#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import random
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

from utils.common45 import (
    # language set (34 langs)
    LANG_CODE_TO_LID_LABEL,
    TARGET_LID_LABELS,

    # shared token table
    load_shared_token_probs,

    # LID
    load_lid_model,
    detect_language,

    # token->language weights
    build_token_language_weights,

    # tokenizer helpers
    ensure_pad_token,

    # probs aggregation
    aggregate_token_probs_to_language_probs,

    # resume/save
    load_escape_sigma,
    save_escape_sigma,
    load_half_similarity,
    save_similarity_progress,
)

# ✅ 使用你现成的 noisy_mgpt.py
from utils.noisy_mgpt import add_activation_noise_to_gpt2_mlp


# ==========================
# Config (mGPT)
# ==========================
MODEL_NAME = "ai-forever/mGPT"

FLORES_DIR = "/ACL_exp/data/flores"
FLORES_SPLIT = "flores_all_dev"

OUTPUT_DIR = "/ACL_exp/Models_LPD/mGPT/flores-st-mc4-45"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ESCAPE_NOISE_JSON = os.path.join(OUTPUT_DIR, "escape_noise_mgpt.json")

TOKEN_LANG_WEIGHTS_JSON = os.path.join(OUTPUT_DIR, "token_lang_weights_mgpt.json")
SHARED_TOKENS_JSON = os.path.join(OUTPUT_DIR, "mgpt_shared_tokens_v2_45.json")

LID_MODEL_NAME = "juliensimon/xlm-v-base-language-id"

SAMPLES_PER_LANG_ESCAPE = 100
SIGMA_GRID = np.arange(0.01, 0.25, 0.01).tolist()
SEED = 42

ALLOWED_LANGS = set(TARGET_LID_LABELS)

MAX_BASELINE_TRIES_MULT = 10
MAX_NEW_TOKENS_STAGE1 = 25

# 你导师关心的：前 N-1 个 token 无噪声，第 N 个 token 才加噪声，并评估第 N 个 token 的语言聚合概率
SWITCH_POS_LIST = [2]  # N = 1,5,10,20


# ==========================
# Data loading
# ==========================
def load_texts_all_and_small():
    flores_path = os.path.join(FLORES_DIR, FLORES_SPLIT)
    print(f"\n[Data] Loading dataset from: {flores_path}")
    ds = load_from_disk(flores_path)

    sentence_cols = [c for c in ds.column_names if c.startswith("sentence_")]
    clean_cols = [c.replace("sentence_", "") for c in sentence_cols]
    clean_to_orig = {clean_cols[i]: sentence_cols[i] for i in range(len(clean_cols))}

    missing_codes = [code for code in LANG_CODE_TO_LID_LABEL.keys() if code not in clean_to_orig]
    if missing_codes:
        print("[Data][Warning] Missing language codes in dataset columns:")
        for m in missing_codes:
            print("  -", m)

    lang_to_all = {lab: [] for lab in TARGET_LID_LABELS}
    lang_to_small = {lab: [] for lab in TARGET_LID_LABELS}

    for lang_code, lab in LANG_CODE_TO_LID_LABEL.items():
        if lang_code not in clean_to_orig:
            continue
        col = clean_to_orig[lang_code]
        texts = [ex[col] for ex in ds if isinstance(ex[col], str) and ex[col].strip()]
        random.shuffle(texts)
        lang_to_all[lab].extend(texts)
        lang_to_small[lab].extend(texts[:SAMPLES_PER_LANG_ESCAPE])

    print("\n[Data] Samples per language:")
    for lab in TARGET_LID_LABELS:
        print(f"  - {lab}: {len(lang_to_small[lab])} small / {len(lang_to_all[lab])} all")

    return lang_to_all, lang_to_small


# ==========================
# Model builder (CausalLM + optional noise)
# ==========================
def build_mgpt_causallm(
    sigma: float,
    device: torch.device,
    tokenizer: Optional[AutoTokenizer] = None,
    inject_noise: bool = True,
):
    """
    构造 mGPT causalLM。
    - 若 inject_noise=True：在 GPT2MLP.act 注入激活噪声，强度 sigma
    - 若 inject_noise=False：不注入噪声（即 clean model）
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        tokenizer = ensure_pad_token(tokenizer)

    print(f"\n[Model] Loading {MODEL_NAME} | inject_noise={inject_noise} | sigma={sigma} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )

    if inject_noise and float(sigma) > 0.0:
        add_activation_noise_to_gpt2_mlp(model, float(sigma))

    model.to(device)
    model.eval()

    # quick stochasticity check (debug) 仅对 noisy 模型有意义
    if inject_noise and float(sigma) > 0.0:
        with torch.no_grad():
            test_ids = torch.randint(low=0, high=min(tokenizer.vocab_size, 32000), size=(1, 16), device=device)
            out1 = model(input_ids=test_ids).logits
            out2 = model(input_ids=test_ids).logits
            diff = (out1 - out2).abs().mean().item()
        print(f"[Debug] sigma={sigma:.2f}, mean |logits1 - logits2| = {diff:.6e}")

    return tokenizer, model


# ==========================
# Stage1: mGPT direct generation (greedy)
# ==========================
def mgpt_generate_reply(
    tokenizer,
    model,
    device,
    text: str,
    max_new_tokens: int = 25,
) -> str:
    """
    直接用 mGPT 生成回复（greedy）。
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_part = gen_ids[0, input_ids.size(1):]
    out = tokenizer.decode(new_part.tolist(), skip_special_tokens=True)
    return (out or "").strip()


def detect_text_lang_label(
    text: str,
    lid_tokenizer,
    lid_model,
    lid_id2label,
    lid_device,
) -> str:
    lab, _ = detect_language(
        text,
        lid_tokenizer,
        lid_model,
        lid_id2label,
        lid_device,
        short_latin_max_len=2,
        top1_top2_margin=0.1,
    )
    return lab


def pick_baseline_consistent_samples(
    src_lang: str,
    all_texts: list,
    tokenizer0,
    model0,
    device,
    lid_tokenizer,
    lid_model,
    lid_id2label,
    lid_device,
    needed: int,
    max_new_tokens: int,
):
    """
    从 all_texts 中挑 needed 条，使得 sigma=0 时，mGPT 生成的回复语言 == src_lang。
    """
    picked = []
    tries = 0
    max_tries = max(needed * MAX_BASELINE_TRIES_MULT, needed)

    for text in all_texts:
        if len(picked) >= needed:
            break
        tries += 1
        if tries > max_tries:
            break

        reply = mgpt_generate_reply(tokenizer0, model0, device, text, max_new_tokens=max_new_tokens)
        if not reply:
            continue

        out_lang = detect_text_lang_label(reply, lid_tokenizer, lid_model, lid_id2label, lid_device)
        if out_lang == src_lang:
            picked.append(text)

    if len(picked) < needed:
        print(f"[Stage1][Baseline] {src_lang}: picked {len(picked)}/{needed} baseline-consistent samples (tries={tries}, pool={len(all_texts)})")
    else:
        print(f"[Stage1][Baseline] {src_lang}: picked {len(picked)}/{needed} baseline-consistent samples")

    return picked


def compute_escape_sigma_by_sentence_confusion(
    src_lang: str,
    baseline_samples: list,
    device,
    lid_tokenizer,
    lid_model,
    lid_id2label,
    lid_device,
    max_new_tokens: int,
    shared_tokenizer: AutoTokenizer,
):
    """
    对每条 baseline sample：
      找最小 sigma，使得“生成回复”的语言 != src_lang；若一直不变，取 SIGMA_GRID[-1]。
    返回：mean_sigma, confused_counts（累计混淆数）
    """
    if not baseline_samples:
        return SIGMA_GRID[-1], {}

    first_confuse_sigma = [None] * len(baseline_samples)
    confused_counts = {}
    num_confused_so_far = 0

    for sigma in SIGMA_GRID:
        # noisy model for this sigma
        _, model = build_mgpt_causallm(float(sigma), device, tokenizer=shared_tokenizer, inject_noise=True)

        newly_confused = 0
        for i, text in enumerate(baseline_samples):
            if first_confuse_sigma[i] is not None:
                continue

            reply = mgpt_generate_reply(shared_tokenizer, model, device, text, max_new_tokens=max_new_tokens)
            out_lang = detect_text_lang_label(reply, lid_tokenizer, lid_model, lid_id2label, lid_device)

            if out_lang != src_lang:
                first_confuse_sigma[i] = sigma
                newly_confused += 1

        num_confused_so_far += newly_confused
        confused_counts[float(sigma)] = int(num_confused_so_far)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if num_confused_so_far >= len(baseline_samples):
            break

    for i in range(len(first_confuse_sigma)):
        if first_confuse_sigma[i] is None:
            first_confuse_sigma[i] = SIGMA_GRID[-1]

    mean_sigma = float(sum(first_confuse_sigma) / len(first_confuse_sigma))
    return mean_sigma, confused_counts


# ==========================
# Stage2 helper (核心修改点)
# 前 N-1 个 token 用 clean model (sigma=0, no noise) greedy 生成
# 第 N 个 token 用 noisy model (sigma=escape_sigma) 取其分布并聚合语言概率
# ==========================
def greedy_generate_prefix_tokens(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    num_new_tokens: int,
) -> torch.Tensor:
    """
    用 greedy 逐步生成 num_new_tokens 个 token，返回追加后的 input_ids。
    注意：这里每步 forward 一次，num_new_tokens 最大 19（当 N=20 时生成 19 个前缀），开销可接受。
    """
    if num_new_tokens <= 0:
        return input_ids

    cur_ids = input_ids
    cur_mask = attention_mask
    device = cur_ids.device

    with torch.no_grad():
        for _ in range(num_new_tokens):
            out = model(input_ids=cur_ids, attention_mask=cur_mask)
            next_logits = out.logits[0, -1]  # [V]
            next_id = int(torch.argmax(next_logits).item())
            next_id_t = torch.tensor([[next_id]], device=device, dtype=cur_ids.dtype)
            cur_ids = torch.cat([cur_ids, next_id_t], dim=1)
            if cur_mask is not None:
                one = torch.ones((cur_mask.size(0), 1), device=device, dtype=cur_mask.dtype)
                cur_mask = torch.cat([cur_mask, one], dim=1)

    return cur_ids


def token_probs_with_noise_switch_at_N(
    tokenizer,
    model_clean,
    model_noisy,
    device,
    text: str,
    N: int,
) -> Optional[torch.Tensor]:
    """
    返回“第 N 个生成 token”的概率分布（softmax 后，cpu float tensor）：
      - 用 clean model 生成前 N-1 个 token（greedy）
      - 然后把 prompt+prefix 喂给 noisy model，取最后位置 logits 作为第 N 个 token 的分布
    """
    if N <= 0:
        return None

    enc = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # 生成前缀：N-1 个 token（无噪声）
    prefix_len = N - 1
    if prefix_len > 0:
        input_ids = greedy_generate_prefix_tokens(model_clean, input_ids, attention_mask, num_new_tokens=prefix_len)
        if attention_mask is not None:
            # greedy_generate_prefix_tokens 已经更新过 mask，但我们这里需要拿到更新后的 mask
            # 由于函数内部生成了 cur_mask，本函数外看不到；因此简单重建 mask（全1）
            attention_mask = torch.ones_like(input_ids, device=device)

    # 用 noisy model 取第 N 个 token 分布
    with torch.no_grad():
        out = model_noisy(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[0, -1].float().cpu()
        probs = torch.softmax(logits, dim=-1)
    return probs


# ==========================
# Main
# ==========================
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lid_device = torch.device("cpu")

    # 0) shared tokenizer (mGPT)
    base_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    base_tokenizer = ensure_pad_token(base_tokenizer)

    # 1) LID model (XLM-V)
    print("[Main] Loading LID model ...")
    lid_tokenizer, lid_model, lid_id2label = load_lid_model(LID_MODEL_NAME)
    lid_model.to(lid_device)

    # 2) shared token probs
    shared_token_probs = load_shared_token_probs(SHARED_TOKENS_JSON)

    # 3) token -> language weights (cache)
    token_lang_weights = build_token_language_weights(
        tokenizer=base_tokenizer,
        shared_token_probs=shared_token_probs,
        lid_tokenizer=lid_tokenizer,
        lid_model=lid_model,
        lid_id2label=lid_id2label,
        lid_device=lid_device,
        target_labels=TARGET_LID_LABELS,    # 34 langs
        symbol_label="symbols",
        unknown_label="unknown",
        cache_path=TOKEN_LANG_WEIGHTS_JSON,
        short_latin_max_len=2,
        top1_top2_margin=0.1,
        progress_step_percent=5,
    )

    # 4) dataset
    lang_to_all_texts, _ = load_texts_all_and_small()

    src_langs = [lab for lab in TARGET_LID_LABELS if len(lang_to_all_texts.get(lab, [])) > 0]
    print("\n[Main] Source languages with samples:")
    for lab in src_langs:
        print(f"  - {lab}: {len(lang_to_all_texts[lab])} all")

    # ==================
    # Stage 1: escape sigma
    # ==================
    escape_sigma = load_escape_sigma(ESCAPE_NOISE_JSON, src_langs)
    langs_needing_escape = [l for l in src_langs if escape_sigma[l] is None]
    print(f"\n[Stage1] Languages needing escape-noise search: {len(langs_needing_escape)}")

    if langs_needing_escape:
        # clean model for baseline-consistent picking
        _, model0 = build_mgpt_causallm(0.0, device, tokenizer=base_tokenizer, inject_noise=False)

        for src_lang in src_langs:
            if escape_sigma[src_lang] is not None:
                continue

            pool = lang_to_all_texts.get(src_lang, [])
            if not pool:
                escape_sigma[src_lang] = SIGMA_GRID[-1]
                print(f"[Stage1] {src_lang}: 0 samples -> escape_sigma={escape_sigma[src_lang]:.2f}")
                save_escape_sigma(ESCAPE_NOISE_JSON, escape_sigma)
                continue

            baseline_samples = pick_baseline_consistent_samples(
                src_lang=src_lang,
                all_texts=pool,
                tokenizer0=base_tokenizer,
                model0=model0,
                device=device,
                lid_tokenizer=lid_tokenizer,
                lid_model=lid_model,
                lid_id2label=lid_id2label,
                lid_device=lid_device,
                needed=SAMPLES_PER_LANG_ESCAPE,
                max_new_tokens=MAX_NEW_TOKENS_STAGE1,
            )

            mean_sigma, confused_counts = compute_escape_sigma_by_sentence_confusion(
                src_lang=src_lang,
                baseline_samples=baseline_samples,
                device=device,
                lid_tokenizer=lid_tokenizer,
                lid_model=lid_model,
                lid_id2label=lid_id2label,
                lid_device=lid_device,
                max_new_tokens=MAX_NEW_TOKENS_STAGE1,
                shared_tokenizer=base_tokenizer,
            )

            escape_sigma[src_lang] = float(mean_sigma)
            print(f"[Stage1][EscapeAvg] {src_lang}: escape_sigma(mean_confuse_sigma)={mean_sigma:.4f}, baseline_n={len(baseline_samples)}")
            if confused_counts:
                last_sigma = max(confused_counts.keys())
                print(f"  [Stage1][Confused@{last_sigma:.2f}] {confused_counts[last_sigma]}/{len(baseline_samples)}")

            save_escape_sigma(ESCAPE_NOISE_JSON, escape_sigma)

        del model0
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print("\n[Stage1] All escape_sigma already available, skip.")

    for lang in src_langs:
        if escape_sigma[lang] is None:
            escape_sigma[lang] = SIGMA_GRID[-1]
    save_escape_sigma(ESCAPE_NOISE_JSON, escape_sigma)

    # ==================
    # Stage 2: language similarity at different switch positions N
    #   - 前 N-1 token 无噪声（clean model）
    #   - 第 N token 加噪声（sigma = escape_sigma[src_lang]）
    #   - 聚合第 N token 的语言概率（NO renormalize）
    # ==================
    print("\n[Stage2] Building ONE clean model (no noise) for all N ...")
    _, clean_model = build_mgpt_causallm(0.0, device, tokenizer=base_tokenizer, inject_noise=False)

    # sigma -> langs
    sigma_to_langs = defaultdict(list)
    for lang in src_langs:
        sigma_to_langs[escape_sigma[lang]].append(lang)

    for N in SWITCH_POS_LIST:
        similarity_json = os.path.join(OUTPUT_DIR, f"language_similarity_mgpt_switchN{N}.json")
        print(f"\n[Stage2] =============================")
        print(f"[Stage2] Switch position N = {N}")
        print(f"[Stage2] Output: {similarity_json}")
        print(f"[Stage2] =============================")

        half_sim = load_half_similarity(similarity_json, src_langs)

        print(f"\n[Stage2][N={N}] Compute half similarity at sigma = escape_sigma[lang], using ALL samples. (NO renormalize)")
        for sigma in sorted(sigma_to_langs.keys()):
            langs_at_sigma = [l for l in sigma_to_langs[sigma] if l not in half_sim]
            if not langs_at_sigma:
                continue

            print(f"\n[Stage2][N={N}] ===== Sigma = {sigma:.4f} ===== for languages: {langs_at_sigma}")

            # noisy model for this sigma
            _, noisy_model = build_mgpt_causallm(float(sigma), device, tokenizer=base_tokenizer, inject_noise=True)

            for src_lang in langs_at_sigma:
                texts = lang_to_all_texts.get(src_lang, [])
                if not texts:
                    print(f"[Stage2][N={N} σ={sigma:.4f}] {src_lang} has 0 all samples -> zeros")
                    half_sim[src_lang] = {lab: 0.0 for lab in TARGET_LID_LABELS}
                    save_similarity_progress(
                        similarity_json,
                        escape_sigma=escape_sigma,
                        half_sim=half_sim,
                        languages=src_langs,
                        sigma_grid=SIGMA_GRID,
                        notes=(
                            f"Stage1: mean first-confuse sigma via mGPT direct generation; "
                            f"Stage2: token-N distribution with noise switch: first N-1 tokens clean, "
                            f"Nth token noisy; aggregation NO renormalize. "
                            f"Noise: utils/noisy_mgpt.py GPT2MLP.act noise. N={N}"
                        ),
                    )
                    continue

                print(f"[Stage2][N={N} σ={sigma:.4f}] Processing {src_lang} with {len(texts)} ALL samples ...")
                sum_lang_probs = defaultdict(float)
                num_valid = 0

                for text in texts:
                    probs = token_probs_with_noise_switch_at_N(
                        tokenizer=base_tokenizer,
                        model_clean=clean_model,
                        model_noisy=noisy_model,
                        device=device,
                        text=text,
                        N=N,
                    )
                    if probs is None:
                        continue

                    lang_probs = aggregate_token_probs_to_language_probs(
                        probs,
                        token_lang_weights,
                        allowed_langs=ALLOWED_LANGS,
                        renormalize=False,
                    )
                    if not lang_probs:
                        continue

                    for k, v in lang_probs.items():
                        sum_lang_probs[k] += float(v)
                    num_valid += 1

                if num_valid == 0:
                    print(f"[Stage2][N={N} σ={sigma:.4f}] Warning: no valid samples for {src_lang} -> zeros")
                    half_sim[src_lang] = {lab: 0.0 for lab in TARGET_LID_LABELS}
                else:
                    avg = {lab: sum_lang_probs.get(lab, 0.0) / num_valid for lab in TARGET_LID_LABELS}
                    self_mass = avg.get(src_lang, 0.0)
                    print(f"[Stage2][N={N} σ={sigma:.4f}] {src_lang}: self_mass={self_mass:.6f}, num_valid={num_valid}")
                    half_sim[src_lang] = avg

                save_similarity_progress(
                    similarity_json,
                    escape_sigma=escape_sigma,
                    half_sim=half_sim,
                    languages=src_langs,
                    sigma_grid=SIGMA_GRID,
                    notes=(
                        f"Stage1: mean first-confuse sigma via mGPT direct generation; "
                        f"Stage2: token-N distribution with noise switch: first N-1 tokens clean, "
                        f"Nth token noisy; aggregation NO renormalize. "
                        f"Noise: utils/noisy_mgpt.py GPT2MLP.act noise. N={N}"
                    ),
                )

            del noisy_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        print(f"\n[Stage2][N={N}] Done: saved to {similarity_json}")

    del clean_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n[Main] Done. escape_sigma saved + 3 similarity JSONs saved (N=5,10,20).")


if __name__ == "__main__":
    main()
