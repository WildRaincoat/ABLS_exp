#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.common45 import (
    # language set (34 langs)
    TARGET_LID_LABELS,

    # ✅ Tatoeba loader lives in utils.common
    load_texts_from_tatoeba_csv,

    # shared token table
    load_shared_token_probs,

    # LID
    load_lid_model,
    detect_language,

    # token->language weights
    build_token_language_weights,

    # tokenizer helpers
    ensure_pad_token,
    get_special_token_ids,               # (Stage2 不再用 special_ids，但保留 import 不影响)
    pick_random_non_special_position,    # (Stage2 不再用，但保留 import 不影响)

    # probs aggregation
    aggregate_token_probs_to_language_probs,

    # resume/save
    load_escape_sigma,
    save_escape_sigma,
    load_half_similarity,
    save_similarity_progress,
)

# ✅ 新的动态噪声注入方式（只加载一次模型，动态调 sigma）
from utils.noisy_qwen3 import (
    attach_noisy_qwen_mlp_activation_patch,
    restore_patched_mlps,
    NoiseController,
)


# ==========================
# Config (Qwen3-8B)
# ==========================
MODEL_NAME = "Qwen/Qwen3-8B"

# ✅ Tatoeba
TATOEBA_SENTENCES_CSV = "/ACL_exp/data/tatoeba/sentences.csv"

OUTPUT_DIR = "/ACL_exp/Models_LPD/Qwen3/tatoeba45"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIMILARITY_JSON = os.path.join(OUTPUT_DIR, "language_similarity_qwen.json")
ESCAPE_NOISE_JSON = os.path.join(OUTPUT_DIR, "escape_noise_qwen.json")

TOKEN_LANG_WEIGHTS_JSON = os.path.join(OUTPUT_DIR, "token_lang_weights_qwen.json")
SHARED_TOKENS_JSON = os.path.join(OUTPUT_DIR, "qwen3_8b_shared_tokens_v2_45.json")

LID_MODEL_NAME = "juliensimon/xlm-v-base-language-id"

SAMPLES_PER_LANG_ESCAPE = 100
SIGMA_GRID = np.arange(0.01, 0.25, 0.01).tolist()
SEED = 42

# ✅ Stage2：每个语言用于 similarity 的最大样本数
MAX_SAMPLES_PER_LANG_SIM = 1500

# Stage2 仍然只关心 34 种语言
ALLOWED_LANGS = set(TARGET_LID_LABELS)

# Stage1 baseline 样本筛选：最多尝试多少条原始样本（防止极端语言卡死）
MAX_BASELINE_TRIES_MULT = 10

# Stage1：Qwen 直接生成句子
MAX_NEW_TOKENS_STAGE1 = 25

# ✅ bit 量化 + CPU offload
OFFLOAD_DIR = os.path.join(OUTPUT_DIR, "offload_int8")
os.makedirs(OFFLOAD_DIR, exist_ok=True)


# ==========================
# Data loading (Tatoeba)
# ==========================
def load_texts_all_and_small():
    """
    用 Tatoeba sentences.csv。
    返回结构保持一致：lang_to_all, lang_to_small
    key 为 TARGET_LID_LABELS 里的 label（English/Spanish/...）
    """
    print(f"\n[Data] Loading Tatoeba sentences from: {TATOEBA_SENTENCES_CSV}")

    lang_to_all, lang_to_small = load_texts_from_tatoeba_csv(
        csv_path=TATOEBA_SENTENCES_CSV,
        target_labels=TARGET_LID_LABELS,
        samples_per_lang_small=SAMPLES_PER_LANG_ESCAPE,
        seed=SEED,
    )

    print("\n[Data] Samples per language:")
    for lab in TARGET_LID_LABELS:
        print(f"  - {lab}: {len(lang_to_small.get(lab, []))} small / {len(lang_to_all.get(lab, []))} all")

    return lang_to_all, lang_to_small


# ==========================
# Model builder (load once + int8 offload + dynamic sigma)
# ==========================
def build_qwen3_int8_offload_once() -> Tuple[AutoTokenizer, torch.nn.Module]:
    print(f"\n[Model] Loading {MODEL_NAME} once (int8 + fp32 cpu offload) ...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, use_fast=False, trust_remote_code=True
    )
    tokenizer = ensure_pad_token(tokenizer)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        offload_folder=OFFLOAD_DIR,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def debug_stochasticity(tokenizer, model, controller: NoiseController, sigma: float):
    controller.set_sigma(float(sigma))
    with torch.no_grad():
        dev = next(model.parameters()).device
        test_ids = torch.randint(
            low=0,
            high=min(tokenizer.vocab_size, 32000),
            size=(1, 16),
            device=dev,
        )
        out1 = model(input_ids=test_ids).logits
        out2 = model(input_ids=test_ids).logits
        diff = (out1 - out2).abs().mean().item()
    print(f"[Debug] sigma={sigma:.2f}, mean |logits1 - logits2| = {diff:.6e}")


# ==========================
# Stage1: Qwen direct generation
# ==========================
def qwen_generate_reply(
    tokenizer,
    model,
    text: str,
    max_new_tokens: int = 25,
) -> str:
    """
    直接用 Qwen 生成回复（greedy）。激活噪声由 controller 动态控制。
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", None)

    model_device = next(model.parameters()).device
    input_ids = input_ids.to(model_device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model_device)

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
    tokenizer,
    model,
    controller: NoiseController,
    lid_tokenizer,
    lid_model,
    lid_id2label,
    lid_device,
    needed: int,
    max_new_tokens: int,
):
    """
    从 all_texts 中挑 needed 条，使得 sigma=0 时，Qwen 生成的回复语言 == src_lang。
    """
    controller.set_sigma(0.0)

    picked = []
    tries = 0
    max_tries = max(needed * MAX_BASELINE_TRIES_MULT, needed)

    for text in all_texts:
        if len(picked) >= needed:
            break
        tries += 1
        if tries > max_tries:
            break

        reply = qwen_generate_reply(tokenizer, model, text, max_new_tokens=max_new_tokens)
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


def compute_escape_sigma_parallel_roundscan(
    src_langs: List[str],
    baseline_samples_by_lang: Dict[str, List[str]],
    tokenizer,
    model,
    controller: NoiseController,
    lid_tokenizer,
    lid_model,
    lid_id2label,
    lid_device,
    max_new_tokens: int,
):
    """
    ✅ 多语言并行（round-robin）扫描 SIGMA_GRID：
    对每个 lang 的每条 baseline sample，记录最小 sigma 使得“生成回复”的语言 != src_lang。
    若一直不变，取 SIGMA_GRID[-1]。

    返回：
      escape_sigma_by_lang: {lang: mean_first_confuse_sigma}
      confused_counts_by_lang: {lang: {sigma: cumulative_confused}}
    """
    trackers = {}
    first_confuse_sigma = {}
    confused_counts_by_lang: Dict[str, Dict[float, int]] = {}

    for lang in src_langs:
        bs = baseline_samples_by_lang.get(lang, [])
        first_confuse_sigma[lang] = [None] * len(bs)
        trackers[lang] = {"bs": bs, "pending": list(range(len(bs)))}
        confused_counts_by_lang[lang] = {}

    if all(len(trackers[l]["pending"]) == 0 for l in trackers):
        return {l: float(SIGMA_GRID[-1]) for l in src_langs}, confused_counts_by_lang

    for sigma in SIGMA_GRID:
        print(f"\n[Stage1] ===== sigma={sigma:.4f} (dynamic, parallel langs) =====")
        controller.set_sigma(float(sigma))
        debug_stochasticity(tokenizer, model, controller, sigma=float(sigma))

        for lang, tr in trackers.items():
            if not tr["pending"]:
                last = 0
                if confused_counts_by_lang[lang]:
                    last = confused_counts_by_lang[lang][max(confused_counts_by_lang[lang].keys())]
                confused_counts_by_lang[lang][float(sigma)] = int(last)
                continue

            still_pending = []
            newly_confused = 0

            for idx in tr["pending"]:
                prompt = tr["bs"][idx]
                reply = qwen_generate_reply(tokenizer, model, prompt, max_new_tokens=max_new_tokens)
                if not reply:
                    still_pending.append(idx)
                    continue

                det = detect_text_lang_label(reply, lid_tokenizer, lid_model, lid_id2label, lid_device)
                if det != lang:
                    first_confuse_sigma[lang][idx] = sigma
                    newly_confused += 1
                else:
                    still_pending.append(idx)

            tr["pending"] = still_pending

            prev = 0
            if confused_counts_by_lang[lang]:
                prev = confused_counts_by_lang[lang][max(confused_counts_by_lang[lang].keys())]
            confused_counts_by_lang[lang][float(sigma)] = int(prev + newly_confused)

        if all(len(tr["pending"]) == 0 for tr in trackers.values()):
            print("[Stage1] All languages finished early.")
            break

    escape_sigma_by_lang = {}
    for lang in src_langs:
        arr = first_confuse_sigma[lang]
        if not arr:
            escape_sigma_by_lang[lang] = float(SIGMA_GRID[-1])
            continue
        for i in range(len(arr)):
            if arr[i] is None:
                arr[i] = SIGMA_GRID[-1]
        escape_sigma_by_lang[lang] = float(sum(arr) / len(arr))

    return escape_sigma_by_lang, confused_counts_by_lang


# ==========================
# Stage2 helper for CausalLM (UPDATED):
# full prompt -> first generated token distribution
# ==========================
def causal_first_generated_token_probs_full_prompt(
    tokenizer,
    model,
    text: str,
):
    """
    ✅ Stage2 对齐到“完整 prompt”：
    把完整输入 prompt 喂给 CausalLM，然后取最后位置 logits：
      P(next_token | full_prompt, sigma)
    这就是“模型生成的第一个 token”的概率分布（不真正 generate）。
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", None)

    if input_ids.size(1) < 1:
        return None

    dev = next(model.parameters()).device
    input_ids = input_ids.to(dev)
    if attention_mask is not None:
        attention_mask = attention_mask.to(dev)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # [1, L, V]

    next_logits = logits[0, -1].float().cpu()
    probs = torch.softmax(next_logits, dim=-1)
    return probs


# ==========================
# Main
# ==========================
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    lid_device = torch.device("cpu")

    # 1) LID model (XLM-V)
    print("[Main] Loading LID model ...")
    lid_tokenizer, lid_model, lid_id2label = load_lid_model(LID_MODEL_NAME)
    lid_model.to(lid_device)
    lid_model.eval()

    # 2) shared token probs
    shared_token_probs = load_shared_token_probs(SHARED_TOKENS_JSON)

    # 3) load Qwen3 once (int8 + offload) and patch noisy mlps
    tokenizer, model = build_qwen3_int8_offload_once()
    print("[Main] Patching MLP forward (dynamic sigma activation noise) ...")
    controller, restore_map = attach_noisy_qwen_mlp_activation_patch(model, controller=None, verbose=True)

    # 4) token -> language weights (cache)  ✅ 直接用同一个 tokenizer
    token_lang_weights = build_token_language_weights(
        tokenizer=tokenizer,
        shared_token_probs=shared_token_probs,
        lid_tokenizer=lid_tokenizer,
        lid_model=lid_model,
        lid_id2label=lid_id2label,
        lid_device=lid_device,
        target_labels=TARGET_LID_LABELS,
        symbol_label="symbols",
        unknown_label="unknown",
        cache_path=TOKEN_LANG_WEIGHTS_JSON,
        short_latin_max_len=2,
        top1_top2_margin=0.1,
        progress_step_percent=5,
    )

    # 5) dataset (Tatoeba)
    lang_to_all_texts, _ = load_texts_all_and_small()

    src_langs = [lab for lab in TARGET_LID_LABELS if len(lang_to_all_texts.get(lab, [])) > 0]
    print("\n[Main] Source languages with samples:")
    for lab in src_langs:
        print(f"  - {lab}: {len(lang_to_all_texts[lab])} all")

    # ==================
    # Stage 1: escape sigma (direct generation confusion avg)
    # ==================
    escape_sigma = load_escape_sigma(ESCAPE_NOISE_JSON, src_langs)
    langs_needing_escape = [l for l in src_langs if escape_sigma[l] is None]
    print(f"\n[Stage1] Languages needing escape-noise search: {len(langs_needing_escape)}")

    if langs_needing_escape:
        # baseline picking @ sigma=0
        controller.set_sigma(0.0)
        debug_stochasticity(tokenizer, model, controller, sigma=0.0)

        baseline_samples_by_lang: Dict[str, List[str]] = {}

        for src_lang in src_langs:
            if escape_sigma[src_lang] is not None:
                continue

            pool = lang_to_all_texts.get(src_lang, [])
            if not pool:
                baseline_samples_by_lang[src_lang] = []
                continue

            baseline_samples_by_lang[src_lang] = pick_baseline_consistent_samples(
                src_lang=src_lang,
                all_texts=pool,
                tokenizer=tokenizer,
                model=model,
                controller=controller,
                lid_tokenizer=lid_tokenizer,
                lid_model=lid_model,
                lid_id2label=lid_id2label,
                lid_device=lid_device,
                needed=SAMPLES_PER_LANG_ESCAPE,
                max_new_tokens=MAX_NEW_TOKENS_STAGE1,
            )

        need_langs = [l for l in src_langs if escape_sigma[l] is None]
        escape_by_lang, confused_counts_by_lang = compute_escape_sigma_parallel_roundscan(
            src_langs=need_langs,
            baseline_samples_by_lang=baseline_samples_by_lang,
            tokenizer=tokenizer,
            model=model,
            controller=controller,
            lid_tokenizer=lid_tokenizer,
            lid_model=lid_model,
            lid_id2label=lid_id2label,
            lid_device=lid_device,
            max_new_tokens=MAX_NEW_TOKENS_STAGE1,
        )

        for src_lang in need_langs:
            bs = baseline_samples_by_lang.get(src_lang, [])
            if not bs:
                escape_sigma[src_lang] = float(SIGMA_GRID[-1])
                print(f"[Stage1] {src_lang}: 0 baseline samples -> escape_sigma={escape_sigma[src_lang]:.4f}")
                save_escape_sigma(ESCAPE_NOISE_JSON, escape_sigma)
                continue

            mean_sigma = float(escape_by_lang[src_lang])
            escape_sigma[src_lang] = mean_sigma
            print(f"[Stage1][EscapeAvg] {src_lang}: escape_sigma(mean_confuse_sigma)={mean_sigma:.4f}, baseline_n={len(bs)}")

            cc = confused_counts_by_lang.get(src_lang, {})
            if cc:
                last_sigma = max(cc.keys())
                print(f"  [Stage1][Confused@{last_sigma:.2f}] {cc[last_sigma]}/{len(bs)}")

            save_escape_sigma(ESCAPE_NOISE_JSON, escape_sigma)

    else:
        print("\n[Stage1] All escape_sigma already available, skip.")

    for lang in src_langs:
        if escape_sigma[lang] is None:
            escape_sigma[lang] = float(SIGMA_GRID[-1])
    save_escape_sigma(ESCAPE_NOISE_JSON, escape_sigma)

    # ==================
    # Stage 2: half similarity (token prob aggregation) — NO renormalize
    #   ✅ 修改点：
    #     - Stage2 改为“完整输入 prompt 的 next-token 分布”（第一个生成 token）
    #     - 仍然：按语言 escape_sigma 动态 set_sigma
    #     - 仍然：每语言最多采样 MAX_SAMPLES_PER_LANG_SIM
    # ==================
    half_sim = load_half_similarity(SIMILARITY_JSON, src_langs)

    sigma_to_langs = defaultdict(list)
    for lang in src_langs:
        sigma_to_langs[escape_sigma[lang]].append(lang)

    print("\n[Stage2] Compute half similarity at sigma = escape_sigma[lang]. (NO renormalize)")
    print(f"[Stage2] Per-language cap for similarity samples: {MAX_SAMPLES_PER_LANG_SIM}")
    print("[Stage2] UPDATED: Use FULL-PROMPT next-token distribution (first generated token).")

    for sigma in sorted(sigma_to_langs.keys()):
        langs_at_sigma = [l for l in sigma_to_langs[sigma] if l not in half_sim]
        if not langs_at_sigma:
            continue

        print(f"\n[Stage2] ===== Sigma = {sigma:.4f} ===== for languages: {langs_at_sigma}")
        controller.set_sigma(float(sigma))
        debug_stochasticity(tokenizer, model, controller, sigma=float(sigma))

        for src_lang in langs_at_sigma:
            all_texts = lang_to_all_texts.get(src_lang, [])
            if not all_texts:
                print(f"[Stage2 σ={sigma:.4f}] {src_lang} has 0 all samples -> zeros")
                half_sim[src_lang] = {lab: 0.0 for lab in TARGET_LID_LABELS}
                save_similarity_progress(
                    SIMILARITY_JSON,
                    escape_sigma=escape_sigma,
                    half_sim=half_sim,
                    languages=src_langs,
                    sigma_grid=SIGMA_GRID,
                    notes="Stage1: mean first-confuse sigma via Qwen direct generation (parallel roundscan, dynamic sigma); "
                          "Stage2: FULL-PROMPT first generated token prob aggregation (next-token at full prompt) NO renormalize. "
                          "Dataset: Tatoeba sentences.csv. "
                          "Noise: utils/noisy_qwen3.py patch MLP forward after activation (dynamic sigma). "
                          "Quantization: int8 bitsandbytes + fp32 cpu offload (device_map=auto). "
                          f"Stage2 per-lang cap={MAX_SAMPLES_PER_LANG_SIM}.",
                )
                continue

            # ✅ Stage2 只取最多 MAX_SAMPLES_PER_LANG_SIM 条（不足就全用）
            if len(all_texts) > MAX_SAMPLES_PER_LANG_SIM:
                rng = random.Random(SEED + (abs(hash(src_lang)) % 10_000_000))
                texts = rng.sample(all_texts, k=MAX_SAMPLES_PER_LANG_SIM)
            else:
                texts = all_texts

            print(f"[Stage2 σ={sigma:.4f}] Processing {src_lang}: use {len(texts)}/{len(all_texts)} samples ...")

            sum_lang_probs = defaultdict(float)
            num_valid = 0

            for text in texts:
                probs = causal_first_generated_token_probs_full_prompt(
                    tokenizer=tokenizer,
                    model=model,
                    text=text,
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
                print(f"[Stage2 σ={sigma:.4f}] Warning: no valid samples for {src_lang} -> zeros")
                half_sim[src_lang] = {lab: 0.0 for lab in TARGET_LID_LABELS}
            else:
                avg = {lab: sum_lang_probs.get(lab, 0.0) / num_valid for lab in TARGET_LID_LABELS}
                self_mass = avg.get(src_lang, 0.0)
                print(f"[Stage2 σ={sigma:.4f}] {src_lang}: self_mass={self_mass:.6f}, num_valid={num_valid}")
                half_sim[src_lang] = avg

            save_similarity_progress(
                SIMILARITY_JSON,
                escape_sigma=escape_sigma,
                half_sim=half_sim,
                languages=src_langs,
                sigma_grid=SIGMA_GRID,
                notes="Stage1: mean first-confuse sigma via Qwen direct generation (parallel roundscan, dynamic sigma); "
                      "Stage2: FULL-PROMPT first generated token prob aggregation (next-token at full prompt) NO renormalize. "
                      "Dataset: Tatoeba sentences.csv. "
                      "Noise: utils/noisy_qwen3.py patch MLP forward after activation (dynamic sigma). "
                      "Quantization: int8 bitsandbytes + fp32 cpu offload (device_map=auto). "
                      f"Stage2 per-lang cap={MAX_SAMPLES_PER_LANG_SIM}.",
            )

    # ✅ restore patched mlps
    restore_patched_mlps(model, restore_map)
    print("\n[Main] Done. escape_sigma + similarity saved.")


if __name__ == "__main__":
    main()
