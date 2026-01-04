#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import random
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
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
    detect_language,  # ✅ 用于检测“伪生成回复”的语言

    # token->language weights
    build_token_language_weights,

    # tokenizer helpers
    ensure_pad_token,
    get_special_token_ids,

    # probs aggregation
    aggregate_token_probs_to_language_probs,

    # resume/save
    load_escape_sigma,
    save_escape_sigma,
    load_half_similarity,
    save_similarity_progress,
)

from utils.noisy_xlml import add_noise_to_xlmr_ffn_activations


# ==========================
# Config
# ==========================
MODEL_NAME = "xlm-roberta-large"

FLORES_DIR = "/ACL_exp/data/flores"
FLORES_SPLIT = "flores_all_dev"

OUTPUT_DIR = "/ACL_exp/Models_LPD/XLM-L/flores45"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIMILARITY_JSON = os.path.join(OUTPUT_DIR, "language_similarity_xlmr.json")
ESCAPE_NOISE_JSON = os.path.join(OUTPUT_DIR, "escape_noise_xlmr.json")

TOKEN_LANG_WEIGHTS_JSON = os.path.join(OUTPUT_DIR, "token_lang_weights_xlmr.json")
SHARED_TOKENS_JSON = os.path.join(OUTPUT_DIR, "xlmr_large_shared_tokens_v2_45.json")

LID_MODEL_NAME = "juliensimon/xlm-v-base-language-id"

SAMPLES_PER_LANG_ESCAPE = 100
SIGMA_GRID = np.arange(0.01, 0.25, 0.01).tolist()
SEED = 42

# Stage2 仍然只关心 34 种语言
ALLOWED_LANGS = set(TARGET_LID_LABELS)

# Stage1 baseline 样本筛选：最多尝试多少条原始样本（防止极端语言卡死）
MAX_BASELINE_TRIES_MULT = 10  # 最多尝试 100*10 条，仍不足就接受现有数量


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
# Model builder (MLM + noise)
# ==========================
def build_noisy_xlmr_mlm(sigma: float, device: torch.device):
    print(f"\n[Model] Loading {MODEL_NAME} with sigma={sigma} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer = ensure_pad_token(tokenizer)

    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )

    patched = add_noise_to_xlmr_ffn_activations(model, sigma)
    print(f"[Debug] sigma={sigma:.2f}, patched_count={patched}")

    model.to(device)
    model.eval()

    with torch.no_grad():
        test_ids = torch.randint(low=0, high=tokenizer.vocab_size, size=(1, 16), device=device)
        out1 = model(input_ids=test_ids).logits
        out2 = model(input_ids=test_ids).logits
        diff = (out1 - out2).abs().mean().item()
    print(f"[Debug] sigma={sigma:.2f}, mean |logits1 - logits2| = {diff:.6e}")

    return tokenizer, model


# ==========================
# Stage1: MLM pseudo-generation (UPDATED)
# ==========================
def pseudo_reply_mlm_full_sentence_sequential(
    tokenizer,
    model,
    device,
    text: str,
) -> str:
    """
    ✅ 新版本（按你的要求）：
    - 不再随机 mask + 拼 25 token
    - 改为：从第 1 个“词/子词 token”开始，一直到最后一个，
      逐位置 mask，并取该位置 top-1 预测；
    - 生成的伪句子长度与输入相同（同一条 input_ids 序列），
      最后 decode 整条序列作为伪句子，再送去 LID。
    - 仍然“fill back”预测值，使上下文随位置推进而演化。
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"][0]  # 1D: [L]
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    special_ids = get_special_token_ids(tokenizer)

    # work on a mutable copy
    ids = input_ids.clone()

    # 顺序从左到右逐位置预测（跳过 special token 位置）
    for pos in range(ids.size(0)):
        if int(ids[pos].item()) in special_ids:
            continue

        masked = ids.clone()
        masked[pos] = tokenizer.mask_token_id

        masked = masked.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_ids=masked, attention_mask=attention_mask).logits

        pos_logits = logits[0, pos].float().cpu()
        pred_id = int(torch.argmax(pos_logits).item())

        # fill back prediction to evolve context
        ids[pos] = pred_id

    # decode 整条序列（长度与输入一致）
    out = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
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
):
    """
    From all_texts, pick up to `needed` samples such that at sigma=0,
    the pseudo reply language == src_lang.
    If cannot reach needed, return what we found, and print count.
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

        reply = pseudo_reply_mlm_full_sentence_sequential(tokenizer0, model0, device, text)
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
):
    """
    For each sample:
      find the smallest sigma in SIGMA_GRID such that detected reply language != src_lang.
      If never changes, assign SIGMA_GRID[-1].
    Return:
      mean_sigma, per_sigma_confused_counts (for logging)
    """
    if not baseline_samples:
        return SIGMA_GRID[-1], {}

    # track per-sample first confused sigma
    first_confuse_sigma = [None] * len(baseline_samples)

    confused_counts = {}  # sigma -> num_confused_at_this_sigma (cumulative vs src_lang)
    num_confused_so_far = 0

    for sigma in SIGMA_GRID:
        tokenizer, model = build_noisy_xlmr_mlm(sigma, device)

        newly_confused = 0
        for i, text in enumerate(baseline_samples):
            if first_confuse_sigma[i] is not None:
                continue

            reply = pseudo_reply_mlm_full_sentence_sequential(tokenizer, model, device, text)
            out_lang = detect_text_lang_label(reply, lid_tokenizer, lid_model, lid_id2label, lid_device)

            if out_lang != src_lang:
                first_confuse_sigma[i] = sigma
                newly_confused += 1

        num_confused_so_far += newly_confused
        confused_counts[float(sigma)] = int(num_confused_so_far)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # early stop if all confused
        if num_confused_so_far >= len(baseline_samples):
            break

    # fill not confused
    for i in range(len(first_confuse_sigma)):
        if first_confuse_sigma[i] is None:
            first_confuse_sigma[i] = SIGMA_GRID[-1]

    mean_sigma = float(sum(first_confuse_sigma) / len(first_confuse_sigma))
    return mean_sigma, confused_counts


# ==========================
# Main
# ==========================
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lid_device = torch.device("cpu")

    # 1) LID model (XLM-V)
    print("[Main] Loading LID model ...")
    lid_tokenizer, lid_model, lid_id2label = load_lid_model(LID_MODEL_NAME)
    lid_model.to(lid_device)

    # 2) shared token probs
    shared_token_probs = load_shared_token_probs(SHARED_TOKENS_JSON)

    # 3) token -> language weights (cache)
    base_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_tokenizer = ensure_pad_token(base_tokenizer)

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
    lang_to_all_texts, lang_to_small_texts = load_texts_all_and_small()

    src_langs = [lab for lab in TARGET_LID_LABELS if len(lang_to_all_texts.get(lab, [])) > 0]
    print("\n[Main] Source languages with samples:")
    for lab in src_langs:
        print(f"  - {lab}: {len(lang_to_all_texts[lab])} all")

    # ==================
    # Stage 1: NEW escape sigma (sentence confusion avg)
    # ==================
    escape_sigma = load_escape_sigma(ESCAPE_NOISE_JSON, src_langs)
    langs_needing_escape = [l for l in src_langs if escape_sigma[l] is None]
    print(f"\n[Stage1] Languages needing escape-noise search: {len(langs_needing_escape)}")

    if langs_needing_escape:
        # baseline model at sigma=0
        tok0, model0 = build_noisy_xlmr_mlm(0.0, device)

        for src_lang in src_langs:
            if escape_sigma[src_lang] is not None:
                continue

            pool = lang_to_all_texts.get(src_lang, [])
            if not pool:
                escape_sigma[src_lang] = SIGMA_GRID[-1]
                print(f"[Stage1] {src_lang}: 0 samples in dataset -> escape_sigma={escape_sigma[src_lang]:.2f}")
                save_escape_sigma(ESCAPE_NOISE_JSON, escape_sigma)
                continue

            # 1) select baseline-consistent samples (sigma=0 reply language == src_lang)
            baseline_samples = pick_baseline_consistent_samples(
                src_lang=src_lang,
                all_texts=pool,
                tokenizer0=tok0,
                model0=model0,
                device=device,
                lid_tokenizer=lid_tokenizer,
                lid_model=lid_model,
                lid_id2label=lid_id2label,
                lid_device=lid_device,
                needed=SAMPLES_PER_LANG_ESCAPE,
            )

            # 2) find per-sample first confuse sigma, then average
            mean_sigma, confused_counts = compute_escape_sigma_by_sentence_confusion(
                src_lang=src_lang,
                baseline_samples=baseline_samples,
                device=device,
                lid_tokenizer=lid_tokenizer,
                lid_model=lid_model,
                lid_id2label=lid_id2label,
                lid_device=lid_device,
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

    # fill any None
    for lang in src_langs:
        if escape_sigma[lang] is None:
            escape_sigma[lang] = SIGMA_GRID[-1]
    save_escape_sigma(ESCAPE_NOISE_JSON, escape_sigma)

    # ==================
    # Stage 2: half similarity (token prob aggregation) — NO renormalize
    # ==================
    half_sim = load_half_similarity(SIMILARITY_JSON, src_langs)

    sigma_to_langs = defaultdict(list)
    for lang in src_langs:
        sigma_to_langs[escape_sigma[lang]].append(lang)

    print("\n[Stage2] Compute half similarity at sigma = escape_sigma[lang], using ALL samples. (NO renormalize)")
    for sigma in sorted(sigma_to_langs.keys()):
        langs_at_sigma = [l for l in sigma_to_langs[sigma] if l not in half_sim]
        if not langs_at_sigma:
            continue

        print(f"\n[Stage2] ===== Sigma = {sigma:.4f} ===== for languages: {langs_at_sigma}")
        tokenizer, model = build_noisy_xlmr_mlm(float(sigma), device)
        special_ids = get_special_token_ids(tokenizer)

        for src_lang in langs_at_sigma:
            texts = lang_to_all_texts.get(src_lang, [])
            if not texts:
                print(f"[Stage2 σ={sigma:.4f}] {src_lang} has 0 all samples -> zeros")
                half_sim[src_lang] = {lab: 0.0 for lab in TARGET_LID_LABELS}
                save_similarity_progress(
                    SIMILARITY_JSON,
                    escape_sigma=escape_sigma,
                    half_sim=half_sim,
                    languages=src_langs,
                    sigma_grid=SIGMA_GRID,
                    notes="Stage1: mean first-confuse sigma via pseudo-reply LID (sequential full-sentence MLM); Stage2: token-prob aggregation NO renormalize.",
                )
                continue

            print(f"[Stage2 σ={sigma:.4f}] Processing {src_lang} with {len(texts)} ALL samples ...")
            sum_lang_probs = defaultdict(float)
            num_valid = 0

            for text in texts:
                enc = tokenizer(text, return_tensors="pt", truncation=True)
                input_ids = enc["input_ids"]
                attention_mask = enc.get("attention_mask", None)

                ids_1d = input_ids[0]

                # Stage2 仍然保持原逻辑：随机挑一个非 special 位置 mask
                # （你要求只改 Stage1）
                # 这里不使用外部 pick_random_non_special_position，直接按原函数调用方式保持一致：
                from utils.common import pick_random_non_special_position
                mask_pos = pick_random_non_special_position(ids_1d, special_ids)
                if mask_pos is None:
                    continue

                masked_input_ids = input_ids.clone()
                masked_input_ids[0, mask_pos] = tokenizer.mask_token_id

                masked_input_ids = masked_input_ids.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                with torch.no_grad():
                    logits = model(input_ids=masked_input_ids, attention_mask=attention_mask).logits

                mask_logits = logits[0, mask_pos].float().cpu()
                probs = torch.softmax(mask_logits, dim=-1)

                lang_probs = aggregate_token_probs_to_language_probs(
                    probs,
                    token_lang_weights,
                    allowed_langs=ALLOWED_LANGS,
                    renormalize=False,  # ✅ 关掉归一化
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
                notes="Stage1: mean first-confuse sigma via pseudo-reply LID (sequential full-sentence MLM); Stage2: token-prob aggregation NO renormalize.",
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n[Main] Done. escape_sigma + similarity saved.")


if __name__ == "__main__":
    main()
