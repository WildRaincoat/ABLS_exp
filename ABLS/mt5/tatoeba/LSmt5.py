#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import random
from collections import defaultdict
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import unicodedata  # ✅ NEW: 用于过滤标点/符号 token

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

from utils.common45 import (
    LANG_CODE_TO_LID_LABEL,
    TARGET_LID_LABELS,

    load_shared_token_probs,

    load_lid_model,
    detect_language,

    build_token_language_weights,

    ensure_pad_token,
    get_special_token_ids,

    aggregate_token_probs_to_language_probs,

    load_escape_sigma,
    save_escape_sigma,
    load_half_similarity,
    save_similarity_progress,

    # ✅ Tatoeba loader（按你 XLM 脚本一样的方式）
    load_texts_from_tatoeba_csv,
)

# ✅ 用你提供的 noisy_mt5.py
from utils.noisy_mt5 import add_activation_noise_to_t5_ffn


# ==========================
# Config (mT5-large)
# ==========================
MODEL_NAME = "google/mt5-large"

# ✅ CHANGED: Tatoeba dataset
TATOEBA_CSV = "/ACL_exp/data/tatoeba/sentences.csv"

OUTPUT_DIR = "/ACL_exp/Models_LPD/mt5/tatoeba-st-mc4-45"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIMILARITY_JSON = os.path.join(OUTPUT_DIR, "language_similarity_mt5_ENta.json")
ESCAPE_NOISE_JSON = os.path.join(OUTPUT_DIR, "escape_noise_mt5_ENta.json")

TOKEN_LANG_WEIGHTS_JSON = os.path.join(OUTPUT_DIR, "token_lang_weights_mt5.json")
SHARED_TOKENS_JSON = os.path.join(OUTPUT_DIR, "mt5large_shared_tokens_v2_45.json")

LID_MODEL_NAME = "juliensimon/xlm-v-base-language-id"

SAMPLES_PER_LANG_ESCAPE = 100
SIGMA_GRID = np.arange(0.5, 10, 0.5).tolist()
SEED = 42

ALLOWED_LANGS = set(TARGET_LID_LABELS)

MAX_BASELINE_TRIES_MULT = 10  # baseline 样本最多尝试倍数

# ✅ NEW: Stage2 mask 过滤阈值（“太短的不 mask”）
MIN_TOKEN_TEXT_LEN_TO_MASK = 3  # 你如果觉得还是太宽松，可以改成 3

# ✅ NEW: Tatoeba 很大：每种语言最多抽多少条 all 样本（避免 Stage2 跑爆）
MAX_SAMPLES_PER_LANG_ALL = 2000


# ==========================
# Data loading (Tatoeba)
# ==========================
def load_texts_all_and_small_tatoeba():
    """
    ✅ CHANGED: 从 Tatoeba CSV 加载数据
    - 复用 common.py 里的 load_texts_from_tatoeba_csv
    - 每语言 all 样本做 cap（默认 2000）
    - small 样本维持 SAMPLES_PER_LANG_ESCAPE
    """
    print(f"\n[Data] Loading Tatoeba CSV from: {TATOEBA_CSV}")

    lang_to_all, lang_to_small = load_texts_from_tatoeba_csv(
        csv_path=TATOEBA_CSV,
        target_labels=TARGET_LID_LABELS,
        samples_per_lang_small=SAMPLES_PER_LANG_ESCAPE,
        seed=SEED,
        min_len=1,
    )

    # ✅ cap all
    for lab in TARGET_LID_LABELS:
        if lab not in lang_to_all:
            lang_to_all[lab] = []
        if len(lang_to_all[lab]) > MAX_SAMPLES_PER_LANG_ALL:
            lang_to_all[lab] = lang_to_all[lab][:MAX_SAMPLES_PER_LANG_ALL]

        if lab not in lang_to_small:
            lang_to_small[lab] = []
        if len(lang_to_small[lab]) > SAMPLES_PER_LANG_ESCAPE:
            lang_to_small[lab] = lang_to_small[lab][:SAMPLES_PER_LANG_ESCAPE]

    print("\n[Data] Samples per language (after cap):")
    for lab in TARGET_LID_LABELS:
        print(f"  - {lab}: {len(lang_to_small[lab])} small / {len(lang_to_all[lab])} all (cap={MAX_SAMPLES_PER_LANG_ALL})")

    return lang_to_all, lang_to_small


# ==========================
# Model builder (Seq2Seq + noise)
# ==========================
def build_noisy_mt5(sigma: float, device: torch.device):
    print(f"\n[Model] Loading {MODEL_NAME} with sigma={sigma} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    tokenizer = ensure_pad_token(tokenizer)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )

    patched = add_activation_noise_to_t5_ffn(model, float(sigma))
    print(f"[Debug] sigma={sigma:.2f}, patched_ffn={patched}")

    model.to(device)
    model.eval()

    # stochasticity check (updated to match new decoding: [start, <extra_id_0>] and take next-step logits)
    with torch.no_grad():
        enc = tokenizer("hello", return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(device)

        extra0_id = _get_extra_id0(tokenizer)
        dec_start = model.config.decoder_start_token_id
        if dec_start is None:
            dec_start = tokenizer.pad_token_id

        decoder_input_ids = torch.tensor([[int(dec_start), int(extra0_id)]], device=device)

        out1 = model(input_ids=input_ids, attention_mask=attn, decoder_input_ids=decoder_input_ids).logits
        out2 = model(input_ids=input_ids, attention_mask=attn, decoder_input_ids=decoder_input_ids).logits

        # take last step logits (step 1, the "fill token" step)
        step1_1 = out1[0, -1]
        step1_2 = out2[0, -1]
        diff = (step1_1 - step1_2).abs().mean().item()

    print(f"[Debug] sigma={sigma:.2f}, mean |fill_step_logits1 - fill_step_logits2| = {diff:.6e}")

    return tokenizer, model


# ==========================
# mT5 infilling helpers
# ==========================
def _get_extra_id0(tokenizer) -> int:
    # 1) 先尝试常规方式
    tid = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    if tid is not None and tid != tokenizer.unk_token_id:
        return int(tid)

    # 2) 尝试从 additional_special_tokens 里找
    if hasattr(tokenizer, "additional_special_tokens") and tokenizer.additional_special_tokens:
        for t in tokenizer.additional_special_tokens:
            if "extra_id_0" in t:
                tid2 = tokenizer.convert_tokens_to_ids(t)
                if tid2 is not None and tid2 != tokenizer.unk_token_id:
                    return int(tid2)

    # 3) 兜底：T5/mT5 的约定：<extra_id_0> 在 vocab 最后一个位置
    fallback = int(tokenizer.vocab_size - 1)

    # 4) 校验一下
    tok = tokenizer.convert_ids_to_tokens([fallback])[0]
    # print(f"[WARN] convert_tokens_to_ids('<extra_id_0>') failed. Fallback extra0_id={fallback}, token='{tok}'")
    return fallback


def _decode_first_fill_token_from_mt5(
    tokenizer,
    model,
    device,
    input_ids_1d: torch.Tensor,
    attention_mask_1d: Optional[torch.Tensor],
) -> Tuple[int, torch.Tensor]:
    """
    ✅ decoder_input_ids = [decoder_start_token_id, <extra_id_0>]
    并取最后一步 logits 作为 fill-token 分布。
    """
    input_ids = input_ids_1d.unsqueeze(0).to(device)
    attn = attention_mask_1d.unsqueeze(0).to(device) if attention_mask_1d is not None else None

    dec_start_id = model.config.decoder_start_token_id
    if dec_start_id is None:
        dec_start_id = tokenizer.pad_token_id

    extra0_id = _get_extra_id0(tokenizer)

    decoder_input_ids = torch.tensor([[int(dec_start_id), int(extra0_id)]], device=device)

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attn,
            decoder_input_ids=decoder_input_ids,
        ).logits

    step_logits = logits[0, -1].float().cpu()  # [V]
    probs = torch.softmax(step_logits, dim=-1)
    pred_id = int(torch.argmax(step_logits).item())
    return pred_id, probs


# ==========================
# Stage1: pseudo-reply (sequential full sentence)
# ==========================
def pseudo_reply_mt5_full_sentence_sequential(
    tokenizer,
    model,
    device,
    text: str,
) -> str:
    enc = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"][0]          # [L]
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask[0]   # [L]

    special_ids = get_special_token_ids(tokenizer)
    extra0_id = _get_extra_id0(tokenizer)

    ids = input_ids.clone()

    for pos in range(ids.size(0)):
        if int(ids[pos].item()) in special_ids:
            continue

        masked = ids.clone()
        masked[pos] = extra0_id

        pred_id, _ = _decode_first_fill_token_from_mt5(
            tokenizer=tokenizer,
            model=model,
            device=device,
            input_ids_1d=masked,
            attention_mask_1d=attention_mask,
        )

        ids[pos] = pred_id

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
    picked = []
    tries = 0
    max_tries = max(needed * MAX_BASELINE_TRIES_MULT, needed)

    for text in all_texts:
        if len(picked) >= needed:
            break
        tries += 1
        if tries > max_tries:
            break

        reply = pseudo_reply_mt5_full_sentence_sequential(tokenizer0, model0, device, text)
        if not reply:
            continue

        out_lang = detect_text_lang_label(reply, lid_tokenizer, lid_model, lid_id2label, lid_device)
        if out_lang == src_lang:
            picked.append(text)

    if len(picked) < needed:
        print(f"[Stage1][Baseline] {src_lang}: picked {len(picked)}/{needed} (tries={tries}, pool={len(all_texts)})")
    else:
        print(f"[Stage1][Baseline] {src_lang}: picked {len(picked)}/{needed}")

    return picked


# ==========================
# ✅ Stage1 NEW: round-robin sigma search (wheel search)
# ==========================
def compute_escape_sigma_round_robin(
    langs: List[str],
    baseline_samples_by_lang: Dict[str, List[str]],
    device,
    lid_tokenizer,
    lid_model,
    lid_id2label,
    lid_device,
    escape_sigma: Dict[str, Optional[float]],
):
    """
    轮式搜索：
    - 对每个 sigma 只加载一次 noisy 模型
    - 对所有“尚未完成”的语言同时推进
    - 某语言的所有 baseline samples 都首次被 confuse 后：
        escape_sigma[lang] = mean(first_confuse_sigma_per_sample)
        然后从 active set 里移除
    """
    # 每个语言：每个样本第一次 confuse 的 sigma
    first_confuse_sigma: Dict[str, List[Optional[float]]] = {}
    # 记录累计 confuse 数（便于你调试/可视化）
    confused_counts: Dict[str, Dict[float, int]] = {}

    active_langs = []
    for lang in langs:
        samples = baseline_samples_by_lang.get(lang, [])
        if not samples:
            escape_sigma[lang] = SIGMA_GRID[-1]
            print(f"[Stage1][RR] {lang}: 0 baseline samples -> escape_sigma={escape_sigma[lang]:.2f}")
            continue
        first_confuse_sigma[lang] = [None] * len(samples)
        confused_counts[lang] = {}
        active_langs.append(lang)

    if not active_langs:
        return escape_sigma

    print(f"\n[Stage1][RR] Round-robin search start. Active langs={len(active_langs)}")

    for sigma in SIGMA_GRID:
        if not active_langs:
            break

        tokenizer, model = build_noisy_mt5(float(sigma), device)

        # 对每个 active lang 推进：只测“尚未 confuse”的样本
        next_active = []
        for lang in active_langs:
            samples = baseline_samples_by_lang[lang]
            slots = first_confuse_sigma[lang]

            newly_confused = 0
            for i, text in enumerate(samples):
                if slots[i] is not None:
                    continue

                reply = pseudo_reply_mt5_full_sentence_sequential(tokenizer, model, device, text)
                out_lang = detect_text_lang_label(reply, lid_tokenizer, lid_model, lid_id2label, lid_device)

                if out_lang != lang:
                    slots[i] = float(sigma)
                    newly_confused += 1

            # 统计当前 sigma 时累计 confuse
            total_confused = sum(1 for x in slots if x is not None)
            confused_counts[lang][float(sigma)] = int(total_confused)

            # 如果该语言所有样本都已 confuse，则可以定 escape_sigma 并移出 active
            if total_confused >= len(slots):
                mean_sigma = float(sum(slots) / len(slots))  # slots 已无 None
                escape_sigma[lang] = mean_sigma
                print(f"[Stage1][RR][Done@{sigma:.2f}] {lang}: escape_sigma(mean)={mean_sigma:.4f}, baseline_n={len(slots)}")
                save_escape_sigma(ESCAPE_NOISE_JSON, escape_sigma)
            else:
                # 还没完成，继续留在 active
                if newly_confused > 0:
                    print(f"[Stage1][RR][σ={sigma:.2f}] {lang}: +{newly_confused}, confused={total_confused}/{len(slots)}")
                next_active.append(lang)

        # 清理
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        active_langs = next_active

    # 兜底：如果 sigma_grid 跑完还有没完成的语言，把剩余 None 填成 max sigma 再求均值
    for lang in active_langs:
        slots = first_confuse_sigma[lang]
        for i in range(len(slots)):
            if slots[i] is None:
                slots[i] = float(SIGMA_GRID[-1])
        mean_sigma = float(sum(slots) / len(slots))
        escape_sigma[lang] = mean_sigma
        print(f"[Stage1][RR][Fallback] {lang}: not fully confused -> set remaining to {SIGMA_GRID[-1]:.2f}, escape_sigma={mean_sigma:.4f}")
        save_escape_sigma(ESCAPE_NOISE_JSON, escape_sigma)

    return escape_sigma


# ==========================
# Stage2 helper (mT5):
# random position -> first fill token distribution
# ==========================
def _normalize_spm_piece(piece: str) -> str:
    # sentencepiece 常见前缀 ▁ 表示空格
    if piece is None:
        return ""
    s = piece.replace("▁", "").strip()
    return s


def _is_punct_or_symbol_only(s: str) -> bool:
    """
    True 表示：字符串里所有非空字符都属于 Unicode P(标点) 或 S(符号) 类别
    """
    if not s:
        return True
    has_any = False
    for ch in s:
        if ch.isspace():
            continue
        has_any = True
        cat = unicodedata.category(ch)  # e.g. 'P*', 'S*', 'L*', 'N*'
        if not (cat.startswith("P") or cat.startswith("S")):
            return False
    return has_any


def is_good_mask_token(tokenizer, token_id: int, min_len: int = MIN_TOKEN_TEXT_LEN_TO_MASK) -> bool:
    """
    ✅ NEW: Stage2 mask 过滤规则
    - 不 mask 字符（长度太短）
    - 不 mask 太短 token
    - 不 mask 纯标点/符号 token
    """
    piece = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
    norm = _normalize_spm_piece(piece)

    # 1) 太短（含“单字符”）直接跳过
    if len(norm) < int(min_len):
        return False

    # 2) 纯标点/符号（比如 "," "." "!" "€" 等）跳过
    if _is_punct_or_symbol_only(norm):
        return False

    return True


def pick_random_maskable_position_1d(
    tokenizer,
    ids_1d: torch.Tensor,
    special_ids: set,
) -> Optional[int]:
    """
    ✅ NEW: 只从“可 mask 的 token”里随机选位置
    """
    candidates = []
    for i in range(ids_1d.numel()):
        tid = int(ids_1d[i].item())
        if tid in special_ids:
            continue
        if not is_good_mask_token(tokenizer, tid, min_len=MIN_TOKEN_TEXT_LEN_TO_MASK):
            continue
        candidates.append(i)

    if not candidates:
        return None
    return int(random.choice(candidates))


def mt5_fill_token_probs_at_random_position(
    tokenizer,
    model,
    device,
    text: str,
    special_ids: set,
) -> Optional[torch.Tensor]:
    """
    Stage2:
    - 选一个“满足过滤条件”的随机非 special 位置 pos
    - 将该位置替换成 <extra_id_0>
    - 用 decoder “内容 token”一步 logits 得到“填空 token”的分布 probs
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"][0]
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask[0]

    if input_ids.numel() < 2:
        return None

    # ✅ CHANGED: 只挑选“可 mask”的位置
    pos = pick_random_maskable_position_1d(tokenizer, input_ids, special_ids)
    if pos is None:
        return None

    extra0_id = _get_extra_id0(tokenizer)

    masked = input_ids.clone()
    masked[pos] = extra0_id

    _, probs = _decode_first_fill_token_from_mt5(
        tokenizer=tokenizer,
        model=model,
        device=device,
        input_ids_1d=masked,
        attention_mask_1d=attention_mask,
    )
    return probs


# ==========================
# Main
# ==========================
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lid_device = torch.device("cpu")

    # 1) LID model
    print("[Main] Loading LID model ...")
    lid_tokenizer, lid_model, lid_id2label = load_lid_model(LID_MODEL_NAME)
    lid_model.to(lid_device)

    # 2) shared token probs
    shared_token_probs = load_shared_token_probs(SHARED_TOKENS_JSON)

    # 3) token->language weights
    base_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    base_tokenizer = ensure_pad_token(base_tokenizer)

    token_lang_weights = build_token_language_weights(
        tokenizer=base_tokenizer,
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

    # 4) dataset  ✅ CHANGED: Flores -> Tatoeba
    lang_to_all_texts, _ = load_texts_all_and_small_tatoeba()

    src_langs = [lab for lab in TARGET_LID_LABELS if len(lang_to_all_texts.get(lab, [])) > 0]
    print("\n[Main] Source languages with samples:")
    for lab in src_langs:
        print(f"  - {lab}: {len(lang_to_all_texts[lab])} all (cap={MAX_SAMPLES_PER_LANG_ALL})")

    # ==================
    # Stage 1: escape sigma  ✅ round-robin search
    # ==================
    escape_sigma = load_escape_sigma(ESCAPE_NOISE_JSON, src_langs)
    langs_needing_escape = [l for l in src_langs if escape_sigma[l] is None]
    print(f"\n[Stage1] Languages needing escape-noise search: {len(langs_needing_escape)}")

    if langs_needing_escape:
        tok0, model0 = build_noisy_mt5(0.0, device)

        # 先为每个语言挑 baseline samples（只做一次）
        baseline_samples_by_lang: Dict[str, List[str]] = {}
        for src_lang in langs_needing_escape:
            pool = lang_to_all_texts.get(src_lang, [])
            if not pool:
                baseline_samples_by_lang[src_lang] = []
                continue

            baseline_samples_by_lang[src_lang] = pick_baseline_consistent_samples(
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

        # 用轮式搜索推进所有语言
        escape_sigma = compute_escape_sigma_round_robin(
            langs=langs_needing_escape,
            baseline_samples_by_lang=baseline_samples_by_lang,
            device=device,
            lid_tokenizer=lid_tokenizer,
            lid_model=lid_model,
            lid_id2label=lid_id2label,
            lid_device=lid_device,
            escape_sigma=escape_sigma,
        )

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
    # Stage 2: half similarity (NO renormalize)
    # ==================
    half_sim = load_half_similarity(SIMILARITY_JSON, src_langs)

    sigma_to_langs = defaultdict(list)
    for lang in src_langs:
        sigma_to_langs[escape_sigma[lang]].append(lang)

    print("\n[Stage2] Compute half similarity at sigma = escape_sigma[lang], using ALL (capped) samples. (NO renormalize)")
    for sigma in sorted(sigma_to_langs.keys()):
        langs_at_sigma = [l for l in sigma_to_langs[sigma] if l not in half_sim]
        if not langs_at_sigma:
            continue

        print(f"\n[Stage2] ===== Sigma = {sigma:.4f} ===== for languages: {langs_at_sigma}")
        tokenizer, model = build_noisy_mt5(float(sigma), device)
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
                    notes="Tatoeba; Stage1: mean first-confuse sigma via sequential <extra_id_0> infilling + LID; Stage2: random-position <extra_id_0> fill-token prob aggregation NO renormalize. Noise: utils/noisy_mt5.py. Stage2 mask filtering: skip short/punct/symbol tokens.",
                )
                continue

            print(f"[Stage2 σ={sigma:.4f}] Processing {src_lang} with {len(texts)} ALL samples (cap={MAX_SAMPLES_PER_LANG_ALL}) ...")
            sum_lang_probs = defaultdict(float)
            num_valid = 0

            for text in texts:
                probs = mt5_fill_token_probs_at_random_position(
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    text=text,
                    special_ids=special_ids,
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
                notes="Tatoeba; Stage1: mean first-confuse sigma via sequential <extra_id_0> infilling + LID; Stage2: random-position <extra_id_0> fill-token prob aggregation NO renormalize. Noise: utils/noisy_mt5.py. Stage2 mask filtering: skip short/punct/symbol tokens.",
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n[Main] Done. escape_sigma + similarity saved.")


if __name__ == "__main__":
    main()
