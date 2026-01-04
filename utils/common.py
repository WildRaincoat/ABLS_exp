#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import string
import unicodedata
import random
from collections import Counter, defaultdict
from typing import Dict, Tuple, Callable, Any, Optional, Iterable, List, Set

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ==========================
# 0) Experiment language set (fixed 34 langs)
# ==========================

LANG_CODE_TO_LID_LABEL = {
    # ===== 已有语言（保留） =====
    "mya_Mymr": "Burmese",
    "fin_Latn": "Finnish",
    "hun_Latn": "Hungarian",
    "ron_Latn": "Romanian",
    "fra_Latn": "French",
    "ita_Latn": "Italian",
    "por_Latn": "Portuguese",
    "spa_Latn": "Spanish",
    "deu_Latn": "German",
    "dan_Latn": "Danish",
    "eng_Latn": "English",
    "tur_Latn": "Turkish",
    "ind_Latn": "Indonesian",
    "zsm_Latn": "Malay",
    "swh_Latn": "Swahili",
    "lit_Latn": "Lithuanian",
    "pol_Latn": "Polish",
    "mar_Deva": "Marathi",
    "mal_Mlym": "Malayalam",
    "tam_Taml": "Tamil",
    "ben_Beng": "Bengali",
    "heb_Hebr": "Hebrew",
    "zho_Hant": "Chinese",
    "zho_Hans": "Chinese",
    "vie_Latn": "Vietnamese",
    "kaz_Cyrl": "Kazakh",
    "kir_Cyrl": "Kyrgyz",
    "bul_Cyrl": "Bulgarian",
    "rus_Cyrl": "Russian",
    "bel_Cyrl": "Belarusian",
    "ukr_Cyrl": "Ukrainian",
    "ell_Grek": "Greek",
    "hye_Armn": "Armenian",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",

    # ===== 新增语言（补充） =====
    # "afr_Latn": "Afrikaans",
    # "arb_Arab": "Arabic",
    # "azj_Latn": "Azerbaijani",
    # "eus_Latn": "Basque",
    # "est_Latn": "Estonian",
    # "kat_Geor": "Georgian",
    # "hin_Deva": "Hindi",
    # "jav_Latn": "Javanese",
    # "lvs_Latn": "Latvian",
    # "khk_Cyrl": "Mongolian",
    # "pes_Arab": "Persian",
    # "swe_Latn": "Swedish",
    # "tel_Telu": "Telugu",
    # "tha_Thai": "Thai",
    # "uzn_Latn": "Uzbek",
    # "yor_Latn": "Yoruba",
}


# ✅ 仍保留 symbols label（用于 detect_language / token 权重表里标注 symbols）
SYMBOL_LANG_LABEL = "symbols"

LID_RAW_TO_CANONICAL = {
    "Mandarin Chinese": "Chinese",
    "Cantonese Chinese": "Chinese",
}

ALL_LANG_CODES = list(LANG_CODE_TO_LID_LABEL.keys())
# ✅ 只保留 34 语言，不包含 symbols
TARGET_LID_LABELS = sorted(set(LANG_CODE_TO_LID_LABEL.values()))


def canonicalize_lid_label(raw_label: str) -> str:
    return LID_RAW_TO_CANONICAL.get(raw_label, raw_label)


# ==========================
# 1) Shared token 表
# ==========================
def load_shared_token_probs(path: str) -> Dict[str, Dict[str, float]]:
    if not os.path.exists(path):
        print(f"[WARN] Shared token file not found: {path}, skip shared-token logic.")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    shared_probs: Dict[str, Dict[str, float]] = {}
    for tok_id_str, info in raw.items():
        langs = info.get("langs", {})
        total = sum(langs.values())
        if total <= 0:
            continue
        shared_probs[tok_id_str] = {lang: cnt / total for lang, cnt in langs.items()}

    print(f"Loaded shared token table for {len(shared_probs)} tokens from {path}")
    return shared_probs


# ==========================
# 2) symbol / digit 检测
# ==========================
def _is_whitespace_or_control_only(text: str) -> bool:
    t = text or ""
    if t == "":
        return True
    for ch in t:
        cat = unicodedata.category(ch)
        if not (cat.startswith("Z") or cat.startswith("C")):
            return False
    return True


def is_symbol_or_digit_only(text: str) -> bool:
    """
    Treat as 'symbols' if token contains NO Unicode Letter (L*),
    EXCEPT tokens that are purely whitespace/control (Z*/C*), which we IGNORE:
      - whitespace/control-only -> False  (not symbols, not language)
      - otherwise: no letters -> True
    """
    t = (text or "")
    if t == "":
        return False
    if _is_whitespace_or_control_only(t):
        return False
    for ch in t:
        if unicodedata.category(ch).startswith("L"):
            return False
    return True


def is_short_latin_fragment(text: str, max_len: int = 2) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    if any(ch not in string.ascii_letters for ch in letters):
        return False
    return len(letters) <= max_len


# ==========================
# 3) XLM-V LID
# ==========================
def load_lid_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    id2label = model.config.id2label
    return tokenizer, model, id2label


def detect_language(
    text: str,
    lid_tokenizer,
    lid_model,
    id2label,
    device,
    short_latin_max_len: int = 2,
    top1_top2_margin: float = 0.1,
) -> Tuple[str, float]:
    text = (text or "").strip()
    if not text:
        return "unknown", 0.0

    if is_symbol_or_digit_only(text):
        return SYMBOL_LANG_LABEL, 1.0

    if is_short_latin_fragment(text, max_len=short_latin_max_len):
        return "unknown", 0.0

    try:
        inputs = lid_tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = lid_model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        topk = torch.topk(probs, k=min(2, probs.shape[-1]))
        top_indices = topk.indices.tolist()
        top_values = topk.values.tolist()

        top1_idx = top_indices[0]
        top1_prob = float(top_values[0])
        raw_label = id2label[top1_idx]
        top1_label = canonicalize_lid_label(raw_label)

        if len(top_indices) > 1:
            top2_prob = float(top_values[1])
            if top1_prob - top2_prob < top1_top2_margin:
                return "unknown", top1_prob

        return top1_label, top1_prob
    except Exception:
        return "unknown", 0.0


# ==========================
# 4) 分布与相似度
# ==========================
def normalize_counter(counter: Counter) -> Dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def cosine_similarity(p: Dict[str, float], q: Dict[str, float], all_keys=None) -> float:
    if all_keys is None:
        all_keys = set(p.keys()) | set(q.keys())
    v1 = [p.get(k, 0.0) for k in all_keys]
    v2 = [q.get(k, 0.0) for k in all_keys]

    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


# ==========================
# 5) sigma 输出目录命名
# ==========================
def sigma_to_dirname(sigma: float) -> str:
    if sigma == 0:
        return "0"
    if sigma < 0.001:
        return f"{sigma:.0e}".replace("+", "")
    return str(sigma).replace(".", "p")


# ==========================
# 6) Sanity check
# ==========================
def sanity_check_logits(get_logits_fn: Callable[[], torch.Tensor], sigma: float):
    print("=== Sanity check: same input twice ===")
    print(f"sigma: {sigma}")

    with torch.no_grad():
        logits1 = get_logits_fn()
        logits2 = get_logits_fn()

    print("logits allclose:", torch.allclose(logits1, logits2))
    diff = (logits1 - logits2).abs()
    print("max abs diff:", diff.max().item())
    print("any nan in logits1:", torch.isnan(logits1).any().item())
    print("any nan in logits2:", torch.isnan(logits2).any().item())


# ==========================
# 8) JSON helpers
# ==========================
def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load json: {path} ({e})")
        return default


def save_json_atomic(obj, path: str, indent: int = 2, ensure_ascii: bool = False):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=ensure_ascii, indent=indent)
    os.replace(tmp_path, path)


# ==========================
# 9) Tokenizer helpers
# ==========================
def ensure_pad_token(tokenizer):
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            tokenizer.pad_token_id = eos
    return tokenizer


def get_special_token_ids(tokenizer) -> set:
    ids = set()
    for name in ["cls_token_id", "sep_token_id", "pad_token_id", "bos_token_id", "eos_token_id"]:
        v = getattr(tokenizer, name, None)
        if v is not None:
            ids.add(int(v))
    if hasattr(tokenizer, "all_special_ids"):
        ids |= set(int(x) for x in tokenizer.all_special_ids)
    return ids


def pick_candidate_positions_1d(input_ids_1d: torch.Tensor, special_ids: set) -> list:
    return [i for i in range(int(input_ids_1d.size(0))) if int(input_ids_1d[i].item()) not in special_ids]


def pick_random_non_special_position(input_ids_1d: torch.Tensor, special_ids: set) -> Optional[int]:
    cands = pick_candidate_positions_1d(input_ids_1d, special_ids)
    if not cands:
        return None
    return random.choice(cands)


# ==========================
# 10) token -> language weights
# ==========================
def build_token_language_weights(
    tokenizer,
    shared_token_probs: Dict[str, Dict[str, float]],
    lid_tokenizer,
    lid_model,
    lid_id2label,
    lid_device: torch.device,
    target_labels: Iterable[str],
    symbol_label: str = "symbols",
    unknown_label: str = "unknown",
    cache_path: Optional[str] = None,
    short_latin_max_len: int = 2,
    top1_top2_margin: float = 0.1,
    progress_step_percent: int = 5,
) -> Dict[int, Dict[str, float]]:
    """
    token_lang_weights maps token_id -> {label: weight}

    label is one of:
      - 34 target languages
      - symbol_label ("symbols")
      - unknown_label ("unknown")

    Design goal: for (almost) every normal token, assign at least one bucket.
    We still skip:
      - special tokens
      - whitespace/control-only tokens
    """
    target_labels = set(target_labels)

    if cache_path is not None and os.path.exists(cache_path):
        print(f"[TokenLang] Loading cached token→language weights: {cache_path}")
        raw = load_json(cache_path, default={}) or {}
        out = {int(k): v for k, v in raw.items()}
        print(f"[TokenLang] Loaded {len(out)} tokens from cache.")
        return out

    vocab_size = int(getattr(tokenizer, "vocab_size", 0))
    if vocab_size <= 0:
        raise ValueError("Tokenizer has invalid vocab_size.")

    token_lang_weights: Dict[int, Dict[str, float]] = {}
    special_ids = get_special_token_ids(tokenizer)

    print(f"[TokenLang] Building token→language weights (vocab_size={vocab_size}) ...")
    next_percent = progress_step_percent

    for tok_id in range(vocab_size):
        tok_id_str = str(tok_id)

        # (0) skip special ids
        if tok_id in special_ids:
            pass

        # (1) shared token table: keep target langs, put the rest into unknown
        elif tok_id_str in shared_token_probs:
            raw_w = {k: float(v) for k, v in shared_token_probs[tok_id_str].items()}
            # raw_w should already sum ~1.0, but be robust:
            s_raw = sum(raw_w.values())
            if s_raw > 0:
                raw_w = {k: v / s_raw for k, v in raw_w.items()}

            w_target = {k: v for k, v in raw_w.items() if k in target_labels}
            s_target = sum(w_target.values())

            w_out = {}
            if s_target > 0:
                w_out.update(w_target)

            # leftover mass -> unknown
            unk = 1.0 - s_target
            if unk < 0.0:
                unk = 0.0
            if unk > 0.0:
                w_out[unknown_label] = unk

            if w_out:
                token_lang_weights[tok_id] = w_out

        else:
            # (2) decode token text
            text = tokenizer.decode([tok_id], skip_special_tokens=True) or ""

            # ignore pure whitespace/control tokens
            if _is_whitespace_or_control_only(text):
                pass
            else:
                # symbols bucket
                if is_symbol_or_digit_only(text):
                    token_lang_weights[tok_id] = {symbol_label: 1.0}

                # short latin fragment -> unknown (your existing rule)
                elif is_short_latin_fragment(text, max_len=short_latin_max_len):
                    token_lang_weights[tok_id] = {unknown_label: 1.0}

                else:
                    lid_label, _ = detect_language(
                        text,
                        lid_tokenizer,
                        lid_model,
                        lid_id2label,
                        lid_device,
                        short_latin_max_len=short_latin_max_len,
                        top1_top2_margin=top1_top2_margin,
                    )

                    if lid_label in target_labels:
                        token_lang_weights[tok_id] = {lid_label: 1.0}
                    elif lid_label == symbol_label:
                        token_lang_weights[tok_id] = {symbol_label: 1.0}
                    else:
                        token_lang_weights[tok_id] = {unknown_label: 1.0}

        progress = int((tok_id + 1) * 100 / vocab_size)
        if progress >= next_percent:
            print(f"  [TokenLang] {progress}% ({tok_id+1}/{vocab_size})")
            next_percent += progress_step_percent
            if next_percent > 100:
                next_percent = 101

    print(f"[TokenLang] Got weights for {len(token_lang_weights)} tokens.")

    if cache_path is not None:
        raw = {str(k): v for k, v in token_lang_weights.items()}
        print(f"[TokenLang] Saving cache: {cache_path}")
        save_json_atomic(raw, cache_path, indent=2, ensure_ascii=False)

    return token_lang_weights



# ==========================
# 11) Aggregate token probs -> language probs
# ==========================
def aggregate_token_probs_to_language_probs(
    token_probs_1d: torch.Tensor,
    token_lang_weights: Dict[int, Dict[str, float]],
    allowed_langs: Optional[set] = None,
    renormalize: bool = True,
) -> Dict[str, float]:
    lang_probs = defaultdict(float)
    vocab_size = int(token_probs_1d.size(0))
    for tid in range(vocab_size):
        w_dict = token_lang_weights.get(tid)
        if not w_dict:
            continue
        p = float(token_probs_1d[tid].item())
        if p == 0.0:
            continue
        for lang, w in w_dict.items():
            if allowed_langs is not None and lang not in allowed_langs:
                continue
            lang_probs[lang] += p * float(w)

    out = dict(lang_probs)
    if renormalize:
        s = sum(out.values())
        if s > 0:
            out = {k: v / s for k, v in out.items()}
    return out


# ==========================
# 12) Progress printing helper
# ==========================
def should_print_progress(idx: int, total: int, next_percent: int) -> Tuple[bool, int, int]:
    if total <= 0:
        return False, 0, next_percent
    progress = int((idx + 1) * 100 / total)
    if progress >= next_percent:
        new_next = next_percent + 5
        if new_next > 100:
            new_next = 101
        return True, progress, new_next
    return False, progress, next_percent


# ==========================
# 13) Resume/save patterns for escape_sigma & half_similarity
# ==========================
def load_escape_sigma(path: str, languages: Iterable[str]) -> Dict[str, Optional[float]]:
    esc = {lang: None for lang in languages}
    data = load_json(path, default={}) or {}
    for lang in languages:
        if lang in data:
            esc[lang] = data[lang]
    return esc


def save_escape_sigma(path: str, escape_sigma: Dict[str, Optional[float]]):
    to_save = {k: v for k, v in escape_sigma.items() if v is not None}
    save_json_atomic(to_save, path, indent=2, ensure_ascii=False)


def load_half_similarity(path: str, languages: Iterable[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    data = load_json(path, default={}) or {}
    half = data.get("half_similarity", {}) if isinstance(data, dict) else {}
    for lang in languages:
        if lang in half:
            out[lang] = half[lang]
    return out


def save_similarity_progress(
    path: str,
    escape_sigma: Dict[str, float],
    half_sim: Dict[str, Dict[str, float]],
    languages: Iterable[str],
    sigma_grid=None,
    notes: str = "",
):
    languages = list(languages)
    completed = [l for l in languages if l in half_sim]

    full_sim = {}
    for i, l1 in enumerate(completed):
        for l2 in completed[i + 1:]:
            v = float(half_sim.get(l1, {}).get(l2, 0.0)) + float(half_sim.get(l2, {}).get(l1, 0.0))
            full_sim[f"{l1}__{l2}"] = v

    out = {
        "half_similarity": half_sim,
        "full_similarity": full_sim,
        "target_languages": languages,
        "sigma_grid": sigma_grid,
        "escape_sigma": {k: float(v) for k, v in escape_sigma.items()},
        "notes": notes,
    }
    save_json_atomic(out, path, indent=2, ensure_ascii=False)


# ==========================
# (NEW) Dataset code alias maps (Flores / Tatoeba) -> canonical labels
# ==========================

# Tatoeba 常见是 ISO639-3/639-1 混用；这里统一映射到你实验用的 canonical label
TATOEBA_CODE_TO_LABEL = {
    # Latin script langs
    "eng": "English",
    "en": "English",

    "deu": "German",
    "de": "German",

    "fra": "French",
    "fr": "French",

    "ita": "Italian",
    "it": "Italian",

    "spa": "Spanish",
    "es": "Spanish",

    "por": "Portuguese",
    "pt": "Portuguese",

    "fin": "Finnish",
    "hu": "Hungarian",
    "hun": "Hungarian",
    "ron": "Romanian",
    "ro": "Romanian",
    "dan": "Danish",
    "da": "Danish",
    "tur": "Turkish",
    "tr": "Turkish",
    "ind": "Indonesian",
    "id": "Indonesian",
    "zsm": "Malay",   # 你截图里出现的
    "msa": "Malay",
    "ms": "Malay",
    "swh": "Swahili",
    "sw": "Swahili",
    "lit": "Lithuanian",
    "lt": "Lithuanian",
    "pol": "Polish",
    "pl": "Polish",

    # Indic / other scripts
    "mar": "Marathi",
    "mal": "Malayalam",
    "tam": "Tamil",
    "ben": "Bengali",
    "heb": "Hebrew",
    "he": "Hebrew",

    # Chinese variants in Tatoeba
    "cmn": "Chinese",   # 你截图里出现的
    "zho": "Chinese",
    "zh": "Chinese",
    "yue": "Chinese",   # 粤语也统一算 Chinese（与你 LID canonical 一致）

    # Cyrillic / others
    "vie": "Vietnamese",
    "vi": "Vietnamese",
    "kaz": "Kazakh",
    "kir": "Kyrgyz",
    "bul": "Bulgarian",
    "rus": "Russian",
    "ru": "Russian",
    "bel": "Belarusian",
    "ukr": "Ukrainian",
    "ell": "Greek",
    "el": "Greek",
    "hye": "Armenian",
    "hy": "Armenian",
    "jpn": "Japanese",
    "ja": "Japanese",
    "kor": "Korean",
    "ko": "Korean",

    # Burmese
    "mya": "Burmese",
    "my": "Burmese",

#--------------------------new--------------------------------------------
    # # Afrikaans
    # "afr": "Afrikaans",
    # "af": "Afrikaans",

    # # Arabic (Modern Standard Arabic in many datasets: "ara")
    # "arb": "Arabic",   # in case dataset uses arb
    # "ara": "Arabic",
    # "ar": "Arabic",

    # # Azerbaijani
    # "aze": "Azerbaijani",
    # "az": "Azerbaijani",

    # # Basque
    # "eus": "Basque",
    # "baq": "Basque",   # legacy ISO639-2
    # "eu": "Basque",

    # # Estonian
    # "est": "Estonian",
    # "et": "Estonian",

    # # Georgian
    # "kat": "Georgian",
    # "geo": "Georgian",  # legacy ISO639-2
    # "ka": "Georgian",

    # # Hindi
    # "hin": "Hindi",
    # "hi": "Hindi",

    # # Javanese
    # "jav": "Javanese",
    # "jv": "Javanese",

    # # Latvian
    # "lvs": "Latvian",
    # "lv": "Latvian",

    # # Mongolian
    # "mon": "Mongolian",
    # "mn": "Mongolian",

    # # Persian (Farsi)
    # "pes": "Persian",  # you used pes_Arab
    # "fas": "Persian",  # ISO639-2
    # "per": "Persian",  # legacy ISO639-2
    # "fa": "Persian",

    # # Swedish
    # "swe": "Swedish",
    # "sv": "Swedish",

    # # Telugu
    # "tel": "Telugu",
    # "te": "Telugu",

    # # Thai
    # "tha": "Thai",
    # "th": "Thai",


    # # Uzbek
    # "uzb": "Uzbek",
    # "uz": "Uzbek",

    # # Yoruba
    # "yor": "Yoruba",
    # "yo": "Yoruba",
}

def map_dataset_code_to_label(code: str) -> Optional[str]:
    """
    Map a dataset-specific language code to your canonical label (XLM-V style),
    e.g. "ita_Latn" -> "Italian", "ita" -> "Italian", "cmn" -> "Chinese".

    Returns None if unknown / unsupported.
    """
    if not code:
        return None
    code = code.strip()

    # Flores: "ita_Latn" style
    if code in LANG_CODE_TO_LID_LABEL:
        return LANG_CODE_TO_LID_LABEL[code]

    # Some callers may pass just the left part "ita" extracted from "ita_Latn"
    if "_" in code:
        left = code.split("_", 1)[0]
        if left in TATOEBA_CODE_TO_LABEL:
            return TATOEBA_CODE_TO_LABEL[left]

    # Tatoeba: "ita", "cmn", etc.
    if code in TATOEBA_CODE_TO_LABEL:
        return TATOEBA_CODE_TO_LABEL[code]

    return None

# ==========================
# (NEW) Tatoeba loader
# ==========================
def load_texts_from_tatoeba_csv(
    csv_path: str,
    target_labels: Iterable[str],
    samples_per_lang_small: int = 100,
    seed: int = 42,
    min_len: int = 1,
) -> Tuple[Dict[str, list], Dict[str, list]]:
    """
    Read Tatoeba sentences.csv and return:
      - lang_to_all[label]   : all sentences (label in target_labels)
      - lang_to_small[label] : first samples_per_lang_small after shuffle

    The CSV is expected to have 3 fields like:
      sentence_id, lang_code, sentence
    It can be tab-separated or comma-separated; this function tries to auto-detect.
    """
    import csv

    rng = random.Random(seed)
    target_labels = list(target_labels)
    target_set = set(target_labels)

    lang_to_all = {lab: [] for lab in target_labels}

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Tatoeba sentences csv not found: {csv_path}")

    # --- detect delimiter (tab vs comma) ---
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
            delim = dialect.delimiter
        except Exception:
            # most common for Tatoeba is tab
            delim = "\t"

        reader = csv.reader(f, delimiter=delim)

        # optional header detection
        first_row = None
        try:
            first_row = next(reader)
        except StopIteration:
            first_row = None

        def looks_like_header(row):
            if not row:
                return False
            joined = " ".join(row).lower()
            return ("sentence" in joined and "lang" in joined) or ("language" in joined)

        if first_row is not None and not looks_like_header(first_row):
            # treat as data row
            row_iter = [first_row]
            row_iter.extend(reader)
        else:
            row_iter = reader

        for row in row_iter:
            if not row:
                continue
            # be robust to extra columns
            if len(row) < 3:
                continue

            # typical: [id, lang, text]
            lang_code = (row[1] or "").strip()
            text = (row[2] or "").strip()
            if len(text) < min_len:
                continue

            label = map_dataset_code_to_label(lang_code)
            if label is None or label not in target_set:
                continue

            lang_to_all[label].append(text)

    # shuffle within each language (keep behavior similar to your Flores loader)
    lang_to_small = {lab: [] for lab in target_labels}
    for lab in target_labels:
        texts = lang_to_all[lab]
        rng.shuffle(texts)
        lang_to_small[lab] = texts[:samples_per_lang_small]

    return lang_to_all, lang_to_small


def load_similarity_meta(path: str) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """
    读取已保存的 similarity 文件，返回:
      - half_similarity: dict[src][tgt] = value
      - saved_target_languages: 保存文件里记录的 target_languages（若无则返回空列表）
    """
    data = load_json(path, default={}) or {}
    half = data.get("half_similarity", {}) if isinstance(data, dict) else {}
    saved_langs = data.get("target_languages", []) if isinstance(data, dict) else []
    # half 可能是 str-keyed src; 保持原样即可
    return half, list(saved_langs) if isinstance(saved_langs, list) else []

def build_incremental_halfsim_plan(
    half_sim: Dict[str, Dict[str, float]],
    current_targets: List[str],
) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """
    给定已有 half_sim 与当前 target 列表，返回：
      - new_sources: 需要“整行重算”的 src（half_sim 里不存在的 src）
      - patch_old_sources: 需要“补齐缺失 target 维度”的 src -> missing_targets 集合
        （典型就是旧语言缺少新加入语言那几列）
    约定：只补缺失 key，不覆盖已有 key。
    """
    cur_set = set(current_targets)

    # 1) 需要整行重算的 src：当前目标语言里有，但 half_sim 里完全没有该 src
    new_sources = set([src for src in current_targets if src not in half_sim])

    # 2) 旧 src 需要补齐的 targets
    patch_old_sources: Dict[str, Set[str]] = {}
    for src, row in half_sim.items():
        if not isinstance(row, dict):
            continue
        missing = cur_set - set(row.keys())
        if missing:
            patch_old_sources[src] = missing

    # 只关心“当前语言集合里的 src”
    patch_old_sources = {src: miss for src, miss in patch_old_sources.items() if src in cur_set}

    return new_sources, patch_old_sources

def merge_half_sim_row_inplace(
    half_sim: Dict[str, Dict[str, float]],
    src: str,
    new_values: Dict[str, float],
    only_if_missing: bool = True,
):
    """
    将 new_values 合并进 half_sim[src]。
    - only_if_missing=True：只填不存在的 key（安全，避免覆盖旧-旧结果）
    """
    if src not in half_sim or not isinstance(half_sim.get(src), dict):
        half_sim[src] = {}
    row = half_sim[src]
    for k, v in new_values.items():
        if (not only_if_missing) or (k not in row):
            row[k] = float(v)

def ensure_half_sim_row_has_all_targets(
    half_sim: Dict[str, Dict[str, float]],
    src: str,
    current_targets: List[str],
    fill_value: float = 0.0,
):
    """
    可选：为了让下游处理更一致，确保 half_sim[src] 对所有 target 都有 key。
    不过你现在 save_similarity_progress 里 get(...,0) 已经兼容缺失 key，
    所以这个不是必须的。
    """
    if src not in half_sim or not isinstance(half_sim.get(src), dict):
        half_sim[src] = {}
    row = half_sim[src]
    for t in current_targets:
        if t not in row:
            row[t] = float(fill_value)
