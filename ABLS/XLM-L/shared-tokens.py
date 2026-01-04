import os
import json
import unicodedata
from collections import Counter, defaultdict

from transformers import AutoTokenizer

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x


# ==========================
# 配置区
# ==========================
MODEL_NAME = "xlm-roberta-large"
MC4_LOCAL_DIR = "/ACL_exp/data/MC4"

OUTPUT_DIR = "/ACL_exp/Models_LPD/XLM-L"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SHARED_TOKENS_JSON = os.path.join(
    OUTPUT_DIR,
    "xlmr_large_shared_tokens_v2_45.json"
)

# ==========================
# 语言映射（保持不变）
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
    # "tam_Taml": "Tamil",
    "ben_Beng": "Bengali",
    # "heb_Hebr": "Hebrew",
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
    # "hye_Armn": "Armenian",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",

    # ===== 新增语言（补充） =====
    "afr_Latn": "Afrikaans",
    "arb_Arab": "Arabic",
    "azj_Latn": "Azerbaijani",
    "eus_Latn": "Basque",
    "est_Latn": "Estonian",
    # "kat_Geor": "Georgian",
    "hin_Deva": "Hindi",
    "jav_Latn": "Javanese",
    "lvs_Latn": "Latvian",
    "khk_Cyrl": "Mongolian",
    "pes_Arab": "Persian",
    "swe_Latn": "Swedish",
    # "tel_Telu": "Telugu",
    "tha_Thai": "Thai",
    "uzn_Latn": "Uzbek",
    "yor_Latn": "Yoruba",
}

# ==========================
# 只用 Latn + Cyrl
# ==========================
USE_ONLY_LATN_CYRL = True
ALLOWED_SCRIPTS = ("_Latn", "_Cyrl")


def read_local_jsonl_texts(path):
    """读取本地 MC4 的 jsonl 文件：每行 {"text": "..."}，只取 text"""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", None)
            if text is None:
                continue
            text = str(text).strip()
            if not text:
                continue
            texts.append(text)
    return texts


def is_unicode_letter_only(s: str) -> bool:
    """字符串每个字符都必须是 Unicode Letter（category 以 'L' 开头）"""
    if not s:
        return False
    for ch in s:
        if not unicodedata.category(ch).startswith("L"):
            return False
    return True


def detect_script_latn_or_cyrl(s: str):
    """
    判断字符串脚本（基于真实字符）：
    - 若全部字符都是 LATIN 字母 -> "Latn"
    - 若全部字符都是 CYRILLIC 字母 -> "Cyrl"
    - 否则（混合/非LatnCyrl）-> None
    """
    if not s:
        return None

    script = None
    for ch in s:
        name = unicodedata.name(ch, "")
        if "LATIN" in name:
            ch_script = "Latn"
        elif "CYRILLIC" in name:
            ch_script = "Cyrl"
        else:
            return None

        if script is None:
            script = ch_script
        elif script != ch_script:
            return None

    return script


def lang_script_of_code(lang_code: str):
    """根据 lang_code 后缀得到语言脚本（仅 Latn/Cyrl）"""
    if lang_code.endswith("_Latn"):
        return "Latn"
    if lang_code.endswith("_Cyrl"):
        return "Cyrl"
    return None


def token_surface_via_decode(tokenizer, tid: int) -> str:
    """
    用 decode 得到 token 的“真实表面形式”用于脚本判断。
    注意：decode 单 token 可能带前导空格；这里 strip 掉两端空白。
    """
    s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
    if s is None:
        return ""
    return s.strip()


def main():
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    special_ids = set(tokenizer.all_special_ids)

    # -------------------------------
    # 选择语言：只取 Latn + Cyrl，且必须在映射表中
    # -------------------------------
    all_lang_codes = list(LANG_CODE_TO_LID_LABEL.keys())
    if USE_ONLY_LATN_CYRL:
        expected_langs = [c for c in all_lang_codes if c.endswith(ALLOWED_SCRIPTS)]
    else:
        expected_langs = all_lang_codes

    # 检查本地文件存在性
    active_langs = []
    for lang_code in expected_langs:
        fp = os.path.join(MC4_LOCAL_DIR, f"{lang_code}.jsonl")
        if os.path.exists(fp):
            active_langs.append(lang_code)

    print(f"\nLocal MC4 dir: {MC4_LOCAL_DIR}")
    print(f"Detected languages in local MC4 within scope (Latn+Cyrl) (N={len(active_langs)}):")
    for c in active_langs:
        print(f"  - {c}  ->  LID label: {LANG_CODE_TO_LID_LABEL.get(c)}")

    missing = set(expected_langs) - set(active_langs)
    if missing:
        print("\n[Warning] Expected but NOT found in local MC4 dir:")
        for m in sorted(missing):
            print("  -", m)

    if not active_langs:
        print("\n[Error] No languages found in local MC4 dir for (Latn+Cyrl). Please check files.")
        return

    # -------------------------------
    # 逐语言 tokenize & count（只统计 TOTAL，不统计句首）
    # -------------------------------
    lang_total_counts = {lang: Counter() for lang in active_langs}

    print("\nCounting tokens (filter: Unicode-L only + token script must be Latn/Cyrl + must match language script)...")

    # cache: 避免重复 decode/判断
    token_ok_cache = {}       # tid -> bool
    token_script_cache = {}   # tid -> "Latn"/"Cyrl"/None

    for lang_code in active_langs:
        lid_label = LANG_CODE_TO_LID_LABEL.get(lang_code, "UNKNOWN_IN_LID")
        lang_script = lang_script_of_code(lang_code)  # "Latn" or "Cyrl"
        path = os.path.join(MC4_LOCAL_DIR, f"{lang_code}.jsonl")
        texts = read_local_jsonl_texts(path)

        print(f"\nProcessing: {lang_code} (LID={lid_label}, script={lang_script})  sentences={len(texts)}  file={path}")

        for text in tqdm(texts, desc=f"  Tokenizing {lang_code}"):
            encoded = tokenizer(text, add_special_tokens=False)
            ids = encoded.get("input_ids", [])
            if not ids:
                continue

            for tid in ids:
                tid = int(tid)
                if tid in special_ids:
                    continue

                ok = token_ok_cache.get(tid)
                if ok is None:
                    surf = token_surface_via_decode(tokenizer, tid)
                    if not is_unicode_letter_only(surf):
                        token_ok_cache[tid] = False
                        token_script_cache[tid] = None
                    else:
                        scr = detect_script_latn_or_cyrl(surf)
                        token_script_cache[tid] = scr
                        token_ok_cache[tid] = (scr in ("Latn", "Cyrl"))
                    ok = token_ok_cache[tid]

                if not ok:
                    continue

                # token 脚本必须与语言脚本匹配（避免 Latn token 计入 Cyrl 语言）
                tok_script = token_script_cache.get(tid)
                if tok_script != lang_script:
                    continue

                lang_total_counts[lang_code][tid] += 1

        print(f"  Done. Unique kept tokens = {len(lang_total_counts[lang_code])}")

    # -------------------------------
    # 合并到 token -> {lid_label: count}（仅 TOTAL）
    # -------------------------------
    print("\nMerging (and mapping codes to LID labels)...")
    token_to_lang_total = defaultdict(dict)

    for lang_code in active_langs:
        lid_label = LANG_CODE_TO_LID_LABEL.get(lang_code, lang_code)
        for tid, cnt in lang_total_counts[lang_code].items():
            token_to_lang_total[int(tid)][lid_label] = int(cnt)

    # -------------------------------
    # shared token 判定：total >=2 langs
    # 输出：一律输出 total_cnt（与位置无关）
    # -------------------------------
    print("\nFiltering shared tokens (sharedness decided by TOTAL; weights = TOTAL only, position-independent)...")

    shared_token_table = {}

    for tid, total_lang_counts in token_to_lang_total.items():
        if len(total_lang_counts) < 2:
            continue
        shared_token_table[int(tid)] = {
            "langs": {lid: int(cnt) for lid, cnt in total_lang_counts.items()}
        }

    print(f"Total kept tokens seen: {len(token_to_lang_total)}")
    print(f"Shared tokens (>=2 langs by TOTAL): {len(shared_token_table)}")

    # -------------------------------
    # 保存
    # -------------------------------
    print(f"\nSaving to {SHARED_TOKENS_JSON}")
    with open(SHARED_TOKENS_JSON, "w", encoding="utf-8") as f:
        json.dump(shared_token_table, f, ensure_ascii=False, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
