from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable, Optional

DOC_TYPE_PATTERNS = [
    "Luật",
    "Bộ luật",
    "Nghị định",
    "Thông tư",
    "Thông tư liên tịch",
    "Quyết định",
    "Nghị quyết",
    "Pháp lệnh",
]

ABBREVIATIONS = {
    "bhxh": "bảo hiểm xã hội",
    "bhtn": "bảo hiểm thất nghiệp",
    "bhyt": "bảo hiểm y tế",
    "hđlđ": "hợp đồng lao động",
    "hddt": "hóa đơn điện tử",
    "tnhh": "trách nhiệm hữu hạn",
    "ctcp": "công ty cổ phần",
    "dn": "doanh nghiệp",
    "dnnvv": "doanh nghiệp nhỏ và vừa",
    "sme": "doanh nghiệp nhỏ và vừa",
    "sở hữu trí tuệ": "sở hữu trí tuệ",
}


def stable_id(*parts: object, length: int = 20) -> str:
    """Create a stable short SHA1 id from arbitrary parts."""
    raw = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]


def normalize_unicode(text: object) -> str:
    if text is None:
        return ""
    return unicodedata.normalize("NFC", str(text))


def normalize_whitespace(text: object) -> str:
    text = normalize_unicode(text)
    text = text.replace("\ufeff", " ")
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"[ \u00a0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_legal_text(text: object) -> str:
    text = normalize_whitespace(text)
    # Remove common page headers/footers while staying conservative.
    text = re.sub(r"(?im)^\s*Trang\s+\d+\s*/\s*\d+\s*$", "", text)
    text = re.sub(r"(?im)^\s*Page\s+\d+\s*(of\s+\d+)?\s*$", "", text)
    return normalize_whitespace(text)


def first_non_empty(row: dict, keys: Iterable[str], default: str = "") -> str:
    lower_map = {str(k).lower().strip(): k for k in row.keys()}
    for key in keys:
        actual = lower_map.get(key.lower().strip())
        if actual is not None:
            val = row.get(actual)
            if val is not None and str(val).strip() and str(val).strip().lower() != "nan":
                return normalize_whitespace(val)
    return default


def infer_doc_type(title: str, code: str = "") -> str:
    sample = f"{title} {code}".strip()
    for pat in DOC_TYPE_PATTERNS:
        if re.search(rf"\b{re.escape(pat)}\b", sample, flags=re.IGNORECASE):
            return pat
    if re.search(r"/NĐ-CP\b", sample, re.IGNORECASE):
        return "Nghị định"
    if re.search(r"/TT-BTC\b|/TT-[A-ZĐ]+\b", sample, re.IGNORECASE):
        return "Thông tư"
    if re.search(r"/QH\d+\b", sample, re.IGNORECASE):
        return "Luật"
    return "Văn bản"


def normalize_doc_code(code: str) -> str:
    code = normalize_whitespace(code)
    code = code.replace(" ", "")
    code = code.replace("ND-CP", "NĐ-CP")
    code = code.replace("NĐCP", "NĐ-CP")
    return code


def compose_contest_doc_title(doc_type: str, doc_code: str, title_or_summary: str) -> str:
    """Return title in contest format: Loại văn bản + Mã văn bản + Trích yếu."""
    doc_type = normalize_whitespace(doc_type)
    doc_code = normalize_doc_code(doc_code)
    title_or_summary = normalize_whitespace(title_or_summary)

    # If already starts with the type and code, preserve it.
    if doc_code and re.search(re.escape(doc_code), title_or_summary, flags=re.IGNORECASE):
        if doc_type and title_or_summary.lower().startswith(doc_type.lower()):
            return title_or_summary

    # Remove duplicated leading type/code from summary.
    summary = title_or_summary
    if doc_type:
        summary = re.sub(rf"^\s*{re.escape(doc_type)}\s+", "", summary, flags=re.IGNORECASE)
    if doc_code:
        summary = re.sub(rf"^\s*{re.escape(doc_code)}\s+", "", summary, flags=re.IGNORECASE)

    parts = [p for p in [doc_type, doc_code, summary] if p]
    return normalize_whitespace(" ".join(parts))


def normalize_article_number(article: str) -> str:
    article = normalize_whitespace(article)
    m = re.search(r"Điều\s+([0-9]+[a-zA-ZÀ-ỹ]?)", article, flags=re.IGNORECASE)
    if m:
        return f"Điều {m.group(1)}"
    if re.fullmatch(r"[0-9]+[a-zA-ZÀ-ỹ]?", article):
        return f"Điều {article}"
    return article


def expand_query_terms(query: str) -> str:
    """Lightweight query expansion for legal Vietnamese abbreviations."""
    q = normalize_whitespace(query).lower()
    extra = []
    for abbr, full in ABBREVIATIONS.items():
        if abbr in q and full not in q:
            extra.append(full)
    return normalize_whitespace(query + " " + " ".join(extra))
