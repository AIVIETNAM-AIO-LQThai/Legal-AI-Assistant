from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pandas as pd

from .text import clean_legal_text, first_non_empty, infer_doc_type, normalize_doc_code, normalize_whitespace


TEXT_KEYS = [
    "text", "content", "full_text", "noi_dung", "noi_dung_van_ban", "article_text",
    "raw_text", "html_text", "plain_text", "body",
]
DOC_CODE_KEYS = [
    "doc_code", "code", "ma_van_ban", "so_ky_hieu", "sohieu", "so_hieu", "document_code",
    "law_id", "id_van_ban", "mã văn bản", "so_van_ban",
]
TITLE_KEYS = [
    "doc_title", "title", "ten_van_ban", "trich_yeu", "summary", "description",
    "official_title", "name", "tên văn bản",
]
DOC_TYPE_KEYS = ["doc_type", "loai_van_ban", "type", "document_type", "loại văn bản"]
ARTICLE_NUMBER_KEYS = ["article_number", "article", "dieu", "so_dieu", "article_id", "điều"]
ARTICLE_TITLE_KEYS = ["article_title", "ten_dieu", "title_article", "tên điều"]
CLAUSE_KEYS = ["clause", "clause_text", "khoan", "noi_dung_khoan"]
SOURCE_URL_KEYS = ["source_url", "url", "link", "href"]
ISSUING_BODY_KEYS = ["issuing_body", "co_quan_ban_hanh", "agency"]
ISSUED_DATE_KEYS = ["issued_date", "ngay_ban_hanh", "ban_hanh", "date_issued"]
EFFECTIVE_DATE_KEYS = ["effective_date", "ngay_hieu_luc", "hieu_luc", "date_effective"]


def flatten_json_records(obj: Any) -> Iterator[Dict[str, Any]]:
    """Yield dictionaries from a JSON object that may be list/dict/nested."""
    if isinstance(obj, list):
        for item in obj:
            yield from flatten_json_records(item)
    elif isinstance(obj, dict):
        # Common dataset wrappers.
        for key in ["data", "items", "records", "rows", "articles", "documents"]:
            if key in obj and isinstance(obj[key], list):
                for item in obj[key]:
                    yield from flatten_json_records(item)
                return
        yield obj
    else:
        return


def read_text_file(path: Path) -> str:
    for enc in ["utf-8", "utf-8-sig", "cp1258", "latin1"]:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore")


def load_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install pypdf to parse PDF files.") from exc
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return clean_legal_text("\n\n".join(pages))


def load_docx_text(path: Path) -> str:
    try:
        import docx
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install python-docx to parse DOCX files.") from exc
    document = docx.Document(str(path))
    return clean_legal_text("\n".join(p.text for p in document.paragraphs))


def load_local_file(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        text = read_text_file(path)
        obj = json.loads(text)
        return [dict(r, raw_path=str(path)) for r in flatten_json_records(obj)]
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(dict(json.loads(line), raw_path=str(path)))
        return rows
    if suffix == ".csv":
        df = pd.read_csv(path)
        return [dict(r, raw_path=str(path)) for r in df.to_dict(orient="records")]
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
        return [dict(r, raw_path=str(path)) for r in df.to_dict(orient="records")]
    if suffix == ".txt":
        return [{"title": path.stem, "text": read_text_file(path), "raw_path": str(path)}]
    if suffix == ".pdf":
        return [{"title": path.stem, "text": load_pdf_text(path), "raw_path": str(path)}]
    if suffix == ".docx":
        return [{"title": path.stem, "text": load_docx_text(path), "raw_path": str(path)}]
    return []


def load_local_rows(input_path: str | Path) -> List[Dict[str, Any]]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    files = [path] if path.is_file() else sorted(
        p for p in path.rglob("*") if p.suffix.lower() in {".json", ".jsonl", ".csv", ".parquet", ".pq", ".txt", ".pdf", ".docx"}
    )
    rows: List[Dict[str, Any]] = []
    for file_path in files:
        rows.extend(load_local_file(file_path))
    return rows


def load_hf_rows(dataset_name: str, split: str = "train", config_name: Optional[str] = None, streaming: bool = False) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install datasets to load Hugging Face datasets.") from exc
    if config_name:
        ds = load_dataset(dataset_name, config_name, split=split, streaming=streaming)
    else:
        ds = load_dataset(dataset_name, split=split, streaming=streaming)
    return [dict(r, source_dataset=dataset_name) for r in ds]


def infer_raw_record(row: Dict[str, Any], source_name: str = "") -> Dict[str, str]:
    """Infer canonical raw fields from a heterogeneous row."""
    doc_code = normalize_doc_code(first_non_empty(row, DOC_CODE_KEYS))
    raw_title = first_non_empty(row, TITLE_KEYS)
    doc_type = first_non_empty(row, DOC_TYPE_KEYS)
    if not doc_type:
        doc_type = infer_doc_type(raw_title, doc_code)

    article_number = first_non_empty(row, ARTICLE_NUMBER_KEYS)
    article_title = first_non_empty(row, ARTICLE_TITLE_KEYS)
    text = first_non_empty(row, TEXT_KEYS)
    clause_text = first_non_empty(row, CLAUSE_KEYS)
    if clause_text and not text:
        text = clause_text

    # If a row has an article field with a full phrase like "Điều 5. ...", use it as text/title fallback.
    if not text:
        for k, v in row.items():
            if isinstance(v, str) and re.search(r"Điều\s+\d+", v):
                text = v
                break

    return {
        "doc_code": doc_code,
        "raw_title": normalize_whitespace(raw_title),
        "doc_type": normalize_whitespace(doc_type),
        "article_number": normalize_whitespace(article_number),
        "article_title": normalize_whitespace(article_title),
        "text": clean_legal_text(text),
        "source_url": first_non_empty(row, SOURCE_URL_KEYS),
        "issuing_body": first_non_empty(row, ISSUING_BODY_KEYS),
        "issued_date": first_non_empty(row, ISSUED_DATE_KEYS),
        "effective_date": first_non_empty(row, EFFECTIVE_DATE_KEYS),
        "source_dataset": normalize_whitespace(row.get("source_dataset", source_name) or source_name),
        "raw_path": normalize_whitespace(row.get("raw_path", "")),
    }
