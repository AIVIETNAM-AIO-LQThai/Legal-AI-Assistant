from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class LegalDocument:
    doc_id: str
    doc_code: str
    doc_type: str
    doc_title: str
    official_title: str
    issuing_body: str = ""
    issued_date: str = ""
    effective_date: str = ""
    source_url: str = ""
    source_dataset: str = ""
    raw_path: str = ""


@dataclass(frozen=True)
class LegalArticle:
    article_id: str
    doc_id: str
    doc_code: str
    doc_title: str
    article_number: str
    article_title: str
    article_text: str
    chapter_title: str = ""
    section_title: str = ""


@dataclass(frozen=True)
class LegalChunk:
    chunk_id: str
    article_id: str
    doc_id: str
    doc_code: str
    doc_title: str
    article_number: str
    article_title: str
    clause_number: str
    point_number: str
    chunk_type: str
    chunk_text: str
    retrieval_text: str
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
