from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

from .schemas import LegalArticle, LegalChunk, LegalDocument
from .text import (
    clean_legal_text,
    compose_contest_doc_title,
    normalize_article_number,
    normalize_doc_code,
    normalize_whitespace,
    stable_id,
)

ARTICLE_RE = re.compile(
    r"(?ms)(^\s*Điều\s+\d+[a-zA-ZÀ-ỹ]?\s*[\.:]?\s*.*?)(?=^\s*Điều\s+\d+[a-zA-ZÀ-ỹ]?\s*[\.:]?\s*|\Z)"
)
ARTICLE_HEAD_RE = re.compile(r"^\s*(Điều\s+\d+[a-zA-ZÀ-ỹ]?)\s*[\.:]?\s*(.*)$", re.IGNORECASE | re.MULTILINE)
CLAUSE_HEAD_RE = re.compile(r"(?m)^\s*(\d+)\s*[\.)]\s+")
POINT_HEAD_RE = re.compile(r"(?m)^\s*([a-zđ])\s*[\.)]\s+", re.IGNORECASE)


def split_article_header(article_block: str) -> Tuple[str, str, str]:
    """Return article_number, article_title, body."""
    article_block = clean_legal_text(article_block)
    lines = article_block.splitlines()
    if not lines:
        return "", "", ""
    first = lines[0]
    m = ARTICLE_HEAD_RE.match(first)
    if not m:
        # Sometimes OCR/text extraction puts header and title in the first paragraph.
        m2 = re.search(r"(Điều\s+\d+[a-zA-ZÀ-ỹ]?)\s*[\.:]?\s*([^\n]*)", article_block, flags=re.IGNORECASE)
        if m2:
            number = normalize_article_number(m2.group(1))
            title = normalize_whitespace(m2.group(2))
            body = normalize_whitespace(article_block[m2.end():])
            return number, title, body
        return "", "", article_block
    number = normalize_article_number(m.group(1))
    title = normalize_whitespace(m.group(2))
    body = normalize_whitespace("\n".join(lines[1:]))
    return number, title, body


def split_articles_from_text(text: str) -> List[Tuple[str, str, str]]:
    """Split a document text into legal articles.

    Returns tuples of: article_number, article_title, article_full_text.
    """
    text = clean_legal_text(text)
    blocks = [m.group(1).strip() for m in ARTICLE_RE.finditer(text)]
    if not blocks and text:
        return [("", "", text)]
    articles = []
    for block in blocks:
        num, title, body = split_article_header(block)
        full = normalize_whitespace("\n".join([f"{num}. {title}".strip(), body]).strip())
        articles.append((num, title, full))
    return articles


def split_by_headings(text: str, heading_re: re.Pattern[str]) -> List[Tuple[str, str, int, int]]:
    """Split text by legal headings.

    Returns tuples: heading_value, block_text, char_start, char_end.
    """
    text = normalize_whitespace(text)
    matches = list(heading_re.finditer(text))
    if not matches:
        return []
    out = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        label = m.group(1)
        block = normalize_whitespace(text[start:end])
        out.append((label, block, start, end))
    return out


def split_long_text(text: str, max_chars: int = 2200) -> List[str]:
    text = normalize_whitespace(text)
    if len(text) <= max_chars:
        return [text] if text else []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 2 <= max_chars:
            current = f"{current}\n\n{p}".strip()
        else:
            if current:
                chunks.append(current)
            if len(p) <= max_chars:
                current = p
            else:
                # Hard fallback for very long paragraphs.
                parts = [p[i:i + max_chars] for i in range(0, len(p), max_chars)]
                chunks.extend(parts[:-1])
                current = parts[-1]
    if current:
        chunks.append(current)
    return chunks


def make_retrieval_text(
    doc_title: str,
    article_number: str,
    article_title: str,
    chunk_text: str,
    clause_number: str = "",
    point_number: str = "",
) -> str:
    labels = [doc_title]
    article_line = " ".join(x for x in [article_number, article_title] if x)
    if article_line:
        labels.append(article_line)
    if clause_number:
        labels.append(clause_number)
    if point_number:
        labels.append(point_number)
    labels.append(chunk_text)
    return normalize_whitespace("\n".join(labels))


def build_document_from_raw(raw: Dict[str, str]) -> LegalDocument:
    doc_code = normalize_doc_code(raw.get("doc_code", ""))
    doc_type = normalize_whitespace(raw.get("doc_type", "")) or "Văn bản"
    raw_title = normalize_whitespace(raw.get("raw_title", ""))
    if not raw_title:
        raw_title = doc_code or stable_id(raw.get("text", ""), raw.get("raw_path", ""))
    doc_title = compose_contest_doc_title(doc_type, doc_code, raw_title)
    doc_id = stable_id(doc_code, doc_title, raw.get("source_url", ""), raw.get("raw_path", ""))
    return LegalDocument(
        doc_id=doc_id,
        doc_code=doc_code or doc_id,
        doc_type=doc_type,
        doc_title=doc_title,
        official_title=raw_title,
        issuing_body=raw.get("issuing_body", ""),
        issued_date=raw.get("issued_date", ""),
        effective_date=raw.get("effective_date", ""),
        source_url=raw.get("source_url", ""),
        source_dataset=raw.get("source_dataset", ""),
        raw_path=raw.get("raw_path", ""),
    )


def make_article(document: LegalDocument, article_number: str, article_title: str, article_text: str) -> LegalArticle:
    article_number = normalize_article_number(article_number)
    article_title = normalize_whitespace(article_title)
    article_text = clean_legal_text(article_text)
    if article_number and not article_text.lower().startswith(article_number.lower()):
        header = normalize_whitespace(" ".join([article_number, article_title]).strip())
        article_text = normalize_whitespace(f"{header}\n{article_text}")
    article_id = stable_id(document.doc_code, document.doc_title, article_number, article_title, article_text[:200])
    return LegalArticle(
        article_id=article_id,
        doc_id=document.doc_id,
        doc_code=document.doc_code,
        doc_title=document.doc_title,
        article_number=article_number,
        article_title=article_title,
        article_text=article_text,
    )


def chunk_article(
    article: LegalArticle,
    max_chars_per_chunk: int = 2200,
    include_title_chunks: bool = True,
    include_article_chunks: bool = True,
    include_clause_chunks: bool = True,
    include_point_chunks: bool = True,
) -> List[LegalChunk]:
    chunks: List[LegalChunk] = []

    def add_chunk(chunk_type: str, text: str, clause: str = "", point: str = "", start: int | None = None, end: int | None = None):
        text_clean = clean_legal_text(text)
        if not text_clean:
            return
        retrieval_text = make_retrieval_text(
            article.doc_title,
            article.article_number,
            article.article_title,
            text_clean,
            clause_number=clause,
            point_number=point,
        )
        chunk_id = stable_id(article.article_id, chunk_type, clause, point, text_clean[:250], start, end)
        chunks.append(LegalChunk(
            chunk_id=chunk_id,
            article_id=article.article_id,
            doc_id=article.doc_id,
            doc_code=article.doc_code,
            doc_title=article.doc_title,
            article_number=article.article_number,
            article_title=article.article_title,
            clause_number=clause,
            point_number=point,
            chunk_type=chunk_type,
            chunk_text=text_clean,
            retrieval_text=retrieval_text,
            char_start=start,
            char_end=end,
        ))

    if include_title_chunks:
        title_text = normalize_whitespace(" ".join([article.doc_title, article.article_number, article.article_title]))
        add_chunk("title", title_text)

    if include_article_chunks:
        for idx, part in enumerate(split_long_text(article.article_text, max_chars=max_chars_per_chunk)):
            add_chunk("article" if idx == 0 else "article_part", part)

    body = article.article_text
    # Remove first header line when possible so clause splitting works cleaner.
    if article.article_number:
        body = re.sub(rf"^\s*{re.escape(article.article_number)}\s*[\.:]?\s*.*?\n", "", body, count=1, flags=re.IGNORECASE)
    clause_blocks = split_by_headings(body, CLAUSE_HEAD_RE)

    if include_clause_chunks and clause_blocks:
        for clause_label, clause_text, c_start, c_end in clause_blocks:
            clause_number = f"Khoản {clause_label}"
            if include_point_chunks:
                point_blocks = split_by_headings(clause_text, POINT_HEAD_RE)
            else:
                point_blocks = []
            if point_blocks:
                # Add clause summary/full text too; useful for broad questions.
                add_chunk("clause", clause_text, clause=clause_number, start=c_start, end=c_end)
                for point_label, point_text, p_start, p_end in point_blocks:
                    add_chunk(
                        "point",
                        point_text,
                        clause=clause_number,
                        point=f"Điểm {point_label.lower()}",
                        start=c_start + p_start,
                        end=c_start + p_end,
                    )
            else:
                for idx, part in enumerate(split_long_text(clause_text, max_chars=max_chars_per_chunk)):
                    add_chunk("clause" if idx == 0 else "clause_part", part, clause=clause_number, start=c_start, end=c_end)

    # If no clauses and article text is long, paragraph chunks help focused retrieval.
    if not clause_blocks:
        for idx, part in enumerate(split_long_text(article.article_text, max_chars=max_chars_per_chunk)):
            if idx == 0 and include_article_chunks:
                continue
            add_chunk("paragraph", part)

    # Deduplicate exact retrieval texts.
    seen = set()
    deduped: List[LegalChunk] = []
    for ch in chunks:
        key = (ch.article_id, ch.chunk_type, ch.retrieval_text)
        if key not in seen:
            seen.add(key)
            deduped.append(ch)
    return deduped


def convert_raw_records_to_canonical(
    raw_records: Sequence[Dict[str, str]],
    max_chars_per_chunk: int = 2200,
) -> Tuple[List[LegalDocument], List[LegalArticle], List[LegalChunk]]:
    """Convert inferred raw records to canonical docs/articles/chunks.

    If a record already has article_number, it is treated as an article-level row.
    Otherwise, its full text is split into articles using Vietnamese legal headings.
    """
    documents_by_id: Dict[str, LegalDocument] = {}
    articles_by_id: Dict[str, LegalArticle] = {}
    chunks_by_id: Dict[str, LegalChunk] = {}

    for raw in raw_records:
        doc = build_document_from_raw(raw)
        documents_by_id[doc.doc_id] = doc
        text = clean_legal_text(raw.get("text", ""))
        if not text:
            continue

        article_number = normalize_article_number(raw.get("article_number", ""))
        article_title = normalize_whitespace(raw.get("article_title", ""))
        article_tuples: List[Tuple[str, str, str]]

        if article_number:
            article_tuples = [(article_number, article_title, text)]
        else:
            article_tuples = split_articles_from_text(text)

        for art_num, art_title, art_text in article_tuples:
            if not art_text:
                continue
            article = make_article(doc, art_num, art_title, art_text)
            articles_by_id[article.article_id] = article
            for chunk in chunk_article(article, max_chars_per_chunk=max_chars_per_chunk):
                chunks_by_id[chunk.chunk_id] = chunk

    return list(documents_by_id.values()), list(articles_by_id.values()), list(chunks_by_id.values())
