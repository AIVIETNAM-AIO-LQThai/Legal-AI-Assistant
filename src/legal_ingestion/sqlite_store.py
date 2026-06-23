from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List, Sequence

from .schemas import LegalArticle, LegalChunk, LegalDocument

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS legal_documents (
    doc_id TEXT PRIMARY KEY,
    doc_code TEXT NOT NULL,
    doc_type TEXT,
    doc_title TEXT NOT NULL,
    official_title TEXT,
    issuing_body TEXT,
    issued_date TEXT,
    effective_date TEXT,
    source_url TEXT,
    source_dataset TEXT,
    raw_path TEXT
);

CREATE TABLE IF NOT EXISTS legal_articles (
    article_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    doc_code TEXT NOT NULL,
    doc_title TEXT NOT NULL,
    article_number TEXT,
    article_title TEXT,
    article_text TEXT NOT NULL,
    chapter_title TEXT,
    section_title TEXT,
    FOREIGN KEY(doc_id) REFERENCES legal_documents(doc_id)
);

CREATE TABLE IF NOT EXISTS legal_chunks (
    chunk_id TEXT PRIMARY KEY,
    article_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    doc_code TEXT NOT NULL,
    doc_title TEXT NOT NULL,
    article_number TEXT,
    article_title TEXT,
    clause_number TEXT,
    point_number TEXT,
    chunk_type TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    retrieval_text TEXT NOT NULL,
    char_start INTEGER,
    char_end INTEGER,
    metadata_json TEXT,
    FOREIGN KEY(article_id) REFERENCES legal_articles(article_id),
    FOREIGN KEY(doc_id) REFERENCES legal_documents(doc_id)
);

CREATE INDEX IF NOT EXISTS idx_docs_code ON legal_documents(doc_code);
CREATE INDEX IF NOT EXISTS idx_articles_doc ON legal_articles(doc_id);
CREATE INDEX IF NOT EXISTS idx_articles_number ON legal_articles(article_number);
CREATE INDEX IF NOT EXISTS idx_chunks_article ON legal_chunks(article_id);
CREATE INDEX IF NOT EXISTS idx_chunks_type ON legal_chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_code_article ON legal_chunks(doc_code, article_number);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection, drop_existing: bool = False) -> None:
    if drop_existing:
        conn.executescript(
            """
            DROP TABLE IF EXISTS legal_chunks;
            DROP TABLE IF EXISTS legal_articles;
            DROP TABLE IF EXISTS legal_documents;
            """
        )
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def insert_documents(conn: sqlite3.Connection, docs: Sequence[LegalDocument]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO legal_documents (
            doc_id, doc_code, doc_type, doc_title, official_title,
            issuing_body, issued_date, effective_date, source_url,
            source_dataset, raw_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                d.doc_id, d.doc_code, d.doc_type, d.doc_title, d.official_title,
                d.issuing_body, d.issued_date, d.effective_date, d.source_url,
                d.source_dataset, d.raw_path,
            )
            for d in docs
        ],
    )
    conn.commit()


def insert_articles(conn: sqlite3.Connection, articles: Sequence[LegalArticle]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO legal_articles (
            article_id, doc_id, doc_code, doc_title, article_number,
            article_title, article_text, chapter_title, section_title
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                a.article_id, a.doc_id, a.doc_code, a.doc_title, a.article_number,
                a.article_title, a.article_text, a.chapter_title, a.section_title,
            )
            for a in articles
        ],
    )
    conn.commit()


def insert_chunks(conn: sqlite3.Connection, chunks: Sequence[LegalChunk]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO legal_chunks (
            chunk_id, article_id, doc_id, doc_code, doc_title, article_number,
            article_title, clause_number, point_number, chunk_type, chunk_text,
            retrieval_text, char_start, char_end, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                c.chunk_id, c.article_id, c.doc_id, c.doc_code, c.doc_title, c.article_number,
                c.article_title, c.clause_number, c.point_number, c.chunk_type, c.chunk_text,
                c.retrieval_text, c.char_start, c.char_end, json.dumps(c.metadata or {}, ensure_ascii=False),
            )
            for c in chunks
        ],
    )
    conn.commit()


def save_all(
    db_path: str | Path,
    docs: Sequence[LegalDocument],
    articles: Sequence[LegalArticle],
    chunks: Sequence[LegalChunk],
    drop_existing: bool = True,
) -> None:
    conn = connect(db_path)
    try:
        init_db(conn, drop_existing=drop_existing)
        insert_documents(conn, docs)
        insert_articles(conn, articles)
        insert_chunks(conn, chunks)
    finally:
        conn.close()


def count_table(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])


def fetch_chunks_for_chroma(conn: sqlite3.Connection, include_chunk_types: List[str] | None = None):
    if include_chunk_types:
        placeholders = ",".join("?" for _ in include_chunk_types)
        sql = f"""
        SELECT * FROM legal_chunks
        WHERE chunk_type IN ({placeholders})
        ORDER BY doc_code, article_number, chunk_type
        """
        return conn.execute(sql, include_chunk_types)
    return conn.execute("SELECT * FROM legal_chunks ORDER BY doc_code, article_number, chunk_type")


def fetch_sample(conn: sqlite3.Connection, limit: int = 5):
    return conn.execute(
        """
        SELECT c.chunk_id, c.doc_code, c.doc_title, c.article_number, c.article_title,
               c.clause_number, c.point_number, c.chunk_type, substr(c.chunk_text, 1, 500) AS chunk_preview
        FROM legal_chunks c
        ORDER BY c.doc_code, c.article_number, c.chunk_type
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
