from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Iterable, List, Sequence

from tqdm import tqdm

from .sqlite_store import connect, fetch_chunks_for_chroma


def batched(items: Sequence, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def load_embedding_model(model_name_or_path: str, device: str | None = None):
    from sentence_transformers import SentenceTransformer

    kwargs = {"trust_remote_code": True}
    if device:
        kwargs["device"] = device
    return SentenceTransformer(model_name_or_path, **kwargs)


def build_chroma_from_sqlite(
    db_path: str | Path,
    persist_dir: str | Path,
    embedding_model: str,
    collection_name: str = "legal_chunks",
    batch_size: int = 64,
    device: str | None = None,
    reset_collection: bool = True,
    include_chunk_types: List[str] | None = None,
) -> None:
    import chromadb

    conn = connect(db_path)
    try:
        rows = list(fetch_chunks_for_chroma(conn, include_chunk_types=include_chunk_types))
    finally:
        conn.close()

    if not rows:
        raise RuntimeError("No chunks found in SQLite. Build the SQLite DB first.")

    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    if reset_collection:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    model = load_embedding_model(embedding_model, device=device)

    for batch in tqdm(list(batched(rows, batch_size)), desc="Indexing chunks into ChromaDB"):
        ids = [r["chunk_id"] for r in batch]
        docs = [r["retrieval_text"] for r in batch]
        embeddings = model.encode(
            docs,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        metadatas = [
            {
                "article_id": r["article_id"],
                "doc_id": r["doc_id"],
                "doc_code": r["doc_code"],
                "doc_title": r["doc_title"],
                "article_number": r["article_number"] or "",
                "article_title": r["article_title"] or "",
                "clause_number": r["clause_number"] or "",
                "point_number": r["point_number"] or "",
                "chunk_type": r["chunk_type"],
            }
            for r in batch
        ]
        collection.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)

    print(f"Done. Chroma collection '{collection_name}' contains {collection.count()} chunks.")


def query_chroma(
    persist_dir: str | Path,
    embedding_model: str,
    query: str,
    collection_name: str = "legal_chunks",
    top_k: int = 10,
    device: str | None = None,
):
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(collection_name)
    model = load_embedding_model(embedding_model, device=device)
    q_emb = model.encode([query], normalize_embeddings=True).tolist()
    return collection.query(query_embeddings=q_emb, n_results=top_k, include=["documents", "metadatas", "distances"])
