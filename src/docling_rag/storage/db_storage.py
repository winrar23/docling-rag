# storage/db_storage.py
"""pgvector-хранилище chunks. Реализует StorageBackend (core/protocols.py)."""
import json

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from docling_rag.core.chunker import Chunk

_META_COLS = "text, source_file, chunk_id, page_number, element_type, headings"


def _to_numpy(v) -> np.ndarray:
    # Installed pgvector's psycopg loader returns a pgvector.Vector object
    # (not a bare list/ndarray) for the `embedding` column; convert explicitly.
    if hasattr(v, "to_numpy"):
        return v.to_numpy()
    return np.asarray(v, dtype=np.float32)


def _row_to_meta(row) -> dict:
    text, source_file, chunk_id, page_number, element_type, headings = row
    return {
        "text": text, "source_file": source_file, "chunk_id": chunk_id,
        "page_number": page_number, "element_type": element_type,
        "headings": headings,
    }


class DBStorage:
    """Соединение открывается на операцию: CLI короткоживущий, пул не нужен."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._dsn)
        register_vector(conn)
        return conn

    def _insert(self, conn: psycopg.Connection, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        # FK: indexer вызывает append ДО registry.upsert — гарантируем родительскую строку
        for src in {c.source_file for c in chunks}:
            conn.execute(
                "INSERT INTO documents (source_file) VALUES (%s) ON CONFLICT DO NOTHING",
                (src,),
            )
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (source_file, chunk_id, page_number, text, element_type, headings, embedding)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [
                    (c.source_file, c.chunk_id, c.page_number, c.text, c.element_type,
                     json.dumps(c.headings, ensure_ascii=False), embeddings[i])
                    for i, c in enumerate(chunks)
                ],
            )

    def _check_lengths(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings"
            )

    def save(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        self._check_lengths(chunks, embeddings)
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")
            self._insert(conn, chunks, embeddings)
            conn.commit()

    def append(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        self._check_lengths(chunks, embeddings)
        with self._connect() as conn:
            self._insert(conn, chunks, embeddings)
            conn.commit()

    def load(self) -> tuple[np.ndarray, list[dict]]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_META_COLS}, embedding FROM chunks ORDER BY source_file, chunk_id"
            ).fetchall()
        if not rows:
            raise FileNotFoundError("Storage is empty: no chunks in database")
        embeddings = np.stack([_to_numpy(r[-1]) for r in rows])
        return embeddings, [_row_to_meta(r[:-1]) for r in rows]

    def delete_by_source(self, source_file: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE source_file = %s", (source_file,))
            conn.commit()

    def count_by_source(self, source_file: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT count(*) FROM chunks WHERE source_file = %s", (source_file,)
            ).fetchone()
        return int(row[0])

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        allowed_sources: set[str] | None = None,
    ) -> list[tuple[dict, float]]:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if allowed_sources is not None and not allowed_sources:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        with self._connect() as conn:
            empty = conn.execute("SELECT NOT EXISTS (SELECT 1 FROM chunks)").fetchone()[0]
            if empty:
                raise FileNotFoundError("Storage is empty: no chunks in database")
            if allowed_sources is None:
                rows = conn.execute(
                    f"SELECT {_META_COLS}, 1 - (embedding <=> %s) AS score"
                    " FROM chunks ORDER BY embedding <=> %s LIMIT %s",
                    (q, q, top_k),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT {_META_COLS}, 1 - (embedding <=> %s) AS score"
                    " FROM chunks WHERE source_file = ANY(%s)"
                    " ORDER BY embedding <=> %s LIMIT %s",
                    (q, sorted(allowed_sources), q, top_k),
                ).fetchall()
        return [(_row_to_meta(r[:-1]), float(r[-1])) for r in rows]
