"""In-memory реализации Protocol'ов для герметичных юнит-тестов (без postgres)."""
from datetime import datetime

import numpy as np

from docling_rag.core.chunker import Chunk
from docling_rag.core.errors import StorageError  # noqa: F401  (паритет исключений с реальными бэкендами)


def _chunk_to_meta(chunk: Chunk) -> dict:
    return {
        "text": chunk.text, "source_file": chunk.source_file, "chunk_id": chunk.chunk_id,
        "page_number": chunk.page_number, "element_type": chunk.element_type,
        "headings": chunk.headings,
    }


class InMemoryStorage:
    """Семантика DBStorage/FileStorage: пустое хранилище -> FileNotFoundError."""

    def __init__(self) -> None:
        self._emb: np.ndarray | None = None
        self._meta: list[dict] = []

    def save(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks/embeddings length mismatch")
        self._emb = embeddings.copy()
        self._meta = [_chunk_to_meta(c) for c in chunks]

    def append(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks/embeddings length mismatch")
        if self._emb is None:
            self.save(chunks, embeddings)
            return
        self._emb = np.vstack([self._emb, embeddings])
        self._meta = self._meta + [_chunk_to_meta(c) for c in chunks]

    def load(self) -> tuple[np.ndarray, list[dict]]:
        if self._emb is None or len(self._meta) == 0:
            raise FileNotFoundError("Storage is empty")
        return self._emb, self._meta

    def delete_by_source(self, source_file: str) -> None:
        if self._emb is None:
            return
        keep = [i for i, m in enumerate(self._meta) if m["source_file"] != source_file]
        if not keep:
            self._emb, self._meta = None, []
            return
        self._emb = self._emb[keep]
        self._meta = [self._meta[i] for i in keep]

    def count_by_source(self, source_file: str) -> int:
        return sum(1 for m in self._meta if m["source_file"] == source_file)

    def search(self, query_embedding, top_k=5, allowed_sources=None):
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if allowed_sources is not None and not allowed_sources:
            return []
        emb, meta = self.load()
        if allowed_sources is not None:
            mask = [i for i, m in enumerate(meta) if m["source_file"] in allowed_sources]
            if not mask:
                return []
            emb, meta = emb[mask], [meta[i] for i in mask]
        scores = emb @ query_embedding
        top = np.argsort(scores)[::-1][:top_k]
        return [(meta[i], float(scores[i])) for i in top]


class InMemoryRegistry:
    def __init__(self) -> None:
        self._docs: dict[str, dict] = {}

    def upsert(self, source_file, title, topic, tags) -> None:
        existing = self._docs.get(source_file, {})
        self._docs[source_file] = {
            "title": title if title is not None else existing.get("title"),
            "topic": topic if topic is not None else existing.get("topic"),
            "tags": list(tags) if tags else existing.get("tags", []),
            "added_at": existing.get("added_at", datetime.now().isoformat(timespec="seconds")),
        }

    def delete(self, source_file: str) -> None:
        self._docs.pop(source_file, None)

    def get(self, source_file: str) -> dict | None:
        return self._docs.get(source_file)

    def load(self) -> dict[str, dict]:
        return dict(self._docs)
