"""In-memory реализации Protocol'ов для герметичных юнит-тестов (без postgres)."""
from datetime import datetime, timezone

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
    """Семантика DBStorage: пустое хранилище -> FileNotFoundError."""

    def __init__(self) -> None:
        self._emb: np.ndarray | None = None
        self._meta: list[dict] = []

    def append(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks/embeddings length mismatch")
        self._emb = embeddings.copy() if self._emb is None else np.vstack([self._emb, embeddings])
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


class InMemorySearchLog:
    """Хранит записи в списке: юниты не открывают соединений к БД."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, float]] = []

    def log(self, query: str, top_score: float) -> None:
        self.entries.append((query, float(top_score)))


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


class InMemoryJobs:
    """Семантика DBJobs без postgres. _rows публичен для тестов (правка heartbeat)."""

    def __init__(self) -> None:
        self._rows: dict[str, dict] = {}
        self._seq = 0

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def create(self, source_file, original_name, title, topic, tags) -> str:
        self._seq += 1
        jid = str(self._seq)
        now = self._now()
        self._rows[jid] = {
            "id": jid, "source_file": source_file, "original_name": original_name,
            "title": title, "topic": topic, "tags": list(tags),
            "status": "queued", "step": None, "chunks_done": None, "chunks_total": None,
            "error": None, "attempts": 0,
            "created_at": now, "started_at": None, "updated_at": now, "finished_at": None,
        }
        return jid

    def get(self, job_id):
        row = self._rows.get(job_id)
        return dict(row) if row else None

    def list(self, limit=20, status=None):
        rows = [dict(r) for r in self._rows.values()
                if status is None or r["status"] == status]
        rows.sort(key=lambda r: r["created_at"], reverse=True)
        return rows[:limit]

    def find_active_by_source(self, source_file):
        for r in self._rows.values():
            if r["source_file"] == source_file and r["status"] in ("queued", "running"):
                return dict(r)
        return None

    def claim_next(self):
        for r in sorted(self._rows.values(), key=lambda r: r["created_at"]):
            if r["status"] == "queued":
                now = self._now()
                r.update(status="running", started_at=now, updated_at=now,
                         attempts=r["attempts"] + 1)
                return dict(r)
        return None

    def update_progress(self, job_id, step, chunks_done=None, chunks_total=None):
        r = self._rows[job_id]
        r.update(step=step, chunks_done=chunks_done, chunks_total=chunks_total,
                 updated_at=self._now())

    def heartbeat(self, job_id):
        self._rows[job_id]["updated_at"] = self._now()

    def complete(self, job_id, chunks_added):
        now = self._now()
        self._rows[job_id].update(status="done", step=None, chunks_done=chunks_added,
                                  updated_at=now, finished_at=now)

    def fail(self, job_id, error):
        now = self._now()
        self._rows[job_id].update(status="failed", error=error, updated_at=now, finished_at=now)

    def requeue_stale(self, stale_seconds, max_attempts):
        from datetime import timedelta
        cutoff = self._now() - timedelta(seconds=stale_seconds)
        n = 0
        for r in self._rows.values():
            if r["status"] == "running" and r["updated_at"] < cutoff:
                n += 1
                if r["attempts"] >= max_attempts:
                    r.update(status="failed", error="воркер умирал, превышен лимит попыток",
                             updated_at=self._now(), finished_at=self._now())
                else:
                    r.update(status="queued", step=None, updated_at=self._now())
        return n
