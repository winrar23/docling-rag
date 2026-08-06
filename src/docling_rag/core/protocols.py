# core/protocols.py
from typing import Protocol, Sequence
import numpy as np
from docling_rag.core.chunker import Chunk


class EmbedderBackend(Protocol):
    """Эмбеддер запросов/чанков: локальная модель или HTTP-клиент embed-сервиса."""

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class StorageBackend(Protocol):
    """
    Protocol for storage backends. Implementations:
    - docling_rag.storage.db_storage.DBStorage (PostgreSQL + pgvector)
    - tests.fakes.InMemoryStorage (unit tests)
    """

    def append(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Append new chunks to existing storage."""
        ...

    def load(self) -> tuple[np.ndarray, list[dict]]:
        """Load all embeddings and metadata. Raises FileNotFoundError if empty."""
        ...

    def delete_by_source(self, source_file: str) -> None:
        """Delete all chunks from the given source file."""
        ...

    def count_by_source(self, source_file: str) -> int:
        """Return number of chunks stored for the given source file."""
        ...

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        allowed_sources: set[str] | None = None,
    ) -> list[tuple[dict, float]]:
        """Find top_k nearest chunks by cosine similarity. allowed_sources filters by source_file."""
        ...


class SearchLogBackend(Protocol):
    """
    Protocol for search-query logs. Implementations:
    - docling_rag.storage.db_search_log.DBSearchLog (PostgreSQL)
    - tests.fakes.InMemorySearchLog (unit tests)
    """

    def log(self, query: str, top_score: float) -> None:
        """Record a search query and the score of its best hit."""
        ...


class DocumentRegistryBackend(Protocol):
    """
    Protocol for document-level metadata registries. Implementations:
    - docling_rag.storage.db_registry.DBRegistry (PostgreSQL)
    - tests.fakes.InMemoryRegistry (unit tests)
    """

    def upsert(
        self,
        source_file: str,
        title: str | None,
        topic: str | None,
        tags: list[str],
        author: str | None = None,
    ) -> None:
        """Add or update document entry. Preserves added_at; None не затирает существующие."""
        ...

    def delete(self, source_file: str) -> None:
        """Remove document entry."""
        ...

    def get(self, source_file: str) -> dict | None:
        """Return entry for source_file or None."""
        ...

    def load(self) -> dict[str, dict]:
        """Return full index as {source_file: {id, title, topic, tags, added_at}}."""
        ...

    def get_by_id(self, doc_id: str) -> tuple[str, dict] | None:
        """Return (source_file, entry) for a surrogate uuid, or None (malformed/unknown)."""
        ...

    def update_metadata(self, source_file: str, fields: dict) -> dict | None:
        """Явно установить переданные поля из {title, author, topic, tags} (None очищает).

        Возвращает обновлённый entry или None, если документа нет.
        """
        ...


class JobBackend(Protocol):
    """
    Protocol for background ingestion jobs. Implementations:
    - docling_rag.storage.db_jobs.DBJobs (PostgreSQL, очередь через FOR UPDATE SKIP LOCKED)
    - tests.fakes.InMemoryJobs (unit tests)

    Job-dict: id, source_file, original_name, title, topic, tags, status
    (queued|running|done|failed), step, chunks_done, chunks_total, error,
    attempts, created_at, started_at, updated_at, finished_at, ocr, ocr_lang.
    """

    def create(self, source_file: str, original_name: str,
               title: str | None, topic: str | None, tags: list[str],
               ocr: str = "auto", ocr_lang: str = "en") -> str:
        """Insert a queued job. Return job_id."""
        ...

    def get(self, job_id: str) -> dict | None:
        """Return job-dict or None."""
        ...

    def list(self, limit: int = 20, status: str | None = None) -> list[dict]:
        """Recent jobs, newest first."""
        ...

    def find_active_by_source(self, source_file: str) -> dict | None:
        """Return a queued/running job for this source, else None (dedup guard)."""
        ...

    def find_latest_by_source(self, source_file: str) -> dict | None:
        """Return the newest job (any status) for this source, else None (catalog card)."""
        ...

    def claim_next(self) -> dict | None:
        """Atomically move one queued job to running; return it or None."""
        ...

    def update_progress(self, job_id: str, step: str,
                        chunks_done: int | None = None,
                        chunks_total: int | None = None) -> None:
        """Set step/counters and bump heartbeat."""
        ...

    def heartbeat(self, job_id: str) -> None:
        """Bump updated_at (liveness), no other change."""
        ...

    def complete(self, job_id: str, chunks_added: int) -> None:
        """Mark done."""
        ...

    def fail(self, job_id: str, error: str) -> None:
        """Mark failed with error text."""
        ...

    def requeue_stale(self, stale_seconds: int, max_attempts: int) -> int:
        """Running jobs with stale heartbeat → queued (or failed if attempts exhausted). Return count."""
        ...
