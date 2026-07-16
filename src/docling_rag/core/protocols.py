# core/protocols.py
from typing import Protocol
import numpy as np
from docling_rag.core.chunker import Chunk


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
    ) -> None:
        """Add or update document entry. Preserves added_at on re-index."""
        ...

    def delete(self, source_file: str) -> None:
        """Remove document entry."""
        ...

    def get(self, source_file: str) -> dict | None:
        """Return entry for source_file or None."""
        ...

    def load(self) -> dict[str, dict]:
        """Return full index as {source_file: {title, topic, tags, added_at}}."""
        ...
