# core/storage.py
from typing import Protocol
import numpy as np
from core.chunker import Chunk


class StorageBackend(Protocol):
    """
    Protocol for storage backends. Implementations:
    - storage.file_storage.FileStorage (MVP)
    - storage.db_storage.DBStorage (Phase 2, pgvector)
    """

    def save(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Save chunks and embeddings (overwrites)."""
        ...

    def append(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Append new chunks to existing storage."""
        ...

    def load(self) -> tuple[np.ndarray, list[dict]]:
        """Load all embeddings and metadata. Raises FileNotFoundError if empty."""
        ...

    def delete_by_source(self, source_file: str) -> None:
        """Delete all chunks from the given source file."""
        ...

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[tuple[dict, float]]:
        """Find top_k nearest chunks by cosine similarity."""
        ...


class DocumentRegistryBackend(Protocol):
    """
    Protocol for document-level metadata registries. Implementations:
    - storage.doc_registry.DocRegistry (MVP)
    - storage.db_registry.DBRegistry (Phase 2, PostgreSQL)
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
