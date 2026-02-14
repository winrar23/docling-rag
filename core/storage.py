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
