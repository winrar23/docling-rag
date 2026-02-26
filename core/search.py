from __future__ import annotations

import numpy as np

from core.embedder import Embedder
from storage.file_storage import FileStorage


def run_search(
    query: str,
    embedder: Embedder,
    storage: FileStorage,
    top_k: int,
    allowed_sources: set[str] | None = None,
) -> list[tuple[dict, float]]:
    """Embed query and search storage. Used by CLI search and agent tool."""
    query_emb: np.ndarray = embedder.embed([query])[0]
    return storage.search(
        query_embedding=query_emb,
        top_k=top_k,
        allowed_sources=allowed_sources,
    )
