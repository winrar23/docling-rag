from __future__ import annotations

from typing import Sequence

import numpy as np

from docling_rag.core.embedder import Embedder
from docling_rag.core.protocols import DocumentRegistryBackend, StorageBackend


def run_search(
    query: str,
    embedder: Embedder,
    storage: StorageBackend,
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


def resolve_allowed_sources(
    registry: DocumentRegistryBackend,
    tags: Sequence[str] = (),
    topic: str | None = None,
) -> set[str] | None:
    """None — no filters; empty set — filters given but nothing matched (=> empty results)."""
    if not tags and not topic:
        return None
    matched: set[str] = set()
    for src, entry in registry.load().items():
        tag_ok = all(t in entry.get("tags", []) for t in tags)
        topic_ok = (entry.get("topic") or "").lower() == topic.lower() if topic else True
        if tag_ok and topic_ok:
            matched.add(src)
    return matched
