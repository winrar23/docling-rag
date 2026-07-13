"""Indexing service: file -> parse -> chunk -> embed -> store. Used by CLI (and API in v2 stage 4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from docling_rag.core.chunker import chunk_document
from docling_rag.core.embedder import Embedder
from docling_rag.core.parser import Parser
from docling_rag.core.protocols import DocumentRegistryBackend, StorageBackend


@dataclass
class IndexReport:
    chunks_added: int = 0
    files_ok: int = 0
    files_failed: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)


def index_files(
    files: Iterable[Path],
    parser: Parser,
    embedder: Embedder,
    storage: StorageBackend,
    registry: DocumentRegistryBackend,
    embedding_model: str,
    title: str | None = None,
    topic: str | None = None,
    tags: Sequence[str] = (),
) -> IndexReport:
    report = IndexReport()
    for file in files:
        source = str(file)
        try:
            source = str(Path(file).resolve())
            doc = parser.parse(file)
            chunks = chunk_document(doc, source_file=source, embedding_model=embedding_model)
            if not chunks:
                report.files_ok += 1
                continue
            embeddings = embedder.embed([c.context_text for c in chunks], batch_size=128)
            storage.delete_by_source(source)
            storage.append(chunks, embeddings)
            registry.upsert(source, title=title, topic=topic, tags=list(tags))
            report.chunks_added += len(chunks)
            report.files_ok += 1
        except Exception as e:  # batch loop: one file's failure must not abort the rest
            report.files_failed += 1
            report.errors.append((source, f"{type(e).__name__}: {e}"))
    return report
