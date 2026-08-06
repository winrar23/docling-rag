"""Indexing service: file -> parse -> chunk -> embed -> store. Used by CLI and API worker (v2 stage 4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from docling_rag.core.chunker import chunk_document
from docling_rag.core.embedder import Embedder
from docling_rag.core.errors import (
    EmbedServiceUnavailableError,
    StorageSchemaMissingError,
    StorageUnavailableError,
)
from docling_rag.core.parser import Parser
from docling_rag.core.protocols import DocumentRegistryBackend, StorageBackend

# Шаги пайплайна — значения совпадают с колонкой jobs.step.
PARSING = "parsing"
CHUNKING = "chunking"
METADATA = "metadata"
EMBEDDING = "embedding"
STORING = "storing"

ProgressCallback = Callable[[str, "int | None", "int | None"], None]

_EMBED_BATCH = 128


@dataclass
class IndexReport:
    chunks_added: int = 0
    files_ok: int = 0
    files_failed: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[tuple[str, str]] = field(default_factory=list)


def index_files(
    files: Iterable[Path],
    parser: Parser,
    embedder: Embedder,
    storage: StorageBackend,
    registry: DocumentRegistryBackend,
    embedding_model: str,
    chunk_max_tokens: int = 512,
    ocr: str = "auto",
    ocr_lang: str = "en",
    metadata_extractor: "Callable[[Sequence], object] | None" = None,
    on_progress: ProgressCallback | None = None,
) -> IndexReport:
    """
    metadata_extractor: опциональный callable(chunks) -> DocMeta
    (docling_rag.core.metadata.DocMeta; не импортируется здесь верхнеуровнево, чтобы
    core/indexer.py не тянул за собой pydantic_ai/agent-extras). Извлечение — fail-soft:
    исключения (кроме инфраструктурных Storage-ошибок) превращаются в report.warnings,
    индексация файла продолжается с заглушкой метаданных.
    """
    def report_progress(step: str, done: int | None = None, total: int | None = None) -> None:
        if on_progress is not None:
            on_progress(step, done, total)

    report = IndexReport()
    for file in files:
        source = str(file)
        try:
            source = str(Path(file).resolve())
            report_progress(PARSING)
            doc = parser.parse(file, ocr=ocr, ocr_lang=ocr_lang)
            report_progress(CHUNKING)
            chunks = chunk_document(
                doc, source_file=source, embedding_model=embedding_model, max_tokens=chunk_max_tokens
            )
            if not chunks:
                report.files_ok += 1
                continue

            meta = None
            if metadata_extractor is not None:
                report_progress(METADATA)
                try:
                    meta = metadata_extractor(chunks)
                except (StorageUnavailableError, StorageSchemaMissingError):
                    raise  # инфраструктура БД — не «мягкая» ошибка метаданных
                except Exception as e:  # fail-soft: LLM/сеть/extras не роняют индексацию
                    report.warnings.append(
                        (source, f"метаданные не извлечены: {type(e).__name__}: {e}")
                    )

            total = len(chunks)
            parts = []
            for start in range(0, total, _EMBED_BATCH):
                batch = [c.context_text for c in chunks[start:start + _EMBED_BATCH]]
                parts.append(embedder.embed(batch, batch_size=_EMBED_BATCH))
                report_progress(EMBEDDING, min(start + _EMBED_BATCH, total), total)
            embeddings = np.vstack(parts)
            report_progress(STORING)
            storage.delete_by_source(source)
            storage.append(chunks, embeddings)
            title = (meta.title if meta else None) or Path(source).stem
            registry.upsert(
                source,
                title=title,
                topic=meta.topic if meta else None,
                tags=list(meta.tags) if meta else [],
                author=meta.author if meta else None,
            )
            report.chunks_added += len(chunks)
            report.files_ok += 1
        except (StorageUnavailableError, StorageSchemaMissingError, EmbedServiceUnavailableError):
            raise  # инфраструктурная ошибка — батч бессмысленен, пробрасываем наверх
        except Exception as e:  # batch loop: one file's failure must not abort the rest
            report.files_failed += 1
            report.errors.append((source, f"{type(e).__name__}: {e}"))
    return report
