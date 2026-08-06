import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from docling_rag.core.errors import (
    EmbedServiceUnavailableError,
    StorageSchemaMissingError,
    StorageUnavailableError,
)
from docling_rag.core.indexer import index_files, IndexReport
from docling_rag.core.chunker import Chunk
from tests.fakes import InMemoryRegistry, InMemoryStorage


def _chunk(source: str) -> Chunk:
    return Chunk(text="t", source_file=source, chunk_id=0, page_number=1,
                 element_type="text", headings=[], context_text="ctx")


def test_index_files_happy_path(tmp_path):
    f = tmp_path / "a.md"; f.write_text("# A\n\ntext")
    storage, registry = InMemoryStorage(), InMemoryRegistry()
    parser, embedder = MagicMock(), MagicMock()
    embedder.embed.return_value = np.ones((1, 384), dtype=np.float32)
    with patch("docling_rag.core.indexer.chunk_document", return_value=[_chunk(str(f.resolve()))]):
        report = index_files([f], parser, embedder, storage, registry, "all-MiniLM-L6-v2")
    assert report == IndexReport(chunks_added=1, files_ok=1, files_failed=0, errors=[], warnings=[])
    _, meta = storage.load()
    assert len(meta) == 1 and meta[0]["source_file"] == str(f.resolve())
    assert registry.get(str(f.resolve()))["title"] == f.stem


def test_index_files_reindex_is_idempotent(tmp_path):
    f = tmp_path / "a.md"; f.write_text("# A\n\ntext")
    storage, registry = InMemoryStorage(), InMemoryRegistry()
    parser, embedder = MagicMock(), MagicMock()
    embedder.embed.return_value = np.ones((1, 384), dtype=np.float32)
    with patch("docling_rag.core.indexer.chunk_document", return_value=[_chunk(str(f.resolve()))]):
        index_files([f], parser, embedder, storage, registry, "m")
        index_files([f], parser, embedder, storage, registry, "m")
    _, meta = storage.load()
    assert len(meta) == 1  # не 2


def test_index_files_passes_chunk_max_tokens_to_chunk_document(tmp_path):
    """chunk_max_tokens must reach chunk_document as max_tokens (non-default 384:
    matching defaults of 512 on both sides would mask a dropped kwarg)."""
    f = tmp_path / "a.md"; f.write_text("# A\n\ntext")
    storage, registry = InMemoryStorage(), InMemoryRegistry()
    parser, embedder = MagicMock(), MagicMock()
    embedder.embed.return_value = np.ones((1, 384), dtype=np.float32)
    with patch("docling_rag.core.indexer.chunk_document",
               return_value=[_chunk(str(f.resolve()))]) as mock_chunk_doc:
        index_files([f], parser, embedder, storage, registry, "all-MiniLM-L6-v2",
                    chunk_max_tokens=384)
    mock_chunk_doc.assert_called_once_with(
        parser.parse.return_value,
        source_file=str(f.resolve()),
        embedding_model="all-MiniLM-L6-v2",
        max_tokens=384,
    )


def test_index_files_continues_after_failure(tmp_path):
    good, bad = tmp_path / "g.md", tmp_path / "b.md"
    good.write_text("g"); bad.write_text("b")
    storage, registry = InMemoryStorage(), InMemoryRegistry()
    parser, embedder = MagicMock(), MagicMock()
    parser.parse.side_effect = [ValueError("boom"), MagicMock()]
    embedder.embed.return_value = np.ones((1, 384), dtype=np.float32)
    with patch("docling_rag.core.indexer.chunk_document", return_value=[_chunk(str(good.resolve()))]):
        report = index_files([bad, good], parser, embedder, storage, registry, "m")
    assert report.files_failed == 1 and report.files_ok == 1
    assert report.errors[0][0] == str(bad.resolve()) and "boom" in report.errors[0][1]


@pytest.mark.parametrize("infra_error", [
    StorageUnavailableError("connection refused"),
    StorageSchemaMissingError('relation "chunks" does not exist'),
])
def test_index_files_reraises_infrastructure_errors(tmp_path, infra_error):
    """Инфраструктурная ошибка хранилища — не «кривой файл»: батч бессмысленен,
    исключение пробрасывается наверх, а не кладётся в report.errors."""
    f = tmp_path / "a.md"; f.write_text("# A\n\ntext")
    storage, registry = MagicMock(), MagicMock()
    storage.delete_by_source.side_effect = infra_error
    parser, embedder = MagicMock(), MagicMock()
    embedder.embed.return_value = np.ones((1, 384), dtype=np.float32)
    with patch("docling_rag.core.indexer.chunk_document", return_value=[_chunk(str(f.resolve()))]):
        with pytest.raises(type(infra_error)):
            index_files([f], parser, embedder, storage, registry, "m")


def test_index_files_reraises_embed_service_unavailable(tmp_path):
    """Отказ embed-сервиса (embed_url в HTTP-режиме) — тоже инфраструктурная ошибка:
    embed-сервис имеет `restart: unless-stopped` и может легитимно временно
    отвалиться. Она не должна тонуть в report.errors (терминальный per-file fail),
    а пробрасываться наверх, чтобы worker мог вернуть джобу через requeue_stale."""
    f = tmp_path / "a.md"; f.write_text("# A\n\ntext")
    storage, registry = MagicMock(), MagicMock()
    parser, embedder = MagicMock(), MagicMock()
    embedder.embed.side_effect = EmbedServiceUnavailableError("connection refused")
    with patch("docling_rag.core.indexer.chunk_document", return_value=[_chunk(str(f.resolve()))]):
        with pytest.raises(EmbedServiceUnavailableError):
            index_files([f], parser, embedder, storage, registry, "m")
    storage.append.assert_not_called()


def test_index_files_reports_progress_steps(monkeypatch):
    """on_progress получает шаги по порядку + батчи embed с done/total."""
    from pathlib import Path
    from unittest.mock import MagicMock
    import numpy as np

    import docling_rag.core.indexer as indexer_mod
    from docling_rag.core.chunker import Chunk
    from tests.fakes import InMemoryStorage, InMemoryRegistry

    fake_chunks = [
        Chunk(text=f"t{i}", source_file="/x.pdf", chunk_id=i, page_number=1,
              element_type="text", headings=[], context_text=f"t{i}")
        for i in range(5)
    ]
    monkeypatch.setattr(indexer_mod, "chunk_document", lambda *a, **k: fake_chunks)

    parser = MagicMock()
    embedder = MagicMock()
    embedder.embed.side_effect = lambda texts, batch_size=128: np.ones((len(texts), 1024), dtype=np.float32)

    events = []
    indexer_mod.index_files(
        [Path("/x.pdf")], parser, embedder, InMemoryStorage(), InMemoryRegistry(),
        embedding_model="m", chunk_max_tokens=512,
        on_progress=lambda step, done, total: events.append((step, done, total)),
    )

    steps = [e[0] for e in events]
    assert steps[0] == indexer_mod.PARSING
    assert indexer_mod.CHUNKING in steps
    assert indexer_mod.STORING == steps[-1]
    embed_events = [e for e in events if e[0] == indexer_mod.EMBEDDING]
    assert embed_events, "должны быть события embedding"
    assert embed_events[-1] == (indexer_mod.EMBEDDING, 5, 5)  # последний батч: done==total==5


def test_index_files_passes_ocr_to_parser(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    parser = MagicMock()
    embedder = MagicMock()
    embedder.embed.return_value = np.zeros((1, 4), dtype=np.float32)
    storage, registry = InMemoryStorage(), InMemoryRegistry()
    with patch("docling_rag.core.indexer.chunk_document", return_value=[]):
        index_files([f], parser, embedder, storage, registry, "model",
                    ocr="off", ocr_lang="ru")
    parser.parse.assert_called_once_with(f, ocr="off", ocr_lang="ru")


def test_index_files_metadata_extractor_fills_registry(tmp_path):
    from docling_rag.core.metadata import DocMeta

    f = tmp_path / "a.md"; f.write_text("# A\n\ntext")
    storage, registry = InMemoryStorage(), InMemoryRegistry()
    parser, embedder = MagicMock(), MagicMock()
    embedder.embed.return_value = np.ones((1, 384), dtype=np.float32)
    calls = []

    def extractor(chunks):
        calls.append(list(chunks))
        return DocMeta(title="Книга", author="Автор А.", topic="базы данных",
                       tags=["postgres"])

    with patch("docling_rag.core.indexer.chunk_document", return_value=[_chunk(str(f.resolve()))]):
        report = index_files([f], parser, embedder, storage, registry, "m",
                             metadata_extractor=extractor)
    entry = registry.get(str(f.resolve()))
    assert entry["title"] == "Книга"
    assert entry["author"] == "Автор А."
    assert entry["topic"] == "базы данных"
    assert entry["tags"] == ["postgres"]
    assert calls, "extractor должен получить чанки"
    assert report.warnings == []


def test_index_files_extractor_failure_is_soft(tmp_path):
    f = tmp_path / "a.md"; f.write_text("# A\n\ntext")
    storage, registry = InMemoryStorage(), InMemoryRegistry()
    parser, embedder = MagicMock(), MagicMock()
    embedder.embed.return_value = np.ones((1, 384), dtype=np.float32)

    def extractor(chunks):
        raise ConnectionError("LM Studio лежит")

    with patch("docling_rag.core.indexer.chunk_document", return_value=[_chunk(str(f.resolve()))]):
        report = index_files([f], parser, embedder, storage, registry, "m",
                             metadata_extractor=extractor)
    assert report.files_ok == 1 and report.files_failed == 0  # индексация не пострадала
    entry = registry.get(str(f.resolve()))
    assert entry["title"] == f.stem       # заглушка — имя файла без расширения
    assert entry["author"] is None
    assert entry["tags"] == []
    assert len(report.warnings) == 1
    assert "метаданные не извлечены" in report.warnings[0][1]


def test_index_files_no_extractor_uses_stub_title(tmp_path):
    f = tmp_path / "a.md"; f.write_text("# A\n\ntext")
    storage, registry = InMemoryStorage(), InMemoryRegistry()
    parser, embedder = MagicMock(), MagicMock()
    embedder.embed.return_value = np.ones((1, 384), dtype=np.float32)
    with patch("docling_rag.core.indexer.chunk_document", return_value=[_chunk(str(f.resolve()))]):
        report = index_files([f], parser, embedder, storage, registry, "m",
                             metadata_extractor=None)
    assert registry.get(str(f.resolve()))["title"] == f.stem
    assert report.warnings == []


def test_index_files_reports_metadata_step(monkeypatch):
    """по образцу test_index_files_reports_progress_steps: собрать шаги в список"""
    from pathlib import Path

    import docling_rag.core.indexer as indexer_mod
    from docling_rag.core.chunker import Chunk
    from docling_rag.core.metadata import DocMeta

    fake_chunks = [
        Chunk(text="t0", source_file="/x.pdf", chunk_id=0, page_number=1,
              element_type="text", headings=[], context_text="t0"),
    ]
    monkeypatch.setattr(indexer_mod, "chunk_document", lambda *a, **k: fake_chunks)

    parser = MagicMock()
    embedder = MagicMock()
    embedder.embed.side_effect = lambda texts, batch_size=128: np.ones((len(texts), 4), dtype=np.float32)

    steps_with_extractor = []
    indexer_mod.index_files(
        [Path("/x.pdf")], parser, embedder, InMemoryStorage(), InMemoryRegistry(),
        embedding_model="m",
        metadata_extractor=lambda chunks: DocMeta(title="T"),
        on_progress=lambda step, done, total: steps_with_extractor.append(step),
    )

    steps_without_extractor = []
    indexer_mod.index_files(
        [Path("/x.pdf")], parser, embedder, InMemoryStorage(), InMemoryRegistry(),
        embedding_model="m",
        metadata_extractor=None,
        on_progress=lambda step, done, total: steps_without_extractor.append(step),
    )

    assert steps_with_extractor == ["parsing", "chunking", "metadata", "embedding", "storing"]
    assert steps_without_extractor == ["parsing", "chunking", "embedding", "storing"]
