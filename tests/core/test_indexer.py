import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from docling_rag.core.errors import StorageSchemaMissingError, StorageUnavailableError
from docling_rag.core.indexer import index_files, IndexReport
from docling_rag.storage.file_storage import FileStorage
from docling_rag.storage.doc_registry import DocRegistry
from docling_rag.core.chunker import Chunk


def _chunk(source: str) -> Chunk:
    return Chunk(text="t", source_file=source, chunk_id=0, page_number=1,
                 element_type="text", headings=[], context_text="ctx")


def test_index_files_happy_path(tmp_path):
    f = tmp_path / "a.md"; f.write_text("# A\n\ntext")
    storage, registry = FileStorage(data_dir=tmp_path), DocRegistry(data_dir=tmp_path)
    parser, embedder = MagicMock(), MagicMock()
    embedder.embed.return_value = np.ones((1, 384), dtype=np.float32)
    with patch("docling_rag.core.indexer.chunk_document", return_value=[_chunk(str(f.resolve()))]):
        report = index_files([f], parser, embedder, storage, registry, "all-MiniLM-L6-v2",
                             title="T", topic="x", tags=("a",))
    assert report == IndexReport(chunks_added=1, files_ok=1, files_failed=0, errors=[])
    _, meta = storage.load()
    assert len(meta) == 1 and meta[0]["source_file"] == str(f.resolve())
    assert registry.get(str(f.resolve()))["title"] == "T"


def test_index_files_reindex_is_idempotent(tmp_path):
    f = tmp_path / "a.md"; f.write_text("# A\n\ntext")
    storage, registry = FileStorage(data_dir=tmp_path), DocRegistry(data_dir=tmp_path)
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
    storage, registry = FileStorage(data_dir=tmp_path), DocRegistry(data_dir=tmp_path)
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
    storage, registry = FileStorage(data_dir=tmp_path), DocRegistry(data_dir=tmp_path)
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
