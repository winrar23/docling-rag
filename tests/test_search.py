import numpy as np
import pytest
from unittest.mock import MagicMock

from docling_rag.core.search import run_search, resolve_allowed_sources


def test_run_search_returns_results():
    """run_search embeds query and calls storage.search."""
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = np.ones((1, 384), dtype=np.float32)

    expected_results = [
        ({"text": "chunk text", "source_file": "doc.pdf"}, 0.9),
    ]
    mock_storage = MagicMock()
    mock_storage.search.return_value = expected_results

    results = run_search("test query", mock_embedder, mock_storage, top_k=5)

    assert results == expected_results
    mock_embedder.embed.assert_called_once_with(["test query"])
    mock_storage.search.assert_called_once()
    call_kwargs = mock_storage.search.call_args
    assert call_kwargs.kwargs["top_k"] == 5
    assert call_kwargs.kwargs["allowed_sources"] is None


def test_run_search_passes_allowed_sources():
    """run_search forwards allowed_sources to storage.search."""
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = np.ones((1, 384), dtype=np.float32)
    mock_storage = MagicMock()
    mock_storage.search.return_value = []

    sources = {"doc1.pdf", "doc2.pdf"}
    run_search("query", mock_embedder, mock_storage, top_k=3, allowed_sources=sources)

    call_kwargs = mock_storage.search.call_args
    assert call_kwargs.kwargs["allowed_sources"] == sources


def test_run_search_propagates_file_not_found():
    """run_search propagates FileNotFoundError from docling_rag.storage.search."""
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = np.ones((1, 384), dtype=np.float32)
    mock_storage = MagicMock()
    mock_storage.search.side_effect = FileNotFoundError("Storage not found")

    with pytest.raises(FileNotFoundError):
        run_search("query", mock_embedder, mock_storage, top_k=5)


def _registry_with(docs):
    reg = MagicMock(); reg.load.return_value = docs; return reg


DOCS = {
    "a.pdf": {"title": "A", "topic": "Arch", "tags": ["arch", "solid"], "added_at": "x"},
    "b.pdf": {"title": "B", "topic": "data", "tags": ["etl"], "added_at": "x"},
}


def test_no_filters_returns_none():
    assert resolve_allowed_sources(_registry_with(DOCS)) is None


def test_tag_filter_and_semantics():
    assert resolve_allowed_sources(_registry_with(DOCS), tags=("arch", "solid")) == {"a.pdf"}


def test_topic_case_insensitive():
    assert resolve_allowed_sources(_registry_with(DOCS), topic="arch") == {"a.pdf"}


def test_no_match_returns_empty_set():
    assert resolve_allowed_sources(_registry_with(DOCS), tags=("nope",)) == set()
