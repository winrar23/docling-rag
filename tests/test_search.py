import numpy as np
import pytest
from unittest.mock import MagicMock

from core.search import run_search


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
    """run_search propagates FileNotFoundError from storage.search."""
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = np.ones((1, 384), dtype=np.float32)
    mock_storage = MagicMock()
    mock_storage.search.side_effect = FileNotFoundError("Storage not found")

    with pytest.raises(FileNotFoundError):
        run_search("query", mock_embedder, mock_storage, top_k=5)
