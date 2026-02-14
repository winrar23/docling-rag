import numpy as np
import pytest
from core.embedder import Embedder


def test_embedder_returns_numpy_array():
    embedder = Embedder()
    result = embedder.embed(["Hello world"])
    assert isinstance(result, np.ndarray)


def test_embedder_output_shape():
    embedder = Embedder()
    texts = ["Hello world", "Semantic search", "SQL query"]
    result = embedder.embed(texts)
    assert result.shape == (3, 384)  # all-MiniLM-L6-v2 → 384 dimensions


def test_embedder_single_text():
    embedder = Embedder()
    result = embedder.embed(["Just one sentence"])
    assert result.shape == (1, 384)


def test_embedder_normalized_vectors():
    """Vectors must be L2-normalized (cosine similarity = dot product)."""
    embedder = Embedder()
    result = embedder.embed(["Normalized vector test"])
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


@pytest.mark.slow
def test_similar_texts_have_high_similarity():
    embedder = Embedder()
    vecs = embedder.embed(["database schema", "schema of database", "python syntax"])
    sim_same = float(np.dot(vecs[0], vecs[1]))
    sim_diff = float(np.dot(vecs[0], vecs[2]))
    assert sim_same > sim_diff, "Semantically close texts must have higher similarity"


def test_embedder_empty_list_returns_empty_array():
    embedder = Embedder()
    result = embedder.embed([])
    assert result.shape == (0, 384)
