import numpy as np
import pytest
from docling_rag.core.embedder import Embedder


def test_embedder_returns_numpy_array():
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    result = embedder.embed(["Hello world"])
    assert isinstance(result, np.ndarray)


def test_embedder_output_shape():
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    texts = ["Hello world", "Semantic search", "SQL query"]
    result = embedder.embed(texts)
    assert result.shape == (3, 384)  # all-MiniLM-L6-v2 → 384 dimensions


def test_embedder_single_text():
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    result = embedder.embed(["Just one sentence"])
    assert result.shape == (1, 384)


def test_embedder_normalized_vectors():
    """Vectors must be L2-normalized (cosine similarity = dot product)."""
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    result = embedder.embed(["Normalized vector test"])
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


@pytest.mark.slow
def test_similar_texts_have_high_similarity():
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    vecs = embedder.embed(["database schema", "schema of database", "python syntax"])
    sim_same = float(np.dot(vecs[0], vecs[1]))
    sim_diff = float(np.dot(vecs[0], vecs[2]))
    assert sim_same > sim_diff, "Semantically close texts must have higher similarity"


def test_embedder_empty_list_returns_empty_array():
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    result = embedder.embed([])
    assert result.shape == (0, 384)


def test_embed_passes_batch_size():
    from unittest.mock import MagicMock
    e = Embedder.__new__(Embedder)
    e._model = MagicMock()
    e._model.encode.return_value = np.ones((2, 4), dtype=np.float32)
    e._dim = 4
    e.embed(["a", "b"], batch_size=128)
    assert e._model.encode.call_args.kwargs["batch_size"] == 128


def test_embedder_init_does_not_use_deprecated_dimension_api():
    """sentence-transformers 5.6 переименовал get_sentence_embedding_dimension →
    get_embedding_dimension; старое имя даёт FutureWarning на каждом создании Embedder."""
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
    assert embedder._dim == 384
    deprecated = [w for w in caught if "get_sentence_embedding_dimension" in str(w.message)]
    assert not deprecated, f"deprecated API used: {deprecated[0].message}"
