import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.chunker import Chunk
from storage.file_storage import FileStorage


def make_chunks(n=3, source="doc.pdf"):
    return [
        Chunk(
            text=f"chunk text {i}",
            source_file=source,
            chunk_id=i,
            page_number=1,
            element_type="text",
        )
        for i in range(n)
    ]


def make_embeddings(n=3, dim=384):
    vecs = np.random.rand(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def storage(tmp_path):
    return FileStorage(data_dir=tmp_path)


def test_storage_saves_and_loads_embeddings(storage):
    chunks = make_chunks(3)
    embeddings = make_embeddings(3)
    storage.save(chunks, embeddings)

    loaded_emb, loaded_meta = storage.load()
    assert loaded_emb.shape == (3, 384)
    np.testing.assert_allclose(loaded_emb, embeddings, atol=1e-6)


def test_storage_saves_metadata(storage):
    chunks = make_chunks(2)
    embeddings = make_embeddings(2)
    storage.save(chunks, embeddings)

    _, loaded_meta = storage.load()
    assert len(loaded_meta) == 2
    assert loaded_meta[0]["text"] == "chunk text 0"
    assert loaded_meta[0]["source_file"] == "doc.pdf"
    assert loaded_meta[0]["chunk_id"] == 0
    assert loaded_meta[0]["page_number"] == 1
    assert loaded_meta[0]["element_type"] == "text"


def test_storage_load_raises_when_empty(storage):
    with pytest.raises(FileNotFoundError):
        storage.load()


def test_storage_creates_data_dir_if_missing(tmp_path):
    new_dir = tmp_path / "new" / "nested"
    storage = FileStorage(data_dir=new_dir)
    chunks = make_chunks(1)
    embeddings = make_embeddings(1)
    storage.save(chunks, embeddings)
    assert (new_dir / "embeddings.npy").exists()


def test_storage_appends_new_chunks(storage):
    chunks1 = make_chunks(2, source="doc1.pdf")
    emb1 = make_embeddings(2)
    storage.save(chunks1, emb1)

    chunks2 = make_chunks(3, source="doc2.pdf")
    emb2 = make_embeddings(3)
    storage.append(chunks2, emb2)

    loaded_emb, loaded_meta = storage.load()
    assert loaded_emb.shape[0] == 5
    assert loaded_meta[4]["source_file"] == "doc2.pdf"


def test_storage_delete_by_source(storage):
    chunks = make_chunks(2, source="old.pdf") + make_chunks(2, source="keep.pdf")
    for i, c in enumerate(chunks):
        c.chunk_id = i
    emb = make_embeddings(4)
    storage.save(chunks, emb)

    storage.delete_by_source("old.pdf")

    loaded_emb, loaded_meta = storage.load()
    assert loaded_emb.shape[0] == 2
    assert all(m["source_file"] == "keep.pdf" for m in loaded_meta)


def test_storage_search_returns_top_k(storage):
    chunks = make_chunks(10)
    emb = make_embeddings(10)
    storage.save(chunks, emb)

    query = make_embeddings(1)[0]
    results = storage.search(query_embedding=query, top_k=3)

    assert len(results) == 3
    for meta, score in results:
        assert "text" in meta
        assert 0.0 <= score <= 1.0


def test_storage_search_sorted_by_score(storage):
    chunks = make_chunks(5)
    emb = make_embeddings(5)
    storage.save(chunks, emb)

    query = emb[2]  # exact match with third vector
    results = storage.search(query_embedding=query, top_k=3)

    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0][1] > 0.99  # first result is exact match
