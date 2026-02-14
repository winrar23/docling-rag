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
        assert -1.0 <= score <= 1.0


def test_storage_search_sorted_by_score(storage):
    chunks = make_chunks(5)
    emb = make_embeddings(5)
    storage.save(chunks, emb)

    query = emb[2]  # exact match with third vector
    results = storage.search(query_embedding=query, top_k=3)

    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0][1] > 0.99  # first result is exact match


def test_storage_append_on_empty_storage(storage):
    """append() on first call (storage empty) should work without error."""
    chunks = make_chunks(2, source="first.pdf")
    emb = make_embeddings(2)
    storage.append(chunks, emb)  # no prior save — must not raise

    loaded_emb, loaded_meta = storage.load()
    assert loaded_emb.shape[0] == 2
    assert loaded_meta[0]["source_file"] == "first.pdf"


def test_storage_delete_all_chunks_empties_storage(storage):
    """delete_by_source when all chunks belong to that source unlinks files."""
    chunks = make_chunks(2, source="only.pdf")
    emb = make_embeddings(2)
    storage.save(chunks, emb)

    storage.delete_by_source("only.pdf")

    with pytest.raises(FileNotFoundError):
        storage.load()


def test_storage_delete_nonexistent_source_is_noop(storage):
    """delete_by_source for a source that doesn't exist — no error."""
    chunks = make_chunks(2)
    emb = make_embeddings(2)
    storage.save(chunks, emb)

    storage.delete_by_source("nonexistent.pdf")  # must not raise

    _, meta = storage.load()
    assert len(meta) == 2  # unchanged


def test_storage_delete_on_empty_storage_is_noop(storage):
    """delete_by_source on empty storage — no error."""
    storage.delete_by_source("anything.pdf")  # must not raise


def test_storage_search_top_k_larger_than_data(storage):
    """search() with top_k > N returns all N results, not an error."""
    chunks = make_chunks(2)
    emb = make_embeddings(2)
    storage.save(chunks, emb)

    query = make_embeddings(1)[0]
    results = storage.search(query_embedding=query, top_k=10)
    assert len(results) == 2  # only 2 available


def test_storage_save_length_mismatch_raises(storage):
    """save() with mismatched chunks/embeddings raises ValueError."""
    chunks = make_chunks(3)
    emb = make_embeddings(5)  # mismatch
    with pytest.raises(ValueError, match="mismatch"):
        storage.save(chunks, emb)


def test_search_with_allowed_sources_filters_results(storage):
    """search() only returns chunks from allowed_sources."""
    chunks_a = make_chunks(3, source="alpha.pdf")
    chunks_b = make_chunks(3, source="beta.pdf")
    all_chunks = chunks_a + chunks_b
    for i, c in enumerate(all_chunks):
        c.chunk_id = i
    emb = make_embeddings(6)
    storage.save(all_chunks, emb)

    query = make_embeddings(1)[0]
    results = storage.search(query_embedding=query, top_k=6, allowed_sources={"alpha.pdf"})

    assert len(results) <= 3
    assert all(meta["source_file"] == "alpha.pdf" for meta, _ in results)


def test_search_with_empty_allowed_sources_returns_empty(storage):
    """Empty allowed_sources set -> no results."""
    chunks = make_chunks(3)
    emb = make_embeddings(3)
    storage.save(chunks, emb)

    query = make_embeddings(1)[0]
    results = storage.search(query_embedding=query, top_k=3, allowed_sources=set())
    assert results == []


def test_search_with_none_allowed_sources_searches_all(storage):
    """allowed_sources=None (default) -> searches all docs, unchanged behaviour."""
    chunks_a = make_chunks(2, source="a.pdf")
    chunks_b = make_chunks(2, source="b.pdf")
    all_chunks = chunks_a + chunks_b
    for i, c in enumerate(all_chunks):
        c.chunk_id = i
    emb = make_embeddings(4)
    storage.save(all_chunks, emb)

    query = make_embeddings(1)[0]
    results = storage.search(query_embedding=query, top_k=4, allowed_sources=None)
    assert len(results) == 4
