import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from docling_rag.api.app import (  # noqa: E402
    app, get_registry, get_search_embedder, get_search_log, get_settings, get_storage,
)
from docling_rag.core.chunker import Chunk  # noqa: E402
from tests.fakes import InMemoryRegistry, InMemorySearchLog, InMemoryStorage  # noqa: E402


class FakeEmbedder:
    def embed(self, texts):
        return np.ones((len(texts), 4), dtype="float32")


@pytest.fixture
def client():
    registry, storage, log = InMemoryRegistry(), InMemoryStorage(), InMemorySearchLog()
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_storage] = lambda: storage
    app.dependency_overrides[get_search_embedder] = lambda: FakeEmbedder()
    app.dependency_overrides[get_search_log] = lambda: log
    app.dependency_overrides[get_settings] = lambda: {"top_k_results": 5}
    yield TestClient(app), registry, storage, log
    app.dependency_overrides.clear()


def _seed(registry, storage, source="/uploads/b.pdf", title="B", tags=()):
    # InMemoryStorage.append принимает объекты Chunk, не dict'ы (см. tests/api/test_catalog.py::_seed_doc).
    registry.upsert(source, title, "sys", list(tags))
    storage.append(
        [Chunk(text="replication text", source_file=source, chunk_id=0,
               page_number=3, element_type="text", headings=["Replication"])],
        np.ones((1, 4), dtype="float32"),
    )


def test_search_returns_results_with_title(client):
    c, registry, storage, log = client
    _seed(registry, storage)
    body = c.get("/search", params={"q": "replication"}).json()
    assert body["query"] == "replication"
    r = body["results"][0]
    assert r["text"] == "replication text" and r["title"] == "B"
    assert r["page_number"] == 3 and r["element_type"] == "text"
    assert isinstance(r["score"], float)


def test_search_logs_top_score(client):
    c, registry, storage, log = client
    _seed(registry, storage)
    c.get("/search", params={"q": "x"})
    assert len(log.entries) == 1 and log.entries[0][0] == "x"


def test_search_log_failure_does_not_break_request(client):
    c, registry, storage, log = client
    _seed(registry, storage)

    def boom(query, top_score):
        raise RuntimeError("лог лежит")
    log.log = boom
    assert c.get("/search", params={"q": "x"}).status_code == 200


def test_search_empty_storage_empty_results(client):
    c, *_ = client
    body = c.get("/search", params={"q": "x"}).json()
    assert body["results"] == []


def test_search_filter_no_match_empty_results(client):
    c, registry, storage, _ = client
    _seed(registry, storage, tags=["arch"])
    body = c.get("/search", params={"q": "x", "tag": "no-such-tag"}).json()
    assert body["results"] == []


def test_search_validation(client):
    c, *_ = client
    assert c.get("/search").status_code == 422              # q обязателен
    assert c.get("/search", params={"q": ""}).status_code == 422
    assert c.get("/search", params={"q": "x", "top_k": 0}).status_code == 422
    assert c.get("/search", params={"q": "x", "top_k": 51}).status_code == 422
