import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from docling_rag.api.app import app, get_jobs, get_registry, get_storage  # noqa: E402
from docling_rag.core.chunker import Chunk  # noqa: E402
from tests.fakes import InMemoryJobs, InMemoryRegistry, InMemoryStorage  # noqa: E402


@pytest.fixture
def client():
    jobs, registry, storage = InMemoryJobs(), InMemoryRegistry(), InMemoryStorage()
    app.dependency_overrides[get_jobs] = lambda: jobs
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_storage] = lambda: storage
    yield TestClient(app), jobs, registry, storage
    app.dependency_overrides.clear()


def _seed_doc(registry, storage, source="/uploads/b.pdf", title="B", chunks=2):
    registry.upsert(source, title, "sys", ["arch"], author="Автор А.")
    storage.append(
        [
            Chunk(text=f"c{i}", source_file=source, chunk_id=i,
                  page_number=1, element_type="text", headings=[])
            for i in range(chunks)
        ],
        np.zeros((chunks, 4), dtype="float32"),
    )


def test_list_documents_card_shape(client):
    c, jobs, registry, storage = client
    _seed_doc(registry, storage)
    jid = jobs.create("/uploads/b.pdf", "b.pdf")
    jobs.claim_next(); jobs.complete(jid, 2)

    body = c.get("/documents").json()
    assert len(body) == 1
    card = body[0]
    assert card["id"] and card["source_file"] == "/uploads/b.pdf"
    assert card["title"] == "B" and card["tags"] == ["arch"]
    assert card["author"] == "Автор А."
    assert card["chunks"] == 2
    assert card["indexing"] == {"status": "done", "job_id": jid}


def test_list_documents_without_jobs_indexing_null(client):
    c, _, registry, storage = client
    _seed_doc(registry, storage)
    assert c.get("/documents").json()[0]["indexing"] is None


def test_list_documents_sorted_added_at_desc(client):
    c, _, registry, storage = client
    _seed_doc(registry, storage, "/uploads/a.pdf", "A")
    _seed_doc(registry, storage, "/uploads/b.pdf", "B")
    registry._docs["/uploads/a.pdf"]["added_at"] = "2026-01-01T00:00:00"
    registry._docs["/uploads/b.pdf"]["added_at"] = "2026-02-01T00:00:00"
    titles = [d["title"] for d in c.get("/documents").json()]
    assert titles == ["B", "A"]


def test_empty_catalog_returns_empty_list(client):
    c, *_ = client
    assert c.get("/documents").json() == []


def test_get_document_by_id_and_404(client):
    c, _, registry, storage = client
    _seed_doc(registry, storage)
    doc_id = registry.load()["/uploads/b.pdf"]["id"]
    assert c.get(f"/documents/{doc_id}").json()["title"] == "B"
    assert c.get("/documents/00000000-0000-0000-0000-000000000000").status_code == 404
    assert c.get("/documents/not-a-uuid").status_code == 404


def test_find_latest_by_source_returns_newest_any_status():
    jobs = InMemoryJobs()
    j1 = jobs.create("/uploads/b.pdf", "b.pdf")
    jobs.claim_next(); jobs.fail(j1, "boom")
    j2 = jobs.create("/uploads/b.pdf", "b.pdf")
    assert jobs.find_latest_by_source("/uploads/b.pdf")["id"] == j2
    assert jobs.find_latest_by_source("/uploads/none.pdf") is None
