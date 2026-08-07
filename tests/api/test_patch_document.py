"""PATCH /documents/{id} — правка метаданных карточки (fake registry/jobs/storage)."""
import uuid

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from docling_rag.api.app import app, get_jobs, get_registry, get_settings, get_storage  # noqa: E402
from docling_rag.core.chunker import Chunk  # noqa: E402
from tests.fakes import InMemoryJobs, InMemoryRegistry, InMemoryStorage  # noqa: E402


@pytest.fixture
def client(tmp_path):
    jobs, registry, storage = InMemoryJobs(), InMemoryRegistry(), InMemoryStorage()
    app.dependency_overrides[get_jobs] = lambda: jobs
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_storage] = lambda: storage
    app.dependency_overrides[get_settings] = lambda: {
        "uploads_dir": str(tmp_path), "max_upload_mb": 1,
    }
    yield TestClient(app), jobs, registry, storage
    app.dependency_overrides.clear()


SOURCE = "/uploads/b.pdf"


def _seed(registry, storage):
    registry.upsert(SOURCE, title="Книга", topic="бд", tags=["старый-тег"], author="Автор А.")
    storage.append(
        [Chunk(text="c", source_file=SOURCE, chunk_id=0,
               page_number=1, element_type="text", headings=[])],
        np.zeros((1, 4), dtype="float32"),
    )
    return registry.load()[SOURCE]["id"]


def test_patch_updates_fields_and_returns_card(client):
    c, _, registry, storage = client
    doc_id = _seed(registry, storage)
    resp = c.patch(f"/documents/{doc_id}",
                   json={"title": "Новое", "author": "Автор Б."})
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "Новое"
    assert body["author"] == "Автор Б."
    assert body["tags"] == ["старый-тег"]  # не переданное — не тронуто
    assert body["chunks"] == 1  # полная карточка, как GET /documents/{id}


def test_patch_null_clears_and_empty_tags_clear(client):
    c, _, registry, storage = client
    doc_id = _seed(registry, storage)
    resp = c.patch(f"/documents/{doc_id}", json={"topic": None, "tags": []})
    assert resp.status_code == 200
    assert resp.json()["topic"] is None
    assert resp.json()["tags"] == []


def test_patch_unknown_document_404(client):
    c, *_ = client
    assert c.patch(f"/documents/{uuid.uuid4()}", json={"title": "X"}).status_code == 404
    assert c.patch("/documents/not-a-uuid", json={"title": "X"}).status_code == 404


def test_patch_active_job_409(client):
    c, jobs, registry, storage = client
    doc_id = _seed(registry, storage)
    jid = jobs.create(SOURCE, "b.pdf")  # queued — воркер затёр бы правку
    resp = c.patch(f"/documents/{doc_id}", json={"title": "X"})
    assert resp.status_code == 409
    assert resp.json()["detail"]["job_id"] == jid
    assert registry.get(SOURCE)["title"] == "Книга"  # ничего не изменено


def test_patch_unknown_field_422(client):
    c, _, registry, storage = client
    doc_id = _seed(registry, storage)
    assert c.patch(f"/documents/{doc_id}", json={"pages": 5}).status_code == 422
