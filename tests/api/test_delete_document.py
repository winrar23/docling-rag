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
    yield TestClient(app), jobs, registry, storage, tmp_path
    app.dependency_overrides.clear()


def _seed(registry, storage, source, title="T"):
    registry.upsert(source, title, None, [])
    storage.append(
        [Chunk(text="c", source_file=source, chunk_id=0,
               page_number=1, element_type="text", headings=[])],
        np.zeros((1, 4), dtype="float32"),
    )
    return registry.load()[source]["id"]


def test_delete_removes_doc_chunks_and_file(client):
    c, _, registry, storage, uploads = client
    source = str(uploads / "b.pdf")
    (uploads / "b.pdf").write_bytes(b"x")
    doc_id = _seed(registry, storage, source)

    resp = c.delete(f"/documents/{doc_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] == "T" and body["chunks"] == 1 and body["file_removed"] is True
    assert registry.get(source) is None
    assert storage.count_by_source(source) == 0
    assert not (uploads / "b.pdf").exists()


def test_delete_file_outside_uploads_not_touched(client, tmp_path_factory):
    c, _, registry, storage, _ = client
    outside = tmp_path_factory.mktemp("elsewhere") / "cli-era.pdf"
    outside.write_bytes(b"x")
    doc_id = _seed(registry, storage, str(outside))
    body = c.delete(f"/documents/{doc_id}").json()
    assert body["file_removed"] is False
    assert outside.exists()


def test_delete_active_job_409(client):
    c, jobs, registry, storage, uploads = client
    source = str(uploads / "b.pdf")
    doc_id = _seed(registry, storage, source)
    jid = jobs.create(source, "b.pdf", None, None, [])  # queued
    resp = c.delete(f"/documents/{doc_id}")
    assert resp.status_code == 409
    assert resp.json()["detail"]["job_id"] == jid
    assert registry.get(source) is not None  # ничего не удалено


def test_delete_unknown_404(client):
    c, *_ = client
    assert c.delete("/documents/00000000-0000-0000-0000-000000000000").status_code == 404
    assert c.delete("/documents/not-a-uuid").status_code == 404
