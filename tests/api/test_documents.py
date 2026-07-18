import io

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")  # python-multipart

from fastapi.testclient import TestClient  # noqa: E402

from docling_rag.api.app import app, get_jobs, get_settings  # noqa: E402
from tests.fakes import InMemoryJobs  # noqa: E402


@pytest.fixture
def client(tmp_path):
    jobs = InMemoryJobs()
    app.dependency_overrides[get_jobs] = lambda: jobs
    app.dependency_overrides[get_settings] = lambda: {"uploads_dir": str(tmp_path)}
    yield TestClient(app), jobs, tmp_path
    app.dependency_overrides.clear()


def test_post_documents_accepts_pdf_returns_202(client):
    c, jobs, uploads = client
    resp = c.post(
        "/documents",
        files={"file": ("book.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
        data={"title": "T", "topic": "sys", "tags": ["arch", "data"]},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued" and body["job_id"]
    job = jobs.get(body["job_id"])
    assert job["original_name"] == "book.pdf" and job["tags"] == ["arch", "data"]
    assert (uploads / "book.pdf").read_bytes() == b"%PDF-1.4 fake"


def test_post_documents_rejects_bad_extension_400(client):
    c, _, _ = client
    resp = c.post("/documents", files={"file": ("notes.txt", io.BytesIO(b"x"), "text/plain")})
    assert resp.status_code == 400


def test_post_documents_dedup_returns_409_with_existing_job(client):
    c, jobs, _ = client
    r1 = c.post("/documents", files={"file": ("b.pdf", io.BytesIO(b"a"), "application/pdf")})
    assert r1.status_code == 202
    r2 = c.post("/documents", files={"file": ("b.pdf", io.BytesIO(b"a"), "application/pdf")})
    assert r2.status_code == 409
    assert r2.json()["detail"]["job_id"] == r1.json()["job_id"]


def test_post_documents_strips_path_traversal(client):
    c, jobs, uploads = client
    resp = c.post("/documents", files={"file": ("../../evil.pdf", io.BytesIO(b"a"), "application/pdf")})
    assert resp.status_code == 202
    assert (uploads / "evil.pdf").exists()  # только basename
