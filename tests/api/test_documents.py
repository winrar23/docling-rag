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
    app.dependency_overrides[get_settings] = lambda: {
        "uploads_dir": str(tmp_path), "max_upload_mb": 1,
    }
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
    assert job["original_name"] == "book.pdf"
    assert (uploads / "book.pdf").read_bytes() == b"%PDF-1.4 fake"


def test_post_documents_rejects_bad_extension_400(client):
    c, jobs, uploads = client
    resp = c.post("/documents", files={"file": ("notes.txt", io.BytesIO(b"x"), "text/plain")})
    assert resp.status_code == 400
    assert jobs.list() == []  # без побочных эффектов:
    assert list(uploads.iterdir()) == []  # ни джобы, ни файла


def test_post_documents_rejects_oversized_413(client):
    c, jobs, uploads = client
    big = io.BytesIO(b"x" * (1024 * 1024 + 1))  # лимит в фикстуре — 1 МБ
    resp = c.post("/documents", files={"file": ("big.pdf", big, "application/pdf")})
    assert resp.status_code == 413
    assert jobs.list() == []  # джоба не создана
    assert list(uploads.iterdir()) == []  # частичный файл убран


def test_post_documents_oversized_keeps_previous_file(client):
    """Превышение лимита при перезаливке не должно портить уже лежащий файл."""
    c, _, uploads = client
    (uploads / "b.pdf").write_bytes(b"old")
    big = io.BytesIO(b"x" * (1024 * 1024 + 1))
    resp = c.post("/documents", files={"file": ("b.pdf", big, "application/pdf")})
    assert resp.status_code == 413
    assert (uploads / "b.pdf").read_bytes() == b"old"


def test_post_documents_dedup_returns_409_with_existing_job(client):
    c, jobs, _ = client
    r1 = c.post("/documents", files={"file": ("b.pdf", io.BytesIO(b"a"), "application/pdf")})
    assert r1.status_code == 202
    r2 = c.post("/documents", files={"file": ("b.pdf", io.BytesIO(b"a"), "application/pdf")})
    assert r2.status_code == 409
    assert r2.json()["detail"]["job_id"] == r1.json()["job_id"]


def test_post_documents_stores_resolved_source_file(tmp_path):
    """jobs.source_file должен совпадать с documents.source_file (indexer резолвит путь).

    uploads_dir через symlink: без resolve в API ключи расходятся — этап B
    не сможет скоррелировать джобу с документом.
    """
    real = tmp_path / "real-uploads"
    real.mkdir()
    link = tmp_path / "link-uploads"
    link.symlink_to(real)
    jobs = InMemoryJobs()
    app.dependency_overrides[get_jobs] = lambda: jobs
    app.dependency_overrides[get_settings] = lambda: {
        "uploads_dir": str(link), "max_upload_mb": 1,
    }
    try:
        c = TestClient(app)
        resp = c.post("/documents", files={"file": ("b.pdf", io.BytesIO(b"a"), "application/pdf")})
        assert resp.status_code == 202
        job = jobs.get(resp.json()["job_id"])
        assert job["source_file"] == str((real / "b.pdf").resolve())
    finally:
        app.dependency_overrides.clear()


def test_post_documents_strips_path_traversal(client):
    c, jobs, uploads = client
    resp = c.post("/documents", files={"file": ("../../evil.pdf", io.BytesIO(b"a"), "application/pdf")})
    assert resp.status_code == 202
    assert (uploads / "evil.pdf").exists()  # только basename


def test_post_documents_accepts_ocr_fields(client):
    c, jobs, _ = client
    resp = c.post(
        "/documents",
        files={"file": ("b.pdf", io.BytesIO(b"a"), "application/pdf")},
        data={"ocr": "off", "ocr_lang": "ru"},
    )
    assert resp.status_code == 202
    job = jobs.get(resp.json()["job_id"])
    assert job["ocr"] == "off" and job["ocr_lang"] == "ru"


def test_post_documents_ocr_defaults(client):
    c, jobs, _ = client
    resp = c.post("/documents", files={"file": ("b.pdf", io.BytesIO(b"a"), "application/pdf")})
    assert resp.status_code == 202
    job = jobs.get(resp.json()["job_id"])
    assert job["ocr"] == "auto" and job["ocr_lang"] == "en"


def test_post_documents_rejects_invalid_ocr_422(client):
    c, jobs, uploads = client
    resp = c.post(
        "/documents",
        files={"file": ("b.pdf", io.BytesIO(b"a"), "application/pdf")},
        data={"ocr": "always"},
    )
    assert resp.status_code == 422
    assert jobs.list() == []  # джоба не создана


def test_post_documents_rejects_invalid_ocr_lang_422(client):
    c, jobs, _ = client
    resp = c.post(
        "/documents",
        files={"file": ("b.pdf", io.BytesIO(b"a"), "application/pdf")},
        data={"ocr_lang": "de"},
    )
    assert resp.status_code == 422
