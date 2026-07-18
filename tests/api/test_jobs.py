import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from docling_rag.api.app import app, get_jobs  # noqa: E402
from tests.fakes import InMemoryJobs  # noqa: E402


@pytest.fixture
def client():
    jobs = InMemoryJobs()
    app.dependency_overrides[get_jobs] = lambda: jobs
    yield TestClient(app), jobs
    app.dependency_overrides.clear()


def test_get_job_returns_status_and_liveness(client):
    c, jobs = client
    jid = jobs.create("/uploads/b.pdf", "b.pdf", "T", None, ["x"])
    jobs.claim_next()
    jobs.update_progress(jid, "embedding", chunks_done=3, chunks_total=10)

    resp = c.get(f"/jobs/{jid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "running" and body["step"] == "embedding"
    assert body["chunks_done"] == 3 and body["chunks_total"] == 10
    assert body["elapsed_sec"] >= 0 and body["heartbeat_age_sec"] >= 0


def test_get_unknown_job_404(client):
    c, _ = client
    assert c.get("/jobs/nope").status_code == 404


def test_list_jobs(client):
    c, jobs = client
    for i in range(2):
        jobs.create(f"/uploads/{i}.pdf", f"{i}.pdf", None, None, [])
    resp = c.get("/jobs?limit=10")
    assert resp.status_code == 200 and len(resp.json()) == 2
