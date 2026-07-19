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


def test_get_queued_job_has_null_elapsed(client):
    c, jobs = client
    jid = jobs.create("/uploads/b.pdf", "b.pdf", None, None, [])  # queued, never claimed
    body = c.get(f"/jobs/{jid}").json()
    assert body["status"] == "queued" and body["elapsed_sec"] is None


def test_list_jobs_rejects_negative_limit(client):
    c, _ = client
    assert c.get("/jobs?limit=-1").status_code == 422


def test_list_jobs_rejects_unknown_status(client):
    c, _ = client
    assert c.get("/jobs?status=bogus").status_code == 422


def test_list_jobs_filters_by_valid_status(client):
    c, jobs = client
    jobs.create("/uploads/a.pdf", "a.pdf", None, None, [])
    jid = jobs.create("/uploads/b.pdf", "b.pdf", None, None, [])
    jobs.claim_next()  # a.pdf -> running
    resp = c.get("/jobs?status=queued")
    assert resp.status_code == 200
    assert [j["id"] for j in resp.json()] == [jid]


def test_terminal_job_liveness_frozen_at_finished_at(client):
    """elapsed_sec/heartbeat_age_sec у done/failed не растут после завершения."""
    from datetime import datetime, timedelta, timezone

    c, jobs = client
    jid = jobs.create("/uploads/b.pdf", "b.pdf", None, None, [])
    jobs.claim_next()
    jobs.complete(jid, chunks_added=3)
    now = datetime.now(timezone.utc)
    jobs._rows[jid]["started_at"] = now - timedelta(seconds=100)
    jobs._rows[jid]["updated_at"] = now - timedelta(seconds=50)
    jobs._rows[jid]["finished_at"] = now - timedelta(seconds=50)

    body = c.get(f"/jobs/{jid}").json()
    assert body["elapsed_sec"] == 50  # finished - started, а не now - started
    assert body["heartbeat_age_sec"] == 0  # finished - updated
