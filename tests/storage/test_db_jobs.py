"""Integration DBJobs: реальная тест-БД docling_rag_test."""
import pytest

from docling_rag.storage.db_jobs import DBJobs

pytestmark = pytest.mark.integration

# db_url/clean_db реэкспортированы в conftest из tests/storage/test_db_backends.py
from tests.storage.test_db_backends import clean_db, db_url  # noqa: F401


@pytest.fixture
def jobs(clean_db):
    import psycopg
    with psycopg.connect(clean_db) as conn:
        conn.execute("TRUNCATE jobs")
        conn.commit()
    return DBJobs(clean_db)


def test_create_get_lifecycle(jobs):
    jid = jobs.create("/uploads/b.pdf", "b.pdf", "T", "topic", ["a", "b"])
    job = jobs.get(jid)
    assert job["status"] == "queued" and job["tags"] == ["a", "b"]
    assert job["source_file"] == "/uploads/b.pdf"

    claimed = jobs.claim_next()
    assert claimed["id"] == jid and claimed["status"] == "running" and claimed["attempts"] == 1
    assert jobs.claim_next() is None

    jobs.update_progress(jid, "embedding", chunks_done=3, chunks_total=7)
    assert jobs.get(jid)["step"] == "embedding" and jobs.get(jid)["chunks_done"] == 3

    jobs.complete(jid, chunks_added=7)
    done = jobs.get(jid)
    assert done["status"] == "done" and done["finished_at"] is not None


def test_find_active_and_dedup(jobs):
    jid = jobs.create("/uploads/b.pdf", "b.pdf", None, None, [])
    assert jobs.find_active_by_source("/uploads/b.pdf")["id"] == jid
    jobs.claim_next(); jobs.complete(jid, 1)
    assert jobs.find_active_by_source("/uploads/b.pdf") is None


def test_fail_sets_error(jobs):
    jid = jobs.create("/uploads/x.pdf", "x.pdf", None, None, [])
    jobs.claim_next()
    jobs.fail(jid, "parse boom")
    assert jobs.get(jid)["status"] == "failed" and jobs.get(jid)["error"] == "parse boom"


def test_requeue_stale(jobs, clean_db):
    import psycopg
    jid = jobs.create("/uploads/s.pdf", "s.pdf", None, None, [])
    jobs.claim_next()
    with psycopg.connect(clean_db) as conn:  # состарить heartbeat
        conn.execute("UPDATE jobs SET updated_at = now() - interval '120 seconds' WHERE id = %s::uuid", (jid,))
        conn.commit()
    n = jobs.requeue_stale(stale_seconds=60, max_attempts=3)
    assert n == 1 and jobs.get(jid)["status"] == "queued"


def test_requeue_stale_exhausted_attempts_preserves_step(jobs, clean_db):
    import psycopg
    jid = jobs.create("/uploads/s.pdf", "s.pdf", None, None, [])
    jobs.claim_next()
    with psycopg.connect(clean_db) as conn:  # исчерпать попытки + состарить heartbeat
        conn.execute(
            "UPDATE jobs SET attempts=3, step='embedding',"
            " updated_at = now() - interval '120 seconds' WHERE id = %s::uuid",
            (jid,),
        )
        conn.commit()
    jobs.requeue_stale(stale_seconds=60, max_attempts=3)
    job = jobs.get(jid)
    assert job["status"] == "failed"
    assert job["step"] == "embedding"  # шаг сохранён при терминальном отказе


def test_get_unknown_returns_none(jobs):
    import uuid
    assert jobs.get(str(uuid.uuid4())) is None


def test_get_malformed_uuid_returns_none(jobs):
    assert jobs.get("not-a-uuid") is None
