"""Integration DBJobs: реальная тест-БД docling_rag_test."""
import psycopg
import pytest

from docling_rag.storage.db_jobs import DBJobs
from docling_rag.storage.db_schema import init_schema

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


def test_claim_next_skip_locked_concurrent(jobs, clean_db):
    """Два конкурентных claim'а не берут одну джобу и не блокируются (FOR UPDATE SKIP LOCKED)."""
    import threading
    import psycopg

    j1 = jobs.create("/uploads/1.pdf", "1.pdf", None, None, [])
    j2 = jobs.create("/uploads/2.pdf", "2.pdf", None, None, [])
    with psycopg.connect(clean_db) as conn:  # j1 старше — первый claim возьмёт именно её
        conn.execute(
            "UPDATE jobs SET created_at = now() - interval '10 seconds' WHERE id = %s::uuid", (j1,)
        )
        conn.commit()

    with psycopg.connect(clean_db) as txn1:  # незакоммиченная транзакция держит лок j1
        claimed1 = txn1.execute(
            "UPDATE jobs SET status='running' WHERE id = ("
            " SELECT id FROM jobs WHERE status='queued' ORDER BY created_at"
            " FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING id"
        ).fetchone()
        assert str(claimed1[0]) == j1

        result = {}
        t = threading.Thread(target=lambda: result.update(job=jobs.claim_next()), daemon=True)
        t.start()
        t.join(timeout=5)
        assert not t.is_alive(), "claim_next завис на чужом локе — SKIP LOCKED не работает"
        assert result["job"]["id"] == j2  # взял вторую джобу, а не заблокировался на первой
        txn1.rollback()


def test_heartbeat_advances_updated_at(jobs, clean_db):
    import psycopg

    jid = jobs.create("/uploads/h.pdf", "h.pdf", None, None, [])
    jobs.claim_next()
    with psycopg.connect(clean_db) as conn:  # состарить heartbeat
        conn.execute(
            "UPDATE jobs SET updated_at = now() - interval '300 seconds' WHERE id = %s::uuid", (jid,)
        )
        conn.commit()
    before = jobs.get(jid)["updated_at"]
    jobs.heartbeat(jid)
    after = jobs.get(jid)["updated_at"]
    assert (after - before).total_seconds() > 250  # updated_at сдвинут к now()


def test_list_orders_limits_and_filters(jobs, clean_db):
    import psycopg

    j1 = jobs.create("/uploads/1.pdf", "1.pdf", None, None, [])
    j2 = jobs.create("/uploads/2.pdf", "2.pdf", None, None, [])
    j3 = jobs.create("/uploads/3.pdf", "3.pdf", None, None, [])
    with psycopg.connect(clean_db) as conn:  # детерминированный порядок created_at: j1 < j2 < j3
        for age, jid in ((30, j1), (20, j2), (10, j3)):
            conn.execute(
                "UPDATE jobs SET created_at = now() - make_interval(secs => %s) WHERE id = %s::uuid",
                (age, jid),
            )
        conn.commit()

    assert [r["id"] for r in jobs.list(limit=2)] == [j3, j2]  # новые первыми, limit режет
    jobs.claim_next()  # j1 (старейшая queued) -> running
    assert [r["id"] for r in jobs.list(status="queued")] == [j3, j2]
    assert [r["id"] for r in jobs.list(status="running")] == [j1]


def test_find_latest_by_source_integration(jobs):
    j1 = jobs.create("/uploads/l.pdf", "l.pdf", None, None, [])
    jobs.claim_next(); jobs.fail(j1, "x")
    j2 = jobs.create("/uploads/l.pdf", "l.pdf", None, None, [])
    assert jobs.find_latest_by_source("/uploads/l.pdf")["id"] == j2


def test_update_progress_preserves_counters_when_none(jobs):
    """Шаг STORING шлёт (None, None) — COALESCE не обнуляет счётчики embed'а."""
    jid = jobs.create("/uploads/b.pdf", "b.pdf", None, None, [])
    jobs.claim_next()
    jobs.update_progress(jid, "embedding", chunks_done=5, chunks_total=5)
    jobs.update_progress(jid, "storing")  # (None, None)
    j = jobs.get(jid)
    assert j["step"] == "storing"
    assert j["chunks_done"] == 5 and j["chunks_total"] == 5


def test_jobs_ocr_fields_roundtrip(jobs):
    jid = jobs.create("/b.pdf", "b.pdf", None, None, [], ocr="off", ocr_lang="ru")
    job = jobs.get(jid)
    assert job["ocr"] == "off" and job["ocr_lang"] == "ru"
    claimed = None
    while (j := jobs.claim_next()) is not None:  # добираемся до своей джобы
        if j["id"] == jid:
            claimed = j
    assert claimed is not None and claimed["ocr"] == "off" and claimed["ocr_lang"] == "ru"


def test_jobs_ocr_defaults_in_db(jobs):
    jid = jobs.create("/b.pdf", "b.pdf", None, None, [])
    job = jobs.get(jid)
    assert job["ocr"] == "auto" and job["ocr_lang"] == "en"


def test_init_schema_twice_keeps_ocr_columns(db_url):
    """Повторный init на существующей схеме — идемпотентен, колонки на месте."""
    init_schema(db_url)
    init_schema(db_url)
    with psycopg.connect(db_url) as conn:
        cols = {r[0] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'jobs'"
        ).fetchall()}
    assert {"ocr", "ocr_lang"} <= cols
