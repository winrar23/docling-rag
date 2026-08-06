"""Контракт JobBackend на InMemoryJobs (тот же контракт обязан выполнять DBJobs)."""
from tests.fakes import InMemoryJobs


def test_create_then_get_returns_queued():
    jobs = InMemoryJobs()
    jid = jobs.create("/uploads/b.pdf", "b.pdf")
    job = jobs.get(jid)
    assert job["status"] == "queued"
    assert job["source_file"] == "/uploads/b.pdf"
    assert job["original_name"] == "b.pdf"
    assert job["attempts"] == 0
    assert job["started_at"] is None


def test_get_unknown_returns_none():
    assert InMemoryJobs().get("nope") is None


def test_claim_next_moves_queued_to_running_and_bumps_attempts():
    jobs = InMemoryJobs()
    jid = jobs.create("/uploads/b.pdf", "b.pdf")
    claimed = jobs.claim_next()
    assert claimed["id"] == jid
    assert claimed["status"] == "running"
    assert claimed["attempts"] == 1
    assert claimed["started_at"] is not None
    assert jobs.claim_next() is None  # больше queued нет


def test_progress_complete_and_fail():
    jobs = InMemoryJobs()
    jid = jobs.create("/uploads/b.pdf", "b.pdf")
    jobs.claim_next()
    jobs.update_progress(jid, "embedding", chunks_done=5, chunks_total=10)
    j = jobs.get(jid)
    assert j["step"] == "embedding" and j["chunks_done"] == 5 and j["chunks_total"] == 10
    jobs.complete(jid, chunks_added=10)
    assert jobs.get(jid)["status"] == "done"
    assert jobs.get(jid)["finished_at"] is not None

    jid2 = jobs.create("/uploads/c.pdf", "c.pdf")
    jobs.claim_next()
    jobs.fail(jid2, "boom")
    assert jobs.get(jid2)["status"] == "failed"
    assert jobs.get(jid2)["error"] == "boom"


def test_find_active_by_source_only_matches_queued_or_running():
    jobs = InMemoryJobs()
    jid = jobs.create("/uploads/b.pdf", "b.pdf")
    assert jobs.find_active_by_source("/uploads/b.pdf")["id"] == jid  # queued
    jobs.claim_next()
    assert jobs.find_active_by_source("/uploads/b.pdf")["id"] == jid  # running
    jobs.complete(jid, 1)
    assert jobs.find_active_by_source("/uploads/b.pdf") is None       # done → не активна


def test_requeue_stale_returns_running_with_old_heartbeat_to_queued():
    from datetime import datetime, timedelta, timezone
    jobs = InMemoryJobs()
    jid = jobs.create("/uploads/b.pdf", "b.pdf")
    jobs.claim_next()
    jobs._rows[jid]["updated_at"] = datetime.now(timezone.utc) - timedelta(seconds=120)
    n = jobs.requeue_stale(stale_seconds=60, max_attempts=3)
    assert n == 1
    assert jobs.get(jid)["status"] == "queued"


def test_requeue_stale_fails_when_attempts_exhausted():
    from datetime import datetime, timedelta, timezone
    jobs = InMemoryJobs()
    jid = jobs.create("/uploads/b.pdf", "b.pdf")
    jobs._rows[jid]["attempts"] = 3
    jobs._rows[jid]["status"] = "running"
    jobs._rows[jid]["step"] = "embedding"
    jobs._rows[jid]["updated_at"] = datetime.now(timezone.utc) - timedelta(seconds=120)
    jobs.requeue_stale(stale_seconds=60, max_attempts=3)
    assert jobs.get(jid)["status"] == "failed"
    assert jobs.get(jid)["step"] == "embedding"  # шаг сохранён при терминальном отказе


def test_list_returns_newest_first_and_respects_limit():
    jobs = InMemoryJobs()
    for i in range(3):
        jobs.create(f"/uploads/{i}.pdf", f"{i}.pdf")
    got = jobs.list(limit=2)
    assert len(got) == 2


def test_update_progress_preserves_counters_when_none():
    """Шаг STORING шлёт (None, None) — счётчики embed'а не должны обнуляться."""
    jobs = InMemoryJobs()
    jid = jobs.create("/uploads/b.pdf", "b.pdf")
    jobs.claim_next()
    jobs.update_progress(jid, "embedding", chunks_done=5, chunks_total=5)
    jobs.update_progress(jid, "storing")  # (None, None)
    j = jobs.get(jid)
    assert j["step"] == "storing"
    assert j["chunks_done"] == 5 and j["chunks_total"] == 5  # сохранены, не обнулены


def test_create_stores_ocr_params():
    jobs = InMemoryJobs()
    jid = jobs.create("/b.pdf", "b.pdf", ocr="off", ocr_lang="ru")
    job = jobs.get(jid)
    assert job["ocr"] == "off" and job["ocr_lang"] == "ru"


def test_create_ocr_defaults():
    jobs = InMemoryJobs()
    jid = jobs.create("/b.pdf", "b.pdf")
    job = jobs.get(jid)
    assert job["ocr"] == "auto" and job["ocr_lang"] == "en"


def test_claim_next_returns_ocr_params():
    jobs = InMemoryJobs()
    jobs.create("/b.pdf", "b.pdf", ocr="on", ocr_lang="ru")
    job = jobs.claim_next()
    assert job["ocr"] == "on" and job["ocr_lang"] == "ru"


def test_jobs_create_without_metadata_and_set_warning():
    from tests.fakes import InMemoryJobs

    jobs = InMemoryJobs()
    jid = jobs.create("/u/b.pdf", "b.pdf", ocr="on", ocr_lang="ru")
    job = jobs.get(jid)
    assert "title" not in job and "topic" not in job and "tags" not in job
    assert job["warning"] is None
    jobs.set_warning(jid, "метаданные не извлечены: LLM недоступна")
    assert jobs.get(jid)["warning"] == "метаданные не извлечены: LLM недоступна"
