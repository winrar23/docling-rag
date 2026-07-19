"""Юниты воркера на InMemoryJobs + фейковый индексатор (без postgres/моделей)."""
from types import SimpleNamespace

from docling_rag.core.indexer import IndexReport, EMBEDDING
from docling_rag.worker.runner import WorkerDeps, process_one_job, make_progress
from tests.fakes import InMemoryJobs


def _deps():
    return WorkerDeps(parser=object(), embedder=object(), storage=object(),
                      registry=object(), embedding_model="m", chunk_max_tokens=512)


def test_process_one_job_success_marks_done():
    jobs = InMemoryJobs()
    jid = jobs.create("/uploads/b.pdf", "b.pdf", "T", None, ["x"])
    job = jobs.claim_next()

    def fake_index(files, *a, on_progress=None, **k):
        on_progress(EMBEDDING, 2, 2)
        return IndexReport(chunks_added=2, files_ok=1)

    process_one_job(jobs, _deps(), job, index_fn=fake_index)
    done = jobs.get(jid)
    assert done["status"] == "done" and done["chunks_done"] == 2


def test_process_one_job_zero_chunks_marks_failed():
    jobs = InMemoryJobs()
    jid = jobs.create("/uploads/b.pdf", "b.pdf", None, None, [])
    job = jobs.claim_next()
    process_one_job(jobs, _deps(), job,
                    index_fn=lambda *a, **k: IndexReport(chunks_added=0, files_ok=1))
    assert jobs.get(jid)["status"] == "failed"


def test_process_one_job_exception_marks_failed():
    jobs = InMemoryJobs()
    jid = jobs.create("/uploads/b.pdf", "b.pdf", None, None, [])
    job = jobs.claim_next()

    def boom(*a, **k):
        raise RuntimeError("docling died")

    process_one_job(jobs, _deps(), job, index_fn=boom)
    j = jobs.get(jid)
    assert j["status"] == "failed" and "docling died" in j["error"]


def test_make_progress_writes_step_to_job():
    jobs = InMemoryJobs()
    jid = jobs.create("/uploads/b.pdf", "b.pdf", None, None, [])
    jobs.claim_next()
    cb = make_progress(jobs, jid)
    cb(EMBEDDING, 3, 10)
    j = jobs.get(jid)
    assert j["step"] == EMBEDDING and j["chunks_done"] == 3 and j["chunks_total"] == 10


def test_run_loop_survives_transient_claim_failure(capsys):
    """Обрыв postgres в claim_next не роняет воркер: stderr + пауза + повтор."""
    import threading

    from docling_rag.worker.runner import run_loop

    jobs = InMemoryJobs()
    jid = jobs.create("/uploads/b.pdf", "b.pdf", None, None, [])
    stop = threading.Event()
    calls = {"n": 0}

    class FlakyJobs:
        """Первый claim_next падает (как при недоступном pg), дальше делегирует."""

        def __getattr__(self, name):
            return getattr(jobs, name)

        def claim_next(self):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("pg down")
            return jobs.claim_next()

    orig_complete = jobs.complete

    def complete_and_stop(job_id, chunks_added):
        orig_complete(job_id, chunks_added)
        stop.set()

    jobs.complete = complete_and_stop

    run_loop(FlakyJobs(), _deps(), poll_interval=0.01, stop=stop,
             index_fn=lambda *a, **k: IndexReport(chunks_added=1, files_ok=1))

    assert jobs.get(jid)["status"] == "done"  # цикл пережил сбой и обработал джобу
    assert calls["n"] >= 2
    assert "pg down" in capsys.readouterr().err


def test_build_deps_wires_from_config(monkeypatch):
    import docling_rag.worker.__main__ as wmain

    monkeypatch.setattr(wmain, "Parser", lambda: "PARSER")
    monkeypatch.setattr(wmain, "Embedder", lambda model: f"EMB:{model}")
    monkeypatch.setattr(wmain, "DBStorage", lambda dsn: f"ST:{dsn}")
    monkeypatch.setattr(wmain, "DBRegistry", lambda dsn: f"RG:{dsn}")

    deps = wmain.build_deps({
        "database_url": "postgresql://x", "embedding_model": "m", "chunk_max_tokens": 256,
    })
    assert deps.embedder == "EMB:m" and deps.storage == "ST:postgresql://x"
    assert deps.chunk_max_tokens == 256
