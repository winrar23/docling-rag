"""Воркер фоновой индексации. Вне core/ — импортирует storage/psycopg-слой.

Тонкий слой: берёт джобу из JobBackend, зовёт core.index_files с колбэком,
держит heartbeat, ставит done/failed. Цикл run_loop — для контейнера.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

from docling_rag.core.indexer import index_files
from docling_rag.core.protocols import (
    DocumentRegistryBackend,
    JobBackend,
    StorageBackend,
)


@dataclass
class WorkerDeps:
    parser: object
    embedder: object
    storage: StorageBackend
    registry: DocumentRegistryBackend
    embedding_model: str
    chunk_max_tokens: int = 512


def make_progress(jobs: JobBackend, job_id: str):
    def _cb(step: str, done: int | None, total: int | None) -> None:
        jobs.update_progress(job_id, step, chunks_done=done, chunks_total=total)
    return _cb


class _Heartbeat:
    """Бьёт jobs.heartbeat раз в interval, пока джоба активна (даже в молчащем parse)."""

    def __init__(self, jobs: JobBackend, job_id: str, interval: float = 10.0) -> None:
        self._jobs, self._id, self._interval = jobs, job_id, interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._jobs.heartbeat(self._id)
            except Exception:  # heartbeat не должен ронять джобу
                pass

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1)


def process_one_job(jobs: JobBackend, deps: WorkerDeps, job: dict, index_fn=index_files) -> None:
    job_id = job["id"]
    try:
        report = index_fn(
            [Path(job["source_file"])], deps.parser, deps.embedder, deps.storage, deps.registry,
            embedding_model=deps.embedding_model, chunk_max_tokens=deps.chunk_max_tokens,
            title=job["title"], topic=job["topic"], tags=job["tags"] or (),
            on_progress=make_progress(jobs, job_id),
        )
        if report.files_failed or report.chunks_added == 0:
            msg = "; ".join(f"{s}: {e}" for s, e in report.errors) or "0 chunks (пустой документ?)"
            jobs.fail(job_id, msg)
        else:
            jobs.complete(job_id, report.chunks_added)
    except Exception as e:
        jobs.fail(job_id, f"{type(e).__name__}: {e}")


def run_loop(jobs: JobBackend, deps: WorkerDeps, *, poll_interval: float = 3.0,
             stale_seconds: int = 60, max_attempts: int = 3, stop: threading.Event | None = None) -> None:
    jobs.requeue_stale(stale_seconds, max_attempts)
    while stop is None or not stop.is_set():
        job = jobs.claim_next()
        if job is None:
            jobs.requeue_stale(stale_seconds, max_attempts)
            time.sleep(poll_interval)
            continue
        with _Heartbeat(jobs, job["id"]):
            process_one_job(jobs, deps, job)
