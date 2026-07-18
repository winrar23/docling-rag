# storage/db_jobs.py
"""Очередь фоновых джоб индексации в PostgreSQL. Реализует JobBackend (core/protocols.py)."""
import psycopg
from psycopg.rows import dict_row

from docling_rag.storage.db_storage import _translate_db_errors

_COLS = ("id, source_file, original_name, title, topic, tags, status, step, "
         "chunks_total, chunks_done, error, attempts, created_at, started_at, "
         "updated_at, finished_at")


def _normalize(row: dict | None) -> dict | None:
    if row is None:
        return None
    row["id"] = str(row["id"])  # uuid -> str (единый контракт с InMemoryJobs)
    return row


class DBJobs:
    """Соединение на операцию — как DBStorage/DBRegistry/DBSearchLog."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self._dsn, row_factory=dict_row)

    def create(self, source_file, original_name, title, topic, tags) -> str:
        with _translate_db_errors(), self._connect() as conn:
            row = conn.execute(
                "INSERT INTO jobs (source_file, original_name, title, topic, tags)"
                " VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (source_file, original_name, title, topic, list(tags)),
            ).fetchone()
            conn.commit()
        return str(row["id"])

    def get(self, job_id):
        with _translate_db_errors(), self._connect() as conn:
            row = conn.execute(
                f"SELECT {_COLS} FROM jobs WHERE id = %s::uuid", (job_id,)
            ).fetchone()
        return _normalize(row)

    def list(self, limit=20, status=None):
        with _translate_db_errors(), self._connect() as conn:
            if status is None:
                rows = conn.execute(
                    f"SELECT {_COLS} FROM jobs ORDER BY created_at DESC LIMIT %s", (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT {_COLS} FROM jobs WHERE status = %s ORDER BY created_at DESC LIMIT %s",
                    (status, limit),
                ).fetchall()
        return [_normalize(r) for r in rows]

    def find_active_by_source(self, source_file):
        with _translate_db_errors(), self._connect() as conn:
            row = conn.execute(
                f"SELECT {_COLS} FROM jobs WHERE source_file = %s"
                " AND status IN ('queued','running') ORDER BY created_at DESC LIMIT 1",
                (source_file,),
            ).fetchone()
        return _normalize(row)

    def claim_next(self):
        with _translate_db_errors(), self._connect() as conn:
            row = conn.execute(
                "UPDATE jobs SET status='running', started_at=now(), updated_at=now(),"
                " attempts=attempts+1 WHERE id = ("
                "  SELECT id FROM jobs WHERE status='queued'"
                "  ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1"
                f") RETURNING {_COLS}"
            ).fetchone()
            conn.commit()
        return _normalize(row)

    def update_progress(self, job_id, step, chunks_done=None, chunks_total=None):
        with _translate_db_errors(), self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET step=%s, chunks_done=%s, chunks_total=%s, updated_at=now()"
                " WHERE id = %s::uuid",
                (step, chunks_done, chunks_total, job_id),
            )
            conn.commit()

    def heartbeat(self, job_id):
        with _translate_db_errors(), self._connect() as conn:
            conn.execute("UPDATE jobs SET updated_at=now() WHERE id = %s::uuid", (job_id,))
            conn.commit()

    def complete(self, job_id, chunks_added):
        with _translate_db_errors(), self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='done', step=NULL, chunks_done=%s,"
                " updated_at=now(), finished_at=now() WHERE id = %s::uuid",
                (chunks_added, job_id),
            )
            conn.commit()

    def fail(self, job_id, error):
        with _translate_db_errors(), self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='failed', error=%s, updated_at=now(), finished_at=now()"
                " WHERE id = %s::uuid",
                (error, job_id),
            )
            conn.commit()

    def requeue_stale(self, stale_seconds, max_attempts):
        with _translate_db_errors(), self._connect() as conn:
            rows = conn.execute(
                "UPDATE jobs SET"
                "  status = CASE WHEN attempts >= %(max)s THEN 'failed' ELSE 'queued' END,"
                "  error = CASE WHEN attempts >= %(max)s THEN 'воркер умирал, превышен лимит попыток' ELSE error END,"
                "  step = CASE WHEN attempts >= %(max)s THEN step ELSE NULL END, updated_at = now(),"
                "  finished_at = CASE WHEN attempts >= %(max)s THEN now() ELSE finished_at END"
                " WHERE status='running' AND updated_at < now() - make_interval(secs => %(stale)s)"
                " RETURNING id",
                {"max": max_attempts, "stale": stale_seconds},
            ).fetchall()
            conn.commit()
        return len(rows)
