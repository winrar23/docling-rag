# api/app.py — этап 4 A: приём книг (ingestion). Каталог/чат — этапы B/C.
import os
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile

from docling_rag.cli.config_loader import load_config
from docling_rag.core.parser import SUPPORTED_EXTENSIONS
from docling_rag.core.protocols import JobBackend
from docling_rag.storage.db_jobs import DBJobs

app = FastAPI(title="docling-rag")


def get_settings() -> dict:
    return load_config()


def get_jobs(settings: dict = Depends(get_settings)) -> JobBackend:
    return DBJobs(settings["database_url"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents", status_code=202)
async def create_document(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    topic: str | None = Form(None),
    tags: list[str] = Form(default=[]),
    settings: dict = Depends(get_settings),
    jobs: JobBackend = Depends(get_jobs),
) -> dict:
    name = os.path.basename(file.filename or "")  # защита от path traversal
    ext = os.path.splitext(name)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Формат {ext or '?'} не поддерживается. Разрешено: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )
    uploads_dir = settings["uploads_dir"]
    source_file = os.path.join(uploads_dir, name)

    existing = jobs.find_active_by_source(source_file)
    if existing is not None:
        raise HTTPException(status_code=409,
                            detail={"message": "Уже индексируется", "job_id": existing["id"]})

    os.makedirs(uploads_dir, exist_ok=True)
    with open(source_file, "wb") as f:
        f.write(await file.read())

    job_id = jobs.create(source_file, name, title, topic, tags)
    return {"job_id": job_id, "status": "queued"}


def _with_liveness(job: dict) -> dict:
    now = datetime.now(timezone.utc)
    started, updated = job.get("started_at"), job.get("updated_at")
    job = dict(job)
    job["elapsed_sec"] = int((now - started).total_seconds()) if started else None
    job["heartbeat_age_sec"] = int((now - updated).total_seconds()) if updated else None
    return job


@app.get("/jobs/{job_id}")
def get_job(job_id: str, jobs: JobBackend = Depends(get_jobs)) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Джоба не найдена")
    return _with_liveness(job)


@app.get("/jobs")
def list_jobs(limit: int = Query(default=20, ge=1, le=100), status: str | None = None,
              jobs: JobBackend = Depends(get_jobs)) -> list[dict]:
    return [_with_liveness(j) for j in jobs.list(limit=limit, status=status)]
