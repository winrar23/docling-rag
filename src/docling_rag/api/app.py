# api/app.py — этап 4 A: приём книг (ingestion). Каталог/чат — этапы B/C.
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from docling_rag.cli.config_loader import load_config
from docling_rag.core.errors import (
    EmbedServiceUnavailableError,
    StorageSchemaMissingError,
    StorageUnavailableError,
)
from docling_rag.core.parser import SUPPORTED_EXTENSIONS
from docling_rag.core.protocols import JobBackend
from docling_rag.storage.db_jobs import DBJobs

app = FastAPI(title="docling-rag")


@app.exception_handler(StorageUnavailableError)
def _storage_unavailable_503(request: Request, exc: StorageUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": f"PostgreSQL недоступен: {exc}"})


@app.exception_handler(StorageSchemaMissingError)
def _schema_missing_503(request: Request, exc: StorageSchemaMissingError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "Схема БД не инициализирована. Выполните: docling-rag init"})


@app.exception_handler(EmbedServiceUnavailableError)
def _embed_unavailable_503(request: Request, exc: EmbedServiceUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": f"Сервис эмбеддингов недоступен: {exc}"})


@lru_cache
def get_settings() -> dict:
    # Конфиг читается один раз на процесс; смена config.yaml требует рестарта API.
    return load_config()


def get_jobs(settings: dict = Depends(get_settings)) -> JobBackend:
    return DBJobs(settings["database_url"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _save_upload(src, dest: str, max_bytes: int) -> None:
    """Чанковая запись в dest+'.part' с лимитом, затем атомарная замена dest.

    Не держит файл в памяти целиком; при превышении лимита прежний dest не трогается.
    """
    part = dest + ".part"
    try:
        with open(part, "wb") as f:
            written = 0
            while chunk := src.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Файл больше лимита {max_bytes // (1024 * 1024)} МБ (ключ конфига max_upload_mb)",
                    )
                f.write(chunk)
        os.replace(part, dest)
    finally:
        if os.path.exists(part):
            os.remove(part)


@app.post("/documents", status_code=202)
def create_document(  # sync def: FastAPI уводит в threadpool — файловый и БД I/O не блокируют event loop
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
    # resolve — тот же ключ, что indexer пишет в documents.source_file
    # (Path.resolve() в index_files); иначе джобу не скоррелировать с документом.
    source_file = str((Path(uploads_dir) / name).resolve())

    existing = jobs.find_active_by_source(source_file)
    if existing is not None:
        raise HTTPException(status_code=409,
                            detail={"message": "Уже индексируется", "job_id": existing["id"]})

    os.makedirs(uploads_dir, exist_ok=True)
    _save_upload(file.file, source_file, int(settings["max_upload_mb"]) * 1024 * 1024)

    job_id = jobs.create(source_file, name, title, topic, tags)
    return {"job_id": job_id, "status": "queued"}


def _with_liveness(job: dict) -> dict:
    now = datetime.now(timezone.utc)
    finished = job.get("finished_at")
    if finished is not None:  # done/failed: счётчики заморожены на моменте завершения
        now = min(now, finished)
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
def list_jobs(limit: int = Query(default=20, ge=1, le=100),
              status: Literal["queued", "running", "done", "failed"] | None = None,
              jobs: JobBackend = Depends(get_jobs)) -> list[dict]:
    return [_with_liveness(j) for j in jobs.list(limit=limit, status=status)]
