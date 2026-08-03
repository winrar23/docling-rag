# api/app.py — этап 4-A (ingestion) + 4-B (каталог, поиск) + 4-C (чат) + 4-D (SPA-статика).
import os
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from docling_rag.cli.config_loader import load_config
from docling_rag.core.embedder import get_embedder
from docling_rag.core.errors import (
    EmbedServiceUnavailableError,
    StorageSchemaMissingError,
    StorageUnavailableError,
    cause_chain,
)
from docling_rag.core.parser import SUPPORTED_EXTENSIONS
from docling_rag.core.protocols import (
    DocumentRegistryBackend,
    EmbedderBackend,
    JobBackend,
    SearchLogBackend,
    StorageBackend,
)
from docling_rag.core.search import resolve_allowed_sources, run_search
from docling_rag.storage.db_jobs import DBJobs
from docling_rag.storage.db_registry import DBRegistry
from docling_rag.storage.db_search_log import DBSearchLog
from docling_rag.storage.db_storage import DBStorage

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


def get_registry(settings: dict = Depends(get_settings)) -> DocumentRegistryBackend:
    return DBRegistry(settings["database_url"])


def get_storage(settings: dict = Depends(get_settings)) -> StorageBackend:
    return DBStorage(settings["database_url"])


@lru_cache(maxsize=2)
def _embedder_singleton(embed_url: str | None, model: str) -> EmbedderBackend:
    return get_embedder({"embed_url": embed_url, "embedding_model": model})


def get_search_embedder(settings: dict = Depends(get_settings)) -> EmbedderBackend:
    return _embedder_singleton(settings.get("embed_url"), settings.get("embedding_model", ""))


def get_search_log(settings: dict = Depends(get_settings)) -> SearchLogBackend:
    return DBSearchLog(settings["database_url"])


# --- Chat (этап 4-C) ---------------------------------------------------------

class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatTurn] = []


class ChatSource(BaseModel):
    file: str          # basename — как в текстовых цитатах агента
    page: int
    headings: list[str]
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]


def _import_agent():
    """Ленивый импорт .[agent]: модуль api обязан импортироваться без pydantic-ai."""
    from docling_rag.core.agent import AgentDeps, build_lmstudio_model, create_agent
    return create_agent, AgentDeps, build_lmstudio_model


def get_chat_model(settings: dict = Depends(get_settings)):
    if not settings.get("agent_enabled"):
        raise HTTPException(status_code=503,
                            detail="Агент выключен: agent_enabled: false в конфиге")
    try:
        _, _, build_model = _import_agent()
    except ImportError:
        raise HTTPException(status_code=503,
                            detail="pydantic-ai не установлен (extra .[agent])")
    return build_model(settings["llm_model"], settings["llm_base_url"], settings["llm_api_key"],
                       timeout_sec=float(settings.get("llm_timeout_sec", 120)))


def _to_message_history(history: list[ChatTurn]) -> list:
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    msgs: list = []
    for turn in history:
        if turn.role == "user":
            msgs.append(ModelRequest(parts=[UserPromptPart(content=turn.content)]))
        else:
            msgs.append(ModelResponse(parts=[TextPart(content=turn.content)]))
    return msgs


def _has_in_chain(e: BaseException, types: tuple[type, ...]) -> bool:
    return any(isinstance(c, types) for c in cause_chain(e))


@app.post("/chat")
def chat(req: ChatRequest,  # sync def → threadpool: run_sync и sync psycopg/embedder внутри tool'а не блокируют event loop
         settings: dict = Depends(get_settings),
         registry: DocumentRegistryBackend = Depends(get_registry),
         storage: StorageBackend = Depends(get_storage),
         embedder: EmbedderBackend = Depends(get_search_embedder),
         search_log: SearchLogBackend = Depends(get_search_log),
         model=Depends(get_chat_model)) -> ChatResponse:
    import httpx  # доступен: get_chat_model уже гарантировал .[agent]
    create_agent, AgentDeps, _ = _import_agent()
    agent = create_agent(model)
    deps = AgentDeps(embedder=embedder, storage=storage, registry=registry,
                     top_k=settings["agent_top_k"], search_log=search_log)
    try:
        result = agent.run_sync(req.message,
                                message_history=_to_message_history(req.history),
                                deps=deps)
    except FileNotFoundError:
        # пустое хранилище: у HTTP нет exit-код контракта CLI (как GET /search → пустые results)
        return ChatResponse(answer="Хранилище пустое. Документов нет.", sources=[])
    except (StorageUnavailableError, StorageSchemaMissingError, EmbedServiceUnavailableError):
        # ДО connect-эвристики: цепочка psycopg/embed-ошибок тоже содержит ConnectionError,
        # иначе виноватым ошибочно назначался бы LM Studio (тот же порядок, что в cli ask)
        raise  # app-уровневые 503-хендлеры
    except Exception as e:
        if _has_in_chain(e, (ConnectionError, httpx.ConnectError, httpx.ConnectTimeout)):
            raise HTTPException(
                status_code=503,
                detail=f"LLM недоступна по адресу {settings['llm_base_url']} — запустите LM Studio",
            ) from e
        if _has_in_chain(e, (httpx.TimeoutException,)):
            raise HTTPException(
                status_code=504,
                detail=f"LLM не ответила за {settings.get('llm_timeout_sec', 120)} секунд",
            ) from e
        raise
    return ChatResponse(
        answer=result.output,
        sources=[ChatSource(file=Path(meta["source_file"]).name,
                            page=meta["page_number"],
                            headings=meta["headings"] or [],
                            score=float(score))
                 for meta, score in deps.sources],
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/search")
def search(q: str = Query(min_length=1),
           top_k: int | None = Query(default=None, ge=1, le=50),
           tag: list[str] = Query(default=[]),
           topic: str | None = None,
           settings: dict = Depends(get_settings),
           registry: DocumentRegistryBackend = Depends(get_registry),
           storage: StorageBackend = Depends(get_storage),
           embedder: EmbedderBackend = Depends(get_search_embedder),
           search_log: SearchLogBackend = Depends(get_search_log)) -> dict:
    k = top_k if top_k is not None else settings["top_k_results"]
    allowed = resolve_allowed_sources(registry, tags=tag, topic=topic)
    if allowed == set():  # фильтры заданы, но ничего не совпало
        return {"query": q, "results": []}
    try:
        found = run_search(q, embedder, storage, top_k=k, allowed_sources=allowed)
    except FileNotFoundError:  # пустое хранилище — у HTTP нет exit-код контракта CLI
        return {"query": q, "results": []}
    titles = registry.load()
    results = [{
        "text": chunk["text"], "score": float(score),
        "source_file": chunk["source_file"],
        "title": (titles.get(chunk["source_file"]) or {}).get("title"),
        "page_number": chunk["page_number"], "headings": chunk["headings"],
        "element_type": chunk["element_type"],
    } for chunk, score in found]
    if results:
        try:
            search_log.log(q, results[0]["score"])
        except Exception as e:  # отказ лога не роняет поиск (как в CLI)
            print(f"предупреждение: лог поиска не записан: {e}", file=sys.stderr)
    return {"query": q, "results": results}


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
    # Literal-дубликаты OCR_MODES/OCR_LANGS (core/parser.py) — FastAPI требует литеральный тип; менять синхронно
    ocr: Literal["auto", "on", "off"] = Form("auto"),
    ocr_lang: Literal["en", "ru"] = Form("en"),
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

    job_id = jobs.create(source_file, name, title, topic, tags, ocr=ocr, ocr_lang=ocr_lang)
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


def _document_card(source: str, entry: dict, storage: StorageBackend, jobs: JobBackend) -> dict:
    job = jobs.find_latest_by_source(source)
    return {
        "id": entry["id"], "source_file": source,
        "title": entry["title"], "topic": entry["topic"], "tags": entry["tags"],
        "added_at": entry["added_at"],
        "chunks": storage.count_by_source(source),
        "indexing": {"status": job["status"], "job_id": job["id"]} if job else None,
    }


@app.get("/documents")
def list_documents(registry: DocumentRegistryBackend = Depends(get_registry),
                   storage: StorageBackend = Depends(get_storage),
                   jobs: JobBackend = Depends(get_jobs)) -> list[dict]:
    cards = [_document_card(s, e, storage, jobs) for s, e in registry.load().items()]
    cards.sort(key=lambda c: c["added_at"], reverse=True)
    return cards


@app.get("/documents/{doc_id}")
def get_document(doc_id: str, registry: DocumentRegistryBackend = Depends(get_registry),
                 storage: StorageBackend = Depends(get_storage),
                 jobs: JobBackend = Depends(get_jobs)) -> dict:
    found = registry.get_by_id(doc_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Документ не найден")
    source, entry = found
    return _document_card(source, entry, storage, jobs)


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str,
                    settings: dict = Depends(get_settings),
                    registry: DocumentRegistryBackend = Depends(get_registry),
                    storage: StorageBackend = Depends(get_storage),
                    jobs: JobBackend = Depends(get_jobs)) -> dict:
    found = registry.get_by_id(doc_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Документ не найден")
    source, entry = found

    active = jobs.find_active_by_source(source)
    if active is not None:  # иначе воркер пересоздал бы документ после удаления
        raise HTTPException(status_code=409,
                            detail={"message": "Идёт индексация", "job_id": active["id"]})

    chunks = storage.count_by_source(source)
    registry.delete(source)          # FK-каскад сносит chunks
    storage.delete_by_source(source)  # идемпотентная страховка (как в cli delete)

    file_removed = False
    path = Path(source)
    if path.parent == Path(settings["uploads_dir"]).resolve() and path.exists():
        try:
            path.unlink()
            file_removed = True
        except OSError as e:  # истина — в БД; файл не критичен
            print(f"предупреждение: файл не удалён: {e}", file=sys.stderr)
    return {"deleted": entry["title"] or source, "chunks": chunks, "file_removed": file_removed}


# --- SPA-статика (этап 4-D) ------------------------------------------------
def mount_static(app: FastAPI, static_dir: str = "static") -> None:
    """Монтирует собранный фронтенд на '/' — строго ПОСЛЕ всех роутов.

    Папки нет (dev на хосте, герметичные тесты) — тихий no-op: API работает без UI.
    """
    if Path(static_dir).is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


mount_static(app)
