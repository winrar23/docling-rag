"""Точка входа воркера в контейнере: docling-rag-worker (entrypoint 'worker')."""
from docling_rag.cli.config_loader import load_config
from docling_rag.core.embedder import get_embedder
from docling_rag.core.parser import Parser
from docling_rag.storage.db_jobs import DBJobs
from docling_rag.storage.db_registry import DBRegistry
from docling_rag.storage.db_storage import DBStorage
from docling_rag.worker.runner import WorkerDeps, run_loop


def build_deps(cfg: dict) -> WorkerDeps:
    dsn = cfg["database_url"]
    return WorkerDeps(
        parser=Parser(),
        embedder=get_embedder(cfg),
        storage=DBStorage(dsn),
        registry=DBRegistry(dsn),
        embedding_model=cfg["embedding_model"],
        chunk_max_tokens=cfg.get("chunk_max_tokens", 512),
    )


def main() -> None:
    cfg = load_config()
    jobs = DBJobs(cfg["database_url"])
    print("worker: запущен, слушаю очередь jobs", flush=True)
    run_loop(jobs, build_deps(cfg))


if __name__ == "__main__":
    main()
