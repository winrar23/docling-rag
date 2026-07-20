# cli/commands.py
from pathlib import Path

import click

from docling_rag.cli.config_loader import load_config, ConfigError
from docling_rag.core.embedder import get_embedder
from docling_rag.core.errors import (
    EmbedServiceUnavailableError,
    StorageError,
    StorageSchemaMissingError,
    StorageUnavailableError,
    cause_chain,
)
from docling_rag.core.indexer import index_files
from docling_rag.core.parser import Parser, SUPPORTED_EXTENSIONS
from docling_rag.core.protocols import SearchLogBackend, StorageBackend
from docling_rag.core.search import resolve_allowed_sources, run_search
from docling_rag.storage.db_registry import DBRegistry
from docling_rag.storage.db_schema import init_schema
from docling_rag.storage.db_search_log import DBSearchLog
from docling_rag.storage.db_storage import DBStorage


def get_storage(cfg: dict) -> StorageBackend:
    return DBStorage(cfg["database_url"])


def get_search_log(cfg: dict) -> SearchLogBackend:
    return DBSearchLog(cfg["database_url"])


def _load_cfg(config: str | None) -> dict:
    try:
        return load_config(config or "config.yaml", required=config is not None)
    except ConfigError as e:
        raise click.ClickException(str(e)) from e


def _mask_dsn(dsn: str) -> str:
    """postgresql://user:СЕКРЕТ@host -> postgresql://user:***@host"""
    import re
    return re.sub(r"(://[^:/@]+):[^@]+@", r"\1:***@", dsn)


def _db_unavailable(cfg: dict, e: Exception) -> click.ClickException:
    return click.ClickException(
        f"PostgreSQL недоступен ({_mask_dsn(cfg['database_url'])}).\n"
        "Запустите: docker compose up -d postgres"
    )


def _schema_missing() -> click.ClickException:
    return click.ClickException("Схема БД не инициализирована. Выполните: docling-rag init")


def _embed_unavailable(e: Exception) -> click.ClickException:
    return click.ClickException(
        f"Сервис эмбеддингов недоступен: {e}\n"
        "Запустите: docker compose up -d embed"
    )


@click.group()
def main() -> None:
    """docling-rag — semantic search over technical documentation."""
    pass


@main.command()
@click.option("--config", default=None, help="Path to config.yaml")
def init(config: str | None) -> None:
    """Initialize database schema (idempotent)."""
    cfg = _load_cfg(config)
    try:
        init_schema(cfg["database_url"])
    except StorageUnavailableError as e:
        raise _db_unavailable(cfg, e) from e
    click.echo(f"Схема БД инициализирована: {_mask_dsn(cfg['database_url'])}")


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--config", default=None, help="Path to config.yaml")
@click.option("--title", default=None, help="Document title")
@click.option("--topic", default=None, help="Domain/topic of the document")
@click.option("--tag", "tags", multiple=True, help="Tag (repeatable: --tag arch --tag solid)")
def add(
    file_path: str,
    config: str | None,
    title: str | None,
    topic: str | None,
    tags: tuple[str, ...],
) -> None:
    """Add a document or directory to the index."""
    cfg = _load_cfg(config)
    path = Path(file_path)
    files = list(path.rglob("*.*")) if path.is_dir() else [path]
    files = [f for f in files if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        raise click.ClickException("Нет поддерживаемых файлов для индексации.")

    parser = Parser()
    embedder = get_embedder(cfg)
    storage = get_storage(cfg)
    registry = DBRegistry(cfg["database_url"])
    try:
        report = index_files(files, parser, embedder, storage, registry, cfg["embedding_model"],
                             chunk_max_tokens=cfg["chunk_max_tokens"], title=title, topic=topic, tags=tags)
    except StorageUnavailableError as e:
        raise _db_unavailable(cfg, e) from e
    except StorageSchemaMissingError as e:
        raise _schema_missing() from e
    except EmbedServiceUnavailableError as e:
        raise _embed_unavailable(e) from e
    for src, err in report.errors:
        click.echo(f"Ошибка при обработке {src}: {err}", err=True)
    click.echo(f"\nДобавлено {report.chunks_added} chunks из {report.files_ok} файлов.")
    if report.files_failed or report.chunks_added == 0:
        raise SystemExit(1)


@main.command()
@click.argument("query")
@click.option("--top-k", default=None, type=click.IntRange(min=1), help="Number of results")
@click.option("--config", default=None, help="Path to config.yaml")
@click.option("--tag", "filter_tags", multiple=True, help="Filter to docs with this tag (repeatable)")
@click.option("--topic", "filter_topic", default=None, help="Filter to docs with this topic (case-insensitive)")
def search(
    query: str,
    top_k: int | None,
    config: str | None,
    filter_tags: tuple[str, ...],
    filter_topic: str | None,
) -> None:
    """Perform semantic search over the documentation."""
    cfg = _load_cfg(config)
    k = top_k if top_k is not None else cfg["top_k_results"]
    storage = get_storage(cfg)
    registry = DBRegistry(cfg["database_url"])

    try:
        allowed_sources = resolve_allowed_sources(registry, tags=filter_tags, topic=filter_topic)
    except StorageUnavailableError as e:
        raise _db_unavailable(cfg, e) from e
    except StorageSchemaMissingError as e:
        raise _schema_missing() from e

    if allowed_sources == set():
        click.echo("Нет документов с такими тегами/темой.")
        return
    embedder = get_embedder(cfg)   # создаётся ПОСЛЕ раннего return — ленивый эмбеддер

    try:
        results = run_search(query, embedder, storage, top_k=k, allowed_sources=allowed_sources)
    except FileNotFoundError:
        raise click.ClickException("Хранилище пустое. Добавьте документы: docling-rag add <path>")
    except StorageError as e:
        raise click.ClickException(f"Хранилище повреждено: {e}. Переиндексируйте документы.") from e
    except StorageUnavailableError as e:
        raise _db_unavailable(cfg, e) from e
    except StorageSchemaMissingError as e:
        raise _schema_missing() from e
    except EmbedServiceUnavailableError as e:
        raise _embed_unavailable(e) from e

    if not results:
        click.echo("Ничего не найдено.")
        return

    click.echo(f"\nРезультаты для: \"{query}\"\n" + "-" * 60)
    for i, (meta, score) in enumerate(results, 1):
        source = Path(meta["source_file"]).name
        page = meta.get("page_number", "?")
        etype = meta.get("element_type", "text")
        headings = meta.get("headings", [])
        heading_str = " > ".join(headings) if headings else ""
        text_preview = meta["text"][:300].replace("\n", " ")
        click.echo(f"\n[{i}] score={score:.3f} | {source} | стр.{page} | {etype}")
        if heading_str:
            click.echo(f"    [{heading_str}]")
        click.echo(f"    {text_preview}...")

    try:
        get_search_log(cfg).log(query, results[0][1])
    except Exception as e:
        # Лог — побочная функция: его отказ не должен ронять уже выданный поиск
        click.echo(f"Предупреждение: не удалось записать лог: {e}", err=True)


@main.command("list")
@click.option("--config", default=None, help="Path to config.yaml")
def list_docs(config: str | None) -> None:
    """Show list of indexed documents."""
    cfg = _load_cfg(config)
    storage = get_storage(cfg)
    registry = DBRegistry(cfg["database_url"])
    try:
        _, metadata = storage.load()
        doc_index = registry.load()
    except FileNotFoundError:
        click.echo("Хранилище пустое. Документов нет.")
        return
    except StorageError as e:
        raise click.ClickException(f"Хранилище повреждено: {e}. Переиндексируйте документы.") from e
    except StorageUnavailableError as e:
        raise _db_unavailable(cfg, e) from e
    except StorageSchemaMissingError as e:
        raise _schema_missing() from e

    sources: dict[str, int] = {}
    for m in metadata:
        src = m["source_file"]
        sources[src] = sources.get(src, 0) + 1

    click.echo(f"\nПроиндексировано документов: {len(sources)}\n" + "-" * 60)
    for src, count in sorted(sources.items()):
        entry = doc_index.get(src, {})
        title = entry.get("title") or "—"
        topic = entry.get("topic") or "—"
        tags_str = "[" + ", ".join(entry.get("tags", [])) + "]" if entry.get("tags") else "[]"
        title_display = (title[:28] + "...") if len(title) > 31 else title
        click.echo(
            f"  {Path(src).name:35s} {count:4d} chunks"
            f" | {title_display:31s} | {topic:18s} | {tags_str}"
        )


@main.command()
@click.argument("source")
@click.option("--config", default=None, help="Path to config.yaml")
def delete(source: str, config: str | None) -> None:
    """Delete a document and all its chunks from the index."""
    cfg = _load_cfg(config)
    p = Path(source)
    key = str(p.resolve()) if p.exists() else source  # осиротевшие записи удаляемы по строке-ключу
    storage = get_storage(cfg)
    registry = DBRegistry(cfg["database_url"])
    try:
        entry = registry.get(key)
        n = storage.count_by_source(key)
        if entry is None and n == 0:
            raise click.ClickException(
                f"Документ не найден: {key}\nТочные пути покажет: docling-rag list"
            )
        registry.delete(key)           # в pg каскад сносит chunks
        storage.delete_by_source(key)  # идемпотентная страховка (и контракт для fake)
    except StorageUnavailableError as e:
        raise _db_unavailable(cfg, e) from e
    except StorageSchemaMissingError as e:
        raise _schema_missing() from e
    title = (entry or {}).get("title") or key
    click.echo(f"Удалено: {title} ({n} chunks)")


def _is_connection_error(e: BaseException) -> bool:
    """Walk the cause chain: httpx/openai wrap ConnectError several levels deep."""
    try:
        import httpx
        conn_types: tuple[type, ...] = (ConnectionError, httpx.ConnectError, httpx.ConnectTimeout)
    except ImportError:
        conn_types = (ConnectionError,)
    return any(isinstance(cur, conn_types) for cur in cause_chain(e))


def _import_agent_module():
    """Import core.agent module. Separated for testability."""
    from docling_rag.core.agent import create_agent, AgentDeps, build_lmstudio_model  # noqa: F401
    return create_agent, AgentDeps, build_lmstudio_model


def _create_and_run_agent(question: str, cfg: dict, top_k: int) -> str:
    """Create agent and run synchronously. Separated for testability."""
    create_agent, AgentDeps, build_lmstudio_model = _import_agent_module()
    agent = create_agent(build_lmstudio_model(cfg["llm_model"], cfg["llm_base_url"], cfg["llm_api_key"]))
    embedder = get_embedder(cfg)
    storage = get_storage(cfg)
    registry = DBRegistry(cfg["database_url"])
    deps = AgentDeps(embedder=embedder, storage=storage, registry=registry, top_k=top_k,
                     search_log=DBSearchLog(cfg["database_url"]))
    result = agent.run_sync(question, deps=deps)
    return result.output


@main.command()
@click.argument("question")
@click.option("--config", default=None, help="Path to config.yaml")
@click.option("--top-k", default=None, type=click.IntRange(min=1), help="Number of search results for agent")
def ask(question: str, config: str | None, top_k: int | None) -> None:
    """Ask a question — agent synthesizes answer from indexed documents."""
    cfg = _load_cfg(config)

    if not cfg["agent_enabled"]:
        raise click.ClickException(
            "Агент отключён. Включите в config.yaml:\n"
            "  agent_enabled: true\n"
            "  llm_model: <ваша модель в LM Studio>"
        )

    try:
        _import_agent_module()
    except ImportError:
        raise click.ClickException(
            "pydantic-ai не установлен. Установите:\n"
            "  uv pip install -e '.[agent]'"
        )

    k = top_k if top_k is not None else cfg["agent_top_k"]

    try:
        answer = _create_and_run_agent(question, cfg, k)
        click.echo(answer)
    except FileNotFoundError:
        raise click.ClickException("Хранилище пустое. Добавьте документы: docling-rag add <path>")
    except StorageError as e:
        raise click.ClickException(f"Хранилище повреждено: {e}. Переиндексируйте документы.") from e
    except StorageUnavailableError as e:
        raise _db_unavailable(cfg, e) from e
    except StorageSchemaMissingError as e:
        raise _schema_missing() from e
    except EmbedServiceUnavailableError as e:
        # ДО generic-хендлера: иначе _is_connection_error() находит вложенный
        # httpx.ConnectError и ошибочно винит LM Studio, хотя лежит embed-сервис.
        raise _embed_unavailable(e) from e
    except Exception as e:
        if _is_connection_error(e):
            raise click.ClickException(
                f"Не удалось подключиться к LLM по адресу {cfg['llm_base_url']}.\n"
                "Убедитесь, что LM Studio запущен."
            ) from e
        raise click.ClickException(f"Ошибка агента: {e}") from e
