# cli/commands.py
from datetime import datetime
from pathlib import Path

import click

from docling_rag.cli.config_loader import load_config, ConfigError
from docling_rag.core.embedder import Embedder
from docling_rag.core.errors import StorageError
from docling_rag.core.indexer import index_files
from docling_rag.core.parser import Parser, SUPPORTED_EXTENSIONS
from docling_rag.core.protocols import StorageBackend
from docling_rag.core.search import resolve_allowed_sources, run_search
from docling_rag.storage.doc_registry import DocRegistry
from docling_rag.storage.file_storage import FileStorage


def get_storage(data_dir: str) -> StorageBackend:
    return FileStorage(data_dir=Path(data_dir))


def _load_cfg(config: str | None) -> dict:
    try:
        return load_config(config or "config.yaml", required=config is not None)
    except ConfigError as e:
        raise click.ClickException(str(e)) from e


@click.group()
def main() -> None:
    """docling-rag — semantic search over technical documentation."""
    pass


@main.command()
@click.option("--data-dir", default=None, help="Storage directory")
@click.option("--config", default=None, help="Path to config.yaml")
def init(data_dir: str | None, config: str | None) -> None:
    """Initialize storage."""
    cfg = _load_cfg(config)
    data_dir = data_dir or cfg["data_dir"]
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    Path(cfg["log_file"]).parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"Инициализировано хранилище: {path.resolve()}")


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--data-dir", default=None, help="Storage directory")
@click.option("--config", default=None, help="Path to config.yaml")
@click.option("--title", default=None, help="Document title")
@click.option("--topic", default=None, help="Domain/topic of the document")
@click.option("--tag", "tags", multiple=True, help="Tag (repeatable: --tag arch --tag solid)")
def add(
    file_path: str,
    data_dir: str | None,
    config: str | None,
    title: str | None,
    topic: str | None,
    tags: tuple[str, ...],
) -> None:
    """Add a document or directory to the index."""
    cfg = _load_cfg(config)
    data_dir = data_dir or cfg["data_dir"]
    path = Path(file_path)
    files = list(path.rglob("*.*")) if path.is_dir() else [path]
    files = [f for f in files if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        raise click.ClickException("Нет поддерживаемых файлов для индексации.")

    parser = Parser()
    embedder = Embedder(model_name=cfg["embedding_model"])
    storage = get_storage(data_dir)
    registry = DocRegistry(data_dir=data_dir)
    report = index_files(files, parser, embedder, storage, registry,
                         cfg["embedding_model"], title=title, topic=topic, tags=tags)
    for src, err in report.errors:
        click.echo(f"Ошибка при обработке {src}: {err}", err=True)
    click.echo(f"\nДобавлено {report.chunks_added} chunks из {report.files_ok} файлов.")
    if report.files_failed or report.chunks_added == 0:
        raise SystemExit(1)


@main.command()
@click.argument("query")
@click.option("--data-dir", default=None, help="Storage directory")
@click.option("--top-k", default=None, type=click.IntRange(min=1), help="Number of results")
@click.option("--config", default=None, help="Path to config.yaml")
@click.option("--tag", "filter_tags", multiple=True, help="Filter to docs with this tag (repeatable)")
@click.option("--topic", "filter_topic", default=None, help="Filter to docs with this topic (case-insensitive)")
def search(
    query: str,
    data_dir: str | None,
    top_k: int | None,
    config: str | None,
    filter_tags: tuple[str, ...],
    filter_topic: str | None,
) -> None:
    """Perform semantic search over the documentation."""
    cfg = _load_cfg(config)
    data_dir = data_dir or cfg["data_dir"]
    k = top_k if top_k is not None else cfg["top_k_results"]
    storage = get_storage(data_dir)
    registry = DocRegistry(data_dir=data_dir)

    allowed_sources = resolve_allowed_sources(registry, tags=filter_tags, topic=filter_topic)
    if allowed_sources == set():
        click.echo("Нет документов с такими тегами/темой.")
        return
    embedder = Embedder(model_name=cfg["embedding_model"])   # constructed AFTER the early return — lazy Embedder

    try:
        results = run_search(query, embedder, storage, top_k=k, allowed_sources=allowed_sources)
    except FileNotFoundError:
        raise click.ClickException("Хранилище пустое. Добавьте документы: docling-rag add <path>")
    except StorageError as e:
        raise click.ClickException(f"Хранилище повреждено: {e}. Переиндексируйте документы.") from e

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
        _log_search(cfg["log_file"], query, results[0][1])
    except OSError as e:
        click.echo(f"Предупреждение: не удалось записать лог: {e}", err=True)


@main.command("list")
@click.option("--data-dir", default=None, help="Storage directory")
@click.option("--config", default=None, help="Path to config.yaml")
def list_docs(data_dir: str | None, config: str | None) -> None:
    """Show list of indexed documents."""
    cfg = _load_cfg(config)
    data_dir = data_dir or cfg["data_dir"]
    storage = get_storage(data_dir)
    registry = DocRegistry(data_dir=data_dir)
    try:
        _, metadata = storage.load()
    except FileNotFoundError:
        click.echo("Хранилище пустое. Документов нет.")
        return
    except StorageError as e:
        raise click.ClickException(f"Хранилище повреждено: {e}. Переиндексируйте документы.") from e

    sources: dict[str, int] = {}
    for m in metadata:
        src = m["source_file"]
        sources[src] = sources.get(src, 0) + 1

    doc_index = registry.load()

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


def _is_connection_error(e: BaseException) -> bool:
    """Walk the cause chain: httpx/openai wrap ConnectError several levels deep."""
    try:
        import httpx
        conn_types: tuple[type, ...] = (ConnectionError, httpx.ConnectError, httpx.ConnectTimeout)
    except ImportError:
        conn_types = (ConnectionError,)
    seen: set[int] = set()
    cur: BaseException | None = e
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, conn_types):
            return True
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    return False


def _import_agent_module():
    """Import core.agent module. Separated for testability."""
    from docling_rag.core.agent import create_agent, AgentDeps  # noqa: F401
    return create_agent, AgentDeps


def _create_and_run_agent(question: str, cfg: dict, data_dir: str, top_k: int) -> str:
    """Create agent and run synchronously. Separated for testability."""
    create_agent, AgentDeps = _import_agent_module()
    agent = create_agent(
        model_name=cfg["llm_model"],
        base_url=cfg["llm_base_url"],
        api_key=cfg["llm_api_key"],
    )
    embedder = Embedder(model_name=cfg["embedding_model"])
    storage = get_storage(data_dir)
    registry = DocRegistry(data_dir=data_dir)
    deps = AgentDeps(embedder=embedder, storage=storage, registry=registry, top_k=top_k)
    result = agent.run_sync(question, deps=deps)
    return result.output


@main.command()
@click.argument("question")
@click.option("--data-dir", default=None, help="Storage directory")
@click.option("--config", default=None, help="Path to config.yaml")
@click.option("--top-k", default=None, type=click.IntRange(min=1), help="Number of search results for agent")
def ask(question: str, data_dir: str | None, config: str | None, top_k: int | None) -> None:
    """Ask a question — agent synthesizes answer from indexed documents."""
    cfg = _load_cfg(config)
    data_dir = data_dir or cfg["data_dir"]

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
        answer = _create_and_run_agent(question, cfg, data_dir, k)
        click.echo(answer)
    except FileNotFoundError:
        raise click.ClickException("Хранилище пустое. Добавьте документы: docling-rag add <path>")
    except StorageError as e:
        raise click.ClickException(f"Хранилище повреждено: {e}. Переиндексируйте документы.") from e
    except Exception as e:
        if _is_connection_error(e):
            raise click.ClickException(
                f"Не удалось подключиться к LLM по адресу {cfg['llm_base_url']}.\n"
                "Убедитесь, что LM Studio запущен."
            ) from e
        raise click.ClickException(f"Ошибка агента: {e}") from e


def _log_search(log_file: str, query: str, top_score: float) -> None:
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | score={top_score:.3f} | {query}\n")
