# cli/commands.py
from datetime import datetime
from pathlib import Path

import click

from cli.config_loader import load_config
from core.chunker import chunk_document
from core.embedder import Embedder
from core.parser import Parser
from core.search import run_search
from storage.doc_registry import DocRegistry
from storage.file_storage import FileStorage


def get_storage(data_dir: str) -> FileStorage:
    return FileStorage(data_dir=Path(data_dir))


@click.group()
def main() -> None:
    """docling-rag — semantic search over technical documentation."""
    pass


@main.command()
@click.option("--data-dir", default="data", help="Storage directory")
@click.option("--config", default="config.yaml", help="Path to config.yaml")
def init(data_dir: str, config: str) -> None:
    """Initialize storage."""
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config)
    Path(cfg["log_file"]).parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"Инициализировано хранилище: {path.resolve()}")


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--data-dir", default="data", help="Storage directory")
@click.option("--config", default="config.yaml", help="Path to config.yaml")
@click.option("--title", default=None, help="Document title")
@click.option("--topic", default=None, help="Domain/topic of the document")
@click.option("--tag", "tags", multiple=True, help="Tag (repeatable: --tag arch --tag solid)")
def add(file_path: str, data_dir: str, config: str, title: str | None, topic: str | None, tags: tuple[str, ...]) -> None:
    """Add a document or directory to the index."""
    cfg = load_config(config)
    path = Path(file_path)
    files = list(path.rglob("*.*")) if path.is_dir() else [path]
    supported = {".pdf", ".docx", ".md", ".txt"}
    files = [f for f in files if f.suffix.lower() in supported]

    if not files:
        click.echo("Нет поддерживаемых файлов для индексации.")
        return

    parser = Parser()
    embedder = Embedder(model_name=cfg["embedding_model"])
    storage = get_storage(data_dir)
    registry = DocRegistry(data_dir=data_dir)

    total_chunks = 0
    for file in files:
        click.echo(f"Обрабатываю: {file.name} ...", nl=False)
        try:
            doc = parser.parse(file)
            chunks = chunk_document(
                doc,
                source_file=str(file),
                embedding_model=cfg["embedding_model"],
            )
            if not chunks:
                click.echo(" (пустой документ, пропускаю)")
                continue
            texts = [c.context_text for c in chunks]
            embeddings = embedder.embed(texts)
            storage.append(chunks, embeddings)
            registry.upsert(str(file), title=title, topic=topic, tags=list(tags))
            total_chunks += len(chunks)
            click.echo(f" {len(chunks)} chunks")
        except Exception as e:
            click.echo("")
            click.echo(f"Ошибка при обработке {file}: {e}", err=True)
            continue

    click.echo(f"\nДобавлено {total_chunks} chunks из {len(files)} файлов.")


@main.command()
@click.argument("query")
@click.option("--data-dir", default="data", help="Storage directory")
@click.option("--top-k", default=None, type=int, help="Number of results")
@click.option("--config", default="config.yaml", help="Path to config.yaml")
@click.option("--tag", "filter_tags", multiple=True, help="Filter to docs with this tag (repeatable)")
@click.option("--topic", "filter_topic", default=None, help="Filter to docs with this topic (case-insensitive)")
def search(
    query: str,
    data_dir: str,
    top_k: int | None,
    config: str,
    filter_tags: tuple[str, ...],
    filter_topic: str | None,
) -> None:
    """Perform semantic search over the documentation."""
    cfg = load_config(config)
    k = top_k if top_k is not None else cfg["top_k_results"]
    embedder = Embedder(model_name=cfg["embedding_model"])
    storage = get_storage(data_dir)
    registry = DocRegistry(data_dir=data_dir)

    allowed_sources: set[str] | None = None
    if filter_tags or filter_topic:
        doc_index = registry.load()
        matched = []
        for src, entry in doc_index.items():
            tag_ok = all(t in entry.get("tags", []) for t in filter_tags) if filter_tags else True
            topic_ok = (
                (entry.get("topic") or "").lower() == filter_topic.lower()
                if filter_topic else True
            )
            if tag_ok and topic_ok:
                matched.append(src)
        if not matched:
            click.echo("Нет документов с такими тегами/темой.")
            return
        allowed_sources = set(matched)

    try:
        results = run_search(query, embedder, storage, top_k=k, allowed_sources=allowed_sources)
    except FileNotFoundError:
        click.echo("Хранилище пустое. Добавьте документы: docling-rag add <path>")
        return

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
        _log_search(cfg["log_file"], query, results[0][1] if results else 0.0)
    except OSError as e:
        click.echo(f"Предупреждение: не удалось записать лог: {e}", err=True)


@main.command("list")
@click.option("--data-dir", default="data", help="Storage directory")
def list_docs(data_dir: str) -> None:
    """Show list of indexed documents."""
    storage = get_storage(data_dir)
    registry = DocRegistry(data_dir=data_dir)
    try:
        _, metadata = storage.load()
    except FileNotFoundError:
        click.echo("Хранилище пустое. Документов нет.")
        return

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


def _import_agent_module():
    """Import core.agent module. Separated for testability."""
    from core.agent import create_agent, AgentDeps  # noqa: F401
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
@click.option("--data-dir", default="data", help="Storage directory")
@click.option("--config", default="config.yaml", help="Path to config.yaml")
@click.option("--top-k", default=None, type=int, help="Number of search results for agent")
def ask(question: str, data_dir: str, config: str, top_k: int | None) -> None:
    """Ask a question — agent synthesizes answer from indexed documents."""
    cfg = load_config(config)

    if not cfg.get("agent_enabled", False):
        click.echo(
            "Агент отключён. Включите в config.yaml:\n"
            "  agent_enabled: true\n"
            "  llm_model: <ваша модель в LM Studio>"
        )
        return

    try:
        _import_agent_module()
    except ImportError:
        click.echo(
            "pydantic-ai не установлен. Установите:\n"
            "  uv pip install -e '.[agent]'"
        )
        return

    k = top_k if top_k is not None else cfg.get("agent_top_k", 5)

    try:
        answer = _create_and_run_agent(question, cfg, data_dir, k)
        click.echo(answer)
    except FileNotFoundError:
        click.echo("Хранилище пустое. Добавьте документы: docling-rag add <path>")
    except Exception as e:
        exc_type = type(e).__name__
        if isinstance(e, ConnectionError) or "ConnectError" in exc_type or "ConnectionRefused" in exc_type:
            click.echo(
                f"Не удалось подключиться к LLM по адресу {cfg['llm_base_url']}.\n"
                "Убедитесь, что LM Studio запущен."
            )
        else:
            click.echo(f"Ошибка агента: {e}", err=True)


def _log_search(log_file: str, query: str, top_score: float) -> None:
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | score={top_score:.3f} | {query}\n")
