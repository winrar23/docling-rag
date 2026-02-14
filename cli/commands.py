# cli/commands.py
from datetime import datetime
from pathlib import Path

import click

from cli.config_loader import load_config
from core.chunker import chunk_elements
from core.embedder import Embedder
from core.parser import Parser
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
            elements = parser.parse(file)
            chunks = chunk_elements(
                elements,
                source_file=str(file),
                chunk_size=cfg["chunk_size"],
                overlap=cfg["chunk_overlap"],
            )
            if not chunks:
                click.echo(" (пустой документ, пропускаю)")
                continue
            texts = [c.text for c in chunks]
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
def search(query: str, data_dir: str, top_k: int | None, config: str) -> None:
    """Perform semantic search over the documentation."""
    cfg = load_config(config)
    k = top_k if top_k is not None else cfg["top_k_results"]
    embedder = Embedder(model_name=cfg["embedding_model"])
    storage = get_storage(data_dir)

    try:
        query_emb = embedder.embed([query])[0]
        results = storage.search(query_embedding=query_emb, top_k=k)
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
        text_preview = meta["text"][:300].replace("\n", " ")
        click.echo(
            f"\n[{i}] score={score:.3f} | {source} | стр.{page} | {etype}\n"
            f"    {text_preview}..."
        )

    try:
        _log_search(cfg["log_file"], query, results[0][1] if results else 0.0)
    except OSError as e:
        click.echo(f"Предупреждение: не удалось записать лог: {e}", err=True)


@main.command("list")
@click.option("--data-dir", default="data", help="Storage directory")
def list_docs(data_dir: str) -> None:
    """Show list of indexed documents."""
    storage = get_storage(data_dir)
    try:
        _, metadata = storage.load()
    except FileNotFoundError:
        click.echo("Хранилище пустое. Документов нет.")
        return

    sources = {}
    for m in metadata:
        src = m["source_file"]
        sources[src] = sources.get(src, 0) + 1

    click.echo(f"\nПроиндексировано документов: {len(sources)}\n" + "-" * 60)
    for src, count in sorted(sources.items()):
        click.echo(f"  {Path(src).name:40s} {count:4d} chunks  ({src})")


def _log_search(log_file: str, query: str, top_score: float) -> None:
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | score={top_score:.3f} | {query}\n")
