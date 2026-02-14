# Document Metadata & Categorization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add document-level metadata (title, topic, tags) stored in `doc_index.json`, expose it via `--title`/`--topic`/`--tag` flags on `add`, filter search with `--tag`/`--topic`, and enrich `list` output.

**Architecture:** New `DocRegistry` class in `storage/doc_registry.py` manages `doc_index.json` with atomic writes. `FileStorage.search` gains an `allowed_sources` pre-filter. CLI commands wire everything together. No changes to chunk schema or embeddings.

**Tech Stack:** Python 3.10+, Click (`multiple=True` for `--tag`), NumPy, json, pathlib — all already in project.

---

## Task 1: DocRegistry — upsert and load

**Files:**
- Create: `storage/doc_registry.py`
- Create: `tests/storage/test_doc_registry.py`

**Step 1: Write the failing tests**

Create `tests/storage/test_doc_registry.py`:

```python
import json
import pytest
from pathlib import Path
from storage.doc_registry import DocRegistry


@pytest.fixture
def registry(tmp_path):
    return DocRegistry(data_dir=tmp_path)


def test_load_returns_empty_when_no_file(registry):
    assert registry.load() == {}


def test_upsert_creates_entry(registry):
    registry.upsert("docs/book.pdf", title="Clean Code", topic="software", tags=["refactoring"])
    data = registry.load()
    assert "docs/book.pdf" in data
    entry = data["docs/book.pdf"]
    assert entry["title"] == "Clean Code"
    assert entry["topic"] == "software"
    assert entry["tags"] == ["refactoring"]
    assert "added_at" in entry


def test_upsert_preserves_added_at_on_second_call(registry):
    registry.upsert("a.pdf", title="First", topic=None, tags=[])
    first_added_at = registry.load()["a.pdf"]["added_at"]

    registry.upsert("a.pdf", title="Updated", topic="new", tags=["x"])
    second = registry.load()["a.pdf"]

    assert second["added_at"] == first_added_at  # preserved
    assert second["title"] == "Updated"           # updated


def test_upsert_none_title_and_topic(registry):
    registry.upsert("b.pdf", title=None, topic=None, tags=[])
    entry = registry.load()["b.pdf"]
    assert entry["title"] is None
    assert entry["topic"] is None
    assert entry["tags"] == []


def test_upsert_multiple_docs(registry):
    registry.upsert("a.pdf", title="A", topic="t1", tags=["x"])
    registry.upsert("b.pdf", title="B", topic="t2", tags=["y"])
    data = registry.load()
    assert len(data) == 2
    assert data["a.pdf"]["title"] == "A"
    assert data["b.pdf"]["title"] == "B"
```

**Step 2: Run to verify they fail**

```bash
python3 -m pytest tests/storage/test_doc_registry.py -v
```
Expected: `ModuleNotFoundError: No module named 'storage.doc_registry'`

**Step 3: Implement `storage/doc_registry.py`**

```python
# storage/doc_registry.py
import json
import os
from datetime import datetime
from pathlib import Path


class DocRegistry:
    """
    Document-level metadata store: doc_index.json
    Keys are source_file paths (same as in metadata.json chunks).
    Implements DocumentRegistryBackend protocol.
    """

    INDEX_FILE = "doc_index.json"

    def __init__(self, data_dir: str | Path = "data") -> None:
        self._dir = Path(data_dir)

    def _index_path(self) -> Path:
        return self._dir / self.INDEX_FILE

    def _atomic_save(self, data: dict) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._dir / "doc_index.tmp.json"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._index_path())
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def load(self) -> dict[str, dict]:
        """Return full index. Returns {} if file does not exist."""
        path = self._index_path()
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def upsert(
        self,
        source_file: str,
        title: str | None,
        topic: str | None,
        tags: list[str],
    ) -> None:
        """Add or update entry. Preserves added_at if entry already exists."""
        data = self.load()
        existing = data.get(source_file, {})
        data[source_file] = {
            "title": title,
            "topic": topic,
            "tags": tags,
            "added_at": existing.get("added_at", datetime.now().isoformat(timespec="seconds")),
        }
        self._atomic_save(data)

    def get(self, source_file: str) -> dict | None:
        return self.load().get(source_file)

    def delete(self, source_file: str) -> None:
        data = self.load()
        if source_file in data:
            del data[source_file]
            self._atomic_save(data)
```

**Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/storage/test_doc_registry.py -v
```
Expected: 5 tests PASS

**Step 5: Commit**

```bash
git add storage/doc_registry.py tests/storage/test_doc_registry.py
git commit -m "feat: add DocRegistry for document-level metadata storage"
```

---

## Task 2: DocRegistry — get and delete

**Files:**
- Modify: `tests/storage/test_doc_registry.py`

**Step 1: Add failing tests for get and delete**

Append to `tests/storage/test_doc_registry.py`:

```python
def test_get_returns_entry(registry):
    registry.upsert("c.pdf", title="C", topic="t", tags=["a"])
    entry = registry.get("c.pdf")
    assert entry is not None
    assert entry["title"] == "C"


def test_get_returns_none_for_missing(registry):
    assert registry.get("missing.pdf") is None


def test_delete_removes_entry(registry):
    registry.upsert("d.pdf", title="D", topic=None, tags=[])
    registry.upsert("e.pdf", title="E", topic=None, tags=[])
    registry.delete("d.pdf")
    data = registry.load()
    assert "d.pdf" not in data
    assert "e.pdf" in data


def test_delete_nonexistent_is_noop(registry):
    registry.upsert("f.pdf", title="F", topic=None, tags=[])
    registry.delete("nonexistent.pdf")  # must not raise
    assert "f.pdf" in registry.load()


def test_delete_on_empty_is_noop(registry):
    registry.delete("anything.pdf")  # must not raise
```

**Step 2: Run to verify they fail**

```bash
python3 -m pytest tests/storage/test_doc_registry.py -v
```
Expected: 5 new tests FAIL (`get` and `delete` not implemented)

> Note: `get` and `delete` are already implemented in Task 1. These tests should actually PASS immediately if Task 1 is done correctly. If they do — great, proceed to commit.

**Step 3: Run all DocRegistry tests**

```bash
python3 -m pytest tests/storage/test_doc_registry.py -v
```
Expected: 10 tests PASS

**Step 4: Commit**

```bash
git add tests/storage/test_doc_registry.py
git commit -m "test: add get/delete coverage for DocRegistry"
```

---

## Task 3: DocumentRegistryBackend Protocol

**Files:**
- Modify: `core/storage.py`

**Step 1: Read the current file**

Read `core/storage.py` — it currently has only `StorageBackend` Protocol.

**Step 2: Add the new Protocol**

Append to `core/storage.py` after the existing `StorageBackend` class:

```python
class DocumentRegistryBackend(Protocol):
    """
    Protocol for document-level metadata registries. Implementations:
    - storage.doc_registry.DocRegistry (MVP)
    - storage.db_registry.DBRegistry (Phase 2, PostgreSQL)
    """

    def upsert(
        self,
        source_file: str,
        title: str | None,
        topic: str | None,
        tags: list[str],
    ) -> None:
        """Add or update document entry. Preserves added_at on re-index."""
        ...

    def delete(self, source_file: str) -> None:
        """Remove document entry."""
        ...

    def get(self, source_file: str) -> dict | None:
        """Return entry for source_file or None."""
        ...

    def load(self) -> dict[str, dict]:
        """Return full index as {source_file: {title, topic, tags, added_at}}."""
        ...
```

**Step 3: Run full unit test suite to verify nothing is broken**

```bash
python3 -m pytest tests/ -m "not integration and not slow" -v
```
Expected: all existing tests PASS

**Step 4: Commit**

```bash
git add core/storage.py
git commit -m "feat: add DocumentRegistryBackend protocol to core/storage.py"
```

---

## Task 4: FileStorage.search — allowed_sources filter

**Files:**
- Modify: `storage/file_storage.py:104-121`
- Modify: `tests/storage/test_file_storage.py`

**Step 1: Write failing tests**

Append to `tests/storage/test_file_storage.py`:

```python
def test_search_with_allowed_sources_filters_results(storage):
    """search() only returns chunks from allowed_sources."""
    chunks_a = make_chunks(3, source="alpha.pdf")
    chunks_b = make_chunks(3, source="beta.pdf")
    all_chunks = chunks_a + chunks_b
    for i, c in enumerate(all_chunks):
        c.chunk_id = i
    emb = make_embeddings(6)
    storage.save(all_chunks, emb)

    query = make_embeddings(1)[0]
    results = storage.search(query_embedding=query, top_k=6, allowed_sources={"alpha.pdf"})

    assert len(results) <= 3
    assert all(meta["source_file"] == "alpha.pdf" for meta, _ in results)


def test_search_with_empty_allowed_sources_returns_empty(storage):
    """Empty allowed_sources set → no results."""
    chunks = make_chunks(3)
    emb = make_embeddings(3)
    storage.save(chunks, emb)

    query = make_embeddings(1)[0]
    results = storage.search(query_embedding=query, top_k=3, allowed_sources=set())
    assert results == []


def test_search_with_none_allowed_sources_searches_all(storage):
    """allowed_sources=None (default) → searches all docs, unchanged behaviour."""
    chunks_a = make_chunks(2, source="a.pdf")
    chunks_b = make_chunks(2, source="b.pdf")
    all_chunks = chunks_a + chunks_b
    for i, c in enumerate(all_chunks):
        c.chunk_id = i
    emb = make_embeddings(4)
    storage.save(all_chunks, emb)

    query = make_embeddings(1)[0]
    results = storage.search(query_embedding=query, top_k=4, allowed_sources=None)
    assert len(results) == 4
```

**Step 2: Run to verify they fail**

```bash
python3 -m pytest tests/storage/test_file_storage.py::test_search_with_allowed_sources_filters_results tests/storage/test_file_storage.py::test_search_with_empty_allowed_sources_returns_empty tests/storage/test_file_storage.py::test_search_with_none_allowed_sources_searches_all -v
```
Expected: FAIL — `search()` does not accept `allowed_sources` keyword argument.

**Step 3: Update `FileStorage.search` signature and logic**

In `storage/file_storage.py`, replace the `search` method (lines 104–121):

```python
def search(
    self,
    query_embedding: np.ndarray,
    top_k: int = 5,
    allowed_sources: set[str] | None = None,
) -> list[tuple[dict, float]]:
    """
    Linear cosine similarity search.
    query_embedding must be L2-normalized.
    Since stored embeddings are normalized, cosine_sim = dot(query, vec).
    allowed_sources: if set, only chunks from those source_files are searched.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    embeddings, metadata = self.load()
    if embeddings.shape[1] != query_embedding.shape[0]:
        raise ValueError(
            f"Dimension mismatch: stored={embeddings.shape[1]}, query={query_embedding.shape[0]}"
        )
    if allowed_sources is not None:
        mask = [i for i, m in enumerate(metadata) if m["source_file"] in allowed_sources]
        if not mask:
            return []
        embeddings = embeddings[mask]
        metadata = [metadata[i] for i in mask]
    scores: np.ndarray = embeddings @ query_embedding  # (N,)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(metadata[i], float(scores[i])) for i in top_indices]
```

**Step 4: Run all storage tests**

```bash
python3 -m pytest tests/storage/ -v
```
Expected: all tests PASS

**Step 5: Commit**

```bash
git add storage/file_storage.py tests/storage/test_file_storage.py
git commit -m "feat: add allowed_sources filter to FileStorage.search"
```

---

## Task 5: CLI add — --title, --topic, --tag flags

**Files:**
- Modify: `cli/commands.py`
- Modify: `tests/test_cli.py`

**Step 1: Write failing tests**

Append to `tests/test_cli.py`:

```python
def test_add_command_calls_doc_registry_upsert(runner, tmp_path):
    """add with --title/--topic/--tag calls DocRegistry.upsert with correct args."""
    test_doc = tmp_path / "book.md"
    test_doc.write_text("# Book\n\nContent here.\n")

    with (
        patch("cli.commands.Parser") as MockParser,
        patch("cli.commands.Embedder") as MockEmbedder,
        patch("cli.commands.FileStorage") as MockStorage,
        patch("cli.commands.chunk_elements") as MockChunker,
        patch("cli.commands.DocRegistry") as MockRegistry,
    ):
        mock_chunk = MagicMock()
        mock_chunk.text = "Content here."
        MockChunker.return_value = [mock_chunk]
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)

        result = runner.invoke(main, [
            "add", str(test_doc),
            "--data-dir", str(tmp_path),
            "--title", "My Book",
            "--topic", "architecture",
            "--tag", "arch",
            "--tag", "solid",
        ])

    assert result.exit_code == 0
    MockRegistry.return_value.upsert.assert_called_once_with(
        str(test_doc),
        title="My Book",
        topic="architecture",
        tags=["arch", "solid"],
    )


def test_add_command_without_metadata_flags_upserts_nones(runner, tmp_path):
    """add without metadata flags calls upsert with None/empty."""
    test_doc = tmp_path / "plain.md"
    test_doc.write_text("# Plain\n\nText.\n")

    with (
        patch("cli.commands.Parser"),
        patch("cli.commands.Embedder") as MockEmbedder,
        patch("cli.commands.FileStorage"),
        patch("cli.commands.chunk_elements") as MockChunker,
        patch("cli.commands.DocRegistry") as MockRegistry,
    ):
        mock_chunk = MagicMock()
        mock_chunk.text = "Text."
        MockChunker.return_value = [mock_chunk]
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)

        result = runner.invoke(main, ["add", str(test_doc), "--data-dir", str(tmp_path)])

    assert result.exit_code == 0
    MockRegistry.return_value.upsert.assert_called_once_with(
        str(test_doc),
        title=None,
        topic=None,
        tags=[],
    )
```

**Step 2: Run to verify they fail**

```bash
python3 -m pytest tests/test_cli.py::test_add_command_calls_doc_registry_upsert tests/test_cli.py::test_add_command_without_metadata_flags_upserts_nones -v
```
Expected: FAIL — `add` command does not have `--title`/`--topic`/`--tag` yet.

**Step 3: Update `add` command in `cli/commands.py`**

Add import at the top of the file (after existing imports):

```python
from storage.doc_registry import DocRegistry
```

Replace the `add` command definition:

```python
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
```

**Step 4: Run new CLI tests**

```bash
python3 -m pytest tests/test_cli.py::test_add_command_calls_doc_registry_upsert tests/test_cli.py::test_add_command_without_metadata_flags_upserts_nones -v
```
Expected: PASS

**Step 5: Run full unit suite**

```bash
python3 -m pytest tests/ -m "not integration and not slow" -v
```
Expected: all PASS

**Step 6: Commit**

```bash
git add cli/commands.py tests/test_cli.py
git commit -m "feat: add --title/--topic/--tag flags to docling-rag add"
```

---

## Task 6: CLI search — --tag and --topic filter

**Files:**
- Modify: `cli/commands.py`
- Modify: `tests/test_cli.py`

**Step 1: Write failing tests**

Append to `tests/test_cli.py`:

```python
def test_search_with_tag_filter_passes_allowed_sources(runner, tmp_path):
    """search --tag filters to docs that have that tag."""
    with (
        patch("cli.commands.Embedder") as MockEmbedder,
        patch("cli.commands.FileStorage") as MockStorage,
        patch("cli.commands.DocRegistry") as MockRegistry,
    ):
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)
        MockRegistry.return_value.load.return_value = {
            "arch.pdf": {"title": "T", "topic": "arch", "tags": ["arch"], "added_at": "2026-01-01"},
            "data.pdf": {"title": "D", "topic": "data", "tags": ["etl"],  "added_at": "2026-01-01"},
        }
        MockStorage.return_value.search.return_value = [
            ({"text": "result", "source_file": "arch.pdf", "page_number": 1, "element_type": "text"}, 0.9)
        ]

        result = runner.invoke(main, [
            "search", "query text",
            "--data-dir", str(tmp_path),
            "--tag", "arch",
        ])

    assert result.exit_code == 0
    call_kwargs = MockStorage.return_value.search.call_args
    assert call_kwargs.kwargs.get("allowed_sources") == {"arch.pdf"} or \
           (call_kwargs.args and {"arch.pdf"} in call_kwargs.args)


def test_search_with_topic_filter_case_insensitive(runner, tmp_path):
    """search --topic filters case-insensitively."""
    with (
        patch("cli.commands.Embedder") as MockEmbedder,
        patch("cli.commands.FileStorage") as MockStorage,
        patch("cli.commands.DocRegistry") as MockRegistry,
    ):
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)
        MockRegistry.return_value.load.return_value = {
            "arch.pdf": {"title": "T", "topic": "Software Architecture", "tags": [], "added_at": "2026-01-01"},
            "data.pdf": {"title": "D", "topic": "data engineering",     "tags": [], "added_at": "2026-01-01"},
        }
        MockStorage.return_value.search.return_value = [
            ({"text": "r", "source_file": "arch.pdf", "page_number": 1, "element_type": "text"}, 0.8)
        ]

        result = runner.invoke(main, [
            "search", "patterns",
            "--data-dir", str(tmp_path),
            "--topic", "software architecture",
        ])

    assert result.exit_code == 0
    call_kwargs = MockStorage.return_value.search.call_args
    allowed = call_kwargs.kwargs.get("allowed_sources") or (call_kwargs.args[2] if len(call_kwargs.args) > 2 else None)
    assert "arch.pdf" in allowed
    assert "data.pdf" not in allowed


def test_search_filter_no_matching_docs_exits_gracefully(runner, tmp_path):
    """search --tag with no matching docs prints message and does not call storage."""
    with (
        patch("cli.commands.Embedder"),
        patch("cli.commands.FileStorage") as MockStorage,
        patch("cli.commands.DocRegistry") as MockRegistry,
    ):
        MockRegistry.return_value.load.return_value = {
            "data.pdf": {"title": "D", "topic": "data", "tags": ["etl"], "added_at": "2026-01-01"},
        }

        result = runner.invoke(main, [
            "search", "query",
            "--data-dir", str(tmp_path),
            "--tag", "nonexistent-tag",
        ])

    assert result.exit_code == 0
    assert "нет документов" in result.output.lower() or "no documents" in result.output.lower()
    MockStorage.return_value.search.assert_not_called()
```

**Step 2: Run to verify they fail**

```bash
python3 -m pytest tests/test_cli.py::test_search_with_tag_filter_passes_allowed_sources tests/test_cli.py::test_search_with_topic_filter_case_insensitive tests/test_cli.py::test_search_filter_no_matching_docs_exits_gracefully -v
```
Expected: FAIL — `search` command has no `--tag`/`--topic` flags.

**Step 3: Update `search` command in `cli/commands.py`**

Replace the `search` command:

```python
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
        query_emb = embedder.embed([query])[0]
        results = storage.search(query_embedding=query_emb, top_k=k, allowed_sources=allowed_sources)
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
```

**Step 4: Run new search tests**

```bash
python3 -m pytest tests/test_cli.py::test_search_with_tag_filter_passes_allowed_sources tests/test_cli.py::test_search_with_topic_filter_case_insensitive tests/test_cli.py::test_search_filter_no_matching_docs_exits_gracefully -v
```
Expected: PASS

**Step 5: Run full unit suite**

```bash
python3 -m pytest tests/ -m "not integration and not slow" -v
```
Expected: all PASS

**Step 6: Commit**

```bash
git add cli/commands.py tests/test_cli.py
git commit -m "feat: add --tag/--topic filter to docling-rag search"
```

---

## Task 7: CLI list — enrich with DocRegistry metadata

**Files:**
- Modify: `cli/commands.py`
- Modify: `tests/test_cli.py`

**Step 1: Write failing tests**

Append to `tests/test_cli.py`:

```python
def test_list_shows_title_topic_tags(runner, tmp_path):
    """list command joins chunk counts with doc registry metadata."""
    with (
        patch("cli.commands.FileStorage") as MockStorage,
        patch("cli.commands.DocRegistry") as MockRegistry,
    ):
        MockStorage.return_value.load.return_value = (
            np.zeros((5, 384), dtype=np.float32),
            [{"source_file": "books/arch.pdf"} for _ in range(5)],
        )
        MockRegistry.return_value.load.return_value = {
            "books/arch.pdf": {
                "title": "Clean Architecture",
                "topic": "software",
                "tags": ["arch", "solid"],
                "added_at": "2026-02-14T10:00:00",
            }
        }

        result = runner.invoke(main, ["list", "--data-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "Clean Architecture" in result.output
    assert "software" in result.output
    assert "arch" in result.output


def test_list_shows_dashes_for_docs_without_registry_entry(runner, tmp_path):
    """list shows — for docs that have no entry in doc_index.json."""
    with (
        patch("cli.commands.FileStorage") as MockStorage,
        patch("cli.commands.DocRegistry") as MockRegistry,
    ):
        MockStorage.return_value.load.return_value = (
            np.zeros((3, 384), dtype=np.float32),
            [{"source_file": "old.pdf"} for _ in range(3)],
        )
        MockRegistry.return_value.load.return_value = {}  # no entries

        result = runner.invoke(main, ["list", "--data-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "—" in result.output or "-" in result.output
```

**Step 2: Run to verify they fail**

```bash
python3 -m pytest tests/test_cli.py::test_list_shows_title_topic_tags tests/test_cli.py::test_list_shows_dashes_for_docs_without_registry_entry -v
```
Expected: FAIL — `list` does not use DocRegistry.

**Step 3: Update `list_docs` command in `cli/commands.py`**

Replace the `list_docs` command:

```python
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
```

**Step 4: Run new list tests**

```bash
python3 -m pytest tests/test_cli.py::test_list_shows_title_topic_tags tests/test_cli.py::test_list_shows_dashes_for_docs_without_registry_entry -v
```
Expected: PASS

**Step 5: Run full unit suite**

```bash
python3 -m pytest tests/ -m "not integration and not slow" -v
```
Expected: all PASS

**Step 6: Commit**

```bash
git add cli/commands.py tests/test_cli.py
git commit -m "feat: enrich docling-rag list with title/topic/tags from DocRegistry"
```

---

## Task 8: Integration test

**Files:**
- Modify: `tests/test_integration.py`

**Step 1: Read existing integration test to understand fixture setup**

Read `tests/test_integration.py` to understand how the real pipeline is invoked.

**Step 2: Add integration test**

Append to `tests/test_integration.py`:

```python
@pytest.mark.integration
def test_add_with_tags_and_search_filter(tmp_path, cli_runner):
    """
    End-to-end: index two docs with different tags, search with --tag filter
    returns only results from the matching doc.
    """
    # Two minimal text files that parse fast
    doc_arch = tmp_path / "architecture.txt"
    doc_arch.write_text("Hexagonal architecture separates core logic from adapters.")

    doc_data = tmp_path / "data_engineering.txt"
    doc_data.write_text("Data pipelines move and transform data between systems.")

    store_dir = tmp_path / "store"

    # Index first doc with tag=arch
    result = cli_runner.invoke(main, [
        "add", str(doc_arch),
        "--data-dir", str(store_dir),
        "--title", "Arch Book",
        "--topic", "architecture",
        "--tag", "arch",
    ])
    assert result.exit_code == 0, result.output

    # Index second doc with tag=data
    result = cli_runner.invoke(main, [
        "add", str(doc_data),
        "--data-dir", str(store_dir),
        "--title", "Data Book",
        "--topic", "data engineering",
        "--tag", "data",
    ])
    assert result.exit_code == 0, result.output

    # Search without filter — should return results from both docs
    result = cli_runner.invoke(main, [
        "search", "logic and systems",
        "--data-dir", str(store_dir),
        "--top-k", "5",
    ])
    assert result.exit_code == 0
    sources_in_output = result.output
    assert "architecture" in sources_in_output.lower() or "arch" in sources_in_output.lower()

    # Search with --tag arch — must not return data doc
    result = cli_runner.invoke(main, [
        "search", "logic and systems",
        "--data-dir", str(store_dir),
        "--tag", "arch",
        "--top-k", "5",
    ])
    assert result.exit_code == 0
    assert "data_engineering" not in result.output
    assert "architecture" in result.output.lower() or "arch" in result.output.lower()
```

**Step 3: Run integration test**

```bash
python3 -m pytest tests/test_integration.py -m integration -s -v
```
Expected: PASS (takes ~10–15 seconds due to model loading)

**Step 4: Run complete test suite**

```bash
python3 -m pytest tests/ -m "not slow" -v
```
Expected: all PASS

**Step 5: Final commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test for tag-filtered search"
```
