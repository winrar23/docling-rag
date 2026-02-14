# Design: Document Metadata & Categorization

**Date:** 2026-02-14
**Status:** Approved
**Scope:** P1 feature — document-level metadata (title, topic, tags) + filtered search

---

## Problem

Currently `metadata.json` stores only chunk-level data: `source_file`, `chunk_id`,
`page_number`, `element_type`. There is no way to attach a human-readable title,
domain/topic, or tags to a document, and `docling-rag list` shows only file paths
and chunk counts.

---

## Goals

- Allow users to label documents with `--title`, `--topic`, `--tag` flags at index time
- Show labels in `docling-rag list`
- Filter search results by tag and/or topic via `docling-rag search ... --tag X`

---

## Non-Goals

- Auto-detection of title/topic from document content (future)
- Full-text search on metadata fields
- Editing tags after indexing (future `update` command)

---

## Data Model

New file: `data/doc_index.json`

```json
{
  "docs/test_file.pdf": {
    "title": "Scalable Data Warehousing Using Data Vault 2.0",
    "topic": "data engineering",
    "tags": ["data-vault", "warehousing"],
    "added_at": "2026-02-14T12:00:00"
  }
}
```

| Field | Type | Notes |
|---|---|---|
| key | `str` | `source_file` path — future FK to `documents.source_file` in Postgres |
| `title` | `str \| null` | optional, free-form |
| `topic` | `str \| null` | optional, free-form domain label |
| `tags` | `list[str]` | zero or more, set via `--tag` (repeatable) |
| `added_at` | ISO 8601 str | set on first `upsert`, not overwritten on re-index |

`metadata.json` (chunks) is unchanged — `source_file` remains the join key.

---

## Architecture

### PostgreSQL migration path

| File storage (MVP) | PostgreSQL (stage 2) |
|---|---|
| `data/metadata.json` | table `chunks` (id, source_file, text, page, element_type, embedding vector(384)) |
| `data/doc_index.json` | table `documents` (id, source_file, title, topic, tags[], added_at) |

The separation introduced here maps 1:1 to a relational schema.
`source_file` string key → `source_file_id` FK at migration time (planned work, not a hack).

---

## Components

### New: `storage/doc_registry.py` — `DocRegistry`

```python
class DocRegistry:
    INDEX_FILE = "doc_index.json"

    def __init__(self, data_dir: str | Path = "data") -> None: ...

    def upsert(
        self,
        source_file: str,
        title: str | None,
        topic: str | None,
        tags: list[str],
    ) -> None:
        """Add or update document entry. Preserves added_at on re-index."""

    def delete(self, source_file: str) -> None: ...
    def get(self, source_file: str) -> dict | None: ...
    def load(self) -> dict[str, dict]: ...  # {source_file: {title, topic, tags, added_at}}
```

- Atomic `_atomic_save` via `os.replace()` — same pattern as `FileStorage`
- `added_at` is set once on first `upsert`, preserved on subsequent calls
- Missing `doc_index.json` → empty dict (graceful, no FileNotFoundError)

### New Protocol: `core/storage.py`

Add `DocumentRegistryBackend` Protocol alongside existing `StorageBackend`.
Two independent protocols → two independent backends (file now, Postgres later).

### Modified: `storage/file_storage.py` — `FileStorage.search`

Add optional `allowed_sources` parameter:

```python
def search(
    self,
    query_embedding: np.ndarray,
    top_k: int = 5,
    allowed_sources: set[str] | None = None,  # None = no filter
) -> list[tuple[dict, float]]:
```

When `allowed_sources` is set, filter metadata index before matrix multiply:

```python
if allowed_sources is not None:
    mask = [i for i, m in enumerate(metadata) if m["source_file"] in allowed_sources]
    embeddings = embeddings[mask]
    metadata = [metadata[i] for i in mask]
```

O(N) pre-filter — cheaper than post-filtering top-k results.

---

## CLI Changes

### `add` command

```bash
docling-rag add book.pdf --title "Clean Architecture" --topic "architecture" \
    --tag arch --tag solid
```

New flags (all optional):
- `--title TEXT`
- `--topic TEXT`
- `--tag TEXT` — repeatable (`multiple=True` in Click), results in `tuple[str]`

After successful chunk indexing: call `DocRegistry.upsert(source_file, title, topic, list(tags))`.

### `search` command

```bash
docling-rag search "dependency inversion" --tag arch
docling-rag search "data vault hubs" --topic "data engineering"
```

New flags (all optional):
- `--tag TEXT` — repeatable, filter to docs where ALL specified tags are present
- `--topic TEXT` — filter to docs with matching topic (case-insensitive)

Logic:
1. Load `DocRegistry`
2. Build `allowed_sources` set from matching docs
3. Pass to `FileStorage.search(allowed_sources=...)`
4. If filter produces empty `allowed_sources` → print "Нет документов с такими тегами/темой." and exit

### `list` command

```
Проиндексировано документов: 2
------------------------------------------------------------
  clean-arch.pdf     312 chunks | Clean Architecture     | architecture       | [arch, solid]
  test_file.pdf        8 chunks | Scalable DWH...        | data engineering   | [data-vault]
```

Join chunk counts from `metadata.json` with labels from `doc_index.json`.
Documents in `metadata.json` without a `doc_index.json` entry show `—` for metadata fields
(backwards compatibility).

---

## Testing

- **Unit tests** for `DocRegistry`: upsert, upsert preserves added_at, delete, missing file
- **Unit tests** for `FileStorage.search` with `allowed_sources`: filters correctly, empty set, None
- **Unit tests** for CLI `add` with `--tag`/`--title`/`--topic` flags (mock DocRegistry)
- **Unit tests** for CLI `search` with `--tag`/`--topic` (mock storage)
- **Integration test**: add two docs with different tags, search with tag filter → only relevant doc returned

---

## Files Touched

| File | Change |
|---|---|
| `storage/doc_registry.py` | **new** |
| `core/storage.py` | add `DocumentRegistryBackend` Protocol |
| `storage/file_storage.py` | add `allowed_sources` to `search` |
| `cli/commands.py` | `add` flags, `search` flags + filter logic, `list` join |
| `tests/test_doc_registry.py` | **new** |
| `tests/test_file_storage.py` | extend with `allowed_sources` tests |
| `tests/test_commands.py` | extend with new flag tests |
