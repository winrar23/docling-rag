import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from docling_rag.cli import main
from docling_rag.core.chunker import Chunk
from docling_rag.core.errors import StorageError, StorageSchemaMissingError, StorageUnavailableError
from tests.fakes import InMemoryRegistry


def test_init_command_initializes_schema(runner, hermetic_config, monkeypatch):
    """init calls init_schema(database_url) — DDL, not a directory."""
    called = {}

    def fake_init_schema(dsn):
        called["dsn"] = dsn

    monkeypatch.setattr("docling_rag.cli.commands.init_schema", fake_init_schema)
    result = runner.invoke(main, ["init"])
    assert result.exit_code == 0
    assert called["dsn"] == hermetic_config["database_url"]
    assert "Схема БД инициализирована" in result.output


def test_init_postgres_down_gives_helpful_error(runner, monkeypatch):
    def raise_unavailable(dsn):
        raise StorageUnavailableError("connection refused")

    monkeypatch.setattr("docling_rag.cli.commands.init_schema", raise_unavailable)
    result = runner.invoke(main, ["init"])
    assert result.exit_code == 1
    assert "PostgreSQL недоступен" in result.output
    assert "docker compose up -d postgres" in result.output


def test_list_command_empty_storage(runner, fake_backends):
    result = runner.invoke(main, ["list"])
    assert result.exit_code == 0
    assert "пустое" in result.output.lower() or "нет" in result.output.lower()


def test_add_command_indexes_file(runner, tmp_path):
    test_doc = tmp_path / "test.md"
    test_doc.write_text("# Test\n\nThis is a test document.\n")

    with (
        patch("docling_rag.cli.commands.Parser") as MockParser,
        patch("docling_rag.cli.commands.Embedder") as MockEmbedder,
        patch("docling_rag.cli.commands.DBStorage") as MockStorage,
        patch("docling_rag.core.indexer.chunk_document") as MockChunkDoc,
        patch("docling_rag.cli.commands.DBRegistry") as MockRegistry,
    ):
        mock_chunk = MagicMock()
        mock_chunk.context_text = "Test heading\nTest content."
        MockChunkDoc.return_value = [mock_chunk]

        embedder_instance = MockEmbedder.return_value
        storage_instance = MockStorage.return_value
        embedder_instance.embed.return_value = np.ones((1, 384), dtype=np.float32)

        result = runner.invoke(main, ["add", str(test_doc)])

    assert result.exit_code == 0
    assert "chunk" in result.output.lower() or "добавлен" in result.output.lower()

    MockParser.return_value.parse.assert_called_once_with(test_doc)
    # Embedding uses context_text, not text
    embedder_instance.embed.assert_called_once_with([mock_chunk.context_text], batch_size=128)
    storage_instance.append.assert_called_once()


def test_add_command_skips_file_on_exception(runner, tmp_path):
    test_doc = tmp_path / "corrupt.pdf"
    test_doc.write_bytes(b"%PDF-1.4 corrupted content")

    with (
        patch("docling_rag.cli.commands.Parser") as MockParser,
        patch("docling_rag.cli.commands.Embedder"),
        patch("docling_rag.cli.commands.DBStorage"),
        patch("docling_rag.cli.commands.DBRegistry"),
    ):
        MockParser.return_value.parse.side_effect = Exception("corrupt PDF")

        result = runner.invoke(main, ["add", str(test_doc)])

    assert result.exit_code != 0
    assert "Ошибка при обработке" in result.output or "corrupt" in result.output.lower()


def test_add_skips_txt_files(runner, tmp_path):
    """Docling can't parse .txt — add must not even try."""
    (tmp_path / "notes.txt").write_text("plain text")
    result = runner.invoke(main, ["add", str(tmp_path / "notes.txt")])
    assert "Нет поддерживаемых файлов" in result.output


def test_search_command_returns_results(runner):
    mock_results = [
        ({"text": "SQL query example SELECT *", "source_file": "doc.pdf",
          "chunk_id": 0, "page_number": 1, "element_type": "code"}, 0.92),
        ({"text": "Database schema description", "source_file": "arch.docx",
          "chunk_id": 1, "page_number": 2, "element_type": "text"}, 0.78),
    ]

    with (
        patch("docling_rag.cli.commands.Embedder") as MockEmbedder,
        patch("docling_rag.cli.commands.DBStorage") as MockStorage,
    ):
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)
        MockStorage.return_value.search.return_value = mock_results

        result = runner.invoke(main, ["search", "SQL query example"])

    assert result.exit_code == 0
    assert "0.920" in result.output or "0.92" in result.output
    assert "doc.pdf" in result.output


def test_search_command_empty_storage(runner):
    with (
        patch("docling_rag.cli.commands.Embedder") as MockEmbedder,
        patch("docling_rag.cli.commands.DBStorage") as MockStorage,
    ):
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)
        MockStorage.return_value.search.side_effect = FileNotFoundError

        result = runner.invoke(main, ["search", "query"])

    assert result.exit_code != 0
    assert "пустое" in result.output.lower() or "нет документов" in result.output.lower()


def test_search_reports_corrupted_storage(runner):
    with patch("docling_rag.cli.commands.Embedder") as MockEmbedder, \
         patch("docling_rag.cli.commands.DBStorage") as MockStorage:
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)
        MockStorage.return_value.search.side_effect = StorageError("2 vs 3")
        result = runner.invoke(main, ["search", "q"])
    assert result.exit_code != 0
    assert "повреждено" in result.output.lower()


def test_search_postgres_down_gives_helpful_error(runner, monkeypatch):
    class BoomStorage:
        def __init__(self, dsn): pass
        def search(self, *a, **kw): raise StorageUnavailableError("connection refused")
        def load(self): raise StorageUnavailableError("connection refused")

    monkeypatch.setattr("docling_rag.cli.commands.DBStorage", BoomStorage)
    monkeypatch.setattr("docling_rag.cli.commands.DBRegistry", lambda dsn: InMemoryRegistry())
    with patch("docling_rag.cli.commands.Embedder") as MockEmbedder:
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)
        result = runner.invoke(main, ["search", "query"])
    assert result.exit_code == 1
    assert "PostgreSQL недоступен" in result.output
    assert "docker compose up -d postgres" in result.output


def test_search_schema_missing_gives_helpful_error(runner, monkeypatch):
    """StorageSchemaMissingError (schema not created yet) -> 'run init' hint."""

    class BoomStorage:
        def __init__(self, dsn): pass
        def search(self, *a, **kw):
            raise StorageSchemaMissingError('relation "chunks" does not exist')
        def load(self):
            raise StorageSchemaMissingError('relation "chunks" does not exist')

    monkeypatch.setattr("docling_rag.cli.commands.DBStorage", BoomStorage)
    monkeypatch.setattr("docling_rag.cli.commands.DBRegistry", lambda dsn: InMemoryRegistry())
    with patch("docling_rag.cli.commands.Embedder") as MockEmbedder:
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)
        result = runner.invoke(main, ["search", "query"])
    assert result.exit_code == 1
    assert "Схема БД не инициализирована" in result.output
    assert "docling-rag init" in result.output


def test_list_postgres_down_gives_helpful_error(runner, monkeypatch):
    class BoomStorage:
        def __init__(self, dsn): pass
        def load(self): raise StorageUnavailableError("connection refused")

    monkeypatch.setattr("docling_rag.cli.commands.DBStorage", BoomStorage)
    monkeypatch.setattr("docling_rag.cli.commands.DBRegistry", lambda dsn: InMemoryRegistry())
    result = runner.invoke(main, ["list"])
    assert result.exit_code == 1
    assert "PostgreSQL недоступен" in result.output
    assert "docker compose up -d postgres" in result.output


def test_add_postgres_down_gives_helpful_error(runner, tmp_path, monkeypatch):
    """Important-фикс: инфраструктурная ошибка при add НЕ должна тонуть в per-file
    отчёте indexer'а — полная цепочка add -> index_files -> storage.append должна
    донести до пользователя «PostgreSQL недоступен» + подсказку compose."""

    class BoomStorage:
        def __init__(self, dsn): pass
        def delete_by_source(self, source_file): pass
        def append(self, *a, **kw): raise StorageUnavailableError("connection refused")

    test_doc = tmp_path / "down.md"
    test_doc.write_text("# T\n\ntext\n")
    monkeypatch.setattr("docling_rag.cli.commands.DBStorage", BoomStorage)
    monkeypatch.setattr("docling_rag.cli.commands.DBRegistry", lambda dsn: InMemoryRegistry())
    with (
        patch("docling_rag.cli.commands.Parser"),
        patch("docling_rag.cli.commands.Embedder") as MockEmbedder,
        patch("docling_rag.core.indexer.chunk_document") as MockChunkDoc,
    ):
        mock_chunk = MagicMock()
        mock_chunk.context_text = "t"
        MockChunkDoc.return_value = [mock_chunk]
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)
        result = runner.invoke(main, ["add", str(test_doc)])
    assert result.exit_code == 1
    assert "PostgreSQL недоступен" in result.output
    assert "docker compose up -d postgres" in result.output
    assert "Ошибка при обработке" not in result.output  # не per-file отчёт, а fail-fast


def test_add_schema_missing_gives_helpful_error(runner, tmp_path, monkeypatch):
    class BoomStorage:
        def __init__(self, dsn): pass
        def delete_by_source(self, source_file):
            raise StorageSchemaMissingError('relation "chunks" does not exist')

    test_doc = tmp_path / "noschema.md"
    test_doc.write_text("# T\n\ntext\n")
    monkeypatch.setattr("docling_rag.cli.commands.DBStorage", BoomStorage)
    monkeypatch.setattr("docling_rag.cli.commands.DBRegistry", lambda dsn: InMemoryRegistry())
    with (
        patch("docling_rag.cli.commands.Parser"),
        patch("docling_rag.cli.commands.Embedder") as MockEmbedder,
        patch("docling_rag.core.indexer.chunk_document") as MockChunkDoc,
    ):
        mock_chunk = MagicMock()
        mock_chunk.context_text = "t"
        MockChunkDoc.return_value = [mock_chunk]
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)
        result = runner.invoke(main, ["add", str(test_doc)])
    assert result.exit_code == 1
    assert "Схема БД не инициализирована" in result.output
    assert "docling-rag init" in result.output


def test_search_does_not_crash_when_log_raises_oserror(runner):
    mock_results = [
        ({"text": "Some result text", "source_file": "doc.pdf",
          "chunk_id": 0, "page_number": 1, "element_type": "text"}, 0.85),
    ]

    import builtins
    real_open = builtins.open

    def patched_open(file, *args, **kwargs):
        if "search_log" in str(file) or str(file).endswith(".log"):
            raise OSError("permission denied")
        return real_open(file, *args, **kwargs)

    with (
        patch("docling_rag.cli.commands.Embedder") as MockEmbedder,
        patch("docling_rag.cli.commands.DBStorage") as MockStorage,
        patch("builtins.open", side_effect=patched_open),
    ):
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)
        MockStorage.return_value.search.return_value = mock_results

        result = runner.invoke(main, ["search", "some query"])

    assert result.exit_code == 0
    assert "doc.pdf" in result.output
    assert "Предупреждение" in result.output or "не удалось записать лог" in result.output


def test_add_command_calls_doc_registry_upsert(runner, tmp_path):
    """add with --title/--topic/--tag calls DBRegistry.upsert with correct args."""
    test_doc = tmp_path / "book.md"
    test_doc.write_text("# Book\n\nContent here.\n")

    with (
        patch("docling_rag.cli.commands.Parser"),
        patch("docling_rag.cli.commands.Embedder") as MockEmbedder,
        patch("docling_rag.cli.commands.DBStorage"),
        patch("docling_rag.core.indexer.chunk_document") as MockChunkDoc,
        patch("docling_rag.cli.commands.DBRegistry") as MockRegistry,
    ):
        mock_chunk = MagicMock()
        mock_chunk.context_text = "Content here."
        MockChunkDoc.return_value = [mock_chunk]
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)

        result = runner.invoke(main, [
            "add", str(test_doc),
            "--title", "My Book",
            "--topic", "architecture",
            "--tag", "arch",
            "--tag", "solid",
        ])

    assert result.exit_code == 0
    MockRegistry.return_value.upsert.assert_called_once_with(
        str(test_doc.resolve()),
        title="My Book",
        topic="architecture",
        tags=["arch", "solid"],
    )


def test_add_command_without_metadata_flags_upserts_nones(runner, tmp_path):
    """add without metadata flags calls upsert with None/empty."""
    test_doc = tmp_path / "plain.md"
    test_doc.write_text("# Plain\n\nText.\n")

    with (
        patch("docling_rag.cli.commands.Parser"),
        patch("docling_rag.cli.commands.Embedder") as MockEmbedder,
        patch("docling_rag.cli.commands.DBStorage"),
        patch("docling_rag.core.indexer.chunk_document") as MockChunkDoc,
        patch("docling_rag.cli.commands.DBRegistry") as MockRegistry,
    ):
        mock_chunk = MagicMock()
        mock_chunk.context_text = "Text."
        MockChunkDoc.return_value = [mock_chunk]
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)

        result = runner.invoke(main, ["add", str(test_doc)])

    assert result.exit_code == 0
    MockRegistry.return_value.upsert.assert_called_once_with(
        str(test_doc.resolve()),
        title=None,
        topic=None,
        tags=[],
    )


def test_add_passes_chunk_max_tokens_from_config(runner, tmp_path, hermetic_config):
    """Full chain commands.add -> index_files -> chunk_document must deliver
    chunk_max_tokens from config. Non-default 384: since index_files and
    chunk_document both default to 512, a dropped kwarg would otherwise pass unnoticed."""
    from unittest.mock import ANY

    # hermetic_config's load_config lambda closes over this dict and copies it
    # at call time — mutating it here changes what the CLI sees in cfg.
    hermetic_config["chunk_max_tokens"] = 384

    test_doc = tmp_path / "tokens.md"
    test_doc.write_text("# T\n\ntext\n")

    with (
        patch("docling_rag.cli.commands.Parser"),
        patch("docling_rag.cli.commands.Embedder") as MockEmbedder,
        patch("docling_rag.cli.commands.DBStorage"),
        patch("docling_rag.core.indexer.chunk_document") as MockChunkDoc,
        patch("docling_rag.cli.commands.DBRegistry"),
    ):
        mock_chunk = MagicMock()
        mock_chunk.context_text = "text"
        MockChunkDoc.return_value = [mock_chunk]
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)

        result = runner.invoke(main, ["add", str(test_doc)])

    assert result.exit_code == 0
    MockChunkDoc.assert_called_once_with(
        ANY,
        source_file=str(test_doc.resolve()),
        embedding_model="all-MiniLM-L6-v2",
        max_tokens=384,
    )


def test_re_add_same_file_does_not_duplicate(runner, tmp_path):
    """Re-adding a file must delete its old chunks first (idempotent add)."""
    test_doc = tmp_path / "b.md"
    test_doc.write_text("# T\n\ntext\n")
    with (
        patch("docling_rag.cli.commands.Parser"),
        patch("docling_rag.cli.commands.Embedder") as MockEmbedder,
        patch("docling_rag.cli.commands.DBStorage") as MockStorage,
        patch("docling_rag.core.indexer.chunk_document") as MockChunkDoc,
        patch("docling_rag.cli.commands.DBRegistry"),
    ):
        mock_chunk = MagicMock()
        mock_chunk.context_text = "t"
        MockChunkDoc.return_value = [mock_chunk]
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)
        runner.invoke(main, ["add", str(test_doc)])
    expected_source = str(test_doc.resolve())
    MockStorage.return_value.delete_by_source.assert_called_once_with(expected_source)
    MockStorage.return_value.append.assert_called_once()


def test_add_uses_resolved_path_as_source(runner, tmp_path):
    test_doc = tmp_path / "c.md"
    test_doc.write_text("# T\n\ntext\n")
    with (
        patch("docling_rag.cli.commands.Parser"),
        patch("docling_rag.cli.commands.Embedder") as MockEmbedder,
        patch("docling_rag.cli.commands.DBStorage"),
        patch("docling_rag.core.indexer.chunk_document") as MockChunkDoc,
        patch("docling_rag.cli.commands.DBRegistry") as MockRegistry,
    ):
        mock_chunk = MagicMock(); mock_chunk.context_text = "t"
        MockChunkDoc.return_value = [mock_chunk]
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)
        runner.invoke(main, ["add", str(test_doc)])
    MockRegistry.return_value.upsert.assert_called_once_with(
        str(test_doc.resolve()), title=None, topic=None, tags=[],
    )


def test_search_with_tag_filter_passes_allowed_sources(runner):
    """search --tag filters to docs that have that tag."""
    with (
        patch("docling_rag.cli.commands.Embedder") as MockEmbedder,
        patch("docling_rag.cli.commands.DBStorage") as MockStorage,
        patch("docling_rag.cli.commands.DBRegistry") as MockRegistry,
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
            "--tag", "arch",
        ])

    assert result.exit_code == 0
    call_kwargs = MockStorage.return_value.search.call_args
    assert call_kwargs.kwargs.get("allowed_sources") == {"arch.pdf"} or \
           (call_kwargs.args and {"arch.pdf"} in call_kwargs.args)


def test_search_with_topic_filter_case_insensitive(runner):
    """search --topic filters case-insensitively."""
    with (
        patch("docling_rag.cli.commands.Embedder") as MockEmbedder,
        patch("docling_rag.cli.commands.DBStorage") as MockStorage,
        patch("docling_rag.cli.commands.DBRegistry") as MockRegistry,
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
            "--topic", "software architecture",
        ])

    assert result.exit_code == 0
    call_kwargs = MockStorage.return_value.search.call_args
    allowed = call_kwargs.kwargs.get("allowed_sources") or (call_kwargs.args[2] if len(call_kwargs.args) > 2 else None)
    assert "arch.pdf" in allowed
    assert "data.pdf" not in allowed


def test_search_filter_no_matching_docs_exits_gracefully(runner):
    """search --tag with no matching docs prints message and does not call storage."""
    with (
        patch("docling_rag.cli.commands.Embedder"),
        patch("docling_rag.cli.commands.DBStorage") as MockStorage,
        patch("docling_rag.cli.commands.DBRegistry") as MockRegistry,
    ):
        MockRegistry.return_value.load.return_value = {
            "data.pdf": {"title": "D", "topic": "data", "tags": ["etl"], "added_at": "2026-01-01"},
        }

        result = runner.invoke(main, [
            "search", "query",
            "--tag", "nonexistent-tag",
        ])

    assert result.exit_code == 0
    assert "нет документов" in result.output.lower() or "no documents" in result.output.lower()
    MockStorage.return_value.search.assert_not_called()


def test_list_shows_title_topic_tags(runner):
    """list command joins chunk counts with doc registry metadata."""
    with (
        patch("docling_rag.cli.commands.DBStorage") as MockStorage,
        patch("docling_rag.cli.commands.DBRegistry") as MockRegistry,
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

        result = runner.invoke(main, ["list"])

    assert result.exit_code == 0
    assert "Clean Architecture" in result.output
    assert "software" in result.output
    assert "arch" in result.output


def test_list_shows_dashes_for_docs_without_registry_entry(runner):
    """list shows — for docs that have no entry in doc_index.json."""
    with (
        patch("docling_rag.cli.commands.DBStorage") as MockStorage,
        patch("docling_rag.cli.commands.DBRegistry") as MockRegistry,
    ):
        MockStorage.return_value.load.return_value = (
            np.zeros((3, 384), dtype=np.float32),
            [{"source_file": "old.pdf"} for _ in range(3)],
        )
        MockRegistry.return_value.load.return_value = {}  # no entries

        result = runner.invoke(main, ["list"])

    assert result.exit_code == 0
    assert "—" in result.output or "-" in result.output


def test_search_shows_headings_in_output(runner):
    """search results display headings when present."""
    mock_results = [
        ({"text": "Some content about patterns", "source_file": "doc.pdf",
          "chunk_id": 0, "page_number": 2, "element_type": "text",
          "headings": ["Chapter 3", "Design Patterns"]}, 0.91),
    ]

    with (
        patch("docling_rag.cli.commands.Embedder") as MockEmbedder,
        patch("docling_rag.cli.commands.DBStorage") as MockStorage,
    ):
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)
        MockStorage.return_value.search.return_value = mock_results

        result = runner.invoke(main, ["search", "patterns"])

    assert result.exit_code == 0
    assert "Chapter 3" in result.output
    assert "Design Patterns" in result.output


# --- ask command tests ---


def test_ask_disabled_by_default(runner):
    """ask with agent_enabled=false shows activation hint."""
    result = runner.invoke(main, ["ask", "test question"])
    assert result.exit_code != 0
    assert "agent_enabled" in result.output or "отключён" in result.output.lower()


def test_ask_shows_install_hint_when_pydantic_ai_missing(runner):
    """ask with agent_enabled=true but pydantic-ai missing shows install hint."""
    with patch("docling_rag.cli.commands.load_config", return_value={
        "agent_enabled": True,
        "llm_base_url": "http://127.0.0.1:1234/v1",
        "llm_api_key": "lm-studio",
        "llm_model": "test-model",
        "agent_top_k": 5,
        "embedding_model": "all-MiniLM-L6-v2",
    }):
        with patch("docling_rag.cli.commands._import_agent_module", side_effect=ImportError("No module named 'pydantic_ai'")):
            result = runner.invoke(main, ["ask", "question"])
    assert result.exit_code != 0
    assert "install" in result.output.lower() or "pip" in result.output.lower() or "[agent]" in result.output


def test_ask_calls_agent_and_prints_output(runner):
    """ask with enabled agent calls _create_and_run_agent and prints result."""
    with patch("docling_rag.cli.commands.load_config", return_value={
        "agent_enabled": True,
        "llm_base_url": "http://127.0.0.1:1234/v1",
        "llm_api_key": "lm-studio",
        "llm_model": "test-model",
        "agent_top_k": 5,
        "embedding_model": "all-MiniLM-L6-v2",
    }):
        with patch("docling_rag.cli.commands._create_and_run_agent", return_value="Data Vault uses hubs, links, and satellites."):
            result = runner.invoke(main, ["ask", "What is Data Vault?"])
    assert result.exit_code == 0
    assert "Data Vault" in result.output


def test_explicit_config_path_missing_fails(runner, tmp_path, monkeypatch):
    monkeypatch.setattr("docling_rag.cli.commands.load_config", __import__("docling_rag.cli.config_loader", fromlist=["load_config"]).load_config)
    result = runner.invoke(main, ["search", "q", "--config", str(tmp_path / "typo.yml")])
    assert result.exit_code != 0
    assert "не найден" in result.output


def test_ask_handles_connection_error(runner):
    """ask prints helpful message when LLM is unreachable."""
    with patch("docling_rag.cli.commands.load_config", return_value={
        "agent_enabled": True,
        "llm_base_url": "http://127.0.0.1:1234/v1",
        "llm_api_key": "lm-studio",
        "llm_model": "test-model",
        "agent_top_k": 5,
        "embedding_model": "all-MiniLM-L6-v2",
    }):
        with patch("docling_rag.cli.commands._create_and_run_agent", side_effect=ConnectionError("Connection refused")):
            result = runner.invoke(main, ["ask", "test"])
    assert result.exit_code != 0
    assert "подключиться" in result.output.lower() or "connection" in result.output.lower() or "lm studio" in result.output.lower()


def test_add_exits_nonzero_when_no_supported_files(runner, tmp_path):
    (tmp_path / "x.csv").write_text("a,b")
    result = runner.invoke(main, ["add", str(tmp_path / "x.csv")])
    assert result.exit_code != 0


def test_top_k_zero_rejected(runner):
    result = runner.invoke(main, ["search", "q", "--top-k", "0"])
    assert result.exit_code != 0
    assert "top-k" in result.output.lower() or "range" in result.output.lower()


def _agent_cfg():
    return {
        "agent_enabled": True, "llm_base_url": "http://127.0.0.1:1234/v1",
        "llm_api_key": "lm-studio", "llm_model": "m", "agent_top_k": 5,
        "embedding_model": "all-MiniLM-L6-v2",
        "top_k_results": 5, "log_file": "",
    }


def test_ask_detects_wrapped_connect_error(runner):
    """openai.APIConnectionError wraps httpx.ConnectError as __cause__ — must be detected."""
    httpx = pytest.importorskip("httpx")
    wrapper = Exception("Connection error.")  # имя типа НЕ содержит 'ConnectError'
    wrapper.__cause__ = httpx.ConnectError("All connection attempts failed")
    with patch("docling_rag.cli.commands.load_config", return_value=_agent_cfg()), \
         patch("docling_rag.cli.commands._create_and_run_agent", side_effect=wrapper):
        result = runner.invoke(main, ["ask", "q"])
    assert result.exit_code != 0
    assert "lm studio" in result.output.lower() or "подключиться" in result.output.lower()


def _mk_chunk(source: str, cid: int) -> Chunk:
    return Chunk(text="t", source_file=source, chunk_id=cid, page_number=1,
                 element_type="text", headings=[], context_text="t")


class TestDeleteCommand:
    def test_delete_removes_document_and_chunks(self, runner, fake_backends):
        storage, registry = fake_backends
        storage.append([_mk_chunk("/books/a.pdf", 0), _mk_chunk("/books/a.pdf", 1)],
                       np.zeros((2, 4), dtype=np.float32))
        registry.upsert("/books/a.pdf", title="Book A", topic=None, tags=[])
        result = runner.invoke(main, ["delete", "/books/a.pdf"])
        assert result.exit_code == 0
        assert "Удалено: Book A (2 chunks)" in result.output
        assert registry.get("/books/a.pdf") is None
        assert storage.count_by_source("/books/a.pdf") == 0

    def test_delete_untitled_shows_source(self, runner, fake_backends):
        storage, registry = fake_backends
        storage.append([_mk_chunk("/books/a.pdf", 0)], np.zeros((1, 4), dtype=np.float32))
        registry.upsert("/books/a.pdf", title=None, topic=None, tags=[])
        result = runner.invoke(main, ["delete", "/books/a.pdf"])
        assert result.exit_code == 0
        assert "Удалено: /books/a.pdf (1 chunks)" in result.output

    def test_delete_missing_exits_1(self, runner, fake_backends):
        result = runner.invoke(main, ["delete", "/books/nope.pdf"])
        assert result.exit_code == 1
        assert "не найден" in result.output
        assert "docling-rag list" in result.output
