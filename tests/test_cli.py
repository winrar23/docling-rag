import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from click.testing import CliRunner

from cli import main


@pytest.fixture
def runner():
    return CliRunner()


def test_init_command_creates_data_dir(runner, tmp_path):
    result = runner.invoke(main, ["init", "--data-dir", str(tmp_path / "mystore")])
    assert result.exit_code == 0
    assert (tmp_path / "mystore").exists()
    assert "Инициализировано" in result.output


def test_list_command_empty_storage(runner, tmp_path):
    with patch("cli.commands.FileStorage") as MockStorage:
        MockStorage.return_value.load.side_effect = FileNotFoundError
        result = runner.invoke(main, ["list", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "пустое" in result.output.lower() or "нет" in result.output.lower()


def test_add_command_indexes_file(runner, tmp_path):
    test_doc = tmp_path / "test.md"
    test_doc.write_text("# Test\n\nThis is a test document.\n")

    with (
        patch("cli.commands.Parser") as MockParser,
        patch("cli.commands.Embedder") as MockEmbedder,
        patch("cli.commands.FileStorage") as MockStorage,
        patch("cli.commands.chunk_elements") as MockChunker,
    ):
        mock_chunk = MagicMock()
        mock_chunk.text = "Test content."
        MockChunker.return_value = [mock_chunk]

        parser_instance = MockParser.return_value
        embedder_instance = MockEmbedder.return_value
        storage_instance = MockStorage.return_value

        embedder_instance.embed.return_value = np.ones((1, 384), dtype=np.float32)

        result = runner.invoke(main, ["add", str(test_doc), "--data-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "chunk" in result.output.lower() or "добавлен" in result.output.lower()

    parser_instance.parse.assert_called_once_with(test_doc)
    embedder_instance.embed.assert_called_once_with([mock_chunk.text])
    storage_instance.append.assert_called_once()


def test_add_command_skips_file_on_exception(runner, tmp_path):
    test_doc = tmp_path / "corrupt.pdf"
    test_doc.write_bytes(b"%PDF-1.4 corrupted content")

    with (
        patch("cli.commands.Parser") as MockParser,
        patch("cli.commands.Embedder"),
        patch("cli.commands.FileStorage"),
    ):
        MockParser.return_value.parse.side_effect = Exception("corrupt PDF")

        result = runner.invoke(main, ["add", str(test_doc), "--data-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "Ошибка при обработке" in result.output or "corrupt" in result.output.lower()


def test_search_command_returns_results(runner, tmp_path):
    mock_results = [
        ({"text": "SQL query example SELECT *", "source_file": "doc.pdf",
          "chunk_id": 0, "page_number": 1, "element_type": "code"}, 0.92),
        ({"text": "Database schema description", "source_file": "arch.docx",
          "chunk_id": 1, "page_number": 2, "element_type": "text"}, 0.78),
    ]

    with (
        patch("cli.commands.Embedder") as MockEmbedder,
        patch("cli.commands.FileStorage") as MockStorage,
    ):
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)
        MockStorage.return_value.search.return_value = mock_results

        result = runner.invoke(
            main, ["search", "SQL query example", "--data-dir", str(tmp_path)]
        )

    assert result.exit_code == 0
    assert "0.920" in result.output or "0.92" in result.output
    assert "doc.pdf" in result.output


def test_search_command_empty_storage(runner, tmp_path):
    with (
        patch("cli.commands.Embedder") as MockEmbedder,
        patch("cli.commands.FileStorage") as MockStorage,
    ):
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)
        MockStorage.return_value.search.side_effect = FileNotFoundError

        result = runner.invoke(main, ["search", "query", "--data-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "пустое" in result.output.lower() or "нет документов" in result.output.lower()


def test_search_does_not_crash_when_log_raises_oserror(runner, tmp_path):
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
        patch("cli.commands.Embedder") as MockEmbedder,
        patch("cli.commands.FileStorage") as MockStorage,
        patch("builtins.open", side_effect=patched_open),
    ):
        MockEmbedder.return_value.embed.return_value = np.ones((1, 384), dtype=np.float32)
        MockStorage.return_value.search.return_value = mock_results

        result = runner.invoke(
            main, ["search", "some query", "--data-dir", str(tmp_path)]
        )

    assert result.exit_code == 0
    assert "doc.pdf" in result.output
    assert "Предупреждение" in result.output or "не удалось записать лог" in result.output


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
