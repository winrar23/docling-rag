from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.parser import Parser


class _FakeTableItem:
    """Minimal stand-in for TableItem so isinstance checks pass in tests."""

    def __init__(self, markdown: str) -> None:
        self._markdown = markdown
        self.prov = []

    def export_to_markdown(self) -> str:
        return self._markdown


@pytest.fixture
def mock_docling_result():
    """Mock result from DocumentConverter.convert()"""
    mock_result = MagicMock()
    mock_doc = MagicMock()

    text_item = MagicMock()
    text_item.text = "This is a paragraph about databases."
    text_item.__class__.__name__ = "TextItem"

    # Use _FakeTableItem instances; tests patch core.parser.TableItem to
    # _FakeTableItem so isinstance(item, TableItem) resolves correctly.
    table_item = _FakeTableItem("| col1 | col2 |\n|------|------|\n| A    | B    |")

    code_item = MagicMock()
    code_item.text = "SELECT * FROM users;"
    code_item.__class__.__name__ = "CodeItem"

    mock_doc.iterate_items.return_value = [
        (text_item, 0),
        (table_item, 0),
        (code_item, 0),
    ]
    mock_result.document = mock_doc
    return mock_result


def test_parser_returns_list_of_elements(mock_docling_result, tmp_path):
    fake_file = tmp_path / "test.pdf"
    fake_file.write_bytes(b"fake pdf content")

    with patch("core.parser.DocumentConverter") as MockConverter, \
            patch("core.parser.TableItem", _FakeTableItem):
        MockConverter.return_value.convert.return_value = mock_docling_result
        parser = Parser()
        elements = parser.parse(fake_file)

    assert isinstance(elements, list)
    assert len(elements) == 3


def test_parser_text_element(mock_docling_result, tmp_path):
    fake_file = tmp_path / "test.pdf"
    fake_file.write_bytes(b"fake")

    with patch("core.parser.DocumentConverter") as MockConverter, \
            patch("core.parser.TableItem", _FakeTableItem):
        MockConverter.return_value.convert.return_value = mock_docling_result
        parser = Parser()
        elements = parser.parse(fake_file)

    assert elements[0]["type"] == "text"
    assert "databases" in elements[0]["text"]
    assert elements[0]["page"] == 1  # mock items have no prov -> defaults to 1


def test_parser_table_element(mock_docling_result, tmp_path):
    fake_file = tmp_path / "test.pdf"
    fake_file.write_bytes(b"fake")

    with patch("core.parser.DocumentConverter") as MockConverter, \
            patch("core.parser.TableItem", _FakeTableItem):
        MockConverter.return_value.convert.return_value = mock_docling_result
        parser = Parser()
        elements = parser.parse(fake_file)

    assert elements[1]["type"] == "table"
    assert "col1" in elements[1]["text"]


def test_parser_code_element(mock_docling_result, tmp_path):
    fake_file = tmp_path / "test.pdf"
    fake_file.write_bytes(b"fake")

    with patch("core.parser.DocumentConverter") as MockConverter, \
            patch("core.parser.TableItem", _FakeTableItem):
        MockConverter.return_value.convert.return_value = mock_docling_result
        parser = Parser()
        elements = parser.parse(fake_file)

    assert elements[2]["type"] == "code"
    assert "SELECT" in elements[2]["text"]


def test_parser_raises_for_missing_file():
    with patch("core.parser.DocumentConverter"):
        parser = Parser()
    with pytest.raises(FileNotFoundError):
        parser.parse(Path("/nonexistent/file.pdf"))


def test_parser_raises_for_unsupported_format(tmp_path):
    bad_file = tmp_path / "test.xyz"
    bad_file.write_text("content")
    with patch("core.parser.DocumentConverter"):
        parser = Parser()
    with pytest.raises(ValueError, match="Unsupported"):
        parser.parse(bad_file)


def test_parser_empty_document_returns_empty_list(tmp_path):
    """Empty document (no items) returns []."""
    fake_file = tmp_path / "empty.pdf"
    fake_file.write_bytes(b"fake")

    mock_result = MagicMock()
    mock_result.document.iterate_items.return_value = []

    with patch("core.parser.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value = mock_result
        parser = Parser()
        elements = parser.parse(fake_file)

    assert elements == []


def test_parser_skips_text_items_with_empty_text(tmp_path):
    """Text items with empty or None text are excluded."""
    fake_file = tmp_path / "test.pdf"
    fake_file.write_bytes(b"fake")

    empty_item = MagicMock()
    empty_item.text = ""
    empty_item.__class__.__name__ = "TextItem"

    none_item = MagicMock()
    none_item.text = None
    none_item.__class__.__name__ = "TextItem"

    real_item = MagicMock()
    real_item.text = "Real content."
    real_item.__class__.__name__ = "TextItem"

    mock_result = MagicMock()
    mock_result.document.iterate_items.return_value = [
        (empty_item, 0), (none_item, 0), (real_item, 0)
    ]

    with patch("core.parser.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value = mock_result
        parser = Parser()
        elements = parser.parse(fake_file)

    assert len(elements) == 1
    assert elements[0]["text"] == "Real content."
