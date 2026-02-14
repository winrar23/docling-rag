from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.parser import Parser


@pytest.fixture
def mock_docling_result():
    """Mock result from DocumentConverter.convert()"""
    mock_result = MagicMock()
    mock_doc = MagicMock()

    text_item = MagicMock()
    text_item.text = "This is a paragraph about databases."
    text_item.__class__.__name__ = "TextItem"

    table_item = MagicMock()
    table_item.export_to_markdown.return_value = "| col1 | col2 |\n|------|------|\n| A    | B    |"
    table_item.__class__.__name__ = "TableItem"

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

    with patch("core.parser.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value = mock_docling_result
        parser = Parser()
        elements = parser.parse(fake_file)

    assert isinstance(elements, list)
    assert len(elements) == 3


def test_parser_text_element(mock_docling_result, tmp_path):
    fake_file = tmp_path / "test.pdf"
    fake_file.write_bytes(b"fake")

    with patch("core.parser.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value = mock_docling_result
        parser = Parser()
        elements = parser.parse(fake_file)

    assert elements[0]["type"] == "text"
    assert "databases" in elements[0]["text"]


def test_parser_table_element(mock_docling_result, tmp_path):
    fake_file = tmp_path / "test.pdf"
    fake_file.write_bytes(b"fake")

    with patch("core.parser.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value = mock_docling_result
        parser = Parser()
        elements = parser.parse(fake_file)

    assert elements[1]["type"] == "table"
    assert "col1" in elements[1]["text"]


def test_parser_code_element(mock_docling_result, tmp_path):
    fake_file = tmp_path / "test.pdf"
    fake_file.write_bytes(b"fake")

    with patch("core.parser.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value = mock_docling_result
        parser = Parser()
        elements = parser.parse(fake_file)

    assert elements[2]["type"] == "code"
    assert "SELECT" in elements[2]["text"]


def test_parser_raises_for_missing_file():
    parser = Parser()
    with pytest.raises(FileNotFoundError):
        parser.parse(Path("/nonexistent/file.pdf"))


def test_parser_raises_for_unsupported_format(tmp_path):
    bad_file = tmp_path / "test.xyz"
    bad_file.write_text("content")
    parser = Parser()
    with pytest.raises(ValueError, match="Unsupported"):
        parser.parse(bad_file)
