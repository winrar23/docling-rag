import pytest
from unittest.mock import MagicMock, patch
from core.parser import Parser


def test_parse_returns_docling_document(tmp_path):
    """parse() should return DoclingDocument (result.document), not list[dict]."""
    fake_file = tmp_path / "test.md"
    fake_file.write_text("# Hello")

    with patch("core.parser.DocumentConverter") as MockConverter:
        mock_result = MagicMock()
        MockConverter.return_value.convert.return_value = mock_result

        parser = Parser()
        result = parser.parse(str(fake_file))

    assert result is mock_result.document


def test_parse_file_not_found_raises():
    parser = Parser()
    with pytest.raises(FileNotFoundError):
        parser.parse("/nonexistent/file.pdf")


def test_parse_unsupported_format_raises(tmp_path):
    bad_file = tmp_path / "data.csv"
    bad_file.write_text("a,b,c")
    parser = Parser()
    with pytest.raises(ValueError, match="Unsupported"):
        parser.parse(str(bad_file))
