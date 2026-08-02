import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from docling_rag.core.parser import Parser, _has_text_layer
from tests.pdf_fixtures import make_empty_pdf, make_text_pdf


def test_parse_returns_docling_document(tmp_path):
    """parse() should return DoclingDocument (result.document), not list[dict]."""
    fake_file = tmp_path / "test.md"
    fake_file.write_text("# Hello")

    with patch("docling_rag.core.parser.DocumentConverter") as MockConverter:
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


def test_txt_not_supported():
    from docling_rag.core.parser import SUPPORTED_EXTENSIONS
    assert ".txt" not in SUPPORTED_EXTENSIONS


_LONG = "This page contains a long extractable text layer for detection purposes, ok. " * 2  # >100 симв.


class TestHasTextLayer:
    def test_digital_pdf_detected(self, tmp_path):
        p = tmp_path / "digital.pdf"
        p.write_bytes(make_text_pdf(_LONG, pages=3))
        assert _has_text_layer(p) == (True, 3, 3)

    def test_scan_pdf_not_detected(self, tmp_path):
        p = tmp_path / "scan.pdf"
        p.write_bytes(make_empty_pdf(pages=3))
        assert _has_text_layer(p) == (False, 0, 3)

    def test_short_text_below_min_chars_counts_as_scan(self, tmp_path):
        p = tmp_path / "stub.pdf"
        p.write_bytes(make_text_pdf("short", pages=2))  # < 100 символов на странице
        assert _has_text_layer(p)[0] is False

    def test_samples_at_most_10_pages(self, tmp_path):
        p = tmp_path / "big.pdf"
        p.write_bytes(make_text_pdf(_LONG, pages=15))
        has_text, text_pages, sampled = _has_text_layer(p)
        assert has_text is True and sampled == 10 and text_pages == 10

    def test_broken_file_counts_as_scan(self, tmp_path):
        p = tmp_path / "broken.pdf"
        p.write_bytes(b"definitely not a pdf")
        assert _has_text_layer(p) == (False, 0, 0)

    def test_zero_page_pdf_counts_as_scan(self, tmp_path):
        p = tmp_path / "zero.pdf"
        p.write_bytes(make_empty_pdf(pages=0))
        assert _has_text_layer(p) == (False, 0, 0)

    def test_len_doc_raises_counts_as_scan(self, tmp_path):
        """Corrupted PDF with broken page tree: len(doc) raises -> (False, 0, 0)."""
        p = tmp_path / "broken_tree.pdf"
        p.write_bytes(make_text_pdf(_LONG, pages=1))  # Valid file for file I/O

        with patch("pypdfium2.PdfDocument") as MockDoc:
            mock_doc = MagicMock()
            mock_doc.__len__.side_effect = Exception("Broken page tree")
            MockDoc.return_value = mock_doc

            assert _has_text_layer(p) == (False, 0, 0)
            mock_doc.close.assert_called_once()
