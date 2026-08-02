import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import RapidOcrOptions

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


def _write_pdf(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(make_text_pdf(_LONG, pages=1))
    return p


def _pdf_opts(mock_converter_cls):
    fmt = mock_converter_cls.call_args.kwargs["format_options"][InputFormat.PDF]
    return fmt.pipeline_options


class TestParserOcrModes:
    def test_ocr_on_forces_do_ocr(self, tmp_path):
        with patch("docling_rag.core.parser.DocumentConverter") as MC:
            Parser().parse(str(_write_pdf(tmp_path)), ocr="on")
        assert _pdf_opts(MC).do_ocr is True

    def test_ocr_off_disables_do_ocr(self, tmp_path):
        with patch("docling_rag.core.parser.DocumentConverter") as MC:
            Parser().parse(str(_write_pdf(tmp_path)), ocr="off")
        assert _pdf_opts(MC).do_ocr is False

    def test_auto_with_text_layer_disables_ocr(self, tmp_path):
        with patch("docling_rag.core.parser.DocumentConverter") as MC, \
             patch("docling_rag.core.parser._has_text_layer", return_value=(True, 8, 10)):
            Parser().parse(str(_write_pdf(tmp_path)), ocr="auto")
        assert _pdf_opts(MC).do_ocr is False

    def test_auto_without_text_layer_enables_ocr(self, tmp_path):
        with patch("docling_rag.core.parser.DocumentConverter") as MC, \
             patch("docling_rag.core.parser._has_text_layer", return_value=(False, 0, 10)):
            Parser().parse(str(_write_pdf(tmp_path)), ocr="auto")
        assert _pdf_opts(MC).do_ocr is True

    def test_ru_lang_sets_cyrillic_rapidocr_params(self, tmp_path):
        with patch("docling_rag.core.parser.DocumentConverter") as MC:
            Parser().parse(str(_write_pdf(tmp_path)), ocr="on", ocr_lang="ru")
        ocr_opts = _pdf_opts(MC).ocr_options
        assert isinstance(ocr_opts, RapidOcrOptions)
        assert ocr_opts.backend == "torch"
        assert ocr_opts.rapidocr_params == {"Rec.lang_type": "cyrillic"}

    def test_en_lang_keeps_default_ocr_options(self, tmp_path):
        with patch("docling_rag.core.parser.DocumentConverter") as MC:
            Parser().parse(str(_write_pdf(tmp_path)), ocr="on", ocr_lang="en")
        assert not isinstance(_pdf_opts(MC).ocr_options, RapidOcrOptions)

    def test_invalid_ocr_mode_raises(self, tmp_path):
        with pytest.raises(ValueError, match="ocr mode"):
            Parser().parse(str(_write_pdf(tmp_path)), ocr="always")

    def test_invalid_ocr_lang_raises(self, tmp_path):
        with pytest.raises(ValueError, match="ocr_lang"):
            Parser().parse(str(_write_pdf(tmp_path)), ocr="on", ocr_lang="de")

    def test_md_skips_detection_and_uses_shared_converter(self, tmp_path):
        md = tmp_path / "note.md"
        md.write_text("# Hello")
        with patch("docling_rag.core.parser.DocumentConverter") as MC, \
             patch("docling_rag.core.parser._has_text_layer") as mock_detect:
            p = Parser()
            p.parse(str(md), ocr="auto", ocr_lang="ru")
            p.parse(str(_write_pdf(tmp_path)), ocr="on", ocr_lang="en")
        mock_detect.assert_not_called()   # для md детект не запускается
        assert MC.call_count == 1          # md и on+en делят один конвертер

    def test_converter_cache_reused_per_mode(self, tmp_path):
        pdf = _write_pdf(tmp_path)
        with patch("docling_rag.core.parser.DocumentConverter") as MC:
            p = Parser()
            p.parse(str(pdf), ocr="on")
            p.parse(str(pdf), ocr="on")
            p.parse(str(pdf), ocr="off")
        assert MC.call_count == 2  # on+en и off; повторный on — из кеша

    def test_off_cache_ignores_lang(self, tmp_path):
        pdf = _write_pdf(tmp_path)
        with patch("docling_rag.core.parser.DocumentConverter") as MC:
            p = Parser()
            p.parse(str(pdf), ocr="off", ocr_lang="en")
            p.parse(str(pdf), ocr="off", ocr_lang="ru")
        assert MC.call_count == 1  # off от языка не зависит

    def test_auto_logs_decision_to_stderr(self, tmp_path, capsys):
        with patch("docling_rag.core.parser.DocumentConverter"), \
             patch("docling_rag.core.parser._has_text_layer", return_value=(True, 8, 10)):
            Parser().parse(str(_write_pdf(tmp_path)), ocr="auto")
        err = capsys.readouterr().err
        assert "OCR: off" in err and "8/10" in err
