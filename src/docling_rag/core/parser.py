import sys
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md"}

OCR_MODES = ("auto", "on", "off")
OCR_LANGS = ("en", "ru")

# Детект текстового слоя (ocr=auto): до 10 страниц равномерно по документу;
# страница «текстовая» при >=100 символах; цифровой PDF при доле >= 0.5.
_DETECT_MAX_PAGES = 10
_DETECT_MIN_CHARS = 100
_DETECT_TEXT_RATIO = 0.5


def _has_text_layer(path: Path) -> tuple[bool, int, int]:
    """(есть текстовый слой, страниц с текстом, страниц проверено).

    Ошибки чтения (битый/зашифрованный PDF) -> (False, 0, 0): считаем сканом,
    OCR включится — лучше медленно, чем пусто.
    """
    import pypdfium2 as pdfium  # транзитивная зависимость docling

    try:
        doc = pdfium.PdfDocument(str(path))
    except Exception:
        return False, 0, 0
    try:
        try:
            n = len(doc)
            if n == 0:
                return False, 0, 0
            count = min(n, _DETECT_MAX_PAGES)
            step = (n - 1) / max(count - 1, 1)
            indices = sorted({round(i * step) for i in range(count)})
            text_pages = 0
            for idx in indices:
                try:
                    text = doc[idx].get_textpage().get_text_range()
                except Exception:
                    continue  # нечитаемая страница = страница без текста
                if len(text.strip()) >= _DETECT_MIN_CHARS:
                    text_pages += 1
            return (text_pages / len(indices)) >= _DETECT_TEXT_RATIO, text_pages, len(indices)
        except Exception:
            return False, 0, 0
    finally:
        doc.close()


def _build_converter(do_ocr: bool, ocr_lang: str) -> DocumentConverter:
    opts = PdfPipelineOptions(do_ocr=do_ocr)
    if do_ocr and ocr_lang == "ru":
        # Кириллица: docling-обёртка мапит lang только на english/latin/chinese,
        # но rapidocr_params уходят в RapidOCR как есть и перекрывают дефолты.
        opts.ocr_options = RapidOcrOptions(
            backend="torch", rapidocr_params={"Rec.lang_type": "cyrillic"}
        )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


class Parser:
    """
    Wraps Docling DocumentConverter.
    Returns DoclingDocument for use with HybridChunker.

    Конвертеры кешируются per (do_ocr, ocr_lang) — максимум три комбинации
    (off / on+en / on+ru); создание DocumentConverter греет модели, это дорого.
    """

    def __init__(self) -> None:
        self._converters: dict[tuple[bool, str], DocumentConverter] = {}

    def _converter(self, do_ocr: bool, ocr_lang: str) -> DocumentConverter:
        key = (do_ocr, ocr_lang if do_ocr else "en")  # off от языка не зависит
        if key not in self._converters:
            self._converters[key] = _build_converter(*key)
        return self._converters[key]

    def parse(self, file_path: str | Path, ocr: str = "auto", ocr_lang: str = "en") -> DoclingDocument:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format: {path.suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        if ocr not in OCR_MODES:
            raise ValueError(f"Unknown ocr mode: {ocr!r}. Supported: {', '.join(OCR_MODES)}")
        if ocr_lang not in OCR_LANGS:
            raise ValueError(f"Unknown ocr_lang: {ocr_lang!r}. Supported: {', '.join(OCR_LANGS)}")

        do_ocr = True
        if path.suffix.lower() == ".pdf":
            if ocr == "auto":
                has_text, text_pages, sampled = _has_text_layer(path)
                do_ocr = not has_text
                state = ("off — обнаружен текстовый слой" if has_text
                         else "on — текстовый слой не обнаружен")
                print(f"OCR: {state} ({text_pages}/{sampled} страниц с текстом)", file=sys.stderr)
            else:
                do_ocr = ocr == "on"
        else:
            ocr_lang = "en"  # OCR-опции не влияют на md/docx — общий конвертер on+en

        result = self._converter(do_ocr, ocr_lang).convert(str(path))
        return result.document
