import sys
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md"}

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
    finally:
        doc.close()


class Parser:
    """
    Wraps Docling DocumentConverter.
    Returns DoclingDocument for use with HybridChunker.
    """

    def __init__(self) -> None:
        self._converter = DocumentConverter()

    def parse(self, file_path: str | Path) -> DoclingDocument:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format: {path.suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        result = self._converter.convert(str(path))
        return result.document
