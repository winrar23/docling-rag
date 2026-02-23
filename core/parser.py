from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}


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
