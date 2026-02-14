from pathlib import Path
from typing import Any

from docling.datamodel.document import TableItem
from docling.document_converter import DocumentConverter

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}


def _get_page(item: Any) -> int:
    """Extract page number from Docling item provenance, defaulting to 1."""
    try:
        prov = getattr(item, "prov", None)
        if prov and len(prov) > 0:
            return int(prov[0].page_no)
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    return 1


class Parser:
    """
    Wraps Docling DocumentConverter.
    Returns normalized elements: {"text": str, "type": str, "page": int}
    """

    def __init__(self) -> None:
        self._converter = DocumentConverter()

    def parse(self, file_path: str | Path) -> list[dict[str, Any]]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format: {path.suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        result = self._converter.convert(str(path))
        doc = result.document

        elements: list[dict[str, Any]] = []

        for item, _level in doc.iterate_items():
            if isinstance(item, TableItem):
                try:
                    text = item.export_to_markdown()
                except Exception:
                    continue  # skip malformed table rather than storing garbage
                elements.append({"text": text, "type": "table", "page": _get_page(item)})

            elif item.__class__.__name__ == "CodeItem":
                # CodeItem is not exported from docling.datamodel.document, so we
                # fall back to class name comparison rather than isinstance.
                text = getattr(item, "text", str(item))
                elements.append({"text": text, "type": "code", "page": _get_page(item)})

            elif hasattr(item, "text") and item.text:
                elements.append({"text": item.text, "type": "text", "page": _get_page(item)})

        return elements
