from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}


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
            class_name = item.__class__.__name__

            if class_name == "TableItem":
                try:
                    text = item.export_to_markdown()
                except Exception:
                    text = str(item)
                elements.append({"text": text, "type": "table", "page": 1})

            elif class_name == "CodeItem":
                text = getattr(item, "text", str(item))
                elements.append({"text": text, "type": "code", "page": 1})

            elif hasattr(item, "text") and item.text:
                elements.append({"text": item.text, "type": "text", "page": 1})

        return elements
