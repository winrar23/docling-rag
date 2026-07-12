# storage/doc_registry.py
import json
import os
from datetime import datetime
from pathlib import Path


class DocRegistry:
    """
    Document-level metadata store: doc_index.json
    Keys are source_file paths (same as in metadata.json chunks).
    Implements DocumentRegistryBackend protocol.
    """

    INDEX_FILE = "doc_index.json"

    def __init__(self, data_dir: str | Path = "data") -> None:
        self._dir = Path(data_dir)

    def _index_path(self) -> Path:
        return self._dir / self.INDEX_FILE

    def _atomic_save(self, data: dict) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._dir / "doc_index.tmp.json"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._index_path())
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def load(self) -> dict[str, dict]:
        """Return full index. Returns {} if file does not exist."""
        path = self._index_path()
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def upsert(
        self,
        source_file: str,
        title: str | None,
        topic: str | None,
        tags: list[str],
    ) -> None:
        """Add or update entry. Preserves added_at if entry already exists."""
        data = self.load()
        existing = data.get(source_file, {})
        data[source_file] = {
            "title": title,
            "topic": topic,
            "tags": tags,
            "added_at": existing.get("added_at", datetime.now().isoformat(timespec="seconds")),
        }
        self._atomic_save(data)

    def get(self, source_file: str) -> dict | None:
        return self.load().get(source_file)

    def delete(self, source_file: str) -> None:
        data = self.load()
        if source_file in data:
            del data[source_file]
            self._atomic_save(data)
