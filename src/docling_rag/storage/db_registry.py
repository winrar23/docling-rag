# storage/db_registry.py
"""Реестр документов в PostgreSQL. Реализует DocumentRegistryBackend (core/protocols.py)."""
import psycopg

from docling_rag.storage.db_storage import _translate_db_errors

_EDITABLE_FIELDS = ("title", "author", "topic", "tags")


def _row_to_entry(row) -> dict:
    id_, title, author, topic, tags, added_at = row
    return {
        "id": str(id_), "title": title, "author": author, "topic": topic, "tags": list(tags),
        "added_at": added_at.isoformat(timespec="seconds"),
    }


class DBRegistry:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def upsert(
        self,
        source_file: str,
        title: str | None,
        topic: str | None,
        tags: list[str],
        author: str | None = None,
    ) -> None:
        # Контракт upsert: added_at сохраняется, None/пустые значения не затирают существующие
        with _translate_db_errors(), psycopg.connect(self._dsn) as conn:
            conn.execute(
                """
                INSERT INTO documents (source_file, title, author, topic, tags)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (source_file) DO UPDATE SET
                    title  = COALESCE(EXCLUDED.title, documents.title),
                    author = COALESCE(EXCLUDED.author, documents.author),
                    topic  = COALESCE(EXCLUDED.topic, documents.topic),
                    tags   = CASE WHEN cardinality(EXCLUDED.tags) > 0
                                  THEN EXCLUDED.tags ELSE documents.tags END
                """,
                (source_file, title, author, topic, list(tags)),
            )
            conn.commit()

    def delete(self, source_file: str) -> None:
        with _translate_db_errors(), psycopg.connect(self._dsn) as conn:
            conn.execute("DELETE FROM documents WHERE source_file = %s", (source_file,))
            conn.commit()

    def get(self, source_file: str) -> dict | None:
        with _translate_db_errors(), psycopg.connect(self._dsn) as conn:
            row = conn.execute(
                "SELECT id, title, author, topic, tags, added_at FROM documents WHERE source_file = %s",
                (source_file,),
            ).fetchone()
        return _row_to_entry(row) if row else None

    def load(self) -> dict[str, dict]:
        with _translate_db_errors(), psycopg.connect(self._dsn) as conn:
            rows = conn.execute(
                "SELECT source_file, id, title, author, topic, tags, added_at FROM documents ORDER BY source_file"
            ).fetchall()
        return {r[0]: _row_to_entry(r[1:]) for r in rows}

    def get_by_id(self, doc_id: str) -> tuple[str, dict] | None:
        import uuid as _uuid
        try:
            _uuid.UUID(str(doc_id))
        except (ValueError, TypeError):
            return None  # malformed -> не найдено (эндпоинт отдаст 404)
        with _translate_db_errors(), psycopg.connect(self._dsn) as conn:
            row = conn.execute(
                "SELECT source_file, id, title, author, topic, tags, added_at"
                " FROM documents WHERE id = %s::uuid", (doc_id,),
            ).fetchone()
        return (row[0], _row_to_entry(row[1:])) if row else None

    def update_metadata(self, source_file: str, fields: dict) -> dict | None:
        """Явный SET переданных полей (правка, в т.ч. очистка) — НЕ COALESCE-upsert индексатора."""
        updates = {k: fields[k] for k in _EDITABLE_FIELDS if k in fields}
        if "tags" in updates:
            updates["tags"] = list(updates["tags"] or [])  # колонка NOT NULL: None → []
        if not updates:
            return self.get(source_file)
        set_sql = ", ".join(f"{k} = %s" for k in updates)  # ключи из белого списка — не инъекция
        with _translate_db_errors(), psycopg.connect(self._dsn) as conn:
            row = conn.execute(
                f"UPDATE documents SET {set_sql} WHERE source_file = %s"
                " RETURNING id, title, author, topic, tags, added_at",
                (*updates.values(), source_file),
            ).fetchone()
            conn.commit()
        return _row_to_entry(row) if row else None
