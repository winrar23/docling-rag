# storage/db_registry.py
"""Реестр документов в PostgreSQL. Реализует DocumentRegistryBackend (core/protocols.py)."""
import psycopg


def _row_to_entry(row) -> dict:
    title, topic, tags, added_at = row
    return {
        "title": title, "topic": topic, "tags": list(tags),
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
    ) -> None:
        # Семантика DocRegistry (MVP): added_at сохраняется, None/пустые значения не затирают существующие
        with psycopg.connect(self._dsn) as conn:
            conn.execute(
                """
                INSERT INTO documents (source_file, title, topic, tags)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (source_file) DO UPDATE SET
                    title = COALESCE(EXCLUDED.title, documents.title),
                    topic = COALESCE(EXCLUDED.topic, documents.topic),
                    tags  = CASE WHEN cardinality(EXCLUDED.tags) > 0
                                 THEN EXCLUDED.tags ELSE documents.tags END
                """,
                (source_file, title, topic, list(tags)),
            )
            conn.commit()

    def delete(self, source_file: str) -> None:
        with psycopg.connect(self._dsn) as conn:
            conn.execute("DELETE FROM documents WHERE source_file = %s", (source_file,))
            conn.commit()

    def get(self, source_file: str) -> dict | None:
        with psycopg.connect(self._dsn) as conn:
            row = conn.execute(
                "SELECT title, topic, tags, added_at FROM documents WHERE source_file = %s",
                (source_file,),
            ).fetchone()
        return _row_to_entry(row) if row else None

    def load(self) -> dict[str, dict]:
        with psycopg.connect(self._dsn) as conn:
            rows = conn.execute(
                "SELECT source_file, title, topic, tags, added_at FROM documents ORDER BY source_file"
            ).fetchall()
        return {r[0]: _row_to_entry(r[1:]) for r in rows}
