# storage/db_search_log.py
"""Лог поисковых запросов в PostgreSQL. Реализует SearchLogBackend (core/protocols.py)."""
import psycopg

from docling_rag.storage.db_storage import _translate_db_errors


class DBSearchLog:
    """Соединение открывается на операцию — как в DBStorage/DBRegistry."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def log(self, query: str, top_score: float) -> None:
        with _translate_db_errors(), psycopg.connect(self._dsn) as conn:
            conn.execute(
                "INSERT INTO searches (query, top_score) VALUES (%s, %s)",
                (query, float(top_score)),
            )
            conn.commit()
