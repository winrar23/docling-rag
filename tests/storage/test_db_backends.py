"""Integration-тесты DBStorage/DBRegistry/DBSearchLog против реального postgres (compose).

Прекондишн: docker compose up -d postgres. Используется отдельная БД docling_rag_test.
"""
import os

import numpy as np
import pytest

from docling_rag.core.chunker import Chunk

psycopg = pytest.importorskip("psycopg")

pytestmark = pytest.mark.integration

TEST_DB = "docling_rag_test"
DEFAULT_TEST_URL = f"postgresql://docling:docling@127.0.0.1:5432/{TEST_DB}"


@pytest.fixture(scope="session")
def db_url():
    url = os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_URL)
    admin_url = url.rsplit("/", 1)[0] + "/postgres"
    try:
        with psycopg.connect(admin_url, autocommit=True) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)
            ).fetchone()
            if not exists:
                conn.execute(f'CREATE DATABASE "{TEST_DB}"')
    except psycopg.OperationalError:
        pytest.skip("postgres недоступен — запустите: docker compose up -d postgres")
    from docling_rag.storage.db_schema import init_schema
    init_schema(url)
    return url


@pytest.fixture
def clean_db(db_url):
    with psycopg.connect(db_url) as conn:
        conn.execute("TRUNCATE documents CASCADE")
        conn.execute("TRUNCATE searches")
        conn.execute("TRUNCATE jobs")
        conn.commit()
    return db_url


def _vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(1024, dtype=np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _chunk(source: str, chunk_id: int, text: str = "hello") -> Chunk:
    return Chunk(
        text=text, source_file=source, chunk_id=chunk_id, page_number=chunk_id + 1,
        element_type="text", headings=["H1", "H2"], context_text=f"H1 > H2\n{text}",
    )


class TestDBStorage:
    def _storage(self, url):
        from docling_rag.storage.db_storage import DBStorage
        return DBStorage(url)

    def test_load_empty_raises_file_not_found(self, clean_db):
        with pytest.raises(FileNotFoundError):
            self._storage(clean_db).load()

    def test_search_empty_raises_file_not_found(self, clean_db):
        with pytest.raises(FileNotFoundError):
            self._storage(clean_db).search(_vec(1), top_k=3)

    def test_append_load_roundtrip_preserves_metadata_shape(self, clean_db):
        s = self._storage(clean_db)
        chunks = [_chunk("/books/a.pdf", 0), _chunk("/books/a.pdf", 1, "world")]
        s.append(chunks, np.stack([_vec(1), _vec(2)]))
        emb, meta = s.load()
        assert emb.shape == (2, 1024)
        assert emb.dtype == np.float32
        assert meta[0] == {
            "text": "hello", "source_file": "/books/a.pdf", "chunk_id": 0,
            "page_number": 1, "element_type": "text", "headings": ["H1", "H2"],
        }

    def test_append_creates_parent_document_row(self, clean_db):
        # indexer вызывает storage.append ДО registry.upsert — FK не должен падать
        s = self._storage(clean_db)
        s.append([_chunk("/books/a.pdf", 0)], _vec(1).reshape(1, -1))
        with psycopg.connect(clean_db) as conn:
            row = conn.execute(
                "SELECT source_file FROM documents WHERE source_file = %s",
                ("/books/a.pdf",),
            ).fetchone()
        assert row is not None

    def test_length_mismatch_raises_value_error(self, clean_db):
        with pytest.raises(ValueError):
            self._storage(clean_db).append([_chunk("/books/a.pdf", 0)], np.stack([_vec(1), _vec(2)]))

    def test_delete_by_source_and_idempotent_readd(self, clean_db):
        s = self._storage(clean_db)
        s.append([_chunk("/books/a.pdf", 0), _chunk("/books/b.pdf", 0)],
                 np.stack([_vec(1), _vec(2)]))
        s.delete_by_source("/books/a.pdf")
        s.append([_chunk("/books/a.pdf", 0)], _vec(3).reshape(1, -1))
        _, meta = s.load()
        assert sorted(m["source_file"] for m in meta) == ["/books/a.pdf", "/books/b.pdf"]
        s.delete_by_source("/books/nope.pdf")  # no-op, не падает

    def test_count_by_source(self, clean_db):
        s = self._storage(clean_db)
        s.append([_chunk("/books/a.pdf", 0), _chunk("/books/a.pdf", 1)],
                 np.stack([_vec(1), _vec(2)]))
        assert s.count_by_source("/books/a.pdf") == 2
        assert s.count_by_source("/books/nope.pdf") == 0

    def test_search_orders_by_similarity_and_respects_top_k(self, clean_db):
        s = self._storage(clean_db)
        q = _vec(42)
        near = (q + 0.01 * _vec(7)); near = (near / np.linalg.norm(near)).astype(np.float32)
        far = _vec(1000)
        s.append([_chunk("/books/a.pdf", 0, "near"), _chunk("/books/a.pdf", 1, "far")],
                 np.stack([near, far]))
        results = s.search(q, top_k=1)
        assert len(results) == 1
        meta, score = results[0]
        assert meta["text"] == "near"
        assert 0.9 < score <= 1.0001

    def test_search_allowed_sources_filter(self, clean_db):
        s = self._storage(clean_db)
        s.append([_chunk("/books/a.pdf", 0), _chunk("/books/b.pdf", 0)],
                 np.stack([_vec(1), _vec(2)]))
        results = s.search(_vec(3), top_k=5, allowed_sources={"/books/b.pdf"})
        assert [m["source_file"] for m, _ in results] == ["/books/b.pdf"]
        assert s.search(_vec(3), top_k=5, allowed_sources=set()) == []


class TestDBRegistry:
    def _registry(self, url):
        from docling_rag.storage.db_registry import DBRegistry
        return DBRegistry(url)

    def test_get_missing_returns_none_and_load_empty_dict(self, clean_db):
        r = self._registry(clean_db)
        assert r.get("/books/nope.pdf") is None
        assert r.load() == {}

    def test_upsert_and_get_shape(self, clean_db):
        r = self._registry(clean_db)
        r.upsert("/books/a.pdf", title="T", topic="arch", tags=["a", "b"])
        entry = r.get("/books/a.pdf")
        assert entry["title"] == "T" and entry["topic"] == "arch" and entry["tags"] == ["a", "b"]
        assert isinstance(entry["added_at"], str) and "T" in entry["added_at"]

    def test_upsert_preserves_added_at_and_does_not_wipe(self, clean_db):
        r = self._registry(clean_db)
        r.upsert("/books/a.pdf", title="T", topic="arch", tags=["a"])
        first = r.get("/books/a.pdf")["added_at"]
        r.upsert("/books/a.pdf", title=None, topic=None, tags=[])
        entry = r.get("/books/a.pdf")
        assert entry["title"] == "T" and entry["topic"] == "arch"
        assert entry["tags"] == ["a"] and entry["added_at"] == first

    def test_delete_cascades_chunks(self, clean_db):
        from docling_rag.storage.db_storage import DBStorage
        r, s = self._registry(clean_db), DBStorage(clean_db)
        s.append([_chunk("/books/a.pdf", 0)], _vec(1).reshape(1, -1))
        r.upsert("/books/a.pdf", title=None, topic=None, tags=[])
        r.delete("/books/a.pdf")
        assert r.get("/books/a.pdf") is None
        assert s.count_by_source("/books/a.pdf") == 0
        r.delete("/books/a.pdf")  # идемпотентно

    def test_db_registry_author_roundtrip_and_coalesce(self, clean_db):
        r = self._registry(clean_db)
        r.upsert("/b.pdf", title="T", topic="db", tags=["a"], author="Иванов И.")
        assert r.get("/b.pdf")["author"] == "Иванов И."
        assert r.load()["/b.pdf"]["author"] == "Иванов И."
        r.upsert("/b.pdf", title=None, topic=None, tags=[], author=None)
        assert r.get("/b.pdf")["author"] == "Иванов И."

    def test_db_registry_update_metadata_partial_and_clear(self, clean_db):
        r = self._registry(clean_db)
        r.upsert("/b.pdf", title="T", topic="db", tags=["a"], author="A")
        entry = r.update_metadata("/b.pdf", {"title": "T2", "topic": None})
        assert entry["title"] == "T2"
        assert entry["topic"] is None          # None очищает
        assert entry["author"] == "A"          # не переданное — не тронуто
        assert entry["tags"] == ["a"]
        assert r.update_metadata("/b.pdf", {"tags": None})["tags"] == []  # None для tags → []
        assert r.update_metadata("/nope.pdf", {"title": "X"}) is None


def test_documents_have_uuid_id_and_get_by_id(clean_db):
    from docling_rag.storage.db_registry import DBRegistry
    reg = DBRegistry(clean_db)
    reg.upsert("/uploads/x.pdf", "X", None, ["t"])
    entry = reg.get("/uploads/x.pdf")
    assert entry["id"]
    source, by_id = reg.get_by_id(entry["id"])
    assert source == "/uploads/x.pdf" and by_id["title"] == "X"
    assert reg.get_by_id("not-a-uuid") is None


def test_init_schema_adds_id_to_legacy_documents(db_url):
    """Миграция: база без колонки id получает её повторным init_schema (идемпотентно)."""
    import psycopg
    from docling_rag.storage.db_schema import init_schema
    with psycopg.connect(db_url) as conn:
        conn.execute("INSERT INTO documents (source_file, title) VALUES ('/legacy.pdf', 'L')"
                     " ON CONFLICT (source_file) DO NOTHING")
        conn.execute("DROP INDEX IF EXISTS documents_id_key")
        conn.execute("ALTER TABLE documents DROP COLUMN IF EXISTS id")
        conn.commit()
    init_schema(db_url)
    with psycopg.connect(db_url) as conn:
        row = conn.execute("SELECT id FROM documents WHERE source_file='/legacy.pdf'").fetchone()
        conn.execute("DELETE FROM documents WHERE source_file='/legacy.pdf'")
        conn.commit()
    assert row[0] is not None  # backfill сработал


@pytest.mark.integration
def test_schema_has_author_and_warning_columns(db_url):
    """Миграция: author в documents, warning в jobs. Идемпотентна при повторном init."""
    import psycopg

    from docling_rag.storage.db_schema import init_schema

    init_schema(db_url)  # повторный init — миграция идемпотентна
    with psycopg.connect(db_url) as conn:
        cols = {
            r[0] for r in conn.execute(
                "SELECT column_name FROM information_schema.columns"
                " WHERE table_name = 'documents'"
            ).fetchall()
        }
        job_cols = {
            r[0] for r in conn.execute(
                "SELECT column_name FROM information_schema.columns"
                " WHERE table_name = 'jobs'"
            ).fetchall()
        }
    assert "author" in cols
    assert "warning" in job_cols


class TestDomainErrorTranslation:
    """Реальный psycopg-стек -> доменные исключения (core/errors.py)."""

    UNREACHABLE_URL = "postgresql://test:test@127.0.0.1:1/test"  # порт 1 — fail fast

    def test_storage_load_unreachable_raises_storage_unavailable(self):
        from docling_rag.core.errors import StorageUnavailableError
        from docling_rag.storage.db_storage import DBStorage
        with pytest.raises(StorageUnavailableError):
            DBStorage(self.UNREACHABLE_URL).load()

    def test_registry_load_unreachable_raises_storage_unavailable(self):
        from docling_rag.core.errors import StorageUnavailableError
        from docling_rag.storage.db_registry import DBRegistry
        with pytest.raises(StorageUnavailableError):
            DBRegistry(self.UNREACHABLE_URL).load()

    def test_init_schema_unreachable_raises_storage_unavailable(self):
        from docling_rag.core.errors import StorageUnavailableError
        from docling_rag.storage.db_schema import init_schema
        with pytest.raises(StorageUnavailableError):
            init_schema(self.UNREACHABLE_URL)

    def test_missing_chunks_table_raises_schema_missing(self, db_url):
        from docling_rag.core.errors import StorageSchemaMissingError
        from docling_rag.storage.db_schema import init_schema
        from docling_rag.storage.db_storage import DBStorage
        with psycopg.connect(db_url) as conn:
            conn.execute("DROP TABLE IF EXISTS chunks CASCADE")
            conn.commit()
        try:
            with pytest.raises(StorageSchemaMissingError):
                DBStorage(db_url).load()
        finally:
            init_schema(db_url)  # восстановить схему для соседних тестов (идемпотентно)

    def test_missing_column_raises_schema_missing(self, db_url):
        """Непромигрированная БД (колонка отсутствует) — живая приёмка поймала здесь
        сырой psycopg.errors.UndefinedColumn -> 500 вместо понятной подсказки."""
        from docling_rag.core.errors import StorageSchemaMissingError
        from docling_rag.storage.db_schema import init_schema
        from docling_rag.storage.db_storage import DBStorage
        with psycopg.connect(db_url) as conn:
            conn.execute("ALTER TABLE chunks DROP COLUMN element_type")
            conn.commit()
        try:
            with pytest.raises(StorageSchemaMissingError):
                DBStorage(db_url).load()
        finally:
            with psycopg.connect(db_url) as conn:
                conn.execute(
                    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS element_type"
                    " text NOT NULL DEFAULT 'text'"
                )
                conn.commit()  # init_schema не восстановит колонку: CREATE TABLE IF NOT EXISTS — no-op на существующей таблице


class TestDBSearchLog:
    """Лог поисковых запросов в БД (заменил файловый лог: в docker он умирал с контейнером)."""

    def _log(self, url):
        from docling_rag.storage.db_search_log import DBSearchLog
        return DBSearchLog(url)

    def test_log_writes_row_with_query_and_score(self, clean_db):
        self._log(clean_db).log("что такое data vault", 0.87)
        with psycopg.connect(clean_db) as conn:
            row = conn.execute("SELECT query, top_score, searched_at FROM searches").fetchone()
        assert row[0] == "что такое data vault"
        assert row[1] == pytest.approx(0.87, abs=1e-4)
        assert row[2] is not None  # searched_at проставляет БД (DEFAULT now())

    def test_log_appends_and_preserves_order(self, clean_db):
        log = self._log(clean_db)
        log.log("первый", 0.5)
        log.log("второй", 0.6)
        with psycopg.connect(clean_db) as conn:
            rows = conn.execute("SELECT query FROM searches ORDER BY id").fetchall()
        assert [r[0] for r in rows] == ["первый", "второй"]

    def test_log_unreachable_raises_storage_unavailable(self):
        from docling_rag.core.errors import StorageUnavailableError
        with pytest.raises(StorageUnavailableError):
            self._log("postgresql://test:test@127.0.0.1:1/test").log("q", 0.1)
