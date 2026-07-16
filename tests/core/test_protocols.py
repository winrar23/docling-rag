from docling_rag.core.protocols import DocumentRegistryBackend, SearchLogBackend, StorageBackend
from docling_rag.storage.db_registry import DBRegistry
from docling_rag.storage.db_search_log import DBSearchLog
from docling_rag.storage.db_storage import DBStorage

# Конструирование с фиктивным dsn НЕ подключается к БД — соединение открывается на операцию.
_FAKE_DSN = "postgresql://test:test@127.0.0.1:1/test"


def test_db_storage_conforms():
    s: StorageBackend = DBStorage(_FAKE_DSN)  # статическая + структурная проверка
    assert s is not None


def test_db_registry_conforms():
    r: DocumentRegistryBackend = DBRegistry(_FAKE_DSN)
    assert r is not None


def test_db_search_log_conforms():
    log: SearchLogBackend = DBSearchLog(_FAKE_DSN)
    assert log is not None


def test_core_does_not_import_storage_package():
    import inspect
    from pathlib import Path

    import docling_rag.core
    import docling_rag.core.indexer as idx
    import docling_rag.core.search as s

    assert "from docling_rag.storage" not in inspect.getsource(s)
    assert "from docling_rag.storage" not in inspect.getsource(idx)

    # agent.py requires pydantic_ai (optional `.[agent]` extra), which may not be
    # installed in every environment. Read its source from disk instead of
    # importing it, so this check runs unconditionally without erroring on
    # installs without pydantic_ai.
    agent_path = Path(docling_rag.core.__file__).parent / "agent.py"
    agent_source = agent_path.read_text(encoding="utf-8")
    assert "from docling_rag.storage" not in agent_source
