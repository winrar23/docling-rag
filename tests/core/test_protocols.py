from docling_rag.core.protocols import StorageBackend, DocumentRegistryBackend
from docling_rag.storage.file_storage import FileStorage
from docling_rag.storage.doc_registry import DocRegistry


def test_file_storage_conforms(tmp_path):
    s: StorageBackend = FileStorage(data_dir=tmp_path)  # статическая + структурная проверка
    assert s is not None


def test_doc_registry_conforms(tmp_path):
    r: DocumentRegistryBackend = DocRegistry(data_dir=tmp_path)
    assert r is not None


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
