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
    import docling_rag.core.search as s
    import docling_rag.core.agent  # noqa: F401 — просто импортируемость
    import inspect
    assert "from docling_rag.storage" not in inspect.getsource(s)
