import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from docling_rag.api.app import app, get_jobs  # noqa: E402
from docling_rag.core.errors import (  # noqa: E402
    EmbedServiceUnavailableError, StorageSchemaMissingError, StorageUnavailableError,
)


class _RaisingJobs:
    def __init__(self, exc): self._exc = exc
    def list(self, limit=20, status=None): raise self._exc
    def get(self, job_id): raise self._exc


@pytest.mark.parametrize("exc,fragment", [
    (StorageUnavailableError("нет коннекта"), "PostgreSQL недоступен"),
    (StorageSchemaMissingError("нет таблиц"), "docling-rag init"),
    (EmbedServiceUnavailableError("embed лежит"), "эмбеддинг"),
])
def test_domain_errors_become_503(exc, fragment):
    app.dependency_overrides[get_jobs] = lambda: _RaisingJobs(exc)
    try:
        client = TestClient(app)
        resp = client.get("/jobs")
        assert resp.status_code == 503
        assert fragment.lower() in resp.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()
