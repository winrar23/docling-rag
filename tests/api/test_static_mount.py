"""Этап 4-D: раздача SPA-статики. API-роуты приоритетнее mount'а; без папки static — no-op."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from docling_rag.api.app import mount_static


def _app_with_health() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app


def test_static_served_and_api_wins(tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<html><body>ui</body></html>", encoding="utf-8")
    app = _app_with_health()
    mount_static(app, str(static))
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}  # API не перекрыт mount'ом
    assert "ui" in client.get("/").text  # index.html с корня (html=True)


def test_mount_skipped_without_static_dir(tmp_path):
    app = _app_with_health()
    mount_static(app, str(tmp_path / "нет-такой-папки"))
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/").status_code == 404  # ничего не смонтировано
