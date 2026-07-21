import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic_ai")
psycopg = pytest.importorskip("psycopg")

from fastapi.testclient import TestClient  # noqa: E402
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart  # noqa: E402
from pydantic_ai.models.function import FunctionModel  # noqa: E402

from docling_rag.api.app import app, get_chat_model, get_search_embedder, get_settings  # noqa: E402
from docling_rag.core.chunker import Chunk  # noqa: E402
from docling_rag.storage.db_registry import DBRegistry  # noqa: E402
from docling_rag.storage.db_storage import DBStorage  # noqa: E402

pytestmark = pytest.mark.integration


class Fake1024Embedder:
    """Схема БД — vector(1024); реальная модель не нужна: тестируем storage/log-путь."""

    def embed(self, texts):
        return np.ones((len(texts), 1024), dtype="float32") / 32.0


def _scripted(messages, info):
    if len(messages) == 1:  # первый вызов модели — дёргаем tool
        return ModelResponse(parts=[ToolCallPart(tool_name="search_documents",
                                                 args={"query": "hubs"})])
    return ModelResponse(parts=[TextPart(content="Data Vault строится на hubs (dv.pdf, стр. 42)")])


def test_chat_against_real_postgres(db_url, clean_db):
    storage, registry = DBStorage(db_url), DBRegistry(db_url)
    registry.upsert("/books/dv.pdf", "DV Book", "dwh", ["arch"])
    storage.append(
        [Chunk(text="Data Vault uses hubs.", source_file="/books/dv.pdf", chunk_id=0,
               page_number=42, element_type="text", headings=["Ch 2"], context_text="ctx")],
        np.ones((1, 1024), dtype="float32") / 32.0,
    )

    app.dependency_overrides[get_settings] = lambda: {
        "database_url": db_url, "top_k_results": 5, "agent_top_k": 3,
        "agent_enabled": True, "llm_model": "m", "llm_base_url": "http://127.0.0.1:1/v1",
        "llm_api_key": "k", "llm_timeout_sec": 5,
        "embed_url": None, "embedding_model": "unused",
    }
    app.dependency_overrides[get_search_embedder] = lambda: Fake1024Embedder()
    app.dependency_overrides[get_chat_model] = lambda: FunctionModel(_scripted)
    try:
        body = TestClient(app).post("/chat", json={"message": "Что такое Data Vault?"}).json()
    finally:
        app.dependency_overrides.clear()

    assert "hubs" in body["answer"]
    assert body["sources"][0]["file"] == "dv.pdf"
    assert body["sources"][0]["page"] == 42

    # сквозной путь лога: агентский поиск записан в searches (реальный DBSearchLog)
    with psycopg.connect(db_url) as conn:
        rows = conn.execute("SELECT query, top_score FROM searches").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "hubs"  # запрос агента, не вопрос пользователя
