import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic_ai")

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pydantic_ai.messages import ModelResponse, TextPart  # noqa: E402
from pydantic_ai.models.function import FunctionModel  # noqa: E402
from pydantic_ai.models.test import TestModel  # noqa: E402

from docling_rag.api.app import (  # noqa: E402
    app, get_chat_model, get_registry, get_search_embedder, get_search_log,
    get_settings, get_storage,
)
from docling_rag.core.chunker import Chunk  # noqa: E402
from tests.fakes import InMemoryRegistry, InMemorySearchLog, InMemoryStorage  # noqa: E402

SETTINGS = {"top_k_results": 5, "agent_top_k": 3, "agent_enabled": True,
            "llm_model": "m", "llm_base_url": "http://127.0.0.1:1/v1",
            "llm_api_key": "k", "llm_timeout_sec": 5}


class FakeEmbedder:
    def embed(self, texts):
        return np.ones((len(texts), 4), dtype="float32")


@pytest.fixture
def client():
    registry, storage, log = InMemoryRegistry(), InMemoryStorage(), InMemorySearchLog()
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_storage] = lambda: storage
    app.dependency_overrides[get_search_embedder] = lambda: FakeEmbedder()
    app.dependency_overrides[get_search_log] = lambda: log
    app.dependency_overrides[get_settings] = lambda: dict(SETTINGS)
    app.dependency_overrides[get_chat_model] = lambda: TestModel()
    yield TestClient(app), registry, storage, log
    app.dependency_overrides.clear()


def _seed(registry, storage):
    registry.upsert("/books/dv.pdf", "DV Book", "dwh", ["arch"])
    storage.append(
        [Chunk(text="Data Vault uses hubs.", source_file="/books/dv.pdf", chunk_id=0,
               page_number=42, element_type="text", headings=["Ch 2"], context_text="ctx")],
        np.ones((1, 4), dtype="float32"),
    )


def test_chat_returns_answer_and_sources(client):
    c, registry, storage, _ = client
    _seed(registry, storage)
    body = c.post("/chat", json={"message": "Что такое Data Vault?"}).json()
    assert isinstance(body["answer"], str) and body["answer"]
    assert body["sources"], "TestModel вызывает tool → sources не пустые"
    src = body["sources"][0]
    assert src["file"] == "dv.pdf"          # basename, не полный путь
    assert src["page"] == 42
    assert src["headings"] == ["Ch 2"]
    assert isinstance(src["score"], float)


def test_chat_logs_agent_search(client):
    c, registry, storage, log = client
    _seed(registry, storage)
    c.post("/chat", json={"message": "Что такое Data Vault?"})
    assert len(log.entries) == 1


def test_chat_history_reaches_model(client):
    c, registry, storage, _ = client
    _seed(registry, storage)
    seen = {}

    def capture(messages, info):
        seen["n_messages"] = len(messages)
        return ModelResponse(parts=[TextPart(content="ок")])

    app.dependency_overrides[get_chat_model] = lambda: FunctionModel(capture)
    c.post("/chat", json={"message": "новый вопрос", "history": [
        {"role": "user", "content": "прошлый вопрос"},
        {"role": "assistant", "content": "прошлый ответ"},
    ]})
    assert seen["n_messages"] == 3  # 2 хода истории + новый запрос


def test_chat_empty_storage_returns_canned_answer(client):
    c, *_ = client  # storage не сеем
    body = c.post("/chat", json={"message": "вопрос"}).json()
    assert body == {"answer": "Хранилище пустое. Документов нет.", "sources": []}


def test_chat_validation_errors(client):
    c, *_ = client
    assert c.post("/chat", json={"message": ""}).status_code == 422
    assert c.post("/chat", json={"message": "q", "history": [
        {"role": "system", "content": "x"}]}).status_code == 422
    assert c.post("/chat", json={}).status_code == 422


def test_chat_agent_disabled_503(client):
    c, *_ = client
    app.dependency_overrides[get_settings] = lambda: dict(SETTINGS, agent_enabled=False)
    del app.dependency_overrides[get_chat_model]  # реальный get_chat_model видит agent_enabled
    r = c.post("/chat", json={"message": "q"})
    assert r.status_code == 503
    assert "агент" in r.json()["detail"].lower()


def test_chat_llm_connect_error_503(client):
    c, registry, storage, _ = client
    _seed(registry, storage)

    def refuse(messages, info):
        raise httpx.ConnectError("connection refused")

    app.dependency_overrides[get_chat_model] = lambda: FunctionModel(refuse)
    r = c.post("/chat", json={"message": "q"})
    assert r.status_code == 503
    assert "LM Studio" in r.json()["detail"]


def test_chat_llm_timeout_504(client):
    c, registry, storage, _ = client
    _seed(registry, storage)

    def hang(messages, info):
        raise httpx.ReadTimeout("timed out")

    app.dependency_overrides[get_chat_model] = lambda: FunctionModel(hang)
    r = c.post("/chat", json={"message": "q"})
    assert r.status_code == 504


def test_chat_storage_error_not_misattributed_to_llm(client):
    """Доменная ошибка хранилища (в цепочке есть ConnectionError) должна дать 503 postgres,
    а не 503 LM Studio — проверка порядка except-веток эндпоинта."""
    from docling_rag.core.errors import StorageUnavailableError

    c, registry, storage, _ = client
    _seed(registry, storage)

    def _broken_search(query_embedding, top_k=5, allowed_sources=None):
        try:
            raise ConnectionError("connection refused")
        except ConnectionError as e:
            raise StorageUnavailableError("PostgreSQL недоступен") from e

    storage.search = _broken_search
    r = c.post("/chat", json={"message": "вопрос"})
    assert r.status_code == 503
    assert "PostgreSQL" in r.json()["detail"]
    assert "LM Studio" not in r.json()["detail"]
