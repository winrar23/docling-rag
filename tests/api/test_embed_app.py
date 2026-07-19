import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from docling_rag.api.embed_app import create_app  # noqa: E402


class FakeEmbedder:
    def embed(self, texts):
        return np.full((len(texts), 4), 0.5, dtype=np.float32)


@pytest.fixture
def client():
    return TestClient(create_app(embedder=FakeEmbedder(), model_name="fake-model"))


def test_health_ok(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_embed_returns_vectors_model_dim(client):
    resp = client.post("/embed", json={"texts": ["a", "b"]})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["embeddings"]) == 2 and len(body["embeddings"][0]) == 4
    assert body["model"] == "fake-model" and body["dim"] == 4


def test_embed_empty_texts_422(client):
    assert client.post("/embed", json={"texts": []}).status_code == 422
