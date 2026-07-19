"""Юниты HTTPEmbedder: httpx.MockTransport, без сети и модели."""
import httpx
import numpy as np
import pytest

from docling_rag.core.embed_client import HTTPEmbedder
from docling_rag.core.errors import EmbedServiceUnavailableError


def _transport_ok():
    def handler(request: httpx.Request) -> httpx.Response:
        import json
        texts = json.loads(request.content)["texts"]
        return httpx.Response(200, json={
            "embeddings": [[0.1] * 4 for _ in texts],
            "model": "deepvk/USER-bge-m3", "dim": 4,
        })
    return httpx.MockTransport(handler)


def test_embed_returns_float32_ndarray():
    emb = HTTPEmbedder("http://embed:8100", transport=_transport_ok())
    vecs = emb.embed(["a", "b"])
    assert isinstance(vecs, np.ndarray)
    assert vecs.dtype == np.float32 and vecs.shape == (2, 4)


def test_connect_error_raises_domain_error():
    def handler(request):
        raise httpx.ConnectError("connection refused")
    emb = HTTPEmbedder("http://embed:8100", transport=httpx.MockTransport(handler))
    with pytest.raises(EmbedServiceUnavailableError):
        emb.embed(["a"])


def test_5xx_raises_domain_error():
    transport = httpx.MockTransport(lambda req: httpx.Response(500, text="boom"))
    emb = HTTPEmbedder("http://embed:8100", transport=transport)
    with pytest.raises(EmbedServiceUnavailableError):
        emb.embed(["a"])


def test_embed_service_error_is_not_storage_error():
    from docling_rag.core.errors import StorageError
    assert not issubclass(EmbedServiceUnavailableError, StorageError)


def test_embed_accepts_batch_size_kwarg_for_indexer_compat():
    """indexer вызывает embed(batch, batch_size=...) для локального Embedder;
    HTTPEmbedder должен принимать (и игнорировать) тот же kwarg."""
    emb = HTTPEmbedder("http://embed:8100", transport=_transport_ok())
    vecs = emb.embed(["a", "b"], batch_size=8)
    assert isinstance(vecs, np.ndarray)
    assert vecs.dtype == np.float32 and vecs.shape == (2, 4)
