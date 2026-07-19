"""Фактори get_embedder: выбор бэкенда по embed_url (модель НЕ грузим)."""
from unittest.mock import patch

from docling_rag.core.embed_client import HTTPEmbedder
from docling_rag.core.embedder import get_embedder


def test_embed_url_set_returns_http_embedder():
    emb = get_embedder({"embed_url": "http://embed:8100", "embedding_model": "m"})
    assert isinstance(emb, HTTPEmbedder)


def test_embed_url_absent_returns_local_embedder():
    with patch("docling_rag.core.embedder.Embedder") as embedder_cls:
        embedder_cls.return_value = "LOCAL"
        assert get_embedder({"embed_url": None, "embedding_model": "m"}) == "LOCAL"
        embedder_cls.assert_called_once_with(model_name="m")


def test_defaults_contain_embed_url_none():
    from docling_rag.cli.config_loader import _DEFAULTS
    assert _DEFAULTS["embed_url"] is None
