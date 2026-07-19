# core/embed_client.py
"""HTTP-клиент embed-сервиса. Реализует EmbedderBackend без загрузки модели."""
from __future__ import annotations

from typing import Sequence

import httpx
import numpy as np

from docling_rag.core.errors import EmbedServiceUnavailableError


class HTTPEmbedder:
    """POST {embed_url}/embed {"texts": [...]} -> ndarray float32 (N, dim)."""

    def __init__(self, base_url: str, timeout: float = 120.0,
                 transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout,
                                    transport=transport)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        try:
            resp = self._client.post("/embed", json={"texts": list(texts)})
            resp.raise_for_status()
        except httpx.HTTPError as e:  # connect/timeout/5xx -> одна доменная ошибка
            raise EmbedServiceUnavailableError(
                f"Сервис эмбеддингов недоступен: {e}"
            ) from e
        return np.asarray(resp.json()["embeddings"], dtype=np.float32)
