# core/embed_client.py
"""HTTP-клиент embed-сервиса. Реализует EmbedderBackend без загрузки модели."""
from __future__ import annotations

from typing import Sequence

import httpx
import numpy as np

from docling_rag.core.errors import EmbedServiceUnavailableError


class HTTPEmbedder:
    """POST {embed_url}/embed {"texts": [...]} -> ndarray float32 (N, dim)."""

    # Read — щедрый: CPU-батч 128x512 токенов на большой книге может не уложиться
    # в старые плоские 120s (риск ReadTimeout). Connect — быстрый: недоступность
    # сервиса (порт не слушает) должна обнаруживаться быстро, а не ждать 600s.
    def __init__(self, base_url: str, timeout: float | httpx.Timeout = httpx.Timeout(600.0, connect=10.0),
                 transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout,
                                    transport=transport)

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        # batch_size принимается для совместимости с интерфейсом локального Embedder
        # (indexer передаёт его явно) и игнорируется: батчинг — забота embed-сервиса.
        try:
            resp = self._client.post("/embed", json={"texts": list(texts)})
            resp.raise_for_status()
        except httpx.HTTPError as e:  # connect/timeout/5xx -> одна доменная ошибка
            raise EmbedServiceUnavailableError(
                f"Сервис эмбеддингов недоступен: {e}"
            ) from e
        return np.asarray(resp.json()["embeddings"], dtype=np.float32)
