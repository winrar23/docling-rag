# api/embed_app.py — embed-сервис: единственный процесс с моделью USER-bge-m3.
"""uvicorn --factory docling_rag.api.embed_app:create_app (entrypoint 'embed').

Модель грузится блокирующе ДО старта сервера: «сервер отвечает» == «модель готова»,
readiness в compose — обычный healthcheck на GET /health со start_period.
"""
from fastapi import FastAPI
from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1)


def create_app(embedder=None, model_name: str | None = None) -> FastAPI:
    if embedder is None:
        from docling_rag.cli.config_loader import load_config
        from docling_rag.core.embedder import Embedder
        cfg = load_config()
        model_name = cfg["embedding_model"]
        embedder = Embedder(model_name=model_name)  # блокирующая загрузка модели

    app = FastAPI(title="docling-rag-embed")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/embed")
    def embed(req: EmbedRequest) -> dict:
        vecs = embedder.embed(req.texts)
        return {"embeddings": vecs.tolist(), "model": model_name, "dim": int(vecs.shape[1])}

    return app
