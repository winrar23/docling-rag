# api/app.py — заглушка этапа 1; REST-эндпоинты каталога/чата появятся на этапе 4
from fastapi import FastAPI

app = FastAPI(title="docling-rag")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
