# syntax=docker/dockerfile:1

# --- Стадия frontend: заглушка до этапа 4 (здесь появится React build) ---
FROM node:22-slim AS frontend
RUN mkdir -p /out/static

# --- Стадия runtime: python + uv, один образ для api и cli ---
FROM python:3.12-slim AS runtime
COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/hf-cache

# libgl1/libglib2.0-0 — рантайм-зависимости opencv в OCR-пайплайне docling
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch/torchvision строго с CPU-индекса ДО остального: дефолтный linux-wheel тянет CUDA (~4 ГБ),
# а PyPI-torchvision бинарно несовместим с torch+cpu (RuntimeError: operator torchvision::nms does not exist).
# Версии запинены по проверенному образу этапа 1: дрейф пары уже ломал рантайм
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src/ src/
COPY tests/ tests/
# editable: /app/src — живой код пакета, dev-режим бинд-маунтит ./src поверх
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e ".[agent,api,dev]"

COPY docker/config.container.yaml ./config.yaml
COPY --from=frontend /out/static ./static
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]
