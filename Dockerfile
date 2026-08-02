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

# Слой зависимостей: только pyproject — правки src/tests не инвалидируют установку стека
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r pyproject.toml --extra agent --extra api --extra dev

# Enforce: deps-слой не имеет права молча переустановить torch-пару (например, CUDA-wheel с PyPI)
RUN python -c "import torch, torchvision; v=(torch.__version__, torchvision.__version__); assert v == ('2.13.0+cpu', '0.28.0+cpu'), v"

# Пре-бейк RapidOCR-моделей: torch-движок качает ~16 МБ .pth в site-packages/rapidocr/models
# при первом парсе PDF; скачивание происходит в конструкторе стадии — запекаем в слой образа.
# backend='torch' обязателен: дефолт onnxruntime в этом образе падает ImportError'ом,
# рантайм (OcrAutoModel) ловит его и фолбэчится на torch — конструируем сразу как рантайм
RUN python -c "\
from docling.datamodel.accelerator_options import AcceleratorOptions; \
from docling.datamodel.pipeline_options import RapidOcrOptions; \
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel; \
RapidOcrModel(enabled=True, artifacts_path=None, options=RapidOcrOptions(backend='torch'), accelerator_options=AcceleratorOptions())"

# Пре-бейк кириллической rec-модели RapidOCR (~10 МБ): ocr_lang=ru качает её при
# первом русском скане — запекаем в слой, чтобы worker не ходил в modelscope в рантайме
RUN python -c "\
from docling.datamodel.accelerator_options import AcceleratorOptions; \
from docling.datamodel.pipeline_options import RapidOcrOptions; \
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel; \
RapidOcrModel(enabled=True, artifacts_path=None, options=RapidOcrOptions(backend='torch', rapidocr_params={'Rec.lang_type': 'cyrillic'}), accelerator_options=AcceleratorOptions())"

COPY src/ src/
# editable: /app/src — живой код пакета, dev-режим бинд-маунтит ./src поверх; deps уже в слое выше
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-deps -e .
COPY tests/ tests/

COPY docker/config.container.yaml ./config.yaml
COPY --from=frontend /out/static ./static
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]
