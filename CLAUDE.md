# docling-rag

CLI-утилита для семантического поиска по технической документации на базе Docling.
RAG-система: Docling → chunking → Sentence Transformers → NumPy cosine search.

**Статус:** MVP реализован. 45 unit-тестов + 1 integration test, все зелёные.

## Stack (MVP)

- Python 3.10–3.12, Docling, Sentence Transformers (`all-MiniLM-L6-v2`), NumPy, Click, PyYAML

## Commands (dev)

```bash
# Установка зависимостей
uv pip install -e ".[dev]"

# Проверить установку
docling-rag --help

# CLI команды
docling-rag init              # инициализировать хранилище в текущей директории
docling-rag add <path>        # добавить документ или папку в индекс
docling-rag search "<query>"  # семантический поиск (топ-5 результатов)
docling-rag list              # список проиндексированных документов
# update <file> — P1, не реализован

# Тесты
python3 -m pytest tests/ -m "not integration and not slow"     # быстрые (45 тестов)
python3 -m pytest tests/test_integration.py -m integration -s  # e2e тест (~10 сек)
```

## Architecture (MVP)

```
docling-rag/
├── cli/
│   ├── commands.py         # Click: init, add, search, list
│   └── config_loader.py    # Загрузка config.yaml + дефолты
├── core/
│   ├── parser.py           # Docling парсер (PDF, DOCX, MD, TXT)
│   ├── chunker.py          # Chunker: 3200 символов, overlap 320, atomic table/code
│   ├── embedder.py         # Sentence Transformers all-MiniLM-L6-v2, L2-нормализация
│   └── storage.py          # Protocol-абстракция (file → pgvector на этапе 2)
├── storage/
│   └── file_storage.py     # NumPy-хранилище с атомарными записями
├── data/
│   ├── embeddings.npy      # Матрица эмбеддингов (N × 384, float32)
│   └── metadata.json       # Метаданные chunks
├── tests/                  # 45 unit + 1 integration
└── config.yaml             # chunk_size, overlap, top_k_results, embedding_model
```

## Gotchas

- **Chunker без LangChain** — собственная реализация разбивки по предложениям; chunk_size в символах (3200 ≈ 800 токенов)
- **Таблицы и code-блоки не разбиваются** — определяются по типу Docling-элемента, хранятся как отдельный chunk
- **storage.py — Protocol-абстракция** — file_storage (MVP) легко заменяется на pgvector без изменения вызывающего кода
- **LLM нет в MVP** — `search` возвращает raw chunks с score, не генерирует ответы
- **Изображения/диаграммы** — только OCR через Docling; Vision LLM (GPT-4V) — этап 2
- **Одна embedding-модель для индексации и поиска** — нельзя менять модель без полной переиндексации
- **Атомарные записи** — `_atomic_save` использует `os.replace()` для предотвращения рассинхронизации `.npy`/`.json`
- **top-k по умолчанию из config** — `--top-k` без явного значения берёт `top_k_results` из `config.yaml`
- **`--config` флаг на всех командах** — `init`, `add`, `search` принимают `--config path/to/config.yaml`; `list` — только `--data-dir`

## Non-Goals (MVP)

Не используется в MVP: ChromaDB, FAISS, LangChain, OpenAI API, веб-интерфейс, БД

## Spec

`docs/Feature_Specification.md` — полная спецификация, приоритеты P0/P1/P2, этапность
