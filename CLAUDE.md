# docling-rag

CLI-утилита для семантического поиска по технической документации на базе Docling.
RAG-система: Docling → chunking → Sentence Transformers → NumPy cosine search.

## Stack (MVP)

- Python 3.10+, Docling, Sentence Transformers (`all-MiniLM-L6-v2`), NumPy, Click

## Commands (dev)

```bash
# Установка зависимостей
uv pip install -e ".[dev]"

# CLI команды
docling-rag init              # инициализировать хранилище в текущей директории
docling-rag add <path>        # добавить документ или папку в индекс
docling-rag search "<query>"  # семантический поиск (топ-5 результатов)
docling-rag list              # список проиндексированных документов
docling-rag update <file>     # переиндексировать конкретный файл (P1)

# Тесты
pytest tests/
```

## Architecture (MVP)

```
docling-rag/
├── cli/                    # Click команды
├── core/
│   ├── parser.py           # Docling парсер (PDF, DOCX, MD, TXT)
│   ├── chunker.py          # Простой chunker (500-1000 токенов, overlap 10%)
│   ├── embedder.py         # Sentence Transformers all-MiniLM-L6-v2
│   └── storage.py          # Абстракция хранилища (file → pgvector на этапе 2)
├── storage/
│   └── file_storage.py     # NumPy-хранилище (MVP)
├── data/
│   ├── embeddings.npy      # Матрица эмбеддингов (N × 384)
│   └── metadata.json       # Метаданные chunks
├── tests/
└── config.yaml             # chunk_size, overlap, top_k_results, embedding_model
```

## Gotchas

- **Chunker без LangChain** — собственная реализация разбивки по предложениям
- **Таблицы и code-блоки не разбиваются** — определяются по типу Docling-элемента, хранятся как отдельный chunk
- **storage.py — абстракция** — file_storage (MVP) легко заменяется на pgvector без изменения вызывающего кода
- **LLM нет в MVP** — `search` возвращает raw chunks с score, не генерирует ответы
- **Изображения/диаграммы** — только OCR через Docling; Vision LLM (GPT-4V) — этап 2
- **Одна embedding-модель для индексации и поиска** — нельзя менять модель без полной переиндексации

## Non-Goals (MVP)

Не используется в MVP: ChromaDB, FAISS, LangChain, OpenAI API, веб-интерфейс, БД

## Spec

`docs/Feature_Specification.md` — полная спецификация, приоритеты P0/P1/P2, этапность
