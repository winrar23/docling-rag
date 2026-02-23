# docling-rag

CLI-утилита для семантического поиска по технической документации на базе Docling.
RAG-система: Docling → chunking → Sentence Transformers → NumPy cosine search.

**Статус:** MVP + document metadata + hybrid chunking реализованы. 57 unit-тестов + 2 integration tests, все зелёные.

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
docling-rag add <path> --title "..." --topic "..." --tag arch --tag solid  # с метаданными
docling-rag search "<query>"  # семантический поиск (топ-5 результатов)
docling-rag search "<query>" --tag arch --topic "architecture"  # с фильтром
docling-rag list              # список проиндексированных документов
# update <file> — P1, не реализован

# Тесты
python3 -m pytest tests/ -m "not integration and not slow"     # быстрые (65 тестов)
python3 -m pytest tests/test_integration.py -m integration -s  # e2e тесты (~30 сек)
```

## Architecture (MVP)

```
docling-rag/
├── cli/
│   ├── commands.py         # Click: init, add, search, list
│   └── config_loader.py    # Загрузка config.yaml + дефолты
├── core/
│   ├── parser.py           # Docling парсер → возвращает DoclingDocument (PDF, DOCX, MD)
│   ├── chunker.py          # HybridChunker (docling-core): structure-aware + token-aware + headings
│   ├── embedder.py         # Sentence Transformers all-MiniLM-L6-v2, L2-нормализация
│   └── storage.py          # StorageBackend + DocumentRegistryBackend Protocol-абстракции
├── storage/
│   ├── file_storage.py     # NumPy-хранилище с атомарными записями
│   └── doc_registry.py     # Метаданные документов (title, topic, tags) → doc_index.json
├── data/
│   ├── embeddings.npy      # Матрица эмбеддингов (N × 384, float32)
│   ├── metadata.json       # Метаданные chunks
│   └── doc_index.json      # Реестр документов (title, topic, tags, added_at)
├── tests/                  # 57 unit + 2 integration
└── config.yaml             # top_k_results, embedding_model (chunk_size удалён — HybridChunker авто)
```

## Gotchas

- **HybridChunker из docling-core** — разбивает по структуре документа (heading → секция), токен-лимит из tokenizer'а (all-MiniLM-L6-v2 → 256 токенов), мёрджит мелкие соседние chunks
- **context_text vs text** — `chunk.context_text` = headings + text (используется для эмбеддингов); `chunk.text` = чистый текст (хранится и отображается в поиске)
- **headings в metadata** — `metadata.json` хранит `headings: list[str]`; `search` отображает их как `[H1 > H2]`
- **Таблицы и code-блоки** — HybridChunker сохраняет их как атомарные chunks (element_type = "table" или "code")
- **storage.py — Protocol-абстракция** — file_storage (MVP) легко заменяется на pgvector без изменения вызывающего кода
- **LLM нет в MVP** — `search` возвращает raw chunks с score, не генерирует ответы
- **Изображения/диаграммы** — только OCR через Docling; Vision LLM (GPT-4V) — этап 2
- **Одна embedding-модель для индексации и поиска** — нельзя менять модель без полной переиндексации
- **Атомарные записи** — `_atomic_save` использует `os.replace()` для предотвращения рассинхронизации `.npy`/`.json`
- **top-k по умолчанию из config** — `--top-k` без явного значения берёт `top_k_results` из `config.yaml`
- **`--config` флаг на всех командах** — `init`, `add`, `search` принимают `--config path/to/config.yaml`; `list` — только `--data-dir`
- **Docling не парсит `.txt`** — для integration tests используй `.md`; поддерживаемые форматы: PDF, DOCX, MD
- **DocRegistry следует паттерну FileStorage** — тот же `_atomic_save` через `os.replace()`, ключ = `source_file`
- **CLI mock-паттерн** — в тестах патчить `cli.commands.chunk_document` + `DocRegistry` + `FileStorage`/`Parser`/`Embedder`
- **Фильтр поиска: пустой match → пустые результаты** — если `--tag`/`--topic` не совпадает ни с одним документом, `search` возвращает пустой список (не fallback на все документы)
- **`--topic` сравнивается case-insensitive** — `"Software"` == `"software"` через `.lower()`

## Non-Goals (MVP)

Не используется в MVP: ChromaDB, FAISS, LangChain, OpenAI API, веб-интерфейс, БД

## Git workflow

- **`main`** — стабильная ветка, всегда рабочая
- **`dev`** — ветка для экспериментальных фич, worktree в `.worktrees/dev/`
- Новые фичи разрабатываются в `dev`, после стабилизации мёрджатся в `main`

```bash
# Переключиться в dev worktree
cd .worktrees/dev

# Список worktrees
git worktree list
```

## Docs

Локальная документация в `docs/` (в .gitignore, не публикуется):
- `docs/Feature_Specification.md` — полная спецификация, P0/P1/P2
- `docs/ARCHITECTURE.md` — компонентная архитектура, потоки данных, инварианты
- `docs/FEATURES.md` — краткий фичелист со статусами
