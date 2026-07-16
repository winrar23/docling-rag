# docling-rag

**v0.2.0** — CLI-утилита для семантического поиска по технической документации. Парсит PDF, DOCX, Markdown, нарезает на chunks с учётом структуры документа (заголовки, таблицы, код), строит векторный индекс в PostgreSQL + pgvector и отвечает на запросы ближайшими по смыслу фрагментами.

> Два режима: `search` — сырые chunks с cosine similarity score; `ask` — ответ на вопрос через локальный LLM-агент (100% оффлайн, через LM Studio).

---

## Быстрый старт (Docker)

CLI работает поверх PostgreSQL и запускается **docker-only**. Требования на хосте: Docker Desktop + LM Studio (только для `ask`).

```bash
git clone https://github.com/winrar23/docling-rag.git
cd docling-rag
cp .env.example .env   # пути volumes и порты — правь под себя

docker compose up -d --wait          # postgres + api (health: :8000/health)
docker compose run --rm cli init     # схема БД (идемпотентно)

# Книги кладутся в ${BOOKS_DIR:-./books} и видны контейнеру как /books
docker compose run --rm cli add /books/my-book.pdf --title "My Book" --topic "software" --tag arch
docker compose run --rm cli search "схема звезда и таблицы фактов"
docker compose run --rm cli list
docker compose run --rm cli delete /books/my-book.pdf
docker compose run --rm cli ask "вопрос"   # LM Studio на хосте, порт 1234

docker compose run --rm cli test tests/ -m "not integration and not slow"  # тесты в контейнере
docker compose --profile dev up api-dev    # hot-reload API на :8001
```

Все данные лежат на путях хоста из `.env` (`PGDATA_DIR`, `HF_CACHE_DIR`, `UPLOADS_DIR`,
`BOOKS_DIR`) — расположение выбираешь сам, named volumes не используются. Индекс живёт
в postgres (`PGDATA_DIR`), файлового хранилища больше нет.

> **Первый `add` скачает embedding-модель `deepvk/USER-bge-m3` (~2.3 ГБ)** в `HF_CACHE_DIR`
> и закэширует её. Индексация на CPU небыстрая — большой PDF может занять минуты.

### PyTorch: CPU или GPU

Образ собирается с **CPU-сборкой** PyTorch: GPU в контейнер не пробрасывается (Docker
Desktop на macOS — это Linux-VM без доступа к Metal), а CUDA-сборка добавила бы ~4 ГБ
бесполезных NVIDIA-библиотек (образ был бы ~6 ГБ вместо 2.4). CPU-сборка работает на
любой машине; на функциональность выбор не влияет — только на скорость `add`/`search`.

При установке на хост (`uv pip install -e ".[dev]"`, для разработки):

- **macOS (Apple Silicon)** — ничего выбирать не нужно: обычный wheel с PyPI уже включает
  поддержку Apple-GPU (backend MPS, работает с объединённой памятью). CUDA-сборка на Mac
  неприменима — CUDA есть только у NVIDIA.
- **Linux/Windows с NVIDIA-картой** — дефолтный `pip install torch` (CUDA-сборка) ускорит
  индексацию и OCR.
- **Linux/Windows без NVIDIA** — экономнее CPU-сборка (минус ~4 ГБ диска):
  `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
  до установки пакета.

---

## Команды

Все команды принимают `[--config config.yaml]`. Подключение к БД: env `DATABASE_URL`
(в compose уже выставлен) приоритетнее ключа `database_url` из конфига.

### `init` — инициализировать схему БД

```bash
docker compose run --rm cli init
```

Создаёт расширение pgvector, таблицы и HNSW-индекс. Идемпотентно — безопасно запускать повторно.

---

### `add` — добавить документы в индекс

```bash
docker compose run --rm cli add <путь> [--title TEXT] [--topic TEXT] [--tag TEXT]...
```

Принимает файл или папку. Поддерживаемые форматы: **PDF, DOCX, MD**.

- `--title` — название документа (свободная строка)
- `--topic` — домен/тема (например: `"software architecture"`, `"data engineering"`)
- `--tag` — тег, можно указывать несколько: `--tag arch --tag solid`
- Таблицы и code-блоки — отдельные неделимые chunks
- При ошибке на конкретном файле — пропускает и продолжает
- Повторный `add` того же файла не дублирует chunks (идемпотентно)

```bash
docker compose run --rm cli add /books/architecture.pdf
docker compose run --rm cli add /books/book.pdf --title "Clean Architecture" --topic "software" --tag arch --tag solid
docker compose run --rm cli add /books/ --topic "project docs"
```

---

### `search` — семантический поиск

```bash
docker compose run --rm cli search "<запрос>" [--top-k 5] [--tag TEXT]... [--topic TEXT]
```

Возвращает топ-K фрагментов по cosine similarity (HNSW-индекс pgvector). Поиск по смыслу, а не по ключевым словам.

- `--tag` — искать только в документах с этим тегом (можно несколько, AND-логика)
- `--topic` — искать только в документах с этой темой (без учёта регистра)
- Если фильтр не совпал ни с одним документом — возвращает пустой список (не fallback)

```bash
docker compose run --rm cli search "как работает партиционирование"
docker compose run --rm cli search "ETL pipeline best practices" --top-k 10
docker compose run --rm cli search "hub and satellite" --topic "data engineering"
docker compose run --rm cli search "layered architecture" --tag arch --tag ddd
```

**Пример вывода:**

```
Результаты для: "схема звезда и таблицы фактов"
────────────────────────────────────────────────────────────

[1] score=0.720 | architecture.md | стр.1 | text | [Chapter 3 > Data Models]
    DWH использует схему звезда с таблицами фактов и измерений...

[2] score=0.651 | design.pdf | стр.4 | text | [Part II > Star Schema]
    Fact table содержит числовые метрики: продажи, количество...
```

---

### `list` — список проиндексированных документов

```bash
docker compose run --rm cli list
```

```
Проиндексировано документов: 3
────────────────────────────────────────────────────────────
  architecture.md            4 chunks | Clean Architecture     | software           | [arch, solid]
  design.pdf                12 chunks | Data Vault 2.0         | data engineering   | [data-vault]
  etl_pipeline.docx          7 chunks | —                      | —                  | []
```

---

### `delete` — удалить документ из индекса

```bash
docker compose run --rm cli delete <путь-источника>
```

Удаляет документ и все его chunks (каскадом). Ключ — путь-источник, как его показывает
`list`. Несуществующий документ → понятная ошибка и exit 1.

```bash
docker compose run --rm cli delete /books/architecture.pdf
```

---

### `ask` — задать вопрос агенту *(требует LM Studio)*

```bash
docker compose run --rm cli ask "<вопрос>" [--top-k 5]
```

Агент сам вызывает семантический поиск по индексу, а затем синтезирует ответ через локальный LLM. 100% оффлайн.

**Требования:**
1. Запустить [LM Studio](https://lmstudio.ai) на хосте (порт 1234) и загрузить любую модель
2. В контейнерном конфиге агент уже включён (`agent_enabled: true`, LLM через `host.docker.internal:1234`)

```bash
docker compose run --rm cli ask "Что такое Data Vault и чем он отличается от Star Schema?"
docker compose run --rm cli ask "Объясни принцип dependency inversion" --tag arch --top-k 10
```

**Пример вывода:**

```
Data Vault — это методология моделирования данных, разработанная Дэном Линстедтом.
В отличие от Star Schema, которая оптимизирована для аналитических запросов,
Data Vault фокусируется на аудируемости и исторических данных через три типа таблиц:
хабы (бизнес-ключи), линки (связи) и сателлиты (атрибуты).

Источники: design.pdf (стр. 12), architecture.md (стр. 3)
```

---

## Конфигурация

Дефолты живут в коде (`src/docling_rag/cli/config_loader.py`) — отдельный `config.yaml` в репозитории не нужен и не хранится. В контейнере запечён `/app/config.yaml` (`docker/config.container.yaml`) с настройками под docker-сеть. Чтобы переопределить дефолты на хосте, создайте свой `config.yaml` в рабочей директории или укажите путь через `--config PATH`:

```yaml
embedding_model: deepvk/USER-bge-m3  # имя с org — как есть; без org — префикс sentence-transformers/
chunk_max_tokens: 512                # токен-лимит чанка (у bge-m3 окно 8192 — авто-лимит слишком крупный)
top_k_results: 5                     # результатов по умолчанию
database_url: postgresql://docling:docling@127.0.0.1:5432/docling_rag  # env DATABASE_URL приоритетнее

# Агентский режим
agent_enabled: true
llm_base_url: "http://127.0.0.1:1234/v1"
llm_api_key: "lm-studio"
llm_model: "local-model"
agent_top_k: 5
```

Значения из файла перекрывают дефолты; env `DATABASE_URL` перекрывает и файл, и дефолт (в compose он уже выставлен). Лог поисковых запросов пишется в таблицу `searches` в БД, а не в файл.

> **Важно:** нельзя менять `embedding_model` после индексации — размерность вектора зашита
> в схему БД (`vector(1024)` под USER-bge-m3), требуется полная переиндексация.

---

## Поддерживаемые форматы

| Формат | Парсинг | Таблицы | Код |
|--------|---------|---------|-----|
| PDF    | Docling | ✓ | ✓ |
| DOCX   | Docling | ✓ | ✓ |
| MD     | Docling | ✓ | ✓ |

---

## Архитектура

```
Файл → Parser (Docling) → DoclingDocument → HybridChunker → Chunks → Embedder → DBStorage (pgvector)
                                                                                      ↓
Запрос → Embedder ───────────────────── [DBRegistry filter] ─── HNSW cosine search → Результаты
                                                                                      ↓
ask → pydantic-ai Agent → search tool ──────────────────────────────────────── LLM ответ
```

**Структура проекта:**

```
docling-rag/
├── src/docling_rag/
│   ├── cli/
│   │   ├── commands.py      # Click: init, add, search, list, delete, ask
│   │   └── config_loader.py # Дефолты в коде + опциональный config.yaml + DATABASE_URL-приоритет
│   ├── core/
│   │   ├── parser.py     # Docling: PDF/DOCX/MD → DoclingDocument
│   │   ├── chunker.py    # HybridChunker: structure-aware, headings, token-limit
│   │   ├── embedder.py   # SentenceTransformer → L2-нормализованные векторы
│   │   ├── indexer.py    # index_files(): file → parse → chunk → embed → store
│   │   ├── search.py     # run_search() — общая логика для search и agent tool
│   │   ├── agent.py      # pydantic-ai Agent с search tool (требует .[agent])
│   │   ├── protocols.py  # Protocol-абстракции: StorageBackend, DocumentRegistryBackend, SearchLogBackend
│   │   └── errors.py     # Доменные ошибки (storage/format/LLM)
│   └── storage/
│       ├── db_schema.py     # DDL: pgvector extension, documents, chunks, searches, HNSW-индекс
│       ├── db_storage.py    # Chunks + эмбеддинги в postgres (psycopg3 + pgvector)
│       ├── db_registry.py   # Реестр документов: title, topic, tags (таблица documents)
│       └── db_search_log.py # Лог поисковых запросов (таблица searches)
├── .claude/
│   └── skills/
│       └── docling-rag-manager/  # Claude Code skill для управления приложением
├── tests/                  # 103 fast + 23 integration + 1 slow
├── compose.yaml            # postgres + api + api-dev + cli
└── pyproject.toml
```

**HybridChunker** разбивает документ по структуре (heading → секция), сохраняет путь заголовков в каждом chunk'е (`[Chapter 1 > Section 1.2]`). Для эмбеддингов используется `context_text` (headings + text), для отображения — чистый `text`.

**Protocol-абстракции** `core/protocols.py`: CLI и core не знают о psycopg — `DBStorage`/`DBRegistry`/`DBSearchLog` подключаются через `StorageBackend`/`DocumentRegistryBackend`/`SearchLogBackend`, юнит-тесты используют in-memory fakes.

---

## Claude Code

Проект включает skill для Claude Code: `.claude/skills/docling-rag-manager/`.

Если работаешь в проекте через Claude Code — агент автоматически знает все команды, флаги и gotchas при обсуждении индексации или поиска.

---

## Разработка (установка на хост — только для тестов)

```bash
# Создать venv (если ещё не создан)
uv venv && source .venv/bin/activate

# Установка с dev-зависимостями
uv pip install -e ".[dev]"

# Установка с поддержкой агента
uv pip install -e ".[agent,dev]"

# Быстрые тесты (103 fast, герметичные — postgres не нужен)
pytest tests/ -m "not integration and not slow"

# Интеграционные тесты (нужен postgres: docker compose up -d postgres;
# первый прогон скачает deepvk/USER-bge-m3 ~2.3 ГБ)
pytest tests/ -m integration

# Agent integration тест
pytest tests/test_agent_integration.py -v -m integration -s
```

---

## Changelog

### v0.2.0
- **PostgreSQL + pgvector вместо файлового хранилища** — `DBStorage`/`DBRegistry` на psycopg3, две таблицы + HNSW-индекс, каскадное удаление; `FileStorage`/`DocRegistry` и флаг `--data-dir` удалены; CLI стал docker-only
- **Embedding-модель `deepvk/USER-bge-m3`** (1024d, ~2.3 ГБ) вместо `all-MiniLM-L6-v2`; новый ключ `chunk_max_tokens: 512`
- **`delete` команда** — удаление документа и его chunks по пути-источнику
- **`database_url` в конфиге + приоритет env `DATABASE_URL`**; понятные ошибки при недоступном postgres и неинициализированной схеме
- **Лог поиска — в таблицу `searches`** вместо файла (в docker файловый лог умирал с контейнером)
- **Уборка:** корневой `config.yaml` удалён (дефолты в коде), мёртвый `StorageBackend.save()` убран

### v0.1.2
- **`ask` команда** — новый режим: задать вопрос и получить синтезированный ответ через локальный LLM (LM Studio). Агент сам вызывает семантический поиск и отвечает на языке вопроса. 100% оффлайн.
- **`core/search.py`** — вынесена общая функция `run_search()`, переиспользуется в `search` и agent tool
- **`core/agent.py`** — pydantic-ai Agent с search tool, динамическим system prompt (список документов), форматированием chunks для LLM
- **Опциональная зависимость** `.[agent]` — `pydantic-ai[openai]>=1.0`

### v0.1.1
- **HybridChunker** — заменил кастомный chunker на `docling-core` `HybridChunker`: structure-aware нарезка по заголовкам, автоматический токен-лимит, headings в результатах поиска
- **Claude Code skill** — `.claude/skills/docling-rag-manager/` для управления приложением через агента

### v0.1.0
- `init`, `add`, `search`, `list`
- Метаданные документов: `--title`, `--topic`, `--tag`, фильтр поиска
- NumPy cosine search, Protocol-абстракции для хранилища
