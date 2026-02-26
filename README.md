# docling-rag

**v0.1.2** — CLI-утилита для семантического поиска по технической документации. Парсит PDF, DOCX, Markdown, нарезает на chunks с учётом структуры документа (заголовки, таблицы, код), строит векторный индекс и отвечает на запросы ближайшими по смыслу фрагментами.

> Два режима: `search` — сырые chunks с cosine similarity score; `ask` — ответ на вопрос через локальный LLM-агент (100% оффлайн, через LM Studio).

---

## Быстрый старт

```bash
# 1. Установка
git clone https://github.com/winrar23/docling-rag.git
cd docling-rag
uv pip install -e .

# 2. Инициализация хранилища
docling-rag init

# 3. Добавить документы (файл или папку)
docling-rag add ./docs/

# 3а. С метаданными (опционально)
docling-rag add architecture.pdf --title "Clean Architecture" --topic "software" --tag arch --tag solid

# 4. Поиск
docling-rag search "схема звезда и таблицы фактов"

# 4а. Поиск с фильтром по тегу
docling-rag search "dependency inversion" --tag arch

# 5. Посмотреть что проиндексировано
docling-rag list
```

Первый запуск скачает модель `all-MiniLM-L6-v2` (~90 МБ) и закэширует её локально.

---

## Команды

### `init` — инициализировать хранилище

```bash
docling-rag init [--data-dir data] [--config config.yaml]
```

Создаёт директорию хранилища и папку для логов. Безопасно запускать повторно.

---

### `add` — добавить документы в индекс

```bash
docling-rag add <путь> [--title TEXT] [--topic TEXT] [--tag TEXT]... [--data-dir data] [--config config.yaml]
```

Принимает файл или папку. Поддерживаемые форматы: **PDF, DOCX, MD**.

- `--title` — название документа (свободная строка)
- `--topic` — домен/тема (например: `"software architecture"`, `"data engineering"`)
- `--tag` — тег, можно указывать несколько: `--tag arch --tag solid`
- Таблицы и code-блоки — отдельные неделимые chunks
- При ошибке на конкретном файле — пропускает и продолжает

```bash
docling-rag add architecture.pdf
docling-rag add book.pdf --title "Clean Architecture" --topic "software" --tag arch --tag solid
docling-rag add ./docs/ --topic "project docs"
```

---

### `search` — семантический поиск

```bash
docling-rag search "<запрос>" [--top-k 5] [--tag TEXT]... [--topic TEXT] [--data-dir data] [--config config.yaml]
```

Возвращает топ-K фрагментов по cosine similarity. Поиск по смыслу, а не по ключевым словам.

- `--tag` — искать только в документах с этим тегом (можно несколько, AND-логика)
- `--topic` — искать только в документах с этой темой (без учёта регистра)
- Если фильтр не совпал ни с одним документом — возвращает пустой список (не fallback)

```bash
docling-rag search "как работает партиционирование"
docling-rag search "ETL pipeline best practices" --top-k 10
docling-rag search "hub and satellite" --topic "data engineering"
docling-rag search "layered architecture" --tag arch --tag ddd
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
docling-rag list [--data-dir data]
```

```
Проиндексировано документов: 3
────────────────────────────────────────────────────────────
  architecture.md            4 chunks | Clean Architecture     | software           | [arch, solid]
  design.pdf                12 chunks | Data Vault 2.0         | data engineering   | [data-vault]
  etl_pipeline.docx          7 chunks | —                      | —                  | []
```

---

### `ask` — задать вопрос агенту *(требует агентский режим)*

```bash
docling-rag ask "<вопрос>" [--top-k 5] [--data-dir data] [--config config.yaml]
```

Агент сам вызывает семантический поиск по индексу, а затем синтезирует ответ через локальный LLM. 100% оффлайн.

**Требования:**
1. Установить зависимость агента: `uv pip install -e ".[agent]"`
2. Запустить [LM Studio](https://lmstudio.ai) и загрузить любую модель
3. Включить агент в `config.yaml`:

```yaml
agent_enabled: true
llm_model: "your-model-name"   # имя модели как оно отображается в LM Studio
llm_base_url: "http://127.0.0.1:1234/v1"
```

```bash
docling-rag ask "Что такое Data Vault и чем он отличается от Star Schema?"
docling-rag ask "Объясни принцип dependency inversion" --tag arch --top-k 10
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

По умолчанию читается `config.yaml` из текущей директории:

```yaml
embedding_model: all-MiniLM-L6-v2   # модель для эмбеддингов
top_k_results: 5                     # результатов по умолчанию
data_dir: data                       # папка хранилища
log_file: logs/search.log            # лог поисковых запросов

# Агентский режим (требует uv pip install -e ".[agent]")
# agent_enabled: false
# llm_base_url: "http://127.0.0.1:1234/v1"
# llm_api_key: "lm-studio"
# llm_model: "local-model"
# agent_top_k: 5
```

> Размер chunks управляется автоматически — `HybridChunker` использует токен-лимит embedding-модели (256 токенов для `all-MiniLM-L6-v2`).

> **Важно:** нельзя менять `embedding_model` после индексации — требуется полная переиндексация.

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
Файл → Parser (Docling) → DoclingDocument → HybridChunker → Chunks → Embedder → FileStorage
                                                                                      ↓
Запрос → Embedder ───────────────────── [DocRegistry filter] ──── cosine search → Результаты
                                                                                      ↓
ask → pydantic-ai Agent → search tool ──────────────────────────────────────── LLM ответ
```

**Структура проекта:**

```
docling-rag/
├── cli/
│   ├── commands.py         # Click: init, add, search, list, ask
│   └── config_loader.py    # Загрузка config.yaml + дефолты
├── core/
│   ├── parser.py           # Docling: PDF/DOCX/MD → DoclingDocument
│   ├── chunker.py          # HybridChunker: structure-aware, headings, token-limit
│   ├── embedder.py         # SentenceTransformer → L2-нормализованные векторы
│   ├── search.py           # run_search() — общая логика для search и agent tool
│   ├── agent.py            # pydantic-ai Agent с search tool (требует .[agent])
│   └── storage.py          # Protocol-абстракции: StorageBackend, DocumentRegistryBackend
├── storage/
│   ├── file_storage.py     # NumPy (.npy) + JSON хранилище chunks
│   └── doc_registry.py     # Реестр документов: title, topic, tags → doc_index.json
├── .claude/
│   └── skills/
│       └── docling-rag-manager/  # Claude Code skill для управления приложением
├── data/                   # всё содержимое в .gitignore
│   ├── embeddings.npy      # Матрица эмбеддингов (N × 384)
│   ├── metadata.json       # Метаданные chunks (включая headings)
│   └── doc_index.json      # Реестр документов (title, topic, tags, added_at)
├── tests/                  # 73 unit-тестов + 3 integration
├── config.yaml
└── pyproject.toml
```

**HybridChunker** разбивает документ по структуре (heading → секция), сохраняет путь заголовков в каждом chunk'е (`[Chapter 1 > Section 1.2]`). Для эмбеддингов используется `context_text` (headings + text), для отображения — чистый `text`.

**Protocol-абстракции** `core/storage.py` позволяют заменить NumPy-файлы на pgvector без изменения CLI-кода.

---

## Claude Code

Проект включает skill для Claude Code: `.claude/skills/docling-rag-manager/`.

Если работаешь в проекте через Claude Code — агент автоматически знает все команды, флаги и gotchas при обсуждении индексации или поиска.

---

## Разработка

```bash
# Установка с dev-зависимостями
uv pip install -e ".[dev]"

# Установка с поддержкой агента
uv pip install -e ".[agent,dev]"

# Быстрые тесты (73 unit)
pytest tests/ -m "not integration and not slow"

# Интеграционные тесты (реальный Docling + модель, ~30 сек)
pytest tests/test_integration.py -v -m integration -s

# Agent integration тест
pytest tests/test_agent_integration.py -v -m integration -s
```

---

## Changelog

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
