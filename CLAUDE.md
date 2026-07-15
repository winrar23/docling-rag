# docling-rag

CLI-утилита для семантического поиска по технической документации на базе Docling.
RAG-система: Docling → chunking → Sentence Transformers → NumPy cosine search.

**Статус:** MVP + document metadata + hybrid chunking + pydantic-ai agent реализованы; stage-0 рефакторинг (src-layout, идемпотентный add, exit-коды, Protocol-типизация, композируемый agent) завершён; v2 этап 1 (Docker) завершён — образ `docling-rag:stage1`, docker compose (postgres + api + cli), env-configurable volumes. 108 unit/fast-тестов + 3 integration tests + 1 slow test, все зелёные.

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
docling-rag ask "<вопрос>"    # задать вопрос агенту (требуется agent_enabled: true + LM Studio)
# update <file> — P1, не реализован

# Тесты
python3 -m pytest tests/ -m "not integration and not slow"          # быстрые (108 тестов, 4 deselected)
python3 -m pytest tests/test_integration.py -m integration -s       # e2e тесты (~30 сек)
python3 -m pytest tests/test_agent_integration.py -m integration -s # agent e2e тест
```

## Commands (docker)

```bash
cp .env.example .env   # пути volumes и порты — правь под себя
docker compose up -d --wait          # postgres + api (health: :8000/health)
docker compose run --rm cli init
docker compose run --rm cli add /books/my-book.pdf --title "My Book"
docker compose run --rm cli search "запрос"
docker compose run --rm cli ask "вопрос"   # LM Studio на хосте, порт 1234
docker compose run --rm cli test tests/ -m "not integration and not slow"  # тесты в контейнере
docker compose --profile dev up api-dev    # hot-reload API на :8001
```

## Architecture

Пакеты живут под `src/docling_rag/` (src-layout; editable install резолвится сюда, не в repo-root).

```
docling-rag/
├── src/docling_rag/
│   ├── cli/
│   │   ├── commands.py      # Click: init, add, search, list, ask + exit-код контракт
│   │   └── config_loader.py # load_config(path, *, required=) + дефолты (агент-ключи включены)
│   ├── core/
│   │   ├── parser.py     # Docling парсер → DoclingDocument; SUPPORTED_EXTENSIONS = {.pdf, .docx, .md}
│   │   ├── chunker.py    # chunk_document(); HybridChunker кеширован per embedding_model (lru_cache)
│   │   ├── embedder.py   # Sentence Transformers, L2-нормализация, настраиваемый batch_size
│   │   ├── indexer.py    # index_files(): file → parse → chunk → embed → store, per-file error isolation
│   │   ├── search.py     # run_search() + resolve_allowed_sources() — переиспользуются CLI search и agent tool
│   │   ├── agent.py      # create_agent(model) + build_lmstudio_model(...); требует .[agent]
│   │   ├── protocols.py  # StorageBackend + DocumentRegistryBackend Protocol — используются в аннотациях (search.py, indexer.py)
│   │   └── errors.py     # StorageError, UnsupportedFormatError, LLMUnavailableError
│   ├── api/
│   │   └── app.py        # FastAPI health-заглушка (GET /health); требует .[api]; REST каталога/чата — этап 4
│   └── storage/
│       ├── file_storage.py  # NumPy-хранилище с атомарными записями (StorageBackend impl)
│       └── doc_registry.py  # Метаданные документов (title, topic, tags) → doc_index.json (DocumentRegistryBackend impl)
├── data/                    # всё содержимое в .gitignore
│   ├── embeddings.npy       # Матрица эмбеддингов (N × 384, float32)
│   ├── metadata.json        # Метаданные chunks
│   └── doc_index.json       # Реестр документов (title, topic, tags, added_at)
├── tests/                   # tests/core/, tests/storage/, tests/api/, tests/test_*.py — 108 fast + 3 integration + 1 slow
├── config.yaml              # top_k_results, embedding_model (chunk_size удалён — HybridChunker авто)
├── Dockerfile               # multi-stage: frontend-заглушка (node) + runtime (python+uv); entrypoint-диспетчер api/test/cli
├── compose.yaml             # postgres + api + api-dev (profile dev) + cli (profile cli), bind-mounts из .env
├── .env.example             # DATA_DIR/PGDATA_DIR/HF_CACHE_DIR/UPLOADS_DIR/BOOKS_DIR + порты
└── docker/
    ├── entrypoint.sh          # api → uvicorn :8000; test → pytest; иначе → docling-rag CLI
    └── config.container.yaml  # запекается в /app/config.yaml: data_dir: /data, LLM через host.docker.internal:1234
```

## Gotchas

- **HybridChunker из docling-core** — разбивает по структуре документа (heading → секция), токен-лимит из tokenizer'а (all-MiniLM-L6-v2 → 256 токенов), мёрджит мелкие соседние chunks
- **HybridChunker кеширован per embedding_model** — `core/chunker.py::_get_chunker()` обёрнут в `@lru_cache(maxsize=4)`: повторные вызовы `chunk_document()` с той же моделью не пересоздают tokenizer/chunker
- **context_text vs text** — `chunk.context_text` = headings + text (используется для эмбеддингов); `chunk.text` = чистый текст (хранится и отображается в поиске)
- **headings в metadata** — `metadata.json` хранит `headings: list[str]`; `search` отображает их как `[H1 > H2]`
- **Таблицы и code-блоки** — HybridChunker сохраняет их как атомарные chunks (element_type = "table" или "code")
- **SUPPORTED_EXTENSIONS — единый источник** — `{".pdf", ".docx", ".md"}` определён только в `core/parser.py`; `.txt` отсутствует (убран из индексации — Docling его некорректно обрабатывал). `cli/commands.py` импортирует эту же константу для фильтрации файлов при `add`, дублирования нет
- **core/protocols.py — Protocol-абстракция, реально используется в аннотациях** — `StorageBackend`/`DocumentRegistryBackend` типизируют `core/search.py` и `core/indexer.py` (не только duck typing); ни один модуль `core/` не импортирует пакет `storage` напрямую. Тест `test_core_does_not_import_storage_package` это проверяет источниковым grep'ом по `core/search.py`, `core/indexer.py` и `core/agent.py` (для agent.py — чтение исходника с диска, без импорта, чтобы тест не падал на установках без `.[agent]`). `FileStorage`/`DocRegistry` (MVP) заменяемы на pgvector-реализации без изменения вызывающего кода
- **LLM нет в MVP** — `search` возвращает raw chunks с score, не генерирует ответы (генерация ответов — только через `ask`/agent)
- **Изображения/диаграммы** — только OCR через Docling; Vision LLM (GPT-4V) — этап 2
- **Одна embedding-модель для индексации и поиска** — нельзя менять модель без полной переиндексации
- **Атомарные записи** — `_atomic_save` использует `os.replace()` для предотвращения рассинхронизации `.npy`/`.json`
- **top-k по умолчанию из config** — `--top-k` без явного значения берёт `top_k_results` из `config.yaml`
- **`--config` флаг на ВСЕХ командах, включая `list`** — `init`, `add`, `search`, `list`, `ask` все принимают `--config path/to/config.yaml` (гоча про "list только --data-dir" устарела — исправлено в T4/T13). Контракт `load_config(path, *, required=)`: без явного `--config` читается `config.yaml` в cwd с `required=False` (нет файла → тихий fallback на дефолты); при явном `--config PATH` вызывается с `required=True` — если файл не существует, `ConfigError` → `click.ClickException` (exit 1). Невалидный YAML или не-dict корень тоже даёт `ConfigError` независимо от `required`
- **DocRegistry следует паттерну FileStorage** — тот же `_atomic_save` через `os.replace()`, ключ = `source_file` (резолвленный путь, см. ниже)
- **CLI mock-паттерн (актуальный)** — патчить `docling_rag.cli.commands.Parser` / `.Embedder` / `.FileStorage` / `.DocRegistry`; `chunk_document` теперь вызывается из `core/indexer.py`, поэтому патчить `docling_rag.core.indexer.chunk_document`, НЕ `docling_rag.cli.commands.chunk_document`
- **Фильтр поиска: пустой match → пустые результаты** — если `--tag`/`--topic` не совпадает ни с одним документом, `search` возвращает пустой список (не fallback на все документы)
- **`--topic` сравнивается case-insensitive** — `"Software"` == `"software"` через `.lower()`
- **Идемпотентный `add` через резолвленные пути** — `add`/`core/indexer.py::index_files()` резолвит путь (`Path(file).resolve()`) ДО вызова `chunk_document`/`registry.upsert`, использует его как `source`; перед `storage.append()` вызывается `storage.delete_by_source(source)`, поэтому повторный `add` того же файла не дублирует chunks, а `registry.upsert` сохраняет `added_at` и не затирает title/topic/tags значениями `None`/пустыми
- **ВАЖНО: документы, проиндексированные ДО этого рефакторинга, не дедуплицируются при re-add** — `delete_by_source`/`upsert` матчат по точной строке ключа; старые записи индекса могут использовать нерезолвленные (относительные/иные) пути, которые не совпадут с новым резолвленным `source`. Для уже существующих индексов, созданных до stage-0, рекомендуется полная переиндексация (`init` + `add` заново), а не точечный re-add
- **Изоляция ошибок по файлу в `index_files()`** — путь инициализируется как `str(file)` ДО `try`, резолвится (`Path(file).resolve()`) внутри `try`; если `.resolve()` падает (symlink loop, permission error), ошибка попадает в `report.errors` для этого файла, а batch не прерывается
- **Exit-код контракт** — `click.ClickException` (невалидный `--config`, повреждённое хранилище `StorageError`, ошибки агента, `agent_enabled: false`) → exit 1. Пустое хранилище (`FileNotFoundError`) → exit 1 только в `search` и `ask`; `list` на пустом хранилище печатает "Хранилище пустое. Документов нет." и завершается нормально с exit 0 (тест `test_list_command_empty_storage` фиксирует `exit_code == 0`). `add` дополнительно делает `raise SystemExit(1)`, если есть `files_failed` ИЛИ `chunks_added == 0` (даже если все файлы формально "ok", но ничего не добавлено); успешные пути → exit 0. Отдельно от этого — собственная валидация параметров Click (`--top-k 0` через `IntRange(min=1)`, несуществующий `file_path` через `click.Path(exists=True)`) даёт `UsageError` → exit 2, а не 1. Ошибки пишутся в stderr (`err=True`)
- **`ask` требует `.[agent]` и LM Studio** — `uv pip install -e ".[agent]"`, `agent_enabled: true` в config.yaml, LM Studio на `127.0.0.1:1234`
- **Обнаружение ошибок соединения с LLM — isinstance по цепочке cause/context, НЕ строковый матч** — httpx/openai заворачивают `httpx.ConnectError`/`ConnectTimeout` на несколько уровней глубже builtin `ConnectionError`. `cli/commands.py::_is_connection_error(e)` проходит `e.__cause__ or e.__context__` по цепочке и проверяет `isinstance(cur, (ConnectionError, httpx.ConnectError, httpx.ConnectTimeout))` (раньше — хрупкий матч `"ConnectError" in type(e).__name__`, теперь так НЕ делается)
- **`_create_and_run_agent` — точка мока для тестов** — сигнатура `_create_and_run_agent(question, cfg, data_dir, top_k) -> str`; в тестах `ask` патчить `docling_rag.cli.commands._create_and_run_agent`, НЕ `create_agent` напрямую
- **Lazy import + testability** — `_import_agent_module()` — отдельная функция для тестируемого lazy import guard, возвращает кортеж `(create_agent, AgentDeps, build_lmstudio_model)`; патчится через `patch("docling_rag.cli.commands._import_agent_module")`
- **pydantic-ai API — composable `create_agent(model)`** — `create_agent(model) -> Agent[AgentDeps, str]` принимает ЛЮБУЮ pydantic-ai `Model` (включая `pydantic_ai.models.test.TestModel`), не строит модель сама; `build_lmstudio_model(model_name, base_url, api_key) -> OpenAIChatModel` собирает `OpenAIChatModel(model_name, provider=OpenAIProvider(base_url=base_url, api_key=api_key))` отдельно — LM Studio говорит на Chat Completions API, поэтому явный `OpenAIChatModel`, а не `"openai:"`-префикс (тот означал бы Responses API). `tests/test_agent.py` покрывает agent tool через `TestModel`: реальный поиск по seeded storage выполняется (`test_agent_tool_executes_real_search`), динамические инструкции подставляют список документов (`test_dynamic_instructions_list_documents`). Импорты: `from pydantic_ai import Agent, RunContext`; `from pydantic_ai.models.openai import OpenAIChatModel`; `from pydantic_ai.providers.openai import OpenAIProvider`; `result.output` для получения ответа
- **Все volumes — bind-mounts из `.env`** — требование пользователя: расположение данных выбирает он. `DATA_DIR`/`PGDATA_DIR`/`HF_CACHE_DIR`/`UPLOADS_DIR`/`BOOKS_DIR`, дефолты `./volumes/*` и `./books`. Named volumes в compose НЕ использовать
- **Entrypoint-диспетчер образа** — `api` → uvicorn :8000, `test` → pytest, иначе → CLI `docling-rag`. Конфиг контейнера запечён в `/app/config.yaml` (`docker/config.container.yaml`): `data_dir: /data`, LLM через `host.docker.internal:1234`
- **torch И torchvision в образе — только CPU-индекс** — `uv pip install --system torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu` ДО установки пакета, иначе linux-wheel притянет CUDA (~4 ГБ). `torchvision` добавлен намеренно (не только `torch`) — PyPI-колесо `torchvision` бинарно несовместимо с CPU-сборкой `torch` с того же индекса и падает в рантайме (`RuntimeError: operator torchvision::nms does not exist`); оба пакета берутся с одного CPU-индекса одной командой. Версии запинены (первый коммит этапа 2); при осознанном апгрейде пары менять обе версии разом и проверять контейнерный тест-прогон
- **postgres в compose поднимается, но приложением не используется до этапа 2** — FileStorage остаётся рабочим бэкендом, данные в `/data`
- **api на этапе 1 — только `GET /health`** — REST каталога/чата появится на этапе 4

## Non-Goals (MVP)

Не используется в MVP: ChromaDB, FAISS, LangChain, OpenAI API, веб-интерфейс, БД

## Git workflow

- **`main`** — стабильная ветка, всегда рабочая
- **`dev`** — ветка для экспериментальных фич, worktree в `.claude/worktrees/dev/`
- Новые фичи разрабатываются в `dev`, после стабилизации мёрджатся в `main`

```bash
# Переключиться в dev worktree
cd .claude/worktrees/dev

# Список worktrees
git worktree list
```

## Claude Code Skills

Скилл для управления приложением: `.claude/skills/docling-rag-manager/SKILL.md`
- Активируется автоматически при обсуждении индексации, поиска, управления документами
- Покрывает: bootstrap (install + init), add, search, list + gotchas
- Протестирован через RED-GREEN-REFACTOR (writing-skills методология)

## Docs

Локальная документация в `docs/` (в .gitignore, не публикуется):
- `docs/PRD.md` — полная спецификация, P0/P1/P2
- `docs/ARCHITECTURE.md` — компонентная архитектура, потоки данных, инварианты
- `docs/FEATURES.md` — краткий фичелист со статусами
