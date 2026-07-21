# docling-rag

CLI-утилита для семантического поиска по технической документации на базе Docling.
RAG-система: Docling → chunking → Sentence Transformers → PostgreSQL+pgvector (HNSW cosine search).

**Статус:** MVP + document metadata + hybrid chunking + pydantic-ai agent реализованы; stage-0 рефакторинг (src-layout, идемпотентный add, exit-коды, Protocol-типизация, композируемый agent) завершён; v2 этап 1 (Docker) завершён — docker compose (postgres + api + cli), env-configurable volumes; v2 этап 2: открывающие коммиты (пины torch==2.13.0/torchvision==0.28.0 cpu, docling==2.113.0, pydantic-ai>=2.0,<3, split deps-слоя, пре-бейк RapidOCR, общий образ `docling-rag:local`) + **pgvector-миграция**: хранилище postgres-only (`DBStorage`/`DBRegistry`), embedding-модель `deepvk/USER-bge-m3` (1024d), команда `delete`, лог поиска в таблице `searches` (`DBSearchLog`), CLI стал docker-only; `FileStorage`/`DocRegistry`, флаг `--data-dir`, корневой `config.yaml`, мёртвый `save()` и файловый `log_file` удалены. **Этап 4-A (ingestion API)** + post-merge polish завершены: `POST /documents` (multipart, лимит `max_upload_mb`, стриминг на диск) → таблица `jobs` (postgres как очередь, `DBJobs`) → фоновый `worker`-сервис (claim через SKIP LOCKED, heartbeat, requeue_stale, переживает обрыв pg) → `GET /jobs/{id}`/`GET /jobs` (live-статус, elapsed заморожен у терминальных). **Этап 4-B (read-API + embed-сервис)** завершён: отдельный `embed`-сервис (единственный процесс с моделью USER-bge-m3, HTTP `POST /embed`) + `HTTPEmbedder`/`get_embedder(cfg)`-фактори (embed_url задан → HTTP-клиент, иначе локальная модель; используют cli/worker/api) → `GET /documents`/`GET /documents/{id}` (карточка: chunks + live indexing-статус) → `DELETE /documents/{id}` (запись + chunks + файл, 409 при активной джобе) → `GET /search` (HTTP-обёртка над тем же `run_search`, что CLI/agent, с логом в `searches`) → app-уровневые 503-хендлеры доменных ошибок хранилища/эмбеддера; схема получила `documents.id` (uuid, идемпотентная миграция). **Этап 4-C (chat-API)** завершён: `POST /chat` (JSON без стриминга, sources из tool-вызовов, история от клиента, лог агентских поисков — TODO п.5 закрыт), `instructions=` вместо `system_prompt=` (переживает message_history), `llm_timeout_sec`. Остаток этапа 4 — только D (веб-UI). 205 fast + 41 integration + 1 slow тест, все зелёные.

## Stack

- Python 3.10–3.12, Docling, Sentence Transformers (`deepvk/USER-bge-m3`), PostgreSQL 17 + pgvector, psycopg3, NumPy, Click, PyYAML

## Commands (docker) — основной способ запуска

CLI требует postgres, поэтому боевой запуск только через compose.

```bash
cp .env.example .env   # пути volumes и порты — правь под себя
docker compose up -d --wait postgres embed api worker   # embed :8100/health (модель), api :8000/health
docker compose run --rm cli init                      # DDL: extension + таблицы + HNSW (идемпотентно; повторный init после апгрейда — см. Gotchas)
docker compose run --rm cli add /books/my-book.pdf --title "My Book" --topic "..." --tag arch
docker compose run --rm cli search "запрос"           # + --tag/--topic/--top-k
docker compose run --rm cli list
docker compose run --rm cli delete /books/my-book.pdf  # документ + его chunks (каскад)
docker compose run --rm cli ask "вопрос"   # LM Studio на хосте, порт 1234
docker compose run --rm cli test tests/ -m "not integration and not slow"  # тесты в контейнере
docker compose --profile dev up api-dev    # hot-reload API на :8001
# update <file> — P1, не реализован

# read-API (этап 4-B), пример
curl -s -X POST http://localhost:8000/documents -F "file=@book.md" -F "title=My Book"  # 202 + job_id
curl -s http://localhost:8000/documents                     # каталог: chunks + indexing.status
curl -s "http://localhost:8000/search?q=запрос"              # семантический поиск по HTTP
curl -s -X DELETE http://localhost:8000/documents/<id>       # {deleted, chunks, file_removed}

# chat-API (этап 4-C), пример
curl -s -X POST http://localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"message": "Что такое Data Vault?"}'   # {answer, sources[]}
```

## Commands (dev на хосте — только тесты)

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,agent,api]"

python3 -m pytest tests/ -m "not integration and not slow"   # быстрые: 205 passed, 42 deselected (герметичны, postgres НЕ нужен)
docker compose up -d postgres                                # прекондишн для integration
python3 -m pytest tests/ -m integration                      # 41 passed (тест-БД docling_rag_test; первый прогон качает USER-bge-m3 ~2.3 ГБ)
```

## Architecture

Пакеты живут под `src/docling_rag/` (src-layout; editable install резолвится сюда, не в repo-root).

```
docling-rag/
├── src/docling_rag/
│   ├── cli/
│   │   ├── commands.py      # Click: init, add, search, list, delete, ask + exit-код контракт
│   │   └── config_loader.py # load_config(path, *, required=) + дефолты + приоритет env DATABASE_URL
│   ├── core/
│   │   ├── parser.py     # Docling парсер → DoclingDocument; SUPPORTED_EXTENSIONS = {.pdf, .docx, .md}
│   │   ├── chunker.py    # chunk_document(); HybridChunker кеширован per (embedding_model, max_tokens)
│   │   ├── embedder.py   # Embedder (Sentence Transformers, L2-норм.) + get_embedder(cfg) фактори: embed_url → HTTP, иначе локальная модель
│   │   ├── embed_client.py # HTTPEmbedder — EmbedderBackend без загрузки модели; POST {embed_url}/embed; сетевые ошибки → EmbedServiceUnavailableError
│   │   ├── indexer.py    # index_files(): file → parse → chunk → embed → store, per-file error isolation
│   │   ├── search.py     # run_search() + resolve_allowed_sources() — переиспользуются CLI search, GET /search и agent tool
│   │   ├── agent.py      # create_agent(model) + build_lmstudio_model(..., timeout_sec=); tool собирает deps.sources и логирует запрос в deps.search_log; требует .[agent]
│   │   ├── protocols.py  # StorageBackend/DocumentRegistryBackend/SearchLogBackend/JobBackend/EmbedderBackend Protocol — в аннотациях (search.py, indexer.py, commands.py, api/app.py)
│   │   └── errors.py     # StorageError, StorageUnavailableError, StorageSchemaMissingError, UnsupportedFormatError, LLMUnavailableError, EmbedServiceUnavailableError
│   ├── api/
│   │   ├── app.py        # FastAPI: GET /health, POST /documents (ingestion), GET /jobs/{id}, GET /jobs, GET/DELETE /documents(/{id}), GET /search, POST /chat; 503-хендлеры доменных ошибок; требует .[api]
│   │   └── embed_app.py  # embed-сервис: create_app() грузит USER-bge-m3 блокирующе, POST /embed {texts} -> {embeddings, model, dim}
│   ├── worker/
│   │   ├── runner.py     # process_one_job/_Heartbeat/run_loop — фоновая индексация джоб; вне core/ (импортирует storage)
│   │   └── __main__.py   # точка входа контейнерного worker-сервиса: build_deps(cfg) (embedder через get_embedder) + run_loop
│   └── storage/
│       ├── db_schema.py     # DDL: CREATE EXTENSION vector + documents(+id uuid) + chunks + searches + jobs + HNSW-индекс; init_schema(dsn), идемпотентно
│       ├── db_storage.py    # chunks+embeddings в pg (StorageBackend impl); _translate_db_errors psycopg→доменные
│       ├── db_registry.py   # documents: title/topic/tags/added_at + id (DocumentRegistryBackend impl); get_by_id(doc_id) для REST-адресации
│       ├── db_search_log.py # searches: query/top_score/searched_at (SearchLogBackend impl)
│       └── db_jobs.py       # jobs: очередь фоновой индексации (JobBackend impl); claim_next через FOR UPDATE SKIP LOCKED
├── tests/                   # tests/core/, tests/storage/, tests/api/, tests/fakes.py, tests/test_*.py — 205 fast + 41 integration + 1 slow
├── Dockerfile               # multi-stage: frontend-заглушка (node) + runtime (python+uv); deps-слой отделён от src, RapidOCR-модели запечены; entrypoint-диспетчер api/embed/test/cli
├── compose.yaml             # postgres + embed + api + worker + api-dev (profile dev) + cli (profile cli); DATABASE_URL в environment, bind-mounts из .env
├── .env.example             # PGDATA_DIR/HF_CACHE_DIR/UPLOADS_DIR/BOOKS_DIR + порты + POSTGRES_*
└── docker/
    ├── entrypoint.sh          # api → uvicorn :8000; embed → uvicorn --factory embed_app:create_app :8100; test → pytest; иначе → docling-rag CLI
    └── config.container.yaml  # запекается в /app/config.yaml: database_url на хост `postgres`, embed_url на хост `embed`, LLM через host.docker.internal:1234
```

Дефолты конфига живут в коде (`cli/config_loader.py::_DEFAULTS`) — репозиторного `config.yaml` НЕТ (удалён: был побайтовым дублем дефолтов и молча их перекрывал). Свой `config.yaml` в cwd или `--config PATH` опциональны.

Схема БД (`db_schema.py`): `documents(source_file PK, id uuid UNIQUE, title, topic, tags text[], added_at)` ← `chunks(id, source_file FK ON DELETE CASCADE, chunk_id, page_number, text, headings jsonb, element_type, embedding vector(1024))` + `chunks_embedding_hnsw` (hnsw, `vector_cosine_ops`); `searches(id, query, top_score, searched_at)` — независимая, без FK на documents (запрос переживает удаление документа); `jobs(id uuid PK, source_file, original_name, title/topic/tags, status queued|running|done|failed, step, chunks_total/done, error, attempts, created/started/updated/finished_at)` — очередь ingestion, без FK (история переживает удаление документа); `jobs.source_file` резолвлен и равен `documents.source_file`. `documents.id` (uuid, `gen_random_uuid()`) + `documents_id_key` (unique index) добавлены этапом 4-B идемпотентной миграцией внутри того же DDL (`ALTER TABLE ... ADD COLUMN IF NOT EXISTS` + `CREATE UNIQUE INDEX IF NOT EXISTS`) — `source_file` остаётся PK, FK-цепочка не меняется, `id` только для REST-адресации карточек.

## Gotchas

### Хранилище (pgvector)

- **postgres — ЕДИНСТВЕННОЕ хранилище** (с pgvector-миграции этапа 2). Файлового бэкенда нет: `FileStorage`/`DocRegistry`, `data_dir`, `--data-dir`, `/data`-маунты удалены. Индекс живёт в `PGDATA_DIR`
- **DSN: env `DATABASE_URL` > ключ `database_url` в config.yaml** — приоритет реализован в `config_loader.load_config()` (env применяется ПОСЛЕ `cfg.update(user_cfg)`). Дефолт ключа — `postgresql://docling:docling@127.0.0.1:5432/docling_rag`; в compose всем сервисам (api/api-dev/cli) выставлен `DATABASE_URL` на хост `postgres`. `database_url` в `docker/config.container.yaml` — понятный фолбэк для `docker run` без compose
- **`vector(1024)` — литерал в DDL, привязан к USER-bge-m3** — смена embedding-модели требует правки DDL И полной переиндексации (не только «переиндексации», как было с NumPy)
- **`init` = только DDL** — `init_schema(dsn)` идемпотентен (`CREATE EXTENSION/TABLE/INDEX IF NOT EXISTS`), печатает `Схема БД инициализирована: <DSN с замаскированным паролем>`; `_mask_dsn()` прячет пароль во ВСЕХ сообщениях. Папок больше не создаёт (файлового лога нет)
- **Лог поиска — таблица `searches`, НЕ файл** — `DBSearchLog.log(query, top_score)` пишется в БД на каждом успешном `search` (при пустом результате не пишется — `top_score` берётся из `results[0]`). Отказ лога НЕ роняет поиск: `except Exception` → предупреждение в stderr. Причина замены: после удаления маунта `/data` файловый лог шёл в `/tmp` контейнера и умирал с `--rm`. Ключ `log_file` и функция `_log_search` удалены
- **`config.yaml` в репозитории НЕТ** — дефолты в `cli/config_loader.py::_DEFAULTS`. Файл-дубликат дефолтов удалён (молча их перекрывал, создавал второй источник правды). `load_config(required=False)` без файла тихо берёт `_DEFAULTS`; свой `config.yaml`/`--config PATH` опциональны. В образе — свой `/app/config.yaml` (`docker/config.container.yaml`)
- **Доменные ошибки хранилища НЕ наследуют `StorageError`** (`core/errors.py`) — иначе существующие `except StorageError` («Хранилище повреждено») перехватили бы инфраструктурные сбои. `storage/db_storage.py::_translate_db_errors()` (contextmanager, используется и `DBRegistry`) переводит `psycopg.OperationalError` → `StorageUnavailableError` («PostgreSQL недоступен» + подсказка `docker compose up -d postgres`), `psycopg.errors.UndefinedTable` → `StorageSchemaMissingError` («Выполните: docling-rag init»), `ProgrammingError` с "vector" в тексте → `StorageSchemaMissingError`. Неожиданные ошибки пробрасываются как есть. **`core/` и `cli/` не импортируют psycopg** — тест-стражник `test_core_does_not_import_storage_package` остаётся зелёным
- **Соединение открывается на операцию** (`DBStorage._connect()`), пула нет — CLI короткоживущий. Конструирование `DBStorage(dsn)`/`DBRegistry(dsn)` к БД НЕ подключается (на этом стоит `tests/core/test_protocols.py` с фиктивным DSN)
- **`headings`: list → json-строка на запись, list на чтение** — psycopg не адаптирует list в jsonb автоматически, поэтому `json.dumps(...)` в `_insert`; из jsonb приходит уже list. Формат dict метаданных chunk'а сохранён байт-в-байт (`text`, `source_file`, `chunk_id`, `page_number`, `element_type`, `headings`)
- **`_to_numpy()` в db_storage** — pgvector-loader возвращает объект `pgvector.Vector` (не list/ndarray) для колонки `embedding`; конвертация явная
- **FK-порядок: `append` сам создаёт родительскую строку** — `indexer` вызывает `storage.append()` ДО `registry.upsert()`, поэтому `_insert` делает `INSERT INTO documents (source_file) ... ON CONFLICT DO NOTHING` перед вставкой chunks, иначе FK падает
- **Контракт пустого хранилища сохранён** — `load()`/`search()` на пустой БД → `FileNotFoundError` (на этом держится exit-код контракт CLI). `search` сначала проверяет `SELECT NOT EXISTS (SELECT 1 FROM chunks)`
- **`delete SOURCE`** — ключ резолвится (`Path(source).resolve()`), если файл существует, иначе берётся строка как есть (осиротевшие записи удаляемы). `registry.delete()` сносит chunks каскадом, `storage.delete_by_source()` — идемпотентная страховка. Ничего не найдено (нет записи И 0 chunks) → `ClickException` «Документ не найден» + подсказка `docling-rag list`, exit 1. Вывод: `Удалено: <title|key> (N chunks)`

### Embedding-модель и чанкинг

- **`deepvk/USER-bge-m3`, 1024d, БЕЗ префиксов query:/passage:** — модель качается один раз (~2.3 ГБ) в `HF_CACHE_DIR` (`HF_HOME=/hf-cache` в образе)
- **Имя модели с `/` (org) используется как есть; без `/` — приклеивается `sentence-transformers/`** — `core/chunker.py::_get_chunker()` резолвит `model_id` для `HuggingFaceTokenizer.from_pretrained`
- **`chunk_max_tokens: 512` — явный ключ конфига, не авто** — у bge-m3 окно 8192 токенов; авто-лимит из tokenizer'а дал бы чанки, убивающие гранулярность поиска. Прокидывается `cli/commands.py::add` → `index_files(..., chunk_max_tokens=...)` → `chunk_document(..., max_tokens=...)`
- **HybridChunker кеширован per (embedding_model, max_tokens)** — `@lru_cache(maxsize=4)` на `_get_chunker(embedding_model, max_tokens)`; в тестах чистить `_get_chunker.cache_clear()`
- **Токенайзер чанкера создаётся с `model_max_length=sys.maxsize`** — он только СЧИТАЕТ токены (лимит чанков — явный `max_tokens`), но transformers сравнивает длину с `model_max_length` из конфига модели и шумит «Token indices sequence length is longer than ...» на секциях длиннее окна; ids в модель не идут, предупреждение ложное — глушится на источнике
- **Одна embedding-модель для индексации и поиска** — нельзя менять модель без полной переиндексации (и правки `vector(N)` в DDL)
- **context_text vs text** — `chunk.context_text` = headings + text (используется для эмбеддингов); `chunk.text` = чистый текст (хранится и отображается в поиске)
- **Таблицы и code-блоки** — HybridChunker сохраняет их как атомарные chunks (element_type = "table" или "code")

### Тесты

- **Быстрый суит герметичен: postgres НЕ нужен** — юниты работают на `tests/fakes.py::InMemoryStorage`/`InMemoryRegistry`/`InMemorySearchLog` (реализуют Protocol'ы, семантика та же: пустое → `FileNotFoundError`). Проверка герметичности: `docker compose stop postgres && pytest -m "not integration and not slow"` — зелёный
- **Два autouse-фикстуры герметизации** (`tests/conftest.py`): `hermetic_config` патчит `load_config`; `hermetic_search_log` патчит `docling_rag.cli.commands.DBSearchLog` на in-memory фейк — иначе КАЖДЫЙ `search`-тест открывал бы соединение к несоединяемому DSN и печатал предупреждение, маскируя реальные сбои. `e2e_config` осознанно переопределяет ОБЕ (зависит от них явно): реальная тест-БД + реальная модель + НАСТОЯЩИЙ `DBSearchLog` (иначе сквозной путь логирования в БД остался бы непокрытым — там и пряталась регрессия)
- **CLI mock-паттерн (актуальный)** — фикстура `fake_backends` (`tests/conftest.py`) патчит `docling_rag.cli.commands.DBStorage`/`.DBRegistry` на in-memory fake'и и отдаёт `(storage, registry)` для сидирования. Патчить `docling_rag.cli.commands.Parser` / `.get_embedder` (НЕ `.Embedder` — с этапа 4-B cli/worker создают эмбеддер через фактори, `MockEmbedder.return_value.embed.return_value = ...`); `init` — патчить `docling_rag.cli.commands.init_schema`; лог — `docling_rag.cli.commands.DBSearchLog`; `chunk_document` вызывается из `core/indexer.py` → патчить `docling_rag.core.indexer.chunk_document`, НЕ `cli.commands.chunk_document`. `ask` — патчить `docling_rag.cli.commands._create_and_run_agent` (сигнатура `(question, cfg, top_k) -> str`, без `data_dir`). API-тесты поиска — `app.dependency_overrides[get_search_embedder] = lambda: FakeEmbedder()` (см. tests/api/test_search_endpoint.py)
- **Герметичный дефолт `database_url` — порт 1** (`tests/conftest.py::_HERMETIC_DEFAULTS`): `postgresql://test:test@127.0.0.1:1/test` — юнит, случайно дошедший до реального соединения, падает быстро и громко. `embedding_model` в герметичных дефолтах — `all-MiniLM-L6-v2` (не тянуть 2.3 ГБ в юнитах)
- **Integration-тесты — ОТДЕЛЬНАЯ БД `docling_rag_test`**, боевая `docling_rag` не трогается. Фикстуры `db_url` (создаёт БД + схему, `pytest.skip` если postgres недоступен) и `clean_db` (`TRUNCATE documents CASCADE`) живут в `tests/storage/test_db_backends.py` и реэкспортируются в `tests/conftest.py` для e2e
- **`e2e_config` осознанно переопределяет autouse `hermetic_config`** — зависит от него явно (порядок фикстур), ре-патчит `load_config` ПОСЛЕ герметичного патча на реальную тест-БД + `deepvk/USER-bge-m3`; function-scoped monkeypatch откатывает оба патча в обратном порядке
- **Счётчики** — 205 fast (42 deselected), 41 integration, 1 slow

### CLI-контракты

- **Exit-код контракт** — `click.ClickException` (невалидный `--config`, `StorageError`, `StorageUnavailableError`, `StorageSchemaMissingError`, ошибки агента, `agent_enabled: false`, `delete` несуществующего) → exit 1. Пустое хранилище (`FileNotFoundError`) → exit 1 только в `search` и `ask`; `list` на пустом хранилище печатает «Хранилище пустое. Документов нет.» и завершается с exit 0 (тест `test_list_command_empty_storage`). `add` дополнительно делает `raise SystemExit(1)`, если есть `files_failed` ИЛИ `chunks_added == 0`. Валидация параметров Click (`--top-k 0` через `IntRange(min=1)`, несуществующий `file_path` через `click.Path(exists=True)`) → `UsageError`, exit 2. Ошибки — в stderr (`err=True`)
- **`--config` на ВСЕХ командах** — `init`, `add`, `search`, `list`, `delete`, `ask`. Контракт `load_config(path, *, required=)`: без явного `--config` читается `config.yaml` в cwd с `required=False` (нет файла → тихий fallback на дефолты); при явном `--config PATH` — `required=True`, файла нет → `ConfigError` → `ClickException` (exit 1). Невалидный YAML или не-dict корень → `ConfigError` независимо от `required`. Неизвестные ключи → warning в stderr
- **top-k по умолчанию из config** — `--top-k` без значения берёт `top_k_results`
- **SUPPORTED_EXTENSIONS — единый источник** — `{".pdf", ".docx", ".md"}` только в `core/parser.py`; `.txt` убран (Docling обрабатывал некорректно). `cli/commands.py` импортирует эту же константу
- **Идемпотентный `add` через резолвленные пути** — `index_files()` резолвит путь (`Path(file).resolve()`) ДО `chunk_document`/`registry.upsert`; перед `storage.append()` вызывается `storage.delete_by_source(source)`; `registry.upsert` сохраняет `added_at` и не затирает title/topic/tags значениями `None`/пустыми (в SQL — `COALESCE` + `CASE WHEN cardinality(EXCLUDED.tags) > 0`)
- **Изоляция ошибок по файлу в `index_files()`** — путь инициализируется как `str(file)` ДО `try`, резолвится внутри `try`; падение `.resolve()` (symlink loop, permission) попадает в `report.errors`, batch не прерывается
- **Фильтр поиска: пустой match → пустые результаты** — `--tag`/`--topic` не совпал ни с одним документом → пустой список (не fallback на все документы). `--topic` сравнивается case-insensitive
- **core/protocols.py — Protocol-абстракция, реально используется в аннотациях** — `StorageBackend` (`append`/`load`/`delete_by_source`/`count_by_source`/`search` — `save()` удалён как мёртвый), `DocumentRegistryBackend` и `SearchLogBackend` (`log`) типизируют `core/search.py`, `core/indexer.py`, `cli/commands.py`. Тест `test_core_does_not_import_storage_package` проверяет источниковым grep'ом `core/search.py`, `core/indexer.py`, `core/agent.py` (для agent.py — чтение исходника с диска, без импорта: тест не должен падать без `.[agent]`)
- **LLM только в `ask`** — `search` возвращает raw chunks со score, ответы не генерирует
- **`ask` требует `.[agent]` и LM Studio** — `agent_enabled: true` (в контейнерном конфиге уже true), LM Studio на `127.0.0.1:1234` (из контейнера — `host.docker.internal:1234`)
- **Обнаружение ошибок соединения с LLM — isinstance по цепочке cause/context, НЕ строковый матч** — `cli/commands.py::_is_connection_error(e)` идёт по `e.__cause__ or e.__context__` и проверяет `isinstance(cur, (ConnectionError, httpx.ConnectError, httpx.ConnectTimeout))`
- **Lazy import + testability** — `_import_agent_module()` возвращает `(create_agent, AgentDeps, build_lmstudio_model)`; патчится через `patch("docling_rag.cli.commands._import_agent_module")`
- **pydantic-ai API — composable `create_agent(model)`** — принимает ЛЮБУЮ pydantic-ai `Model` (включая `TestModel`), не строит модель сама; `build_lmstudio_model(model_name, base_url, api_key) -> OpenAIChatModel` собирает `OpenAIChatModel(model_name, provider=OpenAIProvider(base_url=..., api_key=...))` отдельно — LM Studio говорит на Chat Completions API, поэтому явный `OpenAIChatModel`, а не `"openai:"`-префикс (тот означал бы Responses API). Импорты: `from pydantic_ai import Agent, RunContext`; `from pydantic_ai.models.openai import OpenAIChatModel`; `from pydantic_ai.providers.openai import OpenAIProvider`; `result.output` для ответа

### Ingestion API и worker (этап 4-A)

- **postgres как очередь, без Celery/Redis** — `POST /documents` пишет строку в `jobs` (202 + job_id), CPU-тяжёлая индексация в отдельном `worker`-сервисе; `claim_next` через `FOR UPDATE SKIP LOCKED` (есть integration-тест на конкурентный claim с таймаут-стражем)
- **`POST /documents` — sync def намеренно** — FastAPI уводит его в threadpool, файловый и БД I/O не блокируют event loop. Аплоад пишется чанками 1 МБ в `<dest>.part` + атомарный `os.replace` (весь файл в память не читается); лимит — ключ конфига `max_upload_mb` (дефолт 100), превышение → 413, прежний файл цел
- **`jobs.source_file` резолвится в API** (`Path.resolve()`) — равен `documents.source_file` (indexer резолвит так же); на этом строится корреляция джоба↔документ в каталоге 4-B (`find_latest_by_source`). Дедуп: активная (queued/running) джоба с тем же source_file → 409 + существующий job_id. Partial unique index против TOCTOU отклонён (YAGNI, однопользовательский инструмент)
- **Live-статус** — `GET /jobs/{id}`: `elapsed_sec`/`heartbeat_age_sec` считаются от `started_at`/`updated_at`; у терминальных (done/failed) `now` зажат до `finished_at` — значения заморожены. `?status=` валидируется Literal'ом (мусор → 422). Хартбит: тикер раз в ~10с + каждый тик прогресса; зависшие running возвращает `requeue_stale` (порог 60с, лимит попыток 3)
- **Воркер переживает обрыв postgres** — итерация `run_loop` в try/except (stderr + пауза + повтор), плюс `restart: unless-stopped` на сервисе; недообработанная джоба вернётся через requeue_stale
- **`get_settings` кеширован (lru_cache)** — конфиг читается раз на процесс API; смена config.yaml требует рестарта. В тестах — `app.dependency_overrides` (кеш не мешает), тестовые оверрайды settings должны включать `uploads_dir` и `max_upload_mb`
- **`jobs` — история без FK** — строки джоб переживают удаление документа (как `searches`); `cli delete` их не трогает

### Read-API, поиск и embed-сервис (этап 4-B)

- **`embed`-сервис — единственный процесс с моделью USER-bge-m3** — `api/embed_app.py::create_app()` грузит модель блокирующе ДО старта uvicorn (готовность сервера == готовность модели); `POST /embed {"texts": [...]}` → `{"embeddings": [[...]], "model", "dim"}`. compose: сервис `embed`, healthcheck на GET `:8100/health` (`start_period: 180s` — время загрузки модели); `api`/`worker`/`cli` зависят от `embed: condition: service_healthy`
- **`get_embedder(cfg)` — единственная точка выбора эмбеддинг-бэкенда** (`core/embedder.py`) — ключ конфига `embed_url` задан → `HTTPEmbedder(embed_url)` (`core/embed_client.py`, HTTP-клиент, модель НЕ грузит); иначе → локальный `Embedder(embedding_model)`. `cli`, `worker` и `api` (для `GET /search`) идут через эту фактори. В контейнерном конфиге `embed_url: http://embed:8100`; на хосте (dev/тесты) ключ `None` → локальная модель (герметичные тесты не поднимают embed)
- **`HTTPEmbedder.embed(texts, batch_size=32)` принимает `batch_size`, но игнорирует его** — сохранено ради совместимости сигнатуры с `Embedder.embed()` (`indexer.py` передаёт `batch_size` явно); батчинг — забота embed-сервиса, не клиента. Сетевые сбои (connect/timeout/5xx) → `EmbedServiceUnavailableError` (НЕ наследует `StorageError`, как и остальные инфраструктурные ошибки)
- **503-хендлеры на уровне приложения** (`api/app.py`) — `StorageUnavailableError`, `StorageSchemaMissingError`, `EmbedServiceUnavailableError` перехвачены `@app.exception_handler(...)` и превращены в JSON `503` с понятным `detail` (для отсутствующей схемы — подсказка `docling-rag init`); сами эндпоинты try/except вокруг них не оборачивают
- **`documents.id` — суррогатный uuid для REST-адресации карточек** — `source_file` остаётся PK (FK `chunks` и корреляция `jobs` не меняются); `DBRegistry.get_by_id(doc_id)` валидирует uuid-формат до похода в БД (не-uuid → `None` → 404, а не 500)
- **После апгрейда на 4-B на долгоживущих postgres-инстансах нужен повторный `docker compose run --rm cli init`** — миграция `documents.id`/`documents_id_key` идемпотентна, но применяется только явным `init`, НЕ автоматически при старте `api`/`worker`. Без неё `GET /documents` и `GET /search` отвечают `503` «Выполните: docling-rag init» (`UndefinedColumn` → `StorageSchemaMissingError` в `_translate_db_errors`; обнаружено на живой приёмке 2026-07-19 ещё как сырой 500 — postgres с 34-часовым аптаймом был без миграции, `init` починил)
- **Каталог** — `GET /documents` (карточки: `id`, `source_file`, `title/topic/tags`, `added_at`, `chunks` через `count_by_source`, `indexing: {status, job_id}|null` — самая свежая джоба по `find_latest_by_source`), отсортирован по `added_at` убыв.; `GET /documents/{id}` — та же карточка по одному документу, 404 если не найден
- **`DELETE /documents/{id}`** — 404 если не найден, 409 если есть активная (queued/running) джоба на этот source (иначе воркер пересоздал бы документ после удаления); удаляет registry-запись (FK-каскад сносит chunks) + `storage.delete_by_source` (идемпотентная страховка) + сам файл, но только если он лежит в `uploads_dir` (книги из `BOOKS_DIR`/`/books` не трогает); отвечает `{deleted, chunks, file_removed}`. `jobs`-история не удаляется
- **`GET /search`** — HTTP-обёртка над тем же `run_search`/`resolve_allowed_sources`, что CLI `search` и agent tool; `?tag=`/`?topic=` фильтруют как в CLI (пустой match → пустые `results`, не fallback на все документы); пустое хранилище (`FileNotFoundError`) → `{"results": []}`, а не 5xx (у HTTP нет exit-код контракта CLI); непустой результат логируется в `searches` (`DBSearchLog`, отказ лога не роняет запрос). **Agent tool с этапа 4-C тоже логирует** (`AgentDeps.search_log`, см. подраздел «Chat-API» ниже) — docs/TODO.md п.5 закрыт
- **Эмбеддер для поиска синглтон-кеширован на процесс api** — `_embedder_singleton(embed_url, model)` (`@lru_cache(maxsize=2)`) не пересоздаёт `HTTPEmbedder`/`Embedder` на каждый запрос `GET /search`

### Chat-API (этап 4-C)

- **`instructions=` вместо `system_prompt=` в `create_agent`** — при непустом `message_history` pydantic-ai НЕ отправляет `system_prompt`, `instructions` отправляются каждый run (иначе чат с историей без RAG-правил); тест `test_static_instructions_survive_message_history`
- **`POST /chat` — sync def + `run_sync`** (threadpool, как `POST /documents`); история — role/content-пары от клиента, tool-вызовы прошлых ходов не реплеятся
- **Пустое хранилище в `/chat` → 200 с каноническим answer**, не 5xx (контракт как `GET /search` → пустые results)
- **Порядок except в `/chat`**: доменные Storage/Embed-ошибки re-raise ДО connect-эвристики (иначе psycopg-цепочка винит LM Studio) — тот же принцип, что в `cli ask`
- **Лог агентских поисков** — tool пишет в `deps.search_log` (query агента, не вопрос пользователя; пустая выдача не логируется; отказ лога не роняет run). TODO п.5 закрыт
- Тесты: `hermetic_search_log` теперь отдаёт лог и агентскому пути CLI `ask`; API-тесты чата — `dependency_overrides[get_chat_model] = lambda: TestModel()`

### Docker

- **Все volumes — bind-mounts из `.env`** — требование пользователя: расположение данных выбирает он. `PGDATA_DIR`/`HF_CACHE_DIR`/`UPLOADS_DIR`/`BOOKS_DIR`, дефолты `./volumes/*` и `./books`. Named volumes в compose НЕ использовать
- **`cli` зависит от healthy postgres** (`depends_on: postgres: condition: service_healthy`) — негативный сценарий «postgres лежит» воспроизводится через `docker compose run --rm --no-deps cli ...`
- **Entrypoint-диспетчер образа** — `api` → uvicorn :8000, `embed` → `uvicorn --factory embed_app:create_app` :8100, `test` → pytest, иначе → CLI `docling-rag`. Конфиг контейнера запечён в `/app/config.yaml` (`docker/config.container.yaml`)
- **torch И torchvision в образе — только CPU-индекс** — `uv pip install --system torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu` ДО установки пакета, иначе linux-wheel притянет CUDA (~4 ГБ). `torchvision` добавлен намеренно — PyPI-колесо бинарно несовместимо с CPU-сборкой torch и падает в рантайме (`RuntimeError: operator torchvision::nms does not exist`); оба пакета — с одного CPU-индекса одной командой. Версии запинены; при осознанном апгрейде менять обе разом и проверять контейнерный тест-прогон. Дрейф ловится assert-слоем сразу после deps-установки
- **Пре-бейк RapidOCR-моделей в образе** — rapidocr работает на torch-движке (onnxruntime не ставится) и качает ~16 МБ .pth-моделей при первом парсе PDF; Dockerfile конструирует `RapidOcrModel(... backend='torch')` на этапе сборки. Внутренний импорт `docling.models.stages.ocr.rapid_ocr_model` защищён пином `docling==2.113.0` — апгрейд docling осознанный (пин + ребилд + контейнерный прогон)
- **api теперь покрывает ingestion + read-API + chat-API** — `GET /health`, `POST /documents`, `GET /jobs(/{id})`, `GET/DELETE /documents(/{id})`, `GET /search`, `POST /chat`; веб-UI — этап 4-D
- **OCR-кириллица не поддержана** — bundled-наборы RapidOCR в docling: english/latin/chinese. Актуально только для сканов без текстового слоя; нужен другой движок (easyocr/tesseract) — дальний бэклог

## Non-Goals

Не используется: ChromaDB, FAISS, LangChain, OpenAI API (внешний), веб-интерфейс (этап 4-D)

## Git workflow

- **`main`** — стабильная ветка, всегда рабочая
- Фичи разрабатываются в ветках `feat/*` / `fix/*` от `main`, после ревью и зелёных тестов мёрджатся в `main`
- Worktree создаются ad-hoc при необходимости изоляции (`git worktree add ...`); постоянных worktree нет

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
