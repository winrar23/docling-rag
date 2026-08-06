# storage/db_schema.py
"""DDL схемы pgvector-хранилища. Идемпотентно; используется CLI `init` и тестами."""
import psycopg

from docling_rag.core.errors import StorageUnavailableError

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    source_file text PRIMARY KEY,
    id          uuid NOT NULL DEFAULT gen_random_uuid(),
    title       text,
    author      text,
    topic       text,
    tags        text[] NOT NULL DEFAULT '{}',
    added_at    timestamptz NOT NULL DEFAULT now()
);

-- Суррогатный id для REST-адресации карточек (этап 4-B); source_file остаётся PK,
-- FK chunks и корреляция jobs не меняются. Идемпотентная миграция для существующих баз.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS id uuid NOT NULL DEFAULT gen_random_uuid();
CREATE UNIQUE INDEX IF NOT EXISTS documents_id_key ON documents (id);

CREATE TABLE IF NOT EXISTS chunks (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_file  text NOT NULL REFERENCES documents(source_file) ON DELETE CASCADE,
    chunk_id     int NOT NULL,
    page_number  int NOT NULL DEFAULT 1,
    text         text NOT NULL,
    headings     jsonb NOT NULL DEFAULT '[]',
    element_type text NOT NULL DEFAULT 'text',
    embedding    vector(1024) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Лог поисковых запросов. Не связан с documents: запрос переживает удаление документа,
-- по которому нашёлся, иначе статистика молча теряла бы историю при delete.
CREATE TABLE IF NOT EXISTS searches (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    query       text NOT NULL,
    top_score   real,
    searched_at timestamptz NOT NULL DEFAULT now()
);

-- Фоновые джобы индексации (этап 4 A). id — uuid (gen_random_uuid встроена в pg13+).
CREATE TABLE IF NOT EXISTS jobs (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_file   text NOT NULL,
    original_name text NOT NULL,
    title         text,
    topic         text,
    tags          text[] NOT NULL DEFAULT '{}',
    ocr           text NOT NULL DEFAULT 'auto',
    ocr_lang      text NOT NULL DEFAULT 'en',
    status        text NOT NULL DEFAULT 'queued',
    step          text,
    chunks_total  integer,
    chunks_done   integer,
    error         text,
    warning       text,
    attempts      integer NOT NULL DEFAULT 0,
    created_at    timestamptz NOT NULL DEFAULT now(),
    started_at    timestamptz,
    updated_at    timestamptz NOT NULL DEFAULT now(),
    finished_at   timestamptz
);
CREATE INDEX IF NOT EXISTS jobs_status_created_idx ON jobs (status, created_at);

-- Параметры OCR per-job (авто-OCR, 2026-07-21). Идемпотентная миграция для
-- существующих баз — тот же приём, что documents.id в 4-B; применяется явным init.
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS ocr text NOT NULL DEFAULT 'auto';
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS ocr_lang text NOT NULL DEFAULT 'en';

-- Авто-метаданные (2026-08-06): author в documents, warning шага metadata в jobs.
-- Идемпотентная миграция для существующих баз; применяется явным init.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS author text;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS warning text;
"""


def init_schema(dsn: str) -> None:
    try:
        with psycopg.connect(dsn) as conn:
            conn.execute(DDL)
            conn.commit()
    except psycopg.OperationalError as e:
        raise StorageUnavailableError(str(e)) from e
