# storage/db_schema.py
"""DDL схемы pgvector-хранилища. Идемпотентно; используется CLI `init` и тестами."""
import psycopg

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    source_file text PRIMARY KEY,
    title       text,
    topic       text,
    tags        text[] NOT NULL DEFAULT '{}',
    added_at    timestamptz NOT NULL DEFAULT now()
);

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
"""


def init_schema(dsn: str) -> None:
    with psycopg.connect(dsn) as conn:
        conn.execute(DDL)
        conn.commit()
