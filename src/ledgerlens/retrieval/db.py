"""Postgres connection and schema.

One table holds every chunk with its metadata, its embedding (pgvector) and a
generated full-text column (tsvector), so keyword and vector search run on the
same rows and share the same filters.
"""

import psycopg
from pgvector.psycopg import register_vector

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      text PRIMARY KEY,
    ticker        text NOT NULL,
    company       text NOT NULL,
    form          text NOT NULL,
    fiscal_year   int  NOT NULL,
    filing_date   date NOT NULL,
    accession     text NOT NULL,
    item          text NOT NULL,
    section_title text NOT NULL,
    chunk_index   int  NOT NULL,
    is_table      boolean NOT NULL,
    n_tokens      int  NOT NULL,
    text          text NOT NULL,
    source_url    text NOT NULL,
    content_hash  text NOT NULL,
    embedding     vector({dim}) NOT NULL,
    tsv           tsvector GENERATED ALWAYS AS (
                      setweight(to_tsvector('english', section_title), 'A') ||
                      setweight(to_tsvector('english', text), 'B')
                  ) STORED
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_filter_idx ON chunks (ticker, fiscal_year, item);
"""


def connect(database_url: str) -> psycopg.Connection:
    conn = psycopg.connect(database_url, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection, dim: int) -> None:
    existing = conn.execute(
        "SELECT atttypmod FROM pg_attribute "
        "WHERE attrelid = to_regclass('chunks') AND attname = 'embedding'"
    ).fetchone()
    if existing and existing[0] != dim:
        raise RuntimeError(
            f"chunks.embedding has dimension {existing[0]} but the model produces {dim}. "
            "Re-index from scratch with `ledgerlens-index --rebuild`."
        )
    conn.execute(SCHEMA.format(dim=dim))


def drop_schema(conn: psycopg.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS chunks")
