"""Load chunks.jsonl into Postgres, embedding only what is new or changed.

Example:
    ledgerlens-index            # incremental: embeds new/changed chunks only
    ledgerlens-index --rebuild  # drop the table and re-embed everything
"""

import argparse
import hashlib
import json
import time

from ledgerlens.config import get_settings
from ledgerlens.retrieval.db import connect, drop_schema, init_schema
from ledgerlens.retrieval.embed import Embedder, contextual_text

COLUMNS = [
    "chunk_id", "ticker", "company", "form", "fiscal_year", "filing_date", "accession", "item",
    "section_title", "chunk_index", "is_table", "n_tokens", "text", "source_url",
]


def content_hash(chunk: dict, model_name: str) -> str:
    # Includes the model name so switching models re-embeds everything.
    return hashlib.sha1(f"{model_name}|{contextual_text(chunk)}".encode()).hexdigest()


def run(rebuild: bool = False, batch_size: int = 256) -> None:
    settings = get_settings()
    chunks = [json.loads(line) for line in settings.chunks_path.read_text().splitlines() if line]
    embedder = Embedder(settings.embedding_model)
    conn = connect(settings.database_url)

    if rebuild:
        drop_schema(conn)
    init_schema(conn, embedder.dim)

    stored = dict(conn.execute("SELECT chunk_id, content_hash FROM chunks").fetchall())
    hashes = {c["chunk_id"]: content_hash(c, settings.embedding_model) for c in chunks}
    todo = [c for c in chunks if stored.get(c["chunk_id"]) != hashes[c["chunk_id"]]]
    stale = set(stored) - set(hashes)

    if stale:
        conn.execute("DELETE FROM chunks WHERE chunk_id = ANY(%s)", (list(stale),))
    print(f"{len(chunks)} chunks in file, {len(todo)} to embed, {len(stale)} stale removed")

    started = time.monotonic()
    placeholders = ", ".join(["%s"] * (len(COLUMNS) + 2))
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in COLUMNS[1:] + ["content_hash", "embedding"])
    sql = (
        f"INSERT INTO chunks ({', '.join(COLUMNS)}, content_hash, embedding) "
        f"VALUES ({placeholders}) ON CONFLICT (chunk_id) DO UPDATE SET {updates}"
    )
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        vectors = embedder.embed_documents([contextual_text(c) for c in batch])
        with conn.cursor() as cur:
            cur.executemany(
                sql,
                [
                    [c[col] for col in COLUMNS] + [hashes[c["chunk_id"]], vec]
                    for c, vec in zip(batch, vectors)
                ],
            )
        done = i + len(batch)
        rate = done / (time.monotonic() - started)
        print(f"  embedded {done}/{len(todo)}  ({rate:.0f} chunks/s)", flush=True)

    total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    print(f"Index ready: {total} chunks in Postgres")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks and load them into Postgres.")
    parser.add_argument("--rebuild", action="store_true", help="drop and re-embed everything")
    args = parser.parse_args()
    run(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
