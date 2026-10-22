"""Hybrid search: keyword + vector retrieval, fused with RRF, then reranked.

Why both retrievers:
  * keyword (Postgres full-text) nails exact terms: "R&D", "iPhone", "CET1"
  * vector (pgvector) catches meaning: "supply problems" ~ "component shortages"
Reciprocal Rank Fusion merges the two ranked lists without having to calibrate
their very different scores, and a cross-encoder reranker then reads each
(question, chunk) pair together to pick the final top k.

Example:
    ledgerlens-search "How much did Apple spend on R&D?" --ticker AAPL --year 2025
"""

import argparse
import logging
import textwrap
from dataclasses import dataclass, field

import psycopg

from ledgerlens.config import get_settings
from ledgerlens.retrieval.db import connect
from ledgerlens.retrieval.embed import Embedder, Reranker, contextual_text

log = logging.getLogger("ledgerlens.search")

RRF_K = 60
CANDIDATES = 40  # per retriever
RERANK_POOL = 15  # fused candidates the cross-encoder reads (30 was 2x slower, same accuracy)
MODES = ("keyword", "vector", "hybrid", "rerank")

RESULT_COLUMNS = (
    "chunk_id, ticker, company, form, fiscal_year, item, section_title, "
    "is_table, text, source_url"
)


@dataclass
class Filters:
    tickers: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    items: list[str] = field(default_factory=list)
    forms: list[str] = field(default_factory=list)

    def sql(self) -> tuple[str, list]:
        clauses, params = [], []
        for column, values in (
            ("ticker", [t.upper() for t in self.tickers]),
            ("fiscal_year", self.years),
            ("item", [i.upper() for i in self.items]),
            ("form", self.forms),
        ):
            if values:
                clauses.append(f"{column} = ANY(%s)")
                params.append(values)
        return (" AND ".join(clauses) or "TRUE"), params


@dataclass
class Result:
    chunk: dict
    score: float
    found_by: list[str]

    @property
    def citation(self) -> str:
        c = self.chunk
        return f"{c['ticker']} {c['form']} FY{c['fiscal_year']}, Item {c['item']} ({c['section_title']})"


def reciprocal_rank_fusion(rankings: dict[str, list[str]], k: int = RRF_K) -> list[tuple[str, float, list[str]]]:
    """Merge ranked id lists: score = sum over lists of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    for name, ids in rankings.items():
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            sources.setdefault(chunk_id, []).append(name)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [(cid, scores[cid], sources[cid]) for cid in ordered]


class Searcher:
    def __init__(self, database_url: str, embedding_model: str, reranker_model: str):
        self.conn = connect(database_url)
        self.embedder = Embedder(embedding_model)
        self.reranker = Reranker(reranker_model)
        self.conn.execute("SET hnsw.ef_search = 100")
        try:
            # pgvector >= 0.8: keep scanning the HNSW graph until filtered results fill the limit.
            self.conn.execute("SET hnsw.iterative_scan = relaxed_order")
        except psycopg.Error:
            log.warning("pgvector < 0.8: filtered vector search may return fewer than k results")

    @classmethod
    def from_settings(cls) -> "Searcher":
        s = get_settings()
        return cls(s.database_url, s.embedding_model, s.reranker_model)

    def keyword(self, query: str, filters: Filters, limit: int = CANDIDATES) -> list[str]:
        where, params = filters.sql()
        # OR the query terms together (plainto_tsquery ANDs them, which is too strict for questions).
        rows = self.conn.execute(
            f"""
            WITH q AS (SELECT replace(plainto_tsquery('english', %s)::text, '&', '|') AS t)
            SELECT chunk_id FROM chunks, q
            WHERE t <> '' AND tsv @@ t::tsquery AND {where}
            ORDER BY ts_rank_cd(tsv, t::tsquery, 1) DESC
            LIMIT %s
            """,
            [query, *params, limit],
        ).fetchall()
        return [r[0] for r in rows]

    def vector(self, query: str, filters: Filters, limit: int = CANDIDATES) -> list[str]:
        where, params = filters.sql()
        rows = self.conn.execute(
            f"SELECT chunk_id FROM chunks WHERE {where} ORDER BY embedding <=> %s LIMIT %s",
            [*params, self.embedder.embed_query(query), limit],
        ).fetchall()
        return [r[0] for r in rows]

    def _fetch(self, ids: list[str]) -> dict[str, dict]:
        if not ids:
            return {}
        cur = self.conn.execute(
            f"SELECT {RESULT_COLUMNS} FROM chunks WHERE chunk_id = ANY(%s)", [ids]
        )
        names = [d.name for d in cur.description]
        return {row[0]: dict(zip(names, row)) for row in cur.fetchall()}

    def search(
        self, query: str, filters: Filters | None = None, k: int = 5, mode: str = "rerank"
    ) -> list[Result]:
        filters = filters or Filters()
        if mode == "keyword":
            fused = [(cid, 1.0 / (RRF_K + r), ["keyword"]) for r, cid in enumerate(self.keyword(query, filters), 1)]
        elif mode == "vector":
            fused = [(cid, 1.0 / (RRF_K + r), ["vector"]) for r, cid in enumerate(self.vector(query, filters), 1)]
        else:
            fused = reciprocal_rank_fusion(
                {"keyword": self.keyword(query, filters), "vector": self.vector(query, filters)}
            )

        if mode != "rerank":
            top = fused[:k]
            chunks = self._fetch([cid for cid, _, _ in top])
            return [Result(chunks[cid], score, src) for cid, score, src in top]

        pool = fused[:RERANK_POOL]
        chunks = self._fetch([cid for cid, _, _ in pool])
        scores = self.reranker.score(query, [contextual_text(chunks[cid]) for cid, _, _ in pool])
        reranked = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)[:k]
        return [Result(chunks[cid], score, src) for (cid, _, src), score in reranked]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search indexed SEC filing chunks.")
    parser.add_argument("query")
    parser.add_argument("--ticker", nargs="*", default=[], help="e.g. AAPL MSFT")
    parser.add_argument("--year", nargs="*", type=int, default=[], help="fiscal year(s)")
    parser.add_argument("--item", nargs="*", default=[], help="10-K item(s), e.g. 7 1A")
    parser.add_argument("--mode", choices=MODES, default="rerank")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    searcher = Searcher.from_settings()
    filters = Filters(tickers=args.ticker, years=args.year, items=args.item)
    for i, r in enumerate(searcher.search(args.query, filters, k=args.k, mode=args.mode), 1):
        preview = textwrap.shorten(r.chunk["text"].replace("\n", " "), 220)
        print(f"\n[{i}] {r.citation}   score={r.score:.3f}   via={'+'.join(r.found_by)}")
        print(f"    {preview}")


if __name__ == "__main__":
    main()
