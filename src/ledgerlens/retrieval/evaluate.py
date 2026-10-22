"""Compare search modes on a small labeled query set.

A result counts as relevant when it comes from the expected company and
section and contains one of the expected terms. This is a smoke test for
retrieval only; the full answer-quality evaluation comes in phase 6.

Example:
    ledgerlens-eval-retrieval evals/retrieval_smoke.jsonl
"""

import argparse
import json
import time
from pathlib import Path

from ledgerlens.retrieval.search import MODES, Searcher

K = 5


def is_relevant(chunk: dict, case: dict) -> bool:
    text = chunk["text"].lower()
    return (
        chunk["ticker"] == case["ticker"]
        and chunk["item"] in case["items"]
        and any(term in text for term in case["contains"])
    )


def evaluate(searcher: Searcher, cases: list[dict], mode: str) -> dict:
    hits1 = hits5 = rr = 0.0
    latencies, misses = [], []
    for case in cases:
        started = time.perf_counter()
        results = searcher.search(case["query"], k=K, mode=mode)
        latencies.append(time.perf_counter() - started)
        ranks = [i for i, r in enumerate(results, 1) if is_relevant(r.chunk, case)]
        if ranks:
            hits1 += ranks[0] == 1
            hits5 += 1
            rr += 1 / ranks[0]
        else:
            misses.append(case["id"])
    n = len(cases)
    latencies.sort()
    return {
        "mode": mode,
        "hit@1": hits1 / n,
        "hit@5": hits5 / n,
        "mrr@5": rr / n,
        "p50_ms": 1000 * latencies[n // 2],
        "misses": misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval modes.")
    parser.add_argument("cases", type=Path)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args()

    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    searcher = Searcher.from_settings()
    searcher.search("warm up models", k=1, mode="rerank")

    print(f"{len(cases)} queries, no filters, top {K}\n")
    print(f"{'mode':10} {'hit@1':>6} {'hit@5':>6} {'mrr@5':>6} {'p50 ms':>7}  misses")
    for mode in args.modes:
        m = evaluate(searcher, cases, mode)
        print(
            f"{m['mode']:10} {m['hit@1']:6.0%} {m['hit@5']:6.0%} {m['mrr@5']:6.2f} "
            f"{m['p50_ms']:7.0f}  {', '.join(m['misses']) or '-'}"
        )


if __name__ == "__main__":
    main()
