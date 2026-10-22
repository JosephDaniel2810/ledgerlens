"""Command line entry point for the ingestion pipeline.

Example:
    ledgerlens-ingest --tickers AAPL MSFT NVDA --forms 10-K --limit 3
"""

import argparse
import json
import logging
from collections import Counter

from ledgerlens.config import get_settings
from ledgerlens.ingest.chunk import chunk_section
from ledgerlens.ingest.edgar import EdgarClient
from ledgerlens.ingest.parse import html_to_text, split_sections

log = logging.getLogger("ledgerlens.ingest")

# Sections required for a 10-K to be considered parsed correctly.
REQUIRED_10K_ITEMS = {"1", "1A", "7", "8"}


def run(tickers: list[str], forms: tuple[str, ...], limit: int) -> None:
    settings = get_settings()
    client = EdgarClient(settings.sec_user_agent, settings.raw_dir)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.processed_dir / "chunks.jsonl"

    # Rewrite chunks for the filings we process, keep everything else.
    existing = []
    if out_path.exists():
        existing = [json.loads(line) for line in out_path.read_text().splitlines() if line]

    new_chunks, processed, warnings = [], set(), []
    for ticker in tickers:
        for filing in client.list_filings(ticker, forms=forms, limit=limit):
            doc_path = client.download(filing)
            sections = split_sections(html_to_text(doc_path.read_bytes()), form=filing.form)

            found = {s.item for s in sections}
            if filing.form == "10-K" and (missing := REQUIRED_10K_ITEMS - found):
                warnings.append(f"{ticker} FY{filing.fiscal_year}: missing items {sorted(missing)}")

            meta = filing.to_dict()
            chunks = [c for s in sections for c in chunk_section(s, meta)]
            new_chunks.extend(c.to_dict() for c in chunks)
            processed.add(filing.accession)
            log.info(
                "%s %s FY%s: %d sections, %d chunks",
                ticker, filing.form, filing.fiscal_year, len(sections), len(chunks),
            )

    kept = [c for c in existing if c["accession"] not in processed]
    with out_path.open("w") as f:
        for c in kept + new_chunks:
            f.write(json.dumps(c) + "\n")

    _report(kept + new_chunks, out_path, warnings)


def _report(chunks: list[dict], out_path, warnings: list[str]) -> None:
    by_filing = Counter((c["ticker"], c["form"], c["fiscal_year"]) for c in chunks)
    tables = sum(c["is_table"] for c in chunks)
    tokens = [c["n_tokens"] for c in chunks]
    print(f"\nWrote {len(chunks)} chunks from {len(by_filing)} filings to {out_path}")
    print(f"  tables: {tables}   avg tokens/chunk: {sum(tokens) // max(1, len(tokens))}")
    for (ticker, form, year), n in sorted(by_filing.items()):
        print(f"  {ticker:6} {form:5} FY{year}  {n:5} chunks")
    for w in warnings:
        print(f"  WARNING {w}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download, parse and chunk SEC filings.")
    parser.add_argument("--tickers", nargs="+", required=True, help="e.g. AAPL MSFT NVDA")
    parser.add_argument("--forms", nargs="+", default=["10-K"], help="10-K and/or 10-Q")
    parser.add_argument("--limit", type=int, default=3, help="filings per ticker per run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(args.tickers, tuple(args.forms), args.limit)


if __name__ == "__main__":
    main()
