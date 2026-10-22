"""Section-aware chunking.

Chunks never cross an Item boundary, tables are kept whole when they fit, and
oversized tables are split by rows with the header row repeated, so a number
is never separated from the column it belongs to.
"""

import hashlib
import re
from dataclasses import asdict, dataclass

from ledgerlens.ingest.parse import TABLE_END, TABLE_START, Section

TARGET_TOKENS = 400
MAX_TOKENS = 800


def approx_tokens(text: str) -> int:
    # ~4 characters per token for English prose; good enough for sizing chunks.
    return max(1, len(text) // 4)


@dataclass
class Chunk:
    chunk_id: str
    ticker: str
    company: str
    form: str
    fiscal_year: int
    filing_date: str
    accession: str
    item: str
    section_title: str
    chunk_index: int
    is_table: bool
    n_tokens: int
    text: str
    source_url: str

    def to_dict(self) -> dict:
        return asdict(self)


def split_blocks(text: str) -> list[tuple[str, bool]]:
    """Split section text into (block, is_table) pieces."""
    pattern = re.compile(rf"{re.escape(TABLE_START)}\n(.*?)\n{re.escape(TABLE_END)}", re.DOTALL)
    blocks: list[tuple[str, bool]] = []
    pos = 0
    for m in pattern.finditer(text):
        blocks += [(p, False) for p in _paragraphs(text[pos : m.start()])]
        blocks.append((m.group(1).strip(), True))
        pos = m.end()
    blocks += [(p, False) for p in _paragraphs(text[pos:])]
    return blocks


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _split_large(block: str, is_table: bool) -> list[str]:
    if approx_tokens(block) <= MAX_TOKENS:
        return [block]
    if is_table:
        header, *rows = block.split("\n")
        parts, current = [], [header]
        for row in rows:
            if approx_tokens("\n".join(current + [row])) > MAX_TOKENS and len(current) > 1:
                parts.append("\n".join(current))
                current = [header]
            current.append(row)
        parts.append("\n".join(current))
        return parts
    sentences = re.split(r"(?<=[.!?])\s+", block)
    parts, current = [], ""
    for s in sentences:
        if current and approx_tokens(current + " " + s) > TARGET_TOKENS:
            parts.append(current)
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        parts.append(current)
    return parts


def chunk_section(section: Section, meta: dict) -> list[Chunk]:
    """Pack a section's paragraphs into ~TARGET_TOKENS chunks; tables stand alone."""
    pieces: list[tuple[str, bool]] = []
    buffer: list[str] = []

    def flush():
        if buffer:
            pieces.append(("\n\n".join(buffer), False))
            buffer.clear()

    for block, is_table in split_blocks(section.text):
        for part in _split_large(block, is_table):
            if is_table:
                flush()
                pieces.append((part, True))
            elif buffer and approx_tokens("\n\n".join(buffer + [part])) > TARGET_TOKENS:
                flush()
                buffer.append(part)
            else:
                buffer.append(part)
    flush()

    chunks = []
    for i, (text, is_table) in enumerate(pieces):
        digest = hashlib.sha1(f"{meta['accession']}|{section.item}|{i}".encode()).hexdigest()[:16]
        chunks.append(
            Chunk(
                chunk_id=digest,
                ticker=meta["ticker"],
                company=meta["company"],
                form=meta["form"],
                fiscal_year=meta["fiscal_year"],
                filing_date=meta["filing_date"],
                accession=meta["accession"],
                item=section.item,
                section_title=section.title,
                chunk_index=i,
                is_table=is_table,
                n_tokens=approx_tokens(text),
                text=text,
                source_url=meta["url"],
            )
        )
    return chunks
