"""Turn a 10-K / 10-Q HTML document into clean text split by "Item" sections.

Filings are messy: inline XBRL tags, hidden blocks, layout tables, a table of
contents that repeats every heading, and financial tables where "$", the
number and ")" sit in separate cells. This module handles those cases and
keeps real tables as pipe-delimited rows so numbers stay next to their labels.
"""

import re
import warnings
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

# Inline XBRL filings are XHTML; parsing them as HTML is intended.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

TABLE_START = "[TABLE]"
TABLE_END = "[/TABLE]"

BLOCK_TAGS = ["p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"]

TEN_K_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "[Reserved]",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

ITEM_RE = re.compile(r"^item\s*(\d{1,2}[a-c]?)\s*[.:\-–—]?\s*(.*)$", re.IGNORECASE)
PART_RE = re.compile(r"^part\s+(i{1,3}|iv)\b", re.IGNORECASE)
MAX_HEADING_LEN = 150


@dataclass
class Section:
    item: str  # e.g. "1A" for a 10-K, "II-1A" for a 10-Q
    title: str
    text: str


def html_to_text(html: str | bytes) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "head", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.IGNORECASE)):
        tag.decompose()

    for table in soup.find_all("table"):
        table.replace_with(NavigableString(_table_to_text(table)))

    for tag in soup.find_all(BLOCK_TAGS):
        tag.append(NavigableString("\n"))

    text = soup.get_text().replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _table_to_text(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True).replace("\xa0", " ") for c in tr.find_all(["td", "th"])]
        cells = _merge_cells([c for c in cells if c])
        if cells:
            rows.append(cells)

    if not rows:
        return "\n"
    # Layout tables (one cell per row) are just text in a box; don't mark them as tables.
    if all(len(r) == 1 for r in rows):
        return "\n" + "\n".join(r[0] for r in rows) + "\n"

    body = "\n".join(" | ".join(r) for r in rows)
    return f"\n\n{TABLE_START}\n{body}\n{TABLE_END}\n\n"


def _merge_cells(cells: list[str]) -> list[str]:
    """Join currency symbols and closing parens / percent signs onto their numbers."""
    merged: list[str] = []
    for cell in cells:
        opens_number = merged and merged[-1] in {"$", "(", "($", "$("}
        closes_number = merged and cell in {")", "%", ")%", "%)"}
        if opens_number or closes_number:
            merged[-1] += cell
        else:
            merged.append(cell)
    return merged


def split_sections(text: str, form: str = "10-K") -> list[Section]:
    """Split filing text into Item sections.

    Every heading appears twice (table of contents, then the real section), so
    for each item we keep the longest candidate span, which is the real one.
    """
    lines = text.split("\n")
    headings: list[tuple[int, str, str]] = []  # (line index, item key, title)
    part = ""
    in_table = False

    for i, line in enumerate(lines):
        if line == TABLE_START:
            in_table = True
        elif line == TABLE_END:
            in_table = False
        if in_table or not line or len(line) > MAX_HEADING_LEN:
            continue

        if m := PART_RE.match(line):
            part = m.group(1).upper()
            continue
        m = ITEM_RE.match(line)
        if not m:
            continue
        item, title = m.group(1).upper(), m.group(2).strip()
        if title.lower().startswith(("of ", "and ", "in ", "to ")):
            continue  # a cross-reference like "Item 7 of this report", not a heading
        if not title:
            title = next((ln for ln in lines[i + 1 : i + 4] if ln), "")
        title = re.sub(r"\s*\d+$", "", title)  # drop TOC page numbers
        key = f"{part}-{item}" if form.startswith("10-Q") and part else item
        headings.append((i, key, title))

    # Many filings repeat "PART II / Item 7" as a running header on every page.
    # Collapse consecutive headings for the same item so the section stays whole.
    headings = [h for j, h in enumerate(headings) if j == 0 or h[1] != headings[j - 1][1]]

    best: dict[str, Section] = {}
    for idx, (start, key, title) in enumerate(headings):
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        body = "\n".join(lines[start + 1 : end]).strip()
        if key not in best or len(body) > len(best[key].text):
            if form.startswith("10-K"):
                # Heading text varies wildly by filer; standard 10-K titles keep metadata clean.
                title = TEN_K_TITLES.get(key, title)
            best[key] = Section(item=key, title=title, text=body)

    return sorted(best.values(), key=lambda s: _section_order(s.item))


def _section_order(key: str) -> tuple:
    part, _, item = key.rpartition("-")
    m = re.match(r"(\d+)([A-C]?)", item)
    return (part, int(m.group(1)), m.group(2)) if m else (part, 999, item)
