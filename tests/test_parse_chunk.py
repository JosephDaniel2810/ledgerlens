from ledgerlens.ingest.chunk import approx_tokens, chunk_section, split_blocks
from ledgerlens.ingest.parse import TABLE_START, Section, html_to_text, split_sections

FILING_HTML = """
<html><body>
<ix:header><div>hidden xbrl metadata</div></ix:header>
<div style="display:none">hidden</div>
<p>PART I</p>
<table><tr><td>Item 1.</td><td>Business</td><td>3</td></tr>
       <tr><td>Item 1A.</td><td>Risk Factors</td><td>9</td></tr>
       <tr><td>Item 7.</td><td>MD&amp;A</td><td>20</td></tr></table>
<p>Item 1. Business</p>
<p>We design phones and services. As described in Item 7 of this report, sales grew.</p>
<p>Item 1A. Risk Factors</p>
<p>Supply chain disruption could hurt results.</p>
<p>PART II</p>
<p>Item 7. Management's Discussion and Analysis</p>
<p>Revenue increased year over year.</p>
<table>
  <tr><td></td><td>2024</td><td>2023</td></tr>
  <tr><td>Net sales</td><td>$</td><td>391,035</td><td>$</td><td>383,285</td></tr>
  <tr><td>Change</td><td>2</td><td>%</td><td>(3</td><td>)%</td></tr>
</table>
</body></html>
"""

META = {
    "ticker": "TEST", "company": "Test Inc", "form": "10-K", "fiscal_year": 2024,
    "filing_date": "2024-11-01", "accession": "0000-24-000001", "url": "https://example.com",
}


def test_hidden_content_removed():
    text = html_to_text(FILING_HTML)
    assert "hidden" not in text


def test_financial_table_cells_merged():
    text = html_to_text(FILING_HTML)
    assert "Net sales | $391,035 | $383,285" in text
    assert "Change | 2% | (3)%" in text


def test_sections_skip_table_of_contents_and_cross_references():
    sections = split_sections(html_to_text(FILING_HTML))
    by_item = {s.item: s for s in sections}
    assert list(by_item) == ["1", "1A", "7"]
    assert by_item["1"].title == "Business"
    assert "design phones" in by_item["1"].text
    assert "Supply chain" in by_item["1A"].text
    assert TABLE_START in by_item["7"].text


def test_running_page_headers_do_not_split_a_section():
    html = (
        "<p>Item 7. MD&amp;A</p><p>Page one text.</p>"
        "<p>PART II</p><p>Item 7</p><p>Page two text.</p>"
        "<p>PART II</p><p>Item 7</p><p>Page three text.</p>"
        "<p>Item 8. Financial Statements</p><p>Statements.</p>"
    )
    section = split_sections(html_to_text(html))[0]
    assert section.item == "7" and section.title == "Management's Discussion and Analysis"
    assert all(f"Page {n} text." in section.text for n in ("one", "two", "three"))


def test_ten_q_items_are_keyed_by_part():
    html = "<p>PART I</p><p>Item 1. Financial Statements</p><p>x</p>" \
           "<p>PART II</p><p>Item 1. Legal Proceedings</p><p>y</p>"
    keys = [s.item for s in split_sections(html_to_text(html), form="10-Q")]
    assert keys == ["I-1", "II-1"]


def test_tables_become_their_own_chunks():
    section = next(s for s in split_sections(html_to_text(FILING_HTML)) if s.item == "7")
    chunks = chunk_section(section, META)
    assert [c.is_table for c in chunks] == [False, True]
    assert chunks[1].text.startswith("2024 | 2023")
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_long_prose_is_packed_under_limit():
    para = "Revenue grew because demand was strong. " * 40
    section = Section(item="7", title="MD&A", text="\n\n".join([para] * 10))
    chunks = chunk_section(section, META)
    assert len(chunks) > 1
    assert all(approx_tokens(c.text) <= 800 for c in chunks)


def test_oversized_table_split_repeats_header():
    rows = "\n".join(f"Line item {i} | {i * 1000:,} | {i * 900:,}" for i in range(400))
    text = f"{TABLE_START}\nMetric | 2024 | 2023\n{rows}\n[/TABLE]"
    chunks = chunk_section(Section(item="8", title="Financials", text=text), META)
    assert len(chunks) > 1
    assert all(c.is_table and c.text.startswith("Metric | 2024 | 2023") for c in chunks)
    assert split_blocks(text)[0][1] is True
