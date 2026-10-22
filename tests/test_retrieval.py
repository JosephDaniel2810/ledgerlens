from ledgerlens.retrieval.embed import contextual_text
from ledgerlens.retrieval.evaluate import is_relevant
from ledgerlens.retrieval.index import content_hash
from ledgerlens.retrieval.search import Filters, reciprocal_rank_fusion

CHUNK = {
    "chunk_id": "abc", "ticker": "AAPL", "company": "Apple Inc.", "form": "10-K",
    "fiscal_year": 2025, "item": "7", "section_title": "Management's Discussion and Analysis",
    "text": "Research and development expense increased 10%.",
}


def test_rrf_rewards_agreement_between_retrievers():
    fused = reciprocal_rank_fusion({"keyword": ["a", "b", "c"], "vector": ["c", "a", "d"]})
    ids = [cid for cid, _, _ in fused]
    assert ids[:2] == ["a", "c"]  # found by both beat those found by one
    assert set(ids) == {"a", "b", "c", "d"}
    sources = {cid: src for cid, _, src in fused}
    assert sources["a"] == ["keyword", "vector"] and sources["d"] == ["vector"]


def test_rrf_score_formula():
    fused = reciprocal_rank_fusion({"only": ["x", "y"]}, k=60)
    assert fused[0][1] == 1 / 61 and fused[1][1] == 1 / 62


def test_filters_build_parameterized_sql():
    where, params = Filters(tickers=["aapl"], years=[2025], items=["1a"]).sql()
    assert where == "ticker = ANY(%s) AND fiscal_year = ANY(%s) AND item = ANY(%s)"
    assert params == [["AAPL"], [2025], ["1A"]]
    assert Filters().sql() == ("TRUE", [])


def test_contextual_text_names_company_year_and_section():
    text = contextual_text(CHUNK)
    assert text.startswith("Apple Inc. (AAPL) 10-K fiscal year 2025, Item 7:")
    assert text.endswith(CHUNK["text"])


def test_content_hash_changes_with_text_and_model():
    base = content_hash(CHUNK, "model-a")
    assert base == content_hash(dict(CHUNK), "model-a")
    assert base != content_hash(CHUNK | {"text": "changed"}, "model-a")
    assert base != content_hash(CHUNK, "model-b")


def test_relevance_needs_company_section_and_term():
    case = {"ticker": "AAPL", "items": ["7", "8"], "contains": ["research and development"]}
    assert is_relevant(CHUNK, case)
    assert not is_relevant(CHUNK | {"ticker": "MSFT"}, case)
    assert not is_relevant(CHUNK | {"item": "1A"}, case)
    assert not is_relevant(CHUNK | {"text": "Net sales grew."}, case)
