"""Local embedding and reranking models (ONNX via fastembed, CPU only, no API key)."""

from functools import cached_property

import numpy as np

# bge models are trained with this instruction on the query side only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def contextual_text(chunk: dict) -> str:
    """Text that gets embedded and reranked: the chunk plus where it came from.

    A chunk that just says "Net sales increased 6%" is ambiguous on its own;
    prefixing company, year and section lets search tell Apple FY2025 MD&A
    apart from Microsoft FY2024 MD&A.
    """
    header = (
        f"{chunk['company']} ({chunk['ticker']}) {chunk['form']} fiscal year "
        f"{chunk['fiscal_year']}, Item {chunk['item']}: {chunk['section_title']}"
    )
    return f"{header}\n\n{chunk['text']}"


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name

    @cached_property
    def _model(self):
        from fastembed import TextEmbedding

        return TextEmbedding(self.model_name)

    @property
    def dim(self) -> int:
        return len(self.embed_query("dimension probe"))

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> list[np.ndarray]:
        return list(self._model.embed(texts, batch_size=batch_size))

    def embed_query(self, query: str) -> np.ndarray:
        prefix = BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        return next(iter(self._model.embed([prefix + query])))


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name

    @cached_property
    def _model(self):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return TextCrossEncoder(self.model_name)

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, documents)]
