"""Cross-encoder reranking — second-stage retrieval.

Bi-encoder retrievers (FAISS dense, BM25) are fast but encode query and document
independently. A cross-encoder reads ``(query, candidate)`` jointly through a
transformer to produce sharper relevance scores at the cost of running once per
candidate at query time.

Standard pattern: over-fetch ``fetch_k`` candidates from any base
:class:`~askistanbul.retriever.Retriever`, then keep the top ``k`` by reranker
score.
"""

from __future__ import annotations

from typing import Optional

from .config import config
from .models import RetrievalResult
from .retriever import Retriever


class Reranker:
    """Cross-encoder model wrapper. Scores ``(query, text)`` pairs."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or config.reranker_model
        print(f"[Reranker] Loading model: {self.model_name}")
        self.model = CrossEncoder(self.model_name, device=device)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance score per text, in input order."""
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return scores.tolist() if hasattr(scores, "tolist") else list(scores)


class RerankingRetriever(Retriever):
    """Two-stage retriever — over-fetch from ``base``, narrow with ``reranker``.

    The base retriever's score and method tag are preserved verbatim on the
    returned :class:`RetrievalResult` objects; the cross-encoder score is
    attached as :attr:`RetrievalResult.cescore`. Callers can detect rerank
    via ``result.reranked`` (i.e. ``result.cescore is not None``).
    """

    def __init__(
        self,
        base: Retriever,
        reranker: Optional[Reranker] = None,
    ):
        self.base = base
        self.reranker = reranker or Reranker()

    def retrieve(self, query: str, biencoder_k: int = 20, reranker_k: int = 5) -> list[RetrievalResult]:
        # Over-fetch biencoder_k candidates from the base retriever, then keep the
        # top reranker_k after cross-encoder scoring. biencoder_k > reranker_k is
        # what makes reranking useful (a larger pool to re-sort and trim).
        candidates = self.base.retrieve(query, biencoder_k=biencoder_k)
        if not candidates:
            return []

        rerank_scores = self.reranker.score(
            query, [c.chunk.text for c in candidates]
        )
        scored = sorted(
            zip(candidates, rerank_scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [
            RetrievalResult(
                chunk=c.chunk,
                score=c.score,            # preserve base retriever score
                method=c.method,          # preserve base retriever tag
                cescore=float(s),         # attach cross-encoder score
            )
            for c, s in scored[:reranker_k]
        ]
