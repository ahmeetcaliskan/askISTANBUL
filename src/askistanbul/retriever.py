"""Retrieval — dense (FAISS) and sparse (BM25), behind a shared ABC."""

from __future__ import annotations

import json
import re
import string
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

from .embedder import Embedder
from .models import Chunk, RetrievalResult
from .paths import CHUNKS_FILE, INDEX_DIR


# ---------------------------------------------------------------------------
# BM25 tokenizer — lowercases, strips punctuation, drops stopwords.
# Travel-question intent words (how, what, near, where) are deliberately kept.
# ---------------------------------------------------------------------------

_BM25_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "had", "he", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "this", "these", "those", "to", "was", "were", "will",
    "with", "i", "you", "your", "we", "our", "they", "them", "their",
    "do", "does", "did", "been", "being", "than", "then", "so", "if",
    "about", "into", "over", "also", "more",
}
_BM25_PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}]")


def bm25_tokenize(text: str) -> list[str]:
    text = _BM25_PUNCT_RE.sub(" ", text.lower())
    return [t for t in text.split() if t and t not in _BM25_STOPWORDS]


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class Retriever(ABC):
    """Common interface for dense and sparse retrievers."""

    @abstractmethod
    def retrieve(self, query: str, biencoder_k: int = 20, reranker_k: int = 5) -> list[RetrievalResult]: ...


# ---------------------------------------------------------------------------
# Dense retriever (FAISS + sentence-transformers)
# ---------------------------------------------------------------------------

class DenseRetriever(Retriever):
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        index_dir: Path = INDEX_DIR,
        fetch_k: Optional[int] = 20,
    ):
        self.index, self.chunks, config = Embedder.load_index(index_dir)
        index_model = config["model"]
        self.embedder = embedder if embedder is not None else Embedder(model_name=index_model)
        self.fetch_k = fetch_k
        if self.embedder.model_name != index_model:
            print(
                f"[DenseRetriever] WARNING: index was built with {index_model!r} "
                f"but embedder uses {self.embedder.model_name!r}"
            )
        print(f"[DenseRetriever] Loaded {len(self.chunks)} chunks, model={index_model}")

    def retrieve(self, query: str, biencoder_k: int = 20, reranker_k: int = 5) -> list[RetrievalResult]:
        vec = self.embedder.encode([query], is_query=True)
        scores, indices = self.index.search(vec, biencoder_k)
        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                score=float(score),
                method="dense",
            ))
        return results


# ---------------------------------------------------------------------------
# Sparse retriever (BM25)
# ---------------------------------------------------------------------------

class BM25Retriever(Retriever):
    def __init__(
        self,
        chunks_file: Path = CHUNKS_FILE,
        tokenize_fn: Callable[[str], list[str]] = bm25_tokenize,
    ):
        from rank_bm25 import BM25Okapi

        self.chunks: list[Chunk] = []
        with Path(chunks_file).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.chunks.append(Chunk.from_dict(json.loads(line)))

        self.tokenize_fn = tokenize_fn
        tokenized = [tokenize_fn(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"[BM25Retriever] Indexed {len(self.chunks)} chunks")

    def retrieve(self, query: str, biencoder_k: int = 20, reranker_k: int = 5) -> list[RetrievalResult]:
        # reranker_k is accepted for interface uniformity but unused (no rerank stage).
        tokens = self.tokenize_fn(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:biencoder_k]
        return [
            RetrievalResult(chunk=self.chunks[i], score=float(scores[i]), method="bm25")
            for i in top_indices
        ]


# ---------------------------------------------------------------------------
# Smoke-test CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    from .config import config as _cfg

    parser = argparse.ArgumentParser(description="Query the Istanbul RAG index.")
    parser.add_argument("query", nargs="?", default="best rooftop bars in Beyoglu",
                        help="Natural-language question.")
    parser.add_argument("--biencoder-k", type=int, default=_cfg.biencoder_fetch_k,
                        help="Bi-encoder candidates / result count (default from BIENCODER_FETCH_K).")
    parser.add_argument("--reranker-k", type=int, default=_cfg.reranker_fetch_k,
                        help="Final results after reranking, used with --rerank (default from RERANKER_FETCH_K).")
    parser.add_argument("--method", choices=["dense", "bm25", "both"], default="dense",
                        help="Retrieval backend (default: dense).")
    parser.add_argument("--rerank", action="store_true",
                        help="Apply cross-encoder reranking on top of the base retriever.")
    parser.add_argument("--show-text", action="store_true",
                        help="Print the chunk text alongside scores.")
    args = parser.parse_args()

    def wrap(base: Retriever) -> Retriever:
        if not args.rerank:
            return base
        from .reranker import RerankingRetriever
        return RerankingRetriever(base=base)

    def go(base: Retriever) -> list[RetrievalResult]:
        return wrap(base).retrieve(args.query, biencoder_k=args.biencoder_k,
                                   reranker_k=args.reranker_k)

    def show(label: str, results: list[RetrievalResult]) -> None:
        print(f"\n{'='*60}\n{label}  —  query: {args.query!r}\n{'='*60}")
        for i, r in enumerate(results, 1):
            score_str = f"score={r.score:.4f}"
            if r.cescore is not None:
                score_str += f"  ce={r.cescore:.4f}"
            print(f"\n[{i}] {score_str}  |  {r.chunk.title} / {r.chunk.heading}  ({r.method})")
            if args.show_text:
                print(r.chunk.text[:300].replace("\n", " "))

    if args.method in ("dense", "both"):
        show("DENSE (FAISS)", go(DenseRetriever()))
    if args.method in ("bm25", "both"):
        show("SPARSE (BM25)", go(BM25Retriever()))


if __name__ == "__main__":
    _main()
