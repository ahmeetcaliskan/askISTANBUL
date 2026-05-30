"""
retriever.py
------------
Unified retrieval interface — dense (FAISS) and sparse (BM25).

Both backends expose the same API:
    retriever.retrieve(query: str, k: int = 5) -> list[dict]

Each returned dict is a chunk with an added "score" field.

Usage:
    from retriever import DenseRetriever, BM25Retriever

    dense = DenseRetriever()
    results = dense.retrieve("best rooftop bars in Beyoglu", k=5)
    for r in results:
        print(r["score"], r["title"], r["heading"])
        print(r["text"][:200])
        print()
"""

import json
from pathlib import Path

import numpy as np

INDEX_DIR   = Path(__file__).parent.parent / "data" / "index"
CHUNKS_FILE = Path(__file__).parent.parent / "data" / "chunks" / "all_chunks.jsonl"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Dense Retriever (FAISS + sentence-transformers)
# ---------------------------------------------------------------------------

class DenseRetriever:
    def __init__(self, index_dir: Path = INDEX_DIR):
        from embedder import load_index
        from sentence_transformers import SentenceTransformer

        self.index, self.chunks, config = load_index(index_dir)
        self.model_name = config["model"]
        print(f"[DenseRetriever] Loaded {len(self.chunks)} chunks, model={self.model_name}")
        self.model = SentenceTransformer(self.model_name)

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        # intfloat/multilingual-e5-* models need "query: " prefix at query time
        is_e5 = "e5" in self.model_name.lower()
        query_input = f"query: {query}" if is_e5 else query

        vec = self.model.encode(
            [query_input],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, indices = self.index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            chunk["retrieval_method"] = "dense"
            results.append(chunk)
        return results


# ---------------------------------------------------------------------------
# BM25 Retriever (sparse baseline)
# ---------------------------------------------------------------------------

class BM25Retriever:
    def __init__(self, chunks_file: Path = CHUNKS_FILE):
        from rank_bm25 import BM25Okapi

        self.chunks = []
        with chunks_file.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))

        tokenized = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"[BM25Retriever] Indexed {len(self.chunks)} chunks")

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:k]

        results = []
        for idx in top_indices:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            chunk["retrieval_method"] = "bm25"
            results.append(chunk)
        return results


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="best rooftop bars in Beyoglu")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--method", choices=["dense", "bm25", "both"], default="both")
    args = parser.parse_args()

    def print_results(label: str, results: list[dict]) -> None:
        print(f"\n{'='*60}")
        print(f"{label}  —  query: {args.query!r}")
        print("="*60)
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] score={r['score']:.4f}  |  {r['title']} / {r['heading']}")
            print(r["text"][:300].replace("\n", " "))

    if args.method in ("dense", "both"):
        dense = DenseRetriever()
        print_results("DENSE (FAISS)", dense.retrieve(args.query, args.k))

    if args.method in ("bm25", "both"):
        bm25 = BM25Retriever()
        print_results("SPARSE (BM25)", bm25.retrieve(args.query, args.k))
