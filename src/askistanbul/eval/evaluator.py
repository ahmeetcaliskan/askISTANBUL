"""
eval/evaluator.py
-----------------
Full evaluation harness for AskIstanbul.

Runs the RAG pipeline against the labeled QA set and reports:
  - Precision@5       (retrieval quality)
  - Recall@5          (retrieval completeness)
  - MRR               (rank of first relevant chunk)
  - NDCG@5            (graded ranking quality of relevant chunks)
  - Faithfulness      (generation groundedness, LLM judge)
  - Answer Relevance  (embedding cosine sim, question vs answer)

Compares three conditions:
  - dense-rag     DenseRetriever  + LLM
  - bm25-rag      BM25Retriever   + LLM
  - no-rag        No retrieval    + LLM  (baseline: answer from LLM alone)

Usage:
  python -m askistanbul.eval.evaluator                   # dense-rag only, first 10 questions
  python -m askistanbul.eval.evaluator --conditions all  # all three conditions
  python -m askistanbul.eval.evaluator --n 50 --conditions dense-rag,no-rag
  python -m askistanbul.eval.evaluator --conditions dense-rag --out results/run1.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import config
from ..generator.factory.LLMClientFactory import LLMClientFactory
from ..paths import EVAL_DIR, QA_DRAFT_FILE, PROJECT_ROOT
from ..rag import RAGPipeline
from ..retriever import BM25Retriever, DenseRetriever
from .metrics import (
    aggregate,
    answer_relevance,
    faithfulness,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

RESULTS_DIR = PROJECT_ROOT / "results"
QA_FILE     = EVAL_DIR / "qa_draft_with_relevant_chunks.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_qa(path: Path, n: Optional[int] = None) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items[:n] if n else items


def make_embed_fn(model_name: str):
    """Return a callable embed_fn(texts) → np.ndarray for answer_relevance."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    is_e5 = "e5" in model_name.lower()

    def embed_fn(texts: list[str]) -> np.ndarray:
        prefixed = [f"query: {t}" if is_e5 else t for t in texts]
        return model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    return embed_fn


# ---------------------------------------------------------------------------
# Single-condition evaluation
# ---------------------------------------------------------------------------

def run_condition(
    condition: str,
    qa_items: list[dict],
    llm_client,
    embed_fn,
    k: int = 5,
    verbose: bool = True,
) -> dict:
    """Evaluate one retrieval condition over qa_items.

    Args:
        condition: "dense-rag" | "bm25-rag" | "rerank-rag" | "no-rag"
    Returns:
        dict with per-question rows + aggregate metrics.
    """
    # Build retriever / pipeline
    if condition == "dense-rag":
        retriever = DenseRetriever()
        pipeline  = RAGPipeline(retriever=retriever, generator=llm_client)
    elif condition == "bm25-rag":
        retriever = BM25Retriever()
        pipeline  = RAGPipeline(retriever=retriever, generator=llm_client)
    elif condition == "rerank-rag":
        # Two-stage: dense over-fetch → cross-encoder rerank → top-k
        from ..reranker import Reranker, RerankingRetriever
        base      = DenseRetriever()
        reranker  = Reranker()
        retriever = RerankingRetriever(base=base, reranker=reranker)
        pipeline  = RAGPipeline(retriever=retriever, generator=llm_client)
    elif condition == "no-rag":
        pipeline  = _NoRAGPipeline(llm_client)
        retriever = None
    else:
        raise ValueError(f"Unknown condition: {condition!r}")

    # Per-condition k wiring (fair comparison — every condition feeds the LLM `k`):
    #   dense/bm25 return `k` directly;
    #   rerank over-fetches `biencoder_fetch_k` candidates, then trims to `k`.
    if condition == "rerank-rag":
        bk, rk = config.biencoder_fetch_k, k
    else:
        bk, rk = k, k

    rows: list[dict] = []
    p_at_k_scores: list[float] = []
    recall_scores: list[float] = []
    mrr_scores:    list[float] = []
    ndcg_scores:   list[float] = []
    faith_scores:  list[float] = []
    rel_scores:    list[float] = []

    for i, item in enumerate(qa_items, 1):
        qid      = item["id"]
        question = item["question"]
        gold_ids = item.get("relevant_chunk_ids") or []

        if verbose:
            print(f"  [{i:3d}/{len(qa_items)}] Q{qid}: {question[:70]}...")

        # Retry up to 2 times on transient LLM errors (rate limits, empty responses)
        t0 = time.time()
        ans = None
        last_err = None
        for attempt in range(3):
            try:
                ans = pipeline.answer(question, fetch_k_biencoder=bk, fetch_k_reranker=rk)
                break
            except Exception as exc:
                last_err = exc
                wait = 5 * (attempt + 1)
                print(f"         [retry {attempt+1}/3] {exc} — waiting {wait}s...")
                time.sleep(wait)

        elapsed = time.time() - t0

        if ans is None:
            print(f"         [SKIP] Q{qid} failed after 3 attempts: {last_err}")
            rows.append({"id": qid, "question": question, "error": str(last_err)})
            continue

        retrieved_ids = [r.chunk.chunk_id for r in ans.results] if ans.results else []
        context_texts = [r.chunk.text for r in ans.results]     if ans.results else []
        answer_text   = ans.answer or ""

        # --- Retrieval metrics --------------------------------------------
        pk  = precision_at_k(retrieved_ids, gold_ids, k=k) if gold_ids else None
        rk  = recall_at_k(retrieved_ids, gold_ids, k=k)    if gold_ids else None
        rr  = mrr(retrieved_ids, gold_ids)                  if gold_ids else None
        nd  = ndcg_at_k(retrieved_ids, gold_ids, k=k)       if gold_ids else None

        # --- Faithfulness (LLM judge) ------------------------------------
        if answer_text and context_texts:
            faith, claims = faithfulness(answer_text, context_texts, llm_client)
        else:
            faith, claims = (None, [])

        # --- Answer relevance (embedding cosine) -------------------------
        if answer_text:
            ar = answer_relevance(question, answer_text, embed_fn)
        else:
            ar = None

        row = {
            "id":               qid,
            "category":         item.get("category"),
            "difficulty":       item.get("difficulty"),
            "question":         question,
            "ground_truth":     item.get("ground_truth"),
            "answer":           answer_text,
            "retrieved_ids":    retrieved_ids,
            "gold_ids":         gold_ids,
            "precision_at_k":   pk,
            "recall_at_k":      rk,
            "mrr":              rr,
            "ndcg_at_k":        nd,
            "faithfulness":     faith,
            "answer_relevance": ar,
            "latency_s":        round(elapsed, 2),
            "faithfulness_claims": claims,
        }
        rows.append(row)

        # Accumulate for aggregate (skip None — items without gold_ids)
        if pk    is not None: p_at_k_scores.append(pk)
        if rk    is not None: recall_scores.append(rk)
        if rr    is not None: mrr_scores.append(rr)
        if nd    is not None: ndcg_scores.append(nd)
        if faith is not None: faith_scores.append(faith)
        if ar    is not None: rel_scores.append(ar)

        if verbose:
            pk_str    = f"{pk:.3f}"    if pk    is not None else "n/a"
            rk_str    = f"{rk:.3f}"    if rk    is not None else "n/a"
            rr_str    = f"{rr:.3f}"    if rr    is not None else "n/a"
            nd_str    = f"{nd:.3f}"    if nd    is not None else "n/a"
            faith_str = f"{faith:.3f}" if faith is not None else "n/a"
            ar_str    = f"{ar:.3f}"    if ar    is not None else "n/a"
            print(f"         P@{k}={pk_str}  R@{k}={rk_str}  MRR={rr_str}  NDCG@{k}={nd_str}  "
                  f"faith={faith_str}  rel={ar_str}  ({elapsed:.1f}s)")

    return {
        "condition": condition,
        "k": k,
        "n_questions": len(rows),
        "aggregate": {
            "precision_at_k":   aggregate(p_at_k_scores),
            "recall_at_k":      aggregate(recall_scores),
            "mrr":              aggregate(mrr_scores),
            "ndcg_at_k":        aggregate(ndcg_scores),
            "faithfulness":     aggregate(faith_scores),
            "answer_relevance": aggregate(rel_scores),
        },
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# No-RAG baseline pipeline
# ---------------------------------------------------------------------------

class _NoRAGPipeline:
    """Answer without any retrieval — pure LLM baseline."""

    _SYSTEM = (
        "You are a helpful travel assistant for Istanbul. "
        "Answer the question as accurately as you can from your own knowledge. "
        "Be concise."
    )

    def __init__(self, llm_client):
        self.generator = llm_client

    def answer(self, question: str, fetch_k_biencoder: int = 5, fetch_k_reranker: int = 5):
        from ..rag import Answer
        messages = [
            {"role": "system", "content": self._SYSTEM},
            {"role": "user",   "content": question},
        ]
        text = self.generator.chat(messages)
        return Answer(question=question, results=[], answer=text, citations=[])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AskIstanbul evaluation harness",
                                     allow_abbrev=False)
    parser.add_argument(
        "--conditions", default="dense-rag",
        help='Comma-separated list of conditions to run: dense-rag, bm25-rag, no-rag, all'
             '  (default: dense-rag)',
    )
    parser.add_argument("--n", type=int, default=None,
                        help="Number of QA items to evaluate (default: all)")
    parser.add_argument("--k", type=int, default=5,
                        help="Retrieval top-k (default: 5)")
    parser.add_argument("--client-type", default="openrouter",
                        choices=["ollama", "openrouter"],
                        help="LLM backend for generation + judge (default: openrouter)")
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: results/eval_<timestamp>.json)")
    parser.add_argument("--qa-file", default=str(QA_FILE),
                        help="Path to the QA JSONL file")
    parser.add_argument("--fetch-k", type=int, default=None,
                        help="Reranker over-fetch count for rerank-rag (default: RERANKER_FETCH_K from config)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-question progress output")
    args = parser.parse_args()

    # Resolve conditions
    if args.conditions.strip().lower() == "all":
        conditions = ["dense-rag", "bm25-rag", "rerank-rag", "no-rag"]
    else:
        conditions = [c.strip() for c in args.conditions.split(",")]

    qa_items = load_qa(Path(args.qa_file), n=args.n)
    print(f"Loaded {len(qa_items)} QA items from {args.qa_file}")

    llm_client = LLMClientFactory.create_llm_client(config, args.client_type)

    # Load embedding model once (shared across conditions)
    from ..embedder import Embedder
    from ..paths import INDEX_DIR
    _, _, idx_config = Embedder.load_index(INDEX_DIR)
    embed_fn = make_embed_fn(idx_config["model"])

    all_results = []
    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {condition}  |  k={args.k}  |  n={len(qa_items)}")
        print("="*60)
        result = run_condition(
            condition=condition,
            qa_items=qa_items,
            llm_client=llm_client,
            embed_fn=embed_fn,
            k=args.k,
            verbose=not args.quiet,
        )
        all_results.append(result)

        agg = result["aggregate"]
        print(f"\n--- {condition} summary ---")
        for metric, stats in agg.items():
            print(f"  {metric:20s}  mean={stats['mean']:.3f}  "
                  f"min={stats['min']:.3f}  max={stats['max']:.3f}  n={stats['n']}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (
        RESULTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    payload = {
        "timestamp": datetime.now().isoformat(),
        "k": args.k,
        "client_type": args.client_type,
        "conditions": all_results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
