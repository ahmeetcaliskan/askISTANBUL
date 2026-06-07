"""
eval/ablation.py
----------------
Retrieval-only ablation sweeps for AskIstanbul (proposal Week 4).

These sweeps deliberately skip LLM generation: retrieval metrics
(Precision@k / Recall@k / MRR / NDCG@k) need only the retriever and the
gold ``relevant_chunk_ids``, so the whole grid runs in seconds-to-minutes
on CPU with no API cost.

Three sweeps:

  k          Vary top-k on the *existing* primary index, dense vs BM25.
             chunk_ids are stable, so gold relevance is matched by id.

  encoder    Re-embed the *existing* chunks with a different sentence-
             transformer (e.g. all-MiniLM-L6-v2 vs multilingual-e5-base).
             chunk_ids are stable → id matching.

  chunk      Re-chunk the cleaned corpus at different (chunk_size, overlap)
             then re-embed. chunk_ids CHANGE, so gold relevance can no longer
             be matched by id; we fall back to text-overlap matching (a
             retrieved chunk counts as a hit if it has high token-Jaccard with
             any gold chunk's text from the primary index).

Usage:
  python -m askistanbul.eval.ablation --sweep k
  python -m askistanbul.eval.ablation --sweep encoder
  python -m askistanbul.eval.ablation --sweep chunk
  python -m askistanbul.eval.ablation --sweep all          # everything
  python -m askistanbul.eval.ablation --sweep chunk --n 30 # quick subset
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..chunker import Chunker
from ..embedder import Embedder
from ..paths import CLEAN_DIR, EVAL_DIR, INDEX_DIR, PROJECT_ROOT
from ..retriever import BM25Retriever, DenseRetriever
from .metrics import aggregate, mrr, ndcg_at_k, precision_at_k, recall_at_k

RESULTS_DIR = PROJECT_ROOT / "results"
QA_FILE = EVAL_DIR / "qa_draft_with_relevant_chunks.jsonl"
ABLATION_INDEX_ROOT = PROJECT_ROOT / "data" / "index_ablation"

# Defaults for the sweeps (kept small so the grid finishes quickly).
K_VALUES = [1, 3, 5, 10, 20]
ENCODERS = ["sentence-transformers/all-MiniLM-L6-v2", "intfloat/multilingual-e5-base"]
CHUNK_CONFIGS = [
    (150, 30),
    (200, 50),   # the project default
    (300, 75),
    (400, 100),
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_qa(path: Path, n: Optional[int] = None) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    # Only items with gold labels are useful for retrieval metrics.
    items = [it for it in items if it.get("relevant_chunk_ids")]
    return items[:n] if n else items


def _tokens(text: str) -> set[str]:
    return {t for t in text.lower().split() if t}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b)


# ---------------------------------------------------------------------------
# Metric computation — id-based (stable chunk_ids)
# ---------------------------------------------------------------------------

def eval_by_id(retriever, qa_items: list[dict], k: int) -> dict:
    """Retrieval metrics with id-based gold matching."""
    P, R, M, N = [], [], [], []
    for item in qa_items:
        gold = item["relevant_chunk_ids"]
        results = retriever.retrieve(item["question"], biencoder_k=k)
        rids = [r.chunk.chunk_id for r in results]
        P.append(precision_at_k(rids, gold, k=k))
        R.append(recall_at_k(rids, gold, k=k))
        M.append(mrr(rids, gold))
        N.append(ndcg_at_k(rids, gold, k=k))
    return {
        "precision_at_k": aggregate(P),
        "recall_at_k":    aggregate(R),
        "mrr":            aggregate(M),
        "ndcg_at_k":      aggregate(N),
    }


# ---------------------------------------------------------------------------
# Metric computation — text-based (re-chunked, ids no longer comparable)
# ---------------------------------------------------------------------------

def build_gold_texts(qa_items: list[dict], threshold: float) -> dict:
    """Map each question id → list of gold chunk token-sets from the primary index.

    Lets us judge relevance of a *re-chunked* retrieval by text overlap rather
    than by chunk_id, which is meaningless across different chunking schemes.
    """
    _, chunks, _ = Embedder.load_index(INDEX_DIR)
    by_id = {c.chunk_id: c.text for c in chunks}
    gold_texts: dict = {}
    for item in qa_items:
        toks = []
        for cid in item["relevant_chunk_ids"]:
            txt = by_id.get(cid)
            if txt:
                toks.append(_tokens(txt))
        gold_texts[item["id"]] = toks
    return gold_texts


def eval_by_text(
    retriever,
    qa_items: list[dict],
    gold_texts: dict,
    k: int,
    threshold: float = 0.4,
) -> dict:
    """Retrieval metrics with text-overlap gold matching (for re-chunked indexes)."""
    P, R, M, N = [], [], [], []
    for item in qa_items:
        golds = gold_texts.get(item["id"], [])
        if not golds:
            continue
        results = retriever.retrieve(item["question"], biencoder_k=k)
        rtoks = [_tokens(r.chunk.text) for r in results]

        # hits[i] = does retrieved chunk i overlap *any* gold chunk?
        hits = [
            max((_jaccard(rt, g) for g in golds), default=0.0) >= threshold
            for rt in rtoks
        ]
        # covered = how many distinct gold chunks are matched by some retrieved one
        covered = sum(
            1 for g in golds
            if any(_jaccard(rt, g) >= threshold for rt in rtoks)
        )

        n_hit = sum(hits)
        P.append(n_hit / k)
        R.append(covered / len(golds))
        # MRR: reciprocal rank of first hit
        rr = 0.0
        for rank, h in enumerate(hits, start=1):
            if h:
                rr = 1.0 / rank
                break
        M.append(rr)
        # NDCG with binary gains from `hits`
        import math
        dcg = sum(1.0 / math.log2(r + 1) for r, h in enumerate(hits, start=1) if h)
        n_ideal = min(len(golds), k)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, n_ideal + 1))
        N.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        "precision_at_k": aggregate(P),
        "recall_at_k":    aggregate(R),
        "mrr":            aggregate(M),
        "ndcg_at_k":      aggregate(N),
    }


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def sweep_k(qa_items: list[dict], k_values: list[int]) -> list[dict]:
    print("\n### Sweep: top-k (dense vs bm25, primary index) ###")
    dense = DenseRetriever()
    bm25 = BM25Retriever()
    runs = []
    for method, retr in (("dense", dense), ("bm25", bm25)):
        for k in k_values:
            t0 = time.time()
            metrics = eval_by_id(retr, qa_items, k=k)
            dt = time.time() - t0
            runs.append({"sweep": "k", "method": method, "k": k,
                         "match": "id", "metrics": metrics})
            _print_row(f"{method:5s} k={k:<2d}", metrics, dt)
    return runs


def sweep_encoder(qa_items: list[dict], encoders: list[str]) -> list[dict]:
    print("\n### Sweep: embedding encoder (re-embed primary chunks) ###")
    _, chunks, _ = Embedder.load_index(INDEX_DIR)
    runs = []
    ABLATION_INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    for model_name in encoders:
        idx_dir = ABLATION_INDEX_ROOT / ("enc_" + model_name.split("/")[-1])
        print(f"\n  Embedding {len(chunks)} chunks with {model_name} ...")
        Embedder(model_name=model_name).build_index(chunks, index_dir=idx_dir)
        retr = DenseRetriever(index_dir=idx_dir)
        for k in (5,):
            t0 = time.time()
            metrics = eval_by_id(retr, qa_items, k=k)
            dt = time.time() - t0
            runs.append({"sweep": "encoder", "encoder": model_name, "k": k,
                         "match": "id", "metrics": metrics})
            _print_row(f"{model_name.split('/')[-1]:28s} k={k}", metrics, dt)
    return runs


def sweep_chunk(
    qa_items: list[dict],
    chunk_configs: list[tuple[int, int]],
    encoder: str,
    threshold: float,
) -> list[dict]:
    print("\n### Sweep: chunk_size / overlap (re-chunk + re-embed, text matching) ###")
    gold_texts = build_gold_texts(qa_items, threshold)
    runs = []
    ABLATION_INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    embedder = Embedder(model_name=encoder)  # reuse one model load across configs
    for chunk_size, overlap in chunk_configs:
        tag = f"cs{chunk_size}_ov{overlap}"
        chunks_file = ABLATION_INDEX_ROOT / f"chunks_{tag}.jsonl"
        idx_dir = ABLATION_INDEX_ROOT / f"idx_{tag}"
        print(f"\n  Chunking cs={chunk_size} ov={overlap} ...")
        chunks = Chunker(chunk_size=chunk_size, overlap=overlap).chunk_all(
            clean_dir=CLEAN_DIR, out_file=chunks_file
        )
        embedder.build_index(chunks, index_dir=idx_dir)
        retr = DenseRetriever(embedder=embedder, index_dir=idx_dir)
        for k in (5,):
            t0 = time.time()
            metrics = eval_by_text(retr, qa_items, gold_texts, k=k, threshold=threshold)
            dt = time.time() - t0
            runs.append({"sweep": "chunk", "chunk_size": chunk_size, "overlap": overlap,
                         "encoder": encoder, "k": k, "match": "text",
                         "threshold": threshold, "n_chunks": len(chunks),
                         "metrics": metrics})
            _print_row(f"cs={chunk_size:<3d} ov={overlap:<3d} ({len(chunks)} chunks)", metrics, dt)
    return runs


def _print_row(label: str, metrics: dict, dt: float) -> None:
    p = metrics["precision_at_k"]["mean"]
    r = metrics["recall_at_k"]["mean"]
    m = metrics["mrr"]["mean"]
    n = metrics["ndcg_at_k"]["mean"]
    print(f"    {label:42s}  P={p:.3f}  R={r:.3f}  MRR={m:.3f}  NDCG={n:.3f}  ({dt:.1f}s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AskIstanbul retrieval ablation sweeps")
    parser.add_argument("--sweep", default="all",
                        choices=["k", "encoder", "chunk", "all"],
                        help="Which sweep to run (default: all)")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N labeled QA items (default: all)")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Token-Jaccard threshold for text matching in the chunk sweep")
    parser.add_argument("--chunk-encoder", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Encoder used during the chunk-size sweep")
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--keep-indexes", action="store_true",
                        help="Keep the temporary ablation indexes instead of deleting them")
    args = parser.parse_args()

    qa_items = load_qa(QA_FILE, n=args.n)
    print(f"Loaded {len(qa_items)} labeled QA items from {QA_FILE}")

    runs: list[dict] = []
    if args.sweep in ("k", "all"):
        runs += sweep_k(qa_items, K_VALUES)
    if args.sweep in ("encoder", "all"):
        runs += sweep_encoder(qa_items, ENCODERS)
    if args.sweep in ("chunk", "all"):
        runs += sweep_chunk(qa_items, CHUNK_CONFIGS, args.chunk_encoder, args.threshold)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (
        RESULTS_DIR / f"ablation_{args.sweep}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    payload = {
        "timestamp": datetime.now().isoformat(),
        "sweep": args.sweep,
        "n_questions": len(qa_items),
        "runs": runs,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAblation results saved → {out_path}")

    if not args.keep_indexes and ABLATION_INDEX_ROOT.exists():
        shutil.rmtree(ABLATION_INDEX_ROOT)
        print(f"Cleaned up temporary indexes in {ABLATION_INDEX_ROOT}")


if __name__ == "__main__":
    main()
