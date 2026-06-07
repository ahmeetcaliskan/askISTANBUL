"""
eval/auto_label.py
------------------
Automatically assign relevant_chunk_ids to QA pairs.

For each question, retrieves the top-K chunks and asks an LLM judge whether
each chunk contains information needed to answer the question (given the
ground truth). Chunks marked as supporting are written as relevant_chunk_ids.

Usage:
  python -m askistanbul.eval.auto_label                        # all items, top-20
  python -m askistanbul.eval.auto_label --n 10                 # first 10 only
  python -m askistanbul.eval.auto_label --fetch-k 30           # wider candidate pool
  python -m askistanbul.eval.auto_label --client-type ollama   # local judge
  python -m askistanbul.eval.auto_label --dry-run              # preview without saving
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ..config import config
from ..generator.factory.LLMClientFactory import LLMClientFactory
from ..paths import EVAL_DIR
from ..retriever import DenseRetriever

QA_FILE = EVAL_DIR / "qa_draft_with_relevant_chunks.jsonl"

# ---------------------------------------------------------------------------
# LLM judge prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a strict relevance judge for a travel Q&A system about Istanbul.

You will be given:
- A QUESTION
- A GROUND TRUTH answer
- A PASSAGE from a travel guide

Your task: decide whether the passage contains information that directly
supports or is needed to produce the ground truth answer.

Respond with JSON only (no markdown):
{"relevant": true}   — if the passage supports the ground truth
{"relevant": false}  — if it does not
"""

_USER = """\
QUESTION: {question}

GROUND TRUTH: {ground_truth}

PASSAGE (from: {title} / {heading}):
{text}

Does this passage support the ground truth answer?
"""


def is_relevant(
    question: str,
    ground_truth: str,
    chunk: dict,
    llm_client,
) -> bool:
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER.format(
            question=question,
            ground_truth=ground_truth,
            title=chunk.get("title", ""),
            heading=chunk.get("heading", ""),
            text=chunk.get("text", "")[:800],
        )},
    ]
    try:
        result = llm_client.chat_json(messages, temperature=0.0, max_new_tokens=20)
        return bool(result.get("relevant", False))
    except Exception:
        # Fallback: parse from plain text
        try:
            raw = llm_client.chat(messages, temperature=0.0, max_new_tokens=20)
            raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
            return json.loads(raw).get("relevant", False)
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label relevant_chunk_ids in QA set.")
    parser.add_argument("--qa-file", default=str(QA_FILE))
    parser.add_argument("--n", type=int, default=None, help="Number of items to process")
    parser.add_argument("--fetch-k", type=int, default=20,
                        help="Candidate pool size per question (default: 20)")
    parser.add_argument("--client-type", default="openrouter",
                        choices=["ollama", "openrouter"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results without saving")
    parser.add_argument("--out", default=None,
                        help="Output path (default: overwrites input file)")
    args = parser.parse_args()

    qa_path = Path(args.qa_file)
    items = []
    with qa_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if args.n:
        items = items[:args.n]

    print(f"Loaded {len(items)} QA items")
    print(f"Fetch-k={args.fetch_k}, client={args.client_type}")

    llm_client = LLMClientFactory.create_llm_client(config, args.client_type)
    retriever  = DenseRetriever()

    updated = []
    for i, item in enumerate(items, 1):
        qid          = item["id"]
        question     = item["question"]
        ground_truth = item.get("ground_truth", "")

        print(f"\n[{i:3d}/{len(items)}] Q{qid}: {question[:70]}...")

        # Retrieve candidate chunks
        results   = retriever.retrieve(question, biencoder_k=args.fetch_k)
        relevant_ids: list[str] = []

        for r in results:
            chunk = r.chunk.to_dict()
            verdict = is_relevant(question, ground_truth, chunk, llm_client)
            marker  = "✓" if verdict else "✗"
            print(f"   {marker} {chunk['chunk_id']:45s} | {chunk['heading'][:40]}")
            if verdict:
                relevant_ids.append(chunk["chunk_id"])

        print(f"   → {len(relevant_ids)} relevant chunks found")
        item["relevant_chunk_ids"] = relevant_ids
        updated.append(item)

    if args.dry_run:
        print("\n[dry-run] Not saving.")
        return

    out_path = Path(args.out) if args.out else qa_path
    with out_path.open("w", encoding="utf-8") as fh:
        for item in updated:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(updated)} items → {out_path}")


if __name__ == "__main__":
    main()
