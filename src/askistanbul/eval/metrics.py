"""
eval/metrics.py
---------------
Standalone metric functions for the AskIstanbul evaluation suite.

Retrieval metrics:
  precision_at_k(retrieved_ids, relevant_ids, k)
      → float   Fraction of top-k retrieved chunks that are relevant.

  recall_at_k(retrieved_ids, relevant_ids, k)
      → float   Fraction of all relevant chunks captured in the top-k.
                Important for synthesis questions that have multiple gold chunks.

  mrr(retrieved_ids, relevant_ids)
      → float   Reciprocal rank of the first relevant chunk (1/rank).
                Best for factual questions with a single key answer chunk.

  ndcg_at_k(retrieved_ids, relevant_ids, k)
      → float   Normalised Discounted Cumulative Gain at k (binary relevance).
                Rewards placing relevant chunks near the top of the ranking;
                1.0 means every relevant chunk is packed into the highest ranks.

Generation metrics:
  faithfulness(answer, context_chunks, llm_client)
      → float   Fraction of answer claims supported by the retrieved context.
                Uses an LLM judge; mirrors RAGAS faithfulness.

  answer_relevance(question, answer, embed_fn)
      → float   Cosine similarity between question and answer embeddings.
                Mirrors RAGAS answer relevance (no LLM required).
"""

from __future__ import annotations

import json
import re
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# 1. Precision@k
# ---------------------------------------------------------------------------

def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int | None = None,
) -> float:
    """Fraction of the top-k retrieved chunks that appear in relevant_ids.

    Args:
        retrieved_ids: chunk_ids returned by the retriever, ranked best-first.
        relevant_ids:  gold-standard relevant chunk_ids from the QA set.
        k:             cut-off; defaults to len(retrieved_ids).

    Returns:
        float in [0, 1].  Returns 0.0 if retrieved_ids or relevant_ids is empty.
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0
    top = retrieved_ids[:k] if k else retrieved_ids
    relevant_set = set(relevant_ids)
    hits = sum(1 for cid in top if cid in relevant_set)
    return hits / len(top)


# ---------------------------------------------------------------------------
# 2. Recall@k
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int | None = None,
) -> float:
    """Fraction of all relevant chunks captured in the top-k.

    Complements Precision@k: a synthesis question may need 3 gold chunks;
    Precision tells you how clean the retrieved set is, Recall tells you
    how complete it is.

    Args:
        retrieved_ids: chunk_ids returned by the retriever, ranked best-first.
        relevant_ids:  gold-standard relevant chunk_ids from the QA set.
        k:             cut-off; defaults to len(retrieved_ids).

    Returns:
        float in [0, 1].  Returns 0.0 if relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0
    top = retrieved_ids[:k] if k else retrieved_ids
    relevant_set = set(relevant_ids)
    hits = sum(1 for cid in top if cid in relevant_set)
    return hits / len(relevant_set)


# ---------------------------------------------------------------------------
# 3. MRR (Mean Reciprocal Rank)
# ---------------------------------------------------------------------------

def mrr(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """Reciprocal rank of the first relevant chunk in the retrieved list.

    MRR = 1/rank of the first hit, or 0 if no hit is found.
    Best interpreted for factual questions where one key chunk contains
    the answer; the metric rewards surfacing that chunk as early as possible.

    Args:
        retrieved_ids: chunk_ids returned by the retriever, ranked best-first.
        relevant_ids:  gold-standard relevant chunk_ids from the QA set.

    Returns:
        float in (0, 1].  Returns 0.0 if no relevant chunk is retrieved.
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# 4. NDCG@k (Normalised Discounted Cumulative Gain, binary relevance)
# ---------------------------------------------------------------------------

def _dcg(rels: list[int]) -> float:
    """Discounted Cumulative Gain of a relevance list (rank 1 = first item).

    DCG = sum_i  rel_i / log2(i + 1)   with i the 1-based rank.
    """
    import math
    return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(rels, start=1))


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int | None = None,
) -> float:
    """Normalised DCG at k with binary relevance.

    Unlike Precision@k (which ignores order) and MRR (which only looks at the
    first hit), NDCG rewards ranking *all* relevant chunks as high as possible.
    The ideal ranking puts every relevant chunk first, so IDCG is the DCG of
    min(#relevant, k) ones.

    Args:
        retrieved_ids: chunk_ids returned by the retriever, ranked best-first.
        relevant_ids:  gold-standard relevant chunk_ids from the QA set.
        k:             cut-off; defaults to len(retrieved_ids).

    Returns:
        float in [0, 1].  Returns 0.0 if retrieved_ids or relevant_ids is empty.
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0
    cut = k if k else len(retrieved_ids)
    top = retrieved_ids[:cut]
    relevant_set = set(relevant_ids)

    gains = [1 if cid in relevant_set else 0 for cid in top]
    dcg = _dcg(gains)

    n_ideal = min(len(relevant_set), cut)
    idcg = _dcg([1] * n_ideal)

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# 5. Faithfulness  (LLM judge)
# ---------------------------------------------------------------------------

_FAITHFULNESS_SYSTEM = """\
You are a balanced evaluation judge. Your task is to assess whether each claim
in an AI-generated answer is grounded in the provided context passages.

Respond with a JSON object (no markdown fences) in this exact schema:
{
  "claims": [
    {"claim": "<verbatim claim>", "supported": true | false},
    ...
  ]
}

Rules:
- Extract every distinct factual claim from the answer (typically 3-8).
- Mark "supported": true if the claim is explicitly stated, clearly inferrable,
  or consistent with information in the context — even if phrased differently.
- Mark "supported": false only if the claim directly contradicts the context
  or introduces specific facts (names, numbers, prices) absent from the context.
- Ignore formatting artifacts like citation markers ([1], [2]) — do not treat
  them as claims.
- Give the benefit of the doubt for reasonable paraphrases of context content.
"""

_FAITHFULNESS_USER = """\
### Context
{context}

### Answer
{answer}

For each factual claim in the Answer, judge whether it is supported by the Context above.
"""


def faithfulness(
    answer: str,
    context_chunks: list[str],
    llm_client,
    max_context_chars: int = 8000,
) -> tuple[float, list[dict]]:
    """RAGAS-style faithfulness: fraction of answer claims supported by context.

    Args:
        answer:          Generated answer text.
        context_chunks:  List of retrieved chunk texts.
        llm_client:      Any BaseLLMClient instance (used as judge).
        max_context_chars: Truncate pasted context to avoid token limits.

    Returns:
        (score, claims) where score ∈ [0, 1] and claims is the raw judge output.
    """
    context = "\n\n---\n\n".join(context_chunks)
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n[...truncated...]"

    messages = [
        {"role": "system", "content": _FAITHFULNESS_SYSTEM},
        {"role": "user",   "content": _FAITHFULNESS_USER.format(
            context=context, answer=answer
        )},
    ]

    try:
        result = llm_client.chat_json(messages, temperature=0.0, max_new_tokens=1000)
        claims = result.get("claims", [])
    except Exception as exc:
        # Fallback: try to parse JSON from a plain chat response
        try:
            raw = llm_client.chat(messages, temperature=0.0, max_new_tokens=1000)
            # strip markdown fences if present
            raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()
            result = json.loads(raw)
            claims = result.get("claims", [])
        except Exception:
            return 0.0, []

    if not claims:
        return 0.0, []

    # Normalize: LLMs sometimes return a list of strings instead of dicts.
    # Convert any string item to {"claim": <str>, "supported": False} so the
    # rest of the pipeline always sees uniform dicts.
    normalized = []
    for c in claims:
        if isinstance(c, dict):
            normalized.append(c)
        elif isinstance(c, str):
            # Try to parse the string as JSON first (e.g. '{"claim":...}')
            try:
                parsed = json.loads(c)
                normalized.append(parsed if isinstance(parsed, dict) else {"claim": c, "supported": False})
            except (json.JSONDecodeError, ValueError):
                normalized.append({"claim": c, "supported": False})
        else:
            normalized.append({"claim": str(c), "supported": False})

    supported = sum(1 for c in normalized if c.get("supported", False))
    return supported / len(normalized), normalized


# ---------------------------------------------------------------------------
# 3. Answer Relevance  (embedding cosine similarity)
# ---------------------------------------------------------------------------

def answer_relevance(
    question: str,
    answer: str,
    embed_fn: Callable[[list[str]], np.ndarray],
) -> float:
    """Cosine similarity between question and answer embeddings.

    Mirrors RAGAS answer_relevance without requiring an LLM.

    Args:
        question:  Original user question.
        answer:    Generated answer.
        embed_fn:  Callable that takes a list of strings and returns a
                   (N, dim) float32 numpy array of L2-normalised embeddings.
                   Pass the same model used for retrieval.

    Returns:
        float in [-1, 1]; values near 1 mean the answer is on-topic.
    """
    if not answer or not question:
        return 0.0

    vecs = embed_fn([question, answer])   # shape (2, dim)
    # dot product of L2-normalised vectors = cosine similarity
    return float(np.dot(vecs[0], vecs[1]))


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def aggregate(scores: list[float]) -> dict[str, float]:
    """Return mean, min, max for a list of per-question scores."""
    if not scores:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    arr = np.array(scores)
    return {
        "mean": float(arr.mean()),
        "min":  float(arr.min()),
        "max":  float(arr.max()),
        "n":    len(scores),
    }
