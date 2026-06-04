"""Evaluation subpackage — QA-set generation, retrieval/generation metrics, harness."""

from .metrics import precision_at_k, recall_at_k, mrr, faithfulness, answer_relevance, aggregate

__all__ = ["precision_at_k", "recall_at_k", "mrr", "faithfulness", "answer_relevance", "aggregate"]
