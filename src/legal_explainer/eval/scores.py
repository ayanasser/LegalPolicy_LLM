"""Deterministic (non-LLM) evaluation metrics.

These are computed directly from a system's prediction + the gold reference, with
no judge calls:

  citation_accuracy   answer cites the gold article number
  has_citation        answer cites *any* article
  retrieval_hit@k     gold article is among the top-k retrieved articles
  retrieval_mrr       1 / rank of the gold article (0 if absent)
  context_precision@k fraction of the top-k retrieved that are the gold article
  token_overlap_gold  Jaccard of answer vs gold tokens (cheap relatedness proxy)

The LLM-judged metrics (faithfulness, answer_relevance, context_precision/recall,
answer_correctness) live in `judge.py` / `prompts.py`.
"""
from __future__ import annotations

from .text_utils import cited_article_numbers, jaccard, tokens


def citation_accuracy(answer: str, gold_article: int | None) -> bool:
    if gold_article is None:
        return False
    return gold_article in cited_article_numbers(answer)


def has_citation(answer: str) -> bool:
    return bool(cited_article_numbers(answer))


def retrieval_hit(retrieved: list[int], gold_article: int | None, k: int) -> bool:
    if gold_article is None:
        return False
    return gold_article in (retrieved or [])[:k]


def retrieval_mrr(retrieved: list[int], gold_article: int | None) -> float:
    if gold_article is None:
        return 0.0
    for rank, num in enumerate(retrieved or [], 1):
        if num == gold_article:
            return round(1.0 / rank, 4)
    return 0.0


def context_precision_at_k(retrieved: list[int], gold_article: int | None, k: int) -> float:
    """Fraction of the top-k retrieved articles that match the gold reference.

    With a single gold article this is at most 1/k; it rewards surfacing the
    right article without padding the context with the same number repeatedly."""
    if gold_article is None or not retrieved:
        return 0.0
    topk = retrieved[:k]
    return round(sum(1 for n in topk if n == gold_article) / max(1, len(topk)), 4)


def token_overlap_gold(answer: str, gold_answer: str) -> float:
    return jaccard(tokens(answer), tokens(gold_answer))


def deterministic_metrics(answer: str, retrieved: list[int],
                          gold_article: int | None, gold_answer: str, k: int) -> dict:
    return {
        "citation_accuracy": citation_accuracy(answer, gold_article),
        "has_citation": has_citation(answer),
        "retrieval_hit@k": retrieval_hit(retrieved, gold_article, k),
        "retrieval_mrr": retrieval_mrr(retrieved, gold_article),
        "context_precision@k": context_precision_at_k(retrieved, gold_article, k),
        "token_overlap_gold": token_overlap_gold(answer, gold_answer),
    }
