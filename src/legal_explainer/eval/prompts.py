"""Judge rubric for RAG / answer evaluation.

A single rubric call per row returns the five RAGAS-style LLM-judged metrics
(all on a 0.0–1.0 scale). Deterministic metrics — citation accuracy, retrieval
hit@k, MRR — are computed without the judge (see `scores.py`).

The same rubric serves all three systems:
  * Graph RAG / Bilingual RAG → `contexts` are the retrieved article texts.
  * Fine-tuned (closed-book)   → `contexts` is the single gold article text, so
    faithfulness measures grounding against the law the answer *should* recall.
"""
from __future__ import annotations

JUDGE_SYSTEM = (
    "You are a meticulous evaluator for a legal question-answering system over the "
    "Egyptian Civil Code. You grade one answer at a time against retrieved context "
    "and a gold reference answer. Be strict, objective and consistent. The answer "
    "may be in Arabic or English — grade it in whatever language it is written.\n\n"
    "Return ONLY a JSON object (no markdown, no commentary) with these keys, each a "
    "float from 0.0 to 1.0:\n"
    '  "faithfulness"       — every factual/legal claim in the ANSWER is supported by '
    "the CONTEXT (1.0 = fully grounded, 0.0 = contradicted or invented).\n"
    '  "answer_relevance"   — the ANSWER directly and completely addresses the '
    "QUESTION (1.0 = fully on-point, 0.0 = off-topic/evasive).\n"
    '  "context_precision"  — proportion of the CONTEXT passages that are actually '
    "relevant to answering the QUESTION (1.0 = all relevant, 0.0 = all noise).\n"
    '  "context_recall"     — proportion of the information in the GOLD answer that is '
    "present in the CONTEXT (1.0 = everything needed was retrieved, 0.0 = nothing).\n"
    '  "answer_correctness" — the ANSWER agrees with the GOLD answer on the law '
    "(1.0 = legally equivalent, 0.0 = wrong).\n"
    'Also include "notes": one short sentence justifying the lowest score.\n'
    'Example: {"faithfulness": 0.8, "answer_relevance": 1.0, "context_precision": 0.6, '
    '"context_recall": 1.0, "answer_correctness": 0.9, "notes": "..."}'
)


def build_judge_user_prompt(
    question: str,
    answer: str,
    contexts: list[str],
    gold_answer: str,
    gold_article: int | None,
    closed_book: bool = False,
) -> str:
    if contexts:
        ctx_block = "\n\n".join(
            f"[Context {i}]\n{c.strip()}" for i, c in enumerate(contexts, 1)
        )
    else:
        ctx_block = "(no context was retrieved)"

    note = (
        "NOTE: this is a CLOSED-BOOK system — the CONTEXT below is the gold article "
        "the answer was expected to recall from memory; grade faithfulness against it.\n\n"
        if closed_book else ""
    )
    gold_ref = f"Article {gold_article}" if gold_article else "(unspecified)"
    return (
        f"{note}"
        f"QUESTION:\n{question.strip()}\n\n"
        f"CONTEXT:\n{ctx_block}\n\n"
        f"ANSWER (to grade):\n{(answer or '').strip()}\n\n"
        f"GOLD reference answer:\n{(gold_answer or '').strip()}\n"
        f"GOLD article: {gold_ref}\n\n"
        "Return the JSON verdict now."
    )


# The LLM-judged metric keys, in report order.
LLM_METRICS = (
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "answer_correctness",
)
