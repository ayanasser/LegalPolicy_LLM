"""Assemble the RAFT context-block prompt.

This deliberately reuses ``dataset_builder._format_context_block`` /
``_clean_article_text`` so the prompt the RAFT adapter sees at inference is the
*same shape* it was fine-tuned on (header + ``[Article N]`` blocks + ``Question:``).
If this drifts from the training format, the open-book accuracy gain vanishes —
so this module is intentionally thin.
"""
from __future__ import annotations

import random

from legal_explainer.finetune.dataset_builder import _clean_article_text, _format_context_block

from .retriever import RetrievalResult

_EMPTY_MARKER_EN = "[No relevant Egyptian Civil Code articles for this request.]"
_EMPTY_MARKER_AR = "[لا توجد مواد ذات صلة من القانون المدني المصري لهذا الطلب.]"


def build_prompt(question: str, result: RetrievalResult, lang: str, *,
                 max_article_chars: int = 900, shuffle_seed: int | None = 13) -> str:
    """`question` (already phrased in `lang`) -> the full RAFT-style user message.

    With no retrieved context (empty result — e.g. the 'closed' eval mode or a
    refusal-type request), emit the same empty-context marker the RAFT training
    data used for the oracle-absent case.
    """
    entries = result.context_entries
    if not entries:
        ctx_label = "Context" if lang == "en" else "السياق"
        marker = _EMPTY_MARKER_EN if lang == "en" else _EMPTY_MARKER_AR
        return _format_context_block([(ctx_label, marker)], lang) + question

    pairs = [(e.label(lang), _clean_article_text(e.text(lang), max_article_chars)) for e in entries]
    if shuffle_seed is not None and len(pairs) > 1:
        random.Random(shuffle_seed).shuffle(pairs)        # RAFT training shuffled the entries too
    return _format_context_block(pairs, lang) + question
