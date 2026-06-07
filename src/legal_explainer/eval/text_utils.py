"""Shared text helpers for evaluation (citation / digit / language utilities).

Consolidates the small regex helpers that were previously copy-pasted across
`scripts/eval_csv_closedbook.py`, `scripts/closed_book_recall_eval.py` and
`src/legal_explainer/finetune/raft_rag/eval.py`, so every evaluator extracts
article numbers and detects language the same way.
"""
from __future__ import annotations

import re

# Arabic-Indic digits (٠-٩) → ASCII so "٤٤٦" == "446".
AR2EN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# "Article 5", "Art. 5", "المادة ٥", "مادة 5"
_CITE_RE = re.compile(
    r"(?:article|art\.?|المادة|مادة)\s*[#:]?\s*0*([0-9٠-٩]{1,4})", re.IGNORECASE
)
_NUM_RE = re.compile(r"([0-9٠-٩]{1,4})")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_AR_CHARS = re.compile(r"[؀-ۿ]")

# Common stop-words (AR + EN) so token overlap is not noise-dominated.
_STOP = {
    "في", "من", "إلى", "على", "عن", "أن", "إن", "أو", "و", "ال", "هذا", "هذه",
    "ذلك", "تلك", "هو", "هي", "ما", "ماذا", "كيف", "متى", "أين", "لماذا",
    "كان", "كانت", "يكون", "تكون", "قد", "لو", "لا", "نعم", "بس", "ده", "دي",
    "كل", "مع", "بين", "تحت", "فوق", "بعد", "قبل", "حيث", "إذا", "أم",
    "the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "be", "was",
    "were", "for", "on", "with", "as", "by", "this", "that", "it", "if", "from",
}


def cited_article_numbers(text: str) -> set[int]:
    """Every article number explicitly cited in the text (any digit script)."""
    out: set[int] = set()
    for raw in _CITE_RE.findall(text or ""):
        d = raw.translate(AR2EN_DIGITS)
        if d.isdigit():
            out.add(int(d))
    return out


def parse_reference_number(ref: str | None) -> int | None:
    """Pull the article number out of a `legal_reference` cell, e.g. 'مادة 4' → 4."""
    if not ref:
        return None
    m = _NUM_RE.search(str(ref))
    if not m:
        return None
    d = m.group(1).translate(AR2EN_DIGITS)
    return int(d) if d.isdigit() else None


def detect_language(text: str) -> str:
    """'ar' if ≥20% of non-space chars are Arabic, else 'en'."""
    if not text:
        return "en"
    non_space = len(text.replace(" ", "")) or 1
    return "ar" if len(_AR_CHARS.findall(text)) >= 0.20 * non_space else "en"


def tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall((text or "").lower()) if t not in _STOP and len(t) > 1}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return round(len(a & b) / max(1, len(a | b)), 4)
