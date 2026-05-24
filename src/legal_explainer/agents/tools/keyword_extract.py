"""Deterministic keyword extraction for LightRAG queries — NLTK + Punkt.

LightRAG's `QueryParam` accepts pre-populated `hl_keywords` (high-level themes)
and `ll_keywords` (low-level specifics). If either is non-empty, LightRAG
skips its built-in LLM-based extractor entirely (lightrag/operate.py:3316).

This module produces both lists from a query string using NLTK's Punkt
tokenizer + bilingual (EN + AR) stopword lists from `nltk.corpus.stopwords`.
No LLM call is made.

NLTK data is downloaded on first use if missing — `punkt_tab` (Punkt
tokenizer) and `stopwords` (English + Arabic).
"""

from __future__ import annotations

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from legal_explainer.agents.tools.glossary import _build_lookup, _load_glossary
from legal_explainer.agents.tools.statute import _REF_RE


# ── NLTK bootstrap ───────────────────────────────────────────────────────────
def _ensure_nltk_data() -> None:
    """Download Punkt + stopwords on first use. Idempotent — re-runs are no-ops."""
    for resource, lookup in (
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
    ):
        try:
            nltk.data.find(lookup)
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk_data()
_STOPWORDS = set(stopwords.words("english")) | set(stopwords.words("arabic"))


def _content_tokens(query: str) -> list[str]:
    """Tokenize with Punkt, drop stopwords + 1-2 char tokens + pure punctuation."""
    return [
        tok.lower()
        for tok in word_tokenize(query)
        if tok.isalnum() and len(tok) >= 3 and tok.lower() not in _STOPWORDS
    ]


def _glossary_hits(query: str) -> tuple[set[str], set[str]]:
    """Find glossary entries whose term/alias appears in the (lowercased) query."""
    q_lower = query.lower()
    lookup = _build_lookup()
    glossary = _load_glossary()

    matched_ids: set[str] = set()
    matched_surface: set[str] = set()
    for variant, canonical_id in lookup.items():
        if variant in q_lower:
            matched_ids.add(canonical_id)
            matched_surface.add(variant)

    # Always include both EN + AR canonical terms for every matched concept —
    # gives the entity-vector search both languages to match.
    for cid in matched_ids:
        matched_surface.add(glossary[cid]["en"]["term"].lower())
        matched_surface.add(glossary[cid]["ar"]["term"])

    return matched_surface, matched_ids


def extract_keywords(query: str) -> tuple[list[str], list[str]]:
    """Return (hl_keywords, ll_keywords) for a query. Both lists deduplicated.

    Low-level (specific terms, used for entity matching):
      - article references (Article 89 + المادة 89 variants)
      - glossary surface terms that appeared in the query
      - significant content tokens after stopword removal

    High-level (themes, used for relationship matching):
      - canonical glossary concept ids
      - the de-stopworded query joined as a single thematic string
    """
    low_level: list[str] = []
    high_level: list[str] = []
    seen_low: set[str] = set()
    seen_high: set[str] = set()

    def _add_low(kw: str) -> None:
        kw = kw.strip()
        if kw and kw not in seen_low:
            seen_low.add(kw)
            low_level.append(kw)

    def _add_high(kw: str) -> None:
        kw = kw.strip()
        if kw and kw not in seen_high:
            seen_high.add(kw)
            high_level.append(kw)

    # 1. Article references → low-level (both EN + AR forms for matching)
    for m in _REF_RE.finditer(query):
        _add_low(f"Article {m.group(1)}")
        _add_low(f"المادة {m.group(1)}")

    # 2. Glossary hits → surface to low-level, canonical id to high-level
    surface, ids = _glossary_hits(query)
    for s in surface:
        _add_low(s)
    for cid in ids:
        _add_high(cid.replace("_", " "))

    # 3. Content tokens → low-level
    content_toks = _content_tokens(query)
    for tok in content_toks:
        _add_low(tok)

    # 4. Thematic high-level keyword from the de-stopworded query
    if content_toks:
        _add_high(" ".join(content_toks[:10]))

    # 5. Safety net: never let both lists be empty, or LightRAG will fall
    #    back to its LLM extractor.
    if not high_level:
        _add_high(query.strip())
    if not low_level:
        _add_low(query.strip())

    return high_level, low_level
