"""Glossary tool — deterministic lookup of canonical legal definitions.

Replaces the LLM's fuzzy recall with a hand-curated bilingual dictionary.
The tool resolves both English and Arabic terms (plus common aliases) to the
same canonical entry — so `get_legal_definition("indemnity")` and
`get_legal_definition("تعويض")` both return the `damages` entry.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from claude_agent_sdk import tool

from legal_explainer.agents.config import GLOSSARY_PATH


@lru_cache(maxsize=1)
def _load_glossary() -> dict[str, dict[str, Any]]:
    """Load and cache the glossary JSON. One-shot — no hot-reload."""
    with open(GLOSSARY_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _build_lookup() -> dict[str, str]:
    """Map every term variant (en, ar, aliases) → canonical id."""
    glossary = _load_glossary()
    lookup: dict[str, str] = {}
    for canonical_id, entry in glossary.items():
        for key in (entry["en"]["term"], entry["ar"]["term"], canonical_id):
            lookup[_normalize(key)] = canonical_id
        for alias in entry.get("aliases", []):
            lookup[_normalize(alias)] = canonical_id
    return lookup


def _normalize(s: str) -> str:
    """Lowercase, collapse whitespace, strip diacritics that don't affect meaning."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    # Strip Arabic diacritics (tashkeel) so "تعويض" matches "تَعْوِيض"
    s = re.sub(r"[ً-ْٰ]", "", s)
    return s


def lookup_definition(term: str) -> dict[str, Any]:
    """Pure-Python entrypoint (used by router heuristics + as the tool body)."""
    canonical_id = _build_lookup().get(_normalize(term))
    if canonical_id is None:
        return {
            "found": False,
            "queried_term": term,
            "message": f"'{term}' is not in the curated legal glossary.",
            "available_terms": [
                {"en": e["en"]["term"], "ar": e["ar"]["term"]}
                for e in _load_glossary().values()
            ],
        }
    entry = _load_glossary()[canonical_id]
    return {
        "found": True,
        "queried_term": term,
        "canonical_id": canonical_id,
        "en": entry["en"],
        "ar": entry["ar"],
        "aliases": entry.get("aliases", []),
    }


@tool(
    "get_legal_definition",
    (
        "Return the canonical bilingual (Arabic + English) definition of a legal "
        "term from the curated Egyptian civil-law glossary. Use this when the user "
        "asks 'what is X?' / 'ما معنى X؟' for a single legal term. Do NOT use this "
        "for full statute articles — use check_statute_reference for that. "
        "Returns {found: false, ...} when the term is not in the glossary; honor "
        "that and tell the user instead of inventing a definition."
    ),
    {"term": str},
)
async def get_legal_definition(args: dict[str, Any]) -> dict[str, Any]:
    """MCP tool wrapper around `lookup_definition`."""
    result = lookup_definition(args["term"])
    return {
        "content": [
            {"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}
        ]
    }


def is_definition_query(query: str) -> tuple[bool, str | None]:
    """Cheap rule used by the router: does this query look like a single-term
    definition lookup AND does that term exist in the glossary? Returns
    (True, canonical_id) if yes, (False, None) otherwise.

    This is a heuristic — the orchestrator can override it.
    """
    q = query.strip().lower()
    triggers = [
        r"^what (?:is|does|are)\s+",
        r"^define\s+",
        r"^meaning of\s+",
        r"^ما (?:هو|هي|معنى)\s+",
        r"^تعريف\s+",
    ]
    if not any(re.match(p, q) for p in triggers):
        return False, None
    for variant, canonical_id in _build_lookup().items():
        if variant in q:
            return True, canonical_id
    return False, None