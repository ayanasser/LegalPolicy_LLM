"""Statute lookup tool — deterministic article-keyed index over EgyptianLaw.pdf.

Parses citations like "Article 89", "المادة 89", or "Egyptian Civil Code
Article 89" and returns the verbatim text (Arabic + English) of that article.
Falls back to a structured 'not found' instead of inventing content.

The article index is built once by `python -m legal_explainer.agents.tools.statute build`
and persisted to data/articles_index.json.
"""

from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from typing import Any

from claude_agent_sdk import tool

from legal_explainer.agents.config import ARTICLES_INDEX_PATH, PDF_PATH


# Matches: "Article 89", "article 89", "Art. 89", "المادة 89", "مادة 89"
# Captures the integer article number.
_REF_RE = re.compile(
    r"(?ix)"
    r"(?:article|art\.?|المادة|مادة)\s*"
    r"(?:no\.?\s*)?"
    r"(\d+)"
)


def parse_reference(ref: str) -> int | None:
    """Pull an article number out of free-text. None if no match."""
    m = _REF_RE.search(ref)
    return int(m.group(1)) if m else None


@lru_cache(maxsize=1)
def _load_index() -> dict[str, str]:
    """Article index keyed by stringified number → article text (AR + EN concatenated)."""
    if not ARTICLES_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Article index missing at {ARTICLES_INDEX_PATH}. Build it with: "
            "`python -m legal_explainer.agents.tools.statute build`"
        )
    with open(ARTICLES_INDEX_PATH, encoding="utf-8") as f:
        return json.load(f)


def lookup_article(ref: str) -> dict[str, Any]:
    """Pure-Python entrypoint."""
    article_num = parse_reference(ref)
    if article_num is None:
        return {
            "found": False,
            "queried_reference": ref,
            "reason": "could_not_parse",
            "message": (
                "Could not extract an article number from the reference. Try "
                "formats like 'Article 89' or 'المادة 89'."
            ),
        }

    index = _load_index()
    text = index.get(str(article_num))
    if text is None:
        return {
            "found": False,
            "queried_reference": ref,
            "parsed_article_number": article_num,
            "reason": "not_in_index",
            "message": (
                f"Article {article_num} is not present in the Egyptian Civil "
                "Code index built from EgyptianLaw.pdf."
            ),
        }

    return {
        "found": True,
        "queried_reference": ref,
        "article_number": article_num,
        "text": text,
        "source": "EgyptianLaw.pdf",
    }


@tool(
    "check_statute_reference",
    (
        "Return the verbatim text of a specific article of the Egyptian Civil "
        "Code by citation (e.g. 'Article 89', 'المادة 89'). Use this when the "
        "user names an article number explicitly. Do NOT use this for "
        "open-ended legal questions — use search_legal_documents instead. "
        "Returns {found: false, ...} when the article isn't in the corpus; "
        "honor that rather than fabricating content."
    ),
    {"statute_reference": str},
)
async def check_statute_reference(args: dict[str, Any]) -> dict[str, Any]:
    """MCP tool wrapper."""
    result = lookup_article(args["statute_reference"])
    return {
        "content": [
            {"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}
        ]
    }


# ── Index builder ────────────────────────────────────────────────────────────
# Run once: `python -m legal_explainer.agents.tools.statute build`


def build_index() -> dict[str, str]:
    """Parse EgyptianLaw.pdf into an article-keyed index.

    Strategy: extract full text page-by-page, then split on article markers
    (English "Article N" or Arabic "المادة N"). Both AR and EN versions of
    each article are kept under the same key — the parser concatenates whatever
    text falls between consecutive markers.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(str(PDF_PATH))
    full_text = "\n\n".join(p.get_text("text") for p in doc if p.get_text("text").strip())
    doc.close()

    # Split on article markers, capturing the article number.
    parts = re.split(_REF_RE, full_text)
    # Result: [pre-amble, num1, text1, num2, text2, ...]
    index: dict[str, str] = {}
    for i in range(1, len(parts), 2):
        num = parts[i].strip()
        text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if not text:
            continue
        # Trim very long tails (next article boundary is usually close, but
        # if the regex missed a marker we don't want a giant blob).
        text = text[:8000]
        # If we hit the same article number twice (AR + EN copies), concatenate.
        if num in index:
            index[num] = index[num] + "\n\n---\n\n" + text
        else:
            index[num] = text

    ARTICLES_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ARTICLES_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    return index


def _cli() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        idx = build_index()
        print(f"Indexed {len(idx)} articles → {ARTICLES_INDEX_PATH}")
    else:
        print("Usage: python -m legal_explainer.agents.tools.statute build")
        sys.exit(1)


if __name__ == "__main__":
    _cli()