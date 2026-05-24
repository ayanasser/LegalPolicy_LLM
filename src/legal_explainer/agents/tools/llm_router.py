"""LLM-based query classifier — replaces the rule-based router with a single
Claude call. Pure prompt + JSON parse, no heuristics.

Returns the same `RouterDecision` shape as `tools/router.classify_complexity`
so it's a drop-in for the orchestrator: complexity (simple|medium|complex)
plus, when complexity=simple, the canonical glossary term the user is asking
about (so the orchestrator's glossary shortcut still works).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    query,
)

from legal_explainer.agents.config import ROUTER_MODEL, SDK_ENV
from legal_explainer.agents.tools.glossary import _load_glossary

Complexity = Literal["simple", "medium", "complex"]


@dataclass
class LLMRouterDecision:
    complexity: Complexity
    glossary_term_id: str | None
    reason: str
    raw: str


def _build_system_prompt() -> str:
    """Embed the glossary id list so the LLM knows what 'simple' means.

    Generated once at module import — terms are static."""
    glossary_ids = list(_load_glossary().keys())
    return (
        "You classify a legal-explainer query into exactly one bucket. "
        "Respond with a single JSON object — no prose, no markdown fence.\n\n"
        "Buckets:\n"
        "  - simple:  the user is asking for the definition of ONE specific "
        "legal term, AND that term maps to one of the canonical ids below.\n"
        "  - complex: the user is asking to COMPARE / CONTRAST two or more "
        "things (articles, concepts, doctrines). Look for words like compare, "
        "contrast, difference, vs, versus — or Arabic equivalents like قارن, "
        "ما الفرق, اشرح الفرق.\n"
        "  - medium:  everything else (single-article walkthrough, topical "
        "question, refusal-style request, etc.).\n\n"
        f"Canonical glossary ids: {glossary_ids}\n\n"
        "Output schema:\n"
        '{"complexity": "simple|medium|complex", '
        '"glossary_term_id": "<one id from the list, only if complexity is simple, '
        'else null>", '
        '"reason": "<one short sentence>"}\n\n'
        "If a term sounds like a glossary entry but isn't in the list "
        "verbatim (e.g. user asks about 'tort' which isn't there), classify "
        "as medium — do NOT set glossary_term_id to a value not in the list. "
        "Reply with the JSON only."
    )


_SYSTEM = _build_system_prompt()


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of the model output. Tolerates fenced
    code blocks or leading prose if the model misbehaves."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    blob = m.group(0) if m else text
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return {}


async def classify(user_query: str) -> LLMRouterDecision:
    """One Claude call → structured classification. No fallback heuristics."""
    options = ClaudeAgentOptions(
        system_prompt=_SYSTEM,
        allowed_tools=[],
        model=ROUTER_MODEL,
        env=SDK_ENV,
    )

    text_parts: list[str] = []
    async for message in query(prompt=user_query, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)

    raw = "".join(text_parts).strip()
    parsed = _extract_json(raw)

    complexity_raw = (parsed.get("complexity") or "").strip().lower()
    if complexity_raw not in ("simple", "medium", "complex"):
        complexity_raw = "medium"  # safe default on malformed output

    term_id = parsed.get("glossary_term_id")
    # Defensive: only honor term_id if it's actually in the glossary AND we
    # said simple. Prevents an LLM hallucination from breaking the dispatch.
    if complexity_raw == "simple":
        if not term_id or term_id not in _load_glossary():
            # Said simple but no valid term — downgrade to medium so we
            # don't try to look up a fake glossary id.
            complexity_raw = "medium"
            term_id = None
    else:
        term_id = None

    return LLMRouterDecision(
        complexity=complexity_raw,  # type: ignore[arg-type]
        glossary_term_id=term_id,
        reason=parsed.get("reason", "")[:200] or "(no reason returned)",
        raw=raw,
    )