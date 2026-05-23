"""Query complexity router — rules first, LLM classifier as fallback.

Classifies a user query as `simple`, `medium`, or `complex` so the
orchestrator can pick a cheap path for easy questions and reserve subagent
fan-out for hard ones. Epic 7 task 7.2.

Routing rules (cheap, no LLM):
  - Definition-style queries matching a glossary term     → simple
  - Single-article references ("Article 89" / "المادة 89") → medium
  - Comparison or analysis keywords                       → complex
  - Otherwise hand off to an LLM classifier               → fallback

Returns a dataclass with the decision, the reasoning trail, and the
confidence the rules had — useful for debugging routing mistakes later.
"""

from __future__ import annotations

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
from legal_explainer.agents.tools.glossary import is_definition_query
from legal_explainer.agents.tools.statute import parse_reference

Complexity = Literal["simple", "medium", "complex"]


@dataclass
class RouterDecision:
    complexity: Complexity
    reason: str
    rule_matched: str | None
    used_llm: bool


# Keywords that hint a comparison / multi-step analysis is needed.
# Mixed EN + AR — both languages can appear in user queries.
_COMPLEX_TRIGGERS = [
    "compare", "contrast", "difference", "vs", "versus",
    "analyze", "analyse", "evaluate", "argue",
    "which is", "what are the differences",
    "قارن", "اشرح الفرق", "ما الفرق", "حلل", "ناقش",
]


def _rule_classify(query_text: str) -> RouterDecision | None:
    """Try every rule in order. Return the first match, or None if all miss."""
    q_lower = query_text.lower().strip()

    is_def, term_id = is_definition_query(query_text)
    if is_def:
        return RouterDecision(
            complexity="simple",
            reason=f"definition-style query for glossary term '{term_id}'",
            rule_matched="glossary_term",
            used_llm=False,
        )

    if any(t in q_lower for t in _COMPLEX_TRIGGERS):
        return RouterDecision(
            complexity="complex",
            reason="contains comparison/analysis keyword",
            rule_matched="complex_trigger",
            used_llm=False,
        )

    if parse_reference(query_text) is not None:
        return RouterDecision(
            complexity="medium",
            reason="explicit article reference detected",
            rule_matched="article_reference",
            used_llm=False,
        )

    return None


async def _llm_classify(query_text: str) -> RouterDecision:
    """Fallback for queries the rules can't confidently classify."""
    system = (
        "You classify legal queries into exactly one of three buckets:\n"
        "  - simple:  one-word/phrase definition lookups, glossary-style.\n"
        "  - medium:  single concept, single article, or one-step Q&A.\n"
        "  - complex: comparison, multi-step analysis, drafting, or "
        "anything needing more than one retrieval + synthesis step.\n\n"
        "Respond with EXACTLY one word: simple, medium, or complex. "
        "No punctuation, no explanation."
    )
    options = ClaudeAgentOptions(
        system_prompt=system,
        allowed_tools=[],
        model=ROUTER_MODEL,
        env=SDK_ENV,
    )
    text_parts: list[str] = []
    async for message in query(prompt=query_text, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)

    answer = "".join(text_parts).strip().lower()
    # Defensive parse — model occasionally adds punctuation.
    answer_clean = re.sub(r"[^a-z]", "", answer)
    if answer_clean.startswith("simple"):
        complexity: Complexity = "simple"
    elif answer_clean.startswith("complex"):
        complexity = "complex"
    else:
        complexity = "medium"

    return RouterDecision(
        complexity=complexity,
        reason=f"LLM classifier returned '{answer.strip() or '<empty>'}'",
        rule_matched=None,
        used_llm=True,
    )


async def classify_complexity(query_text: str) -> RouterDecision:
    """Public entrypoint: rules first, LLM fallback."""
    rule_result = _rule_classify(query_text)
    if rule_result is not None:
        return rule_result
    return await _llm_classify(query_text)