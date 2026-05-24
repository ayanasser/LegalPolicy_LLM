"""Explainer subagent — turns Researcher / Comparator findings into prose.

No tool access. Grounds strictly on the JSON it receives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

from legal_explainer.agents.config import EXPLAINER_MODEL, PROMPTS_DIR, SDK_ENV
from legal_explainer.agents.flow import FlowLogger

_SYSTEM = (PROMPTS_DIR / "explainer.md").read_text(encoding="utf-8")


@dataclass
class ExplainerAnswer:
    text: str
    cost_usd: float
    duration_ms: int


def _build_user_prompt(user_query: str, findings: dict[str, Any]) -> str:
    """Pack the original query + Researcher JSON into a single user message."""
    return (
        f"USER QUESTION:\n{user_query}\n\n"
        f"RESEARCHER FINDINGS (JSON):\n"
        f"```json\n{json.dumps(findings, ensure_ascii=False, indent=2)}\n```\n\n"
        "Write the final user-facing answer using only the findings above. "
        "Follow your system instructions exactly — language match, structure, "
        "citations, disclaimer."
    )


async def run_explainer(
    user_query: str,
    findings: dict[str, Any],
    flow: FlowLogger | None = None,
) -> ExplainerAnswer:
    options = ClaudeAgentOptions(
        system_prompt=_SYSTEM,
        allowed_tools=[],
        model=EXPLAINER_MODEL,
        env=SDK_ENV,
    )

    prompt = _build_user_prompt(user_query, findings)
    text_parts: list[str] = []
    cost, duration = 0.0, 0
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
        elif isinstance(message, ResultMessage):
            cost = getattr(message, "total_cost_usd", None) or 0.0
            duration = getattr(message, "duration_ms", 0)

    return ExplainerAnswer(
        text="\n".join(text_parts).strip(),
        cost_usd=cost,
        duration_ms=duration,
    )