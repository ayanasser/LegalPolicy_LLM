"""Comparator subagent — multi-side retrieval + structured comparison.

Output is structured JSON that the Explainer turns into prose.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)
from claude_agent_sdk.types import ToolResultBlock, ToolUseBlock, UserMessage

from legal_explainer.agents.config import (
    COMPARATOR_MODEL,
    PROMPTS_DIR,
    SDK_ENV,
)
from legal_explainer.agents.flow import FlowLogger
from legal_explainer.agents.tools.mcp_server import (
    TOOL_CHECK_STATUTE,
    TOOL_GET_DEFINITION,
    TOOL_SEARCH_DOCS,
    legal_tools_server,
)

_SYSTEM = (PROMPTS_DIR / "comparator.md").read_text(encoding="utf-8")


@dataclass
class ComparisonFindings:
    side_a: dict[str, Any] = field(default_factory=dict)
    side_b: dict[str, Any] = field(default_factory=dict)
    shared_ground: list[str] = field(default_factory=list)
    differences: list[dict[str, str]] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    language_of_query: str = "en"
    raw: str = ""
    cost_usd: float = 0.0
    duration_ms: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "side_a": self.side_a,
            "side_b": self.side_b,
            "shared_ground": self.shared_ground,
            "differences": self.differences,
            "ambiguities": self.ambiguities,
            "language_of_query": self.language_of_query,
        }


def _extract_json_block(text: str) -> dict[str, Any]:
    m = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    blob = (m.group(1) if m else text).strip()
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return {}


async def run_comparator(
    user_query: str, flow: FlowLogger | None = None
) -> ComparisonFindings:
    options = ClaudeAgentOptions(
        system_prompt=_SYSTEM,
        model=COMPARATOR_MODEL,
        env=SDK_ENV,
        mcp_servers={"legal_tools": legal_tools_server},
        allowed_tools=[TOOL_CHECK_STATUTE, TOOL_GET_DEFINITION, TOOL_SEARCH_DOCS],
        permission_mode="bypassPermissions",
    )

    text_parts: list[str] = []
    cost, duration = 0.0, 0
    async for message in query(prompt=user_query, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
                elif isinstance(block, ToolUseBlock) and flow is not None:
                    flow.tool_call(block.name.split("__")[-1], block.input)
        elif isinstance(message, UserMessage) and flow is not None:
            from legal_explainer.agents.subagents.researcher import (
                _summarize_tool_result,
            )
            content = message.content if isinstance(message.content, list) else []
            for block in content:
                if isinstance(block, ToolResultBlock):
                    summary = _summarize_tool_result(block.content)
                    if block.is_error:
                        flow.tool_error("(tool)", summary)
                    else:
                        flow.tool_result("(tool)", summary)
        elif isinstance(message, ResultMessage):
            cost = getattr(message, "total_cost_usd", None) or 0.0
            duration = getattr(message, "duration_ms", 0)

    raw = "\n".join(text_parts)
    parsed = _extract_json_block(raw)

    return ComparisonFindings(
        side_a=parsed.get("side_a", {}),
        side_b=parsed.get("side_b", {}),
        shared_ground=parsed.get("shared_ground", []),
        differences=parsed.get("differences", []),
        ambiguities=parsed.get("ambiguities", []),
        language_of_query=parsed.get("language_of_query", "en"),
        raw=raw,
        cost_usd=cost,
        duration_ms=duration,
    )