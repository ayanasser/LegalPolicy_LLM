"""Researcher subagent — retrieves + extracts; never synthesizes.

Output is a structured JSON blob (key_passages, key_facts, ambiguities) that
the Explainer consumes downstream.
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
from claude_agent_sdk.types import ToolUseBlock

from legal_explainer.agents.config import (
    PROMPTS_DIR,
    RESEARCHER_MODEL,
    SDK_ENV,
)
from legal_explainer.agents.flow import FlowLogger
from legal_explainer.agents.tools.mcp_server import (
    TOOL_CHECK_STATUTE,
    TOOL_GET_DEFINITION,
    TOOL_SEARCH_DOCS,
    TOOL_WEB_SEARCH,
    legal_tools_server,
)

_SYSTEM = (PROMPTS_DIR / "researcher.md").read_text(encoding="utf-8")


@dataclass
class ResearcherFindings:
    """Parsed JSON output. `raw` keeps the original text for traceability."""

    key_passages: list[dict[str, Any]] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    language_of_query: str = "en"
    raw: str = ""
    cost_usd: float = 0.0
    duration_ms: int = 0


def _extract_json_block(text: str) -> dict[str, Any]:
    """Pull the first ```json fenced block out of the response.
    Falls back to parsing the whole string if no fence is found."""
    m = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    blob = (m.group(1) if m else text).strip()
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return {}


async def run_researcher(
    user_query: str, flow: FlowLogger | None = None
) -> ResearcherFindings:
    """Single-turn researcher run. The MCP server is registered so the SDK
    can dispatch tool calls; the allow-list constrains which tools are usable.

    If `flow` is provided, every tool invocation is logged to it as the
    assistant streams its responses."""
    options = ClaudeAgentOptions(
        system_prompt=_SYSTEM,
        model=RESEARCHER_MODEL,
        env=SDK_ENV,
        mcp_servers={"legal_tools": legal_tools_server},
        allowed_tools=[
            TOOL_CHECK_STATUTE,
            TOOL_GET_DEFINITION,
            TOOL_SEARCH_DOCS,
            TOOL_WEB_SEARCH,
        ],
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
        elif isinstance(message, ResultMessage):
            cost = getattr(message, "total_cost_usd", None) or 0.0
            duration = getattr(message, "duration_ms", 0)

    raw = "\n".join(text_parts)
    parsed = _extract_json_block(raw)

    return ResearcherFindings(
        key_passages=parsed.get("key_passages", []),
        key_facts=parsed.get("key_facts", []),
        ambiguities=parsed.get("ambiguities", []),
        language_of_query=parsed.get("language_of_query", "en"),
        raw=raw,
        cost_usd=cost,
        duration_ms=duration,
    )