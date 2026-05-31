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
from claude_agent_sdk.types import ToolResultBlock, ToolUseBlock, UserMessage

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


def _summarize_tool_result(content) -> str:
    """Compact a ToolResultBlock's content into a one-line summary for the
    flow trace. Tool results can be huge (full chunks, full article text);
    this picks out the most informative slice."""
    if content is None:
        return "(empty)"
    # MCP envelope: list of {type: text, text: "..."} dicts
    if isinstance(content, list):
        text_parts = [
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        text = "\n".join(text_parts)
    else:
        text = str(content)

    text = text.strip()
    # Try to surface structured fields from our tool envelopes ({found, ok, ...}).
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            if parsed.get("found") is True and "article_number" in parsed:
                t = parsed.get("text", "") or ""
                return f"found Article {parsed['article_number']}: {t[:160]}…"
            if parsed.get("found") is False:
                return f"not found ({parsed.get('reason', '?')})"
            if parsed.get("ok") is True:
                ents = len(parsed.get("entities", []))
                rels = len(parsed.get("relationships", []))
                chunks = len(parsed.get("chunks", []))
                return f"ok — {ents} entities, {rels} relations, {chunks} chunks"
            if parsed.get("ok") is False:
                return f"no match ({parsed.get('message', '?')})"
            if parsed.get("canonical_id"):
                return f"glossary hit: {parsed['canonical_id']}"
    except (ValueError, json.JSONDecodeError):
        pass
    return text[:200] + ("…" if len(text) > 200 else "")


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
        elif isinstance(message, UserMessage) and flow is not None:
            # Tool results come back to the model wrapped in a UserMessage
            # whose content is a list of ToolResultBlock(s).
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

    return ResearcherFindings(
        key_passages=parsed.get("key_passages", []),
        key_facts=parsed.get("key_facts", []),
        ambiguities=parsed.get("ambiguities", []),
        language_of_query=parsed.get("language_of_query", "en"),
        raw=raw,
        cost_usd=cost,
        duration_ms=duration,
    )