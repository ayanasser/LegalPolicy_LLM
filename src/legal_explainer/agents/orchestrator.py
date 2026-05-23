"""The orchestrator — main entry point for Epic 7 task 7.3.

Implements the conditional-flow design:

    query
      → safety_filter()                  [no LLM]
      → complexity_router()              [rules, LLM only on ambiguity]
      ├── simple   → glossary tool → format → disclaimer       (~1 call)
      ├── medium   → Researcher → Explainer                    (~2 calls)
      └── complex  → Researcher → Explainer (long disclaimer)  (~2 calls)

A separate `run_baseline()` runs the two-role pipeline unconditionally —
that's the 7.1 baseline kept for comparison.

Every step is logged to TRACE_DIR as JSONL for auditability (task 6.3).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from legal_explainer.agents.config import TRACE_DIR
from legal_explainer.agents.flow import FlowLogger
from legal_explainer.agents.safety import check_safety
from legal_explainer.agents.subagents.comparator import run_comparator
from legal_explainer.agents.subagents.explainer import run_explainer
from legal_explainer.agents.subagents.researcher import run_researcher
from legal_explainer.agents.tools.glossary import lookup_definition
from legal_explainer.agents.tools.router import RouterDecision, classify_complexity

_DISCLAIMER_SHORT_EN = (
    "DISCLAIMER: General information about the Egyptian Civil Code, "
    "not legal advice."
)
_DISCLAIMER_SHORT_AR = (
    "تنويه: معلومات عامة عن القانون المدني المصري وليست استشارة قانونية."
)


@dataclass
class OrchestratorResult:
    query_id: str
    user_query: str
    answer: str
    path: str  # simple | medium | complex | refused | baseline
    complexity: str | None
    safety_verdict: str
    duration_ms: int
    total_cost_usd: float
    steps: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraceLogger:
    """Append-only JSONL log. One line per event, indexed by query_id."""

    def __init__(self, query_id: str):
        self.query_id = query_id
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = TRACE_DIR / f"run_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
        self.events: list[dict[str, Any]] = []

    def log(self, step: str, **payload: Any) -> None:
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "query_id": self.query_id,
            "step": step,
            **payload,
        }
        self.events.append(event)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _format_glossary_simple(user_query: str, lookup: dict[str, Any]) -> str:
    """Render a simple-path answer when the glossary tool resolved cleanly."""
    if not lookup.get("found"):
        return (
            f"I don't have a curated definition for '{lookup.get('queried_term')}'. "
            f"{_DISCLAIMER_SHORT_EN}"
        )
    en = lookup["en"]
    ar = lookup["ar"]
    # If the query was in Arabic, lead with Arabic
    is_arabic = any("؀" <= ch <= "ۿ" for ch in user_query)
    if is_arabic:
        body = f"**{ar['term']}** — {ar['definition']}\n\n_English: {en['term']} — {en['definition']}_"
        return f"{body}\n\n{_DISCLAIMER_SHORT_AR}"
    body = f"**{en['term']}** — {en['definition']}\n\n_العربية: {ar['term']} — {ar['definition']}_"
    return f"{body}\n\n{_DISCLAIMER_SHORT_EN}"


def _is_comparison_query(user_query: str, decision: RouterDecision) -> bool:
    """Determine whether to dispatch to the Comparator vs the Researcher."""
    return (
        decision.complexity == "complex"
        and decision.rule_matched == "complex_trigger"
    )


# ── Public entrypoints ───────────────────────────────────────────────────────


async def run_orchestrated(user_query: str) -> OrchestratorResult:
    """The smart path: safety → router → conditional dispatch → synthesize."""
    query_id = uuid.uuid4().hex[:12]
    trace = TraceLogger(query_id)
    flow = FlowLogger(query_id)
    t_start = time.time()
    total_cost = 0.0

    flow.query(user_query)

    # 1. Safety
    safety = check_safety(user_query)
    trace.log("safety", verdict=safety.verdict, reason=safety.reason)
    flow.safety(safety.verdict, safety.reason)
    if safety.verdict == "refuse":
        answer = safety.suggested_response or _DISCLAIMER_SHORT_EN
        flow.done("refused", 0.0)
        return OrchestratorResult(
            query_id=query_id,
            user_query=user_query,
            answer=answer,
            path="refused",
            complexity=None,
            safety_verdict=safety.verdict,
            duration_ms=int((time.time() - t_start) * 1000),
            total_cost_usd=0.0,
            steps=trace.events,
        )

    # 2. Route
    decision = await classify_complexity(user_query)
    trace.log(
        "route",
        complexity=decision.complexity,
        rule_matched=decision.rule_matched,
        used_llm=decision.used_llm,
        reason=decision.reason,
    )
    flow.route(decision.complexity, decision.rule_matched, decision.used_llm)

    # 3. Dispatch
    if decision.complexity == "simple":
        term_id = (
            decision.reason.split("'")[1]
            if decision.reason and "'" in decision.reason
            else user_query
        )
        lookup = lookup_definition(term_id)
        trace.log("tool_call", tool="get_legal_definition", args={"term": term_id})
        flow.tool_call("get_legal_definition", {"term": term_id})
        flow.tool_result(
            "get_legal_definition",
            f"found={lookup.get('found')} canonical={lookup.get('canonical_id', 'n/a')}",
        )
        answer = _format_glossary_simple(user_query, lookup)
        path = "simple"

    elif _is_comparison_query(user_query, decision):
        flow.subagent_start("comparator")
        comparison = await run_comparator(user_query, flow=flow)
        total_cost += comparison.cost_usd
        trace.log(
            "subagent",
            agent="comparator",
            cost_usd=comparison.cost_usd,
            duration_ms=comparison.duration_ms,
        )
        flow.subagent_done("comparator", comparison.duration_ms, comparison.cost_usd)

        flow.subagent_start("explainer")
        explanation = await run_explainer(user_query, comparison.as_dict(), flow=flow)
        total_cost += explanation.cost_usd
        trace.log(
            "subagent",
            agent="explainer",
            cost_usd=explanation.cost_usd,
            duration_ms=explanation.duration_ms,
        )
        flow.subagent_done("explainer", explanation.duration_ms, explanation.cost_usd)
        answer = explanation.text
        path = "complex"

    else:  # medium or non-comparison complex
        flow.subagent_start("researcher")
        findings = await run_researcher(user_query, flow=flow)
        total_cost += findings.cost_usd
        trace.log(
            "subagent",
            agent="researcher",
            cost_usd=findings.cost_usd,
            duration_ms=findings.duration_ms,
            findings_summary={
                "n_passages": len(findings.key_passages),
                "n_facts": len(findings.key_facts),
                "n_ambiguities": len(findings.ambiguities),
            },
        )
        flow.subagent_done(
            "researcher",
            findings.duration_ms,
            findings.cost_usd,
            f"{len(findings.key_passages)} passages, {len(findings.key_facts)} facts, {len(findings.ambiguities)} ambiguities",
        )

        flow.subagent_start("explainer")
        explanation = await run_explainer(
            user_query,
            {
                "key_passages": findings.key_passages,
                "key_facts": findings.key_facts,
                "ambiguities": findings.ambiguities,
                "language_of_query": findings.language_of_query,
            },
            flow=flow,
        )
        total_cost += explanation.cost_usd
        trace.log(
            "subagent",
            agent="explainer",
            cost_usd=explanation.cost_usd,
            duration_ms=explanation.duration_ms,
        )
        flow.subagent_done("explainer", explanation.duration_ms, explanation.cost_usd)
        answer = explanation.text
        path = decision.complexity

    flow.done(path, total_cost)

    return OrchestratorResult(
        query_id=query_id,
        user_query=user_query,
        answer=answer,
        path=path,
        complexity=decision.complexity,
        safety_verdict=safety.verdict,
        duration_ms=int((time.time() - t_start) * 1000),
        total_cost_usd=total_cost,
        steps=trace.events,
    )


async def run_baseline(user_query: str) -> OrchestratorResult:
    """Epic 7 task 7.1: the 'always Researcher → Explainer' baseline.

    No safety filter, no routing — every query takes the full two-role
    pipeline. Kept for the apples-to-apples comparison in task 7.5.
    """
    query_id = uuid.uuid4().hex[:12]
    trace = TraceLogger(query_id)
    flow = FlowLogger(query_id)
    t_start = time.time()
    total_cost = 0.0

    flow.query(user_query)
    flow.info("baseline pipeline — no safety/router")

    flow.subagent_start("researcher")
    findings = await run_researcher(user_query, flow=flow)
    total_cost += findings.cost_usd
    trace.log("subagent", agent="researcher", cost_usd=findings.cost_usd)
    flow.subagent_done(
        "researcher",
        findings.duration_ms,
        findings.cost_usd,
        f"{len(findings.key_passages)} passages, {len(findings.key_facts)} facts",
    )

    flow.subagent_start("explainer")
    explanation = await run_explainer(
        user_query,
        {
            "key_passages": findings.key_passages,
            "key_facts": findings.key_facts,
            "ambiguities": findings.ambiguities,
            "language_of_query": findings.language_of_query,
        },
        flow=flow,
    )
    total_cost += explanation.cost_usd
    trace.log("subagent", agent="explainer", cost_usd=explanation.cost_usd)
    flow.subagent_done("explainer", explanation.duration_ms, explanation.cost_usd)
    flow.done("baseline", total_cost)

    return OrchestratorResult(
        query_id=query_id,
        user_query=user_query,
        answer=explanation.text,
        path="baseline",
        complexity=None,
        safety_verdict="not_checked",
        duration_ms=int((time.time() - t_start) * 1000),
        total_cost_usd=total_cost,
        steps=trace.events,
    )