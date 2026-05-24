"""LangGraph version of the orchestrator (Epic 7 task 7.3, alternative impl).

Same external shape as `orchestrator.run_orchestrated` (in/out signature) so
this is a drop-in for the eval harness. The internal flow is built as a
stateful LangGraph instead of a sequence of if/else dispatches, and the
router is now an LLM call (see `tools/llm_router.py`) rather than rules.

  START
    │
    ▼
  safety  ─── refuse ──→ refusal ─→ END
    │
    │ allow
    ▼
  router  ─── simple ──→ glossary ─→ END
    │
    │ medium
    ▼
  researcher ─→ explainer ─→ END
    │  (or)
    │ complex
    ▼
  comparator ─→ explainer ─→ END

The LangGraph compile() step builds an async-callable app whose state is a
TypedDict updated by each node. `run_orchestrated_lg(query)` wraps invocation
and returns the same OrchestratorResult dataclass as the legacy orchestrator.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from legal_explainer.agents.flow import FlowLogger
from legal_explainer.agents.orchestrator import (
    OrchestratorResult,
    TraceLogger,
    _format_glossary_simple,
)
from legal_explainer.agents.safety import check_safety
from legal_explainer.agents.subagents.comparator import run_comparator
from legal_explainer.agents.subagents.explainer import run_explainer
from legal_explainer.agents.subagents.researcher import run_researcher
from legal_explainer.agents.tools.glossary import lookup_definition
from legal_explainer.agents.tools.llm_router import classify as llm_classify

_DISCLAIMER_SHORT_EN = (
    "DISCLAIMER: General information about the Egyptian Civil Code, "
    "not legal advice."
)


# ── State schema ─────────────────────────────────────────────────────────────
class GraphState(TypedDict, total=False):
    user_query: str
    query_id: str
    # filled by nodes:
    safety_verdict: str
    safety_reason: str | None
    safety_response: str | None
    complexity: str
    glossary_term_id: str | None
    router_reason: str
    findings: dict[str, Any]
    answer: str
    path: str
    total_cost_usd: float
    started_at: float
    # tracing handles — carried through the graph but not part of the answer
    trace: TraceLogger
    flow: FlowLogger


# ── Nodes ────────────────────────────────────────────────────────────────────


async def safety_node(state: GraphState) -> dict[str, Any]:
    verdict = check_safety(state["user_query"])
    state["trace"].log("safety", verdict=verdict.verdict, reason=verdict.reason)
    state["flow"].safety(verdict.verdict, verdict.reason)
    return {
        "safety_verdict": verdict.verdict,
        "safety_reason": verdict.reason,
        "safety_response": verdict.suggested_response,
    }


async def refusal_node(state: GraphState) -> dict[str, Any]:
    answer = state.get("safety_response") or _DISCLAIMER_SHORT_EN
    return {"answer": answer, "path": "refused"}


async def router_node(state: GraphState) -> dict[str, Any]:
    decision = await llm_classify(state["user_query"])
    state["trace"].log(
        "route",
        complexity=decision.complexity,
        rule_matched=None,  # LLM router has no rules
        used_llm=True,
        reason=decision.reason,
        glossary_term_id=decision.glossary_term_id,
    )
    state["flow"].route(decision.complexity, rule=None, used_llm=True)
    return {
        "complexity": decision.complexity,
        "glossary_term_id": decision.glossary_term_id,
        "router_reason": decision.reason,
    }


async def glossary_node(state: GraphState) -> dict[str, Any]:
    term_id = state.get("glossary_term_id") or ""
    lookup = lookup_definition(term_id)
    state["trace"].log("tool_call", tool="get_legal_definition", args={"term": term_id})
    state["flow"].tool_call("get_legal_definition", {"term": term_id})
    answer = _format_glossary_simple(state["user_query"], lookup)
    return {"answer": answer, "path": "simple"}


async def researcher_node(state: GraphState) -> dict[str, Any]:
    state["flow"].subagent_start("researcher")
    findings = await run_researcher(state["user_query"], flow=state["flow"])
    state["trace"].log(
        "subagent",
        agent="researcher",
        cost_usd=findings.cost_usd,
        duration_ms=findings.duration_ms,
    )
    state["flow"].subagent_done(
        "researcher",
        findings.duration_ms,
        findings.cost_usd,
        f"{len(findings.key_passages)} passages, {len(findings.key_facts)} facts",
    )
    return {
        "findings": {
            "key_passages": findings.key_passages,
            "key_facts": findings.key_facts,
            "ambiguities": findings.ambiguities,
            "language_of_query": findings.language_of_query,
        },
        "total_cost_usd": state.get("total_cost_usd", 0.0) + findings.cost_usd,
    }


async def comparator_node(state: GraphState) -> dict[str, Any]:
    state["flow"].subagent_start("comparator")
    comparison = await run_comparator(state["user_query"], flow=state["flow"])
    state["trace"].log(
        "subagent",
        agent="comparator",
        cost_usd=comparison.cost_usd,
        duration_ms=comparison.duration_ms,
    )
    state["flow"].subagent_done("comparator", comparison.duration_ms, comparison.cost_usd)
    return {
        "findings": comparison.as_dict(),
        "total_cost_usd": state.get("total_cost_usd", 0.0) + comparison.cost_usd,
    }


async def explainer_node(state: GraphState) -> dict[str, Any]:
    state["flow"].subagent_start("explainer")
    explanation = await run_explainer(
        state["user_query"], state.get("findings", {}), flow=state["flow"]
    )
    state["trace"].log(
        "subagent",
        agent="explainer",
        cost_usd=explanation.cost_usd,
        duration_ms=explanation.duration_ms,
    )
    state["flow"].subagent_done("explainer", explanation.duration_ms, explanation.cost_usd)
    return {
        "answer": explanation.text,
        "path": state.get("complexity", "medium"),
        "total_cost_usd": state.get("total_cost_usd", 0.0) + explanation.cost_usd,
    }


# ── Edge predicates ──────────────────────────────────────────────────────────


def _route_after_safety(state: GraphState) -> str:
    return "refused" if state["safety_verdict"] == "refuse" else "ok"


def _route_after_router(state: GraphState) -> str:
    return state.get("complexity", "medium")


# ── Build & compile the graph ────────────────────────────────────────────────


def _build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("safety", safety_node)
    graph.add_node("refusal", refusal_node)
    graph.add_node("router", router_node)
    graph.add_node("glossary", glossary_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("comparator", comparator_node)
    graph.add_node("explainer", explainer_node)

    graph.add_edge(START, "safety")
    graph.add_conditional_edges(
        "safety", _route_after_safety, {"refused": "refusal", "ok": "router"}
    )
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {"simple": "glossary", "medium": "researcher", "complex": "comparator"},
    )
    graph.add_edge("researcher", "explainer")
    graph.add_edge("comparator", "explainer")
    graph.add_edge("glossary", END)
    graph.add_edge("explainer", END)
    graph.add_edge("refusal", END)

    return graph.compile()


_app = None


def get_app():
    """Lazy-compile the graph once per process."""
    global _app
    if _app is None:
        _app = _build_graph()
    return _app


# ── Public entry point ───────────────────────────────────────────────────────


async def run_orchestrated_lg(user_query: str) -> OrchestratorResult:
    """LangGraph drop-in for `orchestrator.run_orchestrated`. Same return type."""
    query_id = uuid.uuid4().hex[:12]
    trace = TraceLogger(query_id)
    flow = FlowLogger(query_id)
    t_start = time.time()

    flow.query(user_query)

    initial: GraphState = {
        "user_query": user_query,
        "query_id": query_id,
        "total_cost_usd": 0.0,
        "started_at": t_start,
        "trace": trace,
        "flow": flow,
    }

    app = get_app()
    final_state: GraphState = await app.ainvoke(initial)

    path = final_state.get("path", "medium")
    total_cost = final_state.get("total_cost_usd", 0.0)
    flow.done(path, total_cost)

    return OrchestratorResult(
        query_id=query_id,
        user_query=user_query,
        answer=final_state.get("answer", ""),
        path=path,
        complexity=final_state.get("complexity"),
        safety_verdict=final_state.get("safety_verdict", "not_checked"),
        duration_ms=int((time.time() - t_start) * 1000),
        total_cost_usd=total_cost,
        steps=trace.events,
    )