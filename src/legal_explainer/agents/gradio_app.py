"""Gradio demo for the legal-explainer agent.

Visualises the v2 LangGraph orchestrator end-to-end:

  - User types a question (English or Arabic)
  - The orchestrator runs: safety → router → conditional dispatch → synthesis
  - The UI shows:
      1. The final answer
      2. A step-by-step flow trace (safety verdict, router decision, which
         subagents ran, every tool call with its args, durations, costs)
      3. Aggregate metrics (path taken, total cost, total wall-clock)

Run from project root:
    PYTHONPATH=src python -m legal_explainer.agents.gradio_app
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

import gradio as gr

from legal_explainer.agents.flow import FlowLogger
from legal_explainer.agents.orchestrator_langgraph import run_orchestrated_lg


# ── Flow-event rendering ─────────────────────────────────────────────────────

_STAGE_ICONS = {
    "query":          "💬",
    "safety":         "🛡️",
    "route":          "🧭",
    "subagent_start": "🤖",
    "subagent_done":  "✅",
    "tool_call":      "🔧",
    "tool_result":    "📦",
    "tool_error":     "⚠️",
    "info":           "ℹ️",
    "warn":           "⚠️",
    "done":           "🏁",
}


def _format_event(event: dict[str, Any]) -> str:
    """Render one structured flow event as a markdown bullet."""
    t = event.get("type", "?")
    icon = _STAGE_ICONS.get(t, "·")
    elapsed = event.get("elapsed_s", 0.0)
    ts_tag = f"`+{elapsed:>5.2f}s`"

    if t == "query":
        return f"{ts_tag}  {icon} **Query** — {event.get('text', '')[:200]}"

    if t == "safety":
        verdict = event.get("verdict", "?")
        reason = event.get("reason")
        suffix = f" — {reason}" if reason else ""
        return f"{ts_tag}  {icon} **Safety**: `{verdict}`{suffix}"

    if t == "route":
        cx = event.get("complexity", "?")
        rule = event.get("rule") or "LLM classifier"
        used_llm = event.get("used_llm")
        return (
            f"{ts_tag}  {icon} **Router**: complexity = `{cx}`  "
            f"(decided by: {rule}, used_llm={used_llm})"
        )

    if t == "subagent_start":
        return f"{ts_tag}  {icon} **{event.get('agent', '?').title()}** — starting…"

    if t == "subagent_done":
        agent = event.get("agent", "?")
        dur = event.get("duration_ms", 0) / 1000
        cost = event.get("cost_usd", 0.0)
        summary = event.get("summary") or ""
        suffix = f"  · {summary}" if summary else ""
        return (
            f"{ts_tag}  {icon} **{agent.title()}** done — "
            f"`{dur:.1f}s` · `${cost:.4f}`{suffix}"
        )

    if t == "tool_call":
        tool = event.get("tool", "?")
        args_json = json.dumps(event.get("args", {}), ensure_ascii=False)
        if len(args_json) > 220:
            args_json = args_json[:217] + "…"
        return f"{ts_tag}  &nbsp;&nbsp;&nbsp;&nbsp;{icon} `{tool}` — `{args_json}`"

    if t == "tool_result":
        return f"{ts_tag}  &nbsp;&nbsp;&nbsp;&nbsp;{icon} `{event.get('tool')}` → {event.get('summary', '')[:200]}"

    if t == "tool_error":
        return f"{ts_tag}  &nbsp;&nbsp;&nbsp;&nbsp;{icon} `{event.get('tool')}` **ERROR** — {event.get('error', '')[:200]}"

    if t == "done":
        path = event.get("path", "?")
        cost = event.get("total_cost_usd", 0.0)
        elapsed_total = event.get("elapsed_s", 0.0)
        return (
            f"{ts_tag}  {icon} **Done** — path=`{path}` · "
            f"total `{elapsed_total:.1f}s` · `${cost:.4f}`"
        )

    if t in ("info", "warn"):
        return f"{ts_tag}  {icon} {event.get('message', '')}"

    return f"{ts_tag}  · {json.dumps(event, ensure_ascii=False)}"


def render_flow(events: list[dict[str, Any]]) -> str:
    """Turn a flow event list into a markdown trace."""
    if not events:
        return "_No flow events yet._"
    return "\n\n".join(_format_event(e) for e in events)


def _aggregate_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Pull aggregate stats out of the flow events alone (no OrchestratorResult)."""
    tool_calls = [e for e in events if e.get("type") == "tool_call"]
    route_event = next((e for e in events if e.get("type") == "route"), None)
    safety_event = next((e for e in events if e.get("type") == "safety"), None)
    subagents = sorted({
        e["agent"]
        for e in events
        if e.get("type") in ("subagent_start", "subagent_done")
    })
    return {
        "complexity": route_event.get("complexity") if route_event else None,
        "safety_verdict": safety_event.get("verdict") if safety_event else "—",
        "subagents": subagents,
        "tools_called": sorted({e["tool"] for e in tool_calls}),
        "tool_call_count": len(tool_calls),
    }


def render_metrics(result, events: list[dict[str, Any]]) -> str:
    """Final metrics panel after the orchestrator returned an OrchestratorResult."""
    agg = _aggregate_from_events(events)
    return (
        f"**Status**: ✅ done  \n"
        f"**Path taken**: `{result.path}`  \n"
        f"**Complexity**: `{result.complexity or agg['complexity'] or 'n/a'}`  \n"
        f"**Safety verdict**: `{result.safety_verdict}`  \n"
        f"**Subagents run**: {', '.join(f'`{a}`' for a in agg['subagents']) or 'none'}  \n"
        f"**Tools called**: {', '.join(f'`{t}`' for t in agg['tools_called']) or 'none'}  \n"
        f"**Tool calls**: {agg['tool_call_count']}  \n"
        f"**Duration**: `{result.duration_ms / 1000:.1f}s`  \n"
        f"**Cost**: `${result.total_cost_usd:.4f}`  \n"
        f"**Query ID**: `{result.query_id}`"
    )


def render_partial_metrics(events: list[dict[str, Any]], elapsed_s: float) -> str:
    """Live metrics panel while the orchestrator is still running."""
    agg = _aggregate_from_events(events)
    return (
        f"**Status**: ⏳ running ({elapsed_s:.1f}s elapsed)  \n"
        f"**Path so far**: `{agg['complexity'] or '… deciding …'}`  \n"
        f"**Safety verdict**: `{agg['safety_verdict']}`  \n"
        f"**Subagents run**: {', '.join(f'`{a}`' for a in agg['subagents']) or '…'}  \n"
        f"**Tools called**: {', '.join(f'`{t}`' for t in agg['tools_called']) or '…'}  \n"
        f"**Tool calls so far**: {agg['tool_call_count']}"
    )


# ── Streaming handler ────────────────────────────────────────────────────────

# How often (seconds) to poll flow.events while the orchestrator runs.
# Smaller = snappier UI updates, larger = less Gradio re-render churn.
_POLL_INTERVAL_S = 0.4


async def run_query(user_query: str):
    """Streaming handler — runs the orchestrator in the background and yields
    `(answer, flow_md, metrics_md)` tuples as flow events arrive AND as the
    Explainer streams its answer text token-by-token.

    The orchestrator's FlowLogger has two live-updating fields:
      - `events`   → structured flow events (we render these as the "thinking")
      - `answer_text` → the Explainer's streaming text (rendered in the Answer pane)

    We poll on a short interval and yield whenever either has changed.
    """
    user_query = (user_query or "").strip()
    if not user_query:
        yield (
            "_Type a question above and click Run._",
            "_(no run yet)_",
            "_(no run yet)_",
        )
        return

    flow = FlowLogger(query_id=uuid.uuid4().hex[:12])

    t_start = time.time()
    task = asyncio.create_task(run_orchestrated_lg(user_query, flow_logger=flow))

    # Initial frame — UI flips to "running" immediately.
    yield (
        "_⏳ Routing your query…_",
        "_(no events yet)_",
        "**Status**: ⏳ running (0.0s elapsed)",
    )

    last_event_count = 0
    last_answer_len = 0
    while not task.done():
        await asyncio.sleep(_POLL_INTERVAL_S)
        events_changed = len(flow.events) != last_event_count
        answer_changed = len(flow.answer_text) != last_answer_len

        if not (events_changed or answer_changed):
            continue

        last_event_count = len(flow.events)
        last_answer_len = len(flow.answer_text)

        # During streaming: if we have answer text, show it; otherwise show a
        # "working" placeholder so the Answer pane isn't blank.
        if flow.answer_text:
            answer_display = flow.answer_text + " ▍"  # blinking-cursor sentinel
        else:
            # Render a hint about which subagent is currently active.
            active = next(
                (
                    e["agent"]
                    for e in reversed(flow.events)
                    if e.get("type") == "subagent_start"
                    and not any(
                        d.get("type") == "subagent_done" and d["agent"] == e["agent"]
                        for d in flow.events[flow.events.index(e):]
                    )
                ),
                None,
            )
            if active:
                answer_display = f"_⏳ {active.title()} subagent working…_"
            else:
                answer_display = "_⏳ Working — see the thinking panel below._"

        yield (
            answer_display,
            render_flow(flow.events),
            render_partial_metrics(flow.events, time.time() - t_start),
        )

    # Drain final result.
    try:
        result = await task
    except Exception as e:
        yield (
            f"### Error\n\n```\n{type(e).__name__}: {e}\n```",
            render_flow(flow.events) if flow.events else "_(no flow recorded)_",
            f"**Status**: ❌ failed — `{type(e).__name__}`",
        )
        return

    # Final answer — for the simple/refused paths there's no streamed chunks,
    # so prefer the result's `answer` field as the source of truth.
    final_answer = result.answer or flow.answer_text or "_(no answer text returned)_"
    yield (
        final_answer,
        render_flow(flow.events),
        render_metrics(result, flow.events),
    )


# ── UI ───────────────────────────────────────────────────────────────────────

_EXAMPLE_QUERIES = [
    "What is force majeure?",
    "Walk me through Article 713 of the Egyptian Civil Code.",
    "Compare Article 660 and Article 713 in terms of duties.",
    "ما معنى التعويض في القانون المدني المصري؟",
    "How does Egyptian law handle gifts and revocation?",
    "I was arrested last night, should I sign this contract?",
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Legal Explainer — Egyptian Civil Code Agent",
        theme=gr.themes.Soft(),
        # Theme-aware CSS: use Gradio's CSS variables so colors adapt to
        # light/dark mode automatically. Avoid hard-coded backgrounds.
        css="""
            .answer-pane {
                min-height: 200px;
                font-size: 15px;
                line-height: 1.55;
            }
            .thinking-box {
                background: var(--background-fill-secondary);
                color: var(--body-text-color);
                border: 1px solid var(--border-color-primary);
                border-radius: 8px;
                padding: 12px 14px;
            }
            .thinking-box code {
                background: var(--background-fill-primary) !important;
                color: var(--body-text-color) !important;
                padding: 1px 5px;
                border-radius: 4px;
                font-size: 0.9em;
            }
            .thinking-box p { margin: 0.35em 0; }
        """,
    ) as demo:
        gr.Markdown(
            "# 🇪🇬 Legal Explainer — Egyptian Civil Code Agent\n\n"
            "Multi-agent legal explainer built on **Claude Agent SDK + LangGraph**. "
            "Type a question — the agent routes it (simple / medium / complex), spawns "
            "the right subagents, calls tools, and streams the final answer. "
            "Open the **💭 Thinking** panel to watch every step in real time."
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                query_input = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. Walk me through Article 89 of the Egyptian Civil Code.",
                    lines=3,
                )
                run_btn = gr.Button("Run", variant="primary", size="lg")
                gr.Examples(
                    examples=[[q] for q in _EXAMPLE_QUERIES],
                    inputs=[query_input],
                    label="Try one of these",
                )

            with gr.Column(scale=2):
                gr.Markdown("### 📝 Answer")
                answer_md = gr.Markdown(
                    value="_(the agent's answer will stream here)_",
                    elem_classes=["answer-pane"],
                )

        with gr.Accordion("💭 Thinking — flow, tool calls, subagent timings", open=True):
            with gr.Row():
                with gr.Column(scale=2):
                    flow_md = gr.Markdown(
                        value="_(every safety check, routing decision, subagent start/end, and tool call will appear here as it happens)_",
                        elem_classes=["thinking-box"],
                    )
                with gr.Column(scale=1, min_width=260):
                    metrics_md = gr.Markdown(
                        value="**Status**: (idle)",
                        elem_classes=["thinking-box"],
                    )

        run_btn.click(
            fn=run_query,
            inputs=[query_input],
            outputs=[answer_md, flow_md, metrics_md],
            api_name="run_query",
        )

        # Pressing Enter in the textbox triggers Run too.
        query_input.submit(
            fn=run_query,
            inputs=[query_input],
            outputs=[answer_md, flow_md, metrics_md],
        )

        gr.Markdown(
            "---\n\n"
            "**Stack**: Claude Agent SDK · LangGraph · LightRAG (Egyptian Civil Code KG) · "
            "Ollama Qwen3-Embedding · NLTK · Gradio  \n"
            "**Engine**: v2 (LangGraph state machine + LLM-based router)  \n"
            "**Corpus**: EgyptianLaw.pdf — 2,164 articles indexed"
        )

    return demo


def main() -> None:
    demo = build_ui()
    demo.queue(max_size=16).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
