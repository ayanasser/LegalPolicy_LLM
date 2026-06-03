"""Unified Gradio UI — one chat, every project.

A single chat interface with a dropdown to pick which project answers:

  • Baseline · Qwen2.5-3B-Instruct (no finetune)   ← default
  • Baseline · Llama-3.2-3B (Ollama)
  • Finetuned · Qwen2.5-3B Knowledge adapter
  • Prompt Design · Llama-3.2-3B + legal prompt
  • Neo4j Graph RAG            → shows a retrieval panel
  • Bilingual RAG (Chroma)     → shows a retrieval panel
  • Multi-Agent · LangGraph    → shows an agent-trace panel

Run from project root (legalpolicy env):
    PYTHONPATH=src python -m apps.unified_ui.app

The RAG backends call their FastAPI services over HTTP (start them separately —
see RUN.md). Everything loads lazily, so backends you don't pick cost nothing.
"""
from __future__ import annotations

import os

import gradio as gr
import pandas as pd

from . import config
from .backends import REGISTRY, DEFAULT_LABEL
from .suggestions import load_suggestions

try:
    # Langfuse tracing (MLOps observability). No-ops if Langfuse isn't configured.
    from legal_explainer.observability.langfuse_tracing import trace_question
except Exception:  # pragma: no cover - keep the UI working even if src/ isn't on path
    import contextlib

    @contextlib.contextmanager
    def trace_question(*a, **k):
        class _N:
            def set_output(self, *a, **k):
                pass
        yield _N()

_RETRIEVAL_HEADERS = ["article", "score", "lang", "snippet"]


def _rows_to_df(rows: list[dict] | None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_RETRIEVAL_HEADERS)
    return pd.DataFrame([{h: r.get(h) for h in _RETRIEVAL_HEADERS} for r in rows])


def _to_text(content) -> str:
    """Gradio 6 Chatbot history can carry `content` as a list of parts
    (e.g. [{'type': 'text', 'text': '…'}]). Flatten to a plain string so
    Ollama / chat-template backends receive valid text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        ).strip()
    return "" if content is None else str(content)


def _clean_history(history) -> list[dict]:
    out = []
    for h in (history or []):
        role = h.get("role") if isinstance(h, dict) else None
        if role in ("user", "assistant"):
            out.append({"role": role, "content": _to_text(h.get("content"))})
    return out


def _panels_for(label: str):
    """(retrieval_visible, trace_visible, description_md) for a backend label."""
    b = REGISTRY[label]
    return (
        gr.update(visible=b.kind == "rag"),
        gr.update(visible=b.kind == "agent"),
        f"**{b.label}** — {b.description}",
    )


def build_ui() -> gr.Blocks:
    groups = load_suggestions()

    with gr.Blocks(title="Egyptian Civil Code — Unified Legal Assistant") as demo:
        gr.HTML(
            "<div style='background:linear-gradient(90deg,#1e3a8a,#2563eb,#38bdf8);"
            "color:#fff;padding:18px 22px;border-radius:12px;margin-bottom:10px'>"
            "<h1 style='margin:0;color:#fff'>⚖️ Egyptian Civil Code — Unified Legal Assistant</h1>"
            "<p style='margin:6px 0 0;color:#dbeafe'>One chat · pick a project from the dropdown · "
            "EN / العربية</p></div>"
        )

        with gr.Row():
            # Small control column on the LEFT of the chat.
            with gr.Column(scale=1, min_width=190):
                backend_dd = gr.Dropdown(
                    choices=list(REGISTRY.keys()), value=DEFAULT_LABEL,
                    label="🧩 Project", container=True, filterable=False,
                )
                clear_btn = gr.Button("🧹 Clear", size="sm")
                desc_md = gr.Markdown(_panels_for(DEFAULT_LABEL)[2])

            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=480)
                msg = gr.Textbox(
                    placeholder="Ask about the Egyptian Civil Code (English or العربية)…",
                    submit_btn=True, autofocus=True,
                )
                meta_md = gr.Markdown("")

            with gr.Column(scale=2):
                # Retrieval panel — RAG backends only.
                with gr.Group(visible=False) as retrieval_group:
                    gr.Markdown("### 🔎 Retrieval")
                    retrieval_df = gr.Dataframe(
                        headers=_RETRIEVAL_HEADERS, wrap=True, interactive=False,
                    )
                # Agent-trace panel — multi-agent only.
                with gr.Group(visible=False) as trace_group:
                    gr.Markdown("### 💭 Agent trace")
                    trace_md = gr.Markdown("_(the orchestrator's steps appear here)_")

        with gr.Accordion("💡 Suggested questions", open=True):
            for title, qs in groups.items():
                if qs:
                    gr.Markdown(f"**{title}**")
                    gr.Examples(examples=[[q] for q in qs], inputs=[msg], examples_per_page=8)

        # ── Handlers ──────────────────────────────────────────────────────────
        def on_submit(message, history, backend_label):
            history = _clean_history(history)
            message = _to_text(message).strip()
            if not message:
                return history, gr.update(), gr.update(), "", ""
            backend = REGISTRY[backend_label]
            # Trace each question to Langfuse; trace_name == the project (backend) label.
            with trace_question(backend.label, message, backend_id=backend.id) as _tr:
                reply = backend.generate(message, history)
                _tr.set_output(
                    reply.text,
                    kind=backend.kind,
                    status=reply.meta,
                    n_retrieved=(len(reply.retrieval) if reply.retrieval else 0),
                )
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply.text},
            ]
            df_update = (gr.update(value=_rows_to_df(reply.retrieval))
                         if reply.retrieval is not None else gr.update())
            trace_update = gr.update(value=reply.trace) if reply.trace is not None else gr.update()
            return history, df_update, trace_update, reply.meta, ""

        def on_backend_change(label):
            r_vis, t_vis, desc = _panels_for(label)
            return r_vis, t_vis, desc

        msg.submit(
            on_submit, [msg, chatbot, backend_dd],
            [chatbot, retrieval_df, trace_md, meta_md, msg],
        )
        backend_dd.change(
            on_backend_change, [backend_dd],
            [retrieval_group, trace_group, desc_md],
        )
        clear_btn.click(lambda: ([], "", ""), None, [chatbot, meta_md, msg])

    return demo


def main() -> None:
    demo = build_ui()
    demo.queue()  # serialise — only one model copy fits in 6 GB VRAM
    demo.launch(
        theme=gr.themes.Soft(primary_hue="blue"),
        server_name="0.0.0.0",
        server_port=int(os.environ.get("LP_UI_PORT", str(config.UI_PORT))),
        share=config.UI_SHARE,
        show_error=True,
    )


if __name__ == "__main__":
    main()
