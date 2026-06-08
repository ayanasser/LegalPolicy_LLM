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
import sys
import time
from pathlib import Path

# Make `legal_explainer` importable regardless of how the app is launched.
# Without this, running `python -m apps.unified_ui.app` (instead of via
# scripts/run_unified_ui.sh, which sets PYTHONPATH=src) makes the Langfuse
# import below fail and silently disables tracing — the UI keeps working but
# no traces reach Langfuse.
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import gradio as gr
import pandas as pd

from . import config
from .backends import REGISTRY, DEFAULT_LABEL, Reply
from .suggestions import load_suggestions

try:
    # Langfuse tracing (MLOps observability). No-ops if Langfuse isn't configured.
    from legal_explainer.observability.langfuse_tracing import trace_question
except Exception as _e:  # pragma: no cover - keep the UI working even if src/ isn't on path
    import contextlib

    # Loud warning: tracing is disabled. Better than silently dropping traces.
    print(f"[langfuse] tracing DISABLED — could not import tracing helper: {_e}\n"
          f"           (expected src/ at {_SRC}). Traces will NOT reach Langfuse.")

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


def _stream_with_heartbeat(backend, message, history, hb_every: float = 1.5):
    """Wrap `backend.stream` so the FIRST token's wait never freezes the browser.

    The first message to a backend can block for minutes (a cold 4-bit Qwen load
    ~175s, a RAG service warming BGE-M3, an Ollama model swap). With no output the
    browser shows its 'page unresponsive — wait / exit' dialog. We run the real
    stream in a thread and emit a ticking '⏳ loading…' heartbeat until the first
    token arrives, which keeps the connection alive and tells the user what's up.

    Yields the same `(partial_text, reply)` tuples as `backend.stream`.
    """
    import queue as _queue
    import threading

    q: "_queue.Queue" = _queue.Queue()
    _DONE = object()

    def _producer():
        try:
            for item in backend.stream(message, history):
                q.put(("item", item))
        except Exception as e:  # surface backend errors to the main generator
            q.put(("error", e))
        finally:
            q.put(("done", _DONE))

    threading.Thread(target=_producer, daemon=True).start()

    t0 = time.perf_counter()
    got_first = False
    while True:
        try:
            kind, payload = q.get(timeout=hb_every)
        except _queue.Empty:
            if not got_first:  # still waiting on the first token — beat the clock
                secs = int(time.perf_counter() - t0)
                yield (f"⏳ Loading the model and preparing the answer… ({secs}s)\n\n"
                       f"_First use of a project can take 2–3 minutes on a 6 GB GPU "
                       f"(model load). Subsequent questions are fast._", None)
            continue
        if kind == "done":
            return
        if kind == "error":
            raise payload
        got_first = True
        yield payload


def build_ui() -> gr.Blocks:
    groups = load_suggestions()

    with gr.Blocks(title="Egyptian Civil Code — Unified Legal Assistant",
                   theme=gr.themes.Soft(primary_hue="blue")) as demo:
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
                chatbot = gr.Chatbot(height=480, type="messages")
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
            """Streaming handler — a generator that yields partial answers as the
            model produces tokens, then a final update carrying retrieval/trace/
            metadata. Backends that can't stream yield once (see Backend.stream)."""
            history = _clean_history(history)
            message = _to_text(message).strip()
            if not message:
                yield history, gr.update(), gr.update(), "", ""
                return
            backend = REGISTRY[backend_label]
            # Display history = prior turns + this user turn + a placeholder
            # assistant turn we fill in as tokens stream.
            disp = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": ""},
            ]
            # ── Stream tokens (NO Langfuse context open here) ──────────────────
            # The trace span uses OpenTelemetry contextvars, which can't survive
            # being suspended/resumed across `yield` (Gradio resumes the generator
            # in a different context per token → "Token created in a different
            # Context"). So we stream first and record the trace afterwards, in a
            # single synchronous block with no yields inside it.
            final = None
            df_update = gr.update()      # retrieval panel — updated the moment a
            trace_update = gr.update()   # reply carries retrieval / agent trace,
            status = gr.update()         # so it shows WHILE tokens still stream.
            t0 = time.perf_counter()
            for partial, reply in _stream_with_heartbeat(backend, message, history):
                disp[-1]["content"] = partial or ""
                if reply is not None:
                    final = reply
                    if reply.retrieval is not None:
                        df_update = gr.update(value=_rows_to_df(reply.retrieval))
                    if reply.trace is not None:
                        trace_update = gr.update(value=reply.trace)
                    if reply.meta:
                        status = reply.meta
                yield disp, df_update, trace_update, status, ""
            if final is None:  # safety net — backend yielded no final reply
                final = Reply(text=disp[-1]["content"])
            elapsed_ms = round((time.perf_counter() - t0) * 1000)

            # ── Record the trace (synchronous, no yields) ──────────────────────
            try:
                with trace_question(backend.label, message, backend_id=backend.id) as _tr:
                    # SINGLE dict (dict-merge so a stray "model" key in info can't
                    # collide with the explicit model=).
                    payload = dict(final.info or {})
                    retrieval_meta = payload.pop("retrieval", None)
                    payload.setdefault("elapsed_ms", elapsed_ms)
                    payload.update(
                        model=final.model,
                        model_parameters=final.params,
                        usage=final.usage,
                        documents=final.documents,
                        retrieval_meta=retrieval_meta,
                        prompt=final.prompt,
                        kind=backend.kind,
                        status=final.meta,
                        n_retrieved=(len(final.retrieval) if final.retrieval else 0),
                    )
                    _tr.set_output(final.text, **payload)
            except Exception as e:  # never let tracing break the chat
                print(f"[unified-ui] tracing error (ignored): {e}")

            # Final update — ensure the panels reflect the completed reply.
            disp[-1]["content"] = final.text or disp[-1]["content"]
            if final.retrieval is not None:
                df_update = gr.update(value=_rows_to_df(final.retrieval))
            if final.trace is not None:
                trace_update = gr.update(value=final.trace)
            yield disp, df_update, trace_update, (final.meta or status), ""

        def on_backend_change(label):
            # Keep only ONE model on the 6 GB GPU. When the user picks a project,
            # free whatever the new one won't use so the next load can't OOM:
            #   • picking a local-HF project → unload any resident Ollama model
            #   • picking anything else      → unload the in-process 4-bit Qwen
            # (baseline ⇄ finetuned both use the shared HF load, so neither evicts it.)
            try:
                from .backends import LocalQwenBackend, _free_ollama_vram, _QWEN
                if isinstance(REGISTRY[label], LocalQwenBackend):
                    _free_ollama_vram()
                elif _QWEN.is_loaded():
                    _QWEN.unload()
            except Exception as e:  # never let VRAM housekeeping break the UI
                print(f"[unified-ui] VRAM eviction on switch failed (ignored): {e}")
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
    if config.PRELOAD_LOCAL_MODEL:
        # Pay the ~175s 4-bit load + first-call warm-up now, at startup, so the
        # first baseline/finetuned message is fast. Never block the UI on it.
        from .backends import _QWEN
        print("[unified-ui] LP_PRELOAD=1 — loading + warming the local Qwen "
              "(baseline + finetuned). This takes a few minutes; the UI will "
              "start serving once it's done …", flush=True)
        try:
            import time as _t
            _t0 = _t.perf_counter()
            _QWEN.warmup()
            print(f"[unified-ui] local model ready in {_t.perf_counter() - _t0:.0f}s.",
                  flush=True)
        except Exception as e:  # don't let a preload failure block the UI
            print(f"[unified-ui] preload failed (continuing without it): {e}",
                  flush=True)

    demo = build_ui()
    demo.queue()  # serialise — only one model copy fits in 6 GB VRAM
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("LP_UI_PORT", str(config.UI_PORT))),
        share=config.UI_SHARE,
        show_error=True,
    )


if __name__ == "__main__":
    main()
