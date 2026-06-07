"""Standalone Gradio app for Project 6 — Legal Graph-RAG + Finetuned model.

Self-contained: loads the graph-retrieval pipeline (BGE-M3 + Neo4j + Ollama for
metadata) and the finetuned QLoRA adapter in ONE process — no separate service
needed. The legal prompt drives the flow; the finetuned model writes the answer.

Run from project root (legalpolicy env):
    PYTHONPATH=src python -m apps.legal_graphrag_finetuned.app
Prereqs:
    - NEO4J_* in .env (graph retrieval) + Ollama qwen3:4b (metadata extraction)
    - free GPU (~4-5 GB: BGE-M3 + the finetuned Qwen)
Env:
    LP_PORT=7863   server port
"""
from __future__ import annotations

import asyncio
import os

import gradio as gr
import pandas as pd

from .config import get_settings
from .pipeline import LegalGraphRagFinetuned

_HEADERS = ["article", "score", "lang", "snippet"]
_PIPE: LegalGraphRagFinetuned | None = None


def _pipe() -> LegalGraphRagFinetuned:
    global _PIPE
    if _PIPE is None:
        p = LegalGraphRagFinetuned(get_settings())
        p.startup()
        _PIPE = p
    return _PIPE


def _rows(articles: list[dict]) -> pd.DataFrame:
    if not articles:
        return pd.DataFrame(columns=_HEADERS)
    return pd.DataFrame([{
        "article": a.get("number"),
        "score": round(a.get("score", 0.0), 3),
        "lang": "ar+en",
        "snippet": (a.get("english") or a.get("arabic") or "")[:280],
    } for a in articles])


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Legal Graph-RAG + Finetuned — Egyptian Civil Code") as demo:
        gr.Markdown(
            "# ⚖️ Legal Graph-RAG + Finetuned model\n"
            "**Legal prompt + safety** → **Neo4j graph retrieval** → answer written by the "
            "**finetuned Qwen2.5-3B knowledge adapter**. One self-contained pipeline."
        )
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=460)
                msg = gr.Textbox(placeholder="Ask about the Egyptian Civil Code…", submit_btn=True)
                meta_md = gr.Markdown("")
            with gr.Column(scale=2):
                gr.Markdown("### 🔎 Retrieved articles")
                hits_df = gr.Dataframe(headers=_HEADERS, wrap=True, interactive=False)

        def on_submit(message, history):
            history = history or []
            message = (message or "").strip()
            if not message:
                return history, gr.update(), "", ""
            result = asyncio.run(_pipe().answer(message))
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": result["answer"]},
            ]
            tag = "refused by safety gate" if result.get("refused") else \
                f"graph RAG → finetuned 3B · {len(result['articles'])} articles"
            meta = f"{tag} · {result.get('processing_time_ms', 0)} ms"
            return history, _rows(result["articles"]), meta, ""

        msg.submit(on_submit, [msg, chatbot], [chatbot, hits_df, meta_md, msg])
    return demo


def main() -> None:
    demo = build_ui()
    demo.queue()
    demo.launch(
        theme=gr.themes.Soft(primary_hue="blue"),
        server_name="0.0.0.0",
        server_port=int(os.environ.get("LP_PORT", str(get_settings().ui_port))),
        share=os.environ.get("LP_SHARE", "0") == "1",
        show_error=True,
    )


if __name__ == "__main__":
    main()
