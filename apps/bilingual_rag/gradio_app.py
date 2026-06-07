"""Standalone Gradio app for the Bilingual RAG (Project 4).

Chat over the Egyptian Civil Code with a retrieval panel showing the articles
that grounded each answer. Self-contained — talks to the in-process
BilingualRAGPipeline (Chroma + BGE-M3 + cross-encoder rerank + Qwen via Ollama).

Run from project root (legalpolicy env):
    python -m apps.bilingual_rag.gradio_app
Prereqs:
    1. Build the index once:  python -m apps.bilingual_rag.build_index
    2. Ollama running with the model pulled:  ollama pull qwen2.5:3b-instruct
Env:
    LP_PORT=7861   server port (default 7861, to avoid clashing with 7860)
    LP_SHARE=1     public share link
"""
from __future__ import annotations

import os

import gradio as gr
import pandas as pd

from .config import get_settings
from .pipeline import BilingualRAGPipeline

_PIPE: BilingualRAGPipeline | None = None


def _pipe() -> BilingualRAGPipeline:
    global _PIPE
    if _PIPE is None:
        _PIPE = BilingualRAGPipeline(get_settings())
        _ = _PIPE.collection  # fail fast if the index is missing
    return _PIPE


def _hits_to_df(hits: list[dict]) -> pd.DataFrame:
    if not hits:
        return pd.DataFrame(columns=["rank", "article", "lang", "score", "rerank", "text"])
    rows = []
    for i, h in enumerate(hits, 1):
        rows.append({
            "rank": i,
            "article": h["article_number"],
            "lang": h["language"],
            "score": round(h["score"], 4),
            "rerank": round(h["rerank_score"], 4) if "rerank_score" in h else None,
            "text": h["text"][:300] + ("…" if len(h["text"]) > 300 else ""),
        })
    return pd.DataFrame(rows)


def respond(message, history, top_k, use_rerank):
    message = (message or "").strip()
    if not message:
        return "Please type a question about the Egyptian Civil Code.", _hits_to_df([]), ""
    result = _pipe().answer(message, k=int(top_k), use_rerank=bool(use_rerank))
    df = _hits_to_df(result["hits"])
    meta = (
        f"**Detected language:** `{result.get('detected_language')}`  ·  "
        f"**Keywords:** {', '.join(result.get('keywords') or []) or '—'}  ·  "
        f"**Search query:** {result.get('search_query', '')}  ·  "
        f"**{result.get('processing_time_ms', 0)} ms**"
    )
    return result["answer"], df, meta


EXAMPLES = [
    "What rules apply when there is no written law?",
    "ما هي حقوق المستأجر في عقد الإيجار؟",
    "obligations of the seller in a contract of sale",
    "متى يكون استعمال الحق غير مشروع؟",
    "What is the age of legal majority under the Egyptian Civil Code?",
]


def build_ui() -> gr.Blocks:
    cfg = get_settings()
    with gr.Blocks(title="Bilingual RAG — Egyptian Civil Code") as demo:
        gr.Markdown(
            "# 🔎 Bilingual RAG — Egyptian Civil Code\n"
            f"BGE-M3 + Chroma + cross-encoder rerank + **{cfg.llm_model}** (Ollama). "
            "Ask in **English or العربية** — the retrieved articles appear below."
        )
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=420)
                msg = gr.Textbox(placeholder="Ask about the Egyptian Civil Code…", submit_btn=True)
                with gr.Row():
                    top_k = gr.Slider(1, 10, value=cfg.top_k, step=1, label="Top-k articles")
                    use_rerank = gr.Checkbox(value=cfg.use_reranker, label="Cross-encoder rerank")
                gr.Examples(examples=EXAMPLES, inputs=[msg])
            with gr.Column(scale=2):
                gr.Markdown("### 📚 Retrieved articles")
                meta_md = gr.Markdown("_(retrieval details appear here)_")
                hits_df = gr.Dataframe(
                    headers=["rank", "article", "lang", "score", "rerank", "text"],
                    wrap=True, interactive=False,
                )

        def _on_submit(message, chat_history, k, rerank):
            answer, df, meta = respond(message, chat_history, k, rerank)
            chat_history = (chat_history or []) + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer},
            ]
            return chat_history, df, meta, ""

        msg.submit(_on_submit, [msg, chatbot, top_k, use_rerank], [chatbot, hits_df, meta_md, msg])
    return demo


def main() -> None:
    demo = build_ui()
    demo.queue()
    demo.launch(
        theme=gr.themes.Soft(),
        server_name="0.0.0.0",
        server_port=int(os.environ.get("LP_PORT", "7861")),
        share=os.environ.get("LP_SHARE", "0") == "1",
        show_error=True,
    )


if __name__ == "__main__":
    main()
