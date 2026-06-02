"""Legal Graph-RAG + Finetuned pipeline (Project 6).

Fuses three components into one self-contained pipeline:

    question
      → safety gate + legal system prompt        (Prompt Design, Project 1)
      → Neo4j graph retrieval (BGE-M3 + graph + vector + rerank)   (Project 3)
      → answer written by the finetuned QLoRA knowledge adapter    (Project 2)

Retrieval reuses apps/api's RAGPipeline in-process (single source of truth for
the graph RAG), but its Ollama answer step is bypassed — the final answer is
produced by the finetuned model with the legal prompt + retrieved articles.
"""
from __future__ import annotations

import time

from .config import LGFSettings, get_settings
from .generator import FinetunedGenerator


class LegalGraphRagFinetuned:
    def __init__(self, settings: LGFSettings | None = None) -> None:
        self.cfg = settings or get_settings()
        self._rag = None  # apps.api.pipeline.RAGPipeline (retrieval)
        self._gen = FinetunedGenerator(
            self.cfg.base_model, self.cfg.adapter_dir, self.cfg.max_new_tokens
        )
        # Prompt-design pieces (legal prompt + deterministic safety).
        from legal_explainer.prompt_design.assistant import (
            SYSTEM_PROMPT, detect_refusal_needed, select_disclaimer, REFUSAL_MESSAGES)
        self._system_prompt = SYSTEM_PROMPT
        self._detect_refusal = detect_refusal_needed
        self._select_disclaimer = select_disclaimer
        self._refusals = REFUSAL_MESSAGES

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def startup(self) -> None:
        """Open the graph-retrieval pipeline (BGE-M3 + Neo4j + Ollama for metadata).
        The finetuned model itself loads lazily on the first answer()."""
        from apps.api.config import get_settings as rag_settings
        from apps.api.pipeline import RAGPipeline
        self._rag = RAGPipeline(rag_settings())
        self._rag.startup()

    def shutdown(self) -> None:
        if self._rag is not None:
            self._rag.shutdown()

    # ── Prompt assembly ───────────────────────────────────────────────────────

    def _build_messages(self, question: str, articles: list[dict]) -> list[dict]:
        context = "\n\n".join(
            f"[Article {a.get('number')}]\nEN: {a.get('english','')}\nAR: {a.get('arabic','')}"
            for a in articles
        ) or "(no articles retrieved)"
        disclaimer = self._select_disclaimer(question)
        system = (
            self._system_prompt
            + "\n\n=== RETRIEVAL GROUNDING ===\n"
            "You are given the most relevant articles of the Egyptian Civil Code below. "
            "Base your explanation on THESE articles, cite the article numbers you rely on, "
            "and if they don't cover the question, say so plainly."
        )
        user = (
            f"RETRIEVED ARTICLES:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            f"[SYSTEM NOTE: End your response with this exact disclaimer on a new line "
            f"after '---':\n{disclaimer}]"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # ── End-to-end ────────────────────────────────────────────────────────────

    async def answer(self, question: str, top_k: int | None = None) -> dict:
        if self._rag is None:
            raise RuntimeError("Pipeline not started — call startup() first.")
        top_k = top_k or self.cfg.top_k
        t0 = time.perf_counter()

        # 1. Safety gate (deterministic) — refuse before retrieving / generating.
        category = self._detect_refusal(question)
        if category:
            return {
                "answer": self._refusals[category], "articles": [],
                "refused": True, "processing_time_ms": round((time.perf_counter() - t0) * 1000),
            }

        # 2. Graph retrieval only (no Ollama answer).
        articles = await self._rag.search(question, top_k=top_k)

        # 3. Generate with the finetuned model + legal prompt + retrieved context.
        import asyncio
        messages = self._build_messages(question, articles)
        answer = await asyncio.to_thread(self._gen.generate_chat, messages)

        return {
            "answer": answer, "articles": articles, "refused": False,
            "processing_time_ms": round((time.perf_counter() - t0) * 1000),
        }
