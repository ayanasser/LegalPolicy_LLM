"""
RAG Pipeline
============
Encapsulates all heavy objects (BGE-M3 model, Neo4j driver, Ollama client)
as a single class that is initialised once at API startup.

Stages
------
1. extract_metadata  – Qwen3 extracts keywords / topics / article numbers
2. fetch_by_number   – direct Neo4j lookup for explicit article references
3. fetch_by_keywords – keyword + section graph traversal
4. fetch_by_semantic – BGE-M3 vector search via Neo4j vector index
5. rerank            – merge & score (55% semantic + 35% keyword + 10% direct)
6. generate_answer   – Qwen3 answers using top-k articles as context
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

import ollama as ollama_lib
from FlagEmbedding import BGEM3FlagModel
from neo4j import GraphDatabase

from .config import Settings


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """\
You are a metadata extractor for an Egyptian Civil Law database.
Given a user question (English or Arabic), extract search metadata.

Return ONLY a valid JSON object — no markdown, no extra text:
{
  "keywords_en":      ["english", "legal", "keywords"],
  "keywords_ar":      ["كلمات", "عربية"],
  "legal_topics":     ["broad legal topic 1", "broad legal topic 2"],
  "article_numbers":  [],
  "search_query":     "one English sentence capturing the core legal question"
}

Rules:
- article_numbers: list of integers if articles are explicitly mentioned, else []
- keywords_en / keywords_ar: 3-8 concise terms directly from the question
- legal_topics: 1-4 broad categories (e.g. prescription, contracts, rights)
- search_query: always in English, even if the question is in Arabic
"""

_ANSWER_SYSTEM = """\
You are a highly knowledgeable legal assistant specialising in Egyptian Civil Law.

You receive:
  1. The user's question (English or Arabic)
  2. Relevant law articles from the database (Arabic + English text)

Instructions:
- Answer based ONLY on the provided articles.
- Cite every article you rely on (e.g. "According to Article 7 …").
- If the articles do not contain enough information, say so explicitly.
- Match the response language to the question language.
- Be precise and professional — this is a legal context.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Cypher queries
# ─────────────────────────────────────────────────────────────────────────────

_Q_BY_NUMBER = """
MATCH (a:Article)
WHERE a.number IN $nums
RETURN a.id AS id, a.number AS number,
       a.english AS english, a.arabic AS arabic
"""

_Q_BY_KEYWORD = """
UNWIND $keywords AS kw
MATCH (a:Article)-[:HAS_KEYWORD]->(k:Keyword)
WHERE toLower(k.name) CONTAINS toLower(kw)
WITH  a, count(DISTINCT k) AS hits
ORDER BY hits DESC
LIMIT  $limit
RETURN a.id AS id, a.number AS number,
       a.english AS english, a.arabic AS arabic,
       hits AS kw_hits
"""

_Q_BY_SECTION = """
UNWIND $topics AS topic
MATCH (s:Section)<-[:IN_SECTION]-(a:Article)
WHERE toLower(s.name) CONTAINS toLower(topic)
WITH  a, count(DISTINCT s) AS hits
ORDER BY hits DESC
LIMIT  $limit
RETURN a.id AS id, a.number AS number,
       a.english AS english, a.arabic AS arabic,
       hits AS kw_hits
"""

_Q_SEMANTIC = """
CALL db.index.vector.queryNodes('article_embedding', $topK, $vec)
YIELD node AS a, score
RETURN a.id AS id, a.number AS number,
       a.english AS english, a.arabic AS arabic,
       score AS sem_score
"""

_Q_ARTICLE = """
MATCH (a:Article {number: $number})
RETURN a.id AS id, a.number AS number,
       a.english AS english, a.arabic AS arabic
"""


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline class
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:

    def __init__(self, settings: Settings) -> None:
        self.cfg = settings
        self._embed_model: BGEM3FlagModel | None = None
        self._driver = None
        self._ollama: ollama_lib.Client | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def startup(self) -> None:
        """Load models and open connections.  Called once at API startup."""
        print("[pipeline] Loading BGE-M3 …")
        self._embed_model = BGEM3FlagModel(
            self.cfg.embed_model_name,
            use_fp16=self.cfg.embed_use_fp16,
        )
        print("[pipeline] Connecting to Neo4j …")
        self._driver = GraphDatabase.driver(
            self.cfg.neo4j_uri,
            auth=(self.cfg.neo4j_user, self.cfg.neo4j_password),
        )
        self._driver.verify_connectivity()
        print("[pipeline] Connecting to Ollama …")
        self._ollama = ollama_lib.Client(host=self.cfg.ollama_host)
        # Warm up: quick no-op to ensure the model is loaded in Ollama
        self._ollama.chat(
            model=self.cfg.llm_model,
            messages=[{"role": "user", "content": "/no_think hi"}],
            options={"num_predict": 1},
        )
        print("[pipeline] Ready.")

    def shutdown(self) -> None:
        if self._driver:
            self._driver.close()

    # ── Low-level helpers ─────────────────────────────────────────────────

    def _run_cypher(self, query: str, **params) -> list[dict]:
        with self._driver.session(database=self.cfg.neo4j_database) as s:
            return s.run(query, **params).data()

    def _embed(self, text: str) -> list[float]:
        out = self._embed_model.encode(
            [text],
            batch_size=1,
            max_length=self.cfg.embed_max_length,
        )
        return out["dense_vecs"][0].tolist()

    def _chat(self, system: str, user: str,
              temperature: float, use_json: bool = False) -> str:
        kwargs: dict[str, Any] = {
            "model": self.cfg.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "options": {"temperature": temperature},
        }
        if use_json:
            kwargs["format"] = "json"
        resp = self._ollama.chat(**kwargs)
        raw = resp["message"]["content"]
        # Strip <think>…</think> blocks from Qwen3 chain-of-thought
        return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # ── Stage 1 — Metadata extraction ─────────────────────────────────────

    def extract_metadata(self, question: str) -> dict:
        raw = self._chat(
            system=_EXTRACT_SYSTEM,
            user="/no_think\n" + question,
            temperature=self.cfg.llm_temp_extract,
            use_json=True,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "keywords_en": question.split()[:5],
                "keywords_ar": [],
                "legal_topics": [],
                "article_numbers": [],
                "search_query": question,
            }

    # ── Stage 2 — Retrieval ───────────────────────────────────────────────

    def fetch_by_number(self, numbers: list[int]) -> list[dict]:
        if not numbers:
            return []
        rows = self._run_cypher(_Q_BY_NUMBER, nums=numbers)
        return [
            {**r, "kw_score": 1.0, "sem_score": 1.0, "source": "direct"}
            for r in rows
        ]

    def fetch_by_keywords(
        self,
        keywords_en: list[str],
        keywords_ar: list[str],
        legal_topics: list[str],
        limit: int | None = None,
    ) -> list[dict]:
        limit = limit or self.cfg.retrieval_top_k
        all_kw = keywords_en + keywords_ar + legal_topics
        if not all_kw:
            return []

        kw_rows  = self._run_cypher(_Q_BY_KEYWORD, keywords=all_kw, limit=limit)
        sec_rows = self._run_cypher(
            _Q_BY_SECTION,
            topics=legal_topics or keywords_en or [""],
            limit=limit,
        )

        merged: dict[str, dict] = {}
        for r in kw_rows + sec_rows:
            aid = r["id"]
            if aid not in merged or r["kw_hits"] > merged[aid]["kw_hits"]:
                merged[aid] = r

        max_hits = max((v["kw_hits"] for v in merged.values()), default=1)
        return [
            {
                "id":       v["id"],
                "number":   v["number"],
                "english":  v["english"],
                "arabic":   v["arabic"],
                "kw_score": v["kw_hits"] / max_hits,
                "sem_score": 0.0,
                "source":   "keyword",
            }
            for v in merged.values()
        ]

    def fetch_by_semantic(self, query_text: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or self.cfg.retrieval_top_k
        vec   = self._embed(query_text)
        rows  = self._run_cypher(_Q_SEMANTIC, vec=vec, topK=top_k)
        return [
            {
                "id":       r["id"],
                "number":   r["number"],
                "english":  r["english"],
                "arabic":   r["arabic"],
                "kw_score": 0.0,
                "sem_score": float(r["sem_score"]),
                "source":   "semantic",
            }
            for r in rows
            if float(r["sem_score"]) >= self.cfg.sim_threshold
        ]

    def get_article(self, number: int) -> dict | None:
        rows = self._run_cypher(_Q_ARTICLE, number=number)
        return rows[0] if rows else None

    # ── Stage 3 — Reranking ───────────────────────────────────────────────

    @staticmethod
    def rerank(
        direct:   list[dict],
        keyword:  list[dict],
        semantic: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        Combined score:
          55% semantic cosine similarity
          35% keyword / section overlap
          10% bonus for direct article-number mentions
        """
        pool: dict[str, dict] = {}

        def _add(records: list[dict], sem_w: float, kw_w: float) -> None:
            for r in records:
                aid = r["id"]
                if aid not in pool:
                    pool[aid] = {k: r[k] for k in
                                 ("id", "number", "english", "arabic", "source")}
                    pool[aid]["score"] = 0.0
                pool[aid]["score"] += r.get("sem_score", 0.0) * sem_w
                pool[aid]["score"] += r.get("kw_score",  0.0) * kw_w

        _add(direct,   0.55, 0.35)
        _add(keyword,  0.55, 0.35)
        _add(semantic, 0.55, 0.35)

        direct_ids = {r["id"] for r in direct}
        for aid in direct_ids:
            if aid in pool:
                pool[aid]["score"] += 0.10

        ranked = sorted(pool.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    # ── Stage 4 — Answer generation ───────────────────────────────────────

    @staticmethod
    def _format_context(articles: list[dict]) -> str:
        parts = []
        for a in articles:
            parts.append(
                f"--- {a['id']} (score={a['score']:.3f}) ---\n"
                f"[English]\n{a['english']}\n\n"
                f"[Arabic]\n{a['arabic']}"
            )
        return "\n\n".join(parts)

    def generate_answer(self, question: str, articles: list[dict]) -> str:
        if not articles:
            return "No relevant articles were found in the database for this question."
        context  = self._format_context(articles)
        user_msg = f"RETRIEVED LAW ARTICLES:\n\n{context}\n\n---\n\nQUESTION: {question}"
        return self._chat(
            system=_ANSWER_SYSTEM,
            user=user_msg,
            temperature=self.cfg.llm_temp_answer,
        )

    # ── Public async API ──────────────────────────────────────────────────

    async def ask(self, question: str, top_k: int | None = None) -> dict:
        """Full RAG — returns answer + source articles + metadata + timing."""
        top_k = top_k or self.cfg.answer_top_k
        t0    = time.perf_counter()

        # Stage 1 — metadata (sync, runs in thread)
        meta = await asyncio.to_thread(self.extract_metadata, question)

        # Stage 2 — all three retrieval signals in parallel
        direct_t  = asyncio.to_thread(
            self.fetch_by_number, meta.get("article_numbers", [])
        )
        keyword_t = asyncio.to_thread(
            self.fetch_by_keywords,
            meta.get("keywords_en", []),
            meta.get("keywords_ar", []),
            meta.get("legal_topics", []),
        )
        semantic_t = asyncio.to_thread(
            self.fetch_by_semantic, meta.get("search_query", question)
        )
        direct, keyword, semantic = await asyncio.gather(
            direct_t, keyword_t, semantic_t
        )

        # Stage 3 — rerank
        top_articles = self.rerank(direct, keyword, semantic, top_k)

        # Stage 4 — generate answer
        answer = await asyncio.to_thread(
            self.generate_answer, question, top_articles
        )

        return {
            "answer":             answer,
            "articles":           top_articles,
            "metadata":           meta,
            "processing_time_ms": round((time.perf_counter() - t0) * 1000),
        }

    async def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Retrieval only — no answer generation."""
        top_k = top_k or self.cfg.retrieval_top_k

        meta = await asyncio.to_thread(self.extract_metadata, query)

        direct, keyword, semantic = await asyncio.gather(
            asyncio.to_thread(self.fetch_by_number, meta.get("article_numbers", [])),
            asyncio.to_thread(
                self.fetch_by_keywords,
                meta.get("keywords_en", []),
                meta.get("keywords_ar", []),
                meta.get("legal_topics", []),
            ),
            asyncio.to_thread(
                self.fetch_by_semantic, meta.get("search_query", query)
            ),
        )
        return self.rerank(direct, keyword, semantic, top_k)
