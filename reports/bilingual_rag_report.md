# Bilingual RAG over the Egyptian Civil Code — Project Report

**Project 4** of the Legal Explainer. A bilingual (Arabic + English) Retrieval-Augmented
Generation service over the Egyptian Civil Code: it retrieves the relevant articles from a
vector store and answers the question grounded in them, with inline article citations, in the
user's language.

> **One-line takeaway:** retrieval is strong (hit@5 ≈ 0.6–0.75, and exact for
> article-number questions), but the small 3B answer model is the bottleneck — it often
> refuses to answer even when the correct article was retrieved.

---

## 1. What it is

- A standalone, self-contained service (`apps/bilingual_rag/`) converted from the Kaggle
  notebook `notebooks/bilingual-rag-system-over-the-egyptian-civil-code_fixed_trial.ipynb`.
- Ships two front-ends over one pipeline: a **Gradio** app (`:7861`) and a **FastAPI**
  service (`:8100`). The unified UI calls the FastAPI service over HTTP.
- Answers **only** from retrieved context (no outside knowledge), cites article numbers
  inline (e.g. *(Article 5)* / *(المادة 5)*), and replies in the question's language.

## 2. Architecture & pipeline

```
question
  ├─ exact article-number lookup?  ("المادة 446" / "Article 5")
  │     → direct metadata lookup on article_number in Chroma  → exact article(s)
  └─ else:
        → keyword extraction (Qwen via Ollama)
        → BGE-M3 dense vector search over Chroma (language-filtered)
        → multilingual cross-encoder rerank against the original question
  → grounded bilingual prompt
  → Qwen 2.5 3B answer (Ollama), cited + in-language
```

| Stage | Component |
|---|---|
| Embeddings | **BAAI/bge-m3** (multilingual, 1024-dim) via sentence-transformers |
| Vector store | **Chroma** (persistent, cosine) |
| Reranker | **cross-encoder/mmarco-mMiniLMv2-L12-H384-v1** (multilingual), CPU |
| Answer LLM | **qwen2.5:3b-instruct** via Ollama (temperature 0) |
| Language handling | Arabic-aware detection; search is filtered to the question's language |

## 3. Corpus & indexing

- Source: `data/orig_data.json` — 1,093 bilingual articles (`arabic`, `english`, `metadata`).
- Sentence-aware bilingual chunking (chunk size 800 chars, overlap 120), per language.
- **2,262 chunks** indexed into Chroma (AR + EN), each carrying `article_number`, `language`,
  `section_path`, and chunk position metadata.
- Build once with `./scripts/run_bilingual_rag.sh build`.

## 4. Configuration (key knobs — `apps/bilingual_rag/config.py`)

| Setting | Default | Env override |
|---|---|---|
| Embedder | `BAAI/bge-m3` | `BRAG_EMBED_MODEL` |
| Embed device | `cpu` | `BRAG_EMBED_DEVICE` |
| top_k / candidate_k | 5 / 20 | — |
| Reranker enabled | yes | `BRAG_USE_RERANKER` |
| Restrict-to-language | yes | — |
| Answer LLM | `qwen2.5:3b-instruct` | `BRAG_OLLAMA_MODEL` |
| Chroma dir | `artifacts/bilingual_rag/chroma_db` | `BRAG_CHROMA_DIR` |
| API port | 8100 | `BRAG_API_PORT` |

## 5. Exact article-number lookup (the retrieval fix)

Dense search is great for *meaning* but weak at matching an exact **article number** — asking
*"نص المادة 446"* would return semantically-near articles, not Article 446. The fix: when the
question names an article number (Arabic or English digits, e.g. *"المادة ٤٤٦"*, *"Article 5"*,
*"articles 3 and 7"*), the pipeline does a **direct `article_number` metadata lookup** in
Chroma and returns the exact article(s), bypassing semantic search.

- Code: `extract_article_numbers()` + `fetch_articles_by_number()` in
  `apps/bilingual_rag/pipeline.py`; runs as "Step 0" of `retrieve()`.
- Surfaced through the API as `article_numbers` on the response.

**Verified live:** *"ما نص المادة 1؟"* → `article_numbers: [1]`, `hits: [1]`. On the
article-lookup eval set this lifted retrieval **hit@5 from 0.00 → 0.75+** (the remaining
misses are the answer model, not retrieval).

## 6. Evaluation

**Method.** `scripts/eval_rag.py` runs the service over gold question sets and scores each
answer with a RAGAS-style suite. Judge backend is pluggable — a **local Ollama** model
(automated, default) or **Claude Code** (in-session grading); no external API billing.

**Metrics.**
- *LLM-judged (0–1):* faithfulness, answer relevance, context precision, context recall,
  answer correctness.
- *Deterministic:* citation accuracy (gold article cited), retrieval hit@k, MRR,
  context precision@k, token overlap vs gold.

**Gold sets.**
- `data/general_user_legal_questions.csv` — colloquial general-public questions (AR/EN).
- `data/lawyer_llm_solution_questions.csv` — lawyer-framed questions.
- `data/article_lookup_golden.csv` — "what does Article N say" (forward) and
  "which article says X" (reverse), with explicit `language` / `direction` columns.

### Results (judge = Ollama qwen2.5:3b, k = 5, n = 8 per set — preliminary sample)

| Dataset | Faith | Relev | CtxP | CtxR | Correct | Hit@5 | Cite |
|---|---:|---:|---:|---:|---:|---:|---:|
| general_user_legal_questions | 0.18 | 0.18 | 0.12 | 0.26 | 0.09 | 0.62 | 0.25 |
| lawyer_llm_solution_questions | 0.20 | 0.16 | 0.18 | 0.11 | 0.23 | 0.75 | 0.12 |
| article_lookup_golden | 0.20 | 0.25 | 0.42 | 0.23 | 0.50 | 0.75 | 0.25 |

*(Numbers from an 8-row sample per set; re-run without `--limit` for final figures. Live
artifacts: `reports/eval/rag/SUMMARY.md` + per-run `reports/eval/rag/bilingual-rag__*.jsonl`.)*

### Findings

- **Retrieval works.** Hit@5 is 0.62–0.75 across sets, and exact for article-number queries.
  Context precision is highest on the lookup set (0.42), as expected.
- **The answer model is the bottleneck.** Faithfulness/relevance sit around 0.18 because the
  3B model frequently bails with *"The provided articles do not contain enough information…"*
  even when the right article is in context. Concrete case: *"ما نص المادة 1؟"* retrieves
  exactly Article 1, yet the model still refuses. This is the documented small-answer-model
  limitation, now measured.
- **Citation accuracy is low (0.12–0.25)** — a direct consequence of the bails (a refusal
  cites nothing).

## 7. Known limitations & next steps

1. **Answer model bails under the strict "answer only from context" prompt.** Options:
   soften the prompt, special-case "give me the text of Article N" to return the retrieved
   article verbatim, or swap in a stronger answer model for that step.
2. **Run the full eval** (all rows, not the 8-row sample) for thesis-grade numbers.
3. **Try the fine-tuned 3B as the answer model** (a vector-RAG twin of Project 6) to test
   whether a domain-tuned writer answers grounded questions the stock model refuses.

## 8. How to run

```bash
conda activate legalpolicy

# Build the Chroma index once:
./scripts/run_bilingual_rag.sh build

# Serve it:
./scripts/run_bilingual_rag.sh ui      # Gradio  (:7861)
./scripts/run_bilingual_rag.sh api     # FastAPI (:8100)

# Evaluate (writes reports/eval/rag/SUMMARY.md):
python scripts/eval_rag.py --system bilingual-rag                 # all 3 gold sets
python scripts/eval_rag.py --system bilingual-rag --limit 20      # quick sample
```

## 9. Where the code lives

| Path | Purpose |
|---|---|
| `apps/bilingual_rag/pipeline.py` | retrieval + answer pipeline (incl. exact-article lookup) |
| `apps/bilingual_rag/api.py` | FastAPI service (`/api/v1/ask`, `/search`, `/health`) |
| `apps/bilingual_rag/gradio_app.py` | standalone Gradio UI |
| `apps/bilingual_rag/build_index.py` | one-time Chroma index builder |
| `apps/bilingual_rag/config.py` | all settings (env-overridable) |
| `src/legal_explainer/eval/` | shared RAGAS-style eval module |
| `scripts/eval_rag.py` | evaluation CLI |
| `data/*.csv` | gold question sets |
