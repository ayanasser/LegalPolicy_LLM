# Running the projects + the Unified UI

Five projects live in this repo. This doc shows how to run **each one standalone**
and how to launch the **Unified UI** that fronts all of them.

All commands assume the `legalpolicy` conda env and the project root:

```bash
conda activate legalpolicy
cd /home/aya/master/genai/LegalPolicy_LLM
```

The `scripts/*.sh` helpers default to the env at
`/home/aya/miniconda3/envs/legalpolicy/bin/python` (override with `LP_PY=...`).

---

## 0. One-time setup

```bash
# Ollama models used across the projects
ollama serve &                       # if not already running
ollama pull llama3.2:3b              # prompt design + llama baseline
ollama pull qwen2.5:3b-instruct     # bilingual RAG + qwen baseline (ollama path)
ollama pull qwen3:4b                # Neo4j RAG answer/extraction
ollama pull qwen3-embedding:0.6b    # multi-agent LightRAG retrieval

# BGE-M3 ships only a .bin; transformers 5.x + torch<2.6 needs safetensors.
# Convert once (after bge-m3 has been downloaded at least once):
python scripts/ensure_bge_m3_safetensors.py

# Build the bilingual RAG vector index (one-time, ~a few min on CPU):
./scripts/run_bilingual_rag.sh build
```

`.env` must contain: `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` (multi-agent
glm-5 gateway) and, for the Neo4j RAG, `NEO4J_URI` / `NEO4J_USERNAME` /
`NEO4J_PASSWORD` / `NEO4J_DATABASE`.

> **GPU note (RTX 3050, 6 GB):** the local Qwen (baseline/finetuned) needs ~2.5 GB.
> Don't run it while a QLoRA training job owns the card. The RAG services load
> BGE-M3 on **CPU** by default so they can coexist with the local model.

---

## 1. Prompt Design — CLI (Project 1)

Ollama `llama3.2:3b` + legal system prompt + deterministic safety/refusal + disclaimers.

```bash
./scripts/run_prompt_design.sh
# or:  python "src/Prompt Design/legal_policy_assistant_egypt_v2.py"
```

## 2. Finetuned Knowledge 3B — Gradio (Project 2)  ·  http://localhost:7860

QLoRA adapter `runs/qlora-qwen2.5-3b-knowledge` over Qwen2.5-3B-Instruct (closed-book).

```bash
./scripts/run_finetuned.sh
# or:  python -m app.finetuned_3b_knowledge_chat_app
```

## 3. Neo4j Graph RAG — FastAPI (Project 3)  ·  http://localhost:8000/docs

BGE-M3 + Neo4j Aura graph + vector index + Qwen3 (Ollama). Needs `NEO4J_*` in `.env`.

```bash
./scripts/run_neo4j_rag.sh
# Health:  curl localhost:8000/health
# Ask:     curl -X POST localhost:8000/api/v1/ask -H 'content-type: application/json' \
#               -d '{"question":"What is force majeure?","top_k":5}'
```

## 4. Bilingual RAG — Gradio / FastAPI (Project 4)  ·  UI http://localhost:7861

Converted from `notebooks/bilingual-rag-system-over-the-egyptian-civil-code_fixed_trial.ipynb`.
BGE-M3 + Chroma (persistent) + multilingual cross-encoder rerank + `qwen2.5:3b-instruct`,
with exact article-number lookup (e.g. "نص المادة 446") bypassing semantic search.

```bash
./scripts/run_bilingual_rag.sh build   # one-time index build
./scripts/run_bilingual_rag.sh ui      # standalone Gradio (port 7861)
./scripts/run_bilingual_rag.sh api     # FastAPI service (port 8100)
```

## 5. Multi-Agent — Gradio (Project 5)  ·  http://localhost:7860

LangGraph orchestrator (safety → router → subagents → tools → synthesis) over the
glm-5 gateway + LightRAG KG. Needs `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN`.

```bash
./scripts/run_agents.sh
# or:  PYTHONPATH=src python -m legal_explainer.agents.gradio_app
```

---

## 6. Unified UI  ·  http://localhost:7870

One chat. A dropdown picks the backend; RAG backends show a **retrieval** panel,
the multi-agent backend shows a **trace** panel. Baseline = raw Qwen2.5-3B-Instruct.

Dropdown options: Baseline Qwen2.5-3B · Baseline Llama-3.2-3B · Finetuned Knowledge
adapter · Prompt Design (Llama) · Prompt Design (Qwen) · Neo4j Graph RAG ·
Bilingual RAG · Combined (legal prompt + Graph RAG + Finetuned 3B) · Multi-Agent.

The **Combined** dropdown option calls the Project 6 service over HTTP — start it
first: `PYTHONPATH=src python -m apps.legal_graphrag_finetuned.api` (:8200), or
add `LP_WITH_LGF=1` when running `serve_stack.sh`.

---

## 7. Legal Graph-RAG + Finetuned — standalone project (Project 6)  ·  http://localhost:7863

A separate self-contained project (`apps/legal_graphrag_finetuned/`) that fuses
three components into one pipeline: **legal prompt + safety** (Project 1) →
**Neo4j graph retrieval** (Project 3) → answer written by the **finetuned
Qwen2.5-3B knowledge adapter** (Project 2), instead of the Ollama model the graph
RAG normally calls. Loads retrieval + the finetuned model in ONE process.

```bash
./scripts/run_legal_graphrag_finetuned.sh    # Gradio (:7863)
# or the API:
PYTHONPATH=src python -m apps.legal_graphrag_finetuned.api   # :8200
```

Needs `NEO4J_*` in `.env`, Ollama `qwen3:4b`, and a **free GPU (~4–5 GB** for
BGE-M3 + the finetuned Qwen together). See `apps/legal_graphrag_finetuned/README.md`.

```bash
# UI only (chat + local/ollama/agent backends; RAG backends need their services):
./scripts/run_unified_ui.sh

# Full stack — brings up both RAG services + the UI together:
./scripts/serve_stack.sh
```

The unified UI talks to the RAG services over HTTP:
`LP_NEO4J_RAG_URL` (default `http://localhost:8000`) and
`LP_BILINGUAL_RAG_URL` (default `http://localhost:8100`). If a service is down,
that backend shows a friendly "start the service" message instead of erroring.

Suggested questions are sampled from `data/general_user_legal_questions.csv`,
`data/lawyer_llm_solution_questions.csv`, and `data/qa_pairs_knowledge.jsonl`.
