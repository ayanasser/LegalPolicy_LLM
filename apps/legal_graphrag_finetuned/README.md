# Project 6 — Legal Graph-RAG + Finetuned model

A self-contained project that fuses **three** of the other projects into one
pipeline:

| Component | From | Role here |
|---|---|---|
| Legal system prompt + safety/refusal gate | Project 1 (Prompt Design) | Frames the answer; blocks unsafe asks before retrieval |
| Neo4j graph retrieval (BGE-M3 + graph + vector + rerank) | Project 3 (`apps/api`) | Fetches the relevant Civil Code articles |
| Finetuned Qwen2.5-3B knowledge adapter | Project 2 (`runs/qlora-qwen2.5-3b-knowledge`) | **Writes the final grounded answer** (instead of the Ollama model the graph RAG normally calls) |

```
question
  → safety gate + legal system prompt
  → Neo4j graph retrieval  (top-k articles)
  → finetuned model writes the answer, grounded on those articles + disclaimer
```

## Run

```bash
conda activate legalpolicy
# Standalone Gradio (loads retrieval + finetuned model in ONE process):
./scripts/run_legal_graphrag_finetuned.sh
#   UI  → http://localhost:7863
# Or the API:
PYTHONPATH=src python -m apps.legal_graphrag_finetuned.api   # :8200
```

### Requirements
- `NEO4J_*` in `.env` (graph retrieval) and Ollama `qwen3:4b` (metadata extraction).
- A **free GPU** (~4–5 GB): BGE-M3 + the finetuned Qwen run together in-process.
  Don't run while a QLoRA training job owns the card.

### Note
The finetuned adapter was trained *closed-book, single-turn, no system prompt*.
Feeding it a legal system prompt + retrieved context is out of its training
distribution, so its behavior here is an experimental fusion — it grounds on the
retrieved articles but may format differently from the pure closed-book recall.
