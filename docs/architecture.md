# LegalPolicy_LLM — High-Level System Architecture

Bilingual (EN/AR) legal explainer over the **Egyptian Civil Code** (1,093 articles).
The system fronts **six comparable approaches** — prompt engineering, fine-tuning, two RAG
styles, multi-agent, and a combined pipeline — behind one unified UI, all running on a
single consumer laptop GPU (RTX 3050, 6 GB).

> Two formats are provided:
> - [`architecture.drawio`](architecture.drawio) — open at <https://app.diagrams.net> (File → Open).
> - The Mermaid diagrams below — render on GitHub or any Mermaid viewer.

---

## 1. System overview (component graph)

```mermaid
flowchart TB
    user(["User<br/>(layperson / lawyer)"])

    subgraph UI["USER INTERFACE LAYER"]
        unified["Unified UI · Gradio :7870<br/>apps/unified_ui/app.py<br/>backend dropdown · retrieval & trace panels"]
        reg["backends.py registry<br/>uniform generate(msg, history) → Reply<br/>lazy-loaded adapters"]
        standalone["Standalone Gradio apps<br/>Finetuned :7860 · Bilingual RAG :7861<br/>Agents :7860 · Combined :7863"]
    end

    subgraph BE["BACKEND / APPROACH LAYER"]
        p1["P1 · Prompt Design (baseline)<br/>system prompt + safety gate → Llama-3.2-3B"]
        p2["P2 · Fine-tuned closed-book<br/>Qwen2.5-3B 4-bit + QLoRA adapter"]
        p3["P3 · Neo4j Graph RAG · FastAPI :8000<br/>hybrid retrieval + rerank"]
        p4["P4 · Bilingual Vector RAG · FastAPI :8100<br/>Chroma + cross-encoder rerank"]
        p5["P5 · Multi-Agent · LangGraph<br/>router → Researcher/Explainer/Comparator"]
        p6["P6 · Combined<br/>Neo4j retrieval → Fine-tuned Qwen"]
    end

    subgraph RT["MODEL RUNTIMES"]
        shared["Shared local Qwen (one 4-bit GPU load)<br/>base = baseline · base+adapter = finetuned"]
        ollama["Ollama :11434<br/>Llama-3.2-3B · Qwen3:4b · Qwen2.5:3b"]
        glm["glm-5 via Agent SDK gateway"]
    end

    subgraph DATA["DATA & RETRIEVAL STORES"]
        corpus[("orig_data.json<br/>1,093 bilingual articles")]
        neo4j[("Neo4j Aura<br/>articles + keywords + vector index")]
        chroma[("Chroma<br/>2,262 bilingual chunks")]
        lightrag[("LightRAG graph<br/>articles + entities + relations")]
        bge["BAAI/bge-m3 embeddings<br/>multilingual · 1024-dim · CPU"]
    end

    lf["Langfuse v3 (self-hosted :3000)<br/>per-question traces: latency · #articles · cost"]

    user --> unified
    unified --- reg
    unified -.-> standalone
    reg --> p1 & p2 & p3 & p4 & p5 & p6

    p1 --> ollama
    p2 --> shared
    p3 --> neo4j --> ollama
    p4 --> chroma --> ollama
    p5 --> lightrag
    p5 --> glm
    p6 --> neo4j
    p6 --> shared

    bge -.embeds queries.-> neo4j
    bge -.embeds queries.-> chroma
    corpus -- ingest --> neo4j
    corpus -- chunk+index --> chroma
    corpus -- extract --> lightrag

    unified -.trace.-> lf
```

---

## 2. Request flow (per query)

```mermaid
sequenceDiagram
    participant U as User
    participant UI as Unified UI
    participant B as Selected backend
    participant R as Retrieval store
    participant M as LLM (Qwen / Llama / glm-5)
    participant LF as Langfuse

    U->>UI: question (AR/EN) + backend choice
    UI->>B: generate(message, history)
    Note over B: deterministic safety gate<br/>(refuse case prediction / personal advice / circumvention)
    alt RAG / combined backend (P3, P4, P6)
        B->>R: embed + hybrid/vector retrieve top-k
        R-->>B: grounding articles/chunks
    end
    B->>M: prompt (+ grounding if any)
    M-->>B: answer
    B-->>UI: Reply (answer + retrieval/trace panel)
    UI->>LF: trace_question(latency, #articles, cost, backend)
    UI-->>U: answer + sources
```

---

## 3. Offline training pipeline (produces the P2 adapter)

```mermaid
flowchart LR
    corpus[("orig_data.json<br/>1,093 articles")]
    kb["knowledge_builder.py<br/>11 task families → 22K+ Q&A pairs"]
    ds[("qa_pairs_knowledge*.jsonl")]
    trainer["train_unsloth.py<br/>Unsloth + TRL + PEFT<br/>r=16 α=32 · 4-bit NF4 · 4 epochs"]
    adapter[("runs/qlora-qwen2.5-3b-knowledge")]
    gguf["optional merge → GGUF<br/>ollama create"]
    serve["served by P2 / P6<br/>via shared local Qwen"]

    corpus --> kb --> ds --> trainer --> adapter
    adapter --> gguf
    adapter --> serve
```

---

## 4. Component reference

| # | Component | Entry point | Port | Stack | Retrieval / model |
|---|-----------|-------------|------|-------|-------------------|
| 0 | Unified UI | `apps/unified_ui/app.py` | 7870 | Gradio | routes to all backends |
| 1 | Prompt Design | `src/.../prompt_design/assistant.py` | CLI | Ollama | Llama-3.2-3B + safety gate |
| 2 | Fine-tuned (closed-book) | `app/finetuned_3b_knowledge_chat_app.py` | 7860 | Gradio + local Qwen | Qwen2.5-3B 4-bit + QLoRA |
| 3 | Neo4j Graph RAG | `apps/api/main.py` | 8000 | FastAPI | BGE-M3 → Neo4j Aura → Qwen2.5:3b |
| 4 | Bilingual Vector RAG | `apps/bilingual_rag/api.py` / `gradio_app.py` | 8100 / 7861 | FastAPI + Gradio | BGE-M3 → Chroma → Qwen2.5:3b |
| 5 | Multi-Agent | `src/.../agents/gradio_app.py` | 7860 | LangGraph | LightRAG + tools → glm-5 |
| 6 | Combined | `apps/legal_graphrag_finetuned/app.py` | 7863 | Gradio | Neo4j retrieval → fine-tuned Qwen |
| — | Observability | `deploy/langfuse/docker-compose.yml` | 3000 | Langfuse v3 | Postgres · ClickHouse · Redis · MinIO |
| — | Full stack | `scripts/serve_stack.sh` | 7870 | all services | — |

### Cross-cutting concerns
- **Safety gate** (deterministic refusal rules) fires *before* the LLM in P1, P5, P6.
- **One corpus** (`data/orig_data.json`) feeds every backend and the training pipeline.
- **Shared embedder** (BGE-M3, CPU) backs both RAG stores, freeing GPU for the LLM.
- **VRAM strategy** (6 GB): one 4-bit Qwen load shared between baseline and fine-tuned via PEFT toggle; RAG services run as separate processes alongside Ollama.
- **Evaluation**: `general_user_legal_questions.csv`, `lawyer_llm_solution_questions.csv`, `scenarios_full.jsonl`, golden `article_lookup_golden.csv`.
