# Legal Explainer — Agents Package

A multi-agent legal-explainer system over the Egyptian Civil Code, built on the **Claude Agent SDK**. This package implements **Epic 6 (Tooling & Function Calling)** and **Epic 7 (Multi-Agent Collaboration)** end-to-end: deterministic tools, three specialist subagents, a complexity-routed orchestrator, a rule-based safety filter, and an eval harness that plugs into the repo's existing Claude-judge scoring.

> **v2 status (this revision):** the orchestrator has been re-implemented as a **LangGraph** state machine and the router is now **LLM-based** (a single Claude classification call) instead of regex rules. The v1 (plain Python if/else + rule-based router) implementation is still in the codebase but is no longer the default. See [§ Architecture v2](#architecture-v2--langgraph--llm-router) below. v1 traces from earlier runs live under `reports/agent_traces/`.

---

## Why this design

A single LLM call answering "compare Article 660 and Article 713" tends to skip steps — it retrieves a bit, decides on an answer, and writes it out all in one pass. That's fine for "what is an NDA?" but breaks down on harder questions. Splitting work across specialist roles produces better answers on complex queries and gives clearer intermediate state for debugging.

At the same time, multi-agent setups aren't free — every extra role is another model call. So this package ships **two pipelines side by side**:

- **Baseline** — Researcher → Explainer for every query (two LLM calls, always)
- **Orchestrated** — safety → router → conditional dispatch → synthesis (one call for simple, two for complex)

…and an eval harness to measure whether the extra machinery is worth it.

The other design principle: **the model is bad at exact recall**. So instead of hoping it remembers Article 89 verbatim, the agent has a `check_statute_reference` tool that returns the exact text. Instead of trusting it to recall "force majeure" precisely, it has `get_legal_definition` over a curated bilingual glossary. Tools replace "remember" with "look up" — answers become reliable, cheap, and auditable.

---

## Architecture (visual)

```
                            ┌──────────────────────────────────┐
                            │       User query (en / ar)        │
                            └──────────────┬───────────────────┘
                                           │
                            ┌──────────────▼───────────────────┐
                            │  safety.py — rule-based filter   │
                            │  (off-topic? case-specific?)     │
                            └──────────────┬───────────────────┘
                                           │ allow
                            ┌──────────────▼───────────────────┐
                            │  tools/router.py                 │
                            │  rules → LLM fallback            │
                            │  → simple / medium / complex     │
                            └──────────────┬───────────────────┘
                       ┌───────────────────┼───────────────────┐
                       │                   │                   │
                  simple path         medium path         complex path
                       │                   │                   │
       ┌───────────────▼──────┐  ┌─────────▼─────────┐  ┌──────▼──────────┐
       │ glossary lookup      │  │ Researcher        │  │ Comparator (or  │
       │ (no LLM call)        │  │ → tools: glossary │  │   Researcher)   │
       │                      │  │   statute, RAG,   │  │ → same tools    │
       │                      │  │   web_search      │  │ but called per  │
       │                      │  │                   │  │ side            │
       └───────────────┬──────┘  └─────────┬─────────┘  └──────┬──────────┘
                       │                   │                   │
                       │             ┌─────▼─────────┐ ┌───────▼─────────┐
                       │             │ Explainer     │ │ Explainer       │
                       │             │ (no tools —   │ │ (synthesizes    │
                       │             │  grounds on   │ │  comparison     │
                       │             │  findings)    │ │  into prose)    │
                       │             └─────┬─────────┘ └───────┬─────────┘
                       │                   │                   │
                       └─────────┬─────────┴───────────────────┘
                                 ▼
                       ┌─────────────────────┐
                       │ Final answer +      │
                       │ mandatory disclaimer│
                       │ (lang-matched)      │
                       └─────────────────────┘
```

Underneath, every box is a wrapped `claude_agent_sdk.query()` call with its own system prompt and tool allow-list. The whole flow is traced to `reports/agent_traces/run_YYYYMMDD.jsonl` — one event per step, indexed by query id.

---

## File-by-file walkthrough

### Foundation

| File | Purpose |
|---|---|
| [`config.py`](config.py) | Single source of truth: paths, model selection per role, SDK auth env, RAG tunables, Ollama embedding config. Every other module imports constants from here — don't duplicate them. |
| [`__init__.py`](__init__.py) | Empty marker. |

### `data/`

| File | Purpose |
|---|---|
| [`data/glossary.json`](data/glossary.json) | 15 hand-curated Egyptian-civil-law terms, each with **paired EN + AR** entries (term, definition, related article references) plus an aliases list so `indemnity`, `compensation`, `تعويض`, `جبر` all resolve to the same canonical `damages` entry. This is the source of truth for `get_legal_definition`. |
| `data/articles_index.json` *(built at runtime)* | The output of `python -m legal_explainer.agents.tools.statute build` — a dict mapping `"89"` → verbatim text of Article 89 (AR + EN concatenated). Used by `check_statute_reference`. Built from `EgyptianLaw.pdf` by splitting on `Article N` / `المادة N` markers. |

### `tools/` — what the model can call

Every tool follows the Claude Agent SDK `@tool` pattern, then gets bundled into a single MCP server (`mcp_server.py`) that the subagents register.

| File | Tool name | What it does | When the model should call it |
|---|---|---|---|
| [`tools/glossary.py`](tools/glossary.py) | `get_legal_definition(term)` | Looks up `term` (any language, any alias) in `glossary.json`. Returns the bilingual canonical entry or a structured `{found: false, available_terms: […]}`. Also exposes `is_definition_query(query)` — a cheap heuristic the router uses to detect glossary-style queries without an LLM call. | "What is X?", "ما معنى X؟" — single-term definition. **Not** for full articles. |
| [`tools/statute.py`](tools/statute.py) | `check_statute_reference(statute_reference)` | Parses a free-text article reference (`Article 89`, `المادة 89`, `Art. 89`, `Egyptian Civil Code Article 89`), looks up that number in the article index, and returns the verbatim text. Returns `{found: false, reason: not_in_index}` for missing articles. Also exposes `parse_reference()` used by the router and `build_index()` the one-time PDF parser. |  User names a specific article number. **Not** for topical questions. |
| [`tools/rag_search.py`](tools/rag_search.py) | `search_legal_documents(query, top_k, mode)` | Wraps the LightRAG index built by `ingest_rag.ipynb`. Uses `aquery_data()` — the new structured-retrieval API that returns `{entities, relationships, chunks}` **without ever calling the LLM**. Pre-extracts keywords via [`tools/keyword_extract.py`](tools/keyword_extract.py) and passes them as `hl_keywords` / `ll_keywords` so LightRAG also skips its LLM-based keyword extractor. Has a `_llm_guard` callback that raises if anything tries to invoke an LLM call inside LightRAG. | Open-ended topical questions: "how does Egyptian law handle gifts?" |
| [`tools/keyword_extract.py`](tools/keyword_extract.py) | *(internal helper)* | Deterministic keyword extractor using **NLTK + Punkt tokenizer** + bilingual (EN + AR) stopword lists. Outputs `(hl_keywords, ll_keywords)` from a query in microseconds. Combines article-reference parsing, glossary surface/canonical-id matching, and content-token extraction. NLTK data (`punkt_tab`, `stopwords`) is auto-downloaded on first import. |
| [`tools/web_search.py`](tools/web_search.py) | `web_search(query, max_results)` | DuckDuckGo wrapper (no API key required). Used for queries about recent amendments / news / topics outside the corpus. Anthropic-native web_search isn't usable because the default model route is `glm-5` via proxy — to swap, drop this file and enable `tools={"type": "preset", "preset": "claude_code"}` + `allowed_tools=["WebSearch"]` in `ClaudeAgentOptions`. | Recent events, amendments, anything that won't be in the 1948 civil code. |
| [`tools/router.py`](tools/router.py) | *(not exposed as MCP — internal helper)* | `classify_complexity(query)` — Epic 7 task 7.2. Cheap rule-based first pass: glossary hit → simple, comparison keyword → complex, article reference → medium. Falls back to an LLM classifier only when rules can't decide. Returns a `RouterDecision` with reasoning trail for debugging. |
| [`tools/mcp_server.py`](tools/mcp_server.py) | *(plumbing)* | Calls `create_sdk_mcp_server` once to bundle the four tools above into a single MCP server called `legal_tools`. Exports constants like `TOOL_GET_DEFINITION = "mcp__legal_tools__get_legal_definition"` so subagents reference tools by name without stringly-typed literals. |

### `prompts/` — the system prompts

These live in markdown so they can be tracked as docs, reviewed in PRs, and edited without touching Python.

| File | Audience | Key rules |
|---|---|---|
| [`prompts/researcher.md`](prompts/researcher.md) | Researcher subagent | Choose tools by query type (article ref → statute, topical → RAG, term → glossary, recent → web). Max 3 tool calls. Output **strict JSON** with `key_passages`, `key_facts`, `ambiguities`, `language_of_query`. No prose. Never synthesize. |
| [`prompts/explainer.md`](prompts/explainer.md) | Explainer subagent | No tools. Language match the query (en/ar/bilingual). Definition first, then structured body with citations, then concrete example, then mandatory disclaimer. Never introduce facts the Researcher didn't provide. |
| [`prompts/comparator.md`](prompts/comparator.md) | Comparator subagent | Two-sided structured JSON output: `side_a`, `side_b`, `shared_ground`, `differences`, `ambiguities`. One tool call per side. |
| [`prompts/orchestrator.md`](prompts/orchestrator.md) | Reference doc | Currently unused at runtime (the orchestrator is pure Python — no LLM at the top level). Kept as living documentation of the orchestration contract; if you ever migrate to a Claude-driven orchestrator, this is the system prompt. |

### `subagents/` — wrapped `query()` calls

Each subagent is a small async function: build `ClaudeAgentOptions` with the right prompt + tool allow-list, call `query()`, collect text blocks, parse JSON.

| File | Role | Tools allowed | Output shape |
|---|---|---|---|
| [`subagents/researcher.py`](subagents/researcher.py) | **Researcher** — retrieves & extracts | glossary, statute, RAG, web_search | `ResearcherFindings` dataclass: key_passages, key_facts, ambiguities, language_of_query, raw, cost, duration |
| [`subagents/explainer.py`](subagents/explainer.py) | **Explainer** — prose for the user | none | `ExplainerAnswer` dataclass: text, cost, duration |
| [`subagents/comparator.py`](subagents/comparator.py) | **Comparator** — multi-side analysis | glossary, statute, RAG (no web_search — comparisons are about the corpus) | `ComparisonFindings` dataclass: side_a, side_b, shared_ground, differences, ambiguities |

### Top-level orchestration

| File | Purpose |
|---|---|
| [`safety.py`](safety.py) | Rule-based pre-check (no LLM). Patterns cover specific-case advice, requests to draft binding documents, out-of-jurisdiction queries, personal-data requests. Returns a `SafetyVerdict` with `allow` / `refuse` + a suggested refusal message. The orchestrator honors this before any routing. |
| [`orchestrator.py`](orchestrator.py) | **Epic 7 main entry point.** Exposes `run_orchestrated(query)` and `run_baseline(query)` — both async, both returning an `OrchestratorResult`. The orchestrated path does safety → route → dispatch (simple = glossary direct / medium = Researcher → Explainer / complex = Researcher-or-Comparator → Explainer). Every step is logged via `TraceLogger` to JSONL. |
| [`ingest_rag.ipynb`](ingest_rag.ipynb) | The notebook that built the LightRAG index from `EgyptianLaw.pdf`. Lives here because the storage dir (`rag_storage_egyptian_law/`) is consumed by the RAG tool. |
| [`test_pipeline.ipynb`](test_pipeline.ipynb) | End-to-end smoke test: builds the article index, exercises every tool standalone, tests the router on each bucket, runs the orchestrator on simple/medium/complex/refusal paths, runs the baseline for comparison, demonstrates the tool-use ablation, then runs a 5-case mini-eval. Run this first to verify everything is wired correctly. |

### `eval/` — Epic 7 task 7.5

| File | Purpose |
|---|---|
| [`eval/run_eval.py`](eval/run_eval.py) | Loads cases from `data/qa_pairs_raft_val.jsonl` (or any compatible JSONL), runs them through `run_orchestrated` or `run_baseline`, writes predictions to `reports/agent_eval/predictions_<system>.json`. Compatible with the existing judge harness's input format. Also computes **`retrieval_hit@k`** — did the gold article number appear in the answer? — a metric the existing harness doesn't have. CLI: `--system {orchestrated,baseline}`, `--n <count>`, `--input <path>`. |
| [`eval/compare_baseline.py`](eval/compare_baseline.py) | Reads two predictions files and prints a side-by-side table covering latency p50/p90, mean and total cost, retrieval_hit overall and by language, and the orchestrated path distribution. Quality scoring is delegated to the repo's existing `scripts/closed_book_recall_eval.py` (which takes prediction files in the same shape). |

---

## How a query flows through the system

**Example 1 — simple definition (Arabic):** `"ما معنى التعويض؟"`

1. `safety.check_safety` → allow.
2. `classify_complexity` → rule `glossary_term` matches (no LLM call). Decision: `simple`.
3. Orchestrator skips subagents entirely. Calls `lookup_definition("damages")` directly.
4. `_format_glossary_simple` detects Arabic in the query, builds Arabic-first response with English secondary version + short Arabic disclaimer.
5. Returned: 1 LLM call (just the optional LLM router fallback, not used here) → really **zero LLM calls**.

**Example 2 — medium article ref:** `"Walk me through Article 713 of the Egyptian Civil Code."`

1. Safety → allow.
2. Router rule `article_reference` matches → `medium`.
3. Orchestrator calls `run_researcher`. Researcher's system prompt steers it to call `check_statute_reference("Article 713")` → verbatim text returned. Researcher emits JSON with `key_passages`.
4. Orchestrator passes JSON to `run_explainer`. Explainer (no tools, no retrieval) produces English prose with the citation, example, disclaimer.
5. Total: 2 LLM calls (Researcher + Explainer) + 1 deterministic tool call.

**Example 3 — complex comparison:** `"Compare Article 660 and Article 713 in terms of duties."`

1. Safety → allow.
2. Router rule `complex_trigger` matches (`compare`) → `complex`.
3. Orchestrator routes to `run_comparator`. Comparator calls `check_statute_reference` twice (once per article), emits structured JSON.
4. Comparator output goes to `run_explainer`, which writes the comparison as prose.
5. Total: 2 LLM calls + 2 deterministic tool calls.

**Example 4 — refusal:** `"I was arrested last night, should I sign this contract?"`

1. Safety pattern `specific_case_advice` matches → refuse.
2. Orchestrator returns the canned refusal message + disclaimer.
3. Total: **zero LLM calls**.

---

## Running it

All commands below run from the project root (`/Volumes/Shared/NileUni/GenAI/LegalPolicy_LLM`). The `PYTHONPATH=src` prefix is needed because the package lives under `src/` and there's no `pyproject.toml` yet — export it once per shell if you prefer (`export PYTHONPATH=$PWD/src`).

### 1. Build the article index (one-time, ~1s)
```bash
PYTHONPATH=src python -m legal_explainer.agents.tools.statute build
```

### 2. Verify everything works (smoke test)
Open [`test_pipeline.ipynb`](test_pipeline.ipynb) and run top-to-bottom. It exercises every tool, every router rule, every orchestrator path, and runs a 5-case mini-eval.

### 3. Quick eval — 50 cases on v2 (the default LangGraph + LLM-router engine)

This is the recommended starter eval. Takes **~10-15 minutes** at concurrency 4.

```bash
PYTHONPATH=src python -m legal_explainer.agents.eval.run_eval \
    --system orchestrated \
    --engine langgraph \
    --n 50 \
    --shuffle \
    --concurrency 4 \
    --quiet \
    2> /tmp/v2_eval.stderr
```

Flags explained:
- `--system orchestrated` — run the orchestrator (vs the two-role baseline)
- `--engine langgraph` — v2 (LangGraph graph + LLM router). Use `--engine legacy` to compare against v1.
- `--n 50` — 50 cases. Drop the flag entirely to run all 424.
- `--shuffle` — randomize ordering so the 50 are representative, not just the top of the file
- `--concurrency 4` — 4 queries in flight at once. Each query spawns 2-3 Node subprocesses; >8 has been observed to crash the CLI transport.
- `--quiet` — silences the per-step `FlowLogger` output (it still writes the structured trace JSONL)
- `2> /tmp/v2_eval.stderr` — sends LightRAG INFO logs to a file so they don't clog your terminal

Output: `reports/agent_eval/predictions_orchestrated.json`. While the run is in flight, a streaming `predictions_orchestrated.jsonl.partial` is written after every completed case so a crash never costs you the whole run.

### 4. Snapshot mid-run (read-only, doesn't disrupt the eval)
```bash
PYTHONPATH=src python -m legal_explainer.agents.eval.snapshot --system orchestrated
```
Prints aggregate metrics computed from whatever cases have finished so far: success/error counts, retrieval_hit rate (overall + by language), path distribution, latency p50/mean/p90, cost so far + projection to the full run, per-`kind` breakdown.

### 5. Full sweep + comparison against baseline (Epic 7 task 7.5)
```bash
# Run all 424 RAFT cases on v2 (~1 hour at concurrency 4)
PYTHONPATH=src python -m legal_explainer.agents.eval.run_eval \
    --system orchestrated --engine langgraph --concurrency 4 --quiet \
    2> /tmp/orchestrated.stderr

# Same dataset, baseline pipeline (Researcher → Explainer for every query)
PYTHONPATH=src python -m legal_explainer.agents.eval.run_eval \
    --system baseline --concurrency 4 --quiet \
    2> /tmp/baseline.stderr

# Latency / cost / retrieval comparison
PYTHONPATH=src python -m legal_explainer.agents.eval.compare_baseline \
    --baseline     reports/agent_eval/predictions_baseline.json \
    --orchestrated reports/agent_eval/predictions_orchestrated.json

# Quality scores via the existing Claude judge harness
python scripts/closed_book_recall_eval.py reports/agent_eval/predictions_orchestrated.json
```

### 6. v1 vs v2 head-to-head (optional)

Re-run the same 50 cases through v1 to compare against v2:

```bash
PYTHONPATH=src python -m legal_explainer.agents.eval.run_eval \
    --system orchestrated --engine legacy --n 50 --shuffle --seed 42 \
    --concurrency 4 --quiet
```

The `--seed 42` matches v2's default, so both runs see the exact same 50 cases. Rename the v1 output file before running so it isn't overwritten:
```bash
mv reports/agent_eval/predictions_orchestrated.json reports/agent_eval/predictions_v2.json
# then run the legacy command above, which will write a new predictions_orchestrated.json (the v1 result)
```

### 7. Inspect traces
Every orchestrator run appends to `reports/agent_traces/run_YYYYMMDD.jsonl`. One event per step, tagged with `query_id`. Follow one query end-to-end:
```bash
QUERY_ID=$(jq -r '.predictions[0].query_id' reports/agent_eval/predictions_orchestrated.json)
grep "$QUERY_ID" reports/agent_traces/run_*.jsonl | jq
```

---

## Epic mapping

| Epic 6 task | Where it lives |
|---|---|
| 6.1 Tool catalog (glossary, document search, statute lookup) | `tools/glossary.py`, `tools/rag_search.py`, `tools/statute.py`, `tools/web_search.py` |
| 6.2 Strict JSON tool schemas | Each `@tool` decorator includes a usage-discipline docstring telling the model when to call and when not to. |
| 6.3 Invocation flow + structured errors + logging | Orchestrator dispatches, tools return `{found: false, ...}` on misses, `TraceLogger` records every call. |
| 6.4 Demonstrable tool-use scenario | `test_pipeline.ipynb` section 6 — "with vs without glossary" ablation. |

| Epic 7 task | Where it lives |
|---|---|
| 7.1 Baseline two-role pipeline | `orchestrator.run_baseline` |
| 7.2 Query router | `tools/router.py` — rules + LLM fallback |
| 7.3 Unified orchestrated flow | `orchestrator.run_orchestrated` |
| 7.4 Agent-collaboration showcase | `test_pipeline.ipynb` complex-path cell — shows safety → routing → comparator → explainer with intermediate state |
| 7.5 Baseline vs orchestrated comparison | `eval/run_eval.py` + `eval/compare_baseline.py` |

---

## Architecture v2 — LangGraph + LLM router

The original orchestrator (`orchestrator.py`) was a plain Python function that ran safety → router → if/else dispatch in sequence. It worked, but two pain points emerged in real usage:

1. **The rule-based router was brittle.** It worked when the user query matched a regex (article reference, comparison keyword, or a glossary term in a "what is X?" pattern), but it fell back to an LLM classifier for everything else — and that LLM classifier was wired separately from the rule path, making the control flow hard to trace.
2. **State was implicit.** Each subagent's findings, every cost-tracking accumulation, every flag was a local variable in one long function. Adding a new node (e.g. a Reranker, or a citation-verifier) meant another `if/else` branch and another set of locals.

v2 replaces both with **LangGraph** and a **pure LLM router**.

### What changed

| | v1 (legacy) | v2 (LangGraph + LLM) |
|---|---|---|
| Control flow | Plain Python `if/else` in [`orchestrator.py`](orchestrator.py) | State machine compiled from a graph in [`orchestrator_langgraph.py`](orchestrator_langgraph.py) |
| Router | Regex rules first, LLM fallback only on miss ([`tools/router.py`](tools/router.py)) | Single Claude call returns `{complexity, glossary_term_id, reason}` ([`tools/llm_router.py`](tools/llm_router.py)) |
| State management | Local variables in one function | TypedDict `GraphState` carried between nodes — every node returns a partial update |
| Cost tracking | Manual `total_cost += ...` accumulator | Each node returns `total_cost_usd` increment; LangGraph merges |
| Extensibility | Add another `elif` in the dispatch block | `graph.add_node(...) + graph.add_edge(...)` — declarative |

### The v2 graph

Each box is a node — most of them are async functions that wrap one of the existing subagents/tools, so **no subagent code was rewritten**, just glued together differently.

```
        ┌──────────┐
        │  START   │
        └─────┬────┘
              ▼
        ┌──────────┐
        │  safety  │  (rule-based, kept as-is from v1)
        └─────┬────┘
       refuse │ allow
       ┌──────┴────────────────────┐
       ▼                           ▼
  ┌──────────┐              ┌──────────┐
  │ refusal  │              │  router  │  (LLM call — tools/llm_router.py)
  └────┬─────┘              └──────────┘
       │                          │
       │           ┌──────────────┼──────────────┐
       │     simple│      medium  │      complex │
       │           ▼              ▼              ▼
       │      ┌─────────┐  ┌──────────┐  ┌────────────┐
       │      │ glossary│  │researcher│  │ comparator │
       │      └────┬────┘  └─────┬────┘  └─────┬──────┘
       │           │             │              │
       │           │             └──────┬───────┘
       │           │                    ▼
       │           │              ┌──────────┐
       │           │              │ explainer│
       │           │              └─────┬────┘
       │           │                    │
       └───────────┴────────────────────┘
                      ▼
                  ┌──────┐
                  │ END  │
                  └──────┘
```

### The LLM router

The whole router is now this prompt (literal — see [`tools/llm_router.py`](tools/llm_router.py)):

> Classify a legal-explainer query into exactly one of three buckets. Respond with a single JSON object.
> - **simple**: user asks for the definition of ONE specific term that maps to a canonical id in our 15-term glossary.
> - **complex**: user asks to COMPARE / CONTRAST two or more things.
> - **medium**: everything else.
>
> Output: `{"complexity": "simple|medium|complex", "glossary_term_id": "<id or null>", "reason": "<one sentence>"}`

The list of 15 glossary canonical ids is embedded directly in the system prompt so the LLM knows what "simple" actually means. The Python side then **defensively verifies** the returned `glossary_term_id` is in the glossary before honoring it — if the LLM hallucinates an id, the call is downgraded to medium so the simple-path dispatch never tries to look up a fake term (this was the exact bug in v1 that produced `"I don't have a curated definition for 'simple'"`).

One LLM call per query, ~50-150 tokens out. Cost: ~$0.0005 per route.

### When to use which engine

The eval runner takes an `--engine {langgraph,legacy}` flag (default `langgraph`). Both engines produce predictions in the **same JSON shape**, so `compare_baseline.py` and the existing Claude-judge harness work unchanged. See [§ Running it](#running-it) for the exact commands.

### Tradeoffs

| | Pro | Con |
|---|---|---|
| **v2 (LangGraph + LLM)** | Trivial to add new nodes; declarative graph; routing handles paraphrased queries the regex would miss; the LLM can identify glossary terms even when the user phrasing isn't `"what is X?"` | One extra LLM call per query (~$0.0005, ~1-2s); slightly more dependencies (`langgraph`) |
| **v1 (legacy)** | Zero LLM cost on routing; rule-based is fully traceable; no extra deps | Brittle to paraphrasing; the LLM fallback path was awkward to debug |

For the legal-explainer workload (mostly single-article queries on RAFT, lots of bilingual paraphrasing), the v2 routing accuracy gain matters more than the small per-query cost.

---

## Known limitations / things to fix later

- **`addon_params={"language": "Arabic and English"}` for LightRAG** — set in the RAG tool but the original ingest used `"English"`. The graph built by `ingest_rag.ipynb` may under-extract Arabic entities until re-ingested with the bilingual setting.
- **No reranker** — LightRAG warns about this on every query. Adding a Cohere or local cross-encoder reranker would lift retrieval quality.
- **Web search is DuckDuckGo** — fine for v1, but unstable under load and English-biased. If `web_search` ends up being important, swap for Brave (free 2K/mo tier, official MCP).
- **Glossary is 15 terms** — covers the most common Egyptian civil-law concepts, but real coverage requires growing this to ~50 terms.
- **The Explainer has no tools** — by design, so it can't introduce ungrounded facts. Side effect: if the Researcher's JSON is malformed, the Explainer has nothing to work with and the answer degrades. The fallback path could re-call the Researcher with stricter instructions.
- **No reranker for the eval** — quality scoring uses the existing 5-dim Claude judge. Retrieval correctness (`retrieval_hit@k`) is added here, but a full RAGAS-style breakdown (context_precision, faithfulness, answer_relevancy) would be a nice next step.