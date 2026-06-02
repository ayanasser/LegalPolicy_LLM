"""Backend adapters for the Unified UI.

Every project is exposed through one uniform interface::

    backend.generate(message, history) -> Reply(text, retrieval?, trace?, meta)

so the Gradio layer never has to special-case a project. Heavy resources
(the local Qwen, Ollama, RAG services, the agent graph) are loaded lazily on
first use, so importing this module is cheap and selecting a backend you don't
use costs nothing.

Backends
  1. baseline-qwen        Qwen2.5-3B-Instruct (HF, no adapter, closed-book)
  2. baseline-llama       Llama-3.2-3B (Ollama, raw)
  3. finetuned-knowledge  Qwen2.5-3B + knowledge QLoRA adapter (HF, closed-book)
  4. prompt-design        Llama-3.2-3B + legal system prompt + safety (Project 1)
  5. neo4j-rag            Neo4j Graph RAG service over HTTP        → retrieval panel
  6. bilingual-rag        Bilingual RAG service over HTTP          → retrieval panel
  7. multi-agent          LangGraph multi-agent orchestrator       → trace panel

Backends 1 & 3 share a single 4-bit base-model load via a PEFT adapter toggle.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from . import config


@dataclass
class Reply:
    text: str
    retrieval: list[dict] | None = None  # normalized rows: {article, score, lang, snippet}
    trace: str | None = None             # markdown (agent flow)
    meta: str = ""                       # one-line status under the answer


# ─────────────────────────────────────────────────────────────────────────────
# Shared local Qwen (base + knowledge adapter, one 4-bit load)
# ─────────────────────────────────────────────────────────────────────────────

class _LocalQwenManager:
    """Loads Qwen2.5-3B once in 4-bit and attaches the knowledge adapter.
    Baseline = adapter disabled; Finetuned = adapter enabled. Single GPU copy."""

    def __init__(self) -> None:
        self._tok = None
        self._model = None
        self._lock = threading.Lock()

    def _ensure(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from peft import PeftModel
            from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                      BitsAndBytesConfig)
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
            )
            # Tokenizer from the adapter dir (carries the trained chat template).
            tok = AutoTokenizer.from_pretrained(config.KNOWLEDGE_ADAPTER_DIR,
                                                trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            base = AutoModelForCausalLM.from_pretrained(
                config.BASE_QWEN_MODEL, quantization_config=bnb,
                device_map="auto", trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(
                base, config.KNOWLEDGE_ADAPTER_DIR, adapter_name="knowledge",
            )
            model.eval()
            self._tok, self._model = tok, model

    def generate(self, message: str, use_adapter: bool) -> str:
        """Closed-book, single-turn (matches how the knowledge model was trained
        and evaluated) — history is intentionally not threaded in."""
        return self.generate_chat([{"role": "user", "content": message}], use_adapter)

    def generate_chat(self, messages: list[dict], use_adapter: bool) -> str:
        """Run an arbitrary chat (e.g. system + RAG-grounded user turn) through the
        base model, with the knowledge adapter on or off."""
        import torch
        self._ensure()
        tok, model = self._tok, self._model
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        gen = dict(**inputs, max_new_tokens=config.MAX_NEW_TOKENS,
                   do_sample=False, repetition_penalty=1.0,
                   pad_token_id=tok.eos_token_id)
        ctx = (model.disable_adapter() if not use_adapter
               else _nullcontext())
        with torch.no_grad(), ctx:
            if use_adapter:
                model.set_adapter("knowledge")
            out = model.generate(**gen)
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


_QWEN = _LocalQwenManager()


# ─────────────────────────────────────────────────────────────────────────────
# Backend classes
# ─────────────────────────────────────────────────────────────────────────────

class Backend:
    id: str = ""
    label: str = ""
    kind: str = "chat"        # "chat" | "rag" | "agent"
    description: str = ""

    def generate(self, message: str, history: list[dict]) -> Reply:
        raise NotImplementedError


class LocalQwenBackend(Backend):
    def __init__(self, id, label, use_adapter, description):
        self.id, self.label, self.use_adapter, self.description = id, label, use_adapter, description
        self.kind = "chat"

    def generate(self, message, history):
        text = _QWEN.generate(message, use_adapter=self.use_adapter)
        tag = "knowledge adapter ON" if self.use_adapter else "base model (no adapter)"
        return Reply(text=text, meta=f"closed-book · single-turn · {tag}")


class OllamaChatBackend(Backend):
    """Raw Ollama chat — used for the Llama baseline."""

    def __init__(self, id, label, model, description):
        self.id, self.label, self.model, self.description = id, label, model, description
        self.kind = "chat"
        self._client = None

    def _c(self):
        if self._client is None:
            import ollama
            self._client = ollama.Client(host=config.OLLAMA_HOST)
        return self._client

    def generate(self, message, history):
        msgs = []
        for t in (history or []):
            if t.get("role") in ("user", "assistant") and t.get("content"):
                msgs.append({"role": t["role"], "content": t["content"]})
        msgs.append({"role": "user", "content": message})
        resp = self._c().chat(model=self.model, messages=msgs)
        return Reply(text=resp["message"]["content"].strip(), meta=f"Ollama · {self.model}")


class PromptDesignBackend(Backend):
    """Project 1 — legal system prompt + deterministic safety/refusal + disclaimer,
    served by any Ollama chat model."""

    def __init__(self, id, label, model, description):
        self.id, self.label, self.model, self.description = id, label, model, description
        self.kind = "chat"
        self._asst = None

    def generate(self, message, history):
        if self._asst is None:
            from legal_explainer.prompt_design.assistant import PromptDesignAssistant
            self._asst = PromptDesignAssistant(
                ollama_host=config.OLLAMA_HOST, model=self.model)
        text = self._asst.answer(message, history=history)
        return Reply(text=text, meta=f"Ollama · {self.model} · safety-gated")


class HttpRagBackend(Backend):
    """Calls a RAG FastAPI service's /api/v1/ask and normalizes the retrieval."""

    def __init__(self, id, label, url, shape, description):
        self.id, self.label, self.url, self.shape, self.description = id, label, url, shape, description
        self.kind = "rag"

    def generate(self, message, history):
        import requests
        try:
            r = requests.post(
                f"{self.url}/api/v1/ask",
                json={"question": message, "top_k": 5},
                timeout=config.RAG_HTTP_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
        except requests.exceptions.HTTPError as e:
            # Service is up but the pipeline errored (e.g. CUDA OOM, Neo4j drop).
            # Surface the server's {"detail": ...} body — raise_for_status() hides it.
            detail = ""
            try:
                detail = (e.response.json() or {}).get("detail", "")
            except Exception:
                detail = (e.response.text or "")[:500]
            code = e.response.status_code if e.response is not None else "?"
            return Reply(
                text=(f"⚠️ The **{self.label}** service is running but returned "
                      f"an error (HTTP {code}).\n\n```\n{detail or e}\n```\n\n"
                      f"On a 6 GB GPU this is usually VRAM contention — the answer "
                      f"model couldn't load alongside the other models. Try again, "
                      f"or free GPU memory (see RUN.md)."),
                retrieval=[], meta=f"service error (HTTP {code})",
            )
        except Exception as e:
            return Reply(
                text=(f"⚠️ Could not reach the **{self.label}** service at "
                      f"`{self.url}`.\n\n```\n{type(e).__name__}: {e}\n```\n\n"
                      f"Start it first (see RUN.md)."),
                retrieval=[], meta="service unavailable",
            )
        if self.shape in ("neo4j", "lgf"):
            rows = [{
                "article": a.get("number"),
                "score": round(a.get("score", 0.0), 3),
                "lang": "ar+en",
                "snippet": (a.get("english") or a.get("arabic") or "")[:280],
            } for a in data.get("articles", [])]
            if self.shape == "lgf":
                meta = ("refused by safety gate" if data.get("refused")
                        else f"{data.get('processing_time_ms', 0)} ms · {len(rows)} articles · finetuned 3B")
            else:
                meta = f"{data.get('processing_time_ms', 0)} ms · {len(rows)} articles"
        else:  # bilingual
            rows = [{
                "article": h.get("article_number"),
                "score": round(h.get("rerank_score") if h.get("rerank_score") is not None
                               else h.get("score", 0.0), 3),
                "lang": h.get("language", ""),
                "snippet": (h.get("text") or "")[:280],
            } for h in data.get("hits", [])]
            kw = ", ".join(data.get("keywords") or [])
            meta = f"{data.get('processing_time_ms', 0)} ms · {len(rows)} chunks · kw: {kw}"
        return Reply(text=data.get("answer", ""), retrieval=rows, meta=meta)


class AgentBackend(Backend):
    def __init__(self):
        self.id, self.label, self.kind = "multi-agent", "Multi-Agent · LangGraph orchestrator", "agent"
        self.description = "Safety → router → subagents → tools → synthesis (Project 5)."

    def generate(self, message, history):
        import asyncio
        import uuid
        from legal_explainer.agents.flow import FlowLogger
        from legal_explainer.agents.orchestrator_langgraph import run_orchestrated_lg

        flow = FlowLogger(query_id=uuid.uuid4().hex[:12])
        try:
            result = asyncio.run(run_orchestrated_lg(message, flow_logger=flow))
        except Exception as e:
            return Reply(text=f"### Agent error\n```\n{type(e).__name__}: {e}\n```",
                         trace="_(run failed before producing a trace)_",
                         meta="agent error")
        answer = getattr(result, "answer", None) or flow.answer_text or "_(no answer)_"
        trace = _render_agent_trace(flow.events)
        meta = (f"path=`{getattr(result, 'path', '?')}` · "
                f"{getattr(result, 'duration_ms', 0) / 1000:.1f}s · "
                f"${getattr(result, 'total_cost_usd', 0.0):.4f}")
        return Reply(text=answer, trace=trace, meta=meta)


def _render_agent_trace(events: list[dict]) -> str:
    if not events:
        return "_No flow events recorded._"
    lines = []
    for e in events:
        t = e.get("type", "?")
        if t == "safety":
            lines.append(f"🛡️ **Safety**: `{e.get('verdict','?')}`")
        elif t == "route":
            lines.append(f"🧭 **Router**: complexity = `{e.get('complexity','?')}`")
        elif t == "subagent_start":
            lines.append(f"🤖 **{str(e.get('agent','?')).title()}** — started")
        elif t == "subagent_done":
            lines.append(f"✅ **{str(e.get('agent','?')).title()}** — done "
                         f"(`{e.get('duration_ms',0)/1000:.1f}s`)")
        elif t == "tool_call":
            lines.append(f"&nbsp;&nbsp;🔧 `{e.get('tool','?')}`")
        elif t == "done":
            lines.append(f"🏁 **Done** — path=`{e.get('path','?')}`")
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

def build_registry() -> dict[str, Backend]:
    backends: list[Backend] = [
        LocalQwenBackend("baseline-qwen", "Baseline · Qwen2.5-3B-Instruct (no finetune)",
                         use_adapter=False,
                         description="Raw Qwen2.5-3B-Instruct, closed-book — the comparison baseline."),
        OllamaChatBackend("baseline-llama", "Baseline · Llama-3.2-3B (Ollama)",
                          config.LLAMA_BASELINE_MODEL,
                          description="Raw Llama-3.2-3B via Ollama — a second baseline."),
        LocalQwenBackend("finetuned-knowledge", "Finetuned · Qwen2.5-3B Knowledge adapter",
                         use_adapter=True,
                         description="QLoRA adapter that memorised the Egyptian Civil Code (Project 2)."),
        PromptDesignBackend("prompt-design-llama", "Prompt Design · Llama-3.2-3B + legal prompt",
                            config.PROMPT_DESIGN_MODEL,
                            description="System prompt + deterministic safety/refusal + disclaimer (Project 1)."),
        PromptDesignBackend("prompt-design-qwen", "Prompt Design · Qwen2.5-3B-Instruct + legal prompt",
                            config.PROMPT_DESIGN_QWEN_MODEL,
                            description="Same legal prompt + safety as Project 1, served by Qwen2.5-3B-Instruct."),
        HttpRagBackend("neo4j-rag", "Neo4j Graph RAG", config.NEO4J_RAG_URL, "neo4j",
                       description="BGE-M3 + Neo4j graph + Qwen3 (Project 3)."),
        HttpRagBackend("bilingual-rag", "Bilingual RAG (Chroma)", config.BILINGUAL_RAG_URL, "bilingual",
                       description="BGE-M3 + Chroma + rerank + Qwen (Project 4)."),
        HttpRagBackend("combined-legal-graphrag", "Combined · Legal prompt + Graph RAG + Finetuned 3B",
                       config.LGF_RAG_URL, "lgf",
                       description="Project 6 service (:8200): legal prompt + safety → graph RAG → finetuned answer."),
        AgentBackend(),
    ]
    return {b.label: b for b in backends}


REGISTRY = build_registry()
DEFAULT_LABEL = next(iter(REGISTRY))  # baseline-qwen
