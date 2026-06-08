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

import os
import threading
from dataclasses import dataclass, field

from . import config


@dataclass
class Reply:
    text: str
    retrieval: list[dict] | None = None  # normalized rows: {article, score, lang, snippet}
    trace: str | None = None             # markdown (agent flow)
    meta: str = ""                       # one-line status under the answer
    # ── Trace enrichment (Langfuse) — all optional, populated per backend ──
    model: str | None = None             # the answer model (HF id / Ollama tag / adapter)
    params: dict | None = None           # generation/sampling params (temp, top_p, max_tokens …)
    usage: dict | None = None            # token usage {input, output, total}
    documents: list[dict] | None = None  # full retrieved chunks (untruncated) for the trace
    info: dict | None = None             # any extra structured detail (keywords, topics, timing …)
    prompt: object | None = None         # the EXACT input sent to the model (str or messages list)


def _ollama_get(resp, key, default=None):
    """Ollama responses may be dicts or pydantic ChatResponse objects."""
    if isinstance(resp, dict):
        return resp.get(key, default)
    return getattr(resp, key, default)


def _ollama_content(obj) -> str:
    """Extract message.content from an ollama chat response / stream chunk,
    whether it's a dict or a pydantic ChatResponse/Message object."""
    msg = _ollama_get(obj, "message", None)
    if msg is None:
        return ""
    if isinstance(msg, dict):
        return msg.get("content", "") or ""
    return getattr(msg, "content", "") or ""


def _ollama_usage(resp) -> dict | None:
    """Pull token usage from an Ollama chat response (if present)."""
    pin = _ollama_get(resp, "prompt_eval_count")
    pout = _ollama_get(resp, "eval_count")
    if pin is None and pout is None:
        return None
    usage = {}
    if pin is not None:
        usage["input"] = int(pin)
    if pout is not None:
        usage["output"] = int(pout)
    if pin is not None and pout is not None:
        usage["total"] = int(pin) + int(pout)
    return usage or None


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
            # Only one model fits in 6 GB VRAM — evict any Ollama model first so
            # the 4-bit load below doesn't OOM against a resident Ollama model.
            _free_ollama_vram()
            import torch
            from peft import PeftModel
            from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                      BitsAndBytesConfig)
            # bf16 needs Ampere+ (RTX 30xx/40xx). On Turing (Colab's T4) bf16 is
            # unsupported → fall back to fp16. Override with LP_COMPUTE_DTYPE.
            _dtype_env = os.getenv("LP_COMPUTE_DTYPE", "").lower()
            if _dtype_env in ("bf16", "bfloat16"):
                compute_dtype = torch.bfloat16
            elif _dtype_env in ("fp16", "float16"):
                compute_dtype = torch.float16
            else:
                compute_dtype = (torch.bfloat16 if torch.cuda.is_available()
                                 and torch.cuda.is_bf16_supported() else torch.float16)
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
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

    def warmup(self) -> None:
        """Load the model and run a tiny generation on BOTH the adapter-on and
        adapter-off paths, so the heavy 4-bit load and the first-call CUDA
        kernel/adapter warm-up are paid here (at startup) rather than on the
        user's first message."""
        import torch
        self._ensure()
        tok, model = self._tok, self._model
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        for use_adapter in (True, False):
            ctx = (_nullcontext() if use_adapter else model.disable_adapter())
            with torch.no_grad(), ctx:
                if use_adapter:
                    model.set_adapter("knowledge")
                model.generate(**inputs, max_new_tokens=4, do_sample=False,
                               pad_token_id=tok.eos_token_id)

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        """Free the 4-bit model from VRAM. Called when the user switches to a
        backend that doesn't use it, so the ~3 GB it holds is returned to the
        6 GB GPU for the next model (Ollama / a RAG service's embedder)."""
        with self._lock:
            if self._model is None:
                return
            import gc

            import torch
            del self._model
            self._model = None
            self._tok = None
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            print("[unified-ui] unloaded local Qwen from VRAM.")

    def generate(self, message: str, use_adapter: bool) -> tuple[str, dict, str]:
        """Closed-book, single-turn (matches how the knowledge model was trained
        and evaluated) — history is intentionally not threaded in.
        Returns (text, usage, prompt): usage = {input, output, total} tokens,
        prompt = the exact rendered chat-template string fed to the model."""
        return self.generate_chat([{"role": "user", "content": message}], use_adapter)

    def generate_chat(self, messages: list[dict], use_adapter: bool) -> tuple[str, dict, str]:
        """Run an arbitrary chat (e.g. system + RAG-grounded user turn) through the
        base model, with the knowledge adapter on or off.
        Returns (text, usage, prompt) for the trace."""
        import torch
        self._ensure()
        tok, model = self._tok, self._model
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        n_in = int(inputs["input_ids"].shape[1])
        gen = dict(**inputs, max_new_tokens=config.MAX_NEW_TOKENS,
                   do_sample=False, repetition_penalty=1.0,
                   pad_token_id=tok.eos_token_id)
        ctx = (model.disable_adapter() if not use_adapter
               else _nullcontext())
        with torch.no_grad(), ctx:
            if use_adapter:
                model.set_adapter("knowledge")
            out = model.generate(**gen)
        n_out = int(out.shape[1]) - n_in
        text = tok.decode(out[0][n_in:], skip_special_tokens=True)
        usage = {"input": n_in, "output": n_out, "total": n_in + n_out}
        return text.strip(), usage, prompt

    def generate_stream(self, messages: list[dict], use_adapter: bool, meta_out: dict):
        """Token-streaming variant: yields the cumulative answer text as the
        model produces tokens. After the generator is exhausted, `meta_out` is
        filled with {"usage", "prompt"} for the trace."""
        import threading
        import torch
        from transformers import TextIteratorStreamer
        self._ensure()
        tok, model = self._tok, self._model
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        n_in = int(inputs["input_ids"].shape[1])
        streamer = TextIteratorStreamer(
            tok, skip_prompt=True, skip_special_tokens=True)
        gen = dict(**inputs, max_new_tokens=config.MAX_NEW_TOKENS,
                   do_sample=False, repetition_penalty=1.0,
                   pad_token_id=tok.eos_token_id, streamer=streamer)

        def _run():
            ctx = (model.disable_adapter() if not use_adapter else _nullcontext())
            with torch.no_grad(), ctx:
                if use_adapter:
                    model.set_adapter("knowledge")
                model.generate(**gen)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        text = ""
        for chunk in streamer:
            text += chunk
            yield text.strip()
        thread.join()
        n_out = int(len(tok(text, return_tensors="pt")["input_ids"][0]))
        meta_out["usage"] = {"input": n_in, "output": n_out, "total": n_in + n_out}
        meta_out["prompt"] = prompt


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _free_ollama_vram() -> None:
    """Ask Ollama to unload every resident model so the 6 GB GPU has room for the
    in-process 4-bit Qwen. Best-effort — never raises (Ollama may be down, or the
    client may lack `ps`)."""
    try:
        import ollama
        client = ollama.Client(host=config.OLLAMA_HOST)
        loaded = client.ps().get("models", [])
        for m in loaded:
            name = m.get("name") or m.get("model")
            if name:
                # keep_alive=0 → unload immediately after this (no-op) call.
                client.generate(model=name, prompt="", keep_alive=0)
        if loaded:
            print(f"[unified-ui] freed Ollama VRAM ({len(loaded)} model(s) unloaded).")
    except Exception as e:
        print(f"[unified-ui] could not free Ollama VRAM (ignored): {e}")


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

    def stream(self, message: str, history: list[dict]):
        """Yield (partial_text, reply). Intermediate yields carry the cumulative
        answer text and `reply=None`; the FINAL yield carries the full text and
        the complete `Reply` (metadata/retrieval/usage). Default: no token
        streaming — call generate() and yield once (used by RAG + agent, whose
        answers come back whole from their services)."""
        reply = self.generate(message, history)
        yield reply.text, reply


class LocalQwenBackend(Backend):
    def __init__(self, id, label, use_adapter, description):
        self.id, self.label, self.use_adapter, self.description = id, label, use_adapter, description
        self.kind = "chat"

    def _reply(self, text, usage, prompt) -> Reply:
        tag = "knowledge adapter ON" if self.use_adapter else "base model (no adapter)"
        adapter_name = config.KNOWLEDGE_ADAPTER_DIR.rstrip("/").split("/")[-1]
        model = config.BASE_QWEN_MODEL + (
            f" + QLoRA knowledge adapter ({adapter_name})"
            if self.use_adapter else " (no adapter)")
        params = {
            "decoding": "greedy",
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": config.MAX_NEW_TOKENS,
            "repetition_penalty": 1.0,
            "quantization": "4-bit nf4 (double-quant)",
        }
        info = {
            "mode": "closed-book",
            "single_turn": True,
            "adapter": self.use_adapter,
            "base_model": config.BASE_QWEN_MODEL,
            "adapter_dir": (config.KNOWLEDGE_ADAPTER_DIR if self.use_adapter else None),
            "prompt_template": "Qwen2.5 chat template (from adapter tokenizer)",
        }
        info = {k: v for k, v in info.items() if v is not None}
        return Reply(
            text=text, meta=f"closed-book · single-turn · {tag}",
            model=model, params=params, usage=usage, prompt=prompt, info=info,
        )

    def generate(self, message, history):
        # Closed-book, single-turn (history intentionally not threaded in).
        text, usage, prompt = _QWEN.generate(message, use_adapter=self.use_adapter)
        return self._reply(text, usage, prompt)

    def stream(self, message, history):
        meta_out: dict = {}
        msgs = [{"role": "user", "content": message}]
        text = ""
        for text in _QWEN.generate_stream(msgs, self.use_adapter, meta_out):
            yield text, None
        yield text, self._reply(text, meta_out.get("usage"), meta_out.get("prompt"))


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

    def _msgs(self, message, history):
        msgs = []
        for t in (history or []):
            if t.get("role") in ("user", "assistant") and t.get("content"):
                msgs.append({"role": t["role"], "content": t["content"]})
        msgs.append({"role": "user", "content": message})
        return msgs

    def _reply(self, text, resp, n_msgs) -> Reply:
        info = {"multi_turn": True, "n_history_turns": n_msgs - 1}
        td = _ollama_get(resp, "total_duration")
        if td:
            info["total_duration_ms"] = round(td / 1e6)
        return Reply(
            text=text.strip(), meta=f"Ollama · {self.model}",
            model=self.model,
            params={"runtime": "ollama", "sampling": "model defaults (temperature≈0.8, top_p=0.9, top_k=40)"},
            usage=_ollama_usage(resp), info=info,
        )

    def generate(self, message, history):
        msgs = self._msgs(message, history)
        resp = self._c().chat(model=self.model, messages=msgs)
        return self._reply(_ollama_content(resp), resp, len(msgs))

    def stream(self, message, history):
        msgs = self._msgs(message, history)
        text, last = "", None
        for part in self._c().chat(model=self.model, messages=msgs, stream=True):
            last = part
            text += _ollama_content(part)
            yield text, None
        yield text, self._reply(text, last, len(msgs))


class PromptDesignBackend(Backend):
    """Project 1 — legal system prompt + deterministic safety/refusal + disclaimer,
    served by any Ollama chat model."""

    def __init__(self, id, label, model, description):
        self.id, self.label, self.model, self.description = id, label, model, description
        self.kind = "chat"
        self._asst = None

    def _ensure(self):
        if self._asst is None:
            from legal_explainer.prompt_design.assistant import PromptDesignAssistant
            self._asst = PromptDesignAssistant(
                ollama_host=config.OLLAMA_HOST, model=self.model)
        return self._asst

    def _reply(self, text, info) -> Reply:
        info = dict(info or {})
        usage = info.pop("usage", None)
        prompt = info.pop("prompt_messages", None)  # exact messages sent to the model
        refused = info.get("refused")
        meta = (f"Ollama · {self.model} · refused by safety gate"
                if refused else f"Ollama · {self.model} · safety-gated")
        return Reply(
            text=text, meta=meta,
            model=self.model,
            params={"runtime": "ollama",
                    "system_prompt": "legal-assistant (Project 1)",
                    "safety_gate": "deterministic refusal + disclaimer injection"},
            usage=usage, prompt=prompt, info=info,
        )

    def generate(self, message, history):
        text, info = self._ensure().answer_meta(message, history=history)
        return self._reply(text, info)

    def stream(self, message, history):
        meta_out: dict = {}
        text = ""
        for text in self._ensure().answer_stream(message, history=history, meta_out=meta_out):
            yield text, None
        yield text, self._reply(text, meta_out)


class HttpRagBackend(Backend):
    """Calls a RAG FastAPI service's /api/v1/ask and normalizes the retrieval."""

    def __init__(self, id, label, url, shape, description):
        self.id, self.label, self.url, self.shape, self.description = id, label, url, shape, description
        self.kind = "rag"

    def _error_reply(self, e) -> Reply:
        import requests
        if isinstance(e, requests.exceptions.HTTPError):
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
        return Reply(
            text=(f"⚠️ Could not reach the **{self.label}** service at "
                  f"`{self.url}`.\n\n```\n{type(e).__name__}: {e}\n```\n\n"
                  f"Start it first (see RUN.md)."),
            retrieval=[], meta="service unavailable",
        )

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
        except Exception as e:
            return self._error_reply(e)
        return self._reply_from_data(data)

    def stream(self, message, history):
        """Token-stream the answer from the service's NDJSON /ask/stream endpoint.
        The first line carries retrieval + generation metadata; subsequent lines
        are answer deltas. Falls back to non-streaming generate() if the service
        has no streaming endpoint (404) or anything goes wrong."""
        import json
        import requests
        try:
            r = requests.post(
                f"{self.url}/api/v1/ask/stream",
                json={"question": message, "top_k": 5},
                stream=True, timeout=config.RAG_HTTP_TIMEOUT,
            )
            if r.status_code == 404:  # older service without streaming — fall back
                reply = self.generate(message, history)
                yield reply.text, reply
                return
            r.raise_for_status()
        except Exception as e:
            reply = self._error_reply(e)
            yield reply.text, reply
            return

        meta_fields: dict = {}
        done_fields: dict = {}
        answer = ""
        try:
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                ev = json.loads(line)
                etype = ev.pop("type", None)
                if etype == "meta":
                    meta_fields = ev
                    # Emit a retrieval "preview" right away so the retrieval
                    # panel populates while the answer is still streaming.
                    preview = self._reply_from_data({**meta_fields, "answer": ""})
                    yield "", preview
                elif etype == "delta":
                    answer += ev.get("text", "")
                    yield answer, None
                elif etype == "done":
                    done_fields = ev
                elif etype == "error":
                    reply = Reply(text=f"⚠️ {self.label}: {ev.get('detail','stream error')}",
                                  retrieval=[], meta="service error")
                    yield reply.text, reply
                    return
        except Exception as e:
            reply = self._error_reply(e)
            yield (answer or reply.text), reply
            return

        data = {**meta_fields, "answer": answer,
                "processing_time_ms": done_fields.get("processing_time_ms", 0)}
        yield answer, self._reply_from_data(data)

    def _reply_from_data(self, data) -> Reply:
        ms = data.get("processing_time_ms", 0)
        # Generation block (model + sampling params) — services added it; older
        # services won't have it, so default to None and degrade gracefully.
        gen = data.get("generation") or {}
        gen_model = gen.get("model")
        gen_params = gen.get("params")
        gen_usage = gen.get("usage")

        if self.shape in ("neo4j", "lgf"):
            articles = data.get("articles", [])
            rows = [{
                "article": a.get("number"),
                "score": round(a.get("score", 0.0), 3),
                "lang": "ar+en",
                "snippet": (a.get("english") or a.get("arabic") or "")[:280],
            } for a in articles]
            # Full, untruncated chunks for the trace.
            documents = [{
                "article": a.get("number"),
                "score": round(a.get("score", 0.0), 4),
                "english": a.get("english") or "",
                "arabic": a.get("arabic") or "",
            } for a in articles]
            rmeta = data.get("metadata") or {}
            retrieval_meta = {
                "strategy": ("graph RAG (BGE-M3 + Neo4j, hybrid keyword+semantic+direct)"
                             if self.shape == "neo4j"
                             else "graph RAG → finetuned answer (Project 6)"),
                "top_k_requested": 5,
                "n_retrieved": len(rows),
                "articles": [a.get("number") for a in articles],
                "keywords_en": rmeta.get("keywords_en"),
                "keywords_ar": rmeta.get("keywords_ar"),
                "legal_topics": rmeta.get("legal_topics"),
                "search_query": rmeta.get("search_query"),
            }
            info = {"processing_time_ms": ms}
            if self.shape == "lgf":
                info["refused"] = bool(data.get("refused"))
                meta = ("refused by safety gate" if data.get("refused")
                        else f"{ms} ms · {len(rows)} articles · finetuned 3B")
            else:
                meta = f"{ms} ms · {len(rows)} articles"
        else:  # bilingual
            hits = data.get("hits", [])
            rows = [{
                "article": h.get("article_number"),
                "score": round(h.get("rerank_score") if h.get("rerank_score") is not None
                               else h.get("score", 0.0), 3),
                "lang": h.get("language", ""),
                "snippet": (h.get("text") or "")[:280],
            } for h in hits]
            documents = [{
                "article": h.get("article_number"),
                "language": h.get("language", ""),
                "score": round(h.get("score", 0.0), 4),
                "rerank_score": (round(h["rerank_score"], 4)
                                 if h.get("rerank_score") is not None else None),
                "section_path": h.get("section_path", ""),
                "text": h.get("text") or "",
            } for h in hits]
            kw = ", ".join(data.get("keywords") or [])
            retrieval_meta = {
                "strategy": "BGE-M3 dense + Chroma + multilingual cross-encoder rerank",
                "top_k_requested": 5,
                "n_retrieved": len(rows),
                "reranked": any(h.get("rerank_score") is not None for h in hits),
                "keywords": data.get("keywords"),
                "article_numbers": data.get("article_numbers"),
                "search_query": data.get("search_query"),
                "detected_language": data.get("detected_language"),
            }
            info = {"processing_time_ms": ms, "detected_language": data.get("detected_language")}
            meta = f"{ms} ms · {len(rows)} chunks · kw: {kw}"
        # Drop empty keys so the trace stays readable.
        retrieval_meta = {k: v for k, v in retrieval_meta.items() if v}
        return Reply(
            text=data.get("answer", ""), retrieval=rows, meta=meta,
            model=gen_model, params=gen_params, usage=gen_usage,
            documents=documents, info={**info, "retrieval": retrieval_meta},
        )


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
        path = getattr(result, "path", "?")
        duration_ms = getattr(result, "duration_ms", 0)
        cost = getattr(result, "total_cost_usd", 0.0)
        meta = (f"path=`{path}` · {duration_ms / 1000:.1f}s · ${cost:.4f}")
        # Surface the orchestration flow (router/subagents/tools) into the trace.
        tools = [e.get("tool") for e in flow.events if e.get("type") == "tool_call"]
        subagents = [e.get("agent") for e in flow.events if e.get("type") == "subagent_start"]
        info = {
            "path": path,
            "duration_ms": duration_ms,
            "cost_usd": round(cost, 6) if cost else None,
            "complexity": next((e.get("complexity") for e in flow.events
                                if e.get("type") == "route"), None),
            "safety_verdict": next((e.get("verdict") for e in flow.events
                                    if e.get("type") == "safety"), None),
            "subagents": subagents,
            "tools_called": tools,
            "n_flow_events": len(flow.events),
        }
        info = {k: v for k, v in info.items() if v not in (None, [], "")}
        return Reply(
            text=answer, trace=trace, meta=meta,
            model="LangGraph orchestrator (glm-5 gateway)",
            params={"orchestration": "safety → router → subagents → tools → synthesis"},
            info=info,
        )


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
