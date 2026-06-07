"""System adapters — turn a question into a normalized prediction.

Each system exposes `.predict(question) -> Prediction` with a uniform shape so the
runner and metrics are system-agnostic:

  answer              the generated answer text
  contexts            retrieved passages used as grounding (str list)
  retrieved_articles  ordered article numbers retrieved (int list)
  closed_book         True for systems with no retrieval (the fine-tune)
  elapsed_ms          wall-clock for the call

  * GraphRAGSystem     → POST {graph api}/api/v1/ask   (apps/api, default :8000)
  * BilingualRAGSystem → POST {brag api}/api/v1/ask    (apps/bilingual_rag, :8100)
  * FinetuneSystem     → local QLoRA adapter, closed-book generation
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Prediction:
    answer: str
    contexts: list[str] = field(default_factory=list)
    retrieved_articles: list[int] = field(default_factory=list)
    closed_book: bool = False
    elapsed_ms: int = 0
    error: str | None = None


def _dedup(seq: list[int]) -> list[int]:
    seen, out = set(), []
    for x in seq:
        if x is not None and x not in seen:
            seen.add(x)
            out.append(int(x))
    return out


# ── HTTP RAG services ─────────────────────────────────────────────────────────

class _HttpSystem:
    name = "http"

    def __init__(self, base_url: str, top_k: int = 5, timeout: int = 180) -> None:
        self.base_url = base_url.rstrip("/")
        self.top_k = top_k
        self.timeout = timeout

    def _post(self, payload: dict) -> dict:
        import requests
        r = requests.post(f"{self.base_url}/api/v1/ask", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()


class GraphRAGSystem(_HttpSystem):
    name = "graph-rag"

    def predict(self, question: str) -> Prediction:
        t0 = time.perf_counter()
        try:
            data = self._post({"question": question, "top_k": self.top_k})
        except Exception as e:  # noqa: BLE001 - surface as a row-level error
            return Prediction(answer="", error=str(e),
                              elapsed_ms=round((time.perf_counter() - t0) * 1000))
        arts = data.get("articles", [])
        contexts = [
            "\n".join(p for p in (a.get("english"), a.get("arabic")) if p)
            for a in arts
        ]
        return Prediction(
            answer=data.get("answer", ""),
            contexts=contexts,
            retrieved_articles=_dedup([a.get("number") for a in arts]),
            elapsed_ms=round((time.perf_counter() - t0) * 1000),
        )


class BilingualRAGSystem(_HttpSystem):
    name = "bilingual-rag"

    def predict(self, question: str) -> Prediction:
        t0 = time.perf_counter()
        try:
            data = self._post({"question": question, "top_k": self.top_k})
        except Exception as e:  # noqa: BLE001
            return Prediction(answer="", error=str(e),
                              elapsed_ms=round((time.perf_counter() - t0) * 1000))
        hits = data.get("hits", [])
        return Prediction(
            answer=data.get("answer", ""),
            contexts=[h.get("text", "") for h in hits],
            retrieved_articles=_dedup([h.get("article_number") for h in hits]),
            elapsed_ms=round((time.perf_counter() - t0) * 1000),
        )


# ── Local fine-tuned adapter (closed-book) ────────────────────────────────────

class FinetuneSystem:
    """Closed-book generation from a QLoRA adapter (4-bit base).

    Mirrors the hardened decoding used by scripts/eval_csv_closedbook.py
    (low temp + repetition penalty + no-repeat-ngram) to avoid the bullet-loop
    degeneration. No retrieval: `contexts` is left empty here and the runner
    injects the gold article text as the grounding the answer should recall."""

    name = "finetuned"

    def __init__(self, adapter_dir: str, base_model: str | None = None,
                 max_new_tokens: int = 256, temperature: float = 0.2,
                 top_p: float = 0.85, repetition_penalty: float = 1.15,
                 no_repeat_ngram_size: int = 3, language_tag: bool = True) -> None:
        self.adapter_dir = adapter_dir
        self.base_model = base_model
        self.gen = dict(max_new_tokens=max_new_tokens, temperature=temperature,
                        top_p=top_p, repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size)
        self.language_tag = language_tag
        self._model = None
        self._tok = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer, BitsAndBytesConfig
        print(f"[eval] loading adapter {self.adapter_dir} (4-bit) …")
        tok = AutoTokenizer.from_pretrained(self.adapter_dir, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        self._model = AutoPeftModelForCausalLM.from_pretrained(
            self.adapter_dir, quantization_config=bnb, device_map="auto",
            trust_remote_code=True)
        self._model.eval()
        self._tok = tok

    def predict(self, question: str) -> Prediction:
        import torch
        from .text_utils import detect_language
        self._load()
        q = question
        if self.language_tag:
            q = ("[AR] " if detect_language(question) == "ar" else "[EN] ") + question
        chat = [{"role": "user", "content": q}]
        prompt = self._tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self._tok(prompt, return_tensors="pt").to(self._model.device)
        plen = inputs["input_ids"].shape[1]
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=self.gen["max_new_tokens"],
                do_sample=self.gen["temperature"] > 0,
                temperature=max(self.gen["temperature"], 1e-5), top_p=self.gen["top_p"],
                repetition_penalty=self.gen["repetition_penalty"],
                no_repeat_ngram_size=self.gen["no_repeat_ngram_size"],
                pad_token_id=(self._tok.pad_token_id or self._tok.eos_token_id),
            )
        ans = self._tok.decode(out[0, plen:], skip_special_tokens=True).strip()
        return Prediction(answer=ans, closed_book=True,
                          elapsed_ms=round((time.perf_counter() - t0) * 1000))


def build_system(kind: str, **kw):
    if kind == "graph-rag":
        return GraphRAGSystem(kw["base_url"], top_k=kw.get("top_k", 5))
    if kind == "bilingual-rag":
        return BilingualRAGSystem(kw["base_url"], top_k=kw.get("top_k", 5))
    if kind == "finetuned":
        return FinetuneSystem(kw["adapter_dir"], base_model=kw.get("base_model"))
    raise ValueError(f"unknown system kind: {kind!r}")
