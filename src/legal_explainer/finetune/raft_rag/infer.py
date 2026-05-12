"""RAFT-RAG inference pipeline:  retrieve  ->  RAFT-format prompt  ->  generate.

Generation uses hardened decoding params (repetition penalty, no-repeat-ngram,
low temperature) — the fix for the bullet-loops / runaway-repetition / stray
non-Latin-script degeneration seen in earlier runs.

The RAFT adapter can have been trained with the plain TRL pipeline *or* with
Unsloth — either way it is a standard PEFT adapter directory and loads here via
``AutoPeftModelForCausalLM`` (base re-loaded in 4-bit). The base model
(Qwen2.5-1.5B / 3B Instruct) is read from the adapter's config.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .index import PROJECT_ROOT, detect_lang
from .prompt import build_prompt
from .retriever import HybridRetriever, RetrievalResult


@dataclass
class GenConfig:
    adapter_dir: str
    max_new_tokens: int = 384
    temperature: float = 0.2
    top_p: float = 0.85
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 3


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load_adapter(adapter_dir):
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer, BitsAndBytesConfig

    p = _abs(adapter_dir)
    if not p.exists():
        raise FileNotFoundError(
            f"RAFT adapter not found at {p}. Train it first (e.g. "
            f"`python scripts/train_unsloth.py --config src/legal_explainer/finetune/configs/qlora_qwen3b_raft.yaml`) "
            f"or set generation.adapter_dir in the raft_rag config to an existing adapter."
        )
    tok = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(p), quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model.eval()
    return model, tok


class RaftRagPipeline:
    def __init__(self, retriever: HybridRetriever, model, tokenizer, gen: GenConfig, *,
                 k: int = 1, n_distractors: int = 1, max_article_chars: int = 900,
                 shuffle_seed: int | None = 13):
        self.retriever = retriever
        self.model = model
        self.tok = tokenizer
        self.gen = gen
        self.k = k
        self.n_distractors = n_distractors
        self.max_article_chars = max_article_chars
        self.shuffle_seed = shuffle_seed

    # --- factory -----------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict, *, load_model: bool = True) -> "RaftRagPipeline":
        rc = cfg.get("retrieval", {}) or {}
        pc = cfg.get("prompt", {}) or {}
        gc = cfg.get("generation", {}) or {}
        retr = HybridRetriever.load(
            _abs(cfg.get("artifacts_dir", "artifacts/raft_rag")),
            rrf_k=rc.get("rrf_k", 60), candidate_pool=rc.get("candidate_pool", 20),
        )
        gen = GenConfig(
            adapter_dir=gc.get("adapter_dir", "runs/qlora-qwen2.5-3b-raft"),
            max_new_tokens=gc.get("max_new_tokens", 384),
            temperature=gc.get("temperature", 0.2),
            top_p=gc.get("top_p", 0.85),
            repetition_penalty=gc.get("repetition_penalty", 1.15),
            no_repeat_ngram_size=gc.get("no_repeat_ngram_size", 3),
        )
        model, tok = (_load_adapter(gen.adapter_dir) if load_model else (None, None))
        return cls(retr, model, tok, gen,
                   k=rc.get("k", 1), n_distractors=rc.get("n_distractors", 1),
                   max_article_chars=pc.get("max_article_chars", 900),
                   shuffle_seed=pc.get("shuffle_seed", 13))

    # --- generation --------------------------------------------------------
    def _generate(self, question: str, lang: str, result: RetrievalResult) -> dict:
        import torch
        prompt = build_prompt(question, result, lang,
                              max_article_chars=self.max_article_chars, shuffle_seed=self.shuffle_seed)
        chat = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        plen = inputs["input_ids"].shape[1]
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.gen.max_new_tokens,
                do_sample=self.gen.temperature > 0,
                temperature=max(self.gen.temperature, 1e-5),
                top_p=self.gen.top_p,
                repetition_penalty=self.gen.repetition_penalty,
                no_repeat_ngram_size=self.gen.no_repeat_ngram_size,
                pad_token_id=(self.tok.pad_token_id or self.tok.eos_token_id),
            )
        dt = time.perf_counter() - t0
        answer = self.tok.decode(out[0, plen:], skip_special_tokens=True).strip()
        return {"answer": answer, "prompt": prompt, "gen_seconds": round(dt, 2)}

    # --- public ------------------------------------------------------------
    def ask(self, question: str, *, lang: str | None = None, mode: str = "rag",
            force_key: str | None = None) -> dict:
        """mode: 'rag' (use the retriever), 'closed' (no context — closed-book),
        'oracle' (force `force_key` into the context — measures adapter ceiling)."""
        lang = lang or detect_lang(question)
        if mode == "closed":
            result = RetrievalResult(oracle=[], distractors=[])
        elif mode == "oracle":
            if not force_key:
                raise ValueError("mode='oracle' needs force_key")
            result = self.retriever.force(force_key, n_distractors=self.n_distractors)
        else:
            result = self.retriever.retrieve(question, lang=lang, k=self.k, n_distractors=self.n_distractors)
        rec = {
            "question": question, "lang": lang, "mode": mode,
            "retrieved_oracle": [e.key for e in result.oracle],
            "retrieved_distractors": [e.key for e in result.distractors],
            "explicit_id": result.explicit_id,
        }
        if self.model is None:                            # retrieval-only (no GPU/model loaded)
            rec["answer"] = None
            return rec
        rec.update(self._generate(question, lang, result))
        return rec
