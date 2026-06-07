"""Finetuned-model generator for Project 6 — self-contained.

Loads Qwen2.5-3B-Instruct in 4-bit with the knowledge QLoRA adapter and answers
an arbitrary chat (system + RAG-grounded user turn). Lazy + thread-safe.
"""
from __future__ import annotations

import threading


class FinetunedGenerator:
    def __init__(self, base_model: str, adapter_dir: str, max_new_tokens: int = 600) -> None:
        self.base_model = base_model
        self.adapter_dir = adapter_dir
        self.max_new_tokens = max_new_tokens
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
            tok = AutoTokenizer.from_pretrained(self.adapter_dir, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model, quantization_config=bnb,
                device_map="auto", trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base, self.adapter_dir)
            model.eval()
            self._tok, self._model = tok, model

    def generate_chat(self, messages: list[dict]) -> str:
        import torch
        self._ensure()
        tok, model = self._tok, self._model
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, repetition_penalty=1.0,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
