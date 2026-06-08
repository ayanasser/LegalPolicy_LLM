"""Importable wrapper around the Prompt Design assistant (Project 1).

The original implementation lives in `src/Prompt Design/legal_policy_assistant_egypt_v2.py`,
a folder whose name contains a space and so can't be imported as a normal module.
This wrapper loads that file by path and re-exports the pieces the unified UI
needs — keeping the big SYSTEM_PROMPT, refusal patterns, and disclaimer logic as a
single source of truth — plus a non-interactive `PromptDesignAssistant` that
returns a full answer string (the original ships only a streaming CLI loop).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC = _PROJECT_ROOT / "src" / "Prompt Design" / "legal_policy_assistant_egypt_v2.py"


def _load_original():
    spec = importlib.util.spec_from_file_location("_prompt_design_v2", _SRC)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Prompt Design module from {_SRC}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # safe: CLI is guarded by __name__ == "__main__"
    return mod


_pd = _load_original()

# Re-exported single-source-of-truth constants / helpers.
SYSTEM_PROMPT = _pd.SYSTEM_PROMPT
PROMPT_VERSION = _pd.PROMPT_VERSION
detect_refusal_needed = _pd.detect_refusal_needed
select_disclaimer = _pd.select_disclaimer
REFUSAL_MESSAGES = _pd.REFUSAL_MESSAGES


class PromptDesignAssistant:
    """Egyptian Civil Law explainer: safety gate → disclaimer injection → Ollama chat.

    Mirrors the original `LegalAssistant.chat` flow but returns the full answer
    (no terminal streaming), so it drops cleanly into a Gradio backend.
    """

    def __init__(self, ollama_host: str = "http://localhost:11434",
                 model: str = "llama3.2:3b") -> None:
        import ollama
        self._client = ollama.Client(host=ollama_host)
        self.model = model

    def answer(self, message: str, history: list[dict] | None = None) -> str:
        return self.answer_meta(message, history)[0]

    # ── Shared prep ─────────────────────────────────────────────────────────
    def _prepare(self, message, history):
        """Run the safety gate and build the chat messages. Returns either
        ('refusal', text, meta) or ('chat', messages, meta_base)."""
        message = (message or "").strip()
        if not message:
            return "refusal", "Please ask a question about Egyptian civil law.", {"status": "empty_question"}

        category = detect_refusal_needed(message)
        if category:
            return "refusal", REFUSAL_MESSAGES[category], {
                "refused": True, "refusal_category": category,
                "safety_gate": "triggered", "model": self.model,
                "note": "deterministic refusal — model not called",
            }

        disclaimer = select_disclaimer(message)
        augmented = (
            f"{message}\n\n"
            f"[SYSTEM NOTE: End your response with this exact disclaimer on a new "
            f"line after '---':\n{disclaimer}]"
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in (history or []):
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": augmented})
        meta_base = {
            "refused": False, "safety_gate": "passed", "model": self.model,
            "disclaimer": disclaimer, "n_history_turns": len(messages) - 2,
            "system_prompt": SYSTEM_PROMPT, "prompt_messages": messages,
        }
        return "chat", messages, meta_base

    @staticmethod
    def _usage(resp) -> dict | None:
        g = (lambda k: resp.get(k) if isinstance(resp, dict) else getattr(resp, k, None))
        pin, pout = g("prompt_eval_count"), g("eval_count")
        if pin is None and pout is None:
            return None
        return {
            **({"input": int(pin)} if pin is not None else {}),
            **({"output": int(pout)} if pout is not None else {}),
            **({"total": int(pin) + int(pout)} if (pin is not None and pout is not None) else {}),
        }

    @staticmethod
    def _content(resp) -> str:
        if isinstance(resp, dict):
            return resp["message"]["content"]
        return resp.message.content

    def answer_meta(self, message: str, history: list[dict] | None = None) -> tuple[str, dict]:
        """Like :meth:`answer` but also returns a metadata dict describing the
        safety verdict, the injected disclaimer, the model, and token usage —
        used to enrich observability traces."""
        kind, a, meta = self._prepare(message, history)
        if kind == "refusal":
            return a, meta
        resp = self._client.chat(model=self.model, messages=a)
        usage = self._usage(resp)
        if usage:
            meta["usage"] = usage
        return self._content(resp).strip(), meta

    def answer_stream(self, message: str, history: list[dict] | None = None,
                      meta_out: dict | None = None):
        """Token-streaming variant: yields the cumulative answer text. After the
        generator is exhausted, `meta_out` holds the same metadata dict that
        :meth:`answer_meta` returns. Deterministic refusals yield once."""
        meta_out = meta_out if meta_out is not None else {}
        kind, a, meta = self._prepare(message, history)
        if kind == "refusal":
            meta_out.update(meta)
            yield a
            return
        text, last = "", None
        for part in self._client.chat(model=self.model, messages=a, stream=True):
            last = part
            text += self._content(part) or ""
            yield text
        usage = self._usage(last)
        if usage:
            meta["usage"] = usage
        meta_out.update(meta)
