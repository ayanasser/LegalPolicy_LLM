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
        message = (message or "").strip()
        if not message:
            return "Please ask a question about Egyptian civil law."

        # 1. Safety gate — deterministic refusal before hitting the model.
        category = detect_refusal_needed(message)
        if category:
            return REFUSAL_MESSAGES[category]

        # 2. Inject the appropriate disclaimer instruction.
        disclaimer = select_disclaimer(message)
        augmented = (
            f"{message}\n\n"
            f"[SYSTEM NOTE: End your response with this exact disclaimer on a new "
            f"line after '---':\n{disclaimer}]"
        )

        # 3. Build messages = system + prior turns + augmented user turn.
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in (history or []):
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": augmented})

        resp = self._client.chat(model=self.model, messages=messages)
        return resp["message"]["content"].strip()
