"""Pluggable LLM judge for RAG evaluation.

Two backends, per the project's no-API-billing constraint:

  * **ollama**       — fully automated; a local model (default Qwen 2.5 3B) scores
                       every row programmatically. Cheap, offline, reproducible.
  * **claude-code**  — file handoff; the runner writes the rendered judge prompts
                       to a `*.judge_tasks.jsonl`, you (Claude Code) score them in
                       this session and write `*.judge_verdicts.jsonl`, then the
                       report phase ingests them. Higher-quality grading, zero API.

A "verdict" is a dict with the five keys in `prompts.LLM_METRICS`, each a float in
[0, 1], plus an optional "notes" string.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .prompts import JUDGE_SYSTEM, LLM_METRICS, build_judge_user_prompt

# ── Verdict parsing ───────────────────────────────────────────────────────────

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdict(raw: str) -> dict:
    """Best-effort extraction of the JSON verdict from a judge's reply.

    Missing/garbage scores become None so they are excluded from aggregates
    rather than silently counted as zero."""
    out: dict = {m: None for m in LLM_METRICS}
    out["notes"] = ""
    m = _JSON_RE.search(raw or "")
    if not m:
        return out
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out
    for k in LLM_METRICS:
        v = data.get(k)
        if isinstance(v, (int, float)):
            out[k] = max(0.0, min(1.0, float(v)))
    out["notes"] = str(data.get("notes", ""))[:300]
    return out


# ── Judge task (one per row) ──────────────────────────────────────────────────

@dataclass
class JudgeTask:
    row_id: str
    system_prompt: str
    user_prompt: str
    closed_book: bool = False

    def to_json(self) -> dict:
        return {
            "id": self.row_id,
            "closed_book": self.closed_book,
            "system": self.system_prompt,
            "prompt": self.user_prompt,
        }


def make_task(row_id: str, question: str, answer: str, contexts: list[str],
              gold_answer: str, gold_article: int | None,
              closed_book: bool = False) -> JudgeTask:
    return JudgeTask(
        row_id=row_id,
        system_prompt=JUDGE_SYSTEM,
        user_prompt=build_judge_user_prompt(
            question, answer, contexts, gold_answer, gold_article, closed_book
        ),
        closed_book=closed_book,
    )


# ── Ollama judge (automated) ──────────────────────────────────────────────────

class OllamaJudge:
    """Scores a JudgeTask with a local Ollama model."""

    backend = "ollama"

    def __init__(self, model: str = "qwen2.5:3b-instruct",
                 host: str = "http://localhost:11434", temperature: float = 0.0) -> None:
        import ollama as ollama_lib
        self.model = model
        self.temperature = temperature
        self._client = ollama_lib.Client(host=host)

    def score(self, task: JudgeTask) -> dict:
        resp = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": task.system_prompt},
                {"role": "user", "content": task.user_prompt},
            ],
            format="json",
            options={"temperature": self.temperature, "num_predict": 400},
        )
        raw = resp["message"]["content"]
        # Strip <think>…</think> in case a reasoning model is configured.
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        return parse_verdict(raw)


# ── Claude-Code judge (file handoff) ──────────────────────────────────────────

class ClaudeCodeJudge:
    """Marker backend — scoring happens out-of-band via file handoff.

    The runner detects this backend and, instead of calling `.score()`, writes
    the tasks to disk for Claude Code to grade in-session. Calling `.score()`
    directly is a programming error."""

    backend = "claude-code"

    def score(self, task: JudgeTask) -> dict:  # pragma: no cover - guard
        raise RuntimeError(
            "claude-code judge is scored via file handoff, not programmatically. "
            "Use the 'judge' phase to emit tasks, then ingest verdicts."
        )


def make_judge(backend: str, model: str = "qwen2.5:3b-instruct",
               host: str = "http://localhost:11434", temperature: float = 0.0):
    if backend == "ollama":
        return OllamaJudge(model=model, host=host, temperature=temperature)
    if backend == "claude-code":
        return ClaudeCodeJudge()
    raise ValueError(f"unknown judge backend: {backend!r} (use 'ollama' or 'claude-code')")
