"""Flow logger — prints a human-readable trace of orchestrator activity.

Single-line events on stderr, tagged by stage. Examples:

  [12:30:01.4 a3b1f2] SAFETY     allow
  [12:30:01.5 a3b1f2] ROUTE      medium (rule=article_reference)
  [12:30:01.5 a3b1f2] SUBAGENT   researcher start
  [12:30:02.1 a3b1f2] TOOL       check_statute_reference args={"statute_reference":"Article 713"}
  [12:30:03.2 a3b1f2] SUBAGENT   researcher done  (4.7s, $0.012, 3 passages)
  [12:30:04.9 a3b1f2] SUBAGENT   explainer  done  (3.1s, $0.008)

Set FLOW_LOG_LEVEL env var to 'silent' to suppress, or to a path to ALSO mirror
events into a file. By default events are emitted to stderr so they don't
pollute notebook cell outputs (which capture stdout).
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

_LEVEL = os.getenv("FLOW_LOG_LEVEL", "normal")  # silent | normal | <filepath>
_LEVEL_IS_SILENT = _LEVEL == "silent"
_LEVEL_IS_FILE = _LEVEL not in ("silent", "normal") and os.access(
    os.path.dirname(_LEVEL) or ".", os.W_OK
)


def _ts() -> str:
    t = time.time()
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 10)}"


def _short(value: Any, n: int = 80) -> str:
    s = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return s if len(s) <= n else s[:n - 1] + "…"


@dataclass
class FlowLogger:
    """Per-query flow logger. Attach one instance to each orchestrator run."""

    query_id: str
    started_at: float = field(default_factory=time.time)

    def _emit(self, stage: str, message: str) -> None:
        if _LEVEL_IS_SILENT:
            return
        line = f"[{_ts()} {self.query_id}] {stage:<10} {message}"
        print(line, file=sys.stderr, flush=True)
        if _LEVEL_IS_FILE:
            with open(_LEVEL, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def query(self, user_query: str) -> None:
        self._emit("QUERY", _short(user_query, 120))

    def safety(self, verdict: str, reason: str | None) -> None:
        msg = verdict if not reason else f"{verdict} ({reason})"
        self._emit("SAFETY", msg)

    def route(self, complexity: str, rule: str | None, used_llm: bool) -> None:
        rule_txt = f"rule={rule}" if rule else "llm-classifier"
        self._emit("ROUTE", f"{complexity:<7} ({rule_txt}, used_llm={used_llm})")

    def subagent_start(self, name: str) -> None:
        self._emit("SUBAGENT", f"{name:<10} start")

    def subagent_done(
        self, name: str, duration_ms: int, cost_usd: float, summary: str = ""
    ) -> None:
        s = f"{name:<10} done   ({duration_ms / 1000:.1f}s, ${cost_usd:.4f})"
        if summary:
            s += f"  {summary}"
        self._emit("SUBAGENT", s)

    def tool_call(self, tool: str, args: dict[str, Any]) -> None:
        self._emit("TOOL", f"{tool}  args={_short(args)}")

    def tool_result(self, tool: str, summary: str) -> None:
        self._emit("TOOL", f"{tool}  → {_short(summary, 100)}")

    def tool_error(self, tool: str, error: str) -> None:
        self._emit("TOOL", f"{tool}  !! {_short(error, 100)}")

    def info(self, message: str) -> None:
        self._emit("INFO", message)

    def warn(self, message: str) -> None:
        self._emit("WARN", message)

    def done(self, path: str, total_cost: float) -> None:
        elapsed = time.time() - self.started_at
        self._emit("DONE", f"path={path}, {elapsed:.1f}s total, ${total_cost:.4f}")


@contextmanager
def quiet():
    """Temporarily silence flow logging — useful inside eval sweeps."""
    global _LEVEL_IS_SILENT
    prev = _LEVEL_IS_SILENT
    _LEVEL_IS_SILENT = True
    try:
        yield
    finally:
        _LEVEL_IS_SILENT = prev