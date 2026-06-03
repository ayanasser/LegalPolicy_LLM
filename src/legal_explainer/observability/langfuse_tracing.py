"""Langfuse tracing helper (MLOps observability).

One reusable entry point that every project / the unified UI calls to log a
trace per question. The **trace name is the project name** (Langfuse v4 names a
trace after its root observation), so traces in the Langfuse UI are grouped by
project (e.g. "Neo4j Graph RAG", "Finetuned · Qwen2.5-3B Knowledge adapter").

Design goals:
  * **Never break the app.** If Langfuse isn't configured (no keys) or the
    server is unreachable, this degrades to a no-op — the UI keeps working.
  * **Host-agnostic.** Works against self-hosted (http://localhost:3000) or
    Langfuse Cloud — it just reads LANGFUSE_* from the environment / .env.

Usage:
    from legal_explainer.observability.langfuse_tracing import trace_question

    with trace_question("Neo4j Graph RAG", question, backend_id="neo4j-rag") as tr:
        answer = run_backend(...)
        tr.set_output(answer, metadata={"n_articles": 5, "ms": 1234})
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Load .env so LANGFUSE_PUBLIC_KEY / SECRET_KEY / HOST are available even when
# the launching process didn't export them.
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except Exception:
    pass

_client = None
_init_tried = False


def _enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def get_client():
    """Lazily build (and cache) the Langfuse client. Returns None if disabled
    or if the SDK/credentials can't be initialised."""
    global _client, _init_tried
    if _client is not None:
        return _client
    if _init_tried or not _enabled():
        return None
    _init_tried = True
    try:
        from langfuse import Langfuse
        _client = Langfuse()  # reads LANGFUSE_* from env
    except Exception as e:  # pragma: no cover
        print(f"[langfuse] disabled — could not init client: {e}")
        _client = None
    return _client


def flush_traces() -> None:
    """Synchronously flush pending traces (for short-lived scripts / tests)."""
    client = get_client()
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass


class _TraceHandle:
    """Lightweight handle passed to the caller to attach the answer + metadata."""

    def __init__(self, span):
        self._span = span
        self.output = None
        self.metadata: dict = {}

    def set_output(self, output, **metadata):
        self.output = output
        if metadata:
            self.metadata.update(metadata)


@contextlib.contextmanager
def trace_question(project_name: str, question: str, backend_id: str | None = None,
                   session_id: str | None = None, user_id: str | None = None):
    """Context manager that records one trace named after `project_name`.

    The wrapped block's latency is captured automatically. Call
    `handle.set_output(answer, **metadata)` inside the block.
    """
    client = get_client()
    if client is None:
        # No-op handle — app still works without Langfuse.
        yield _TraceHandle(None)
        return

    meta = {"backend_id": backend_id} if backend_id else {}
    try:
        with client.start_as_current_observation(
            name=project_name, as_type="span",
            input={"question": question}, metadata=meta,
        ) as span:
            handle = _TraceHandle(span)
            try:
                yield handle
            finally:
                try:
                    span.update(output=handle.output,
                                metadata={**meta, **handle.metadata})
                    # Promote name/io to the trace level (trace name == project).
                    span.set_trace_io(input={"question": question},
                                      output=handle.output)
                except Exception:
                    pass
        # NB: we deliberately do NOT call client.flush() here — flush blocks
        # (retries) when the server is down, which would slow the UI. The
        # background batch exporter sends spans within a few seconds when the
        # stack is up, and is harmless when it isn't. Call flush_traces()
        # explicitly (e.g. in a script) if you need a synchronous send.
    except Exception as e:  # never let tracing break a request
        print(f"[langfuse] trace error (ignored): {e}")
        yield _TraceHandle(None)
