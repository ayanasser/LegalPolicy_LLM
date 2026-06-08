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
    """Lightweight handle passed to the caller to attach the answer + rich
    structured detail. Everything beyond ``output`` is optional and degrades
    gracefully — pass what each backend knows.

    Fields that get promoted into a proper nested observation in the trace:
      * ``documents`` + ``retrieval_meta`` → a child ``retriever`` observation
        (the retrieved chunks, scores, keywords, timing …).
      * ``model`` / ``model_parameters`` / ``usage`` → a child ``generation``
        observation (the answer model, its sampling params, token usage).
    Anything passed as plain kwargs lands in the span/trace metadata.
    """

    def __init__(self, span):
        self._span = span
        self.output = None
        self.metadata: dict = {}
        self.model = None
        self.model_parameters: dict | None = None
        self.usage: dict | None = None
        self.documents = None
        self.retrieval_meta: dict | None = None
        self.prompt = None

    def set_output(self, output, *, model=None, model_parameters=None,
                   usage=None, documents=None, retrieval_meta=None,
                   prompt=None, **metadata):
        self.output = output
        if model is not None:
            self.model = model
        if model_parameters is not None:
            self.model_parameters = model_parameters
        if usage is not None:
            self.usage = usage
        if documents is not None:
            self.documents = documents
        if retrieval_meta is not None:
            self.retrieval_meta = retrieval_meta
        if prompt is not None:
            self.prompt = prompt
        # Drop None-valued kwargs so the trace metadata stays clean.
        clean = {k: v for k, v in metadata.items() if v is not None}
        if clean:
            self.metadata.update(clean)


def _emit_child_observations(span, question, handle):
    """Create the nested retriever / generation observations (best-effort)."""
    # 1) Retrieval — the chunks the RAG backend pulled in.
    if handle.documents:
        try:
            with span.start_as_current_observation(
                name="retrieval", as_type="retriever",
                input={"question": question},
                output=handle.documents,
                metadata=handle.retrieval_meta or {},
            ):
                pass
        except Exception:
            pass
    # 2) Generation — the answer model + its sampling params + token usage.
    #    Input = the EXACT prompt/messages sent to the model when the backend
    #    exposes it (local HF + prompt-design); otherwise fall back to the question.
    if handle.model or handle.model_parameters or handle.usage:
        try:
            with span.start_as_current_observation(
                name="generation", as_type="generation",
                input=(handle.prompt if handle.prompt is not None
                       else {"question": question}),
                output=handle.output,
                model=handle.model,
                model_parameters=handle.model_parameters or {},
                usage_details=handle.usage or None,
            ):
                pass
        except Exception:
            pass


@contextlib.contextmanager
def trace_question(project_name: str, question: str, backend_id: str | None = None,
                   session_id: str | None = None, user_id: str | None = None):
    """Context manager that records one trace named after `project_name`.

    The wrapped block's latency is captured automatically. Call
    `handle.set_output(answer, ...)` inside the block to attach the answer and
    any model / retrieval detail (see :class:`_TraceHandle`).
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
                    # Nested retriever + generation observations (rich detail).
                    _emit_child_observations(span, question, handle)
                    full_meta = {**meta, **handle.metadata}
                    if handle.model:
                        full_meta["answer_model"] = handle.model
                    if handle.model_parameters:
                        full_meta["model_parameters"] = handle.model_parameters
                    if handle.usage:
                        full_meta["token_usage"] = handle.usage
                    if handle.retrieval_meta:
                        full_meta["retrieval"] = handle.retrieval_meta
                    span.update(output=handle.output, metadata=full_meta)
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
