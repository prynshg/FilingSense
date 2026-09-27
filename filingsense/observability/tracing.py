"""Observability: traces, per-node latency, token/cost accounting.

Local-first: every query appends JSONL spans to data/traces.jsonl with
zero config — no signup, no network, works offline. Langfuse is opt-in:
set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY and the same spans are
mirrored there (best-effort; a Langfuse outage never breaks a query).

The Tracer protocol keeps the pipeline vendor-neutral, the same way
DecisionGate does for the decision layer.

Usage:
    tracer = get_tracer()
    with tracer.trace("filingsense.ask", question=q) as t:
        with tracer.span("route", model="freejev@jev") as s:
            ...
            s["confidence"] = 0.94   # attrs are just a dict
    # -> one JSONL record per span in data/traces.jsonl
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol
import threading

from filingsense.config import settings

TRACE_PATH = Path("data/traces.jsonl")


def _has_key(value: str | None) -> bool:
    """Non-empty and not an unreplaced <placeholder> from .env.example."""
    v = (value or "").strip()
    return bool(v) and not (v.startswith("<") and v.endswith(">"))


class Tracer(Protocol):
    @contextmanager
    def trace(self, name: str, **attrs): ...
    @contextmanager
    def span(self, name: str, **attrs): ...


class LocalTracer:
    """Append-only JSONL spans. One trace per query; spans nest via parent_id.

    Trace state is thread-local (Step 15): the API server runs each request
    in its own thread, and concurrent traces must not share a span stack.
    """

    def __init__(self, path: str | Path = TRACE_PATH):
        self.path = Path(path)
        self._tls = threading.local()

    @property
    def _stack(self) -> list[str]:
        return self._tls.__dict__.setdefault("stack", [])

    def _get_trace_id(self) -> str | None:
        return self._tls.__dict__.get("trace_id")

    def _set_trace_id(self, value: str | None) -> None:
        self._tls.__dict__["trace_id"] = value

    @contextmanager
    def trace(self, name: str, **attrs):
        self._set_trace_id(uuid.uuid4().hex[:16])
        with self.span(name, **attrs) as a:
            yield a
        self._set_trace_id(None)

    @contextmanager
    def span(self, name: str, **attrs):
        sid = uuid.uuid4().hex[:12]
        rec = {
            "trace_id": self._get_trace_id(),
            "span_id": sid,
            "parent_id": self._stack[-1] if self._stack else None,
            "name": name,
            "start_ts": time.time(),
            "attrs": dict(attrs),
        }
        self._stack.append(sid)
        t0 = time.perf_counter()
        try:
            yield rec["attrs"]  # the live dict; mutations are recorded
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            self._stack.pop()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")


class _LangfuseMirror:
    """Best-effort Langfuse mirror. Shares the live attrs dict, so anything
    the pipeline records locally shows up in Langfuse too. Never raises."""

    def __init__(self):
        self._client = None
        self._trace = None
        try:
            from langfuse import Langfuse
            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        except Exception:
            self._client = None

    @contextmanager
    def trace(self, name: str, live: dict):
        if self._client is None:
            yield
            return
        try:
            self._trace = self._client.trace(name=name, input=dict(live))
            yield
            self._trace.update(output=dict(live))
            self._client.flush()
        except Exception:
            yield

    @contextmanager
    def span(self, name: str, live: dict):
        if self._client is None or self._trace is None:
            yield
            return
        span = None
        try:
            span = self._trace.span(name=name, input=dict(live))
        except Exception:
            pass
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if span is not None:
                try:
                    span.update(output={
                        **dict(live),
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    })
                except Exception:
                    pass


class FanoutTracer:
    """Local JSONL always; Langfuse mirror when keys are configured."""

    def __init__(self):
        self.local = LocalTracer()
        self.mirror = (_LangfuseMirror()
                       if _has_key(settings.langfuse_public_key) else None)

    @contextmanager
    def trace(self, name: str, **attrs):
        with self.local.trace(name, **attrs) as a:
            if self.mirror is None:
                yield a
            else:
                with self.mirror.trace(name, live=a):
                    yield a

    @contextmanager
    def span(self, name: str, **attrs):
        with self.local.span(name, **attrs) as a:
            if self.mirror is None:
                yield a
            else:
                with self.mirror.span(name, live=a):
                    yield a


_tracer: FanoutTracer | None = None


def get_tracer() -> FanoutTracer:
    global _tracer
    if _tracer is None:
        _tracer = FanoutTracer()
    return _tracer


# ---- cost accounting -------------------------------------------------
# The whole stack runs on free tiers; the tracer accounts tokens anyway so
# "it costs $0" is a measured claim, not a hope. FreeJev bills in credits:
# 1 credit per 1K input tokens (10k free on signup, no card).

def estimate_tokens(text: str) -> int:
    """~4 chars/token for English. Good enough for attribution; Gemini's
    real counts come from usage_metadata (Step 14b)."""
    return max(1, len(text) // 4)


def freejev_cost_credits(input_text: str) -> float:
    return round(estimate_tokens(input_text) / 1000, 4)
