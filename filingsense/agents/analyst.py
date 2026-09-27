"""Analyst agent: answers ONLY from retrieved evidence, with citations.

The system prompt is the compliance boundary — every factual claim must
cite its evidence label, and "Not in the filings." is a valid, complete
answer. Runs on the configured analyst model (Gemini free tier).

Evidence is passed in as pre-formatted text; the caller (test script,
LangGraph node, API) decides what to retrieve. Labels look like [S1],
[S2], ... and are assigned by format_evidence().
"""
from __future__ import annotations

from google import genai
from google.genai import types
from pydantic import BaseModel
import threading

from filingsense.config import settings

SYSTEM = """You are a financial filings analyst. Answer the user's question
using ONLY the evidence below. Rules:
- Every factual claim must cite the evidence label(s) it comes from, like [S1].
  A claim without a citation is wrong.
- If the evidence does not contain the answer, set abstained=true and reply
  with exactly: "Not in the filings." Do not infer, do not use outside knowledge.
- Quote numbers exactly as they appear in the evidence; never round or
  convert units silently.
- Keep the answer under 150 words unless the question needs more."""


class Claim(BaseModel):
    text: str
    evidence: list[str]  # e.g. ["S1", "S3"]


class AnalystAnswer(BaseModel):
    answer: str
    claims: list[Claim]
    abstained: bool = False


def format_evidence(chunks: list[dict]) -> str:
    """Number retrieved chunks S1..Sn with source tags for the prompt.

    Chunk dicts come from HybridRetriever.search(); metadata may carry
    source and start_sec (earnings-call chunks).
    """
    lines = []
    for i, c in enumerate(chunks, 1):
        md = c.get("metadata", {}) or {}
        tag = f"{md.get('source', '10-K')} chunk {c.get('id', '?')}"
        if md.get("start_sec") is not None:
            tag += f" @ {md['start_sec']:.0f}s"
        lines.append(f"[S{i} | {tag}]\n{c['text']}")
    return "\n\n".join(lines)


def run_analyst(question: str, evidence: str,
                model: str | None = None) -> AnalystAnswer:
    """Answer `question` using only `evidence` (from format_evidence)."""
    client = genai.Client(api_key=settings.gemini_api_key)
    resp = client.models.generate_content(
        model=model or settings.analyst_model,
        contents=(f"EVIDENCE:\n{evidence}\n\nQUESTION: {question}\n\n"
                  "Answer with citations."),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            response_mime_type="application/json",
            response_schema=AnalystAnswer,
        ),
    )
        # Real token counts for the trace (Step 14). Module-level on purpose:
    # run_analyst's return type is the public contract; usage is telemetry.
    global last_usage
        # Real token counts for the trace. Thread-local (Step 15): the API
    # server runs requests on threads, and two concurrent answers must not
    # overwrite each other's telemetry. run_analyst's return type stays the
    # public contract; usage is telemetry.
    um = getattr(resp, "usage_metadata", None)
    _usage_tls.last_usage = {
        "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
        "candidates_tokens": getattr(um, "candidates_token_count", 0) or 0,
    } if um is not None else {}
    return resp.parsed


_usage_tls = threading.local()


def get_last_usage() -> dict:
    """Token telemetry from the most recent run_analyst call on this thread."""
    return getattr(_usage_tls, "last_usage", {})


