"""LLM-backed DecisionGate on Google Gemini's free tier.

Same DecisionGate interface as every other engine. Two purposes:
(1) the pipeline runs with zero paid vendors; (2) the honest baseline
Jev has to beat in the eval harness.
"""
from __future__ import annotations

import time

from google import genai
from google.genai import types
from pydantic import BaseModel

from filingsense.config import settings
from filingsense.decisions.base import Decision


class _YesNo(BaseModel):
    answer: bool
    confidence: float


class _Choice(BaseModel):
    choice: str
    confidence: float


class _Score(BaseModel):
    score: int
    confidence: float


class LLMFallbackGate:
    name = "llm-fallback"

    def __init__(self):
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.critic_model  # e.g. gemini-2.5-flash

    def _ask(self, schema: type[BaseModel], question: str, state: str,
             extra: str = "") -> Decision:
        t0 = time.perf_counter()
        resp = self.client.models.generate_content(
            model=self.model,
            contents=(f"STATE:\n{state}\n\nQUESTION: {question}\n{extra}\n"
                      "Return the value plus your confidence 0..1."),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        p = resp.parsed
        if isinstance(p, _YesNo):
            val = p.answer
        elif isinstance(p, _Choice):
            val = p.choice
        else:
            val = p.score
        return Decision(value=val, confidence=p.confidence,
                        model=f"llm-fallback@{self.model}",
                        latency_ms=(time.perf_counter() - t0) * 1000)

    def ask_yes_no(self, question: str, state: str) -> Decision:
        return self._ask(_YesNo, question, state)

    def ask_choice(self, question: str, options: list[str], state: str) -> Decision:
        return self._ask(_Choice, question, state,
                         extra=f"OPTIONS (reply with exactly one): {options}")

    def ask_score(self, question: str, state: str, levels: int = 5) -> Decision:
        return self._ask(_Score, question, state,
                         extra=f"Reply with an integer score 1..{levels}.")
