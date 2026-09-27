"""Decision-layer interface.

Jev is the primary implementation; an LLM fallback keeps the pipeline
working without a TypeSafe key — and is the honest baseline Jev has to
beat in the eval harness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Decision:
    value: object
    confidence: float                      # 0..1
    probabilities: dict[str, float] = field(default_factory=dict)
    model: str = ""
    latency_ms: float = 0.0
    attempts: int = 1
    raw: dict = field(default_factory=dict)


class DecisionGate(Protocol):
    name: str

    def ask_yes_no(self, question: str, state: str) -> Decision: ...
    def ask_choice(self, question: str, options: list[str], state: str) -> Decision: ...
    def ask_score(self, question: str, state: str, levels: int = 5) -> Decision: ...


def get_gate() -> DecisionGate:
    """Jev when configured, else the LLM fallback.

    Provider priority: TypeSafe direct > FreeJev (free) > LLM fallback.
    The pipeline never hard-depends on a 10-day-old API.
    """
    from filingsense.config import settings

    if settings.jev_enabled and settings.typesafe_api_key:
        from filingsense.decisions.jev import JevGate
        return JevGate()
    if settings.jev_enabled and settings.freejev_api_key:
        from filingsense.decisions.jev import FreeJevGate
        return FreeJevGate()
    from filingsense.decisions.llm_fallback import LLMFallbackGate
    return LLMFallbackGate()

