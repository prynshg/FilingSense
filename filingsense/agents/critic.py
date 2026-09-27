"""Critic agent: verifies each Analyst claim against its cited evidence.

Each claim becomes one typed yes/no question for the decision gate (Jev,
or the LLM fallback): "is this claim fully supported by these chunks?"
Low gate confidence escalates for human review. This is the cascade
pattern: cheap, fast decisions first, expensive scrutiny only where needed.

Crucially, the critic sees ONLY the chunks the analyst cited — never the
full evidence set. It audits the citation, not the answer: a claim that
is true but cites chunks that don't support it still fails.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from filingsense.agents.analyst import AnalystAnswer
from filingsense.decisions.base import Decision, DecisionGate, get_gate

ESCALATE_BELOW = 0.7


@dataclass
class ClaimVerdict:
    claim: str
    evidence: list[str]
    supported: bool    # the gate's yes/no answer
    confidence: float  # the gate's confidence in that answer
    escalated: bool    # confidence < ESCALATE_BELOW -> needs human review
    decision: Decision | None = None


@dataclass
class CriticReport:
    verdicts: list[ClaimVerdict] = field(default_factory=list)

    @property
    def all_supported(self) -> bool:
        return all(v.supported for v in self.verdicts)

    @property
    def escalations(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if v.escalated]

    @property
    def trusted_claims(self) -> list[ClaimVerdict]:
        """Supported AND confident — safe to show the user as-is."""
        return [v for v in self.verdicts if v.supported and not v.escalated]


def _label_to_index(label: str) -> int | None:
    # Labels are "S1".."Sn" from format_evidence(); S1 -> chunks[0].
    if len(label) > 1 and label[0] == "S" and label[1:].isdigit():
        return int(label[1:]) - 1
    return None


def run_critic(answer: AnalystAnswer, chunks: list[dict],
               gate: DecisionGate | None = None) -> CriticReport:
    """Verify each claim of `answer` against its cited chunks.

    `chunks` is the same ordered list that was passed to format_evidence().
    `gate` defaults to get_gate() (Jev when configured, else LLM fallback).
    """
    gate = gate or get_gate()
    report = CriticReport()
    if answer.abstained or not answer.claims:
        return report
    for claim in answer.claims:
        idxs = [i for lab in claim.evidence
                if (i := _label_to_index(lab)) is not None and i < len(chunks)]
        cited = "\n\n".join(f"[S{i + 1}]\n{chunks[i]['text']}" for i in idxs)
        if not cited:
            # Claim cites nothing retrievable -> automatic fail.
            report.verdicts.append(ClaimVerdict(
                claim=claim.text, evidence=claim.evidence,
                supported=False, confidence=1.0, escalated=False))
            continue
        d = gate.ask_yes_no(
            question=("Is the claim below fully and exactly supported by the "
                      "cited chunks? A claim that rounds, converts units, or "
                      "adds detail not present in the chunks is NOT supported."),
            state=f"CLAIM: {claim.text}\n\nCITED CHUNKS:\n{cited}",
        )
        report.verdicts.append(ClaimVerdict(
            claim=claim.text,
            evidence=claim.evidence,
            supported=bool(d.value),
            confidence=d.confidence,
            escalated=d.confidence < ESCALATE_BELOW,
            decision=d,
        ))
    return report
