"""Step 10: Critic verifies each Analyst claim against its cited chunks.

Provision query -> 1 claim ("$14.2 billion"), critic judges the rounding.
Apple query -> abstention, no claims, empty report.
"""
from filingsense.agents.analyst import format_evidence, run_analyst
from filingsense.agents.critic import run_critic
from filingsense.decisions.base import get_gate
from filingsense.retrieval.hybrid import HybridRetriever

retriever = HybridRetriever()
retriever.load_sidecar("data/chunks.pkl")
gate = get_gate()
print(f"gate: {gate.name}\n")

for q in [
    "What was JPMorgan's provision for credit losses in 2025?",
    "What was Apple's revenue in 2025?",
]:
    chunks = retriever.search(q, top_k=5)
    ans = run_analyst(q, format_evidence(chunks))
    print(f"Q: {q}\nA: {ans.answer} (abstained={ans.abstained})")
    report = run_critic(ans, chunks, gate)
    for v in report.verdicts:
        flag = "ESCALATE" if v.escalated else ("OK" if v.supported else "FAIL")
        print(f"  [{flag}] (conf {v.confidence:.2f}) "
              f"{v.claim[:90]} {v.evidence}")
    if not report.verdicts:
        print("  (no claims to verify)")
    print(f"  trusted: {len(report.trusted_claims)}/{len(report.verdicts)}")
    print()

