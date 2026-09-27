"""Replay the remaining abstain-misses with evidence visible, so the fix
(if any) is aimed at the true cause instead of the metric."""
from filingsense.agents.analyst import format_evidence
from filingsense.agents.graph import build_graph
from filingsense.decisions.base import get_gate
from filingsense.retrieval.hybrid import HybridRetriever

QS = {
    "g018": "How many employees did JPMorgan have at end of 2025?",
    "g039": "By how much did debit and credit sales volume grow, per the call?",
    "g041": ("How many net new checking accounts did JPMorgan add in 2025, "
             "per the call?"),
    "g043": ("What full-year revenue and ROTCE did management report "
             "excluding significant items?"),
}

retriever = HybridRetriever()
retriever.load_sidecar("data/chunks.pkl")
graph = build_graph(retriever, get_gate())
for gid, q in QS.items():
    out = graph.invoke({"question": q})
    print(f"=== {gid}: answerable={out.get('answerable')} "
          f"route={out.get('route')}")
    print(f"    final: {out.get('final', '')[:150]}")
    if out.get("answerable") and out.get("route"):
        print("    evidence given to analyst:")
        print(format_evidence(out.get("chunks", []))[:1500])
    print()
