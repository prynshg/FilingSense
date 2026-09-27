"""Full pipeline CLI: answer one question, print the decision audit log,
and save a JSON audit trail.

    python scripts/ask.py --question "What was JPM's provision for credit losses in 2025?"
    python scripts/ask.py --question "..." --audit data/audit_provision.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from filingsense.agents.graph import build_graph
from filingsense.decisions.base import get_gate
from filingsense.retrieval.hybrid import HybridRetriever

from filingsense.agents.graph import build_graph, invoke_with_trace

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--audit", default="data/last_audit.json",
                    help="where to write the JSON audit log")
    args = ap.parse_args()

    retriever = HybridRetriever()
    retriever.load_sidecar("data/chunks.pkl")  # BM25 side; vectors in Qdrant

    out = invoke_with_trace(build_graph(retriever, get_gate()), args.question)

    print(out["final"])
    print("\n--- decision audit log ---")
    for d in out.get("decisions", []):
        print(d)

    audit = {
        "question": args.question,
        "answerable": out.get("answerable"),
        "route": out.get("route"),
        "final": out.get("final"),
        "decisions": out.get("decisions", []),
        "critic": out.get("critic", {}),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(args.audit, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"\naudit written to {args.audit}")


if __name__ == "__main__":
    main()
