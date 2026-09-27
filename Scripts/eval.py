"""Step 13: eval harness — run the golden set through the pipeline, score it.

Metrics (each maps to a Ragas concept, without the extra dependency):
  answerability accuracy, abstention precision/recall  (safety)
  routing accuracy on answered questions                (context precision)
  exact-match rate        (reference_kind == exact)     (answer relevancy)
  keyword recall          (reference_kind == key_facts; rough — human review)
  critic faithfulness     (trusted claims / total claims)
  escalation rate, per-node mean latency                (cost/latency)

Run:  python scripts/eval.py --limit 4     # smoke test
      python scripts/eval.py               # full 44 (~15 min on FreeJev)
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

from filingsense.agents.graph import build_graph
from filingsense.decisions.base import get_gate
from filingsense.retrieval.hybrid import HybridRetriever
from filingsense.agents.graph import build_graph, invoke_with_trace



GOLDEN = "data/golden.json"
STOP = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "or",
        "with", "by", "from", "at", "as", "is", "was", "were", "are",
        "be", "its", "it", "that", "this", "which", "per"}


def norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def keywords(s: str) -> set[str]:
    return {w.strip(".,;:()\"'").lower() for w in s.split()
            if len(w) > 3} - STOP


def score(item: dict, out: dict) -> dict:
    final = out.get("final", "")
    abstained = final.strip() == "Not in the filings."
    expected_answerable = item["answerable"]
    row = {
        "id": item["id"],
        "question": item["question"],
        "answerable": expected_answerable,
        "expected_route": item["expected_route"],
        "pred_abstained": abstained,
        "pred_route": out.get("route"),
        "answerability_ok": abstained == (not expected_answerable),
        "final": final[:300],
    }
    if expected_answerable and not abstained:
        row["route_ok"] = out.get("route") == item["expected_route"]
        if item["reference_kind"] == "exact":
            row["exact_ok"] = norm(item["reference"]) in norm(final)
        elif item["reference_kind"] == "key_facts":
            kr, ka = keywords(item["reference"]), keywords(final)
            row["keyword_recall"] = round(len(kr & ka) / max(1, len(kr)), 3)
    verdicts = out.get("critic", {}).get("verdicts", [])
    trusted = [v for v in verdicts if v["supported"] and not v["escalated"]]
    row["n_claims"] = len(verdicts)
    row["n_trusted"] = len(trusted)
    row["n_escalated"] = out.get("critic", {}).get("escalations", 0)
    row["latency"] = {d["node"]: d["latency_ms"]
                      for d in out.get("decisions", [])}
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="data/eval_results.json")
    args = ap.parse_args()

    items = json.load(open(GOLDEN))
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} questions; building pipeline...")
    retriever = HybridRetriever()
    retriever.load_sidecar("data/chunks.pkl")
    graph = build_graph(retriever, get_gate())

    rows, t0 = [], time.perf_counter()
    for i, item in enumerate(items, 1):
        q0 = time.perf_counter()
        try:
            out = invoke_with_trace(graph, item["question"])
            row = score(item, out)
        except Exception as e:  # record and continue; eval must not die
            row = {"id": item["id"], "question": item["question"],
                   "error": f"{type(e).__name__}: {e}"}
        row["secs"] = round(time.perf_counter() - q0, 1)
        rows.append(row)
        flag = row.get("error") or (
            "OK" if row.get("answerability_ok") else "ABSTAIN-MISS")
        print(f"[{i}/{len(items)}] {item['id']} {flag} ({row['secs']}s)",
              flush=True)

    ok = [r for r in rows if "error" not in r]
    answered = [r for r in ok if r["answerable"] and not r["pred_abstained"]]
    abst_pred = [r for r in ok if r["pred_abstained"]]
    abst_true = [r for r in ok if not r["answerable"]]
    exact_rows = [r for r in answered if "exact_ok" in r]
    kf_rows = [r for r in answered if "keyword_recall" in r]
    all_claims = sum(r["n_claims"] for r in ok)
    summary = {
        "n": len(items),
        "errors": len(rows) - len(ok),
        "answerability_acc": round(
            sum(r["answerability_ok"] for r in ok) / max(1, len(ok)), 3),
        "abstention_precision": round(
            sum(1 for r in abst_pred if not r["answerable"])
            / max(1, len(abst_pred)), 3),
        "abstention_recall": round(
            sum(1 for r in abst_true if r["pred_abstained"])
            / max(1, len(abst_true)), 3),
        "answer_recall": round(len(answered) / max(
            1, sum(1 for r in ok if r["answerable"])), 3),
        "routing_acc": round(
            sum(r.get("route_ok", False) for r in answered)
            / max(1, len(answered)), 3),
        "exact_match": round(
            sum(r["exact_ok"] for r in exact_rows) / max(1, len(exact_rows)), 3),
        "mean_keyword_recall": round(
            sum(r["keyword_recall"] for r in kf_rows) / max(1, len(kf_rows)), 3),
        "critic_faithfulness": round(
            sum(r["n_trusted"] for r in ok) / max(1, all_claims), 3),
        "escalation_rate": round(
            sum(r["n_escalated"] for r in ok) / max(1, all_claims), 3),
        "mean_secs": round(
            sum(r["secs"] for r in rows) / max(1, len(rows)), 1),
        "total_mins": round((time.perf_counter() - t0) / 60, 1),
    }
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k:22s} {v}")

    report = {"timestamp": datetime.now(timezone.utc).isoformat(),
              "summary": summary, "rows": rows}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nreport written to {args.out}")

    needs_review = [r for r in kf_rows if r["keyword_recall"] < 0.5]
    if needs_review:
        print("\nkey_facts questions with keyword_recall < 0.5 "
              "(human review advised):")
        for r in needs_review:
            print(f"  {r['id']} ({r['keyword_recall']}): "
                  f"{r['question'][:70]}")


if __name__ == "__main__":
    main()
