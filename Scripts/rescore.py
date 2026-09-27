"""Re-score a saved eval report with the corrected abstention definition.

Step 13 lesson: the pipeline intentionally continues past low-confidence
gate refusals (ABSTAIN_BELOW=0.6). 'The system abstained' must mean the
final answer IS 'Not in the filings.' — not that the gate voted no.
Re-scoring recorded outputs is instant; no 31-minute re-run needed.
Writes data/eval_results_rescored.json (original untouched).
"""
import json

STOP = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "or",
        "with", "by", "from", "at", "as", "is", "was", "were", "are",
        "be", "its", "it", "that", "this", "which", "per"}


def norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def keywords(s: str) -> set[str]:
    return {w.strip(".,;:()\"'").lower() for w in s.split()
            if len(w) > 3} - STOP


def main() -> None:
    report = json.load(open("data/eval_results.json"))
    golden = {q["id"]: q for q in json.load(open("data/golden.json"))}

    for row in report["rows"]:
        if "error" in row:
            continue
        item = golden[row["id"]]
        ab = row.get("final", "").strip() == "Not in the filings."
        row["pred_abstained"] = ab
        row["answerability_ok"] = ab == (not item["answerable"])
        if item["answerable"] and not ab:
            row["route_ok"] = row.get("pred_route") == item["expected_route"]
            if item["reference_kind"] == "exact":
                row["exact_ok"] = norm(item["reference"]) in norm(
                    row.get("final", ""))
            elif item["reference_kind"] == "key_facts":
                kr = keywords(item["reference"])
                ka = keywords(row.get("final", ""))
                row["keyword_recall"] = round(len(kr & ka) / max(1, len(kr)), 3)

    ok = [r for r in report["rows"] if "error" not in r]
    answered = [r for r in ok if r["answerable"] and not r["pred_abstained"]]
    abst_pred = [r for r in ok if r["pred_abstained"]]
    abst_true = [r for r in ok if not r["answerable"]]
    exact_rows = [r for r in answered if "exact_ok" in r]
    kf_rows = [r for r in answered if "keyword_recall" in r]
    all_claims = sum(r["n_claims"] for r in ok)
    summary = {
        "n": len(ok),
        "errors": len(report["rows"]) - len(ok),
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
    }
    print("=== rescored summary ===")
    for k, v in summary.items():
        print(f"  {k:22s} {v}")
    still_bad = [r["id"] for r in ok if not r.get("answerability_ok")]
    print("remaining answerability misses:", still_bad or "none")

    report["summary"] = summary
    report["rescored"] = True
    with open("data/eval_results_rescored.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("written to data/eval_results_rescored.json")


if __name__ == "__main__":
    main()
