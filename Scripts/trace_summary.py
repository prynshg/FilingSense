"""Summarize data/traces.jsonl: per-node latency and cost totals.

    python scripts/trace_summary.py
    python scripts/trace_summary.py --path data/traces.jsonl --last 5
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/traces.jsonl")
    ap.add_argument("--last", type=int, default=0,
                    help="only the N most recent traces (0 = all)")
    args = ap.parse_args()

    spans = [json.loads(l) for l in Path(args.path).open()]
    traces: dict[str, list[dict]] = defaultdict(list)
    for r in spans:
        traces[r["trace_id"]].append(r)
    if args.last:
        tids = sorted(traces, key=lambda t: max(
            r["start_ts"] for r in traces[t]))[-args.last:]
    else:
        tids = list(traces)
    spans = [r for t in tids for r in traces[t]]

    by_node: dict[str, list[float]] = defaultdict(list)
    credits = cost_usd = 0.0
    prompt_tok = cand_tok = 0
    retried = errors = 0
    for r in spans:
        a = r.get("attrs", {}) or {}
        by_node[r["name"]].append(r["latency_ms"])
        credits += a.get("credits_est") or 0
        cost_usd += a.get("cost_usd_est") or 0
        prompt_tok += a.get("prompt_tokens") or 0
        cand_tok += a.get("candidates_tokens") or 0
        if (a.get("attempts") or 1) > 1:
            retried += 1
        if r.get("error"):
            errors += 1

    print(f"{len(tids)} traces, {len(spans)} spans")
    print(f"\n{'node':22s} {'n':>4} {'mean_ms':>9} {'p95_ms':>8}")
    for name, lat in sorted(by_node.items()):
        lat.sort()
        p95 = lat[int(0.95 * (len(lat) - 1))]
        print(f"{name:22s} {len(lat):>4} "
              f"{sum(lat) / len(lat):>9.0f} {p95:>8.0f}")
    print(f"\nFreeJev credits used (est): {credits:.4f} "
          f"(~{credits / 1000:.1%} of the 10k free signup credits)")
    print(f"TypeSafe-direct cost (est): ${cost_usd:.6f}")
    print(f"Gemini tokens: {prompt_tok} prompt + {cand_tok} candidates "
          f"(free tier: $0.00)")
    print(f"spans with retries: {retried}, spans with errors: {errors}")


if __name__ == "__main__":
    main()
