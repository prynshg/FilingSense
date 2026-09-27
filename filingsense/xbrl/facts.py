"""Extract exact figures from SEC XBRL company facts.

The companyfacts JSON nests data as:
    facts -> {taxonomy} -> {tag} -> units -> {unit} -> [datapoints]
Each datapoint looks like:
    {"start": ..., "end": "2025-12-31", "val": 14212000000,
     "accn": "0001628280-26-008131", "fy": 2025, "fp": "FY",
     "form": "10-K", "filed": "2026-02-13", "frame": "CY2025"}

Gotcha: comparative columns in one filing (2025/2024/2023 side by side)
can share fy, form AND filed date — the filing's attributes, not the
value's. Disambiguate with frame/end, which describe the value's period.
"""
from __future__ import annotations


def get_fact(
    data: dict,
    tag: str,
    *,
    taxonomy: str = "us-gaap",
    form: str = "10-K",
    fy: int,
    fp: str = "FY",
    unit: str = "USD",
) -> dict | None:
    """Latest-filed datapoint for a tag and fiscal period, or None."""
    try:
        points = data["facts"][taxonomy][tag]["units"][unit]
    except KeyError:
        return None
    cands = [
        p for p in points
        if p.get("form") == form and p.get("fy") == fy and p.get("fp") == fp
    ]
    if not cands:
        return None
    # Prefer the frame matching the requested year (CY2025 / FY2025),
    # else the period end-date year, else fall back to the raw candidates.
    framed = [p for p in cands if p.get("frame") in (f"CY{fy}", f"FY{fy}")]
    pool = framed or [p for p in cands if str(p.get("end", "")).startswith(str(fy))]
    pool = pool or cands
    # Prefer the most recently filed (handles 10-K/A amendments).
    return max(pool, key=lambda p: p["filed"])
