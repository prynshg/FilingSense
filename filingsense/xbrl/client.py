"""Download (and cache) SEC XBRL company facts.

Endpoint: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:0>10}.json
Every tagged numeric fact the company has ever filed, machine-readable.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx

from filingsense.config import settings


def fetch_company_facts(cik: str, cache_dir: str | Path = "data") -> dict:
    """Return the companyfacts JSON for a CIK, using a local cache."""
    cik_padded = cik.zfill(10)
    cache = Path(cache_dir) / f"xbrl_{cik_padded}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    resp = httpx.get(url, headers={"User-Agent": settings.sec_user_agent}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    cache.write_text(json.dumps(data), encoding="utf-8")
    return data
