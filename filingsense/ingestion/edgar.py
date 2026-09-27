"""Download 10-K filings from SEC EDGAR.

No API key needed — but the SEC requires a descriptive User-Agent header
(set SEC_USER_AGENT in .env). Be polite: a handful of requests per filing.
"""
from __future__ import annotations

import re
import warnings

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from filingsense.config import settings

HEADERS = {"User-Agent": settings.sec_user_agent, "Accept-Encoding": "gzip"}
SEC = "https://www.sec.gov"
DATA = "https://data.sec.gov"

# 10-Ks are iXBRL (XHTML) documents. We parse with the HTML parser on purpose
# (it handles the presentation tables best); the warning is expected, not a bug.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def _get(url: str) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r


def get_cik(ticker: str) -> str:
    """Map a ticker (e.g. 'JPM') to its zero-padded 10-digit CIK."""
    data = _get(f"{SEC}/files/company_tickers.json").json()
    for row in data.values():
        if row["ticker"].upper() == ticker.upper():
            return str(row["cik_str"]).zfill(10)
    raise ValueError(f"Ticker {ticker!r} not found in SEC company_tickers.json")


def list_10k_filings(cik: str, limit: int = 3) -> list[dict]:
    """Most recent 10-K filings for a CIK, newest first.

    We keep the SEC's own `primaryDocument` field — the authoritative
    filename of the main 10-K document — so we never guess which file
    in the filing index is the real one.
    """
    subs = _get(f"{DATA}/submissions/CIK{cik}.json").json()
    recent = subs["filings"]["recent"]
    out = []
    for i in range(len(recent["form"])):
        if recent["form"][i] == "10-K":
            out.append({
                "accession": recent["accessionNumber"][i],
                "filed": recent["filingDate"][i],
                "report_date": recent["reportDate"][i],
                "primary": recent["primaryDocument"][i],
            })
        if len(out) >= limit:
            break
    return out


def download_10k_text(cik: str, accession: str, primary: str) -> dict:
    """Download the main 10-K document and return cleaned text.

    `primary` is the SEC's own primaryDocument filename — the authoritative
    main document, so we never guess from the filing index.

    Two things get stripped:
    - script/style tags (website chrome, not filing content)
    - ix:header — the hidden inline-XBRL metadata block. Without this,
      chunks get polluted with taxonomy URLs and embeddings drift.

    Tables are preserved as pipe-delimited rows so financial statements
    survive chunking (Step 3).
    """
    nodash = accession.replace("-", "")
    url = f"{SEC}/Archives/edgar/data/{int(cik)}/{nodash}/{primary}"
    soup = BeautifulSoup(_get(url).text, "lxml")

    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all("ix:header"):
        tag.decompose()

    # Preserve tables as pipe-rows before stripping tags.
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all(["tr"]):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append("| " + " | ".join(cells) + " |")
        table.replace_with("\n" + "\n".join(rows) + "\n")

    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return {"accession": accession, "url": url, "text": text.strip()}


def ingest_company(ticker: str, limit: int = 2) -> list[dict]:
    """End-to-end: ticker -> list of cleaned 10-K documents."""
    cik = get_cik(ticker)
    docs = []
    for filing in list_10k_filings(cik, limit=limit):
        doc = download_10k_text(cik, filing["accession"], filing["primary"])
        doc.update({"ticker": ticker.upper(), "cik": cik,
                    "filed": filing["filed"], "report_date": filing["report_date"]})
        docs.append(doc)
    return docs
