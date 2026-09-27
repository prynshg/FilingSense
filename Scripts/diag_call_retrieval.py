"""Diagnose call-route abstentions: is the retrieval funnel dropping the
3 call chunks, or does the analyst see them and still refuse?"""
from filingsense.retrieval.hybrid import HybridRetriever

QUESTIONS = {
    "g036": "What was JPMorgan's CET1 ratio at the end of 2025, according to the earnings call?",
    "g039": "By how much did debit and credit sales volume grow, per the call?",
    "g018": "How many employees did JPMorgan have at end of 2025?",
}

r = HybridRetriever()
r.load_sidecar("data/chunks.pkl")
for gid, q in QUESTIONS.items():
    chunks = r.search(q, top_k=8)
    call = [c for c in chunks
            if (c.get("metadata", {}) or {}).get("source") == "earnings_call"]
    print(f"{gid}: top-8 contains {len(call)} call chunk(s)")
    for c in chunks:
        md = c.get("metadata", {}) or {}
        print(f"   {c['id']}  src={md.get('source')}")
    for c in call:
        print(f"   >>> call chunk text: {c['text'][:200]}")
    print()
