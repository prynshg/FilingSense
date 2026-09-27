"""Step 2 check: download one 10-K and inspect it."""
from filingsense.ingestion.edgar import ingest_company

docs = ingest_company("JPM", limit=1)
d = docs[0]
print("ticker:", d["ticker"], "| filed:", d["filed"])
print("url:", d["url"])
print(f"chars: {len(d['text']):,}")
print("---- first 800 chars ----")
print(d["text"][:800])
lines = d["text"].splitlines()
table_rows = [l for l in lines if l.strip().startswith("|")]
print(f"---- table rows preserved: {len(table_rows)} ----")
for l in table_rows[:3]:
    print(l[:160])
