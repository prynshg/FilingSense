"""Step 3 check: chunk the JPM 10-K and inspect the chunks."""
import json
from pathlib import Path

from filingsense.ingestion.chunking import chunk_document
from filingsense.ingestion.edgar import ingest_company

cache = Path("data/jpm_10k.json")
if cache.exists():
    doc = json.loads(cache.read_text(encoding="utf-8"))
    print("(loaded cached download)")
else:
    doc = ingest_company("JPM", limit=1)[0]
    cache.write_text(json.dumps(doc), encoding="utf-8")
    print("(downloaded fresh, cached to data/jpm_10k.json)")

chunks = chunk_document(doc)
print(f"chunks: {len(chunks)}")
sizes = [len(c["text"]) for c in chunks]
print(f"avg chars: {sum(sizes)/len(sizes):.0f} | max: {max(sizes):,} | min: {min(sizes)}")

table_chunks = [c for c in chunks if c["metadata"]["has_table"]]
print(f"chunks containing tables: {len(table_chunks)}")

# Sanity check: a complete pipe-row starts AND ends with '|'.
# A row split mid-way would leave a fragment failing this.
broken = sum(
    1 for c in chunks for line in c["text"].splitlines()
    if line.strip().startswith("|") and not line.strip().endswith("|")
)
print(f"split-in-half table rows: {broken}")

print("\n--- sample table chunk ---")
print(table_chunks[5]["text"][:900])
