"""Step 4 check: index chunks into Qdrant + BM25, run hybrid search."""
import json
import pickle
import time
from pathlib import Path

from filingsense.ingestion.chunking import chunk_document
from filingsense.retrieval.hybrid import HybridRetriever

doc = json.loads(Path("data/jpm_10k.json").read_text(encoding="utf-8"))
chunks = chunk_document(doc)
print(f"chunks: {len(chunks)}")

retriever = HybridRetriever()
t0 = time.time()
chunks = retriever.upsert(chunks)  # embed + Qdrant upsert + BM25 build
print(f"indexed in {time.time() - t0:.0f}s")

# Sidecar so later steps (API server) can rebuild BM25 without re-embedding.
Path("data/chunks.pkl").write_bytes(pickle.dumps({"chunks": chunks}))
print("saved data/chunks.pkl")

queries = [
    "What was the provision for credit losses in 2025?",
    "JPMorgan Chase total assets as of December 31 2025",
    "How does the firm describe its approach to climate risk?",
]
for q in queries:
    print(f"\n=== {q} ===")
    for i, c in enumerate(retriever.search(q, top_k=3)):
        snippet = c["text"][:200].replace("\n", " ")
        print(f"[{i+1}] chunk {c['metadata']['chunk_id']}: {snippet}...")
