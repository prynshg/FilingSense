"""Step 9: Analyst answers only from retrieved evidence, with citations.

Query 1 (answerable): provision for credit losses -> expects a cited figure.
Query 2 (unanswerable): Apple revenue -> expects abstention.
"""
import time

from filingsense.agents.analyst import format_evidence, run_analyst
from filingsense.retrieval.hybrid import HybridRetriever

retriever = HybridRetriever()
retriever.load_sidecar("data/chunks.pkl")  # BM25 side; vectors live in Qdrant

QUERIES = [
    "What was JPMorgan's provision for credit losses in 2025?",
    "What was Apple's revenue in 2025?",
]

for q in QUERIES:
    chunks = retriever.search(q, top_k=5)
    print(f"Q: {q}")
    print(f"  retrieved {len(chunks)} chunks: "
          f"{[c.get('id', '?') for c in chunks]}")
    t0 = time.perf_counter()
    ans = run_analyst(q, format_evidence(chunks))
    print(f"  ({(time.perf_counter() - t0):.0f}s, abstained={ans.abstained})")
    print(f"  A: {ans.answer}")
    for cl in ans.claims:
        print(f"    - {cl.text[:100]} {cl.evidence}")
    print()
