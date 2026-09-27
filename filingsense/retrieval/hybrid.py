"""Hybrid retrieval: dense (Qdrant ANN) + sparse (BM25, in-memory), fused with RRF.

BM25 lives in memory for the scaffold — Qdrant holds the dense side and acts
as the ANN index. Move BM25 to Qdrant sparse vectors when the corpus grows
past a few filings.
"""
from __future__ import annotations

from pathlib import Path

from rank_bm25 import BM25Okapi

from filingsense.config import settings
from filingsense.retrieval.store import get_client, point_id


def rrf(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for rank, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class HybridRetriever:
    def __init__(self):
        from fastembed import TextEmbedding

        self.embedder = TextEmbedding(model_name=settings.embedding_model,threads=1)
        self.client = get_client()
        self._chunks: list[dict] = []
        self._id2pos: dict[str, int] = {}
        self._bm25: BM25Okapi | None = None

    # ---- indexing ----
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [e.tolist() for e in self.embedder.embed(texts)]

    def upsert(self, chunks: list[dict]) -> list[dict]:
        """Embed + write to Qdrant, then build the BM25 side."""
        from filingsense.retrieval import store

        chunks = [dict(c, _point_id=point_id(c["id"])) for c in chunks]
        store.upsert_chunks(chunks, self.embed([c["text"] for c in chunks]))
        self.build_bm25(chunks)
        return chunks

    def build_bm25(self, chunks: list[dict]) -> None:
        self._chunks = chunks
        self._id2pos = {c["_point_id"]: i for i, c in enumerate(chunks)}
        self._bm25 = BM25Okapi([c["text"].lower().split() for c in chunks])

    def load_sidecar(self, path: str | Path) -> None:
        """Rebuild BM25 + id map from cached chunks WITHOUT re-upserting.

        Use this in the API server: the vectors already live in Qdrant.
        """
        import pickle

        with open(path, "rb") as f:
            self.build_bm25(pickle.load(f)["chunks"])

    # ---- search ----
    def search(self, query: str, top_k: int = 10,
               dense_k: int = 20, sparse_k: int = 20) -> list[dict]:
        """Chunk dicts ranked by RRF fusion of dense and sparse ranks."""
        if self._bm25 is None:
            raise RuntimeError("Retriever has no index: call upsert() or load_sidecar() first.")
        qvec = next(self.embedder.embed([query])).tolist()
        dense = self.client.query_points(
            collection_name=settings.qdrant_collection,
            query=qvec, limit=dense_k,
        ).points
        dense_ids = [str(p.id) for p in dense]

        scores = self._bm25.get_scores(query.lower().split())
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        sparse_ids = [self._chunks[i]["_point_id"] for i in order[:sparse_k]]

        fused = rrf([dense_ids, sparse_ids])[:top_k]
        return [self._chunks[self._id2pos[pid]]
                for pid in fused if pid in self._id2pos]
    
    def search_within(self, query: str, source: str,
                  top_k: int = 6) -> list[dict]:

        subset = [c for c in self._chunks
                if (c.get("metadata", {}) or {}).get("source") == source]
        if not subset:
            return self.search(query, top_k=top_k)  # no slice: don't strand
        if len(subset) <= top_k:
            return subset  # slice fits in context; ranking adds nothing
        bm25 = BM25Okapi([c["text"].lower().split() for c in subset])
        scores = bm25.get_scores(query.lower().split())
        order = sorted(range(len(subset)), key=lambda i: scores[i],
                    reverse=True)
        return [subset[i] for i in order[:top_k]]
