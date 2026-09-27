"""Qdrant vector store: collection setup + chunk upserts.

Point ids are deterministic UUIDs derived from the chunk id, so re-running
ingestion never creates duplicates.
"""
from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from filingsense.config import settings

EMBEDDING_DIM = 384  # BAAI/bge-small-en-v1.5 via fastembed
NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/filingsense")


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, chunk_id))


def get_client() -> QdrantClient:
    if not settings.qdrant_url or not settings.qdrant_api_key:
        raise RuntimeError("Set QDRANT_URL and QDRANT_API_KEY in .env")
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=60,
    )


def ensure_collection(client: QdrantClient | None = None) -> None:
    client = client or get_client()
    if not client.collection_exists(settings.qdrant_collection):
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def upsert_chunks(chunks: list[dict], vectors: list[list[float]]) -> None:
    """Upsert in small batches — one giant request can time out on write."""
    client = get_client()
    ensure_collection(client)
    total = 0
    for i in range(0, len(chunks), 100):
        batch = chunks[i : i + 100]
        vecs = vectors[i : i + 100]
        points = [
            PointStruct(
                id=c.get("_point_id") or point_id(c["id"]),
                vector=v,
                payload={"text": c["text"], **c["metadata"]},
            )
            for c, v in zip(batch, vecs)
        ]
        client.upsert(collection_name=settings.qdrant_collection, points=points)
        total += len(points)
        print(f"  upserted {total}/{len(chunks)} points", flush=True)
