"""One-time fix: rebuild data/chunks.pkl to include the earnings-call chunks.

The call chunks were upserted to Qdrant in Step 6 but never added to the
BM25 sidecar, so sidecar-based search silently dropped them (Step 9 note).
Chunk IDs are deterministic (jpm_call_000, ...) and point UUIDs derive from
them, so rebuilding the dicts from Qdrant payloads restores the exact mapping
— no re-transcription, no re-embedding needed.
"""
import pickle

from filingsense.retrieval.store import point_id

from qdrant_client import QdrantClient
from qdrant_client.http import models

from filingsense.config import settings

SIDECAR = "data/chunks.pkl"
META_KEYS = ("ticker", "source", "chunk_id", "start_sec", "end_sec",
             "has_tables")

client = QdrantClient(url=settings.qdrant_url,
                      api_key=settings.qdrant_api_key, timeout=60)
try:
    client.create_payload_index(
        collection_name=settings.qdrant_collection,
        field_name="source",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
except Exception:
    pass  # index already exists or not permitted; client-side filter below

points, offset = [], None
while True:
    batch, offset = client.scroll(
        collection_name=settings.qdrant_collection,
        limit=100, offset=offset, with_payload=True, with_vectors=False)
    points.extend(batch)
    if offset is None:
        break
call_pts = [p for p in points if p.payload.get("source") == "earnings_call"]

with open(SIDECAR, "rb") as f:
    data = pickle.load(f)
chunks = data["chunks"]
# Backfill _point_id on any chunk missing it (deterministic, so recompute
# is safe for chunks that already have it).
for c in chunks:
    c.setdefault("_point_id", point_id(c["id"]))
have = {c["id"] for c in chunks}
added = 0
for p in sorted(call_pts, key=lambda p: p.payload.get("chunk_id", 0)):
    md = p.payload
    cid = f"jpm_call_{int(md['chunk_id']):03d}"
    if cid in have:
        continue
    chunks.append({
        "id": cid,
        "_point_id": point_id(cid),
        "text": md["text"],
        "metadata": {k: md[k] for k in META_KEYS if k in md},
    })
    added += 1
with open(SIDECAR, "rb") as f:
    data = pickle.load(f)
chunks = data["chunks"]
# Backfill _point_id on any chunk missing it (deterministic, so recompute
# is safe for chunks that already have it).
for c in chunks:
    c.setdefault("_point_id", point_id(c["id"]))
have = {c["id"] for c in chunks}
added = 0
for p in sorted(call_pts, key=lambda p: p.payload.get("chunk_id", 0)):
    md = p.payload
    cid = f"jpm_call_{int(md['chunk_id']):03d}"
    if cid in have:
        continue
    chunks.append({
        "id": cid,
        "_point_id": point_id(cid),
        "text": md["text"],
        "metadata": {k: md[k] for k in META_KEYS if k in md},
    })
    added += 1
with open(SIDECAR, "rb") as f:
    data = pickle.load(f)
chunks = data["chunks"]
# Backfill _point_id on any chunk missing it (deterministic, so recompute
# is safe for chunks that already have it).
for c in chunks:
    c.setdefault("_point_id", point_id(c["id"]))
have = {c["id"] for c in chunks}
added = 0
for p in sorted(call_pts, key=lambda p: p.payload.get("chunk_id", 0)):
    md = p.payload
    cid = f"jpm_call_{int(md['chunk_id']):03d}"
    if cid in have:
        continue
    chunks.append({
        "id": cid,
        "_point_id": point_id(cid),
        "text": md["text"],
        "metadata": {k: md[k] for k in META_KEYS if k in md},
    })
    added += 1


with open(SIDECAR, "wb") as f:
    pickle.dump(data, f)
print(f"sidecar now has {len(chunks)} chunks ({added} call chunks added)")
