"""Dump the earnings-call chunks from Qdrant so Step 12's golden-set
questions about the call can be grounded in what the transcript actually
covers (not what we imagine it covers)."""
from qdrant_client import QdrantClient
from qdrant_client.http import models

from filingsense.config import settings

client = QdrantClient(url=settings.qdrant_url,
                      api_key=settings.qdrant_api_key, timeout=60)

CALL_FILTER = models.Filter(must=[
    models.FieldCondition(key="source",
                          match=models.MatchValue(value="earnings_call"))
])


def _scroll_filtered():
    """Filtered scroll needs a payload index on `source` — create it."""
    client.create_payload_index(
        collection_name=settings.qdrant_collection,
        field_name="source",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    points, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=CALL_FILTER,
        limit=10,
        with_payload=True,
        with_vectors=False,
    )
    return points


def _scroll_client_side():
    """No-index fallback: scroll everything, filter in Python."""
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break
    return [p for p in points if p.payload.get("source") == "earnings_call"]


try:
    points = _scroll_filtered()
except Exception as e:
    print(f"(indexed filter unavailable: {e}\n falling back to client-side filter)")
    points = _scroll_client_side()

print(f"found {len(points)} earnings-call chunks\n")
for p in points:
    md = {k: v for k, v in p.payload.items() if k != "text"}
    print(f"--- {md} ---")
    print(p.payload.get("text", "")[:800])
    print()
