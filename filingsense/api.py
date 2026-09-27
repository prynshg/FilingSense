"""FastAPI backend: serve the FilingSense pipeline over HTTP.

The expensive resources — retriever (705-chunk sidecar + Qdrant client),
gate (httpx connection pool), compiled LangGraph — are built ONCE in the
lifespan and shared across requests. Building them per-request would
re-read the sidecar and re-handshake every HTTPS connection on each call.

Run from the project root:
    uvicorn filingsense.api:app --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from filingsense.agents.graph import build_graph, invoke_with_trace
from filingsense.decisions.base import get_gate
from filingsense.retrieval.hybrid import HybridRetriever


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class DecisionOut(BaseModel):
    node: str
    gate: str
    value: Any
    confidence: float
    latency_ms: float
    attempts: int


class AskResponse(BaseModel):
    question: str
    answer: str
    answerable: bool
    route: str | None
    decisions: list[DecisionOut]
    critic: dict
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    retriever = HybridRetriever()
    retriever.load_sidecar("data/chunks.pkl")  # BM25 side; vectors in Qdrant
    app.state.graph = build_graph(retriever, get_gate())
    yield


app = FastAPI(title="FilingSense", lifespan=lifespan)


@app.get("/")
def root():
    return {"message": "FilingSense API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    # Render / HF Spaces hit this to know the service is up (Step 17).
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # NOTE: plain `def`, not `async def`. FastAPI runs sync endpoints in a
    # threadpool, so this ~20s network-bound pipeline never blocks the
    # event loop for other requests.
    t0 = time.perf_counter()
    try:
        out = invoke_with_trace(app.state.graph, req.question)
    except Exception as e:  # noqa: BLE001 — surface pipeline errors as 502
        raise HTTPException(
            status_code=502, detail=f"pipeline failed: {type(e).__name__}: {e}"
        ) from e
    return AskResponse(
        question=req.question,
        answer=out.get("final", ""),
        answerable=bool(out.get("answerable")),
        route=out.get("route"),
        decisions=out.get("decisions", []),
        critic=out.get("critic", {}),
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
