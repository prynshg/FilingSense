"""TypeSafe Jev: the System One decision layer.

Jev takes state + typed questions and returns typed values with calibrated
probabilities — no text generation. Used here for: abstention gating, chunk
reranking, claim verification, and query routing.

Wire shape verified against https://docs.typesafe.ai (quickstart) on
2026-09-26 — the endpoint is /v1/systemone (not /v1/decide), questions are
a dict keyed by caller-chosen IDs, and Noul answers carry no confidence::

    POST https://api.typesafe.ai/v1/systemone
    {"state": ..., "model": "jev-1.13.0",
     "questions": {"<id>": {"type": "noul|choice|score",
                            "instructions": "...",
                            "criteria": {...} | [...]}}}
    -> {"model": "jev-1.13.0",
        "answers": {"<id>": {"type": ..., "choice"|"noul"|"score": ...,
                             "confidence": ..., "probabilities": {...},
                             "legend": {...}}},
        "usage": {"input_tokens": ..., "output_tokens": ...}}

We call the REST API directly (instead of typesafe-sdk) to keep the wire
shape visible and the dependency surface minimal — this file isolates the
API surface so only it changes if TypeSafe evolves it.

Pin TYPESAFE_MODEL_VERSION (jev-latest moves) and log every decision
(see observability).
"""
from __future__ import annotations

import json
import time
import uuid

import httpx

from filingsense.config import settings
from filingsense.decisions.base import Decision
from filingsense.observability.tracing import estimate_tokens

# Generic rubric for ask_score when the caller only gives a level count.
_SCORE_LABELS = ["very unlikely", "unlikely", "neutral", "likely", "very likely"]

def _retryable(exc: Exception) -> bool:
    """Timeouts, dropped connections, 429 and 5xx are worth retrying.
    Other 4xx (400/401/404) won't fix themselves — fail fast."""
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


def _post_with_retry(client: httpx.Client, url: str, payload: dict,
                     max_attempts: int = 3) -> tuple[httpx.Response, int]:
    """POST with exponential backoff on transient failures.

    FreeJev's request_id makes retries safe against double-billing a
    dropped connection. Returns (response, attempts).
    """
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp, attempt
        except (httpx.TimeoutException, httpx.ConnectError,
                httpx.HTTPStatusError) as e:
            if attempt < max_attempts and _retryable(e):
                time.sleep(2 ** (attempt - 1))
                continue
            raise
    raise AssertionError("unreachable")  # keeps type-checkers calm


def _noul_decision(ans: dict, model: str, latency_ms: float, raw: dict) -> Decision:
    p_yes = float(ans["noul"])
    # Noul carries no confidence field; concentration of the
    # distribution is |p - 0.5| * 2  (1.0 at certainty, 0.0 at 0.5).
    return Decision(
        value=p_yes >= 0.5,
        confidence=abs(p_yes - 0.5) * 2.0,
        probabilities={"yes": p_yes, "no": 1.0 - p_yes},
        model=model,
        latency_ms=latency_ms,
        raw=raw,
    )


def _choice_decision(ans: dict, model: str, latency_ms: float, raw: dict) -> Decision:
    return Decision(
        value=ans["choice"],
        confidence=float(ans.get("confidence", 0.0)),
        probabilities={k: float(v) for k, v in ans.get("probabilities", {}).items()},
        model=model,
        latency_ms=latency_ms,
        raw=raw,
    )


def _score_decision(ans: dict, model: str, latency_ms: float, raw: dict) -> Decision:
    return Decision(
        # Score is a probability-weighted rubric position and may be
        # fractional (e.g. 1.6) on a 0-indexed legend — by design.
        value=ans["score"],
        confidence=float(ans.get("confidence", 0.0)),
        probabilities={k: float(v) for k, v in ans.get("probabilities", {}).items()},
        model=model,
        latency_ms=latency_ms,
        raw=raw,
    )


def _score_criteria(levels: int) -> list[str]:
    if not 2 <= levels <= 10:
        raise ValueError("Jev score levels must be 2..10")
    if levels <= len(_SCORE_LABELS):
        return _SCORE_LABELS[:levels]
    return [f"level {i}" for i in range(levels)]


class JevGate:
    name = "jev"

    def __init__(self):
        self.client = httpx.Client(
            base_url=settings.typesafe_base_url,
            headers={"Authorization": f"Bearer {settings.typesafe_api_key}"},
            timeout=10.0,
        )
        self.model_version = settings.typesafe_model_version or "jev-1.13.0"

    def _ask(self, qtype: str, question: str, state: str,
         criteria: object = None) -> tuple[dict, str, float, int, int, dict]:
        """One typed question; returns
        (answer, model, latency_ms, attempts, tokens_in_est, raw_body)."""
        t0 = time.perf_counter()
        q: dict = {"type": qtype, "instructions": question}
        if criteria is not None:
            q["criteria"] = criteria
        payload = {
            "state": state,
            "model": self.model_version,
            "questions": {"q": q},
        }
        resp, attempts = _post_with_retry(self.client, "/v1/systemone", payload)
        body = resp.json()
        latency_ms = (time.perf_counter() - t0) * 1000
        tokens_in = estimate_tokens(json.dumps(payload))
        return (body["answers"]["q"], body.get("model", ""), latency_ms,
                attempts, tokens_in, body)

    def _tag(self, model: str) -> str:
        return f"jev@{model or self.model_version}"

    def ask_yes_no(self, question: str, state: str) -> Decision:
        ans, model, latency_ms, attempts, tokens_in, raw = self._ask(
            "noul", question, state)
        d = _noul_decision(ans, self._tag(model), latency_ms, raw)
        d.attempts = attempts
        d.raw["tokens_in_est"] = tokens_in
        d.raw["cost_usd_est"] = round(tokens_in / 1e6 * 0.042, 6)  # $0.042/M in
        return d

    def ask_choice(self, question: str, options: list[str], state: str) -> Decision:
        if len(options) > 255:
            raise ValueError("Jev supports at most 255 options")
        ans, model, latency_ms, attempts, tokens_in, raw = self._ask(
            "choice", question, state, criteria={o: o for o in options})
        d = _choice_decision(ans, self._tag(model), latency_ms, raw)
        d.attempts = attempts
        d.raw["tokens_in_est"] = tokens_in
        d.raw["cost_usd_est"] = round(tokens_in / 1e6 * 0.042, 6)
        return d

    def ask_score(self, question: str, state: str, levels: int = 5) -> Decision:
        ans, model, latency_ms, attempts, tokens_in, raw = self._ask(
            "score", question, state, criteria=_score_criteria(levels))
        d = _score_decision(ans, self._tag(model), latency_ms, raw)
        d.attempts = attempts
        d.raw["tokens_in_est"] = tokens_in
        d.raw["cost_usd_est"] = round(tokens_in / 1e6 * 0.042, 6)
        return d



class FreeJevGate:
    """FreeJev (freejev.org): free third-party Jev access.

    Same typed questions as TypeSafe, different envelope — reverse-engineered
    from the open-source freejev-client (npm) on 2026-09-26::

        POST https://freejev.org/api/v1/decisions
        Authorization: Bearer <FREEJEV_API_KEY>
        {"state": ..., "questions": {"<id>": {...}}, "request_id": "<uuid>"}

    Differences from TypeSafe direct: no `model` field (they serve one
    always-current alias); `request_id` is required and makes retries
    idempotent (reuse it after a dropped connection — results aren't stored,
    but you won't be billed twice). The response is expected in the
    TypeSafe shape (answers keyed by question id); if it ever differs, the
    error below prints the raw body so the mapping can be fixed.
    """
    name = "freejev"

    def __init__(self):
        self.client = httpx.Client(
            base_url=settings.freejev_base_url,
            headers={"Authorization": f"Bearer {settings.freejev_api_key}"},
            timeout=30.0,
        )

    def _ask(self, qtype: str, question: str, state: str,
         criteria: object = None) -> tuple[dict, float, int, int, dict]:
        """One typed question; returns
        (answer, latency_ms, attempts, tokens_in_est, raw_body)."""
        t0 = time.perf_counter()
        q: dict = {"type": qtype, "instructions": question}
        if criteria is not None:
            q["criteria"] = criteria
        payload = {
            "state": state,
            "questions": {"q": q},
            "request_id": uuid.uuid4().hex,
        }
        resp, attempts = _post_with_retry(
            self.client, "/api/v1/decisions", payload)
        body = resp.json()
        latency_ms = (time.perf_counter() - t0) * 1000
        tokens_in = estimate_tokens(json.dumps(payload))
        try:
            ans = body["answers"]["q"]
        except (KeyError, TypeError):
            raise RuntimeError(
                "Unexpected FreeJev response shape: "
                + json.dumps(body)[:2000]
            )
        return ans, latency_ms, attempts, tokens_in, body

    def ask_yes_no(self, question: str, state: str) -> Decision:
        ans, latency_ms, attempts, tokens_in, raw = self._ask("noul", question, state)
        d = _noul_decision(ans, "freejev@jev", latency_ms, raw)
        d.attempts = attempts
        d.raw["tokens_in_est"] = tokens_in
        d.raw["credits_est"] = round(tokens_in / 1000, 4)  # 1 credit / 1K in
        return d

    def ask_choice(self, question: str, options: list[str], state: str) -> Decision:
        if len(options) > 255:
            raise ValueError("Jev supports at most 255 options")
        ans, latency_ms, attempts, tokens_in, raw = self._ask(
            "choice", question, state, criteria={o: o for o in options})
        d = _choice_decision(ans, "freejev@jev", latency_ms, raw)
        d.attempts = attempts
        d.raw["tokens_in_est"] = tokens_in
        d.raw["credits_est"] = round(tokens_in / 1000, 4)
        return d

    def ask_score(self, question: str, state: str, levels: int = 5) -> Decision:
        ans, latency_ms, attempts, tokens_in, raw = self._ask(
            "score", question, state, criteria=_score_criteria(levels))
        d = _score_decision(ans, "freejev@jev", latency_ms, raw)
        d.attempts = attempts
        d.raw["tokens_in_est"] = tokens_in
        d.raw["credits_est"] = round(tokens_in / 1000, 4)
        return d

