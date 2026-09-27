"""LangGraph pipeline: answerable? -> route -> retrieve -> analyst -> critic.

The decision gate (Jev) sits at every branch point; the Gemini analyst
only writes prose. State, arithmetic, and control flow stay in plain code.

Deliberately NO per-chunk rerank node: at ~4.5s per FreeJev decision,
scoring a dozen chunks would cost a minute per query. RRF fusion already
ranks; Step 13's eval will say whether reranking earns its cost.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from filingsense.agents import analyst as analyst_mod
from filingsense.agents import critic as critic_mod
from filingsense.agents.analyst import format_evidence
from filingsense.decisions.base import DecisionGate, get_gate
from filingsense.retrieval.hybrid import HybridRetriever
from filingsense.xbrl.client import fetch_company_facts
from filingsense.xbrl.facts import get_fact
from filingsense.config import settings
from filingsense.observability.tracing import get_tracer


ABSTAIN_BELOW = 0.6
CIK = "19617"  # JPMorgan Chase & Co. filer CIK (Step 5)
FY = 2025

ROUTES = ["10-K text", "XBRL facts", "earnings call transcript"]

# Source descriptions for the router (Step 13). The answerability gate got
# these in Step 11; the router was still choosing from bare names, so
# "how does management explain..." questions drifted to the call even when
# the golden intent was MD&A discussion in the 10-K.
ROUTE_DESCRIPTIONS = {
    "10-K text": (
        "the audited annual filing: MD&A discussion of results and drivers, "
        "risk factors (Item 1A), business segments, accounting policies, "
        "legal proceedings"
    ),
    "XBRL facts": (
        "exact reported figures only: total assets, provision for credit "
        "losses, net income, deposits (FY2025)"
    ),
    "earnings call transcript": (
        "management's spoken remarks on the Q4/FY2025 call (~10 min): "
        "results overview, consumer health, card business, outlook"
    ),
}


SOURCES_STATE = (
    "Available sources: (1) JPMorgan's FY2025 10-K filing text, "
    "(2) XBRL structured facts (exact figures for: total assets, "
    "provision for credit losses, net income, total deposits), "
    "(3) Q4/FY2025 earnings call transcript."
)

# Minimal keyword -> XBRL tag map (Step 5 tags). The router picks the source;
# this picks which facts to pull. Unmapped questions get all four facts and
# the analyst abstains if none answer the question.
TAG_KEYWORDS = {
    "Assets": ["asset"],
    "ProvisionForLoanLeaseAndOtherLosses":
        ["provision", "credit loss", "loan loss", "allowance"],
    "NetIncomeLoss": ["net income", "net earnings", "profit"],
    "Deposits": ["deposit"],
}
TAG_LABELS = {
    "Assets": "Total assets",
    "ProvisionForLoanLeaseAndOtherLosses": "Provision for credit losses",
    "NetIncomeLoss": "Net income",
    "Deposits": "Total deposits",
}

_xbrl_cache: dict | None = None


def _xbrl_data() -> dict:
    """Fetch once per process; the SEC API is the slow part."""
    global _xbrl_cache
    if _xbrl_cache is None:
        _xbrl_cache = fetch_company_facts(CIK)
    return _xbrl_cache


def _xbrl_chunks(question: str) -> list[dict]:
    """XBRL facts as pseudo-chunks, so analyst/critic treat them uniformly."""
    data = _xbrl_data()
    q = question.lower()
    tags = [t for t, kws in TAG_KEYWORDS.items()
            if any(k in q for k in kws)] or list(TAG_KEYWORDS)
    chunks = []
    for tag in tags:
        fact = get_fact(data, tag, fy=FY)
        if fact is None:
            continue
        chunks.append({
            "id": f"XBRL:{tag}",
            "text": (f"{TAG_LABELS[tag]} (FY{FY}, 10-K): ${fact['val']:,} "
                     f"(filed {fact['filed']})"),
            "metadata": {"source": "XBRL"},
        })
    return chunks


class State(TypedDict, total=False):
    question: str
    answerable: bool
    route: str                       # one of ROUTES
    chunks: list[dict]                # retrieved evidence (or XBRL pseudo-chunks)
    answer: dict                     # AnalystAnswer as dict
    critic: dict                     # verdicts summary
    decisions: list[dict]            # audit log of every gate call
    final: str


def _log(state: State, node: str, d) -> None:
    state.setdefault("decisions", []).append({
        "node": node,
        "gate": d.model,
        "value": d.value,
        "confidence": round(d.confidence, 3),
        "latency_ms": round(d.latency_ms, 1),
        "attempts": d.attempts
    })

def _span_attrs(d) -> dict:
    """Decision -> trace span attributes (latency, cost, confidence)."""
    return {
        "model": d.model,
        "confidence": round(d.confidence, 3),
        "attempts": d.attempts,
        "credits_est": d.raw.get("credits_est"),
        "cost_usd_est": d.raw.get("cost_usd_est"),
    }


def build_graph(retriever: HybridRetriever, gate: DecisionGate | None = None):
    gate = gate or get_gate()
    tracer = get_tracer()

    def check_answerable(state: State) -> State:
        with tracer.span("check_answerable") as s:
            d = gate.ask_yes_no(
            f'User query: "{state["question"]}". '
            "Can this be answered from the available sources?",
            SOURCES_STATE,
            )

            _log(state, "check_answerable", d)
            s.update(_span_attrs(d))
            state["answerable"] = bool(d.value)
            if not d.value and d.confidence >= ABSTAIN_BELOW:
                state["final"] = "Not in the filings."
        return state

    def route(state: State) -> State:
        with tracer.span("route") as s:
            options = "\n".join(f"- {r}: {ROUTE_DESCRIPTIONS[r]}" for r in ROUTES)
            d = gate.ask_choice(
                "Which source best answers this question?\n"
                f"Sources:\n{options}",
                ROUTES,
                state["question"],
            )
            _log(state, "route", d)
            s.update(_span_attrs(d))
            s["choice"] = str(d.value)
            state["route"] = str(d.value)
        return state


    def retrieve(state: State) -> State:
        with tracer.span("retrieve", route=state["route"]) as s:
            if state["route"] == "XBRL facts":
                state["chunks"] = _xbrl_chunks(state["question"])
            elif state["route"] == "earnings call transcript":
                # Route-aware: rank within the call slice, don't filter the
                # full-corpus top-k down to nothing (Step 13 fix).
                state["chunks"] = retriever.search_within(
                    state["question"], source="earnings_call", top_k=6)
            else:
                state["chunks"] = retriever.search(state["question"], top_k=8)
            s["n_chunks"] = len(state["chunks"])
        return state


    def run_analyst_node(state: State) -> State:
        with tracer.span("analyst", model=settings.analyst_model) as s:
            ans = analyst_mod.run_analyst(
                state["question"], format_evidence(state["chunks"]))
            state["answer"] = ans.model_dump()
            s.update(analyst_mod.get_last_usage())
            s["abstained"] = ans.abstained
            s["n_claims"] = len(ans.claims)
            if ans.abstained:
                state["final"] = ans.answer
        return state

    def run_critic_node(state: State) -> State:
        with tracer.span("critic", model=settings.critic_model) as s:
            ans = analyst_mod.AnalystAnswer(**state["answer"])
            rep = critic_mod.run_critic(ans, state["chunks"], gate)
            state["critic"] = {
                "all_supported": rep.all_supported,
                "escalations": len(rep.escalations),
                "verdicts": [
                    {"claim": v.claim[:120], "supported": v.supported,
                     "confidence": round(v.confidence, 3), "escalated": v.escalated}
                    for v in rep.verdicts
                ],
            }
            for v in rep.verdicts:
                if v.decision is not None:
                    _log(state, "critic", v.decision)
            s["n_claims"] = len(rep.verdicts)
            s["n_escalated"] = len(rep.escalations)
            s["all_supported"] = rep.all_supported
            # Critic's per-claim gate calls have no child spans (they run
            # inside run_critic); roll their cost up here instead.
            s["credits_est"] = round(sum(
                (v.decision.raw.get("credits_est") or 0)
                for v in rep.verdicts if v.decision is not None), 4)
            final = state["answer"]["answer"]
            if rep.escalations:
                final += (f"\n\n[Flagged for review: {len(rep.escalations)} "
                          f"claim(s) below confidence threshold.]")
            state["final"] = final
        return state

    g = StateGraph(State)
    g.add_node("check_answerable", check_answerable)
    g.add_node("route", route)
    g.add_node("retrieve", retrieve)
    g.add_node("analyst", run_analyst_node)
    g.add_node("critic", run_critic_node)
    g.set_entry_point("check_answerable")
    g.add_conditional_edges(
        "check_answerable",
        lambda s: "end" if "final" in s else "route",
        {"end": END, "route": "route"},
    )
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "analyst")
    g.add_conditional_edges(
        "analyst",
        lambda s: "end" if s.get("answer", {}).get("abstained") else "critic",
        {"end": END, "critic": "critic"},
    )
    g.add_edge("critic", END)
    return g.compile()


def invoke_with_trace(graph, question: str) -> dict:
    """One trace per question; every node records a child span.

    Use this instead of graph.invoke() in ask.py / eval.py so runs are
    observable by default (Step 14).
    """
    with get_tracer().trace("filingsense.ask", question=question):
        return graph.invoke({"question": question})
