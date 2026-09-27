"""Step 8: the same five gate decisions, now through a real Jev engine.

Needs TYPESAFE_API_KEY or FREEJEV_API_KEY in .env. Refuses to run on the
LLM fallback — the whole point is to see Jev's own answers, confidences,
and latency.
"""
import sys

from filingsense.decisions.base import get_gate

STATE = (
    "Available sources: (1) JPMorgan Chase FY2025 10-K filing text — narrative "
    "plus tables; (2) SEC XBRL structured facts for JPM FY2025 — exact figures "
    "for assets, provisions, income, deposits; (3) transcript of the JPM Q4/FY2025 "
    "earnings call (first 10 minutes: introductions and headline results). "
    "No web access, no other companies, no future or predicted data."
)

QUERIES = [
    "What was JPMorgan's provision for credit losses in 2025?",
    "What did management say about the economic outlook on the earnings call?",
    "What are JPMorgan's main risk factors?",
    "What was Apple's revenue in 2025?",
    "What will JPM's stock price be next month?",
]


def main() -> None:
    gate = get_gate()
    if gate.name not in ("jev", "freejev"):
        print("No Jev key set — nothing to test.")
        print("Add TYPESAFE_API_KEY or FREEJEV_API_KEY to .env, then re-run.")
        sys.exit(2)
    print(f"gate: {gate.name}\n")
    for q in QUERIES:
        print(f"Q: {q}")
        answerable = gate.ask_yes_no(
            f'User query: "{q}". Can this be answered from the available sources?',
            state=STATE,
        )
        print(f"  answerable: {answerable.value} "
              f"(confidence {answerable.confidence:.2f}, {answerable.latency_ms:.0f}ms)")
        if answerable.value:
            route = gate.ask_choice(
                f'User query: "{q}". Which source should lead the answer?',
                options=["10-K text", "XBRL facts", "earnings call transcript"],
                state=STATE,
            )
            print(f"  route to:   {route.value} "
                  f"(confidence {route.confidence:.2f}, {route.latency_ms:.0f}ms)")
        else:
            print("  route to:   — (abstain)")
        print()


if __name__ == "__main__":
    main()
