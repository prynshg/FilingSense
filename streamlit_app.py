"""FilingSense UI: answers with citations, plus the audit trail.

The part recruiters actually read isn't the answer — it's the receipts:
which source was chosen, what each gate decided, what the critic verified.

Run the API first (Step 15), then from the project root:
    streamlit run streamlit_app.py
Point at a deployed API with (PowerShell):
    $env:FILINGSENSE_API_URL = "https://your-app.onrender.com"
"""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("FILINGSENSE_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="FilingSense", layout="wide")
st.title("FilingSense — 10-K Q&A with receipts")
st.caption(
    "Multi-agent RAG over JPMorgan's FY2025 10-K, XBRL facts, and earnings call. "
    "Every answer ships with its decision audit log."
)

EXAMPLES = {
    "Provision (XBRL)": "What was the provision for credit losses in 2025?",
    "Checking accounts (call)": (
        "How many net new checking accounts did JPMorgan add in 2025, per the call?"
    ),
    "MD&A (10-K text)": (
        "How does management explain the change in provision for credit losses?"
    ),
    "Unanswerable (abstains)": "What was Apple's revenue in 2025?",
}

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", API_URL)
    st.header("Try one")
    for label, q in EXAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state["question"] = q

question = st.text_input("Ask about the filing", key="question")

if st.button("Ask", type="primary") and question:
    try:
        with st.spinner("Analyst → Critic are working (takes ~20–60s)..."):
            r = httpx.post(f"{api_url}/ask", json={"question": question},
                           timeout=180.0)
        r.raise_for_status()
    except httpx.ConnectError:
        st.error(
            f"Can't reach the API at {api_url}. Start it first:\n\n"
            "`python -m uvicorn filingsense.api:app --port 8000`"
        )
        st.stop()
    except httpx.TimeoutException:
        st.error("The pipeline timed out (>3 min) — the decision API may be "
                 "having a bad hour. Try again.")
        st.stop()
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text[:300]}")
        st.stop()

    data = r.json()
    if data["answer"].strip() == "Not in the filings.":
        st.warning("Not in the filings. The system abstained rather than guessing.")
    else:
        st.subheader("Answer")
        st.write(data["answer"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latency", f"{data['latency_ms'] / 1000:.1f}s")
    c2.metric("Route", data["route"] or "—")
    c3.metric("Gate decisions", len(data["decisions"]))
    c4.metric("Critic escalations", data["critic"].get("escalations", "—"))

    with st.expander("Decision audit log"):
        st.dataframe(data["decisions"], use_container_width=True)
    with st.expander("Critic report"):
        st.json(data["critic"])
