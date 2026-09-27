#!/bin/bash
# One container, two processes: the FastAPI backend (internal) and the
# Streamlit UI (public). Render sets $PORT for the public port; HF Spaces
# expects 7860; the default covers local `docker run`.
set -e
UI_PORT="${PORT:-7860}"
python -m uvicorn filingsense.api:app --host 0.0.0.0 --port 8000 &
export FILINGSENSE_API_URL="http://localhost:8000"
exec python -m streamlit run streamlit_app.py \
  --server.port "$UI_PORT" --server.address 0.0.0.0 --server.headless true
