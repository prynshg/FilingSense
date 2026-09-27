FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + the BM25 sidecar (vectors live in Qdrant Cloud, not in the image).
COPY filingsense/ filingsense/
COPY streamlit_app.py .
COPY data/chunks.pkl data/chunks.pkl
COPY start.sh .
# Normalize line endings: an editor on Windows may save start.sh as CRLF,
# which breaks bash inside the container.
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh

# Secrets are NEVER baked in: .env is dockerignored; keys come from the
# host environment (Render dashboard, or --env-file for local runs).
EXPOSE 7860 8000
CMD ["./start.sh"]
