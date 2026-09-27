"""Central configuration. Everything comes from the environment; see .env.example."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Embeddings (fastembed runs locally — no key needed)
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Qdrant Cloud (URL + key from cloud.qdrant.io)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "filingsense"

    # SEC EDGAR (the SEC requires a descriptive User-Agent)
    sec_user_agent: str = "FilingSense/0.1 (contact: you@example.com)"

    # --- Decision layer ---
    typesafe_api_key: str = ""
    typesafe_base_url: str = "https://api.typesafe.ai"
    typesafe_model_version: str = "jev-1.13.0"  # pinned 2026-09-26; jev-latest moves
    jev_enabled: bool = True

    # FreeJev (freejev.org) — free third-party Jev access, 10k credits on signup.
    freejev_api_key: str = ""
    freejev_base_url: str = "https://freejev.org"


    # Langfuse observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # LLM — Google Gemini free tier (no card needed)
    gemini_api_key: str = ""
    analyst_model: str = "gemini-3.5-flash-lite"
    critic_model: str = "gemini-3.5-flash-lite"



settings = Settings()
