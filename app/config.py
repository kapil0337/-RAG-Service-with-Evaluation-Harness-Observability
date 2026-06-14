"""Centralized application configuration loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the RAG service.

    All values can be overridden via environment variables or a `.env` file.
    See `.env.example` for the full list of supported variables.
    """

    # --- Database ---
    database_url: str = "postgresql+psycopg2://rag:rag@localhost:5432/ragdb"

    # --- Groq (OpenAI-compatible API) ---
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # --- Reranking ---
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enable_reranking: bool = True

    # --- Chunking / retrieval defaults ---
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 5
    rerank_top_k: int = 3

    # --- Storage ---
    upload_dir: str = "./data/uploads"

    # --- Observability: Langfuse (optional) ---
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return a cached `Settings` instance.

    Cached so the environment is only parsed once and the embedding/reranker
    models can rely on a stable configuration object.
    """
    return Settings()


settings = get_settings()
