"""
Configuration — reads from environment variables / .env file.
All Neo4j and Ollama settings can be overridden at runtime.
"""
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Neo4j ──────────────────────────────────────────────────────────────
    neo4j_uri: str      = Field("neo4j+s://785ea338.databases.neo4j.io", alias="NEO4J_URI")
    neo4j_user: str     = Field("785ea338",                              alias="NEO4J_USERNAME")
    neo4j_password: str = Field(...,                                     alias="NEO4J_PASSWORD")
    neo4j_database: str = Field("785ea338",                              alias="NEO4J_DATABASE")

    # ── Ollama / LLM ───────────────────────────────────────────────────────
    ollama_host: str           = Field("http://localhost:11434", alias="OLLAMA_HOST")
    llm_model: str             = Field("qwen3:4b",               alias="OLLAMA_MODEL")
    llm_temp_extract: float    = 0.0    # metadata extraction — deterministic
    llm_temp_answer: float     = 0.3    # answer generation

    # ── Embeddings ─────────────────────────────────────────────────────────
    embed_model_name: str = Field("BAAI/bge-m3", alias="EMBED_MODEL")
    embed_use_fp16: bool  = True
    embed_max_length: int = 512

    # ── RAG tuning ─────────────────────────────────────────────────────────
    retrieval_top_k: int       = 15     # candidates fetched per signal
    answer_top_k: int          = 5      # articles passed to the LLM
    sim_threshold: float       = 0.75   # minimum semantic score to keep

    # ── API server ─────────────────────────────────────────────────────────
    api_host: str        = "0.0.0.0"
    api_port: int        = 8000
    cors_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — import and call wherever settings are needed."""
    return Settings()
