import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """
    Single source of truth for all project configuration.
    """

    # ── LLM Configuration (Groq) ──────────────────────────────────────────────
    LLM_TEMPERATURE: float   = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str          = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # ── Tool API Keys ─────────────────────────────────────────────────────────
    FIRECRAWL_API_KEY: str | None = os.getenv("FIRECRAWL_API_KEY")
    GITHUB_TOKEN: str | None      = os.getenv("GITHUB_TOKEN")

    # ── Graph limits ──────────────────────────────────────────────────────────
    MAX_REVISIONS: int = int(os.getenv("MAX_REVISIONS", "6"))
    AGENT_MODE: str    = os.getenv("AGENT_MODE", "parallel") 

    # ── LangSmith ─────────────────────────────────────────────────────────────
    LANGCHAIN_TRACING_V2: str     = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_ENDPOINT: str       = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    LANGCHAIN_API_KEY: str | None = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT: str        = os.getenv("LANGCHAIN_PROJECT", "venture-graph")


settings = Settings()