import os
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

class Settings():
    """System-wide configuration settings."""

    # LLM Engine Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    HUGGINGFACE_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
    HUGGINGFACE_API: str | None = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # Graph Execution Limits
    MAX_REVISIONS: int = int(os.getenv("MAX_REVISIONS", "6"))

    # API Keys (Centralizing validation)
    FIRECRAWL_API_KEY: str | None = os.getenv("FIRECRAWL_API_KEY")
    GITHUB_TOKEN: str | None = os.getenv("GITHUB_TOKEN")

# Instantiate a global settings object
settings = Settings()