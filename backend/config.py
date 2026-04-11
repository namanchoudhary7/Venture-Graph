import os
from dotenv import load_dotenv

load_dotenv()

class Settings():
    """System-wide configuration settings."""

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "huggingface")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))


    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str          = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str          = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL: str          = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    # LLM Engine Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    HUGGINGFACE_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
    HUGGINGFACE_API: str | None = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # Graph Execution Limits
    MAX_REVISIONS: int = int(os.getenv("MAX_REVISIONS", "6"))

    # API Keys
    FIRECRAWL_API_KEY: str | None = os.getenv("FIRECRAWL_API_KEY")
    GITHUB_TOKEN: str | None = os.getenv("GITHUB_TOKEN")

    # --- LangSmith Observability ---
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    LANGCHAIN_API_KEY: str | None = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "venture-graph")

settings = Settings()