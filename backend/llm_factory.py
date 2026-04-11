"""
backend/llm_factory.py
 
The ONLY place in the codebase where an LLM object is constructed.
 
Every node, worker, and eval script imports from here:
    from backend.llm_factory import get_llm, get_llm_with_tools, get_parser
 
To switch provider: change LLM_PROVIDER in your .env. Nothing else changes.
 
Supported providers
───────────────────
  ollama       → local Ollama server (free, private)
  huggingface  → HuggingFace Inference API
  openai       → OpenAI API  (gpt-4o-mini, gpt-4o, etc.)
  groq         → Groq API    (llama-3.3-70b — very fast)
  anthropic    → Anthropic   (claude-haiku, claude-sonnet, etc.)
 
Adding a new provider
─────────────────────
  1. Add its settings to config.py
  2. Add its env keys to .env.example
  3. Add an elif branch in _build_llm() below
  That's it — zero changes needed anywhere else.
"""

from functools import lru_cache
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
 
from backend.config import settings
from backend.schemas import VCEvaluationOutput

def _build_llm() -> BaseChatModel:
    """
    Internal factory — constructs and returns the appropriate LangChain
    chat model based on settings.LLM_PROVIDER.
    """
    provider = settings.LLM_PROVIDER.lower().strip()
    temp     = settings.LLM_TEMPERATURE
 
    # ── Ollama (local) ───────────────────────────────────────────────────────
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        print(f"[LLM] Provider: Ollama | Model: {settings.OLLAMA_MODEL}")
        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=temp,
        )
 
    # ── HuggingFace Inference API ────────────────────────────────────────────
    elif provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        if not settings.HUGGINGFACE_API:
            raise EnvironmentError(
                "LLM_PROVIDER=huggingface but HUGGINGFACEHUB_API_TOKEN is not set in .env"
            )
        print(f"[LLM] Provider: HuggingFace | Model: {settings.HUGGINGFACE_MODEL}")
        endpoint = HuggingFaceEndpoint(
            model=settings.HUGGINGFACE_MODEL,
            temperature=temp,
            huggingfacehub_api_token=settings.HUGGINGFACE_API,
        )
        return ChatHuggingFace(llm=endpoint)
 
    # ── OpenAI ───────────────────────────────────────────────────────────────
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        if not settings.OPENAI_API_KEY:
            raise EnvironmentError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set in .env"
            )
        print(f"[LLM] Provider: OpenAI | Model: {settings.OPENAI_MODEL}")
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=temp,
            api_key=settings.OPENAI_API_KEY,
        )
 
    # ── Groq ─────────────────────────────────────────────────────────────────
    elif provider == "groq":
        from langchain_groq import ChatGroq
        if not settings.GROQ_API_KEY:
            raise EnvironmentError(
                "LLM_PROVIDER=groq but GROQ_API_KEY is not set in .env"
            )
        print(f"[LLM] Provider: Groq | Model: {settings.GROQ_MODEL}")
        return ChatGroq(
            model=settings.GROQ_MODEL,
            temperature=temp,
            api_key=settings.GROQ_API_KEY,
        )
 
    # ── Anthropic ────────────────────────────────────────────────────────────
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        if not settings.ANTHROPIC_API_KEY:
            raise EnvironmentError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set in .env"
            )
        print(f"[LLM] Provider: Anthropic | Model: {settings.ANTHROPIC_MODEL}")
        return ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            temperature=temp,
            api_key=settings.ANTHROPIC_API_KEY,
        )
 
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider}'. "
            f"Valid options: ollama, huggingface, openai, groq, anthropic"
        )
    
@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """
    Returns a cached LLM instance.
    The same object is reused across all nodes — no redundant initialisation.
 
    Usage:
        from backend.llm_factory import get_llm
        llm = get_llm()
        response = llm.invoke(messages)
    """
    return _build_llm()

def get_llm_with_tools(tools: list) -> BaseChatModel:
    """
    Returns the LLM bound to a list of LangChain tools.
    Called once per graph compilation — not cached since tools may vary.
 
    Usage:
        from backend.llm_factory import get_llm_with_tools
        llm_with_tools = get_llm_with_tools([market_research, tech_assessment])
        response = llm_with_tools.invoke(messages)
    """
    return get_llm().bind_tools(tools)

def get_parser() -> PydanticOutputParser:
    """
    Returns a cached Pydantic output parser for VCEvaluationOutput.
 
    Usage:
        from backend.llm_factory import get_parser
        parser = get_parser()
        parsed = parser.invoke(response)
    """
    return PydanticOutputParser(pydantic_object=VCEvaluationOutput)