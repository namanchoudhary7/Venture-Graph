"""
backend/llm.py

Exclusive Groq LLM initialization for Venture-Graph.

Every node, worker, and eval script imports from here:
    from backend.llm import get_groq_llm, get_parser
"""

from functools import lru_cache
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq

from backend.config import settings
from backend.schemas import VCEvaluationOutput

@lru_cache(maxsize=1)
def get_groq_llm() -> BaseChatModel:
    """
    Returns a cached ChatGroq instance.
    The same object is reused across all nodes — no redundant initialization.
    """
    if not settings.GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY is not set in .env")
        
    print(f"[LLM] Provider: Groq | Model: {settings.GROQ_MODEL}")
    return ChatGroq(
        model=settings.GROQ_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        api_key=settings.GROQ_API_KEY,
    )

def get_parser() -> PydanticOutputParser:
    """
    Returns a cached Pydantic output parser for VCEvaluationOutput.
    
    Usage:
        from backend.llm import get_parser
        parser = get_parser()
        parsed = parser.invoke(response)
    """
    return PydanticOutputParser(pydantic_object=VCEvaluationOutput)