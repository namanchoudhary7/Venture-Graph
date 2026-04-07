from typing import TypedDict, Annotated, Any
import operator

def append_list(a:list, b:list)->list:
    """Reducer function to append to a list in LangGraph state."""
    if a is None: return b
    return a+b

class AgentState(TypedDict):
    # Core Input
    input_idea: str

    # Internal Reasoning & History (Crucial for ReAct)
    messages: Annotated[list[Any], append_list] # Stores Langchain/LLM message history
    errors: Annotated[list[str], append_list]   # Tracks tool failures for self-correction
    revision_count: int                         # Circuit breaker for infinite loops

    # Unstructured Data Dump (Populated by Tools)
    market_data_raw: dict[str, Any] | None
    sentiment_data_raw: str | None
    tech_metrics_raw: dict[str, Any] | None

    # Final Structured Output (Populated by Synthesizer Node)
    final_report: dict[str, Any] | None       # Must map to VCEvaluationOutput schema