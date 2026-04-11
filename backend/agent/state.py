from typing import TypedDict, Annotated, Any
import operator

def append_list(a:list, b:list)->list:
    """Reducer function to append to a list in LangGraph state."""
    if a is None: return b
    return a+b

class AgentState(TypedDict):
    input_idea: str

    messages: Annotated[list[Any], append_list]
    errors: Annotated[list[str], append_list]   
    revision_count: int                         

    rag_context: str | None
    
    market_data_raw: dict[str, Any] | None
    sentiment_data_raw: str | None
    tech_metrics_raw: dict[str, Any] | None

    final_report: dict[str, Any] | None       