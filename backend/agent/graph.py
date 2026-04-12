"""
backend/agent/graph.py

Execution mode is controlled by AGENT_MODE in .env (via settings):
  - parallel   → Send() fan-out, ~3x faster (default)
  - sequential → original ReAct loop
"""

import os
from langgraph.graph import StateGraph, END
from backend.agent.state import AgentState
from backend.config import settings

# LangSmith bootstrap — must happen before any LangChain import
os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"]   = settings.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_PROJECT"]    = settings.LANGCHAIN_PROJECT
if settings.LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY


def route_after_research(state: AgentState):
    """Conditional routing for the sequential ReAct loop."""
    messages     = state.get("messages", [])
    last_message = messages[-1]

    if state.get("revision_count", 0) >= settings.MAX_REVISIONS:
        print("[SYSTEM] Circuit breaker triggered. Routing to synthesis.")
        return "synthesis"

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"[SYSTEM] Agent called tools: {[t['name'] for t in last_message.tool_calls]}")
        return "tools"

    print("[SYSTEM] Agent ready for synthesis.")
    return "synthesis"


def _build_sequential_graph():
    from backend.agent.nodes import tool_node, research_node, synthesis_node

    workflow = StateGraph(AgentState)
    workflow.add_node("researcher", research_node)
    workflow.add_node("tools",      tool_node)
    workflow.add_node("synthesis",  synthesis_node)

    workflow.set_entry_point("researcher")
    workflow.add_conditional_edges(
        "researcher",
        route_after_research,
        {"tools": "tools", "synthesis": "synthesis"},
    )
    workflow.add_edge("tools",     "researcher")
    workflow.add_edge("synthesis", END)

    return workflow.compile()


def _build_parallel_graph():
    from backend.agent.parallel_nodes import compile_parallel_graph
    return compile_parallel_graph()


def compile_graph():
    """Returns the compiled graph for the configured AGENT_MODE."""
    mode = settings.AGENT_MODE.lower().strip()

    if mode == "parallel":
        print("[SYSTEM] Compiling PARALLEL graph (Send() fan-out).")
        return _build_parallel_graph()
    else:
        print("[SYSTEM] Compiling SEQUENTIAL graph (ReAct loop).")
        return _build_sequential_graph()