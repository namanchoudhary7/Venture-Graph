from langgraph.graph import StateGraph, END
from backend.agent.state import AgentState
from backend.agent.nodes_hf import tool_node, research_node, synthesis_node
from backend.config import settings

def route_after_research(state: AgentState):
    """Conditional routing logic."""
    messages = state.get("messages", [])
    last_message = messages[-1]

    # 1. Circuit Breaker: Prevent infinite loops if tools keep failing
    if state.get('revision_count', 0) >= settings.MAX_REVISIONS:
        print(f"[SYSTEM] Circuit breaker triggered ({settings.MAX_REVISIONS} revisions). Routing to synthesis.")
        return "synthesis"
    
    # 2. Route to Tools: If the LLM generated a tool call
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print(f"[SYSTEM] Agent called tools: {[t['name'] for t in last_message.tool_calls]}")
        return "tools"
    
    # 3. Route to Synthesis: If the LLM determines it is done
    content = str(last_message.content)
    if "READY_FOR_SYNTHESIS" in content or not last_message.tool_calls:
        print("[SYSTEM] Agent is ready for synthesis.")
        return "synthesis"
    
    return "synthesis"

def compile_graph():
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(AgentState)
    workflow.add_node("researcher", research_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("synthesis", synthesis_node)

    workflow.set_entry_point("researcher")

    workflow.add_conditional_edges(
        "researcher",
        route_after_research,
        {
            "tools": "tools",
            "synthesis": "synthesis"
        }
    )

    workflow.add_edge("tools", "researcher")
    workflow.add_edge("synthesis", END)

    return workflow.compile()