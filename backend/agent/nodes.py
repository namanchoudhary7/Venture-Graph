from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama

from backend.agent.state import AgentState
from backend.agent.prompts import SYNTHESIS_PROMPT, SYSTEM_PROMPT
from backend.schemas import VCEvaluationOutput
from backend.tools.firecrawl import scrape_competitor_website
from backend.tools.github import assess_tech_stack
from backend.tools.hackernews import analyze_developer_sentiment
from backend.config import settings
from backend.llm_factory import get_llm, get_llm_with_tools, get_parser

from backend.memory.rag_memory import (
    retrieve_similar,
    format_context_for_prompt,
    store_evaluation,
)

# 1. Wrap the raw tools into LangChain structured tools
@tool
def market_research(url: str)->str:
    """Scrapes a competitor website to extract pricing and features. Pass a valid, exact URL (e.g., https://stripe.com)."""
    return scrape_competitor_website(url=url)

@tool 
def tech_assessment(query:str)->str:
    """Searches GitHub to assess the viability of a tech stack. Pass a concise, 1-3 word query (e.g., 'langgraph multi-agent')."""
    return assess_tech_stack(query=query)

@tool
def sentiment_analysis(query:str)->str:
    """Searches Hacker News to gauge developer sentiment. Pass a concise keyword (e.g., 'supabase')."""
    return analyze_developer_sentiment(query=query)

tools = [market_research, tech_assessment, sentiment_analysis]
tool_node = ToolNode(tools=tools)

# 3. Define the Node Logic
def research_node(state: AgentState):
    """
    ReAct agent that decides which tools to call.
 
    On the very first turn (no messages yet) it:
      1. Retrieves up to 3 semantically similar past evaluations from ChromaDB.
      2. Prepends them to SYSTEM_PROMPT so the agent can reference patterns.
    """
    messages = state.get("messages", [])
    if not messages:
        idea = state["input_idea"]

        similar_evals  = retrieve_similar(idea, n_results=3)
        rag_context    = format_context_for_prompt(similar_evals)

        if similar_evals:
            print(f"[RAG] Injecting {len(similar_evals)} similar past evaluations into prompt.")
        
        enriched_system = SYSTEM_PROMPT
        if rag_context:
            enriched_system = f"{SYSTEM_PROMPT}\n\n{rag_context}"

        messages = [
            SystemMessage(content=enriched_system),
            HumanMessage(content=f"STARTUP IDEA: {idea}")
        ]
    llm_with_tools = get_llm_with_tools(tools)
    response = llm_with_tools.invoke(messages)
    rev_count = state.get('revision_count', 0) + 1

    return {"messages": [response], "revision_count": rev_count}

def synthesis_node(state: AgentState):
    """
    Final node that forces the JSON contract, then stores the result in ChromaDB.
    """
    messages = state.get("messages", [])
    parser              = get_parser()
    format_instructions = parser.get_format_instructions()
    system_content = f"""{SYNTHESIS_PROMPT}

Strict Formatting Instructions:
{format_instructions}

CRITICAL: You must output ONLY valid JSON. Do NOT include any conversational text, preambles, postambles, or markdown formatting (do not use ```json blocks). Your output must start exactly with {{ and end exactly with }}.
"""
    synthesis_input = [SystemMessage(content=system_content)] + messages
    llm           = get_llm()
    response      = llm.invoke(synthesis_input)
    parsed_report = parser.invoke(response)
    report_dict   = parsed_report.model_dump()

    idea = state.get("input_idea", "")
    try:
        store_evaluation(idea, report_dict)
    except Exception as e:
        # Never let storage failure break the response
        print(f"[RAG] Warning: failed to store evaluation — {e}")
 
    return {"final_report": report_dict}