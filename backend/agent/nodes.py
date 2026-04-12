"""
backend/agent/nodes.py  —  Sequential ReAct nodes

LLM is sourced entirely from llm_factory. No provider-specific imports here.
To change model/provider: edit LLM_PROVIDER in .env. Done.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from backend.agent.state import AgentState
from backend.agent.prompts import SYNTHESIS_PROMPT, SYSTEM_PROMPT
from backend.tools.firecrawl import scrape_competitor_website
from backend.tools.github import assess_tech_stack
from backend.tools.hackernews import analyze_developer_sentiment

from backend.llm import get_groq_llm
from backend.schemas import VCEvaluationOutput
from backend.memory.rag_memory import (
    retrieve_similar,
    format_context_for_prompt,
    store_evaluation,
)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def market_research(url: str) -> str:
    """Scrapes a competitor website to extract pricing and features. Pass a valid, exact URL (e.g., https://stripe.com)."""
    return scrape_competitor_website(url=url)

@tool
def tech_assessment(query: str) -> str:
    """Searches GitHub to assess the viability of a tech stack. Pass a concise, 1-3 word query (e.g., 'langgraph multi-agent')."""
    return assess_tech_stack(query=query)

@tool
def sentiment_analysis(query: str) -> str:
    """Searches Hacker News to gauge developer sentiment. Pass a concise keyword (e.g., 'supabase')."""
    return analyze_developer_sentiment(query=query)

tools     = [market_research, tech_assessment, sentiment_analysis]
tool_node = ToolNode(tools=tools)

# ---------------------------------------------------------------------------
# Node logic
# ---------------------------------------------------------------------------

def research_node(state: AgentState) -> dict:
    """ReAct agent — decides which tools to call next."""
    messages = state.get("messages", [])

    if not messages:
        idea          = state["input_idea"]
        similar_evals = retrieve_similar(idea, n_results=3)
        rag_context   = format_context_for_prompt(similar_evals)

        if similar_evals:
            print(f"[RAG] Injecting {len(similar_evals)} similar past evaluations.")

        enriched_system = SYSTEM_PROMPT
        if rag_context:
            enriched_system = f"{SYSTEM_PROMPT}\n\n{rag_context}"

        messages = [
            SystemMessage(content=enriched_system),
            HumanMessage(content=f"STARTUP IDEA: {idea}"),
        ]

    llm_with_tools = get_groq_llm().bind_tools(tools)
    response       = llm_with_tools.invoke(messages)
    rev_count      = state.get("revision_count", 0) + 1


def synthesis_node(state: AgentState) -> dict:
    """Forces the JSON contract via native structured output, applies guards, stores to ChromaDB."""
    messages = state.get("messages", [])
    
    # No need for manual format instructions injected into the prompt
    synthesis_input = [SystemMessage(content=SYNTHESIS_PROMPT)] + messages

    # Bind structured output natively to Groq
    structured_llm = get_groq_llm().with_structured_output(VCEvaluationOutput)
    parsed_report  = structured_llm.invoke(synthesis_input)
    report_dict    = parsed_report.model_dump()

    # Python-level post-processing guard
    report_dict = _apply_consistency_rules(report_dict)

    try:
        store_evaluation(state.get("input_idea", ""), report_dict)
    except Exception as e:
        print(f"[RAG] Warning: failed to store evaluation — {e}")

    return {"final_report": report_dict}


def _apply_consistency_rules(report: dict) -> dict:
    """
    Enforces business logic that small LLMs often ignore from prompts alone.
    Called after parsing — operates on the plain dict before it hits the frontend.
    """
    status    = report.get("status", "NEEDS_WORK")
    score     = report.get("confidence_score", 50)
    market    = report.get("market_assessment", {})
    tech      = report.get("technical_feasibility", {})
    saturated = market.get("market_saturation_warning", False)
    buildable = tech.get("is_buildable", True)
    gh_repos  = tech.get("github_repos_found", 0)

    # Rule 1: saturated market must never be VALIDATE
    if saturated and status == "VALIDATE":
        print("[GUARD] Overriding VALIDATE → NEEDS_WORK: market is saturated.")
        report["status"] = "NEEDS_WORK"
        status = "NEEDS_WORK"

    # Rule 2: VALIDATE with is_buildable=False is a contradiction
    if status == "VALIDATE" and not buildable:
        print("[GUARD] Overriding VALIDATE → NEEDS_WORK: is_buildable is False.")
        report["status"] = "NEEDS_WORK"
        status = "NEEDS_WORK"

    # Rule 3: confidence score must be consistent with status
    if status == "REJECT" and score > 45:
        print(f"[GUARD] Clamping confidence {score} → 40 for REJECT decision.")
        report["confidence_score"] = 40

    if status == "VALIDATE" and gh_repos == 0 and score > 65:
        print(f"[GUARD] Clamping confidence {score} → 65: VALIDATE with 0 GitHub repos.")
        report["confidence_score"] = 65

    if status == "NEEDS_WORK" and (score < 30 or score > 75):
        clamped = max(30, min(75, score))
        print(f"[GUARD] Clamping confidence {score} → {clamped} for NEEDS_WORK.")
        report["confidence_score"] = clamped

    # Rule 4: average_stars must be an integer
    avg = tech.get("average_stars")
    if avg is not None:
        report["technical_feasibility"]["average_stars"] = round(avg)

    return report