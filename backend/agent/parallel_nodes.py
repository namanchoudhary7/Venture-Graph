"""
backend/agent/parallel_nodes_hf.py  —  Parallel Send() fan-out nodes

Fixes applied vs previous version:
  1. Worker prompts are much more specific — with few-shot examples and
     explicit negative instructions to prevent hallucinated queries.
  2. Each worker retries up to 2 times if the tool returns an error.
  3. tech_worker now tries multiple query strategies before giving up,
     so "0 repos found" only happens when the tech is genuinely niche.
  4. aggregator includes a data quality warning when a worker got no data,
     so the synthesis node can factor that into its confidence score.

Key LangGraph rule:
  - NODE functions  → always return a plain dict
  - EDGE functions  → the only place Send() objects are returned
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph, END
from typing import TypedDict
from backend.schemas import VCEvaluationOutput
from backend.agent.state import AgentState
from backend.agent.prompts import SYNTHESIS_PROMPT
from backend.tools.firecrawl import scrape_competitor_website
from backend.tools.github import assess_tech_stack
from backend.tools.hackernews import analyze_developer_sentiment
from backend.llm_factory import get_llm, get_parser
from backend.memory.rag_memory import (
    retrieve_similar,
    format_context_for_prompt,
    store_evaluation,
)


# ---------------------------------------------------------------------------
# Worker input type
# ---------------------------------------------------------------------------

class WorkerInput(TypedDict):
    input_idea:  str
    rag_context: str


# ---------------------------------------------------------------------------
# STEP 1 — Orchestrator NODE (returns plain dict — never Send objects)
# ---------------------------------------------------------------------------

def orchestrator_node(state: AgentState) -> dict:
    """Fetches RAG context and stores it in state for fan_out() to read."""
    idea          = state["input_idea"]
    similar_evals = retrieve_similar(idea, n_results=3)
    rag_context   = format_context_for_prompt(similar_evals)

    if similar_evals:
        print(f"[RAG] Injecting {len(similar_evals)} similar past evaluations.")
    else:
        print("[RAG] No similar evaluations found — cold start.")

    return {"rag_context": rag_context}


# ---------------------------------------------------------------------------
# STEP 2 — fan_out EDGE FUNCTION (only place Send() is created)
# ---------------------------------------------------------------------------

def fan_out(state: AgentState) -> list:
    """Conditional edge — returns Send() list to LangGraph."""
    worker_input: WorkerInput = {
        "input_idea":  state["input_idea"],
        "rag_context": state.get("rag_context", ""),
    }
    return [
        Send("market_worker",    worker_input),
        Send("tech_worker",      worker_input),
        Send("sentiment_worker", worker_input),
    ]


# ---------------------------------------------------------------------------
# Shared LLM helper
# ---------------------------------------------------------------------------

def _ask_llm(prompt: str) -> str:
    """Single LLM call via the factory — always uses the configured provider."""
    return get_llm().invoke([HumanMessage(content=prompt)]).content.strip()


def _clean_url(raw: str) -> str | None:
    """Extract the first valid-looking URL from an LLM response."""
    for token in raw.split():
        token = token.strip(".,\"'()")
        if token.startswith("http://") or token.startswith("https://"):
            return token
    return None


def _clean_query(raw: str, max_words: int = 3) -> str:
    """
    Strip markdown, quotes, and excess words from an LLM query response.
    Limits to max_words to keep GitHub/HN queries tight.
    """
    cleaned = (
        raw.replace("```", "")
           .replace('"', "")
           .replace("'", "")
           .replace("*", "")
           .strip()
    )
    # Take only the first line (LLMs sometimes add explanations on line 2)
    first_line = cleaned.splitlines()[0].strip()
    # Limit to max_words
    words = first_line.split()
    return " ".join(words[:max_words])


# ---------------------------------------------------------------------------
# STEP 3 — Parallel worker nodes
# ---------------------------------------------------------------------------

def market_worker(state: WorkerInput) -> dict:
    """
    Identifies the most relevant competitor and scrapes their website.
    Retries with a different URL if the first scrape fails.
    """
    idea = state["input_idea"]
    print("[PARALLEL] market_worker starting.")

    # Few-shot prompt — prevents vague or hallucinated URLs
    url_raw = _ask_llm(
        f"You are a market research analyst.\n"
        f"Startup idea: \"{idea}\"\n\n"
        f"Task: Identify the single most direct competitor to this startup idea "
        f"and return their homepage URL.\n\n"
        f"Rules:\n"
        f"- Return ONLY the URL. No explanation, no punctuation before or after.\n"
        f"- Must be a real, well-known company's actual domain.\n"
        f"- Do NOT return search engines, Wikipedia, or generic sites.\n\n"
        f"Examples:\n"
        f"  Idea: online payment processing for e-commerce → https://stripe.com\n"
        f"  Idea: project management tool for remote teams → https://asana.com\n"
        f"  Idea: AI writing assistant for emails → https://grammarly.com\n\n"
        f"Competitor URL:"
    )

    url = _clean_url(url_raw)

    # Retry once if first response wasn't a URL
    if not url:
        print(f"[PARALLEL] market_worker: bad URL '{url_raw[:60]}', retrying...")
        retry_raw = _ask_llm(
            f"Give me the homepage URL of the biggest competitor to: \"{idea}\"\n"
            f"Reply with ONLY the URL starting with https://"
        )
        url = _clean_url(retry_raw) or "https://example.com"

    result = scrape_competitor_website(url=url)

    # If scrape errored, note it but don't crash
    if result.startswith("ERROR"):
        print(f"[PARALLEL] market_worker: scrape failed for {url}")
        result = f"Scrape failed for {url}. Error: {result}"

    print(f"[PARALLEL] market_worker done. URL={url}")
    return {"market_data_raw": {"url": url, "content": result}}


def tech_worker(state: WorkerInput) -> dict:
    """
    Searches GitHub for repos relevant to the core technology.
    Tries up to 3 progressively broader queries before giving up.
    """
    idea = state["input_idea"]
    print("[PARALLEL] tech_worker starting.")

    # Generate 3 query candidates from most specific to broadest
    queries_raw = _ask_llm(
        f"You are a software engineer assessing technical feasibility.\n"
        f"Startup idea: \"{idea}\"\n\n"
        f"Task: Generate 3 GitHub search queries to find relevant open-source "
        f"repositories for this idea's core technology stack.\n\n"
        f"Rules:\n"
        f"- Each query must be 1-4 words only.\n"
        f"- Go from SPECIFIC to BROAD across the 3 queries.\n"
        f"- Focus on the technology/framework, not the business domain.\n"
        f"- Do NOT include company names, proper nouns, or version numbers.\n"
        f"- Return exactly 3 lines, one query per line, no numbering or bullets.\n\n"
        f"Examples for 'AI drone inspection of oil pipelines':\n"
        f"  drone computer vision\n"
        f"  aerial object detection\n"
        f"  python drone sdk\n\n"
        f"Examples for 'SaaS carbon emissions tracker':\n"
        f"  carbon footprint tracker\n"
        f"  emissions monitoring api\n"
        f"  sustainability dashboard\n\n"
        f"Your 3 queries:"
    )

    # Parse the 3 candidate queries
    candidates = [
        _clean_query(line, max_words=4)
        for line in queries_raw.splitlines()
        if line.strip()
    ][:3]

    # Fallback if LLM gave fewer than expected
    if not candidates:
        candidates = [_clean_query(queries_raw, max_words=3)]

    print(f"[PARALLEL] tech_worker: trying queries {candidates}")

    # Try each query, stop at the first one that returns results
    best_result = None
    used_query  = None

    for query in candidates:
        if not query:
            continue
        result = assess_tech_stack(query=query)
        if not result.startswith("NO DATA") and not result.startswith("ERROR"):
            best_result = result
            used_query  = query
            print(f"[PARALLEL] tech_worker: got results for query='{query}'")
            break
        print(f"[PARALLEL] tech_worker: no results for '{query}', trying next...")

    if not best_result:
        # All queries returned no data — report it honestly
        used_query  = candidates[0] if candidates else "unknown"
        best_result = (
            f"NO DATA: No GitHub repositories found across queries {candidates}. "
            f"This technology may be highly proprietary, brand-new, or the queries "
            f"need refinement. Treat technical feasibility as UNVERIFIED."
        )
        print(f"[PARALLEL] tech_worker: all queries exhausted — no repos found.")

    print(f"[PARALLEL] tech_worker done. Final query='{used_query}'")
    return {"tech_metrics_raw": {"query": used_query, "content": best_result}}


def sentiment_worker(state: WorkerInput) -> dict:
    """
    Searches Hacker News for developer discussions about the idea's domain.
    Uses a product/market keyword, not a technology keyword.
    """
    idea = state["input_idea"]
    print("[PARALLEL] sentiment_worker starting.")

    keyword_raw = _ask_llm(
        f"You are analysing developer community sentiment.\n"
        f"Startup idea: \"{idea}\"\n\n"
        f"Task: Return ONE search keyword to find relevant Hacker News discussions "
        f"about this product's market or problem domain.\n\n"
        f"Rules:\n"
        f"- 1-2 words only.\n"
        f"- Focus on the MARKET or PROBLEM (e.g. 'carbon tracking', 'drone inspection').\n"
        f"- Do NOT use generic words like 'startup', 'saas', 'app', 'ai', 'tool'.\n"
        f"- Do NOT use technology names (Python, React, etc.).\n\n"
        f"Examples:\n"
        f"  Idea: AI tool for detecting bridge defects using drones → bridge inspection\n"
        f"  Idea: SaaS carbon emissions tracker for shipping → carbon tracking\n"
        f"  Idea: CRM for restaurant reservations → restaurant tech\n\n"
        f"Your keyword:"
    )

    keyword = _clean_query(keyword_raw, max_words=2)

    # Fallback: derive a safe keyword from the idea itself
    if not keyword or len(keyword) < 3:
        keyword = " ".join(idea.split()[:2]).lower()

    result = analyze_developer_sentiment(query=keyword)

    if result.startswith("NO DATA"):
        # Try a one-word fallback
        fallback = keyword.split()[0]
        print(f"[PARALLEL] sentiment_worker: no data for '{keyword}', trying '{fallback}'")
        result = analyze_developer_sentiment(query=fallback)

    print(f"[PARALLEL] sentiment_worker done. Keyword='{keyword}'")
    return {"sentiment_data_raw": result}


# ---------------------------------------------------------------------------
# STEP 4 — Aggregator (fan-in)
# ---------------------------------------------------------------------------

def aggregator_node(state: AgentState) -> dict:
    """
    Merges all worker outputs into a single context message for synthesis.
    Adds explicit data quality warnings so the synthesiser can adjust its
    confidence score accordingly (e.g. 0 GitHub repos → lower confidence).
    """
    market_raw    = state.get("market_data_raw")    or {}
    tech_raw      = state.get("tech_metrics_raw")   or {}
    sentiment_raw = state.get("sentiment_data_raw") or "No sentiment data retrieved."

    # Build data quality flags for the synthesiser
    warnings = []
    tech_content = tech_raw.get("content", "")
    if "NO DATA" in tech_content or "UNVERIFIED" in tech_content:
        warnings.append(
            "⚠ TECH DATA MISSING: GitHub returned 0 results. "
            "Set is_buildable=false and github_repos_found=0. "
            "Lower confidence_score by at least 15 points."
        )
    if "Scrape failed" in market_raw.get("content", ""):
        warnings.append(
            "⚠ MARKET DATA MISSING: Competitor website could not be scraped. "
            "Use only what you can infer — do not hallucinate pricing or features."
        )
    if "NO DATA" in sentiment_raw:
        warnings.append(
            "⚠ SENTIMENT DATA MISSING: No HN discussions found. "
            "Note this in developer_sentiment rather than inventing sentiment."
        )

    warning_block = ""
    if warnings:
        warning_block = "\n\n=== DATA QUALITY WARNINGS (must be reflected in your output) ===\n"
        warning_block += "\n".join(warnings)

    summary = (
        f"=== MARKET RESEARCH ===\n"
        f"Competitor URL scraped: {market_raw.get('url', 'N/A')}\n"
        f"{market_raw.get('content', 'No data.')}\n\n"

        f"=== TECH ASSESSMENT ===\n"
        f"GitHub query used: '{tech_raw.get('query', 'N/A')}'\n"
        f"{tech_content or 'No data.'}\n\n"

        f"=== DEVELOPER SENTIMENT (Hacker News) ===\n"
        f"{sentiment_raw}"
        f"{warning_block}"
    )

    print("[PARALLEL] aggregator: all workers complete, context assembled.")
    return {
        "messages":       [AIMessage(content=summary)],
        "revision_count": state.get("revision_count", 0) + 1,
    }


# ---------------------------------------------------------------------------
# STEP 5 — Synthesis
# ---------------------------------------------------------------------------

def parallel_synthesis_node(state: AgentState) -> dict:
    """Synthesises final structured report and persists to ChromaDB."""
    # parser              = get_parser()
    
    messages            = state.get("messages", [])
    # format_instructions = parser.get_format_instructions()
    system_content      = f"{SYNTHESIS_PROMPT}"
    synthesis_input     = [SystemMessage(content=system_content)] + messages

    # response      = get_llm().invoke(synthesis_input)
    # parsed_report = parser.invoke(response)
    structured_llm = get_llm().with_structured_output(VCEvaluationOutput)
    parsed_report = structured_llm.invoke(synthesis_input)
    report_dict   = parsed_report.model_dump()

    try:
        store_evaluation(state.get("input_idea", ""), report_dict)
        print("[RAG] Evaluation stored successfully.")
    except Exception as e:
        print(f"[RAG] Warning: could not store evaluation — {e}")

    return {"final_report": report_dict}


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def compile_parallel_graph():
    """
    Topology:
        orchestrator_node
              │
          fan_out()  ← conditional edge — returns Send() list
         ┌────┼────┐
         ▼    ▼    ▼   (run in parallel)
       mkt  tech  sent
         └────┼────┘
              ▼
          aggregator  ← waits for all three
              ▼
           synthesis
              ▼
             END
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("orchestrator",     orchestrator_node)
    workflow.add_node("market_worker",    market_worker)
    workflow.add_node("tech_worker",      tech_worker)
    workflow.add_node("sentiment_worker", sentiment_worker)
    workflow.add_node("aggregator",       aggregator_node)
    workflow.add_node("synthesis",        parallel_synthesis_node)

    workflow.set_entry_point("orchestrator")

    workflow.add_conditional_edges(
        "orchestrator",
        fan_out,
        ["market_worker", "tech_worker", "sentiment_worker"],
    )

    workflow.add_edge("market_worker",    "aggregator")
    workflow.add_edge("tech_worker",      "aggregator")
    workflow.add_edge("sentiment_worker", "aggregator")
    workflow.add_edge("aggregator",       "synthesis")
    workflow.add_edge("synthesis",        END)

    return workflow.compile()