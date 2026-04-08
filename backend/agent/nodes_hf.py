from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.prebuilt import ToolNode

from backend.agent.state import AgentState
from backend.agent.prompts import SYNTHESIS_PROMPT, SYSTEM_PROMPT
from backend.schemas import VCEvaluationOutput
from backend.tools.firecrawl import scrape_competitor_website
from backend.tools.github import assess_tech_stack
from backend.tools.hackernews import analyze_developer_sentiment
from backend.config import settings

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

# 2. Initialize the Local Inference Engine
_llm = HuggingFaceEndpoint(
    model=settings.HUGGINGFACE_MODEL, 
    temperature=settings.LLM_TEMPERATURE, 
    huggingfacehub_api_token=settings.HUGGINGFACE_API
)
llm = ChatHuggingFace(llm=_llm)
llm_with_tools = llm.bind_tools(tools)

parser = PydanticOutputParser(pydantic_object=VCEvaluationOutput)

# 3. Define the Node Logic
def research_node(state: AgentState):
    """The ReAct agent that decides which tools to call."""
    messages = state.get("messages", [])
    if not messages:
        # Inject the system prompt and user idea on the first run
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"STARTUP IDEA: {state['input_idea']}")
        ]

    response = llm_with_tools.invoke(messages)

    rev_count = state.get('revision_count', 0) + 1

    return {"messages": [response], "revision_count": rev_count}

def synthesis_node(state: AgentState):
    """The final node that forces the JSON contract."""
    messages = state.get("messages", [])
    
    # Get the JSON schema instructions required by the parser
    format_instructions = parser.get_format_instructions()
    
    # Inject those instructions into the synthesis prompt
    system_content = f"{SYNTHESIS_PROMPT}\n\nStrict Formatting Instructions:\n{format_instructions}"
    synthesis_input = [SystemMessage(content=system_content)] + messages

    # Get the raw response from the standard LLM
    response = llm.invoke(synthesis_input)

    # Parse the raw text back into the VCEvaluationOutput Pydantic object
    parsed_report = parser.invoke(response)

    return {"final_report": parsed_report.model_dump()}