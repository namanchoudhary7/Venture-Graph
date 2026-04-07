import os 
from firecrawl import Firecrawl
from dotenv import load_dotenv

# Load environment variables explicitly for isolated testing
load_dotenv()

def scrape_competitor_website(url:str)->str:
    """
    Scrapes a target URL using Firecrawl to bypass anti-bot protections.
    Returns the page content as clean, LLM-readable Markdown.
    """
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key or api_key == "your_copied_key":
        return "ERROR: FIRECRAWL_API_KEY is missing or invalid in .env file."

    try:
        app = Firecrawl(api_key=api_key)

        # We request markdown specifically to strip out DOM noise
        result = app.scrape(url=url, formats=['markdown'])

        if hasattr(result, 'markdown') and result.markdown:
            markdown_content = result.markdown
            # We truncate to 5000 characters to prevent blowing out the LLM context window
            if len(markdown_content) > 5000:
                return markdown_content[:5000] + "\n\n...[CONTENT TRUNCATED FOR CONTEXT LIMIT]..."
            return markdown_content
        else:
            return f"ERROR: Firecrawl returned successful status but no markdown payload. Raw: {str(result)[:200]}"
        
    except Exception as e:
        # We return the error as a string so the LangGraph agent can read it and self-correct
        return f"ERROR: Scraping failed for {url}. System Exception: {str(e)}"