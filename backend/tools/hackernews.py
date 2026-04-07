import requests

def analyze_developer_sentiment(query:str)->str:
    """
    Searches Hacker News via Algolia API for discussions regarding the startup idea/tech.
    Returns top thread titles to gauge developer sentiment and market interest.
    """

    url = f"https://hn.algolia.com/api/v1/search?query={query}&tags=story&hitsPerPage=5"

    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return f"ERROR: Hacker News API returned status code {response.status_code}."

        data = response.json()
        hits = data.get("hits", [])
        
        if not hits:
            return f"NO DATA: No discussions found on Hacker News for query '{query}'."

        report = f"### Hacker News Sentiment Analysis for '{query}'\n"
        
        for hit in hits:
            title = hit.get('title', 'No Title')
            points = hit.get('points', 0)
            comments = hit.get('num_comments', 0)
            date = hit.get('created_at', '')[:10] # Get YYYY-MM-DD
            
            report += f"- **{title}**\n"
            report += f"  - Score: {points} | Comments: {comments} | Date: {date}\n"
            
        return report

    except Exception as e:
        return f"ERROR: Hacker News API request failed. System Exception: {str(e)}"