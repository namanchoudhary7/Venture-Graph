import os, requests
from dotenv import load_dotenv

load_dotenv()

def assess_tech_stack(query:str)->str:
    """
    Searches GitHub for repositories matching the tech stack query.
    Returns aggregated metrics (stars, forks) to assess technical viability.
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token or token == "your_github_classic_pat_here":
        return "ERROR: GITHUB_TOKEN is missing or invalid in .env file."   
    
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    # We search for the top 3 most relevant repositories
    url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc&per_page=3"

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 403:
            return "ERROR: GitHub API rate limit exceeded or token lacks permissions."
        elif response.status_code != 200:
            return f"ERROR: GitHub API returned status code {response.status_code}. Response: {response.text}"
        
        data = response.json()
        items = data.get('items', [])

        if not items:
            return f"NO DATA: No GitHub repositories found for query '{query}'. Technology may be too niche or proprietary."
        
        report = f"### GitHub Technical Assessment for '{query}'\n"
        report += f"Total repositories found globally: {data.get('total_count', 0)}\n\n"

        for repo in items:
            report += f"- **{repo['name']}** ({repo['full_name']})\n"
            report += f"  - Stars: {repo['stargazers_count']} | Forks: {repo['forks_count']} | Open Issues: {repo['open_issues_count']}\n"
            report += f"  - Description: {repo['description']}\n"
            report += f"  - Last Updated: {repo['updated_at']}\n\n"
            
        return report

    except Exception as e:
        return f"ERROR: GitHub API request failed. System Exception: {str(e)}"