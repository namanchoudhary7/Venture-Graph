import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.tools.firecrawl import scrape_competitor_website
from backend.tools.github import assess_tech_stack
from backend.tools.hackernews import analyze_developer_sentiment

def run_tests():
    print("====================================")
    print("INITIATING PHASE 2 TOOL VERIFICATION")
    print("====================================\n")

    # 1. Firecrawl Test
    print("[1/3] Testing Firecrawl (Market Research)...")
    fc_result = scrape_competitor_website("https://stripe.com/pricing")
    if fc_result.startswith("ERROR"):
        print(f"  [X] FAILED: {fc_result}")
    else:
        print(f"  [+] PASSED. Extracted {len(fc_result)} characters of Markdown.")

    # 2. GitHub Test
    print("\n[2/3] Testing GitHub API (Tech Assessment)...")
    gh_result = assess_tech_stack("langgraph multi-agent")
    if gh_result.startswith("ERROR"):
        print(f"  [X] FAILED: {gh_result}")
    else:
        print("  [+] PASSED. Output preview:")
        print("      " + gh_result.replace("\n", "\n      ")[:300] + "...")

    # 3. Hacker News Test
    print("\n[3/3] Testing Hacker News API (Sentiment Analysis)...")
    hn_result = analyze_developer_sentiment("langgraph")
    if hn_result.startswith("ERROR"):
        print(f"  [X] FAILED: {hn_result}")
    else:
        print("  [+] PASSED. Output preview:")
        print("      " + hn_result.replace("\n", "\n      ")[:300] + "...")

    print("\n====================================")

if __name__ == "__main__":
    run_tests()