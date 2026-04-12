"""
eval/benchmark.py

LLM-as-Judge evaluation pipeline for Venture-Graph.

Fixes applied:
  - initial_state now includes "rag_context": None (was causing KeyError in parallel graph)
  - build_judge_llm() now calls get_llm() from the factory — respects LLM_PROVIDER
  - duplicate check_expectations() function removed
  - run_graph() uses settings.AGENT_MODE, not hardcoded mode

Usage:
    python -m eval.benchmark                 # run all fixtures
    python -m eval.benchmark --idea "custom" # single custom idea
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixture ideas
# ---------------------------------------------------------------------------

FIXTURE_IDEAS = [
    {
        "id":               "blue_ocean_aerospace",
        "idea":             "A SaaS platform for real-time structural health monitoring of spacecraft components using embedded IoT sensors and ML anomaly detection.",
        "expected_status":  "VALIDATE",
        "expected_score_min": 70,
    },
    {
        "id":               "saturated_payments",
        "idea":             "An online payment processing platform for e-commerce businesses, competing directly with Stripe and PayPal.",
        "expected_status":  "REJECT",
        "expected_score_min": 0,
        "expected_score_max": 50,
    },
    {
        "id":               "niche_biotech",
        "idea":             "AI-driven drug interaction prediction tool specifically for rare paediatric oncology treatments, sold to hospital pharmacy teams.",
        "expected_status":  "VALIDATE",
        "expected_score_min": 65,
    },
    {
        "id":               "needs_work_crm",
        "idea":             "A CRM tool for small restaurants to manage reservations, loyalty points, and email marketing.",
        "expected_status":  "NEEDS_WORK",
        "expected_score_min": 30,
        "expected_score_max": 75,
    },
    {
        "id":               "legal_tech_niche",
        "idea":             "An automated contract review tool trained on maritime shipping law, targeting freight forwarding companies.",
        "expected_status":  "VALIDATE",
        "expected_score_min": 60,
    },
]

JUDGE_SYSTEM_PROMPT = """You are a rigorous AI evaluation judge assessing a VC analyst agent's output.

Score on FOUR dimensions, each 0-25 (total max 100):

1. REASONING_QUALITY (0-25): Is the final_verdict well-reasoned and grounded in
   specific data (competitor names, GitHub stats, HN sentiment)? Penalise vague statements.

2. SCHEMA_COMPLIANCE (0-25): Are all required fields present and correctly typed?
   Is confidence_score an integer 0-100? Penalise missing fields or type errors.

3. HALLUCINATION_RISK (0-25): Does the report invent specific numbers or company names
   not plausibly from real tools? 25 = no hallucinations. 0 = clearly fabricated.

4. VERDICT_ALIGNMENT (0-25): Does status logically follow from market_saturation_warning
   and confidence_score? VALIDATE + saturation=True is misaligned.

Return ONLY valid JSON, no markdown:
{
  "reasoning_quality":  <int 0-25>,
  "schema_compliance":  <int 0-25>,
  "hallucination_risk": <int 0-25>,
  "verdict_alignment":  <int 0-25>,
  "total_score":        <int 0-100>,
  "judge_notes":        "<1-2 sentences>"
}"""


# ---------------------------------------------------------------------------
# Graph runner
# ---------------------------------------------------------------------------

def run_graph(idea: str) -> dict | None:
    """Invoke the compiled graph and return the final_report dict."""
    from backend.agent.graph import compile_graph

    g = compile_graph()

    initial_state = {
        "input_idea":         idea,
        "messages":           [],
        "errors":             [],
        "revision_count":     0,
        "rag_context":        None,   # FIX: was missing, caused KeyError in parallel graph
        "market_data_raw":    None,
        "sentiment_data_raw": None,
        "tech_metrics_raw":   None,
        "final_report":       None,
    }

    for event in g.stream(initial_state):
        for node_name, state_data in event.items():
            if node_name == "synthesis" and state_data.get("final_report"):
                return state_data["final_report"]
    return None


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def build_judge_llm():
    """
    FIX: Now uses get_llm() from the factory instead of hardcoding HuggingFace.
    Respects LLM_PROVIDER from .env — switching provider automatically updates
    both the agent and the judge.
    """
    from backend.llm_factory import get_llm
    print(f"[EVAL] Building judge LLM via factory (provider: {__import__('backend.config', fromlist=['settings']).settings.LLM_PROVIDER})")
    return get_llm()


def judge_report(idea: str, report: dict, judge_llm) -> dict:
    """Ask the judge LLM to score a single report."""
    from langchain_core.messages import SystemMessage, HumanMessage

    user_content = (
        f"STARTUP IDEA: {idea}\n\n"
        f"AGENT REPORT:\n{json.dumps(report, indent=2)}"
    )
    response = judge_llm.invoke([
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])

    raw = response.content.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "reasoning_quality":  0,
            "schema_compliance":  0,
            "hallucination_risk": 0,
            "verdict_alignment":  0,
            "total_score":        0,
            "judge_notes":        f"PARSE ERROR: {raw[:200]}",
        }


# ---------------------------------------------------------------------------
# Expectation checker  (defined ONCE — duplicate removed)
# ---------------------------------------------------------------------------

def check_expectations(fixture: dict, report: dict) -> dict:
    """Compare actual output against expected status and score range."""
    actual_status = report.get("status", "UNKNOWN")
    actual_score  = report.get("confidence_score", -1)

    status_pass = actual_status == fixture.get("expected_status", actual_status)
    score_pass  = True

    if "expected_score_min" in fixture:
        score_pass = score_pass and actual_score >= fixture["expected_score_min"]
    if "expected_score_max" in fixture:
        score_pass = score_pass and actual_score <= fixture["expected_score_max"]

    return {
        "status_pass":   status_pass,
        "score_pass":    score_pass,
        "actual_status": actual_status,
        "actual_score":  actual_score,
    }


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(ideas: list[dict] | None = None):
    from tabulate import tabulate

    fixtures   = ideas or FIXTURE_IDEAS
    judge_llm  = build_judge_llm()
    results    = []
    table_rows = []

    print(f"\n{'='*70}")
    print(f"  Venture-Graph Benchmark  —  {len(fixtures)} fixtures  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}\n")

    for fx in fixtures:
        idea = fx["idea"]
        print(f"[{fx['id']}] Running graph...")
        t0      = time.time()
        report  = run_graph(idea)
        elapsed = round(time.time() - t0, 1)

        if not report:
            print(f"  ⚠  Graph returned no report. Skipping.\n")
            results.append({"id": fx["id"], "error": "no_report"})
            continue

        print(f"  ✓  Graph done in {elapsed}s. Running judge...")
        scores = judge_report(idea, report, judge_llm)
        expect = check_expectations(fx, report)

        row = {
            "id":              fx["id"],
            "idea_short":      idea[:55] + "...",
            "elapsed_s":       elapsed,
            "actual_status":   expect["actual_status"],
            "expected_status": fx.get("expected_status", "—"),
            "status_pass":     "✓" if expect["status_pass"] else "✗",
            "actual_score":    expect["actual_score"],
            "judge_total":     scores["total_score"],
            "reasoning":       scores["reasoning_quality"],
            "schema":          scores["schema_compliance"],
            "hallucination":   scores["hallucination_risk"],
            "alignment":       scores["verdict_alignment"],
            "judge_notes":     scores["judge_notes"],
            "full_report":     report,
            "judge_scores":    scores,
        }
        results.append(row)

        table_rows.append([
            fx["id"],
            expect["actual_status"],
            row["status_pass"],
            expect["actual_score"],
            scores["total_score"],
            f"{elapsed}s",
        ])

        print(f"  Judge total: {scores['total_score']}/100 | Notes: {scores['judge_notes'][:80]}\n")

    # Summary table
    print("\n" + tabulate(
        table_rows,
        headers=["Fixture", "Status", "Pass?", "Conf.Score", "Judge/100", "Time"],
        tablefmt="rounded_outline",
    ))

    scored = [r for r in results if "judge_total" in r]
    if scored:
        avg_judge = round(sum(r["judge_total"] for r in scored) / len(scored), 1)
        passes    = sum(1 for r in scored if r["status_pass"] == "✓")
        print(f"\n  Average judge score : {avg_judge}/100")
        print(f"  Status pass rate    : {passes}/{len(scored)}")

    # Persist
    out_dir  = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"benchmark_{ts}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved → {out_file}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Venture-Graph LLM-as-Judge benchmark")
    parser.add_argument("--idea", type=str, default=None,
                        help="Run a single custom idea instead of all fixtures")
    args = parser.parse_args()

    if args.idea:
        run_benchmark([{"id": "custom", "idea": args.idea}])
    else:
        run_benchmark()