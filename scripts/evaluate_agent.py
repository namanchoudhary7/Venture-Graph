import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent.graph import compile_graph

def run_evaluations():
    print("====================================")
    print("INITIATING PHASE 6: PIPELINE EVALUATION")
    print("====================================\n")

    graph = compile_graph()

    # Define our unit tests
    test_cases = [
        {
            "name": "TEST 1: The Saturated Market (Expected: Fail/Needs Work)",
            "idea": "A new generic payment gateway that does exactly what Stripe does, but built in Python.",
            "expected_status": ["NEEDS_WORK", "REJECT"],
            "max_score": 75 # The agent should recognize the saturation and dock points
        },
        {
            "name": "TEST 2: The Niche Blue Ocean (Expected: Validate)",
            "idea": "An open-source AI agent orchestrator built in Rust specifically for aerospace embedded systems.",
            "expected_status": ["VALIDATE"],
            "min_score": 75 # The agent should recognize the strong tech stack and niche viability
        }
    ]

    passed = 0

    for test in test_cases:
        print(f"Running {test['name']}...")
        
        initial_state = {
            "input_idea": test["idea"],
            "messages": [],
            "errors": [],
            "revision_count": 0,
            "final_report": None
        }
        
        # Invoke graph silently (no streaming)
        final_state = graph.invoke(initial_state)
        report = final_state.get("final_report", {})
        
        status = report.get("status", "UNKNOWN")
        score = report.get("confidence_score", 0)
        
        print(f"  -> Agent Decision: {status}")
        print(f"  -> Confidence Score: {score}")
        
        # Automated Assertions
        status_match = status in test.get("expected_status", [])
        
        score_match = True
        if "max_score" in test and score > test["max_score"]:
            score_match = False
        if "min_score" in test and score < test["min_score"]:
            score_match = False
            
        if status_match and score_match:
            print("  [+] TEST PASSED\n")
            passed += 1
        else:
            print("  [X] TEST FAILED (Assertion Mismatch)\n")

    print("====================================")
    print(f"EVALUATION COMPLETE: {passed}/{len(test_cases)} PASSED.")
    
    if passed == len(test_cases):
        print("SYSTEM IS PRODUCTION READY. CI/CD GATES OPEN.")
    else:
        print("AGENT LOGIC REQUIRES TUNING. CHECK PROMPTS.")

if __name__ == "__main__":
    run_evaluations()