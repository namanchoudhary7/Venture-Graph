import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent.graph import compile_graph

def run_test():

    print("====================================")
    print("INITIATING PHASE 3 GRAPH COMPILATION")
    print("====================================\n")

    graph = compile_graph()

    # Initialize state
    initial_state = {
        "input_idea": "An open-source, multi-agent orchestrator built in Rust that competes with LangGraph.",
        "messages": [],
        "errors": [],
        "revision_count": 0,
        "market_data_raw": None,
        "sentiment_data_raw": None,
        "tech_metrics_raw": None,
        "final_report": None
    }

    print("Invoking Graph (This may take 30-60 seconds locally depending on your GPU)...")

    # Stream the graph execution to watch the agent think
    for event in graph.stream(initial_state):
        for key, value in event.items():
            print(f"\n--- Node Completed: {key} ---")
            if key == "synthesis":
                print("\nFINAL VALIDATED JSON OUTPUT:")
                print(json.dumps(value.get("final_report"), indent=2))

if __name__ == "__main__":
    run_test()