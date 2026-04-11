import json
import uuid
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import StartupIdeaInput
from backend.agent.graph import compile_graph

app = FastAPI(title="Venture-Graph API", description="Multi-Agent Startup Evaluator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = compile_graph()

@app.post('/api/evaluate')
async def evaluate_startup(payload: StartupIdeaInput):
    """
    Streaming endpoint. Yields graph node transitions in real-time,
    followed by the final synthesised JSON payload.
 
    Each request gets a unique run_id that appears as the LangSmith trace ID,
    making it easy to link a specific user session to its full trace tree.
    """
    run_id   = str(uuid.uuid4())        
    run_name = f"evaluate | {payload.idea[:60]}"

    async def event_generator():
        initial_state = {
            "input_idea":       payload.idea,
            "messages":         [],
            "errors":           [],
            "revision_count":   0,
            "rag_context":      None,
            "market_data_raw":  None,
            "sentiment_data_raw": None,
            "tech_metrics_raw": None,
            "final_report":     None,
        }

        yield f"data: {json.dumps({'type': 'status', 'message': 'Initializing Agent Pipeline...', 'run_id': run_id})}\n\n"

        langsmith_config = {
            "run_name": run_name,
            "run_id":   run_id,
            "tags":     ["venture-graph", "production"],
            "metadata": {
                "idea":    payload.idea,
                "run_id":  run_id,
            },
        }
        
        for event in graph.stream(initial_state, config=langsmith_config):
            for node_name, state_data in event.items():

                yield f"data: {json.dumps({'type': 'status', 'message': f'Executing Node: {node_name.upper()}'})}\n\n"

                if node_name == "synthesis" and state_data.get("final_report"):
                    yield f"data: {json.dumps({'type': 'result', 'data': state_data['final_report'], 'run_id': run_id})}\n\n"

    return StreamingResponse(event_generator(), media_type='text/event-stream')

@app.get('/health')
def health_check():
    """Simple probe to verify the server is alive."""
    return {"status": "Venture-Graph Backend Online"}