import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import StartupIdeaInput
from backend.agent.graph import compile_graph

app = FastAPI(title="Venture-Graph API", description="Multi-Agent Startup Evaluator")

# Critical: Allow our upcoming Streamlit frontend (usually localhost:8501) to talk to this backend

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compile graph once at startup to save memory and latency
graph = compile_graph()

@app.post('/api/evaluate')
async def evaluate_startup(payload: StartupIdeaInput):
    """
    Streaming endpoint. Yields graph node transitions in real-time, 
    followed by the final synthesized JSON payload.
    """

    async def event_generator():
        initial_state = {
            "input_idea" : payload.idea,
            "messages": [],
            "errors": [],
            "revision_count": 0,
            "market_data_raw": None,
            "sentiment_data_raw": None,
            "tech_metrics_raw": None,
            "final_report": None
        }

        yield f"data: {json.dumps({'type': 'status', 'message': 'Initializing Agent Pipeline...'})}\n\n"

        # Stream the graph execution
        for event in graph.stream(initial_state):
            for node_name, state_data in event.items():

                # Yield the current node to the frontend so it can display "Agent is thinking..."
                yield f"data: {json.dumps({'type': 'status', 'message': f'Executing Node: {node_name.upper()}'})}\n\n"

                # If we hit the synthesis node, extract and yield the final Pydantic JSON
                if node_name == "synthesis" and state_data.get("final_report"):
                    yield f"data: {json.dumps({'type': 'result', 'data': state_data['final_report']})}\n\n"

    return StreamingResponse(event_generator(), media_type='text/event-stream')

@app.get('/health')
def health_check():
    """Simple probe to verify the server is alive."""
    return {"status": "Venture-Graph Backend Online"}