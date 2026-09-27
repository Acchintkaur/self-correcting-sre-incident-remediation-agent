# main.py
import json
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.workflow import app_graph
import app.nodes as nodes  # Imported so we can access nodes.client for clean teardown


# -----------------------------------------------------------------------------
# 1. Lifespan Management (Graceful Setup & Teardown)
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Ensures vector store handles and background resources are initialized
    and torn down cleanly without triggering Windows lock or runtime errors.
    """
    print("\n[STARTUP] Self-Correcting Incident Copilot API initialized.")
    yield
    print("\n[SHUTDOWN] Releasing Qdrant storage locks and closing resources...")
    try:
        nodes.close_client()
        print("[SHUTDOWN] Storage client closed successfully.")
    except Exception as exc:
        print(f"[SHUTDOWN WARNING] Error releasing client lock: {exc}")


# -----------------------------------------------------------------------------
# 2. FastAPI Application Setup
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Self-Correcting Incident Runbook API",
    description=(
        "Production-grade Agentic RAG system with Evaluator-Optimizer feedback loops "
        "and real-time Server-Sent Events (SSE) node telemetry."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local testing and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# 3. Pydantic Request & Response Models
# -----------------------------------------------------------------------------
class IncidentQueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        examples=["What is the procedure to failover a Patroni PostgreSQL leader node safely?"],
        description="The technical or operational incident question."
    )
    max_retries: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum self-correction loops allowed for document or groundedness verification."
    )


# -----------------------------------------------------------------------------
# 4. Asynchronous Event Stream Generator
# -----------------------------------------------------------------------------
async def sse_event_generator(query: str, max_retries: int) -> AsyncGenerator[str, None]:
    """
    Executes the LangGraph StateGraph asynchronously via `astream` and yields
    structured SSE payloads for node state transitions and generated content.
    """
    initial_state = {
        "question": query,
        "original_question": query,
        "documents": [],
        "generation": "",
        "retry_count": 0,
        "max_retries": max_retries,
        "hallucination_feedback": None,
        "web_fallback": False,
    }

    # Initial start event
    yield f"data: {json.dumps({'event': 'started', 'query': query})}\n\n"

    try:
        # astream emits each node's output dictionary as soon as that node finishes
        async for output in app_graph.astream(initial_state):
            for node_name, state_delta in output.items():

                if state_delta is None or not isinstance(state_delta, dict):
                    state_delta = {}
                    
                event_data = {
                    "event": "node_complete",
                    "node": node_name,
                    "retries": state_delta.get("retry_count", 0),
                }

                # Attach descriptive metadata depending on which node just ran
                if node_name == "retrieve":
                    docs = state_delta.get("documents", [])
                    event_data["message"] = f"Retrieved {len(docs)} candidate chunks from Qdrant."
                elif node_name == "grade_documents":
                    docs = state_delta.get("documents", [])
                    event_data["message"] = f"Evaluation complete: {len(docs)} verified relevant chunks retained."
                elif node_name == "transform_query":
                    rewritten = state_delta.get("question", "")
                    event_data["message"] = f"Search recall low. Query reformulated to: '{rewritten}'"
                elif node_name == "generate":
                    event_data["message"] = "Remediation instructions generated from verified context."
                elif node_name == "grade_hallucination":
                    feedback = state_delta.get("hallucination_feedback")
                    if feedback:
                        event_data["message"] = f"Hallucination caught by auditor! Injecting critique for self-correction: {feedback}"
                    else:
                        event_data["message"] = "Zero hallucinations detected. Output verified strictly against source context."
                elif node_name == "grade_answer_usefulness":
                    event_data["message"] = "Answer intent verified against original query."

                yield f"data: {json.dumps(event_data)}\n\n"

                # If this node produced or refined the remediation answer, stream the text payload
                if "generation" in state_delta and state_delta["generation"]:
                    yield f"data: {json.dumps({'event': 'answer_payload', 'answer': state_delta['generation']})}\n\n"

        # Signal completion
        yield f"data: {json.dumps({'event': 'done'})}\n\n"

    except Exception as err:
        yield f"data: {json.dumps({'event': 'error', 'detail': str(err)})}\n\n"


# -----------------------------------------------------------------------------
# 5. Endpoints
# -----------------------------------------------------------------------------
@app.get("/healthz", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint for container probes or load balancers."""
    return {
        "status": "healthy",
        "service": "self-correcting-rag-incident-copilot",
        "vector_store": "qdrant_embedded"
    }


@app.post("/api/v1/incident/stream")
async def stream_incident_resolution(request: IncidentQueryRequest):
    """
    SSE streaming endpoint. Connect using EventSource or standard HTTP streaming
    to observe real-time evaluator-optimizer graph transitions and receive the final answer.
    """
    return StreamingResponse(
        sse_event_generator(request.query, request.max_retries),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Prevents Nginx/reverse-proxies from buffering chunks
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000)