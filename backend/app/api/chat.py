import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from loguru import logger

from app.graph.workflow import app as graph_app

router = APIRouter()


class ChatRequest(BaseModel):
    """
    Validation schema for incoming chat requests.
    Tracks user prompt and persistent thread identifier.
    """
    message: str
    thread_id: str


@router.post("/stream")
async def stream_chat(payload: ChatRequest) -> StreamingResponse:
    """
    Exposes a streaming Server-Sent Events (SSE) connection that executes
    the LangGraph state machine, yielding active node logs and model tokens.
    """
    logger.info(f"API Request: POST /stream - Thread ID: {payload.thread_id}")

    async def event_generator():
        config = {"configurable": {"thread_id": payload.thread_id}}
        try:
            # Stream events from the compiled LangGraph workflow
            async for event in graph_app.astream_events(
                {"question": payload.message},
                config=config,
                version="v2"
            ):
                kind = event.get("event")
                name = event.get("name")

                # 1. Yield node start signals
                if kind == "on_chain_start" and name in {"retrieve", "grade_documents", "generate", "rewrite"}:
                    yield f"data: {json.dumps({'event': 'node_start', 'data': name})}\n\n"

                # 2. Yield node end signals
                elif kind == "on_chain_end" and name in {"retrieve", "grade_documents", "generate", "rewrite"}:
                    yield f"data: {json.dumps({'event': 'node_end', 'data': name})}\n\n"

                # 3. Yield token streams from Chat Model
                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield f"data: {json.dumps({'event': 'token', 'data': chunk.content})}\n\n"

        except Exception as e:
            logger.error(f"Stream error encountered: {str(e)}")
            yield f"data: {json.dumps({'event': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
