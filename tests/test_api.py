import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from app.main import app

def test_health_endpoint():
    """Verify that the health check endpoint returns 200 and healthy status."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@pytest.mark.asyncio
async def test_stream_chat_endpoint(monkeypatch):
    """Test `/api/chat/stream` post and check SSE responses with mock events."""
    async def mock_stream(*args, **kwargs):
        yield {
            "event": "on_chain_start",
            "name": "retrieve",
            "data": {}
        }
        yield {
            "event": "on_chat_model_stream",
            "name": "chat_model",
            "data": {"chunk": MagicMock(content="Jon")}
        }
        yield {
            "event": "on_chat_model_stream",
            "name": "chat_model",
            "data": {"chunk": MagicMock(content=" Snow")}
        }
        yield {
            "event": "on_chain_end",
            "name": "retrieve",
            "data": {}
        }

    monkeypatch.setattr("app.api.chat.graph_app.astream_events", mock_stream)

    # Use TestClient with event streaming support
    with TestClient(app) as client:
        payload = {"message": "Who is Jon Snow?", "thread_id": "test-session-thread"}
        response = client.post("/api/chat/stream", json=payload)
        
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        
        # Verify SSE stream lines
        lines = [line if isinstance(line, str) else line.decode("utf-8") for line in response.iter_lines()]
        data_lines = [line[5:].strip() for line in lines if line.startswith("data:")]
        
        assert len(data_lines) == 4
        
        event_1 = json.loads(data_lines[0])
        assert event_1["event"] == "node_start"
        assert event_1["data"] == "retrieve"

        event_2 = json.loads(data_lines[1])
        assert event_2["event"] == "token"
        assert event_2["data"] == "Jon"

        event_3 = json.loads(data_lines[2])
        assert event_3["event"] == "token"
        assert event_3["data"] == " Snow"

        event_4 = json.loads(data_lines[3])
        assert event_4["event"] == "node_end"
        assert event_4["data"] == "retrieve"
