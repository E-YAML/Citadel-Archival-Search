import sys
import os
import pytest
from unittest.mock import AsyncMock, MagicMock

# Set mock environment variables at import time so Pydantic settings validation passes
os.environ["PROJECT_NAME"] = "Test Citadel Archival Search"
os.environ["GROQ_API_KEY"] = "gsk_mock_api_key_for_testing_purposes"
os.environ["QDRANT_URL"] = "http://localhost:6333"
os.environ["QDRANT_API_KEY"] = "mock_qdrant_api_key"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "mock_neo4j_password"

# Ensure backend directory is in the python path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_path not in sys.path:
    sys.path.append(backend_path)

@pytest.fixture
def mock_qdrant(monkeypatch):
    """Mock the QdrantService client to prevent database requests during testing."""
    from app.services.qdrant_service import QdrantService
    mock_client = AsyncMock()
    # Mock collection_exists and other methods
    mock_client.collection_exists.return_value = True
    monkeypatch.setattr(QdrantService, "get_client", lambda self: mock_client)
    return mock_client

@pytest.fixture
def mock_neo4j(monkeypatch):
    """Mock the Neo4jService driver to prevent Neo4j requests during testing."""
    from app.services.neo4j_service import Neo4jService
    mock_driver = MagicMock()
    mock_session = MagicMock()
    
    # Session run mock setup
    mock_session.run = AsyncMock()
    
    # Make session context manager work
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    
    mock_driver.session.return_value = mock_session
    monkeypatch.setattr(Neo4jService, "get_driver", lambda self: mock_driver)
    return mock_session
