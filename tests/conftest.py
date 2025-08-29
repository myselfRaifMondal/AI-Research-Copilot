import os
import pytest
from unittest.mock import AsyncMock

# Set test environment variables BEFORE any app imports
os.environ["OPENAI_API_KEY"] = "sk-test-key-for-testing"
os.environ["API_KEY"] = "dev-api-key" 
os.environ["USE_PINECONE"] = "false"
os.environ["DEBUG"] = "true"

# Now safe to import app modules
from fastapi.testclient import TestClient
from app.server import app


@pytest.fixture
def test_client():
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_query_response():
    """Mock RAG chain query response."""
    return {
        "answer": "This is a test answer about machine learning in healthcare research.",
        "sources": [
            {
                "source": "test_paper.pdf",
                "page": 1,
                "score": 0.85,
                "content_preview": "Machine learning has shown significant promise in healthcare applications..."
            },
            {
                "source": "test_paper.pdf", 
                "page": 3,
                "score": 0.78,
                "content_preview": "The methodology involved training neural networks on clinical data..."
            }
        ],
        "raw_llm_output": "This is a test answer about machine learning in healthcare research.",
        "query_metadata": {
            "retrieved_chunks": 2,
            "model_used": "gpt-3.5-turbo",
            "avg_relevance_score": 0.815
        }
    }
