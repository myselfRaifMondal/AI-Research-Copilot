import pytest
from unittest.mock import Mock, patch, AsyncMock

# Import from conftest which sets up environment
from .conftest import test_client, mock_query_response


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    @patch('app.vectorstore.vector_store.get_stats')
    def test_health_endpoint(self, mock_get_stats, test_client):
        """Test health check endpoint returns correct format."""
        mock_get_stats.return_value = {
            "type": "FAISS",
            "index_size": 0,
            "local_path": "vectordb/faiss_index"
        }
        
        response = test_client.get("/health")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "vector_store" in data
        
        assert data["status"] == "healthy"
        assert data["vector_store"]["type"] == "FAISS"
    
    @patch('app.vectorstore.vector_store.get_stats')
    def test_health_endpoint_error(self, mock_get_stats, test_client):
        """Test health endpoint when vector store fails."""
        mock_get_stats.side_effect = Exception("Vector store error")
        
        response = test_client.get("/health")
        
        assert response.status_code == 500


class TestQueryEndpoint:
    """Test query endpoint functionality."""
    
    @patch('app.chain.rag_chain.query')
    def test_query_endpoint_success(self, mock_query, test_client, mock_query_response):
        """Test successful query endpoint."""
        # Create a proper async mock
        mock_query.return_value = AsyncMock(return_value=mock_query_response)()
        
        # Make request with proper auth
        response = test_client.post(
            "/query",
            json={"question": "What are the main findings about machine learning in healthcare?"},
            headers={"Authorization": "Bearer dev-api-key"}
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "query_metadata" in data
        assert "timestamp" in data
        
        assert data["answer"] == mock_query_response["answer"]
        assert len(data["sources"]) == 2
        assert data["sources"][0]["source"] == "test_paper.pdf"
    
    def test_query_endpoint_no_auth(self, test_client):
        """Test query endpoint without authentication."""
        response = test_client.post(
            "/query",
            json={"question": "Test question"}
        )
        
        assert response.status_code == 403  # FastAPI returns 403 for missing auth
    
    def test_query_endpoint_invalid_auth(self, test_client):
        """Test query endpoint with invalid authentication."""
        response = test_client.post(
            "/query",
            json={"question": "Test question"},
            headers={"Authorization": "Bearer invalid-key"}
        )
        
        assert response.status_code == 401
    
    def test_query_endpoint_empty_question(self, test_client):
        """Test query endpoint with empty question."""
        response = test_client.post(
            "/query",
            json={"question": ""},
            headers={"Authorization": "Bearer dev-api-key"}
        )
        
        assert response.status_code == 422  # Validation error
    
    @patch('app.chain.rag_chain.query')
    def test_query_endpoint_max_sources(self, mock_query, test_client):
        """Test query endpoint with max_sources parameter."""
        mock_response = {
            "answer": "Test answer",
            "sources": [{"source": f"doc{i}.pdf", "page": 1, "score": 0.9} for i in range(10)],
            "query_metadata": {}
        }
        
        mock_query.return_value = AsyncMock(return_value=mock_response)()
        
        response = test_client.post(
            "/query", 
            json={"question": "Test question", "max_sources": 3},
            headers={"Authorization": "Bearer dev-api-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 3


class TestRAGChain:
    """Test RAG chain functionality."""
    
    @patch('app.vectorstore.vector_store.similarity_search')
    @patch('app.chain.rag_chain.llm')
    def test_query_sync_success(self, mock_llm, mock_similarity_search):
        """Test synchronous RAG chain query."""
        # Import here to avoid circular import issues
        from langchain.schema import Document
        from app.chain import rag_chain
        
        mock_docs = [
            (Document(
                page_content="Machine learning in healthcare shows great promise.",
                metadata={"source": "paper1.pdf", "page": 1}
            ), 0.15),  # Lower score = higher similarity
            (Document(
                page_content="Neural networks can predict patient outcomes.",
                metadata={"source": "paper1.pdf", "page": 2}
            ), 0.22)
        ]
        mock_similarity_search.return_value = mock_docs
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Machine learning has shown significant potential in healthcare applications."
        mock_llm.invoke.return_value = mock_response
        mock_llm.model_name = "gpt-3.5-turbo"
        
        # Test query
        result = rag_chain.query_sync("What is machine learning's role in healthcare?")
        
        assert isinstance(result, dict)
        assert "answer" in result
        assert "sources" in result
        assert "raw_llm_output" in result
        
        assert result["answer"] == mock_response.content
        assert len(result["sources"]) == 2
        assert result["sources"][0]["source"] == "paper1.pdf"
        assert result["sources"][0]["score"] == 0.85  # 1 - 0.15
    
    @patch('app.vectorstore.vector_store.similarity_search')
    def test_query_sync_no_documents(self, mock_similarity_search):
        """Test RAG chain when no documents are found."""
        from app.chain import rag_chain
        
        mock_similarity_search.return_value = []
        
        result = rag_chain.query_sync("Test question")
        
        assert result["answer"].startswith("I couldn't find relevant information")
        assert result["sources"] == []
    
    @patch('app.vectorstore.vector_store.similarity_search')
    def test_query_sync_error_handling(self, mock_similarity_search):
        """Test RAG chain error handling."""
        from app.chain import rag_chain
        
        mock_similarity_search.side_effect = Exception("Vector search failed")
        
        result = rag_chain.query_sync("Test question")
        
        assert "error" in result["answer"].lower()
        assert result["sources"] == []


class TestIngestionEndpoint:
    """Test document ingestion endpoint."""
    
    def test_ingest_endpoint_no_auth(self, test_client):
        """Test ingestion endpoint without authentication."""
        response = test_client.post("/ingest")
        
        assert response.status_code == 403  # FastAPI returns 403 for missing auth
    
    def test_ingest_endpoint_invalid_file_type(self, test_client):
        """Test ingestion with invalid file type."""
        files = {"file": ("test.docx", b"fake content", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        
        response = test_client.post(
            "/ingest",
            files=files,
            headers={"Authorization": "Bearer dev-api-key"}
        )
        
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]
    
    def test_ingest_endpoint_file_too_large(self, test_client):
        """Test ingestion with file too large."""
        large_content = b"x" * (51 * 1024 * 1024)  # 51MB
        files = {"file": ("large.txt", large_content, "text/plain")}
        
        response = test_client.post(
            "/ingest",
            files=files,
            headers={"Authorization": "Bearer dev-api-key"}
        )
        
        assert response.status_code == 400
        assert "File too large" in response.json()["detail"]


class TestStatsEndpoint:
    """Test system statistics endpoint."""
    
    @patch('app.vectorstore.vector_store.get_stats')
    def test_stats_endpoint(self, mock_get_stats, test_client):
        """Test stats endpoint returns system information."""
        mock_get_stats.return_value = {
            "type": "FAISS",
            "index_size": 5,
            "local_path": "vectordb/faiss_index"
        }
        
        response = test_client.get(
            "/stats",
            headers={"Authorization": "Bearer dev-api-key"}
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert "vector_store" in data
        assert "active_tasks" in data
        assert "system_info" in data
        
        assert data["vector_store"]["type"] == "FAISS"
        assert data["vector_store"]["index_size"] == 5


class TestAsyncFunctionality:
    """Test async functionality in RAG chain."""
    
    @patch('app.vectorstore.vector_store.similarity_search')
    @patch('app.chain.rag_chain.llm')
    @pytest.mark.asyncio
    async def test_async_query(self, mock_llm, mock_similarity_search):
        """Test async query functionality."""
        from langchain.schema import Document
        from app.chain import rag_chain
        
        # Mock documents
        mock_docs = [
            (Document(
                page_content="Test content",
                metadata={"source": "test.pdf", "page": 1}
            ), 0.1)
        ]
        mock_similarity_search.return_value = mock_docs
        
        # Mock async LLM response
        mock_response = Mock()
        mock_response.content = "Test async response"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm.model_name = "gpt-3.5-turbo"
        
        # Test async query
        result = await rag_chain.query("Test async question")
        
        assert isinstance(result, dict)
        assert "answer" in result
        assert result["answer"] == "Test async response"
        
        # Verify async method was called
        mock_llm.ainvoke.assert_called_once()
