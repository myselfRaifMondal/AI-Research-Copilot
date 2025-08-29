import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from app.ingest import ingester
from app.vectorstore import vector_store


@pytest.fixture
def sample_txt_file():
    """Create a temporary text file for testing."""
    content = """
    This is a sample research document for testing.
    
    Introduction
    This document contains multiple paragraphs and sections that will be used to test
    the document ingestion pipeline of the AI Research Copilot system.
    
    Methodology  
    The methodology section describes the approach taken in this research.
    It includes detailed explanations of the experimental setup and data collection.
    
    Results
    The results section presents the findings from the research study.
    Key metrics and statistical analysis are provided here.
    
    Conclusion
    The conclusion summarizes the main findings and their implications.
    Future work directions are also discussed in this section.
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name
    
    # Cleanup
    if os.path.exists(f.name):
        os.remove(f.name)


@pytest.fixture
def sample_pdf_content():
    """Mock PDF content for testing."""
    return "This is sample PDF content for testing the ingestion pipeline."


class TestDocumentIngester:
    """Test document ingestion functionality."""
    
    def test_extract_text_from_txt(self, sample_txt_file):
        """Test text extraction from TXT file."""
        text, metadata = ingester.extract_text_from_txt(sample_txt_file)
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert "sample research document" in text.lower()
        
        assert isinstance(metadata, dict)
        assert "source" in metadata
        assert "pages" in metadata
        assert "title" in metadata
        assert metadata["pages"] == 1
        assert metadata["source"] == os.path.basename(sample_txt_file)
    
    @patch('app.ingest.PdfReader')
    def test_extract_text_from_pdf(self, mock_pdf_reader, sample_pdf_content):
        """Test PDF text extraction with mocked PDF reader."""
        # Mock PDF reader
        mock_page = Mock()
        mock_page.extract_text.return_value = sample_pdf_content
        
        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_reader.metadata = None
        
        mock_pdf_reader.return_value = mock_reader
        
        # Test extraction
        with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_pdf:
            text, metadata = ingester.extract_text_from_pdf(temp_pdf.name)
            
            assert isinstance(text, str)
            assert sample_pdf_content in text
            assert isinstance(metadata, dict)
            assert metadata["pages"] == 1
            assert metadata["source"] == os.path.basename(temp_pdf.name)
    
    def test_chunk_document(self, sample_txt_file):
        """Test document chunking functionality."""
        text, metadata = ingester.extract_text_from_txt(sample_txt_file)
        documents = ingester.chunk_document(text, metadata, sample_txt_file)
        
        assert isinstance(documents, list)
        assert len(documents) > 0
        
        # Check document structure
        for doc in documents:
            assert hasattr(doc, 'page_content')
            assert hasattr(doc, 'metadata')
            assert isinstance(doc.page_content, str)
            assert isinstance(doc.metadata, dict)
            
            # Check metadata preservation
            assert "source" in doc.metadata
            assert "chunk_id" in doc.metadata
            assert "page" in doc.metadata
            assert "total_chunks" in doc.metadata
            assert doc.metadata["total_chunks"] == len(documents)
    
    @patch('app.vectorstore.vector_store.add_documents')
    def test_ingest_file_sync(self, mock_add_documents, sample_txt_file):
        """Test synchronous file ingestion."""
        # Mock vector store
        mock_add_documents.return_value = ["doc_1", "doc_2"]
        
        result = ingester.ingest_file_sync(sample_txt_file)
        
        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["file_path"] == sample_txt_file
        assert result["filename"] == os.path.basename(sample_txt_file)
        assert "chunks_created" in result
        assert "total_characters" in result
        assert result["chunks_created"] > 0
        
        # Verify vector store was called
        mock_add_documents.assert_called_once()
    
    def test_ingest_file_sync_unsupported_format(self):
        """Test ingestion with unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix='.docx') as temp_file:
            result = ingester.ingest_file_sync(temp_file.name)
            
            assert result["status"] == "error"
            assert "Unsupported file type" in result["error"]
    
    def test_ingest_file_sync_nonexistent_file(self):
        """Test ingestion with non-existent file."""
        fake_path = "/path/that/does/not/exist.txt"
        result = ingester.ingest_file_sync(fake_path)
        
        assert result["status"] == "error"
        assert "error" in result


class TestVectorStoreIntegration:
    """Test vector store integration."""
    
    @patch('app.vectorstore.vector_store.add_documents')
    @patch('app.vectorstore.vector_store.get_stats')
    def test_vector_store_mock_integration(self, mock_get_stats, mock_add_documents, sample_txt_file):
        """Test vector store integration with mocks."""
        # Mock vector store responses
        mock_add_documents.return_value = ["doc_1", "doc_2"]
        mock_get_stats.return_value = {
            "type": "FAISS",
            "index_size": 2,
            "local_path": "vectordb/faiss_index"
        }
        
        # Test ingestion
        result = ingester.ingest_file_sync(sample_txt_file)
        
        assert result["status"] == "success"
        
        # Verify vector store interactions
        mock_add_documents.assert_called_once()
        
        # Test stats retrieval
        stats = vector_store.get_stats()
        assert stats["type"] == "FAISS"
        assert stats["index_size"] == 2


@pytest.mark.integration
class TestIngestionEndToEnd:
    """Integration tests for complete ingestion pipeline."""
    
    @pytest.mark.skipif(
        not os.getenv("RUN_INTEGRATION_TESTS"),
        reason="Integration tests require RUN_INTEGRATION_TESTS=1"
    )
    def test_complete_ingestion_pipeline(self, sample_txt_file):
        """Test complete ingestion pipeline (requires real vector store)."""
        # This test requires actual vector store setup
        # Skip in CI unless explicitly enabled
        
        result = ingester.ingest_file_sync(sample_txt_file)
        
        assert result["status"] == "success"
        assert result["chunks_created"] > 0
        
        # Verify documents are searchable
        from app.chain import rag_chain
        query_result = rag_chain.query_sync("What is this document about?")
        
        assert "answer" in query_result
        assert len(query_result["sources"]) > 0

