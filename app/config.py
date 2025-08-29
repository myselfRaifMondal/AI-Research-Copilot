import os
from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_version: str = "2024-02-01"
    
    # Vector Database Configuration
    use_pinecone: bool = False
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "us-east1-gcp"
    pinecone_index_name: str = "research-copilot"
    
    # API Configuration
    api_key: str = "dev-api-key"
    jwt_secret: str = "change-me-in-production"
    
    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_k: int = 5
    
    # Application Configuration
    debug: bool = False
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"


# Global settings instance
settings = Settings()

# Prompt templates centralized here for easy modification
SYSTEM_PROMPT = """You are a helpful research assistant. Answer questions based ONLY on the provided context from academic papers and documents. 

IMPORTANT GUIDELINES:
1. Only use information from the provided context - do not use your general knowledge
2. If the context doesn't contain enough information to answer the question, say so
3. Always cite your sources using the format [filename:page] when referencing specific information
4. Be precise and academic in your tone
5. If multiple sources support a claim, cite all relevant sources

Context from research documents:
{context}

Question: {question}

Please provide a comprehensive answer with proper citations."""

CITATION_PROMPT = """Based on the provided context, extract and format citations in academic style. 
Include page numbers when available.

Context: {context}
"""

# Chunking strategy explanation
CHUNK_CONFIG_NOTES = """
Chunking Strategy:
- chunk_size=1000: Optimal balance between context and specificity for academic papers
- chunk_overlap=150: Ensures important concepts aren't split across chunks
- RecursiveCharacterTextSplitter: Preserves sentence and paragraph boundaries
"""

