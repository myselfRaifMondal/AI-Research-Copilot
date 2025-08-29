"""Configuration settings and prompt templates."""

import os
from typing import Optional
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = ConfigDict(env_file=".env")
    
    # Local AI Models Configuration
    use_local_models: bool = True
    local_llm_model: str = "microsoft/DialoGPT-medium"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_tokens: int = 512
    
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


# Global settings instance
settings = Settings()

# Prompt templates optimized for local models
SYSTEM_PROMPT = """You are a helpful research assistant. Answer questions based ONLY on the provided context from academic papers and documents. 

Guidelines:
1. Only use information from the provided context
2. If context doesn't contain enough information, say so
3. Cite sources using [filename:page] format
4. Be precise and academic in tone
5. Keep responses concise and focused

Context: {context}

Question: {question}

Answer:"""

CITATION_PROMPT = """Extract and format citations from the provided context.

Context: {context}
"""

# Model configuration notes
MODEL_CONFIG_NOTES = """
Local Model Setup:
- LLM: microsoft/DialoGPT-medium (good balance of quality and speed)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (fast and accurate)
- No API keys required - runs completely offline
- First run will download models (~1GB total)
"""
