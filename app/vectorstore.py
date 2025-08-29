"""Vector database abstraction with local embeddings."""

import os
import logging
from typing import List, Tuple, Optional, Any, Dict
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import numpy as np

from .config import settings

logger = logging.getLogger(__name__)


class LocalEmbeddings(Embeddings):
    """Local embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.local_embedding_model
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("Embedding model loaded successfully")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()


class VectorStoreManager:
    """
    Unified interface for vector databases with local embeddings.
    
    Uses sentence-transformers for completely free, local embeddings.
    """
    
    def __init__(self):
        self.embeddings = self._initialize_embeddings()
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_embeddings(self) -> LocalEmbeddings:
        """Initialize local embeddings model."""
        try:
            return LocalEmbeddings()
        except Exception as e:
            logger.error(f"Failed to initialize local embeddings: {e}")
            raise
    
    def _initialize_vectorstore(self):
        """Initialize vector store based on configuration."""
        if settings.use_pinecone and settings.pinecone_api_key:
            self._initialize_pinecone()
        else:
            self._initialize_faiss()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone vector store."""
        try:
            from langchain_pinecone import PineconeVectorStore
            import pinecone as pc
            
            # Initialize Pinecone client
            pc.init(
                api_key=settings.pinecone_api_key,
                environment=settings.pinecone_environment
            )
            
            # Check if index exists, create if not
            index_name = settings.pinecone_index_name
            if index_name not in pc.list_indexes():
                logger.info(f"Creating Pinecone index: {index_name}")
                pc.create_index(
                    name=index_name,
                    dimension=384,  # sentence-transformers/all-MiniLM-L6-v2 dimension
                    metric="cosine"
                )
                
            # Initialize vector store
            self.vectorstore = PineconeVectorStore(
                index_name=index_name,
                embedding=self.embeddings
            )
            logger.info(f"Initialized Pinecone vector store: {index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            logger.info("Falling back to FAISS")
            self._initialize_faiss()
    
    def _initialize_faiss(self):
        """Initialize FAISS vector store for local development."""
        faiss_path = "vectordb/faiss_index"
        
        try:
            if os.path.exists(f"{faiss_path}.faiss"):
                # Load existing FAISS index
                self.vectorstore = FAISS.load_local(
                    faiss_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Loaded existing FAISS index")
            else:
                # Will create on first ingestion
                self.vectorstore = None
                logger.info("FAISS index will be created on first ingestion")
                
        except Exception as e:
            logger.error(f"FAISS initialization error: {e}")
            self.vectorstore = None
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to vector store."""
        try:
            if not documents:
                return []
            
            if self.vectorstore is None:
                # Create new vector store
                if settings.use_pinecone and settings.pinecone_api_key:
                    from langchain_pinecone import PineconeVectorStore
                    self.vectorstore = PineconeVectorStore.from_documents(
                        documents,
                        embedding=self.embeddings,
                        index_name=settings.pinecone_index_name
                    )
                else:
                    self.vectorstore = FAISS.from_documents(documents, self.embeddings)
                    # Save FAISS index locally
                    os.makedirs("vectordb", exist_ok=True)
                    self.vectorstore.save_local("vectordb/faiss_index")
                
                logger.info(f"Created new vector store with {len(documents)} documents")
            else:
                # Add to existing vector store
                if isinstance(self.vectorstore, FAISS):
                    self.vectorstore.add_documents(documents)
                    self.vectorstore.save_local("vectordb/faiss_index")
                else:
                    self.vectorstore.add_documents(documents)
                
                logger.info(f"Added {len(documents)} documents to existing vector store")
            
            return [doc.metadata.get("id", "") for doc in documents]
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = None,
        score_threshold: float = 0.7
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search."""
        if self.vectorstore is None:
            logger.warning("Vector store not initialized")
            return []
        
        k = k or settings.retrieval_k
        
        try:
            # Get documents with scores
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # Filter by score threshold (lower scores = higher similarity for cosine distance)
            filtered_results = [
                (doc, score) for doc, score in docs_with_scores
                if score <= (1 - score_threshold)
            ]
            
            logger.info(f"Retrieved {len(filtered_results)} relevant documents")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            if isinstance(self.vectorstore, FAISS):
                return {
                    "type": "FAISS",
                    "index_size": self.vectorstore.index.ntotal if self.vectorstore else 0,
                    "local_path": "vectordb/faiss_index",
                    "embedding_model": self.embeddings.model_name
                }
            elif self.vectorstore:
                return {
                    "type": "Pinecone",
                    "index_name": settings.pinecone_index_name,
                    "environment": settings.pinecone_environment,
                    "embedding_model": self.embeddings.model_name
                }
            else:
                return {
                    "type": "Not initialized", 
                    "index_size": 0,
                    "embedding_model": self.embeddings.model_name
                }
                
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {"type": "Error", "error": str(e)}


# Global vector store manager
vector_store = VectorStoreManager()
