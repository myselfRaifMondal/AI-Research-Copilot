"""Vector database abstraction with local embeddings - RECURSION FIXED."""

import os
import logging
from typing import List, Tuple, Optional, Any, Dict
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

from .config import settings

logger = logging.getLogger(__name__)


class LocalEmbeddings(Embeddings):
    """Fixed local embeddings without recursion issues."""
    
    def __init__(self, model_name: str = None):
        # Set attributes directly without calling methods
        self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self._model = None
        logger.info(f"Embedding model configured: {self.model_name}")
    
    def _ensure_model_loaded(self):
        """Load model only when needed."""
        if self._model is None:
            logger.info(f"Loading SentenceTransformer: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("SentenceTransformer loaded successfully")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not texts:
            return []
        
        self._ensure_model_loaded()
        embeddings = self._model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        self._ensure_model_loaded()
        embedding = self._model.encode([text], convert_to_tensor=False)
        return embedding[0].tolist()


class VectorStoreManager:
    """Unified interface for vector databases with local embeddings."""
    
    def __init__(self):
        logger.info("Initializing VectorStoreManager...")
        self.embeddings = None
        self.vectorstore = None
        
        try:
            self.embeddings = LocalEmbeddings()
            logger.info("✅ Embeddings initialized")
            
            self._initialize_vectorstore()
            logger.info("✅ VectorStoreManager initialization complete")
            
        except Exception as e:
            logger.error(f"❌ VectorStoreManager initialization failed: {e}")
            self.embeddings = None
            self.vectorstore = None
    
    def _initialize_vectorstore(self):
        """Initialize vector store."""
        if settings.use_pinecone and settings.pinecone_api_key:
            self._initialize_pinecone()
        else:
            self._initialize_faiss()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone vector store."""
        try:
            from langchain_pinecone import PineconeVectorStore
            import pinecone as pc
            
            pc.init(
                api_key=settings.pinecone_api_key,
                environment=settings.pinecone_environment
            )
            
            index_name = settings.pinecone_index_name
            if index_name not in pc.list_indexes():
                logger.info(f"Creating Pinecone index: {index_name}")
                pc.create_index(name=index_name, dimension=384, metric="cosine")
            
            self.vectorstore = PineconeVectorStore(
                index_name=index_name,
                embedding=self.embeddings
            )
            logger.info(f"✅ Pinecone vector store initialized: {index_name}")
            
        except Exception as e:
            logger.error(f"❌ Pinecone initialization failed: {e}")
            logger.info("Falling back to FAISS")
            self._initialize_faiss()
    
    def _initialize_faiss(self):
        """Initialize FAISS vector store."""
        try:
            faiss_dir = "vectordb/faiss_index"
            logger.info(f"Looking for FAISS index in: {faiss_dir}")
            
            # Check for required files
            faiss_file = os.path.join(faiss_dir, "index.faiss")
            pkl_file = os.path.join(faiss_dir, "index.pkl")
            
            logger.info(f"Checking files:")
            logger.info(f"  {faiss_file} -> exists: {os.path.exists(faiss_file)}")
            logger.info(f"  {pkl_file} -> exists: {os.path.exists(pkl_file)}")
            
            if os.path.exists(faiss_file) and os.path.exists(pkl_file):
                logger.info("✅ FAISS files found, loading index...")
                
                self.vectorstore = FAISS.load_local(
                    faiss_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                vectors_count = self.vectorstore.index.ntotal
                logger.info(f"✅ FAISS index loaded successfully with {vectors_count} vectors")
                
            else:
                logger.warning("❌ FAISS index files not found")
                logger.info("Vector store will be created when documents are added")
                self.vectorstore = None
                
        except Exception as e:
            logger.error(f"❌ FAISS initialization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.vectorstore = None
    
    def similarity_search(self, query: str, k: int = None, score_threshold: float = 0.7) -> List[Tuple[Document, float]]:
        """Perform similarity search."""
        if self.vectorstore is None:
            logger.warning("Vector store not initialized")
            return []
        
        k = k or settings.retrieval_k
        
        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
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
                    "index_size": self.vectorstore.index.ntotal,
                    "local_path": "vectordb/faiss_index",
                    "embedding_model": self.embeddings.model_name if self.embeddings else "Unknown"
                }
            elif self.vectorstore:
                return {
                    "type": "Pinecone",
                    "index_name": settings.pinecone_index_name,
                    "environment": settings.pinecone_environment,
                    "embedding_model": self.embeddings.model_name if self.embeddings else "Unknown"
                }
            else:
                return {
                    "type": "Not initialized",
                    "index_size": 0,
                    "embedding_model": self.embeddings.model_name if self.embeddings else "Unknown"
                }
                
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"type": "Error", "error": str(e)}


# Global vector store manager
vector_store = VectorStoreManager()
