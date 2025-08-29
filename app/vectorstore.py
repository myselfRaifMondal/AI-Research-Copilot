"""Vector database abstraction with local embeddings - FIXED PATHS."""

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
        self.model_name = model_name or settings.local_embedding_model
        logger.info(f"Loading embedding model: {self.model_name}")
        self._model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model safely."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not texts:
            return []
        self._initialize_model()
        embeddings = self._model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        self._initialize_model()
        embedding = self._model.encode([text], convert_to_tensor=False)
        return embedding[0].tolist()


class VectorStoreManager:
    """
    Unified interface for vector databases with local embeddings.
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
        """Initialize vector store with proper path handling."""
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
                pc.create_index(
                    name=index_name,
                    dimension=384,
                    metric="cosine"
                )
                
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
        """Initialize FAISS vector store with flexible path detection."""
        try:
            # Try different possible locations for FAISS files
            possible_paths = [
                "vectordb/faiss_index",  # Standard format: vectordb/faiss_index.faiss, vectordb/faiss_index.pkl
                "vectordb/faiss_index/index"  # Directory format: vectordb/faiss_index/index.faiss, vectordb/faiss_index/index.pkl
            ]
            
            loaded = False
            
            for faiss_path in possible_paths:
                try:
                    faiss_file = f"{faiss_path}.faiss"
                    pkl_file = f"{faiss_path}.pkl"
                    
                    logger.info(f"Checking for FAISS files at: {faiss_path}")
                    logger.info(f"Looking for: {faiss_file} and {pkl_file}")
                    
                    if os.path.exists(faiss_file) and os.path.exists(pkl_file):
                        logger.info(f"Found FAISS files at: {faiss_path}")
                        
                        self.vectorstore = FAISS.load_local(
                            faiss_path,
                            self.embeddings,
                            allow_dangerous_deserialization=True
                        )
                        
                        logger.info(f"✅ Loaded existing FAISS index with {self.vectorstore.index.ntotal} vectors")
                        loaded = True
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to load from {faiss_path}: {e}")
                    continue
            
            if not loaded:
                logger.info("No existing FAISS index found - will create on first document addition")
                self.vectorstore = None
                
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
                    # Save FAISS index in directory format (what FAISS expects)
                    os.makedirs("vectordb/faiss_index", exist_ok=True)
                    self.vectorstore.save_local("vectordb/faiss_index/index")
                
                logger.info(f"Created new vector store with {len(documents)} documents")
            else:
                # Add to existing vector store
                if isinstance(self.vectorstore, FAISS):
                    self.vectorstore.add_documents(documents)
                    # Save in the format that was successfully loaded
                    os.makedirs("vectordb/faiss_index", exist_ok=True)
                    self.vectorstore.save_local("vectordb/faiss_index/index")
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
                    "embedding_model": getattr(self.embeddings, 'model_name', 'Unknown')
                }
                
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {"type": "Error", "error": str(e)}


# Global vector store manager
vector_store = VectorStoreManager()
