"""Vector database abstraction - CLEAN IMPLEMENTATION."""

import os
import logging
from typing import List, Tuple, Optional, Any, Dict
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    logger.warning("sentence-transformers not available")
    HAS_SENTENCE_TRANSFORMERS = False

from .config import settings


class SimpleLocalEmbeddings(Embeddings):
    """Simple local embeddings without any recursion issues."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        logger.info(f"Embeddings configured: {model_name}")
    
    def _load_model(self):
        """Load model only when first needed."""
        if self._model is None:
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError("sentence-transformers package required")
            
            logger.info(f"Loading SentenceTransformer: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("✅ SentenceTransformer loaded successfully")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents."""
        if not texts:
            return []
        
        self._load_model()
        try:
            embeddings = self._model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            # Return zero embeddings as fallback
            return [[0.0] * 384 for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed single query."""
        self._load_model()
        try:
            embedding = self._model.encode([text], convert_to_tensor=False, show_progress_bar=False)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            # Return zero embedding as fallback
            return [0.0] * 384


class VectorStoreManager:
    """Clean vector store manager without recursion issues."""
    
    def __init__(self):
        logger.info("🔄 Initializing VectorStoreManager...")
        self.embeddings = None
        self.vectorstore = None
        
        try:
            # Initialize embeddings
            self.embeddings = SimpleLocalEmbeddings()
            logger.info("✅ Embeddings initialized")
            
            # Try to load existing FAISS index
            self._safe_load_faiss()
            
            logger.info("✅ VectorStoreManager ready")
            
        except Exception as e:
            logger.error(f"❌ VectorStoreManager init failed: {e}")
    
    def _safe_load_faiss(self):
        """Safely attempt to load FAISS index."""
        try:
            faiss_dir = "vectordb/faiss_index"
            
            # Check if files exist
            faiss_file = os.path.join(faiss_dir, "index.faiss")
            pkl_file = os.path.join(faiss_dir, "index.pkl")
            
            logger.info(f"Checking FAISS files:")
            logger.info(f"  📄 {faiss_file} -> {os.path.exists(faiss_file)}")
            logger.info(f"  📄 {pkl_file} -> {os.path.exists(pkl_file)}")
            
            if os.path.exists(faiss_file) and os.path.exists(pkl_file):
                # Check file sizes to ensure they're not corrupted
                faiss_size = os.path.getsize(faiss_file)
                pkl_size = os.path.getsize(pkl_file)
                
                logger.info(f"  📊 FAISS file size: {faiss_size:,} bytes")
                logger.info(f"  📊 PKL file size: {pkl_size:,} bytes")
                
                if faiss_size > 1000 and pkl_size > 1000:  # Reasonable minimum sizes
                    logger.info("🔄 Attempting to load FAISS index...")
                    
                    # Load with strict error handling
                    self.vectorstore = FAISS.load_local(
                        faiss_dir,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    
                    count = self.vectorstore.index.ntotal
                    logger.info(f"✅ FAISS index loaded: {count:,} vectors")
                    
                else:
                    logger.warning("⚠️  FAISS files too small - possibly corrupted")
                    self.vectorstore = None
            else:
                logger.info("ℹ️  No existing FAISS index found")
                self.vectorstore = None
                
        except Exception as e:
            logger.error(f"❌ Failed to load FAISS index: {e}")
            logger.info("🔧 Will create new index when documents are added")
            self.vectorstore = None
    
    def recreate_index(self):
        """Force recreate the FAISS index from scratch."""
        logger.info("🔄 Recreating FAISS index...")
        
        # Remove existing files
        faiss_dir = "vectordb/faiss_index"
        try:
            if os.path.exists(os.path.join(faiss_dir, "index.faiss")):
                os.remove(os.path.join(faiss_dir, "index.faiss"))
            if os.path.exists(os.path.join(faiss_dir, "index.pkl")):
                os.remove(os.path.join(faiss_dir, "index.pkl"))
            logger.info("🗑️  Removed corrupted FAISS files")
        except Exception as e:
            logger.warning(f"Could not remove old files: {e}")
        
        self.vectorstore = None
        logger.info("✅ Ready to create new index")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to vector store."""
        if not documents:
            return []
        
        try:
            if self.vectorstore is None:
                logger.info(f"🔄 Creating new FAISS index with {len(documents)} documents...")
                
                # Create new vector store
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
                
                # Save it
                os.makedirs("vectordb/faiss_index", exist_ok=True)
                self.vectorstore.save_local("vectordb/faiss_index")
                
                count = self.vectorstore.index.ntotal
                logger.info(f"✅ Created FAISS index: {count:,} vectors")
            else:
                logger.info(f"➕ Adding {len(documents)} documents to existing index...")
                
                self.vectorstore.add_documents(documents)
                self.vectorstore.save_local("vectordb/faiss_index")
                
                count = self.vectorstore.index.ntotal
                logger.info(f"✅ Updated FAISS index: {count:,} total vectors")
            
            return [doc.metadata.get("id", "") for doc in documents]
            
        except Exception as e:
            logger.error(f"❌ Error adding documents: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5, score_threshold: float = 0.7) -> List[Tuple[Document, float]]:
        """Perform similarity search."""
        if self.vectorstore is None:
            logger.warning("⚠️  Vector store not initialized")
            return []
        
        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # Filter by threshold
            filtered_results = [
                (doc, score) for doc, score in docs_with_scores
                if score <= (1 - score_threshold)
            ]
            
            logger.info(f"🔍 Found {len(filtered_results)} relevant documents")
            return filtered_results
            
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            if self.vectorstore:
                return {
                    "type": "FAISS",
                    "index_size": self.vectorstore.index.ntotal,
                    "local_path": "vectordb/faiss_index",
                    "embedding_model": self.embeddings.model_name if self.embeddings else "Unknown"
                }
            else:
                return {
                    "type": "Not initialized",
                    "index_size": 0,
                    "embedding_model": self.embeddings.model_name if self.embeddings else "Unknown"
                }
        except Exception as e:
            return {"type": "Error", "error": str(e)}


# Global instance
vector_store = VectorStoreManager()
