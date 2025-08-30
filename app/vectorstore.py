"""Vector database abstraction - CORRUPTION-RESISTANT VERSION."""

import os
import logging
from typing import List, Tuple, Optional, Any, Dict
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

from .config import settings

logger = logging.getLogger(__name__)


class CorruptionResistantEmbeddings(Embeddings):
    """Corruption-resistant local embeddings."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        logger.info(f"Embeddings configured: {model_name}")
    
    def _ensure_loaded(self):
        """Load model with error handling."""
        if self._model is None:
            try:
                logger.info(f"Loading SentenceTransformer: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                logger.info("✅ SentenceTransformer loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
                raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with error handling."""
        if not texts:
            return []
        
        self._ensure_loaded()
        try:
            embeddings = self._model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"❌ Embedding error: {e}")
            # Return zero vectors as fallback to prevent corruption
            return [[0.0] * 384 for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query with error handling."""
        self._ensure_loaded()
        try:
            embedding = self._model.encode([text], convert_to_tensor=False, show_progress_bar=False)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"❌ Query embedding error: {e}")
            return [0.0] * 384


class VectorStoreManager:
    """Corruption-resistant vector store manager."""
    
    def __init__(self):
        logger.info("🔄 Initializing VectorStoreManager...")
        self.embeddings = CorruptionResistantEmbeddings()
        self.vectorstore = None
        self._safe_load_existing_index()
    
    def _safe_load_existing_index(self):
        """Safely load existing index or create new one."""
        faiss_dir = "vectordb/faiss_index"
        
        if not os.path.exists(faiss_dir):
            logger.info("📁 No existing index directory found")
            return
        
        faiss_file = os.path.join(faiss_dir, "index.faiss")
        pkl_file = os.path.join(faiss_dir, "index.pkl")
        
        if not (os.path.exists(faiss_file) and os.path.exists(pkl_file)):
            logger.info("📄 FAISS files not found")
            return
        
        # Check file sizes
        faiss_size = os.path.getsize(faiss_file)
        pkl_size = os.path.getsize(pkl_file)
        
        logger.info(f"📊 File sizes: FAISS={faiss_size:,}, PKL={pkl_size:,}")
        
        if faiss_size < 1000 or pkl_size < 1000:
            logger.warning("⚠️  Files too small, likely corrupted")
            self._remove_corrupted_files(faiss_dir)
            return
        
        # Attempt to load with timeout protection
        try:
            logger.info("🔄 Attempting to load FAISS index...")
            
            import signal
            def timeout_handler(signum, frame):
                raise TimeoutError("FAISS loading timed out")
            
            # Set 30-second timeout for loading
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
            
            try:
                self.vectorstore = FAISS.load_local(
                    faiss_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                signal.alarm(0)  # Cancel timeout
                
                count = self.vectorstore.index.ntotal
                logger.info(f"✅ Successfully loaded FAISS index: {count:,} vectors")
                
            except Exception as load_error:
                signal.alarm(0)  # Cancel timeout
                raise load_error
                
        except (RecursionError, TimeoutError, Exception) as e:
            logger.error(f"❌ Failed to load index: {e}")
            logger.info("🔧 Removing corrupted files...")
            self._remove_corrupted_files(faiss_dir)
            self.vectorstore = None
    
    def _remove_corrupted_files(self, faiss_dir: str):
        """Remove corrupted FAISS files."""
        try:
            import shutil
            if os.path.exists(faiss_dir):
                shutil.rmtree(faiss_dir)
                logger.info("🗑️  Removed corrupted FAISS directory")
        except Exception as e:
            logger.error(f"Failed to remove corrupted files: {e}")
    
    def force_recreate_index(self):
        """Force recreation of the index."""
        logger.info("🔄 Forcing index recreation...")
        self._remove_corrupted_files("vectordb/faiss_index")
        self.vectorstore = None
        logger.info("✅ Ready for fresh index creation")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents with corruption protection."""
        if not documents:
            return []
        
        try:
            if self.vectorstore is None:
                logger.info(f"🔄 Creating fresh FAISS index with {len(documents)} documents...")
                
                # Create with error handling
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
                
                # Save with corruption protection
                os.makedirs("vectordb/faiss_index", exist_ok=True)
                self.vectorstore.save_local("vectordb/faiss_index")
                
                # Verify save was successful
                if os.path.exists("vectordb/faiss_index/index.faiss"):
                    count = self.vectorstore.index.ntotal
                    logger.info(f"✅ Created and saved fresh index: {count:,} vectors")
                else:
                    raise Exception("Failed to save FAISS index")
            else:
                logger.info(f"➕ Adding {len(documents)} documents to existing index...")
                
                old_count = self.vectorstore.index.ntotal
                self.vectorstore.add_documents(documents)
                self.vectorstore.save_local("vectordb/faiss_index")
                
                new_count = self.vectorstore.index.ntotal
                logger.info(f"✅ Updated index: {old_count:,} -> {new_count:,} vectors")
            
            return [doc.metadata.get("id", "") for doc in documents]
            
        except Exception as e:
            logger.error(f"❌ Error adding documents: {e}")
            # If corruption occurs during save, clean up
            self._remove_corrupted_files("vectordb/faiss_index")
            self.vectorstore = None
            raise
    
    def similarity_search(self, query: str, k: int = 5, score_threshold: float = 0.7) -> List[Tuple[Document, float]]:
        """Search with error handling."""
        if self.vectorstore is None:
            logger.warning("⚠️  Vector store not initialized")
            return []
        
        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
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
        """Get statistics with error handling."""
        try:
            if self.vectorstore:
                return {
                    "type": "FAISS",
                    "index_size": self.vectorstore.index.ntotal,
                    "local_path": "vectordb/faiss_index",
                    "embedding_model": self.embeddings.model_name
                }
            else:
                return {
                    "type": "Not initialized",
                    "index_size": 0,
                    "embedding_model": self.embeddings.model_name
                }
        except Exception as e:
            return {"type": "Error", "error": str(e)}


# Global instance
vector_store = VectorStoreManager()
