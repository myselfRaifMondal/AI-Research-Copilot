import os
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import numpy as np

class SimpleLocalEmbeddings(Embeddings):
    def __init__(self):
        print("Loading sentence transformer...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("Model loaded successfully")
    
    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text):
        embedding = self.model.encode([text])
        return embedding[0].tolist()

try:
    print("🔄 Testing FAISS loading with simple embeddings...")
    
    # Check if files exist in correct location
    if os.path.exists("vectordb/faiss_index.faiss") and os.path.exists("vectordb/faiss_index.pkl"):
        print("✅ FAISS files found in correct location")
        
        # Initialize embeddings without recursion
        embeddings = SimpleLocalEmbeddings()
        
        # Load FAISS index
        vectorstore = FAISS.load_local(
            "vectordb/faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        print(f"✅ SUCCESS: Loaded {vectorstore.index.ntotal} document chunks!")
        
        # Test search
        results = vectorstore.similarity_search("research methodology", k=3)
        print(f"✅ Search test successful: {len(results)} results")
        
        for i, result in enumerate(results):
            print(f"  Result {i+1}: {result.metadata.get('source', 'Unknown')} - {result.page_content[:100]}...")
            
    else:
        print("❌ FAISS files not found in expected location")
        print("Current vectordb contents:")
        if os.path.exists("vectordb"):
            for item in os.listdir("vectordb"):
                print(f"  {item}")
                
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
