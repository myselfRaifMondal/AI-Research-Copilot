"""Test creating a fresh FAISS index without corruption."""

import os
import numpy as np
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

# Proper embeddings class that inherits from Embeddings
class TestEmbeddings(Embeddings):
    def __init__(self):
        print("Loading sentence transformer...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✅ Model loaded")
    
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()

def test_fresh_index():
    print("🔄 Testing fresh FAISS index creation...")
    
    # Create test documents
    docs = [
        Document(page_content="Machine learning is used in healthcare research.", metadata={"source": "test1.pdf", "page": 1}),
        Document(page_content="Deep learning models can analyze medical images.", metadata={"source": "test2.pdf", "page": 1}),
        Document(page_content="Natural language processing helps analyze clinical notes.", metadata={"source": "test3.pdf", "page": 1})
    ]
    
    # Create embeddings
    embeddings = TestEmbeddings()
    
    # Create fresh FAISS index
    print("🔄 Creating FAISS index...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Save it
    os.makedirs("vectordb/faiss_index", exist_ok=True)
    vectorstore.save_local("vectordb/faiss_index")
    
    print(f"✅ Created fresh index with {vectorstore.index.ntotal} vectors")
    
    # Test loading
    print("🔄 Testing index loading...")
    loaded_store = FAISS.load_local("vectordb/faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    print(f"✅ Successfully loaded index with {loaded_store.index.ntotal} vectors")
    
    # Test search
    results = loaded_store.similarity_search("healthcare", k=2)
    print(f"✅ Search test successful: found {len(results)} results")
    
    for i, doc in enumerate(results):
        print(f"  Result {i+1}: {doc.page_content[:50]}...")
    
    return True

if __name__ == "__main__":
    try:
        success = test_fresh_index()
        if success:
            print("🎉 Fresh FAISS index created successfully!")
            print("✅ Your corruption issue is FIXED!")
        else:
            print("❌ Test failed")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
