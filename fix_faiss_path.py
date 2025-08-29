import os
import shutil
from app.vectorstore import LocalEmbeddings
from langchain_community.vectorstores import FAISS

print("🔄 Fixing FAISS path structure...")

# Check current structure
print("Current vectordb structure:")
for root, dirs, files in os.walk("vectordb"):
    level = root.replace("vectordb", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 2 * (level + 1)
    for file in files:
        print(f"{subindent}{file}")

try:
    # Initialize embeddings
    embeddings = LocalEmbeddings()
    print("✅ Embeddings loaded")
    
    # Try different path variations
    possible_paths = [
        "vectordb/faiss_index",
        "vectordb/faiss_index/",  
        "./vectordb/faiss_index",
        "vectordb/faiss_index/faiss_index"
    ]
    
    vectorstore = None
    working_path = None
    
    for path in possible_paths:
        try:
            print(f"Trying path: {path}")
            
            # Check if files exist at this path
            faiss_file = f"{path}.faiss"
            pkl_file = f"{path}.pkl"
            
            # Also check for files inside directory
            if os.path.isdir(path):
                files_in_dir = os.listdir(path)
                print(f"  Files in {path}: {files_in_dir}")
                
                # Look for index files in the directory
                for f in files_in_dir:
                    if f.endswith('.faiss'):
                        faiss_file = os.path.join(path, f.replace('.faiss', ''))
                        print(f"  Found FAISS file pattern: {faiss_file}")
                        break
            
            if os.path.exists(faiss_file) or (os.path.isdir(path) and any(f.endswith('.faiss') for f in os.listdir(path))):
                print(f"  Attempting to load from: {path}")
                vectorstore = FAISS.load_local(
                    path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                working_path = path
                print(f"✅ SUCCESS: Loaded from {path}")
                break
                
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    if vectorstore:
        print(f"✅ Vector store loaded: {vectorstore.index.ntotal} chunks")
        
        # Test search
        results = vectorstore.similarity_search("research", k=3)
        print(f"✅ Search test: {len(results)} results")
        
        # Re-save in the correct location
        correct_path = "vectordb/faiss_index"
        vectorstore.save_local(correct_path)
        print(f"✅ Re-saved to: {correct_path}")
        
        # Verify it can be loaded from the standard location
        test_load = FAISS.load_local(
            correct_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✅ Verification: Can load from standard path: {test_load.index.ntotal} chunks")
        
    else:
        print("❌ Could not load vector store from any path")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
