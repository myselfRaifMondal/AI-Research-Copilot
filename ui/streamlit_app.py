import streamlit as st
import requests
import json
import os
from typing import Dict, Any, Optional
import time

# Page configuration
st.set_page_config(
    page_title="AI Research Copilot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "dev-api-key")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin: 0.5rem 0;
    }
    .query-stats {
        background-color: #EFF6FF;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        color: #1E40AF;
    }
</style>
""", unsafe_allow_html=True)


def make_api_request(endpoint: str, method: str = "GET", data: Dict = None, files: Dict = None) -> Optional[Dict]:
    """Make authenticated API request with error handling."""
    headers = {"Authorization": f"Bearer {API_KEY}"}
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "POST":
            if files:
                response = requests.post(url, headers=headers, files=files)
            else:
                headers["Content-Type"] = "application/json"
                response = requests.post(url, headers=headers, json=data)
        else:
            response = requests.get(url, headers=headers)
        
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json().get('detail', 'Unknown error')
                st.error(f"Error details: {error_detail}")
            except:
                pass
        return None


def display_sources(sources: list):
    """Display source citations in an organized format."""
    if not sources:
        return
    
    st.subheader("📚 Sources")
    
    for i, source in enumerate(sources, 1):
        with st.container():
            st.markdown(f"""
            <div class="source-box">
                <strong>Source {i}: {source.get('source', 'Unknown')}</strong> 
                (Page {source.get('page', 'N/A')}, Relevance: {source.get('score', 0):.2f})<br>
                <em>{source.get('content_preview', 'No preview available')}</em>
            </div>
            """, unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔬 AI Research Copilot</h1>', unsafe_allow_html=True)
    st.markdown("*Upload research papers and get AI-powered answers with citations*")
    
    # Sidebar for system info and configuration
    with st.sidebar:
        st.header("⚙️ System Info")
        
        # Health check
        health_data = make_api_request("/health")
        if health_data:
            st.success("✅ API Connected")
            if health_data.get("vector_store"):
                vs_info = health_data["vector_store"]
                st.info(f"📊 Vector DB: {vs_info.get('type', 'Unknown')}")
                if vs_info.get('index_size'):
                    st.info(f"📄 Documents: {vs_info.get('index_size', 0)}")
        else:
            st.error("❌ API Disconnected")
            st.stop()
        
        # System stats
        if st.button("🔄 Refresh Stats"):
            stats = make_api_request("/stats")
            if stats:
                st.json(stats)
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload Documents", "❓ Ask Questions", "📈 System Stats"])
    
    # Tab 1: Document Upload
    with tab1:
        st.header("📤 Upload Research Documents")
        st.markdown("Supported formats: PDF, TXT, HTML (max 50MB)")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'html', 'htm'],
            help="Upload research papers, articles, or documents to analyze"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.info(f"**File:** {uploaded_file.name} ({uploaded_file.size:,} bytes)")
            
            if st.button("🚀 Start Ingestion", type="primary"):
                with st.spinner("Uploading and processing document..."):
                    
                    # Prepare file for upload
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    
                    # Upload file
                    result = make_api_request("/ingest", method="POST", files=files)
                    
                    if result:
                        if result.get("status") == "accepted":
                            st.success("✅ File uploaded successfully!")
                            st.info(f"**Task ID:** {result.get('task_id')}")
                            
                            # Store task ID in session state for tracking
                            st.session_state.last_task_id = result.get('task_id')
                            
                            # Display file info
                            file_info = result.get('file_info', {})
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("File Size", f"{file_info.get('size_bytes', 0):,} bytes")
                            with col2:
                                st.metric("File Type", file_info.get('file_type', 'Unknown'))
                            with col3:
                                st.metric("Status", "Processing...")
                            
                            # Auto-refresh status
                            placeholder = st.empty()
                            for i in range(30):  # Check for 30 seconds
                                time.sleep(2)
                                status_result = make_api_request(f"/ingest/status/{result.get('task_id')}")
                                if status_result:
                                    status = status_result.get('status')
                                    with placeholder.container():
                                        if status == "completed":
                                            st.success("🎉 Document processed successfully!")
                                            if 'result' in status_result:
                                                ingestion_result = status_result['result']
                                                st.info(f"Created {ingestion_result.get('chunks_created', 0)} text chunks")
                                            break
                                        elif status == "failed":
                                            st.error(f"❌ Processing failed: {status_result.get('error', 'Unknown error')}")
                                            break
                                        else:
                                            progress = status_result.get('progress', 0)
                                            st.info(f"🔄 Processing... ({progress}%)")
                        else:
                            st.error("❌ Upload failed")
                    else:
                        st.error("❌ Failed to upload file")
        
        # Task status checker
        st.header("📋 Check Task Status")
        task_id = st.text_input(
            "Task ID", 
            value=st.session_state.get('last_task_id', ''),
            help="Enter task ID to check ingestion status"
        )
        
        if task_id and st.button("Check Status"):
            status_result = make_api_request(f"/ingest/status/{task_id}")
            if status_result:
                status = status_result.get('status')
                if status == "completed":
                    st.success("✅ Task completed successfully!")
                    if 'result' in status_result:
                        st.json(status_result['result'])
                elif status == "failed":
                    st.error(f"❌ Task failed: {status_result.get('error')}")
                else:
                    st.info(f"🔄 Status: {status} ({status_result.get('progress', 0)}%)")
            else:
                st.error("❌ Task not found")
    
    # Tab 2: Query Interface  
    with tab2:
        st.header("❓ Ask Research Questions")
        
        # Query input
        question = st.text_area(
            "Enter your research question:",
            height=100,
            placeholder="e.g., What are the main findings about machine learning in healthcare?",
            help="Ask questions about the documents you've uploaded"
        )
        
        # Query options
        col1, col2 = st.columns(2)
        with col1:
            max_sources = st.slider("Maximum sources", 1, 10, 5)
        with col2:
            if st.button("🔍 Ask Question", type="primary", disabled=not question.strip()):
                if question.strip():
                    with st.spinner("Searching knowledge base and generating answer..."):
                        # Query API
                        query_data = {
                            "question": question.strip(),
                            "max_sources": max_sources
                        }
                        
                        result = make_api_request("/query", method="POST", data=query_data)
                        
                        if result:
                            # Display answer
                            st.subheader("🤖 AI Answer")
                            st.markdown(result.get('answer', 'No answer generated'))
                            
                            # Display query metadata
                            metadata = result.get('query_metadata', {})
                            if metadata:
                                st.markdown(f"""
                                <div class="query-stats">
                                    <strong>Query Stats:</strong> 
                                    Retrieved {metadata.get('retrieved_chunks', 0)} chunks | 
                                    Model: {metadata.get('model_used', 'Unknown')} |
                                    Avg. Relevance: {metadata.get('avg_relevance_score', 0):.2f}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Display sources
                            sources = result.get('sources', [])
                            if sources:
                                display_sources(sources)
                            else:
                                st.warning("⚠️ No relevant sources found in the knowledge base")
                        else:
                            st.error("❌ Query failed")
        
        # Sample questions
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "What are the main research methodologies discussed?",
            "What are the key findings and conclusions?",
            "What are the limitations mentioned in the study?",
            "What future research directions are suggested?",
            "How does this research relate to previous work?"
        ]
        
        for sample_q in sample_questions:
            if st.button(f"📝 {sample_q}", key=f"sample_{hash(sample_q)}"):
                st.session_state.sample_question = sample_q
                st.experimental_rerun()
        
        # Auto-fill sample question
        if 'sample_question' in st.session_state:
            st.text_area(
                "Selected question:",
                value=st.session_state.sample_question,
                key="selected_question",
                height=60
            )
            if st.button("🔍 Ask This Question", type="primary"):
                # Process the selected question
                pass
    
    # Tab 3: System Statistics
    with tab3:
        st.header("📈 System Statistics")
        
        if st.button("🔄 Refresh Statistics"):
            stats = make_api_request("/stats")
            
            if stats:
                # Vector store stats
                st.subheader("🗄️ Vector Database")
                vs_stats = stats.get('vector_store', {})
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Database Type", vs_stats.get('type', 'Unknown'))
                with col2:
                    st.metric("Documents", vs_stats.get('index_size', 0))
                with col3:
                    st.metric("Index Status", "Active" if vs_stats.get('index_size', 0) > 0 else "Empty")
                
                # Task statistics
                st.subheader("⚙️ Processing Tasks")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Active Tasks", stats.get('active_tasks', 0))
                with col2:
                    st.metric("Completed", stats.get('completed_tasks', 0))
                with col3:
                    st.metric("Failed", stats.get('failed_tasks', 0))
                
                # System configuration
                st.subheader("⚙️ System Configuration")
                system_info = stats.get('system_info', {})
                
                config_data = {
                    "Debug Mode": system_info.get('debug_mode', False),
                    "Chunk Size": system_info.get('chunk_size', 'Unknown'),
                    "Retrieval K": system_info.get('retrieval_k', 'Unknown'),
                    "API Base URL": API_BASE_URL
                }
                
                st.json(config_data)
            else:
                st.error("❌ Failed to retrieve statistics")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🔬 **AI Research Copilot** | Built with LangChain, FastAPI, and Streamlit | "
        "[API Docs](http://localhost:8000/docs) | [GitHub](https://github.com/your-repo)"
    )


if __name__ == "__main__":
    main()

