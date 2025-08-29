import os
import logging
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import settings
from .auth import verify_api_key, rate_limiter
from .ingest import ingester
from .chain import rag_chain
from .vectorstore import vector_store

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="AI Research Copilot",
    description="RAG-powered research assistant for academic papers and documents",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,  # Disable in production
    redoc_url="/redoc" if settings.debug else None
)

# CORS middleware for development
if settings.debug:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501"],  # Streamlit default port
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Query request model."""
    question: str = Field(..., min_length=1, max_length=1000, description="Research question")
    max_sources: Optional[int] = Field(default=5, ge=1, le=10, description="Maximum number of sources")


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    sources: List[Dict[str, Any]]
    query_metadata: Dict[str, Any]
    timestamp: str


class IngestionResponse(BaseModel):
    """Ingestion response model."""
    status: str
    message: str
    file_info: Dict[str, Any]
    task_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    version: str
    vector_store: Dict[str, Any]


# Background task storage (in production, use Redis or database)
ingestion_tasks = {}


async def background_ingestion(task_id: str, file_path: str, filename: str):
    """Background task for file ingestion."""
    try:
        logger.info(f"Starting background ingestion: {task_id}")
        ingestion_tasks[task_id] = {"status": "processing", "progress": 0}
        
        result = await ingester.ingest_file(file_path)
        
        ingestion_tasks[task_id] = {
            "status": "completed" if result["status"] == "success" else "failed",
            "result": result,
            "progress": 100,
            "completed_at": datetime.utcnow().isoformat()
        }
        
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
            
        logger.info(f"Background ingestion completed: {task_id}")
        
    except Exception as e:
        logger.error(f"Background ingestion failed: {task_id}: {e}")
        ingestion_tasks[task_id] = {
            "status": "failed",
            "error": str(e),
            "progress": 0,
            "completed_at": datetime.utcnow().isoformat()
        }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns system status and vector store information.
    """
    try:
        vector_stats = vector_store.get_stats()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            vector_store=vector_stats
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/ingest", response_model=IngestionResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Document ingestion endpoint with background processing.
    
    Supports PDF, TXT, and HTML files up to 50MB.
    Returns task ID for tracking ingestion progress.
    """
    try:
        # Rate limiting check
        if not rate_limiter.is_allowed(api_key, limit=10, window=3600):  # 10 uploads per hour
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Validate file type and size
        allowed_types = {".pdf", ".txt", ".html", ".htm"}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_types}"
            )
        
        # Check file size (50MB limit)
        max_size = 50 * 1024 * 1024  # 50MB
        content = await file.read()
        if len(content) > max_size:
            raise HTTPException(status_code=400, detail="File too large (max 50MB)")
        
        # Save uploaded file temporarily
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Generate task ID and start background processing
        task_id = f"ingest_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{len(ingestion_tasks)}"
        
        background_tasks.add_task(
            background_ingestion,
            task_id=task_id,
            file_path=file_path,
            filename=file.filename
        )
        
        logger.info(f"Started ingestion task: {task_id} for file: {file.filename}")
        
        return IngestionResponse(
            status="accepted",
            message="File uploaded successfully. Ingestion started in background.",
            file_info={
                "filename": file.filename,
                "size_bytes": len(content),
                "file_type": file_ext
            },
            task_id=task_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/ingest/status/{task_id}")
async def get_ingestion_status(
    task_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get status of background ingestion task."""
    if task_id not in ingestion_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return ingestion_tasks[task_id]


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Query endpoint for RAG-powered question answering.
    
    Retrieves relevant documents and generates contextual answers with citations.
    """
    try:
        # Rate limiting check
        if not rate_limiter.is_allowed(api_key, limit=100, window=3600):  # 100 queries per hour
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        logger.info(f"Processing query: {request.question[:100]}...")
        
        # Process query through RAG chain
        result = await rag_chain.query(request.question)
        
        # Limit sources if requested
        sources = result["sources"][:request.max_sources] if request.max_sources else result["sources"]
        
        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            query_metadata=result.get("query_metadata", {}),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/stats")
async def get_system_stats(api_key: str = Depends(verify_api_key)):
    """Get system statistics (admin endpoint)."""
    try:
        vector_stats = vector_store.get_stats()
        
        return {
            "vector_store": vector_stats,
            "active_tasks": len(ingestion_tasks),
            "completed_tasks": len([t for t in ingestion_tasks.values() if t.get("status") == "completed"]),
            "failed_tasks": len([t for t in ingestion_tasks.values() if t.get("status") == "failed"]),
            "system_info": {
                "debug_mode": settings.debug,
                "chunk_size": settings.chunk_size,
                "retrieval_k": settings.retrieval_k
            }
        }
    except Exception as e:
        logger.error(f"Stats endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc) if settings.debug else "Server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    # For development
    uvicorn.run(
        "app.server:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )

