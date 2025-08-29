import logging
from typing import List, Dict, Any
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.callbacks import StdOutCallbackHandler
from .config import settings, SYSTEM_PROMPT
from .vectorstore import vector_store

logger = logging.getLogger(__name__)


class RAGChain:
    """
    Retrieval-Augmented Generation chain with citation support.
    
    Features:
    - Context-aware retrieval with score thresholding
    - Citation extraction and formatting
    - Robust error handling and fallback mechanisms
    """
    
    def __init__(self):
        self.llm = self._initialize_llm()
        self.prompt_template = self._create_prompt_template()
    
    def _initialize_llm(self) -> ChatOpenAI:
        """
        Initialize LLM with Azure OpenAI fallback support.
        
        Production notes:
        - Use gpt-3.5-turbo for cost efficiency vs gpt-4 for quality
        - Implement model routing based on query complexity
        - Add timeout and retry logic for production resilience
        """
        try:
            if settings.azure_openai_api_key and settings.azure_openai_endpoint:
                # Azure OpenAI configuration
                return ChatOpenAI(
                    openai_api_type="azure",
                    azure_endpoint=settings.azure_openai_endpoint,
                    openai_api_key=settings.azure_openai_api_key,
                    openai_api_version=settings.azure_openai_api_version,
                    model_name="gpt-35-turbo",  # Azure model name format
                    temperature=0.1,  # Low temperature for factual accuracy
                    max_tokens=1000,
                    request_timeout=30
                )
            else:
                # Standard OpenAI configuration
                return ChatOpenAI(
                    model="gpt-3.5-turbo",  # Cost-effective choice
                    temperature=0.1,
                    max_tokens=1000,
                    openai_api_key=settings.openai_api_key,
                    request_timeout=30
                )
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create prompt template with citation enforcement."""
        return PromptTemplate(
            template=SYSTEM_PROMPT,
            input_variables=["context", "question"]
        )
    
    def _format_context(self, documents: List[tuple]) -> str:
        """
        Format retrieved documents into context string with source tracking.
        
        Args:
            documents: List of (Document, score) tuples
        """
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, (doc, score) in enumerate(documents, 1):
            # Extract metadata for citations
            filename = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            
            # Format each document chunk
            context_part = f"""
Source {i} [{filename}:page {page}] (relevance: {1-score:.2f}):
{doc.page_content}
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _extract_sources(self, documents: List[tuple]) -> List[Dict[str, Any]]:
        """Extract source information for citation."""
        sources = []
        for doc, score in documents:
            source_info = {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "score": float(1 - score),  # Convert to similarity score
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources.append(source_info)
        
        return sources
    
    async def query(self, question: str) -> Dict[str, Any]:
        """
        Process a query through the RAG chain.
        
        Returns:
            Dict with answer, sources, and metadata
        """
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # Step 1: Retrieve relevant documents
            retrieved_docs = vector_store.similarity_search(
                query=question,
                k=settings.retrieval_k,
                score_threshold=0.7
            )
            
            if not retrieved_docs:
                return {
                    "answer": "I couldn't find relevant information in the knowledge base to answer your question.",
                    "sources": [],
                    "raw_llm_output": None,
                    "query_metadata": {
                        "retrieved_chunks": 0,
                        "model_used": self.llm.model_name
                    }
                }
            
            # Step 2: Format context and create prompt
            context = self._format_context(retrieved_docs)
            
            # Step 3: Generate response using LLM
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            
            llm_response = await self.llm.ainvoke(prompt)
            answer = llm_response.content
            
            # Step 4: Extract and format sources
            sources = self._extract_sources(retrieved_docs)
            
            logger.info(f"Successfully processed query with {len(sources)} sources")
            
            return {
                "answer": answer,
                "sources": sources,
                "raw_llm_output": answer,
                "query_metadata": {
                    "retrieved_chunks": len(retrieved_docs),
                    "model_used": self.llm.model_name,
                    "avg_relevance_score": sum(1-score for _, score in retrieved_docs) / len(retrieved_docs)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "raw_llm_output": None,
                "query_metadata": {
                    "error": str(e),
                    "retrieved_chunks": 0,
                    "model_used": getattr(self.llm, 'model_name', 'unknown')
                }
            }
    
    def query_sync(self, question: str) -> Dict[str, Any]:
        """Synchronous version of query method for non-async contexts."""
        try:
            # Retrieve documents
            retrieved_docs = vector_store.similarity_search(
                query=question,
                k=settings.retrieval_k
            )
            
            if not retrieved_docs:
                return {
                    "answer": "I couldn't find relevant information in the knowledge base to answer your question.",
                    "sources": [],
                    "raw_llm_output": None
                }
            
            # Format context and generate response
            context = self._format_context(retrieved_docs)
            prompt = self.prompt_template.format(context=context, question=question)
            
            llm_response = self.llm.invoke(prompt)
            answer = llm_response.content
            sources = self._extract_sources(retrieved_docs)
            
            return {
                "answer": answer,
                "sources": sources,
                "raw_llm_output": answer,
                "query_metadata": {
                    "retrieved_chunks": len(retrieved_docs),
                    "model_used": self.llm.model_name
                }
            }
            
        except Exception as e:
            logger.error(f"Error in synchronous query: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "raw_llm_output": None
            }


# Global RAG chain instance
rag_chain = RAGChain()

