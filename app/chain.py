"""RAG chain implementation with local Hugging Face models."""

import logging
from typing import List, Dict, Any
import torch
from transformers import pipeline, AutoTokenizer
from langchain.schema import Document
from .config import settings, SYSTEM_PROMPT
from .vectorstore import vector_store

logger = logging.getLogger(__name__)


class RAGChain:
    """
    Retrieval-Augmented Generation chain with local Hugging Face models.
    
    Features:
    - Local text generation (no API calls)
    - Citation support
    - Robust error handling
    """
    
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Using device: {'GPU' if self.device == 0 else 'CPU'}")
        
        self.generator = self._initialize_generator()
        self.tokenizer = self._initialize_tokenizer()
    
    def _initialize_generator(self):
        """Initialize local text generation model."""
        try:
            logger.info(f"Loading model: {settings.local_llm_model}")
            
            # Use a text generation model instead of dialogue
            generator = pipeline(
                'text-generation',
                model='microsoft/DialoGPT-medium',  # Good for conversations
                tokenizer='microsoft/DialoGPT-medium',
                device=self.device,
                max_length=settings.max_tokens,
                do_sample=True,
                temperature=0.1,  # Low temperature for factual responses
                pad_token_id=50256  # GPT-2 pad token
            )
            
            logger.info("Text generation model loaded successfully")
            return generator
            
        except Exception as e:
            logger.error(f"Failed to initialize generator: {e}")
            logger.info("Falling back to a simpler model...")
            
            # Fallback to a smaller model
            return pipeline(
                'text-generation',
                model='gpt2',
                device=self.device,
                max_length=300
            )
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer for text processing."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(settings.local_llm_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {e}")
            return None
    
    def _format_context(self, documents: List[tuple]) -> str:
        """Format retrieved documents into context string."""
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, (doc, score) in enumerate(documents, 1):
            filename = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            
            context_part = f"""
Source {i} [{filename}:page {page}] (relevance: {1-score:.2f}):
{doc.page_content}
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _extract_sources(self, documents: List[tuple]) -> List[Dict[str, Any]]:
        """Extract source information for citations."""
        sources = []
        for doc, score in documents:
            source_info = {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "score": float(1 - score),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources.append(source_info)
        
        return sources
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response using local model."""
        try:
            # Truncate prompt if too long
            max_prompt_length = 400  # Leave room for response
            if len(prompt) > max_prompt_length:
                prompt = prompt[:max_prompt_length] + "..."
            
            # Generate response
            result = self.generator(
                prompt,
                max_new_tokens=200,
                num_return_sequences=1,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            
            # Extract generated text (remove the input prompt)
            generated_text = result[0]['generated_text']
            if generated_text.startswith(prompt):
                response = generated_text[len(prompt):].strip()
            else:
                response = generated_text.strip()
            
            # Clean up response
            response = response.split('\n')[0] if '\n' in response else response
            
            if not response:
                response = "I need more specific information to provide a detailed answer."
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an issue generating a response. Please try rephrasing your question."
    
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
                    "answer": "I couldn't find relevant information in the knowledge base to answer your question. Please try uploading relevant documents first.",
                    "sources": [],
                    "raw_llm_output": None,
                    "query_metadata": {
                        "retrieved_chunks": 0,
                        "model_used": settings.local_llm_model,
                        "local_model": True
                    }
                }
            
            # Step 2: Format context
            context = self._format_context(retrieved_docs)
            
            # Step 3: Create prompt
            prompt = SYSTEM_PROMPT.format(
                context=context[:800],  # Limit context length
                question=question
            )
            
            # Step 4: Generate response
            answer = self._generate_response(prompt)
            
            # Step 5: Extract sources
            sources = self._extract_sources(retrieved_docs)
            
            logger.info(f"Successfully processed query with {len(sources)} sources")
            
            return {
                "answer": answer,
                "sources": sources,
                "raw_llm_output": answer,
                "query_metadata": {
                    "retrieved_chunks": len(retrieved_docs),
                    "model_used": settings.local_llm_model,
                    "local_model": True,
                    "avg_relevance_score": sum(1-score for _, score in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": f"I encountered an error while processing your question. Please try again or check if documents are properly uploaded.",
                "sources": [],
                "raw_llm_output": None,
                "query_metadata": {
                    "error": str(e),
                    "retrieved_chunks": 0,
                    "model_used": settings.local_llm_model,
                    "local_model": True
                }
            }
    
    def query_sync(self, question: str) -> Dict[str, Any]:
        """Synchronous version of query method."""
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
            prompt = SYSTEM_PROMPT.format(context=context[:800], question=question)
            
            answer = self._generate_response(prompt)
            sources = self._extract_sources(retrieved_docs)
            
            return {
                "answer": answer,
                "sources": sources,
                "raw_llm_output": answer,
                "query_metadata": {
                    "retrieved_chunks": len(retrieved_docs),
                    "model_used": settings.local_llm_model,
                    "local_model": True
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
