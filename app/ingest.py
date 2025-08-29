import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pypdf import PdfReader

from .config import settings
from .vectorstore import vector_store

logger = logging.getLogger(__name__)


class DocumentIngester:
    """
    Document ingestion pipeline with robust PDF parsing and chunking.
    
    Supports:
    - PDF files with complex layouts
    - Plain text files  
    - HTML files (basic)
    - Metadata extraction and preservation
    """
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],  # Preserve document structure
            length_function=len
        )
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def extract_text_from_pdf(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        """
        Extract text from PDF with metadata and page tracking.
        
        Handles:
        - Multi-page documents
        - Complex layouts with columns
        - Metadata extraction (title, author, creation date)
        """
        try:
            reader = PdfReader(file_path)
            
            # Extract document metadata
            metadata = {
                "source": os.path.basename(file_path),
                "pages": len(reader.pages),
                "title": None,
                "author": None,
                "creation_date": None
            }
            
            # Try to extract PDF metadata
            if reader.metadata:
                metadata["title"] = getattr(reader.metadata, "title", None)
                metadata["author"] = getattr(reader.metadata, "author", None)
                metadata["creation_date"] = getattr(reader.metadata, "creation_date", None)
            
            # Extract text from all pages
            full_text = ""
            page_texts = []
            
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        page_texts.append((page_num, page_text))
                        full_text += f"\n\n--- Page {page_num} ---\n\n{page_text}"
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue
            
            if not full_text.strip():
                raise ValueError("No readable text found in PDF")
            
            logger.info(f"Extracted text from PDF: {len(page_texts)} pages, {len(full_text)} characters")
            return full_text, metadata
            
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise ValueError(f"Failed to process PDF: {str(e)}")
    
    def extract_text_from_txt(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        """Extract text from plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            metadata = {
                "source": os.path.basename(file_path),
                "pages": 1,
                "title": Path(file_path).stem,
                "file_size": os.path.getsize(file_path)
            }
            
            return content, metadata
            
        except Exception as e:
            logger.error(f"TXT extraction error: {e}")
            raise ValueError(f"Failed to process TXT file: {str(e)}")
    
    def extract_text_from_html(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        """Extract text from HTML file (basic implementation)."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Basic HTML tag removal (for production, use BeautifulSoup)
            import re
            text = re.sub(r'<[^>]+>', '', content)
            text = re.sub(r'\s+', ' ', text).strip()
            
            metadata = {
                "source": os.path.basename(file_path),
                "pages": 1,
                "title": Path(file_path).stem,
                "file_size": os.path.getsize(file_path)
            }
            
            return text, metadata
            
        except Exception as e:
            logger.error(f"HTML extraction error: {e}")
            raise ValueError(f"Failed to process HTML file: {str(e)}")
    
    def chunk_document(
        self, 
        text: str, 
        metadata: Dict[str, Any], 
        file_path: str
    ) -> List[Document]:
        """
        Split document into chunks with metadata preservation.
        
        Strategy:
        - Chunk size 1000: Balance between context and specificity
        - Overlap 150: Prevent concept fragmentation
        - Page tracking for citation purposes
        """
        try:
            chunks = self.text_splitter.split_text(text)
            
            documents = []
            for i, chunk in enumerate(chunks):
                # Estimate page number based on text position (rough approximation)
                char_position = text.find(chunk)
                estimated_page = min(
                    (char_position // 2000) + 1,  # ~2000 chars per page estimate
                    metadata.get("pages", 1)
                )
                
                chunk_metadata = {
                    **metadata,
                    "chunk_id": i,
                    "page": estimated_page,
                    "total_chunks": len(chunks),
                    "file_path": file_path,
                    "chunk_size": len(chunk)
                }
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                ))
            
            logger.info(f"Created {len(documents)} chunks from document")
            return documents
            
        except Exception as e:
            logger.error(f"Chunking error: {e}")
            raise
    
    async def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest a single file asynchronously.
        
        Returns ingestion results with statistics and status.
        """
        try:
            logger.info(f"Starting ingestion of: {file_path}")
            
            # Validate file exists and is readable
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_ext = Path(file_path).suffix.lower()
            
            # Extract text based on file type
            if file_ext == '.pdf':
                text, metadata = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self.extract_text_from_pdf, file_path
                )
            elif file_ext == '.txt':
                text, metadata = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self.extract_text_from_txt, file_path
                )
            elif file_ext in ['.html', '.htm']:
                text, metadata = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self.extract_text_from_html, file_path
                )
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            # Create document chunks
            documents = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.chunk_document, text, metadata, file_path
            )
            
            # Add to vector store
            document_ids = await asyncio.get_event_loop().run_in_executor(
                self.executor, vector_store.add_documents, documents
            )
            
            result = {
                "status": "success",
                "file_path": file_path,
                "filename": os.path.basename(file_path),
                "file_type": file_ext,
                "chunks_created": len(documents),
                "total_characters": len(text),
                "metadata": metadata,
                "processing_time": None  # Could add timing
            }
            
            logger.info(f"Successfully ingested {file_path}: {len(documents)} chunks")
            return result
            
        except Exception as e:
            logger.error(f"Ingestion failed for {file_path}: {e}")
            return {
                "status": "error",
                "file_path": file_path,
                "filename": os.path.basename(file_path) if file_path else "Unknown",
                "error": str(e),
                "chunks_created": 0
            }
    
    def ingest_file_sync(self, file_path: str) -> Dict[str, Any]:
        """Synchronous version for CLI and testing."""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            # Extract text
            if file_ext == '.pdf':
                text, metadata = self.extract_text_from_pdf(file_path)
            elif file_ext == '.txt':
                text, metadata = self.extract_text_from_txt(file_path)
            elif file_ext in ['.html', '.htm']:
                text, metadata = self.extract_text_from_html(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            # Create chunks and add to vector store
            documents = self.chunk_document(text, metadata, file_path)
            vector_store.add_documents(documents)
            
            return {
                "status": "success",
                "file_path": file_path,
                "filename": os.path.basename(file_path),
                "chunks_created": len(documents),
                "total_characters": len(text)
            }
            
        except Exception as e:
            logger.error(f"Synchronous ingestion failed: {e}")
            return {
                "status": "error",
                "file_path": file_path,
                "error": str(e)
            }


# Global ingester instance
ingester = DocumentIngester()


# CLI script functionality
def main():
    """CLI entry point for document ingestion."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Ingest documents into AI Research Copilot")
    parser.add_argument("files", nargs="+", help="Files to ingest")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    
    print("🚀 AI Research Copilot - Document Ingestion")
    print(f"📊 Vector store: {'Pinecone' if settings.use_pinecone else 'FAISS'}")
    print(f"📄 Processing {len(args.files)} files...")
    
    success_count = 0
    for file_path in args.files:
        try:
            result = ingester.ingest_file_sync(file_path)
            if result["status"] == "success":
                print(f"✅ {result['filename']}: {result['chunks_created']} chunks")
                success_count += 1
            else:
                print(f"❌ {result['filename']}: {result['error']}")
        except Exception as e:
            print(f"❌ {file_path}: {str(e)}")
    
    print(f"\n🎯 Completed: {success_count}/{len(args.files)} files processed successfully")
    
    if success_count > 0:
        stats = vector_store.get_stats()
        print(f"📈 Vector store stats: {stats}")


if __name__ == "__main__":
    main()

