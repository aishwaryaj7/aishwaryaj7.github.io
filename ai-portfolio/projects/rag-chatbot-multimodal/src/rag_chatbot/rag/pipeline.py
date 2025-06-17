"""Main RAG pipeline orchestrating document processing and querying."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from ..core.config import settings
from ..core.logger import get_logger
from ..document_processing.extractor import MultiModalDocumentExtractor
from .vector_store import WeaviateVectorStore

logger = get_logger(__name__)


class RAGPipeline:
    """Production-grade RAG pipeline with multi-modal document support."""
    
    def __init__(self):
        """Initialize the RAG pipeline."""
        self.vector_store: Optional[WeaviateVectorStore] = None
        self.document_extractor: Optional[MultiModalDocumentExtractor] = None
        self.llm: Optional[ChatOpenAI] = None
        self.is_initialized = False
        
        # RAG prompt template
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert assistant for analyzing invoices, contracts, and business documents. 
Your task is to provide accurate, helpful answers based on the provided context.

Guidelines:
- Use ONLY the information provided in the context
- If the answer isn't in the context, say "I don't have enough information to answer that question"
- Be specific and cite relevant details from the documents
- For financial information, be precise with numbers and dates
- When referring to documents, mention the document name and page number if available

Context from documents:
{context}"""),
            ("human", "{question}")
        ])
    
    async def initialize(self) -> None:
        """Initialize all components of the RAG pipeline."""
        try:
            logger.info("Initializing RAG pipeline")
            
            # Validate OpenAI API key
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key is required")
            
            # Initialize components
            self.vector_store = WeaviateVectorStore()
            await self.vector_store.connect()
            
            self.document_extractor = MultiModalDocumentExtractor()
            
            self.llm = ChatOpenAI(
                model=settings.openai_model,
                temperature=settings.openai_temperature,
                api_key=settings.openai_api_key,
                max_tokens=1000
            )
            
            self.is_initialized = True
            logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
    async def process_document(
        self, 
        file_path: Path, 
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a document through the RAG pipeline.
        
        Args:
            file_path: Path to the document file
            document_id: Unique identifier for the document
            metadata: Optional additional metadata
            
        Returns:
            Processing results with statistics
        """
        if not self.is_initialized:
            raise RuntimeError("RAG pipeline not initialized")
        
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Processing document {file_path} with ID {document_id}")
            
            # Extract content from document
            extracted_content = await self.document_extractor.extract_content(file_path)
            
            # Prepare chunks for vector store
            chunks = await self._prepare_chunks(extracted_content, document_id, metadata)
            
            # Add to vector store
            chunk_ids = await self.vector_store.add_documents(chunks, document_id)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            result = {
                "document_id": document_id,
                "file_path": str(file_path),
                "chunks_created": len(chunk_ids),
                "processing_time_seconds": processing_time,
                "content_types": self._get_content_type_stats(chunks),
                "status": "success"
            }
            
            logger.info(
                f"Successfully processed document {document_id}",
                chunks_created=len(chunk_ids),
                processing_time_seconds=processing_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process document {document_id}: {e}")
            return {
                "document_id": document_id,
                "file_path": str(file_path),
                "chunks_created": 0,
                "status": "failed",
                "error": str(e)
            }
    
    async def query(
        self,
        query: str,
        max_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG pipeline for information.
        
        Args:
            query: User query
            max_results: Maximum number of results to retrieve
            filters: Optional filters for retrieval
            include_sources: Whether to include source documents
            
        Returns:
            Response with answer and sources
        """
        if not self.is_initialized:
            raise RuntimeError("RAG pipeline not initialized")
        
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # Retrieve relevant documents
            search_results = await self.vector_store.search(
                query=query,
                limit=max_results,
                filters=filters,
                include_similarity=True
            )
            
            if not search_results:
                return {
                    "answer": "I don't have any relevant information to answer your question. Please try rephrasing or upload relevant documents.",
                    "sources": [],
                    "query": query
                }
            
            # Prepare context from retrieved documents
            context = self._format_context(search_results)
            
            # Generate response using LLM
            chain = (
                {"context": lambda x: context, "question": RunnablePassthrough()}
                | self.rag_prompt
                | self.llm
                | StrOutputParser()
            )
            
            answer = await chain.ainvoke(query)
            
            # Format sources if requested
            sources = []
            if include_sources:
                sources = [
                    {
                        "id": result["id"],
                        "content": result["content"][:500] + "..." if len(result["content"]) > 500 else result["content"],
                        "document_name": result["document_name"],
                        "page_number": result["page_number"],
                        "similarity_score": result.get("similarity_score"),
                        "content_type": result["content_type"]
                    }
                    for result in search_results
                ]
            
            logger.info(
                f"Query processed successfully",
                sources_found=len(search_results),
                answer_length=len(answer)
            )
            
            return {
                "answer": answer,
                "sources": sources,
                "query": query,
                "total_sources": len(search_results)
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise
    
    async def _prepare_chunks(
        self, 
        extracted_content: Dict[str, Any], 
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Prepare document chunks for vector store insertion."""
        
        chunks = []
        base_metadata = metadata or {}
        
        # Process text content
        for i, text_item in enumerate(extracted_content.get("text_content", [])):
            chunk = {
                "content": text_item["content"],
                "content_type": "text",
                "document_name": extracted_content.get("file_name", ""),
                "page_number": text_item.get("page_number", 0),
                "chunk_index": i,
                "file_path": extracted_content.get("file_path", ""),
                "extraction_timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    **base_metadata,
                    "category": text_item.get("category", "Text"),
                    "extraction_metadata": text_item.get("metadata", {})
                }
            }
            chunks.append(chunk)
        
        # Process tables
        for i, table_item in enumerate(extracted_content.get("tables", [])):
            chunk = {
                "content": table_item["content"],
                "content_type": "table",
                "document_name": extracted_content.get("file_name", ""),
                "page_number": table_item.get("page_number", 0),
                "chunk_index": len(chunks) + i,
                "file_path": extracted_content.get("file_path", ""),
                "extraction_timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    **base_metadata,
                    "extraction_metadata": table_item.get("metadata", {})
                }
            }
            chunks.append(chunk)
        
        # Process images (OCR text)
        for i, image_item in enumerate(extracted_content.get("images", [])):
            if image_item.get("content"):  # Only if OCR extracted text
                chunk = {
                    "content": image_item["content"],
                    "content_type": "image_text",
                    "document_name": extracted_content.get("file_name", ""),
                    "page_number": image_item.get("page_number", 0),
                    "chunk_index": len(chunks) + i,
                    "file_path": extracted_content.get("file_path", ""),
                    "extraction_timestamp": datetime.utcnow().isoformat(),
                    "metadata": {
                        **base_metadata,
                        "extraction_metadata": image_item.get("metadata", {})
                    }
                }
                chunks.append(chunk)
        
        logger.info(f"Prepared {len(chunks)} chunks for document {document_id}")
        return chunks
    
    def _format_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results into context for the LLM."""
        
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            document_info = f"Document: {result['document_name']}"
            if result['page_number']:
                document_info += f", Page {result['page_number']}"
            if result.get('similarity_score'):
                document_info += f" (Relevance: {result['similarity_score']:.2f})"
            
            context_part = f"[Source {i}] {document_info}\nContent: {result['content']}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _get_content_type_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get statistics on content types in chunks."""
        
        stats = {}
        for chunk in chunks:
            content_type = chunk.get("content_type", "unknown")
            stats[content_type] = stats.get(content_type, 0) + 1
        
        return stats
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.vector_store:
                await self.vector_store.disconnect()
            
            self.is_initialized = False
            logger.info("RAG pipeline cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        
        stats = {
            "is_initialized": self.is_initialized,
            "vector_store_connected": False,
            "total_documents": 0,
            "supported_file_types": []
        }
        
        if self.vector_store and self.vector_store.is_connected:
            try:
                vector_stats = await self.vector_store.get_collection_stats()
                stats.update({
                    "vector_store_connected": True,
                    "total_documents": vector_stats["total_objects"],
                    "content_type_distribution": vector_stats["content_type_distribution"]
                })
            except Exception as e:
                logger.warning(f"Failed to get vector store stats: {e}")
        
        if self.document_extractor:
            stats["supported_file_types"] = self.document_extractor.get_supported_formats()
        
        return stats 