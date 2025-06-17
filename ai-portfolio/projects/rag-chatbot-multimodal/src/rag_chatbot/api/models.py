"""Pydantic models for API requests and responses."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    include_sources: bool = Field(True, description="Whether to include source documents")
    max_results: int = Field(5, ge=1, le=20, description="Maximum number of results to retrieve")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters for retrieval")


class SourceDocument(BaseModel):
    """Source document information."""

    id: str = Field(..., description="Document chunk ID")
    content: str = Field(..., description="Document content")
    document_name: str = Field(..., description="Source document name")
    page_number: int = Field(..., description="Page number")
    similarity_score: Optional[float] = Field(None, description="Similarity score")
    content_type: str = Field(..., description="Type of content")


class SourceInfo(BaseModel):
    """Simplified source information for compatibility."""

    file_name: str = Field(..., description="Source file name")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    content_preview: str = Field(..., description="Content preview")
    document_type: Optional[str] = Field(None, description="Document type")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    
    message: str = Field(..., description="Generated response")
    conversation_id: str = Field(..., description="Conversation ID")
    sources: List[SourceDocument] = Field(default_factory=list, description="Source documents")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class UploadRequest(BaseModel):
    """Request model for document upload."""
    
    process_immediately: bool = Field(True, description="Whether to process the document immediately")
    extract_images: bool = Field(True, description="Whether to extract images from PDFs")
    chunk_strategy: str = Field("by_title", description="Chunking strategy")


class UploadResponse(BaseModel):
    """Response model for document upload."""
    
    document_id: str = Field(..., description="Unique document ID")
    filename: str = Field(..., description="Original filename")
    file_size_mb: float = Field(..., description="File size in MB")
    status: str = Field(..., description="Processing status")
    chunks_created: int = Field(..., description="Number of chunks created")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Upload timestamp")


class DocumentStatus(BaseModel):
    """Document processing status."""
    
    document_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Processing status")
    chunks_created: int = Field(..., description="Number of chunks created")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    
    documents: List[DocumentStatus] = Field(..., description="List of documents")
    total: int = Field(..., description="Total number of documents")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Page size")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: str = Field(..., description="Application version")
    vector_store_connected: bool = Field(..., description="Vector store connection status")
    total_documents: int = Field(..., description="Total documents in vector store")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking") 