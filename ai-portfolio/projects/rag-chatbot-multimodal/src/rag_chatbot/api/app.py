"""Main FastAPI application with production-grade monitoring and observability."""

import time
from contextlib import asynccontextmanager
from typing import Any, Dict
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from ..core.config import settings
from ..core.logger import setup_logging, get_logger
from ..rag.pipeline import RAGPipeline
from .models import (
    ChatRequest, ChatResponse, UploadResponse, 
    HealthResponse, ErrorResponse, DocumentListResponse
)

# Setup logging and monitoring
setup_logging()
logger = get_logger(__name__)

# Initialize Sentry for error tracking
if settings.debug:
    sentry_sdk.init(
        integrations=[FastApiIntegration()],
        traces_sample_rate=0.1,
    )

# Global RAG pipeline instance
rag_pipeline: RAGPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global rag_pipeline
    
    # Startup
    logger.info("Starting RAG Chatbot API")
    try:
        # Create data directories
        settings.create_directories()
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline()
        await rag_pipeline.initialize()
        
        logger.info("RAG Chatbot API started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down RAG Chatbot API")
    if rag_pipeline:
        await rag_pipeline.cleanup()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="RAG Chatbot API",
        description="Production-grade multi-modal RAG chatbot for invoice and contract processing",
        version=settings.app_version,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = str(uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    
    # Exception handlers
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", request_id=getattr(request.state, 'request_id', None))
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                error_code="INTERNAL_ERROR",
                request_id=getattr(request.state, 'request_id', None)
            ).dict()
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.detail,
                error_code=f"HTTP_{exc.status_code}",
                request_id=getattr(request.state, 'request_id', None)
            ).dict()
        )
    
    # Routes
    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Root endpoint."""
        return {"message": "RAG Chatbot API", "version": settings.app_version}
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        try:
            # Check vector store connection
            vector_store_connected = False
            total_documents = 0
            
            if rag_pipeline and rag_pipeline.vector_store:
                try:
                    stats = await rag_pipeline.vector_store.get_collection_stats()
                    vector_store_connected = stats["is_connected"]
                    total_documents = stats["total_objects"]
                except Exception as e:
                    logger.warning(f"Failed to get vector store stats: {e}")
            
            return HealthResponse(
                status="healthy",
                version=settings.app_version,
                vector_store_connected=vector_store_connected,
                total_documents=total_documents
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=503, detail="Service unavailable")
    
    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest, http_request: Request):
        """Chat endpoint for RAG queries."""
        start_time = time.time()
        
        try:
            if not rag_pipeline:
                raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
            
            # Generate conversation ID if not provided
            conversation_id = request.conversation_id or str(uuid4())
            
            # Process the query
            result = await rag_pipeline.query(
                query=request.message,
                max_results=request.max_results,
                filters=request.filters,
                include_sources=request.include_sources
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Log the interaction
            logger.info(
                "Chat query processed",
                conversation_id=conversation_id,
                query_length=len(request.message),
                results_count=len(result.get("sources", [])),
                processing_time_ms=processing_time,
                request_id=getattr(http_request.state, 'request_id', None)
            )
            
            return ChatResponse(
                message=result["answer"],
                conversation_id=conversation_id,
                sources=result.get("sources", []),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Chat query failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/upload", response_model=UploadResponse)
    async def upload_document(
        file: UploadFile = File(...),
        http_request: Request = None
    ):
        """Upload and process a document."""
        start_time = time.time()
        
        try:
            if not rag_pipeline:
                raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
            
            # Validate file
            if not file.filename:
                raise HTTPException(status_code=400, detail="No filename provided")
            
            file_extension = "." + file.filename.split(".")[-1].lower()
            if file_extension not in settings.allowed_file_types:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File type {file_extension} not supported"
                )
            
            # Check file size
            file_content = await file.read()
            file_size_mb = len(file_content) / (1024 * 1024)
            
            if file_size_mb > settings.max_file_size_mb:
                raise HTTPException(
                    status_code=400,
                    detail=f"File size ({file_size_mb:.2f} MB) exceeds limit ({settings.max_file_size_mb} MB)"
                )
            
            # Process the document
            document_id = str(uuid4())
            
            # Save file temporarily
            temp_file_path = settings.upload_dir / f"{document_id}_{file.filename}"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_content)
            
            try:
                # Process with RAG pipeline
                result = await rag_pipeline.process_document(
                    file_path=temp_file_path,
                    document_id=document_id
                )
                
                processing_time = (time.time() - start_time) * 1000
                
                # Log the upload
                logger.info(
                    "Document uploaded and processed",
                    document_id=document_id,
                    filename=file.filename,
                    file_size_mb=file_size_mb,
                    chunks_created=result["chunks_created"],
                    processing_time_ms=processing_time,
                    request_id=getattr(http_request.state, 'request_id', None)
                )
                
                return UploadResponse(
                    document_id=document_id,
                    filename=file.filename,
                    file_size_mb=file_size_mb,
                    status="processed",
                    chunks_created=result["chunks_created"],
                    processing_time_ms=processing_time
                )
                
            finally:
                # Clean up temporary file
                if temp_file_path.exists():
                    temp_file_path.unlink()
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Document upload failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/documents", response_model=DocumentListResponse)
    async def list_documents(page: int = 1, page_size: int = 10):
        """List uploaded documents."""
        try:
            if not rag_pipeline:
                raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
            
            # This would typically come from a database
            # For now, return mock data
            return DocumentListResponse(
                documents=[],
                total=0,
                page=page,
                page_size=page_size
            )
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/documents/{document_id}")
    async def delete_document(document_id: str):
        """Delete a document and its chunks."""
        try:
            if not rag_pipeline:
                raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
            
            deleted_count = await rag_pipeline.vector_store.delete_document(document_id)
            
            logger.info(f"Deleted document {document_id} with {deleted_count} chunks")
            
            return {"message": f"Deleted {deleted_count} chunks for document {document_id}"}
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Setup OpenTelemetry instrumentation
    if settings.enable_tracing:
        FastAPIInstrumentor.instrument_app(app)
    
    return app


# Create app instance
app = create_app() 