"""
Production-grade multi-modal RAG chatbot for invoice and contract processing.

This package provides a complete RAG pipeline with:
- Multi-modal document processing (PDF, images)
- Vector search with Weaviate
- Production monitoring with OpenTelemetry
- FastAPI backend with audit logging
"""

__version__ = "0.1.0"
__author__ = "Aishwarya"

from .core.config import Settings
from .core.logger import setup_logging

__all__ = ["Settings", "setup_logging"] 