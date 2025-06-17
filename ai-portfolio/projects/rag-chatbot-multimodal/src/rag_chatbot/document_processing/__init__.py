"""Multi-modal document processing module."""

from .extractor import MultiModalDocumentExtractor

# Alias for backward compatibility
DocumentExtractor = MultiModalDocumentExtractor

__all__ = ["DocumentExtractor", "MultiModalDocumentExtractor"]