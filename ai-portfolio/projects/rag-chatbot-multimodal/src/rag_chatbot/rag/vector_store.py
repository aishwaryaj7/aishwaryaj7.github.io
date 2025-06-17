"""Modern Weaviate vector store implementation with multi-modal support."""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import weaviate
from weaviate.client import WeaviateClient
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter, MetadataQuery

from ..core.config import settings
from ..core.logger import get_logger

logger = get_logger(__name__)


class WeaviateVectorStore:
    """Production-grade Weaviate vector store with multi-modal document support."""
    
    def __init__(self):
        """Initialize Weaviate connection."""
        self.client: Optional[WeaviateClient] = None
        self.collection_name = settings.weaviate_index_name
        self.is_connected = False
    
    async def connect(self) -> None:
        """Connect to Weaviate instance."""
        try:
            if settings.weaviate_api_key and settings.weaviate_api_key.strip():
                # Connect to Weaviate Cloud
                self.client = weaviate.connect_to_wcs(
                    cluster_url=settings.weaviate_url,
                    auth_credentials=weaviate.auth.AuthApiKey(settings.weaviate_api_key)
                )
            else:
                # Connect to local Weaviate
                host = settings.weaviate_url.replace('http://', '').replace('https://', '').split(':')[0]
                self.client = weaviate.connect_to_local(
                    host=host,
                    port=8080
                )
            
            # Test connection
            if self.client.is_ready():
                self.is_connected = True
                logger.info("Successfully connected to Weaviate")
                await self._setup_schema()
            else:
                raise ConnectionError("Failed to connect to Weaviate")
                
        except Exception as e:
            logger.error(f"Error connecting to Weaviate: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Weaviate."""
        if self.client:
            self.client.close()
            self.is_connected = False
            logger.info("Disconnected from Weaviate")
    
    async def _setup_schema(self) -> None:
        """Set up the Weaviate schema for multi-modal documents."""
        try:
            # Check if collection exists
            if self.client.collections.exists(self.collection_name):
                logger.info(f"Collection {self.collection_name} already exists")
                return
            
            # Create collection with multi-modal schema
            self.client.collections.create(
                name=self.collection_name,
                description="Multi-modal document collection for RAG chatbot",
                vectorizer_config=Configure.Vectorizer.text2vec_openai(
                    model="text-embedding-3-small"
                ),
                generative_config=Configure.Generative.openai(
                    model="gpt-4-1106-preview"
                ),
                properties=[
                    Property(
                        name="content",
                        data_type=DataType.TEXT,
                        description="Main text content of the document chunk"
                    ),
                    Property(
                        name="content_type",
                        data_type=DataType.TEXT,
                        description="Type of content (text, image, table, etc.)"
                    ),
                    Property(
                        name="document_id",
                        data_type=DataType.TEXT,
                        description="Unique identifier for the source document"
                    ),
                    Property(
                        name="document_name",
                        data_type=DataType.TEXT,
                        description="Name of the source document"
                    ),
                    Property(
                        name="chunk_index",
                        data_type=DataType.INT,
                        description="Index of the chunk within the document"
                    ),
                    Property(
                        name="page_number",
                        data_type=DataType.INT,
                        description="Page number in the source document"
                    ),
                    Property(
                        name="metadata",
                        data_type=DataType.OBJECT,
                        description="Additional metadata for the document chunk"
                    ),
                    Property(
                        name="file_path",
                        data_type=DataType.TEXT,
                        description="Path to the source file"
                    ),
                    Property(
                        name="extraction_timestamp",
                        data_type=DataType.DATE,
                        description="Timestamp when content was extracted"
                    ),
                    Property(
                        name="embedding_model",
                        data_type=DataType.TEXT,
                        description="Model used for generating embeddings"
                    ),
                ]
            )
            
            logger.info(f"Created collection {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error setting up schema: {e}")
            raise
    
    async def add_documents(
        self, 
        documents: List[Dict[str, Any]], 
        document_id: str,
        batch_size: int = 100
    ) -> List[str]:
        """
        Add document chunks to the vector store.
        
        Args:
            documents: List of document chunks with content and metadata
            document_id: Unique identifier for the document
            batch_size: Number of documents to process in each batch
            
        Returns:
            List of chunk IDs that were added
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            collection = self.client.collections.get(self.collection_name)
            chunk_ids = []
            
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_objects = []
                
                for j, doc in enumerate(batch):
                    chunk_id = str(uuid4())
                    chunk_ids.append(chunk_id)
                    
                    # Prepare data object
                    data_object = DataObject(
                        uuid=chunk_id,
                        properties={
                            "content": doc.get("content", ""),
                            "content_type": doc.get("content_type", "text"),
                            "document_id": document_id,
                            "document_name": doc.get("document_name", ""),
                            "chunk_index": i + j,
                            "page_number": doc.get("page_number", 0),
                            "metadata": doc.get("metadata", {}),
                            "file_path": doc.get("file_path", ""),
                            "extraction_timestamp": doc.get("extraction_timestamp"),
                            "embedding_model": "text-embedding-3-small",
                        }
                    )
                    batch_objects.append(data_object)
                
                # Insert batch
                result = collection.data.insert_many(batch_objects)
                
                if result.errors:
                    logger.warning(f"Batch insert had {len(result.errors)} errors")
                    for error in result.errors:
                        logger.error(f"Insert error: {error}")
                
                logger.info(f"Inserted batch {i//batch_size + 1} with {len(batch)} documents")
            
            logger.info(f"Successfully added {len(chunk_ids)} chunks for document {document_id}")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_similarity: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using hybrid search.
        
        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional filters for the search
            include_similarity: Whether to include similarity scores
            
        Returns:
            List of relevant documents with metadata
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            collection = self.client.collections.get(self.collection_name)
            
            # Build query with filters
            where_filter = None
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    filter_conditions.append(Filter.by_property(key).equal(value))
                
                if len(filter_conditions) == 1:
                    where_filter = filter_conditions[0]
                else:
                    where_filter = Filter.all_of(filter_conditions)
            
            # Perform hybrid search (semantic + keyword)
            response = collection.query.hybrid(
                query=query,
                limit=limit,
                where=where_filter,
                return_metadata=MetadataQuery(score=include_similarity, explain_score=True) if include_similarity else None
            )
            
            # Format results
            results = []
            for obj in response.objects:
                result = {
                    "id": str(obj.uuid),
                    "content": obj.properties.get("content", ""),
                    "content_type": obj.properties.get("content_type", ""),
                    "document_id": obj.properties.get("document_id", ""),
                    "document_name": obj.properties.get("document_name", ""),
                    "chunk_index": obj.properties.get("chunk_index", 0),
                    "page_number": obj.properties.get("page_number", 0),
                    "metadata": obj.properties.get("metadata", {}),
                    "file_path": obj.properties.get("file_path", ""),
                }
                
                if include_similarity and obj.metadata:
                    result["similarity_score"] = obj.metadata.score
                    result["explain_score"] = obj.metadata.explain_score
                
                results.append(result)
            
            logger.info(f"Found {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise
    
    async def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks for a specific document.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            Number of chunks deleted
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            collection = self.client.collections.get(self.collection_name)
            
            # Delete all chunks for the document
            result = collection.data.delete_many(
                where=Filter.by_property("document_id").equal(document_id)
            )
            
            deleted_count = result.matches
            logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            raise
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        if not self.is_connected:
            await self.connect()
        
        try:
            collection = self.client.collections.get(self.collection_name)
            
            # Get total object count
            total_objects = collection.aggregate.over_all(total_count=True).total_count
            
            # Get objects by content type
            content_types = collection.aggregate.over_all(
                group_by="content_type"
            ).groups
            
            stats = {
                "total_objects": total_objects,
                "content_type_distribution": {
                    group.grouped_by["value"]: group.total_count 
                    for group in content_types
                },
                "collection_name": self.collection_name,
                "is_connected": self.is_connected,
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            raise
    
    def __del__(self):
        """Cleanup on object destruction."""
        if self.is_connected and self.client:
            try:
                self.client.close()
            except Exception:
                pass 