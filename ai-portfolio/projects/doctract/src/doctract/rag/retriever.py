from typing import Optional, Any, List

from llama_index.core.schema import NodeWithScore
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.postgres import PGVectorStore


class VectorDatabaseRetriever(BaseRetriever):
    """
    A retriever that queries a PGVector-backed vector store using dense embeddings.

    This class encapsulates the retrieval logic for semantic search by generating
    embeddings from the query string and matching them with nodes stored in the
    underlying PGVectorStore.
    """

    def __init__(
        self,
        vector_store_backend: PGVectorStore,
        embedding_model: Any,
        similarity_search_mode: str = "default",
        top_k_similar_results: int = 2,
    ) -> None:
        """
        Initialize the vector database retriever.

        Args:
            vector_store_backend (PGVectorStore): The vector store to query.
            embedding_model (Any): The embedding model for encoding queries.
            similarity_search_mode (str): Retrieval strategy mode (e.g., "default", "mmr").
            top_k_similar_results (int): Number of top results to return.
        """
        self._vector_store = vector_store_backend
        self._embedding_model = embedding_model
        self._similarity_search_mode = similarity_search_mode
        self._top_k_similar_results = top_k_similar_results
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve the most relevant nodes from the vector store based on the query embedding.

        Args:
            query_bundle (QueryBundle): Contains the raw query string and related metadata.

        Returns:
            List[NodeWithScore]: A ranked list of nodes with associated similarity scores.
        """
        query_embedding = self._embedding_model.get_query_embedding(
            query_bundle.query_str
        )

        query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._top_k_similar_results,
            mode=self._similarity_search_mode,
        )

        result = self._vector_store.query(query)

        nodes_with_scores: List[NodeWithScore] = []
        for idx, node in enumerate(result.nodes):
            similarity_score: Optional[float] = (
                result.similarities[idx] if result.similarities else None
            )
            nodes_with_scores.append(NodeWithScore(node=node, score=similarity_score))

        return nodes_with_scores