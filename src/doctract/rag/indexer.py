from typing import List, Any
from llama_index.core.schema import TextNode


def create_text_nodes_with_metadata(
    text_chunks: List[str],
    source_document_indices: List[int],
    original_documents: List,
) -> List[TextNode]:
    """
    Creates TextNode objects from text chunks and assigns metadata
    from the corresponding source documents.

    Args:
        text_chunks (List[str]): List of segmented text strings.
        source_document_indices (List[int]): Index map linking each chunk to its document.
        original_documents (List): Original document objects with metadata.

    Returns:
        List[TextNode]: List of enriched TextNode objects.
    """
    text_nodes: List[TextNode] = []

    for chunk_index, chunk_text in enumerate(text_chunks):
        source_document = original_documents[source_document_indices[chunk_index]]
        node = TextNode(text=chunk_text)
        node.metadata = source_document.metadata
        text_nodes.append(node)

    return text_nodes


def embed_text_nodes_with_model(
    text_nodes: List[TextNode],
    embedding_model: Any
) -> List[TextNode]:
    """
    Generates embeddings for each TextNode using the given embedding model.

    Args:
        text_nodes (List[TextNode]): The list of TextNodes to embed.
        embedding_model (BaseEmbedding): The embedding model to use.

    Returns:
        List[TextNode]: TextNodes with embedding vectors attached.
    """
    for node in text_nodes:
        content_to_embed = node.get_content(metadata_mode="all")
        node.embedding = embedding_model.get_text_embedding(content_to_embed)

    return text_nodes