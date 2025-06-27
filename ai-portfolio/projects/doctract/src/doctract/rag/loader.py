from pathlib import Path
from typing import List, Tuple

from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader

from doctract.constants import DATA_DIR


def load_pdf_document_from_data_dir(file_path: Path) -> List:
    """
    Loads a PDF document from the local data directory using PyMuPDF.

    Args:
        filename (str): Name of the PDF file to load.

    Returns:
        List: A list of Document objects parsed from the PDF.
    """
    pdf_loader = PyMuPDFReader()
    parsed_documents = pdf_loader.load(file_path=file_path)
    return parsed_documents


def split_document_into_text_chunks(documents: List, chunk_size: int = 1024) -> Tuple[
    List[str], List[int]]:
    """
    Splits loaded documents into chunks using sentence-based splitting.

    Args:
        documents (List): List of documents loaded from PDF.
        chunk_size (int): Max characters per chunk. Default is 1024.

    Returns:
        Tuple[List[str], List[int]]:
            - List of text chunks
            - List of indices indicating the origin document for each chunk
    """
    sentence_splitter = SentenceSplitter(chunk_size=chunk_size)
    chunked_texts: List[str] = []
    document_indices: List[int] = []

    for document_index, document in enumerate(documents):
        current_chunks = sentence_splitter.split_text(document.text)
        chunked_texts.extend(current_chunks)
        document_indices.extend([document_index] * len(current_chunks))

    return chunked_texts, document_indices