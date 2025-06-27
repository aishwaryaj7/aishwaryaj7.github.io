from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
import streamlit as st


def load_huggingface_embedding_model(
        model_name: str = "BAAI/bge-small-en") -> HuggingFaceEmbedding:
    """
    Initializes a HuggingFace embedding model for text embeddings.

    Args:
        model_name (str): Name or path of the HuggingFace model to load.
                          Defaults to 'BAAI/bge-small-en'.

    Returns:
        HuggingFaceEmbedding: Initialized embedding model.
    """
    embedding_model = HuggingFaceEmbedding(model_name=model_name)
    return embedding_model


@st.cache_resource
def load_llama_cpp_language_model(
        model_url: str = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        ,
        temperature: float = 0.1,
        max_tokens: int = 256,
        context_window: int = 3900,
        n_gpu_layers: int = 1,
        verbose: bool = True,
) -> LlamaCPP:
    """

    Args:
        model_url (str): URL to the model file (.gguf) on Hugging Face.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum number of new tokens to generate.
        context_window (int): Size of the context window.
        n_gpu_layers (int): Number of layers to offload to GPU.
        verbose (bool): Whether to enable verbose logging.

    Returns:
        LlamaCPP: Initialized language model.
    """
    llm_model = LlamaCPP(
        model_url=model_url,
        # model_path="models/mistral/mistral-7b-v0.1.Q2_K.gguf",
        temperature=temperature,
        max_new_tokens=max_tokens,
        context_window=context_window,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": n_gpu_layers},
        verbose=verbose,
    )
    return llm_model