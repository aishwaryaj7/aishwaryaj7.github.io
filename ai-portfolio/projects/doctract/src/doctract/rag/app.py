from doctract.rag.api.models import load_huggingface_embedding_model, \
    load_llama_cpp_language_model
from doctract.rag.loader import load_pdf_document_from_data_dir, \
    split_document_into_text_chunks
from doctract.rag.indexer import create_text_nodes_with_metadata, \
    embed_text_nodes_with_model
from doctract.rag.retriever import VectorDatabaseRetriever
from doctract.rag.api.psql import initialize_pgvector_store
from llama_index.core.query_engine import RetrieverQueryEngine

from doctract.constants import DATA_DIR

import streamlit as st


def main():
    vector_store = initialize_pgvector_store()
    embedding_model = load_huggingface_embedding_model()
    llm_model = load_llama_cpp_language_model()
    documents = load_pdf_document_from_data_dir(filename="pdf/paper1.pdf")
    text_chunks, doc_idxs = split_document_into_text_chunks(documents=documents)
    nodes = create_text_nodes_with_metadata(text_chunks=text_chunks,
                                            source_document_indices=doc_idxs,
                                            original_documents=documents)
    nodes = embed_text_nodes_with_model(
        text_nodes=nodes,
        embedding_model=embedding_model
    )
    vector_store.add(nodes)
    query_str = "Can you tell me about the document?"
    retriever = VectorDatabaseRetriever(
        vector_store_backend=vector_store,
        embedding_model=embedding_model
    )
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm_model)
    response = query_engine.query(query_str)
    print(str(response))
    print(response.source_nodes[0].get_content())


if __name__ == "__main__":
    # ------------------------------
    # Streamlit Config
    # ------------------------------
    st.set_page_config(page_title="Document Extraction & RAG Assistant", layout="wide")
    st.title("📚 Doctract - Document Extraction & RAG Assistant: Talk to Your Documents")

    st.markdown(""" 👋 Welcome to Doctract Document Extractor and RAG Assistant — a local-first AI tool to chat with your PDFs.

Powered by:

    🧠 Local LLM (Llama.cpp)
    
    🧬 HuggingFace embeddings
    
    🧭 PGVector search via llama-index
    
    🛡️ All processing is private & runs locally.
You also get full control over:

    📏 Chunk size tuning (for optimal context handling)

    🧮 Number of results to retrieve

    🔄 Search strategy: Default or Hybrid

So go ahead: Upload a file, adjust your settings, and start chatting with your documents like never before. 🗣️📄✨
    """)

    st.sidebar.title("📌 About This App")
    st.sidebar.markdown("""
    **Doctract RAG Assistant** is a Streamlit application that demonstrates the core components of a RAG (Retrieval-Augmented Generation) pipeline using local models.

    ### 🔧 Components:
    - **PDF Parsing**: via `PyMuPDF`
    - **Text Chunking**: Sentence splitter with customizable chunk size
    - **Embeddings**: HuggingFace BGE-small
    - **Vector Store**: PostgreSQL + PGVector
    - **Retriever**: Similarity search with top-K and mode options
    - **LLM**: Llama 2 via `llama.cpp` backend

    Use it to **chat with any PDF**—be it research papers, policy docs, or technical guides.
    """)

    tab1, tab2 = st.tabs(["🎥 Application Demo", "🧠 RAG Chat Interface"])

    with tab1:

        st.header("🎬 Doctract Application Working Demo")
        st.markdown("Watch the video below to understand how DocuChat works.")
        st.video("data/rag_demo.mp4")

    with tab2:

        # ------------------------------
        # Session State Initialization
        # ------------------------------
        for key in ["documents", "text_chunks", "doc_idxs", "nodes", "query_engine",
                    "vector_store", "embedding_model", "llm_model"]:
            if key not in st.session_state:
                st.session_state[key] = None

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # ------------------------------
        # Sidebar Hyperparameters
        # ------------------------------
        st.sidebar.header("⚙️ RAG Settings")

        chunk_size = st.sidebar.slider("Chunk Size", min_value=256, max_value=2048,
                                       step=8,
                                       value=1024)
        top_k = st.sidebar.number_input("Top-K Similar Results", min_value=1,
                                        max_value=10,
                                        value=2)
        query_mode = st.sidebar.selectbox("Similarity Search Mode",
                                          options=["default", "hybrid"])

        # ------------------------------
        # PDF Upload and Processing
        # ------------------------------
        uploaded_file = st.file_uploader("📎 Upload your PDF", type=["pdf"])

        if uploaded_file:
            with st.spinner("🔍 Loading models and initializing vector store..."):
                # Save uploaded PDF to data dir
                UPLOAD_FILE_PATH = DATA_DIR / "uploaded" / uploaded_file.name
                with open(UPLOAD_FILE_PATH, "wb") as f:
                    f.write(uploaded_file.read())

                # Load components
                st.session_state.vector_store = initialize_pgvector_store()
                st.session_state.embedding_model = load_huggingface_embedding_model()
                st.session_state.llm_model = load_llama_cpp_language_model()

                st.session_state.documents = load_pdf_document_from_data_dir(
                    file_path=UPLOAD_FILE_PATH)

                st.success("📄 Document loaded successfully!")

                # Chunking
                st.session_state.text_chunks, st.session_state.doc_idxs = split_document_into_text_chunks(
                    documents=st.session_state.documents,
                    chunk_size=chunk_size  # Pass custom chunk size
                )

                # Chunking
                st.session_state.text_chunks, st.session_state.doc_idxs = split_document_into_text_chunks(
                    documents=st.session_state.documents,
                    chunk_size=chunk_size  # Pass custom chunk size
                )

                st.info("✅ Text chunks prepared. Now create index.")

        # ------------------------------
        # Create Index Button
        # ------------------------------
        if st.session_state.documents:
            if st.button("🚀 Create indexes in Vector Store"):
                with st.spinner("🔗 Creating text node index..."):
                    nodes = create_text_nodes_with_metadata(
                        text_chunks=st.session_state.text_chunks,
                        source_document_indices=st.session_state.doc_idxs,
                        original_documents=st.session_state.documents
                    )
                    nodes = embed_text_nodes_with_model(
                        text_nodes=nodes,
                        embedding_model=st.session_state.embedding_model
                    )
                    st.session_state.vector_store.add(nodes)
                    st.session_state.nodes = nodes

                    st.success(
                        "📚 Document embedded and indexed!")

        # ------------------------------
        # Query Interface
        # Build retriever + engine
        # ------------------------------
        retriever = VectorDatabaseRetriever(
            vector_store_backend=st.session_state.vector_store,
            embedding_model=st.session_state.embedding_model,
            top_k_similar_results=top_k,
            similarity_search_mode=query_mode,
        )
        query_engine = RetrieverQueryEngine.from_args(retriever,
                                                      llm=st.session_state.llm_model)
        st.session_state.query_engine = query_engine

        st.success(
            "✅ Query Engine ready for chatting!")

        if st.session_state.query_engine:
            st.divider()
            st.subheader("💬 Chat with your document")

            user_input = st.chat_input("Ask something about the document...")

            if user_input:
                with st.spinner("🧠 Thinking..."):
                    response = st.session_state.query_engine.query(user_input)

                    # Add to history
                    st.session_state.chat_history.append(
                        {"user": user_input, "assistant": str(response)}
                    )

            # Render conversation history
            for chat in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.markdown(chat["user"])
                with st.chat_message("assistant"):
                    st.markdown(chat["assistant"])

    # main()