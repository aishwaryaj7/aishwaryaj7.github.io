# DocTract - Document RAG Assistant

Privacy-first document processing system with local AI inference, vector search, and conversational interfaces for intelligent document analysis.

---

## 🎯 **What This Project Does**

- 📄 **Document Processing** - Upload and process PDFs with intelligent text extraction
- 🧠 **Local AI Inference** - Llama 2 model running locally for complete privacy
- 🔍 **Vector Search** - PostgreSQL + PGVector for semantic document retrieval
- 💬 **Conversational Interface** - Ask questions about documents in natural language
- 🔒 **Privacy-First** - All processing happens locally, no data leaves your system

---

## 🌐 **Live Demo**

**📺 Demo Video**: [YouTube Demo](https://youtu.be/9pInYAGtVZA)

---

## 🚀 **How to Use**

### **Local Setup**

```bash
# Navigate to project directory
cd ai-portfolio/projects/doctract

# Install dependencies
uv sync

# Set up PostgreSQL with PGVector
# (Ensure PostgreSQL is running with pgvector extension)

# Run the Streamlit app
uv run streamlit run src/doctract/rag/app.py
```

### **Using the App**

1. **Upload PDF** - Drag and drop any PDF document
2. **Ask Questions** - Type natural language questions about the content
3. **Get Answers** - Receive contextual responses from local AI
4. **Adjust Settings** - Tune retrieval parameters for optimal results

---

## 🔧 **Key Features**

- **Local AI Processing**: Llama 2 via Llama.cpp for complete privacy
- **Vector Search**: PostgreSQL + PGVector for semantic document retrieval
- **Smart Chunking**: Intelligent text splitting with configurable parameters
- **Interactive Interface**: Streamlit UI with real-time configuration
- **Fast Retrieval**: Sub-second response times with optimized search

---

## 🛠️ **Tech Stack**

**AI/ML**: Llama 2, HuggingFace Embeddings, Llama.cpp
**Database**: PostgreSQL, PGVector
**Frontend**: Streamlit
**Processing**: PyMuPDF, LlamaIndex
**Language**: Python 3.11+

---

## 🤝 **Skills Demonstrated**

- RAG Systems & Vector Databases
- Local AI Deployment & Privacy-First Design
- Document Processing & Text Extraction
- Interactive UI Development
- System Architecture & Modular Design

---

