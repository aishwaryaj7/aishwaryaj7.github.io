# Building DocTract: My Journey into RAG and Local AI

**By Aishwarya Jauhari Sharma**  
*Published: June 2025*

---

## 🎯 **Why I Built DocTract**

A few months ago, I found myself drowning in research papers. As someone working in AI/ML, I constantly need to reference academic papers, technical documentation, and industry reports. The problem? I'd spend more time searching through documents than actually learning from them.

That's when I decided to build something that could change how I interact with documents. Not just another PDF reader, but an intelligent assistant that could understand my documents and have conversations about them. The result was DocTract - a local-first document processing system that turns any PDF into a conversational partner.

## 🤔 **The Challenge That Excited Me**

The idea seemed simple: upload a PDF, ask questions, get answers. But as I dug deeper, I realized this involved some fascinating technical challenges:

- How do you break down a document so an AI can understand it?
- How do you find the most relevant parts when someone asks a question?
- How do you keep everything private and running locally?
- How do you make the responses actually useful, not just generic?

These questions led me down the rabbit hole of RAG (Retrieval-Augmented Generation) - a technique that combines information retrieval with language generation.

## 🧠 **Understanding RAG: The Heart of DocTract**

RAG was a game-changer for me. Instead of trying to stuff entire documents into an AI model (which doesn't work well), RAG breaks the problem into two parts:

1. **Retrieval**: Find the most relevant pieces of information
2. **Generation**: Use those pieces to generate a helpful response

Think of it like having a really smart research assistant. When you ask a question, they first go through all your documents to find the relevant sections, then use that information to give you a comprehensive answer.

Here's how I implemented the core RAG pipeline in DocTract:

```python
def process_document_for_rag(pdf_path, chunk_size=1024):
    # Step 1: Extract text from PDF
    documents = load_pdf_document_from_data_dir(pdf_path)
    
    # Step 2: Split into manageable chunks
    text_chunks, doc_indices = split_document_into_text_chunks(
        documents=documents, 
        chunk_size=chunk_size
    )
    
    # Step 3: Create searchable nodes with metadata
    nodes = create_text_nodes_with_metadata(
        text_chunks=text_chunks,
        source_document_indices=doc_indices,
        original_documents=documents
    )
    
    return nodes
```

The magic happens in how you chunk the text. Too small, and you lose context. Too large, and the AI gets confused. I settled on 1024 characters as a sweet spot, but made it configurable so users can experiment.

## 🔍 **The Vector Search Revolution**

One of the coolest things I learned was how vector search works. Traditional search looks for exact word matches. Vector search understands meaning.

When you ask "What are the main findings?", it doesn't just look for documents containing those exact words. Instead, it converts your question into a mathematical representation (a vector) and finds document chunks with similar meanings.

I used PostgreSQL with the PGVector extension for this. Here's the retrieval logic:

```python
class VectorDatabaseRetriever(BaseRetriever):
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Convert query to vector representation
        query_embedding = self._embedding_model.get_query_embedding(
            query_bundle.query_str
        )
        
        # Search for similar vectors in the database
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._top_k_similar_results,
            mode=self._similarity_search_mode,
        )
        
        # Return the most relevant chunks
        query_result = self._vector_store.query(vector_store_query)
        return query_result.nodes
```

What blew my mind was how well this works. Ask about "performance metrics" and it finds sections about "evaluation results" or "benchmark scores" - concepts that are related but use different words.

## 🏠 **Going Local: Privacy and Performance**

One decision I'm really proud of is making DocTract completely local. Your documents never leave your machine. This wasn't just about privacy (though that's important) - it was about performance and cost too.

I used Llama 2 through Llama.cpp, which lets you run powerful language models locally. The key was using quantized models - they're smaller and faster while maintaining good quality:

```python
def load_llama_cpp_language_model():
    llm_model = LlamaCPP(
        model_url="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        temperature=0.1,  # Low temperature for factual responses
        max_new_tokens=256,
        context_window=3900,
        model_kwargs={"n_gpu_layers": 1},  # Use GPU if available
        verbose=True,
    )
    return llm_model
```

The Q4_K_M quantization reduces the model size significantly while keeping response quality high. On my laptop, I get responses in 2-10 seconds, which feels snappy for the complexity of what's happening.

## 📚 **Document Processing: The Unsung Hero**

Before any AI magic can happen, you need to extract text from PDFs properly. This sounds simple but is surprisingly tricky. PDFs can have complex layouts, images, tables, and weird formatting.

I chose PyMuPDF because it handles these edge cases well:

```python
def load_pdf_document_from_data_dir(file_path: Path) -> List:
    pdf_loader = PyMuPDFReader()
    parsed_documents = pdf_loader.load(file_path=file_path)
    return parsed_documents
```

The real challenge was chunking. You want chunks that are:
- Large enough to contain complete thoughts
- Small enough for the AI to process effectively  
- Semantically coherent (don't break in the middle of sentences)

I used LlamaIndex's SentenceSplitter, which respects sentence boundaries:

```python
def split_document_into_text_chunks(documents: List, chunk_size: int = 1024):
    sentence_splitter = SentenceSplitter(chunk_size=chunk_size)
    chunked_texts = []
    document_indices = []
    
    for doc_idx, document in enumerate(documents):
        chunks = sentence_splitter.split_text(document.text)
        chunked_texts.extend(chunks)
        document_indices.extend([doc_idx] * len(chunks))
    
    return chunked_texts, document_indices
```

## 🎛️ **Making It User-Friendly**

Technical capabilities mean nothing if people can't use them easily. I built the interface with Streamlit because it lets you create interactive web apps with just Python.

The key insight was making the RAG parameters configurable. Different documents work better with different settings:
- Research papers might need larger chunks to preserve context
- Legal documents might need more retrieved results for comprehensive answers
- Technical manuals might benefit from different search strategies

Users can adjust chunk size, number of results, and search mode in real-time and see how it affects their results.

## 🚀 **What I Learned About RAG Systems**

Building DocTract taught me that RAG isn't just about the technology - it's about understanding how people actually want to interact with information.

**Key insights:**
1. **Context is everything**: The quality of your chunks determines the quality of your answers
2. **Retrieval is as important as generation**: If you can't find the right information, even the best LLM can't help
3. **User control matters**: Different documents and use cases need different approaches
4. **Local deployment is viable**: You don't need cloud APIs for powerful AI applications

## 🔮 **What's Next**

DocTract opened my eyes to the potential of RAG systems. I'm already thinking about improvements:
- Support for more document types (Word, PowerPoint, web pages)
- Better handling of tables and images
- Multi-document conversations
- Integration with note-taking systems

The most exciting part? This is just the beginning. As local AI models get better and more efficient, tools like DocTract will become even more powerful while staying completely private.

## 💭 **Final Thoughts**

Building DocTract was one of those projects where I learned something new every day. RAG systems, vector databases, local AI deployment - each piece was a puzzle that taught me something about how modern AI applications really work.

The best part is seeing it in action. When you upload a complex research paper and ask "What's the main contribution?" and get a clear, accurate answer in seconds - that's when you realize we're living in the future.

If you're interested in document AI or RAG systems, I'd encourage you to try building something similar. The tools are more accessible than ever, and the learning experience is incredible.
