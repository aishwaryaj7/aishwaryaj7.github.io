from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.postgres import PGVectorStore
from constants import DATA_DIR
from llama_index.readers.file import PyMuPDFReader
import psycopg2

# Init Embedding Model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"

# Init LLM
llm = LlamaCPP(
    model_url=model_url,
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": 1},
    verbose=True,
)

# Init Vector Store
db_name = "vector_db"
host = "localhost"
password = "password"
port = "5432"
user = "jerry"

# Connect to default DB to drop and recreate `vector_db`
default_conn = psycopg2.connect(
    dbname="postgres",  # or some admin db
    host=host,
    password=password,
    port=port,
    user=user,
)
default_conn.autocommit = True

with default_conn.cursor() as c:
    c.execute(f"DROP DATABASE IF EXISTS {db_name}")
    c.execute(f"CREATE DATABASE {db_name}")
default_conn.close()

# Reconnect to the new vector_db
conn = psycopg2.connect(
    dbname=db_name,
    host=host,
    password=password,
    port=port,
    user=user,
)
conn.autocommit = True

with conn.cursor() as c:
    c.execute("CREATE EXTENSION IF NOT EXISTS vector")

vector_store = PGVectorStore.from_params(
    database=db_name,
    host=host,
    password=password,
    port=port,
    user=user,
    table_name="llama2_paper",
    embed_dim=384,  # openai embedding dimension
)


# Build an ingestion pipeline

loader = PyMuPDFReader()
documents = loader.load(file_path= DATA_DIR / "llama2.pdf")

text_parser = SentenceSplitter(
    chunk_size=1024,
    # separator=" ",
)
text_chunks = []
# maintain relationship with source doc index, to help inject doc metadata in (3)
doc_idxs = []
for doc_idx, doc in enumerate(documents):
    cur_text_chunks = text_parser.split_text(doc.text)
    text_chunks.extend(cur_text_chunks)
    doc_idxs.extend([doc_idx] * len(cur_text_chunks))


# Manually construct nodes from text chunks
from llama_index.core.schema import TextNode

nodes = []
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(
        text=text_chunk,
    )
    src_doc = documents[doc_idxs[idx]]
    node.metadata = src_doc.metadata
    nodes.append(node)


for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding

vector_store.add(nodes)


# Build the retrieval pipeline
query_str = "Can you tell me about the key concepts for safety finetuning"
query_embedding = embed_model.get_query_embedding(query_str)

# construct vector store query
from llama_index.core.vector_stores import VectorStoreQuery

query_mode = "default"
# query_mode = "sparse"
# query_mode = "hybrid"

vector_store_query = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
)

query_result = vector_store.query(vector_store_query)
print(query_result.nodes[0].get_content())


from llama_index.core.schema import NodeWithScore
from typing import Optional

nodes_with_scores = []
for index, node in enumerate(query_result.nodes):
    score: Optional[float] = None
    if query_result.similarities is not None:
        score = query_result.similarities[index]
    nodes_with_scores.append(NodeWithScore(node=node, score=score))





print('_')

"""
from constants import DATA_DIR
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader(DATA_DIR / "pdf").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What are these documents about?")
print(response)
"""