"""
rag_pipeline.py
----------------
Core Retrieval-Augmented Generation (RAG) pipeline for the AI-Powered
College Assistant.

Responsibilities:
1. Load raw text documents from the /data folder (college knowledge base).
2. Split documents into smaller overlapping chunks.
3. Generate vector embeddings for each chunk using SentenceTransformers.
4. Store / persist the embeddings in a local ChromaDB collection.
5. Given a user query, retrieve the most relevant chunks (context)
   to be passed to the LLM (Groq).
"""

import os
import glob
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "college_knowledge_base"

# Chunking parameters
CHUNK_SIZE = 700        # characters per chunk
CHUNK_OVERLAP = 120     # overlap between consecutive chunks

# Embedding model (runs locally, no API key required)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# --------------------------------------------------------------------------
# Document loading & chunking
# --------------------------------------------------------------------------
def load_documents(data_dir: str = DATA_DIR) -> List[Dict]:
    """
    Load every .txt file from the data directory.
    Returns a list of dicts: {"source": filename, "text": content}
    """
    documents = []
    for filepath in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append({
            "source": os.path.basename(filepath),
            "text": text
        })
    return documents


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split a long piece of text into overlapping chunks so that semantic
    context isn't lost at chunk boundaries.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += (chunk_size - overlap)

    return chunks


def build_chunk_records(documents: List[Dict]) -> List[Dict]:
    """
    Convert loaded documents into a flat list of chunk records ready
    for embedding. Each record includes an id, the chunk text, and
    metadata (source file name).
    """
    records = []
    for doc in documents:
        chunks = chunk_text(doc["text"])
        for idx, chunk in enumerate(chunks):
            records.append({
                "id": f"{doc['source']}_{idx}",
                "text": chunk,
                "metadata": {"source": doc["source"], "chunk_index": idx}
            })
    return records


# --------------------------------------------------------------------------
# ChromaDB vector store
# --------------------------------------------------------------------------
def get_chroma_client():
    """Return a persistent ChromaDB client stored on disk."""
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_embedding_function():
    """Local sentence-transformer embedding function (no external API needed)."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )


def get_or_create_collection(client=None, reset: bool = False):
    """
    Get the college knowledge base collection, creating it if needed.
    If reset=True, any existing collection is deleted and rebuilt.
    """
    client = client or get_chroma_client()
    embed_fn = get_embedding_function()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # collection didn't exist yet

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def ingest_documents(reset: bool = True) -> int:
    """
    Full ingestion pipeline: load -> chunk -> embed -> store in ChromaDB.
    Returns the number of chunks ingested.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client, reset=reset)

    documents = load_documents()
    if not documents:
        raise FileNotFoundError(
            f"No .txt files found in {DATA_DIR}. Add college documents first."
        )

    records = build_chunk_records(documents)

    collection.add(
        ids=[r["id"] for r in records],
        documents=[r["text"] for r in records],
        metadatas=[r["metadata"] for r in records],
    )

    return len(records)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
def retrieve_context(query: str, top_k: int = 4) -> List[Dict]:
    """
    Given a user query, retrieve the top_k most relevant chunks from
    ChromaDB along with their source metadata.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client, reset=False)

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    retrieved = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        retrieved.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "score": 1 - dist  # cosine similarity approx (higher = more relevant)
        })

    return retrieved


def format_context_for_prompt(retrieved_chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into a single context string, tagged with
    their source file, ready to be inserted into the LLM prompt.
    """
    if not retrieved_chunks:
        return "No relevant information found in the knowledge base."

    formatted_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        formatted_blocks.append(
            f"[Source {i}: {chunk['source']}]\n{chunk['text']}"
        )
    return "\n\n".join(formatted_blocks)


# --------------------------------------------------------------------------
# CLI helper — run this file directly to (re)build the vector database
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("Ingesting college documents into ChromaDB...")
    count = ingest_documents(reset=True)
    print(f"Done. {count} chunks stored in collection '{COLLECTION_NAME}'.")