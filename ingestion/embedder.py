"""Embedding wrapper and ChromaDB storage with Parent-Child support."""

import hashlib

import chromadb
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

from config import (
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    CHROMA_PARENT_COLLECTION,
)

_chroma_client = None


def _get_chroma_client() -> chromadb.ClientAPI:
    """Get or create a shared PersistentClient for ChromaDB."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _chroma_client


class LocalEmbeddings(Embeddings):
    """LangChain-compatible wrapper around sentence-transformers."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode([text], show_progress_bar=False)
        return embedding[0].tolist()


def _make_chunk_id(source: str, page: int | str, chunk_index: int, prefix: str = "c", parent_index: int | str = 0) -> str:
    """Create a deterministic ID to prevent duplicates."""
    key = f"{source}::p{page}::pi{parent_index}::{prefix}{chunk_index}"
    return hashlib.sha256(key.encode()).hexdigest()


def get_vectorstore(embedding_fn: Embeddings | None = None) -> Chroma:
    """Get or create the child chunk vector store (for searching)."""
    if embedding_fn is None:
        embedding_fn = LocalEmbeddings()

    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_fn,
        client=_get_chroma_client(),
    )


def get_parent_store(embedding_fn: Embeddings | None = None) -> Chroma:
    """Get or create the parent chunk vector store (for context retrieval)."""
    if embedding_fn is None:
        embedding_fn = LocalEmbeddings()

    return Chroma(
        collection_name=CHROMA_PARENT_COLLECTION,
        embedding_function=embedding_fn,
        client=_get_chroma_client(),
    )


def get_collection_count(embedding_fn: Embeddings | None = None) -> int:
    """Get the number of child chunks in the collection."""
    try:
        vs = get_vectorstore(embedding_fn)
        return vs._collection.count()
    except Exception:
        return 0


def get_parent_count(embedding_fn: Embeddings | None = None) -> int:
    """Get the number of parent chunks in the collection."""
    try:
        vs = get_parent_store(embedding_fn)
        return vs._collection.count()
    except Exception:
        return 0


def ingest(
    chunk_data: dict | list[Document],
    embedding_fn: Embeddings | None = None,
    batch_size: int = 100,
) -> int:
    """Embed and store document chunks in ChromaDB.

    Accepts either:
    - dict from chunk_documents() with "children" and "parents" keys
    - list[Document] for backward compatibility (treated as children only)

    Returns the total number of chunks ingested (children + parents).
    """
    if isinstance(chunk_data, list):
        # Backward compatibility: flat list of chunks
        return _ingest_docs(chunk_data, get_vectorstore(embedding_fn), embedding_fn, batch_size, "c")

    children = chunk_data.get("children", [])
    parents = chunk_data.get("parents", [])

    child_count = _ingest_docs(children, get_vectorstore(embedding_fn), embedding_fn, batch_size, "child")
    parent_count = _ingest_docs(parents, get_parent_store(embedding_fn), embedding_fn, batch_size, "parent")

    return child_count + parent_count


def _ingest_docs(
    chunks: list[Document],
    vectorstore: Chroma,
    embedding_fn: Embeddings | None,
    batch_size: int,
    id_prefix: str,
) -> int:
    """Ingest a list of documents into a specific vector store."""
    if not chunks:
        return 0

    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        ids = [
            _make_chunk_id(
                chunk.metadata.get("source", "unknown"),
                chunk.metadata.get("page", 0),
                chunk.metadata.get("chunk_index", start + i),
                prefix=id_prefix,
                parent_index=chunk.metadata.get("parent_chunk_index", 0),
            )
            for i, chunk in enumerate(batch)
        ]

        vectorstore.add_documents(documents=batch, ids=ids)
        total += len(batch)

    return total
