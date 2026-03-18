"""Embedding wrapper and ChromaDB storage."""

import hashlib

from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

from config import EMBEDDING_MODEL, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME


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


def _make_chunk_id(source: str, chunk_index: int) -> str:
    """Create a deterministic ID from source and chunk index to prevent duplicates."""
    key = f"{source}::{chunk_index}"
    return hashlib.sha256(key.encode()).hexdigest()


def get_vectorstore(embedding_fn: Embeddings | None = None) -> Chroma:
    """Get or create the ChromaDB vector store."""
    if embedding_fn is None:
        embedding_fn = LocalEmbeddings()

    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_fn,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def ingest(chunks: list[Document], embedding_fn: Embeddings | None = None) -> int:
    """Embed and store document chunks in ChromaDB.

    Returns the number of chunks ingested.
    Uses deterministic IDs to prevent duplicate ingestion.
    """
    if not chunks:
        return 0

    vectorstore = get_vectorstore(embedding_fn)

    ids = [
        _make_chunk_id(chunk.metadata.get("source", "unknown"), chunk.metadata.get("chunk_index", i))
        for i, chunk in enumerate(chunks)
    ]

    vectorstore.add_documents(documents=chunks, ids=ids)
    return len(chunks)
