"""Tests for the retrieval pipeline."""

import shutil
import tempfile
import os

import pytest
from langchain_core.documents import Document

import config
from ingestion.embedder import LocalEmbeddings, ingest, get_vectorstore
from ingestion.chunker import chunk_documents
from retrieval.retriever import retrieve


@pytest.fixture
def temp_chroma(tmp_path, monkeypatch):
    """Use a temporary directory for ChromaDB during tests."""
    monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma_test"))
    yield


@pytest.fixture
def embedding_fn():
    return LocalEmbeddings()


def test_embedding_shape(embedding_fn):
    """Embedding vector has correct dimension."""
    vec = embedding_fn.embed_query("fire safety regulations")
    assert len(vec) == config.EMBEDDING_DIMENSION


def test_similar_embeddings(embedding_fn):
    """Similar sentences have higher cosine similarity than dissimilar ones."""
    import numpy as np

    v1 = np.array(embedding_fn.embed_query("fire safety on ships"))
    v2 = np.array(embedding_fn.embed_query("fire protection aboard vessels"))
    v3 = np.array(embedding_fn.embed_query("chocolate cake recipe"))

    sim_close = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    sim_far = np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))
    assert sim_close > sim_far


def test_ingest_and_retrieve(temp_chroma, embedding_fn):
    """Ingest documents and retrieve relevant chunks."""
    docs = [
        Document(page_content="SOLAS Chapter II-2 covers fire protection, fire detection and fire extinction on ships.", metadata={"source": "solas.pdf", "page": 1}),
        Document(page_content="MARPOL Annex VI addresses air pollution from ships including SOx and NOx emissions.", metadata={"source": "marpol.pdf", "page": 5}),
    ]
    chunks = chunk_documents(docs)
    count = ingest(chunks, embedding_fn=embedding_fn)
    assert count == len(chunks)

    results = retrieve("fire safety", k=1, embedding_fn=embedding_fn)
    assert len(results) >= 1
    assert "fire" in results[0].page_content.lower()
