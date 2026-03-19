"""Tests for the hybrid retrieval pipeline with Parent-Child support."""

import pytest
from langchain_core.documents import Document

import config
from ingestion.embedder import LocalEmbeddings, ingest, get_vectorstore
from ingestion.chunker import chunk_documents, chunk_documents_flat
from retrieval.retriever import retrieve, _reciprocal_rank_fusion, _tokenize
from retrieval.compressor import compress


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


def test_tokenize():
    """BM25 tokenizer splits and lowercases."""
    tokens = _tokenize("SOLAS Regulation 7.2")
    assert "solas" in tokens
    assert "regulation" in tokens


def test_reciprocal_rank_fusion():
    """RRF merges two ranked lists correctly."""
    doc_a = Document(page_content="Fire safety", metadata={})
    doc_b = Document(page_content="Life saving", metadata={})
    doc_c = Document(page_content="Navigation", metadata={})

    vector_docs = [doc_a, doc_b]
    bm25_docs = [doc_b, doc_c]

    fused = _reciprocal_rank_fusion(vector_docs, bm25_docs)
    assert fused[0].page_content == "Life saving"
    assert len(fused) == 3


def test_compression(embedding_fn):
    """Contextual compression keeps relevant sentences."""
    docs = [Document(
        page_content="Fire safety is critical on ships. The weather was sunny. "
                     "Fire detection systems must be installed in engine rooms. "
                     "The captain had breakfast at 7am.",
        metadata={"source": "test.pdf"},
    )]

    compressed = compress("fire safety requirements", docs)
    assert len(compressed) == 1
    # Should keep fire-related sentences, possibly drop irrelevant ones
    assert "fire" in compressed[0].page_content.lower()


def test_ingest_and_retrieve(temp_chroma, embedding_fn, monkeypatch):
    """Ingest documents and retrieve relevant chunks with hybrid search."""
    monkeypatch.setattr(config, "RERANKER_ENABLED", False)
    monkeypatch.setattr(config, "COMPRESSION_ENABLED", False)

    docs = [
        Document(
            page_content="SOLAS Chapter II-2 covers fire protection, fire detection and fire extinction on ships. "
                         "This regulation applies to all passenger ships and cargo ships of 500 gross tonnage and above.",
            metadata={"source": "solas.pdf", "page": 1, "organization": "sample"},
        ),
        Document(
            page_content="MARPOL Annex VI addresses air pollution from ships including SOx and NOx emissions. "
                         "Ships must use fuel oil with a sulphur content not exceeding 0.50% m/m.",
            metadata={"source": "marpol.pdf", "page": 5, "organization": "sample"},
        ),
    ]
    chunk_data = chunk_documents(docs)
    count = ingest(chunk_data, embedding_fn=embedding_fn)
    assert count > 0

    results = retrieve("fire safety", k=1, embedding_fn=embedding_fn)
    assert len(results) >= 1
    assert "fire" in results[0].page_content.lower()
