"""Central configuration constants for the Maritime Safety RAG application."""

import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# LLM settings
LLM_MODEL = "llama3"
LLM_TEMPERATURE = 0.1

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Reranker (cross-encoder) settings
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_ENABLED = True

# Text splitting settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
RETRIEVAL_K = 5               # final results returned
RETRIEVAL_FETCH_K = 20         # candidates fetched before reranking
BM25_WEIGHT = 0.3              # weight for keyword search in hybrid fusion
VECTOR_WEIGHT = 0.7            # weight for semantic search in hybrid fusion

# Data directory
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

# ChromaDB settings
CHROMA_PERSIST_DIR = os.path.join(_PROJECT_ROOT, "chroma_db")
CHROMA_COLLECTION_NAME = "maritime_safety"

# Source organizations
SOURCE_ORGS = {
    "maib": "MAIB (UK)",
    "ntsb": "NTSB (US)",
    "tsb": "TSB (Canada)",
    "sample": "Sample Documents",
}
