"""Central configuration constants for the Maritime Safety RAG application."""

import os

# LLM settings
LLM_MODEL = "llama3"
LLM_TEMPERATURE = 0.1

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Text splitting settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
RETRIEVAL_K = 5

# ChromaDB settings
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
CHROMA_COLLECTION_NAME = "maritime_safety"
