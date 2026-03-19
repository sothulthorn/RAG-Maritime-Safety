"""Contextual compression: extract only the relevant sentences from retrieved chunks.

Instead of sending entire chunks (which contain irrelevant sentences mixed in),
this compresses each chunk to only the sentences that are relevant to the query.
This reduces noise in the LLM context and improves answer quality.
"""

import numpy as np
from langchain_core.documents import Document

from ingestion.embedder import LocalEmbeddings

_embeddings: LocalEmbeddings | None = None


def _get_embeddings() -> LocalEmbeddings:
    """Reuse the shared LocalEmbeddings instance (avoids reloading the model)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = LocalEmbeddings()
    return _embeddings


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving meaningful units."""
    import re
    # Split on sentence boundaries but keep abbreviations intact
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Also split on newlines for structured text
    result = []
    for s in sentences:
        parts = s.split("\n")
        for part in parts:
            part = part.strip()
            if len(part) > 15:  # skip very short fragments
                result.append(part)
    return result


def compress(query: str, documents: list[Document], threshold: float = 0.3) -> list[Document]:
    """Compress documents by keeping only query-relevant sentences.

    For each document, computes sentence-level similarity to the query and
    keeps only sentences above the threshold. If no sentences pass,
    keeps the top 3 most relevant.

    Args:
        query: The user's search query.
        documents: Retrieved documents to compress.
        threshold: Minimum cosine similarity to keep a sentence.

    Returns:
        Compressed documents with only relevant content.
    """
    if not documents:
        return []

    emb = _get_embeddings()
    query_embedding = np.array(emb.embed_query(query))

    compressed = []
    for doc in documents:
        sentences = _split_sentences(doc.page_content)
        if not sentences:
            compressed.append(doc)
            continue

        # Compute similarity of each sentence to the query
        sentence_embeddings = np.array(emb.embed_documents(sentences))
        similarities = np.dot(sentence_embeddings, query_embedding) / (
            np.linalg.norm(sentence_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Keep sentences above threshold, or top 5 if none pass
        scored = sorted(zip(similarities, sentences), key=lambda x: x[0], reverse=True)
        relevant = [(sim, sent) for sim, sent in scored if sim >= threshold]

        if len(relevant) < 3:
            relevant = scored[:5]

        # Preserve original order of kept sentences
        kept_sentences = {sent for _, sent in relevant}
        ordered = [s for s in sentences if s in kept_sentences]

        compressed_text = " ".join(ordered)
        compressed_doc = Document(
            page_content=compressed_text,
            metadata={**doc.metadata, "compressed": True, "original_length": len(doc.page_content)},
        )
        compressed.append(compressed_doc)

    return compressed
