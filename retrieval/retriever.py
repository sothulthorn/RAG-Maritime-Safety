"""Hybrid retrieval: vector similarity + BM25 keyword search, with reranking."""

import re

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from config import (
    RETRIEVAL_K,
    RETRIEVAL_FETCH_K,
    BM25_WEIGHT,
    VECTOR_WEIGHT,
    RERANKER_ENABLED,
)
from ingestion.embedder import get_vectorstore, LocalEmbeddings
from retrieval.reranker import rerank


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


def _reciprocal_rank_fusion(
    vector_docs: list[Document],
    bm25_docs: list[Document],
    vector_weight: float = VECTOR_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    rrf_k: int = 60,
) -> list[Document]:
    """Combine vector and BM25 results using weighted Reciprocal Rank Fusion."""
    scores: dict[int, float] = {}
    doc_map: dict[int, Document] = {}

    for rank, doc in enumerate(vector_docs):
        doc_id = id(doc)
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0) + vector_weight / (rrf_k + rank + 1)

    for rank, doc in enumerate(bm25_docs):
        # Check for duplicate by content match
        matched_id = None
        for existing_id, existing_doc in doc_map.items():
            if existing_doc.page_content == doc.page_content:
                matched_id = existing_id
                break

        if matched_id:
            scores[matched_id] += bm25_weight / (rrf_k + rank + 1)
        else:
            doc_id = id(doc)
            doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0) + bm25_weight / (rrf_k + rank + 1)

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids]


def retrieve(
    query: str,
    k: int = RETRIEVAL_K,
    embedding_fn=None,
    source_filter: str | None = None,
) -> list[Document]:
    """Hybrid retrieval: vector + BM25, with optional reranking and source filtering.

    Args:
        query: The search query.
        k: Number of final results to return.
        embedding_fn: Optional embedding function (uses default if None).
        source_filter: Filter by organization (e.g., "maib", "ntsb", "tsb").

    Returns:
        List of top-k relevant documents after hybrid search and reranking.
    """
    if embedding_fn is None:
        embedding_fn = LocalEmbeddings()

    vectorstore = get_vectorstore(embedding_fn)
    fetch_k = RETRIEVAL_FETCH_K

    # Build ChromaDB filter for source organization
    where_filter = None
    if source_filter:
        where_filter = {"organization": source_filter}

    # --- Vector search ---
    search_kwargs = {"k": fetch_k}
    if where_filter:
        search_kwargs["filter"] = where_filter

    vector_docs = vectorstore.similarity_search(query, **search_kwargs)

    if not vector_docs:
        return []

    # --- BM25 keyword search over the vector candidates ---
    # We run BM25 over a broader set from ChromaDB to find keyword matches
    # that vector search might have missed
    all_candidates = vectorstore.similarity_search(query, k=min(fetch_k * 3, 100))
    if where_filter:
        all_candidates = [
            d for d in all_candidates
            if d.metadata.get("organization") == source_filter
        ]

    if all_candidates:
        corpus_tokens = [_tokenize(doc.page_content) for doc in all_candidates]
        bm25 = BM25Okapi(corpus_tokens)
        query_tokens = _tokenize(query)
        bm25_scores = bm25.get_scores(query_tokens)

        scored = sorted(
            zip(bm25_scores, all_candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        bm25_docs = [doc for _, doc in scored[:fetch_k]]
    else:
        bm25_docs = []

    # --- Hybrid fusion ---
    fused = _reciprocal_rank_fusion(vector_docs, bm25_docs)

    # --- Reranking ---
    if RERANKER_ENABLED and len(fused) > k:
        results = rerank(query, fused[:fetch_k], top_k=k)
    else:
        results = fused[:k]

    return results
