"""Hybrid retrieval with Parent-Child lookup and contextual compression.

Pipeline:
1. Search CHILD chunks (small, precise) via hybrid vector+BM25
2. Rerank candidates with cross-encoder
3. Look up PARENT chunks (large, full context) for matched children
4. Compress parent chunks to keep only query-relevant sentences
5. Return compressed parents for LLM context
"""

import re

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from config import (
    RETRIEVAL_K,
    RETRIEVAL_FETCH_K,
    BM25_WEIGHT,
    VECTOR_WEIGHT,
    RERANKER_ENABLED,
    COMPRESSION_ENABLED,
)
from ingestion.embedder import get_vectorstore, get_parent_store, LocalEmbeddings
from retrieval.reranker import rerank
from retrieval.compressor import compress


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


def _find_parent(child: Document, embedding_fn) -> Document | None:
    """Look up the parent chunk for a matched child chunk.

    Searches the parent collection for the chunk from the same source/page
    that contains the child's content.
    """
    parent_store = get_parent_store(embedding_fn)

    try:
        parent_count = parent_store._collection.count()
    except Exception:
        parent_count = 0

    if parent_count == 0:
        return None

    source = child.metadata.get("source", "")
    page = child.metadata.get("page")
    parent_idx = child.metadata.get("parent_chunk_index")

    # Try to find parent by metadata match
    where_filter = {"source": source}
    if page is not None:
        where_filter["page"] = page

    try:
        candidates = parent_store.similarity_search(
            child.page_content, k=5, filter=where_filter
        )
    except Exception:
        candidates = parent_store.similarity_search(child.page_content, k=5)

    if not candidates:
        return None

    # If we know the parent_chunk_index, prefer exact match
    if parent_idx is not None:
        for c in candidates:
            if c.metadata.get("chunk_index") == parent_idx:
                return c

    # Otherwise return the parent whose content contains the child's text
    child_snippet = child.page_content[:100]
    for c in candidates:
        if child_snippet in c.page_content:
            return c

    # Fallback: return most similar parent from same source
    return candidates[0] if candidates else None


def _expand_to_parents(children: list[Document], embedding_fn) -> list[Document]:
    """Replace child chunks with their parent chunks, deduplicating."""
    seen_parents = set()
    parents = []

    for child in children:
        parent = _find_parent(child, embedding_fn)
        if parent:
            # Deduplicate by content prefix
            key = (parent.metadata.get("source"), parent.metadata.get("page"),
                   parent.metadata.get("chunk_index"))
            if key not in seen_parents:
                seen_parents.add(key)
                parents.append(parent)
        else:
            # No parent found — use the child itself
            key = (child.metadata.get("source"), child.metadata.get("page"),
                   child.page_content[:80])
            if key not in seen_parents:
                seen_parents.add(key)
                parents.append(child)

    return parents


def retrieve(
    query: str,
    k: int = RETRIEVAL_K,
    embedding_fn=None,
    source_filter: str | None = None,
    fast: bool = False,
) -> list[Document]:
    """Full retrieval pipeline: hybrid search → rerank → parent expansion → compression.

    Args:
        query: The search query.
        k: Number of final results to return.
        embedding_fn: Optional embedding function (uses default if None).
        source_filter: Filter by organization (e.g., "maib", "ntsb", "tsb").
        fast: If True, skip reranking and compression for speed.

    Returns:
        List of top-k relevant documents with full context.
    """
    if embedding_fn is None:
        embedding_fn = LocalEmbeddings()

    vectorstore = get_vectorstore(embedding_fn)
    fetch_k = RETRIEVAL_FETCH_K

    # Build ChromaDB filter
    where_filter = None
    if source_filter:
        where_filter = {"organization": source_filter}

    # --- 1. Vector search on child chunks ---
    search_kwargs = {"k": fetch_k}
    if where_filter:
        search_kwargs["filter"] = where_filter

    vector_docs = vectorstore.similarity_search(query, **search_kwargs)

    if not vector_docs:
        return []

    # --- 2. BM25 keyword search ---
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

    # --- 3. Hybrid fusion ---
    fused = _reciprocal_rank_fusion(vector_docs, bm25_docs)

    # --- 4. Reranking ---
    if not fast and RERANKER_ENABLED and len(fused) > k:
        top_children = rerank(query, fused[:fetch_k], top_k=k)
    else:
        top_children = fused[:k]

    # --- 5. Parent expansion ---
    results = _expand_to_parents(top_children, embedding_fn)

    # --- 6. Contextual compression ---
    if not fast and COMPRESSION_ENABLED and results:
        results = compress(query, results)

    return results[:k]
