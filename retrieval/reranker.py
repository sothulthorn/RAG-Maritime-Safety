"""Cross-encoder reranker for improving retrieval precision."""

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from config import RERANKER_MODEL


_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    """Lazy-load the cross-encoder model."""
    global _model
    if _model is None:
        _model = CrossEncoder(RERANKER_MODEL)
    return _model


def rerank(query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
    """Rerank documents using a cross-encoder model.

    Takes candidate documents from initial retrieval and reranks them
    by computing a relevance score for each (query, document) pair.
    Returns the top-k most relevant documents.
    """
    if not documents:
        return []

    if len(documents) <= top_k:
        return documents

    model = _get_model()

    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)

    scored_docs = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True,
    )

    return [doc for _, doc in scored_docs[:top_k]]
