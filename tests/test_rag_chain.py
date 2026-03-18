"""Tests for the RAG chain formatting utilities."""

from langchain_core.documents import Document

from retrieval.rag_chain import _format_context, _extract_sources


def test_format_context_with_page():
    """Context includes source label with page number."""
    docs = [Document(
        page_content="Fire safety regulation text.",
        metadata={"source": "solas.pdf", "page": 3},
    )]
    result = _format_context(docs)
    assert "[Source: solas.pdf, Page 3]" in result
    assert "Fire safety regulation text." in result


def test_format_context_without_page():
    """Context works for sources without page numbers."""
    docs = [Document(
        page_content="Web content about maritime law.",
        metadata={"source": "https://example.com"},
    )]
    result = _format_context(docs)
    assert "[Source: https://example.com]" in result


def test_extract_sources():
    """Source extraction returns structured metadata."""
    docs = [
        Document(page_content="Some content here.", metadata={"source": "doc.pdf", "page": 1}),
        Document(page_content="More content.", metadata={"source": "https://imo.org"}),
    ]
    sources = _extract_sources(docs)
    assert len(sources) == 2
    assert sources[0]["source"] == "doc.pdf"
    assert sources[0]["page"] == 1
    assert "page" not in sources[1]
    assert sources[1]["source"] == "https://imo.org"
