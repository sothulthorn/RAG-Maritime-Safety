"""Tests for the RAG chain formatting utilities."""

from langchain_core.documents import Document

from retrieval.rag_chain import _format_context, _extract_sources, _is_complex_query


def test_format_context_with_page():
    """Context includes source label with page number."""
    docs = [Document(
        page_content="Fire safety regulation text.",
        metadata={"source": "solas.pdf", "page": 3, "organization": "sample"},
    )]
    result = _format_context(docs)
    assert "[Source: solas.pdf, Page 3" in result
    assert "Fire safety regulation text." in result


def test_format_context_with_section():
    """Context includes section metadata."""
    docs = [Document(
        page_content="Regulation 7 details.",
        metadata={"source": "solas.pdf", "section": "Regulation 7 — Detection"},
    )]
    result = _format_context(docs)
    assert "Section: Regulation 7" in result


def test_format_context_deduplicates():
    """Duplicate content is removed from context."""
    docs = [
        Document(page_content="Same content here.", metadata={"source": "a.pdf"}),
        Document(page_content="Same content here.", metadata={"source": "b.pdf"}),
    ]
    result = _format_context(docs)
    assert result.count("Same content here.") == 1


def test_extract_sources():
    """Source extraction returns structured metadata."""
    docs = [
        Document(page_content="Some content here.", metadata={"source": "doc.pdf", "page": 1, "organization": "maib"}),
        Document(page_content="More content.", metadata={"source": "report.pdf", "organization": "ntsb"}),
    ]
    sources = _extract_sources(docs)
    assert len(sources) == 2
    assert sources[0]["source"] == "doc.pdf"
    assert sources[0]["page"] == 1
    assert sources[0]["organization"] == "MAIB"
    assert sources[1]["organization"] == "NTSB"


def test_is_complex_query():
    """Complex queries are detected correctly."""
    assert _is_complex_query("Compare fire safety requirements between passenger and cargo ships")
    assert _is_complex_query("What is the difference between SOLAS and MARPOL?")
    assert not _is_complex_query("What are the fire detection requirements?")
    assert not _is_complex_query("Tell me about SOLAS Chapter II-2")
