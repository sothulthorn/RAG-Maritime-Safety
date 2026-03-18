"""Tests for text chunking."""

from langchain_core.documents import Document

from ingestion.chunker import chunk_documents, _detect_section
from config import CHUNK_SIZE


def test_chunk_long_document():
    """A document longer than CHUNK_SIZE gets split into multiple chunks."""
    long_text = "Maritime safety regulation. " * 200  # ~5600 chars
    docs = [Document(page_content=long_text, metadata={"source": "test.txt"})]

    chunks = chunk_documents(docs)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= CHUNK_SIZE + 50  # small tolerance


def test_chunk_preserves_metadata():
    """Chunks inherit parent metadata and get chunk_index."""
    docs = [Document(
        page_content="Short text about SOLAS fire safety regulations.",
        metadata={"source": "solas.pdf", "page": 3, "organization": "sample"},
    )]

    chunks = chunk_documents(docs)
    assert len(chunks) >= 1
    assert chunks[0].metadata["source"] == "solas.pdf"
    assert chunks[0].metadata["page"] == 3
    assert chunks[0].metadata["organization"] == "sample"
    assert "chunk_index" in chunks[0].metadata


def test_chunk_short_document():
    """A short document stays as one chunk."""
    docs = [Document(page_content="Short text.", metadata={"source": "test.txt"})]

    chunks = chunk_documents(docs)
    assert len(chunks) == 1
    assert chunks[0].page_content == "Short text."


def test_detect_section_regulation():
    """Detects regulation headings."""
    text = "Regulation 7 — Detection and Alarm\n7.1 Purpose: to detect fire."
    section = _detect_section(text)
    assert section is not None
    assert "Regulation 7" in section


def test_detect_section_chapter():
    """Detects chapter headings."""
    text = "Chapter II-2: Construction\nFire protection requirements."
    section = _detect_section(text)
    assert section is not None
    assert "Chapter II" in section


def test_detect_section_none():
    """Returns None for text without section headings."""
    text = "Some plain text about maritime safety."
    section = _detect_section(text)
    assert section is None
