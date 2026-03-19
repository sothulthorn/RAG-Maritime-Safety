"""Tests for Parent-Child text chunking."""

from langchain_core.documents import Document

from ingestion.chunker import chunk_documents, chunk_documents_flat, _detect_section
from config import CHILD_CHUNK_SIZE, PARENT_CHUNK_SIZE


def test_parent_child_structure():
    """chunk_documents returns dict with children and parents."""
    long_text = "Maritime safety regulation. " * 200  # ~5600 chars
    docs = [Document(page_content=long_text, metadata={"source": "test.txt"})]

    result = chunk_documents(docs)
    assert "children" in result
    assert "parents" in result
    assert "child_to_parent" in result
    assert len(result["children"]) > len(result["parents"])


def test_child_chunks_smaller_than_parent():
    """Child chunks are smaller than parent chunks."""
    long_text = "Maritime safety regulation. " * 200
    docs = [Document(page_content=long_text, metadata={"source": "test.txt"})]

    result = chunk_documents(docs)
    for child in result["children"]:
        assert len(child.page_content) <= CHILD_CHUNK_SIZE + 50

    for parent in result["parents"]:
        assert len(parent.page_content) <= PARENT_CHUNK_SIZE + 50


def test_chunk_preserves_metadata():
    """Chunks inherit parent metadata and get chunk type."""
    docs = [Document(
        page_content="Short text about SOLAS fire safety regulations.",
        metadata={"source": "solas.pdf", "page": 3, "organization": "sample"},
    )]

    result = chunk_documents(docs)
    assert len(result["children"]) >= 1
    child = result["children"][0]
    assert child.metadata["source"] == "solas.pdf"
    assert child.metadata["organization"] == "sample"
    assert child.metadata["chunk_type"] == "child"

    assert len(result["parents"]) >= 1
    parent = result["parents"][0]
    assert parent.metadata["chunk_type"] == "parent"


def test_flat_chunking_backward_compat():
    """chunk_documents_flat returns a flat list."""
    docs = [Document(page_content="Short text.", metadata={"source": "test.txt"})]
    chunks = chunk_documents_flat(docs)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


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
