"""Tests for document loaders."""

import os
import tempfile

import pytest

from ingestion.loader import load_text, load_pdf, load_document


def test_load_text_file():
    """Load a .txt file and verify content and metadata."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("SOLAS Chapter II-2: Fire protection, fire detection and fire extinction.")
        path = f.name

    try:
        docs = load_text(path)
        assert len(docs) == 1
        assert "SOLAS" in docs[0].page_content
        assert "source" in docs[0].metadata
    finally:
        os.unlink(path)


def test_load_text_empty():
    """Empty files return no documents."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("")
        path = f.name

    try:
        docs = load_text(path)
        assert len(docs) == 0
    finally:
        os.unlink(path)


def test_load_document_dispatcher_txt():
    """Dispatcher correctly routes .txt files."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Test content")
        path = f.name

    try:
        docs = load_document(path)
        assert len(docs) == 1
    finally:
        os.unlink(path)


def test_load_document_unsupported():
    """Unsupported extensions raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document("file.xyz")
