"""Tests for document loaders."""

import os
import tempfile

import pytest

from ingestion.loader import load_text, load_document, _detect_organization, list_data_files


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
    """Unsupported extensions return empty list."""
    docs = load_document("file.xyz")
    assert docs == []


def test_detect_organization():
    """Organization detection from file paths."""
    assert _detect_organization("/data/maib/report.pdf") == "maib"
    assert _detect_organization("/data/ntsb/MAR2101.pdf") == "ntsb"
    assert _detect_organization("/data/tsb/M23C0305.pdf") == "tsb"
    assert _detect_organization("/data/sample/solas.txt") == "sample"
    assert _detect_organization("/data/other/file.pdf") == "other"
