"""Document loaders for PDF, web, and text files, plus batch loading from data/."""

import os
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from pypdf import PdfReader

from config import DATA_DIR, SOURCE_ORGS


def load_pdf(path: str) -> list[Document]:
    """Load a PDF file, returning one Document per page."""
    reader = PdfReader(path)
    filename = os.path.basename(path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": filename, "page": i + 1, "file_path": path},
            ))
    return docs


def load_text(path: str) -> list[Document]:
    """Load a plain text or markdown file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    if not text.strip():
        return []

    return [Document(
        page_content=text,
        metadata={"source": os.path.basename(path), "file_path": path},
    )]


def load_document(path: str) -> list[Document]:
    """Detect document type and load accordingly."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext in (".txt", ".md"):
        return load_text(path)
    else:
        return []


def _detect_organization(file_path: str) -> str:
    """Detect source organization from file path (maib/, ntsb/, tsb/, etc.)."""
    parts = Path(file_path).parts
    for part in parts:
        lower = part.lower()
        if lower in SOURCE_ORGS:
            return lower
    return "other"


def load_all_documents(data_dir: str = DATA_DIR) -> list[Document]:
    """Recursively load all supported documents from the data directory.

    Adds 'organization' metadata based on subdirectory name.
    """
    supported_exts = {".pdf", ".txt", ".md"}
    all_docs = []
    data_path = Path(data_dir)

    if not data_path.exists():
        return []

    files = sorted(data_path.rglob("*"))
    for file_path in files:
        if file_path.suffix.lower() not in supported_exts:
            continue
        if not file_path.is_file():
            continue

        try:
            docs = load_document(str(file_path))
            org = _detect_organization(str(file_path))
            for doc in docs:
                doc.metadata["organization"] = org
            all_docs.extend(docs)
        except Exception as e:
            print(f"  [WARN] Failed to load {file_path.name}: {e}")

    return all_docs


def list_data_files(data_dir: str = DATA_DIR) -> dict[str, list[str]]:
    """List all loadable files grouped by organization."""
    supported_exts = {".pdf", ".txt", ".md"}
    data_path = Path(data_dir)
    grouped: dict[str, list[str]] = {}

    if not data_path.exists():
        return grouped

    for file_path in sorted(data_path.rglob("*")):
        if file_path.suffix.lower() not in supported_exts or not file_path.is_file():
            continue
        org = _detect_organization(str(file_path))
        grouped.setdefault(org, []).append(file_path.name)

    return grouped
