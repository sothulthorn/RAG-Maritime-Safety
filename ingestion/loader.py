"""Document loaders for PDF, web, and text files."""

import os

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from pypdf import PdfReader


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


def load_web(url: str) -> list[Document]:
    """Load a web page, stripping HTML to plain text."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    if not text.strip():
        return []

    return [Document(
        page_content=text,
        metadata={"source": url},
    )]


def load_text(path: str) -> list[Document]:
    """Load a plain text or markdown file."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        return []

    return [Document(
        page_content=text,
        metadata={"source": os.path.basename(path), "file_path": path},
    )]


def load_document(path_or_url: str) -> list[Document]:
    """Detect document type and load accordingly.

    Supports: .pdf, .txt, .md files, and http(s) URLs.
    """
    if path_or_url.startswith(("http://", "https://")):
        return load_web(path_or_url)

    ext = os.path.splitext(path_or_url)[1].lower()
    if ext == ".pdf":
        return load_pdf(path_or_url)
    elif ext in (".txt", ".md"):
        return load_text(path_or_url)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
