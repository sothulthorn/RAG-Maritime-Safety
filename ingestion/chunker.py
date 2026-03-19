"""Structure-aware Parent-Child text splitting for maritime documents.

Strategy:
  1. Split each document into PARENT chunks (~1500 chars) — full context units
  2. Split each parent into CHILD chunks (~300 chars) — precise retrieval units
  3. Store both: search on children, return parents for LLM context

This gives the best of both worlds:
  - Small chunks match queries precisely (high retrieval accuracy)
  - Large parent chunks give the LLM full context (better answer quality)
"""

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
)

# Maritime-aware separators
_SECTION_SEPARATORS = [
    "\n\nRegulation ",
    "\n\nChapter ",
    "\n\nAnnex ",
    "\n\nPart ",
    "\n\nSection ",
    "\n\nFinding",
    "\n\nRecommendation",
    "\n\nConclusion",
    "\n\nAnalysis",
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]


def chunk_documents(documents: list[Document]) -> dict:
    """Split documents into parent and child chunks.

    Returns:
        dict with:
        - "children": list of small chunks for vector search
        - "parents": list of large chunks for LLM context
        - "child_to_parent": mapping from child chunk_id to parent chunk_id
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=_SECTION_SEPARATORS,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=_SECTION_SEPARATORS,
    )

    all_parents = []
    all_children = []
    child_to_parent = {}

    for doc in documents:
        # Create parent chunks
        parent_chunks = parent_splitter.split_documents([doc])

        for p_idx, parent in enumerate(parent_chunks):
            parent.metadata["chunk_index"] = p_idx
            parent.metadata["chunk_type"] = "parent"

            parent_id = _make_parent_id(parent)

            section = _detect_section(parent.page_content)
            if section:
                parent.metadata["section"] = section

            all_parents.append(parent)

            # Create child chunks from this parent
            child_chunks = child_splitter.split_documents([parent])

            for c_idx, child in enumerate(child_chunks):
                child.metadata["chunk_index"] = c_idx
                child.metadata["parent_chunk_index"] = p_idx
                child.metadata["chunk_type"] = "child"

                if section:
                    child.metadata["section"] = section

                child_id = _make_child_id(child)
                child_to_parent[child_id] = parent_id

                all_children.append(child)

    return {
        "children": all_children,
        "parents": all_parents,
        "child_to_parent": child_to_parent,
    }


def chunk_documents_flat(documents: list[Document]) -> list[Document]:
    """Legacy flat chunking for backward compatibility with tests.

    Returns only child-level chunks as a flat list.
    """
    result = chunk_documents(documents)
    return result["children"]


def _make_parent_id(doc: Document) -> str:
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", 0)
    idx = doc.metadata.get("chunk_index", 0)
    return f"{source}::p{page}::parent{idx}"


def _make_child_id(doc: Document) -> str:
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", 0)
    p_idx = doc.metadata.get("parent_chunk_index", 0)
    c_idx = doc.metadata.get("chunk_index", 0)
    return f"{source}::p{page}::parent{p_idx}::child{c_idx}"


def _detect_section(text: str) -> str | None:
    """Extract section heading from the start of a chunk if present."""
    patterns = [
        r"^(Regulation\s+\d+[.\d]*\s*[—–-]?\s*.+?)(?:\n|$)",
        r"^(Chapter\s+[IVXLC\d]+[.\d]*\s*[—–:-]?\s*.+?)(?:\n|$)",
        r"^(Annex\s+[IVXLC\d]+\s*[—–:-]?\s*.+?)(?:\n|$)",
        r"^(Part\s+[A-Z\d]+\s*[—–:-]?\s*.+?)(?:\n|$)",
        r"^(Section\s+\d+[.\d]*\s*[—–:-]?\s*.+?)(?:\n|$)",
        r"^(\d+\.\d+[.\d]*\s+.+?)(?:\n|$)",
    ]
    for pattern in patterns:
        match = re.match(pattern, text.strip(), re.IGNORECASE)
        if match:
            return match.group(1).strip()[:100]
    return None
