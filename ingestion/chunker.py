"""Structure-aware text splitting for maritime documents."""

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

# Patterns that indicate section boundaries in maritime regulatory text
_SECTION_SEPARATORS = [
    "\n\nRegulation ",       # SOLAS/MARPOL regulation headings
    "\n\nChapter ",          # Convention chapter headings
    "\n\nAnnex ",            # MARPOL annexes
    "\n\nPart ",             # Document parts
    "\n\nSection ",          # Generic sections
    "\n\nFinding",           # Accident report findings
    "\n\nRecommendation",    # Accident report recommendations
    "\n\nConclusion",        # Report conclusions
    "\n\nAnalysis",          # Report analysis sections
    "\n\n",                  # Paragraph breaks
    "\n",                    # Line breaks
    ". ",                    # Sentences
    " ",
    "",
]


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks using maritime-aware separators.

    Respects regulation boundaries, report sections, and paragraph structure.
    Each chunk inherits its parent document's metadata and gets a chunk_index.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=_SECTION_SEPARATORS,
    )

    all_chunks = []
    for doc in documents:
        text = doc.page_content

        # Try to extract a section heading for the chunk metadata
        chunks = splitter.split_documents([doc])
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

            # Detect if this chunk starts with a known section heading
            section = _detect_section(chunk.page_content)
            if section:
                chunk.metadata["section"] = section

            all_chunks.append(chunk)

    return all_chunks


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
