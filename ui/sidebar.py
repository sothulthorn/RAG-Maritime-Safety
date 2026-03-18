"""Document upload and management sidebar."""

import os
import tempfile

import streamlit as st

from ingestion.loader import load_document
from ingestion.chunker import chunk_documents
from ingestion.embedder import ingest, get_vectorstore, LocalEmbeddings


def _get_embedding_fn():
    """Get cached embedding function."""
    if "embedding_fn" not in st.session_state:
        st.session_state.embedding_fn = LocalEmbeddings()
    return st.session_state.embedding_fn


def _ingest_file(uploaded_file):
    """Process and ingest an uploaded file."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        docs = load_document(tmp_path)
        # Replace temp path metadata with original filename
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
            if "file_path" in doc.metadata:
                doc.metadata["file_path"] = uploaded_file.name

        chunks = chunk_documents(docs)
        count = ingest(chunks, embedding_fn=_get_embedding_fn())
        return count
    finally:
        os.unlink(tmp_path)


def _ingest_url(url: str):
    """Process and ingest a web URL."""
    docs = load_document(url)
    chunks = chunk_documents(docs)
    return ingest(chunks, embedding_fn=_get_embedding_fn())


def _get_collection_stats():
    """Get document count from ChromaDB."""
    try:
        vs = get_vectorstore(_get_embedding_fn())
        collection = vs._collection
        return collection.count()
    except Exception:
        return 0


def render_sidebar():
    """Render the document management sidebar."""
    st.sidebar.header("Document Management")

    # File upload
    st.sidebar.subheader("Upload Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files and st.sidebar.button("Ingest Files", key="ingest_files"):
        total_chunks = 0
        progress = st.sidebar.progress(0)
        for i, f in enumerate(uploaded_files):
            try:
                count = _ingest_file(f)
                total_chunks += count
                st.sidebar.success(f"{f.name}: {count} chunks")
            except Exception as e:
                st.sidebar.error(f"{f.name}: {e}")
            progress.progress((i + 1) / len(uploaded_files))
        if total_chunks:
            st.sidebar.info(f"Total: {total_chunks} chunks ingested")

    # URL ingestion
    st.sidebar.subheader("Ingest from URL")
    url = st.sidebar.text_input("Enter a web URL", key="url_input")
    if url and st.sidebar.button("Ingest URL", key="ingest_url"):
        try:
            with st.sidebar:
                with st.spinner("Fetching and ingesting..."):
                    count = _ingest_url(url)
                st.success(f"Ingested {count} chunks from URL")
        except Exception as e:
            st.sidebar.error(f"Failed to ingest URL: {e}")

    # Collection stats
    st.sidebar.divider()
    st.sidebar.subheader("Collection Info")
    chunk_count = _get_collection_stats()
    st.sidebar.metric("Total Chunks in Database", chunk_count)

    # Clear collection
    if chunk_count > 0 and st.sidebar.button("Clear All Documents", type="secondary"):
        try:
            vs = get_vectorstore(_get_embedding_fn())
            vs.delete_collection()
            st.sidebar.success("Collection cleared")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed to clear: {e}")
