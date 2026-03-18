"""Sidebar: collection stats, source filtering, and ingestion controls."""

import streamlit as st

from config import SOURCE_ORGS
from ingestion.loader import load_all_documents, list_data_files
from ingestion.chunker import chunk_documents
from ingestion.embedder import ingest, get_vectorstore, get_collection_count, LocalEmbeddings


def _get_embedding_fn():
    """Get cached embedding function."""
    if "embedding_fn" not in st.session_state:
        st.session_state.embedding_fn = LocalEmbeddings()
    return st.session_state.embedding_fn


def render_sidebar():
    """Render the document management sidebar."""
    st.sidebar.header("Maritime Safety RAG")

    # --- Source Filter ---
    st.sidebar.subheader("Source Filter")
    filter_options = {"All Sources": None}
    for key, label in SOURCE_ORGS.items():
        filter_options[label] = key

    selected_label = st.sidebar.selectbox(
        "Filter answers by source",
        options=list(filter_options.keys()),
        key="source_filter_select",
    )
    st.session_state.source_filter = filter_options[selected_label]

    # --- Collection Stats ---
    st.sidebar.divider()
    st.sidebar.subheader("Document Database")
    chunk_count = get_collection_count(_get_embedding_fn())
    st.sidebar.metric("Total Chunks", chunk_count)

    # Show available data files
    data_files = list_data_files()
    if data_files:
        with st.sidebar.expander("Available Data Files"):
            for org, files in data_files.items():
                label = SOURCE_ORGS.get(org, org.upper())
                st.markdown(f"**{label}**: {len(files)} files")
    else:
        st.sidebar.warning("No data files found in data/ directory. Run the scraper first.")

    # --- Ingest Button ---
    st.sidebar.divider()
    if chunk_count == 0:
        st.sidebar.warning("Database is empty. Click below to ingest documents.")

    if st.sidebar.button("Ingest All Documents", type="primary", key="ingest_btn"):
        _run_ingestion()

    # --- Clear Button ---
    if chunk_count > 0 and st.sidebar.button("Clear Database", type="secondary", key="clear_btn"):
        try:
            vs = get_vectorstore(_get_embedding_fn())
            vs.delete_collection()
            st.sidebar.success("Database cleared.")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed to clear: {e}")


def _run_ingestion():
    """Load, chunk, and ingest all documents from data/ directory."""
    with st.sidebar:
        with st.spinner("Loading documents from data/ ..."):
            docs = load_all_documents()

        if not docs:
            st.error("No documents found in data/ directory.")
            return

        st.info(f"Loaded {len(docs)} document pages. Chunking...")

        with st.spinner("Chunking documents..."):
            chunks = chunk_documents(docs)

        st.info(f"Created {len(chunks)} chunks. Embedding and storing...")

        progress = st.progress(0)
        embedding_fn = _get_embedding_fn()
        batch_size = 100
        total_ingested = 0

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            count = ingest(batch, embedding_fn=embedding_fn, batch_size=len(batch))
            total_ingested += count
            progress.progress(min(1.0, (start + batch_size) / len(chunks)))

        progress.progress(1.0)
        st.success(f"Ingested {total_ingested} chunks into the database.")
        st.rerun()
