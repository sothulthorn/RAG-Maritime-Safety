"""Sidebar: collection stats, source filtering, and ingestion controls."""

import streamlit as st

from config import SOURCE_ORGS
from ingestion.loader import load_all_documents, list_data_files
from ingestion.chunker import chunk_documents
from ingestion.embedder import (
    ingest, get_vectorstore, get_collection_count, get_parent_count, LocalEmbeddings,
)


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

    child_count = get_collection_count(_get_embedding_fn())
    parent_count = get_parent_count(_get_embedding_fn())

    col1, col2 = st.sidebar.columns(2)
    col1.metric("Child Chunks", child_count)
    col2.metric("Parent Chunks", parent_count)

    if child_count > 0:
        st.sidebar.caption(
            "Child chunks (small) are used for precise search. "
            "Parent chunks (large) provide full context to the LLM."
        )

    # Show available data files
    data_files = list_data_files()
    if data_files:
        with st.sidebar.expander("Available Data Files"):
            for org, files in data_files.items():
                label = SOURCE_ORGS.get(org, org.upper())
                st.markdown(f"**{label}**: {len(files)} files")
    else:
        st.sidebar.warning("No data files found in data/ directory. Run the scraper first.")

    # --- Pipeline Info ---
    st.sidebar.divider()
    st.sidebar.subheader("Pipeline Features")
    st.sidebar.caption(
        "1. Hybrid Search (Vector + BM25)\n"
        "2. Cross-Encoder Reranking\n"
        "3. Parent-Child Retrieval\n"
        "4. Contextual Compression\n"
        "5. Query Decomposition\n"
        "6. Answer Verification"
    )

    # --- Ingest Button ---
    st.sidebar.divider()
    if child_count == 0:
        st.sidebar.warning("Database is empty. Click below to ingest documents.")

    if st.sidebar.button("Ingest All Documents", type="primary", key="ingest_btn"):
        _run_ingestion()

    # --- Clear Button ---
    if child_count > 0 and st.sidebar.button("Clear Database", type="secondary", key="clear_btn"):
        try:
            from ingestion.embedder import get_parent_store
            vs = get_vectorstore(_get_embedding_fn())
            ps = get_parent_store(_get_embedding_fn())
            vs.delete_collection()
            ps.delete_collection()
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

        st.info(f"Loaded {len(docs)} document pages. Chunking (parent + child)...")

        with st.spinner("Creating parent and child chunks..."):
            chunk_data = chunk_documents(docs)

        child_count = len(chunk_data["children"])
        parent_count = len(chunk_data["parents"])
        st.info(f"Created {child_count} child chunks + {parent_count} parent chunks. Embedding and storing...")

        progress = st.progress(0)
        embedding_fn = _get_embedding_fn()

        # Ingest children
        total_children = len(chunk_data["children"])
        batch_size = 100
        ingested = 0
        for start in range(0, total_children, batch_size):
            batch_children = chunk_data["children"][start:start + batch_size]
            ingest(batch_children, embedding_fn=embedding_fn, batch_size=len(batch_children))
            ingested += len(batch_children)
            progress.progress(ingested / (total_children + parent_count))

        # Ingest parents
        total_parents = len(chunk_data["parents"])
        from ingestion.embedder import _ingest_docs, get_parent_store
        parent_store = get_parent_store(embedding_fn)
        for start in range(0, total_parents, batch_size):
            batch_parents = chunk_data["parents"][start:start + batch_size]
            _ingest_docs(batch_parents, parent_store, embedding_fn, len(batch_parents), "parent")
            ingested += len(batch_parents)
            progress.progress(min(1.0, ingested / (total_children + parent_count)))

        progress.progress(1.0)
        st.success(f"Ingested {total_children} child + {total_parents} parent chunks.")
        st.rerun()
