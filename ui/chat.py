"""Chat interface components."""

import streamlit as st

from retrieval.rag_chain import answer_question


def render_chat():
    """Render the chat interface."""
    st.header("Maritime Safety Assistant")
    st.caption("Ask questions about maritime safety regulations and accident investigation reports")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_metadata(msg)

    # Chat input
    if prompt := st.chat_input("Ask a maritime safety question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    source_filter = st.session_state.get("source_filter")
                    result = answer_question(prompt, source_filter=source_filter)
                    answer = result["answer"]
                    sources = result["sources"]
                    confidence = result.get("confidence", "")
                    verified = result.get("verified", False)
                    verification_details = result.get("verification_details", "")
                except Exception as e:
                    answer = f"Error generating answer: {e}\n\nMake sure Ollama is running with the llama3 model (`ollama serve` and `ollama pull llama3`)."
                    sources = []
                    confidence = ""
                    verified = False
                    verification_details = ""

            st.markdown(answer)

            msg_data = {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "confidence": confidence,
                "verified": verified,
                "verification_details": verification_details,
            }
            _render_metadata(msg_data)

        st.session_state.messages.append(msg_data)


def _render_metadata(msg: dict):
    """Render confidence badge, sources, and verification details."""
    confidence = msg.get("confidence", "")
    verified = msg.get("verified", False)
    sources = msg.get("sources", [])
    verification_details = msg.get("verification_details", "")

    # Confidence badge
    if confidence:
        color_map = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red", "UNVERIFIED": "gray"}
        color = color_map.get(confidence, "gray")
        badge = f":{color}[Confidence: {confidence}]"
        if verified:
            badge += " | :blue[Verified]"
        st.caption(badge)

    # Sources
    if sources:
        with st.expander("Sources"):
            for src in sources:
                _render_source(src)

    # Verification details
    if verification_details and verified:
        with st.expander("Verification Details"):
            st.text(verification_details)


def _render_source(src: dict):
    """Render a single source entry."""
    source_label = src.get("source", "Unknown")
    page = src.get("page")
    org = src.get("organization", "")
    section = src.get("section", "")

    header = f"**{source_label}**"
    if page:
        header += f" — Page {page}"
    if org:
        header += f" ({org})"

    st.markdown(header)
    if section:
        st.caption(f"Section: {section}")
    st.text(src.get("snippet", ""))
    st.divider()
