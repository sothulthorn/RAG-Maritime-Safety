"""Chat interface components for Explainable RAG."""

import streamlit as st

from retrieval.rag_chain import answer_question


def render_chat():
    """Render the chat interface."""
    st.header("Explainable Maritime Safety Assistant")
    st.caption("Ask questions about maritime safety — get answers with evidence, reasoning, and source citations")

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
            with st.spinner("Searching documents, generating answer, and building explanation..."):
                try:
                    source_filter = st.session_state.get("source_filter")
                    result = answer_question(prompt, source_filter=source_filter)
                    answer = result["answer"]
                    sources = result["sources"]
                    confidence = result.get("confidence", "")
                    verified = result.get("verified", False)
                    verification_details = result.get("verification_details", "")
                    evidence = result.get("evidence", [])
                    unsupported_claims = result.get("unsupported_claims", [])
                    reasoning_steps = result.get("reasoning_steps", [])
                    key_principles = result.get("key_principles", [])
                except Exception as e:
                    answer = f"Error generating answer: {e}\n\nMake sure Ollama is running with the llama3 model (`ollama serve` and `ollama pull llama3`)."
                    sources = []
                    confidence = ""
                    verified = False
                    verification_details = ""
                    evidence = []
                    unsupported_claims = []
                    reasoning_steps = []
                    key_principles = []

            st.markdown(answer)

            msg_data = {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "confidence": confidence,
                "verified": verified,
                "verification_details": verification_details,
                "evidence": evidence,
                "unsupported_claims": unsupported_claims,
                "reasoning_steps": reasoning_steps,
                "key_principles": key_principles,
            }
            _render_metadata(msg_data)

        st.session_state.messages.append(msg_data)


def _render_metadata(msg: dict):
    """Render confidence, evidence, reasoning, sources, and verification."""
    confidence = msg.get("confidence", "")
    verified = msg.get("verified", False)
    sources = msg.get("sources", [])
    verification_details = msg.get("verification_details", "")
    evidence = msg.get("evidence", [])
    unsupported_claims = msg.get("unsupported_claims", [])
    reasoning_steps = msg.get("reasoning_steps", [])
    key_principles = msg.get("key_principles", [])

    # Confidence badge
    if confidence:
        color_map = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red", "UNVERIFIED": "gray"}
        color = color_map.get(confidence, "gray")
        badge = f":{color}[Confidence: {confidence}]"
        if verified:
            badge += " | :blue[Verified]"
        st.caption(badge)

    # Evidence / Citations
    if evidence:
        with st.expander("Evidence / Citations"):
            for ev in evidence:
                claim = ev.get("claim", "")
                source = ev.get("source", "")
                quote = ev.get("quote", "")
                st.markdown(f"**Claim:** {claim}")
                st.markdown(f"**Source:** {source}")
                if quote:
                    st.caption(f'"{quote}"')
                st.divider()

            if unsupported_claims:
                st.warning("**Unsupported Claims:**")
                for claim in unsupported_claims:
                    st.caption(f"- {claim}")

    # Reasoning Trace
    if reasoning_steps:
        with st.expander("Reasoning Trace"):
            for i, step in enumerate(reasoning_steps, 1):
                st.markdown(f"**Step {i}:** {step}")

            if key_principles:
                st.divider()
                st.markdown("**Key Maritime Safety Principles:**")
                for principle in key_principles:
                    st.markdown(f"- {principle}")

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
