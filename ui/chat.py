"""Chat interface with expander-based explainability (native Streamlit components only)."""

import streamlit as st

from retrieval.rag_chain import answer_question


def render_chat():
    """Render the chat interface with inline expanders per message."""
    st.header("Explainable Maritime Safety Assistant")
    st.caption("Ask questions about maritime safety — get answers with evidence, reasoning, and source citations")

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
                msg_data = _generate_answer(prompt)

            st.markdown(msg_data["content"])
            _render_metadata(msg_data)

        st.session_state.messages.append(msg_data)


def _generate_answer(prompt: str) -> dict:
    """Call the RAG pipeline and return the message data dict."""
    try:
        source_filter = st.session_state.get("source_filter")
        result = answer_question(prompt, source_filter=source_filter)
        return {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "confidence": result.get("confidence", ""),
            "verified": result.get("verified", False),
            "verification_details": result.get("verification_details", ""),
            "evidence": result.get("evidence", []),
            "unsupported_claims": result.get("unsupported_claims", []),
            "reasoning_steps": result.get("reasoning_steps", []),
            "key_principles": result.get("key_principles", []),
        }
    except Exception as e:
        return {
            "role": "assistant",
            "content": (
                f"Error generating answer: {e}\n\n"
                "Make sure Ollama is running with the llama3 model "
                "(`ollama serve` and `ollama pull llama3`)."
            ),
            "sources": [],
            "confidence": "",
            "verified": False,
            "verification_details": "",
            "evidence": [],
            "unsupported_claims": [],
            "reasoning_steps": [],
            "key_principles": [],
        }


def _render_metadata(msg: dict):
    """Render confidence badge and expanders for evidence, reasoning, sources."""
    confidence = msg.get("confidence", "")
    verified = msg.get("verified", False)
    sources = msg.get("sources", [])
    evidence = msg.get("evidence", [])
    unsupported_claims = msg.get("unsupported_claims", [])
    reasoning_steps = msg.get("reasoning_steps", [])
    key_principles = msg.get("key_principles", [])
    # Confidence badge
    if confidence:
        color_map = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}
        color = color_map.get(confidence, "gray")
        dots = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(confidence, 0)
        dot_display = "●" * dots + "○" * (3 - dots)

        parts = [f":{color}[{dot_display} {confidence}]"]
        if verified:
            parts.append(":blue[✓ Verified]")
        parts.append(f"{len(sources)} sources")
        st.caption(" · ".join(parts))

    # Evidence / Citations
    if evidence:
        with st.expander(f"Evidence / Citations ({len(evidence)} claims)"):
            for i, ev in enumerate(evidence, 1):
                claim = ev.get("claim", "")
                source = ev.get("source", "")
                quote = ev.get("quote", "")
                st.markdown(f"**[{i}]** {claim}")
                st.caption(f"Source: {source}")
                if quote:
                    st.markdown(f"> *\"{quote}\"*")
                st.divider()

            if unsupported_claims:
                st.warning("**Unsupported Claims**")
                for claim in unsupported_claims:
                    st.caption(f"- {claim}")

    # Reasoning Trace
    if reasoning_steps:
        with st.expander("Reasoning Trace"):
            for i, step in enumerate(reasoning_steps, 1):
                st.markdown(f"**Step {i}:** {step}")

            if key_principles:
                st.divider()
                st.markdown("**Key Maritime Safety Principles**")
                for principle in key_principles:
                    st.markdown(f"- {principle}")


