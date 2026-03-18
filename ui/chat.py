"""Chat interface components."""

import streamlit as st

from retrieval.rag_chain import answer_question


def render_chat():
    """Render the chat interface."""
    st.header("Maritime Safety Assistant")
    st.caption("Ask questions about maritime safety regulations and documents")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        source_label = src.get("source", "Unknown")
                        page = src.get("page")
                        if page:
                            source_label += f", Page {page}"
                        st.markdown(f"**{source_label}**")
                        st.text(src.get("snippet", ""))
                        st.divider()

    # Chat input
    if prompt := st.chat_input("Ask a maritime safety question..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    result = answer_question(prompt)
                    answer = result["answer"]
                    sources = result["sources"]
                except Exception as e:
                    answer = f"Error generating answer: {e}\n\nMake sure Ollama is running with the llama3 model (`ollama serve` and `ollama pull llama3`)."
                    sources = []

            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        source_label = src.get("source", "Unknown")
                        page = src.get("page")
                        if page:
                            source_label += f", Page {page}"
                        st.markdown(f"**{source_label}**")
                        st.text(src.get("snippet", ""))
                        st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
