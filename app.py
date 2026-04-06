"""Explainable RAG for Maritime Safety — Streamlit entry point."""

import streamlit as st

from ui.sidebar import render_sidebar
from ui.chat import render_chat

st.set_page_config(
    page_title="Explainable Maritime Safety RAG",
    page_icon="🚢",
    layout="wide",
)

render_sidebar()
render_chat()
