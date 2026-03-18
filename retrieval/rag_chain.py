"""RAG pipeline: retrieve context, build prompt, call LLM."""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from config import LLM_MODEL, LLM_TEMPERATURE, RETRIEVAL_K
from retrieval.retriever import retrieve

SYSTEM_PROMPT = """You are a maritime safety expert assistant. Answer the question based ONLY on the provided context from official maritime safety documents. If the context does not contain enough information to answer, say so clearly.

For each claim in your answer, reference the source document and section when available."""

CONTEXT_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""


def _format_context(docs) -> str:
    """Format retrieved documents into a labeled context string."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")
        label = f"[Source: {source}"
        if page:
            label += f", Page {page}"
        label += "]"
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n".join(parts)


def _extract_sources(docs) -> list[dict]:
    """Extract source metadata from retrieved documents."""
    sources = []
    for doc in docs:
        source_info = {
            "source": doc.metadata.get("source", "Unknown"),
            "snippet": doc.page_content[:200],
        }
        if "page" in doc.metadata:
            source_info["page"] = doc.metadata["page"]
        sources.append(source_info)
    return sources


def answer_question(query: str, k: int = RETRIEVAL_K) -> dict:
    """Run the full RAG pipeline: retrieve, format, generate.

    Returns:
        dict with "answer" (str) and "sources" (list[dict])
    """
    # 1. Retrieve relevant chunks
    docs = retrieve(query, k=k)

    if not docs:
        return {
            "answer": "I couldn't find any relevant information in the ingested documents. Please make sure maritime safety documents have been uploaded.",
            "sources": [],
        }

    # 2. Format context
    context = _format_context(docs)

    # 3. Build prompt and call LLM
    llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=CONTEXT_TEMPLATE.format(context=context, question=query)),
    ]
    response = llm.invoke(messages)

    # 4. Return answer with sources
    return {
        "answer": response.content,
        "sources": _extract_sources(docs),
    }
