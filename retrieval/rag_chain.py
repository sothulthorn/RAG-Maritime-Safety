"""RAG pipeline with query decomposition and answer verification.

Full pipeline:
1. Query decomposition (for complex questions)
2. Hybrid retrieval + parent expansion + compression
3. LLM generation with grounding prompt
4. Answer verification against source documents
5. Return verified answer with sources and confidence
"""

import re

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from config import LLM_MODEL, LLM_TEMPERATURE, RETRIEVAL_K, VERIFICATION_ENABLED
from retrieval.retriever import retrieve
from retrieval.verifier import verify_answer

SYSTEM_PROMPT = """You are a maritime safety expert assistant. Answer the question based ONLY on the provided context from official maritime safety documents and accident investigation reports.

Rules:
1. Provide a thorough, detailed answer using ALL relevant information from the context.
2. Structure your answer with clear sections or bullet points when comparing multiple topics.
3. For each claim, cite the specific source document (filename and page number when available).
4. When citing accident investigation findings, include the vessel name, incident type, and specific findings.
5. Include specific details: regulation numbers, vessel names, dates, casualty figures, and technical details from the documents.
6. If the context covers the topic partially, answer what you can and clearly state what aspects are not covered in the available documents.
7. Do not guess or extrapolate beyond what the documents state."""

CONTEXT_TEMPLATE = """Context:
{context}

Question: {question}

Provide a detailed, well-structured answer with specific citations from the source documents above. If comparing topics, organize your answer by each topic with supporting evidence:"""

DECOMPOSITION_PROMPT = """Break down the following maritime safety question into 2-3 simpler sub-questions that would help retrieve relevant information. Return ONLY the sub-questions, one per line, no numbering or bullets.

Question: {question}

Sub-questions:"""


def _format_context(docs) -> str:
    """Format retrieved documents into a labeled context string."""
    parts = []
    seen_content = set()
    for doc in docs:
        content_key = doc.page_content[:100]
        if content_key in seen_content:
            continue
        seen_content.add(content_key)

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")
        org = doc.metadata.get("organization", "")
        section = doc.metadata.get("section", "")

        label = f"[Source: {source}"
        if page:
            label += f", Page {page}"
        if org:
            label += f", Org: {org.upper()}"
        if section:
            label += f", Section: {section}"
        label += "]"
        parts.append(f"{label}\n{doc.page_content}")

    return "\n\n".join(parts)


def _extract_sources(docs) -> list[dict]:
    """Extract source metadata from retrieved documents."""
    sources = []
    seen = set()
    for doc in docs:
        source_key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:50])
        if source_key in seen:
            continue
        seen.add(source_key)

        source_info = {
            "source": doc.metadata.get("source", "Unknown"),
            "snippet": doc.page_content[:200],
        }
        if "page" in doc.metadata:
            source_info["page"] = doc.metadata["page"]
        if "organization" in doc.metadata:
            source_info["organization"] = doc.metadata["organization"].upper()
        if "section" in doc.metadata:
            source_info["section"] = doc.metadata["section"]
        sources.append(source_info)
    return sources


def _is_complex_query(query: str) -> bool:
    """Detect if a query likely needs decomposition."""
    complex_indicators = [
        r"\bcompare\b", r"\bdifference\b", r"\bvs\.?\b",
        r"\bbetween\b.*\band\b", r"\brelation\b",
        r"\bhow does .+ relate to\b", r"\bwhat are the .+ and .+\b",
    ]
    query_lower = query.lower()
    return any(re.search(p, query_lower) for p in complex_indicators)


def _decompose_query(query: str) -> list[str]:
    """Break a complex query into sub-queries using the LLM."""
    llm = ChatOllama(model=LLM_MODEL, temperature=0.0)
    messages = [
        HumanMessage(content=DECOMPOSITION_PROMPT.format(question=query)),
    ]
    response = llm.invoke(messages)
    sub_queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    return sub_queries[:3] if sub_queries else [query]


def answer_question(
    query: str,
    k: int = RETRIEVAL_K,
    source_filter: str | None = None,
) -> dict:
    """Run the full RAG pipeline with verification.

    Pipeline:
    1. Decompose complex queries into sub-queries
    2. Retrieve relevant chunks (hybrid + rerank + parent expansion + compression)
    3. Generate answer with grounding prompt
    4. Verify answer against source documents
    5. Return verified answer with confidence and sources

    Returns:
        dict with "answer", "sources", "confidence", "verified", "verification_details"
    """
    # 1. Retrieve (with optional decomposition)
    if _is_complex_query(query):
        sub_queries = _decompose_query(query)
        # Also include the original query for direct matches
        all_queries = [query] + sub_queries
        all_docs = []
        seen_content = set()
        for sq in all_queries:
            # Fetch more per sub-query for complex questions
            docs = retrieve(sq, k=k * 2, source_filter=source_filter)
            for doc in docs:
                key = doc.page_content[:100]
                if key not in seen_content:
                    seen_content.add(key)
                    all_docs.append(doc)
        docs = all_docs[:k * 3]  # allow more context for complex queries
    else:
        docs = retrieve(query, k=k, source_filter=source_filter)

    if not docs:
        return {
            "answer": "I couldn't find any relevant information in the document database. The database may be empty — check the sidebar to verify documents have been ingested.",
            "sources": [],
            "confidence": "LOW",
            "verified": False,
            "verification_details": "",
        }

    # 2. Format context and generate answer
    context = _format_context(docs)

    llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=CONTEXT_TEMPLATE.format(context=context, question=query)),
    ]
    response = llm.invoke(messages)
    raw_answer = response.content

    # 3. Verify answer against sources
    if VERIFICATION_ENABLED:
        verification = verify_answer(raw_answer, context)
        final_answer = verification["verified_answer"]
        confidence = verification["confidence"]
        was_verified = True
        verification_details = verification["verification_details"]
    else:
        final_answer = raw_answer
        confidence = "UNVERIFIED"
        was_verified = False
        verification_details = ""

    return {
        "answer": final_answer,
        "sources": _extract_sources(docs),
        "confidence": confidence,
        "verified": was_verified,
        "verification_details": verification_details,
    }
