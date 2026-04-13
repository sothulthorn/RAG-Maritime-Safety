"""Explainable RAG pipeline with query decomposition, verification, and explanation.

Full pipeline:
1. Query decomposition (for complex questions)
2. Hybrid retrieval + parent expansion + compression
3. LLM generation with Chain-of-Thought grounding prompt
4. Answer verification against source documents
5. Explainability: evidence citations + reasoning traces
6. Return explainable answer with sources, confidence, evidence, and reasoning
"""

import re

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from config import (
    LLM_MODEL, LLM_TEMPERATURE, RETRIEVAL_K,
    VERIFICATION_ENABLED, EXPLAINABILITY_ENABLED,
)
from retrieval.retriever import retrieve
from retrieval.verifier import verify_answer
from retrieval.explainer import generate_explanation

SYSTEM_PROMPT = """You are a maritime safety expert assistant. Answer the question based ONLY on the provided context from official maritime safety documents and accident investigation reports.

Rules:
1. Provide a thorough, detailed answer using ALL relevant information from the context.
2. Structure your answer with clear sections or bullet points when comparing multiple topics.
3. Use footnote-style citations: place a numbered reference like [1], [2], etc. at the end of each sentence or claim that comes from a source document. Do NOT add a References section at the end — the citation details are handled separately by the system.
4. When citing accident investigation findings, include the vessel name, incident type, and specific findings in the answer text — but do not mention report filenames inline.
5. Include specific details: regulation numbers, vessel names, dates, casualty figures, and technical details from the documents.
6. If the context covers the topic partially, answer what you can and clearly state what aspects are not covered in the available documents.
7. Do not guess or extrapolate beyond what the documents state.
8. Use Chain-of-Thought reasoning: think through the question step by step before providing your final answer.

Example format:
Improper loading is a primary factor in the capsizing of fishing vessels [1]. Crews often lack formal training in vessel stability [2]. Vessel modifications without reassessment can raise the centre of gravity significantly [3]."""

CONTEXT_TEMPLATE = """Context:
{context}

Question: {question}

Think through this step by step using the source documents, then provide a detailed, well-structured answer with footnote-style citations [1], [2], etc. at the end of sentences. Do NOT add a References section — the system handles that separately. If comparing topics, organize your answer by each topic with supporting evidence:"""

DECOMPOSITION_PROMPT = """Break down the following maritime safety question into 2-3 simpler sub-questions that would help retrieve relevant information. Return ONLY the sub-questions, one per line, no numbering or bullets.

Question: {question}

Sub-questions:"""


def _clean_answer(text: str) -> str:
    """Clean LLM answer: strip references section and inline filename citations."""
    # 1. Strip References/Sources/Citations/Note section at the end
    result = re.split(
        r"\n\s*\*{0,2}(?:References?|Sources?|Citations?|Note)\s*:?\s*\*{0,2}\s*(?:\n|$)",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    # 2. Strip trailing numbered lists that look like references
    result = re.sub(
        r"\n(?:\d+\.\s+(?:\[?\w+[\w\s\-]*(?:\.pdf)?\]?[,\s]*(?:Page|p\.?)\s*\d+.*?)(?:\n|$))+\s*$",
        "",
        result,
        flags=re.IGNORECASE,
    )

    # 3. Replace inline filename citations like [M98C0026.pdf, Page 8] or
    #    [Source: filename, Page X] with clean footnote numbers [n]
    counter = {"n": 0}
    seen = {}

    def _replace_cite(match):
        key = match.group(0).lower()
        if key not in seen:
            counter["n"] += 1
            seen[key] = counter["n"]
        return f"[{seen[key]}]"

    result = re.sub(
        r"\[(?:Source:\s*)?[\w\-]+\.pdf(?:,?\s*(?:Page|p\.?)\s*\d+)?\]",
        _replace_cite,
        result,
        flags=re.IGNORECASE,
    )

    # 4. Also catch patterns like "the TSB report [M98C0026.pdf, Page 8]" where
    #    the brackets contain a filename without the [Source:] prefix
    result = re.sub(
        r"\[[\w\-]+\.pdf\]",
        _replace_cite,
        result,
        flags=re.IGNORECASE,
    )

    return result.rstrip()


def _build_evidence_from_docs(docs) -> list[dict]:
    """Build evidence list directly from retrieved documents.

    Each document becomes an evidence entry with its number matching
    the [1], [2] context labels used in the answer.
    """
    evidence = []
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

        source_label = source
        if page:
            source_label += f", Page {page}"
        if org:
            source_label += f" ({org.upper()})"
        if section:
            source_label += f", Section: {section}"

        # Use first 150 chars as the quote snippet
        quote = doc.page_content[:150].strip()
        if len(doc.page_content) > 150:
            quote += "..."

        evidence.append({
            "claim": f"Source document: {source}",
            "source": source_label,
            "quote": quote,
        })

    return evidence


def _format_context(docs) -> str:
    """Format retrieved documents into a numbered context string.

    Uses [1], [2], etc. so the LLM naturally cites with footnote numbers.
    """
    parts = []
    seen_content = set()
    num = 0
    for doc in docs:
        content_key = doc.page_content[:100]
        if content_key in seen_content:
            continue
        seen_content.add(content_key)
        num += 1

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")
        org = doc.metadata.get("organization", "")
        section = doc.metadata.get("section", "")

        meta_parts = [source]
        if page:
            meta_parts.append(f"Page {page}")
        if org:
            meta_parts.append(org.upper())
        if section:
            meta_parts.append(f"Section: {section}")
        meta = ", ".join(meta_parts)

        parts.append(f"[{num}] ({meta})\n{doc.page_content}")

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
    """Run the full Explainable RAG pipeline.

    Pipeline:
    1. Decompose complex queries into sub-queries
    2. Retrieve relevant chunks (hybrid + rerank + parent expansion + compression)
    3. Generate answer with Chain-of-Thought grounding prompt
    4. Verify answer against source documents
    5. Generate explainability output (evidence + reasoning traces)
    6. Return explainable answer with confidence, sources, evidence, and reasoning

    Returns:
        dict with:
        - answer: the final answer text
        - sources: list of source metadata
        - confidence: HIGH/MEDIUM/LOW/UNVERIFIED
        - verified: bool
        - verification_details: raw verification output
        - evidence: list of evidence citations (claim -> source -> quote)
        - unsupported_claims: claims not found in sources
        - reasoning_steps: Chain-of-Thought reasoning steps
        - key_principles: maritime safety principles underpinning the answer
        - explanation_raw: raw explanation output for debugging
    """
    # 1. Retrieve (with optional decomposition)
    if _is_complex_query(query):
        sub_queries = _decompose_query(query)
        all_queries = [query] + sub_queries
        all_docs = []
        seen_content = set()
        for sq in all_queries:
            docs = retrieve(sq, k=k * 2, source_filter=source_filter)
            for doc in docs:
                key = doc.page_content[:100]
                if key not in seen_content:
                    seen_content.add(key)
                    all_docs.append(doc)
        docs = all_docs[:k * 3]
    else:
        docs = retrieve(query, k=k, source_filter=source_filter)

    if not docs:
        return {
            "answer": "I couldn't find any relevant information in the document database. The database may be empty — check the sidebar to verify documents have been ingested.",
            "sources": [],
            "confidence": "LOW",
            "verified": False,
            "verification_details": "",
            "evidence": [],
            "unsupported_claims": [],
            "reasoning_steps": [],
            "key_principles": [],
            "explanation_raw": "",
        }

    # 2. Format context and generate answer
    context = _format_context(docs)
    sources = _extract_sources(docs)

    llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=CONTEXT_TEMPLATE.format(context=context, question=query)),
    ]
    response = llm.invoke(messages)
    raw_answer = _clean_answer(response.content)

    # 3. Verify answer against sources
    if VERIFICATION_ENABLED:
        verification = verify_answer(raw_answer, context)
        final_answer = _clean_answer(verification["verified_answer"])
        confidence = verification["confidence"]
        was_verified = True
        verification_details = verification["verification_details"]
    else:
        final_answer = raw_answer
        confidence = "UNVERIFIED"
        was_verified = False
        verification_details = ""

    # 4. Build evidence from retrieved docs (deterministic, not LLM-dependent)
    evidence = _build_evidence_from_docs(docs)

    # 5. Generate reasoning traces
    if EXPLAINABILITY_ENABLED:
        explanation = generate_explanation(
            answer=final_answer,
            context=context,
            question=query,
            sources=sources,
        )
        reasoning_steps = explanation["reasoning_steps"]
        key_principles = explanation["key_principles"]
        explanation_raw = explanation.get("reasoning_raw", "")
    else:
        reasoning_steps = []
        key_principles = []
        explanation_raw = ""

    return {
        "answer": final_answer,
        "sources": sources,
        "confidence": confidence,
        "verified": was_verified,
        "verification_details": verification_details,
        "evidence": evidence,
        "unsupported_claims": [],
        "reasoning_steps": reasoning_steps,
        "key_principles": key_principles,
        "explanation_raw": explanation_raw,
    }
