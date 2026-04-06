"""Explainability module: generates evidence citations and reasoning traces.

After the LLM generates an answer, this module produces:
1. Evidence/Citations — maps each claim to specific source documents
2. Reasoning Traces — Chain-of-Thought logic explaining why the answer was given

This is the core of the Explainable RAG system: not just WHAT the answer is,
but WHY that answer was given and WHERE it comes from.
"""

import re

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from config import LLM_MODEL


EVIDENCE_PROMPT = """You are a maritime safety citation analyst. Your job is to trace each claim in an answer back to specific source documents.

SOURCE DOCUMENTS:
{context}

QUESTION: {question}

ANSWER TO ANALYZE:
{answer}

For each key claim or fact in the answer, identify the specific source document that supports it.

Respond in this EXACT format:
EVIDENCE:
- Claim: [specific claim from the answer] | Source: [document name, page/section] | Quote: [brief relevant quote from source]
- Claim: [specific claim from the answer] | Source: [document name, page/section] | Quote: [brief relevant quote from source]
(list all claims)

UNSUPPORTED_CLAIMS:
- [any claims not found in sources, or "None"]"""


REASONING_PROMPT = """You are a maritime safety reasoning analyst. Explain the Chain-of-Thought logic behind an answer about maritime safety.

QUESTION: {question}

ANSWER: {answer}

RETRIEVED SOURCES:
{sources_summary}

Explain step-by-step WHY this answer was given. Focus on:
1. Why the retrieved documents are relevant to the question
2. How the key conclusions were derived from the evidence
3. What maritime safety principles or regulations connect the evidence to the answer

Respond in this EXACT format:
REASONING_TRACE:
- Step 1: [first reasoning step explaining relevance of retrieved documents]
- Step 2: [how key facts were extracted and connected]
- Step 3: [how the final answer was synthesized from evidence]
(add more steps if needed, but keep each step concise)

KEY_PRINCIPLES:
- [maritime safety principle or regulation that underpins the answer]
(list 1-3 key principles)"""


def generate_evidence(answer: str, context: str, question: str) -> dict:
    """Extract evidence citations linking answer claims to source documents.

    Args:
        answer: The LLM-generated answer.
        context: The formatted context from retrieved documents.
        question: The original user question.

    Returns:
        dict with:
        - evidence: list of dicts with claim, source, quote
        - unsupported_claims: list of claims without source support
        - raw_output: full LLM response
    """
    if not context or not answer:
        return {
            "evidence": [],
            "unsupported_claims": [],
            "raw_output": "No context available for evidence extraction",
        }

    llm = ChatOllama(model=LLM_MODEL, temperature=0.0)

    prompt = EVIDENCE_PROMPT.format(
        context=context[:4000],
        question=question,
        answer=answer,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        result_text = response.content

        evidence = _parse_evidence(result_text)
        unsupported = _parse_unsupported(result_text)

        return {
            "evidence": evidence,
            "unsupported_claims": unsupported,
            "raw_output": result_text,
        }
    except Exception as e:
        return {
            "evidence": [],
            "unsupported_claims": [],
            "raw_output": f"Evidence extraction failed: {e}",
        }


def generate_reasoning_trace(
    answer: str, question: str, sources: list[dict],
) -> dict:
    """Generate Chain-of-Thought reasoning trace for the answer.

    Args:
        answer: The LLM-generated answer.
        question: The original user question.
        sources: List of source metadata dicts.

    Returns:
        dict with:
        - reasoning_steps: list of reasoning step strings
        - key_principles: list of maritime safety principles
        - raw_output: full LLM response
    """
    if not answer:
        return {
            "reasoning_steps": [],
            "key_principles": [],
            "raw_output": "No answer available for reasoning trace",
        }

    sources_summary = _format_sources_summary(sources)

    llm = ChatOllama(model=LLM_MODEL, temperature=0.0)

    prompt = REASONING_PROMPT.format(
        question=question,
        answer=answer,
        sources_summary=sources_summary,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        result_text = response.content

        steps = _parse_reasoning_steps(result_text)
        principles = _parse_key_principles(result_text)

        return {
            "reasoning_steps": steps,
            "key_principles": principles,
            "raw_output": result_text,
        }
    except Exception as e:
        return {
            "reasoning_steps": [],
            "key_principles": [],
            "raw_output": f"Reasoning trace failed: {e}",
        }


def generate_explanation(
    answer: str, context: str, question: str, sources: list[dict],
) -> dict:
    """Generate full explainability output: evidence + reasoning trace.

    This is the main entry point for the explainer module.

    Returns:
        dict with:
        - evidence: list of evidence citations
        - unsupported_claims: claims without source backing
        - reasoning_steps: Chain-of-Thought steps
        - key_principles: underpinning maritime principles
        - evidence_raw: raw evidence LLM output
        - reasoning_raw: raw reasoning LLM output
    """
    evidence_result = generate_evidence(answer, context, question)
    reasoning_result = generate_reasoning_trace(answer, question, sources)

    return {
        "evidence": evidence_result["evidence"],
        "unsupported_claims": evidence_result["unsupported_claims"],
        "reasoning_steps": reasoning_result["reasoning_steps"],
        "key_principles": reasoning_result["key_principles"],
        "evidence_raw": evidence_result["raw_output"],
        "reasoning_raw": reasoning_result["raw_output"],
    }


# --- Parsing helpers ---

def _parse_evidence(text: str) -> list[dict]:
    """Parse evidence entries from LLM output."""
    evidence = []
    pattern = re.compile(
        r"-\s*Claim:\s*(.+?)\s*\|\s*Source:\s*(.+?)\s*\|\s*Quote:\s*(.+)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        evidence.append({
            "claim": match.group(1).strip(),
            "source": match.group(2).strip(),
            "quote": match.group(3).strip(),
        })
    return evidence


def _parse_unsupported(text: str) -> list[str]:
    """Parse unsupported claims from LLM output."""
    unsupported = []
    section_match = re.search(
        r"UNSUPPORTED_CLAIMS:\s*\n(.*?)(?:\n\n|\Z)", text, re.DOTALL | re.IGNORECASE,
    )
    if not section_match:
        return []

    section = section_match.group(1)
    for line in section.strip().split("\n"):
        line = line.strip().lstrip("- ").strip()
        if line and line.lower() != "none":
            unsupported.append(line)
    return unsupported


def _parse_reasoning_steps(text: str) -> list[str]:
    """Parse reasoning steps from LLM output."""
    steps = []
    pattern = re.compile(r"-\s*Step\s+\d+:\s*(.+)", re.IGNORECASE)
    for match in pattern.finditer(text):
        steps.append(match.group(1).strip())
    return steps


def _parse_key_principles(text: str) -> list[str]:
    """Parse key principles from LLM output."""
    principles = []
    section_match = re.search(
        r"KEY_PRINCIPLES:\s*\n(.*?)(?:\n\n|\Z)", text, re.DOTALL | re.IGNORECASE,
    )
    if not section_match:
        return []

    section = section_match.group(1)
    for line in section.strip().split("\n"):
        line = line.strip().lstrip("- ").strip()
        if line:
            principles.append(line)
    return principles


def _format_sources_summary(sources: list[dict]) -> str:
    """Format sources list into a readable summary."""
    if not sources:
        return "No sources available."

    parts = []
    for i, src in enumerate(sources, 1):
        label = src.get("source", "Unknown")
        page = src.get("page")
        org = src.get("organization", "")
        section = src.get("section", "")

        entry = f"- Doc{i}: {label}"
        if page:
            entry += f" (Page {page})"
        if org:
            entry += f" [{org}]"
        if section:
            entry += f" — {section}"
        parts.append(entry)
    return "\n".join(parts)
