"""
Evaluation metrics for comparing RAG vs plain LLM answers.

Metrics:
1. Key Fact Coverage  — what % of expected facts appear in the answer
2. Citation Accuracy  — do cited sources actually exist in the database
3. Hallucination Score — does the answer invent facts not in the context
4. Answer Relevance   — does the answer address the question
5. Faithfulness       — is the answer grounded in retrieved context
"""

import re

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from config import LLM_MODEL


def key_fact_coverage(answer: str, key_facts: list[str]) -> float:
    """Check what fraction of expected key facts appear in the answer.

    Returns a score between 0.0 and 1.0.
    """
    if not key_facts:
        return 1.0

    answer_lower = answer.lower()
    hits = sum(1 for fact in key_facts if fact.lower() in answer_lower)
    return hits / len(key_facts)


def citation_accuracy(sources: list[dict], source_keywords: list[str]) -> float:
    """Check if cited sources contain expected keywords.

    A source is considered valid if at least one expected keyword appears
    in the source filename. Returns fraction of keywords matched.
    """
    if not source_keywords:
        # No specific sources expected — score based on whether ANY sources were cited
        return 1.0 if sources else 0.0

    source_text = " ".join(
        s.get("source", "") + " " + s.get("snippet", "")
        for s in sources
    ).lower()

    hits = sum(1 for kw in source_keywords if kw.lower() in source_text)
    return hits / len(source_keywords)


def source_grounding(answer: str, sources: list[dict]) -> float:
    """Check if the answer references information that appears in the source snippets.

    Extracts named entities / specific terms from the answer and checks
    if they appear in the source material.
    """
    if not sources or not answer:
        return 0.0

    source_text = " ".join(s.get("snippet", "") for s in sources).lower()

    # Extract specific terms from answer (capitalized words, numbers, vessel names)
    specific_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer)
    specific_terms += re.findall(r'\b\d+\b', answer)

    if not specific_terms:
        return 0.5  # neutral if no specific terms to check

    grounded = sum(1 for term in specific_terms if term.lower() in source_text)
    return min(1.0, grounded / max(len(specific_terms), 1))


def llm_faithfulness(question: str, answer: str, context: str) -> dict:
    """Use LLM-as-judge to evaluate faithfulness and hallucination.

    Returns dict with:
    - faithfulness_score: 0.0 to 1.0
    - hallucination_score: 0.0 to 1.0 (lower is better)
    - relevance_score: 0.0 to 1.0
    - reasoning: explanation
    """
    judge_prompt = f"""You are an impartial evaluator assessing the quality of an AI answer about maritime safety.

CONTEXT (retrieved documents):
{context[:3000]}

QUESTION: {question}

ANSWER BEING EVALUATED:
{answer}

Evaluate the answer on these criteria. For each, give a score from 0 to 10:

1. FAITHFULNESS: Is every claim in the answer supported by the context? (10 = fully grounded, 0 = completely made up)
2. HALLUCINATION: Does the answer contain facts NOT present in the context? (10 = no hallucination, 0 = entirely fabricated)
3. RELEVANCE: Does the answer actually address the question asked? (10 = perfectly relevant, 0 = completely off-topic)

Respond in EXACTLY this format (numbers only, no extra text):
FAITHFULNESS: <score>
HALLUCINATION: <score>
RELEVANCE: <score>
REASONING: <one line explanation>"""

    try:
        llm = ChatOllama(model=LLM_MODEL, temperature=0.0)
        response = llm.invoke([HumanMessage(content=judge_prompt)])
        text = response.content

        faithfulness = _extract_score(text, "FAITHFULNESS")
        hallucination = _extract_score(text, "HALLUCINATION")
        relevance = _extract_score(text, "RELEVANCE")

        reasoning_match = re.search(r"REASONING:\s*(.+)", text, re.IGNORECASE)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        return {
            "faithfulness_score": faithfulness / 10.0,
            "hallucination_score": hallucination / 10.0,
            "relevance_score": relevance / 10.0,
            "reasoning": reasoning,
        }
    except Exception as e:
        return {
            "faithfulness_score": 0.0,
            "hallucination_score": 0.0,
            "relevance_score": 0.0,
            "reasoning": f"Evaluation error: {e}",
        }


def _extract_score(text: str, label: str) -> float:
    """Extract a numeric score after a label."""
    match = re.search(rf"{label}:\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return min(10.0, max(0.0, float(match.group(1))))
    return 0.0


def compute_all_metrics(
    question: str,
    answer: str,
    sources: list[dict],
    key_facts: list[str],
    source_keywords: list[str],
    context: str = "",
    use_llm_judge: bool = True,
) -> dict:
    """Compute all metrics for a single question-answer pair.

    Returns a dict with all metric scores.
    """
    results = {
        "key_fact_coverage": key_fact_coverage(answer, key_facts),
        "citation_accuracy": citation_accuracy(sources, source_keywords),
        "source_grounding": source_grounding(answer, sources),
    }

    if use_llm_judge and context:
        llm_scores = llm_faithfulness(question, answer, context)
        results.update(llm_scores)
    else:
        results.update({
            "faithfulness_score": 0.0,
            "hallucination_score": 0.0,
            "relevance_score": 0.0,
            "reasoning": "LLM judge skipped",
        })

    return results
