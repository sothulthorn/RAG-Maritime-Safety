"""
Evaluation metrics for the Explainable RAG system.

Four core metrics aligned with the Explainable RAG evaluation framework:
1. Retrieval Accuracy  — Is the retrieved document actually relevant? (Recall/Precision)
2. Answer Correctness  — Is the generated answer factually correct? (Ground Truth + LLM Judge)
3. Faithfulness        — Is the answer strictly based on retrieved documents? (No hallucinations)
4. Explainability Score — Is the logic clear and are citations accurate?
"""

import re

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from config import LLM_MODEL


# ---------------------------------------------------------------------------
# 1. Retrieval Accuracy
# ---------------------------------------------------------------------------

def retrieval_accuracy(
    sources: list[dict],
    key_facts: list[str],
    source_keywords: list[str],
) -> dict:
    """Measure retrieval quality via recall and precision proxies.

    Recall proxy: what fraction of expected source keywords appear in retrieved sources.
    Precision proxy: what fraction of retrieved sources contain at least one expected keyword.

    Returns dict with recall, precision, and combined f1 score.
    """
    if not source_keywords:
        has_sources = 1.0 if sources else 0.0
        return {"recall": has_sources, "precision": has_sources, "f1": has_sources}

    source_text = " ".join(
        s.get("source", "") + " " + s.get("snippet", "")
        for s in sources
    ).lower()

    # Recall: fraction of expected keywords found in retrieved sources
    keyword_hits = sum(1 for kw in source_keywords if kw.lower() in source_text)
    recall = keyword_hits / len(source_keywords)

    # Precision: fraction of retrieved sources that contain at least one keyword
    if not sources:
        precision = 0.0
    else:
        relevant_sources = 0
        for s in sources:
            s_text = (s.get("source", "") + " " + s.get("snippet", "")).lower()
            if any(kw.lower() in s_text for kw in source_keywords):
                relevant_sources += 1
        precision = relevant_sources / len(sources)

    # F1 score
    if recall + precision > 0:
        f1 = 2 * (recall * precision) / (recall + precision)
    else:
        f1 = 0.0

    return {"recall": round(recall, 3), "precision": round(precision, 3), "f1": round(f1, 3)}


# ---------------------------------------------------------------------------
# 2. Answer Correctness
# ---------------------------------------------------------------------------

def answer_correctness(
    question: str,
    answer: str,
    key_facts: list[str],
    context: str = "",
    use_llm_judge: bool = True,
) -> dict:
    """Evaluate factual correctness of the answer.

    Combines:
    - Key fact coverage: fraction of expected facts present in the answer
    - LLM judge score: how correct the answer is given context + question

    Returns dict with fact_coverage, llm_score, and combined score.
    """
    # Key fact coverage
    if key_facts:
        answer_lower = answer.lower()
        hits = sum(1 for fact in key_facts if fact.lower() in answer_lower)
        fact_coverage = hits / len(key_facts)
    else:
        fact_coverage = 1.0

    # LLM judge correctness
    llm_score = 0.0
    llm_reasoning = ""
    if use_llm_judge and context:
        judge_result = _llm_judge_correctness(question, answer, context)
        llm_score = judge_result["score"]
        llm_reasoning = judge_result["reasoning"]

    # Combined score (weighted average: 40% fact coverage + 60% LLM judge)
    if use_llm_judge and context:
        combined = 0.4 * fact_coverage + 0.6 * llm_score
    else:
        combined = fact_coverage

    return {
        "fact_coverage": round(fact_coverage, 3),
        "llm_score": round(llm_score, 3),
        "combined": round(combined, 3),
        "reasoning": llm_reasoning,
    }


def _llm_judge_correctness(question: str, answer: str, context: str) -> dict:
    """Use LLM-as-judge to score answer correctness."""
    prompt = f"""You are an impartial evaluator. Score the correctness of this maritime safety answer.

CONTEXT (retrieved documents):
{context[:3000]}

QUESTION: {question}

ANSWER:
{answer}

Score the answer's factual correctness from 0 to 10:
- 10 = Every fact is accurate and well-supported by the context
- 5 = Some facts are correct but others are wrong or missing
- 0 = The answer is completely incorrect

Respond in EXACTLY this format:
CORRECTNESS: <score>
REASONING: <one line explanation>"""

    try:
        llm = ChatOllama(model=LLM_MODEL, temperature=0.0)
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content

        score = _extract_score(text, "CORRECTNESS") / 10.0
        reasoning_match = re.search(r"REASONING:\s*(.+)", text, re.IGNORECASE)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        return {"score": score, "reasoning": reasoning}
    except Exception as e:
        return {"score": 0.0, "reasoning": f"Judge error: {e}"}


# ---------------------------------------------------------------------------
# 3. Faithfulness
# ---------------------------------------------------------------------------

def faithfulness(
    question: str,
    answer: str,
    context: str = "",
    use_llm_judge: bool = True,
) -> dict:
    """Evaluate whether the answer is strictly based on retrieved documents.

    Checks for hallucinations — facts in the answer that are NOT in the context.

    Returns dict with faithfulness_score, hallucination_score, and reasoning.
    """
    if not use_llm_judge or not context:
        # Fall back to source grounding heuristic
        return {
            "faithfulness_score": 0.0,
            "hallucination_score": 0.0,
            "combined": 0.0,
            "reasoning": "LLM judge skipped or no context",
        }

    prompt = f"""You are an impartial evaluator assessing whether an AI answer is faithful to its source documents.

CONTEXT (retrieved documents):
{context[:3000]}

QUESTION: {question}

ANSWER BEING EVALUATED:
{answer}

Evaluate on these two criteria (score 0-10 each):

1. FAITHFULNESS: Is every claim in the answer supported by the context? (10 = fully grounded, 0 = completely made up)
2. HALLUCINATION: Does the answer contain facts NOT present in the context? (10 = no hallucination at all, 0 = entirely fabricated)

Respond in EXACTLY this format:
FAITHFULNESS: <score>
HALLUCINATION: <score>
REASONING: <one line explanation>"""

    try:
        llm = ChatOllama(model=LLM_MODEL, temperature=0.0)
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content

        faith_score = _extract_score(text, "FAITHFULNESS") / 10.0
        halluc_score = _extract_score(text, "HALLUCINATION") / 10.0
        reasoning_match = re.search(r"REASONING:\s*(.+)", text, re.IGNORECASE)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        combined = (faith_score + halluc_score) / 2.0

        return {
            "faithfulness_score": round(faith_score, 3),
            "hallucination_score": round(halluc_score, 3),
            "combined": round(combined, 3),
            "reasoning": reasoning,
        }
    except Exception as e:
        return {
            "faithfulness_score": 0.0,
            "hallucination_score": 0.0,
            "combined": 0.0,
            "reasoning": f"Evaluation error: {e}",
        }


# ---------------------------------------------------------------------------
# 4. Explainability Score
# ---------------------------------------------------------------------------

def explainability_score(
    answer: str,
    evidence: list[dict],
    reasoning_steps: list[str],
    sources: list[dict],
    context: str = "",
    use_llm_judge: bool = True,
) -> dict:
    """Evaluate the quality of explanations: citation accuracy + reasoning clarity.

    Combines:
    - Citation coverage: fraction of evidence items that reference real sources
    - Reasoning completeness: whether reasoning steps form a logical chain
    - LLM judge: overall explainability assessment

    Returns dict with citation_score, reasoning_score, llm_score, and combined.
    """
    # Citation coverage: check evidence items reference actual sources
    citation_score = _evaluate_citations(evidence, sources)

    # Reasoning completeness: basic structural check
    reasoning_score = _evaluate_reasoning(reasoning_steps)

    # LLM judge for overall explainability
    llm_score = 0.0
    llm_reasoning = ""
    if use_llm_judge and context:
        judge_result = _llm_judge_explainability(answer, evidence, reasoning_steps, context)
        llm_score = judge_result["score"]
        llm_reasoning = judge_result["reasoning"]

    # Combined (equal weight)
    if use_llm_judge and context:
        combined = (citation_score + reasoning_score + llm_score) / 3.0
    else:
        combined = (citation_score + reasoning_score) / 2.0

    return {
        "citation_score": round(citation_score, 3),
        "reasoning_score": round(reasoning_score, 3),
        "llm_score": round(llm_score, 3),
        "combined": round(combined, 3),
        "reasoning": llm_reasoning,
    }


def _evaluate_citations(evidence: list[dict], sources: list[dict]) -> float:
    """Check what fraction of evidence citations reference real retrieved sources."""
    if not evidence:
        return 0.0

    source_text = " ".join(
        s.get("source", "") + " " + s.get("snippet", "")
        for s in sources
    ).lower()

    grounded = 0
    for ev in evidence:
        ev_source = ev.get("source", "").lower()
        ev_quote = ev.get("quote", "").lower()
        # Check if evidence source or quote text appears in actual sources
        if any(word in source_text for word in ev_source.split() if len(word) > 3):
            grounded += 1
        elif any(word in source_text for word in ev_quote.split() if len(word) > 3):
            grounded += 1

    return grounded / len(evidence)


def _evaluate_reasoning(reasoning_steps: list[str]) -> float:
    """Evaluate reasoning quality based on structure and completeness."""
    if not reasoning_steps:
        return 0.0

    score = 0.0

    # Has at least 2 steps
    if len(reasoning_steps) >= 2:
        score += 0.4

    # Has at least 3 steps (thorough reasoning)
    if len(reasoning_steps) >= 3:
        score += 0.2

    # Steps have meaningful content (not too short)
    meaningful = sum(1 for s in reasoning_steps if len(s) > 20)
    if reasoning_steps:
        score += 0.4 * (meaningful / len(reasoning_steps))

    return min(1.0, score)


def _llm_judge_explainability(
    answer: str,
    evidence: list[dict],
    reasoning_steps: list[str],
    context: str,
) -> dict:
    """Use LLM to judge overall explainability quality."""
    evidence_text = "\n".join(
        f"- Claim: {e.get('claim', '')} | Source: {e.get('source', '')} | Quote: {e.get('quote', '')}"
        for e in evidence
    ) if evidence else "No evidence provided."

    reasoning_text = "\n".join(
        f"- Step {i+1}: {s}" for i, s in enumerate(reasoning_steps)
    ) if reasoning_steps else "No reasoning provided."

    prompt = f"""You are evaluating the EXPLAINABILITY of a maritime safety RAG system's output.

ANSWER:
{answer}

EVIDENCE CITATIONS:
{evidence_text}

REASONING TRACE:
{reasoning_text}

CONTEXT (source documents):
{context[:2000]}

Score the explainability from 0 to 10:
- 10 = Citations are accurate, reasoning is clear and logical, a human can fully understand WHY this answer was given
- 5 = Some citations and reasoning provided but incomplete or partially unclear
- 0 = No meaningful explanation provided

Respond in EXACTLY this format:
EXPLAINABILITY: <score>
REASONING: <one line explanation>"""

    try:
        llm = ChatOllama(model=LLM_MODEL, temperature=0.0)
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content

        score = _extract_score(text, "EXPLAINABILITY") / 10.0
        reasoning_match = re.search(r"REASONING:\s*(.+)", text, re.IGNORECASE)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        return {"score": score, "reasoning": reasoning}
    except Exception as e:
        return {"score": 0.0, "reasoning": f"Judge error: {e}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    evidence: list[dict] | None = None,
    reasoning_steps: list[str] | None = None,
    use_llm_judge: bool = True,
) -> dict:
    """Compute all four Explainable RAG metrics for a single question-answer pair.

    Returns a dict with:
    - retrieval_accuracy: {recall, precision, f1}
    - answer_correctness: {fact_coverage, llm_score, combined}
    - faithfulness: {faithfulness_score, hallucination_score, combined}
    - explainability: {citation_score, reasoning_score, llm_score, combined}
    """
    retrieval = retrieval_accuracy(sources, key_facts, source_keywords)

    correctness = answer_correctness(
        question, answer, key_facts, context, use_llm_judge,
    )

    faith = faithfulness(
        question, answer, context, use_llm_judge,
    )

    explain = explainability_score(
        answer=answer,
        evidence=evidence or [],
        reasoning_steps=reasoning_steps or [],
        sources=sources,
        context=context,
        use_llm_judge=use_llm_judge,
    )

    return {
        "retrieval_accuracy": retrieval,
        "answer_correctness": correctness,
        "faithfulness": faith,
        "explainability": explain,
    }
