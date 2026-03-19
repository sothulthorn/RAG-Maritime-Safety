"""Answer verification: cross-checks the generated answer against source documents.

After the LLM generates an answer, this module runs a second pass that:
1. Extracts specific claims from the answer
2. Checks if each claim is supported by the retrieved context
3. Flags unsupported claims and optionally corrects them

This is what general LLMs CANNOT do — they have no source documents to verify against.
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from config import LLM_MODEL


VERIFICATION_PROMPT = """You are a fact-checker for maritime safety information. Your job is to verify an answer against source documents.

SOURCE DOCUMENTS (ground truth):
{context}

ANSWER TO VERIFY:
{answer}

For each factual claim in the answer, check if it is supported by the source documents.

Respond in this EXACT format:
VERIFIED_CLAIMS:
- [claim text] → SUPPORTED
- [claim text] → UNSUPPORTED (not found in sources)
- [claim text] → PARTIALLY SUPPORTED (detail: what's different)

CONFIDENCE: [HIGH/MEDIUM/LOW]
CORRECTIONS: [List any corrections needed, or "None needed"]
VERIFIED_ANSWER: [Rewrite the answer keeping ONLY supported claims. Remove or flag any unsupported claims. Keep the same style and structure.]"""


def verify_answer(answer: str, context: str) -> dict:
    """Verify an answer against the retrieved context.

    Args:
        answer: The LLM-generated answer to verify.
        context: The formatted context from retrieved documents.

    Returns:
        dict with:
        - verified_answer: corrected answer with only supported claims
        - confidence: HIGH/MEDIUM/LOW
        - verification_details: raw verification output
        - was_modified: whether the answer was changed
    """
    if not context or not answer:
        return {
            "verified_answer": answer,
            "confidence": "LOW",
            "verification_details": "No context available for verification",
            "was_modified": False,
        }

    llm = ChatOllama(model=LLM_MODEL, temperature=0.0)

    prompt = VERIFICATION_PROMPT.format(
        context=context[:4000],  # limit context size
        answer=answer,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        result_text = response.content

        # Extract confidence
        confidence = "MEDIUM"
        if "CONFIDENCE: HIGH" in result_text.upper():
            confidence = "HIGH"
        elif "CONFIDENCE: LOW" in result_text.upper():
            confidence = "LOW"

        # Extract verified answer
        verified_answer = answer  # default to original
        was_modified = False

        if "VERIFIED_ANSWER:" in result_text:
            parts = result_text.split("VERIFIED_ANSWER:")
            if len(parts) > 1:
                verified = parts[1].strip()
                if verified and len(verified) > 20:
                    verified_answer = verified
                    was_modified = verified_answer.strip() != answer.strip()

        return {
            "verified_answer": verified_answer,
            "confidence": confidence,
            "verification_details": result_text,
            "was_modified": was_modified,
        }

    except Exception as e:
        return {
            "verified_answer": answer,
            "confidence": "LOW",
            "verification_details": f"Verification failed: {e}",
            "was_modified": False,
        }
