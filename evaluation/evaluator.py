"""
Explainable RAG evaluation pipeline: runs test questions through RAG and plain LLM,
then scores both using the four-metric evaluation framework.

Metrics:
1. Retrieval Accuracy (Recall/Precision/F1)
2. Answer Correctness (Fact Coverage + LLM Judge)
3. Faithfulness (No Hallucinations)
4. Explainability Score (Citation Accuracy + Reasoning Clarity)

Usage:
    python -m evaluation.evaluator                  # Run full evaluation
    python -m evaluation.evaluator --no-llm-judge   # Skip LLM-as-judge (faster)
    python -m evaluation.evaluator --max 5          # Only evaluate first 5 questions
"""

import argparse
import json
import os
import time
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from config import LLM_MODEL, LLM_TEMPERATURE
from retrieval.rag_chain import answer_question, _format_context
from retrieval.retriever import retrieve
from evaluation.test_set import TEST_SET
from evaluation.metrics import compute_all_metrics


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

METRIC_NAMES = [
    "retrieval_accuracy",
    "answer_correctness",
    "faithfulness",
    "explainability",
]


def _ask_plain_llm(question: str) -> str:
    """Ask the same LLM without any RAG context — simulates a general LLM."""
    llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    messages = [
        SystemMessage(content="You are a maritime safety expert. Answer the question as accurately as possible. Cite specific sources, regulations, or reports when you can."),
        HumanMessage(content=question),
    ]
    response = llm.invoke(messages)
    return response.content


def evaluate_single(test_case: dict, use_llm_judge: bool = True) -> dict:
    """Evaluate a single test case through both RAG and plain LLM."""
    question = test_case["question"]
    key_facts = test_case["key_facts"]
    source_keywords = test_case["source_keywords"]

    print(f"\n  Q: {question}")

    # --- RAG answer ---
    print("    Running RAG...")
    rag_start = time.time()
    rag_result = answer_question(question)
    rag_time = time.time() - rag_start
    rag_answer = rag_result["answer"]
    rag_sources = rag_result["sources"]
    rag_evidence = rag_result.get("evidence", [])
    rag_reasoning = rag_result.get("reasoning_steps", [])

    # Get context for LLM judge
    rag_docs = retrieve(question)
    rag_context = _format_context(rag_docs) if rag_docs else ""

    rag_metrics = compute_all_metrics(
        question=question,
        answer=rag_answer,
        sources=rag_sources,
        key_facts=key_facts,
        source_keywords=source_keywords,
        context=rag_context,
        evidence=rag_evidence,
        reasoning_steps=rag_reasoning,
        use_llm_judge=use_llm_judge,
    )

    # --- Plain LLM answer (no RAG) ---
    print("    Running plain LLM...")
    plain_start = time.time()
    plain_answer = _ask_plain_llm(question)
    plain_time = time.time() - plain_start

    plain_metrics = compute_all_metrics(
        question=question,
        answer=plain_answer,
        sources=[],
        key_facts=key_facts,
        source_keywords=source_keywords,
        context=rag_context,
        evidence=[],
        reasoning_steps=[],
        use_llm_judge=use_llm_judge,
    )

    return {
        "id": test_case["id"],
        "question": question,
        "category": test_case["category"],
        "ground_truth": test_case["ground_truth"],
        "rag": {
            "answer": rag_answer,
            "sources": rag_sources,
            "evidence": rag_evidence,
            "reasoning_steps": rag_reasoning,
            "metrics": rag_metrics,
            "response_time": round(rag_time, 2),
        },
        "plain_llm": {
            "answer": plain_answer,
            "sources": [],
            "evidence": [],
            "reasoning_steps": [],
            "metrics": plain_metrics,
            "response_time": round(plain_time, 2),
        },
    }


def compute_summary(results: list[dict]) -> dict:
    """Compute aggregate metrics across all test cases."""
    summary = {"rag": {}, "plain_llm": {}}

    for model in ["rag", "plain_llm"]:
        for metric in METRIC_NAMES:
            values = [
                r[model]["metrics"].get(metric, {}).get("combined", 0)
                for r in results
            ]
            if values:
                summary[model][metric] = {
                    "mean": round(sum(values) / len(values), 3),
                    "min": round(min(values), 3),
                    "max": round(max(values), 3),
                }

        times = [r[model]["response_time"] for r in results]
        summary[model]["avg_response_time"] = round(sum(times) / len(times), 2) if times else 0

    # Compute win/loss/tie for each metric
    summary["comparison"] = {}
    for metric in METRIC_NAMES:
        rag_wins = 0
        plain_wins = 0
        ties = 0
        for r in results:
            rag_val = r["rag"]["metrics"].get(metric, {}).get("combined", 0)
            plain_val = r["plain_llm"]["metrics"].get(metric, {}).get("combined", 0)
            if abs(rag_val - plain_val) < 0.05:
                ties += 1
            elif rag_val > plain_val:
                rag_wins += 1
            else:
                plain_wins += 1
        summary["comparison"][metric] = {
            "rag_wins": rag_wins,
            "plain_llm_wins": plain_wins,
            "ties": ties,
        }

    # Per-category breakdown
    categories = set(r["category"] for r in results)
    summary["by_category"] = {}
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        summary["by_category"][cat] = {}
        for model in ["rag", "plain_llm"]:
            summary["by_category"][cat][model] = {}
            for metric in METRIC_NAMES:
                values = [
                    r[model]["metrics"].get(metric, {}).get("combined", 0)
                    for r in cat_results
                ]
                if values:
                    summary["by_category"][cat][model][metric] = round(
                        sum(values) / len(values), 3
                    )

    return summary


def run_evaluation(max_questions: int = 0, use_llm_judge: bool = True) -> dict:
    """Run the full evaluation pipeline."""
    test_cases = TEST_SET
    if max_questions > 0:
        test_cases = test_cases[:max_questions]

    print(f"Running Explainable RAG evaluation on {len(test_cases)} questions...")
    print(f"LLM Judge: {'enabled' if use_llm_judge else 'disabled'}")

    results = []
    for i, tc in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] {tc['id']}")
        result = evaluate_single(tc, use_llm_judge=use_llm_judge)
        results.append(result)

    summary = compute_summary(results)

    output = {
        "timestamp": datetime.now().isoformat(),
        "model": LLM_MODEL,
        "num_questions": len(results),
        "llm_judge_enabled": use_llm_judge,
        "evaluation_framework": "Explainable RAG (4-metric)",
        "summary": summary,
        "results": results,
    }

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(RESULTS_DIR, f"eval_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("EXPLAINABLE RAG EVALUATION COMPLETE")
    print(f"{'='*60}")
    _print_summary(summary)
    print(f"\nFull results saved to: {output_path}")

    return output


def _print_summary(summary: dict):
    """Print a formatted summary table."""
    print(f"\n{'Metric':<25} {'RAG':>10} {'Plain LLM':>10} {'Winner':>10}")
    print("-" * 60)

    metric_labels = {
        "retrieval_accuracy": "Retrieval Accuracy",
        "answer_correctness": "Answer Correctness",
        "faithfulness": "Faithfulness",
        "explainability": "Explainability",
    }

    for metric, label in metric_labels.items():
        rag_val = summary["rag"].get(metric, {}).get("mean", 0)
        plain_val = summary["plain_llm"].get(metric, {}).get("mean", 0)

        if abs(rag_val - plain_val) < 0.03:
            winner = "Tie"
        elif rag_val > plain_val:
            winner = "RAG"
        else:
            winner = "Plain LLM"

        print(f"{label:<25} {rag_val:>9.1%} {plain_val:>9.1%} {winner:>10}")

    print(f"\n{'Avg Response Time':<25} {summary['rag']['avg_response_time']:>8.1f}s {summary['plain_llm']['avg_response_time']:>8.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Explainable RAG evaluation pipeline")
    parser.add_argument("--max", type=int, default=0, help="Max questions to evaluate (0=all)")
    parser.add_argument("--no-llm-judge", action="store_true", help="Skip LLM-as-judge scoring")
    args = parser.parse_args()

    run_evaluation(max_questions=args.max, use_llm_judge=not args.no_llm_judge)
