"""
Explainable RAG Evaluation Report — Streamlit dashboard.

Visualizes the four-metric evaluation framework:
1. Retrieval Accuracy
2. Answer Correctness
3. Faithfulness
4. Explainability Score

Usage:
    streamlit run evaluation/report.py
"""

import json
from pathlib import Path

import streamlit as st
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"

METRIC_LABELS = {
    "retrieval_accuracy": "Retrieval Accuracy",
    "answer_correctness": "Answer Correctness",
    "faithfulness": "Faithfulness",
    "explainability": "Explainability",
}


def _results_list() -> list[Path]:
    """List all evaluation result files."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("eval_*.json"), reverse=True)


def main():
    st.set_page_config(page_title="Explainable RAG Evaluation", page_icon="📊", layout="wide")
    st.title("Explainable RAG — Evaluation Dashboard")

    result_files = _results_list()
    if not result_files:
        st.error("No evaluation results found. Run the evaluation first:")
        st.code("python -m evaluation.evaluator")
        return

    selected_file = st.sidebar.selectbox(
        "Select evaluation run",
        result_files,
        format_func=lambda f: f.stem,
    )

    with open(selected_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data["summary"]
    results = data["results"]

    # --- Header Info ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", data["model"])
    col2.metric("Questions", data["num_questions"])
    col3.metric("LLM Judge", "Enabled" if data.get("llm_judge_enabled") else "Disabled")
    col4.metric("Framework", data.get("evaluation_framework", "Explainable RAG"))

    st.divider()

    # --- Evaluation Dashboard (4 metrics overview) ---
    st.header("Evaluation Dashboard")

    dash_cols = st.columns(4)
    for i, (metric_key, label) in enumerate(METRIC_LABELS.items()):
        rag_val = summary["rag"].get(metric_key, {}).get("mean", 0)
        with dash_cols[i]:
            st.metric(label, f"{rag_val:.0%}")
            st.progress(rag_val)

    st.divider()

    # --- Overall Comparison Table ---
    st.header("RAG vs Plain LLM Comparison")

    comparison_data = []
    for metric, label in METRIC_LABELS.items():
        rag_val = summary["rag"].get(metric, {}).get("mean", 0)
        plain_val = summary["plain_llm"].get(metric, {}).get("mean", 0)
        diff = rag_val - plain_val
        comp = summary.get("comparison", {}).get(metric, {})

        comparison_data.append({
            "Metric": label,
            "RAG": f"{rag_val:.1%}",
            "Plain LLM": f"{plain_val:.1%}",
            "Difference": f"{diff:+.1%}",
            "RAG Wins": comp.get("rag_wins", 0),
            "LLM Wins": comp.get("plain_llm_wins", 0),
            "Ties": comp.get("ties", 0),
        })

    st.table(comparison_data)

    # --- Visual Comparison Chart ---
    st.header("Score Comparison Chart")
    chart_data = {}
    for metric, label in METRIC_LABELS.items():
        rag_val = summary["rag"].get(metric, {}).get("mean", 0)
        plain_val = summary["plain_llm"].get(metric, {}).get("mean", 0)
        chart_data[label] = {"RAG": rag_val, "Plain LLM": plain_val}

    chart_df = pd.DataFrame(chart_data).T
    st.bar_chart(chart_df)

    # --- Response Time ---
    st.header("Response Time")
    time_col1, time_col2 = st.columns(2)
    time_col1.metric(
        "RAG Avg Response Time",
        f"{summary['rag']['avg_response_time']:.1f}s",
    )
    time_col2.metric(
        "Plain LLM Avg Response Time",
        f"{summary['plain_llm']['avg_response_time']:.1f}s",
    )

    # --- Per-Category Breakdown ---
    st.header("Performance by Category")
    categories = summary.get("by_category", {})
    for cat, cat_data in categories.items():
        with st.expander(f"Category: {cat.replace('_', ' ').title()}"):
            cat_comparison = []
            for metric, label in METRIC_LABELS.items():
                rag_val = cat_data.get("rag", {}).get(metric, 0)
                plain_val = cat_data.get("plain_llm", {}).get(metric, 0)
                cat_comparison.append({
                    "Metric": label,
                    "RAG": f"{rag_val:.1%}",
                    "Plain LLM": f"{plain_val:.1%}",
                    "Difference": f"{rag_val - plain_val:+.1%}",
                })
            st.table(cat_comparison)

    # --- Individual Results ---
    st.header("Individual Question Results")

    for i, result in enumerate(results):
        with st.expander(f"Q{i+1}: {result['question']}"):
            st.caption(f"Category: {result['category']} | ID: {result['id']}")
            st.markdown(f"**Ground Truth:** {result['ground_truth']}")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("RAG Answer")
                st.markdown(result["rag"]["answer"])
                st.caption(f"Response time: {result['rag']['response_time']}s")

                if result["rag"]["sources"]:
                    st.markdown("**Sources cited:**")
                    for src in result["rag"]["sources"]:
                        source_label = src.get("source", "Unknown")
                        page = src.get("page")
                        if page:
                            source_label += f" (p.{page})"
                        st.caption(f"- {source_label}")

                # Evidence citations
                rag_evidence = result["rag"].get("evidence", [])
                if rag_evidence:
                    st.markdown("**Evidence Citations:**")
                    for ev in rag_evidence:
                        st.caption(
                            f"- *{ev.get('claim', '')}* → "
                            f"{ev.get('source', '')} — \"{ev.get('quote', '')}\""
                        )

                # Reasoning trace
                rag_reasoning = result["rag"].get("reasoning_steps", [])
                if rag_reasoning:
                    st.markdown("**Reasoning Trace:**")
                    for j, step in enumerate(rag_reasoning, 1):
                        st.caption(f"Step {j}: {step}")

                st.markdown("**Scores:**")
                metrics = result["rag"]["metrics"]
                for metric, label in METRIC_LABELS.items():
                    val = metrics.get(metric, {}).get("combined", 0)
                    st.progress(val, text=f"{label}: {val:.0%}")

            with col2:
                st.subheader("Plain LLM Answer")
                st.markdown(result["plain_llm"]["answer"])
                st.caption(f"Response time: {result['plain_llm']['response_time']}s")

                st.markdown("**No sources, evidence, or reasoning** (no retrieval)")

                st.markdown("**Scores:**")
                metrics = result["plain_llm"]["metrics"]
                for metric, label in METRIC_LABELS.items():
                    val = metrics.get(metric, {}).get("combined", 0)
                    st.progress(val, text=f"{label}: {val:.0%}")

            # Judge reasoning
            for model_key, model_label in [("rag", "RAG"), ("plain_llm", "Plain LLM")]:
                for metric_key in METRIC_LABELS:
                    reasoning = result[model_key]["metrics"].get(metric_key, {}).get("reasoning", "")
                    if reasoning:
                        st.caption(f"{model_label} {METRIC_LABELS[metric_key]}: {reasoning}")

    # --- Key Takeaway ---
    st.divider()
    st.header("Key Takeaway")

    rag_scores = {
        metric: summary["rag"].get(metric, {}).get("mean", 0)
        for metric in METRIC_LABELS
    }
    plain_scores = {
        metric: summary["plain_llm"].get(metric, {}).get("mean", 0)
        for metric in METRIC_LABELS
    }

    st.markdown(f"""
| Metric | RAG | Plain LLM | Advantage |
|---|---|---|---|
| Retrieval Accuracy | {rag_scores['retrieval_accuracy']:.0%} | {plain_scores['retrieval_accuracy']:.0%} | **{'+' if rag_scores['retrieval_accuracy'] > plain_scores['retrieval_accuracy'] else ''}{(rag_scores['retrieval_accuracy'] - plain_scores['retrieval_accuracy'])*100:.0f}pp** |
| Answer Correctness | {rag_scores['answer_correctness']:.0%} | {plain_scores['answer_correctness']:.0%} | **{'+' if rag_scores['answer_correctness'] > plain_scores['answer_correctness'] else ''}{(rag_scores['answer_correctness'] - plain_scores['answer_correctness'])*100:.0f}pp** |
| Faithfulness | {rag_scores['faithfulness']:.0%} | {plain_scores['faithfulness']:.0%} | **{'+' if rag_scores['faithfulness'] > plain_scores['faithfulness'] else ''}{(rag_scores['faithfulness'] - plain_scores['faithfulness'])*100:.0f}pp** |
| Explainability | {rag_scores['explainability']:.0%} | {plain_scores['explainability']:.0%} | **{'+' if rag_scores['explainability'] > plain_scores['explainability'] else ''}{(rag_scores['explainability'] - plain_scores['explainability'])*100:.0f}pp** |

The Explainable RAG system provides **verifiable, source-grounded answers** with transparent
reasoning traces and evidence citations — not just WHAT the answer is, but WHY it was given
and WHERE it comes from.
""")


if __name__ == "__main__":
    main()
