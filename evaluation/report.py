"""
Streamlit evaluation report page.

Usage:
    streamlit run evaluation/report.py
"""

import json
import os
from pathlib import Path

import streamlit as st

RESULTS_DIR = Path(__file__).parent / "results"


def _load_latest_results() -> dict | None:
    """Load the most recent evaluation results."""
    if not RESULTS_DIR.exists():
        return None

    files = sorted(RESULTS_DIR.glob("eval_*.json"), reverse=True)
    if not files:
        return None

    with open(files[0], "r", encoding="utf-8") as f:
        return json.load(f)


def _results_list() -> list[Path]:
    """List all evaluation result files."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("eval_*.json"), reverse=True)


def main():
    st.set_page_config(page_title="RAG Evaluation Report", page_icon="📊", layout="wide")
    st.title("RAG vs Plain LLM — Evaluation Report")

    # Select results file
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
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", data["model"])
    col2.metric("Questions Evaluated", data["num_questions"])
    col3.metric("LLM Judge", "Enabled" if data.get("llm_judge_enabled") else "Disabled")

    st.divider()

    # --- Overall Comparison Table ---
    st.header("Overall Comparison")

    metric_labels = {
        "key_fact_coverage": "Key Fact Coverage",
        "citation_accuracy": "Citation Accuracy",
        "source_grounding": "Source Grounding",
        "faithfulness_score": "Faithfulness",
        "hallucination_score": "Non-Hallucination",
        "relevance_score": "Relevance",
    }

    # Build comparison data
    comparison_data = []
    for metric, label in metric_labels.items():
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

    # --- Visual Comparison ---
    st.header("Score Comparison Chart")
    chart_data = {}
    for metric, label in metric_labels.items():
        rag_val = summary["rag"].get(metric, {}).get("mean", 0)
        plain_val = summary["plain_llm"].get(metric, {}).get("mean", 0)
        chart_data[label] = {"RAG": rag_val, "Plain LLM": plain_val}

    import pandas as pd
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
            for metric, label in metric_labels.items():
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

                st.markdown("**Scores:**")
                metrics = result["rag"]["metrics"]
                for metric, label in metric_labels.items():
                    val = metrics.get(metric, 0)
                    st.progress(val, text=f"{label}: {val:.0%}")

            with col2:
                st.subheader("Plain LLM Answer")
                st.markdown(result["plain_llm"]["answer"])
                st.caption(f"Response time: {result['plain_llm']['response_time']}s")

                st.markdown("**No sources cited** (no retrieval)")

                st.markdown("**Scores:**")
                metrics = result["plain_llm"]["metrics"]
                for metric, label in metric_labels.items():
                    val = metrics.get(metric, 0)
                    st.progress(val, text=f"{label}: {val:.0%}")

            if result["rag"]["metrics"].get("reasoning"):
                st.info(f"**RAG Judge Reasoning:** {result['rag']['metrics']['reasoning']}")
            if result["plain_llm"]["metrics"].get("reasoning"):
                st.warning(f"**Plain LLM Judge Reasoning:** {result['plain_llm']['metrics']['reasoning']}")

    # --- Key Takeaway ---
    st.divider()
    st.header("Key Takeaway")

    rag_fact = summary["rag"].get("key_fact_coverage", {}).get("mean", 0)
    plain_fact = summary["plain_llm"].get("key_fact_coverage", {}).get("mean", 0)
    rag_cite = summary["rag"].get("citation_accuracy", {}).get("mean", 0)
    plain_cite = summary["plain_llm"].get("citation_accuracy", {}).get("mean", 0)
    rag_hall = summary["rag"].get("hallucination_score", {}).get("mean", 0)
    plain_hall = summary["plain_llm"].get("hallucination_score", {}).get("mean", 0)

    st.markdown(f"""
    | Metric | RAG | Plain LLM | Advantage |
    |---|---|---|---|
    | Key Fact Coverage | {rag_fact:.0%} | {plain_fact:.0%} | **{'+' if rag_fact > plain_fact else ''}{(rag_fact - plain_fact)*100:.0f}pp** |
    | Citation Accuracy | {rag_cite:.0%} | {plain_cite:.0%} | **{'+' if rag_cite > plain_cite else ''}{(rag_cite - plain_cite)*100:.0f}pp** |
    | Non-Hallucination | {rag_hall:.0%} | {plain_hall:.0%} | **{'+' if rag_hall > plain_hall else ''}{(rag_hall - plain_hall)*100:.0f}pp** |

    The RAG system provides **verifiable, source-grounded answers** from actual maritime safety documents,
    while the plain LLM generates plausible-sounding answers that cannot be traced to real sources.
    """)


if __name__ == "__main__":
    main()
