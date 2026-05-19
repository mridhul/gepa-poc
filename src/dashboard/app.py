"""Streamlit dashboard for GEPA POC results and monitoring."""

import sys
from pathlib import Path

# Streamlit runs this file as __main__, so ensure project root is importable.
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json

from src import database

# Configure page
st.set_page_config(page_title="GEPA POC Dashboard", layout="wide", initial_sidebar_state="expanded")

# Utility: get database connection
@st.cache_resource
def get_db_conn():
    """Get database connection (cached)."""
    db_path = Path(__file__).parent.parent.parent / "data" / "gepa.db"
    return database.get_db(str(db_path), check_same_thread=False)

# Main layout
st.title("🎯 GEPA Talent Lens POC Dashboard")

conn = get_db_conn()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Overview", "Baseline Results", "Optimization Progress", "Prompt Evolution", "Example Evaluations"]
)

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================
if page == "Overview":
    st.header("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        total_applicants = database.count_applicants(conn)
        st.metric("Total Applicants", total_applicants)

    with col2:
        train_applicants = len(database.get_applicants_by_split(conn, "train"))
        st.metric("Training Set", train_applicants)

    with col3:
        holdout_applicants = len(database.get_applicants_by_split(conn, "holdout"))
        st.metric("Holdout Set", holdout_applicants)

    # Hire ratio
    all_applicants = database.get_all_applicants(conn)
    if all_applicants:
        train_apps = [a for a in all_applicants if a["split"] == "train"]
        holdout_apps = [a for a in all_applicants if a["split"] == "holdout"]

        train_hired = sum(1 for a in train_apps if a["hired"] == 1)
        holdout_hired = sum(1 for a in holdout_apps if a["hired"] == 1)

        fig = go.Figure(data=[
            go.Bar(name="Hired", x=["Train", "Holdout"], y=[train_hired, holdout_hired], marker_color="green"),
            go.Bar(name="Rejected", x=["Train", "Holdout"], y=[len(train_apps) - train_hired, len(holdout_apps) - holdout_hired], marker_color="red"),
        ])
        fig.update_layout(barmode="group", title="Hire Distribution by Split", xaxis_title="Split", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    # Sample resumes
    st.subheader("Sample Resumes")
    if all_applicants:
        for i, app in enumerate(all_applicants[:3]):
            with st.expander(f"Applicant: {app['name']}"):
                resume = app["resume_json"]
                if isinstance(resume, str):
                    resume = json.loads(resume)
                st.json(resume)

# ============================================================================
# PAGE 2: BASELINE RESULTS
# ============================================================================
elif page == "Baseline Results":
    st.header("📈 Baseline Evaluation Results")

    logs = database.get_iteration_logs(conn)
    baseline_logs = [l for l in logs if l["iteration"] == 0]

    if not baseline_logs:
        st.warning("No baseline results found. Run --phase baseline first.")
    else:
        # Metrics table
        st.subheader("Baseline Metrics")
        metrics_data = []
        for log in baseline_logs:
            metrics_data.append({
                "Split": log["split"],
                "Accuracy": round(log["accuracy"], 3) if log["accuracy"] else 0,
                "Precision": round(log["precision_h"], 3) if log["precision_h"] else 0,
                "Recall": round(log["recall_h"], 3) if log["recall_h"] else 0,
                "F1": round(log["f1_h"], 3) if log["f1_h"] else 0,
                "Spearman ρ": round(log["spearman_rho"], 3) if log["spearman_rho"] else 0,
            })

        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True)

        # Confusion matrix visualization (if we have data)
        st.subheader("Score Distribution")
        candidate_id = baseline_logs[0]["candidate_id"]
        evals = database.get_evaluations_for_candidate(conn, candidate_id, split="train")

        if evals:
            scores = [e["aggregate_score"] for e in evals]
            hired_labels = [database.get_candidate(conn, e["candidate_id"]) for e in evals]

            fig = go.Figure(data=[
                go.Histogram(x=scores, name="Aggregate Scores", nbinsx=10)
            ])
            fig.update_layout(title="Distribution of Aggregate Evaluation Scores", xaxis_title="Score (1-5)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 3: OPTIMIZATION PROGRESS
# ============================================================================
elif page == "Optimization Progress":
    st.header("🚀 Optimization Progress")

    logs = database.get_iteration_logs(conn)
    if not logs:
        st.warning("No optimization results yet. Run --phase optimize first.")
    else:
        # Filter to train split for progress tracking
        train_logs = [l for l in logs if l["split"] == "train"]

        if train_logs:
            df_progress = pd.DataFrame(train_logs)[["iteration", "candidate_id", "accuracy", "precision_h", "is_pareto"]]
            df_progress = df_progress.sort_values("iteration")

            # Accuracy + Precision over iterations
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_progress["iteration"],
                y=df_progress["accuracy"],
                name="Accuracy",
                mode="lines+markers",
            ))
            fig.add_trace(go.Scatter(
                x=df_progress["iteration"],
                y=df_progress["precision_h"],
                name="Precision (Hired)",
                mode="lines+markers",
            ))
            fig.update_layout(
                title="Accuracy and Precision Over Iterations",
                xaxis_title="Iteration",
                yaxis_title="Score",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Pareto frontier scatter
            pareto_logs = [l for l in train_logs if l["is_pareto"] == 1]
            non_pareto_logs = [l for l in train_logs if l["is_pareto"] == 0]

            fig = go.Figure()
            if pareto_logs:
                fig.add_trace(go.Scatter(
                    x=[l["accuracy"] for l in pareto_logs],
                    y=[l["precision_h"] for l in pareto_logs],
                    mode="markers",
                    name="Pareto Frontier",
                    marker=dict(size=10, color="gold", symbol="star"),
                    text=[f"Iteration {l['iteration']}, ID {l['candidate_id']}" for l in pareto_logs],
                    hovertemplate="%{text}<br>Accuracy: %{x:.3f}<br>Precision: %{y:.3f}",
                ))
            if non_pareto_logs:
                fig.add_trace(go.Scatter(
                    x=[l["accuracy"] for l in non_pareto_logs],
                    y=[l["precision_h"] for l in non_pareto_logs],
                    mode="markers",
                    name="Evaluated",
                    marker=dict(size=6, color="lightgray"),
                    text=[f"Iteration {l['iteration']}, ID {l['candidate_id']}" for l in non_pareto_logs],
                    hovertemplate="%{text}<br>Accuracy: %{x:.3f}<br>Precision: %{y:.3f}",
                ))

            fig.update_layout(
                title="Pareto Frontier (2-Objective)",
                xaxis_title="Accuracy",
                yaxis_title="Precision (Hired)",
                hovermode="closest"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Cost tracking
        st.subheader("Cost Tracking")
        total_cost = database.get_total_cost(conn)
        total_calls = database.get_total_calls(conn)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Cost", f"${total_cost:.2f}")
        with col2:
            st.metric("Total LLM Calls", total_calls)

        # LLM call logs by phase
        call_logs = database.get_llm_call_log(conn)
        if call_logs:
            df_calls = pd.DataFrame(call_logs)
            phase_costs = df_calls.groupby("phase")["cost_usd"].sum().reset_index()

            fig = px.bar(phase_costs, x="phase", y="cost_usd", title="Cost by Phase", labels={"cost_usd": "Cost (USD)"})
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 4: PROMPT EVOLUTION
# ============================================================================
elif page == "Prompt Evolution":
    st.header("📝 Prompt Evolution")

    logs = database.get_iteration_logs(conn)
    train_logs = [l for l in logs if l["split"] == "train"]

    if not train_logs:
        st.warning("No optimization results yet.")
    else:
        # Find best candidate
        best_log = max(train_logs, key=lambda l: l["accuracy"])
        best_id = best_log["candidate_id"]

        # Find baseline
        baseline_logs = [l for l in logs if l["iteration"] == 0 and l["split"] == "train"]
        if baseline_logs:
            baseline_id = baseline_logs[0]["candidate_id"]

            baseline_candidate = database.get_candidate(conn, baseline_id)
            best_candidate = database.get_candidate(conn, best_id)

            if baseline_candidate and best_candidate:
                baseline_prompts = json.loads(baseline_candidate["prompts_json"])
                best_prompts = json.loads(best_candidate["prompts_json"])

                st.subheader("Baseline vs. Optimized Prompts")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("### Baseline (Human-Authored)")
                    for i, p in enumerate(baseline_prompts):
                        with st.expander(f"{i+1}. {p['name']}"):
                            st.write(p["prompt"])

                with col2:
                    st.write("### Optimized (GEPA)")
                    for i, p in enumerate(best_prompts):
                        with st.expander(f"{i+1}. {p['name']}"):
                            st.write(p["prompt"])

                st.subheader("Candidate Evolution")
                st.write(f"Best candidate: ID {best_id} (Iteration {best_log['iteration']})")
                st.write(f"Accuracy: {best_log['accuracy']:.3f} | Precision: {best_log['precision_h']:.3f}")

# ============================================================================
# PAGE 5: EXAMPLE EVALUATIONS
# ============================================================================
elif page == "Example Evaluations":
    st.header("💡 Example Applicant Evaluations")

    logs = database.get_iteration_logs(conn)
    train_logs = [l for l in logs if l["split"] == "train"]

    if not train_logs:
        st.warning("No evaluation results yet.")
    else:
        baseline_log = next((l for l in logs if l["iteration"] == 0 and l["split"] == "train"), None)
        best_log = max((l for l in train_logs if l["iteration"] > 0), key=lambda l: l["accuracy"], default=None)

        if baseline_log and best_log:
            baseline_id = baseline_log["candidate_id"]
            best_id = best_log["candidate_id"]

            applicants = database.get_applicants_by_split(conn, "train")

            st.write(f"Comparing baseline (ID {baseline_id}) vs. best (ID {best_id})")

            if applicants:
                # Show first 3 applicants with scores
                for app in applicants[:3]:
                    with st.expander(f"Applicant: {app['name']}"):
                        baseline_evals = database.get_evaluations_for_candidate(conn, baseline_id)
                        best_evals = database.get_evaluations_for_candidate(conn, best_id)

                        baseline_eval = next((e for e in baseline_evals if e["applicant_id"] == app["id"]), None)
                        best_eval = next((e for e in best_evals if e["applicant_id"] == app["id"]), None)

                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**Baseline Evaluation**")
                            if baseline_eval:
                                st.write(f"Aggregate Score: **{baseline_eval['aggregate_score']:.1f}/5**")
                                st.write(f"Prediction: **{'HIRE' if baseline_eval['predicted_hired'] else 'REJECT'}**")
                                st.write(f"Ground Truth: **{'HIRE' if app['hired'] else 'REJECT'}**")
                                scores = json.loads(baseline_eval["scores_json"])
                                for score in scores:
                                    st.write(f"- {score['lens']}: {score['score']}")

                        with col2:
                            st.write("**Optimized Evaluation**")
                            if best_eval:
                                st.write(f"Aggregate Score: **{best_eval['aggregate_score']:.1f}/5**")
                                st.write(f"Prediction: **{'HIRE' if best_eval['predicted_hired'] else 'REJECT'}**")
                                st.write(f"Ground Truth: **{'HIRE' if app['hired'] else 'REJECT'}**")
                                scores = json.loads(best_eval["scores_json"])
                                for score in scores:
                                    st.write(f"- {score['lens']}: {score['score']}")
