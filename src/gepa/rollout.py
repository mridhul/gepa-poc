"""Rollout and evaluation of GEPA candidates."""

import sqlite3
import json
from ..llm.client import LLMClient
from ..evaluator.prompt_executor import evaluate_applicant_set
from ..evaluator.metrics import compute_metrics_from_db, get_worst_cases as get_worst_cases_metrics
from .. import database


def rollout_candidate(
    conn: sqlite3.Connection,
    client: LLMClient,
    candidate_id: int,
    split: str = "train",
    threshold: float = 3.0,
) -> dict:
    """
    Evaluate a candidate on a data split.
    Idempotent: skips LLM calls if evaluations already exist.
    Returns metrics dict.
    """

    # Check if evaluations already exist for this split (idempotent)
    if database.evaluations_exist_for_candidate(conn, candidate_id, split=split):
        metrics = compute_metrics_from_db(conn, candidate_id, split=split)
        return {
            "accuracy": metrics.accuracy,
            "precision_hired": metrics.precision_hired,
            "recall_hired": metrics.recall_hired,
            "f1_hired": metrics.f1_hired,
            "spearman_rho": metrics.spearman_rho,
            "n_samples": metrics.n_samples,
        }

    # Get applicants for this split
    applicants = database.get_applicants_by_split(conn, split)

    # Evaluate
    results = evaluate_applicant_set(client, conn, applicants, candidate_id, threshold)

    # Compute and return metrics
    metrics = compute_metrics_from_db(conn, candidate_id, split=split)
    return {
        "accuracy": metrics.accuracy,
        "precision_hired": metrics.precision_hired,
        "recall_hired": metrics.recall_hired,
        "f1_hired": metrics.f1_hired,
        "spearman_rho": metrics.spearman_rho,
        "n_samples": metrics.n_samples,
    }


def get_worst_cases(
    conn: sqlite3.Connection,
    candidate_id: int,
    split: str = "train",
    k: int = 10,
) -> list[dict]:
    """Get top-k failure cases for a candidate."""

    failures = get_worst_cases_metrics(conn, candidate_id, split=split, k=k)

    # Format for reflector
    formatted_failures = []
    for failure in failures:
        applicant = failure["applicant"]
        resume_json = applicant["resume_json"]
        if isinstance(resume_json, str):
            resume_json = json.loads(resume_json)

        formatted_failures.append({
            "applicant_name": applicant.get("name", "Unknown"),
            "resume": resume_json,
            "predicted_hired": failure["predicted_hired"],
            "actual_hired": failure["actual_hired"],
            "aggregate_score": failure["aggregate_score"],
            "scores_json": failure["scores_json"],
        })

    return formatted_failures
