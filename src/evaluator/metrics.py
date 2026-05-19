"""Metrics calculation for evaluations."""

import sqlite3
import json
from dataclasses import dataclass
from typing import Optional
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from .. import database


@dataclass
class MetricsResult:
    accuracy: float
    precision_hired: float
    recall_hired: float
    f1_hired: float
    spearman_rho: float
    spearman_pvalue: float
    n_samples: int
    n_correct: int


def compute_metrics(
    evaluations: list[dict],
    applicants: list[dict],
    threshold: float = 3.0,
) -> MetricsResult:
    """Compute metrics from evaluations."""

    if not evaluations:
        return MetricsResult(
            accuracy=0.0,
            precision_hired=0.0,
            recall_hired=0.0,
            f1_hired=0.0,
            spearman_rho=0.0,
            spearman_pvalue=1.0,
            n_samples=0,
            n_correct=0,
        )

    # Build ID -> applicant mapping
    applicant_map = {a["id"]: a for a in applicants}

    # Extract predictions and ground truth
    predictions = []
    ground_truth = []
    scores = []

    for eval_dict in evaluations:
        applicant_id = eval_dict["applicant_id"]
        if applicant_id not in applicant_map:
            continue

        applicant = applicant_map[applicant_id]
        predicted_hired = eval_dict["predicted_hired"]
        actual_hired = applicant["hired"]

        predictions.append(predicted_hired)
        ground_truth.append(actual_hired)

        # For Spearman: use aggregate score
        agg_score = eval_dict["aggregate_score"]
        scores.append(agg_score)

    if not predictions:
        return MetricsResult(
            accuracy=0.0,
            precision_hired=0.0,
            recall_hired=0.0,
            f1_hired=0.0,
            spearman_rho=0.0,
            spearman_pvalue=1.0,
            n_samples=0,
            n_correct=0,
        )

    # Accuracy
    acc = accuracy_score(ground_truth, predictions)
    n_correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)

    # Precision, recall, F1 for hired class (class=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth,
        predictions,
        labels=[1],
        zero_division=0.0,
    )
    precision = precision[0]
    recall = recall[0]
    f1 = f1[0]

    # Spearman rank correlation
    try:
        # For Spearman: scores vs ground truth (binary)
        rho, pvalue = spearmanr(scores, ground_truth)
    except Exception:
        rho, pvalue = 0.0, 1.0

    return MetricsResult(
        accuracy=acc,
        precision_hired=precision,
        recall_hired=recall,
        f1_hired=f1,
        spearman_rho=rho,
        spearman_pvalue=pvalue,
        n_samples=len(predictions),
        n_correct=n_correct,
    )


def compute_metrics_from_db(
    conn: sqlite3.Connection,
    candidate_id: int,
    split: Optional[str] = None,
) -> MetricsResult:
    """Compute metrics from evaluations stored in DB."""

    evaluations = database.get_evaluations_for_candidate(conn, candidate_id, split=split)

    if split == "train":
        applicants = database.get_applicants_by_split(conn, "train")
    elif split == "holdout":
        applicants = database.get_applicants_by_split(conn, "holdout")
    else:
        applicants = database.get_all_applicants(conn)

    return compute_metrics(evaluations, applicants)


def get_worst_cases(
    conn: sqlite3.Connection,
    candidate_id: int,
    split: str = "train",
    k: int = 10,
) -> list[dict]:
    """Get top-k failure cases."""

    evaluations = database.get_evaluations_for_candidate(conn, candidate_id, split=split)
    applicants_map = {a["id"]: a for a in database.get_applicants_by_split(conn, split)}

    failures = []
    for eval_dict in evaluations:
        applicant_id = eval_dict["applicant_id"]
        applicant = applicants_map.get(applicant_id)
        if not applicant:
            continue

        predicted_hired = eval_dict["predicted_hired"]
        actual_hired = applicant["hired"]

        if predicted_hired != actual_hired:
            # This is a failure
            agg_score = eval_dict["aggregate_score"]
            confidence = abs(agg_score - 3.0)  # Distance from threshold
            error_magnitude = confidence

            failures.append({
                "applicant_id": applicant_id,
                "applicant": applicant,
                "predicted_hired": predicted_hired,
                "actual_hired": actual_hired,
                "aggregate_score": agg_score,
                "scores_json": eval_dict["scores_json"],
                "error_magnitude": error_magnitude,
            })

    # Sort by error magnitude (worst first)
    failures.sort(key=lambda x: x["error_magnitude"], reverse=True)

    return failures[:k]
