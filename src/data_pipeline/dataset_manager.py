"""Dataset management: creation, splitting, validation."""

import sqlite3
from sklearn.model_selection import StratifiedShuffleSplit
from ..llm.client import LLMClient
from ..config import Config
from .synthetic_generator import generate_all, oracle_hire_decision
from .. import database


def create_dataset(
    conn: sqlite3.Connection,
    client: LLMClient,
    config: Config,
) -> None:
    """Create synthetic dataset (idempotent)."""

    # Check if already exists
    existing_count = database.count_applicants(conn)
    if existing_count > 0:
        print(f"Dataset already exists with {existing_count} applicants, skipping generation.")
        return

    print(f"Generating {config.dataset.total} synthetic resumes...")
    resumes = generate_all(
        client,
        config.dataset.total,
        config.job_description,
        config.dataset.batch_size,
        max_tokens=config.llm.max_tokens_generation,
    )

    # Apply oracle labels
    print("Applying oracle labels...")
    labeled_resumes = []
    for resume in resumes:
        hired = oracle_hire_decision(resume, config.oracle)
        labeled_resumes.append((resume, hired))

    # Perform stratified split
    print("Performing stratified train/holdout split...")
    applicant_ids = list(range(len(labeled_resumes)))
    hire_labels = [hired for _, hired in labeled_resumes]

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.dataset.holdout,
        random_state=42,
    )

    train_indices, holdout_indices = next(splitter.split(applicant_ids, hire_labels))

    # Insert into database with splits
    print("Inserting into database...")
    for idx, (resume, hired) in enumerate(labeled_resumes):
        split = "train" if idx in train_indices else "holdout"
        database.insert_applicant(
            conn,
            name=resume.get("name", f"Applicant_{idx}"),
            resume_json=resume,
            hired=hired,
            split=split,
            source="synthetic",
        )

    print(f"Dataset created: {len(train_indices)} train, {len(holdout_indices)} holdout")


def validate_dataset(conn: sqlite3.Connection) -> dict:
    """Validate dataset quality."""
    applicants = database.get_all_applicants(conn)

    if not applicants:
        return {"error": "No applicants in database"}

    train_applicants = [a for a in applicants if a["split"] == "train"]
    holdout_applicants = [a for a in applicants if a["split"] == "holdout"]

    train_hired = sum(1 for a in train_applicants if a["hired"] == 1)
    holdout_hired = sum(1 for a in holdout_applicants if a["hired"] == 1)

    train_hire_ratio = train_hired / len(train_applicants) if train_applicants else 0
    holdout_hire_ratio = holdout_hired / len(holdout_applicants) if holdout_applicants else 0

    stats = {
        "total_applicants": len(applicants),
        "train_count": len(train_applicants),
        "holdout_count": len(holdout_applicants),
        "train_hired": train_hired,
        "train_rejected": len(train_applicants) - train_hired,
        "train_hire_ratio": round(train_hire_ratio, 3),
        "holdout_hired": holdout_hired,
        "holdout_rejected": len(holdout_applicants) - holdout_hired,
        "holdout_hire_ratio": round(holdout_hire_ratio, 3),
    }

    # Validation checks
    issues = []
    if len(applicants) < 100:
        issues.append(f"Dataset too small: {len(applicants)} applicants")
    if train_hire_ratio < 0.2 or train_hire_ratio > 0.6:
        issues.append(f"Train hire ratio {train_hire_ratio:.1%} out of expected range [20%, 60%]")
    if holdout_hire_ratio < 0.2 or holdout_hire_ratio > 0.6:
        issues.append(f"Holdout hire ratio {holdout_hire_ratio:.1%} out of expected range [20%, 60%]")

    stats["validation_issues"] = issues
    stats["is_valid"] = len(issues) == 0

    return stats
