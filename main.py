#!/usr/bin/env python3
"""GEPA POC orchestrator: runs phases of the optimization system."""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

from src.config import load_config
from src.database import get_db, init_schema
from src.llm.client import create_llm_client
from src.data_pipeline.dataset_manager import create_dataset, validate_dataset
from src.evaluator.metrics import compute_metrics_from_db
from src.gepa.optimizer import run_optimization
from src.gepa.rollout import rollout_candidate
from src import database


# Load environment variables
load_dotenv()


def phase_prep(conn, client, config):
    """Phase: Data preparation."""
    print("\n" + "="*60)
    print("PHASE: DATA PREPARATION")
    print("="*60)

    create_dataset(conn, client, config)

    print("\nValidating dataset...")
    stats = validate_dataset(conn)
    if not stats.get("is_valid"):
        print(f"⚠️  Dataset validation issues:")
        for issue in stats.get("validation_issues", []):
            print(f"  - {issue}")
    else:
        print("✅ Dataset validated")

    print("\nDataset Stats:")
    for key, value in stats.items():
        if key not in ["validation_issues", "is_valid"]:
            print(f"  {key}: {value}")


def phase_baseline(conn, client, config):
    """Phase: Baseline evaluation."""
    print("\n" + "="*60)
    print("PHASE: BASELINE EVALUATION")
    print("="*60)

    # Check if dataset exists
    if database.count_applicants(conn) == 0:
        print("❌ No dataset found. Run --phase prep first.")
        sys.exit(1)

    # Insert baseline prompts as iteration=0 candidate
    baseline_prompts = [
        {"name": p.name, "prompt": p.prompt} for p in config.baseline_prompts
    ]
    baseline_id = database.insert_prompt_candidate(
        conn, iteration=0, parent_id=None, prompts=baseline_prompts
    )
    print(f"Created baseline candidate: {baseline_id}")

    # Get applicants
    train_applicants = database.get_applicants_by_split(conn, "train")
    holdout_applicants = database.get_applicants_by_split(conn, "holdout")

    # Rollout on both splits
    print("\nEvaluating baseline on training set...")
    train_metrics = rollout_candidate(conn, client, baseline_id, split="train")

    print("\nEvaluating baseline on holdout set...")
    holdout_metrics = rollout_candidate(conn, client, baseline_id, split="holdout")

    # Log baseline results
    database.insert_iteration_log(
        conn,
        iteration=0,
        candidate_id=baseline_id,
        split="train",
        accuracy=train_metrics["accuracy"],
        precision_h=train_metrics["precision_hired"],
        recall_h=train_metrics["recall_hired"],
        f1_h=train_metrics["f1_hired"],
        spearman_rho=train_metrics["spearman_rho"],
        is_pareto=1,
        notes="Baseline (human-authored)",
    )
    database.insert_iteration_log(
        conn,
        iteration=0,
        candidate_id=baseline_id,
        split="holdout",
        accuracy=holdout_metrics["accuracy"],
        precision_h=holdout_metrics["precision_hired"],
        recall_h=holdout_metrics["recall_hired"],
        f1_h=holdout_metrics["f1_hired"],
        spearman_rho=holdout_metrics["spearman_rho"],
        is_pareto=1,
        notes="Baseline (human-authored)",
    )

    print("\n✅ Baseline Evaluation Results:")
    print(f"  Train accuracy: {train_metrics['accuracy']:.3f}")
    print(f"  Train precision (hired): {train_metrics['precision_hired']:.3f}")
    print(f"  Holdout accuracy: {holdout_metrics['accuracy']:.3f}")
    print(f"  Holdout precision (hired): {holdout_metrics['precision_hired']:.3f}")


def phase_optimize(conn, client, config, resume=False):
    """Phase: GEPA optimization."""
    print("\n" + "="*60)
    print("PHASE: GEPA OPTIMIZATION")
    print("="*60)

    # Check if baseline exists
    baseline_logs = [log for log in database.get_iteration_logs(conn) if log["iteration"] == 0]
    if not baseline_logs:
        print("❌ No baseline found. Run --phase baseline first.")
        sys.exit(1)

    print(f"Budget: ${database.get_total_cost(conn):.2f} / ${config.llm.budget_hard_cap_usd:.2f}")
    print(f"Calls: {database.get_total_calls(conn)} / {config.llm.max_calls_per_run}")

    best_id = run_optimization(conn, client, config, resume=resume)

    print(f"\n✅ Optimization Complete")
    print(f"Budget used: ${database.get_total_cost(conn):.2f} / ${config.llm.budget_hard_cap_usd:.2f}")
    print(f"Total calls: {database.get_total_calls(conn)} / {config.llm.max_calls_per_run}")


def phase_evaluate(conn, client, config):
    """Phase: Evaluation and comparison."""
    print("\n" + "="*60)
    print("PHASE: EVALUATE AND COMPARE")
    print("="*60)

    # Get best candidate from Pareto pool
    state = database.get_optimization_state(conn)
    if not state or state["status"] == "running":
        print("⚠️  Optimization not complete. Running evaluation on best found so far...")

    pareto_ids = state.get("pareto_pool_ids", []) if state else []
    if not pareto_ids:
        print("❌ No candidates found")
        sys.exit(1)

    # Find best by training accuracy
    logs = database.get_iteration_logs(conn)
    train_logs = [l for l in logs if l["split"] == "train" and l["candidate_id"] in pareto_ids]
    if not train_logs:
        print("❌ No training evaluations found")
        sys.exit(1)

    best_candidate_id = max(train_logs, key=lambda l: l["accuracy"])["candidate_id"]

    baseline_id = database.get_baseline_candidate_id(conn)
    if not baseline_id:
        print("❌ No baseline found. Run --phase baseline first.")
        sys.exit(1)

    # Evaluate baseline and best on holdout (fresh metrics, not stale logs)
    print(f"\nEvaluating baseline ({baseline_id}) on holdout set...")
    baseline_metrics = rollout_candidate(conn, client, baseline_id, split="holdout")

    print(f"\nEvaluating best candidate ({best_candidate_id}) on holdout set...")
    best_metrics = rollout_candidate(conn, client, best_candidate_id, split="holdout")

    database.insert_iteration_log(
        conn,
        iteration=0,
        candidate_id=baseline_id,
        split="holdout",
        accuracy=baseline_metrics["accuracy"],
        precision_h=baseline_metrics["precision_hired"],
        recall_h=baseline_metrics["recall_hired"],
        f1_h=baseline_metrics["f1_hired"],
        spearman_rho=baseline_metrics["spearman_rho"],
        is_pareto=1,
        notes="Baseline (human-authored)",
    )

    # Log best result
    database.insert_iteration_log(
        conn,
        iteration=99,  # Special iteration number for final evaluation
        candidate_id=best_candidate_id,
        split="holdout",
        accuracy=best_metrics["accuracy"],
        precision_h=best_metrics["precision_hired"],
        recall_h=best_metrics["recall_hired"],
        f1_h=best_metrics["f1_hired"],
        spearman_rho=best_metrics["spearman_rho"],
        is_pareto=1,
        notes="Final holdout evaluation",
    )

    # Print comparison
    print("\n" + "="*60)
    print("HOLDOUT SET COMPARISON")
    print("="*60)

    baseline_acc = baseline_metrics["accuracy"]
    best_acc = best_metrics["accuracy"]
    if baseline_acc > 0:
        improvement = (best_acc - baseline_acc) / baseline_acc
    else:
        improvement = best_acc - baseline_acc  # absolute gain when baseline is 0

    print(f"\nBaseline (human-authored, candidate {baseline_id}):")
    print(f"  Accuracy: {baseline_acc:.3f}")
    print(f"  Precision (hired): {baseline_metrics['precision_hired']:.3f}")
    print(f"  Recall (hired): {baseline_metrics['recall_hired']:.3f}")

    print(f"\nOptimized (GEPA):")
    print(f"  Accuracy: {best_acc:.3f}")
    print(f"  Precision (hired): {best_metrics['precision_hired']:.3f}")
    print(f"  Recall (hired): {best_metrics['recall_hired']:.3f}")

    print(f"\nImprovement:")
    if baseline_acc > 0:
        print(f"  Accuracy: {improvement*100:+.1f}%")
        success = improvement >= 0.15
    else:
        print(f"  Accuracy: {improvement:+.3f} absolute (baseline was 0%)")
        success = improvement >= 0.15
    if success:
        label = f"{improvement*100:.1f}%" if baseline_acc > 0 else f"{improvement:.3f} absolute"
        print(f"  ✅ SUCCESS: {label} improvement (target: ≥15%)")
    else:
        label = f"{improvement*100:.1f}%" if baseline_acc > 0 else f"{improvement:.3f} absolute"
        print(f"  ⚠️  Below target: {label} improvement (target: ≥15%)")

    print(f"\nBest candidate ID: {best_candidate_id}")


def phase_dashboard():
    """Phase: Launch Streamlit dashboard."""
    print("\n" + "="*60)
    print("PHASE: DASHBOARD")
    print("="*60)
    print("\nLaunching Streamlit dashboard...")

    project_root = Path(__file__).parent.resolve()
    streamlit_path = project_root / "src" / "dashboard" / "app.py"
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(project_root) + (os.pathsep + pythonpath if pythonpath else "")
    )
    subprocess.run(
        ["streamlit", "run", str(streamlit_path)],
        env=env,
        cwd=str(project_root),
    )


def phase_smoke_test(conn, client, config):
    """Smoke test: verify end-to-end with minimal data."""
    print("\n" + "="*60)
    print("SMOKE TEST")
    print("="*60)

    # Create minimal dataset (10 applicants)
    print("Smoke test: creating minimal dataset (10 applicants)...")
    config_copy = config
    config_copy.dataset.total = 10
    config_copy.dataset.train = 7
    config_copy.dataset.holdout = 3
    config_copy.gepa.max_iterations = 3

    # Run all phases
    phase_prep(conn, client, config_copy)
    phase_baseline(conn, client, config_copy)
    phase_optimize(conn, client, config_copy)
    phase_evaluate(conn, client, config_copy)

    print("\n✅ Smoke test passed!")


def main():
    parser = argparse.ArgumentParser(description="GEPA POC orchestrator")
    parser.add_argument(
        "--phase",
        choices=["all", "prep", "baseline", "optimize", "evaluate", "dashboard", "smoke-test"],
        default="all",
        help="Phase to run",
    )
    parser.add_argument("--resume", action="store_true", help="Resume optimization from checkpoint")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--db", default="data/gepa.db", help="Database path")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Initialize database
    conn = get_db(args.db)
    init_schema(conn)

    # Initialize LLM client (Bedrock or Ollama)
    llm_timeout = (
        config.ollama.timeout_seconds
        if config.llm_provider == "ollama" and config.ollama
        else config.llm.timeout_seconds
    )
    client = create_llm_client(
        llm_provider=config.llm_provider,
        model_rollout=config.llm.model_rollout,
        model_reflection=config.llm.model_reflection,
        model_generation=config.llm.model_generation,
        max_retries=config.llm.max_retries,
        timeout_seconds=llm_timeout,
        budget_hard_cap=config.llm.budget_hard_cap_usd,
        budget_warn_threshold=config.llm.budget_warn_threshold_usd,
        max_calls_per_run=config.llm.max_calls_per_run,
        db_conn=conn,
        ollama_base_url=config.ollama.base_url if config.ollama else None,
        ollama_model=config.ollama.model if config.ollama else None,
    )

    # Run phases
    try:
        if args.phase == "all":
            phase_prep(conn, client, config)
            phase_baseline(conn, client, config)
            phase_optimize(conn, client, config)
            phase_evaluate(conn, client, config)
            print("\n" + "="*60)
            print("✅ ALL PHASES COMPLETE")
            print("="*60)
        elif args.phase == "prep":
            phase_prep(conn, client, config)
        elif args.phase == "baseline":
            phase_baseline(conn, client, config)
        elif args.phase == "optimize":
            phase_optimize(conn, client, config, resume=args.resume)
        elif args.phase == "evaluate":
            phase_evaluate(conn, client, config)
        elif args.phase == "dashboard":
            phase_dashboard()
        elif args.phase == "smoke-test":
            phase_smoke_test(conn, client, config)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
