"""Main GEPA optimization loop."""

import sqlite3
from tqdm import tqdm
from ..llm.client import LLMClient, BudgetExhaustedError
from ..config import Config
from .. import database
from .pareto import ParetoCandidate, update_pareto_frontier, select_parent, reconstruct_from_logs
from .rollout import rollout_candidate, get_worst_cases
from .seed_generator import generate_seed_prompts
from .reflector import reflect_and_mutate


def run_optimization(
    conn: sqlite3.Connection,
    client: LLMClient,
    config: Config,
    resume: bool = False,
) -> int:
    """
    Main GEPA optimization loop.
    Returns best_candidate_id.
    """

    # Load or initialize state
    state = database.get_optimization_state(conn)
    if resume and state and state["status"] == "paused":
        print(f"Resuming optimization from iteration {state['current_iteration']}")
        current_iteration = state["current_iteration"]
        pareto_pool_ids = state["pareto_pool_ids"]
        total_cost = state.get("total_cost_usd", 0.0)
        total_calls = state.get("total_llm_calls", 0)
        optimal_threshold = state.get("optimal_threshold", 3.0)
    else:
        print("Starting new optimization run")
        current_iteration = 0
        pareto_pool_ids = []
        total_cost = 0.0
        total_calls = 0

        # Generate seed candidates
        print("Generating seed candidates...")
        seed_prompts_list = generate_seed_prompts(
            client, config.job_description, config.gepa.seed_candidates
        )

        # Rollout seeds and init Pareto pool
        pareto_pool = []
        for seed_idx, seed_prompts in enumerate(seed_prompts_list):
            seed_id = database.insert_prompt_candidate(
                conn, iteration=0, parent_id=None, prompts=seed_prompts
            )

            metrics = rollout_candidate(
                conn, client, seed_id, split="train", threshold=3.0
            )

            candidate = ParetoCandidate(
                candidate_id=seed_id,
                accuracy=metrics["accuracy"],
                precision_hired=metrics["precision_hired"],
            )
            pareto_pool, added = update_pareto_frontier(
                pareto_pool, candidate, config.gepa.pareto_max_size
            )

            database.insert_iteration_log(
                conn,
                iteration=0,
                candidate_id=seed_id,
                split="train",
                accuracy=metrics["accuracy"],
                precision_h=metrics["precision_hired"],
                recall_h=metrics["recall_hired"],
                f1_h=metrics["f1_hired"],
                spearman_rho=metrics["spearman_rho"],
                is_pareto=1 if added else 0,
                notes=f"Seed {seed_idx}",
            )

        pareto_pool_ids = [c.candidate_id for c in pareto_pool]
        optimal_threshold = 3.0
        database.upsert_optimization_state(
            conn, "paused", 0, pareto_pool_ids, database.get_total_calls(conn),
            database.get_total_cost(conn), optimal_threshold
        )

    # Main GEPA loop
    try:
        for iteration in range(current_iteration + 1, config.gepa.max_iterations + 1):
            print(f"\n=== Iteration {iteration}/{config.gepa.max_iterations} ===")

            # Reconstruct Pareto pool from DB
            logs = database.get_iteration_logs(conn)
            pareto_pool = reconstruct_from_logs(logs, pareto_pool_ids)

            if not pareto_pool:
                print("Pareto pool is empty, stopping")
                break

            # Select parent
            parent = select_parent(pareto_pool, strategy=config.gepa.selection_strategy)
            print(f"Selected parent {parent.candidate_id} (acc={parent.accuracy:.3f}, prec={parent.precision_hired:.3f})")

            # Get worst cases
            parent_candidate = database.get_candidate(conn, parent.candidate_id)
            failures = get_worst_cases(conn, parent.candidate_id, split="train", k=config.gepa.worst_cases_k)
            print(f"Found {len(failures)} failure cases")

            # Reflect and mutate
            new_prompts = reflect_and_mutate(
                client, parent_candidate, failures, config.job_description
            )

            # Insert new candidate
            new_id = database.insert_prompt_candidate(
                conn, iteration=iteration, parent_id=parent.candidate_id, prompts=new_prompts
            )

            # Rollout
            metrics = rollout_candidate(
                conn, client, new_id, split="train", threshold=optimal_threshold
            )

            # Update Pareto frontier
            new_candidate = ParetoCandidate(
                candidate_id=new_id,
                accuracy=metrics["accuracy"],
                precision_hired=metrics["precision_hired"],
            )
            pareto_pool, added = update_pareto_frontier(
                pareto_pool, new_candidate, config.gepa.pareto_max_size
            )

            pareto_pool_ids = [c.candidate_id for c in pareto_pool]

            # Log iteration
            database.insert_iteration_log(
                conn,
                iteration=iteration,
                candidate_id=new_id,
                split="train",
                accuracy=metrics["accuracy"],
                precision_h=metrics["precision_hired"],
                recall_h=metrics["recall_hired"],
                f1_h=metrics["f1_hired"],
                spearman_rho=metrics["spearman_rho"],
                is_pareto=1 if added else 0,
                notes=f"Parent={parent.candidate_id}",
            )

            print(f"  Metrics: acc={metrics['accuracy']:.3f}, prec={metrics['precision_hired']:.3f}")
            print(f"  Pareto pool size: {len(pareto_pool)}")

            # Checkpoint
            if iteration % config.gepa.checkpoint_every == 0:
                database.upsert_optimization_state(
                    conn, "paused", iteration, pareto_pool_ids, database.get_total_calls(conn),
                    database.get_total_cost(conn), optimal_threshold
                )
                print(f"Checkpoint saved")

    except BudgetExhaustedError as e:
        print(f"\n⚠️  {e}")
        database.upsert_optimization_state(
            conn, "paused_budget", iteration - 1, pareto_pool_ids,
            database.get_total_calls(conn), database.get_total_cost(conn), optimal_threshold
        )
        print("Optimization paused due to budget limit")

    # Mark as done
    database.upsert_optimization_state(
        conn, "done", config.gepa.max_iterations, pareto_pool_ids,
        database.get_total_calls(conn), database.get_total_cost(conn), optimal_threshold
    )

    # Return best candidate (highest accuracy)
    best_candidate = max(pareto_pool, key=lambda c: c.accuracy)
    print(f"\n✅ Optimization complete. Best candidate: {best_candidate.candidate_id}")

    return best_candidate.candidate_id
