"""Pareto frontier management for GEPA."""

import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParetoCandidate:
    candidate_id: int
    accuracy: float
    precision_hired: float


def dominates(a: ParetoCandidate, b: ParetoCandidate) -> bool:
    """Check if candidate a dominates candidate b."""
    a_better_or_equal_acc = a.accuracy >= b.accuracy
    a_better_or_equal_prec = a.precision_hired >= b.precision_hired

    a_strictly_better_acc = a.accuracy > b.accuracy
    a_strictly_better_prec = a.precision_hired > b.precision_hired

    return (a_better_or_equal_acc and a_better_or_equal_prec) and (
        a_strictly_better_acc or a_strictly_better_prec
    )


def update_pareto_frontier(
    current_pool: list[ParetoCandidate],
    new_candidate: ParetoCandidate,
    max_size: int = 20,
) -> tuple[list[ParetoCandidate], bool]:
    """
    Attempt to add new_candidate to the Pareto pool.
    Returns (updated_pool, was_added: bool).
    """

    # Check if new_candidate is dominated by any existing candidate
    for existing in current_pool:
        if dominates(existing, new_candidate):
            return current_pool, False

    # Remove all candidates that new_candidate dominates
    updated_pool = [c for c in current_pool if not dominates(new_candidate, c)]

    # Add new_candidate
    updated_pool.append(new_candidate)

    # If pool exceeds max_size, remove the worst candidate
    if len(updated_pool) > max_size:
        # Remove candidate with lowest combined score
        worst = min(updated_pool, key=lambda c: c.accuracy + c.precision_hired)
        updated_pool.remove(worst)

    return updated_pool, True


def select_parent(
    pool: list[ParetoCandidate],
    strategy: str = "random",
) -> Optional[ParetoCandidate]:
    """Select a parent from the Pareto pool."""

    if not pool:
        return None

    if strategy == "random":
        return random.choice(pool)

    elif strategy == "tournament":
        # Tournament selection: pick best of 3 random candidates
        candidates = random.sample(pool, min(3, len(pool)))
        return max(candidates, key=lambda c: c.accuracy + c.precision_hired)

    elif strategy == "best_accuracy":
        return max(pool, key=lambda c: c.accuracy)

    else:
        # Default to random
        return random.choice(pool)


def reconstruct_from_logs(
    iteration_logs: list[dict],
    candidate_ids: list[int],
) -> list[ParetoCandidate]:
    """Reconstruct Pareto candidates from iteration logs."""

    candidates_dict = {}
    for log in iteration_logs:
        cid = log["candidate_id"]
        if cid in candidate_ids:
            if cid not in candidates_dict or log["iteration"] > candidates_dict[cid]["iteration"]:
                candidates_dict[cid] = log

    candidates = [
        ParetoCandidate(
            candidate_id=cid,
            accuracy=log.get("accuracy", 0.0) or 0.0,
            precision_hired=log.get("precision_h", 0.0) or 0.0,
        )
        for cid, log in candidates_dict.items()
    ]

    return candidates
