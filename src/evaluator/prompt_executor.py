"""Prompt execution for evaluating applicants."""

import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
from ..llm.client import LLMClient
from .. import database


@dataclass
class EvaluationResult:
    applicant_id: int
    candidate_id: int
    scores: list[dict]
    aggregate_score: float
    predicted_hired: int


def parse_eval_response(raw: str) -> dict:
    """Parse evaluation response from LLM with fallbacks."""
    try:
        # Try direct JSON parse
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from response
    try:
        match = re.search(r"\{[^{}]*\}", raw)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass

    # Fallback: return neutral score with error
    return {"score": 3, "rationale": f"PARSE_ERROR: {raw[:100]}"}


def evaluate_single(
    client: LLMClient,
    applicant: dict,
    prompts: list[dict],
    candidate_id: int,
    threshold: float = 3.0,
) -> EvaluationResult:
    """Evaluate a single applicant against a set of prompts."""
    resume_json = applicant["resume_json"]
    if isinstance(resume_json, str):
        resume_json = json.loads(resume_json)

    scores = []
    total_score = 0.0

    for prompt_dict in prompts:
        lens_name = prompt_dict.get("name", "unknown")
        prompt_text = prompt_dict.get("prompt", "")

        # Create strict evaluation prompt
        system_prompt = "You are an expert technical recruiter evaluating job applicants."
        eval_prompt = f"""{lens_name}
{prompt_text}

Applicant Profile (JSON):
{json.dumps(resume_json, indent=2)}"""

        response = client.call(
            eval_prompt,
            system=system_prompt,
            max_tokens=256,
            phase="rollout",
        )

        parsed = parse_eval_response(response.content)
        score = parsed.get("score", 3)
        rationale = parsed.get("rationale", "")

        # Ensure score is int in range [1, 5]
        if not isinstance(score, int):
            score = int(score) if isinstance(score, float) else 3
        score = max(1, min(5, score))

        scores.append({
            "lens": lens_name,
            "score": score,
            "rationale": rationale,
        })
        total_score += score

    aggregate_score = total_score / len(prompts) if prompts else 3.0
    predicted_hired = 1 if aggregate_score >= threshold else 0

    return EvaluationResult(
        applicant_id=applicant["id"],
        candidate_id=candidate_id,
        scores=scores,
        aggregate_score=aggregate_score,
        predicted_hired=predicted_hired,
    )


def evaluate_applicant_set(
    client: LLMClient,
    conn: sqlite3.Connection,
    applicants: list[dict],
    candidate_id: int,
    threshold: float = 3.0,
    show_progress: bool = True,
) -> list[EvaluationResult]:
    """Evaluate all applicants against a prompt candidate."""

    # Get prompts for this candidate
    candidate = database.get_candidate(conn, candidate_id)
    if not candidate:
        raise ValueError(f"Candidate {candidate_id} not found")

    prompts_json = candidate["prompts_json"]
    if isinstance(prompts_json, str):
        prompts = json.loads(prompts_json)
    else:
        prompts = prompts_json

    results = []
    iterator = tqdm(applicants, desc=f"Evaluating candidate {candidate_id}") if show_progress else applicants

    for applicant in iterator:
        result = evaluate_single(client, applicant, prompts, candidate_id, threshold)
        results.append(result)

        # Insert evaluation immediately
        database.insert_evaluation(
            conn,
            candidate_id=result.candidate_id,
            applicant_id=result.applicant_id,
            scores=result.scores,
            aggregate_score=result.aggregate_score,
            predicted_hired=result.predicted_hired,
        )

    return results
