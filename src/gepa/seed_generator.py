"""Seed prompt generation for GEPA."""

import json
from ..llm.client import LLMClient
from ..config import JobDescription


SEED_GENERATION_PROMPT = """You are an expert prompt engineer designing evaluation prompts for hiring.

Design {n_seeds} alternative sets of 5 evaluation prompts for this job role.

Job Title: {job_title}
Required Skills: {required_skills}
Preferred Skills: {preferred_skills}
Selection Criteria:
{selection_criteria}

For each of the {n_seeds} prompt sets, design 5 independent evaluation lenses that together capture
the most important hiring signals for this role. Each prompt should be specific, measurable, and
actionable.

The prompts will be shown to evaluators who score applicants 1-5.

Return a JSON array with {n_seeds} objects, each with this structure:
{{
  "prompts": [
    {{"name": "Lens Name", "prompt": "Detailed evaluation prompt..."}},
    ...5 prompts total...
  ]
}}

Each prompt should be 2-3 sentences, specific to the role and selection criteria.
Vary the perspective and focus across the 5 lenses within each set.
Make the {n_seeds} sets diverse from each other - different angles on evaluation.

Return ONLY valid JSON. No other text.
"""


def generate_seed_prompts(
    client: LLMClient,
    job_description: JobDescription,
    n_seeds: int = 3,
) -> list[list[dict]]:
    """Generate seed prompt sets."""

    prompt = SEED_GENERATION_PROMPT.format(
        n_seeds=n_seeds,
        job_title=job_description.title,
        required_skills=", ".join(job_description.required_skills),
        preferred_skills=", ".join(job_description.preferred_skills),
        selection_criteria=job_description.selection_criteria,
    )

    response = client.call(prompt, phase="generation", max_tokens=4096)

    try:
        data = json.loads(response.content)
        if isinstance(data, list):
            prompt_sets = [item.get("prompts", []) for item in data]
        else:
            prompt_sets = [data.get("prompts", [])]
    except json.JSONDecodeError:
        print(f"Failed to parse seed prompts: {response.content[:200]}")
        prompt_sets = []

    # Filter to valid sets (must have exactly 5 prompts)
    valid_sets = [ps for ps in prompt_sets if len(ps) == 5]

    if not valid_sets:
        print(f"No valid seed prompt sets generated, returning empty")

    return valid_sets[:n_seeds]
