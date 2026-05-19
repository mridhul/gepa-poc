"""Meta-LLM reflection and prompt mutation for GEPA."""

import json
from ..llm.client import LLMClient
from ..config import JobDescription


REFLECTION_PROMPT_TEMPLATE = """You are an expert at improving AI evaluation systems through targeted mutation.

CURRENT EVALUATION SYSTEM (5 prompts):
{current_prompts}

JOB DESCRIPTION:
Title: {job_title}
Required Skills: {required_skills}
Preferred Skills: {preferred_skills}
Selection Criteria:
{selection_criteria}

FAILURE CASES (candidates the system incorrectly classified):
{failures}

Each failure shows:
- Applicant profile (name, experience, skills, education, work history)
- System's prediction (hired/rejected) vs actual decision
- Per-lens scores and rationales that led to the wrong conclusion

TASK:
1. Analyze the failure cases to identify patterns
2. Determine which of the 5 prompts is most responsible for the errors
3. Diagnose the root cause (e.g., "overweighs PhD vs practical experience", "misses leadership signals")
4. Propose ONE specific edit to ONE prompt that would address the most common failure pattern

Return ONLY valid JSON:
{{
  "diagnosis": "<2-3 sentence explanation of the failure pattern>",
  "prompt_index": <0-4>,
  "prompt_name": "<unchanged name from above>",
  "proposed_prompt": "<full replacement prompt text, 2-3 sentences>",
  "reasoning": "<why this change addresses the failures>"
}}

Be surgical and specific. The change should directly address the identified pattern.
"""


def reflect_and_mutate(
    client: LLMClient,
    parent_candidate: dict,
    failure_cases: list[dict],
    job_description: JobDescription,
) -> list[dict]:
    """
    Use meta-LLM to reflect on failures and propose mutation.
    Returns new prompt set (4 unchanged + 1 mutated).
    """

    # Parse parent prompts
    parent_prompts_json = parent_candidate.get("prompts_json")
    if isinstance(parent_prompts_json, str):
        parent_prompts = json.loads(parent_prompts_json)
    else:
        parent_prompts = parent_prompts_json

    # Format current prompts for reflection
    formatted_prompts = "\n".join([
        f'{i}. {p["name"]}\n   {p["prompt"]}'
        for i, p in enumerate(parent_prompts)
    ])

    # Format failure cases
    formatted_failures = "\n\n".join([
        f'Applicant: {f["applicant_name"]}\n'
        f'  Predicted: {"hired" if f["predicted_hired"] else "rejected"} | '
        f'Actual: {"hired" if f["actual_hired"] else "rejected"}\n'
        f'  Score: {f["aggregate_score"]:.1f}/5\n'
        f'  Details: {json.dumps(f.get("scores_json", []), indent=4)}'
        for f in failure_cases[:10]  # Limit to 10 failures
    ])

    reflection_prompt = REFLECTION_PROMPT_TEMPLATE.format(
        current_prompts=formatted_prompts,
        job_title=job_description.title,
        required_skills=", ".join(job_description.required_skills),
        preferred_skills=", ".join(job_description.preferred_skills),
        selection_criteria=job_description.selection_criteria,
        failures=formatted_failures if failure_cases else "(No failures provided)",
    )

    response = client.call(reflection_prompt, phase="reflection", max_tokens=1024)

    try:
        proposal = json.loads(response.content)
    except json.JSONDecodeError:
        print(f"Failed to parse reflection response: {response.content[:200]}")
        # Return unchanged prompts as fallback
        return parent_prompts

    # Apply mutation: change one prompt
    prompt_index = proposal.get("prompt_index", 0)
    if not (0 <= prompt_index < len(parent_prompts)):
        prompt_index = 0

    new_prompts = parent_prompts.copy()
    new_prompts[prompt_index] = {
        "name": proposal.get("prompt_name", parent_prompts[prompt_index]["name"]),
        "prompt": proposal.get("proposed_prompt", parent_prompts[prompt_index]["prompt"]),
    }

    return new_prompts
