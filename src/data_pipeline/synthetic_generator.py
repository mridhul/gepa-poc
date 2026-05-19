"""Synthetic resume generation."""

import json
import hashlib
import random
from typing import Optional
from ..llm.client import LLMClient
from ..config import JobDescription, OracleConfig


GENERATION_PROMPT_TEMPLATE = """Generate {count} realistic software engineer resumes in valid JSON format.

Job context: {job_title}
Required skills: {required_skills}
Preferred skills: {preferred_skills}

For each resume, create a JSON object with this exact structure:
{{
  "name": "Full Name",
  "years_experience": <integer 0-20>,
  "skills": ["skill1", "skill2", ...],
  "education": {{
    "degree": "BS|MS|PhD|Bootcamp|None",
    "field": "field of study",
    "university": "university name"
  }},
  "work_history": [
    {{
      "company": "company name",
      "role": "job title",
      "years": <integer>,
      "highlights": ["achievement1", "achievement2", ...]
    }}
  ],
  "projects": ["project description", ...],
  "summary": "2-3 sentence professional summary"
}}

Vary the profiles across experience levels (0-2 yrs, 3-5 yrs, 6-10 yrs, 10+ yrs) and skillsets.
Make them realistic and diverse.

Return ONLY a JSON array of {count} resume objects. No other text.
"""


def generate_batch(
    client: LLMClient,
    count: int,
    job_description: JobDescription,
    batch_number: int,
    max_tokens: int = 4096,
) -> list[dict]:
    """Generate a batch of resumes."""
    prompt = GENERATION_PROMPT_TEMPLATE.format(
        count=count,
        job_title=job_description.title,
        required_skills=", ".join(job_description.required_skills),
        preferred_skills=", ".join(job_description.preferred_skills),
    )

    response = client.call(prompt, phase="generation", max_tokens=max_tokens)

    try:
        # Try to parse as JSON array
        resumes = json.loads(response.content)
        if not isinstance(resumes, list):
            resumes = [resumes]
    except json.JSONDecodeError:
        # Try to extract JSON array from response
        import re
        match = re.search(r"\[.*\]", response.content, re.DOTALL)
        if match:
            try:
                resumes = json.loads(match.group())
            except json.JSONDecodeError:
                print(f"Failed to parse resumesbatch {batch_number}: {response.content[:200]}")
                resumes = []
        else:
            print(f"No JSON found in batch {batch_number}")
            resumes = []

    return resumes


def generate_all(
    client: LLMClient,
    total: int,
    job_description: JobDescription,
    batch_size: int = 10,
    max_tokens: int = 4096,
) -> list[dict]:
    """Generate all resumes in batches."""
    all_resumes = []
    num_batches = (total + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        # Last batch might be smaller
        batch_count = min(batch_size, total - batch_num * batch_size)
        print(f"Generating batch {batch_num + 1}/{num_batches} ({batch_count} resumes)...")

        batch_resumes = generate_batch(
            client, batch_count, job_description, batch_num, max_tokens=max_tokens
        )
        all_resumes.extend(batch_resumes)

        if len(all_resumes) >= total:
            all_resumes = all_resumes[:total]
            break

    print(f"Generated {len(all_resumes)} resumes")
    return all_resumes


def oracle_hire_decision(resume: dict, oracle_config: OracleConfig) -> int:
    """Deterministic oracle: decide hire (1) or reject (0) based on rules + seeded noise."""

    # Base rule: min years experience + required skills
    years = resume.get("years_experience", 0)
    skills = [s.lower() for s in resume.get("skills", [])]

    hire_prob = 0.0

    # Check if required skills present
    required_present = sum(
        1 for req_skill in oracle_config.required_skills
        if any(req_skill.lower() in skill for skill in skills)
    )

    if required_present >= len(oracle_config.required_skills) and years >= oracle_config.min_years_experience:
        hire_prob = 0.85  # High probability if meets base criteria
    elif required_present >= len(oracle_config.required_skills) - 1 and years >= oracle_config.min_years_experience - 1:
        hire_prob = 0.50  # Medium probability if close
    elif years >= oracle_config.min_years_experience:
        hire_prob = 0.30  # Some probability if has experience but missing skills
    else:
        hire_prob = 0.10  # Low baseline

    # Secondary signal: leadership/mentoring boost
    work_history = resume.get("work_history", [])
    all_highlights = []
    for job in work_history:
        all_highlights.extend(job.get("highlights", []))

    mentor_keywords = ["mentoring", "mentored", "led team", "team lead", "leadership", "managed", "lead"]
    has_mentor_signal = any(
        any(keyword in highlight.lower() for keyword in mentor_keywords)
        for highlight in all_highlights
    )

    if has_mentor_signal:
        hire_prob = min(1.0, hire_prob * 1.2)  # +20% boost

    # Apply seeded noise based on resume name hash (for reproducibility)
    name = resume.get("name", "")
    seed = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
    random.seed(seed)

    noise = random.random()
    if noise < oracle_config.noise_rate:
        # Flip the label
        decision = 0 if hire_prob > 0.5 else 1
    else:
        decision = 1 if hire_prob > 0.5 else 0

    return decision
