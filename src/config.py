"""Configuration loading and validation."""

import os
from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class JobDescription:
    title: str
    required_skills: list[str]
    preferred_skills: list[str]
    selection_criteria: str


@dataclass
class OracleConfig:
    required_skills: list[str]
    min_years_experience: int
    noise_rate: float


@dataclass
class DatasetConfig:
    total: int
    train: int
    holdout: int
    hire_ratio: float
    batch_size: int


@dataclass
class GEPAConfig:
    max_iterations: int
    pareto_max_size: int
    checkpoint_every: int
    num_prompts: int
    worst_cases_k: int
    seed_candidates: int
    selection_strategy: str = "random"


@dataclass
class LLMConfig:
    model_rollout: str
    model_reflection: str
    model_generation: str
    max_tokens_rollout: int
    max_tokens_reflection: int
    max_tokens_generation: int
    max_retries: int
    timeout_seconds: int
    budget_hard_cap_usd: float
    budget_warn_threshold_usd: float
    max_calls_per_run: int


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "mistral"
    timeout_seconds: int = 600


@dataclass
class BaselinePrompt:
    name: str
    prompt: str


@dataclass
class Config:
    job_description: JobDescription
    oracle: OracleConfig
    dataset: DatasetConfig
    gepa: GEPAConfig
    llm: LLMConfig
    baseline_prompts: list[BaselinePrompt]
    llm_provider: str = "bedrock"  # "bedrock" or "ollama"
    ollama: Optional[OllamaConfig] = None


def load_config(path: str = "config.yaml") -> Config:
    """Load and parse config.yaml, injecting environment variable overrides."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # Build dataclass objects
    job_desc = JobDescription(**raw["job_description"])
    oracle = OracleConfig(**raw["oracle"])
    dataset = DatasetConfig(**raw["dataset"])
    gepa = GEPAConfig(**raw["gepa"])
    llm = LLMConfig(**raw["llm"])
    baseline_prompts = [BaselinePrompt(**p) for p in raw["baseline_prompts"]]

    # Detect LLM provider from environment
    llm_provider = os.environ.get("LLM_PROVIDER", "bedrock").lower()

    # Load Ollama config if using Ollama
    ollama = None
    if llm_provider == "ollama":
        ollama = OllamaConfig(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.environ.get("OLLAMA_MODEL", "mistral"),
            timeout_seconds=int(os.environ.get("OLLAMA_TIMEOUT", "600")),
        )

    config = Config(
        job_description=job_desc,
        oracle=oracle,
        dataset=dataset,
        gepa=gepa,
        llm=llm,
        baseline_prompts=baseline_prompts,
        llm_provider=llm_provider,
        ollama=ollama,
    )

    # Apply environment variable overrides (convention: GEPA_<SECTION>_<KEY>)
    # For now, skip this for simplicity in POC
    # Full implementation would walk the dataclass and apply os.environ matches

    return config
