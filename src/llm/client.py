"""LLM client factory for Bedrock or Ollama."""

import time
import sqlite3
from dataclasses import dataclass
from typing import Optional, Union
import os
from anthropic import AnthropicBedrock
from .cost_tracker import calculate_cost, get_budget_status
from .ollama_client import OllamaClient
from .. import database


class BudgetExhaustedError(Exception):
    """Raised when LLM budget is exhausted."""
    pass


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    model: str
    cost_usd: float


class LLMClient:
    def __init__(self, model_rollout: str, model_reflection: str, model_generation: str,
                 max_retries: int, timeout_seconds: int, budget_hard_cap: float,
                 budget_warn_threshold: float, max_calls_per_run: int,
                 db_conn: sqlite3.Connection):
        """Initialize LLM client with Bedrock."""
        self.model_rollout = model_rollout
        self.model_reflection = model_reflection
        self.model_generation = model_generation
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.budget_hard_cap = budget_hard_cap
        self.budget_warn_threshold = budget_warn_threshold
        self.max_calls_per_run = max_calls_per_run
        self.db_conn = db_conn

        # Initialize Bedrock client
        try:
            self.client = AnthropicBedrock(
                region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Bedrock client: {e}")

    def call(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 512,
        phase: str = "rollout",
        system: Optional[str] = None,
    ) -> LLMResponse:
        """Call LLM with retry, budget checking, and logging."""

        if model is None:
            model = self.model_rollout

        # Budget check before call
        total_cost = database.get_total_cost(self.db_conn)
        total_calls = database.get_total_calls(self.db_conn)

        if total_cost >= self.budget_hard_cap:
            raise BudgetExhaustedError(
                f"Budget exhausted: ${total_cost:.2f} >= ${self.budget_hard_cap:.2f}"
            )

        if total_calls >= self.max_calls_per_run:
            raise BudgetExhaustedError(
                f"Max calls exhausted: {total_calls} >= {self.max_calls_per_run}"
            )

        # Warn if approaching threshold
        if total_cost >= self.budget_warn_threshold:
            print(
                f"⚠️  Budget warning: ${total_cost:.2f} / ${self.budget_hard_cap:.2f} "
                f"({total_cost/self.budget_hard_cap*100:.1f}%)"
            )

        # Retry logic
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                messages = [{"role": "user", "content": prompt}]
                kwargs = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": messages,
                }
                if system:
                    kwargs["system"] = system

                response = self.client.messages.create(**kwargs)

                latency_ms = int((time.time() - start_time) * 1000)
                prompt_tokens = response.usage.input_tokens
                completion_tokens = response.usage.output_tokens
                cost = calculate_cost(model, prompt_tokens, completion_tokens)

                # Log the call
                database.log_llm_call(
                    self.db_conn,
                    phase=phase,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    cost_usd=cost,
                )

                return LLMResponse(
                    content=response.content[0].text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    model=model,
                    cost_usd=cost,
                )

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {e}")
                wait_time = 2 ** attempt
                print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)


def create_llm_client(
    llm_provider: str,
    model_rollout: str,
    model_reflection: str,
    model_generation: str,
    max_retries: int,
    timeout_seconds: int,
    budget_hard_cap: float,
    budget_warn_threshold: float,
    max_calls_per_run: int,
    db_conn: sqlite3.Connection,
    ollama_base_url: Optional[str] = None,
    ollama_model: Optional[str] = None,
) -> Union[LLMClient, OllamaClient]:
    """Factory function to create appropriate LLM client."""

    if llm_provider.lower() == "ollama":
        if not ollama_base_url or not ollama_model:
            raise ValueError("ollama_base_url and ollama_model required for Ollama provider")
        print(f"🔧 Using Ollama (local inference)")
        return OllamaClient(
            base_url=ollama_base_url,
            model=ollama_model,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            db_conn=db_conn,
        )
    elif llm_provider.lower() == "bedrock":
        print(f"🔧 Using AWS Bedrock")
        return LLMClient(
            model_rollout=model_rollout,
            model_reflection=model_reflection,
            model_generation=model_generation,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            budget_hard_cap=budget_hard_cap,
            budget_warn_threshold=budget_warn_threshold,
            max_calls_per_run=max_calls_per_run,
            db_conn=db_conn,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {llm_provider}. Use 'bedrock' or 'ollama'")
