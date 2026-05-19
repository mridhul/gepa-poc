"""Ollama client for local LLM inference."""

import time
import sqlite3
import requests
from dataclasses import dataclass
from typing import Optional
from .cost_tracker import get_budget_status
from .. import database


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    model: str
    cost_usd: float  # Always 0 for local Ollama


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        max_retries: int,
        timeout_seconds: int,
        db_conn: sqlite3.Connection,
    ):
        """Initialize Ollama client."""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.db_conn = db_conn

        # Verify Ollama is running
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✅ Connected to Ollama at {self.base_url}")
                models = response.json().get("models", [])
                if models:
                    print(f"   Available models: {[m['name'] for m in models]}")
                else:
                    print(f"   ⚠️  No models installed. Run: ollama pull {model}")
            else:
                print(f"⚠️  Ollama returned status {response.status_code}")
        except requests.ConnectionError:
            print(f"❌ Cannot connect to Ollama at {self.base_url}")
            print(f"   Start Ollama with: ollama serve")
            raise RuntimeError(f"Ollama not running at {self.base_url}")
        except Exception as e:
            print(f"⚠️  Error connecting to Ollama: {e}")

    def call(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 512,
        phase: str = "rollout",
        system: Optional[str] = None,
    ) -> LLMResponse:
        """Call Ollama with retry logic."""

        if model is None:
            model = self.model

        # For Ollama, cost is always 0 (local inference)
        cost_usd = 0.0

        # Retry logic
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                # Build full prompt with system message if provided
                full_prompt = prompt
                if system:
                    full_prompt = f"{system}\n\n{prompt}"

                # Call Ollama API
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": 0.7,
                        },
                    },
                    timeout=self.timeout_seconds,
                )

                if response.status_code != 200:
                    raise RuntimeError(f"Ollama returned {response.status_code}: {response.text}")

                result = response.json()
                latency_ms = int((time.time() - start_time) * 1000)

                # Ollama returns token counts
                prompt_tokens = result.get("prompt_eval_count", 0)
                completion_tokens = result.get("eval_count", 0)
                content = result.get("response", "")

                # Log the call (cost is 0 for Ollama)
                database.log_llm_call(
                    self.db_conn,
                    phase=phase,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    cost_usd=0.0,
                )

                return LLMResponse(
                    content=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    model=model,
                    cost_usd=0.0,
                )

            except requests.ConnectionError as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Ollama connection failed after {self.max_retries} retries: {e}")
                wait_time = 2 ** attempt
                print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Ollama call failed after {self.max_retries} retries: {e}")
                wait_time = 2 ** attempt
                print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
