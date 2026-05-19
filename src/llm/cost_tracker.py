"""LLM cost calculation and tracking."""

PRICE_TABLE = {
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {
        "input_per_1k": 0.003,
        "output_per_1k": 0.015,
    },
    "anthropic.claude-opus-4-5": {
        "input_per_1k": 0.015,
        "output_per_1k": 0.075,
    },
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost for an LLM call."""
    if model not in PRICE_TABLE:
        # Fallback to Sonnet pricing for unknown models
        prices = PRICE_TABLE["anthropic.claude-3-5-sonnet-20241022-v2:0"]
    else:
        prices = PRICE_TABLE[model]

    input_cost = (prompt_tokens / 1000.0) * prices["input_per_1k"]
    output_cost = (completion_tokens / 1000.0) * prices["output_per_1k"]
    return input_cost + output_cost


def get_budget_status(
    total_cost: float, hard_cap: float, total_calls: int, max_calls: int
) -> dict:
    """Get budget status."""
    remaining = hard_cap - total_cost
    pct_used = (total_cost / hard_cap * 100.0) if hard_cap > 0 else 0.0
    calls_remaining = max_calls - total_calls

    return {
        "total_cost_usd": round(total_cost, 2),
        "remaining_usd": round(remaining, 2),
        "hard_cap_usd": hard_cap,
        "pct_used": round(pct_used, 1),
        "total_calls": total_calls,
        "calls_remaining": calls_remaining,
        "max_calls": max_calls,
    }
