"""OpenRouter pricing for supported models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    model_id: str
    name: str
    type: str  # 'dense' or 'moe'
    input_per_1m: float   # USD per 1M input tokens
    output_per_1m: float  # USD per 1M output tokens
    context_window: int

    def compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.input_per_1m + completion_tokens * self.output_per_1m) / 1_000_000


PRICING: dict[str, ModelPricing] = {
    "google/gemma-4-31b-it": ModelPricing(
        model_id="google/gemma-4-31b-it",
        name="Gemma-4-31B-IT",
        type="dense",
        input_per_1m=0.10,
        output_per_1m=0.10,
        context_window=128_000,
    ),
    "google/gemma-4-26b-a4b-it": ModelPricing(
        model_id="google/gemma-4-26b-a4b-it",
        name="Gemma-4-26B-A4B-IT",
        type="moe",
        input_per_1m=0.08,
        output_per_1m=0.08,
        context_window=128_000,
    ),
}

KNOWN_MODEL_IDS = set(PRICING.keys())


def compute_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return cost in USD for the given token counts.

    Raises:
        ValueError: if model_id is not in the pricing table.
    """
    if model_id not in PRICING:
        raise ValueError(f"Invalid model ID '{model_id}'")
    return PRICING[model_id].compute_cost(prompt_tokens, completion_tokens)
