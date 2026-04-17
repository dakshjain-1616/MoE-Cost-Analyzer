"""Runs benchmark prompts against OpenRouter models and records metrics."""

import asyncio
import random
import time
from dataclasses import dataclass

from openai import AsyncOpenAI, APIStatusError, APIConnectionError
from tqdm.asyncio import tqdm

from .pricing import KNOWN_MODEL_IDS, compute_cost

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_REFERER = "https://github.com/neo-ai/moe-cost-analyzer"
MAX_RETRIES = 3


@dataclass
class RunResult:
    task_id: str
    model_id: str
    prompt: str
    response_text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    error: str | None = None


async def _call_model(
    client: AsyncOpenAI,
    model_id: str,
    task_id: str,
    prompt: str,
    dry_run: bool,
) -> RunResult:
    """Call a single model for one task with retry logic."""
    if dry_run:
        # Simulate realistic differences: MoE is faster & cheaper
        is_moe = "a4b" in model_id
        base_latency = random.gauss(900 if is_moe else 1300, 100)
        p_tokens, c_tokens = 60, 20
        return RunResult(
            task_id=task_id,
            model_id=model_id,
            prompt=prompt,
            response_text="Positive",
            latency_ms=max(200, base_latency),
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=p_tokens + c_tokens,
            cost_usd=compute_cost(model_id, p_tokens, c_tokens),
        )

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            response = await client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=64,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            usage = response.usage
            p_tokens = usage.prompt_tokens if usage else 0
            c_tokens = usage.completion_tokens if usage else 0
            text = response.choices[0].message.content or ""

            return RunResult(
                task_id=task_id,
                model_id=model_id,
                prompt=prompt,
                response_text=text,
                latency_ms=latency_ms,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=(p_tokens + c_tokens),
                cost_usd=compute_cost(model_id, p_tokens, c_tokens),
            )

        except APIStatusError as e:
            if e.status_code == 429 or e.status_code >= 500:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
            return RunResult(
                task_id=task_id,
                model_id=model_id,
                prompt=prompt,
                response_text="",
                latency_ms=0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                error=str(e),
            )
        except APIConnectionError as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return RunResult(
                task_id=task_id,
                model_id=model_id,
                prompt=prompt,
                response_text="",
                latency_ms=0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                error=str(e),
            )

    # Exhausted all retries without a successful response
    return RunResult(
        task_id=task_id, model_id=model_id, prompt=prompt,
        response_text="", latency_ms=0, prompt_tokens=0,
        completion_tokens=0, total_tokens=0, cost_usd=0.0,
        error="Max retries exceeded",
    )


async def run_benchmark(
    api_key: str,
    tasks: list[dict],
    dense_model: str,
    moe_model: str,
    dry_run: bool = False,
) -> tuple[list[RunResult], list[RunResult]]:
    """Run all tasks against both models concurrently.

    Returns:
        Tuple of (dense_results, moe_results).

    Raises:
        ValueError: if either model ID is not recognized.
    """
    for mid in (dense_model, moe_model):
        if mid not in KNOWN_MODEL_IDS:
            raise ValueError(f"Invalid model ID '{mid}'")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={"HTTP-Referer": OPENROUTER_REFERER},
    )

    sem = asyncio.Semaphore(5)  # max concurrent requests

    async def bounded(model_id: str, task: dict) -> RunResult:
        async with sem:
            return await _call_model(client, model_id, task["id"], task["prompt"], dry_run)

    coroutines = []
    for task in tasks:
        coroutines.append(bounded(dense_model, task))
        coroutines.append(bounded(moe_model, task))

    try:
        all_results = await tqdm.gather(
            *coroutines,
            desc="Calling models",
            unit="req",
            total=len(coroutines),
        )
    finally:
        await client.close()

    dense_results = [all_results[i] for i in range(0, len(all_results), 2)]
    moe_results = [all_results[i] for i in range(1, len(all_results), 2)]
    return dense_results, moe_results
