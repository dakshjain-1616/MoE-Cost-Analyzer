"""Compute statistics and build decision matrix from benchmark results."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .runner import RunResult


@dataclass
class ModelStats:
    model_id: str
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_cost_per_query: float
    total_cost: float
    total_tokens: int
    error_rate: float


def _compute_stats(results: list[RunResult]) -> ModelStats:
    successful = [r for r in results if not r.error]
    latencies = [r.latency_ms for r in successful]
    costs = [r.cost_usd for r in successful]

    return ModelStats(
        model_id=results[0].model_id if results else "unknown",
        avg_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        p50_latency_ms=float(np.percentile(latencies, 50)) if latencies else 0.0,
        p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
        avg_cost_per_query=float(np.mean(costs)) if costs else 0.0,
        total_cost=float(np.sum(costs)),
        total_tokens=sum(r.total_tokens for r in successful),
        error_rate=1.0 - len(successful) / len(results) if results else 0.0,
    )


def _pct_change(moe_val: float, dense_val: float) -> str:
    if dense_val == 0:
        return "N/A"
    pct = (moe_val - dense_val) / dense_val * 100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


def build_decision_matrix(
    dense_results: list[RunResult],
    moe_results: list[RunResult],
) -> tuple[ModelStats, ModelStats, pd.DataFrame]:
    """Build a decision matrix comparing both models."""
    dense_stats = _compute_stats(dense_results)
    moe_stats = _compute_stats(moe_results)

    rows = [
        ("Avg Latency (ms)", f"{dense_stats.avg_latency_ms:.1f}", f"{moe_stats.avg_latency_ms:.1f}",
         _pct_change(moe_stats.avg_latency_ms, dense_stats.avg_latency_ms)),
        ("P50 Latency (ms)", f"{dense_stats.p50_latency_ms:.1f}", f"{moe_stats.p50_latency_ms:.1f}",
         _pct_change(moe_stats.p50_latency_ms, dense_stats.p50_latency_ms)),
        ("P95 Latency (ms)", f"{dense_stats.p95_latency_ms:.1f}", f"{moe_stats.p95_latency_ms:.1f}",
         _pct_change(moe_stats.p95_latency_ms, dense_stats.p95_latency_ms)),
        ("Avg Cost / Query (USD)", f"${dense_stats.avg_cost_per_query:.6f}", f"${moe_stats.avg_cost_per_query:.6f}",
         _pct_change(moe_stats.avg_cost_per_query, dense_stats.avg_cost_per_query)),
        ("Total Cost (USD)", f"${dense_stats.total_cost:.4f}", f"${moe_stats.total_cost:.4f}",
         _pct_change(moe_stats.total_cost, dense_stats.total_cost)),
        ("Total Tokens", str(dense_stats.total_tokens), str(moe_stats.total_tokens),
         _pct_change(moe_stats.total_tokens, dense_stats.total_tokens)),
        ("Error Rate", f"{dense_stats.error_rate:.1%}", f"{moe_stats.error_rate:.1%}", "N/A"),
    ]

    matrix = pd.DataFrame(rows, columns=["Metric", "Dense", "MoE", "MoE vs Dense"])
    return dense_stats, moe_stats, matrix


def recommend(
    dense_stats: ModelStats,
    moe_stats: ModelStats,
    sla_latency_ms: float,
    sla_cost_per_1k: float,
) -> str:
    """Return a recommendation string based on SLA constraints."""
    moe_meets_latency = moe_stats.avg_latency_ms <= sla_latency_ms
    cost_per_1k = moe_stats.avg_cost_per_query * 1000
    moe_meets_cost = cost_per_1k <= sla_cost_per_1k

    if moe_meets_latency and moe_meets_cost:
        return (
            f"USE MoE — latency {moe_stats.avg_latency_ms:.0f}ms < {sla_latency_ms:.0f}ms SLA "
            f"and cost ${cost_per_1k:.4f}/1K < ${sla_cost_per_1k:.4f}/1K SLA"
        )
    elif moe_meets_latency:
        return (
            f"MARGINAL — MoE meets latency SLA ({moe_stats.avg_latency_ms:.0f}ms) "
            f"but cost ${cost_per_1k:.4f}/1K exceeds ${sla_cost_per_1k:.4f}/1K SLA"
        )
    elif moe_meets_cost:
        return (
            f"MARGINAL — MoE meets cost SLA (${cost_per_1k:.4f}/1K) "
            f"but latency {moe_stats.avg_latency_ms:.0f}ms exceeds {sla_latency_ms:.0f}ms SLA"
        )
    else:
        return (
            f"STICK WITH DENSE — MoE fails both: "
            f"latency {moe_stats.avg_latency_ms:.0f}ms > {sla_latency_ms:.0f}ms, "
            f"cost ${cost_per_1k:.4f}/1K > ${sla_cost_per_1k:.4f}/1K"
        )


def save_csv(
    dense_results: list[RunResult],
    moe_results: list[RunResult],
    output_path: str | Path,
) -> None:
    """Save per-query results to CSV."""
    rows = []
    for r in dense_results + moe_results:
        rows.append({
            "task_id": r.task_id,
            "model_id": r.model_id,
            "latency_ms": r.latency_ms,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "total_tokens": r.total_tokens,
            "cost_usd": r.cost_usd,
            "error": r.error or "",
        })
    pd.DataFrame(rows).sort_values(["task_id", "model_id"]).to_csv(output_path, index=False)
