"""Unit tests for src/analyzer.py using synthetic RunResult data."""

import pytest
from src.runner import RunResult
from src.analyzer import build_decision_matrix, recommend


def _make_results(model_id: str, latency_ms: float, cost_usd: float, n: int = 100) -> list[RunResult]:
    return [
        RunResult(
            task_id=f"task_{i:03d}",
            model_id=model_id,
            prompt="dummy",
            response_text="Positive",
            latency_ms=latency_ms + (i % 5) * 10,  # slight variance
            prompt_tokens=60,
            completion_tokens=5,
            total_tokens=65,
            cost_usd=cost_usd,
        )
        for i in range(1, n + 1)
    ]


DENSE_MODEL = "google/gemma-4-31b-it"
MOE_MODEL = "google/gemma-4-26b-a4b-it"

# Dense: 1300ms / $0.0000065; MoE: ~30% lower latency, ~25% lower cost
DENSE_LATENCY = 1300.0
MOE_LATENCY = DENSE_LATENCY * 0.70   # 30% faster
DENSE_COST = 6.5e-6
MOE_COST = DENSE_COST * 0.75         # 25% cheaper


@pytest.fixture
def dense_results():
    return _make_results(DENSE_MODEL, DENSE_LATENCY, DENSE_COST)


@pytest.fixture
def moe_results():
    return _make_results(MOE_MODEL, MOE_LATENCY, MOE_COST)


def test_matrix_moe_latency_lower(dense_results, moe_results):
    dense_stats, moe_stats, matrix = build_decision_matrix(dense_results, moe_results)
    assert moe_stats.avg_latency_ms < dense_stats.avg_latency_ms


def test_matrix_moe_cost_lower(dense_results, moe_results):
    dense_stats, moe_stats, _ = build_decision_matrix(dense_results, moe_results)
    assert moe_stats.avg_cost_per_query < dense_stats.avg_cost_per_query


def test_matrix_latency_pct_around_30(dense_results, moe_results):
    dense_stats, moe_stats, _ = build_decision_matrix(dense_results, moe_results)
    pct = (moe_stats.avg_latency_ms - dense_stats.avg_latency_ms) / dense_stats.avg_latency_ms * 100
    assert -35 <= pct <= -25, f"Expected ~-30% latency change, got {pct:.1f}%"


def test_matrix_cost_pct_around_25(dense_results, moe_results):
    dense_stats, moe_stats, _ = build_decision_matrix(dense_results, moe_results)
    pct = (moe_stats.avg_cost_per_query - dense_stats.avg_cost_per_query) / dense_stats.avg_cost_per_query * 100
    assert -30 <= pct <= -20, f"Expected ~-25% cost change, got {pct:.1f}%"


def test_matrix_has_expected_columns(dense_results, moe_results):
    _, _, matrix = build_decision_matrix(dense_results, moe_results)
    assert list(matrix.columns) == ["Metric", "Dense", "MoE", "MoE vs Dense"]


def test_recommend_use_moe_when_meets_sla(dense_results, moe_results):
    dense_stats, moe_stats, _ = build_decision_matrix(dense_results, moe_results)
    rec = recommend(dense_stats, moe_stats, sla_latency_ms=2000, sla_cost_per_1k=1.0)
    assert rec.startswith("USE MoE")


def test_recommend_stick_dense_when_latency_too_high(dense_results, moe_results):
    dense_stats, moe_stats, _ = build_decision_matrix(dense_results, moe_results)
    # Set SLA tighter than MoE can achieve
    rec = recommend(dense_stats, moe_stats, sla_latency_ms=100, sla_cost_per_1k=0.000001)
    assert "STICK WITH DENSE" in rec or "MARGINAL" in rec


def test_error_rate_zero_for_clean_results(dense_results, moe_results):
    dense_stats, moe_stats, _ = build_decision_matrix(dense_results, moe_results)
    assert dense_stats.error_rate == 0.0
    assert moe_stats.error_rate == 0.0


def test_partial_errors_reflected_in_error_rate():
    results = _make_results(DENSE_MODEL, 1000, 5e-6, n=10)
    results[0].error = "timeout"
    results[1].error = "timeout"
    dense_stats, _, _ = build_decision_matrix(results, _make_results(MOE_MODEL, 700, 3e-6, n=10))
    assert dense_stats.error_rate == pytest.approx(0.2)
