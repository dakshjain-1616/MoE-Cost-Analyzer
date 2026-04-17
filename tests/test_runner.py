"""Unit tests for src/runner.py — empty benchmark and invalid model ID handling."""

import pytest

from src.pricing import compute_cost, KNOWN_MODEL_IDS

DENSE_MODEL = "google/gemma-4-31b-it"
MOE_MODEL = "google/gemma-4-26b-a4b-it"


def test_compute_cost_dense():
    cost = compute_cost(DENSE_MODEL, 1_000_000, 0)
    assert cost == pytest.approx(0.10)


def test_compute_cost_moe():
    cost = compute_cost(MOE_MODEL, 1_000_000, 0)
    assert cost == pytest.approx(0.08)


def test_compute_cost_moe_cheaper_than_dense():
    c_dense = compute_cost(DENSE_MODEL, 100, 20)
    c_moe = compute_cost(MOE_MODEL, 100, 20)
    assert c_moe < c_dense


def test_invalid_model_raises():
    with pytest.raises(ValueError, match="Invalid model ID"):
        compute_cost("google/unknown-model", 100, 20)


def test_known_model_ids_contains_both():
    assert DENSE_MODEL in KNOWN_MODEL_IDS
    assert MOE_MODEL in KNOWN_MODEL_IDS
