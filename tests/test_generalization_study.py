# Tests for generalization summaries and paired effect sizes.
from __future__ import annotations

from src.experiments.generalization_study import _paired_effect_size, _summarize_generalization


def test_paired_effect_size_is_zero_for_equal_scores() -> None:
    # Equal paired scores should have no effect.
    assert _paired_effect_size([0.5, 0.7], [0.5, 0.7]) == 0.0


def test_paired_effect_size_is_positive_when_adaptive_wins() -> None:
    # Higher adaptive scores should produce a positive effect.
    assert _paired_effect_size([0.8, 0.9], [0.5, 0.6]) > 0.0


def test_generalization_summary_aggregates_model_and_task() -> None:
    # Verify summaries group records by model, task, and policy.
    records = [
        {
            "model": "m1",
            "task": "retrieval",
            "budget_ratio": 0.5,
            "policy": "adaptive",
            "quality_score": 0.8,
        },
        {
            "model": "m1",
            "task": "retrieval",
            "budget_ratio": 0.5,
            "policy": "uniform",
            "quality_score": 0.6,
        },
    ]
    summary = _summarize_generalization(records)
    assert summary[0]["count"] == 1
    assert summary[0]["mean_quality"] == 0.8
    assert summary[0]["adaptive_vs_uniform_effect"] == 0.2
