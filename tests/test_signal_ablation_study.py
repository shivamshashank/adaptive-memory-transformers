# Tests for ablation statistics and failure classification.
from __future__ import annotations

from src.experiments.signal_ablation_study import _bootstrap_mean_interval, _summarize_records


def test_bootstrap_interval_contains_mean_for_constant_values() -> None:
    # Constant samples should yield a degenerate confidence interval.
    interval = _bootstrap_mean_interval([0.5, 0.5, 0.5], samples=100, seed=7)
    assert interval == (0.5, 0.5)


def test_summarize_records_reports_group_statistics_and_failures() -> None:
    # Verify summaries identify when adaptive loses to a fixed policy.
    records = [
        {
            "policy": "adaptive",
            "signal_variant": "all",
            "category": "early",
            "budget_ratio": 0.5,
            "quality_score": 0.8,
        },
        {
            "policy": "uniform",
            "signal_variant": "all",
            "category": "early",
            "budget_ratio": 0.5,
            "quality_score": 0.9,
        },
    ]
    summary = _summarize_records(records)
    assert summary[0]["mean_quality"] == 0.8
    assert summary[0]["count"] == 1
    assert summary[0]["adaptive_beats_best_fixed"] is False
