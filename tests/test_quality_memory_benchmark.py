# Tests for benchmark quality proxies and frontier summaries.
from __future__ import annotations

from src.experiments.quality_memory_benchmark import (
    _quality_proxy_for_selection,
    _summarize_policy_frontier,
)


def test_quality_proxy_for_selection_matches_attention_mass() -> None:
    # Verify retained attention mass is normalized by total mass.
    attention = [0.1, 0.7, 0.2]
    retained = [1, 2]
    assert _quality_proxy_for_selection(retained, attention) == 0.9


def test_summarize_policy_frontier_keeps_budgets_and_scores() -> None:
    # Verify frontier summaries preserve policy scores and ordering.
    records = [
        {"policy": "uniform", "budget_ratio": 1.0, "retained_tokens": 3, "quality_score": 1.0},
        {"policy": "adaptive", "budget_ratio": 0.5, "retained_tokens": 2, "quality_score": 0.8},
    ]
    summary = _summarize_policy_frontier(records)
    assert summary[0]["policy"] == "uniform"
    assert summary[1]["policy"] == "adaptive"
    assert summary[1]["quality_score"] == 0.8
