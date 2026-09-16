# Tests for exact layer budget allocation.
from __future__ import annotations

from src.experiments.layer_adaptive_allocation import (
    _aggregate_layer_scores,
    _layer_budget_allocation,
)


def test_aggregate_layer_scores_ranks_highest_layers_first() -> None:
    # Verify layer scores are ordered from highest to lowest.
    scores = [0.20, 0.70, 0.10]
    assert _aggregate_layer_scores(scores) == [0.70, 0.20, 0.10]


def test_layer_budget_allocation_respects_total_budget() -> None:
    # Verify allocation preserves the exact total budget.
    allocation = _layer_budget_allocation([0.1, 0.6, 0.3], total_budget=10)
    assert sum(allocation) == 10
    assert allocation[1] >= allocation[0]
    assert allocation[1] >= allocation[2]
