# Tests for adaptive token scoring and budgeted selection.
from __future__ import annotations

from src.adaptive_policy import AdaptiveImportancePolicy, score_token_importance


def test_score_token_importance_combines_signals() -> None:
    # Verify all configured signals contribute to the score.
    attention = [0.8, 0.1, 0.1]
    recency = [0.2, 0.2, 0.6]
    frequency = [0.9, 0.1, 0.0]

    scores = score_token_importance(attention, recency, frequency)

    assert len(scores) == 3
    assert scores[0] > scores[1]
    assert scores[2] > scores[1]
    assert max(scores) > 0.0


def test_adaptive_policy_selects_top_budget_tokens() -> None:
    # Verify adaptive selection reserves a sink and local window.
    policy = AdaptiveImportancePolicy()
    attention = [0.1, 0.2, 0.7, 0.0]
    recency = [0.1, 0.1, 0.1, 0.7]
    frequency = [0.1, 0.1, 0.8, 0.0]

    kept = policy.select_indices(
        total_tokens=4,
        budget_tokens=2,
        attention_scores=attention,
        recency_scores=recency,
        frequency_scores=frequency,
    )

    assert kept == [0, 1]


def test_adaptive_policy_reserves_sinks_and_attention_heavy_hitters() -> None:
    policy = AdaptiveImportancePolicy()
    attention = [0.01] * 12
    attention[6] = 10.0
    attention[7] = 9.0

    kept = policy.select_indices(
        total_tokens=12,
        budget_tokens=6,
        attention_scores=attention,
        sink_tokens=2,
        local_window_ratio=0.0,
    )

    assert kept[:2] == [0, 1]
    assert {6, 7}.issubset(set(kept))
    assert len(kept) == 6


def test_adaptive_policy_keeps_full_cache_when_budget_exceeds_total() -> None:
    # Verify an oversized budget retains the complete sequence.
    policy = AdaptiveImportancePolicy()
    kept = policy.select_indices(
        total_tokens=4,
        budget_tokens=10,
        attention_scores=[0.2, 0.3, 0.4, 0.1],
        recency_scores=[0.2, 0.2, 0.3, 0.3],
        frequency_scores=[0.2, 0.2, 0.3, 0.3],
    )
    assert kept == [0, 1, 2, 3]
