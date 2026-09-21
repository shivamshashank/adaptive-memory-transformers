# Tests for fixed cache policy selection and budget accounting.
from __future__ import annotations

import pytest

from amt.policies import (
    AttentionRecencyPolicy,
    FullCachePolicy,
    RecencyCachePolicy,
    RetentionSignals,
    UniformCachePolicy,
    budget_tokens_for_ratio,
    create_policy,
    rank_normalize,
)


def test_full_policy_keeps_every_token() -> None:
    # Verify the full policy ignores compression budgets.
    policy = FullCachePolicy()
    assert policy.select_indices(12, 8) == list(range(12))


def test_recency_policy_keeps_latest_tokens() -> None:
    # Verify recency keeps the final sequence positions.
    policy = RecencyCachePolicy()
    assert policy.select_indices(10, 3) == [7, 8, 9]


def test_uniform_policy_spreads_across_sequence() -> None:
    # Verify uniform selection spans the sequence.
    policy = UniformCachePolicy()
    result = policy.select_indices(10, 4)
    assert result == [0, 3, 6, 9]


def test_budget_ratio_produces_a_bounded_token_count() -> None:
    assert budget_tokens_for_ratio(20, 0.25) == 5
    assert budget_tokens_for_ratio(20, 0.0) == 0
    assert budget_tokens_for_ratio(20, 2.0) == 20


def test_policy_factory_rejects_unknown_names() -> None:
    assert create_policy("recency").name == "recency"
    with pytest.raises(ValueError, match="Unsupported cache policy"):
        create_policy("unknown")


def test_rank_normalization_handles_scale_ties_and_constants() -> None:
    assert rank_normalize((1.0, 1.0, 9.0, 0.0)) == (0.5, 0.5, 1.0, 0.0)
    assert rank_normalize((10.0, 10.0, 90.0, 0.0)) == (0.5, 0.5, 1.0, 0.0)
    assert rank_normalize((0.0, 0.0)) == (0.5, 0.5)
    assert rank_normalize(()) == ()
    assert rank_normalize((4.0,)) == (0.5,)


def test_each_signal_changes_the_actual_selected_set() -> None:
    signals = RetentionSignals((0, 3, 9, 20), ((9.0, 8.0, 1.0, 0.0),))
    attention = AttentionRecencyPolicy(attention_weight=0.9, recency_weight=0.1)
    recency = AttentionRecencyPolicy(attention_weight=0.1, recency_weight=0.9)
    assert attention.select_indices(4, 2, signals=signals) == [0, 1]
    assert recency.select_indices(4, 2, signals=signals) == [2, 3]
    changed = RetentionSignals(signals.positions, ((0.0, 1.0, 8.0, 9.0),))
    assert attention.select_indices(4, 2, signals=changed) == [2, 3]


def test_layer_ranks_are_combined_instead_of_raw_magnitudes() -> None:
    signals = RetentionSignals((0, 1, 2), ((1000.0, 0.0, 1.0), (0.0, 2.0, 1.0)))
    policy = create_policy("attention")
    assert policy.select_indices(3, 1, signals=signals) == [2]


def test_zero_attention_has_deterministic_newer_first_ties() -> None:
    signals = RetentionSignals((0, 10, 30), ((0.0, 0.0, 0.0),))
    assert create_policy("attention").select_indices(3, 2, signals=signals) == [1, 2]


def test_protection_uses_original_positions_and_never_exceeds_budget() -> None:
    signals = RetentionSignals((2, 10, 30, 31), ((9.0, 8.0, 1.0, 0.0),))
    policy = AttentionRecencyPolicy(1.0, 0.0, sink_tokens=1, local_window=1)
    assert policy.select_indices(4, 2, signals=signals) == [0, 3]
    # Original position zero was deleted; physical slot zero must NOT become a sink.
    assert policy.select_indices(4, 1, signals=signals) == [3]
    huge_protection = AttentionRecencyPolicy(sink_tokens=11, local_window=100)
    for budget in range(1, 5):
        selected = huge_protection.select_indices(4, budget, signals=signals)
        assert len(selected) == budget
        assert selected == sorted(set(selected))
    assert huge_protection.select_indices(4, 1, signals=signals) == [0]
    # A gap in original positions must not be mistaken for a two-token local window.
    gap = RetentionSignals((0, 3, 20), ((9.0, 0.0, 1.0),))
    assert AttentionRecencyPolicy(1.0, 0.0, local_window=2).select_indices(3, 2, signals=gap) == [
        0,
        2,
    ]


def test_missing_attention_is_only_allowed_when_its_weight_is_zero() -> None:
    positions = RetentionSignals((0, 9, 20))
    with pytest.raises(ValueError, match="Attention scores are required"):
        create_policy("attention").select_indices(3, 1, signals=positions)
    assert AttentionRecencyPolicy(0.0, 1.0).select_indices(3, 1, signals=positions) == [2]
    with pytest.raises(ValueError, match="position"):
        create_policy("adaptive").select_indices(3, 1)


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_invalid_weights_are_rejected(weight: float) -> None:
    with pytest.raises(ValueError, match="Weights"):
        AttentionRecencyPolicy(attention_weight=weight)


@pytest.mark.parametrize(
    "signals",
    [
        RetentionSignals((0, 1), ((1.0,),)),
        RetentionSignals((1, 0), ((0.5, 0.5),)),
        RetentionSignals((0, 0), ((0.5, 0.5),)),
        RetentionSignals((0, 1), ((float("nan"), 0.5),)),
        RetentionSignals((0, 1), ((-0.1, 0.5),)),
    ],
)
def test_misaligned_or_invalid_signals_are_rejected(signals: RetentionSignals) -> None:
    with pytest.raises(ValueError):
        create_policy("adaptive").select_indices(2, 1, signals=signals)


def test_adaptive_budget_edges_and_configuration_errors() -> None:
    policy = create_policy("adaptive")
    assert policy.select_indices(0, 2) == []
    assert policy.select_indices(2, 0) == []
    assert policy.select_indices(1, 3, signals=RetentionSignals((7,), ((1.0,),))) == [0]
    with pytest.raises(ValueError, match="positive"):
        AttentionRecencyPolicy(0.0, 0.0)
    with pytest.raises(ValueError, match="counts"):
        AttentionRecencyPolicy(sink_tokens=-1)
    with pytest.raises(ValueError, match="local_window"):
        policy.select_indices(3, 1, recent_window=2)


def test_recency_window_cannot_override_capacity() -> None:
    assert RecencyCachePolicy().select_indices(10, 2, recent_window=9) == [8, 9]
