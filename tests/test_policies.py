# Tests for fixed cache policy selection and budget accounting.
from __future__ import annotations

import pytest

from amt.policies import (
    FullCachePolicy,
    RecencyCachePolicy,
    UniformCachePolicy,
    budget_tokens_for_ratio,
    create_policy,
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
