# Tests for fixed cache policy selection and budget accounting.
from __future__ import annotations

from src.cache_policies import (
    FullCachePolicy,
    RecencyCachePolicy,
    UniformCachePolicy,
    compare_policies_at_budget,
    make_selection,
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


def test_make_selection_returns_budgeted_result() -> None:
    # Verify selection metadata reports the requested budget.
    selection = make_selection("recency", 20, 5)
    assert selection.policy == "recency"
    assert selection.total_tokens == 20
    assert selection.budget_tokens == 5
    assert selection.keep_indices == [15, 16, 17, 18, 19]
    assert selection.retained_tokens == 5


def test_compare_policies_at_budget_uses_equal_capacity() -> None:
    # Verify fixed policies share one retained-token capacity.
    by_policy = compare_policies_at_budget(20, 0.25)
    assert set(by_policy) == {"full", "recency", "uniform"}
    assert by_policy["full"].retained_tokens == 20
    assert by_policy["recency"].retained_tokens == 5
    assert by_policy["uniform"].retained_tokens == 5
