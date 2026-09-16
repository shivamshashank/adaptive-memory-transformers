# Fixed full, recency, and uniform cache-retention policies.
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

VALID_BUDGETS = (1.0, 0.75, 0.5, 0.25, 0.1)


class CachePolicy(Protocol):
    name: str

    def select_indices(
        self,
        total_tokens: int,
        budget_tokens: int,
        *,
        recent_window: int | None = None,
    ) -> list[int]: ...


@dataclass(frozen=True)
class CacheSelection:
    policy: str
    total_tokens: int
    budget_tokens: int
    keep_indices: list[int]

    @property
    def retained_tokens(self) -> int:
        # Report the number of retained positions.
        return len(self.keep_indices)


class FullCachePolicy:
    name = "full"

    def select_indices(
        self,
        total_tokens: int,
        budget_tokens: int,
        *,
        recent_window: int | None = None,
    ) -> list[int]:
        # Retain every token regardless of the requested budget.
        del budget_tokens, recent_window
        return list(range(max(total_tokens, 0)))


class RecencyCachePolicy:
    name = "recency"

    def select_indices(
        self,
        total_tokens: int,
        budget_tokens: int,
        *,
        recent_window: int | None = None,
    ) -> list[int]:
        # Retain the most recent token positions.
        if total_tokens <= 0:
            return []
        if budget_tokens <= 0:
            return []
        if budget_tokens >= total_tokens:
            return list(range(total_tokens))

        window = recent_window if recent_window is not None else budget_tokens
        effective_window = min(max(window, 1), total_tokens)
        if effective_window >= total_tokens:
            return list(range(total_tokens))

        start = total_tokens - effective_window
        return list(range(start, total_tokens))


class UniformCachePolicy:
    name = "uniform"

    def select_indices(
        self,
        total_tokens: int,
        budget_tokens: int,
        *,
        recent_window: int | None = None,
    ) -> list[int]:
        # Spread retained positions across the sequence.
        del recent_window
        if total_tokens <= 0:
            return []
        if budget_tokens <= 0:
            return []
        if budget_tokens >= total_tokens:
            return list(range(total_tokens))

        positions = np.linspace(0, total_tokens - 1, num=budget_tokens, dtype=int)
        keep = sorted(int(value) for value in np.unique(positions))
        return keep


def select_cache_indices(
    policy: str,
    total_tokens: int,
    budget_tokens: int,
    *,
    recent_window: int | None = None,
) -> list[int]:
    # Dispatch a policy name to its index selector.
    if policy == "full":
        return FullCachePolicy().select_indices(
            total_tokens, budget_tokens, recent_window=recent_window
        )
    if policy == "recency":
        return RecencyCachePolicy().select_indices(
            total_tokens, budget_tokens, recent_window=recent_window
        )
    if policy == "uniform":
        return UniformCachePolicy().select_indices(
            total_tokens, budget_tokens, recent_window=recent_window
        )
    raise ValueError(f"Unsupported cache policy: {policy}")


def make_selection(
    policy: str,
    total_tokens: int,
    budget_tokens: int,
    *,
    recent_window: int | None = None,
) -> CacheSelection:
    # Build selection metadata for one policy and budget.
    keep_indices = select_cache_indices(
        policy,
        total_tokens,
        budget_tokens,
        recent_window=recent_window,
    )
    return CacheSelection(
        policy=policy,
        total_tokens=total_tokens,
        budget_tokens=budget_tokens,
        keep_indices=keep_indices,
    )


def budget_tokens_for_ratio(total_tokens: int, ratio: float) -> int:
    # Convert a fractional budget into an integer token count.
    if total_tokens <= 0:
        return 0
    if ratio <= 0:
        return 0
    if ratio >= 1:
        return total_tokens
    return max(1, min(total_tokens, int(round(total_tokens * ratio))))


def compare_policies_at_budget(
    total_tokens: int,
    ratio: float,
    *,
    recent_window: int | None = None,
) -> dict[str, CacheSelection]:
    # Return fixed-policy selections at one matched budget.
    budget = budget_tokens_for_ratio(total_tokens, ratio)
    return {
        policy: make_selection(
            policy,
            total_tokens,
            budget,
            recent_window=recent_window,
        )
        for policy in ("full", "recency", "uniform")
    }
