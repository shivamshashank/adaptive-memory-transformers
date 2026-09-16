"""Fixed full, recency, and uniform cache-retention policies."""

from __future__ import annotations

from typing import Protocol


class CachePolicy(Protocol):
    name: str

    def select_indices(
        self,
        total_tokens: int,
        budget_tokens: int,
        *,
        recent_window: int | None = None,
    ) -> list[int]: ...


class FullCachePolicy:
    name = "full"

    def select_indices(
        self,
        total_tokens: int,
        budget_tokens: int,
        *,
        recent_window: int | None = None,
    ) -> list[int]:
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
        del recent_window
        if total_tokens <= 0:
            return []
        if budget_tokens <= 0:
            return []
        if budget_tokens >= total_tokens:
            return list(range(total_tokens))

        if budget_tokens == 1:
            return [0]
        last = total_tokens - 1
        denominator = budget_tokens - 1
        return [(index * last) // denominator for index in range(budget_tokens)]


def create_policy(name: str) -> CachePolicy:
    policies: dict[str, CachePolicy] = {
        "full": FullCachePolicy(),
        "recency": RecencyCachePolicy(),
        "uniform": UniformCachePolicy(),
    }
    try:
        return policies[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported cache policy: {name}") from exc


def budget_tokens_for_ratio(total_tokens: int, ratio: float) -> int:
    if total_tokens <= 0:
        return 0
    if ratio <= 0:
        return 0
    if ratio >= 1:
        return total_tokens
    return max(1, min(total_tokens, int(round(total_tokens * ratio))))
