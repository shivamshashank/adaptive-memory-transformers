"""Fixed and attention/recency cache-retention policies with explicit budgets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RetentionSignals:
    """Scores aligned to CURRENT cache slots, with original absolute positions."""

    positions: tuple[int, ...]
    attention_by_layer: tuple[tuple[float, ...], ...] = ()


def rank_normalize(values: tuple[float, ...]) -> tuple[float, ...]:
    """Map to [0, 1] using average ranks for ties; constant signals give 0.5."""
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Signals must be finite")
    if len(values) < 2:
        return (0.5,) * len(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / (2 * (len(order) - 1))
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return tuple(ranks)


class CachePolicy(Protocol):
    @property
    def name(self) -> str: ...

    def select_indices(
        self,
        total_tokens: int,
        budget_tokens: int,
        *,
        recent_window: int | None = None,
        signals: RetentionSignals | None = None,
    ) -> list[int]: ...


class FullCachePolicy:
    name = "full"

    def select_indices(
        self,
        total_tokens: int,
        budget_tokens: int,
        *,
        recent_window: int | None = None,
        signals: RetentionSignals | None = None,
    ) -> list[int]:
        del budget_tokens, recent_window, signals
        return list(range(max(total_tokens, 0)))


class RecencyCachePolicy:
    name = "recency"

    def select_indices(
        self,
        total_tokens: int,
        budget_tokens: int,
        *,
        recent_window: int | None = None,
        signals: RetentionSignals | None = None,
    ) -> list[int]:
        del signals
        if total_tokens <= 0:
            return []
        if budget_tokens <= 0:
            return []
        if budget_tokens >= total_tokens:
            return list(range(total_tokens))

        window = recent_window if recent_window is not None else budget_tokens
        effective_window = min(max(window, 1), total_tokens, budget_tokens)
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
        signals: RetentionSignals | None = None,
    ) -> list[int]:
        del recent_window, signals
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


@dataclass(frozen=True)
class AttentionRecencyPolicy:
    """Rank current-query attention and recency; no history or novelty claim.

    All layers share one selection. Sink tokens are original positions below
    sink_tokens. If protections exceed capacity, earliest sinks win first,
    then newest local tokens. Remaining slots use scores, with newer-first ties.
    """

    attention_weight: float = 0.5
    recency_weight: float = 0.5
    sink_tokens: int = 0
    local_window: int = 0
    name: str = "adaptive"

    def __post_init__(self) -> None:
        weights = (self.attention_weight, self.recency_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights) or not math.isfinite(sum(weights)):
            raise ValueError("Weights must be finite and nonnegative")
        if sum(weights) == 0:
            raise ValueError("At least one weight must be positive")
        for count in (self.sink_tokens, self.local_window):
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("Protection counts must be nonnegative integers")

    def select_indices(
        self,
        total_tokens: int,
        budget_tokens: int,
        *,
        recent_window: int | None = None,
        signals: RetentionSignals | None = None,
    ) -> list[int]:
        if recent_window is not None:
            raise ValueError("Configure local_window on the attention policy")
        if total_tokens <= 0 or budget_tokens <= 0:
            return []
        if signals is None or len(signals.positions) != total_tokens:
            raise ValueError("Signals must provide a position for every cache slot")
        positions = signals.positions
        if any(not isinstance(p, int) or isinstance(p, bool) or p < 0 for p in positions):
            raise ValueError("Positions must be nonnegative integers")
        if any(a >= b for a, b in zip(positions, positions[1:])):
            raise ValueError("Positions must be sorted and unique")
        layers = signals.attention_by_layer
        if self.attention_weight > 0 and not layers:
            raise ValueError("Attention scores are required for a positive attention weight")
        if any(len(layer) != total_tokens for layer in layers):
            raise ValueError("Attention scores must align with every cache slot")
        if any(not math.isfinite(v) or v < 0 for layer in layers for v in layer):
            raise ValueError("Attention scores must be finite and nonnegative")
        if budget_tokens >= total_tokens:
            return list(range(total_tokens))
        ranked_layers = [rank_normalize(layer) for layer in layers]
        recency = rank_normalize(tuple(float(p) for p in positions))
        weight_sum = self.attention_weight + self.recency_weight
        scores = [
            (
                self.attention_weight
                * (sum(layer[i] for layer in ranked_layers) / len(layers) if layers else 0.0)
                + self.recency_weight * recency[i]
            )
            / weight_sum
            for i in range(total_tokens)
        ]
        sinks = [i for i, p in enumerate(positions) if p < self.sink_tokens]
        local = [
            i
            for i in reversed(range(total_tokens))
            if positions[i] > positions[-1] - self.local_window and i not in sinks
        ]
        protected = (sinks + local)[:budget_tokens]
        selected = set(protected)
        ranked = sorted(range(total_tokens), key=lambda i: (scores[i], positions[i]), reverse=True)
        for index in ranked:
            if len(selected) == budget_tokens:
                break
            selected.add(index)
        return sorted(selected)


def create_policy(name: str) -> CachePolicy:
    policies: dict[str, CachePolicy] = {
        "full": FullCachePolicy(),
        "recency": RecencyCachePolicy(),
        "uniform": UniformCachePolicy(),
        "attention": AttentionRecencyPolicy(
            attention_weight=1.0, recency_weight=0.0, name="attention"
        ),
        "adaptive": AttentionRecencyPolicy(),
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
    return max(1, min(total_tokens, round(total_tokens * ratio)))
