# Adaptive token selection using attention, recency, and frequency signals.
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AdaptiveResult:
    policy: str
    total_tokens: int
    budget_tokens: int
    keep_indices: list[int]


class AdaptiveImportancePolicy:
    name = "adaptive"

    def select_indices(
        self,
        total_tokens: int,
        budget_tokens: int,
        *,
        attention_scores: Sequence[float] | None = None,
        recency_scores: Sequence[float] | None = None,
        frequency_scores: Sequence[float] | None = None,
        sink_tokens: int = 4,
        local_window_ratio: float = 0.1,
    ) -> list[int]:
        # Select the highest-scoring token positions within a budget.
        if total_tokens <= 0:
            return []
        if budget_tokens <= 0:
            return []
        if budget_tokens >= total_tokens:
            return list(range(total_tokens))

        attention = list(attention_scores) if attention_scores is not None else [0.0] * total_tokens
        recency = list(recency_scores) if recency_scores is not None else [0.0] * total_tokens
        frequency = list(frequency_scores) if frequency_scores is not None else [0.0] * total_tokens

        if len(attention) < total_tokens:
            attention = attention + [0.0] * (total_tokens - len(attention))
        if len(recency) < total_tokens:
            recency = recency + [0.0] * (total_tokens - len(recency))
        if len(frequency) < total_tokens:
            frequency = frequency + [0.0] * (total_tokens - len(frequency))

        sink_count = min(max(sink_tokens, 0), budget_tokens, total_tokens)
        local_count = min(
            max(int(budget_tokens * local_window_ratio), 0),
            budget_tokens - sink_count,
            total_tokens - sink_count,
        )
        kept = set(range(sink_count))
        local_start = max(sink_count, total_tokens - local_count)
        kept.update(range(local_start, total_tokens))
        attention_ranked = sorted(
            (index for index in range(total_tokens) if index not in kept),
            key=lambda index: attention[index],
            reverse=True,
        )
        kept.update(attention_ranked[: budget_tokens - len(kept)])
        return sorted(kept)


def score_token_importance(
    attention_scores: Sequence[float],
    recency_scores: Sequence[float],
    frequency_scores: Sequence[float],
) -> list[float]:
    # Combine normalized importance signals with fixed weights.
    if not attention_scores and not recency_scores and not frequency_scores:
        return []

    length = max(len(attention_scores), len(recency_scores), len(frequency_scores))
    normalized_attention = _normalize_scores(attention_scores, length)
    normalized_recency = _normalize_scores(recency_scores, length)
    normalized_frequency = _normalize_scores(frequency_scores, length)

    return [
        0.5 * normalized_attention[i] + 0.3 * normalized_recency[i] + 0.2 * normalized_frequency[i]
        for i in range(length)
    ]


def _normalize_scores(values: Sequence[float], length: int) -> list[float]:
    # Scale a score vector to the unit interval.
    if length == 0:
        return []
    data = list(values)[:length]
    if len(data) < length:
        data = data + [0.0] * (length - len(data))
    if not data:
        return [0.0] * length
    max_val = max(data)
    min_val = min(data)
    if max_val == min_val:
        return [1.0 if max_val > 0 else 0.0 for _ in data]
    return [(value - min_val) / (max_val - min_val) for value in data]
