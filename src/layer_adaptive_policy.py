# Proportional allocation of a shared token budget across layers.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LayerBudget:
    layer_index: int
    score: float
    allocated_tokens: int


class LayerAdaptivePolicy:
    name = "layer_adaptive"

    def allocate_budgets(
        self,
        layer_scores: Sequence[float],
        total_budget: int,
    ) -> list[int]:
        # Allocate the shared budget in proportion to layer scores.
        if total_budget <= 0:
            return [0 for _ in layer_scores]
        if not layer_scores:
            return []

        scores = [max(0.0, float(score)) for score in layer_scores]
        total_score = sum(scores)
        if total_score <= 0:
            base = total_budget // len(scores)
            remainder = total_budget % len(scores)
            allocations = [base for _ in scores]
            for idx in range(remainder):
                allocations[idx] += 1
            return allocations

        raw = [total_budget * score / total_score for score in scores]
        allocations = [int(math.floor(value)) for value in raw]
        remaining = total_budget - sum(allocations)

        order = sorted(
            range(len(scores)),
            key=lambda idx: (raw[idx] - allocations[idx], scores[idx]),
            reverse=True,
        )
        for idx in order[:remaining]:
            allocations[idx] += 1

        return allocations


def allocate_layer_budgets(
    layer_scores: Sequence[float],
    total_budget: int,
) -> list[int]:
    # Apply the default layer allocation policy.
    return LayerAdaptivePolicy().allocate_budgets(layer_scores, total_budget)
