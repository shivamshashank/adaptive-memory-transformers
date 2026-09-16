# Tests for retaining selected key/value cache positions.
from __future__ import annotations

import torch

from src.experiments.attention_cache_evaluation import _prune_past_key_values


def _make_cache() -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    # Build a small cache fixture with two layers.
    layer0_keys = torch.arange(1 * 2 * 6 * 3).reshape(1, 2, 6, 3).float()
    layer0_values = torch.arange(1 * 2 * 6 * 3).reshape(1, 2, 6, 3).float()
    layer1_keys = torch.arange(1 * 2 * 6 * 3, 2 * 1 * 2 * 6 * 3).reshape(1, 2, 6, 3).float()
    layer1_values = torch.arange(2 * 1 * 2 * 6 * 3, 3 * 1 * 2 * 6 * 3).reshape(1, 2, 6, 3).float()
    return ((layer0_keys, layer0_values), (layer1_keys, layer1_values))


def test_prune_past_key_values_keeps_selected_positions() -> None:
    # Verify pruning preserves selected positions and shapes.
    cache = _make_cache()
    pruned = _prune_past_key_values(cache, [0, 2, 5])

    assert len(pruned) == 2
    assert pruned[0][0].shape == (1, 2, 3, 3)
    assert pruned[0][1].shape == (1, 2, 3, 3)
    assert pruned[1][0].shape == (1, 2, 3, 3)
    assert pruned[1][1].shape == (1, 2, 3, 3)
