from __future__ import annotations

import torch
from transformers import DynamicCache

from src.experiments.attention_cache_evaluation import _prune_past_key_values
from src.experiments.compressed_decoding_evaluation import _cache_bytes, _normalize_answer


def test_normalize_answer_ignores_case_and_whitespace() -> None:
    assert _normalize_answer("  Amber-17\n") == "amber-17"


def test_dynamic_cache_pruning_preserves_type_and_selected_size() -> None:
    keys = torch.zeros((1, 2, 4, 3))
    values = torch.ones((1, 2, 4, 3))
    cache = DynamicCache(ddp_cache_data=[(keys, values)])
    pruned = _prune_past_key_values(cache, [0, 3])
    assert isinstance(pruned, DynamicCache)
    assert pruned.get_seq_length() == 2
    assert _cache_bytes(pruned) == 2 * 1 * 2 * 2 * 3 * 4
