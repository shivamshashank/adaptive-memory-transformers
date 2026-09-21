"""Greedy Qwen decoding with position-preserving, shared-layer cache pruning."""

from dataclasses import dataclass
from typing import cast

import torch
from transformers import Qwen2ForCausalLM

from amt.cache import Qwen2CacheAdapter
from amt.decoding import DecodeTrace, _InspectableCache, snapshot_cache
from amt.policies import CachePolicy, RetentionSignals, policy_requires_attention


@dataclass(frozen=True)
class CompressedDecodeTrace:
    """Decode evidence with post-pruning absolute positions for every step."""

    decode: DecodeTrace
    retained_positions: tuple[tuple[int, ...], ...]


@torch.inference_mode()
def run_compressed_greedy_decode(
    model: Qwen2ForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    policy: CachePolicy,
    budget_tokens: int,
) -> CompressedDecodeTrace:
    """Prune after each forward; subsequent queries see only the retained cache.

    FullCachePolicy deliberately ignores the budget. Budgeted policies retain
    at most budget_tokens AFTER pruning; prefill and append have transient
    allocations. This function is a correctness runner, not a latency benchmark.
    step_logits[0] is predicted by UNPRUNED prefill. A one-token completion must
    not be used to measure compression quality; evaluation.py prunes context
    before appending an unseen query instead.
    """
    if max_new_tokens < 1 or budget_tokens < 1:
        raise ValueError("max_new_tokens and budget_tokens must be at least 1")
    adapter = Qwen2CacheAdapter(model)
    sequences = input_ids.detach().clone()
    current_input = input_ids
    logits_history = []
    snapshots = []
    positions = []
    collect_attention = policy_requires_attention(policy)
    for _ in range(max_new_tokens):
        logits = adapter.forward(current_input, collect_attention=collect_attention)[:, -1, :]
        logits_history.append(logits.detach().clone())
        signals = RetentionSignals(adapter.positions, adapter.attention_by_layer)
        adapter.retain(
            policy.select_indices(len(adapter.positions), budget_tokens, signals=signals)
        )
        snapshots.append(snapshot_cache(cast(_InspectableCache, adapter.cache)))
        positions.append(adapter.positions)
        current_input = logits.argmax(dim=-1, keepdim=True)
        sequences = torch.cat((sequences, current_input), dim=1)
    return CompressedDecodeTrace(
        decode=DecodeTrace(sequences, tuple(logits_history), tuple(snapshots)),
        retained_positions=tuple(positions),
    )
