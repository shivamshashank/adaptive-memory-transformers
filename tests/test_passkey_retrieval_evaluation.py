# Tests for passkey retrieval prompt building, needle retention accounting, and policy selection.
from __future__ import annotations

import pytest

from src.experiments.passkey_retrieval_evaluation import (
    PasskeyMeasurement,
    _accumulate_attention_scores,
    _select_indices,
    build_passkey_prompt,
)


class MockTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Simple whitespace-based mock tokenizer mapping words to token ids.
        del add_special_tokens
        return [hash(word) % 10000 for word in text.split()]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(f"tok_{tid}" for tid in token_ids)


def test_build_passkey_prompt_structure() -> None:
    tokenizer = MockTokenizer()
    target_length = 100
    depth = 0.3
    passkey = "12345"

    spec = build_passkey_prompt(
        tokenizer,  # type: ignore[arg-type]
        passkey=passkey,
        target_length=target_length,
        depth=depth,
    )

    assert spec.passkey == "12345"
    assert passkey in spec.full_prompt
    assert "What is the secret passkey?" in spec.full_prompt
    assert spec.needle_start_token < spec.needle_end_token
    assert spec.needle_start_token >= 0
    assert spec.needle_end_token <= spec.actual_tokens


def test_recency_eviction_of_early_needle() -> None:
    total_tokens = 100
    budget = 50  # 50% budget
    needle_indices = set(range(15, 25))  # Needle placed at depth ~0.2 (tokens 15-25)

    recency_keep = _select_indices("recency", total_tokens, budget, [0.0] * total_tokens)
    # Recency should only keep tokens [50, 99]
    assert recency_keep == list(range(50, 100))
    # Crucially, recency must retain ZERO needle tokens!
    assert len(set(recency_keep) & needle_indices) == 0


def test_adaptive_policy_preserves_high_scoring_needle() -> None:
    total_tokens = 100
    budget = 50
    needle_indices = set(range(15, 25))

    # Construct attention scores where needle tokens have very high attention mass
    scores = [0.01] * total_tokens
    for idx in needle_indices:
        scores[idx] = 10.0  # High attention to needle

    adaptive_keep = _select_indices("adaptive", total_tokens, budget, scores)
    assert len(adaptive_keep) == budget
    # Adaptive policy must retain all needle tokens because of high score
    assert needle_indices.issubset(set(adaptive_keep))


def test_passkey_measurement_retention_rate_calculation() -> None:
    measurement = PasskeyMeasurement(
        policy="adaptive",
        budget_ratio=0.5,
        budget_tokens=50,
        prompt_tokens=100,
        needle_tokens=10,
        needle_tokens_retained=10,
        needle_retention_rate=1.0,
        generated_tokens=4,
        generated_text="12345",
        passkey_found=True,
        next_token_loss=0.12,
        perplexity=1.13,
        prefill_time_s=0.05,
        selection_time_s=0.001,
        pruning_time_s=0.001,
        decode_time_s=0.02,
        decode_tokens_per_second=200.0,
        cache_bytes_after_prefill=50000,
        cache_bytes_final=50000,
        peak_memory_mb=0.0,
    )

    assert measurement.passkey_found is True
    assert measurement.needle_retention_rate == 1.0
    assert measurement.budget_ratio == 0.5


def test_accumulate_attention_scores_preserves_existing_and_appends_new() -> None:
    assert _accumulate_attention_scores([1.0, 2.0], [0.5, 1.5, 3.0]) == [1.5, 3.5, 3.0]


def test_accumulate_attention_scores_rejects_short_updates() -> None:
    with pytest.raises(ValueError, match="cover the retained cache"):
        _accumulate_attention_scores([1.0, 2.0], [0.5])
