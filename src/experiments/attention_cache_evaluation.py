# Real-signal cache evaluation and key/value cache pruning.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import DynamicCache

from src.adaptive_policy import AdaptiveImportancePolicy
from src.cache_policies import VALID_BUDGETS, budget_tokens_for_ratio, compare_policies_at_budget
from src.importance import DEFAULT_MODEL, run_importance_analysis


def _prune_past_key_values(
    cache: tuple[tuple[torch.Tensor, torch.Tensor], ...] | DynamicCache,
    keep_indices: list[int],
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...] | DynamicCache:
    if not keep_indices:
        raise ValueError("keep_indices must not be empty")
    # Retain selected sequence positions in every cache layer.
    legacy_layers = (
        cache
        if isinstance(cache, tuple)
        else tuple((layer.keys, layer.values) for layer in cache.layers)
    )
    if not legacy_layers:
        raise ValueError("cache must contain at least one layer")

    pruned: list[tuple[torch.Tensor, torch.Tensor]] = []
    for key_tensor, value_tensor in legacy_layers:
        seq_len = int(key_tensor.shape[-2])
        if any(index < 0 or index >= seq_len for index in keep_indices):
            raise ValueError("keep_indices must be valid cache positions")
        selected = torch.tensor(keep_indices, dtype=torch.long, device=key_tensor.device)
        pruned.append(
            (
                key_tensor[..., selected, :].clone(),
                value_tensor[..., selected, :].clone(),
            )
        )
    if isinstance(cache, DynamicCache):
        return DynamicCache(ddp_cache_data=pruned)
    return tuple(pruned)


def _signal_vectors_from_prompt(
    prompt: str, model_name: str
) -> tuple[list[float], list[float], list[float]]:
    result = run_importance_analysis(model_name=model_name, prompt=prompt)
    # Build attention, recency, and frequency vectors from a prompt.
    attention_scores = [entry["attention_mass"] for entry in result.position_importance]
    recency_scores = [entry["recency_signal"] for entry in result.position_importance]
    frequency_scores = [
        max(0.0, 1.0 - (idx / max(len(attention_scores), 1)))
        for idx in range(len(attention_scores))
    ]
    return attention_scores, recency_scores, frequency_scores


def _compare_with_real_signal(
    prompt: str,
    model_name: str,
    total_tokens: int | None = None,
) -> list[dict[str, object]]:
    attention_scores, recency_scores, frequency_scores = _signal_vectors_from_prompt(
        prompt, model_name
    )
    # Compare fixed and adaptive policies using real model signals.
    effective_total = total_tokens or len(attention_scores)
    if effective_total <= 0:
        raise ValueError("total_tokens must be positive")
    if effective_total > len(attention_scores):
        pad = [0.0] * (effective_total - len(attention_scores))
        attention_scores = attention_scores + pad
        recency_scores = recency_scores + pad
        frequency_scores = frequency_scores + pad

    records: list[dict[str, object]] = []
    for ratio in VALID_BUDGETS:
        fixed = compare_policies_at_budget(effective_total, ratio)
        for policy_name, selection in fixed.items():
            records.append(
                {
                    "policy": policy_name,
                    "budget_ratio": ratio,
                    "budget_tokens": selection.budget_tokens,
                    "retained_tokens": selection.retained_tokens,
                    "keep_indices": selection.keep_indices,
                }
            )

        budget = budget_tokens_for_ratio(effective_total, ratio)
        adaptive = AdaptiveImportancePolicy().select_indices(
            effective_total,
            budget,
            attention_scores=attention_scores,
            recency_scores=recency_scores,
            frequency_scores=frequency_scores,
        )
        records.append(
            {
                "policy": "adaptive",
                "budget_ratio": ratio,
                "budget_tokens": budget,
                "retained_tokens": len(adaptive),
                "keep_indices": adaptive,
            }
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare full, recency, uniform, and adaptive retention using a model-derived attention signal."
    )
    # Parse evaluation options and save the comparison records.
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of memory in transformers is",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/attention_cache_evaluation.json"),
    )
    parser.add_argument(
        "--total-tokens",
        type=int,
        default=None,
        help="Optional override for the number of token positions evaluated.",
    )
    args = parser.parse_args()

    records = _compare_with_real_signal(args.prompt, args.model, total_tokens=args.total_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
