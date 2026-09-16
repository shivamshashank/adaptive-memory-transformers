# Layer-aware allocation comparison under a matched total budget.
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.adaptive_policy import AdaptiveImportancePolicy
from src.cache_policies import VALID_BUDGETS, budget_tokens_for_ratio
from src.importance import run_importance_analysis
from src.layer_adaptive_policy import LayerAdaptivePolicy


def _aggregate_layer_scores(layer_scores: Sequence[float]) -> list[float]:
    """Return layer importance scores sorted from highest to lowest."""
    # Order layer scores for comparison and reporting.
    return sorted((float(score) for score in layer_scores), reverse=True)


def _layer_budget_allocation(layer_scores: Sequence[float], total_budget: int) -> list[int]:
    """Allocate a fixed total token budget across layers using proportional importance."""
    # Allocate one total budget across layers by importance.
    if total_budget <= 0:
        return [0 for _ in layer_scores]
    if not layer_scores:
        return []

    scores = [max(0.0, float(score)) for score in layer_scores]
    policy = LayerAdaptivePolicy()
    allocations = policy.allocate_budgets(scores, total_budget)
    return allocations


def layer_adaptive_selection(
    layer_scores: Sequence[float],
    tokens_per_layer: Sequence[int],
    total_budget: int,
) -> list[int]:
    """Return the per-layer token budget using the same layer-weighted allocation scheme."""
    # Return the per-layer token budgets for a matched allocation.
    if len(tokens_per_layer) != len(layer_scores):
        raise ValueError("tokens_per_layer must contain one capacity per layer")
    capacities = [max(0, int(value)) for value in tokens_per_layer]
    target = min(max(total_budget, 0), sum(capacities))
    allocation = [0 for _ in capacities]
    scores = [max(0.0, float(value)) for value in layer_scores]
    while sum(allocation) < target:
        candidates = [
            index for index, capacity in enumerate(capacities) if allocation[index] < capacity
        ]
        if not candidates:
            break
        # Largest-remainder allocation with capacities prevents a high-scoring
        # layer from consuming every slot before lower layers receive theirs.
        chosen = max(
            candidates,
            key=lambda index: (
                (target * scores[index] / sum(scores) if sum(scores) else target / len(scores))
                - allocation[index],
                -index,
            ),
        )
        allocation[chosen] += 1
    return allocation


def _layer_attention_scores(model_name: str, prompt: str) -> list[float]:
    # Extract one attention-mass score for every model layer.
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            output_attentions=True,
            use_cache=False,
        )

    if outputs.attentions is None:
        raise ValueError("The model did not return attention tensors.")

    scores: list[float] = []
    for layer_attention in outputs.attentions:
        layer = layer_attention[0]
        per_position = layer.mean(dim=0).sum(dim=0)
        scores.append(float(per_position.sum().item()))
    return scores


def _layer_adaptive_keep_indices(
    token_scores: Sequence[float],
    layer_scores: Sequence[float],
    total_budget: int,
) -> list[dict[str, object]]:
    # Build layer allocation records from token and layer scores.
    if total_budget <= 0:
        return []

    budgets = _layer_budget_allocation(layer_scores, total_budget)
    ranked = sorted(
        range(len(token_scores)), key=lambda idx: float(token_scores[idx]), reverse=True
    )
    layer_entries: list[dict[str, object]] = []

    for layer_index, layer_budget in enumerate(budgets):
        if layer_budget <= 0:
            layer_entries.append({"layer": layer_index, "budget": 0, "keep_indices": []})
            continue
        keep = ranked[:layer_budget]
        layer_entries.append(
            {
                "layer": layer_index,
                "budget": layer_budget,
                "keep_indices": sorted(keep),
                "layer_score": float(layer_scores[layer_index])
                if layer_index < len(layer_scores)
                else 0.0,
            }
        )

    return layer_entries


def compare_global_vs_layer_adaptive(
    prompt: str,
    model_name: str,
    total_tokens: int | None = None,
) -> dict[str, object]:
    # Compare global and layer-aware selection at one matched budget.
    result = run_importance_analysis(model_name=model_name, prompt=prompt)
    token_scores = [item["attention_mass"] for item in result.position_importance]
    if total_tokens is not None:
        if total_tokens > len(token_scores):
            token_scores = token_scores + [0.0] * (total_tokens - len(token_scores))
        else:
            token_scores = token_scores[:total_tokens]

    layer_scores = _layer_attention_scores(model_name, prompt)
    total_budget = total_tokens or len(token_scores)
    global_budget = budget_tokens_for_ratio(total_budget, 0.5)
    global_keep = AdaptiveImportancePolicy().select_indices(
        total_budget,
        global_budget,
        attention_scores=token_scores,
        recency_scores=[
            item["recency_signal"] for item in result.position_importance[: len(token_scores)]
        ],
        frequency_scores=[
            max(0.0, 1.0 - (idx / max(len(token_scores), 1))) for idx in range(len(token_scores))
        ],
    )

    # Global retention keeps global_budget positions in every layer.  Allocate the
    # same total number of layer-token slots for the layer-aware alternative.
    layer_keep = _layer_adaptive_keep_indices(
        token_scores,
        layer_scores,
        global_budget * len(layer_scores),
    )
    return {
        "model_name": model_name,
        "prompt": prompt,
        "total_tokens": total_budget,
        "global_budget": global_budget,
        "global_keep": global_keep,
        "layer_budget": sum(item["budget"] for item in layer_keep),
        "layer_allocations": layer_keep,
        "layer_scores": layer_scores,
        "layer_score_summary": _aggregate_layer_scores(layer_scores),
        "budget_ratios": {
            ratio: budget_tokens_for_ratio(total_budget, ratio) for ratio in VALID_BUDGETS
        },
    }


def main() -> None:
    # Parse layer-allocation options and write the comparison artifact.
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare a global adaptive policy against a layer-aware allocation policy."
    )
    parser.add_argument("--model", type=str, default="hf-internal-testing/tiny-random-gpt2")
    parser.add_argument("--prompt", type=str, default="The future of memory in transformers is")
    parser.add_argument(
        "--output", type=Path, default=Path("results/layer_adaptive_allocation.json")
    )
    args = parser.parse_args()

    comparison = compare_global_vs_layer_adaptive(args.prompt, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
