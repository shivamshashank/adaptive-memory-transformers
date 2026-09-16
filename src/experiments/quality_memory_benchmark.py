# Quality-memory frontier benchmark for fixed and adaptive policies.
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptive_policy import AdaptiveImportancePolicy
from src.cache_policies import VALID_BUDGETS, budget_tokens_for_ratio, compare_policies_at_budget
from src.importance import DEFAULT_MODEL, run_importance_analysis


def _quality_proxy_for_selection(
    retained_indices: Sequence[int], attention_scores: Sequence[float]
) -> float:
    # Measure retained attention mass as the benchmark quality proxy.
    if not retained_indices:
        return 0.0
    if not attention_scores:
        return 0.0
    total = sum(float(value) for value in attention_scores)
    if total <= 0:
        return 0.0
    retained = sum(
        float(attention_scores[idx]) for idx in retained_indices if idx < len(attention_scores)
    )
    return round(retained / total, 10)


def _summarize_policy_frontier(records: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    # Order benchmark records from full to compressed budgets.
    return [
        dict(record)
        for record in sorted(
            records,
            key=lambda item: float(item.get("budget_ratio", 1.0)),
            reverse=True,
        )
    ]


def _run_policy_frontier(
    prompt: str, model_name: str, total_tokens: int | None = None
) -> list[dict[str, object]]:
    # Generate the fixed-versus-adaptive quality frontier.
    result = run_importance_analysis(model_name=model_name, prompt=prompt)
    attention_scores = [entry["attention_mass"] for entry in result.position_importance]
    effective_total = total_tokens or len(attention_scores)
    if effective_total <= 0:
        raise ValueError("total_tokens must be positive")
    if effective_total > len(attention_scores):
        attention_scores = attention_scores + [0.0] * (effective_total - len(attention_scores))
    else:
        attention_scores = attention_scores[:effective_total]

    recency_scores = [
        entry["recency_signal"] for entry in result.position_importance[: len(attention_scores)]
    ]
    frequency_scores = [
        max(0.0, 1.0 - (idx / max(len(attention_scores), 1)))
        for idx in range(len(attention_scores))
    ]

    records: list[dict[str, object]] = []
    for ratio in VALID_BUDGETS:
        budget = budget_tokens_for_ratio(effective_total, ratio)
        fixed = compare_policies_at_budget(effective_total, ratio)
        for policy_name, selection in fixed.items():
            records.append(
                {
                    "policy": policy_name,
                    "budget_ratio": ratio,
                    "budget_tokens": selection.budget_tokens,
                    "retained_tokens": selection.retained_tokens,
                    "quality_score": _quality_proxy_for_selection(
                        selection.keep_indices, attention_scores
                    ),
                    "keep_indices": selection.keep_indices,
                }
            )

        adaptive_keep = AdaptiveImportancePolicy().select_indices(
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
                "retained_tokens": len(adaptive_keep),
                "quality_score": _quality_proxy_for_selection(adaptive_keep, attention_scores),
                "keep_indices": adaptive_keep,
            }
        )

    return records


def main() -> None:
    # Parse benchmark options and write the frontier artifact.
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the main quality-memory benchmark frontier for fixed and adaptive cache policies."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--prompt", type=str, default="The future of memory in transformers is")
    parser.add_argument(
        "--output", type=Path, default=Path("results/quality_memory_benchmark.json")
    )
    parser.add_argument("--total-tokens", type=int, default=None)
    args = parser.parse_args()

    records = _run_policy_frontier(args.prompt, args.model, total_tokens=args.total_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
