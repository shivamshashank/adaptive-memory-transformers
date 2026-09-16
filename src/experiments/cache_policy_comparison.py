# Matched-budget comparison of fixed and adaptive cache policies.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptive_policy import AdaptiveImportancePolicy
from src.cache_policies import VALID_BUDGETS, budget_tokens_for_ratio, compare_policies_at_budget


def _summarize_by_ratio(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_ratio: dict[float, list[dict[str, object]]] = {}
    # Aggregate policy records by matched budget ratio.
    for record in records:
        ratio = float(record["budget_ratio"])
        by_ratio.setdefault(ratio, []).append(record)

    for ratio in sorted(by_ratio):
        entries = by_ratio[ratio]
        rows.append(
            {
                "budget_ratio": ratio,
                "policies": {
                    str(entry["policy"]): {
                        "budget_tokens": entry["budget_tokens"],
                        "retained_tokens": entry["retained_tokens"],
                        "keep_indices": entry["keep_indices"],
                    }
                    for entry in entries
                },
            }
        )
    return rows


def _adaptive_selection_for_ratio(
    total_tokens: int,
    ratio: float,
    *,
    attention_scores: list[float] | None = None,
    recency_scores: list[float] | None = None,
    frequency_scores: list[float] | None = None,
) -> dict[str, object]:
    budget = budget_tokens_for_ratio(total_tokens, ratio)
    # Select adaptive positions for one synthetic budget ratio.
    if attention_scores is None:
        attention_scores = [0.1 + (idx / max(total_tokens, 1)) for idx in range(total_tokens)]
    if recency_scores is None:
        recency_scores = [
            0.1 + (1.0 - idx / max(total_tokens - 1, 1)) for idx in range(total_tokens)
        ]
    if frequency_scores is None:
        frequency_scores = [1.0 if idx % 2 == 0 else 0.0 for idx in range(total_tokens)]

    policy = AdaptiveImportancePolicy()
    kept = policy.select_indices(
        total_tokens,
        budget,
        attention_scores=attention_scores,
        recency_scores=recency_scores,
        frequency_scores=frequency_scores,
    )
    return {
        "policy": "adaptive",
        "total_tokens": total_tokens,
        "budget_ratio": ratio,
        "budget_tokens": budget,
        "retained_tokens": len(kept),
        "keep_indices": kept,
    }


def _compare_policies(total_tokens: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    # Produce matched-budget records for all policy families.
    for ratio in VALID_BUDGETS:
        fixed = compare_policies_at_budget(total_tokens, ratio)
        for policy_name, selection in fixed.items():
            records.append(
                {
                    "policy": policy_name,
                    "total_tokens": selection.total_tokens,
                    "budget_ratio": ratio,
                    "budget_tokens": selection.budget_tokens,
                    "retained_tokens": selection.retained_tokens,
                    "keep_indices": selection.keep_indices,
                }
            )

        adaptive = _adaptive_selection_for_ratio(total_tokens, ratio)
        records.append(adaptive)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the simple adaptive importance policy against fixed baselines at matched budgets."
    )
    # Parse comparison options and write raw and summarized records.
    parser.add_argument(
        "--total-tokens",
        type=int,
        default=32,
        help="Total tokens available for the comparison.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cache_policy_comparison.json"),
        help="Path for the JSON comparison output.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/cache_policy_summary.json"),
        help="Path for the compact ratio-level summary output.",
    )
    args = parser.parse_args()

    records = _compare_policies(args.total_tokens)
    summary = _summarize_by_ratio(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("raw comparison:")
    print(json.dumps(records, indent=2))
    print("\nsummary by ratio:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
