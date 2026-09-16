# Budget checks for full, recency, and uniform cache baselines.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache_policies import VALID_BUDGETS, compare_policies_at_budget


class _RunRecord(TypedDict):
    policy: str
    total_tokens: int
    budget_ratio: float
    budget_tokens: int
    retained_tokens: int
    keep_indices: list[int]


def _run_policy_budget_checks(total_tokens: int = 32) -> list[_RunRecord]:
    records: list[_RunRecord] = []
    # Check fixed policy retention across the configured budgets.
    for ratio in VALID_BUDGETS:
        by_policy = compare_policies_at_budget(total_tokens, ratio)
        for policy_name, selection in by_policy.items():
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
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fixed-policy baseline records for Phase 5 budgets."
    )
    # Parse budget-check options and write the policy records.
    parser.add_argument(
        "--total-tokens",
        type=int,
        default=32,
        help="Total token count used to evaluate the budgeted policies.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fixed_cache_baselines.json"),
        help="Where to write the JSON records.",
    )
    args = parser.parse_args()

    records = _run_policy_budget_checks(total_tokens=args.total_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
