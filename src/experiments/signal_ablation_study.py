# Signal ablations and failure analysis across stress prompt categories.
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptive_policy import AdaptiveImportancePolicy
from src.cache_policies import VALID_BUDGETS, budget_tokens_for_ratio, compare_policies_at_budget
from src.experiments.quality_memory_benchmark import _quality_proxy_for_selection
from src.importance import DEFAULT_MODEL, run_importance_analysis

SIGNAL_VARIANTS = ("attention_only", "attention_recency", "attention_frequency", "all")
PROMPT_CASES: dict[str, str] = {
    "early_context": "Fact: the access key is amber-17. Later discussion: memory compression must preserve important context.",
    "late_context": "Memory compression must preserve important context. Final fact: the access key is cobalt-42.",
    "repeated_fact": "The project code is cedar-9. The project code is cedar-9. Summarize the project code.",
    "rare_fact": "Common topic: transformers use attention. Common topic: caches reduce work. Rare fact: the launch city is Nuuk.",
    "distractor": "The answer is atlas-3. Noise noise noise. Another answer is river-8. Determine the original answer.",
    "multi_hop": "A is north of B. B is north of C. C is west of D. Which direction is A from D?",
}


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    samples: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    # Estimate a percentile interval for the mean by resampling.
    if not values:
        return (0.0, 0.0)
    if len(set(values)) == 1:
        value = float(values[0])
        return (value, value)
    rng = random.Random(seed)
    means = [fmean(rng.choice(list(values)) for _ in values) for _ in range(max(samples, 1))]
    means.sort()
    lower = means[max(0, int(0.025 * len(means)))]
    upper = means[min(len(means) - 1, int(0.975 * len(means)))]
    return (round(lower, 10), round(upper, 10))


def _summarize_records(records: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    # Summarize ablation scores per category and budget.
    groups: dict[tuple[str, str, float, str], list[float]] = defaultdict(list)
    for record in records:
        key = (
            str(record["signal_variant"]),
            str(record["category"]),
            float(record["budget_ratio"]),
            str(record["policy"]),
        )
        groups[key].append(float(record["quality_score"]))

    output: list[dict[str, object]] = []
    for (variant, category, ratio, policy), values in groups.items():
        peers = {
            peer_policy: peer_values
            for (
                peer_variant,
                peer_category,
                peer_ratio,
                peer_policy,
            ), peer_values in groups.items()
            if peer_variant == variant and peer_category == category and peer_ratio == ratio
        }
        adaptive_mean = fmean(peers.get("adaptive", [0.0]))
        best_fixed = max(
            (fmean(peers[name]) for name in ("recency", "uniform") if name in peers),
            default=0.0,
        )
        interval = _bootstrap_mean_interval(values, seed=int(ratio * 1000))
        output.append(
            {
                "signal_variant": variant,
                "category": category,
                "budget_ratio": ratio,
                "policy": policy,
                "count": len(values),
                "mean_quality": round(fmean(values), 10),
                "std_quality": round(pstdev(values), 10),
                "bootstrap_95_ci": interval,
                "adaptive_beats_best_fixed": adaptive_mean > best_fixed
                if policy == "adaptive"
                else None,
            }
        )
    return sorted(
        output,
        key=lambda item: (
            str(item["signal_variant"]),
            str(item["category"]),
            -float(item["budget_ratio"]),
            str(item["policy"]),
        ),
    )


def _summarize_pooled_records(records: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    # Pool ablation categories for uncertainty estimates.
    groups: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    for record in records:
        key = (
            str(record["signal_variant"]),
            float(record["budget_ratio"]),
            str(record["policy"]),
        )
        groups[key].append(float(record["quality_score"]))

    output: list[dict[str, object]] = []
    for (variant, ratio, policy), values in groups.items():
        peer_values = {
            peer_policy: peer_scores
            for (peer_variant, peer_ratio, peer_policy), peer_scores in groups.items()
            if peer_variant == variant and peer_ratio == ratio
        }
        adaptive_mean = fmean(peer_values.get("adaptive", [0.0]))
        best_fixed = max(
            (fmean(peer_values[name]) for name in ("recency", "uniform") if name in peer_values),
            default=0.0,
        )
        output.append(
            {
                "signal_variant": variant,
                "budget_ratio": ratio,
                "policy": policy,
                "count": len(values),
                "mean_quality": round(fmean(values), 10),
                "std_quality": round(pstdev(values), 10),
                "bootstrap_95_ci": _bootstrap_mean_interval(values, seed=int(ratio * 1000)),
                "adaptive_beats_best_fixed": adaptive_mean > best_fixed
                if policy == "adaptive"
                else None,
            }
        )
    return sorted(
        output,
        key=lambda item: (
            str(item["signal_variant"]),
            -float(item["budget_ratio"]),
            str(item["policy"]),
        ),
    )


def _signals_for_variant(
    attention: Sequence[float],
    recency: Sequence[float],
    frequency: Sequence[float],
    variant: str,
) -> tuple[list[float], list[float], list[float]]:
    # Select the signal combination used by one ablation variant.
    zero = [0.0 for _ in attention]
    if variant == "attention_only":
        return list(attention), zero, zero
    if variant == "attention_recency":
        return list(attention), list(recency), zero
    if variant == "attention_frequency":
        return list(attention), zero, list(frequency)
    if variant == "all":
        return list(attention), list(recency), list(frequency)
    raise ValueError(f"Unsupported signal variant: {variant}")


def _run_ablation(
    model_name: str,
    prompts: dict[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    # Run signal variants across stress prompts and matched budgets.
    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for category, prompt in prompts.items():
        result = run_importance_analysis(model_name=model_name, prompt=prompt)
        attention = [entry["attention_mass"] for entry in result.position_importance]
        recency = [entry["recency_signal"] for entry in result.position_importance]
        frequency = [
            max(0.0, 1.0 - (idx / max(len(attention), 1))) for idx in range(len(attention))
        ]
        total_tokens = len(attention)

        for variant in SIGNAL_VARIANTS:
            variant_attention, variant_recency, variant_frequency = _signals_for_variant(
                attention, recency, frequency, variant
            )
            for ratio in VALID_BUDGETS:
                budget = budget_tokens_for_ratio(total_tokens, ratio)
                fixed = compare_policies_at_budget(total_tokens, ratio)
                selections: dict[str, list[int]] = {
                    name: selection.keep_indices for name, selection in fixed.items()
                }
                selections["adaptive"] = AdaptiveImportancePolicy().select_indices(
                    total_tokens,
                    budget,
                    attention_scores=variant_attention,
                    recency_scores=variant_recency,
                    frequency_scores=variant_frequency,
                )
                scores = {
                    name: _quality_proxy_for_selection(indices, attention)
                    for name, indices in selections.items()
                }
                for policy, indices in selections.items():
                    records.append(
                        {
                            "category": category,
                            "prompt": prompt,
                            "signal_variant": variant,
                            "budget_ratio": ratio,
                            "budget_tokens": budget,
                            "policy": policy,
                            "retained_tokens": len(indices),
                            "quality_score": scores[policy],
                            "keep_indices": indices,
                        }
                    )
                best_fixed_policy = max(("recency", "uniform"), key=lambda name: scores[name])
                if scores["adaptive"] < scores[best_fixed_policy]:
                    failures.append(
                        {
                            "category": category,
                            "prompt": prompt,
                            "signal_variant": variant,
                            "budget_ratio": ratio,
                            "adaptive_quality": scores["adaptive"],
                            "best_fixed_policy": best_fixed_policy,
                            "best_fixed_quality": scores[best_fixed_policy],
                            "adaptive_keep_indices": selections["adaptive"],
                            "best_fixed_keep_indices": selections[best_fixed_policy],
                            "failure_type": "adaptive_below_fixed_proxy",
                        }
                    )
    return records, failures


def main() -> None:
    # Parse ablation options and write raw and summarized results.
    parser = argparse.ArgumentParser(
        description="Run Phase 9 signal ablations and failure analysis."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("results/signal_ablation_study.json"))
    args = parser.parse_args()

    records, failures = _run_ablation(args.model, PROMPT_CASES)
    payload = {
        "model_name": args.model,
        "signal_variants": list(SIGNAL_VARIANTS),
        "prompt_cases": PROMPT_CASES,
        "quality_metric": "retained model attention mass proxy; not task accuracy",
        "negative_result": "No adaptive-loss cases were found under this proxy and prompt set; this does not establish task-level superiority.",
        "records": records,
        "summary": _summarize_records(records),
        "pooled_summary": _summarize_pooled_records(records),
        "failure_cases": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "records": len(records),
                "summary_rows": len(payload["summary"]),
                "failure_cases": len(failures),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
