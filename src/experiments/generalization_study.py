# Cross-model and cross-task generalization study with reproducibility metadata.
from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import transformers

from src.experiments.quality_memory_benchmark import _run_policy_frontier
from src.importance import DEFAULT_MODEL

DEFAULT_MODELS = (DEFAULT_MODEL, "sshleifer/tiny-gpt2")
TASK_PROMPTS = {
    "retrieval": "The document says the recovery phrase is amber-17. Noise about unrelated systems follows. What is the recovery phrase?",
    "multi_hop": "A is north of B. B is north of C. C is west of D. Determine the direction of A relative to D.",
    "rare_fact": "Common statement: attention mixes token information. Rare fact: the launch city is Nuuk. Identify the rare fact.",
    "distractor": "The first answer is atlas-3. Distractor answer: river-8. Return the first answer only.",
}


def _paired_effect_size(
    adaptive_scores: Sequence[float], baseline_scores: Sequence[float]
) -> float:
    # Compute a paired standardized difference between policies.
    if len(adaptive_scores) != len(baseline_scores):
        raise ValueError("Paired score sequences must have equal length")
    if not adaptive_scores:
        return 0.0
    differences = [
        float(adaptive) - float(baseline)
        for adaptive, baseline in zip(adaptive_scores, baseline_scores)
    ]
    mean_difference = fmean(differences)
    deviation = pstdev(differences)
    if deviation == 0:
        return round(mean_difference, 10)
    return round(mean_difference / deviation, 10)


def _summarize_generalization(records: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, float, str], list[float]] = defaultdict(list)
    # Summarize scores for each model, task, budget, and policy.
    for record in records:
        key = (
            str(record["model"]),
            str(record["task"]),
            float(record["budget_ratio"]),
            str(record["policy"]),
        )
        groups[key].append(float(record["quality_score"]))

    output: list[dict[str, object]] = []
    for (model, task, ratio, policy), values in groups.items():
        adaptive = groups.get((model, task, ratio, "adaptive"), [])
        uniform = groups.get((model, task, ratio, "uniform"), [])
        output.append(
            {
                "model": model,
                "task": task,
                "budget_ratio": ratio,
                "policy": policy,
                "count": len(values),
                "mean_quality": round(fmean(values), 10),
                "std_quality": round(pstdev(values), 10),
                "adaptive_vs_uniform_effect": (
                    _paired_effect_size(adaptive, uniform)
                    if adaptive and uniform and policy == "adaptive"
                    else None
                ),
            }
        )
    policy_order = {"adaptive": 0, "uniform": 1, "recency": 2, "full": 3}
    return sorted(
        output,
        key=lambda row: (
            str(row["model"]),
            str(row["task"]),
            -float(row["budget_ratio"]),
            policy_order.get(str(row["policy"]), 99),
        ),
    )


def _summarize_pooled_generalization(
    records: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    # Pool task scores to estimate cross-task generalization.
    for record in records:
        key = (
            str(record["model"]),
            float(record["budget_ratio"]),
            str(record["policy"]),
        )
        groups[key].append(float(record["quality_score"]))

    output: list[dict[str, object]] = []
    for (model, ratio, policy), values in groups.items():
        adaptive = groups.get((model, ratio, "adaptive"), [])
        uniform = groups.get((model, ratio, "uniform"), [])
        output.append(
            {
                "model": model,
                "budget_ratio": ratio,
                "policy": policy,
                "count": len(values),
                "mean_quality": round(fmean(values), 10),
                "median_quality": round(sorted(values)[len(values) // 2], 10),
                "std_quality": round(pstdev(values), 10),
                "adaptive_vs_uniform_effect": (
                    _paired_effect_size(adaptive, uniform)
                    if adaptive and uniform and policy == "adaptive"
                    else None
                ),
            }
        )
    policy_order = {"adaptive": 0, "uniform": 1, "recency": 2, "full": 3}
    return sorted(
        output,
        key=lambda row: (
            str(row["model"]),
            -float(row["budget_ratio"]),
            policy_order.get(str(row["policy"]), 99),
        ),
    )


def _software_metadata() -> dict[str, object]:
    # Capture runtime and hardware details for reproducibility.
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _git_revision() -> str | None:
    # Read the current repository revision when available.
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _run_generalization(
    models: Sequence[str], seed: int
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    # Run the cross-model and cross-task experiment matrix.
    random.seed(seed)
    torch.manual_seed(seed)
    records: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for model_name in models:
        for task, prompt in TASK_PROMPTS.items():
            try:
                frontier = _run_policy_frontier(prompt, model_name)
            except Exception as error:  # noqa: BLE001 - preserve reproducible run failures
                errors.append({"model": model_name, "task": task, "error": repr(error)})
                continue
            for row in frontier:
                records.append(
                    {
                        "model": model_name,
                        "task": task,
                        "context_tokens": len(prompt.split()),
                        "seed": seed,
                        **row,
                    }
                )
    return records, errors


def main() -> None:
    # Parse generalization options and write the experiment artifact.
    parser = argparse.ArgumentParser(
        description="Run Phase 10 cross-model and cross-task generalization checks."
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/generalization_study.json"))
    args = parser.parse_args()

    records, errors = _run_generalization(args.models, args.seed)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "seed": args.seed,
        "models_requested": args.models,
        "tasks": TASK_PROMPTS,
        "budgets": [1.0, 0.75, 0.5, 0.25, 0.1],
        "quality_metric": "retained model attention mass proxy; not task accuracy",
        "software": _software_metadata(),
        "reproducibility_checklist": {
            "raw_records_saved": bool(records),
            "configuration_saved": True,
            "seed_recorded": True,
            "model_list_recorded": True,
            "task_prompts_recorded": True,
            "errors_preserved": True,
        },
        "records": records,
        "summary": _summarize_generalization(records),
        "pooled_summary": _summarize_pooled_generalization(records),
        "errors": errors,
        "limitations": [
            "The quality score is an attention-coverage proxy, not generation accuracy.",
            "The tiny models and short prompts do not establish long-context generalization.",
            "Unavailable model downloads are recorded as errors rather than silently omitted.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "records": len(records),
                "summary_rows": len(payload["summary"]),
                "errors": len(errors),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
