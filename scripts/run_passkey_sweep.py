"""Run a reproducible passkey depth and context-length sweep."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.passkey_retrieval_evaluation import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_NEEDLE_DEPTH,
    DEFAULT_PASSKEY,
    evaluate_passkey_retrieval,
)


def _parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("values must be comma-separated numbers in [0.0, 1.0]")
    return values


def _parse_int_list(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("values must be comma-separated integers") from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("context lengths must be positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run passkey retrieval over depths and context lengths."
    )
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--passkey", default=DEFAULT_PASSKEY)
    parser.add_argument("--depths", type=_parse_float_list, default=[DEFAULT_NEEDLE_DEPTH])
    parser.add_argument("--context-lengths", type=_parse_int_list, default=[DEFAULT_CONTEXT_LENGTH])
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("results/passkey_retrieval_sweep.json"))
    args = parser.parse_args()

    runs: list[dict[str, object]] = []
    for context_length in args.context_lengths:
        for depth in args.depths:
            prompt_spec, records = evaluate_passkey_retrieval(
                model_name=args.model,
                passkey=args.passkey,
                context_length=context_length,
                depth=depth,
                max_new_tokens=args.max_new_tokens,
            )
            runs.append(
                {
                    "context_length_requested": context_length,
                    "context_length_actual": prompt_spec.actual_tokens,
                    "needle_depth": depth,
                    "prompt_spec": asdict(prompt_spec),
                    "records": [asdict(record) for record in records],
                }
            )

    payload = {
        "benchmark": "Passkey Retrieval (Needle-in-a-Haystack)",
        "model_name": args.model,
        "passkey": args.passkey,
        "depths": args.depths,
        "context_lengths": args.context_lengths,
        "max_new_tokens": args.max_new_tokens,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"status": "success", "runs": len(runs), "output": str(args.output)}, indent=2)
    )


if __name__ == "__main__":
    main()
