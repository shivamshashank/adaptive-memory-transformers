"""Command-line interface for reproducible baseline runs and reports."""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence, cast

from amt.baseline import DEFAULT_MODEL, BaselineResult, run_full_cache_baseline


@dataclass(frozen=True)
class BaselinePlan:
    experiment_id: str = "manual-baseline"
    model: str = DEFAULT_MODEL
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    prompt: str = "The future of memory in transformers is"
    prompt_lengths: tuple[int | None, ...] = (None,)
    max_new_tokens: int = 16
    seeds: tuple[int, ...] = (42,)
    repeats_per_seed: int = 1
    dtypes: tuple[str, ...] = ("float32",)
    device: str | None = None
    output_dir: Path = Path("results/baseline")


def _section(data: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section [{name}] must be a table")
    return cast(dict[str, object], value)


def _optional_str(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def load_plan(path: Path) -> BaselinePlan:
    with path.open("rb") as handle:
        raw: object = tomllib.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a table")
    data = cast(dict[str, object], raw)
    schema_version = _positive_int(data.get("schema_version"), "schema_version")
    if schema_version != 1:
        raise ValueError(f"Unsupported schema_version: {schema_version}")
    experiment_id = _optional_str(data.get("experiment_id"), "experiment_id")
    if experiment_id is None:
        raise ValueError("experiment_id is required")
    model = _section(data, "model")
    generation = _section(data, "generation")
    reproducibility = _section(data, "reproducibility")
    outputs = _section(data, "outputs")

    model_id = _optional_str(model.get("id", DEFAULT_MODEL), "model.id") or DEFAULT_MODEL
    prompt = _optional_str(generation.get("prompt"), "generation.prompt")
    if prompt is None:
        raise ValueError("generation.prompt is required")

    seeds_value = reproducibility.get("seeds", [42])
    if not isinstance(seeds_value, list) or not seeds_value:
        raise ValueError("reproducibility.seeds must be a non-empty array")
    seeds = tuple(_nonnegative_int(seed, "reproducibility.seeds") for seed in seeds_value)

    prompt_tokens = _positive_int(generation.get("prompt_tokens", 1), "generation.prompt_tokens")
    output_value = _optional_str(outputs.get("directory"), "outputs.directory")
    if output_value is None:
        raise ValueError("outputs.directory is required")

    return BaselinePlan(
        experiment_id=experiment_id,
        model=model_id,
        model_revision=_optional_str(model.get("revision"), "model.revision"),
        tokenizer_revision=_optional_str(
            model.get("tokenizer_revision"), "model.tokenizer_revision"
        ),
        prompt=prompt,
        prompt_lengths=(prompt_tokens,),
        max_new_tokens=_positive_int(
            generation.get("max_new_tokens", 16), "generation.max_new_tokens"
        ),
        seeds=seeds,
        repeats_per_seed=_positive_int(
            reproducibility.get("repeats_per_seed", 1),
            "reproducibility.repeats_per_seed",
        ),
        dtypes=(_optional_str(model.get("dtype", "float32"), "model.dtype") or "float32",),
        device=_optional_str(model.get("device"), "model.device"),
        output_dir=Path(output_value),
    )


def _save_result(result: BaselineResult, output_dir: Path, run_number: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"baseline_{time.time_ns()}_{run_number:03d}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def _run_baseline(args: argparse.Namespace) -> int:
    plan = load_plan(args.config) if args.config else BaselinePlan()
    if args.model is not None:
        plan = replace(plan, model=args.model)
    if args.prompt is not None:
        plan = replace(plan, prompt=args.prompt)
    if args.max_new_tokens is not None:
        plan = replace(plan, max_new_tokens=args.max_new_tokens)
    if args.output_dir is not None:
        plan = replace(plan, output_dir=args.output_dir)

    run_number = 0
    for dtype in plan.dtypes:
        for prompt_length in plan.prompt_lengths:
            for seed in plan.seeds:
                for _ in range(plan.repeats_per_seed):
                    run_number += 1
                    result = run_full_cache_baseline(
                        experiment_id=plan.experiment_id,
                        model_name=plan.model,
                        model_revision=plan.model_revision,
                        tokenizer_revision=plan.tokenizer_revision,
                        prompt=plan.prompt,
                        max_new_tokens=plan.max_new_tokens,
                        device=plan.device,
                        seed=seed,
                        prompt_length=prompt_length,
                        dtype=dtype,
                    )
                    result_path = _save_result(result, plan.output_dir, run_number)
                    print(json.dumps(result.to_dict(), indent=2))
                    print(f"Saved result to {result_path}")
    return 0


def _run_summary(args: argparse.Namespace) -> int:
    from amt.reporting import summarize_results

    summary = summarize_results(args.input_dir, args.output_dir)
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="amt", description="Adaptive KV-cache research tools.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    baseline = subcommands.add_parser("baseline", help="Run the full-cache reference baseline.")
    baseline.add_argument("--config", type=Path, help="Versioned TOML experiment configuration.")
    baseline.add_argument("--model", help="Model identifier or local path.")
    baseline.add_argument("--prompt", help="Prompt text.")
    baseline.add_argument("--max-new-tokens", type=int, help="Number of tokens to generate.")
    baseline.add_argument("--output-dir", type=Path, help="Directory for JSON records.")
    baseline.set_defaults(handler=_run_baseline)

    summary = subcommands.add_parser("summarize", help="Summarize baseline JSON records.")
    summary.add_argument("--input-dir", type=Path, default=Path("results/baseline"))
    summary.add_argument("--output-dir", type=Path)
    summary.set_defaults(handler=_run_summary)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = cast(Callable[[argparse.Namespace], int], args.handler)
    return handler(args)
