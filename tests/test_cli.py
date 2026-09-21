"""Tests for the unified command-line interface and versioned configuration."""

from pathlib import Path

from amt.cli import build_parser, load_plan


def test_smoke_config_is_executable_by_the_cli() -> None:
    plan = load_plan(Path("configs/smoke/v1.toml"))

    assert plan.experiment_id == "smoke-v1"
    assert plan.model == "hf-internal-testing/tiny-random-gpt2"
    assert plan.model_revision == "71034c5d8bde858ff824298bdedc65515b97d2b9"
    assert plan.prompt_lengths == (32,)
    assert plan.seeds == (41, 42, 43)


def test_cli_requires_a_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(["summarize", "--input-dir", "results/example"])

    assert args.command == "summarize"


def test_primary_model_candidate_is_pinned() -> None:
    plan = load_plan(Path("configs/models/qwen2.5-1.5b-instruct.toml"))

    assert plan.model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert plan.model_revision == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert plan.tokenizer_revision == plan.model_revision
    assert plan.dtypes == ("bfloat16",)
    assert plan.device == "mps"
