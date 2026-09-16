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
