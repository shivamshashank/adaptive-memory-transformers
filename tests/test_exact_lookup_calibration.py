"""Construction, gate and evidence tests for exact-lookup calibration."""

import hashlib
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from amt.validation import write_json

experiment = SimpleNamespace(
    **runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/calibrate_exact_lookup.py"))
)


def test_dataset_is_fresh_balanced_and_excludes_confirmation():
    data = experiment.make_dataset()
    experiment.validate_data(data)
    assert len(data) == len({example["id"] for example in data}) == 32
    assert {example["seed"] for example in data} == set(experiment.DEVELOPMENT_SEEDS)
    assert all(example["seed"] != experiment.CONFIRMATION_SEED for example in data)
    assert all(example["query"].count(example["target_name"]) == 2 for example in data)
    encoded = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
    assert hashlib.sha256(encoded).hexdigest() == experiment.DATASET_SHA256
    assert {
        (seed, facts, color): sum(
            example["seed"] == seed and example["fact_count"] == facts and example["gold"] == color
            for example in data
        )
        for seed in experiment.DEVELOPMENT_SEEDS
        for facts in (16, 32)
        for color in experiment.COLORS
    } == {
        (seed, facts, color): 2
        for seed in experiment.DEVELOPMENT_SEEDS
        for facts in (16, 32)
        for color in experiment.COLORS
    }


def make_rows(data, errors=()):
    rows = []
    for index, example in enumerate(data):
        prediction = "blue" if index in errors and example["gold"] != "blue" else example["gold"]
        if index in errors and example["gold"] == "blue":
            prediction = "red"
        scores = {color: float(color == prediction) for color in experiment.COLORS}
        rows.append(
            {
                "example_id": example["id"],
                **{
                    key: example[key]
                    for key in ("seed", "gold", "fact_count", "target_fact_index", "target_depth")
                },
                **experiment.choose(scores, example["gold"]),
                "color_logits": scores,
                "raw_greedy_text": prediction,
                "budget_tokens": 10,
                "scoring_cache_tokens": 12,
                "dense_check": "passed",
                "dense_max_abs_difference": 0.0,
                "dense_color_logits": scores,
            }
        )
    return rows


def test_gate_passes_at_declared_minimum_and_rejects_cell_failure():
    data = experiment.make_dataset()
    errors = {0, 8, 16, 24}
    assert experiment.summarize(make_rows(data, errors), "complete")["gate"] == "passed"
    errors.add(25)
    assert experiment.summarize(make_rows(data, errors), "complete")["gate"] == "failed"


@pytest.fixture
def evidence(tmp_path):
    data = experiment.make_dataset()
    write_json(tmp_path / "dataset.json", data)
    write_json(
        tmp_path / "run.json",
        {
            "dataset_sha256": hashlib.sha256((tmp_path / "dataset.json").read_bytes()).hexdigest(),
            "confirmation_seed": experiment.CONFIRMATION_SEED,
            "confirmation_seed_consumed": False,
        },
    )
    write_json(
        tmp_path / "inputs.json",
        {
            "examples": [
                {"id": example["id"], "context_ids": list(range(10)), "query_ids": [10, 11]}
                for example in data
            ]
        },
    )
    rows = make_rows(data)
    (tmp_path / "rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    experiment.save(tmp_path, rows, "complete")
    return tmp_path, rows


def test_verifier_passes_and_rejects_corrupted_scores(evidence):
    path, rows = evidence
    assert experiment.verify(path)["status"] == "passed"
    rows[0]["prediction"] = "green"
    (path / "rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(AssertionError):
        experiment.verify(path)
