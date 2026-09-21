"""Multi-seed construction, paired analyses, gates and evidence checks."""

import hashlib
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from amt.validation import write_json

experiment = SimpleNamespace(
    **runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/run_query_aware_larger.py"))
)


def test_dataset_is_frozen_balanced_and_excludes_confirmation():
    data = experiment.make_dataset()
    experiment.validate_data(data)
    assert len(data) == len({example["id"] for example in data}) == 32
    assert {example["seed"] for example in data} == set(experiment.DEVELOPMENT_SEEDS)
    assert all(example["seed"] != experiment.CONFIRMATION_SEED for example in data)
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


def test_frozen_paired_statistics():
    assert experiment.exact_paired_p(0, 0) == 1
    assert experiment.exact_paired_p(4, 0) == 0.125
    assert experiment.exact_paired_p(3, 1) == 0.625
    assert experiment.bootstrap_interval([1, 1, 1, 1]) == [1.0, 1.0]
    assert experiment.bootstrap_interval([]) is None


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
                {
                    "id": example["id"],
                    "context_ids": list(range(10)),
                    "query_ids": [10, 11],
                    "question_probe_ids": [10],
                    "target_span": [2, 3],
                }
                for example in data
            ]
        },
    )
    rows = []
    for example in data:
        scores = {color: float(color == example["gold"]) for color in experiment.COLORS}
        for policy in experiment.POLICIES:
            full = policy == "full"
            retained = list(range(10)) if full else [0, 2, 3, 8, 9]
            query_aware = policy == "adaptive-query+first"
            rows.append(
                {
                    "example_id": example["id"],
                    "policy": policy,
                    "first_token_protected": not full,
                    **{
                        key: example[key]
                        for key in (
                            "seed",
                            "gold",
                            "fact_count",
                            "target_fact_index",
                            "target_depth",
                        )
                    },
                    **experiment.choose(scores, example["gold"]),
                    "raw_greedy_text": example["gold"],
                    "color_logits": scores,
                    "budget_tokens": len(retained),
                    "retained_positions": retained,
                    "target_retained_tokens": 2,
                    "target_retained_fraction": 1.0,
                    "retained_kv_bytes": 100 * len(retained),
                    "scoring_cache_tokens": len(retained) + 2,
                    "scoring_kv_bytes": 100 * (len(retained) + 2),
                    "selector_attention_by_layer": [[0.1] * 10] if query_aware else [],
                    "probe_seconds": 1.0 if query_aware else 0,
                    "probe_cache_tokens": 11 if query_aware else None,
                    "dense_check": "passed" if full else None,
                    "dense_color_logits": scores if full else None,
                }
            )
    (tmp_path / "rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    experiment.save(tmp_path, rows, "complete")
    return tmp_path, rows


def test_verifier_and_decision_rule(evidence):
    path, rows = evidence
    assert experiment.verify(path)["status"] == "passed"
    assert experiment.summarize(rows, "complete")["confirmation_decision"] == "diagnose"
    for policy in ("recency+first", "adaptive+first"):
        candidates = [row for row in rows if row["policy"] == policy]
        for row in candidates[:4]:
            row["correct"] = False
    summary = experiment.summarize(rows, "complete")
    assert summary["confirmation_decision"] == "proceed"
    assert all(
        item["query_correct"] - item["baseline_correct"] == 4 for item in summary["comparisons"]
    )


def test_decision_rejects_cell_regression(evidence):
    _, rows = evidence
    for policy in ("recency+first", "adaptive+first"):
        candidates = [row for row in rows if row["policy"] == policy]
        for row in candidates[:4]:
            row["correct"] = False
    query_cell = [
        row
        for row in rows
        if row["policy"] == "adaptive-query+first"
        and row["seed"] == experiment.DEVELOPMENT_SEEDS[1]
        and row["fact_count"] == 32
    ]
    query_cell[0]["correct"] = False
    assert experiment.summarize(rows, "complete")["confirmation_decision"] == "diagnose"


@pytest.mark.parametrize(
    "field,value",
    [
        ("prediction", "purple"),
        ("budget_tokens", 99),
        ("retained_positions", [0, 1, 1, 2, 3]),
        ("target_retained_tokens", 99),
        ("scoring_kv_bytes", 999),
        ("dense_color_logits", dict.fromkeys(experiment.COLORS, 100.0)),
    ],
)
def test_verifier_rejects_corruption(evidence, field, value):
    path, rows = evidence
    rows[0][field] = value
    (path / "rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(AssertionError):
        experiment.verify(path)
