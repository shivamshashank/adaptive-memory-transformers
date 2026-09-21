"""Saved-evidence checks must reject malformed records, not just accept good ones."""

import hashlib
import json
import runpy
from pathlib import Path

import pytest

from amt.evaluation import make_examples, summarize
from amt.validation import write_json

verify = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/verify_pilot.py"))[
    "verify"
]


@pytest.fixture
def evidence(tmp_path):
    data = make_examples(1)
    key = data[0]["id"]
    gold = data[0]["gold"]
    write_json(tmp_path / "dataset.json", data)
    write_json(
        tmp_path / "inputs.json",
        {"examples": [{"id": key, "context_ids": [1] * 10, "query_ids": [2, 3]}]},
    )
    settings = [("full", None)] + [
        (name, ratio)
        for ratio in (0.5, 0.25)
        for name in ("recency", "uniform", "attention", "adaptive")
    ]
    rows = []
    for name, ratio in settings:
        budget = 10 if ratio is None else int(10 * ratio)
        rows.append(
            {
                "example_id": key,
                "policy": name,
                "ratio": ratio,
                "gold": gold,
                "prediction": gold,
                "correct": True,
                "choice_logits": {label: float(label == gold) for label in "ABCD"},
                "context_tokens": 10,
                "query_tokens": 2,
                "budget_tokens": budget,
                "retained_positions": list(range(budget)),
                "scoring_cache_tokens": budget + 2,
                "budget_audit": "passed",
                "retained_kv_bytes": 16 * budget,
                "scoring_kv_bytes": 16 * (budget + 2),
                "seconds": 1.0,
            }
        )
    (tmp_path / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    write_json(
        tmp_path / "run.json",
        {
            "dataset_sha256": hashlib.sha256((tmp_path / "dataset.json").read_bytes()).hexdigest(),
            "ratios": [0.5, 0.25],
            "expected_evaluations": 9,
        },
    )
    write_json(
        tmp_path / "summary.json",
        {"status": "complete", "completed": 9, "expected": 9, "groups": summarize(rows)},
    )
    return tmp_path


def test_verifier_accepts_consistent_evidence(evidence):
    assert verify(evidence)["status"] == "passed"


@pytest.mark.parametrize("corruption", ["duplicate", "prediction", "budget", "hash"])
def test_verifier_rejects_corruption(evidence, corruption):
    path = evidence / "rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if corruption == "duplicate":
        rows[-1] = rows[-2]
    elif corruption == "prediction":
        rows[0]["prediction"] = "D"
    elif corruption == "budget":
        rows[1]["scoring_cache_tokens"] += 1
    else:
        with (evidence / "dataset.json").open("a") as handle:
            handle.write("\n")
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(AssertionError):
        verify(evidence)
