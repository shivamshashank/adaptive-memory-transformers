"""Direct-color construction, decision gate and evidence corruption tests."""

import hashlib
import json
import runpy
from collections import Counter
from pathlib import Path

import pytest

from amt.validation import write_json

calibration = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/calibrate_direct_color.py")
)
COLORS = calibration["COLORS"]


def test_dataset_is_paired_and_semantically_valid():
    data = calibration["make_calibration"]()
    calibration["validate_data"](data)
    assert len(data) == len({example["id"] for example in data}) == 64
    assert Counter(example["vocabulary"] for example in data) == {
        "original": 32,
        "renamed": 32,
    }
    assert Counter((example["base_id"], example["vocabulary"]) for example in data) == {
        (base_id, vocabulary): 1
        for base_id in {example["base_id"] for example in data}
        for vocabulary in ("original", "renamed")
    }
    assert all("A:" not in example["query"] for example in data)
    data[0]["gold"] = "purple"
    with pytest.raises(ValueError):
        calibration["validate_data"](data)


def test_choose_is_deterministic_and_validated():
    assert calibration["choose"](
        {color: float(index) for index, color in enumerate(COLORS)}, "yellow"
    ) == {
        "prediction": "yellow",
        "correct": True,
    }
    tied = dict.fromkeys(COLORS, 1.0)
    assert calibration["choose"](tied, "red")["prediction"] == "red"
    with pytest.raises(ValueError):
        calibration["choose"]({"red": 1.0}, "red")
    with pytest.raises(ValueError):
        calibration["choose"](tied | {"red": float("nan")}, "red")


@pytest.fixture
def evidence(tmp_path):
    data = calibration["make_calibration"]()
    write_json(tmp_path / "dataset.json", data)
    write_json(
        tmp_path / "run.json",
        {
            "dataset_sha256": hashlib.sha256((tmp_path / "dataset.json").read_bytes()).hexdigest(),
            "colors": list(COLORS),
        },
    )
    write_json(
        tmp_path / "inputs.json",
        {
            "examples": [
                {"id": example["id"], "context_ids": [1, 2], "query_ids": [3]} for example in data
            ],
            "bare_color_ids": {color: [index] for index, color in enumerate(COLORS)},
            "space_color_ids": {color: [index + 10] for index, color in enumerate(COLORS)},
        },
    )
    rows = []
    for example in data:
        logits = {color: float(color == example["gold"]) for color in COLORS}
        rows.append(
            {
                "example_id": example["id"],
                **{
                    key: example[key]
                    for key in ("base_id", "vocabulary", "gold", "target_fact_index")
                },
                **calibration["choose"](logits, example["gold"]),
                "space_prediction": example["gold"],
                "color_logits": logits,
                "space_color_logits": logits,
                "dense_color_logits": logits,
                "dense_space_color_logits": logits,
                "raw_greedy_text": example["gold"],
                "dense_check": "passed",
                "dense_max_abs_difference": 0.0,
                "context_tokens": 2,
                "scoring_cache_tokens": 3,
            }
        )
    (tmp_path / "rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    calibration["save"](tmp_path, rows, "complete")
    return tmp_path, rows


def test_complete_gate_and_pairing(evidence):
    path, rows = evidence
    assert calibration["verify"](path)["status"] == "passed"
    summary = calibration["summarize"](rows, "complete")
    assert summary["task_gate"] == "passed"
    assert len(summary["renamed_pairs"]) == 32
    assert all(pair["same_prediction"] and pair["same_raw"] for pair in summary["renamed_pairs"])
    assert calibration["summarize"](rows, "running")["task_gate"] == "pending"


def test_gate_requires_raw_validity_and_each_vocabulary(evidence):
    _, rows = evidence
    original = [row for row in rows if row["vocabulary"] == "original"]
    for row in original[:4]:
        row["correct"] = False
    assert calibration["summarize"](rows, "complete")["task_gate"] == "failed"
    for row in original[:4]:
        row["correct"] = True
        row["raw_greedy_text"] = "The"
    assert calibration["summarize"](rows, "complete")["task_gate"] == "failed"


@pytest.mark.parametrize(
    "field,value",
    [
        ("prediction", "purple"),
        ("space_prediction", "purple"),
        ("scoring_cache_tokens", 99),
        ("dense_check", "failed"),
        ("dense_color_logits", dict.fromkeys(COLORS, 100.0)),
    ],
)
def test_verifier_rejects_corruption(evidence, field, value):
    path, rows = evidence
    rows[0][field] = value
    (path / "rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(AssertionError):
        calibration["verify"](path)
