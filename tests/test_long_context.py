"""Long-context construction, gates, target spans and evidence checks."""

import hashlib
import json
import runpy
from collections import Counter
from pathlib import Path

import pytest

from amt.validation import write_json

experiment = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/run_long_context.py")
)
COLORS = experiment["COLORS"]


def test_dataset_is_balanced_and_semantically_valid():
    data = experiment["make_dataset"]()
    experiment["validate_data"](data)
    assert len(data) == len({example["id"] for example in data}) == 16
    assert Counter(example["fact_count"] for example in data) == {16: 8, 32: 8}
    assert Counter((example["fact_count"], example["gold"]) for example in data) == {
        (facts, color): 2 for facts in (16, 32) for color in COLORS
    }
    assert experiment["target_positions"](16) == (0, 2, 4, 6, 9, 11, 13, 15)
    assert experiment["target_positions"](32) == (0, 4, 9, 13, 18, 22, 27, 31)
    data[0]["gold"] = "purple"
    with pytest.raises(ValueError):
        experiment["validate_data"](data)


def test_target_span_uses_offsets_and_rejects_disagreement():
    text = "prefix The Unit00 locker is red. suffix"

    class CharacterTokenizer:
        def __call__(self, value, **kwargs):
            assert value == text
            assert kwargs == {"add_special_tokens": False, "return_offsets_mapping": True}
            return {
                "input_ids": list(range(len(text))),
                "offset_mapping": [(i, i + 1) for i in range(len(text))],
            }

    example = {"target_name": "Unit00", "gold": "red"}
    tokenized = {"context_prefix": text, "context_ids": list(range(len(text)))}
    span = experiment["target_span"](CharacterTokenizer(), tokenized, example)
    start = text.index("The Unit00 locker is red.")
    assert span == list(range(start, start + len("The Unit00 locker is red.")))
    tokenized["context_ids"] = [999]
    with pytest.raises(ValueError):
        experiment["target_span"](CharacterTokenizer(), tokenized, example)


@pytest.fixture
def evidence(tmp_path):
    data = experiment["make_dataset"]()
    write_json(tmp_path / "dataset.json", data)
    write_json(
        tmp_path / "run.json",
        {"dataset_sha256": hashlib.sha256((tmp_path / "dataset.json").read_bytes()).hexdigest()},
    )
    write_json(
        tmp_path / "inputs.json",
        {
            "examples": [
                {
                    "id": example["id"],
                    "context_ids": list(range(10)),
                    "query_ids": [10, 11],
                    "target_span": [2, 3],
                }
                for example in data
            ]
        },
    )
    rows = []
    for example in data:
        logits = {color: float(color == example["gold"]) for color in COLORS}
        for policy, retained in (
            ("full", list(range(10))),
            ("recency+first", [0, 6, 7, 8, 9]),
            ("adaptive+first", [0, 2, 3, 8, 9]),
        ):
            full = policy == "full"
            target_kept = sum(position in retained for position in (2, 3))
            rows.append(
                {
                    "example_id": example["id"],
                    "policy": policy,
                    "first_token_protected": not full,
                    **{
                        key: example[key]
                        for key in ("gold", "fact_count", "target_fact_index", "target_depth")
                    },
                    **experiment["choose"](logits, example["gold"]),
                    "raw_greedy_text": example["gold"],
                    "color_logits": logits,
                    "budget_tokens": len(retained),
                    "retained_positions": retained,
                    "target_retained_tokens": target_kept,
                    "target_retained_fraction": target_kept / 2,
                    "retained_kv_bytes": 100 * len(retained),
                    "scoring_cache_tokens": len(retained) + 2,
                    "scoring_kv_bytes": 100 * (len(retained) + 2),
                    "dense_check": "passed" if full else None,
                    "dense_max_abs_difference": 0.0 if full else None,
                    "dense_color_logits": logits if full else None,
                }
            )
    (tmp_path / "rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    experiment["save"](tmp_path, rows, "complete")
    return tmp_path, rows


def test_verifier_and_complete_decisions(evidence):
    path, rows = evidence
    assert experiment["verify"](path)["status"] == "passed"
    summary = experiment["summarize"](rows, "complete")
    assert summary["baseline_gate"] == "passed"
    assert summary["adaptive_larger_run_decision"] == "diagnose"
    assert len(summary["paired_adaptive_minus_recency"]) == 16
    for row in [row for row in rows if row["policy"] == "recency+first"][:2]:
        row["correct"] = False
    assert experiment["summarize"](rows, "complete")["adaptive_larger_run_decision"] == "proceed"


def test_baseline_gate_requires_each_length(evidence):
    _, rows = evidence
    full_16 = [row for row in rows if row["policy"] == "full" and row["fact_count"] == 16]
    for row in full_16[:2]:
        row["correct"] = False
    assert experiment["summarize"](rows, "complete")["baseline_gate"] == "failed"
    assert experiment["summarize"](rows, "running")["baseline_gate"] == "pending"


@pytest.mark.parametrize(
    "field,value",
    [
        ("prediction", "purple"),
        ("budget_tokens", 99),
        ("retained_positions", [0, 1, 1, 2, 3]),
        ("target_retained_tokens", 99),
        ("scoring_kv_bytes", 999),
        ("dense_color_logits", dict.fromkeys(COLORS, 100.0)),
    ],
)
def test_verifier_rejects_corruption(evidence, field, value):
    path, rows = evidence
    rows[0][field] = value
    (path / "rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(AssertionError):
        experiment["verify"](path)
