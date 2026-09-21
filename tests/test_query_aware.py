"""Query-probe boundaries, decision rule and saved-evidence checks."""

import hashlib
import json
import runpy
from pathlib import Path

import pytest

from amt.validation import write_json

experiment = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/run_query_aware.py"))
COLORS = experiment["COLORS"]
POLICIES = experiment["POLICIES"]


def test_frozen_dataset_hash_matches_prior_experiment():
    data = experiment["make_dataset"]()
    experiment["validate_data"](data)
    encoded = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
    assert hashlib.sha256(encoded).hexdigest() == experiment["DATASET_SHA256"]


def test_question_probe_is_exact_prefix_and_excludes_instruction():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "<user>" + messages[0]["content"] + "</user><assistant>"

        def encode(self, text, **kwargs):
            return list(text.encode())

    example = experiment["make_dataset"]()[0]
    result = experiment["tokenize_with_probe"](Tokenizer(), example)
    probe = bytes(result["question_probe_ids"]).decode()
    assert probe == f"What color is the {example['target_name']} locker?\n"
    assert "Answer with" not in probe
    assert result["query_ids"][: len(result["question_probe_ids"])] == result["question_probe_ids"]


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
                    "question_probe_ids": [10],
                    "target_span": [2, 3],
                }
                for example in data
            ]
        },
    )
    rows = []
    for example in data:
        scores = {color: float(color == example["gold"]) for color in COLORS}
        for policy in POLICIES:
            full = policy == "full"
            retained = list(range(10)) if full else [0, 2, 3, 8, 9]
            target_kept = sum(position in retained for position in (2, 3))
            query_aware = policy == "adaptive-query+first"
            rows.append(
                {
                    "example_id": example["id"],
                    "policy": policy,
                    "first_token_protected": not full,
                    **{
                        key: example[key]
                        for key in ("gold", "fact_count", "target_fact_index", "target_depth")
                    },
                    **experiment["choose"](scores, example["gold"]),
                    "raw_greedy_text": example["gold"],
                    "color_logits": scores,
                    "budget_tokens": len(retained),
                    "retained_positions": retained,
                    "target_retained_tokens": target_kept,
                    "target_retained_fraction": target_kept / 2,
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
    experiment["save"](tmp_path, rows, "complete")
    return tmp_path, rows


def test_verifier_and_query_aware_decision(evidence):
    path, rows = evidence
    assert experiment["verify"](path)["status"] == "passed"
    summary = experiment["summarize"](rows, "complete")
    assert summary["baseline_gate"] == "passed"
    assert summary["query_aware_decision"] == "diagnose"
    assert len(summary["paired_changes"]) == 16
    for policy in ("recency+first", "adaptive+first"):
        for row in [row for row in rows if row["policy"] == policy][:2]:
            row["correct"] = False
    assert experiment["summarize"](rows, "complete")["query_aware_decision"] == "proceed"


def test_decision_requires_each_length(evidence):
    _, rows = evidence
    for policy in ("recency+first", "adaptive+first"):
        candidates = [row for row in rows if row["policy"] == policy and row["fact_count"] == 16]
        for row in candidates[:2]:
            row["correct"] = False
    query_32 = [
        row for row in rows if row["policy"] == "adaptive-query+first" and row["fact_count"] == 32
    ]
    query_32[0]["correct"] = False
    assert experiment["summarize"](rows, "complete")["query_aware_decision"] == "diagnose"


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


def test_verifier_rejects_missing_probe(evidence):
    path, rows = evidence
    query = next(row for row in rows if row["policy"] == "adaptive-query+first")
    query["probe_seconds"] = 0
    (path / "rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(AssertionError):
        experiment["verify"](path)
