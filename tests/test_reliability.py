"""Control construction and saved-evidence checks require no pretrained model."""

import hashlib
import json
import runpy
from collections import Counter
from pathlib import Path

import pytest

from amt.evaluation import score_choices
from amt.validation import write_json

diagnostic = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/check_retrieval_reliability.py")
)


def test_controls_balanced_and_semantics_preserved():
    data = diagnostic["make_controls"]()
    diagnostic["validate_data"](data)
    assert len(data) == len({e["id"] for e in data}) == 64
    assert Counter((e["target_fact_index"], e["vocabulary"], e["gold"]) for e in data) == {
        (p, v, c): 1 for p in range(8) for v in ("original", "renamed") for c in "ABCD"
    }
    for position in range(8):
        group = [e for e in data if e["target_fact_index"] == position]
        assert len({e["base_id"] for e in group}) == 1
        assert len({e["target_color"] for e in group}) == 1
    data[0]["gold"] = "Z"
    with pytest.raises(ValueError):
        diagnostic["validate_data"](data)


@pytest.fixture
def evidence(tmp_path):
    data = diagnostic["make_controls"]()
    write_json(tmp_path / "dataset.json", data)
    write_json(
        tmp_path / "run.json",
        {"dataset_sha256": hashlib.sha256((tmp_path / "dataset.json").read_bytes()).hexdigest()},
    )
    write_json(
        tmp_path / "inputs.json",
        {"examples": [{"id": e["id"], "context_ids": [1, 2], "query_ids": [3]} for e in data]},
    )
    rows = []
    for e in data:
        scores = [float(c == e["gold"]) for c in "ABCD"]
        rows.append(
            {
                "example_id": e["id"],
                **{k: e[k] for k in ("gold", "vocabulary", "rotation", "target_fact_index")},
                **score_choices(scores, e["gold"]),
                "raw_greedy_text": e["gold"],
                "predicted_color": e["target_color"],
                "dense_check": "passed",
                "dense_choice_logits": dict(zip("ABCD", scores)),
                "context_tokens": 2,
                "scoring_cache_tokens": 3,
            }
        )
    (tmp_path / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    diagnostic["save"](tmp_path, rows, "complete")
    return tmp_path, rows


def test_gate_and_pairs(evidence):
    path, rows = evidence
    assert diagnostic["verify"](path)["status"] == "passed"
    summary = diagnostic["summarize"](rows, "complete")
    assert summary["longer_context_gate"] == "passed"
    assert len(summary["renamed_pairs"]) == 32
    assert all(g["unique_predicted_colors"] == 1 for g in summary["rotation_groups"])
    assert diagnostic["summarize"](rows, "running")["longer_context_gate"] == "pending"
    for r in rows[:4]:
        r["raw_greedy_text"] = "The"
    assert diagnostic["summarize"](rows, "complete")["longer_context_gate"] == "failed"


@pytest.mark.parametrize(
    "field,value",
    [
        ("prediction", "Z"),
        ("scoring_cache_tokens", 99),
        ("vocabulary", "other"),
        ("predicted_color", "purple"),
        ("dense_choice_logits", dict.fromkeys("ABCD", 1000.0)),
    ],
)
def test_verifier_rejects_corruption(evidence, field, value):
    path, rows = evidence
    rows[0][field] = value
    (path / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(AssertionError):
        diagnostic["verify"](path)


def test_gate_requires_each_vocabulary(evidence):
    _, rows = evidence
    original = [r for r in rows if r["vocabulary"] == "original"]
    for r in original[:4]:
        r["correct"] = False
    # 60/64 globally is insufficient if one vocabulary only achieves 28/32.
    assert diagnostic["summarize"](rows, "complete")["longer_context_gate"] == "failed"
