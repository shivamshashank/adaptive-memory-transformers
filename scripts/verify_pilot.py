"""Independently check saved pilot evidence. Does not rerun model inference."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def verify(path: Path) -> dict:
    settings = json.loads((path / "run.json").read_text())
    data = json.loads((path / "dataset.json").read_text())
    inputs = json.loads((path / "inputs.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    rows = [json.loads(line) for line in (path / "rows.jsonl").read_text().splitlines()]
    assert summary["status"] == "complete", "Run is not complete"
    assert (
        hashlib.sha256((path / "dataset.json").read_bytes()).hexdigest()
        == settings["dataset_sha256"]
    )
    examples = {x["id"]: x for x in data}
    tokenized = {x["id"]: x for x in inputs["examples"]}
    assert len(examples) == len(data) == len(tokenized)
    assert set(examples) == set(tokenized)
    for example in data:
        name = example["query"].split(" locker?")[0].removeprefix("What color is the ")
        fact = next(
            line
            for line in example["context"].splitlines()
            if line.startswith(f"The {name} locker is ")
        )
        color = fact.removeprefix(f"The {name} locker is ").removesuffix(".")
        assert example["options"]["ABCD".index(example["gold"])] == color
    if settings.get("dataset_version") == "v2":
        assert len(data) % 32 == 0
        assert Counter((x["target_fact_index"], x["gold"]) for x in data) == {
            (position, label): len(data) // 32 for position in range(8) for label in "ABCD"
        }
    expected = {(key, "full", None) for key in examples}
    expected |= {
        (key, policy, ratio)
        for key in examples
        for ratio in settings["ratios"]
        for policy in ("recency", "uniform", "attention", "adaptive")
    }
    actual = [(r["example_id"], r["policy"], r["ratio"]) for r in rows]
    assert len(actual) == len(set(actual)) and set(actual) == expected
    assert (
        len(rows) == summary["completed"] == summary["expected"] == settings["expected_evaluations"]
    )
    groups = defaultdict(list)
    full = {r["example_id"]: r for r in rows if r["policy"] == "full"}
    for row in rows:
        key = row["example_id"]
        assert row["gold"] == examples[key]["gold"]
        assert all(math.isfinite(value) for value in row["choice_logits"].values())
        assert row["prediction"] == max("ABCD", key=row["choice_logits"].__getitem__)
        assert row["correct"] == (row["prediction"] == row["gold"])
        n = len(tokenized[key]["context_ids"])
        q = len(tokenized[key]["query_ids"])
        budget = n if row["ratio"] is None else max(1, math.floor(n * row["ratio"]))
        assert row["context_tokens"] == n and row["query_tokens"] == q
        assert row["budget_tokens"] == len(row["retained_positions"]) == budget
        assert row["retained_positions"] == sorted(set(row["retained_positions"]))
        assert all(0 <= position < n for position in row["retained_positions"])
        assert row["scoring_cache_tokens"] == budget + q
        assert row["budget_audit"] == "passed"
        bytes_per_token = full[key]["retained_kv_bytes"] / n
        assert row["retained_kv_bytes"] == bytes_per_token * budget
        assert row["scoring_kv_bytes"] == bytes_per_token * (budget + q)
        groups[(row["policy"], row["ratio"])].append(row)
    assert len(summary["groups"]) == len(groups)
    assert len({(x["policy"], x["ratio"]) for x in summary["groups"]}) == len(groups)
    for group in summary["groups"]:
        selected = groups[(group["policy"], group["ratio"])]
        assert group["n"] == len(selected) == len(data)
        correct = sum(r["correct"] for r in selected)
        assert group["correct"] == correct and group["accuracy"] == correct / len(data)
        subset = [r for r in selected if full[r["example_id"]]["correct"]]
        assert group["full_correct_n"] == len(subset)
        assert group["accuracy_on_full_correct"] == (
            sum(r["correct"] for r in subset) / len(subset) if subset else None
        )
        assert group["mean_retained_kv_bytes"] == sum(
            r["retained_kv_bytes"] for r in selected
        ) / len(selected)
        assert math.isclose(group["total_seconds"], sum(r["seconds"] for r in selected))
    return {
        "status": "passed",
        "rows": len(rows),
        "examples": len(data),
        "groups": len(groups),
        "checks": [
            "dataset hash",
            "gold parsed from facts",
            (
                "v2 position/label balance"
                if settings.get("dataset_version") == "v2"
                else "v1: position/label balance not asserted"
            ),
            "all expected unique cells",
            "predictions",
            "accuracy and subset aggregates",
            "retained positions",
            "budget and KV byte arithmetic",
            "timing aggregates",
        ],
        "scope": "Saved evidence consistency, not independent proof of model outputs or peak memory",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    # Do not run with python -O: verification deliberately uses assertions.
    if not __debug__:
        raise RuntimeError("Run verification without Python optimization (-O)")
    print(json.dumps(verify(args.run), indent=2))
