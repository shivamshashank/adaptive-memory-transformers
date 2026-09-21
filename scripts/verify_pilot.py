"""Independently check saved pilot evidence. Does not rerun model inference."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def verify_prefix_pairs(rows, pairs):
    by_key = {(r["example_id"], r["policy"], r["ratio"]): r for r in rows}
    expected_pairs = {
        (r["example_id"], r["policy"].removesuffix("+first"), r["ratio"])
        for r in rows
        if r["policy"].endswith("+first")
    }
    assert len(pairs) == len(expected_pairs)
    assert {(p["example_id"], p["policy"], p["ratio"]) for p in pairs} == expected_pairs
    for pair in pairs:
        off = by_key[(pair["example_id"], pair["policy"], pair["ratio"])]
        on = by_key[(pair["example_id"], pair["policy"] + "+first", pair["ratio"])]
        expected = off["retained_positions"].copy()
        if 0 not in expected:
            expected.remove(min(expected))
            expected.append(0)
        assert on["retained_positions"] == sorted(expected)
        unchanged = on["retained_positions"] == off["retained_positions"]
        assert pair["selection_unchanged"] == unchanged
        if unchanged:
            assert on["raw_greedy_id"] == off["raw_greedy_id"]
            assert on["prediction"] == off["prediction"]
            assert all(
                abs(on["choice_logits"][c] - off["choice_logits"][c])
                <= 2e-4 + 1e-5 * abs(off["choice_logits"][c])
                for c in "ABCD"
            )
        valid_off = off["raw_greedy_text"].strip() in ("A", "B", "C", "D")
        valid_on = on["raw_greedy_text"].strip() in ("A", "B", "C", "D")
        assert pair["validity_delta"] == int(valid_on) - int(valid_off)
        assert pair["accuracy_delta"] == int(on["correct"]) - int(off["correct"])


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
    policies = ("recency", "uniform", "attention", "adaptive")
    if settings.get("prefix_ablation"):
        assert settings["ratios"] == [0.5] and len(data) == 32
        assert settings["seed"] == 20260920 and settings["dataset_version"] == "v2"
        assert settings["protection_rule"] == "swap-oldest-selected-for-position-zero-v1"
        policies += tuple(name + "+first" for name in policies)
    expected |= {
        (key, policy, ratio)
        for key in examples
        for ratio in settings["ratios"]
        for policy in policies
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
        if settings.get("prefix_ablation"):
            assert n == 82 and (row["ratio"] is None or budget == 41)
            assert row["first_token_protected"] == row["policy"].endswith("+first")
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
        if "valid_next_tokens" in group:
            assert group["valid_next_tokens"] == sum(
                r.get("raw_greedy_text", "").strip() in ("A", "B", "C", "D") for r in selected
            )
            assert group["raw_next_correct"] == sum(
                r.get("raw_greedy_text", "").strip() == r["gold"] for r in selected
            )
        subset = [r for r in selected if full[r["example_id"]]["correct"]]
        assert group["full_correct_n"] == len(subset)
        assert group["accuracy_on_full_correct"] == (
            sum(r["correct"] for r in subset) / len(subset) if subset else None
        )
        assert group["mean_retained_kv_bytes"] == sum(
            r["retained_kv_bytes"] for r in selected
        ) / len(selected)
        assert math.isclose(group["total_seconds"], sum(r["seconds"] for r in selected))
    if settings.get("prefix_ablation"):
        verify_prefix_pairs(rows, json.loads((path / "paired_changes.json").read_text()))
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
