"""Frozen CPU-only longer-context matched-budget development experiment."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from amt.evaluation import audit_cache, tokenize_example
from amt.policies import FirstTokenProtectedPolicy, RetentionSignals, create_policy
from amt.validation import snapshot_sources, write_json

COLORS = ("red", "blue", "green", "yellow")
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SEED = 20260921
DATASET_SHA256 = "2a88cf55ef59b7909af5de5f2a32725b52f94074e0a2c3e24fa20e34664d7a62"


def target_positions(facts):
    return tuple(round(slot * (facts - 1) / 7) for slot in range(8))


def make_dataset():
    rng = random.Random(SEED)
    result = []
    for facts in (16, 32):
        for slot, target in enumerate(target_positions(facts)):
            values = [rng.choice(COLORS) for _ in range(facts)]
            values[target] = COLORS[slot % len(COLORS)]
            lines = [
                f"The Unit{index:02d} locker is {color}." for index, color in enumerate(values)
            ]
            context = "Remember these locker colors.\n" + "\n".join(lines) + "\nEnd of records.\n"
            query = (
                f"What color is the Unit{target:02d} locker?\n"
                "Answer with only the color word (red, blue, green, or yellow).\n"
                "Answer:"
            )
            result.append(
                {
                    "id": f"long-{facts}-{slot}",
                    "split": "development",
                    "fact_count": facts,
                    "target_fact_index": target,
                    "target_depth": target / (facts - 1),
                    "target_name": f"Unit{target:02d}",
                    "gold": values[target],
                    "context": context,
                    "query": query,
                }
            )
    return result


def validate_data(data):
    if data != make_dataset():
        raise ValueError("Dataset differs from frozen longer-context construction")
    for example in data:
        prefix = f"The {example['target_name']} locker is "
        facts = [line for line in example["context"].splitlines() if line.startswith(prefix)]
        if len(facts) != 1 or facts[0].removeprefix(prefix).removesuffix(".") != example["gold"]:
            raise ValueError("Gold does not match the uniquely requested fact")
        if (
            len([line for line in example["context"].splitlines() if line.startswith("The Unit")])
            != example["fact_count"]
        ):
            raise ValueError("Declared fact count disagrees with context")


def choose(scores, gold):
    if (
        set(scores) != set(COLORS)
        or gold not in COLORS
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in scores.values()
        )
    ):
        raise ValueError("Four finite color scores and valid gold required")
    prediction = max(COLORS, key=scores.__getitem__)
    return {"prediction": prediction, "correct": prediction == gold}


def target_span(tokenizer, tokenized, example):
    context_text = tokenized["context_prefix"]
    fact = f"The {example['target_name']} locker is {example['gold']}."
    start = context_text.index(fact)
    end = start + len(fact)
    encoded = tokenizer(context_text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != tokenized["context_ids"]:
        raise ValueError("Offset tokenizer disagrees with frozen context token IDs")
    span = [
        index
        for index, (left, right) in enumerate(encoded["offset_mapping"])
        if right > start and left < end
    ]
    if not span:
        raise ValueError("Target fact has no token span")
    return span


def metric(rows):
    return {
        "n": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "raw_valid": sum(row["raw_greedy_text"].strip() in COLORS for row in rows),
        "raw_correct": sum(row["raw_greedy_text"].strip() == row["gold"] for row in rows),
        "mean_target_retained_fraction": (
            sum(row["target_retained_fraction"] for row in rows) / len(rows) if rows else None
        ),
        "mean_retained_kv_bytes": (
            sum(row["retained_kv_bytes"] for row in rows) / len(rows) if rows else None
        ),
    }


def summarize(rows, status):
    full = {row["example_id"]: row for row in rows if row["policy"] == "full"}
    policies = ("full", "recency+first", "adaptive+first")
    groups = []
    for policy in policies:
        for facts in (16, 32):
            group = [row for row in rows if row["policy"] == policy and row["fact_count"] == facts]
            if group:
                item = {"policy": policy, "fact_count": facts, **metric(group)}
                subset = [row for row in group if full.get(row["example_id"], {}).get("correct")]
                item["full_correct_n"] = len(subset)
                item["correct_on_full_correct"] = sum(row["correct"] for row in subset)
                groups.append(item)
    full_rows = [row for row in rows if row["policy"] == "full"]
    baseline_passed = (
        status == "complete"
        and len(full_rows) == 16
        and metric(full_rows)["correct"] >= 14
        and metric(full_rows)["raw_correct"] >= 14
        and metric(full_rows)["raw_valid"] == 16
        and all(
            sum(row["correct"] for row in full_rows if row["fact_count"] == facts) >= 7
            and sum(
                row["raw_greedy_text"].strip() == row["gold"]
                for row in full_rows
                if row["fact_count"] == facts
            )
            >= 7
            for facts in (16, 32)
        )
        and all(row["dense_check"] == "passed" for row in full_rows)
    )
    pairs = []
    lookup = {(row["example_id"], row["policy"]): row for row in rows}
    for example_id in sorted(full):
        recency = lookup.get((example_id, "recency+first"))
        adaptive = lookup.get((example_id, "adaptive+first"))
        if recency is not None and adaptive is not None:
            pairs.append(
                {
                    "example_id": example_id,
                    "fact_count": recency["fact_count"],
                    "full_correct": full[example_id]["correct"],
                    "accuracy_delta": int(adaptive["correct"]) - int(recency["correct"]),
                    "target_retention_delta": adaptive["target_retained_fraction"]
                    - recency["target_retained_fraction"],
                }
            )
    recency_correct = sum(lookup[key]["correct"] for key in lookup if key[1] == "recency+first")
    adaptive_correct = sum(lookup[key]["correct"] for key in lookup if key[1] == "adaptive+first")
    promising = (
        baseline_passed
        and len(pairs) == 16
        and adaptive_correct - recency_correct >= 2
        and all(
            sum(
                row["correct"]
                for row in rows
                if row["policy"] == "adaptive+first" and row["fact_count"] == facts
            )
            >= sum(
                row["correct"]
                for row in rows
                if row["policy"] == "recency+first" and row["fact_count"] == facts
            )
            for facts in (16, 32)
        )
    )
    return {
        "status": status,
        "completed": len(rows),
        "expected": 48,
        "groups": groups,
        "paired_adaptive_minus_recency": pairs,
        "baseline_gate": "passed"
        if baseline_passed
        else "failed"
        if status == "complete"
        else "pending",
        "adaptive_larger_run_decision": "proceed"
        if promising
        else "diagnose"
        if status == "complete"
        else "pending",
    }


def save(output, rows, status):
    summary = summarize(rows, status)
    write_json(output / "summary.json", summary)
    lines = [
        "# Modest longer-context matched-budget experiment",
        "",
        f"Status: {status}; {len(rows)}/48 evaluations.",
        "",
        "| Policy | Facts | Correct | Raw correct | Mean target fact retained | Mean retained KV bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in summary["groups"]:
        lines.append(
            f"| {group['policy']} | {group['fact_count']} | {group['correct']}/{group['n']} | "
            f"{group['raw_correct']}/{group['n']} | {group['mean_target_retained_fraction']:.1%} | "
            f"{group['mean_retained_kv_bytes']:.0f} |"
        )
    lines += [
        "",
        f"Baseline gate: **{summary['baseline_gate']}**.",
        f"Adaptive larger-run decision: **{summary['adaptive_larger_run_decision']}**.",
        "",
        "Development data only. Accuracy is paired; timing is not a speed benchmark.",
        "The 50% budget covers context tokens before the unpruned query is appended.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify(output):
    data = json.loads((output / "dataset.json").read_text())
    validate_data(data)
    settings = json.loads((output / "run.json").read_text())
    assert (
        settings["dataset_sha256"]
        == hashlib.sha256((output / "dataset.json").read_bytes()).hexdigest()
    )
    inputs = json.loads((output / "inputs.json").read_text())
    examples = {example["id"]: example for example in data}
    tokens = {item["id"]: item for item in inputs["examples"]}
    rows = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
    expected = {
        (key, policy) for key in examples for policy in ("full", "recency+first", "adaptive+first")
    }
    assert len(rows) == 48
    assert {(row["example_id"], row["policy"]) for row in rows} == expected
    full = {row["example_id"]: row for row in rows if row["policy"] == "full"}
    for row in rows:
        example = examples[row["example_id"]]
        for field in ("gold", "fact_count", "target_fact_index", "target_depth"):
            assert row[field] == example[field]
        assert all(
            row[key] == value for key, value in choose(row["color_logits"], row["gold"]).items()
        )
        tokenized = tokens[row["example_id"]]
        context = len(tokenized["context_ids"])
        budget = context if row["policy"] == "full" else math.floor(context * 0.5)
        assert row["budget_tokens"] == len(row["retained_positions"]) == budget
        assert row["retained_positions"] == sorted(set(row["retained_positions"]))
        assert all(0 <= position < context for position in row["retained_positions"])
        if row["policy"] != "full":
            assert 0 in row["retained_positions"]
        assert row["scoring_cache_tokens"] == budget + len(tokenized["query_ids"])
        bytes_per_token = full[row["example_id"]]["retained_kv_bytes"] / context
        assert row["retained_kv_bytes"] == bytes_per_token * budget
        assert row["scoring_kv_bytes"] == bytes_per_token * row["scoring_cache_tokens"]
        expected_target = sum(
            position in row["retained_positions"] for position in tokenized["target_span"]
        )
        assert row["target_retained_tokens"] == expected_target
        assert row["target_retained_fraction"] == expected_target / len(tokenized["target_span"])
        if row["policy"] == "full":
            assert not row["first_token_protected"]
            assert row["dense_check"] == "passed"
            assert math.isfinite(row["dense_max_abs_difference"])
            for color in COLORS:
                assert abs(
                    row["color_logits"][color] - row["dense_color_logits"][color]
                ) <= 2e-4 + 1e-5 * abs(row["dense_color_logits"][color])
        else:
            assert row["dense_check"] is None and row["first_token_protected"]
    assert json.loads((output / "summary.json").read_text()) == summarize(rows, "complete")
    return {
        "status": "passed",
        "rows": len(rows),
        "scope": "Saved consistency, not an independent model rerun",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/long-context"))
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("Do not use Python -O for evidence verification")
    if args.verify:
        print(json.dumps(verify(args.verify), indent=2))
        return 0
    root = Path(__file__).resolve().parents[1]
    os.environ.update(
        {
            "HF_HOME": str(root / ".model-cache/huggingface"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    output = args.output_dir.resolve() / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    output.mkdir(parents=True, exist_ok=False)
    data = make_dataset()
    validate_data(data)
    write_json(output / "dataset.json", data)
    dataset_hash = hashlib.sha256((output / "dataset.json").read_bytes()).hexdigest()
    if dataset_hash != DATASET_SHA256:
        raise ValueError("Frozen longer-context dataset hash changed")
    snapshot_sources(root, output)
    write_json(
        output / "run.json",
        {
            "model": MODEL,
            "revision": REVISION,
            "seed": SEED,
            "device": "cpu",
            "dtype": "float32",
            "attention": "eager",
            "python": sys.version,
            "platform": platform.platform(),
            "versions": {
                name: importlib.metadata.version(name) for name in ("torch", "transformers")
            },
            "protocol": "research/long_context_protocol.md",
            "ratio": 0.5,
            "expected": 48,
            "policies": ["full", "recency+first", "adaptive+first"],
            "dataset_sha256": dataset_hash,
        },
    )
    rows = []
    save(output, rows, "running")
    print(f"Long-context report: {output / 'report.md'}", flush=True)
    with (
        (output / "run.log").open("w", buffering=1) as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        try:
            import torch
            from transformers import AutoTokenizer, Qwen2ForCausalLM

            from amt.cache import Qwen2CacheAdapter
            from amt.policies import policy_requires_attention

            torch.manual_seed(SEED)
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL, revision=REVISION, local_files_only=True
            )
            model = Qwen2ForCausalLM.from_pretrained(
                MODEL,
                revision=REVISION,
                local_files_only=True,
                dtype=torch.float32,
                attn_implementation="eager",
            ).eval()
            bare = {color: tokenizer.encode(color, add_special_tokens=False) for color in COLORS}
            spaced = {
                color: tokenizer.encode(" " + color, add_special_tokens=False) for color in COLORS
            }
            if any(len(ids) != 1 for ids in (*bare.values(), *spaced.values())):
                raise ValueError("Frozen scorer requires single-token colors")
            tokenized = [tokenize_example(tokenizer, example, "v2") for example in data]
            for example, tokens in zip(data, tokenized):
                tokens["target_span"] = target_span(tokenizer, tokens, example)
                expected_context = 161 if example["fact_count"] == 16 else 289
                if (
                    len(tokens["context_ids"]) != expected_context
                    or len(tokens["query_ids"]) != 32
                    or len(tokens["target_span"]) != 8
                ):
                    raise ValueError("Frozen token lengths or target span changed")
            write_json(
                output / "inputs.json",
                {"examples": tokenized, "bare_color_ids": bare, "space_color_ids": spaced},
            )
            bare_ids = [bare[color][0] for color in COLORS]
            space_ids = [spaced[color][0] for color in COLORS]
            with (
                torch.inference_mode(),
                (output / "rows.jsonl").open("w", buffering=1, encoding="utf-8") as handle,
            ):
                for example, tokens in zip(data, tokenized):
                    for name in ("full", "recency", "adaptive"):
                        started = time.perf_counter()
                        adapter = Qwen2CacheAdapter(model)
                        policy = create_policy(name)
                        protected = name != "full"
                        if protected:
                            policy = FirstTokenProtectedPolicy(policy)
                        adapter.forward(
                            torch.tensor([tokens["context_ids"]]),
                            collect_attention=policy_requires_attention(policy),
                        )
                        budget = (
                            len(tokens["context_ids"])
                            if name == "full"
                            else math.floor(len(tokens["context_ids"]) * 0.5)
                        )
                        signals = RetentionSignals(adapter.positions, adapter.attention_by_layer)
                        adapter.retain(
                            policy.select_indices(len(adapter.positions), budget, signals=signals)
                        )
                        retained = adapter.positions
                        retained_bytes = audit_cache(adapter, budget)
                        logits = adapter.forward(torch.tensor([tokens["query_ids"]]))[0, -1]
                        scoring_bytes = audit_cache(adapter, budget + len(tokens["query_ids"]))
                        colors = dict(zip(COLORS, logits[bare_ids].tolist()))
                        spaces = dict(zip(COLORS, logits[space_ids].tolist()))
                        primary = choose(colors, example["gold"])
                        raw_id = logits.argmax().item()
                        dense_check = None
                        dense_difference = None
                        dense_colors = None
                        if name == "full":
                            dense = model(
                                torch.tensor([tokens["context_ids"] + tokens["query_ids"]]),
                                use_cache=False,
                            ).logits[0, -1]
                            torch.testing.assert_close(logits, dense, atol=2e-4, rtol=1e-5)
                            dense_check = "passed"
                            dense_difference = (logits - dense).abs().max().item()
                            dense_colors = dict(zip(COLORS, dense[bare_ids].tolist()))
                            del dense
                        target_kept = sum(
                            position in retained for position in tokens["target_span"]
                        )
                        row = {
                            "example_id": example["id"],
                            "policy": policy.name,
                            "first_token_protected": protected,
                            **{
                                key: example[key]
                                for key in (
                                    "gold",
                                    "fact_count",
                                    "target_fact_index",
                                    "target_depth",
                                )
                            },
                            **primary,
                            "space_prediction": choose(spaces, example["gold"])["prediction"],
                            "color_logits": colors,
                            "space_color_logits": spaces,
                            "raw_greedy_id": raw_id,
                            "raw_greedy_text": tokenizer.decode([raw_id]),
                            "context_tokens": len(tokens["context_ids"]),
                            "query_tokens": len(tokens["query_ids"]),
                            "budget_tokens": budget,
                            "retained_positions": retained,
                            "target_retained_tokens": target_kept,
                            "target_retained_fraction": target_kept / len(tokens["target_span"]),
                            "retained_kv_bytes": retained_bytes,
                            "scoring_cache_tokens": len(adapter.positions),
                            "scoring_kv_bytes": scoring_bytes,
                            "attention_by_layer": signals.attention_by_layer,
                            "dense_check": dense_check,
                            "dense_max_abs_difference": dense_difference,
                            "dense_color_logits": dense_colors,
                            "seconds": time.perf_counter() - started,
                        }
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                        rows.append(row)
                        save(output, rows, "running")
                        print(
                            f"{len(rows)}/48: {example['id']} {policy.name} pred={row['prediction']} gold={row['gold']}",
                            flush=True,
                        )
                        del adapter, logits
            save(output, rows, "complete")
            write_json(output / "verification.json", verify(output))
        except (Exception, KeyboardInterrupt):
            traceback.print_exc()
            save(output, rows, "failed / interrupted")
            return 1
    print(f"Complete: {output / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
