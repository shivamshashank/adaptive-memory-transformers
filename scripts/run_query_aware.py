"""Frozen two-pass query-aware cache-selection development experiment."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
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
POLICIES = ("full", "recency+first", "adaptive+first", "adaptive-query+first")


def target_positions(facts):
    return tuple(round(slot * (facts - 1) / 7) for slot in range(8))


def make_dataset():
    import random

    rng = random.Random(SEED)
    result = []
    for facts in (16, 32):
        for slot, target in enumerate(target_positions(facts)):
            values = [rng.choice(COLORS) for _ in range(facts)]
            values[target] = COLORS[slot % 4]
            context = (
                "Remember these locker colors.\n"
                + "\n".join(
                    f"The Unit{index:02d} locker is {color}." for index, color in enumerate(values)
                )
                + "\nEnd of records.\n"
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
                    "query": (
                        f"What color is the Unit{target:02d} locker?\n"
                        "Answer with only the color word (red, blue, green, or yellow).\n"
                        "Answer:"
                    ),
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


def tokenize_with_probe(tokenizer, example):
    tokenized = tokenize_example(tokenizer, example, "v2")
    question = example["query"].splitlines()[0] + "\n"
    probe_prefix = tokenized["context_prefix"] + question
    probe_prefix_ids = tokenizer.encode(probe_prefix, add_special_tokens=False)
    context_ids = tokenized["context_ids"]
    full_ids = context_ids + tokenized["query_ids"]
    if (
        probe_prefix_ids[: len(context_ids)] != context_ids
        or full_ids[: len(probe_prefix_ids)] != probe_prefix_ids
    ):
        raise ValueError("Question-probe boundary changes tokenization")
    question_ids = probe_prefix_ids[len(context_ids) :]
    if not question_ids:
        raise ValueError("Question probe is empty")
    return tokenized | {"question_probe_ids": question_ids, "question_probe_text": question}


def target_span(tokenizer, tokenized, example):
    text = tokenized["context_prefix"]
    fact = f"The {example['target_name']} locker is {example['gold']}."
    start, end = text.index(fact), text.index(fact) + len(fact)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != tokenized["context_ids"]:
        raise ValueError("Offset tokenizer disagrees with context IDs")
    span = [
        i
        for i, (left, right) in enumerate(encoded["offset_mapping"])
        if right > start and left < end
    ]
    if not span:
        raise ValueError("Target fact has no token span")
    return span


def metrics(rows):
    return {
        "n": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "raw_valid": sum(row["raw_greedy_text"].strip() in COLORS for row in rows),
        "raw_correct": sum(row["raw_greedy_text"].strip() == row["gold"] for row in rows),
        "mean_target_retained_fraction": sum(row["target_retained_fraction"] for row in rows)
        / len(rows)
        if rows
        else None,
        "mean_retained_kv_bytes": sum(row["retained_kv_bytes"] for row in rows) / len(rows)
        if rows
        else None,
        "total_probe_seconds": sum(row["probe_seconds"] for row in rows),
    }


def summarize(rows, status):
    full = {row["example_id"]: row for row in rows if row["policy"] == "full"}
    groups = []
    for policy in POLICIES:
        for facts in (16, 32):
            group = [row for row in rows if row["policy"] == policy and row["fact_count"] == facts]
            if group:
                subset = [row for row in group if full.get(row["example_id"], {}).get("correct")]
                groups.append(
                    {
                        "policy": policy,
                        "fact_count": facts,
                        **metrics(group),
                        "full_correct_n": len(subset),
                        "correct_on_full_correct": sum(row["correct"] for row in subset),
                    }
                )
    full_rows = list(full.values())
    baseline = (
        status == "complete"
        and len(full_rows) == 16
        and metrics(full_rows)["correct"] >= 14
        and metrics(full_rows)["raw_correct"] >= 14
        and metrics(full_rows)["raw_valid"] == 16
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
    lookup = {(row["example_id"], row["policy"]): row for row in rows}
    pairs = []
    for example_id in sorted(full):
        if all((example_id, policy) in lookup for policy in POLICIES[1:]):
            pairs.append(
                {
                    "example_id": example_id,
                    "fact_count": full[example_id]["fact_count"],
                    "full_correct": full[example_id]["correct"],
                    "query_minus_recency": int(
                        lookup[example_id, "adaptive-query+first"]["correct"]
                    )
                    - int(lookup[example_id, "recency+first"]["correct"]),
                    "query_minus_context_adaptive": int(
                        lookup[example_id, "adaptive-query+first"]["correct"]
                    )
                    - int(lookup[example_id, "adaptive+first"]["correct"]),
                }
            )

    def correct_on_full(policy, facts=None):
        return sum(
            row["correct"]
            for row in rows
            if row["policy"] == policy
            and full.get(row["example_id"], {}).get("correct")
            and (facts is None or row["fact_count"] == facts)
        )

    query_score = correct_on_full("adaptive-query+first")
    proceed = (
        baseline
        and len(pairs) == 16
        and query_score - correct_on_full("recency+first") >= 2
        and query_score - correct_on_full("adaptive+first") >= 2
        and all(
            correct_on_full("adaptive-query+first", facts)
            >= correct_on_full(baseline_policy, facts)
            for facts in (16, 32)
            for baseline_policy in ("recency+first", "adaptive+first")
        )
    )
    return {
        "status": status,
        "completed": len(rows),
        "expected": 64,
        "groups": groups,
        "paired_changes": pairs,
        "baseline_gate": "passed" if baseline else "failed" if status == "complete" else "pending",
        "query_aware_decision": "proceed"
        if proceed
        else "diagnose"
        if status == "complete"
        else "pending",
    }


def save(output, rows, status):
    summary = summarize(rows, status)
    write_json(output / "summary.json", summary)
    lines = [
        "# Two-pass query-aware selection experiment",
        "",
        f"Status: {status}; {len(rows)}/64 evaluations.",
        "",
        "| Policy | Facts | Correct | Correct on full-correct | Target retained | KV bytes | Probe seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in summary["groups"]:
        lines.append(
            f"| {group['policy']} | {group['fact_count']} | {group['correct']}/{group['n']} | "
            f"{group['correct_on_full_correct']}/{group['full_correct_n']} | "
            f"{group['mean_target_retained_fraction']:.1%} | {group['mean_retained_kv_bytes']:.0f} | "
            f"{group['total_probe_seconds']:.2f} |"
        )
    lines += [
        "",
        f"Baseline gate: **{summary['baseline_gate']}**.",
        f"Query-aware larger-run decision: **{summary['query_aware_decision']}**.",
        "",
        "Query-aware uses an extra discarded full-context probe; it is not an efficiency result.",
        "All policies use equal retained context budgets on the fresh scoring pass.",
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
    assert len(rows) == 64
    assert {(row["example_id"], row["policy"]) for row in rows} == {
        (example_id, policy) for example_id in examples for policy in POLICIES
    }
    full = {row["example_id"]: row for row in rows if row["policy"] == "full"}
    for row in rows:
        example, tokenized = examples[row["example_id"]], tokens[row["example_id"]]
        for field in ("gold", "fact_count", "target_fact_index", "target_depth"):
            assert row[field] == example[field]
        assert all(
            row[key] == value for key, value in choose(row["color_logits"], row["gold"]).items()
        )
        context, query = len(tokenized["context_ids"]), len(tokenized["query_ids"])
        budget = context if row["policy"] == "full" else math.floor(context * 0.5)
        assert row["budget_tokens"] == len(row["retained_positions"]) == budget
        assert row["retained_positions"] == sorted(set(row["retained_positions"]))
        assert all(0 <= position < context for position in row["retained_positions"])
        assert row["scoring_cache_tokens"] == budget + query
        bytes_per_token = full[row["example_id"]]["retained_kv_bytes"] / context
        assert row["retained_kv_bytes"] == bytes_per_token * budget
        assert row["scoring_kv_bytes"] == bytes_per_token * (budget + query)
        kept = sum(position in row["retained_positions"] for position in tokenized["target_span"])
        assert row["target_retained_tokens"] == kept
        assert row["target_retained_fraction"] == kept / len(tokenized["target_span"])
        if row["policy"] == "full":
            assert not row["first_token_protected"] and row["dense_check"] == "passed"
            assert row["probe_seconds"] == 0 and row["selector_attention_by_layer"] == []
            for color in COLORS:
                assert abs(
                    row["color_logits"][color] - row["dense_color_logits"][color]
                ) <= 2e-4 + 1e-5 * abs(row["dense_color_logits"][color])
        else:
            assert row["first_token_protected"] and 0 in row["retained_positions"]
            assert row["dense_check"] is None
        if row["policy"] == "adaptive-query+first":
            assert row["probe_seconds"] > 0 and row["probe_cache_tokens"] == context + len(
                tokenized["question_probe_ids"]
            )
            assert all(len(layer) == context for layer in row["selector_attention_by_layer"])
        else:
            assert row["probe_seconds"] == 0 and row["probe_cache_tokens"] is None
    assert json.loads((output / "summary.json").read_text()) == summarize(rows, "complete")
    return {
        "status": "passed",
        "rows": len(rows),
        "scope": "Saved consistency, not independent model inference",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/query-aware"))
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
        raise ValueError("Frozen dataset hash changed")
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
            "protocol": "research/query_aware_protocol.md",
            "ratio": 0.5,
            "expected": 64,
            "policies": list(POLICIES),
            "query_attention_reduction": "mean tokens and heads per layer",
            "dataset_sha256": dataset_hash,
        },
    )
    rows = []
    save(output, rows, "running")
    print(f"Query-aware report: {output / 'report.md'}", flush=True)
    with (
        (output / "run.log").open("w", buffering=1) as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        try:
            import torch
            from transformers import AutoTokenizer, Qwen2ForCausalLM

            from amt.cache import Qwen2CacheAdapter

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
            if any(len(ids) != 1 for ids in bare.values()):
                raise ValueError("Frozen scorer requires single-token bare colors")
            tokenized = [tokenize_with_probe(tokenizer, example) for example in data]
            for example, tokens in zip(data, tokenized):
                tokens["target_span"] = target_span(tokenizer, tokens, example)
                expected_context = 161 if example["fact_count"] == 16 else 289
                if (
                    len(tokens["context_ids"]) != expected_context
                    or len(tokens["query_ids"]) != 32
                    or len(tokens["question_probe_ids"]) != 9
                    or len(tokens["target_span"]) != 8
                ):
                    raise ValueError("Frozen token lengths changed")
            write_json(output / "inputs.json", {"examples": tokenized, "bare_color_ids": bare})
            choice_ids = [bare[color][0] for color in COLORS]
            with (
                torch.inference_mode(),
                (output / "rows.jsonl").open("w", buffering=1, encoding="utf-8") as handle,
            ):
                for example, tokens in zip(data, tokenized):
                    context_count = len(tokens["context_ids"])
                    for policy_name in POLICIES:
                        started = time.perf_counter()
                        probe_seconds = 0.0
                        probe_cache_tokens = None
                        selector_attention = ()
                        if policy_name == "adaptive-query+first":
                            probe_started = time.perf_counter()
                            probe = Qwen2CacheAdapter(model)
                            probe.forward(torch.tensor([tokens["context_ids"]]))
                            probe.forward(
                                torch.tensor([tokens["question_probe_ids"]]),
                                collect_attention=True,
                                attention_query_reduction="mean",
                            )
                            selector_attention = tuple(
                                tuple(layer[:context_count]) for layer in probe.attention_by_layer
                            )
                            probe_seconds = time.perf_counter() - probe_started
                            probe_cache_tokens = len(probe.positions)
                            del probe
                        adapter = Qwen2CacheAdapter(model)
                        collect_context = policy_name == "adaptive+first"
                        adapter.forward(
                            torch.tensor([tokens["context_ids"]]), collect_attention=collect_context
                        )
                        if collect_context:
                            selector_attention = adapter.attention_by_layer
                        base_name = (
                            "full"
                            if policy_name == "full"
                            else "recency"
                            if policy_name == "recency+first"
                            else "adaptive"
                        )
                        policy = create_policy(base_name)
                        protected = policy_name != "full"
                        if protected:
                            policy = FirstTokenProtectedPolicy(policy)
                        budget = context_count if not protected else math.floor(context_count * 0.5)
                        signals = RetentionSignals(adapter.positions, selector_attention)
                        adapter.retain(
                            policy.select_indices(context_count, budget, signals=signals)
                        )
                        retained = adapter.positions
                        retained_bytes = audit_cache(adapter, budget)
                        logits = adapter.forward(torch.tensor([tokens["query_ids"]]))[0, -1]
                        scoring_bytes = audit_cache(adapter, budget + len(tokens["query_ids"]))
                        color_logits = dict(zip(COLORS, logits[choice_ids].tolist()))
                        primary = choose(color_logits, example["gold"])
                        raw_id = logits.argmax().item()
                        dense_check = None
                        dense_difference = None
                        dense_colors = None
                        if policy_name == "full":
                            dense = model(
                                torch.tensor([tokens["context_ids"] + tokens["query_ids"]]),
                                use_cache=False,
                            ).logits[0, -1]
                            torch.testing.assert_close(logits, dense, atol=2e-4, rtol=1e-5)
                            dense_check = "passed"
                            dense_difference = (logits - dense).abs().max().item()
                            dense_colors = dict(zip(COLORS, dense[choice_ids].tolist()))
                            del dense
                        target_kept = sum(
                            position in retained for position in tokens["target_span"]
                        )
                        row = {
                            "example_id": example["id"],
                            "policy": policy_name,
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
                            "color_logits": color_logits,
                            "raw_greedy_id": raw_id,
                            "raw_greedy_text": tokenizer.decode([raw_id]),
                            "budget_tokens": budget,
                            "retained_positions": retained,
                            "target_retained_tokens": target_kept,
                            "target_retained_fraction": target_kept / len(tokens["target_span"]),
                            "retained_kv_bytes": retained_bytes,
                            "scoring_cache_tokens": len(adapter.positions),
                            "scoring_kv_bytes": scoring_bytes,
                            "selector_attention_by_layer": selector_attention,
                            "probe_seconds": probe_seconds,
                            "probe_cache_tokens": probe_cache_tokens,
                            "dense_check": dense_check,
                            "dense_max_abs_difference": dense_difference,
                            "dense_color_logits": dense_colors,
                            "seconds": time.perf_counter() - started,
                        }
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                        rows.append(row)
                        save(output, rows, "running")
                        print(
                            f"{len(rows)}/64: {example['id']} {policy_name} pred={row['prediction']} gold={row['gold']}",
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
