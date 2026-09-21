"""Larger predeclared multi-seed query-aware development validation."""

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
POLICIES = ("full", "recency+first", "adaptive+first", "adaptive-query+first")
DEVELOPMENT_SEEDS = (20260922, 20260923)
CONFIRMATION_SEED = 20261001
ANALYSIS_SEED = 20260924
BOOTSTRAP_SAMPLES = 10_000
DATASET_SHA256 = "bc020e2e1a666e64fbea043bd9b35420fb13bd079fe17f63056535026a862f24"


def target_positions(facts):
    return tuple(round(slot * (facts - 1) / 7) for slot in range(8))


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
    probe_prefix_ids = tokenizer.encode(
        tokenized["context_prefix"] + question, add_special_tokens=False
    )
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
        index
        for index, (left, right) in enumerate(encoded["offset_mapping"])
        if right > start and left < end
    ]
    if not span:
        raise ValueError("Target fact has no token span")
    return span


def make_dataset():
    result = []
    for seed in DEVELOPMENT_SEEDS:
        rng = random.Random(seed)
        for facts in (16, 32):
            for slot, target in enumerate(target_positions(facts)):
                values = [rng.choice(COLORS) for _ in range(facts)]
                values[target] = COLORS[slot % 4]
                context = (
                    "Remember these locker colors.\n"
                    + "\n".join(
                        f"The Unit{index:02d} locker is {color}."
                        for index, color in enumerate(values)
                    )
                    + "\nEnd of records.\n"
                )
                result.append(
                    {
                        "id": f"larger-{seed}-{facts}-{slot}",
                        "split": "development",
                        "seed": seed,
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
        raise ValueError("Dataset differs from frozen larger-development construction")
    if any(example["seed"] == CONFIRMATION_SEED for example in data):
        raise ValueError("Confirmation seed must remain untouched")
    for example in data:
        prefix = f"The {example['target_name']} locker is "
        facts = [line for line in example["context"].splitlines() if line.startswith(prefix)]
        if len(facts) != 1 or facts[0].removeprefix(prefix).removesuffix(".") != example["gold"]:
            raise ValueError("Gold does not match the uniquely requested fact")


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


def exact_paired_p(wins, losses):
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant
    return min(1.0, 2 * tail)


def bootstrap_interval(differences):
    if not differences:
        return None
    rng = random.Random(ANALYSIS_SEED)
    n = len(differences)
    samples = sorted(
        sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_SAMPLES)
    )
    return [samples[int(0.025 * BOOTSTRAP_SAMPLES)], samples[int(0.975 * BOOTSTRAP_SAMPLES)]]


def comparison(rows, full, baseline_policy):
    lookup = {(row["example_id"], row["policy"]): row for row in rows}
    differences = []
    pairs = []
    for example_id in sorted(full):
        if not full[example_id]["correct"]:
            continue
        query = lookup[example_id, "adaptive-query+first"]
        baseline = lookup[example_id, baseline_policy]
        delta = int(query["correct"]) - int(baseline["correct"])
        differences.append(delta)
        pairs.append(
            {
                "example_id": example_id,
                "seed": query["seed"],
                "fact_count": query["fact_count"],
                "delta": delta,
            }
        )
    wins = differences.count(1)
    losses = differences.count(-1)
    return {
        "baseline": baseline_policy,
        "n": len(differences),
        "query_correct": sum(
            lookup[key, "adaptive-query+first"]["correct"] for key in full if full[key]["correct"]
        ),
        "baseline_correct": sum(
            lookup[key, baseline_policy]["correct"] for key in full if full[key]["correct"]
        ),
        "wins": wins,
        "losses": losses,
        "ties": differences.count(0),
        "paired_accuracy_difference": sum(differences) / len(differences) if differences else None,
        "bootstrap_95_interval": bootstrap_interval(differences),
        "exact_two_sided_p": exact_paired_p(wins, losses),
        "pairs": pairs,
    }


def summarize(rows, status):
    full = {row["example_id"]: row for row in rows if row["policy"] == "full"}
    groups = []
    for policy in POLICIES:
        for seed in DEVELOPMENT_SEEDS:
            for facts in (16, 32):
                group = [
                    row
                    for row in rows
                    if row["policy"] == policy
                    and row["seed"] == seed
                    and row["fact_count"] == facts
                ]
                if group:
                    subset = [
                        row for row in group if full.get(row["example_id"], {}).get("correct")
                    ]
                    groups.append(
                        {
                            "policy": policy,
                            "seed": seed,
                            "fact_count": facts,
                            **metrics(group),
                            "full_correct_n": len(subset),
                            "correct_on_full_correct": sum(row["correct"] for row in subset),
                        }
                    )
    full_rows = list(full.values())
    baseline = (
        status == "complete"
        and len(full_rows) == 32
        and all(
            len(seed_rows := [row for row in full_rows if row["seed"] == seed]) == 16
            and metrics(seed_rows)["correct"] >= 14
            and metrics(seed_rows)["raw_correct"] >= 14
            and metrics(seed_rows)["raw_valid"] == 16
            and all(
                sum(row["correct"] for row in seed_rows if row["fact_count"] == facts) >= 7
                for facts in (16, 32)
            )
            for seed in DEVELOPMENT_SEEDS
        )
        and all(row["dense_check"] == "passed" for row in full_rows)
    )
    comparisons = (
        [
            comparison(rows, full, baseline_policy)
            for baseline_policy in ("recency+first", "adaptive+first")
        ]
        if len(rows) == 128
        else []
    )

    def cell_correct(policy, seed, facts):
        return sum(
            row["correct"]
            for row in rows
            if row["policy"] == policy
            and row["seed"] == seed
            and row["fact_count"] == facts
            and full.get(row["example_id"], {}).get("correct")
        )

    proceed = (
        baseline
        and len(comparisons) == 2
        and all(item["query_correct"] - item["baseline_correct"] >= 4 for item in comparisons)
        and all(
            cell_correct("adaptive-query+first", seed, facts)
            >= cell_correct(baseline_policy, seed, facts)
            for seed in DEVELOPMENT_SEEDS
            for facts in (16, 32)
            for baseline_policy in ("recency+first", "adaptive+first")
        )
    )
    return {
        "status": status,
        "completed": len(rows),
        "expected": 128,
        "groups": groups,
        "comparisons": comparisons,
        "baseline_gate": "passed" if baseline else "failed" if status == "complete" else "pending",
        "confirmation_decision": "proceed"
        if proceed
        else "diagnose"
        if status == "complete"
        else "pending",
        "confirmation_seed_consumed": False,
    }


def save(output, rows, status):
    summary = summarize(rows, status)
    write_json(output / "summary.json", summary)
    lines = [
        "# Larger multi-seed query-aware development validation",
        "",
        f"Status: {status}; {len(rows)}/128 evaluations.",
        "",
        "| Policy | Seed | Facts | Correct | Correct on full-correct | Target retained | KV bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in summary["groups"]:
        lines.append(
            f"| {group['policy']} | {group['seed']} | {group['fact_count']} | "
            f"{group['correct']}/{group['n']} | {group['correct_on_full_correct']}/{group['full_correct_n']} | "
            f"{group['mean_target_retained_fraction']:.1%} | {group['mean_retained_kv_bytes']:.0f} |"
        )
    lines += [
        "",
        f"Baseline gate: **{summary['baseline_gate']}**.",
        f"Confirmation decision: **{summary['confirmation_decision']}**.",
        "Confirmation seed consumed: **no**.",
        "",
        "Bootstrap intervals and exact paired tests are descriptive development analyses.",
        "Query-aware still uses an extra discarded full-context probe.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify(output):
    data = json.loads((output / "dataset.json").read_text())
    validate_data(data)
    settings = json.loads((output / "run.json").read_text())
    assert settings["confirmation_seed"] == CONFIRMATION_SEED
    assert settings["confirmation_seed_consumed"] is False
    assert (
        settings["dataset_sha256"]
        == hashlib.sha256((output / "dataset.json").read_bytes()).hexdigest()
    )
    inputs = json.loads((output / "inputs.json").read_text())
    examples = {example["id"]: example for example in data}
    tokens = {item["id"]: item for item in inputs["examples"]}
    rows = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 128
    assert {(row["example_id"], row["policy"]) for row in rows} == {
        (example_id, policy) for example_id in examples for policy in POLICIES
    }
    full = {row["example_id"]: row for row in rows if row["policy"] == "full"}
    for row in rows:
        example, tokenized = examples[row["example_id"]], tokens[row["example_id"]]
        for field in ("seed", "gold", "fact_count", "target_fact_index", "target_depth"):
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
            for color in COLORS:
                assert abs(
                    row["color_logits"][color] - row["dense_color_logits"][color]
                ) <= 2e-4 + 1e-5 * abs(row["dense_color_logits"][color])
        else:
            assert row["first_token_protected"] and 0 in row["retained_positions"]
            assert row["dense_check"] is None
        if row["policy"] == "adaptive-query+first":
            assert row["probe_seconds"] > 0
            assert row["probe_cache_tokens"] == context + len(tokenized["question_probe_ids"])
            assert all(len(layer) == context for layer in row["selector_attention_by_layer"])
        else:
            assert row["probe_seconds"] == 0 and row["probe_cache_tokens"] is None
    assert json.loads((output / "summary.json").read_text()) == summarize(rows, "complete")
    return {"status": "passed", "rows": len(rows), "confirmation_seed_consumed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/query-aware-larger"))
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("Do not use Python -O for evidence verification")
    if args.verify:
        print(json.dumps(verify(args.verify), indent=2))
        return 0
    if DATASET_SHA256.startswith("TO_BE_"):
        raise RuntimeError("Freeze dataset hash before inference")
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
        raise ValueError("Frozen larger-development dataset hash changed")
    snapshot_sources(root, output)
    write_json(
        output / "run.json",
        {
            "model": MODEL,
            "revision": REVISION,
            "development_seeds": list(DEVELOPMENT_SEEDS),
            "confirmation_seed": CONFIRMATION_SEED,
            "confirmation_seed_consumed": False,
            "analysis_seed": ANALYSIS_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "device": "cpu",
            "dtype": "float32",
            "attention": "eager",
            "python": sys.version,
            "platform": platform.platform(),
            "versions": {
                name: importlib.metadata.version(name) for name in ("torch", "transformers")
            },
            "protocol": "research/query_aware_larger_protocol.md",
            "ratio": 0.5,
            "expected": 128,
            "policies": list(POLICIES),
            "dataset_sha256": dataset_hash,
        },
    )
    rows = []
    save(output, rows, "running")
    print(f"Larger query-aware report: {output / 'report.md'}", flush=True)
    with (
        (output / "run.log").open("w", buffering=1) as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        try:
            import torch
            from transformers import AutoTokenizer, Qwen2ForCausalLM

            from amt.cache import Qwen2CacheAdapter

            torch.manual_seed(DEVELOPMENT_SEEDS[0])
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
                        probe_seconds, probe_cache_tokens, selector_attention = 0.0, None, ()
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
                        dense_check = dense_difference = dense_colors = None
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
                                    "seed",
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
                            f"{len(rows)}/128: {example['id']} {policy_name} pred={row['prediction']} gold={row['gold']}",
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
