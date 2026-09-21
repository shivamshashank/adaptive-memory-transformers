"""Offline, development-only delayed-query retrieval pilot. No quality claims."""

from __future__ import annotations

import argparse
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
from typing import Any, cast

from amt.validation import snapshot_sources, write_json


def audit_cache(adapter: Any, expected_tokens: int) -> int:
    """Verify physical length/storage for every K/V tensor, not just metadata."""
    if len(adapter.positions) != expected_tokens:
        raise RuntimeError("Retained-position count does not match assigned capacity")
    total = 0
    for layer in adapter.cache.layers:
        for tensor in (layer.keys, layer.values):
            if tensor is None or tensor.shape[-2] != expected_tokens:
                raise RuntimeError("Physical cache length does not match assigned capacity")
            logical_bytes = tensor.numel() * tensor.element_size()
            if tensor.untyped_storage().nbytes() != logical_bytes:
                raise RuntimeError("Cache tensor retains a larger backing storage")
            total += logical_bytes
    return total


def make_examples(
    count: int = 20, seed: int = 20260919, version: str = "v1"
) -> list[dict[str, Any]]:
    """Balanced answer labels, varying fact locations; development data only."""
    if count < 1:
        raise ValueError("count must be positive")
    if version not in ("v1", "v2"):
        raise ValueError("Unknown dataset version")
    if version == "v2" and count % 32:
        raise ValueError("v2 requires multiples of 32 for complete position/label balancing")
    rng = random.Random(seed)
    colors = ["red", "blue", "green", "yellow"]
    names = ["Aster", "Birch", "Cedar", "Dahlia", "Elm", "Fern", "Grove", "Hazel"]
    cells = [(position, label) for position in range(8) for label in range(4)] * (count // 32)
    if version == "v2":
        rng.shuffle(cells)
    examples = []
    for index in range(count):
        target = index % len(names) if version == "v1" else cells[index][0]
        values = [rng.choice(colors) for _ in names]
        options = colors.copy()
        rng.shuffle(options)
        label = index % 4 if version == "v1" else cells[index][1]
        values[target] = options[label]
        facts = [f"The {name} locker is {color}." for name, color in zip(names, values)]
        context = "Remember these locker colors.\n" + "\n".join(facts) + "\nEnd of records.\n"
        query = (
            f"What color is the {names[target]} locker?\n"
            + "\n".join(f"{letter}: {color}" for letter, color in zip("ABCD", options))
            + "\nAnswer with only the correct letter.\nAnswer:"
        )
        examples.append(
            {
                "id": (
                    f"dev-{seed}-{index:03d}" if version == "v1" else f"dev-v2-{seed}-{index:03d}"
                ),
                "split": "development",
                "context": context,
                "query": query,
                "gold": "ABCD"[label],
                "target_fact_index": target,
                "target_color": values[target],
                "options": options,
            }
        )
    return examples


def tokenize_example(tokenizer: Any, example: dict[str, Any], version: str) -> dict[str, Any]:
    """Split a rendered chat ONLY at a verified token boundary before the question."""
    if version == "v1":
        context_text = example["context"]
        text = context_text + example["query"]
    else:
        content = example["context"] + example["query"]
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        if text.count(content) != 1:
            raise ValueError("Chat template does not preserve the unique user content")
        context_text = text[: text.index(content)] + example["context"]
    context_ids = tokenizer.encode(context_text, add_special_tokens=False)
    full_ids = tokenizer.encode(text, add_special_tokens=False)
    if full_ids[: len(context_ids)] != context_ids:
        raise ValueError("Context/query boundary changes tokenization")
    query_ids = full_ids[len(context_ids) :]
    if not context_ids or not query_ids:
        raise ValueError("Empty context or query")
    return {
        "id": example["id"],
        "context_ids": context_ids,
        "query_ids": query_ids,
        "rendered_prompt": text,
        "context_prefix": context_text,
    }


def score_choices(scores: list[float], gold: str) -> dict[str, Any]:
    if len(scores) != 4 or gold not in "ABCD" or len(gold) != 1:
        raise ValueError("Four scores and one gold label A-D required")
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("Non-finite logits")
    prediction = "ABCD"[max(range(4), key=scores.__getitem__)]
    return {
        "prediction": prediction,
        "correct": prediction == gold,
        "choice_logits": dict(zip("ABCD", scores)),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full = {row["example_id"]: row["correct"] for row in rows if row["policy"] == "full"}
    groups: dict[tuple[str, float | None], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["policy"], row["ratio"]), []).append(row)
    summaries = []
    for (policy, ratio), group in groups.items():
        reference_correct = [row for row in group if full.get(row["example_id"], False)]
        summaries.append(
            {
                "policy": policy,
                "ratio": ratio,
                "n": len(group),
                "correct": sum(row["correct"] for row in group),
                "accuracy": sum(row["correct"] for row in group) / len(group),
                "full_correct_n": len(reference_correct),
                "accuracy_on_full_correct": (
                    sum(row["correct"] for row in reference_correct) / len(reference_correct)
                    if reference_correct
                    else None
                ),
                "mean_retained_kv_bytes": sum(row["retained_kv_bytes"] for row in group)
                / len(group),
                "total_seconds": sum(row["seconds"] for row in group),
            }
        )
    return summaries


def save_report(output: Path, rows: list[dict[str, Any]], expected: int, status: str) -> None:
    summary = summarize(rows)
    write_json(
        output / "summary.json",
        {
            "status": status,
            "completed": len(rows),
            "expected": expected,
            "groups": summary,
        },
    )
    lines = [
        "# Delayed-query retrieval: development pilot",
        "",
        f"Status: {status}; completed {len(rows)}/{expected} evaluations.",
        "",
        "CPU float32, eager attention, cached pinned Qwen. No paid compute.",
        "",
        "Read facts → prune context cache → append question → score A/B/C/D.",
        "",
        "| Policy | Context budget | Correct / N | Accuracy | Mean retained KV bytes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in summary:
        ratio = "full" if item["ratio"] is None else f"{item['ratio']:.0%}"
        lines.append(
            f"| {item['policy']} | {ratio} | {item['correct']}/{item['n']} | "
            f"{item['accuracy']:.1%} | {item['mean_retained_kv_bytes']:.0f} |"
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "This is synthetic short-context development data, not held-out research evidence.",
        "Accuracy is constrained multiple-choice (chance 25%), not free-text generation.",
        "Full-cache failures remain in the denominator; summary.json also reports results",
        "on full-cache-correct examples. Do not infer significance from this small pilot.",
        "Budget applies to retained CONTEXT tokens before the query. Query tokens are appended",
        "without pruning for scoring, equally for all policies. KV bytes are not peak RAM.",
        "Attention selection uses the last context token, without seeing the future question.",
        "Runtime includes instrumentation and is not a speed benchmark. Full cache runs once",
        "per example; adaptive weights are fixed 0.5 attention / 0.5 recency, no protections.",
        "",
        "## Evidence",
        "",
        "- dataset.json: exact facts, questions, options and gold labels",
        "- inputs.json: actual token IDs and candidate IDs",
        "- rows.jsonl: scores, raw greedy token, retained positions, signals and timing",
        "- summary.json: aggregates and full-cache-correct subset",
        "- run.json / source_snapshot.zip / run.log: settings, code and execution log",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--dataset-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.5, 0.25])
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluation"))
    args = parser.parse_args(argv)
    if args.dataset_version == "v2" and args.examples % 32:
        parser.error("v2 requires --examples to be a positive multiple of 32")
    if (
        args.examples < 1
        or len(set(args.ratios)) != len(args.ratios)
        or any(not math.isfinite(ratio) or not 0 < ratio < 1 for ratio in args.ratios)
    ):
        parser.error("Positive example count and unique ratios strictly between 0 and 1 required")
    root = Path(__file__).resolve().parents[2]
    output = args.output_dir.resolve() / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    output.mkdir(parents=True, exist_ok=False)
    print(f"Pilot report: {output / 'report.md'}", flush=True)
    rows: list[dict[str, Any]] = []
    expected = args.examples * (1 + 4 * len(args.ratios))
    save_report(output, rows, expected, "running")
    os.environ.update(
        {
            "HF_HOME": str(root / ".model-cache/huggingface"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    snapshot_sources(root, output)
    data = make_examples(args.examples, args.seed, args.dataset_version)
    write_json(output / "dataset.json", data)
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    revision = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    write_json(
        output / "run.json",
        {
            "model": model_id,
            "revision": revision,
            "seed": args.seed,
            "ratios": args.ratios,
            "expected_evaluations": expected,
            "split": "development",
            "dataset_version": args.dataset_version,
            "prompt_format": "chat" if args.dataset_version == "v2" else "plain",
            "scoring": "argmax over single-token bare A/B/C/D; ties prefer first label",
            "device": "cpu",
            "dtype": "float32",
            "attention": "eager",
            "python": sys.version,
            "platform": platform.platform(),
            "versions": {
                name: importlib.metadata.version(name) for name in ("torch", "transformers")
            },
            "dataset_sha256": hashlib.sha256((output / "dataset.json").read_bytes()).hexdigest(),
            "policy_settings": {
                "adaptive": {"attention": 0.5, "recency": 0.5, "sink_tokens": 0, "local_window": 0}
            },
        },
    )
    # Redirect both streams so model-loader messages and failures are preserved too.
    with (output / "run.log").open("w", buffering=1) as log:
        import contextlib

        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                import torch
                from transformers import AutoTokenizer, Qwen2ForCausalLM

                from amt.cache import Qwen2CacheAdapter
                from amt.policies import AttentionRecencyPolicy, RetentionSignals, create_policy

                torch.manual_seed(args.seed)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    revision=revision,
                    local_files_only=True,
                )
                model = Qwen2ForCausalLM.from_pretrained(
                    model_id,
                    revision=revision,
                    local_files_only=True,
                    dtype=torch.float32,
                    attn_implementation="eager",
                ).eval()
                choices = [tokenizer.encode(letter, add_special_tokens=False) for letter in "ABCD"]
                if any(len(ids) != 1 for ids in choices):
                    raise ValueError("Candidate labels must each tokenize to exactly one token")
                choice_ids = [ids[0] for ids in choices]
                tokenized = [
                    tokenize_example(tokenizer, example, args.dataset_version) for example in data
                ]
                write_json(
                    output / "inputs.json", {"examples": tokenized, "choice_ids": choice_ids}
                )
                with (
                    torch.inference_mode(),
                    (output / "rows.jsonl").open("w", buffering=1) as handle,
                ):
                    for example, tokens in zip(data, tokenized):
                        settings: list[tuple[str, float | None]] = [("full", None)]
                        settings += [
                            (name, ratio)
                            for ratio in args.ratios
                            for name in ("recency", "uniform", "attention", "adaptive")
                        ]
                        for name, ratio in settings:
                            started = time.perf_counter()
                            adapter = Qwen2CacheAdapter(model)
                            policy = create_policy(name)
                            collect = isinstance(policy, AttentionRecencyPolicy)
                            adapter.forward(
                                torch.tensor([tokens["context_ids"]]), collect_attention=collect
                            )
                            budget = (
                                len(tokens["context_ids"])
                                if ratio is None
                                else max(1, math.floor(len(tokens["context_ids"]) * ratio))
                            )
                            signals = RetentionSignals(
                                adapter.positions, adapter.attention_by_layer
                            )
                            adapter.retain(
                                policy.select_indices(
                                    len(adapter.positions), budget, signals=signals
                                )
                            )
                            retained = adapter.positions
                            kv_bytes = audit_cache(adapter, budget)
                            # Query is deliberately unseen until AFTER context pruning.
                            logits = adapter.forward(torch.tensor([tokens["query_ids"]]))[0, -1]
                            scoring_bytes = audit_cache(adapter, budget + len(tokens["query_ids"]))
                            scores = logits[choice_ids].tolist()
                            raw_id = cast(int, logits.argmax().item())
                            measured_seconds = time.perf_counter() - started
                            dense_difference = None
                            if name == "full" and args.dataset_version == "v2":
                                dense = model(
                                    torch.tensor([tokens["context_ids"] + tokens["query_ids"]]),
                                    use_cache=False,
                                ).logits[0, -1]
                                torch.testing.assert_close(logits, dense, atol=2e-4, rtol=1e-5)
                                dense_difference = (logits - dense).abs().max().item()
                                del dense
                            row = {
                                "example_id": example["id"],
                                "policy": name,
                                "ratio": ratio,
                                "gold": example["gold"],
                                **score_choices(scores, example["gold"]),
                                "raw_greedy_id": raw_id,
                                "raw_greedy_text": tokenizer.decode([raw_id]),
                                "context_tokens": len(tokens["context_ids"]),
                                "query_tokens": len(tokens["query_ids"]),
                                "budget_tokens": budget,
                                "retained_positions": retained,
                                "retained_kv_bytes": kv_bytes,
                                "scoring_cache_tokens": len(adapter.positions),
                                "scoring_kv_bytes": scoring_bytes,
                                "budget_audit": "passed",
                                "attention_by_layer": signals.attention_by_layer,
                                "seconds": measured_seconds,
                                "dense_reference_max_abs_difference": dense_difference,
                            }
                            handle.write(json.dumps(row, allow_nan=False) + "\n")
                            rows.append(row)
                            save_report(output, rows, expected, "running")
                            print(
                                f"{len(rows)}/{expected}: {example['id']} {name} {ratio}: "
                                f"pred={row['prediction']} gold={row['gold']} seconds={row['seconds']:.2f}",
                                flush=True,
                            )
                            del adapter, logits
                save_report(output, rows, expected, "complete")
            except (Exception, KeyboardInterrupt):
                traceback.print_exc()
                save_report(output, rows, expected, "failed / interrupted")
                return 1
    print(f"Complete: {output / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
