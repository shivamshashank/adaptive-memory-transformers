"""Calibrate one frozen exact-lookup prompt on fresh development examples."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
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
from amt.validation import snapshot_sources, write_json

COLORS = ("red", "blue", "green", "yellow")
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DEVELOPMENT_SEEDS = (20260925, 20260926)
CONFIRMATION_SEED = 20261001
DATASET_SHA256 = "81cce90fd1103daefbbea128ba76df6c0121e1bb29a8c70521905063ab5f42ff"


def target_positions(facts):
    return tuple(round(slot * (facts - 1) / 7) for slot in range(8))


def choose(scores, gold):
    if set(scores) != set(COLORS) or gold not in COLORS:
        raise ValueError("Four color scores and a valid gold color are required")
    prediction = max(COLORS, key=scores.__getitem__)
    return {"prediction": prediction, "correct": prediction == gold}


def make_dataset():
    result = []
    for seed in DEVELOPMENT_SEEDS:
        rng = random.Random(seed)
        for facts in (16, 32):
            for slot, target in enumerate(target_positions(facts)):
                values = [rng.choice(COLORS) for _ in range(facts)]
                values[target] = COLORS[slot % 4]
                name = f"Unit{target:02d}"
                result.append(
                    {
                        "id": f"exact-lookup-{seed}-{facts}-{slot}",
                        "split": "development",
                        "seed": seed,
                        "fact_count": facts,
                        "target_fact_index": target,
                        "target_depth": target / (facts - 1),
                        "target_name": name,
                        "gold": values[target],
                        "context": (
                            "Remember these locker colors.\n"
                            + "\n".join(
                                f"The Unit{index:02d} locker is {color}."
                                for index, color in enumerate(values)
                            )
                            + "\nEnd of records.\n"
                        ),
                        "query": (
                            f"Look up the exact {name} record above. "
                            f"What color is the {name} locker?\n"
                            "Answer with only the color word (red, blue, green, or yellow).\n"
                            "Answer:"
                        ),
                    }
                )
    return result


def validate_data(data):
    if data != make_dataset():
        raise ValueError("Dataset differs from its frozen construction")
    if any(example["seed"] == CONFIRMATION_SEED for example in data):
        raise ValueError("Confirmation seed must remain untouched")
    for example in data:
        prefix = f"The {example['target_name']} locker is "
        matches = [line for line in example["context"].splitlines() if line.startswith(prefix)]
        if (
            len(matches) != 1
            or matches[0].removeprefix(prefix).removesuffix(".") != example["gold"]
        ):
            raise ValueError("Gold does not match the uniquely requested record")


def summarize(rows, status):
    groups = []
    for seed in DEVELOPMENT_SEEDS:
        for facts in (16, 32):
            group = [row for row in rows if row["seed"] == seed and row["fact_count"] == facts]
            if group:
                groups.append(
                    {
                        "seed": seed,
                        "fact_count": facts,
                        "n": len(group),
                        "correct": sum(row["correct"] for row in group),
                        "raw_valid": sum(row["raw_greedy_text"].strip() in COLORS for row in group),
                        "raw_correct": sum(
                            row["raw_greedy_text"].strip() == row["gold"] for row in group
                        ),
                    }
                )
    complete = status == "complete" and len(rows) == 32
    gate = complete and all(
        len(seed_rows := [row for row in rows if row["seed"] == seed]) == 16
        and sum(row["correct"] for row in seed_rows) >= 14
        and sum(row["raw_greedy_text"].strip() == row["gold"] for row in seed_rows) >= 14
        and all(row["raw_greedy_text"].strip() in COLORS for row in seed_rows)
        and all(
            sum(row["correct"] for row in seed_rows if row["fact_count"] == facts) >= 7
            for facts in (16, 32)
        )
        for seed in DEVELOPMENT_SEEDS
    )
    gate = gate and all(row["dense_check"] == "passed" for row in rows)
    return {
        "status": status,
        "completed": len(rows),
        "expected": 32,
        "groups": groups,
        "correct": sum(row["correct"] for row in rows),
        "raw_correct": sum(row["raw_greedy_text"].strip() == row["gold"] for row in rows),
        "gate": "passed" if gate else "failed" if complete else "pending",
        "confirmation_seed_consumed": False,
    }


def save(output, rows, status):
    summary = summarize(rows, status)
    write_json(output / "summary.json", summary)
    lines = [
        "# Exact-lookup prompt calibration",
        "",
        f"Status: {status}; {len(rows)}/32 evaluations.",
        "",
        "| Seed | Facts | Correct | Raw correct | Raw valid |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in summary["groups"]:
        lines.append(
            f"| {group['seed']} | {group['fact_count']} | {group['correct']}/{group['n']} | "
            f"{group['raw_correct']}/{group['n']} | {group['raw_valid']}/{group['n']} |"
        )
    lines += [
        "",
        f"Calibration gate: **{summary['gate']}**.",
        "Confirmation seed consumed: **no**.",
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
    examples = {example["id"]: example for example in data}
    inputs = json.loads((output / "inputs.json").read_text())
    tokenized = {item["id"]: item for item in inputs["examples"]}
    rows = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 32
    assert {row["example_id"] for row in rows} == set(examples)
    for row in rows:
        example = examples[row["example_id"]]
        tokens = tokenized[row["example_id"]]
        for field in ("seed", "gold", "fact_count", "target_fact_index", "target_depth"):
            assert row[field] == example[field]
        assert all(
            row[key] == value for key, value in choose(row["color_logits"], row["gold"]).items()
        )
        assert row["budget_tokens"] == len(tokens["context_ids"])
        assert row["scoring_cache_tokens"] == len(tokens["context_ids"]) + len(tokens["query_ids"])
        assert row["dense_check"] == "passed"
        assert row["dense_max_abs_difference"] <= 2e-4
        for color in COLORS:
            assert abs(row["color_logits"][color] - row["dense_color_logits"][color]) <= (
                2e-4 + 1e-5 * abs(row["dense_color_logits"][color])
            )
    assert json.loads((output / "summary.json").read_text()) == summarize(rows, "complete")
    return {"status": "passed", "rows": len(rows), "confirmation_seed_consumed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/exact-lookup"))
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
        raise ValueError("Frozen dataset hash changed")
    snapshot_sources(root, output)
    write_json(
        output / "run.json",
        {
            "model": MODEL,
            "revision": REVISION,
            "development_seeds": list(DEVELOPMENT_SEEDS),
            "confirmation_seed": CONFIRMATION_SEED,
            "confirmation_seed_consumed": False,
            "device": "cpu",
            "dtype": "float32",
            "attention": "eager",
            "python": sys.version,
            "platform": platform.platform(),
            "versions": {
                name: importlib.metadata.version(name) for name in ("torch", "transformers")
            },
            "protocol": "research/exact_lookup_calibration_protocol.md",
            "expected": 32,
            "dataset_sha256": dataset_hash,
        },
    )
    rows = []
    save(output, rows, "running")
    print(f"Exact-lookup calibration: {output / 'report.md'}", flush=True)
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
            choice_ids = [bare[color][0] for color in COLORS]
            tokenized = [tokenize_example(tokenizer, example, "v2") for example in data]
            for example, tokens in zip(data, tokenized):
                expected_context = 161 if example["fact_count"] == 16 else 289
                if len(tokens["context_ids"]) != expected_context or len(tokens["query_ids"]) != 42:
                    raise ValueError("Frozen token lengths changed")
            write_json(output / "inputs.json", {"examples": tokenized, "bare_color_ids": bare})
            with (
                torch.inference_mode(),
                (output / "rows.jsonl").open("w", buffering=1, encoding="utf-8") as handle,
            ):
                for example, tokens in zip(data, tokenized):
                    started = time.perf_counter()
                    adapter = Qwen2CacheAdapter(model)
                    adapter.forward(torch.tensor([tokens["context_ids"]]))
                    logits = adapter.forward(torch.tensor([tokens["query_ids"]]))[0, -1]
                    dense = model(
                        torch.tensor([tokens["context_ids"] + tokens["query_ids"]]),
                        use_cache=False,
                    ).logits[0, -1]
                    torch.testing.assert_close(logits, dense, atol=2e-4, rtol=1e-5)
                    scores = dict(zip(COLORS, logits[choice_ids].tolist()))
                    row = {
                        "example_id": example["id"],
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
                        **choose(scores, example["gold"]),
                        "color_logits": scores,
                        "raw_greedy_id": logits.argmax().item(),
                        "raw_greedy_text": tokenizer.decode([logits.argmax().item()]),
                        "budget_tokens": len(tokens["context_ids"]),
                        "scoring_cache_tokens": len(adapter.positions),
                        "retained_kv_bytes": audit_cache(adapter, len(adapter.positions)),
                        "dense_check": "passed",
                        "dense_max_abs_difference": (logits - dense).abs().max().item(),
                        "dense_color_logits": dict(zip(COLORS, dense[choice_ids].tolist())),
                        "seconds": time.perf_counter() - started,
                    }
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                    rows.append(row)
                    save(output, rows, "running")
                    print(
                        f"{len(rows)}/32: {example['id']} pred={row['prediction']} gold={row['gold']}",
                        flush=True,
                    )
                    del adapter, logits, dense
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
