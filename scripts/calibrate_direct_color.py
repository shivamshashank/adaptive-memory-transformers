"""Frozen direct-color full-cache calibration using only cached model files."""

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

from amt.evaluation import audit_cache, make_examples, tokenize_example
from amt.validation import snapshot_sources, write_json

COLORS = ("red", "blue", "green", "yellow")
ORIGINAL = ("Aster", "Birch", "Cedar", "Dahlia", "Elm", "Fern", "Grove", "Hazel")
RENAMED = ("Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry")
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def make_calibration():
    result = []
    for base in make_examples(32, 20260920, "v2"):
        position = base["target_fact_index"]
        for vocabulary, names in (("original", ORIGINAL), ("renamed", RENAMED)):
            context = base["context"]
            for old, new in zip(ORIGINAL, names):
                context = context.replace(f"The {old} locker", f"The {new} locker")
            result.append(
                {
                    "id": f"direct-{base['id']}-{vocabulary}",
                    "base_id": base["id"],
                    "split": "development",
                    "vocabulary": vocabulary,
                    "context": context,
                    "query": (
                        f"What color is the {names[position]} locker?\n"
                        "Answer with only the color word (red, blue, green, or yellow).\n"
                        "Answer:"
                    ),
                    "gold": base["target_color"],
                    "target_fact_index": position,
                }
            )
    return result


def validate_data(data):
    if data != make_calibration():
        raise ValueError("Dataset differs from frozen direct-color construction")
    for example in data:
        target = example["query"].split(" locker?")[0].removeprefix("What color is the ")
        prefix = f"The {target} locker is "
        facts = [line for line in example["context"].splitlines() if line.startswith(prefix)]
        if len(facts) != 1 or facts[0].removeprefix(prefix).removesuffix(".") != example["gold"]:
            raise ValueError("Gold does not match the uniquely requested fact")
        if example["gold"] not in COLORS:
            raise ValueError("Gold must be a declared color")


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
        raise ValueError("Four finite numeric color scores and a valid gold color required")
    prediction = max(COLORS, key=scores.__getitem__)
    return {"prediction": prediction, "correct": prediction == gold}


def group_metrics(rows):
    return {
        "n": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "raw_valid": sum(r["raw_greedy_text"].strip() in COLORS for r in rows),
        "raw_correct": sum(r["raw_greedy_text"].strip() == r["gold"] for r in rows),
        "space_agreement": sum(r["space_prediction"] == r["prediction"] for r in rows),
    }


def summarize(rows, status):
    by_vocabulary = {
        name: group_metrics([r for r in rows if r["vocabulary"] == name])
        for name in ("original", "renamed")
    }
    lookup = {(r["base_id"], r["vocabulary"]): r for r in rows}
    pairs = []
    for base_id in sorted({r["base_id"] for r in rows}):
        original = lookup.get((base_id, "original"))
        renamed = lookup.get((base_id, "renamed"))
        if original is not None and renamed is not None:
            pairs.append(
                {
                    "base_id": base_id,
                    "accuracy_delta": int(renamed["correct"]) - int(original["correct"]),
                    "same_prediction": renamed["prediction"] == original["prediction"],
                    "same_raw": renamed["raw_greedy_text"].strip()
                    == original["raw_greedy_text"].strip(),
                }
            )
    passed = (
        status == "complete"
        and len(rows) == 64
        and all(
            metrics["n"] == 32
            and metrics["correct"] >= 29
            and metrics["raw_valid"] == 32
            and metrics["raw_correct"] >= 29
            for metrics in by_vocabulary.values()
        )
        and all(r["dense_check"] == "passed" for r in rows)
    )
    return {
        "status": status,
        "completed": len(rows),
        "expected": 64,
        "overall": group_metrics(rows),
        "by_vocabulary": by_vocabulary,
        "by_gold": {
            color: group_metrics([r for r in rows if r["gold"] == color]) for color in COLORS
        },
        "renamed_pairs": pairs,
        "task_gate": "passed" if passed else "failed" if status == "complete" else "pending",
    }


def save(output, rows, status):
    summary = summarize(rows, status)
    write_json(output / "summary.json", summary)
    lines = [
        "# Direct-color baseline calibration",
        "",
        f"Status: {status}; {len(rows)}/64 cases.",
        "",
        "| Vocabulary | Constrained correct | Valid raw color | Raw correct | Bare/space agree |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in summary["by_vocabulary"].items():
        lines.append(
            f"| {name} | {metrics['correct']}/{metrics['n']} | "
            f"{metrics['raw_valid']}/{metrics['n']} | {metrics['raw_correct']}/{metrics['n']} | "
            f"{metrics['space_agreement']}/{metrics['n']} |"
        )
    lines += [
        "",
        f"Task gate: **{summary['task_gate']}**.",
        "",
        "Development calibration on paired name variants, not held-out confirmation.",
        "Constrained correctness uses frozen bare single-token color scores.",
        "See summary.json for color and rename breakdowns; exact evidence is saved locally.",
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
    assert settings["colors"] == list(COLORS)
    inputs = json.loads((output / "inputs.json").read_text())
    assert all(len(ids) == 1 for ids in inputs["bare_color_ids"].values())
    assert all(len(ids) == 1 for ids in inputs["space_color_ids"].values())
    tokens = {item["id"]: item for item in inputs["examples"]}
    examples = {item["id"]: item for item in data}
    rows = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 64 and len({r["example_id"] for r in rows}) == 64
    assert set(tokens) == set(examples) == {r["example_id"] for r in rows}
    for row in rows:
        example = examples[row["example_id"]]
        for field in ("base_id", "vocabulary", "gold", "target_fact_index"):
            assert row[field] == example[field]
        expected = choose(row["color_logits"], example["gold"])
        assert all(row[key] == value for key, value in expected.items())
        assert (
            row["space_prediction"]
            == choose(row["space_color_logits"], example["gold"])["prediction"]
        )
        assert row["dense_check"] == "passed"
        assert math.isfinite(row["dense_max_abs_difference"])
        assert row["dense_max_abs_difference"] >= 0
        for color in COLORS:
            assert abs(row["color_logits"][color] - row["dense_color_logits"][color]) <= (
                2e-4 + 1e-5 * abs(row["dense_color_logits"][color])
            )
            assert abs(
                row["space_color_logits"][color] - row["dense_space_color_logits"][color]
            ) <= 2e-4 + 1e-5 * abs(row["dense_space_color_logits"][color])
        tokenized = tokens[row["example_id"]]
        assert row["context_tokens"] == len(tokenized["context_ids"])
        assert row["scoring_cache_tokens"] == len(tokenized["context_ids"]) + len(
            tokenized["query_ids"]
        )
    assert json.loads((output / "summary.json").read_text()) == summarize(rows, "complete")
    return {
        "status": "passed",
        "rows": len(rows),
        "scope": "Saved evidence consistency, not a model rerun",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/direct-color"))
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
    data = make_calibration()
    validate_data(data)
    write_json(output / "dataset.json", data)
    snapshot_sources(root, output)
    write_json(
        output / "run.json",
        {
            "model": MODEL,
            "revision": REVISION,
            "seed": 20260920,
            "device": "cpu",
            "dtype": "float32",
            "attention": "eager",
            "python": sys.version,
            "platform": platform.platform(),
            "versions": {
                name: importlib.metadata.version(name) for name in ("torch", "transformers")
            },
            "protocol": "research/direct_color_protocol.md",
            "expected": 64,
            "colors": list(COLORS),
            "primary_scorer": "argmax over bare single-token color logits",
            "dataset_sha256": hashlib.sha256((output / "dataset.json").read_bytes()).hexdigest(),
        },
    )
    rows = []
    save(output, rows, "running")
    print(f"Direct-color report: {output / 'report.md'}", flush=True)
    with (
        (output / "run.log").open("w", buffering=1) as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        try:
            import torch
            from transformers import AutoTokenizer, Qwen2ForCausalLM

            from amt.cache import Qwen2CacheAdapter

            torch.manual_seed(20260920)
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
                raise ValueError("Frozen scorer requires one token per color form")
            tokenized = [tokenize_example(tokenizer, example, "v2") for example in data]
            write_json(
                output / "inputs.json",
                {
                    "examples": tokenized,
                    "bare_color_ids": bare,
                    "space_color_ids": spaced,
                },
            )
            bare_ids = [bare[color][0] for color in COLORS]
            space_ids = [spaced[color][0] for color in COLORS]
            with (
                torch.inference_mode(),
                (output / "rows.jsonl").open("w", buffering=1, encoding="utf-8") as handle,
            ):
                for example, tokens in zip(data, tokenized):
                    started = time.perf_counter()
                    adapter = Qwen2CacheAdapter(model)
                    adapter.forward(torch.tensor([tokens["context_ids"]]))
                    logits = adapter.forward(torch.tensor([tokens["query_ids"]]))[0, -1]
                    cache_bytes = audit_cache(
                        adapter, len(tokens["context_ids"]) + len(tokens["query_ids"])
                    )
                    dense = model(
                        torch.tensor([tokens["context_ids"] + tokens["query_ids"]]), use_cache=False
                    ).logits[0, -1]
                    torch.testing.assert_close(logits, dense, atol=2e-4, rtol=1e-5)
                    color_logits = dict(zip(COLORS, logits[bare_ids].tolist()))
                    space_logits = dict(zip(COLORS, logits[space_ids].tolist()))
                    dense_colors = dict(zip(COLORS, dense[bare_ids].tolist()))
                    dense_spaces = dict(zip(COLORS, dense[space_ids].tolist()))
                    primary = choose(color_logits, example["gold"])
                    raw_id = logits.argmax().item()
                    row = {
                        "example_id": example["id"],
                        **{
                            key: example[key]
                            for key in ("base_id", "vocabulary", "gold", "target_fact_index")
                        },
                        **primary,
                        "space_prediction": choose(space_logits, example["gold"])["prediction"],
                        "color_logits": color_logits,
                        "space_color_logits": space_logits,
                        "dense_color_logits": dense_colors,
                        "dense_space_color_logits": dense_spaces,
                        "raw_greedy_id": raw_id,
                        "raw_greedy_text": tokenizer.decode([raw_id]),
                        "dense_check": "passed",
                        "dense_max_abs_difference": (logits - dense).abs().max().item(),
                        "context_tokens": len(tokens["context_ids"]),
                        "query_tokens": len(tokens["query_ids"]),
                        "scoring_cache_tokens": len(adapter.positions),
                        "scoring_kv_bytes": cache_bytes,
                        "seconds": time.perf_counter() - started,
                    }
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                    rows.append(row)
                    save(output, rows, "running")
                    print(
                        f"{len(rows)}/64: {example['id']} pred={row['prediction']} "
                        f"raw={row['raw_greedy_text']!r} gold={row['gold']}",
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
