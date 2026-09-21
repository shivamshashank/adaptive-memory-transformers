"""Frozen CPU-only full-cache diagnostic; no downloads or compression claims."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from amt.evaluation import audit_cache, make_examples, score_choices, tokenize_example
from amt.validation import snapshot_sources, write_json

ORIGINAL = ("Aster", "Birch", "Cedar", "Dahlia", "Elm", "Fern", "Grove", "Hazel")
RENAMED = ("Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry")
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def make_controls():
    bases = {}
    for example in make_examples(32, 20260920, "v2"):
        bases.setdefault(example["target_fact_index"], example)
    result = []
    for position, base in sorted(bases.items()):
        for vocabulary, names in (("original", ORIGINAL), ("renamed", RENAMED)):
            context = base["context"]
            for old, new in zip(ORIGINAL, names):
                context = context.replace(f"The {old} locker", f"The {new} locker")
            for rotation in range(4):
                options = base["options"][rotation:] + base["options"][:rotation]
                query = (
                    f"What color is the {names[position]} locker?\n"
                    + "\n".join(f"{letter}: {color}" for letter, color in zip("ABCD", options))
                    + "\nAnswer with only the correct letter.\nAnswer:"
                )
                result.append(
                    base
                    | {
                        "id": f"reliability-{position}-{vocabulary}-{rotation}",
                        "base_id": base["id"],
                        "vocabulary": vocabulary,
                        "rotation": rotation,
                        "context": context,
                        "query": query,
                        "options": options,
                        "gold": "ABCD"[options.index(base["target_color"])],
                    }
                )
    return result


def validate_data(data):
    if data != make_controls():
        raise ValueError("Dataset differs from the frozen control construction")
    for example in data:
        target = example["query"].split(" locker?")[0].removeprefix("What color is the ")
        prefix = f"The {target} locker is "
        facts = [line for line in example["context"].splitlines() if line.startswith(prefix)]
        if len(facts) != 1:
            raise ValueError("Target must have exactly one fact")
        color = facts[0].removeprefix(prefix).removesuffix(".")
        if color != example["options"]["ABCD".index(example["gold"])]:
            raise ValueError("Gold disagrees with parsed fact")


def metrics(rows):
    return {
        "n": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "valid": sum(r["raw_greedy_text"].strip() in tuple("ABCD") for r in rows),
        "raw_correct": sum(r["raw_greedy_text"].strip() == r["gold"] for r in rows),
    }


def summarize(rows, status):
    vocab = {v: metrics([r for r in rows if r["vocabulary"] == v]) for v in ("original", "renamed")}
    lookup = {(r["target_fact_index"], r["vocabulary"], r["rotation"]): r for r in rows}
    pairs = []
    for position in range(8):
        for rotation in range(4):
            off = lookup.get((position, "original", rotation))
            on = lookup.get((position, "renamed", rotation))
            if off is not None and on is not None:
                pairs.append(
                    {
                        "position": position,
                        "rotation": rotation,
                        "accuracy_delta": int(on["correct"]) - int(off["correct"]),
                        "same_color": on["predicted_color"] == off["predicted_color"],
                    }
                )
    rotations = []
    for position in range(8):
        for vocabulary in vocab:
            group = [
                r
                for r in rows
                if r["target_fact_index"] == position and r["vocabulary"] == vocabulary
            ]
            if len(group) == 4:
                rotations.append(
                    {
                        "position": position,
                        "vocabulary": vocabulary,
                        "unique_predicted_colors": len({r["predicted_color"] for r in group}),
                        "correct": sum(r["correct"] for r in group),
                    }
                )
    passed = (
        status == "complete"
        and len(rows) == 64
        and all(
            m["n"] == 32 and m["correct"] >= 29 and m["raw_correct"] >= 29 for m in vocab.values()
        )
        and all(r["dense_check"] == "passed" for r in rows)
    )
    return {
        "status": status,
        "completed": len(rows),
        "expected": 64,
        "overall": metrics(rows),
        "by_vocabulary": vocab,
        "by_gold": {c: metrics([r for r in rows if r["gold"] == c]) for c in "ABCD"},
        "renamed_pairs": pairs,
        "rotation_groups": rotations,
        "longer_context_gate": "passed"
        if passed
        else "failed"
        if status == "complete"
        else "pending",
    }


def save(output, rows, status):
    summary = summarize(rows, status)
    write_json(output / "summary.json", summary)
    lines = [
        "# Full-cache reliability diagnostic",
        "",
        f"Status: {status}; {len(rows)}/64 cases.",
        "",
        "| Vocabulary | Correct | Valid next letter | Raw next correct |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, m in summary["by_vocabulary"].items():
        lines.append(
            f"| {name} | {m['correct']}/{m['n']} | {m['valid']}/{m['n']} | {m['raw_correct']}/{m['n']} |"
        )
    lines += [
        "",
        f"Longer-context gate: **{summary['longer_context_gate']}**.",
        "",
        "64 paired development cases from eight fact sets, not independent held-out trials.",
        "Renaming also changes tokenization. No mechanism or significance claim.",
        "Inputs, raw predictions, choice logits, dense checks and logs are saved alongside this report.",
        "See summary.json for per-label metrics, rotation consistency and renamed pairs.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def verify(output):
    data = json.loads((output / "dataset.json").read_text())
    validate_data(data)
    rows = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
    settings = json.loads((output / "run.json").read_text())
    assert (
        settings["dataset_sha256"]
        == hashlib.sha256((output / "dataset.json").read_bytes()).hexdigest()
    )
    assert len(rows) == 64 and len({r["example_id"] for r in rows}) == 64
    examples = {e["id"]: e for e in data}
    inputs = json.loads((output / "inputs.json").read_text())
    tokens = {t["id"]: t for t in inputs["examples"]}
    assert set(tokens) == set(examples) == {r["example_id"] for r in rows}
    for row in rows:
        example = examples[row["example_id"]]
        for field in ("gold", "vocabulary", "rotation", "target_fact_index"):
            assert row[field] == example[field]
        expected = score_choices([row["choice_logits"][c] for c in "ABCD"], example["gold"])
        assert all(row[k] == v for k, v in expected.items())
        assert row["predicted_color"] == example["options"]["ABCD".index(row["prediction"])]
        assert row["dense_check"] == "passed"
        for c in "ABCD":
            assert abs(
                row["choice_logits"][c] - row["dense_choice_logits"][c]
            ) <= 2e-4 + 1e-5 * abs(row["dense_choice_logits"][c])
        t = tokens[row["example_id"]]
        assert row["context_tokens"] == len(t["context_ids"])
        assert row["scoring_cache_tokens"] == len(t["context_ids"]) + len(t["query_ids"])
    assert json.loads((output / "summary.json").read_text()) == summarize(rows, "complete")
    return {
        "status": "passed",
        "rows": len(rows),
        "scope": "Saved consistency; not an independent model rerun",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/reliability"))
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("Do not use Python -O for evidence checks")
    if args.verify:
        print(json.dumps(verify(args.verify), indent=2))
        return 0
    root = Path(__file__).resolve().parents[1]
    os.environ["HF_HOME"] = str(root / ".model-cache/huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    output = args.output_dir.resolve() / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    output.mkdir(parents=True, exist_ok=False)
    data = make_controls()
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
            "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers")},
            "protocol": "research/reliability_protocol.md",
            "expected": 64,
            "dataset_sha256": hashlib.sha256((output / "dataset.json").read_bytes()).hexdigest(),
        },
    )
    rows = []
    save(output, rows, "running")
    print(f"Reliability report: {output / 'report.md'}", flush=True)
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
            candidates = [tokenizer.encode(c, add_special_tokens=False) for c in "ABCD"]
            if any(len(c) != 1 for c in candidates):
                raise ValueError("Single-token A-D required")
            choice_ids = [c[0] for c in candidates]
            tokenized = [tokenize_example(tokenizer, e, "v2") for e in data]
            write_json(output / "inputs.json", {"examples": tokenized, "choice_ids": choice_ids})
            with torch.inference_mode(), (output / "rows.jsonl").open("w", buffering=1) as handle:
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
                    scored = score_choices(logits[choice_ids].tolist(), example["gold"])
                    raw_id = logits.argmax().item()
                    row = {
                        "example_id": example["id"],
                        **{
                            k: example[k]
                            for k in ("gold", "vocabulary", "rotation", "target_fact_index")
                        },
                        **scored,
                        "predicted_color": example["options"]["ABCD".index(scored["prediction"])],
                        "raw_greedy_id": raw_id,
                        "raw_greedy_text": tokenizer.decode([raw_id]),
                        "dense_choice_logits": dict(zip("ABCD", dense[choice_ids].tolist())),
                        "dense_max_abs_difference": (logits - dense).abs().max().item(),
                        "dense_check": "passed",
                        "context_tokens": len(tokens["context_ids"]),
                        "scoring_cache_tokens": len(adapter.positions),
                        "scoring_kv_bytes": cache_bytes,
                        "seconds": time.perf_counter() - started,
                    }
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                    rows.append(row)
                    save(output, rows, "running")
                    print(
                        f"{len(rows)}/64: {example['id']} pred={row['prediction']} gold={row['gold']}",
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
