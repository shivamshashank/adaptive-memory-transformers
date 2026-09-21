"""Diagnose saved v2 results without changing data or rerunning model inference."""

import argparse
import json
import os
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def classify_answer(text):
    """Next-token validity only: do not label a multi-token completion invalid."""
    return text.strip() in ("A", "B", "C", "D")


def span_indices(offsets, start, end):
    """Tokens whose text overlaps a half-open character span."""
    return [i for i, (a, b) in enumerate(offsets) if a < end and b > start]


def analyze(data, rows, token_spans):
    examples = {x["id"]: x for x in data}
    full = [r for r in rows if r["policy"] == "full"]
    breakdown = {}
    for field in ("gold", "target_fact_index", "target_color"):
        cells = defaultdict(list)
        for row in full:
            cells[str(examples[row["example_id"]][field])].append(row)
        breakdown[field] = {
            k: {"correct": sum(r["correct"] for r in v), "n": len(v)}
            for k, v in sorted(cells.items())
        }
    failures = []
    for row in full:
        if row["correct"]:
            continue
        example = examples[row["example_id"]]
        predicted_color = example["options"]["ABCD".index(row["prediction"])]
        colors = [
            line.split(" locker is ")[1].rstrip(".")
            for line in example["context"].splitlines()
            if " locker is " in line
        ]
        failures.append(
            {
                "id": row["example_id"],
                "gold": row["gold"],
                "prediction": row["prediction"],
                "target_position": example["target_fact_index"],
                "target_fact": example["context"].splitlines()[1 + example["target_fact_index"]],
                "predicted_color": predicted_color,
                "other_positions_with_predicted_color": [
                    i for i, c in enumerate(colors) if c == predicted_color
                ],
                "prediction_minus_gold_logit": row["choice_logits"][row["prediction"]]
                - row["choice_logits"][row["gold"]],
                "raw_token": row["raw_greedy_text"],
                "context": example["context"],
                "query": example["query"],
                "choice_logits": row["choice_logits"],
            }
        )
    groups = defaultdict(list)
    augmented = []
    for row in rows:
        spans = token_spans[row["example_id"]]
        retained = set(row["retained_positions"])
        extra = {
            "id": row["example_id"],
            "policy": row["policy"],
            "ratio": row["ratio"],
            "correct": row["correct"],
            "prediction": row["prediction"],
            "valid_next_token": classify_answer(row["raw_greedy_text"]),
            "raw_token": row["raw_greedy_text"],
            "position_zero_retained": 0 in retained,
            "header_retained": sorted(retained & set(spans["header"])),
            "header_total": len(spans["header"]),
            "target_fact_retained": sorted(retained & set(spans["target_fact"])),
            "target_fact_total": len(spans["target_fact"]),
        }
        augmented.append(extra)
        groups[(row["policy"], row["ratio"])].append(extra)
    summaries = []
    for (policy, ratio), group in groups.items():
        summaries.append(
            {
                "policy": policy,
                "ratio": ratio,
                "n": len(group),
                "valid_next_tokens": sum(r["valid_next_token"] for r in group),
                "correct": sum(r["correct"] for r in group),
                "predictions": dict(Counter(r["prediction"] for r in group)),
                "position_zero_retained": sum(r["position_zero_retained"] for r in group),
                "mean_header_retained": sum(len(r["header_retained"]) for r in group) / len(group),
                "whole_target_fact_retained": sum(
                    len(r["target_fact_retained"]) == r["target_fact_total"] for r in group
                ),
            }
        )
    contingency = []
    for retained in (False, True):
        group = [
            r
            for r in augmented
            if r["policy"] != "full" and r["position_zero_retained"] == retained
        ]
        contingency.append(
            {
                "position_zero_retained": retained,
                "n": len(group),
                "valid_next_tokens": sum(r["valid_next_token"] for r in group),
            }
        )
    return {
        "full_breakdown": breakdown,
        "full_failures": failures,
        "groups": summaries,
        "compressed_position_zero_association": contingency,
        "rows": augmented,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.update(
        {
            "HF_HOME": str(root / ".model-cache/huggingface"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    from transformers import AutoTokenizer
    from verify_pilot import verify

    verification = verify(args.run)
    data = json.loads((args.run / "dataset.json").read_text())
    inputs = {x["id"]: x for x in json.loads((args.run / "inputs.json").read_text())["examples"]}
    settings = json.loads((args.run / "run.json").read_text())
    rows = [json.loads(line) for line in (args.run / "rows.jsonl").read_text().splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(
        settings["model"], revision=settings["revision"], local_files_only=True
    )
    spans = {}
    for example in data:
        saved = inputs[example["id"]]
        prefix = saved["context_prefix"]
        encoded = tokenizer(prefix, add_special_tokens=False, return_offsets_mapping=True)
        assert encoded["input_ids"] == saved["context_ids"]
        start = prefix.index(example["context"])
        fact = example["context"].splitlines()[1 + example["target_fact_index"]]
        fact_start = prefix.index(fact, start)
        spans[example["id"]] = {
            "header": span_indices(encoded["offset_mapping"], 0, start),
            "target_fact": span_indices(
                encoded["offset_mapping"], fact_start, fact_start + len(fact)
            ),
            "tokens": [
                {"position": i, "id": token, "text": tokenizer.decode([token])}
                for i, token in enumerate(saved["context_ids"])
            ],
        }
    result = analyze(data, rows, spans)
    result.update(
        {
            "source_run": str(args.run.resolve()),
            "source_verification": verification,
            "dataset_sha256": settings["dataset_sha256"],
            "token_spans": spans,
            "scope": "Post-hoc descriptive diagnosis, not causal inference or a new benchmark",
        }
    )
    output = (
        root
        / "results/diagnostics"
        / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# V2 failure and prefix diagnosis",
        "",
        f"Source: `{args.run}`",
        "",
        "Saved evidence independently verified; no model inference or result changes.",
        "",
        "## Full-cache failures",
        "",
        "| ID suffix | Fact | Gold | Predicted | Logit gap |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for r in result["full_failures"]:
        lines.append(
            f"| {r['id'].split('-')[-1]} | {r['target_fact']} | {r['gold']} | {r['prediction']} ({r['predicted_color']}) | {r['prediction_minus_gold_logit']:.3f} |"
        )
    lines += [
        "",
        "## Full-cache breakdown (descriptive only)",
        "",
        "```json",
        json.dumps(result["full_breakdown"], indent=2),
        "```",
        "",
        "## Cache and answer-format comparison",
        "",
        "| Policy | Budget | Correct | Valid next letter | Retained position 0 | Mean header tokens | Whole target fact retained |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for g in result["groups"]:
        lines.append(
            f"| {g['policy']} | {g['ratio']} | {g['correct']}/{g['n']} | {g['valid_next_tokens']}/{g['n']} | {g['position_zero_retained']}/{g['n']} | {g['mean_header_retained']:.2f} | {g['whole_target_fact_retained']}/{g['n']} |"
        )
    lines += [
        "",
        "## Limits",
        "",
        "Prefix retention and validity are correlated with policy choice; this is not a causal test.",
        "Fact position and entity identity are confounded in v2. Four examples per entity/depth are too few for general conclusions.",
        "Whole-fact token retention is diagnostic only: retained hidden states can encode other tokens' information.",
        "Invalid next letter does not establish that a longer generated completion would be invalid.",
        "All examples are previously inspected development data. No test-set or significance claim.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    print(output / "report.md")
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("full_breakdown", "groups", "compressed_position_zero_association")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
