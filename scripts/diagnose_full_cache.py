"""Read-only model diagnosis of the frozen development pilot; saves a new run."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(
    {
        "HF_HOME": str(ROOT / ".model-cache/huggingface"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
)


def main() -> None:
    # Import after offline/cache settings have been established.
    import torch
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    from amt.cache import Qwen2CacheAdapter
    from amt.validation import write_json

    source = ROOT / "results/evaluation/20260918T232554Z-e48814cb"
    output = ROOT / "results/diagnostics" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    print(f"Diagnosis: {output}", flush=True)
    examples = json.loads((source / "dataset.json").read_text())
    inputs = {x["id"]: x for x in json.loads((source / "inputs.json").read_text())["examples"]}
    prior = {
        x["example_id"]: x
        for x in map(json.loads, (source / "rows.jsonl").read_text().splitlines())
        if x["policy"] == "full"
    }
    settings = json.loads((source / "run.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        settings["model"], revision=settings["revision"], local_files_only=True
    )
    model = Qwen2ForCausalLM.from_pretrained(
        settings["model"],
        revision=settings["revision"],
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    candidate_ids = {
        prefix: [tokenizer.encode(prefix + letter, add_special_tokens=False) for letter in "ABCD"]
        for prefix in ("", " ")
    }
    assert all(len(ids) == 1 for group in candidate_ids.values() for ids in group)

    def describe(logits):
        result = {
            "raw_token_id": logits.argmax().item(),
            "raw_text": tokenizer.decode([logits.argmax().item()]),
        }
        for prefix, group in candidate_ids.items():
            scores = logits[[ids[0] for ids in group]].tolist()
            result["bare" if prefix == "" else "spaced"] = {
                "prediction": "ABCD"[max(range(4), key=scores.__getitem__)],
                "scores": scores,
            }
        return result

    rows = []
    with torch.inference_mode():
        for example in examples:
            key = example["id"]
            saved = inputs[key]
            # Check gold directly from the text, not generator target metadata.
            name = example["query"].split(" locker?")[0].removeprefix("What color is the ")
            fact = next(
                line
                for line in example["context"].splitlines()
                if line.startswith(f"The {name} locker is ")
            )
            color = fact.removeprefix(f"The {name} locker is ").removesuffix(".")
            parsed_gold = "ABCD"[example["options"].index(color)]
            assert parsed_gold == example["gold"]
            combined = saved["context_ids"] + saved["query_ids"]
            assert (
                tokenizer.encode(example["context"] + example["query"], add_special_tokens=False)
                == combined
            )
            assert tokenizer.decode(combined) == example["context"] + example["query"]
            dense = model(torch.tensor([combined]), use_cache=False).logits[0, -1]
            adapter = Qwen2CacheAdapter(model)
            adapter.forward(torch.tensor([saved["context_ids"]]))
            split = adapter.forward(torch.tensor([saved["query_ids"]]))[0, -1]
            close = torch.allclose(dense, split, atol=2e-4, rtol=1e-5)
            # A separate diagnostic condition, never substituted into the old pilot.
            chat_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": example["context"] + example["query"]}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            chat = model(torch.tensor([chat_ids]), use_cache=False).logits[0, -1]
            row = {
                "id": key,
                "gold": parsed_gold,
                "previous_prediction": prior[key]["prediction"],
                "dense": describe(dense),
                "split": describe(split),
                "chat": describe(chat),
                "dense_split_close": close,
                "max_abs_difference": (dense - split).abs().max().item(),
                "prior_choice_max_abs_difference": max(
                    abs(a - b)
                    for a, b in zip(
                        describe(split)["bare"]["scores"], prior[key]["choice_logits"].values()
                    )
                ),
                "chat_input_ids": chat_ids,
            }
            rows.append(row)
            write_json(
                output / "diagnosis.json",
                {
                    "source": str(source),
                    "dataset_sha256": hashlib.sha256(
                        (source / "dataset.json").read_bytes()
                    ).hexdigest(),
                    "settings": settings,
                    "candidate_ids": candidate_ids,
                    "rows": rows,
                    "complete": len(rows) == len(examples),
                },
            )
            print(
                f"{len(rows)}/20 {key}: gold={parsed_gold} plain={row['dense']['bare']['prediction']} "
                f"chat={row['chat']['bare']['prediction']} dense/split={close}",
                flush=True,
            )
            del adapter, dense, split, chat
    for condition in ("dense", "split", "chat"):
        for form in ("bare", "spaced"):
            correct = sum(r[condition][form]["prediction"] == r["gold"] for r in rows)
            print(f"{condition} {form}: {correct}/20", flush=True)


if __name__ == "__main__":
    main()
