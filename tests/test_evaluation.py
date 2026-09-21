import math

import pytest

from amt.evaluation import (
    audit_cache,
    make_examples,
    save_report,
    score_choices,
    summarize,
    tokenize_example,
)


def test_v2_cross_balances_depth_and_label():
    from collections import Counter

    data = make_examples(32, version="v2")
    assert data == make_examples(32, version="v2")
    assert data != make_examples(32, seed=1, version="v2")
    assert Counter((x["target_fact_index"], x["gold"]) for x in data) == {
        (position, label): 1 for position in range(8) for label in "ABCD"
    }
    assert not {x["id"] for x in data} & {x["id"] for x in make_examples()}
    for x in data:
        assert x["options"]["ABCD".index(x["gold"])] == x["target_color"]
        assert (
            x["context"].splitlines()[x["target_fact_index"] + 1].endswith(x["target_color"] + ".")
        )
    with pytest.raises(ValueError):
        make_examples(20, version="v2")


def test_chat_split_exact_and_question_unseen():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "<user>" + messages[0]["content"] + "</user><assistant>"

        def encode(self, text, **kwargs):
            return list(text.encode())

    example = make_examples(32, version="v2")[0]
    result = tokenize_example(Tokenizer(), example, "v2")
    assert bytes(result["context_ids"] + result["query_ids"]).decode() == result["rendered_prompt"]
    assert example["query"] not in result["context_prefix"]
    assert bytes(result["query_ids"]).decode().startswith(example["query"])


def test_chat_split_rejects_changed_boundary():
    class BadTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def encode(self, text, **kwargs):
            return [len(text)]

    with pytest.raises(ValueError, match="boundary"):
        tokenize_example(BadTokenizer(), make_examples()[0], "v2")


def test_dataset_deterministic_balanced_and_consistent():
    examples = make_examples()
    assert examples == make_examples()
    assert examples != make_examples(seed=20)
    assert len({item["id"] for item in examples}) == 20
    assert [item["gold"] for item in examples].count("A") == 5
    for item in examples:
        assert item["options"]["ABCD".index(item["gold"])] == item["target_color"]
        assert item["split"] == "development"
        assert item["target_color"] in item["context"].splitlines()[1 + item["target_fact_index"]]


def test_choice_scoring():
    assert score_choices([1, 4, 2, 3], "B")["correct"]
    assert not score_choices([1, 4, 2, 3], "A")["correct"]
    assert score_choices([1, 1, 1, 1], "A")["prediction"] == "A"
    with pytest.raises(ValueError):
        score_choices([1, math.nan, 2, 3], "A")


def test_summary_keeps_full_failures_and_pairs_by_id(tmp_path):
    rows = [
        dict(
            example_id="a",
            policy="full",
            ratio=None,
            correct=False,
            retained_kv_bytes=100,
            seconds=1,
        ),
        dict(
            example_id="b",
            policy="full",
            ratio=None,
            correct=True,
            retained_kv_bytes=100,
            seconds=1,
        ),
        dict(
            example_id="a",
            policy="adaptive",
            ratio=0.5,
            correct=True,
            retained_kv_bytes=50,
            seconds=1,
        ),
        dict(
            example_id="b",
            policy="adaptive",
            ratio=0.5,
            correct=False,
            retained_kv_bytes=50,
            seconds=1,
        ),
    ]
    summary = summarize(rows)
    assert summary[1]["accuracy"] == 0.5
    assert summary[1]["accuracy_on_full_correct"] == 0
    assert summary[1]["full_correct_n"] == 1
    save_report(tmp_path, rows, 9, "running")
    assert "running" in (tmp_path / "report.md").read_text()


def test_invalid_count():
    with pytest.raises(ValueError):
        make_examples(0)


def test_budget_audit_rejects_wrong_length_and_backing_storage():
    from types import SimpleNamespace

    import torch

    tensor = torch.zeros(1, 2, 8, 4)
    layer = SimpleNamespace(keys=tensor, values=tensor.clone())
    adapter = SimpleNamespace(positions=tuple(range(8)), cache=SimpleNamespace(layers=[layer]))
    assert audit_cache(adapter, 8) == 512
    with pytest.raises(RuntimeError, match="position"):
        audit_cache(adapter, 4)
    adapter.positions = tuple(range(4))
    layer.keys = tensor[:, :, :4, :]
    layer.values = tensor[:, :, :4, :]
    with pytest.raises(RuntimeError, match="backing storage"):
        audit_cache(adapter, 4)


def test_delayed_query_full_matches_dense_and_compressed_budget():
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from amt.cache import Qwen2CacheAdapter
    from amt.policies import RetentionSignals, create_policy

    torch.manual_seed(19)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval()
    context = torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29]])
    query = torch.tensor([[3, 7, 11]])
    with torch.inference_mode():
        dense = model(torch.cat((context, query), dim=1)).logits[:, -1]
        for name in ("full", "recency", "uniform", "attention", "adaptive"):
            adapter = Qwen2CacheAdapter(model)
            adapter.forward(context, collect_attention=name in ("attention", "adaptive"))
            signals = RetentionSignals(adapter.positions, adapter.attention_by_layer)
            adapter.retain(create_policy(name).select_indices(8, 4, signals=signals))
            count = 8 if name == "full" else 4
            audit_cache(adapter, count)
            result = adapter.forward(query)[:, -1]
            audit_cache(adapter, count + 3)
            assert adapter.positions[-3:] == (8, 9, 10)
            if name == "full":
                torch.testing.assert_close(result, dense, atol=1e-5, rtol=1e-5)
