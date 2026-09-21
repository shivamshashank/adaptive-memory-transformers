"""Model-backed equivalence tests for standard and manual cached decoding."""

import os
from collections.abc import Iterator

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

from amt.decoding import (
    DecodeTrace,
    run_manual_full_cache_decode,
    run_reference_greedy_decode,
    tolerance_for_dtype,
)


@pytest.fixture
def tiny_gpt2() -> Iterator[GPT2LMHeadModel]:
    torch.manual_seed(7)
    config = GPT2Config(
        vocab_size=64,
        n_positions=64,
        n_ctx=64,
        n_embd=32,
        n_layer=2,
        n_head=4,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
        use_cache=True,
    )
    model = GPT2LMHeadModel(config).eval()
    yield model


def _assert_equivalent_traces(
    manual: DecodeTrace,
    reference: DecodeTrace,
    dtype: torch.dtype,
) -> None:
    assert len(manual.step_logits) == len(reference.step_logits)

    tolerance = tolerance_for_dtype(dtype)
    for step, (manual_logits, reference_logits) in enumerate(
        zip(manual.step_logits, reference.step_logits, strict=True),
        start=1,
    ):
        try:
            torch.testing.assert_close(
                manual_logits.float(),
                reference_logits.float(),
                atol=tolerance.absolute,
                rtol=tolerance.relative,
            )
        except AssertionError as exc:
            raise AssertionError(f"logits diverged at generation step {step}") from exc

    assert torch.equal(manual.sequences, reference.sequences)
    assert manual.cache_snapshots[-1] == reference.cache_snapshots[-1]


def test_manual_full_cache_decode_matches_transformers_generation(
    tiny_gpt2: GPT2LMHeadModel,
) -> None:
    input_ids = torch.tensor([[1, 5, 9, 13, 17]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    max_new_tokens = 4

    reference = run_reference_greedy_decode(
        tiny_gpt2,
        input_ids,
        attention_mask,
        max_new_tokens,
    )
    manual = run_manual_full_cache_decode(
        tiny_gpt2,
        input_ids,
        attention_mask,
        max_new_tokens,
    )

    assert len(reference.step_logits) == max_new_tokens
    assert len(manual.step_logits) == max_new_tokens
    _assert_equivalent_traces(manual, reference, tiny_gpt2.dtype)


def test_manual_decode_records_expected_cache_growth_and_bytes(
    tiny_gpt2: GPT2LMHeadModel,
) -> None:
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    trace = run_manual_full_cache_decode(
        tiny_gpt2,
        input_ids,
        torch.ones_like(input_ids),
        max_new_tokens=3,
    )

    assert [snapshot.sequence_length for snapshot in trace.cache_snapshots] == [3, 4, 5]
    for snapshot in trace.cache_snapshots:
        expected_shape = (1, 4, snapshot.sequence_length, 8)
        assert len(snapshot.layers) == 2
        assert all(layer.keys == expected_shape for layer in snapshot.layers)
        assert all(layer.values == expected_shape for layer in snapshot.layers)

        expected_bytes = 2 * 2 * 4 * snapshot.sequence_length * 8 * 4
        assert snapshot.total_bytes == expected_bytes


@pytest.mark.parametrize(
    ("dtype", "absolute", "relative"),
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ],
)
def test_numerical_tolerances_are_declared_by_dtype(
    dtype: torch.dtype,
    absolute: float,
    relative: float,
) -> None:
    tolerance = tolerance_for_dtype(dtype)

    assert tolerance.absolute == absolute
    assert tolerance.relative == relative


def test_decode_rejects_invalid_inputs(tiny_gpt2: GPT2LMHeadModel) -> None:
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with pytest.raises(ValueError, match="max_new_tokens"):
        run_manual_full_cache_decode(
            tiny_gpt2,
            input_ids,
            torch.ones_like(input_ids),
            max_new_tokens=0,
        )

    with pytest.raises(ValueError, match="attention_mask"):
        run_reference_greedy_decode(
            tiny_gpt2,
            input_ids,
            torch.ones((1, 2), dtype=torch.long),
            max_new_tokens=1,
        )


@pytest.mark.model_download
@pytest.mark.skipif(
    os.environ.get("AMT_RUN_MODEL_INTEGRATION") != "1",
    reason="set AMT_RUN_MODEL_INTEGRATION=1 to download the pinned tiny model",
)
def test_pinned_tiny_gpt2_matches_transformers_generation() -> None:
    model_id = "hf-internal-testing/tiny-random-gpt2"
    revision = "71034c5d8bde858ff824298bdedc65515b97d2b9"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision).eval()
    inputs = tokenizer("A small deterministic cache test", return_tensors="pt")

    reference = run_reference_greedy_decode(
        model,
        inputs["input_ids"],
        inputs["attention_mask"],
        max_new_tokens=4,
    )
    manual = run_manual_full_cache_decode(
        model,
        inputs["input_ids"],
        inputs["attention_mask"],
        max_new_tokens=4,
    )

    _assert_equivalent_traces(manual, reference, model.dtype)


@pytest.mark.model_download
@pytest.mark.skipif(
    os.environ.get("AMT_RUN_QWEN_INTEGRATION") != "1",
    reason="set AMT_RUN_QWEN_INTEGRATION=1 to download the pinned Qwen model",
)
def test_pinned_qwen_matches_transformers_generation() -> None:
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    revision = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    device = os.environ.get("AMT_MODEL_DEVICE", "cpu")
    dtype_name = os.environ.get("AMT_MODEL_DTYPE", "float32")
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtypes[dtype_name]

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
    ).eval()
    model.to(device)
    inputs = tokenizer("A deterministic Qwen cache test", return_tensors="pt").to(device)

    reference = run_reference_greedy_decode(
        model,
        inputs["input_ids"],
        inputs["attention_mask"],
        max_new_tokens=3,
    )
    manual = run_manual_full_cache_decode(
        model,
        inputs["input_ids"],
        inputs["attention_mask"],
        max_new_tokens=3,
    )

    _assert_equivalent_traces(manual, reference, model.dtype)
