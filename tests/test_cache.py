"""Independent dense masked-cache oracle for physically pruned Qwen caches."""

import os

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2 import modeling_qwen2

from amt.cache import Qwen2CacheAdapter
from amt.compressed import run_compressed_greedy_decode
from amt.decoding import run_reference_greedy_decode, tolerance_for_dtype
from amt.policies import (
    FullCachePolicy,
    RecencyCachePolicy,
    UniformCachePolicy,
    create_policy,
)


@pytest.fixture(params=["eager", "sdpa"])
def qwen(request: pytest.FixtureRequest) -> Qwen2ForCausalLM:
    torch.manual_seed(19)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
        eos_token_id=None,
    )
    config._attn_implementation = request.param
    return Qwen2ForCausalLM(config).eval()


def assert_logits_close(
    actual: torch.Tensor, expected: torch.Tensor, absolute: float | None = None
) -> None:
    tolerance = tolerance_for_dtype(actual.dtype)
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=tolerance.absolute if absolute is None else absolute,
        rtol=tolerance.relative,
    )


@torch.inference_mode()
def assert_dense_masked_equivalence(model: Qwen2ForCausalLM, absolute: float | None = None) -> None:
    """Retain dense historical KVs but hide removed columns on future steps.

    Previously computed KVs must NOT be recomputed after deleting prompt tokens:
    their hidden states legitimately include the earlier, pre-pruning context.
    Dense writes use explicit original positions and an independent visibility
    vector. Compare logits AND all surviving K/V entries after every append.
    """
    adapter = Qwen2CacheAdapter(model)
    dense = DynamicCache(config=model.config)
    visible: list[bool] = []
    processed = 0
    # Early/middle/recent retention; repeat, remove everything, then append again.
    stages = [
        ([1, 5, 9, 13, 17, 21, 25, 29], [0, 3, 7]),
        ([31, 33], [0, 2, 4]),
        ([35], [1, 3]),
        ([37, 39], []),
        ([41], [0]),
    ]
    for tokens, keep in stages:
        ids = torch.tensor([tokens], device=model.device, dtype=torch.long)
        new_count = len(tokens)
        visible.extend([True] * new_count)
        # Dense physical indices equal original positions; construct independently.
        mask = torch.full(
            (1, 1, new_count, processed + new_count),
            float("-inf"),
            dtype=model.dtype,
            device=model.device,
        )
        for query in range(new_count):
            for key in range(processed + query + 1):
                if visible[key]:
                    mask[0, 0, query, key] = 0
        expected = model(
            input_ids=ids,
            position_ids=torch.arange(
                processed, processed + new_count, device=model.device
            ).unsqueeze(0),
            attention_mask=mask,
            past_key_values=dense,
            use_cache=True,
            logits_to_keep=0,
        ).logits
        actual = adapter.forward(ids)
        assert_logits_close(actual, expected, absolute)
        assert torch.equal(actual.argmax(-1), expected.argmax(-1))
        processed += new_count
        assert adapter.seen_tokens == processed
        assert adapter.positions == tuple(i for i, active in enumerate(visible) if active)
        indices = torch.tensor(adapter.positions, device=model.device, dtype=torch.long)
        for compressed_layer, dense_layer in zip(adapter.cache.layers, dense.layers, strict=True):
            assert_logits_close(
                compressed_layer.keys, dense_layer.keys.index_select(-2, indices), absolute
            )
            assert_logits_close(
                compressed_layer.values, dense_layer.values.index_select(-2, indices), absolute
            )
        retained = tuple(adapter.positions[index] for index in keep)
        adapter.retain(keep)
        visible = [index in retained for index in range(processed)]
        assert adapter.positions == retained
        assert adapter.seen_tokens == processed
        for layer in adapter.cache.layers:
            assert layer.keys.shape[-2] == len(keep)
            assert layer.values.shape[-2] == len(keep)
            # No sliced view retains the original full allocation.
            for tensor in (layer.keys, layer.values):
                assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()


def test_pruned_cache_matches_independent_dense_masked_oracle(qwen: Qwen2ForCausalLM) -> None:
    assert_dense_masked_equivalence(qwen)


def test_pruning_keeps_keys_and_values_bitwise_unchanged(qwen: Qwen2ForCausalLM) -> None:
    adapter = Qwen2CacheAdapter(qwen)
    adapter.forward(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]))
    before = [(layer.keys.clone(), layer.values.clone()) for layer in adapter.cache.layers]
    adapter.retain([0, 3, 7])
    assert adapter.positions == (0, 3, 7)
    for layer, (keys, values) in zip(adapter.cache.layers, before, strict=True):
        assert torch.equal(layer.keys, keys[:, :, [0, 3, 7], :])
        assert torch.equal(layer.values, values[:, :, [0, 3, 7], :])
    adapter.retain([1])
    assert adapter.positions == (3,)
    assert adapter.seen_tokens == 8


def assert_full_retention_matches_generation(
    qwen: Qwen2ForCausalLM, absolute: float | None = None
) -> None:
    ids = torch.tensor([[1, 5, 9, 13, 17]], device=qwen.device)
    actual = run_compressed_greedy_decode(qwen, ids, 4, FullCachePolicy(), budget_tokens=1)
    expected = run_reference_greedy_decode(qwen, ids, torch.ones_like(ids), 4)
    assert torch.equal(actual.decode.sequences, expected.sequences)
    for left, right in zip(actual.decode.step_logits, expected.step_logits, strict=True):
        assert_logits_close(left, right, absolute)
    assert actual.decode.cache_snapshots[-1] == expected.cache_snapshots[-1]
    assert actual.retained_positions == tuple(tuple(range(length)) for length in range(5, 9))


def test_full_retention_matches_standard_generation(qwen: Qwen2ForCausalLM) -> None:
    assert_full_retention_matches_generation(qwen)


@pytest.mark.parametrize("policy", [RecencyCachePolicy(), UniformCachePolicy()])
def test_budgeted_generation_tracks_positions_and_bytes(qwen: Qwen2ForCausalLM, policy) -> None:
    ids = torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29]])
    result = run_compressed_greedy_decode(qwen, ids, 5, policy, budget_tokens=3)
    assert result.decode.sequences.shape == (1, 13)
    previous: set[int] = set(range(8))
    for step, (positions, snapshot) in enumerate(
        zip(result.retained_positions, result.decode.cache_snapshots, strict=True)
    ):
        available = previous | ({7 + step} if step else set())
        assert set(positions) <= available
        assert len(positions) == snapshot.sequence_length == 3
        assert snapshot.total_bytes == 2 * 2 * 2 * 3 * 8 * 4
        previous = set(positions)


@pytest.mark.parametrize("indices", [[-1], [8], [1, 1], [3, 0], [True], [1.5]])
def test_invalid_retention_leaves_cache_unchanged(qwen: Qwen2ForCausalLM, indices) -> None:
    adapter = Qwen2CacheAdapter(qwen)
    adapter.forward(torch.tensor([[1, 2, 3, 4]]))
    original = adapter.cache.layers[0].keys.clone()
    with pytest.raises(ValueError):
        adapter.retain(indices)
    assert adapter.positions == (0, 1, 2, 3)
    torch.testing.assert_close(adapter.cache.layers[0].keys, original)


def test_rejects_unsupported_inputs(qwen: Qwen2ForCausalLM) -> None:
    adapter = Qwen2CacheAdapter(qwen)
    for ids in [torch.ones((2, 3), dtype=torch.long), torch.tensor([1]), torch.empty((1, 0))]:
        with pytest.raises(ValueError, match="shape"):
            adapter.forward(ids)
    with pytest.raises(ValueError, match="dtype"):
        adapter.forward(torch.ones((1, 2)))
    qwen.train()
    with pytest.raises(ValueError, match="eval"):
        Qwen2CacheAdapter(qwen)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("layer_types", ["sliding_attention", "full_attention"], "full_attention"),
        ("rope_parameters", {"rope_type": "dynamic"}, "default RoPE"),
        ("_attn_implementation", "flash_attention_2", "eager and sdpa"),
    ],
)
def test_rejects_unvalidated_model_modes(qwen: Qwen2ForCausalLM, field, value, message) -> None:
    setattr(qwen.config, field, value)
    with pytest.raises(ValueError, match=message):
        Qwen2CacheAdapter(qwen)


@pytest.mark.parametrize("new_tokens,budget", [(0, 3), (3, 0), (-1, 3)])
def test_runner_rejects_invalid_limits(qwen: Qwen2ForCausalLM, new_tokens, budget) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        run_compressed_greedy_decode(
            qwen, torch.tensor([[1, 2]]), new_tokens, UniformCachePolicy(), budget
        )


@torch.inference_mode()
def assert_wrong_position_detected(qwen: Qwen2ForCausalLM, absolute: float | None = None) -> None:
    """Mutation control: a renumbered query must differ from the correct result."""
    good = Qwen2CacheAdapter(qwen)
    bad = Qwen2CacheAdapter(qwen)
    ids = torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29]], device=qwen.device)
    good.forward(ids)
    bad.forward(ids)
    good.retain([0, 3, 7])
    bad.retain([0, 3, 7])
    token = torch.tensor([[31]], device=qwen.device)
    correct = good.forward(token)
    # Reproduce the original bug: use retained length (3), not processed length (8).
    wrong = qwen(
        input_ids=token,
        position_ids=torch.tensor([[3]], device=qwen.device),
        attention_mask=torch.zeros((1, 1, 1, 4), device=qwen.device, dtype=qwen.dtype),
        past_key_values=bad.cache,
        use_cache=True,
    ).logits
    with pytest.raises(AssertionError):
        assert_logits_close(correct, wrong, absolute)


def test_oracle_detects_wrong_query_position(qwen: Qwen2ForCausalLM) -> None:
    assert_wrong_position_detected(qwen)


@torch.inference_mode()
def assert_attention_collection_and_generation(model: Qwen2ForCausalLM) -> None:
    ids = torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29]], device=model.device)
    expected = model(input_ids=ids, output_attentions=True, use_cache=True)
    adapter = Qwen2CacheAdapter(model)
    actual = adapter.forward(ids, collect_attention=True)
    assert_logits_close(actual, expected.logits, 2e-4)
    for scores, weights in zip(adapter.attention_by_layer, expected.attentions, strict=True):
        torch.testing.assert_close(
            torch.tensor(scores), weights[0, :, -1, :].float().mean(0).cpu(), atol=1e-6, rtol=1e-5
        )
    mean_adapter = Qwen2CacheAdapter(model)
    mean_adapter.forward(ids, collect_attention=True, attention_query_reduction="mean")
    for scores, weights in zip(mean_adapter.attention_by_layer, expected.attentions, strict=True):
        torch.testing.assert_close(
            torch.tensor(scores), weights[0].float().mean((0, 1)).cpu(), atol=1e-6, rtol=1e-5
        )
    original_scores = adapter.attention_by_layer
    adapter.retain([0, 3, 7])
    assert adapter.attention_by_layer == tuple(
        tuple(layer[i] for i in (0, 3, 7)) for layer in original_scores
    )
    adapter.forward(torch.tensor([[31]], device=model.device), collect_attention=True)
    assert adapter.positions == (0, 3, 7, 8)
    assert all(
        len(layer) == 4 and abs(sum(layer) - 1) < 1e-5 for layer in adapter.attention_by_layer
    )
    adapter.forward(torch.tensor([[33]], device=model.device))
    assert adapter.attention_by_layer == ()  # Never reuse stale scores.

    for name in ("attention", "adaptive"):
        policy = create_policy(name)
        short = run_compressed_greedy_decode(model, ids, 1, policy, 3)
        long = run_compressed_greedy_decode(model, ids, 4, policy, 3)
        assert short.retained_positions[0] == long.retained_positions[0]
        assert_logits_close(short.decode.step_logits[0], long.decode.step_logits[0])
        prior = set(range(8))
        for step, (positions, snapshot) in enumerate(
            zip(long.retained_positions, long.decode.cache_snapshots, strict=True)
        ):
            assert len(positions) == snapshot.sequence_length == 3
            assert set(positions) <= prior | ({7 + step} if step else set())
            prior = set(positions)
        full = run_compressed_greedy_decode(model, ids, 4, policy, 100)
        reference = run_compressed_greedy_decode(model, ids, 4, FullCachePolicy(), 100)
        assert torch.equal(full.decode.sequences, reference.decode.sequences)
        for left, right in zip(full.decode.step_logits, reference.decode.step_logits, strict=True):
            assert_logits_close(left, right)


def test_attention_collection_and_generation(qwen: Qwen2ForCausalLM) -> None:
    if qwen.config._attn_implementation == "sdpa":
        adapter = Qwen2CacheAdapter(qwen)
        with pytest.raises(ValueError, match="eager"):
            adapter.forward(torch.tensor([[1, 2, 3]]), collect_attention=True)
        assert adapter.seen_tokens == 0
        assert adapter.positions == ()
    else:
        assert_attention_collection_and_generation(qwen)


def test_rejects_unknown_attention_query_reduction(qwen: Qwen2ForCausalLM) -> None:
    adapter = Qwen2CacheAdapter(qwen)
    with pytest.raises(ValueError, match="reduction"):
        adapter.forward(torch.tensor([[1, 2, 3]]), attention_query_reduction="median")  # type: ignore[arg-type]


@pytest.mark.model_download
@pytest.mark.skipif(
    os.environ.get("AMT_RUN_QWEN_INTEGRATION") != "1",
    reason="set AMT_RUN_QWEN_INTEGRATION=1 for pinned Qwen attention collection",
)
def test_pinned_qwen_attention_selection() -> None:
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    assert_attention_collection_and_generation(model)


@pytest.mark.model_download
@pytest.mark.parametrize("attention_precision", ["native", "float64"])
@pytest.mark.skipif(
    os.environ.get("AMT_RUN_QWEN_INTEGRATION") != "1",
    reason="set AMT_RUN_QWEN_INTEGRATION=1 for pinned Qwen compression validation",
)
def test_pinned_qwen_pruned_cache_semantics(
    monkeypatch: pytest.MonkeyPatch, attention_precision: str
) -> None:
    if attention_precision == "float64":
        # Diagnostic only: isolate attention reduction error without doubling
        # the 1.5B model weights. Production keeps the standard backend.
        def precise_attention(module, query, key, value, attention_mask, scaling, **kwargs):
            key = modeling_qwen2.repeat_kv(key.double(), module.num_key_value_groups)
            value = modeling_qwen2.repeat_kv(value.double(), module.num_key_value_groups)
            scores = query.double() @ key.transpose(-2, -1) * scaling
            if attention_mask is not None:
                scores = scores + attention_mask.double()
            weights = scores.softmax(-1)
            output = (weights @ value).transpose(1, 2).contiguous().to(query.dtype)
            return output, weights.to(query.dtype)

        monkeypatch.setattr(modeling_qwen2, "eager_attention_forward", precise_attention)
    device = os.environ.get("AMT_MODEL_DEVICE", "cpu")
    if attention_precision == "float64" and device != "cpu":
        pytest.skip("The float64 attention diagnostic is CPU-only")
    model = (
        AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            dtype=getattr(torch, os.environ.get("AMT_MODEL_DTYPE", "float32")),
            attn_implementation=(
                "eager"
                if attention_precision == "float64"
                else os.environ.get("AMT_MODEL_ATTENTION", "sdpa")
            ),
        )
        .eval()
        .to(os.environ.get("AMT_MODEL_DEVICE", "cpu"))
    )
    # Native float32 compares different reduction shapes; this separate bound
    # was established after the strict float64-attention diagnostic passed.
    absolute = 2e-4 if model.dtype == torch.float32 and attention_precision == "native" else None
    assert_dense_masked_equivalence(model, absolute)
    assert_wrong_position_detected(model, absolute)
    assert_full_retention_matches_generation(model, absolute)
