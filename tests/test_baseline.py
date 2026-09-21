"""Unit tests for baseline calculations that do not download a model."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from amt.baseline import _CausalLM, _torch_dtype_for_name, estimate_kv_cache_bytes


def test_estimated_cache_bytes_uses_kv_heads() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=2,
            num_key_value_heads=4,
            num_attention_heads=8,
            head_dim=16,
            hidden_size=128,
        ),
        dtype=torch.float32,
    )

    assert estimate_kv_cache_bytes(cast(_CausalLM, model), seq_len=10) == 10_240


def test_estimated_cache_bytes_falls_back_to_attention_heads() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=2,
            num_key_value_heads=None,
            num_attention_heads=8,
            head_dim=16,
            hidden_size=128,
        ),
        dtype=torch.float32,
    )

    assert estimate_kv_cache_bytes(cast(_CausalLM, model), seq_len=10) == 20_480


def test_dtype_aliases_are_normalized() -> None:
    assert _torch_dtype_for_name("fp32", "cpu") == torch.float32
    assert _torch_dtype_for_name("bf16", "cpu") == torch.bfloat16


def test_cpu_float16_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="not reliably supported"):
        _torch_dtype_for_name("float16", "cpu")


@pytest.mark.parametrize("count", [1, 4])
def test_baseline_uses_one_fixed_length_run(monkeypatch, count):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    import amt.baseline as baseline

    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        eos_token_id=None,
    )
    model = Qwen2ForCausalLM(config).eval()
    calls = []
    hook = model.register_forward_hook(lambda *args: calls.append(1))

    class Inputs(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token = "pad"
        eos_token = None
        eos_token_id = None

        def __call__(self, text, **kwargs):
            return Inputs(
                input_ids=torch.tensor([[1, 5, 9]]),
                attention_mask=torch.ones(1, 3, dtype=torch.long),
            )

        def decode(self, ids, **kwargs):
            return str(ids.tolist())

    monkeypatch.setattr(
        baseline, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())
    )
    monkeypatch.setattr(
        baseline, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=lambda *a, **k: model)
    )
    result = baseline.run_full_cache_baseline(max_new_tokens=count, device="cpu")
    hook.remove()
    assert len(calls) == count  # One prefill plus N-1 decoding calls; no second generation.
    assert result.generated_tokens == count
    assert result.actual_kv_cache_seq_len == 3 + count - 1
    assert result.actual_kv_cache_bytes == result.estimated_kv_cache_bytes
    assert result.total_time_s == result.prefill_time_s + result.decode_time_s
    assert result.decode_tokens_per_second == (count - 1) / result.decode_time_s
    assert result.peak_memory_mb is None
    with torch.inference_mode():
        expected = model.generate(
            torch.tensor([[1, 5, 9]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            max_new_tokens=count,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
        )
    assert result.generated_text == str(expected[0, 3:].tolist())


def test_non_cuda_peak_memory_is_unavailable_not_zero(monkeypatch):
    from amt.baseline import _peak_gpu_memory_mb

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _peak_gpu_memory_mb("cpu") is None
    assert _peak_gpu_memory_mb("mps") is None
