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
