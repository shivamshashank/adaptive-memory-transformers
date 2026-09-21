"""Deterministic reference and manual full-cache decoding utilities.

This module is the Day 2 correctness oracle.  It deliberately performs no
compression: the manual path must first agree with standard Transformers
generation while retaining the complete cache.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from transformers import GenerationConfig
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.modeling_outputs import CausalLMOutputWithPast


class _InspectableCacheLayer(Protocol):
    keys: torch.Tensor
    values: torch.Tensor


class _InspectableCache(Protocol):
    layers: Sequence[_InspectableCacheLayer]

    def get_seq_length(self) -> int: ...


class _CausalLM(Protocol):
    dtype: torch.dtype

    def generate(self, **kwargs: object) -> GenerateDecoderOnlyOutput | torch.Tensor: ...

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: _InspectableCache | None = None,
        use_cache: bool = True,
        return_dict: bool = True,
        logits_to_keep: int = 1,
    ) -> CausalLMOutputWithPast: ...


@dataclass(frozen=True)
class NumericalTolerance:
    """Absolute and relative tolerances for comparing two logit tensors."""

    absolute: float
    relative: float


@dataclass(frozen=True)
class LayerCacheShape:
    """Key and value tensor shapes for one Transformer layer."""

    keys: tuple[int, ...]
    values: tuple[int, ...]


@dataclass(frozen=True)
class CacheSnapshot:
    """Inspectable cache metadata captured after one model invocation."""

    sequence_length: int
    total_bytes: int
    layers: tuple[LayerCacheShape, ...]


@dataclass(frozen=True)
class DecodeTrace:
    """Generated tokens, prediction logits, and cache growth for one decode."""

    sequences: torch.Tensor
    step_logits: tuple[torch.Tensor, ...]
    cache_snapshots: tuple[CacheSnapshot, ...]


def tolerance_for_dtype(dtype: torch.dtype) -> NumericalTolerance:
    """Return the pre-declared comparison tolerance for a floating-point dtype."""

    tolerances = {
        torch.float32: NumericalTolerance(absolute=1e-5, relative=1e-5),
        torch.float16: NumericalTolerance(absolute=1e-3, relative=1e-3),
        torch.bfloat16: NumericalTolerance(absolute=1e-2, relative=1e-2),
    }
    try:
        return tolerances[dtype]
    except KeyError as exc:
        raise ValueError(f"No decoding tolerance declared for dtype {dtype}") from exc


def snapshot_cache(cache: _InspectableCache) -> CacheSnapshot:
    """Capture cache length, per-layer shapes, and physical tensor bytes."""

    total_bytes = 0
    layer_shapes: list[LayerCacheShape] = []
    for layer in cache.layers:
        keys = layer.keys
        values = layer.values
        total_bytes += keys.numel() * keys.element_size()
        total_bytes += values.numel() * values.element_size()
        layer_shapes.append(
            LayerCacheShape(
                keys=tuple(keys.shape),
                values=tuple(values.shape),
            )
        )
    return CacheSnapshot(
        sequence_length=cache.get_seq_length(),
        total_bytes=total_bytes,
        layers=tuple(layer_shapes),
    )


def _validate_decode_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> None:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")


def run_reference_greedy_decode(
    model: _CausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> DecodeTrace:
    """Run a neutral greedy decode through standard Transformers generation."""

    _validate_decode_inputs(input_ids, attention_mask, max_new_tokens)
    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.0,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        use_cache=True,
        return_dict_in_generate=True,
        output_logits=True,
        eos_token_id=None,
        pad_token_id=None,
    )
    with torch.inference_mode():
        output = cast(
            GenerateDecoderOnlyOutput,
            model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            ),
        )

    if output.logits is None:
        raise RuntimeError("Transformers generation did not return logits")
    if output.past_key_values is None:
        raise RuntimeError("Transformers generation did not return a KV cache")

    return DecodeTrace(
        sequences=output.sequences.detach().clone(),
        step_logits=tuple(logits.detach().clone() for logits in output.logits),
        cache_snapshots=(snapshot_cache(cast(_InspectableCache, output.past_key_values)),),
    )


def run_manual_full_cache_decode(
    model: _CausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> DecodeTrace:
    """Greedily decode token by token while retaining the entire KV cache."""

    _validate_decode_inputs(input_ids, attention_mask, max_new_tokens)
    sequences = input_ids.detach().clone()
    current_attention_mask = attention_mask.detach().clone()
    step_logits: list[torch.Tensor] = []
    cache_snapshots: list[CacheSnapshot] = []

    with torch.inference_mode():
        prefill = model(
            input_ids=input_ids,
            attention_mask=current_attention_mask,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        if prefill.past_key_values is None:
            raise RuntimeError("Model prefill did not return a KV cache")
        if prefill.logits is None:
            raise RuntimeError("Model prefill did not return logits")

        cache = cast(_InspectableCache, prefill.past_key_values)
        logits = prefill.logits[:, -1, :]
        step_logits.append(logits.detach().clone())
        cache_snapshots.append(snapshot_cache(cache))

        next_token = logits.argmax(dim=-1, keepdim=True)
        sequences = torch.cat((sequences, next_token), dim=1)

        for _ in range(1, max_new_tokens):
            current_attention_mask = torch.cat(
                (
                    current_attention_mask,
                    torch.ones(
                        (current_attention_mask.shape[0], 1),
                        dtype=current_attention_mask.dtype,
                        device=current_attention_mask.device,
                    ),
                ),
                dim=1,
            )
            decode_output = model(
                input_ids=next_token,
                attention_mask=current_attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            if decode_output.past_key_values is None:
                raise RuntimeError("Model decode step did not return a KV cache")
            if decode_output.logits is None:
                raise RuntimeError("Model decode step did not return logits")

            cache = cast(_InspectableCache, decode_output.past_key_values)
            logits = decode_output.logits[:, -1, :]
            step_logits.append(logits.detach().clone())
            cache_snapshots.append(snapshot_cache(cache))

            next_token = logits.argmax(dim=-1, keepdim=True)
            sequences = torch.cat((sequences, next_token), dim=1)

    return DecodeTrace(
        sequences=sequences,
        step_logits=tuple(step_logits),
        cache_snapshots=tuple(cache_snapshots),
    )
