"""Position-preserving cache adapter for full-attention Qwen2 models.

Qwen2.5 uses the Qwen2 implementation. Cached keys already contain rotary
position information: gather them unchanged and position new queries using
the total processed length, never the compressed cache length.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from transformers import Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache, DynamicLayer


class Qwen2CacheAdapter:
    """Own one unpadded sequence and a shared retained-position map across layers.

    This deliberately supports only eager/SDPA full attention with default RoPE.
    Padding, batches, sliding windows, and different retention per layer need
    separate validation. Create a fresh adapter for each independent sequence.
    """

    def __init__(self, model: Qwen2ForCausalLM) -> None:
        if not isinstance(model, Qwen2ForCausalLM):
            raise ValueError("The cache adapter supports Qwen2ForCausalLM only")
        if model.training:
            raise ValueError("Set the model to eval() before cached inference")
        config = model.config
        if not config.layer_types or any(kind != "full_attention" for kind in config.layer_types):
            raise ValueError("Only full_attention layers are supported")
        if not config.rope_parameters or config.rope_parameters.get("rope_type") != "default":
            raise ValueError("Only default RoPE is supported")
        if config._attn_implementation not in ("eager", "sdpa"):
            raise ValueError("Only eager and sdpa attention are supported")
        self.model = model
        self.cache = DynamicCache(config=config)
        self.positions: tuple[int, ...] = ()
        self.seen_tokens = 0
        self.attention_by_layer: tuple[tuple[float, ...], ...] = ()

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        collect_attention: bool = False,
        attention_query_reduction: Literal["last", "mean"] = "last",
    ) -> torch.Tensor:
        """Append tokens at original positions and return logits for each query.

        The first call prefills the prompt. Later calls append a token or chunk.
        All input tokens are real tokens; this API does not accept padded input.
        """
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [1, nonempty sequence]")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must have dtype torch.long")
        if attention_query_reduction not in ("last", "mean"):
            raise ValueError("Attention query reduction must be 'last' or 'mean'")
        if collect_attention and self.model.config._attn_implementation != "eager":
            raise ValueError(
                "Attention collection requires eager attention; load the model explicitly"
            )
        self.attention_by_layer = ()
        query_count = input_ids.shape[1]
        new_positions = tuple(range(self.seen_tokens, self.seen_tokens + query_count))
        key_positions = self.positions + new_positions
        query_ids = torch.tensor(new_positions, device=input_ids.device, dtype=torch.long)
        key_ids = torch.tensor(key_positions, device=input_ids.device, dtype=torch.long)

        # Columns describe physical cache slots, but causality uses absolute positions.
        allowed = key_ids.unsqueeze(0) <= query_ids.unsqueeze(1)
        mask = torch.zeros(allowed.shape, device=input_ids.device, dtype=self.model.dtype)
        mask.masked_fill_(~allowed, float("-inf"))
        output = self.model(
            input_ids=input_ids,
            position_ids=query_ids.unsqueeze(0),
            attention_mask=mask[None, None, :, :],
            past_key_values=self.cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=0,
            output_attentions=collect_attention,
        )
        self.positions = key_positions
        self.seen_tokens += query_count
        if collect_attention:
            if output.attentions is None or len(output.attentions) != len(self.cache.layers):
                raise RuntimeError("Model did not return attention for every layer")
            if attention_query_reduction == "last":
                reduced = (
                    weights[0, :, -1, :].float().mean(dim=0) for weights in output.attentions
                )
            else:
                reduced = (weights[0].float().mean(dim=(0, 1)) for weights in output.attentions)
            self.attention_by_layer = tuple(tuple(scores.cpu().tolist()) for scores in reduced)
        return output.logits

    @torch.inference_mode()
    def retain(self, indices: Sequence[int]) -> None:
        """Gather sorted unique physical slots in every layer, without re-rotation.

        Indices address the CURRENT cache, not the original prompt. Positions
        remain absolute even after repeated pruning or an empty retained set.
        """
        selected = tuple(indices)
        if any(not isinstance(index, int) or isinstance(index, bool) for index in selected):
            raise ValueError("Retained indices must be integers")
        if any(index < 0 or index >= len(self.positions) for index in selected):
            raise ValueError("Retained index is outside the current cache")
        if any(left >= right for left, right in zip(selected, selected[1:])):
            raise ValueError("Retained indices must be sorted and unique")
        if selected == tuple(range(len(self.positions))):
            return
        for layer in self.cache.layers:
            if not isinstance(layer, DynamicLayer) or layer.keys is None or layer.values is None:
                raise RuntimeError("Expected initialized dynamic attention cache layers")
            index_tensor = torch.tensor(selected, dtype=torch.long, device=layer.keys.device)
            # index_select allocates compact tensors, rather than views holding old storage.
            layer.keys = layer.keys.index_select(-2, index_tensor)
            layer.values = layer.values.index_select(-2, index_tensor)
        self.positions = tuple(self.positions[index] for index in selected)
        self.attention_by_layer = tuple(
            tuple(scores[index] for index in selected) for scores in self.attention_by_layer
        )
