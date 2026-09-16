"""Matched-policy decode evaluation for KV-cache retention experiments.

The evaluator intentionally reports task metrics separately from the existing
retained-attention proxy experiments.  It supports small causal models for
smoke runs; larger context sweeps should be driven by versioned configurations.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from src.adaptive_policy import AdaptiveImportancePolicy
from src.cache_policies import VALID_BUDGETS, budget_tokens_for_ratio, select_cache_indices
from src.experiments.attention_cache_evaluation import _prune_past_key_values
from src.importance import DEFAULT_MODEL


@dataclass(frozen=True)
class DecodeMeasurement:
    policy: str
    budget_ratio: float
    budget_tokens: int
    prompt_tokens: int
    generated_tokens: int
    generated_text: str
    next_token_loss: float | None
    perplexity: float | None
    exact_answer: bool | None
    prefill_time_s: float
    selection_time_s: float
    pruning_time_s: float
    decode_time_s: float
    decode_tokens_per_second: float
    cache_bytes_after_prefill: int
    cache_bytes_final: int
    peak_memory_mb: float


def _cache_bytes(cache: DynamicCache) -> int:
    return sum(
        layer.keys.numel() * layer.keys.element_size()
        + layer.values.numel() * layer.values.element_size()
        for layer in cache.layers
    )


def _normalize_answer(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _attention_scores(attentions: Sequence[torch.Tensor], sequence_length: int) -> list[float]:
    scores = [0.0] * sequence_length
    for attention in attentions:
        # [batch, heads, query, key] -> average key attention across heads/queries.
        key_mass = attention[0].mean(dim=0).mean(dim=0)
        for index in range(sequence_length):
            scores[index] += float(key_mass[index].item())
    return scores


def _select_indices(
    policy: str, total_tokens: int, budget: int, scores: Sequence[float]
) -> list[int]:
    if policy in {"full", "recency", "uniform"}:
        return select_cache_indices(policy, total_tokens, budget)
    if policy == "adaptive":
        recency = [(index + 1) / max(total_tokens, 1) for index in range(total_tokens)]
        return AdaptiveImportancePolicy().select_indices(
            total_tokens,
            budget,
            attention_scores=scores,
            recency_scores=recency,
            frequency_scores=[0.0] * total_tokens,
        )
    raise ValueError(f"Unsupported policy: {policy}")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _prune_dynamic_cache(cache: DynamicCache, indices: list[int]) -> tuple[DynamicCache, float]:
    start = time.perf_counter()
    pruned = _prune_past_key_values(cache, indices)
    _sync()
    elapsed = time.perf_counter() - start
    if not isinstance(pruned, DynamicCache):
        raise TypeError("DynamicCache pruning must return a DynamicCache")
    return pruned, elapsed


def _clone_cache(cache: DynamicCache) -> DynamicCache:
    return DynamicCache(
        ddp_cache_data=[(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
    )


def _teacher_forced_loss(
    *,
    model: object,
    cache: DynamicCache,
    initial_logits: torch.Tensor,
    continuation_ids: torch.Tensor | None,
    policy: str,
    budget: int,
    initial_scores: Sequence[float],
) -> float | None:
    if continuation_ids is None or continuation_ids.numel() == 0:
        return None
    current_cache = _clone_cache(cache)
    scores = list(initial_scores)
    logits = initial_logits
    losses: list[float] = []
    with torch.no_grad():
        for target in continuation_ids[0]:
            losses.append(float(-torch.log_softmax(logits[0, -1, :], dim=-1)[target].item()))
            output = model(
                input_ids=target.reshape(1, 1),
                past_key_values=current_cache,
                output_attentions=True,
                use_cache=True,
            )
            current_cache = output.past_key_values
            logits = output.logits
            if policy != "full":
                if output.attentions is None:
                    raise ValueError(
                        "Teacher-forced compressed evaluation requires attention outputs"
                    )
                scores = _attention_scores(output.attentions, current_cache.get_seq_length())
                keep = _select_indices(policy, current_cache.get_seq_length(), budget, scores)
                current_cache, _ = _prune_dynamic_cache(current_cache, keep)
                scores = [scores[index] for index in keep]
    return sum(losses) / len(losses)


def evaluate_prompt(
    *,
    model_name: str,
    prompt: str,
    expected_answer: str | None,
    continuation: str | None,
    max_new_tokens: int,
    device: str | None = None,
) -> list[DecodeMeasurement]:
    """Evaluate all policies with matched initial cache capacities.

    Each compressed policy is pruned after prefill and after every decode step,
    so its cache is bounded by the configured capacity between model calls.
    """
    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    model.to(device_name)
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device_name)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    prompt_tokens = int(input_ids.shape[1])
    continuation_ids = (
        tokenizer(continuation, return_tensors="pt")["input_ids"].to(device_name)
        if continuation is not None
        else None
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    _sync()
    start = time.perf_counter()
    with torch.no_grad():
        prefill = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=True,
        )
    _sync()
    prefill_time = time.perf_counter() - start
    if prefill.attentions is None or not isinstance(prefill.past_key_values, DynamicCache):
        raise ValueError("The selected model must return eager attentions and DynamicCache values")
    initial_scores = _attention_scores(prefill.attentions, prompt_tokens)

    measurements: list[DecodeMeasurement] = []
    for ratio in VALID_BUDGETS:
        budget = budget_tokens_for_ratio(prompt_tokens, ratio)
        for policy in ("full", "recency", "uniform", "adaptive"):
            cache = _clone_cache(prefill.past_key_values)
            policy_budget = prompt_tokens if policy == "full" else budget
            selection_start = time.perf_counter()
            keep = _select_indices(policy, prompt_tokens, policy_budget, initial_scores)
            selection_time = time.perf_counter() - selection_start
            pruning_time = 0.0
            current_scores = [initial_scores[index] for index in keep]
            if policy != "full":
                cache, pruning_time = _prune_dynamic_cache(cache, keep)
            cache_bytes_after_prefill = _cache_bytes(cache)
            next_token_loss = _teacher_forced_loss(
                model=model,
                cache=cache,
                initial_logits=prefill.logits,
                continuation_ids=continuation_ids,
                policy=policy,
                budget=policy_budget,
                initial_scores=current_scores,
            )
            next_token = prefill.logits[:, -1:, :].argmax(dim=-1)
            generated: list[int] = []
            decode_start = time.perf_counter()
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    if attention_mask is not None:
                        attention_mask = torch.ones(
                            (1, cache.get_seq_length() + 1),
                            dtype=attention_mask.dtype,
                            device=device_name,
                        )
                    output = model(
                        input_ids=next_token,
                        attention_mask=attention_mask,
                        past_key_values=cache,
                        output_attentions=True,
                        use_cache=True,
                    )
                    target = output.logits[:, -1, :].argmax(dim=-1)
                    generated.append(int(target.item()))
                    next_token = target[:, None]
                    cache = output.past_key_values
                    if policy != "full":
                        if output.attentions is None:
                            raise ValueError(
                                "Compressed evaluation requires attention outputs during decode"
                            )
                        step_scores = _attention_scores(output.attentions, cache.get_seq_length())
                        selection_start = time.perf_counter()
                        keep = _select_indices(
                            policy, cache.get_seq_length(), policy_budget, step_scores
                        )
                        selection_time += time.perf_counter() - selection_start
                        cache, elapsed = _prune_dynamic_cache(cache, keep)
                        pruning_time += elapsed
            _sync()
            decode_time = time.perf_counter() - decode_start
            generated_text = tokenizer.decode(generated, skip_special_tokens=True)
            measurements.append(
                DecodeMeasurement(
                    policy=policy,
                    budget_ratio=ratio,
                    budget_tokens=policy_budget,
                    prompt_tokens=prompt_tokens,
                    generated_tokens=len(generated),
                    generated_text=generated_text,
                    next_token_loss=next_token_loss,
                    perplexity=math.exp(next_token_loss) if next_token_loss is not None else None,
                    exact_answer=(
                        _normalize_answer(generated_text) == _normalize_answer(expected_answer)
                        if expected_answer is not None
                        else None
                    ),
                    prefill_time_s=prefill_time,
                    selection_time_s=selection_time,
                    pruning_time_s=pruning_time,
                    decode_time_s=decode_time,
                    decode_tokens_per_second=(len(generated) / decode_time) if decode_time else 0.0,
                    cache_bytes_after_prefill=cache_bytes_after_prefill,
                    cache_bytes_final=_cache_bytes(cache),
                    peak_memory_mb=(
                        float(torch.cuda.max_memory_allocated() / 1024**2)
                        if torch.cuda.is_available()
                        else 0.0
                    ),
                )
            )
    return measurements


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run matched-policy compressed decoding on one controlled prompt."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--prompt", default="The recovery phrase is amber-17. Return the recovery phrase."
    )
    parser.add_argument("--expected-answer", default="amber-17")
    parser.add_argument(
        "--continuation",
        default=None,
        help="Known continuation used for teacher-forced loss and perplexity.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--output", type=Path, default=Path("results/compressed_decoding_evaluation.json")
    )
    args = parser.parse_args()
    records = evaluate_prompt(
        model_name=args.model,
        prompt=args.prompt,
        expected_answer=args.expected_answer,
        continuation=args.continuation,
        max_new_tokens=args.max_new_tokens,
    )
    payload = {
        "quality_metric": "teacher-forced next-token loss/perplexity when a continuation is supplied, plus exact greedy generated-answer match; retained attention mass is not used as a task metric",
        "limitations": [
            "Arbitrary token pruning reindexes retained cache positions; architectures requiring absolute cache positions need a model-specific adapter.",
            "One prompt and one seed are a smoke test, not a statistical result.",
        ],
        "records": [asdict(record) for record in records],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
