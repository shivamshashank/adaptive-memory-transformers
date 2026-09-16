# Passkey retrieval (Needle-in-a-Haystack) evaluation for matched KV-cache policies.
from __future__ import annotations

import argparse
import json
import math
import os
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

DEFAULT_PASSKEY = "84920"
DEFAULT_CONTEXT_LENGTH = 256
DEFAULT_NEEDLE_DEPTH = 0.25

FILLER_SENTENCE = (
    "The grass is green and the sky is blue. "
    "A digital computer processes information using binary transistors. "
    "Data centers operate continuously to maintain global distributed services. "
    "Machine learning models require rigorous empirical evaluation and testing. "
    "Algorithms optimize objective functions across multi-dimensional search spaces. "
)


@dataclass(frozen=True)
class PasskeyPrompt:
    full_prompt: str
    passkey: str
    expected_continuation: str
    target_tokens: int
    actual_tokens: int
    needle_depth: float
    needle_start_token: int
    needle_end_token: int


@dataclass(frozen=True)
class PasskeyMeasurement:
    policy: str
    budget_ratio: float
    budget_tokens: int
    prompt_tokens: int
    needle_tokens: int
    needle_tokens_retained: int
    needle_retention_rate: float
    generated_tokens: int
    generated_text: str
    passkey_found: bool
    next_token_loss: float | None
    perplexity: float | None
    prefill_time_s: float
    selection_time_s: float
    pruning_time_s: float
    decode_time_s: float
    decode_tokens_per_second: float
    cache_bytes_after_prefill: int
    cache_bytes_final: int
    peak_memory_mb: float


def _ensure_ssl_cert() -> None:
    # Ensure macOS Python environments can access certificates for HF downloads.
    if "SSL_CERT_FILE" not in os.environ:
        try:
            import certifi

            os.environ["SSL_CERT_FILE"] = certifi.where()
        except ImportError:
            pass


def _cache_bytes(cache: DynamicCache) -> int:
    # Sum total bytes across all key and value tensors in the cache.
    return sum(
        layer.keys.numel() * layer.keys.element_size()
        + layer.values.numel() * layer.values.element_size()
        for layer in cache.layers
    )


def _sync() -> None:
    # Synchronize CUDA device if available for accurate timing.
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _attention_scores(attentions: Sequence[torch.Tensor], sequence_length: int) -> list[float]:
    # Average key attention mass across heads and query positions for each layer.
    scores = [0.0] * sequence_length
    for attention in attentions:
        key_mass = attention[0].mean(dim=0).mean(dim=0)
        for index in range(min(sequence_length, key_mass.shape[0])):
            scores[index] += float(key_mass[index].item())
    return scores


def _accumulate_attention_scores(
    previous: Sequence[float], current: Sequence[float]
) -> list[float]:
    if len(current) < len(previous):
        raise ValueError("current attention scores must cover the retained cache")
    return [previous[index] + current[index] for index in range(len(previous))] + list(
        current[len(previous) :]
    )


def _select_indices(
    policy: str, total_tokens: int, budget: int, scores: Sequence[float]
) -> list[int]:
    # Select retained indices according to the configured cache policy.
    if policy in {"full", "recency", "uniform"}:
        return select_cache_indices(policy, total_tokens, budget)
    if policy == "adaptive":
        recency = [(idx + 1) / max(total_tokens, 1) for idx in range(total_tokens)]
        return AdaptiveImportancePolicy().select_indices(
            total_tokens,
            budget,
            attention_scores=scores,
            recency_scores=recency,
            frequency_scores=[0.0] * total_tokens,
        )
    raise ValueError(f"Unsupported policy: {policy}")


def _prune_dynamic_cache(cache: DynamicCache, indices: list[int]) -> tuple[DynamicCache, float]:
    # Prune DynamicCache layers and record elapsed time.
    start = time.perf_counter()
    pruned = _prune_past_key_values(cache, indices)
    _sync()
    elapsed = time.perf_counter() - start
    if not isinstance(pruned, DynamicCache):
        raise TypeError("DynamicCache pruning must return a DynamicCache")
    return pruned, elapsed


def _clone_cache(cache: DynamicCache) -> DynamicCache:
    # Create an independent clone of DynamicCache layers.
    return DynamicCache(
        ddp_cache_data=[(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
    )


def _position_ids(position: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([[position]], dtype=torch.long, device=device)


def build_passkey_prompt(
    tokenizer: AutoTokenizer,
    *,
    passkey: str = DEFAULT_PASSKEY,
    target_length: int = DEFAULT_CONTEXT_LENGTH,
    depth: float = DEFAULT_NEEDLE_DEPTH,
) -> PasskeyPrompt:
    # Construct a controlled needle-in-a-haystack prompt with exact needle boundaries.
    prefix = "There is an important secret hidden in the text below. Read carefully and answer the question at the end.\n\n"
    needle = f"The secret passkey is {passkey}. Remember this passkey. "
    query = "\n\nWhat is the secret passkey? The secret passkey is"
    continuation = f" {passkey}"

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    query_ids = tokenizer.encode(query, add_special_tokens=False)

    fixed_tokens = len(prefix_ids) + len(needle_ids) + len(query_ids)
    filler_budget = max(0, target_length - fixed_tokens)

    filler_token_ids = tokenizer.encode(FILLER_SENTENCE, add_special_tokens=False)
    repeated_filler: list[int] = []
    while len(repeated_filler) < filler_budget:
        repeated_filler.extend(filler_token_ids)
    repeated_filler = repeated_filler[:filler_budget]

    pre_needle_budget = int(round(depth * filler_budget))
    pre_filler_ids = repeated_filler[:pre_needle_budget]
    post_filler_ids = repeated_filler[pre_needle_budget:]

    pre_filler_text = tokenizer.decode(pre_filler_ids)
    post_filler_text = tokenizer.decode(post_filler_ids)

    full_prompt = prefix + pre_filler_text + needle + post_filler_text + query
    encoded_all = tokenizer.encode(full_prompt, add_special_tokens=False)

    needle_start = len(prefix_ids) + len(pre_filler_ids)
    needle_end = needle_start + len(needle_ids)

    return PasskeyPrompt(
        full_prompt=full_prompt,
        passkey=passkey,
        expected_continuation=continuation,
        target_tokens=target_length,
        actual_tokens=len(encoded_all),
        needle_depth=depth,
        needle_start_token=needle_start,
        needle_end_token=needle_end,
    )


def _compute_teacher_forced_loss(
    model: AutoModelForCausalLM,
    cache: DynamicCache,
    continuation_ids: torch.Tensor,
    initial_logits: torch.Tensor,
    policy: str,
    budget: int,
    initial_prompt_tokens: int,
    initial_scores: Sequence[float],
) -> float | None:
    # Compute cross-entropy loss over the target continuation tokens under cache compression.
    if continuation_ids.numel() == 0:
        return None
    current_cache = _clone_cache(cache)
    cache_positions = list(range(initial_prompt_tokens))
    cumulative_scores = list(initial_scores)
    logits = initial_logits
    losses: list[float] = []

    with torch.no_grad():
        for step_index, target_id in enumerate(continuation_ids[0]):
            target_scalar = int(target_id.item())
            step_loss = -torch.log_softmax(logits[0, -1, :], dim=-1)[target_scalar].item()
            losses.append(float(step_loss))

            output = model(
                input_ids=target_id.reshape(1, 1),
                position_ids=_position_ids(initial_prompt_tokens + step_index, target_id.device),
                past_key_values=current_cache,
                output_attentions=True,
                use_cache=True,
            )
            current_cache = output.past_key_values
            logits = output.logits
            cache_positions.append(initial_prompt_tokens + step_index)

            if policy != "full":
                if output.attentions is None:
                    raise ValueError("Compressed evaluation requires attention outputs")
                step_scores = _attention_scores(output.attentions, current_cache.get_seq_length())
                cumulative_scores = _accumulate_attention_scores(cumulative_scores, step_scores)
                keep = _select_indices(
                    policy, current_cache.get_seq_length(), budget, cumulative_scores
                )
                current_cache, _ = _prune_dynamic_cache(current_cache, keep)
                cache_positions = [cache_positions[index] for index in keep]
                cumulative_scores = [cumulative_scores[index] for index in keep]

    return sum(losses) / len(losses) if losses else None


def evaluate_passkey_retrieval(
    *,
    model_name: str,
    passkey: str = DEFAULT_PASSKEY,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    depth: float = DEFAULT_NEEDLE_DEPTH,
    max_new_tokens: int = 6,
    device: str | None = None,
) -> tuple[PasskeyPrompt, list[PasskeyMeasurement]]:
    # Run matched-policy evaluation on a synthetic passkey retrieval task.
    _ensure_ssl_cert()
    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_spec = build_passkey_prompt(
        tokenizer,
        passkey=passkey,
        target_length=context_length,
        depth=depth,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    model.to(device_name)
    model.eval()

    inputs = tokenizer(prompt_spec.full_prompt, return_tensors="pt").to(device_name)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    prompt_tokens = int(input_ids.shape[1])

    continuation_ids = tokenizer(
        prompt_spec.expected_continuation,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].to(device_name)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    _sync()
    start_prefill = time.perf_counter()
    with torch.no_grad():
        prefill = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=True,
        )
    _sync()
    prefill_time = time.perf_counter() - start_prefill

    if prefill.attentions is None or not isinstance(prefill.past_key_values, DynamicCache):
        raise ValueError("Model must return eager attentions and DynamicCache")

    initial_scores = _attention_scores(prefill.attentions, prompt_tokens)
    needle_token_set = set(range(prompt_spec.needle_start_token, prompt_spec.needle_end_token))
    total_needle_tokens = max(1, len(needle_token_set))

    measurements: list[PasskeyMeasurement] = []

    for ratio in VALID_BUDGETS:
        budget = budget_tokens_for_ratio(prompt_tokens, ratio)
        for policy in ("full", "recency", "uniform", "adaptive"):
            cache = _clone_cache(prefill.past_key_values)
            policy_budget = prompt_tokens if policy == "full" else budget

            selection_start = time.perf_counter()
            keep = _select_indices(policy, prompt_tokens, policy_budget, initial_scores)
            selection_time = time.perf_counter() - selection_start
            pruning_time = 0.0

            if policy != "full":
                cache, pruning_time = _prune_dynamic_cache(cache, keep)

            cache_bytes_after_prefill = _cache_bytes(cache)
            retained_needle_tokens = len(set(keep) & needle_token_set)
            retention_rate = retained_needle_tokens / total_needle_tokens

            next_token_loss = _compute_teacher_forced_loss(
                model=model,
                cache=cache,
                continuation_ids=continuation_ids,
                initial_logits=prefill.logits,
                policy=policy,
                budget=policy_budget,
                initial_prompt_tokens=prompt_tokens,
                initial_scores=[initial_scores[index] for index in keep],
            )

            # Autoregressive greedy decode
            next_token = prefill.logits[:, -1:, :].argmax(dim=-1)
            generated: list[int] = []
            cache_positions = list(range(prompt_tokens))
            cumulative_scores = [initial_scores[index] for index in keep]
            decode_start = time.perf_counter()

            with torch.no_grad():
                for _ in range(max_new_tokens):
                    cur_mask = (
                        torch.ones(
                            (1, cache.get_seq_length() + 1),
                            dtype=attention_mask.dtype,
                            device=device_name,
                        )
                        if attention_mask is not None
                        else None
                    )
                    out = model(
                        input_ids=next_token,
                        position_ids=_position_ids(
                            prompt_tokens + len(generated), next_token.device
                        ),
                        attention_mask=cur_mask,
                        past_key_values=cache,
                        output_attentions=True,
                        use_cache=True,
                    )
                    target = out.logits[:, -1, :].argmax(dim=-1)
                    generated.append(int(target.item()))
                    next_token = target[:, None]
                    cache = out.past_key_values
                    cache_positions.append(prompt_tokens + len(generated))

                    if policy != "full":
                        if out.attentions is None:
                            raise ValueError("Compressed evaluation requires attention outputs")
                        step_scores = _attention_scores(out.attentions, cache.get_seq_length())
                        cumulative_scores = _accumulate_attention_scores(
                            cumulative_scores, step_scores
                        )
                        sel_start = time.perf_counter()
                        step_keep = _select_indices(
                            policy,
                            cache.get_seq_length(),
                            policy_budget,
                            cumulative_scores,
                        )
                        selection_time += time.perf_counter() - sel_start
                        cache, step_prune = _prune_dynamic_cache(cache, step_keep)
                        cache_positions = [cache_positions[index] for index in step_keep]
                        cumulative_scores = [cumulative_scores[index] for index in step_keep]
                        pruning_time += step_prune

            _sync()
            decode_time = time.perf_counter() - decode_start
            generated_text = tokenizer.decode(generated, skip_special_tokens=True)
            passkey_found = passkey in generated_text

            peak_mem = (
                float(torch.cuda.max_memory_allocated() / (1024**2))
                if torch.cuda.is_available()
                else 0.0
            )

            measurements.append(
                PasskeyMeasurement(
                    policy=policy,
                    budget_ratio=ratio,
                    budget_tokens=policy_budget,
                    prompt_tokens=prompt_tokens,
                    needle_tokens=total_needle_tokens,
                    needle_tokens_retained=retained_needle_tokens,
                    needle_retention_rate=round(retention_rate, 4),
                    generated_tokens=len(generated),
                    generated_text=generated_text,
                    passkey_found=passkey_found,
                    next_token_loss=next_token_loss,
                    perplexity=math.exp(next_token_loss) if next_token_loss is not None else None,
                    prefill_time_s=prefill_time,
                    selection_time_s=selection_time,
                    pruning_time_s=pruning_time,
                    decode_time_s=decode_time,
                    decode_tokens_per_second=(len(generated) / decode_time) if decode_time else 0.0,
                    cache_bytes_after_prefill=cache_bytes_after_prefill,
                    cache_bytes_final=_cache_bytes(cache),
                    peak_memory_mb=peak_mem,
                )
            )

    return prompt_spec, measurements


def main() -> None:
    # CLI entry point for passkey retrieval evaluation.
    parser = argparse.ArgumentParser(
        description="Run passkey retrieval evaluation on matched KV-cache policies."
    )
    parser.add_argument(
        "--model", default="HuggingFaceTB/SmolLM-135M", help="Model name or local checkpoint path."
    )
    parser.add_argument("--passkey", default=DEFAULT_PASSKEY, help="Secret passkey string.")
    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help="Target context sequence length.",
    )
    parser.add_argument(
        "--needle-depth",
        type=float,
        default=DEFAULT_NEEDLE_DEPTH,
        help="Fractional depth to place needle (0.0 to 1.0).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=6, help="Maximum generated tokens.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/passkey_retrieval.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    prompt_spec, records = evaluate_passkey_retrieval(
        model_name=args.model,
        passkey=args.passkey,
        context_length=args.context_length,
        depth=args.needle_depth,
        max_new_tokens=args.max_new_tokens,
    )

    payload = {
        "benchmark": "Passkey Retrieval (Needle-in-a-Haystack)",
        "model_name": args.model,
        "prompt_spec": asdict(prompt_spec),
        "records": [asdict(rec) for rec in records],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "success",
                "model": args.model,
                "context_length": prompt_spec.actual_tokens,
                "needle_depth": args.needle_depth,
                "records_count": len(records),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
