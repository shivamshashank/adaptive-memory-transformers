# Full-cache inference baseline with timing, memory, and hardware metadata.
from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence, cast

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "hf-internal-testing/tiny-random-gpt2"


class _TorchRandom(Protocol):
    def manual_seed(self, seed: int) -> torch.Generator: ...


class _TokenizedInputs(Mapping[str, torch.Tensor]):
    def to(self, *args: object, **kwargs: object) -> "_TokenizedInputs":
        del args, kwargs
        return self
        # Preserve the typed token mapping for the baseline runner.


class _Tokenizer(Protocol):
    pad_token: str | None
    eos_token: str | None
    eos_token_id: int | list[int] | None

    def __call__(self, text: str, *, return_tensors: Literal["pt"]) -> _TokenizedInputs: ...

    def decode(self, token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str: ...


class _TokenizerFactory(Protocol):
    def from_pretrained(self, pretrained_model_name_or_path: str) -> _Tokenizer: ...


class _CausalLM(Protocol):
    config: "_ModelConfig"
    dtype: torch.dtype

    def to(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "_CausalLM": ...

    def eval(self) -> "_CausalLM": ...

    def generate(self, **kwargs: object) -> torch.Tensor: ...

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: "_Cache | None" = None,
        use_cache: bool = True,
    ) -> "_ModelOutput": ...


class _CacheLayer(Protocol):
    keys: torch.Tensor
    values: torch.Tensor


class _Cache(Protocol):
    layers: Sequence[_CacheLayer]

    def get_seq_length(self) -> int: ...


class _ModelOutput(Protocol):
    logits: torch.Tensor
    past_key_values: _Cache


class _ModelFactory(Protocol):
    def from_pretrained(self, pretrained_model_name_or_path: str) -> _CausalLM: ...


class _ModelConfig(Protocol):
    num_hidden_layers: int
    num_key_value_heads: int
    num_attention_heads: int
    head_dim: int | None
    hidden_size: int


class _CudaDeviceProperties(Protocol):
    total_memory: int


class _CudaModule(Protocol):
    def get_device_properties(self, device: int | str | None = None) -> _CudaDeviceProperties: ...


@dataclass
class BaselineResult:
    model_name: str
    model_revision: str | None
    prompt: str
    seed: int
    max_new_tokens: int
    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    total_time_s: float
    tokens_per_second: float
    prefill_time_s: float
    decode_time_s: float
    decode_tokens_per_second: float
    estimated_kv_cache_bytes: int
    actual_kv_cache_bytes: int
    actual_kv_cache_seq_len: int
    actual_kv_cache_shapes: list[dict[str, list[int]]]
    device: str
    dtype: str
    peak_memory_mb: float
    python_version: str
    torch_version: str
    transformers_version: str
    platform: str
    cpu_info: str
    gpu_device_name: str | None
    gpu_total_memory_mb: float | None
    cuda_version: str | None
    model_config: dict[str, int | None]
    cuda_device_name: str | None

    def to_dict(self) -> dict[str, object]:
        # Serialize baseline measurements for JSON output.
        return asdict(self)


def _estimate_kv_cache_bytes(model: _CausalLM, seq_len: int) -> int:
    config = model.config
    # Estimate cache storage from model dimensions and element size.
    num_layers = config.num_hidden_layers
    if hasattr(config, "num_key_value_heads"):
        kv_heads = config.num_key_value_heads
    else:
        kv_heads = config.num_attention_heads

    head_dim = cast(int | None, getattr(config, "head_dim", None))
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads

    elem_bytes = torch.empty((), dtype=model.dtype).element_size()
    estimated = 2 * num_layers * kv_heads * seq_len * head_dim * elem_bytes
    return int(estimated)


def _peak_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024**2))
    # Read peak allocated GPU memory when CUDA is available.
    return 0.0


def _torch_dtype_for_name(dtype_name: str, device: str) -> torch.dtype:
    normalized = dtype_name.lower()
    # Convert a CLI dtype name into a supported torch dtype.
    mapping: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Supported: {sorted(mapping)}")

    if device == "cpu" and normalized in {"float16", "fp16"}:
        raise RuntimeError(
            "float16 is not reliably supported for this model on CPU; use bfloat16 or float32."
        )
    return mapping[normalized]


def _synchronize_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Synchronize CUDA before taking timing measurements.


def _measure_prefill_and_decode(
    model: _CausalLM,
    inputs: _TokenizedInputs,
    max_new_tokens: int,
) -> tuple[float, float, _Cache]:
    input_ids = inputs["input_ids"]
    # Measure prompt prefill and autoregressive decode separately.
    attention_mask = inputs.get("attention_mask")

    _synchronize_device()
    prefill_start = time.perf_counter()
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    _synchronize_device()
    prefill_time = time.perf_counter() - prefill_start

    cache = output.past_key_values
    next_token = output.logits[:, -1:, :].argmax(dim=-1)
    decode_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(attention_mask[:, :1])],
                    dim=1,
                )
            output = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            next_token = output.logits[:, -1:, :].argmax(dim=-1)
    _synchronize_device()
    decode_time = time.perf_counter() - decode_start
    return prefill_time, decode_time, cache


def _actual_cache_metadata(cache: _Cache) -> tuple[int, list[dict[str, list[int]]]]:
    cache_bytes = 0
    shapes: list[dict[str, list[int]]] = []
    # Collect actual cache bytes and tensor shapes from the model cache.
    for layer in cache.layers:
        cache_bytes += layer.keys.numel() * layer.keys.element_size()
        cache_bytes += layer.values.numel() * layer.values.element_size()
        shapes.append(
            {
                "keys": list(layer.keys.shape),
                "values": list(layer.values.shape),
            }
        )
    return cache_bytes, shapes


def _prompt_for_length(tokenizer: _Tokenizer, prompt: str, target_tokens: int | None) -> str:
    if target_tokens is None:
        return prompt
    # Expand and truncate a prompt to a requested token length.

    prompt_inputs = tokenizer(prompt, return_tensors="pt")
    prompt_ids = prompt_inputs["input_ids"][0]
    repetitions = max(1, (target_tokens + int(prompt_ids.shape[0]) - 1) // int(prompt_ids.shape[0]))
    expanded_prompt = " ".join([prompt] * repetitions)
    expanded_inputs = tokenizer(expanded_prompt, return_tensors="pt")
    expanded_ids = expanded_inputs["input_ids"][0][:target_tokens]
    return tokenizer.decode(expanded_ids, skip_special_tokens=True)


def _model_config_metadata(model: _CausalLM) -> dict[str, int | None]:
    config = model.config
    # Extract cache-relevant model configuration fields.
    return {
        "num_hidden_layers": cast(int, getattr(config, "num_hidden_layers", None)),
        "num_attention_heads": cast(int, getattr(config, "num_attention_heads", None)),
        "num_key_value_heads": cast(int | None, getattr(config, "num_key_value_heads", None)),
        "hidden_size": cast(int, getattr(config, "hidden_size", None)),
        "head_dim": cast(int | None, getattr(config, "head_dim", None)),
    }


def _model_revision(model: _CausalLM) -> str | None:
    config = model.config
    # Return the model revision when the loaded model exposes one.
    revision = getattr(config, "revision", None)
    if revision is not None:
        return str(revision)
    return None


def _gpu_total_memory_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    # Read total memory for the active CUDA device.
    device_index = 0
    cuda_module = cast(_CudaModule, torch.cuda)
    props = cuda_module.get_device_properties(device_index)
    return float(props.total_memory / (1024**2))


def _save_result(result: BaselineResult, output_dir: Path, run_number: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"baseline_{time.time_ns()}_{run_number:03d}.json"
    # Write one baseline measurement as a numbered JSON file.
    result_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return result_path


def run_full_cache_baseline(
    model_name: str = DEFAULT_MODEL,
    prompt: str = "The future of memory in transformers is",
    max_new_tokens: int = 16,
    device: str | None = None,
    seed: int = 42,
    prompt_length: int | None = None,
    dtype: str = "float32",
) -> BaselineResult:
    cast(_TorchRandom, torch).manual_seed(seed)
    # Execute full-cache inference and collect all baseline measurements.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = _torch_dtype_for_name(dtype, device_name)

    tokenizer = cast(_TokenizerFactory, AutoTokenizer).from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt = _prompt_for_length(tokenizer, prompt, prompt_length)

    model = cast(_ModelFactory, AutoModelForCausalLM).from_pretrained(model_name)
    model.to(device_name)
    model.to(dtype=torch_dtype)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    inputs = tokenizer(prompt, return_tensors="pt").to(device_name)
    prompt_tokens = int(inputs["input_ids"].shape[1])

    prefill_time, decode_time, measured_cache = _measure_prefill_and_decode(
        model,
        inputs,
        max_new_tokens,
    )
    actual_cache_bytes, actual_cache_shapes = _actual_cache_metadata(measured_cache)
    actual_cache_seq_len = measured_cache.get_seq_length()
    del measured_cache

    _synchronize_device()
    start = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    _synchronize_device()
    total_time = time.perf_counter() - start

    generated_text = tokenizer.decode(
        generated_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    generated_tokens = int(generated_ids.shape[1] - prompt_tokens)
    full_seq_len = int(generated_ids.shape[1])
    estimated_bytes = _estimate_kv_cache_bytes(model, full_seq_len)
    peak_memory = _peak_gpu_memory_mb()

    result = BaselineResult(
        model_name=model_name,
        model_revision=_model_revision(model),
        prompt=prompt,
        seed=seed,
        max_new_tokens=max_new_tokens,
        generated_text=generated_text,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        total_time_s=total_time,
        tokens_per_second=(generated_tokens / total_time) if total_time > 0 else 0.0,
        prefill_time_s=prefill_time,
        decode_time_s=decode_time,
        decode_tokens_per_second=((max_new_tokens / decode_time) if decode_time > 0 else 0.0),
        estimated_kv_cache_bytes=estimated_bytes,
        actual_kv_cache_bytes=actual_cache_bytes,
        actual_kv_cache_seq_len=actual_cache_seq_len,
        actual_kv_cache_shapes=actual_cache_shapes,
        device=device_name,
        dtype=str(model.dtype),
        peak_memory_mb=peak_memory,
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        platform=platform.platform(),
        cpu_info=platform.processor() or platform.machine() or "unknown",
        gpu_device_name=(torch.cuda.get_device_name() if torch.cuda.is_available() else None),
        gpu_total_memory_mb=_gpu_total_memory_mb(),
        cuda_version=(torch.version.cuda if torch.cuda.is_available() else None),
        model_config=_model_config_metadata(model),
        cuda_device_name=(torch.cuda.get_device_name() if torch.cuda.is_available() else None),
    )
    return result


def main() -> None:
    import argparse
    # Parse baseline options and write the requested measurement files.

    parser = argparse.ArgumentParser(description="Run the full-cache baseline generation harness.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Small causal LM name or local path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of memory in transformers is",
        help="Prompt to send to the model.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=16, help="Number of tokens to generate."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--prompt-lengths", type=int, nargs="+", help="Target prompt lengths in tokens."
    )
    parser.add_argument(
        "--repeats", type=int, default=1, help="Number of runs for each prompt length."
    )
    parser.add_argument(
        "--dtypes",
        type=str,
        nargs="+",
        default=["float32"],
        choices=["float32", "float16", "bfloat16"],
        help="Supported precisions to evaluate. CPU float16 is skipped automatically when unsupported.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/baseline"),
        help="Directory for JSON results.",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.prompt_lengths and any(length < 1 for length in args.prompt_lengths):
        parser.error("--prompt-lengths values must be positive")

    prompt_lengths = args.prompt_lengths or [None]
    run_number = 0
    for dtype in args.dtypes:
        for prompt_length in prompt_lengths:
            for repeat in range(args.repeats):
                run_number += 1
                try:
                    result = run_full_cache_baseline(
                        model_name=args.model,
                        prompt=args.prompt,
                        max_new_tokens=args.max_new_tokens,
                        seed=args.seed + repeat,
                        prompt_length=prompt_length,
                        dtype=dtype,
                    )
                except RuntimeError as exc:
                    print(
                        f"Skipping dtype={dtype} for prompt_length={prompt_length}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                result_path = _save_result(result, args.output_dir, run_number)
                print(json.dumps(result.to_dict(), indent=2))
                print(f"Saved result to {result_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
