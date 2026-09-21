# Decoding equivalence and numerical tolerances

## Purpose

The compressed-cache experiments are meaningful only if the custom decoding
path behaves like standard Transformers generation before any cache entries are
removed. `src/amt/decoding.py` therefore provides two independent paths:

1. `run_reference_greedy_decode` uses `PreTrainedModel.generate` as the oracle.
2. `run_manual_full_cache_decode` performs prefill and then feeds one selected
   token at a time while retaining the complete KV cache.

The reference uses a neutral generation configuration: greedy selection, no
sampling, no early EOS termination, and no model-specific logits processors.
This isolates cache-decoding semantics from sampling and chat-generation policy.

## Equivalence contract

For a prompt and a fixed generated-token count, the two paths must satisfy all
of the following:

- generated token IDs are exactly equal;
- prefill logits match within the declared dtype tolerance;
- logits match after every autoregressive step within the same tolerance;
- the final cache sequence length and per-layer key/value shapes agree; and
- measured cache bytes equal the sum of the physical key/value tensors.

`torch.testing.assert_close` applies the following elementwise condition:

```text
absolute_difference <= absolute_tolerance + relative_tolerance * abs(reference)
```

## Declared tolerances

| Dtype | Absolute tolerance | Relative tolerance | Validation status |
|---|---:|---:|---|
| `float32` | `1e-5` | `1e-5` | Validated on the local and pinned tiny GPT-2 tests |
| `float16` | `1e-3` | `1e-3` | Declared; target-device validation pending |
| `bfloat16` | `1e-2` | `1e-2` | Validated on pinned Qwen2.5-1.5B using Apple M1 MPS |

Token IDs must match exactly for every dtype. A token mismatch cannot be waived
because its logits are numerically close.

The lower-precision thresholds are initial pre-declarations, not permission to
ignore large errors. A failure must be diagnosed first. Any tolerance change
must record the model, dtype, device, maximum observed error, reason, and date.

## Cache-length convention

The prefill logits predict the first generated token. That token enters the KV
cache only when it is fed back to predict the second token. Consequently, a
trace that produces `N` tokens has cache snapshots with lengths:

```text
prompt_length, prompt_length + 1, ..., prompt_length + N - 1
```

Running an additional model call merely to place the final generated token in
the cache would predict an unused extra token and distort decode timing.

## Commands

Run the offline test, which constructs a tiny GPT-2 model entirely in memory:

```bash
uv run pytest tests/test_decoding.py
```

Run the optional integration test against the immutable Hub revision configured
for the smoke model:

```bash
AMT_RUN_MODEL_INTEGRATION=1 uv run pytest tests/test_decoding.py
```

The integration test downloads model artifacts when they are not already in the
local Hugging Face cache. Normal CI skips it so pull requests do not depend on
network access.

## Completed Day 2 gate

The Day 2 gate passed for pinned `Qwen/Qwen2.5-1.5B-Instruct` revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306` in BF16 on Apple M1 MPS. Standard
Transformers generation and manual full-cache decoding produced identical raw
logits, greedy tokens, and final cache metadata.

Run the Qwen check from an already populated local model cache with:

```bash
HF_HUB_OFFLINE=1 \
AMT_RUN_QWEN_INTEGRATION=1 \
AMT_MODEL_DEVICE=mps \
AMT_MODEL_DTYPE=bfloat16 \
uv run pytest tests/test_decoding.py::test_pinned_qwen_matches_transformers_generation
```

CUDA latency and memory measurements remain future systems experiments; they
are not required for this semantic equivalence result.

## Day 3: comparisons across different cache shapes

The Day 2 tolerances above remain unchanged. Day 3 compares a physically pruned
cache with a dense cache whose deleted columns are masked. These paths perform
attention reductions over different physical dimensions, so their rounding is
not necessarily the same, even when their mathematical semantics agree.

The following investigation was performed on 2026-09-18 with pinned
`Qwen/Qwen2.5-1.5B-Instruct` revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306` on the Apple M1 host:

| Configuration | Observation |
| --- | --- |
| MPS BF16, native eager attention | Failed original `1e-2` tolerance; first failing comparison reported maximum error `0.28125`. Not qualified for compressed-cache equivalence. |
| CPU float32, native eager attention | Failed original `1e-5` tolerance; first failing comparison reported error `4.74453e-5`. |
| CPU float32, native SDPA | Failed original `1e-5` and provisional `1e-4` absolute tolerances. Full-schedule characterization found maximum absolute error `1.35422e-4` across 285 logit/K/V tensor comparisons; greedy choices matched. |
| CPU float32 weights, test-only float64 attention | Entire pruning schedule passed original `1e-5` absolute/relative tolerances, isolating sensitivity to attention arithmetic. This is a diagnostic, not production inference. |
| CPU float32 SDPA, deliberately incorrect query position | Maximum absolute logit difference `6.41412`; the numerical oracle detects this positional bug. |

For the **native float32 pinned-model Day 3 test only**, the pilot comparison
uses `atol=2e-4`, `rtol=1e-5`. This is an empirical tolerance chosen after the
above diagnosis, not a preregistered result or a universal error bound. Exact
greedy-token equality remains mandatory. The wrong-position control must also
fail at this tolerance. Small Qwen fixtures and the float64-attention diagnostic
retain the stricter original bound.

No BF16 tolerance was relaxed. BF16 compressed inference, other devices, larger
contexts and longer pruning schedules need separate numerical qualification
before being used for research claims. Re-running the optional native integration
test in BF16 can still fail; this is a recorded limitation, not an expected pass.

Part 4 regression on 2026-09-18 also passed the same pinned-model native
float32 comparison using **eager** attention at `atol=2e-4`, `rtol=1e-5`, and
the strict float64-attention diagnostic. The new attention collection/generation
test passed on eager CPU float32. This does not extend validation to BF16 or
long contexts.
