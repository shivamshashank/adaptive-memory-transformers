# Day 3: position-correct compressed decoding

## The problem in plain language

The KV cache stores intermediate attention representations of past tokens. Each
entry has a physical slot in memory and an original position in the sequence.
Those numbers agree before pruning, but can differ afterwards:

| Physical slot | Original position before pruning | Original position after keeping 0, 3, 7 |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 1 | 3 |
| 2 | 2 | 7 |

Eight tokens have been processed, so the next query belongs at position **8**,
even though only three cached tokens remain. After appending that query, the
retained positions are `(0, 3, 7, 8)`. Selecting slots `[1, 3]` now keeps original
positions `(3, 8)`. It must never resurrect a previously removed position.

## Code walkthrough

### `src/amt/cache.py`: `Qwen2CacheAdapter`

The adapter owns the Transformers `DynamicCache` and two pieces of metadata:

- `positions`: original positions corresponding to the current physical slots.
  Day 3 uses the same retained slots in every layer.
- `seen_tokens`: total processed tokens. Pruning never decreases this counter.

`forward(input_ids)` performs these steps:

1. Validate a nonempty, single-sequence integer token tensor.
2. Assign new positions starting at `seen_tokens`.
3. Combine existing positions with the new query positions.
4. Build an additive attention mask: `0` permits attention; negative infinity
   blocks it. A key is permitted only when its original position is no greater
   than the query's position. This also prevents future-token leakage in chunks.
5. Call Qwen with explicit `position_ids`, the custom mask and the owned cache.
6. Update position metadata and return prediction logits for all new tokens.

Qwen2.5 uses the Qwen2 model implementation. That implementation applies rotary
position embeddings (RoPE) to keys before caching them. `retain(indices)` uses
`index_select` to gather cached keys and values **without modifying their
contents**. There is no second rotation or renumbering. It also gathers the
position map, while leaving `seen_tokens` unchanged.

`index_select` creates compact storage; it does not leave a small view holding
the original large allocation. Temporary allocations still occur while gathering.

The adapter accepts sorted, unique indices into the current cache. Invalid
indices are rejected before pruning. An empty retained set is supported, and
subsequent tokens still receive their original absolute positions.

### `src/amt/compressed.py`: `run_compressed_greedy_decode`

This is the first end-to-end compressed decoding path:

1. Prefill the prompt and predict the next token.
2. Ask the selected policy which cache slots to retain.
3. Prune and record retained positions, tensor shapes and KV bytes.
4. Append the predicted token using its original position.
5. Repeat until the requested number of tokens is generated.

It reuses the existing full, recency and uniform policies. Adaptive importance
scoring is Day 4 work. `FullCachePolicy` ignores the budget by design.

The trace includes the final predicted token in `sequences`; that final token
has not yet been processed into the cache. This matches the Day 2 convention.

Example, after loading an evaluation-mode Qwen2 model and tokenizing a prompt:

```python
from amt.compressed import run_compressed_greedy_decode
from amt.policies import UniformCachePolicy

trace = run_compressed_greedy_decode(
    model=model,
    input_ids=inputs["input_ids"],  # one sequence, no padding
    max_new_tokens=8,
    policy=UniformCachePolicy(),
    budget_tokens=32,
)
print(trace.retained_positions)
print(trace.decode.cache_snapshots[-1].total_bytes)
```

## How correctness is checked

`tests/test_cache.py` contains an independent dense masked-cache oracle. It
retains every historical KV entry, but masks removed positions on future calls.
The production adapter physically removes those entries. Their outputs should
agree within the relevant numerical tolerance.

The oracle deliberately does not reconstruct a shorter prompt: doing so would
change the earlier hidden states, which is a different computation. Earlier
retained representations may legitimately contain information from tokens that
were present when those representations were computed.

Tests compare logits, greedy token choices and every surviving layer's K/V
entries through repeated pruning. Other tests check full-retention equivalence
against standard generation, exact unchanged tensor values after gathering,
compact storage, invalid selections and supported configuration boundaries.

A deliberate wrong-position control sets a query to the compressed cache length
instead of its original position. The test requires the numerical comparison to
detect this bug.

## Scope and limitations

- One unpadded sequence; one shared retention map across all layers.
- Qwen2 full-attention layers, default RoPE, eager or SDPA attention only.
- No sliding-window, FlashAttention, offloading, batching or per-layer budgets.
- The explicit mask is quadratic during prefill and returns all query logits.
  This prioritizes correctness and is not the long-context benchmark runner.
- Budgets apply after pruning. Prefill holds the full prompt cache; a decode
  append temporarily exceeds the retained capacity before the next pruning.
- Reported KV bytes exclude masks, position metadata, model weights, transient
  allocations and allocator reservations. They are not peak device memory.
- The dense oracle exists only in tests; production does not retain a hidden
  full-cache copy.
- Correct cache semantics do not establish quality preservation or speedup.

## Validation commands

```bash
uv run pytest tests/test_cache.py

HF_HOME="$PWD/.model-cache/huggingface" HF_HUB_OFFLINE=1 \
AMT_RUN_QWEN_INTEGRATION=1 AMT_MODEL_DEVICE=cpu AMT_MODEL_DTYPE=float32 \
uv run pytest tests/test_cache.py::test_pinned_qwen_pruned_cache_semantics
```

The optional test uses immutable Qwen revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. `HF_HUB_OFFLINE=1` requires an
existing local model download. `AMT_MODEL_ATTENTION` selects `eager` or `sdpa`.

The pinned test has native and float64-attention diagnostic variants. The latter
keeps model weights in float32, changes attention arithmetic only within the
test, and runs on CPU. See [numerical evidence](NUMERICAL_TOLERANCES.md#day-3-comparisons-across-different-cache-shapes)
for the native float32 tolerance, original failed runs and unresolved BF16
qualification. These are short correctness fixtures, not long-context results.
