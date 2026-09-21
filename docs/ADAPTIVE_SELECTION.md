# Part 4: attention and recency selection

## What was built

The current policy ranks cached tokens by current-query attention and recency.
It is an inference-only heuristic, not a trained controller, a reproduction of
H2O, or a claim of a new algorithm. Historical attention, EMA and layer-specific
budgets are deferred under the reduced, zero-paid-compute project scope.

## Follow one generation step

1. Qwen processes the current token (or prompt on the first call).
2. The adapter reads attention for the last query in this call. It averages the
   attention probabilities across query heads separately in each layer.
3. The policy rank-normalizes those values within each layer, then averages the
   layer ranks to produce one attention rank per current cache slot.
4. Original token positions are rank-normalized to give a recency rank.
5. The policy combines the two ranks and chooses the survivors.
6. The adapter physically prunes keys, values, position metadata and score
   metadata together. The next model invocation sees the pruned cache.

Only information available at the current step is used. A generated token is
not processed until the next iteration. The prefill selection uses only the
last prompt query; it does not average over all prompt queries. There is no
cross-step accumulation of attention in this version.

## The score

For token slot i, with nonnegative weights a and r:

```text
A[i] = mean over layers(rank(mean over heads(last-query attention to i)))
R[i] = rank(original position of i)
score[i] = (a * A[i] + r * R[i]) / (a + r)
```

Rank normalization maps the lowest rank to 0 and highest to 1. Ties share their
average rank; a constant signal receives 0.5 everywhere. Each layer is normalized
before aggregation, so raw attention scale does not determine its influence.
Recency ranks encode order, not the magnitude of age gaps. Original positions
also determine the protected local window and the original sink region.

Example: attention prefers tokens at positions 0 and 3, while recency prefers
positions 9 and 20. Weights `(0.9, 0.1)` select the first pair; `(0.1, 0.9)`
select the second. Regression tests require this change to occur.

Default adaptive weights are `(0.5, 0.5)` as an unvalidated starting point.
Attention-only uses `(1.0, 0.0)`. These are not tuned or claimed optimal.

## Code responsibilities

- `src/amt/cache.py`: `forward(..., collect_attention=True)` requests Qwen's
  attention weights and reduces them to per-layer score vectors. It rejects
  SDPA for this request before model execution. `retain` gathers scores with
  the cache, and subsequent calls clear stale scores.
- `src/amt/policies.py`: `RetentionSignals` carries aligned positions and scores.
  `rank_normalize` implements tie-aware normalization. `AttentionRecencyPolicy`
  validates inputs, combines signals, enforces protection rules and selects
  tokens. `create_policy("attention")` and `create_policy("adaptive")` expose
  the two standard configurations.
- `src/amt/compressed.py`: passes current signals into the policy only after
  forward execution, then prunes. Fixed policies accept and ignore the signals.

## Protection, budgets and ties

Both `sink_tokens` and `local_window` default to zero. They can be configured
on the policy, but protections consume the SAME retained-token budget.

- Sinks mean original positions less than `sink_tokens`. A token moved into
  physical slot zero by pruning does not become an original sink token.
- A local window of W protects retained positions greater than `latest - W`.
  Missing positions are not filled or resurrected.
- If protections exceed capacity: keep earliest sinks first, then newest local
  tokens, stopping at the budget.
- Fill remaining capacity by combined score. Equal scores prefer newer tokens.
- Return selected indices sorted in sequence order for the adapter.
- Missing attention with a positive attention weight is an error. All-zero
  attention is valid and uses the stated tie rule. Non-finite, negative or
  misaligned scores are rejected.

The existing recency policy was also repaired: `recent_window` can no longer
override the retained-token budget.

## Example

Load the pinned Qwen model with `attn_implementation="eager"`, in evaluation
mode, and tokenize a single unpadded prompt. Then:

```python
from amt.compressed import run_compressed_greedy_decode
from amt.policies import AttentionRecencyPolicy

policy = AttentionRecencyPolicy(
    attention_weight=0.75,
    recency_weight=0.25,
    sink_tokens=1,
    local_window=4,
)
trace = run_compressed_greedy_decode(
    model, inputs["input_ids"], max_new_tokens=8,
    policy=policy, budget_tokens=32,
)
print(trace.retained_positions)
```

These example weights are illustrative, not empirical findings. The baseline
CLI remains unchanged; this functionality is currently a Python API.

## What the tests establish

Pure-policy tests cover signal effects, scale invariance, per-layer normalization,
ties, zero/missing/invalid scores, original-position protections and small budgets.
Model-backed tests compare collected scores against Qwen's direct attention
outputs, check alignment after pruning, run both policies through repeated
generation, and verify full-retention output equivalence. The first selection
is unchanged when the requested generation horizon is extended.

The optional pinned-model test uses Qwen2.5-1.5B revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, CPU float32, eager attention:

```bash
HF_HOME="$PWD/.model-cache/huggingface" HF_HUB_OFFLINE=1 \
AMT_RUN_QWEN_INTEGRATION=1 uv run pytest \
  tests/test_cache.py::test_pinned_qwen_attention_selection
```

## Limits before experiments

Eager attention can materialize large attention matrices during prefill. Score
reduction, transfer to CPU, sorting and pruning all have overhead. This path is
for correctness development; it has not demonstrated a memory or latency win.
Later timing comparisons must declare backend differences and include overhead,
with an eager baseline to isolate policy effects. BF16 compressed inference is
still not numerically qualified. Use the recorded Day 3 scope and tolerances.

Attention is a selection signal, not an answer-quality metric. The next phase
must use independent task scores or text loss, held-out examples, and a properly
specified published comparison method before any research advantage is claimed.
