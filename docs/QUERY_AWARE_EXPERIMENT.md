# Query-aware adaptive selection experiment

## Why this experiment exists

The previous adaptive policy saw only the context before selecting tokens. With
many structurally identical records, it could not know which record a later
question would request. It scored 8/16 versus protected recency's 7/16, below the
predeclared advancement rule.

This experiment tests a single change: condition token selection on the question.
The [frozen protocol](../research/query_aware_protocol.md) defines the intervention,
equal-budget comparisons and decision rule before inference.

## Preventing answer leakage

Query-aware selection uses two passes:

1. A probe cache reads the full context and the nine-token question sentence.
2. Attention is averaged across question tokens and heads, separately by layer.
3. The selector chooses context tokens at the same 50% budget.
4. The probe cache is completely discarded.
5. A fresh context cache is built, physically pruned to that selection, and only
   then receives the complete question, instruction and held-back answer cue.

Therefore, hidden states produced while the probe saw full context cannot carry
the answer into scoring. This is an experimental selector, not yet an efficient
runtime method: it adds a full-context probe and its time/peak memory are worse.

## Comparisons

- Full cache: correctness reference.
- Protected recency: strongest simple baseline.
- Protected context-adaptive: prior fixed attention/recency policy.
- Protected query-adaptive: identical weights, but attention comes from question
  tokens instead of the final context token.

All compressed scoring passes retain 80/161 or 144/289 context tokens. Exact
positions, target-fact retention, physical cache bytes, outputs, logits, probe
signals and timings are saved. Full-cache rows must match dense inference.

`Qwen2CacheAdapter.forward` now supports `attention_query_reduction="mean"` while
retaining the prior default `"last"`. Unit tests independently compare both
reductions against model-returned attention tensors.

## Checklist

- [x] Freeze intervention, controls, budgets and decision threshold.
- [x] Hold probe states out of the scoring pass.
- [x] Verify exact 9-token question probe and unchanged dataset hash.
- [x] Test attention reduction, boundaries, evidence checks and decision rules.
- [x] Run all 64 scored evaluations and 16 discarded probes.
- [x] Verify artifacts and reproduce the three prior policy arms.
- [x] Report whether query-aware selection passes its advancement rule.

Advancement requires at least two additional correct answers over both compressed
baselines among full-correct cases, with no loss at either context size. Passing
would justify a larger development experiment—not a superiority claim.

## Findings — 2026-09-21

Local run: `results/query-aware/20260921T185433Z-1f070bb7`.
The independent saved-evidence verifier passed all 64 policy/example cells. The
baseline gate passed, and the query-aware decision was **proceed**.

| Policy | 16 facts | 32 facts | Overall | Correct among 14 full-correct cases | Mean target fact retained |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 7/8 | 7/8 | 14/16 | 14/14 | 100.0% |
| Protected recency | 4/8 | 3/8 | 7/16 | 7/14 | 52.3% |
| Protected context-adaptive | 4/8 | 4/8 | 8/16 | 8/14 | 57.0% |
| Protected query-adaptive | 5/8 | 5/8 | **10/16** | **10/14** | **65.6%** |

All 64 raw outputs were valid color words and raw correctness matched constrained
correctness. Full-cache staged/dense logits matched exactly. The 48 full, recency
and context-adaptive rows reproduced the previous run exactly: retained positions,
gold labels, predictions, raw token IDs and correctness all matched; maximum
absolute color-logit difference was 0.

On full-correct examples, query-aware improved three cases and regressed none
versus recency. Versus context-adaptive it improved three and regressed one,
meeting the frozen net +2 rule. It scored 5/7 at each fact count, so it was not
worse than either baseline at either size.

The selection change is directionally interpretable. Query-aware retained the
target color token in 12/16 cases versus 7/16 for context-adaptive, and retained
the complete eight-token target fact in 6/16 versus 3/16. It stopped universally
retaining generic `The` tokens (7/16 versus 16/16) and retained `locker` in 15/16.
These are post-outcome diagnostics, not causal proof or a tuned feature rule.

## Limitations and next step

Query-aware still answered only 10/14 full-correct examples and failed one case
despite retaining the complete target fact. The same 16 development examples
have now informed multiple design decisions. They cannot support a final claim.

The 16 probes added about 54.38 seconds on this machine and each temporarily used
the complete context cache. The experiment therefore demonstrates selection
utility, not lower peak memory, lower latency or end-to-end efficiency.

Next, freeze a larger development run on newly generated examples before looking
at outcomes. Keep the direct-color task, both fact counts, equal budgets and all
three compressed baselines. Add at least one second seed and a 25% budget only if
CPU cost remains acceptable. Keep a separate untouched confirmation seed for the
eventual manuscript result. Do not tune the 0.5/0.5 weights on this run.
