# Query-aware selection probe — 2026-09-21

Declared before inference. Development proof-of-concept only; no efficiency,
novelty, significance, superiority or generalization claim.

## Motivation and frozen intervention

The query-unaware 50% experiment produced adaptive 8/16 versus recency 7/16,
below its +2 advancement rule. Its selector never saw which record was requested.
This experiment tests whether question-conditioned attention improves selection.

- Reuse the exact 16-example dataset, model, chat prompt, direct-color scorer,
  16/32-fact sizes and 50% context budgets from `long_context_protocol.md`.
  Dataset SHA-256 remains
  `2a88cf55ef59b7909af5de5f2a32725b52f94074e0a2c3e24fa20e34664d7a62`.
- Compare full cache, protected recency, protected context-adaptive (the existing
  last-context-token attention/recency policy), and protected query-adaptive.
  Context-adaptive and query-adaptive both use fixed 0.5 attention / 0.5 recency.
- For query-adaptive only, run a selection probe over the full context followed
  by ONLY the question sentence (`What color is the UnitNN locker?\n`). Average
  attention across its query tokens and heads per layer, then slice to context
  positions. Select the 50% context budget with the shared protected wrapper.
- Discard the entire probe cache. Build a fresh context cache, retain the selected
  context positions, append the complete original query/instruction/answer cue,
  then score. Thus probe hidden states cannot carry the answer past pruning.
- Other policies use the same fresh scoring path. Query-adaptive has an additional
  full-context probe and is NOT a latency, peak-memory or compute improvement.
  Timing is logged only for audit. No prompt, ratio, weight or dataset tuning.
- 16 examples × 4 policies = 64 scored evaluations, plus 16 discarded probes.

## Checks and decision rule

Freeze exact token lengths: 161 or 289 context tokens, 9 question-probe tokens,
32 complete query tokens and eight target-fact tokens. Compressed budgets remain
80 and 144. Independently verify gold, boundaries, unique policy cells, physical
cache budgets/storage, target span retention and full-cache dense equivalence.
Save probe signals, selections, logits, raw outputs, source, logs and partial runs.

The baseline gate remains ≥14/16 full constrained and raw correct, ≥7/8 at each
size, all raw outputs valid colors, and all dense checks passing. Primary policy
comparison is paired accuracy on full-correct examples.

Query-aware selection advances to a larger development run only if the baseline
gate passes, it obtains at least two more full-correct answers than BOTH protected
recency and context-adaptive overall, and it is not worse than either at either
fact count. This is a project decision threshold, not statistical evidence.
Otherwise preserve the result and diagnose token selection. Fresh confirmation
examples remain unused.

Command: `uv run --frozen --group dev python scripts/run_query_aware.py`
