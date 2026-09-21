# Larger multi-seed query-aware development validation — 2026-09-21

Declared before inference. This remains development data, not the reserved final
confirmation and not a claim of significance, superiority or efficiency.

## Frozen data and execution

- Development seeds: 20260922 and 20260923. For each seed generate the same
  balanced 16 examples as the first longer-context experiment: eight target
  depths and balanced target colors at both 16 and 32 facts. Total 32 examples.
- Dataset SHA-256:
  `bc020e2e1a666e64fbea043bd9b35420fb13bd079fe17f63056535026a862f24`.
- Confirmation seed 20261001 is RESERVED. Do not generate, inspect, evaluate or
  tune against it in this run. Every artifact records that it remains unconsumed.
- Same pinned cached Qwen2.5-1.5B-Instruct, CPU float32/eager, direct-color chat
  prompt, bare single-token scorer, 16/32 facts and 50% context budget.
- Same four policies: full, protected recency, protected context-adaptive and
  protected two-pass query-adaptive. Query probes are discarded before a fresh
  scoring pass. Fixed adaptive weights remain 0.5 attention / 0.5 recency.
- Exact lengths remain 161/289 context tokens, 9 question-probe tokens, 32 full
  query tokens and 8 target-fact tokens. Budgets remain 80/144 context tokens.
- 32 examples × 4 policies = 128 scored evaluations plus 32 discarded probes.
  No prompt, weight, ratio, seed, threshold or example changes after outcomes.

## Metrics and predeclared decision

Baseline gate, applied separately to EACH development seed: full cache must score
at least 14/16 constrained and raw correct, at least 7/8 at each fact count, all
raw outputs must be valid colors, and all full/dense checks must pass.

Primary comparisons use only full-correct examples and are paired by example.
Report query-aware minus protected recency and query-aware minus context-adaptive:
wins/losses/ties, accuracy difference, deterministic 10,000-resample paired
percentile bootstrap interval (analysis seed 20260924), and an exact two-sided
binomial/McNemar sign test over discordant pairs. These inferential summaries are
descriptive because data and design remain developmental; no multiplicity claim.

Proceed to the untouched confirmation seed only if:

1. both per-seed baseline gates pass;
2. query-aware obtains at least four additional correct full-correct examples
   over EACH compressed baseline in the pooled 32-example development run; and
3. query-aware is not worse than either baseline in any of the four seed × fact-
   count cells on full-correct examples.

Otherwise preserve results, keep confirmation untouched and diagnose. Query-aware
uses an extra full-context probe, so even success does not establish memory,
latency or compute efficiency.

Command:
`uv run --frozen --group dev python -m scripts.run_query_aware_larger`
