# Research Protocol v1

**Status:** frozen for the ten-day validation sprint
**Project:** Adaptive KV-Cache Compression
**Primary purpose:** determine whether a causal, position-correct adaptive
retention policy improves task quality over strong fixed policies at the same
physical KV-cache capacity.

Any change to this protocol must be recorded in a dated deviation log before the
held-out confirmation run. Pilot failures may narrow the study, but results must
not be used to silently redefine success.

## Research question

At an equal physical KV-cache budget, can an adaptive policy retain more
long-context task quality than recency, uniform, and attention-only retention
without hiding signal-collection, selection, or pruning overhead?

## Hypotheses

**H1:** On the held-out controlled benchmark, the proposed adaptive policy has a
positive paired task-quality difference from the strongest fixed budget-matched
baseline at one or more pre-declared low-memory budgets, with a 95% paired
bootstrap confidence interval excluding zero.

**H0:** The proposed policy provides no statistically distinguishable task-quality
advantage over the strongest fixed budget-matched baseline. This is an acceptable
and reportable outcome.

## Claim boundaries

The project may claim only what the final evidence directly supports:

1. Custom-cache correctness when 100% retention matches reference decoding within
   a declared numerical tolerance.
2. Measured task quality, cache bytes, and runtime under the recorded protocol.
3. An adaptive advantage only when it survives a paired comparison against the
   strongest matched-budget fixed baseline.

The project must not describe retained attention mass as task quality, a tiny-model
smoke test as long-context evidence, or full cache as a restricted-budget baseline.

## Primary comparison

- **Proposed method:** causal adaptive retention using rank-normalized historical
  attention demand, true recency from original positions, current-query attention
  for the next decoding step, a fixed attention-sink allowance, and a fixed local
  window.
- **Primary baseline:** the strongest of recency and uniform retention on the
  development split, chosen before confirmation.
- **Secondary baseline:** attention-only retention.
- **Upper bound:** full cache, evaluated once per example rather than repeated at
  every compressed budget.

All budgeted policies must use the same model, revision, tokenizer, prompt,
precision, decoding settings, generated-token limit, and cache adapter.

## Budgets and contexts

- Development budgets: 50%, 25%, and 12.5% of the reference token capacity.
- Confirmation budgets: the same three values, frozen before the run.
- Initial development contexts: 4K, 8K, and 16K tokens where supported by the
  chosen model and available hardware.
- A context or budget cell is invalid if the full-cache reference cannot solve the
  task or the model does not natively support the requested context.

The full experiment may be narrowed before confirmation when compute is
insufficient. Such narrowing must be documented before inspecting confirmation
outcomes.

## Model acceptance

The primary model and tokenizer must be pinned to immutable revisions. A candidate
model is accepted only when:

- the model licence permits the intended public research artifact;
- the native context window covers the configured evaluation length;
- the full-cache reference passes the benchmark sensitivity threshold;
- the custom cache passes 100%-retention equivalence tests; and
- the model fits the recorded hardware without offloading that would invalidate
  latency or memory comparisons.

The immutable tiny-model configuration in `configs/smoke/v1.toml` validates the
software pipeline only. It is not the primary research model.

## Data and split discipline

- Use a versioned controlled long-context development suite, preferably a small
  RULER-derived subset or an equivalent locally generated suite.
- Freeze disjoint development and confirmation example identifiers.
- Tune coefficients, EMA horizon, sink allowance, and local window only on the
  development split.
- Do not tune on LongBench, NeedleBench, or the confirmation split.
- Save prompt, target, task, context length, depth, source split, seed, and scorer
  version for every example.

## Primary and secondary metrics

**Primary metric**

- Official task accuracy or task score, paired by exact example.

**Secondary quality metrics**

- Teacher-forced next-token loss and perplexity where a valid continuation exists.
- Exact retrieval or answer match by task.
- Retained attention mass as a diagnostic only.

**Systems metrics**

- Physical KV-cache tensor bytes.
- Peak allocated GPU memory.
- Prefill latency.
- Steady-state decode latency and tokens per second.
- Signal-collection, selection, and pruning overhead.

## Randomness and repetition

- Development seeds: 41, 42, and 43.
- Confirmation seeds: 101, 202, and 303.
- Greedy decoding is used for deterministic retrieval comparisons unless a task's
  official protocol requires another decoding method.
- Timing cells require warm-up runs and at least 30 measured repetitions on CUDA.

## Statistical analysis

- Pair policies by model, example ID, task, context, budget, seed, and decoding
  configuration.
- Report the paired mean difference and a 95% paired bootstrap confidence interval.
- Use 10,000 bootstrap resamples for confirmatory quality comparisons.
- Report task-level results before any macro average.
- Report sample counts, failed-run counts, and missing-data reasons.
- Plot quality versus physical cache bytes and quality versus decode latency.

## Exclusion and failure rules

- A full-cache failure makes the benchmark cell non-discriminative; retain and
  label it, but exclude it from the primary compression comparison.
- OOM, timeout, malformed output, or scorer failure is recorded as a failed run and
  is never silently removed.
- Any budget violation invalidates that policy run.
- Any 100%-retention equivalence failure blocks compressed research claims for that
  model family.
- Any use of future attention or target information makes a policy an oracle; it
  must be relabelled and excluded from the primary comparison.
- Results produced from a dirty working tree remain diagnostic unless the exact
  diff is archived with the run.

## Reproducibility record

Every run must record:

- immutable configuration and exact command;
- Git SHA and dirty state;
- model and tokenizer identifiers and revisions;
- dataset or generator version and example manifest hash;
- Python, PyTorch, Transformers, CUDA, driver, and dependency-lock hashes;
- device, dtype, cache type, seed, start time, and finish time;
- raw per-example output, failure state, and measured metrics.

## Sprint success criteria

The ten-day sprint succeeds when:

1. a clean clone passes compilation, formatting, linting, type checks, and tests;
2. 100%-retention custom-cache decoding matches the reference;
3. every signal ablation changes the selection when expected;
4. all budgeted policies remain within physical cache limits;
5. the full-cache model solves the accepted pilot benchmark cells;
6. the pilot reports paired task and systems metrics from raw records; and
7. a reader can reproduce the main pilot table or figure with one documented
   command.
