# PhD CV Project — 10-Day Execution Checklist

## Goal

Turn the current adaptive KV-cache compression prototype into a credible,
reproducible research pilot suitable for a PhD application and public portfolio.

The ten-day target is **not** to claim a general research breakthrough. The target
is a correct implementation, a defensible pilot experiment, reproducible evidence,
and a concise technical report. A negative result is acceptable if it is measured
and explained rigorously.

## Current implemented foundation

- [x] Clear research question and hypothesis.
- [x] Full-cache inference baseline.
- [x] Analytical and measured KV-cache size reporting.
- [x] Prefill latency, decode latency, and throughput measurement.
- [x] Full, recency, and uniform cache-selection policies.
- [x] Position-aware Qwen2 cache pruning and fixed-policy generation.
- [x] Automated unit tests and optional pinned-model integration tests.

Earlier proxy experiment scripts were removed during the Day 1 cleanup. Minimal
attention/recency scoring is now implemented. The Part 5 CPU development evaluator
is separate from the planned long-context benchmark and layer-budget allocation;
old prototype results are not evidence for the current implementation.

## Non-negotiable research blockers

- [x] Prove that custom-cache decoding at 100% retention is equivalent to normal
      Transformers decoding.
- [x] Preserve correct original token positions after arbitrary cache pruning
      in the supported Qwen2 adapter (see Day 3 numerical scope).
- [x] Make attention and recency actually affect selection in the reduced Part 4
      scope. Historical attention remains deferred, not implemented.
- [x] Remove the fake "frequency" signal (absent from the current package).
- [ ] Replace retained-attention coverage as the primary quality metric with an
      independent task metric.
- [ ] Apply layer-specific budgets to the physical per-layer caches during real
      decoding.
- [ ] Use benchmark examples that the full-cache reference model can solve.
- [x] Track all source, tests, scripts, documentation, and configurations in Git.
- [ ] Record exact model, tokenizer, dependency, hardware, seed, and Git revisions.

## Day 1 — Repository and reproducibility foundation

### Tasks

- [x] Review `.gitignore` and confirm that only generated/private files are ignored.
- [x] Add the real package, tests, configuration, and documentation to version control.
- [x] Add `pyproject.toml` with project metadata and Python version constraints.
- [x] Add separate runtime and development dependencies.
- [x] Generate and commit a reproducible dependency lockfile.
- [x] Add formatting, linting, and type-checking configuration.
- [x] Add a CI workflow that runs compilation, unit tests, linting, and type checks.
- [x] Create `research/protocol_v1.md` defining H1, H0, primary comparison,
      metrics, budgets, seeds, and exclusion rules.
- [x] Create the first immutable experiment configuration under `configs/`.

### Exit gate

- [x] A clean clone can install the environment and pass all fast checks using
      documented commands.

### Completion record

- Completed: 2026-09-16.
- Locked environment: `pyproject.toml` plus `uv.lock`, with Python 3.11–3.14
  support and separate runtime/development dependency groups.
- Immutable smoke configuration: `configs/smoke/v1.toml`, pinned to a specific
  model and tokenizer revision.
- Frozen protocol: `research/protocol_v1.md`.
- Local quality gate passed from a clean clone. The active `amt` package is
  fully included in MyPy; superseded proxy experiments were removed during the
  post-Day-1 repository cleanup.

### Post-foundation cleanup

- Completed: 2026-09-16.
- Consolidated the retained baseline, fixed policies, reporting, and config
  loading into one installable `amt` package and CLI.
- Removed the duplicate requirements file, wrappers, stale planning documents,
  and proxy experiment modules that could not support the registered claims.
- Verified a locked isolated install, CLI startup, compilation, formatting,
  linting, full-package MyPy, documentation links, and all 12 focused tests.

## Day 2 — Reference decoding oracle

### Tasks

- [x] Implement a deterministic reference test against standard Transformers
      decoding.
- [x] Compare logits after prefill.
- [x] Compare logits across multiple autoregressive decode steps.
- [x] Compare generated token IDs under greedy decoding.
- [x] Test a locally constructed and an immutable downloaded tiny GPT-style model.
- [x] Test the intended primary model family (`Qwen2.5-1.5B-Instruct`).
- [x] Define and document numerical tolerances by dtype.
- [x] Test cache tensor shapes, sequence lengths, and byte accounting.
- [x] Add a regression test for 100% retention equivalence.

### Exit gate

- [x] The custom cache at 100% retention matches reference logits and generated
      tokens within the declared tolerance on the intended primary model family.

### Current evidence

- Completed: deterministic CPU equivalence on a locally constructed GPT-2 and
  `hf-internal-testing/tiny-random-gpt2` at immutable revision
  `71034c5d8bde858ff824298bdedc65515b97d2b9`.
- Exact greedy token equality and float32 logit equivalence passed across prefill
  and multiple decode steps.
- Cache sequence growth, per-layer tensor shapes, and physical byte accounting
  are covered by regression tests.
- Completed: pinned `Qwen/Qwen2.5-1.5B-Instruct` revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` passed in BF16 on Apple M1 MPS.
- The Qwen run exposed and fixed an oracle confound: Transformers 5.13 inherited
  the model's repetition penalty unless every neutral greedy control was
  explicitly declared. With that penalty removed, logits, greedy tokens, and
  final cache metadata match.

### Completion record

- Completed: 2026-09-17.
- Hardware: Apple M1 GPU through PyTorch MPS, 16 GB unified memory.
- Primary candidate: `Qwen/Qwen2.5-1.5B-Instruct` in BF16.
- No paid compute was used.

## Day 3 — Position-correct compressed cache

### Tasks

- [x] Introduce a small model-family cache adapter instead of directly treating a
      pruned `DynamicCache` as a complete abstraction.
- [x] Track every retained token's original position.
- [x] Pass correct `position_ids` and/or `cache_position` during decoding.
- [x] Construct the attention mask from retained positions correctly.
- [x] Test non-contiguous retained positions.
- [x] Test repeated pruning over several decode steps.
- [x] Test early, middle, and recent retained tokens.
- [x] Confirm that no removed position silently reappears or becomes reindexed.

### Exit gate

- [x] Arbitrary retained-position sets pass model-backed semantic tests without
      position corruption.

### Completion record

- Completed: 2026-09-18, within the scope below.
- Implemented `Qwen2CacheAdapter` and fixed-policy compressed greedy decoding.
- Scope: one unpadded sequence, full attention, default RoPE, identical retained
  positions across layers. Multi-token append, repeated pruning and empty
  retention are covered. Per-layer budgets remain Day 7 work.
- All 40 Day 3 tests passed, including pinned Qwen2.5-1.5B CPU float32 and
  test-only float64-attention diagnosis; tiny Qwen fixtures cover eager and SDPA.
- Final full suite: 61 passed, no skips, with both pinned-model integration
  flags enabled and networking disabled. Formatting, lint, MyPy, compilation,
  CLI startup and lockfile checks passed. Two dependency deprecation warnings
  remain in Python package metadata loading.
- Compared actual outputs and surviving K/V tensors against an independent
  dense masked-cache oracle; deliberate wrong positions are detected.
- Native pinned-model float32 uses an empirically diagnosed comparison tolerance
  of `atol=2e-4`, `rtol=1e-5`. The diagnostic retains the original strict bound.
- BF16 compressed-cache equivalence remains unqualified following a failed MPS
  test. See `NUMERICAL_TOLERANCES.md` for all observed errors and scope.
- Walkthrough: `POSITION_CORRECT_CACHE.md`. No paid compute or new downloads.

## Day 4 — Correct adaptive policy signals (reduced scope)

Scope revised on 2026-09-18 for the user's no-paid-GPU plan. See
`research/deviations.md`; history/EMA and layer-specific allocation are deferred.

### Tasks

- [x] Route the combined importance score into `select_indices`.
- [x] Rank-normalize attention per layer before averaging; normalize recency.
- [x] Define recency from original token positions.
- [x] Remove "frequency"; do not claim a historical-attention signal.
- [x] Use current-query attention only to decide retention for the next step.
- [x] Make weights, sink-token count and local-window size configurable.
- [x] Add tests showing that changing each signal can change the retained set.
- [x] Add tests for ties, missing signals, zero signals, and very small budgets.

### Exit gate

- [x] Attention-dominant and recency-dominant variants produce different selections
      on controlled fixtures; attention-only and adaptive generation pass
      model-backed checks. The original history/all-signal gate is deferred.

### Completion record

- Implemented per-layer last-query attention collection, head averaging, rank
  normalization and shared-budget attention/recency selection.
- Explicit eager-attention requirement prevents silently changing SDPA models.
- Pinned Qwen2.5-1.5B CPU float32 test passed with attention collection, repeated
  pruning, both new policies and full-retention equivalence. No new model
  downloads or paid compute. BF16 remains unqualified.
- Fixed recency's `recent_window` capacity override and added a regression test.
- Full local quality gate passed. All three optional Part 3/4 pinned-Qwen cache
  tests also passed on CPU float32, including native eager-attention and the
  float64-attention diagnostic. Optional Day 2 downloads were not rerun here.
- Walkthrough: `ADAPTIVE_SELECTION.md`. No benchmark advantage is established.

## Day 5 — Unified matched-budget evaluator

### Reduced CPU development scope (2026-09-19)

- [x] Second, separately versioned balanced chat pilot: 32 examples / 288 rows,
      verified against declared `research/pilot_v2_protocol.md`. Every full-cache
      example matched dense inference; final quality checks passed (93 tests,
      five optional skips). See `PILOT_V2_FINDINGS.md`: no adaptive advantage;
      constrained always-A behavior and raw non-answer outputs require diagnosis.

- [x] Implement a shared delayed-query scoring path for all five policies.
- [x] Run full cache once per example; use identical context budgets for the
      four compressed policies (50% and 25%).
- [x] Save exact dataset, token IDs, label scores, retained positions, physical
      cache bytes, source snapshot and incremental logs/reports.
- [x] Audit physical K/V lengths and backing-storage sizes after context pruning
      and question append; test the audit against a deliberately oversized view.
- [x] Verify the tiny-model full-cache delayed-query path against dense inference.
- [x] Complete and inspect the 20-example / 180-evaluation development pilot.
      Run `20260918T232554Z-e48814cb`: all 180 rows complete and all physical
      budget audits passed. Independent recomputation confirmed row uniqueness,
      candidate predictions, accuracy/subset aggregates, capacities and dataset
      hash. Full cache 16/20; uniform beat adaptive at both tested budgets.
      See `EVALUATION_PILOT.md` for results and next steps.

The original checklist below is intentionally not all checked: the reduced
pilot budgets context before an uncompressed query, not total cache throughout
multi-token generation. See `EVALUATION_PILOT.md` and `research/deviations.md`.

### Tasks

- [ ] Create one evaluation path shared by every policy.
- [ ] Evaluate full cache once as an upper bound, not once per compressed budget.
- [ ] Compare recency, uniform, attention-only, and proposed adaptive policies.
- [ ] Use identical prompts, seeds, model revision, dtype, and generation settings.
- [ ] Enforce identical compressed-cache capacity for every budgeted method.
- [ ] Add an automatic budget audit after prefill and every decode step.
- [ ] Detect hidden cache copies remaining on CPU or GPU.
- [ ] Include signal collection, selection, and pruning in runtime measurements.
- [ ] Preserve failures, timeouts, and invalid outputs in the raw results.

### Exit gate

- [ ] No compressed policy exceeds its assigned capacity, and all policies are
      evaluated through the same decoding and measurement path.

## Day 6 — Independent quality benchmark

### Tasks

- [ ] Stop using retained attention mass as the primary quality metric.
- [ ] Keep attention coverage only as a diagnostic measurement.
- [ ] Add task accuracy and teacher-forced loss/perplexity as independent metrics.
- [ ] Add a small, versioned RULER development subset or equivalent controlled suite.
- [ ] Retain passkey retrieval as a diagnostic stress test.
- [ ] Use multiple needle depths and context lengths.
- [ ] Verify that the full-cache model solves each accepted benchmark cell.
- [ ] Exclude or label cells where the reference model cannot solve the task.
- [ ] Separate development examples from held-out confirmation examples.
- [ ] Save the example IDs, prompts, targets, scorer version, and raw completions.

### Exit gate

- [ ] The full-cache reference succeeds reliably enough for compression-related
      degradation or preservation to be measurable.

## Day 7 — Real layer-adaptive decoding

### Tasks

- [ ] Apply each calculated layer budget `B_l` to that layer's physical cache.
- [ ] Compare layer-adaptive allocation with equal-per-layer allocation.
- [ ] Match total layer-token capacity exactly between methods.
- [ ] Record the selected budget and retained positions for every layer.
- [ ] Enforce per-layer and total budget limits after every decode step.
- [ ] Include layer-allocation computation in overhead measurements.
- [ ] Add integration tests for different layer counts and limited capacities.
- [ ] Defer head-level allocation unless the layer implementation is validated.

### Exit gate

- [ ] Saved cache shapes prove that the requested layer-specific allocations were
      applied during actual decoding under an equal total budget.

## Day 8 — Statistics and experiment ledger

### Tasks

- [ ] Save one raw record per model/example/policy/budget/seed combination.
- [ ] Add a run manifest containing the exact command and immutable configuration.
- [ ] Record Git SHA and whether the working tree was dirty.
- [ ] Record model and tokenizer identifiers plus immutable revisions.
- [ ] Record Python, PyTorch, Transformers, CUDA, driver, GPU, dtype, and cache type.
- [ ] Record analytical cache bytes, tensor cache bytes, and peak allocated VRAM.
- [ ] Measure prefill, steady-state decoding, signal collection, selection, and pruning.
- [ ] Add warm-up runs and synchronized timing on CUDA.
- [ ] Add paired bootstrap confidence intervals for policy differences.
- [ ] Define how OOM, timeout, invalid output, and missing rows are counted.
- [ ] Add a validation command that rejects incomplete or mismatched experiment groups.

### Exit gate

- [ ] Every summary row can be traced to raw examples, configuration, code revision,
      model revision, and hardware metadata.

## Day 9 — Pilot and red-team review

### Pilot matrix

- [ ] Use one capable primary model.
- [ ] Use at least two or three context lengths feasible on available hardware.
- [ ] Use 50%, 25%, and 12.5% cache budgets.
- [ ] Run full, recency, uniform, attention-only, and proposed adaptive policies.
- [ ] Use multiple fixed seeds.
- [ ] Include retrieval and at least one reasoning-style task.

### Red-team checks

- [ ] Confirm 100% custom-cache equivalence again on the pilot model.
- [ ] Confirm no policy uses future information.
- [ ] Confirm every cache remains within budget at every step.
- [ ] Confirm ablation variants actually produce different selections.
- [ ] Reorder prompts and verify conclusions do not depend on execution order.
- [ ] Inspect cases where simple fixed policies beat the adaptive method.
- [ ] Inspect the largest loss/perplexity regressions manually.
- [ ] Confirm no confirmation examples were used for tuning.

### Exit gate

- [ ] Proceed only if semantic equivalence, budget compliance, raw logging, and
      benchmark sensitivity all pass. Otherwise document a no-go decision and fix
      the failed gate before larger experiments.

## Day 10 — PhD CV research artifact

### Tasks

- [ ] Generate every table and figure directly from saved raw records.
- [ ] Produce a quality-versus-cache-memory plot.
- [ ] Produce a quality-versus-decode-latency plot.
- [ ] Report confidence intervals and sample counts.
- [ ] Include negative results and fixed-policy wins.
- [ ] Update the README with exact reproduction commands.
- [ ] Update the README, roadmap, protocol, and this checklist honestly.
- [ ] Write a concise limitations and threats-to-validity section.
- [ ] Write a 4–6 page technical report or extended research note.
- [ ] Add a clear repository status: prototype, validated pilot, or confirmatory study.
- [ ] Create one CV bullet that describes only demonstrated results.
- [ ] Decide whether the project is ready for a larger GPU campaign.

### Exit gate

- [ ] A reviewer can understand the question, reproduce the pilot, inspect raw
      evidence, and see exactly which claims are and are not supported.

## Recommended CV wording

### Safe wording now

- [ ] Use wording similar to:

  > Built a PyTorch and Hugging Face research prototype for adaptive Transformer
  > KV-cache compression, including budgeted retention policies, cache pruning,
  > memory/latency instrumentation, synthetic retrieval evaluation, and automated
  > tests.

- [ ] Do not claim that the adaptive method currently improves quality, memory, or
      latency.

### Wording after the ten-day sprint

- [ ] Use wording similar to:

  > Developed a reproducible evaluation framework for position-correct KV-cache
  > compression, comparing adaptive and fixed policies under matched memory budgets
  > using task quality, latency, memory, and paired statistical analysis.

### Wording only after confirmatory experiments

- [ ] State the actual number of models, tasks, lengths, budgets, and examples.
- [ ] Report the measured effect rather than saying the method is simply "better."
- [ ] Link the public repository, technical report or preprint, and reproducibility
      artifact.

## Work after the ten-day sprint

- [ ] Freeze the confirmatory protocol before running it.
- [ ] Run the held-out RULER confirmation set.
- [ ] Evaluate LongBench without tuning on it.
- [ ] Run NeedleBench as a retrieval-and-reasoning stress test.
- [ ] Replicate on a second, architecturally different model family.
- [ ] Run longer contexts and more seeds on appropriate CUDA hardware.
- [ ] Produce final paired statistical analysis and Pareto frontiers.
- [ ] Release compact results and checksums publicly.
- [ ] Archive large artifacts through an appropriate public research repository.
- [ ] Expand the technical report into a workshop paper or preprint if justified.

## Definition of PhD-CV ready

- [ ] Public repository contains the actual implementation and tests.
- [ ] One command reproduces the main pilot table or figure.
- [ ] Cache correctness is supported by model-backed equivalence tests.
- [ ] Baselines use genuinely matched memory budgets.
- [ ] Quality is measured independently of the selection signal.
- [ ] Multiple seeds and paired uncertainty estimates are reported.
- [ ] Raw failures and negative results are preserved.
- [ ] Claims in the README, report, and CV match the evidence.
- [ ] The project has a short technical report with limitations.
- [ ] The repository is understandable to a prospective supervisor within ten
      minutes.
