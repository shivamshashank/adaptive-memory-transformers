# PhD CV Project — 10-Day Execution Checklist

## Goal

Turn the current adaptive KV-cache compression prototype into a credible,
reproducible research pilot suitable for a PhD application and public portfolio.

The ten-day target is **not** to claim a general research breakthrough. The target
is a correct implementation, a defensible pilot experiment, reproducible evidence,
and a concise technical report. A negative result is acceptable if it is measured
and explained rigorously.

## Already available

- [x] Clear research question and hypothesis.
- [x] Full-cache inference baseline.
- [x] Analytical and measured KV-cache size reporting.
- [x] Prefill latency, decode latency, and throughput measurement.
- [x] Full, recency, and uniform cache-selection policies.
- [x] Initial adaptive token-selection policy.
- [x] `DynamicCache` pruning support.
- [x] Synthetic passkey retrieval evaluator and sweep runner.
- [x] Initial layer-budget allocation logic.
- [x] Ablation, quality-memory, and generalization experiment scripts.
- [x] Initial raw results and plots.
- [x] Automated unit tests: 27 currently passing.
- [x] Project architecture knowledge graph in `.ua/knowledge-graph.json`.

## Non-negotiable research blockers

- [ ] Prove that custom-cache decoding at 100% retention is equivalent to normal
      Transformers decoding.
- [ ] Preserve correct original token positions after arbitrary cache pruning.
- [ ] Make recency and historical-attention signals actually affect adaptive
      selection.
- [ ] Remove or rename the current fake "frequency" signal.
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

- [ ] Implement a deterministic reference test against standard Transformers
      decoding.
- [ ] Compare logits after prefill.
- [ ] Compare logits across multiple autoregressive decode steps.
- [ ] Compare generated token IDs under greedy decoding.
- [ ] Test at least one tiny GPT-style model and the intended primary model family.
- [ ] Define and document numerical tolerances by dtype.
- [ ] Test cache tensor shapes, sequence lengths, and byte accounting.
- [ ] Add a regression test for 100% retention equivalence.

### Exit gate

- [ ] The custom cache at 100% retention matches reference logits and generated
      tokens within the declared tolerance.

## Day 3 — Position-correct compressed cache

### Tasks

- [ ] Introduce a small model-family cache adapter instead of directly treating a
      pruned `DynamicCache` as a complete abstraction.
- [ ] Track every retained token's original position.
- [ ] Pass correct `position_ids` and/or `cache_position` during decoding.
- [ ] Construct the attention mask from retained positions correctly.
- [ ] Test non-contiguous retained positions.
- [ ] Test repeated pruning over several decode steps.
- [ ] Test early, middle, and recent retained tokens.
- [ ] Confirm that no removed position silently reappears or becomes reindexed.

### Exit gate

- [ ] Arbitrary retained-position sets pass model-backed semantic tests without
      position corruption.

## Day 4 — Correct adaptive policy signals

### Tasks

- [ ] Route the combined importance score into `select_indices`.
- [ ] Rank-normalize each signal per layer before combining it.
- [ ] Define true recency from original token positions.
- [ ] Replace "frequency" with a correctly named historical-attention-demand
      statistic, or remove it.
- [ ] Use current-query attention only to decide retention for the next step.
- [ ] Make weights, sink-token count, local-window size, and EMA horizon configurable.
- [ ] Add tests showing that changing each signal can change the retained set.
- [ ] Add tests for ties, missing signals, zero signals, and very small budgets.

### Exit gate

- [ ] Attention-only, attention-plus-recency, attention-plus-history, and all-signal
      variants produce meaningfully different selections on controlled fixtures.

## Day 5 — Unified matched-budget evaluator

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
