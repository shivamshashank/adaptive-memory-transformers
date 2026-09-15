# Research Phases

## Adaptive KV-Cache Compression for Long-Context Transformers

This document turns the project brief into ten bounded phases. Each phase has a
research question, implementation work, experiments, outputs, and an exit
criterion. A phase is complete when its evidence is recorded, not merely when
its code exists.

## Phase 1: Literature review and gap definition

### Question

What has already been established about KV-cache growth, token eviction,
attention-based retention, quantization, pooling, sparse attention, adaptive
memory, and long-context evaluation?

### Work

- Build a literature matrix with problem, method, model, budget, evaluation,
  finding, limitation, and reproducibility notes.
- Cover MHA, MQA, GQA, decoding memory bandwidth, token selection, KV
  quantization, memory compression, and lost-in-the-middle behavior.
- Study AToM and closely related adaptive-memory work without reproducing it
  uncritically.
- Define the smallest defensible research gap.
- Freeze the primary research question, H1, H0, primary metric, and primary
  comparison before tuning the adaptive method.

### Deliverables

- `research/literature-matrix.md`
- `research/research-question.md`
- Initial evaluation protocol and threat-to-validity register.

### Exit criterion

The project can explain what is new, what is not new, and what result would
falsify H1. If the literature closes the proposed gap, revise the question
before implementation continues.

## Phase 2: Full-cache baseline inference

### Question

Can a clean inference harness measure standard full-cache generation reliably?

### Work

- Load one small openly available causal model through Hugging Face
  Transformers.
- Record model name and revision, parameter count, layers, hidden size,
  attention heads, KV heads, context limit, dtype, GPU, CUDA, and library
  versions.
- Implement tokenize, prefill, cache construction, autoregressive decode, and
  cache append measurement.
- Compare output behavior with standard Transformers generation under fixed
  decoding settings.

### Experiments

- Short deterministic prompts on CPU or GPU.
- Multiple prompt lengths where supported.
- Greedy decoding with a fixed generated-token budget.

### Deliverables

- Baseline runner and configuration.
- Per-run JSON metadata and timing results.
- Baseline validation notes.

### Exit criterion

The harness completes repeated runs, reports consistent tensor shapes and
metrics, and agrees with standard generation closely enough to establish the
reference condition.

## Phase 3: KV-cache memory scaling

### Question

How does cache memory scale with sequence length, model structure, and
precision?

### Work

- Implement the analytical estimate:

  ```text
  2 × layers × KV heads × sequence length × head dimension × bytes
  ```

- Measure actual cache tensor sizes and peak allocated GPU memory.
- Sweep sequence lengths such as 2K, 4K, 8K, 16K, and 32K when hardware allows.
- Repeat selected measurements for supported precisions.
- Record prefill latency, decode latency, and decode throughput.

### Experiments

- Analytical versus observed KV bytes.
- KV bytes versus sequence length.
- Peak GPU memory versus sequence length.
- Decode throughput versus sequence length.

### Deliverables

- `results/memory_scaling/` raw outputs.
- Reproducible Figure 1: KV bytes versus sequence length.
- Reproducible Figure 2: peak GPU memory versus sequence length.
- Reproducible Figure 3: throughput versus sequence length.

### Exit criterion

The scaling relationship and deviations from the analytical estimate are
quantified. Any unexplained discrepancy is documented before compression work.

## Phase 4: Token importance analysis

### Question

Do measurable attention, position, recency, frequency, layer, or head signals
correlate with which cached states remain useful later?

### Work

- Instrument attention or an equivalent supported signal without changing the
  baseline evaluation protocol.
- Aggregate attention mass by token position, layer, and head.
- Compute recency, repeated-attention frequency, and position features.
- Separate observations used for method development from held-out evaluation.
- Avoid treating attention received now as ground-truth future utility.

### Experiments

- Layer by token-position heatmaps.
- Head by token-position heatmaps.
- Recency and attention-mass distributions.
- Correlation between an early importance signal and later token utility.

### Deliverables

- Importance-analysis data schema.
- Figures for layer, head, position, recency, and frequency patterns.
- A pre-declared candidate signal set.

### Exit criterion

The project has evidence for or against a useful signal. If no signal is
predictive, retain that negative result and test simple baselines anyway.

## Phase 5: Fixed compression baselines

### Question

How much quality and efficiency do simple, non-adaptive policies preserve at
equal cache budgets?

### Work

- Implement full cache, recency retention, and uniform retention.
- Define exact retention semantics and budget accounting.
- Preserve a common recent-token window if required by the decode mechanism.
- Ensure policies use the same prompts, model, dtype, and generated-token limit.

### Experiments

- Budgets at 100%, 75%, 50%, 25%, and 10%.
- Perplexity or next-token loss on controlled sequences.
- Retrieval and synthetic long-context tasks where available.
- Memory, latency, and throughput for every budget.

### Deliverables

- Policy interface and unit tests.
- Baseline comparison table.
- Raw per-example and per-run outputs.

### Exit criterion

The simple baselines are correct, budget-compliant, and strong enough to make
an adaptive result meaningful. A new method is not evaluated against full cache
alone.

## Phase 6: First adaptive token policy

### Question

Can a simple measurable importance score improve over fixed retention at the
same budget?

### Work

- Normalize candidate signals before combining them.
- Start with a transparent score such as:

  ```text
  importance = alpha * attention + beta * recency + gamma * frequency
  ```

- Choose weights on development data only.
- Select tokens under the exact same budgets used in Phase 5.
- Measure the cost of collecting the signal and selecting entries.

### Experiments

- Attention only.
- Recency only.
- Frequency only.
- Attention plus recency.
- Attention plus frequency.
- Recency plus frequency.
- All three signals.

### Deliverables

- Adaptive token-selection implementation.
- Development-set tuning record.
- First quality-memory-latency comparison.

### Exit criterion

The method either shows a reproducible advantage under at least one pre-defined
condition or provides a clear, localized failure that informs Phase 7. No
weights are changed after looking at final test results.

## Phase 7: Layer and head adaptive allocation

### Question

Is a global token budget sufficient, or should different layers or heads
receive different cache capacity?

### Work

- Compare global fixed allocation with layer-adaptive allocation.
- Add head-level allocation only if the model architecture and runtime make it
  meaningful and measurable.
- Derive allocation from signals available at inference or from a development
  procedure, not manual inspection of final test outputs.
- Track metadata describing the allocation selected for every run.

### Experiments

- Equal budget across layers.
- Signal-derived layer budget.
- Equal budget across heads where supported.
- Signal-derived layer/head budget.
- Fixed recent window plus adaptive remainder.

### Deliverables

- Allocation policy and budget audit.
- Layer/head allocation visualizations.
- Complexity and overhead analysis.

### Exit criterion

Any adaptive allocation advantage survives matched total memory and includes
selection overhead. Otherwise, the simpler global policy remains the primary
candidate.

## Phase 8: Main quality-efficiency benchmark

### Question

Across context lengths and budgets, which policy gives the best measured
quality-memory-latency frontier?

### Work

- Freeze the evaluation protocol, model revision, seeds, prompts, and metrics.
- Run the primary comparison across supported context lengths and budgets.
- Use suitable long-context tasks such as needle retrieval, multi-needle
  retrieval, controlled QA, summarization, or reasoning.
- Report results per example as well as aggregate means.

### Experiments

- Full, recency, uniform, and selected adaptive policies.
- Context lengths from 2K upward within hardware limits.
- Compression budgets from 10% to 100%.
- Quality, cache memory, peak memory, latency, throughput, and overhead.

### Deliverables

- Frozen benchmark configuration.
- Main results table.
- Quality-memory-latency Pareto figure.
- Experiment log with hardware and runtime metadata.

### Exit criterion

The primary comparison is complete without post-hoc budget or metric changes.
The result is classified as support, no meaningful advantage, inconclusive, or
failure of H1.

## Phase 9: Ablations and failure analysis

### Question

Which component matters, and where does compression break?

### Work

- Run the signal ablations from Phase 6 under matched budgets.
- Compare global, layer, and head allocation.
- Analyze long-range dependencies, beginning-of-context facts, repeated facts,
  rare facts, distractors, and multi-hop reasoning.
- Search for a compression-quality phase transition without assuming one
  exists.
- Use paired examples to inspect what was discarded and what was needed later.

### Experiments

- Multi-needle and distractor-heavy contexts.
- Repeated versus rare information.
- Early-context versus late-context retrieval.
- Multi-hop chains requiring several retained facts.
- Aggressive budgets near 10% and intermediate budgets.

### Deliverables

- Ablation table with uncertainty estimates.
- Failure-case catalogue with raw examples.
- Error taxonomy and mechanism hypotheses.
- Explicit negative-result section.

### Exit criterion

The project can state not only whether a policy works, but under what
conditions it fails and which signal caused the observed behavior.

## Phase 10: Generalization, statistics, and release

### Question

Does the conclusion survive changes in model, task, context, and budget, and
can another researcher reproduce it?

### Work

- Test a second small model if computationally feasible.
- Add at least one substantially different task setting.
- Report mean, median where useful, standard deviation, bootstrap confidence
  intervals, and effect sizes for primary paired comparisons.
- Apply paired tests proportionately and correct for multiple comparisons when
  needed.
- Regenerate all figures from raw results.
- Write limitations, literature positioning, and the paper without overstating
  novelty or generality.

### Experiments

- Cross-model replication.
- Cross-task replication.
- Cross-context-length replication.
- Cross-budget replication.
- Sensitivity to seed and measurement noise.

### Deliverables

- Reproducibility checklist.
- Final results and figures.
- Research log and configuration archive.
- Manuscript and references.
- Clean README and release package.
- Optional Zenodo or other legitimate archive after the work is complete.

### Exit criterion

A clean environment can run the documented smoke tests and regenerate reported
figures from committed configurations and raw outputs. Claims in the manuscript
match the evidence, limitations are explicit, and negative results are retained.

## Cross-phase experiment record

Every experiment should record at least:

```text
Date
Question
Hypothesis
Configuration
Model and revision
Dataset or prompt source
Policy
Seed
Hardware and software versions
Expected outcome
Actual outcome
Interpretation
Next experiment
```

## Decision rules

- Prefer a stronger ablation over adding a larger model when the core mechanism
  is not yet understood.
- Prefer a simple baseline over a complex adaptive policy when their frontiers
  are indistinguishable.
- Treat latency and selection overhead as part of the method, not as an
  implementation detail to omit.
- Do not tune on the final test set.
- Do not convert a heuristic into a claim of learned memory.
- If H1 fails, investigate why and report the failure as the result.
