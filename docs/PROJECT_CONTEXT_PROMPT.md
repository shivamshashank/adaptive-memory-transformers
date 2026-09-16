# Adaptive Memory Transformers: Complete Project Context Prompt

Copy the prompt below when handing this repository to another coding or research assistant.

---

## Prompt

You are working on the repository `adaptive-memory-transformers`, an independent research project by Shivam Shashank. Read the complete repository before making changes. Do not assume that a phase label is the correct name for a file; the code has been organized by responsibility under `src/experiments/`.

### Researcher profile and research fit

The researcher is an AI/ML researcher focused on trustworthy and efficient generative AI, evidence-grounded reasoning, LLM evaluation, and efficient Transformer inference. Relevant background includes:

- MSc Computer Science at the University of Birmingham, Distinction, 81%.
- Dissertation: CloudGraph, a GraphRAG-powered multi-agent root-cause-analysis system.
- CloudGraph work on graph confidence propagation, graph-provenance claim scoring, temporal operational knowledge graphs, multi-hop retrieval, evaluation leakage, pseudo-replication, paired bootstrap confidence intervals, Wilcoxon tests, and regression testing.
- Independent research on adaptive KV-cache compression for long-context Transformers.
- More than two years of production distributed-systems experience at Hewlett Packard Enterprise using Python, Go, Kubernetes, AWS, GCP, Terraform, CI/CD, observability, and incident analysis.
- Technical strengths relevant here: PyTorch, Hugging Face Transformers, LLM evaluation, retrieval-augmented generation, statistical analysis, reproducibility, and production-quality systems.

Keep the project aligned with this profile: make the work rigorous, evidence-grounded, reproducible, statistically honest, and relevant to trustworthy and efficient long-context AI. Do not overclaim novelty or generality.

### Research question

Can adaptive allocation of KV-cache capacity provide a better quality-efficiency trade-off than full, recency, uniform, or other fixed compression policies under the same memory budget?

The main hypothesis is:

> At equivalent cache budgets, adaptive retention may improve the quality-memory-latency frontier over fixed or purely recency-based compression.

The null hypothesis is:

> Adaptive importance-based compression provides no statistically meaningful advantage over simpler policies under equivalent budgets.

The project must preserve negative results. A higher attention-coverage proxy is not the same as higher task accuracy.

### Read these documents first

1. [README.md](../README.md): research question, architecture, metrics, policies, design constraints, and reproducibility contract.
2. [docs/PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md): current directory tree, file responsibilities, inputs, outputs, and commands.
3. [phases.md](../phases.md): research questions, experiments, deliverables, exit criteria, and decision rules.
4. `requirements.txt`: pinned runtime dependencies.
5. The complete `tests/` directory before modifying behavior.

Read every line of the relevant source and test files, not only the first matching function. Preserve user changes in a dirty worktree.

### Current repository structure

```text
adaptive-memory-transformers/
├── README.md
├── phases.md
├── requirements.txt
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── LICENSE
├── .gitignore
├── docs/
│   ├── PROJECT_CONTEXT_PROMPT.md
│   └── PROJECT_STRUCTURE.md
├── src/
│   ├── __init__.py
│   ├── adaptive_policy.py
│   ├── baseline.py
│   ├── cache_policies.py
│   ├── importance.py
│   ├── layer_adaptive_policy.py
│   └── experiments/
│       ├── __init__.py
│       ├── attention_cache_evaluation.py
│       ├── cache_policy_comparison.py
│       ├── fixed_cache_baselines.py
│       ├── generalization_study.py
│       ├── layer_adaptive_allocation.py
│       ├── quality_memory_benchmark.py
│       └── signal_ablation_study.py
├── scripts/
│   ├── memory_scaling_summary.py
│   └── run_baseline.py
├── tests/
│   ├── test_adaptive_policy.py
│   ├── test_attention_cache_evaluation.py
│   ├── test_cache_policies.py
│   ├── test_generalization_study.py
│   ├── test_layer_adaptive_allocation.py
│   ├── test_quality_memory_benchmark.py
│   └── test_signal_ablation_study.py
└── results/
    └── generated experiment JSON, Markdown, and plots
```

Generated directories such as `.venv/`, `__pycache__/`, `.pytest_cache/`, and `.git/` are not source files.

### File-by-file responsibilities

#### Reusable source modules

- `src/__init__.py`: package marker and package-level description.
- `src/baseline.py`: full-cache causal-LM inference; accepts model, prompt, seed, dtype, device, prompt length, and generation settings; outputs generated text, token counts, prefill/decode timing, throughput, analytical and actual KV-cache bytes, cache shapes, hardware metadata, software versions, and model configuration.
- `src/adaptive_policy.py`: combines normalized attention, recency, and frequency scores; accepts token count, budget, and score vectors; outputs retained token indices.
- `src/cache_policies.py`: implements full, recency, and uniform policies; accepts sequence length and budget; outputs selected indices and `CacheSelection` records.
- `src/importance.py`: loads a model with eager attention and computes per-position attention mass and recency signals; accepts model and prompt; outputs structured importance metadata and optionally `results/importance/importance_analysis.json`.
- `src/layer_adaptive_policy.py`: allocates a total integer token budget across layers from layer scores; accepts layer scores and total budget; outputs one budget per layer with an exact total.

#### Experiment entry points

- `src/experiments/fixed_cache_baselines.py`: checks fixed policy behavior at configured budgets; outputs policy-budget records.
- `src/experiments/attention_cache_evaluation.py`: compares fixed policies with adaptive selection from real attention signals and contains cache-pruning helpers; outputs matched-budget selection JSON.
- `src/experiments/cache_policy_comparison.py`: compares fixed and adaptive policies at synthetic matched budgets; outputs raw and ratio-level summaries.
- `src/experiments/layer_adaptive_allocation.py`: compares global adaptive selection with layer-aware allocation; accepts model and prompt; outputs layer scores, allocations, retained indices, and budget metadata.
- `src/experiments/quality_memory_benchmark.py`: runs the main fixed-versus-adaptive frontier at 100%, 75%, 50%, 25%, and 10%; outputs policy records and retained-attention quality proxy scores.
- `src/experiments/signal_ablation_study.py`: runs attention-only, attention-plus-recency, attention-plus-frequency, and all-signal ablations across early, late, repeated, rare, distractor, and multi-hop prompt categories; outputs raw records, pooled statistics, bootstrap intervals, and failure cases.
- `src/experiments/generalization_study.py`: runs cross-model and cross-task checks; accepts model list and seed; outputs raw records, pooled summaries, paired effect sizes, software/hardware metadata, reproducibility checklist, and preserved errors.

#### Utility scripts

- `scripts/run_baseline.py`: direct CLI wrapper around `src.baseline.main`; accepts baseline CLI options and writes baseline results.
- `scripts/memory_scaling_summary.py`: reads per-run baseline JSON files; outputs `summary.json`, `summary.md`, and memory, prefill, and throughput plots.

#### Tests

- `tests/test_adaptive_policy.py`: adaptive signal combination and budget behavior.
- `tests/test_attention_cache_evaluation.py`: key/value cache pruning positions and shapes.
- `tests/test_cache_policies.py`: full, recency, uniform, and matched-budget fixed policies.
- `tests/test_generalization_study.py`: paired effect sizes and grouped summaries.
- `tests/test_layer_adaptive_allocation.py`: layer ordering and exact allocation totals.
- `tests/test_quality_memory_benchmark.py`: attention-coverage proxy and frontier ordering.
- `tests/test_signal_ablation_study.py`: bootstrap intervals and failure classification.

### Completed work and evidence

The repository has completed the following implementation sequence:

1. Full-cache baseline with static typing, direct script execution, model metadata, cache metadata, prefill/decode timing, dtype handling, and hardware information.
2. Memory-scaling aggregation with JSON, Markdown, and plots.
3. Precision support and validation.
4. Attention-based token importance with eager attention to avoid empty attention outputs.
5. Fixed full, recency, and uniform cache policies at matched budgets.
6. Adaptive token selection using attention, recency, and frequency signals.
7. Cache-pruning helper and real-signal comparison.
8. Layer-aware budget allocation under a matched total budget.
9. Quality-memory frontier benchmark using retained attention mass.
10. Signal ablations and failure analysis with bootstrap intervals.
11. Cross-model and cross-task generalization smoke study using `hf-internal-testing/tiny-random-gpt2` and `sshleifer/tiny-gpt2`.
12. Responsibility-based file organization under `src/experiments/`, documentation in `docs/`, concise comments in Python files, and README references.

Verified repository state:

- Full relevant test suite passes: 18 tests.
- Python compilation passes for `src`, `scripts`, and `tests`.
- `git diff --check` passes.
- Phase 9 generated 480 records and no adaptive-loss cases under the attention-coverage proxy.
- Phase 10 generated 160 records across two models and four task-style prompt categories with zero model/task errors.

### What the evidence currently supports

The experiments support the narrow statement:

> Adaptive selection retains more model-derived attention mass than recency and uniform selection under matched token budgets on the tested small-model prompts.

The experiments do not yet support claims of:

- improved next-token loss or perplexity;
- improved generation quality;
- improved controlled retrieval or multi-hop QA accuracy;
- lower end-to-end latency after including selection and pruning overhead;
- long-context generalization at 2K, 4K, 8K, 16K, or 32K tokens;
- broad model or task generality;
- learned memory or a generally useful compression method.

The quality score is an attention-coverage proxy, not task accuracy. The tested models and prompts are small and short. The very large paired effect sizes in Phase 10 are partly caused by low within-task variation and should not be treated as definitive statistical proof.

### Highest-priority next steps

Implement these in order:

1. Build a real compressed-decoding evaluator that runs full, recency, uniform, and adaptive policies under identical prompts, seeds, model revisions, generation settings, and total budgets.
2. Compute task-level metrics: next-token loss/perplexity and controlled retrieval or QA accuracy. Use paired examples and preserve raw per-example outputs.
3. Measure actual analytical cache bytes, actual tensor bytes, peak memory, prefill latency, decode latency, selection latency, pruning latency, and throughput for every policy.
4. Add a real context-length sweep constrained by hardware, starting with feasible short, medium, and longer contexts before claiming long-context generalization.
5. Add multiple seeds and enough examples per task for meaningful confidence intervals and paired tests; do not report a large effect size from one score per task as conclusive.
6. Compare global adaptive allocation with layer-aware allocation under the same true decode-time protocol. Keep the simpler policy primary if layer allocation does not improve quality after overhead.
7. Generate figures directly from raw JSON and commit the configuration used for each reported result.
8. Update the final research log and README limitations. Retain negative results.

### Engineering constraints

- Use Python 3.14 in `.venv` and the pinned dependencies in `requirements.txt`.
- Keep every Python file statically typed; do not introduce `Any`.
- Prefer `Protocol`, `TypedDict`, dataclasses, and precise unions for dynamic external APIs.
- Use existing policy and measurement abstractions before adding new ones.
- Keep experiment entry points under `src/experiments/` and utility CLIs under `scripts/`.
- Use descriptive responsibility-based names, not phase numbers.
- Add only short comments where they clarify a method or non-obvious block.
- Preserve public APIs unless the requested change requires a migration.
- Do not use synthetic proxy results as evidence for task-level claims.
- Do not modify unrelated user changes or commit changes unless explicitly asked.
- After every edit, run the narrowest relevant test first, then the full suite when practical.

### Standard validation commands

```bash
./.venv/bin/python -m pytest -q
./.venv/bin/python -m compileall -q src scripts tests
./.venv/bin/python scripts/run_baseline.py --help
./.venv/bin/python src/experiments/quality_memory_benchmark.py --help
./.venv/bin/python src/experiments/generalization_study.py --help
git diff --check
```

### Expected assistant behavior

Before editing:

1. Read the relevant file completely.
2. State one local hypothesis about the behavior or missing requirement.
3. Identify one cheap test that could disconfirm it.
4. Make the smallest focused edit.

After editing:

1. Run a focused executable validation immediately.
2. Repair local failures before broadening scope.
3. Run the full relevant test suite and compilation checks.
4. Report exactly what changed, what was verified, what remains uncertain, and which evidence supports the conclusion.

When proposing research conclusions, distinguish clearly between measured evidence, proxy evidence, assumptions, and uncompleted work.

---

## End of Prompt

This prompt is a maintained snapshot of the repository context. Update it when the directory structure, evaluation protocol, research evidence, or next-step priority changes.
