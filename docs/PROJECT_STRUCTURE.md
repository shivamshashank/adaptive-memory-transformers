# Project Structure and Usage

This document describes the current repository layout, the purpose of each tracked project file, and the inputs and outputs of the executable components.

For the research motivation, policy definitions, KV-cache formulas, planned measurements, and phase exit criteria, see the main [README.md](../README.md). The longer research roadmap is in [phases.md](../phases.md).

## Directory Tree

```text
adaptive-memory-transformers/
├── .github/workflows/
│   └── ci.yml
├── configs/
│   └── smoke/
│       └── v1.toml
├── docs/
│   ├── IMPLEMENTATION_CHECKLIST.md
│   ├── PHD_CV_10_DAY_CHECKLIST.md
│   └── PROJECT_STRUCTURE.md
├── research/
│   └── protocol_v1.md
├── scripts/
│   ├── memory_scaling_summary.py
│   ├── run_baseline.py
│   └── run_passkey_sweep.py
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
│       ├── compressed_decoding_evaluation.py
│       ├── fixed_cache_baselines.py
│       ├── generalization_study.py
│       ├── layer_adaptive_allocation.py
│       ├── passkey_retrieval_evaluation.py
│       ├── quality_memory_benchmark.py
│       └── signal_ablation_study.py
├── tests/
│   ├── test_adaptive_policy.py
│   ├── test_attention_cache_evaluation.py
│   ├── test_cache_policies.py
│   ├── test_compressed_decoding_evaluation.py
│   ├── test_generalization_study.py
│   ├── test_layer_adaptive_allocation.py
│   ├── test_passkey_retrieval_evaluation.py
│   ├── test_quality_memory_benchmark.py
│   └── test_signal_ablation_study.py
├── results/                  Generated JSON, Markdown, and plot artifacts
├── README.md                 Project overview and research design
├── phases.md                 Research roadmap and exit criteria
├── pyproject.toml            Project metadata and quality-tool configuration
├── uv.lock                   Reproducible dependency resolution
├── requirements.txt          Runtime dependency mirror
├── CONTRIBUTING.md           Contribution workflow
├── CODE_OF_CONDUCT.md        Community conduct rules
├── SECURITY.md               Security reporting policy
├── LICENSE                   Project license
└── .gitignore                Ignored local and generated files
```

`.venv/`, `.git/`, `__pycache__/`, and `.pytest_cache/` are local or generated directories and are not part of the source layout.

## Top-Level Documentation and Configuration

### `README.md`

The main project guide. It explains the research question, adaptive KV-cache motivation, policy families, planned budgets and context lengths, metrics, architecture, and intended repository structure.

Input: None.

Output: Documentation for users and contributors. See [README.md](../README.md).

### `phases.md`

The research roadmap. It defines the questions, experiments, deliverables, and exit criteria for each phase, including the Phase 9 ablations and Phase 10 generalization requirements.

Input: None.

Output: Research planning and evaluation criteria.

### `pyproject.toml`, `uv.lock`, and `requirements.txt`

`pyproject.toml` defines project metadata, runtime and development dependency
groups, and pytest, Ruff, and MyPy configuration. `uv.lock` pins the complete
cross-platform dependency graph. `requirements.txt` mirrors the direct runtime
dependencies for compatibility with environments that do not use `uv`.

Input: Python environment creation and quality-tool execution.

Output: A reproducible dependency environment.

Example:

```bash
python3 -m pip install uv==0.11.26
uv sync --frozen --group dev
```

### `configs/smoke/v1.toml`

Immutable CPU smoke configuration with pinned model and tokenizer revisions,
generation settings, seeds, and output metadata requirements. It validates the
software pipeline and is not a research benchmark.

### `research/protocol_v1.md`

Frozen ten-day validation protocol defining H1/H0, claim boundaries, primary
comparison, budgets, metrics, split discipline, failure rules, and sprint exit
criteria.

### `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, and `LICENSE`

Repository governance, contribution, security-reporting, and licensing documents. They do not participate in model execution.

## Reusable Source Modules

All reusable implementation modules live directly under `src/`. They expose Python APIs and do not require phase-specific naming.

### `src/baseline.py`

Runs the full-cache causal language-model baseline.

Inputs:

- Model name
- Prompt text
- Maximum generated tokens
- Device and dtype settings
- Optional prompt target length and seed, depending on the CLI arguments

Outputs:

- Generated text
- Prompt and generated token counts
- Prefill and decode timings
- Decode throughput
- Analytical and actual KV-cache sizes
- Cache tensor shapes
- Model, dtype, device, hardware, and software metadata

The result is written as JSON when invoked through the baseline runner.

### `src/cache_policies.py`

Defines the fixed cache policies and shared budget utilities:

- `FullCachePolicy`
- `RecencyCachePolicy`
- `UniformCachePolicy`
- Budget conversion and matched-policy selection helpers

Inputs: Total token count, budget token count or budget ratio, and optionally a recent window.

Outputs: Retained token indices and `CacheSelection` metadata.

### `src/adaptive_policy.py`

Defines `AdaptiveImportancePolicy`, which ranks token positions using normalized attention, recency, and frequency signals.

Inputs:

- Total token count
- Retained-token budget
- Optional attention, recency, and frequency score vectors

Outputs: Sorted retained token indices.

### `src/importance.py`

Loads a causal model with eager attention and computes a model-derived token importance signal.

Inputs:

- Model identifier
- Prompt text

Outputs:

- Sequence length
- Layer and head counts
- Total attention mass
- Per-position attention mass
- Per-position recency signal

The CLI writes `results/importance/importance_analysis.json` by default.

### `src/layer_adaptive_policy.py`

Allocates one total token budget across layers in proportion to non-negative layer scores.

Inputs: Layer score vector and total token budget.

Outputs: One integer budget per layer whose sum equals the requested total budget.

## Experiment Entry Points

All experiment entry points are in `src/experiments/`. They can be run from the repository root with the project virtual environment.

### `fixed_cache_baselines.py`

Checks full, recency, and uniform policy selections across fixed budgets.

Input: Total token count, supplied through the CLI.

Output: JSON policy-budget records, including selected indices and retained counts.

### `attention_cache_evaluation.py`

Compares fixed policies with the adaptive policy using a model-derived attention signal and provides cache-pruning helpers.

Inputs:

- Model identifier
- Prompt text
- Optional total token count

Output: JSON records containing policy, budget, and retained indices. The default output is `results/attention_cache_evaluation.json`.

### `cache_policy_comparison.py`

Runs a synthetic matched-budget comparison of fixed and adaptive cache policies and summarizes results by budget ratio.

Inputs: Optional output paths and total-token settings supplied by the CLI.

Outputs:

- Raw policy comparison JSON
- Summary JSON grouped by budget ratio

### `compressed_decoding_evaluation.py`

Runs actual autoregressive decoding with full, recency, uniform, and adaptive
cache policies. Compressed policies are pruned after prefill and each decode
step. It records cache tensor bytes, selection and pruning overhead, decode
throughput, greedy exact-answer match, and teacher-forced loss/perplexity when
a known continuation is supplied.

Inputs: Model identifier, prompt, optional expected answer and continuation,
and generated-token count.

Output: JSON policy records. The default output is
`results/compressed_decoding_evaluation.json`.

Important: arbitrary cache pruning reindexes retained positions, so models
requiring absolute cache positions need a model-specific adapter before results
are generalized.

### `passkey_retrieval_evaluation.py`

Runs synthetic Needle-in-a-Haystack (passkey retrieval) evaluation across full,
recency, uniform, and adaptive cache policies at matched budgets. Generates
controlled prompts with needles inserted at configurable fractional context
depths, tracking needle token retention rate, passkey extraction, teacher-forced
cross-entropy loss/perplexity, actual cache memory footprint, and selection/pruning
runtime overhead.

Inputs: Model identifier, secret passkey string, context length, fractional needle depth,
and generated token count.

Output: JSON specification and measurement records. Default output is
`results/passkey_retrieval.json`.

### `layer_adaptive_allocation.py`

Compares global adaptive selection with layer-aware budget allocation using real model attention outputs.

Inputs:

- Model identifier
- Prompt text

Outputs: JSON containing global retained indices, layer scores, layer allocations, and matched budget metadata. The default output is `results/layer_adaptive_allocation.json`.

### `quality_memory_benchmark.py`

Runs the main fixed-versus-adaptive quality-memory frontier using retained attention mass as a quality proxy.

Inputs:

- Model identifier
- Prompt text
- Optional total token count

Outputs: JSON records for full, recency, uniform, and adaptive policies at 100%, 75%, 50%, 25%, and 10% budgets. The default output is `results/quality_memory_benchmark.json`.

Important: its quality score is an attention-coverage proxy, not task accuracy.

### `signal_ablation_study.py`

Runs Phase 9 signal ablations across attention-only, attention-plus-recency, attention-plus-frequency, and all-signal variants.

Inputs: Model identifier and output path.

Outputs:

- Raw records across prompt categories and budgets
- Per-case summaries
- Pooled means, standard deviations, and bootstrap intervals
- Raw failure cases
- An explicit negative-result statement

The default output is `results/signal_ablation_study.json`.

### `generalization_study.py`

Runs cross-model and cross-task generalization checks.

Inputs:

- One or more model identifiers
- Random seed
- Output path

Outputs:

- Raw cross-model and cross-task records
- Per-task summaries
- Pooled summaries with mean, median, standard deviation, and paired effect size
- Software and hardware metadata
- Reproducibility checklist
- Preserved model/task errors

The default output is `results/generalization_study.json`.

## Utility Scripts

### `scripts/run_baseline.py`

A thin direct-execution wrapper around `src.baseline.main`. It adds the repository root to `sys.path`, allowing this command to work from the repository root:

```bash
./.venv/bin/python scripts/run_baseline.py --help
```

Input: The baseline CLI arguments.

Output: The baseline JSON result and generated text, according to the options passed to `src.baseline`.

### `scripts/memory_scaling_summary.py`

Aggregates per-run baseline JSON files, computes mean/std/min/max statistics, writes a Markdown table, and generates KV-memory, prefill-time, and decode-throughput plots.

Inputs:

- Directory of baseline JSON files, default `results/memory_scaling`
- Optional output directory

Outputs:

- `summary.json`
- `summary.md`
- `kv_bytes_vs_sequence.png`
- `prefill_time_vs_sequence.png`
- `decode_throughput_vs_sequence.png`

## Tests

Tests are grouped by responsibility rather than research phase.

### `test_adaptive_policy.py`

Checks signal combination, top-budget token selection, and full-budget behavior for the adaptive selector.

### `test_attention_cache_evaluation.py`

Checks that key/value cache pruning retains the requested positions and tensor shapes.

### `test_cache_policies.py`

Checks full, recency, uniform, and equal-budget fixed-policy behavior.

### `test_compressed_decoding_evaluation.py`

Checks DynamicCache-compatible pruning, cache-byte accounting, and answer
normalization without downloading a model.

### `test_passkey_retrieval_evaluation.py`

Checks passkey prompt synthesis, needle token index boundary calculations,
recency eviction behavior on early needles, and adaptive selection preservation
of high-scoring needle tokens.

### `test_generalization_study.py`

Checks paired effect-size calculations and cross-model/task summary aggregation.

### `test_layer_adaptive_allocation.py`

Checks layer-score ordering and exact total layer-budget allocation.

### `test_quality_memory_benchmark.py`

Checks attention-coverage scoring and benchmark frontier ordering.

### `test_signal_ablation_study.py`

Checks bootstrap intervals and ablation summary failure classification.

Run all tests with:

```bash
./.venv/bin/python -m pytest -q
```

## Results and Reproducibility

`results/` contains generated experiment artifacts and is not a source-code package. Raw JSON should be preserved alongside summaries and plots so figures and reported values can be regenerated.

A typical workflow is:

```bash
./.venv/bin/python scripts/run_baseline.py --help
./.venv/bin/python src/experiments/quality_memory_benchmark.py --help
./.venv/bin/python src/experiments/signal_ablation_study.py --help
./.venv/bin/python src/experiments/generalization_study.py --help
./.venv/bin/python -m pytest -q
```

For the research rationale, metric definitions, policy comparison rules, and limitations, refer back to [README.md](../README.md). For the full sequence of experiments and exit criteria, see [phases.md](../phases.md).
