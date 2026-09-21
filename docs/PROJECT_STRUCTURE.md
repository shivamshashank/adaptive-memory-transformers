# Project structure

The repository intentionally contains only the evidence-producing baseline,
fixed-policy utilities, reporting code, tests, configuration, and research
documents that are valid at the current stage. Superseded proxy experiments are
not retained as active source code.

```text
adaptive-memory-transformers/
├── .github/workflows/ci.yml
├── configs/models/qwen2.5-1.5b-instruct.toml
├── configs/smoke/v1.toml
├── docs/
│   ├── ADAPTIVE_SELECTION.md
│   ├── NUMERICAL_TOLERANCES.md
│   ├── EVALUATION_PILOT.md
│   ├── POSITION_CORRECT_CACHE.md
│   ├── PHD_CV_10_DAY_CHECKLIST.md
│   ├── PROJECT_STRUCTURE.md
│   └── VALIDATION_REPORTS.md
├── research/protocol_v1.md
├── research/deviations.md
├── src/amt/
│   ├── __init__.py
│   ├── __main__.py
│   ├── baseline.py
│   ├── cache.py
│   ├── cli.py
│   ├── compressed.py
│   ├── decoding.py
│   ├── evaluation.py
│   ├── policies.py
│   ├── reporting.py
│   └── validation.py
├── tests/
│   ├── test_baseline.py
│   ├── test_cache.py
│   ├── test_cli.py
│   ├── test_decoding.py
│   ├── test_evaluation.py
│   ├── test_policies.py
│   ├── test_reporting.py
│   └── test_validation.py
├── 10_phase_phd_roadmap.md
├── pyproject.toml
├── uv.lock
└── README.md
```

## Source package

`src/amt/baseline.py` contains the full-cache reference runner and its cache,
latency, environment, model-revision, and hardware measurements. It does not
implement compressed decoding.

`src/amt/decoding.py` contains the Day 2 correctness oracle: neutral standard
Transformers generation, explicit token-by-token full-cache decoding, numerical
tolerance declarations, and inspectable cache snapshots. It deliberately does
not prune the cache.

`src/amt/policies.py` contains full-cache, recency, uniform, attention-only and
attention-plus-recency selection, aligned signal data, tie-aware rank
normalization, budget conversion and policy construction.

`src/amt/cache.py` owns Qwen2 cache state, retains original positions through
pruning, and supplies explicit positions and causal masks for new queries.
`src/amt/compressed.py` connects that adapter to fixed policies for greedy
generation with per-step retained-position and KV-byte traces. See
`docs/POSITION_CORRECT_CACHE.md` for a code walkthrough and supported scope.
Part 4 adds optional eager-attention collection and connects its scores to
selection; see `docs/ADAPTIVE_SELECTION.md` for the formula and tests.

`src/amt/reporting.py` aggregates baseline JSON records and generates a summary
table and plots from the saved measurements.

`src/amt/cli.py` is the baseline/reporting CLI entry point. It loads the versioned
TOML smoke configuration, runs baseline grids, writes JSON records, and invokes
the reporting pipeline.

`python -m amt.validation` is the dedicated offline validation/report script.
It runs code checks and pytest, captures logs and JUnit results, snapshots source,
and saves a five-policy model demonstration with exact inputs, selection signals,
outputs and full logits. Add `--qwen` to use the cached primary model on CPU.
See `docs/VALIDATION_REPORTS.md`.

## Supported commands

Create the locked environment and inspect the CLI:

```bash
python3 -m pip install uv==0.11.26
uv sync --frozen --group dev
uv run amt --help
```

Run the committed smoke configuration:

```bash
uv run amt baseline --config configs/smoke/v1.toml
```

Summarize saved baseline records:

```bash
uv run amt summarize --input-dir results/smoke-v1
```

Run the quality gate:

```bash
uv run python -m compileall -q src
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy
uv run pytest
uv run pre-commit run --all-files
git diff --check
```

Run the optional pinned-model decoding equivalence check:

```bash
AMT_RUN_MODEL_INTEGRATION=1 uv run pytest tests/test_decoding.py
```

## Deliberately absent code

The former `src/experiments/` modules were prototype scripts built around
retained-attention proxy scores and direct `DynamicCache` pruning. The research
roadmap identifies those approaches as insufficient for evidence-grade claims,
so the modules and their self-referential tests were removed instead of being
presented as completed experiments.

Position-correct Qwen2 cache adaptation and minimal causal attention/recency
scoring are implemented. Historical attention, layer allocation, benchmark
adapters and matched-budget evaluation remain future work. The repository makes
no compressed-cache performance claim.
