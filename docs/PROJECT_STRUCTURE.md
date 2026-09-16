# Project structure

The repository intentionally contains only the evidence-producing baseline,
fixed-policy utilities, reporting code, tests, configuration, and research
documents that are valid at the current stage. Superseded proxy experiments are
not retained as active source code.

```text
adaptive-memory-transformers/
├── .github/workflows/ci.yml
├── configs/smoke/v1.toml
├── docs/
│   ├── PHD_CV_10_DAY_CHECKLIST.md
│   └── PROJECT_STRUCTURE.md
├── research/protocol_v1.md
├── src/amt/
│   ├── __init__.py
│   ├── __main__.py
│   ├── baseline.py
│   ├── cli.py
│   ├── policies.py
│   └── reporting.py
├── tests/
│   ├── test_baseline.py
│   ├── test_cli.py
│   ├── test_policies.py
│   └── test_reporting.py
├── 10_phase_phd_roadmap.md
├── pyproject.toml
├── uv.lock
└── README.md
```

## Source package

`src/amt/baseline.py` contains the full-cache reference runner and its cache,
latency, environment, model-revision, and hardware measurements. It does not
implement compressed decoding.

`src/amt/policies.py` contains deterministic full-cache, recency, and uniform
index-selection policies plus budget conversion and policy construction. These
utilities are kept because they have clear semantics and focused tests.

`src/amt/reporting.py` aggregates baseline JSON records and generates a summary
table and plots from the saved measurements.

`src/amt/cli.py` is the single supported entry point. It loads the versioned
TOML smoke configuration, runs baseline grids, writes JSON records, and invokes
the reporting pipeline.

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

## Deliberately absent code

The former `src/experiments/` modules were prototype scripts built around
retained-attention proxy scores and direct `DynamicCache` pruning. The research
roadmap identifies those approaches as insufficient for evidence-grade claims,
so the modules and their self-referential tests were removed instead of being
presented as completed experiments.

Position-correct cache adaptation, causal adaptive scoring, benchmark adapters,
and matched-budget evaluation will return as tested package modules during the
corresponding roadmap days. Until then, the repository makes no compressed-cache
performance claim.
