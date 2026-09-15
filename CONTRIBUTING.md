# Contributing to Adaptive KV-Cache Compression

Thank you for contributing to **Adaptive KV-Cache Compression**, an independent
research project studying adaptive KV-cache allocation for long-context
Transformers. Contributions are welcome in code, experiments, documentation,
literature review, evaluation, and reproducibility improvements.

By participating in this project, you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## How Can I Contribute?

- **Report bugs:** Include the command, configuration, environment, and a
  minimal reproducible example.
- **Propose experiments:** Describe the question, hypothesis, baseline, budget,
  metrics, and expected falsification condition.
- **Improve implementation:** Add focused tests for cache policies, budgeting,
  configuration, or result processing.
- **Improve documentation:** Clarify methods, assumptions, limitations, or
  reproduction steps.
- **Review research claims:** Check that claims are supported by recorded
  experiments and that negative results are not omitted.

## Development Setup

The planned implementation uses **Python**, **PyTorch**, and Hugging Face
**Transformers**. The repository is currently documentation-only, so these
commands describe the intended setup once implementation is added.

Prerequisites:

- Python 3.11 or newer
- A virtual environment or Conda environment
- PyTorch and Transformers
- An NVIDIA CUDA GPU for performance experiments, where available
- CPU support for unit tests and small smoke tests

When `requirements.txt` is available, the intended setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Working on Code and Experiments

Keep changes narrow and preserve the research boundary:

- `src/cache/`: Cache interfaces and retention policies.
- `src/compression/`: Compression implementations.
- `src/importance/`: Attention, recency, and frequency signals.
- `src/evaluation/`: Measurement and benchmark harnesses.
- `configs/`: Versioned experiment configurations.
- `experiments/`: Experiment-specific runners and analysis.
- `results/`: Raw generated outputs; do not hand-edit them.
- `figures/`: Reproducible generated figures.

For experiment changes, record the model revision, dataset or prompt source,
seed, dtype, hardware, cache budget, metrics, and result location.

## Formatting, Linting, and Validation

Run the checks supported by the implementation:

```bash
python3 -m compileall -q src scripts
PYTHONPATH=. python3 scripts/run_baseline.py --help
git diff --check
```

When formatting and linting tools are introduced, document their versions and
commands here and pin them where practical.

## Testing

Code changes should include focused unit or integration tests. Research changes
should also include the configuration and raw output needed to inspect them.

Planned commands:

```bash
python3 -m pytest -q
python3 -m pytest --cov=src -q
```

Planned test coverage includes KV-cache byte calculations across MHA, GQA, and
MQA; cache index selection; exact budget accounting; importance normalization;
configuration validation; deterministic baseline comparison; compressed-cache
shape validity; and reproducible figure generation.

GPU benchmarks are not unit-test assertions. Record hardware, software
versions, seeds, dtype, model revision, and wall-clock conditions with every
benchmark result.

## Research Integrity

- Tune method weights and thresholds on development data only.
- Freeze the primary evaluation protocol before the final comparison.
- Compare all policies at equivalent cache budgets.
- Preserve failed runs and negative findings.
- Report selection overhead rather than hiding it in implementation details.
- Do not claim novelty, generality, or speedups without evidence.
- Update [phases.md](phases.md) when the research plan materially changes.

## Commit Guidelines

Use concise Conventional Commit-style messages where practical:

- `feat: add recency cache policy`
- `fix: correct cache budget accounting`
- `docs: clarify evaluation protocol`
- `test: add GQA cache-size test`
- `research: add importance ablation`

## Submitting a Pull Request

```bash
git checkout -b research/short-description
git diff --check
git commit -m "research: describe experiment"
git push origin research/short-description
```

Pull requests should describe the research or engineering question addressed,
files changed, validation performed, and remaining limitations. They should not
include credentials, private datasets, model access tokens, or generated
artifacts that can be regenerated from committed code and configuration.
