<div align="center">

# Adaptive KV-Cache Compression

### Adaptive memory allocation for efficient long-context Transformer inference

This project investigates whether a Transformer can dynamically decide which
past representations to retain, compress, or discard during autoregressive
inference while preserving language-model and long-context performance.

The research question is not *"can KV-cache compression save memory?"* It is:

> **Can adaptive allocation of KV-cache capacity provide a better
> quality-efficiency trade-off than fixed or recency-based compression under
> the same memory budget?**

<br />

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![License](https://img.shields.io/github/license/shivamshashank/adaptive-memory-transformers?style=flat-square)

**Status:** CPU cache correctness and direct-color task calibrated · query-aware
selection remained stronger in a larger two-seed development run, but the
predeclared full-context calibration gate failed · one fresh exact-lookup prompt
repair also failed · confirmation seed remains untouched · no superiority or
speedup claims

</div>

---

## 📌 Overview

Completed diagnosis: [modest longer-context matched-budget experiment](docs/LONG_CONTEXT_EXPERIMENT.md).
Current milestone: [two-pass query-aware selection](docs/QUERY_AWARE_EXPERIMENT.md).
Larger validation: [multi-seed query-aware development run](docs/QUERY_AWARE_LARGER.md).
Reliability diagnosis: [larger full-cache failures](docs/FULL_CACHE_LARGER_DIAGNOSIS.md).
Calibrated task: [direct-color task calibration](docs/DIRECT_COLOR_CALIBRATION.md).
Why it is required: [retrieval reliability findings](docs/RETRIEVAL_RELIABILITY.md).
Completed experiment: [protected-first-token ablation](docs/PROTECTED_PREFIX_ABLATION.md).
Previous evidence: [v2 findings](docs/PILOT_V2_FINDINGS.md) and
[diagnosis checklist](docs/V2_DIAGNOSIS_CHECKLIST.md).

During autoregressive decoding, a Transformer stores keys and values for past
tokens so that every new token can attend to previous context without
recomputing the entire sequence. This KV cache grows with sequence length and
can become a major memory and bandwidth bottleneck for long-context inference.

The project will compare four families of cache policy:

| Policy | Description |
|---|---|
| **Full cache** | Retain every key and value; the reference condition. |
| **Recency** | Retain the most recent tokens under a fixed budget. |
| **Uniform** | Retain tokens using a fixed, non-adaptive allocation. |
| **Adaptive** | Allocate capacity using measurable token, layer, and head signals. |

The study is explicitly allowed to produce a negative result. The primary
hypothesis is:

> **H1:** At equivalent cache budgets, adaptive retention improves the
> quality-memory-latency frontier over fixed or purely recency-based policies.

The null hypothesis is:

> **H0:** Adaptive importance-based compression provides no statistically
> meaningful advantage over simpler policies under equivalent budgets.

No method will be described as novel, state of the art, or generally useful
unless the literature review and experiments support that claim.

## 🎯 Research objectives

1. Derive and empirically verify how KV-cache memory scales with layers, KV
	 heads, sequence length, head dimension, and precision.
2. Determine whether cached tokens have unequal future utility across positions,
	 layers, and attention heads.
3. Compare full, recency, uniform, and importance-based policies at equal
	 memory budgets.
4. Test whether adaptive allocation improves quality without hiding latency or
	 selection overhead.
5. Use ablations to identify whether attention, recency, frequency, or their
	 combinations actually matter.
6. Identify failure cases involving long-range dependencies, rare facts,
	 repeated information, and multi-hop reasoning.
7. Test generalization across context lengths, tasks, compression ratios, and,
	 where feasible, a second small causal model.

## 🧪 Research design

Every claim is tied to an experiment, every experiment has a baseline, and all
comparisons use equivalent cache budgets. Development settings are separated
from final evaluation settings, random seeds are recorded, raw outputs are
preserved, and figures are generated from saved results rather than manually
transcribed numbers.

### Planned budget sweep

```text
100% · 75% · 50% · 25% · 10%
```

### Planned context sweep

```text
2K · 4K · 8K · 16K · 32K
```

The actual range will be constrained by the selected model, hardware, and
measurement validity. The first implementation uses one small openly available
causal model before adding scale.

### Primary measurements

| Category | Metrics |
|---|---|
| Memory | Analytical KV bytes, allocated GPU memory, peak GPU memory |
| Runtime | Prefill latency, decode latency, total latency, tokens/second |
| Language modelling | Next-token loss and perplexity |
| Long context | Retrieval accuracy, multi-needle accuracy, task score |
| Statistics | Mean, median, standard deviation, bootstrap confidence intervals, effect size |
| Reproducibility | Seed, model revision, dtype, hardware, software versions, config |

The main result should be a quality-memory-latency Pareto frontier, not one
unqualified scalar score.

## 🏗️ Architecture

```mermaid
flowchart TB
		CFG["YAML experiment config"] --> RUN["Experiment runner"]
		RUN --> TOK["Tokenizer"]
		TOK --> PRE["Prefill: prompt -> hidden states"]
		PRE --> CACHE[("KV cache")]
		CACHE --> POLICY["Cache policy"]
		POLICY --> DEC["Autoregressive decode"]
		DEC --> APPEND["Append new K/V states"]
		APPEND --> POLICY
		DEC --> METRICS["Metrics collector"]
		CACHE --> METRICS
		METRICS --> RAW["Raw JSON/CSV results"]
		RAW --> ANALYSIS["Analysis and figures"]

		subgraph POLICIES["Comparable policies"]
				FULL["Full cache"]
				REC["Recency"]
				UNI["Uniform"]
				ADAPT["Adaptive importance"]
		end
		POLICY -. selects .-> FULL
		POLICY -. selects .-> REC
		POLICY -. selects .-> UNI
		POLICY -. selects .-> ADAPT
```

The policy boundary is the central experimental boundary. Model loading,
tokenization, prompts, budgets, decoding settings, and metrics must remain
matched when policies are compared.

### KV-cache memory model

For a standard key-value cache, the analytical estimate is:

```text
KV bytes = 2 × layers × KV heads × sequence length × head dimension × bytes per element
```

The factor of two accounts for keys and values. For grouped-query or
multi-query attention, the number of KV heads can be smaller than the number
of query heads. The estimate will be checked against actual tensors and GPU
memory measurements rather than treated as a complete runtime model.

### Research data flow

```mermaid
flowchart LR
		PROMPT["Controlled prompt or benchmark example"] --> BASE["Full-cache baseline"]
		PROMPT --> COMP["Compressed inference"]
		BASE --> Q1["Quality"]
		BASE --> E1["Memory and latency"]
		COMP --> Q2["Quality"]
		COMP --> E2["Memory and latency"]
		Q1 & Q2 --> STAT["Paired comparison and confidence intervals"]
		E1 & E2 --> FRONT["Pareto frontier"]
		STAT & FRONT --> FIND["Supported, inconclusive, or negative finding"]
```

## 🧱 Technical stack

### Languages

- **Python 3.11+** for models, experiments, evaluation, analysis, and figures.
- **TOML/JSON** for experiment configuration and machine-readable outputs.
- **Markdown** for research notes, methods, and reproducibility documentation.

### Core ML and numerical tools

- **PyTorch** for tensor operations, inference, cache manipulation, and GPU
	measurement.
- **Hugging Face Transformers** for openly available causal language models,
	tokenization, and comparison with standard generation.
- **Hugging Face Datasets** where a dataset-backed evaluation is appropriate.
- **NumPy and Pandas** for result processing.
- **SciPy** for bootstrap intervals and proportionate statistical tests.
- **Matplotlib** for reproducible research figures.

### Experiment and engineering tools

- YAML configuration files for every run.
- Git for versioned code, configs, and research decisions.
- Local JSON/CSV results as the default tracker; Weights & Biases may be added
	if it reduces, rather than increases, reproducibility friction.
- Optional CUDA GPU execution with CPU support for smoke tests and analytical
	components.

This project intentionally does not introduce Kubernetes, Terraform, FastAPI,
LangChain, Neo4j, or Qdrant. Those technologies belong to other projects in
the portfolio and are not needed for this research question.

## 📂 Target repository structure

The current directory tree, file responsibilities, inputs, outputs, and
execution entry points are documented in
[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).

## ⚡ Implementation quick start

The initial full-cache inference harness is available. The locked development
environment is the canonical setup:

```bash
python3 -m pip install uv==0.11.26
uv sync --frozen --group dev
uv run amt baseline --config configs/smoke/v1.toml
```

`pyproject.toml` is the single dependency declaration and `uv.lock` pins the
complete environment. The immutable CPU smoke configuration is in
`configs/smoke/v1.toml`; the frozen research contract is in
`research/protocol_v1.md`.

The planned baseline will record prompt length, generated length, prefill and
decode time, decode throughput, estimated KV-cache bytes, device, and peak GPU
memory when CUDA is available. Generated results will belong in `results/` and
will not be hand-edited.

## 🔬 Experiments

The experiment sequence is documented in
[10_phase_phd_roadmap.md](10_phase_phd_roadmap.md). In brief:

1. Literature review and gap definition.
2. Full-cache inference harness.
3. Analytical and empirical memory scaling.
4. Attention and token-importance analysis.
5. Fixed compression baselines.
6. First importance-based adaptive policy.
7. Layer and head adaptive allocation.
8. Main quality-efficiency benchmark.
9. Ablations and failure analysis.
10. Generalization, statistical synthesis, and paper release.

### Evaluation controls

- Use the same model revision, prompt set, dtype, decoding parameters, and
	generated-token budget when comparing policies.
- Tune thresholds and weights on development data only.
- Freeze the final evaluation protocol before running the primary comparison.
- Report selection overhead; a policy that saves memory but costs more latency
	must be described that way.
- Preserve raw per-example results, not only aggregate tables.
- Report negative results and cases where a simple baseline wins.

## 🧪 Testing and validation

Testing is layered because this is both a research codebase and a numerical
measurement project.

### Fast checks

```bash
uv run python -m compileall -q src
uv run amt --help
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy
uv run pytest
uv run pre-commit run --all-files
git diff --check
```

The offline test suite constructs a tiny GPT-2 model locally. An optional test
also validates the decoding oracle against the immutable downloaded smoke model:

```bash
AMT_RUN_MODEL_INTEGRATION=1 uv run pytest tests/test_decoding.py
```

The declared dtype tolerances and the exact equivalence contract are documented
in [docs/NUMERICAL_TOLERANCES.md](docs/NUMERICAL_TOLERANCES.md).

### Planned unit tests

- KV-cache byte calculations across MHA, GQA, and MQA configurations.
- Cache index selection for full, recency, uniform, and adaptive policies.
- Budget accounting and exact retention counts.
- Stable normalization of importance signals.
- Configuration validation and seed propagation.
- Result schema validation.

### Planned integration tests

- Compare baseline logits or generated output with standard Transformers
	generation under deterministic settings.
- Verify that compressed cache shapes remain valid through multiple decode
	steps.
- Verify that no policy exceeds its configured budget.
- Run a tiny local model on a short synthetic prompt end to end.
- Recreate a figure from raw results in a clean environment.

GPU benchmarks are not unit tests. They will be run as recorded experiments
with hardware, software versions, seeds, and wall-clock conditions stored in
the result metadata.

## 📚 Documentation map

| Document | Purpose |
|---|---|
| [10_phase_phd_roadmap.md](10_phase_phd_roadmap.md) | Research roadmap, deliverables, experiments, and exit criteria |
| [README.md](README.md) | Project scope, architecture, stack, and reproducibility contract |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution workflow and repository standards |
| [SECURITY.md](SECURITY.md) | Responsible security disclosure |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Community expectations |
| [research/protocol_v1.md](research/protocol_v1.md) | Frozen hypotheses, comparisons, metrics, and exclusion rules |
| [docs/PHD_CV_10_DAY_CHECKLIST.md](docs/PHD_CV_10_DAY_CHECKLIST.md) | Ten-day implementation and validation checklist |

Planned research documents include a literature matrix, preregistered
evaluation protocol, experiment log, failure-analysis report, and manuscript.

## Validation reports

See [PR scope, compatibility changes and follow-ups](docs/PR_SCOPE.md) for the
current research-prototype boundary. Tracked pilot summaries are descriptive
negative results, not evidence of adaptive superiority.

To save tests, logs and model input/output traces in one timestamped folder, run
`uv run --frozen --group dev python -m amt.validation`. Add `--qwen` for the
cached primary model. See [validation report instructions](docs/VALIDATION_REPORTS.md).

## CPU development evaluator

`uv run --frozen --group dev python -m amt.evaluation` runs a 20-example
delayed-query retrieval pilot with all five policies, using the cached Qwen model
offline. It saves prompts, token IDs, scores, cache audits and a readable report.
The 50%/25% budgets apply to retained context, with question tokens appended
afterwards. This is not the continuous-generation or long-context benchmark.
See [pilot instructions and limitations](docs/EVALUATION_PILOT.md).

For the separately versioned, position/label-balanced chat pilot, use
`uv run --frozen --group dev python -m amt.evaluation --dataset-version v2 --examples 32 --seed 20260920`.
Its settings are declared in [the v2 development protocol](research/pilot_v2_protocol.md).

## ⚠️ Current limitations

This repository has a reproducible full-cache baseline and deterministic fixed
policy utilities. It does not yet establish that adaptive compression improves
quality, memory, or latency. In particular:

- Full-retention equivalence is validated on tiny GPT-2 models and pinned
  Qwen2.5-1.5B in BF16 on Apple M1 MPS.
- Position-correct Qwen2 compressed decoding is implemented for one unpadded
  sequence with shared retention across layers. See the
  [Day 3 walkthrough](docs/POSITION_CORRECT_CACHE.md).
- Compressed-cache semantics are validated on tiny Qwen models and pinned
  Qwen2.5-1.5B in CPU float32 with documented cross-shape numerical tolerances.
  BF16 compressed-cache equivalence remains unqualified; the Day 2 BF16 result
  applies only to full retention. See [numerical evidence](docs/NUMERICAL_TOLERANCES.md).
- The correctness runner uses explicit masks and full prefill; it is not yet
  the long-context memory/latency benchmark pipeline.
- Minimal attention-only and attention-plus-recency policies are implemented.
  Attention collection requires eager attention and has unmeasured overhead.
  History/EMA is deferred. See the [Part 4 walkthrough](docs/ADAPTIVE_SELECTION.md)
  and [scope decisions](research/deviations.md).
- No benchmark result or statistical conclusion is reported here.
- Model and context coverage will initially be limited by available hardware.
- Analytical KV size does not equal end-to-end peak GPU memory.
- Attention importance is an observed signal, not proof of future token utility.
- A selection policy may introduce latency or implementation overhead.
- Results from one small model cannot support claims about foundation models in
	general.
- Literature review must establish the defensible gap before novelty is claimed.

## 🤝 Contributing

Research contributions should include the question being tested, the expected
outcome, the baseline, the configuration, and the validation command. Keep
changes narrow and preserve raw outputs and negative findings.

```bash
git checkout -b research/short-description
uv run python -m compileall -q src
git diff --check
git commit -m "research: describe experiment"
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## 📄 License and citation

This project is released under the MIT License; see [LICENSE](LICENSE).

The project is an independent research work in progress. Do not cite it as a
paper, preprint, or published result until a manuscript and corresponding
archive actually exist.

## 👤 Author

**Shivam Shashank** — independent research project on adaptive KV-cache
compression for long-context Transformers.

- 🌐 Portfolio: [shivam-shashank.me](https://www.shivam-shashank.me/)
- 💼 LinkedIn:
	[shivam-shashank-2b5766217](https://www.linkedin.com/in/shivam-shashank-2b5766217/)
- 📧 Email: [shivamkumar872000@gmail.com](mailto:shivamkumar872000@gmail.com)
- 🐙 GitHub: [shivamshashank](https://github.com/shivamshashank)
