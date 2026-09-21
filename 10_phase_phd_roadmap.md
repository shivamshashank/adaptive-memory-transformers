# Adaptive Memory Transformers — 10-phase research roadmap

**Purpose:** turn the present Adaptive Memory Transformers prototype into a small, rigorous ML-systems research artifact that strengthens a September 2027 PhD application. The target is a defensible result, even if the result is negative—not an impressive-looking demo.

**Working title:** *Position-correct adaptive KV-cache allocation for long-context decoding.*

**Core question:** at the same KV-cache budget, can a **causal**, position-correct adaptive policy preserve long-context task quality better than strong fixed retention policies, while reducing memory and retaining practical decoding speed?

**Primary hypothesis (H1):** the proposed adaptive policy improves task quality over the best pre-registered fixed baseline at one or more low-memory budgets, with a paired confidence interval that excludes zero on the primary controlled benchmark.

**Null / equally valuable outcome (H0):** after correcting cache semantics and comparing against matched baselines, no adaptive advantage remains. That is publishable as a precise negative result if the artifact and diagnosis are strong.

This roadmap deliberately fixes the issues found in the audit: the current `recency` and `frequency` signals do not affect selection, the "frequency" signal is an age proxy, per-layer allocation is not applied to the decoder, the full-cache comparison is not budget matched, and pruning a `DynamicCache` can lose the original positions required by the model.

## Research contract — freeze this before coding

### Claim boundaries

The project may claim only the following if supported by the final results:

1. **Functional correctness:** the custom cache is semantically equivalent to normal decoding at 100% retention, within a stated numerical tolerance.
2. **Memory-quality trade-off:** task quality, cache bytes, and latency change by measured amounts under a fixed experimental protocol.
3. **Causal policy benefit (if observed):** any advantage over pre-registered, budget-matched baselines.

It must **not** claim that retained-attention mass is model quality, that a toy passkey task demonstrates general long-context ability, or that a full cache is a matched low-memory baseline.

### Initial method specification

At decoding step `t`, for cached position `i` and layer `l`, the policy chooses tokens for the next step using only information available by time `t`:

`score(i,l,t) = a_l * rank(EMA_attention(i,l)) + b_l * rank(recency(i,t)) + c_l * rank(historical_attention_demand(i,l)) + d_l * rank(current_query_attention(i,l))`

- `EMA_attention`: exponentially decayed attention received over previous decode queries.
- `recency`: a stated monotonic function of the **true original position**. It is a real recency signal, not an unused argument.
- `historical_attention_demand`: the running attention-demand statistic for that position. Do not call this token frequency; it is not token frequency.
- `current_query_attention`: attention from the just-completed query; it can decide what is retained for the following step, never for the query already scored.
- Every feature is rank-normalised per layer before combining it. Coefficients and EMA horizon are selected once on the development split, then frozen.

Start with the simpler and easier-to-audit design: the same retained-position set in every layer. The layer-adaptive extension is a separate comparison: it must allocate a real `B_l` to each layer *and apply it in that layer's cache*, against an equal-per-layer budget. Never infer layer adaptivity merely from reported scores.

## Tech stack

Use a compact, reproducible Python stack. Pin exact versions and model revisions in a lockfile; the names below are roles, not permission to use floating `latest` dependencies.

| Area | Choice | Why / rule |
|---|---|---|
| Runtime | Python 3.11, `uv`, `pyproject.toml` + `uv.lock` | A clone must create the same environment. |
| Model execution | PyTorch, Hugging Face Transformers, Accelerate | Build the correctness reference here. Transformers documents distinct dynamic, static and quantized cache behaviours, so record the cache type explicitly. |
| Custom cache | A small model-family adapter around original cache tensors, positions and attention masks | Do **not** mutate/prune `DynamicCache` and assume reindexed positions are valid. |
| GPU path | CUDA + BF16; FlashAttention only as an optional, separately recorded optimisation | Use Apple MPS/CPU for unit tests and smoke tests only—never compare its speed with CUDA results. |
| Benchmarking | Official RULER runner; LongBench / NeedleBench adapters; `lm-evaluation-harness` only for standard short-context regression checks | Keep the long-context runner under project control and retain raw per-example output. |
| Config / orchestration | Hydra or typed YAML configs, one command per immutable run | No notebook-only experiments. |
| Data and results | Hugging Face Datasets where appropriate; Parquet/JSONL for per-example records; SHA-256 manifests | Version, hash and document every download; do not commit restricted data. |
| Statistics / plots | NumPy, SciPy, Pandas or Polars, Matplotlib/Seaborn | Paired bootstrap CIs and plots generated from raw tables. |
| Quality controls | `pytest`, `ruff`, `mypy` or `pyright`, `pre-commit`, GitHub Actions | Tests and source must be tracked and visible in the public repository. |
| Optional serving comparison | vLLM only after the reference implementation is validated | Treat it as a throughput integration study, not evidence for the policy itself. |

Hugging Face notes that cache type changes the memory/speed trade-off and that a dynamic cache grows with generation; this is exactly why the implementation, cache type, and position handling must be recorded per run. [Transformers cache strategies](https://huggingface.co/docs/transformers/main/en/kv_cache)

### Compute tiers

- **Tier 0 — local laptop:** static analysis, unit tests, synthetic 0.5–1.5B smoke model, and one short 4K pilot. No headline numbers.
- **Tier 1 — primary evidence:** a CUDA GPU with enough VRAM for BF16 cache experiments, beginning with a 3B-ish open causal model at 8K–32K context and batch size 1. Record GPU, driver, CUDA, model revision, precision and maximum allocated VRAM.
- **Tier 2 — replication:** a second, architecturally different 7B-ish open model on a larger GPU at pre-registered lengths. If this compute is unavailable, narrow the paper's claim to one model; do not substitute a tiny random model and call it replication.

Suggested concrete starting pair, subject to access/licence checks: `Qwen/Qwen2.5-3B-Instruct` for the primary model and `mistralai/Mistral-7B-Instruct-v0.3` for replication. Pin each to an immutable Hub revision and test the adapters separately.

## Dataset and benchmark plan

| Dataset / suite | Role | Split discipline | Primary metrics |
|---|---|---|---|
| [RULER](https://arxiv.org/abs/2404.06654) | **Primary controlled benchmark.** It covers 13 long-context retrieval and multi-hop tasks and permits controlled length/complexity generation. | Generate a development set with fixed public seeds, then a disjoint held-out seed list for confirmation. Select policy coefficients only on development. | Official task score, per-task score, error rate by length. |
| [LongBench](https://arxiv.org/abs/2308.14508) | **External generalisation.** Its 21 datasets span six long-context task categories. | Never tune on LongBench. Use the published prompts/metrics; document exclusions caused by model context or licence. | Published task metric plus macro average with no hidden reweighting. |
| [NeedleBench](https://arxiv.org/abs/2407.11963) | **Stress test.** It separates retrieval and reasoning over a broad length/depth range. | Final challenge only; pre-register tested lengths within the model's native context window. Do not advertise 1M-token performance unless actually evaluated. | Retrieval and reasoning score vs. depth and length. |
| Long-document continuation set (optional, secondary) | Tests next-token log-likelihood / distribution drift when a cache is pruned during continued text. | Fixed held-out texts, no policy tuning. Publish the source, licence and selection script. | NLL/perplexity delta and KL divergence from full-cache logits. |

RULER is preferable to the current passkey-only evidence because it was designed to vary sequence length and task complexity; LongBench supplies task diversity; NeedleBench makes retrieval-vs-reasoning failures visible. Use all three as complementary evidence, not as a single aggregate leaderboard.

## Ten phases

### Phase 1 — Freeze the research question and protocol

**Time:** days 1–3
**Objective:** make the study falsifiable before implementation choices blur the result.

**Build**

- Write `research/protocol_v1.md`: H1/H0, target model families, policy definitions, baselines, budgets, primary metric, exclusion rules, and analysis plan.
- Create `configs/protocol/v1.yaml` with the seed lists, length schedule, decoding settings, budget fractions and metric definitions.
- State one primary comparison: proposed causal policy versus the strongest fixed, matched-budget baseline on held-out RULER. Everything else is secondary.
- Add a research log. Every deviation gets a dated entry and its reason.

**Exit gate:** a reader can predict what would count as a win, loss, or inconclusive result without reading code.

### Phase 2 — Turn the prototype into a visible research artifact

**Time:** days 3–6
**Objective:** repair the current public-readiness failure: source, tests and experiment code are untracked while only a handful of root files are in Git.

**Build**

- Commit the real package, tests, documentation, lockfile, and configs. Keep raw datasets, credentials, virtual environments and large transient checkpoints out of Git.
- Replace the README's performance language with an accurate **prototype / validation in progress** status.
- Add `REPRODUCE.md`, a data card directory, an experiment-card template, `CITATION.cff`, and a license/third-party notices section.
- Add CI for install, lint, type checks and the full unit suite. Save coverage but do not optimise for a vanity percentage.
- Preserve the present prototype in a tagged branch (`prototype-before-position-correctness`) so later conclusions are traceable.

**Exit gate:** a clean external clone can run `uv sync` (or documented equivalent), `pytest`, and a 30-second smoke experiment. A visitor can see the actual implementation.

### Phase 3 — Implement a position-correct cache reference

**Time:** days 6–14
**Objective:** solve the validity-critical implementation problem before measuring quality or speed.

**Build**

- Replace direct cache pruning with an explicit `PositionAwareCache` / model-family adapter. Store, per layer: key, value, **original absolute position**, retained index map, cache length, and attention mask mapping.
- Pass true `cache_position` / position IDs and a causal attention mask that matches retained original positions. Never renumber non-contiguous tokens as though they were adjacent.
- Support greedy, batch-size-one decoding first. Beam search, speculative decoding and paged serving are out of scope until correctness is proved.
- Write a small reference decoder that reconstructs the exact no-pruning call for comparison. Keep all tensors BF16 for performance tests and add FP32 micro-tests where numerical tolerances are tighter.

**Required tests**

1. 100% retention: logits agree with the normal decoder (`max_abs_diff` threshold stated separately for FP32 and BF16); greedy token sequence is identical on fixed prompts.
2. Retained positions are strictly increasing, unique, in range, and preserved across every decode step.
3. An intentionally permuted/reindexed cache fails the semantic test—proving the test can detect the error it is meant to prevent.
4. Cache bytes equal the tensor allocation implied by the retained K/V shapes; no hidden full cache remains resident.
5. One test per supported architecture family and one long prompt crossing an eviction boundary.

**Exit gate:** full-retention equivalence passes in CI. If it does not, no pruning result may be reported.

### Phase 4 — Make the adaptive policy real, causal and ablatable

**Time:** days 14–19
**Objective:** turn the current score bookkeeping into an actual policy that changes cached tensors.

**Build**

- Extend `src/amt/policies.py` with a common selector interface: `observe(step_state)`, `select(layer, budget)`, and `apply(cache, indices)`; keep model-specific cache handling in a separate adapter module.
- Implement the four signals in the research contract. Unit-test that independently changing each signal changes selected indices under a crafted, deterministic fixture.
- Reserve any fixed sink tokens explicitly and charge them to the budget. Keep a policy audit log per step: raw signal summaries, normalised ranks, selected original indices, and cache bytes.
- First implement global selection across all layers. Implement actual per-layer `B_l` allocation only after the global version works; unit-test that allocated budgets sum to `B` and that each layer receives/apply its own selection.
- Choose score weights, horizon and sink count using only RULER development data; persist the complete search table, including bad configurations.

**Exit gate:** signal ablations demonstrably alter *selection* and per-layer allocation changes *real cache sizes*, not just a reporting table.

### Phase 5 — Establish fair baselines and non-circular measurements

**Time:** days 19–23
**Objective:** remove the current comparison and metric traps.

**Pre-register these policies**

| Policy | Budget treatment | Role |
|---|---|---|
| Full cache | 100% only | Upper bound, never a matched low-memory baseline. |
| Recent window | Exactly `B` positions | Simple strong recency baseline. |
| Uniform / strided | Exactly `B` positions | Position-coverage baseline. |
| Sink + recent window | Exactly `B`, including sinks | Streaming-style long-context baseline. |
| Attention-only heavy hitters | Exactly `B` positions | Tests whether adaptivity beats the attention signal alone. |
| Proposed adaptive | Exactly `B` positions | Causal multi-signal method. |
| Oracle future-attention selector | Clearly labelled diagnostic only | Ceiling analysis; never a deployable baseline. |

**Build**

- Give all budgeted methods the same cache adapter, prompt handling, decoding strategy and generated-token limit.
- Replace retained-attention mass as a quality result with task score, continuation NLL/KL (if used), exact cache bytes, peak allocated GPU memory, prefill time and steady-state decode latency.
- Measure latency after warm-ups, with synchronisation, fixed clocks where available, and at least 30 repetitions per throughput cell. Report median and percentile interval; prefill and decode separately.
- Ensure the `FullCachePolicy` is used once as the upper bound, not repeated under 75/50/25/10% labels.

**Exit gate:** the result table can never imply that full cache obeyed a restricted budget, and no primary quality metric is mathematically optimised by the selector itself.

### Phase 6 — Integrate data, task adapters and data cards

**Time:** days 23–28
**Objective:** replace toy/random prompts with an auditable benchmark suite.

**Build**

- Vendor no datasets; instead write versioned download/preparation scripts with source URL, commit/tag, licence, SHA-256 and expected example count.
- Add `datasets/ruler.py`, `datasets/longbench.py`, `datasets/needlebench.py` and a normalised `Example` schema containing ID, prompt, target, task, length, depth, source split and scoring callback.
- Create RULER development/confirmation seed manifests. The code must refuse to tune on IDs in the held-out manifest.
- Use the benchmark authors' official scoring where available. Save raw model completions and scorer version, then compute metrics from those raw records.
- Write data cards covering languages, synthetic-vs-natural distribution, licences, sensitive content and known mismatch with interactive chat workloads.

**Exit gate:** a fresh machine recreates the same input manifest and validates its hashes; every final result row traces back to an example ID and raw completion.

### Phase 7 — Build the experiment system and reproducibility ledger

**Time:** days 28–35
**Objective:** make each result a rerunnable, inspectable experiment rather than a CSV produced by a one-off script.

**Build**

- Extend the single `amt` CLI with an evidence-grade `run --config` path. Keep superseded prototype code in Git history rather than in the active source tree.
- Create a per-run directory containing immutable config, Git SHA and dirty state, package lock hash, model revision, tokenizer revision, device info, seed, stdout/stderr, raw JSONL and an aggregated Parquet table.
- Add a `manifest.json` that records command, start/finish time, cache implementation, context length, policy, budget, model and dataset hashes.
- Add a validation command that rejects mixed model revisions, duplicate example IDs, incomplete groups, out-of-budget caches and leakage of development examples into confirmation.
- Use `lm-evaluation-harness` for a small set of standard short-context regression tests only; it supports reusable configuration and common model backends, which helps catch broad regressions without replacing the custom long-context evaluator. [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

**Exit gate:** deleting the output directory and running the documented command recreates a pilot table and its figure from only tracked code plus downloaded data.

### Phase 8 — Pilot, red-team, and stop/go review

**Time:** days 35–42
**Objective:** find invalidity before paying for the full experiment grid.

**Pilot design**

- One primary 3B-ish model; RULER development only; 4K, 8K and 16K; three budgets (50%, 25%, 12.5%); all pre-registered budgeted policies.
- Include every benchmark task family, even if the example count is deliberately small. Do not report these as final performance results.
- Run 30 latency repetitions on two representative task/length cells, separately after warm-up.

**Red-team checklist**

- Can the full-cache reference and 100%-retention custom cache ever disagree? Diagnose first.
- Is any full cache retained invisibly on GPU or CPU?
- Do masked/pruned indices ever regain an invented adjacent position?
- Does a signalled ablation actually alter the selected cache?
- Do results reverse when prompts are reordered or when a new seed is used?
- Does any policy use future information? If yes, relabel it oracle and exclude it from the main claim.

**Stop/go rule:** proceed only if semantic tests pass, budgets are respected, raw logs are complete and the pilot is operationally stable. If the adaptive policy loses, continue—this is a diagnosis opportunity, not a reason to tune on the confirmation split.

### Phase 9 — Run the confirmatory research campaign

**Time:** weeks 7–11, parallel with PhD outreach
**Objective:** obtain statistically interpretable, external evidence.

**Pre-register the final matrix before launch**

| Study | Model(s) | Contexts / budgets | Policies | Purpose |
|---|---|---|---|---|
| Controlled primary | Primary 3B-ish model | 8K, 16K, 32K; 50%, 25%, 12.5%, 6.25% | Full upper bound once; all five matched-budget policies | Main causal comparison on held-out RULER. |
| Architecture replication | Second 7B-ish family | A pre-stated feasible subset, ideally 16K and 32K; same budgets where possible | Full, strongest fixed policy, attention-only, proposed | Checks that one model's positional behaviour is not the whole story. |
| External generalisation | Primary model, then replication if budget permits | Published LongBench limits / model native window | Full, best fixed, proposed | Task diversity, no tuning. |
| Stress test | Primary model | Pre-stated NeedleBench lengths/depths within native context | Full, best fixed, proposed | Retrieval-versus-reasoning failure analysis. |
| Systems study | Both models as available | 16K and 32K, chosen budgets | Full, best fixed, proposed | Cache bytes, GPU peak, prefill, steady decode latency. |

Do **not** quietly reduce the official confirmation example count after viewing outcomes. If compute limits force a narrower study, reduce the number of pre-registered lengths or models *before* confirmation and say so plainly.

**Analysis plan**

- Pair methods by exact example, model, context length, seed and generation settings.
- Report each task family and budget separately before any aggregate. Use a macro average only as secondary.
- Use a paired bootstrap confidence interval (for example, 10,000 resamples) for quality deltas. Report the effect estimate, interval, sample count, failure count and missing-data reason.
- Plot quality-vs-cache-bytes and quality-vs-decode-latency Pareto curves. Show all points, not only the best budget.
- Predefine how invalid JSON, OOM, timeout and failed generations count. Usually they are failures, not rows to silently discard.
- Run a final independently seeded robustness set **once**, without further hyperparameter selection.

**Exit gate:** every headline figure has raw per-example records, a reproducible command, an exact model/data revision and a stated confidence interval.

### Phase 10 — Convert the result into a research artifact and PhD evidence

**Time:** weeks 11–16; begin the writing while Phase 9 runs
**Objective:** make the work legible to faculty and admissions readers.

**Research outputs**

- A 6–8 page workshop-style paper or technical report: problem, cache-semantics threat model, method, protocol, results, failures, limitations and reproducibility statement.
- A public `results/` release with compact derived tables/figures, seed/config manifests and checksums. Place large raw artifacts in Zenodo/OSF/Hugging Face with a DOI only after verifying that the record is public and resolves.
- An `artifact_evaluation.md` with the one-command small reproduction and a costed full-reproduction path.
- A 2–3 minute screen-recorded demo that shows an input, selected original positions, cache bytes and a link to the raw run—not a cherry-picked answer alone.
- A project page with a precise one-sentence result. If the method does not win, make the contribution the diagnosis: e.g. *after enforcing position-correct semantics and matched budgets, attention-plus-recency did not outperform the fixed baseline on X; the released suite identifies where it fails.*

**PhD conversion**

- Add a one-page research brief: research question, your technical contribution, one figure, robustness/limitation, and next question.
- Use it in supervisor outreach as a link, not an attachment dump. Describe the project as **in progress** until the confirmatory study is complete.
- Update the CV only with verified numbers. Do not claim a paper, DOI, benchmark coverage or performance improvement before it is public and reproducible.
- Connect the next research question to ML systems: cache eviction under distribution shift, learned-but-causal budget allocation, cache quantisation plus selection, or serving-level integration—one question, not four new projects.

**Exit gate:** an independent reader can clone the code, understand the result and limitations in 10 minutes, and reproduce a small slice without contacting you.

## 30-day application-ready sprint

You can begin PhD applications/outreach on day 31 without pretending the whole paper is finished. The deliverable is an **honest, credible research trajectory**.

| Days | Roadmap work | Day-31 evidence |
|---|---|---|
| 1–3 | Phase 1 | Frozen question, hypotheses, protocol and seeds. |
| 3–6 | Phase 2 | Public repo with the actual source, tests and clear prototype status. |
| 6–14 | Phase 3 | Position-correct reference plus full-retention semantic-equivalence tests. |
| 14–19 | Phase 4 | A causal policy whose signals and layer allocation actually affect cache contents. |
| 19–23 | Phase 5 | Matched baselines and valid metrics. |
| 23–28 | Phase 6 | RULER/LongBench/NeedleBench data adapters, manifests and data cards. |
| 28–30 | Phase 7 start | One-command pilot runner, immutable experiment manifest and an initial smoke run. |

On day 31, start supervisor outreach and application drafting in parallel with Phases 7–10. Your truthful description then is: **“I am building a reproducible evaluation framework for causal, position-correct KV-cache allocation; the implementation and benchmark protocol are public, and the confirmatory evaluation is under way.”** Do not use comparative performance wording until Phase 9.

## Non-negotiable anti-patterns

- Do not report CPU/MPS latency as a GPU systems result.
- Do not use a random tiny model or passkey accuracy as final evidence.
- Do not call an age/position proxy “frequency.”
- Do not choose weights, models, lengths or examples after inspecting held-out results.
- Do not compare a budgeted policy against full cache as though both consume the same memory.
- Do not use retained attention mass as the main quality metric; the selector optimises it by construction.
- Do not introduce vLLM, quantised KV caches, batching, multi-GPU serving and layer adaptation into the main result simultaneously. They confound the causal question.
- Do not wait to publish code: the present untracked-source state is more damaging to PhD credibility than a modest or negative result.

## What would make this strong for PhD applications

The strongest version is not “I built an adaptive memory transformer.” It is: **“I found and repaired a subtle validity problem in cache-compression evaluation, designed a causal matched-budget study, and released a reproducible result with its limitations.”** That reads as research maturity in ML systems and trustworthy empirical ML—the profile that gives a supervisor something concrete to discuss.
