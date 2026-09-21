# Protected first-token ablation

## What this PR tests

The merged prototype physically prunes Qwen's KV cache and records reproducible
development results. Its v2 diagnosis found that policies dropping token 0 also
failed to produce a raw A–D next token. That association was confounded by policy.

This PR tests a narrower question: does swapping token 0 back into the same
41-token budget change answer formatting or constrained-choice accuracy?
It does not test a new adaptive algorithm or establish a causal mechanism.

The [frozen protocol](../research/protected_prefix_protocol.md) specifies the
32 examples, seed, model and minimal-swap rule before inspecting these outcomes.
Uniform and attention-only selections already containing token 0 are unchanged
controls. Recency and adaptive can change by exactly one retained token.

## Code map in plain language

- `src/amt/policies.py`: a wrapper first asks the original policy which tokens
  to keep. If protection is enabled and token 0 is absent, it replaces the oldest
  selected token with token 0. It cannot recover a token already evicted earlier.
- `src/amt/compressed.py`: recognizes attention requirements through the wrapper,
  so wrapping an attention-based policy does not disable attention collection.
- `src/amt/evaluation.py`: runs full cache once and each compressed policy twice
  per example. It saves the raw next token, A–D logits, retained positions and
  paired changes. A valid next letter is not necessarily a correct answer.
- `scripts/verify_pilot.py`: independently recomputes the expected swap, metrics
  and pair counts from saved evidence, and checks unchanged-control outputs.
- `tests/test_prefix_protection.py`: tests boundaries, malformed signals,
  unchanged controls, tiny-model inference and deliberate verifier corruption.

## Reproduce without paid compute

The pinned model must already exist in `.model-cache/huggingface`. Evaluation is
offline CPU float32, not a GPU benchmark. Raw artifacts remain local and ignored
by git; source snapshots preserve the exact evaluated code.

```bash
uv sync --frozen --group dev
uv run --frozen --group dev python -m amt.validation
uv run --frozen --group dev python -m amt.evaluation --dataset-version v2 --examples 32 --seed 20260920 --ratios 0.5 --prefix-ablation
uv run --frozen --group dev python scripts/verify_pilot.py results/evaluation/YOUR_RUN
```

`report.md` is the readable report; `run.log` contains execution output;
`dataset.json` and `inputs.json` contain text and token inputs; `rows.jsonl`
contains all model results; `paired_changes.json` records each on/off comparison.
The experiment has 288 evaluations and 128 compressed pairs.

## Completion checklist

- [x] Freeze intervention and equal-budget protocol.
- [x] Implement shared wrapper, paired outputs and independent checks.
- [x] Pass offline validation: 124 tests passed, 5 optional integrations skipped.
- [x] Complete all 288 cached-model evaluations.
- [x] Verify artifacts and all 64 unchanged-control pairs.
- [x] Compare rerun off arms with the original v2 evidence: 160 matching rows.
- [x] Report findings and limitations before choosing a follow-up experiment.

## Results: 2026-09-21

Local run: `results/evaluation/20260921T123428Z-12dae271`.
Validation: `results/validation/20260921T123336Z-1e9fae4c`.
These paths identify gitignored local artifacts, not published downloads.

All entries below use the same 32 development examples. The compressed arms
retain 41 context tokens (2,351,104 KV bytes); full retains 82 (4,702,208 bytes).
Question tokens add equally to each cache after pruning.

| Policy | Constrained correct, off → on | Valid raw next letter, off → on | Raw next letter correct, off → on |
| --- | ---: | ---: | ---: |
| Recency | 8 → 23 | 0 → 30 | 0 → 21 |
| Uniform | 13 → 13 | 32 → 32 | 13 → 13 |
| Attention-only | 9 → 9 | 32 → 32 | 9 → 9 |
| Adaptive | 8 → 19 | 0 → 27 | 0 → 15 |

Full cache: 22/32 constrained correct, 32/32 valid next letters and 22/32 raw
next letters correct. Raw correctness is stricter here: constrained scoring picks
the largest A–D logit even when another vocabulary token wins overall.

Paired constrained-accuracy changes (improved / regressed / unchanged):

- Recency: 20 / 5 / 7.
- Adaptive: 16 / 5 / 11.
- Uniform and attention-only: each 0 / 0 / 32.

Validity improved on 30 recency and 27 adaptive pairs, with no validity regressions.
Uniform and attention-only were unchanged on every selection and passed the
declared output-consistency checks. The independent evidence verifier passed.

Reproducibility comparison against `20260919T103159Z-57ebffee`: all 160 full/off
rows matched retained positions, gold labels, predictions and raw token IDs;
maximum absolute A–D logit difference was **0**. Dataset bytes and parsed token
inputs also matched exactly. Runtime is intentionally not compared as a speed test.

## Interpretation and next step

The equal-budget swap improves formatting and accuracy for recency and adaptive
on this development set. It does not help every example: each has five accuracy
regressions. Protected recency outperforms protected adaptive here (23 vs 19),
so there is still no adaptive-superiority result. Recency's 23 vs full's 22 is
not evidence that compression generally improves accuracy.

This intervention replaces another token; it does not isolate chat framing from
attention-sink effects or establish generalization. Full-cache accuracy is only
68.75%, and the data were already used for diagnosis.

Next PR: freeze a separate development protocol for full-cache label/entity
controls and longer contexts, retain protected recency as a strong matched-budget
baseline, and set acceptance criteria before running. Reserve a genuinely unseen
confirmation set until the evaluation design is settled. Do not tune weights on
these 32 examples or treat them as publication-ready evidence.

Longer contexts, held-out confirmation, 25% budgets and mechanism attribution
remain outside this PR. No merge or publication-readiness claim is implied.
