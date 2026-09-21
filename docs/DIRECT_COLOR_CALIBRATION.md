# Direct-color task calibration

## Purpose

The A–D task failed its full-cache reliability gate even though staged cache and
ordinary dense inference matched exactly. This milestone removes the extra step
of translating a remembered color into a shuffled letter. It asks for the color
word itself and determines whether that is a reliable development task.

This does not repair or overwrite earlier results. It is a separately versioned
task, and it does not evaluate cache compression.

## What is fixed before inference

The [protocol](../research/direct_color_protocol.md) fixes one prompt, all 32 v2
fact sets, two entity vocabularies, the cached model revision, CPU float32/eager
execution and a 64-case paired dataset. It also fixes the primary scorer and gate.

All four colors are one token in bare form and one token with a leading space.
The primary score compares bare `red`, `blue`, `green`, `yellow` token logits at
the assistant-message boundary. Leading-space scores and unrestricted raw output
are diagnostics; they cannot replace the declared primary score after outcomes.

The task passes only if each 32-case name set has at least 29 constrained-correct
and 29 raw-correct answers, all raw tokens are valid colors, and every staged
cache result matches ordinary dense inference within tolerance.

## Code and evidence

- `scripts/calibrate_direct_color.py` constructs and validates the exact dataset,
  runs the model offline, audits physical KV storage, compares dense logits, saves
  every input/output and generates a readable report.
- `tests/test_direct_color.py` checks pairing, semantic gold extraction, scoring,
  per-vocabulary thresholds, raw validity and rejection of corrupted evidence.
- `dataset.json`, `inputs.json`, `rows.jsonl`, `summary.json`, `run.json`,
  `verification.json`, `source_snapshot.zip` and `run.log` form the local evidence.

```bash
uv run --frozen --group dev python scripts/calibrate_direct_color.py
uv run --frozen --group dev python scripts/calibrate_direct_color.py --verify results/direct-color/YOUR_RUN
```

The model must already be cached. No paid GPU or network download is required.

## Checklist

- [x] Verify color tokenization before selecting the scorer.
- [x] Freeze one prompt, scorer, dataset and pass/fail gate.
- [x] Add construction, threshold and evidence-corruption tests.
- [x] Pass the full offline suite: 141 passed, 5 optional integrations skipped.
- [x] Complete and independently verify all 64 cached-model cases.
- [x] Record findings without changing the gate or scorer.
- [x] Decide whether to proceed to a modest longer-context compression protocol.

## Findings — 2026-09-21

Local run: `results/direct-color/20260921T174749Z-bfb7cc14`.
Dataset SHA-256: `7f394d9ede75f55600fd509f9ee26a2fa6c6cc69323959118f85cae64d356cf4`.
The saved-evidence verifier passed.

| Vocabulary | Constrained correct | Valid raw color | Raw correct | Bare/space agree |
| --- | ---: | ---: | ---: | ---: |
| Original | 32/32 | 32/32 | 32/32 | 31/32 |
| Renamed | 32/32 | 32/32 | 32/32 | 32/32 |
| Overall | 64/64 | 64/64 | 64/64 | 63/64 |

The predeclared task gate **passed**. All 32 original/renamed pairs had the same
primary and raw predictions. Results by gold color were red 22/22, blue 10/10,
green 20/20 and yellow 12/12; this color distribution is not balanced.

Staged full-cache and ordinary dense full-vocabulary logits matched exactly in
all 64 cases (maximum absolute difference 0.0), and physical KV storage checks
passed. Original and renamed prompt chunks were 81–82 context tokens and 30–31
query tokens, so this is still a short-context task.

The sole bare/space disagreement was original example 026: the frozen primary
bare-token scorer and raw greedy token predicted the correct `red`, while the
leading-space diagnostic selected `blue`. This is retained, not rescored. It
supports using the explicitly declared assistant-boundary form and shows why
token-form conventions must remain recorded.

Removing the A–D mapping made this development baseline reliable on these cases.
It does not prove generalization, because the task and examples were used for
calibration. It also does not show that any compressed policy preserves quality.

## Next experiment

Proceed to a separately frozen modest longer-context development comparison.
Use direct-color answers, full cache, protected recency and protected adaptive
at exactly equal context budgets. Start with CPU-feasible lengths and a small
number of examples; preserve inference/logging checks and selection overhead.
Do not include unprotected policies as competitive baselines, tune adaptive
weights on these examples, or consume the future confirmation set yet.

If the gate passes, the next compression comparison must include full cache,
protected recency and protected adaptive at equal context budgets. Fresh examples
must remain untouched for later confirmation. If the gate fails, task design—not
adaptive-policy tuning—is the next problem.
