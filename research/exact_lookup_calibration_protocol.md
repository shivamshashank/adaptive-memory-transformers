# Exact-lookup prompt calibration — 2026-09-21

Declared before model inference. This is a fresh development calibration, not
confirmation and not a compression-policy comparison.

## Diagnosis motivating the single change

The six full-context failures in the larger development run had correct unique
labels, valid raw color outputs and exact staged/dense agreement. The unrestricted
raw token agreed with the constrained color prediction in all six cases, ruling
out the four-color scorer as the cause. Four errors had a competing-color margin
below 1.0 logit; two were high-margin retrieval errors. Query-aware compression
answered four of these six cases correctly, consistent with context interference.

## Frozen calibration

- Generate 32 new development examples from seeds 20260925 and 20260926: eight
  target depths at 16 facts and eight at 32 facts per seed. Colors remain balanced.
- Keep the record format, chat template, model, revision, CPU float32/eager mode,
  bare-color scorer and dense-cache check unchanged.
- Make exactly one prompt change: repeat the requested identifier in an explicit
  lookup instruction: `Look up the exact UnitNN record above. What color is the
  UnitNN locker?` The answer constraint and `Answer:` cue remain unchanged.
- Dataset SHA-256:
  `81cce90fd1103daefbbea128ba76df6c0121e1bb29a8c70521905063ab5f42ff`.
- Frozen tokenizer inspection gives 161/289 context tokens and exactly 42 query
  tokens for the 16/32-fact cases.
- Confirmation seed 20261001 remains ungenerated and uninspected.
- Run all 32 examples. Do not reject examples, try alternative wording or change
  the threshold after observing results.

## Gate

For each seed separately, require at least 14/16 constrained and raw correct,
at least 7/8 correct at each fact count, 16/16 valid raw color tokens and all
dense checks. Passing permits freezing this prompt for a new compression run;
failure means the current small model is not a sufficiently reliable full-cache
reference for this synthetic task. It does not permit lowering the gate.

Command: `uv run --frozen --group dev python -m scripts.calibrate_exact_lookup`
