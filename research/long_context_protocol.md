# Modest longer-context matched-budget experiment — 2026-09-21

Declared before model inference. Development experiment only; no held-out,
significance, generalization, superiority or speedup claim.

## Frozen design

- Generate 16 direct-color examples with seed 20260921: eight contexts with 16
  facts and eight with 32 facts. Entities are `Unit00`…`Unit31`; each fact is
  `The UnitNN locker is COLOR.` Distractor colors are seeded. At each context
  size, target fact indices are eight evenly spaced positions including both ends.
  Target colors cycle red, blue, green, yellow twice per size.
  Frozen dataset SHA-256: `2a88cf55ef59b7909af5de5f2a32725b52f94074e0a2c3e24fa20e34664d7a62`.
- Prompt: v2 single-user chat. The context ends with `End of records.` The query
  asks for the target unit and requests only red/blue/green/yellow, ending in
  `Answer:`. Primary scorer is the frozen bare single-token color argmax; save
  leading-space scores and unrestricted raw next token as diagnostics.
- Pinned cached Qwen2.5-1.5B-Instruct revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, CPU float32/eager, no downloads.
- Run full cache, protected recency and protected adaptive (fixed 0.5 attention /
  0.5 recency) for every example. Compressed policies retain exactly floor(50%
  of context tokens); query tokens are appended after pruning and are not charged
  to the context budget. Protection uses the already-declared minimal swap of
  original position 0 for the oldest selected position. No other sink/local rule.
- Total: 16 × 3 = 48 evaluations. No alternative prompts, weights, ratios,
  rejection, resampling or answer-based exclusions after outcomes.
- Pre-inference tokenizer inspection gives exactly 161 context + 32 query tokens
  for every 16-fact case, and 289 + 32 for every 32-fact case. Target fact spans
  contain eight context tokens. Compressed budgets are therefore 80 and 144.

## Evidence and checks

Independently parse gold colors. Verify exact chat token boundaries and record
the token span intersecting the target fact. For every row audit assigned budget,
retained positions and physical K/V storage. Save retained target-span count and
fraction, selection signals, logits, raw outputs, timing, settings, inputs, source
snapshot, partial reports and logs. Timing includes instrumentation and is not a
speed benchmark. KV bytes are tensor storage, not total process memory.

For full cache only, compare staged-cache and ordinary dense full-vocabulary
logits at atol=2e-4, rtol=1e-5. The baseline usability gate requires all 16 dense
checks, at least 14/16 constrained AND raw correct overall, at least 7/8 of each
at both fact counts, and all raw next tokens valid colors.

Primary compression reporting is accuracy among examples the full cache answers
correctly, paired by example, plus raw validity/correctness. Report each length,
target depth, target-span retention, improved/regressed/unchanged pairs and exact
memory bytes. Adaptive is only "promising for a larger development run" if it is
at least +2 correct overall versus protected recency, is not worse at either
context size, and the baseline gate passes. This is a project decision threshold,
not statistical evidence. Otherwise diagnose selection before scaling. Preserve
negative results and all failures.

Command: `uv run --frozen --group dev python scripts/run_long_context.py`
