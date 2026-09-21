# Full-cache label/entity reliability diagnostic — 2026-09-21

Declared before running this diagnostic. Development only; not held-out evidence.

## Fixed experiment

- From v2 seed 20260920, take the first generated example at each of eight fact
  positions, without consulting predictions. Preserve its facts and target color.
- Cross four cyclic rotations of its option order with two entity vocabularies:
  original Aster/Birch/Cedar/Dahlia/Elm/Fern/Grove/Hazel and renamed
  Alice/Bob/Carol/David/Emma/Frank/Grace/Henry. Replace names consistently in facts
  and query. Do not change colors, target position, prompt or answer instructions.
- 8 base cases × 4 rotations × 2 vocabularies = 64 full-cache evaluations.
  Every position has all four gold labels in each vocabulary. Cases are paired
  and correlated, not 64 independent samples. Renaming also changes tokenization.
- Same cached pinned Qwen2.5-1.5B-Instruct revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, CPU float32/eager, seed 20260920,
  v2 chat rendering and bare single-token A–D scoring. No weight/prompt tuning.
- Check split-cache vs dense full-vocabulary logits on EVERY case, using
  atol=2e-4, rtol=1e-5. Save dense choice logits and maximum absolute difference.
  Independently parse gold from text and verify context/query token boundaries.

## Metrics and decision rule

Report constrained accuracy, raw next-letter validity/correctness, accuracy by
gold label and vocabulary, color predictions across rotations, and renamed pairs
improved/regressed/unchanged. Include every case and preserve partial failures.

Engineering gate for advancing this task to longer contexts: at least 29/32
constrained correct AND 29/32 raw-next correct in EACH vocabulary; all 64 split
vs dense checks pass. This is a development acceptance rule, not a statistical
significance threshold or proof of unbiased behavior. No exclusions or reruns
with changed settings to manufacture a pass.

If the gate fails, report the failure and propose a separately frozen task/prompt
calibration experiment before scaling contexts. Protected recency remains the
required strong baseline for subsequent compression experiments. Do not run
held-out confirmation or tune adaptive weights in this diagnostic.

Command: `uv run --frozen --group dev python scripts/check_retrieval_reliability.py`
