# First-token minimal-swap development ablation

Declared 2026-09-21 before this experiment runs. Base: merged PR #1 (`db51ac7`).
This is exploratory development data already inspected, not held-out confirmation.

## Fixed settings

- Same v2 dataset: 32 examples, seed 20260920, eight positions × four labels.
  Expected dataset SHA-256: `4677b328abbb80e0eed87a32f0c3d1285d8d55d2055438fe7c7e7e385ec1224e`.
- Same pinned Qwen2.5-1.5B-Instruct, CPU float32/eager, offline cache, chat template,
  exact pre-question token split and bare single-token A-D argmax scorer as v2.
- Full cache once per example. Four compressed policies, protection off/on,
  50% retained-context budget only: floor(82 × 0.5) = 41 tokens in both arms.
  32 × (1 + 4 × 2) = 288 evaluations. Query tokens remain additional as in v2.
- Attention-only weights 1/0; adaptive 0.5/0.5; no other sink/local protection.

## Minimal intervention, shared across policies

1. Compute the original policy's B-token selection on the full aligned signals.
2. Off: return it exactly unchanged.
3. On and original position 0 already selected: return it exactly unchanged.
4. On and position 0 missing: replace the selected token with the SMALLEST
   original position by position 0. Sort cache slots, preserve absolute positions.
5. Require exactly B physical retained tokens; do not recompute ranks or scores.
   Position 0 must be present in the input cache; protection cannot resurrect
   an already evicted token. Budget 1 becomes just token 0; zero capacity stays empty.

The contrast estimates this particular first-token-for-oldest-selected-token
swap, not the effect of adding a token without opportunity cost. It does not
identify why the token matters (chat framing, attention sinks or another cause).
Uniform and attention-only that already keep token 0 serve as unchanged-selection
controls. Require matching outputs within declared float32 tolerance (2e-4,1e-5)
for those identical-selection pairs; retain and report any disagreement.

## Metrics, validation and decision

- Primary diagnostic: raw immediate next token, stripped of whitespace, is one
  of A-D. This is not multi-token answer validity.
- Retain original constrained-choice accuracy; also report raw-next-letter
  correctness. Save improved, worsened and unchanged pairs for both validity
  and constrained accuracy, keyed by example ID rather than execution order.
- Full-cache dense comparisons, physical-length/storage checks, source snapshot,
  raw outputs, settings, partial/failure reports and dataset hash remain required.
- Rerun all off arms, do not silently substitute old results. Compare old/off
  outputs as a reproducibility check when prior local artifacts are available.
- No threshold tuning, answer-based exclusions, weights changes or replacing
  primary scores after seeing results. Report all 32 examples and optionally
  the same full-cache-correct subset. No significance/superiority claim.
- If validity improves but accuracy does not, report that distinction. If neither
  improves, the hypothesis was not supported by this intervention. Either outcome
  is a completed experiment. Larger contexts/25% budgets/held-out work are deferred.

Command:
`uv run --frozen --group dev python -m amt.evaluation --dataset-version v2 --examples 32 --seed 20260920 --ratios 0.5 --prefix-ablation`
