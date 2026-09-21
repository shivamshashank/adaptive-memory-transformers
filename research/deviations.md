# Protocol deviations and scope decisions

## 2026-09-19 — Balanced chat-formatted development v2

Before running v2, declared `pilot_v2_protocol.md`: 32 position × label cells,
seed 20260920, chat formatting with a verified pre-question split, fixed bare
A-D scoring and unchanged policy weights/budgets. All policies rerun; old scores
remain unchanged. This is a second development pilot, not held-out confirmation.

## 2026-09-19 — Part 5 CPU development pilot

Recorded before the first evaluator pilot run. This is a narrowed development
experiment, not completion of the original continuous-budget/long-context gate.

- Twenty synthetic eight-fact locker-color questions; seed 20260919.
- Full cache once per example; four compressed policies at 50% and 25% of
  context token count (floor rounding), giving 180 evaluations.
- CPU float32, eager attention, pinned cached Qwen2.5-1.5B-Instruct; no downloads.
- Read context, prune once, then append the previously unseen question and score
  the four single-token labels A-D. Report constrained-choice accuracy, not
  free-generation quality. This avoids scoring a token predicted before pruning.
- Budgets apply to retained context; query tokens are additional and equal for
  every method. Check physical K/V lengths and backing-storage sizes before and
  after query append. This is not a global peak-memory or hidden-copy audit.
- Attention is from the last context token, not the future question. Adaptive
  weights stay 0.5/0.5, no sink/local protections. No tuning during this run.
- All examples are development-only. Full-cache failures remain in overall
  accuracy; separately report accuracy on the full-cache-correct subset.
- Runtime includes selection and collection overhead but is instrumented and
  fixed-order, so cannot support speed claims. No significance/novelty claims.
- Multi-step generation budget enforcement, timeout supervision, global live
  tensor-copy detection and held-out confirmation remain separate pending work.

## 2026-09-18 — Reduced Part 4, no paid GPU compute

Recorded before benchmark development/confirmation results exist. The user
selected Adaptive as the main new PhD application project, with no GPU rental,
and approved the reduced Part 4 plan. The frozen `protocol_v1.md` remains as a
record of the original broader proposal; this log records the changed scope.

- Implement current-query attention and original-position recency only.
- Omit frequency, historical attention and EMA from this version. Do not claim
  history ablations or H2O equivalence.
- Rank attention per layer, then average ranks for one shared retained set.
  Different physical budgets per layer remain deferred.
- Default weights 0.5/0.5 and zero protected tokens are development defaults,
  not choices validated on benchmark data.
- CPU float32 is the numerical validation configuration. Attention collection
  explicitly requires eager attention; do not silently switch a benchmark model.
- Use tiny Qwen fixtures for fast tests and the already cached 1.5B model for
  short integration checks. No model training or external paid compute.
- Proposed smaller context/budget grid must still be runtime-piloted and frozen
  in a separate experiment configuration before confirmation. No existing
  context grid, significance threshold or success criterion has been evaluated.

The reduced Part 4 gate is functional signal collection and demonstrated signal
influence on selection, plus model-backed generation tests. It is not evidence
for improved quality, speed, peak memory, novelty or publication readiness.
