# Balanced chat pilot v2: findings

Run: `results/evaluation/20260919T103159Z-57ebffee`.
Declared protocol: `research/pilot_v2_protocol.md`.
All 288 evaluations completed; original v1 data/results remain unchanged.

## Verification

- 32 unique development examples; all eight positions × four labels covered once.
- Nine complete groups, 32 rows each. Full cache evaluated once per example.
- Independent verifier passed dataset hash, gold facts, balance, expected unique
  cells, predictions, aggregates, subset scores, budgets and byte arithmetic.
- All physical cache/storage audits passed.
- All 32 full-cache split/dense comparisons passed, maximum absolute logit
  difference 0.0 in this run.
- Final validation run `20260919T104300Z-02c2929b`: 93 tests passed, five optional
  model tests skipped; formatting/lint/types passed. Separate verifier lint also
  passed. No GPU rental, downloads or training.

## Primary result: constrained A-D accuracy

| Policy | 50% context budget | 25% context budget |
| --- | ---: | ---: |
| Recency | 8/32 (25.0%) | 8/32 (25.0%) |
| Uniform | 13/32 (40.6%) | 12/32 (37.5%) |
| Attention | 9/32 (28.1%) | 12/32 (37.5%) |
| Adaptive | 8/32 (25.0%) | 8/32 (25.0%) |

Full cache: 22/32 (68.8%). No adaptive advantage is demonstrated. Neither this
small development dataset nor the old dataset is held-out confirmation.

Dataset and prompt both changed from v1; the lower full-cache percentage cannot
be attributed to chat formatting alone. The earlier 18/20 chat diagnostic was
on the ORIGINAL examples, not a guarantee of 90% on new balanced examples.

## Important diagnostic: primary scores hide non-answer output

The declared primary metric deliberately restricts choices to A-D; the saved
unrestricted next token reveals a separate failure. This metric was not changed
after seeing results.

| Policy | Budget | Unrestricted next token matches gold after whitespace trim | Disagrees with constrained prediction |
| --- | --- | ---: | ---: |
| Full | full | 22/32 | 0/32 |
| Recency | 50% | 0/32 | 32/32 |
| Uniform | 50% | 13/32 | 0/32 |
| Attention | 50% | 9/32 | 0/32 |
| Adaptive | 50% | 0/32 | 32/32 |
| Recency | 25% | 0/32 | 32/32 |
| Uniform | 25% | 11/32 | 3/32 |
| Attention | 25% | 12/32 | 0/32 |
| Adaptive | 25% | 0/32 | 32/32 |

This is a next-token diagnostic, NOT accuracy after a multi-token completion.
Adaptive selects A in all 32 cases at BOTH budgets under constrained scoring.
Its unrestricted token is instead punctuation, an end marker, or words such as
`The` and `Remember`. Recency has a similar non-answer problem. Thus 25% is an
always-A pattern on balanced gold labels, not evidence of successful retrieval.
There are 131 raw/constrained disagreements across the complete run.

Loss of important chat-prefix tokens is one hypothesis, not yet a demonstrated
cause. Current policies have no protected sink/local tokens. Do not claim the
failure is solved by protecting tokens without a matched-budget ablation.

## Next steps, in priority order

Saved-evidence diagnosis completed on 2026-09-21. See
`V2_DIAGNOSIS_CHECKLIST.md` for the findings and explicit pending experiment.

1. Inspect full-cache errors and label/entity effects. Fixed locker identity is
   still associated with position; v2 removes gold-label/position confounding,
   not every dataset bias. Establish a competent baseline before larger runs.
2. Inspect retained chat-prefix positions and non-answer outputs. Predeclare a
   small, matched-total-budget protected-prefix ablation shared fairly across
   relevant baselines. Do not change weights opportunistically during the run.
3. Report constrained accuracy AND answer-format validity in future protocols;
   retain this run's original primary metric and raw outputs.
4. Freeze a reliable development setup, then expand context lengths/depths and
   design held-out confirmation. No paid GPU is needed for the next diagnosis.

Continuous-generation budget enforcement, timeout supervision, global hidden-copy
auditing and peak-memory measurement remain outside this completed pilot.
