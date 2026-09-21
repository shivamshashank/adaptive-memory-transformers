# V2 diagnosis and next-step checklist — 2026-09-21

## Completed without new model inference

- [x] Reverified all 288 source rows and their dataset hash with `verify_pilot.py`.
- [x] Enumerated all ten full-cache failures, including exact facts/questions,
      predicted colors, other matching fact positions and gold/prediction logits.
- [x] Broke down full-cache accuracy by label, fact position and target color.
- [x] Mapped saved prefix token IDs back to character spans using the pinned
      cached tokenizer; checked that token IDs match saved inputs exactly.
- [x] Compared retained header tokens, position zero and whole target-fact spans.
- [x] Separated next-token answer-format validity from constrained accuracy.
- [x] Preserved original datasets, predictions and primary scores unchanged.

Evidence: `results/diagnostics/20260921T111552Z-6b366455/report.md` and
`analysis.json`. Reproduce with:

```bash
uv run --frozen --group dev python scripts/analyze_pilot_failures.py results/evaluation/20260919T103159Z-57ebffee
```

## Findings

The rendered header occupies 24 of 82 prefix tokens. Position zero is the opening
`<|im_start|>` token. At the 25% budget, only 20 context tokens can survive, so
protecting the entire 24-token header would violate the budget.

Across compressed rows, position zero absent: **0/128 valid next letters**;
position zero present: **125/128 valid next letters**. Recency and adaptive always
discard position zero; uniform and attention-only always retain it in this run.
This association is therefore confounded with policy choice. It is NOT a causal
estimate, 256 independent observations, or proof that token zero alone fixes it.

Full-cache accuracy by gold label: A **3/8**, B **8/8**, C **6/8**, D **5/8**.
Accuracy by entity/depth: Aster 4/4, Birch 1/4, Cedar 3/4, Dahlia 2/4,
Elm 3/4, Fern 3/4, Grove 4/4, Hazel 2/4. Entity identity and depth are inseparable
in this dataset. These small post-hoc cells suggest follow-up controls, not a
proven label bias, positional defect or model mechanism.

At 50% budget recency retains the whole target fact in 24/32 cases but produces
no answer letters as the immediate next token. Whole-fact survival is therefore
not sufficient for valid next-token behavior here. Conversely, partial KVs can
carry information about other context tokens, so whole-fact retention is not a
necessary condition for a correct answer either.

## Next controlled experiment (pending; no outcome claimed)

- [ ] Implement a shared first-token protection wrapper for all four compressed
      policies, not an adaptive-only exception.
- [ ] Compare protection off versus position zero protected, with **the same
      total 41-token capacity** (50% of 82), reducing other retained tokens by
      one as needed. Preserve original positions and aligned attention signals.
- [ ] Use all 32 existing development examples; do not select only failures or
      favourable examples. Full cache once plus four policies × two conditions
      gives 288 evaluations. Test 50% first; do not claim a 25% result.
- [ ] Freeze exact selection/tie behavior and settings before execution; unit
      test the wrapper, physical budgets and unchanged no-protection behavior.
- [ ] Record paired changes in next-letter validity, constrained accuracy and
      raw tokens, including unchanged and worsened cases. Report per policy.
- [ ] Keep weights, prompt, model, candidate scorer and seed fixed. Do not
      attribute any change to adaptive weighting when protection is the change.
- [ ] Treat the result as exploratory on previously inspected development data.
      If protection helps, independently verify before any held-out claim.

Separately pending for baseline diagnosis: counterbalance entity/depth and rotate
answer-option labels for the same underlying facts, with a protocol declared
before inference. No larger context benchmark or GPU spending is warranted yet.
