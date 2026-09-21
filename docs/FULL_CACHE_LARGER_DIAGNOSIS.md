# Larger full-cache reliability diagnosis

## Outcome

The larger run's six full-context errors are model/task retrieval failures, not
label, constrained-scoring, staged-cache or dense-inference bugs. One frozen
prompt repair was evaluated on 32 fresh development examples and failed the
predeclared reliability gate. Confirmation seed 20261001 remains ungenerated
and uninspected.

## Evidence from the original larger run

- Every question named one unique record and every saved gold color matched it.
- Every unrestricted raw next token was a valid color and agreed with the
  constrained four-color prediction on the six failures.
- Staged full-cache and ordinary dense full-vocabulary logits matched exactly
  on all 32 examples (maximum absolute difference 0.0).
- Four failures had a gold-versus-predicted margin below 1.0 logit; two were
  confident retrieval errors with margins of -5.57 and -12.12.
- Query-aware compression answered four of the six full-context failures
  correctly. This shows that full context is not an accuracy oracle on this
  task and is consistent with distractor interference.

## Single frozen repair

The [predeclared protocol](../research/exact_lookup_calibration_protocol.md)
made one change: the query repeated the requested identifier in an explicit
exact-record lookup instruction. It used two fresh seeds, preserved balanced
colors and target depths, and kept the model, context, chat template, scorer,
dense check and strict per-seed gate unchanged.

Verified run: `results/exact-lookup/20260921T195414Z-0eee3f6c`.

| Seed | 16 facts | 32 facts | Overall |
| ---: | ---: | ---: | ---: |
| 20260925 | 6/8 | 7/8 | 13/16 |
| 20260926 | 8/8 | 5/8 | 13/16 |
| Combined | 14/16 | 12/16 | 26/32 |

All raw tokens were valid colors, raw and constrained correctness agreed, saved
evidence verification passed, and the maximum staged/dense difference was 0.0.
The calibration gate failed. The repaired prompt obtained the same 26/32 total
as the preceding larger run and did not improve per-seed reliability.

Failures also repeated at fixed balanced-design slots across independent seeds:
32-fact depth 0.419 with yellow, 32-fact depth 0.581 with red, and the final
yellow position were each missed in both prompt experiments. This is evidence
of systematic model/task interaction, not random malformed output. It does not
identify an internal causal mechanism.

## Interpretation for the project

Do not lower the old gate, try additional wording on these outcomes or consume
the confirmation seed. Qwen2.5-1.5B full context is not reliable enough to act
as a correctness oracle for this synthetic longer-context construction.

The ground-truth comparison remains potentially useful. In the larger run,
query-aware scored 26/32, full scored 26/32, context-adaptive scored 19/32 and
recency scored 16/32. Query-aware versus full had four wins and four losses;
therefore compression sometimes removed harmful distractors as well as losing
useful evidence. An all-example paired analysis is post-outcome and descriptive,
not a replacement primary result.

## Next research decision

The highest-integrity next move is a separately predeclared development protocol
that treats known ground-truth accuracy across every example as primary and full
cache as a comparator rather than an oracle. It must retain paired equal-budget
baselines, report query-probe compute, and use fresh development seeds before
confirmation. If a full-cache correctness oracle is essential instead, change
the model or task and calibrate it separately; do not keep tuning this prompt.
