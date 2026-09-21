# Retrieval reliability: project milestone

PR work is paused. This milestone tests the evaluation task itself before adding
longer contexts or a more complicated adaptive policy.

## Why this comes first

The previous protected-prefix experiment improved compressed answers, but full
cache still answered only 22/32 original v2 examples correctly. If the reference
model struggles with the question format, a compression comparison can mix
retrieval loss with sensitivity to answer labels or entity names.

We therefore keep facts fixed and rotate which option is A/B/C/D, then rename
the lockers consistently. A correct model should preserve the requested color
across these equivalent questions. Renaming is an input intervention, not a
token-length-matched test or proof of an internal mechanism.

## What the code does

`scripts/check_retrieval_reliability.py` is deliberately separate from the frozen
v1/v2 evaluator, so historical results and protocols remain unchanged.

1. `make_controls`: picks the first v2 example at each fact position, without
   consulting outcomes, then creates 64 paired label/name variants.
2. `validate_data`: checks the declared construction and independently reads each
   requested color from the fact text to validate its gold answer.
3. `main`: renders the v2 chat prompt, runs full cache in context/query stages,
   then runs ordinary dense inference on the same entire token sequence. All
   vocabulary logits must match within the declared float32 tolerance.
4. `summarize`: counts correct and valid answers, checks whether predicted colors
   change across option rotations, and pairs original/renamed outcomes.
5. `verify`: checks saved labels, candidate scores, cache lengths, dense candidate
   agreement, dataset identity and report consistency. It does not independently
   rerun the model or prove that every saved measurement is authentic.

The [protocol](../research/reliability_protocol.md) freezes the settings and
requires at least 29/32 constrained AND raw-next correct per name set before
advancing this task to longer-context experiments. This is a practical project
gate, not a statistical hypothesis test.

## Run and inspect

```bash
uv run --frozen --group dev python scripts/check_retrieval_reliability.py
uv run --frozen --group dev python scripts/check_retrieval_reliability.py --verify results/reliability/YOUR_RUN
```

No paid compute or downloads: the pinned model must already be cached locally.
The run saves exact text/token inputs, gold labels, raw next tokens, A–D logits,
dense comparison results, cache sizes, timings, settings, a source snapshot and
logs. `report.md` is the readable entry point; `summary.json` contains paired
analyses. Raw artifacts are local and gitignored, not published attachments.

## Checklist

- [x] Freeze label/entity controls and the acceptance rule before inference.
- [x] Implement reproducible runner, paired summaries and saved-evidence checks.
- [x] Add eight tests for balance, semantics, thresholds and corrupted artifacts.
- [x] Complete and verify all 64 model evaluations.
- [x] Record the results and choose the next experiment without changing this gate.

## Findings — 2026-09-21

Local evidence: `results/reliability/20260921T140014Z-584307a0`.
Dataset SHA-256: `583680bdd4349b51fae04cb3bea06708fdfe3fe26da6880caf38c147174868dc`.
Validation: `results/validation/20260921T140022Z-d98a28fc` — 132 tests passed,
5 optional integrations skipped; formatting, lint, types and tiny-model demo pass.

| Name set | Constrained correct | Valid next letter | Raw next correct |
| --- | ---: | ---: | ---: |
| Original | 21/32 (65.6%) | 32/32 | 21/32 |
| Renamed | 19/32 (59.4%) | 32/32 | 19/32 |
| Overall | 40/64 (62.5%) | 64/64 | 40/64 |

The predeclared gate **failed**. Both name sets fall below 29/32. These are
eight paired base fact sets, not 64 independent random examples.

- Full-vocabulary split-cache vs ordinary dense logits matched exactly in all
  64 cases: maximum absolute difference 0.0. Cache storage checks also passed.
- Gold-label scores: A 7/16, B 15/16, C 9/16, D 9/16. These conditional accuracies
  show uneven behavior; they do not by themselves establish an internal bias.
- Only 6/16 position/name groups predicted the same color across all four option
  rotations. In the other 10 groups, equivalent option reorderings changed the
  predicted color while the facts and requested color were fixed.
- Renaming changed the predicted color in 10/32 pairs: accuracy improved in 3,
  regressed in 5, and was unchanged in 24 (including 2 changed wrong answers).
- Original and renamed contexts used 82 and 81 tokens respectively. Consequently,
  name identity and tokenization are not isolated causes in this comparison.
- All 64 raw next tokens were valid letters. Unlike the earlier unprotected
  compressed results, invalid immediate answer formatting is not the issue here.

This identifies task/model sensitivity, not its internal mechanism, and does not
prove a general retrieval defect. The saved-evidence verification passed. Original
v2 and prefix-ablation findings remain historical results, not silently rescored.

## Next required project step

Do not yet scale this multiple-choice task or tune adaptive weights against it.
First freeze a small **direct-color answer calibration**: keep the facts and target
questions, but request the color itself without the A–D mapping. Verify candidate
tokenization before choosing a single-token or sequence scorer; save raw output
and gold-normalization rules. This tests whether removing the label-mapping step
makes the baseline reliable; it is a different task and must be versioned separately.

Declare its pass/fail rule before inference, use development examples only, retain
the present failure report, and require fresh independent examples before scaling.
If calibration succeeds, compare full cache, protected recency and protected
adaptive at the same context budget on modestly longer CPU-feasible inputs.
Do not select the better-looking prompt and describe its development score as
held-out confirmation. No new model, paid GPU or PR action is required.

Protected recency remains the required baseline for future compression studies.
These eight fact sets are development data; reserve separate unseen examples for
later confirmation and do not tune adaptive weights on this diagnostic.
