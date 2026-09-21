# Modest longer-context matched-budget experiment

## Question

On the calibrated direct-color task, does protected adaptive retention perform
better than protected recency when both physically retain the same 50% context
budget? Full cache is the reference. This is the first comparison after removing
the unreliable A–D answer-mapping step.

The [frozen protocol](../research/long_context_protocol.md) defines all settings,
checks and decision thresholds before inference. This is development evidence,
not held-out confirmation or a publication claim.

## Design in simple terms

There are eight questions at each of two sizes. The requested fact moves from the
start to the end of the context, while target colors are balanced. Full cache
keeps everything. Recency keeps the newest half plus protected token 0. Adaptive
uses fixed equal attention/recency weights and the same first-token protection.

| Facts | Context tokens | Query tokens | Compressed context budget | Cases |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 161 | 32 | 80 | 8 |
| 32 | 289 | 32 | 144 | 8 |

Each target fact has eight context tokens. We record how many remain after
pruning, which helps distinguish lost evidence from a model error after retention.
The query arrives only after pruning; policies do not see it when choosing tokens.

## Saved evidence

`scripts/run_long_context.py` constructs and validates the dataset, runs all 48
evaluations offline, saves inputs and outputs, audits physical cache sizes, and
checks every full-cache result against ordinary dense inference. Its `--verify`
mode independently checks the saved dataset, predictions, positions, budgets,
target-span retention, byte arithmetic, paired cells and aggregate report.

`tests/test_long_context.py` covers data balance, text-to-token target spans,
per-length baseline gates, the adaptive decision rule and corrupted evidence.

```bash
uv run --frozen --group dev python scripts/run_long_context.py
uv run --frozen --group dev python scripts/run_long_context.py --verify results/long-context/YOUR_RUN
```

The cached pinned model runs on CPU; no paid GPU or network download is needed.

## Checklist

- [x] Freeze data, prompt, model, policies, budget, metrics and decision rules.
- [x] Verify exact token lengths, eight-token target spans and dataset hash.
- [x] Add construction, gate, pairing, byte and corruption tests.
- [x] Complete all 48 model evaluations.
- [x] Verify saved evidence and report every failure.
- [x] Decide whether adaptive merits a larger development run or needs diagnosis.

The baseline gate requires reliable full-cache answers. Adaptive can advance only
if it gains at least two correct answers over protected recency and is not worse
at either length. Passing that rule would justify another experiment, not prove
superiority. Failing it is a useful negative result and triggers diagnosis.

## Findings — 2026-09-21

Local run: `results/long-context/20260921T180618Z-e536bebd`.
Dataset SHA-256: `2a88cf55ef59b7909af5de5f2a32725b52f94074e0a2c3e24fa20e34664d7a62`.
The saved-evidence verifier passed all 48 unique policy/example cells.

| Policy | 16 facts | 32 facts | Overall | Correct among 14 full-correct cases |
| --- | ---: | ---: | ---: | ---: |
| Full | 7/8 | 7/8 | 14/16 | 14/14 |
| Protected recency | 4/8 | 3/8 | 7/16 | 7/14 |
| Protected adaptive | 4/8 | 4/8 | 8/16 | 8/14 |

All 48 raw next tokens were valid color words; raw correctness equalled the
declared constrained accuracy. Full-cache staged and ordinary dense logits
matched exactly in all 16 reference cases (maximum absolute difference 0.0).
The baseline gate passed exactly at its minimum: 14/16 overall and 7/8 per size.

Adaptive versus recency was improved on 3 paired examples, worse on 2 and equal
on 11: net +1 correct. It tied recency at 16 facts and gained one at 32 facts.
This fails the predeclared +2 rule, so the decision is **diagnose**, not scale.

The physical context-cache bytes were exactly matched within each size:

- 16 facts: full 9,232,384 bytes; compressed 4,587,520 bytes (80/161 tokens).
- 32 facts: full 16,572,416 bytes; compressed 8,257,536 bytes (144/289 tokens).

These are K/V tensor bytes, not total RAM or a measured 2× end-to-end saving.

## Selection diagnosis

Protected recency retained an average 52.3% of each eight-token target fact;
protected adaptive retained 57.0%. Recency fully retained 8/16 target facts,
whereas adaptive fully retained only 3/16, because adaptive spread capacity over
fragments across the context. Adaptive retained the target color token in 7/16
cases; recency did so in 9/16. Both answered some cases without that token and
failed some cases with it, so these small counts do not identify causality.

The adaptive selector always retained the generic `The` and punctuation tokens
from target lines but retained the target entity/color much less consistently.
That is evidence that end-of-context attention plus recency is not sufficiently
aligned with the later question. The query is deliberately unseen during pruning,
so the policy cannot know which of many structurally identical records matters.

This result does not support adaptive superiority. It also does not show that
recency is generally optimal: both compressed policies lost half the full-cache
correct answers at this budget. The two full-cache failures and small development
sample further limit interpretation.

## Next required design change

Before scaling, test a query-aware selection stage. A realistic design is:

1. Prefill the context and the question body without the final answer cue.
2. Aggregate attention from question tokens to context positions.
3. Prune to an equal physical budget using those query-conditioned signals.
4. Append the held-back `Answer:` cue and score the direct color.

This preserves a genuinely unseen scoring token while allowing the retention
policy to know the user query. It needs a new frozen protocol and correctness
tests; it must compare query-aware adaptive against protected recency and the
current query-unaware adaptive policy at the same budget. Do not tune weights on
these 16 outcomes or describe the proposed method as novel before literature review.
