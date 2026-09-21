# Larger multi-seed query-aware development validation

The first query-aware experiment passed its project threshold on 16 previously
used development examples. This milestone checks whether that direction persists
on 32 newly generated examples across two seeds before touching confirmation data.

The [frozen protocol](../research/query_aware_larger_protocol.md) fixes seeds,
dataset hash, policies, token lengths, budgets, paired analyses and decision rule.
Confirmation seed 20261001 remains explicitly ungenerated and uninspected.

## Experiment size

| Development seeds | Examples | Policies | Scored evaluations | Discarded probes |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 32 | 4 | 128 | 32 |

The comparison remains CPU-only and uses the cached pinned model. Query-aware
selection still pays for an additional full-context probe, so this validates
selection quality only—not deployable efficiency.

The analysis reports paired bootstrap intervals and exact discordant-pair tests,
but treats them as descriptive development statistics. The decision additionally
requires a four-answer pooled advantage over both baselines and no loss in any
seed × context-size cell.

## Result

Run: `results/query-aware-larger/20260921T191645Z-5d0055c7`

The independent verifier passed all 128 scored rows. It confirmed unique and
complete policy/example cells, exact token and byte budgets, target positions,
discarded-probe evidence, dense-cache equivalence and an unconsumed confirmation
seed. Full-cache logits matched the dense reference exactly (maximum absolute
difference 0.0).

| Policy | All examples | Full-correct subset |
| --- | ---: | ---: |
| Full | 26/32 | 26/26 |
| Recency + protected first token | 16/32 | 14/26 |
| Context-adaptive + protected first token | 19/32 | 16/26 |
| Query-aware + protected first token | 26/32 | 22/26 |

On the predeclared full-correct subset, query-aware improved by 8/26 over
recency and by 6/26 over context-adaptive. The paired 95% bootstrap intervals
for those differences were [1/26, 15/26] and [1/26, 11/26], respectively. The
descriptive exact two-sided discordant-pair p-values were 0.0574 and 0.0703.
Query-aware was not worse in any seed × fact-count cell.

Mean target-fact retention at the same 50% scored-cache budget was 52.3% for
recency, 56.6% for context-adaptive and 66.8% for query-aware. This supports the
mechanistic interpretation that the query probe more often identifies the
relevant fact. It does not establish deployable efficiency: the query-aware
selector used 153.44 seconds of additional full-context probe computation over
the 32 examples, and that probe is then discarded.

## Decision

**Diagnose; do not run confirmation yet.** Each development seed achieved only
13/16 full-context correct, below the frozen 14/16 per-seed gate (and below its
7/8 requirement in one fact-count cell). The comparative advancement criteria
were met, but the calibration gate takes precedence. This is evidence that the
selection idea is promising and that the evaluation task/model pairing is not
yet reliable enough for the reserved confirmation run. Seed 20261001 remains
ungenerated and uninspected.

```bash
uv run --frozen --group dev python -m scripts.run_query_aware_larger
uv run --frozen --group dev python -m scripts.run_query_aware_larger --verify results/query-aware-larger/YOUR_RUN
```

## Checklist

- [x] Save the prior milestone in local commit `e1306c6`.
- [x] Freeze two new development seeds and dataset hash.
- [x] Reserve confirmation seed 20261001 without generating it.
- [x] Freeze paired analysis and advancement criteria.
- [x] Pass construction, evidence-corruption and decision-rule tests.
- [x] Complete and verify all 128 scored evaluations and 32 probes.
- [x] Apply the frozen decision rule: diagnose; confirmation is not yet justified.

## Next checklist

- [ ] Inspect the six full-context errors without changing the reserved seed.
- [ ] Determine whether failures come from prompt ambiguity, token scoring or
      the small model's retrieval reliability.
- [ ] Freeze one minimal calibration repair using development data only.
- [ ] Require the repaired full baseline to pass a new predeclared development
      gate before considering seed 20261001.
- [ ] Keep query-probe overhead explicit in every claim and report.
