# Part 5: CPU retrieval pilot

## Balanced chat development v2

Completed run and interpretation: [PILOT_V2_FINDINGS.md](PILOT_V2_FINDINGS.md).
All 288 rows verified. Adaptive did not beat uniform and its constrained answers
collapsed to A; read the raw-output diagnostic before interpreting its 25% score.

```bash
uv run --frozen --group dev python -m amt.evaluation --dataset-version v2 --examples 32 --seed 20260920
```

V2 uses all 32 combinations of eight fact positions and four answer labels, in
seeded random order. Chat formatting is shared by every policy; its exact token
boundary before the question is checked. The full-cache path is independently
validated against dense inference on each example. Protocol and limitations:
`research/pilot_v2_protocol.md`. V2 requires a multiple of 32 examples to preserve
cross-balancing; the default/no-flag command below still runs the original v1.

Dataset and prompt both change: do not interpret changes from v1 scores as the
effect of chat formatting alone. Compare all policies within each run.

Independently verify a completed run's saved evidence with:

```bash
uv run --frozen --group dev python scripts/verify_pilot.py results/evaluation/YOUR_RUN
```

This checks the dataset hash, gold answers parsed from facts, v2 balance, exact
expected row set, scores, aggregates, retained positions and KV byte arithmetic.
It does not rerun inference or prove peak-memory behavior. Never use Python `-O`
for this verifier because its consistency checks use assertions. Unit tests
exercise both valid records and deliberate corruption.

## Original v1

Run from the repository root:

```bash
uv run --frozen --group dev python -m amt.evaluation
```

The default is 20 development examples and two context-cache ratios (50%, 25%).
Full cache runs once per example; four compressed policies run at each ratio:
180 scored evaluations total. All inference uses the already cached pinned
Qwen2.5-1.5B-Instruct, CPU float32/eager attention. Missing cached weights fail
without downloading. A smaller smoke check is `--examples 1`.

## What happens

1. `make_examples` creates deterministic facts, questions and gold A-D labels.
   Options are shuffled and gold labels balanced across the 20 questions.
2. The model reads eight facts using the position-correct cache adapter.
3. Each policy selects a retained context set at the assigned capacity.
   Attention-based policies see only context attention, not the future question.
4. `audit_cache` checks every physical K/V length and backing storage size.
5. The question is appended at its original positions. Its tokens are additional
   to the context budget, identically for all methods.
6. The highest-scoring A-D token is the constrained answer. The unrestricted
   greedy token is also recorded so that formatting problems are visible.
7. Each completed row is flushed to disk and the readable report is updated.

This ordering matters: a normal first generated token is predicted before our
decoder's first pruning operation. That token alone cannot test compression.

## Evidence files

Each invocation prints a new `results/evaluation/<run>/report.md` path:

- `dataset.json`: actual facts, questions, options and gold answers.
- `inputs.json`: exact context/query token IDs and candidate IDs.
- `rows.jsonl`: all candidate scores, constrained/raw answers, retained positions,
  attention signals, physical cache bytes and elapsed time.
- `summary.json`: counts, accuracy and the full-cache-correct subset.
- `run.json`: immutable model revision, versions, seed and dataset hash.
- `source_snapshot.zip`: the source and protocol used, including uncommitted code.
- `run.log`: loading messages, per-evaluation progress and failure traceback.

Exceptions/interruption mark the report incomplete and preserve finished rows.
There is no automatic per-example timeout yet. Inspect `run.log` while it runs.

## What this does not establish

This pilot does not demonstrate a long-context advantage, statistical
significance, novelty, publication readiness or free-generation accuracy.
Chance constrained-choice accuracy is 25%. The small tasks may be too easy.
Retained KVs can contain information from other context tokens before pruning;
deleting a token's cache entry is not proof that all its information was removed.
Logical KV bytes are not peak process memory. Storage checks do not detect
arbitrary hidden copies elsewhere in the process. Instrumented runtime is not a
speed benchmark. All data is development data, not held-out confirmation.

The original Day 5 gate additionally asks for continuous generation-budget
audits, global hidden-copy detection and timeout recording. Those are not
silently marked complete by this reduced context-retention pilot.

## Completed pilot: 2026-09-19 local time

Run: `results/evaluation/20260918T232554Z-e48814cb` (folder timestamp is UTC).
All 180 evaluations completed, all cache audits passed. Independent recomputation
checked unique cells, gold/prediction correctness, overall/subset accuracy,
capacity arithmetic and the saved dataset SHA-256.

| Policy | 50% context budget | 25% context budget |
| --- | ---: | ---: |
| Recency | 3/20 (15%) | 5/20 (25%) |
| Uniform | 9/20 (45%) | 7/20 (35%) |
| Attention | 4/20 (20%) | 2/20 (10%) |
| Adaptive | 7/20 (35%) | 4/20 (20%) |

Full cache: 16/20 (80%). These small development results do not establish an
adaptive advantage. On the 16 examples full cache answered correctly, uniform
scored 9/16 and 7/16 versus adaptive 6/16 and 2/16 respectively.

### Next decision gate

1. Inspect the four full-cache errors before claiming the task is a valid
   compression benchmark. Check raw greedy tokens, prompt formatting and option
   scores. Diagnose first; do not silently remove failed examples.
   **Completed:** see `FULL_CACHE_DIAGNOSIS.md`. Dense and custom full-cache
   logits matched exactly on all 20 examples; chat formatting fixed two errors.
   Original pilot scores remain unchanged.
2. Improve development dataset construction: answer labels currently cycle with
   target fact location, introducing a position/label association. Randomize
   these independently and balance both in a separately versioned dataset.
3. Test context length and needle depth on development data; explain why
   last-context-token attention may poorly predict a later unseen question.
4. Freeze the revised protocol/settings before generating held-out examples.
   Do not reuse this pilot as confirmation or tune on held-out results.
5. Only pursue an adaptive superiority claim if it survives stronger matched
   baselines and an adequately powered held-out paired comparison. Otherwise
   narrow the contribution to validated cache semantics and measured limitations.

The reduced Part 5 deliverable is complete. The original broader Day 5 gate
remains partially open; this document makes those boundaries explicit.
