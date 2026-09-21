# PR scope: validated KV-cache research prototype

This PR delivers cache correctness, reproducible development evaluation and
diagnostics. It does NOT establish adaptive superiority or publication readiness.

## Audit triage and compatibility

- Baseline timing: replaced two separate generations with one fixed-length greedy
  run. Prefill predicts token one; N-1 decode calls predict the rest. Total timing
  is the sum of instrumented phase times, not an end-to-end model-load benchmark.
  EOS does not stop this fixed-length measurement. Actual/estimated KV bytes now
  refer to the same cache length (prompt + N - 1). Old baseline timing records
  are historical and must not be pooled with these measurements.
- CPU/MPS peak GPU memory is unavailable (`null`), not zero. CUDA reports peak
  allocated tensor memory, not all GPU/process memory. MPS timing synchronizes
  the device, but MPS compressed correctness remains unqualified.
- First-token semantics: compressed.py explicitly documents unpruned first-token
  logits. The evaluator prunes BEFORE appending its unseen query. Do not score
  a one-token completion from the former as compressed retrieval.
- Budget rounding: the generic legacy helper rounds; the versioned evaluator
  floors. These are intentionally documented distinct contracts for this PR.
  Pilot v1/v2 floor rounding is frozen; do not silently rewrite old experiments.
- Requested baseline prompt length is an approximate text-expansion target;
  results record the actual retokenized count. It is not an exact-length contract.
- Historical audit prose is not authoritative evidence. Its significance claim
  is unsupported by these small development pilots and is not included in the PR.

## Portable evidence and reproduction

Tracked summaries: `PILOT_V2_FINDINGS.md`, `FULL_CACHE_DIAGNOSIS.md`,
`V2_DIAGNOSIS_CHECKLIST.md`, with settings in `research/pilot_v2_protocol.md`.
These contain counts and limitations readable without the local results folder.
Raw run paths in those documents identify LOCAL, gitignored artifacts; they are
not downloadable attachments. No raw-output archive is published by this PR.

```bash
uv sync --frozen --group dev
uv run --frozen pre-commit run --all-files
uv run --frozen --group dev python -m amt.validation
# Requires the documented model revision ALREADY cached under .model-cache/huggingface:
uv run --frozen --group dev python -m amt.validation --qwen
uv run --frozen --group dev python -m amt.evaluation --dataset-version v2 --examples 32 --seed 20260920
uv run --frozen --group dev python scripts/verify_pilot.py results/evaluation/YOUR_RUN
uv run --frozen --group dev python scripts/analyze_pilot_failures.py results/evaluation/YOUR_RUN
```

Normal CI does not download model weights; optional integration tests explicitly
skip. The fixed original-run diagnostic `diagnose_full_cache.py` additionally
requires the original local v1 run; it is not a clean-clone smoke command.

## Deliberately deferred

- Protected-first-token ablation, causal explanation of format failures.
- Baseline label/entity controls, longer contexts and held-out confirmation.
- Continuous-generation total-budget enforcement, evaluator timeouts, global
  hidden-copy detection and process peak-memory instrumentation.

Review and green CI are required before merge. No merge is performed by PR preparation.
