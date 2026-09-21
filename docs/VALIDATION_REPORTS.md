# Run tests and save the complete validation evidence

From the repository root, run:

```bash
uv run --frozen --group dev python -m amt.validation
```

This checks formatting, lint, types and the full local test suite, then runs all
five cache policies on a deterministic random tiny Qwen. It does not download
models. A tiny random model has no meaningful language ability; its outputs
demonstrate the mechanics only.

For the already downloaded Qwen2.5-1.5B model:

```bash
uv run --frozen --group dev python -m amt.validation --qwen
```

The Qwen option enables the pinned-Qwen integration tests and uses the real
tokenizer/model for the demonstration. It uses CPU float32 and eager attention,
with the cached immutable revision. It does not use paid GPU services or permit
model downloads. Missing cached files produce a logged failure. The independent
download-dependent tiny GPT-2 test remains skipped in both modes.

Optional parameters:

```bash
uv run --frozen --group dev python -m amt.validation --qwen \
  --prompt "The capital of France is" \
  --budget 4 --max-new-tokens 6 --timeout 1200
```

`--timeout` is the maximum seconds for EACH check, not the whole run. The default
is 900 seconds. Keep demonstration prompts and generation lengths small: the
current eager-attention correctness runner is not a large-context benchmark.
The custom prompt is used only in Qwen mode; tiny mode uses fixed synthetic IDs.

Each invocation creates a new folder under `results/validation/`, identified by
UTC time and a unique suffix. The terminal prints its exact location. Reports
are ignored by Git; earlier runs are never overwritten.

| File | Contents |
| --- | --- |
| `report.md` | Readable check results and per-policy input/output summaries |
| `results.json` | Commands, run settings, package versions, durations, exit codes and test counts |
| `all_logs.txt` | All completed check logs combined into one text file |
| `pytest.log` | Verbose test outcomes, captured stdout/stderr and failure tracebacks/local variables |
| `pytest.xml` | Machine-readable per-test results and captured test output |
| `format.log`, `lint.log`, `types.log` | Complete code-quality check output |
| `git_revision.log`, `git_status.log` | Git revision and dirty/untracked state |
| `source_snapshot.zip` | Actual source, tests, configs, protocol and documentation, including uncommitted code |
| `demo.log` | Model loading output and generated token/retention summaries |
| `demo_inputs.json` | Exact model config, input IDs, prompt, seed, backend, dtype and budget |
| `demo_outputs.json` | Per-step inputs, generated IDs/text, attention-policy selection signals, selected slots, retained positions, cache shapes/bytes and top-five logits |
| `demo_logits.pt` | All vocabulary logits for every demonstration policy and step |
| `policy_fixture.json` | A controlled scoring example and its weight-dependent outputs |

Logs and reports are updated as checks finish. A failed check does not prevent
later independent checks from running. Timeout or interruption terminates the
check's process group; the runner preserves partial evidence and exits nonzero.
An interrupted run stops subsequent checks. Files not yet produced by a failed
or interrupted check may be absent.

The runner returns exit code 0 only when all scheduled checks pass. Skipped
pytest cases are counted explicitly and do not mean those tests were validated.

## Reading full logits

```python
import torch

logits = torch.load("results/validation/YOUR_RUN/demo_logits.pt", weights_only=True)
print(logits["adaptive"].shape)  # [generation_steps, batch_size=1, vocabulary_size]
```

The JSON contains raw selection inputs for attention/adaptive policies. Fixed
policies do not consume those signals, so their selection field is null.
Unit-test fixture definitions are saved in the source snapshot; successful
pytest tests do not serialize every local tensor. Assertions and captured output
remain in the logs/XML. The explicit model demonstration saves its complete
input/output evidence and full logits separately.

Validation durations are not inference benchmarks. Retained KV bytes are not
peak RAM/GPU memory, and full cache deliberately ignores the compressed budget.
The report is evidence of software behavior, not evidence of superior answer
quality. Current floating-point limitations remain in `NUMERICAL_TOLERANCES.md`.
