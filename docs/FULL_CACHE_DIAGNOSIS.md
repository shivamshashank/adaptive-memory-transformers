# Full-cache failure diagnosis — 2026-09-19

## Outcome

The four errors in the original pilot are reproducible with ordinary uncompressed
model inference. No gold-label, token-boundary or custom full-cache discrepancy
was found on these 20 examples. Prompt formatting contributes: the model's chat
template fixes two failures, but does not fix the other two.

This identifies an empirical model/prompt limitation on this dataset, not the
model's internal reason for making each mistake. It is not evidence of a general
retrieval limitation or a result about compressed policies under chat formatting.

## Evidence and procedure

- Original run: `results/evaluation/20260918T232554Z-e48814cb`.
- Diagnostic: `results/diagnostics/20260919T102630Z/diagnosis.json`.
- Reproducer: `uv run --frozen --group dev python scripts/diagnose_full_cache.py`.
- Same pinned Qwen model, CPU float32/eager, offline cached weights; no training.
- Checked ALL 20 development examples, not only previously failed ones.
- Independently parsed the requested locker and its color from the saved text;
  all gold labels agreed with the options.
- Verified separately tokenized context/query IDs equal tokenization of their
  concatenated text, and decoding reproduces that text exactly.
- Compared ordinary `use_cache=False` inference on the complete token sequence
  with the custom two-stage full-cache adapter. Full-vocabulary last-token
  logits matched exactly in this run (maximum absolute difference 0.0); the
  diagnostic acceptance tolerance was atol=2e-4, rtol=1e-5.
- Recomputed bare-letter candidate scores matched saved pilot scores exactly.
- Checked bare A-D IDs (32–35) and leading-space alternatives (362,425,356,422).
  Their predictions agreed on every original full-cache example: both 16/20.
- Wrapped the same context/question in the tokenizer's user-message chat
  template with an assistant generation prompt. Both candidate forms scored
  18/20. Chat raw greedy tokens were also the corresponding bare letters.

| Example suffix | Requested fact | Gold | Plain answer | Chat answer |
| --- | --- | --- | --- | --- |
| 002 | Cedar is yellow | C | B (red) | B (red) |
| 004 | Elm is yellow | A | D (blue) | A (yellow) |
| 009 | Birch is green | B | C (red) | B (green) |
| 012 | Elm is blue | A | D (red) | D (red) |

Original plain raw greedy tokens included a leading space (e.g. ` B`). The
scorer's bare-letter restriction is a convention that should be documented and
revisited, but it did not cause these four errors. This check covers full cache
only; do not assume all compressed predictions are invariant to token forms.

The initial diagnostic attempt stopped before producing results because the
installed tokenizer returned a dictionary-like chat encoding by default. The
reproducer explicitly requests `return_dict=False`; the subsequent run completed
all 20 examples. Original evaluator code and pilot artifacts were not changed.

## Next steps

- [x] Diagnose the four full-cache errors without deleting or rescoring old data.
- [ ] Build a separately versioned development dataset with independently
      randomized answer labels and fact locations.
- [ ] Declare prompt and scoring conventions before the next run. Chat formatting
      is a reasonable development candidate, not a silently substituted result.
- [ ] If adopting chat formatting, validate its delayed-query token boundary and
      rerun ALL policies at matched budgets; do not compare chat full cache with
      old plain compressed scores.
- [ ] Retain and analyze the two remaining failures; test further development
      examples before treating this as a reliable benchmark.
- [ ] Freeze settings before any held-out confirmation. None of these 20 examples
      is held out, and no significance or adaptive-advantage claim is supported.
