# Direct-color baseline calibration — 2026-09-21

Declared before model inference. This is development calibration, not held-out
confirmation and not a compression experiment.

## Frozen construction

- Start from all 32 v2 examples generated with seed 20260920. Preserve every
  fact, target position and target color. Remove the A–D options and ask:
  `What color is the NAME locker?` followed by
  `Answer with only the color word (red, blue, green, or yellow).\nAnswer:`.
- Run each fact set with the original entity vocabulary and with the fixed rename
  Aster…Hazel → Alice…Henry, replacing facts and query consistently.
  Total: 32 × 2 = 64 paired full-cache cases. Renaming changes tokenization.
- Same v2 chat template, pinned cached Qwen2.5-1.5B-Instruct revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, CPU float32/eager and seed 20260920.
  No downloads, prompt alternatives, rejection, resampling or answer-based edits.

## Frozen scorer and checks

Tokenizer inspection before this declaration found `red`, `blue`, `green` and
`yellow` are each one bare token; their leading-space forms are also each one
token. Primary constrained prediction is argmax over the four BARE color-token
logits because the model is at an assistant-message boundary. Also save leading-
space predictions, all eight token IDs/logits and the unrestricted raw next token.
Raw correctness requires `decode(argmax).strip()` to equal the gold color exactly.
No multi-token normalization, substring matching or post-outcome scorer changes.

Independently parse the requested fact and gold from text. Verify the chat
context/query token split. For every case compare staged full-cache and ordinary
dense full-vocabulary logits at atol=2e-4, rtol=1e-5, and audit physical cache
storage. Preserve partial/failure reports, exact inputs, rows, logs and sources.

## Gate and interpretation

Advance this task design only if EACH vocabulary has at least 29/32 primary
constrained correct AND 29/32 raw-next correct, all 64 dense checks pass, and all
64 raw next tokens are one of the four color words after stripping whitespace.
This practical development gate is not a significance test or a general model
claim. Report paired rename changes and bare/space prediction agreement.

If it passes, freeze a distinct modest-longer-context compression protocol using
full cache, protected recency and protected adaptive at equal budgets, then keep
fresh examples for confirmation. If it fails, retain the result and redesign the
task before compression. Do not tune adaptive weights on these examples.

Command: `uv run --frozen --group dev python scripts/calibrate_direct_color.py`
