# Development pilot v2 — declared before model results

Date: 2026-09-19. This is development data, not held-out confirmation.

Command: `uv run --frozen --group dev python -m amt.evaluation --dataset-version v2 --examples 32 --seed 20260920`

- Complete crossed design: eight target fact positions × four gold labels,
  one example per cell, shuffled with the declared seed. Random distractor
  colors and shuffled options; no rejection/resampling based on model answers.
- New IDs start `dev-v2-`. Original v1 artifacts and generator remain available.
- Same eight locker names in fixed order: entity identity is still linked to
  depth. This pilot removes label/depth association, not every possible bias.
- Pinned Qwen2.5-1.5B-Instruct revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; CPU float32/eager; local files only.
- Render the tokenizer's chat template for one user message containing facts
  then the question. Use its assistant generation prompt; no custom system text.
- Split tokens immediately after context text, before the unseen question.
  Assert that tokenizing the prefix is exactly a prefix of the full rendered
  token sequence. Save rendered text, prefix text and both token chunks.
- All five policies use that same input. Full cache once per example; compressed
  context budgets floor(0.5 × prefix length) and floor(0.25 × prefix length).
  Chat prefix/header tokens count in that budget. Query/footer tokens are extra
  at scoring time. Physical K/V storage audits apply before and after append.
- Adaptive weights fixed 0.5/0.5, no sink/local protections. Attention-only and
  adaptive signals come from the last context token, not the future question.
- Primary score: highest next-token logit among bare A/B/C/D (single-token IDs
  checked at runtime); ties take the first label. Save unrestricted greedy token
  as a diagnostic. No free-text-generation or sequence-probability claim.
- Validate every full-cache example against ordinary uncompressed inference at
  atol=2e-4, rtol=1e-5. Stop and preserve partial results on validation failure.
  Dense-reference check time is excluded from the recorded policy timing.
- 32 examples × (1 + 4 × 2) = 288 rows. Require all unique cells, 32 per group,
  correct aggregate recomputation, dataset hash and physical budget audits.
- Keep full-cache errors in overall accuracy and separately report the
  full-cache-correct subset. No data-dependent exclusions or tuning in this run.
- Instrumented fixed-order timing is not a performance benchmark. Logical KV
  bytes are not peak RAM. No significance or superiority claim from this pilot.
- Dataset and prompt BOTH change from v1: differences from old percentages
  cannot be attributed solely to either change. Compare policies within v2 only.

Next decision: inspect full-cache competence and paired policy errors before
changing settings. Longer contexts, variable entity/depth assignment, held-out
data and continuous generation-budget enforcement remain separate work.
