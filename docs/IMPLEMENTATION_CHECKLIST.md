# Implementation Checklist

## Completed in this change

- [x] Correct model identity in attention-importance outputs.
- [x] Synchronize CUDA around the baseline generation timing interval.
- [x] Validate cache-pruning indices and support Transformers `DynamicCache`.
- [x] Make layer-attention evaluation place its model and inputs on the same device.
- [x] Account for global policy capacity across every layer when comparing it with layer-aware allocation.
- [x] Add a matched-policy compressed-decoding evaluator with next-token loss, exact-answer accuracy, cache-byte, latency, and throughput measurements.
- [x] Implement synthetic Passkey Retrieval (Needle-in-a-Haystack) benchmark evaluating full, recency, uniform, and adaptive policies on real causal language models.
- [x] Add focused tests for cache compatibility, evaluator metrics, passkey needle retention accounting, and model identity.

## Required before research claims

- [ ] Run the evaluator on a fixed, versioned prompt suite with multiple seeds.
- [ ] Add paired confidence intervals and paired significance tests to saved per-example results.
- [ ] Add feasible short/medium/long context-length configurations and record hardware limits.
- [ ] Compare global and layer-aware allocation in the real decode protocol.
- [ ] Generate figures only from saved evaluator JSON and commit the exact configurations.

## Interpretation guardrails

- [ ] Keep retained-attention-mass experiments labelled as a proxy, never task accuracy.
- [ ] Do not claim a memory or latency improvement unless selection and pruning overhead are included.
- [ ] Preserve policy failures and incomplete runs in raw artifacts.
