import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from amt.cache import Qwen2CacheAdapter
from amt.evaluation import audit_cache, main, paired_changes
from amt.policies import (
    FirstTokenProtectedPolicy,
    RetentionSignals,
    create_policy,
    policy_requires_attention,
)


def test_independent_verifier_rejects_wrong_swap():
    import runpy
    from pathlib import Path

    verify = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/verify_pilot.py"))[
        "verify_prefix_pairs"
    ]
    off = dict(
        example_id="a",
        policy="recency",
        ratio=0.5,
        correct=False,
        raw_greedy_text="The",
        retained_positions=[4, 5],
    )
    on = off | dict(policy="recency+first", raw_greedy_text="A", retained_positions=[0, 5])
    pair = dict(
        example_id="a",
        policy="recency",
        ratio=0.5,
        accuracy_delta=0,
        validity_delta=1,
        selection_unchanged=False,
    )
    verify([off, on], [pair])
    with pytest.raises(AssertionError):
        verify([off, on | dict(retained_positions=[0, 4])], [pair])
    with pytest.raises(AssertionError):
        verify([off, on], [pair | dict(validity_delta=0)])


@pytest.mark.parametrize("name", ["recency", "uniform", "attention", "adaptive"])
@pytest.mark.parametrize("budget", [0, 1, 3, 8, 20])
def test_shared_minimal_swap_and_disabled_equivalence(name, budget):
    signals = RetentionSignals(
        (0, 3, 5, 9, 12, 15, 18, 22), ((8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0),)
    )
    base = create_policy(name)
    original = base.select_indices(8, budget, signals=signals)
    assert (
        FirstTokenProtectedPolicy(base, False).select_indices(8, budget, signals=signals)
        == original
    )
    protected = FirstTokenProtectedPolicy(base).select_indices(8, budget, signals=signals)
    expected = original if not original or 0 in original else sorted([0] + original[1:])
    assert protected == expected
    assert len(protected) == min(8, budget)
    assert policy_requires_attention(FirstTokenProtectedPolicy(base)) == (
        name in ("attention", "adaptive")
    )


def test_rejects_resurrection_and_invalid_alignment():
    wrapper = FirstTokenProtectedPolicy(create_policy("recency"))
    for positions in ((1, 3, 5), (0, 0, 1), (0, -1, 4), (False, 2, 4)):
        with pytest.raises(ValueError):
            wrapper.select_indices(3, 2, signals=RetentionSignals(positions))
    with pytest.raises(ValueError):
        wrapper.select_indices(3, 2)
    with pytest.raises(ValueError):
        FirstTokenProtectedPolicy(create_policy("full")).select_indices(3, 2)
    with pytest.raises(ValueError, match="capacity"):
        wrapper.select_indices(5, 4, recent_window=1, signals=RetentionSignals(tuple(range(5))))


def test_frozen_cli_rejects_wrong_settings():
    with pytest.raises(SystemExit) as exc:
        main(["--prefix-ablation"])
    assert exc.value.code == 2


def test_pairing_is_by_identity_and_keeps_regressions():
    a = dict(
        example_id="a",
        policy="recency",
        ratio=0.5,
        correct=True,
        raw_greedy_text="The",
        retained_positions=[2, 3],
    )
    b = a | dict(
        policy="recency+first", correct=False, raw_greedy_text=" B", retained_positions=[0, 3]
    )
    pairs = paired_changes([b, a])
    assert pairs == [
        dict(
            example_id="a",
            policy="recency",
            ratio=0.5,
            validity_delta=1,
            accuracy_delta=-1,
            selection_unchanged=False,
        )
    ]
    assert paired_changes([b]) == []


@torch.inference_mode()
def test_tiny_qwen_protected_cache_query_positions_and_unchanged_controls():
    torch.manual_seed(19)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval()
    for name in ("recency", "uniform", "attention", "adaptive"):
        traces = []
        for enabled in (False, True):
            adapter = Qwen2CacheAdapter(model)
            policy = FirstTokenProtectedPolicy(create_policy(name), enabled)
            adapter.forward(
                torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29]]),
                collect_attention=policy_requires_attention(policy),
            )
            signals = RetentionSignals(adapter.positions, adapter.attention_by_layer)
            selected = policy.select_indices(8, 4, signals=signals)
            adapter.retain(selected)
            audit_cache(adapter, 4)
            if enabled:
                assert 0 in adapter.positions
            logits = adapter.forward(torch.tensor([[3, 7, 11]]))
            audit_cache(adapter, 7)
            assert adapter.positions[-3:] == (8, 9, 10)
            traces.append((selected, logits))
        if traces[0][0] == traces[1][0]:
            torch.testing.assert_close(traces[0][1], traces[1][1], atol=0, rtol=0)
