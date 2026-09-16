"""Tests for deterministic aggregation of baseline measurements."""

from amt.reporting import BaselineRecord, summarize_records


def _record(prompt_tokens: int, seed: int, cache_bytes: int) -> BaselineRecord:
    return {
        "prompt_tokens": prompt_tokens,
        "actual_kv_cache_seq_len": prompt_tokens + 4,
        "estimated_kv_cache_bytes": cache_bytes,
        "actual_kv_cache_bytes": cache_bytes,
        "prefill_time_s": 1.0,
        "decode_time_s": 2.0,
        "decode_tokens_per_second": 2.0,
        "seed": seed,
    }


def test_summary_groups_runs_by_prompt_length() -> None:
    summary = summarize_records([_record(16, 1, 100), _record(16, 2, 200), _record(32, 1, 300)])

    assert summary["num_runs"] == 3
    assert summary["prompt_lengths"] == [16, 32]
    assert summary["rows"][0]["actual_kv_cache_bytes"]["mean"] == 150.0
