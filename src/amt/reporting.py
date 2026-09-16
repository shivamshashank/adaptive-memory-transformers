"""Aggregate baseline results and plot KV-cache memory scaling."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Protocol, Sequence, TypedDict, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class _FigureProtocol(Protocol):
    def tight_layout(self) -> None: ...
    def savefig(self, fname: str | Path, *, dpi: int) -> None: ...


class _AxesProtocol(Protocol):
    def plot(
        self,
        x: Sequence[int] | Sequence[float],
        y: Sequence[float],
        *,
        marker: str,
        linewidth: int,
    ) -> object: ...

    def set_title(self, title: str) -> object: ...
    def set_xlabel(self, xlabel: str) -> object: ...
    def set_ylabel(self, ylabel: str) -> object: ...
    def grid(self, visible: bool, *, linestyle: str, alpha: float) -> None: ...


class BaselineRecord(TypedDict):
    prompt_tokens: int
    actual_kv_cache_seq_len: int
    estimated_kv_cache_bytes: int
    actual_kv_cache_bytes: int
    prefill_time_s: float
    decode_time_s: float
    decode_tokens_per_second: float
    seed: int


class SummaryStats(TypedDict):
    mean: float
    std: float
    min: float
    max: float
    n: int


class SummaryRow(TypedDict):
    prompt_tokens: int
    final_sequence_length: SummaryStats
    estimated_kv_cache_bytes: SummaryStats
    actual_kv_cache_bytes: SummaryStats
    prefill_time_s: SummaryStats
    decode_time_s: SummaryStats
    decode_tokens_per_second: SummaryStats
    seed_values: list[int]


class Summary(TypedDict):
    num_runs: int
    prompt_lengths: list[int]
    rows: list[SummaryRow]


def _load_result(path: Path) -> BaselineRecord:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return cast(BaselineRecord, loaded)


def _stats(values: list[float]) -> SummaryStats:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}

    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "n": len(values),
    }


def summarize_records(results: list[BaselineRecord]) -> Summary:
    """Group baseline records by prompt length and compute descriptive statistics."""
    by_prompt: dict[int, list[BaselineRecord]] = {}
    for result in results:
        prompt_tokens = int(result["prompt_tokens"])
        by_prompt.setdefault(prompt_tokens, []).append(result)

    rows: list[SummaryRow] = []
    for prompt_tokens in sorted(by_prompt):
        runs: list[BaselineRecord] = by_prompt[prompt_tokens]
        row: SummaryRow = {
            "prompt_tokens": prompt_tokens,
            "final_sequence_length": _stats(
                [float(run["actual_kv_cache_seq_len"]) for run in runs]
            ),
            "estimated_kv_cache_bytes": _stats(
                [float(run["estimated_kv_cache_bytes"]) for run in runs]
            ),
            "actual_kv_cache_bytes": _stats([float(run["actual_kv_cache_bytes"]) for run in runs]),
            "prefill_time_s": _stats([float(run["prefill_time_s"]) for run in runs]),
            "decode_time_s": _stats([float(run["decode_time_s"]) for run in runs]),
            "decode_tokens_per_second": _stats(
                [float(run["decode_tokens_per_second"]) for run in runs]
            ),
            "seed_values": [int(run["seed"]) for run in runs],
        }
        rows.append(row)

    return {
        "num_runs": len(results),
        "prompt_lengths": [row["prompt_tokens"] for row in rows],
        "rows": rows,
    }


def _plot_series(
    x_values: list[int],
    y_values: list[float],
    ylabel: str,
    filename: str,
    title: str,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    typed_fig = cast(_FigureProtocol, fig)
    typed_ax = cast(_AxesProtocol, ax)

    typed_ax.plot(x_values, y_values, marker="o", linewidth=2)
    typed_ax.set_title(title)
    typed_ax.set_xlabel("Final sequence length")
    typed_ax.set_ylabel(ylabel)
    typed_ax.grid(True, linestyle="--", alpha=0.4)
    typed_fig.tight_layout()
    typed_fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def _write_markdown_table(summary: Summary, output_path: Path) -> None:
    lines = [
        "# Memory scaling summary",
        "",
        "| prompt_tokens | final_seq_len_mean | final_seq_len_std | estimated_kv_cache_bytes_mean | actual_kv_cache_bytes_mean | prefill_time_s_mean | decode_tokens_per_second_mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in summary["rows"]:
        seq = row["final_sequence_length"]
        est = row["estimated_kv_cache_bytes"]
        actual = row["actual_kv_cache_bytes"]
        prefill = row["prefill_time_s"]
        throughput = row["decode_tokens_per_second"]
        lines.append(
            f"| {row['prompt_tokens']} | {seq['mean']:.1f} | {seq['std']:.1f} | {est['mean']:.1f} | {actual['mean']:.1f} | {prefill['mean']:.6f} | {throughput['mean']:.2f} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_results(input_dir: Path, output_dir: Path | None = None) -> Summary:
    """Aggregate baseline JSON records and generate tables and plots."""
    output_dir = output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        _load_result(path)
        for path in sorted(input_dir.glob("*.json"))
        if path.is_file() and path.name != "summary.json"
    ]
    if not results:
        raise FileNotFoundError(f"No JSON result files found in {input_dir}")

    summary = summarize_records(results)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    _write_markdown_table(summary, output_dir / "summary.md")

    seq_lengths = [int(row["final_sequence_length"]["mean"]) for row in summary["rows"]]
    kv_means = [float(row["actual_kv_cache_bytes"]["mean"]) for row in summary["rows"]]
    prefill_means = [float(row["prefill_time_s"]["mean"]) for row in summary["rows"]]
    decode_throughput = [float(row["decode_tokens_per_second"]["mean"]) for row in summary["rows"]]

    _plot_series(
        seq_lengths,
        kv_means,
        "Actual KV bytes",
        "kv_bytes_vs_sequence.png",
        "KV cache memory vs sequence length",
        output_dir,
    )
    _plot_series(
        seq_lengths,
        prefill_means,
        "Prefill time (s)",
        "prefill_time_vs_sequence.png",
        "Prefill time vs sequence length",
        output_dir,
    )
    _plot_series(
        seq_lengths,
        decode_throughput,
        "Decode throughput (tokens/s)",
        "decode_throughput_vs_sequence.png",
        "Decode throughput vs sequence length",
        output_dir,
    )

    return summary
