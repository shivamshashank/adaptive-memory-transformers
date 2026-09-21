"""Run offline validation and preserve logs, source, model inputs and outputs.

Usage: python -m amt.validation [--qwen]
The default demonstration is a random tiny model, NOT a quality benchmark.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
import uuid
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_check(
    name: str,
    command: list[str],
    root: Path,
    output: Path,
    env: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    """Capture both output streams and terminate the process group on timeout."""
    log = output / f"{name}.log"
    started = time.monotonic()
    status = "failed"
    code = 127
    with log.open("w", encoding="utf-8") as handle:
        handle.write(f"Command: {json.dumps(command)}\nWorking directory: {root}\n\n")
        handle.flush()
        try:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
            try:
                code = process.wait(timeout=timeout)
                status = "passed" if code == 0 else "failed"
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except ProcessLookupError:
                    pass
                process.wait()
                status = "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "interrupted"
                code = 124 if status == "timeout" else 130
                handle.write(f"\nRunner stopped this check: {status}\n")
        except OSError:
            traceback.print_exc(file=handle)
    return {
        "name": name,
        "command": command,
        "status": status,
        "exit_code": code,
        "duration_seconds": round(time.monotonic() - started, 3),
        "log": log.name,
    }


def pytest_summary(path: Path) -> dict[str, int] | None:
    if not path.exists():
        return None
    try:
        root = ET.parse(path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        counts = {
            key: sum(int(s.get(key, "0")) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        counts["passed"] = counts["tests"] - sum(
            counts[k] for k in ("failures", "errors", "skipped")
        )
        return counts
    except (ET.ParseError, ValueError):
        return None


def snapshot_sources(root: Path, output: Path) -> None:
    """Snapshot project code/configuration, never credentials or model weights."""
    candidates = [root / name for name in ("pyproject.toml", "uv.lock", "README.md")]
    for folder in ("src", "tests", "scripts", "configs", "research", "docs"):
        candidates.extend(
            path
            for path in (root / folder).rglob("*")
            if path.suffix in (".py", ".toml", ".md") and "__pycache__" not in path.parts
        )
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(candidates)):
            if path.is_file() and not path.is_symlink():
                archive.write(path, path.relative_to(root))


def demonstrate(output: Path, qwen: bool, prompt: str, budget: int, new_tokens: int) -> None:
    """Exercise the actual compressed runner and persist its complete logit tensors."""
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    from amt.compressed import run_compressed_greedy_decode
    from amt.policies import AttentionRecencyPolicy, RetentionSignals, create_policy

    torch.manual_seed(19)
    tokenizer = None
    revision = None
    if qwen:
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        revision = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=True
        )
        if tokenizer is None:
            raise RuntimeError(f"Failed to load tokenizer for {model_id}")
        model = Qwen2ForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.float32,
            attn_implementation="eager",
            local_files_only=True,
        ).eval()
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    else:
        model_id = "random-tiny-qwen2-seed-19"
        config = Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            attention_dropout=0.0,
            eos_token_id=None,
        )
        config._attn_implementation = "eager"
        model = Qwen2ForCausalLM(config).eval()
        ids = torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29]])
    model.to("cpu")
    inputs = {
        "model": model_id,
        "revision": revision,
        "seed": 19,
        "device": "cpu",
        "dtype": "float32",
        "attention_backend": "eager",
        "model_config": model.config.to_dict(),
        "prompt_text": prompt if qwen else None,
        "input_ids": ids.tolist(),
        "budget_tokens": budget,
        "max_new_tokens": new_tokens,
        "note": "Correctness demonstration, not a quality or performance benchmark.",
    }
    write_json(output / "demo_inputs.json", inputs)
    policies: dict[str, Any] = {}
    tensors = {}
    for name in ("full", "recency", "uniform", "attention", "adaptive"):
        policy = create_policy(name)
        decisions: list[dict[str, Any]] = []

        class LoggedAttentionPolicy(AttentionRecencyPolicy):
            def select_indices(
                self,
                total_tokens: int,
                budget_tokens: int,
                *,
                recent_window: int | None = None,
                signals: RetentionSignals | None = None,
            ) -> list[int]:
                selected = super().select_indices(
                    total_tokens, budget_tokens, recent_window=recent_window, signals=signals
                )
                decisions.append(
                    {
                        "signals": asdict(signals) if signals is not None else None,
                        "selected_slot_indices": selected,
                    }
                )
                return selected

        if isinstance(policy, AttentionRecencyPolicy):
            policy = LoggedAttentionPolicy(**asdict(policy))
        trace = run_compressed_greedy_decode(model, ids, new_tokens, policy, budget)
        generated = trace.decode.sequences[0, ids.shape[1] :].tolist()
        steps = []
        for step, (logits, positions, cache) in enumerate(
            zip(
                trace.decode.step_logits,
                trace.retained_positions,
                trace.decode.cache_snapshots,
                strict=True,
            )
        ):
            values, indices = logits[0].topk(5)
            steps.append(
                {
                    "step": step + 1,
                    "processed_input_ids": ids[0].tolist() if step == 0 else [generated[step - 1]],
                    "selection": decisions[step] if decisions else None,
                    "generated_token_id": generated[step],
                    "retained_original_positions": positions,
                    "cache": asdict(cache),
                    "top5_token_ids": indices.tolist(),
                    "top5_raw_logits": values.tolist(),
                }
            )
        policies[name] = {
            "configuration": asdict(policy)
            if isinstance(policy, AttentionRecencyPolicy)
            else {"name": name},
            "sequence_ids": trace.decode.sequences.tolist(),
            "generated_ids": generated,
            "generated_text": tokenizer.decode(generated, skip_special_tokens=True)
            if tokenizer
            else None,
            "steps": steps,
        }
        tensors[name] = torch.stack(trace.decode.step_logits).cpu()
        # Save incremental evidence even if a later policy raises an error.
        write_json(output / "demo_outputs.json", policies)
        torch.save(tensors, output / "demo_logits.pt")
        print(f"{name}: generated={generated}; retained={trace.retained_positions}", flush=True)
    fixture = RetentionSignals((0, 3, 9, 20), ((9.0, 8.0, 1.0, 0.0),))
    write_json(
        output / "policy_fixture.json",
        {
            "inputs": asdict(fixture),
            "budget": 2,
            "attention_dominant_indices": AttentionRecencyPolicy(0.9, 0.1).select_indices(
                4, 2, signals=fixture
            ),
            "recency_dominant_indices": AttentionRecencyPolicy(0.1, 0.9).select_indices(
                4, 2, signals=fixture
            ),
        },
    )


def write_report(output: Path, metadata: dict[str, Any], checks: list[dict[str, Any]]) -> bool:
    success = len(checks) == metadata["expected_checks"] and all(
        check["status"] == "passed" for check in checks
    )
    summary = pytest_summary(output / "pytest.xml")
    with (output / "all_logs.txt").open("w", encoding="utf-8") as combined:
        for check in checks:
            combined.write(f"\n===== {check['name']} ({check['status']}) =====\n")
            log_path = output / check["log"]
            if log_path.exists():
                combined.write(log_path.read_text(encoding="utf-8", errors="replace"))
    write_json(
        output / "results.json",
        {"metadata": metadata, "success": success, "pytest": summary, "checks": checks},
    )
    lines = [
        "# Local validation report",
        "",
        f"Run: {metadata['started_utc']}",
        f"Status: {'PASS' if success else 'INCOMPLETE / FAILED'}",
        "",
        "CPU float32; network downloads disabled. No paid compute.",
        "",
        "## Check results",
        "",
        "| Check | Status | Seconds | Full log |",
        "| --- | --- | ---: | --- |",
    ]
    lines.extend(
        f"| {c['name']} | {c['status']} | {c['duration_seconds']} | [{c['log']}]({c['log']}) |"
        for c in checks
    )
    lines += [
        "",
        f"Pytest counts: `{json.dumps(summary)}`",
        "",
        "[All check logs in one file](all_logs.txt)",
        "",
        "Test-by-test outcomes, stdout/stderr and assertion diagnostics are in pytest.log and pytest.xml.",
        "Test fixture inputs are preserved in source_snapshot.zip. Successful tests do not dump every local variable.",
        "",
        "## Model inputs and outputs",
        "",
    ]
    input_file, output_file = output / "demo_inputs.json", output / "demo_outputs.json"
    if input_file.exists():
        inputs = json.loads(input_file.read_text())
        lines += [
            f"Model: `{inputs['model']}`",
            f"Input IDs: `{inputs['input_ids']}`",
            f"Budget: {inputs['budget_tokens']}; new tokens: {inputs['max_new_tokens']}",
            "",
        ]
        if inputs["prompt_text"] is not None:
            lines += ["Prompt:", "", "```text", inputs["prompt_text"], "```", ""]
    if output_file.exists():
        for name, result in json.loads(output_file.read_text()).items():
            lines += [f"### {name}", "", f"Generated IDs: `{result['generated_ids']}`", ""]
            if result["generated_text"] is not None:
                lines += ["```text", result["generated_text"], "```", ""]
            lines += [
                "| Step | Token ID | Retained positions | KV bytes |",
                "| ---: | ---: | --- | ---: |",
            ]
            for step in result["steps"]:
                lines.append(
                    f"| {step['step']} | {step['generated_token_id']} | "
                    f"{step['retained_original_positions']} | {step['cache']['total_bytes']} |"
                )
            lines.append("")
    lines += [
        "## Files and interpretation",
        "",
        "- demo_inputs.json: exact model configuration, prompt/token IDs and run parameters.",
        "- demo_outputs.json: sequences, top-5 raw logits, retained positions and per-layer cache shapes/bytes.",
        "- demo_logits.pt: every vocabulary logit for every demonstration step and policy.",
        "- policy_fixture.json: controlled scoring inputs and two weight-dependent selections.",
        "- results.json: metadata, commands, timings, exit codes and test totals.",
        "- source_snapshot.zip: source, tests, configuration, protocol and documentation at run time.",
        "",
        "A random tiny model demonstrates mechanics, not language quality. Full retention ignores the",
        "budget intentionally. KV bytes exclude model weights and transient allocations. Check timings",
        "are validation durations, not inference benchmarks. Passing does not establish adaptive superiority.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return success


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwen", action="store_true", help="Use cached Qwen 1.5B and enable its integration tests"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/validation"))
    parser.add_argument("--timeout", type=float, default=900, help="Seconds allowed per check")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--demo-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (
        not math.isfinite(args.timeout)
        or args.timeout <= 0
        or args.budget < 1
        or args.max_new_tokens < 1
    ):
        parser.error("timeout, budget and max-new-tokens must be positive")
    root = Path(__file__).resolve().parents[2]
    if args.demo_only:
        demonstrate(args.output_dir, args.qwen, args.prompt, args.budget, args.max_new_tokens)
        return 0
    started = datetime.now(timezone.utc).isoformat()
    output = args.output_dir.resolve() / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    output.mkdir(parents=True, exist_ok=False)
    print(f"Writing validation evidence to {output}", flush=True)
    env = dict(os.environ)
    env.update(
        {
            "HF_HOME": str(root / ".model-cache/huggingface"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "AMT_RUN_MODEL_INTEGRATION": "0",
            "AMT_RUN_QWEN_INTEGRATION": "1" if args.qwen else "0",
            "AMT_MODEL_DEVICE": "cpu",
            "AMT_MODEL_DTYPE": "float32",
            "AMT_MODEL_ATTENTION": "eager",
            "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": "19",
            "PYTEST_ADDOPTS": "",
        }
    )
    metadata: dict[str, Any] = {
        "started_utc": started,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "mode": "qwen" if args.qwen else "tiny",
        "arguments": vars(args) | {"output_dir": str(output)},
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "pytest", "ruff", "mypy")
        },
    }
    snapshot_sources(root, output)
    demo = [
        sys.executable,
        "-m",
        "amt.validation",
        "--demo-only",
        "--output-dir",
        str(output),
        "--prompt",
        args.prompt,
        "--budget",
        str(args.budget),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.qwen:
        demo.append("--qwen")
    commands = [
        ("git_revision", ["git", "rev-parse", "HEAD"]),
        ("git_status", ["git", "status", "--short", "--branch"]),
        ("format", [sys.executable, "-m", "ruff", "format", "--check", "src", "tests", "scripts"]),
        ("lint", [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"]),
        ("types", [sys.executable, "-m", "mypy"]),
        (
            "pytest",
            [
                sys.executable,
                "-m",
                "pytest",
                "-vv",
                "-rA",
                "--showlocals",
                f"--junitxml={output / 'pytest.xml'}",
                "-o",
                "junit_logging=all",
                "-o",
                "junit_log_passing_tests=true",
            ],
        ),
        ("demo", demo),
    ]
    metadata["expected_checks"] = len(commands)
    checks: list[dict[str, Any]] = []
    write_report(output, metadata, checks)
    for name, command in commands:
        print(f"Running {name}...", flush=True)
        result = run_check(name, command, root, output, env, args.timeout)
        checks.append(result)
        write_report(output, metadata, checks)
        print(f"  {result['status']} ({result['duration_seconds']}s)", flush=True)
        if result["status"] == "interrupted":
            break
    success = write_report(output, metadata, checks)
    print(f"{'PASS' if success else 'INCOMPLETE / FAILED'}: {output / 'report.md'}", flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
