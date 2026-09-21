"""The reporting runner must preserve failures rather than claim a false pass."""

import json
import os
import sys
from pathlib import Path

from amt.validation import pytest_summary, run_check, write_report


def test_runner_preserves_both_streams_and_failure_code(tmp_path: Path) -> None:
    result = run_check(
        "failure",
        [
            sys.executable,
            "-c",
            "import sys; print('input=3'); print('error=bad', file=sys.stderr); sys.exit(7)",
        ],
        tmp_path,
        tmp_path,
        dict(os.environ),
        10,
    )
    assert result["status"] == "failed"
    assert result["exit_code"] == 7
    log = (tmp_path / result["log"]).read_text()
    assert "input=3" in log and "error=bad" in log
    metadata = {"started_utc": "test", "expected_checks": 1}
    assert not write_report(tmp_path, metadata, [result])
    assert json.loads((tmp_path / "results.json").read_text())["success"] is False


def test_runner_records_timeout(tmp_path: Path) -> None:
    result = run_check(
        "timeout",
        [sys.executable, "-c", "import time; time.sleep(60)"],
        tmp_path,
        tmp_path,
        dict(os.environ),
        0.1,
    )
    assert result["status"] == "timeout"
    assert result["exit_code"] == 124
    assert "timeout" in (tmp_path / "timeout.log").read_text()


def test_incomplete_report_is_not_a_pass(tmp_path: Path) -> None:
    result = {"name": "one", "status": "passed", "duration_seconds": 0, "log": "one.log"}
    metadata = {"started_utc": "test", "expected_checks": 2}
    assert not write_report(tmp_path, metadata, [result])
    assert "INCOMPLETE / FAILED" in (tmp_path / "report.md").read_text()
    metadata["expected_checks"] = 1
    assert write_report(tmp_path, metadata, [result])


def test_junit_counts_include_skips_errors_and_failures(tmp_path: Path) -> None:
    path = tmp_path / "pytest.xml"
    assert pytest_summary(path) is None
    path.write_text(
        '<testsuites><testsuite tests="8" failures="1" errors="1" skipped="2"/></testsuites>'
    )
    assert pytest_summary(path) == {
        "tests": 8,
        "passed": 4,
        "failures": 1,
        "errors": 1,
        "skipped": 2,
    }
    path.write_text("<broken")
    assert pytest_summary(path) is None
