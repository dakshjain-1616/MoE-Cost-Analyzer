"""Integration tests for analyze.py CLI edge cases."""

import os
import subprocess
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
ANALYZE = str(PROJECT_ROOT / "analyze.py")
EMPTY_BENCHMARK = str(PROJECT_ROOT / "tests" / "fixtures" / "empty_benchmark.json")


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, ANALYZE] + args,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )


def test_empty_benchmark_prints_no_tasks():
    result = _run([EMPTY_BENCHMARK, "--dry-run"])
    output = result.stdout + result.stderr
    assert "No tasks to evaluate." in output
    assert result.returncode == 0


def test_invalid_model_id_prints_error(tmp_path):
    # Write a one-task benchmark
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({
        "name": "test",
        "tasks": [{"id": "t1", "prompt": "hello", "expected_labels": ["Positive"]}]
    }))
    result = _run([str(bench), "--dense-model", "google/does-not-exist", "--dry-run"])
    output = result.stdout + result.stderr
    assert "Invalid model ID" in output
    assert result.returncode != 0


def test_dry_run_completes_successfully():
    bench = str(PROJECT_ROOT / "benchmark.json")
    result = _run([bench, "--dry-run", "--output", "/tmp/test_results.csv"])
    assert result.returncode == 0
    assert "Recommendation" in result.stdout or "USE MoE" in result.stdout or "DENSE" in result.stdout


def test_dry_run_creates_csv():
    output_path = "/tmp/dry_run_results.csv"
    bench = str(PROJECT_ROOT / "benchmark.json")
    result = _run([bench, "--dry-run", "--output", output_path])
    assert result.returncode == 0
    assert os.path.exists(output_path)
    # CSV should have header + 200 data rows (100 tasks × 2 models)
    lines = Path(output_path).read_text().strip().splitlines()
    assert len(lines) == 201  # 1 header + 200 rows
