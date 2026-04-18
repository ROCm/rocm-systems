"""Tests for kernelbench_runner.run + parse."""

import csv
from io import StringIO
from pathlib import Path

import pytest

from tests.benchmarks.kernelbench_runner import parse_kernelbench_output, RunResult


FIXTURE = Path(__file__).parent / "fixtures" / "kernelbench_sample.csv"


def test_parse_returns_run_results():
    raw = FIXTURE.read_text() if FIXTURE.exists() else _sample_raw()
    results = parse_kernelbench_output(raw)
    assert all(isinstance(r, RunResult) for r in results)
    assert len(results) >= 2


def test_parse_extracts_kernel_id_baseline_optimized_ns():
    raw = _sample_raw()
    results = parse_kernelbench_output(raw)
    r = results[0]
    assert r.kernel_id
    assert r.baseline_ns > 0
    assert r.optimized_ns > 0
    assert isinstance(r.pr_applied, bool)


def _sample_raw() -> str:
    output = StringIO()
    w = csv.DictWriter(output, fieldnames=["kernel", "baseline_ns", "optimized_ns", "perfxpert_recommended"])
    w.writeheader()
    w.writerow({"kernel": "gemm_rocblas", "baseline_ns": "1000000", "optimized_ns": "750000", "perfxpert_recommended": "True"})
    w.writerow({"kernel": "softmax_fused", "baseline_ns": "500000", "optimized_ns": "500000", "perfxpert_recommended": "False"})
    return output.getvalue()
