"""Tests for tritonbench_runner.run + parse."""

import json
from pathlib import Path

import pytest

from tests.benchmarks.tritonbench_runner import parse_tritonbench_output, RunResult


FIXTURE = Path(__file__).parent / "fixtures" / "tritonbench_sample.json"


def test_parse_returns_run_results():
    raw = FIXTURE.read_text() if FIXTURE.exists() else _sample_raw()
    results = parse_tritonbench_output(raw)
    assert all(isinstance(r, RunResult) for r in results)
    assert len(results) >= 3


def test_parse_extracts_kernel_id_baseline_optimized_ns():
    raw = _sample_raw()
    results = parse_tritonbench_output(raw)
    r = results[0]
    assert r.kernel_id
    assert r.baseline_ns > 0
    assert isinstance(r.analysis_succeeded, bool)
    assert isinstance(r.recommendation_count, int)


def _sample_raw() -> str:
    return json.dumps({
        "suite": "tritonbench-rocm",
        "mode": "analysis_only",
        "results": [
            {"kernel": "matmul_f16_256x256", "baseline_ns": 1_000_000, "analysis_succeeded": True, "recommendation_count": 2, "report_path": "/tmp/matmul.json"},
            {"kernel": "softmax_f32_1024", "baseline_ns": 500_000, "analysis_succeeded": True, "recommendation_count": 1, "report_path": "/tmp/softmax.json"},
            {"kernel": "layernorm_f16_4096", "baseline_ns": 800_000, "analysis_succeeded": False, "recommendation_count": 0, "report_path": None},
        ],
    })
