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
    assert r.optimized_ns > 0
    assert isinstance(r.pr_applied, bool)


def _sample_raw() -> str:
    return json.dumps({
        "suite": "tritonbench-rocm",
        "version": "0.2.0",
        "results": [
            {"kernel": "matmul_f16_256x256",    "baseline_ns": 1_000_000, "optimized_ns":   700_000, "perfxpert_recommended": True},
            {"kernel": "softmax_f32_1024",      "baseline_ns":   500_000, "optimized_ns":   300_000, "perfxpert_recommended": True},
            {"kernel": "layernorm_f16_4096",    "baseline_ns":   800_000, "optimized_ns":   800_000, "perfxpert_recommended": False},
        ],
    })
