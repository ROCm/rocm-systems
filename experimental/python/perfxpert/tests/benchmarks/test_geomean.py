"""Tests for weighted-geomean computation + 1.2× gate."""

import math

import pytest

from tests.benchmarks.geomean import (
    weighted_geomean,
    assert_geomean_above,
    filter_recommended,
)
from tests.benchmarks.tritonbench_runner import RunResult


def _mk(kernel, base_ns, opt_ns, pr=True):
    return RunResult(kernel, base_ns, opt_ns, pr)


def test_weighted_geomean_unweighted_baseline():
    # When all have equal baseline, geomean = geometric mean of speedups
    results = [
        _mk("k1", 100, 100),   # 1.0× speedup
        _mk("k2", 100, 50),    # 2.0× speedup
        _mk("k3", 100, 25),    # 4.0× speedup
    ]
    g = weighted_geomean(results)
    # geomean(1.0, 2.0, 4.0) = (1*2*4)^(1/3) = 8^(1/3) = 2.0
    assert math.isclose(g, 2.0, rel_tol=0.01)


def test_filter_recommended_drops_no_op_entries():
    results = [
        _mk("k1", 100, 70, pr=True),
        _mk("k2", 200, 200, pr=False),
        _mk("k3", 500, 250, pr=True),
    ]
    filtered = filter_recommended(results)
    assert [r.kernel_id for r in filtered] == ["k1", "k3"]


def test_assert_geomean_above_passes_when_fast():
    results = [_mk("k1", 200, 100)]  # 2.0× speedup
    assert_geomean_above(results, threshold=1.2)  # no exception


def test_assert_geomean_above_raises_when_slow():
    results = [_mk("k1", 105, 100)]  # 1.05× speedup
    with pytest.raises(AssertionError) as e:
        assert_geomean_above(results, threshold=1.2)
    assert "1.05" in str(e.value) or "< 1.20" in str(e.value)


def test_weighted_geomean_weights_by_baseline_ns():
    # Large baseline dominates geomean — validates we're weighting correctly
    results = [
        _mk("small",    100, 50),       # 2.0× but tiny weight
        _mk("dominant", 100_000, 100_000),  # 1.0× but huge weight
    ]
    g = weighted_geomean(results)
    assert 1.0 <= g <= 1.05  # big baseline dominates → geomean near 1.0
