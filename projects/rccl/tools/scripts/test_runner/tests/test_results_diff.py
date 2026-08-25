#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""Host tests for the per-config results diff (init-pipeline correctness gate)."""

import os
import sys

_RUNNER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER_ROOT not in sys.path:
    sys.path.insert(0, _RUNNER_ROOT)

from lib.results_diff import diff_results, index_results  # noqa: E402


def _serial(suite, name, result):
    return {"suite": suite, "test_name": name, "result": result}


def _topline(suite, name, result):
    return {"suite": suite, "test_name": name, "result": result,
            "record_type": "parent_summary", "counts_toward_topline": True}


def _sub(suite, name, result):
    return {"suite": suite, "test_name": name, "name": name + ".x", "result": result,
            "record_type": "sub_entry", "counts_toward_topline": False}


def test_index_skips_sub_entries():
    recs = [_topline("S", "A", "PASSED"), _sub("S", "A", "FAILED")]
    idx = index_results(recs)
    assert idx == {("S", "A"): "PASSED"}  # sub_entry ignored, parent counted


def test_identical_runs_gate_pass():
    base = [_serial("S", "A", "PASSED"), _serial("S", "B", "SKIPPED")]
    cand = [_topline("S", "A", "PASSED"), _topline("S", "B", "SKIPPED")]
    d = diff_results(base, cand)
    assert d["gate_ok"] and d["matched"] == 2
    assert not d["regressions"] and not d["only_baseline"]


def test_regression_fails_gate():
    base = [_serial("S", "A", "PASSED")]
    cand = [_topline("S", "A", "FAILED")]
    d = diff_results(base, cand)
    assert not d["gate_ok"]
    assert d["regressions"] == [(("S", "A"), "PASSED", "FAILED")]


def test_fix_is_not_a_regression():
    base = [_serial("S", "A", "FAILED")]
    cand = [_topline("S", "A", "PASSED")]
    d = diff_results(base, cand)
    assert d["gate_ok"]
    assert d["fixes"] == [(("S", "A"), "FAILED", "PASSED")]


def test_dropped_coverage_fails_gate():
    base = [_serial("S", "A", "PASSED"), _serial("S", "B", "PASSED")]
    cand = [_topline("S", "A", "PASSED")]
    d = diff_results(base, cand)
    assert not d["gate_ok"]
    assert d["only_baseline"] == [(("S", "B"), "PASSED")]


def test_new_config_does_not_fail_gate():
    base = [_serial("S", "A", "PASSED")]
    cand = [_topline("S", "A", "PASSED"), _topline("S", "B", "PASSED")]
    d = diff_results(base, cand)
    assert d["gate_ok"]
    assert d["only_candidate"] == [(("S", "B"), "PASSED")]


def test_exclude_masks_known_preexisting():
    base = [_serial("S", "AllReduce_CuMem1", "SKIPPED"), _serial("S", "A", "PASSED")]
    cand = [_topline("S", "AllReduce_CuMem1", "FAILED"), _topline("S", "A", "PASSED")]
    # Without exclude the CuMem run is a regression; excluded, the gate passes.
    assert not diff_results(base, cand)["gate_ok"]
    d = diff_results(base, cand, exclude=["*_CuMem1"])
    assert d["gate_ok"] and d["matched"] == 1


def test_timeout_variants_are_failures():
    base = [_serial("S", "A", "PASSED")]
    for bad in ("TIMED_OUT", "TIMEOUT", "INFRA_ERROR", "CANCELLED"):
        d = diff_results(base, [_topline("S", "A", bad)])
        assert not d["gate_ok"], bad


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
