#!/usr/bin/env python3
"""
Unit tests for the test_runner parallel-execution policy (`--jobs` concurrency).

These are pure-Python, GPU-free tests of the decision logic that governs concurrent
execution: the co-tenancy admissibility policy (`_can_parallelize`), the per-entry GPU
footprint (`_gpu_demand`), the aggregate GPU budget semaphore (`GpuBudget`), the
`serial_only` configuration -> suite -> test inheritance/override, and the CLI
flag-combination validation (rerun-failed + jobs fail-fast).

Run with:  pytest projects/rccl/tools/scripts/test_runner/tests/test_parallel_policy.py
"""

import os
import sys
import threading
import time

import pytest

# The test_runner modules live one directory up. Add both the package root (so
# `import test_runner` resolves `from lib...`) and the lib dir (so test_executor's
# `from test_config import ...` fallback resolves) to the path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LIB = os.path.join(_ROOT, "lib")
for _p in (_ROOT, _LIB):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Aliased so pytest doesn't try to collect the imported "Test*" classes as test cases.
from test_executor import TestExecutor as _Executor, GpuBudget  # noqa: E402
from test_config import TestConfigProcessor as _ConfigProcessor  # noqa: E402
import test_runner  # noqa: E402


class _Args:
    """Minimal stand-in for the parsed argparse namespace."""

    def __init__(self, **kw):
        self.jobs = kw.get("jobs", 1)
        self.max_parallel_gpus = kw.get("max_parallel_gpus", None)
        self.rerun_failed = kw.get("rerun_failed", False)
        self.stop_on_rerun_failure = kw.get("stop_on_rerun_failure", False)


def _executor(max_parallel_gpus=8, detected_gpus=8):
    """A TestExecutor with just enough state for the pure policy methods (no real init).

    max_parallel_gpus=None models the flag being left at its default, which makes the
    budget resolve to the detected node size.
    """
    ex = _Executor.__new__(_Executor)
    ex._gpus_per_node = detected_gpus
    ex._gpus_per_node_detected = True
    ex.args = _Args(max_parallel_gpus=max_parallel_gpus)
    return ex


# --------------------------------------------------------------------------- #
# _gpu_demand: per-entry GPU footprint
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("test,expected", [
    # single-process gtest: footprint is num_gpus (enumerated in-process), NOT num_ranks
    ({"num_ranks": 1, "num_gpus": 8}, 8),   # all-GPU fixtures suite
    ({"num_ranks": 1, "num_gpus": 3}, 3),   # asymmetric-visibility style
    ({"num_ranks": 1, "num_gpus": 1}, 1),   # single-GPU gtest
    # single-node MPI: footprint is num_ranks (one GPU/rank); num_gpus is node capacity
    ({"num_ranks": 2, "num_gpus": 8}, 2),
    ({"num_ranks": 8, "num_gpus": 8}, 8),
    # "auto" ranks on a single node resolve to num_gpus * nodes
    ({"num_ranks": "auto", "num_gpus": 8}, 8),
    # multi-node: per-node footprint is num_gpus
    ({"num_ranks": 16, "num_gpus": 8, "num_nodes": 2}, 8),
    # unparseable rank value falls back to node size, never crashes
    ({"num_ranks": "weird", "num_gpus": 8}, 8),
])
def test_gpu_demand(test, expected):
    assert _executor()._gpu_demand(test, {}) == expected


def test_gpu_demand_inherits_from_suite_config():
    ex = _executor()
    # num_gpus provided only at the suite/config level is inherited by the entry.
    assert ex._gpu_demand({"num_ranks": 1}, {"num_gpus": 4}) == 4


def test_gpu_demand_auto_uses_detected_node_size():
    ex = _executor(detected_gpus=6)
    assert ex._gpu_demand({"num_ranks": 1, "num_gpus": "auto"}, {}) == 6


# --------------------------------------------------------------------------- #
# _can_parallelize: co-tenancy admissibility
# --------------------------------------------------------------------------- #

def test_serial_only_never_co_tenants():
    ex = _executor()
    assert ex._can_parallelize({"num_ranks": 2, "serial_only": True}, {}) is False


def test_serial_only_false_overrides_inherited_true():
    # Regression for the config-block OR bug: a test-level serial_only:false must win.
    ex = _executor()
    assert ex._can_parallelize({"num_ranks": 2, "serial_only": False}, {}) is True


def test_entry_larger_than_budget_runs_serially():
    ex = _executor(max_parallel_gpus=4)
    # 8-rank entry cannot fit a 4-GPU co-tenancy budget -> not eligible for the pool.
    assert ex._can_parallelize({"num_ranks": 8, "num_gpus": 8}, {}) is False


def test_entry_equal_to_budget_is_eligible_but_runs_alone():
    # demand == budget is admitted to the pool; the GpuBudget then serialises it by
    # forcing it to hold all permits. Eligibility (True) is correct here.
    ex = _executor(max_parallel_gpus=8)
    assert ex._can_parallelize({"num_ranks": 1, "num_gpus": 8}, {}) is True


def test_small_mpi_entry_co_tenants():
    ex = _executor()
    assert ex._can_parallelize({"num_ranks": 2, "num_gpus": 8}, {}) is True


# --------------------------------------------------------------------------- #
# GpuBudget: aggregate GPU semaphore
# --------------------------------------------------------------------------- #

def test_budget_bounds_concurrent_sum():
    """Two 3-permit holders fit an 8-budget; a third 3-permit acquire must block until
    one releases, proving the aggregate sum (not per-entry) is bounded."""
    b = GpuBudget(8)
    b.acquire(3)
    b.acquire(3)  # 6/8 in use

    acquired_third = threading.Event()

    def _take():
        b.acquire(3)   # needs 3, only 2 free -> must block
        acquired_third.set()

    t = threading.Thread(target=_take, daemon=True)
    t.start()
    assert not acquired_third.wait(0.3), "third acquire should block while only 2 permits free"
    b.release(3)       # now 5 free
    assert acquired_third.wait(2.0), "third acquire should proceed after a release"
    t.join(2.0)


def test_budget_clamps_oversized_request():
    """An entry demanding more than the whole budget still runs (alone), never deadlocks."""
    b = GpuBudget(8)
    got = b.acquire(999)          # clamped to 8
    assert got == 8
    b.release(999)                # clamp on release too; never exceeds total
    assert b._avail == 8


def test_budget_release_does_not_exceed_total():
    b = GpuBudget(4)
    b.acquire(2)
    b.release(2)
    b.release(2)                  # stray release must not push avail above total
    assert b._avail == 4


# --------------------------------------------------------------------------- #
# serial_only inheritance: configuration -> suite -> test
# --------------------------------------------------------------------------- #

def _apply_defaults(config_defaults, tests):
    proc = _ConfigProcessor.__new__(_ConfigProcessor)
    return proc._apply_test_defaults(tests, config_defaults)


def test_serial_only_inherited_from_config_default():
    out = _apply_defaults({"serial_only": True}, [{"name": "t"}])
    assert out[0]["serial_only"] is True


def test_serial_only_test_level_overrides_default():
    out = _apply_defaults({"serial_only": True}, [{"name": "t", "serial_only": False}])
    assert out[0]["serial_only"] is False


# --------------------------------------------------------------------------- #
# CLI flag-combination validation
# --------------------------------------------------------------------------- #

def test_rerun_failed_with_jobs_is_rejected():
    err = test_runner.validate_arg_combinations(_Args(jobs=4, rerun_failed=True))
    assert err is not None and "rerun-failed" in err


def test_rerun_failed_serial_is_allowed():
    assert test_runner.validate_arg_combinations(_Args(jobs=1, rerun_failed=True)) is None


def test_jobs_without_rerun_is_allowed():
    assert test_runner.validate_arg_combinations(_Args(jobs=4, rerun_failed=False)) is None


def test_stop_on_rerun_requires_rerun_failed():
    err = test_runner.validate_arg_combinations(_Args(stop_on_rerun_failure=True, rerun_failed=False))
    assert err is not None and "stop-on-rerun-failure" in err


# --------------------------------------------------------------------------- #
# _parallel_gpu_budget: the budget defaults to the detected node size
# --------------------------------------------------------------------------- #

def test_budget_defaults_to_detected_node_size():
    # --max-parallel-gpus not given -> follow the hardware, not a literal 8.
    assert _executor(max_parallel_gpus=None, detected_gpus=2)._parallel_gpu_budget() == 2
    assert _executor(max_parallel_gpus=None, detected_gpus=8)._parallel_gpu_budget() == 8
    assert _executor(max_parallel_gpus=None, detected_gpus=16)._parallel_gpu_budget() == 16


def test_budget_explicit_flag_wins_over_detection():
    assert _executor(max_parallel_gpus=4, detected_gpus=8)._parallel_gpu_budget() == 4
    assert _executor(max_parallel_gpus=16, detected_gpus=8)._parallel_gpu_budget() == 16


def test_budget_falls_back_to_8_when_detection_failed():
    # gpus_per_node returns 0 when it cannot tell; keep _resolve_gpu_count's convention.
    assert _executor(max_parallel_gpus=None, detected_gpus=0)._parallel_gpu_budget() == 8


def test_single_gpu_node_does_not_admit_co_tenants():
    """Regression: with a hard-coded budget of 8, four 1-GPU entries piled onto one GPU."""
    ex = _executor(max_parallel_gpus=None, detected_gpus=1)
    assert ex._parallel_gpu_budget() == 1
    one_gpu = {"num_ranks": 1, "num_gpus": "auto"}
    assert ex._gpu_demand(one_gpu, {}) == 1
    # Eligible, but the budget of 1 lets exactly one hold it at a time.
    b = GpuBudget(ex._parallel_gpu_budget())
    b.acquire(ex._gpu_demand(one_gpu, {}))
    assert b._avail == 0, "a second 1-GPU entry must not fit alongside the first"


def test_two_gpu_node_bounds_co_tenancy_to_two():
    ex = _executor(max_parallel_gpus=None, detected_gpus=2)
    b = GpuBudget(ex._parallel_gpu_budget())
    b.acquire(1); b.acquire(1)
    assert b._avail == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
