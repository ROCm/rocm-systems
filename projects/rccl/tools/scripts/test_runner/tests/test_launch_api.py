#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Launch-API regression tests for the split ``run_test`` (init-pipeline Phase 0.2).

These guard the seam introduced when the monolithic ``TestExecutor.run_test`` was
split into ``_build_launch_spec`` -> ``_spawn`` / ``_spawn_captured`` ->
``_wait_and_infer``. The properties tested are exactly the ones the init-overlap
POC silently violated (see ``test/common/ForkSafetyInvariant.md``):

  * a real 2 s test must report ~2 s, never ``0.000 s`` (the POC's tell-tale bug);
  * a nonzero / signal return code can never be turned into PASSED by a partial or
    stale gtest JSON (18 SIGSEGVs "passed" in the POC);
  * duration is measured across the process wait, not across spec building;
  * the capture-fd launch primitive cannot deadlock on a full pipe buffer.

The pure and fake-process tests run on any OS (verified on Windows). The tests
that spawn a real process group are POSIX-only (the runner targets Linux
clusters) and are skipped elsewhere.

Run:  cd tools/scripts/test_runner && python -m pytest tests/test_launch_api.py -v
"""

import os
import subprocess
import sys
import time

import pytest

# Make ``lib`` importable when pytest is invoked from the test_runner root.
_RUNNER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER_ROOT not in sys.path:
    sys.path.insert(0, _RUNNER_ROOT)

# Aliased so pytest does not try to collect the imported ``Test*`` classes.
from lib.test_executor import (  # noqa: E402
    ExitCode,
    LaunchSpec,
    infer_gtest_result_from_json_file,
)
from lib.test_executor import TestExecutor as Executor  # noqa: E402
from lib.test_executor import TestResult as Result  # noqa: E402

POSIX_ONLY = pytest.mark.skipif(
    os.name != "posix",
    reason="spawns a real process group (start_new_session / killpg): Linux/POSIX only",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _bare_executor():
    """A TestExecutor with no __init__ side effects (no config, no dirs).

    ``_wait_and_infer`` only touches ``self._terminate_process_group`` on the
    timeout/error branches, so a stub keeps the fake-process tests OS-portable
    (real ``_terminate_process_group`` calls ``os.getpgid``, which is POSIX-only).
    """
    ex = Executor.__new__(Executor)
    ex._terminated = []
    ex._terminate_process_group = lambda proc: ex._terminated.append(proc)
    return ex


class _FakeProc:
    """Stand-in for a Popen: ``wait`` optionally sleeps then returns ``rc``, or
    raises ``TimeoutExpired`` to exercise the timeout branch."""

    def __init__(self, rc=0, sleep=0.0, raise_timeout=False):
        self._rc = rc
        self._sleep = sleep
        self._raise_timeout = raise_timeout
        self.pid = -1

    def wait(self, timeout=None):
        if self._raise_timeout:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
        if self._sleep:
            time.sleep(self._sleep)
        return self._rc


def _spec(**kw):
    base = dict(name="T", cmd="true", run_cwd=".", env={}, timeout=0, is_gtest=False)
    base.update(kw)
    return LaunchSpec(**base)


def _write_gtest_json(tmp_path, result):
    """Write a minimal one-leaf gtest JSON report and return its path."""
    p = tmp_path / "gtest.json"
    failures = '[{"failure": "boom"}]' if result == "FAILED" else "[]"
    res = "COMPLETED" if result in ("PASSED", "FAILED") else "SKIPPED"
    p.write_text(
        '{"testsuites":[{"testsuite":[{"name":"MyTest",'
        f'"result":"{res}","failures":{failures}}}]}}]}}',
        encoding="utf-8",
    )
    return str(p)


# --------------------------------------------------------------------------- #
# infer_gtest_result_from_json_file -- pure, runs everywhere
# --------------------------------------------------------------------------- #
def test_crash_returncode_cannot_become_passed(tmp_path):
    """A crash (SIGSEGV -> 139) with a JSON full of passing tests is FAILED, not
    PASSED. This is the POC's core corruption: 18 signal-11s reported success."""
    json_path = _write_gtest_json(tmp_path, "PASSED")
    assert infer_gtest_result_from_json_file(json_path, 139) == Result.RESULT_FAILED.value
    assert infer_gtest_result_from_json_file(json_path, -11) == Result.RESULT_FAILED.value
    assert infer_gtest_result_from_json_file(json_path, 1) == Result.RESULT_FAILED.value


def test_success_json_is_passed(tmp_path):
    json_path = _write_gtest_json(tmp_path, "PASSED")
    assert infer_gtest_result_from_json_file(json_path, 0) == Result.RESULT_PASSED.value


def test_skipped_only_json_is_skipped(tmp_path):
    json_path = _write_gtest_json(tmp_path, "SKIPPED")
    assert infer_gtest_result_from_json_file(json_path, 0) == Result.RESULT_SKIPPED.value


def test_failed_leaf_json_is_failed(tmp_path):
    json_path = _write_gtest_json(tmp_path, "FAILED")
    assert infer_gtest_result_from_json_file(json_path, 0) == Result.RESULT_FAILED.value


def test_timeout_returncode_is_timeout(tmp_path):
    json_path = _write_gtest_json(tmp_path, "PASSED")
    assert (
        infer_gtest_result_from_json_file(json_path, ExitCode.EXIT_TIMEOUT)
        == Result.RESULT_TIMEOUT.value
    )


def test_missing_json_rc0_falls_back_passed():
    assert infer_gtest_result_from_json_file("", 0) == Result.RESULT_PASSED.value


# --------------------------------------------------------------------------- #
# _wait_and_infer -- fake process, runs everywhere
# --------------------------------------------------------------------------- #
def test_duration_measures_the_wait_not_zero():
    """A process that runs ~0.3 s must report ~0.3 s -- the regression that
    catches the POC's cluster of ``0.000 s`` completions."""
    ex = _bare_executor()
    start = time.time()
    res = ex._wait_and_infer(_FakeProc(rc=0, sleep=0.3), _spec(), start)
    assert res["result"] == Result.RESULT_PASSED.value
    assert res["duration"] >= 0.25, f"duration collapsed to {res['duration']}"


def test_wait_and_infer_crash_is_failed(tmp_path):
    """Even with a fully-passing gtest JSON, a signal exit is FAILED."""
    ex = _bare_executor()
    spec = _spec(is_gtest=True, gtest_json_path=_write_gtest_json(tmp_path, "PASSED"))
    res = ex._wait_and_infer(_FakeProc(rc=-11), spec, time.time())
    assert res["result"] == Result.RESULT_FAILED.value
    assert res["exit_code"] == -11


def test_wait_and_infer_nongtest_success():
    ex = _bare_executor()
    res = ex._wait_and_infer(_FakeProc(rc=0), _spec(is_gtest=False), time.time())
    assert res["result"] == Result.RESULT_PASSED.value


def test_wait_and_infer_timeout_branch():
    ex = _bare_executor()
    proc = _FakeProc(raise_timeout=True)
    res = ex._wait_and_infer(proc, _spec(timeout=1), time.time())
    assert res["result"] == Result.RESULT_TIMEOUT.value
    assert proc in ex._terminated, "process group must be torn down on timeout"


def test_wait_and_infer_carries_metadata():
    ex = _bare_executor()
    spec = _spec(
        binary="rccl-UnitTests", num_nodes=2, num_gpus=8, num_ranks=16,
        exec_mode="mpi", perf_nthreads=1, emit_log_path="/tmp/x.log",
    )
    res = ex._wait_and_infer(_FakeProc(rc=0), spec, time.time())
    assert res["binary"] == "rccl-UnitTests"
    assert res["num_ranks"] == 16
    assert res["exec_mode"] == "mpi"
    assert res["log_file"] == "/tmp/x.log"


# --------------------------------------------------------------------------- #
# run_test wrapper -- start_time is taken AFTER spec building
# --------------------------------------------------------------------------- #
def test_run_test_duration_excludes_spec_build_time(monkeypatch):
    """run_test must capture start_time immediately before _spawn, so a slow
    _build_launch_spec never inflates the measured execution time."""
    ex = _bare_executor()

    def slow_build(tc, sc):
        time.sleep(0.4)
        return _spec(), None

    ex._build_launch_spec = slow_build
    ex._spawn = lambda spec: _FakeProc(rc=0, sleep=0.1)
    res = ex.run_test({}, {})
    assert res["result"] == Result.RESULT_PASSED.value
    # 0.1 s of wait, not 0.4 s of build + 0.1 s of wait.
    assert res["duration"] < 0.35, f"spec-build time leaked into duration: {res['duration']}"


def test_run_test_returns_early_result_without_spawning():
    ex = _bare_executor()
    early = {"name": "S", "result": Result.RESULT_SKIPPED.value, "duration": 0}
    ex._build_launch_spec = lambda tc, sc: (None, early)

    def _boom(spec):
        raise AssertionError("_spawn must not be called for an early result")

    ex._spawn = _boom
    assert ex.run_test({}, {}) is early


# --------------------------------------------------------------------------- #
# _spawn_captured -- primitive contract
# --------------------------------------------------------------------------- #
def test_spawn_captured_requires_log_path():
    ex = _bare_executor()
    with pytest.raises(ValueError):
        ex._spawn_captured(_spec(emit_log_path=None))


@POSIX_ONLY
def test_spawn_captured_no_deadlock_on_large_output(tmp_path):
    """A real file fd (not subprocess.PIPE) must let a process emit far more than
    a pipe buffer (~64 KB) without wait() deadlocking; all bytes land in the log."""
    ex = _bare_executor()
    log = tmp_path / "cap.log"
    # ~500 KB of output -- would wedge a PIPE that nobody drains.
    spec = _spec(
        cmd="for i in $(seq 1 5000); do printf '%s\\n' "
        "'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'; done",
        run_cwd=str(tmp_path),
        emit_log_path=str(log),
    )
    proc, fd = ex._spawn_captured(spec)
    try:
        rc = proc.wait(timeout=30)
    finally:
        fd.close()
    assert rc == 0
    assert log.stat().st_size > 400_000


@POSIX_ONLY
def test_spawn_and_wait_real_process_timing(tmp_path):
    """End-to-end through _spawn + _wait_and_infer: a 1 s sleeper reports ~1 s."""
    ex = Executor.__new__(Executor)  # real _terminate_process_group here
    spec = _spec(cmd="sleep 1", run_cwd=str(tmp_path), is_gtest=False)
    start = time.time()
    proc = ex._spawn(spec)
    res = ex._wait_and_infer(proc, spec, start)
    assert res["result"] == Result.RESULT_PASSED.value
    assert 0.8 <= res["duration"] <= 5.0


@POSIX_ONLY
def test_spawn_real_nonzero_is_failed(tmp_path):
    ex = Executor.__new__(Executor)
    spec = _spec(cmd="exit 3", run_cwd=str(tmp_path), is_gtest=False)
    start = time.time()
    proc = ex._spawn(spec)
    res = ex._wait_and_infer(proc, spec, start)
    assert res["result"] == Result.RESULT_FAILED.value


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
