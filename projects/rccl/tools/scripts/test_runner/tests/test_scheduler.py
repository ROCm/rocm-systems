#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Host tests for the mixed-mode PipelineScheduler (init-pipeline Phase 4).

Proves the plan section-5 invariants without a GPU, using a fake entry that
drives the real Rendezvous + real subprocesses, plus injected fakes for the
concurrency edge cases (launch faults, cancellation). Portable (Windows + WSL).

The core invariants are checked post-hoc from runner-observed monotonic stamps:
  * INITIALIZING + READY <= init_pool  -> max overlap of [t_launch, t_go) <= pool
  * EXECUTING <= 1                      -> max overlap of [t_go, t_exit) <= 1
  * Model A                             -> every entry has exactly one terminal
                                          result; terminal == settled == total
"""

import os
import subprocess
import sys
import threading
import time

import pytest

_RUNNER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER_ROOT not in sys.path:
    sys.path.insert(0, _RUNNER_ROOT)

from lib.pipeline import (  # noqa: E402
    KIND_PIPELINE,
    KIND_SERIAL,
    PipelineScheduler,
    Rendezvous,
    RESULT_CANCELLED,
    RESULT_FAILED,
    RESULT_INFRA_ERROR,
    RESULT_TIMED_OUT,
    SchedEntry,
)

FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fake_entry.py")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _entries(tmp_path, specs):
    base = str(tmp_path)
    run = Rendezvous.new_run_uuid()
    out = []
    for i, s in enumerate(specs):
        kind = s.get("kind", KIND_PIPELINE)
        rdv = Rendezvous.for_entry(base, run, i) if kind == KIND_PIPELINE else None
        e = SchedEntry(seq=i, label=s["label"], kind=kind, rendezvous=rdv,
                       log_path=os.path.join(base, f"e{i}.log"))
        e._params = s
        out.append(e)
    return out


def _real_io(timeline):
    def spawn(entry):
        s = entry._params
        fd = open(entry.log_path, "wb")
        args = [sys.executable, FAKE, str(s["warm"]), str(s["exec"]),
                str(s["code"]), entry.label, timeline]
        if s.get("skip_ready"):
            args.append("--skip-ready")
        env = {**os.environ, "RCCL_TEST_READY_GO": "1"}
        if entry.rendezvous is not None:
            env["RCCL_TEST_RENDEZVOUS_DIR"] = entry.rendezvous.dir
        proc = subprocess.Popen(args, env=env, stdout=fd, stderr=subprocess.STDOUT)
        return proc, fd

    def wait_exit(proc, deadline):
        timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return "timeout"

    def infer(entry, rc):
        return "PASSED" if rc == 0 else RESULT_FAILED

    def terminate(proc):
        try:
            proc.kill()
        except OSError:
            pass

    return spawn, wait_exit, infer, terminate


def _max_overlap(intervals):
    """Max number of half-open [a, b) intervals overlapping at any instant."""
    events = []
    for a, b in intervals:
        if a is None or b is None or b <= a:
            continue
        events.append((a, 1))
        events.append((b, -1))
    events.sort(key=lambda x: (x[0], x[1]))  # close (-1) before open (+1) at a tie
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


def _assert_model_a(entries):
    for e in entries:
        assert e.finalized, f"entry {e.seq} not finalized"
        assert e.result is not None, f"entry {e.seq} has no result"


# --------------------------------------------------------------------------- #
# Real-subprocess invariant tests
# --------------------------------------------------------------------------- #
def test_all_pipeline_pass_and_bounds(tmp_path):
    timeline = os.path.join(str(tmp_path), "tl.txt")
    entries = _entries(tmp_path, [{"label": f"P{i}", "warm": 0.3, "exec": 0.15, "code": 0}
                                  for i in range(5)])
    spawn, wait_exit, infer, terminate = _real_io(timeline)
    sched = PipelineScheduler(entries, spawn=spawn, wait_exit=wait_exit, infer=infer,
                              terminate=terminate, init_pool=2, init_timeout=30, exec_timeout=30)
    sched.run()

    assert all(e.result == "PASSED" for e in entries)
    _assert_model_a(entries)
    assert sched.terminal == sched.settled == sched.total
    # INITIALIZING + READY <= init_pool
    init_iv = [(e.t_launch, e.t_go) for e in entries]
    assert _max_overlap(init_iv) <= 2, "init-pool bound violated"
    # EXECUTING <= 1
    exec_iv = [(e.t_go, e.t_exit) for e in entries]
    assert _max_overlap(exec_iv) <= 1, "executions overlapped"


def test_init_overlaps_execution(tmp_path):
    """With pool=3, later entries reach READY while an earlier one executes."""
    timeline = os.path.join(str(tmp_path), "tl.txt")
    entries = _entries(tmp_path, [{"label": f"P{i}", "warm": 0.3, "exec": 0.3, "code": 0}
                                  for i in range(3)])
    spawn, wait_exit, infer, terminate = _real_io(timeline)
    sched = PipelineScheduler(entries, spawn=spawn, wait_exit=wait_exit, infer=infer,
                              terminate=terminate, init_pool=3, init_timeout=30, exec_timeout=30)
    sched.run()
    assert all(e.result == "PASSED" for e in entries)
    # Some entry became READY before the first entry finished executing.
    first_exec_end = min(e.t_exit for e in entries)
    assert any(e.t_ready is not None and e.t_ready < first_exec_end for e in entries[1:])


def test_mixed_mode_serial_never_overlaps(tmp_path):
    timeline = os.path.join(str(tmp_path), "tl.txt")
    entries = _entries(tmp_path, [
        {"label": "P0", "warm": 0.2, "exec": 0.2, "code": 0, "kind": KIND_PIPELINE},
        {"label": "S1", "warm": 0.1, "exec": 0.2, "code": 0, "kind": KIND_SERIAL},
        {"label": "P2", "warm": 0.2, "exec": 0.2, "code": 0, "kind": KIND_PIPELINE},
    ])
    spawn, wait_exit, infer, terminate = _real_io(timeline)
    sched = PipelineScheduler(entries, spawn=spawn, wait_exit=wait_exit, infer=infer,
                              terminate=terminate, init_pool=2, init_timeout=30, exec_timeout=30)
    sched.run()
    assert all(e.result == "PASSED" for e in entries)
    exec_iv = [(e.t_go, e.t_exit) for e in entries]
    assert _max_overlap(exec_iv) <= 1, "a serial unit overlapped another execution"


def test_exit_before_ready_failed(tmp_path):
    timeline = os.path.join(str(tmp_path), "tl.txt")
    entries = _entries(tmp_path, [
        {"label": "bad", "warm": 0.05, "exec": 0.0, "code": 3, "skip_ready": True},
        {"label": "ok", "warm": 0.1, "exec": 0.1, "code": 0},
    ])
    spawn, wait_exit, infer, terminate = _real_io(timeline)
    sched = PipelineScheduler(entries, spawn=spawn, wait_exit=wait_exit, infer=infer,
                              terminate=terminate, init_pool=2, init_timeout=30, exec_timeout=30)
    sched.run()
    bad = next(e for e in entries if e.label == "bad")
    ok = next(e for e in entries if e.label == "ok")
    assert bad.result == RESULT_FAILED and bad.phase == "init"
    assert ok.result == "PASSED"
    _assert_model_a(entries)


def test_exec_timeout(tmp_path):
    timeline = os.path.join(str(tmp_path), "tl.txt")
    entries = _entries(tmp_path, [{"label": "slow", "warm": 0.1, "exec": 5.0, "code": 0}])
    spawn, wait_exit, infer, terminate = _real_io(timeline)
    sched = PipelineScheduler(entries, spawn=spawn, wait_exit=wait_exit, infer=infer,
                              terminate=terminate, init_pool=2, init_timeout=30, exec_timeout=0.5)
    sched.run()
    assert entries[0].result == RESULT_TIMED_OUT and entries[0].phase == "exec"


# --------------------------------------------------------------------------- #
# Injected-fake edge cases (deterministic)
# --------------------------------------------------------------------------- #
class _FakeProc:
    def __init__(self, rc=0):
        self._rc = rc
        self.returncode = None
        self.pid = -1

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = self._rc
        return self._rc


def test_spawn_infra_error_completes(tmp_path):
    """A launch fault on one entry is contained: it becomes INFRA_ERROR, the run
    still reaches completion, every entry is terminal, and failed() is True."""
    entries = _entries(tmp_path, [{"label": f"P{i}", "warm": 0, "exec": 0, "code": 0}
                                  for i in range(4)])

    def spawn(entry):
        if entry.seq == 1:
            raise RuntimeError("boom")
        # others: reach READY immediately, exit 0 on wait
        if entry.rendezvous is not None:
            with open(entry.rendezvous.ready_path, "w") as f:
                f.write("ready\n")
        return _FakeProc(0), None

    sched = PipelineScheduler(
        entries, spawn=spawn,
        wait_exit=lambda p, d: p.wait(0),
        infer=lambda e, rc: "PASSED",
        terminate=lambda p: None,
        init_pool=2, init_timeout=30, exec_timeout=30, poll_interval=0.01)
    sched.run()

    _assert_model_a(entries)
    assert sched.terminal == sched.settled == sched.total
    assert sched.failed()
    assert any(e.result == RESULT_INFRA_ERROR for e in entries)


def test_cancellation_all_terminal(tmp_path):
    """request_stop() partway leaves no entry unfinished; completion still arrives."""
    entries = _entries(tmp_path, [{"label": f"P{i}", "warm": 0, "exec": 0, "code": 0}
                                  for i in range(6)])
    sched_ref = {}

    launched = []

    def spawn(entry):
        launched.append(entry.seq)
        if len(launched) == 1:
            sched_ref["s"].request_stop()  # stop after the very first launch
        if entry.rendezvous is not None:
            with open(entry.rendezvous.ready_path, "w") as f:
                f.write("ready\n")
        return _FakeProc(0), None

    sched = PipelineScheduler(
        entries, spawn=spawn,
        wait_exit=lambda p, d: p.wait(0),
        infer=lambda e, rc: "PASSED",
        terminate=lambda p: None,
        init_pool=2, init_timeout=30, exec_timeout=30, poll_interval=0.01)
    sched_ref["s"] = sched
    sched.run()

    _assert_model_a(entries)
    assert sched.terminal == sched.settled == sched.total
    # At least some entries were cancelled (never admitted after stop).
    assert any(e.result == RESULT_CANCELLED for e in entries)


def test_finish_entry_idempotent(tmp_path):
    entries = _entries(tmp_path, [{"label": "P0", "warm": 0, "exec": 0, "code": 0}])
    sched = PipelineScheduler(entries, spawn=lambda e: (_FakeProc(0), None),
                              wait_exit=lambda p, d: 0, infer=lambda e, rc: "PASSED",
                              terminate=lambda p: None)
    e = entries[0]
    sched._finish_entry(e, "PASSED", "exec", terminate=False)
    sched._finish_entry(e, RESULT_FAILED, "exec", terminate=False)  # ignored
    assert e.result == "PASSED"
    assert sched.terminal == 1 and sched.settled == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
