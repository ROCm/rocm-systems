#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Host tests for the init-pipeline two-entry overlap orchestration (Phase 3).

These prove the mechanism the hardware Gate A4 checks, without a GPU: entries
initialize concurrently but execute strictly one at a time. A fake entry
(`_fake_entry.py`) stands in for the real gtest binary and drives the real
`Rendezvous` token protocol, so the orchestration + timing model are exercised
end to end. Everything here is portable (verified on Windows and WSL/Linux).

Run:  cd tools/scripts/test_runner && python -m pytest tests/test_pipeline.py -v
"""

import os
import subprocess
import sys
import time

import pytest

_RUNNER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUNNER_ROOT not in sys.path:
    sys.path.insert(0, _RUNNER_ROOT)

from lib.pipeline import (  # noqa: E402
    OverlapEntry,
    Rendezvous,
    run_overlap_batch,
    RESULT_FAILED,
    RESULT_TIMED_OUT,
)

FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fake_entry.py")


# --------------------------------------------------------------------------- #
# Harness: real Rendezvous + real subprocesses, injected into run_overlap_batch
# --------------------------------------------------------------------------- #
def _build_entries(tmp_path, specs):
    base = str(tmp_path)
    run = Rendezvous.new_run_uuid()
    entries = []
    for i, s in enumerate(specs):
        rdv = Rendezvous.for_entry(base, run, i)
        log_path = os.path.join(base, f"entry_{i}.log")
        e = OverlapEntry(seq=i, label=s["label"], rendezvous=rdv, log_path=log_path)
        e._params = s  # test-only: warm/exec/code/skip carried for the fake spawn
        entries.append(e)
    return entries


def _make_io(timeline):
    def spawn(entry):
        s = entry._params
        fd = open(entry.log_path, "wb")
        args = [sys.executable, FAKE,
                str(s["warm"]), str(s["exec"]), str(s["code"]), entry.label, timeline]
        if s.get("skip_ready"):
            args.append("--skip-ready")
        env = {**os.environ,
               "RCCL_TEST_RENDEZVOUS_DIR": entry.rendezvous.dir,
               "RCCL_TEST_READY_GO": "1"}
        proc = subprocess.Popen(args, env=env, stdout=fd, stderr=subprocess.STDOUT)
        return proc, fd

    def wait_exit(proc, deadline):
        timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return "timeout"

    def infer(entry, rc):
        return "PASSED" if rc == 0 else "FAILED"

    terminated = []

    def terminate(proc):
        terminated.append(proc)
        try:
            proc.kill()
        except OSError:
            pass

    return spawn, wait_exit, infer, terminate, terminated


def _parse_intervals(timeline):
    rows = []
    if os.path.exists(timeline):
        with open(timeline) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 3:
                    rows.append((parts[0], float(parts[1]), float(parts[2])))
    return rows


# --------------------------------------------------------------------------- #
# Rendezvous unit tests (pure)
# --------------------------------------------------------------------------- #
def test_rendezvous_go_is_atomic(tmp_path):
    r = Rendezvous.for_entry(str(tmp_path), "run", 0)
    assert not os.path.exists(r.go_path)
    r.write_go()
    assert os.path.exists(r.go_path)
    assert not os.path.exists(r.go_path + ".tmp")  # temp cleaned by rename


def test_rendezvous_ready_detection(tmp_path):
    r = Rendezvous.for_entry(str(tmp_path), "run", 3)
    assert r.ready() is False
    with open(r.ready_path, "w") as f:
        f.write("ready\n")
    assert r.ready() is True
    assert r.dir.endswith(os.path.join("rendezvous", "run", "entry_3"))


# --------------------------------------------------------------------------- #
# Orchestration tests (real subprocesses, portable)
# --------------------------------------------------------------------------- #
def test_concurrent_init_overlaps(tmp_path):
    """Two entries that each warm ~0.4 s must both reach READY in ~0.4 s total
    (concurrent), not ~0.8 s (serial) -- init overlaps."""
    timeline = os.path.join(str(tmp_path), "timeline.txt")
    entries = _build_entries(tmp_path, [
        {"label": "A", "warm": 0.4, "exec": 0.05, "code": 0},
        {"label": "B", "warm": 0.4, "exec": 0.05, "code": 0},
    ])
    spawn, wait_exit, infer, terminate, _ = _make_io(timeline)
    run_overlap_batch(entries, spawn=spawn, wait_exit=wait_exit, infer=infer,
                      terminate=terminate, init_timeout=30, exec_timeout=30)

    assert all(e.result == "PASSED" for e in entries)
    span_to_all_ready = max(e.t_ready for e in entries) - min(e.t_launch for e in entries)
    assert span_to_all_ready < 0.7, f"init did not overlap (took {span_to_all_ready:.3f}s)"


def test_serial_execution_no_overlap(tmp_path):
    """Executions must not overlap: the next GO is only written after the prior
    entry exits."""
    timeline = os.path.join(str(tmp_path), "timeline.txt")
    entries = _build_entries(tmp_path, [
        {"label": "A", "warm": 0.1, "exec": 0.3, "code": 0},
        {"label": "B", "warm": 0.1, "exec": 0.3, "code": 0},
    ])
    spawn, wait_exit, infer, terminate, _ = _make_io(timeline)
    run_overlap_batch(entries, spawn=spawn, wait_exit=wait_exit, infer=infer,
                      terminate=terminate, init_timeout=30, exec_timeout=30)

    assert all(e.result == "PASSED" for e in entries)
    intervals = sorted(_parse_intervals(timeline), key=lambda r: r[1])
    assert len(intervals) == 2
    (_, _, end0), (_, start1, _) = intervals[0], intervals[1]
    assert start1 >= end0, f"executions overlapped: {intervals}"


def test_exit_before_ready_is_failed(tmp_path):
    """An entry that dies before READY is FAILED (init phase) and never released;
    a healthy sibling still executes and passes."""
    timeline = os.path.join(str(tmp_path), "timeline.txt")
    entries = _build_entries(tmp_path, [
        {"label": "bad", "warm": 0.05, "exec": 0.0, "code": 7, "skip_ready": True},
        {"label": "good", "warm": 0.1, "exec": 0.1, "code": 0},
    ])
    spawn, wait_exit, infer, terminate, _ = _make_io(timeline)
    run_overlap_batch(entries, spawn=spawn, wait_exit=wait_exit, infer=infer,
                      terminate=terminate, init_timeout=30, exec_timeout=30)

    bad = next(e for e in entries if e.label == "bad")
    good = next(e for e in entries if e.label == "good")
    assert bad.result == RESULT_FAILED and bad.phase == "init"
    assert good.result == "PASSED" and good.phase == "exec"


def test_isolated_logs(tmp_path):
    """Each entry's captured log contains only its own bytes."""
    timeline = os.path.join(str(tmp_path), "timeline.txt")
    entries = _build_entries(tmp_path, [
        {"label": "AAA", "warm": 0.1, "exec": 0.1, "code": 0},
        {"label": "BBB", "warm": 0.1, "exec": 0.1, "code": 0},
    ])
    spawn, wait_exit, infer, terminate, _ = _make_io(timeline)
    run_overlap_batch(entries, spawn=spawn, wait_exit=wait_exit, infer=infer,
                      terminate=terminate, init_timeout=30, exec_timeout=30)

    a = open(entries[0].log_path).read()
    b = open(entries[1].log_path).read()
    assert "AAA" in a and "BBB" not in a
    assert "BBB" in b and "AAA" not in b


def test_phase_timings_monotonic(tmp_path):
    timeline = os.path.join(str(tmp_path), "timeline.txt")
    entries = _build_entries(tmp_path, [
        {"label": "A", "warm": 0.15, "exec": 0.15, "code": 0},
    ])
    spawn, wait_exit, infer, terminate, _ = _make_io(timeline)
    run_overlap_batch(entries, spawn=spawn, wait_exit=wait_exit, infer=infer,
                      terminate=terminate, init_timeout=30, exec_timeout=30)

    e = entries[0]
    assert e.t_launch <= e.t_ready <= e.t_go <= e.t_exit
    t = e.phase_timings()
    assert t["time_to_ready"] >= 0
    assert t["ready_queue_wait"] >= 0
    assert t["execution_time"] >= 0
    assert t["total"] >= t["execution_time"]


def test_exec_timeout_terminates(tmp_path):
    """An entry that runs past exec_timeout is TIMED_OUT and torn down."""
    timeline = os.path.join(str(tmp_path), "timeline.txt")
    entries = _build_entries(tmp_path, [
        {"label": "slow", "warm": 0.1, "exec": 5.0, "code": 0},
    ])
    spawn, wait_exit, infer, terminate, terminated = _make_io(timeline)
    run_overlap_batch(entries, spawn=spawn, wait_exit=wait_exit, infer=infer,
                      terminate=terminate, init_timeout=30, exec_timeout=0.4)

    e = entries[0]
    assert e.result == RESULT_TIMED_OUT and e.phase == "exec"
    assert e.proc in terminated


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
