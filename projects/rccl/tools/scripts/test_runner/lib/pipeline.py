#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Init-pipeline orchestration (Phase 3: minimal two-entry overlap).

This is the Python side of the init-parallel / execute-serial pipeline. It pairs
with the C++ READY/GO rendezvous (``test/common/Rendezvous.*``): each launched
test entry warms its RCCL device code, atomically publishes a ``ready`` token and
blocks; the orchestrator releases entries one ``go`` at a time and waits for each
to fully exit before releasing the next.

Two properties this module guarantees and that its host tests prove:
  * concurrent initialization -- all entries are launched before any is released,
    so their (dominant) device-code warmup overlaps;
  * serial execution -- exactly one entry runs between its GO and its exit, and
    the next GO is not written until the previous entry's process tree has
    exited, so no two tested executions overlap.

This is deliberately the *minimal* orchestration for Gate A4 (a simple stream,
released in READY order). It is NOT the full Phase-4 scheduler -- there is no
bounded init-pool, no mixed serial/pipeline units, and no coordinator eligibility
structure. Those land in ``PipelineScheduler`` later; the ``Rendezvous`` and the
phase-timing model here are the reusable foundation.

All process I/O is injected (``spawn``/``wait_exit``/``infer``/``terminate``) so
the orchestration logic is unit-testable on any OS with a fake entry, while the
real runner supplies process-group-aware implementations.
"""

import os
import shutil
import time
import uuid
from dataclasses import dataclass

READY_TOKEN = "ready"
GO_TOKEN = "go"

# Terminal results for entries that never reached execution.
RESULT_FAILED = "FAILED"        # exited before READY
RESULT_TIMED_OUT = "TIMED_OUT"  # init- or exec-phase timeout


class Rendezvous:
    """Filesystem READY/GO rendezvous (Python side).

    Mirrors ``test/common/Rendezvous.*``: the test binary writes ``ready`` (after
    warmup) and the runner writes ``go`` (to release execution), each via a
    temp-file + atomic ``os.replace`` so neither side observes a torn token. Each
    entry gets its own directory under a per-run UUID namespace, so a stale token
    from a previous run can never release a new entry.
    """

    def __init__(self, entry_dir):
        self.dir = entry_dir
        os.makedirs(self.dir, exist_ok=True)

    @classmethod
    def for_entry(cls, base_dir, run_uuid, seq):
        """Rendezvous dir for entry ``seq`` of run ``run_uuid`` under ``base_dir``."""
        return cls(os.path.join(base_dir, "rendezvous", run_uuid, f"entry_{seq}"))

    @property
    def ready_path(self):
        return os.path.join(self.dir, READY_TOKEN)

    @property
    def go_path(self):
        return os.path.join(self.dir, GO_TOKEN)

    def ready(self):
        """True once the binary has published its READY token."""
        return os.path.exists(self.ready_path)

    def write_go(self):
        """Atomically publish the GO token (temp + rename in the same dir)."""
        tmp = self.go_path + ".tmp"
        with open(tmp, "w") as f:
            f.write("go\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.go_path)

    def cleanup(self):
        """Remove this entry's rendezvous dir (best effort)."""
        shutil.rmtree(self.dir, ignore_errors=True)

    @staticmethod
    def new_run_uuid():
        return uuid.uuid4().hex

    @staticmethod
    def gc_run(base_dir, run_uuid):
        """Remove one run's rendezvous tree (best effort). A fresh run_uuid per
        run means we never touch a concurrent run's directories."""
        shutil.rmtree(os.path.join(base_dir, "rendezvous", run_uuid), ignore_errors=True)


@dataclass
class OverlapEntry:
    """One entry in a two-/N-entry overlap batch."""
    seq: int
    label: str
    rendezvous: Rendezvous
    log_path: str = None
    proc: object = None
    capture_fd: object = None
    # Runner-observed monotonic stamps (never rank-local clocks -- not comparable).
    t_launch: float = None
    t_ready: float = None
    t_go: float = None
    t_exit: float = None
    result: str = None
    phase: str = None       # phase in which the entry ended: 'init' | 'exec'
    exit_code: int = None

    def phase_timings(self):
        """Phase durations from runner-observed stamps; a phase never reached is
        reported as ``None`` (unavailable), never 0."""
        def _d(a, b):
            return (b - a) if (a is not None and b is not None) else None
        return {
            "time_to_ready": _d(self.t_launch, self.t_ready),
            "ready_queue_wait": _d(self.t_ready, self.t_go),   # NOT charged to execution
            "execution_time": _d(self.t_go, self.t_exit),
            "total": _d(self.t_launch, self.t_exit),
        }


def _close_fd(entry):
    if entry.capture_fd is not None:
        try:
            entry.capture_fd.close()
        except OSError:
            pass
        entry.capture_fd = None


def run_overlap_batch(entries, *, spawn, wait_exit, infer, terminate,
                      init_timeout=None, exec_timeout=None,
                      monotonic=time.monotonic, poll_interval=0.05,
                      log=lambda *a: None):
    """Run a small batch of entries: overlap their init, execute them serially.

    Args:
        entries: list of OverlapEntry (each with a fresh Rendezvous + log_path).
        spawn(entry) -> (proc, capture_fd): launch the entry non-blocking. The
            entry's env must carry RCCL_TEST_RENDEZVOUS_DIR = entry.rendezvous.dir
            (and RCCL_TEST_READY_GO) so the binary warms + parks at READY.
        wait_exit(proc, deadline) -> int | 'timeout': block for the process tree
            to exit; 'timeout' if ``deadline`` (monotonic) passes first.
        infer(entry, rc) -> str: map a real exit code to a result string.
        terminate(proc) -> None: tear down a process (group) on timeout.
        init_timeout / exec_timeout: seconds; None = wait indefinitely (the
            runner owns timeouts). poll_interval: READY-poll cadence.

    Returns the same ``entries`` list, each populated with stamps + result.
    """
    # 1) Launch everything first so device-code warmup overlaps across entries.
    for e in entries:
        e.t_launch = monotonic()
        e.proc, e.capture_fd = spawn(e)
        log(f"[pipeline] launched entry {e.seq} ({e.label})")

    # 2) Wait until each entry reaches READY (or dies / times out). Record the
    #    order in which they become READY -- that is the simple release stream.
    pending = list(entries)
    ready_order = []
    while pending:
        still = []
        for e in pending:
            if e.rendezvous.ready():
                e.t_ready = monotonic()
                ready_order.append(e)
                log(f"[pipeline] entry {e.seq} READY")
            elif e.proc.poll() is not None:
                # Exited before publishing READY -> init failure (never released).
                e.t_exit = monotonic()
                e.exit_code = e.proc.returncode
                e.result = RESULT_FAILED
                e.phase = "init"
                _close_fd(e)
                log(f"[pipeline] entry {e.seq} exited before READY -> FAILED")
            elif init_timeout is not None and (monotonic() - e.t_launch) > init_timeout:
                e.result = RESULT_TIMED_OUT
                e.phase = "init"
                terminate(e.proc)
                e.t_exit = monotonic()
                _close_fd(e)
                log(f"[pipeline] entry {e.seq} init timeout -> TIMED_OUT")
            else:
                still.append(e)
        pending = still
        if pending:
            time.sleep(poll_interval)

    # 3) Release READY entries one at a time. Waiting for full exit before the
    #    next GO is what makes tested executions strictly serial (no overlap).
    for e in ready_order:
        e.t_go = monotonic()
        e.rendezvous.write_go()
        log(f"[pipeline] entry {e.seq} GO")
        deadline = (monotonic() + exec_timeout) if exec_timeout else None
        rc = wait_exit(e.proc, deadline)
        e.t_exit = monotonic()
        e.phase = "exec"
        if rc == "timeout":
            e.result = RESULT_TIMED_OUT
            terminate(e.proc)
            log(f"[pipeline] entry {e.seq} exec timeout -> TIMED_OUT")
        else:
            e.exit_code = rc
            e.result = infer(e, rc)
            log(f"[pipeline] entry {e.seq} exec done rc={rc} -> {e.result}")
        _close_fd(e)

    return entries
