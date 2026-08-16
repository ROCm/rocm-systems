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
import queue
import shutil
import threading
import time
import uuid
from dataclasses import dataclass

READY_TOKEN = "ready"
GO_TOKEN = "go"

# Terminal results.
RESULT_FAILED = "FAILED"         # exited before READY, or a failing exec exit code
RESULT_TIMED_OUT = "TIMED_OUT"   # init- or exec-phase timeout
RESULT_CANCELLED = "CANCELLED"   # loader stopped before admitting the entry
RESULT_INFRA_ERROR = "INFRA_ERROR"  # scheduler/launch fault (distinct from a test failure)

# Entry kinds for the mixed-mode scheduler.
KIND_PIPELINE = "pipeline"  # loader-launched, warms to READY, executor sends GO
KIND_SERIAL = "serial"      # executor-launched only (never stops at READY)


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


@dataclass
class SchedEntry:
    """A unit scheduled by PipelineScheduler.

    kind == KIND_PIPELINE: the loader launches it, it warms and parks at READY,
    the executor sends GO. Its init overlaps other entries.
    kind == KIND_SERIAL: the loader never launches it; the executor launches it
    only when it owns the sole execution slot (so a non-protocol binary can never
    begin executing while another entry executes).
    """
    seq: int
    label: str
    kind: str = KIND_PIPELINE
    rendezvous: Rendezvous = None   # pipeline entries only
    log_path: str = None
    parent: object = None           # optional parent-config id for roll-up/ordering
    sibling_index: int = 0          # position within the parent's sweep (legacy/config order)
    sibling_total: int = 1          # sub-entries for this parent (1 = not a split sweep)
    # runtime process state
    proc: object = None
    capture_fd: object = None
    # runner-observed monotonic stamps
    t_launch: float = None
    t_ready: float = None
    t_go: float = None
    t_exit: float = None
    result: str = None
    phase: str = None               # 'pending' | 'init' | 'exec'
    exit_code: int = None
    # ownership / lifecycle flags -- ALL guarded by PipelineScheduler.state_lock
    finalized: bool = False
    settled: bool = False
    slot_owned: bool = False
    executor_owned: bool = False

    def phase_timings(self):
        def _d(a, b):
            return (b - a) if (a is not None and b is not None) else None
        return {
            "time_to_ready": _d(self.t_launch, self.t_ready),
            "ready_queue_wait": _d(self.t_ready, self.t_go),
            "execution_time": _d(self.t_go, self.t_exit),
            "total": _d(self.t_launch, self.t_exit),
        }


@dataclass
class _ParentGroup:
    """Legacy-ordering state for one split fork sweep (plan section 5 #10)."""
    total: int
    next_index: int = 0                 # the only sibling index currently eligible
    cancelled: bool = False             # a sibling failed -> fail-fast the rest
    ready_by_index: dict = None         # sibling_index -> entry that reached READY early

    def __post_init__(self):
        if self.ready_by_index is None:
            self.ready_by_index = {}


class PipelineScheduler:
    """Mixed-mode init-pipeline scheduler (plan section 5).

    One loader thread, one watcher thread per in-flight pipeline entry, and a
    single executor thread, coordinated by a bounded slot semaphore and a
    per-entry filesystem rendezvous. Invariants:

      * INITIALIZING + READY <= init_pool  (one BoundedSemaphore; pipeline only)
      * EXECUTING <= 1                      (a single executor thread)
      * every entry reaches exactly one terminal state (Model A accounting)

    Pipeline entries overlap their init with the currently-executing entry; a
    serial entry never begins executing while another entry executes (the loader
    never launches it -- only the executor does, when it owns the sole slot).

    Process I/O is injected so the concurrency logic is unit-testable with a fake
    entry, while the real runner supplies process-group-aware implementations:
      spawn(entry) -> (proc, capture_fd)
      wait_exit(proc, deadline) -> int | 'timeout'
      infer(entry, rc) -> result string
      terminate(proc) -> None
    """

    _SENTINEL = object()

    def __init__(self, entries, *, spawn, wait_exit, infer, terminate,
                 init_pool=2, init_timeout=None, exec_timeout=None,
                 fork_sweep_policy="legacy", release_order="ready",
                 poll_interval=0.05, monotonic=time.monotonic, log=lambda *a: None):
        self.entries = list(entries)
        self.total = len(self.entries)
        self._spawn = spawn
        self._wait_exit = wait_exit
        self._infer = infer
        self._terminate = terminate
        self.init_pool = max(1, int(init_pool))
        self.init_timeout = init_timeout
        self.exec_timeout = exec_timeout
        # 'legacy' preserves each split sweep's original sub-entry order + fail-fast
        # (serial-equivalent); 'independent' runs sub-entries in READY order with no
        # fail-fast (max overlap, different semantics).
        self.fork_sweep_policy = fork_sweep_policy
        # Cross-entry release order: 'ready' executes entries as they reach READY
        # (per-parent legacy ordering still applies within a split sweep);
        # 'config' executes strictly in config (seq) order across all entries --
        # deterministic, with intentional head-of-line blocking on the next seq.
        self.release_order = release_order
        self.poll_interval = poll_interval
        self.monotonic = monotonic
        self.log = log

        # Cross-entry config-order gate (release_order='config'): entries execute
        # in ascending seq; only the lowest not-yet-finished seq is eligible.
        self._by_seq = {e.seq: e for e in self.entries}
        self._ordered_seqs = sorted(self._by_seq)
        self._exec_ptr = 0
        self._ready_seqs = set()

        # Per-parent ordering state for legacy fork sweeps: only the next expected
        # sibling of a parent is executor-eligible; a sibling failure cancels the
        # parent's later siblings (fail-fast), matching the serial isCorrect
        # short-circuit. Groups are keyed by parent; singletons are never gated.
        self._parents = {}
        for e in self.entries:
            if e.sibling_total > 1 and e.parent not in self._parents:
                self._parents[e.parent] = _ParentGroup(total=e.sibling_total)

        self.slot = threading.BoundedSemaphore(self.init_pool)
        self.exec_q = queue.Queue()
        self.state_lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.completed = threading.Event()
        self.worker_exc = queue.Queue()
        self.terminal = 0     # entries whose terminal state has been CLAIMED
        self.settled = 0      # entries whose cleanup has FINISHED
        self._visited = set()  # seqs the loader has admitted/handled (loader thread only)
        self._watchers = []

    # ---- helpers -----------------------------------------------------------
    def _close_fd(self, entry):
        if entry.capture_fd is not None:
            try:
                entry.capture_fd.close()
            except OSError:
                pass
            entry.capture_fd = None

    def _release_slot_if_owned(self, entry):
        with self.state_lock:
            if not entry.slot_owned:
                return
            entry.slot_owned = False
        self.slot.release()

    def _finish_entry(self, entry, result, phase, *, terminate):
        """Idempotent terminal transition. Split so completion is only signalled
        after teardown is SETTLED (plan section 5 #11): claim `terminal` under the
        lock, then terminate/close/record OUTSIDE it, then bump `settled` in a
        finally and signal completion only when settled == total."""
        with self.state_lock:
            if entry.finalized:
                return
            entry.finalized = True
            self.terminal += 1
        try:
            if terminate and entry.proc is not None:
                self._terminate(entry.proc)
            self._release_slot_if_owned(entry)
            self._close_fd(entry)
            entry.result = result
            entry.phase = phase
        finally:
            with self.state_lock:
                self.settled += 1
                reached = (self.settled == self.total)
            if reached and not self.completed.is_set():
                self.completed.set()
                self.exec_q.put(self._SENTINEL)

    # ---- legacy per-parent ordering (plan section 5 #10) --------------------
    def _grouped(self, entry):
        """True if this entry is a sibling of a split fork sweep under legacy
        ordering (singletons and 'independent' policy are never gated)."""
        return (self.fork_sweep_policy == "legacy"
                and entry.sibling_total > 1
                and entry.parent in self._parents)

    def _parent_cancelled(self, entry):
        if not self._grouped(entry):
            return False
        with self.state_lock:
            return self._parents[entry.parent].cancelled

    def _on_ready(self, entry):
        """Decide what to do when a grouped entry reaches READY: 'release'
        (its turn -> execute), 'wait' (a lower-index sibling goes first), or
        'cancel' (the parent already failed-fast). Ungrouped -> always release."""
        if not self._grouped(entry):
            return "release"
        with self.state_lock:
            g = self._parents[entry.parent]
            if g.cancelled:
                return "cancel"
            g.ready_by_index[entry.sibling_index] = entry
            return "release" if entry.sibling_index == g.next_index else "wait"

    def _advance_parent(self, entry):
        """After a grouped entry reaches a terminal result, promote the next
        sibling (on pass/skip) or fail-fast the parent's remaining siblings.
        Preserves the serial nested-loop order + isCorrect short-circuit."""
        if not self._grouped(entry):
            return
        to_release, to_cancel = [], []
        with self.state_lock:
            g = self._parents[entry.parent]
            if entry.result in ("PASSED", "SKIPPED"):
                g.next_index = entry.sibling_index + 1
                nxt = g.ready_by_index.get(g.next_index)
                if nxt is not None and not nxt.finalized:
                    to_release.append(nxt)     # it was waiting; now it's its turn
            else:
                g.cancelled = True             # fail-fast: cancel every later sibling
                for e in self.entries:
                    if (e.parent == entry.parent and e.sibling_total > 1
                            and e.sibling_index > entry.sibling_index
                            and not e.finalized):
                        to_cancel.append(e)
        for e in to_release:
            self.exec_q.put(e)
        for e in to_cancel:
            self._finish_entry(e, RESULT_CANCELLED, "pending", terminate=True)

    # ---- cross-entry config-order gate (release_order='config') -------------
    def _config_next_locked(self):
        """Advance the exec pointer past finalized seqs; return the current
        next-eligible seq, or None. Caller must hold state_lock."""
        while (self._exec_ptr < len(self._ordered_seqs)
               and self._by_seq[self._ordered_seqs[self._exec_ptr]].finalized):
            self._exec_ptr += 1
        if self._exec_ptr < len(self._ordered_seqs):
            return self._ordered_seqs[self._exec_ptr]
        return None

    def _config_on_ready(self, entry):
        """Config-order gate: register the entry as ready-to-execute and release it
        only if it is the next expected seq; otherwise hold it."""
        with self.state_lock:
            self._ready_seqs.add(entry.seq)
            return "release" if self._config_next_locked() == entry.seq else "wait"

    def _advance_config(self, entry):
        """After a terminal result under config order: apply legacy fork fail-fast
        (cancel later siblings of a failed split sweep), then release the next
        config-eligible entry if it is ready."""
        to_cancel = []
        if (self.fork_sweep_policy == "legacy" and entry.sibling_total > 1
                and entry.result not in ("PASSED", "SKIPPED")):
            with self.state_lock:
                g = self._parents.get(entry.parent)
                if g is not None:
                    g.cancelled = True
                for e in self.entries:
                    if (e.parent == entry.parent and e.sibling_total > 1
                            and e.sibling_index > entry.sibling_index and not e.finalized):
                        to_cancel.append(e)
        for e in to_cancel:
            self._finish_entry(e, RESULT_CANCELLED, "pending", terminate=True)
        to_release = []
        with self.state_lock:
            nxt = self._config_next_locked()
            if nxt is not None:
                e = self._by_seq[nxt]
                if nxt in self._ready_seqs and not e.executor_owned and not e.finalized:
                    to_release.append(e)
        for e in to_release:
            self.exec_q.put(e)

    def _on_ready_dispatch(self, entry):
        return self._config_on_ready(entry) if self.release_order == "config" else self._on_ready(entry)

    def _advance_dispatch(self, entry):
        if self.release_order == "config":
            self._advance_config(entry)
        else:
            self._advance_parent(entry)

    # ---- threads -----------------------------------------------------------
    def _loader(self):
        try:
            for entry in self.entries:
                try:
                    if self.stop_flag.is_set():
                        self._visited.add(entry.seq)
                        self._finish_entry(entry, RESULT_CANCELLED, "pending", terminate=False)
                        continue
                    if entry.finalized:
                        # Already cancelled by a sibling's fail-fast (or elsewhere).
                        self._visited.add(entry.seq)
                        continue
                    if entry.kind == KIND_SERIAL:
                        # Hand to executor; the loader NEVER spawns a serial unit.
                        # Under config order it is gated to its seq position.
                        self._visited.add(entry.seq)
                        if self.release_order == "config":
                            if self._config_on_ready(entry) == "release":
                                self.exec_q.put(entry)
                        else:
                            self.exec_q.put(entry)
                        continue
                    if self._grouped(entry) and self._parent_cancelled(entry):
                        # Legacy fail-fast: don't launch a cancelled parent's sibling.
                        self._visited.add(entry.seq)
                        self._finish_entry(entry, RESULT_CANCELLED, "pending", terminate=False)
                        continue
                    # Pipeline: block on a slot (INITIALIZING+READY <= init_pool).
                    got = False
                    while not self.stop_flag.is_set():
                        if self.slot.acquire(timeout=self.poll_interval):
                            got = True
                            break
                    if not got:
                        self._visited.add(entry.seq)
                        self._finish_entry(entry, RESULT_CANCELLED, "pending", terminate=False)
                        continue
                    with self.state_lock:
                        entry.slot_owned = True
                    self._visited.add(entry.seq)
                    entry.t_launch = self.monotonic()
                    entry.proc, entry.capture_fd = self._spawn(entry)
                    self._start_watcher(entry)
                    self.log(f"[sched] launched pipeline entry {entry.seq} ({entry.label})")
                except BaseException as e:  # noqa: BLE001 -- contain per-entry launch faults
                    self._visited.add(entry.seq)
                    self._finish_entry(entry, RESULT_INFRA_ERROR, "init", terminate=True)
                    self.worker_exc.put(e)
                    self.stop_flag.set()
        finally:
            # Finalize every entry the loop never reached, so settled can reach
            # total even on an outer exception (plan section 5 #3).
            for entry in self.entries:
                if entry.seq not in self._visited:
                    self._finish_entry(entry, RESULT_CANCELLED, "pending", terminate=False)

    def _start_watcher(self, entry):
        t = threading.Thread(target=self._watcher, args=(entry,), daemon=True)
        self._watchers.append(t)
        t.start()

    def _watcher(self, entry):
        try:
            deadline = None if self.init_timeout is None else entry.t_launch + self.init_timeout
            while True:
                if self._grouped(entry) and self._parent_cancelled(entry):
                    # A sibling failed while this one was initializing: fail-fast it.
                    self._finish_entry(entry, RESULT_CANCELLED, "init", terminate=True)
                    return
                if entry.rendezvous is not None and entry.rendezvous.ready():
                    with self.state_lock:
                        entry.t_ready = self.monotonic()
                    decision = self._on_ready_dispatch(entry)
                    if decision == "release":
                        self.exec_q.put(entry)   # slot still owned; executor takes over
                    elif decision == "cancel":
                        self._finish_entry(entry, RESULT_CANCELLED, "init", terminate=True)
                    # 'wait': held; the advance path releases it when its turn comes.
                    return
                if entry.proc.poll() is not None:
                    entry.t_exit = self.monotonic()
                    entry.exit_code = entry.proc.returncode
                    self._finish_entry(entry, RESULT_FAILED, "init", terminate=False)
                    self._advance_dispatch(entry)  # a sibling died in init -> advance/fail-fast
                    return
                if deadline is not None and self.monotonic() > deadline:
                    self._finish_entry(entry, RESULT_TIMED_OUT, "init", terminate=True)
                    self._advance_dispatch(entry)
                    return
                if self.stop_flag.is_set():
                    return                       # shutdown() will finalize it
                time.sleep(self.poll_interval)
        except BaseException as e:  # noqa: BLE001
            self._finish_entry(entry, RESULT_INFRA_ERROR, "init", terminate=True)
            self.worker_exc.put(e)
            self.stop_flag.set()

    def _executor(self):
        while True:
            unit = self.exec_q.get()
            if unit is self._SENTINEL:
                return
            # Atomic READY->EXECUTING handoff (plan section 5 #5): claim ownership
            # or skip an already-finalized (e.g. cancelled) unit.
            with self.state_lock:
                if unit.finalized:
                    skip = True
                else:
                    unit.executor_owned = True
                    skip = False
            if skip:
                self._release_slot_if_owned(unit)
                continue
            try:
                if unit.kind == KIND_SERIAL:
                    unit.t_launch = self.monotonic()
                    unit.proc, unit.capture_fd = self._spawn(unit)
                    unit.t_go = unit.t_launch     # no queue wait for a serial unit
                    deadline = None if self.exec_timeout is None else self.monotonic() + self.exec_timeout
                    rc = self._wait_exit(unit.proc, deadline)
                else:
                    self._release_slot_if_owned(unit)  # free slot -> loader refills NOW (overlap)
                    unit.t_go = self.monotonic()
                    unit.rendezvous.write_go()
                    deadline = None if self.exec_timeout is None else self.monotonic() + self.exec_timeout
                    rc = self._wait_exit(unit.proc, deadline)
                unit.t_exit = self.monotonic()
                if rc == "timeout":
                    self._finish_entry(unit, RESULT_TIMED_OUT, "exec", terminate=True)
                else:
                    unit.exit_code = rc
                    self._finish_entry(unit, self._infer(unit, rc), "exec", terminate=False)
                # Promote the next unit: per-parent (ready order) or per-seq (config).
                self._advance_dispatch(unit)
            except BaseException as e:  # noqa: BLE001 -- contain GO/wait/infer faults
                self._finish_entry(unit, RESULT_INFRA_ERROR, "exec", terminate=True)
                self.worker_exc.put(e)
                self.stop_flag.set()

    def _shutdown(self):
        """Finalize entries abandoned on stop. An executor-owned entry is only
        terminated (its owning executor records the result); everything else that
        is not yet finalized is cancelled/infra-errored here (plan section 5 #4/#5)."""
        self.stop_flag.set()
        for e in self.entries:
            with self.state_lock:
                owned = e.executor_owned
                fin = e.finalized
            if fin:
                continue
            if owned:
                if e.proc is not None:
                    self._terminate(e.proc)
            else:
                result = RESULT_CANCELLED if e.t_go is None else RESULT_INFRA_ERROR
                self._finish_entry(e, result, e.phase or "pending", terminate=True)

    def request_stop(self):
        """Ask the scheduler to stop launching new work (e.g. Ctrl-C / fatal)."""
        self.stop_flag.set()

    # ---- driver ------------------------------------------------------------
    def run(self):
        """Run to completion; return the entries (each with a terminal result).

        Raises nothing for test failures (those are per-entry results). If a
        scheduler/launch fault occurred, the exception(s) are available via
        ``worker_exc`` and the caller should treat the run as an infra error.
        """
        loader_t = threading.Thread(target=self._loader, daemon=True)
        exec_t = threading.Thread(target=self._executor, daemon=True)
        loader_t.start()
        exec_t.start()
        try:
            loader_t.join()
            # If a fault stopped the run, finalize whatever the watchers abandoned
            # so `settled` can reach `total` (otherwise completion never arrives).
            if self.stop_flag.is_set():
                self._shutdown()
            self.completed.wait()
        except KeyboardInterrupt:
            self._shutdown()
            self.completed.wait()
            raise
        finally:
            exec_t.join()
            for w in self._watchers:
                w.join(timeout=1.0)
        return self.entries

    def failed(self):
        """True if any scheduler/launch fault was recorded (not a test failure)."""
        return not self.worker_exc.empty()
