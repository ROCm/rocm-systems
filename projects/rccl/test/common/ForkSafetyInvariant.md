# Fork-safety invariant for the init-pipeline test harness

**Status:** normative for any `RCCL_TEST_READY_GO` / init-pipeline warmup code.
**Applies to:** `test/common/` (`main.cpp`, `main_fixtures.cpp`, `main_mpi.cpp`,
`TestBed.cpp`, `TestBedChild.cpp`, `MPIEnvironment.cpp`) and the Python runner in
`tools/scripts/test_runner/`.

This document records the hard rule that the init-overlap POC (2026-08-15,
OCI gfx942 / ROCm 7.0.2) proved must not be broken, so no future warmup work
re-introduces the same crash. There is currently **no** warmup hook in the
mains on `develop` — this is documentation and a review checklist, not a revert.

---

## The invariant (non-negotiable)

> In the fork-based test binaries, **no pipeline code may initialize HIP, RCCL,
> profiling, or any accelerator runtime in the TestBed parent before the
> execution children for that configuration are forked.**

Before the fork, the parent may only do CPU-only work: environment parsing, pipe
setup, rendezvous-path construction, logging. It must **not** call, directly or
transitively:

- `hipSetDevice` or any implicit context-creating HIP call,
- `ncclCommInitRank` / `ncclCommInitAll` / any `ncclCommInit*`,
- HIP-initializing profiler setup,
- any helper whose HIP behavior is unknown.

The correct place to warm the device code is **inside the already-forked
execution children/ranks** (via `CHILD_WARMUP` in `TestBed`/`TestBedChild`, and
inside `MPIEnvironment::SetUp()` for the MPI binary), never in the fork parent.

---

## Why — the POC failure signature (do not reproduce)

The POC put the warmup in the **TestBed parent** (`main.cpp`, before
`RUN_ALL_TESTS`): `hipSetDevice` + `ncclCommInitAll` on a per-rank device, then a
file barrier. Result on hardware:

- **SIGSEGV at `test/common/TestBed.cpp` `PIPE_WRITE(childId, cmd)`** inside
  `TestBed::SetCollectiveArgs`, reached from `RunSimpleSweep` →
  `AllReduce_OutOfPlace_Test::TestBody`.
- **18 × `signal 11`** across the suite; every entry "completed" in **`0.000 s`**.
- Isolated smoke: warmup + barrier worked (`ready` written, blocked correctly),
  but the real collective after release exited **139 (SIGSEGV)**.

**Root cause (rocgdb A/B):** `TestBed` **`fork()`s one child process per rank**
and drives them over pipes — this happens **even at `UT_PROCESS_MASK=1`**
(single-process mode still forks; verified at `TestBed.cpp:107`). The parent
hook initialized HIP in the parent, so `TestBed` then forked its rank-children
from a **HIP-dirty parent**. HIP state does not survive `fork()`, the children
were corrupted, and the first `PIPE_WRITE` segfaulted. Same failure class as the
rocprofv3 and cuMem "HIP-init + fork don't mix" bugs.

**Deeper point:** even without the crash, the parent's warmed device code
**cannot help the forked rank-children** — they run the real `ncclCommInitRank`
in fresh processes the parent cannot pre-warm. Parent-side warmup can neither run
safely nor deliver the overlap. The earlier "9–14× speedup" micro-benchmark
numbers were **warm-then-crash artifacts** (wall-time only, pass/fail never
checked); one run even landed on a 0-GPU node and silently no-op'd. All prior
init-overlap *measurements* are void.

MPI is the safe first slice: `MPIEnvironment::SetUp()` runs one process per rank
via `mpirun` and does **not** fork rank-children, so it structurally sidesteps
this hazard.

---

## Enforcement

1. **Source review.** For every warmup/rendezvous change, confirm no HIP/RCCL/
   profiler call sits on a code path that runs in the TestBed parent before the
   fork at `TestBed.cpp:107` (or before `initialize_devices()` for MPI). Treat any
   helper of unknown HIP behavior as forbidden pre-fork.

2. **Runtime PID diagnostic (Gate A2).** Every warmup HIP/RCCL call must log its
   PID. Assert:
   - `warmup PID == InitComms PID == execution PID` (the process that warms a
     device is the one that runs the test on it);
   - the warmed device set ⊇ the assigned device set;
   - **no warmup HIP/RCCL call runs with the TestBed parent's PID.**

3. **Defense-in-depth.** When `RCCL_TEST_READY_GO` is set, assert the sweep
   produced exactly one child generation (`generation == 1`). This is a backstop
   only — eligibility/routing is decided by the runner **before launch** from the
   config inventory, never from a runtime signal.

4. **Workload-validity gates (runner).** Reject a run as *invalid* (not slow/fast
   — void) before accepting any timing: signal termination (SIGSEGV / signal 11 /
   exit 139 / core) is always a failure; a nonzero/negative return code cannot be
   overridden into PASSED by a partial gtest JSON; no unexplained cluster of
   `0.000 s` durations; expected GPU count and visible-device env logged and
   nonzero. See `tools/scripts/test_runner/tests/test_launch_api.py`.
