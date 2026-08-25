# 6. Formal correctness: what has to be true, and how to check it

The safety claim behind kernel replay is one sentence: *the application cannot observe that the
kernel ran P times instead of once.* This section turns that sentence into a state-transition
model, extracts the proof obligations, identifies which ones the current implementation discharges
and which it merely assumes, and gives a verification plan that is executable rather than
aspirational.

## 6.1 State model

Fix one GPU agent $a$ and one process. Partition all state a kernel on $a$ can read or write:

| Symbol | Contents | Restored by `restore()`? |
|---|---|---|
| $T$ | tracked coarse-grained VRAM owned by $a$ + module-scope `__device__`/`__constant__` variables | **yes**, in full |
| $U_{alloc}$ | managed/unified memory, stream-ordered pool memory, VMEM-mapped memory | no |
| $U_{flag}$ | coarse VRAM carrying `HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG` | no (side inventory) |
| $U_{host}$ | host memory, fine-grained memory, kernarg buffers | no |
| $U_{peer}$ | VRAM owned by another agent, reachable over XGMI/P2P, and IPC-attached memory owned by another process | no |
| $H$ | caches, TLBs, scratch, clock/power state, hardware counters | no |
| $Q$ | AQL queue contents, read/write indices, signal values | managed separately (§6.4) |

Write $\sigma = (T, U, H, Q)$ with $U = U_{alloc} \cup U_{flag} \cup U_{host} \cup U_{peer}$.

A kernel dispatch $k$ denotes a relation $\llbracket k \rrbracket \subseteq \Sigma \times \Sigma$ —
a *relation*, not a function, because floating-point atomics, unsynchronized races, and
scheduler-dependent reduction order make GPU kernels legitimately nondeterministic. Let
$R(k) \subseteq T \cup U$ and $W(k) \subseteq T \cup U$ be its read and write sets.

The application observes state only through: subsequent kernels' reads, explicit copies
(`hipMemcpy*`, `hsa_memory_copy`, `hsa_amd_memory_async_copy`), host loads of mapped or fine-grained
memory, and HSA signal values. Define the observation function $obs(\sigma)$ as the projection of
$\sigma$ onto $(T \cup U, Q_{signals})$; $H$ is deliberately *not* observable — that is an
assumption, and §6.6 shows where it fails.

## 6.2 The replay transform

With $s = \mathrm{snap}(\sigma_0)$ the captured image of $T$ at window entry, and
$\rho(\sigma, s)$ the state with $T$ overwritten by $s$:

$$\sigma_0 \xrightarrow{\text{drain}} \sigma_0' \xrightarrow{k} \sigma_1 \xrightarrow{\rho(\cdot,s)} \cdots \xrightarrow{k} \sigma_P$$

with **no** $\rho$ after the final pass (`should_continue_replay` returns false and the loop breaks
before the restore). The application resumes from $\sigma_P$.

**Soundness claim.** $obs(\sigma_P) \in obs(\llbracket k \rrbracket(\sigma_0))$ — the state the
application resumes from is one of the states a single execution could have produced.

## 6.3 Proof obligations

The claim decomposes into six obligations. Each is stated so that it is either machine-checkable,
runtime-testable, or must be documented as an assumption.

**O1 — Write-set coverage.** $W(k) \subseteq T$.
*Why:* if the kernel writes $u \in U$, then after P passes $u$ holds the result of P applications
instead of one. For an accumulating write (`+=`, an atomic add, a counter, a histogram, a
free-list pointer bump) the final value is wrong by a factor of P; for an idempotent write
(`x = f(inputs)` with unchanged inputs) it is accidentally correct.
*Status:* **assumed, never checked.** This is the single most consequential unverified obligation.
It is violated by construction for every workload in §2 that allocates through managed memory,
`hipMallocAsync`, or VMEM (which includes PyTorch with `expandable_segments`).

**O2 — Restore exactness.** $\rho(\sigma_i, s)|_T = \sigma_0|_T$.
*Status:* discharged for the bytes it copies, with two gaps.
* *Freed regions are skipped.* `with_inventory_check` returns `nullopt` when the address is no longer
  a live tracked allocation of at least the recorded size, and `restore()` logs and continues. Benign:
  a region that no longer exists cannot be observed.
* *Address reuse (ABA) is not detected.* The liveness predicate is keyed on
  `(base address, size ≥ recorded)`. The allocation wrappers are explicitly **not** covered by the
  per-agent replay lock (the comment in `memory_snapshot.cpp` says so), so another thread may
  `free(p)` and then `alloc()` a fresh region that lands at the same base with a size ≥ the old one
  during the window. `restore()` then writes the *old* allocation's bytes into the *new*
  allocation. Caching allocators make same-address reuse the common case rather than a rarity, and
  the window can be seconds long. **This is a latent silent-corruption bug**; the fix is a
  monotonically increasing generation counter per inventory slot, compared at snap and at restore.

**O3 — Isolation.** No actor other than $k$ writes $T$ during $[\mathrm{snap}, \sigma_P]$.
*Status:* partially discharged. Enumerate the actors:

| Actor | Handled? | Mechanism / gap |
|---|---|---|
| Another replay on $a$ | yes | per-agent writer lock |
| A non-replayed dispatch on $a$, not yet submitted | yes | it must take the reader lock across its submit |
| A non-replayed dispatch on $a$ already in flight | yes | agent-wide drain polls `active_async_packets()` to zero |
| SDMA / blit copies (`hsa_amd_memory_async_copy`, `hipMemcpyAsync`) | **no** | they never enter an AQL queue, so neither the lock nor the drain sees them; documented as a TODO |
| A peer agent $b$ writing $a$'s memory over XGMI | **no** | $b$ holds a *different* mutex; agent-scoped isolation is not isolation once `hsa_amd_agents_allow_access` has been called |
| Another process writing IPC-shared memory | **no** | out of scope, and undetected |
| Host CPU stores to mapped device memory (large-BAR, MI300A) | **no** | not gated |
| RDMA writes from a NIC (GPU-aware MPI, RCCL over IB) | **no** | not gated; §3 |
| Video decode / display engines writing device buffers | **no** | not gated; matters for the robotics and media cases in §2 |

The first three rows are exactly what the design documents claim, and they are correctly
implemented. The remaining seven rows are the honest scope of the isolation hole, and every one of
them is reachable from a real production workload.

**O4 — Exactly-once completion.** The application's completion signal is decremented exactly once,
after the last executed pass.
*Status:* discharged. Per-pass submissions pass `is_replay_pass=true`, which suppresses the
app-signal barrier (`queue.cpp:825`); after the loop a single barrier packet carrying
`app_completion_signal` is written to the same queue. Because the barrier is enqueued on the same
queue after all pass packets, and AQL barrier-AND with the barrier bit set orders it after prior
packets on that queue, the decrement happens-after the final pass. Deferring it out of the pass loop
is what makes early-exit and indefinite loops correct.

**O5 — Trace and identity consistency.** One logical dispatch produces one dispatch id and, from the
application's viewpoint, one dispatch.
*Status:* discharged by the reserved-dispatch-id threading, with a `ROCP_FATAL_IF` guarding misuse.
Note the deliberate consequence: P kernel-dispatch *records* share one dispatch id, distinguished
only by `replay_pass`. Any consumer that assumes dispatch ids are unique per record will
double-count; that is an output-format contract that needs stating in the tool documentation.

**O6 — Measurement validity.** Counters collected in pass $i$ and pass $j$ describe the same
computation.
*Status:* **not discharged, and not achievable in general.** It requires O1–O3 (identical inputs)
*and* comparable hardware state, and §1.4 shows the hardware state is not comparable because caches
are neither restored nor flushed. It additionally requires $\llbracket k \rrbracket$ to be
deterministic, which is false for any kernel using floating-point atomics or racy reductions.

## 6.4 Liveness: the two ways this hangs or dies

Correctness is not the only property. Two liveness hazards follow directly from taking a
non-recursive writer lock and then blocking:

**L1 — Bounded drains abort the process.** `replay_wait_or_fatal` gives 12 slices of 5 s and then
calls `ROCP_FATAL`; `replay_drain_agent_or_fatal` has a 60 s deadline and the same outcome. Any
workload where the agent legitimately does not go idle within 60 s — a persistent kernel, a
spin-waiting cooperative kernel, a collective whose peer is blocked, a queue whose completion
handler is itself blocked — turns a profiling request into a process abort. For a multi-hour
training run under a scheduler, killing the job is a worse outcome than declining to profile. This
should be a decline-and-continue path with an error record, not a `FATAL`.

**L2 — Reentrancy deadlocks with no timeout.** `std::shared_mutex` is not recursive. While the
window holds the unique lock, any attempt on *any* thread to submit a dispatch on the same agent
blocks on the shared lock — including the replaying thread itself. So if a tool's `CONFIG`/`PASS`
callback, or a completion-handler callback the window drains, launches a kernel on the replaying
agent (directly, or indirectly via a library call that happens to do a GPU memset), the process
deadlocks permanently. Unlike L1 there is no timeout on the lock acquisition, so this is an
unbounded hang. Mitigations, in increasing order of robustness: document the prohibition; set a
thread-local "in replay window" flag and make the reader-lock path fail fast with a diagnostic when
it is set on the same thread; use `try_lock_for` on the reader side so the failure is bounded.

## 6.5 A TLA+ model that is worth writing

The protocol is small enough to model-check exhaustively, and the interesting bugs above are
schedule-dependent, which is exactly what TLC finds and unit tests do not. Proposed specification:

*Constants:* `Agents`, `Threads`, `Regions`, `MaxPasses`.

*Variables:* `lock[a] ∈ [mode: {free, shared, unique}, holders: SUBSET Threads]`,
`inFlight[a] ⊆ Dispatches`, `mem[r] ∈ Nat` (a version number per region),
`snapshot[a] ∈ [Regions → Nat] ∪ {NoSnap}`, `live[r] ∈ BOOLEAN`, `gen[r] ∈ Nat`,
`sig[d] ∈ Nat`, `pass[a] ∈ Nat`, `pc[t]` per thread.

*Actions:* `AcquireShared`, `AcquireUnique`, `SubmitNormal`, `DrainQueue`, `DrainAgent`, `Snap`,
`PassSubmit`, `KernelWrite` (bumps `mem[r]` for `r ∈ W`), `Restore`, `SignalOnce`, `ReleaseLock`,
plus the adversarial actions that model the isolation hole: `AsyncCopy(r)`, `PeerWrite(r)`,
`FreeRegion(r)`, `AllocRegion(r)` (with and without `gen` bumping), `Finalize`.

*Invariants worth checking:*

1. `AtMostOneWindow`: at most one thread is between `Snap` and `ReleaseLock` per agent.
2. `NoForeignWriteInWindow`: while a window is open on `a`, no `KernelWrite` from a non-pass
   dispatch on `a` occurs. (Expected to **hold** with only `SubmitNormal`; expected to **fail** the
   moment `AsyncCopy` or `PeerWrite` is enabled — that failure is the formal statement of the
   documented gap, and having TLC produce the counterexample trace is more convincing than prose.)
3. `SignalExactlyOnce`: `[]( sig[d] decremented at most once )` and
   `<>( sig[d] decremented )` — safety and liveness for O4.
4. `RestoreOnlyLiveSameGeneration`: `Restore` never writes a region whose `gen` changed since
   `Snap`. (Expected to **fail** today; this is the ABA bug of O2, and the counterexample is a
   three-step trace: `Snap`, `FreeRegion(r)`, `AllocRegion(r)` at the same address, `Restore`.)
5. `NoDeadlock`: `[]<>(lock[a].mode = free)`. (Expected to **fail** when a pass callback is modelled
   as issuing `AcquireShared` on the replaying thread — the formal statement of L2.)
6. `FinalStateEquivalence`: with all adversarial actions disabled and `W ⊆ TrackedRegions`, the
   final `mem` equals the single-execution `mem`. This is the machine-checked version of the
   soundness claim, under exactly the assumptions O1–O3 name.

The value here is not the proof; it is that invariants 2, 4 and 5 are *expected to fail* and the
failing traces are the precise, reviewable statements of three known defects. A model where the
known bugs show up as counterexamples is a model you can trust for the parts that pass. Apalache can
take the same spec for symbolic checking if the state space with 3 threads / 2 agents / 3 regions
proves too large for TLC.

## 6.6 What to test, in priority order

Model checking covers the protocol; the implementation needs differential and property-based tests.
Ordered by (probability of catching a real defect) × (severity):

**Tier A — obligations that are currently assumed.**

1. *Untracked-write detector.* Instrument a test kernel that writes one `hipMalloc` buffer and one
   `hipMallocManaged` buffer. Replay with P = 4. Assert the tracked buffer equals the
   single-execution result and assert the managed buffer equals 4×. The second assertion documents
   the violation as a test rather than a paragraph, and turns into a regression test the day
   coverage is extended.
2. *ABA restore test.* Thread A replays a long kernel; thread B, during the window, frees a tracked
   buffer and immediately allocates the same size (HIP's allocator will very often return the same
   address). Assert the new allocation's contents are not overwritten. **Expected to fail today.**
3. *Async-copy race test.* Thread B issues `hipMemcpyAsync` into a tracked buffer during the window.
   Assert the buffer holds the copied data afterwards. **Expected to fail today**, and it is the
   cheapest possible demonstration of the SDMA gap.
4. *Reentrancy test.* A tool `PASS` callback launches a trivial kernel on the replaying agent.
   Assert the process does not hang (i.e. that the mitigation from L2 exists). **Expected to hang
   today** — so it needs a watchdog-based harness, or it must be written after the mitigation.

**Tier B — protocol behaviour under concurrency.**

5. Multi-threaded submission: N threads dispatching on one agent while one dispatch replays; assert
   every non-replayed dispatch's output is intact (the in-tree
   `tests/kernel-replay-concurrency/` client is the right shape; it needs to be run under stress and
   with thread-sanitizer).
6. Multi-GPU: concurrent replays on two agents; assert independence, then repeat *with*
   `hsa_amd_agents_allow_access` enabled and a peer-writing kernel, which should fail and thereby
   document the $U_{peer}$ hole.
7. Teardown mid-window: `abort()`/`exit()` from another thread during a window; assert no
   use-after-free in the fini path (the code has explicit `get_fini_status()` guards; they deserve a
   test that exercises them).
8. Queue creation/destruction during a window (the agent drain deliberately re-reads the live queue
   set under a brief lock; that is worth a stress test).

**Tier C — differential validation at application level.**

9. *Numerical equivalence under replay.* Run an application twice — once clean, once with
   `--kernel-replay-beta-enabled` — and compare application output. Bit-exactness is the right
   oracle only for kernels whose nondeterminism is excluded by construction; for everything else
   use a tolerance. §7 specifies per-application oracles.
10. *Counter pass-invariance (the canary).* This is the highest-value new test, and it should become
    a product feature rather than a test: include one cheap counter (e.g. `SQ_WAVES`, or
    grid-size-derived wave count) in **every** counter group, so it is collected in every pass. If
    its value differs across passes for the same dispatch id, then either the inputs were not
    identical (O1/O3 violated) or the kernel is nondeterministic. Either way the user must be told,
    because every derived metric that combines groups is then suspect. Linux `perf` does the moral
    equivalent by reporting `time_enabled/time_running` so the user can see multiplexing-induced
    error; kernel replay currently reports nothing.
11. *Repeated-group self-validation.* Run P+1 passes where the extra pass repeats pass 0's counter
    group, and compare. This is a direct empirical test of "the passes saw identical inputs",
    per-dispatch, at the cost of one extra pass. Combined with a device-side hash of the tracked
    footprint after each pass (cheap: HBM bandwidth, not host-link bandwidth), it detects untracked
    input drift, concurrent-writer interference, and kernel nondeterminism, and distinguishes them:
    identical hash + differing counters ⇒ hardware/measurement nondeterminism; differing hash ⇒
    state leakage. **This should be a supported mode** (`--kernel-replay-verify`), because it
    converts an unprovable assumption into a per-run measurement.

**Tier D — fault injection.** Force `snap()` to fail on the *k*-th region (memory pressure path),
force a `restore()` copy to fail, force a drain to time out, force `pass_count_cb` to return 0 with
and without a continue callback, and make a tool callback throw. Each of these paths exists in the
code and each currently ends in either a graceful decline or a `ROCP_FATAL`; a test matrix that pins
which is which prevents a future refactor from silently converting a decline into an abort.

## 6.7 The trusted base, stated as assumptions

For the record, the soundness claim of §6.2 holds only under these assumptions. Each is listed with
its status so that nothing is quietly load-bearing:

| # | Assumption | Status |
|---|---|---|
| A1 | The kernel's write set lies entirely within tracked coarse-grained VRAM owned by the replaying agent, or within module-scope variables | **unchecked**; violated by managed/async/VMEM allocations |
| A2 | No SDMA copy, peer-agent kernel, other process, host store, or NIC RDMA writes tracked memory during the window | **unchecked**; documented gap |
| A3 | No tracked allocation is freed and re-allocated at the same base address during the window | **unchecked**; ABA bug |
| A4 | Tool callbacks do not submit GPU work on the replaying agent | **undocumented**; unbounded hang if violated |
| A5 | The agent goes idle within 60 s of the window opening | **enforced by abort**, not by declining |
| A6 | Kernel behaviour depends only on memory contents, not on hardware state (caches, clocks) | true for results, **false for measurements** (§1.4) |
| A7 | The kernel is deterministic given identical inputs | false in general; determines whether O6 is meaningful |
| A8 | Host RAM can hold the full tracked footprint | enforced by graceful decline |

A reader who wants a one-line summary of this section: **O4 and O5 are proved by construction, O2
and O3 are half-proved with named holes, and O1 and O6 are assumptions that the mechanism has no
means of checking — which is why §6.6's canary and verify modes matter more than any amount of
additional documentation.**
