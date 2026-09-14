# RFC: asynchronous MMA execution in RocJITsu

**Proposal:** offload expensive MMA handlers to persistent CPU helpers. Keep
async opt-in: large gfx1250 WMMA, plus K32 and selected CDNA4 MFMA with
cached admission. Exclude memory offload. This accelerates functional simulation; it does not model GPU timing.

## Core idea

Issue in program order; execute and complete independent MMAs out of order.
A scoreboard makes consumers and overwrites wait for conflicting work. The
issuer executes safe instructions, including other MMAs. Helpers are shared
across issuers/XCDs.

Host out-of-order execution has a finite window; it cannot generally expose a
later simulated instruction across an MMA's long nested loops. Helpers supply
additional cores. Coroutines alone would not provide that parallelism.

## What makes it practical

| Mechanism | Why it is needed |
|---|---|
| **Persistent shared pool; immediate fallback** | Avoid per-job thread creation and per-CU pools. Exhaustion reads a fork guard and at most two capacity words; reservation retries are bounded. On failure, resolve dependencies and execute inline without waiting for unrelated jobs. |
| **Short warm wait, then sleep** | Poll for 512 pauses, then use a private Linux futex (`atomic::wait` fallback). Warm empty-job latency falls **10.51 to 1.00 us**. Sleep-intent bits prevent missed/redundant wakeups; idle helpers park. |
| **Small payload, stable storage** | Publish instruction/wave pointers and FP environment; no matrix copy. Allocate lazy registers before publication, retain instructions until completion, and destroy them on their allocating thread. |
| **Scoreboard; independent completion** | Track register reads/writes and aliases. Enforce read-after-write, write-after-read and write-after-write dependencies, while allowing shared inputs. Reclaim completed jobs without waiting for an older unrelated MMA. |
| **Bounded windows** | Drain at control flow, mode changes, wait-idle, unsupported operations and CU rescheduling. The prototype issues at most 32 instructions from one wave per step, then joins; it does not overlap different waves on one CU. |
| **Admission; reserve issuer work** | Decode up to eight following instructions without executing them; stop at dependencies or unsafe boundaries. Offload an independent group's prefix and reserve its final MMA for the issuer. Preserve the earliest reservation to avoid offloading all useful work. |
| **Cache lookahead decodes and accepted/rejected plans** | Unchanged loops reuse verdicts within a bounded per-CU working set (16K plans, 64K decodes). Cache pressure clears retained history; revisits rebuild. Revalidate bytes after I-cache invalidation. Warm lookup: **7-8 ns/site**. Ordinary execution decoding is unchanged. |
| **ISA gating; lazy activation** | Unsupported targets retain ordinary execution. Cached encoding filters skip the async entry for non-MFMA instructions on CDNA4 and non-candidates on gfx1250. Create/poll windows only after an offload opportunity; otherwise avoid full scoreboard costs. Whole-binary layout effects can still change timings. |

Cold wakeups and immediate joins still cost more than warm empty transport:
dense K128 submit-and-join overhead measured **3.44 us** with warm waits.

## Which operations qualify?

A useful approximate cost test is:

```text
min(handler time, independent issuer work before the next wait)
    > handoff + bookkeeping + added contention/locality cost
```

An operation also needs these properties:

- **Expensive execution and overlap.** Gfx1250 K32/K64/K128 callbacks measured
  roughly 32/61/122 us. Immediate consumers can erase the gain; GPU latency or
  K alone cannot determine eligibility.
- **Pure register semantics.** Dependencies must be enumerable and independent
  operations must commute. The prototype requires full EXEC, fixed mode and
  direct register addressing.
- **No precise rollback or per-instruction observation.** Unexpected failures
  drain outstanding jobs but cannot undo later results. Debugging, traps and
  plugins requiring ordered architectural snapshots disable this path.
- **Spare CPU capacity.** Helpers have a tunable shared budget. Added-core gains
  do not establish efficiency at a fixed core count.

Scalar/vector ALU handlers are usually too short. Loads/stores add aliasing,
memory ordering, cache ownership, faults and wait semantics. Ordered retirement
cannot undo an already performed store or load; explicit GPU waits alone are
insufficient. Memory offload increased dispatch/wall time by **20.4%/15.4%**
even with 32 independent ALU steps per load and was removed.
Sparse and multi-block MFMA remain separate experiments.

Keep synchronous before/after hooks separate from async issue/retirement: their
state-inspection guarantees differ. Plugins opt in once; any plugin requiring
synchronous observation disables offload for the group. Throughput and kernel
logging handle both pairs, including synchronous fallback.
Callbacks run on the issuer; register hooks can run concurrently on helpers.
Throughput retains counts and dispatch timing, marks affected handler timing
invalid and emits null timing-derived values. ConSan keeps synchronous execution:
its dependency-event reads can race with wait retirement, and diagnostics need
the originating instruction context.

## Evidence and proposed policy

Six rotated rounds, **eight CU workers in every policy**, plus four shared
helpers for async, on reserved physical cores 80-95 via `agent-reserved-run`.
Host: Threadripper PRO 9995WX; Clang 23, `-O2`, no LTO; measured 2026-09-13.
Gluon uses original 1024 x 1024 x 1024 GEMMs; K32 is f16/bf16, K64/K128 FP8. Dispatch is
throughput-plugin active time; process wall includes startup and numerical
checks. Entries are **dispatch / wall changes versus ordinary execution**;
negative means faster:

| Workload | Unrestricted async | Async with cached admission |
|---|---:|---:|
| Gluon K32 f16 | -6.2% / -1.5% | **-6.6% / -1.9%** |
| Gluon K32 bf16 | -6.6% / -2.4% | **-7.5% / -2.5%** |
| Gluon K64 FP8 | **-20.9% / -6.8%** | -20.3% / -6.5% |
| Gluon K128 FP8 | **-23.0% / -7.5%** | -21.8% / -7.2% |
| IREE K32 f16 | +3.9% / +2.8% | **-2.3% / -1.0%** |
| HIP K32 dependent chain | +8.3% / +6.8% | **+0.3% / 0.0%** |

Unrestricted K128 reduces dispatch **1.248 to 0.961 s**, wall **3.880 to 3.590 s**.
Admission cuts IREE submissions **40,344 to 3,142**; dependent HIP submits none
and approaches ordinary performance. A later six-round check still found about
0.9% dispatch / 0.6% wall overhead on that chain with zero offloads: cached
admission and the async issue path are not free.

**On gfx1250, use admission for K32 and unrestricted offload for K64/K128.** The table tests
admission on all sizes (mode 3); the proposed policy gates only K32:

```sh
RJ_MATRIX_COEXEC=4 RJ_ASYNC_WMMA_MIN_K=32 RJ_MMA_ADMISSION=2 \
RJ_MMA_SHARED_HELPERS=4 RJ_MMA_HELPERS=7 RJ_MMA_LOOKAHEAD=8
```

The measured admission column uses the same helper limits and lookahead.
On gfx1250, mode 4 alone enables neither K32 nor admission.
Gfx1250 large-MMA admission differences are small and have overlapping ranges.

The **162 gfx1250 timed processes** passed numerical checks and identical instruction
signatures. Broader Gluon suites did not establish an aggregate win; K32 Gluon
uses 5-6% more CPU time. Whole-machine scaling, hardware numerics, full-runtime
concurrency qualification and concurrent-VM scaling remain outstanding. Inherited
helpers fail closed after fork; the child executes synchronously. The shared pool
has process lifetime, including detached-runtime teardown.

## CDNA4 extension

Mode 4 also selects dense FP16, FP8/BF8 and scaled/unscaled f8f6f4 MFMA on
gfx950, with cached admission by default. Other targets retain their previous
selection. Explicit `RJ_ASYNC_MFMA=0` disables MFMA; `RJ_MMA_ADMISSION=0`
allows unrestricted offload for experiments. Multi-block MFMA stays opt-in.

The original Triton matmul kernel, 1024³, gives these six-round results using
the same eight-worker/four-helper setup. FP8 means E4M3; BF8 means E5M2.

| Input / emitted MFMA / output tile | Dispatch change | Process wall change |
|---|---:|---:|
| FP8 / 16x16x32 / 64x64 | -28.2% | -8.7% |
| BF8 / 16x16x32 / 64x64 | -29.1% | -9.5% |
| FP8 / 16x16x128 / 64x64 | -39.5% | -11.6% |
| BF8 / 16x16x128 / 64x64 | -38.8% | -11.1% |
| FP8 / 32x32x64 / 128x128 | -21.7% | -4.7% |
| BF8 / 32x32x64 / 128x128 | -21.8% | -4.4% |

Deltas are medians of paired percentage changes. All 132 final-binary runs
passed numerical and instruction-signature checks. The source-built IREE
MXFP4 kernel improves **42.0% dispatch / 37.6% wall**.

The 32x32x64 FP8 opcode loses 6.4% dispatch time on a 64x64 output tile with
unrestricted offload. Admission rejects every job there; the final policy
measures -2.0% dispatch / -0.6% wall. The 128x128 tile supplies independent
accumulator groups. CDNA4 therefore needs both a handler allowlist and cached
independence checks.

Admission is still not free: the dependent HIP FP8 chain measured +3.6%
dispatch / +2.6% wall in the main run, then -0.1% / -0.3% in a six-round
follow-up; zero helpers measured +0.6% / 0.0%. Small timing differences are
noisy. The encoding filter cuts extra host instructions on a rejected FP16
kernel from 0.59% to 0.26%; it does not eliminate admission bookkeeping.
