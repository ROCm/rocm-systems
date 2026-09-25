# Range replay test roadmap

This page plans the correctness and performance tests range replay still needs, one set per
week from October 2026 through September 2027. Nothing here is implemented yet. Each week's
tests land as their own pull request, stacked on the previous week's, so every PR stays small
enough to review in one sitting and can be merged or dropped on its own.

## What is covered today

| Layer | Where | What it pins |
| --- | --- | --- |
| Unit, no GPU | `range_replay/tests/` (81 tests) | eligibility and decline bookkeeping, kernarg layout, pass packets, digests, the public ABI, the single-subscriber claim, CONFIG → PASS → CLOSE ordering, the queue hook's early returns |
| Samples, GPU | `samples/range_replay/` | a replayed range with `divergence_count == 0`, a tool that opts out, a multi-queue decline |
| Integration, GPU | `tests/range-replay-local-context/` | every per-dispatch service sees the replayed dispatches and honors per-pass toggles |
| Performance, GPU, nightly | `tests/range-replay-perf/` | pass-count scaling, and amortization of the window cost over dispatches |

## What is missing

Most decline reasons are asserted only as bookkeeping: nothing on hardware drives an application
into them and checks that the range is declined and the application is unaffected. Three bugs
fixed in #10864 have no regression test:

- device-writing copies went unseen when nothing traced memory copies;
- `hsa_amd_memory_async_batch_copy` bypassed the decline;
- a graph launched from another thread during a replay window could corrupt application data.

The recurring cost range replay imposes on runs that never open a range is not measured either.

## The obvious ones first

These are the first eight weeks. Each guards a bug that already happened or a decline path users
will hit first.

| Week of | ID | Kind | Test |
| --- | --- | --- | --- |
| 2026-10-05 | C1, C2 | correctness | A `hipMemcpyAsync` host-to-device inside a range, with **no** memory-copy tracing configured, declines with `MEMORY_COPY_IN_RANGE`; so does an `hsa_amd_memory_async_batch_copy` (skipped where the HSA runtime lacks it). The application's result is intact in both. |
| 2026-10-12 | C8 | correctness | A second thread launches HIP graphs in a loop on the same agent while the first thread's range is being replayed. Every buffer both threads wrote ends with the value each application run expects. |
| 2026-10-19 | C3, C4 | correctness | `hipMalloc` and `hipFree` inside a range decline with `ALLOCATION_CHANGED_IN_RANGE`; an empty range closes with `NO_DISPATCH`. |
| 2026-10-26 | C5, C6 | correctness | A graph launch inside a range declines with `GRAPH_LAUNCH`; another thread's kernel on the same agent while a range is open declines it with `CONCURRENT_DISPATCH`. |
| 2026-11-02 | P1 | performance | Overhead when range replay is configured but no range is ever opened: per-dispatch submission time against a run with no tool, since the interception gate is process-wide. |
| 2026-11-09 | P2, P3 | performance | `hipMemcpyAsync` throughput with range replay configured, now that the copy wrappers are installed for it; and graph-launch throughput, now that graphs no longer take the fast path while range replay is active. |
| 2026-11-16 | C9 | correctness | An in-place kernel and an atomic histogram inside a replayed range: the application's result is intact and `divergence_count == 0` with verification on. |
| 2026-11-23 | C7, C14 | correctness | A range that exceeds the recording budget declines with `PROGRAM_TOO_LARGE` and reports the full observed dispatch count; `begin` twice on one thread and `end` with no range return `INVALID_ARGUMENT` end to end. |

## The rest of the year

| Week of | ID | Kind | Test |
| --- | --- | --- | --- |
| 2026-11-30 | C10 | correctness | A kernel that updates a module-scope `__device__` variable inside a range: the variable is restored between passes and the application sees one update. |
| 2026-12-07 | C11 | correctness | Kernarg staging: 64 dispatches with distinct arguments, arguments near the 4 KiB limit, and kernels with no arguments, all replayed with the right arguments on every pass. |
| 2026-12-14 | P4 | performance | Snapshot and restore time against footprint, 16 MiB to 2 GiB; guards the slope. |
| 2026-12-21 | C12 | correctness | Open-ended passes: `pass_count_cb` returns 0 and `replay_continue_cb` stops after *k*; exactly *k* passes run. |
| 2026-12-28 | — | — | Buffer week: fix whatever the earlier weeks turned up. |
| 2027-01-04 | C13 | correctness | Several ranges in sequence with different ids, and the context stopped and restarted between them; one CLOSE per range, each with its own status. |
| 2027-01-11 | P5 | performance | A thousand one-dispatch ranges: the per-range fixed cost, against kernel replay's per-dispatch cost. |
| 2027-01-18 | C15 | correctness | Range replay and kernel replay configured in one process: neither steals the other's dispatches, and each claims its own service. |
| 2027-01-25 | P6 | performance | The cost of `ROCPROF_RANGE_REPLAY_VERIFY=1` against verification off. |
| February | C16, C20, P7, C21 | mixed | The application exits with a range still open, without hanging. A kernel reading host memory that changes between passes makes verification report divergence. Kernarg staging cost from 1 to 512 dispatches per range. A range on the queue-interposition path declines with `UNSUPPORTED_QUEUE_PATH`. |
| March | P8, P9, C19, C22 | mixed | Reader-lock contention on a four-stream application. Latency distribution of the replay window (p50, p99). PC sampling alongside range replay: collection is agent-wide and ignores toggles, as documented. A stress test with many threads opening ranges on one agent, all declining cleanly. |
| April | C17, P10 | mixed | `MULTI_AGENT` on multi-GPU runners, once CI has them. Host memory high-water mark while snapshotting large footprints. |
| May – September | — | — | One test a week from new findings and from features range replay gains, such as the multi-GPU abandonment notification; the plan is revisited on the first week of each month and this table updated. |

## Conventions for every week's PR

- Stacked on the previous week's PR, with the week and the test IDs in the title.
- GPU tests skip cleanly without a device; performance tests register only for the nightly job,
  like `tests/range-replay-perf`, and must confirm every range was replayed before accepting a
  number.
- A correctness test that drives a decline also checks the application's own result, because a
  decline must never change what the application computes.
- The PR description links the CI run and summarizes any failure: the test, what it asserted,
  the root cause, and whether it reproduces on `develop`.
