# Range replay performance

Range replay's cost model is the reason the service exists, so it is also what its performance
tests measure.

## The cost model

A replayed range pays three kinds of cost:

| Cost | Scales with | Paid |
| --- | --- | --- |
| Replay window: agent drain, entry and exit snapshots of the tracked inventory, restore at range exit | Tracked device memory footprint | Once per range |
| Restore of the entry snapshot before each replayed pass | Tracked device memory footprint | Once per pass |
| Dispatch execution and kernarg staging | Dispatches x passes | Per dispatch, per pass |

Kernel replay opens a window per dispatch, so a phase of `K` dispatches pays the window cost and
the per-pass restores `K` times. Range replay opens one window for the whole range and pays them
once. That difference is the service's entire justification, and it grows with `K`.

It also means the two services are not interchangeable at small `K`. For a single dispatch the two
do the same work, and range replay adds the bookkeeping of recording the range for no benefit. The
win begins as soon as a range holds more than one dispatch and increases from there.

## What range replay shares with kernel replay

Range replay takes and restores its snapshots with kernel replay's snapshot code
(`kernel_replay/memory_snapshot`). Anything that makes a kernel replay snapshot or restore cheaper
--- fewer regions captured, cheaper host staging, faster copies --- lowers range replay's per-range
and per-pass costs the same way, with no change on the range replay side.

The window around the snapshots is range replay's own, and follows kernel replay's:

- the barrier that drains the range's queue before the entry snapshot completes on a signal taken
  from the SDK's signal pool, rather than on one created and destroyed per range;
- the kernarg block the passes read is kept for the agent's next range (below);
- each range's phases are logged at INFO (below).

Two differences from kernel replay matter when changing the shared snapshot code:

- Range replay holds two snapshots of the agent at once: the entry snapshot for the whole range,
  and the exit snapshot from the close until the exit restore, plus a third while the divergence
  check runs. Host staging recycled from one snapshot to the next of the same agent serves the
  entry snapshot. It serves the exit snapshot only if the staging the previous range released is
  still held when the exit snapshot is taken, which a policy that frees every buffer a snapshot did
  not take, as soon as that snapshot finishes, does not do.
- The divergence check hashes the snapshots' host copies. A snapshot backed by device memory would
  have to be read back, or hashed on the device, before the check could compare it.

### The kernarg block is kept between ranges

Each pass reads its arguments from one kernarg block the executor allocates from the agent's
kernarg pool and maps for the agent. Rather than freeing it when the range closes, the executor
keeps it, and the next range on the same agent reuses it when it is at least as large as that range
needs and no more than twice as large. A block outside that bound is freed, and the range allocates
its own. Each agent holds at most one block, it is kept only after the last pass that read it has
drained, it never enters a snapshot (the memory tracker does not track kernarg pool memory), and it
is released with the process.

### Reading the phase timings

With `ROCPROFILER_LOG_LEVEL=info`, each replayed range logs two lines:

```text
range replay: range <id> entry: drain <t> ms, snapshot <t> ms (<bytes> bytes in <n> regions)
range replay: range <id> (<n> dispatches) phases: lock+drain <t> ms, exit snapshot <t> ms
    (<bytes> bytes in <n> regions), kernarg staging <t> ms (<bytes> bytes, reused|allocated|none),
    <n> passes <t> ms, <n> restores <t> ms, verify <t> ms|off, exit restore <t> ms, window <t> ms
```

The first is logged when the entry snapshot is taken, before the range's first dispatch reaches the
GPU, and the second when the window closes. The restores grow with the pass count, the passes with
dispatches times passes, and everything else is paid once per range, which is the split the scaling
and amortization tests below depend on. A kernarg staging block that is `allocated` on every range
of a steady loop means consecutive ranges' kernarg footprints differ by more than a factor of two.

## What the tests measure

Both live in `tests/range-replay-perf/` and are registered only when
`ROCPROFILER_BUILD_NIGHTLY_PERF_CTESTS` is on, matching `tests/kernel-replay-perf/`. Wall-time
tests on a shared, multi-tenant CI runner measure the neighbours as much as the code, so they are a
trend across nightly runs rather than a per-commit gate.

### `test-range-replay-perf-scaling`

Holds the range fixed and varies the pass count, `P=2` against `P=5`, bounding how fast wall time
may grow.

The baseline is `P=2` rather than `P=1`. A pass count below 2 means the range is recorded and
closed without any re-execution, so a `P=1` sample times the application rather than the replay,
and a ratio against it reports the cost of enabling replay at all instead of per-pass scaling.
Comparing two replayed configurations keeps the once-per-range fixed cost on both sides of the
ratio, which is what makes the ratio a per-pass measurement. Because that fixed cost does not scale
with `P`, the expected ratio is well below `P_high / P_base`.

### `test-range-replay-perf-amortization`

Holds the pass count and the memory footprint fixed and sweeps the dispatches per range across
`1, 2, 4, 8`. Perfect amortization puts the ratio of the longest range to the shortest near `1.0`;
paying the window cost per dispatch would put it near `8.0`. The cap sits between the two.

This is the check with no kernel replay analogue, and no other test in the suite can see what it
sees: the scaling test holds the dispatch count constant, so a regression that reintroduced
per-dispatch windows would pass it unchanged.

For the sweep to mean anything the dispatches have to be cheap relative to the snapshot. The
workload's kernel therefore writes a small fixed slice rather than the whole ballast --- sizing the
working set to the footprint would tie the two costs together, and the sweep would measure GPU work
growing linearly with the dispatch count instead of whether the window cost was amortized.

## Why a declined range cannot be timed

Every decline path abandons the range during recording: no snapshot is taken, no pass loop runs and
nothing is restored. A declined range is therefore dramatically *faster* than a replayed one.

A perf harness that only timed the application would read a newly introduced decline --- a
regression that made ordinary ranges ineligible --- as a large improvement, and go green. So the
numbers are only collected alongside proof that they measure a replay: the tool asserts that every
range reached `CLOSE` with status `REPLAYED` and recorded the dispatch count the application
issued, and the python drivers refuse to record a sample otherwise.

## Interpreting a failure

A ratio breach is a starting point, not a verdict. Check in order:

1. Whether the run was noisy. Each configuration is sampled several times and compared on medians,
   and the reported spread says how much to trust the number.
2. Whether the footprint changed. Both costs move with the tracked inventory, so a change in what
   the memory tracker considers live moves the fixed cost without any replay regression.
3. For an amortization breach specifically, whether something began opening a window per dispatch
   again --- for example a snapshot or drain that moved inside the recorded dispatch loop rather
   than staying at the range boundary.
4. Which phase moved. Rerun the failing configuration with `ROCPROFILER_LOG_LEVEL=info` and compare
   the per-range phase lines against a good run: a change in the snapshot or restore times points at
   the shared snapshot code, a change in the passes at dispatch execution.
