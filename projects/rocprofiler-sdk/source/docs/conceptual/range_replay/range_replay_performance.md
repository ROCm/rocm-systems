# Range replay performance

Range replay's cost model is the reason the service exists, so it is also what its performance
tests measure.

## The cost model

A replayed range pays two kinds of cost:

| Cost | Scales with | Paid |
| --- | --- | --- |
| Replay window: agent drain, entry snapshot of the tracked inventory, restore between passes and at range exit | Tracked device memory footprint | Once per range |
| Dispatch execution and kernarg staging | Dispatches x passes | Per dispatch, per pass |

Kernel replay opens a window per dispatch, so a phase of `K` dispatches pays the window cost `K`
times. Range replay opens one window for the whole range and pays it once. That difference is the
service's entire justification, and it grows with `K`.

It also means the two services are not interchangeable at small `K`. For a single dispatch the two
do the same work, and range replay adds the bookkeeping of recording the range for no benefit. The
win begins as soon as a range holds more than one dispatch and increases from there.

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
