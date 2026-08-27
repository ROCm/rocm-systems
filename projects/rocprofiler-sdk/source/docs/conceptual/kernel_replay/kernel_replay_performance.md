(kernel-replay-performance)=
# Kernel Replay Performance Assessment

This page states what kernel replay actually costs, why the current regression tests cannot detect
the regressions that would matter in production, and how the cost behaves as device capacity and
host-link bandwidth diverge across accelerator generations. It is an assessment rather than a guide:
the reasoning below is derived from the code and from the shape of the cost model rather than from
any particular measurement.

## The cost model

Replay expands one dispatch into a drain, a snapshot, a loop of passes, and a restore between
passes. The snapshot is a full copy of the agent's tracked device memory into host RAM.

```text
for each replayed dispatch:
      drain          every queue on the agent to idle
      snap           whole tracked footprint, device -> pageable host RAM
      for each pass:
          kernel     re-execute the dispatch
          restore    whole footprint, host -> device   (skipped after the last pass)
      free           release the host copy
```

Three properties of that shape drive everything else:

**The unit of work is the agent, not the kernel.** `snap()` walks the allocation tracker's whole
inventory for the agent. It makes no attempt to determine which allocations the kernel actually
reaches, so a kernel that writes one megabyte still costs a copy of every tracked byte the process
has allocated on that GPU.

**The unit of repetition is the dispatch.** `snap()` is called from the replay window in
`hsa/queue.cpp`, once per replayed dispatch, and the returned `device_snapshot_t` is destroyed when
that dispatch's window ends. Nothing is carried across dispatches.

**The copies are full copies.** There is no dirty-page tracking; this is documented as future work in
[Memory snapshot and restore](kernel_replay_memory_snapshot.md).

So the bytes crossing the host link over a run are approximately:

```text
bytes ~= dispatches x tracked_footprint x passes
```

per dispatch: one snapshot plus `passes - 1` restores, which is `passes` copies of the footprint.
This is the same model the existing tests encode in
`tests/kernel-replay-perf/perf_cost_model.py`, so the model itself is not in dispute. What follows
is about the range over which it has been exercised.

## Why the current tests cannot catch a production regression

The four kernel-replay perf tests run at 16 to 64 MB of ballast and 4 to 16 dispatches. The largest,
`test-kernel-replay-perf-scaling`, is 64 MB with 8 dispatches at 5 passes, so it moves about
**2.5 GB** in total.

A deliberately modest real workload -- 1 GB resident, 10,000 dispatches, 5 counter groups -- moves
about **50 TB**. That is roughly four orders of magnitude more traffic, and it is four orders of
magnitude in *both* factors that matter, footprint and dispatch count.

Nothing in the suite covers the region where kernel replay would actually be used, which has several
consequences:

- A regression that only appears at large footprint (for example, a change that makes per-region
  host allocation quadratic, or that stops reusing a staging buffer) is invisible at 64 MB.
- A regression in per-dispatch fixed cost is diluted: at 8 dispatches, a 10 ms per-dispatch
  regression is 80 ms against a ceiling of roughly 7300 ms.
- The cost model's own margins are generous by design -- a 4 GB/s floor and a 10x overhead
  multiplier -- so the assertions only fire on catastrophic breakage, never on drift. A change that
  made replay twice as slow would pass every test in the suite.

The consequence worth stating plainly: **the suite is a smoke test that replay still works, not a
performance gate.**

## Why the scaling baseline is P=2, not P=1

The scaling tests compare a baseline pass count against a higher one. The baseline is **P=2**, and
that choice is load-bearing.

A `pass_count_cb` returning `1` means the dispatch is **not replayed**: it takes the ordinary
single-dispatch path with no snapshot and no restore (see `experimental/kernel_replay.h`). Timing
P=1 therefore times a bare dispatch, with no snapshot and no restore in it at all.

Replay cost is dominated by the **fixed** cost of snapshot and restore rather than by the pass
count, so a P=1 baseline does not measure pass scaling -- it measures the price of enabling replay
at all. That ratio is bounded by how long a snapshot takes relative to a bare dispatch, which is
orders of magnitude and cannot be expressed as a linear-scaling cap.

The baseline is therefore P=2: the smallest pass count that is actually replayed. Both sides of the
comparison then pay the same fixed snapshot cost, the ratio sits near unity, and it moves only when
**per-pass** cost changes -- which is the regression the test exists to catch.

### The pass count barely moves wall time

The medians above are **not monotonic**: P=2 (836 ms) measures above both P=3 (763 ms) and P=5
(811 ms). That is not an inversion in the implementation, it is variance. Within a single
configuration the spread reaches 66% (P=2 ranges 745 to 1240 ms), while the difference *between*
pass counts is about 3%. The noise is an order of magnitude larger than the effect, so any single
sample can order the configurations arbitrarily. This is why the tests compare medians rather than
single runs.

The bytes genuinely do scale -- one snapshot plus N-1 restores of the ballast per dispatch, so
1.0 GB at P=2, 1.5 GB at P=3 and 2.5 GB at P=5 across 8 launches. Wall time not tracking that means
**DMA bandwidth is not the bottleneck**. Decomposed against the measurements:

- **Fixed, per dispatch (~90 ms x 8 = ~720 ms):** agent-wide sibling-queue drain, snapshot inventory
  discovery over the tracked allocation map, and per-agent writer-lock acquisition. Paid once per
  dispatch regardless of the pass count.
- **Per pass (~3 ms x 8 = ~25 ms per additional pass):** the actual restore copy.

That is roughly a **30:1 fixed-to-variable ratio**, which is why P=2 through P=5 all cluster in the
750-850 ms band. The fixed cost is replay's own work rather than process startup or HIP
initialization: the same binary at P=1 completes the timed loop in 2.9 ms, because the timer covers
only the in-app loop.

The practical consequence is that these tests are a **weak per-pass regression detector**. With the
fixed cost anchoring both sides of the ratio, a per-pass cost regression would have to be very large
to push a ~1.0 ratio past a 5.0 cap. Catching per-pass drift wants either the slope across the
nightly `--passes 2 3 5 8` sweep or the direct snapshot/restore bandwidth measurement in
`snap_bandwidth.cpp`, not end-to-end wall time.

### Alternatives considered

- **(b) Keep the P=1 baseline, make the scaling check advisory** behind
  `ROCPROFILER_PERF_STRICT_CEILING`, the way the absolute ceiling already works. Rejected: it
  removes the only per-commit scaling signal, leaving nothing that fails on a real regression.
- **(c) Keep P=1 and raise the caps** to measured values (~400x and ~100x). Rejected: it bakes a
  machine- and build-type-specific constant into the gate, and a genuine per-pass regression would
  hide comfortably under a 400x cap.

### Known gaps

- These numbers come from a **Debug** build. Release will differ, so the caps are deliberately loose
  rather than tuned to one machine.
- `snap_bandwidth_meets_floor_*` are timing-sensitive and can fail under concurrent GPU load while
  passing on rerun. They gate a bandwidth floor, not correctness.
- The fixed-cost-dominates result above is itself a finding: at these sizes the suite barely
  exercises per-pass scaling, which reinforces the smoke-test caveat in the previous section.

## Specific performance problems in the current implementation

These are candidates identified by reading the code. Each needs measurement before it is treated as
a defect, and the benchmark harness described below is what would measure them.

### Host staging is pageable

`mem_block_t::host_copy` is a `std::vector<char>`. `hsa_memory_copy` into pageable host memory
cannot DMA directly; the driver stages through its own pinned buffers, which costs an extra copy and
serializes against that staging capacity.

The suggestive evidence is in the tests themselves: `snap_bandwidth.cpp` sets its floor at
**4 GB/s** and describes it as conservative. PCIe Gen5 x16 is roughly 64 GB/s theoretical and
around 50 GB/s achievable with pinned memory. A floor an order of magnitude below the link rate is
what one would expect from pageable staging. Whether the achieved rate is actually near the floor or
comfortably above it is not currently reported anywhere, because the tests assert a bound rather
than record the value.

Allocating the staging buffer from a host memory pool would be the obvious experiment.

### The staging buffer is allocated and freed per dispatch

Because the snapshot is a local that dies with the dispatch, every replayed dispatch performs a
`resize()` per tracked region and frees it again. At a gigabyte-scale footprint this is a large
allocation churn on every dispatch: first-touch page faults on the way in, and return-to-OS on the
way out. A per-agent staging buffer retained across dispatches, sized to the high-water footprint,
would remove all of it.

### The whole footprint is copied regardless of what the kernel writes

This is the largest single lever. Real kernels typically write a small fraction of resident memory:
an attention kernel touches activations while weights sit resident and unmodified. Dirty-page
tracking, or even a coarse per-allocation write-set derived from kernel arguments, would change the
cost from "footprint" to "working set" -- potentially orders of magnitude.

### The agent-wide drain and the exclusive writer lock serialize the GPU

Before snapshotting, the replay window drains every queue on the agent and holds a per-agent writer
lock for the whole window. Any application that overlaps compute with copies, or that uses several
streams, loses that overlap entirely for the duration.

Microbenchmarks with one stream cannot see this at all, and every existing perf test uses one
stream. The concurrency test at `tests/kernel-replay-concurrency/` runs two streams but asserts
correctness, not throughput.

### Host RAM high-water is never measured

The peak host memory a replayed run requires is at least the tracked device footprint. No test
records it, so the point at which replay becomes impossible on a given host is unknown.

## Behavior across architectures

Device capacity and the host link matter more than compute here, because the cost is a memory copy
over the host link. Two consequences follow from that alone, without reference to any particular
part:

- **Host capacity, not device capacity, is the hard limit.** A snapshot of a full-VRAM application
  needs host memory equal to the device footprint. On high-capacity accelerators that exceeds the
  host memory typically provisioned per GPU in a dense node, and on multi-device packages replaying
  on several devices at once is bounded by host capacity before anything else.
- **Snapshot time scales with footprint over host-link bandwidth.** Since the copy happens once per
  replayed dispatch, a large footprint turns into seconds of added latency per dispatch.

The trend matters more than any single number. Device memory bandwidth and capacity are growing far
faster than the host link, so the ratio between them widens with each generation. Kernel replay's
cost is paid on the slow side of that ratio while the value it delivers scales with the fast side,
which means **the technique gets relatively more expensive on each generation** unless the amount
copied is decoupled from the footprint.

The practical reading: kernel replay is well suited to small- and medium-footprint work, and
application replay remains the better answer for large-footprint runs until dirty tracking exists.
That trade-off should be measured per architecture rather than asserted, which is what the harness
is for.

## Applicability to real applications

Three properties of the mainstream ROCm machine-learning stack interact badly with replay today, and
they are worth knowing before interpreting any benchmark:

- **HIP graphs.** Replay declines graph launches: the interceptor warns once and the graph runs
  un-replayed. Frameworks that capture graphs -- vLLM and PyTorch both do by default in their common
  configurations -- would therefore see replay silently do nothing for the captured region.
- **Stream-ordered / caching allocators.** `hipMallocAsync` and pool allocators are not hooked by
  the memory tracker, so that memory is neither snapshotted nor restored. PyTorch's caching
  allocator is exactly this shape.
- **Multi-GPU.** There is no multi-GPU test, and no cross-process coordination. A kernel that takes
  part in a collective must not be replayed.

None of these are defects in the sense of a wrong answer inside the supported envelope; they are the
boundary of that envelope. They are listed here because a benchmark run against a real framework
would otherwise produce numbers that look fine while measuring almost nothing.

## What the benchmark work adds

- Workloads that span the real range of footprint and dispatch count instead of stopping at 64 MB
  and 16 dispatches, including the shapes that break the single-stream assumption.
- Reporting of measured values -- achieved snapshot bandwidth, bytes copied, host RAM high-water --
  rather than only pass/fail against a loose ceiling, so drift is visible.
- A path to run the same workloads across accelerator generations and compare, since the conclusion
  above is architecture-dependent.

## See also

- [Memory snapshot and restore](kernel_replay_memory_snapshot.md) -- what is captured and excluded
- [Concurrency and isolation](kernel_replay_concurrency_and_isolation.md) -- the drain and the lock
- {ref}`using-kernel-replay-rocprofv3` -- when to prefer application replay
