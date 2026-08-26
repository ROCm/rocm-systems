(kernel-replay-performance)=
# Kernel Replay Performance Assessment

This page states what kernel replay actually costs, why the current regression tests cannot detect
the regressions that would matter in production, and how the cost behaves on MI250X, MI325X and
MI455X. It is an assessment rather than a guide: several of the numbers below are derived from the
code and from published hardware specifications, not measured, and each one says which it is.

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
over the host link.

- **MI250X (gfx90a)** -- 64 GB HBM2e per GCD, two GCDs per package, presented as two devices. On a
  Frontier node there are four packages (eight GCDs, 512 GB of HBM total) against 512 GB of host
  DDR4. Aggregate device memory equals host memory, so replaying on several GCDs at once is bounded
  by host capacity before anything else.
- **MI325X (gfx942)** -- 256 GB HBM3E. This is the only family that runs rocprofiler-sdk CI today.
  A full-VRAM application needs 256 GB of host RAM for a single snapshot.
- **MI455X (gfx125X)** -- 432 GB HBM4 at 23.3 TB/s. A full-VRAM snapshot needs 432 GB of host RAM,
  which exceeds the host memory typically provisioned per GPU in a dense node. At an optimistic
  50 GB/s that snapshot takes about **8.6 seconds**, once per dispatch.

The trend matters more than any single number. Device bandwidth is growing far faster than the host
link: the HBM-to-PCIe ratio is roughly 100:1 on MI250X and MI325X, and roughly 360:1 on MI455X at
23.3 TB/s against a Gen5 x16 link. Kernel replay's cost is paid on the slow side of that ratio while
the value it delivers scales with the fast side, so **the technique gets relatively more expensive on
each generation** unless the amount copied is decoupled from the footprint.

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
- A path to run the same workloads on MI325X, MI455X and MI250X and compare, since the conclusion
  above is architecture-dependent.

## See also

- [Memory snapshot and restore](kernel_replay_memory_snapshot.md) -- what is captured and excluded
- [Concurrency and isolation](kernel_replay_concurrency_and_isolation.md) -- the drain and the lock
- {ref}`using-kernel-replay-rocprofv3` -- when to prefer application replay
