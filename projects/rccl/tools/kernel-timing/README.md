# Dispatch-attached kernel timing experiments

Standalone probes investigating whether RCCL can get accurate per-kernel timing
without the two stream-marker packets `CollTrace` records today, and without
rocprof. Nothing here is built or linked into RCCL; these exist to establish and
re-verify the findings below.

```bash
make ARCH=gfx90a ROCM_PATH=/work/lmeadows/rocm/srock
./launch_cost 2000 50
```

## Findings

**Stream markers are both expensive and wrong.** `cudaEventRecord` before and
after a launch costs ~9.3 us per dispatch and overstates kernel duration by a
roughly constant ~8 us, which is +85% on a 5 us kernel. `hipExtModuleLaunchKernel`
with events attached to the dispatch avoids the extra packets.

**Attach only the stop event.** Both dispatch timestamps are recoverable from the
stop event alone, and the start event is what costs — it inserts a barrier. Per
launch, over 2000 back-to-back dispatches (`launch_cost`):

| scheme | 5 us kernel | 50 us kernel |
| --- | --- | --- |
| no timing | 5.758 us | 50.719 us |
| stream markers | +9.283 us | +9.276 us |
| attached start+stop | +8.180 us | +8.170 us |
| attached stop only | +0.008 us | +0.003 us |

**`hipEventElapsedTime` cannot report the dispatch interval.** It measures
completion-to-completion, so for an ext-launch pair it returns kernel-end minus
the *start event's* completion, including the launch gap ahead of the kernel.
That is the ~4-5 us discrepancy against rocprof's kernel trace; it is a property
of the API, not a rocprof bug.

**Absolute timestamps are recoverable, in the CLOCK_BOOTTIME domain.** ROCr fills
`start_ts`/`end_ts` in the dispatch's completion signal (`amd_signal_t`) in raw
25 MHz GPU ticks; ROCclr converts them and caches the pair on the event. Both the
converted cache and the underlying signal are reachable from the stop event, and
feeding the signal to the documented `hsa_amd_profiling_get_dispatch_time` gives
values bit-identical to the cache on 25 of 25 dispatches
(`hsa_signal_crosscheck`). The conversion is a fixed affine map: re-converting
one dispatch's ticks over 2 s shows zero drift.

Because the domain is CLOCK_BOOTTIME, timestamps are directly comparable across
all GPUs in a node with no calibration, and merge with a rocprof trace as-is.

**Read the timestamps from ROCr, not from the runtime's cache.** The converted
pair ROCclr caches is not reliably present: on gfx950 it was found for one device
of four, and its location differs between ROCm builds. The signal handle is the
stable thing to locate; asking `hsa_amd_profiling_get_dispatch_time` for the
numbers on every harvest costs nothing measurable and removes any dependence on
how the runtime caches or converts them.

**Ask the right agent.** `hsa_amd_profiling_get_dispatch_time` converts ticks
with the calibration of whichever agent is passed and does not check that the
signal belongs to it, so the wrong agent returns a plausible but skewed answer —
on a 4-GPU run, 21 of 48 dispatches on one rank then reported starting before
their own launch. The agent has to be matched to the device by PCI address;
HIP's device order and HSA's agent order need not agree.

**Caveats.**

- `hipExtModuleLaunchKernel` takes *global work size* (grid x block), not grid
  dimensions in blocks. Passing grid dimensions silently yields `gridDim == 1`.
- Under stream capture the launch is legal, but graph replay never populates the
  events; `hipEventElapsedTime` then returns `hipErrorInvalidHandle`. Graph plans
  need a fallback.
- Under rocprof, the pointer chase finds nothing — rocprof replaces the
  completion signals. Discovery fails cleanly and the caller falls back, so the
  two cannot be collected in one run.
- On the very first dispatch of a process the cache and the HSA API once differed
  by 420 ns (spans still matching to 1 ns). Never reproduced in steady state;
  cause unknown.

## In-tree validation (gfx950, 4x MI350X)

Against `all_reduce_perf -b 1M -e 64M -f 8 -g 4 -n 50 -w 10`, with
`RCCL_KERNEL_TIMING=1` and rccl-tests writing the drained records
(`check_trace.py`, `compare_rocprof.py`):

- 1524 of 1524 dispatches timed, no gaps in the per-rank sequence, every window
  well-formed and non-overlapping, every timestamp inside the host-clock window
  of the run.
- Durations against a rocprofv3 `--kernel-trace` of the same workload on the same
  runtime: **+0.06%** at 8 MB and **+0.02%** at 64 MB. At 1 MB the drain API
  reports 3.6 us *less* than rocprof (33.0 vs 36.6 us) — small collectives are
  where rocprof's own perturbation shows up.
- Cost of leaving it enabled, by message size: +5 us at 8 KB, +2.5 us at 64 KB,
  and nothing measurable from 512 KB up. Harvesting must stay off the launch
  path to get this: querying an event per launch makes the runtime flush the
  queue and cost ~190 us per dispatch.

### Latency-bound sizes (8-128 B)

An all-reduce this small is ~20 us of pure latency, so the cost of attaching the
event is at its most visible: **+3.4 us, about 17%**, steady-state across 8, 32
and 128 B. Two things make that number easy to get wrong. The first timed loop
in a process is slower with timing on than later ones (+7 us vs +3.4), so a
single-size run measures one-time cost as if it were per-dispatch. And a caller
that never drains fills the in-flight queue, after which timing costs far more
than it saves.

Sampling does not help. Timing 1 dispatch in 2, 4, 16 or 64 gives no consistent
reduction and 1-in-4 measured *worse* than timing everything, which suggests the
cost is not per-timed-dispatch so much as the queue carrying a mix of dispatches
with and without completion signals. A `KERNEL_TIMING_SAMPLE` knob was written
and then dropped for this reason.

This is also where the drain API is worth the most. Against rocprof on the same
workload it reports 17.6 us median where rocprof reports 23.9 us, i.e. rocprof
inflates these collectives by **27%** -- at this scale a rank spends most of its
kernel waiting for peers, so anything that delays a launch lengthens the kernel
it is trying to measure. The per-rank medians the drain API records for one
32 B run were 11.2, 14.9, 18.6 and 22.7 us: real skew that a single aggregate
number hides.

## `evtstamp.h`

Recovers the absolute dispatch start/end from a stop event. No runtime ABI is
hardcoded: `discover()` runs two calibration launches of different known
durations and keeps only the pointer chain whose end-minus-start reproduces
both, so a layout change makes discovery fail rather than return wrong numbers.
Memory walking goes through `pread` on `/proc/self/mem`, so following a value
that is not a pointer returns an error instead of faulting.

On the ROCm build this was developed against, discovery lands on
`event -> *(+88) -> *(+248) -> start@+88, end@+96`, with the ROCr signal handle
at `+16` of the same object.

## Programs

| program | purpose |
| --- | --- |
| `clock_domains` | HSA system clock vs CLOCK_BOOTTIME (identical, rate ratio 0.999999) |
| `ext_launch_gating` | argument convention, stream-capture legality, first accuracy and cost passes |
| `marker_vs_attached` | isolates launch-path cost from event cost; accuracy on a busy stream |
| `rocprof_crosscheck` | attached-event durations vs rocprofv3 `--kernel-trace` |
| `timestamp_discovery` | pointer-walks the event object graph to locate the timestamps |
| `timestamp_validation` | host-clock bracketing, and stop-event-only mode |
| `launch_cost` | per-launch cost of each timing scheme (table above) |
| `hsa_signal_crosscheck` | finds the ROCr signal, compares against `hsa_amd_profiling_get_dispatch_time`, measures conversion drift |
| `check_trace.py` | sanity-checks a trace written by rccl-tests: counts, gaps, overlaps, malformed windows |
| `compare_rocprof.py` | duration distributions from the drain API vs a rocprofv3 kernel trace |
| `tiny_compare.py` | the same, for latency-bound sizes where there is nothing to bin |
