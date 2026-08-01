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

**Caveats.**

- `hipExtModuleLaunchKernel` takes *global work size* (grid x block), not grid
  dimensions in blocks. Passing grid dimensions silently yields `gridDim == 1`.
- Under stream capture the launch is legal, but graph replay never populates the
  events; `hipEventElapsedTime` then returns `hipErrorInvalidHandle`. Graph plans
  need a fallback.
- Under rocprof, the pointer chase finds nothing — rocprof's interception
  relocates the fields. Discovery fails cleanly and the caller falls back.
- On the very first dispatch of a process the cache and the HSA API once differed
  by 420 ns (spans still matching to 1 ns). Never reproduced in steady state;
  cause unknown.

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
