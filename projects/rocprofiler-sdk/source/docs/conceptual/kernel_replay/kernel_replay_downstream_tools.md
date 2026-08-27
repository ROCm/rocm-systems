# Kernel replay and the downstream tools

What it would take for rocprofiler-compute, rocprofiler-systems and the rocprofv3 "lite trace" fast
path to use kernel replay, and which of them it is actually worth doing for.

Neither rocprofiler-compute nor rocprofiler-systems mentions kernel replay anywhere today. That is
expected — it is a beta SDK feature — but it means everything below is a plan rather than a
description.

## The short version

Kernel replay's value proposition is specific: it collapses N application runs into one. So its
payoff to a downstream tool is proportional to how many times that tool runs the application today.

| Tool | Application runs for a full counter profile today | Payoff |
| --- | --- | --- |
| rocprofiler-compute | Between roughly 13 and 20, depending on architecture | Large |
| rocprofiler-systems | 1 | Small |
| rocprofv3 lite trace | Not applicable — collects no counters | None by design |

rocprofiler-compute is where the work belongs. rocprofiler-systems already collects everything in a
single run and would only benefit in the case where a user asks for more counters than fit in one
hardware pass, which it does not currently handle at all.

## rocprofiler-compute

### What it does today

For a default profile it bin-packs the requested counters into hardware-sized groups, writes one
`perfmon/pmc_perf_<N>.yaml` per group, and then **runs the application once per file**. The loop is
in `profiler_base.run_profiling()`. The test workloads in the repository record exactly this:
`[Run 1/13]` through `[Run 13/13]` for `vcopy`, each one relaunching the binary.

That is the cost kernel replay exists to remove. Collapsing 13 or 20 runs to one is the single
largest change available to compute's profiling time, and it also removes a correctness hazard that
application replay carries: every group is collected from a *different execution*, so any run-to-run
variation in the workload shows up as inconsistency between counters that are supposed to describe
the same dispatch.

Compute already has one single-run alternative, `--iteration-multiplexing`, but it is a different
mechanism — the native tool rotates counter configurations across *repeated invocations of the same
kernel*, and analysis imputes what is missing. It needs each kernel to run at least N times and it
does not snapshot memory. Kernel replay and iteration multiplexing are two answers to the same
question and would need to be mutually exclusive.

### The blocker that matters

Compute re-derives dispatch identity from timestamps:

```python
dispatch_ids = csv_ops.GroupIdAssigner(
    ["PID", "Kernel_Name", ..., "Start_Timestamp", "End_Timestamp"],
    "Dispatch_ID",
)
```

Under application replay this is fine, because each pass is a separate process and the SDK's own
dispatch ids are not comparable across runs. Under kernel replay it is exactly wrong: the passes of
one dispatch share a dispatch id but have *different timestamps*, because they are different
executions of the same kernel. Re-deriving identity from the timestamp would give each pass its own
synthetic id and the counter groups would never merge back onto one row.

The fix is to stop re-deriving under replay and use the SDK's `dispatch_id`, which is stable across
passes by construction. Everything downstream of that is closer to working than it looks:
`process_rocpd_csv()` already pivots multiple counter names onto a single dispatch row, which is the
shape replay produces.

### The other work

- **The native tool would need to implement the replay callbacks.** Compute's default path preloads
  its own `librocprofiler-compute-tool.so` with `ROCPROF_COUNTER_COLLECTION=0`, so the SDK's counter
  collection is not what is running. That tool would have to subscribe to the replay domain, supply
  `pass_count_cb`, and select its counter configuration by `current_pass` — the shape in
  `samples/kernel_replay/counters_client.cpp`. Alternatively compute falls back to SDK counter
  collection when replay is on.
- **The per-pass output directories go away.** `run_prof()` currently writes each pass under its own
  directory and copies `out/pass_1/` into `out/pmc_1/`; with one run there is one output.
- **`concat_result_csvs()` becomes unnecessary** for counter passes, since there is one result file
  rather than one per pass.
- **PC sampling and roofline stay separate runs.** Replay is counters-only, and roofline is an
  independent micro-benchmark rather than a counter pass.

### Suggested order

Add the flag and make it mutually exclusive with `--iteration-multiplexing`; collapse the run loop;
implement replay in the native tool or force SDK counter collection; then fix the post-processing so
dispatch identity survives. The post-processing fix is the one that decides whether the rest works,
so it is worth proving first with a hand-run replay profile before changing the orchestration.

## rocprofiler-systems

rocprofiler-systems does not shell out to `rocprofv3`. It links the SDK and registers as a client
tool in `library/rocprofiler-sdk.cpp`, and it runs the application **once**.

For dispatch-attached counters (`ROCPROFSYS_ROCM_EVENTS`) it builds a single counter configuration
per agent containing every requested counter, and `dispatch_counting_service_callback()` hands that
same configuration to every dispatch. There is no pass loop to collapse, because there is no pass
loop at all — and correspondingly no handling for the case where the requested counters do not fit
in one hardware pass.

That last point is the only real motivation here. Kernel replay would let systems accept a counter
list larger than one pass, which today either fails or silently truncates. The change is contained:
create N configurations per agent instead of one, subscribe to the replay domain, and select by
`current_pass` in the existing dispatch callback.

Two things would not follow along:

- **The polled counter path is not dispatch-scoped.** `ROCPROFSYS_GPU_PERF_COUNTERS` samples on a
  timer through the device counting service, not per dispatch, so replay's per-dispatch model does
  not map onto it. It would stay single-pass.
- **The output pipeline has no notion of a pass.** Systems writes to Perfetto and rocpd through its
  own trace cache rather than parsing rocprofv3 output. If replay produces multiple counter samples
  against one dispatch, the writers need a pass dimension or they will double-count.

Given that systems is a tracing and sampling tool first, and that its counter collection already
fits in one run for the counter lists people actually use with it, this is worth doing only when
someone hits the counters-exceed-one-pass case.

## rocprofv3 lite trace

"Lite trace" is not a separate library or project — there is no `rocprofiler-lite` in the tree. It is
an unmerged rocprofiler-sdk runtime mode on the `rocprofv3-lite-trace-fast-path` branch, reached by
`ROCPROF_LITE_TRACE=1` or `rocprofv3 --lite-trace`. The name comes from the record file it writes,
`rocprofiler-lite-trace-<pid>.records`.

Its purpose is to collect kernel dispatch names and timestamps and nothing else, at the lowest
overhead reachable. It allows exactly one buffer tracing kind
(`ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH`) and one callback kind (code object), writes compact
records straight into an mmap store that bypasses the normal buffer pipeline, and uses a separate
`FastPathWriteInterceptor` that accepts only single-packet kernel dispatch submissions. Counter
collection is refused outright: `rocprofiler_configure_callback_dispatch_counting_service` returns
`ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT` when lite trace is on, and the tool config treats
`ROCPROF_COUNTERS` as a fatal combination.

**Kernel replay should not be added to lite trace.** The two are opposed by construction. Replay
requires counter collection, multi-pass re-submission, a memory snapshot and restore around every
dispatch, and per-pass context toggling — which is precisely the machinery lite trace exists to
avoid. Supporting it would mean linking the whole `kernel_replay/` subsystem, relaxing every service
block that defines the mode, and disabling the fast path for any dispatch that gets replayed. At
that point the mode is no longer lite, and the user is better served by the full SDK with
`--kernel-replay-beta-enabled`.

The one thing that *will* be needed when lite trace merges is a guard. The fast path's entry
condition is currently a check on notifiers:

```cpp
if(kernel_trace_fast_path_enabled() && queue.get_notifiers() == 0)
```

Replay is gated separately, on `kernel_replay::has_active_replay_contexts()`. Whichever lands
second has to make the fast path yield to an active replay context, or a run with both enabled would
take the single-shot path and silently collect no replay at all — the same failure mode as HIP
graphs, and just as quiet.
