# AQL Shared-Memory Monitor

`experimental/aql-shm-monitor` currently contains three pieces:

- `libaqlmon_runtime_contract.so`
  A small runtime-facing negotiation ABI in `aqlmon/runtime_contract.h`.
- `libaql_shm_monitor.so`
  The current capture backend. It still runs as an `LD_PRELOAD` HSA interposer.
- `aqlmon_launch`
  A small launcher that only preloads `libaql_shm_monitor.so`:
  `aqlmon_launch ... -- ./app`

## Design Summary

The POC keeps `rocprofiler-register` generic:

- `rocprofiler-register` exposes a one-time runtime activation callback API.
- runtimes register a callback once
- `rocprofiler-register` fires that callback on tool activation
  - `STARTUP` when a tool is active during normal API-table propagation
  - `ATTACH` when a tool becomes active through late attach

`aqlmon` policy is not embedded in `rocprofiler-register`.

Instead, a linked runtime calls the `aqlmon` contract directly when that generic callback fires.

The current negotiation result is process-global: the first successful runtime negotiation becomes
the active completion-signal ownership mode for the process.

HIP/CLR is the first example runtime:

- HIP links `libaqlmon_runtime_contract.so`
- on the generic activation callback, HIP calls `aqlmon_runtime_negotiate(...)`
- HIP asks for `RUNTIME_PROVIDED` plus `KERNEL_DISPATCH_SIGNALS`
- if `aqlmon` grants that request, HIP sets a per-kernel `AttachCompletionSignal` bit
- if `aqlmon` denies it, HIP leaves that mode disabled and the preload monitor falls back to its
  own signal pool

ROCclr then reuses its existing `attach_signal` path to attach a `ProfilingSignal` without
enabling generic profiling for every command.

The full target design is documented in:

- [docs/design/2026-04-22-aql-shm-monitor-design-decisions.md](docs/design/2026-04-22-aql-shm-monitor-design-decisions.md)

That design goes further than the current branch implementation:

- generic activation through `rocprofiler-register`
- runtime-to-`AQLMON` completion-signal ownership negotiation
- shadow write-pointer publication with no packet copying
- minimal async completion handling
- code-object lifetime records in the same shared-memory stream

In the target design, `rocprofiler-sdk` is the intended consumer of that shared-memory stream. The
sample reader and trace exporter in this directory are validation tools only.

## Current Scope

What is implemented:

- generic runtime activation callback support in `rocprofiler-register`
- HIP/CLR runtime negotiation against the linked `aqlmon` contract
- runtime-provided vs monitor-provided completion-signal mode selection
- current preload monitor consuming the negotiated mode
- `rocprofiler-sdk` shm receiver for kernel-dispatch completion correlation
- simple launcher for packet-dump validation
- eligible kernel-trace-only sessions start the SDK shm receiver for AQLMON before
  queue-controller setup
- legacy kernel-intercept fallback is preserved until the receiver is actually operational
- if a session requests counter collection or another queue-intercept-dependent service, kernel
  dispatch tracing stays on the old queue-intercept path for that process state
- validated pure `rocprofv3 --kernel-trace` GraphBench process-launch runs, including the large
  graph batch path at `--size 8192`
- on the validated HIP/CLR publish order, publish progression follows the shadow doorbell
  watermark plus packet-header validity; per-slot reservation metadata is only advisory producer
  metadata and is not used as a hard publish gate

What is not implemented yet:

- removing the current `LD_PRELOAD` dependency of the heavy monitor backend
- automatic `rocprofv3` preload/policy selection for all session types
- an AQLMON-backed mixed `kernel-trace + counter-collection` session; mixed sessions stay on the
  old queue-intercept path by design in the current slice
- runtime-owned graph replay completion coverage for zero-signal graph packets

## Validation Snapshot

Current branch validation has covered:

- direct `rocprofv3 --kernel-trace` GraphBench process-launch runs on the fixed path
- pure kernel-trace GraphBench at `--size 2048 --iters 1000 --topology straight`
- pure kernel-trace GraphBench at `--size 8192 --iters 1000 --topology straight`
- containerized GraphBench and PyTorch ROCm benchmark harness setup for local validation

For local benchmarking and validation on this branch, `rocprofv3 -f json` is preferred. That
keeps kernel-trace validation on the AQLMON path without depending on rocpd schema packaging in
the local overlay.

## Build

```bash
cmake -S /home/aelwazir/work/rocm-systems-dev/.worktrees/experimental-aql-shm-monitor/experimental/aql-shm-monitor \
  -B /home/aelwazir/work/rocm-systems-dev/.worktrees/experimental-aql-shm-monitor/experimental/aql-shm-monitor/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /home/aelwazir/work/rocm-systems-dev/.worktrees/experimental-aql-shm-monitor/experimental/aql-shm-monitor/build -j
```

## Install

```bash
cmake --install /home/aelwazir/work/rocm-systems-dev/.worktrees/experimental-aql-shm-monitor/experimental/aql-shm-monitor/build \
  --prefix /home/aelwazir/install/aqlmon-stack
```

Build HIP/CLR against that prefix so `amdhip64` can find:

- `include/aqlmon/runtime_contract.h`
- `lib/libaqlmon_runtime_contract.so`

## Common Launch Example

The included launcher is only for packet-dump validation:

```bash
/home/aelwazir/install/aqlmon-stack/bin/aqlmon_launch \
  --shm-name /aqlmon-demo \
  -- ./your_hip_app
```

`aqlmon_launch` sets:

- `LD_PRELOAD=...libaql_shm_monitor.so`
- optional `AQLMONITOR_SHM_NAME`

This launcher does not exercise `rocprofiler-register`. The runtime activation path is linked
into HIP/CLR and is intended for a real tool flow later.

## Inspect

```bash
/home/aelwazir/install/aqlmon-stack/bin/aqlmon_read /aqlmon-demo
```

## Export Chrome Trace

```bash
/home/aelwazir/install/aqlmon-stack/bin/aqlmon_trace_json \
  /aqlmon-demo \
  /tmp/aqlmon-demo-trace.json
```

`aqlmon_trace_json` correlates packet, completion, and code-object records by
`pid + queue_id + dispatch_id` and writes Chrome/Perfetto JSON.

## Runtime Contract

The public ABI is in [`aqlmon/runtime_contract.h`](source/include/aqlmon/runtime_contract.h).

The runtime passes:

- `proposed_mode`
- `proposed_capabilities`

`aqlmon` returns:

- `selected_mode`
- `granted_capabilities`

Status codes:

- `AQLMON_STATUS_SUCCESS`
  `aqlmon` accepted the runtime request as-is.
- `AQLMON_STATUS_DENIED`
  `aqlmon` selected a different mode or reduced capabilities.
- `AQLMON_STATUS_ERROR_*`
  invalid ABI or invalid arguments.

Relevant policy controls:

- `AQLMONITOR_COMPLETION_SIGNAL_POLICY=runtime|monitor`

## Data Path

The producer thread remains minimal:

- shadow write-index updates
- normal packet writes by the runtime
- shadow doorbell bookkeeping

The monitor publisher thread does the expensive work:

- observes header-valid packets
- writes packet records to shm
- advances the real WDID
- rings the real doorbell

Current implementation note:

- the completion lifetime fix still uses a mutex-protected lifetime map on the publisher-side
  completion-tracking path
- the current negotiation and routing decision is still process-global in the implementation slice
- the long-term design target remains to reduce that to a cheaper cached fast path

Completion handling is also off the producer thread:

- runtime-provided completion signals are preferred when granted
- otherwise `aqlmon` falls back to its own pooled signal injection
- a separate completion thread emits timing records into the same shm ring

## Limitations

- The heavy monitor backend still depends on `LD_PRELOAD`.
- The current HIP integration only requests `KERNEL_DISPATCH_SIGNALS`.
- Zero-signal packets are ignored when runtime-provided mode is active.
- Multi-process fan-in to one shm object is still not coordinated.
- Negotiation state is still process-global in the current implementation slice, not per
  profiler session.
- The validated `rocprofiler-sdk` shm path is currently for kernel-trace-oriented sessions.
- Counter collection still uses the existing queue-intercept path. If counters are requested,
  kernel-dispatch tracing stays on that old path as well.
- Preload-only counter sessions are not a supported validation mode for this branch. The intended
  direction is to leave counter collection on the old path and use AQLMON for kernel-dispatch
  tracing.
