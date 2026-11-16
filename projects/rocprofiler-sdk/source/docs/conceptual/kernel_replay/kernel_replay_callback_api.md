(kernel-replay-callback-api)=
# Kernel Replay — Callback API and Tool Configuration

Kernel replay is exposed as a **callback tracing service**, not through a dedicated
`rocprofiler_configure_*` entry point. A tool subscribes to the
`ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY` domain through the ordinary
`rocprofiler_configure_callback_tracing_service()` call and drives the replay loop from the callbacks
it receives.

The API is experimental. Its public header is
`source/include/rocprofiler-sdk/experimental/kernel_replay.h`. The domain and payload are expected
to change before a stable release. This page describes the SDK domain itself; `rocprofv3` exposes a
narrower slice of it through its own command-line option.

Decoupling replay from counter collection is the point of the design: a tool can use replay for
hardware counters, kernel timing statistics, PC sampling, thread trace, SPM, or anything else, and
it decides per pass which of its services are active, and for dispatch thread trace which part of
the GPU each pass traces. ``rocprofv3`` wires replay to dispatch counter collection, and, with
``--att-remap``, to the remapper's per-pass dispatch and thread trace target; SPM, PC sampling, and
mixing services across passes require a custom tool (see :ref:`kernel-replay-sdk-api` and the
``samples/kernel_replay/`` examples).

Replay is **not** a dedicated counting service mutually exclusive with ordinary dispatch counting.
There is no `rocprofiler_configure_kernel_replay_counting_service()`, no pass-count environment
variable, and no `--kernel-replay-passes` option in this API.

## API surface

### Domain and operations

```c
ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY   // rocprofiler_callback_tracing_kind_t

typedef enum rocprofiler_kernel_replay_operation_t
{
    ROCPROFILER_KERNEL_REPLAY_NONE   = 0,
    ROCPROFILER_KERNEL_REPLAY_CONFIG = 1,  ///< Pass-count / loop configuration, once per dispatch
    ROCPROFILER_KERNEL_REPLAY_PASS,        ///< Per-pass begin/end notification
    ROCPROFILER_KERNEL_REPLAY_LAST,
} rocprofiler_kernel_replay_operation_t;
```

Both operations deliver `PHASE_ENTER` and `PHASE_EXIT` callbacks. `CONFIG` fires once per candidate
dispatch; `PASS` fires once per executed pass.

Replay activation is keyed on the `CONFIG` operation specifically: the interceptor looks for an
active context whose callback tracer covers `ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY` with
`ROCPROFILER_KERNEL_REPLAY_CONFIG`. Subscribing to all operations (passing `NULL, 0` for the
operations array) satisfies this.

`ROCPROFILER_KERNEL_REPLAY_SNAPSHOT` and `ROCPROFILER_KERNEL_REPLAY_RESTORE` operations are reserved
as future work so a tool can observe snapshot/restore phases; they are not implemented.

### Payload

One flat struct carries both operations; there are no unions. Which members are meaningful depends on
the current operation.

```c
typedef struct rocprofiler_callback_tracing_kernel_replay_data_t
{
    uint64_t                           size;
    rocprofiler_kernel_dispatch_info_t dispatch_info;   // always populated by the SDK

    // [CONFIG] tool-provided; the tool sets these during CONFIG PHASE_ENTER
    rocprofiler_kernel_replay_pass_count_cb_t replay_pass_count;
    rocprofiler_kernel_replay_continue_cb_t   replay_continue;

    // [PASS] read-only, populated by the SDK
    uint64_t current_pass;    // 0-indexed
    uint64_t total_passes;    // 0 for an indefinite loop

    // [PASS] SDK-provided; the tool calls these during PASS PHASE_ENTER
    rocprofiler_kernel_replay_context_cb_t replay_start_context;
    rocprofiler_kernel_replay_context_cb_t replay_stop_context;
    rocprofiler_kernel_replay_thread_trace_params_cb_t replay_set_thread_trace_parameters;

    // [CONFIG/PASS] read-only, populated by the SDK: the dispatch the application submitted
    const rocprofiler_kernel_replay_dispatch_t* app_dispatch;
    // [PASS] SDK-provided; the tool calls it during PASS PHASE_ENTER
    rocprofiler_kernel_replay_dispatch_cb_t     replay_set_dispatch;
} rocprofiler_callback_tracing_kernel_replay_data_t;
```

`replay_set_thread_trace_parameters` was appended after the first release of the payload. A tool
checks that `size` covers it before calling it; see
[Per-pass thread trace targets](#per-pass-thread-trace-targets).

During `CONFIG`, the pass-info fields are zero. During `PASS`, the config callback pointers are
zero and must not be modified.

The SDK maintains a single `rocprofiler_user_data_t` for the whole replay sequence. A tool can write
per-dispatch state into `user_data` during `CONFIG` `PHASE_ENTER`, and the same value is delivered to
every subsequent `PASS` callback and to both `replay_pass_count` and `replay_continue` for that
dispatch — so a tool does not need its own side table keyed on dispatch.

### Dispatch identity across passes

One dispatch id is reserved for the logical dispatch before the `CONFIG` callback fires, and every
submit reuses it: all replay passes, and the single fall-through run if replay is declined. So
`replay_pass_count`, every `CONFIG` and `PASS` callback, and every record produced by any pass carry the
same `dispatch_info.dispatch_id`, and the dispatch counter is bumped exactly once for the logical
dispatch. Passes are distinguished by `current_pass`.

The application observes **exactly one** kernel completion for every loop that ends. The
application's completion signal is suppressed on every pass and fired once after the loop ends
(including after an indefinite loop breaks out), not at pass `N-1`, so a fixed pass count, an early
exit, and an indefinite loop that terminates are indistinguishable from the application's side. An
indefinite loop that never terminates is the one case where this does not hold; see *Terminating an
indefinite loop* below.

Each pass produces its own kernel start/end timestamps in dispatch tracing and counter records.
Those timestamps differ per pass even though `dispatch_info.dispatch_id` is the same for all of
them. Distinguish passes with `current_pass` (or the JSON `replay_pass` field on counter records),
not with `dispatch_id` alone.

### Correlation ids

The SDK captures the thread, internal, and ancestor correlation ids once when the replay window
opens and threads them through every `CONFIG` and `PASS` callback for that dispatch. They are the
same on every pass, just like `dispatch_info.dispatch_id`. Record callbacks therefore arrive once
per pass with identical correlation and dispatch ids but different start/end timestamps.

Do not key side tables on `dispatch_id` alone when storing per-pass counter or trace data — merge
on `(dispatch_id, current_pass)` or use pass-local state set during `PASS` `PHASE_ENTER`. On
``rocprofv3``, the counter JSON `replay_pass` field carries the pass index.

## Pass-count semantics

`replay_pass_count` is the switch that decides whether a dispatch is replayed at all.

| `replay_pass_count` | `replay_continue` | Behavior |
|---|---|---|
| left `NULL` | (ignored) | **Not replayed.** The dispatch runs once, no snapshot is taken, and execution continues as usual. This is the per-dispatch opt-out. |
| returns `1` | (ignored) | **Not replayed.** One pass needs no snapshot or restore, so the dispatch takes the ordinary single-dispatch path. |
| returns `N > 1` | `NULL` | Fixed loop of exactly `N` passes. |
| returns `N > 1` | provided | Up to `N` passes; `replay_continue` may break out early. It cannot extend the loop past `N`. |
| returns `0` | provided | Indefinite loop until `replay_continue` returns zero. `total_passes` is reported as `0`. |
| returns `0` | `NULL` | Rejected: the SDK warns and the dispatch is not replayed. |

`replay_continue` is consulted after a pass completes and after that pass's `PHASE_EXIT`, to decide
whether a further pass follows. Returning non-zero continues the loop; returning zero breaks out.
Because the break happens before `restore()`, the last executed pass leaves device memory in the
state the application expects.

Since it only ever decides whether a *further* pass runs, a fixed loop of `N` passes does not
consult it after the final pass — it sees `current_pass` `0` through `N-2`. An indefinite loop does
consult it after every pass, including the one that ends the loop. Per-pass bookkeeping belongs in
the `PASS` `PHASE_EXIT` callback, which fires for every pass in both modes.

``rocprofv3`` never sets ``replay_continue``; early exit and indefinite loops are custom-tool
features only.

There is no environment variable that overrides this. A tool returns whatever count it needs —
for example the number of counter groups collectable on the dispatch's agent.

### Terminating an indefinite loop

An indefinite plan hands the termination decision entirely to the tool. The pass loop is unbounded:
there is no pass cap, no wall-clock budget, and no diagnostic for a loop that runs long. A
`replay_continue` that never returns zero therefore replays the dispatch forever, and because the
application's completion signal is deferred until after the loop, it is never fired -- the
application blocks on that dispatch for the life of the process.

A tool returning `0` from `replay_pass_count` must therefore guarantee that `replay_continue`
eventually returns zero on every path, including its own error paths. A condition derived from the
collected data -- a target sample count, a convergence check -- needs a fallback for the case where
the data never satisfies it. A tool that cannot make that guarantee should return a fixed `N > 1`
and stop early with `replay_continue` instead, which is bounded by construction.

## Localized context control

Tools frequently want different services active on different passes — for example collect hardware
counters on passes 0..N-2 and kernel dispatch timing on the last pass only. Calling the global
`rocprofiler_start_context()` / `rocprofiler_stop_context()` would leak those changes into other,
non-replayed dispatches. Instead the tool calls the localized enable/disable callbacks delivered in
the `PASS` payload:

```c
payload->replay_stop_context(my_timing_ctx);
```

Contexts are configured and started **globally before replay** (outside the replay callbacks).
The localized enable/disable calls only mask which of those already-active contexts participate in
each pass; they do not create contexts, do not invoke global start/stop, and never mutate global
state.

Semantics:

- **Only legal during `PASS` `PHASE_ENTER`.** Calls made outside that window return
  `ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR` and record nothing.
- **Sticky across passes.** A context disabled in one pass stays disabled until it is enabled again
  within the same replay loop, and vice versa.
- **Scoped to the replay loop.** Each context's pre-replay active or inactive state is in effect
  again once the loop completes.

**Service combination limits.** Dispatch counter collection and PC sampling are mutually exclusive
on MI2xx/MI3xx when both would run on the **same** replay pass (clock gating). ATT and SPM cannot
safely share one pass because both inject AQL instrumentation. Use separate passes and separate
contexts — locally stop one service before starting another — for services that **consult** the
override map at dispatch time (dispatch counters, SPM, kernel dispatch tracing, and dispatch thread
trace). PC sampling and device counting are agent-wide today and **ignore** localized toggles, so
they keep collecting on every pass even when a tool records a local stop. Do not combine dispatch
counter collection with PC sampling under replay until PC sampling honors localized overrides.

See
[Concurrency and isolation](kernel_replay_concurrency_and_isolation.md#localized-context-control-and-thread-scope)
for how the override map is scoped.

### Per-pass thread trace targets

Thread trace on one dispatch covers one target CU (or WGP) per shader engine, and on some
architectures one SIMD per run. Tracing every SIMD of a WGP, or several WGPs, therefore takes one
run per target. Under replay those runs are passes of the same dispatch from identical starting
memory, so their traces describe the same work. The tool selects each pass's target with the
`replay_set_thread_trace_parameters` callback delivered in the `PASS` payload:

```c
const rocprofiler_thread_trace_parameter_t simd = {
    ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT, {payload->current_pass}};
payload->replay_set_thread_trace_parameters(my_att_ctx, &simd, 1);
```

Semantics, matching the localized context toggles:

- **Only legal during `PASS` `PHASE_ENTER`**, for a context active when the loop began
  (`ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR` and `ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED`
  otherwise). The context must have dispatch thread trace configured on the replayed agent.
- **Only the target parameters.** `ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU`,
  `ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT` and
  `ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK`. Parameters a call leaves out keep the
  value the context was configured with, and a call with no parameters returns to the configured
  target.
- **Validated up front.** A value outside the configuration limits, a shader engine mask naming an
  engine the agent does not have, or a mask that leaves less than 128 KiB of the trace buffer per
  traced engine returns `ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT` and keeps the previous
  selection.
- **Sticky across passes, scoped to the loop.** The selection holds until changed within the loop
  and is dropped when the loop completes. The configured parameters never change.
- **One trace buffer.** The SDK builds the start and stop packets for each distinct target on first
  use and reuses them for later passes and dispatches; every target writes into the context's one
  trace buffer, which is safe because traced dispatches never overlap.

The thread trace dispatch callback still runs on every pass and decides whether that pass is traced,
and it can tag the trace with the pass through its `userdata_shader` output so the shader data
callback can attribute each pass's data. `samples/kernel_replay/att_targets_client.cpp` traces the
four SIMDs of one WGP over four passes.

### Correlation IDs across passes

The SDK reads the **internal** correlation ID once before the replay loop and threads the same value
through CONFIG and every PASS callback. The **external** correlation ID is produced by the tool's
own callback on each submit; a tool that increments it blindly will emit a different external ID per
pass. Use `current_pass` (or `replay_pass` on counter records) as the reliable per-pass
discriminator, not the external correlation ID.

### Replacing a pass's dispatch

A pass normally resubmits the application's packet. A tool that needs a pass to run something
else, for example a rewritten copy of the kernel that reads extra kernel arguments or runs on a
padded grid, calls `replay_set_dispatch` during `PASS` `PHASE_ENTER`:

```c
rocprofiler_kernel_replay_dispatch_t dispatch = *payload->app_dispatch;
dispatch.kernel_object   = rewritten_kernel_object;   // loaded through HSA by the tool
dispatch.kernarg_address = tool_kernargs;             // valid until PASS PHASE_EXIT
dispatch.grid_size.x    *= 2;
payload->replay_set_dispatch(&dispatch);
```

- **One pass at a time.** A pass that does not call it submits the application's dispatch; `NULL`
  clears a replacement made earlier in the same callback.
- **Same packet type.** The SDK builds the pass's packet from the application's, keeping its type,
  header, fences and dependency signal. An ordinary dispatch packet takes `cluster_size`
  {0, 0, 0}; a gfx1250 extended dispatch packet takes a nonzero `cluster_size` and a grid that is a
  whole number of clusters, and its cluster counts are recomputed. HIP on gfx1250 submits ordinary
  launches as extended packets with `cluster_size` {1, 1, 1}, so start from `app_dispatch`.
- **Validated.** The kernel object must belong to a code object loaded through HSA on the replayed
  agent, so the SDK knows its kernel id; sizes must be nonzero and fit the packet; a kernel with
  kernel arguments needs a `kernarg_address`. A rejected call returns
  `ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT`.
- **Nothing else changes.** Snapshot and restore, the profiling packets injected around the
  kernel, the dispatch id and the deferred application completion all behave as for any pass.
  Services observing the pass see the replacement's kernel id and geometry; the payload's
  `dispatch_info` keeps describing the application's dispatch.
- **The application keeps the last pass's output**, so let the final pass submit the application's
  dispatch.

`app_dispatch` describes what the application submitted, including its `kernarg_address`, which
stays unchanged until the loop ends. A tool copies the application's kernel arguments from there
when its replacement extends them. Inside the replay callbacks, move memory with host writes or
`hsa_memory_copy`, as the snapshot does: launching work on an application queue from a callback
deadlocks against the replay lock. `tests/kernel-replay-dispatch/` loads a substitute kernel,
replaces one pass with its own kernel arguments and a doubled grid, and checks the result.

## Configuring the service

Configuring the service is an ordinary callback tracing subscription. Doing so also switches on the
replay allocation tracker, so a run that never configures the service pays no tracking cost.

```c
rocprofiler_context_id_t ctx = {0};
rocprofiler_create_context(&ctx);

rocprofiler_configure_callback_tracing_service(ctx,
                                               ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
                                               NULL,   // all operations
                                               0,
                                               tool_kernel_replay_callback,
                                               NULL);  // callback_args

rocprofiler_start_context(ctx);
```

The callback installs `replay_pass_count` during `CONFIG` `PHASE_ENTER` and reads the pass index during
`PASS`:

```c
void
tool_kernel_replay_callback(rocprofiler_callback_tracing_record_t record,
                            rocprofiler_user_data_t* /* user_data */,
                            void* /* callback_args */)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY) return;

    auto* payload =
        static_cast<rocprofiler_callback_tracing_kernel_replay_data_t*>(record.payload);

    if(record.operation == ROCPROFILER_KERNEL_REPLAY_CONFIG &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        // How many passes for this dispatch? Return 1 to skip replay.
        payload->replay_pass_count = tool_pass_count_callback;
    }
    else if(record.operation == ROCPROFILER_KERNEL_REPLAY_PASS)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
            publish_current_pass(payload->current_pass);
        else
            clear_current_pass();
    }
}
```

Because the replay loop runs synchronously on the submitting thread, the `PASS` callback and any
service callbacks fired by that pass run on the same thread within the pass. That is what lets a tool
publish the pass index in thread-local state during `PASS` `PHASE_ENTER` and have its counter
dispatch callback pick it up, clearing it again on `PHASE_EXIT` so ordinary dispatches never observe
a stale value.

`replay_pass_count` is called by the SDK, once per dispatch, after `CONFIG` `PHASE_ENTER` returns:

```c
uint64_t
tool_pass_count_callback(rocprofiler_kernel_dispatch_info_t dispatch_info,
                         rocprofiler_user_data_t /* user_data */)
{
    // e.g. one pass per counter group collectable on THIS dispatch's agent
    return groups_for_agent(dispatch_info.agent_id);
}
```

Deriving the count from `dispatch_info.agent_id` rather than from a global is the usual pattern
when collectable counter groups differ per agent: pass `i` must map to group `i` with no wrap or
skip on an agent with a different or partial group set.

## API reference

The payload struct is documented in the `CALLBACK_TRACING_SERVICE` Doxygen group and appears on the
{ref}`callback_tracing_reference` page along with the rest of the callback tracing API. There is no
separate kernel replay Doxygen group. A walkthrough for tool authors is
{ref}`kernel-replay-sdk-api`. See
{ref}`using-kernel-replay` for a configure / `replay_pass_count` / local-context how-to.

## rocprofv3 integration

`rocprofv3` exposes replay through one flag:

```bash
rocprofv3 --pmc <counters...> --replay-mode kernel --kernel-replay-beta-enabled -- <app>
rocprofv3 --att --att-remap --replay-mode kernel --kernel-replay-beta-enabled -- <app>
```

- `--replay-mode kernel` requires `--pmc`, or `--att` with `--att-remap`, and
  `--kernel-replay-beta-enabled`, and the CLI fails with a diagnostic if either is missing. `--pmc`
  and `--att` cannot be combined under replay.
- The tool library creates the kernel replay context when the flag is given, and not otherwise.
- There is no pass-count knob. With `--pmc` the tool derives the pass count from the number of
  counter groups collectable on the dispatch's agent. With `--att-remap` the remapper runs an
  indefinite loop: at every `PASS` `PHASE_ENTER` the tool applies its plan through
  `replay_set_dispatch` and `replay_set_thread_trace_parameters`, and `replay_continue` follows it.
  Only dispatches thread trace will capture are remapped.
- With `--pmc` the flag does not wire `replay_continue` or the localized start/stop context
  callbacks.
- Without the flag, multiple `--pmc` groups continue to use application replay, where the whole
  application is re-run once per group.

JSON counter records include a `replay_pass` field. CSV `counter_collection.csv` does **not** add a
`Replay_Pass` column. With `--att-remap`, thread trace output is written and decoded once per pass, into
`ui_output_agent_<agent>_dispatch_<dispatch>_pass_<pass>/`, which also holds a
`replay_target.json` naming the pass's target. See {ref}`using-kernel-replay-rocprofv3`.

## Source reference

All paths are relative to `projects/rocprofiler-sdk/`.

| Component | File | Symbol |
|---|---|---|
| Payload struct | `source/include/rocprofiler-sdk/experimental/kernel_replay.h` | `rocprofiler_callback_tracing_kernel_replay_data_t` |
| Domain enumerator | `source/include/rocprofiler-sdk/fwd.h` | `ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY` |
| Operation enum | `source/include/rocprofiler-sdk/fwd.h` | `rocprofiler_kernel_replay_operation_t` |
| Service configuration hook | `source/lib/rocprofiler-sdk/callback_tracing.cpp` | `rocprofiler_configure_callback_tracing_service()` |
| CONFIG callbacks and plan | `source/lib/rocprofiler-sdk/kernel_replay/replay_callbacks.cpp` | `execute_config_phase_enter()`, `execute_config_phase_exit()` |
| PASS callbacks | `source/lib/rocprofiler-sdk/kernel_replay/replay_callbacks.cpp` | `execute_pass_phase_enter()`, `execute_pass_phase_exit()` |
| Continue decision | `source/lib/rocprofiler-sdk/kernel_replay/replay_callbacks.cpp` | `should_continue_replay()` |
| Dispatch info population | `source/lib/rocprofiler-sdk/kernel_replay/replay_callbacks.cpp` | `make_dispatch_info()` |
| Operation name/id queries | `source/lib/rocprofiler-sdk/kernel_replay/kernel_replay.cpp` | `name_by_id()`, `id_by_name()` |
| Localized context callbacks | `source/lib/rocprofiler-sdk/kernel_replay/local_context.cpp` | `replay_local_enable_context()`, `replay_local_disable_context()`, `replay_local_set_thread_trace_parameters()` |
| Dispatch replacement | `source/lib/rocprofiler-sdk/kernel_replay/dispatch_override.cpp` | `describe_dispatch()`, `build_dispatch_packet()`; setter `replay_set_dispatch()` in `replay_callbacks.cpp` |
| Per-pass thread trace packets | `source/lib/rocprofiler-sdk/thread_trace/core.cpp` | `DispatchThreadTracer::resolve_target()`, `ThreadTracerAgent::get_start_packet()` |
| Replay loop and dispatch-id reservation | `source/lib/rocprofiler-sdk/hsa/queue.cpp` | `WriteInterceptor` |
| Tool-side subscription | `source/lib/rocprofiler-sdk-tool/tool.cpp` | `kernel_replay_callback()`, `kernel_replay_pass_count_callback()` |
| Tool-side flag | `source/lib/rocprofiler-sdk-tool/config.hpp` | `kernel_replay` |
| CLI flags | `source/bin/rocprofv3.py` | `--replay-mode kernel`, `--kernel-replay-beta-enabled` |
| Tests | `source/lib/rocprofiler-sdk/kernel_replay/tests/` | `local_context.cpp` |
| Thread trace target tests | `source/lib/rocprofiler-sdk/thread_trace/tests/` | `replay_target_test.cpp` |
