(kernel-replay-callback-api)=
# Kernel Replay — Callback API and Tool Configuration

Kernel replay is exposed as a **callback tracing service**, not through a dedicated
`rocprofiler_configure_*` entry point. A tool subscribes to the
`ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY` domain through the ordinary
`rocprofiler_configure_callback_tracing_service()` call and drives the replay loop from the callbacks
it receives.

The API is experimental. Its public header is
`source/include/rocprofiler-sdk/experimental/kernel_replay.h`. The domain and payload are expected
to change before a stable release. Command-line `rocprofv3` wiring is the stacked tool integration
PR, not this SDK change.

Decoupling replay from counter collection is the point of the design: a tool can use replay for
hardware counters, kernel timing statistics, PC sampling, thread trace, SPM, or anything else, and
it decides per pass which of its services are active. ``rocprofv3`` wires replay to dispatch counter
collection only; SPM, PC sampling, and thread trace require a custom tool (see
:ref:`kernel-replay-sdk-api` and the ``samples/kernel_replay/`` examples).

An earlier prototype (`rocprofiler_configure_kernel_replay_counting_service()`) was a dedicated
counting service mutually exclusive with ordinary dispatch counting. That configure function, a
pass-count environment variable, and `--kernel-replay-passes` are **not** in this API.

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
    uint64_t (*pass_count_cb)(rocprofiler_kernel_dispatch_info_t dispatch_info,
                              rocprofiler_user_data_t            user_data);
    int (*replay_continue_cb)(rocprofiler_kernel_dispatch_info_t dispatch_info,
                              uint64_t                           current_pass,
                              uint64_t                           total_passes,
                              rocprofiler_user_data_t            user_data);

    // [PASS] read-only, populated by the SDK
    uint64_t current_pass;    // 0-indexed
    uint64_t total_passes;    // 0 for an indefinite loop

    // [PASS] SDK-provided; the tool calls these during PASS PHASE_ENTER
    rocprofiler_status_t (*replay_local_enable_context_cb)(rocprofiler_context_id_t context_id);
    rocprofiler_status_t (*replay_local_disable_context_cb)(rocprofiler_context_id_t context_id);

    uint8_t reserved_padding[64];  // reserved for extensions w/o ABI break
} rocprofiler_callback_tracing_kernel_replay_data_t;
```

During `CONFIG`, the pass-info fields are zero. During `PASS`, the config callback pointers are
zero and must not be modified.

The SDK maintains a single `rocprofiler_user_data_t` for the whole replay sequence. A tool can write
per-dispatch state into `user_data` during `CONFIG` `PHASE_ENTER`, and the same value is delivered to
every subsequent `PASS` callback and to both `pass_count_cb` and `replay_continue_cb` for that
dispatch — so a tool does not need its own side table keyed on dispatch.

### Dispatch identity across passes

One dispatch id is reserved for the logical dispatch before the `CONFIG` callback fires, and every
submit reuses it: all replay passes, and the single fall-through run if replay is declined. So
`pass_count_cb`, every `CONFIG` and `PASS` callback, and every record produced by any pass carry the
same `dispatch_info.dispatch_id`, and the dispatch counter is bumped exactly once for the logical
dispatch. Passes are distinguished by `current_pass`.

The application observes **exactly one** kernel completion regardless of the pass count, an early
exit, or an indefinite loop. The application's completion signal is suppressed on every pass and
fired once after the loop ends (including after an indefinite loop breaks out), not at pass `N-1`.

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
``rocprofv3``, the counter JSON `replay_pass` field (stacked tool PR) carries the pass index.

## Pass-count semantics

`pass_count_cb` is the switch that decides whether a dispatch is replayed at all.

| `pass_count_cb` | `replay_continue_cb` | Behavior |
|---|---|---|
| left `NULL` | (ignored) | **Not replayed.** The dispatch runs once, no snapshot is taken, and execution continues as usual. This is the per-dispatch opt-out. |
| returns `1` | (ignored) | **Not replayed.** One pass needs no snapshot or restore, so the dispatch takes the ordinary single-dispatch path. |
| returns `N > 1` | `NULL` | Fixed loop of exactly `N` passes. |
| returns `N > 1` | provided | Up to `N` passes; `replay_continue_cb` may break out early. It cannot extend the loop past `N`. |
| returns `0` | provided | Indefinite loop until `replay_continue_cb` returns zero. `total_passes` is reported as `0`. |
| returns `0` | `NULL` | Rejected: the SDK warns and the dispatch is not replayed. |

`replay_continue_cb` is consulted after each pass completes and after that pass's `PHASE_EXIT`.
Returning non-zero continues the loop; returning zero breaks out. Because the break happens before
`restore()`, the last executed pass leaves device memory in the state the application expects.

``rocprofv3`` never sets ``replay_continue_cb``; early exit and indefinite loops are custom-tool
features only.

There is no environment variable that overrides this. A tool returns whatever count it needs —
for example the number of counter groups collectable on the dispatch's agent.

## Localized context control

Tools frequently want different services active on different passes — for example collect hardware
counters on every pass but PC sampling on the last pass only. Calling the global
`rocprofiler_start_context()` / `rocprofiler_stop_context()` would leak those changes into other,
non-replayed dispatches. Instead the tool calls the localized enable/disable callbacks delivered in
the `PASS` payload:

```c
payload->replay_local_disable_context_cb(my_pc_sampling_ctx);
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
contexts — locally stop one service before starting another. Kernel dispatch tracing honors the
overrides by dropping disabled contexts from the pass's tracing data. Counter collection, SPM, and
thread trace consult the override map at dispatch time and skip building AQL packets for disabled
contexts. PC sampling and device counting are agent-wide today: the toggle is recorded but
collection continues regardless, so separate passes are still required to avoid running counters and
PC sampling together.

See
[Concurrency and isolation](kernel_replay_concurrency_and_isolation.md#localized-context-control-and-thread-scope)
for how the override map is scoped.

### Correlation IDs across passes

The SDK reads the **internal** correlation ID once before the replay loop and threads the same value
through CONFIG and every PASS callback. The **external** correlation ID is produced by the tool's
own callback on each submit; a tool that increments it blindly will emit a different external ID per
pass. Use `current_pass` (or `replay_pass` on counter records) as the reliable per-pass
discriminator, not the external correlation ID.

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

The callback installs `pass_count_cb` during `CONFIG` `PHASE_ENTER` and reads the pass index during
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
        payload->pass_count_cb = tool_pass_count_callback;
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

`pass_count_cb` is called by the SDK, once per dispatch, after `CONFIG` `PHASE_ENTER` returns:

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
separate kernel replay Doxygen group (the earlier prototype's `kernel_replay_service` group is
gone). A walkthrough for tool authors is {ref}`kernel-replay-sdk-api`. See
{ref}`using-kernel-replay` for a configure / `pass_count_cb` / local-context how-to.

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
| Localized context callbacks | `source/lib/rocprofiler-sdk/kernel_replay/local_context.cpp` | `replay_local_enable_context()`, `replay_local_disable_context()` |
| Replay loop and dispatch-id reservation | `source/lib/rocprofiler-sdk/hsa/queue.cpp` | `WriteInterceptor` |
| Tests | `source/lib/rocprofiler-sdk/kernel_replay/tests/` | `local_context.cpp` |
