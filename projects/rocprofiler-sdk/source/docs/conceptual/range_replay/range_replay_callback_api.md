(range-replay-callback-api)=
# Range Replay — Callback API and Tool Configuration

Range replay is exposed as a **callback tracing service** plus two API calls that mark the range.
A tool subscribes to the `ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY` domain through the ordinary
`rocprofiler_configure_callback_tracing_service()` call, then brackets the phase it wants replayed
with `rocprofiler_range_replay_begin()` and `rocprofiler_range_replay_end()`.

The API is experimental. Its public header is
`source/include/rocprofiler-sdk/experimental/range_replay.h`. The domain, the payload, and the two
entry points are expected to change before a stable release.

## API surface

### Domain and operations

```c
ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY   // rocprofiler_callback_tracing_kind_t

typedef enum rocprofiler_range_replay_operation_t
{
    ROCPROFILER_RANGE_REPLAY_NONE   = 0,
    ROCPROFILER_RANGE_REPLAY_CONFIG = 1,  ///< Pass-count configuration, once per range
    ROCPROFILER_RANGE_REPLAY_PASS,        ///< Per-pass begin/end notification
    ROCPROFILER_RANGE_REPLAY_CLOSE,       ///< Outcome of the range, once per range
    ROCPROFILER_RANGE_REPLAY_LAST,
} rocprofiler_range_replay_operation_t;
```

All three operations deliver `PHASE_ENTER` and `PHASE_EXIT`. The phases mean something slightly
different for `CONFIG` than they do in kernel replay: both run synchronously inside
`rocprofiler_range_replay_begin()`, because they are a configuration exchange rather than a bracket
around a loop that has not started yet. `PASS` fires once per *re-executed* pass. `CLOSE` fires once
per opened range, including for a range that was declined before any pass ran.

`CLOSE` is the operation that has no kernel replay counterpart, and it exists because a range can be
declined long after it was configured — the tool asks for four passes at `CONFIG`, and only at
`CLOSE` can the SDK say whether it got them.

### Marking a range

```c
rocprofiler_status_t rocprofiler_range_replay_begin(uint64_t range_id);
rocprofiler_status_t rocprofiler_range_replay_end(void);
```

`range_id` is tool-defined and is echoed back in every callback for that range. Ranges are
**thread-scoped** and do not nest: `begin` on a thread that already has one open returns
`ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT`, and `end` closes the range belonging to the calling
thread. Only the dispatches submitted by the marking thread are recorded.

`begin` also fails if no context configured a range replay service
(`ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND`) or if no such context is currently active
(`ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED`). `end` returns
`ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT` when no range is open, and otherwise returns success
whether the range was replayed or declined — the outcome is reported through the `CLOSE` callback's
`status`, not through the return value. `end` blocks the calling thread for the duration of the
replay.

### Payload

One flat struct carries all three operations; there are no unions. Which members are meaningful
depends on the current operation.

```c
typedef struct rocprofiler_callback_tracing_range_replay_data_t
{
    uint64_t size;
    uint64_t range_id;

    uint64_t (*pass_count_cb)(uint64_t range_id, rocprofiler_user_data_t user_data);
    int (*replay_continue_cb)(uint64_t range_id, uint64_t current_pass,
                              uint64_t total_passes, rocprofiler_user_data_t user_data);

    uint64_t current_pass;
    uint64_t total_passes;

    rocprofiler_status_t (*replay_local_start_context_cb)(rocprofiler_context_id_t);
    rocprofiler_status_t (*replay_local_stop_context_cb)(rocprofiler_context_id_t);

    rocprofiler_agent_id_t            agent_id;
    uint64_t                          dispatch_count;
    rocprofiler_range_replay_status_t status;
    uint64_t                          divergence_count;
} rocprofiler_callback_tracing_range_replay_data_t;
```

| Member | Populated on | Meaning |
|---|---|---|
| `range_id` | all | the id the tool passed to `begin` |
| `pass_count_cb` | `CONFIG` | **tool writes.** Total passes for this range, counting the application's own execution as pass 0 |
| `replay_continue_cb` | `CONFIG` | **tool writes, optional.** Continue-decision after each re-executed pass |
| `current_pass` | `PASS` | 1-based index of the re-executed pass |
| `total_passes` | `PASS` | what `pass_count_cb` returned, or 0 when open-ended |
| `replay_local_start_context_cb` / `..._stop_...` | `PASS` `PHASE_ENTER` | localized context toggles |
| `agent_id` | `PASS`, `CLOSE` | agent the range bound to; zero handle if nothing was recorded |
| `dispatch_count` | `PASS`, `CLOSE` | dispatches observed in the range |
| `status` | `CLOSE` | `REPLAYED`, or why the range was declined |
| `divergence_count` | `CLOSE` | snapshot regions that differed between pass 0 and the final pass |

`dispatch_count` at `CLOSE` reports the dispatches *observed*, not the dispatches retained. A
declined range drops its recorded packets immediately (they hold a kernarg copy each), but "how many
dispatches were in the phase" is still the number the tool wants to see.

`user_data` written by the tool during `CONFIG` is captured once and passed to every later callback
for that range, so a tool can carry per-range state without its own side table.

## Pass-count semantics

Identical to kernel replay's, with pass 0 reinterpreted: it is the application's own execution rather
than the first SDK-driven pass.

| `pass_count_cb` | Effect |
|---|---|
| left `NULL` | the range is observed but never re-executed (per-range opt-out) |
| returns `N > 1` | the recording is re-executed `N - 1` times after the live run |
| returns `0` | open-ended re-execution; requires `replay_continue_cb` |
| returns `1` | opt-out, same as `NULL` |

An opt-out is not silently dropped: the range still gets a `CLOSE` callback, with status
`ROCPROFILER_RANGE_REPLAY_STATUS_NO_PASS_COUNT`, so a tool can distinguish "I chose not to replay
this range" from "the SDK could not".

`replay_continue_cb` returns non-zero to keep going and zero to stop. It is consulted after each
re-executed pass, and is required when `pass_count_cb` returns 0.

## Localized context control

The `PASS` toggles have exactly the semantics of kernel replay's: valid only while the tool's `PASS`
`PHASE_ENTER` callback is running, sticky across passes of the range, scoped to that range's replay
loop, and unable to promote a context that is globally inactive. The implementation is the same
thread-local override machinery, reused unchanged from
`kernel_replay/local_context.hpp`. See
[Kernel replay — localized context control](../kernel_replay/kernel_replay_callback_api.md#localized-context-control)
for the full contract.

The one behavioral difference follows from pass 0 being the application's: a toggle cannot affect it.
By the time the tool's first `PASS` callback runs, the application's execution of the range is
already complete. A tool that wants a specific counter group on the live run must have it active
before `begin`.

## Configuring the service

Range replay is **single-subscriber**. A second context attempting to configure the domain gets
`ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED`, because a range runs one plan and two
subscribers would each supply their own pass count.

```c
rocprofiler_configure_callback_tracing_service(
    ctx,
    ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY,
    nullptr,  // all operations: CONFIG, PASS, CLOSE
    0,
    tool_range_replay_callback,
    nullptr);
```

Configuring the domain switches on the same allocation inventory kernel replay uses — the snapshot
cannot be taken without it — and enables HSA queue interception, since range replay both records
dispatches from `WriteInterceptor` and re-submits them through it.

Queue interposition (the write-pointer virtualization path) has its own submission handling that this
beta does not cover. When it is enabled, `begin` warns once and the range is recorded and then
declined with `ROCPROFILER_RANGE_REPLAY_STATUS_UNSUPPORTED_QUEUE_PATH`, rather than replayed through
a path that cannot re-submit recorded packets correctly.

## Source reference

All paths are relative to `projects/rocprofiler-sdk/`.

| Component | File | Symbol |
|---|---|---|
| Public header | `source/include/rocprofiler-sdk/experimental/range_replay.h` | — |
| `begin` / `end` | `source/lib/rocprofiler-sdk/range_replay/api.cpp` | `rocprofiler_range_replay_begin`, `..._end` |
| Subscription and single-subscriber check | `source/lib/rocprofiler-sdk/callback_tracing.cpp` | `rocprofiler_configure_callback_tracing_service` |
| Pass plan and callback delivery | `source/lib/rocprofiler-sdk/range_replay/replay_callbacks.cpp` | `execute_config_callback`, `execute_pass_phase_enter`, `execute_close_callback` |
| Pass-count / continue decisions | `source/lib/rocprofiler-sdk/range_replay/replay_callbacks.cpp` | `should_continue_replay()` |
| Service activity gates | `source/lib/rocprofiler-sdk/range_replay/replay_callbacks.cpp` | `has_active_range_replay_contexts()`, `has_registered_range_replay_context()` |
| Queue interception enablement | `source/lib/rocprofiler-sdk/hsa/queue_controller.cpp` | `enable_queue_intercept()` |
