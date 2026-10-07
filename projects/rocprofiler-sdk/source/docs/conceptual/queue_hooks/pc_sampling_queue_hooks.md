# PC Sampling Queue Hooks

How PC sampling is wired into dispatch interception without the per-queue callback registry. The
mechanism shared by the migrated services is described on the queue callback registry removal page.
PC sampling is the simplest case: it has no enter hook, and its exit hook never reads the context
registry. This page covers what is specific to it.

Paths are relative to `projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/`. Symbols are named
rather than cited by line number.

## 1. Hooks, predicates and call sites

| Element | Location |
|---|---|
| `pc_sampling::is_configured_on_agent` | `pc_sampling/queue_hooks.hpp`, `pc_sampling/queue_hooks.cpp` |
| `pc_sampling::kernel_dispatch_phase_exit_hook` | `pc_sampling/queue_hooks.hpp`, `pc_sampling/queue_hooks.cpp` |
| `pc_sampling::hsa::kernel_completion_cb` | `pc_sampling/hsa_adapter.hpp`, `pc_sampling/hsa_adapter.cpp` |
| Gate in `no_real_consumers` | `hsa/queue.cpp`, `WriteInterceptor` |
| Exit hook call site | `hsa/queue.cpp`, `AsyncSignalHandler` |
| Marker packet | `hsa/queue.cpp`, `WriteInterceptor`, `pc_sampling::hsa::generate_marker_packet_for_kernel` |

There is no enter hook. The marker packet that correlates samples with a dispatch was never produced
by the registered callback (its `write_interceptor` returned no packet); the write interceptor adds
it inline, just before the kernel packet, whenever `is_pc_sample_service_configured()` is true for
the queue's agent.

`is_configured_on_agent` wraps `is_pc_sample_service_configured` behind the
`ROCPROFILER_SDK_HSA_PC_SAMPLING` compile gate. `queue_hooks.cpp` is added to the object library
before PC sampling's early return in `pc_sampling/CMakeLists.txt`, while `service.cpp` (which
defines `is_pc_sample_service_configured`) is not built when PC sampling is compiled out, so the
gate has to stay at the call site for the write interceptor to link in that configuration.

## 2. What the migration changes

Before the migration, `pc_sampling_service_finish_configuration()` registered one callback for every
queue on every agent (`add_callback(std::nullopt, ...)`) when it finished configuring a service at
HSA initialization. Its `batch_packets` returned true, its `write_interceptor` returned no packet,
and its `signal_completion` called `kernel_completion_cb`. Its only lasting effect on the write path
was to make `queue.get_notifiers()` non-zero on every queue, which pulled every queue in the process
through interception.

Now:

- `no_real_consumers` includes `!pc_sampling::is_configured_on_agent(<queue's agent>)`. A queue is
  kept on the interception path only when its own agent has a PC sampling session, so a session on
  one GPU does not cost the other GPUs their fast path.
- Packet batching is unaffected, as before (the old callback allowed it).
- `PCSAgentSession` no longer carries the registry id (`intercept_cb_id`) that the old callback
  needed, since nothing is registered.

`is_pc_sample_service_configured` returns true when `is_hsa_initialized()` is set and the agent has
an entry in the global session map. `post_hsa_init_start_active_service()` sets that flag at HSA
initialization, after configuring the sessions that exist by then; if there are none, it returns and
leaves the flag clear. The predicate does not look at whether the service is started, so a
configured but stopped session still keeps its agent's queues on the interception path.

## 3. Completion

In `AsyncSignalHandler` the per-packet order is `kernel_dispatch::dispatch_complete`, the registry's
`signal_completion` loop, the PC sampling exit hook, the SPM and counter collection exit hooks, then
`profiler_serializer::kernel_completion_signal`.

`kernel_dispatch_phase_exit_hook` returns at once when there is no session, checks the queue and its
agent, returns when `is_pc_sample_service_configured()` is false for that agent, and otherwise calls
`hsa::kernel_completion_cb`. That function returns when the session has no correlation id, looks up
the agent's `PCSAgentSession` (fatal if missing), and tells its CID manager that the correlation
id's asynchronous activity has completed.

No context set is read. The routing key is the queue's agent, and the session found through it is
the owner, so the split between active and registered contexts that the other services need does
not arise. Sessions are never removed from the global session map, so a dispatch that completes
after its context has stopped still finds its session.

## 4. Start and stop

**Start.** `pc_sampling::start_service()` is the last thing `context::start_context` does, after the
start-in-progress marker has been released (section 4 of the queue callback registry removal page).
It compare-exchanges the service's `enabled` flag from false to true and returns
`ROCPROFILER_STATUS_ERROR` if the flag was already set; `context::start_context` returns that
status. If `is_hsa_initialized()` is set it starts sampling on each of the service's agent sessions;
otherwise `post_hsa_init_start_active_service()` starts every enabled service at HSA initialization.

**Stop.** `context::stop_context` treats PC sampling like device counter collection rather than like
the queue-interposed services: `pc_sampling::stop_service()` runs in the last phase, after the
active slot is cleared, while the context is still in the stopping set. It compare-exchanges
`enabled` from true to false and returns `ROCPROFILER_STATUS_ERROR` if the flag was not set;
`context::stop_context` ignores the result. If `is_hsa_initialized()` is set it stops sampling on
each agent session and flushes that agent's internal buffers. Nothing drains the GPU queues for PC
sampling.

## 5. Tests

All are in `pcs-test`, which sets `SKIP_REGULAR_EXPRESSION "PC sampling unavailable"`.

`pc_sampling/tests/queue_hooks_test.cpp`:

| Test | Pins |
|---|---|
| `pc_sampling_queue_hooks.is_configured_on_agent_unconfigured` | the gate is shut when no service is configured |
| `pc_sampling_queue_hooks.exit_hook_null_session_is_noop` | the exit hook returns without a session |
| `pc_sampling_queue_hooks.is_configured_on_agent_configured` | seeding the HSA-init flag and a global session-map entry opens the gate, and removing the entry with the flag still set shuts it again |

None of them needs a GPU. `is_configured_on_agent_configured` is compiled only when
`ROCPROFILER_SDK_HSA_PC_SAMPLING` is enabled. It seeds a null session entry, which is enough because
the gate only calls `find()`.

`pc_sampling/tests/local_context.cpp`: `pc_sampling.local_context_override_does_not_toggle_enabled`
and `pc_sampling.local_context_override_restart_does_not_toggle_enabled` pin that a kernel-replay
local stop or start is recorded but leaves the service's `enabled` flag alone.

## 6. Known gaps

1. **No test checks completion delivery through the configured path.** The unit tests cover only
   the null-session and gate cases. `pc-sampling-integration-test`
   (`projects/rocprofiler-sdk/tests/pc_sampling`) runs kernels with PC sampling configured on a real
   agent and traces correlation-id retirement, but it only prints the retired ids, so nothing
   asserts that the CID manager saw each completion.
2. **Start after the marker.** `pc_sampling::start_service()` runs after the start-in-progress
   marker is released, so a concurrent `rocprofiler_stop_context` can claim the context before PC
   sampling has started. Its `stop_service()` call then finds `enabled` clear and returns
   `ROCPROFILER_STATUS_ERROR`, which is ignored, and the start's own `start_service()` call then
   enables sampling on a context that is no longer active. Sampling stays on:
   `rocprofiler_stop_context` returns `ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND` for the inactive
   context, and the next `rocprofiler_start_context` activates it but returns
   `ROCPROFILER_STATUS_ERROR` because `enabled` is still set. Only a stop after that turns sampling
   off. Device counter collection, which also starts after the marker is released, has the same
   window.
3. **Configured, not started, still intercepts.** Because the gate keys off configuration, and
   sessions are never removed from the global map, an agent with a configured session keeps paying
   the interception cost for the rest of the process, whether or not the service is started. That is
   narrower than before the migration, when the registered callback made every agent pay it.
4. **Kernel replay passes cannot turn PC sampling off.** PC sampling does not read
   `kernel_replay::local_context_override()`, so it runs on every pass, including passes meant only
   for counter collection. The kernel-replay samples that use PC sampling are therefore disabled in
   ctest (`KR_PC_SAMPLING_PASS_PARTITION_UNSUPPORTED` in
   `projects/rocprofiler-sdk/samples/kernel_replay/CMakeLists.txt`).
