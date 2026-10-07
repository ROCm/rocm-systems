# Thread Trace Queue Hooks

Dispatch thread trace does not register with the per-queue callback registry
(`QueueController::add_callback`). The queue interception code in `hsa/queue.cpp` calls its enter
and exit hooks directly. The mechanism shared by the services that no longer use the registry, and
the reason the enter hook reads active contexts while the exit hook reads registered ones, are
described on the *Queue Callback Registry Removal — Software Architecture* page. This page covers
what is specific to thread trace.

Paths are relative to `projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/`. Symbols are named
rather than cited by line number. Test names are GoogleTest names; in the build tree, ctest prefixes
each one with `unit.`.

Only dispatch thread trace (`context::dispatch_thread_trace`, a `DispatchThreadTracer`) has queue
hooks. Device thread trace (`context::device_thread_trace`, a `DeviceThreadTracer`) adds no packets
to dispatches: `hsa/queue.cpp` refers to it only when an intercepted queue is created, to enable
profiling on that queue. It never registered with the registry and is not covered here.

## 1. Hooks, predicates and call sites

| Element | Location |
|---|---|
| `thread_trace::kernel_dispatch_phase_enter_hook` | `thread_trace/queue_hooks.hpp`, `thread_trace/queue_hooks.cpp` |
| `thread_trace::kernel_dispatch_phase_exit_hook` | `thread_trace/queue_hooks.hpp`, `thread_trace/queue_hooks.cpp` |
| `thread_trace::is_any_active`, `thread_trace::is_active_on_agent` | `thread_trace/queue_hooks.hpp`, `thread_trace/queue_hooks.cpp` |
| Context filter used by both hooks and both predicates: `ctx && ctx->dispatch_thread_trace != nullptr` | `thread_trace/queue_hooks.cpp`, `thread_trace_contexts_filter()` |
| Owner stamp: `GetOwner`/`SetOwner` | `hsa/aql_packet.hpp`, `TraceControlAQLPacket` |
| Producer tag `THREAD_TRACE_CLIENT_ID` | `hsa/queue_hooks/client_ids.hpp` |
| Enter hook call site | `hsa/queue.cpp`, the `process_packet_batch` lambda in `WriteInterceptor` |
| Exit hook call site | `hsa/queue.cpp`, `AsyncSignalHandler` |

`WriteInterceptor` evaluates `thread_trace::is_active_on_agent()` once per call, for the queue's
agent. The result is folded into `no_real_consumers`, so a queue on an agent with an active dispatch
thread trace context is intercepted even when nothing is registered with the registry. It also sets
`should_batch_packets = false`, so `process_packet_batch` runs once per packet of the write instead
of once for the whole write. Inside `process_packet_batch`, the enter hook runs after the registry's
`write_interceptor` loop and the SPM enter hook, and before the serializer's `kernel_dispatch`
barriers are added, so the serialization request it ORs into `is_serialized` takes effect on the
same dispatch.

`AsyncSignalHandler` runs on the HSA runtime's async signal handler thread. When a session's
completion signal fires, it handles each dispatch in the session in turn: it calls
`kernel_dispatch::dispatch_complete`, the registry's `signal_completion` loop, the PC sampling exit
hook, the SPM exit hook, the thread trace exit hook, and then
`profiler_serializer::kernel_completion_signal`. The thread trace exit hook, and the tool's
shader-data callback inside it, therefore finish before the serializer releases the next serialized
dispatch on that agent.

`is_any_active()` has no caller outside the tests; the interceptor uses only the per-agent form.

When dispatch thread trace used the registry, `DispatchThreadTracer::start_context` added one
registry entry covering every queue on every agent (`add_callback(std::nullopt, ...)`), with
`batch_packets` returning false, and only `resource_deinit()` removed it. After the first start,
every queue in the process was intercepted and unbatched until finalization. Both now apply only to
queues on agents that an active context is configured for, and only while that context is active.

## 2. Which contexts each phase reads

- The **enter hook** walks `context::get_active_contexts` with the filter, skips a tracer whose
  `collects_on()` is false for the queue's agent, and calls `DispatchThreadTracer::pre_kernel_call`.
  A returned packet is appended to `inst_pkt` tagged `THREAD_TRACE_CLIENT_ID`, and the returned
  serialization request is ORed into `is_serialized`. Only an active context can add
  instrumentation.
- The **exit hook** returns at once when no `inst_pkt` entry is tagged `THREAD_TRACE_CLIENT_ID`, as
  the SPM exit hook does, so only completions of traced dispatches read the registry. Otherwise it
  walks `context::get_registered_contexts` with the filter and calls `post_kernel_call` on every
  dispatch thread tracer in that snapshot, active or not. A dispatch still on the GPU when its
  context stops must still deliver its trace, which the active set cannot provide. Each tracer's
  `post_kernel_call` returns at once when it has no packet outstanding.

The exit hook holds raw context pointers after the snapshot they were read from is released. That
is safe because `deregister_client_contexts` retires a context instead of destroying it, so the
pointer stays valid until static teardown (see the comment on `get_registered_contexts` in
`context/context.hpp`). A retired context is no longer in the snapshot, so the exit hook does not
visit it. Outside the tests, `deregister_client_contexts` is called only for a tool whose configure
function returns no result.

## 3. What `pre_kernel_call` returns

`DispatchThreadTracer::pre_kernel_call` returns a packet, or none, and a serialization request. It
takes a shared lock on `agents_map_mut` and holds it for the rest of the call, including the tool's
dispatch callback. The checks run in this order:

1. The queue's agent has no entry in `agents`: no packet and no serialization request.
   `resource_init()` fills `agents` with one `ThreadTracerAgent` for each configured agent that HSA
   exposes; `thread_trace::initialize()` calls it when the HSA runtime registers.
   `resource_deinit()` empties it (section 5).
2. `enabled` is clear: no packet; the serialization request is the agent's `bSerialize` setting.
3. A kernel replay pass has forced this context off (`kernel_replay::local_context_override`): no
   packet; `bSerialize`.
4. The tool's dispatch callback returns `ROCPROFILER_THREAD_TRACE_CONTROL_NONE`: no packet;
   `bSerialize`.
5. `ThreadTracerAgent::get_start_packet()` returns null because the agent's trace resources do not
   exist yet: no packet; `bSerialize`. In the default resource mode,
   `ROCPROFILER_THREAD_TRACE_PARAMETER_RESOURCE_MODE_CODE_OBJECT`, the resources are built when the
   agent's first code object is loaded, or when the `ThreadTracerAgent` is constructed if one is
   already loaded. `ROCPROFILER_THREAD_TRACE_PARAMETER_RESOURCE_MODE_HSA` builds them when the
   `ThreadTracerAgent` is constructed.
6. Otherwise the packet is stamped (section 4), `post_move_data` is incremented, the before- and
   after-kernel packets are populated, and the serialization request is true.

`bSerialize` is set by `ROCPROFILER_THREAD_TRACE_PARAMETER_SERIALIZE_ALL`. With it, every dispatch
on the agent that reaches `pre_kernel_call` asks for serialization, whether or not it is traced.

## 4. How a completion finds its owner

Walking every registered tracer is correct only if each tracer claims its own packets and no others.
Thread trace keeps no per-packet map, so ownership travels on the packet:

1. `pre_kernel_call` takes a start packet, a copy of the control packet of the agent's
   `ThreadTracerAgent`; `ThreadTracerAgent::get_start_packet()` increments that agent's
   `active_traces`. `pre_kernel_call` then calls `SetOwner(tracer_id)` and increments
   `post_move_data`.
2. `tracer_id` comes from `DispatchThreadTracer::allocate_tracer_id()`, a process-wide counter that
   starts at 1 and is never reused. A packet that outlives its tracer cannot be claimed by a later
   tracer that happens to occupy the same address, and a packet's default owner, 0, matches no
   tracer.
3. `post_kernel_call` returns at once when `post_move_data` is below 1. Otherwise it skips
   `inst_pkt` entries that are not tagged `THREAD_TRACE_CLIENT_ID`, are not a
   `TraceControlAQLPacket`, or have `GetOwner() != tracer_id`. For a packet it owns, it takes a
   shared lock on `agents_map_mut` and decrements `post_move_data`. If the packet has after-kernel
   packets, which a packet built by `pre_kernel_call` always has, it looks up `GetAgent()` in
   `agents` and calls `ThreadTracerAgent::iterate_data()` with the packet's handle and the
   dispatch's user data. `iterate_data()` passes the trace data to the tool's shader-data callback
   and decrements `active_traces`.

The packet holds no reference to its `ThreadTracerAgent`; `agents` owns every agent. A completion
that arrives after `resource_deinit()` has emptied `agents` finds no agent and delivers nothing, but
it still decrements `post_move_data`. What the agent's destructor delivers instead is in section 8.

When dispatch thread trace used the registry, `post_kernel_call` claimed every
`TraceControlAQLPacket` in `inst_pkt`, whichever tracer had produced it. The lookup in `agents`, and
a completion after `resource_deinit()` delivering nothing, are unchanged from then.

## 5. Start, stop and finalization

`context::start_context` and `context::stop_context` release the contexts mutex while services
start and stop. The registry's in-progress set (`registry_state::stopping`, returned by
`get_stopping_contexts()`) holds the id of every context whose start or stop is under way.
`wait_for_stopping_contexts()` waits until that set is empty, and `context::start_context`,
`context::stop_context`, `deactivate_client_contexts` and `deregister_client_contexts` each call it
under the mutex before they act. A start or stop in progress therefore holds off every other start,
stop, deactivate and deregister, not only those on overlapping agents.

**Start.** Under the mutex, `context::start_context` waits for the in-progress set to empty, runs
the conflict scan (section 6), reserves an active slot, increments the active count, and adds the
context to the in-progress set. It then drops the mutex, notifies queue interposition, publishes the
context into the slot with a compare-exchange, and starts dispatch counter collection, SPM, device
thread trace and dispatch thread trace, in that order. `DispatchThreadTracer::start_context` calls
`enable_serialization(configured_agents())` and then sets `enabled`. Serialization is acquired
first, so a dispatch that sees `enabled` set finds its agent's serializer already enabled.
`DispatchThreadTracer::start_context` does not check `enabled` before it acquires; it relies on
`context::start_context`, which returns early for a context that is already active.

Between the publish and `DispatchThreadTracer::start_context`, the enter hook already sees the
context. With `enabled` still clear, `pre_kernel_call` returns no packet and the `bSerialize`
setting, as it does during a stop.

The context stays in the in-progress set until those four services have started, so a concurrent
stop, deactivate, deregister or start waits until the tracer has acquired serialization and set
`enabled`. PC sampling also starts before the context leaves the set; device counter collection
starts after it (`counters::start_agent_ctx()`, which calls the tool's profile callback
synchronously). No thread trace test drives a concurrent start and stop.
`spm_core.concurrent_context_start_stop_does_not_pin_serialization` and
`spm_core.concurrent_overlapping_context_starts_admit_exactly_one` in `spm/tests/core.cpp` exercise
the same `context::start_context` path with SPM contexts.

**Stop.** `context::stop_context` runs in four phases:

1. Locked: wait for the in-progress set to empty, find the context's active slot, take a reference
   to the context, and add its id to the in-progress set.
2. Unlocked, slot still populated: stop dispatch counter collection, SPM, device thread trace and
   dispatch thread trace. `DispatchThreadTracer::stop_context` clears `enabled` and, only if it was
   set, calls `disable_serialization(configured_agents())`.
3. Locked: compare-exchange the slot to null and decrement the active count.
4. Unlocked: notify queue interposition, then stop device counter collection and PC sampling.

A scope guard removes the id from the in-progress set and wakes waiters when `context::stop_context`
returns.

Between phases 2 and 3 the context is still active, so the enter hook still reaches
`pre_kernel_call`. With `enabled` clear it returns no packet and the `bSerialize` setting, so a
tracer configured with `ROCPROFILER_THREAD_TRACE_PARAMETER_SERIALIZE_ALL` keeps untraced dispatches
serialized until `disable_serialization` has run. When dispatch thread trace used the registry,
`pre_kernel_call` checked `enabled` first and returned no serialization request.

What the stop guarantees:

- A `pre_kernel_call` that reads `enabled` after the stop has cleared it injects no packet. The stop
  does not wait for calls already past that check, so a dispatch whose `pre_kernel_call` read
  `enabled` before it was cleared can still inject a packet, including after
  `rocprofiler_stop_context` has returned.
- A traced dispatch still in flight is delivered later through the exit hook, on the HSA async
  signal handler thread.
- The serialization reference the tracer took at start is released once: by the stop that clears
  `enabled`, or, if no stop ran, by `resource_deinit()` at finalization.

There is **no drain**. For a context whose only service is dispatch thread trace, nothing on the
stop path waits for in-flight dispatches: `DispatchThreadTracer::stop_context` does not call
`hsa::queue_controller_sync()`, and `notify_queue_interposition_consumer_context_stopped` at most
decrements a counter. `rocprofiler_stop_context` can return while traced dispatches are
still running, and the shader-data callback can run after it returns. When dispatch thread trace
used the registry, its stop did not drain either.

**Finalization.** `registration::finalize` calls `hsa::queue_controller_fini()`, which begins with
`hsa::queue_controller_sync()`, and then `thread_trace::finalize()`, which calls `resource_deinit()`
on every dispatch thread tracer in the registered snapshot. `resource_deinit()` releases
serialization only if `enabled` is still set, so it never releases a second time after a stop. It
then swaps `agents` out under `agents_map_mut` and destroys the `ThreadTracerAgent`s only after
releasing the lock. A completion can be waiting for that lock in `post_kernel_call` on the HSA async
signal handler thread, and on the ROCr/HSA thread-trace backend (used where direct KFD thread trace
is unavailable, or forced with `ROCPROFILER_SQTT_FORCE_HSA=1`) an agent's destructor calls
`hsa_queue_destroy()`, which waits on that same thread. Destroying the agents under the lock, or on
that thread, hangs the process at exit.
`thread_trace.resource_deinit_does_not_release_foreign_serialization_owner` pins that a stop
followed by `resource_deinit()` leaves another owner's serialization in place.

## 6. Per-agent gating

A dispatch thread trace context is configured one agent at a time, and only while the SDK is
initializing, that is, from a tool's configure or initialize function. Each
`rocprofiler_configure_dispatch_thread_trace_service(context, agent, ...)` call that succeeds sets
the agent's entry in the tracer's `params` map.

- `collects_on(agent)` is true when the agent is in `params`. The enter hook and
  `is_active_on_agent` both use it.
- `intersects(other)` is true when the two `params` maps share an agent. A tracer compared with
  itself returns `!params.empty()`.
- `configured_agents()` returns the keys of `params`. Serialization is acquired and released for
  exactly that set, through the queue controller's per-agent serialization refcount
  (`QueueController::update_serialization`). An agent's serializer is toggled only when its
  effective count crosses zero, and a release at zero is ignored.
- `context::start_context` rejects a dispatch thread trace context whose configured agents intersect
  those of an active dispatch thread trace context with `ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT`.
  Contexts on disjoint agents can be active together.

The queue controller treats an empty agent set as every GPU agent.
`rocprofiler_configure_dispatch_thread_trace_service` creates the tracer before it validates its
arguments, so a call that fails validation can leave a context with a tracer whose `params` is
empty. Such a tracer traces nothing, is not active on any agent and conflicts with no other context,
but starting its context acquires serialization on every GPU agent.

When dispatch thread trace used the registry, it acquired and released serialization for every
agent, and `context::start_context` had no conflict rule for dispatch thread trace contexts.

## 7. Tests

`thread-trace-queue-hooks-test`, `thread_trace/tests/queue_hooks_test.cpp`:

| Test | Pins |
|---|---|
| `ThreadTraceQueueHooks.IsAnyActiveReturnsFalseWhenNoContextActive` | `is_any_active()` is false with no context started |
| `ThreadTraceQueueHooks.IsActiveOnAgentReturnsFalseWhenNoContextActive` | `is_active_on_agent()` is false for two agent handles with no context started |
| `ThreadTraceQueueHooks.StopContextInFlightCompletionRoutesViaHookPath` | a packet injected through the enter hook before `rocprofiler_stop_context` is claimed through the exit hook afterwards, taking `pending_post_moves()` back to 0 |
| `ThreadTraceQueueHooks.CompletionRoutingStaysWithTheProducingTracer` | two tracers on one agent, the second started after the first is stopped: each tracer's completion is claimed by that tracer only |
| `ThreadTraceQueueHooks.CompletionAfterResourceDeinitStillDrains` | a completion that arrives after `resource_deinit()` has emptied `agents` still takes `pending_post_moves()` back to 0, and delivers nothing |
| `ThreadTraceQueueHooks.StoppingContextKeepsSerializeAllRequest` | after a SERIALIZE_ALL tracer's context is stopped, `pre_kernel_call` returns no packet and a serialization request |

The last four call `GTEST_SKIP()` when the queue controller has no supported agent that HSA exposes.
The skip message says no ATT-capable agent is available, but the check does not test ATT support.
Each of these four calls `resource_init()` itself to fill its tracer's `agents` map, and drives the
hooks with fake queues; the injected packets are never submitted to the GPU.

`thread-trace-per-agent-test`, `thread_trace/tests/per_agent_scoping_test.cpp`:

| Test | Pins |
|---|---|
| `thread_trace_per_agent.collects_on_and_intersects_match_configured_agents` | `collects_on`, `intersects` and `configured_agents` on two tracers built directly |
| `thread_trace_per_agent.disjoint_agent_contexts_can_be_active_together` | contexts on disjoint agents both start |
| `thread_trace_per_agent.is_active_on_agent_follows_the_configured_agents` | the interceptor predicate is false before start, true only for the configured agent while active, and false after stop |
| `thread_trace_per_agent.overlapping_agent_contexts_still_conflict` | a second context on the same agent gets `ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT` |
| `thread_trace_per_agent.serialization_is_scoped_to_the_contexts_agents` | serialization is enabled only on the configured agent, and released on stop |

`overlapping_agent_contexts_still_conflict` needs one supported agent; the others need two and skip
otherwise. This target sets `SKIP_REGULAR_EXPRESSION`, so ctest reports those skips as skipped. None
of these tests calls `resource_init()`, so every tracer's `agents` map stays empty.

`thread-trace-packet-test`, `thread_trace/tests/local_context.cpp`:

| Test | Pins |
|---|---|
| `thread_trace.resource_deinit_does_not_release_foreign_serialization_owner` | with serialization also held by another owner, a tracer's start, stop and `resource_deinit()` leave it enabled |
| `thread_trace.local_context_override_skips_pre_kernel_call` | meant to pin that a replay pass forcing the context off skips the dispatch callback; skips (section 8) |
| `thread_trace.local_context_override_forced_on_still_invokes_dispatch_cb` | meant to pin that a replay pass forcing the context on still calls the dispatch callback; skips (section 8) |

## 8. Known gaps

1. **No drain on stop.** Shader data for a dispatch in flight at stop arrives after
   `rocprofiler_stop_context` returns, on the HSA async signal handler thread (section 5).
2. **A packet injected during the stop.** The stop does not wait for `pre_kernel_call` calls that
   are already past the `enabled` check (section 5). Such a dispatch is traced and its trace is
   delivered through the exit hook. If its agent's serializer has been disabled by the time the
   dispatch's serializer packets are built, the serializer adds no ready or block barriers for it.
3. **A completion after finalization.** `registration::finalize` syncs the queues before
   `thread_trace::finalize()`, but a traced dispatch can outlive that sync. If it is still in flight
   when `resource_deinit()` destroys its `ThreadTracerAgent`, the destructor, on the finalizing
   thread, stops the trace and waits for the stop. With a single trace buffer it then passes the
   data collected so far to the shader-data callback, with the tracer's callback user data instead
   of the dispatch's. The dispatch's completion later finds no agent and delivers nothing
   (section 4).
4. **Tests that skip or stop short.**
   - `thread-trace-queue-hooks-test` sets no `SKIP_REGULAR_EXPRESSION`, so when its four GPU tests
     skip, ctest reports them as passed.
   - `thread-trace-packet-test` sets no `SKIP_REGULAR_EXPRESSION` either. Its two
     `local_context_override_*` tests never start the tracer, so `pre_kernel_call` returns at the
     `enabled` check before it calls the dispatch callback. The baseline check that each test runs
     before its override assertions therefore fails, and both tests skip wherever their setup
     succeeds; ctest reports them as passed.
   - `StoppingContextKeepsSerializeAllRequest` calls `pre_kernel_call` after the stop has finished.
     It pins the return value, not a dispatch going through the enter hook during the stop.
   - No test checks that the enter hook skips a queue on an agent the context is not configured
     for. The per-agent tests never call `resource_init()`, so with an empty `agents` map
     `pre_kernel_call` would return no packet for such a queue anyway.
   - No thread trace test drives a concurrent start and stop (section 5).
