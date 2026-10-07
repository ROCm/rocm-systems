# SPM Queue Hooks

How dispatch SPM (`context::dispatch_spm`, a `spm_dispatch_counter_collection_service`) is wired
into dispatch interception without the per-queue callback registry. The mechanism shared by the
migrated services, and why the enter hook reads active contexts while the exit hook reads
registered ones, is described on the queue callback registry removal page. This page covers what
is specific to SPM.

Paths are relative to `projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/`. Symbols are named
rather than cited by line number.

SPM is experimental (`rocprofiler-sdk/experimental/spm.h`). Its configure calls and
`rocprofiler_spm_dispatch_counting_service_set_agents` return
`ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED` unless `ROCPROFILER_SPM_BETA_ENABLED` is set.

## 1. Hooks, predicates and call sites

| Element | Location |
|---|---|
| `spm::kernel_dispatch_phase_enter_hook` | `spm/queue_hooks.hpp`, `spm/queue_hooks.cpp` |
| `spm::kernel_dispatch_phase_exit_hook` | `spm/queue_hooks.hpp`, `spm/queue_hooks.cpp` |
| `spm::is_any_active`, `spm::is_active_on_agent` | `spm/queue_hooks.hpp`, `spm/queue_hooks.cpp` |
| Context filter shared by all four, `dispatch_spm != nullptr` | `spm/queue_hooks.cpp`, `spm_contexts_filter()` |
| Per-dispatch work: `spm::pre_kernel_call`, `spm::post_kernel_call` | `spm/dispatch_handlers.cpp` |
| Owner record: `spm_counter_callback_info::packet_return_map` | `spm/core.hpp` |
| Producer tag `SPM_CLIENT_ID` | `hsa/queue_hooks/client_ids.hpp` |
| Enter hook call site | `hsa/queue.cpp`, `WriteInterceptor` |
| Exit hook call site | `hsa/queue.cpp`, `AsyncSignalHandler` |

In `WriteInterceptor`, `spm::is_active_on_agent()` is evaluated once per call for the queue's
agent. It is folded into `no_real_consumers`, so a queue on an agent with an active SPM context is
intercepted even when nothing is registered with the registry, and it sets
`should_batch_packets = false`. The enter hook runs after the registry's `write_interceptor` loop
and before the serializer's `kernel_dispatch` barriers are added, so the `bSerial` it ORs into
`is_serialized` takes effect on the same dispatch.

In `AsyncSignalHandler` the per-packet order is `kernel_dispatch::dispatch_complete`, the
registry's `signal_completion` loop, the PC sampling, SPM and counter collection exit hooks, then
`profiler_serializer::kernel_completion_signal`. `kfd_stop()` and the DISPATCH_END record therefore
happen before the serializer lets the next serialized dispatch go.

`is_any_active()` has no caller outside the tests; the interceptor uses only the per-agent form.

Before the migration, `spm::start_context` added one registry entry per SPM callback, each covering
every queue on every agent (`add_callback(std::nullopt, ...)`) with `batch_packets` returning false,
and `spm::stop_context` removed them after its drain. While any SPM context was active, every queue
in the process was intercepted and unbatched. Both now apply only to queues on agents that an active
SPM context collects on.

## 2. Which contexts each phase reads

- The **enter hook** walks `context::get_active_contexts` with the filter, skips a context whose
  `collects_on()` is false for the queue's agent, and calls `pre_kernel_call` once per callback in
  `dispatch_spm->callbacks`. A returned packet is appended to `inst_pkt` tagged `SPM_CLIENT_ID`,
  and the returned `bSerial` is ORed into `is_serialized`. Only an active context may add
  instrumentation. For an SPM context, every return path of `pre_kernel_call` asks for
  serialization. That includes the disabled path, which returns an `EmptyAQLPacket` instead of an
  SPM packet (section 3).
- The **exit hook** returns at once unless some `inst_pkt` entry is tagged `SPM_CLIENT_ID`. It then
  walks `context::get_registered_contexts` with the filter and calls `post_kernel_call` for every
  callback of every SPM context, active or not. Two kinds of dispatch can complete after their
  context has left the active set: one whose queue the stop's drain gave up on (section 4), and
  one submitted between the end of the drain and the slot being cleared, which carries an empty
  fallback entry. Both must still be claimed.

The exit hook holds raw context pointers after the snapshot they were read from is released. That
is safe because `deregister_client_contexts` retires a context instead of destroying it, so the
pointer stays valid until static teardown (see the comment on `get_registered_contexts` in
`context/context.hpp`).

## 3. How a completion finds its owner

Each SPM callback keeps `packet_return_map`, keyed by the address of every packet it produced:

1. `pre_kernel_call` records each packet it returns. An `SPMPacket` is recorded with the
   `spm_counter_config` it was built from. The `EmptyAQLPacket` fallback is recorded with a null
   config. The fallback is returned when `enabled` is clear (or a kernel-replay pass has switched
   the context off), when no dispatch callback is set, when the tool's callback selects no config,
   and when creating the start-handshake signals or registering their handler fails.
2. `post_kernel_call` takes the map's write lock and looks up every `inst_pkt` entry by address,
   whatever its tag, and erases each entry it finds. For an `SPMPacket` it also:
   - destroys the second handshake barrier's dependency signal;
   - calls `kfd_stop()`;
   - calls the record callback with `ROCPROFILER_SPM_RECORD_FLAG_DISPATCH_END`, if one is set;
   - clears the packet and moves it out of `inst_pkt`;
   - after the lock, returns it to the config's packet cache.
3. A callback that did not produce the packet has no entry for its address and does nothing.

The address identifies the producer only while the entry exists. Every intercepted dispatch's
session reaches `AsyncSignalHandler`, and the entry is erased there before the packet is reused or
freed. The exception is the handler's early return after finalization
(`registration::get_fini_status() > 0`), which frees the session without running the exit hook.
`WriteInterceptor` forwards packets untouched from then on, so the stale entry is never looked up.

## 4. Start and stop

**Start.** Under the contexts mutex, `context::start_context` waits until no context is being
started or stopped (`wait_for_stopping_contexts`), runs the conflict scan (section 5), reserves an
active slot, increments the active count and adds the context's id to the stopping set, which
doubles as a start-in-progress marker. It then drops the mutex, publishes the context into the slot
with a compare-exchange, and starts the services. `spm::start_context` calls
`enable_serialization(agents)` and then sets `enabled`. Serialization is acquired first, so the
first dispatch that sees `enabled` is already serialized. The acquire is unconditional (section 7).
The marker is held until dispatch counter collection, SPM, both thread trace services and PC
sampling have started, so a stop, another start, `deactivate_client_contexts`,
`deregister_client_contexts` or `spm::set_dispatch_agents` that arrives in the meantime waits until
those services have started.

**Stop.** `context::stop_context` runs in four phases:

1. Locked: wait until no context is being started or stopped, find the context's slot, take a
   reference to the context, and add its id to the stopping set.
2. Unlocked, slot still populated: stop the queue-interposed services. `spm::stop_context` clears
   `enabled` and, only if it was the call that cleared it, then calls
   `hsa::queue_controller_sync()` and `disable_serialization(agents)`. A stop that finds `enabled`
   already clear returns without draining or releasing serialization.
3. Locked: compare-exchange the slot to null and decrement the active count.
4. Unlocked: notify queue interposition, then stop device counter collection and PC sampling.

A scope guard removes the context from the stopping set when `stop_context` returns.

Between phases 2 and 3 the context is still active, so the enter hook still reaches
`pre_kernel_call`. With `enabled` clear, it returns the empty fallback and `serialize=true`, so
dispatches submitted during the drain stay serialized until `disable_serialization` has run. The
comments in `spm::stop_context` and `context::stop_context` explain why this ordering matters: if
the slot were cleared first, the enter hook would stop asking for serialization while the agent's
serializer was still enabled. Before the migration the same effect came from the registry entries,
which `spm::stop_context` removed only after `disable_serialization`.

**Drain.** `hsa::queue_controller_sync()` calls `Queue::sync()` on every queue the controller
holds, on every agent, not only the context's. `Queue::sync()` waits a single five-second slice
(`drain_slice` in `hsa/queue.cpp`) for the queue's in-flight count to reach zero, warns if it does
not, and returns false. `hsa::queue_controller_sync()` returns true only if every queue drained, and
`spm::stop_context` does not look at the result. The in-flight count is decremented by
`Queue::async_complete()` at the end of `AsyncSignalHandler`, after the exit hooks. A queue that
drains has therefore run the exit hook for every SPM dispatch it had in flight.

What a stop that clears `enabled` guarantees:

- Once `enabled` is clear, `pre_kernel_call` builds no SPM packet. Dispatches on the context's
  agents for the rest of the stop get the empty fallback, and are serialized until
  `disable_serialization` has run.
- If every queue drains within its slice, every SPM dispatch in flight at the stop has had its
  `kfd_stop()` and DISPATCH_END record before serialization is released and before
  `rocprofiler_stop_context` returns.
- If a drain times out, the stop still releases serialization and returns. The late completion is
  claimed afterwards through the exit hook, on the HSA async handler thread.
- It releases exactly one serialization reference.

## 5. Per-agent gating

`spm_dispatch_counter_collection_service::agents` is the set of GPU agents the context collects on.
An empty set means every GPU agent, which is what the configure calls produce.
`rocprofiler_spm_dispatch_counting_service_set_agents(context, agents, num_agents)` replaces the
set, and `num_agents` of zero restores every agent. It returns:

- `ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI` if the aqlprofile SPM interface cannot be constructed;
- `ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED` without `ROCPROFILER_SPM_BETA_ENABLED`;
- `ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT` for a null `agents` with a nonzero count, or a
  non-GPU agent;
- `ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID` for an unregistered context;
- `ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND` for a context without an SPM dispatch service;
- `ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED` for an active context;
- `ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND` for an unknown agent.

The header states the call order (after configure, before start) but lists none of these codes.

- `collects_on(agent)` is true when the set is empty or contains the agent. The enter hook and
  `is_active_on_agent` both use it.
- `intersects(other)` is true when either set is empty or the two share an agent.
- Serialization is acquired and released for `agents` through the queue controller's per-agent
  serialization refcount (`QueueController::update_serialization`). An empty set takes the unscoped
  `all` count, which thread trace also uses; counter collection passes its own agent set, as SPM
  does. An agent's serializer is toggled only when its effective count (`all` plus its own) crosses
  zero, and a decrement below zero is ignored. Before the migration `enable_serialization()` and
  `disable_serialization()` were plain switches, so an SPM stop turned serialization off under a
  counter collection or thread trace context that was still running.
- `context::start_context` rejects an SPM context whose agent set intersects an active SPM
  context's with `ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT`. Contexts on disjoint agents run
  together, and two unrestricted SPM contexts conflict. Before the migration there was no such
  rule: two SPM contexts could be active at once and both instrumented every dispatch.
- `spm::set_dispatch_agents` holds the contexts mutex for the whole call, waits in
  `wait_for_stopping_contexts` for any start or stop in progress, and refuses a context that is in
  the active array. Together with the start marker, this keeps the agent set fixed from the
  conflict scan in `context::start_context` until the context's stop has finished, so
  `spm::stop_context` releases serialization for the same agents that `spm::start_context`
  acquired it for.
- The drain in `spm::stop_context` is not scoped (section 4). A context restricted to one agent
  still waits on the queues of every agent.

## 6. Tests

All SPM unit tests are in `spm-counter-test`, which sets `ROCPROFILER_SPM_BETA_ENABLED=True` and
`SKIP_REGULAR_EXPRESSION "SPM unavailable"`.

`spm/tests/queue_hooks_test.cpp`. These tests need no HSA runtime: the two hook tests pass a null
queue or session, and the hooks return before touching it.

| Test | Pins |
|---|---|
| `spm_queue_hooks.is_any_active_false_when_no_context_active` | `is_any_active()` with no context started |
| `spm_queue_hooks.is_active_on_agent_false_when_no_context_active` | `is_active_on_agent()` with no context started |
| `spm_queue_hooks.exit_hook_skips_when_inst_pkt_has_no_spm_client_id` | the exit hook returns without touching its null session when no entry is tagged `SPM_CLIENT_ID` |
| `spm_queue_hooks.enter_hook_noop_when_no_context_active` | with no active context the enter hook adds nothing and leaves `is_serialized` unchanged |

`spm/tests/core.cpp`:

| Test | Pins |
|---|---|
| `spm_queue_hooks.stop_context_in_flight_completion_routes_via_hook_path` | a packet that `spm::pre_kernel_call` recorded before `rocprofiler_stop_context` is erased from `packet_return_map` by the exit hook afterwards |
| `spm_core.start_stop_callback_ctx`, `spm_core.start_stop_buffered_ctx` | start sets `enabled` and stop clears it |
| `spm_core.stop_context_removes_callbacks` | `enabled` is clear after stop |
| `spm_core.stop_context_sync_and_restart` | a stopped context starts again with `enabled` set |
| `spm_core.set_agents_restricts_collection` | `collects_on` follows `set_agents`, and the restricted context starts and stops |
| `spm_core.disjoint_contexts_no_conflict` | a second context on the same agent gets `ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT`; contexts on two disjoint agents both start |
| `spm_core.concurrent_context_start_stop_does_not_pin_serialization` | two threads each start and stop one context, restricted to one agent, 1000 times; every start succeeds, every stop returns success or `ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND`, and afterwards the agent is not serialized, no SPM context is active and `enabled` is clear |
| `spm_core.concurrent_overlapping_context_starts_admit_exactly_one` | in each of 1000 iterations, two contexts restricted to the same agent are started from two threads; exactly one start succeeds and the other returns `ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT`, and once the winner is stopped the agent is not serialized and no SPM context is active |
| `spm_core.set_agents_does_not_race_context_start` | one context holds agent 0 while a thread keeps moving a second context between agents 0 and 1 with `set_agents`, which returns success or `ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED`; each of 1000 starts of the second context succeeds or returns `ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT`, and a start that succeeds never collects on agent 0; afterwards no agent is serialized and no SPM context is active |

`stop_context_in_flight_completion_routes_via_hook_path`, `set_agents_restricts_collection`,
`disjoint_contexts_no_conflict` and the first two concurrency tests log "SPM unavailable" when there
is no SPM-capable agent, so ctest reports them as skipped; `set_agents_does_not_race_context_start`
logs "SPM unavailable on two agents" with fewer than two. `disjoint_contexts_no_conflict` runs its
disjoint half only with two SPM-capable agents. With one agent it passes on the same-agent half
alone.

Each concurrency test repeats its race 1000 times within the test and finalizes the SDK when it
finishes. The comment above each names the interleaving that the start-in-progress marker closes.

`spm/tests/local_context.cpp`: `spm_core.local_context_override_stops_pre_kernel_call`,
`spm_core.local_context_override_restarts_pre_kernel_call` and
`spm_core.local_context_start_cannot_promote_globally_stopped` pin the kernel-replay override in
`pre_kernel_call`'s enabled check. The last one also shows that with `enabled` clear the tool's
callback is not called and a packet, the empty fallback, is still returned. None of the three
checks the serialize flag.

## 7. Known gaps

1. **Serialization reference taken on every start.** `spm::start_context` calls
   `enable_serialization(agents)` even when `enabled` is already set, while `spm::stop_context`
   releases only when it clears `enabled`. `context::start_context` reaches `spm::start_context`
   only for a context that is not active, so the two stay balanced unless a context's slot is
   cleared without a stop. `deactivate_client_contexts` does that. It runs after
   `stop_client_contexts` when a tool finalizes or detaches, so a context started between the two
   calls keeps `enabled` set. If that context is started again, the start takes a second reference
   and the next stop releases only one, so its agents stay serialized. Counter collection acquires
   only on the disabled-to-enabled transition and is not affected.
2. **Drain result ignored.** `spm::stop_context` releases serialization and returns whether or not
   the queues drained, with no warning beyond the one `Queue::sync()` logs; counter collection logs
   the timeout. A dispatch the drain gave up on is still claimed through the exit hook, but after
   `rocprofiler_stop_context` has returned, and possibly while other dispatches on the agent
   already run unserialized alongside it.
3. **Tests that do not reach the property they name.**
   - No unit test runs the enter hook with an active context. The integration tests under
     `projects/rocprofiler-sdk/tests/spm` and `tests/rocprofv3/spm` do, but only with unrestricted
     contexts: nothing outside the unit tests calls
     `rocprofiler_spm_dispatch_counting_service_set_agents`. Agent filtering in the enter hook and
     in `is_active_on_agent()`, and the `serialize=true` return during a stop, are not checked by
     any test.
   - `set_agents_restricts_collection` checks only `collects_on` and `enabled`, although its
     comment says it verifies serialization scoping and enter-hook filtering. The concurrency tests
     check that the restricted agent ends unserialized, but no test checks that a restricted context
     leaves the other agents unserialized while it runs.
   - `stop_context_in_flight_completion_routes_via_hook_path` uses a no-op record callback, so
     DISPATCH_END delivery is not asserted. It also passes when `pre_kernel_call` returns the empty
     fallback, because that entry is erased too. Its comment says it does not call
     `pre_kernel_call` directly, but it does; only the completion goes through the hook.
   - `stop_context_removes_callbacks` and `stop_context_sync_and_restart` check only `enabled`.
     Neither checks that serialization was released.
   - Nothing tests the return codes of `rocprofiler_spm_dispatch_counting_service_set_agents`.
