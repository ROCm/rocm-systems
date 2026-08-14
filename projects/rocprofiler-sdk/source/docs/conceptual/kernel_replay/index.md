# Kernel Replay (Experimental)

Hardware has a limited number of counter registers per block. When a tool requests more counters than
can be collected in a single pass, it has to collect them across several passes. The traditional
answer is *application replay*: re-run the whole application once per counter group. **Kernel replay**
does it at the granularity of a single dispatch instead — re-execute one kernel `N` times within one
application run, restoring device memory between executions so every pass sees identical inputs.

| Approach | Scope | Memory handling | Cost |
|---|---|---|---|
| Application replay (multiple `--pmc` groups) | whole application, re-run per group | none needed; each run is a fresh process | `O(N x` app runtime`)` |
| **Kernel replay** | one dispatch, re-executed in place | device memory snapshot and restore between passes | `O(N x` kernel time `+ N x` snap/restore`)` |
| Counter group rotation | amortized across dispatches | none; different dispatches sample different groups | `O(1 x` app runtime`)` |

Kernel replay is **experimental**. The public header lives under
`rocprofiler-sdk/experimental/`, the `rocprofv3` flag is `--kernel-replay-beta-enabled`, and both are
expected to change before a stable release. Several waits inside the replay window abort the process
on expiry rather than proceeding on questionable state — a deliberate choice for a beta feature,
described in [Concurrency and isolation](kernel_replay_concurrency_and_isolation.md).

## How it fits together

Replay is driven entirely from the HSA queue `WriteInterceptor`. There is no replay worker thread: a
replayed dispatch expands, synchronously on the submitting thread, into a drain, a device memory
snapshot, and a loop of passes with a restore between them.

```text
experimental/kernel_replay.h            public payload struct (callback tracing domain)
        |
        +-- callback_tracing.cpp        subscription; switches on the allocation tracker
        |
        +-- kernel_replay/
        |     replay_callbacks.cpp      CONFIG + PASS callbacks, pass-count/continue decisions
        |     local_context.cpp         per-pass localized context control (thread-local overrides)
        |     memory_tracker.cpp        HSA allocate/free hooks, per-agent allocation inventory
        |     memory_snapshot.cpp       snap()/restore(), module-scope variable capture
        |     utils.cpp                 trackable-allocation classifier
        |
        +-- hsa/queue.cpp               the replay window: per-agent lock, drains, pass loop
```

Because replay is a callback tracing service rather than a counter-collection mode, it is not tied to
hardware counters. A tool decides what each pass is for and, through localized context control, which
of its services are active on which pass.

## Documentation in this section

- **[Callback API and tool configuration](kernel_replay_callback_api.md)** — the public API surface:
  the `ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY` domain, its two operations, the payload struct,
  pass-count semantics, localized context control, and how a tool (including `rocprofv3`) configures
  replay.
- **[Concurrency and isolation](kernel_replay_concurrency_and_isolation.md)** — how the
  snapshot-to-restore window is isolated: the per-agent reader/writer lock, the agent-wide drain,
  agent-scoped snapshots, the async completion handler drain, the bounded-wait and abort convention,
  and what is deliberately left un-isolated.
- **[Memory snapshot and restore](kernel_replay_memory_snapshot.md)** — what is captured and what is
  excluded, module-scope `__device__` variable capture, the decline-rather-than-corrupt failure
  policy, HIP graph behavior, and the known limitations.
- **[Callback tracing API design](kernel_replay_callback_api_design.md)** — the design rationale and
  history behind the callback API: why replay was decoupled from counter collection, the shape
  decisions taken, and the recorded review findings. Read the pages above for current behavior; this
  one records how the design got there and what remains open.
