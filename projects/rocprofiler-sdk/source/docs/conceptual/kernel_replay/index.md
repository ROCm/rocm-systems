(kernel-replay-conceptual)=
# Kernel Replay (Experimental)

Hardware has a limited number of counter registers per block. When a tool requests more counters than
can be collected in a single pass, it has to collect them across several passes. The traditional
answer is *application replay*: re-run the whole application once per counter group. **Kernel replay**
does it at the granularity of a single dispatch instead — re-execute one kernel `N` times within one
application run, restoring device memory between executions so every pass sees identical inputs.

| Approach | Scope | Memory handling | Cost |
|---|---|---|---|
| Application replay (multiple `--pmc` groups) | whole application, re-run per group | none needed; each run is a fresh process | `O(N ×` app runtime`)` |
| **Kernel replay** | one dispatch, re-executed in place | device memory snapshot and restore between passes | `O(N ×` kernel time `+ N ×` snap/restore`)` |
| Counter group rotation | amortized across dispatches | none; different dispatches sample different groups | `O(1 ×` app runtime`)` |

The three approaches differ in what they must reproduce, and that is what decides where each one is
sound. Application replay reproduces nothing: each run is a fresh process, so correctness is free and
the cost is the whole application, `N` times. Counter group rotation also reproduces nothing, but it
answers a different question — it attributes different counter groups to *different* dispatches, so
it cannot correlate two groups on the same execution. Kernel replay is the only one that gives every
counter group the same execution of the same kernel, and it pays for that by having to reconstruct
that execution's inputs. Everything difficult about kernel replay follows from that one obligation.

Kernel replay is **experimental**. The public header lives under `rocprofiler-sdk/experimental/`.
Both the API and any later command-line flag are expected to change before a stable release. Where a
dispatch cannot be replayed soundly — memory the snapshot cannot cover, a footprint that will not
fit, work that will not drain — the window declines and runs the dispatch once, reporting why; see
[Concurrency and isolation](kernel_replay_concurrency_and_isolation.md).

This is the kernel replay **callback tracing API**. An earlier experimental counting-service
prototype is not the current contract: there is no dedicated configure function, no pass-count
environment variable, and no dirty-page hashing. Snapshot and restore are a full in-memory copy.
Host-side and/or device-side hashing of dirty regions is expected in a future version.

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
        |     replay_diagnostics.cpp    admission control, decline reasons, per-dispatch reporting
        |     queue_hooks.cpp           the replay window: per-agent lock, drains, pass loop
        |     utils.cpp                 trackable-allocation classifier
        |
        +-- hsa/queue.cpp               dispatch eligibility gate; calls run_replay_window()
```

Because replay is a callback tracing service rather than a counter-collection mode, it is not tied to
hardware counters. A tool decides what each pass is for and, through localized context control, which
of its services are active on which pass. A custom tool can collect every counter group in one run,
or use the same domain for timing, PC sampling, or thread trace. Command-line `rocprofv3` wiring is
the stacked tool integration PR.

## Documentation in this section

**Tool authors:** start with {ref}`using-kernel-replay`.

**SDK / tool developers:**

- **[Callback API and tool configuration](kernel_replay_callback_api.md)** — the public API surface:
  the `ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY` domain, its two operations, the payload struct,
  pass-count semantics, localized context control, and how a tool configures replay.
- **[Concurrency and isolation](kernel_replay_concurrency_and_isolation.md)** — how the
  snapshot-to-restore window is isolated: the per-agent reader/writer lock, the agent-wide drain,
  agent-scoped snapshots, the async completion handler drain, the bounded-wait and abort convention,
  and what is deliberately left un-isolated.
- **[Memory snapshot and restore](kernel_replay_memory_snapshot.md)** — what is captured and what is
  excluded, the full in-memory copy (no dirty-page hashing), module-scope `__device__` variable
  capture, the decline-rather-than-corrupt failure policy, HIP graph behavior, and planned hashing.
- **[Callback tracing API design](kernel_replay_callback_api_design.md)** — the design rationale and
  history behind the callback API. Read the pages above for current behavior; this one records how
  the design got there and what remains open.
