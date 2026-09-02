# Kernel replay in rocprof-compute

## System Context

### Terms

| Term | Meaning |
| --- | --- |
| **Bucket** | A group of hardware counters small enough to fit one hardware pass. |
| **Logical dispatch** | One kernel launch as the application issued it, however many times a replay strategy executes it. |
| **Native tool** | A `rocprof-compute` library. |
| **Context** | A rocprofiler-SDK container holding one or more profiling services, started and stopped as a unit. |

### Counter collection today

| Strategy | Execution model | Pros | Cons |
| --- | --- | --- | --- |
| **Application replay** | Launch and run the complete workload once per bucket. | Every bucket, across corresponding replayed dispatch occurrences — nothing is estimated from other occurrences. | Repeats process startup, runtime initialization and host work. |
| **Iteration multiplexing** | Launch once and let the native tool rotate buckets across comparable dispatches. Analysis imputes each dispatch's missing counters. | Avoids repeated launches for workloads with enough dispatches. | Never collects every bucket from one logical dispatch. Undersampled kernels cannot produce a complete metric set. |

- Every configuration consumes the same buckets. Application replay works across all of them;
  iteration multiplexing hard-errors without the native tool.
- Application replay is therefore the only strategy today that satisfies a multi-bucket request
  without borrowing counters from neighbouring dispatches.

### Co-active profiling services

Counter collection never runs alone. Every counter collection invocation must produce kernel-dispatch records,
because those records carry the kernel records.

| Service | Context owner today | Per dispatch? | Consequence under replay |
| --- | --- | --- | --- |
| Counter collection | Native tool, shared with code-object tracing | Yes | The point of the feature: one bucket per pass. |
| Code-object tracing | Native tool, shared with counter collection | No | Unaffected. |
| Kernel dispatch tracing | `rocprofiler-sdk-tool` | Yes | *N* records for one logical dispatch. |
| Marker / ROCTx tracing | `rocprofiler-sdk-tool` | Host-side, spans all passes | Region duration absorbs replay overhead. |
| PC sampling | `rocprofiler-sdk-tool`, separate workload invocation | Agent-wide | Not a pass at all today. |

### Prerequisites

Kernel replay controls services by starting and stopping contexts around individual passes. That
control is only available over contexts the native tool owns, and only over services sharing a
context with nothing that must stay running. Neither condition holds today.

| Prerequisite | Why kernel replay needs it |
| --- | --- |
| The native tool owns kernel dispatch tracing and emits its dispatch records | A tool cannot locally stop a context it does not own. |
| The native tool owns PC sampling, in the same workload invocation as counter collection | A separate invocation has no pass to occupy. |
| Counter collection moves to its own context, split from code-object tracing | A context starts and stops as a unit, and the PC sampling pass must stop counters without stopping code-object tracing. |
| PC sampling honours per-pass local context overrides | PC sampling is agent-wide in the SDK, so it does not consult the override at all. |

These are tracked outside this design. Everything below assumes they are satisfied, and a run that
finds them unsatisfied is rejected rather than degraded.

### SDK replay mechanism

The SDK's kernel replay service provides the mechanism this design relies on.
`rocprofiler-sdk/experimental/kernel_replay.h` is the authoritative API.

| Domain / operation | SDK | Profiling tool |
| --- | --- | --- |
| `KERNEL_REPLAY` `CONFIG` enter | Once per dispatch reaching the replay gate | Supply `pass_count_cb` (fixed-pass path). |
| `pass_count_cb` | Once per dispatch, before any pass | Return the pass count. May consult the dispatch's agent information. |
| `KERNEL_REPLAY` `PASS` enter / exit | Delimits each execution inside a replay window | Select the counter profile for this pass. |
| `KERNEL_REPLAY` `CONFIG` exit | After the last pass | — |

| `pass_count_cb` returns | SDK behavior |
| --- | --- |
| `1` | Ordinary single-execution path. No replay window, no snapshot, no `PASS` callbacks. |
| Greater than `1` | Opens a replay window with that many passes. |

#### Per-pass context control

The SDK supplies `replay_start_context` and `replay_stop_context` on the `PASS` record. The tool
calls them to position one of its own contexts for the remainder of the replay loop.

| Rule | Consequence for this design |
| --- | --- |
| Callable only during `PASS` `PHASE_ENTER`. | Every toggle is placed at a named pass boundary. |
| Sticky across passes. | A context is positioned once, not re-issued every pass. |
| Loop-scoped. Pre-replay state is restored when the loop ends, and global context state is never modified. | The tool never undoes its own toggles. |
| A local enable only undoes a prior local disable; it cannot promote a globally inactive context. | Every context must already be globally started before the first dispatch. |
| Kernel dispatch tracing honours a local stop only. Dispatch counter collection consults the override on every dispatch. | Tracing is suppressed by stopping it and leaving it stopped. |
| Exactly one context may configure a `KERNEL_REPLAY` service. A second returns `ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED`. | Only the native tool configures the service. |

PC sampling is the one exception to the table above: it is agent-wide, so it ignores these overrides
entirely. The design depends on that being fixed — see [Prerequisites](#prerequisites).

```mermaid
sequenceDiagram
    participant App as Application thread
    participant SDK as SDK replay service
    participant Tool as Profiling tool
    participant Agent as GPU agent and queues
    participant Snapshot as Host snapshot manager

    App->>SDK: Submit dispatch
    SDK->>Tool: KERNEL_REPLAY CONFIG enter
    Tool-->>SDK: Register pass_count_cb
    SDK->>Tool: pass_count_cb for this dispatch
    Tool-->>SDK: Return pass count N
    alt N equals 1: opt out
        SDK->>Tool: KERNEL_REPLAY CONFIG exit
        SDK->>Agent: Execute ordinary dispatch
        Agent-->>App: One application-visible completion
    else N is greater than 1
        SDK->>SDK: Acquire per-agent writer lock
        Note over SDK,Agent: Hold the application completion signal
        SDK->>Agent: Drain prior submitting and sibling queue work
        SDK->>Snapshot: Snapshot supported agent state
        loop Repeat N times
            SDK->>Tool: KERNEL_REPLAY PASS enter
            SDK->>Agent: Execute the same dispatch
            Agent-->>SDK: Pass completes and handler drains
            SDK->>Tool: KERNEL_REPLAY PASS exit
            opt Another pass remains
                SDK->>Snapshot: Restore captured agent state
            end
        end
        SDK->>Tool: KERNEL_REPLAY CONFIG exit
        SDK-->>App: Signal one application-visible completion
        SDK->>SDK: Release per-agent writer lock
    end
```

- **Snapshot coverage.** Tracked, agent-owned, coarse-grained device allocations that are neither
  kernarg nor executable memory. The SDK separately discovers and captures module-scope `__device__`
  and `__constant__` storage visible to the agent.
- **Isolation.** A replay window isolates a GPU agent.
- **Restoration.** The snapshot is restored between passes, so every pass starts from identical
  device state.

#### Replay coverage and limitations

| State or feature | Replay guarantee |
| --- | --- |
| Allocated device memory (`hipMalloc`) | Snapshotted and restored between passes. |
| Module-scope `__device__` / `__constant__` state | Discovered and captured by the SDK. |
| Unified, managed, `hipMallocAsync` allocations | Not restored. No equivalence guarantee. |
| Cache state | Not restored. Cache-sensitive counter values may vary between passes. |
| HIP graph launches | Unsupported. |
| Multi-packet or multi-dispatch submissions | Not replayed. Only single-packet, single-dispatch submissions are. |
| Asynchronous SDMA or HSA copies | Not fenced by the replay window. |
| Host memory | Required at least as much as the tracked device memory footprint. |
| Multi-rank dispatches | Unsafe. |

## Problem statement

When a counter request produces *N* buckets and *N* is greater than one:

| Per counter-collection run | Application replay | Kernel replay |
| --- | --- | --- |
| Workload launches | *N* | 1 |
| Process startup, runtime initialization | *N*× | 1× |
| Host-side work | *N*× | 1× |
| Executions of a replay-eligible dispatch | *N* (one per launch) | *N* (one per pass) |
| Executions of every other dispatch | *N* | 1 |
| Added cost | — | Snapshot, restore, queue drain, agent isolation |

Only the selected kernel needs profiling *N* times. For workloads with a large startup cost,
application replay requires a lot of time.

- **Iteration multiplexing does not close this gap.** It avoids the repeated launches, but collects
  counters from different dispatches and fills each dispatch's gaps during analysis. It cannot
  collect every counter from repeated executions of *one* logical dispatch in a single workload run.
- **This is not free.** Kernel replay drops the repeated full-workload launches but adds snapshot,
  restoration, dispatch replay, and isolation costs. It will not be faster for every workload.

The flows below compare counter collection when *N* is greater than one. It does not include other
services outside counter collection.

### Application replay

```mermaid
flowchart TD
    Buckets["N buckets<br/>N greater than 1"]
    AppSelect["Choose bucket i"]
    AppLaunch["Launch profiling and full workload"]
    AppRun["Startup, runtime init,<br/>host work, kernel execution"]
    AppCollect["Collect bucket i"]
    AppDone{"All N buckets?"}
    AppResult["Complete<br/>N full workload launches"]

    Buckets --> AppSelect --> AppLaunch --> AppRun --> AppCollect --> AppDone
    AppDone -- next bucket --> AppSelect
    AppDone -- yes --> AppResult
```

### Kernel replay

```mermaid
flowchart TD
    Buckets["N buckets<br/>N greater than 1"]
    KernelLaunch["Launch profiling and full workload once"]
    KernelSetup["Startup and runtime init once"]
    KernelDispatch{"Next dispatch?"}
    KernelOrdinary["Execute unselected dispatch once"]
    KernelReplay["Execute selected dispatch N times<br/>one bucket per pass"]
    KernelContinue["Continue host-side execution"]
    KernelDone{"Workload complete?"}
    KernelResult["Complete<br/>one full workload launch"]

    Buckets --> KernelLaunch --> KernelSetup --> KernelDispatch
    KernelDispatch -- not selected --> KernelOrdinary --> KernelContinue
    KernelDispatch -- selected --> KernelReplay --> KernelContinue
    KernelContinue --> KernelDone
    KernelDone -- no, next dispatch --> KernelDispatch
    KernelDone -- yes --> KernelResult
```

## Requirements

### Functional requirements

#### Mode selection

| ID | Requirement |
| --- | --- |
| **FR-1** | `rocprof-compute profile` exposes `--replay-mode {application,kernel}`, defaults to `application`. `--iteration-multiplexing` remains independent. |
| **FR-2** | For *N* buckets, kernel mode uses one workload invocation for counter collection and collects one bucket in each of *N* passes, for every replay-eligible dispatch. No user-supplied pass count. A zero-bucket request bypasses kernel replay and retains the existing path. |
| **FR-3** | Kernel replay requires the native tool. The non-native counter-collection backend and `--no-native-tool` are unsupported. A declined native tool, or a ROCm version that does not support one, is a hard error naming the unmet condition, before any workload runs. |

#### Passes and buckets

| ID | Requirement |
| --- | --- |
| **FR-4** | Bucket membership matches application replay. `_ACCUM` pairing, TCC grouping, and same-bucket priority are unchanged, and every bucket still fits one hardware pass. |
| **FR-5** | Pass count equals the bucket count, or the bucket count plus one when `--pc-sampling` is selected. |
| **FR-6** | The PC sampling pass runs with counter collection disabled, and alters neither bucket membership nor the ordering of the counter passes preceding it. It is realized by locally stopping the counter context and locally starting the PC sampling context at that pass. |

#### Output and identity

| ID | Requirement |
| --- | --- |
| **FR-7** | One kernel-replay invocation produces one consolidated counter result using the existing naming and discovery convention. The contract the analysis path consumes is unchanged; kernel replay adds one native artifact and one ingestion step upstream of it. |
| **FR-8** | All passes of one logical dispatch resolve to one `Dispatch_ID` group holding the complete counter set. |
| **FR-9** | Start and end timestamps follow the same cross-pass normalization semantics used for application-replay results. |
| **FR-10** | Exactly one kernel dispatch record reaches the result per logical dispatch, from pass 0. |
| **FR-11** | Marker region durations are corrected by subtracting the replay overhead of the dispatches the region encloses. |

#### Composition and filtering

| ID | Requirement |
| --- | --- |
| **FR-12** | Kernel replay with `--iteration-multiplexing` or `--attach-pid` is a hard error. `--roof-only`, `--set`, and `--block` interoperate unchanged. |
| **FR-13** | A dispatch whose kernel is excluded is not replayed. |
| **FR-14** | `--dispatch` retains its per-kernel dispatch filtering. A replay-ineligible kernel dispatch is not replayed. |
| **FR-15** | Kernel replay stays available for multi-rank workloads and emits a kernel-replay-specific diagnostic describing the risk. |
| **FR-16** | Only the native tool configures a `KERNEL_REPLAY` service. `rocprofiler-sdk-tool` must not, because only one context may configure one. |

#### Failure

| ID | Requirement |
| --- | --- |
| **FR-17** | An SDK below the supported version floor is a hard error stating the required version. |
| **FR-18** | If the SDK declines a device-memory snapshot, abandon the profile without retry and recommend application replay in the diagnostic. |
| **FR-19** | If the upstream replay mechanism aborts, report the failed run without attempting recovery. |
| **FR-20** | If counters were requested but an agent has no usable counter profiles, do not silently degrade the dispatch to one pass. |
| **FR-21** | A second `KERNEL_REPLAY` service configuration is a hard error naming `ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED`. |
| **FR-22** | An unsatisfied prerequisite is a hard error naming that prerequisite, before any workload runs. Kernel replay never proceeds with multiplied dispatch records, a counter context it cannot stop independently, or a missing PC sampling pass. |

### Non-functional requirements

| ID | Requirement |
| --- | --- |
| **NFR-1** | For deterministic workload and counters, each kernel-replay counter value matches the corresponding application-replay counter value for the same logical dispatch. |
| **NFR-2** | Fail closed whenever a complete counter set cannot be delivered. Never collect partial counter data silently. |
| **NFR-3** | Workflows that do not select kernel replay keep their current collection behavior, and kernel-replay output preserves the existing analysis input contract. |

## Design

### Context inventory and lifecycle

This section presumes the [prerequisites](#prerequisites) are satisfied. Given them, the native tool
owns five contexts. Splitting them is what makes per-pass control expressible: each one is
positioned differently across the pass loop.

| Context | Carries | Created | Globally started | Local toggle inside the loop | Globally stopped |
| --- | --- | --- | --- | --- | --- |
| Code object | Code-object tracing | Tool initialization | HSA runtime load | None | Tool finalization |
| Counter | Dispatch counter collection | Tool initialization | HSA runtime load | Stopped at pass *N* enter, the PC sampling pass only | Tool finalization |
| Kernel trace | Kernel dispatch tracing | Tool initialization | HSA runtime load | Stopped at pass 1 enter, so only pass 0 is traced | Tool finalization |
| PC sampling | PC sampling | Tool initialization | HSA runtime load | Stopped at pass 0 enter, started at pass *N* enter | Tool finalization |
| Replay | The `KERNEL_REPLAY` `CONFIG` and `PASS` service | Tool initialization | HSA runtime load | Not applicable | Tool finalization |

- **Created at tool initialization**, alongside the code-object and dispatch-counting services the
  native tool already configures there.
- **Globally started when the HSA runtime loads**, not at initialization. Starting a context spawns
  HSA worker threads, which deadlock on `fork()` in non-GPU preloaded shells. The existing deferral
  applies to every context, not only the current one.
- **Globally stopped at tool finalization**, before output generation, so every buffered record is
  complete when the artifacts are written.
- **Local toggles are issued once**, at the named pass boundary, and never re-issued or undone. The
  SDK restores each context's pre-replay position when the loop ends.

```mermaid
sequenceDiagram
    participant SDK as SDK replay service
    participant Tool as Native tool

    SDK->>Tool: CONFIG enter
    Tool-->>SDK: pass_count_cb returns N+1
    SDK->>Tool: PASS 0 enter
    Tool->>SDK: stop PC sampling context
    Note over Tool,SDK: Counters and kernel tracing both collect
    SDK->>Tool: PASS 1 enter
    Tool->>SDK: stop kernel trace context
    Note over Tool,SDK: Passes 1..N-1 collect counters only
    SDK->>Tool: PASS N enter
    Tool->>SDK: stop counter context
    Tool->>SDK: start PC sampling context
    Note over Tool,SDK: Counter-disabled PC sampling pass
    SDK->>Tool: CONFIG exit
    SDK->>SDK: Restore every context to its pre-replay position
```

Without PC sampling the loop is the same minus the pass-0 and pass-*N* toggles: only the kernel
trace context is stopped, at pass 1.

### Counter groups and the native-tool boundary

- **The counter-grouping logic stays the same for both modes.**
  `_ACCUM` pairing, TCC grouping, and same-bucket priority policies produce the same *N* buckets for
  both modes.
- **All *N* groups go to a single tool invocation.** The native tool derives the pass
  count from the per-agent counter profiles, adding one when PC sampling is selected.
- **Consolidated results keep the existing naming convention.**
- **Results have unified structure across modes.**
- **No separate pass-count option or environment value.**

```mermaid
flowchart TD
    subgraph ComputeSetup["rocprof-compute · setup"]
        direction TB
        Buckets["N counter buckets"]
    end

    subgraph NativeTool["Native tool"]
        direction TB
        Profiles["Per-agent profile vector<br/>and pass_count_cb"]
        Collect["Select group i on pass i<br/>and collect counters"]
        Write["Write per-pass PMC results"]
        Reject["Reject invalid profile<br/>never treat as filter opt-out"]
        Trace["Kernel dispatch tracing<br/>on its own context"]
        Collect --> Write
    end

    subgraph SDK["rocprofiler-sdk"]
        direction TB
        Replay["Kernel replay service<br/>N passes per logical dispatch"]
    end

    subgraph ComputeResults["rocprof-compute · results"]
        direction TB
        Normalize["Normalize cross-pass identity<br/>and timestamps"]
        Output["Consolidated PMC rows"]
        Analysis["Existing join and analysis path"]
        Normalize --> Output --> Analysis
    end

    Buckets -- "ROCPROF_COUNTERS groups<br/>and native replay setup" --> Profiles
    Profiles -. "register pass_count_cb" .-> Replay
    Replay -- "invoke pass_count_cb" --> Profiles
    Profiles -- "admitted: vector size<br/>filtered: 1" --> Replay
    Profiles -- "counter request with<br/>missing or empty vector" --> Reject
    Replay -- "PASS enter / exit" --> Collect
    Replay -- "replayed dispatches" --> Trace
    Collect -. "stop the trace context<br/>at pass 1 enter" .-> Trace
    Write -- "per-pass PMC rows" --> Normalize
    Trace -- "one pass-0<br/>dispatch record" --> Normalize
```

### The pass-count decision

Kernel and dispatch filters establish whether the dispatch is profiled before counter-bucket
availability is considered. An admitted zero-bucket request bypasses kernel replay and
follows the existing path. For an admitted kernel, `pass_count_cb` determines the
pass count exactly once before any replay pass.

```mermaid
flowchart TD
    Kernel{"Kernel admitted by<br/>the kernel filter?"}
    Range{"Dispatch admitted by<br/>the dispatch filter?"}
    Filtered["Do not profile<br/>the dispatched kernel"]
    Request{"Counter request has<br/>at least one bucket?"}
    Zero["Bypass kernel replay<br/>use existing non-counter path"]
    Gate{"Dispatch reaches<br/>the replay gate?"}
    Ordinary["Execute once outside replay<br/>retain ordinary dispatch handling"]
    Start["pass_count_cb runs once"]
    Vector{"Agent has the expected<br/>non-empty profile vector?"}
    Fatal["Fatal: agent/profile mismatch<br/>reject the whole profile"]
    Size{"Profile-vector size?"}
    Single["Return 1 as admitted<br/>select the sole profile"]
    PC{"PC sampling selected?"}
    N["Return N<br/>one counter bucket per pass"]
    NPlus["Return N+1<br/>N counter passes, then one<br/>counter-disabled PC sampling pass"]

    Kernel -- no --> Filtered
    Kernel -- yes --> Range
    Range -- no --> Filtered
    Range -- yes --> Request
    Request -- no --> Zero
    Request -- yes --> Gate
    Gate -- no --> Ordinary
    Gate -- yes --> Start --> Vector
    Vector -- no --> Fatal
    Vector -- yes --> PC
    PC -- yes --> NPlus
    PC -- no --> Size
    Size -- one --> Single
    Size -- more than one --> N
```

| Pass count returned | When | What the execution selects |
| --- | --- | --- |
| `1` — filtered | A confirmed filter miss — the kernel or dispatch is excluded. | Not profiled. |
| `1` — admitted | The dispatch is admitted and its agent's profile vector contains exactly one bucket, no PC sampling. | SDK selects the sole profile and produces a complete one-bucket result. |
| *N*, where *N* > 1 | Admitted, no PC sampling. *N* is the size of the profile vector for this dispatch's agent. | Pass *i* selects entry *i* of that vector. |
| *N*+1 | Admitted, PC sampling selected. *N* is the non-zero size of the profile vector. | Passes 0 through *N*−1 map one-to-one onto the vector. Pass *N* selects no counter profile and runs counter-disabled. |

### Output and dispatch ID

A single dispatch produces one consolidated counter result. Before it is
written, every pass for a logical dispatch must be collapsed.

| Step | What happens |
| --- | --- |
| 1. Correlate | Group the pass rows by logical-dispatch ID. |
| 2. Normalize timestamps | Give the group one canonical start/end pair using pass 0's logical duration. |
| 3. Hand off | The existing identity and pivot contracts see a single set per dispatch. |

Pass identity survives transiently for normalization and diagnostics, but it does not reach
the counter result output. Non-replay identity behavior is untouched.

### Co-active service composition

| Service or output | Under kernel replay | Mechanism |
| --- | --- | --- |
| Kernel dispatch tracing | One dispatch record per logical dispatch, from pass 0 | The native tool owns the trace context and stops it at pass 1 enter. It stays stopped for the rest of the loop. |
| Top Stats and dispatch information | One logical counter dispatch and its pass-0 duration | These outputs consume the consolidated dispatch data. |
| Code-object tracing | Unchanged | Load-time only. Replay never multiplies it. |
| Marker / ROCTx | Emitted once, spans all passes; durations corrected | Replay overhead is subtracted from any region enclosing a replayed dispatch. |
| PC sampling | One extra pass, appended after the counter passes | Runs counter-disabled on its own context. |
| Roofline | Unchanged | |

#### Kernel dispatch records

Native-tool ownership of kernel dispatch tracing is a [prerequisite](#prerequisites), not something
this design delivers. It has one consequence beyond the pass loop: the native tool produces the
dispatch records too.

- `ROCPROF_KERNEL_TRACE` is set to `0` for `rocprofiler-sdk-tool` under kernel replay, so the two
  libraries cannot both emit dispatch records for the same dispatch.
- The native tool's dispatch records are ingested alongside its counter records before the
  consolidated result is produced, following the path its counter records already take. The record
  format is a low-level design decision and is not specified here.
- Exactly one dispatch record per logical dispatch reaches the result. This is not cosmetic: the
  dispatch record carries the timestamps and launch metadata that counter rows are joined against,
  so *N* records would multiply the joined result and defeat dispatch-ID collapsing.
- **Risk for the low-level design.** The result database is created by `rocprofiler-sdk-tool`.
  Confirm it is still created once `ROCPROF_KERNEL_TRACE` is `0`.

#### Marker region durations

A marker region is emitted once by the host but encloses every pass, so its duration subsumes the
replay overhead. That overhead is subtracted rather than tolerated.

- The native tool derives each logical dispatch's replay overhead from the `KERNEL_REPLAY` `CONFIG`
  and `PASS` callbacks it already receives: the replay window's wall time, less the single execution
  the application would have performed without replay. This needs no per-pass kernel tracing, so
  stopping the trace context at pass 1 does not affect it.
- Post-processing subtracts that overhead from any marker region enclosing the dispatch, so reported
  region durations approximate an unreplayed run.
- **Residual limitation.** Overhead that cannot be attributed to a specific enclosed dispatch is not
  subtracted.

### Mode and option compatibility

`--replay-mode` accepts `application` and `kernel`. Application replay stays the default, CLI help
flags kernel replay as experimental.

| Option or condition | With kernel replay |
| --- | --- |
| `--iteration-multiplexing` | Rejected |
| `--attach-pid` | Rejected. Live attach cannot open a replay window over an already-running process. |
| `--no-native-tool`, the non-native counter-collection backend | Rejected |
| `--pc-sampling` | Accepted, *N*+1 passes |
| `--roof-only`, `--set`, `--block` | Accepted, unchanged |
| Multi-rank | Accepted, with a kernel-replay-specific diagnostic |

### Failure behavior

Every failure below rejects partial replay data. None falls back to application replay, and none
quietly turns a multi-bucket request into a single pass.

| Condition | Required behavior |
| --- | --- |
| Native tool not selected while kernel replay is selected | Hard error from the argument combination alone, before discovery. |
| Native tool unavailable: unsupported ROCm version or unresolvable library | Hard error after discovery, naming which of the two failed. |
| SDK below the supported version floor | Hard error stating the required version. |
| Iteration multiplexing or live attach selected with kernel replay | Hard error before profiling starts. |
| Missing or unexpectedly empty per-agent profile vector when counters were requested | Hard error describing the profile mismatch. |
| SDK declines the device-memory snapshot | Abandon the entire profile without retry, reject incomplete output, and recommend application replay. |
| Upstream drain timeout or process abort | Abort the failed run. |
| A second `KERNEL_REPLAY` service configuration | Hard error naming `ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED`. |
| An unsatisfied prerequisite: a service the native tool does not own, counter collection still sharing the code-object context, or PC sampling ignoring per-pass overrides | Hard error naming that prerequisite, before profiling starts. |

One wrinkle: a declined snapshot currently leaves a successful process status behind, so the
required outcome cannot lean on subprocess failure alone. Classifying an upstream warning string is
the signal available today. A structured detection mechanism remains an open question.

## Implementation phases

| Phase | Delivers | Observable after this phase |
| --- | --- | --- |
| **1. Mode selection and validation** | The experimental `--replay-mode {application,kernel}` surface and rejections with their error diagnostics. Replay execution stays disabled. | Mode selection and correct rejection. Existing application-replay output does not change. |
| **2. Native-tool kernel replay** | Coalesced counter groups delivered to the SDK invocation; the multi-group admission check extended to admit kernel replay; `pass_count_cb` implemented from the per-agent profile-vector size; both filters applied; and diagnosed failure on missing or unexpectedly empty counter profiles. | Replayed dispatches collect every bucket in one run, including valid one-bucket requests, while zero-bucket requests retain the existing bypass. Application replay keeps working throughout. |
| **3. Consolidated output and dispatch ID** | Per-pass context positioning, one consolidated counter result using the existing naming convention, cross-pass timestamp and dispatch ID normalization, and marker duration correction. | Analysis consumes kernel-replay profiling output; Top Stats, dispatch information and marker regions report non-multiplied values. |

Phases 2 and 3 assume the [prerequisites](#prerequisites) are already satisfied. Phase 1 does not:
its rejections include the prerequisite check, so an unsatisfied prerequisite fails before any
replay code runs.

## Validation, security and debuggability

### Validation

| # | Check | Pass criterion |
| --- | --- | --- |
| 1 | **Counter accuracy** | For a deterministic workload requesting more than one bucket; for each logical dispatch, every kernel-replay counter value matches its corresponding application-replay counter value. Evaluate cache-sensitive counters separately, as a documented limitation. |
| 2 | **Completeness and identity** | For each admitted dispatch, including an admitted one-bucket dispatch, the observed counter-pass count equals the application-replay count, every counter appears exactly once, and all passes collapse to one `Dispatch_ID`. Incomplete results fail before analysis. |
| 3 | **Filtering** | Kernels excluded by the kernel or dispatch filter are not profiled and no errors are thrown. The same `--dispatch` range selects the same dispatches in both replay modes. |
| 4 | **Application-replay comparison** | Profile the same deterministic workload and multi-bucket counter request in both modes. Compare corresponding buckets for each logical dispatch; bucket membership, counter values, and final analysis results agree, and existing application replay is unchanged. |
| 5 | **Kernel tracing** | Exactly one kernel dispatch record per logical dispatch, from pass 0, with the counter rows joining to it one-to-one and one resulting `Dispatch_ID`. That record supplies the kernel duration in analysis. |
| 6 | **PC sampling** | Each admitted dispatch replays *N*+1 times. Every bucket appears exactly once across passes 0 through *N*−1, and pass *N* produces PC sampling output and no counter rows. |
| 7 | **Compatibility** | Every accepted option behaves as expected — PC sampling, roofline selection, a kernel-replay-specific multi-rank diagnostic that names the collective kernel risk, and default-off — and every rejected combination is rejected: iteration multiplexing, live attach. |
| 8 | **Configuration rejection** | Each unmet condition on its own — no native tool, unsupported ROCm version, unresolvable library, each unsatisfied prerequisite — fails before profiling starts, with a diagnostic naming that specific condition. |
| 9 | **Failure paths** | An unsupported SDK, a missing or unexpectedly empty profile vector when counters were requested, a declined snapshot, and an upstream abort each fail, and each proves no one-pass fallback happened. A zero-bucket request follows the existing bypass and is not misclassified as a missing-profile failure. |
| 10 | **Diagnostics** | Every run identifies the replay mode, and every failure names its specific unmet condition. |
| 11 | **Context lifecycle** | Every native context is created at tool initialization, started when the HSA runtime loads, and stopped at tool finalization. No local toggle is issued outside `PASS` `PHASE_ENTER`, and the kernel trace context is stopped exactly once. |
| 12 | **Marker correction** | A marker region enclosing a replayed dispatch reports a duration within tolerance of the same region under application replay. |

### Security

- Kernel replay adds no network interface and no new privilege boundary.
- Its one security-sensitive operation is the SDK-managed host snapshot of application device memory.
  That snapshot holds application data and should stay inside the profiled process's trust
  boundary, and `rocprof-compute` must never persist its contents in result artifacts or print them
  in diagnostics.
- Snapshot allocation and restore failures are availability failures, and follow the same fail-closed
  rule as every other incomplete replay condition.

### Debuggability

A diagnostic must name the specific unmet condition. It must identify:

- Selected replay mode and backend
- Which contexts the native tool owns, and which prerequisite is unsatisfied when a run is rejected
- Multi-rank detection under kernel replay
- SDK capability and agent
- Counter-to-pass mapping
- Which failure occurred: option conflict, unavailable native collection, missing profile, snapshot
  decline, or upstream abort
- That partial results are unusable
- A snapshot decline, with a recommendation to use application replay

## Open questions

| # | Question | Why it matters |
| --- | --- | --- |
| 1 | Should kernel replay exclude collective kernels? | This will make multi-rank profiling with kernel replay possible, but the metrics won't reflect the statistics for all kernel invocations. |
| 2 | Are cache-related metrics trustworthy at all under kernel replay? | Nothing restores cache state between passes, so cache-related counters do not represent the true cache behavior. |
| 3 | How should `rocprof-compute` detect a declined snapshot? | Today the only signal is classifying an upstream warning string, and the process status can still report success. |
