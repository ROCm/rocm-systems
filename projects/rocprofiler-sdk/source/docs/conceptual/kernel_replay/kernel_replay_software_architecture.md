# Kernel Replay — Software Architecture

Companion to `kernel_replay_callback_api_design.md`. That document specifies the public API and its
semantics; this one records where the implementation lives and what order things happen in.

Paths are relative to `projects/rocprofiler-sdk/source/`, except in the test table, which is given in
full because the tests live outside `source/`. Elements are named by symbol rather than cited by line
number, since line numbers rot faster than the code they point at.

## 1. Component architecture

```mermaid
flowchart TB
  T1["<b>Tool</b><br/>rocprofv3 --kernel-replay-beta-enabled<br/>or any SDK client"]
  A4["<b>rocprofiler_configure_callback_tracing_service</b><br/>domain = KERNEL_REPLAY<br/>side effects: enable memory tracker,<br/>set process-global replay flag"]
  H7["<b>enable_queue_intercept</b><br/>KERNEL_REPLAY alone forces<br/>AQL queue interception"]
  H1["<b>WriteInterceptor</b> — app launch thread"]
  H2{"<b>replay gate</b><br/>replay contexts active<br/>AND pkt_count == 1<br/>AND one dispatch packet?"}
  NORM["normal single-pass path<br/><i>replay is skipped</i>"]
  K1A["<b>CONFIG PHASE_ENTER</b><br/>tool installs pass_count_cb<br/>SDK calls it to get N"]
  H3["<b>acquire agent_replay_mutex</b><br/>one std::mutex per rocp agent"]
  D1["<b>drain</b> this queue via barrier + CPU wait<br/>then Queue::sync every queue on the agent"]
  K5A["<b>memory_snapshot::snap(agent)</b><br/>tracked coarse-grained VRAM owned by<br/>this agent copied to host vectors"]
  LOOP["<b>pass loop</b>, pass = 0 .. N-1"]
  K1B["<b>PASS PHASE_ENTER</b><br/>toggles armed; tool may call<br/>replay_local_start/stop_context_cb"]
  H5["<b>process_packet_batch</b><br/>same code as the single-pass path<br/>shared dispatch_id, app signal suppressed,<br/>pass_done barrier appended"]
  R2["<b>AQL queue</b> -> GPU executes the pass"]
  W["<b>CPU wait on pass_done</b><br/>GPU barrier only"]
  K1C["<b>PASS PHASE_EXIT</b> then should_continue_replay"]
  K5B["<b>memory_snapshot::restore</b><br/>every captured block written back"]
  K1D["<b>CONFIG PHASE_EXIT</b><br/>then fire the application completion signal once"]
  H6["<b>AsyncSignalHandler</b> — HSA signal thread<br/>counter/dispatch records, serializer release,<br/>signal + packet cleanup"]
  K2["<b>local_context</b><br/>thread-local override map,<br/>lives for the whole loop, sticky"]
  SVC["<b>Per-dispatch consumers of the override</b><br/>kernel dispatch tracing, dispatch counter<br/>collection, dispatch thread trace"]
  SVCX["<b>Not wired to the override</b><br/>PC sampling, SPM,<br/>device counter collection"]
  K3["<b>memory_tracker</b><br/>HSA alloc/free wrappers keep an inventory<br/>of ptr -> size + owning agent"]
  K4["<b>query_alloc</b> filter<br/>coarse-grained AND NOT kernarg"]
  RT["<b>HSA API table</b><br/>wrappers installed unconditionally at init"]

  T1 --> A4
  A4 --> H7
  A4 --> K3
  H7 --> H1
  H1 --> H2
  H2 -- no --> NORM
  H2 -- yes --> K1A
  K1A -- "N <= 1 or no cb" --> NORM
  K1A -- "N > 1 or indefinite" --> H3
  H3 --> D1 --> K5A --> LOOP
  LOOP --> K1B --> H5 --> R2 --> W --> K1C
  K1C -- continue --> K5B --> LOOP
  K1C -- stop --> K1D
  R2 -.completion signals.-> H6
  K1B --> K2
  K2 -. queried at dispatch time .-> SVC
  H5 --> SVC
  SVC -.-> SVCX
  K5A <--> K3
  K5B <--> K3
  K3 --> K4
  RT --> K3

  classDef newcode fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000
  classDef touched fill:#fff8e1,stroke:#f9a825,stroke-width:2px,color:#000
  classDef gap fill:#eceff1,stroke:#546e7a,stroke-width:2px,stroke-dasharray:5 3,color:#000
  classDef api fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000
  classDef tool fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#000
  class K1A,K1B,K1C,K1D,K2,K3,K4,K5A,K5B newcode
  class H1,H2,H3,H5,H6,H7,D1,LOOP,W,R2,SVC,RT,NORM touched
  class SVCX gap
  class A4 api
  class T1 tool
```

Green is the new subsystem, yellow is existing code modified to accommodate it, grey dashed is a
service that does not consult the per-pass override.

## 2. Component inventory

### 2.1 Public API

| Element | Location |
|---|---|
| Domain `ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY` | `include/rocprofiler-sdk/fwd.h` |
| `rocprofiler_kernel_replay_operation_t` (`NONE`, `CONFIG`, `PASS`) | `include/rocprofiler-sdk/fwd.h` |
| `rocprofiler_callback_tracing_kernel_replay_data_t` | `include/rocprofiler-sdk/experimental/kernel_replay.h` |
| `pass_count_cb`, `replay_continue_cb` | `experimental/kernel_replay.h` |
| `current_pass`, `total_passes` | `experimental/kernel_replay.h` |
| `replay_local_start_context_cb`, `replay_local_stop_context_cb` | `experimental/kernel_replay.h` |
| Enum label and `static_assert` on the domain count | `include/rocprofiler-sdk/cxx/enum_string.hpp` |
| Service configuration and its side effects | `lib/rocprofiler-sdk/callback_tracing.cpp` |
| Operation name / id tables | `kernel_replay/kernel_replay.cpp` |

### 2.2 `kernel_replay/` subsystem

| Element | Location |
|---|---|
| Process-global service gate, `has_active_replay_contexts` | `kernel_replay/replay_callbacks.cpp` |
| `make_dispatch_info` | `kernel_replay/replay_callbacks.cpp` |
| `execute_config_phase_enter`, `execute_config_phase_exit` | `kernel_replay/replay_callbacks.cpp` |
| `execute_pass_phase_enter` (arms toggles), `execute_pass_phase_exit` | `kernel_replay/replay_callbacks.cpp` |
| `should_continue_replay` | `kernel_replay/replay_callbacks.cpp` |
| `replay_plan_t`, `pass_context_state_t` | `kernel_replay/replay_callbacks.hpp` |
| Thread-local override map and arm flag | `kernel_replay/local_context.cpp` |
| `record_override` (rejects unarmed calls) | `kernel_replay/local_context.cpp` |
| `scoped_local_context_control` | `kernel_replay/local_context.{cpp,hpp}` |
| `replay_local_start_context` / `_stop_context` | `kernel_replay/local_context.cpp` |
| `local_context_has_overrides`, `local_context_override` | `kernel_replay/local_context.cpp` |
| Tracking flag, alloc/free wrappers (pool + region) | `kernel_replay/memory_tracker.cpp` |
| `record_alloc` / `record_free` (fini-guarded) | `kernel_replay/memory_tracker.cpp` |
| `snap_inventory(agent)` — agent-filtered | `kernel_replay/memory_tracker.cpp` |
| `update_table` (core + amd_ext) | `kernel_replay/memory_tracker.cpp` |
| `query_alloc` pool filter | `kernel_replay/utils.cpp` |
| `snap(agent)`, `restore(snapshot)` | `kernel_replay/memory_snapshot.cpp` |
| `mem_block_t`, `device_snapshot_t` | `kernel_replay/memory_snapshot.hpp` |
| HSA table install site | `lib/rocprofiler-sdk/registration.cpp` |

### 2.3 Queue interception

| Element | Location |
|---|---|
| `replay_pass_state_t`, `agent_replay_mutex` | `hsa/queue.cpp` |
| `has_active_replay_contexts` call and interceptor gate | `hsa/queue.cpp` |
| Local-override pruning of dispatch-tracing contexts | `hsa/queue.cpp` |
| Shared `dispatch_id` across passes | `hsa/queue.cpp` |
| Application completion signal suppressed per pass | `hsa/queue.cpp` |
| `pass_done` barrier appended | `hsa/queue.cpp` |
| Replay gate and replay window (lock, drain, snap, loop, restore, signal) | `hsa/queue.cpp` |
| `AsyncSignalHandler` | `hsa/queue.cpp` |
| `Queue::sync` (five-second hint, warn-only on timeout) | `hsa/queue.cpp` |
| `enable_queue_intercept` and the interposition predicate | `hsa/queue_controller.cpp` |

### 2.4 Services that honor the per-pass override

| Service | Location | Mechanism |
|---|---|---|
| Kernel dispatch tracing | `hsa/queue.cpp` | Removes disabled contexts from the per-packet callback and buffered context vectors |
| Dispatch counter collection | `counters/dispatch_handlers.cpp` | Override folded into the computed `is_enabled` |
| Dispatch thread trace | `thread_trace/core.cpp` | Returns `{nullptr, bSerialize}` — skips the trace, keeps serialization |
| PC sampling | not wired | — |
| SPM (`dispatch_spm`) | not wired | — |
| Device counter collection | not wired | — |

### 2.5 Tool integration (`rocprofv3`)

| Element | Location |
|---|---|
| `--kernel-replay-beta-enabled` CLI flag | `bin/rocprofv3.py` |
| `--pmc` requirement and environment plumbing | `bin/rocprofv3.py` |
| Counter-group merge (single application run, no per-group relaunch) | `bin/rocprofv3.py` |
| `config::kernel_replay` (`ROCPROF_KERNEL_REPLAY`) | `lib/rocprofiler-sdk-tool/config.hpp` |
| Per-thread published pass index | `lib/rocprofiler-sdk-tool/tool.cpp` |
| `replay_user_data_t` pack/unpack (tid + pass in one 64-bit slot) | `tool.cpp` |
| `get_replay_profile` (pass index to counter group) | `tool.cpp` |
| Group selection in the dispatch callback | `tool.cpp` |
| `replay_pass` on the record | `tool.cpp`, `lib/output/counter_info.hpp` |
| `kernel_replay_pass_count_callback` | `tool.cpp` |
| `kernel_replay_callback` (CONFIG + PASS) | `tool.cpp` |
| Context creation and start | `tool.cpp` |

`Replay_Pass` is serialized to JSON but is not emitted in `counter_collection.csv`; the header list
and row writer are both in `lib/output/generateCSV.cpp`.

### 2.6 Tests

| Test | Location |
|---|---|
| Snapshot / restore unit tests | `kernel_replay/tests/snap_restore.cpp` |
| Local context override unit tests | `kernel_replay/tests/local_context_test.cpp` |
| Integration binary (vecAdd, saxpy, vecScale; self-validating) | `projects/rocprofiler-sdk/tests/bin/kernel-replay/kernel_replay.cpp` |
| End-to-end rocprofv3 test across counter groups | `projects/rocprofiler-sdk/tests/rocprofv3/kernel-replay/CMakeLists.txt` |
| Structural validation of passes | `projects/rocprofiler-sdk/tests/rocprofv3/kernel-replay/validate.py` |

## 3. Sequence — one replayed dispatch, N passes

```mermaid
sequenceDiagram
    autonumber
    participant APP as App launch thread (HIP)
    participant WI as WriteInterceptor<br/>hsa/queue.cpp
    participant KR as kernel_replay<br/>replay_callbacks + local_context
    participant TOOL as Tool callback
    participant MEM as memory_snapshot<br/>+ memory_tracker
    participant GPU as AQL queue / GPU
    participant ASH as AsyncSignalHandler<br/>HSA signal thread

    APP->>WI: kernel launch, one dispatch packet
    WI->>KR: has_active_replay_contexts
    KR-->>WI: true
    WI->>KR: execute_config_phase_enter
    KR->>TOOL: CONFIG PHASE_ENTER
    TOOL-->>KR: sets pass_count_cb, optional replay_continue_cb
    KR->>TOOL: pass_count_cb(dispatch_info, user_data)
    TOOL-->>KR: N
    Note over KR: N <= 1 and not indefinite<br/>=> CONFIG PHASE_EXIT and fall through<br/>to the normal single-pass path

    WI->>WI: lock agent_replay_mutex (exclusive, whole window)
    WI->>GPU: drain barrier packet
    GPU-->>WI: drain signal reaches 0 (blocking CPU wait)
    WI->>GPU: Queue::sync on every queue of this agent
    Note over WI,GPU: Queue::sync has a five-second hint and only warns on timeout
    WI->>MEM: snap(agent)
    MEM->>MEM: snap_inventory(agent) then hsa_memory_copy D2H per block
    MEM-->>WI: device_snapshot_t (host vectors)
    WI->>KR: scoped_local_context_control (override map for the loop)

    loop pass = 0 .. N-1  (or until replay_continue_cb returns 0)
        WI->>KR: execute_pass_phase_enter(pass)
        KR->>KR: set_toggles_armed(true)
        KR->>TOOL: PASS PHASE_ENTER (current_pass, total_passes)
        TOOL-->>KR: optional replay_local_start/stop_context_cb
        KR->>KR: set_toggles_armed(false)
        WI->>WI: store pass_done = 1
        WI->>WI: process_packet_batch(replay_state)
        Note over WI: shared dispatch_id, app completion signal suppressed,<br/>services consult the override at dispatch time,<br/>pass_done barrier appended last
        WI->>GPU: submit start / kernel / read / stop / pass_done
        GPU-->>WI: pass_done reaches 0 (blocking CPU wait)
        GPU--)ASH: per-pass completion signal
        ASH-)TOOL: counter and dispatch records for this pass
        WI->>KR: execute_pass_phase_exit(pass)
        WI->>KR: should_continue_replay
        alt continue
            WI->>MEM: restore(snapshot) — every captured block written back
        else stop
        end
    end

    WI->>KR: execute_config_phase_exit
    WI->>GPU: barrier carrying the application completion signal
    WI->>WI: destroy drain and pass_done signals, release agent_replay_mutex
    WI-->>APP: return from WriteInterceptor
    GPU-->>APP: application completion signal fires once
```

The pass loop waits on `pass_done`, which is a GPU barrier. It is not synchronized with
`AsyncSignalHandler`, so the restore for a pass can begin before the handler has finished reading
that pass out.

## 4. Current scope

What the implementation covers today, as a reader of the code needs to know it:

- Replay applies to a single-dispatch write of one packet. Multi-packet writes and graph launches
  take the normal path.
- Exclusion is a per-agent mutex acquired inside the replay branch, so it covers replayed dispatches
  against each other rather than against all dispatches on the agent.
- Snapshots are held in host memory as one buffer per tracked allocation, and restore writes back
  every captured block rather than only modified ones.
- The tracked inventory comes from the HSA pool and region allocators. Allocations that establish
  their address through `hsa_amd_vmem_map`, and module-scope device variables, are not tracked.
- Host memory is deliberately out of scope, as is cache state, so cache-sensitive counters can vary
  across passes.
- Both replay waits use an unbounded timeout, and the agent-wide drain is best-effort because
  `Queue::sync` warns and returns rather than failing on timeout.
- The per-pass override is consulted by kernel dispatch tracing, dispatch counter collection and
  dispatch thread trace. PC sampling, SPM and device counter collection do not consult it, and the
  override cannot activate a context that is globally inactive.
- Snapshot and restore failures are logged and skipped rather than failing the replay.
- There is no unit test of the replay loop itself, and no multi-threaded, multi-stream, multi-GPU or
  graph-launch coverage.
