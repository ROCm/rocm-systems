# ROCprofiler-SDK & rocprofv3 — GPU Profiling Overview

> Focused on the **kernel replay** feature (PR #7358) and its relationship to the broader profiling architecture.

---

## 1. Fundamentals of GPU Profiling on AMD Hardware

GPU profiling on AMD Instinct/RDNA hardware operates at three layers:

| Layer | What it provides | Mechanism |
|-------|-----------------|-----------|
| **Hardware counters** | Per-shader-engine register values (SQ_WAVES, SQ_INSTS_VALU, etc.) | AQL profile packets configure GRBM/SQ/TA/GL2C counters via PM4 commands |
| **Kernel dispatch tracing** | Timestamps, grid/workgroup sizes, kernel IDs | HSA queue interception + signal completion callbacks |
| **Streaming telemetry** | Continuous time-series of counter values | SPM (Streaming Performance Monitor) hardware writes ring buffers at configurable intervals |

The fundamental constraint: **hardware has a limited number of counter registers per block**. When you request more counters than can be collected in a single pass, the profiler must either:
- Run the application multiple times (rocprofv3 multi-pass: `--pmc "group1" --pmc "group2"`)
- Re-execute individual kernels with memory restore (**kernel replay** — this PR)

---

## 2. ROCprofiler-SDK Architecture

### Library Stack

```
┌──────────────────────────────────────────────────────────────┐
│  User Application (hipMalloc, hipLaunchKernel, etc.)         │
├──────────────────────────────────────────────────────────────┤
│  HIP Runtime → HSA Runtime                                   │
├──────────────────────────────────────────────────────────────┤
│  librocprofiler-sdk.so (the profiling SDK)                   │
│    ├── Registration (ROCP_TOOL_LIBRARIES discovery)          │
│    ├── Context management                                    │
│    ├── HSA table interception (queue write, signals, memory) │
│    ├── Counter collection (dispatch + device modes)          │
│    ├── Kernel replay (multi-pass per-dispatch)  ← THIS PR   │
│    ├── SPM (streaming performance monitor)                   │
│    ├── PC sampling (instruction-level)                       │
│    ├── Thread trace (wavefront execution trace)              │
│    ├── Tracing (HIP/HSA/KFD/RCCL/Marker APIs)              │
│    └── Code object tracking                                  │
├──────────────────────────────────────────────────────────────┤
│  librocprofiler-sdk-tool.so (rocprofv3 tool library)         │
│    Configures SDK services, manages output (CSV/JSON/OTF2)   │
├──────────────────────────────────────────────────────────────┤
│  rocprofv3 (Python CLI)                                      │
│    Parses --pmc/--spm/--kernel-trace, sets env vars,         │
│    LD_PRELOADs SDK + tool, launches the app                  │
└──────────────────────────────────────────────────────────────┘
```

### How rocprofv3 Connects to the SDK

1. `rocprofv3.py` translates CLI flags → `ROCPROF_*` env vars
2. Sets `LD_PRELOAD=librocprofiler-sdk-tool.so:librocprofiler-sdk.so`
3. SDK's `registration.cpp` discovers tools via `ROCP_TOOL_LIBRARIES`
4. Calls tool's `rocprofiler_configure()` → `tool_init()` sets up contexts/services
5. HSA queue interception begins; counter packets are injected per dispatch

### Key Source Locations

| Component | File | Entry Point |
|-----------|------|-------------|
| CLI launcher | [`rocprofv3.py:2192`](https://github.com/ROCm/rocm-systems/blob/develop/projects/rocprofiler-sdk/source/bin/rocprofv3.py#L2192) | `main()` |
| Tool library | [`tool.cpp:3951`](https://github.com/ROCm/rocm-systems/blob/develop/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk-tool/tool.cpp#L3951) | `rocprofiler_configure()` |
| SDK registration | `source/lib/rocprofiler-sdk/registration.cpp` | `ROCP_TOOL_LIBRARIES` |
| Context creation | `source/lib/rocprofiler-sdk/context/context.hpp` | `allocate_context()` |

---

## 3. Kernel Replay and Its Place in the Architecture

### What Kernel Replay Does

Kernel replay solves the multi-pass counter collection problem **at the per-dispatch level** rather than re-running the entire application:

| Approach | Scope | Memory safety | Latency |
|----------|-------|---------------|---------|
| rocprofv3 multipass (`--pmc` groups) | Whole application re-run | N/A (fresh process) | O(N × app runtime) |
| **Kernel replay** | Single dispatch re-execution | Device memory snapshot/restore | O(N × kernel time + snap/restore) |
| Counter group rotation (`ROCPROF_COUNTER_GROUPS_INTERVAL`) | Amortized across dispatches | No replay; samples different dispatches | O(1 × app runtime) |

### Kernel Replay vs SPM — Similarities and Differences

| Aspect | Kernel Replay | SPM |
|--------|--------------|-----|
| **Goal** | Collect all requested counters for one kernel | Continuous time-series of counters across all kernels |
| **Mechanism** | Re-execute kernel N times, restore memory between passes | Hardware writes counter samples to ring buffer at clock intervals |
| **Context service struct** | `kernel_replay_service` | `spm_dispatch_counter_collection_service` |
| **Mutual exclusion** | With regular dispatch counting on same context | With regular dispatch counting on same context |
| **AQL packet injection** | Reuses `process_packet_batch` counter start/stop packets per pass | Injects SPM start/stop PM4 packets around dispatch |
| **Queue interception** | Same HSA WriteInterceptor in `queue.cpp` | Same HSA WriteInterceptor in `queue.cpp` |
| **Counter config** | `rocprofiler_counter_config_id_t` (shared with dispatch counting) | `rocprofiler_counter_config_id_t` (offset to avoid ID collision) |
| **Data callback** | `record_callback` per pass | `spm_data_callback` per sample window |
| **Beta gating** | `ROCPROFILER_SDK_EXPERIMENTAL` attribute | `ROCPROFILER_SPM_BETA_ENABLED` env var |
| **Memory management** | Full device memory snapshot/restore (file-backed, dirty-page diffing) | SPM ring buffer in device memory (no restore needed) |
| **Packet caching** | N/A (reuses existing counter packets) | `AQLPacket` cache in `spm_counter_config` |

### Shared Infrastructure

Both kernel replay and SPM share:
- The HSA queue WriteInterceptor (`queue.cpp`)
- `process_packet_batch` lambda for packet transformation
- Counter configuration APIs (`rocprofiler_create_counter_config`)
- The `context` struct as the service bundle
- AQL profile packet generation (`source/lib/rocprofiler-sdk/aql/`)
- Signal-based synchronization (`hsa_signal_wait`)

---

## 4. Key Contributions by Topic and Contributor

### Counter Collection & Dispatch Profiling

| Contributor | Key Work |
|-------------|----------|
| **Benjamin Welton** | Queue Interposition (#5219) — shadow write pointer alternative; ioctl runtime selection (#5562) |
| **Venkateshwar Reddy Kandula** | Optimize HSA queue write interceptor and async signal handler (#5869) |
| **Sushma Vaddireddy** | Optimize PM4 read + offload completion callback to consumer thread (#6100); gfx1250 support (#5436) |
| **Meng Cao** | Remove counter callbacks on stop (#6832); replace `getenv` with direct environ reads (#6091) |
| **Vivek Kale** (you) | Simplify kernel replay to minimal self-contained feature (this PR) |
| **Jonathan R. Madsen** | Original HSA queue write interceptor optimization (#4276); unit test infrastructure (#921) |

### SPM (Streaming Performance Monitor)

| Contributor | Key Work |
|-------------|----------|
| **SrirakshaNag** | SPM core library implementation (#4337); SPM context/tool support + output format (#4342); SPM counter metrics flag (#4139) |
| **Saurabh Verma** | Fix SPM packet memory leak (#6807) |

### PC Sampling

| Contributor | Key Work |
|-------------|----------|
| **Vladimir Indic** | Multiple contexts with PC sampling (#2449); trap handler latency masking (#1109, #1385); GFX1250 PCS fix (#5652) |
| **Julian Jose** | PC sampling with attach/detach + code_object tracking (#3921) |

### Thread Trace (ATT)

| Contributor | Key Work |
|-------------|----------|
| **Giovanni Lenzi Baraldi** | Double/triple buffering for gfx9/gfx12 (#3233); logging improvements (#5691); async copy fixes (#5684, #5666); gfx9 read offset (#6784) |
| **Gopesh Bhardwaj** | Architecture-gated test skip (#6687) |

### HSA Layer & Queue Infrastructure

| Contributor | Key Work |
|-------------|----------|
| **Benjamin Welton** | Queue interposition (#5219); Abseil logging migration (#4668) |
| **Venkateshwar Reddy Kandula** | Queue write interceptor + signal handler optimization (#5869) |
| **Mythreya Kuricheti** | AQL profile migration (#4933) |
| **Jonathan R. Madsen** | Queue write interceptor original design (#4276) |

### Build, Testing, CI

| Contributor | Key Work |
|-------------|----------|
| **Ian Trowbridge** | Top contributor by volume — CI, build, testing infrastructure |
| **Mark Meserve** | Attachment tests (#3933); general testing |
| **Dana Robinson** | Codespell/code quality fixes (#6964) |
| **Jakub Kuderski** | Recent build/config contributions |

---

## 5. How Kernel Replay Impacts/Connects to Other Subsystems

```
kernel_replay.h/.cpp (Public API + configuration)
       │
       ├──→ context/context.hpp (kernel_replay_service struct, mutual exclusion)
       │
       ├──→ counters/core.cpp (configure_callback_dispatch — reuses counter path)
       │
       ├──→ hsa/memory_tracker.hpp/.cpp (allocation inventory for snap/restore)
       │         └── Hooks hsa_memory_allocate/free, hsa_amd_memory_pool_allocate/free
       │
       ├──→ hsa/memory_snapshot.hpp/.cpp (device memory save/restore)
       │         └── File-backed, dirty-page diffing, module variable capture
       │
       └──→ hsa/queue.cpp (WriteInterceptor replay loop)
                 ├── Shares process_packet_batch with regular dispatch counting
                 ├── Shares signal infrastructure with SPM
                 └── Shares serializer/async handler with all kernel profiling
```

### What Would Break If Kernel Replay Changes

- Any change to `process_packet_batch` packet ordering affects replay pass correctness
- Changes to HSA memory allocation wrappers (used by runtime) impact the memory tracker
- Signal handling changes in `AsyncSignalHandler` affect per-pass record delivery
- Counter config/profile rotation logic must remain compatible with `fixed_dispatch_id`

---

## 6. Current Development Activity (Recent 50 Commits)

| Area | Recent activity |
|------|----------------|
| CI/Build | Multi-arch CI, occupancy tests gating, format fixes (Ian Trowbridge) |
| RCCL | Broadcasting operations update (rocshmem) |
| HIP | Doxygen cooperative groups fix, fsanitize option handling |
| hipFile | Relocatable unit-test assets revert |
| ROCr (HSA runtime) | Kernel trampoline install, AQLPROFILE extension gating |
| Kernel Replay | Your branch: simplified API + memory snapshot + benchmarks |

---

## 7. Environment Variables Reference

| Variable | Layer | Purpose |
|----------|-------|---------|
| `ROCPROFILER_KERNEL_REPLAY_PASSES` | SDK | Number of replay passes per dispatch (default: 1 = disabled) |
| `ROCPROFILER_SPM_BETA_ENABLED` | SDK | Gate SPM feature (required for SPM to activate) |
| `ROCPROFILER_REPLAY_SNAPSHOT_CHUNK_BYTES` | SDK | Chunk size for DMA transfers during snap/restore (default: 64 MB) |
| `ROCPROFILER_REPLAY_SNAPSHOT_DIR` | SDK | Directory for snapshot files (default: `/tmp/rocprofiler_replay_<pid>`) |
| `ROCPROF_COUNTER_COLLECTION` | Tool | Enable dispatch counter collection in rocprofv3 |
| `ROCPROF_COUNTERS` / `ROCPROF_COUNTER_GROUPS` | Tool | Counter names / multi-group definitions |
| `ROCPROF_COUNTER_GROUPS_INTERVAL` | Tool | Dispatches between counter group rotation |
| `ROCP_TOOL_LIBRARIES` | SDK | Path to tool .so for registration |

---

## 8. rocprofv3 Integration (`--kernel-replay`)

The `rocprofiler_configure_kernel_replay_counting_service()` API remains **experimental**, but it is now wired into the rocprofv3 tool and CLI:

- **CLI**: `rocprofv3 --pmc COUNTER1 COUNTER2 ... --kernel-replay --kernel-replay-passes N -- <app>`
  - `--kernel-replay` (requires `--pmc`) sets `ROCPROF_KERNEL_REPLAY=1`.
  - `--kernel-replay-passes N` sets `ROCPROFILER_KERNEL_REPLAY_PASSES=N` (consumed by the SDK replay loop in `hsa/queue.cpp`).
- **Tool** (`tool.cpp`): when `ROCPROF_KERNEL_REPLAY` is set, counter collection is configured through `rocprofiler_configure_kernel_replay_counting_service()` instead of `rocprofiler_configure_callback_dispatch_counting_service()` (the two are mutually exclusive). The existing dispatch/record callbacks are reused unchanged, so each replay pass produces one counter record.
- **Output** (`generateCSV.cpp`): when replay is active, `counter_collection.csv` gains a 0-based `Replay_Pass` column. All passes of a replayed dispatch share the same `Dispatch_Id` and are distinguished by `Replay_Pass`. The column is omitted entirely for non-replay runs, leaving the default CSV schema unchanged.

This lets a single application run produce `N` fully-profiled counter records per dispatch (snapshot/restore between passes), as opposed to rocprofv3 multi-pass mode which re-runs the entire application once per counter group.
