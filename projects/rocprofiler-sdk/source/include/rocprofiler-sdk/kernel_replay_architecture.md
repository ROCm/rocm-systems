# Kernel Replay — Software Architecture

> Branch: `users/vkale/kernel-replay-simplified` (PR #7358)

---

## Architecture Diagram

![Kernel Replay Architecture](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/include/rocprofiler-sdk/kernel_replay_architecture_v2.png?raw=true)

---

## Source Reference Table

### 1. Public API

| Component | File | Lines | GitHub |
|-----------|------|-------|--------|
| `rocprofiler_configure_kernel_replay_counting_service()` declaration | `kernel_replay.h` | 64–70 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/include/rocprofiler-sdk/kernel_replay.h#L64-L70) |
| Implementation | `kernel_replay.cpp` | 33–68 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/kernel_replay.cpp#L33-L68) |

### 2. Context Layer

| Component | File | Lines | GitHub |
|-----------|------|-------|--------|
| `kernel_replay_service` struct | `context.hpp` | 103–109 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/context/context.hpp#L103-L109) |
| `dispatch_counter_collection_service` struct | `context.hpp` | 78–89 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/context/context.hpp#L78-L89) |
| `context.kernel_replay` field | `context.hpp` | 160 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/context/context.hpp#L160) |
| `kernel_replay_is_enabled()` helper | `context.hpp` | 176–183 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/context/context.hpp#L176-L183) |

### 3. HSA Interception Layer

#### 3.1 Memory Tracker

| Component | File | Lines | GitHub |
|-----------|------|-------|--------|
| Public interface | `memory_tracker.hpp` | 42–68 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/memory_tracker.hpp#L42-L68) |
| HSA hook wrappers (`pool_allocate_wrapper`, etc.) | `memory_tracker.cpp` | 69–103 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/memory_tracker.cpp#L69-L103) |
| `update_table()` — installs hooks | `memory_tracker.cpp` | 139–159 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/memory_tracker.cpp#L139-L159) |
| `snap_inventory()` | `memory_tracker.cpp` | 132–137 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/memory_tracker.cpp#L132-L137) |

#### 3.2 Queue WriteInterceptor (Replay Loop)

| Component | File | Lines | GitHub |
|-----------|------|-------|--------|
| `replay_pass_state` struct | `queue.cpp` | 93–98 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L93-L98) |
| `kernel_replay_passes()` (env-driven pass count) | `queue.cpp` | 102–114 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L102-L114) |
| `kernel_replay_active()` | `queue.cpp` | 117–123 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L117-L123) |
| `process_packet_batch` lambda | `queue.cpp` | 431–769 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L431-L769) |
| Replay gate + loop | `queue.cpp` | 771–832 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L771-L832) |
| Drain barrier + snap | `queue.cpp` | 785–797 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L785-L797) |
| Per-pass loop body | `queue.cpp` | 807–824 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L807-L824) |
| `fixed_dispatch_id` shared across passes | `queue.cpp` | 574–579 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L574-L579) |
| App completion barrier (final pass only) | `queue.cpp` | 691–698 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L691-L698) |
| `pass_done` barrier append | `queue.cpp` | 760–761 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L760-L761) |
| `write_interceptor()` call site | `queue.cpp` | 621–639 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L621-L639) |

#### 3.3 Memory Snapshot

| Component | File | Lines | GitHub |
|-----------|------|-------|--------|
| `Snapshot` class definition | `memory_snapshot.hpp` | 46–65 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/memory_snapshot.hpp#L46-L65) |
| `snap()` — full device→host→file save | `memory_snapshot.cpp` | 149–226 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/memory_snapshot.cpp#L149-L226) |
| `snap_module_variables()` | `memory_snapshot.cpp` | 229–321 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/memory_snapshot.cpp#L229-L321) |
| `restore()` — dirty-page diffing + H2D writeback | `memory_snapshot.cpp` | 323–403 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/memory_snapshot.cpp#L323-L403) |
| `restore_module_variables()` | `memory_snapshot.cpp` | 406–430 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/memory_snapshot.cpp#L406-L430) |
| MurmurHash3 implementation | `memory_snapshot.cpp` | 62–106 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/memory_snapshot.cpp#L62-L106) |
| `dma_copy()` helper | `memory_snapshot.cpp` | 108–114 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/memory_snapshot.cpp#L108-L114) |

### 4. Counter Collection

| Component | File | Lines | GitHub |
|-----------|------|-------|--------|
| `configure_callback_dispatch()` | `counters/core.cpp` | 268 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/counters/core.cpp#L268) |

### 5. Benchmarks & Tests

| Component | File | Lines | GitHub |
|-----------|------|-------|--------|
| Replay benchmarks (VecAdd, AtomicInc, LargeBuffer) | `replay_benchmarks.cpp` | 59–195 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/counters/tests/benchmarks/replay_benchmarks.cpp#L59-L195) |
| Benchmark kernels | `kernels.hip` | 1–43 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/counters/tests/benchmarks/kernels.hip#L1-L43) |
| Benchmark CMakeLists | `CMakeLists.txt` | 1–31 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/counters/tests/benchmarks/CMakeLists.txt#L1-L31) |
| Counter tests (kernel_replay mutual exclusion) | `counters/tests/core.cpp` | 28 (diff) | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/counters/tests/core.cpp) |

### 6. rocprofv3 Tool & CLI Integration

Kernel replay is exposed through the `rocprofv3` tool so it can be driven from the command line. When enabled, counter collection is routed through the replay service instead of the regular dispatch counting service (the two are mutually exclusive), the existing dispatch/record callbacks are reused unchanged, and a `Replay_Pass` column is added to the counter CSV.

| Component | File | Purpose |
|-----------|------|---------|
| `--kernel-replay` / `--kernel-replay-passes` CLI flags | `source/bin/rocprofv3.py` | Sets `ROCPROF_KERNEL_REPLAY=1` and `ROCPROFILER_KERNEL_REPLAY_PASSES=N`; validates `--kernel-replay` requires `--pmc`. |
| `ROCPROF_KERNEL_REPLAY` config flag | `rocprofiler-sdk-tool/config.hpp` | Tool-side opt-in read from the environment. |
| Replay-service branch | `rocprofiler-sdk-tool/tool.cpp` | When opted in, calls `rocprofiler_configure_kernel_replay_counting_service()` instead of `rocprofiler_configure_callback_dispatch_counting_service()`; assigns a per-dispatch `replay_pass` index in `counter_record_callback`. |
| `replay_pass` record field | `lib/output/counter_info.hpp` | Per-record 0-based pass index (+ serialization for JSON/rocpd). |
| `Replay_Pass` CSV column (gated) | `lib/output/generateCSV.cpp`, `lib/output/csv.hpp` | 20-column encoder used only when replay is active; default 19-column schema is unchanged otherwise. |
| `kernel_replay` output config | `lib/output/output_config.hpp/.cpp` | Carries `ROCPROF_KERNEL_REPLAY` into the CSV generator to gate the column. |

**Usage:** `rocprofv3 --pmc SQ_WAVES SQ_INSTS_VALU --kernel-replay --kernel-replay-passes 3 -- <app>` → `counter_collection.csv` with `N` rows per dispatch, distinguished by `Replay_Pass`.

---

## Sequence Diagram

![Kernel Replay Sequence Diagram](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/include/rocprofiler-sdk/kernel_replay_sequence_diagram_v2.png?raw=true)

---

## Sequence — Step-by-Step Source References

| Step | Action | File | Lines | GitHub |
|------|--------|------|-------|--------|
| 1 | Dispatch intercepted by WriteInterceptor | `queue.cpp` | 778–779 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L778-L779) |
| 2 | Gate: `kernel_replay_active()` + pass count | `queue.cpp` | 117–123, 103–113 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L117-L123) |
| 3 | Submit drain barrier + wait | `queue.cpp` | 785–793 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L785-L793) |
| 4 | `snap_inventory()` | `memory_tracker.cpp` | 132–137 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/memory_tracker.cpp#L132-L137) |
| 5 | `snapshot.snap()` | `memory_snapshot.cpp` | 149–226 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/memory_snapshot.cpp#L149-L226) |
| 5a | DMA device→host (chunked) | `memory_snapshot.cpp` | 193 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/memory_snapshot.cpp#L193) |
| 5b | Write file + MurmurHash3 per page | `memory_snapshot.cpp` | 200–207 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/memory_snapshot.cpp#L200-L207) |
| 5c | `snap_module_variables()` | `memory_snapshot.cpp` | 229–321 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/memory_snapshot.cpp#L229-L321) |
| 6 | Set `is_final`, reset `pass_done` | `queue.cpp` | 809–810 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L809-L810) |
| 7 | `process_packet_batch(..., &replay_state)` | `queue.cpp` | 812–816 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L812-L816) |
| 8 | `write_interceptor()` → AQL counter packets | `queue.cpp` | 621–639 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L621-L639) |
| 9a | `fixed_dispatch_id` shared across passes | `queue.cpp` | 574–579 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L574-L579) |
| 9b | App completion barrier (final pass only) | `queue.cpp` | 691–698 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L691-L698) |
| 9c | `pass_done` barrier appended | `queue.cpp` | 760–761 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L760-L761) |
| 10 | `writer(transformed_packets)` | `queue.cpp` | 768 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L768) |
| 11 | `hsa_signal_wait(pass_done)` | `queue.cpp` | 818–819 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L818-L819) |
| 12 | `snapshot.restore()` (not final) | `queue.cpp` | 823 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L823) |
| 12a | DMA D2H (read current GPU pages) | `memory_snapshot.cpp` | 348 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/memory_snapshot.cpp#L348) |
| 12b | Hash + diff vs `checksums1` | `memory_snapshot.cpp` | 375–394 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/memory_snapshot.cpp#L375-L394) |
| 12c | DMA H2D (dirty pages only) | `memory_snapshot.cpp` | 366 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/memory_snapshot.cpp#L366) |
| 12d | `restore_module_variables()` | `memory_snapshot.cpp` | 406–430 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/memory_snapshot.cpp#L406-L430) |
| 13 | Signal cleanup | `queue.cpp` | 829–830 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L829-L830) |
| 14 | App completion signal fires | `queue.cpp` | 691–698 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L691-L698) |
| 15 | `record_callback` via AsyncSignalHandler | `queue.cpp` | 152–274 | [link](https://github.com/ROCm/rocm-systems/blob/users/vkale/kernel-replay-simplified/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue.cpp#L152-L274) |
