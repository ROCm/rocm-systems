# GIN_ANVIL design — test plan and traceability

This document describes the **GIN_ANVIL** (GIN over MI300-class xGMI Anvil SDMA) design in RCCL, the implementation approach, the automated tests added under `test/device/`, known coverage gaps, a traceability matrix, and exact build/run commands.

---

## 1. Summary of the main goals of the design

- **Single-node GPU-initiated transport**: Provide a GIN backend (`NCCL_NET_DEVICE_GIN_ANVIL` / `NCCL_GIN_TYPE_ANVIL = 5`) that uses **Anvil SDMA queues** between peer GPUs on one node, avoiding IB proxy progress where the design allows.
- **Correctness with LSA-style symmetric windows**: Collective windows registered through the devr LSA flat layout expose a **rank-0 base** and **stride** so device code can form peer VAs via `ncclGinAnvilRankPtr`.
- **Latency vs throughput policy**: Transfers **smaller than 256 bytes** use **GPU load/store** on peer-mapped memory; larger transfers use **SDMA** with **configurable chunking** (`NCCL_GIN_ANVIL_SDMA_CHUNK_MB`, with in-kernel clamp/fallback).
- **Signaling and counters**: Indexed signals use **pinned shareable cuMem** on the host path so SDMA fused signal or GPU atomic fallback can target the same VA; optional **completion counters** integrate with `putCounter` when no signal is fused.
- **Flush / ordering**: `ncclGinApi_Flush` tracks a **dirty bitmask per peer** across SDMA channels and issues `quiet` on queues that were used.

Non-goals for this document: full multi-node support (`ncclGinAnvilCreateContext` rejects `nNodes > 1`).

---

## 2. Overview of implementation ideas and approaches

| Layer | Location | Role |
|--------|-----------|------|
| **Device API** | `src/include/nccl_device/gin/anvil/gin_anvil.h` | Template specializations (`ncclGinApi_Put`, `Flush`, `GetCounterPtr`, …) and inline helpers (`PeerQueue`, `Memcpy`, chunking, signals). |
| **Shared structs** | `src/include/nccl_device/gin/anvil/gin_anvil_device_host_common.h` | `ncclGinAnvilGPUContext`, `ncclGinAnvilMemHandle`, version constant. |
| **Host plugin** | `src/gin/gin_plugin_anvil.cc` | Stub `ncclGin_t` vtable (listen/connect/properties); real create/register/destroy live in `gin_host_anvil.cc` / `gin_host.cc`. |
| **Host runtime** | `src/gin/gin_host_anvil.cc` (under `ENABLE_ROCSHMEM_GIN`) | `ncclGinAnvilCreateContext`: single-node check, signal cuMem + IPC import, rocshmem Anvil endpoint + per-peer SDMA queue handles copied to device, `ncclGinAnvilGPUContext` array, register/deregister LSA windows. |
| **SDMA packet layer** | rocshmem `sdma/anvil_device.hpp` | `put`, `putCounter`, `quiet`, queue device handles used by `gin_anvil.h`. |

**Build gating**: `NCCL_GIN_ANVIL_ENABLE` is set in `gin_device_common.h` when compiling for AMD HIP and `ENABLE_ROCSHMEM` or `ENABLE_ROCSHMEM_GIN` is defined. The new unit tests compile **`GinAnvilDeviceTests.cpp` only** with `ENABLE_ROCSHMEM_GIN` so `gin_anvil.h` is instantiated without requiring the full RCCL library to be built with GIN.

---

## 3. Each testcase added — what it exercises

All cases live in `test/device/GinAnvilDeviceTests.cpp` (GoogleTest, HIP kernels). Names below match `TEST_F(GinAnvilDeviceTest, …)`.

### Helpers and small pure functions

| Test | Exercises |
|------|------------|
| **RankPtr_Stride** | `ncclGinAnvilRankPtr`: rank indexing with non-zero stride. |
| **Memcpy_ZeroBytes** | `ncclGinAnvilMemcpy`: `bytes == 0` early return. |
| **Memcpy_UnalignedSmall** | `ncclGinAnvilMemcpy`: branch where `(uintptr_t)d \| s` alignment prevents uint64 fast path; byte loop / `__builtin_memcpy` path. |
| **Memcpy_AlignedUint64Path** | `ncclGinAnvilMemcpy`: uint64-aligned bulk copy + tail bytes. |
| **SdmaPutChunks_ZeroBytes** | `ncclGinAnvilSdmaPutChunks`: `bytes == 0` return before any `put`. |
| **SelectSdmaChannel_NumChEdgeCases** | `ncclGinAnvilSelectSdmaChannel`: `numCh <= 1` vs modulo spread when `numCh > 1`. |
| **PeerQueue_NullQueuesOrZeroNumCh** | `ncclGinAnvilPeerQueue`: `queues == nullptr`; `numCh < 1` clamp to 1. |
| **MarkSdmaDirty_PeerBounds** | `ncclGinAnvilMarkSdmaDirty`: `peer` in `[0,31]` vs out of range (no atomic OR). |
| **LocalSignalOp_Branches** | `ncclGinAnvilLocalSignalOp`: null pointer; `ncclGinSignalInc` vs `ncclGinSignalAdd`; Inc forces arg to 1. |
| **RemoteGpuSignalOp_Branches** | `ncclGinAnvilRemoteGpuSignalOp`: null; Inc vs Add on device-visible signal word. |

### `ncclGinApi_Put` — self (peer == rank)

| Test | Exercises |
|------|------------|
| **Put_Self_HandleNull** | `ncclGinAnvilGetCtx` → null `handle`; immediate return. |
| **Put_Self_NoWins_NoSignal** | Self path `!hasWins`, no signal: fall-through return. |
| **Put_Self_NoWins_LocalSignal** | Self `!hasWins` + indexed signal + `ncclGinAnvilLocalSignalOp` / `LocalSignalPtr`. |
| **Put_Self_Wins_NullWindow** | `dstWin == nullptr \|\| srcWin == nullptr`. |
| **Put_Self_Wins_InvalidBases** | `dstRank0Base == 0 \|\| srcRank0Base == 0 \|\| stride == 0`. |
| **Put_Self_Wins_MemcpyAndOptionalSignal** | Self memcpy + optional signal; **hasCounter** branch with `atomicAdd` on counter. |

### `ncclGinApi_Put` — cross-peer (peer != rank)

| Test | Exercises |
|------|------------|
| **Put_Peer_QueueNull** | `ncclGinAnvilPeerQueue` returns null → return before signal/memcpy. |
| **Put_Peer_HasSignal_SigPtrNull** | `hasSignal` + `PeerSignalPtr` null → early return. |
| **Put_Peer_NoWins_RemoteSignal** | `!hasWins` + remote signal op (dummy non-null `q` required by control flow before this block). |
| **Put_Peer_NoWins_NoSignal** | `!hasWins` + `!hasSignal`: return after queue non-null check. |
| **Put_Peer_SubThreshold_Memcpy** | `bytes < ncclGinAnvilSdmaThresholdBytes`: memcpy + optional remote signal + optional counter; **does not** call `rocshmem::anvil::put`. |
| **Put_Peer_Wins_NullWindow** | Null `dstWin`/`srcWin` on peer path. |
| **Put_Peer_Wins_InvalidBases** | Zero bases/stride on peer path. |

### Other template APIs

| Test | Exercises |
|------|------------|
| **GetCounterPtr_And_ResetCounter** | `ncclGinApi_GetCounterPtr`: null ctx handle / null counters / valid pointer. `ncclGinApi_ResetCounter`: null handle; store zero. |
| **GetSignalPtr** | `ncclGinApi_GetSignalPtr` delegating to `LocalSignalPtr`; null `signals`. |
| **ResetSignal_IsNoOp** | `ncclGinApi_ResetSignal::call` body (signal ignored). |
| **Flush_NullCtx** | `ncclGinApi_Flush`: null `aCtx`. |
| **Flush_DirtyMaskZero_Collective** | Full CTA calls `Flush` with `sdmaDirtyMask == 0`: peer loop skips `quiet` (no dirty peers). |
| **IndexedSignalId_OnlyWhenIndexed** | `hasSignal` with type **INDEXED** sets `signalId`; **NONE** leaves default id path in Put self signal branch (smoke). |

### Intentionally not implemented as automated unit tests

| API / branch | Reason |
|----------------|--------|
| **`ncclGinApi_PutValue<NCCL_NET_DEVICE_GIN_ANVIL>`** | Calls `__builtin_unreachable()` — must not be invoked in conforming code. |
| **Large-message SDMA paths** (`ncclGinAnvilSdmaPutChunks` with `bytes > 0`, `putCounter`, `MarkSdmaDirty` after real `put`) | Require a **valid** `SdmaQueueDeviceHandle` and hardware; bogus handles risk infinite loops in `ReserveQueueSpace`. |
| **`ncclGinApi_Flush` with dirty ≠ 0 and non-null queues** | Calls `rocshmem::anvil::quiet(*q)` — same hardware dependency. |
| **`gin_host_anvil.cc`**: env parsing, multi-rank IPC, `ncclGinAnvilCreateContext` success/failure graph, `ncclGinAnvilRegister` LSA resolution | Needs full communicator, drivers, and optional multi-GPU CI; covered in **coverage gaps** and manual/integration tier. |
| **`gin_plugin_anvil.cc` vtable** | Thin stubs; optional future test linking exported `ncclGinAnvilPlugin` if visibility allows. |

---

## 4. Coverage gaps

1. **Real SDMA submission**: `rocshmem::anvil::put`, `putCounter`, `quiet` — require MI300-class Anvil setup and valid queues from `rocshmem::anvil::anvil.getSdmaQueue`.
2. **`ncclGinAnvilSdmaPutChunks`**: non-zero `bytes` branches (while-loop over chunks, `chunkBytes == 0` defaulting to 8 MiB, clamp `chunkBytes < 65536`).
3. **`ncclGinApi_Put` peer branch** for `bytes >= ncclGinAnvilSdmaThresholdBytes`: all three combinations of `(hasSignal, hasCounter)` that call into SDMA helpers.
4. **Host-only paths** in `gin_host_anvil.cc`: `ginAnvilParseNumSdmaChannels`, `ginAnvilParseSdmaChunkBytes`, `ginAnvilAlignSignalCuMemBytes`, `ginAnvilGrantSignalPeerAccess`, `setupSignalBases` failure modes, `ncclGinAnvilCreateContext`/`Register`/`Destroy` integration.
5. **`ncclGinAnvilQueryLastError` / `hasError` flag** mutation (currently no producer sets `hasError` in the shown code).
6. **Multi-context** stress (`nContexts > 1`) and **multi-channel** flush iteration with real queues.

---

## 5. Traceability matrix

| Requirement / behavior | Primary implementation | Automated test(s) | Gap / manual |
|------------------------|------------------------|-------------------|--------------|
| Rank LSA addressing | `ncclGinAnvilRankPtr`, mem handles | `RankPtr_Stride`, `Put_Self_Wins_*` | — |
| Small vs large byte policy threshold | `ncclGinAnvilSdmaThresholdBytes` | `Put_Peer_SubThreshold_Memcpy` | Large side → HW |
| SDMA chunking policy | `ncclGinAnvilSdmaPutChunks`, `sdmaChunkBytes` | `SdmaPutChunks_ZeroBytes` only | Non-zero bytes → HW |
| Channel selection | `ncclGinAnvilSelectSdmaChannel` | `SelectSdmaChannel_NumChEdgeCases` | — |
| Queue lookup | `ncclGinAnvilPeerQueue` | `PeerQueue_NullQueuesOrZeroNumCh`, `Put_Peer_QueueNull` | — |
| Dirty tracking | `ncclGinAnvilMarkSdmaDirty` | `MarkSdmaDirty_PeerBounds` | Post-put dirty + flush → HW |
| Flush / quiet | `ncclGinApi_Flush` | `Flush_*` | `quiet` with real q → HW |
| Self memcpy | `ncclGinApi_Put` self + wins | `Put_Self_Wins_MemcpyAndOptionalSignal` | — |
| Local / remote signals | `LocalSignalOp`, `RemoteGpuSignalOp` | Multiple Put / signal tests | SDMA fused signal |
| Counters | counter pointer + `ResetCounter` | `GetCounterPtr_*`, `Put_Self_Wins_*` | `putCounter` SDMA path → HW |
| Single-node enforcement | `gin_host_anvil.cc` | — | Integration / MPI |
| Plugin identity | `gin_plugin_anvil.cc` | — | Optional link test |
| PutValue unsupported | `gin_anvil.h` | — | By design unreachable |

---

## 6. How to build the tests and exact command lines

### Preconditions

- ROCm **6.4+** (same gate as `rccl-UnitTestsFixtures` in `test/CMakeLists.txt`).
- rocshmem **SDMA headers** available at configure time, either:
  - RCCL configured with **`-DENABLE_ROCSHMEM=ON`** or **`-DENABLE_ROCSHMEM_GIN=ON`** (sets `ROCSHMEM_SDMA_SRC_DIR` via `ROCSHMEM.cmake`), or
  - A mono-repo layout where **`../rocshmem/src/sdma/anvil_device.hpp`** exists relative to the RCCL source root (CMake probes this for the fixtures target).

### Configure and build (example)

From the RCCL build directory:

```bash
cmake -S /path/to/rccl -B /path/to/rccl/build \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DGPU_TARGETS=gfx942 \
  -DBUILD_TESTS=ON \
  -DENABLE_ROCSHMEM_GIN=ON \
  -DROCSHMEM_SOURCE_DIR=/path/to/rocshmem
cmake --build /path/to/rccl/build --target rccl-UnitTestsFixtures -j$(nproc)
```

If rocshmem is a sibling of `rccl` under `projects/`, you may omit `ROCSHMEM_SOURCE_DIR` when the default `../rocshmem/src` probe succeeds.

### Run only GIN Anvil fixture tests

```bash
cd /path/to/rccl/build/test
./rccl-UnitTestsFixtures --gtest_filter='GinAnvilDeviceTest.*'
```

### Run all fixtures (includes existing `GinDeviceTest`, etc.)

```bash
./rccl-UnitTestsFixtures
```

### Optional: RCCL with full rocshmem (integration / future MPI)

```bash
cmake -S /path/to/rccl -B /path/to/rccl/build \
  -DBUILD_TESTS=ON \
  -DENABLE_ROCSHMEM=ON \
  -DROCSHMEM_SOURCE_DIR=/path/to/rocshmem \
  ...
```

Use application-level tests (e.g. `rccl-tests` with `NCCL_GIN_TYPE=5`) for end-to-end validation on supported hardware.

---

## Revision history

| Date | Change |
|------|--------|
| 2026-06-11 | Initial test plan, traceability, and `GinAnvilDeviceTests.cpp` suite. |
