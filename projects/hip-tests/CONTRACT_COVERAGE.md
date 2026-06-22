# HIP Contract Test Coverage Tracker

This document tracks the public HIP API semantic contract coverage in `projects/hip-tests/catch/contract`.

The percentages below are approximate API-name coverage against declarations parsed from `projects/hip/include/hip/hip_runtime_api.h`. They are not behavioral coverage: one API can have many modes and edge cases, and the contract layer intentionally covers only small, portable semantic guarantees.

## Snapshot

- Contract tests: 82
- Declared HIP runtime APIs parsed from `hip_runtime_api.h`: 495
- Declared HIP runtime APIs directly exercised by contract tests: 61
- Approximate declared API-name coverage: 12.3%
- Additional public macro exercised: `hipLaunchKernelGGL`

## Contract domains

| Contract domain | Tests |
|---|---:|
| `memory` | 5 |
| `transfer` | 4 |
| `runtime` | 10 |
| `device` | 7 |
| `stream_event` | 7 |
| `async_transfer` | 4 |
| `memset` | 6 |
| `error_api` | 6 |
| `kernel` | 4 |
| `graph` | 5 |
| `occupancy` | 3 |
| `graph_capture` | 4 |
| `graph_kernel` | 3 |
| `graph_event` | 3 |
| `graph_topology` | 5 |
| `graph_clone` | 3 |
| `graph_update` | 3 |

## Coverage by API category

| Category | Covered | Total parsed | Approx. coverage |
|---|---:|---:|---:|
| Error handling | 3 | 3 | 100.0% |
| Event | 5 | 8 | 62.5% |
| Occupancy | 3 | 7 | 42.9% |
| Graph / capture | 25 | 96 | 26.0% |
| Stream | 5 | 23 | 21.7% |
| Runtime / device | 9 | 45 | 20.0% |
| Kernel launch / function attrs | 1 | 13 | 7.7% |
| Memory / copy / memset | 9 | 137 | 6.6% |
| Other runtime APIs | 1 | 56 | 1.8% |
| Module / library loading | 0 | 29 | 0.0% |
| Texture / surface | 0 | 44 | 0.0% |
| Context / driver | 0 | 16 | 0.0% |
| Extension / proc address | 0 | 13 | 0.0% |
| IPC | 0 | 5 | 0.0% |

## Covered APIs

### Memory / copy / memset

```text
hipMalloc
hipFree
hipMemcpy
hipMemcpyAsync
hipMemset
hipMemsetAsync
hipMemsetD8
hipMemsetD16
hipMemsetD32
```

### Runtime / device

```text
hipInit
hipGetDeviceCount
hipGetDevice
hipSetDevice
hipDeviceGetAttribute
hipGetDeviceProperties
hipDeviceSynchronize
hipRuntimeGetVersion
hipDriverGetVersion
```

### Error handling

```text
hipGetErrorName
hipGetErrorString
hipGetLastError
hipPeekAtLastError
```

### Stream / event

```text
hipStreamCreate
hipStreamDestroy
hipStreamSynchronize
hipStreamQuery
hipStreamWaitEvent
hipEventCreate
hipEventDestroy
hipEventRecord
hipEventSynchronize
hipEventQuery
```

### Kernel launch

```text
hipLaunchKernel
hipLaunchKernelGGL
```

### Graph / capture / graph nodes

```text
hipGraphCreate
hipGraphDestroy
hipGraphInstantiate
hipGraphLaunch
hipGraphExecDestroy
hipGraphAddEmptyNode
hipGraphAddMemcpyNode1D
hipGraphAddMemsetNode
hipGraphAddKernelNode
hipGraphAddEventRecordNode
hipGraphAddEventWaitNode
hipGraphExecMemcpyNodeSetParams1D
hipGraphExecMemsetNodeSetParams
hipGraphExecKernelNodeSetParams
hipGraphKernelNodeGetParams
hipGraphGetNodes
hipGraphGetRootNodes
hipGraphGetEdges
hipGraphNodeGetDependencies
hipGraphNodeGetDependentNodes
hipGraphClone
hipStreamBeginCapture
hipStreamEndCapture
hipStreamIsCapturing
hipStreamGetCaptureInfo
```

### Occupancy

```text
hipOccupancyMaxActiveBlocksPerMultiprocessor
hipOccupancyMaxPotentialBlockSize
hipOccupancyAvailableDynamicSMemPerBlock
```

## Largest remaining gaps

1. Memory surface beyond basics: arrays, pitched/2D/3D copies, host memory, memory pools, VMM, managed memory, peer copies.
2. Texture and surface APIs.
3. Module, library, and code-loading APIs.
4. Context and driver-style APIs.
5. Advanced graph APIs: graph update, node find in clone, host nodes, child graphs, memory alloc/free nodes, attributes, debug, user objects.
6. IPC, peer, and multigpu APIs.

## Update procedure

When adding or removing contract tests:

1. Recompute the contract-domain test counts from `catch/contract/*/hip_*_contract.cc`.
2. Recompute directly exercised `hip*` APIs from the contract sources.
3. Recompute the public API denominator from `projects/hip/include/hip/hip_runtime_api.h`.
4. Update this document in the same branch as the contract-test change.
5. Keep percentages labeled as approximate API-name coverage, not behavioral coverage.
