# HIP Contract Test Coverage Tracker

This document tracks the public HIP API semantic contract coverage in `projects/hip-tests/catch/contract`.

The percentages below are approximate API-name coverage against declarations parsed from `projects/hip/include/hip/hip_runtime_api.h`. They are not behavioral coverage: one API can have many modes and edge cases, and the contract layer intentionally covers only small, portable semantic guarantees.

## Snapshot

- Contract tests: 138
- Declared HIP runtime APIs parsed from `hip_runtime_api.h`: 495
- Declared HIP runtime APIs directly exercised by contract tests: 118
- Approximate declared API-name coverage: 23.8%
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
| `host_memory` | 5 |
| `pitched_memory` | 4 |
| `array_memory` | 4 |
| `managed_memory` | 5 |
| `memory_pool` | 6 |
| `memory_pool_lifecycle` | 4 |
| `memory_pool_access` | 3 |
| `vmm` | 5 |
| `copy3d` | 4 |
| `array3d` | 3 |
| `texture` | 7 |
| `context` | 6 |

## Coverage by API category

| Category | Covered | Total parsed | Approx. coverage |
|---|---:|---:|---:|
| Error handling | 3 | 3 | 100.0% |
| Event | 5 | 8 | 62.5% |
| Occupancy | 3 | 7 | 42.9% |
| Graph / capture | 25 | 96 | 26.0% |
| Stream | 5 | 23 | 21.7% |
| Runtime / device | 21 | 45 | 46.7% |
| Kernel launch / function attrs | 1 | 13 | 7.7% |
| Memory / copy / memset | 45 | 137 | 32.8% |
| Other runtime APIs | 1 | 56 | 1.8% |
| Module / library loading | 0 | 29 | 0.0% |
| Texture / surface | 7 | 44 | 15.9% |
| Context / driver | 2 | 16 | 12.5% |
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
hipHostMalloc
hipHostFree
hipHostRegister
hipHostUnregister
hipHostGetDevicePointer
hipHostGetFlags
hipMallocPitch
hipMemcpy2D
hipMalloc3D
hipMemcpy3D
hipMallocArray
hipFreeArray
hipMemcpy2DToArray
hipMemcpy2DFromArray
hipMalloc3DArray
hipArrayGetInfo
hipMallocManaged
hipMemPrefetchAsync
hipMallocAsync
hipFreeAsync
hipMemPoolGetAttribute
hipMemPoolSetAttribute
hipMemPoolCreate
hipMemPoolDestroy
hipMemPoolTrimTo
hipMallocFromPoolAsync
hipMemPoolSetAccess
hipMemPoolGetAccess
```

### Virtual memory management

```text
hipMemGetAllocationGranularity
hipMemAddressReserve
hipMemAddressFree
hipMemCreate
hipMemRelease
hipMemMap
hipMemUnmap
hipMemSetAccess
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
hipDeviceGetDefaultMemPool
hipDeviceGetMemPool
hipDeviceSetMemPool
hipDeviceGet
hipDeviceGetName
hipDeviceComputeCapability
hipDeviceTotalMem
hipDeviceGetUuid
hipDeviceGetPCIBusId
hipDevicePrimaryCtxRetain
hipDevicePrimaryCtxGetState
hipDevicePrimaryCtxRelease
```

### Context / driver

```text
hipCtxGetCurrent
hipCtxGetDevice
```

### Texture / surface

```text
hipCreateTextureObject
hipDestroyTextureObject
hipGetTextureObjectResourceDesc
hipGetTextureObjectTextureDesc
hipCreateSurfaceObject
hipDestroySurfaceObject
hipGetChannelDesc
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

1. Memory surface beyond current basics: peer copies, advanced VMM operations (multi-device access descriptors, export/import handles, and protection-mode matrices), and remaining advanced memory-pool operations (pool export/import handles and multi-device access descriptors). Host allocation/registration, pitched allocation with 2D copies, basic array allocation with 2D array copies, 3D pitched allocation with host-device 3D copies, 3D array allocation with 3D copy-to/from-array, managed allocation with prefetch, default memory pools with stream-ordered allocation, basic virtual memory management (granularity, address reserve/free, allocation handle create/release, map/unmap, and single-device access), explicit memory-pool lifecycle (create/destroy, release-threshold, trim, and pool-specific async allocation), and current-device memory-pool access control are now covered.
2. Texture and surface APIs beyond current basics: texture/surface object create/destroy with resource and texture descriptor round-trips and channel-descriptor queries are now covered; texture reference APIs, mipmapped arrays, and bound/linear texture variants remain.
3. Module, library, and code-loading APIs.
4. Context and driver-style APIs beyond current basics: device-handle, name, compute-capability, total-memory, UUID, and PCI bus-id queries, primary-context retain/get-state/release, and current-context/device queries are now covered; context create/destroy, push/pop/set-current, cache and shared-memory config, and context peer access remain.
5. Advanced graph APIs: graph update, node find in clone, host nodes, child graphs, memory alloc/free nodes, attributes, debug, user objects.
6. IPC, peer, and multigpu APIs.

## Update procedure

When adding or removing contract tests:

1. Recompute the contract-domain test counts from `catch/contract/*/hip_*_contract.cc`.
2. Recompute directly exercised `hip*` APIs from the contract sources.
3. Recompute the public API denominator from `projects/hip/include/hip/hip_runtime_api.h`.
4. Update this document in the same branch as the contract-test change.
5. Keep percentages labeled as approximate API-name coverage, not behavioral coverage.
