# HIP Contract Test Coverage Tracker

This document tracks the public HIP API semantic contract coverage in `projects/hip-tests/catch/contract`.

The percentages below are approximate API-name coverage against declarations parsed from `projects/hip/include/hip/hip_runtime_api.h`. They are not behavioral coverage: one API can have many modes and edge cases, and the contract layer intentionally covers only small, portable semantic guarantees.

## Snapshot

- Contract tests: 217
- Declared HIP runtime APIs parsed from `hip_runtime_api.h`: 495
- Declared HIP runtime APIs directly exercised by contract tests: 179
- Approximate declared API-name coverage: 36.2%
- Additional public macro exercised: `hipLaunchKernelGGL`
- Additional non-runtime-header APIs exercised: HIPRTC (`hiprtcCreateProgram`, `hiprtcCompileProgram`, `hiprtcGetCodeSize`, `hiprtcGetCode`, `hiprtcDestroyProgram`); these are not declared in `hip_runtime_api.h` and are excluded from the coverage denominator and covered counts.

## Contract domains

| Contract domain | Tests |
|---|---:|
| `memory` | 5 |
| `transfer` | 4 |
| `runtime` | 10 |
| `device` | 7 |
| `device_config` | 6 |
| `stream_event` | 7 |
| `async_transfer` | 4 |
| `memset` | 6 |
| `error_api` | 6 |
| `extension` | 6 |
| `kernel` | 4 |
| `graph` | 5 |
| `occupancy` | 3 |
| `graph_capture` | 4 |
| `graph_kernel` | 3 |
| `graph_event` | 3 |
| `graph_topology` | 5 |
| `graph_clone` | 3 |
| `graph_update` | 9 |
| `graph_node_types` | 5 |
| `graph_child` | 3 |
| `graph_host` | 3 |
| `graph_mem_nodes` | 4 |
| `graph_node_attributes` | 3 |
| `graph_node_find` | 3 |
| `graph_user_objects` | 4 |
| `graph_node_enabled` | 3 |
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
| `ipc` | 5 |
| `module` | 7 |
| `module_load_ex` | 4 |
| `library` | 17 |

## Coverage by API category

| Category | Covered | Total parsed | Approx. coverage |
|---|---:|---:|---:|
| Error handling | 3 | 3 | 100.0% |
| Event | 6 | 8 | 75.0% |
| Occupancy | 3 | 7 | 42.9% |
| Graph / capture | 53 | 96 | 55.2% |
| Stream | 5 | 23 | 21.7% |
| Runtime / device | 28 | 45 | 62.2% |
| Kernel launch / function attrs | 2 | 13 | 15.4% |
| Memory / copy / memset | 45 | 137 | 32.8% |
| Other runtime APIs | 1 | 56 | 1.8% |
| Module / library loading | 15 | 29 | 51.7% |
| Texture / surface | 7 | 44 | 15.9% |
| Context / driver | 2 | 16 | 12.5% |
| Extension / proc address | 4 | 13 | 30.8% |
| IPC | 5 | 5 | 100.0% |

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
hipDeviceGetCacheConfig
hipDeviceSetCacheConfig
hipDeviceGetSharedMemConfig
hipDeviceGetLimit
hipDeviceSetLimit
hipGetDeviceFlags
hipDeviceGetStreamPriorityRange
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
hipEventCreateWithFlags
hipEventDestroy
hipEventRecord
hipEventSynchronize
hipEventQuery
```

### Kernel launch / function attrs

```text
hipLaunchKernel
hipLaunchKernelGGL
hipFuncGetAttribute
```

### Module / library loading

```text
hipModuleLoadData
hipModuleLoadDataEx
hipModuleUnload
hipModuleGetFunction
hipModuleGetGlobal
hipModuleLaunchKernel
hipLibraryLoadData
hipLibraryUnload
hipLibraryGetKernel
hipLibraryGetKernelCount
hipLibraryEnumerateKernels
hipLibraryGetGlobal
hipKernelGetFunction
hipKernelGetLibrary
hipKernelGetName
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
hipGraphAddChildGraphNode
hipGraphChildGraphNodeGetGraph
hipGraphAddHostNode
hipGraphHostNodeGetParams
hipGraphNodeGetType
hipGraphAddDependencies
hipGraphRemoveDependencies
hipGraphExecMemcpyNodeSetParams1D
hipGraphExecMemsetNodeSetParams
hipGraphExecKernelNodeSetParams
hipGraphExecHostNodeSetParams
hipGraphExecChildGraphNodeSetParams
hipGraphExecEventRecordNodeSetEvent
hipGraphExecEventWaitNodeSetEvent
hipGraphExecUpdate
hipGraphKernelNodeGetParams
hipGraphGetNodes
hipGraphGetRootNodes
hipGraphGetEdges
hipGraphNodeGetDependencies
hipGraphNodeGetDependentNodes
hipGraphClone
hipGraphNodeFindInClone
hipGraphAddMemAllocNode
hipGraphMemAllocNodeGetParams
hipGraphAddMemFreeNode
hipGraphMemFreeNodeGetParams
hipDeviceGetGraphMemAttribute
hipDeviceGraphMemTrim
hipStreamBeginCapture
hipStreamEndCapture
hipStreamIsCapturing
hipStreamGetCaptureInfo
hipUserObjectCreate
hipUserObjectRetain
hipUserObjectRelease
hipGraphRetainUserObject
hipGraphReleaseUserObject
hipGraphNodeSetEnabled
hipGraphNodeGetEnabled
hipGraphKernelNodeSetAttribute
hipGraphKernelNodeGetAttribute
```

### Occupancy

```text
hipOccupancyMaxActiveBlocksPerMultiprocessor
hipOccupancyMaxPotentialBlockSize
hipOccupancyAvailableDynamicSMemPerBlock
```

### Extension / proc address

```text
hipGetProcAddress
hipApiName
hipGetStreamDeviceId
hipExtGetLastError
```

### IPC

```text
hipIpcGetMemHandle
hipIpcOpenMemHandle
hipIpcCloseMemHandle
hipIpcGetEventHandle
hipIpcOpenEventHandle
```

## Largest remaining gaps

1. Memory surface beyond current basics: peer copies, advanced VMM operations (multi-device access descriptors, export/import handles, and protection-mode matrices), and remaining advanced memory-pool operations (pool export/import handles and multi-device access descriptors). Host allocation/registration, pitched allocation with 2D copies, basic array allocation with 2D array copies, 3D pitched allocation with host-device 3D copies, 3D array allocation with 3D copy-to/from-array, managed allocation with prefetch, default memory pools with stream-ordered allocation, basic virtual memory management (granularity, address reserve/free, allocation handle create/release, map/unmap, and single-device access), explicit memory-pool lifecycle (create/destroy, release-threshold, trim, and pool-specific async allocation), and current-device memory-pool access control are now covered.
2. Texture and surface APIs beyond current basics: texture/surface object create/destroy with resource and texture descriptor round-trips and channel-descriptor queries are now covered; texture reference APIs, mipmapped arrays, and bound/linear texture variants remain.
3. Module, library, and code-loading APIs: HIPRTC-backed module load-from-data, load-from-data-with-options (`hipModuleLoadDataEx`, including JIT option handling), unload, function and global lookup, and module kernel launch (with `hipFuncGetAttribute`) are now covered; the HIPRTC-backed library-loading family is now covered too (library load-from-data, unload, kernel lookup, kernel count, kernel enumeration, global lookup, and the `hipKernel*` accessors for function/library/name). Module load from file/fat-binary, tex-ref and function-count queries, cooperative module launches, and module occupancy helpers remain.
4. Context and driver-style APIs beyond current basics: device-handle, name, compute-capability, total-memory, UUID, and PCI bus-id queries, primary-context retain/get-state/release, current-context/device queries, device cache-config get/set, shared-memory-config query, device-limit get/set, device-flag query, and stream-priority-range query are now covered; context create/destroy, push/pop/set-current, shared-memory config setter, and context peer access remain.
5. Advanced graph APIs: node type queries, explicit add/remove dependencies, child graph nodes with sub-graph retrieval, host nodes with param round-trips, node find in clone, memory alloc/free nodes with param round-trips plus device graph-memory attribute and trim helpers, user objects (create/retain/release and graph retain/release), per-node enable/disable state (set/get), kernel-node attribute set/get round-trips (cooperative and access-policy-window) with invalid-input rejection, and executable-graph update paths (whole-graph `hipGraphExecUpdate` with topology-change reporting plus per-node exec setters for host, child-graph, event-record, and event-wait nodes) are now covered; remaining node attribute variants and debug dot export remain.
6. IPC memory and event handle round-trips (get/open/close for memory, get/open for events) are now covered; peer access and multigpu APIs remain.
7. Extension and proc-address APIs beyond current basics: dynamic API-name lookup, API-name-to-string mapping, per-stream device-id queries, and thread-local extended error state are now covered; external memory/semaphore import/export, extended kernel launch and CU-mask stream variants, logging controls, and link-type queries remain.

## Update procedure

When adding or removing contract tests:

1. Recompute the contract-domain test counts from `catch/contract/*/hip_*_contract.cc`.
2. Recompute directly exercised `hip*` APIs from the contract sources.
3. Recompute the public API denominator from `projects/hip/include/hip/hip_runtime_api.h`.
4. Update this document in the same branch as the contract-test change.
5. Keep percentages labeled as approximate API-name coverage, not behavioral coverage.
