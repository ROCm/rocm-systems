# HIP Contract Test Coverage Tracker

This document tracks the public HIP API semantic contract coverage in `projects/hip-tests/catch/contract`.

The percentages below are approximate API-name coverage against declarations parsed from `projects/hip/include/hip/hip_runtime_api.h`. They are not behavioral coverage: one API can have many modes and edge cases, and the contract layer intentionally covers only small, portable semantic guarantees.

## Snapshot

- Contract tests: 470
- Declared HIP runtime APIs parsed from `hip_runtime_api.h`: 494
- Declared HIP runtime APIs directly exercised by contract tests: 354
- Approximate declared API-name coverage: 71.7%
- Additional public macro exercised: `hipLaunchKernelGGL`
- Additional non-runtime-header APIs exercised: HIPRTC (`hiprtcCreateProgram`, `hiprtcCompileProgram`, `hiprtcGetCodeSize`, `hiprtcGetCode`, `hiprtcGetProgramLogSize`, `hiprtcGetProgramLog`, `hiprtcDestroyProgram`); these are not declared in `hip_runtime_api.h` and are excluded from the coverage denominator and covered counts.

## Contract domains

| Contract domain | Tests |
|---|---:|
| `memory` | 5 |
| `transfer` | 4 |
| `driver_memcpy` | 6 |
| `driver_memcpy_2d` | 5 |
| `runtime` | 10 |
| `device` | 7 |
| `device_identity` | 7 |
| `device_texture_query` | 6 |
| `peer_query` | 5 |
| `peer_access` | 6 |
| `peer_copy` | 5 |
| `device_config` | 6 |
| `stream_event` | 7 |
| `stream_props` | 7 |
| `stream_callbacks` | 5 |
| `stream_attributes` | 5 |
| `stream_memory_ops` | 6 |
| `stream_cu_mask` | 4 |
| `stream_attach` | 5 |
| `async_transfer` | 4 |
| `memset` | 6 |
| `driver_memset_async` | 6 |
| `driver_memset_2d` | 6 |
| `driver_memset_async_2d3d` | 5 |
| `error_api` | 6 |
| `driver_error` | 6 |
| `extension` | 6 |
| `driver_entry_point` | 5 |
| `kernel` | 4 |
| `call_config` | 2 |
| `kernel_launch` | 7 |
| `func_attributes` | 8 |
| `graph` | 5 |
| `occupancy` | 3 |
| `occupancy_ext` | 4 |
| `graph_capture` | 4 |
| `stream_capture_mode` | 6 |
| `graph_kernel` | 3 |
| `graph_event` | 3 |
| `graph_topology` | 5 |
| `graph_clone` | 3 |
| `graph_update` | 9 |
| `graph_exec_lifecycle` | 5 |
| `graph_node_types` | 5 |
| `graph_child` | 3 |
| `graph_host` | 3 |
| `graph_mem_nodes` | 4 |
| `graph_node_attributes` | 3 |
| `graph_node_params` | 7 |
| `graph_node_setters` | 5 |
| `graph_symbol_copy_nodes` | 3 |
| `graph_node_find` | 3 |
| `graph_user_objects` | 4 |
| `graph_node_enabled` | 3 |
| `host_memory` | 5 |
| `host_alloc_aliases` | 7 |
| `pitched_memory` | 4 |
| `driver_pitched_memory` | 5 |
| `array_memory` | 4 |
| `array_copy` | 5 |
| `array_copy_ext` | 7 |
| `driver_array` | 6 |
| `managed_memory` | 5 |
| `mem_advise` | 6 |
| `pointer_info` | 7 |
| `pointer_query` | 6 |
| `memory_pool` | 6 |
| `memory_pool_lifecycle` | 4 |
| `memory_pool_access` | 3 |
| `vmm` | 5 |
| `copy3d` | 4 |
| `async_copy3d` | 6 |
| `driver_copy3d` | 5 |
| `array3d` | 3 |
| `texture` | 7 |
| `driver_texture_object` | 6 |
| `context` | 6 |
| `context_mutation` | 5 |
| `context_config` | 6 |
| `ipc` | 5 |
| `module` | 7 |
| `module_exec` | 8 |
| `module_load_ex` | 4 |
| `module_load_file` | 6 |
| `jit_link` | 6 |
| `library` | 14 |
| `kernel_object_attributes` | 3 |
| `symbol_copy` | 6 |

## Coverage by API category

| Category | Covered | Total parsed | Approx. coverage |
|---|---:|---:|---:|
| Error handling | 5 | 5 | 100.0% |
| Event | 8 | 8 | 100.0% |
| Occupancy | 13 | 13 | 100.0% |
| Graph / capture | 77 | 96 | 80.2% |
| Stream | 23 | 23 | 100.0% |
| Runtime / device | 34 | 45 | 75.6% |
| Kernel launch / function attrs | 15 | 19 | 78.9% |
| Memory / copy / memset | 111 | 137 | 81.0% |
| Other runtime APIs | 1 | 56 | 1.8% |
| Module / library loading | 27 | 29 | 93.1% |
| Texture / surface | 13 | 44 | 29.5% |
| Context / driver | 16 | 16 | 100.0% |
| Extension / proc address | 6 | 13 | 46.2% |
| IPC | 5 | 5 | 100.0% |

## Covered APIs

### Memory / copy / memset

```text
hipMalloc
hipFree
hipMemcpy
hipMemcpyAsync
hipMemcpyWithStream
hipMemcpyHtoD
hipMemcpyDtoH
hipMemcpyDtoD
hipMemcpyHtoDAsync
hipMemcpyDtoHAsync
hipMemcpyDtoDAsync
hipMemset
hipMemsetAsync
hipMemsetD8
hipMemsetD16
hipMemsetD32
hipMemsetD8Async
hipMemsetD16Async
hipMemsetD32Async
hipMemset2D
hipMemsetD2D8
hipMemsetD2D16
hipMemsetD2D32
hipMemsetD2D8Async
hipMemsetD2D16Async
hipMemsetD2D32Async
hipMemset2DAsync
hipMemset3D
hipMemset3DAsync
hipHostMalloc
hipHostAlloc
hipMallocHost
hipMemAllocHost
hipHostFree
hipFreeHost
hipHostRegister
hipHostUnregister
hipHostGetDevicePointer
hipHostGetFlags
hipMallocPitch
hipMemAllocPitch
hipMemcpy2D
hipMemcpy2DAsync
hipMemcpyParam2D
hipMemcpyParam2DAsync
hipMalloc3D
hipMemcpy3D
hipMemcpy3DAsync
hipDrvMemcpy3D
hipDrvMemcpy3DAsync
hipMallocArray
hipFreeArray
hipMemcpy2DToArray
hipMemcpy2DFromArray
hipMemcpyToArray
hipMemcpyFromArray
hipMemcpyHtoA
hipMemcpyAtoH
hipMemcpyDtoA
hipMemcpyAtoD
hipMemcpyHtoAAsync
hipMemcpyAtoHAsync
hipMemcpy2DArrayToArray
hipMemcpy2DToArrayAsync
hipMemcpy2DFromArrayAsync
hipMalloc3DArray
hipArrayGetInfo
hipArrayCreate
hipArrayDestroy
hipArrayGetDescriptor
hipArray3DCreate
hipArray3DGetDescriptor
hipMallocManaged
hipMemPrefetchAsync
hipMemAdvise
hipMemRangeGetAttribute
hipMemRangeGetAttributes
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
hipExtMallocWithFlags
hipPointerGetAttributes
hipPointerGetAttribute
hipDrvPointerGetAttributes
hipPointerSetAttribute
hipMemGetAddressRange
hipMemGetInfo
hipMemPtrGetInfo
hipMemcpyToSymbol
hipMemcpyFromSymbol
hipMemcpyToSymbolAsync
hipMemcpyFromSymbolAsync
hipMemcpyPeer
hipMemcpyPeerAsync
hipMemcpy3DPeer
hipMemcpy3DPeerAsync
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
hipDeviceGetByPCIBusId
hipChooseDevice
hipDeviceCanAccessPeer
hipDeviceGetP2PAttribute
hipDeviceEnablePeerAccess
hipDeviceDisablePeerAccess
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
hipCtxCreate
hipCtxDestroy
hipCtxSetCurrent
hipCtxPushCurrent
hipCtxPopCurrent
hipCtxSynchronize
hipCtxGetApiVersion
hipCtxGetCacheConfig
hipCtxSetCacheConfig
hipCtxGetSharedMemConfig
hipCtxSetSharedMemConfig
hipCtxGetFlags
hipCtxEnablePeerAccess
hipCtxDisablePeerAccess
```

### Texture / surface

```text
hipCreateTextureObject
hipDestroyTextureObject
hipGetTextureObjectResourceDesc
hipGetTextureObjectTextureDesc
hipTexObjectCreate
hipTexObjectDestroy
hipTexObjectGetResourceDesc
hipTexObjectGetTextureDesc
hipTexObjectGetResourceViewDesc
hipDeviceGetTexture1DLinearMaxWidth
hipCreateSurfaceObject
hipDestroySurfaceObject
hipGetChannelDesc
```

### Error handling

```text
hipGetErrorName
hipGetErrorString
hipDrvGetErrorName
hipDrvGetErrorString
hipGetLastError
hipPeekAtLastError
```

### Stream / event

```text
hipStreamCreate
hipStreamCreateWithFlags
hipStreamCreateWithPriority
hipStreamDestroy
hipStreamSynchronize
hipStreamQuery
hipStreamWaitEvent
hipStreamGetFlags
hipStreamGetPriority
hipStreamGetDevice
hipStreamGetId
hipStreamAddCallback
hipStreamGetAttribute
hipStreamSetAttribute
hipStreamCopyAttributes
hipStreamWriteValue32
hipStreamWriteValue64
hipStreamWaitValue32
hipStreamWaitValue64
hipStreamBatchMemOp
hipStreamAttachMemAsync
hipExtStreamCreateWithCUMask
hipExtStreamGetCUMask
hipEventCreate
hipEventCreateWithFlags
hipEventDestroy
hipEventRecord
hipEventRecordWithFlags
hipEventSynchronize
hipEventQuery
hipEventElapsedTime
```

### Kernel launch / function attrs

```text
hipLaunchKernel
hipLaunchHostFunc
hipLaunchKernelGGL
hipLaunchCooperativeKernel
hipExtLaunchKernel
hipGetSymbolAddress
hipGetSymbolSize
hipFuncGetAttribute
hipFuncGetAttributes
hipFuncSetAttribute
hipFuncSetCacheConfig
hipFuncSetSharedMemConfig
hipGetFuncBySymbol
hipConfigureCall
hipSetupArgument
hipLaunchByPtr
```

### Module / library loading

```text
hipModuleLoad
hipModuleLoadFatBinary
hipModuleLoadData
hipModuleLoadDataEx
hipModuleUnload
hipModuleGetFunction
hipModuleGetFunctionCount
hipModuleGetGlobal
hipModuleLaunchKernel
hipModuleLaunchCooperativeKernel
hipLinkCreate
hipLinkAddData
hipLinkAddFile
hipLinkComplete
hipLinkDestroy
hipLibraryLoadData
hipLibraryUnload
hipLibraryGetKernel
hipLibraryGetKernelCount
hipLibraryEnumerateKernels
hipLibraryGetGlobal
hipKernelGetFunction
hipKernelGetLibrary
hipKernelGetName
hipKernelGetAttribute
hipKernelSetAttribute
hipKernelGetParamInfo
```

### Graph / capture / graph nodes

```text
hipGraphCreate
hipGraphDestroy
hipGraphInstantiate
hipGraphInstantiateWithFlags
hipGraphUpload
hipGraphLaunch
hipGraphExecGetFlags
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
hipGraphMemcpyNodeGetParams
hipGraphMemcpyNodeSetParams
hipGraphMemsetNodeGetParams
hipGraphMemsetNodeSetParams
hipGraphEventRecordNodeGetEvent
hipGraphEventWaitNodeGetEvent
hipGraphDestroyNode
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
hipStreamGetCaptureInfo_v2
hipThreadExchangeStreamCaptureMode
hipUserObjectCreate
hipUserObjectRetain
hipUserObjectRelease
hipGraphRetainUserObject
hipGraphReleaseUserObject
hipGraphNodeSetEnabled
hipGraphNodeGetEnabled
hipGraphKernelNodeSetAttribute
hipGraphKernelNodeGetAttribute
hipGraphKernelNodeSetParams
hipGraphHostNodeSetParams
hipGraphMemcpyNodeSetParams1D
hipGraphEventRecordNodeSetEvent
hipGraphEventWaitNodeSetEvent
hipGraphKernelNodeCopyAttributes
hipGraphAddMemcpyNodeToSymbol
hipGraphAddMemcpyNodeFromSymbol
hipGraphMemcpyNodeSetParamsToSymbol
hipGraphMemcpyNodeSetParamsFromSymbol
hipGraphExecMemcpyNodeSetParamsToSymbol
hipGraphExecMemcpyNodeSetParamsFromSymbol
```

### Occupancy

```text
hipOccupancyMaxActiveBlocksPerMultiprocessor
hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
hipOccupancyMaxPotentialBlockSize
hipOccupancyAvailableDynamicSMemPerBlock
hipOccupancyMaxActiveClusters
hipOccupancyMaxPotentialClusterSize
hipModuleOccupancyMaxPotentialBlockSize
hipModuleOccupancyMaxPotentialBlockSizeWithFlags
hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
hipOccupancyMaxPotentialBlockSizeWithFlags
hipOccupancyMaxPotentialBlockSizeVariableSMem
hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags
```

### Extension / proc address

```text
hipGetProcAddress
hipGetDriverEntryPoint
hipApiName
hipGetStreamDeviceId
hipExtGetLastError
hipExtGetLinkTypeAndHopCount
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

1. Memory surface beyond current basics: advanced VMM operations (multi-device access descriptors, export/import handles, and protection-mode matrices), and remaining advanced memory-pool operations (pool export/import handles and multi-device access descriptors). Host allocation/registration and alternate host/device allocation entry points (`hipHostAlloc`, `hipMallocHost`, `hipMemAllocHost`, `hipFreeHost`, `hipExtMallocWithFlags`), directed driver-style 1D and struct-based 2D copies (`hipMemcpyHtoD`, `hipMemcpyDtoH`, `hipMemcpyDtoD`, `hipMemcpyParam2D`, plus async variants), directed async byte/word/dword memset (`hipMemsetD8Async`, `hipMemsetD16Async`, `hipMemsetD32Async`), directed 2D/3D memset (`hipMemset2D`, `hipMemset2DAsync`, `hipMemset3D`, `hipMemset3DAsync`, `hipMemsetD2D8`, `hipMemsetD2D16`, `hipMemsetD2D32`, and D2D async variants), runtime and driver-style pitched allocation with 2D copies, basic array allocation with 2D array copies plus legacy, driver-style, and async array-copy entry points (`hipMemcpyToArray`, `hipMemcpyFromArray`, `hipMemcpyHtoA`, `hipMemcpyAtoH`, `hipMemcpyDtoA`, `hipMemcpyAtoD`, `hipMemcpyHtoAAsync`, `hipMemcpyAtoHAsync`, `hipMemcpy2DArrayToArray`, `hipMemcpy2DToArrayAsync`, `hipMemcpy2DFromArrayAsync`), 3D pitched allocation with runtime and driver-style host-device 3D copies, 3D array allocation with 3D copy-to/from-array, driver-style array create/destroy and descriptor queries (`hipArrayCreate`, `hipArrayDestroy`, `hipArrayGetDescriptor`, `hipArray3DCreate`, `hipArray3DGetDescriptor`), managed allocation with prefetch, managed-memory advice and range attribute queries (set/unset read-mostly, preferred-location, and accessed-by advice via `hipMemAdvise`, plus single-attribute and multi-attribute range queries via `hipMemRangeGetAttribute` and `hipMemRangeGetAttributes`), default memory pools with stream-ordered allocation, basic virtual memory management (granularity, address reserve/free, allocation handle create/release, map/unmap, and single-device access), explicit memory-pool lifecycle (create/destroy, release-threshold, trim, and pool-specific async allocation), current-device memory-pool access control, pointer and memory-capacity queries (`hipPointerGetAttributes`, `hipPointerGetAttribute`, `hipDrvPointerGetAttributes`, `hipPointerSetAttribute`, `hipMemGetAddressRange`, `hipMemGetInfo`, `hipMemPtrGetInfo`), and device-global symbol copies (synchronous and stream-ordered copy-to/from-symbol with byte-offset placement and invalid/out-of-bounds rejection: `hipMemcpyToSymbol`, `hipMemcpyFromSymbol`, `hipMemcpyToSymbolAsync`, `hipMemcpyFromSymbolAsync`) are now covered. Single-device peer copies are now covered too: synchronous and async 1D peer copy plus 3D peer copy (`hipMemcpyPeer`, `hipMemcpyPeerAsync`, `hipMemcpy3DPeer`, `hipMemcpy3DPeerAsync`), exercised via self-device (dst==src) round-trips that behave like ordinary device-to-device copies, plus an invalid-device-ordinal rejection; cross-device peer transfers require multi-GPU hosts and remain uncovered. Remaining symbol-copy gaps are the driver-style/2D memcpy families.
2. Texture and surface APIs beyond current basics: runtime and driver-style texture object create/destroy with resource, texture, and resource-view descriptor round-trips, image-gated device texture-width queries, surface object create/destroy, and channel-descriptor queries are now covered; texture reference APIs, mipmapped arrays, and bound/linear texture variants remain.
3. Module, library, and code-loading APIs: module load from file/fat-binary, HIPRTC-backed module load-from-data, load-from-data-with-options (`hipModuleLoadDataEx`, including JIT option handling), unload, function and global lookup, module kernel launch (with `hipFuncGetAttribute`), function-count queries, cooperative module launch, and module occupancy helpers (max potential block size and max active blocks per multiprocessor, including both with-flags variants), and JIT linker lifecycle and invalid-input paths are now covered; the HIPRTC-backed library-loading family is now covered too (library load-from-data, unload, kernel lookup, kernel count, kernel enumeration, global lookup, and the `hipKernel*` accessors for function/library/name), plus the `hipKernel_t` attribute accessors and parameter-info query (`hipKernelGetAttribute`, `hipKernelSetAttribute` accepted-or-unsupported, `hipKernelGetParamInfo` first-parameter layout). Tex-ref queries and multi-device cooperative module launch remain.
4. Context and driver-style APIs beyond current basics: device-handle, name, compute-capability, total-memory, UUID, and PCI bus-id queries, primary-context retain/get-state/release, current-context/device queries, device cache-config get/set, shared-memory-config query, device-limit get/set, device-flag query, and stream-priority-range query are now covered; driver-style context lifecycle and current-context mutation are now covered too (context create/destroy, set-current, push/pop-current, context synchronize, and API-version query). Driver-style context configuration and peer access are now covered as well: per-context cache-config get/set (`hipCtxGetCacheConfig`, `hipCtxSetCacheConfig`), shared-memory-config get/set (`hipCtxGetSharedMemConfig`, `hipCtxSetSharedMemConfig`), the context flag query (`hipCtxGetFlags`), and context peer access enable/disable (`hipCtxEnablePeerAccess`, `hipCtxDisablePeerAccess`). This brings the driver-style context (`hipCtx*`) API family to full name coverage. PCI bus-id-to-device round-trips, device selection (`hipChooseDevice`), single-device peer-access queries (`hipDeviceCanAccessPeer`), peer-attribute queries (`hipDeviceGetP2PAttribute`), and peer-access enable/disable contracts (`hipDeviceEnablePeerAccess`, `hipDeviceDisablePeerAccess`) are now covered too. Remaining driver-style device-management gaps are the primary-context reset/set-flags helpers.
5. Advanced graph APIs: node type queries, explicit add/remove dependencies, child graph nodes with sub-graph retrieval, host nodes with param round-trips, node find in clone, stream capture-mode exchange and v2 capture-info queries, memory alloc/free nodes with param round-trips plus device graph-memory attribute and trim helpers, user objects (create/retain/release and graph retain/release), per-node enable/disable state (set/get), kernel-node attribute set/get round-trips (cooperative and access-policy-window) with invalid-input rejection, graph node parameter get/set round-trips and event getter contracts, graph node destruction, executable-graph instantiate-with-flags/upload/flag query paths, and executable-graph update paths (whole-graph `hipGraphExecUpdate` with topology-change reporting plus per-node exec setters for host, child-graph, event-record, and event-wait nodes) are now covered; the pre-instantiation node parameter setters are covered too (`hipGraphKernelNodeSetParams`, `hipGraphHostNodeSetParams`, `hipGraphMemcpyNodeSetParams1D`, `hipGraphEventRecordNodeSetEvent`, `hipGraphEventWaitNodeSetEvent`, and kernel-node attribute propagation via `hipGraphKernelNodeCopyAttributes`), verified by set-then-get round-trips and launched observable effects. The device-global symbol-copy graph nodes are now covered as well (`hipGraphAddMemcpyNodeToSymbol`, `hipGraphAddMemcpyNodeFromSymbol`, their graph-node set-params variants `hipGraphMemcpyNodeSetParamsToSymbol`/`hipGraphMemcpyNodeSetParamsFromSymbol`, and the executable-graph set-params variants `hipGraphExecMemcpyNodeSetParamsToSymbol`/`hipGraphExecMemcpyNodeSetParamsFromSymbol`), verified by to/from-symbol round-trips through a device global with node and exec-node re-parameterization. Remaining node attribute variants and debug dot export remain.
6. IPC memory and event handle round-trips (get/open/close for memory, get/open for events) are now covered; peer access and multigpu APIs remain.
7. Extension and proc-address APIs beyond current basics: dynamic API-name lookup, driver-entry-point lookup, API-name-to-string mapping, per-stream device-id queries, and thread-local extended error state are now covered; external memory/semaphore import/export, extended kernel launch variants, and logging controls remain.
8. Occupancy APIs: the occupancy API family is now at full name coverage. Beyond the earlier basics — max-active-blocks-per-multiprocessor (with the module variants and their with-flags forms), the non-module with-flags variant (`hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`), max-potential-block-size, available-dynamic-shared-memory-per-block, and the cluster occupancy helpers (`hipOccupancyMaxActiveClusters`, `hipOccupancyMaxPotentialClusterSize`) — the non-module with-flags potential-block-size variant (`hipOccupancyMaxPotentialBlockSizeWithFlags`) and the variable-shared-memory potential-block-size helpers (`hipOccupancyMaxPotentialBlockSizeVariableSMem`, `hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags`) are now covered too, exercising the with-flags/plain equivalence and a zero-dynamic-shared-memory functor.
9. Stream and event APIs beyond current basics: stream creation with flags and priority, flag/priority/device/id property round-trips, event timing (`hipEventElapsedTime`, `hipEventRecordWithFlags`), host-side stream callbacks (`hipStreamAddCallback`), stream attribute get/set/copy (`hipStreamGetAttribute`, `hipStreamSetAttribute`, `hipStreamCopyAttributes`), and stream memory operations (32/64-bit write-value `hipStreamWriteValue32`/`64`, 32/64-bit wait-value `hipStreamWaitValue32`/`64`, and batch memory ops `hipStreamBatchMemOp`, all gated on `hipDeviceAttributeCanUseStreamWaitValue`) are now covered, bringing the event API family to full name coverage; CU-mask stream create/query variants and memory-attach (`hipStreamAttachMemAsync`) are now covered, bringing the stream API family to full name coverage; device-resource queries (`hipStreamGetDevResource`) and the capture-to-graph/update-dependencies variants remain.
10. Kernel launch and function-attribute APIs beyond current basics: the function-attribute family is now covered, including the struct-form attribute query (`hipFuncGetAttributes`), the scalar attribute query (`hipFuncGetAttribute`), per-function attribute and hint setters (`hipFuncSetAttribute`, `hipFuncSetCacheConfig`, `hipFuncSetSharedMemConfig`), and symbol-to-function resolution (`hipGetFuncBySymbol`); host-pointer kernel launch (`hipLaunchKernel`) and host-function launch (`hipLaunchHostFunc`) are covered too. Cooperative launch (`hipLaunchCooperativeKernel`), the AMD extended launch entry point (`hipExtLaunchKernel`), and the device-global symbol address/size queries (`hipGetSymbolAddress`, `hipGetSymbolSize`) are now covered as well. The extended-launch contract exercises the `hipLaunchKernelEx` C++ template wrapper, but that name is not declared in `hip_runtime_api.h` (the header declares the underlying `hipLaunchKernelExC` entry point), so it does not count toward name coverage under the header-parse method. The legacy call-configuration helpers (`hipConfigureCall`, `hipSetupArgument`, `hipLaunchByPtr`) are now covered too, via a configure/setup/launch-by-pointer round-trip that writes an expected value plus a repeated-staging independence check; the null-function-pointer negative is intentionally omitted because `hipLaunchByPtr(nullptr)` corrupts the runtime heap on this backend (returns an error but aborts at teardown with `free(): invalid pointer`). Remaining launch and symbol APIs include the multi-device cooperative launch (`hipLaunchCooperativeKernelMultiDevice`, `hipExtLaunchMultiKernelMultiDevice`) and the driver-style extended launch entry points (`hipLaunchKernelExC`, `hipDrvLaunchKernelEx`).

## Update procedure

When adding or removing contract tests:

1. Recompute the contract-domain test counts from `catch/contract/*/test_hip_*_contract.cc`.
2. Recompute directly exercised `hip*` APIs from the contract sources.
3. Recompute the public API denominator from `projects/hip/include/hip/hip_runtime_api.h`.
4. Update this document in the same branch as the contract-test change.
5. Keep percentages labeled as approximate API-name coverage, not behavioral coverage.
