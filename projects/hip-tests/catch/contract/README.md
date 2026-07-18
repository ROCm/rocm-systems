# HIP Semantic Contract Tests

This directory contains public HIP API semantic contract tests.

Contract tests validate externally observable HIP behavior through public HIP APIs only. They are small, deterministic, device-required tests intended to fail like semantic contract violations, not stress, performance, platform-policy, or customer-scenario tests.

The first domains are:

- `memory`: allocation and free contracts
- `transfer`: synchronous copy contracts
- `driver_memcpy`: driver-style directed 1D synchronous and stream-ordered copy contracts
- `driver_memcpy_2d`: driver-style struct-based 2D synchronous and stream-ordered copy contracts
- `runtime`: runtime initialization, device visibility, version, and error-state contracts
- `device`: portable current-device property contracts
- `device_identity`: device PCI identity, device selection, and single-device peer-query contracts
- `device_texture_query`: image-gated device texture-width query contracts
- `peer_query`: peer-to-peer attribute and AMD link-type query invalid-input contracts
- `peer_access`: peer-access enable/disable invalid-input and lifecycle contracts
- `peer_copy`: single-device self-peer 1D/3D copy and invalid-device peer-copy contracts
- `stream_event`: stream and event lifecycle, query, synchronization, and wait-event ordering contracts
- `stream_props`: stream creation flag, priority, identity, device, and event timing contracts
- `stream_callbacks`: stream callback and host-function ordering contracts
- `stream_attributes`: stream attribute set, get, copy, and invalid-input contracts
- `stream_memory_ops`: stream wait/write value and batch memory operation contracts
- `stream_cu_mask`: AMD stream CU-mask create and query contracts
- `stream_attach`: managed-memory stream attach contracts
- `async_transfer`: async copy visibility and invalid-kind consistency contracts
- `mem_batch_copy`: batch async device-to-device copy round-trip, per-copy attribute, and invalid-input contracts
- `mem_batch_copy_3d`: batch async 3D pointer copy host-device-host round-trip and invalid-input contracts
- `mem_batch_discard`: batch async memory discard and discard-and-prefetch accepted-or-unsupported and invalid-input contracts
- `memset`: byte, word, dword, and async memset contracts
- `driver_memset_async`: driver-style directed async byte, word, and dword memset contracts
- `driver_memset_2d`: driver-style directed 2D byte, word, and dword memset contracts
- `driver_memset_async_2d3d`: driver-style async 2D and 3D memset contracts
- `error_api`: error name and string API contracts without backend-specific text assumptions
- `driver_error`: driver-style error name and string API contracts
- `kernel`: tiny in-source kernel launch contracts
- `call_config`: legacy configure/setup/launch-by-pointer call-configuration contracts
- `func_attributes`: runtime function attribute query, symbol, and configuration contracts
- `kernel_launch`: cooperative, extended, symbol, and AMD extension kernel launch contracts
- `multi_device_launch`: discrete-GPU-gated cooperative, extended, and module-based multi-device kernel launch contracts
- `driver_launch_ex`: discrete-GPU-gated driver-style extended kernel launch and SM-resource group-split contracts
- `kernel_name_ref`: AMD kernel-name reflection contracts by function handle and host function pointer
- `graphics_interop`: graphics-interop resource map, unmap, unregister, and mapped-pointer/array query invalid-input rejection contracts
- `symbol_copy`: device symbol copy, offset, async, and invalid-symbol contracts
- `graph`: graph lifecycle plus simple memcpy and memset node contracts
- `occupancy`: portable occupancy query contracts for a tiny in-source kernel
- `occupancy_ext`: occupancy with-flags and cluster query contracts
- `occupancy_variable`: potential-block-size with-flags and variable-shared-memory occupancy contracts
- `graph_capture`: stream capture lifecycle and captured memcpy graph contracts
- `stream_capture_mode`: stream capture mode exchange and v2 capture-info contracts
- `capture_to_graph`: capture-into-provided-graph and capture-dependency-update contracts
- `graph_kernel`: graph kernel node contracts with tiny in-source kernels
- `graph_event`: graph event record and wait node contracts
- `graph_topology`: graph node, root, edge, dependency, and dependent introspection contracts
- `graph_clone`: graph clone lifecycle and cloned memcpy graph contracts
- `graph_update`: graph exec whole-graph update and memcpy, memset, host, child graph, and event node parameter update contracts
- `graph_exec_lifecycle`: executable graph instantiate-with-flags, upload, and flag query contracts
- `graph_node_types`: graph node type introspection and dependency edit contracts
- `graph_child`: child graph node creation, introspection, and embedded execution contracts
- `graph_host`: host graph node callback, parameter, and node-type contracts
- `graph_mem_nodes`: graph memory allocation/free node and graph memory attribute contracts
- `graph_generic_node`: unified generic graph node add and pre/post-instantiation parameter setter contracts
- `graph_memcpy3d_node`: struct-based 3D memcpy graph node add, parameter round-trip, and pre/post-instantiation setter contracts
- `graph_batch_mem_op`: batch memory operation graph node add, write-value launch, parameter round-trip, and executable setter contracts
- `driver_graph_node`: AMD-gated driver-style context-bound 3D memcpy and memset graph node add, parameter round-trip, and executable setter contracts
- `graph_instantiate_params`: params-struct graph instantiation result and upload-stream launch contracts
- `graph_debug`: graph dot-export file-output contracts
- `graph_node_find`: graph node lookup in cloned graph contracts
- `graph_user_objects`: graph user object create, retain, release, and graph lifetime contracts
- `graph_node_enabled`: executable graph node enable/disable query and behavior contracts
- `graph_node_attributes`: graph kernel node attribute set/get and invalid-input contracts
- `graph_node_params`: graph memcpy, memset, event, and lifecycle node parameter contracts
- `graph_node_setters`: pre-instantiation graph node parameter setter and attribute-copy contracts
- `graph_symbol_copy_nodes`: graph memcpy-to/from-symbol node add, set-params, and exec-set-params contracts
- `host_memory`: host allocation, registration, device-pointer, and flag contracts
- `host_alloc_aliases`: alternate host allocation/free entry point and extended allocation contracts
- `pitched_memory`: pitched allocation and host/device 2D copy contracts
- `driver_pitched_memory`: driver-style pitched allocation, 2D copy, and unaligned 2D copy contracts
- `array_memory`: HIP array allocation and 2D array copy contracts
- `array_copy`: legacy and async HIP array copy contracts
- `array_copy_ext`: remaining driver-style and async HIP array copy contracts
- `driver_array`: driver-style HIP array creation, destruction, and descriptor contracts
- `mipmapped_array`: image-gated mipmapped array allocation, level retrieval, and memory-requirement query contracts
- `managed_memory`: managed allocation, visibility, free, and prefetch contracts
- `mem_advise`: managed-memory advice and range-attribute contracts
- `mem_advise_v2`: location-based managed-memory advise and prefetch contracts
- `pointer_info`: pointer attribute, address-range, and memory-capacity query contracts
- `pointer_query`: driver-style pointer attribute, pointer-attribute set, and allocation-size query contracts
- `memory_pool`: default memory pool, release-threshold, and stream-ordered allocation contracts
- `vmm`: virtual memory management granularity, reserve, map, access, and roundtrip contracts
- `vmm_handle`: VMM allocation-handle retain, property query, and dma-buf address-range export contracts
- `copy3d`: 3D pitched allocation and host-device 3D copy contracts
- `async_copy3d`: stream-ordered 3D copy, 3D memset, and memcpy-with-stream contracts
- `driver_copy3d`: driver-style 3D copy contracts
- `array3d`: 3D array allocation and 3D copy-to/from-array contracts
- `texture`: texture and surface object creation and descriptor-introspection contracts
- `driver_texture_object`: driver-style texture object creation and descriptor-introspection contracts
- `texture_reference`: image-gated deprecated texture-reference scalar set/get contracts (address mode, filter mode, flags, format, max anisotropy)
- `texture_reference_symbol`: image-gated symbol-backed texture-reference bind, module-texref address/array round-trip, and deprecated-stub mipmap contracts
- `context`: driver-style device and context query contracts
- `context_mutation`: driver-style context create, set-current, push/pop, synchronize, and API-version contracts
- `context_config`: driver-style context cache/shared-memory config, flags, and peer-access contracts
- `device_config`: device configuration query, limit, flag, and stream-priority contracts
- `device_lifecycle`: device flag, shared-memory-config, valid-device selection, and primary-context flag/reset lifecycle contracts
- `green_context`: device/stream SM resource query, SM split, and green execution context creation, stream, and event contracts
- `memory_pool_lifecycle`: explicit memory pool lifecycle, release-threshold, trim, and pool-specific async allocation contracts
- `memory_pool_access`: current-device memory pool access-control contracts
- `mem_location_pool`: location-based memory pool set/get and access-query contracts
- `mempool_shareable_handle`: shareable memory-pool handle export/import and pointer export/import contracts
- `extension`: proc-address resolution and AMD extension API contracts (`hipGetProcAddress`, `hipApiName`, `hipGetStreamDeviceId`, `hipExtGetLastError`)
- `logging`: AMD-gated extended logging enable/disable and parameter-configuration contracts
- `profiler`: deprecated profiler start/stop accepted-or-unsupported contracts
- `driver_entry_point`: driver entry-point symbol resolution contracts
- `ipc`: capability-gated same-process IPC memory and event handle contracts
- `external_resource`: external memory and semaphore import, mapping, destroy, and signal/wait invalid-input rejection contracts
- `module`: HIPRTC-backed module load, function, global, launch, and function-attribute contracts
- `module_load_ex`: HIPRTC-backed module load-data-with-options contracts
- `module_load_file`: module load-from-file and fat-binary invalid-input contracts
- `module_exec`: HIPRTC-backed module function-count, occupancy, and cooperative launch contracts
- `jit_link`: AMD-gated JIT linker lifecycle and invalid-input contracts
- `library`: AMD-gated HIPRTC-backed library load, kernel, global, and kernel-object contracts
- `kernel_object_attributes`: AMD-gated HIPRTC-backed hipKernel_t attribute and parameter-info contracts
- `library_file`: AMD-gated HIPRTC-backed library load-from-file and managed-symbol contracts

Some domains are capability-gated. For example, `array_memory` skips on devices without image/array support, and `pitched_memory` skips on runtime paths where `hipMallocPitch` reports out of memory for tiny allocations. The AMD-specific extension contracts in `extension` are compiled only on the AMD backend, while the portable `hipGetProcAddress` contracts run on both backends. The `library` domain is likewise compiled only on the AMD backend, since the `hipLibrary*`/`hipKernel*` object APIs are AMD-side in this tree. These gates indicate an unsupported local capability, not a contract failure.

Run the layer with:

```bash
cmake --build <build-dir> --target contract_tests
ctest --test-dir <build-dir> -L contract --output-on-failure
```

## Backend coverage

The contract suite is portable across the AMD (ROCm/HIP) backend and the NVIDIA
HIP-over-CUDA backend (`HIP_PLATFORM=nvidia`). All 115 domains compile on both
backends. Domains that exercise AMD-only or CUDA-removed entry points are gated
with `#if HT_AMD`, so on NVIDIA they compile to an empty binary (no registered
test cases) rather than failing to build.

### Finding backend divergences (`BACKEND-DIFF`)

Every place a test accommodates a real behavioral or API-availability difference
between the AMD and NVIDIA backends is tagged with a `BACKEND-DIFF:` comment
marker. Grep for it to review or revisit these sites:

```bash
grep -rn "BACKEND-DIFF" projects/hip-tests/catch/contract
```

Each marked comment explains the difference and notes what parity would require,
so the markers double as a worklist for reconciling behavior if a backend closes
the gap. The marker covers whole-file and per-test `#if HT_AMD` gates, divergent
`#if HT_AMD ... #else ... #endif` assertions, and CUDA-version availability
gates. Purely mechanical portability shims are intentionally not marked (they
carry ordinary explanatory comments instead): the `hipFree(0)` primary-context
primes, the HIPRTC `--offload-arch` vs `--fmad=false` compile-option branches,
and the `hipDeviceptr_t` pointer/integer type-conversion helpers.

| | AMD (gfx908, ROCm 7.15) | NVIDIA (H100, CUDA 13.1) |
|---|---|---|
| Domains | 115 | 115 |
| Domains compiled (non-empty) | 115 | 106 |
| Domains compile-gated (empty) | 0 | 9 |
| Registered test cases (ctest) | 578 | 515 |
| Passed | 578 | 494 |
| Failed | 0 | 0 |
| Skipped | 0 | 21 |

Counts are from `ctest -L contract` (one registered test per `HIP_TEST_CASE`).
The AMD figures are for a single-GPU MI100; on AMD, capability-gated cases that
have nothing to exercise (for example single-GPU peer access) resolve as passing
skips rather than being reported separately by ctest.

### Compilation and execution time

Representative wall-clock timings for a full clean build and a full serial
`ctest -L contract` run. Both were measured on 16-core allocations.

| Stage | AMD (MI100 host, gfx908) | NVIDIA (H100 host, sm_90) |
|---|---|---|
| Clean compile (`contract_tests`, `-j16`) | ~466 s | ~282 s |
| Execution (`ctest -L contract`, serial) | ~65 s | ~225 s |

Notes:
- The AMD build here is a multi-architecture fat binary
  (gfx906/gfx908/gfx90a/gfx942/gfx950/gfx1100/gfx1201), which dominates its
  longer compile time; a single-architecture AMD build compiles substantially
  faster. The NVIDIA build targets a single architecture (`sm_90`).
- NVIDIA execution is slower largely because several driver-API-first tests run
  under one-process-per-test isolation and pay CUDA primary-context
  initialization on each launch; the AMD runtime auto-initializes and the local
  datacenter GPU has lower per-launch overhead.

### Domains compile-gated out on NVIDIA (empty binary)

These 9 domains are AMD-only (or use APIs removed from recent CUDA) and are
wrapped in a whole-file guard, so they register no test cases on NVIDIA. Most use
`#if HT_AMD`; `texture_reference_symbol` uses
`#if HT_AMD || (HT_NVIDIA && defined(CUDA_VERSION) && CUDA_VERSION < CUDA_12000)`
because the legacy texref API it exercises existed on NVIDIA only through CUDA 11:

| Domain | Reason absent on NVIDIA |
|---|---|
| `call_config` | `hipConfigureCall`/`hipSetupArgument`/`hipLaunchByPtr` legacy launch — AMD only |
| `kernel_name_ref` | `hipKernelNameRef`/`hipKernelNameRefByPtr` name reflection — AMD only |
| `texture_reference` | deprecated `hipTexRef*` driver texref API — removed in CUDA 12+ |
| `texture_reference_symbol` | symbol-backed `hipTexRef*` module texref API — removed in CUDA 12+ |
| `multi_device_launch` | `hipExtLaunchMultiKernelMultiDevice` + incomplete `cudaLaunchParams` |
| `occupancy_variable` | `hipOccupancyMaxPotentialBlockSizeVariableSMem` family — AMD only |
| `stream_cu_mask` | AMD stream CU-mask API — AMD only |
| `jit_link` | `hipLink*` / `hipJitInputSpirv` JIT linker — AMD only |
| `logging` | AMD extended logging control API — AMD only |

In addition, several domains that do compile on NVIDIA gate only their AMD-only
tests behind `#if HT_AMD` (for example `library`, `kernel_object_attributes`,
`library_file`, `driver_graph_node`, `host_alloc_aliases`, `pointer_query`,
`device_config`, `occupancy_ext`, `driver_texture_object`, `texture`), so their
portable subset still runs on NVIDIA.

### Tests skipped at execution on NVIDIA

The 21 runtime skips fall into two groups.

NVIDIA-backend-specific safety gates — the AMD backend validates the argument and
returns an error, but the CUDA entry point these map to dereferences the bad
argument and would fault (SIGSEGV), so the negative/rejection contract is only
exercised on AMD:

| Test | API | Reason |
|---|---|---|
| `Contract_Extension_GetProcAddress_NullArgs_AreRejected` | `hipGetProcAddress`→`cuGetProcAddress` | null args not validated (fault) |
| `Contract_DriverEntryPoint_NullFuncPtr_IsRejected` | `hipGetDriverEntryPoint` | null output not validated |
| `Contract_PointerInfo_GetAttributes_NullOutput_IsRejected` | `hipPointerGetAttributes`→`cudaPointerGetAttributes` | null output not validated (fault) |
| `Contract_StreamAttributes_GetAttribute_RejectsInvalidInputs` | `hipStreamGetAttribute`→`cudaStreamGetAttribute` | null value-out not validated (fault) |
| `Contract_ExternalResource_SignalSemaphore_NullHandle_IsRejected` | `hipSignalExternalSemaphoresAsync` | null handle not validated (fault) |
| `Contract_Ipc_GetMemHandle_NullArgs_AreRejected` | `hipIpcGetMemHandle`→`cudaIpcGetMemHandle` | null args not validated (fault) |
| `Contract_Ipc_MemHandle_SameProcessRoundTrip` | `hipIpcOpenMemHandle` | same-process IPC open unsupported on NVIDIA |
| `Contract_Ipc_EventHandle_SameProcessRoundTrip` | `hipIpcOpenEventHandle` | same-process IPC open unsupported on NVIDIA |

Universal capability skips — these depend on device/runtime capability and skip
on any host lacking it (they also skip on a single-GPU or non-XNACK AMD host):

| Test(s) | Capability required |
|---|---|
| `Contract_PeerAccess_EnableTwice_ThenDisable_RoundTripsWhenAvailable` | two or more GPUs |
| `Contract_DriverLaunchEx_DevSmResourceSplit_ProducesBoundedGroup` | SM resource splitting |
| `Contract_VmmHandle_GetHandleForAddressRange_DmaBufFd_IsQueryableWhenSupported` | dma-buf handle export |
| `Contract_StreamMemoryOps_*` (6: WriteValue32/64, WaitValueGte, WaitValue64Gte, BatchMemOp, RejectsInvalidInputs) | stream wait/write value ops |
| `Contract_GraphBatchMemOp_*` (4: AddNode, SetParams, ExecSetParams, GetParams) | stream wait/write value ops in graph nodes |
