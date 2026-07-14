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
