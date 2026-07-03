# HIP Semantic Contract Tests

This directory contains public HIP API semantic contract tests.

Contract tests validate externally observable HIP behavior through public HIP APIs only. They are small, deterministic, device-required tests intended to fail like semantic contract violations, not stress, performance, platform-policy, or customer-scenario tests.

The first domains are:

- `memory`: allocation and free contracts
- `transfer`: synchronous copy contracts
- `runtime`: runtime initialization, device visibility, version, and error-state contracts
- `device`: portable current-device property contracts
- `stream_event`: stream and event lifecycle, query, synchronization, and wait-event ordering contracts
- `stream_props`: stream creation flag, priority, identity, device, and event timing contracts
- `async_transfer`: async copy visibility and invalid-kind consistency contracts
- `memset`: byte, word, dword, and async memset contracts
- `error_api`: error name and string API contracts without backend-specific text assumptions
- `kernel`: tiny in-source kernel launch contracts
- `graph`: graph lifecycle plus simple memcpy and memset node contracts
- `occupancy`: portable occupancy query contracts for a tiny in-source kernel
- `occupancy_ext`: occupancy with-flags and cluster query contracts
- `graph_capture`: stream capture lifecycle and captured memcpy graph contracts
- `graph_kernel`: graph kernel node contracts with tiny in-source kernels
- `graph_event`: graph event record and wait node contracts
- `graph_topology`: graph node, root, edge, dependency, and dependent introspection contracts
- `graph_clone`: graph clone lifecycle and cloned memcpy graph contracts
- `graph_update`: graph exec whole-graph update and memcpy, memset, host, child graph, and event node parameter update contracts
- `graph_node_types`: graph node type introspection and dependency edit contracts
- `graph_child`: child graph node creation, introspection, and embedded execution contracts
- `graph_host`: host graph node callback, parameter, and node-type contracts
- `graph_mem_nodes`: graph memory allocation/free node and graph memory attribute contracts
- `graph_node_find`: graph node lookup in cloned graph contracts
- `graph_user_objects`: graph user object create, retain, release, and graph lifetime contracts
- `graph_node_enabled`: executable graph node enable/disable query and behavior contracts
- `graph_node_attributes`: graph kernel node attribute set/get and invalid-input contracts
- `host_memory`: host allocation, registration, device-pointer, and flag contracts
- `pitched_memory`: pitched allocation and host/device 2D copy contracts
- `array_memory`: HIP array allocation and 2D array copy contracts
- `managed_memory`: managed allocation, visibility, free, and prefetch contracts
- `memory_pool`: default memory pool, release-threshold, and stream-ordered allocation contracts
- `vmm`: virtual memory management granularity, reserve, map, access, and roundtrip contracts
- `copy3d`: 3D pitched allocation and host-device 3D copy contracts
- `array3d`: 3D array allocation and 3D copy-to/from-array contracts
- `texture`: texture and surface object creation and descriptor-introspection contracts
- `context`: driver-style device and context query contracts
- `device_config`: device configuration query, limit, flag, and stream-priority contracts
- `memory_pool_lifecycle`: explicit memory pool lifecycle, release-threshold, trim, and pool-specific async allocation contracts
- `memory_pool_access`: current-device memory pool access-control contracts
- `extension`: proc-address resolution and AMD extension API contracts (`hipGetProcAddress`, `hipApiName`, `hipGetStreamDeviceId`, `hipExtGetLastError`)
- `ipc`: capability-gated same-process IPC memory and event handle contracts
- `module`: HIPRTC-backed module load, function, global, launch, and function-attribute contracts
- `module_load_ex`: HIPRTC-backed module load-data-with-options contracts
- `module_exec`: HIPRTC-backed module function-count, occupancy, and cooperative launch contracts
- `library`: AMD-gated HIPRTC-backed library load, kernel, global, and kernel-object contracts

Some domains are capability-gated. For example, `array_memory` skips on devices without image/array support, and `pitched_memory` skips on runtime paths where `hipMallocPitch` reports out of memory for tiny allocations. The AMD-specific extension contracts in `extension` are compiled only on the AMD backend, while the portable `hipGetProcAddress` contracts run on both backends. The `library` domain is likewise compiled only on the AMD backend, since the `hipLibrary*`/`hipKernel*` object APIs are AMD-side in this tree. These gates indicate an unsupported local capability, not a contract failure.

Run the layer with:

```bash
cmake --build <build-dir> --target contract_tests
ctest --test-dir <build-dir> -L contract --output-on-failure
```
