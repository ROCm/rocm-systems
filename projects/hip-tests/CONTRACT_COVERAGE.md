# HIP Contract Test Coverage Tracker

This document is a compact status snapshot for the HIP public-API contract tier in
[`projects/hip-tests/catch/contract`](catch/contract/). It complements the generated
[`catch/TEST_PLAN.md`](catch/TEST_PLAN.md), which is the detailed per-test inventory of
`// @asserts:` invariants.

The coverage percentage is approximate **API-name coverage** against declarations parsed
from [`projects/hip/include/hip/hip_runtime_api.h`](../hip/include/hip/hip_runtime_api.h).
It is not behavioral coverage: one API can have many modes and edge cases, and the
contract tier intentionally pins only small, portable semantic guarantees.

## Snapshot

<!-- contract-coverage-snapshot
contract_tests: 612
contract_domains: 118
declared_apis: 498
covered_apis: 487
uncovered_allowlisted: 11
coverage_pct: 97.8
-->

- Snapshot date: 2026-08-26
- Snapshot commit: `920418c484`
- Contract tests: 612
- Contract domains: 118
- Declared HIP runtime APIs parsed from `hip_runtime_api.h`: 498
- Declared HIP runtime APIs directly exercised by contract tests: 487
- Intentionally uncovered, allowlisted APIs: 11
- Approximate declared API-name coverage: 97.8%
- Additional public macro exercised: `hipLaunchKernelGGL`
- Additional non-runtime-header APIs exercised: HIPRTC (`hiprtcCreateProgram`,
  `hiprtcCompileProgram`, `hiprtcGetCodeSize`, `hiprtcGetCode`,
  `hiprtcGetProgramLogSize`, `hiprtcGetProgramLog`, `hiprtcDestroyProgram`) and
  OpenGL interop entry points declared in `hip/hip_gl_interop.h` (`hipGLGetDevices`,
  `hipGraphicsGLRegisterBuffer`, `hipGraphicsGLRegisterImage`). These are excluded
  from the `hip_runtime_api.h` denominator.

## Relationship to generated artifacts

- [`catch/TEST_PLAN.md`](catch/TEST_PLAN.md) is the detailed, generated inventory. It
  lists each test case, the API it asserts, and the one-line invariant extracted from
  the source `// @asserts:` tag.
- [`catch/contract/uncovered_apis.txt`](catch/contract/uncovered_apis.txt) is the
  source of truth for APIs that are deliberately not covered by the device-only
  contract harness.
- [`catch/contract/tools/check_contract_coverage.py`](catch/contract/tools/check_contract_coverage.py)
  is the static drift checker. It recomputes the declared, covered, uncovered, and
  stale-allowlist sets, verifies the numeric snapshot block above, and is run by the
  HIP contract-test coverage workflow.

Keep this file short and human-readable. Do not duplicate the complete generated test
plan or the full covered-API listing here.

## Current allowlisted gaps

The checker currently reports these 11 uncovered declared APIs, all intentionally
allowlisted:

### External-semaphore graph nodes

These APIs require a valid external semaphore node. The device-only contract harness
cannot construct such a node without an external producer, and the add path faults when
fed an invalid handle on some runtime paths.

- `hipGraphAddExternalSemaphoresSignalNode`
- `hipGraphAddExternalSemaphoresWaitNode`
- `hipGraphExecExternalSemaphoresSignalNodeSetParams`
- `hipGraphExecExternalSemaphoresWaitNodeSetParams`
- `hipGraphExternalSemaphoresSignalNodeGetParams`
- `hipGraphExternalSemaphoresSignalNodeSetParams`
- `hipGraphExternalSemaphoresWaitNodeGetParams`
- `hipGraphExternalSemaphoresWaitNodeSetParams`

### Deprecated texture-reference border color

The AMD runtime currently implements these entry points with an unconditional
`assert(false)` after the image-support check, so calling them aborts assert-enabled test
binaries instead of exposing portable state to assert.

- `hipTexRefGetBorderColor`
- `hipTexRefSetBorderColor`

### Windows-only deprecated texture-reference bind

- `hipBindTextureToMipmappedArray` — the AMD Linux runtime rejects the device-side bind
  with `hipErrorInvalidTexture`; the positive path is Windows-only, matching the in-tree
  unit-test guard.

## Contract domains

| Contract domain | Tests |
|---|---:|
| `array3d` | 3 |
| `array_copy` | 5 |
| `array_copy_ext` | 8 |
| `array_memory` | 4 |
| `async_copy3d` | 6 |
| `async_transfer` | 4 |
| `call_config` | 3 |
| `capture_to_graph` | 3 |
| `context` | 6 |
| `context_config` | 6 |
| `context_mutation` | 5 |
| `copy3d` | 4 |
| `device` | 7 |
| `device_config` | 6 |
| `device_identity` | 9 |
| `device_lifecycle` | 7 |
| `device_reset` | 1 |
| `device_texture_query` | 6 |
| `driver_array` | 6 |
| `driver_copy3d` | 5 |
| `driver_entry_point` | 5 |
| `driver_error` | 6 |
| `driver_graph_node` | 5 |
| `driver_launch_ex` | 3 |
| `driver_memcpy` | 6 |
| `driver_memcpy_2d` | 5 |
| `driver_memset_2d` | 6 |
| `driver_memset_async` | 6 |
| `driver_memset_async_2d3d` | 5 |
| `driver_pitched_memory` | 7 |
| `driver_texture_object` | 6 |
| `error_api` | 6 |
| `extension` | 6 |
| `external_resource` | 8 |
| `func_attributes` | 8 |
| `graph` | 5 |
| `graph_batch_mem_op` | 4 |
| `graph_capture` | 4 |
| `graph_child` | 3 |
| `graph_clone` | 3 |
| `graph_debug` | 2 |
| `graph_event` | 3 |
| `graph_exec_lifecycle` | 5 |
| `graph_generic_node` | 3 |
| `graph_host` | 3 |
| `graph_instantiate_params` | 2 |
| `graph_kernel` | 3 |
| `graph_mem_nodes` | 7 |
| `graph_memcpy3d_node` | 4 |
| `graph_node_attributes` | 3 |
| `graph_node_enabled` | 3 |
| `graph_node_find` | 3 |
| `graph_node_params` | 7 |
| `graph_node_setters` | 5 |
| `graph_node_types` | 5 |
| `graph_symbol_copy_nodes` | 3 |
| `graph_topology` | 5 |
| `graph_update` | 9 |
| `graph_user_objects` | 4 |
| `graphics_interop` | 8 |
| `green_context` | 6 |
| `host_alloc_aliases` | 7 |
| `host_memory` | 5 |
| `ipc` | 5 |
| `jit_link` | 7 |
| `kernel` | 4 |
| `kernel_launch` | 7 |
| `kernel_name_ref` | 3 |
| `kernel_object_attributes` | 3 |
| `library` | 14 |
| `library_file` | 3 |
| `logging` | 3 |
| `managed_memory` | 5 |
| `mem_advise` | 6 |
| `mem_advise_v2` | 4 |
| `mem_batch_copy` | 3 |
| `mem_batch_copy_3d` | 4 |
| `mem_batch_discard` | 7 |
| `mem_location_pool` | 4 |
| `mem_map_array` | 2 |
| `memory` | 5 |
| `memory_pool` | 6 |
| `memory_pool_access` | 3 |
| `memory_pool_lifecycle` | 4 |
| `mempool_shareable_handle` | 3 |
| `memset` | 6 |
| `mipmapped_array` | 4 |
| `module` | 7 |
| `module_exec` | 12 |
| `module_load_ex` | 4 |
| `module_load_file` | 6 |
| `multi_device_launch` | 4 |
| `occupancy` | 3 |
| `occupancy_ext` | 4 |
| `occupancy_variable` | 6 |
| `peer_access` | 6 |
| `peer_copy` | 5 |
| `peer_query` | 5 |
| `pitched_memory` | 4 |
| `pointer_info` | 7 |
| `pointer_query` | 6 |
| `profiler` | 3 |
| `runtime` | 10 |
| `stream_attach` | 5 |
| `stream_attributes` | 5 |
| `stream_callbacks` | 5 |
| `stream_capture_mode` | 6 |
| `stream_cu_mask` | 5 |
| `stream_event` | 7 |
| `stream_memory_ops` | 6 |
| `stream_props` | 7 |
| `symbol_copy` | 6 |
| `texture` | 8 |
| `texture_reference` | 6 |
| `texture_reference_symbol` | 10 |
| `transfer` | 4 |
| `vmm` | 5 |
| `vmm_handle` | 4 |

## Keeping coverage honest

Coverage drift is guarded automatically. The checker parses the declared public APIs
from `hip_runtime_api.h`, scans the contract sources for API calls, loads the explicit
allowlist, and fails when any declared API has neither a test nor a justification. It
also fails on stale allowlist entries so the allowlist stays truthful.

When adding a HIP API or changing contract coverage:

1. Add a contract test under [`catch/contract/`](catch/contract/) when the API has a
   portable device-only invariant.
2. If the API genuinely cannot be covered in this harness, add it to
   [`catch/contract/uncovered_apis.txt`](catch/contract/uncovered_apis.txt) with a
   specific reason.
3. Run:

   ```bash
   python3 projects/hip-tests/catch/contract/tools/check_contract_coverage.py --check
   python3 projects/hip-tests/catch/tools/gen_test_plan.py --check
   ```

4. Update this snapshot only when the counts or intentionally-uncovered rationale change.
