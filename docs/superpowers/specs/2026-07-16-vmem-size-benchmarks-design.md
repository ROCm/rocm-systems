# Vmem/BO size-sweep allocation benchmarks

**Date:** 2026-07-16
**Location:** `projects/rocr-runtime/rocrtst/suites/aie-performance/`

## Goal

Compare the cost of allocating memory on the NPU across a fixed set of sizes
(8 KB, 16 KB, 32 KB, 64 KB) using two backends:

- **HSA vmem** — `hsa_amd_vmem_*` virtual-memory API against the AIE agent.
- **XRT** — `xrt::bo` buffer objects.

The two backends are delivered as two new standalone Google Benchmark
executables so results can be compared side by side.

## Scope

- Two new source files and two new CMake targets.
- The existing `vmem_alloc_hsa.cpp` (which sweeps allocation *count* at a fixed
  4096-byte size) is left untouched.
- NPU/AIE hardware only. Cases skip cleanly when the device or toolchain
  artifacts are unavailable.

## Size sweep

Both benchmarks drive the size via Google Benchmark's range mechanism:

```
->RangeMultiplier(2)->Range(8192, 65536)->Unit(benchmark::kMicrosecond)
```

`state.range(0)` **is** the allocation size in bytes (8192, 16384, 32768,
65536). Exactly one allocation is performed per timing iteration.

Each case sets `state.SetBytesProcessed(state.iterations() * size)` so the
reporter shows throughput per size.

## File: `vmem_size_hsa.cpp` (target `vmem-size-hsa`)

NPU/AIE only. Reuses the discovery helpers established in `vmem_alloc_hsa.cpp`:

- `find_agent<HSA_DEVICE_TYPE_AIE>` — first AIE agent.
- A global-segment pool whose runtime allocation granule is non-zero and
  `<= 8192`, so every swept size is representable without the granule rounding
  up the smallest request.

HSA lifetime (`hsa_init` / `hsa_shut_down`), agent discovery, and pool
discovery happen once at the top of each benchmark function, outside the timed
loop. If the agent or a suitable pool is missing the case is skipped via
`SkipWithError`, not failed.

### Cases

1. **`VMemHandleAIE`** — physical allocation only. Times, per iteration:
   - `hsa_amd_vmem_handle_create(pool, size, MEMORY_TYPE_NONE, 0, &handle)`
   - `hsa_amd_vmem_handle_release(handle)`

2. **`VMemMappedAIE`** — full usable allocation. Times, per iteration:
   - `hsa_amd_vmem_handle_create`
   - `hsa_amd_vmem_address_reserve(&va, size, 0, 0)`
   - `hsa_amd_vmem_map(va, size, 0, handle, 0)`
   - `hsa_amd_vmem_set_access(va, size, &access_desc, 1)` where the descriptor
     grants the AIE agent read/write access.
   - Teardown: `hsa_amd_vmem_unmap(va, size)`,
     `hsa_amd_vmem_address_free(va, size)`, `hsa_amd_vmem_handle_release`.

On any failure inside the loop the partial state is torn down in reverse order
to avoid leaking handles/reservations across iterations, then the case is
skipped.

### Registration

```
BENCHMARK(VMemHandleAIE)->RangeMultiplier(2)->Range(8192, 65536)->Unit(benchmark::kMicrosecond);
BENCHMARK(VMemMappedAIE)->RangeMultiplier(2)->Range(8192, 65536)->Unit(benchmark::kMicrosecond);
```

## File: `vmem_size_xrt.cpp` (target `vmem-size-xrt`)

XRT has no virtual-memory reserve/map API; its allocation primitive is
`xrt::bo`. Allocating a `bo` requires a valid memory group id, which is
obtained from the suite's kernel:

- Open `xrt::device(0)`, load the suite xclbin (`DEFAULT_XCLBIN_PATH`).
- Find the kernel whose name begins with `MLIR_AIE`.
- Register the xclbin, create an `xrt::hw_context`, construct the kernel, and
  take `kernel.group_id(3)` (the host-only input group used by the existing XRT
  benchmarks).

All of this is setup outside the timed loop. If the device, xclbin, or kernel
is unavailable, the case is skipped via `SkipWithError`.

### Cases

1. **`BoAllocXRT`** — times, per iteration:
   - `xrt::bo(device, size, XRT_BO_FLAGS_HOST_ONLY, group)` construction
   - destruction (scope exit)

2. **`BoAllocMapXRT`** — times, per iteration:
   - `xrt::bo(...)` construction
   - `bo.map<void*>()`
   - destruction

`benchmark::DoNotOptimize` / `ClobberMemory` guard against the allocation being
optimized away.

### Registration

```
BENCHMARK(BoAllocXRT)->RangeMultiplier(2)->Range(8192, 65536)->Unit(benchmark::kMicrosecond);
BENCHMARK(BoAllocMapXRT)->RangeMultiplier(2)->Range(8192, 65536)->Unit(benchmark::kMicrosecond);
```

## CMake changes

Append two `add_executable` blocks to `CMakeLists.txt`, following the existing
style:

```cmake
# vmem size-sweep (HSA) microbenchmark
add_executable(vmem-size-hsa vmem_size_hsa.cpp)
target_compile_features(vmem-size-hsa PRIVATE cxx_std_20)
target_link_libraries(vmem-size-hsa PRIVATE
  benchmark::benchmark_main
  hsa-runtime64::hsa-runtime64)

# vmem/bo size-sweep (XRT) microbenchmark
add_executable(vmem-size-xrt vmem_size_xrt.cpp)
target_compile_features(vmem-size-xrt PRIVATE cxx_std_20)
target_include_directories(vmem-size-xrt PRIVATE "${XRT_INC_DIR}")
target_compile_definitions(vmem-size-xrt PRIVATE
  DEFAULT_XCLBIN_PATH=${FINAL_XCLBIN})
target_link_directories(vmem-size-xrt PRIVATE "${XRT_LIB_DIR}")
target_link_libraries(vmem-size-xrt PRIVATE
  benchmark::benchmark_main
  xrt_coreutil)
add_dependencies(vmem-size-xrt aie_kernel_artifacts)
```

## Testing / verification

- Build succeeds: `vmem-size-hsa` and `vmem-size-xrt` compile and link.
- On NPU hardware, each executable runs and reports four rows per case
  (8K/16K/32K/64K). On machines without the device, cases skip rather than
  crash.
- Correctness of the allocation itself is out of scope — these are timing
  microbenchmarks; the workflow mirrors already-validated patterns in the
  suite.

## Non-goals

- No size×count grid; count is fixed at 1 per iteration.
- No GPU-agent variant (the request is NPU-specific).
- No changes to `vmem_alloc_hsa.cpp`.
