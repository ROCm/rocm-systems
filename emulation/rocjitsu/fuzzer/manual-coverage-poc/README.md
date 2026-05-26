# Manual HIP Device-Coverage PoC

This is an isolated ROCm/HIP proof of concept. It does not use NVBit and does
not modify the existing CUDA implementation.

The purpose is to validate the coverage data path before building an AMDGCN
binary-instrumentation backend:

```text
HIP kernel hook calls
  -> device coverage counters
  -> host copy
  -> AFL trace_bits upper half
```

## Files

- `device_kernel.hip`: HIP kernel with manual `record_edge(...)` calls at
  basic-block-like points. These calls are placeholders for future AMDGCN
  assembly-inserted hooks.
- `host_afl_driver.cpp`: ordinary C++ host driver. It maps `__AFL_SHM_ID`, loads
  the HIP code object, launches the kernel, copies device counters back, and
  merges them into AFL's bitmap.
- `Makefile`: builds the HIP code object with `hipcc`; optionally builds the
  host driver with AFL's compiler wrapper.

## Build

Standalone HIP/memory-path check:

```bash
cd rocm/manual-coverage-poc
make standalone GPU_ARCH=native
```

AFL-visible coverage check:

```bash
cd rocm/manual-coverage-poc
make afl AFL_CXX=../../Tools/AFLplusplus/afl-clang-fast++ GPU_ARCH=native
```

If `native` is not supported in your ROCm setup, pass an explicit target such as
`GPU_ARCH=gfx942`.

## Run

Standalone mode validates that the kernel updates device coverage and the host
can read it:

```bash
mkdir -p build
printf '1' > build/seed
ROCM_POC_VERBOSE=1 ./build/manual_coverage_standalone \
  ./build/manual_device_coverage.hsaco build/seed
```

With AFL tooling, the interesting bytes should appear in the upper half of the
64K bitmap, starting at byte id `32768`:

```bash
mkdir -p build
printf '1' > build/seed
../../Tools/AFLplusplus/afl-showmap -o - -- \
  ./build/manual_coverage_afl ./build/manual_device_coverage.hsaco build/seed
```

Different input values (`1`, `2`, `3`, other) should produce different
device-side edge bytes.

## Boundary Being Tested

The host coverage runtime owns:

- AFL shared-memory mapping.
- HIP allocation for device counters and previous-basic-block state.
- Kernel launch synchronization.
- Device-to-host counter copy.
- Quantization and merge into `trace_bits`.

The device instrumentation backend only needs to guarantee:

```text
At each instrumented device basic-block entry, update
coverage[32768 + edge_hash] using per-work-item previous-BB state.
```

In this PoC, that backend is manual C++ calls in `device_kernel.hip`. In a real
ROCm port, it would be AMDGCN assembly surgery that preserves the same hook
semantics.
