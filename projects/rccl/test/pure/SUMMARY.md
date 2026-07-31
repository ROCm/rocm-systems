# rccl-PureUnitTests — CPU-Only Test Binary

**Date:** 2026-07-31
**Compiler:** hipcc --offload-host-only (mandatory)
**GTest:** 1.14.0 (system, RCCL-vendored, or FetchContent)
**Result:** 272 tests pass, 0 failures, 7 disabled

## What This Is

A standalone test binary that compiles with hipcc `--offload-host-only`
(no GPU codegen) against real hipified RCCL headers and runs on CPU-only
nodes. It contains functional tests that compile and call real RCCL
production source, migrated from GPU-dependent test binaries where they
were trapped behind `hip::device` link dependencies.

## Runtime Dependencies

Zero HIP/ROCm/HSA libraries:
```
libstdc++.so.6, libm.so.6, libgcc_s.so.1, libc.so.6
```

## Test Suites

### Compiling real RCCL `.cc` source (150 active tests)

| Source File | Suite | Tests | Real RCCL source compiled |
|-------------|-------|-------|---------------------------|
| MemManagerTests.cpp | MemManager* | 61 | mem_manager.cc |
| AltRsmiTests.cpp | AltRsmiTest | 44 | alt_rsmi.cc |
| BootstrapBidirTests.cpp | BootstrapBidir | 16 | bootstrap.cc (via bootstrap_wrapper.cc) |
| TimeoutTests.cpp | TimeoutTests | 8 | init_stubs.cpp (real RCCL types) |
| VersionInfoTests.cpp | VersionInfoTests | 7 | kernel_config.cc |
| IommuPassthrough_test.cpp | IommuPassthroughTest | 6 | kernel_config.cc |
| EnqueueCountTests.cpp | EnqueueCountTests | 4 | kernel_config.cc |
| RomeTopoConsensusTests.cpp | RomeTopoConsensus | 4 | rome_topo_consensus.cc |

### Header-only, no real `.cc` source (122 tests)

These test files existed in `rccl-UnitTestsFixtures` / `rccl-UnitTestsFixturesDebug`
but were trapped behind `hip::device` link dependencies. They only `#include`
RCCL headers (inline, constexpr, template functions).

| Source File | Suite | Tests | Headers tested |
|-------------|-------|-------|----------------|
| BitOpsTests.cpp | BitOps* | 115 | bitops.h |
| DdaCollCommonTests.cpp | DdaCollCommon | 6 | CollCommon.h |
| MiscTests.cpp | MiscTests | 1 | comm.h |

### Disabled tests (7)

Seven `MemManagerRealMem` tests are disabled (`DISABLED_` prefix). They call
real HIP allocation APIs that require GPU hardware. Will be re-enabled once
a HIP mock layer is available.

## How It Works

All files compile with **hipcc `--offload-host-only`** against real hipified
RCCL headers from the build tree. No stub headers, no g++ fallback. The
linker flag `-no-hip-rt` prevents linking the HIP runtime.

RCCL source files compiled into the binary:
- Directly: `kernel_config.cc`, `alt_rsmi.cc`, `rome_topo_consensus.cc`
- Via wrapper files: `mem_manager.cc`, `bootstrap.cc` (through `rccl-source-wrappers` static library)

## Build

```bash
cd projects/rccl/test/pure
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DROCM_PATH=/path/to/rocm
cmake --build build -j$(nproc)
./build/rccl-PureUnitTests
```

Prerequisite: hipified sources must exist at `../../build/hipify/`. Generate
them by running the main RCCL build or `cmake --build build --target hipify_all`.
