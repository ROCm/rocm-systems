# rccl-PureUnitTests — CPU-Only Test Binary

**Date:** 2026-07-28
**Compiler:** g++ 13.3.0 + hipcc --offload-host-only (optional)
**GTest:** 1.14.0 (system)
**Result:** 285 tests across 41 suites; all pass

## What This Is

A standalone test binary that compiles with g++ and hipcc `--offload-host-only`
(no GPU codegen) and runs on CPU-only nodes. It contains functional tests that
compile and call real RCCL production source, migrated from GPU-dependent test
binaries where they were trapped behind `hip::device` link dependencies.

## Runtime Dependencies

Zero HIP/ROCm/HSA libraries:
```
libstdc++.so.6, libm.so.6, libgcc_s.so.1, libc.so.6
```

## Test Suites

### Compiling real RCCL `.cc` source (142 tests)

| Source File | Suite | Tests | Real RCCL source compiled |
|-------------|-------|-------|---------------------------|
| AltRsmiTests.cpp | AltRsmiTest | 44 | alt_rsmi.cc (g++) |
| MemManagerTests.cpp | MemManager* | 68 | mem_manager.cc (hipcc) |
| VersionInfoTests.cpp | VersionInfoTests | 7 | kernel_config.cc (g++) |
| NullParentTests.cpp | NullParentTest | 6 | paths.cc, search.cc (hipcc) |
| EnqueueCountTests.cpp | EnqueueCountTests | 4 | kernel_config.cc (g++) |
| RomeTopoConsensusTests.cpp | RomeTopoConsensus | 4 | rome_topo_consensus.cc (g++) |
| IommuPassthrough_test.cpp | IommuPassthroughTest | 6 | kernel_config.cc (shared) |

### Header-only, no real `.cc` source (143 tests)

These test files existed in `rccl-UnitTestsFixtures` / `rccl-UnitTestsFixturesDebug`
but were trapped behind `hip::device` link dependencies. They only `#include`
RCCL headers (inline, constexpr, template functions) resolved via stub headers.

| Source File | Suite | Tests | Headers tested |
|-------------|-------|-------|----------------|
| BitOpsTests.cpp | BitOps* | 106 | bitops.h |
| BootstrapBidirTests.cpp | BootstrapBidir | 16 | bootstrap.h (via hand-rolled stub) |
| TimeoutTests.cpp | TimeoutTests | 8 | comm.h |
| DdaCollCommonTests.cpp | DdaCollCommon | 6 | CollCommon.h |
| MiscTests.cpp | MiscTests | 1 | comm.h |

## How It Works

The binary avoids HIP/ROCm dependencies through two mechanisms:

1. **Stub headers** in `test/pure/stubs/` shadow the real HIP/HSA/NCCL headers
   (placed first in the include path via `-iquote`). They provide type
   definitions and no-op API functions.

2. **hipcc `--offload-host-only`** compiles real RCCL source files (hipified)
   with full header resolution but no GPU code. Used for tests that need real
   RCCL struct layouts (NullParentTests, MemManagerTests).

RCCL source files compiled directly into the binary:
- Via g++: `kernel_config.cc`, `alt_rsmi.cc`, `rome_topo_consensus.cc`
- Via hipcc: `paths.cc`, `search.cc`, `mem_manager.cc` (through wrapper files)

## Build

```bash
cd projects/rccl/test/pure
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
./build/rccl-PureUnitTests
```

hipcc wrapper tests auto-disable if hipcc or hipified sources are not found.
