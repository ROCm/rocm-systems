# rccl-PureUnitTests — CPU-Only Test Binary

**Date:** 2026-07-27
**Host:** Dell Strix Halo laptop (gfx1151 iGPU present, not used by this binary)
**Compiler:** g++ 13.3.0 (Ubuntu 24.04)
**GTest:** 1.14.0 (system)
**Result:** 222 tests, 0 failures, ~180ms wall time

## What This Is

A standalone test binary that compiles with plain g++ (no hipcc, no ROCm toolchain)
and runs on CPU-only nodes. It extracts logically CPU-only tests from RCCL's existing
test binaries — tests that were previously trapped behind `hip::device` link dependencies
and could only run on nodes with GPUs.

## Runtime Dependencies

Zero HIP/ROCm/HSA libraries:
```
libstdc++.so.6, libm.so.6, libgcc_s.so.1, libc.so.6
```

## Test Suites (12 test files, 222 tests)

| Source File | Suite | Tests | What It Covers |
|-------------|-------|-------|----------------|
| BitOpsTests.cpp | BitOps* | 106 | Bit manipulation: DIVUP, ROUNDUP, ALIGN, u32fp8, hash |
| IommuPassthrough_test.cpp | IommuPassthroughTest | 6 | IOMMU passthrough detection via sysfs parsing |
| VersionInfoTests.cpp | VersionInfoTests | 7 | HIP/ROCm version info string formatting |
| DdaCollCommonTests.cpp | DdaCollCommon | 6 | DDA collective common utilities (half/bf16 ops) |
| RomeTopoConsensusTests.cpp | RomeTopoConsensus | 4 | Rome topology consensus algorithm |
| MiscTests.cpp | MiscTests | 1 | ncclTaskCollSorter emptiness check |
| TimeoutTests.cpp | TimeoutTests | 8 | ncclTimeout result code contract (enum, strings, async error) |
| EnqueueCountTests.cpp | EnqueueCountTests | 4 | ncclFuncSendCount/RecvCount/MaxSendRecvCount |
| BootstrapBidirTests.cpp | BootstrapBidir | 16 | Bootstrap bidirectional AllGather gating contract |
| AltRsmiTests.cpp | AltRsmiTest | 44 | Alt-RSMI sysfs/DRM topology parsing |
| NullParentTests.cpp | NullParentTest | 11 | Null gpu.parent guard in topo path/search (5cb9833) |
| *(support)* ProcessIsolatedTestRunner.cpp | — | — | Fork-based test isolation framework |

## How It Works

The binary avoids HIP/ROCm dependencies through two mechanisms:

1. **Stub headers** in `test/pure/stubs/` shadow the real HIP/HSA/NCCL headers
   (placed first in the include path). They provide type definitions and no-op
   API functions — enough to compile RCCL source headers without the HIP runtime.

2. **Stub implementations** in `rccl_stubs.cpp` and `bootstrap_bidir_stub.cpp`
   replace functions normally provided by `librccl.so` (ncclDebugLog, ncclGetErrorString,
   ncclCommSetAsyncError, ncclCommGetAsyncError, bootstrapBidirEnabled).

RCCL source files are compiled directly into the binary (not linked against librccl.so):
- `src/misc/kernel_config.cc` — kernel config parsing
- `src/misc/alt_rsmi.cc` — alternative RSMI implementation
- `src/graph/rome_topo_consensus.cc` — Rome topology consensus

## Build

```bash
cd projects/rccl/test/pure
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./rccl-PureUnitTests
```

## Test Run Log

Full log: `test-run-20260727-115907.log` (67KB, 211/211 passed — before NullParentTests addition)

## Remaining Feasible Candidates

| Test File | Est. Tests | Effort | Blocker |
|-----------|------------|--------|---------|
| ParameterApiTests.cpp | ~35 | Medium | Needs param.cc + env plugin stubs |
| ParamTests.cpp | ~8 | Medium | Needs param.cc + topo.h |
| NetSocketTests.cpp | ~10 | Medium | Needs net.h dependency chain |
| TopoEnvPolicyTests.cpp | ~20 | High | Needs full topo subsystem (.cc files) |
| DdaAlltoAllThresholdTests.cpp | many | High | Needs DDA .cc + full ncclComm |
| DdaFabricEligibilityTests.cpp | many | High | Needs DDA fabric .cc files |
| DdaIpcEligibilityTests.cpp | many | High | Needs DDA IPC .cc files |
