# rccl-PureUnitTests — CPU-Only Test Binary

**Date:** 2026-07-28
**Compiler:** g++ 13.3.0 + hipcc --offload-host-only (optional)
**GTest:** 1.14.0 (system)
**Result:** 312 tests across 53 suites; 284 pass, 28 expected failures

## What This Is

A standalone test binary that compiles with g++ and hipcc `--offload-host-only`
(no GPU codegen) and runs on CPU-only nodes. It contains two categories of tests:

1. **Functional tests** (~285 tests) — compile and call real RCCL production
   source. Migrated from GPU-dependent test binaries where they were trapped
   behind `hip::device` link dependencies.

2. **Known-defect tests** (~27 tests) — reproduce known buggy code patterns
   in isolation using local types. These document and assert the presence of
   specific defects. They do NOT compile production RCCL source, so they will
   not automatically pass when a fix lands. When a fix is applied, the
   corresponding test should be updated or removed. 19 of these are expected
   to fail.

## Runtime Dependencies

Zero HIP/ROCm/HSA libraries:
```
libstdc++.so.6, libm.so.6, libgcc_s.so.1, libc.so.6
```

## Test Suites

### Functional tests (compile real RCCL source)

| Source File | Suite | Tests | Real RCCL source compiled |
|-------------|-------|-------|---------------------------|
| BitOpsTests.cpp | BitOps* | 106 | — |
| AltRsmiTests.cpp | AltRsmiTest | 44 | alt_rsmi.cc (g++) |
| MemManagerTests.cpp | MemManager* | 68 | mem_manager.cc (hipcc) |
| BootstrapBidirTests.cpp | BootstrapBidir | 16 | — (stub reimplementation) |
| TimeoutTests.cpp | TimeoutTests | 8 | — |
| VersionInfoTests.cpp | VersionInfoTests | 7 | kernel_config.cc (g++) |
| IommuPassthrough_test.cpp | IommuPassthroughTest | 6 | — |
| DdaCollCommonTests.cpp | DdaCollCommon | 6 | — |
| NullParentTests.cpp | NullParentTest | 6 | paths.cc, search.cc (hipcc) |
| EnqueueCountTests.cpp | EnqueueCountTests | 4 | kernel_config.cc (g++) |
| RomeTopoConsensusTests.cpp | RomeTopoConsensus | 4 | rome_topo_consensus.cc (g++) |
| MiscTests.cpp | MiscTests | 1 | — |

### Known-defect tests (local pattern reproduction, do not compile RCCL source)

| Source File | Suites | Tests | Expected failures |
|-------------|--------|-------|-------------------|
| NetIbCastTests.cpp | IbCastFifo, IbCastCompletion, IbWrIdPacking, IbCastRemDevIdx | 9 | 5 |
| TransportBoundsTests.cpp | CollNetSizeGuard, PatConnectMask, TransportConstants | 4 | 4 |
| ChannelGroupTests.cpp | ChannelBatchSize, GroupTaskQueue | 2 | 2 |
| BufferBoundsTests.cpp | CollTraceBuffer, GdrSupportMatrix, AllocatorResize | 3 | 3 |
| PipeReadTests.cpp | PipeRead | 2 | 1 |
| SyncRegressionTests.cpp | IbWcStatusHint, PatPreconnect, SocketMagic | 7 | 4 |

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
