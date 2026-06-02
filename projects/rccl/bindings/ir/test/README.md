# IR_test — Comprehensive functional test for `librccl_device.bc`

`IR_test` verifies every active API exported by `nccl_device_wrapper__impl.h`,
covering the full bitcode path end-to-end: C-ABI thunks → vtable dispatch →
per-implementation device code. It runs on real GPU hardware as a **GoogleTest**
binary (`IRDeviceTest.*`), so it integrates with `--gtest_filter`, the RCCL
`test_runner`, and the pytest harness in `test/ir-device/`.

## Files

| File | Purpose |
|------|---------|
| `IR_test.cpp` | GoogleTest source — GPU kernels + `TEST_F(IRDeviceTest, …)` cases |
| `run_IR_test.sh` | Build-and-run wrapper (links GTest) with preflight checks and env knobs |

## APIs tested

| GTest case | API / group |
|------------|-------------|
| `A_GetPeerPointerTeam` | `ncclGetPeerPointerTeam` — 20 pointer-arithmetic cases |
| `B1_CoopInitThread` | `ncclCoopAnyInitThread` + `ncclCoopSize/ThreadRank/NumThreads` |
| `B2_CoopInitWarp` | `ncclCoopAnyInitWarp` + accessors |
| `B3a_…/B3b_…/B3c_…/B3d_CoopInitLanes_*` | `ncclCoopAnyInitLanes` — full/sparse/single-bit/lane-63 masks |
| `B4a_…/B4b_CoopInitWarpSpan_*` | `ncclCoopAnyInitWarpSpan` — 1-warp and 2-warp spans |
| `B5_CoopInitCta` | `ncclCoopAnyInitCta` — block sizes 64 and 128 |
| `B6_CoopSync` | `ncclCoopSync` — all five coop types |
| `B7a_LsaBarrierSessionStructural` | `ncclLsaBarrierSession_C` sizeof / alignment |
| `B7b_LsaBarrierSessionRuntime` | `ncclLsaBarrierSession{Init,Arrive,Wait,Sync}` — **SKIP** (require a live `ncclDevComm`) |

## Prerequisites

1. **ROCm** installed (default `/opt/rocm`).
2. **RCCL CMake build** run at least once to populate the hipify-staged headers
   and generate `nccl.h` / `rccl.h`:
   ```bash
   cmake -B build/release -DEMIT_LLVM_IR=ON -DBITCODE_LIB_ARCH=<arch> .
   cmake --build build/release
   ```
3. **`librccl_device.bc`** present at `build/release/lib/librccl_device.bc`
   (built by the `llvm_ir` CMake target, enabled by `-DEMIT_LLVM_IR=ON`).

## Running

```bash
# From the repo root or any directory:
cd bindings/ir/test

# Basic — uses defaults (arch=gfx950, all GPUs, build/release):
bash run_IR_test.sh

# Typical invocation on a gfx942 machine, single GPU:
ARCH=gfx942 GPU=0 bash run_IR_test.sh

# Custom build directory and bitcode:
BUILD=/path/to/rccl/build ARCH=gfx942 GPU=0 bash run_IR_test.sh
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARCH` | `gfx950` | `--offload-arch` passed to `hipcc` and expected bitcode target |
| `ROCM_PATH` | `/opt/rocm` | ROCm installation root |
| `BUILD` | `<repo>/build/release` | RCCL CMake build directory |
| `BC` | `$BUILD/lib/librccl_device.bc` | Path to the bitcode library |
| `GTEST_ROOT` | `$BUILD/gtest` | GoogleTest install prefix (`include/` + `lib{,64}/libgtest.a`) |
| `GPU` | *(unset — all GPUs)* | `HIP_VISIBLE_DEVICES` value (e.g. `0`) |
| `OUTDIR` | `/tmp/ir_test` | Directory for the compiled test binary and `.ll` dump |
| `BUILD_ONLY` | `0` | `1` = compile the test and exit (used by the pytest harness) |
| `RUN_ARGS` | *(empty)* | Extra args for the binary, e.g. `--gtest_filter=IRDeviceTest.B6_CoopSync` |

## Expected output

Standard GoogleTest output, e.g.:

```
[IR_test] device 0: AMD Instinct MI300X  warpSize=64
[==========] Running 12 tests from 1 test suite.
[ RUN      ] IRDeviceTest.A_GetPeerPointerTeam
[       OK ] IRDeviceTest.A_GetPeerPointerTeam (3 ms)
...
[ RUN      ] IRDeviceTest.B7b_LsaBarrierSessionRuntime
[  SKIPPED ] IRDeviceTest.B7b_LsaBarrierSessionRuntime (0 ms)
[==========] 12 tests ran.
[  PASSED  ] 11 tests.
[  SKIPPED ] 1 test.
```

Exit code `0` = all run cases passed (skips do not fail); `1` = at least one
failure; `2` = preflight error (missing bitcode, headers, GTest, etc.).

The script also writes a human-readable LLVM IR dump of the bitcode to
`$OUTDIR/librccl_device.ll` for offline inspection.
