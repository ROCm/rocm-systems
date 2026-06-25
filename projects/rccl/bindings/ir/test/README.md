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

The build (`hipcc` at `-O0`, with the bitcode routed to the AMDGPU device-side
LTO link) and the run are driven by the pytest harness in `test/ir-device/`.

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

The test is built and run through the pytest harness, which compiles
`IR_test.cpp` once (auto-emitting `librccl_device.bc` if it is missing) and then
runs the GoogleTest cases, skipping cleanly when prerequisites are absent:

```bash
cd test/ir-device
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# Typical invocation on a gfx942 machine:
ARCH=gfx942 pytest -v
```

### Environment variables

The harness reads these (see `test/ir-device/tests/conftest.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ARCH` | `gfx942` | `--offload-arch` passed to `hipcc` and expected bitcode target |
| `ROCM_PATH` | `/opt/rocm` | ROCm installation root |
| `RCCL_BUILD` | `<repo>/build/release` | RCCL CMake build directory |
| `GTEST_ROOT` | `$RCCL_BUILD/gtest` | GoogleTest install prefix (`include/` + `lib{,64}/libgtest.a`) |
| `IR_OUTDIR` | `<workdir>/ir_test_build` | Directory for the compiled test binary |

To run the compiled binary directly (after a pytest run leaves it under
`IR_OUTDIR`), invoke it with a `--gtest_filter`, e.g.:

```bash
ir_test_build/IR_test.exe --gtest_filter=IRDeviceTest.B6_CoopSync
```

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
failure.
