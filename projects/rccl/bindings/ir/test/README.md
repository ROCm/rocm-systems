# IR_test — Comprehensive functional test for `librccl_device.bc`

`IR_test` verifies every active API exported by `nccl_device_wrapper__impl.h`,
covering the full bitcode path end-to-end: C-ABI thunks → vtable dispatch →
per-implementation device code. It runs on real GPU hardware and reports each
API individually with `[ PASS ]`, `[ FAIL ]`, or `[ SKIP ]`.

## Files

| File | Purpose |
|------|---------|
| `IR_test.cpp` | Test source — GPU kernels, host driver, result reporting |
| `run_IR_test.sh` | Build-and-run wrapper with preflight checks and env knobs |

## APIs tested

| Test ID | API / group |
|---------|-------------|
| A.1 | `ncclGetPeerPointerTeam` — 20 pointer-arithmetic cases |
| B1.1–B1.3 | `ncclCoopAnyInitThread` + `ncclCoopSize/ThreadRank/NumThreads` |
| B2.1–B2.3 | `ncclCoopAnyInitWarp` + accessors |
| B3.1–B3.12 | `ncclCoopAnyInitLanes` — full/sparse/single-bit mask; lane 63 on 64-lane warps |
| B4.1–B4.6 | `ncclCoopAnyInitWarpSpan` — 1-warp and 2-warp spans |
| B5.1–B5.6 | `ncclCoopAnyInitCta` — block sizes 64 and 128 |
| B6.1–B6.5 | `ncclCoopSync` — all five coop types |
| B7.1 | `ncclLsaBarrierSession_C` sizeof / alignment (structural) |
| B7.2–B7.5 | `ncclLsaBarrierSession{Init,Arrive,Wait,Sync}` — **SKIP** (require a live `ncclDevComm`) |

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
| `GPU` | *(unset — all GPUs)* | `HIP_VISIBLE_DEVICES` value (e.g. `0`) |
| `OUTDIR` | `/tmp/ir_test` | Directory for the compiled test binary and `.ll` dump |

## Expected output

```
[IR_test] GPU 0: AMD Instinct MI300X  warpSize=64
  [ PASS ] A.1    ncclGetPeerPointerTeam (20 cases)           20/20
  [ PASS ] B1.1   ncclCoopAnyInitThread: size==1              64/64
  ...
  [ PASS ] B6.5   ncclCoopSync (Cta blk=128): kernel completes  128/128
  [ PASS ] B7.1   ncclLsaBarrierSession_C: sizeof/align        1/1
  [ SKIP ] B7.2   ncclLsaBarrierSessionInit   requires live ncclDevComm ...
  ...
[IR_test] TOTAL: 37 passed, 0 failed, 4 skipped across 1 GPU(s)
```

Exit code `0` = all non-skipped tests passed; `1` = at least one failure;
`2` = preflight error (missing bitcode, headers, etc.).

The script also writes a human-readable LLVM IR dump of the bitcode to
`$OUTDIR/librccl_device.ll` for offline inspection.
