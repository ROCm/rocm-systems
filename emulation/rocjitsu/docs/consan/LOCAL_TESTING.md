# Local ConSan Testing

This is a short-lived local runbook for the current workspace. Unlike the
team-facing docs, this file may name local paths and current machine state.

Current local GPU: RDNA4 `gfx1201`.

## Local Paths

- rocm-systems repo: `/home/benoit/workspace/TheRock/rocm-systems`
- TheRock ROCm install: `/home/benoit/workspace/TheRock-build/dist/rocm`
- rocJITsu build: `/home/benoit/workspace/TheRock/rocm-systems/emulation/rocjitsu/build`
- ConSan HSA hook:
  `/home/benoit/workspace/TheRock/rocm-systems/emulation/rocjitsu/build/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so`
- hip-moi build: `/home/benoit/workspace/hip-moi-build`
- IREE build: `/home/benoit/workspace/iree-build`
- rocjitsu-test-corpus: `/home/benoit/workspace/rocjitsu-test-corpus`

## Ground Rules

- Use TheRock's ROCm build for runtime tests:
  `/home/benoit/workspace/TheRock-build/dist/rocm`.
- Limit GPU test parallelism to about `8`.
- Do not run multiple IREE `ctest` sweeps against the same build directory at
  the same time; several tests share temporary paths.
- Build C++ with all useful CPU cores. The `-j8` rule is for GPU tests, not
  compilation.

Common environment:

```sh
export RJ_ROCM=/home/benoit/workspace/TheRock-build/dist/rocm
export RJ_HOOK=/home/benoit/workspace/TheRock/rocm-systems/emulation/rocjitsu/build/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so
export ROCM_PATH=$RJ_ROCM
export HIP_PATH=$RJ_ROCM
export LD_LIBRARY_PATH=$RJ_ROCM/lib
export HSA_TOOLS_LIB=$RJ_HOOK
```

## Build And Unit Tests

Build:

```sh
cmake --build /home/benoit/workspace/TheRock/rocm-systems/emulation/rocjitsu/build \
  --target rocjitsu_tests rocjitsu_dbi_hooks
```

Focused ConSan unit/synthetic tests:

```sh
ROCM_PATH=$RJ_ROCM HIP_PATH=$RJ_ROCM LD_LIBRARY_PATH=$RJ_ROCM/lib \
/home/benoit/workspace/TheRock/rocm-systems/emulation/rocjitsu/build/tests/rocjitsu_tests \
  '--gtest_filter=ConSan.*:ConSanMoi.*:InstructionBuilder.*'
```

Known local result:

- Full non-benchmark `rocjitsu_tests`: 1311/1311 passed after R1C.
- R1C spill encoder/layout/descriptor focus: 36/36 passed.

Focused gfx1201 spill hardware smoke:

```sh
ctest --test-dir /home/benoit/workspace/TheRock/rocm-systems/emulation/rocjitsu/build \
  -R '^ConSanSpillHipTest.Gfx1201VgprScratchRoundTrip$' \
  --output-on-failure
```

Known local result:

- 1/1 passed. The kernel has a 32-byte fixed private segment and executes the
  same address-free `scratch_store_b32` / `scratch_load_b32` encodings emitted
  by the R1C backend around a deliberately clobbered live VGPR.

## SuperCollider Coverage

Broad IREE e2e compatibility:

```sh
HSA_TOOLS_LIB=$RJ_HOOK \
ROCM_PATH=$RJ_ROCM HIP_PATH=$RJ_ROCM LD_LIBRARY_PATH=$RJ_ROCM/lib \
RJ_CONSAN_FLAVOR=supercollider \
RJ_CONSAN_REQUIRE_PATCH=1 \
ctest --test-dir /home/benoit/workspace/iree-build \
  -R '^iree/tests/e2e/.*(rocm_hip|rocm-rocm)' \
  --parallel 8 --output-on-failure
```

Known local result:

- IREE e2e broad sweep: 209/209 passed on `gfx1201`.

hip-moi compatibility:

```sh
HSA_TOOLS_LIB=$RJ_HOOK \
ROCM_PATH=$RJ_ROCM HIP_PATH=$RJ_ROCM LD_LIBRARY_PATH=$RJ_ROCM/lib \
RJ_CONSAN_FLAVOR=supercollider \
RJ_CONSAN_CHECK_TRAP_MODE=lds \
ctest --test-dir /home/benoit/workspace/hip-moi-build \
  --parallel 8 --output-on-failure
```

Known local result:

- hip-moi full suite: 189/189 passed on `gfx1201` with
  `RJ_CONSAN_CHECK_TRAP_MODE=lds`.
- The default `RJ_CONSAN_CHECK_TRAP_MODE=all` is not currently the broad
  hip-moi compatibility recipe; it has hit ambiguous flat/local-cave cases.

## MOI Coverage

Use `RJ_CONSAN_FLAVOR=moi` plus one explicit engine:

```sh
RJ_CONSAN_MOI_ENGINE=record_replay
RJ_CONSAN_MOI_ENGINE=sampled
RJ_CONSAN_MOI_ENGINE=inline_shadow
```

### MOI Record/Replay

Broad IREE compatibility:

```sh
HSA_TOOLS_LIB=$RJ_HOOK \
ROCM_PATH=$RJ_ROCM HIP_PATH=$RJ_ROCM LD_LIBRARY_PATH=$RJ_ROCM/lib \
RJ_CONSAN_FLAVOR=moi \
RJ_CONSAN_MOI_ENGINE=record_replay \
RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE=65536 \
RJ_CONSAN_TMP_VGPR=104 \
RJ_CONSAN_MAX_PATCHES=4 \
ctest --test-dir /home/benoit/workspace/iree-build \
  -R '^iree/tests/e2e/.*(rocm_hip|rocm-rocm)' \
  --parallel 8 --output-on-failure
```

Known local result:

- IREE e2e broad sweep: 209/209 passed on `gfx1201`.

Guarded TileAndFuse non-vacuity run:

```sh
HSA_TOOLS_LIB=$RJ_HOOK \
ROCM_PATH=$RJ_ROCM HIP_PATH=$RJ_ROCM LD_LIBRARY_PATH=$RJ_ROCM/lib \
RJ_CONSAN_FLAVOR=moi \
RJ_CONSAN_MOI_ENGINE=record_replay \
RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE=65536 \
RJ_CONSAN_TMP_VGPR=104 \
RJ_CONSAN_MAX_PATCHES=4 \
RJ_CONSAN_REQUIRE_PATCH=1 \
RJ_CONSAN_MOI_REQUIRE_RECORDS=1 \
ctest --test-dir /home/benoit/workspace/iree-build \
  -R '^iree/tests/e2e/matmul/e2e_matmul_rocm_.*large_rdna4_tileandfusewmma.*_rocm_hip$' \
  --parallel 8 --output-on-failure
```

Known local result:

- TileAndFuse guarded subset: 5/5 passed on `gfx1201`.

### MOI Sampled

Broad IREE compatibility:

```sh
HSA_TOOLS_LIB=$RJ_HOOK \
ROCM_PATH=$RJ_ROCM HIP_PATH=$RJ_ROCM LD_LIBRARY_PATH=$RJ_ROCM/lib \
RJ_CONSAN_FLAVOR=moi \
RJ_CONSAN_MOI_ENGINE=sampled \
RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE=65536 \
RJ_CONSAN_TMP_VGPR=104 \
RJ_CONSAN_MAX_PATCHES=4 \
ctest --test-dir /home/benoit/workspace/iree-build \
  -R '^iree/tests/e2e/.*(rocm_hip|rocm-rocm)' \
  --parallel 8 --output-on-failure
```

Known local result:

- IREE e2e broad sweep: 209/209 passed on `gfx1201`.

Guarded TileAndFuse non-vacuity run:

```sh
HSA_TOOLS_LIB=$RJ_HOOK \
ROCM_PATH=$RJ_ROCM HIP_PATH=$RJ_ROCM LD_LIBRARY_PATH=$RJ_ROCM/lib \
RJ_CONSAN_FLAVOR=moi \
RJ_CONSAN_MOI_ENGINE=sampled \
RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE=65536 \
RJ_CONSAN_TMP_VGPR=104 \
RJ_CONSAN_MAX_PATCHES=4 \
RJ_CONSAN_REQUIRE_PATCH=1 \
RJ_CONSAN_MOI_REQUIRE_RECORDS=1 \
ctest --test-dir /home/benoit/workspace/iree-build \
  -R '^iree/tests/e2e/matmul/e2e_matmul_rocm_.*large_rdna4_tileandfusewmma.*_rocm_hip$' \
  --parallel 8 --output-on-failure
```

Known local result:

- TileAndFuse guarded subset: 5/5 passed on `gfx1201`.

### MOI Inline Shadow

Targeted TileAndFuse run:

```sh
HSA_TOOLS_LIB=$RJ_HOOK \
ROCM_PATH=$RJ_ROCM HIP_PATH=$RJ_ROCM LD_LIBRARY_PATH=$RJ_ROCM/lib \
RJ_CONSAN_FLAVOR=moi \
RJ_CONSAN_MOI_ENGINE=inline_shadow \
RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE=262144 \
RJ_CONSAN_TMP_VGPR=240 \
RJ_CONSAN_MOI_INIT_OWNER_EPOCH=1 \
RJ_CONSAN_MOI_OWNER_SOURCE=hw_id \
RJ_CONSAN_MOI_OWNER_SGPR=100 \
RJ_CONSAN_MOI_OWNER_VGPR=250 \
RJ_CONSAN_MOI_EPOCH_VGPR=251 \
RJ_CONSAN_MAX_PATCHES=1 \
RJ_CONSAN_REQUIRE_PATCH=1 \
RJ_CONSAN_MOI_REQUIRE_RECORDS=1 \
ctest --test-dir /home/benoit/workspace/iree-build \
  -R '^iree/tests/e2e/matmul/e2e_matmul_rocm_.*rdna4_tileandfusewmma.*rocm_hip$' \
  --parallel 8 --output-on-failure
```

Known local result:

- TileAndFuse guarded subset: 5/5 passed on `gfx1201`.
- Broad IREE e2e inline-shadow is not yet clean. A broad sweep reached late
  tests but timed out in `check_rocm_hip_scan_configured.mlir` and then hung in
  a softmax test. Treat this as an open MOI broad-readiness bug.

## rocjitsu-test-corpus

Current local status:

- `gfx1201` CTS corpus under SuperCollider: 59/59 passed.
- IREE corpus: compile-only cases passed, but runtime cases were blocked by
  the corpus using an `iree-run-module` binary without HIP driver support.
- Kernels corpus: configure was blocked by a `hipblas` dependency missing from
  the local TheRock ROCm dist. A hip-matmul-only config or adding hipBLAS would
  unblock this.

## Architecture Matrix

| Architecture | Local live-GPU coverage |
| --- | --- |
| `gfx1201` | Yes. Results above are current local coverage. |
| `gfx942` | Planned target, not validated on this machine. |
| `gfx950` | Planned target, not validated on this machine. |
| `gfx1250` | Planned target, not validated on this machine. |
