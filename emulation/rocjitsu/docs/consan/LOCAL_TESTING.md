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
  '--gtest_filter=ConSanResourcePlan.*:ConSanMoi.*:SpillManager.*:InstructionBuilder.*'
```

Known local result:

- Full `rocjitsu_tests`, including registered benchmark-style tests: 1418/1418
  passed after the R1D implementation.
- Current resource/MOI/spill-manager focus: 148/148 passed after enabling
  zero-to-nonzero dispatch scratch.

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

Focused MOI spill vertical and live regression slice:

```sh
ctest --test-dir /home/benoit/workspace/TheRock/rocm-systems/emulation/rocjitsu/build \
  -R '^(ConSanSpillHipTest|ConSanMoiHipTest\.)' \
  --parallel 8 --output-on-failure
```

Known local result:

- 15/15 passed. This includes forced-spill record/replay and sampled tests whose
  original kernel private size is zero. The hook raises the patched kernel's
  dispatch-private size (to 12 and 20 bytes respectively); both tests keep
  eight values live across the patched LDS access, verify every value after
  restoration, and require a visible MOI record/entry.
- The forced tier and kernel selector are internal CTest controls, not public
  ConSan configuration.

Automatic inline-shadow persistent VGPR checks:

```sh
ctest --test-dir /home/benoit/workspace/TheRock/rocm-systems/emulation/rocjitsu/build \
  -R '^ConSanInlineShadowTest\.Dbi(ReportsCrossWaveRace|BarrierEpochOrdersCrossWaveAccesses)$' \
  --parallel 2 --output-on-failure
```

Known local result:

- 2/2 passed without owner/epoch VGPR numbers or
  `RJ_CONSAN_MOI_INIT_OWNER_EPOCH`. Scratch and the diagnostic EXEC-save SGPR
  remain explicit in this R1E slice.

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

R1D descriptor-pressure check:

- `check_rocm_hip_scan_configured.mlir` completed 5/5 checks under a 30-second
  timeout with forced spill planning scoped to the large
  `scan_64x256xf32` kernel.
- That kernel has 640 DS operations currently classified as unsupported access
  kinds, so it reaches a precise pre-allocation blocker rather than a spill
  patch. There was no hang and no silent high-VGPR borrowing.
- Live forced-spill coverage uses a kernel compiled with zero private bytes.
  Record/replay and sampled instrumentation both bind the patched kernel symbol
  to its loaded kernel object and raise the AQL dispatch-private size before
  execution.

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
