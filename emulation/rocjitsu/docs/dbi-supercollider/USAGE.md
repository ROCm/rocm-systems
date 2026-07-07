# DBI SuperCollider MVP Usage

This is the current runbook for the SuperCollider-style LDS race MVP in
rocJITsu. The mode is instrumentation-only on the local RDNA4 `gfx1201` device;
it is not meant to translate the code object to another architecture.

## Build

From the `rocm-systems` tree:

```sh
cmake --build emulation/rocjitsu/build --target rocjitsu_dbi_hooks
cmake --build emulation/rocjitsu/build --target rocjitsu_tests
```

The HSA tools library built by the first command is:

```text
emulation/rocjitsu/build/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so
```

## How The Hook Is Loaded

Use the HSA tools path:

```sh
env \
  HSA_TOOLS_LIB=/home/benoit/workspace/TheRock/rocm-systems/emulation/rocjitsu/build/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so \
  RJ_DBI_SUPERCOLLIDER=1 \
  RJ_DBI_LOG=1 \
  RJ_DBI_SC_DELAY=2 \
  ./app
```

`HSA_TOOLS_LIB` is the same HSA-level mechanism used by other ROCm tools:

- `projects/rocr-runtime/runtime/hsa-runtime/core/util/flag.h` reads
  `HSA_TOOLS_LIB` in the ROCr runtime.
- `projects/rocprofiler/doc/ROCProfiler_V1_API_spec.md` documents
  `HSA_TOOLS_LIB` as the library loaded by the HSA runtime for rocprofiler.
- `projects/rocr-debug-agent/docs/how-to/user-guide.rst` uses
  `HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2 ./my_program`.
- The HSA API surface being wrapped is documented in
  `projects/rocr-runtime/runtime/hsa-runtime/inc/hsa.h` and
  `projects/rocr-runtime/runtime/hsa-runtime/inc/hsa_api_trace.h`; the relevant
  calls are the code-object reader creation APIs and
  `hsa_executable_load_agent_code_object`.

The MVP uses this HSA tools loader path. A separate waitcheck-style
`LD_PRELOAD` path is not needed for the current DBI SuperCollider work.

## Environment Variables

- `RJ_DBI_SUPERCOLLIDER=1`: enable the DBI SuperCollider mode.
- `RJ_DBI_LOG=1`: emit compact info logs. Higher numeric values enable more
  verbose per-kernel diagnostics.
- `RJ_DBI_DUMP_DIR=/tmp/rj-dbi-dump`: write original memory-backed code objects,
  and patched code objects when a patch is produced, as `.hsaco` files for
  `llvm-readelf` / `llvm-objdump` inspection.
- `RJ_DBI_SC_CHECK_TRAP_MODE=lds|flat`: select the primary check/trap
  instrumentation path. The default is `lds`, so normal native `ds_*` LDS
  instrumentation only needs `RJ_DBI_SUPERCOLLIDER=1`. Set
  `RJ_DBI_SC_CHECK_TRAP_MODE=flat` for likely group/LDS `flat_*`
  instrumentation.
- `RJ_DBI_SC_DELAY=N`: configure the delay between the original LDS/flat access
  and the duplicate/readback check.
- `RJ_DBI_SC_DELAY_MODE=nop|sleep|sleep_var`: choose the delay encoding. The
  default `nop` mode emits `N` simple `s_nop 0` instructions. The `sleep` mode
  emits one `s_sleep N` instruction when `N > 0`; `N` must fit the 16-bit
  `s_sleep` immediate. The `sleep_var` mode emits one `s_sleep_var` instruction
  when `N > 0`.
- `RJ_DBI_SC_DELAY_VAR_SSRC=N`: scalar source operand encoding for
  `RJ_DBI_SC_DELAY_MODE=sleep_var`. The default is `106`, the RDNA4 `vcc_lo`
  operand encoding already preserved by the injected native-DS compare path.
- `RJ_DBI_SC_MAX_PATCHES=N`: bound how many native-DS check/trap sites can be
  instrumented in a single code object. The default is `1`. The first
  multi-site implementation selects non-overlapping padded sites and distinct
  reachable local NOP caves in file order.
- `RJ_DBI_SC_TMP_VGPR=N`: force a scratch VGPR. Use this for hand-shaped tests
  whose kernel descriptor reserves the selected VGPR.
- `RJ_DBI_SC_REQUIRE_PATCH=1`: test guard. If a code object has a supported MVP
  site but no patch is emitted, fail the load. Code objects with no supported
  sites still pass through.
- `RJ_DBI_SC_FAULT_DROP_BARRIER=1`: demo-only synchronization fault injection.
  After the primary proof/instrumentation pass, rewrite one decoded
  `s_barrier*` instruction to `s_nop 0`.
- `RJ_DBI_SC_FAULT_BARRIER_INDEX=N`: select which decoded barrier to drop in a
  code object. The default is `0`.
- `RJ_DBI_FAIL_CLOSED=1`: debugging guard that rejects unsupported kernels
  rather than quietly skipping them.

The primary check/trap paths are selected by `RJ_DBI_SC_CHECK_TRAP_MODE`, not
composed. The native LDS path is the default. The flat path is a separate pass
because composing native DS and flat/VFLAT rewriting would need a shared
patch-range reservation plan and patching from already-modified bytes. Barrier
fault injection is composable because it runs after the selected primary mode.
The bring-up-only probes below are explicit debug overrides of the default
check/trap path.

Additional bring-up-only probes also exist:

- `RJ_DBI_SC_PROBE_NOP=1`
- `RJ_DBI_SC_PROBE_TRAMPOLINE_NOP=1`
- `RJ_DBI_SC_PROBE_ENDPGM=1`
- `RJ_DBI_SC_PROBE_LDS_ENDPGM=1`
- `RJ_DBI_SC_PROBE_FLAT_TRAP=1`

Those are useful for debugging code-object mutation but are not the race-check
mode. `RJ_DBI_SC_PROBE_TRAMPOLINE_NOP=1` skips ROCclr runtime helper-only code
objects and code objects with no supported DBI candidate sites.
`RJ_DBI_SC_PROBE_FLAT_TRAP=1` rewrites likely group/LDS `flat_load/store` sites
in place as `s_trap 0; s_nop 0; s_nop 0`; it is intentionally destructive and
should make an executing patched site fail.

## Coverage Signal

With `RJ_DBI_LOG=1`, look for:

```text
SuperCollider DBI summary reader=... kernels=... candidates=... skips=... rejects=... supported_lds_sites=... flat_sites=... patches=... modified=...
```

Interpretation:

- `modified=true` and `patches>0`: this code object was actually patched.
- `inline-barrier-nop-rewrite`: demo-only sync-fault injection rewrote one
  selected decoded barrier to `s_nop 0`. It intentionally skips ROCclr
  runtime-helper-only code objects.
- `supported_lds_sites=0`: the current native-DS MVP found no supported
  instructions in this code object.
- `local-cave-lds-load-check-trap` / `local-cave-lds-store-check-trap` patch
  logs mean a compact native DS site was redirected through uncovered local NOP
  slack and returned to the original fallthrough. The focused IREE WMMA
  ROCm/HIP e2e object now patches one `ds_load_2addr_b64` site this way and
  passes under `RJ_DBI_SC_REQUIRE_PATCH=1`.
- `flat_sites>0`: flat/generic memory sites were decoded and logged at
  `RJ_DBI_LOG=2`.
- `function_flat_maybe_group_hints>0`: helper-function flat sites are likely
  LDS/shared accesses. The destructive flat proof can patch these with
  `RJ_DBI_SC_PROBE_FLAT_TRAP=1`. Padded likely group flat loads and stores can
  be checked with `RJ_DBI_SC_CHECK_TRAP_MODE=flat`; ordinary unpadded
  hip-moi helper sites can now be checked one selected site at a time through a
  conservative local NOP cave when one is reachable.
- `local-cave-flat-load-check-trap` / `local-cave-flat-store-check-trap` patch
  logs mean an unpadded flat helper site was redirected through uncovered local
  NOP slack and returned to the original fallthrough. The focused hip-moi
  `NoPipelineProd16x8` object now patches one such site and passes cleanly.
- `skips>0` with `modified=false`: the hook observed the code object but chose
  pass-through for policy or coverage reasons.
- `rejects>0`: the hook found something it considered unsafe or unsupported.
- Proof-mode warnings such as `skipped ROCclr runtime helper code object` and
  `skipped code object without supported DBI candidate sites` mean the proof
  mode intentionally avoided an expensive or unsafe trampoline search.
- For `RJ_DBI_SC_CHECK_TRAP_MODE=flat`, an unpadded skip warning now includes
  `supported_candidates=...`, `scratchable_candidates=...`, and
  `max_observed_padding_words=...`, and
  `append_cave_reachable_candidates=...`, `uncovered_nop_caves=...`,
  `max_uncovered_nop_cave_words=...`, and
  `local_cave_reachable_candidates=...` so the local-cave/code-growth gap is
  measurable. In the focused hip-moi `NoPipelineProd16x8` object, appended
  end-of-text caves are not directly reachable, but conservative uncovered NOP
  caves are reachable for all 31 likely group flat helper candidates observed so
  far.

The broad hip-moi matmul kernels observed so far use flat/generic and scratch
forms rather than native `ds_load/store_*`. The focused IREE e2e kernels do
expose compact native DS sites. The current hook can identify and destructively
trap likely group/LDS flat helper-function sites. The non-destructive
race-checking paths currently cover padded native DS sites, one selected compact
native DS site when a local cave is reachable, padded likely group flat sites,
and one selected unpadded likely group flat site when a conservative local NOP
cave is reachable.

## What The MVP Instruments

The current live-safe path instruments native non-atomic LDS instructions with
enough trailing `s_nop 0` padding to fit an in-place sequence:

- `ds_load_b32`
- `ds_load_b64`
- `ds_load_b128`
- `ds_load_2addr_b32`
- `ds_load_2addr_b64`
- `ds_load_2addr_stride64_b32`
- `ds_load_2addr_stride64_b64`
- `ds_store_b32`
- `ds_store_b64`
- `ds_store_b128`
- likely group/LDS `flat_load_b32`
- likely group/LDS `flat_load_b64`
- likely group/LDS `flat_load_b128`
- likely group/LDS `flat_store_b32`
- likely group/LDS `flat_store_b64`
- likely group/LDS `flat_store_b128`

For loads, the patch duplicates the load after the requested NOP delay and traps
if the two values differ. For stores, the patch reads back the stored LDS value
after the requested NOP delay and traps if it differs from the original stored
value. The injected compares preserve `vcc_lo` by saving it to a
liveness-selected SGPR. For native DS and likely group flat helper sites, the
patch may either use trailing padding or redirect one selected site through a
conservative uncovered local NOP cave.

## Repeatable Local Tests

rocJITsu positive/negative regression:

```sh
ctest --test-dir /home/benoit/workspace/TheRock/rocm-systems/emulation/rocjitsu/build \
  -R 'DbiSuperColliderLdsTest' \
  --parallel 1 \
  --output-on-failure
```

Selected hip-moi smoke:

```sh
env \
  HSA_TOOLS_LIB=/home/benoit/workspace/TheRock/rocm-systems/emulation/rocjitsu/build/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so \
  RJ_DBI_SUPERCOLLIDER=1 \
  RJ_DBI_LOG=1 \
  RJ_DBI_SC_DELAY=2 \
  ctest --test-dir /home/benoit/workspace/hip-moi-build \
    -R 'JakubRdna4MatmulReference|HipMoiRdna4Pingpong' \
    --parallel 8 \
    --output-on-failure
```

Focused hip-moi flat local-cave smoke:

```sh
env \
  HSA_TOOLS_LIB=/home/benoit/workspace/TheRock/rocm-systems/emulation/rocjitsu/build/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so \
  RJ_DBI_SUPERCOLLIDER=1 \
  RJ_DBI_LOG=1 \
  RJ_DBI_SC_CHECK_TRAP_MODE=flat \
  RJ_DBI_SC_DELAY=1 \
  ctest --test-dir /home/benoit/workspace/hip-moi-build \
    -R NoPipelineProd16x8 \
    --parallel 1 \
    --output-on-failure -V
```

Focused IREE ROCM/ROCDL smoke:

```sh
env \
  HSA_TOOLS_LIB=/home/benoit/workspace/TheRock/rocm-systems/emulation/rocjitsu/build/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so \
  RJ_DBI_SUPERCOLLIDER=1 \
  RJ_DBI_LOG=1 \
  RJ_DBI_SC_DELAY=2 \
  ctest --test-dir /home/benoit/workspace/iree-build \
    -R 'iree/compiler/plugins/target/ROCM/test/smoketest.mlir.test|iree/compiler/plugins/target/ROCM/test/smoketest_hsaco.mlir.test|iree/compiler/Codegen/LLVMGPU/test/convert_to_rocdl_gfx1201.mlir.test|iree/compiler/Codegen/LLVMGPU/test/ROCDL/config_tile_and_fuse_gfx1201.mlir.test|iree/compiler/Codegen/LLVMGPU/test/ROCDL/pipeline_full_smoketests.mlir.test' \
    --parallel 8 \
    --output-on-failure
```

Focused IREE native-DS local-cave smoke:

```sh
env \
  HSA_TOOLS_LIB=/home/benoit/workspace/TheRock/rocm-systems/emulation/rocjitsu/build/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so \
  LD_LIBRARY_PATH=/home/benoit/workspace/TheRock-build/dist/rocm/lib:$LD_LIBRARY_PATH \
  RJ_DBI_SUPERCOLLIDER=1 \
  RJ_DBI_LOG=1 \
  RJ_DBI_SC_DELAY=2 \
  RJ_DBI_SC_REQUIRE_PATCH=1 \
  ctest --test-dir /home/benoit/workspace/iree-build \
    -R 'rocm_hip_wmma_matmul_f16_wmma_matmul_f16' \
    --parallel 1 \
    --output-on-failure
```

Keep GPU test fanout near 8.

## Limitations

- Flat/generic LDS provenance is conservative. The hook can identify likely
  group/LDS helper-function flat sites and destructively trap them. The
  non-destructive check/trap path can now patch one selected ordinary unpadded
  hip-moi helper site through a local NOP cave, but it does not yet instrument
  every eligible flat site.
- No shadow memory and no non-trapping report buffer yet; `s_trap` is the MVP
  signal.
- Same-value races and unlucky schedules can be missed.
- Appended trampoline/code-growth patching remains bring-up-only. Proof NOP
  mode is gated to candidate-bearing non-ROCclr code objects; use the padded or
  local-cave check/trap paths for MVP race checks.
- Scratch selection is liveness-based and fail-closed. The patcher does not grow
  the kernel's VGPR allocation metadata or spill arbitrary live registers, so a
  site with no free scratch VGPR/SGPR run is skipped.
- Atomic, fence-heavy, global-memory, async-copy, and barrier-epoch cases are
  unsupported and should be skipped or rejected with diagnostics.
