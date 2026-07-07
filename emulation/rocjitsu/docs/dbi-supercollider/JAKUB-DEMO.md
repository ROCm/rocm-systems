# Jakub DBI Demo Notes: IREE e2e

Date: 2026-07-07

## Summary

rocJITsu intercepts IREE ROCm/HIP GPU code-object loads through the HSA tools
path, inspects final native RDNA4 / `gfx1201` machine code, and rewrites
selected LDS instructions before the code object is loaded.

The main demo runs selected IREE e2e tests while requiring rocJITsu to actually
rewrite instrumentable LDS code. A 10-test IREE linalg/matmul/StableHLO set
passes under that requirement. The clearest concrete example is one IREE WMMA
kernel where rocJITsu rewrites a compact `ds_load_2addr_b64` into a branch to a
local NOP cave containing:

- the original LDS load,
- a delay,
- a duplicate LDS load,
- value comparisons,
- `s_trap 0` on mismatch.

This MVP uses trap-first reporting, not full SuperCollider race reporting. It
demonstrates the core DBI substrate that we need: HSA-level interception,
final-ISA inspection, LDS instruction selection, binary patching, and
SuperCollider-style redundant access checks.

The verification mechanisms make the DBI work visible, instead of merely relying
on already-correct IREE kernels to pass.

## Vocabulary

- A **code object** is the loaded GPU binary.
- An **LDS site** is a final machine instruction that accesses AMD LDS/shared
  memory, for example a `ds_load_*` or `ds_store_*` instruction.
- A **supported LDS site** is an LDS site whose opcode and layout this MVP
  knows how to rewrite.
- A **DBI patch** is a byte-level rewrite of that final GPU binary at load time.
  In this demo, the patch replaces an LDS instruction with a branch to a small
  probe sequence, then branches back to the original fallthrough.
- **Requiring a patch** means setting `RJ_DBI_SC_REQUIRE_PATCH=1`. It is a test
  guard: if a code object has supported LDS sites but rocJITsu cannot patch any
  of them, loading that code object fails.

## Verification Mechanisms

IREE e2e kernels are supposed to be correct. A passing test alone does not prove
that rocJITsu decoded anything, found any LDS instruction, or modified the
binary. A passing test could be a vacuous result: the hook was present, but the
final GPU code ran unchanged.

The demo uses two separate mechanisms to make that ambiguity visible.

### `RJ_DBI_SC_REQUIRE_PATCH=1`

This guard tells rocJITsu to fail loudly instead of silently skipping a code
object that looked instrumentable.

With `RJ_DBI_SC_REQUIRE_PATCH=1`, rocJITsu rejects a code object if all three
of these are true:

- the code object contains an LDS instruction that this MVP recognizes,
- instrumentation mode is enabled for that instruction kind,
- rocJITsu cannot actually rewrite any such instruction.

A code object is allowed to load in either of two cases:

- it has no supported LDS sites, so rocJITsu lets it pass through, or
- it has supported LDS sites, and rocJITsu successfully rewrites at least one
  of them.

This is the main non-vacuity check for the positive demo. If a selected IREE
test passes while this guard is enabled, then code objects that contained
supported LDS instructions did receive at least one DBI patch.

### `RJ_DBI_SC_FAULT_DROP_BARRIER=1`

This diagnostic takes the opposite approach: instead of proving that a correct
test still passes after instrumentation, it deliberately makes one loaded code
object wrong.

With `RJ_DBI_SC_FAULT_DROP_BARRIER=1`, rocJITsu selects one decoded
`s_barrier*` instruction and rewrites it to `s_nop 0`. That is not the
SuperCollider algorithm; it is a deliberately destructive synchronization
fault. Its purpose is to answer a narrower question: can rocJITsu find and
modify synchronization instructions in the final native IREE GPU binary in a
way that has an observable runtime effect?

In the WMMA diagnostic, the control run passes, while the
barrier-to-NOP run reaches the IREE BF16 WMMA kernel and then times out. That
does not constitute a polished race report, but it does show that the DBI path
is not merely observing code-object loads. It is changing final `gfx1201`
instructions in a way the workload can feel.

## Setup Assumptions

This note is only about IREE ROCm/HIP e2e test cases. It assumes:

- `ROCM_SYSTEMS_DIR`: a checkout of `ROCm/rocm-systems` on
  `dbi-supercollider`.
- `ROCJITSU_BUILD_DIR`: an already-built rocJITsu CMake build directory from
  that checkout.
- `IREE_BUILD_DIR`: an already-built IREE CMake build directory with CTest
  metadata and HIP runtime tests available.
- `ROCM_DIST_DIR`: a ROCm installation or build distribution whose `lib/`
  directory contains `libamdhip64.so`, if that library is not already visible
  through the dynamic loader path.

No IREE configuration steps are shown here. Point `IREE_BUILD_DIR` at an
existing build directory.

Common environment:

```sh
export ROCM_SYSTEMS_DIR=/path/to/rocm-systems
export ROCJITSU_BUILD_DIR="$ROCM_SYSTEMS_DIR/emulation/rocjitsu/build"
export IREE_BUILD_DIR=/path/to/iree-build
export ROCM_DIST_DIR=/path/to/rocm

export HSA_TOOLS_LIB="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"
export LD_LIBRARY_PATH="$ROCM_DIST_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RJ_DBI_SUPERCOLLIDER=1
export RJ_DBI_LOG=1
```

This is native `gfx1201` instrumentation through the HSA tools hook. rocJITsu
is not translating the code object to another architecture.

## Primary Patch Mode Selection

The SuperCollider DBI hook has one primary check/trap selector:

- unset, or `RJ_DBI_SC_CHECK_TRAP_MODE=lds`: native `ds_*` LDS
  instrumentation,
- `RJ_DBI_SC_CHECK_TRAP_MODE=flat`: likely group/LDS `flat_*`
  instrumentation.

The native LDS path is the default because it is the IREE demo path. The flat
path is still available as a separate pass. The current patcher does not compose
native DS and flat/VFLAT rewriting in one pass because that needs a shared
patch-range reservation plan and patching from already-modified bytes. Barrier
fault injection is different: `RJ_DBI_SC_FAULT_DROP_BARRIER=1` runs after the
selected primary mode and is intentionally composable with native LDS
check/trap.

The IREE demo uses the native LDS mode. Flat/VFLAT support is documented later as
a separate path.

## Compatibility Smoke: Full RDNA4 Matmul e2e

This broad compatibility check verifies that the HSA tools hook sits underneath
IREE's HIP HAL path for the complete RDNA4 ROCm/HIP matmul e2e set exposed by
this build:

```sh
ctest --test-dir "$IREE_BUILD_DIR" \
  -R 'iree/tests/e2e/matmul/.*rocm_hip' \
  --parallel 8 \
  --output-on-failure
```

Observed result:

```text
100% tests passed, 0 tests failed out of 13
```

Covered tests include f16/f8/i8 WMMA, TileAndFuse WMMA, transposed-B variants,
and RDNA4 DT f16/f8/i8 matmul tests.

Representative DBI inventory from IREE e2e runs:

```text
rocm_specific WMMA: kernels=2 candidates=2 supported_lds_sites=32
  _wmma_matmul_f16_identity... lds_reads=72 lds_writes=8 barriers=6
  _wmma_matmul_bf16_identity... lds_reads=72 lds_writes=8 barriers=6

linalg narrow matmuls: kernels=6 candidates=4 supported_lds_sites=24

f16 TileAndFuse matmul: supported_lds_sites=57
f16 TileAndFuse transposed-B matmul: supported_lds_sites=73
f8 TileAndFuse matmul: supported_lds_sites=41
i8 TileAndFuse transposed-B matmul: supported_lds_sites=41
```

The inventory confirms that IREE e2e provides compact native `ds_*` kernels and
concrete LDS-heavy runtime workloads under the hook.

## Main Demo: Require A Real LDS Patch

This is the main demo. It uses the default native LDS check/trap mode and sets
`RJ_DBI_SC_REQUIRE_PATCH=1`, so the run only passes if rocJITsu either finds no
supported LDS sites in a loaded code object or successfully rewrites at least
one supported LDS site before loading it.

```sh
export RJ_DBI_SC_DELAY_MODE=sleep
export RJ_DBI_SC_DELAY=1
export RJ_DBI_SC_MAX_PATCHES=4
export RJ_DBI_SC_REQUIRE_PATCH=1

ctest --test-dir "$IREE_BUILD_DIR" \
  -R 'check_rocm_hip_narrow_n_matmuls|e2e_matmul_rocm_f16_large_rdna4_tileandfusewmma(_tb)?_rocm_hip|e2e_matmul_rocm_f8E4M3FN_large_rdna4_tileandfusewmma_rocm_hip|e2e_matmul_rocm_f8e4M3FN_large_rdna4_tileandfusewmma_tb_rocm_hip|e2e_matmul_rocm_i8_large_rdna4_tileandfusewmma_tb_rocm_hip|e2e_matmul_rdna4_dt_f8E4M3FN_rocm_hip|check_rocm_hip_stream_dot' \
  --parallel 1 \
  --output-on-failure
```

Observed result:

```text
100% tests passed, 0 tests failed out of 10
```

| Test | Result | What the pass proves |
| --- | --- | --- |
| `check_rocm_hip_narrow_n_matmuls.mlir` | PASS | Multi-site local-cave LDS instrumentation no longer trips an illegal instruction |
| `e2e_matmul_rocm_f16_large_rdna4_tileandfusewmma_rocm_hip` | PASS | Compact TileAndFuse kernels can be patched through the appended `.text` cave fallback |
| `e2e_matmul_rocm_f16_large_rdna4_tileandfusewmma_tb_rocm_hip` | PASS | The transposed-B TileAndFuse local-cave case passes with descriptor-bounded scratch selection |
| `e2e_matmul_rocm_f8E4M3FN_large_rdna4_tileandfusewmma_rocm_hip` | PASS | The same DBI mode covers another TileAndFuse datatype variant |
| `e2e_matmul_rocm_f8e4M3FN_large_rdna4_tileandfusewmma_tb_rocm_hip` | PASS | The transposed-B TileAndFuse f8 variant passes under the same guard |
| `e2e_matmul_rocm_i8_large_rdna4_tileandfusewmma_tb_rocm_hip` | PASS | The TileAndFuse i8 transposed-B variant loads and executes under the hook |
| `e2e_matmul_rdna4_dt_f8E4M3FN_rocm_hip` | PASS | The DT f8 store/readback path no longer trips an illegal instruction |
| `check_rocm_hip_stream_dot.mlir` | PASS | StableHLO dot remains correct after the patcher skips an unsafe descriptor-edge B64 duplicate-load site and patches another supported site |
| `check_rocm_hip_stream_dot_bf16.mlir` | PASS | The StableHLO BF16 dot variant is compatible with the patch-required DBI mode |
| `check_rocm_hip_stream_dot_general.mlir` | PASS | The broader StableHLO dot_general path is compatible with the same hook/configuration |

A compact WMMA example used in the patch anatomy logs:

```text
[rocjitsu-dbi-hooks] SuperCollider DBI proof patch ... kind=local-cave-lds-load-check-trap anchor=0x3cc trampoline=0x810 original_size=8 scratch_vgpr=104
```

## Delay And Multi-Site Controls

The duplicate/check sequence supports three delay encodings:

- `RJ_DBI_SC_DELAY_MODE=nop`: emit `RJ_DBI_SC_DELAY` `s_nop 0` instructions.
- `RJ_DBI_SC_DELAY_MODE=sleep`: emit one deterministic `s_sleep N` when
  `RJ_DBI_SC_DELAY=N` is nonzero.
- `RJ_DBI_SC_DELAY_MODE=sleep_var`: emit one `s_sleep_var` from a scalar source
  operand when `RJ_DBI_SC_DELAY` is nonzero. The default source is RDNA4
  `vcc_lo` operand encoding 106.

The WMMA control also passes when the delay is encoded as `s_sleep` or
`s_sleep_var`. Example `sleep_var` run:

```sh
export RJ_DBI_SC_DELAY_MODE=sleep_var
export RJ_DBI_SC_DELAY=1
export RJ_DBI_SC_MAX_PATCHES=4
export RJ_DBI_SC_REQUIRE_PATCH=1

ctest --test-dir "$IREE_BUILD_DIR" \
  -R '^iree/tests/e2e/rocm_specific/check_rocm_hip_wmma_matmul_f16_wmma_matmul_f16\.mlir$' \
  --parallel 1 \
  --output-on-failure
```

Observed result:

```text
100% tests passed, 0 tests failed out of 1
Test #1914 ... Passed 0.24 sec
```

`RJ_DBI_SC_MAX_PATCHES=N` bounds native-DS multi-site instrumentation in a
single code object. Selection is file-ordered and non-overlapping, and uses at
most one selected site per local NOP cave.

## Patch Anatomy: WMMA `ds_load_2addr_b64`

The WMMA dump evidence can be regenerated with:

```sh
export RJ_DBI_DUMP_DIR=/tmp/rj-dbi-jakub-demo-20260707
```

That produces original and patched HSACO dumps. In the original code object,
the selected site is a compact native `ds_load_2addr_b64`:

```text
ds_load_u16_d16 v62, v30 offset:4928                       // 000000001EC4
ds_load_2addr_b64 v[68:71], v29 offset1:1                  // 000000001ECC
ds_load_2addr_b64 v[72:75], v29 offset0:4 offset1:5        // 000000001ED4
```

In the patched code object, that 8-byte site becomes a branch plus fill NOP:

```text
ds_load_u16_d16 v62, v30 offset:4928                       // 000000001EC4
s_branch 272                                               // 000000001ECC <...+0x810>
s_nop 0                                                    // 000000001ED0
ds_load_2addr_b64 v[72:75], v29 offset0:4 offset1:5        // 000000001ED4
```

The local cave contains the original load, delay, duplicate load to scratch
VGPRs, `dscnt` wait, VCC-preserving comparisons, traps on mismatch, and a
return branch:

```text
ds_load_2addr_b64 v[68:71], v29 offset1:1                  // 000000002310
s_nop 0                                                    // 000000002318
s_nop 0                                                    // 00000000231C
ds_load_2addr_b64 v[104:107], v29 offset1:1                // 000000002320
s_wait_dscnt 0x0                                           // 000000002328
s_mov_b32 s14, vcc_lo                                      // 00000000232C
v_cmp_ne_u32_e32 vcc_lo, v68, v104                         // 000000002330
s_cbranch_vccz 1                                           // 000000002334
s_trap 0                                                   // 000000002338
...
v_cmp_ne_u32_e32 vcc_lo, v71, v107                         // 000000002354
s_cbranch_vccz 1                                           // 000000002358
s_trap 0                                                   // 00000000235C
s_mov_b32 vcc_lo, s14                                      // 000000002360
s_branch 65243                                             // 000000002364 <...+0x3d4>
```

The only part controlled by `RJ_DBI_SC_DELAY_MODE` is the delay between the
original LDS access and the duplicate/readback LDS access. In the WMMA patch,
that means the instructions between:

```text
ds_load_2addr_b64 v[68:71], v29 offset1:1
```

and:

```text
ds_load_2addr_b64 v[104:107], v29 offset1:1
```

The effect is:

| Mode | Example settings | Delay emitted in the cave | Cave-size effect |
| --- | --- | --- | --- |
| `nop` | `RJ_DBI_SC_DELAY_MODE=nop`, `RJ_DBI_SC_DELAY=2` | `s_nop 0`; `s_nop 0` | one 32-bit word per requested NOP |
| `sleep` | `RJ_DBI_SC_DELAY_MODE=sleep`, `RJ_DBI_SC_DELAY=2` | `s_sleep 2` | one 32-bit word total when delay is nonzero |
| `sleep_var` | `RJ_DBI_SC_DELAY_MODE=sleep_var`, `RJ_DBI_SC_DELAY=1`, `RJ_DBI_SC_DELAY_VAR_SSRC=106` | `s_sleep_var <ssrc0=106>`; 106 is RDNA4 `vcc_lo` by default | one 32-bit word total when delay is nonzero |

The `sleep_var` examples show the source operand by encoding; a disassembler
may spell encoding 106 as `vcc_lo`.

The middle of the probe is the part that changes:

```text
nop mode, RJ_DBI_SC_DELAY=2:
  ds_load_2addr_b64 v[68:71], v29 offset1:1
  s_nop 0
  s_nop 0
  ds_load_2addr_b64 v[104:107], v29 offset1:1

sleep mode, RJ_DBI_SC_DELAY=2:
  ds_load_2addr_b64 v[68:71], v29 offset1:1
  s_sleep 2
  ds_load_2addr_b64 v[104:107], v29 offset1:1

sleep_var mode, default RJ_DBI_SC_DELAY_VAR_SSRC:
  ds_load_2addr_b64 v[68:71], v29 offset1:1
  s_sleep_var <ssrc0=106/vcc_lo>
  ds_load_2addr_b64 v[104:107], v29 offset1:1
```

Everything after the duplicate LDS load is the same: `s_wait_dscnt 0`, save
`vcc_lo`, compare original VGPRs against scratch VGPRs, trap on mismatch,
restore `vcc_lo`, and branch back.

Why this is the central evidence:

- The selected IREE WMMA site is a compact native `ds_load_2addr_b64`.
- That instruction represents two 64-bit LDS loads in one 8-byte instruction,
  so the check needs four scratch VGPR dwords and four comparisons.
- rocJITsu replaces the original site with `s_branch cave; s_nop 0`.
- The local cave executes the original LDS load, the requested delay, a
  duplicate LDS load into scratch VGPRs, `s_wait_dscnt 0`, value comparisons,
  and `s_trap 0` on mismatch.
- The injected compares preserve `vcc_lo` by saving it to a liveness-selected
  SGPR and restoring it before returning to the original fallthrough.

The run requires an actual patch, the patched IREE e2e module still passes, and
the patch is applied to native `gfx1201` code loaded through the HSA tools path.

## Sync-Fault Diagnostic: Drop One Barrier

Without touching the IREE compiler, rocJITsu can deliberately rewrite one
decoded `s_barrier*` instruction to `s_nop 0`:

```sh
export RJ_DBI_SC_DELAY=2
export RJ_DBI_SC_REQUIRE_PATCH=1
export RJ_DBI_SC_FAULT_DROP_BARRIER=1
export RJ_DBI_SC_FAULT_BARRIER_INDEX=0

ctest --test-dir "$IREE_BUILD_DIR" \
  -R '^iree/tests/e2e/rocm_specific/check_rocm_hip_wmma_matmul_f16_wmma_matmul_f16.mlir$' \
  --parallel 1 \
  --output-on-failure
```

Control run, with the same LDS check/trap mode but no barrier fault:

```text
100% tests passed, 0 tests failed out of 1
Test #1914 ... Passed 0.24 sec
```

Fault-injection run:

```text
[rocjitsu-dbi-hooks] DBI SuperCollider barrier fault skipped ROCclr runtime helper code object
[rocjitsu-dbi-hooks] DBI SuperCollider barrier fault rewrote s_barrier_signal in kernel:_wmma_matmul_bf16_identity_dispatch_0_matmul_128x128x128_bf16
[rocjitsu-dbi-hooks] SuperCollider DBI proof patch ... kind=local-cave-lds-load-check-trap anchor=0x3cc trampoline=0x810 original_size=8 scratch_vgpr=104
[rocjitsu-dbi-hooks] SuperCollider DBI proof patch ... kind=inline-barrier-nop-rewrite anchor=0xab4 trampoline=0xab4 original_size=4 scratch_vgpr=-
[       OK ] module.wmma_matmul_f16_identity (132 ms)
[ RUN      ] module.wmma_matmul_bf16_identity
***Timeout  60.11 sec
```

Interpretation:

- The ROCclr helper-only code object is explicitly skipped.
- The IREE WMMA code object receives the LDS check/trap patch.
- The same IREE code object also receives a barrier-to-NOP rewrite.
- The F16 test passes because `RJ_DBI_SC_FAULT_BARRIER_INDEX=0` selects the
  first barrier in the BF16 kernel in this code object ordering.
- The BF16 test then hangs until CTest kills it at 60 seconds.

This is not a full SuperCollider race report. It is a compact DBI diagnostic
showing that rocJITsu can identify synchronization instructions in final native
`gfx1201` code, compose that rewrite with the LDS duplicate/check patch, and
make an LDS-heavy IREE e2e kernel visibly fail without creating a special IREE
compiler variant.

## Patch Selection And Placement Guardrails

The native LDS check/trap path uses the following guardrails for compact IREE
kernels:

- automatic scratch VGPR selection is bounded by the AMDHSA kernel
  descriptor's allocated VGPR count,
- injected VCC-preserving compares prefer a free SGPR above the kernel's maximum
  referenced SGPR,
- local-cave selection is limited to one patch per kernel, avoiding overlapping
  patch/cave plans in compact code,
- if a code object has a single `.text` section and no inline or local cave is
  selected, rocJITsu can append a small trampoline cave to `.text` and branch to
  it,
- plain `ds_load_b64` and compact B64 two-address loads avoid using a duplicate
  scratch register run that ends exactly at the descriptor allocation edge.

The native LDS IREE matrix uses the same "reject if we should have patched but
could not" guard:

```sh
export RJ_DBI_SC_DELAY_MODE=sleep
export RJ_DBI_SC_DELAY=1
export RJ_DBI_SC_MAX_PATCHES=4
export RJ_DBI_SC_REQUIRE_PATCH=1
```

The covered test families are:

- `tests/e2e/linalg/check_rocm_hip_narrow_n_matmuls.mlir`
- `tests/e2e/matmul/*tileandfusewmma*rocm_hip`
- `tests/e2e/stablehlo_ops/check_rocm_hip_stream_dot.mlir`
- `tests/e2e/stablehlo_ops/check_rocm_hip_stream_dot_bf16.mlir`
- `tests/e2e/stablehlo_ops/check_rocm_hip_stream_dot_general.mlir`

The result:

```text
100% tests passed, 0 tests failed out of 10
```

rocJITsu patches selected compact IREE LDS kernels through inline padding, local
caves, or appended `.text` caves, and those tests still pass under the HSA tools
path on native `gfx1201`.

## Current IREE Native-DS Demo Scope

The IREE e2e demo exercises the native LDS / DS path. In that path, rocJITsu
covers:

- `ds_load_b{32,64,128}`,
- `ds_load_u16_d16`,
- `ds_load_u16_d16_hi`,
- `ds_load_2addr_b{32,64}`,
- `ds_load_2addr_stride64_b{32,64}`,
- `ds_store_b{32,64,128}`.

The patch can be placed three ways:

- inline, when the original LDS site has enough trailing `s_nop 0` padding,
- through a reachable uncovered local NOP cave,
- through an appended `.text` cave when the code object has a single `.text`
  section and the branch range works.

Native-DS d16 load support covers padded `ds_load_u16_d16` and
`ds_load_u16_d16_hi`. The duplicate path seeds scratch with the original full
destination dword before repeating the halfword load, so the full-dword compare
remains meaningful.

## Flat/VFLAT Support Exists Separately

The native-DS list is not the whole DBI SuperCollider scope. rocJITsu also has a
conservative flat/VFLAT check/trap path for likely LDS/group-memory accesses:

- `flat_load_b{32,64,128}`,
- `flat_store_b{32,64,128}`.

That path is separate from the IREE native-DS demo. It matters because some
HIP-generated kernels, especially the hip-moi helper-function code we inspected,
use final `flat_*` instructions even when the source-level intent is shared/LDS
memory. Since final machine code no longer carries a clean source-language
address-space label, rocJITsu only treats flat accesses as LDS candidates when
its provenance heuristic classifies them as likely group memory.

The flat/VFLAT path can patch padded sites, and can patch one selected unpadded
likely-group site through a reachable local NOP cave. Broader multi-site flat
instrumentation is still feature work.

Not in scope for this demo:

- other 8/16-bit native LDS forms beyond `ds_load_u16_d16(_hi)`,
- flat check/trap multi-site instrumentation,
- non-trapping report buffers,
- randomized sleep sampling policy beyond the scalar-source
  `s_sleep_var` mechanism,
- synchronization fault injection beyond the single selected decoded
  `s_barrier*` to `s_nop 0` rewrite.

## Remaining Feature Work

The next work is feature depth:

1. Strengthen `sleep_var` from "variable scalar source" into a real randomized
   or sampled delay policy.
2. Replace trap-only reporting with a device-visible report buffer once we want
   non-fatal diagnostics.
3. Expand flat/generic multi-site instrumentation beyond the conservative
   one-selected-site shape.
4. Add more native LDS widths and atomics only when the MVP needs them.
