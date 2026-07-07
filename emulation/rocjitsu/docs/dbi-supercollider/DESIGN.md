# DBI SuperCollider Design Notes

## Audience

This note is for someone who has read the SuperCollider paper and wants to know
how this rocJITsu DBI implementation compares, especially around AMD LDS and
flat/generic memory.

The short version: the implementation follows the paper's redundant-access
shape, but currently as a post-register-allocation binary patcher. The paper's
DataCollider motivation describes the key redundant-read idea this way:

> "the algorithm issues a redundant read to the same address, separated by a random delay"

Our MVP preserves that value-checking idea, with deterministic NOP delay,
deterministic `s_sleep`, or scalar-source `s_sleep_var`, and the report path is
currently `s_trap`.

## Where The Instrumentation Runs

The MVP uses the HSA tools path:

```sh
HSA_TOOLS_LIB=.../librocjitsu_dbi_hooks.so
RJ_DBI_SUPERCOLLIDER=1
```

The hook wraps the HSA code-object load path, obtains the memory-backed code
object bytes, asks `try_patch_supercollider_dbi(...)` to inspect and possibly
rewrite them, then loads a replacement memory-backed reader if the patcher
returns modified bytes.

On the local machine this is native RDNA4 / `gfx1201` instrumentation. rocJITsu
is not translating the code object to another architecture.

Relevant implementation files:

- `lib/rocjitsu/src/rocjitsu/hooks/rj_hsa_dbi_hooks.cpp`
- `lib/rocjitsu/src/rocjitsu/code/patch/supercollider_dbi.h`
- `lib/rocjitsu/src/rocjitsu/code/patch/supercollider_dbi.cpp`

## SuperCollider Mechanism

The intended LDS/group-memory shape is:

1. Preserve the original weak memory access.
2. Delay without adding synchronization.
3. Repeat or read back the same address.
4. Wait only as needed to consume the injected read result.
5. Compare the original value or stored value with the duplicate/readback value.
6. Report by trapping on mismatch.

For loads, the duplicate access is another load from the same address into
scratch VGPRs. For stores, the duplicate access is a synthesized readback from
the stored address into scratch VGPRs.

This is intentionally not happens-before instrumentation. It accepts false
negatives. The paper-level no-false-positive argument only applies when the
instrumented access is actually shared/non-private and the injected check is
itself correct.

## Demo-Only Barrier Fault Injection

The Jakub demo also has an opt-in sync-fault mode:

```sh
RJ_DBI_SC_FAULT_DROP_BARRIER=1
RJ_DBI_SC_FAULT_BARRIER_INDEX=0
```

This is not part of the SuperCollider redundant-access algorithm. It is a
pragmatic diagnostic knob for the question, "can rocJITsu modify synchronization
in a real IREE code object and make the result observably wrong?"

Implementation shape:

1. Run normal inventory/preflight.
2. Run the primary proof/instrumentation mode. By default this is native
   LDS check/trap; use `RJ_DBI_SC_CHECK_TRAP_MODE=flat` to select the
   flat/VFLAT path instead.
3. Decode kernel and local-function ranges again.
4. Select the Nth decoded 4-byte `s_barrier*` instruction.
5. Rewrite that single instruction to `s_nop 0` in the current output ELF
   bytes, preserving any LDS check/trap patch already applied.

The mode intentionally skips ROCclr runtime-helper-only code objects. On the
focused IREE WMMA e2e test, the control run passes under native LDS check/trap
mode, while `RJ_DBI_SC_FAULT_DROP_BARRIER=1` with barrier index 0 also emits an
`inline-barrier-nop-rewrite` patch and makes the BF16 test hang until CTest's
60-second timeout.

That timeout is not a polished race report. It is a useful first diagnostic:
the DBI path can identify synchronization instructions in final native
`gfx1201` code, combine that rewrite with the LDS duplicate/check patch, and
make an LDS-heavy IREE e2e kernel visibly fail without creating a special IREE
compiler variant.

## Currently Instrumented Instructions

The current non-atomic native LDS check/trap path supports:

- `ds_load_b32`
- `ds_load_b64`
- `ds_load_b128`
- `ds_load_2addr_b32`
- `ds_load_2addr_b64`
- `ds_load_2addr_stride64_b32`
- `ds_load_2addr_stride64_b64`
- `ds_load_u16_d16`
- `ds_load_u16_d16_hi`
- `ds_store_b32`
- `ds_store_b64`
- `ds_store_b128`

The current likely group/LDS flat check/trap path supports RDNA4 12-byte VFLAT:

- `flat_load_b32`
- `flat_load_b64`
- `flat_load_b128`
- `flat_store_b32`
- `flat_store_b64`
- `flat_store_b128`

The implementation deliberately does not instrument:

- atomics,
- fence-heavy cases,
- async copies,
- global memory,
- generic/private accesses classified as private or maybe-private,
- unsupported flat widths such as b8/b16/b96,
- arbitrary `flat_*` accesses with unknown address-space provenance.

## Native DS Patch Shape

For padded native LDS sites, the patch is in-place. Each selected instruction
must have enough trailing `s_nop 0` padding to hold the check sequence. For
compact native LDS sites without padding, the patcher can instead redirect
selected 8-byte sites through reachable uncovered local NOP caves.

`RJ_DBI_SC_MAX_PATCHES=N` bounds how many native-DS check/trap sites can be
patched in one code object. Selection is greedy and conservative: candidates
are considered in file order, selected byte ranges may not overlap, and each
local cave is consumed at most once.

For a load:

```text
original ds_load_b*
delay: NOP sequence, one s_sleep, or one s_sleep_var, selected by RJ_DBI_SC_DELAY_MODE
duplicate ds_load_b* to scratch VGPRs
s_wait_dscnt 0
s_mov_b32 free_sgpr, vcc_lo
v_cmp_ne_u32 per dword
s_cbranch_vccz skip_trap
s_trap 0
s_mov_b32 vcc_lo, free_sgpr
```

For `ds_load_u16_d16` and `ds_load_u16_d16_hi`, the duplicate load writes only
one 16-bit half of its destination VGPR. The patcher therefore waits for the
original d16 load, copies the original full destination dword into the scratch
VGPR with `v_mov_b32_e32`, delays, then repeats the same d16 load into scratch.
The untouched half of scratch still matches the original destination, so a
full-dword `v_cmp_ne_u32` catches a changed low or high half without a separate
masking sequence.

For a store:

```text
original ds_store_b*
delay: NOP sequence, one s_sleep, or one s_sleep_var, selected by RJ_DBI_SC_DELAY_MODE
synthesized ds_load_b* readback to scratch VGPRs
s_wait_dscnt 0
s_mov_b32 free_sgpr, vcc_lo
v_cmp_ne_u32 per dword against original data VGPRs
s_cbranch_vccz skip_trap
s_trap 0
s_mov_b32 vcc_lo, free_sgpr
```

The `v_cmp_ne_u32` instructions write VCC, so the patcher saves and restores
`vcc_lo` in a liveness-selected SGPR around the injected compare/trap sequence.
This matters for compact IREE kernels: the WMMA kernel that first exposed this
path had live VCC state used by later scalar branches, and clobbering VCC caused
wrong output even when the duplicated LDS read matched.

Automatic scratch VGPR selection is deliberately conservative. The patcher
starts above the maximum VGPR referenced by the kernel, caps the chosen run by
the AMDHSA kernel descriptor's allocated VGPR count, and does not grow the
descriptor behind the loader's back. For injected compares, it similarly prefers
a free SGPR above the kernel's maximum referenced SGPR before falling back to
lower liveness-proven SGPRs.

One RDNA4 edge case is intentionally skipped: when a plain `ds_load_b64` or a
compact B64 two-address load would use a duplicate scratch run ending exactly at
the descriptor's allocated VGPR boundary, the patcher declines that site and
continues searching. That rule came from StableHLO stream-dot debugging, where a
descriptor-edge `v14:v15` duplicate B64 load was legal-looking but produced an
illegal-instruction failure on the real `gfx1201` path. A later supported LDS
site in the same code object can still be patched, so this is a candidate
selection guard rather than a whole-code-object skip.

For local-cave native DS patches, the original 8-byte DS instruction is replaced
with:

```text
s_branch cave
s_nop 0
```

The cave body contains the original DS access, the check sequence, and an
`s_branch` back to the original fallthrough. Both branches must fit the direct
`s_branch` immediate. Sites inside an active `s_clause` are skipped.

If no inline or local-cave placement is selected and the code object has a
single `.text` section, the patcher can append a cave to `.text`, branch to the
new code, then branch back to the original fallthrough. This is the placement
that unblocked compact IREE TileAndFuse kernels with many supported LDS sites
but no large enough reachable uncovered local NOP cave.

The two-address native DS loads need special width accounting. For example,
`ds_load_2addr_b64` performs two 64-bit LDS loads in one 8-byte instruction, so
the destination spans four VGPR dwords and the duplicate/check sequence needs
four scratch VGPRs and four comparisons. The patcher intentionally does not
treat that as a plain two-dword `b64` load.

This path is unit-proven for b32/b64/b128 padded sites, native-DS local caves,
appended `.text` caves, and `ds_load_2addr_b64` local caves. It is live-proven
on hand-shaped padded tests and on the IREE ROCm/HIP e2e demo set. The focused
WMMA example patches:

```text
kind=local-cave-lds-load-check-trap anchor=0x3cc trampoline=0x810 original_size=8 scratch_vgpr=104
```

Ordinary hip-moi matmul code did not provide the native `ds_*` sites we
initially expected, which is why the flat path below is still important.

## Why Flat/VFLAT Is In Scope

Jakub is right to be suspicious of flat instrumentation: hand-authored kernels
usually try to use explicit DS instructions for LDS. The reason flat matters
here is empirical. The RDNA4 hip-moi matmul objects we are using as the
inner-loop corpus decode with zero native non-atomic `ds_*` sites in the
ordinary benchmark kernels, while the helper functions contain many flat/VFLAT
memory operations. After function-range decoding and provenance tracking, the
focused `NoPipelineProd16x8` object reports:

```text
function_flat_sites=1508
function_flat_maybe_group_hints=31
patches=1
modified=true
```

The live demo patch is:

```text
kind=local-cave-flat-store-check-trap anchor=0xb760 trampoline=0x20 original_size=12 scratch_vgpr=6
```

In other words, flat support is not a theoretical broadening for hand-written
assembly. It is how the current compiled hip-moi corpus exposes the LDS-like
helper accesses we need for an MVP demonstration.

## Why `__shared__` Can Still Become `flat_*`

The hip-moi source really does use `__shared__`. In the representative
reference kernel
`~/workspace/hip-moi/tests/reference/rdna4_jakub_matmul_reference.hip`, the
staging arrays are declared as:

```c++
__shared__ f16x8 a_s[kStages][KGroup][MTiles][kWaveSize];
__shared__ f16x8 b_s[kStages][KGroup][NTiles][kWaveSize];
```

So the observation is not that the storage stopped being LDS. The observation
is that, after HIP/LLVM lowering and optimization, some accesses to that LDS
storage are performed through AMDGPU's flat/generic address path rather than
through explicit DS instructions.

The likely reason is address-space erosion at the optimized machine-code
boundary. The source wraps LDS staging in local lambdas such as
`regs_to_shared`, `shared_to_compute_regs`, and `compute_stage`. In the final
code object, rocJITsu sees many of the relevant memory operations in local
`.text` helper-function ranges, not only in the kernel descriptor entry ranges.
Those helper bodies construct and consume 64-bit flat pointers. Once the
instruction selector is looking at a generic/flat pointer value instead of an
obvious LDS-address-space pointer, it cannot simply choose a DS instruction:
`ds_*` operations consume an LDS offset-style address, while `flat_*` operations
consume a full generic pointer. Emitting `flat_load_*` / `flat_store_*`
preserves the semantics of a pointer that might, in general, name one of
several AMDGPU address spaces.

On AMDGPU, `flat_*` does not mean "global only." It is the generic memory path.
An address derived from the shared/LDS aperture can still reach LDS through a
flat instruction. The signal we used to recover this was the pointer
construction around `src_shared_base`: Session 23 found that the likely
group/LDS helper-function flat sites are reached by tracking pointer halves
from `src_shared_base` through vector add/carry address construction. The
disassembly/log evidence lines are therefore consistent with "LDS accessed
through a flat pointer," not with "the compiler put `__shared__` storage in
global memory."

A useful mental model:

- `ds_*`: compiler still has a direct LDS address-space access and can encode an
  LDS offset-style operation.
- `flat_*`: compiler has a generic/flat pointer; the pointer value decides at
  runtime which aperture is addressed.
- hip-moi case: source storage is `__shared__`, but parts of the optimized
  helper/function code manipulate the address as a flat pointer derived from
  `src_shared_base`, so final instructions are `flat_load_b128` and
  `flat_store_b128`.

This is also why DBI has to be more conservative than a compiler pass. A
compiler pass can ask the IR for the address space. Our post-RA patcher sees
only the final `flat_*` instruction and nearby dataflow, so it records
`MaybeGroup` unless the local machine-code provenance is strong enough to say
more.

## How Flat Provenance Works

An AMD flat instruction does not by itself say "this is LDS." The patcher
therefore attaches a conservative address-space hint to decoded flat sites.

Current public hints are:

- `Group`
- `Private`
- `MaybeGroup`
- `MaybePrivate`
- `Global`
- `Unknown`

The tracker follows simple pointer construction patterns through scalar and
vector registers:

- `src_shared_base` seeds a group/LDS pointer.
- `src_private_base` seeds a private/thread-local pointer.
- Moves and simple pair construction preserve known low/high pointer halves.
- Some vector add/carry patterns degrade a known group/private pointer into
  `MaybeGroup` or `MaybePrivate`.
- Unknown or unsupported transformations drop the hint back to `Unknown`.

Only `Group` and `MaybeGroup` are considered instrumentable by the current flat
proof paths. `Private`, `MaybePrivate`, `Global`, and `Unknown` are skipped.

That is a deliberate MVP tradeoff. `MaybeGroup` is not a formal proof from the
ISA that a given flat access targets LDS. It means the local dataflow seen by
the decoder is consistent with a pointer derived from `src_shared_base`, and not
with an observed private/global derivation. This is strong enough for targeted
hip-moi bring-up, but not strong enough for a paper-level no-false-positive
claim on arbitrary binaries.

## Flat Patch Shape

RDNA4 VFLAT `flat_load/store_b{32,64,128}` instructions are 12 bytes. The
in-place flat path is analogous to the native DS path, but the original and
duplicate/readback accesses are three-word VFLAT instructions.

For a flat load, the duplicate load is created by retargeting `vdst` to scratch
VGPRs.

For a flat store, the readback is synthesized by changing the VFLAT opcode to
the matching load width, setting `vdst` to scratch VGPRs, and clearing the store
source field.

The current local-cave path is used when the selected flat site lacks trailing
padding:

1. Replace the original 12-byte flat instruction with `s_branch` plus two
   `s_nop 0` fill words.
2. Find a conservative uncovered `s_nop 0` run outside known kernel/function
   symbol ranges.
3. Emit the original flat access, delay, duplicate/readback, wait, compare, and
   trap sequence into that cave.
4. End the cave sequence with `s_branch` back to the original fallthrough.

Both forward and return branches must fit the direct `s_branch` immediate. If
no padded site or reachable local cave exists, the patcher skips and logs why.

## Comparison To The Paper

Aligned:

- Preserves the original access.
- Performs a redundant access to the same address.
- Compares values and reports on mismatch.
- Avoids shadow memory and happens-before state.
- Treats partial instrumentation as a false-negative risk, not inherently a
  false-positive risk.

Divergent:

- The paper instruments before register allocation; this implementation patches
  final RDNA4 machine code.
- The paper uses randomized sleep to perturb scheduling; the current
  `RJ_DBI_SC_DELAY_MODE=sleep_var` mode emits one `s_sleep_var` instruction
  from a scalar source operand. Its default source is `vcc_lo`, so this is
  variable perturbation rather than a full randomized sampling policy.
- The paper reports through a runtime buffer; this MVP reports with `s_trap`.
- The paper can reason from compiler IR address spaces; this DBI path has to
  infer flat/generic address space from final machine-code dataflow.
- The paper has same-address intra-warp store checks for same-value lost
  updates; this MVP does not yet have that check.

The biggest remaining near-term correction is randomized delay policy, not the
basic binary patching substrate. The basic `s_sleep_var` mechanism exists; a
better sampled source or per-site seed is still open.

## Reporting And Diagnostics

With `RJ_DBI_LOG=1`, the hook reports:

```text
SuperCollider DBI summary reader=... kernels=... supported_lds_sites=... flat_sites=... function_flat_maybe_group_hints=... patches=... modified=...
```

Important fields:

- `modified=true`: a replacement code object was loaded.
- `patches=N`: number of emitted patches.
- `supported_lds_sites`: native DS sites that match current policy.
- `flat_sites`: decoded flat/VFLAT memory sites.
- `function_flat_maybe_group_hints`: helper-function flat sites likely derived
  from `src_shared_base`.

The current IREE demo-relevant native-DS proof patch kinds are
`local-cave-lds-load-check-trap`, `local-cave-lds-store-check-trap`, and the same
patch kinds backed by an appended `.text` cave. The current hip-moi flat proof
patch kind is `local-cave-flat-store-check-trap`.

## Open Design Questions

- Should `MaybeGroup` remain enabled by default for hip-moi, or become an
  explicit "heuristic flat LDS" mode?
- What is the right sampled seed source for a SuperCollider-like `s_sleep_var`
  delay window?
- How much scratch VGPR or descriptor growth is acceptable for a DBI MVP?
- Should the next reporting step be a fixed device-visible report buffer, a
  trap with richer metadata dumps, or both?
- How soon do we need same-address intra-wave store checks for same-value LDS
  races?
