# CDNA4 vs CDNA3 ISA diff notes for rocjitsu DBT

This document summarizes the CDNA4 to CDNA3 instruction differences that matter
for translating gfx950 code to gfx942. It combines generated rocjitsu ISA table
diffs with AMD markdown ISA notes.

The key mechanical result is reassuring: for instructions present on both
targets, generated rocjitsu tables did not show opcode moves or encoding-family
changes. Most risk comes from CDNA4-only instructions, changed documented side
semantics, and code-object/resource metadata.

## Sources

- `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3`
- `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4`
- `/home/kunwar/Work/amdgpu-isa-manuals/cdna3/README.md`
- `/home/kunwar/Work/amdgpu-isa-manuals/cdna4/README.md`
- `.codex/AMDGPUUsage.rst.txt`

## Mechanical generated-table summary

Generated source diff summary:

| Item | Result |
|---|---:|
| CDNA3 generated instruction records | 1354 |
| CDNA4 generated instruction records | 1423 |
| Added in CDNA4 | 75 |
| Removed from CDNA4 | 6 |
| Same mnemonic opcode changes | 0 |
| Same mnemonic encoding-family changes | 0 |
| Visible operand signature changes | 63 |

Important structural result:

| Encoding structure | CDNA3 vs CDNA4 |
|---|---|
| `MubufMachineInst` | Same bitfield layout |
| `MtbufMachineInst` | Same bitfield layout |
| `DsMachineInst` | Same bitfield layout |
| `FlatMachineInst` | Same bitfield layout |
| `Vop3pMachineInst` | Same bitfield layout |
| Scalar encodings | Same bitfield layouts for SOP1/SOP2/SOPC/SOPK/SOPP/SMEM |

Implication: the DBT generally does not need per-field bit reshuffling for these
formats. If translation fails, suspect legality/semantics/resource metadata
rather than silent bitfield relocation.

## Added instructions in CDNA4

### DS / LDS

| Opcode | CDNA4 mnemonic | CDNA3 status |
|---:|---|---|
| 224 | `ds_read_b64_tr_b4` | invalid/reserved |
| 225 | `ds_read_b96_tr_b6` | invalid/reserved |
| 226 | `ds_read_b64_tr_b8` | invalid/reserved |
| 227 | `ds_read_b64_tr_b16` | invalid/reserved |

These are transpose LDS read instructions. They are not aliases for normal DS
reads.

### MUBUF

| Opcode | CDNA4 mnemonic | CDNA3 status |
|---:|---|---|
| 82 | `buffer_atomic_pk_add_bf16` | absent |

`buffer_load_dwordx3` and `buffer_load_dwordx4` exist as mnemonics on both
targets, but CDNA4 newly documents using them with `lds=1`.

### VOP1

CDNA4 adds:

| Opcode | Mnemonic |
|---:|---|
| 75 | `v_exp_legacy_f32_e32` |
| 76 | `v_log_legacy_f32_e32` |
| 88 | `v_prng_b32_e32` |
| 89 | `v_permlane16_swap_b32_e32` |
| 90 | `v_permlane32_swap_b32_e32` |
| 91 | `v_cvt_f32_bf16_e32` |

### VOP2

| Opcode | Mnemonic |
|---:|---|
| 22 | `v_dot2c_f32_bf16_e32` |

### VOP3

CDNA4 adds many BF16/FP8/FP6/BF6/FP4 conversion and scaled-conversion
instructions. Key additions for current DBT work:

| Opcode | Mnemonic |
|---:|---|
| 278 | `v_dot2c_f32_bf16` |
| 395 | `v_exp_legacy_f32` |
| 396 | `v_log_legacy_f32` |
| 408 | `v_prng_b32` |
| 409 | `v_permlane16_swap_b32` |
| 410 | `v_permlane32_swap_b32` |
| 411 | `v_cvt_f32_bf16` |
| 563 | `v_bitop3_b16` |
| 564 | `v_bitop3_b32` |
| 613 | `v_ashr_pk_i8_i32` |
| 614 | `v_ashr_pk_u8_i32` |
| 615 | `v_cvt_pk_f16_f32` |
| 616 | `v_cvt_pk_bf16_f32` |
| 678 | `v_cvt_sr_f16_f32` |
| 679 | `v_cvt_sr_bf16_f32` |
| 680 | `v_minimum3_f32` |
| 681 | `v_maximum3_f32` |

The full CDNA4-only conversion family includes `v_cvt_scalef32_*` forms for
BF16, F16, F32, BF8, FP8, BF6, FP6, and FP4. These have no simple CDNA3
single-instruction equivalents.

### VOP3P and MFMA/SMFMAC

CDNA4 adds non-MFMA VOP3P instructions:

| Opcode | Mnemonic |
|---:|---|
| 26 | `v_dot2_f32_bf16` |
| 27 | `v_pk_minimum3_f16` |
| 28 | `v_pk_maximum3_f16` |

CDNA4 adds wide-K dense MFMA forms and sparse MFMA forms including:

| Mnemonic |
|---|
| `v_mfma_f32_16x16x32_f16` |
| `v_mfma_f32_32x32x16_f16` |
| `v_mfma_f32_16x16x32_bf16` |
| `v_mfma_f32_32x32x16_bf16` |
| `v_mfma_i32_16x16x64_i8` |
| `v_mfma_i32_32x32x32_i8` |
| `v_mfma_f32_16x16x128_f8f6f4` |
| `v_mfma_f32_32x32x64_f8f6f4` |
| `v_smfmac_f32_16x16x64_f16` |
| `v_smfmac_f32_32x32x32_f16` |
| `v_smfmac_f32_16x16x64_bf16` |
| `v_smfmac_f32_32x32x32_bf16` |
| `v_smfmac_i32_16x16x128_i8` |
| `v_smfmac_i32_32x32x64_i8` |

These are semantic supersets, not aliases. Lowering them to CDNA3 needs
multi-instruction expansion.

## Removed or reserved in CDNA4

### DS GWS operations

CDNA4 reserves/removes these CDNA3 DS GWS operations:

| Opcode | CDNA3 mnemonic | CDNA4 status |
|---:|---|---|
| 152 | `ds_gws_sema_release_all` | invalid/reserved |
| 153 | `ds_gws_init` | invalid/reserved |
| 154 | `ds_gws_sema_v` | invalid/reserved |
| 155 | `ds_gws_sema_br` | invalid/reserved |
| 156 | `ds_gws_sema_p` | invalid/reserved |
| 157 | `ds_gws_barrier` | invalid/reserved |

### MFMA XF32 forms

Generated CDNA4 sources remove these generated CDNA3 MFMA forms:

| CDNA3 mnemonic |
|---|
| `v_mfma_f32_16x16x8_xf32` |
| `v_mfma_f32_32x32x4_xf32` |

## Operand metadata differences for common mnemonics

No common mnemonic changed opcode or encoding family. Some generated operand
metadata differs:

| Family | Difference |
|---|---|
| DS | `ds_write_b8` data width modeled as 32 on CDNA3 and 8 on CDNA4 |
| DS | `ds_write_b16` data width modeled as 32 on CDNA3 and 16 on CDNA4 |
| SMEM | `s_atc_probe` / `s_atc_probe_buffer` data width modeled as 32 on CDNA3 and 8 on CDNA4 |
| SOP1 | `s_sext_i32_i8` source width modeled as 16 on CDNA3 and 8 on CDNA4 |
| SOPK/SOPP | `simm16` operands visible as 32 on CDNA3 and 16 on CDNA4 |
| VOP3 | `v_alignbit_b32` / `v_alignbyte_b32` `src2` modeled as 32 on CDNA3 and 8 on CDNA4 |
| VOP3 | `v_mad_i32_i24` / `v_mad_u32_u24` source inputs modeled as 32 on CDNA3 and 24 on CDNA4 |
| VOP3P | Packed 16-bit ops use 16-bit operand metadata on CDNA3 and 32-bit container metadata on CDNA4 |

These are mostly tooling/modeling risks. They can affect liveness and def-use
if operand widths are used mechanically.

## MUBUF and direct-to-LDS differences

MUBUF encoding is structurally identical:

| Field | Bits |
|---|---|
| `offset` | `[11:0]` |
| `offen` | `[12]` |
| `idxen` | `[13]` |
| `sc0` | `[14]` |
| `sc1` | `[15]` |
| `lds` | `[16]` |
| `nt` | `[17]` |
| `op` | `[24:18]` |
| `encoding` | `[31:26]` |
| `vaddr` | `[39:32]` |
| `vdata` | `[47:40]` |
| `srsrc` | `[52:48]` |
| `acc` | `[55]` |
| `soffset` | `[63:56]` |

Normal buffer memory address:

```text
MEM_ADDR =
  Base + soffset + inst_offset + voffset
  + stride * (vindex + optional TIDinWave)
```

`voffset` is ignored if `OFFEN=0`.

`vindex` is ignored if `IDXEN=0`.

`TIDinWave` in the memory address is controlled by the resource descriptor
`ADD_TID_ENABLE`.

CDNA4 direct MUBUF-to-LDS address:

```text
LDS_ADDR =
  LDSbase + M0_offset + inst_offset + TIDinWave * element_bytes
```

For CDNA4 `buffer_load_dwordx4 ... lds`:

```text
LDS_ADDR = LDSbase + M0_offset + inst_offset + TIDinWave * 16
```

For CDNA4 `buffer_load_dwordx3 ... lds`, the lane stride is still 16 bytes and
the fourth dword slot is skipped.

Direct-to-LDS support differs:

| Target | MUBUF direct-to-LDS documented subset |
|---|---|
| CDNA3 | `buffer_load_{ubyte,sbyte,ushort,sshort,dword,format_x}` |
| CDNA4 | CDNA3 subset plus `buffer_load_dwordx3` and `buffer_load_dwordx4` |

M0 width differs:

| Target | Direct MUBUF-to-LDS `M0` offset |
|---|---|
| CDNA3 | `M0[15:0]`, 16-bit byte offset |
| CDNA4 | `M0[17:0]`, 18-bit byte offset |

DBT risks:

| Risk | Impact |
|---|---|
| Missing LDS `inst_offset` in fallback | `buffer_load ... offset:N lds` must add `N` to the explicit `ds_write` address as well as to the memory load |
| `dwordx3` hole | Lowering must preserve 16-byte lane stride and skip the fourth dword |
| `M0` width | CDNA4 can name LDS offsets CDNA3 cannot represent if the kernel relies on more than gfx942 LDS capacity |
| LDS clamping/masking | Native direct-to-LDS masks lanes whose LDS destination is illegal; `buffer_load + ds_write` does not automatically reproduce this |
| EXEC semantics | Compute physical lane id under full wave if needed, but perform actual memory load and LDS write under guest EXEC |
| Waitcnt semantics | Expanded fallback must drain VMEM before DS write and drain LGKM before dependent LDS reads |

For the current ResNet dispatch 11, the visible CDNA4 instructions are
`buffer_load_dwordx4 ..., 0 offen lds`, so the missing LDS `inst_offset` bug is
real but probably not the cause of that dispatch.

## DS / LDS differences

DS encoding layout is structurally identical:

```text
offset0[7:0], offset1[15:8], gds/reserved[16], op[24:17],
acc[25], encoding[31:26], addr[39:32], data0[47:40],
data1[55:48], vdst[63:56]
```

Important differences:

| Area | CDNA3 | CDNA4 |
|---|---|---|
| DS opcodes 224-227 | invalid/reserved | transpose LDS reads |
| DS opcodes 152-157 | GWS semaphore/barrier ops | invalid/reserved |
| DS bit 16 | `GDS` for GWS ops | generated field still exists, docs describe GDS as reserved for these cases |
| LDS capacity per CU | 64 KiB | 160 KiB |
| LDS banks | 32 | 64 |
| LDS bank entries | 512 dwords/bank | 640 dwords/bank |

Transpose LDS reads:

| Instruction | Notes |
|---|---|
| `ds_read_b64_tr_b4` | CDNA4-only transpose load |
| `ds_read_b96_tr_b6` | CDNA4-only transpose load |
| `ds_read_b64_tr_b8` | CDNA4-only transpose load |
| `ds_read_b64_tr_b16` | CDNA4-only transpose load used by MFMA input paths |

`ds_read_b64_tr_b16` notes:

| Requirement | Detail |
|---|---|
| EXEC | Must be all ones for native CDNA4 transpose reads |
| LDS address | Must satisfy data-size alignment |
| VGPR alignment | DS ops reading/writing 64-bit or larger data require even VGPR alignment, except `ds_read_b96_tr_b6` |
| Layout | Two transpose-read instructions load the full matrix slice: one covers K groups `0..3` and `8..11`, the other covers `4..7` and `12..15` |

DBT risks:

| Risk | Impact |
|---|---|
| Exact MFMA input layout | The result is a register/lane layout contract, not just a memory load |
| Full EXEC requirement | Emulation under partial EXEC must either reject, normalize, or carefully preserve inactive architectural state |
| Source-lane disabled behavior | `ds_bpermute` applies EXEC to source reads; emulation must not accidentally zero values from lanes that native transpose would require to be active |
| Waitcnt and scratch pressure | Native transpose is one DS op; emulation introduces multiple DS/VALU/permutation operations |

## VALU, VOP, MFMA differences

Common VOP mnemonics keep generated opcode numbers across CDNA3 and CDNA4.

High-risk CDNA4-only VOP families:

| Family | Risk |
|---|---|
| `v_bitop3_b16/b32` | Truth table is encoded in modifier fields: `truth_table = (omod << 6) | (abs << 3) | neg`; modifiers are data bits, not normal arithmetic modifiers |
| `v_cvt_pk_f16_f32` | CDNA3 has `v_cvt_pkrtz_f16_f32`, but CDNA4 instruction follows normal F32-to-F16 rounding path; lowering needs two scalar conversions plus bitwise pack |
| `v_cvt_pk_bf16_f32` and scaled conversions | Need explicit conversion, scaling, packing, and rounding; FP4/FP6/BF6 have no simple CDNA3 equivalent |
| `v_permlane16_swap_b32` / `v_permlane32_swap_b32` | Cross-lane behavior plus CDNA4-specific hazards; generated model treats operands as destination-like |
| Wide-K MFMA | Must split into multiple CDNA3 narrower-K MFMAs while preserving accumulator file, temp destination, and source layout |
| SMFMAC | `ACC_CD` controls C/D file, but sparse index `SRC2` is architectural VGPR data; do not handle like dense MFMA source C |

ACC/VGPR encoding is structurally the same:

| Register file | Encoded range |
|---|---|
| destination VGPR | `0..255` |
| destination AccVGPR | `512..767` |
| source VGPR | `256..511` |
| source AccVGPR | `768..1023` |

MFMA file controls:

| Field | Meaning |
|---|---|
| `ACC[0]` | Selects source A file |
| `ACC[1]` | Selects source B file |
| `ACC_CD` | Selects C/D accumulator file |

GPR indexing risk:

| Rule | DBT impact |
|---|---|
| `M0[7:0]` is the unsigned index | Preserve M0 through expansions |
| `M0[15:12]` selects indexed dest/src operands | Liveness must treat potentially indexed VGPRs as live |
| Indexing applies only to VGPR operands | Do not apply to SGPRs or inline constants |
| Some VALU forms implicitly read destination | Expansions must preserve implicit destination reads under indexing |

## Scalar, waitcnt, and metadata differences

No scalar/control/system mnemonics were added or removed in the focused
generated tables:

| Class | CDNA3 count | CDNA4 count |
|---|---:|---:|
| SOP1 | 54 | 54 |
| SOP2 | 53 | 53 |
| SOPC | 20 | 20 |
| SOPK | 21 | 21 |
| SOPP | 32 | 32 |
| SMEM | 84 | 84 |

Important semantic/resource differences:

| Area | CDNA3 | CDNA4 | DBT impact |
|---|---|---|---|
| `EXPCNT` | Present and meaningful | Documented as unused in state overview | Do not blindly reinterpret wait counters |
| `LGKMCNT` | Counts LDS, GDS, scalar memory, messages | Counts LDS, scalar memory, messages | GDS waits are target-sensitive |
| `IB_STS` | Has `VM_CNT`, `EXP_CNT`, `LGKM_CNT` | Has `VM_CNT`, `LGKM_CNT` | `s_getreg_b32 hwreg(IB_STS)` needs semantic care |
| `M0` descriptor | LDS, GWS, GPR indexing | LDS, GPR indexing | GWS use is not portable |
| `LDS_ALLOC` | `LDS_BASE[7:0]`, `LDS_SIZE[20:12]` | `LDS_BASE[8:0]`, `LDS_SIZE[21:12]` | `s_getreg_b32 hwreg(LDS_ALLOC)` needs remapping if used |
| LDS allocation granularity | 512-byte blocks/alignment | 1280-byte blocks/alignment | Code-object metadata must be recomputed for gfx942 |
| SGPR limits | 16 to 102 logical SGPRs | same | no scalar-register-count delta |
| VGPR limits | up to 512 total, 256 regular plus 256 accumulation | same | no VGPR-count delta |

Wait-state notes relevant to generated expansions:

| Hazard | Required care |
|---|---|
| `S_SETREG` followed by `S_GETREG` or `S_SETREG` to same register | needs inserted wait states |
| `S_SETREG MODE.vskip` before vector op | needs wait states |
| `SALU writes M0` before LDS add-TID instruction, `buffer_store_LDS_dword`, scratch/global with LDS=1 | needs wait state |
| `VALU writes SGPR/VCC` before VMEM using that SGPR | needs wait states |
| `VALU writes SGPR/VCC` before VALU reads SGPR as constant | needs wait states |
| OPSEL/SDWA result consumed by next VALU | needs wait state |
| Transcendental VALU result consumed by non-trans VALU | needs wait state |

The broad diagnostic test that inserted `s_nop 2` after every instruction in the
current direct-to-LDS fallback did not change the ResNet dispatch 11 mismatch.
That makes a simple missing wait state inside that fallback less likely.

## Current ResNet direct-load dispatch risk assessment

The current first bad dispatch with direct loads enabled is:

```text
main$async_dispatch_11::main$async_dispatch_11_matmul_like_128x28x28x64_f16xf16xf32
```

Observed trace behavior:

| Property | Result |
|---|---|
| Inputs | match within tolerance |
| Output | large mismatch |
| Representative max abs diff | about 2.8 |
| Mismatch count | about 84657 / 100352 |

Visible offending instruction family:

```asm
buffer_load_dwordx4 v1, s[0:3], 0 offen lds
buffer_load_dwordx4 v2, s[0:3], 0 offen lds
buffer_load_dwordx4 v6, s[16:19], 0 offen lds
```

For these instructions:

| Question | Current assessment |
|---|---|
| Is VOFFSET dropped? | No. The visible `v1`/`v2`/`v6` is the `VADDR`/VOFFSET for `offen`, and the fallback preserves it as the ordinary buffer-load address operand |
| Is immediate offset missing from memory load? | No. The fallback preserves `src.offset` on the ordinary buffer load |
| Is immediate offset missing from LDS write? | Yes, in general. Dispatch 11 uses offset 0, so this likely does not explain this case |
| Is lane stride wrong for dwordx4? | The fallback uses `lane_id * 16`, matching CDNA4 docs |
| Did conservative NOP insertion help? | No. The mismatch was unchanged |
| Could LDS clamping/masking differ? | Yes. Native direct-to-LDS masks illegal LDS writes; fallback `ds_write_b128` does not currently synthesize that mask |
| Could code-object LDS metadata differ? | Yes. gfx950 LDS allocation granularity and size differ from gfx942; raw metadata copying can produce wrong host allocation/clamping assumptions |
| Could scratch liveness still be wrong? | Yes. No shared-opcode diff explains this away |
| Could transpose-read emulation still be wrong? | Less likely if isolated transpose tests pass, but still a high-risk instruction family |

Most plausible next checks for dispatch 11:

| Priority | Check |
|---:|---|
| 1 | Decode runtime kernel descriptor/resource metadata after translation and verify gfx942 LDS size/granularity are recomputed, not copied from gfx950 |
| 2 | Check whether any active lanes in dispatch 11 would write outside allocated LDS for the direct-to-LDS replacement; if yes, synthesize the CDNA4 clamping mask before `ds_write_b128` |
| 3 | Decode original MUBUF raw fields for each dispatch 11 direct-to-LDS instruction and compare field-by-field with emitted ordinary MUBUF |
| 4 | Verify branch-island return targets for all dispatch 11 expansions |
| 5 | Re-check scratch SGPR/VGPR liveness around the direct-to-LDS expansions, especially `v[80:84]` and `s[0:3]`, `s[10:11]` |
| 6 | Fix the known LDS `inst_offset` bug even though it probably does not explain this dispatch |

## Concrete known bugs or TODOs

| Item | Status |
|---|---|
| `buffer_load_dwordx{3,4} ... lds` fallback must add `inst_offset` to explicit LDS write address | Known bug |
| `buffer_load_dwordx3 ... lds` fallback must preserve 16-byte lane stride and skip fourth dword | Required if not already implemented |
| Direct-to-LDS fallback should consider native LDS clamping/masking behavior | Open risk |
| gfx950 to gfx942 code-object metadata must recompute LDS allocation/resource fields | Open risk |
| CDNA4 `ds_read_*_tr_*` emulation must enforce or safely handle full EXEC requirement | Open risk |
| CDNA4 wide-K MFMA expansions must preserve temp destination file and accumulator file | Previously fixed once, still high-risk class |
| CDNA4 `v_cvt_pk_f16_f32` lowering should use scalar conversions plus bitwise pack, not `v_pack` if exact bit behavior matters | Previously fixed once, still worth keeping in tests |
| GPR indexing regions should inform liveness; all potentially indexed VGPRs are live while indexing is enabled | Implemented directionally; keep testing |

