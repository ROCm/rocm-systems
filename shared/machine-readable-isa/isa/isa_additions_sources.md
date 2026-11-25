# CDNA1 SRC2, RDNA2 ATC and RDNA4 barrier additions

The field templates for these three overlays use schema 1.1.1. In the paths
below, an instruction name selects the XML `Instruction` with that
`InstructionName`. The base XML files are pinned in
[ROCm/rocm-systems at b661f7d9](https://github.com/ROCm/rocm-systems/tree/b661f7d935fe37ca1b93af38b2a5c04ecc429e95/shared/machine-readable-isa/isa).

## CDNA1 `amdgpu_isa_cdna1_ds_src2.xml`

The 31 mnemonic/opcode pairs are the MI100 SRC2 instructions. Each definition
in LLVM maps its one-address pseudo to `DS_Real_m0_vi`. The gaps in the opcode
sequence are intentional. The MC file has a block for each mnemonic with
literal encoding bytes and variants for the offset and GDS fields.
[MI100 instruction table](https://github.com/kuhar/amdgpu-isa-manuals/blob/4ca6ba42c7d1e0ec0dcf2758399117be2b84dc04/cdna1/README.md#L3959)
[LLVM opcode definitions](https://github.com/llvm/llvm-project/blob/254f05a4807997ae410e157f1673c8c26020a88c/llvm/lib/Target/AMDGPU/DSInstructions.td#L2103-L2133)
[LLVM assembly vectors](https://github.com/llvm/llvm-project/blob/254f05a4807997ae410e157f1673c8c26020a88c/llvm/test/MC/AMDGPU/gfx9_asm_ds.s#L97-L147)

Use the `amdgpu_isa_cdna1.xml` instruction obtained by removing `_SRC2` from
the name as the template for the flags, functional group, `ENC_DS`, the
`default` condition and the `ADDR` VGPR operand. Use its implicit
`OPR_DSMEM` input/output format and operand size for the data-share entries.
In operand order, the added instructions read ADDR, read the second memory
location and write the first. All except WRITE also read the first location.
The memory inputs and outputs have no explicit field and are marked implicit.
The description names the operation between `DS_` and `_SRC2` and records
the destination. Write the identifier as 64 binary digits for
`0xd8000000 | (opcode << 17)`, with the operand and offset fields zero.

## RDNA2 `amdgpu_isa_rdna2_atc_probe.xml`

The RDNA2 SMEM table gives `S_ATC_PROBE` opcode 38 and `S_ATC_PROBE_BUFFER`
opcode 39. LLVM’s GFX10 definitions and byte vectors cover the immediate and
SGPR-offset forms of both instructions.
[RDNA2 instruction table](https://github.com/kuhar/amdgpu-isa-manuals/blob/4ca6ba42c7d1e0ec0dcf2758399117be2b84dc04/rdna2/README.md#L3255-L3256)
[LLVM opcode definitions](https://github.com/llvm/llvm-project/blob/254f05a4807997ae410e157f1673c8c26020a88c/llvm/lib/Target/AMDGPU/SMInstructions.td#L1311-L1312)
[LLVM assembly vectors](https://github.com/llvm/llvm-project/blob/254f05a4807997ae410e157f1673c8c26020a88c/llvm/test/MC/AMDGPU/gfx10_asm_smem.s#L9-L25)

Use the same-named `amdgpu_isa_rdna1.xml` instruction for all instruction
fields, including the description, flags and functional group. RDNA2's
operand-type table supplies `OPR_SIMM16` for the SDATA immediate in place of
RDNA1's `OPR_SIMM8` (see RDNA2 `S_MOVK_I32`). Preserve the 8-bit operand
size and `FMT_NUM_B8`. The SBASE and SOFFSET fields also match RDNA2
`S_LOAD_DWORD` and `S_BUFFER_LOAD_DWORD`, respectively for the plain and
buffer forms. Write the identifier as 64 binary digits for
`0xf4000000 | (opcode << 18)`, with all operand fields zero.

## RDNA4 `amdgpu_isa_rdna4_barrier_state.xml`

RDNA4 SOP1 assigns `S_GET_BARRIER_STATE` opcode 80. LLVM encodes both M0 and
the inline barrier identifiers with that opcode.
[RDNA4 instruction table](https://github.com/kuhar/amdgpu-isa-manuals/blob/4ca6ba42c7d1e0ec0dcf2758399117be2b84dc04/rdna4/README.md#L5662)
[LLVM definitions](https://github.com/llvm/llvm-project/blob/254f05a4807997ae410e157f1673c8c26020a88c/llvm/lib/Target/AMDGPU/SOPInstructions.td#L2346-L2351)
[LLVM assembly vectors](https://github.com/llvm/llvm-project/blob/254f05a4807997ae410e157f1673c8c26020a88c/llvm/test/MC/AMDGPU/gfx12_asm_sop1.s#L747-L755)

Use `S_GET_BARRIER_STATE` in `amdgpu_isa_cdna5.xml` for the flags,
description, SDST and barrier ID operands, and MESSAGE/STATIC group.
Substitute the nonliteral `ENC_SOP1` encoding condition from
`S_BARRIER_SIGNAL` in `amdgpu_isa_rdna4.xml`. The 32-bit binary identifier
is `0xbe800000 | (80 << 8)`.
