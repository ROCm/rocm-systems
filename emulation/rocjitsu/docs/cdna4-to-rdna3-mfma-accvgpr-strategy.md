# CDNA4-to-RDNA3 MFMA and AccVGPR Strategy

This note records the current disposition for the CDNA4 matrix bucket in the
CDNA4-to-RDNA3 translator. It complements the broader taxonomy in
`cdna4-to-rdna3-taxonomy.md`.

## Current Decision

No MFMA, SMFMAC, or standalone AccVGPR read/write lowering is enabled for
CDNA4-to-RDNA3 yet. The defensible current implementation is a fail-closed
semantic guard for this bucket: CDNA4 `v_mfma_*`, `v_smfmac_*`, and
`v_accvgpr_*` instructions are treated as requiring semantic expansion even
when the generated legalization table contains duplicate `Lower` rows for the
same key. This prevents silent field-only encoding into an RDNA3 VOP3P opcode.

## Source and Target Facts

CDNA4 MFMA instructions can store A, B, C, and D matrix fragments in either
VGPRs or AccVGPRs. The CDNA4 ISA describes AccVGPRs as matrix-core-only
registers and states that the MFMA `ACC` and `ACC_CD` bits choose VGPR versus
AccVGPR operands. Data moves use `V_ACCVGPR_*` instructions.

RDNA3 supports WMMA for 16x16 tiles in wave32 and wave64 modes. AMD's RDNA3
WMMA guidance documents different fragment behavior by wave mode: in wave64,
A and B fragment contents must be replicated from lanes 0-15 into lanes 32-47
and 48-63, and C/D fragments use four VGPRs per lane.

Those facts rule out a single-instruction rewrite from CDNA4 MFMA to RDNA3
WMMA. CDNA4 MFMA input fragments are K-sliced across the wavefront, while RDNA3
WMMA expects replicated fragments. The translator must replace the surrounding
data-preparation idiom, not just the matrix instruction.

## Bucket Categorization

The complex bucket contains 66 CDNA4 VOP3P_MFMA rows:

| Category | Count | RDNA3 direction |
| --- | ---: | --- |
| Dense FP32 output MFMA, F32/F16/BF16/I8 inputs | 38 | Some 16x16 forms can become WMMA only after input-layout replacement. 32x32 and multi-block forms need tiling. |
| Sparse SMFMAC | 28 | Needs sparse metadata semantics and WMMA/SWMMAC availability analysis; no direct RDNA3 lowering. |

Related AccVGPR rows are:

| Instruction | Issue |
| --- | --- |
| `v_accvgpr_read` | Reads an AccVGPR value into a VGPR. RDNA3 has no AccVGPR operand class; mapping AccVGPRs to high unified indices is not enough because ordinary VOP sources cannot encode arbitrary `v[256+]` operands. |
| `v_accvgpr_write` | Writes a scalar/VGPR source into AccVGPR state. Correct lowering needs a chosen physical VGPR allocation and all later matrix consumers rewritten consistently. |
| `v_accvgpr_mov_b32_e32` | AccVGPR-to-AccVGPR copy. Safe only after the same AccVGPR virtualization or matrix-idiom replacement exists. |
| `v_accvgpr_mov_b32` | VOP3 AccVGPR move with source modifiers. Needs modifier-preserving lowering in the chosen virtual-register plan. |

## Required Lowering Design

The first real translation subset should be pattern-level, not instruction-level:

1. Match a full compiler idiom around a supported MFMA shape, starting with
   `v_mfma_f32_16x16x16_f16` and `v_mfma_f32_16x16x16_bf16`.
2. Include the source data-prep instructions that create the CDNA4 MFMA
   fragment layout, especially CDNA4 transpose LDS loads when present.
3. Rewrite data prep into RDNA3 WMMA fragment layout, including required A/B
   replication for wave64.
4. Emit RDNA3 `v_wmma_*` and any output lane permutation proven by a matrix
   layout table or the AMD Matrix Instruction Calculator.
5. Replace AccVGPR readback only as part of the matched idiom, so the chosen
   accumulator storage and final VGPR writes are consistent.
6. Validate numerically with simulator-level matrix tests before enabling the
   lowering for binary translation.

Standalone AccVGPR read/write support is a separate design. It needs either a
whole-kernel virtual AccVGPR-to-VGPR allocation pass or a narrower peephole that
can prove each AccVGPR value is produced and consumed within one matched matrix
idiom. The current intra-block liveness helper is not enough to prove this
across basic-block exits.

## Follow-Up Work

The current implementation leaves these paths explicitly unsupported and
recorded as follow-up implementation work:

- Pattern-level CDNA4 dense MFMA to RDNA3 WMMA lowering for 16x16 F16/BF16.
- AccVGPR virtualization or idiom-local elimination for CDNA4-to-RDNA3.
- Sparse SMFMAC disposition for RDNA3.
- Software-emulation fallback for MFMA shapes without RDNA3 WMMA coverage.

## References

- AMD CDNA4 Instruction Set Architecture, section 7.1 and VOP3P MFMA opcode
  descriptions.
- AMD GPUOpen, "How to accelerate AI applications on RDNA 3 using WMMA."
