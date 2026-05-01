# CDNA4-to-RDNA3 Translation Taxonomy

This report covers the current CDNA4-to-RDNA3 DBT state in:

- `lib/rocjitsu/src/rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna3.h`
- `lib/rocjitsu/src/rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna3.h`
- `lib/rocjitsu/src/rocjitsu/code/dbt/semantic_translator.cpp`
- `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4`
- `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3`

The generated legalization header is alias-expanded for raw
`Instruction::encoding_id()` values. It contains 31,594 physical rows:

| Action | Alias-expanded header rows |
| --- | ---: |
| Identity | 1,407 |
| Substitute | 13,666 |
| Lower | 9,146 |
| Expand | 7,375 |
| Illegal | 0 |

Those rows are not unique implementation work items. Alias expansion currently
also creates duplicate `(src_encoding_id, src_opcode)` keys: 30,310 unique keys,
1,223 duplicated keys, and 1,135 duplicated keys with conflicting action or
target-opcode payloads. The duplicate rows are a generated-table integrity issue
for later implementation planning, but they do not change the taxonomy unit used
below: one unexpanded CDNA4 source instruction row as emitted by
`amdisa.legalization.LegalizationGenerator`.

The unexpanded CDNA4-to-RDNA3 legalization view has 1,496 rows:

| Action | Unique source instruction rows |
| --- | ---: |
| Identity | 66 |
| Substitute | 361 |
| Lower | 638 |
| Expand | 431 |
| Illegal | 0 |

## Counts By Encoding Family

| Family | Total | Identity | Substitute | Lower | Expand |
| --- | ---: | ---: | ---: | ---: | ---: |
| DS | 126 | 0 | 0 | 106 | 20 |
| FLAT | 54 | 0 | 0 | 15 | 39 |
| MTBUF | 16 | 0 | 0 | 8 | 8 |
| MUBUF | 74 | 0 | 0 | 60 | 14 |
| SMEM | 84 | 0 | 0 | 11 | 73 |
| SOP1 | 54 | 2 | 33 | 7 | 12 |
| SOP2 | 53 | 9 | 42 | 0 | 2 |
| SOPC | 20 | 16 | 2 | 0 | 2 |
| SOPK | 21 | 3 | 14 | 3 | 1 |
| SOPP | 32 | 1 | 27 | 1 | 3 |
| VOP1 | 88 | 23 | 44 | 2 | 19 |
| VOP2 | 62 | 0 | 27 | 17 | 18 |
| VOP3 | 500 | 0 | 1 | 375 | 124 |
| VOP3P | 38 | 0 | 0 | 26 | 12 |
| VOP3P_MFMA | 66 | 0 | 0 | 0 | 66 |
| VOP3_SDST_ENC | 10 | 0 | 0 | 7 | 3 |
| VOPC | 198 | 12 | 171 | 0 | 15 |

## Difficulty Buckets

The buckets below are ordered from easiest to hardest. Counts are unique source
instruction rows and sum to 1,496.

| Bucket | Count | Definition |
| --- | ---: | --- |
| Identity copy | 66 | Same mnemonic, target-compatible opcode, and field layout. Copy source words or round-trip through the encoder without changing opcode fields. |
| Opcode substitute | 361 | Same field layout, but source and target opcodes differ. Rewrite only the opcode field through the generated encoding translator. |
| In-place field remap/lower | 629 | Same instruction footprint and no semantic expansion required, but fields, flags, names, or target opcodes differ. Use generated decode/encode field transfer plus targeted tests. Excludes `s_waitcnt` and MTBUF. |
| Same-size semantic rewrite | 1 | Semantic difference is not safely represented by field transfer alone, but the replacement fits in the source footprint. Today this is `s_waitcnt`. |
| Size-expanding rewrite | 353 | No single RDNA3 instruction preserves the CDNA4 behavior, and the row is not matrix/AccVGPR or MTBUF. Requires multi-instruction lowering and code-cave/stub handling. |
| Complex matrix/AccVGPR emulation | 70 | All 66 `VOP3P_MFMA` rows (`v_mfma_*` and `v_smfmac_*`) plus four AccVGPR rows. Requires lane-layout, accumulator-file, and VGPR allocation design. |
| Residual unsupported/unknown | 16 | MTBUF rows. The legalization view has 8 Lower and 8 Expand MTBUF rows, but the CDNA4-to-RDNA3 encoding translator has no MTBUF encoder case. |

Bucket notes:

- `s_waitcnt` is counted as the one same-size semantic rewrite. The generator
  marks it `Lower`, but the CDNA4-to-RDNA3 semantic table handles it before the
  encoding path by decoding the CDNA4 GFX9-layout counters and re-encoding a
  same-size RDNA3 GFX11-layout `s_waitcnt`.
- `v_lshl_add_u64` is counted in size-expanding rewrite. It is currently the
  only CDNA4-to-RDNA3 size-expanding arithmetic rule wired in the semantic
  translator.
- The complex matrix/AccVGPR bucket is all `Action::Expand`: 66
  `VOP3P_MFMA` rows, `v_accvgpr_mov_b32_e32`, `v_accvgpr_mov_b32`,
  `v_accvgpr_read`, and `v_accvgpr_write`.
- MTBUF is residual because the generated encoder switch handles VOP2, VOPC,
  VOP1, SOP2, SOPK, SOP1, SOPC, SOPP, SMEM, VOP3, VOP3_SDST_ENC, VOP3P, DS,
  FLAT, and MUBUF, but not MTBUF.

## Currently Wired Special Cases

The CDNA4-to-RDNA3 translator is selected in `BinaryTranslator` through
`translate_encoding_cdna4_to_rdna3()` and `kLegalization_cdna4_to_rdna3`.
The generated encoder handles VOP2, VOPC, VOP1, SOP2, SOPK, SOP1, SOPC, SOPP,
SMEM, VOP3, VOP3_SDST_ENC, VOP3P, DS, FLAT, and MUBUF. FLAT is split by `seg`:
flat, scratch, and global forms are re-encoded through the matching RDNA3
structure.

## Memory-family Memory-Family Audit Notes

The supported same-size CDNA4-to-RDNA3 memory path is the generated field
transfer for ordinary SMEM, MUBUF, FLAT, FLAT_GLBL-to-FLAT_GLOBAL,
FLAT_SCRATCH, and DS rows whose source-only domain bits are clear. Targeted
tests cover representative SMEM buffer loads, MUBUF loads and atomics, FLAT
flat/scratch/global segment routing, DS atomics, null scalar operand remapping,
and GFX940-to-GFX11 coherency remapping (`sc0 -> glc`, `sc1 -> slc`,
`nt -> dlc`).

CDNA4 source-only memory-domain fields that RDNA3 cannot represent in the same
instruction are now explicit residuals rather than silent drops:

| Family | Source-only field | Residual reason |
| --- | --- | --- |
| SMEM | `nv` | No RDNA3 same-size SMEM bit preserves the CDNA4 non-volatile/cache behavior. |
| MUBUF | `lds`, `acc` | `lds` changes the destination memory domain; `acc` changes the register file. |
| FLAT | `lds`, `acc` | `lds` changes the destination memory domain; `acc` changes the register file. |
| FLAT_GLBL, FLAT_SCRATCH | `acc` | Requires AccVGPR/register-file strategy outside this same-size memory audit. |
| DS | `acc` | Requires AccVGPR/register-file strategy outside this same-size memory audit. |

Non-zero values for those fields make the generated CDNA4-to-RDNA3 encoding
translator return no translation. Expansion or explicit diagnostic policy for
those residuals belongs with the existing simple-expand, AccVGPR/MFMA, and
unsupported-diagnostics follow-up work rather than this same-size memory audit.

Two CDNA4-to-RDNA3 semantic rules are wired:

| Source row | Bucket | Current behavior |
| --- | --- | --- |
| `SOPP:s_waitcnt` | Same-size semantic rewrite | Converts the GFX9-layout CDNA4 waitcnt immediate to the RDNA3 GFX11 immediate layout in place. |
| `VOP3:v_lshl_add_u64` | Size-expanding rewrite | Emits `v_add_co_u32` and `v_add_co_ci_u32`. The RDNA3 path deliberately skips the RDNA4-only `s_wait_alu`. |

Unhandled `Action::Expand` rows warn and NOP-fill in `BinaryTranslator`.
Unhandled encoding families whose `Action` is not Expand can fall back to
source-copy behavior when `translate_encoding_cdna4_to_rdna3()` returns an empty
translation result. That makes MTBUF and unsupported diagnostics separate
follow-up work even though this taxonomy is complete.

## Recommended Implementation Breakdown

| Area | Scope | Builds on |
| --- | --- | --- |
| Coverage harness | Coverage harness for CDNA4-to-RDNA3 translation, including per-bucket fixtures and warning assertions. | This taxonomy |
| Identity and substitute hardening | Harden identity and opcode-substitute rows, including alias-expanded lookup behavior, duplicate-key expectations, trailing literals, and branch/SOPP smoke cases. | coverage harness |
| Generated legalization duplicate audit | Fix or explicitly document duplicate `(encoding_id, opcode)` keys in alias-expanded legalization headers before relying on header rows as implementation units. | coverage harness and identity/substitute hardening |
| In-place field lower audit | Validate the 629 in-place lower rows by family. Split memory families (SMEM/MUBUF/FLAT/DS) from VALU/SALU families if the task is too large. | coverage harness and identity/substitute hardening |
| Same-size semantic lowerings | Same-size semantic lowerings: precise RDNA3 waitcnt, Lower target opcode preservation, and GFX940/GFX9-to-GFX11 coherency remaps. | coverage harness and identity/substitute hardening |
| Simple expand lowerings | Implement the 353 non-matrix, non-MTBUF Expand rows that have local instruction-sequence lowerings, using code caves and tests for branch-return correctness. | coverage harness |
| MTBUF disposition | Either add MTBUF to the generated CDNA4-to-RDNA3 encoder or classify all 16 MTBUF rows as explicitly unsupported with diagnostics. | coverage harness |
| MFMA/AccVGPR to RDNA3 | Design and implement the 70-row complex matrix/AccVGPR bucket, including lane layout, accumulator remapping, and VGPR pressure diagnostics. | coverage harness |
| Unsupported diagnostics | Make unsupported rows deterministic: no silent source-copy for unhandled Lower families, no NOP-fill without an actionable warning, and diagnostic counts for residual unsupported rows. | coverage harness |

## Representative Examples

| Bucket | Examples |
| --- | --- |
| Identity copy | `s_mov_b32`, `s_mov_b64`, integer `s_cmp_*`, `s_nop`, selected `v_cvt_*`, selected `v_cmpx_*_i16/u16_e32`. |
| Opcode substitute | Most SOPP branches/control ops, many SOP1 scalar ops, VOP2 arithmetic such as `v_add_f32_e32`, and most VOPC comparisons. |
| In-place field remap/lower | DS atomics and LDS ops, FLAT load/store base forms, MUBUF format/load/store rows, SMEM load/cache rows, VOP3 compare and arithmetic rows. |
| Same-size semantic rewrite | `s_waitcnt`. |
| Size-expanding rewrite | `v_lshl_add_u64`, scalar control rows without RDNA3 equivalents, packed or D16 memory rows, many non-matrix `Action::Expand` VOP/SMEM rows. |
| Complex matrix/AccVGPR emulation | `v_mfma_*`, `v_smfmac_*`, `v_accvgpr_mov_b32*`, `v_accvgpr_read`, `v_accvgpr_write`. |
| Residual unsupported/unknown | MTBUF rows until encoder support or explicit rejection is added. |
