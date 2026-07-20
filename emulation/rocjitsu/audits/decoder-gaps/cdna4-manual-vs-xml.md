# CDNA4 Manual vs XML Gaps

Architecture: CDNA4

Manual source: `workspace_docs/amdgpu-isa-manuals/cdna4/README.md`

XML source: `shared/machine-readable-isa/isa/amdgpu_isa_cdna4.xml`

## Coverage

| Manual section | Status | Notes |
| --- | --- | --- |
| 1.1 terminology and 2.1 through 2.3 Program Organization | Audited statically | Checked dispatch/workgroup/wavefront/work-item terms, 64-lane wavefront model, SALU/VALU roles, EXEC-masked vector execution, dispatch grid/work-item indexing, LDS topology overlap, and Section 2.3 device-memory consistency/acknowledgment prose against XML global metadata and instruction descriptions. |
| 3.1 through 3.6 Kernel State | Audited statically | Checked state inventory, PC/EXEC/STATUS/MODE prose, status/mode bitfields, GPR/LDS allocation, aliasing, alignment, out-of-range behavior, and LDS allocation/clamping. |
| 3.7 through 3.13 Kernel State | Audited statically | Checked M0 descriptor roles, SCC, VCC/VCCZ update and alias-hazard prose, trap/exception/TRAPSTS/memory-violation state, HW_ID/XCC_ID bitfields, and compute GPR initialization. |
| 4.1 through 4.2 Program Flow Control | Audited statically | Checked program-control and branch instruction inventory, direct PC operations, trap return, debug conditional branches, `S_CALL_B64`, `S_SETVSKIP`, and fork/join branch encodings against XML opcode records. |
| 4.3 through 4.6 Program Flow Control | Audited statically | Checked workgroup/barrier behavior, wait-counter producer/decrement/ordering rules, manually inserted wait-state hazard tables, and arbitrary divergent-control fork/join pseudocode against XML operand and instruction metadata. |
| 5.1 through 5.2 Scalar ALU Operations | Audited statically | Checked SALU field inventory, scalar selector tables, literal availability, selector reserved rows, source/destination out-of-range behavior, and 64-bit SGPR pair alignment against XML operand records. |
| 5.3 through 5.7 Scalar ALU Operations | Audited statically | Checked SCC producer/consumer rules, signed overflow versus carry behavior, comparison/conditional/bitwise SCC effects, max tie predicates, saveexec/wrexec EXEC side effects, and M0-relative moves. |
| 5.8 Scalar ALU Access Instructions | Audited statically | Checked HWREG operand partition, CDNA4 hardware-register ID map, `IB_STS`/allocation subfields, `S_SETREG_IMM32_B32`, and SETREG write/spacing prose. |
| 6.1 through 6.6 Vector ALU Operations | Audited statically | Checked VALU encoding families, source/literal/M0/implicit-VCC restrictions, literal expansion, instruction inventory, output modifier state interactions, FP round/denorm rules, clamp-bit overloads, VGPR indexing, and SDWA/OPSEL hazard prose. |
| 6.7 Packed Math | Audited statically | Checked packed opcode inventory, MIX selector/ABS overloads, packed 32-bit OPSEL behavior, and alignment/scalar-pair prose. |
| 7.1 through 7.1.4 Dense MFMA rules and layouts | Audited statically | Checked dense MFMA instruction inventory, block/cycle/pass metadata, ACC/ACC_CD register-bank prose, contiguous/aligned register-block requirements, input/output lane formulas, packed item layout, and dense state-control rules. |
| 7.1.5 and 7.1.6 F8F6F4 MFMA rules | Audited statically | Checked A/B matrix format selection via `CBSZ`/`BLGP`, F8F6F4 broadcast-field overrides, denorm/round/execution-mask behavior, source legality, alignment, and modifier/clamp notes. |
| 7.1.6 MFMA Broadcasting values | Audited statically | Checked `CBSZ`/`ABID` A-lane broadcast formula and legality, `BLGP` B-lane permutation table, F64 `BLGP`-as-negation override, and overlap with F8F6F4 field repurposing. |
| 7.2 Block Scaled Matrices | Audited statically | Checked VOP3PX2 load-scale prefix, E8M0 scale semantics, ABID[0], scale source byte selection, scale-source legality, per-instruction scale lifetime, and scale data layout. |
| 7.3 BF8 / FP8 and Smaller Formats and Conversions | Audited statically | Checked small-float table, non-scaled FP8/BF8 converts, scaled FP8/BF8/FP4/FP6/BF6 converts, SDWA rules, `FP16_OVFL`, source legality, `SH_MEM_CONFIG[8]`, scale semantics, stochastic multi-pass wording, and FP4/FP6/BF6 saturation/underflow behavior. Related Chapter 12 instruction-definition coverage is recorded in the rows below. |
| 7.4 Floating-point handling details and formats | Audited statically | Checked per-datatype MFMA denormal handling, F32 A/B input MODE dependence, matrix-C/result denorm preservation, F64 RNE/denorm behavior, I8 integer sign-extension wording, and DGEMM exception support. |
| 7.5 Sparse Matrices | Audited statically | Checked SMFMAC inventory, 4:2 sparse-A/index-VGPR prose, SRC2/ACC_CD rules, `CBSZ`/`ABID` sparse-index selection, index layouts, state/control rows, and VOP3P-MAI field overlap. |
| 7.6 Dependency Resolution: Required Independent Instructions | Audited statically | Checked required NOP/independent-VALU table for DLop/XDLOP/SGEMM/DGEMM/SMFMA interactions, exact-opcode forwarding exceptions, Source C overlap predicates, SrcA/SrcB waits, `V_CMPX*` EXEC-mask forwarding, and XDL/SMFMA-vs-VALU WAR anti-dependency. |
| 8.1 through 8.4 Scalar Memory Operations | Audited statically | Checked all Chapter 8 sections for SMEM encoding fields, scalar/global/scratch/buffer addressing, source-overwrite and clause rules, atomics, cache/time/discard behavior, dependency-counter behavior, and alignment/bounds rules. |
| 9.1 through 9.2 Vector Memory Operations | Audited statically | Checked all Chapter 9 sections: VMEM buffer taxonomy, MUBUF/MTBUF field behavior, VGPR address/data use, buffer data-format selection, D16, address/range/swizzle/resource semantics, buffer-to-LDS, SC/NT cache controls, and Chapter 9.2 floating memory atomic numeric rules for buffer/L2 opcodes. |
| 10.1 through 10.7 Flat Memory Instructions | Audited statically | Checked all Chapter 10 sections for FLAT/GLOBAL/SCRATCH field behavior, address forms, aperture and scratch-space prose, SVE/LDS bit descriptions, direct LDS movement, data movement, atomics, ordering, timing, and VM/LGKM counter rules. |
| 11.1 through 11.4 Data Share Operations | Audited statically | Checked all Chapter 11 sections for LDS geometry/allocation/clamping, dataflow and buffer-to-LDS context, indexed load/store and atomic address rules, two-offset behavior, ADDTID, swizzle/permute lane behavior, floating LDS atomics, append/consume, GDS field wording, and MFMA transpose-load constraints. |
| Representative VALU, DOT, compare, and GPR-index definitions | Audited statically | Checked representative VOP2/VOP3/VOPC/VOP3P DOT, integer/floating VALU, compare/CMPX, carry-out, CNDMASK, and `S_SET_GPR_IDX_*` instruction records. |
| VOP3P instruction definitions | Audited statically | Checked all 104 Chapter 12.10 VOP3P opcode headings 0 through 127 against XML inventory, including packed integer/F16/F32 forms, DOT, MIX, `V_PK_MOV_B32`, `V_ACCVGPR_*`, dense/sparse MFMA/SMFMAC entries, non-scale F8F6F4 MFMA definitions 45-46, and `V_MFMA_SCALE_F32_*_F8F6F4` XML entries. |
| VOP3A/VOP3B instruction definitions | Audited statically | Checked the full Chapter 12.11 VOP3A/VOP3B inventory and selected semantics against XML, including VOP3B's ten scalar-destination opcodes, all native-only `ENC_VOP3` rows after excluding promoted VOP1/VOP2/VOPC aliases, VOP3A/VOP3B field maps, promotion drift, XML-only table rows, F32 min/max3 MODE behavior, `V_CVT_PKRTZ_F16_F32` RTZ description coverage, `V_CVT_PKACCUM_U8_F32` destination-byte dataflow, native F16 OPSEL writeback, `V_BITOP3_*` truth-table field overloads, `V_LSHL_ADD_U64` shift-count rules, F64 min/max MODE behavior, and readlane/writelane EXEC override prose. |
| Instruction definitions 640-659 | Audited statically for VOP3A native arithmetic/lane/count/trig slice | Checked F64 arithmetic/ldexp/min/max, integer multiply, readlane/writelane lane selectors, `V_BCNT`/`V_MBCNT`, 64-bit shifts, `V_TRIG_PREOP_F64`, and `V_BFM_B32`; opcode 654 is absent from both manual and XML. |
| Instruction definitions 660-673 | Audited statically for VOP3A packed conversion/add/sub/pack/legacy-mul slice | Checked packed normalized conversions, packed F32-to-F16 RTZ conversion, packed integer narrowing conversions, signed add/sub saturation notes, `V_PACK_B32_F16`, and `V_MUL_LEGACY_F32`; opcode 667 is absent from both manual and XML. |
| Instruction definitions 680-681 | Audited statically for VOP3A F32 IEEE min/max3 slice | Checked `V_MINIMUM3_F32` and `V_MAXIMUM3_F32` opcode/operand metadata and MODE/DX10_CLAMP notes; XML/prose gap remains `CDNA4-XML-081`. |
| DPP/SDWA instruction limitations | Audited statically | Checked Chapter 12.16 DPP and SDWA exclusion lists against XML DPP/SDWA extension encodings; VOPC DPP availability/limitation drift is tracked separately in `CDNA4-XML-073`, and F64 VOP1 DPP leakage is tracked in `CDNA4-XML-088`. |
| Dense MFMA instruction definitions | Audited statically | Checked representative F32, F16, BF16, I8, F64, FP8, and BF8 dense MFMA definitions for operand shapes, block counts, and pass/cycle metadata. |
| Sparse MFMA instruction definitions | Audited statically | Checked representative BF16, F16, I8, BF8, and FP8 SMFMAC definitions against XML operand metadata and VOP3P-MAI field rules. |
| SMEM instruction definitions | Audited statically for scalar-memory definition slice | Checked Chapter 12.6 definitions and the Chapter 13.2 SMEM opcode table against XML inventory, including scalar/global/scratch/buffer load/store entries, cache maintenance, time, discard, and scalar/buffer atomic families. XML-only ATC probe entries are tracked as source drift. |
| MUBUF/MTBUF instruction definitions | Audited statically for vector-buffer definition slice | Checked Chapter 12.13/12.14 definitions and the Chapter 13.5 MTBUF/MUBUF opcode tables against XML inventory, plus typed, formatted, raw, D16/D16_HI, LDS-capable, cache-maintenance, atomic, and floating-atomic buffer definition metadata. |
| FLAT/GLOBAL/SCRATCH instruction definitions | Audited statically for flat-memory definition slice | Checked Chapter 12.15 definitions and Chapter 13.6 FLAT/GLOBAL/SCRATCH opcode tables against XML inventory, plus overlapping segment encodings, direct-to-LDS entries, atomics, floating atomics, and representative operand metadata. |
| DS instruction definitions | Audited statically for data-share definition slice | Checked Chapter 12.12 definitions and the Chapter 13.4 DS opcode table against XML inventory, plus loads/stores, READ2/WRITE2/ST64, ADDTID, SWIZZLE/PERMUTE/BPERMUTE, packed LDS atomics, APPEND/CONSUME, and transpose-load entries. |
| VOP2 instruction definitions | Audited statically for VOP2 definition slice | Checked all Chapter 12.7 opcode definitions 0 through 61 against XML inventory, VOP2-as-VOP3 `+0x100` mappings, literal-only `_MK`/`_AK` exceptions, DOT2C BF16 modifier prose, FP min/max pseudocode, `V_MAC_F16` OPSEL behavior, `V_LDEXP_F16` literal format, carry-out forms, and bitwise no-modifier notes. |
| VOP1 instruction definitions | Audited statically for VOP1 definition slice | Checked all 85 detailed Chapter 12.8 VOP1 opcode definitions against XML inventory, VOP1-as-VOP3 `+0x140` mappings, Chapter 13 VOP3A table drift, XML-only opcode rows, `V_READFIRSTLANE_B32` EXEC/no-modifier prose, untyped read-write swap operands, FP8/BF8 widening SDWA notes, `V_PRNG_B32`, and VOP1 lookup/exception edge cases. |
| VOPC instruction definitions | Audited statically for VOPC definition slice | Checked all 198 Chapter 12.9 VOPC opcode definitions against XML inventory, VOPC-as-VOP3A aliases, compare predicate tables, class-mask pseudocode, literal/SDWA/DPP second-word forms, DPP limitation rows, and CMPX implicit EXEC metadata. |
| Scalar ALU instruction definitions | Audited statically | Checked the full SOP1, SOP2, SOPK, and SOPC opcode tables and instruction-definition inventories, representative scalar ALU and access instruction definitions, implicit SCC operands, implicit M0 operands, HWREG operands, literal alternatives, instruction-specific no-literal SOP1/SOP2 forms, and the `S_SET_GPR_IDX_ON` raw-mode operand exception; also checked Chapter 12/13 SOPP opcode-table inventory and control/wait/message definitions. |
| Instruction definitions 84-87 | Audited statically for FP8/BF8 widening conversion definitions | Checked SDWA byte/word selection and defaults, ignored SDWA modifier fields, packed 64-bit destination metadata, and VOP3 aliases. |
| Instruction definitions 565-572 | Audited statically for scaled FP8/BF8 conversion definitions | Checked scale exponent source, source/destination `OPSEL` selection, old-destination preservation, stochastic seed use, source classes, and MODE/`FP16_OVFL`/DPP limitations. |
| Instruction definitions 573-611 | Audited statically | Checked OPSEL byte selection, old-destination preservation, wide packed FP6/BF6 layouts, stochastic PRNG advancement, scale extraction, and operand classes. |
| Instruction definitions 613-618 | Audited statically for the conversion slice | Checked `V_ASHR_PK_{I8,U8}_I32`, `V_CVT_PK_{F16,BF16}_F32`, and scaled packed BF16-from-FP8/BF8 forms. |
| Instruction definitions 674-677 | Audited statically for non-scaled FP8/BF8 F32 narrow converts | Checked RNE/stochastic seed behavior, `OPSEL` byte/word writes, old-destination preservation, source legality, and ignored OMOD/clamp notes. |
| Instruction definitions 678-679 | Audited statically for the conversion slice | Checked stochastic F32-to-F16/BF16 narrow converts, OPSEL destination selection, old-destination preservation, and forced-RNE/OMOD notes. |
| 13.1 Scalar ALU and Control Formats | Audited statically | Checked SOP1/SOP2/SOPK/SOPC/SOPP field maps, opcode inventories, literal alternatives, instruction-specific SOP1/SOP2 literal exclusions, implicit SCC operands, implicit M0 relative-move operands, the SOPK literal exception, and the `S_SET_GPR_IDX_ON` raw-mode operand exception. |
| 13.2 Scalar Memory Format | Audited statically | Checked SMEM field inventory, opcode table, signed immediate offset description, `SOFFSET_EN`, `GLC`, `NV`, `SBASE`, and `SDATA` metadata. |
| 13.3.1-13.3.3 and 13.3.7-13.3.9 Vector ALU base and extension formats | Audited statically | Checked VOP2/VOP1/VOPC field maps, opcode inventories, literal/SDWA/DPP second-word availability, SDWA/SDWAB/DPP field maps, DPP control tables, and DPP limitation cross-checks. SDWAB `SD` field prose drift is tracked in `CDNA4-XML-090`. |
| 13.3.4 VOP3A / 13.3.5 VOP3B | Audited statically | Checked VOP3A and VOP3B field maps, OPSEL and modifier fields, VOP3B scalar-destination field prose, the ten-opcode VOP3B table, and VOP3A opcode-table alignment with Chapter 12.11 and XML. |
| 13.3.6 VOP3P / 13.3.6.1 VOP3P-MAI | Audited statically | Checked generic VOP3P and VOP3P-MAI field positions, the manual encoding-width drift, and the full VOP3P opcode table against Chapter 12.10 and XML inventory. |
| 13.5 Vector Memory Buffer Formats | Audited statically | Checked MUBUF/MTBUF field maps, typed/untyped split, DFMT/NFMT field placement, SRSRC scaling prose, LDS bit, SC/NT fields, and opcode tables. |
| 13.6 Flat Formats | Audited statically | Checked FLAT/GLOBAL/SCRATCH field maps, opcode tables, bit-13 naming, SADDR/ADDR/ACC operand roles, and direct-to-LDS opcode entries. |
| 13.4 LDS Format | Audited statically | Checked DS field map, `ACC`, `GDS`, offset fields, operand fields, opcode width, and opcode table coverage against Chapter 11 prose. |
| CDNA4 manual sections through Chapter 13 | Complete | Chapter-level status is tracked in `audit-scope.md`; CDNA4 Chapters 1 through 13 now have section-level manual/XML coverage in this report. |

## Gaps

### CDNA4-XML-001: Small-float numeric semantics are prose-only

Manual evidence:

- `cdna4/README.md:2300` through `:2307` defines FP8, BF8, FP6, BF6, and FP4
  numeric encodings, including bias, zero, Inf/NaN encodings, max values, min
  normal values, and min denormal values.

XML evidence:

- `FMT_NUM_BF6`, `FMT_NUM_FP4`, and `FMT_NUM_FP6` at
  `amdgpu_isa_cdna4.xml:101660` through `:101920` record bit layouts and
  `N/A` descriptions, but not biases, ranges, Inf/NaN absence, or min-normal /
  min-denormal values.
- `FMT_NUM_BF8` at `amdgpu_isa_cdna4.xml:101698` through `:101734` and
  `FMT_NUM_FP8` at `:101926` through `:101960` likewise record only bit layout
  and a `N/A` description.
- Packed small-format entries such as `FMT_NUM_PK2_FP4`, `FMT_NUM_PK32_BF6`,
  and `FMT_NUM_PK32_FP6` at `amdgpu_isa_cdna4.xml:105200` through `:105220`,
  `:106395` through `:106420`, and `:110143` through `:110170` preserve
  component layout but not the numeric interpretation of each component.

Impact:

XML-driven semantics cannot recover the numeric interpretation, edge encodings,
or range limits from the data-format entries alone.

### CDNA4-XML-002: `FP16_OVFL` and `SH_MEM_CONFIG[8]` requirements are absent

Manual evidence:

- `cdna4/README.md:2402` through `:2416` defines the F32 to FP8/BF8
  `FP16_OVFL` result table.
- `cdna4/README.md:2418` says `SH_MEM_CONFIG` bit 8 must be set for correct
  BF8/FP8 operation results.

XML evidence:

- The non-scaled narrow conversion entries are present at
  `amdgpu_isa_cdna4.xml:72364` through `:72520`.
- The scaled F32-to-FP8/BF8 entries are present at
  `amdgpu_isa_cdna4.xml:68345` through `:68525`.
- Searching the CDNA4 XML for `FP16_OVFL` or `SH_MEM_CONFIG` returns no
  matches.

Impact:

The XML cannot describe the state-dependent Inf/overflow behavior for either
non-scaled or scaled FP8/BF8 conversions.

### CDNA4-XML-003: `CVT_SR_*` / `CVT_PK_*` source legality contradicts the manual

Manual evidence:

- `cdna4/README.md:2406` says `CVT_SR_*` and `CVT_PK_*` support only VGPR
  inputs, not SGPRs, literal constants, or inline constants.

XML evidence:

- `V_CVT_PK_FP8_F32` uses `OPR_SRC_NOLIT` for `SRC0` and `OPR_SRC_SIMPLE` for
  `SRC1` at `amdgpu_isa_cdna4.xml:72378` through `:72388`.
- `V_CVT_SR_FP8_F32` uses the same broad source classes at
  `amdgpu_isa_cdna4.xml:72466` through `:72476`; BF8 follows the same pattern
  beginning at `:72496`.
- `OPR_SRC_NOLIT` includes scalar-register subtypes at
  `amdgpu_isa_cdna4.xml:124697` through `:124709`, and `OPR_SRC_SIMPLE`
  includes scalar-register subtypes at `:127127` through `:127138`.

Impact:

The XML advertises non-VGPR source forms that the prose manual rejects. LLVM
assembler and hardware behavior remain separate oracle questions.

### CDNA4-XML-004: SDWA byte/word legality and ignored fields for FP8/BF8 widening converts are not encoded

Manual evidence:

- `V_CVT_F32_FP8` uses BYTE0-3 SDWA selection, defaults to BYTE0 without SDWA,
  and allows only BYTE selects at `cdna4/README.md:9268` through `:9290`.
- `V_CVT_F32_BF8` repeats the BYTE-only rule at `cdna4/README.md:9292`
  through `:9311`.
- `V_CVT_PK_F32_FP8` and `V_CVT_PK_F32_BF8` use WORD0/WORD1 selection and
  allow only WORD selects at `cdna4/README.md:9313` through `:9343`.
- Table 31 says the affected VOP1 converts ignore `abs`, `neg`, and `sext` at
  `cdna4/README.md:2320` through `:2325`, and section 7.3 says the other SDWA
  fields are ignored for converts from 8-bit formats at `:2374` through `:2376`.

XML evidence:

- The SDWA encoding exposes generic `CLAMP`, `DST_SEL`, `DST_UNUSED`, source
  modifier, source select, and sign-extension fields at
  `amdgpu_isa_cdna4.xml:5916` through `:5944` and `:6016` through `:6044`.
- The affected SDWA instruction entries use 8-bit or 16-bit operands, for
  example `amdgpu_isa_cdna4.xml:55221` through `:55235` and `:55449` through
  `:55463`, but do not encode BYTE-only, WORD-only, non-SDWA default selection,
  or instruction-specific ignored-field rules.

Impact:

Consumers need manual overrides to validate legal `SRC0_SEL` values, to know
which source byte/word is implied when SDWA is absent, and to suppress generic
SDWA modifier/destination effects that the prose excludes for this conversion
family.

### CDNA4-XML-005: Partial-destination preservation is under-modeled

Manual evidence:

- Non-scaled packed FP8/BF8 converts preserve the other half of the destination
  at `cdna4/README.md:17858` through `:17895`.
- Non-scaled stochastic converts preserve unwritten bytes at
  `cdna4/README.md:17902` through `:17925` and `:17933` through `:17955`.
- Scaled packed and stochastic FP8/BF8 converts preserve other destination bits
  at `cdna4/README.md:16612` through `:16659`.
- Stochastic F32-to-F16/BF16 converts also preserve the unwritten destination
  word selected by `OPSEL[3]` at `cdna4/README.md:17963` through `:18002`.

XML evidence:

- Non-scaled `V_CVT_PK_FP8_F32`, `V_CVT_PK_BF8_F32`, `V_CVT_SR_FP8_F32`, and
  `V_CVT_SR_BF8_F32` mark `VDST` as output-only at
  `amdgpu_isa_cdna4.xml:72372` through `:72376`, `:72416` through `:72420`,
  `:72460` through `:72464`, and `:72504` through `:72508`.
- Scaled `V_CVT_SCALEF32_{PK,SR}_{FP8,BF8}_F32` entries also mark `VDST` as
  output-only at `amdgpu_isa_cdna4.xml:68356` through `:68360`, `:68409`
  through `:68413`, `:68462` through `:68466`, and `:68515` through `:68519`.
- `V_CVT_SR_F16_F32` and `V_CVT_SR_BF16_F32` describe OPSEL-selected 16-bit
  writes but still mark `VDST` as output-only at `amdgpu_isa_cdna4.xml:72548`
  through `:72552` and `:72592` through `:72596`.

Impact:

The XML misses the old-destination dependency for partial 16-bit and 8-bit
writes.

### CDNA4-XML-006: Scaled conversion semantics and source controls are mostly prose-only

Manual evidence:

- The scaled conversion inventory appears at `cdna4/README.md:2327` through
  `:2356`.
- `cdna4/README.md:2370` through `:2373` says stochastic FP4/FP6 multi-pass
  conversions advance an internal PRNG value every pass without writing the
  source VGPR.
- `cdna4/README.md:2378` through `:2387` gives scaled-conversion rules:
  F4/F6/F8 input modifiers, F32 denorm MODE handling, FP16-overflow support for
  F8 destinations, OMOD/DPP restrictions, E8M0 scale bias 127, inline-constant
  scale handling, and FP4/FP6 saturation/underflow behavior.
- Scaled FP8/BF8 pseudocode extracts an exponent and applies scale helpers at
  `cdna4/README.md:16612` through `:16710`.
- Scaled FP4/FP6/BF6 definitions at `cdna4/README.md:16714` through `:17309`
  add FP4 byte selection and preservation, FP6/BF6 32-component packed layouts,
  interleaved `2XPK16` output ordering, and stochastic per-element PRNG
  advancement.
- Scaled packed BF16-from-FP8/BF8 definitions add `OPSEL[0]` source-word
  selection and E8M0-style scale extraction at `cdna4/README.md:17378` through
  `:17403`.

XML evidence:

- The XML has instruction shells and aliases for scaled FP8/BF8 conversions at
  `amdgpu_isa_cdna4.xml:68345` through `:68525`, `:68898` through `:69465`,
  and `:70862` through `:70935`.
- The XML has instruction shells, aliases, operand widths, and broad
  descriptions for scaled FP4/FP6/BF6 conversions at
  `amdgpu_isa_cdna4.xml:68745` through `:68880` and `:69486` through `:70655`.
- Scale operands are ordinary `FMT_NUM_F32` / `OPR_SRC_SIMPLE` operands, for
  example `amdgpu_isa_cdna4.xml:68774` through `:68778`, `:69810` through
  `:69813`, and `:70110` through `:70113`.
- `OPR_SRC_SIMPLE` includes scalar-register source classes at
  `amdgpu_isa_cdna4.xml:127127` through `:127138`, while the manual says scale
  comes from a VGPR or inline constant.
- Searching the CDNA4 XML for `E8M0` returns no matches.

Impact:

The XML can identify the instruction names and operands, but cannot generate or
validate the scale exponent semantics, modifier restrictions, denorm behavior,
scale-source legality, stochastic PRNG sequence, FP4/FP6 saturation/underflow,
or FP16-overflow behavior without manual overrides.

### CDNA4-XML-007: FP8/BF8 forwarding hazard is prose-only

Manual evidence:

- `cdna4/README.md:2360` says the `CVT_*_F32` instructions in this table do not
  support 4-cycle forwarding and require a NOP or unrelated destination write
  between conversions writing the low/high half or bytes of the same
  destination register.

XML evidence:

- The audited FP8/BF8 conversion instruction entries do not carry a scheduling,
  forwarding, or producer-hazard annotation.

Impact:

Instruction scheduling and hazard-aware fuzzing cannot derive this rule from
XML alone.

### CDNA4-XML-008: VOP3P `OPSEL_HI2` is merged into `OP_SEL_HI` and under-described

Manual evidence:

- The CDNA4 VOP3P field table names bit 14 separately as `OPSEL_HI2`, says it
  selects low or high for high sources, and maps high-source selector bits as
  source 0 = bit 14, source 1 = bit 60, and source 2 = bit 59 at
  `cdna4/README.md:25990` through `:26048`.

XML evidence:

- The CDNA4 XML has `OP_SEL_HI` with a two-range bit layout covering bits
  59:60 and bit 14 at `amdgpu_isa_cdna4.xml:2272` through `:2285`.
- Its description still says only "high opcode src0 and src1" and does not name
  `OP_SEL_HI_2` or document bit 14's source mapping.

Impact:

The bit is present, but XML consumers cannot recover the manual field name or
the per-source high-selector mapping from the field description alone.

### CDNA4-XML-009: Packed 32-bit VOP3P dword selection and `V_PK_MOV_B32` special rules are prose-only

Manual evidence:

- `cdna4/README.md:1571` says packed 32-bit instructions operate on two dwords,
  require two-dword alignment, do not support output modifiers, and use
  `OPSEL`/`OPSEL_HI` to select the first or second source dword.
- `V_PK_FMA_F32`, `V_PK_MUL_F32`, and `V_PK_ADD_F32` read source dwords
  `[31:0]` and `[63:32]` at `cdna4/README.md:12190` through `:12217`.
- `V_PK_MOV_B32` uses only `OPSEL[0]` and `OPSEL[1]` for the two output dwords,
  treats sources as 64-bit operands, supports VGPR gather from the even register
  or next odd register, and restricts two-SGPR forms to the same SGPR at
  `cdna4/README.md:12219` through `:12245`.

XML evidence:

- Generic `ENC_VOP3P` still describes two 16-bit operations at
  `amdgpu_isa_cdna4.xml:2208` and describes selector fields as 16-bit half
  selectors at `:2262` through `:2285`.
- The packed F32 and `V_PK_MOV_B32` entries use 64-bit `FMT_NUM_PK2_F32` and
  `FMT_NUM_PK2_B32` operands at `amdgpu_isa_cdna4.xml:74332` through `:74493`,
  and the shared operand definitions carry generic 64-bit SGPR/VGPR
  even-alignment text at `:124715` and `:125846`.
- The XML entries do not encode the packed 32-bit dword selector override,
  `V_PK_MOV_B32`'s special use of `OPSEL` without `OPSEL_HI`, its VGPR gather
  semantics, its unsupported-output-modifier statement, or the same-SGPR
  restriction for two scalar sources.

Impact:

CDNA4 XML is explicit that these are packed two-component 32-bit operands, but
the packed 32-bit selector contract and `V_PK_MOV_B32` special restrictions
still require manual prose.

### CDNA4-XML-010: MIX-specific VOP3P selector and modifier overloads are prose-only

Manual evidence:

- Section 6.7 says `V_MAD_MIX_*` are not packed math but use the VOP3P encoding
  at `cdna4/README.md:1569`.
- Each MIX instruction says `{OPSEL_HI[i], OPSEL[i]}` chooses between a full
  32-bit F32 input and the low/high F16 half, and says `NEG_HI` acts as an
  absolute-value modifier at `cdna4/README.md:11943`, `:11970`, and `:11993`.

XML evidence:

- Generic `ENC_VOP3P` describes `NEG_HI` as high-half floating negation at
  `amdgpu_isa_cdna4.xml:2242` and describes the selector fields as generic
  16-bit half selectors at `:2262` through `:2285`.
- `V_MAD_MIX_F32`, `V_MAD_MIXLO_F16`, and `V_MAD_MIXHI_F16` are present at
  `amdgpu_isa_cdna4.xml:73732`, `:73782`, and `:73832`, but their entries do
  not encode the MIX-only selector mapping or `NEG_HI` absolute-value behavior.

Impact:

Generated code needs manual-derived special cases for MIX source selection and
`NEG_HI`; the XML field descriptions imply the generic packed-half behavior.

### CDNA4-XML-011: Block-scaled MFMA scale semantics are prose-only

Manual evidence:

- Section 7.2 defines block-scaled matrix behavior, including block size 32,
  E8M0 scale bias 127, `0xff` as NaN, and scale application after each block's
  dot product at `cdna4/README.md:2236` through `:2242`.
- The MFMA block-scale section says scale values are supplied by a combined
  four-dword load-scale plus MFMA instruction and only apply to one instruction
  at `cdna4/README.md:2246` through `:2254`.
- The F8F6F4 MFMA rules say `ABID[0]=1` must be set for the scale forms,
  while `ABID[0]=0` forces scale 1.0 and runs without scale source at
  `cdna4/README.md:2042` through `:2050`.
- Scale data layouts for the 16x16x128 and 32x32x64 shapes are described at
  `cdna4/README.md:2278` through `:2294`.

XML evidence:

- The XML has `V_MFMA_SCALE_F32_16X16X128_F8F6F4` and
  `V_MFMA_SCALE_F32_32X32X64_F8F6F4` entries with `ENC_VOP3PX2` and scale
  operands at `amdgpu_isa_cdna4.xml:77932` through `:78047`.
- Those entries reuse ordinary MFMA descriptions and do not mention E8M0,
  block size, `ABID[0]` scale enable behavior, per-instruction scale lifetime,
  scale application point, `0xff` NaN, or scale lane/K layout.
- Searching the CDNA4 XML for `E8M0` or `ABID[0]` returns no matches.

Impact:

XML consumers can identify the scale opcodes and operand fields, but cannot
derive the scale arithmetic, enable behavior, lifetime, or layout without the
manual prose.

### CDNA4-XML-012: VOP3PX2 scale source selection is under-modeled

Manual evidence:

- The load-scale prefix maps `SRC0` and `SRC1` byte selection through
  `{OP_SEL_HI[0], OP_SEL[0]}` and `{OP_SEL_HI[1], OP_SEL[1]}`, with codes for
  bytes `[7:0]`, `[15:8]`, `[23:16]`, and `[31:24]`, at
  `cdna4/README.md:2265` through `:2275`.
- The manual limits scale values to VGPRs or inline constants, using the float
  exponent portion, at `cdna4/README.md:2276`.

XML evidence:

- `ENC_VOP3PX2` includes `SCALE_SRC0` and `SCALE_SRC1` fields at
  `amdgpu_isa_cdna4.xml:2077` through `:2095`, but their descriptions are only
  "First operand for instruction" and "Second operand for instruction."
- The same encoding keeps generic 16-bit-half descriptions for `OP_SEL` and
  `OP_SEL_HI` at `amdgpu_isa_cdna4.xml:2053` through `:2076`; it does not
  describe the scale-prefix byte-selection override.
- The scale operands use `OPR_SRC_SIMPLE` at `amdgpu_isa_cdna4.xml:77964`
  through `:77974` and `:78026` through `:78036`. `OPR_SRC_SIMPLE` includes
  scalar-register source classes at `amdgpu_isa_cdna4.xml:127127` through
  `:127138`, while the manual describes only VGPR and inline-constant scale
  sources.

Impact:

The XML exposes the bits, but not the special byte-selector meaning or the
manual's narrower scale-source contract.

### CDNA4-XML-013: F8F6F4 MFMA format and modifier rules are prose-only

Manual evidence:

- Section 7.1.5 says `V_MFMA_F32_*_F8F6F4` uses `CBSZ[2:0]` for matrix A
  format and `BLGP[2:0]` for matrix B format, with values 0=`FP8 E4M3`,
  1=`BF8 E5M2`, 2=`FP6 E2M3`, 3=`BF6 E3M2`, and 4=`FP4 E2M1`, at
  `cdna4/README.md:2010` through `:2028`.
- The same rule table says these instructions ignore MODE denorm control, use
  RNE regardless of MODE rounding, ignore the execution mask, require even
  VGPR alignment, restrict `SRC0`/`SRC1` to VGPR/AccVGPR, and allow `SRC2` as
  VGPR/AccVGPR/constant at `cdna4/README.md:2029` through `:2040`.
- Sections 7.1.6.2 and 7.1.6.5 explicitly say `CBSZ` and `BLGP` are
  repurposed as data-type controls for F8F6F4 MFMA, behaving as if broadcast is
  disabled, at `cdna4/README.md:2196` through `:2234`.
- The detailed opcode notes repeat the format table, say `NEG[1:0]` and
  `ABS[1:0]` must be zero, allow `NEG[2]` and `ABS[2]` for matrix C, say
  `CLAMP` is not supported, and require RNE at `cdna4/README.md:12121`
  through `:12134` and `:12154` through `:12172`.

XML evidence:

- The non-scale F8F6F4 MFMA opcode entries are present with `VOP3P_MFMA`
  encoding, opcode 45/46, 256-bit A/B sources, 128/512-bit F32 C/D operands,
  and the expected VGPR/AccVGPR/constant source classes at
  `amdgpu_isa_cdna4.xml:74232` through `:74314`.
- Generic `VOP3P_MFMA` field descriptions still describe `CBSZ` as A-matrix
  swizzling/broadcast control, `BLGP` as the ordinary B-matrix lane-group
  control, and `CLAMP` as `[0.0, 1.0]` output clamping at
  `amdgpu_isa_cdna4.xml:1996` through `:2009`, `:2210` through `:2219`, and
  `:2251` through `:2284`.
- The XML entries do not encode the F8F6F4-specific `CBSZ`/`BLGP` data-format
  table, the broadcast-disable override, modifier validity constraints, the
  C-only modifier allowance, the detailed `CLAMP` unsupported note, or the
  MODE-independent denorm/round/execution-mask rules. Searching the CDNA4 XML
  for `FP_ROUND`, `FP_DENORM`, or `FP16_OVFL` finds no matching state metadata.

Impact:

XML consumers can recover opcode inventory and operand classes, but cannot
validate or emulate the F8F6F4-specific field meanings and state/modifier rules
without manual-derived overrides.

### CDNA4-XML-014: Dense MFMA layout, timing, and state rules are prose-only

Manual evidence:

- Section 7.1 defines dense MFMA semantics as
  `D[b,i,j] = C[b,i,j] + sum_k A[b,i,k] * B[b,k,j]`, says `ACC[0]` and
  `ACC[1]` select VGPR versus AccVGPR storage for matrices A and B, says
  `ACC_CD` selects VGPR versus AccVGPR storage for C and D, and requires
  contiguous register blocks whose first register is aligned to the number of
  registers required by the operand at `cdna4/README.md:1632` through `:1650`.
- Table 28 lists dense MFMA variants, block counts, and cycles at
  `cdna4/README.md:1684` through `:1719`.
- The dense MFMA rule table, excluding F8F6F4, says MFMA ignores MODE denorm
  control while keeping input/output denorms, forces RNE, does not support
  exceptions, ignores the execution mask, requires VGPR alignment for
  `SRC0`/`SRC1`/`SRC2`/`VDST`, allows inline/constant `SRC2`, and describes a
  clamp/`FP16_OVFL` overflow policy at `cdna4/README.md:1721` through `:1738`.
  Section 7.4 later qualifies the denorm and exception rows per data type; see
  `CDNA4-XML-018`.
- Sections 7.1.3 and 7.1.4 define the storage layout examples and formulas:
  input `K_L`, item selection, A/B lane formulas, output `H`, `B_I`, `M_I`,
  `G`, item formula, lane formula, little-endian packing for 64/32/16/8/4/6-bit
  items, and the special F64 output layout at `cdna4/README.md:1741` through
  `:2005`.
- Detailed dense MFMA instruction definitions record per-op pass counts, for
  example F32/F16/I8/F64 definitions at `cdna4/README.md:12418` through
  `:13088`.

XML evidence:

- The `VOP3P_MFMA` encoding records fields for `ABID`, `ACC`, `ACC_CD`,
  `BLGP`, `CBSZ`, `OP`, `SRC0`, `SRC1`, `SRC2`, and `VDST` at
  `amdgpu_isa_cdna4.xml:7398` through `:7590`.
- Representative dense MFMA opcode entries record opcode, operand classes,
  operand sizes, and packed data-format names, for example
  `V_MFMA_F32_32X32X1_2B_F32` through `V_MFMA_F32_32X32X2_F32` at
  `amdgpu_isa_cdna4.xml:74964` through `:75210`, I8 at `:75794` through
  `:75826`, and F64 at `:76978` through `:77063`.
- Packed data formats such as `FMT_NUM_PK16_F32`, `FMT_NUM_PK32_F32`,
  `FMT_NUM_PK4_F32`, and `FMT_NUM_PK4_F64` record component count and bit
  layout at `amdgpu_isa_cdna4.xml:103523` through `:103535`, `:109206`
  through `:109220`, and `:113042` through `:113180`, but not the MFMA
  lane/item placement formulas.
- Operand classes such as `OPR_SRC_VGPR_OR_ACCVGPR`,
  `OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST`, and `OPR_VGPR_OR_ACCVGPR` carry generic
  "64-bit and wider" even-alignment prose at `amdgpu_isa_cdna4.xml:130844`
  through `:130855`, `:133418` through `:133430`, and `:142268` through
  `:142279`. They do not encode the manual's stronger per-instruction
  first-register alignment to the full register-block width.
- Searching the CDNA4 XML for the dense MFMA formula names, pass/cycle terms,
  `FP16_OVFL`, denorm/round-mode state, and exception behavior finds no
  instruction-specific metadata. The `VOP3P_MFMA` field map also has no
  `CLAMP` field even though the dense MFMA rule table describes clamp behavior.

Impact:

XML consumers can enumerate dense MFMA opcodes and operand widths, but cannot
derive layout math, cycle/pass metadata, precise register-block legality,
execution-state behavior, or the dense MFMA clamp/overflow contract without
manual-derived rules. The clamp row also needs manual reconciliation because it
is not represented in the VOP3P-MAI bit layout.

### CDNA4-XML-015: MFMA broadcast permutations and F64 field overrides are under-described

Manual evidence:

- Section 7.1.6 says `CBSZ`, `ABID`, and `BLGP` affect lane retrieval after
  the A/B input layout is computed at `cdna4/README.md:2158` through `:2166`.
- Section 7.1.6.1 defines `S = 64 / (1 << CBSZ)`, the A-lane permutation
  `p_a(l_a) = (l_a % S) + (S * ABID)`, the largest legal `CBSZ` value as 4,
  an undefined case when `(1 << CBSZ)` exceeds the instruction's block count,
  and an illegal case when `ABID >= (1 << CBSZ)` at `cdna4/README.md:2164`
  through `:2188`.
- Section 7.1.6.3 defines every `BLGP` value 0 through 7 with exact B-lane
  formulas, including the rotate-16 form for value 3, at
  `cdna4/README.md:2200` through `:2220`.
- Section 7.1.6.4 says F64 MFMA ignores `CBSZ`/`ABID` and repurposes
  `BLGP[0]`, `BLGP[1]`, and `BLGP[2]` as implicit negation of matrices A, B,
  and C at `cdna4/README.md:2220` through `:2230`.
- The manual's VOP3P-MAI field table also states legal `CBSZ` bounds and says
  F64 MFMA uses the `BLGP` field as `NEG` instead of lane-group permutation at
  `cdna4/README.md:26051` through `:26074`.

XML evidence:

- The `VOP3P_MFMA` XML field map records `ABID`, `BLGP`, and `CBSZ` bit
  positions at `amdgpu_isa_cdna4.xml:7398` through `:7590`.
- The XML `ABID`/`CBSZ` descriptions only say A-matrix swizzling is
  fine-grained and independent of `BLGP`, and the `BLGP` description only says
  it controls B-matrix input swizzling among 16-lane groups at
  `amdgpu_isa_cdna4.xml:7487` through `:7528`.
- The XML does not encode the `CBSZ` legal-value limit, the per-instruction
  `(1 << CBSZ) <= blocks` constraint, the `ABID < (1 << CBSZ)` constraint, the
  exact A/B lane-permutation formulas, or the F64 `BLGP`-as-A/B/C-negation
  override. Searching for the F64 MFMA entries only recovers ordinary operand
  metadata at `amdgpu_isa_cdna4.xml:76978` through `:77063`.

Impact:

XML consumers cannot validate MFMA broadcast encodings or emulate ordinary
MFMA/F64 broadcast-field semantics from XML alone. The XML exposes the bits, but
the actionable lane formulas and alternate F64 meaning require manual-derived
rules.

### CDNA4-XML-016: Sparse MFMA structure and selector rules are mostly prose-only

Manual evidence:

- Section 7.5 says `V_SMFMAC` performs sparse matrix multiply-add using a 4:2
  sparse matrix A, dense matrix B, and C/D in the destination register block at
  `cdna4/README.md:2432` through `:2440`.
- The SMFMAC inventory and cycle table covers BF16, F16, I8, BF8, and FP8
  sparse forms at `cdna4/README.md:2442` through `:2460`.
- SMFMAC uses `SRC2` as sparse-index storage rather than an ordinary C matrix,
  while the accumulator C matrix is loaded from `VDST`, at `cdna4/README.md:2462`
  through `:2480`.
- The sparse rules require `SRC2` to be an Arch VGPR, require `index0 < index1`
  and `index0 != index1` in each sparse index pair, require even VGPR alignment
  for `SRC0`, `SRC1`, and `VDST`, and say `CBSZ`/`ABID` only select sparse
  index sets rather than affecting A broadcast at `cdna4/README.md:2472`
  through `:2478`.
- The sparse state table records denorm, round, exception, execution-mask,
  clamp/`FP16_OVFL`, and operand-restriction behavior at
  `cdna4/README.md:2484` through `:2496`.
- Sections 7.5.1.1 through 7.5.1.3 define 16-bit and 8-bit sparse index
  layouts, including how `CBSZ` and `ABID` select alternate index sets for
  16-bit forms and how large 8-bit/F8 forms ignore those selectors, at
  `cdna4/README.md:2504` through `:2692`.

XML evidence:

- The XML `VOP3P_MFMA` field map records generic `ABID`, `ACC_CD`, `BLGP`, and
  `CBSZ` fields at `amdgpu_isa_cdna4.xml:7480` through `:7530`, but keeps the
  ordinary A/B swizzle descriptions. Its `ACC_CD` description mentions a
  destination-only flag concept, but the instruction entries do not carry a
  searchable SMFMAC-specific semantic payload for that flag.
- Representative SMFMAC entries such as `V_SMFMAC_F32_16X16X64_BF16`,
  `V_SMFMAC_I32_16X16X128_I8`, and
  `V_SMFMAC_F32_16X16X128_BF8_BF8` are present at
  `amdgpu_isa_cdna4.xml:74710` through `:74825`, with `SRC2` encoded as a
  32-bit `OPR_SRC_VGPR` operand.
- Additional SMFMAC entries appear later in the VOP3P-MAI opcode table, for
  example `amdgpu_isa_cdna4.xml:75279` through `:75330`, `:76291` through
  `:76823`, and `:77508` through `:77883`.
- The XML entries do not encode the 4:2 sparse reconstruction contract,
  sparse-index pair legality, sparse index layouts, `CBSZ`/`ABID` index-set
  selection and ignore rules, pass/cycle metadata, or the SMFMAC state/control
  table.

Impact:

XML consumers can enumerate CDNA4 SMFMAC opcodes and recover basic operand
classes, including the important `SRC2` VGPR-only shape, but cannot validate or
emulate sparse matrix semantics from XML alone.

### CDNA4-XML-017: MAI dependency-wait rules are prose-only

Manual evidence:

- Section 7.6 says its table identifies timing conditions requiring user-inserted
  NOPs or independent VALU instructions, and defines `DLop`, `XDLOP`, `DGEMM`,
  and `PASS` at `cdna4/README.md:2697` through `:2709`.
- The table records specific wait counts for non-DLop VALU writes feeding
  MFMA/SMFMA reads, DLop exact-opcode Source C forwarding, disabled SrcA/SrcB
  forwarding, XDL/SMFMA/SGEMM/DGEMM Source C overlaps, SrcA/SrcB reads, VM/LDS/
  FLAT/Export overlaps, and VALU RAW+WAW cases at `cdna4/README.md:2711`
  through `:2779`.
- The same section also records a `V_CMPX*` EXEC-mask forwarding hazard for
  `V_MFMA*`, plus an XDL/SMFMA Source C read versus VALU write WAR
  anti-dependency, at `cdna4/README.md:2780` through `:2787`.

XML evidence:

- The CDNA4 XML exposes generic `OPR_WAITCNT` bit fields and generic functional
  groups including `VALU`, `VMEM`, `EXPORT`, and `WAVE_CONTROL` at
  `amdgpu_isa_cdna4.xml:146132` through `:146207`.
- Representative SMFMAC entries carry ordinary operand metadata and the
  `VALU`/`MFMA` subgroup, for example
  `V_SMFMAC_F32_16X16X64_BF16` at `amdgpu_isa_cdna4.xml:74714` through
  `:74754`.
- The XML does not encode the Section 7.6 wait table, the DLop/XDLOP/SGEMM/
  DGEMM scheduling classes, pass-count-derived wait tiers, exact-same-opcode
  forwarding exceptions, Source C overlap predicates, `V_CMPX*` execution-mask
  forwarding hazard, or the XDL/SMFMA-versus-VALU WAR anti-dependency.

Impact:

XML consumers can identify MFMA/SMFMA instructions and wait-counter operands,
but cannot derive the required independent-instruction scheduling contract for
MAI hazards from XML alone. Any scheduler, validator, or diagnostic tool needs
manual-derived rules for this table.

### CDNA4-XML-018: MAI floating-point mode and exception rules are prose-only

Manual evidence:

- Section 7.4 says MAI denormal handling varies by instruction datatype and, in
  some cases, MODE state at `cdna4/README.md:2420` through `:2423`.
- `V_MFMA_F32_*_F32` instructions honor MODE denormal flags for their 32-bit
  inputs, while matrix-C input and result-matrix output ignore `MODE.denorm` and
  do not flush denormals at `cdna4/README.md:2424` through `:2425`.
- MFMA instructions taking sub-32-bit float inputs (`F16`, `BF16`, `BF8`,
  `FP8`) ignore `MODE.denorm` and preserve denormals, while
  `V_MFMA_F64_*_F64` ignores MODE, rounds RNE, and allows input/output denorms
  at `cdna4/README.md:2426` through `:2427`.
- `V_MFMA_I32_*_I8` does integer multiply-add, ignores MODE, and sign-extends
  16-bit intermediate products/results before accumulation at
  `cdna4/README.md:2428`.
- The matrix core generally does not support arithmetic exceptions, except that
  DGEMM matrix operations do support exceptions at `cdna4/README.md:2430`.

XML evidence:

- Representative F32-input MFMA entries such as
  `V_MFMA_F32_32X32X1_2B_F32` record operand data formats and sizes, including
  F32 `SRC0`/`SRC1` and packed F32 `SRC2`/`VDST`, at
  `amdgpu_isa_cdna4.xml:74964` through `:75008`.
- Representative F64/DGEMM entries such as `V_MFMA_F64_16X16X4_F64` and
  `V_MFMA_F64_4X4X4_4B_F64` record F64 operand formats and sizes at
  `amdgpu_isa_cdna4.xml:76978` through `:77070`.
- The XML instruction records expose data formats and broad `VALU`/`MFMA`
  subgroup classification, but do not encode operand-specific denormal
  handling, F32 A/B input MODE dependence, matrix-C/result denorm preservation,
  sub-32-bit MODE-ignore behavior, F64 forced-RNE behavior, I8 intermediate
  sign-extension text, or DGEMM arithmetic-exception support. Searching for
  `FP_DENORM`, `FP_ROUND`, `DGEMM`, and MFMA-specific exception metadata finds
  no structured entries for these rules.

Impact:

XML consumers can classify the MFMA operand data formats but cannot derive the
manual's per-datatype floating-point state contract from XML alone. Correct
simulation, validation, or compiler diagnostics need manual-derived rules for
which MFMA inputs observe MODE, which operands always preserve denormals, and
which MAI operations support exceptions.

### CDNA4-XML-019: SMEM addressing, dependency, clause, and bounds rules are prose-only

Manual evidence:

- Chapter 8 says SMEM loads/stores move one to sixteen dwords through the
  scalar data cache, write one to four dwords, and do no byte/short formatting
  at `cdna4/README.md:2791` through `:2793`.
- Table 39 defines `IMM`, `GLC`, `SDATA`, `SBASE`, `OFFSET`, `NV`, and `SOE`
  behavior at `cdna4/README.md:2797` through `:2824`, including store/atomic
  offset restrictions and SDATA/SBASE register-class and alignment rules.
- Section 8.2.1.1 gives separate scalar/global, scratch, and buffer SMEM
  address formulas, including `IMM`/`SOE` selection, M0/SGPR offset forms,
  scratch 64-byte offset scaling, low-bit masking, private-access conversion,
  scalar-buffer descriptor fields, and stride-only bounds use at
  `cdna4/README.md:2826` through `:2890`.
- Sections 8.2.2 through 8.4 describe scalar atomic return behavior, cache/time
  instructions, scalar-memory source-overwrite and clause hazards, `LGKM_CNT`
  increment/decrement behavior, out-of-order/partial return behavior, and
  SDATA/SBASE alignment/range and out-of-bounds rules at
  `cdna4/README.md:2892` through `:2938`.
- Chapter 13.2 records the SMEM field map and states that `OFFSET` can be an
  immediate signed byte offset or an SGPR address holding an unsigned byte
  offset at `cdna4/README.md:25035` through `:25110`.

XML evidence:

- `ENC_SMEM` carries field bit positions and short field descriptions at
  `amdgpu_isa_cdna4.xml:741` through `:955`, including `SOFFSET_EN`, `IMM`,
  `GLC`, `NV`, `SBASE`, `SDATA`, `SOFFSET`, and signed immediate-offset text.
- The XML has the audited SMEM instruction entries and opcode inventory, for
  example scalar loads/stores/cache/timer/discard entries at
  `amdgpu_isa_cdna4.xml:28166` through `:29538` and scalar atomics beginning at
  `:29660`.
- `FMT_RSRC_SCALAR` records only a generic 128-bit descriptor payload at
  `amdgpu_isa_cdna4.xml:115275` through `:115293`.
- Searching the CDNA4 XML finds no scalar-memory clause, source-overwrite,
  scratch scaling, low-bit masking, buffer-resource bounds, private scalar
  access conversion, or `LGKM_CNT` multi-dword weighting semantics.

Impact:

The XML is sufficient to decode the SMEM fields, but an XML-only consumer cannot
derive the address formulas, dependency-counter behavior, legality restrictions,
or scalar-buffer/bounds behavior needed for semantic emulation and validation.

### CDNA4-XML-020: Scalar atomic return is not gated by `GLC` in operand metadata

Manual evidence:

- Table 39 says `GLC=1` makes scalar atomics return the pre-op value at
  `cdna4/README.md:2811`.
- Section 8.2.2 says scalar atomics use the same address calculations as scalar
  loads/stores and return the pre-operation value to `SDATA` only when `GLC` is
  set at `cdna4/README.md:2892` through `:2895`.

XML evidence:

- Representative scalar-buffer and scalar atomics such as `S_BUFFER_ATOMIC_ADD`
  and `S_ATOMIC_ADD` are encoded with `ENC_SMEM` and classified as `SMEM` /
  `ATOMIC`, but mark `SDATA` as `Input="false"` and `Output="true"`
  unconditionally at `amdgpu_isa_cdna4.xml:29660` through `:29706` and
  `:30804` through `:30850`.
- The generic `GLC` field description at `amdgpu_isa_cdna4.xml:858` through
  `:867` only says the operation is globally coherent and does not encode the
  atomic-specific conditional return contract.

Impact:

An XML-driven generator can model scalar atomic data registers as always written
and never read, even though the manual requires the data operand as an input and
only writes the pre-op value back when `GLC` is set.

### CDNA4-XML-021: Buffer offset and address-VGPR rules are partly misstated

Manual evidence:

- The MUBUF/MTBUF field table says `VADDR` supplies offset, index, or both, and
  `SOFFSET` supplies an unsigned byte offset from an SGPR, M0, or inline
  constant at `cdna4/README.md:2988` through `:3004`.
- The address-VGPR table says zero, one, or two VGPRs are consumed based on
  `IDXEN`/`OFFEN` at `cdna4/README.md:3006` through `:3022`.
- Section 9.1.5 says the instruction offset is present regardless of `OFFEN`,
  and that the offset can come from an SGPR or VGPR and from the instruction
  itself at `cdna4/README.md:3061` through `:3073`.
- Chapter 13.5 repeats the MTBUF/MUBUF field maps, including `OFFEN`, `IDXEN`,
  `VADDR`, `SRSRC`, and `SOFFSET`, at `cdna4/README.md:26356` through `:26420`.

XML evidence:

- `ENC_MUBUF` and `ENC_MTBUF` carry the relevant field bit positions, but their
  `OFFEN` descriptions say that only one of the VGPR offset and instruction
  offset may be sent at `amdgpu_isa_cdna4.xml:3364` through `:3365` and
  `:3560` through `:3561`.
- Representative MUBUF entries encode `VADDR` as a fixed 64-bit operand, for
  example `BUFFER_LOAD_FORMAT_D16_X` at `amdgpu_isa_cdna4.xml:24405` through
  `:24410` and `BUFFER_ATOMIC_ADD` at `:26350` through `:26355`.

Impact:

The XML is sufficient to decode the fields, but an XML-only consumer can derive
the wrong address expression when `OFFEN` is set and can over-consume VADDR
registers for instructions that use zero or one address VGPR.

### CDNA4-XML-022: Buffer resource, range, swizzle, and unbound rules are prose-only

Manual evidence:

- Section 9.1.5 defines buffer addressing from a resource base, SGPR offset,
  VGPR index/offset, stride, element size, `ADD_TID`, swizzle fields, and
  `NumRecords` at `cdna4/README.md:3061` through `:3100`.
- Section 9.1.5.1 defines distinct private, raw, and structured range-checking
  modes, the `dst_sel = SEL_1` out-of-bounds exception, and all-or-nothing
  versus per-component behavior at `cdna4/README.md:3105` through `:3128`.
- Sections 9.1.5.2 and 9.1.8 define swizzled-addressing formulas and the
  128-bit buffer-resource descriptor fields, including base, stride, swizzle,
  `NumRecords`, `dst_sel`, `NFMT`, `DFMT`, user VM, `ADD_TID`, `NV`, type, and
  the all-zero unbound-resource behavior at `cdna4/README.md:3129` through
  `:3147` and `:3190` through `:3220`.

XML evidence:

- The `FMT_RSRC*` data formats describe a 128-bit buffer resource constant as a
  single opaque `Descriptor` field at `amdgpu_isa_cdna4.xml:115255` through
  `:115360`.
- The checked MUBUF/MTBUF encodings and instruction entries do not encode the
  descriptor bit layout, address formulas, swizzle formulas, private/raw/
  structured range modes, unbound-resource behavior, or `dst_sel` OOB
  exception.

Impact:

XML consumers cannot model most architecturally important buffer-resource
semantics without manual prose or a separate descriptor schema.

### CDNA4-XML-023: Buffer data-format, D16, and `dst_sel` semantics are incomplete

Manual evidence:

- Section 9.1.3 describes read/write data VGPR counts and buffer data-format
  conversion at `cdna4/README.md:3023` through `:3029`.
- Section 9.1.4 says data size and type are controlled by `DFMT`, `NFMT`,
  `dst_sel`, and opcode, distinguishes MTBUF instruction format from MUBUF
  resource/derived format, states that an INVALID resource format remains
  unbound rather than being replaced by a derived format, and defines D16
  packing at `cdna4/README.md:3031` through `:3059`.
- Section 9.1.6 says ECC-enabled 16-bit memory loads write a full 32-bit VGPR
  with unused bits zeroed at `cdna4/README.md:3175` through `:3184`.
- Chapter 13.5 lists MTBUF `DFMT`/`NFMT` values and MUBUF D16 opcodes at
  `cdna4/README.md:26364` through `:26444`.

XML evidence:

- `ENC_MTBUF` exposes `DFMT` and `NFMT` fields at
  `amdgpu_isa_cdna4.xml:3510` through `:3547`, and representative formatted
  buffer entries describe conversion at `amdgpu_isa_cdna4.xml:24391` through
  `:24448`.
- The resource descriptor remains opaque in `FMT_RSRC*` at
  `amdgpu_isa_cdna4.xml:115255` through `:115360`, and the XML entries do not
  encode the full data-format enum semantics, `dst_sel` channel selection,
  INVALID/unbound override rule, D16 pair packing, or the ECC writeback rule.

Impact:

The XML tells a generator that formatted buffer operations exist, but not enough
to implement the typed/resource data conversion and destination-channel
behavior specified by the manual.

### CDNA4-XML-024: Buffer-to-LDS and vector cache controls are prose-only

Manual evidence:

- Section 9.1.9 limits memory-buffer load-to-LDS to a subset of MUBUF
  instructions, defines the LDS offset as `M0[17:0]`, and gives active-mask and
  LDS-allocation clamping rules at `cdna4/README.md:3222` through `:3247`.
- Section 9.1.10 gives vector-memory `SC[1:0]`/`NT` load, store, atomic,
  invalidate, and writeback cache-policy tables, including a `TG_SPLIT`
  special case, beginning at `cdna4/README.md:3274`.

XML evidence:

- `ENC_MUBUF` only says that the `LDS` bit writes data to LDS instead of a VGPR
  at `amdgpu_isa_cdna4.xml:3344` through `:3345`.
- `ENC_MUBUF` and `ENC_MTBUF` expose `SC0`, `SC1`, and `NT` fields with short
  descriptions at `amdgpu_isa_cdna4.xml:3353` through `:3405` and `:3549`
  through `:3601`.
- `BUFFER_WBL2` and `BUFFER_INV` entries are present at
  `amdgpu_isa_cdna4.xml:26161` through `:26190`, but the XML does not encode
  the LDS-capable opcode subset, `M0[17:0]` LDS offset, LDS clamping rule, full
  cache-policy tables, or `TG_SPLIT` refinement.

Impact:

Decoding the fields from XML is possible, but execution and validation need
manual-derived rules for LDS destinations and cache policy.

### CDNA4-XML-025: Buffer atomic `VDATA` input and conditional return are under-modeled

Manual evidence:

- Section 9.1 says buffer atomics take data from VGPRs and optionally return
  the pre-operation value at `cdna4/README.md:2964`.
- Section 9.1.3 says atomics read data from the VGPRs starting at `VDATA`; if
  the atomic returns a value, it is returned to the same `VDATA` registers at
  `cdna4/README.md:3029`.

XML evidence:

- Representative buffer atomic descriptions correctly refer to `SC0` for the
  conditional return, for example `BUFFER_ATOMIC_ADD` at
  `amdgpu_isa_cdna4.xml:26336` through `:26337`.
- The same entry marks `VDATA` as `Input="false"` and `Output="true"`
  unconditionally at `amdgpu_isa_cdna4.xml:26344` through `:26349`, even
  though `VDATA` supplies the atomic input and is written only for return forms.

Impact:

XML-driven dataflow can miss the atomic source dependency and can report a
VGPR write for no-return atomics. This is distinct from the older stale-`GLC`
wording: CDNA4 names `SC0`, but the operand contract remains conditional.

### CDNA4-XML-063: Buffer floating atomic numeric rules are prose-only

Manual evidence:

- Chapter 9.2 says floating memory atomics execute in LDS and L2 and can be
  issued as LDS, Buffer, Flat, Global, and Scratch instructions at
  `cdna4/README.md:3385` through `:3389`.
- The same chapter says all float atomic ADD opcodes use RNE rounding, defines
  MODE-based denormal controls, and gives L2-specific behavior: packed F16/BF16
  and F64 add/min/max do not flush denormals, while F32 add flushes denormals,
  at `cdna4/README.md:3393` through `:3428`.
- Chapter 9.2 also defines NaN, signed-zero, compare-swap, min/max ordering,
  and add edge cases for floating atomics at `cdna4/README.md:3430` through
  `:3475`.

XML evidence:

- CDNA4 XML contains the buffer floating atomic opcodes and formats, including
  `BUFFER_ATOMIC_ADD_F32`, `BUFFER_ATOMIC_PK_ADD_F16`,
  `BUFFER_ATOMIC_ADD_F64`, `BUFFER_ATOMIC_MIN_F64`,
  `BUFFER_ATOMIC_MAX_F64`, and `BUFFER_ATOMIC_PK_ADD_BF16` at
  `amdgpu_isa_cdna4.xml:27007` through `:27345`.
- Those records describe the high-level operation and conditional `SC0` return,
  plus operand data formats such as `FMT_NUM_F32`, `FMT_NUM_PK2_F16`,
  `FMT_NUM_F64`, and `FMT_NUM_PK2_BF16`, but do not encode RNE rounding,
  L2-specific denormal policy, packed lane behavior, or NaN/signed-zero
  selection rules.

Impact:

The XML identifies buffer floating atomics and their memory data formats, but a
generator cannot derive exact L2 floating-atomic semantics from XML alone.

### CDNA4-XML-064: VOP2 FP min/max exact selection rules are prose-only

Manual evidence:

- `V_MIN_F32` and `V_MAX_F32` define signaling-NaN quieting under
  `WAVE_MODE.IEEE`, NaN operand selection, signed-zero tie selection, and the
  `V_MAX_F32` IEEE versus non-IEEE equality predicate at `cdna4/README.md:7334`
  through `:7383`.
- `V_MAX_F16` and `V_MIN_F16` repeat the same instruction-local NaN and
  signed-zero rules for half precision, plus F16 round/exception/saturation
  notes, at `cdna4/README.md:7809` through `:7864`.

XML evidence:

- XML records `V_MIN_F32` and `V_MAX_F32` as ordinary F32 min/max instructions,
  with operand formats and VOP2/DPP/SDWA/VOP3 encodings, at
  `amdgpu_isa_cdna4.xml:57399` through `:57527` and `:57543` through `:57671`.
- XML records `V_MAX_F16` and `V_MIN_F16` with F16 operand formats and the same
  encoding-family metadata at `amdgpu_isa_cdna4.xml:62369` through `:62498` and
  `:62513` through `:62641`.
- These records do not encode signaling-NaN quieting, NaN operand preference,
  signed-zero tie rules, or `WAVE_MODE.IEEE`-dependent predicates.

Impact:

XML consumers can identify the opcode and operand shape, but cannot derive the
bit-exact FP min/max result selection rules that matter for NaNs, signed zeros,
and IEEE-mode equality cases.

### CDNA4-XML-065: `V_DOT2C_F32_BF16` VOP3 modifier remapping is prose-only

Manual evidence:

- The `V_DOT2C_F32_BF16` definition says `ABS[1:0]` are used as `NEG_HI[1:0]`
  during translation and that `NEG`/`ABS` input modifiers do not affect the
  accumulator source at `cdna4/README.md:7482` through `:7499`.

XML evidence:

- XML exposes the VOP2, literal, DPP, and VOP3 encodings for
  `V_DOT2C_F32_BF16`, including packed BF16 source formats and the VOP3
  `SRC0`/`SRC1` operands, at `amdgpu_isa_cdna4.xml:59127` through `:59229`.
- The instruction record does not state that VOP3 `ABS` bits are high-lane
  negation controls or that the normal modifier interpretation does not apply
  to the destination accumulator.

Impact:

A generator using only XML can treat DOT2C BF16 VOP3 modifiers as ordinary
source `ABS`/`NEG` bits, or ignore them entirely, instead of applying the
manual's packed-lane modifier remap.

### CDNA4-XML-066: F16 VOP2 destination-half behavior is prose-only

Manual evidence:

- `V_MAC_F16` documents a non-standard OPSEL rule: `OPSEL[3]` writes the
  computed F16 value into the high half while preserving the low half, otherwise
  the high half is zero and the result is written into the low half, at
  `cdna4/README.md:7689` through `:7707`.
- `V_MADMK_F16` and `V_MADAK_F16` explicitly compute an F16 result and write
  `D0 = { 16'0, tmp.f16 }`, while also forbidding VOP3 and input/output
  modifiers, at `cdna4/README.md:7709` through `:7733`.

XML evidence:

- XML's `V_MAC_F16` description mentions a "non-standard rule for OPSEL" and
  records F16 operands and VOP2/DPP/VOP3 encodings at
  `amdgpu_isa_cdna4.xml:61084` through `:61187`, but does not encode the exact
  half-preserve versus zero-high writeback rule.
- XML records `V_MADMK_F16` and `V_MADAK_F16` as literal-only F16 forms with
  16-bit operands at `amdgpu_isa_cdna4.xml:61203` through `:61352`, but does not
  encode the forced zeroing of the high half.

Impact:

The XML supplies the operand sizes and the absence of promoted encodings for the
literal forms, but downstream decoders or semantics generators still need manual
prose to distinguish true16 read-modify-write behavior from zero-high writes.

### CDNA4-XML-067: VOP2 no-modifier exclusions are prose-only

Manual evidence:

- `V_AND_B32`, `V_OR_B32`, `V_XOR_B32`, and `V_XNOR_B32` explicitly say input
  and output modifiers are not supported at `cdna4/README.md:7446` through
  `:7480` and `:8019` through `:8029`.
- `V_FMAMK_F32`, `V_FMAAK_F32`, `V_MADMK_F16`, and `V_MADAK_F16` say they
  cannot use VOP3 and cannot use input/output modifiers at
  `cdna4/README.md:7501` through `:7523` and `:7709` through `:7733`.

XML evidence:

- XML exposes VOP3 forms for the bitwise B32 opcodes, including generic VOP3
  source operands that share the architecture-wide modifier fields; for example
  `V_XNOR_B32` has VOP3 metadata at `amdgpu_isa_cdna4.xml:64523` through
  `:64658`.
- XML correctly omits VOP3/DPP/SDWA encodings for the four literal-only
  `_MK`/`_AK` forms, recording only `VOP2_INST_LITERAL` variants at
  `amdgpu_isa_cdna4.xml:59246` through `:59395` and `:61203` through `:61352`,
  but it does not separately encode the input/output-modifier prohibition.

Impact:

Encoding-family availability is mostly inferable from XML, but per-instruction
modifier legality is not explicit. Consumers that mechanically expose generic
VOP3 modifier fields need the manual exclusions to avoid accepting or applying
unsupported modifiers.

### CDNA4-XML-068: VOP1 XML contains opcode records absent from detailed manual tables

Manual evidence:

- Chapter 12.8 defines 85 detailed VOP1 rows from `V_NOP` through
  `V_CVT_F32_BF16`, with holes at opcodes 55, 75, and 76; the adjacent
  definitions skip from `V_CLREXCP 53` to `V_MOV_B64 56` at
  `cdna4/README.md:8878` through `:8882`, and from `V_COS_F16 74` to
  `V_CVT_NORM_I16_F16 77` at `:9178` through `:9201`.
- The Chapter 13 VOP1 opcode table also omits opcode 55, 75, and 76 at
  `cdna4/README.md:25267` through `:25318`.
- `V_EXP_LEGACY_F32` and `V_LOG_LEGACY_F32` appear only in the high-level
  transcendental-operation inventory at `cdna4/README.md:866`. A source search
  found no detailed manual hit for `V_SCREEN_PARTITION_4SE_B32`.

XML evidence:

- XML records `V_SCREEN_PARTITION_4SE_B32` as VOP1 opcode 55 with literal, DPP,
  SDWA, and VOP3 encodings at `amdgpu_isa_cdna4.xml:52139` through `:52237`.
- XML records `V_EXP_LEGACY_F32` and `V_LOG_LEGACY_F32` as VOP1 opcodes 75 and
  76, also with literal, DPP, SDWA, and VOP3 encodings, at
  `amdgpu_isa_cdna4.xml:54419` through `:54517` and `:54533` through `:54572`.

Impact:

The checked-in XML exposes three VOP1 instruction records that cannot be
validated from the detailed CDNA4 manual definition table. Generators need an
oracle outside the detailed manual rows to decide whether these are intended
CDNA4 opcodes, source omissions, or XML-only drift.

### CDNA4-XML-069: VOP1 promoted-opcode tables conflict between manual sections

Manual evidence:

- Chapter 12.8.1 says VOP1 instructions can be encoded as VOP3 with opcode
  `+0x140`, though the sentence uses the word `VOP2`, at
  `cdna4/README.md:9408` through `:9410`.
- Chapter 13's VOP3A opcode table instead prints the VOP1 promoted rows at
  `+0x180`, for example `V_NOP` 384, `V_MOV_B32` 385,
  `V_READFIRSTLANE_B32` 386, `V_SAT_PK_U8_I16` 463, and
  `V_CVT_F32_BF16` 475 at `cdna4/README.md:25609` through `:25703`.

XML evidence:

- XML consistently uses `VOP1 opcode + 0x140` for promoted VOP3 forms, such as
  `V_READFIRSTLANE_B32` opcode 2 to VOP3 opcode 322 at
  `amdgpu_isa_cdna4.xml:46858` through `:46898`, `V_EXP_LEGACY_F32` opcode 75
  to VOP3 opcode 395 at `:54419` through `:54517`, and
  `V_PERMLANE16_SWAP_B32` opcode 89 to VOP3 opcode 409 at `:55730` through
  `:55770`.

Impact:

Chapter 12.8.1 and XML agree on the promoted VOP3 opcode arithmetic, but the
Chapter 13 VOP3A table gives a conflicting offset. A decoder or disassembler
cross-checking manual tables without this override can reject the XML values or
emit the wrong VOP3 opcode numbers for VOP1 aliases.

### CDNA4-XML-070: `V_READFIRSTLANE_B32` EXEC override and no-modifier rule are prose-only

Manual evidence:

- The `V_READFIRSTLANE_B32` definition says `EXEC == 0` forces lane 0, otherwise
  the lowest active lane is selected, then `D0.b32 = VGPR[lane][SRC0.u32]` at
  `cdna4/README.md:8078` through `:8092`.
- The notes say the VGPR read overrides the EXEC mask and that input/output
  modifiers are not supported because the operation is untyped at
  `cdna4/README.md:8094` through `:8096`.

XML evidence:

- XML records the VOP1 and VOP3 encodings and the `OPR_SREG_NOVCC` destination
  plus `OPR_VGPR_OR_LDS` source at `amdgpu_isa_cdna4.xml:46858` through
  `:46898`, but does not encode the all-disabled EXEC lane-0 rule, the
  EXEC-mask override, or the no-modifier rule.

Impact:

XML consumers can recover the opcode and operand classes, but need manual prose
to emulate the all-disabled EXEC case and to avoid applying generic VOP3 or SDWA
modifier behavior to this untyped scalarizing instruction.

### CDNA4-XML-071: Untyped VOP1 swap operands are read-write but XML marks them output-only

Manual evidence:

- `V_SWAP_B32` reads both `D0` and `S0`, writes both operands, and says input and
  output modifiers are not supported at `cdna4/README.md:9250` through `:9262`.
- `V_PERMLANE16_SWAP_B32` and `V_PERMLANE32_SWAP_B32` read and write both
  `SRC0` and `VDST` VGPRs across row-pair swaps, and their notes say `ABS`,
  `NEG`, and `OMOD` should be zeroed at `cdna4/README.md:9358` through `:9394`.

XML evidence:

- XML marks `V_SWAP_B32` `SRC0` as `Input="false"` and `Output="true"` in both
  VOP1 and VOP3 encodings at `amdgpu_isa_cdna4.xml:54989` through `:55029`.
- XML marks the `SRC0` operand of both permlane swap instructions the same way
  at `amdgpu_isa_cdna4.xml:55730` through `:55828`.
- These XML records identify only the default VOP1 and VOP3 encodings for the
  swap/permlane family, but do not separately encode the modifier-zero rule.

Impact:

Dataflow built from XML can miss the read dependency on `SRC0` for swap-style
VOP1/VOP3 instructions, and modifier validators still need instruction-local
manual prose to reject or zero generic VOP3 modifier bits.

### CDNA4-XML-072: VOP1 lookup, pseudorandom, and exception edge semantics are prose-only

Manual evidence:

- `V_CVT_OFF_F32_I4` uses a 16-entry conversion-offset table indexed by
  `S0[3:0]` at `cdna4/README.md:8267` through `:8268`.
- `V_FRACT_F32` documents DX-style negative fractional behavior and clamps the
  result to `0x3f7fffff` at `cdna4/README.md:8400` through `:8414`.
- `V_RCP_IFLAG_F32` can raise only the integer divide-by-zero exception, not
  floating-point exceptions, at `cdna4/README.md:8529` through `:8540`.
- `V_PRNG_B32` gives the LFSR formula and period note at
  `cdna4/README.md:9345` through `:9356`.

XML evidence:

- XML records the affected instruction names and operand metadata, for example
  `V_CVT_OFF_F32_I4` at `amdgpu_isa_cdna4.xml:48017`,
  `V_FRACT_F32` at `:49309`, `V_RCP_IFLAG_F32` at `:50221`, and
  `V_PRNG_B32` at `:55616`.
- Those XML records do not encode the conversion-offset lookup table, DX
  fractional clamp, integer-exception-only behavior, PRNG feedback constant, or
  period special case.

Impact:

XML-driven semantics can identify these VOP1 opcodes but cannot derive their
bit-exact lookup, exception, clamp, or LFSR behavior without manual prose.

### CDNA4-XML-073: VOPC DPP availability and limitations are not represented

Manual evidence:

- Chapter 13's VOPC `SRC0` selector table includes `250 = DPP` at
  `cdna4/README.md:25338` through `:25364`.
- The DPP microcode-format description says a DPP second dword can follow
  VOP1, VOP2, or VOPC instructions in place of a literal constant at
  `cdna4/README.md:26198` through `:26204`.
- Chapter 12.16.1 then forbids DPP for `V_CMP_CLASS_F64`,
  `V_CMPX_CLASS_F64`, all F64 compare families, and all I64/U64 compare
  families at `cdna4/README.md:24581` through `:24614`.

XML evidence:

- XML records VOPC literal and SDWA extension encodings as `VOPC_INST_LITERAL`
  and `VOPC_VOP_SDWA_SDST_ENC` at `amdgpu_isa_cdna4.xml:7600` and `:7910`.
- A representative `V_CMP_CLASS_F32` record exposes `ENC_VOPC`,
  `VOPC_INST_LITERAL`, and `VOPC_VOP_SDWA_SDST_ENC` operands at
  `amdgpu_isa_cdna4.xml:78056` through `:78145`, but has no VOPC DPP
  encoding record.
- A source search found no `VOPC_VOP_DPP` encoding in
  `amdgpu_isa_cdna4.xml`; only VOP1 and VOP2 DPP encodings are named.

Impact:

The manual describes VOPC DPP as generally available with explicit 64-bit
compare exclusions, but XML-derived consumers cannot discover that VOPC DPP
surface or its instruction-family limitation matrix.

### CDNA4-XML-074: VOPC compare and class predicates are prose-only

Manual evidence:

- Chapter 12.9 defines OP16 floating compare predicates and OP8 integer compare
  predicates at `cdna4/README.md:9431` through `:9455` and `:9468` through
  `:9479`.
- The class-compare definitions list the 10 mask bits for signaling NaN, quiet
  NaN, infinities, normals, denormals, and signed zero, and give pseudocode for
  `V_CMP_CLASS_F32` at `cdna4/README.md:9502` through `:9539`. F64 and F16
  repeat the same class contract, with the F16 literal-format note at
  `:9670` through `:9755`.
- Negated FP compare definitions include NaN-sensitive behavior in the detailed
  instruction prose, for example `V_CMP_NGE_F32` through `V_CMP_NGT_F32` at
  `cdna4/README.md:9848` through `:9890`.

XML evidence:

- XML records the affected instruction names, operand formats, and encoding
  families. For example, `V_CMP_CLASS_F32` at
  `amdgpu_isa_cdna4.xml:78056` through `:78145` identifies the class-mask
  operand shape but not the mask-bit meanings or classification pseudocode.
- Ordinary compare entries such as `V_CMP_F_F32` around
  `amdgpu_isa_cdna4.xml:82827` through `:82925` identify opcodes and operand
  formats, but do not encode the OP16/OP8 truth tables, unordered/ordered NaN
  predicates, or negated-compare edge rules.

Impact:

XML consumers can enumerate VOPC opcodes and operand sizes, but need manual
prose to implement bit-exact compare results for NaNs, signed zeros, class masks,
and the predicate-family opcode offsets.

### CDNA4-XML-075: VOPC compact tables contain source drift

Manual evidence:

- The Chapter 12 compact floating-family table spells the F64 families as
  `V_CMPS_{COMPF}_F64` and `V_CMPSX_{COMPF}_F64` at
  `cdna4/README.md:9457` through `:9466`, while detailed definitions and
  Chapter 13 use `V_CMP_*_F64` and `V_CMPX_*_F64`.
- Chapter 12's integer-family table describes `V_CMPX_{COMPI}_I16`,
  `V_CMPX_{COMPI}_I32`, and `V_CMPX_{COMPI}_I64` as unsigned compares at
  `cdna4/README.md:9481` through `:9496`, even though the mnemonic and detailed
  definitions are signed.
- The markdown extraction also corrupts the unordered `U` predicate row in the
  compact compare tables at `cdna4/README.md:9448` and `:25351`; the detailed
  definitions contain the expected unordered NaN predicate.

XML evidence:

- XML uses the detailed-definition names and signed formats, for example
  `V_CMPX_LT_I32` has `FMT_NUM_I32` operands at
  `amdgpu_isa_cdna4.xml:96026` through `:96063`.
- XML records F64 compare names without the compact-table `S` spelling, for
  example `V_CMP_F_F64` and adjacent aliases beginning around
  `amdgpu_isa_cdna4.xml:86900`.

Impact:

The detailed manual rows and XML agree, but table-based manual scraping can
derive stale names or signedness from the compact VOPC summary tables unless it
cross-checks the detailed instruction definitions.

### CDNA4-XML-076: `V_DOT2_F32_BF16` source-2 modifier exception is prose-only

Manual evidence:

- `V_DOT2_F32_BF16` computes two BF16 products, adds `S2.f32`, and notes that
  `NEG` and `ABS` input modifiers do not affect `S2` at
  `cdna4/README.md:11890` through `:11903`.

XML evidence:

- The XML record correctly marks `SRC0` and `SRC1` as `FMT_NUM_PK2_BF16` and
  `SRC2` as `FMT_NUM_F32` at `amdgpu_isa_cdna4.xml:73582` through `:73613`.
- The same record carries only the broad instruction description and operand
  metadata; it does not encode the instruction-local exception that source
  modifiers skip `SRC2`.

Impact:

An XML-only VOP3P generator can apply ordinary source-modifier rules to the F32
accumulator source even though the manual says those modifiers are ignored for
`S2`.

### CDNA4-XML-077: Packed F16 min/max3 MODE behavior is prose-only

Manual evidence:

- The MODE table defines `DX10_CLAMP` as vector-ALU NaN-to-zero behavior and
  `IEEE` as signaling-NaN propagation/quieting state at `cdna4/README.md:491`
  through `:500`.
- `V_PK_MINIMUM3_F16` and `V_PK_MAXIMUM3_F16` say signaling NaNs propagate, then
  add that `DX10_CLAMP` forces NaNs to zero and `IEEE` is forced to 1 for the
  operation at `cdna4/README.md:11905` through `:11937`.

XML evidence:

- The XML instruction records for `V_PK_MINIMUM3_F16` and
  `V_PK_MAXIMUM3_F16` contain descriptions and packed-F16 operand metadata at
  `amdgpu_isa_cdna4.xml:73632` through `:73710`.
- Those records do not link the instructions to MODE bit 8, do not encode the
  NaN-to-zero behavior when `DX10_CLAMP` is set, and do not encode the forced
  `IEEE=1` override.

Impact:

XML consumers can see that these are IEEE minimum/maximum operations, but cannot
recover their instruction-specific MODE dependence and NaN override from XML
alone.

### CDNA4-XML-078: FMAC destination-accumulator reads are not encoded

Manual evidence:

- `V_FMAC_F32` multiplies two inputs and accumulates into the destination, with
  pseudocode `D0.f32 = fma(S0.f32, S1.f32, D0.f32)`, at
  `cdna4/README.md:15401` through `:15407`.
- `V_PK_FMAC_F16` does the same component-wise for packed half precision,
  reading both destination halves as accumulators at `cdna4/README.md:15409`
  through `:15415`.

XML evidence:

- XML records `V_FMAC_F32` and `V_PK_FMAC_F16` VOP2 and VOP3 encodings, but the
  `VDST` operand is marked `Input="false"` and `Output="true"` in those records
  at `amdgpu_isa_cdna4.xml:64285` through `:64506`.
- The XML descriptions mention accumulation into the destination, but the
  operand dataflow does not mark `VDST` as an input dependency for either scalar
  or packed FMAC form.

Impact:

Instruction consumers need a manual-derived override to treat `VDST` as both
source and destination for FMAC. A dataflow-only XML consumer can miss the
accumulator read, especially for generated VOP3A forms where the destination is
not otherwise listed as a source operand.

### CDNA4-XML-079: VOP3B scalar-result semantics are only partly described

Manual evidence:

- Chapter 12.11 and 13.3.5 say VOP3B is the unique-scalar-destination encoding
  used only by six add/sub carry opcodes, two div-scale opcodes, and two 64-bit
  MAD opcodes at `cdna4/README.md:13388` through `:13410` and `:25895` through
  `:25910`.
- The carry and borrow forms define exact unsigned carry predicates and note
  that VOP3 carry-in comes from the SGPR pair at `S2` at
  `cdna4/README.md:14978` through `:15076`.
- `V_DIV_SCALE_F32` and `V_DIV_SCALE_F64` give the exponent/denorm branch
  tables for the result and condition-code mask at `cdna4/README.md:15866`
  through `:15952`.
- `V_MAD_U64_U32` and `V_MAD_I64_I32` define 65-bit multiply-add results, with
  the high bit going to the scalar mask, at `cdna4/README.md:16054` through
  `:16078`.

XML evidence:

- XML provides the `VOP3_SDST_ENC` field layout at `amdgpu_isa_cdna4.xml:7262`
  through `:7396` and per-instruction operand records for the VOP3B family at
  `:59529` through `:60499`, `:66171` through `:66270`, and `:66593` through
  `:66692`.
- Those records preserve the opcode names, operand classes, and broad
  descriptions, but do not machine-encode the carry/borrow predicates,
  carry-in source mapping, div-scale branch conditions, or 65-bit MAD result
  split.

Impact:

XML consumers can decode VOP3B operands, but require manual-derived semantics to
emulate the scalar mask result or validate a semantic lowering for these
instructions.

### CDNA4-XML-080: VOP3B `SDST` field prose is compare-specific

Manual evidence:

- Chapter 12.11 says VOP3B allows specifying a unique scalar destination and is
  used only by the add/sub carry, div-scale, and 64-bit MAD opcodes at
  `cdna4/README.md:13388` through `:13410`.
- Chapter 13.3.5 repeats the same ten-opcode VOP3B list and defines `SDST` as
  `Scalar destination` at `cdna4/README.md:25895` through `:25920`.

XML evidence:

- The generic `VOP3_SDST_ENC` field map describes `SDST` as `Destination for
  compare result` at `amdgpu_isa_cdna4.xml:7345` through `:7346`.
- The per-instruction records make clear that VOP3B `SDST` is broader than a
  compare result: `V_ADD_CO_U32` stores a carry-out mask in `SDST` at
  `amdgpu_isa_cdna4.xml:59529` through `:59558`, `V_DIV_SCALE_F32` stores its
  post-scaling mask at `:66171` through `:66205`, and `V_MAD_U64_U32` stores
  overflow/carryout at `:66593` through `:66628`.

Impact:

Per-instruction operands are sufficient to recover the destination width and
class, but consumers that read only the generic VOP3B field description can
misclassify `SDST` as a compare-only destination instead of the scalar
destination used by the full VOP3B opcode family.

### CDNA4-XML-081: VOP3A F32 min/max3 MODE behavior is prose-only

Manual evidence:

- The MODE table defines `DX10_CLAMP` as vector-ALU NaN-to-zero behavior and
  `IEEE` as signaling-NaN propagation/quieting state at `cdna4/README.md:491`
  through `:500`.
- `V_MINIMUM3_F32` and `V_MAXIMUM3_F32` say signaling NaNs propagate, then add
  that `DX10_CLAMP` forces NaNs to zero and `IEEE` is forced to 1 for these
  operations at `cdna4/README.md:18004` through `:18030`.

XML evidence:

- XML records `V_MINIMUM3_F32` and `V_MAXIMUM3_F32` as `ENC_VOP3` opcodes 680
  and 681 with F32 operands at `amdgpu_isa_cdna4.xml:71008` through `:71099`.
- Those records describe only the IEEE minimum/maximum operation. They do not
  link the opcodes to MODE bit 8, the NaN-to-zero `DX10_CLAMP` behavior, or the
  forced `IEEE=1` override.

Impact:

XML consumers can recover the VOP3A opcode and operand shape, but cannot infer
the instruction-specific MODE dependence for these scalar F32 min/max3
operations without manual-derived metadata.

### CDNA4-XML-082: `V_CVT_PKACCUM_U8_F32` destination-as-source dataflow is not encoded

Manual evidence:

- `V_CVT_PKACCUM_U8_F32` writes one selected byte of the destination register
  and says this opcode uses `src_c` to pass the destination in as a source at
  `cdna4/README.md:16236` through `:16246`.

XML evidence:

- XML describes the selected-byte destination update for VOP3A opcode 496, but
  the operand list has only output-only `VDST`, `SRC0`, and `SRC1`; there is no
  `SRC2`/`src_c` operand and `VDST` is not marked as an input at
  `amdgpu_isa_cdna4.xml:67001` through `:67033`.

Impact:

XML consumers can see that a destination byte is updated, but cannot recover the
old-destination dependency that preserves the other three bytes.

### CDNA4-XML-083: Native F16 VOP3A destination-half preservation is prose-only

Manual evidence:

- Native F16 VOP3A rows `V_MAD_F16`, `V_MAD_U16`, `V_MAD_I16`, `V_FMA_F16`,
  and `V_DIV_FIXUP_F16` preserve the opposite destination half for both
  `OPSEL[3]` choices at `cdna4/README.md:16422` through `:16522`.
- Legacy rows `V_MAD_LEGACY_U16`, `V_MAD_LEGACY_I16`, `V_FMA_LEGACY_F16`, and
  `V_DIV_FIXUP_LEGACY_F16` preserve the low half when `OPSEL[3]=1`, but zero
  the high half when `OPSEL[3]=0`, at `cdna4/README.md:16104` through
  `:16233`.

XML evidence:

- The generic `OP_SEL` field prose says destination `OP_SEL[3]=0` writes the
  low half and zeroes the high half, while `OP_SEL[3]=1` writes the high half
  and preserves the low half, at `amdgpu_isa_cdna4.xml:2912` through `:2913`.
- Native F16 VOP3A operand records mark `VDST` output-only for this family at
  `amdgpu_isa_cdna4.xml:67945` through `:68175`; legacy operand records do the
  same at `:66751` through `:67000`.

Impact:

The generic XML field text contradicts the native F16 preserve-high behavior
for `OPSEL[3]=0`, and the operand metadata does not expose the old-destination
read needed for either native preserve-half writes or legacy high-half writes.
`CDNA4-XML-066` covers the related VOP2 F16 destination-half rules.

### CDNA4-XML-084: `V_BITOP3_B16/B32` truth-table field overload is prose-only

Manual evidence:

- `V_BITOP3_B16` and `V_BITOP3_B32` build an 8-bit truth table from
  `{ OMOD[1:0], ABS[2:0], NEG[2:0] }` and say normal output modifier,
  absolute-value, and negation controls are disabled at
  `cdna4/README.md:16538` through `:16610`.

XML evidence:

- Generic VOP3A XML field descriptions still describe `ABS`, `NEG`, and `OMOD`
  as ordinary source/output modifiers at `amdgpu_isa_cdna4.xml:2852` through
  `:2893`, while the per-instruction records for these opcodes list only the
  ordinary operands at `:68245` through `:68344`.

Impact:

XML-only consumers can apply floating modifier semantics or reject integer
truth-table encodings instead of treating those fields as the `BITOP3` SIMM8
truth table.

### CDNA4-XML-085: `V_LSHL_ADD_U64` shift-count restriction is prose-only

Manual evidence:

- `V_LSHL_ADD_U64` says the shift count must be between 0 and 4, higher shift
  counts are unsupported, and the design treats unsupported counts as a shift
  of zero at `cdna4/README.md:16524` through `:16536`.

XML evidence:

- XML records `SRC1` as an ordinary 32-bit source for opcode 520 with no range
  or unsupported-count behavior at `amdgpu_isa_cdna4.xml:68195` through
  `:68227`.

Impact:

A decoder or executor derived only from XML will naturally implement an ordinary
variable 64-bit shift instead of the CDNA4 instruction-specific 0-through-4
count rule.

### CDNA4-XML-086: F64 min/max MODE-sensitive selection rules are prose-only

Manual evidence:

- `V_MIN_F64` and `V_MAX_F64` branch on `WAVE_MODE.IEEE`, signaling NaNs, quiet
  NaNs, signed-zero ties, and `V_MAX_F64` equality behavior at
  `cdna4/README.md:17430` through `:17478`; the MODE `IEEE` bit is defined at
  `cdna4/README.md:491` through `:500`.

XML evidence:

- XML records ordinary F64 operand shapes and descriptions for opcodes 642 and
  643 at `amdgpu_isa_cdna4.xml:69424` through `:69504`, but no exact NaN,
  signed-zero, or MODE-dependent selection contract.

Impact:

XML consumers can enumerate the instructions and operands, but need manual
semantics to match edge cases around NaNs, signed zero, and IEEE-mode equality.

### CDNA4-XML-087: Native readlane/writelane EXEC override, lane masks, and no-modifier rules are prose-only

Manual evidence:

- `V_READLANE_B32` says the VGPR read overrides `EXEC`, uses
  `S1.u32[5:0]` as the lane selector, and does not support input/output
  modifiers at `cdna4/README.md:17540` through `:17552`.
- `V_WRITELANE_B32` says the VGPR write overrides `EXEC`, uses
  `S1.u32[5:0]` as the lane selector, and does not support input/output
  modifiers at `cdna4/README.md:17554` through `:17567`.

XML evidence:

- XML captures the scalar/VGPR/lane-select operand classes for opcodes 649 and
  650 at `amdgpu_isa_cdna4.xml:69732` through `:69802`, but
  `OPR_SSRC_LANESEL` does not encode the `[5:0]` truncation, EXEC-mask
  override, or modifier ban.

Impact:

Generic VOP3A consumers need manual overrides for these native untyped lane
transfer opcodes, similar to the promoted `V_READFIRSTLANE_B32` exception
tracked in `CDNA4-XML-070`.

### CDNA4-XML-088: F64 VOP1 DPP exclusions leak into XML encodings

Manual evidence:

- Chapter 12.16.1 says the listed instructions cannot use DPP; the list includes
  the F64 VOP1 conversion and unary operations from `V_CVT_I32_F64` through
  `V_FRACT_F64`, plus `V_CLREXCP` and `V_SWAP_B32`, at
  `cdna4/README.md:24581` through `:24605`.
- Chapter 12.16.2 separately lists the instructions that cannot use SDWA at
  `cdna4/README.md:24615` through `:24629`.

XML evidence:

- XML still advertises a `VOP1_VOP_DPP` encoding for representative forbidden
  F64 VOP1 rows such as `V_CVT_I32_F64` at
  `amdgpu_isa_cdna4.xml:46915` through `:46994` and `V_CVT_F32_F64` at
  `:48131` through `:48190`.
- The same DPP mismatch appears for `V_CVT_F64_I32` (`:47010`/`:47052`),
  `V_CVT_U32_F64` (`:48758`/`:48800`), `V_CVT_F64_U32`
  (`:48853`/`:48895`), `V_TRUNC_F64` (`:48948`/`:48990`),
  `V_CEIL_F64` (`:49043`/`:49085`), `V_FLOOR_F64`
  (`:49214`/`:49256`), `V_FREXP_EXP_I32_F64` (`:51589`/`:51631`),
  `V_FREXP_MANT_F64` (`:51684`/`:51726`), and `V_FRACT_F64`
  (`:51779`/`:51821`).
- A static inventory found no analogous XML SDWA leak for the Chapter 12.16.2
  list, and XML correctly omits DPP records for `V_CLREXCP` and `V_SWAP_B32`;
  VOPC DPP availability and VOPC-specific exclusions remain the separate
  `CDNA4-XML-073` gap.

Impact:

XML-derived decoders or assemblers can treat DPP encodings as legal for a subset
of F64 VOP1 instructions that the CDNA4 manual explicitly excludes.

### CDNA4-XML-089: Program organization, dispatch, and device-memory consistency are prose-only

Manual evidence:

- Section 1.1 defines dispatch, workgroup, 64-work-item wavefronts, work-items,
  literal constants, SALU, VALU, and microcode-format terminology at
  `cdna4/README.md:324` through `:340`.
- Chapter 2 says kernels are grouped into 64-work-item wavefronts, all control
  flow is handled by SALU instructions, vector ALU and vector-memory work is
  gated by `EXEC`, and vector compare/carry-out results return bit-per-work-item
  masks to SGPRs at `cdna4/README.md:344` through `:362`.
- Section 2.1 says dispatches cover 1D/2D/3D grids, generate wavefronts, and
  initialize each work-item with a unique grid index at `cdna4/README.md:364`
  through `:366`.
- Sections 2.2.1 and 2.3 describe the 160 KiB / 64-bank / 32-atomic-unit LDS
  topology, L2/L1 hierarchy, cache-less loads, load-clause overlap behavior,
  atomic return acknowledgments, relaxed consistency, and per-PE/per-channel
  scatter-write ordering at `cdna4/README.md:378` through `:393`.

XML evidence:

- The top-level XML architecture record only carries the architecture name and
  numeric ID at `amdgpu_isa_cdna4.xml:1` through `:13`.
- XML has isolated instruction and field descriptions, such as VOPC compare
  `EXEC` wording at `amdgpu_isa_cdna4.xml:1614` and LDS operand/field text at
  `:3123` through `:3125` and `:3343` through `:3345`, but no global records
  for wavefront size, dispatch-grid dimensionality, SALU/VALU roles,
  EXEC-mask scope, LDS bank/atomic topology, or the Section 2.3 memory
  consistency/acknowledgment/order model.
- Existing narrower entries cover several downstream pieces:
  `CDNA4-XML-032` for LDS geometry/allocation/clamping, `CDNA4-XML-038` for
  wave-state helper semantics, `CDNA4-XML-041` for launch initialization,
  `CDNA4-XML-043` for barrier behavior, and `CDNA4-XML-044` for wait-counter
  semantics.

Impact:

XML consumers can enumerate instructions and some operands, but need manual
prose for the processor-organization assumptions and top-level memory model
that determine how decoded instructions execute together.

### CDNA4-XML-090: SDWAB `SD` field description loses the VCC destination case

Manual evidence:

- Chapter 13.3.8 describes SDWAB as the second word for VOPC SDWA forms with a
  scalar destination and defines `SD` as the selector between VCC and a normal
  SGPR at `cdna4/README.md:26170` through `:26186`.

XML evidence:

- `VOPC_VOP_SDWA_SDST_ENC` carries the `SD` field at bit 47, but describes
  `SD=0` as using bits 8:15 as SDWA destination control at
  `amdgpu_isa_cdna4.xml:6536` and `:6740` through `:6746`; `SDST` is still
  listed separately at `:6750` through `:6755`.
- `VOP2_VOP_SDWA_SDST_ENC` has the same generic `SD` description at
  `amdgpu_isa_cdna4.xml:5706` and `:5783` through `:5785`.

Impact:

XML consumers can recover the raw `SD` bit and `SDST` bits, but cannot derive
the manual's VCC-versus-normal-SGPR destination selection from the XML field
prose. A generated disassembler or validator that trusts that prose can
mishandle VOPC SDWA forms with `SD=0`.

### CDNA4-XML-091: `V_BCNT_U32_B32` base-add semantics are missing from XML description

Manual evidence:

- `V_BCNT_U32_B32` initializes the result with `S1.u32` and then adds the count
  of set bits in `S0` at `cdna4/README.md:17569` through `:17580`.

XML evidence:

- XML describes opcode 651 as counting bits in one vector input at
  `amdgpu_isa_cdna4.xml:69820` through `:69821`.
- The operand metadata still records `SRC1` as an input at
  `amdgpu_isa_cdna4.xml:69840` through `:69844`, but the description does not
  say that `SRC1` is an accumulator/base addend.

Impact:

An XML-driven semantic consumer can model this opcode as unary popcount and
discard the instruction's second source contribution.

### CDNA4-XML-092: `V_TRIG_PREOP_F64` range-reduction algorithm is prose-only

Manual evidence:

- `V_TRIG_PREOP_F64` defines the 2/pi lookup, `S1[4:0]` segment selection,
  exponent thresholds, large-input scaling, and round-toward-zero behavior at
  `cdna4/README.md:17668` through `:17694`.

XML evidence:

- XML carries only a short instruction description and the `VDST F64`,
  `SRC0 F64`, and `SRC1 B32` operand shape at
  `amdgpu_isa_cdna4.xml:70084` through `:70108`.

Impact:

XML consumers need the manual prose or a hand-coded table to implement the
range-reduction behavior for this opcode.

### CDNA4-XML-026: FLAT/GLOBAL/SCRATCH addressing and aperture rules are prose-only

Manual evidence:

- Chapter 10 says FLAT addresses cover video/system memory, LDS, and scratch,
  and that aperture registers determine routing while unmapped regions generate
  memory-violation errors at `cdna4/README.md:3477` through `:3479`.
- Section 10.1 defines `SEG`, `SVE`, `OFFSET`, `SADDR`, and implied `M0`
  behavior for FLAT/GLOBAL/SCRATCH at `cdna4/README.md:3483` through `:3499`,
  with scratch `SVE` address modes at `:3561` through `:3568`.
- Section 10.3 defines 32-bit versus 64-bit `PTR32` mode, flat scratch address
  conversion, and LDS-return address formulas at `cdna4/README.md:3586`
  through `:3608`; sections 10.4 through 10.7 define global and scratch
  address formulas, `ADDRESS_MODE`, LDS-access restrictions, scratch swizzle,
  automatic `FLAT_SCRATCH`, and data movement at `:3619` through `:3663`.
- Chapter 13.6 gives the FLAT field map and repeats the conditional role of
  `ADDR`/`SADDR` for FLAT/GLOBAL/SCRATCH at `cdna4/README.md:26488` through
  `:26499`.

XML evidence:

- `ENC_FLAT`, `ENC_FLAT_GLBL`, and `ENC_FLAT_SCRATCH` expose field bit
  positions, segment constants, offset widths, `SADDR`, `SC0`, `SC1`, `NT`,
  and `SVE`/bit 13 metadata at `amdgpu_isa_cdna4.xml:3730` through `:3889`,
  `:3890` through `:4093`, and `:4094` through `:4263`.
- Representative flat/global/scratch entries expose load/store/atomic
  operands, for example `FLAT_LOAD_UBYTE` at
  `amdgpu_isa_cdna4.xml:14590` through `:14625`.
- The XML does not encode `PTR32`, `ADDRESS_MODE`, aperture base/limit routing,
  memory-violation policy, private-aperture scratch conversion, scratch swizzle
  formulas, or the automatic `FLAT_SCRATCH` request state.

Impact:

The XML is adequate for raw field decode, but not for address-space routing,
address-size selection, fault behavior, or scratch-private address formation.

### CDNA4-XML-027: Bit 13 `SVE`/`LDS` and direct-to-LDS metadata are under-described

Manual evidence:

- The Section 10.1 field table names bit 13 `SVE` and describes scratch VGPR
  enablement at `cdna4/README.md:3495`, with the scratch `SVE` address-mode
  table at `:3561` through `:3568`.
- The same table says `M0` is used for SCRATCH and GLOBAL only when `LDS=1` at
  `cdna4/README.md:3499`, and Section 10.3 gives the LDS destination formulas
  using `M0[17:2]`, instruction offset, and thread ID at `:3604` through
  `:3608`.
- Sections 10.4 and 10.5 say GLOBAL and SCRATCH allow direct data movement
  between LDS and memory at `cdna4/README.md:3630` and `:3649`.
- Chapter 13.6 names bit 13 `LDS`, says it transfers data between LDS and
  memory instead of VGPRs and memory, and lists `GLOBAL_LOAD_LDS_*` and
  `SCRATCH_LOAD_LDS_*` opcodes at `cdna4/README.md:26488`,
  `:26566` through `:26578`, and `:26595` through `:26600`.

XML evidence:

- `ENC_FLAT`, `ENC_FLAT_GLBL`, and `ENC_FLAT_SCRATCH` all name bit 13 `SVE`
  with a scratch-VGPR-enable description at `amdgpu_isa_cdna4.xml:3867`
  through `:3876`, `:4071` through `:4080`, and `:4241` through `:4250`.
- The XML includes direct-to-LDS opcodes, for example
  `GLOBAL_LOAD_LDS_UBYTE` and `SCRATCH_LOAD_LDS_UBYTE` at
  `amdgpu_isa_cdna4.xml:19024` through `:19055` and `:22777` through
  `:22813`.
- Those LDS entries still expose `VDST` as an ordinary
  `OPR_VGPR_OR_ACCVGPR` output, and the XML does not encode the LDS address
  formulas, the opcode-specific meaning of bit 13, or the rule that direct-LDS
  movement bypasses VGPR data return.

Impact:

An XML-only generator can preserve the bit, but cannot decide whether to expose
it as scratch address enablement or direct-LDS movement, and can incorrectly
model LDS load forms as writing VGPR destinations.

### CDNA4-XML-028: FLAT wait-counter, ordering, and timing rules are prose-only

Manual evidence:

- Section 10.2 says FLAT instructions are internally executed as both LDS and
  Buffer operations, increment both `VM_CNT` and `LGKM_CNT`, and complete only
  after both counters decrement at `cdna4/README.md:3570` through `:3576`.
- Sections 10.2.1 and 10.2.2 say FLAT instructions can complete out of order,
  that same-VGPR returns have unknown results, and that only `S_WAITCNT 0` is
  sensible after FLAT loads because the VM and LGKM paths can race at
  `cdna4/README.md:3578` through `:3584`.
- Sections 10.4 and 10.5 say GLOBAL and SCRATCH use only `VM_CNT`, not
  `LGKM_CNT`, and define the LDS-access restrictions for those segments at
  `cdna4/README.md:3630` through `:3651`.

XML evidence:

- The XML exposes flat/global/scratch instruction entries and `SC0`/`SC1`/`NT`
  fields, but no instruction metadata records dual counter increments,
  counter-retirement conditions, ordering, same-destination hazards, or the
  `S_WAITCNT 0` restriction.

Impact:

The XML cannot drive wait-counter or hazard modeling for FLAT instructions
without manual-derived timing rules.

### CDNA4-XML-029: FLAT/GLOBAL floating atomic numeric rules are missing

Manual evidence:

- Section 10.3.1 says floating-point atomics must set `SC[0]=0`, that FP32
  atomics flush denormals to zero, FP64 and FP16 atomics do not flush denormals,
  and rounding is fixed round-to-nearest-even at `cdna4/README.md:3611`
  through `:3617`.
- The opcode inventory includes F32, packed F16/BF16, and F64 flat/global
  floating atomics at `cdna4/README.md:3546` through `:3551` and
  `:26513` through `:26523`, with matching GLOBAL entries at `:26553`
  through `:26558`.

XML evidence:

- Representative CDNA4 XML atomic descriptions correctly refer to `SC0` for
  conditional return, for example `FLAT_ATOMIC_SWAP` at
  `amdgpu_isa_cdna4.xml:15756` through `:15757`.
- Representative floating atomic entries expose F32 and packed F16 data
  formats, for example `FLAT_ATOMIC_ADD_F32` and
  `FLAT_ATOMIC_PK_ADD_F16` at `amdgpu_isa_cdna4.xml:16601` through
  `:16720`.
- The XML does not encode the floating-atomic `SC0=0` legality requirement,
  FP32 denormal flushing, FP16/FP64 denormal preservation, or fixed RNE
  rounding behavior.

Impact:

XML consumers can see that floating atomics exist, but not the numeric and
return-mode restrictions needed for correct execution or validation.

### CDNA4-XML-030: Flat-memory `NV` is a manual-only ambiguity

Manual evidence:

- The Section 10.1 field table lists an `NV` field and says it marks
  non-volatile memory at `cdna4/README.md:3495` through `:3497`.
- Chapter 13.6's FLAT field map has no `NV` bit between `NT`, `OP`, `SC1`, and
  `ENCODING` at `cdna4/README.md:26488` through `:26499`.

XML evidence:

- `ENC_FLAT`, `ENC_FLAT_GLBL`, and `ENC_FLAT_SCRATCH` expose `NT`, `OP`,
  `SC1`, and the encoding bits with no FLAT-family `NV` field at
  `amdgpu_isa_cdna4.xml:3797` through `:3825`, `:4001` through `:4029`,
  and `:4171` through `:4199`.

Impact:

The flat-memory prose mentions non-volatile behavior, but neither the
microcode-format table nor XML exposes a FLAT `NV` bit. This should be treated
as a manual/XML ambiguity until resolved against hardware or an authoritative
encoding source.

### CDNA4-XML-031: FLAT `ACC` field descriptions disagree about the selected operand

Manual evidence:

- The Section 10.1 field table says `ACC` means `DATA` is an Accumulation VGPR
  at `cdna4/README.md:3493`.
- Chapter 13.6 says `ACC` means `VDATA` is an Accumulation VGPR at
  `cdna4/README.md:26498`, while the same table uses `DATA` and `VDST` as the
  actual encoded field names at `:26495` through `:26499`.

XML evidence:

- `ENC_FLAT`, `ENC_FLAT_GLBL`, and `ENC_FLAT_SCRATCH` describe `ACC` as
  selecting whether `VDST` uses an accumulator VGPR at
  `amdgpu_isa_cdna4.xml:3757` through `:3764`, `:3961` through `:3968`, and
  `:4131` through `:4138`.
- Per-instruction operands are more specific: a representative load exposes
  `VDST` as `OPR_VGPR_OR_ACCVGPR` at `amdgpu_isa_cdna4.xml:14598` through
  `:14608`, while a representative store exposes `DATA` as
  `OPR_VGPR_OR_ACCVGPR` at `:15022` through `:15032`, and atomics expose both
  `VDST` and `DATA` as accumulator-capable at `:15764` through `:15780`.

Impact:

The XML operand records can drive most decode/display behavior, but the generic
field description conflicts with the manual and can mislead consumers that
derive the `ACC` contract from field metadata instead of per-instruction
operands.

### CDNA4-XML-032: LDS geometry, allocation, and M0 clamp rules are prose-only

Manual evidence:

- Section 11.1 describes CDNA4 LDS as 160 KiB per CU, split into 64 banks of 640
  dwords, and says bank conflicts serialize indexed and atomic operations at
  `cdna4/README.md:3679` through `:3687`.
- Section 11.3.1 says all LDS operations require `M0` initialization and uses
  `M0[16:0]` as the LDS segment byte-size for address clamping at
  `cdna4/README.md:3728` through `:3730`.
- Section 3.6.1 and 3.6.5 define LDS out-of-range behavior and allocation:
  out-of-range reads return zero, writes are discarded, allocation is in
  1280-byte aligned blocks, and the clamp uses the smaller of SPI allocation
  size and `M0` at `cdna4/README.md:538` through `:548` and `:577` through
  `:581`.

XML evidence:

- `ENC_DS` records the raw DS fields, including `GDS`, `OFFSET0`, `OFFSET1`,
  `OP`, `ADDR`, `DATA0`, `DATA1`, and `VDST`, at
  `amdgpu_isa_cdna4.xml:3110` through `:3205`.
- The XML field metadata does not encode LDS bank count/width, bank-conflict
  serialization, required `M0` initialization, `M0[16:0]` clamp meaning,
  allocation granularity, or out-of-range LDS read/write behavior.

Impact:

XML consumers can decode DS bitfields, but cannot derive CDNA4 LDS bounds,
allocation, or timing-relevant bank behavior without manual prose.

### CDNA4-XML-033: DS address formulas, ADDTID, and duplicate-offset behavior are prose-only

Manual evidence:

- Section 11.3.1 defines single-address DS calculation as
  `LDS_BASE + VGPR[ADDR] + {InstrOffset1,InstrOffset0}` at
  `cdna4/README.md:3747` through `:3749`.
- The same section defines two-address `READ2`/`WRITE2` formulas using
  `InstrOffset0 * ADJ` and `InstrOffset1 * ADJ`, with `ADJ` selected by data
  width, and says equal offsets specify one access using only `DATA0` at
  `cdna4/README.md:3751` through `:3767`.
- `DS_WRITE_ADDTID_B32` and `DS_READ_ADDTID_B32` instruction definitions use
  `{OFFSET1,OFFSET0} + M0[15:0] + laneID * 4` at
  `cdna4/README.md:20450` through `:20455` and `:21799` through `:21807`.

XML evidence:

- `ENC_DS` says `OFFSET0` and `OFFSET1` are DS offsets whose use depends on
  individual instructions at `amdgpu_isa_cdna4.xml:3173` through `:3190`.
- The XML contains `DS_WRITE2_B32`, `DS_WRITE2ST64_B32`, `DS_WRXCHG2_RTN_B32`,
  and `DS_READ2_B32` entries around `amdgpu_isa_cdna4.xml:8969`, `:9018`,
  `:10351`, and `:10864`, but the entries do not encode the general formulas,
  width-dependent `ADJ`, or duplicate-offset single-access rule.
- `DS_READ_ADDTID_B32` exposes an implicit `M0` operand at
  `amdgpu_isa_cdna4.xml:14005` through `:14028`, but the XML does not encode the
  `M0[15:0] + laneID * 4` address formula. The `DS_WRITE_ADDTID_B32` entry
  at `amdgpu_isa_cdna4.xml:9447` through `:9474` likewise lacks executable
  formula metadata.

Impact:

Generated code needs manual-derived DS address semantics. XML-only consumers can
see the fields but cannot determine when offsets are concatenated, scaled,
collapsed, or interpreted through ADDTID's `M0[15:0]` formula.

### CDNA4-XML-034: DS swizzle and permute lane semantics are under-described

Manual evidence:

- `DS_SWIZZLE_B32` has FFT, rotate, quad, and 32-lane bit-mask modes, invalid
  source-lane zeroing, and complete lane-mapping pseudocode at
  `cdna4/README.md:20852` through `:20973`.
- `DS_PERMUTE_B32` and `DS_BPERMUTE_B32` do not access LDS memory but have
  precise source/destination lane formulas, `EXEC` handling, high-bit ignoring,
  and collision behavior at `cdna4/README.md:20975` through `:21014` and
  `:21031` through `:21083`.

XML evidence:

- `DS_SWIZZLE_B32` only says that no data is written to LDS memory at
  `amdgpu_isa_cdna4.xml:11122` through `:11150`.
- `DS_PERMUTE_B32` and `DS_BPERMUTE_B32` have short descriptions and operands at
  `amdgpu_isa_cdna4.xml:11161` through `:11230`, but no lane formulas,
  collision rule, invalid-thread behavior, or complete `EXEC` semantics.

Impact:

The XML identifies the swizzle/permute opcodes and operands, but cannot drive
correct lane-level execution or validation for these non-LDS-memory DS
instructions.

### CDNA4-XML-035: LDS floating atomic numeric rules are prose-only

Manual evidence:

- Chapter 9.2 says floating memory atomics execute in LDS and L2 and defines
  RNE rounding, MODE-dependent denormal handling, packed F16/BF16 handling, and
  NaN behavior at `cdna4/README.md:3387` through `:3443`.
- Section 11.3.1 reiterates that LDS floating atomic denormal behavior is
  controlled by `MODE.FP_DENORM` and rounding is fixed RNE at
  `cdna4/README.md:3769` through `:3783`.

XML evidence:

- CDNA4 XML contains floating DS atomic entries, including packed F16/BF16 forms.
  For example, `DS_PK_ADD_F16` and `DS_PK_ADD_BF16` expose packed DSMEM side
  effects at `amdgpu_isa_cdna4.xml:9349` through `:9425`.
- Those entries carry operand/data-format metadata but do not encode
  MODE-dependent denormal policy, fixed RNE rounding, or NaN-selection behavior.

Impact:

The XML identifies floating LDS atomic opcodes and side effects, but cannot
drive numerically exact atomic execution without manual-derived floating-point
state rules.

### CDNA4-XML-036: LDS transpose-load constraints and B6 naming are under-specified

Manual evidence:

- Section 11.4 says MFMA transpose loads require `EXEC` all ones, LDS address
  alignment to data size, even-aligned VGPRs for 64-bit-or-larger data except
  `DS_READ_B96_TR_B6`, and two-instruction matrix-load layouts at
  `cdna4/README.md:3785` through `:3799`.
- The DS opcode table lists opcode 225 as `DS_READ_B96_TR_B6` at
  `cdna4/README.md:26336` through `:26339`.

XML evidence:

- Transpose load XML entries expose operand sizes and descriptions at
  `amdgpu_isa_cdna4.xml:14329` through `:14486`.
- `DS_READ_B96_TR_B6` additionally carries `DS_READ_B128_TR_B6` as an alias at
  `amdgpu_isa_cdna4.xml:14372` through `:14375`, but the audited manual only
  lists the B96 form.
- The XML entries do not encode the all-ones `EXEC` requirement, alignment
  checks, even-VGPR rule and B96 exception, or the two-instruction matrix layout
  rules.

Impact:

XML-only decode can name the transpose opcodes, but cannot validate the manual's
execution preconditions or unambiguously explain the B6/B96 aliasing.

### CDNA4-XML-037: CDNA4 DS `GDS` classification is inconsistent

Manual evidence:

- The Chapter 11 DS field table says `GDS` is `0 = LDS, 1 = Reserved` at
  `cdna4/README.md:3715` through `:3719`.
- `DS_CONSUME` and `DS_APPEND` instruction descriptions still mention GDS
  addressing at `cdna4/README.md:21846` through `:21852`.
- A static search of the audited CDNA4 manual found no `DS_GWS_*` instruction
  definitions or opcode-table entries.

XML evidence:

- `ENC_DS` describes the `GDS` bit as `1=GDS, 0=LDS` at
  `amdgpu_isa_cdna4.xml:3163` through `:3170`.
- The XML also repeats GDS wording in `DS_CONSUME` and `DS_APPEND` descriptions
  at `amdgpu_isa_cdna4.xml:14157` through `:14223`, while a static search found
  no CDNA4 `DS_GWS_*` instruction entries.

Impact:

CDNA4 consumers cannot decide from XML alone whether `GDS=1` DS encodings should
be treated as reserved, executable GDS forms, or append/consume-only exceptions.

### CDNA4-XML-038: Wave-state register fields and helper semantics are prose-only

Manual evidence:

- Chapter 3.1 lists visible wave state, including `EXEC`, `EXECZ`, `VCC`,
  `VCCZ`, `SCC`, `STATUS`, `MODE`, `M0`, `HW_ID`, `XCC_ID`, `TRAPSTS`, trap
  temporaries, and wait counters at `cdna4/README.md:401` through `:437`.
- Sections 3.2 through 3.5 define PC-relative behavior, `EXEC` masking and
  `EXECZ`, the complete `STATUS` field table, and the complete `MODE` field
  table at `cdna4/README.md:439` through `:512`.

XML evidence:

- `S_GETREG_B32`, `S_SETREG_B32`, and `S_SETREG_IMM32_B32` expose `OPR_HWREG`
  operands at `amdgpu_isa_cdna4.xml:45510` through `:45605`.
- `OPR_HWREG` names hardware-register IDs, including `HW_REG_MODE`,
  `HW_REG_STATUS`, `HW_REG_TRAPSTS`, `HW_REG_HW_ID`, `HW_REG_GPR_ALLOC`,
  `HW_REG_LDS_ALLOC`, and `HW_REG_XCC_ID`, at
  `amdgpu_isa_cdna4.xml:116837` through `:116970`, but the register and ID
  descriptions are `N/A`.
- The XML does not encode the `STATUS` or `MODE` bitfield tables, which fields
  are read-only versus writable, which helper bits summarize `EXEC`/`VCC`, or
  how PC-relative state interacts with branches and `S_TRAP`.

Impact:

XML consumers can parse HWREG operands and ID numbers, but cannot recover the
Chapter 3 state-register semantics, access policy, or helper-bit relationships
without manual prose.

### CDNA4-XML-039: GPR allocation, aliasing, alignment, and out-of-range behavior are prose-only

Manual evidence:

- Section 3.6.1 defines SGPR, VGPR, LDS, memory-return, atomic-return, and
  multiple-destination out-of-range behavior at `cdna4/README.md:518` through
  `:550`.
- Sections 3.6.2 through 3.6.5 define SGPR allocation granularity, physical VCC
  aliasing, privileged trap SGPR reservation, SGPR/VGPR alignment rules, VGPR
  and AccVGPR pools, and LDS allocation/clamping at `cdna4/README.md:552`
  through `:581`.

XML evidence:

- Generic VGPR and SGPR operand definitions carry broad even-alignment text for
  64-bit and wider operands at `amdgpu_isa_cdna4.xml:129556` through `:129564`
  and `:136450` through `:136453`.
- TTMP operand enumerators describe trap temporaries as privileged at
  `amdgpu_isa_cdna4.xml:122809` through `:122870`.
- The XML does not encode SGPR allocation units, the logical 0-101 SGPR view,
  VCC's highest-SGPR physical aliasing, the extra trap SGPR reservation, source
  out-of-range substitution, destination nullification rules, multiple-return
  checks, or the LDS `min(lds_size, M0)` clamp rule. The LDS-specific portion
  overlaps `CDNA4-XML-032`.

Impact:

Operand metadata can identify register classes and some generic alignment text,
but cannot drive CDNA4 allocation legality or out-of-range execution behavior
without manual-derived rules.

### CDNA4-XML-040: M0, SCC, and VCC/VCCZ semantics are mostly prose-only

Manual evidence:

- Section 3.7 assigns `M0` roles for LDS addressing and indirect GPR indexing at
  `cdna4/README.md:583` through `:590`.
- Section 3.8 classifies which scalar operation families set `SCC`, which leave
  it unchanged, and how it is used as carry-in and branch/conditional-move state
  at `cdna4/README.md:592` through `:604`.
- Section 3.9 defines VCC compare results, `EXEC` masking, full VCC writes,
  VCCZ updates for vector compares and scalar writes to VCC, physical VCC
  aliasing, and the scalar-write-to-physical-VCC hazard at `cdna4/README.md:606`
  through `:624`.

XML evidence:

- The XML describes `M0` generically as a special register for LDS/GDS
  addresses, relative indices, and send messages at
  `amdgpu_isa_cdna4.xml:119180` through `:119181` and repeats similar operand
  text at `:122903` through `:122904`.
- `SRC_VCCZ`, `SRC_EXECZ`, and `SRC_SCC` operand values have short helper
  descriptions at `amdgpu_isa_cdna4.xml:119645` through `:119657`.
- The XML does not encode the `M0` bit layouts from Chapter 3, SCC producer
  taxonomy, VCC full-write behavior, VCCZ update-on-scalar-write rule, physical
  alias hazard, or the compare pseudocode `VCC[n] = EXEC[n] & test`.

Impact:

XML-only execution metadata cannot reconstruct the wave-state side effects and
helper-bit updates needed for exact SCC/VCC/M0 behavior.

### CDNA4-XML-041: Trap, memory-violation, HW_ID, and launch-initialization state is not machine-readable

Manual evidence:

- Sections 3.10 and 3.10.1 define TTMP privilege, trap payload packing into
  `TTMP0`/`TTMP1`, `STATUS.TRAP_EN`, `MODE.EXCP_EN`, and `TRAPSTS` sticky fields
  at `cdna4/README.md:626` through `:668`.
- Section 3.11 defines memory-violation sources, non-sources, buffer-to-LDS
  behavior, sticky `TRAPSTS.mem_viol`, trap enable interaction, and imprecise PC
  reporting at `cdna4/README.md:670` through `:689`.
- Sections 3.12 and 3.13 define `HW_ID`/`XCC_ID` fields and compute launch
  initialization for packed `VGPR0`, optional system SGPRs including `TG_SIZE`,
  and `TTMP4` through `TTMP11` at `cdna4/README.md:691` through `:731`.

XML evidence:

- `OPR_HWREG` names `HW_REG_TRAPSTS`, `HW_REG_HW_ID`, and `HW_REG_XCC_ID` at
  `amdgpu_isa_cdna4.xml:116862` through `:116950`, but the descriptions are
  `N/A` and no bitfield metadata is attached.
- TTMP operand descriptions name privileged trap temporaries at
  `amdgpu_isa_cdna4.xml:122809` through `:122870`, but do not encode the
  Chapter 3 trap payload, launch payload, or uninitialized-TTMP rule.
- A static search found no XML metadata for `MEM_VIOL`, `SAVECTX`,
  `ILLEGAL_INST`, `EXCP_CYCLE`, `DP_RATE`, or `TG_SIZE` beyond the
  `HW_REG_TRAPSTS` name itself.

Impact:

XML consumers cannot derive trap-status layout, memory-violation reporting,
hardware-ID bitfields, or compute-launch register initialization from the
machine-readable records.

### CDNA4-XML-042: Program-control stack, trap-return, and debug-branch side effects are prose-only

Manual evidence:

- Section 4.1 lists `S_RFE` as return from trap handler, `S_SLEEP` with a
  64-to-8128 cycle delay, `S_SENDMSG`, `S_WAKEUP`, and related control
  instructions at `cdna4/README.md:736` through `:752`.
- Section 4.2 lists conditional debug branches, direct PC operations,
  `S_SETVSKIP`, and `S_CALL_B64`, including the `PC+4` return value and branch
  target formula, at `cdna4/README.md:754` through `:779`.
- Section 4.6 defines arbitrary divergent control flow, including the six-deep
  fork/join stack, CSP mode bits, `{exec[63:0], PC[47:2]}` stack entries,
  pass/fail mask selection, and `EXEC`/PC/SGPR update pseudocode at
  `cdna4/README.md:868` through `:924`.

XML evidence:

- The XML records opcode names and broad branch flags for `S_CBRANCH_JOIN`,
  including implicit `EXEC` and `PC` outputs, at
  `amdgpu_isa_cdna4.xml:34757` through `:34789`.
- `S_CBRANCH_G_FORK` and `S_CBRANCH_I_FORK` are present with terse
  branch-stack descriptions at `amdgpu_isa_cdna4.xml:41040` through `:41071`
  and `:45465` through `:45500`.
- The debug branch opcodes are present with short system/user flag
  descriptions at `amdgpu_isa_cdna4.xml:46410` through `:46536`.
- `S_CALL_B64` has the high-level return-and-branch description at
  `amdgpu_isa_cdna4.xml:45616` through `:45635`.
- The XML does not encode the fork/join pseudocode, CSP storage layout,
  branch-stack depth, path-selection rules, trap-return state updates, debug
  flag source bits, or the precise sleep/wakeup scheduling side effects.

Impact:

XML-driven control-flow semantics can identify the affected opcodes and some
implicit PC/EXEC effects, but cannot reconstruct CDNA4 divergent-control,
trap-return, debug-branch, or sleep/wakeup behavior without manual-derived
rules.

### CDNA4-XML-043: Workgroup barrier membership and release behavior are prose-only

Manual evidence:

- Section 4.3 states that a workgroup can contain up to 16 wavefronts / 1024
  work-items and that `S_BARRIER` waits for all other wavefronts in the same
  workgroup at the same instruction, with early-terminated waves considered to
  have satisfied the barrier, at `cdna4/README.md:781` through `:783`.

XML evidence:

- `S_BARRIER` is present with the short description "Synchronize waves within a
  threadgroup" and no operands at `amdgpu_isa_cdna4.xml:46000` through
  `:46023`.
- The XML record does not encode same-workgroup membership, same-instruction
  matching, the 16-wavefront / 1024-work-item limits, or the early-terminated
  wave rule.

Impact:

The XML can identify the barrier opcode, but cannot drive validation or exact
barrier release behavior from machine-readable metadata alone.

### CDNA4-XML-044: Wait-counter producer, decrement, and ordering semantics are prose-only

Manual evidence:

- Section 4.4 defines `VM_CNT`, `LGKM_CNT`, and `EXP_CNT` behavior at
  `cdna4/README.md:785` through `:817`.
- The prose assigns producer/decrement rules by instruction class: vector
  memory reads/writes for `VM_CNT`, LDS/scalar-memory/FLAT/SENDMSG operations
  for `LGKM_CNT`, scalar-memory reads counted by returned dwords, and
  same-type ordering plus scalar-memory out-of-order restrictions.

XML evidence:

- `S_WAITCNT` is present with an `OPR_WAITCNT` operand at
  `amdgpu_isa_cdna4.xml:46056` through `:46087`.
- `OPR_WAITCNT` exposes the `EXP`, `LGKM`, and split `VM` bitfields at
  `amdgpu_isa_cdna4.xml:146132` through `:146170`.
- The XML does not encode which instruction families increment or decrement
  each counter, scalar-memory dword-count behavior, `S_SENDMSG` participation
  in `LGKM_CNT`, return ordering, or the scalar-memory `S_WAITCNT 0` rule.

Impact:

The XML is sufficient for immediate decoding and display, but not for deriving
the architectural wait-counter dependency contract.

### CDNA4-XML-045: Required software-inserted wait-state hazards are prose-only

Manual evidence:

- Section 4.5 lists required manually inserted wait states and independent
  instruction counts for hazards involving `S_SETREG`, `S_SETVSKIP`,
  `MODE.vskip`, VALU-to-helper branch dependencies, readlane/lane-selection,
  `V_DIV_FMAS`, flat/buffer store data overwrite, SALU writes used by VMEM or
  `S_SENDMSG`, DPP, mixed VCC aliasing, `S_RFE`, M0-to-LDS/addressed
  operations, `S_MOVEREL`, `V_CMPX`, and transcendental operations at
  `cdna4/README.md:819` through `:867`.

XML evidence:

- The XML records `S_NOP` as an immediately executed instruction that delays
  issue of the next instruction at `amdgpu_isa_cdna4.xml:45665` through
  `:45694`.
- The affected instruction records do not carry producer/consumer hazard
  metadata, required independent-instruction counts, or the table-specific
  exception predicates.

Impact:

Schedulers, validators, and emulators cannot infer the CDNA4 software wait-state
contract from XML metadata alone.

### CDNA4-XML-046: SALU selector fallback and reserved-source details are incomplete

Manual evidence:

- `cdna4/README.md:948` through `:1022` defines the scalar source/destination
  selector tables, the SALU literal exception, out-of-range source and
  destination behavior, and 64-bit SGPR pair alignment.
- The same table marks selector 239 as reserved, and says an out-of-range
  source SGPR reads SGPR0 while an out-of-range destination SGPR suppresses the
  SGPR write but still allows SCC and saveexec EXEC updates.

XML evidence:

- XML scalar source operand tables include selector 239 as
  `SRC_POPS_EXITING_WAVE_ID`, for example at `amdgpu_isa_cdna4.xml:119680`
  through `:119682` and in later source operand classes.
- `OPR_SDST` at `amdgpu_isa_cdna4.xml:117027` through `:117040` and
  `OPR_SSRC` at `:137700` through `:137716` carry broad scalar operand
  metadata, but static XML search did not find the out-of-range source fallback
  or destination no-write side-effect rule.

Impact:

XML consumers cannot recover CDNA4's SALU out-of-range behavior or trust the
reserved/source-special selector split from XML alone.

### CDNA4-XML-047: Signed SALU add/sub SCC descriptions contradict overflow prose

Manual evidence:

- `cdna4/README.md:1023` through `:1033` defines SCC and distinguishes signed
  arithmetic overflow from unsigned carry-out.
- The Chapter 5 arithmetic table says `S_ADD_I32` and `S_SUB_I32` set SCC on
  overflow at `cdna4/README.md:1040` through `:1044`.

XML evidence:

- The `S_ADD_I32` record at `amdgpu_isa_cdna4.xml:35645` describes a signed
  32-bit add but says the carry-out bit goes into SCC.
- The `S_SUB_I32` record at `amdgpu_isa_cdna4.xml:35784` has the same signed
  description/carry-out wording mismatch.
- The XML records implicit SCC operands for these instructions, but the
  machine-readable record does not disambiguate signed overflow from carry.

Impact:

Codegen driven only by XML prose can derive the wrong SCC predicate for signed
scalar add/sub.

### CDNA4-XML-048: `S_MAX_{I32,U32}` tie SCC predicate is not machine-readable

Manual evidence:

- The detailed `S_MAX_I32` and `S_MAX_U32` definitions use inclusive predicates:
  `SCC = S0.i32 >= S1.i32` and `SCC = S0.u32 >= S1.u32` at
  `cdna4/README.md:3940` through `:3953`.

XML evidence:

- The XML `S_MAX_I32` and `S_MAX_U32` records at
  `amdgpu_isa_cdna4.xml:36519` through `:36658` describe setting SCC iff the
  first value is selected, but do not encode the tie predicate.

Impact:

An XML-only semantic derivation cannot determine whether equal operands should
set or clear SCC.

### CDNA4-XML-049: M0-relative scalar move indexing is under-described

Manual evidence:

- The Chapter 5 summary says `S_MOVRELS_{B32,B64}` and
  `S_MOVRELD_{B32,B64}` move through `SGPR[S0+M0]` / `SGPR[D+M0]`, that M0 is
  an unsigned index, and that the index must be even for 64-bit forms at
  `cdna4/README.md:1128` through `:1130`.
- The detailed B64 definitions repeat that `addr += M0.u32[31:0]` and that the
  M0 index and operand address must be even at `cdna4/README.md:5298` through
  `:5355`.

XML evidence:

- `S_MOVRELS_B64` and `S_MOVRELD_B64` exist in XML at
  `amdgpu_isa_cdna4.xml:34588` and `:34698`, and their records include implicit
  M0 use metadata.
- The XML instruction records do not encode the raw `S0+M0` / `D+M0` formula,
  M0's unsigned-index interpretation, or the even-index legality requirement
  for 64-bit forms.

Impact:

XML consumers can see that M0 participates, but cannot infer the address
formula or legality checks needed for the relative scalar moves.

### CDNA4-XML-050: HWREG map and subfield metadata are incomplete or inconsistent

Manual evidence:

- Chapter 5.8 defines `SIMM16 = {size[4:0], offset[4:0], hwRegId[5:0]}` and
  the CDNA4 hardware-register ID map at `cdna4/README.md:1138` through `:1195`.
- The same section marks HWREG IDs 8 through 15 reserved and gives bitfield
  layouts for `IB_STS`, `GPR_ALLOC`, and `LDS_ALLOC`.

XML evidence:

- `OPR_HWREG` at `amdgpu_isa_cdna4.xml:116838` through `:116995` records the
  ID/OFFSET/SIZE bit partition and many register names, but most descriptions
  are `N/A`.
- The XML names IDs 8 through 15 as `PC_LO`, `PC_HI`, `INST_DW0`, `INST_DW1`,
  `IB_DBG0`, `IB_DBG1`, `FLUSH_IB`, and `SH_MEM_BASES`, while the CDNA4 manual
  Chapter 5 table marks those IDs reserved.
- The XML record does not provide the manual's access permissions or the
  `IB_STS`, `GPR_ALLOC`, and `LDS_ALLOC` subfield layouts.

Impact:

XML-driven HWREG decoders need manual cross-reference to avoid stale IDs and to
recover access/register-layout semantics.

### CDNA4-XML-051: SETREG write masks, side effects, and spacing rules are prose-only

Manual evidence:

- The Chapter 5 access table says an `S_NOP` is required between consecutive
  `S_SETREG` writes to the same register at `cdna4/README.md:1141`.
- Detailed `S_SETREG_B32` and `S_SETREG_IMM32_B32` pseudocode masks writes with
  `HwRegWriteMask(hwRegId, WAVE_STATUS.PRIV)` and notes side effects at
  `cdna4/README.md:4580` through `:4625`.

XML evidence:

- `S_GETREG_B32`, `S_SETREG_B32`, and `S_SETREG_IMM32_B32` records expose the
  `OPR_HWREG` operand at `amdgpu_isa_cdna4.xml:45510` through `:45610`, and
  `S_SETREG_IMM32_B32` also exposes `OPR_SIMM32`.
- Those XML records do not encode `HwRegWriteMask`, privilege filtering,
  register-specific side effects, or the consecutive-SETREG spacing rule.

Impact:

Validators and emulators cannot infer correct SETREG write legality or required
software scheduling constraints from XML metadata alone.

### CDNA4-XML-052: VALU scalar-source budget restrictions are prose-only

Manual evidence:

- Chapter 6.2.1 allows VALU inputs from VGPRs, SGPRs, inline constants,
  literals, M0, and EXEC, but then limits each instruction to at most one SGPR
  source and at most one literal, with literals disallowed when an SGPR or M0 is
  also used at `cdna4/README.md:1241` through `:1256`.
- The same section says `ADDC`, `SUBB`, and `CNDMASK` implicitly use VCC and
  therefore cannot use an additional SGPR or literal at `cdna4/README.md:1257`
  through `:1259`.

XML evidence:

- Representative VOP3 records expose broad per-operand classes: `V_ADD_F32`
  uses `OPR_SRC_NOLIT` and `OPR_SRC_SIMPLE` at
  `amdgpu_isa_cdna4.xml:56232` through `:56252`, while `V_CNDMASK_B32` adds an
  explicit `OPR_SREG` third source at `:56082` through `:56108`.
- The operand classes themselves list scalar-source subtypes independently, for
  example `OPR_SRC_NOLIT` at `amdgpu_isa_cdna4.xml:124697` through `:124709`
  and `OPR_SRC_SIMPLE` at `:127127` through `:127138`.
- The XML records do not encode a cross-operand scalar-read budget, the
  SGPR-versus-literal exclusion, or the implicit-VCC special case for
  `ADDC`/`SUBB`/`CNDMASK`.

Impact:

XML-only validators and testcase generators can assemble operand combinations
that the CDNA4 VALU prose declares illegal.

### CDNA4-XML-053: SDWA/OPSEL dependency hazards are prose-only

Manual evidence:

- Chapter 6.2.1 says DOT instructions must not use SDWA or OPSEL, and that a
  VALU instruction using SDWA or OPSEL must not have its result consumed by the
  next VALU instruction; an independent instruction or `V_NOP` must be inserted
  at `cdna4/README.md:1263` through `:1267`.

XML evidence:

- XML exposes the raw selector/encoding machinery, such as `OP_SEL` for VOP3 at
  `amdgpu_isa_cdna4.xml:2912` through `:2913` and representative SDWA
  encodings such as `V_ADD_F32`'s `VOP2_VOP_SDWA` form at `:56207` through
  `:56229`.
- The affected instruction records do not carry DOT-specific SDWA/OPSEL
  illegality or the next-VALU producer/consumer hazard predicate.

Impact:

Schedulers and validators driven by XML cannot infer this required
software-inserted VALU separation.

### CDNA4-XML-054: VOP3 output-modifier MODE and denormal rules are prose-only

Manual evidence:

- Chapter 6.2.2 says VOP3 floating-point results may use OMOD and CLAMP, but
  output modifiers apply only to floating-point results, are ignored for
  integer/bit results, are ignored when output denormals are enabled, flush
  denormals when output denormals are disabled, flush `-0` to `+0`, and are
  ignored when IEEE mode is set at `cdna4/README.md:1279` through `:1289`.

XML evidence:

- `ENC_VOP3` records raw `CLAMP` and `OMOD` fields at
  `amdgpu_isa_cdna4.xml:2861` through `:2869` and `:2891` through `:2899`.
- The field records say only that CLAMP clamps to `[0.0, 1.0]` and that OMOD is
  applied before clamping; they do not encode operand-result type filtering,
  MODE.IEEE gating, FP_DENORM gating, denormal flushing, or `-0` handling.

Impact:

XML consumers cannot derive correct VOP3 output-modifier behavior without the
manual prose and MODE bit semantics.

### CDNA4-XML-055: VALU FP round/denorm modes and conversion/DOT overrides are prose-only

Manual evidence:

- Chapter 6.4 defines MODE-controlled FP_ROUND and FP_DENORM fields for VALU
  floating-point behavior and notes that floating-point `V_DOT2` instructions
  ignore denorm/round modes and flush input and output denormals at
  `cdna4/README.md:1479` through `:1487`.
- `V_CVT_PK_{F16,BF16}_F32` and `V_CVT_SR_{F16,BF16}_F32` force
  round-nearest-even around the conversion at `cdna4/README.md:17352` through
  `:17376` and `:17963` through `:18002`; the stochastic forms also say OMOD
  and clamp are ignored at `:17958` through `:18002`.

XML evidence:

- Representative DOT records such as `V_DOT2C_F32_F16` and `V_DOT2_F32_F16`
  expose operands and basic descriptions at `amdgpu_isa_cdna4.xml:63809`
  through `:63882` and `:73882` through `:73914`.
- Conversion records such as `V_CVT_PK_{F16,BF16}_F32` and
  `V_CVT_SR_{F16,BF16}_F32` expose operands and broad descriptions at
  `amdgpu_isa_cdna4.xml:70774` through `:70846` and `:72540` through `:72612`.
- Those instruction records do not encode MODE-dependent rounding/denormal
  behavior, conversion-specific forced RNE, stochastic OMOD/clamp ignoring, or
  the floating `V_DOT2` flush override.

Impact:

An XML-only emulator cannot determine when VALU FP operations honor MODE state,
when conversion opcodes override rounding or modifiers, or when DOT2 must ignore
that state and flush denormals.

### CDNA4-XML-056: ALU clamp-bit semantic overloads are prose-only

Manual evidence:

- Chapter 6.5 says the VOP3 clamp bit signals FP exceptions for `V_CMP`,
  saturates integer results to the representable extrema, and clamps
  floating-point results to `[0.0, 1.0]` at `cdna4/README.md:1489` through
  `:1491`.

XML evidence:

- `ENC_VOP3` describes `CLAMP` only as clamping output to `[0.0, 1.0]` at
  `amdgpu_isa_cdna4.xml:2861` through `:2863`.
- Representative integer and compare records, such as `V_ADD_U32` at
  `amdgpu_isa_cdna4.xml:63624` through `:63647`, `V_CMP_F_F32` at `:82827`
  through `:82925`, and `V_CMPX_F_F32` at `:84699` through `:84729`, do not
  encode the integer saturation or compare exception-signaling meanings.
- The same gap applies to the Chapter 12.11 signed add/sub records:
  `V_ADD_I32`, `V_SUB_I32`, `V_ADD_I16`, and `V_SUB_I16` describe ordinary
  arithmetic at `amdgpu_isa_cdna4.xml:70480` through `:70655`, while their
  manual definitions add signed-domain saturation notes.

Impact:

XML consumers can see the raw clamp bit but not the instruction-class-specific
meaning needed for integer ALU and compare operations.

### CDNA4-XML-057: VGPR indexing bit layout and operand-role mapping are prose-only

Manual evidence:

- Chapter 6.6 defines GPR indexing as MODE-gated VGPR indexing through M0,
  with `M0[7:0]` as the index and `M0[15:12]` as dest/src2/src1/src0 enable
  bits, and says indexed accesses apply only to VGPR operands and must not go
  out of range at `cdna4/README.md:1493` through `:1522`.
- Chapter 6.6.2 then remaps those role bits for `v_readlane`,
  `v_readfirstlane`, `v_writelane`, `v_mac_*`, `v_madak`, `v_madmk`, reverse
  shifts, `v_cvt_pkaccum`, and SDWA read-modify-write forms at
  `cdna4/README.md:1524` through `:1538`.

XML evidence:

- XML records the `S_SET_GPR_IDX_IDX`, `S_SET_GPR_IDX_ON`,
  `S_SET_GPR_IDX_OFF`, and `S_SET_GPR_IDX_MODE` instructions with implicit M0
  operands at `amdgpu_isa_cdna4.xml:34897` through `:34921`,
  `:44429` through `:44457`, and `:46571` through `:46620`.
- Those records do not encode MODE.gpr_idx_en, the M0 bit layout, the
  source/destination role mapping table, or the out-of-range indexed-VGPR
  illegality.

Impact:

XML consumers can tell that M0 participates in GPR indexing, but cannot recover
the bit placement or per-instruction operand-role semantics.

### CDNA4-XML-058: `S_ATC_PROBE*` SMEM opcodes are XML-only for CDNA4

Manual evidence:

- Chapter 13.2's SMEM opcode table lists opcodes 0-37, 40-41, 64-76, 96-108,
  128-140, and 160-172 at `cdna4/README.md:25060` through `:25107`, but has no
  entries for opcodes 38 or 39.
- Searching the CDNA4 manual prose finds no `S_ATC_PROBE` or
  `S_ATC_PROBE_BUFFER` instruction definition or semantic description.

XML evidence:

- The CDNA4 XML defines `S_ATC_PROBE` at opcode 38 and
  `S_ATC_PROBE_BUFFER` at opcode 39, both as `ENC_SMEM` instructions that probe
  or prefetch an address into the scalar data cache at
  `amdgpu_isa_cdna4.xml:29408` through `:29478`.
- A direct inventory comparison found 82 manual SMEM opcode-table entries and
  84 XML `ENC_SMEM` records; the only XML-only opcodes were 38 and 39.

Impact:

Manual-derived validators will reject or fail to classify these two XML-defined
SMEM opcodes, while XML-derived decoders expose a CDNA4 instruction surface that
has no corroborating behavior in the CDNA4 manual.

### CDNA4-XML-059: SOPP opcode table omits XML-only ordered-PS and co-execution entries

Manual evidence:

- Chapter 12.5 lists detailed SOPP instruction definitions from `S_NOP`
  opcode 0 through `S_SET_GPR_IDX_MODE` opcode 29 at `cdna4/README.md:5718`
  through `:6078`; a direct search of the CDNA4 manual finds no
  `S_ENDPGM_ORDERED_PS_DONE` or `S_SET_VALU_COEXEC_MODE` entry.
- Chapter 13.1.5 Table 74 lists SOPP opcodes 0 through 29, then transitions to
  the SMEM format at `cdna4/README.md:25015` through `:25035`.

XML evidence:

- XML records `S_ENDPGM_ORDERED_PS_DONE` as `ENC_SOPP` opcode 30 with
  program-terminator metadata at `amdgpu_isa_cdna4.xml:46638` through
  `:46645`.
- XML records `S_SET_VALU_COEXEC_MODE` as `ENC_SOPP` opcode 31 with a
  `SIMM16` operand and vector-ALU co-execution-mode description at
  `amdgpu_isa_cdna4.xml:46663` through `:46675`.

Impact:

Manual-derived CDNA4 SOPP inventories stop at opcode 29 and can classify
opcodes 30 and 31 as reserved or absent, while XML-derived decoders and
generated rocjitsu expose both encodings.

### CDNA4-XML-060: SOP2 opcode table omits XML-only `S_RFE_RESTORE_B64`

Manual evidence:

- Chapter 12.1 lists detailed SOP2 instruction definitions from `S_ADD_U32`
  opcode 0 through `S_PACK_HH_B32_B16` opcode 52, but skips from
  `S_ABSDIFF_I32` opcode 42 to `S_MUL_HI_U32` opcode 44 at
  `cdna4/README.md:4280` through `:4313`.
- Chapter 13.1.1 Table 66 likewise lists SOP2 opcodes 0 through 42 and 44
  through 52, with no opcode-43 row at `cdna4/README.md:24755` through
  `:24787`.
- A direct search of the CDNA4 manual finds no `S_RFE_RESTORE_B64`
  instruction entry; Chapter 4 only describes the broader return-from-exception
  control-flow behavior.

XML evidence:

- XML records `S_RFE_RESTORE_B64` as `ENC_SOP2` opcode 43, with 64-bit `SSRC0`,
  32-bit `SSRC1`, and implicit PC output at `amdgpu_isa_cdna4.xml:41300`
  through `:41327`.
- A direct inventory comparison found 52 manual SOP2 opcode-table entries and
  53 XML `ENC_SOP2` records; opcode 43 was the only XML-only SOP2 entry.

Impact:

Manual-derived CDNA4 SOP2 inventories can classify opcode 43 as reserved or
absent, while XML-derived decoders expose a trap-return instruction that has no
corresponding detailed CDNA4 manual definition.

### CDNA4-XML-061: `S_SET_GPR_IDX_ON` exposes literal variants for a raw 4-bit mode field

Manual/oracle evidence:

- The detailed `S_SET_GPR_IDX_ON` definition says the raw bits of the `SRC1`
  field are used to set the enable bits, then assigns `M0[15:12]` from
  `SRC1[3:0]` and explicitly calls this the direct content of the raw `S1` field
  at `cdna4/README.md:5682` through `:5698`.
- The generic SOPC field table permits selector 255 as a literal source at
  `cdna4/README.md:24930` through `:24979`, but the instruction-specific
  `S_SET_GPR_IDX_ON` text narrows operand 1 to the raw 4-bit mode field. As a
  cross-check, `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx950` accepts
  `s_set_gpr_idx_on s0, 15` and `s_set_gpr_idx_on 0x12345678, 15`, but rejects
  `s_set_gpr_idx_on s0, 16` and `s_set_gpr_idx_on s0, 0x12345678`.

XML evidence:

- XML records the default `S_SET_GPR_IDX_ON` second operand as field `SSRC1`
  with `OPR_SIMM4`, which matches the 4-bit mode intent at
  `amdgpu_isa_cdna4.xml:44429` through `:44459`.
- The same instruction also has generated `SOPC_INST_LITERAL` alternatives for
  `has_lit_1` and `has_lit_0_has_lit_1`, replacing operand 1 with `SIMM32` while
  retaining `OPR_SIMM4`, at `amdgpu_isa_cdna4.xml:44490` through `:44547`.

Impact:

XML consumers cannot tell that only source 0 may use the SOPC extension literal
for this instruction. A decoder or assembler generated directly from the XML can
accept or require an extension word for operand 1 even though the ISA text and
LLVM assembler treat that operand as a raw 4-bit mode field.

### CDNA4-XML-062: Generic SMEM `SBASE` field describes buffer descriptor payload for ordinary SMEM

Manual evidence:

- Table 39 says `SBASE` is an SGPR pair with an implied low bit that provides a
  base address, or, for buffer instructions, a four-SGPR resource constant whose
  used fields are base, stride, and `num_records`, at `cdna4/README.md:2803`
  through `:2814`.
- Section 8.2.1.1 states that all scalar/global/scratch address components are
  byte quantities at `cdna4/README.md:2846` through `:2849`.
- Chapter 13.2's SMEM field table repeats that `SBASE` is an SGPR pair for the
  base address or an SGPR quad for buffer `V#` at `cdna4/README.md:25047`
  through `:25057`.

XML evidence:

- The generic `ENC_SMEM` field description says `SBASE` names bits `[6:1]` of
  an aligned SGPR pair specifying `{size[16], base[48]}` in dword units at
  `amdgpu_isa_cdna4.xml:907` through `:920`.
- Per-instruction operand metadata partly corrects the role split: ordinary
  `S_LOAD_DWORD` uses `SBASE` as a 64-bit `FMT_BUF` operand at
  `amdgpu_isa_cdna4.xml:28180` through `:28185`, while
  `S_BUFFER_LOAD_DWORD` uses `SBASE` as a 128-bit `FMT_RSRC_SCALAR` operand at
  `:28572` through `:28577`.

Impact:

An XML consumer that reads the encoding field metadata directly can treat all
SMEM `SBASE` operands as a dword-unit `{size,base}` descriptor, contradicting
the manual's byte-address SGPR-pair rule for ordinary scalar/global/scratch
SMEM. The instruction records mitigate decode/display width, but the generic
field prose still carries the wrong semantic contract.

## No-Gap Notes

- CDNA4 Chapter 12.10 has 104 detailed VOP3P opcode headings, and each one has
  a matching XML `ENC_VOP3P` or `VOP3P_MFMA` record with the same opcode
  number. The only additional VOP3P-family XML records are the two
  `ENC_VOP3PX2` scaled F8F6F4 forms sharing opcodes 45 and 46; those are
  already covered by the block-scale gaps above.
- CDNA4 Chapter 12.11 VOP3B's ten-opcode inventory matches XML
  `VOP3_SDST_ENC` records for opcodes 281 through 286, 480, 481, 488, and 489.
  The VOP3B gaps above are about semantic and generic-field prose, not missing
  instruction rows.
- CDNA4 XML separates `VOP3_SDST_ENC` from generic `ENC_VOP3` and omits the
  `ABS` field from `VOP3_SDST_ENC`, matching Chapter 13.3.5's VOP3B field map.
  Generic `ENC_VOP3` keeps `ABS` and `OP_SEL`, matching Chapter 13.3.4's VOP3A
  field map for ordinary VOP3A forms.
- CDNA4 XML preserves the Chapter 12.11 `V_CVT_PKRTZ_F16_F32`
  round-toward-zero description and VOP3 operand shape for opcode 662 at
  `amdgpu_isa_cdna4.xml:70260` through `:70295`. The matching runtime issue is
  recorded as `CDNA4-RJ-113`, not as a manual/XML gap.
- Chapter 12.11 parsed to 507 detailed instruction-definition headings. After
  excluding already-audited promoted VOP1/VOP2/VOPC rows and the ten VOP3B
  rows, all 162 native-only `ENC_VOP3` rows have matching XML opcode records.
- CDNA4 Chapter 12.11 definitions 640-659 have matching XML `ENC_VOP3` opcode
  records for opcodes 640-653 and 655-659; opcode 654 is absent from both the
  manual and XML. The gaps above are semantic/prose gaps: `V_BCNT_U32_B32`'s
  `SRC1` addend, readlane/writelane lane masks, and `V_TRIG_PREOP_F64`
  range-reduction details.
- CDNA4 Chapter 12.11 definitions 660-673 have matching XML `ENC_VOP3` opcode
  records for opcodes 660-666 and 668-673; opcode 667 is absent from both the
  manual and XML. `V_CVT_PKRTZ_F16_F32` and `V_MUL_LEGACY_F32` preserve the
  instruction-local RTZ and DX9 zero-multiply descriptions in XML. The signed
  add/sub saturation note remains covered by the generic clamp-bit overload gap
  `CDNA4-XML-056`.
- CDNA4 Chapter 12.11 definitions 680-681 have matching XML `ENC_VOP3` opcode
  records for `V_MINIMUM3_F32` and `V_MAXIMUM3_F32`. The remaining semantic
  issue for this pair is the MODE/DX10_CLAMP prose gap recorded as
  `CDNA4-XML-081`; the following repeated VOPC class/compare definitions are
  covered by the full VOPC definition slice.
- Chapter 12.11 repeats the VOP1 promoted-opcode source drift already recorded
  in `CDNA4-XML-069`: the detailed VOP3A headings and Chapter 13.3.4 table use
  the conflicting `+0x180` offset, while XML uses the Chapter 12.8.1 `+0x140`
  mapping.
- The three XML-only primary VOP3A rows in this slice,
  `V_SCREEN_PARTITION_4SE_B32`, `V_EXP_LEGACY_F32`, and `V_LOG_LEGACY_F32`, are
  the same XML-only VOP1 records already tracked by `CDNA4-XML-068`.
- CDNA4 Chapter 13.3.6 VOP3P Table 90 matches the audited Chapter 12.10
  detailed-definition inventory for the non-scale opcode rows. The VOP3P gaps
  above are about instruction-local semantics, source/modifier exceptions, and
  scale-prefix metadata, not missing base opcode-table rows.
- CDNA4 Chapter 13.3.6 lists the VOP3P and VOP3P-MAI `ENCODING` rows as
  `[31:24]` while showing the 9-bit value `110100111` at
  `cdna4/README.md:26000` and `:26066`; XML models the same encoding as
  9 bits at offset 23 at `amdgpu_isa_cdna4.xml:2222` through `:2227` and
  `:7537` through `:7542`. Treat this as manual/extraction drift rather than
  an XML decode gap.
- CDNA4 XML records `V_ACCVGPR_READ` and `V_ACCVGPR_WRITE` with `_B32` aliases,
  32-bit operand sizes, and the expected `OPR_SRC_ACCVGPR` /
  `OPR_ACCVGPR` register classes at `amdgpu_isa_cdna4.xml:76209` through
  `:76272`.
- CDNA4 XML includes the audited Chapter 6.3 VALU opcode inventory for real
  instruction names in the table; apparent misses such as `V_CMP`/`V_CMPX` are
  family headers, and `_MK`/`_AK` forms are present as concrete opcode entries.
- CDNA4 XML records VALU literal expansion in the `SRC_LITERAL` predefined
  value, including 16-bit low-half use and 64-bit integer/floating expansion at
  `amdgpu_isa_cdna4.xml:124690` through `:124692`. The Chapter 6 gaps above
  are about operand-combination legality and stateful execution semantics, not
  absence of literal expansion text.
- CDNA4 XML records raw VOP3 `ABS`, `NEG`, `CLAMP`, `OMOD`, and `OP_SEL`
  fields, plus representative VOP2/VOPC SDWA and literal encodings. The VALU
  modifier and hazard gaps above are about missing cross-field, cross-operand,
  and MODE-dependent semantics rather than missing bit positions.
- CDNA4 XML records all 62 audited Chapter 12.7 VOP2 opcode definitions, with
  opcodes 0 through 61 matching the manual table. The promoted VOP3 opcode
  value is consistently `VOP2 opcode + 0x100` for ordinary promoted forms, and
  XML correctly omits VOP3/DPP/SDWA encodings for the literal-only `_MK`/`_AK`
  forms. The VOP2 gaps above are about instruction-local semantics and modifier
  legality, not missing opcode rows.
- CDNA4 XML records all 85 detailed Chapter 12.8 VOP1 opcode definitions, with
  opcode numbers matching the detailed manual rows. It also uses
  `VOP1 opcode + 0x140` for promoted VOP3 aliases, matching Chapter 12.8.1.
  `CDNA4-XML-068` and `CDNA4-XML-069` record the separate XML-only row drift and
  Chapter 13 promoted-table conflict.
- CDNA4 XML records `V_CMPX`'s implicit EXEC output for VOP3, VOPC, and literal
  encodings, for example at `amdgpu_isa_cdna4.xml:84725` through `:84729` and
  `:84754` through `:84758`. The compare clamp gap is limited to the clamp-bit
  FP-exception meaning, while `CDNA4-XML-073` records the separate absence of
  VOPC DPP extension metadata.
- CDNA4 XML records the `S_SET_GPR_IDX_*` instruction inventory and implicit M0
  operands. The VGPR-indexing gap above is about missing bit layout, MODE state,
  and per-instruction role mapping.
- Chapter 6.2.3's out-of-range VGPR behavior overlaps the broader GPR
  allocation/out-of-range gap already recorded as `CDNA4-XML-039`.
- The CDNA4 XML contains the audited VOP3P packed-math opcode inventory,
  including packed 16-bit arithmetic, MIX, DOT, packed F32 arithmetic, and
  `V_PK_MOV_B32`. The gaps above are semantic and legality gaps, not missing
  opcode entries.
- CDNA4 XML does include the audited Chapter 4 program-control opcode
  inventory, including ordinary branches, debug branches, `S_BARRIER`,
  `S_WAITCNT`, `S_SLEEP`, `S_SETPRIO`, `S_SENDMSG`, `S_SENDMSGHALT`, `S_TRAP`,
  `S_ENDPGM_SAVED`, `S_CBRANCH_{I,G}_FORK`, `S_CBRANCH_JOIN`, and
  `S_CALL_B64`. The Chapter 4 XML gaps above are about missing execution
  semantics, not absent instruction records.
- CDNA4 XML includes the partitioned `OPR_WAITCNT` bit layout for `EXP`,
  `LGKM`, and split `VM`. The wait-counter gap above is about missing producer,
  decrement, and ordering rules, not missing immediate-field decoding.
- CDNA4 XML does include the HWREG ID enum for `MODE`, `STATUS`, `TRAPSTS`,
  `HW_ID`, allocation registers, TBA/TMA, and `XCC_ID`. The Chapter 3 HWREG
  gaps above are about missing bitfield semantics and state behavior, not absent
  register ID names.
- CDNA4 XML records SOP1/SOP2/SOPK/SOPC field maps and opcode inventories for
  the audited Chapter 5 scalar ALU instructions. The SALU gaps above are about
  semantic predicates, operand legality, and access side effects rather than
  missing base instruction rows.
- CDNA4 XML records the full audited SOP1 opcode inventory: all 54 manual
  opcode rows are present, opcode holes 47 and 49 are absent, and XML omits
  `SOP1_INST_LITERAL` alternatives for the instruction-specific no-literal
  forms `S_GETPC_B64`, `S_SETPC_B64`, `S_SWAPPC_B64`, `S_RFE_B64`,
  `S_MOVRELS_B32`, and `S_MOVRELS_B64`.
- CDNA4 XML records the full generated SOP2 opcode inventory, including all 52
  manual Table 66 entries plus the XML-only opcode 43 recorded as
  `CDNA4-XML-060`. XML supplies `SOP2_INST_LITERAL` alternatives for ordinary
  SOP2 forms and omits them only for `S_RFE_RESTORE_B64` in this audited
  inventory.
- CDNA4 XML records the full audited SOPK opcode inventory: all 21 manual/XML
  opcode rows are present, opcode 19 is absent/reserved, `S_SETREG_IMM32_B32`
  is the only `SOPK_INST_LITERAL` form, and `S_CALL_B64` is modeled as a
  PC-relative call with an `OPR_LABEL` SIMM16 source plus implicit PC
  input/output operands.
- CDNA4 XML records the full audited SOPC opcode inventory: all 20 detailed
  manual definitions and all 20 Table 72 opcode rows match XML `ENC_SOPC`
  records for opcodes 0 through 19. `CDNA4-XML-061` records the separate
  `S_SET_GPR_IDX_ON` source-1 literal overgeneralization.
- CDNA4 XML records the SOPP field map and all 32 generated SOPP opcode records
  for opcodes 0 through 31. `CDNA4-XML-059` records the separate manual-source
  omission for opcodes 30 and 31.
- CDNA4 XML records literal alternatives for SOP1/SOP2/SOPC and the explicit
  `OPR_SIMM32` input for `S_SETREG_IMM32_B32`; the literal gaps above are about
  instruction-specific exceptions and downstream semantics, not complete absence
  of literal metadata.
- CDNA4 XML records implicit SCC operands on the audited SCC-producing and
  SCC-consuming SALU instructions. The SCC gaps above are about predicate
  details and contradictory prose, not missing implicit SCC operand entries.
- CDNA4 XML records implicit M0 operands on `S_MOVRELS_*` and `S_MOVRELD_*`.
  The M0-relative move gap is about missing address formulas and even-index
  legality, not absence of M0 use.
- CDNA4 XML does include the `VOP3P_MFMA` field inventory for `CBSZ`, `ABID`,
  and `BLGP` with their bit positions. The broadcast gap above is about missing
  formulas, legality predicates, and instruction-specific field overrides, not
  absent field bits.
- CDNA4 XML does include the dense MFMA opcode inventory, alias names, packed
  operand formats, operand sizes, and broad VGPR/AccVGPR/constant operand
  classes for the audited dense F32, F16, BF16, I8, F64, FP8, and BF8 MFMA
  forms. The dense MFMA gaps above are about semantic/layout/timing/legality
  metadata, not missing instruction records.
- CDNA4 XML does include the audited SMFMAC opcode inventory, aliases, operand
  widths, packed data formats, and the sparse-specific `SRC2` VGPR-only operand
  class. The sparse MFMA gap above is about missing semantic/layout/state
  metadata, not missing instruction records.
- CDNA4 XML does include generic `OPR_WAITCNT` operand fields and broad
  `VALU`/`MFMA` functional grouping for matrix instructions. The MAI
  dependency-wait gap above is about missing per-instruction hazard predicates,
  wait counts, and forwarding exceptions, not absence of all wait or MFMA
  classification.
- CDNA4 XML does include MFMA operand data formats for the audited F32, F64,
  sub-32-bit float, and I8 families. The Section 7.4 gap above is about missing
  per-datatype floating-point state and exception semantics, not missing
  datatype names or operand widths.
- CDNA4 XML includes the audited manual SMEM opcode-table inventory,
  `ENC_SMEM` field bit positions, `OPR_SMEM_OFFSET` selector enumeration, and
  `FMT_RSRC_SCALAR`. A direct Chapter 12.6/13.2 comparison found 82 manual
  SMEM instruction definitions and 82 manual opcode-table entries; XML contains
  those same entries plus the two XML-only `S_ATC_PROBE*` opcodes recorded in
  `CDNA4-XML-058`. The SMEM gaps above are about behavior that is not
  recoverable from those field and descriptor records, not missing base SMEM
  instruction records.
- CDNA4 XML does include the MUBUF/MTBUF opcode inventory, `ENC_MUBUF` and
  `ENC_MTBUF` field bit positions, typed/untyped buffer split,
  `DFMT`/`NFMT`, `LDS`, `SC0`/`SC1`, `NT`, and `SRSRC` scaling text. Direct
  table comparison matched all 16 MTBUF and 74 MUBUF manual opcode-table
  entries to XML `ENC_MTBUF`/`ENC_MUBUF` records; the XML also carries
  `BUFFER_INV` with `BUFFER_INVL2` as an alias. The vector-buffer gaps above
  are about behavior that is not recoverable from those field records, not
  missing base buffer instruction records.
- Unlike CDNA3, the audited CDNA4 buffer atomic descriptions use `SC0` for the
  optional return contract rather than stale `GLC` wording. `CDNA4-XML-025`
  records the separate operand dataflow and conditional-return metadata gap,
  while `CDNA4-XML-063` records the buffer floating-atomic numeric rules.
- CDNA4 XML does include the FLAT/GLOBAL/SCRATCH opcode inventory,
  `ENC_FLAT`, `ENC_FLAT_GLBL`, and `ENC_FLAT_SCRATCH` field bit positions,
  segment constants, `SC0`/`SC1`, `NT`, `ACC`, direct-to-LDS entries, and
  representative floating atomic data formats. Direct table comparison matched
  all 54 FLAT, 61 GLOBAL, and 27 SCRATCH manual opcode-table entries to XML
  records. The flat-memory gaps above are about semantic, timing, and ambiguous
  field behavior that is not recoverable from those records.
- Unlike CDNA3, the audited CDNA4 FLAT/GLOBAL atomic descriptions use `SC0`
  for optional return wording rather than stale `GLC` terminology. The
  remaining floating-atomic gap is legality and numeric behavior.
- CDNA4 XML does include the non-scale F8F6F4 MFMA opcode entries, operand
  widths, `FMT_NUM_PK8_B32` source formats, F32 accumulator/destination
  formats, VGPR/AccVGPR source classes for `SRC0`/`SRC1`, and
  VGPR/AccVGPR/constant source class for `SRC2`.
- CDNA4 XML does include the audited DS opcode inventory, operand fields, ACC
  destination/source classes, ST64/read2/write2 opcode rows, packed F16/BF16
  LDS atomic DSMEM side effects, and transpose-load operand widths. Direct
  table comparison matched all 126 manual DS opcode-table entries to XML
  `ENC_DS` records; opcode 225 also carries XML alias `DS_READ_B128_TR_B6` for
  the manual's `DS_READ_B96_TR_B6` spelling. The LDS gaps above are about
  missing formulas, constraints, and CDNA4-specific semantics, not absent base
  DS instruction records.
- CDNA4 XML does include `ENC_VOP3PX2`, `SCALE_SRC0`, `SCALE_SRC1`, opcode 45,
  opcode 46, and the 128/512-bit destination plus 256-bit source operand sizes
  for the audited block-scaled MFMA instructions.
- Unlike CDNA3, CDNA4 XML uses explicit packed data formats for packed 32-bit
  forms: `FMT_NUM_PK2_F32` and `FMT_NUM_PK2_B32`.
- The CDNA4 XML carries generic even-alignment text for 64-bit SGPR and VGPR
  operand values. The packed 32-bit XML gap is about VOP3P-specific dword
  selector behavior and `V_PK_MOV_B32` special cases, not absence of all
  alignment prose.
- For scaled FP4/FP6/BF6 conversions, CDNA4 XML carries the audited opcode
  inventory, aliases, operand widths, packed data formats, and VGPR-only classes
  for the 64-bit and wider data payload operands, for example
  `V_CVT_SCALEF32_SR_PK_FP4_F32` at `amdgpu_isa_cdna4.xml:68815` through
  `:68820`, `V_CVT_SCALEF32_2XPK16_FP6_F32` at `:69791` through `:69813`, and
  `V_CVT_SCALEF32_PK32_F32_FP6` at `:70003` through `:70019`.
- CDNA4 XML does include the audited opcode inventory, aliases, operand widths,
  and packed data formats for conversion definitions 613-618 and 678-679. The
  new notes above are about missing semantic dependencies and mode overrides,
  not missing instruction records.
- The detailed MIX instruction pseudocode and XML descriptions both say
  multiply-add. Section 6.7 separately says the MIX multiply-add is fused, so
  that wording remains a manual-internal ambiguity rather than an XML-only
  omission from this static pass.
