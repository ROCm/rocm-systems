# RDNA4 Manual vs XML Gaps

Architecture: RDNA4 / gfx12

Manual source: `workspace_docs/amdgpu-isa-manuals/rdna4/README.md`

XML source: `shared/machine-readable-isa/isa/amdgpu_isa_rdna4.xml`

## Coverage

| Manual section | Status | Notes |
| --- | --- | --- |
| 1.1-1.2 Introduction, terminology, hardware overview, LDS/cache/device memory | Audited statically | Checked document scope, notation/suffix terminology, wave/workgroup/WGP/CU/SIMD/LDS/VMEM terms, host/command-processor/memory-controller overview, LDS/cache hierarchy, FP-exception interrupt capability, and relaxed memory/acknowledgment prose against XML architecture metadata, data-format records, and instruction descriptions. Detailed operational gaps remain tracked in later semantic sections. |
| 2 intro / 2.1-2.5 Shader concepts, wave32/wave64, shader types, work-groups, padding, WQM | Audited statically | Checked wave execution model, compute/graphics shader launch concepts, workgroup/CU/WGP mode constraints, shader-padding requirement, and explicit/temporary WQM behavior against XML functional groups, operand descriptions, instruction entries, and existing detailed semantic gaps. |
| 3.1 State Overview | Audited statically | Most wave-visible state exists only as prose-level architecture state, not XML data. |
| 3.2 Control State: PC and EXEC | Audited statically | PC width/alignment and EXEC skip rules are prose-only. |
| 3.3 Storage State: SGPR, VGPR, LDS | Audited statically | Register OOR/alignment, dynamic VGPR, and LDS behavior are only partially represented. |
| 3.4 Wave State Registers | Audited statically | HWREG IDs exist, but most bitfields and side effects are absent; `IB_STS2` is missing. |
| 3.5 Initial Wave State | Audited statically | Shader-stage SGPR/VGPR/LDS launch payloads are outside XML. |
| 4.1 Common Instruction Fields | Audited statically | Checked inline constants, literal constants, source selector table, and unused-field defaults; DPP and aperture details are cross-checked in later architecture-specific rows. |
| 4.1.1 Cache Controls: SCOPE and Temporal-Hint | Audited statically | XML carries field widths/positions, but most cache policy semantics are prose-only. |
| 5.1/5.2 Program flow and control instructions | Audited statically | Checked SOPP/SOP1 control instructions and Chapter 5 table coverage; follow-on sections cover clauses, messages, branches, barriers, dependency cases, and ALU scheduling. |
| 5.3 Instruction Clauses | Audited statically | XML has field decode, but clause membership and stream rules are prose-only. |
| 5.4 Send Message Types | Audited statically | XML has the main enums, but payload, legality, and return-message wait behavior are incomplete. |
| 5.5 Branching | Audited statically | XML has branch inventory and label offsets, but some branch-wide and PC detail remains prose/descriptive. |
| 5.6 Work-groups and Barriers | Audited statically | XML has split-barrier instruction shells, but work-group state, barrier completion, and one RDNA4 opcode are missing or inconsistent. |
| 5.7 intro / 5.7.1 Memory Dependency Counters | Audited statically | XML has split-wait opcodes and operand bitfields, plus some cache-operation counter annotations, but producer taxonomy, ordering, overflow-stall, and exception/event details are mostly prose-only. |
| 5.7.2 Specific Dependency Cases | Audited statically | Checked state-register read-after-update delay cases for `S_SETHALT`/`S_SETPRIO` before `S_GETREG`; WMMA hazard cross-reference remains tracked under 7.12. |
| 5.8 ALU Instruction Software Scheduling | Audited statically | Checked `S_DELAY_ALU` field coding, dependency-code table, stream-history semantics, clause overlap, and XML-only code-8 wording. |
| 6.1/6.2 SALU formats and operands | Audited statically | Checked SALU format inventory, scalar operand selector table, source/destination and literal rules, out-of-range/alignment overlap, and `S_SETREG_IMM32_B32` literal exception metadata. |
| 6.3-6.7 SALU SCC, arithmetic, conditional, compare, and bit-wise operations | Audited statically | Checked SCC write/read rules, signed-overflow versus carry-out wording, WREXEC destination restrictions, relative SGPR moves, bit-compare masks, and representative arithmetic/bitwise instruction definitions. |
| 6.8-6.10 SALU floating point, state access, and memory aperture query | Audited statically | Checked scalar FP inventory, F16 register-half rules, MODE/exception dependence, GETREG/SETREG metadata, and aperture source-selector formulas. |
| 7.1 Microcode Encodings | Audited statically | Checked VOP1/VOP2/VOPC/VOP3/VOP3SD/VOP3P field summaries, literal forms, modifier metadata, OPSEL rules, VOP3SD inventory, and non-promotable VOP1/VOP2 exceptions. |
| 7.2 Operands | Audited statically | Checked VALU selector table, non-standard operand uses, source-combination restrictions, OPSEL allow-list, output modifier/clamp rules, round/denorm mode prose, SGPR mask/carry behavior, wave64 SGPR notes, OOR GPR behavior, and PERMLANE adjacency rules. |
| 7.3 Instructions | Audited statically | Checked the non-VOP3P VALU inventory/classification table, compact compare-family summary, XML primary and alias names, and concrete compare opcode expansion. |
| 7.4 16-bit Math and VGPRs | Audited statically | Checked true16 VGPR half naming/addressing, compact VOP1/VOP2/VOPC half-bit encoding, VOP3/VINTERP OPSEL wording, and 256/512 16-bit VGPR reach. |
| 7.5/7.7 8-bit packed math and VOP3P fields | Audited statically | Checked FP8/BF8 format basics, packed opcode inventory, generic VOP3P fields, MIX selector/ABS overloads, DOT4/DOT8 modifier rules, and scalar/inline constants; WMMA/SWMMAC details are tracked under 7.12. |
| 7.8 Dual Issue VALU (VOPD) | Audited statically | Checked VOPD field tables, literal forms, pair legality rules, source-cache rules, opcode tables, and paired-exception behavior against XML. |
| 7.9 Cross-Lane and Data Parallel Processing (DPP) | Audited statically | Checked DPP marker constants, DPP8/DPP16 field layouts, DPP_CTRL semantics, BC/FI behavior, VOPC/CMPX cases, OPSEL rules, source modifiers, and support-table coverage against XML. |
| 7.10/7.11 Pseudo-scalar transcendental and VGPR indexing | Audited statically | Checked pseudo-scalar execution/operand rules and relative VGPR-indexing formulas against XML. |
| 7.12 WMMA/SWMMAC | Audited statically | Checked instruction inventory, operand roles, sparse-index metadata, layout formulas, modifier rules, forced full-wave execution, and hazards. XML gaps for the prose-only layout, modifier, sparse-index, full-wave, and hazard rules are recorded below. |
| 8.1-8.5 Scalar memory encoding, addressing, buffer resources, narrow loads, S_DCACHE_INV, dependency checking, SMEM groups, alignment, bounds, and prefetch | Audited statically | Checked SMEM fields/opcodes, raw-pointer and buffer-resource addressing prose, SDATA/SBASE/SOFFSET operand metadata, 8/16-bit load descriptions, cache invalidation, `KMcnt` dependency rules, scalar-memory clause/group rules, alignment/range-check consequences, and scalar prefetch rules. Detailed SMEM instruction definitions are tracked in Chapter 16.6. |
| 9 intro / 9.1-9.6 Vector memory buffer instructions, data, addressing, alignment, and resource descriptors | Audited statically | Checked VBUFFER field layout, typed/untyped buffer inventory, TFE, VGPR address/data usage, data-format and D16/D16_HI rules, descriptor fields, range checks, swizzling, alignment, and unbound/resource-mismatch prose. Detailed VBUFFER instruction definitions are tracked in Chapter 16.16. |
| 10 intro / 10.1-10.10 Vector memory image instructions, TFE/LWE, D16/A16/G16, no-sampler/sampler address tables, VGPR usage, image resources, samplers, data formats, data dependencies, ray tracing, and PRT | Audited statically | Checked VIMAGE/VSAMPLE field layout, opcode-family inventory, TFE/LWE status returns, D16/A16/G16 packing, no-sampler versus sampler address tables, DMASK/data VGPR usage, image atomics, sample denormal policy, image-resource descriptor layout, sampler descriptor layout, buffer/image data-format enumeration, VMEM dependency-counter rules, BVH ray-tracing VADDR/descriptor/return rules, and partially resident texture enable/status rules. LDS BVH stack operations referenced by Chapter 10.9 are covered by the Chapter 12 LDS stack slice. |
| 11 intro / 11.1-11.6 Global, Scratch, and Flat address-space operations | Audited statically | Checked VFLAT/VGLOBAL/VSCRATCH field layout, conditional address modes, opcode-family inventory, ADDTID shape, flat aperture/prose semantics, scratch-base use, special atomic notes, cache-counter metadata, exact addressing formulas, scratch swizzle, flat LDS/hole behavior, memory-error policy, data VGPR layout, atomic return-data shape, D16/D16_HI behavior, block VGPR load/store mask/address/OOR rules, and WMMA transpose-load EXEC/wave/layout rules. |
| 12 intro / 12.1/12.1.1/12.1.2 Local data share overview, CU/WGP modes, and access methods | Audited statically | Checked LDS topology, bank/conflict prose, per-workgroup allocation limits, LDSbase/size state, CU/WGP mode references, and the indexed/direct/parameter access taxonomy. Instruction-subsection details are tracked in separate rows. |
| 12.2/12.2.1 LDS Parameter Loads | Audited statically | Checked `DS_PARAM_LOAD` availability, `LDS_PARAM_RDY`, M0/new-primitive/offset layout, LDS address derivation, quad lane fill, 16-bit attribute packing, destination out-of-range behavior, and EXPcnt/wait hazards. DS_DIRECT shared hazard details are included only where Chapter 12.2 explicitly groups them with parameter loads. |
| 12.3/12.3.1 VALU Parameter Interpolation | Audited statically | Checked `V_INTERP_*` field layout, fixed DPP8 lane selection, FMA/RTZ formulas, `WAIT_EXP`, OPSEL, clamp, forwarding restrictions, no-exception behavior, XML operand widths, and opcode coverage. |
| 12.4 LDS Direct Load | Audited statically | Checked `DS_DIRECT_LOAD` CU-only availability, M0 address/datatype format, whole-quad broadcast, sign/zero extension, and EXPcnt/shared VDSDIR hazard text. |
| 12.5/12.5.1 Data Share Indexed and Atomic Access | Audited statically | Checked VDS fields, indexed load/store and ADDTID address formulas, 2-address offset scaling, DScnt ordering, atomic return/no-return inventory, same-address atomic ordering, FP atomic denormal controls, and operand-class ambiguity around atomic constants. Lane-permute and stack sections are audited in separate rows. |
| 12.5.2 LDS Lane-permute Ops | Audited statically | Checked `DS_PERMUTE_B32`, `DS_BPERMUTE_B32`, `DS_BPERMUTE_FI_B32`, and `DS_SWIZZLE_B32` opcode/operand coverage, no-LDS-storage wording, lane mapping, wave32/wave64 index width, EXEC and disabled-lane rules, FI source-read exception, byte-index/offset behavior, and DS_SWIZZLE mode pseudocode from Chapter 16.15. DS stack/BVH is audited separately in the 12.5.3 row. |
| 12.5.3 DS Stack Operations for Ray Tracing | Audited statically | Checked `DS_BVH_STACK_PUSH4_POP1_RTN_B32`, `DS_BVH_STACK_PUSH8_POP1_RTN_B32`, and `DS_BVH_STACK_PUSH8_POP2_RTN_B64` opcode/operand coverage, packed stack-address state, stack-size and ray-tracing flags, DATA_VALID filtering, push/pop/invalidation behavior, B64 second-pop behavior, and older RDNA3/RDNA3.5 stack-op boundary. |
| 13.1-13.3 Float Memory Atomics | Audited statically | Checked fixed-RNE atomic add, denormal-control tables, NaN handling, minNum/maxNum signed-zero behavior, exception suppression, and float-add special cases against DS, buffer, flat/global/scratch, and image atomic XML records. |
| 14 Export: Position, Color/MRT | Audited statically | Checked `ENC_VEXPORT`, `EXPORT` operands, target enum metadata, `DONE`/`EN`/`ROW_EN`, 16-bit packing, pixel/primitive export obligations, dual-source blend lane masks, export-ready/skip-export behavior, and `EXPcnt` dependency prose. |
| 15 Microcode Formats | Audited statically | Checked Chapter 15 summary, reserved-zero rule, DPP/literal exclusivity, and raw field maps for scalar, VALU, VINTERP, VDSDIR, VDS, VBUFFER, VIMAGE, VSAMPLE, VFLAT/VGLOBAL/VSCRATCH, and VEXPORT encodings. Opcode-table semantics are tracked in the corresponding instruction-family rows. |
| 16.1 SOP2 Instructions | Audited statically | Checked the full SOP2 instruction-definition inventory after normalizing markdown conversion artifacts in the pack rows, literal-extension alternatives, literal-only `S_FMAAK_F32`/`S_FMAMK_F32`, scalar pack half-selection rows, and representative scalar FP/F16 rows against XML. |
| 16.2 SOPK Instructions | Audited statically | Checked the full SOPK instruction-definition inventory, ordinary no-literal forms, `S_SETREG_IMM32_B32` literal requirement, branch label operand, HWREG operands, and `S_VERSION` metadata fields against XML. |
| 16.3 SOP1 Instructions | Audited statically | Checked the normalized SOP1 instruction-definition inventory, markdown conversion artifact around `S_WQM_B32`, opcode holes, relative SGPR-indexing formulas, direct PC/trap/message/barrier/dynamic-VGPR/sleep/control rows, and scalar FP/F16 rows against XML. |
| 16.4 SOPC Instructions | Audited statically | Checked the full SOPC instruction-definition inventory, opcode holes, literal-extension alternatives, SCC outputs, integer compare formulas, bit-compare index masks, and floating compare predicates against XML. |
| 16.5 SOPP Instructions | Audited statically | Checked the full SOPP instruction-definition inventory after normalizing the malformed performance-counter/wait markdown table, opcode holes, wait/control/message/barrier operand fields, and reserved/compatibility opcode drift against XML. |
| 16.6 SMEM Instructions | Audited statically | Checked the full SMEM instruction-definition inventory, scalar-load widths and signedness, buffer-load forms, scalar-cache invalidation, scalar prefetch operands, and XML-only ATC probe drift. |
| 16.7 VOP2 Instructions | Audited statically | Checked the full VOP2 instruction-definition inventory, literal-only FMAMK/FMAAK forms, DPP/literal alternatives, implicit VCC operands, carry-in/out rows, packed-FMAC classification, and overlap with earlier VALU metadata gaps. |
| 16.8 VOP1 Instructions | Audited statically | Checked the full VOP1 instruction-definition inventory, VOP3 aliases, true16/FP8 selector wording, non-promotable forms, and representative special cases including readfirstlane, pipeflush, swap, permlane, and relative-indexed moves. |
| 16.9 VOPC Instructions | Audited statically | Checked the full VOPC instruction-definition inventory after normalizing markdown conversion artifacts, compact and VOP3 compare-result operands, compare-family expansion, class-compare masks, true16/DPP/literal variants, and overlap with earlier VALU metadata gaps. |
| 16.10 VOP3P Instructions | Audited statically | Checked the full VOP3P instruction-definition inventory after normalizing markdown conversion artifacts, opcode holes, DOT/MIX/packed-F16/WMMA/SWMMAC rows, DPP alternatives, VOP3P field metadata, and overlap with earlier packed-math and WMMA XML gaps. |
| 16.11 VOPD Instructions | Audited statically | Checked the full VOPD X/Y instruction-definition inventory, OPX/OPY asymmetry, Y-only integer/bitwise opcodes, literal forms, representative slot semantics, and overlap with earlier VOPD and VALU metadata gaps. |
| 16.12 VOP3 & VOP3SD Instructions | Audited statically | Checked the full VOP3/VOP3SD instruction-definition inventory after normalizing markdown conversion artifacts, VOP3SD opcode restriction, VOP3 compare/CMPX result operands, DPP/literal alternatives, modifier/OPSEL metadata, and overlap with earlier VALU metadata gaps. |
| 16.13 VINTERP Instructions | Audited statically | Checked the six detailed VINTERP instruction definitions, opcode/operand-width coverage, fixed DPP8 selector formulas, F16/RTZ OPSEL roles, and overlap with earlier Chapter 12.3 VINTERP XML gaps. |
| 16.14 Parameter and Direct Load from LDS Instructions | Audited statically | Checked `DS_PARAM_LOAD`/`DS_DIRECT_LOAD` definition inventory, VDSDIR opcode/alias coverage, raw field layout, M0/whole-quad/FP16-packing/direct-datatype wording, and overlap with earlier Chapter 12 LDS parameter/direct-load gaps. |
| 16.15 VDS Instructions | Audited statically | Checked the full VDS instruction-definition inventory after normalizing markdown conversion artifacts, opcode holes, 1-address/2-address/stride64 load-store forms, integer/FP/packed/special atomic rows, D16 and B96/B128 data movement, lane-permute/swizzle/BVH rows, and overlap with earlier Chapter 12 and Chapter 13 DS gaps. |
| 16.16 VBUFFER Instructions | Audited statically | Checked the full VBUFFER instruction-definition inventory, opcode holes, formatted/typed/TBUFFER and D16 forms, plain loads/stores, integer/FP/packed atomics, return-hint wording, and overlap with earlier Chapter 9 and Chapter 13 buffer gaps. |
| 16.17 VIMAGE Instructions | Audited statically | Checked the full VIMAGE instruction-definition inventory, opcode holes, image load/store/atomic rows, GET_RESINFO, BVH ray rows, packed/FP atomic rows, and overlap with earlier Chapter 10 and Chapter 13 image gaps. |
| 16.18 VSAMPLE Instructions | Audited statically | Checked the full VSAMPLE instruction-definition inventory, opcode holes, MSAA load, sample suffix families, G16 forms, gather variants, GATHER4H, GET_LOD, and overlap with earlier Chapter 10 sample/gather gaps. |
| 16.19 VEXPORT Instructions | Audited statically | Checked the singleton VEXPORT definition/prose, `EXPORT` XML entry, field/operand coverage, generated singleton decode/class path, and overlap with earlier Chapter 14 export gaps. |
| 16.20 FLAT, Scratch and Global Instructions | Audited statically | Checked the full VFLAT/VSCRATCH/VGLOBAL instruction-definition inventory, opcode holes, scalar/global/scratch address rows, D16 forms, block loads/stores, transpose loads, integer/FP/packed atomics, special atomic predicates, and overlap with earlier Chapter 11 and Chapter 13 gaps. |
| 7.6.1 Data Convert Instruction Types | Audited statically | Checked normal/single, packed, and stochastic conversion categories plus conversion-wide modifier, rounding, OMOD, CLAMP, `FP16_OVFL`, and OPSEL restrictions. FP8/BF8 byte placement and overflow details remain tracked in 7.6.2. |
| 7.6.2 OPSEL with FP8 and BF8 Conversions | Audited | Focused on `V_CVT_{PK,SR}_{FP8,BF8}_F32`. |
| RDNA4 manual sections through Chapter 16 | Complete | All manual chapters and instruction-definition sections have coverage rows above; detailed gaps and no-gap notes below record the manual-only/XML-drift findings and the narrow matches. |

## Gaps

### RDNA4-XML-001: `FP16_OVFL` mode state is absent from XML

Manual evidence:

- `rdna4/README.md:1064` starts the MODE register section.
- `rdna4/README.md:1074` defines `FP16_OVFL` as MODE bit 23.
- `rdna4/README.md:3166` says the flag applies to F32 to FP8/BF8 conversions.

XML evidence:

- Searching `amdgpu_isa_rdna4.xml` for `FP16_OVFL` returns no matches.
- The relevant instruction entries only describe operands and generic
  conversion text: `V_CVT_PK_FP8_F32` at `amdgpu_isa_rdna4.xml:119351`,
  `V_CVT_PK_BF8_F32` at `:119517`, `V_CVT_SR_FP8_F32` at `:119683`,
  and `V_CVT_SR_BF8_F32` at `:119849`.

Impact:

The XML cannot be used alone to discover that these conversions are
hardware-state-dependent, nor which MODE bit controls the alternate behavior.

### RDNA4-XML-002: FP8/BF8 overflow and Inf result table is prose-only

Manual evidence:

- `rdna4/README.md:3166` introduces the F32 to FP8/BF8 `FP16_OVFL` rule.
- `rdna4/README.md:3171` through `:3174` gives the table:
  NaN remains NaN; FP8 Inf/overflow is `max_E4M3` when `FP16_OVFL=1` and NaN
  when `FP16_OVFL=0`; BF8 Inf/overflow is `max_E5M2` when `FP16_OVFL=1` and
  Inf when `FP16_OVFL=0`.

XML evidence:

- `amdgpu_isa_rdna4.xml:119351` through `:119849` lists the four affected
  instructions, their data formats, and their source sizes.
- No XML entry in that range records the NaN/Inf/overflow table or the
  `FP16_OVFL=0` versus `FP16_OVFL=1` split.

Impact:

Generated semantics or tests based only on XML can silently hard-code one mode.

### RDNA4-XML-003: F8 mode numeric encodings are not captured

Manual evidence:

- `rdna4/README.md:3107` through `:3116` introduces FP8/BF8 formats and says
  numeric ranges depend on data format and bias selection.
- `rdna4/README.md:3118` through `:3124` gives the `F8_Mode` table for BF8 and
  FP8, including bias, signed-zero behavior, Inf/NaN encodings, max value, min
  normal, and min denormal.

XML evidence:

- `FMT_NUM_BF8` at `amdgpu_isa_rdna4.xml:174451` through `:174487` records an
  8-bit float with 1 sign bit, 5 exponent bits, and 2 mantissa bits, but has no
  description text or mode/range table.
- `FMT_NUM_FP8` at `amdgpu_isa_rdna4.xml:174603` through `:174625` records an
  8-bit float with 1 sign bit, 4 exponent bits, and 3 mantissa bits, but has no
  description text or mode/range table.

Impact:

The XML records the storage layout but not which numeric interpretation applies
under each `F8_Mode`.

### RDNA4-XML-004: FP8/BF8 conversion modifier restrictions are not machine-readable

Manual evidence:

- `rdna4/README.md:3128` through `:3135` lists common data-conversion
  restrictions, including unsupported input modifiers, OMOD, CLAMP for some
  conversions, and OPSEL restrictions.
- `rdna4/README.md:3164` states that these FP8/BF8 conversion instructions do
  not support ABS, NEG, OMOD, or CLAMP.

XML evidence:

- The generic VOP3 encoding exposes `ABS`, `CLAMP`, `NEG`, and `OMOD` fields at
  `amdgpu_isa_rdna4.xml:2518`, `:2528`, `:2548`, and `:2558`.
- Those field descriptions point to `OPF_NO*` lists, but the audited
  instruction entries for `V_CVT_{PK,SR}_{FP8,BF8}_F32` do not carry a
  per-instruction machine-readable restriction flag.

Impact:

XML consumers need a manual override to reject or ignore unsupported modifiers
for these conversion instructions.

### RDNA4-XML-005: `V_CVT_PK_*_F32` destination preservation is not modeled as an input

Manual evidence:

- `rdna4/README.md:3158` says `CVT_PK_FP8_F32` and `CVT_PK_BF8_F32` write
  either `DST[15:0]` or `DST[31:16]` based on `OPSEL[3]` and preserve the
  other half.
- `rdna4/README.md:22714` through `:22720` gives the same behavior in
  `V_CVT_PK_FP8_F32` pseudocode.
- `rdna4/README.md:22735` through `:22740` gives the same behavior for
  `V_CVT_PK_BF8_F32`.

XML evidence:

- For `V_CVT_PK_FP8_F32`, `VDST` is `Input="false" Output="true"` at
  `amdgpu_isa_rdna4.xml:119359` through `:119364`.
- For `V_CVT_PK_BF8_F32`, `VDST` is `Input="false" Output="true"` at
  `amdgpu_isa_rdna4.xml:119525` through `:119530`.
- By contrast, stochastic byte writes use `Input="true" Output="true"` for
  `VDST` at `amdgpu_isa_rdna4.xml:119691` through `:119696` and `:119857`
  through `:119862`.

Impact:

The XML under-specifies the read-modify-write dependency for packed FP8/BF8
converts. A consumer that trusts `VDST` input flags can miss the old-destination
dependency and preservation tests.

### RDNA4-XML-006: OPSEL byte/half placement details are prose-only

Manual evidence:

- `rdna4/README.md:3156` through `:3164` defines the special OPSEL mapping for
  FP8/BF8 conversions.
- `rdna4/README.md:22764` through `:22773` maps `V_CVT_SR_FP8_F32` OPSEL bits
  `[3:2]` to destination byte lanes and preserves the other bytes.
- `rdna4/README.md:22791` through `:22800` gives the same byte-lane behavior
  for `V_CVT_SR_BF8_F32`.

XML evidence:

- The instruction descriptions mention OPSEL for destination placement, but the
  instruction entries do not encode which OPSEL bits select which byte/half or
  which destination bits are preserved.

Impact:

XML consumers need manual overrides for exact sub-dword write placement.

### RDNA4-XML-007: `V_CVT_F32_{FP8,BF8}` byte OPSEL mapping is missing

Manual evidence:

- `rdna4/README.md:3162` says `V_CVT_F32_FP8` and `V_CVT_F32_BF8` select a
  byte from `SRC0` using `OPSEL[0:1]`.
- `rdna4/README.md:14280` through `:14290` gives the `V_CVT_F32_FP8`
  pseudocode: `0 -> [7:0]`, `2 -> [15:8]`, `1 -> [23:16]`, `3 -> [31:24]`.
- `rdna4/README.md:14303` through `:14313` gives the same reversed-bit byte
  mapping for `V_CVT_F32_BF8`.

XML evidence:

- The generic VOP3 `OPSEL` field at `amdgpu_isa_rdna4.xml:2584` through
  `:2585` describes 16-bit source and destination half selection.
- `V_CVT_F32_FP8` entries begin at `amdgpu_isa_rdna4.xml:70622`, and
  `V_CVT_F32_BF8` entries begin at `amdgpu_isa_rdna4.xml:70790`, but the
  entries do not encode this 8-bit reversed-byte mapping.

Impact:

The XML describes the operand as 8 bits, but not which byte is selected by each
OPSEL value in VOP3 form.

### RDNA4-XML-008: Stochastic rounding algorithm is prose-only

Manual evidence:

- `rdna4/README.md:3147` through `:3155` describes the stochastic conversion
  type at a high level.
- `rdna4/README.md:22756` through `:22763` defines the FP8 stochastic temporary
  value using `S1[31:12]`.
- `rdna4/README.md:22786` through `:22790` defines the BF8 stochastic temporary
  value using `S1[31:11]`.

XML evidence:

- The XML describes `SRC1` as `FMT_NUM_U32` for stochastic converts at
  `amdgpu_isa_rdna4.xml:119703` through `:119708` and `:119869` through
  `:119874`, but does not describe which seed bits are consumed.

Impact:

Seed handling cannot be generated or checked from XML alone.

### RDNA4-XML-009: HWREG bitfield layouts and state-register side effects are prose-only

Manual evidence:

- `rdna4/README.md:983` through `:1008` lists wave-state HWREG IDs.
- `rdna4/README.md:1010` through `:1074` defines `STATUS`, `STATE_PRIV`, and
  `MODE` bitfields, including `STATE_PRIV.SCC` and `MODE.FP16_OVFL`.
- `rdna4/README.md:1149` through `:1351` defines HW_ID, trap/exception, and
  shader-cycle register details.

XML evidence:

- `OPR_HWREG` at `amdgpu_isa_rdna4.xml:181768` through `:181886` defines only
  the `ID`, `OFFSET`, and `SIZE` encoding fields plus register IDs/descriptions.
- The XML does not provide machine-readable bit members for the wave-state
  registers, write masks, read-only fields, or the per-register side effects
  described in Chapter 3.

Impact:

XML consumers can decode `s_getreg_b32` / `s_setreg_b32` operands, but cannot
derive which register bits exist or which writes should be permitted.

### RDNA4-XML-010: `IB_STS2` HWREG ID is missing

Manual evidence:

- `rdna4/README.md:1008` lists HWREG ID 28 as `IB_STS2`.

XML evidence:

- `OPR_HWREG` at `amdgpu_isa_rdna4.xml:181783` through `:181877` enumerates
  MODE, STATUS, STATE_PRIV, allocation, perf snapshot, exception, scratch,
  HW_ID, and shader-cycle IDs, but has no predefined value for ID 28.

Impact:

An XML-driven decoder/disassembler can still decode the raw 6-bit ID, but it
cannot name `IB_STS2` or classify the register from XML alone.

### RDNA4-XML-011: PC width, alignment, and launch initialization are not captured

Manual evidence:

- `rdna4/README.md:701` through `:707` says PC is a DWORD-aligned 48-bit byte
  address, low two bits are forced to zero, wave launch initializes it to the
  first instruction, most PC-relative operations use the next-instruction PC,
  and `S_TRAP` saves the current instruction PC.

XML evidence:

- `OPR_PC` at `amdgpu_isa_rdna4.xml:181927` through `:181934` describes `pc` as
  `pc[63:0]`.
- `OPR_LABEL` at `amdgpu_isa_rdna4.xml:181909` through `:181921` records the
  branch target field relative to `PC+4`, but not the 48-bit width,
  forced-zero low bits, or trap-PC exception.

Impact:

PC state semantics need manual rules outside the XML operand model.

### RDNA4-XML-012: Wave32 `EXEC`/`VCC` active-lane semantics are not represented

Manual evidence:

- `rdna4/README.md:709` through `:715` says wave32 hardware acts only on
  `EXEC[31:0]` and `EXECZ` reflects only active wave32 lanes.
- `rdna4/README.md:1124` through `:1132` gives the same low-32-bit rule for
  `VCCZ` in wave32.

XML evidence:

- `OPR_EXEC` at `amdgpu_isa_rdna4.xml:181742` through `:181751` describes
  `exec` as a 64-bit vector execute mask.
- `OPR_VCC` at `amdgpu_isa_rdna4.xml:193992` similarly models `vcc` as a
  64-bit operand.

Impact:

XML consumers need ISA/wave-size policy to compute active-lane masks and
summary conditions correctly.

### RDNA4-XML-013: `EXEC==0` instruction skip rules are absent

Manual evidence:

- `rdna4/README.md:717` through `:742` defines detailed timing-visible skip and
  no-skip cases for VALU, VOPC, memory, LDS, export, WQM-like operations, and
  outstanding counters when `EXEC==0`.

XML evidence:

- XML instruction flags such as those around `S_GETREG_B32` at
  `amdgpu_isa_rdna4.xml:37510` encode branch/terminator/immediate-execution
  properties, but not the Chapter 3 per-family skip/nullification rules.

Impact:

An emulator or analyzer cannot derive correct zero-EXEC issue/nullification
behavior from XML alone.

### RDNA4-XML-014: SGPR alignment, region-crossing, out-of-range, and TTMP write rules are incomplete

Manual evidence:

- `rdna4/README.md:760` through `:775` defines SGPR alignment requirements.
- `rdna4/README.md:778` through `:816` defines scalar encoding regions,
  out-of-range read/write behavior, TTMP write privilege, failed-TTMP SCC
  preservation, SMEM result behavior, and `S_MOVREL` region rules.

XML evidence:

- Scalar operand classes beginning at `amdgpu_isa_rdna4.xml:181938` enumerate
  register names and include a generic alignment description for some scalar
  operands, but do not encode the region-crossing and behavioral consequences
  above.

Impact:

XML-driven operand classes are sufficient for naming many scalar operands, but
not for modeling architected illegal/out-of-range behavior.

### RDNA4-XML-015: VGPR out-of-range and dynamic-VGPR rules are incomplete

Manual evidence:

- `rdna4/README.md:820` through `:858` defines VGPR allocation and
  out-of-range behavior, including dest nullification, source substitution with
  VGPR0, VOPD modulo-4 source addressing, and multi-destination suppression.
- `rdna4/README.md:860` through `:899` defines dynamic-VGPR launch mode,
  wave32-only support, block sizes, max/min blocks, `S_ALLOC_VGPR` SCC
  success/failure, all-or-nothing allocation, and `DEALLOC_VGPRS` incompatibility.

XML evidence:

- `OPR_VGPR` at `amdgpu_isa_rdna4.xml:194073` records the VGPR namespace.
- `S_ALLOC_VGPR` at `amdgpu_isa_rdna4.xml:21643` through `:21680` describes
  the broad operation and implicit SCC output.
- `msg_dealloc_vgprs` at `amdgpu_isa_rdna4.xml:182657` says the message is
  incompatible with dynamic VGPR mode.
- The XML does not encode the rest of the Chapter 3 allocation, OOR, and
  retry-contract details.

Impact:

Dynamic VGPR and VGPR-OOR behavior require manual augmentation even though the
instruction and message names exist in XML.

### RDNA4-XML-016: Memory/LDS alignment and out-of-range behavior is prose-only

Manual evidence:

- `rdna4/README.md:901` through `:927` defines memory OOR, destination
  nullification, `SH_MEM_CONFIG.alignment_mode`, and atomic alignment behavior.
- `rdna4/README.md:929` through `:958` defines LDS allocation, MEMVIOL,
  per-DWORD OOR load zeroing, partial store behavior, and source/dest VGPR OOR
  handling.
- `rdna4/README.md:4090` through `:4094` introduces the same alignment-mode
  `MEMVIOL` rule for buffer-memory accesses, and `rdna4/README.md:4538`
  through `:4560` defines formatted-buffer alignment, non-formatted
  `SH_MEM_CONFIG.alignment_mode` behavior, and atomic alignment requirements.

XML evidence:

- DS and memory microcode fields exist, for example DS field descriptions near
  `amdgpu_isa_rdna4.xml:3432` and flat/global field descriptions near
  `amdgpu_isa_rdna4.xml:4567`.
- These field descriptions do not encode the Chapter 3 alignment-mode register
  dependency or per-lane OOR/nullification outcomes.
- VBUFFER fields include `IOFFSET`, `RSRC`, `FORMAT`, `OFFEN`, `IDXEN`,
  `SCOPE`, `TH`, and `TFE` at `amdgpu_isa_rdna4.xml:3649` through `:3745`,
  but no field or instruction metadata ties buffer accesses to the
  alignment-mode register or `MEMVIOL` outcomes.

Impact:

Memory correctness tests need manual-derived or hardware-derived expectations;
the XML does not carry enough semantic state.

### RDNA4-XML-017: Trap/exception/TBA/TMA semantics are only nominally represented

Manual evidence:

- `rdna4/README.md:1253` through `:1327` defines trap temporary privilege,
  trap enable behavior, TBA/TMA access, unmaskable exception behavior, and
  exception register fields.

XML evidence:

- `OPR_HWREG` includes exception and trap-control register IDs at
  `amdgpu_isa_rdna4.xml:181833` through `:181843`.
- `S_SENDMSG_RTN` message names for trap-related registers exist near
  `amdgpu_isa_rdna4.xml:182719`, but the Chapter 3 privilege and exception
  behavior is not machine-readable.

Impact:

Trap and debugger-visible state must be modeled from the manual, not generated
from XML alone.

### RDNA4-XML-018: Initial wave state is absent from XML

Manual evidence:

- `rdna4/README.md:1353` through `:1627` defines launch-time initialization for
  `EXEC`, `SCRATCH_BASE`, shader-stage SGPRs, compute `TTMP7`/`TTMP8`/`TTMP9`,
  compute packed work-item VGPR0, pixel-shader inputs, and PS LDS preload.

XML evidence:

- The RDNA4 XML has instruction/operand definitions, but no launch-state model
  for `SPI_PS_INPUT_*`, `COMPUTE_PGM_RSRC*`, AQL dispatch packet fields, or
  stage-specific initial SGPR/VGPR/LDS payloads.

Impact:

Launch-state emulation and debugger-visible state cannot be generated from XML.

### RDNA4-XML-019: Shader-cycle counter width conflicts with the manual

Manual evidence:

- `rdna4/README.md:1332` through `:1343` describes shader time as a 64-bit
  counter, read via low/high parts with wraparound-safe sequencing.

XML evidence:

- `hw_reg_shader_cycles_lo` and `hw_reg_shader_cycles_hi` descriptions at
  `amdgpu_isa_rdna4.xml:181868` through `:181882` call the counter 60 bits.

Impact:

The manual and XML disagree on counter width; consumers need an external
decision before modeling rollover precisely.

### RDNA4-XML-020: Inline-constant materialization rules are prose-only

Manual evidence:

- `rdna4/README.md:1646` through `:1652` defines inline constants and says
  float constants are not converted when used by non-float instructions.
- The same lines say 16-bit operations use the 16-bit float bits in the low
  half with zeros in the high half, and integer constants are sign-extended for
  64-bit sources.

XML evidence:

- The XML enumerates inline integer and float source selector values, including
  `src_0` through `src_64`, `src_neg_1` through `src_neg_16`, and the floating
  constants at `amdgpu_isa_rdna4.xml:184874` through `:185321`.
- No XML text in that enum range records non-float use, 16-bit zero-extension,
  or 64-bit integer sign-extension behavior.

Impact:

An XML-only consumer can recover the selector values, but not the source-size
dependent bit pattern that should be supplied to execution or disassembly.

### RDNA4-XML-021: Literal extension and placement rules are only partially captured

Manual evidence:

- `rdna4/README.md:1654` through `:1661` says a 32-bit literal can supply
  smaller or larger operands.
- It distinguishes signed 64-bit integer sign-extension, unsigned/binary
  64-bit zero-extension, double-float high-half placement with low zeros, and
  lower-width or packed operations that use the 32-bit data, typically LSBs.

XML evidence:

- `OPR_SIMM32` is described as a `32-bit integer constant` with a 32-bit
  `VALUE` field at `amdgpu_isa_rdna4.xml:182805` through `:182818`.
- Literal source enum descriptions contain some extension prose in the source
  selector tables, but they do not provide a complete per-operand rule for the
  F64 high-half case, signed 64-bit integer sign-extension, unsigned/binary
  64-bit zero-extension, and packed/short LSB use.

Impact:

Codegen needs operand-format and instruction-semantic classification outside
the raw `OPR_SIMM32` width to construct the literal operand correctly.

### RDNA4-XML-022: `NULL` destination instruction-nullification is incomplete

Manual evidence:

- `rdna4/README.md:1669` says `NULL` reads return zero, writes are ignored, and
  when used as a SALU destination it nullifies the instruction.

XML evidence:

- XML `null` descriptions say reads return zero and destinations are not
  written at `amdgpu_isa_rdna4.xml:182573` through `:182575` and
  `:184853` through `:184856`.
- Searching the XML for `nullifies` and `SALU destination` finds no matching
  rule.

Impact:

A model generated from XML alone may still execute non-destination side effects
for a SALU instruction whose destination selector should nullify the operation.

### RDNA4-XML-023: Aperture source selectors are named but semantically empty

Manual evidence:

- `rdna4/README.md:1682` through `:1685` identifies `SHARED_BASE`,
  `SHARED_LIMIT`, `PRIVATE_BASE`, and `PRIVATE_LIMIT` as memory aperture
  definitions.
- Chapter 6.10 says these source constants originate from `SH_MEM_BASES`,
  that zero shared/private bases disable the aperture, and defines the 64-bit
  `SHARED_BASE`, `SHARED_LIMIT`, `PRIVATE_BASE`, and `PRIVATE_LIMIT` formulas
  at `rdna4/README.md:2571` through `:2599`.

XML evidence:

- XML enumerates `src_shared_base`, `src_shared_limit`, `src_private_base`, and
  `src_private_limit` at `amdgpu_isa_rdna4.xml:185329` through `:185346`, but
  those entries have empty descriptions.

Impact:

The XML exposes the magic source values but not the state-register derivation,
64-bit base/limit formulas, or zero-base disable behavior needed to model
address-space aperture state.

### RDNA4-XML-024: SCOPE enum and cache-coherence behavior are missing

Manual evidence:

- `rdna4/README.md:1707` through `:1720` defines `SCOPE` as `CU`, `SE`, `DEV`,
  and `SYS`.
- `rdna4/README.md:1722` through `:1731` defines cache hit, write
  acknowledgement, read-modify-write locality, forwarding, and memory-pool
  dependent cache-scope behavior.

XML evidence:

- SMEM has a 2-bit `SCOPE` field at `amdgpu_isa_rdna4.xml:887` through `:893`.
- VMEM formats have equivalent 2-bit fields, for example VBUFFER at
  `amdgpu_isa_rdna4.xml:3719` through `:3725`.
- These field descriptions only say `Scope of memory operation`; the XML does
  not define the enum names or the cache/coherence algorithm.

Impact:

The XML records where the bits are, but cannot drive correct cache-scope or
coherence modeling without manual prose.

### RDNA4-XML-025: Acquire/release and writeback/invalidate policies are prose-only

Manual evidence:

- `rdna4/README.md:1733` through `:1752` defines release writeback and acquire
  invalidate rules based on `ISA.SCOPE > CACHE_SCOPE`.
- `rdna4/README.md:1796` through `:1803` gives the VMEM writeback/invalidate
  CU/L2 action matrix for CU, SE, DEV, and SYS scopes.

XML evidence:

- XML instruction descriptions name global cache operations, for example
  `GLOBAL_INV`, `GLOBAL_WB`, and `GLOBAL_WBINV` around
  `amdgpu_isa_rdna4.xml:54052` through `:54076` and `:55746` through `:55747`.
- The XML does not encode the `ISA.SCOPE > CACHE_SCOPE` decision rule or the
  CU/L2 policy matrix.

Impact:

Fence and cache-maintenance behavior needs manual policy data, not just opcode
and field decode.

### RDNA4-XML-026: TH load/store/atomic policy tables are missing

Manual evidence:

- `rdna4/README.md:1754` through `:1782` defines load/store TH policies such
  as `RT`, `NT`, `HT`, `LU`, `NT_RT`, `RT_NT`, `NT_HT`, `WB`, and `NT_WB`.
- `rdna4/README.md:1784` through `:1795` says atomic TH is split into return,
  NT, and cascade/deferred-scope bits.

XML evidence:

- SMEM has a 2-bit `TH` field at `amdgpu_isa_rdna4.xml:917` through `:923`.
- VMEM formats have 3-bit `TH` fields, for example VBUFFER at
  `amdgpu_isa_rdna4.xml:3749` through `:3755`.
- The shared XML description is only `Temporal hints and atomic return request`;
  searches for the TH policy names, `TH[0]`, and cascade terminology do not find
  a machine-readable table.

Impact:

Load/store retention policy, atomic return selection, and cascading atomic
scope behavior are not recoverable from XML alone.

### RDNA4-XML-027: SMEM-specific scope and TH caveats are missing

Manual evidence:

- `rdna4/README.md:1805` through `:1811` says SMEM uses the same 2-bit scope
  enum, but CU scope is coherent with scalar threads on the same CU, not vector
  threads.
- The same lines say SMEM uses only the 2-bit TH subset and does not support
  independent near-cache/far-cache hints; SMEM loads with `scope==CU` are not
  coherent with VMEM stores or atomics.

XML evidence:

- XML SMEM fields identify 2-bit `SCOPE` and 2-bit `TH` at
  `amdgpu_isa_rdna4.xml:887` through `:923`.
- The XML does not include the scalar-vs-vector CU coherence caveat or the SMEM
  TH subset rule.

Impact:

Generated memory modeling can accidentally treat SMEM and VMEM scope semantics
as interchangeable.

### RDNA4-XML-028: Common unused-field canonicalization is missing

Manual evidence:

- `rdna4/README.md:1642` says not every instruction uses every field in its
  encoding. Unused fields that can specify an SGPR source or destination are
  typically set to `NULL`; other unused fields are typically zero.

XML evidence:

- XML instruction encodings define the physical fields for each format, but no
  matching general rule for unused-field canonical values was found by searching
  the XML for `unused`, `set to zero`, `zero when unused`, or `nullifies`.

Impact:

An encoder or disassembler using only XML does not know the manual-preferred
canonical values for fields that are present in a format but not used by a
specific instruction.

### RDNA4-XML-029: Chapter 5 control-format coverage is incomplete

Manual evidence:

- `rdna4/README.md:1819` through `:1837` introduces Chapter 5 program-flow
  formats as `SOPP` and `SOP1` and describes `SIMM16` as a signed 16-bit
  integer constant.
- `rdna4/README.md:1885` lists `S_VERSION` in the Chapter 5 control-instruction
  table.

XML evidence:

- The XML `ENC_SOPP` format says `SIMM16` signedness is determined by opcode at
  `amdgpu_isa_rdna4.xml:459`, while the generic `OPR_SIMM16` operand type says
  only "integer constant" at `amdgpu_isa_rdna4.xml:182769`.
- `S_VERSION` is present, but uses `ENC_SOPK` at
  `amdgpu_isa_rdna4.xml:37370` through `:37382`, not one of the formats named
  by this Chapter 5 slice.

Impact:

Consumers need instruction-specific format knowledge beyond the local Chapter 5
format prose and beyond the generic `OPR_SIMM16` operand type.

### RDNA4-XML-030: Chapter 5 trap, return, and halt details are only partial

Manual evidence:

- `rdna4/README.md:1846` says `S_ENDPGM_SAVED` is intended only within a trap
  handler.
- `rdna4/README.md:1850` through `:1852` defines `S_TRAP` details: 4-bit
  `TrapID`, waiting for outstanding instructions, `{TTMP1,TTMP0}` save layout,
  `PC = TBA`, `PRIV = 1`, host-generated traps, saved-PC behavior, and
  reserved `TrapID 0`.
- `rdna4/README.md:1853` through `:1854` says `S_RFE_B64` returns from an
  exception by loading `PC`, clearing `STATUS.PRIV`, and may only be used within
  a trap handler.
- `rdna4/README.md:1855` through `:1862` defines `S_SETHALT` bit meanings,
  user-vs-trap privilege restrictions, delayed `HALT` behavior while `PRIV=1`,
  and `FATAL_HALT` behavior.

XML evidence:

- `S_TRAP` says only "Enter the trap handler" with a generic `OPR_SIMM16`
  operand at `amdgpu_isa_rdna4.xml:37923` through `:37937`.
- `S_RFE_B64` describes clearing `PRIV` and jumping to the source address at
  `amdgpu_isa_rdna4.xml:21427` through `:21443`, but omits the trap-handler
  restriction.
- `S_SETHALT` says only that it sets or clears `HALT` or `FATAL_HALT` with a
  generic `OPR_SIMM16` at `amdgpu_isa_rdna4.xml:37724` through `:37739`.
- `S_ENDPGM_SAVED` describes saved-context termination at
  `amdgpu_isa_rdna4.xml:38323` through `:38330`, but does not encode the trap
  handler-only intent.

Impact:

Trap handling, halt behavior, and privilege legality cannot be regenerated from
the XML instruction summaries alone.

### RDNA4-XML-031: NOP, sleep-var, and wakeup behavior is under-specified

Manual evidence:

- `rdna4/README.md:1868` says `S_NOP` repeats `SIMM16[6:0]` times and acts like
  a short `S_SLEEP`.
- `rdna4/README.md:1870` says `S_SLEEP_VAR` sleeps for approximately
  `SGPR_value[6:0] * 64` cycles and is woken by `S_WAKEUP`.
- `rdna4/README.md:1871` says `S_WAKEUP` wakes other waves in the same
  work-group from `S_SLEEP` or `S_SLEEP_VAR`, and leaves non-sleeping waves
  unaffected.

XML evidence:

- `S_NOP` has only a generic `OPR_SIMM16` operand and "small, fixed amount"
  delay description at `amdgpu_isa_rdna4.xml:37666` through `:37678`.
- `OPR_SLEEP` does encode `S_SLEEP` duration and sleep-forever fields at
  `amdgpu_isa_rdna4.xml:182859` through `:182892`, but `S_SLEEP_VAR` uses a
  generic `OPR_SSRC` at `amdgpu_isa_rdna4.xml:21695` through `:21710`.
- `S_WAKEUP` mentions waking waves from `S_SLEEP` at
  `amdgpu_isa_rdna4.xml:38345` through `:38355`, but does not name
  `S_SLEEP_VAR` or the unaffected-wave rule.

Impact:

Timing and scheduler tests need manual-derived behavior for the non-immediate
sleep forms, even though the immediate `S_SLEEP` operand type carries more
metadata.

### RDNA4-XML-032: `S_CLAUSE` length rules conflict with the manual

Manual evidence:

- `rdna4/README.md:1873` says the clause type comes from the instruction after
  `s_clause`, the length is `SIMM16[5:0] + 1`, and `SIMM16[5:0]` must be
  `1-32`, not `0` or `63`.

XML evidence:

- `S_CLAUSE` uses `OPR_CLAUSE` at `amdgpu_isa_rdna4.xml:37782` through
  `:37797`.
- `OPR_CLAUSE.LENGTH` says the programmed range is `[1, 62]` at
  `amdgpu_isa_rdna4.xml:181523` through `:181532`.
- The XML operand type describes length and `BREAK_SPAN`, but does not encode
  "type of instruction after `s_clause`".

Impact:

Manual and XML disagree on which `S_CLAUSE` lengths are legal; codegen cannot
pick the architectural range from XML alone.

### RDNA4-XML-033: Barrier state and work-group semantics are incomplete

Reported by: Dirac subreviewer; local audit.

Manual evidence:

- `rdna4/README.md:2019` through `:2021` defines work-groups as waves on the
  same WGP and splits the barrier operation into `S_BARRIER_SIGNAL` arrival
  followed by `S_BARRIER_WAIT`.
- `rdna4/README.md:2025` through `:2033` defines barrier validity after all
  work-group waves have been created, completion after each wave has signaled
  or terminated, early-termination behavior, and the single-wave/no-work-group
  `S_NOP` rule.
- `rdna4/README.md:2043` through `:2045` defines the work-group and trap
  barriers, including the trap-barrier rule that user-shader signal/wait
  operations are ignored.
- `rdna4/README.md:2051` through `:2061` defines `memberCount`,
  `signaledCount`, reset behavior, `barrierComplete`, `trapBarrierComplete`,
  and the relationship to `STATE_PRIV` and `S_GET_BARRIER_STATE`.
- `rdna4/README.md:2083` through `:2086` gives the split-barrier instruction
  table, including `ISFIRST` `SCC`, `KMcnt`, and the
  `S_GET_BARRIER_STATE` result layout
  `{ 0, signalCnt[6:0], 5'b0, memberCnt[6:0], 3'b0, valid }`.
- The detailed instruction definitions repeat the state changes:
  `S_BARRIER_SIGNAL` increments `signalCnt` and calls
  `CheckBarrierComplete` at `rdna4/README.md:10328` and `:10331`;
  `S_BARRIER_SIGNAL_ISFIRST` sets `SCC` from whether `signalCnt` was zero at
  `:10355`; `S_GET_BARRIER_STATE` packs `signalCnt`, `memberCnt`, and `valid`
  at `:10367` and `:10373`; and `S_BARRIER_WAIT` maps the immediate to
  `WAVE_BARRIER_COMPLETE` bits and clears the completion bit after waiting at
  `:11317` through `:11325`.

XML evidence:

- `S_BARRIER_SIGNAL` and `S_BARRIER_SIGNAL_ISFIRST` exist at
  `amdgpu_isa_rdna4.xml:21549` and `:21591`; the `ISFIRST` entry includes an
  implicit `SCC` output at `:21605` through `:21609`.
- `S_BARRIER_WAIT` exists at `amdgpu_isa_rdna4.xml:38010` through `:38025`.
- Exact searches for `S_GET_BARRIER_STATE`, `GET_BARRIER_STATE`, and
  `BARRIER_STATE` found no instruction entry in `amdgpu_isa_rdna4.xml`.
- `amdgpu_isa_gfx1250.xml` does contain `S_GET_BARRIER_STATE` at
  `:23277`, so this is specific to the checked-in generic RDNA4 XML, not to
  every GFX12-like XML input.
- `OPR_SSRC_BARRIER_ID` has an empty description at
  `amdgpu_isa_rdna4.xml:192740`; it lists numeric selectors for `-2`, `-1`,
  `0..31`, and `m0`, but not the special trap/work-group meanings, privilege
  rules, count semantics, or inline-only restrictions for those barriers.
- `hw_reg_wave_state_priv` is present and records readable-by-all /
  writable-when-`PRIV=1` at `amdgpu_isa_rdna4.xml:181793` through `:181799`,
  but the actual `barrierComplete` and `trapBarrierComplete` bit layout is not
  described in XML.

Impact:

The XML captures some barrier names and operands, but not the full work-group
contract or barrier state layout, and generic RDNA4 XML omits a manual- and
LLVM-accepted `S_GET_BARRIER_STATE` opcode entirely.

### RDNA4-XML-034: `S_SETPRIO` priority formula is absent

Manual evidence:

- `rdna4/README.md:1872` says `S_SETPRIO` writes 2 bits of `USER_PRIO`, defines
  `0` as low and `3` as high, and gives the combined priority formula using
  `SysPrio` and `WaveAge`.

XML evidence:

- `S_SETPRIO` says only "Change wave user priority" and uses generic
  `OPR_SIMM16` at `amdgpu_isa_rdna4.xml:38368` through `:38383`.

Impact:

A scheduler model cannot derive the priority field layout or formula from XML.

### RDNA4-XML-035: Send-message encoding and wait behavior are partial

Reported by: local audit; Goodall subreviewer.

Manual evidence:

- `rdna4/README.md:1887` says `S_SENDMSG` uses `SIMM[9:0]` as the message type
  and has no enforced `S_WAIT_*CNT` before it.
- `rdna4/README.md:1888` says `S_SENDMSG_RTN_B32/B64` uses `KMcnt`, returns to
  an SGPR or aligned SGPR pair, takes an enum operand rather than an SGPR code,
  and leaves `VCCZ` undefined if writing `VCC`.
- `rdna4/README.md:1959` says `S_SENDMSG_RTN_B*` encodes the message type in
  the `SSRC0` instruction field, not an SGPR, and carries any payload in `M0`.
- `rdna4/README.md:1963` through `:1965` says return messages increment
  `KMcnt` by 2 and then decrement once when sent and once when data returns.

XML evidence:

- `OPR_SENDMSG` partitions the operand as a 7-bit `MSG`, bit-7 `HAS_RTN`, and
  bit-8 `SYSTEM` at `amdgpu_isa_rdna4.xml:182622` through `:182676`.
- `OPR_SENDMSG_RTN` has the analogous return-message partition and message
  enum values at `amdgpu_isa_rdna4.xml:182684` through `:182760`.
- The instruction entries for `S_SENDMSG_RTN_B32/B64` at
  `amdgpu_isa_rdna4.xml:21479` and `:21514` do not encode `KMcnt` tracking or
  the `VCCZ` undefined side effect.
- `S_WAIT_KMCNT` exists at `amdgpu_isa_rdna4.xml:38787`, but the XML has no
  producer-side rule connecting `S_SENDMSG_RTN_B32/B64` to that counter.

Impact:

The XML is useful for message enum decoding, but not sufficient for wait-counter
or side-effect modeling, and its field partition does not match the manual's
coarser `SIMM[9:0]` wording.

### RDNA4-XML-036: `S_ICACHE_INV` scope wording differs from the manual

Manual evidence:

- `rdna4/README.md:1890` says `S_ICACHE_INV` invalidates the first-level shader
  instruction cache for the WGP associated with the issuing wave.

XML evidence:

- The XML description says "Invalidate entire first-level instruction cache" at
  `amdgpu_isa_rdna4.xml:38590` through `:38600`, with no WGP association.

Impact:

Cache-invalidation scope is not explicit enough in XML for a precise WGP-level
instruction-cache model.

### RDNA4-XML-037: Instruction-clause type membership is prose-only

Manual evidence:

- `rdna4/README.md:1901` through `:1905` says a clause is an uninterrupted
  series of instructions of one type, with the clause type implicitly defined
  by the instruction immediately after `S_CLAUSE`.
- `rdna4/README.md:1907` through `:1917` lists the valid clause types:
  non-flat loads, non-flat stores, non-flat atomics, flat load/store/atomic,
  LDS indexed load/store/atomic plus `BVH_stack`, SMEM, and VALU.
- `rdna4/README.md:1921` through `:1932` lists instruction classes that are
  illegal in a clause and instructions that are legal in any clause but not as
  the first instruction.

XML evidence:

- `S_CLAUSE` says only "Mark the beginning of a clause" at
  `amdgpu_isa_rdna4.xml:37782` through `:37797`.
- `OPR_CLAUSE` describes `BREAK_SPAN` and `LENGTH` fields at
  `amdgpu_isa_rdna4.xml:181513` through `:181536`.
- Searches for the manual clause-type names and rule phrases, including
  `Non-Flat`, `Flat Load`, `BVH_stack`, `illegal within a clause`, and
  `clause type`, found no machine-readable clause membership table in
  `amdgpu_isa_rdna4.xml`.
- `DS_PARAM_LOAD` and `DS_DIRECT_LOAD` are ordinary XML instructions near
  `amdgpu_isa_rdna4.xml:49933` and `:49982`, with no marker for the manual's
  illegal-in-clause rule.
- FP64 VALU instructions such as `V_ADD_F64` and `V_MUL_F64` carry normal
  `FMT_NUM_F64` VALU definitions near `amdgpu_isa_rdna4.xml:71464` and
  `:72428`, with no clause-exclusion metadata.

Impact:

An XML-only validator cannot decide which following instruction starts a legal
clause type, nor whether later instructions remain within that type.

### RDNA4-XML-038: Clause start, skip, and break behavior is prose-only

Manual evidence:

- `rdna4/README.md:1933` says `S_TRAP` is legal within a clause, including as
  the first instruction after `S_CLAUSE`, and ends the clause.
- `rdna4/README.md:1935` says pseudo-scalar `V_S_*` instructions are VALU ops
  and may be used in a VALU clause.
- `rdna4/README.md:1937` defines the VALU-clause `EXEC==0` behavior at clause
  start and mid-clause.
- `rdna4/README.md:1939` through `:1947` says `S_DELAY_ALU` must precede
  `S_CLAUSE` when used before a clause, because the first instruction after
  `S_CLAUSE` declares the clause type.
- `rdna4/README.md:1947` says if the first instruction after `S_CLAUSE` is
  skipped, no clause starts and later instructions execute individually.
- `rdna4/README.md:1951` through `:1955` lists clause-break causes: VALU
  exceptions, host wave commands, context save, halts, kills, and trap-handler
  entry.

XML evidence:

- XML has individual instruction entries for `S_TRAP`, `S_DELAY_ALU`, and
  `S_CLAUSE`, but no relationship tying their relative instruction-stream
  positions to clause start/end behavior.
- `OPR_DELAY` captures `S_DELAY_ALU` hazard fields at
  `amdgpu_isa_rdna4.xml:181541` and following, but does not encode the
  `S_CLAUSE` ordering rule.
- Searches for `EXEC==0`, `first instruction`, `break a clause`, and
  `S_DELAY_ALU` near clause prose found no XML clause control-flow rule.

Impact:

Clause correctness requires manual-derived instruction-stream state, not just
the `S_CLAUSE` immediate field.

### RDNA4-XML-039: Return-message payload metadata and table source column are incomplete

Reported by: Goodall subreviewer; local audit.

Manual evidence:

- `rdna4/README.md:1959` says `S_SENDMSG_RTN_B*` encodes the message type in
  `SSRC0`, carries any payload in `M0`, and writes the destination SGPR named by
  `SDST`.
- `rdna4/README.md:1979` repeats that the return-message code is encoded in the
  `SSRC0` instruction field, not read from an SGPR.
- The return-message table header at `rdna4/README.md:1981` still labels the
  code column as `SIMM16[7:0]`, which conflicts with the surrounding prose.

XML evidence:

- `S_SENDMSG_RTN_B32` and `S_SENDMSG_RTN_B64` use `SSRC0` with
  `OPR_SENDMSG_RTN` at `amdgpu_isa_rdna4.xml:21493` through `:21498` and
  `:21528` through `:21533`, matching the prose rather than the table header.
- Unlike `S_SENDMSG`, which declares implicit `M0` at
  `amdgpu_isa_rdna4.xml:38412` through `:38415`, the return-message instruction
  entries list only `SDST` and `SSRC0`.

Impact:

An XML-only consumer cannot discover `M0` payload use for return messages, while
a manual-table scraper can also be misled by the `SIMM16` header despite the
prose and XML using `SSRC0`.

### RDNA4-XML-040: Send-message legal-code and payload/result details are incomplete or inconsistent

Reported by: Goodall subreviewer; local audit.

Manual evidence:

- `rdna4/README.md:1967` says all unlisted `S_SENDMSG` codes are reserved and
  illegal, and `rdna4/README.md:1971` lists code `0x00` as reserved.
- `rdna4/README.md:1973` says `HS_TESSFACTOR` uses `M0[0]`: `0` means all zero
  tessellation factors, and `1` means all one.
- `rdna4/README.md:1987` says user shaders sending `RTN_SAVE_WAVE` are converted
  to `MSG_ILLEGAL_RTN`.
- `rdna4/README.md:1989` says `RTN_GET_SE_HW_ID` returns `SE_ID` in
  `data[3:0]` and `AID_ID` in `data[11:8]`.

XML evidence:

- `OPR_SENDMSG.MSG` lists values `1`, `2`, `3`, and `9` at
  `amdgpu_isa_rdna4.xml:182645` through `:182668`, but has no explicit reserved
  `0x00` entry and no machine-readable "all other codes illegal" rule.
- The XML `msg_hs_tessfactor` description at
  `amdgpu_isa_rdna4.xml:182651` through `:182654` collapses `M0[0]` to
  `1 = "all are zero or one"`, losing the manual's `0` versus `1` distinction.
- `msg_rtn_save_wave` at `amdgpu_isa_rdna4.xml:182728` through `:182731` carries
  the trap-handler restriction, but names the user-shader conversion target as
  `MSG_RTN_ILLEGAL_MSG`, not the manual's `MSG_ILLEGAL_RTN`.
- `msg_rtn_get_se_hw_id` at `amdgpu_isa_rdna4.xml:182743` through `:182746`
  says it gets `SE_ID` and `AID_ID`, but does not encode the result bit
  positions.
- `msg_rtn_get_tba_to_pc` appears as internal hardware-only value `0x86` at
  `amdgpu_isa_rdna4.xml:182738` through `:182741`, while the scoped manual
  Table 22 skips from `0x85` to `0x87`.

Impact:

The XML enum tables are a useful starting point, but cannot be treated as a
complete legality or payload/result specification for send-message validation
or emulation.

### RDNA4-XML-041: Branch whole-wave and condition semantics are only descriptive

Reported by: Helmholtz subreviewer; local audit.

Manual evidence:

- `rdna4/README.md:1996` says branch instructions affect the entire wave, not
  individual work-items.
- `rdna4/README.md:2009` maps conditional branches to `SCC==1`, `SCC==0`,
  `VCC==0`, `VCC!=0`, `EXEC==0`, and `EXEC!=0`.
- `rdna4/README.md:2015` says scalar compares set `SCC`, while vector compares
  set `VCC`, which then feeds `VCCZ` / `VCCNZ` branch conditions.

XML evidence:

- Branch instruction entries expose `IsBranch` / `IsConditionalBranch`, for
  example `S_BRANCH` at `amdgpu_isa_rdna4.xml:38054` through `:38061` and
  `S_CBRANCH_SCC0` at `:38083` through `:38091`.
- `S_CBRANCH_VCCZ` and `S_CBRANCH_EXECZ` descriptions name the summary
  conditions at `amdgpu_isa_rdna4.xml:38161` through `:38162` and `:38231`
  through `:38232`, and their implicit operands point at `OPR_VCC` / `EXEC` at
  `:38175` through `:38179` and `:38245` through `:38249`.
- The branch entries do not encode a machine-readable predicate such as
  "branch when the active-lane VCC mask is zero" or "branch controls the whole
  wave." Chapter 3 wave32 `EXEC`/`VCC` active-lane semantics are already tracked
  separately in `RDNA4-XML-012`.

Impact:

XML consumers can identify branch opcodes and their implicit condition
registers, but still need manual-derived rules for whole-wave control flow and
the exact zero/nonzero condition predicates.

### RDNA4-XML-042: Branch PC-base descriptions mix current-PC and next-PC wording

Reported by: Helmholtz subreviewer; local audit.

Manual evidence:

- `rdna4/README.md:2002` says `S_BRANCH`, `S_CBRANCH`, `S_SWAPPC`, `S_CALL`, and
  `S_GETPC` operate on the PC when it points to the instruction after the
  current one, so `S_BRANCH 0` is a NOP.
- `rdna4/README.md:2012` says `S_GETPC_B64` retrieves the current PC value,
  zero-extended, where the table context defines that value as the next
  instruction PC.

XML evidence:

- `OPR_LABEL.TARGET` says branch labels are signed dword targets relative to
  `PC+4` at `amdgpu_isa_rdna4.xml:181913` through `:181920`.
- `S_GETPC_B64` says it stores the address of the next instruction at
  `amdgpu_isa_rdna4.xml:21314` through `:21315`.
- `S_CALL_B64`, `S_BRANCH`, and `S_CBRANCH_SCC0` descriptions say "relative to
  the current PC" at `amdgpu_isa_rdna4.xml:37621` through `:37622`, `:38061`
  through `:38062`, and `:38091` through `:38092`, relying on the operand type
  to recover the next-PC base.
- `OPR_PC` only describes `pc[63:0]` at `amdgpu_isa_rdna4.xml:181927` through
  `:181934`; it does not state the `S_GETPC_B64` zero-extension detail.

Impact:

The XML operand type carries the important `PC+4` label rule, but free-text
instruction descriptions can mislead consumers that do not also apply
`OPR_LABEL` and manual-specific `S_GETPC_B64` rules.

### RDNA4-XML-043: Split-barrier operand legality is inconsistent across sources

Reported by: Dirac subreviewer; local audit.

Manual evidence:

- Chapter 5.6 says barrier instructions can use `M0` or an inline constant for
  the barrier number, except `S_BARRIER_WAIT` can only take an inline constant
  at `rdna4/README.md:2065` through `:2067`.
- The same section says references to the work-group and trap barriers can only
  be made with inline constants at `rdna4/README.md:2067`.
- The detailed `S_BARRIER_SIGNAL` and `S_BARRIER_SIGNAL_ISFIRST` definitions
  instead say support for `M0` is reserved for other architectures at
  `rdna4/README.md:10307` and `:10338`, while their pseudocode still includes
  an `IsM0(SRC0)` path.

XML and oracle evidence:

- RDNA4 XML uses `OPR_SSRC_BARRIER_ID` for `S_BARRIER_SIGNAL` and
  `S_BARRIER_SIGNAL_ISFIRST` source operands at
  `amdgpu_isa_rdna4.xml:21560` and `:21602`, and that operand type includes an
  `m0` predefined value at `amdgpu_isa_rdna4.xml:192917`.
- RDNA4 XML also declares `SOP1_INST_LITERAL` alternatives for the two signal
  instructions at `amdgpu_isa_rdna4.xml:21566` through `:21573` and `:21613`
  through `:21620`.
- In contrast, `amdgpu_isa_gfx1250.xml:23208` through `:23264` has only the
  base `ENC_SOP1` forms for the same signal instructions.
- A local LLVM oracle check with
  `llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding` accepted
  `s_barrier_signal m0`, `s_get_barrier_state s0, m0`, and rejected
  `s_barrier_wait m0`; the same command rejected arbitrary 32-bit literal
  forms such as `s_barrier_signal 0x12345678`.

Impact:

Consumers of RDNA4 XML cannot determine the legal split-barrier source forms
without consulting the manual and an assembler oracle. The XML also appears to
expose literal-extension forms for the signal instructions that LLVM rejects for
`gfx1200`.

### RDNA4-XML-044: Memory dependency counter producer and ordering semantics are mostly prose-only

Reported by: local audit; Ramanujan subreviewer.

Manual evidence:

- `rdna4/README.md:2107` says each wave has per-wave counters that increment
  when instructions issue, decrement when instructions complete, and put the
  wave effectively asleep while `S_WAIT_*CNT` waits for a threshold.
- `rdna4/README.md:2167` through `:2175` gives same-type ordering, SMEM
  out-of-order behavior, counter-overflow issue stalls, ordered decrement for
  load completion, and the rule that counters count instructions rather than
  threads.
- `rdna4/README.md:2181` through `:2209` classifies producer instructions and
  decrement points for `LOADcnt`, `SAMPLEcnt`, `BVHcnt`, `STOREcnt`, `DScnt`,
  `KMcnt`, and `EXPcnt`.
- `rdna4/README.md:2213` through `:2215` says scalar memory uses `KMcnt`,
  larger SMEM operations increment by two, scalar loads can return out of order
  even to the same address, and scalar-cache invalidates require
  `S_WAIT_KMCNT <= 0`.
- `rdna4/README.md:3962` through `:3970` restates the scalar-memory-specific
  dependency contract: SMEM loads can return out of order or partially across
  cache lines, `KMcnt` increments by one for a single-dword fetch or cache
  invalidate, increments by two for two-or-more-dword fetches, decrements by an
  equal amount when the instruction completes, and cache invalidates are not
  known complete until `KMcnt == 0`.
- `rdna4/README.md:4088` says vector-memory buffer loads and stores can
  complete out of order with respect to each other when they are different
  memory-operation types.
- `rdna4/README.md:4612` repeats the out-of-order completion rule for image
  memory operation types, including loads, stores, and samples.
- `rdna4/README.md:2231` through `:2243` says every Flat instruction
  increments both `DScnt` and `LOADcnt`/`STOREcnt`, and that its LDS and global
  halves complete independently with complementary `EXEC` masks.
- `rdna4/README.md:2253` through `:2255` defines export `EXPcnt` WAR hazards,
  export-family ordering, and the late `EXEC` read hazard.
- `rdna4/README.md:2259` says `S_SENDMSG` uses `KMcnt` until the message has
  been sent out of the WGP.
- `rdna4/README.md:2148` says `S_WAIT_EVENT` bit 1 waits for export-ready,
  other bits are reserved, and exceptions wait for this event before being
  processed.

XML evidence:

- The split wait instruction entries encode the wait opcodes and short
  threshold descriptions at `amdgpu_isa_rdna4.xml:38613` through `:38865`.
- `OPR_WAIT_MEM_DS` encodes combined wait `DS` bits at offset 0 and `MEM` bits
  at offset 8 at `amdgpu_isa_rdna4.xml:195495` through `:195518`.
- `OPR_WAIT_EVENT` captures only the `EXPORT_READY` field at
  `amdgpu_isa_rdna4.xml:195475` through `:195490`; it does not encode reserved
  bits or exception ordering.
- XML instruction entries for producer classes such as `S_LOAD_B32`,
  `DS_PARAM_LOAD`, and `EXPORT` describe operands and broad instruction
  behavior, but searches around `amdgpu_isa_rdna4.xml:15969`,
  `:49933`, and `:50025` found no machine-readable counter
  increment/decrement timing or cross-counter ordering contract.
- SMEM entries expose operand size and opcode information, for example
  `S_LOAD_B64` at `amdgpu_isa_rdna4.xml:16018` through `:16055`,
  `S_LOAD_B512` at `:16165` through `:16205`, and `S_DCACHE_INV` at
  `:16919` through `:16930`, while `S_WAIT_KMCNT` exposes only the
  `SIMM16[4:0]` threshold at `:38787` through `:38805`; none of these entries
  ties SMEM producer width, invalidates, out-of-order completion, or partial
  returns to `KMcnt`.
- Global cache-maintenance instructions are a narrow exception: XML annotates
  `GLOBAL_INV` with `LOAD_CNT` and `GLOBAL_WB` / `GLOBAL_WBINV` with
  `STORE_CNT` near `amdgpu_isa_rdna4.xml:54052`, `:54075`, and `:55746`, but
  does not encode completion scope or ordered wait behavior.

Impact:

Generated semantics based on XML can decode the wait instructions, but cannot
derive the hardware dependency contract: which ordinary instructions produce
which counters, when those counters decrement, which waits are sensible, how
Flat and export hazards split, or how events interact with exceptions.

### RDNA4-XML-045: VOP3P DOT modifier semantics are instruction-specific prose

Reported by: Popper subreviewer; local audit.

Manual evidence:

- The packed-math VOP3P field table gives generic `NEG`, `NEG_HI`, `OPSEL`,
  `OPSEL_HI`, and `CM` descriptions at `rdna4/README.md:3199` through
  `:3208`.
- The manual then gives instruction-specific overrides: `V_WMMA...IU...` and
  `V_DOT4...IU...` use `NEG[1:0]` as signedness bits and require
  `NEG_HI=0` at `rdna4/README.md:3224` through `:3227`.
- `DOT4_F32_{FP8,BF8}_{FP8,BF8}` requires `OPSEL=0`, `OPSEL_HI=7`, allows
  `NEG`/`NEG_HI` only on `SRC2`, treats `NEG_HI` as `ABS`, disallows
  `OMOD`/`CLAMP`, uses round-to-nearest-even, and reports no exceptions at
  `rdna4/README.md:3228` through `:3230`.

XML evidence:

- The XML VOP3P format exposes only generic field behavior for `CLAMP`, `NEG`,
  `NEG_HI`, `OPSEL`, and `OPSEL_HI` at `amdgpu_isa_rdna4.xml:13839`,
  `:13859`, `:13869`, `:13889`, and `:13899`.
- The FP8/BF8 DOT4 instruction entries distinguish operand formats and opcodes,
  but do not encode these modifier requirements:
  `V_DOT4_F32_FP8_BF8` at `amdgpu_isa_rdna4.xml:125593` through `:125623`,
  `V_DOT4_F32_BF8_FP8` at `:125857` through `:125887`,
  `V_DOT4_F32_FP8_FP8` at `:126121` through `:126151`, and
  `V_DOT4_F32_BF8_BF8` at `:126385` through `:126415`.

Impact:

XML consumers cannot derive the DOT-specific signedness, selector, clamp, ABS,
rounding, or exception contract without manual overrides.

### RDNA4-XML-046: Packed-math inline-constant behavior is prose-only

Reported by: Popper subreviewer; local audit.

Manual evidence:

- Chapter 7.7.2 says inline constants have packed-math-specific OPSEL behavior
  at `rdna4/README.md:3253` through `:3257`.
- It then says packed math instructions, excluding WMMA, with source float data
  sizes smaller than 16 bits do not work with inline constants at
  `rdna4/README.md:3259`.
- The same table gives DOT integer exceptions where 8-bit and 4-bit integer
  inline constants work and OPSEL is ignored for inline source data at
  `rdna4/README.md:3263` through `:3268`.

XML evidence:

- The affected DOT operands use the broad `OPR_SRC` source class, for example
  `V_DOT4_F32_FP8_BF8` uses it for `SRC0`, `SRC1`, and `SRC2` at
  `amdgpu_isa_rdna4.xml:125607` through `:125623`.
- `OPR_SRC` is defined as all scalar or vector sources at
  `amdgpu_isa_rdna4.xml:184216` through `:184225`, includes inline constants
  beginning at `:184874`, and includes the 32-bit literal selector at
  `:186629` through `:186635`.

Impact:

The XML source class admits inline/literal forms but does not say which packed
math opcodes ignore OPSEL for inline data, which require OPSEL to broadcast a
16-bit inline value, or which sub-16-bit floating-point DOT forms do not work
with inline constants at all.

### RDNA4-XML-047: WMMA/SWMMAC whole-wave and sparse-index behavior is not encoded

Reported by: Popper subreviewer; local audit.

Manual evidence:

- Chapter 7.12 states that each WMMA or SWMMAC operates on one matrix striped
  across all lanes, not one matrix per lane, at `rdna4/README.md:3528`.
- The FP8/BF8 WMMA modifier restriction says there is no OPSEL, ABS, NEG,
  OMOD, DPP, `FP16_OVFL`, or clamp support; A/B must be VGPRs and C may be a
  VGPR or inline constant at `rdna4/README.md:3232` through `:3234`.
- Sparse index VGPRs contain two wave32 or four wave64 index sets selected by
  `OPSEL[0]` or `OPSEL[1:0]` at `rdna4/README.md:3822`.

XML evidence:

- Dense FP8/BF8 WMMA entries record operand classes and sizes, such as
  `V_WMMA_F32_16X16X16_FP8_FP8` at `amdgpu_isa_rdna4.xml:126937` through
  `:126974`, but do not encode the whole-wave execution contract or modifier
  restrictions.
- FP8/BF8 SWMMAC entries record the sparse-index source as
  `FMT_WMMA_INDEX_SET`, for example at `amdgpu_isa_rdna4.xml:127506` through
  `:127543` and `:127553` through `:127590`, but do not encode the wave32/64
  OPSEL index-set selection rule.
- `FMT_WMMA_INDEX_SET` encodes sixteen two-bit components and the
  `idx0 < idx1` rule at `amdgpu_isa_rdna4.xml:181155` through `:181180`, but
  not how those index sets are selected from the VGPR.

Impact:

An XML-only consumer can decode the operand shapes but cannot reconstruct how
WMMA/SWMMAC consumes a whole wave, which modifier bits are illegal or reused,
or how sparse index VGPRs are partitioned by wave size and OPSEL.

### RDNA4-XML-048: SWMMAC is not grouped with WMMA despite shared hazard rules

Reported by: Popper subreviewer; local audit.

Manual evidence:

- Chapter 7.12.1 defines the hazard table's `WMMA` term to mean either WMMA or
  SWMMAC at `rdna4/README.md:3585` through `:3587`.
- The following hazard rows require waits or stalls between WMMA/SWMMAC uses of
  overlapping matrix registers at `rdna4/README.md:3593` through `:3599`.

XML evidence:

- Dense FP8/BF8 WMMA entries have functional group `VALU` and subgroup `WMMA`,
  for example `amdgpu_isa_rdna4.xml:126972` through `:126974`.
- FP8/BF8 SWMMAC entries have functional group `VALU` but no `WMMA` subgroup at
  `amdgpu_isa_rdna4.xml:127541` through `:127543`, `:127588` through `:127590`,
  `:127635` through `:127637`, and `:127682` through `:127684`.

Impact:

XML consumers that use subgroups to derive hazard handling can miss that SWMMAC
must participate in the same manual hazard rules as dense WMMA.

### RDNA4-XML-049: WMMA/SWMMAC matrix layout formulas are prose-only

Reported by: Euclid subreviewer; local audit.

Manual evidence:

- Chapter 7.12.2 gives the row/column to lane/VGPR/start-position procedure at
  `rdna4/README.md:3601` through `:3617`.
- The manual gives wave32 and wave64 A/B/C/D layout formulas at
  `rdna4/README.md:3619` through `:3658`.
- It records wave-size-specific VGPR usage for dense and sparse matrix shapes at
  `rdna4/README.md:3675` through `:3689`.
- Dense result layouts are given as explicit wave32 and wave64 formulas at
  `rdna4/README.md:3743` through `:3774`.

XML evidence:

- WMMA operand entries point to data formats such as
  `FMT_WMMA_DC_16X16_F32` and `FMT_WMMA_AB_16X16_F16`; for example
  `V_WMMA_F32_16X16X16_F16` lists those operands at
  `amdgpu_isa_rdna4.xml:126649` through `:126680`.
- SWMMAC entries similarly record operand formats and sizes, for example
  `V_SWMMAC_F32_16X16X32_F16` at `amdgpu_isa_rdna4.xml:127177` through
  `:127207`.
- The DC data-format description says the total number of VGPRs depends on wave
  size at `amdgpu_isa_rdna4.xml:180319` through `:180327`, but the format data
  records packed component bitmaps rather than the manual's lane/VGPR formulas.

Impact:

The XML has enough shape information to name the operands, but not enough to
derive correct per-lane register locations, wave32/wave64 VGPR counts, or
matrix-result layouts without manual rules.

### RDNA4-XML-050: WMMA modifier bit meanings and zero requirements are prose-only

Reported by: Euclid subreviewer; local audit.

Manual evidence:

- For IU4/IU8 WMMA, `NEG[1:0]` encodes A/B signedness and `NEG[2]` plus all
  `NEG_HI` bits must be zero at `rdna4/README.md:3577` through `:3579`.
- For F16/BF16 WMMA, `NEG[1:0]` and `NEG_HI[1:0]` apply to A/B low/high halves,
  while `{NEG_HI[2], NEG[2]}` acts as `{ABS, NEG}` on `SRC2`, at
  `rdna4/README.md:3581`.
- For FP8/BF8 WMMA, `NEG` must be zero for A and B at
  `rdna4/README.md:3583`, and the packed-math subsection separately says those
  forms do not support OPSEL, ABS, NEG, OMOD, DPP, `FP16_OVFL`, or clamp at
  `rdna4/README.md:3232` through `:3234`.

XML evidence:

- The VOP3P format has only generic `NEG` and `NEG_HI` field descriptions at
  `amdgpu_isa_rdna4.xml:2945` through `:2956` and generic OPSEL prose at
  `:2975` through `:2976`.
- WMMA instruction entries record operand formats and opcodes, but do not carry
  per-instruction bit meanings or zero/unsupported-modifier requirements.

Impact:

XML consumers cannot derive which WMMA modifier bits are signedness selectors,
which bits are source modifiers, and which encodings should be rejected or
ignored for each WMMA data type.

### RDNA4-XML-051: WMMA/SWMMAC forced full-wave execution and hazards are prose-only

Reported by: Euclid subreviewer; local audit.

Manual evidence:

- Chapter 3 says zero-`EXEC` VALU skipping does not apply to WMMA or SWMMAC at
  `rdna4/README.md:717` through `:742`.
- Dense WMMA pseudocode saves `EXEC`, sets it to all ones, evaluates the matrix
  operation, then restores `EXEC` at `rdna4/README.md:17333` through `:17342`.
- SWMMAC pseudocode does the same full-wave `EXEC` sequence at
  `rdna4/README.md:17565` through `:17570`.
- Chapter 7.12.1 lists required `V_NOP` or independent VALU spacing and stall
  cases for WMMA/SWMMAC hazards at `rdna4/README.md:3585` through `:3599`.

XML evidence:

- WMMA and SWMMAC entries describe the high-level matrix operation and list
  operands, for example `amdgpu_isa_rdna4.xml:126649` through `:126687` and
  `:127177` through `:127214`.
- Those entries do not encode the forced full-wave `EXEC` behavior, zero-EXEC
  exception, required inter-instruction spacing, or stall cases.

Impact:

An XML-only simulator or validator cannot derive when WMMA/SWMMAC must issue
despite `EXEC==0`, which lanes participate in the operation, or which adjacent
matrix instructions require scheduling barriers for correct hardware behavior.

### RDNA4-XML-052: VOPD literal-length conditions omit `SRCY0 == 255`

Reported by: Tesla subreviewer; local audit.

Manual evidence:

- Chapter 15.3.7 says VOPD can be followed by a 32-bit literal constant at
  `rdna4/README.md:7535`.
- The VOPD source table gives `SRCX0 == 255` as a literal constant and says
  `SRCY0` uses the same enumeration at `rdna4/README.md:7577` through `:7583`.
- Chapter 7.8 also permits at most one literal constant, or a shared literal,
  for the dual-op pair at `rdna4/README.md:3300`.

XML evidence:

- The 64-bit `VOPDXY` encoding condition checks only `SRCX0 != 255` at
  `amdgpu_isa_rdna4.xml:15518`.
- The 96-bit `VOPDXY_INST_LITERAL` conditions likewise key only on
  `SRCX0 == 255` / `SRCX0 != 255` at `amdgpu_isa_rdna4.xml:15746` and
  `:15782`.
- No corresponding condition ties the 96-bit form to `SRCY0 == 255`.

Impact:

An XML-derived decoder can mis-size a valid VOPD pair whose Y operation is the
only one using a literal source, treating a 96-bit instruction as 64 bits.

### RDNA4-XML-053: VOPD asymmetric dependency rule is lost

Reported by: Tesla subreviewer; local audit.

Manual evidence:

- Chapter 7.8 says the X operation must not overwrite sources of the Y
  operation, while the Y operation may overwrite sources of the X operation
  without creating a hazard, at `rdna4/README.md:3302`.

XML evidence:

- The VOPD description says only that the two instructions must be independent
  at `amdgpu_isa_rdna4.xml:15575`, duplicated for the literal form at
  `:15839`.
- The XML does not preserve the one-way exception that permits Y to overwrite
  X sources.

Impact:

XML-only validation cannot distinguish the legal Y-overwrites-X-source case from
the illegal X-overwrites-Y-source case.

### RDNA4-XML-054: VOPD source-bank rule omits the same-VGPR exception

Reported by: Tesla subreviewer; local audit.

Manual evidence:

- Chapter 7.8 allows `SRC0x`/`SRC0y` or `SRC1x`/`SRC1y` to use the same bank
  when they are the same VGPR and same-sized operand, at
  `rdna4/README.md:3290`.

XML evidence:

- The XML VOPD prose describes the source-cache ports and says a bank cannot
  read two `SRC0` or two `SRC1/2` operands at once, at
  `amdgpu_isa_rdna4.xml:15553`.
- The same-VGPR/same-sized exception from the manual is not included in that
  XML description.

Impact:

A validator built from the XML prose can over-reject legal VOPD pairs that reuse
the exact same VGPR source.

### RDNA4-XML-055: OPY `MOV` source-port rule is ambiguous between manual and XML

Reported by: Tesla subreviewer; local audit.

Manual evidence:

- Chapter 7.8 scopes the special `SRC2` read-port behavior to `V_MOV_B32` in
  OPY when OPX is also `V_MOV_B32`, at `rdna4/README.md:3296`.

XML evidence:

- The XML VOPD description says unconditionally that OPY `MOV` uses the `SRC2`
  read port instead of `SRC0`, at `amdgpu_isa_rdna4.xml:15561`.

Impact:

The manual and XML can drive different source-bank legality decisions for OPY
`MOV` paired with a non-`MOV` OPX instruction.

### RDNA4-XML-056: VOPD paired-exception coalescing is missing from XML

Reported by: Tesla subreviewer; local audit.

Manual evidence:

- Chapter 7.8 says VOPD instruction pairs generate only one exception if either
  or both operations raise an exception, at `rdna4/README.md:3350`.

XML evidence:

- The VOPD description block at `amdgpu_isa_rdna4.xml:15534` through `:15585`
  does not include paired-exception behavior.
- Searching the RDNA4 XML for the manual's "single exception" wording returns
  no match.

Impact:

An XML-only semantic model cannot infer whether paired VALU exceptions should be
coalesced or counted per slot.

### RDNA4-XML-057: DPP16 `DPP_CTRL` lane semantics are not machine-readable

Reported by: Newton subreviewer; local audit.

Manual evidence:

- Chapter 7.9 says DPP is specified through `DPP8`, `DPP8FI`, or `DPP16`
  marker constants and that the actual source VGPR comes from the DPP DWORD at
  `rdna4/README.md:3374`.
- Chapter 7.9.1 lists the valid `DPP_CTRL` ranges for quad permute, row shift,
  row rotate, mirror, half mirror, row share, and row xmask at
  `rdna4/README.md:3444` through `:3454`.
- Chapter 15.3.8 gives the same enumeration plus lane formulas at
  `rdna4/README.md:7639` through `:7649`.

XML evidence:

- The DPP16 format checks `SRC0 == 250` at `amdgpu_isa_rdna4.xml:6240`
  through `:6249`.
- The XML `DPP_CTRL` field is only a 9-bit "Data-parallel primitive control"
  field at `amdgpu_isa_rdna4.xml:6283` through `:6288`, repeated for VOP3P at
  `:14058` through `:14064`.
- Searching the RDNA4 XML for the manual enum names such as `DPP_QUAD`,
  `DPP_ROW`, `ROW_SHARE`, or `ROW_XMASK` returns no matches.

Impact:

The XML can identify the raw DPP16 control field but cannot validate reserved
values or derive the cross-lane source mapping without manual prose.

### RDNA4-XML-058: DPP row/bank mask write semantics are incomplete

Reported by: Newton subreviewer; local audit.

Manual evidence:

- Chapter 7.9.1 says `row_mask` and `bank_mask` apply only to the VGPR
  destination write, do not affect source VGPR fetch, and give VOPC disabled
  lanes a zero SGPR/VCC bit at `rdna4/README.md:3434` through `:3435`.
- The same tables distinguish wave32 and wave64 lane groups at
  `rdna4/README.md:3434` through `:3435`.
- Chapter 15.3.8 repeats that row/bank masks do not impact the source fetch at
  `rdna4/README.md:7632` through `:7633`.

XML evidence:

- The XML `BANK_MASK` and `ROW_MASK` fields say only that lane groups are
  enabled at `amdgpu_isa_rdna4.xml:6263` through `:6264` and `:6323` through
  `:6324`, repeated for VOP3P at `:14028` through `:14029` and `:14148`
  through `:14149`.
- The XML text does not record the destination-only rule, the VOPC zero-result
  rule, or the wave32-specific interpretation of upper row bits.

Impact:

An XML-only model can incorrectly mask source reads, preserve masked VOPC bits,
or apply wave64 row groups in wave32 mode.

### RDNA4-XML-059: DPP16 `BC` and `FI` interaction table is absent

Reported by: Newton subreviewer; local audit.

Manual evidence:

- Chapter 7.9.1 defines `BC` and `FI` at `rdna4/README.md:3441` through
  `:3442`.
- Table 39 gives the four-way result for out-of-range, in-range disabled, and
  active source lanes at `rdna4/README.md:3456` through `:3465`.

XML evidence:

- The XML has separate short field descriptions for `BOUND_CTRL` and `FI` at
  `amdgpu_isa_rdna4.xml:6273` through `:6274` and `:6303` through `:6304`,
  repeated for VOP3P at `:14038` through `:14039` and `:14078` through
  `:14079`.
- The XML does not encode the combined table, including the difference between
  disabled source lanes and out-of-range source lanes.

Impact:

Consumers need the manual to distinguish zero-source, no-write, and normal-read
cases for DPP16 invalid lanes.

### RDNA4-XML-060: `V_CMP`/`V_CMPX` DPP zeroing rules are missing

Reported by: Newton subreviewer; local audit.

Manual evidence:

- Chapter 7.9.1 says VOPC masked lanes receive zero bits at
  `rdna4/README.md:3434` through `:3435`.
- It also says DPP with `V_CMP` or `V_CMPX` and `bound_ctrl=0` writes zero for
  lanes whose EXEC mask bit is zero, and that `FI=1` does not turn on an
  inactive lane for `V_CMPX`, at `rdna4/README.md:3467`.

XML evidence:

- The XML contains compact and VOP3 compare instruction entries plus DPP
  encodings, for example `V_CMPX_LT_F16` around
  `amdgpu_isa_rdna4.xml:146270`, but the DPP format field descriptions at
  `:6263` through `:6324` and `:14028` through `:14149` do not capture the
  compare-specific zeroing behavior.

Impact:

XML-derived compare-DPP semantics can preserve or re-enable bits that the manual
says must be zero.

### RDNA4-XML-061: DPP8FI behavior is encoded only as a source sentinel

Reported by: Newton subreviewer; local audit.

Manual evidence:

- Chapter 7.9.2 distinguishes normal DPP8 from DPP8FI: DPP8 reads zero from
  inactive source lanes, while DPP8FI fetches data from inactive lanes at
  `rdna4/README.md:3471`.
- The same paragraph says DPP8 follows DPP16 `BC = 1` behavior and assumes
  source lanes are in range at `rdna4/README.md:3473`.

XML evidence:

- The DPP8 condition recognizes `SRC0 == 233` and `SRC0 == 234` at
  `amdgpu_isa_rdna4.xml:6500` through `:6547`.
- The DPP8 bitmap then exposes only `LANE_SEL_0` and subsequent lane selector
  fields plus the real source VGPR at `amdgpu_isa_rdna4.xml:6571` and nearby
  field entries; there is no structured `FI` or `BC` field in the DPP8 format.

Impact:

If an XML consumer normalizes both sentinel values to a DPP8 encoding without
retaining the original `SRC0` constant, DPP8FI collapses into ordinary DPP8.

### RDNA4-XML-062: DPP-specific OPSEL legality is missing

Reported by: Newton subreviewer; local audit.

Manual evidence:

- Chapter 7.9.1 says DPP with VOP3/VOP3P requires OPSEL to keep low results
  using low inputs and high results using high inputs at `rdna4/README.md:3423`.
- Chapter 7.9.2 repeats the same rule for DPP8 at `rdna4/README.md:3473`.

XML evidence:

- The XML VOP3 and VOP3P DPP encodings carry generic OPSEL descriptions, for
  example VOP3P `OPSEL` at `amdgpu_isa_rdna4.xml:14118` through `:14119` and
  `OPSEL_HI` at `:14128` through `:14129`.
- Those field descriptions do not encode the DPP-only cross-field legality
  constraint.

Impact:

An XML-only validator can accept DPP half-selection combinations that the manual
requires hardware code to avoid.

### RDNA4-XML-063: DPP16 source-modifier ownership is prose-only

Reported by: Newton subreviewer; local audit.

Manual evidence:

- Chapter 7.9.1 says DPP16 has `SRC*_IMOD` ABS/NEG controls and that VOP3 uses
  VOP3 ABS/NEG while VOP3P uses `NEG`/`NEG_HI`; the DPP16 ABS/NEG bits are
  ignored for those encodings at `rdna4/README.md:3425` through `:3428`.
- Chapter 15.3.8 lists the DPP16 `SRC0_NEG`, `SRC0_ABS`, `SRC1_NEG`, and
  `SRC1_ABS` bits at `rdna4/README.md:7628` through `:7631`.

XML evidence:

- The XML exposes the DPP16 source-modifier fields, and each description says
  the field is ignored for VOP3/VOP3P, for example
  `amdgpu_isa_rdna4.xml:6343` through `:6344` and `:14168` through `:14169`.
- The override is encoded as free-form field prose, not as a machine-readable
  rule tied to VOP1/VOP2/VOPC versus VOP3/VOP3P execution.

Impact:

Generated consumers must parse or hard-code the manual prose to know when DPP16
suffix source modifiers apply and when base VOP3/VOP3P modifiers replace them.

### RDNA4-XML-064: VOP3P DPP allow-list is ambiguous for DOT2

Reported by: Newton subreviewer; local audit.

Manual evidence:

- Chapter 7.9 Table 38 says VOP3P `V_DOT4_I32_IU8`, `V_DOT4_U32_U8`,
  `V_DOT8_I32_IU4`, `V_DOT8_U32_U4`, `V_PK_*`, and WMMA have `NO DPP`, while
  `V_FMA_MIX*` has `Allow DPP`, at `rdna4/README.md:3399` through `:3400`.

XML evidence:

- The XML gives DPP encodings to `V_DOT2_F32_F16` at
  `amdgpu_isa_rdna4.xml:122682` through `:122685`.
- The XML gives DPP encodings to `V_DOT2_F32_BF16` at
  `amdgpu_isa_rdna4.xml:124070` through `:124073`.
- `V_FMA_MIX_F32` is the explicitly manual-allowed shape and starts at
  `amdgpu_isa_rdna4.xml:124615`.

Impact:

If Table 38 is intended as the exhaustive VOP3P DPP allow-list, the XML
over-accepts DOT2 DPP forms. If DOT2 is intended to be legal, the manual table
is incomplete and needs a prose erratum.

### RDNA4-XML-065: Pseudo-scalar single-lane and `EXEC` override is absent

Reported by: Descartes subreviewer; local audit.

Manual evidence:

- Chapter 7.10 says `V_S_*` pseudo-scalar transcendental ops operate on a
  single lane of data with both source and destination in SGPRs, at
  `rdna4/README.md:3492`.
- The same notes state that `EXEC` is ignored and the instructions execute
  even when `EXEC==0`, at `rdna4/README.md:3507`.

XML evidence:

- The ten affected instruction entries exist from `V_S_EXP_F32` through
  `V_S_SQRT_F16` at `amdgpu_isa_rdna4.xml:109686` through `:110181`.
- The entries expose ordinary `InstructionFlags`, with `IsImmediatelyExecuted`
  set to `FALSE` in the representative `V_S_EXP_F32` entry at
  `amdgpu_isa_rdna4.xml:109680` through `:109685`.
- The XML records `FunctionalGroup VALU` / `Subgroup TRANSCENDENTAL`, for
  example at `amdgpu_isa_rdna4.xml:109728` through `:109731`, but has no
  machine-readable bit for the pseudo-scalar single-lane or `EXEC`-ignore
  contract.

Impact:

A generator using the XML alone sees ordinary VOP3/VALU transcendental
instructions and has to learn the `EXEC==0` exception from manual prose.

### RDNA4-XML-066: Pseudo-scalar destination and F16 half rules are not encoded

Reported by: Descartes subreviewer; local audit.

Manual evidence:

- Chapter 7.10 says half-SGPRs are not supported for 16-bit data, input data is
  expected in bits `[15:0]`, and the full 32-bit result is written with the
  upper half zeroed, at `rdna4/README.md:3504`.
- The same notes say `VCC` may not be used as a destination and `OPSEL[3]`
  must be zero, at `rdna4/README.md:3508` through `:3509`.
- The destination is still specified in `VDST`, not `SDST`, at
  `rdna4/README.md:3510`.

XML evidence:

- Representative `V_S_EXP_F32` and `V_S_EXP_F16` entries model `VDST` as
  `OPR_SREG`, at `amdgpu_isa_rdna4.xml:109694` through `:109698` and
  `:109749` through `:109753`.
- `OPR_SREG` is described as any scalar GPR operand including `VCC` and `NULL`
  at `amdgpu_isa_rdna4.xml:189665` through `:189667`, and lists `vcc_lo` /
  `vcc_hi` at `:190285` through `:190292`.
- `V_S_EXP_F16` uses `FMT_NUM_PK2_F16` for `VDST`, while that data format is a
  packed two-component 32-bit value at `amdgpu_isa_rdna4.xml:175048` through
  `:175052`.
- The generic VOP3 `OPSEL` field says `OPSEL[3]` selects low or high 16-bit
  destination half and preserves the unwritten bits at
  `amdgpu_isa_rdna4.xml:2584` through `:2585`.

Impact:

The XML over-accepts `VCC` destinations and presents the F16 destination like a
generic packed half destination, losing the manual's lower-half-only,
upper-zeroed pseudo-scalar rule.

### RDNA4-XML-067: Pseudo-scalar FP mode and exception behavior is prose-only

Reported by: Descartes subreviewer; local audit.

Manual evidence:

- Chapter 7.10 says these ops use the usual `DENORMAL` and `ROUND` mode bits at
  `rdna4/README.md:3505`.
- The same notes say they produce exceptions like their VALU equivalents at
  `rdna4/README.md:3506`.

XML evidence:

- XML has separate `S_ROUND_MODE` and `S_DENORM_MODE` instruction entries at
  `amdgpu_isa_rdna4.xml:37952` and `:37981`.
- The `V_S_*` instruction entries do not link to those mode bits or to the
  exception behavior of their VALU equivalents; they carry only ordinary
  operand, opcode, and functional-group metadata.

Impact:

Generated semantics cannot infer the pseudo-scalar FP environment or exception
model without a manual-derived side table.

### RDNA4-XML-068: VGPR indexing `M0` bit slices and unsigned index rules are missing

Reported by: Descartes subreviewer; local audit.

Manual evidence:

- Chapter 7.11 says the indexing instructions use a value in `M0` and that
  indices are unsigned, at `rdna4/README.md:3514`.
- The table assigns `M0[31:0]` to `V_MOVRELD_B32`, `V_MOVRELS_B32`, and
  `V_MOVRELSD_B32`, and assigns source `M0[9:0]` plus destination
  `M0[25:16]` to `V_MOVRELSD_2_B32` and `V_SWAPREL_B32`, at
  `rdna4/README.md:3520` through `:3524`.

XML evidence:

- The instruction entries only expose an implicit, non-partitioned `M0`
  operand through `OPR_SDST_M0`, for example
  `amdgpu_isa_rdna4.xml:66279` through `:66283`.
- `OPR_SDST_M0` describes `M0` as a special register for relative indices at
  `amdgpu_isa_rdna4.xml:182610` through `:182617`, but carries no bit-slice or
  unsigned-index metadata.

Impact:

XML consumers cannot distinguish full-width relative indexing from the split
source/destination index forms without hand-coding the manual table.

### RDNA4-XML-069: VGPR indexed move/swap equations and OOR consequences are not represented

Reported by: Descartes subreviewer; local audit.

Manual evidence:

- Chapter 7.11 gives exact assignments for `V_MOVRELD_B32`, `V_MOVRELS_B32`,
  `V_MOVRELSD_B32`, `V_MOVRELSD_2_B32`, and `V_SWAPREL_B32`, at
  `rdna4/README.md:3520` through `:3524`.
- Chapter 3.3.2.2 defines indexed operand out-of-range conditions, including
  `Index > 255` and `(V + M0) >= VGPR_SIZE`, at `rdna4/README.md:842` through
  `:846`.
- The same section says destination VGPR out of range treats the instruction as
  a NOP, `V_SWAPREL` discards if either destination is out of range, and VALU
  source VGPR out of range acts as source VGPR0, at `rdna4/README.md:850`
  through `:855`.

XML evidence:

- The five instruction entries are present at `amdgpu_isa_rdna4.xml:65731`,
  `:65939`, `:66099`, `:66259`, and `:70078`.
- Their descriptions are broad relative-index summaries, for example
  `V_MOVRELSD_2_B32` says it uses different offsets at
  `amdgpu_isa_rdna4.xml:66259` through `:66260`, and `V_SWAPREL_B32` says it
  swaps two relatively-indexed vector registers at `:70078` through `:70079`.
- The XML does not encode the assignment formulas, the swap order, or the
  Chapter 3 out-of-range consequences for indexed sources and destinations.

Impact:

A decoder generator can discover the opcodes and operands, but not the
architectural behavior needed to emulate relative VGPR indexing or invalid
indices.

### RDNA4-XML-070: VOP3SD `SDST` field metadata conflicts with the manual

Reported by: Herschel subreviewer; local audit.

Manual evidence:

- Chapter 7.1 defines VOP3SD as a 64-bit VALU encoding with three inputs and a
  scalar destination at `rdna4/README.md:2623`.
- The VOP3 variant text says VOP3SD has `SDST` instead of `OPSEL` and `ABS`,
  is used only for carry-out add/sub, div-scale, and mad-carry instructions,
  and is not used for `V_CMP*`, at `rdna4/README.md:2643` through `:2649`.
- The generic field table describes `SDST` as the scalar result destination,
  disallows `M0` and `EXEC`, and permits `NULL`, at
  `rdna4/README.md:2665`.

XML evidence:

- The `VOP3_SDST_ENC` format field calls `SDST` the "Destination operand for
  compare result" and exposes a 7-bit field at
  `amdgpu_isa_rdna4.xml:10870` through `:10875`.
- Instruction entries then use `OPR_SREG` for concrete VOP3SD operands, for
  example `V_ADD_CO_CI_U32` at `amdgpu_isa_rdna4.xml:78315` through `:78319`.
- `OPR_SREG` itself excludes `M0` and `EXEC` and includes `null`, at
  `amdgpu_isa_rdna4.xml:189665` through `:190299`, so the operand enum is
  narrower than the misleading format-field prose.

Impact:

The XML carries enough concrete operand data for many VOP3SD instructions, but
the microcode-format metadata does not match the manual's VOP3SD description.
Consumers using only the format field cannot distinguish VOP3SD scalar outputs
from compare-result destinations, and must special-case the physical field
width versus the manual's generic `SDST` field table.

### RDNA4-XML-071: Generic VOP3 modifier legality is not structured

Reported by: Herschel subreviewer; local audit.

Manual evidence:

- Chapter 7.1 says VOP3 provides `NEG`, `ABS`, and `OMOD` for floating-point
  only, at `rdna4/README.md:2631` through `:2636`.
- The field table gives the `OMOD` numeric mapping, restricts `NEG` and `ABS`
  to floating-point inputs, and defines `CM` compare-signal and clamp behavior,
  including `-0` clamping to `+0`, at `rdna4/README.md:2666` through `:2672`.
- Chapter 7.2.2.1 and 7.2.3.1 add opcode-specific allow/deny lists: input
  modifiers are forbidden for `READLANE`, `WRITELANE`, integer/bitwise,
  `PERMLANE`, `SWAP`, some DOT/CVT families, and output modifiers/clamp are
  ignored or unsupported for another opcode table, at `rdna4/README.md:2764`
  through `:2886`.
- Chapter 7.2.3.1 also says `CLAMP==1` clamps any NaN result to zero,
  exceptions are reported before clamp, and `OMOD!=0` has specific
  denorm-flush and exception-reporting behavior, at `rdna4/README.md:2841`
  through `:2868`.

XML evidence:

- The generic VOP3 fields expose `ABS`, `CLAMP`, `NEG`, and `OMOD` as raw
  fields with prose descriptions at `amdgpu_isa_rdna4.xml:2518`,
  `:2528`, `:2548`, and `:2558`.
- The `OMOD` field description references `OMOD_OFF` but does not encode the
  manual's `0/1/2/3` mapping as enum data at `amdgpu_isa_rdna4.xml:2558`
  through `:2565`.
- The `ABS`/`CLAMP`/`NEG`/`OMOD` descriptions reference `OPF_NOABS`,
  `OPF_NOCLAMP`, `OPF_NONEG`, and `OPF_NOOMOD`, but searching the XML shows
  those names only in prose references rather than as opcode-level tables.
- The `CLAMP` field description omits the manual's NaN-to-zero,
  exception-ordering, qNaN signaling/non-signaling split, and `-0` to `+0`
  clamp details.

Impact:

XML consumers can decode the raw bits but cannot derive opcode-specific
modifier legality, the complete output-modifier enum, or all clamp/compare
corner semantics without re-reading the manual.

### RDNA4-XML-072: Generic VOP3 `OPSEL` legality and overloads are prose-only

Reported by: Herschel subreviewer; local audit.

Manual evidence:

- Chapter 7.1 defines `OPSEL[0:3]` for 16-bit source and destination half
  selection, says `OPSEL` must be zero for non-16-bit operands/results, gives
  `V_PERMLANE*` overloads, restricts `DOT2_F16`/`DOT2_BF16`, permits SGPR
  half selection, and warns that later operations may overload `OPSEL`, at
  `rdna4/README.md:2673`.
- Chapter 7.2.2.3 gives the explicit opcode allow-list for ordinary VOP3
  `OPSEL` use, at `rdna4/README.md:2803` through `:2829`.

XML evidence:

- The generic VOP3 `OPSEL` field describes only basic 16-bit high/low source
  and destination half selection with destination preservation, at
  `amdgpu_isa_rdna4.xml:2584` through `:2585`.

Impact:

The XML gives the bit location and ordinary half-selection meaning, but not the
format-wide legality rule, the Chapter 7.2 opcode allow-list, or the opcode
families whose `OPSEL` bits have different meanings. This is the root cause
behind several narrower DOT, DPP, pseudo-scalar, and WMMA selector gaps recorded
elsewhere.

### RDNA4-XML-073: VOP3P `larger VGPR range` purpose is absent

Reported by: Herschel subreviewer; local audit.

Manual evidence:

- The Chapter 7.1 encoding table defines VOP3P as a 64-bit VALU format for
  instructions using packed math or requiring a larger VGPR range, at
  `rdna4/README.md:2624`.
- The following VOP3P paragraph emphasizes packed math at
  `rdna4/README.md:2653`.

XML evidence:

- `ENC_VOP3P` describes only packed math instructions that perform two 16-bit
  operations in one instruction, at `amdgpu_isa_rdna4.xml:2916` through
  `:2921`.

Impact:

The XML format description can mislead generators or audits into treating VOP3P
as exclusively packed 16-bit math. Non-packed uses that are present for register
range or matrix-style reasons must be discovered from instruction entries or
manual prose.

### RDNA4-XML-074: VALU source-combination legality is prose-only

Reported by: local audit.

Manual evidence:

- Chapter 7.2.2.2 says not every source combination expressible in the
  microcode format is legal, defines the "scalar value" accounting set, caps
  instructions at two scalar values, permits only one literal constant, forbids
  literals with DPP, counts implicit `VCC` uses for selected opcodes, adds a
  one-scalar-value rule for 64-bit shifts, and gives same-SGPR/same-literal
  accounting details, at `rdna4/README.md:2783` through `:2801`.

XML evidence:

- `OPR_SRC` is a broad per-operand source class covering scalar and vector
  sources at `amdgpu_isa_rdna4.xml:184216` through `:184225`.
- VOP3-family instruction entries enumerate literal-extension alternatives per
  source, including multi-literal-looking condition names such as
  `has_lit_0_has_lit_1` for `V_ADD_CO_CI_U32`, at
  `amdgpu_isa_rdna4.xml:78123` through `:78146`.
- The XML has no cross-operand metadata for scalar-value counting, implicit
  `VCC` accounting, no-literal-with-DPP legality, wave64 second-pass SGPR
  checks, or same-SGPR/same-literal equivalence.

Impact:

XML consumers can decode each operand independently, but cannot validate whether
the source tuple is legal without re-implementing the manual's cross-operand
rules.

### RDNA4-XML-075: `READLANE`/`WRITELANE` lane-number masking is not encoded

Reported by: local audit.

Manual evidence:

- Chapter 7.2.1 says the `V_READLANE` lane-select is limited to the valid
  wave32/wave64 range by ignoring upper bits of the lane number, at
  `rdna4/README.md:2747`.
- The instruction definitions make this explicit for both `V_READLANE_B32` and
  `V_WRITELANE_B32` by selecting `S1.u32[4:0]` in wave32 and `S1.u32[5:0]` in
  wave64, at `rdna4/README.md:22496` through `:22531`.

XML evidence:

- `OPR_SSRC_LANESEL` says lane-select operands are generally scalar integer
  values in the range `0..63`, at `amdgpu_isa_rdna4.xml:192924` through
  `:192929`.
- `V_READLANE_B32` uses `OPR_SSRC_LANESEL` for `SRC1` at
  `amdgpu_isa_rdna4.xml:118052` through `:118057`, and `V_WRITELANE_B32` uses
  the same lane-select type at `:118093` through `:118097`, but the instruction
  entries do not encode the upper-bit-ignore rule.

Impact:

XML consumers may range-check or use the raw lane value instead of modeling the
architectural low-bit masking described by the manual.

### RDNA4-XML-076: SGPR mask/carry and wave64 pass rules are prose-only

Reported by: local audit.

Manual evidence:

- Chapter 7.2.3 says `V_CMP`/`V_CMPX` write full masks regardless of `EXEC`,
  inactive lanes write zero into masks/carry-outs, and VOP2 carry-out writes
  `VCC` while VOP3 writes an arbitrary SGPR pair, at `rdna4/README.md:2831`
  through `:2839`.
- Chapter 7.2.3.2, 7.2.5, and 7.2.6 list VALU instructions that read/write
  SGPR masks or carry, the implicit `VCC` forms, `V_CMPX` as the only VALU
  writer of `EXEC`, the wave64 prohibition on reading and writing the same SGPR
  value, and the two-consecutive-SGPR handling for mask/carry inputs, at
  `rdna4/README.md:2888` through `:2939`.

XML evidence:

- Some concrete operands and implicit operands are present, for example
  `V_ADD_CO_CI_U32` has VOP2 `VCC` carry operands and a VOP3SD `SDST` carry-out
  at `amdgpu_isa_rdna4.xml:78161` through `:78339`.
- The XML has no instruction-stream or wave64-pass metadata for same-SGPR
  read/write hazards, mask-source second-pass increments, inactive-lane zeroing,
  or the full manual mask/carry catalog.

Impact:

The XML identifies many scalar mask/carry operands but does not carry enough
behavioral metadata for a wave64-accurate validator or emulator.

### RDNA4-XML-077: Round/denorm mode bitfields and per-op exceptions are incomplete

Reported by: local audit.

Manual evidence:

- Chapter 7.2.4 defines `MODE.FP_ROUND[3:0]`, `MODE.FP_DENORM[7:4]`,
  half-precision sharing the double-precision fields, float atomics unaffected
  by the bits, DOT2 fixed round/denorm behavior, and opcodes that force input
  and output denorm flushing, at `rdna4/README.md:2892` through `:2911`.

XML evidence:

- `S_ROUND_MODE` and `S_DENORM_MODE` exist only as SOPP instructions with a
  `SIMM16` operand and short descriptions at `amdgpu_isa_rdna4.xml:37952`
  through `:37995`.
- Searches for `FP_ROUND` and `FP_DENORM` in `amdgpu_isa_rdna4.xml` did not
  find the Chapter 7.2 mode bitfield table or the per-op exception lists.

Impact:

The XML exposes the instructions that can set the modes, but not the state
layout and opcode-specific rounding/denorm behavior needed to interpret those
modes.

### RDNA4-XML-078: `PERMLANE`-after-`CMPX` hazard is absent

Reported by: local audit.

Manual evidence:

- Chapter 7.2.8 says `V_PERMLANE*` may not occur immediately after `V_CMPX`
  and requires inserting another VALU opcode such as `V_NOP`, at
  `rdna4/README.md:2949` through `:2951`.

XML evidence:

- The XML instruction entries for `V_PERMLANE16_B32` and
  `V_PERMLANEX16_B32` describe their gather behavior at
  `amdgpu_isa_rdna4.xml:104256` through `:104288` and `:104520` through
  `:104550`.
- `V_CMPX_*` and `V_PERMLANE*` are ordinary instruction entries in the XML; no
  adjacency hazard or stream-validation metadata is present.

Impact:

An XML-only validator cannot reject or warn on the forbidden `V_CMPX` followed
immediately by `V_PERMLANE*` sequence.

### RDNA4-XML-079: Compact true16 VGPR half-addressing is prose-only

Reported by: local audit.

Manual evidence:

- Chapter 7.4 says non-packed 16-bit VALU instructions can separately address
  the two halves of a 32-bit VGPR, naming them `V0.L` and `V0.H` at
  `rdna4/README.md:3073` through `:3083`.
- For VOP1, VOP2, and VOPC encodings, the manual defines
  `SRC/DST[6:0]` as the 32-bit VGPR address, `SRC/DST[7]` as the high/low
  half selector, and says only 256 16-bit VGPRs are addressable at
  `rdna4/README.md:3085` through `:3091`.
- For VOP3, VOP3P, and VINTERP encodings, the manual says `SRC/DST[7:0]`
  remains the 32-bit VGPR address and `OPSEL` selects the half, giving a wave
  access to 512 16-bit VGPRs at `rdna4/README.md:3095` through `:3097`.

XML evidence:

- The `ENC_VOP1` field table exposes `SRC0` and `VDST` widths/offsets, but the
  descriptions are generic operand labels and do not split bit 7 into a true16
  high-half selector or state the 256 16-bit VGPR limit:
  `amdgpu_isa_rdna4.xml:1202` through `:1217`.
- The `ENC_VOPC` and `ENC_VOP2` field tables likewise expose 8-bit `VSRC1` and
  `VDST` fields as generic operands, without the Chapter 7.4 true16 split:
  `amdgpu_isa_rdna4.xml:1569` through `:1584` and `:1819` through `:1844`.
- XML does contain generic OPSEL text for VOP3 and VINTERP true16
  half-selection and destination preservation at `amdgpu_isa_rdna4.xml:2584`
  through `:2585` and `:3129` through `:3130`, but does not encode the compact
  VOP1/VOP2/VOPC bit-7 rule or the 256-versus-512 16-bit VGPR reach.

Impact:

Consumers using XML field tables alone cannot reconstruct the compact true16
VGPR namespace, disassembly names such as `.l`/`.h`, or the addressability
change between compact encodings and OPSEL-based encodings.

### RDNA4-XML-080: MIX-specific VOP3P selector and modifier overloads are prose-only

Reported by: Fermat subreviewer; local audit.

Manual evidence:

- Chapter 7.7 says `NEG_HI` acts as an absolute-value modifier for
  `V_FMA_MIX*` at `rdna4/README.md:3205`.
- The MIX subsection says `MIX`, `MIXLO`, and `MIXHI` reinterpret
  `{OPSEL_HI[i], OPSEL[i]}` as three two-bit source selectors, with `00` and
  `01` selecting `Src[31:0]` as FP32, `10` selecting `Src[15:0]` as FP16, and
  `11` selecting `Src[31:16]` as FP16 at `rdna4/README.md:3210` through
  `:3222`.
- The detailed instruction definitions repeat the same MIX source-selection
  rule and `NEG_HI`-as-ABS behavior for `V_FMA_MIX_F32`,
  `V_FMA_MIXLO_F16`, and `V_FMA_MIXHI_F16` at
  `rdna4/README.md:17185` through `:17249`.

XML evidence:

- The VOP3P field descriptions expose only generic packed-half behavior:
  `NEG_HI` is described as high-operation input negation at
  `amdgpu_isa_rdna4.xml:13868`, `OPSEL` as low-operation half selection at
  `:13888`, and `OPSEL_HI` as high-operation half selection at `:13898`.
- The `V_FMA_MIX_F32`, `V_FMA_MIXLO_F16`, and `V_FMA_MIXHI_F16` instruction
  entries at `amdgpu_isa_rdna4.xml:124615`, `:124941`, and `:125267` carry
  operand formats and fused-FMA descriptions, but not the per-source two-bit
  selector interpretation or the `NEG_HI` absolute-value override.

Impact:

XML-only consumers can decode the MIX opcodes, but cannot derive whether each
source is read as FP32, low FP16, or high FP16 from the selector bits, nor that
`NEG_HI` becomes ABS rather than high-half negation for this instruction class.

### RDNA4-XML-081: SOPP/SOPK literal availability is overbroad in encoding descriptions

Manual evidence:

- `rdna4/README.md:2375` says literal constants are available to all SALU
  microcode formats except SOPP and SOPK, with `S_SETREG_IMM32_B32` as the
  SOPK exception.
- `rdna4/README.md:2564` describes `S_SETREG_IMM32_B32` as a 64-bit SOPK
  instruction whose data comes from a literal constant.
- The SOPK opcode table lists `S_SETREG_IMM32_B32` at opcode 19 in
  `rdna4/README.md:6446`.

XML evidence:

- The `ENC_SOPP` description lists `SOPP + LITERAL (64 bits)` at
  `amdgpu_isa_rdna4.xml:435`.
- The parent `ENC_SOPK` description lists `SOPK + LITERAL (64 bits)` at
  `amdgpu_isa_rdna4.xml:503`, and the `SOPK_INST_LITERAL` alternate repeats
  the same generic phrase at `amdgpu_isa_rdna4.xml:6087`.
- The concrete `S_SETREG_IMM32_B32` entry narrows the literal encoding to
  `SOPK_INST_LITERAL` opcode 19 at `amdgpu_isa_rdna4.xml:37587` through
  `:37591`, with a fieldless `OPR_SIMM32` input at `:37603`.

Impact:

XML consumers that inspect encoding descriptions rather than the full concrete
instruction table can conclude that arbitrary SOPP/SOPK instructions have
literal-extension forms. The structural SOPK record is narrower than the
description text, so this is a metadata precision gap rather than a raw opcode
inventory gap.

### RDNA4-XML-082: Signed SALU add/sub descriptions say carry-out instead of signed overflow

Manual evidence:

- Chapter 6.3 defines SCC for signed add/sub as signed overflow at
  `rdna4/README.md:2391`.
- Chapter 6.4 marks `S_ADD_CO_I32` and `S_SUB_CO_I32` as `Overflow` and says
  their SCC result is overflow at `rdna4/README.md:2402` and `:2407`.
- Chapter 6.4 also marks the SOPK alias `S_ADDK_I32` as `Overflow` at
  `rdna4/README.md:2419`.

XML evidence:

- `S_ADD_CO_I32` says the signed 32-bit add stores the carry-out bit into SCC
  at `amdgpu_isa_rdna4.xml:22840` through `:22844`.
- `S_SUB_CO_I32` says the signed 32-bit subtract stores the carry-out bit into
  SCC at `amdgpu_isa_rdna4.xml:22979` through `:22983`.
- `S_ADDK_CO_I32` / `S_ADDK_I32` says the signed 16-bit-immediate add stores
  the carry-out bit into SCC at `amdgpu_isa_rdna4.xml:37439` through
  `:37443`.

Impact:

An XML-only semantic generator can implement unsigned carry-out for signed
forms whose manual contract is signed overflow. The difference is observable:
`0x7fffffff + 1` has signed overflow without unsigned carry-out, while
`0xffffffff + 1` has unsigned carry-out without signed overflow.

### RDNA4-XML-083: SALU floating-point MODE and exception behavior is prose-only

Reported by: local audit.

Manual evidence:

- Chapter 6.8 says scalar floating-point rounding and denormal handling follow
  `MODE.round` and `MODE.denorm` at `rdna4/README.md:2548`.
- The same section says scalar floating-point arithmetic instructions can
  trigger floating-point exceptions and that those exceptions are handled like
  VALU-pipe exceptions at `rdna4/README.md:2550`.

XML evidence:

- Scalar floating-point instruction entries expose formats, operand sizes, and
  the `SALU` / `FLOATING_POINT` grouping, for example `S_CVT_F32_I32` at
  `amdgpu_isa_rdna4.xml:21957`, `S_CVT_F16_F32` at
  `amdgpu_isa_rdna4.xml:22177`, and `S_ADD_F16` at
  `amdgpu_isa_rdna4.xml:30625`.
- `S_ROUND_MODE` and `S_DENORM_MODE` exist as SOPP instructions with a generic
  `SIMM16` operand at `amdgpu_isa_rdna4.xml:37952` through `:37995`, but the
  XML does not attach MODE dependencies or exception behavior to the scalar FP
  arithmetic/conversion instructions.

Impact:

This is the Chapter 6.8 scalar-FP counterpart to `RDNA4-XML-077`. XML-only
generators can discover the opcodes and operand formats but not that scalar FP
results and exception state are mode-dependent.

### RDNA4-XML-084: Scalar F16 storage and high-half rules are not structured

Reported by: local audit.

Manual evidence:

- Chapter 6.8 says `S_CVT_HI_F32_F16` converts `S0[31:16].f16` at
  `rdna4/README.md:2546` and again in the detailed definition at
  `rdna4/README.md:10522`.
- Chapter 6.8 says scalar F16 instructions do not encode half SGPRs, operate
  on the low `bit[15:0]` part of the named SGPR, and set destination
  `bit[31:16]` to zero at `rdna4/README.md:2552`.
- The detailed `S_CVT_PK_RTZ_F16_F32` definition describes a packed
  half-precision result at `rdna4/README.md:8856`.

XML evidence:

- `S_CVT_HI_F32_F16` carries prose saying the input comes from the high 16 bits,
  but both the normal and literal source operands are still modeled as
  `FMT_NUM_F16` / `OperandSize 16` at `amdgpu_isa_rdna4.xml:22287` through
  `:22324`.
- Representative scalar F16 instructions such as `S_CVT_F16_F32` and
  `S_ADD_F16` expose 16-bit `SDST` operands at
  `amdgpu_isa_rdna4.xml:22177` through `:22208` and
  `amdgpu_isa_rdna4.xml:30625` through `:30649`, but the XML does not say the
  destination is a whole SGPR write whose high half is zeroed.
- `S_CVT_PK_RTZ_F16_F32` describes a packed result but still marks `SDST` as
  `FMT_NUM_F16` / `OperandSize 16` at `amdgpu_isa_rdna4.xml:30508` through
  `:30525`.

Impact:

The XML mixes logical F16 element width with raw 32-bit scalar storage. A
generator that trusts only `OperandSize 16` can mask literal extension words,
preserve a destination high half that hardware zeroes, or invent an
old-destination dependency for instructions that overwrite their scalar result.

### RDNA4-XML-085: SMEM access-size-dependent address masking is not structured

Reported by: local audit.

Manual evidence:

- Chapter 8.1.1 says `S_LOAD` address components are byte quantities, but
  their low bits are ignored according to access size: two low bits for dword
  loads, one low bit for 16-bit loads, and no low-bit masking for 8-bit loads
  at `rdna4/README.md:3903` through `:3911`.
- Chapter 8.1.2 gives the same access-size masking rule for the buffer
  resource base and buffer offsets at `rdna4/README.md:3923` through `:3937`.
- Chapter 8.1.3 separately states that 16-bit loads must be 16-bit aligned at
  `rdna4/README.md:3945` through `:3954`.
- Chapter 8.4.1 restates the final forced-alignment rule: the base address is
  forced to dword alignment, dword-or-larger loads force dword alignment,
  16-bit loads force two-byte alignment, and byte loads have no forced
  alignment at `rdna4/README.md:4000` through `:4002`.

XML evidence:

- The SMEM `IOFFSET` field description says the signed byte offset is aligned
  to one dword at `amdgpu_isa_rdna4.xml:843` through `:849`.
- The SMEM `SOFFSET` field description also says the unsigned byte offset is
  aligned to one dword at `amdgpu_isa_rdna4.xml:907` through `:913`.
- Narrow instruction entries expose implicit memory operand sizes, for example
  `S_LOAD_I8` uses an 8-bit `OPR_GPUMEM` operand at
  `amdgpu_isa_rdna4.xml:16260` through `:16290`, and `S_LOAD_I16` uses a
  16-bit `OPR_GPUMEM` operand at `amdgpu_isa_rdna4.xml:16352` through
  `:16382`. The XML does not connect those sizes to per-component low-bit
  masking rules.

Impact:

An XML-only address generator can incorrectly dword-align 8-bit and 16-bit
SMEM accesses, or mask the final summed address instead of each address
component. The instruction memory size is present, but the alignment formula is
manual prose.

### RDNA4-XML-086: SMEM buffer-resource address and bounds formula is prose-only

Reported by: local audit.

Manual evidence:

- Chapter 8.1.2 says `S_BUFFER_LOAD` gets its base address from the buffer
  resource and uses only `base_address`, `stride`, and `num_records` at
  `rdna4/README.md:3915` through `:3923`.
- The same section says scalar memory does not support swizzled buffers,
  `stride` is used only for bounds checking, negative `IOFFSET` causes
  `MEMVIOL`, and the manual pseudocode defines `resource[47:0]`,
  `resource[61:48]`, `resource[95:64]`, `m_size`, and `stride_scale` behavior
  at `rdna4/README.md:3921` through `:3942`.
- Chapter 8.4.1 defines scalar-buffer address out-of-range as
  `offset >= ((stride == 0 ? 1 : stride) * num_records)`, where `offset` is
  `IOFFSET + {M0 or sgpr-offset}`, and says out-of-range dwords in a buffer
  load return zero at `rdna4/README.md:4004` through `:4010`.

XML evidence:

- The SMEM `SBASE` field description says buffer descriptors use only base,
  stride, and `NUM_RECORDS` at `amdgpu_isa_rdna4.xml:873` through `:883`.
- Concrete `S_BUFFER_LOAD_*` entries use `FMT_RSRC_SCALAR` / `OPR_SREG` /
  `OperandSize 128` for `SBASE`, for example `S_BUFFER_LOAD_B32` at
  `amdgpu_isa_rdna4.xml:16444` through `:16477` and `S_BUFFER_LOAD_I16` at
  `amdgpu_isa_rdna4.xml:16827` through `:16856`.
- The XML does not structure the descriptor bit slices, bounds formula,
  per-dword out-of-range zeroing, `stride_scale` ignore rule, or `MEMVIOL`
  behavior.

Impact:

XML consumers can distinguish raw-pointer loads from scalar-buffer loads, but
cannot implement correct scalar-buffer address calculation or bounds behavior
without manual prose.

### RDNA4-XML-087: SMEM destination alignment and null-prefetch rules are incomplete

Reported by: local audit.

Manual evidence:

- Chapter 8.1 says `SDATA` destinations for two-dword loads must be even, while
  loads of three or more dwords must be aligned to a multiple of four SGPRs at
  `rdna4/README.md:3865`.
- The same table permits SGPR, VCC, and `NULL` destinations but not `EXEC` or
  `M0` at `rdna4/README.md:3865`.
- Chapter 8.1 says a `NULL` destination can be used for "prefetch data with
  acknowledge" at `rdna4/README.md:3897` through `:3899`.
- Chapter 8.4 repeats that `SDST` must be even for two-dword fetches and a
  multiple of four for larger fetches, with invalid data possible when the rule
  is not followed, at `rdna4/README.md:3984` through `:3986`.

XML evidence:

- `OPR_SREG` includes VCC and `NULL` and excludes `M0` / `EXEC`, so the broad
  selector set is available at `amdgpu_isa_rdna4.xml:189665` through
  `:190298`.
- The generic `OPR_SREG` alignment text covers 64-bit even alignment and
  128-bit-and-wider multiple-of-four alignment at
  `amdgpu_isa_rdna4.xml:189669` through `:189673`.
- `S_LOAD_B96` and `S_BUFFER_LOAD_B96` expose 96-bit `SDATA` operands at
  `amdgpu_isa_rdna4.xml:16214` through `:16243` and
  `amdgpu_isa_rdna4.xml:16689` through `:16718`, but no structured rule says
  this three-dword case also requires multiple-of-four destination alignment.
  The XML `NULL` description says destinations are not written at
  `amdgpu_isa_rdna4.xml:190294` through `:190298`, but it does not attach the
  SMEM prefetch-with-acknowledge behavior to `SDATA`.

Impact:

Basic destination selector legality is recoverable, but XML-only validation
can miss the 96-bit alignment case and cannot distinguish a `NULL` SMEM load as
the manual's acknowledged prefetch form.

### RDNA4-XML-088: SMEM group and `S_DCACHE_INV` clause restrictions are prose-only

Reported by: local audit.

Manual evidence:

- Chapter 8.3 defines a clause as a sequence starting with `S_CLAUSE` and
  continuing for 2-63 instructions at `rdna4/README.md:3974`.
- Chapter 8.3 defines a group as a same-type instruction sequence that ends at a
  non-SMEM instruction, says scalar memory instructions are issued in groups,
  and says hardware does not force one wave to execute an entire group before
  issuing another wave at `rdna4/README.md:3976`.
- Chapter 8.3 says `INV` must be in a group by itself and may not be in a clause
  at `rdna4/README.md:3978` through `:3980`.

XML evidence:

- `S_CLAUSE` has only the generic "Mark the beginning of a clause" description
  and an `OPR_CLAUSE` immediate operand at `amdgpu_isa_rdna4.xml:37782` through
  `:37797`.
- `OPR_CLAUSE` encodes `BREAK_SPAN` and `LENGTH` fields at
  `amdgpu_isa_rdna4.xml:181513` through `:181534`, but has no clause-type or
  group-type field.
- `S_DCACHE_INV` is an operandless SMEM instruction at
  `amdgpu_isa_rdna4.xml:16919` through `:16930`; no XML metadata marks it as
  group-alone or forbidden after `S_CLAUSE`.

Impact:

XML-only consumers can decode `S_CLAUSE` and `S_DCACHE_INV`, but cannot infer the
SMEM instruction-stream rule that scalar-memory groups stop at non-SMEM
instructions or that scalar invalidates are excluded from clauses and groups with
other SMEM operations.

### RDNA4-XML-089: SMEM SGPR range-check fallback results are prose-only

Reported by: local audit.

Manual evidence:

- Chapter 8.4 says `S_BUFFER_LOAD` `SBASE` must identify an even SGPR and that
  an out-of-range `SBASE` uses the value from `SGPR0` at
  `rdna4/README.md:3988` through `:3990`.
- Chapter 8.4.1 says hardware checks source and destination SGPRs for
  out-of-range conditions at `rdna4/README.md:3996` through `:3998`.
- Chapter 8.4.1 says out-of-range source SGPR data is replaced with zero, while
  an out-of-range destination SGPR suppresses writeback for the instruction at
  `rdna4/README.md:4012` through `:4018`.

XML evidence:

- The SMEM `SBASE` field records the implied zero low bit and scalar-buffer
  descriptor fields at `amdgpu_isa_rdna4.xml:873` through `:883`, but does not
  encode the out-of-range fallback to `SGPR0`.
- `OPR_SMEM_OFFSET` describes scalar-memory offsets as SGPR, `M0`, or immediate
  sources at `amdgpu_isa_rdna4.xml:182903` through `:182916`; `OPR_SREG`
  enumerates scalar registers and generic alignment at
  `amdgpu_isa_rdna4.xml:189665` through `:189672`. Neither operand type encodes
  the SMEM-specific source-zero or destination-no-write consequences.
- Concrete SMEM entries expose `SBASE`, `SOFFSET`, and `SDATA` operands, for
  example `S_BUFFER_LOAD_B32` at `amdgpu_isa_rdna4.xml:16444` through `:16477`,
  but no instruction metadata binds these operands to current-wave SGPR range
  checks.

Impact:

An XML-only SMEM implementation cannot infer when to substitute `SGPR0`, return
zero for an out-of-range source, or suppress destination writeback for an
out-of-range scalar-memory load destination.

### RDNA4-XML-090: Scalar prefetch address, length, mode, and cache effects are prose-only

Reported by: local audit.

Manual evidence:

- Chapter 8.5 says scalar prefetch instructions request instruction or data
  prefetch into first-level caches, do not use `KMcnt`, apply `SCOPE`/`TH` cache
  reuse policies only to data-prefetch, and are skipped as `S_NOP` when
  `MODE.SCALAR_PREFETCH_EN == 0` at `rdna4/README.md:4020` through `:4028`.
- Chapter 8.5 says prefetch length is `SOFFSET + SDATA`, with upper bits dropped
  modulo 32, representing 1-32 chunks of 128 bytes at
  `rdna4/README.md:4028` through `:4030`.
- Chapter 8.5.1 and 8.5.2 give per-instruction address formulas: base or
  PC-relative address plus signed `IOFFSET`, forced 128-byte alignment,
  `PC + 8` for PC-relative forms, buffer-resource base bits, no bounds checking
  for `S_BUFFER_PREFETCH_DATA`, and negative buffer `IOFFSET` dropping the
  request without `MEMVIOL` at `rdna4/README.md:4034` through `:4063`.
- The detailed instruction definitions repeat the `MODE.SCALAR_PREFETCH_EN`
  guard, no completion/error status, length modulo/add-one calculation, and cache
  target calls at `rdna4/README.md:11941` through `:12070`.

XML evidence:

- XML contains the five prefetch opcodes and broad operand shapes:
  `S_PREFETCH_INST` at `amdgpu_isa_rdna4.xml:17023` through `:17060`,
  `S_PREFETCH_INST_PC_REL` at `:17070` through `:17101`,
  `S_PREFETCH_DATA` at `:17111` through `:17148`,
  `S_BUFFER_PREFETCH_DATA` at `:17158` through `:17195`, and
  `S_PREFETCH_DATA_PC_REL` at `:17205` through `:17236`.
- `SDATA` is typed as `OPR_SIMM5` for these entries, so the immediate field width
  is recoverable, but the XML entries do not encode the `SOFFSET + SDATA`
  length expression, the add-one/128-byte scaling, or PC-relative `PC + 8`
  address base.
- The XML descriptions do not encode `MODE.SCALAR_PREFETCH_EN`, no-`KMcnt`
  behavior, no completion/error status, data-only `SCOPE`/`TH` cache-policy
  applicability, or the buffer-prefetch negative-`IOFFSET` drop rule.

Impact:

XML-only consumers can decode and disassemble scalar prefetch instructions, but
cannot derive their mode gating, cache target/effect, address and length
calculation, wait-counter nonparticipation, or buffer-prefetch edge cases.

### RDNA4-XML-091: Vector-buffer descriptor, addressing, and range-check semantics are not structured

Reported by: local audit.

Manual evidence:

- Chapter 9 says all buffer operations use a 128-bit `V#` resource constant
  that defines the buffer address and characteristics at `rdna4/README.md:4086`.
- Table 47 says `VADDR` uses one VGPR for index, one VGPR for offset, or both,
  `IOFFSET` is a signed 24-bit byte offset that must be non-negative, and
  `RSRC` selects four consecutive SGPRs aligned to a multiple of four at
  `rdna4/README.md:4128` through `:4145`.
- Chapter 9.4 defines index/offset/stride addressing, `const_add_tid_enable`,
  `const_swizzle_enable`, and `const_index_stride` at `rdna4/README.md:4386`
  through `:4408`.
- Chapter 9.4.1 defines `OOB_SELECT`, per-component versus all-or-nothing
  range checks, zero-return/drop behavior, and `SEL_1` out-of-range load results
  at `rdna4/README.md:4434` through `:4458`.
- Chapter 9.4.1.1 through 9.4.2 define structured, raw, scratch, scalar, and
  swizzled address formulas at `rdna4/README.md:4459` through `:4530`.
- Chapter 9.6 defines the full buffer-resource bit layout, including base,
  stride, swizzle, `Num_records`, `Dst_sel`, format, stride scale,
  index stride, add-tid, compression fields, `OOB_SELECT`, and type at
  `rdna4/README.md:4562` through `:4588`.

XML evidence:

- `ENC_VBUFFER` exposes the instruction word fields `FORMAT`, `IDXEN`,
  `IOFFSET`, `OFFEN`, `RSRC`, `SCOPE`, `SOFFSET`, `TFE`, `TH`, `VADDR`, and
  `VDATA` at `amdgpu_isa_rdna4.xml:3527` through `:3777`.
- The `IOFFSET` field description says only "Signed byte offset" at
  `amdgpu_isa_rdna4.xml:3669` through `:3675`; it does not encode the manual's
  non-negative restriction.
- The `RSRC` field description records only that the SGPR must be multiple-of-4
  aligned at `amdgpu_isa_rdna4.xml:3709` through `:3715`.
- `FMT_RSRC*` data formats are modeled as one opaque 128-bit `Descriptor` at
  `amdgpu_isa_rdna4.xml:176246` through `:176447`.
- Searches in the RDNA4 XML found no structured `OOB_SELECT`, `Dst_sel`,
  `swizzle`, `add_tid`, `index_stride`, `stride_scale`, or vector-buffer
  `Num_records` descriptor metadata.

Impact:

An XML-only decoder can recover the instruction-word field layout and broad
operand shapes, but cannot derive legal `IOFFSET` values, buffer-resource field
extraction, structured/raw/scratch/swizzled address formulas, range-check
classification, or out-of-range return/drop behavior.

### RDNA4-XML-092: Vector-buffer data selection and D16 conversion tables are prose-only

Reported by: local audit.

Manual evidence:

- Chapter 9.1 separates untyped buffers, where the resource constant supplies
  data format, from typed buffers, where the instruction supplies data format,
  at `rdna4/README.md:4074` through `:4084`.
- Chapter 9.2 lists address VGPR usage, consecutive load/store data VGPR usage,
  atomic-with-return VDATA reuse, and the VGPR/memory layout tables for
  ordinary, formatted, and D16 buffer operations at `rdna4/README.md:4265`
  through `:4331`.
- Chapter 9.3 says buffer data is controlled by resource format, `dst_sel`, and
  opcode; defines which instruction families use resource, instruction, or
  implied format; says resource `INVALID` remains an unbound-resource marker;
  and defines first-component replication at `rdna4/README.md:4335` through
  `:4362`.
- Chapter 9.3.1 says D16 loads/stores pack two 16-bit values per VGPR, use
  low-half first, distinguish D16 from D16_HI, truncate float32-to-float16, and
  round other conversions to nearest even at `rdna4/README.md:4363` through
  `:4375`.

XML evidence:

- Representative XML entries describe the broad source of data format, such as
  `BUFFER_LOAD_FORMAT_X` using the resource descriptor at
  `amdgpu_isa_rdna4.xml:38874` through `:38910`, and typed buffers using the
  instruction format at `amdgpu_isa_rdna4.xml:43094` through `:43130`.
- Representative D16_HI XML entries say the result is stored into the high
  16 bits, for example `BUFFER_LOAD_D16_HI_U8` at
  `amdgpu_isa_rdna4.xml:40698` through `:40738`.
- Those descriptions do not encode the Chapter 9 data-control table, `Dst_sel`
  descriptor fields, first-component replication, `INVALID`/unbound exception,
  resource/instruction mismatch outcomes, D16 packing order, or conversion
  rounding/truncation choices as structured metadata.

Impact:

XML consumers can identify many buffer opcodes and some textual intent, but
must still use the manual to implement component selection, format-source
selection, D16 packing/conversion, and unbound/mismatch behavior.

### RDNA4-XML-093: Vector-buffer `TFE` status destination is only a bitfield description

Reported by: local audit.

Manual evidence:

- Table 47 says `TFE` enables PRT fault reporting and, when a fetch returns a
  NACK, writes status to the VGPR after the last fetch-destination VGPR at
  `rdna4/README.md:4145`.
- Chapter 3.3.4 says destination VGPR out-of-range checks include the extra PRT
  VGPR and nullify the fetch if that extra VGPR would be out of range,
  regardless of whether the texture system actually returns it at
  `rdna4/README.md:909` through `:910`.

XML evidence:

- `ENC_VBUFFER` exposes a raw `TFE` bit described as texture-fail enable at
  `amdgpu_isa_rdna4.xml:3739` through `:3745`.
- VBUFFER instruction operands do not add a conditional extra destination VGPR
  when `TFE=1`; for example `BUFFER_LOAD_FORMAT_X` exposes only the ordinary
  `VDATA` destination at `amdgpu_isa_rdna4.xml:38882` through `:38887`.

Impact:

The XML identifies that a TFE bit exists, but an XML-only dependency,
disassembly, or emulator path cannot infer the extra status writeback or the
extra destination-range check.

### RDNA4-XML-094: Image `TFE` / `LWE` status returns are not structured

Reported by: local audit.

Manual evidence:

- `rdna4/README.md:4666` through `:4678` says `TFE` and `LWE` can cause an
  extra VGPR write after all fetch-destination VGPRs; `LWE` works only for
  sampler operations and is ignored for non-sampler operations.
- `rdna4/README.md:4684` through `:4698` defines the conditional status return,
  says no extra value is returned when no thread sees a texture fault or LOD
  warning, defines the payload as
  `TEXEL_FAIL | (LOD_WARNING << 1) | (LOD << 16)`, and gives
  `TEXEL_FAIL` precedence over `LOD_WARNING`.

XML evidence:

- `ENC_VIMAGE` exposes only a raw `TFE` bit with a short "Texture Fail Enable"
  description at `amdgpu_isa_rdna4.xml:3066` through `:3074`.
- `ENC_VSAMPLE` exposes raw `LWE` and `TFE` bits at
  `amdgpu_isa_rdna4.xml:3288` through `:3296` and `:3358` through `:3365`.
- Searching `amdgpu_isa_rdna4.xml` for `TEXEL_FAIL`, `LOD_WARNING`,
  `LOD_CLAMPED`, and `min_lod_warning` returns no matches.

Impact:

The XML can decode the selector bits, but cannot derive the dynamic extra
destination VGPR, the status payload layout, the conditional no-status case, or
the sampler-only `LWE` rule.

### RDNA4-XML-095: Image address, `ACNT`, `A16`, and `G16` tables are prose-only

Reported by: local audit.

Manual evidence:

- `rdna4/README.md:4718` through `:4777` gives no-sampler address typing,
  cubemap `face_id = slice * 6 + face`, MSAA limitations, opcode-by-`DIM`
  address tables, and the `ACNT` definition.
- `rdna4/README.md:4781` through `:4785` gives sampler address typing,
  cubemap `face_id = slice * 8 + face`, sampler-specific MSAA restrictions, and
  `IMAGE_MSAA_LOAD`'s no-sampler-in-VSAMPLE rule.
- `rdna4/README.md:4853` through `:4873` defines sampler suffix address parts,
  packed signed texel offsets, derivative counts, the body count formula, and
  the bias-versus-derivative exclusion.
- `rdna4/README.md:4893` through `:4903` defines VADDR component gathering,
  VIMAGE versus VSAMPLE sequential address groups, A16 pair packing, and the
  ray-tracing grouping exception.

XML evidence:

- `ENC_VIMAGE` has raw `VADDR0` through `VADDR4` fields at
  `amdgpu_isa_rdna4.xml:3086` through `:3134`; `ENC_VSAMPLE` has raw
  `VADDR0` through `VADDR3` fields at `:3394` through `:3432`.
- Representative instruction entries expose broad or fixed address operands:
  `IMAGE_LOAD` has an unnamed 128-bit VGPR operand at
  `amdgpu_isa_rdna4.xml:56396` through `:56408`; `IMAGE_SAMPLE` has a
  96-bit F32 address operand at `:170268` through `:170285`; and
  `IMAGE_SAMPLE_D` has a 288-bit F32 address operand at `:170312` through
  `:170332`.
- The XML does not encode the per-opcode/per-`DIM` `ACNT` tables, the different
  cubemap formulas, the sampler suffix ordering rules, the signed 6-bit offset
  subfields, the bias/derivative exclusion, `A16` "pairs cannot be split"
  rule, or `G16` derivative packing tables.

Impact:

XML-derived consumers can see that VADDR fields exist, but cannot reconstruct
the legal address component layout, operand grouping, or VGPR consumption rules
for image and sample instructions.

### RDNA4-XML-096: Image `DMASK`, data-VGPR counts, and atomic legality are only partially descriptive

Reported by: local audit.

Manual evidence:

- `rdna4/README.md:4610` says `DMASK` selects returned components and that
  missing load components come from `T#.DST_SEL`.
- `rdna4/README.md:4638` says `IMAGE_MSAA_LOAD` behaves like gather4 and
  returns four VGPRs, or two when `D16=1`.
- `rdna4/README.md:4909` through `:4927` says data VGPR count is determined by
  `DMASK`, stores write whole elements, missing surface components are filled by
  replicating the first supplied VGPR, and D16 stores care only about the
  `DMASK` popcount.
- `rdna4/README.md:4931` through `:4939` restricts image atomics to 32- or
  64-bit-per-pixel surfaces, lists legal atomic `DMASK` values, and says atomic
  returns write back to the same VDATA VGPRs.

XML evidence:

- `ENC_VIMAGE` and `ENC_VSAMPLE` carry substantial `DMASK` prose at
  `amdgpu_isa_rdna4.xml:2982` through `:2997` and `:3254` through `:3269`,
  including D16 write packing, `DMASK==0`, and gather4 return counts.
- That prose is not structured into operand-count, legality, or resource-format
  metadata, and it says writes missing components get zero at
  `amdgpu_isa_rdna4.xml:2985` through `:2989` and `:3257` through `:3261`,
  while the manual's store rule fills missing surface components by replicating
  the first VGPR.
- Representative entries use fixed VDATA operand sizes: `IMAGE_LOAD` publishes
  128-bit VDATA at `amdgpu_isa_rdna4.xml:54214` through `:54219`,
  `IMAGE_GATHER4` publishes 128-bit VDATA at `:169941` through `:169946`, and
  `IMAGE_ATOMIC_CMPSWAP` publishes a 32-bit VDATA operand at `:54698` through
  `:54703`.
- The image resource descriptor is opaque at `amdgpu_isa_rdna4.xml:173057`, so
  `T#.DST_SEL`, resource data format compatibility, and 32-bit versus 64-bit
  atomic surface selection are not structured here.

Impact:

The XML contains helpful `DMASK` notes, but consumers still need the manual to
derive dynamic data VGPR counts, store fill behavior, D16 component counts,
atomic legal `DMASK` values, and return-VGPR aliasing.

### RDNA4-XML-097: Image sample denormal and out-of-range policy is prose-only

Reported by: local audit.

Manual evidence:

- `rdna4/README.md:4610` says image operations do not generate `MemViol` for
  out-of-range accesses and instead apply clamp modes.
- `rdna4/README.md:4941` through `:4943` says sample operations flush denormals
  while loads do not modify denormals.

XML evidence:

- The image and sample encoding fields expose `DIM`, `SCOPE`, `TH`, `TFE`,
  `LWE`, `UNORM`, and resource operands, but searches for image/sample
  `MemViol`, out-of-range clamp policy, and sample denormal behavior found no
  corresponding metadata near `amdgpu_isa_rdna4.xml:2902` through `:3436` or
  the `IMAGE_*` / `IMAGE_SAMPLE*` instruction entries.

Impact:

XML-based simulators cannot infer image memory fault behavior or the
sample-versus-load denormal policy without manual prose and resource-descriptor
semantics.

### RDNA4-XML-098: Image resource descriptor bitfields are opaque

Reported by: local audit and Lagrange subagent.

Manual evidence:

- `rdna4/README.md:4963` says the image resource/T# defines memory location,
  dimensions, tiling, and data format; resources are stored in four or eight
  consecutive SGPRs.
- `rdna4/README.md:4969` through `:4984` defines the 128-bit resource fields,
  including base address, max/base/last mip levels, format, width, height,
  `dst_sel_x` through `dst_sel_w`, border-color swizzle, and resource type.
- `rdna4/README.md:4985` through `:4996` defines the 256-bit extension fields,
  including depth or pitch bits, `pitch_msb`, base array, `UAV3D`,
  `min_lod_warn`, corner-sample mode, and `min_lod`.
- `rdna4/README.md:4998` says an all-zero resource is treated as unbound,
  returns zeros, and does not generate a memory transaction.

XML evidence:

- `ENC_VIMAGE` and `ENC_VSAMPLE` expose only the raw `R128` instruction bit as
  "Texture resource size" at `amdgpu_isa_rdna4.xml:3922` through `:3927` and
  `:4225` through `:4230`.
- Their `DIM` descriptions say the instruction field should match
  `TYPE[3:1]` from the resource descriptor at `amdgpu_isa_rdna4.xml:3858`
  through `:3859` and `:4151` through `:4152`, but XML does not define the
  descriptor `TYPE` field or its valid/reserved values.
- `FMT_IMG` is a single unsigned 256-bit `Descriptor` field at
  `amdgpu_isa_rdna4.xml:173057` through `:173075`; `FMT_IMG_BVH` is similarly
  a single 128-bit BVH descriptor at `:173077` through `:173095`.
- Representative ordinary image/sample operands still name `RSRC` as
  256-bit `FMT_IMG`, for example `IMAGE_LOAD` at
  `amdgpu_isa_rdna4.xml:56400` through `:56405` and `IMAGE_SAMPLE` at
  `:170273` through `:170278`.

Impact:

XML-only consumers can tell that an image resource operand and `R128` selector
exist and that `DIM` should be compatible with a resource type, but cannot
derive the T# field layout, channel-selection semantics, resource type
legality, 4-SGPR versus 8-SGPR operand footprint, mip/pitch/depth
interpretation, LOD warning/clamp behavior, or all-zero unbound-resource rule
without manual prose.

### RDNA4-XML-099: Image sampler descriptor bitfields are opaque

Reported by: local audit and Meitner subagent.

Manual evidence:

- `rdna4/README.md:5002` says the sampler resource/S# defines texture-map
  operations, primarily address clamping and filtering, is supplied with every
  sample instruction, and occupies four consecutive SGPRs.
- `rdna4/README.md:5006` through `:5020` defines sampler low-word fields for
  per-axis clamp/wrap mode, max anisotropy ratio, depth-compare function,
  force-unnormalized coordinates, anisotropy threshold and bias, motion-
  compensation coordinate truncation, degamma controls, coordinate truncation,
  cube-wrap disable, and filter mode.
- `rdna4/README.md:5024` through `:5036` defines skip-degamma, minimum and
  maximum LOD, LOD bias fields, XY/Z/mip filters, border-color pointer, and
  border-color type.

XML evidence:

- `FMT_SAMP` is a single unsigned 128-bit `Descriptor` field at
  `amdgpu_isa_rdna4.xml:175350` through `:175368`.
- `ENC_VSAMPLE` exposes only the raw `SAMP` SGPR selector, with multiple-of-4
  alignment, at `amdgpu_isa_rdna4.xml:4245` through `:4250`.
- Representative sample instructions expose `SAMP` as a 128-bit `FMT_SAMP`
  operand, for example `IMAGE_SAMPLE` at `amdgpu_isa_rdna4.xml:170280` through
  `:170285`.
- The instruction-side `UNORM` description says it is logically ORed with
  `SAMP.FORCE_UNNORMALIZED` at `amdgpu_isa_rdna4.xml:4285` through `:4292`, but
  the sampler descriptor field itself is not structured.
- XML sample suffix descriptions expose instruction-side LOD, bias, clamp, and
  PCF address-data requirements, for example at
  `amdgpu_isa_rdna4.xml:170349`, `:170396`, `:170490`, and `:171853`, but not
  the sampler descriptor fields that define LOD limits/biases or the depth-
  compare function.

Impact:

XML-derived consumers can identify that a sampler operand exists and that the
instruction `UNORM` bit interacts with a sampler force-unnormalized bit, but
cannot derive the sampler's clamp, filter, anisotropy, PCF/depth compare,
degamma, coordinate rounding, LOD, border-color, or force-unnormalized behavior
from XML alone.

Ambiguity note:

Chapter 10.6 leaves unassigned sampler bit ranges between the documented fields
at `rdna4/README.md:5028` through `:5035`. The general instruction-table rule at
`rdna4/README.md:6312` says reserved instruction fields must be zero, but this
sampler descriptor slice does not independently label those descriptor holes.
The opaque XML descriptor also gives no reserved-mask or zeroing guidance for
them.

### RDNA4-XML-100: Buffer/image data-format enumeration is absent

Reported by: local audit; Chandrasekhar subagent.

Manual evidence:

- Chapter 10.7 says the table details all data formats usable by image and
  buffer resources at `rdna4/README.md:5040`.
- Table 62 enumerates numeric surface-format values 0-89 at
  `rdna4/README.md:5046` through `:5084`, including normalized, scaled,
  integer, float, packed 10/11-bit, sRGB, depth/stencil-like, and channel-order
  forms.
- The same table enumerates compressed and special values 109-122, 205-206,
  and 227 at `rdna4/README.md:5080` through `:5097`, including BCn, YCBCR, and
  `6E4_FLOAT` formats.

XML evidence:

- Searching `amdgpu_isa_rdna4.xml` for representative Table 62 names such as
  `BC1`, `YCBCR`, `6E4`, `10_10_10_2`, `GB_GR`, `32_FLOAT_CLAMP`, and `X24`
  returns no matches.
- XML `DataFormats` defines generic operand/value shapes such as `FMT_ANY`,
  `FMT_IMG`, `FMT_NUM_F32`, packed `FMT_NUM_PK*`, and `FMT_RSRC_TYPED`, but not
  the numbered surface-format enum from manual Table 62.
- `ENC_MTBUF` exposes a raw 7-bit `FORMAT` field with only the description
  "Format for typed buffer" at `amdgpu_isa_rdna4.xml:3649` through `:3656`.
- Formatted buffer and typed-buffer instruction descriptions mention
  descriptor- or instruction-controlled format conversion, for example
  `BUFFER_LOAD_FORMAT_X` at `amdgpu_isa_rdna4.xml:38874` through `:38875`,
  `BUFFER_STORE_FORMAT_X` at `:39086` through `:39087`, and typed-buffer
  descriptions around `:43095`, but they do not enumerate format values or
  properties.
- Image load/store descriptions similarly say format conversion is specified
  by the resource descriptor at `amdgpu_isa_rdna4.xml:56382` and `:56628`, but
  image resource descriptors are opaque as recorded in `RDNA4-XML-098`.

Impact:

XML-only consumers cannot map data-format numbers to names, component widths,
component counts, numeric classes, normalized/scaled/sRGB/depth/compressed/YCBCR
behavior, invalid/reserved treatment, or value legality. The 7-bit MTBUF
instruction `FORMAT` field also cannot represent manual values 205, 206, or
227, while the XML does not clarify which Table 62 values are legal for
instruction-side typed-buffer formats versus image/resource descriptor formats.
The generic XML `FMT_NUM_*` entries are only loose bit-shape analogies, not
aliases for the numbered surface-format enum.

### RDNA4-XML-101: VMEM dependency-counter and source-read timing rules are absent

Reported by: local audit; Kuhn subagent.

Manual evidence:

- Chapter 10.8 says issuing a VM instruction schedules address and store-data
  VGPR reads to the texture unit, and an ALU instruction that tries to write
  that data before it has been sent to the texture unit is stalled at
  `rdna4/README.md:5101`.
- Chapter 10.8 says shader authors must wait for VMEM read completion before
  reading data fetched from the TC, names `LOADcnt` and `STOREcnt`, and says
  ray-tracing image BVH instructions are tracked with `BVHcnt` at
  `rdna4/README.md:5103` through `:5105`.
- The referenced data-dependency section says each wave has outstanding-memory
  counters that increment on issue and decrement on completion at
  `rdna4/README.md:2103` through `:2107`.
- It also states that instructions of the same type return in issue order
  except SMEM, but different types may complete out of order; samples stay in
  order with samples and are unordered with loads, stores, and BVH ops at
  `rdna4/README.md:2133`.
- The memory-counter table gives counter widths and increment/decrement classes
  for `LOADcnt`, `SAMPLEcnt`, `BVHcnt`, and `STOREcnt` at
  `rdna4/README.md:2167` through `:2188`, including `STOREcnt` completion at
  the memory hierarchy level selected by `SCOPE`.
- The vector-memory dependency subsection says `LOADcnt` covers loads,
  atomic-with-return, and cache-invalidate; `SAMPLEcnt` covers sample/gather;
  `STOREcnt` covers stores and atomic-without-return; global invalidates use
  `LOADcnt`; global write-back/write-back-invalidate use `STOREcnt`; and loads,
  stores, and samples are ordered only within their own classes at
  `rdna4/README.md:2217` through `:2227`.

XML evidence:

- XML preserves the operand fields that identify address/data VGPRs, for example
  VBUFFER `VADDR`/`VDATA` at `amdgpu_isa_rdna4.xml:3759` through `:3770`,
  VIMAGE `VADDR*`/`VDATA` at `:3972` through `:4023`, VFLAT `VADDR`/`VSRC` at
  `:4670` through `:4691`, and VGLOBAL `VADDR`/`VSRC` at `:5080` through
  `:5101`, but it does not encode the texture-unit source-read timing or ALU
  overwrite stall rule.
- XML has the split wait instructions, for example `S_WAIT_LOADCNT`,
  `S_WAIT_STORECNT`, `S_WAIT_SAMPLECNT`, and `S_WAIT_BVHCNT` at
  `amdgpu_isa_rdna4.xml:38613` through `:38701`, but those descriptions only
  encode threshold predicates.
- The combined wait operand splits DS and memory threshold fields at
  `amdgpu_isa_rdna4.xml:195495` through `:195511`; it does not identify which
  VMEM instructions increment/decrement the counters or their ordering model.
- XML's generic `VMEM` functional-group text at `amdgpu_isa_rdna4.xml:195537`
  through `:195538` says vector-memory instructions transfer data to and from
  vector registers, but does not distinguish `LOADcnt`, `STOREcnt`,
  `SAMPLEcnt`, or `BVHcnt` classes.
- A few global cache instructions explicitly mention counter effects, such as
  `GLOBAL_INV` using `LOAD_CNT` at `amdgpu_isa_rdna4.xml:54052` through
  `:54053`, `GLOBAL_WB` using `STORE_CNT` at `:54075` through `:54076`, and
  `GLOBAL_WBINV` using `STORE_CNT` at `:55746` through `:55747`; the broader
  Chapter 5.7.1/10.8 class table is not structured.

Impact:

XML-derived consumers can decode the wait instructions and classify many
instructions as VMEM, but cannot infer when VMEM source VGPRs remain protected
from ALU overwrites, which image/buffer/global/flat/scratch/sample/BVH
instructions increment each counter, when counters decrement, how overflow is
avoided, which classes are ordered or unordered with each other, or why
`S_WAIT_LOADCNT`, `S_WAIT_STORECNT`, `S_WAIT_SAMPLECNT`, and `S_WAIT_BVHCNT`
are needed for specific data hazards.

### RDNA4-XML-102: BVH VADDR grouping, A16 footprints, and read/write roles are not structured

Reported by: local audit; Locke subagent.

Manual evidence:

- Chapter 10.9 says the 32-bit and 64-bit BVH ray instructions support `A16`
  for `ray_dir` and `ray_inv_dir`, while `image_bvh_dual_intersect_ray` and
  `image_bvh8_intersect_ray` do not support `A16=1`, at
  `rdna4/README.md:5124`.
- The instruction forms show dynamic address footprints:
  `image_bvh_intersect_ray` uses `vgpr_a[11]` or `vgpr_a[8]` under `A16=1`,
  `image_bvh64_intersect_ray` uses `vgpr_a[12]` or `vgpr_a[9]` under `A16=1`,
  and the dual/BVH8 forms use `vgpr_a[12]` and `vgpr_a[11]`, at
  `rdna4/README.md:5129` through `:5134`.
- Table 63 maps `node_pointer`, `ray_extent`, `ray_origin`, `ray_dir`, and
  `ray_inv_dir` into the five VADDR groups and defines the packed f16 `A16`
  layout at `rdna4/README.md:5154` through `:5173`.
- Table 64 marks the dual/BVH8 ray origin and direction VGPRs as input and
  output parameters, and the node pointers as inputs, at
  `rdna4/README.md:5189` through `:5197`.
- Section 10.9.2 summarizes that BVH instructions use `VADDR0` through
  `VADDR4` as component groups, with `VADDR4` unused in `A16=1` mode at
  `rdna4/README.md:5232` through `:5241`.

XML evidence:

- `ENC_VIMAGE` exposes raw `VADDR0` through `VADDR4` bitfields at
  `amdgpu_isa_rdna4.xml:3086` through `:3126`, but those fields have only
  generic first/second/third/fourth/fifth-source descriptions.
- The BVH instruction entries flatten address VGPRs into one unnamed `FMT_ANY`
  operand: 352 bits for `IMAGE_BVH_INTERSECT_RAY` at
  `amdgpu_isa_rdna4.xml:55422` through `:55425`, 384 bits for
  `IMAGE_BVH64_INTERSECT_RAY` at `:55469` through `:55472`, 384 bits for
  `IMAGE_BVH_DUAL_INTERSECT_RAY` at `:55516` through `:55519`, and 352 bits
  for `IMAGE_BVH8_INTERSECT_RAY` at `:55563` through `:55566`.
- The first two entries are fixed at their `A16=0` address sizes, so the
  `A16=1` 8-VGPR and 9-VGPR forms are not represented as operand-size
  variants.
- The dual and BVH8 address operands are marked `Input="false" Output="true"`
  at `amdgpu_isa_rdna4.xml:55516` and `:55563`, even though the manual says
  most fields are inputs and only ray origin/direction are input/output
  parameters.

Impact:

An XML-only decoder can preserve the raw VADDR selector bits, but cannot infer
which VADDR group supplies each ray field, which groups become packed f16 under
`A16`, that `VADDR4` is unused in `A16=1`, that dual/BVH8 disallow `A16=1`, or
which VGPR ranges are read versus conditionally written.

### RDNA4-XML-103: BVH field restrictions and return-order override are prose-only

Reported by: local audit; Locke subagent.

Manual evidence:

- Chapter 10.9 lists BVH encoding restrictions: `DMASK=0xf`, `D16=0`,
  `R128=1`, `UNRM=1`, `DIM=0`, `LWE=0`, `TFE=0`, and `SSAMP=0`, at
  `rdna4/README.md:5219` through `:5226`.
- The same restriction block says BVH return-order settings are ignored and the
  in-order load return queue is used at `rdna4/README.md:5228`.
- The instruction-definition chapter repeats the restrictions and says they are
  software/compiler requirements not enforced by hardware; improper values may
  be ignored or lead to undefined behavior at `rdna4/README.md:28612` through
  `:28624`.

XML evidence:

- `ENC_VIMAGE` exposes generic `A16`, `D16`, `DIM`, `DMASK`, `R128`, `TFE`,
  `SCOPE`, and `TH` fields at `amdgpu_isa_rdna4.xml:2952` through `:3136`,
  but the BVH instruction entries at `:55405` through `:55581` do not attach
  fixed or legal values to those fields for BVH opcodes.
- Searches for the restriction predicates, the "not enforced by HW" rule, and
  the in-order return queue override found no corresponding structured XML
  metadata outside the generic field definitions and BVH subgroup tags.

Impact:

XML consumers cannot distinguish a legal BVH encoding from an encoding with
unsupported field values, cannot infer that some manual restriction names are
not RDNA4 `ENC_VIMAGE` fields, and cannot model BVH return ordering from the
instruction entries.

### RDNA4-XML-104: BVH resource descriptor and return-mode semantics are opaque

Reported by: local audit; Locke subagent.

Manual evidence:

- Table 65 defines the BVH texture descriptor bit layout, including
  `base_address[47:8]`, reserved zero fields, `sort_triangles_first`,
  `box_sorting_heuristic`, `box_grow_value`, `box_sort_en`, `size[47:6]`,
  `box_node_64B`, `wide_sort_en`, `instance_en`, `pointer_flags`,
  `triangle_return_mode`, and required `type=0x08`, at `rdna4/README.md:5243`
  through `:5267`.
- Table 66 defines the triangle return mode payloads: hit/miss mode returns
  `t_num`, `t_denom`, `triangle_id`, and `hit_status`; barycentric mode returns
  `t_num`, `t_denom`, `I_num`, and `J_num`, at `rdna4/README.md:5269`
  through `:5279`.
- The BVH result prose describes sorted child pointers, wide sorting, triangle
  IDs or barycentrics, instance-node return data, and ShapeID/GeoID payloads at
  `rdna4/README.md:5175`, `:5199` through `:5207`, and `:5213` through
  `:5215`.

XML evidence:

- `FMT_IMG_BVH` is a single unsigned 128-bit `Descriptor` field with the
  description "128-bit BVH resource constant" at
  `amdgpu_isa_rdna4.xml:173077` through `:173096`.
- BVH instruction operands refer to that opaque `FMT_IMG_BVH` resource, for
  example `IMAGE_BVH_INTERSECT_RAY` at `amdgpu_isa_rdna4.xml:55427` through
  `:55431` and `IMAGE_BVH8_INTERSECT_RAY` at `:55568` through `:55572`.
- The BVH instruction descriptions identify the basic operation and node
  pointer size, but do not structure descriptor bitfields, reserved masks,
  sorting controls, instance-node controls, pointer-flag behavior, return-mode
  layouts, ShapeID/GeoID payloads, or bounds-check sizing rules.

Impact:

The XML can identify a BVH resource operand and the four BVH opcodes, but
cannot support descriptor validation, address/bounds reconstruction,
triangle-return-mode decoding, sorted-child interpretation, instance-node
updates, or pointer-flag behavior without the manual prose.

### RDNA4-XML-105: PRT enablement and status-destination rules are prose-only

Reported by: local audit; Ohm subagent.

Manual evidence:

- Chapter 10.10 says PRT supports texture maps with non-resident LODs, and that
  a missing MIP fetch returns an extra DWORD of status in VGPRs indicating
  fetch failure at `rdna4/README.md:5287`.
- The same paragraph says any missing texel causes a texture-cache NACK, a
  nonzero value is written for each failing thread, the value may represent the
  requested LOD, and the shader must allocate, initialize, and check the extra
  VGPR at `rdna4/README.md:5287`.
- Chapter 10.10 states that PRT is enabled when texture-resource
  `MIN_LOD_WARN` is nonzero, normal textures cannot NACK, and PRT writes NACK
  status to `DST_VGPR+Num_VGPRS`, for example VGPR8 after a four-VGPR sample
  return in VGPRs 4-7, at `rdna4/README.md:5289`.
- The image-resource descriptor table defines `min_lod_warn` in bits 177:165
  and `min_lod` in bits 198:186 at `rdna4/README.md:4989` and `:4996`.
- Chapter 3.3.4 says the destination out-of-range check includes the extra PRT
  VGPR and nullifies the fetch if that VGPR would be out of range, regardless
  of whether the texture system actually returns it, at
  `rdna4/README.md:907` through `:910`.

XML evidence:

- `FMT_IMG` is a single unsigned 256-bit descriptor field at
  `amdgpu_isa_rdna4.xml:173057` through `:173075`, so the texture-resource
  `MIN_LOD_WARN` and `MIN_LOD` fields are not structured.
- `ENC_VIMAGE` exposes only a raw `TFE` bit with a short "Texture Fail Enable"
  description at `amdgpu_isa_rdna4.xml:3066` through `:3074`; `ENC_VSAMPLE`
  similarly exposes raw `LWE` and `TFE` bits at `:3288` through `:3296` and
  `:3358` through `:3365`.
- Representative image/sample instruction operands keep only the ordinary data
  destination size: `IMAGE_LOAD` has a 128-bit `VDATA` output at
  `amdgpu_isa_rdna4.xml:54214` through `:54219`, and `IMAGE_SAMPLE` has a
  128-bit `VDATA` output at `:168961` through `:168966`; no conditional
  `TFE`/PRT status operand is present.
- The VIMAGE/VSAMPLE `DMASK` descriptions do preserve the special
  `DMASK==0` rule, including that TFE status is not generated when the fetch
  is dropped, at `amdgpu_isa_rdna4.xml:2995` and `:3267`.
- Searches for `MIN_LOD_WARN`, `min_lod_warn`, `NACK`, `TEXEL_FAIL`,
  `LOD_WARNING`, and `Num_VGPRS` found no structured XML counterpart for
  Chapter 10.10's enable condition, status payload, or destination-range rule.

Impact:

XML consumers can see that image/sample encodings contain status-enable bits,
and they get the narrow `DMASK==0` exception, but cannot infer when a resource
is PRT-capable, where the status VGPR is located after a dynamic data return,
that normal textures cannot NACK, what the status payload means, or that the
extra PRT VGPR participates in destination out-of-range nullification even when
no status is returned.

### RDNA4-XML-106: Flat/global/scratch address modes and aperture semantics are prose-only

Reported by: local audit; Schrodinger subagent.

Manual evidence:

- Chapter 11 says Flat per-thread addresses may resolve to global, scratch,
  LDS, or invalid memory, while not resolving to GPRs or LDS-parameters, at
  `rdna4/README.md:5295` through `:5303`.
- Scratch uses per-wave `SCRATCH_BASE`, and Flat/Scratch implicitly use it
  when private memory is involved, at `rdna4/README.md:5307` and
  `:5410`.
- The field table gives conditional operand meanings: Flat `SADDR` is unused,
  Global `VADDR` is a 64-bit address when `SADDR` is null and a 32-bit byte
  offset when `SADDR` is present, Scratch `VADDR` is a signed byte offset only
  when `SVE=1`, and Scratch `SADDR` is a signed 32-bit component at
  `rdna4/README.md:5318` through `:5325`.
- Flat aperture classification happens before adding `IOFFSET`, invalid
  addresses are routed separately, and LDS-addressed lanes use a logical
  address bounds/remap rule at `rdna4/README.md:5412` through `:5418`.
- Global uses only global memory and no LDS bandwidth, while Scratch is a
  swizzled private memory path with no aperture check, at
  `rdna4/README.md:5422` and `:5431` through `:5435`.

XML evidence:

- XML preserves the raw instruction fields for the three encodings:
  `ENC_VFLAT` at `amdgpu_isa_rdna4.xml:3632` through `:3737`,
  `ENC_VSCRATCH` at `:3786` through `:3891`, and `ENC_VGLOBAL` at
  `:3981` through `:4086`.
- Representative global loads expose `VADDR` as a 64-bit VGPR operand and
  `SADDR` as a 64-bit SGPR operand unconditionally at
  `amdgpu_isa_rdna4.xml:50676` through `:50700`, so the XML operand record
  does not express the `SADDR`-null mode split.
- Representative scratch loads expose both `VADDR` and `SADDR` as 32-bit
  operands at `amdgpu_isa_rdna4.xml:171958` through `:171992`; the raw `SVE`
  bit exists in the encoding, but the operand list does not express that
  `VADDR` is absent when `SVE=0` or that the offsets are signed.
- Representative flat loads describe only the "flat aperture" and an implicit
  `OPR_GPUMEM` operand at `amdgpu_isa_rdna4.xml:47732` through `:47770`;
  they do not model per-lane aperture selection, LDS remapping, invalid
  routing, or `SCRATCH_BASE` as an implicit instruction dependency.
- XML has `HW_REG_WAVE_SCRATCH_BASE_LO/HI` names at
  `amdgpu_isa_rdna4.xml:180732` through `:180738`, but the flat/scratch
  instruction entries do not tie those state registers to the addressing
  semantics.

Impact:

The XML is enough to decode the raw bits, but it is not enough to derive the
Chapter 11 address calculation contract: Flat `SADDR` unusedness, conditional
source widths and presence, signed scratch offsets, per-lane Flat aperture
selection, pre-`IOFFSET` classification, LDS remapping, private scratch-base
use, or the Scratch no-aperture/no-LDS rule.

### RDNA4-XML-107: Chapter 11 return-preOp-only atomic notes are not machine-readable

Reported by: local audit; Schrodinger subagent.

Manual evidence:

- Table 67 marks `FLAT_ATOMIC_COND_SUB_U32` and
  `GLOBAL_ATOMIC_COND_SUB_U32` as supporting only "return preOp" at
  `rdna4/README.md:5388` through `:5389`.
- The same table marks `GLOBAL_ATOMIC_SUB_CLAMP_U32` and
  `GLOBAL_ATOMIC_ORDERED_ADD_B64` as supporting only "return preOp" at
  `rdna4/README.md:5390` through `:5394`.

XML evidence:

- `FLAT_ATOMIC_COND_SUB_U32` says the old flat value is stored only if the
  temporal hint enables atomic return at `amdgpu_isa_rdna4.xml:50122` through
  `:50132`.
- `GLOBAL_ATOMIC_SUB_CLAMP_U32` and `GLOBAL_ATOMIC_COND_SUB_U32` use the same
  generic temporal-hint-controlled return wording at
  `amdgpu_isa_rdna4.xml:52032` through `:52046` and `:53550` through
  `:53564`.
- `GLOBAL_ATOMIC_ORDERED_ADD_B64` has a `VDST` output operand but no
  machine-readable marker for a required return-preOp form at
  `amdgpu_isa_rdna4.xml:54144` through `:54164`.

Impact:

XML consumers cannot tell that the Chapter 11 table gives stricter legality for
these atomic forms. A decoder or emulator generated from XML alone can treat
the no-return encodings as legal or semantically meaningful.

### RDNA4-XML-108: Global write-back cache ops have conflicting subgroup metadata

Reported by: local audit; Schrodinger subagent.

Manual evidence:

- Chapter 11 says `GLOBAL_INV` is tracked with `LOADcnt`, while `GLOBAL_WB`
  and `GLOBAL_WBINV` are tracked with `STOREcnt`, at `rdna4/README.md:5406`.

XML evidence:

- `GLOBAL_WB` says it increments/decrements `STORE_CNT` at
  `amdgpu_isa_rdna4.xml:51746` through `:51747`, but its functional subgroup
  is `LOAD` at `:51760`.
- `GLOBAL_WBINV` says it increments/decrements `STORE_CNT` at
  `amdgpu_isa_rdna4.xml:53528` through `:53529`, but its functional subgroup
  is `LOAD` at `:53542`.

Impact:

An XML consumer using functional subgroups for producer classification will
place the write-back operations in the wrong counter bucket, even though the
description text and manual both identify them as store-counted operations.

### RDNA4-XML-109: `GLOBAL_STORE_ADDTID_B32` describes an SGPR base as immediate

Reported by: local audit.

Manual evidence:

- Chapter 11 says the two ADDTID global instructions use SGPRs and `IOFFSET`,
  and no VGPRs for addressing, at `rdna4/README.md:5424` through `:5427`.

XML evidence:

- `GLOBAL_STORE_ADDTID_B32` says the base address is provided "as an immediate
  value" at `amdgpu_isa_rdna4.xml:51677`.
- The same XML entry exposes `SADDR` as a 64-bit `OPR_SREG` operand at
  `amdgpu_isa_rdna4.xml:51687` through `:51696`, matching the manual and
  contradicting the free-text description.

Impact:

The operand metadata is usable, but tools that trust instruction descriptions
can misdocument or mishandle the ADDTID store addressing form.

### RDNA4-XML-110: Chapter 11.2 addressing-mode formulas are not structured

Reported by: Bacon subreviewer; local audit.

Manual evidence:

- Table 68 selects Scratch `SV`, `SS`, `ST`, and `SVS` from `SVE` plus
  `SADDR`, Flat/Global `GV` from null `SADDR`, Global `GT` by opcode, and LDS
  by opcode at `rdna4/README.md:5448` through `:5458`.
- Chapter 11.2 gives the Global formulas `VGPRU64 + IOFFSETI24`,
  `SGPRU64 + VGPRU32 + IOFFSETI24`, and
  `SGPRU64 + IOFFSETI24 + ThreadID*4` at `rdna4/README.md:5460` through
  `:5462`.
- The same section gives scratch `SV`, `SS`, `SVS`, and `ST` formulas using
  `SCRATCH_BASE` and `SWIZZLE(...)` at `rdna4/README.md:5468` through `:5475`,
  and flat `GV` plus aperture selection at `:5491` through `:5505`.

XML evidence:

- XML preserves only raw field descriptions for `ENC_VFLAT`, `ENC_VSCRATCH`,
  and `ENC_VGLOBAL`, including generic `SADDR`, `SVE`, `VADDR`, and `IOFFSET`
  fields at `amdgpu_isa_rdna4.xml:3632` through `:3737`, `:3786` through
  `:3891`, and `:3981` through `:4086`.
- Representative scratch instructions list both `VADDR` and `SADDR` as
  unconditional 32-bit operands at `amdgpu_isa_rdna4.xml:171958` through
  `:171992`.
- Representative global instructions list both `VADDR` and `SADDR` as
  unconditional operands at `amdgpu_isa_rdna4.xml:50689` through `:50699`, and
  ADDTID instructions omit `VADDR` while using `SADDR` at `:51626` through
  `:51648`.

Impact:

The XML can decode the field bits and broad operands, but not the actual
Chapter 11.2 mode matrix or formula selection. XML-only consumers must already
know when `VADDR` is a 64-bit address versus a 32-bit offset, when a scratch
source is absent, and when `ThreadID*4` is part of the global address.

### RDNA4-XML-111: Flat aperture predicates and per-aperture results are prose-only

Reported by: Bacon subreviewer; local audit.

Manual evidence:

- Flat addressing uses the `GV` mode and performs the aperture test using only
  the base `ADDR`, not `IOFFSET`, at `rdna4/README.md:5491`.
- Chapter 11.2 gives exact `isLDS`, `isScratch`, `isHole`, and `isGlobal`
  predicates based on `ADDR[63:32]`, `ADDR[63:47]`, and
  `SH_MEM_BASES`-derived bases at `rdna4/README.md:5493` through `:5501`.
- The Memory Aperture Query section defines `SHARED_BASE`, `SHARED_LIMIT`,
  `PRIVATE_BASE`, `PRIVATE_LIMIT`, and the hole range at `rdna4/README.md:2573`
  through `:2600`.
- The per-aperture table defines Global, Scratch, LDS, and Hole outcomes,
  including LDS 17-bit truncation, wrap behavior, the only LDS range check, and
  Hole as Memory Violation at `rdna4/README.md:5505` through `:5515`.

XML evidence:

- `ENC_VFLAT` exposes raw field positions and a generic `VADDR` description at
  `amdgpu_isa_rdna4.xml:3632` through `:3737`.
- Representative flat load operands include a 64-bit `VADDR` and an implicit
  memory operand at `amdgpu_isa_rdna4.xml:47735` through `:47755`, but no
  per-lane aperture, LDS, scratch, or invalid-memory outcome metadata.
- XML enumerates `SRC_SHARED_BASE`, `SRC_SHARED_LIMIT`, `SRC_PRIVATE_BASE`,
  and `SRC_PRIVATE_LIMIT` constants, but describes them as `N/A` and does not
  tie them to flat-memory predicates at `amdgpu_isa_rdna4.xml:184211` through
  `:184227`.

Impact:

An XML-derived decoder or emulator cannot recover Flat's pre-`IOFFSET`
classification rule, mixed-lane memory-space routing, LDS U17 wrapping/range
check, scratch remap, or Hole-to-Memory-Violation result from the machine-readable
spec alone.

### RDNA4-XML-112: Scratch swizzle, signed-offset, and ST alignment constraints are prose-only

Reported by: Bacon subreviewer; local audit.

Manual evidence:

- Scratch `SV`, `SS`, `SVS`, and `ST` formulas feed signed source offsets plus
  `IOFFSETI24` into `SWIZZLE(..., ThreadID)` at `rdna4/README.md:5470` through
  `:5475`.
- The combined offsets inside `SWIZZLE()` must be non-negative, and SGPR/VGPR
  offsets are signed 32-bit byte offsets at `rdna4/README.md:5477` through
  `:5479`.
- In Scratch `ST` mode, `IOFFSET` must be aligned to payload size: 4-byte
  aligned for one DWORD and 16-byte aligned for four DWORDs at
  `rdna4/README.md:5481`.

XML evidence:

- `ENC_VSCRATCH` describes `IOFFSET` as a 24-bit signed offset and says `SVE`
  plus `SADDR` determines four scratch modes at `amdgpu_isa_rdna4.xml:3800`
  through `:3801` and `:3850` through `:3851`.
- The `NULL` scalar operand value says that scratch addressing does not use an
  SGPR when `SADDR` is null at `amdgpu_isa_rdna4.xml:181457` through `:181459`.
- Searches found no XML counterpart for scratch `SWIZZLE`, the non-negative
  combined-offset rule, the signedness of scratch SGPR/VGPR source offsets, or
  ST payload-size alignment.

Impact:

XML-only consumers can identify the raw scratch sources, but cannot derive the
actual private-memory lane mapping or legality checks for negative combined
offsets and unaligned thread-private scratch transfers.

### RDNA4-XML-113: `ENC_VGLOBAL` overstates `VADDR` width for GVS mode

Reported by: Bacon subreviewer.

Manual evidence:

- Chapter 11.2 says Global `GV` uses `VGPRU64 + IOFFSETI24`, while Global
  `GVS` uses `SGPRU64 + VGPRU32 + IOFFSETI24` at
  `rdna4/README.md:5460` through `:5462`.

XML evidence:

- Representative ordinary global load operands list `VADDR` as a 64-bit VGPR
  and `SADDR` as a 64-bit SGPR at `amdgpu_isa_rdna4.xml:50689` through
  `:50699`.
- Representative global store operands likewise list `VADDR` as a 64-bit VGPR
  even when `SADDR` is present at `amdgpu_isa_rdna4.xml:51001` through
  `:51017`.

Impact:

Tools that use XML operand widths literally can read, display, or depend on
two VGPRs for GVS encodings where the manual defines only a 32-bit VGPR byte
offset plus the 64-bit SGPR base.

### RDNA4-XML-114: Scratch/global `VADDR` field descriptions call every address flat

Reported by: Bacon subreviewer.

Manual evidence:

- Chapter 11.2 distinguishes Flat `VADDR` as a 64-bit per-thread address,
  Global `VADDR` as either a 64-bit address or 32-bit offset depending on
  `SADDR`, and Scratch `VADDR` as a signed byte offset selected by `SVE`, at
  `rdna4/README.md:5460` through `:5479` and `:5491`.

XML evidence:

- The `ENC_VSCRATCH` `VADDR` field description says "Source flat address VGPR"
  at `amdgpu_isa_rdna4.xml:3870` through `:3871`.
- The `ENC_VGLOBAL` `VADDR` field description also says "Source flat address
  VGPR" at `amdgpu_isa_rdna4.xml:4065` through `:4066`.

Impact:

The XML's free-text field descriptions blur three different address meanings,
making it easy for generated documentation or metadata-driven tools to treat
global offsets and scratch offsets as full flat addresses.

### RDNA4-XML-115: Chapter 11.3 memory-error policy is prose-only

Reported by: local audit.

Manual evidence:

- Chapter 11.3 says Cache and LDS can report bad-address errors for invalid
  addresses outside any aperture, writes to read-only global pages, misaligned
  data, and LDS out-of-range addresses outside `[0, LDS_SIZE-1]`, at
  `rdna4/README.md:5517` through `:5526`.
- The bad-address policy says stores outside the valid range do not store,
  reads return zero, invalid-address aperture checking happens before adding
  address offsets, and all other checks are performed after offsets are added
  at `rdna4/README.md:5527`.
- Addressing errors from either LDS or VMEM set the wave's `MEMVIOL` bit and
  cause an exception/trap at `rdna4/README.md:5529`.

XML evidence:

- The Flat/Global/Scratch encodings expose raw address fields but no
  per-instruction memory-error policy at `amdgpu_isa_rdna4.xml:3632` through
  `:3737`, `:3786` through `:3891`, and `:3981` through `:4086`.
- XML has only generic memory-operand categories for this purpose:
  `OPR_DSMEM` "reads or writes DS memory" at `amdgpu_isa_rdna4.xml:180619`
  through `:180625`, and `OPR_GPUMEM` "reads or writes GPU memory" at
  `:180641` through `:180642`.
- Representative concrete entries such as `FLAT_LOAD_B32`,
  `FLAT_STORE_B32`, and `DS_LOAD_B32` expose ordinary operands at
  `amdgpu_isa_rdna4.xml:47732` through `:47755`, `:48004` through `:48028`,
  and `:43332` through `:43356`, with no conditional zero-return or
  suppressed-store result metadata.
- XML does define nominal `HW_REG_WAVE_EXCP_FLAG_PRIV`,
  `HW_REG_WAVE_EXCP_FLAG_USER`, and `HW_REG_WAVE_TRAP_CTRL` register names at
  `amdgpu_isa_rdna4.xml:180717` through `:180728`, and a generic `TRAP`
  functional group at `:194401` through `:194402`. `S_TRAP` itself says only
  "Enter the trap handler" at `:34714` through `:34715`; XML does not connect
  Flat/Global/Scratch memory errors to `MEMVIOL` or trap delivery.
- The RDNA4 XML has isolated memory-error prose for other instruction families,
  such as scalar-buffer negative `IOFFSET` producing `MEMVIOL` at
  `amdgpu_isa_rdna4.xml:693` through `:695`, but searches found no equivalent
  Chapter 11.3 policy metadata for Flat, Global, or Scratch.

Impact:

XML-only consumers cannot derive which Flat/Global/Scratch bad addresses return
zero, suppress stores, set `MEMVIOL`, or trap, nor can they distinguish the
manual's pre-offset invalid-aperture check from post-offset alignment,
read-only-page, and LDS range checks.

### RDNA4-XML-116: D16 load partial-destination preservation is not structured

Reported by: local audit; Curie the 2nd subagent.

Manual evidence:

- Chapter 11.4 says `"D16"` instructions use only 16 bits of the VGPR,
  `D16_HI` instructions read or write only the high 16 bits, and `D16`
  instructions use the low 16 bits, at `rdna4/README.md:5535`.
- The Chapter 11 opcode inventory includes Flat, Global, and Scratch
  `D16`/`D16_HI` load forms at `rdna4/README.md:5331` through `:5344`.

XML evidence:

- XML free text names the selected half for representative Flat D16 loads:
  `FLAT_LOAD_D16_U8` stores into the low 16 bits at
  `amdgpu_isa_rdna4.xml:50707` through `:50711`, and
  `FLAT_LOAD_D16_HI_U8` stores into the high 16 bits at
  `:50839` through `:50843`.
- Those same entries still expose `VDST` as an ordinary 32-bit output-only
  operand: `Input="false" Output="true"` and `OperandSize 32` at
  `amdgpu_isa_rdna4.xml:50718` through `:50723` and `:50850` through `:50855`.
- Global and Scratch D16 load entries have the same shape, for example
  `GLOBAL_LOAD_D16_U8` / `GLOBAL_LOAD_D16_HI_U8` at
  `amdgpu_isa_rdna4.xml:53564` through `:53579` and `:53714` through `:53729`,
  and `SCRATCH_LOAD_D16_U8` / `SCRATCH_LOAD_D16_HI_U8` at
  `:173634` through `:173649` and `:173784` through `:173799`.
- The raw VFLAT/VGLOBAL field descriptions only identify `VDST` as
  destination data and `VSRC` as source data at `amdgpu_isa_rdna4.xml:4680`
  through `:4697` and `:5090` through `:5107`.

Impact:

The XML text is enough for a human to see low-versus-high placement, but the
operand metadata describes a full 32-bit write-only VGPR. XML-only dataflow or
code-generation consumers cannot derive the old-destination read/merge needed
to preserve the other 16 bits.

### RDNA4-XML-117: Atomic return `VDST` is modeled as an unconditional output

Reported by: Curie the 2nd subagent; local audit.

Manual evidence:

- The Chapter 11 field table says `VDST` is the destination for data returned
  by loads or atomics that return the pre-op value, and says `TH` controls
  whether atomics return the pre-op value, at `rdna4/README.md:5320` through
  `:5322`.
- Chapter 11.4 repeats that `VDST` holds return data "if any" at
  `rdna4/README.md:5533`.
- The common RDNA4 temporal-hint prose defines atomic `TH[0]` as the
  return/no-return selector at `rdna4/README.md:1784` through `:1795`.

XML evidence:

- Representative atomic descriptions mention the conditional return, for
  example `FLAT_ATOMIC_ADD_U32` stores the original flat value into a vector
  register iff the temporal hint enables atomic return at
  `amdgpu_isa_rdna4.xml:51173`.
- The same XML entry nevertheless marks `VDST` as
  `Input="false" Output="true"` with `OperandSize 32` at
  `amdgpu_isa_rdna4.xml:51180` through `:51185`.
- This is a Chapter 11 data-shape consequence of the broader TH policy gap
  recorded in `RDNA4-XML-026`; the instruction operand metadata does not expose
  the conditional destination.

Impact:

XML-only def-use consumers can over-report a `VDST` definition for
non-returning atomics, or require hand-written TH logic outside the operand
model to suppress it.

### RDNA4-XML-118: Block VGPR transfer mask and sparse holes are not structured

Reported by: local audit; Hooke the 2nd subagent.

Manual evidence:

- Chapter 11.5 says block instructions move up to 32 consecutive VGPRs to or
  from memory at `rdna4/README.md:5539`.
- It says `M0` carries the bitmask of which VGPRs to load/store, with the LSB
  selecting the first VGPR, at `rdna4/README.md:5549`.
- It says skipped VGPRs leave skipped memory locations rather than compacting
  the memory block, and that the block address advances as if `IOFFSET`
  increased by 4 bytes for every VGPR regardless of the `M0` bit, at
  `rdna4/README.md:5547` and `:5553`.
- It states that `M0==0` transfers no data at `rdna4/README.md:5564`.
- The per-opcode pseudocode for `SCRATCH_*_BLOCK` and `GLOBAL_*_BLOCK` repeats
  the `for i in 0:31` / `if M0[i]` rule and uses `i * 4` for the memory slot at
  `rdna4/README.md:29999` through `:30021` and `:30670` through `:30692`.

XML evidence:

- The four block entries expose full 1024-bit VGPR and memory operands plus an
  implicit 32-bit `OPR_SDST_M0` operand at
  `amdgpu_isa_rdna4.xml:55959` through `:55985`,
  `:56017` through `:56037`, `:174042` through `:174068`, and `:174100`
  through `:174120`.
- The concrete XML descriptions only say "Load a block of data" or "Store a
  block of data" at `amdgpu_isa_rdna4.xml:55952`, `:56004`,
  `:174035`, and `:174087`.
- The generic `OPR_SDST_M0` operand class records only that selector value 125
  is `M0`, at `amdgpu_isa_rdna4.xml:182611`; it does not identify this operand
  as a per-DWORD transfer mask or define the LSB-first, skip, or zero-mask
  behavior.

Impact:

XML-only consumers can see that `M0` is present and that the maximum transfer is
1024 bits, but cannot derive the sparse per-DWORD transfer contract, the
no-compaction memory holes, the `IOFFSET += 4` stepping through skipped slots,
or the `M0==0` no-transfer rule.

### RDNA4-XML-119: Block-specific address-mode exclusions are not attached to block op entries

Reported by: local audit; Hooke the 2nd subagent.

Manual evidence:

- Chapter 11.5 says global block operations support `GV` and `GVS` but not
  `GT`, while scratch block operations support `SS`, `SV`, and `SVS` but not
  `ST`, at `rdna4/README.md:5551`.
- The same section says block addressing advances one DWORD slot per VGPR
  position, independent of skipped transfers, at `rdna4/README.md:5553`.

XML evidence:

- The block entries use the generic `ENC_VGLOBAL` and `ENC_VSCRATCH`
  encodings at `amdgpu_isa_rdna4.xml:55955`, `:56007`, `:174038`, and
  `:174090`.
- The generic `ENC_VGLOBAL` description still lists the broader global address
  modes, including `GT`, at `amdgpu_isa_rdna4.xml:4989`.
- The generic `ENC_VSCRATCH` description still lists the broader scratch
  address modes, including `ST`, at `amdgpu_isa_rdna4.xml:4760`.
- XML has raw `IOFFSET`, `VADDR`, and `SADDR` field metadata for those
  encodings at `amdgpu_isa_rdna4.xml:4600` through `:4638`, `:4670` through
  `:4698`, `:4816` through `:4884`, and `:5040` through `:5108`, but no
  opcode-specific rule for block-mode exclusions or per-DWORD address stepping.
- Manual prose uses `GVS`, while the generic XML description labels the same
  scalar-plus-vector-looking global mode as `GS` at
  `amdgpu_isa_rdna4.xml:4984`; the XML block entries do not resolve that naming
  difference.

Impact:

An XML-only legality checker can inherit ordinary Global/Scratch address modes
for block instructions and accept unsupported `GT` or `ST` forms, while a
semantic generator still needs manual-specific logic for sparse block address
stepping.

### RDNA4-XML-120: Block-load source/destination VGPR relation is prose-only

Reported by: Hooke the 2nd subagent; local audit.

Manual evidence:

- Chapter 11.5 states that block loads must load data into their own source
  VGPRs at `rdna4/README.md:5555`.

XML evidence:

- `GLOBAL_LOAD_BLOCK` exposes `VDST` as an output VGPR and `VADDR`/`SADDR` as
  input operands at `amdgpu_isa_rdna4.xml:55959` through `:55975`.
- `SCRATCH_LOAD_BLOCK` exposes the same independent `VDST`, `VADDR`, and
  `SADDR` operand roles at `amdgpu_isa_rdna4.xml:174042` through `:174062`.
- No XML operand relation or legality predicate ties `VDST` to a source VGPR
  field or states how the manual's "own source VGPRs" requirement should be
  checked.

Impact:

The manual leaves a block-load operand relation that cannot be recovered from
the XML operand list. XML-only assemblers, validators, and dataflow consumers
need external prose knowledge to decide whether a given `VDST`/source-VGPR
combination is legal.

### RDNA4-XML-121: Block VGPR per-DWORD out-of-range behavior is absent

Reported by: local audit; Hooke the 2nd subagent.

Manual evidence:

- Chapter 11.5.1 says out-of-range address VGPRs follow ordinary Global and
  Scratch rules at `rdna4/README.md:5570`.
- It then gives block-specific per-DWORD data-register behavior: load
  destinations are individually checked and out-of-range destinations are
  ignored, while store sources are individually checked and out-of-range
  sources read from `VGPR0`, at `rdna4/README.md:5572` through `:5582`.

XML evidence:

- The block entries model data as ordinary 1024-bit `OPR_VGPR` operands at
  `amdgpu_isa_rdna4.xml:55959` through `:55964`, `:56017` through `:56022`,
  `:174042` through `:174047`, and `:174100` through `:174105`.
- The generic `OPR_VGPR` operand class describes vector GPRs at
  `amdgpu_isa_rdna4.xml:194073`, but does not encode per-DWORD range checks,
  ignored destination writes, or `VGPR0` substitution for out-of-range store
  sources.

Impact:

XML-only consumers cannot model the mixed in-range/out-of-range behavior of a
single block operation. Treating the 1024-bit operand as one ordinary VGPR span
either over-rejects valid partially in-range block operations or misses the
hardware's `VGPR0` fallback for store source holes.

### RDNA4-XML-122: WMMA transpose-load `EXEC` contract is not encoded

Reported by: Mill the 2nd subagent; local audit.

Manual evidence:

- Chapter 11.6.2 says `GLOBAL_LOAD_TR_B128` and `GLOBAL_LOAD_TR_B64` act like
  `S_NOP` when `EXEC==0`; otherwise `EXEC` must be all ones or behavior is
  undefined, at `rdna4/README.md:5610`.

XML evidence:

- The XML entries for `GLOBAL_LOAD_TR_B128` and `GLOBAL_LOAD_TR_B64` expose
  only generic instruction flags and VGLOBAL operands at
  `amdgpu_isa_rdna4.xml:56106` through `:56145` and `:56153` through `:56192`.
- Those entries do not mention `EXEC`, zero-`EXEC` `S_NOP` behavior, or the
  all-ones precondition.

Impact:

XML-only validators and emulators cannot distinguish these transpose loads from
ordinary predicated global loads. They need manual prose to reject, diagnose, or
otherwise special-case nonzero partial-`EXEC` use, and to know that zero-`EXEC`
is architecturally an `S_NOP` for these opcodes.

### RDNA4-XML-123: WMMA transpose-load lane mapping and wave-size behavior are prose-only

Reported by: Mill the 2nd subagent; local audit.

Manual evidence:

- Chapter 11.6.1 gives row-major and column-major matrix memory address
  formulas at `rdna4/README.md:5594` through `:5603`.
- Chapter 11.6.2 says each wave32 lane loads contiguous memory for a specific
  matrix element group and gives a lane-to-matrix-element diagram at
  `rdna4/README.md:5612` through `:5618`.
- The `GLOBAL_LOAD_TR_B128` row says wave32 loads into four consecutive VGPRs,
  while wave64 loads into two consecutive VGPRs, uses only addresses from lanes
  0-31, and ignores addresses in lanes 32-63, at `rdna4/README.md:5622`.
- The `GLOBAL_LOAD_TR_B64` row says wave32 loads into two consecutive VGPRs,
  while wave64 loads into one VGPR with the same lane 0-31 address rule, at
  `rdna4/README.md:5623`.
- The instruction selection table maps memory order, element size, wave size,
  and desired VGPR layout to ordinary versus transpose global load opcodes at
  `rdna4/README.md:5627` through `:5638`.

XML evidence:

- XML descriptions say the two opcodes load and transpose a 16x16 matrix, but
  provide no lane map, row/column formula, wave32/wave64 output rule, or
  instruction-selection table at `amdgpu_isa_rdna4.xml:56113` through `:56114`
  and `:56160` through `:56161`.
- XML models `GLOBAL_LOAD_TR_B128` with a fixed 128-bit `VDST` and 128-bit
  implicit memory operand at `amdgpu_isa_rdna4.xml:56121` through `:56142`.
- XML models `GLOBAL_LOAD_TR_B64` with a fixed 64-bit `VDST` and 64-bit
  implicit memory operand at `amdgpu_isa_rdna4.xml:56168` through `:56189`.
- The broader WMMA layout formulas are already tracked in `RDNA4-XML-049`; this
  entry records the Chapter 11.6 global-load-transpose-specific lane sourcing,
  memory-order selection, and wave-size-dependent destination width rules.

Impact:

The XML is enough to identify the opcodes and their nominal wave32 data size,
but not enough to generate or validate the actual transpose layout. In
particular, an XML-only implementation can wrongly read addresses from wave64
lanes 32-63 or write the wave32 per-lane destination width in wave64 mode.

### RDNA4-XML-124: LDS topology, bank conflicts, and hardware-atomic placement are prose-only

Reported by: Huygens the 2nd subagent; local audit.

Manual evidence:

- Chapter 12 says LDS bandwidth relies on simultaneous memory-bank access and
  that indexed and atomic same-bank conflicts are serialized by hardware, at
  `rdna4/README.md:5646`.
- Chapter 12.1 says a WGP has 128kB LDS split into 64 DWORD-wide banks, further
  divided into two 32-bank sets; each bank is a 512x32 two-port RAM with one
  read and one write port per clock, at `rdna4/README.md:5654`.
- Chapter 12.1 says LDS atomics execute in LDS hardware rather than ALUs, at
  `rdna4/README.md:5656`.
- The hardware overview also says the LDS has 64 banks, each with 512 entries
  of 4 bytes, and 64 integer atomic units, at `rdna4/README.md:528`.

XML evidence:

- `OPR_DSMEM` only describes an operation that reads or writes DS memory at
  `amdgpu_isa_rdna4.xml:181735`.
- DS instructions expose ordinary DS-memory operands and the broad
  `VMEM` / `DATA_SHARE` functional subgroup, for example `DS_ADD_U32` at
  `amdgpu_isa_rdna4.xml:43970` through `:44001`, `DS_STORE_B32` at
  `:44580` through `:44606`, and `DS_LOAD_B32` at `:46103` through `:46129`.
- The XML has no structured bank-count, bank-set, bank-width, port-count,
  bank-conflict, or LDS-atomic-unit metadata in the checked LDS operand and
  instruction records.

Impact:

The XML can identify DS memory effects but cannot drive a bank-aware scheduler,
timing model, or atomic-hardware classification. An XML-only consumer cannot
recover the bank-conflict serialization rule or the physical LDS topology from
the instruction records.

### RDNA4-XML-125: LDS allocation registers and CU/WGP placement constraints are not structured

Reported by: Huygens the 2nd subagent; local audit.

Manual evidence:

- Chapter 12 says LDS space is allocated per workgroup or wave and recorded in
  dedicated non-shader-writable `LDSbase`/size registers that restrict all LDS
  accesses to owned space, at `rdna4/README.md:5650`.
- Chapter 12.1 says a single workgroup can request up to 64kB, at
  `rdna4/README.md:5654`.
- Chapter 12.1.1 says workgroups dispatch in CU or WGP mode, at
  `rdna4/README.md:5660`.
- The referenced workgroup section says workgroups may share up to 64kB LDS,
  CU mode keeps the allocation on the same LDS side as the CU, WGP mode uses a
  large contiguous LDS allocation with the same maximum allocation size, and
  `DS_PARAM_LOAD` / `DS_DIRECT_LOAD` are not supported in WGP mode, at
  `rdna4/README.md:620` through `:632`.
- Chapter 3.3.5 adds that LDS allocations are 0-64kB in 1024-byte blocks, do
  not wrap, CU-mode accesses cannot cross to the other side, and WGP-mode
  allocations may straddle the CU0/CU1 boundary, at `rdna4/README.md:931`
  through `:941`.

XML evidence:

- Searching `amdgpu_isa_rdna4.xml` for `WGP_MODE`, `GROUP_SEGMENT`, and
  `workgroup` yields no matching LDS allocation schema.
- The only checked LDS-allocation-like HWREG enum is
  `hw_reg_wave_lds_alloc`, described only as "Wave LDS allocation," at
  `amdgpu_isa_rdna4.xml:181803`.
- Individual DS entries expose generic `OPR_DSMEM` operands and `DATA_SHARE`
  subgroup labels but do not encode `LDSbase`/size state, non-writability,
  64kB request limits, 1024-byte allocation granularity, CU-side crossing
  restrictions, WGP straddling, or VDSDIR WGP-mode exclusion.

Impact:

XML-only dispatch validators and emulators cannot derive the LDS allocation
contract from the machine-readable ISA. They need manual prose or ABI-side
knowledge to enforce per-workgroup LDS limits and to map LDS references through
the correct CU/WGP allocation state.

### RDNA4-XML-126: Parameter-load availability and readiness gating are prose-only

Reported by: Sartre the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.2 says pixel waves preload vertex-attribute data into LDS before
  the wave starts, at `rdna4/README.md:5684`.
- Chapter 12.2 says pixel shader waves stall if `DS_DIRECT_LOAD` or
  `DS_PARAM_LOAD` issues before the LDS parameter data is ready, at
  `rdna4/README.md:5688`.
- Chapter 12.2.1 says parameter loads are available in CU mode only and are not
  available in WGP mode, at `rdna4/README.md:5704`.
- The STATUS table defines the pixel-shader-only `LDS_PARAM_RDY` bit at
  `rdna4/README.md:1035`.

XML evidence:

- `DS_PARAM_LOAD` is an ordinary VDSDIR instruction with a default encoding at
  `amdgpu_isa_rdna4.xml:49933` through `:49941`.
- The VDSDIR encoding default condition is unconditional at
  `amdgpu_isa_rdna4.xml:3199`.
- `OPR_HWREG` only enumerates `hw_reg_wave_status` as a generic hardware
  register at `amdgpu_isa_rdna4.xml:181788` through `:181790`.
- Searching the checked RDNA4 XML for `LDS_READY` and `LDS_PARAM_RDY` yields no
  matches.

Impact:

The XML can identify the opcode, but cannot tell a consumer that the operation
is pixel/CU-mode-specific or that issue must stall until the parameter LDS state
is marked ready.

### RDNA4-XML-127: `DS_PARAM_LOAD` M0 layout and LDS address derivation are not structured

Reported by: Sartre the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.2.1 defines the implicit M0 layout as
  `{1'b0, new_prim_mask[15:1], lds_param_offset[15:0]}`, at
  `rdna4/README.md:5720`.
- Chapter 12.2.1 says M0 must be initialized before issuing the instruction, at
  `rdna4/README.md:5722`.
- Chapter 12.2.1 defines the omitted/implied first-quad primitive bit and
  `new_prim_mask` meaning at `rdna4/README.md:5724` through `:5726`.
- Chapter 12.2.1 says `lds_param_offset` bits `[6:0]` must be zero and gives
  the full LDS address formula, at `rdna4/README.md:5730` through `:5736`.
- The M0 register table repeats the DS_PARAM_LOAD M0 shape and the Wave32 mask
  subset at `rdna4/README.md:1093`.

XML evidence:

- `DS_PARAM_LOAD` only marks a generic implicit `OPR_SDST_M0` source at
  `amdgpu_isa_rdna4.xml:49961` through `:49965`.
- `OPR_SDST_M0` is the generic M0 operand description at
  `amdgpu_isa_rdna4.xml:182611` through `:182619`.
- The instruction description mentions `NewPrimMask` at
  `amdgpu_isa_rdna4.xml:49937`, but does not encode M0 bit partitions,
  Wave32 truncation, offset alignment, primitive derivation, or the address
  formula.

Impact:

An XML-only implementation cannot compute the LDS address or validate M0
encoding/alignment rules for parameter loads.

### RDNA4-XML-128: Per-quad lane-fill, EXEC, and out-of-range destination behavior are incomplete

Reported by: Sartre the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.1.2 says parameter interpolation loads place three parameters
  into three VGPR lanes and write zero to the fourth lane, at
  `rdna4/README.md:5676` through `:5680`.
- Chapter 12.2.1 says the three parameters are spread into lanes 0, 1, and 2
  of each quad, at `rdna4/README.md:5706` through `:5708`.
- Chapter 12.2.1 says a destination VGPR out of range still performs the load
  but forces `EXEC` to zero, at `rdna4/README.md:5742`.
- Chapter 12.2.1 says `DS_PARAM_LOAD` and `DS_DIRECT_LOAD` use `EXEC` per
  quad and write all four threads in a quad when any pixel in that quad is
  enabled, at `rdna4/README.md:5744`.
- Chapter 12.2.1 says the load is skipped when `EXEC==0` and `EXPcnt==0`, at
  `rdna4/README.md:5762`.

XML evidence:

- The `DS_PARAM_LOAD` description says data is placed into lanes 0-3 "as
  follows," at `amdgpu_isa_rdna4.xml:49937`, but no lane-fill table is encoded
  after the description.
- The operand list exposes only `VDST`, `ATTR`, implicit `OPR_DSMEM`, and
  implicit M0 at `amdgpu_isa_rdna4.xml:49944` through `:49965`.
- `VDST` is a normal output `OPR_VGPR` at
  `amdgpu_isa_rdna4.xml:49944` through `:49949`, and `OPR_VGPR` is generic at
  `amdgpu_isa_rdna4.xml:194073` through `:194083`.

Impact:

The XML does not contain enough information to reproduce the quad-wide write
mask, the fourth-lane zero, the all-quad write rule, the zero-EXEC skip rule,
or the destination-out-of-range side effect.

### RDNA4-XML-129: 16-bit parameter packing mode is absent from `DS_PARAM_LOAD` metadata

Reported by: Sartre the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.2.1 says 16-bit parameters can pack two attributes into one
  DWORD, describes an alternate unpacked mode, and says the same
  `DS_PARAM_LOAD` returns the packed DWORD containing two attributes, at
  `rdna4/README.md:5748`.
- Chapter 12.2.1 says interpolation then uses packed mixed-precision FMA, DPP,
  and OPSEL, while barycentric coordinates are still 32-bit, at
  `rdna4/README.md:5750`.
- The instruction definition says FP16 loads pack attribute `2*ATTR` in the low
  16 bits and `2*ATTR+1` in the high 16 bits, at
  `rdna4/README.md:25210` through `:25212`.

XML evidence:

- The `DS_PARAM_LOAD` destination is simply a 32-bit `FMT_NUM_B32` VGPR at
  `amdgpu_isa_rdna4.xml:49944` through `:49949`.
- No checked `DS_PARAM_LOAD` operand or instruction metadata records the
  attribute-pair packing, alternate unpacked mode, or the FP16 attribute-number
  mapping.

Impact:

The XML records a 32-bit destination but not the data-dependent meaning of the
two 16-bit halves. A consumer cannot derive the packed FP16 parameter contract
from the instruction metadata.

### RDNA4-XML-130: Parameter-load EXPcnt ordering is not attached to the instruction

Reported by: Sartre the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.2 says LDS parameter loads use `EXPcnt` for outstanding reads and
  decrement `EXPcnt` when the data arrives in VGPRs, at `rdna4/README.md:5686`.
- Chapter 12.2.1 says `DS_DIRECT_LOAD` and `DS_PARAM_LOAD` use `EXPcnt`, need
  `S_WAIT_EXPCNT` before use, and may complete out of order with exports, at
  `rdna4/README.md:5756` through `:5766`.
- Chapter 12.2.1 says VINTERP has a `wait_EXPcnt` field for the load-use
  hazard, at `rdna4/README.md:5760`.
- Chapter 12.2.1 defines the VDSDIR `WAITVDST` and `WAITVMVS` source-read WAR
  controls at `rdna4/README.md:5715` through `:5719` and `:5768` through
  `:5772`.

XML evidence:

- XML has a separate `S_WAIT_EXPCNT` instruction at
  `amdgpu_isa_rdna4.xml:38729` through `:38740`.
- XML has a VINTERP `WAIT_EXP` field at
  `amdgpu_isa_rdna4.xml:3179` through `:3188`.
- VDSDIR raw fields include `WAIT_VA_VDST` and `WAIT_VM_VSRC` at
  `amdgpu_isa_rdna4.xml:3266` through `:3280`.
- The `DS_PARAM_LOAD` entry at `amdgpu_isa_rdna4.xml:49933` through `:49973`
  does not carry a structured producer/counter/ordering annotation. The broader
  wait-counter producer taxonomy gap is also recorded in `RDNA4-XML-044`.

Impact:

The XML carries some wait fields in nearby encodings, but not the full
instruction-level dependency contract that makes parameter loads `EXPcnt`
producers and constrains their ordering relative to exports and consumers.

### RDNA4-XML-144: VINTERP fixed-DPP interpolation semantics are not structured

Manual evidence:

- Chapter 12.3 says VINTERP performs FMA-based interpolation with built-in DPP
  and `fetch invalid = 1`, so neighboring lanes with `EXEC==0` are read rather
  than replaced by zero, at `rdna4/README.md:5774` through `:5787`.
- The VINTERP field table defines opcode-specific P10/P2 behavior, `WAIT_EXP`,
  OPSEL source/destination half selection, clamp, and the rule that OPSEL must
  be zero for non-16-bit operands/results at `rdna4/README.md:5792` through
  `:5817`.
- The 16-bit subsection and detailed instruction definitions specify the fixed
  DPP8 lane selectors, fused FMA formulas, RTZ variants, OPSEL source/dest
  roles, and no-exception/no-forwarding restrictions at
  `rdna4/README.md:5819` through `:5865` and `rdna4/README.md:25062`
  through `:25194`.

XML evidence:

- The XML `ENC_VINTERP` description records no-DPP/no-literal and forwarding
  restrictions, and the raw fields include `WAIT_EXP`, `OPSEL`, `CLAMP`,
  `NEG`, sources, and destination at `amdgpu_isa_rdna4.xml:3073` through
  `:3188`.
- The six VINTERP instruction entries carry opcodes and F16/F32 operand sizes
  at `amdgpu_isa_rdna4.xml:57958` through `:58233`.
- Those records do not structure the fixed DPP8 lane-select tables, implicit
  `fetch invalid = 1` behavior, fused-FMA formula/order, RTZ result rounding,
  OPSEL legality by operand/result width, no-exception behavior, or the
  opcode-specific distinction between P10 sources (`S0` and `S2`) and P2
  sources (`S0` only).

Impact:

XML-only consumers can decode VINTERP and know the basic operand widths, but
cannot derive the actual interpolation dataflow or legality/rounding details
without manual prose.

### RDNA4-XML-131: `DS_DIRECT_LOAD` CU-only and readiness gating are prose-only

Reported by: Anscombe the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.4 says direct access is allowed only in CU mode, not WGP mode, at
  `rdna4/README.md:5869`.
- Chapter 12.2 says pixel shader waves stall if `DS_DIRECT_LOAD` or
  `DS_PARAM_LOAD` issues before LDS parameter data is ready, at
  `rdna4/README.md:5688`.
- The STATUS table defines `LDS_PARAM_RDY` at `rdna4/README.md:1035`.

XML evidence:

- `DS_DIRECT_LOAD` uses the VDSDIR default encoding and opcode 1 at
  `amdgpu_isa_rdna4.xml:49982` through `:50014`.
- The VDSDIR default condition is unconditional at
  `amdgpu_isa_rdna4.xml:3199`.
- `OPR_HWREG` only enumerates generic `hw_reg_wave_status` at
  `amdgpu_isa_rdna4.xml:181788` through `:181790`.
- Searching the checked RDNA4 XML for `LDS_READY` and `LDS_PARAM_RDY` yields no
  matches.

Impact:

The XML can identify the direct-load opcode but cannot derive that it is
CU-mode-only or that pixel waves may stall on LDS readiness before issue.

### RDNA4-XML-132: `DS_DIRECT_LOAD` whole-quad broadcast and EXPcnt behavior are not structured

Reported by: Anscombe the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.4 says `DS_DIRECT_LOAD` reads one LDS DWORD, returns it to one
  VGPR, broadcasts it to all active lanes, evaluates `EXEC` per quad, and
  writes all four pixels in a quad if any pixel in that quad is enabled, at
  `rdna4/README.md:5871`.
- Chapter 12.4 says `DS_DIRECT_LOAD` uses `EXPcnt` to track completion, at
  `rdna4/README.md:5871`.
- Chapter 5.7.1 says `DS_PARAM_LOAD` and `DS_DIRECT_LOAD` increment `EXPcnt`
  on issue, decrement when complete, are ordered with each other, and are
  unordered with exports, at `rdna4/README.md:2202` through `:2209`.
- Chapter 12.2.1 says direct and parameter loads are skipped when `EXEC==0`
  and `EXPcnt==0`, at `rdna4/README.md:5762`.

XML evidence:

- The XML description says `DS_DIRECT_LOAD` reads a single 32-bit LDS value to
  all lanes, at `amdgpu_isa_rdna4.xml:49986`.
- The operand list exposes only `VDST`, implicit `OPR_DSMEM`, and implicit M0
  at `amdgpu_isa_rdna4.xml:49993` through `:50014`.
- XML has separate `S_WAIT_EXPCNT` and wait-count operand records at
  `amdgpu_isa_rdna4.xml:38729` through `:38740` and
  `amdgpu_isa_rdna4.xml:195359` through `:195365`, but the `DS_DIRECT_LOAD`
  entry has no structured counter-producer, ordering, per-quad `EXEC`, or
  zero-EXEC skip metadata.

Impact:

XML-only consumers can know the operation is a broadcast load in prose, but
cannot generate the quad-wide write mask or the direct-load `EXPcnt` ordering
contract from structured metadata.

### RDNA4-XML-133: `DS_DIRECT_LOAD` M0 datatype and extension rules are description-only

Reported by: Anscombe the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.4 defines `M0[15:0]` as a DWORD-aligned byte address and
  `M0[18:16]` as the data type selector, at `rdna4/README.md:5876` through
  `:5886`.
- Chapter 12.4 says signed byte/short loads sign-extend and unsigned byte/short
  loads zero-extend to 32 bits before writing the VGPR, at
  `rdna4/README.md:5891`.
- The M0 table repeats the `DS_DIRECT_LOAD` M0 layout at
  `rdna4/README.md:1094`.
- The instruction definition repeats the same address/datatype contract at
  `rdna4/README.md:25216` through `:25218`.

XML evidence:

- The `DS_DIRECT_LOAD` description includes the address and datatype values at
  `amdgpu_isa_rdna4.xml:49986`.
- The operand list represents M0 only as a generic implicit `OPR_SDST_M0`
  source at `amdgpu_isa_rdna4.xml:50004` through `:50009`.
- The generic M0 operand description has no direct-load bitfield or extension
  metadata at `amdgpu_isa_rdna4.xml:182611` through `:182619`.

Impact:

The XML prose is useful for a human reader, but the machine-readable operand
metadata cannot drive datatype decoding, reserved-value validation, alignment
checking, or sign/zero extension behavior.

### RDNA4-XML-134: VDS address formulas and ADDTID offset semantics are only partially structured

Reported by: Feynman the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.5 defines single-address DS addressing as
  `LDS_BASE + VGPR[ADDR] + {OFFSET1,OFFSET0}`, at `rdna4/README.md:5951`
  through `:5955`.
- Chapter 12.5 defines double-address DS addressing with separate `OFFSET0`
  and `OFFSET1` fields scaled by 4 for 8/16/32-bit data and by 8 for 64-bit
  data, at `rdna4/README.md:5960` through `:5965`.
- Chapter 12.5 defines ADDTID addressing as
  `LDS_BASE + {OFFSET1, OFFSET0} + TID(0..63)*4 + M0`, explicitly with no
  VGPR address and DWORD-aligned M0, at `rdna4/README.md:5967` through
  `:5974`.
- The VDS field table says M0 is used only for the ADDTID forms and represents
  a byte address, at `rdna4/README.md:5920` through `:5922`.

XML evidence:

- `ENC_VDS` carries raw `ADDR`, `DATA0`, `DATA1`, `OFFSET0`, `OFFSET1`, `OP`,
  and `VDST` fields at `amdgpu_isa_rdna4.xml:3432` through `:3522`.
- The `OFFSET0` and `OFFSET1` descriptions say only to see individual
  instructions for use, at `amdgpu_isa_rdna4.xml:3483` through `:3495`.
- `DS_STORE_ADDTID_B32` and `DS_LOAD_ADDTID_B32` descriptions mention an
  immediate base and lane-ID offset, but their structured operands carry only
  `DATA0` or `VDST`, `OPR_DSMEM`, and generic implicit M0 at
  `amdgpu_isa_rdna4.xml:49401` through `:49428` and `:49444` through `:49471`.
- The generic M0 operand description only says M0 passes additional control
  information and DS addresses, at `amdgpu_isa_rdna4.xml:182611` through
  `:182619`.

Impact:

The XML gives enough field and opcode data to decode VDS forms, but an XML-only
consumer cannot generate the indexed, 2-address, stride64, or ADDTID address
formulas without manual prose.

### RDNA4-XML-135: Indexed and atomic DS DScnt producer ordering is not attached to DS instructions

Reported by: Feynman the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.5 says LDS indexed and atomic instructions increment `DScnt` on
  issue, decrement it when complete, and stay in order with other LDS
  instructions from the same wave, at `rdna4/README.md:5903`.

XML evidence:

- XML has wait-side support for `S_WAIT_DSCNT`,
  `S_WAIT_LOADCNT_DSCNT`, and `S_WAIT_STORECNT_DSCNT` at
  `amdgpu_isa_rdna4.xml:38758`, `:38816`, and `:38845`.
- `OPR_WAIT_MEM_DS` describes the split DS wait field at
  `amdgpu_isa_rdna4.xml:195495` through `:195512`.
- Representative DS entries such as `DS_ADD_U32` and `DS_LOAD_B32` carry
  `VMEM` / `DATA_SHARE` functional grouping and operand metadata, but no
  structured counter-producer or same-wave ordering annotation, at
  `amdgpu_isa_rdna4.xml:43966` through `:44002` and
  `amdgpu_isa_rdna4.xml:46096` through `:46135`.
- The broader wait-counter producer taxonomy gap is recorded in
  `RDNA4-XML-044`.

Impact:

The XML can parse wait instructions and DS instructions independently, but it
does not bind indexed/atomic DS operations to the `DScnt` producer and
in-order-within-wave contract from the manual.

### RDNA4-XML-136: Same-address DS atomic lane ordering and RMW atomicity are prose-only

Reported by: Feynman the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.5.1 says same-address lanes perform in an unspecified order, but
  each lane completes the full read-modify-write before another lane operates
  on the data, at `rdna4/README.md:5978`.

XML evidence:

- DS atomic entries encode the arithmetic operation and implicit DS memory
  read/write operands, for example `DS_ADD_U32` at
  `amdgpu_isa_rdna4.xml:43966` through `:43996`.
- The checked XML has no lane-conflict order, same-address serialization, or
  per-lane RMW atomicity metadata attached to DS atomic instructions.
- The broader LDS bank topology and bank-conflict gap is recorded in
  `RDNA4-XML-124`.

Impact:

Opcode and operand data alone cannot drive a faithful atomic scheduler or race
model for multiple active lanes touching the same LDS address.

### RDNA4-XML-137: DS floating-point atomic MODE denormal controls are not attached to the atomic entries

Reported by: Feynman the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.5.1 says floating-point atomic ops use the MODE register to
  control denormal flushing behavior, at `rdna4/README.md:5984`.
- Chapter 13.2 says LDS-indexed atomics use `MODE.denormal`, with
  `denorm_single` affecting F32 and `denorm_double` affecting F64 and F16, at
  `rdna4/README.md:6082` through `:6086`.
- The denormal table says `ADD_F32` and `PK_ADD_F16/_BF16` use MODE-controlled
  input and output denormal flushing for LDS DS atomics, at
  `rdna4/README.md:6095` through `:6105`.

XML evidence:

- XML has `S_DENORM_MODE` as a separate instruction at
  `amdgpu_isa_rdna4.xml:37981`.
- FP DS atomic entries such as `DS_ADD_F32` and `DS_PK_ADD_F16` list only
  address, data, and implicit DS-memory operands at
  `amdgpu_isa_rdna4.xml:44893` through `:44923` and
  `amdgpu_isa_rdna4.xml:48749` through `:48779`.
- No checked FP DS atomic entry carries structured MODE, `FP_DENORM`, input
  denorm, or output denorm metadata.

Impact:

The XML identifies the FP atomic opcodes and packed data formats, but cannot
tell a generator which wave-state bits affect their input/output denormal
behavior.

### RDNA4-XML-138: DS atomic constant-source wording conflicts with the structured operand class

Reported by: Feynman the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.5.1 says VGPR data sources for atomics can only be VGPRs or
  constant values, not SGPRs, at `rdna4/README.md:5984`.

XML evidence:

- Representative DS atomic `DATA0` operands use `OPR_VGPR`, for example
  `DS_ADD_U32` at `amdgpu_isa_rdna4.xml:43980` through `:43984`.
- `OPR_VGPR` says the operand must be a vector GPR and uses an 8-bit operand
  field, at `amdgpu_isa_rdna4.xml:194073` through `:194080`.
- Local assembler sanity checks rejected `ds_add_u32 v0, 0`,
  `ds_add_u32 v0, 1`, and `ds_add_u32 v0, -1` for `gfx1200`, while accepting
  `ds_add_u32 v0, v1`.

Impact:

The manual sentence, XML operand class, and LLVM assembler behavior do not
agree on whether DS atomic data sources can be constants. This should be
treated as a spec-data ambiguity and cross-checked before generalizing operand
classes from either source alone.

### RDNA4-XML-139: LDS lane-permute mapping, EXEC, and FI rules are prose-only

Reported by: Cicero the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.5.2 says `DS_PERMUTE` supports arbitrary swizzles across 32 lanes
  for wave32 and across all 64 lanes for wave64, at `rdna4/README.md:6019`.
- Chapter 12.5.2 defines the forward scatter and backward gather formulas,
  `Dst[index[0..31]] = src[0..31]` and
  `Dst[0..31] = src[index[0..31]]`, and says wave64 replaces `31` with `63`,
  at `rdna4/README.md:6023` through `:6025`.
- Chapter 12.5.2 says EXEC is honored for both source reads and destination
  writes except for `DS_BPERMUTE_FI_B32`, which reads all lanes and uses EXEC
  only for writes; out-of-range index values wrap using index bits `[6:2]` for
  wave32 and `[7:2]` for wave64; disabled source lanes read as zero, at
  `rdna4/README.md:6027`.
- Chapter 12.5.2 says indexes are byte values, `offset0` is added before use,
  and `VDST`, `ADDR`, and `DATA0` have the destination, index, and source-data
  roles, at `rdna4/README.md:6029`.
- The Chapter 12.5.2 figure adds that `DS_PERMUTE` collisions use the
  highest-numbered source lane and unused destinations return zero, in
  `workspace_docs/amdgpu-isa-manuals/rdna4/assets/_page_163_Figure_2.jpeg`.

XML evidence:

- XML has `DS_PERMUTE_B32`, `DS_BPERMUTE_B32`, and `DS_BPERMUTE_FI_B32` entries
  with short descriptions and `VDST`, `ADDR`, and `DATA0` operands at
  `amdgpu_isa_rdna4.xml:49487` through `:49513`, `:49529` through `:49555`,
  and `:49571` through `:49597`.
- `DS_BPERMUTE_FI_B32` is distinguished only by the description "fetch data for
  invalid lanes"; there is no structured source-EXEC exception, disabled-source
  zeroing, or destination-EXEC metadata, at `amdgpu_isa_rdna4.xml:49571`
  through `:49597`.
- The shared VDS `OFFSET0` and `OFFSET1` fields say only to see individual
  instructions for how they are used, at `amdgpu_isa_rdna4.xml:3483` through
  `:3495`.
- The checked XML has no structured lane-mapping formulas, wave32/wave64 index
  bit selection, collision priority, or unused-destination zeroing metadata for
  the lane-permute opcodes.

Impact:

The XML can decode the permute-family opcodes and operands, but cannot drive a
correct scatter/gather/FI implementation or analysis without manual prose and
figure-derived rules.

### RDNA4-XML-140: `DS_SWIZZLE_B32` mode tables and invalid-source behavior are not encoded

Reported by: Cicero the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.5 lists `DS_SWIZZLE_B32` as a DWORD swizzle that writes no LDS
  memory, at `rdna4/README.md:5947`.
- Chapter 7.9 summarizes `DS_SWIZZLE` as an LDS operation that swizzles within
  a group of 32 lanes from a fixed menu of rotate, broadcast, and swap forms,
  at `rdna4/README.md:3369`.
- The Chapter 16.15 `DS_SWIZZLE_B32` definition says invalid-thread reads
  return `0x0` and lists FFT, rotate, group-of-4, and group-of-32 modes, at
  `rdna4/README.md:25807` through `:25815`.
- The same definition gives offset ranges and pseudocode for FFT
  (`offset >= 0xe000`), rotate (`0xc000 <= offset < 0xe000`), group-of-4
  (`offset[15] == 1`), and group-of-32 (`offset[15] == 0`) behavior, at
  `rdna4/README.md:25817` through `:25916`.

XML evidence:

- XML has `DS_SWIZZLE_B32` opcode 53 with `VDST` and `ADDR` operands at
  `amdgpu_isa_rdna4.xml:46060` through `:46080`.
- The XML description only says "Dword swizzle, no data is written to LDS
  memory" at `amdgpu_isa_rdna4.xml:46061`.
- Searches of the checked XML find no `FFT`, `rotate`, `thread_valid`,
  `QUAD_PERM`, or DS-swizzle mode table metadata.

Impact:

The XML preserves the opcode shell, but not the source-lane mapping modes,
invalid-source zeroing, or control-bit interpretation required to emulate or
disassemble DS swizzle forms beyond the mnemonic and raw offset bits.

### RDNA4-XML-141: BVH stack offset fields have instruction-specific meanings that are prose-only

Reported by: Euler the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.5.3 defines `OFFSET0` bits `[4:0]` as `StackSize` and
  `OFFSET1[1:0]` as primitive-range and triangle-pair optimization flags, at
  `rdna4/README.md:6048` through `:6053`.
- The Chapter 16.15 entries repeat that `OFFSET0[4:0]` carries the per-shader
  patched stack size and `OFFSET1[0]`/`OFFSET1[1]` carry the triangle-size
  optimization and primitive-range flags, at `rdna4/README.md:27040` through
  `:27042`, `:27098` through `:27100`, and `:27156` through `:27158`.

XML evidence:

- The shared `ENC_VDS` field definitions expose only generic `OFFSET0` and
  `OFFSET1` fields and say to see individual instructions for their use, at
  `amdgpu_isa_rdna4.xml:3483` through `:3495`.
- The three `DS_BVH_STACK_*` entries do not attach stack-size, primitive-range,
  or triangle-pair flag semantics to those fields, at
  `amdgpu_isa_rdna4.xml:49701` through `:49835`.

Impact:

XML consumers can preserve raw bits, but cannot interpret nonzero stack-op
offset fields as stack size and ray-tracing traversal flags without manual
prose.

### RDNA4-XML-142: BVH stack packed `ADDR` state layout is not structured

Reported by: Euler the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.5.3 says `ADDR` is both a source and destination VGPR carrying
  the LDS stack address and updated address, with packed `stack_base`,
  `stack_size`, and `stack_index` fields, at `rdna4/README.md:6055`.
- The address-field table further defines `valid_entries`, `entries_to_tlas`,
  `ring_addr`, `stack_base_addr`, `has_tlas_in_stack`, `has_overflowed`, and
  `blas_to_tlas_pop`, at `rdna4/README.md:6060` through `:6069`.

XML evidence:

- XML marks `ADDR` as a 32-bit VGPR input/output operand for the three stack
  opcodes at `amdgpu_isa_rdna4.xml:49715` through `:49720`, `:49763` through
  `:49768`, and `:49811` through `:49816`.
- No structured XML metadata describes the packed address fields, ring-buffer
  counters, overflow state, or BLAS/TLAS transition bits.

Impact:

The XML captures that `ADDR` is in-out, but cannot drive stack-address decode,
address update, or traversal-state analysis from operand metadata alone.

### RDNA4-XML-143: BVH stack push/pop behavior is reduced to descriptions and operand widths

Reported by: Euler the 2nd subagent; local audit.

Manual evidence:

- The `DS_BVH_STACK_PUSH4_POP1_RTN_B32` entry defines `DATA_VALID`, last-node
  filtering, three push passes, return-or-pop fallback, LDS invalidation, and
  updated address return, at `rdna4/README.md:27048` through `:27091`.
- The push8/pop1 form repeats the same contract with seven push passes and an
  eighth candidate return, at `rdna4/README.md:27106` through `:27149`.
- The push8/pop2 B64 form adds a second pop into the high return dword, at
  `rdna4/README.md:27164` through `:27213`.

XML evidence:

- XML has only the high-level one-sentence descriptions plus `VDST`, `ADDR`,
  `DATA0`, and `DATA1` operand shapes for the three stack opcodes at
  `amdgpu_isa_rdna4.xml:49701` through `:49835`.
- It records push4 `DATA1` as `FMT_NUM_B128`, push8 `DATA1` as
  `FMT_NUM_B256`, and pop2 `VDST` as `FMT_NUM_B64`, but does not encode
  per-dword push counts, validity filtering, memory invalidation, pop fallback,
  or second-pop behavior.

Impact:

An XML-derived decoder can identify the opcodes and operand footprints, but not
implement the BVH short-stack contract without the manual pseudocode.

### RDNA4-XML-145: Float memory atomic numeric policy is prose-only

Reported by: local audit.

Manual evidence:

- Chapter 13 says floating-point atomics can be issued through LDS, buffer, and
  flat/global/scratch instructions, and that memory atomics do not report
  numeric exceptions, at `rdna4/README.md:6072` through `:6076`.
- Chapter 13.1 fixes float-atomic-add rounding to round-to-nearest-even and
  says `MODE.round` is ignored, at `rdna4/README.md:6078` through `:6080`.
- Chapter 13.2 gives a denormal-control table that differs between LDS DS,
  LDS through flat, L2 cache, and data-fabric paths for `ADD_F32`,
  `PK_ADD_F16/_BF16`, and compare-store/swap forms, at
  `rdna4/README.md:6082` through `:6110`.
- Chapter 13.3 defines NaN quieting/propagation, memory-atomic
  MINNUM/MAXNUM behavior, signed-zero ordering, and float-add special cases at
  `rdna4/README.md:6113` through `:6167`.

XML evidence:

- XML enumerates representative float atomic opcodes and implicit data formats:
  buffer F32 and packed F16/BF16 records at `amdgpu_isa_rdna4.xml:39846`
  through `:40098`, flat records at `:50177` through `:50405`, global records
  at `:53616` through `:54082`, image records at `:55593` through `:55817`,
  and DS packed records at `:48749` and `:48795`.
- Some descriptions name `minimumNumber()` or packed two-component data, but
  no checked XML field attaches fixed-RNE rounding, exception suppression,
  the path-dependent denormal table, NaN quieting/payload selection,
  signed-zero ordering, or float-add special cases to those opcodes.
- The DS-only denormal-metadata subset is recorded separately in
  `RDNA4-XML-137`; this entry is the broader Chapter 13 contract across all
  float-memory-atomic issuers.

Impact:

XML-derived code can recover the opcode names and coarse operand formats, but
cannot faithfully generate or validate RDNA4 float-atomic numeric behavior
without the manual prose.

### RDNA4-XML-146: Export shader-stage contract is prose-only

Reported by: local audit.

Manual evidence:

- Chapter 14 says exports copy VGPR data to position, color, or Z export
  buffers, use `EXEC`, may export to each target only once, and require the
  last pixel or position export to set `DONE`, at `rdna4/README.md:6171`.
- Chapter 14 says 16-bit exports move packed 32-bit VGPR data and rely on the
  shader plus receiving configuration registers to define the 16-bit element
  type, at `rdna4/README.md:6173`.
- The field and component tables define `DONE`, target values, 32-bit and
  16-bit `EN` meanings, `ROW_EN`, and M0 row use at
  `rdna4/README.md:6181` through `:6204`.
- Sections 14.1-14.3 define pixel export obligations, stable final `EXEC`
  masks, `STATUS.SKIP_EXPORT`, OREO ordering, dual-source blend lane masks,
  primitive/position `DONE` requirements, and two-phase `EXPcnt` dependency
  behavior at `rdna4/README.md:6206` through `:6258`.

XML evidence:

- XML carries the raw `ENC_VEXPORT` field layout for `DONE`, `EN`, `ROW_EN`,
  `TGT`, and `VSRC0` through `VSRC3` at `amdgpu_isa_rdna4.xml:3447` through
  `:3534`.
- XML has one `EXPORT` instruction entry with broad prose, `TGT`, four VGPR
  sources, and implicit `EXEC`/M0 operands at `amdgpu_isa_rdna4.xml:47475`
  through `:47522`; `OPR_TGT` has predefined target names at
  `amdgpu_isa_rdna4.xml:192783`.
- The checked XML does not encode one-export-per-target validation,
  stage-specific `DONE` obligations, 16-bit packed data contracts,
  dual-source blend lane masks/back-to-back ordering, final pixel-valid mask
  restrictions, OREO conflict behavior, `STATUS.SKIP_EXPORT` semantics, or
  export-specific two-phase source-read timing. The broad dependency-counter
  metadata gap is also recorded in `RDNA4-XML-044`.

Impact:

XML-derived code can decode the export instruction, but cannot validate or
model the shader-stage export contract without the Chapter 14 prose.

### RDNA4-XML-147: `EXPORT` models `TGT` as a 128-bit output operand

Reported by: local audit.

Manual evidence:

- Chapter 14 defines `Target` as a 6-bit export-target selector at
  `rdna4/README.md:6182` through `:6188`.
- The exported data comes from `VSRC0` through `VSRC3` or packed half-dword
  source pairs selected by `EN`, at `rdna4/README.md:6189` through `:6204`.

XML evidence:

- The `EXPORT` instruction entry marks `TGT` as an explicit output operand
  with `OperandSize` 128 at `amdgpu_isa_rdna4.xml:47487` through `:47491`.
- The same instruction then lists the four 32-bit VGPR source operands at
  `amdgpu_isa_rdna4.xml:47493` through `:47512`.

Impact:

Consumers that trust XML operand direction see a 128-bit destination operand
for a 6-bit selector field, which can mislead def-use metadata, disassembly
shape, and export-side-effect modeling.

### RDNA4-XML-148: Memory-format `NV` bits are XML-only in Chapter 15 field maps

Reported by: local audit.

Manual evidence:

- Chapter 15.2.1 Table 96 lists SMEM fields `SBASE`, `SDATA`, `OP`,
  `SCOPE`, `TH`, `ENCODING`, `IOFFSET`, and `SOFFSET` at
  `rdna4/README.md:6679` through `:6688`, with no bit-20 `NV` row.
- Chapter 15.3.12 Table 111 lists VBUFFER fields and leaves bit 7 unnamed at
  `rdna4/README.md:7855` through `:7869`.
- Chapter 15.3.13/15.3.14 and 15.3.15 list VIMAGE, VSAMPLE, and
  FLAT/GLOBAL/SCRATCH fields without an `NV` row at `rdna4/README.md:7932`
  through `:7946`, `:7992` through `:8003`, and `:8070` through `:8080`.

XML evidence:

- XML defines `NV` in `ENC_SMEM` at `amdgpu_isa_rdna4.xml:853`.
- XML defines `NV` in `ENC_VBUFFER`, `ENC_VIMAGE`, `ENC_VSAMPLE`,
  `ENC_VFLAT`, `ENC_VSCRATCH`, and `ENC_VGLOBAL` at
  `amdgpu_isa_rdna4.xml:3679`, `:3902`, `:4205`, `:4610`, `:4796`, and
  `:5020`.

Impact:

Manual-only consumers of Chapter 15 do not see the non-volatile selector bit,
while XML/generated-code consumers can decode and print it. This is a source
drift issue in the opposite direction from the usual XML omissions: the
machine-readable source is richer than the manual field tables.

### RDNA4-XML-149: `S_VERSION` upper-byte bits conflict with the manual zero rule

Reported by: local audit.

Manual evidence:

- Chapter 16.2 says `S_VERSION` is a no-op/comment used by tools, and its
  hardware argument is ignored at `rdna4/README.md:9121` through `:9127`.
- The same definition says `SIMM16[7:0]` specifies the microcode version and
  `SIMM16[15:8]` must be zero at `rdna4/README.md:9131`.

XML evidence:

- XML models `S_VERSION` as `ENC_SOPK` with an `OPR_VERSION` `SIMM16` operand
  at `amdgpu_isa_rdna4.xml:37370` through `:37382`.
- `OPR_VERSION` gives bit 15 the field name `MDP`, bit 14 `W32`, bit 13
  `W64`, and bits 0 through 7 `VERSION` at
  `amdgpu_isa_rdna4.xml:194002` through `:194070`.

Impact:

XML-derived tools can treat bits 13 through 15 as meaningful version-header
fields even though the RDNA4 manual requires the entire upper byte to be zero.
Bits 8 through 12 also remain neither named nor marked reserved in the XML
operand format.

### RDNA4-XML-150: SOP1 scalar relative-index equations and `M0` slices are not represented

Reported by: local audit.

Manual evidence:

- Chapter 6.1 summarizes `S_MOVRELS_{B32,B64}` and `S_MOVRELD_{B32,B64}` as
  `D = SGPR[S0+M0]` and `SGPR[D+M0] = S0`, says the B64 index must be even, and
  says `M0` is an unsigned index at `rdna4/README.md:2504`.
- Chapter 16.3 gives the detailed `S_MOVRELS_B32/B64` and `S_MOVRELD_B32/B64`
  formulas as `addr = SRC0/DST; addr += M0.u32[31:0]` at
  `rdna4/README.md:10135` through `:10160` and `:10168` through `:10196`.
- `S_MOVRELSD_2_B32` uses source index `M0[9:0]` and destination index
  `M0[25:16]` at `rdna4/README.md:10204` through `:10210`.

XML evidence:

- The XML entries for `S_MOVRELS_B32/B64`, `S_MOVRELD_B32/B64`, and
  `S_MOVRELSD_2_B32` exist at `amdgpu_isa_rdna4.xml:17457`, `:17500`,
  `:17543`, `:17610`, and `:17677`, with an implicit `OPR_SDST_M0` operand for
  `M0`.
- `OPR_SDST_M0` only describes `M0` as a register used for relative indices and
  other control data at `amdgpu_isa_rdna4.xml:181494` through `:181503`; it does
  not encode the per-instruction assignment formula, the unsigned raw-index
  rule, B64 evenness, or the split `M0[9:0]` / `M0[25:16]` slices.

Impact:

XML consumers can discover that scalar relative moves implicitly use `M0`, but
cannot derive the actual source/destination index calculation or distinguish the
full-width and split-index forms without manual augmentation.

### RDNA4-XML-151: SOPC bit-compare index masks are not represented

Reported by: local audit.

Manual evidence:

- `S_BITCMP0_B32` and `S_BITCMP1_B32` extract `S0.u32[S1.u32[4:0]]` at
  `rdna4/README.md:10687` through `:10700`.
- `S_BITCMP0_B64` and `S_BITCMP1_B64` extract `S0.u64[S1.u32[5:0]]` at
  `rdna4/README.md:10703` through `:10718`.

XML evidence:

- XML entries for `S_BITCMP0_B32`, `S_BITCMP1_B32`, `S_BITCMP0_B64`, and
  `S_BITCMP1_B64` exist at `amdgpu_isa_rdna4.xml:30161`, `:30276`, `:30391`,
  and `:30506`.
- The descriptions say the bit is selected by an index in the second scalar
  input, and the second operand is a 32-bit `OPR_SSRC`, for example
  `S_BITCMP0_B32` at `amdgpu_isa_rdna4.xml:30161` through `:30180`.
- The XML does not encode the B32 `S1[4:0]` mask, the B64 `S1[5:0]` mask, or a
  structured relationship between source width and index-mask width.

Impact:

XML consumers can discover the bit-compare opcode and operand shapes, but cannot
derive the architectural modulo-32/modulo-64 index behavior from structured XML
metadata.

### RDNA4-XML-152: SOPP opcode inventory conflicts on `S_WAITCNT` and trace opcodes

Reported by: local audit.

Manual evidence:

- Chapter 15.1 Table 83 lists SOPP opcode 9 as `S_WAITCNT` and marks opcodes 58
  and 59 reserved at `rdna4/README.md:6644` through `:6659`.
- Chapter 16.5 defines `S_WAITCNT 9` as equivalent to `S_WAIT_IDLE`, says modern
  code should use the specialized `S_WAIT_*` instructions, and says the operand
  is ignored for compatibility at `rdna4/README.md:11260` through `:11262`.

XML evidence:

- XML has `ENC_SOPP` entries for `S_WAIT_ALU` opcode 8 and `S_WAIT_IDLE` opcode
  10 at `amdgpu_isa_rdna4.xml:34622` through `:34665`, but no
  `InstructionName` entry for `S_WAITCNT`.
- XML defines `S_TTRACEDATA` as SOPP opcode 58 and `S_TTRACEDATA_IMM` as SOPP
  opcode 59 at `amdgpu_isa_rdna4.xml:35366` through `:35410`.

Impact:

XML-derived opcode inventories drop the manual's compatibility wait-idle opcode
and can regenerate executable trace opcodes for two encodings the RDNA4 manual
marks reserved.

### RDNA4-XML-153: SMEM XML exposes ATC probe opcodes absent from the manual

Reported by: local audit.

Manual evidence:

- Chapter 15.2 Table 85 lists SMEM opcodes 0 through 5, 8 through 11, 16 through
  21, 24 through 27, 33, and 36 through 40; it has no entries for opcodes 34 or
  35 at `rdna4/README.md:6690` through `:6706`.
- Chapter 16.6 defines the same scalar load, scalar-buffer load, `S_DCACHE_INV`,
  and scalar prefetch instructions, then proceeds to Chapter 16.7 without
  defining `S_ATC_PROBE` or `S_ATC_PROBE_BUFFER` at `rdna4/README.md:11569`
  through `:12075`.

XML evidence:

- XML defines `S_ATC_PROBE` as `ENC_SMEM` opcode 34 with `SDATA`, `SBASE`, and
  `SOFFSET` operands at `amdgpu_isa_rdna4.xml:13149` through `:13175`.
- XML defines `S_ATC_PROBE_BUFFER` as `ENC_SMEM` opcode 35 with the same source
  shape but a 128-bit scalar resource base at `amdgpu_isa_rdna4.xml:13193`
  through `:13219`.

Impact:

XML-derived SMEM inventories can regenerate two probe/prefetch instructions in
manual opcode holes. Manual-derived validators or test oracles will disagree on
whether opcodes 34 and 35 are legal RDNA4 SMEM instructions.

### RDNA4-XML-154: VALU min/max-number edge semantics are prose-only

Reported by: local audit.

Manual evidence:

- Chapter 16.7 defines `V_MIN_NUM_F32` / `V_MAX_NUM_F32` with full
  `minimumNumber()` / `maximumNumber()` pseudocode, including signaling-NaN
  invalid flagging, NaN quieting/source selection, signed-zero tie ordering,
  denorm-mode notes, exception flags, and saturation support at
  `rdna4/README.md:12331` through `:12393`.
- Chapter 16.11 repeats the same detailed min/max-number contract for VOPD
  `V_DUAL_MAX_NUM_F32` and `V_DUAL_MIN_NUM_F32` in both X and Y definitions at
  `rdna4/README.md:17946` through `:18011` and `:18168` through `:18231`.

XML evidence:

- XML instruction records expose the coarse `minimumNumber()` /
  `maximumNumber()` descriptions for representative VOP2/VOP3 entries such as
  `V_MIN_NUM_F32` and `V_MAX_NUM_F32` at
  `amdgpu_isa_rdna4.xml:75595` through `:75868`, and for VOPD
  `V_DUAL_MAX_NUM_F32` / `V_DUAL_MIN_NUM_F32` at
  `amdgpu_isa_rdna4.xml:169552` through `:169669`.
- Those records carry operand formats and encoding alternatives, but no
  structured metadata for signaling-NaN invalid behavior, quiet-NaN payload or
  source selection, signed-zero ordering, denorm-mode dependency, or the
  per-op exception/saturation contract.
- Searching the checked-in RDNA4 XML for `TRAPSTS`, `isSignalNAN`, and
  `cvtToQuietNAN` finds no corresponding machine-readable semantic rules.

Impact:

XML-derived VALU generators can recover that these opcodes are min/max-number
operations, but cannot reproduce the manual's exact NaN, signed-zero,
denormal, exception, and saturation behavior without prose augmentation.

### RDNA4-XML-155: Special DS atomic update predicates are not structured

Reported by: local audit.

Manual evidence:

- Chapter 16.15 defines `DS_CONDXCHG32_RTN_B64` as two independent 32-bit
  conditional exchanges at aligned addresses, gated by `DATA[31]` and
  `DATA[63]`, clearing the high bit of each written dword at
  `rdna4/README.md:26620` through `:26645`.
- `DS_COND_SUB_U32` and `DS_COND_SUB_RTN_U32` subtract only when the old LDS
  value is greater than or equal to the source value, at
  `rdna4/README.md:26648` through `:26657` and `:26778` through `:26787`.
- `DS_SUB_CLAMP_U32` and `DS_SUB_CLAMP_RTN_U32` clamp underflow to zero rather
  than wrapping, at `rdna4/README.md:26660` through `:26673` and
  `:26790` through `:26803`.

XML evidence:

- XML records `DS_CONDXCHG32_RTN_B64` with the high-level conditional-exchange
  description and `VDST`, `ADDR`, `DATA0`, and implicit DS-memory operands at
  `amdgpu_isa_rdna4.xml:48605` through `:48640`.
- XML records `DS_COND_SUB_U32` and `DS_SUB_CLAMP_U32` with descriptions and
  generic 32-bit DS read/write operands at `amdgpu_isa_rdna4.xml:48657`
  through `:48740`.
- XML records the return variants with the same description-plus-operand shape
  at `amdgpu_isa_rdna4.xml:49193` through `:49282`.
- The checked XML does not provide structured atomic sub-op metadata for the
  per-dword MSB predicates, high-bit clearing, aligned paired addresses,
  conditional subtract predicate, or saturating subtract behavior.

Impact:

XML consumers can decode these opcode shells and read the free-text
descriptions, but cannot infer the special RMW update function from structured
operand metadata. A generator that classifies atomics by width and mnemonic
tokens can collapse these opcodes into ordinary compare-swap or subtract unless
it carries manual-prose overrides.

### RDNA4-XML-156: Special VBUFFER atomic update predicates are not structured

Reported by: local audit.

Manual evidence:

- `BUFFER_ATOMIC_SUB_CLAMP_U32` clamps unsigned underflow to zero instead of
  wrapping, at `rdna4/README.md:27742` through `:27762`.
- `BUFFER_ATOMIC_COND_SUB_U32` subtracts only when the old buffer value is
  greater than or equal to the data value, otherwise leaving memory unchanged,
  at `rdna4/README.md:28033` through `:28047`.

XML evidence:

- XML records `BUFFER_ATOMIC_SUB_CLAMP_U32` as opcode 55 with a prose
  clamp-to-zero description and generic U32 memory operands at
  `amdgpu_isa_rdna4.xml:41334` through `:41410`.
- XML records `BUFFER_ATOMIC_COND_SUB_U32` as opcode 80 with a prose
  conditional-subtract description and the same generic U32 operand shape at
  `amdgpu_isa_rdna4.xml:42738` through `:42800`.
- The checked XML does not provide structured atomic sub-op metadata for the
  saturating subtract update or the conditional `old >= src` predicate.

Impact:

XML consumers can distinguish the opcode names and read the descriptions, but
cannot derive the update rule programmatically from operand or atomic metadata;
both opcodes look like ordinary U32 memory read-modify-write operations without
manual prose interpretation.

### RDNA4-XML-157: Special Flat/Global atomic update predicates are not structured

Reported by: local audit; Planck the 2nd subagent.

Manual evidence:

- `FLAT_ATOMIC_SUB_CLAMP_U32` and `GLOBAL_ATOMIC_SUB_CLAMP_U32` clamp unsigned
  underflow to zero rather than wrapping, at `rdna4/README.md:29391` through
  `:29404` and `:30325` through `:30338`.
- `FLAT_ATOMIC_COND_SUB_U32` and `GLOBAL_ATOMIC_COND_SUB_U32` subtract only
  when the old memory value is greater than or equal to the data value, at
  `rdna4/README.md:29670` through `:29679` and `:30606` through `:30615`.
- `GLOBAL_ATOMIC_ORDERED_ADD_B64` compares the memory ID with `DATA[31:0]`,
  conditionally increments the memory ID modulo `VGT_GS_MAX_WAVE_ID`, adds
  `DATA[63:32]` to the memory value, and returns the old pair, at
  `rdna4/README.md:30744` through `:30765`.

XML evidence:

- XML records `FLAT_ATOMIC_SUB_CLAMP_U32` and `FLAT_ATOMIC_COND_SUB_U32` with
  prose descriptions and generic U32 GPU-memory operands at
  `amdgpu_isa_rdna4.xml:51279` through `:51324` and `:52544` through
  `:52594`.
- XML records `GLOBAL_ATOMIC_SUB_CLAMP_U32` and `GLOBAL_ATOMIC_COND_SUB_U32`
  with the same description-plus-generic-operand shape at
  `amdgpu_isa_rdna4.xml:54342` through `:54400` and `:55769` through
  `:55825`.
- XML records `GLOBAL_ATOMIC_ORDERED_ADD_B64` with only a prose summary and
  generic B64 memory operands at `amdgpu_isa_rdna4.xml:56323` through
  `:56383`.
- The checked XML does not provide structured atomic sub-op metadata for the
  saturating subtract update, conditional subtract predicate, ordered-add ID
  match, modulo ID increment, or split 32-bit memory-pair update. The stricter
  return-mode legality for these special forms remains covered separately by
  `RDNA4-XML-107`.

Impact:

XML consumers can decode the opcode shells and read the descriptions, but
cannot infer these special RMW update functions from structured operand or
atomic metadata. A generator that classifies atomics by width and mnemonic
tokens can collapse these opcodes into ordinary subtract or add unless it
carries manual-prose overrides.

### RDNA4-XML-158: Chapter 7.6 conversion-wide restrictions are only prose

Reported by: local audit.

Manual evidence:

- Chapter 7.6 lists common data-conversion restrictions: input modifiers
  `neg` and `abs` are unsupported, rounding is RNE except for stochastic
  rounded operations, OMOD is unused, CLAMP is unsupported for BF8/FP8 or
  smaller-float conversions, `FP16_OVFL` applies to FP16/BF16 destinations, and
  OPSEL is unsupported for packed or stochastic converts, at
  `rdna4/README.md:3128` through `:3135`.
- Chapter 7.6.1 classifies normal/single, packed, and stochastic rounded
  conversions at `rdna4/README.md:3139` through `:3155`.

XML evidence:

- The generic VOP3 encoding exposes `ABS`, `CLAMP`, `NEG`, `OMOD`, and `OPSEL`
  fields at `amdgpu_isa_rdna4.xml:2518`, `:2528`, `:2548`, `:2558`, and
  `:2584`.
- The field descriptions refer to `OPF_NO*` lists, but the checked XML does not
  expose instruction-level modifier-restriction data that can be joined to the
  `V_CVT*` entries.
- The XML `V_CVT*` instruction entries provide names, operand formats, and some
  prose descriptions, but do not provide structured conversion-family metadata
  for the Chapter 7.6 normal/single, packed, and stochastic categories or their
  shared no-modifier/no-OMOD/no-OPSEL rules.

Impact:

XML-only consumers cannot infer the common conversion legality contract from
structured data. They need manual overrides to reject or ignore unsupported
modifiers and selector fields, and to decide which conversion opcodes are
normal, packed, or stochastic for shared rule application.

### RDNA4-XML-159: Chapter 5.7.2 state-register dependency cases are prose-only

Reported by: local audit.

Manual evidence:

- Chapter 5.7.2 says `S_SETHALT` followed by `S_GETREG` of `STATE_PRIV` or
  `STATUS` requires an intervening `S_NOP` or any other instruction, at
  `rdna4/README.md:2261` through `:2265`.
- The same list says `S_SETPRIO` followed by `S_GETREG STATE_PRIV` requires
  the same separation, at `rdna4/README.md:2263` through `:2266`.
- The section also points to the WMMA hazard rules at
  `rdna4/README.md:2267`; those matrix hazards remain tracked separately under
  the Chapter 7.12 WMMA entries.

XML evidence:

- XML has instruction shells for `S_GETREG_B32`, `S_SETHALT`, `S_SETPRIO`, and
  `S_NOP` at `amdgpu_isa_rdna4.xml:34268`, `:34494`, `:35196`, and `:34430`.
- `OPR_HWREG` lists `HW_REG_WAVE_STATUS` and `HW_REG_WAVE_STATE_PRIV` at
  `amdgpu_isa_rdna4.xml:180672` through `:180679`.
- XML exposes `S_DELAY_ALU` / `OPR_DELAY` and `S_WAIT_ALU` / `OPR_WAIT_ALU`
  operand structures at `amdgpu_isa_rdna4.xml:34590` through `:34636`,
  `:180425` through `:180535`, and `:194241` through `:194285`.
- None of these records connects `S_SETHALT` or `S_SETPRIO` as producers of a
  state-register read hazard, nor marks `S_GETREG` of `STATUS` or `STATE_PRIV`
  as the consumer requiring one intervening instruction.

Impact:

XML consumers can decode the individual instructions and display the hardware
register operands, but cannot derive the Chapter 5.7.2 scheduling rule. A
validator, code patcher, or DBT pass needs manual knowledge to reject or repair
the immediate `S_SETHALT`/`S_SETPRIO` to `S_GETREG` sequences.

### RDNA4-XML-160: `S_DELAY_ALU` stream semantics are only partially represented

Reported by: local audit.

Manual evidence:

- Chapter 5.8 says `S_DELAY_ALU` is inserted before the instruction to delay
  and describes dependencies on previous instructions rather than just a local
  immediate value, at `rdna4/README.md:2271` through `:2279`.
- `INSTID` counts backwards through issued VALU instructions: branched-over
  VALU instructions do not count, but VALU instructions skipped because
  `EXEC==0` do count, at `rdna4/README.md:2283` through `:2285`.
- `SKIP` counts every instruction type before the second dependency, an
  unconsumed earlier `S_DELAY_ALU` is replaced by a later one, and
  `S_DELAY_ALU` applies to any opcode even when it is not useful, at
  `rdna4/README.md:2287` through `:2291`.
- The dependency-code table defines codes `1..4` as previous ordinary VALU
  instructions, `5..7` as previous transcendental VALU instructions, code `8`
  as reserved, and codes `9..11` as one through three SALU-cycle waits, at
  `rdna4/README.md:2295` through `:2301`.

XML evidence:

- XML has an `S_DELAY_ALU` instruction entry with `OPR_DELAY` at
  `amdgpu_isa_rdna4.xml:34590` through `:34604`.
- `OPR_DELAY` exposes `INSTID0`, `INSTID1`, and `INSTSKIP` bit positions and
  predefined values at `amdgpu_isa_rdna4.xml:180425` through `:180613`.
- Those field records do not encode the instruction-stream history rules:
  issued-VALU counting, branch versus `EXEC==0` treatment, replacement of
  unconsumed delay metadata, or the "any opcode" consumption rule.
- XML gives code `8` the name `INSTID_FMA_ACCUM_CYCLE_1` and description
  "Single cycle penalty for FMA accumulation (reserved)" at
  `amdgpu_isa_rdna4.xml:180480` through `:180482` and again for `INSTID1` at
  `:180552` through `:180554`, while the manual table lists code `8` only as
  reserved.

Impact:

XML consumers can decode the raw `S_DELAY_ALU` operand fields, but cannot
validate or simulate the stream-history semantics without manual prose. The
code-8 wording also risks treating a reserved manual value as a meaningful FMA
accumulation delay.

### RDNA4-XML-161: Chapter 1 architecture and host-control overview is not machine-readable

Reported by: local audit.

Manual evidence:

- Chapter 1 defines the manual's scope as RDNA4 instruction set and
  shader-program accessible state at `rdna4/README.md:362`, then introduces
  RDNA4 as a parallel micro-architecture at `rdna4/README.md:364`.
- The terminology table defines dispatch, work-group, wave, WGP, CU, SIMD32,
  LDS, VMEM, sampler, texture resource, and buffer resource terms at
  `rdna4/README.md:387` through `:429`.
- Chapter 1.2 says the device includes a processor array, command processor,
  and memory controller; the command processor reads host-written
  memory-mapped registers and sends hardware-generated completion interrupts,
  while the memory controller has direct access to device and host-specified
  system memory, at `rdna4/README.md:486`.
- Chapter 1.2.1 through 1.2.3 describe independent WGP pipelines, automatic
  instruction fetch into on-chip caches, hardware FP-exception interrupts, LDS
  geometry and the 64KiB per-workgroup allocation limit, cache hierarchy,
  cache-scope acknowledgments, and relaxed consistency at
  `rdna4/README.md:502`, `:508`, `:510`, `:528` through `:532`, and
  `:536` through `:542`.

XML evidence:

- The top-level XML architecture metadata records only `AMD RDNA 4` and
  architecture ID `10` at `amdgpu_isa_rdna4.xml:11` through `:12` before
  entering instruction encoding records.
- Searching XML for WGP, SIMD32, shader array/engine, command processor, memory
  controller, relaxed/serial consistency, FP exception, and hardware-generated
  host interrupt wording finds no architecture-topology or host-control schema;
  the only host-interrupt hit is the `S_SENDMSG` software interrupt description
  at `amdgpu_isa_rdna4.xml:181532`.
- Related per-instruction or per-feature details are present only in narrower
  records or gaps: data-format layouts exist under the `FMT_NUM_*` records,
  while wave, LDS, cache/scope, memory-ordering, descriptor, and export-stage
  semantics are tracked by later entries such as `RDNA4-XML-012` through
  `RDNA4-XML-027`, `RDNA4-XML-091`, `RDNA4-XML-098` through
  `RDNA4-XML-100`, `RDNA4-XML-124`, `RDNA4-XML-125`, `RDNA4-XML-145`, and
  `RDNA4-XML-146`.

Impact:

XML consumers cannot derive the Chapter 1 device topology, host launch and
completion relationship, FP-exception interrupt capability, memory hierarchy,
LDS topology, or relaxed-consistency overview from XML alone. Validators,
emulators, and documentation generators need manual prose or separate
architecture configuration for that layer, even when instruction-level field
decode comes from XML.

### RDNA4-XML-162: Chapter 2 wave32/wave64 issue model is prose-only

Reported by: local audit.

Manual evidence:

- Chapter 2.1 says both wave32 and wave64 are supported for all operations
  except VOPD, and that shaders are compiled and run for one fixed wave size,
  at `rdna4/README.md:569` through `:571`.
- Wave32 issues each instruction once, while wave64 typically issues VALU and
  vector-memory instructions, including LDS, texture, buffer, flat, scratch,
  and global, twice; scalar ALU, scalar memory, branches, messages, and exports
  issue once, at `rdna4/README.md:571` through `:573`.
- Chapter 2.1 records wave64 skip-half rules for zero `EXEC` halves, the
  VMEM outstanding-operation exception, and the rule that VALU instructions
  writing SGPRs are not half-skipped, at `rdna4/README.md:575` through `:579`.
- The same section says both wave64 passes use the pre-instruction wave state,
  with an exception for VALU instructions that read and write the same SGPR;
  the second pass increments selected carry/divergence inputs and outputs, and
  wave32 ignores upper `EXEC`/`VCC` halves, at `rdna4/README.md:581` through
  `:593`.

XML evidence:

- XML encodes the top-level instruction families and operands, and one mask
  operand description notes that 64-bit masks may be truncated to 32 bits in
  wave32 at `amdgpu_isa_rdna4.xml:173686`.
- Searches found no field or architecture record for the fixed shader wave
  size, the VOPD exception, the functional-family issue-once/issue-twice
  split, wave64 half-skip rules, or the pre-instruction-state rule for both
  wave64 passes.
- Related narrower XML gaps remain recorded separately: `RDNA4-XML-012` covers
  wave32 mask semantics, `RDNA4-XML-013` covers `EXEC==0` skip behavior,
  `RDNA4-XML-052` through `RDNA4-XML-056` cover VOPD-specific restrictions,
  and `RDNA4-XML-076` covers SGPR mask/carry wave64 pass details.

Impact:

XML-only consumers can identify instruction families and operands, but cannot
derive RDNA4's wave32/wave64 issue scheduling, skipped-half behavior, or
wave64 scalar-input/pass interaction from XML alone.

### RDNA4-XML-163: Shader-type launch modes are absent from XML

Reported by: local audit.

Manual evidence:

- Chapter 2.2 describes compute shaders as generic dispatch programs over
  1D, 2D, or 3D grids, with hardware walking the grid, creating waves, and
  initializing each work-item with a unique grid address and index, at
  `rdna4/README.md:599`.
- The same section defines pixel, geometry, and hull shader launches, says the
  normal geometry engine launch initializes VGPRs with primitive index and
  vertex-buffer data, and describes mesh and amplification shader launch modes
  at `rdna4/README.md:603` through `:614`.

XML evidence:

- Searches found no shader-stage launch schema, grid-walk model, work-item
  initialization model, geometry-engine launch payload, mesh shader mode, or
  amplification shader mode in XML.
- XML contains only narrow stage-adjacent enumerator prose, such as
  `MSG_HS_TESSFACTOR` at `amdgpu_isa_rdna4.xml:181536` through `:181537` and
  `MSG_GS_ALLOC_REQ` at `:181548` through `:181549`.
- The more detailed per-stage initial-state payload gap is tracked separately
  as `RDNA4-XML-018`; this entry records that Chapter 2's shader-type and
  launch-mode taxonomy itself is not machine-readable.

Impact:

XML cannot drive or validate compute-vs-graphics shader wave creation, normal
geometry-engine VGPR setup, mesh-shader launch conversion, or amplification
shader control without manual prose or another launch ABI source.

### RDNA4-XML-164: Shader padding requirement is prose-only

Reported by: local audit.

Manual evidence:

- Chapter 2.4 says RDNA4 aggressively prefetches instructions, requires every
  shader to be padded with 64 extra DWORDs, or 256 bytes, beyond the end, and
  recommends `S_CODE_END` as padding to avoid prefetching unmapped or
  uninitialized memory, at `rdna4/README.md:638`.

XML evidence:

- XML models `S_CODE_END` as an instruction that causes an illegal-instruction
  interrupt and marks the end of a shader buffer for debug tools at
  `amdgpu_isa_rdna4.xml:34842` through `:34843`.
- XML also models `S_ENDPGM` as a program terminator at
  `amdgpu_isa_rdna4.xml:35121` through `:35122`.
- Searches found no XML representation of the 64-DWORD/256-byte padding length,
  the instruction-prefetch rationale, or the code-object validation rule.

Impact:

XML consumers can decode the suggested padding instruction, but cannot validate
the Chapter 2 shader-buffer padding requirement from XML alone.

### RDNA4-XML-165: Temporary WQM trigger list is not structured

Reported by: local audit.

Manual evidence:

- Chapter 2.5 defines whole quad mode as temporarily enabling all four threads
  in a pixel quad when any thread in that quad is active, at
  `rdna4/README.md:642`.
- It says `S_WQM_B32` and `S_WQM_B64` explicitly modify `EXEC` into a
  whole-quad mask, at `rdna4/README.md:644`.
- It then lists the instruction classes where WQM is temporarily applied:
  `V_INTERP_*`, `DS_PARAM_LOAD`, `DS_DIRECT_LOAD`, and sample operations that
  use neighboring-pixel data for LOD, at `rdna4/README.md:646` through `:650`.

XML evidence:

- XML models explicit `S_WQM_B32` and `S_WQM_B64` entries at
  `amdgpu_isa_rdna4.xml:15065` through `:15133`.
- XML instruction descriptions contain scattered quad or LOD prose, such as
  `DS_PARAM_LOAD` per-quad wording and sample LOD descriptions, but searches
  found no machine-readable relation that identifies the Chapter 2 temporary
  WQM trigger set across VINTERP, VDSDIR, and VSAMPLE instructions.
- Detailed per-instruction semantic gaps remain tracked in later sections,
  including `RDNA4-XML-126`, `RDNA4-XML-128`, `RDNA4-XML-131`,
  `RDNA4-XML-132`, and the Chapter 10 sample/LOD entries.

Impact:

XML-only consumers can decode explicit `S_WQM` instructions, but cannot infer
which non-`S_WQM` instructions temporarily widen `EXEC` for quad participation
without manual prose.

## No Gap Found In This Slice

- Chapter 2 instruction-role narrow match: XML has top-level functional group
  descriptions for SALU, SMEM, VALU, VMEM, and EXPORT at
  `amdgpu_isa_rdna4.xml:194367` through `:194386`, matching the broad
  processor-component roles listed in Chapter 2. The missing pieces are the
  cross-instruction execution, launch, and wave-scheduling contracts recorded
  in `RDNA4-XML-162` through `RDNA4-XML-165`.
- Chapter 2 workgroup boundary: the broad workgroup/LDS/barrier constraints in
  Chapter 2 overlap later detailed manual sections and are already tracked by
  entries such as `RDNA4-XML-033`, `RDNA4-XML-124`, and `RDNA4-XML-125`.
- Chapter 2 explicit WQM and padding-instruction narrow match: XML has
  `S_WQM_B32`, `S_WQM_B64`, `S_CODE_END`, and `S_ENDPGM` instruction entries.
  The gaps are the temporary WQM trigger relation and the 64-DWORD shader
  padding requirement, not the existence of those instruction shells.
- Chapter 1 narrow match: XML preserves the RDNA4 architecture identity and
  basic floating data-format bit layouts that correspond to several suffix
  definitions. Examples include BF16 at `amdgpu_isa_rdna4.xml:173297`, BF8 at
  `:173335`, F16 at `:173373`, F32 at `:173411`, F64 at `:173449`, and FP8 at
  `:173487`. The gap is the surrounding architecture, topology, host-control,
  and memory-system prose recorded in `RDNA4-XML-161`, not those basic format
  records.
- Chapter 5.8 narrow match: XML captures the raw `S_DELAY_ALU` `INSTID0`,
  `INSTID1`, and `INSTSKIP` bit layout and most dependency-code names through
  `OPR_DELAY`. It also tags the named transcendental VALU instruction families
  with the `TRANSCENDENTAL` subgroup, including `V_EXP`, `V_LOG`, `V_RCP`,
  `V_RSQ`, `V_SQRT`, `V_SIN`, and `V_COS` entries; the gap above is in the
  dynamic stream semantics and reserved-code drift.
- Chapter 5.7.2 narrow match: XML contains the opcode shells and operand
  formats needed to identify the named instructions and the affected hardware
  register IDs. The missing piece is the cross-instruction producer/consumer
  dependency relation recorded in `RDNA4-XML-159`.
- Chapter 7.6.1 narrow match: the XML has opcode and operand coverage for the
  audited `V_CVT*` conversion families, including representative normal/single,
  packed, and stochastic forms. The missing piece is the conversion-family
  rule metadata recorded in `RDNA4-XML-158`, not the basic instruction
  inventory.
- Chapter 15 broad no-gap: XML and the manual agree on the top-level format
  widths for SOP*, SMEM, VOP*, VOPD, DPP, VINTERP, VDSDIR, VDS,
  VBUFFER/VIMAGE/VSAMPLE, VFLAT/VGLOBAL/VSCRATCH, and VEXPORT encodings; the
  raw-field source drift found in this pass is limited to the memory-format
  `NV` bit recorded in `RDNA4-XML-148`.
- Chapter 15 literal/DPP narrow match: XML represents literal and DPP extension
  alternatives as separate encoding records, matching the manual's rule that no
  instruction may use both a DPP control DWORD and a literal constant.
- Chapter 16.1 SOP2 inventory narrow match: after normalizing markdown
  conversion artifacts around the scalar pack rows, all 74 manual SOP2
  definitions from `rdna4/README.md:8216` through `:9107` have matching XML
  `ENC_SOP2`/literal records. The four pack rows are present in the manual at
  `rdna4/README.md:8717` through `:8748` and in XML beginning with
  `S_PACK_LL_B32_B16` at `amdgpu_isa_rdna4.xml:29060`.
- Chapter 16.1 SOP2 literal narrow match: XML models ordinary SOP2 literal
  alternatives and the literal-only FMA forms. `S_FMAAK_F32` uses
  `SOP2_INST_LITERAL` as its default encoding with a fieldless `OPR_SIMM32`
  addend at `amdgpu_isa_rdna4.xml:30115` through `:30145`, while
  `S_FMAMK_F32` places the literal operand between `SSRC0` and `SSRC1` at
  `amdgpu_isa_rdna4.xml:30252` through `:30283`, matching the manual's
  `fma(S0, S1, SIMM32)` and `fma(S0, SIMM32, S1)` formulas.
- Chapter 16.1 SOP2 semantic boundary: the remaining issues found while
  checking this slice are already tracked by earlier SALU entries, including
  signed overflow wording in `RDNA4-XML-082`, scalar FP MODE/exception prose in
  `RDNA4-XML-083`, and scalar F16 high-half/storage metadata in
  `RDNA4-XML-084`.
- Chapter 16.2 SOPK inventory narrow match: all nine manual SOPK definitions
  from `rdna4/README.md:9107` through `:9292` have matching XML instruction
  records and opcode values, including the reserved holes between opcodes 2 and
  15 and after opcode 20.
- Chapter 16.2 SOPK literal narrow match: XML gives ordinary SOPK instructions
  only `ENC_SOPK` records, while `S_SETREG_IMM32_B32` uses
  `SOPK_INST_LITERAL` with a fieldless `OPR_SIMM32` operand at
  `amdgpu_isa_rdna4.xml:37587` through `:37605`, matching the manual's
  required 32-bit literal extension word.
- Chapter 16.2 SOPK semantic boundary: `S_ADDK_CO_I32` carries the same
  signed-overflow wording issue already tracked in `RDNA4-XML-082`, and
  `S_GETREG_B32` / `S_SETREG_B32` depend on the HWREG table and write-mask
  prose already tracked in the Chapter 3 and Chapter 6 state-register entries.
- Chapter 16.3 SOP1 inventory boundary: after normalizing the markdown-converted
  `S_WQM_B32` definition at `rdna4/README.md:9720`, the manual lists 85 SOP1
  definitions from `rdna4/README.md:9292` through `:10573`. XML matches the
  checked opcode values for all definitions except the already-recorded
  `S_GET_BARRIER_STATE` omission in `RDNA4-XML-033`.
- Chapter 16.3 SOP1 alias narrow match: XML includes canonical opcode entries
  for the `S_CTZ`/`S_CLZ`/`S_CLS`/`S_NOT*` families plus assembler alias names,
  and the detailed manual definitions match the opcode values after alias
  normalization.
- Chapter 16.3 SOP1 semantic boundary: scalar FP/F16, barrier, message-return,
  trap-return, dynamic-VGPR, sleep, HWREG, and SCC side effects reuse behavior
  already tracked in the Chapter 3, Chapter 5, and Chapter 6 entries; the new
  XML metadata issue from this SOP1 pass is the relative-SGPR index formula in
  `RDNA4-XML-150`.
- Chapter 16.4 SOPC inventory narrow match: all 46 manual SOPC definitions from
  `rdna4/README.md:10573` through `:10985` have matching XML instruction
  records and opcode values, including the reserved opcode holes after 17, at
  64, after 78, and after 94.
- Chapter 16.4 SOPC literal narrow match: every SOPC entry checked has one
  default `ENC_SOPC` record plus `SOPC_INST_LITERAL` alternatives for
  `has_lit_0`, `has_lit_1`, and `has_lit_0_has_lit_1`; representative
  `S_CMP_EQ_I32` records appear at `amdgpu_isa_rdna4.xml:28785` through
  `:28880`.
- Chapter 16.4 SOPC semantic boundary: XML carries operand widths, data formats,
  and implicit SCC outputs for scalar compares, but formula-level details remain
  descriptive rather than structured. The new missing structured detail from
  this pass is the bit-compare index mask in `RDNA4-XML-151`; scalar FP
  MODE/exception and scalar F16 storage rules remain in `RDNA4-XML-083` and
  `RDNA4-XML-084`.
- Chapter 16.5 SOPP inventory boundary: the manual lists 40 SOPP opcode
  definitions after normalizing the malformed markdown table around
  `S_INCPERFLEVEL`, `S_DECPERFLEVEL`, and the split wait instructions at
  `rdna4/README.md:11518` through `:11553`. XML has 41 `ENC_SOPP` entries; the
  only inventory differences found in this pass are the `S_WAITCNT` omission and
  opcode 58/59 trace additions recorded in `RDNA4-XML-152`.
- Chapter 16.5 SOPP operand narrow match: XML uses dedicated operands for the
  structured SOPP bitfields that the manual defines in detail, including
  `OPR_SLEEP` for `S_SLEEP`, `OPR_CLAUSE` for `S_CLAUSE`, `OPR_DELAY` for
  `S_DELAY_ALU`, `OPR_WAIT_ALU` for `S_WAIT_ALU`, `OPR_WAIT_EVENT` for
  `S_WAIT_EVENT`, and `OPR_WAIT_MEM_DS` for the combined memory/DS waits.
- Chapter 16.5 SOPP semantic boundary: trap/halt/sleep/clause/barrier/message,
  wait-counter, cache-invalidation, and branch-PC details mostly reuse the
  earlier Chapter 5 XML gaps. The new source issue found in the full SOPP
  definition pass is the reserved/compatibility opcode inventory drift recorded
  in `RDNA4-XML-152`.
- Chapter 16.6 SMEM inventory boundary: after excluding the XML-only ATC probe
  opcodes recorded in `RDNA4-XML-153`, the manual's 26 scalar load,
  scalar-buffer load, `S_DCACHE_INV`, and scalar prefetch definitions match XML
  opcode values and operand-family coverage.
- Chapter 16.6 SMEM semantic boundary: load address construction, buffer
  resource handling, destination/range/alignment restrictions, clause/group
  rules, and scalar prefetch cache behavior reuse the existing Chapter 8 XML
  gaps `RDNA4-XML-085` through `RDNA4-XML-090`; the new source issue found in
  this instruction-definition pass is the ATC probe inventory drift in
  `RDNA4-XML-153`.
- Chapter 16.7 VOP2 inventory boundary: the manual lists 49 VOP2 definitions at
  `rdna4/README.md:12075` through `:13650`. XML has matching opcode coverage
  once the four always-literal `V_FMAMK_F32`, `V_FMAAK_F32`, `V_FMAMK_F16`, and
  `V_FMAAK_F16` rows are counted through their `VOP2_INST_LITERAL` records
  rather than plain `ENC_VOP2`.
- Chapter 16.7 VOP2 operand narrow match: XML carries the compact VOP2 implicit
  `VCC` operands for `V_CNDMASK_B32`, `V_ADD_CO_CI_U32`,
  `V_SUB_CO_CI_U32`, and `V_SUBREV_CO_CI_U32`, for example
  `V_CNDMASK_B32` at `amdgpu_isa_rdna4.xml:69306` and `V_ADD_CO_CI_U32` at
  `:76384`. Broader source-combination and mask/carry legality metadata remains
  covered by `RDNA4-XML-074` and `RDNA4-XML-076`.
- Chapter 16.7 VOP2 semantic boundary: detailed FP min/max signed-zero/NaN,
  DPP, true16, source-combination, and packed-FMAC behavior mostly reuses the
  earlier Chapter 7 XML gaps; no new XML/source issue was found beyond those
  existing entries in this VOP2 definition pass.
- Chapter 16.8 VOP1 inventory boundary: the manual lists 90 VOP1 definitions
  from `rdna4/README.md:12824` through the FP8/BF8 tail at
  `:14348`, and XML has matching opcode/name coverage for plain `ENC_VOP1`
  plus the corresponding VOP3 aliases from `rdna4/README.md:14350` onward.
- Chapter 16.8 VOP1 selector narrow match: XML carries the expected operand
  widths for the FP8/BF8 packed converts at `amdgpu_isa_rdna4.xml:69116` and
  `:69211`; the manual's note that packed VOP1 forms use the usual OPSEL16
  rules at `rdna4/README.md:14332` through `:14348` aligns with the generic
  compact true16 source rule from Chapter 7.4.
- Chapter 16.8 VOP1 special-form boundary: XML includes the scalar destination
  for `V_READFIRSTLANE_B32`, operandless `V_PIPEFLUSH`, VOP1-only
  `V_PERMLANE64_B32`, and the relative-indexed/swap forms. The remaining
  semantic details overlap with existing prose-only Chapter 7 gaps rather than
  a new XML inventory issue in this pass.
- Chapter 14 narrow match: XML does carry the raw VEXPORT field map for
  `DONE`, `EN`, `ROW_EN`, `TGT`, and `VSRC0` through `VSRC3`, and it has
  predefined `OPR_TGT` names for MRT, Z, position, primitive, and dual-source
  blend targets at `amdgpu_isa_rdna4.xml:3447` through `:3534` and
  `:192783`.
- Chapter 11.1 narrow match: XML carries the raw RDNA4 VFLAT, VSCRATCH, and
  VGLOBAL instruction-word field layouts, including `SADDR`, `OP`, `SVE`,
  `SCOPE`, `TH`, `VSRC`, `VDST`, `VADDR`, `IOFFSET`, and the XML-only `NV`
  field recorded in `RDNA4-XML-148`, at
  `amdgpu_isa_rdna4.xml:3632` through `:3737`, `:3786` through `:3891`, and
  `:3981` through `:4086`. The missing pieces are the conditional and
  aperture-sensitive semantics recorded in `RDNA4-XML-106`.
- Chapter 11.1 narrow match: every Table 67 opcode name in the audited opening
  slice has XML coverage. XML also includes Chapter 11 opcodes whose detailed
  descriptions appear later in the manual, including flat/global
  `*_CMPSWAP_B64` and sub-clamp forms, block global/scratch loads/stores, and
  global transpose loads at `amdgpu_isa_rdna4.xml:48788`, `:49426`, `:52748`,
  `:53748`, `:53803`, `:53920`, `:53970`, `:172912`, and `:172967`.
- Chapter 11.1 narrow match: the no-resource-constant/sampler statement matches
  the XML operand shape for representative Flat/Global/Scratch forms, which
  expose VGPR, SGPR, and implicit `OPR_GPUMEM` operands but no resource or
  sampler descriptors at `amdgpu_isa_rdna4.xml:47735` through `:47770`,
  `:50679` through `:50700`, and `:171958` through `:171992`.
- Chapter 11.1 narrow match: `GLOBAL_LOAD_ADDTID_B32` omits `VADDR` and uses
  `VDST`, `SADDR`, and implicit memory operands at
  `amdgpu_isa_rdna4.xml:51630` through `:51652`, matching the manual's
  no-VGPR-addressing note.
- Chapter 11.2 narrow match: XML carries separate `ENC_VFLAT`, `ENC_VSCRATCH`,
  and `ENC_VGLOBAL` encodings for the three memory-instruction classes at
  `amdgpu_isa_rdna4.xml:3564`, `:3749`, and `:3903`.
- Chapter 11.2 narrow match: `IOFFSET` is consistently described as a 24-bit
  signed offset for Flat, Scratch, and Global encodings at
  `amdgpu_isa_rdna4.xml:3646` through `:3647`, `:3800` through `:3801`, and
  `:3995` through `:3996`, matching the Chapter 11.2 formula inputs.
- Chapter 11.2 narrow match: representative Scratch operands use 32-bit
  `VADDR` and `SADDR` fields at `amdgpu_isa_rdna4.xml:171975` through
  `:171985`, matching the manual's 32-bit source-offset widths before applying
  the signedness and mode-presence rules recorded above.
- Chapter 11.2 duplicate note: the buffer swizzled-addressing reference at
  `rdna4/README.md:5483` through `:5488` points back to the Chapter 9 buffer
  swizzle formula; the XML gap for buffer descriptor fields and swizzled buffer
  addressing is already recorded in `RDNA4-XML-091`.
- Chapter 11.3 narrow match: XML exposes generic wave exception and trap
  register/function names at `amdgpu_isa_rdna4.xml:180717` through `:180728`
  and `:194401` through `:194402`; the missing piece is the memory-error
  producer policy recorded in `RDNA4-XML-115`.
- Chapter 11.4 narrow match: the manual's "DATA field" wording maps to the
  `VSRC` field named in the Chapter 11 field table at `rdna4/README.md:5319`;
  XML exposes `VSRC`, not a separate `DATA` field, for VFLAT/VSCRATCH/VGLOBAL
  at `amdgpu_isa_rdna4.xml:4690`, `:4876`, and `:5100`.
- Chapter 11.4 narrow match: zero-to-four consecutive DWORD data payloads are
  represented by a single base VGPR field plus operand size metadata. For
  example, `FLAT_LOAD_B128` has 128-bit `VDST` metadata at
  `amdgpu_isa_rdna4.xml:50411` through `:50414`, and `FLAT_STORE_B128` has
  128-bit `VSRC` metadata at `:50681` through `:50684`.
- Chapter 11.4 wording note: XML descriptions for B16 D16 loads say
  "unsigned data", for example `FLAT_LOAD_D16_B16` at
  `amdgpu_isa_rdna4.xml:50795` through `:50799`, but the operand metadata uses
  `FMT_NUM_B16` at `:50819` and the manual's no-conversion rule means this is
  a textual precision issue rather than a distinct structured-data gap.
- Chapter 11.5 narrow match: XML contains all four block load/store opcodes,
  with `GLOBAL_LOAD_BLOCK` and `GLOBAL_STORE_BLOCK` at
  `amdgpu_isa_rdna4.xml:55951` and `:56003`, and `SCRATCH_LOAD_BLOCK` and
  `SCRATCH_STORE_BLOCK` at `:174034` and `:174086`.
- Chapter 11.5 narrow match: the maximum 32-DWORD transfer width is visible as
  1024-bit VGPR and memory operands on the block entries, for example
  `GLOBAL_LOAD_BLOCK` at `amdgpu_isa_rdna4.xml:55959` through `:55980` and
  `SCRATCH_STORE_BLOCK` at `:174100` through `:174116`; the sparse active-DWORD
  behavior is the separate gap recorded in `RDNA4-XML-118`.
- Chapter 11.5 narrow match: all four block entries include implicit `M0` as a
  32-bit operand at `amdgpu_isa_rdna4.xml:55982`, `:56034`, `:174065`, and
  `:174117`; XML omits the mask semantics recorded in `RDNA4-XML-118`.
- Chapter 11.5 reconfirms `RDNA4-XML-101`: the manual says one block transfer
  increments `LOADcnt` or `STOREcnt` once and decrements when the whole block
  completes at `rdna4/README.md:5566`, while XML does not attach wait-counter
  producer metadata to the block entries beyond their generic VMEM functional
  group at `amdgpu_isa_rdna4.xml:55990`, `:56042`, `:174073`, and `:174125`.
- Chapter 11.6 narrow match: XML contains both transpose-load opcodes with the
  manual opcode numbers: `GLOBAL_LOAD_TR_B128` opcode 87 and
  `GLOBAL_LOAD_TR_B64` opcode 88 at `amdgpu_isa_rdna4.xml:56113` through
  `:56119` and `:56160` through `:56166`; the manual opcode table lists those
  opcodes at `rdna4/README.md:8157` through `:8158`.
- Chapter 11.6 narrow match: XML uses `ENC_VGLOBAL` with `VDST`, `VADDR`,
  `SADDR`, and implicit global-memory operands for both transpose loads at
  `amdgpu_isa_rdna4.xml:56116` through `:56145` and `:56163` through `:56192`,
  matching the manual's statement that the fields are identical to
  `GLOBAL_LOAD_B64` and `GLOBAL_LOAD_B128` at `rdna4/README.md:5625`.
- Chapter 11.6 narrow match: XML and the manual instruction-definition entries
  agree on the high-level descriptions and nominal vector sizes for the two
  opcodes: `GLOBAL_LOAD_TR_B128` stores into a 128-bit vector register and
  `GLOBAL_LOAD_TR_B64` stores into a 64-bit vector register at
  `amdgpu_isa_rdna4.xml:56114`, `:56121`, `:56161`, and `:56168`, and
  `rdna4/README.md:30712` through `:30716`.
- Chapter 11.6 wording note: the prose table says `GLOBAL_LOAD_TR_B64` fills
  "16-bits in each of 8 VGPRs" after an 8-byte read at `rdna4/README.md:5623`,
  while both XML and the manual instruction-definition entry describe a 64-bit
  vector result. This appears to be a manual wording issue rather than an XML
  omission.
- Chapter 11.6 reconfirms `RDNA4-XML-101`: the manual says transpose loads are
  tracked with `LOADcnt` at `rdna4/README.md:5625`, while XML only gives them a
  generic `VMEM`/`TEXTURE` functional group at `amdgpu_isa_rdna4.xml:56147` and
  `:56194`.
- Chapter 10.1 narrow match: XML carries the raw RDNA4 VIMAGE/VSAMPLE
  instruction-word field layout for `DIM`, `R128`, `D16`, `A16`, `OP`,
  `DMASK`, `VDATA`, `RSRC`, `SCOPE`, `TH`, `TFE`, `LWE`, `UNORM`, `SAMP`,
  `VADDR*`, and the XML-only `NV` field recorded in `RDNA4-XML-148`, at
  `amdgpu_isa_rdna4.xml:2902` through `:3138` and `:3149` through `:3436`;
  the missing pieces are the dynamic and table-driven semantics recorded in
  `RDNA4-XML-094` through `RDNA4-XML-105`.
- Chapter 10.1 narrow match: the XML `UNORM` description at
  `amdgpu_isa_rdna4.xml:4285` through `:4292` captures the manual's normalized
  versus unnormalized coordinate rule, store/atomic requirement, non-sampler
  no-effect rule, UINT input rule, and logical-OR with sampler
  `FORCE_UNNORMALIZED`.
- Chapter 10.1-10.3 narrow match: XML has broad opcode inventory for the
  audited VIMAGE and VSAMPLE families, including loads, stores, atomics,
  `IMAGE_GET_RESINFO`, `IMAGE_MSAA_LOAD`, `IMAGE_SAMPLE*`, G16 sample forms,
  `IMAGE_GATHER4*`, `IMAGE_GATHER4H`, and `IMAGE_GET_LOD`.
- Chapter 10.1/10.4 narrow match: XML `DMASK` prose includes D16 write packing,
  `DMASK==0`, and gather4/MSAA-like return counts at
  `amdgpu_isa_rdna4.xml:3868` through `:3883` and `:4161` through `:4176`;
  the gap above is that these rules are not fully structured and do not cover
  all manual store/atomic details.
- Chapter 10.3 narrow match: XML instruction descriptions partially preserve
  sample suffix intent for LOD, bias, PCF, offsets, derivatives, LOD clamp, and
  G16 forms, for example around `amdgpu_isa_rdna4.xml:170302`, `:170349`,
  `:170490`, `:170725`, `:171665`, and `:172605`. For PCF forms, this captures
  the extra z-compare address data, but the sampler descriptor's depth-compare
  function remains covered by `RDNA4-XML-099`.
- Chapter 10.1/resource narrow match: XML records image and sampler operands as
  opaque descriptors and carries the raw VIMAGE/VSAMPLE `R128` bit. `FMT_IMG`
  is at `amdgpu_isa_rdna4.xml:173057`, `FMT_IMG_BVH` at `:173077`, and
  `FMT_SAMP` at `:175350`; the missing structured resource detail is recorded
  in `RDNA4-XML-098`.
- Chapter 10.1/resource narrow match: XML records VIMAGE/VSAMPLE `RSRC` as the
  SGPR selector for the resource descriptor and preserves the multiple-of-4
  alignment requirement at `amdgpu_isa_rdna4.xml:3932` through `:3937` and
  `:4235` through `:4240`.
- Chapter 10.5 narrow match: XML records that `IMAGE_GET_RESINFO` returns
  `{ num_mip_levels, depth, height, width }` and performs no memory access at
  `amdgpu_isa_rdna4.xml:57478` through `:57479`; descriptor field extraction
  remains unstructured and is covered by `RDNA4-XML-098`.
- Chapter 10.6 narrow match: XML records the sampler operand footprint as an
  opaque 128-bit descriptor (`FMT_SAMP`) and preserves the VSAMPLE `SAMP`
  selector alignment requirement at `amdgpu_isa_rdna4.xml:175350` through
  `:175368` and `:4245` through `:4250`; the missing field-level sampler
  semantics are recorded in `RDNA4-XML-099`.
- Chapter 10.6 narrow match: XML explicitly preserves the instruction-side
  `UNORM` interaction with `SAMP.FORCE_UNNORMALIZED` at
  `amdgpu_isa_rdna4.xml:4285` through `:4292`, although the sampler bit itself
  remains opaque.
- Chapter 10.3/10.6 narrow match: `IMAGE_MSAA_LOAD` uses `ENC_VSAMPLE` but has
  no `SAMP` operand in XML at `amdgpu_isa_rdna4.xml:170214` through `:170239`,
  matching the manual's statement that MSAA load does not use a sampler.
- Chapter 10.7 narrow match: XML preserves the raw instruction-side typed-buffer
  `FORMAT` field as a 7-bit `ENC_MTBUF` field at
  `amdgpu_isa_rdna4.xml:3649` through `:3656`; the missing enum mapping and
  format semantics are recorded in `RDNA4-XML-100`.
- Chapter 10.7 narrow match: XML models image and typed-buffer resource
  operands as opaque descriptors, with `FMT_IMG` at
  `amdgpu_isa_rdna4.xml:173057` through `:173075` and `FMT_RSRC_TYPED` at
  `:175230` through `:175248`. This preserves the operand footprint, while the
  descriptor-internal data-format field remains unstructured as recorded in
  `RDNA4-XML-098` and `RDNA4-XML-100`.
- Chapter 10.8 narrow match: XML exposes the RDNA4 split wait instructions
  `S_WAIT_LOADCNT`, `S_WAIT_STORECNT`, `S_WAIT_SAMPLECNT`, and `S_WAIT_BVHCNT`
  at `amdgpu_isa_rdna4.xml:38613` through `:38701`, matching the manual's named
  counters at the instruction-inventory level.
- Chapter 10.8 narrow match: XML encodes the combined memory/DS wait operand as
  `OPR_WAIT_MEM_DS`, with `DS` in bits 5:0 and `MEM` in bits 13:8 at
  `amdgpu_isa_rdna4.xml:195495` through `:195511`, matching the immediate
  layout for `S_WAIT_LOADCNT_DSCNT` and `S_WAIT_STORECNT_DSCNT`.
- Chapter 10.9 narrow match: XML contains RDNA4 BVH opcode entries and aliases
  for `IMAGE_BVH_INTERSECT_RAY`/`BVH_INTERSECT_RAY`,
  `IMAGE_BVH64_INTERSECT_RAY`/`BVH64_INTERSECT_RAY`,
  `IMAGE_BVH_DUAL_INTERSECT_RAY`/`BVH_DUAL_INTERSECT_RAY`, and
  `IMAGE_BVH8_INTERSECT_RAY`/`BVH8_INTERSECT_RAY` at
  `amdgpu_isa_rdna4.xml:55405` through `:55581`, and tags them as `VMEM`,
  `TEXTURE`, and `BVH`.
- Chapter 10.9 narrow match: XML records the non-`A16` BVH operand footprints:
  4/10 destination VGPRs, 11/12 address VGPRs, and 128-bit `FMT_IMG_BVH`
  resource operands at `amdgpu_isa_rdna4.xml:55416` through `:55572`. The
  missing `A16` variants, VADDR grouping, and descriptor-internal semantics are
  recorded in `RDNA4-XML-102` through `RDNA4-XML-104`.
- Chapter 10.10 narrow match: XML exposes the raw `TFE`/`LWE` selector bits for
  image/sample encodings and preserves the `DMASK==0` exception that suppresses
  TFE status at `amdgpu_isa_rdna4.xml:3066`, `:3288`, `:3358`, `:2995`, and
  `:3267`. The missing PRT enablement, status destination, payload, and
  out-of-range rules are recorded in `RDNA4-XML-105`.
- Chapter 9.1 narrow match: XML carries the RDNA4 VBUFFER instruction-word
  field layout for `OP`, `SOFFSET`, `VADDR`, `VDATA`, `IOFFSET`, `RSRC`,
  `FORMAT`, `OFFEN`, `IDXEN`, `SCOPE`, `TH`, `TFE`, and the XML-only `NV`
  field recorded in `RDNA4-XML-148`; the missing pieces are the descriptor and
  behavioral semantics recorded in
  `RDNA4-XML-091` through `RDNA4-XML-093`.
- Chapter 9.1 narrow match: XML has broad opcode and operand coverage for the
  audited typed and untyped buffer families, including plain load/store,
  formatted load/store, D16/D16_HI variants, TBUFFER variants, and atomics.
- Chapter 9.2 narrow match: XML operand sizes are enough to recover ordinary
  consecutive VGPR counts for many buffer loads/stores and atomics, while the
  semantic packing and status-destination details remain in the gaps above.
- Chapter 8.1 narrow match: XML has the SMEM field bit positions and widths for
  `SBASE`, `SDATA`, `OP`, `SCOPE`, `TH`, `ENCODING`, `IOFFSET`, `SOFFSET`, and
  the XML-only `NV` field recorded in `RDNA4-XML-148`, at
  `amdgpu_isa_rdna4.xml:843` through `:923`.
- Chapter 8.1 narrow match: the audited SMEM opcode inventory is present in
  XML for `S_LOAD_B32/B64/B96/B128/B256/B512`, `S_LOAD_I8/U8/I16/U16`,
  `S_BUFFER_LOAD_B32/B64/B96/B128/B256/B512`,
  `S_BUFFER_LOAD_I8/U8/I16/U16`, and `S_DCACHE_INV`.
- Chapter 8.1.2 narrow match: XML distinguishes raw-pointer `S_LOAD` bases as
  64-bit `FMT_BUF` operands and scalar-buffer bases as 128-bit
  `FMT_RSRC_SCALAR` operands; the missing piece is the descriptor semantics
  recorded in `RDNA4-XML-086`.
- Chapter 8.1.3 narrow match: XML records signed and unsigned narrow-load
  intent through `S_LOAD_I8/U8/I16/U16` and `S_BUFFER_LOAD_I8/U8/I16/U16`
  instruction descriptions, destination formats, and implicit 8-bit/16-bit
  memory operand sizes.
- Chapter 8.1.4 narrow match: `S_DCACHE_INV` exists with opcode 33, SMEM
  encoding, and no operands at `amdgpu_isa_rdna4.xml:16919` through `:16930`.
- Chapter 8.4 narrow match: SBASE evenness is structurally encoded in XML
  because the SMEM `SBASE` field stores bits `[6:1]` with an implied zero low
  bit at `amdgpu_isa_rdna4.xml:873` through `:883`; the missing behavior is the
  out-of-range fallback recorded in `RDNA4-XML-089`.
- Chapter 8.5 narrow match: XML has opcode and operand coverage for all five
  scalar prefetch forms, and the `SDATA` immediate length contribution is
  visibly constrained by `OPR_SIMM5`; the missing pieces are the semantic formulas
  and cache/mode effects in `RDNA4-XML-090`.
- Chapter 8.2 narrow match: XML has `S_WAIT_KMCNT` with SOPP opcode 71 and a
  `SIMM16[4:0]` threshold description at `amdgpu_isa_rdna4.xml:38787` through
  `:38805`; the missing pieces are the SMEM producer and ordering semantics in
  `RDNA4-XML-044`.
- Chapter 8.2 narrow match: XML SMEM load operand sizes are sufficient to
  distinguish one-dword loads from wider loads in simple consumers, but the
  one-count versus two-count `KMcnt` rule is not encoded as structured metadata.
- Chapter 6.1/6.2 narrow match: the concrete RDNA4 XML instruction table
  restricts the SOPK literal encoding to `S_SETREG_IMM32_B32` / opcode 19,
  matching the manual's one SOPK exception despite the broader encoding-family
  description above.
- Chapter 6.3-6.7 narrow match: the XML does expose SCC as an implicit operand
  on arithmetic, compare, bitwise, WREXEC, and conditional-move entries. The
  gap above is limited to the prose description of signed add/sub SCC meaning.
- Chapter 6.4 narrow match: the detailed instruction definitions and XML agree
  that `S_LSHL{1,2,3,4}_ADD_U32` shifts the first source before adding the
  second source and sets SCC from unsigned overflow; this matches the detailed
  Chapter 16 definitions despite the compact Chapter 6.4 table row's ambiguous
  `D!=0` wording.
- Chapter 6.4 narrow match: the XML canonical names `S_ADD_NC_U64` and
  `S_SUB_NC_U64` carry aliases for the Chapter 6 table names `S_ADD_U64` and
  `S_SUB_U64`, so the no-SCC 64-bit add/sub forms are represented.
- Chapter 6.7 narrow match: the `S_AND_NOT{0,1}_WREXEC_B{32,64}` explicit
  destination uses `OPR_SREG` in XML, and `OPR_SREG` includes SGPR, TTMP, VCC,
  and `NULL` but not `M0` or `EXEC`; the manual's `D cannot be EXEC`
  restriction is therefore structurally recoverable.
- Chapter 6.8 narrow match: the scalar floating-point instruction inventory,
  broad `SALU` / `FLOATING_POINT` grouping, and F16/F32 operand formats are
  present in XML; the gaps above are in mode/exception semantics and raw scalar
  storage rules, not basic opcode coverage.
- Chapter 6.9 narrow match: XML has `S_GETREG_B32`, `S_SETREG_B32`, and
  `S_SETREG_IMM32_B32` with `OPR_HWREG` operands and the literal source for
  `S_SETREG_IMM32_B32`; the missing HWREG bitfields and side effects remain
  tracked under `RDNA4-XML-009`, while the broader SOPP/SOPK literal wording is
  tracked under `RDNA4-XML-081`.
- Chapter 6.10 narrow match: XML names source selector values 235 through 238
  for `SHARED_BASE`, `SHARED_LIMIT`, `PRIVATE_BASE`, and `PRIVATE_LIMIT`; the
  missing part is the aperture formula and state behavior recorded in
  `RDNA4-XML-023`.
- The XML has the requested FP8/BF8 conversion instruction entries and opcodes
  for the audited normal, packed, and stochastic conversion names.
- The XML matches the basic FP8/BF8 storage widths: FP8 E4M3 and BF8 E5M2.
- `V_CVT_PK_FP8_F32` and `V_CVT_PK_BF8_F32` round-to-nearest-even wording is
  present in the XML descriptions at `amdgpu_isa_rdna4.xml:119351` and
  `:119517`.
- `V_CVT_PK_F32_FP8` / `V_CVT_PK_F32_BF8` 16-bit source half selection appears
  covered by the generic VOP3 OPSEL field prose. Their definition inventory is
  covered by the full Chapter 16.12 row above.
- Chapter 3 narrow matches: scalar register numeric mappings, VGPR namespace,
  `S_GETREG_B32`/`S_SETREG_B32` `OPR_HWREG` ID/OFFSET/SIZE encoding, TBA/TMA
  `S_SENDMSG_RTN` names, and the shader-cycle rollover read procedure are at
  least nominally represented.
- Chapter 4.1 narrow matches: the XML carries the basic scalar and vector source
  selector numeric map, including SGPRs, VCC, TTMP, `NULL`, `M0`, `EXEC`,
  integer inline constants, float inline constants, `SCC`, literal constant,
  and VGPR values.
- Chapter 4.1.1 narrow matches: the XML includes `SCOPE` and `TH` bit positions
  and widths for SMEM and VMEM formats; the missing pieces are the enum tables
  and behavioral policy.
- Chapter 5.1/5.2 narrow matches: most manual-listed program-flow instruction
  names exist in XML, including `S_ENDPGM`, `S_ENDPGM_SAVED`, `S_TRAP`,
  `S_RFE_B64`, `S_SETHALT`, `S_SLEEP`, `S_SLEEP_VAR`, `S_WAKEUP`,
  `S_SETPRIO`, `S_CLAUSE`, `S_BARRIER_SIGNAL`,
  `S_BARRIER_SIGNAL_ISFIRST`, `S_BARRIER_WAIT`, `S_VERSION`, `S_CODE_END`,
  `S_SENDMSG`, `S_SENDMSG_RTN_B32/B64`, `S_SENDMSGHALT`, `S_ICACHE_INV`,
  `S_ROUND_MODE`, and `S_DENORM_MODE`; `S_GET_BARRIER_STATE` is the standout
  missing instruction in this slice.
- Chapter 5.2 narrow matches: `OPR_SLEEP` captures `S_SLEEP` duration and
  sleep-forever fields, `OPR_CLAUSE` captures length and break-span fields, and
  `OPR_SENDMSG` / `OPR_SENDMSG_RTN` capture message enum structure.
- Chapter 5.3 narrow matches: XML includes `S_CLAUSE`, `S_DELAY_ALU`, and
  `S_WAIT_ALU` instruction entries plus detailed `OPR_DELAY` and
  `OPR_WAIT_ALU` operand field layouts; the missing pieces are the clause
  membership and instruction-stream rules.
- Chapter 5.3 narrow matches: XML functional groups/subgroups expose some raw
  ingredients for clause classification, such as `SMEM`, `VALU`, `VMEM`,
  `LOAD`, `STORE`, `SAMPLE`, `BVH`, `ATOMIC`, `FLAT`, and `DATA_SHARE`, and
  `V_S_*` pseudo-scalar transcendental instructions are classified as VALU.
- Chapter 5.4 narrow matches: XML names the main `S_SENDMSG` and
  `S_SENDMSG_RTN_B32/B64` instructions, uses `OPR_SENDMSG` for the SOPP
  immediate, uses `OPR_SENDMSG_RTN` for the SOP1 `SSRC0` field, declares
  `S_SENDMSG` implicit `M0`, includes the listed non-return message codes
  `0x01`, `0x02`, `0x03`, and `0x09`, and includes most manual-listed return
  message codes.
- Chapter 5.5 narrow matches: XML has `S_GETPC_B64`, `S_SETPC_B64`,
  `S_SWAPPC_B64`, `S_CALL_B64`, `S_BRANCH`, and the six manual-listed
  `S_CBRANCH_*` forms; branch operands use `FMT_NUM_I16` plus `OPR_LABEL`, and
  `OPR_LABEL` captures signed dword offsets relative to `PC+4`.
- Chapter 5.5 narrow matches: XML branch flags distinguish non-branch
  `S_GETPC_B64`, direct `S_BRANCH`/`S_CBRANCH_*`/`S_CALL_B64`, and indirect
  `S_SETPC_B64`/`S_SWAPPC_B64`; the SET/SWAP/CALL/GET entries also expose the
  relevant 64-bit scalar operands and implicit PC operands.
- Chapter 5.6 narrow matches: XML includes `S_BARRIER_SIGNAL`,
  `S_BARRIER_SIGNAL_ISFIRST`, and `S_BARRIER_WAIT`; `ISFIRST` exposes an
  implicit `SCC` output, and `S_BARRIER_WAIT` is modeled as SOPP opcode `20`
  with a `SIMM16` operand rather than an `M0` source.
- Chapter 5.6 narrow matches: XML partially captures `STATE_PRIV` access
  privilege through `hw_reg_wave_state_priv`, which is readable by all and
  writable only when `PRIV=1`; the missing piece is the barrier-completion bit
  layout and behavior.
- Chapter 5.7.1 narrow matches: XML matches the detailed manual names and SOPP
  opcodes for `S_WAIT_LOADCNT`, `S_WAIT_STORECNT`, `S_WAIT_SAMPLECNT`,
  `S_WAIT_BVHCNT`, `S_WAIT_EXPCNT`, `S_WAIT_DSCNT`, `S_WAIT_KMCNT`,
  `S_WAIT_LOADCNT_DSCNT`, and `S_WAIT_STORECNT_DSCNT`.
- Chapter 5.7.1 narrow matches: XML matches the detailed manual combined-wait
  bit layout, with `DS = SIMM16[5:0]` and `MEM = SIMM16[13:8]`, despite the
  Chapter 5.7 summary table's broader `SIMM16[7:0]` / `SIMM16[15:8]` wording.
- Chapter 5.7.1 narrow matches: XML captures the `S_WAIT_EVENT` export-ready
  bit and zero-as-no-wait behavior; reserved-bit and exception-ordering details
  remain in the gap above.
- Chapter 7.7 narrow match: the XML VOP3P DPP encoding alternatives are
  present only for `V_DOT2_F32_F16`, `V_DOT2_F32_BF16`, and `V_FMA_MIX*`
  entries in this slice; the audited DOT4/DOT8/WMMA entries do not expose DPP
  alternatives despite the generic VOP3P format description.
- Chapter 7.7 narrow match: after normalizing markdown escapes, every
  manual-listed packed VOP3P opcode in the audited opcode set exists as an XML
  instruction entry; the gaps above are in modifier and source-constant
  semantics, not basic packed opcode inventory.
- Chapter 7.7 narrow match: XML carries the generic VOP3P `CLAMP`, `NEG`,
  `NEG_HI`, `OPSEL`, and `OPSEL_HI` field positions and widths across the base,
  literal, and DPP encodings; the missing pieces are the instruction-specific
  overloads tracked above.
- Chapter 7.8 narrow match: the XML VOPD prose captures the broad wave32-only,
  no-VALU-modifier, scalar-input, source-cache, `SRC2`, destination parity, and
  no-DPP restrictions, but several details above are missing, ambiguous, or
  only prose.
- Chapter 7.8 narrow match: the XML instruction inventory and encoding
  identifier masks cover the audited VOPD OPX/OPY opcode names and the 64-bit
  and 96-bit VOPD field shapes.
- Chapter 7.9 narrow match: XML structurally represents DPP as a `SRC0`
  sentinel plus a following DPP extension word; `SRC0 == 250` selects DPP16 and
  `SRC0 == 233` or `234` selects the two DPP8 marker values.
- Chapter 7.9 narrow match: the raw DPP8 lane selector bit layout is present in
  XML through `LANE_SEL_0` and adjacent selector fields, and DPP16 exposes raw
  `VSRC0`, `DPP_CTRL`, `FI`, `BOUND_CTRL`, source-modifier, row-mask, and
  bank-mask fields.
- Chapter 7.9 narrow match: representative instruction availability is encoded
  per instruction rather than only through Table 38; for example, `V_MOV_B32`
  has DPP variants while `V_NOP` and `V_READFIRSTLANE_B32` omit them.
- Chapter 7.10 narrow match: XML contains all ten pseudo-scalar `V_S_*`
  instruction names, with VOP3 opcodes 640 through 649 and broad
  `VALU`/`TRANSCENDENTAL` classification.
- Chapter 7.10 narrow match: XML captures the manual's `VDST`-not-`SDST`
  placement for pseudo-scalar destinations; the gaps above are in special
  destination legality and F16 result behavior.
- Chapter 7.11 narrow match: XML contains all five VGPR-indexing instruction
  names and marks `M0` as an implicit operand, and `V_SWAPREL_B32` marks both
  explicit operands as input/output.
- Chapter 7.1 narrow match: XML contains the core encoding records and raw
  field layouts for VOP1, VOP2, VOPC, VOP3, VOP3P, VOP3SD, and VOPD; the gaps
  above are in legality and semantic metadata rather than basic field presence.
- Chapter 7.1 narrow match: VOP3SD instruction inventory matches the manual's
  limited list and excludes `V_CMP*`; VOP3SD instruction operands use
  `OPR_SREG`, whose value set includes `null` and excludes `M0`/`EXEC`.
- Chapter 7.1 narrow match: the manual's non-promotable VOP1/VOP2 exception
  list is reflected by absence of VOP3 encodings for `SWAP`, `SWAPREL`,
  `PERMLANE64_B32`, `FMAMK`, `FMAAK`, and `PK_FMAC`.
- Chapter 7.1 narrow match: XML records 96-bit literal alternatives for
  VOP3-family formats; the manual's note that excessive VOP3 literal use may
  reduce performance is outside this decoder/semantics audit scope.
- Chapter 7.2 narrow match: XML carries the basic VALU source selector table,
  including SGPR/VCC/TTMP/NULL/M0/EXEC, inline constants, SCC, the literal
  selector, DPP selector values, aperture selector values, and VGPR values; the
  gaps above are in cross-operand legality and behavior metadata.
- Chapter 7.2 narrow match: XML captures the non-standard operand classes for
  `V_READLANE_B32`, `V_WRITELANE_B32`, and `V_READFIRSTLANE_B32` at a field
  level; the gaps above are in lane masking and broader stream legality.
- Chapter 7.3 narrow match: the XML contains every explicit non-VOP3P VALU
  mnemonic listed in the Chapter 7.3 inventory table after accounting for XML
  aliases such as `V_CVT_FLR_I32_F32`, `V_CVT_PKRTZ_F16_F32`,
  `V_CVT_PKNORM_*`, and `V_FMA_LEGACY_F32`.
- Chapter 7.3 narrow match: table classification is recoverable from concrete
  XML encoding names. VOP1 and VOP2 entries expose `ENC_VOP1` or `ENC_VOP2`,
  VOP3 entries expose `ENC_VOP3` or `VOP3_SDST_ENC`, and the always-literal
  `FMAMK`/`FMAAK` VOP2 forms expose `VOP2_INST_LITERAL`.
- Chapter 7.3 ambiguity note: the compact compare-family rows mention integer
  `F`/`LG`/`T` and floating `F`/`T` predicates, but the detailed Chapter 15
  opcode table and XML concrete instruction names omit `F`/`T` compare opcodes
  and use integer `NE` names. The XML matches the detailed opcode table rather
  than that summary wording.
- Chapter 16.9 VOPC inventory narrow match: after normalizing markdown
  conversion artifacts for a few detailed compare headings, the manual's 162
  concrete compact VOPC definitions match the XML `ENC_VOPC` entries, generated
  opcode constants, and compact VOPC decoder table. The `F`/`TRU` compare
  operation table rows remain non-concrete summary operations, as noted in the
  Chapter 7.3 ambiguity note.
- Chapter 16.9 VOPC operand narrow match: XML records the compact compare
  result operand for non-`CMPX` VOPC forms as `OPR_VCC`, for example
  `V_CMP_LT_F16` at `amdgpu_isa_rdna4.xml:125999` through `:126013`, and
  records `CMPX` result operands as `OPR_EXEC` plus an implicit
  `OPR_SDST_EXEC`, for example `V_CMPX_LT_F16` at
  `amdgpu_isa_rdna4.xml:144846` through `:144872`.
- Chapter 16.9 VOPC class-compare narrow match: XML carries the concrete
  `V_CMP_CLASS_*` and `V_CMPX_CLASS_*` instruction entries, while the manual's
  ten-bit class-mask order is explicit in the Chapter 16.9 definitions at
  `rdna4/README.md:15513` through `:15555` and `:16641` through `:16677`.
- Chapter 16.10 VOP3P inventory narrow match: after normalizing markdown
  conversion artifacts for several detailed headings, the manual's 56 concrete
  VOP3P definitions match the XML `ENC_VOP3P` entries. The detailed definition
  section runs from `rdna4/README.md:16795` through `:17801`, and XML entries
  span `amdgpu_isa_rdna4.xml:118013` through `:125799`.
- Chapter 16.10 VOP3P opcode-table narrow match: the manual's Chapter 15.3.6
  VOP3P opcode table lists the same packed, DOT, MIX, WMMA, and SWMMAC opcodes
  at `rdna4/README.md:7498` through `:7529`. The XML exposes matching raw
  VOP3P field and instruction records; the missing/ambiguous pieces remain the
  instruction-specific modifier, selector, DPP, WMMA layout, and hazard prose
  already tracked in `RDNA4-XML-045` through `RDNA4-XML-051`,
  `RDNA4-XML-064`, `RDNA4-XML-073`, and `RDNA4-XML-080`.
- Chapter 16.11 VOPD inventory narrow match: after normalizing markdown
  escapes, the manual lists 14 X-slot definitions and 17 Y-slot definitions
  from `rdna4/README.md:17811` through `:18293`. XML carries the same 17
  `V_DUAL_*` instruction names from `amdgpu_isa_rdna4.xml:168356` through
  `:170148`.
- Chapter 16.11 VOPD slot-asymmetry narrow match: XML models `OPX` as a
  4-bit field and `OPY` as a 5-bit field at
  `amdgpu_isa_rdna4.xml:15599` through `:15614`, and the Y-only opcodes
  `V_DUAL_ADD_NC_U32`, `V_DUAL_LSHLREV_B32`, and `V_DUAL_AND_B32` expose only
  `VDSTY` / `SRCY0` / `VSRCY1` operands at
  `amdgpu_isa_rdna4.xml:170016` through `:170190`.
- Chapter 16.11 VOPD semantic boundary: core slot descriptions for fused FMA,
  literal FMAAK/FMAMK, DX9-zero multiply, CNDMASK, DOT2ACC, shift, add, and
  bitwise AND are present as XML instruction descriptions/operands. The
  remaining VOPD literal, legality, source-cache, paired-exception, and
  min/max edge-metadata issues are already recorded in `RDNA4-XML-052` through
  `RDNA4-XML-056` and `RDNA4-XML-154`.
- Chapter 16.12 VOP3/VOP3SD inventory narrow match: after normalizing ten
  markdown conversion artifacts where instruction headings were emitted as
  fenced-code lines, all 444 manual definitions from `rdna4/README.md:18317`
  through `:25054` match XML VOP3-family instruction records by name and
  opcode.
- Chapter 16.12 VOP3SD boundary: the manual's ten VOP3SD-only opcodes at
  `rdna4/README.md:18303` through `:18313` match the XML `VOP3_SDST_ENC`
  instruction set, and no `V_CMP*`/`V_CMPX*` entry uses VOP3SD. The remaining
  VOP3SD field metadata issue is already recorded in `RDNA4-XML-070`.
- Chapter 16.12 VOP3 semantic boundary: XML carries the VOP3/VOP3SD literal and
  DPP alternatives, compare result operand classes, and generic modifier/OPSEL
  fields. The missing instruction-specific pieces remain the earlier generic
  VOP3 modifier, OPSEL, DPP compare-zeroing, and CMPX hazard gaps recorded in
  `RDNA4-XML-060`, `RDNA4-XML-071`, `RDNA4-XML-072`, and `RDNA4-XML-078`, plus
  the min/max edge-metadata gap in `RDNA4-XML-154`.
- Chapter 16.13 VINTERP inventory narrow match: the manual's six detailed
  VINTERP definitions from `rdna4/README.md:25062` through `:25194` match the
  six XML `ENC_VINTERP` records at `amdgpu_isa_rdna4.xml:57958` through
  `:58233` by name and opcode.
- Chapter 16.13 VINTERP operand-width boundary: XML distinguishes F32,
  F16/F32, and RTZ F16/F32 operand widths, including the F16 P2 destination
  forms. This matches the instruction names and OPSEL destination-half notes;
  the remaining unstructured fixed-DPP, FMA, RTZ, OPSEL-legality, and wait
  semantics are already recorded in `RDNA4-XML-144`.
- Chapter 16.14 VDSDIR inventory narrow match: the manual's `DS_PARAM_LOAD`
  opcode 0 and `DS_DIRECT_LOAD` opcode 1 definitions at `rdna4/README.md:25202`
  through `:25218` match the checked XML `ENC_VDSDIR` records at
  `amdgpu_isa_rdna4.xml:49933` through `:50014`, including the
  `LDS_PARAM_LOAD` and `LDS_DIRECT_LOAD` aliases.
- Chapter 16.14 VDSDIR semantic boundary: XML carries the raw VDSDIR field
  layout for `ATTR`, `ATTR_CHAN`, `VDST`, `WAIT_VA_VDST`, `OP`, and
  `WAIT_VM_VSRC` at `amdgpu_isa_rdna4.xml:3217` through `:3280`, plus
  implicit M0/LDS operands for the two load definitions. The detailed
  CU-only, readiness, `EXPcnt`, M0 format, quad fill/broadcast, FP16 packing,
  and direct-load datatype/extension gaps remain the earlier
  `RDNA4-XML-126` through `RDNA4-XML-133`.
- Chapter 16.15 VDS inventory narrow match: after normalizing markdown
  conversion artifacts where some headings were emitted as fenced-code lines,
  all 123 manual definitions from `rdna4/README.md:25220` through `:27240`
  match the checked XML `ENC_VDS` records from `amdgpu_isa_rdna4.xml:43966`
  through `:49896` by name and opcode.
- Chapter 16.15 VDS semantic boundary: XML covers the shared `ENC_VDS` raw
  fields plus visible operand footprints for ordinary loads/stores, 2-address
  and stride64 forms, integer/FP atomics, byte/short and D16 loads/stores,
  ADDTID, lane permute/swizzle, BVH stack, and B96/B128 data movement. The
  missing structured semantics remain the Chapter 12/13 DS entries
  `RDNA4-XML-134` through `RDNA4-XML-143`, `RDNA4-XML-145`, and the new
  special-atomic predicate gap in `RDNA4-XML-155`.
- Chapter 16.16 VBUFFER inventory narrow match: after normalizing markdown
  conversion artifacts, all 89 manual definitions from
  `rdna4/README.md:27241` through `:28332` match the checked XML
  `ENC_VBUFFER` records from `amdgpu_isa_rdna4.xml:38874` through `:43917` by
  name and opcode.
- Chapter 16.16 VBUFFER semantic boundary: XML carries the shared
  `ENC_VBUFFER` raw field layout plus visible operand footprints for formatted
  loads/stores, D16/D16_HI variants, plain buffer loads/stores, integer and
  floating-point atomics, packed F16/BF16 atomics, and typed `TBUFFER_*`
  operations. Descriptor/address/data-format behavior remains the earlier
  Chapter 9 entries `RDNA4-XML-091` through `RDNA4-XML-093`, TFE remains
  `RDNA4-XML-105`, float-memory-atomic numeric policy remains
  `RDNA4-XML-145`, and the new VBUFFER special-atomic update gap is
  `RDNA4-XML-156`.
- Chapter 16.17 VIMAGE inventory narrow match: after normalizing markdown
  conversion artifacts, all 33 manual definitions from
  `rdna4/README.md:28332` through `:28859` match the checked XML
  `ENC_VIMAGE` records from `amdgpu_isa_rdna4.xml:56381` through `:57907` by
  name and opcode.
- Chapter 16.17 VIMAGE semantic boundary: XML carries the shared `ENC_VIMAGE`
  raw field layout plus visible operand footprints for image loads, stores,
  atomics, `IMAGE_GET_RESINFO`, BVH ray-tracing forms, and packed F16/BF16
  atomics. The missing dynamic image semantics remain the earlier Chapter 10
  entries `RDNA4-XML-094` through `RDNA4-XML-105`; float-memory-atomic numeric
  policy remains `RDNA4-XML-145`.
- Chapter 16.18 VSAMPLE inventory narrow match: after normalizing markdown
  conversion artifacts, all 58 manual definitions from
  `rdna4/README.md:28859` through `:29103` match the checked XML
  `ENC_VSAMPLE` records from `amdgpu_isa_rdna4.xml:170214` through `:172887`
  by name and opcode.
- Chapter 16.18 VSAMPLE semantic boundary: XML carries the shared
  `ENC_VSAMPLE` raw field layout plus visible operand footprints for
  `IMAGE_MSAA_LOAD`, ordinary sample suffix families, G16 derivative forms,
  gather variants, `IMAGE_GATHER4H`, and `IMAGE_GET_LOD`. The missing dynamic
  sample/gather semantics remain the earlier Chapter 10 entries
  `RDNA4-XML-094` through `RDNA4-XML-105`.
- Chapter 16.19 VEXPORT singleton narrow match: the manual's VEXPORT prose
  defines a single export operation at `rdna4/README.md:29103` through
  `:29111`, matching the checked XML `EXPORT` / alias `EXP` entry with
  `ENC_VEXPORT` opcode 0 at `amdgpu_isa_rdna4.xml:50025` through `:50090`.
- Chapter 16.19 VEXPORT semantic boundary: XML carries the raw
  `ENC_VEXPORT` field layout and visible `TGT`/`VSRC0..3` operands plus
  implicit `EXEC` and M0 operands. The missing shader-stage contract,
  16-bit packing, `DONE`/`ROW_EN`, status, and `EXPcnt` behavior remain the
  earlier Chapter 14 entries `RDNA4-XML-044` and `RDNA4-XML-146`.
- Chapter 16.20 VFLAT/VSCRATCH/VGLOBAL inventory narrow match: after
  normalizing markdown escaped underscores, all 144 manual definitions from
  `rdna4/README.md:29121` through `:30765` match the checked XML
  `ENC_VFLAT`, `ENC_VSCRATCH`, and `ENC_VGLOBAL` records by name and opcode:
  55 VFLAT records from `amdgpu_isa_rdna4.xml:50083` through `:52804`, 24
  VSCRATCH records from `:172926` through `:174078`, and 65 VGLOBAL records
  from `:52856` through `:56315`.
- Chapter 16.20 semantic boundary: XML carries the shared VFLAT/VSCRATCH/VGLOBAL
  raw field layouts and visible operand footprints for ordinary loads/stores,
  D16/D16_HI variants, scalar-address ADDTID globals, cache ops, block
  transfers, transpose loads, integer atomics, FP atomics, and packed F16/BF16
  atomics. The missing address-mode, aperture, scratch, error, D16, block,
  transpose, return-TH, and float-memory-atomic details remain the earlier
  Chapter 11/13 entries `RDNA4-XML-106` through `RDNA4-XML-123` and
  `RDNA4-XML-145`; the new structured special-atomic update gap is
  `RDNA4-XML-157`.
- Chapter 7.4 narrow match: XML carries the generic VOP3 and VINTERP OPSEL
  descriptions for 16-bit source high/low selection and destination half
  preservation; the gap above is the compact VOP1/VOP2/VOPC true16 namespace
  and the manual's addressability limits.
- Chapter 7.12 narrow match: XML instruction inventory matches the manual's
  WMMA/SWMMAC table for the audited names, and the basic operand roles are
  present. Dense WMMA `SRC2` is `OPR_SRC_VGPR_OR_INLINE`, while SWMMAC marks
  `VDST` as both input and output and uses `FMT_WMMA_INDEX_SET` for `SRC2`.
- Chapter 7.12 narrow match: SWMMAC `VDST` input/output marking in XML matches
  Table 42's C-and-D accumulator role. The assembler/disassembler surface still
  prints that operand once; rocjitsu's duplicate text form is tracked as
  `RDNA4-RJ-061`, not as an XML operand-role absence.
- Chapter 7.12 narrow match: `FMT_WMMA_INDEX_SET` records the 16 two-bit sparse
  index entries and the `idx0 < idx1` pair constraint; the missing piece is the
  wave32/wave64 OPSEL-based index-set selection rule.
- Chapter 12.1.2 narrow match: indexed LDS load/store/atomic instruction
  structure is represented by `ENC_VDS`, whose fields include `ADDR`, `DATA0`,
  `DATA1`, and `VDST` for local-data-share operations at
  `amdgpu_isa_rdna4.xml:3432` through `:3465`; concrete DS entries also carry
  `OPR_DSMEM` implicit memory operands and `DATA_SHARE` subgroup labels.
- Chapter 12.1.2 narrow match: direct and parameter LDS load instruction
  entries are present as `DS_DIRECT_LOAD` / `LDS_DIRECT_LOAD` and
  `DS_PARAM_LOAD` / `LDS_PARAM_LOAD`, with `VDSDIR` encoding and `M0`/LDS
  operand metadata at `amdgpu_isa_rdna4.xml:49933` through `:50014`. Their
  detailed parameter-load CU-only, EXPcnt, M0 format, and lane-fill gaps are
  recorded in `RDNA4-XML-126` through `RDNA4-XML-130`; the DS_DIRECT-specific
  details are recorded in `RDNA4-XML-131` through `RDNA4-XML-133`.
- Chapter 12.2.1 narrow match: XML carries the raw VDSDIR fields for `OP`,
  `VDST`, `ATTR`, `ATTR_CHAN`, `WAIT_VA_VDST`, and `WAIT_VM_VSRC` at
  `amdgpu_isa_rdna4.xml:3217` through `:3280`.
- Chapter 12.2.1 narrow match: XML enumerates `attr0` through `attr32` in
  `OPR_ATTR`, matching the manual's attribute-number range, at
  `amdgpu_isa_rdna4.xml:181342` through `:181506`.
- Chapter 12.3 narrow match: XML carries the `ENC_VINTERP` field layout for
  `VDST`, `WAIT_EXP`, `OPSEL`, `CLAMP`, `OP`, `SRC0`, `SRC1`, `SRC2`, and
  `NEG`, plus the no-DPP/no-literal and parameter-forwarding prose, at
  `amdgpu_isa_rdna4.xml:3073` through `:3188`.
- Chapter 12.3 narrow match: XML enumerates all six VINTERP opcodes and
  distinguishes F32, F16/F32, and RTZ F16/F32 operand widths at
  `amdgpu_isa_rdna4.xml:57958` through `:58233`.
- Chapter 12.4 narrow match: XML has `DS_DIRECT_LOAD` with alias
  `LDS_DIRECT_LOAD`, VDSDIR opcode 1, `VDST`, implicit `OPR_DSMEM`, and
  implicit M0 operands at `amdgpu_isa_rdna4.xml:49982` through `:50014`.
- Chapter 12.4 narrow match: XML carries the direct-load address/datatype
  values in the instruction description, including `UBYTE`, `USHORT`, `DWORD`,
  `SBYTE`, and `SSHORT`, at `amdgpu_isa_rdna4.xml:49986`; the missing
  structured datatype/extension metadata is recorded in `RDNA4-XML-133`.
- Chapter 12.5 narrow match: XML carries the raw `ENC_VDS` field layout for
  `ADDR`, `DATA0`, `DATA1`, `OFFSET0`, `OFFSET1`, `OP`, and `VDST` at
  `amdgpu_isa_rdna4.xml:3432` through `:3522`; address-formula and offset-use
  gaps are recorded in `RDNA4-XML-134`.
- Chapter 12.5 narrow match: XML enumerates representative indexed load/store
  families, including one-address, 2-address, stride64, sign/zero-extending
  byte/short loads, D16 low/high variants, and ADDTID forms, for example at
  `amdgpu_isa_rdna4.xml:46100`, `:46140`, `:46184`, `:46232`, `:48933`, and
  `:49401`.
- Chapter 12.5.1 narrow match: XML enumerates the atomic return/no-return
  inventory for integer, floating, packed F16/BF16, compare/store, and
  store-exchange families, for example `DS_ADD_U32` at
  `amdgpu_isa_rdna4.xml:43966`, `DS_ADD_RTN_U32` at `:45027`,
  F64 atomics at `:47290` and `:48311`, and packed F16/BF16 atomics at
  `:48749` and `:48795`.
- Chapter 12.5.2 narrow match: XML opcodes match the manual for
  `DS_SWIZZLE_B32` opcode 53, `DS_PERMUTE_B32` opcode 178,
  `DS_BPERMUTE_B32` opcode 179, and `DS_BPERMUTE_FI_B32` opcode 205 at
  `rdna4/README.md:7812`, `:7817`, and `:7828`, and
  `amdgpu_isa_rdna4.xml:46066`, `:49493`, `:49535`, and `:49577`.
- Chapter 12.5.2 narrow match: XML operand shape matches the manual's
  `VDST`, `ADDR`, and `DATA0` roles for the three permute opcodes at
  `rdna4/README.md:6029` and `amdgpu_isa_rdna4.xml:49495` through `:49512`,
  `:49537` through `:49554`, and `:49579` through `:49596`.
- Chapter 12.5.2 narrow match: XML preserves the no-LDS-storage description for
  the permute family and swizzle summary at `amdgpu_isa_rdna4.xml:46061`,
  `:49488`, `:49530`, and `:49572`, matching `rdna4/README.md:6021` and
  `:5947`; the missing lane/mode semantics are recorded in `RDNA4-XML-139`
  and `RDNA4-XML-140`.
- Chapter 12.5.3 narrow match: XML opcode inventory matches the manual for
  `DS_BVH_STACK_PUSH4_POP1_RTN_B32`, `DS_BVH_STACK_PUSH8_POP1_RTN_B32`, and
  `DS_BVH_STACK_PUSH8_POP2_RTN_B64`; the manual lists opcodes 224, 225, and
  226 at `rdna4/README.md:27036`, `:27094`, and `:27152`, and XML records the
  same opcode values at `amdgpu_isa_rdna4.xml:49707`, `:49755`, and `:49803`.
- Chapter 12.5.3 narrow match: XML operand widths match the manual's stack-op
  footprint: push4 uses `DATA1` B128 at `amdgpu_isa_rdna4.xml:49727` through
  `:49732`, the push8 forms use `DATA1` B256 at `:49775` through `:49780` and
  `:49823` through `:49828`, and pop2 returns B64 through `VDST` at `:49805`
  through `:49810`.
- Chapter 12.5.3 narrow match: XML does mark the stack `ADDR` operand as
  input/output for all three RDNA4 stack opcodes at
  `amdgpu_isa_rdna4.xml:49715`, `:49763`, and `:49811`; the packed layout and
  update semantics are the missing pieces recorded in `RDNA4-XML-142`.
- RDNA3/RDNA3.5 boundary note: older RDNA families expose a single
  `DS_BVH_STACK_RTN_B32` opcode rather than RDNA4's three push/pop forms;
  local XML has those entries in `amdgpu_isa_rdna3.xml:24285` and
  `amdgpu_isa_rdna3_5.xml:21850`, and the RDNA3 manual defines its stack size
  through `OFFSET1[5:4]` at `rdna3/README.md:23534`. That older contract needs
  a separate cross-family slice.

## Cross-Architecture Notes Found During This Slice

During the RDNA4 Chapter 5.6 pass, `amdgpu_isa_gfx1250.xml` was useful as a
contrast case: it contains `S_GET_BARRIER_STATE` at
`shared/machine-readable-isa/isa/amdgpu_isa_gfx1250.xml:23277`, but its
description records only `KMCNT` timing and does not encode the manual's packed
`valid` / `memberCnt` / `signalCnt` result layout. This should be revisited in
a dedicated gfx1250/MI450 slice.

The same `FP16_OVFL` prose table appears in CDNA3 and CDNA4 manuals:

- `cdna3/README.md:2034` through `:2048`
- `cdna4/README.md:2402` through `:2418`

CDNA3/CDNA4 add a source-legality statement that is not represented by their
XML operand classes: the manual says `CVT_SR_*` and `CVT_PK_*` support only
VGPR inputs, not SGPRs, literal, or inline constants. CDNA3 XML uses
`OPR_SRC_NOLIT` / `OPR_SRC_SIMPLE` at `amdgpu_isa_cdna3.xml:64566` through
`:64576` and similar entries; CDNA4 XML uses the same classes at
`amdgpu_isa_cdna4.xml:72378` through `:72388` and similar entries. CDNA4 XML
defines those classes as including SGPR/special/inline subtypes at
`amdgpu_isa_cdna4.xml:124697` through `:124709` and `:127127` through
`:127138`.

LLVM assembler sanity check: `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx942`
accepted `v_cvt_pk_fp8_f32 v0, s0, v2` and `v_cvt_pk_fp8_f32 v0, 1.0, v2`,
so assembler behavior appears to follow the XML-like broad source classes
rather than the CDNA3 prose restriction. Hardware behavior remains unaudited.
