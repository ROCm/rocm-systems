# CDNA3 Manual vs XML Gaps

Architecture: CDNA3 / MI300

Manual source: `workspace_docs/amdgpu-isa-manuals/cdna3/README.md`

XML source: `shared/machine-readable-isa/isa/amdgpu_isa_cdna3.xml`

## Coverage

| Manual section | Status | Notes |
| --- | --- | --- |
| 1.1 terminology and 2.1 through 2.3 Program Organization | Audited statically | Checked host/command-processor/memory-controller overview, dispatch/workgroup/wavefront/work-item terms, 64-lane wavefront model, SALU/VALU roles, EXEC-masked vector execution, dispatch grid/work-item indexing, LDS/GWS topology overlap, and Section 2.3 device-memory consistency/acknowledgment prose against XML global metadata and instruction descriptions. |
| 3.1 through 3.6 Kernel State | Audited statically | Checked state inventory including `FLAT_SCRATCH`, `XNACK_MASK`, and wait counters, PC/EXEC/STATUS/MODE prose, status and mode bitfields, GPR/LDS allocation and aliasing rules, out-of-range behavior, and LDS allocation/clamping. |
| 3.7 through 3.13 Kernel State | Audited statically | Checked M0 descriptor roles, SCC summary, VCC/VCCZ update and alias-hazard prose, trap/exception/TRAPSTS/memory-violation state, HW_ID/XCC_ID bitfields, and compute GPR initialization. |
| 4.1 through 4.2 Program Flow Control | Audited statically | Checked control/branch instruction inventories, direct PC operations, trap return, debug conditional branches, status/message/sleep/wakeup controls, and perf/thread-trace rows against XML opcode flags, operands, and descriptions. |
| 4.3 Workgroups | Audited statically | Checked workgroup wavefront limits, `S_BARRIER` release behavior, early-terminated wave handling, trap-handler legality, wait-counter interaction, and `STATUS.IN_BARRIER` against XML opcode metadata. |
| 4.4 Data Dependency Resolution | Audited statically | Checked `S_WAITCNT` counter threshold layout, VM/LGKM/EXP producer and decrement prose, scalar-memory dword-count behavior, `S_SENDMSG`, FLAT dual-counter behavior, out-of-order restrictions, and GWS `EXP_CNT` semantics against XML opcode and operand metadata. |
| 4.5 Manually Inserted Wait States (NOPs) | Audited statically | Checked the required software-inserted wait-state table, affected register and instruction families, trans-op inventory, `S_NOP` delay semantics, and adjacent `S_ICACHE_INV` spacing prose against XML opcode, operand, and description metadata. |
| 4.6 Arbitrary Divergent Control Flow | Audited statically | Checked `S_CBRANCH_{G,I}_FORK` and `S_CBRANCH_JOIN`, compiler-emitted fork/join block structure, CSP/branch-stack requirements, pass/fail mask selection, EXEC/PC update pseudocode, and XML branch/operand metadata. |
| 5.1 through 5.2 Scalar ALU Operations | Audited statically | Checked SALU operand table, source/destination selector legality, literal availability, out-of-range source/destination behavior, and 64-bit SGPR-pair alignment prose. |
| 5.3 through 5.7 Scalar ALU Operations | Audited statically | Checked scalar condition-code prose, arithmetic carry/overflow/nonzero rules, conditional SCC consumers, comparison writers, bitwise/shift/bitfield SCC effects, and representative detailed SOP1/SOP2/SOPC/SOPK definitions. |
| 5.8 Access Instructions | Audited | Checked `S_GETREG_B32`, `S_SETREG_B32`, and `S_SETREG_IMM32_B32` HWREG selector layout, hardware-register inventory, allocation/status field packing, SETREG write-mask/side-effect prose, and representative SOPK definitions. |
| 6.1 through 6.6 Vector ALU Operations | Audited statically | Checked VOP1/VOP2/VOP3/VOPC encoding overview, VALU source-selector table, scalar/literal/M0 restrictions, literal expansion, instruction inventory, VOP3 input/output modifier rules, EXEC-masked writes, carry/compare destinations, out-of-range VGPR behavior, FP denormalized and rounding modes, ALU clamp bit usage, and VGPR indexing. |
| 6.7 Packed Math | Audited statically | Checked packed opcode inventory, MIX selector/ABS overloads, packed 32-bit OPSEL behavior, and alignment/scalar-pair prose. |
| 7.1 through 7.1.5 Matrix fused-multiply-add (MFMA) | Audited statically | Checked dense MFMA operation shape, ACC/ACC_CD register-bank prose, contiguous/aligned register-block requirements, state-control rows, XF32 reduced-precision wording, input/output layout formulas, and ordinary/F64 broadcast-field semantics. |
| 7.2 BF8 and FP8 Formats and Conversions | Audited statically | Checked numeric-format table, conversion table, SDWA selection, stochastic OPSEL, `FP16_OVFL`, source legality, and `SH_MEM_CONFIG[8]`. |
| 7.3 Floating-point handling details and formats | Audited statically | Checked per-family MFMA denorm/MODE behavior, F64 rounding/denorm behavior, I8 MODE independence, and DGEMM exception support. |
| 7.4 Sparse Matrices | Audited statically | Checked SMFMAC inventory, 4:2 sparse-A/index-VGPR prose, C/D accumulate behavior, `SRC2`/`ACC_CD` rules, `CBSZ`/`ABID` sparse-index selection, index layouts, and index/alignment restrictions. |
| 7.5 Dependency Resolution: Required Independent Instructions | Audited statically | Checked DLop/XDLOP/DGEMM/pass definitions and Table 37 required-wait classes against XML metadata. |
| 8.1 through 8.4 Scalar Memory Operations | Audited statically | Checked SMEM encoding fields, scalar/global/scratch/buffer addressing, source-overwrite and clause rules, atomics, cache ops, timer ops, dependency-counter behavior, and alignment/bounds rules. |
| 9.1 Vector Memory Buffer Instructions | Audited statically | Checked VMEM buffer instruction taxonomy, MUBUF/MTBUF field behavior, VGPR address/data use, buffer data-format selection, D16, address/range/swizzle/resource semantics, buffer-to-LDS, SC/NT cache controls, and buffer data-format tables. |
| 9.2 Float Memory Atomics | Audited statically | Checked float atomic issuer coverage, RNE add rounding, LDS/L2 denormal rules, NaN quieting and selection, signed-zero ordering, compare-store equality, and float-add edge cases against XML DS, MUBUF, FLAT, and GLOBAL float-atomic entries. |
| 10.1 through 10.7 Flat Memory Instructions | Audited statically | Checked FLAT/GLOBAL/SCRATCH field behavior, address forms, aperture and scratch-base prose, SVE/LDS bit descriptions, `NV`/`ACC` field drift, direct LDS movement and operand metadata, data movement, atomics, ordering, timing, and VM/LGKM counter rules. |
| 11.1 through 11.4 Data Share Operations | Audited | Checked LDS overview/dataflow, bank/conflict and M0 clamp prose, indexed load/store/atomic address rules, two-offset rules, permute/swizzle behavior, ADDTID, packed LDS atomics, and GWS programming restrictions. |
| SOP1 instruction definitions | Audited statically | Checked full Chapter 12.3 SOP1 opcode inventory, literal variants, direct PC/RFE rows, saveexec/wrexec formulas, bit-count/bit-scan/ABS edge examples, bitset old-destination rows, quad-mask/bitreplicate rows, M0-relative scalar moves, fork/join overlap, and GPR-index control row. |
| SOP2 instruction definitions | Audited statically | Checked full Chapter 12.1 SOP2 opcode definitions 0-52, detailed arithmetic/carry/bitfield/absdiff/shifted-add/pack formulas, literal variants, implicit SCC/PC operands, and opcode 43 XML/manual drift. |
| SOPK instruction definitions | Audited statically | Checked full Chapter 12.2 SOPK opcode inventory, opcode 19 hole, literal-only opcode 20 form, SIMM16 sign/zero extension rules, old-destination ADDK/MULK dataflow, fork/call PC formulas, and HWREG access rows. |
| SOPC instruction definitions | Audited statically | Checked full Chapter 12.4 SOPC opcode inventory, literal variants, signed/unsigned compare formulas, bit-compare index masks, VSKIP state row, `S_SET_GPR_IDX_ON` raw `SIMM4` semantics, and implicit M0/SCC metadata. |
| SOPP instruction definitions | Audited statically | Checked full Chapter 12.5 SOPP opcode inventory, ordinary/debug branch formulas, wait-count field layout, barrier/status/sleep/priority/message/trap/cache/perf rows, GPR-index control rows, send-message payload table, and opcode 30/31 XML/manual drift. |
| 12.10 VOP3P Instructions | Audited statically | Checked the full Chapter 12.10 opcode inventory across `ENC_VOP3P` and `VOP3P_MFMA`, including packed 16-bit arithmetic, MIX, DOT, packed F32 arithmetic, `V_PK_MOV_B32`, `V_ACCVGPR_READ`/`WRITE`, dense MFMA, and sparse SMFMAC rows. |
| 12.11 VOP3A & VOP3B Instructions | Audited statically | Checked the full Chapter 12.11 opcode inventory across `ENC_VOP3` and `VOP3_SDST_ENC`, VOP3B scalar-destination opcode set, VOP1 promoted-opcode drift, generic VOP3 literal conflict, readfirstlane/minmax/accumulator/OPSEL/packed-conversion prose-only overlaps, `V_LSHL_ADD_U64` shift-count behavior, and VOP3A/VOP3B field-table metadata. |
| SMEM instruction definitions | Audited statically | Checked full Chapter 12.6 SMEM opcode inventory, scalar/global/scratch/buffer load-store rows, cache maintenance, time, discard, scalar and buffer atomic pseudocode, and XML-only ATC probe drift. |
| VOP2 instruction definitions | Audited statically | Checked full Chapter 12.7 opcode inventory, opcode 22 hole, ordinary VOP3 promotion, literal-only exceptions, literal/DPP/SDWA reachability, FP min/max edge rules, 24-bit multiply rows, carry/borrow forms, F16 MAC/literal destination-half behavior, 16-bit shift/ldexp source sizing, dot accumulators, packed FMAC, and bitwise/no-modifier notes. |
| 12.16 Instruction Limitations | Audited statically | Checked DPP and SDWA prohibited-instruction lists against XML extension-row availability, F32 `MADMK`/`MADAK` name normalization against `FMAMK`/`FMAAK`, legal VOPC DPP availability, and 64-bit DPP control restrictions. |
| MUBUF/MTBUF instruction definitions | Audited statically | Checked the full Chapter 12.13 and 12.14 opcode inventories: 73 MUBUF and 16 MTBUF rows, including formatted, raw, LDS-capable, cache-maintenance, and atomic buffer entries against XML operands and descriptions. |
| FLAT/GLOBAL/SCRATCH instruction definitions | Audited statically | Checked the full Chapter 12.15 opcode inventory and Chapter 13.6 opcode tables for FLAT, GLOBAL, and SCRATCH, plus D16 preservation, LDS transfer rows, atomic return wording, integer RMW formulas, F64 atomic description drift, offset widths, `SADDR`, `SC`, `NT`, `ACC`, and bit 13 naming. |
| DS instruction definitions | Audited statically | Checked full Chapter 12.12 LDS/GWS opcode inventory, DS loads/stores, READ2/WRITE2/ST64, narrow and D16 reads/writes, integer/floating/packed atomics, ADDTID, APPEND/CONSUME, WRAP, CONDXCHG32, permute/swizzle, B96/B128, and GWS semaphore/barrier definitions. |
| SOP* instruction definitions | Audited statically | Checked representative SOP1/SOP2/SOPK/SOPC/SOPP entries for operand metadata, literal-extension alternatives, implicit SCC operands, SCC predicate semantics for arithmetic/compare/bitwise operations, HWREG access operands, `S_WAITCNT` operand metadata, and `S_NOP` delay metadata. |
| Dense MFMA instruction definitions | Audited statically | Checked representative F32, F16, BF16, I8, XF32, F64, FP8, and BF8 dense MFMA definitions for operand shapes and field coverage. |
| Sparse MFMA instruction definitions | Audited statically | Checked representative F16, BF16, I8, BF8, and FP8 SMFMAC definitions against XML operand metadata and VOP3P-MAI field rules. |
| 13.1 Scalar ALU and Control Formats | Audited statically | Checked SOP1/SOP2/SOPK/SOPC/SOPP field maps and opcode tables across sections 13.1.1 through 13.1.5, including scalar source-selector tables, literal-extension variants, `S_SETREG_IMM32_B32`, branch/fork/wait/message/GPR-index rows, and recorded SOP2/SOPP opcode drift. |
| 13.3.1 through 13.3.3 VOP2/VOP1/VOPC | Audited statically | Checked VOP2, VOP1, and VOPC field maps and opcode tables, promoted VOP3 opcode tables, literal/DPP/SDWA/SDWAB extension availability, VOPC compare destination metadata, VOP2 literal-only exceptions, VOP1 opcode 55 and promoted-opcode drift, F64 VOP1 DPP leakage, and VOPC DPP availability. |
| 13.3.4 through 13.3.5 VOP3A/VOP3B | Audited statically | Checked VOP3A and VOP3B field maps, source-selector tables, promoted-opcode tables, VOP3B `SDST` description, and scalar-destination opcode table coverage for the Chapter 12.11 slice. |
| 13.3.7 through 13.3.9 SDWA/SDWAB/DPP | Audited statically | Checked SDWA/SDWAB field maps, generic DPP control ranges, VOP1/VOP2/VOPC extension availability, and the 64-bit-input `DPP_ROW*` restriction. |
| 13.2 Scalar Memory Format | Audited statically | Checked SMEM field inventory, full opcode table, signed immediate offset description, `SOFFSET_EN`, `GLC`, `NV`, `SBASE`, `SDATA`, and opcode 38/39 XML/manual drift. |
| 13.4.1 DS Format | Audited statically | Checked DS field map, `ACC`, `GDS`, offset fields, opcode width, operand fields, opcode table coverage, and GWS opcode rows against Chapter 11 prose and the full Chapter 12.12 instruction inventory. |
| 13.5 Vector Memory Buffer Formats | Audited statically | Checked MUBUF/MTBUF field maps, typed/untyped split, DFMT/NFMT field placement, SRSRC scaling prose, LDS bit, SC/NT fields, and opcode tables. |
| 13.6 Flat Formats | Audited statically | Checked FLAT field map, segment encodings, offset widths, `SADDR`, `SC`, `NT`, `ACC`, bit 13 naming, and opcode tables. |
| 12.8 VOP1 Instructions | Audited statically | Checked full VOP1 opcode inventory, promoted VOP3 aliases, XML-only opcode 55, `V_READFIRSTLANE_B32` lane/no-modifier contract, move modifiers, conversion/fract/transcendental/FREXP/CLREXCP/SAT semantics, `V_SWAP_B32` and `V_ACCVGPR_MOV_B32` dataflow, FP8/BF8 overlap, and Chapter 12.16 DPP/SDWA limitations. |
| 12.9 VOPC Instructions | Audited statically | Checked full VOPC opcode inventory, compare predicate/range tables, class-mask pseudocode, VOP3A aliases and `CLAMP` behavior, VCC/EXEC/SGPR destination rules, literal/SDWA/DPP extension metadata, and Chapter 12.16 DPP/SDWA limitations. |
| 13.3.6 through 13.3.6.1 VOP3P/VOP3P-MAI | Audited statically | Checked generic VOP3P and VOP3P-MAI field positions and opcode-table coverage for packed 16-bit arithmetic, MIX, DOT, packed F32, `V_PK_MOV_B32`, dense MFMA, and sparse SMFMAC rows, including packed width/count prose, dword selector special cases, source alignment/scalar-pair rules, and dense/sparse overlap. |
| CDNA3 manual sections through Chapter 13 | Complete | Chapter-level status is tracked in `audit-scope.md`; CDNA3 Chapters 1 through 13 now have section-level manual/XML coverage in this report. |

## Gaps

### CDNA3-XML-001: FNUZ FP8/BF8 numeric semantics are prose-only

Manual evidence:

- `cdna3/README.md:1985` through `:1989` defines CDNA3 FP8/BF8 numeric
  encodings, including bias, zero, Inf/NaN encodings, max value, min normal,
  and min denormal.

XML evidence:

- `FMT_NUM_BF8` at `amdgpu_isa_cdna3.xml:91589` through `:91625` records only
  an 8-bit float layout with sign, exponent, and mantissa fields.
- `FMT_NUM_FP8` at `amdgpu_isa_cdna3.xml:91741` through `:91775` records only
  the FP8 bit layout.
- Searching the CDNA3 XML for `FNUZ` or `fnuz` returns no matches.

Impact:

The XML identifies storage shape but not the CDNA3-specific FNUZ numeric
interpretation needed by conversion helpers, decoder tests, or prose/XML
comparisons.

### CDNA3-XML-002: `FP16_OVFL` and `SH_MEM_CONFIG[8]` requirements are absent

Manual evidence:

- `cdna3/README.md:2034` through `:2046` defines the F32 to FP8/BF8
  `FP16_OVFL` result table.
- `cdna3/README.md:2048` says `SH_MEM_CONFIG` bit 8 must be set for correct
  BF8/FP8 operation results.

XML evidence:

- The four narrow conversion entries are present at
  `amdgpu_isa_cdna3.xml:64552` through `:64692`, but they only carry operand
  and high-level description data.
- Searching the CDNA3 XML for `FP16_OVFL` or `SH_MEM_CONFIG` returns no
  matches.

Impact:

An XML consumer cannot discover that FP8/BF8 conversion results depend on MODE
state and a scalar memory configuration bit.

### CDNA3-XML-003: `CVT_SR_*` / `CVT_PK_*` source legality contradicts the manual

Manual evidence:

- `cdna3/README.md:2038` says `CVT_SR_*` and `CVT_PK_*` support only VGPR
  inputs, not SGPRs, literal constants, or inline constants.

XML evidence:

- `V_CVT_PK_FP8_F32` uses `OPR_SRC_NOLIT` for `SRC0` and `OPR_SRC_SIMPLE` for
  `SRC1` at `amdgpu_isa_cdna3.xml:64566` through `:64576`.
- `V_CVT_SR_FP8_F32` uses the same broad source classes at
  `amdgpu_isa_cdna3.xml:64648` through `:64658`; BF8 follows the same pattern
  beginning at `:64675`.
- `OPR_SRC_NOLIT` includes scalar-register subtypes at
  `amdgpu_isa_cdna3.xml:101454` through `:101465`, and `OPR_SRC_SIMPLE`
  includes scalar-register subtypes at `:103880` through `:103890`.

Impact:

The XML advertises legal encodings that the prose manual says are illegal. LLVM
assembler behavior and hardware behavior still need a focused oracle/probe
before deciding whether the manual or XML should drive rocjitsu here.

### CDNA3-XML-004: SDWA byte/word legality for FP8/BF8 widening converts is not encoded

Manual evidence:

- `V_CVT_F32_FP8` uses BYTE0-3 SDWA selection, defaults to BYTE0 without SDWA,
  and allows only BYTE selects at `cdna3/README.md:8766` through `:8789`.
- `V_CVT_F32_BF8` repeats the BYTE-only rule at `cdna3/README.md:8790`
  through `:8809`.
- `V_CVT_PK_F32_FP8` and `V_CVT_PK_F32_BF8` use WORD0/WORD1 selection and
  allow only WORD selects at `cdna3/README.md:8811` through `:8839`.

XML evidence:

- The SDWA encoding exposes a generic 3-bit `SRC0_SEL` field at
  `amdgpu_isa_cdna3.xml:5735` through `:5741`.
- The affected SDWA instruction entries use 8-bit or 16-bit operands, for
  example `amdgpu_isa_cdna3.xml:51077` through `:51091` and `:51299` through
  `:51314`, but do not encode the BYTE-only, WORD-only, or non-SDWA default
  selection rules.

Impact:

Generated decoders can see that SDWA exists, but cannot derive which
`SRC0_SEL` values are legal or what implicit sub-dword slice applies without
manual knowledge.

### CDNA3-XML-005: Packed FP8/BF8 data-format specificity is missing from the checked-in XML

Manual evidence:

- `V_CVT_PK_F32_FP8` and `V_CVT_PK_F32_BF8` are described as packed
  two-component FP8/BF8 sources at `cdna3/README.md:8811` through `:8839`.
- `V_CVT_PK_FP8_F32` and `V_CVT_PK_BF8_F32` write packed two-byte results at
  `cdna3/README.md:15916` through `:15956`.

Checked-in XML evidence:

- The checked-in CDNA3 XML has no `FMT_NUM_PK2_FP8` or `FMT_NUM_PK2_BF8`
  data-format names.
- Packed widening entries use scalar `FMT_NUM_FP8` or `FMT_NUM_BF8` with
  `OperandSize` 16 at `amdgpu_isa_cdna3.xml:51252` through `:51256` and
  `:51363` through `:51367`.
- Packed narrowing entries also use scalar `FMT_NUM_FP8` or `FMT_NUM_BF8` with
  16-bit destinations at `amdgpu_isa_cdna3.xml:64560` through `:64564` and
  `:64601` through `:64605`.
- The newer workspace reference XML at
  `/home/aulu/Workspace/workspace_docs/amdgpu_isa_cdna3.xml` does use
  `FMT_NUM_PK2_FP8` and `FMT_NUM_PK2_BF8` for these packed rows; this gap is
  specific to the checked-in XML source used by rocjitsu.

Impact:

The packed-two-component semantic is recoverable only by instruction name,
operand size, and manual prose, not by a distinct XML data format.

### CDNA3-XML-006: FP8/BF8 forwarding hazard is prose-only

Manual evidence:

- `cdna3/README.md:2009` says the `CVT_*_F32` instructions in this table do not
  support 4-cycle forwarding and require a NOP or unrelated destination write
  between conversions writing the low/high half or bytes of the same
  destination register.

XML evidence:

- The audited FP8/BF8 conversion instruction entries do not carry a scheduling,
  forwarding, or producer-hazard annotation.

Impact:

Instruction scheduling and hazard-aware fuzzing cannot derive this rule from
XML alone.

### CDNA3-XML-007: Packed 32-bit VOP3P dword selection and `V_PK_MOV_B32` special rules are prose-only

Manual evidence:

- `cdna3/README.md:1527` says packed 32-bit instructions operate on two dwords,
  require two-dword alignment, do not support output modifiers, and use
  `OPSEL`/`OPSEL_HI` to select the first or second source dword.
- `V_PK_FMA_F32`, `V_PK_MUL_F32`, and `V_PK_ADD_F32` read source dwords
  `[31:0]` and `[63:32]` at `cdna3/README.md:11495` through `:11529`.
- `V_PK_MOV_B32` uses only `OPSEL[0]` and `OPSEL[1]` for the two output dwords,
  treats sources as 64-bit operands, applies SGPR/VGPR alignment restrictions,
  supports VGPR gather from the even register or next odd register, and restricts
  two-SGPR forms to the same SGPR at `cdna3/README.md:11532` through `:11558`.

XML evidence:

- Generic `ENC_VOP3P` describes two 16-bit operations at
  `amdgpu_isa_cdna3.xml:1986` and describes `OP_SEL`, `OP_SEL_HI`, and
  `OP_SEL_HI_2` only as 16-bit half selectors at `:2040` through `:2061`.
- The packed F32 and `V_PK_MOV_B32` entries do use 64-bit operands at
  `amdgpu_isa_cdna3.xml:65983` through `:66135`, and the shared operand
  predefined values carry generic 64-bit SGPR/VGPR even-alignment text at
  `amdgpu_isa_cdna3.xml:103893` and `:105022`.
- The XML entries do not encode the packed 32-bit dword selector override,
  `V_PK_MOV_B32`'s special use of `OPSEL` without `OPSEL_HI`, its VGPR gather
  semantics, its unsupported-output-modifier statement, or the same-SGPR
  restriction for two scalar sources.

Impact:

XML-only consumers can discover that these operands are 64-bit and generically
even-aligned, but cannot derive the packed 32-bit selector contract or the
`V_PK_MOV_B32`-specific scalar/gather restrictions without manual prose.

### CDNA3-XML-008: MIX-specific VOP3P selector and modifier overloads are prose-only

Manual evidence:

- Section 6.7 says `V_MAD_MIX_*` are not packed math but use the VOP3P encoding
  at `cdna3/README.md:1525`.
- Each MIX instruction says `{OPSEL_HI[i], OPSEL[i]}` chooses between a full
  32-bit F32 input and the low/high F16 half, and says `NEG_HI` acts as an
  absolute-value modifier at `cdna3/README.md:11329`, `:11354`, and `:11377`.

XML evidence:

- Generic `ENC_VOP3P` describes `NEG_HI` as high-half floating negation at
  `amdgpu_isa_cdna3.xml:2020` and describes the selector fields as generic
  16-bit half selectors at `:2040` through `:2061`.
- `V_MAD_MIX_F32`, `V_MAD_MIXLO_F16`, and `V_MAD_MIXHI_F16` are present at
  `amdgpu_isa_cdna3.xml:65513`, `:65560`, and `:65607`, but their entries do
  not encode the MIX-only selector mapping or `NEG_HI` absolute-value behavior.

Impact:

Generated code needs a hard-coded MIX special case or manual-derived metadata;
the XML field descriptions alone imply the wrong `NEG_HI` behavior for these
opcodes.

### CDNA3-XML-009: Dense MFMA layout, state, and register-block rules are prose-only

Manual evidence:

- Section 7.1 defines MFMA as one or more outer-product matrix
  multiplications, records the `D = C + A * B` block formula, and explains
  `ACC` versus `ACC_CD` register-bank selection at `cdna3/README.md:1547`
  through `:1560`.
- The same section requires MFMA source and destination register blocks to be
  contiguous and aligned to the full register-block width at
  `cdna3/README.md:1562`.
- Table 28 lists dense MFMA variants, block counts, cycle counts, state-control
  behavior, `FP16_OVFL` clamp behavior, forced RNE, exception behavior,
  execution-mask behavior, and source restrictions at `cdna3/README.md:1595`
  through `:1638`.
- Section 7.3 later refines the denorm story by saying ordinary
  `V_MFMA_F32_*_F32` honors MODE denormal flags, while XF32 and matrix C/D do
  not flush denormals, at `cdna3/README.md:2050` through `:2059`. This appears
  to narrow or conflict with the broader dense-MFMA denorm row in 7.1.2.
- Sections 7.1.3 and 7.1.4 give concrete and formula-based input/output
  register/lane layouts, including packed 8/16/32-bit items and special F64
  output layout, at `cdna3/README.md:1646` through `:1908`.

XML evidence:

- The `VOP3P_MFMA` encoding exposes `ABID`, `ACC`, `ACC_CD`, `BLGP`, `CBSZ`,
  source, and destination fields at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna3.xml:7106` through `:7222`.
- Representative dense MFMA entries record opcode, operand type, data format,
  and operand size, for example the XF32 entries at
  `amdgpu_isa_cdna3.xml:66153` through `:66237` and the F64 entries at
  `:67661` through `:67754`.
- The XML entries do not encode the MFMA block formula, dense matrix layout
  formulas, cycle counts, register-block alignment rule, execution-state table,
  the section 7.3 denorm refinement, or clamp/`FP16_OVFL` behavior.

Impact:

XML consumers can enumerate dense MFMA encodings and operand widths, but cannot
derive layout math, register-block legality, timing metadata, or state-dependent
behavior without manual prose.

### CDNA3-XML-010: XF32 reduced-precision MFMA behavior is prose-only

Manual evidence:

- Table 28 describes the `V_MFMA_F32_{*}_XF32` forms as F32-data matrix
  multiply with reduced multiplication precision at `cdna3/README.md:1621`
  through `:1622`.
- The XF32 subsection says these instructions take 32-bit floats but round the
  mantissa to 10 bits before reduced-precision multiplication at
  `cdna3/README.md:1640` through `:1642`.

XML evidence:

- `V_MFMA_F32_16X16X8_XF32` and `V_MFMA_F32_32X32X4_XF32` are present at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna3.xml:66153` through
  `:66237`, but all operands use `FMT_NUM_F32`.
- Searching the CDNA3 XML for `reduced`, `XF32` semantics, or the 10-bit
  mantissa rule finds only instruction names and unrelated mantissa text.

Impact:

The XML exposes XF32 opcode availability but not the numerical distinction from
ordinary F32 MFMA.

### CDNA3-XML-011: MFMA broadcast permutations and F64 field overrides are under-described

Manual evidence:

- Section 7.1.5 says `CBSZ`, `ABID`, and `BLGP` affect the lanes used to
  retrieve matrix A and B values at `cdna3/README.md:1910` through `:1916`.
- The manual defines `CBSZ`/`ABID` legality, the largest legal `CBSZ`, the
  `ABID < (1 << CBSZ)` rule, and the exact A-lane permutation formula at
  `cdna3/README.md:1918` through `:1942`.
- Table 29 defines all ordinary `BLGP` values 0 through 7 at
  `cdna3/README.md:1952` through `:1967`.
- F64 MFMA ignores `CBSZ` and `ABID` and repurposes `BLGP[0:2]` as implicit
  negation of A, B, and C at `cdna3/README.md:1969` through `:1979`.

XML evidence:

- The `VOP3P_MFMA` field map records `ABID`, `BLGP`, and `CBSZ` bit positions
  and gives only broad swizzling descriptions at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna3.xml:7175` through `:7216`.
- F64 MFMA entries such as `V_MFMA_F64_16X16X4_F64` and
  `V_MFMA_F64_4X4X4_4B_F64` record ordinary operands at
  `amdgpu_isa_cdna3.xml:67661` through `:67754`, but do not mark the F64
  `BLGP`-as-negation override.

Impact:

XML-only consumers cannot validate MFMA broadcast encodings or derive the F64
field override from the field map and opcode entries alone.

### CDNA3-XML-012: MFMA floating-point mode and DGEMM exception rules are prose-only

Manual evidence:

- The dense MFMA control table gives broad denorm, clamp, rounding, exception,
  execution-mask, and source-behavior rows at `cdna3/README.md:1628` through
  `:1638`.
- Section 7.3 refines that table by saying ordinary `V_MFMA_F32_*_F32` honors
  MODE denormal flags, XF32 ignores `MODE.denorm`, matrix C input and D output
  do not flush denormals, sub-32-bit floating MFMA ignores MODE denorms, F64
  MFMA ignores MODE and rounds to nearest even while allowing denorms, and I8
  MFMA is integer/MODE-independent at `cdna3/README.md:2052` through `:2059`.
- The same section says the matrix core does not support arithmetic exceptions
  except that DGEMM matrix operations do support exceptions at
  `cdna3/README.md:2061`.

XML evidence:

- The `VOP3P_MFMA` encoding exposes only the opcode selector and microcode
  fields such as `ABID`, `ACC`, `ACC_CD`, `BLGP`, and `CBSZ` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna3.xml:7106` through `:7222`.
- Representative MFMA entries, such as XF32 at
  `amdgpu_isa_cdna3.xml:66153` through `:66237` and F64 at `:67661` through
  `:67754`, record operand formats and sizes but not per-family MODE, denorm,
  rounding, or exception behavior.
- Searching the CDNA3 XML for `FP_DENORM`, `FP_ROUND`, `MODE`, or MFMA
  exception metadata finds no MFMA-specific entries.

Impact:

An XML consumer cannot derive the Section 7.3 distinction between ordinary F32,
XF32, sub-32-bit floating, F64, and I8 MFMA behavior. In particular, DGEMM
exception support and F64 nearest-even/denorm behavior require manual prose.

### CDNA3-XML-013: MAI dependency waits and pass classes are prose-only

Manual evidence:

- Section 7.5 defines dependency conditions that require inserted NOPs or
  independent VALU instructions, and defines `DLop`, `XDLOP`, `DGEMM`, and
  `PASS` at `cdna3/README.md:2191` through `:2203`.
- Table 37 lists required waits for VALU-to-MFMA reads, DLop forwarding,
  XDL/SMFMA overlap cases, SGEMM overlap cases, DGEMM/F64 cases, and
  `V_CMPX*` execution-mask forwarding at `cdna3/README.md:2205` through
  `:2302`.

XML evidence:

- The `VOP3P_MFMA` encoding and MFMA instruction entries enumerate opcode
  fields and operands, but do not carry pass counts, DLop/XDLOP/DGEMM classes,
  result-overlap rules, forwarding classes, or required-wait values.
- Searching the CDNA3 XML for Table 37 concepts such as `PASS`, `forwarding`,
  `DGEMM`, and `Required Waits` does not expose scheduler metadata for these
  MFMA hazards.

Impact:

Scheduling, validator, and fuzz-oracle consumers cannot discover CDNA3 MAI wait
requirements from XML alone. The required waits depend on instruction family,
pass count, operand role, exact opcode match, and register overlap, all of which
are only described in the manual table.

### CDNA3-XML-014: Sparse MFMA structure, index selection, and layouts are prose-only

Manual evidence:

- Section 7.4 defines `V_SMFMAC` as a multiply-accumulate using 4:2 sparse
  matrix A, dense matrix B, and C/D in the destination register block at
  `cdna3/README.md:2065` through `:2069`.
- Table 32 records sparse F16, BF16, I8, BF8, and FP8 variants, block counts,
  and cycle counts at `cdna3/README.md:2071` through `:2084`.
- The sparse rules say A occupies two VGPRs per lane, B occupies four VGPRs
  per lane, C shares the destination VGPR offset, `SRC2` holds all sparse
  indexes and is VGPR-only, each index pair requires `index0 < index1` and
  `index0 != index1`, `SRC0`/`SRC1`/`VDST` VGPRs must be even-aligned, and
  `CBSZ`/`ABID` only select sparse-index data rather than ordinary A-matrix
  broadcast at `cdna3/README.md:2086` through `:2104`.
- Sections 7.4.1.1 and 7.4.1.2 define the 16-bit and 8-bit B-matrix layouts
  and sparse-index layouts at `cdna3/README.md:2112` through `:2190`.
- The detailed SMFMAC instruction definitions repeat sparse A, dense B, D as
  both output and accumulator, sparse-index use, and pass counts for F16/BF16/I8
  at `cdna3/README.md:11945` through `:12063` and for BF8/FP8 at
  `:12233` through `:12377`.

XML evidence:

- The `VOP3P_MFMA` encoding exposes `ABID`, `ACC`, `ACC_CD`, `BLGP`, and
  `CBSZ` fields at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna3.xml:7106` through `:7222`;
  its `ACC_CD` description only references an `OPF_ACC_CD_ONLY_D` flag, and
  that flag name appears nowhere else in the checked-in CDNA3 XML.
- Representative SMFMAC entries such as `V_SMFMAC_F32_16X16X32_F16`,
  `V_SMFMAC_I32_32X32X32_I8`, and
  `V_SMFMAC_F32_16X16X64_BF8_BF8` are present at
  `amdgpu_isa_cdna3.xml:67355` through `:67644` and `:68171` through `:68205`.
- Those entries encode basic operand size/type metadata, including
  `SRC2` as `OPR_SRC_VGPR` and `VDST` as both input and output, but they do not
  encode the 4:2 reconstruction contract, sparse-index pair legality,
  `CBSZ`/`ABID` index-set selection, index/B-matrix layouts, even-alignment
  rule, pass/cycle metadata, or the SMFMAC-specific `ACC_CD` rule in a
  searchable per-instruction flag.

Impact:

XML consumers can enumerate CDNA3 SMFMAC opcodes and recover basic operand
classes, but cannot validate or emulate sparse matrix behavior from XML alone.
The sparse-index selection and layout rules require manual-derived metadata.

### CDNA3-XML-015: I8 SMFMAC `SRC1` data format contradicts the manual

Manual evidence:

- The sparse I8 inventory row says `V_SMFMAC_I32_*_I8` is sparse matrix
  multiply of I8 data at `cdna3/README.md:2079` through `:2080`.
- The detailed `V_SMFMAC_I32_16X16X64_I8` and
  `V_SMFMAC_I32_32X32X32_I8` definitions say matrix A is sparse signed 8-bit
  data and matrix B is dense signed 8-bit data at `cdna3/README.md:12025`
  through `:12057`.

XML evidence:

- The dense CDNA3 I8 MFMA entries use `FMT_NUM_I8` for both `SRC0` and `SRC1`,
  for example `V_MFMA_I32_32X32X4_2B_I8` at
  `amdgpu_isa_cdna3.xml:66782` through `:66793` and
  `V_MFMA_I32_32X32X16_I8` at `:66935` through `:66945`.
- `V_SMFMAC_I32_16X16X64_I8` encodes `SRC0` as `FMT_NUM_I8`, but encodes
  `SRC1` as `FMT_NUM_I32` at `amdgpu_isa_cdna3.xml:67576` through `:67586`.
- `V_SMFMAC_I32_32X32X32_I8` repeats the same pattern at
  `amdgpu_isa_cdna3.xml:67627` through `:67637`.

Impact:

The XML contradicts the manual for the dense B operand of CDNA3 sparse I8
SMFMAC. A generator or oracle that relies on XML data formats could classify
matrix B as 32-bit integer data even though the opcode and prose require packed
signed 8-bit input data.

### CDNA3-XML-016: SMEM addressing, dependency, clause, and bounds rules are prose-only

Manual evidence:

- Chapter 8 says scalar-memory loads and stores access SGPR data through the
  scalar data cache, with load/store width limits and no byte/short data
  conversion at `cdna3/README.md:2313` through `:2316`.
- Table 38 defines `IMM`, `SOE`, `GLC`, `SDATA`, `SBASE`, and `NV` behavior at
  `cdna3/README.md:2325` through `:2342`.
- Section 8.1.1 gives separate formulas for scalar/global, scratch, and buffer
  scalar-memory addressing, including `IMM`/`SOE` selection, M0/SGPR offset
  forms, scratch 64-byte offset scaling, low-bit masking, private-access
  conversion, buffer-resource descriptor fields, and stride-only bounds use at
  `cdna3/README.md:2348` through `:2407`.
- Sections 8.2 through 8.4 describe scalar-memory source-overwrite and clause
  hazards, `LGKMCNT` increment/decrement behavior, out-of-order/partial return
  behavior, and SDATA/SBASE alignment/range and out-of-bounds rules at
  `cdna3/README.md:2432` through `:2456`.
- Chapter 13.2 records the SMEM field map and states that `OFFSET` can be an
  immediate signed byte offset or an SGPR address holding an unsigned byte
  offset at `cdna3/README.md:23039` through `:23114`.

XML evidence:

- `ENC_SMEM` carries field bit positions and short field descriptions at
  `amdgpu_isa_cdna3.xml:741` through `:943`, including `SOFFSET_EN`, `IMM`,
  `GLC`, `NV`, `SBASE`, `SDATA`, `SOFFSET`, and signed immediate-offset text.
- The XML has the audited SMEM instruction entries and opcode inventory, for
  example scalar loads/stores/cache/timer/discard entries at
  `amdgpu_isa_cdna3.xml:25780` through `:26350`.
- Searching the CDNA3 XML finds no scalar-memory clause, source-overwrite,
  scratch scaling, low-bit masking, buffer-resource bounds, private scalar
  access conversion, or `LGKMCNT` multi-dword weighting semantics.

Impact:

The XML is sufficient to decode the SMEM fields, but an XML-only consumer cannot
derive the address formulas, dependency-counter behavior, legality restrictions,
or buffer/bounds behavior needed for semantic emulation and validation.

### CDNA3-XML-017: Scalar atomic return and grouping metadata is inconsistent

Manual evidence:

- Section 8.1.2 says scalar atomics support the same set of operations as
  vector memory atomics, use the same address calculations, and return the
  pre-atomic value only when `GLC` is set at `cdna3/README.md:2414` through
  `:2416`.
- Section 8.3 says scalar memory instructions issue through scalar-memory
  dependency handling, and section 8.4 gives scalar-memory alignment/range rules
  at `cdna3/README.md:2432` through `:2456`.

XML evidence:

- Scalar atomic instructions are encoded with `ENC_SMEM`, but representative
  `S_BUFFER_ATOMIC_*` entries classify their functional group as `VMEM` /
  `ATOMIC` at `amdgpu_isa_cdna3.xml:26350` through `:26950`.
- The same entries mark `SDATA` as both input and output unconditionally, while
  the generic `GLC` field description at `amdgpu_isa_cdna3.xml:785` through
  `:789` only says the operation is globally coherent and does not encode the
  conditional return-data contract.

Impact:

An XML-driven generator can classify scalar atomics as vector-memory atomics and
can model their data operand as always written, even though the manual treats
them as SMEM instructions whose return value is gated by `GLC`.

### CDNA3-XML-018: Buffer offset and address-VGPR rules are partly misstated

Manual evidence:

- The MUBUF/MTBUF field table says `VADDR` supplies offset, index, or both, and
  says `SOFFSET` supplies an unsigned byte offset from an SGPR, M0, or inline
  constant at `cdna3/README.md:2506` through `:2513`.
- The address-VGPR table says zero, one, or two VGPRs are consumed based on
  `IDXEN`/`OFFEN` at `cdna3/README.md:2551` through `:2568`.
- Section 9.1.5 says the instruction offset is present regardless of `OFFEN`,
  and that the offset can come from an SGPR or VGPR and from the instruction
  itself at `cdna3/README.md:2606` through `:2618`.

XML evidence:

- `ENC_MUBUF` and `ENC_MTBUF` carry the relevant field bit positions, but their
  `OFFEN` descriptions say that only one of the VGPR offset and instruction
  offset may be sent at `amdgpu_isa_cdna3.xml:3083` through `:3085` and
  `:3279` through `:3281`.
- Representative `TBUFFER_LOAD_FORMAT_X` and `BUFFER_LOAD_FORMAT_X` entries
  encode `VADDR` as a fixed 64-bit operand at `amdgpu_isa_cdna3.xml:20948`
  through `:20953` and `:21716` through `:21720`.

Impact:

The XML is sufficient to decode the fields, but an XML-only consumer can derive
the wrong address expression when `OFFEN` is set and can over-consume VADDR
registers for instructions that use zero or one address VGPR.

### CDNA3-XML-019: Buffer resource, range, swizzle, and unbound rules are prose-only

Manual evidence:

- Section 9.1.5 defines buffer addressing from a resource base, SGPR offset,
  VGPR index/offset, stride, element size, `ADD_TID`, swizzle fields, and
  `NumRecords` at `cdna3/README.md:2606` through `:2648`.
- Section 9.1.5.1 defines distinct private, raw, and structured range-checking
  modes and the `dst_sel = SEL_1` out-of-bounds exception at
  `cdna3/README.md:2650` through `:2673`.
- Sections 9.1.5.2 and 9.1.8 define swizzled-addressing formulas and the
  128-bit buffer-resource descriptor fields, including base, stride, swizzle,
  `NumRecords`, `dst_sel`, `NFMT`, `DFMT`, user VM, `ADD_TID`, `NV`, type, and
  the all-zero unbound-resource behavior at `cdna3/README.md:2674` through
  `:2692` and `:2735` through `:2765`.

XML evidence:

- The `FMT_RSRC*` data formats describe a 128-bit buffer resource constant as a
  single opaque `Descriptor` field at `amdgpu_isa_cdna3.xml:92039` through
  `:92277`.
- The checked MUBUF/MTBUF encodings and instruction entries do not encode the
  descriptor bit layout, address formulas, swizzle formulas, private/raw/
  structured range modes, unbound-resource behavior, or `dst_sel` OOB
  exception.

Impact:

XML consumers cannot model most architecturally important buffer-resource
semantics without manual prose or a separate descriptor schema.

### CDNA3-XML-020: Buffer data-format, D16, and `dst_sel` semantics are incomplete

Manual evidence:

- Section 9.1.3 describes read/write data VGPR counts and format conversion at
  `cdna3/README.md:2568` through `:2574`.
- Section 9.1.4 says data size and type are controlled by `DFMT`, `NFMT`,
  `dst_sel`, and opcode, distinguishes MTBUF instruction format from MUBUF
  resource/derived format, states that an INVALID resource format remains
  unbound rather than being replaced by a derived format, and defines D16
  packing at `cdna3/README.md:2576` through `:2605`.
- Section 9.1.6 says ECC-enabled 16-bit memory loads write a full 32-bit VGPR
  with unused bits zeroed at `cdna3/README.md:2720` through `:2729`.
- Section 9.1.11 begins the buffer data-format enum table at
  `cdna3/README.md:2883` through `:2910`.

XML evidence:

- `ENC_MTBUF` exposes `DFMT` and `NFMT` fields at
  `amdgpu_isa_cdna3.xml:3228` through `:3265`, and representative formatted
  buffer entries describe conversion at `:20934` through `:21127` and `:21702`
  through `:21943`.
- The resource descriptor remains opaque in `FMT_RSRC*` at
  `amdgpu_isa_cdna3.xml:92039` through `:92277`, and the XML entries do not
  encode the full data-format enum semantics, `dst_sel` channel selection,
  INVALID/unbound override rule, D16 pair packing, or the ECC writeback rule.

Impact:

The XML tells a generator that formatted buffer operations exist, but not enough
to implement the typed/resource data conversion and destination-channel behavior
specified by the manual.

### CDNA3-XML-021: Buffer-to-LDS and vector cache controls are prose-only

Manual evidence:

- Section 9.1.9 limits memory-buffer load-to-LDS to a subset of MUBUF
  instructions, defines the LDS offset as `M0[15:0]`, and gives active-mask and
  LDS-allocation clamping rules at `cdna3/README.md:2767` through `:2790`.
- Section 9.1.10 gives vector-memory `SC[1:0]`/`NT` load, store, atomic,
  invalidate, and writeback cache-policy tables, including a `TG_SPLIT` special
  case, at `cdna3/README.md:2791` through `:2882`.

XML evidence:

- `ENC_MUBUF` only says that the `LDS` bit writes data to LDS instead of a VGPR
  at `amdgpu_isa_cdna3.xml:3062` through `:3065`.
- `ENC_MUBUF` and `ENC_MTBUF` expose `SC0`, `SC1`, and `NT` fields with short
  descriptions at `amdgpu_isa_cdna3.xml:3072` through `:3079`, `:3112`
  through `:3124`, `:3268` through `:3276`, and `:3308` through `:3320`.
- The XML does not encode the LDS-capable opcode subset, `M0[15:0]` LDS
  offset, LDS clamping rule, full cache-policy tables, or `TG_SPLIT`
  refinement.

Impact:

Decoding the fields from XML is possible, but execution and validation need
manual-derived rules for LDS destinations and cache policy.

### CDNA3-XML-022: Buffer atomic descriptions use stale `GLC` terminology

Manual evidence:

- Section 9.1.10.2 says buffer atomics use `SC0` to select whether the pre-op
  memory value is returned, while `SC1` selects device/system scope and `NT`
  selects allocation policy at `cdna3/README.md:2856` through `:2862`.

XML evidence:

- The `ENC_MUBUF` field metadata correctly says `SC0` means atomic with return
  at `amdgpu_isa_cdna3.xml:3112` through `:3115`.
- Individual buffer atomic descriptions still say the original value is stored
  into a vector register iff the `GLC` bit is set; representative entries begin
  with `BUFFER_ATOMIC_SWAP` and `BUFFER_ATOMIC_CMPSWAP` at
  `amdgpu_isa_cdna3.xml:23671` through `:23720`.

Impact:

The XML is internally inconsistent: field metadata names the CDNA3 `SC0`
contract, while the instruction prose uses the older `GLC` wording. Consumers
that parse descriptions can attach the return-data contract to a nonexistent
MUBUF field.

### CDNA3-XML-023: FLAT/GLOBAL/SCRATCH addressing and aperture rules are prose-only

Manual evidence:

- Chapter 10 says FLAT instructions use a VGPR flat address and aperture
  registers to distinguish video/system, LDS, and scratch spaces; unmapped
  regions generate memory violations at `cdna3/README.md:3019` through `:3021`.
- Section 10.1 defines segment selection, offset widths, `SADDR`, scratch
  `SVE` modes, and the implied `M0` LDS-address contribution at
  `cdna3/README.md:3023` through `:3042`, with the four scratch `SVE` address
  modes shown at `:3096` through `:3104`.
- Section 10.3 defines 32-bit versus 64-bit address mode, flat scratch address
  conversion, and LDS-return address formulas at `cdna3/README.md:3121`
  through `:3142`; sections 10.4 and 10.5 define global and scratch address
  formulas at `:3154` through `:3184`.

XML evidence:

- `ENC_FLAT`, `ENC_FLAT_GLBL`, and `ENC_FLAT_SCRATCH` expose field positions,
  segment constants, offset widths, and operand fields at
  `amdgpu_isa_cdna3.xml:3376` through `:3980`.
- Representative flat/global/scratch entries expose operands for loads, stores,
  atomics, and LDS forms, but the XML does not encode `PTR32` or
  `ADDRESS_MODE`, aperture base/limit routing, memory-violation policy,
  scratch swizzle formulas, automatic flat-scratch state, or the direct-LDS
  address formulas.

Impact:

The XML is adequate for raw field decode, but a decoder or emulator cannot
recover the address-space routing, address-size selection, or fault behavior
without manual-derived rules.

### CDNA3-XML-024: Bit 13 `SVE`/`LDS` semantics are inconsistent and under-described

Manual evidence:

- The Chapter 10.1 field table names bit 13 `SVE` and describes it as scratch
  VGPR enable at `cdna3/README.md:3023` through `:3042`, with the `SVE`
  address-mode table at `:3096` through `:3104`.
- The Chapter 13.6.1 FLAT format table names the same bit `LDS` and describes
  it as selecting LDS-memory transfer rather than VGPR-memory transfer at
  `cdna3/README.md:24435` through `:24458`.

XML evidence:

- `ENC_FLAT`, `ENC_FLAT_GLBL`, and `ENC_FLAT_SCRATCH` name bit 13 `SVE`; the
  global and scratch encodings carry the same short scratch-VGPR-enable
  description at `amdgpu_isa_cdna3.xml:3376` through `:3980`.
- The XML opcode inventory contains LDS-specific global/scratch load entries,
  but the encoding metadata does not state when bit 13 is scratch VADDR enable
  versus direct-LDS transfer.

Impact:

This is a manual/XML metadata ambiguity, not just a missing field. Codegen needs
manual-specific policy to decide whether bit 13 should be surfaced as `SVE`,
`LDS`, or both depending on segment and opcode family.

### CDNA3-XML-025: FLAT wait-counter, ordering, and timing rules are prose-only

Manual evidence:

- Section 10.2 says FLAT instructions are internally executed as both LDS and
  Buffer operations, increment both `VM_CNT` and `LGKM_CNT`, and complete only
  when both counters have decremented at `cdna3/README.md:3105` through `:3111`.
- Sections 10.2.1 and 10.2.2 say FLAT operations can complete out of order, two
  returns to the same VGPR have unknown result, and only `S_WAITCNT 0` is
  sensible because VM and LGKM paths can race at `cdna3/README.md:3113`
  through `:3119`.

XML evidence:

- The XML exposes flat/global/scratch instruction entries and `SC`/`NT` cache
  fields, but no instruction metadata records dual counter increments,
  counter-retirement conditions, ordering, same-destination hazards, or the
  `S_WAITCNT 0` restriction.

Impact:

The XML cannot drive wait-counter or hazard modeling for FLAT instructions by
itself.

### CDNA3-XML-026: FLAT/GLOBAL atomic descriptions use stale `GLC` wording and miss FP rules

Manual evidence:

- Section 10.3.1 says floating-point atomics must set `SC[0]=0`, that FP32
  atomics flush denormals to zero, FP64 and FP16 atomics do not flush denormals,
  and rounding is fixed round-to-nearest-even at `cdna3/README.md:3146`
  through `:3152`.
- Representative atomic instruction definitions say return data is written iff
  the `SC0` bit is set at `cdna3/README.md:21283` through `:21307` and
  `:21529` through `:21690`.

XML evidence:

- `ENC_FLAT` field metadata correctly describes `SC0` as atomic with return at
  `amdgpu_isa_cdna3.xml:3557` through `:3558`.
- Individual flat/global atomic descriptions still say the old value is stored
  iff the `GLC` bit is set; representative entries include `FLAT_ATOMIC_SWAP`
  around `amdgpu_isa_cdna3.xml:14901` through `:14930` and the corresponding
  global atomic entries around `:17835` and later.
- The XML does not encode the floating-point atomic `SC0=0`, denormal, or
  rounding requirements.

Impact:

As with MUBUF atomics, XML consumers that use instruction descriptions can bind
the return contract to stale terminology and will miss FP atomic legality and
numeric behavior.

### CDNA3-XML-027: Direct LDS movement rules lack executable metadata

Manual evidence:

- Section 10.4 says GLOBAL instructions can move data directly between LDS and
  memory at `cdna3/README.md:3154` through `:3167`.
- Section 10.5 says SCRATCH instructions can also move data directly between
  LDS and memory at `cdna3/README.md:3169` through `:3184`.
- Section 10.3 gives the LDS address formulas using the hardware LDS base,
  `M0[17:2] * 4`, instruction offset, and `ThreadID` scaling at
  `cdna3/README.md:3137` through `:3142`.

XML evidence:

- The XML includes global and scratch LDS load entries, for example
  `GLOBAL_LOAD_LDS_*` around `amdgpu_isa_cdna3.xml:17599` through `:17820`
  and scratch LDS entries around `:20674` through `:20882`.
- Ordinary FLAT/GLOBAL/SCRATCH load rows expose an implicit `OPR_SDST_M0`
  operand, including `FLAT_LOAD_UBYTE_D16`, `GLOBAL_LOAD_DWORD`, and
  `SCRATCH_LOAD_SHORT_D16`, even though Chapter 10 says `M0` is used for
  SCRATCH and GLOBAL only when `LDS=1` and that FLAT cannot return data
  directly to LDS, at `amdgpu_isa_cdna3.xml:14650`, `:17531`, and `:20549`.
- The direct-LDS rows still expose `VDST` as a VGPR/AccVGPR output even though
  their descriptions say the result is stored into data share, for example
  `GLOBAL_LOAD_LDS_UBYTE` and `SCRATCH_LOAD_LDS_UBYTE` at
  `amdgpu_isa_cdna3.xml:17599` and `:20674`.
- Those entries do not encode the `M0` bit slice, thread scaling, LDS
  allocation behavior, or the relationship between the LDS opcode forms and bit
  13 metadata.

Impact:

The opcode inventory is present, but direct-LDS execution needs manual-derived
address and destination rules.

### CDNA3-XML-028: LDS bank geometry, conflicts, and M0 clamp rules are prose-only

Manual evidence:

- Section 11.1 describes LDS as 64 KiB per compute unit, split into 32 banks of
  512 dwords, with dwords placed serially across banks and bank conflicts
  serialized for indexed and atomic operations at `cdna3/README.md:3206`
  through `:3222`.
- Section 11.3.1 says LDS operations require `M0` to be initialized, and that
  `M0[16:0]` contains the byte-size of the LDS segment and clamps the final
  address at `cdna3/README.md:3240` through `:3265`.

XML evidence:

- `ENC_DS` records raw fields, including `GDS`, `OFFSET0`, `OFFSET1`, `OP`,
  `ADDR`, `DATA0`, `DATA1`, and `VDST`, at
  `amdgpu_isa_cdna3.xml:2683` through `:2933`.
- The XML field metadata does not encode LDS bank count/width, bank-conflict
  serialization, two-cycle versus 64-cycle behavior, required `M0`
  initialization, `M0[16:0]` meaning, or clamping policy.

Impact:

The XML can drive DS bitfield decode, but it cannot drive LDS bounds, timing, or
bank-conflict modeling without manual prose.

### CDNA3-XML-029: DS address formulas and duplicate-offset behavior are prose-only

Manual evidence:

- Section 11.3.1 defines single-address LDS address calculation as
  `LDS_BASE + VGPR[ADDR] + {InstrOffset1,InstrOffset0}` at
  `cdna3/README.md:3282` through `:3284`.
- The same section defines two-address formulas using `InstrOffset0 * ADJ` and
  `InstrOffset1 * ADJ`, where `ADJ` depends on data width, and separately says
  `*ST64` opcodes multiply offsets by 64 elements at `cdna3/README.md:3290`
  through `:3298`.
- Setting both two-address offsets to the same value specifies only one
  address, causes only one read/write, and uses only `DATA0` at
  `cdna3/README.md:3300` through `:3302`.

XML evidence:

- `ENC_DS` only says `OFFSET0` and `OFFSET1` are DS offsets whose use depends on
  individual instructions at `amdgpu_isa_cdna3.xml:2893` through `:2912`.
- Some ST64 instruction descriptions mention stride-64 offset scaling, for
  example `DS_WRITE2ST64_B32` at `amdgpu_isa_cdna3.xml:8655` through `:8701`,
  but the XML does not encode the general one-address/two-address formulas,
  per-width `ADJ`, or duplicate-offset single-access rule.

Impact:

Generated code needs manual-derived DS address semantics. XML-only consumers can
see the fields but cannot determine whether a given opcode should concatenate,
scale, issue two accesses, or collapse duplicate offsets.

### CDNA3-XML-030: LDS floating atomic FP-mode and rounding rules are prose-only

Manual evidence:

- Section 11.3.1 says LDS floating atomic denormal handling is controlled by
  `MODE.FP_DENORM` and rounding is fixed round-to-nearest-even at
  `cdna3/README.md:3314` through `:3318`.

XML evidence:

- The XML contains floating DS atomic entries, including F32/F64 and packed
  F16/BF16 forms, but the entries carry operand and format metadata only.
- Searching the CDNA3 XML for `FP_DENORM`, denormal-mode text, or RNE rules does
  not expose these LDS atomic requirements.

Impact:

The XML identifies floating atomic opcodes but cannot drive numerically exact
MODE-dependent behavior.

### CDNA3-XML-031: GWS execution and encoding restrictions are under-specified

Manual evidence:

- Section 11.4 says all GWS instructions must be followed immediately by
  `s_waitcnt 0` and that VGPRs used by GWS instructions must be even at
  `cdna3/README.md:3320` through `:3328`.
- Section 12.12 says GWS instructions operate only on a single lane, the first
  active lane in the EXEC mask, and that `GDS` is set for GWS and clear for LDS
  at `cdna3/README.md:18086` through `:18108`.
- The detailed GWS definitions calculate `rid[5:0]` from `gds_base[5:0]` and
  `offset0[5:0]`, define semaphore/barrier state counter behavior, and read the
  data-bearing operand from the first valid thread at `cdna3/README.md:19711`
  through `:19832`.
- `DS_GWS_INIT`, `DS_GWS_SEMA_BR`, and `DS_GWS_BARRIER` each caution that the
  VGPR operand must be even-aligned because hardware treats it as a 64-bit read,
  even though only 32 bits are used, at `cdna3/README.md:19745`, `:19782`, and
  `:19832`.

XML evidence:

- `ENC_DS` exposes the `GDS` bit as `1=GDS, 0=LDS` at
  `amdgpu_isa_cdna3.xml:2883` through `:2892`, and the DS opcode table contains
  GWS entries.
- The GWS instruction entries at `amdgpu_isa_cdna3.xml:13362` through `:13505`
  carry short descriptions and operand fields, but their encoding conditions do
  not require `GDS=1`.
- The XML does not encode the required following wait instruction,
  first-active-lane execution, `rid` calculation, semaphore/barrier state
  transitions, or the data-bearing GWS operand's even 64-bit-read restriction.

Impact:

Assembler, validator, or emulator logic that relies only on XML cannot enforce
GWS instruction-sequence, encoding, lane-selection, state-machine, or
register-legality requirements.

### CDNA3-XML-032: Packed F16/BF16 LDS atomics omit implicit DSMEM operands

Manual evidence:

- Section 11.3.1 describes LDS atomics as operations that combine VGPR data with
  LDS data and optionally return the pre-operation LDS value to VGPRs at
  `cdna3/README.md:3238` through `:3261`.
- The DS opcode table includes packed F16/BF16 LDS atomic add forms at
  `cdna3/README.md:24263` through `:24297`.

XML evidence:

- Ordinary DS atomics such as `DS_ADD_F32` carry implicit `OPR_DSMEM` input and
  output operands in the XML around `amdgpu_isa_cdna3.xml:8921` through `:8966`.
- The packed no-return entries `DS_PK_ADD_F16` and `DS_PK_ADD_BF16` at
  `amdgpu_isa_cdna3.xml:8975` through `:9037`, and the return entries
  `DS_PK_ADD_RTN_F16` and `DS_PK_ADD_RTN_BF16` at `:13561` through `:13636`,
  expose explicit VGPR operands but do not show the implicit `OPR_DSMEM`
  operands visible on neighboring atomic entries.

Impact:

XML consumers that use implicit operands for side-effect classification can miss
the LDS read/modify/write behavior on the packed F16/BF16 atomic entries.

### CDNA3-XML-033: SALU selector fallback and reserved-source details are incomplete

Manual evidence:

- Section 5.2 says SALU cannot use VGPRs or LDS, that 32-bit literals are
  available to all SALU formats except SOPP and SOPK, and that source selector
  255 consumes the following instruction dword at `cdna3/README.md:983`.
- The same section says out-of-range source SGPRs read SGPR0, out-of-range
  destination SGPRs suppress the SGPR write while still writing SCC and EXEC
  saveexec side effects, and 64-bit SGPR data must use an even-aligned pair at
  `cdna3/README.md:985` through `:989`.
- The scalar operand table marks source selector 239 as reserved and selectors
  249 through 250 as reserved at `cdna3/README.md:961` through `:971`; the
  duplicated SOP field tables repeat the 239 and 249-250 reserved rows, for
  example `cdna3/README.md:22733` through `:22748`.

XML evidence:

- The XML does model the SOP1/SOP2/SOPC 64-bit literal encodings and the
  one-opcode SOPK literal encoding for `S_SETREG_IMM32_B32` at
  `amdgpu_isa_cdna3.xml:3982` through `:5043`.
- The XML scalar source operand table carries generic SGPR pair-alignment prose
  on the SGPR entries and exposes special source selectors, including
  `src_pops_exiting_wave_id` at selector 239, at
  `amdgpu_isa_cdna3.xml:114438` through `:117683`.
- The XML entries for representative SALU instructions include implicit SCC
  operands, for example `S_ADD_U32` at `amdgpu_isa_cdna3.xml:31954` through
  `:31984`, `S_CMP_EQ_I32` at `:38994` through `:39018`, and
  `S_CBRANCH_SCC0` at `:42150` through `:42168`.
- The XML operand tables do not encode the runtime source-SGPR fallback to
  SGPR0, the destination no-write rule for out-of-range destination SGPRs, or
  the manual/XML disagreement on whether selector 239 is reserved or POPS.

Impact:

XML-only consumers can recover the ordinary SOP field maps, literal-extension
forms, and many SCC side effects, but still need manual knowledge for SALU
allocation-bound behavior and for resolving the scalar-source selector 239
disagreement.

### CDNA3-XML-034: `S_MAX_{I32,U32}` tie SCC predicate is not machine-readable

Manual evidence:

- The detailed `S_MAX_I32` and `S_MAX_U32` definitions say the first source is
  selected on ties by assigning `SCC = S0 >= S1` and then selecting
  `D0 = SCC ? S0 : S1` at `cdna3/README.md:3469` through `:3482`.
- CDNA4 and RDNA4 detailed manuals use the same inclusive max predicate at
  `cdna4/README.md:3940` through `:3953` and `rdna4/README.md:8447` through
  `:8460`, so the inclusive tie behavior is not isolated to one CDNA3 entry.

XML evidence:

- The CDNA3 XML entries for `S_MAX_I32` and `S_MAX_U32` say only "set SCC iff
  the first value is selected" at `amdgpu_isa_cdna3.xml:33082` through
  `:33112` and `:33218` through `:33248`.
- Those entries expose the implicit SCC output operand but do not encode the
  actual comparison predicate or tie-selection rule.

Impact:

An XML consumer can learn that `S_MAX_{I32,U32}` writes SCC, but cannot derive
that equality sets SCC. Code generation or tests driven from XML plus summary
tables can accidentally choose strict `>` while still producing the correct
numeric destination for equal operands.

### CDNA3-XML-035: HWREG map and operand metadata are incomplete or inconsistent

Manual evidence:

- Section 5.8 defines `SIMM16 = {size[4:0], offset[4:0], hwRegId[5:0]}`,
  with offset 0 through 31 and logical size 1 through 32, at
  `cdna3/README.md:1115` through `:1119`.
- Table 19 defines CDNA3 hardware register IDs, including `MODE` at 1,
  read-only `STATUS` at 2, `TRAPSTS` at 3, `HW_ID` at 4, `GPR_ALLOC` at 5,
  `LDS_ALLOC` at 6, `IB_STS` at 7, reserved IDs 8 through 15, trap-base/memory
  IDs 16 through 19, `XCC_ID` at 20, and performance-snapshot IDs 21 through 24
  at `cdna3/README.md:1121` through `:1140`.
- Tables 20 through 22 define bit layouts for `IB_STS`, `GPR_ALLOC`, and
  `LDS_ALLOC`, including split VM count bits, allocation bases/sizes, and LDS
  base/size units at `cdna3/README.md:1144` through `:1167`.

XML evidence:

- The CDNA3 XML does define `OPR_HWREG` as a partitioned operand with `ID`,
  `OFFSET`, and `SIZE` fields at `amdgpu_isa_cdna3.xml:93610` through
  `:93773`.
- The `OPR_HWREG` field descriptions are empty, and its predefined-value
  descriptions do not carry the manual's R/W/read-only/debug/allocation/status
  field descriptions.
- The CDNA3 XML assigns IDs 8 through 15 to `hw_reg_pc_lo`, `hw_reg_pc_hi`,
  `hw_reg_inst_dw0`, `hw_reg_inst_dw1`, `hw_reg_ib_dbg0`, `hw_reg_ib_dbg1`,
  `hw_reg_flush_ib`, and `hw_reg_sh_mem_bases` at
  `amdgpu_isa_cdna3.xml:93660` through `:93722`, while the CDNA3 and CDNA4
  manuals reserve those IDs.
- The CDNA3 XML instruction entries for `S_GETREG_B32`, `S_SETREG_B32`, and
  `S_SETREG_IMM32_B32` mark the `SIMM16`/`OPR_HWREG` operand with
  `OperandSize` 32 at `amdgpu_isa_cdna3.xml:41899` through `:41984`, despite
  the 16-bit encoded field and `FMT_NUM_B16`. The CDNA4 entries use
  `OperandSize` 16 for the same operand at `amdgpu_isa_cdna4.xml:45518`
  through `:45605`.

Impact:

XML-only consumers can recover the HWREG bit partition, but not the CDNA3
register semantics or allocation/status subfields. They also see an ID map that
contradicts the CDNA3/CDNA4 prose for IDs 8 through 15 and an operand-size
inconsistency that can leak into generated operand metadata.

### CDNA3-XML-036: SETREG write masks, side effects, and spacing are prose-only

Manual evidence:

- Section 5.8 says `S_SETREG_B32` writes the LSBs of the source SGPR to a
  hardware register, and software must insert an `S_NOP` between two
  consecutive `S_SETREG` writes to the same register at `cdna3/README.md:1108`
  through `:1113`.
- The detailed `S_SETREG_B32` and `S_SETREG_IMM32_B32` pseudocode applies
  `HwRegWriteMask(hwRegId, WAVE_STATUS.PRIV)`, preserves bits outside the
  writable mask, and notes that side effects may trigger when some bits are
  modified at `cdna3/README.md:4074` through `:4119`.

XML evidence:

- The CDNA3 XML entries for `S_SETREG_B32` and `S_SETREG_IMM32_B32` only carry
  operand metadata and high-level descriptions at `amdgpu_isa_cdna3.xml:41934`
  through `:41992`.
- `OPR_HWREG` does not encode per-register writable masks, privilege-sensitive
  write permissions, side effects, or the consecutive-SETREG spacing
  requirement.

Impact:

An XML-derived emulator or validator cannot tell which HWREG bits are writable,
which writes require privilege, which writes have side effects, or which
instruction sequences require a scheduling NOP.

### CDNA3-XML-037: Wave-state register fields and helper semantics are prose-only

Manual evidence:

- Chapter 3 defines PC as a byte address initialized at wave creation, with
  `S_GET_PC`, `S_SET_PC`, and `S_SWAP_PC` transferring PC through an
  even-aligned SGPR pair and branches relative to the instruction after the
  branch at `cdna3/README.md:400` through `:406`.
- EXEC is a 64-bit vector execution mask that affects vector ALU, vector memory,
  and LDS instructions but not scalar execution or branches; `EXECZ` is a helper
  branch condition, and the manual points shaders to `CBRANCH`/`VSKIP` when
  EXEC is likely zero at `cdna3/README.md:408` through `:416`.
- The STATUS table gives concrete bit positions for `EXECZ`, `VCCZ`,
  `IN_TG`, `IN_BARRIER`, `HALT`, `TRAP`, `VALID`, `ECC_ERR`, `PERF_EN`,
  conditional-debug bits, `ALLOW_REPLAY`, `FATAL_HALT`, `SCRATCH_EN`, and
  `IDLE` at `cdna3/README.md:418` through `:448`.
- The MODE table gives concrete fields for `FP_ROUND`, `FP_DENORM`,
  `DX10_CLAMP`, `IEEE`, `DEBUG`, `EXCP_EN`, `FP16_OVFL`, `DISABLE_PERF`,
  `GPR_IDX_EN`, `VSKIP`, and `CSP` at `cdna3/README.md:450` through `:473`.

XML evidence:

- A focused search of `amdgpu_isa_cdna3.xml` only finds isolated instruction
  descriptions for `S_SETVSKIP`, `S_CBRANCH_VCCZ`, `S_CBRANCH_EXECZ`, and
  wait-count fields at `amdgpu_isa_cdna3.xml:40786` through `:40787`,
  `:42218` through `:42321`, and `:122864` through `:122884`; it does not expose
  the Chapter 3 STATUS or MODE register field layout as machine-readable state.
- `OPR_HWREG` has the raw `{ID, OFFSET, SIZE}` partition but an empty register
  description at `amdgpu_isa_cdna3.xml:93610` through `:93617`, so the state
  table and per-field semantics are not recoverable from the operand metadata.

Impact:

XML-derived decoders can recognize some branch and HWREG encodings, but cannot
derive the architectural wave-state register map, MODE/STATUS field semantics,
or the relationship between helper bits such as `EXECZ`/`VCCZ` and the EXEC/VCC
state.

### CDNA3-XML-038: GPR/LDS allocation, aliasing, and out-of-range behavior are prose-only

Manual evidence:

- Chapter 3 defines out-of-range SGPR/VGPR behavior: source reads fall back to
  register 0, destination writes are suppressed, LDS out-of-range reads return
  zero, LDS out-of-range writes are discarded, and memory/LDS/GDS return
  operations are nullified if any destination VGPR in the return range is
  out-of-range at `cdna3/README.md:475` through `:513`.
- SGPR allocation is 16 to 102 dwords in units of 16; VCC is physically stored
  in SGPRs 106 and 107 as the source/destination VCC alias, with trap handler
  SGPRs reserved after VCC at `cdna3/README.md:515` through `:517`.
- VGPR allocation is in groups of eight dwords, split across regular VGPR and
  AccVGPR pools with up to 256 of each, and 64-bit/GWS operations require
  even-aligned VGPRs at `cdna3/README.md:528` through `:538`.
- LDS is allocated in contiguous 512-byte blocks on 512-byte alignment, does not
  wrap, and LDS clamping uses the smaller of the SPI allocation size and M0 at
  `cdna3/README.md:540` through `:544`.

XML evidence:

- The SGPR operand description only says `NUM_SGPR` SGPRs exist and records
  even/quad alignment for wide values at `amdgpu_isa_cdna3.xml:93808` through
  `:93811`; it does not encode the 16-dword allocation granularity, 102 logical
  SGPR limit, VCC physical alias, trap-reserve layout, or out-of-range fallback.
- The AccVGPR and VGPR operand descriptions similarly record generic
  `NUM_ACCVGPR`/`NUM_VGPR` counts and even-alignment text at
  `amdgpu_isa_cdna3.xml:92305` through `:92308` and `:117713` through
  `:117715`, but not the allocation split or destination-nullification rules.
- `OPR_VCC` only describes `VCC[63:0]` at
  `amdgpu_isa_cdna3.xml:117698` through `:117704`, and `OPR_FLAT_SCRATCH`
  carries an empty top-level description at `amdgpu_isa_cdna3.xml:93598`
  through `:93607`.

Impact:

An XML-only legality checker or emulator cannot enforce CDNA3 allocation
granularity, VCC aliasing, trap SGPR reserves, LDS allocation/clamping, or the
specified out-of-range behavior for register and LDS accesses.

### CDNA3-XML-039: M0, SCC, and VCC helper semantics are mostly prose-only

Manual evidence:

- Chapter 3.7 defines M0 as a per-wavefront 32-bit memory descriptor used for
  LDS addressing, GWS, and indirect GPR addressing, including exact bit layouts
  for LDS and GWS at `cdna3/README.md:546` through `:554`.
- Chapter 3.8 defines SCC producer categories and consumer roles: comparisons
  set true/false, arithmetic uses carry out, bit/logical operations use
  result-nonzero, moves do not alter SCC, and SCC feeds carry-in, conditional
  moves, and branches at `cdna3/README.md:556` through `:568`.
- Chapter 3.9 says VCCZ updates whenever VCC updates, including scalar writes
  to VCC; vector compares fully write VCC under EXEC; VCC physically resides in
  the SGPR file; and scalar writes to the physical VCC alias have a branch
  hazard when followed by a VCCZ conditional branch at `cdna3/README.md:570`
  through `:588`.

XML evidence:

- The XML records only a generic `m0` description, "Special register used to
  hold LDS/GDS addresses, relative indices, and send-message values", in the
  scalar source and destination operand tables at `amdgpu_isa_cdna3.xml:94431`
  through `:94433` and `:94465` through `:94472`; it does not encode the
  Chapter 3 bit layouts or per-use meaning.
- The XML does include coarse SCC and VCC operands, including
  `OPR_SSRC_SPECIAL_SCC` as "Scalar condition code" at
  `amdgpu_isa_cdna3.xml:117687` through `:117693` and `OPR_VCC` as
  `VCC[63:0]` at `:117698` through `:117704`.
- The VOPC descriptions record that vector compares are masked by EXEC, but
  the XML does not encode the VCCZ update-on-scalar-write rule, the full-write
  contract, the physical SGPR aliasing hazard, or the SCC producer taxonomy.

Impact:

An XML-only emulator or verifier can name these special operands, but cannot
derive their Chapter 3 state semantics, helper-bit update rules, or VCC alias
hazard without manual prose.

### CDNA3-XML-040: Trap, memory-violation, XNACK, HW_ID, and launch-initialization state is not machine-readable

Manual evidence:

- The Chapter 3.1 state table defines `FLAT_SCRATCH`, `XNACK_MASK`,
  `TRAPSTS`, `TBA`, `TMA`, `TTMP0-TTMP15`, `VMCNT`, `EXPCNT`, and `LGKMCNT`
  as shader-visible state at `cdna3/README.md:362` through `:389`.
- Chapter 3.10 defines TTMP privilege behavior, trap entry payload, TBA/TMA
  access, `STATUS.TRAP_EN`, trap-handler SGPR reservation, and `MODE.EXCP_EN`
  exception enables at `cdna3/README.md:590` through `:618`.
- Chapter 3.10.1 defines TRAPSTS sticky exception/trap fields, including
  `SAVECTX`, `ILLEGAL_INST`, address-watch bits, `EXCP_CYCLE`, and `DP_RATE` at
  `cdna3/README.md:619` through `:632`.
- Chapter 3.11 defines memory-violation sources, non-sources, buffer-to-LDS
  EXEC masking, sticky `TRAPSTS.mem_viol`, trap-handler interaction, and
  imprecise saved-PC behavior at `cdna3/README.md:634` through `:654`.
- Chapter 3.12 gives `HW_ID` and `XCC_ID` bitfields, and Chapter 3.13 defines
  VGPR0, system SGPR, and TTMP initialization at `cdna3/README.md:656`
  through `:693`.

XML evidence:

- The XML HWREG table carries raw `{ID, OFFSET, SIZE}` fields and names such as
  `trapsts`, `hw_id`, `tba_lo`, `tma_lo`, and `xcc_id` at
  `amdgpu_isa_cdna3.xml:93610` through `:93775`, but most descriptions are
  empty and the table does not encode the Chapter 3 bitfields or side effects.
- The TTMP operand entry contains partial trap-handler commentary at
  `amdgpu_isa_cdna3.xml:94370` through `:94390`, but not the user-mode
  read-as-zero/write-ignore rule or the full trap-entry state transition.
- The scalar selector tables name `XNACK_MASK_LO` and `XNACK_MASK_HI` with
  only short "see XNACK replay mechanism" descriptions at
  `amdgpu_isa_cdna3.xml:103593` through `:103600`; they do not encode how the
  address-translation-failure mask is produced, cleared, or consumed.
- Searching the CDNA3 XML for `SAVECTX`, `ILLEGAL_INST`, `ADDR_WATCH`,
  `EXCP_CYCLE`, `DP_RATE`, and `MEM_VIOL` finds no machine-readable state
  fields, and the XML has no representation for Chapter 3.13 launch-time
  register initialization.

Impact:

Code generated from XML can parse an HWREG operand and name some special
selectors, but cannot model CDNA3 trap/exception state, XNACK replay mask
state, memory-violation side effects, HW_ID/XCC_ID contents, or initial
wavefront register state without the manual.

### CDNA3-XML-041: Program-control branch-stack and status side effects are under-described

Manual evidence:

- Chapter 4.1 lists control instructions that terminate, trap/return, alter
  wave priority, sleep, send messages, and wake sleeping waves at
  `cdna3/README.md:701` through `:717`.
- Chapter 4.2 lists ordinary predicate branches, debug-status conditional
  branches, PC get/set/swap, fork/join divergent-control branches, and VSKIP at
  `cdna3/README.md:719` through `:735`.
- The detailed fork/join prose defines a six-deep CSP stack in SGPRs, stack
  entries containing `{EXEC, PC}`, pass/fail mask selection by population
  count, and JOIN restore behavior at `cdna3/README.md:842` through `:890`.
- Detailed instruction definitions say `S_RFE_B64` clears `WAVE_STATUS.PRIV`
  and jumps to the scalar address at `cdna3/README.md:4654` through `:4664`,
  and say `S_TRAP` writes TTMP0/1, jumps to TBA, and sets PRIV at
  `cdna3/README.md:5478` through `:5494`.

XML evidence:

- The XML carries useful opcode and operand entries for this slice, including
  `S_RFE_B64` at `amdgpu_isa_cdna3.xml:30260` through `:30305`,
  `S_CBRANCH_JOIN` at `:31370` through `:31405`, `S_CBRANCH_G_FORK` at
  `:37505` through `:37535`, and `S_CBRANCH_I_FORK` at `:41860` through
  `:41884`.
- The fork/join XML descriptions only say "Conditional branch using
  branch-stack" or "Conditional branch join point"; `S_CBRANCH_JOIN` exposes
  implicit EXEC and PC outputs, but the entries do not expose CSP storage, the
  six stack entries, SGPR stack read/write side effects, pass/fail mask
  selection, or the conditions that decide when PC/EXEC are restored.
- Debug conditional branches are flagged as conditional branches with a label
  operand at `amdgpu_isa_cdna3.xml:42720` through `:42833`, but the entries do
  not include implicit `STATUS.COND_DBG_SYS` or `STATUS.COND_DBG_USER`
  operands.
- `S_RFE_B64`'s text mentions clearing PRIV, and `S_SENDMSGHALT`'s text
  mentions HALT at `amdgpu_isa_cdna3.xml:30268` through `:30285` and `:42547`
  through `:42570`, but those status writes are not represented as structured
  implicit operands or state effects.
- The detailed SOPP definitions say `S_INCPERFLEVEL` and `S_DECPERFLEVEL`
  update performance counters and `S_TTRACEDATA` sends M0 to the thread-trace
  stream at `cdna3/README.md:5503` through `:5513`; the XML rows at
  `amdgpu_isa_cdna3.xml:44453`, `:44485`, and `:44517` carry descriptions and
  operands, but not structured performance-counter or thread-trace state
  effects.

Impact:

XML-only code can decode the Chapter 4.1/4.2 opcodes, but cannot build an
accurate control-flow emulator or side-effect verifier for fork/join, debug
branch predicates, trap return, message-halt, priority, performance/thread-trace
controls, or wake/sleep behavior without manual semantics.

### CDNA3-XML-042: Workgroup barrier membership and wait-counter semantics are prose-only

Manual evidence:

- Chapter 3 defines `STATUS.IN_BARRIER` bit 12 as "Wavefront is waiting at a
  barrier" at `cdna3/README.md:430` through `:437`.
- Chapter 4.3 says workgroups may contain up to 16 wavefronts or 1024
  work-items, that `S_BARRIER` waits until all other wavefronts reach the same
  instruction, and that early `S_ENDPGM` termination leaves only the remaining
  live waves to satisfy the barrier at `cdna3/README.md:746` through `:748`.
- The detailed `S_BARRIER` definition says it waits for waves in the threadgroup
  that have not yet been created, ignores already terminated waves, is legal
  inside trap handlers, and does not wait for counters to become zero before
  issuing at `cdna3/README.md:5400` through `:5406`.

XML evidence:

- The checked-in XML `S_BARRIER` entry only carries the SOPP opcode and the
  short description "Synchronize waves within a threadgroup" at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna3.xml:42346` through
  `:42367`.
- The entry has no operands, implicit status effects, workgroup-membership
  metadata, live-wave membership rule, trap-handler legality flag, or
  wait-counter interaction rule.

Impact:

The XML can identify `S_BARRIER`, but it cannot drive barrier scheduling,
status-bit reporting, workgroup-size validation, or the rule that barriers do
not drain wait counters.

### CDNA3-XML-043: Wait-counter producer, decrement, and ordering semantics are prose-only

Manual evidence:

- Chapter 4.4 says `S_WAITCNT` waits for `VM_CNT`, `LGKM_CNT`, and `EXP_CNT`
  to be at or below specified thresholds, and that same-type operations return
  in order while different instruction types can return out of order at
  `cdna3/README.md:752` through `:758`.
- The same section defines `VM_CNT` producer/decrement rules for MUBUF, MTBUF,
  and FLAT instructions at `cdna3/README.md:760` through `:763`.
- It defines `LGKM_CNT` producer/decrement rules for LDS, GWS, scalar-memory,
  FLAT, `S_SENDMSG`, and SMEM return dwords at `cdna3/README.md:764` through
  `:778`.
- It defines `EXP_CNT` as the GWS VGPR-export count, including GWS issue and
  grant/decrement points at `cdna3/README.md:779` through `:784`.

XML evidence:

- The checked-in XML `S_WAITCNT` entry only gives a generic description and
  the SOPP `SIMM16`/`OPR_WAITCNT` operand at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna3.xml:42397` through
  `:42425`.
- The `OPR_WAITCNT` partition records the `EXP`, `LGKM`, `VM`, and `VM_HI`
  threshold bit fields and "do not wait" encodings at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna3.xml:122857` through
  `:122900`.
- The XML does not record which instruction families increment or decrement
  each counter, scalar-memory dword-count behavior, FLAT dual-counter behavior,
  GWS `EXP_CNT` behavior, or the scalar-memory/FLAT `S_WAITCNT 0` ordering
  restrictions.

Impact:

The XML is sufficient to decode and print a wait-count operand, but it cannot
drive architectural wait-counter accounting, hazard modeling, or tests for the
producer-specific ordering rules without manual-specific policy.

### CDNA3-XML-044: Software-inserted wait-state hazard table is prose-only

Manual evidence:

- Chapter 4.5 says hardware does not check several dependency classes, so the
  shader must resolve them by inserting NOPs or independent instructions at
  `cdna3/README.md:786` through `:788`.
- Table 11 lists required waits for `S_SETREG`/`S_GETREG`, `MODE.VSKIP`,
  VALU-produced VCC/EXEC/SGPR/VGPR values, `V_READLANE`/`V_WRITELANE`,
  `V_DIV_FMAS`, FLAT/GLOBAL/SCRATCH/BUFFER write-data hazards, M0 consumers,
  TRAPSTS/RFE, DPP, OPSEL/SDWA bit-position changes, and trans-op consumers at
  `cdna3/README.md:790` through `:822`.
- Table 12 enumerates the trans-op instruction set for the Table 11 trans-op
  wait row at `cdna3/README.md:824` through `:831`.
- The detailed `S_ICACHE_INV` row says 16 separate `S_NOP` instructions or a
  jump/branch must follow instruction-cache invalidation to ensure internal
  instruction buffers are also invalidated at `cdna3/README.md:5498` through
  `:5504`.

XML evidence:

- A targeted search of the checked-in CDNA3 XML for Table 11 phrases such as
  `wait state`, `Required Software`, `five wait`, `VCC can be accessed`,
  `OPSEL or SDWA`, and `Trans op` returned no matches.
- Representative XML instruction entries carry opcode, operand, and short
  description metadata only: `S_SETVSKIP` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna3.xml:40780` through
  `:40805`, `S_SETREG_B32`/`S_SETREG_IMM32_B32` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna3.xml:41934` through
  `:41992`, `S_NOP` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna3.xml:42048` through
  `:42068`, `S_ICACHE_INV` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna3.xml:44428`, and
  `V_READLANE_B32`/`V_WRITELANE_B32` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna3.xml:63609` through
  `:63670`.
- The XML exposes many DPP encodings, but no structured producer/consumer,
  register-alias, trans-op, or required-wait-count metadata for the Table 11
  hazards.

Impact:

The XML can decode the individual instructions and `S_NOP`, but it cannot
drive a validator, scheduler, or emulator hazard model for CDNA3's manually
inserted wait-state requirements without manual-specific tables.

### CDNA3-XML-045: VALU scalar-source budget restrictions are prose-only

Manual evidence:

- Chapter 6.2.1 says VALU inputs may be VGPRs, SGPRs, inline constants,
  literals, M0, and EXEC, but only one SGPR can be read per instruction, only one
  literal can be used, and literals are allowed only when no SGPR or M0 source is
  used at `cdna3/README.md:1217` through `:1229`.
- The same subsection says `ADDC`, `SUBB`, and `CNDMASK` implicitly use VCC as a
  scalar source, so these opcodes cannot use an additional SGPR or literal
  source at `cdna3/README.md:1231` through `:1233`.
- The detailed definitions clarify the VOP3 form: `V_CNDMASK_B32` may take the
  VCC source from scalar GPR `S2`, and `V_ADDC_CO_U32`/`V_SUBB_CO_U32` may take
  the VCC source from the SGPR pair at `S2.u`, at `cdna3/README.md:6727` through
  `:6738`, `:7072` through `:7085`, `:13652` through `:13663`, and `:13960`
  through `:13973`. This makes `S2` the scalar source for VOP3, leaving no
  budget for another scalar or literal in `S0`/`S1`.

XML evidence:

- The XML records per-operand source classes, but not a cross-operand scalar
  read-count rule. Representative VOP3 `V_ADD_F32` encodes `SRC0` as
  `OPR_SRC_NOLIT` and `SRC1` as `OPR_SRC_SIMPLE` at
  `amdgpu_isa_cdna3.xml:51731` through `:51753`.
- `OPR_SRC_NOLIT` includes scalar-register, M0, and scalar-source subtypes at
  `amdgpu_isa_cdna3.xml:101454` through `:101465`, and `OPR_SRC_SIMPLE` includes
  the corresponding scalar-source subtypes at `:103880` through `:103890`.
- Representative carry op `V_ADDC_CO_U32` records broad `SRC0`/`SRC1` classes
  plus an `OPR_SREG` carry-in source at `amdgpu_isa_cdna3.xml:55370` through
  `:55400`; the entry does not encode that the explicit `S2` scalar source
  consumes the scalar-source budget and therefore restricts `S0`/`S1`.

Impact:

An XML-only validator or generator can infer operand-local selector ranges, but
cannot tell which combinations of VALU sources exceed CDNA3's scalar read-port
contract or conflict with the VCC/S2 source used by carry and conditional-mask
forms.

### CDNA3-XML-046: VOP3 output-modifier MODE and denormal rules are prose-only

Manual evidence:

- Chapter 6.2.2 says VOP3 floating-point output modifiers multiply the result by
  0.5/1/2/4 and optionally clamp to `[0, +1]`, but output modifiers are ignored
  when output denormals are enabled or when `MODE.IEEE` is 1; when output
  denormals are disabled, denormals are flushed to zero and `-0` is flushed to
  `+0` at `cdna3/README.md:1261` through `:1263`.

XML evidence:

- Generic `ENC_VOP3` exposes `CLAMP` as "clamp output to [0.0, 1.0]" and `OMOD`
  as "Output modifier for instruction" at
  `amdgpu_isa_cdna3.xml:2580` through `:2612`.
- Those field descriptions do not encode the `MODE.FP_DENORM` or `MODE.IEEE`
  conditions, the denormal flush rule, or the `-0` to `+0` rule. The raw MODE
  register layout itself is already tracked as prose-only state in
  `CDNA3-XML-037`.

Impact:

XML-derived execution code can see that VOP3 carries `OMOD` and `CLAMP`, but
cannot derive when CDNA3 requires those fields to be ignored or when extra
floating-point result flushing is required.

### CDNA3-XML-047: VALU FP round/denorm modes and V_DOT2 override are prose-only

Manual evidence:

- Chapter 6.4 says the shader program controls floating-point rounding and
  denormal input/result handling through MODE fields set by `S_SETREG` at
  `cdna3/README.md:1435` through `:1438`.
- The same section says floating-point `V_DOT2` instructions do not support
  denormal or rounding modes and flush input and output denormals at
  `cdna3/README.md:1439`.
- Table 26 defines `FP_ROUND[3:0]` as separate single-precision and
  double/half-precision round modes, and `FP_DENORM[7:4]` as separate
  single-precision and double/half-precision denormal modes at
  `cdna3/README.md:1441` through `:1446`.

XML evidence:

- The CDNA3 XML has representative `V_DOT2C_F32_F16` and `V_DOT2_F32_F16`
  opcode entries with operand formats and basic descriptions at
  `amdgpu_isa_cdna3.xml:59038` through `:59064` and `:65654` through `:65687`,
  but those entries do not encode the manual's denormal-flush override.
- The audited XML has no `FP_ROUND` or `FP_DENORM` field records; MODE state is
  not recoverable beyond the separate raw-register gaps already recorded in
  `CDNA3-XML-037`.

Impact:

An XML-only execution generator cannot know which MODE bits control ordinary
VALU rounding and denormal behavior, and cannot distinguish ordinary FP VALU
mode handling from the special `V_DOT2` flush-in/flush-out rule.

### CDNA3-XML-048: ALU clamp-bit semantic overloads are prose-only

Manual evidence:

- Chapter 6.5 says the clamp bit has different meanings by opcode family:
  `V_CMP` clamp requests a signaling compare on FP exceptions, integer
  operations saturate to the largest/smallest representable integer, and
  floating-point operations clamp to `[0.0, 1.0]` at `cdna3/README.md:1448`
  through `:1450`.
- Chapter 12.9.1 repeats the VOPC/VOP3A compare-specific rule: when `CLAMP` is
  set, the compare signals an exception if either input is NaN; when clear, NaN
  does not signal an exception, at `cdna3/README.md:11084` through `:11105`.

XML evidence:

- Generic XML `CLAMP` fields describe only a floating clamp or a generic output
  clamp, for example VOP3P/VOP3 fields at `amdgpu_isa_cdna3.xml:1990` through
  `:1991`, `:2580` through `:2581`, and `:7003` through `:7004`; VOP1/VOP2
  promoted forms only say "If true, clamp output" at `:5626` through `:5627`
  and `:6519` through `:6520`.
- Representative `V_CMP_F_F32` and `V_CMPX_F_F32` XML records describe the
  compare result and operands at `amdgpu_isa_cdna3.xml:73236` through `:73263`
  and `:75060` through `:75091`, but they do not carry the signaling-compare
  meaning of the clamp bit.
- Integer VOP3 entries such as `V_ADD_U32` and `V_ADD_I32` expose the VOP3
  encoding and integer operand formats at `amdgpu_isa_cdna3.xml:58719` through
  `:58742` and `:64306` through `:64333`, but the opcode records do not
  distinguish signed and unsigned saturation semantics for `CLAMP`.

Impact:

The XML can identify the raw clamp bit, but not the opcode-family-dependent
contract needed to implement compare exception signaling or integer saturation.

### CDNA3-XML-049: VGPR indexing bit layout and operand-role mapping are prose-only

Manual evidence:

- Chapter 6.6 says VGPR indexing uses M0 as an index for selected VALU VGPR
  sources or destinations at `cdna3/README.md:1452` through `:1455`.
- Table 27 defines the `S_SET_GPR_IDX_*` operations, including
  `mode.gpr_idx_en`, `M0[7:0]`, and the mode field in `M0[15:12]`, at
  `cdna3/README.md:1460` through `:1465`.
- The prose then defines `M0[15]` as destination enable, `M0[14]` as source-2
  enable, `M0[13]` as source-1 enable, `M0[12]` as source-0 enable, restricts
  indexing to VGPR operands, and says out-of-range indexed VGPRs are illegal at
  `cdna3/README.md:1469` through `:1479`.
- Section 6.6.2 gives per-instruction role remapping for unusual source and
  destination use, including readlane, writelane, MAC/MAD, shifts,
  `v_cvt_pkaccum`, and SDWA preserve forms at `cdna3/README.md:1481` through
  `:1495`.

XML evidence:

- The XML records the `S_SET_GPR_IDX_IDX`, `S_SET_GPR_IDX_ON`,
  `S_SET_GPR_IDX_OFF`, and `S_SET_GPR_IDX_MODE` opcodes with generic
  descriptions and implicit M0 operands at `amdgpu_isa_cdna3.xml:31502`
  through `:31525`, `:40878` through `:40907`, and `:42864` through `:42908`.
- Those opcode records do not encode the `MODE.gpr_idx_en` side effect,
  `M0[7:0]` versus `M0[15:12]` layout, source/destination enable-bit meanings,
  the "VGPR only" restriction, out-of-range illegality, or the unusual
  instruction role table.

Impact:

XML-derived operand readers can know that these control instructions touch M0,
but cannot derive which VALU operand slots should be indexed or which M0 bits
make a given indexed VGPR access legal.

### CDNA3-XML-050: Packed FP8/BF8-to-F32 conversion destination alignment is prose-only

Manual evidence:

- Table 31 says `CVT_PK_F32_FP8` and `CVT_PK_F32_BF8` write `dst,dst+1` and
  that the destination must be even at `cdna3/README.md:2002` through `:2003`.

XML evidence:

- The XML records `V_CVT_PK_F32_FP8` and `V_CVT_PK_F32_BF8` with 64-bit
  `VDST` operands at `amdgpu_isa_cdna3.xml:51238` through `:51405`, but the
  operand records use the generic `OPR_VGPR` type and do not encode the
  instruction-specific even-base legality rule.

Impact:

An XML consumer can infer that these converts span two consecutive VGPRs, but
cannot reject odd destination bases from XML metadata alone.

### CDNA3-XML-051: F32-to-FP8/BF8 conversion modifiers, `OP_SEL`, byte order, and seed bits are under-described

Manual evidence:

- Table 31 says `CVT_PK_FP8_F32` and `CVT_PK_BF8_F32` use `OP_SEL[3]`, ignore
  `CLAMP`/`OMOD`, and support `NEG`/`ABS` at `cdna3/README.md:1996` through
  `:1998`.
- The same table says `CVT_SR_FP8_F32` and `CVT_SR_BF8_F32` use
  `OP_SEL[3:2]`, ignore `CLAMP`/`OMOD`, and support `NEG`/`ABS` at
  `cdna3/README.md:1999` through `:2001`.
- The stochastic-rounding prose says `OP_SEL[3:2]` selects the destination byte
  and preserves the remaining 24 bits at `cdna3/README.md:2014` through
  `:2032`.
- The detailed packed conversion definitions write `{ f32_to_fp8(S1),
  f32_to_fp8(S0) }` or `{ f32_to_bf8(S1), f32_to_bf8(S0) }` into the selected
  half, so the result-byte order is part of the instruction contract at
  `cdna3/README.md:15916` through `:15956`.
- The detailed stochastic conversion definitions use `S1[31:12]` for FP8 and
  `S1[31:11]` for BF8 seed input to the mantissa add at
  `cdna3/README.md:15958` through `:16010`.

XML evidence:

- Generic `ENC_VOP3` exposes `ABS`, `CLAMP`, `NEG`, `OMOD`, and `OP_SEL` at
  `amdgpu_isa_cdna3.xml:2570` through `:2631`, but the generic `OP_SEL`
  description is the ordinary 16-bit source/destination selector.
- The four F32-to-FP8/BF8 conversion entries describe packed or stochastic
  conversion and mention `OPSEL` at `amdgpu_isa_cdna3.xml:64552` through
  `:64692`, but they do not encode the manual's per-instruction modifier
  contract, the `OP_SEL[3]` versus `OP_SEL[3:2]` split, the packed result-byte
  order, the stochastic seed-bit ranges, or the byte-preserve writeback
  formula.

Impact:

An XML-derived implementation can see the raw VOP3 fields, but cannot derive
which modifier fields are semantic, which must be ignored, or how the
packed and stochastic forms repurpose source order and `OP_SEL` to merge
sub-dword results into the destination.

### CDNA3-XML-052: FP8/BF8 widening-convert SDWA ignored-field contract is prose-only

Manual evidence:

- Table 31 says `CVT_F32_FP8`, `CVT_F32_BF8`, `CVT_PK_F32_FP8`, and
  `CVT_PK_F32_BF8` use SDWA source selection and ignore `ABS`, `NEG`, and
  `SEXT` at `cdna3/README.md:2002` through `:2005`.
- The section 7.2 prose says VOP1 8-bit-format converts use only the SDWA
  `SRC0` VGPR and `SRC0_SELECT`, and that the other SDWA fields are ignored at
  `cdna3/README.md:2011` through `:2012`.
- The detailed instruction definitions repeat the byte-only and word-only
  `SRC0_SEL` rules for the four widening converts at `cdna3/README.md:8766`
  through `:8839`.

XML evidence:

- The SDWA encoding exposes generic `SRC0_SEL`, source modifier, destination
  select, destination-unused, and clamp fields at
  `amdgpu_isa_cdna3.xml:5716` through `:5798`.
- The four widening-convert entries include SDWA alternatives at
  `amdgpu_isa_cdna3.xml:51016` through `:51405`, but those opcode records do
  not encode that only `SRC0_SEL` is semantic for these instructions and that
  the other SDWA fields must be ignored.

Impact:

An XML-only decoder or executor can discover that the SDWA encoding exists, but
cannot tell that these conversion opcodes must not apply the generic SDWA
source modifiers, destination merge controls, or clamp behavior.

### CDNA3-XML-053: Program organization, dispatch, GWS, and device-memory consistency are prose-only

Manual evidence:

- Chapter 1 describes the CDNA command processor, memory controller, host/kernel
  split, cache invalidation/flush commands, DMA-style memory-controller
  behavior, hardware interrupts, floating-point exception detection, automatic
  instruction fetch, and latency hiding at `cdna3/README.md:249` through
  `:277`.
- Section 1.1 defines dispatch, workgroup, 64-work-item wavefronts, work-items,
  literal constants, SALU, VALU, and microcode-format terminology at
  `cdna3/README.md:279` through `:295`.
- Chapter 2 says kernels are grouped into 64-work-item wavefronts, control flow
  is handled by SALU instructions, vector ALU and vector-memory work is gated
  by `EXEC`, and vector compare/carry-out results return bit-per-work-item
  masks to SGPRs at `cdna3/README.md:299` through `:317`.
- Section 2.1 says dispatches cover 1D/2D/3D grids, generate wavefronts, and
  initialize each work-item with a unique grid index at `cdna3/README.md:319`
  through `:321`.
- Sections 2.2.1 through 2.3 describe the 64 KiB / 32-bank / 32-atomic-unit LDS
  topology, the GWS unit, L2/L1 hierarchy, cache-less loads, load-clause overlap
  behavior, atomic return acknowledgments, relaxed consistency, and
  per-PE/per-channel scatter-write ordering at `cdna3/README.md:333` through
  `:354`.

XML evidence:

- The top-level XML architecture record only carries the architecture name and
  numeric ID at `amdgpu_isa_cdna3.xml:1` through `:13`.
- XML has useful local descriptions, such as VOPC compare `EXEC` wording at
  `amdgpu_isa_cdna3.xml:1608`, DS/GDS/LDS field text at `:2843` through
  `:2885`, the MUBUF `LDS` field at `:3062` through `:3064`, `OPR_DSMEM` at
  `:93587` through `:93593`, and functional-group descriptions at
  `:122906` through `:122921`.
- Those records still do not provide global metadata for wavefront size,
  dispatch-grid dimensionality, SALU/VALU control roles, EXEC-mask scope,
  LDS/GWS topology, command-processor host interaction, floating-exception
  recording, or the Section 2.3 memory consistency/acknowledgment/order model.
- Existing narrower entries cover several downstream pieces:
  `CDNA3-XML-028` for LDS bank/clamp rules, `CDNA3-XML-031` for GWS execution
  restrictions, `CDNA3-XML-035` through `CDNA3-XML-039` for wave/kernel state
  and launch initialization, `CDNA3-XML-042` for barrier behavior, and
  `CDNA3-XML-043` for wait-counter semantics.

Impact:

XML consumers can enumerate instructions, functional groups, and some operands,
but still need manual prose for processor-organization assumptions and the
top-level memory model that determine how decoded instructions execute together.

### CDNA3-XML-054: Floating-memory atomic numeric rules are prose-only

Manual evidence:

- Chapter 9.2 says floating memory atomics execute in LDS and L2, can be issued
  as LDS, Buffer, Flat, Global, and Scratch instructions, and that the chapter
  defines rounding, denormal, and NaN behavior at `cdna3/README.md:2927`
  through `:2931`.
- Float atomic ADD opcodes use round-to-nearest-even, and the denormal table
  assigns LDS and L2 behavior for packed F16/BF16 add, F32 add, F32/F64
  min/max, compare-store, and F64 add at `cdna3/README.md:2935` through
  `:2970`.
- Chapter 9.2.3 defines SNaN quieting, NaN propagation/selection, signed-zero
  ordering, compare-store's `+0 == -0` equality rule, and float-add special
  cases at `cdna3/README.md:2972` through `:3017`.

XML evidence:

- The XML exposes LDS float atomic opcode shells and operand formats, including
  F32 compare/min/max/add and packed F16/BF16 add at
  `amdgpu_isa_cdna3.xml:8762` through `:9012`, with F64 forms beginning at
  `amdgpu_isa_cdna3.xml:11547`, `amdgpu_isa_cdna3.xml:11599`,
  `amdgpu_isa_cdna3.xml:11645`, and `amdgpu_isa_cdna3.xml:12019`.
- Buffer float atomic entries expose the CDNA3 MUBUF F32 add, packed F16 add,
  and F64 add/min/max opcode records at `amdgpu_isa_cdna3.xml:24295` through
  `:24527`.
- Flat and global float atomic entries expose the corresponding opcode records,
  including packed BF16 flat/global forms, at `amdgpu_isa_cdna3.xml:15577`
  through `:15838` and `:18523` through `:18789`.
- These records state the broad operation, operands, and functional group, but
  do not encode fixed add rounding, per-issuer denormal policies,
  SNaN-to-QNaN conversion, NaN payload selection, signed-zero ordering, or
  compare-store's floating equality rule.

Impact:

XML-only consumers can enumerate the CDNA3 float atomic opcodes and operand
widths, but cannot derive the numeric contract needed for faithful emulation,
validation, or edge-case test generation.

### CDNA3-XML-055: SOP2 detailed formulas and edge examples are not machine-readable

Manual evidence:

- Chapter 12.1 gives the detailed SOP2 definitions from `S_ADD_U32` through
  `S_PACK_HH_B32_B16` at `cdna3/README.md:3363` through `:3920`.
- `S_BFM_B32/B64` define exact width and offset bit slices from `S0` and `S1`
  at `cdna3/README.md:3717` through `:3730`.
- `S_BFE_U32/I32/U64/I64` define `S1[4:0]` or `S1[5:0]` as the offset and
  `S1[22:16]` as the field width at `cdna3/README.md:3743` through `:3779`.
- `S_ABSDIFF_I32` defines a 32-bit signed subtract followed by
  negate-if-negative, and gives overflow examples for `0x80000000` inputs at
  `cdna3/README.md:3809` through `:3832`.
- `S_LSHL{1,2,3,4}_ADD_U32` define shifted-add carry-out formulas at
  `cdna3/README.md:3852` through `:3895`, and the scalar pack rows define exact
  low/high half selections at `cdna3/README.md:3898` through `:3920`.

XML evidence:

- The XML has SOP2 opcode entries from `S_ADD_U32` through
  `S_PACK_HH_B32_B16` at `amdgpu_isa_cdna3.xml:31954` through `:38985`.
- The XML descriptions for `S_BFM_B32`, `S_BFE_U32`, `S_ABSDIFF_I32`, and the
  pack rows are short summaries at `amdgpu_isa_cdna3.xml:36618` through
  `:36619`, `:36966` through `:36967`, `:37622` through `:37623`, and `:38646`
  through `:38879`.
- Those entries expose operands, sizes, literal alternatives, and implicit SCC
  where applicable, but do not structure the bit-slice formulas, shifted-add
  carry equations, `S_ABSDIFF_I32` overflow examples, or scalar-pack source-half
  and destination-half placement rules.

Impact:

XML-only generators can build the CDNA3 SOP2 opcode inventory and operands, but
need manual-specific overrides or tests for exact bitfield, absolute-difference,
shifted-add, and pack semantics.

### CDNA3-XML-056: SOP2 opcode 43 `S_RFE_RESTORE_B64` is XML-only

Manual evidence:

- Chapter 12.1 lists `S_ABSDIFF_I32` as opcode 42 and then jumps to
  `S_MUL_HI_U32` as opcode 44, leaving no opcode 43 definition at
  `cdna3/README.md:3809` through `:3836`.
- Chapter 4.1 lists `S_RFE` generically at `cdna3/README.md:712` through
  `:713`, and the microcode opcode table lists `S_RFE_B64` at
  `cdna3/README.md:22903`. No manual match for `S_RFE_RESTORE_B64` was found in
  the CDNA3 markdown manual.

XML evidence:

- The CDNA3 XML includes `S_RFE_RESTORE_B64` as an `ENC_SOP2` opcode 43 branch
  instruction with 64-bit `SSRC0`, 32-bit `SSRC1`, implicit PC output, literal
  alternatives, and TRAP functional group at `amdgpu_isa_cdna3.xml:37758`
  through `:37860`.

Impact:

The checked-in XML exposes a CDNA3 SOP2 instruction row that is absent from the
CDNA3 ISA manual prose/table. Generators and tests that treat XML as complete
can surface `S_RFE_RESTORE_B64` without a matching CDNA3 manual contract.

### CDNA3-XML-057: SOP1 relative-addressing rules and edge examples are prose-only

Manual evidence:

- Chapter 12.3 gives detailed SOP1 formulas and examples for WQM, bit count,
  first-bit scans, leading-bit scans, `S_ABS_I32`, quad-mask, and bit-replicate
  rows at `cdna3/README.md:4235` through `:5030`.
- `S_MOVRELS_B32/B64` and `S_MOVRELD_B32/B64` define the effective scalar
  register address as the instruction source/destination SGPR address plus
  `M0.u`, and the 64-bit forms require even `M0.u` plus an even source or
  destination address at `cdna3/README.md:4808` through `:4877`.
- `S_CBRANCH_JOIN` gives branch-stack/CSP pseudocode at
  `cdna3/README.md:4879` through `:4896`; the broader fork/join XML gap is
  tracked separately in `CDNA3-XML-041`.

XML evidence:

- The XML has matching `ENC_SOP1` opcode entries for the Chapter 12.3 rows at
  `amdgpu_isa_cdna3.xml:28543` through `:31945`, including implicit SCC, EXEC,
  PC, and M0 operands where applicable.
- The bit-count, bit-scan, `S_ABS_I32`, and bit-replicate entries use short
  descriptions at `amdgpu_isa_cdna3.xml:29143`, `:29399`, `:29615`, `:31438`,
  and `:31900`, but do not structure the manual's edge examples or exact
  formulas.
- The `S_MOVRELS_*` and `S_MOVRELD_*` entries expose M0 as an implicit operand
  at `amdgpu_isa_cdna3.xml:31120` through `:31363`, but do not encode the
  address formula `addr += M0.u` or the even-M0/even-address requirement for
  64-bit forms.

Impact:

XML-only generators can build the SOP1 opcode inventory and basic operands, but
need manual-derived rules for relative SGPR addressing, 64-bit relative-move
alignment, branch-stack join behavior, and edge-sensitive unary tests.

### CDNA3-XML-058: SOPC bit-index, VSKIP, and raw-mode details are not represented precisely

Manual/oracle evidence:

- Chapter 12.4 defines `S_BITCMP0_B32`/`S_BITCMP1_B32` as indexing with
  `S1.u32[4:0]`, and the B64 forms as indexing with `S1.u32[5:0]`, at
  `cdna3/README.md:5154` through `:5186`.
- `S_SETVSKIP` says `VSKIP = S0.u32[S1.u32[4:0]]`, lists the vector, memory,
  image, LDS, and flat instruction classes skipped by VSKIP, and states that
  VSKIPped memory instructions do not manipulate wait counters at
  `cdna3/README.md:5188` through `:5205`.
- `S_SET_GPR_IDX_ON` says source 0 supplies the index, the raw bits of the
  `SRC1` field set the enable bits, `M0[15:12] = SRC1.u32[3:0]`, and this is
  direct raw-field content at `cdna3/README.md:5207` through `:5223`.
- As a cross-check, `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx942` accepts
  `s_set_gpr_idx_on s0, 15` and `s_set_gpr_idx_on 0x12345678, 15`, but rejects
  `s_set_gpr_idx_on s0, 16` and `s_set_gpr_idx_on s0, 0x12345678`.

XML evidence:

- The `S_BITCMP*` XML entries describe bit extraction at
  `amdgpu_isa_cdna3.xml:40338` through `:40777`, but do not encode the
  instruction-specific 5-bit versus 6-bit index masks.
- XML describes `S_SETVSKIP` only as enabling or disabling VSKIP mode and gives
  it two ordinary `OPR_SSRC` operands plus generic SOPC literal variants at
  `amdgpu_isa_cdna3.xml:40786` through `:40865`; it does not encode the bit
  index mask, VSKIP state output, skipped instruction classes, or wait-counter
  policy.
- XML gives `S_SET_GPR_IDX_ON` a default `OPR_SIMM4` second operand and implicit
  M0 input/output operands at `amdgpu_isa_cdna3.xml:40878` through `:40907`, but
  the generic literal variants also expose `SSRC1` as `SIMM32` with `OPR_SIMM4`
  at `:40939` through `:40995`. The broader GPR-index mode layout and remap
  omissions are tracked in `CDNA3-XML-049`.

Impact:

XML-only generators can decode the SOPC rows, but cannot derive the bit-index
masks, the full VSKIP state/wait-counter contract, or the special raw-field
treatment for `S_SET_GPR_IDX_ON`. Generic literal expansion can assign an
extension word to a field the manual and LLVM assembler define as direct 4-bit
instruction bits.

### CDNA3-XML-059: SOPP opcode table omits XML-only ordered-PS and co-execution entries

Manual evidence:

- Chapter 12.5 lists detailed SOPP instruction definitions from `S_NOP`
  opcode 0 through `S_SET_GPR_IDX_MODE` opcode 29 at `cdna3/README.md:5243`
  through `:5601`; a direct search of the CDNA3 manual finds no
  `S_ENDPGM_ORDERED_PS_DONE` or `S_SET_VALU_COEXEC_MODE` entry.
- Chapter 13.1.5 Table 74 lists SOPP opcodes 0 through 29, then transitions to
  the SMEM format at `cdna3/README.md:23020` through `:23042`.

XML evidence:

- XML records `S_ENDPGM_ORDERED_PS_DONE` as `ENC_SOPP` opcode 30 with
  program-terminator metadata at `amdgpu_isa_cdna3.xml:42925` through
  `:42936`.
- XML records `S_SET_VALU_COEXEC_MODE` as `ENC_SOPP` opcode 31 with a
  `SIMM16` operand and vector-ALU co-execution-mode description at
  `amdgpu_isa_cdna3.xml:42947` through `:42966`.

Impact:

Manual-derived CDNA3 SOPP inventories stop at opcode 29 and can classify
opcodes 30 and 31 as reserved or absent, while XML-derived decoders and
generated rocjitsu expose both encodings.

### CDNA3-XML-060: `S_ATC_PROBE*` SMEM opcodes are XML-only for CDNA3

Manual evidence:

- Chapter 12.6 lists detailed SMEM instruction definitions from
  `S_LOAD_DWORD` opcode 0 through `S_ATOMIC_DEC_X2` opcode 172, but skips
  opcode 38 and opcode 39 between `S_MEMREALTIME` and `S_DCACHE_DISCARD` at
  `cdna3/README.md:5618` through `:6721`.
- Chapter 13.2 Table 75 lists SMEM opcodes 0-37, 40-41, 64-76, 96-108,
  128-140, and 160-172 at `cdna3/README.md:23055` through `:23115`, but has no
  entries for opcodes 38 or 39.
- Searching the CDNA3 manual prose finds no `S_ATC_PROBE` or
  `S_ATC_PROBE_BUFFER` instruction definition or semantic description.

XML evidence:

- The CDNA3 XML defines `S_ATC_PROBE` at opcode 38 and
  `S_ATC_PROBE_BUFFER` at opcode 39, both as `ENC_SMEM` instructions that probe
  or prefetch an address into the scalar data cache at
  `amdgpu_isa_cdna3.xml:26207` through `:26289`.
- A direct inventory comparison found 82 manual SMEM instruction definitions and
  84 XML `ENC_SMEM` records; the only XML-only opcodes were 38 and 39.

Impact:

Manual-derived validators will reject or fail to classify these two XML-defined
SMEM opcodes, while XML-derived decoders expose a CDNA3 instruction surface that
has no corroborating behavior in the CDNA3 manual.

### CDNA3-XML-061: VOP2 FP min/max edge semantics are prose-only

Manual evidence:

- `V_MIN_F32` and `V_MAX_F32` define explicit SNaN quieting, NaN operand
  selection, signed-zero selection, and IEEE/non-IEEE equality behavior at
  `cdna3/README.md:6845` through `:6895`.
- `V_MAX_F16` and `V_MIN_F16` repeat the same edge-contract family for 16-bit
  floating point at `cdna3/README.md:7301` through `:7357`.

XML evidence:

- The checked-in XML rows for `V_MIN_F32` and `V_MAX_F32` record operands,
  encodings, and short descriptions at `amdgpu_isa_cdna3.xml:52871` through
  `:53144`, but not the NaN, SNaN, signed-zero, or MODE/IEEE tie rules.
- The XML rows for `V_MAX_F16` and `V_MIN_F16` likewise record operand and
  encoding alternatives at `amdgpu_isa_cdna3.xml:57628` through `:57900`, but
  not the detailed edge semantics.

Impact:

XML-only consumers can identify the min/max opcodes and operand shapes, but
cannot derive the ISA's ordered operand selection, SNaN quieting, signed-zero
tie behavior, or IEEE-mode predicate differences.

### CDNA3-XML-062: F16 VOP2 destination-half behavior is prose-only

Manual evidence:

- `V_MAC_F16` says `OPSEL[3]` selects whether the old low or high destination
  half participates in the multiply-add and which half of `VDST` is updated at
  `cdna3/README.md:7177` through `:7192`.
- `V_MADMK_F16` and `V_MADAK_F16` define literal F16 multiply-add forms that
  write the low half and zero the high half at `cdna3/README.md:7194` through
  `:7221`.

XML evidence:

- The `V_MAC_F16` XML row includes operands and a short note about non-standard
  `OPSEL` handling at `amdgpu_isa_cdna3.xml:56369` through `:56476`, but does
  not encode the old-destination half selection or read-modify-write contract.
- The `V_MADMK_F16` and `V_MADAK_F16` XML rows at
  `amdgpu_isa_cdna3.xml:56485` through `:56632` expose the literal-only forms
  and descriptions, but the structured metadata does not distinguish the
  zero-high write behavior from true16 preserved-half writeback.

Impact:

Generators need manual prose to distinguish true16 read-modify-write behavior
from the literal forms that explicitly clear the high half of the destination.
The XML operand records alone are not enough to decide which destination half is
read, preserved, or zeroed.

### CDNA3-XML-063: VOP2 packed-accumulate formulas are prose-only

Manual evidence:

- `V_DOT2C_F32_F16`, `V_DOT2C_I32_I16`, `V_DOT4C_I32_I8`,
  `V_DOT8C_I32_I4`, `V_FMAC_F32`, and `V_PK_FMAC_F16` spell out accumulator
  dataflow, lane extraction, and packed-component formulas at
  `cdna3/README.md:7441` through `:7513`.

XML evidence:

- The corresponding XML rows record the operand classes, VOP2 literal/DPP
  alternatives, and VOP3 promoted alternatives at
  `amdgpu_isa_cdna3.xml:59038` through `:59695`, but do not encode the per-lane
  dot-product extractions, accumulator use, or packed halfword FMA formulas as
  structured semantics.

Impact:

An XML-derived executor can see that these are accumulator-style VOP2 rows, but
must still import manual prose or separate semantic knowledge to implement the
actual dot/FMAs and packed-lane accumulation.

### CDNA3-XML-064: VOP2 no-modifier exclusions are prose-only

Manual evidence:

- `V_AND_B32`, `V_OR_B32`, `V_XOR_B32`, and `V_XNOR_B32` explicitly say that
  input modifiers and output modifiers are not supported at
  `cdna3/README.md:6957` through `:6991` and `:7515` through `:7525`.
- `V_FMAMK_F32`, `V_FMAAK_F32`, `V_MADMK_F16`, and `V_MADAK_F16` are
  literal-only VOP2 rows that say they do not support VOP3, input modifiers, or
  output modifiers at `cdna3/README.md:6993` through `:7017` and
  `:7194` through `:7221`.

XML evidence:

- XML records the literal-only F32 and F16 rows without promoted VOP3
  alternatives at `amdgpu_isa_cdna3.xml:54563` through `:54718` and
  `:56485` through `:56632`.
- The XML still represents most VOP2 instructions through generic VOP2/DPP/SDWA
  and promoted VOP3 row shapes rather than carrying an instruction-specific
  structured "no input/output modifiers" legality flag.

Impact:

The absence of VOP3 rows is visible for the literal-only forms, but
instruction-specific modifier legality still comes from prose. XML-only
validators need additional rules to reject modifier-shaped encodings for the
affected bitwise and literal-only rows.

### CDNA3-XML-065: VOP1 XML contains opcode 55 absent from detailed manual tables

Manual evidence:

- The Chapter 12.8 VOP1 detailed definitions skip from `V_CLREXCP` opcode 53
  to `V_MOV_B64` opcode 56 at `cdna3/README.md:8374` through `:8378`.
- The Chapter 13.3.1 VOP1 opcode table repeats the same hole by jumping from
  opcode 53 to opcode 56 at `cdna3/README.md:23289` through `:23294`.

XML evidence:

- The checked-in XML defines `V_SCREEN_PARTITION_4SE_B32` as VOP1 opcode 55,
  with default, literal, DPP, SDWA, and promoted VOP3 encodings at
  `amdgpu_isa_cdna3.xml:48291` through `:48400`.

Impact:

Manual-only inventory consumers see opcode 55 as a hole, while XML-derived
decoders expose an extra instruction. This needs an architecture-specific
source-of-truth decision before opcode 55 can be treated as a documented CDNA3
contract.

### CDNA3-XML-066: VOP1 promoted-opcode tables conflict between manual sections

Manual evidence:

- Chapter 12.8.1 says VOP1 instructions using VOP3 encoding use the base
  opcode plus `0x140` at `cdna3/README.md:8843` through `:8845`.
- Chapter 13.3.4 Table 84 instead lists promoted VOP1 rows starting at opcode
  384, including `V_NOP` 384, `V_MOV_B32` 385, `V_READFIRSTLANE_B32` 386,
  `V_SAT_PK_U8_I16` 463, and `V_SWAP_B32` 465 at
  `cdna3/README.md:23605` through `:23695`.

XML evidence:

- The checked-in XML follows the Chapter 12.8.1 `+0x140` rule, for example
  `V_MOV_B32` opcode 1 promotes to VOP3 opcode 321 at
  `amdgpu_isa_cdna3.xml:43022` through `:43120`, and
  `V_SAT_PK_U8_I16` opcode 79 promotes to opcode 399 at
  `amdgpu_isa_cdna3.xml:50740` through `:50838`.

Impact:

The prose rule and XML agree with observed assembler behavior, but the Chapter
13 table can mislead manual-derived opcode-table consumers into generating or
rejecting the wrong promoted VOP1 opcodes.

### CDNA3-XML-067: `V_READFIRSTLANE_B32` EXEC override and no-modifier rule are prose-only

Manual evidence:

- `V_READFIRSTLANE_B32` says an all-disabled `EXEC` mask selects lane 0,
  otherwise the lowest active lane is used; it also says the VGPR read
  overrides `EXEC`, and input/output modifiers are not supported at
  `cdna3/README.md:7574` through `:7590`.

XML evidence:

- The checked-in XML records the scalar destination, vector/LDS source operand,
  literal alternative, and promoted VOP3 form at
  `amdgpu_isa_cdna3.xml:43133` through `:43182`, but it does not carry the
  all-disabled lane rule, the `EXEC` override, or the no-modifier contract as
  structured metadata.

Impact:

An XML-derived executor can infer the operands and encodings, but still needs
manual prose to implement the lane-selection edge case and to reject
modifier-shaped forms.

### CDNA3-XML-068: VOP1 move, conversion, lookup, and exception edge semantics are prose-only

Manual evidence:

- `V_MOV_B32` and `V_MOV_B64` allow floating input modifiers only when the
  moved value is treated as F32 or F64 at `cdna3/README.md:7549` through
  `:7572` and `:8378` through `:8388`.
- VOP1 conversion rows spell out saturation, NaN-to-zero, `CLAMP`-controlled
  inexact exceptions, rounding/denorm behavior, and source/destination format
  details, for example `V_CVT_I32_F64` at `cdna3/README.md:7594` through
  `:7606`, F32 integer conversions at `:7644` through `:7670`, and F16/F64
  conversions at `:8390` through `:8440`.
- `V_CVT_OFF_F32_I4` includes a 16-entry lookup table at
  `cdna3/README.md:7722` through `:7767`; `V_FRACT_F32` and `V_FRACT_F64`
  define DX-style negative behavior, rounding-mode use, and max-below-one
  clamps at `:7896` through `:7912` and `:8324` through `:8339`;
  `V_RCP_IFLAG_F32` defines integer divide-by-zero exception behavior at
  `:8025` through `:8046`; FREXP rows define NaN/Inf/zero/subnormal edge
  behavior at `:8288` through `:8369` and `:8560` through `:8589`; and
  `V_CLREXCP` clears VALU exception state at `:8374`.

XML evidence:

- The corresponding checked-in XML rows record instruction names, operands,
  encodings, and short descriptions, for example `V_MOV_B32` at
  `amdgpu_isa_cdna3.xml:43022` through `:43120`, `V_CVT_I32_F64` at
  `:43206`, `V_CVT_OFF_F32_I4` at `:44278`, `V_FRACT_F32` at `:45531`,
  `V_RCP_IFLAG_F32` at `:46422`, `V_FREXP_EXP_I32_F64` at `:47759`,
  `V_CLREXCP` at `:48257`, and `V_MOV_B64` at `:48402`, but do not encode
  these edge formulas, lookup-table contents, exception controls, or
  conditional modifier rules as structured semantics.

Impact:

XML-only semantic generation can recover the broad dataflow but not the edge
contracts that define architectural behavior for conversion, reciprocal,
fractional, FREXP, MOV-modifier, and exception-clear instructions.

### CDNA3-XML-069: `V_SAT_PK_U8_I16` signed packed dataflow conflicts with checked-in XML

Manual evidence:

- `V_SAT_PK_U8_I16` says two signed 16-bit integer inputs are saturated over an
  unsigned 8-bit range and packed, with formulas using `S0[15:0].i16` and
  `S0[31:16].i16`, at `cdna3/README.md:8725` through `:8746`.

XML evidence:

- The checked-in XML description says "two 16-bit unsigned integer inputs" and
  uses generic `FMT_NUM_B16`/`FMT_NUM_B32` metadata rather than signed packed
  I16 input and packed U8 output formats at `amdgpu_isa_cdna3.xml:50740`
  through `:50838`.

Impact:

XML-derived semantics can infer the wrong signedness and packing contract for
negative I16 inputs unless the manual prose overrides the checked-in XML row.

### CDNA3-XML-070: Chapter 12.16 F64 VOP1 DPP exclusions leak into XML encodings

Manual evidence:

- Chapter 12.16.1 says DPP cannot be used with `V_READFIRSTLANE_B32`, the F64
  VOP1 conversion/unary block from `V_CVT_I32_F64` through `V_FRACT_F64`,
  `V_CLREXCP`, and `V_SWAP_B32` at `cdna3/README.md:22583` through `:22616`.

XML evidence:

- The checked-in XML omits DPP for `V_CLREXCP` and `V_SWAP_B32`, but still
  exposes DPP alternatives for several F64 VOP1 rows in that prohibited range,
  including `V_CVT_I32_F64` at `amdgpu_isa_cdna3.xml:43206` through `:43248`,
  `V_CVT_F32_F64` at `:44389` through `:44431`, `V_TRUNC_F64` at `:45182`
  through `:45224`, `V_FREXP_EXP_I32_F64` at `:47759` through `:47801`, and
  `V_FRACT_F64` at `:47943` through `:47985`.

Impact:

Generated validators that trust the checked-in XML row set can accept DPP
forms that the manual explicitly excludes for F64 VOP1 instructions.

### CDNA3-XML-071: VOPC DPP extension availability is absent from checked-in XML

Manual evidence:

- Chapter 13.3.9 says the DPP extension word can follow `VOP1`, `VOP2`, or
  `VOPC` instructions in place of a literal constant at
  `cdna3/README.md:24155` through `:24168`.
- Chapter 12.16.1 then narrows that general rule by listing prohibited DPP
  instructions, including the 64-bit VOPC compare families, at
  `cdna3/README.md:22583` through `:22616`. Non-64 VOPC compare/class rows are
  not in that exclusion list.

XML evidence:

- The checked-in CDNA3 XML defines and attaches `VOP1_VOP_DPP` and
  `VOP2_VOP_DPP` encodings, for example at
  `amdgpu_isa_cdna3.xml:5238`, `:6158`, and representative VOP1/VOP2 opcode
  rows at `:42992` and `:51524`, but searching the file finds no
  `VOPC_*DPP` encoding name or VOPC instruction row with a DPP encoding.
- Representative legal non-64 VOPC rows such as `V_CMP_F_F32` expose only
  `ENC_VOP3`, `ENC_VOPC`, `VOPC_INST_LITERAL`, and
  `VOPC_VOP_SDWA_SDST_ENC` at `amdgpu_isa_cdna3.xml:73236` through `:73345`.

Impact:

An XML-only decoder or validator cannot learn that non-64 VOPC DPP encodings
exist, nor can it distinguish those legal forms from the 64-bit compare
families excluded by Chapter 12.16.1.

### CDNA3-XML-072: VOPC compare predicates and class masks are prose-only

Manual evidence:

- Chapter 12.9 defines the 16 floating compare operation offsets and 8 integer
  compare operation offsets in tables at `cdna3/README.md:8870` through
  `:8931`.
- The detailed floating rows are NaN-sensitive. For example `V_CMP_U_F16` uses
  `isNAN(S0) || isNAN(S1)`, and `V_CMP_NGE_F16` warns that NaN inputs make the
  negated predicate different from an ordered `<`, at `cdna3/README.md:9268`
  through `:9292`.
- `V_CMP_CLASS_*` and `V_CMPX_CLASS_*` define the 10-bit class mask ordering,
  including signaling/quiet NaN, infinities, normals, denormals, and signed
  zero, at `cdna3/README.md:8936` through `:9015`.

XML evidence:

- The checked-in XML carries opcode names, operand formats, and short
  descriptions for representative compare rows such as `V_CMP_F_F32`,
  `V_CMP_U_F32`, `V_CMP_NGE_F32`, `V_CMP_F_I32`, and `V_CMPX_F_I32`, but does
  not encode the operation-offset truth tables or NaN-sensitive formulas as
  structured semantics at `amdgpu_isa_cdna3.xml:73236`, `:74148`, `:74262`,
  `:84260`, and `:86084`.
- The class rows describe a "10 bit mask" but do not structure the mask-bit
  order or the denormal/signed-zero tests; `V_CMP_CLASS_F32` and
  `V_CMPX_CLASS_F32` are representative at `amdgpu_isa_cdna3.xml:68579`
  through `:68840`.

Impact:

XML-derived semantics can recover broad operand and opcode data, but cannot
derive exact ordered/unordered floating predicates, negated-NaN behavior, or
class-mask bit meanings without manual prose.

### CDNA3-XML-073: VOPC summary tables and VOP3A wording drift from detailed rows

Manual evidence:

- The Chapter 12.9 summary range table names the F64 families as
  `V_CMPS_*_F64`/`V_CMPSX_*_F64`, while the detailed rows and opcode tables use
  `V_CMP_*_F64`/`V_CMPX_*_F64`, at `cdna3/README.md:8896` through `:8905`.
- The integer summary table uses `LG`/`TRU` operation labels and swaps several
  signed/unsigned descriptions; detailed rows use names such as
  `V_CMP_NE_I16` and signed `S0.i16` dataflow at `cdna3/README.md:8909`
  through `:8931` and `:10200` through `:10222`.
- Chapter 12.9.1 describes VOP3A `VDST` generically as a VGPR destination at
  `cdna3/README.md:11084` through `:11105`, while Chapter 13.3.3 says VOP3
  compare results can target an arbitrary SGPR at `cdna3/README.md:23320`
  through `:23340`.

XML evidence:

- The checked-in XML matches the detailed instruction names and SGPR
  destination metadata rather than the summary drift. `V_CMP_CLASS_F32`
  `ENC_VOP3` records `VDST` as `OPR_SREG` at
  `amdgpu_isa_cdna3.xml:68579` through `:68615`, and `V_CMP_F_F64` appears as
  a `V_CMP_*` row at `:77204`.
- Integer rows similarly use the signedness implied by the mnemonic/data
  format rather than the contradictory range-table descriptions.

Impact:

Tools that scrape manual summary tables instead of detailed rows or XML can
misname F64 compare families, infer the wrong integer signedness for some
ranges, or incorrectly treat VOP3A compare destinations as VGPRs.

### CDNA3-XML-074: Packed 16-bit VOP3P operand widths are inconsistent

Manual evidence:

- Chapter 12.10 describes the packed 16-bit VOP3P rows as two 16-bit
  component operations packed into a dword destination. For example,
  `V_PK_MAD_I16`, `V_PK_MUL_LO_U16`, `V_PK_MAX_I16`, `V_PK_MAD_U16`,
  `V_PK_MAX_U16`, and `V_PK_MIN_F16` all write `D0.b32` after producing both
  low and high 16-bit components at `cdna3/README.md:11115` through `:11130`,
  `:11191` through `:11219`, `:11248` through `:11265`, and `:11307` through
  `:11322`.

XML evidence:

- The checked-in XML marks some packed 16-bit rows as 16-bit operands:
  `V_PK_MAD_I16` and `V_PK_MUL_LO_U16` at
  `amdgpu_isa_cdna3.xml:64723` through `:64788`, `V_PK_MAX_I16`,
  `V_PK_MIN_I16`, and `V_PK_MAD_U16` at `:65016` through `:65121`,
  `V_PK_MAX_U16` and `V_PK_MIN_U16` at `:65227` through `:65285`, and
  `V_PK_MIN_F16`/`V_PK_MAX_F16` at `:65438` through `:65497`.
- Adjacent packed 16-bit rows such as `V_PK_ADD_I16`, `V_PK_SUB_I16`,
  `V_PK_ADD_U16`, `V_PK_SUB_U16`, `V_PK_FMA_F16`, `V_PK_ADD_F16`, and
  `V_PK_MUL_F16` use 32-bit operands for the same packed-dword dataflow at
  `amdgpu_isa_cdna3.xml:64811` through `:64870`, `:65145` through `:65203`,
  and `:65310` through `:65415`.

Impact:

The XML mixes element-width and packed-register-width metadata for the same
instruction family. XML-only consumers cannot reliably infer both facts needed
for these rows: the operands occupy one 32-bit packed VGPR lane, while inline
constants and selectors may need 16-bit element interpretation.

### CDNA3-XML-075: Packed VOP3P shift-count bit slicing is prose-only

Manual evidence:

- `V_PK_LSHLREV_B16`, `V_PK_LSHRREV_B16`, and `V_PK_ASHRREV_I16` use only
  `S0.u32[3:0]` for the low component shift count and `S0.u32[19:16]` for the
  high component shift count at `cdna3/README.md:11155` through `:11184`.

XML evidence:

- The corresponding XML rows record generic packed shift descriptions and
  32-bit `FMT_NUM_B16`/`FMT_NUM_I16` operands at
  `amdgpu_isa_cdna3.xml:64886` through `:64992`, but do not encode the
  component-specific nibble extraction from source 0.

Impact:

The shift count is not simply the selected 16-bit component value; only the low
four bits of each selected component participate. That bit slicing must be
recovered from manual prose or instruction-specific implementation knowledge.

### CDNA3-XML-076: VOP3B `SDST` field description is compare-specific

Manual evidence:

- Chapter 12.11 says VOP3B is the scalar-destination encoding used only by
  `V_ADD_CO_U32`, `V_SUB_CO_U32`, `V_SUBREV_CO_U32`, `V_ADDC_CO_U32`,
  `V_SUBB_CO_U32`, `V_SUBBREV_CO_U32`, `V_DIV_SCALE_F32`, `V_DIV_SCALE_F64`,
  `V_MAD_U64_U32`, and `V_MAD_I64_I32` at `cdna3/README.md:12393` through
  `:12413`.
- Chapter 13.3.5 labels `SDST [14:8]` simply as "Scalar destination" and
  repeats the same ten-opcode VOP3B list at `cdna3/README.md:23861` through
  `:23942`.

XML evidence:

- The `VOP3_SDST_ENC` field map is present, but its `SDST` field description
  says "Destination for compare result" at `amdgpu_isa_cdna3.xml:6970`
  through `:7055`.
- The instruction rows themselves expose `SDST` for the non-compare VOP3B
  opcodes, for example `V_DIV_SCALE_F32` at `amdgpu_isa_cdna3.xml:61285`
  through `:61324` and `V_MAD_U64_U32` at `:61683` through `:61775`.

Impact:

Field-level XML documentation misclassifies VOP3B's scalar destination as a
compare-only result even though the encoding is dedicated to carry, division
scale, and wide multiply-add scalar outputs.

### CDNA3-XML-077: VOP3B carry/divide-scale/wide-MAD semantics are prose-only

Manual evidence:

- The VOP3B add/sub carry rows define carry-out and borrow masks, arbitrary
  SGPR-pair VCC destinations, source carry-in from `S2`, and integer-domain
  saturation support at `cdna3/README.md:13909` through `:14010`.
- `V_DIV_SCALE_F32` and `V_DIV_SCALE_F64` define division-macro scaling and
  condition-mask behavior at `cdna3/README.md:14804` through `:14890`.
- `V_MAD_U64_U32` and `V_MAD_I64_I32` define 65-bit result/carry formulas at
  `cdna3/README.md:14991` through `:15011`.

XML evidence:

- The corresponding XML rows record operands and short descriptions, for
  example `V_ADD_CO_U32`/`V_ADDC_CO_U32` at
  `amdgpu_isa_cdna3.xml:54844` through `:54872` and `:55369` through `:55397`,
  `V_DIV_SCALE_F32` at `:61285` through `:61324`, and `V_MAD_U64_U32` at
  `:61683` through `:61775`.
- The XML does not encode per-lane carry/borrow equations, carry-in mask
  indexing, saturation behavior, divide scaling edge cases, or 65-bit
  wide-MAD overflow formulas as structured semantics.

Impact:

An XML-only executor or validator can discover the VOP3B operand shape, but not
the arithmetic and mask contracts that make the scalar destination meaningful.

### CDNA3-XML-078: VOP3 literal legality is internally contradictory

Manual evidence:

- The VOP3 overview says 32-bit literal constants are available for 32-bit
  microcode formats "but not VOP3" at `cdna3/README.md:1195` through `:1199`.
- The VOP2, VOP1, and VOPC promoted-to-VOP3 notes repeat that VOP3 extra
  control bits come "in exchange for not being able to use a literal constant"
  at `cdna3/README.md:7531`, `:8845`, and `:11087`.
- The generic VOP3A and VOP3B field tables still list source selector `255` as
  "Literal constant" at `cdna3/README.md:23597` and `:23924`.

XML evidence:

- Generic `ENC_VOP3` and `VOP3_SDST_ENC` expose source-selector fields, and
  the shared operand-type tables include literal predefined values such as
  `src_literal` at `amdgpu_isa_cdna3.xml:101442` through `:101449`.
- Individual VOP3 instruction rows generally use source classes such as
  `OPR_SRC_NOLIT` and `OPR_SRC_SIMPLE`, for example `V_MOV_B32` at
  `amdgpu_isa_cdna3.xml:43102` through `:43117`.

Impact:

Consumers that scrape only the manual field tables or generic XML source tables
can accept an extension-literal VOP3 form that the manual prose says is not a
legal VOP3 instruction form. The no-literal rule must be taken from prose and
instruction-specific operand handling rather than the generic selector table.

### CDNA3-XML-079: `V_LSHL_ADD_U64` unsupported shift-count rule is prose-only

Manual evidence:

- `V_LSHL_ADD_U64` says its shift count must be between 0 and 4, higher counts
  are unsupported, and the design treats unsupported counts as a shift of zero
  at `cdna3/README.md:15462` through `:15475`.

XML evidence:

- The XML row records `V_LSHL_ADD_U64` as a 64-bit source/add instruction with
  a 32-bit `SRC1` at `amdgpu_isa_cdna3.xml:63193` through `:63225`, but does
  not encode the legal count range or the "unsupported means shift zero" rule.

Impact:

XML-derived execution and validation cannot distinguish `V_LSHL_ADD_U64` from
ordinary masked 64-bit shifts. Counts 5 and above need manual-derived handling.

### CDNA3-XML-080: Specialized VOP3A formulas and writeback rules are prose-only

Manual evidence:

- `V_QSAD_PK_U16_U8`, `V_MQSAD_PK_U16_U8`, and `V_MQSAD_U32_U8` define
  multiple byte-window SAD/MSAD formulas and packed 64/128-bit outputs at
  `cdna3/README.md:14954` through `:14989`.
- `V_CVT_PKACCUM_U8_F32` packs one converted byte into either `S2` or the old
  destination depending on the opcode form at `cdna3/README.md:14718` through
  `:14729` and `:15170` through `:15183`.
- `V_TRIG_PREOP_F64` defines the 2/PI segment lookup, exponent-dependent shift,
  round-toward-zero behavior, and large-exponent scaling at
  `cdna3/README.md:15727` through `:15762`.
- `V_CVT_PKRTZ_F16_F32` explicitly ignores the current rounding mode and uses
  round-toward-zero at `cdna3/README.md:15783` through `:15808`; the adjacent
  packed normalized conversions define per-component conversion formulas.

XML evidence:

- The corresponding XML rows record operand widths and short descriptions, for
  example the QSAD/MQSAD group at `amdgpu_isa_cdna3.xml:61543` through
  `:61659`, `V_CVT_PKACCUM_U8_F32` at `:62071` through `:62098`,
  `V_TRIG_PREOP_F64` at `:63937` through `:63968`, and packed conversion rows
  around `:64019` through `:64142`.
- Those rows do not encode the byte windows, old-destination/source-C merge
  convention, 2/PI lookup algorithm, per-opcode writeback rules, or RTZ
  rounding override as structured semantics.

Impact:

Instruction decoders can enumerate these VOP3A rows from XML, but executing or
validating their corner behavior still requires manual prose or hand-maintained
semantic metadata.

### CDNA3-XML-081: DS opcode inventory is complete in XML, but the Chapter 13.4 table appears to omit two rows

Manual evidence:

- Chapter 12.12 defines `DS_MIN_RTN_F32 50` at `cdna3/README.md:18669` and
  `DS_READ2_B64 119` at `cdna3/README.md:19648`.
- The Chapter 13.4.1 opcode table jumps from opcode 49 to 51 and from opcode
  118 to 120, with a blank row between the table fragments at
  `cdna3/README.md:24273` through `:24280`.

XML evidence:

- The checked-in XML contains `DS_MIN_RTN_F32` opcode 50 at
  `amdgpu_isa_cdna3.xml:10135` and `DS_READ2_B64` opcode 119 at
  `amdgpu_isa_cdna3.xml:13176`.
- The local Chapter 12.12 inventory found matching manual/XML instruction
  definitions for all 128 non-invalid `ENC_DS` rows after accounting for
  markdown headings formatted as fenced code blocks.

Impact:

The machine-readable XML and detailed instruction definitions are complete, but
consumers that scrape only the Chapter 13.4 opcode table text can miss opcodes
50 and 119. This may be a markdown/PDF extraction artifact; the audit did not
visually verify the original PDF table rendering.

### CDNA3-XML-082: DS atomic RMW formulas are prose-only

Manual evidence:

- Chapter 12.12 gives instruction-specific update formulas for integer atomics,
  including `INC`/`DEC` wrap predicates, signed/unsigned min/max selection,
  bitwise ops, `MSKOR`, and compare-store source/compare ordering at
  `cdna3/README.md:18143`, `:18251`, and `:18293`.
- The return forms repeat the same formulas with old-value writeback, and
  floating variants add NaN/Inf/denorm notes throughout the DS atomic rows.

XML evidence:

- Representative XML rows such as `DS_MSKOR_B32` and `DS_CMPST_B32` expose
  operand formats and implicit `OPR_DSMEM` side effects at
  `amdgpu_isa_cdna3.xml:8523` and `:8710`, but do not encode the executable
  predicates, operand ordering, tie behavior, or old-value return formulas.

Impact:

The XML is useful for decode and side-effect classification, but an emulator or
validator still needs manual-derived semantics for exact DS atomic behavior.

### CDNA3-XML-083: Special DS addressing rules are not fully machine-readable

Manual evidence:

- Chapter 11 defines single-address, two-address, ST64, and duplicate-offset DS
  addressing at `cdna3/README.md:3282` through `:3302`.
- `DS_WRITE_ADDTID_B32` and `DS_READ_ADDTID_B32` use
  `{OFFSET1, OFFSET0} + M0[15:0] + laneID * 4` rather than an `ADDR` VGPR at
  `cdna3/README.md:18410` and `:19835`.
- `DS_CONSUME` and `DS_APPEND` split their LDS and GDS address rules: LDS uses
  `instr_offset`, while GDS uses `M0.base + instr_offset`, at
  `cdna3/README.md:19882` through `:19888`.

XML evidence:

- `ENC_DS` only says `OFFSET0` and `OFFSET1` use depends on the instruction at
  `amdgpu_isa_cdna3.xml:2893` through `:2912`.
- The ADDTID XML rows expose implicit `OPR_SDST_M0` and omit explicit `ADDR` at
  `amdgpu_isa_cdna3.xml:9047` and `:13521`, but their descriptions do not
  encode the `M0[15:0] + laneID * 4` formula.
- The XML descriptions for `DS_CONSUME` and `DS_APPEND` say `M0.base +
  instr_offset` for "LDS & GDS" at `amdgpu_isa_cdna3.xml:13645` and `:13685`,
  collapsing the manual's LDS/GDS distinction.

Impact:

XML-only consumers can decode these rows, but cannot derive the full access
count and address contract for ADDTID, READ2/WRITE2/ST64, duplicate-offset, or
APPEND/CONSUME without manual prose.

### CDNA3-XML-084: DS D16 half-source and preserve-destination behavior is not structured

Manual evidence:

- D16 write rows use the high half of the data register for `_HI` forms, such as
  `DATA[23:16]` and `DATA[31:16]`, and D16 read rows preserve the other half of
  `VDST` at `cdna3/README.md:19285` through `:19301`.

XML evidence:

- Representative D16 read rows mark `VDST` as an output-only 32-bit operand with
  narrow DSMEM input sizes at `amdgpu_isa_cdna3.xml:11773` through `:12001`.
- The XML descriptions mention low/high placement in prose, but do not expose a
  structured read-modify-write dependency on the old destination half or a
  structured source-half selector for high-half writes.

Impact:

Def-use, liveness, and execution models cannot infer D16 old-destination reads
or source-half selection from XML operands alone.

### CDNA3-XML-085: DS swizzle and permute lane-routing formulas are prose-only

Manual evidence:

- `DS_SWIZZLE_B32` defines FFT, rotate, and two basic modes, plus disabled-source
  zeroing, at `cdna3/README.md:18799` through `:18913`.
- `DS_PERMUTE_B32` and `DS_BPERMUTE_B32` define address divide-by-four, ignored
  high bits, EXEC source/destination behavior, disabled-source zeroing, and
  collision arbitration at `cdna3/README.md:18915` through `:19004`.

XML evidence:

- The XML rows for `DS_SWIZZLE_B32`, `DS_PERMUTE_B32`, and
  `DS_BPERMUTE_B32` carry short descriptions and operand metadata at
  `amdgpu_isa_cdna3.xml:10636` through `:10747`, but no lane-routing formulas
  or EXEC/collision rules.

Impact:

XML consumers can identify these non-LDS-memory lane-routing instructions, but
cannot implement their exact lane mapping without manual-derived logic.

### CDNA3-XML-086: `DS_CONDXCHG32_RTN_B64` write-mask semantics are prose-only

Manual evidence:

- `DS_CONDXCHG32_RTN_B64` aligns the address with `& 0xfff8`, reads two dwords,
  writes each half only when the corresponding input MSB is set, clears that MSB
  on write, and returns both old dwords at `cdna3/README.md:19688` through
  `:19709`.

XML evidence:

- The XML row gives a short description and generic 64-bit DSMEM input/output
  operands at `amdgpu_isa_cdna3.xml:13310`, but does not encode the address
  alignment, per-half conditional writes, or MSB-clearing behavior.

Impact:

This opcode cannot be implemented correctly from XML operand metadata alone.

### CDNA3-XML-087: DS `ACC` source operand metadata may be broader than the manual field prose

Manual evidence:

- Chapter 13.4.1 describes the `ACC` field as "`VDST` is Accumulation VGPR" and
  separately describes `ADDR`, `DATA0`, and `DATA1` as VGPR fields at
  `cdna3/README.md:24216` through `:24221`.
- Chapter 12.12's common field prose similarly calls `DATA0` and `DATA1` source
  VGPRs at `cdna3/README.md:18102`.

XML evidence:

- Many DS `DATA0`/`DATA1` operands use `OPR_VGPR_OR_ACCVGPR`, including
  representative atomic rows around `amdgpu_isa_cdna3.xml:8537` and `:8589`.
- The `ENC_DS` field description still says `ACC` specifies whether `VDST` uses
  an accumulator VGPR at `amdgpu_isa_cdna3.xml:2834`.

Impact:

The XML may allow accumulator source operands where the manual's DS format prose
documents only an accumulator destination selector. This entry is high
uncertainty until assembler or hardware legality is checked; the XML may be
using `OPR_VGPR_OR_ACCVGPR` as the intended operand-class authority.

### CDNA3-XML-088: MUBUF integer atomic formulas are prose-only

Manual evidence:

- Chapter 12.13 gives executable formulas for MUBUF integer atomics, including
  compare-swap source/compare ordering at `cdna3/README.md:20434`, `INC`/`DEC`
  wrap predicates at `cdna3/README.md:20561` through `:20581`, and the 64-bit
  X2 equivalents at `cdna3/README.md:20670` and `:20792` through `:20810`.

XML evidence:

- The checked-in XML rows carry opcodes, operand widths, and short descriptions
  for representative entries such as `BUFFER_ATOMIC_CMPSWAP`,
  `BUFFER_ATOMIC_INC`, `BUFFER_ATOMIC_DEC`, and `BUFFER_ATOMIC_CMPSWAP_X2` at
  `amdgpu_isa_cdna3.xml:23719`, `:24199`, `:24247`, and `:24583`, but they do
  not encode structured update formulas or the compare/source half ordering.
- `CDNA3-XML-022` separately tracks stale `GLC` return wording, and
  `CDNA3-XML-054` tracks floating memory atomic numeric rules; neither records
  the integer RMW formulas.

Impact:

An XML-only consumer can identify the atomic opcode and operand sizes, but
cannot derive exact compare-swap packing, signed/unsigned min/max behavior, or
`INC`/`DEC` wrap edge conditions from structured metadata.

### CDNA3-XML-089: F64 memory atomic min/max descriptions reuse integer text

Manual evidence:

- Chapter 12.15 defines `FLAT_ATOMIC_MIN_F64` and `FLAT_ATOMIC_MAX_F64` as
  double-precision floating-point min/max operations using `.f64` dataflow at
  `cdna3/README.md:21486` through `:21505`.
- The corresponding `GLOBAL_ATOMIC_MIN_F64` and `GLOBAL_ATOMIC_MAX_F64` rows
  use the same double-precision floating-point wording and `.f64` formulas at
  `cdna3/README.md:22385` through `:22404`.
- The Chapter 12.13 `BUFFER_ATOMIC_MIN_F64` and
  `BUFFER_ATOMIC_MAX_F64` rows also define double-precision floating-point
  min/max behavior at `cdna3/README.md:20617` and later.

XML evidence:

- The XML rows are named `FLAT_ATOMIC_MIN_F64`/`MAX_F64`,
  `GLOBAL_ATOMIC_MIN_F64`/`MAX_F64`, and `BUFFER_ATOMIC_MIN_F64`/`MAX_F64`,
  and their operands are 64-bit, but their descriptions say "minimum signed
  integer" or "maximum signed integer" at
  `amdgpu_isa_cdna3.xml:15733`, `:15785`, `:18682`, `:18735`, `:24439`, and
  `:24487`.
- The same descriptions also use stale `GLC` return wording, which overlaps
  the broader return-wording gaps in `CDNA3-XML-022` and `CDNA3-XML-026`.

Impact:

Name and operand metadata point to F64 atomics, but description-based consumers
can misclassify the operation as integer min/max and miss the floating-point
numeric contract.

### CDNA3-XML-090: FLAT/GLOBAL integer atomic formulas are prose-only

Manual evidence:

- Chapter 12.15 gives executable formulas for FLAT integer atomics, including
  compare-swap source/compare ordering at `cdna3/README.md:21296`, `INC`/`DEC`
  wrap predicates at `cdna3/README.md:21367` through `:21388`, and the 64-bit
  X2 equivalents at `cdna3/README.md:21545` and `:21667` through `:21685`.
- The GLOBAL integer atomic rows repeat the same RMW formulas and edge
  predicates in the global-address form at `cdna3/README.md:22184` through
  `:22460`.

XML evidence:

- The XML rows carry opcodes, operand widths, and short descriptions for
  representative `FLAT_ATOMIC_CMPSWAP`, `FLAT_ATOMIC_INC`,
  `FLAT_ATOMIC_DEC`, and X2 entries at
  `amdgpu_isa_cdna3.xml:14953`, `:15480`, `:15531`, and `:15941`; the GLOBAL
  equivalents begin at `:17887` and later.
- The XML does not encode structured RMW formulas, the compare/source half
  ordering for compare-swap, signed versus unsigned min/max comparison rules,
  or the `INC`/`DEC` wrap predicates.
- `CDNA3-XML-026` separately tracks stale atomic return wording and
  floating-point atomic rules; `CDNA3-XML-054` tracks floating memory atomic
  numeric rules.

Impact:

An XML-only consumer can identify the flat/global atomic opcode and operand
sizes, but cannot derive exact integer RMW behavior without manual-derived
rules.

### CDNA3-XML-091: FLAT partial-destination preservation is prose-only

Manual evidence:

- FLAT, SCRATCH, and GLOBAL D16 load rows explicitly preserve the untouched
  half of `VDATA`, for example `cdna3/README.md:21228` through `:21280`,
  `:21862` through `:21914`, and `:22108` through `:22160`.
- `FLAT_ATOMIC_CMPSWAP` says `RETURN_DATA[1]` is not modified at
  `cdna3/README.md:21298`, and `FLAT_ATOMIC_CMPSWAP_X2` says
  `RETURN_DATA[2:3]` is not modified at `cdna3/README.md:21547`.

XML evidence:

- D16 rows expose a 32-bit `VDST` operand and low/high placement descriptions,
  for example `FLAT_LOAD_UBYTE_D16` and `_HI` at
  `amdgpu_isa_cdna3.xml:14625` and `:14671`, but do not mark the destination as
  read-modify-write or preserve-half.
- The flat compare-swap rows carry destination operand widths and descriptions
  but no structured "upper return dwords unchanged" metadata at
  `amdgpu_isa_cdna3.xml:14953` and `:15941`.

Impact:

Generated execution or def-use code cannot discover from XML alone which
destination bits/dwords must be preserved for D16 loads and flat compare-swap
returns.

### CDNA3-XML-092: 64-bit DPP control legality is prose-only

Manual evidence:

- Chapter 13.3.9 says a DPP extension word can follow VOP1, VOP2, or VOPC
  instructions at `cdna3/README.md:24155` through `:24168`.
- The same DPP control table lists `DPP_ROW*` as the `0x150` through `0x15f`
  control range and then states that for 64-bit input data the only legal DPP
  type is `DPP_ROW*` at `cdna3/README.md:24176` through `:24199`.

XML evidence:

- The generic `VOP1_VOP_DPP` and `VOP2_VOP_DPP` microcode formats expose
  `DPP_CTRL` only as an unconstrained 9-bit "Data-parallel primitive control"
  field at `amdgpu_isa_cdna3.xml:5238`, `:5391`, `:6158`, and `:6294`; their
  `has_dpp` conditions only classify `SRC0 == 250`.
- Representative 64-bit DPP-capable rows such as `V_MOV_B64` and
  `V_FMAC_F64` record 64-bit source operands and DPP alternatives at
  `amdgpu_isa_cdna3.xml:48402` through `:48459` and `:52050` through `:52113`,
  but do not encode the `DPP_CTRL` subset restriction for those 64-bit inputs.

Impact:

An XML-derived validator can discover that 64-bit DPP rows exist, but cannot
derive from structured metadata that only the `DPP_ROW*` control family is
legal for those rows.

### CDNA3-XML-093: Flat-family `NV` and `ACC` field metadata is inconsistent

Manual evidence:

- The Chapter 10.1 field table lists `ACC` as applying to `DATA`, lists
  `SEG=3` as reserved, lists `SVE`, and then lists `NV` as a non-volatile
  read/write bit at `cdna3/README.md:3023` through `:3042`.
- The Chapter 13.6 FLAT field map lists bit 13 as `LDS`, `SEG`, `SC0`, `NT`,
  `OP`, `SC1`, `ADDR`, `DATA`, `SADDR`, `ACC`, and `VDST`, but has no `NV`
  field at `cdna3/README.md:24435` through `:24458`.
- The Chapter 13.6 `ACC` row says `VDATA` is the Accumulation VGPR at
  `cdna3/README.md:24457`, which agrees with the Chapter 10.1 `DATA` role but
  not with an output-only interpretation.

XML evidence:

- The generic `ENC_FLAT` field map records `ACC` as selecting whether `VDST`
  uses an accumulator VGPR at `amdgpu_isa_cdna3.xml:3477`, and no flat-family
  field map records an `NV` bit.
- Per-instruction operands show that the meaning is opcode-role dependent:
  stores mark `DATA` as AccVGPR-capable, loads mark `VDST`, and atomics mark
  both roles, for example at `amdgpu_isa_cdna3.xml:14593`, `:14633`, and
  `:14909`.

Impact:

The prose and field maps disagree about whether flat-family bit 55 is an input
or output accumulator selector and whether a Chapter 10 `NV` control exists at
all. XML-only consumers need opcode-level operand metadata and manual context
to avoid treating `ACC` as only a `VDST` selector or inventing an absent `NV`
field.

## No-Gap Notes

- Counting both `ENC_VOP3` and `VOP3_SDST_ENC`, every named instruction in
  CDNA3 Chapter 12.11 has a checked-in XML row. The apparent XML-only rows
  `V_QSAD_PK_U16_U8`, `V_MAD_U32_U16`, and `V_CMPX_NEQ_F16` are present in the
  manual body/table but formatted as code-block text rather than normal bold
  headings at `cdna3/README.md:14954`, `:15185`, and `:16556`.
- The ten manual VOP3B scalar-destination opcodes are present in the checked-in
  XML as `VOP3_SDST_ENC` rows with matching opcodes. The VOP3B gaps above are
  about field-description wording and prose-only semantics, not missing
  encoding coverage.
- The CDNA3 Chapter 12.8 VOP1 opcode inventory matches the checked-in XML for
  all manual-listed rows: manual opcodes 0-8, 10-53, 56-74, 77-79, 81-82, and
  84-87 are present in both sources. Opcode 55 is the only extra checked-in XML
  VOP1 row found in this slice, and is tracked by `CDNA3-XML-065`.
- The CDNA3 Chapter 12.9 VOPC opcode inventory matches the checked-in XML for
  all 198 detailed manual rows: the class opcodes 16-21, FP compare ranges
  32-127, and integer compare ranges 160-255 are present with matching
  `ENC_VOPC` opcodes.
- VOP3A VOPC opcode table entries use the same opcodes as the corresponding
  VOPC rows in the checked-in XML; this slice did not find a VOP1-style
  promoted-opcode offset drift for VOPC compares.
- The checked-in XML does model the key CMPX destination split at operand level:
  VOPC CMPX rows write VCC and implicit EXEC, while VOP3 CMPX rows expose an
  explicit scalar destination plus implicit EXEC. The remaining VOPC gaps above
  are about prose-only semantics, missing DPP availability, and manual summary
  drift rather than absence of all destination metadata.
- The checked-in XML records `V_SWAP_B32` as read-write dataflow for both
  `VDST` and `SRC0`, and records `V_ACCVGPR_MOV_B32` with accumulator operand
  classes at `amdgpu_isa_cdna3.xml:50851` through `:50973`. The remaining
  `V_SWAP_B32` concerns are instruction-extension legality and runtime behavior,
  not missing XML input/output metadata.
- The FP8/BF8 VOP1 conversion rows 84-87 were already recorded in
  `CDNA3-XML-050` through `CDNA3-XML-052` for packed destination alignment,
  F32-to-FP8/BF8 modifier/`OP_SEL` overrides, and SDWA ignored-field behavior;
  this VOP1 inventory pass did not find an additional opcode-inventory gap for
  those rows.
- The CDNA3 Chapter 12.7 VOP2 opcode inventory matches the checked-in XML:
  manual opcodes 0-21 and 23-61 are present in both sources, opcode 22 is
  absent from both, ordinary promoted rows use the documented VOP3 opcode
  offset of `0x100`, and the literal-only `_MK`/`_AK` rows omit promoted VOP3
  alternatives. The gaps above are about detailed semantics and legality, not
  missing base VOP2 rows.
- For the Chapter 12.16 SDWA prohibited-instruction list, every listed row that
  exists in the checked-in CDNA3 XML omits an SDWA alternative. For example,
  `V_MAC_F16` and `V_FMAC_F32` have DPP but no SDWA rows at
  `amdgpu_isa_cdna3.xml:56369` and `:59502`, while `V_READFIRSTLANE_B32`,
  `V_CLREXCP`, and `V_SWAP_B32` have no DPP/SDWA rows at `:43133`, `:48257`,
  and `:50851`. The runtime over-acceptance is tracked separately in
  `CDNA3-RJ-119`.
- Chapter 12.16 names `V_MADMK_F32` and `V_MADAK_F32` in the DPP/SDWA
  limitation lists, while the CDNA3 opcode table and XML use
  `V_FMAMK_F32`/`V_FMAAK_F32`. The XML rows for those F32 literal-only forms
  carry only literal encodings, so this audit treats the name split as manual
  naming drift rather than an XML extension-row leak.
- The CDNA3 Chapter 12.5 SOPP opcode inventory matches the checked-in XML for
  manual opcodes 0 through 29 at `cdna3/README.md:5243` through `:5601` and
  `amdgpu_isa_cdna3.xml:42048` through `:42904`. `CDNA3-XML-059` records the
  separate XML-only opcode 30 and 31 rows.
- The CDNA3 Chapter 12.3 SOP1 opcode inventory matches the checked-in XML:
  opcodes 0-46, 48, and 50-55 are present, while opcode holes 47 and 49 are
  absent from both sources at `cdna3/README.md:4173` through `:5030` and
  `amdgpu_isa_cdna3.xml:28543` through `:31945`. `CDNA3-XML-057` records
  missing formulas and edge contracts, not missing base SOP1 rows.
- The CDNA3 Chapter 12.4 SOPC opcode inventory matches the checked-in XML:
  opcodes 0 through 19 are present in both sources at `cdna3/README.md:5034`
  through `:5239` and `amdgpu_isa_cdna3.xml:38994` through `:41220`.
  `CDNA3-XML-058` records missing formulas and instruction-specific raw-field
  contracts, not missing base SOPC rows.
- Except for the `S_RFE_RESTORE_B64` opcode 43 drift recorded in
  `CDNA3-XML-056` and the SOPP opcode 30/31 drift recorded in
  `CDNA3-XML-059`, the CDNA3 XML covers the audited SOP1, SOP2, SOPC, SOPK, and
  SOPP field maps and opcode inventories for the SALU formats slice. The gaps
  above are about selector semantics, detailed formulas, source drift, and
  runtime rules, not missing base SOP encodings.
- The CDNA3 XML models the SALU literal split at row granularity:
  literal-capable SOP1/SOP2/SOPC rows carry `SOP*_INST_LITERAL` alternatives,
  default-only PC forms such as `S_GETPC_B64`, `S_SETPC_B64`, and
  `S_SWAPPC_B64` omit them, SOPP has no literal alternative, and SOPK's only
  64-bit literal encoding is the `S_SETREG_IMM32_B32` exception.
  `CDNA3-XML-058` records the separate `S_SET_GPR_IDX_ON` source-1 raw-field
  exception.
- The CDNA3 Chapter 12.2 SOPK opcode inventory matches the checked-in XML after
  accounting for the literal-only `S_SETREG_IMM32_B32` encoding: the manual and
  XML both omit opcode 19, both expose opcode 20 as a 32-bit-literal HWREG form,
  and both list opcode 21 as a PC-relative `S_CALL_B64` row at
  `cdna3/README.md:3924` through `:4169` and
  `amdgpu_isa_cdna3.xml:41234` through `:42039`.
- The XML records the SOPK old-destination dataflow needed by `S_ADDK_I32` and
  `S_MULK_I32`: `SDST` is both input and output for both rows, and only
  `S_ADDK_I32` has an implicit SCC output at `amdgpu_isa_cdna3.xml:41789`
  through `:41849`.
- The CDNA3 XML does record VALU literal expansion semantics in the generic
  source-literal predefined values: 16-bit operands use the low 16 bits, 64-bit
  unsigned operands zero-extend, 64-bit signed operands sign-extend, and 64-bit
  floating operands place the literal in the high 32 bits of the double at
  `amdgpu_isa_cdna3.xml:97728` through `:97731`,
  `:101447` through `:101449`, and `:115579` through `:115581`. The VALU
  literal gaps above are about cross-operand legality, not this expansion text.
- The CDNA3 XML contains the audited VOP3P packed-math opcode inventory,
  including packed 16-bit arithmetic, MIX, DOT, packed F32 arithmetic, and
  `V_PK_MOV_B32`. The gaps above are semantic and legality gaps, not missing
  opcode entries.
- The full CDNA3 Chapter 12.10 VOP3P opcode inventory matches the checked-in
  XML when both `ENC_VOP3P` and `VOP3P_MFMA` rows are counted: 81 manual
  definitions, 81 XML encodings, no manual-only rows, no XML-only rows, and no
  opcode mismatches in the local inventory script.
- `V_ACCVGPR_READ` and `V_ACCVGPR_WRITE` are present in the checked-in XML with
  `_B32` aliases and 32-bit accumulator/register operand classes at
  `amdgpu_isa_cdna3.xml:67020` through `:67079`. Although the manual's
  `V_ACCVGPR_WRITE` prose says "architectural vector register", LLVM accepts
  the broad source selector forms used by real kernels such as inline `0` and
  scalar sources, so this static pass does not classify the XML's
  `OPR_SRC_NOLIT` source class as a hard XML contradiction.
- The CDNA3 XML contains representative VALU opcode inventory entries for the
  audited Chapter 6.3 through 6.6 slice, including ordinary FP/INT VOP3
  instructions, compare instructions, VOP2 promoted forms, `V_DOT2`, and the
  `S_SET_GPR_IDX_*` controls. The gaps above are about prose-only semantics and
  legality metadata, not missing base opcode records.
- The CDNA3 XML carries generic even-alignment text for 64-bit SGPR and VGPR
  operand values. The packed 32-bit XML gap is about VOP3P-specific dword
  selector behavior and `V_PK_MOV_B32` special cases, not absence of all
  alignment prose.
- The detailed MIX instruction pseudocode and XML descriptions both say
  multiply-add. Section 6.7 separately says the MIX multiply-add is fused, so
  that wording remains a manual-internal ambiguity rather than an XML-only
  omission from this static pass.
- The CDNA3 XML contains the dense MFMA opcode inventory and the VOP3P-MFMA
  field map, including `ACC` and `ACC_CD`. The dense MFMA gaps above are about
  missing semantics, formulas, and legality constraints, not missing opcode
  records.
- The CDNA3 XML contains the audited SMFMAC opcode inventory and records
  `SRC2` as a VGPR-only operand on the sampled sparse entries. The sparse MFMA
  gaps above are about missing semantic/layout metadata and one I8 operand
  format contradiction, not missing SMFMAC opcode records.
- The CDNA3 XML records the audited SMEM field inventory, `SOFFSET_EN`, and
  signed immediate-offset handling. The SMEM gaps above are about behavior that
  is not recoverable from those field records, not missing base SMEM encoding
  coverage, except for the XML-only `S_ATC_PROBE*` rows recorded in
  `CDNA3-XML-060`.
- Except for the `S_ATC_PROBE*` source drift recorded in `CDNA3-XML-060`, the
  CDNA3 XML contains the full audited Chapter 12.6 SMEM instruction-definition
  inventory: scalar/global/scratch/buffer loads and stores, cache maintenance,
  timer, discard, scalar-buffer atomics, and scalar-global atomics.
- The CDNA3 XML records the base MUBUF and MTBUF field bit positions, including
  `OFFSET`, `OFFEN`, `IDXEN`, `VADDR`, `VDATA`, `SRSRC`, `SOFFSET`, `SC0`,
  `SC1`, `NT`, `ACC`, and the MUBUF-only `LDS` bit.
- The CDNA3 XML contains representative MTBUF formatted, MUBUF formatted, MUBUF
  raw, cache-maintenance, and buffer-atomic opcode entries for the audited
  vector-buffer slice.
- The CDNA3 XML distinguishes typed formatted buffer resource operands from raw
  buffer resource operands at a coarse data-format-name level, for example
  `FMT_RSRC_TYPED` on formatted buffer operations and `FMT_RSRC_SCRATCH` on raw
  loads/atomics. The gaps above are about missing descriptor semantics rather
  than total absence of resource operand metadata.
- The CDNA3 XML records the raw FLAT/GLOBAL/SCRATCH field layouts, segment
  encodings, `SADDR`, `SC0`, `SC1`, `NT`, `ACC`, and offset-width differences
  for the audited flat-memory slice.
- The CDNA3 XML contains the audited FLAT, GLOBAL, and SCRATCH opcode
  inventories for Chapter 12.15/13.6. GLOBAL and SCRATCH both matched exactly;
  FLAT's only inventory-script mismatch is the manual Markdown's fenced
  `FLAT_LOAD_UBYTE 16` heading at `cdna3/README.md:21060`, not a real opcode
  omission. Manual/XML table counts are FLAT 54, GLOBAL 59, and SCRATCH 27.
- The Chapter 12.15 opening legend mentions `NV` and omits `ACC`; related
  Chapter 10/13.6/XML field drift is tracked in `CDNA3-XML-093`. This does not
  indicate a missing flat/global/scratch opcode row.
- The CDNA3 XML records the raw DS field layout and full `ENC_DS` instruction
  inventory for the audited Chapter 12.12 slice, including `ACC`, `GDS`, offset
  fields, source/destination fields, load/store/atomic opcodes, late ADDTID and
  B96/B128 rows, and the six GWS opcode entries. The apparent XML-only rows
  `DS_WRAP_RTN_B32`, `DS_ADD_RTN_F32`, `DS_READ_ADDTID_B32`,
  `DS_PK_ADD_RTN_F16`, and `DS_PK_ADD_RTN_BF16` are present in the manual body
  but formatted as fenced-code headings rather than normal bold headings.
- The CDNA3 XML carries useful coarse DS descriptions for narrow
  sign/zero-extending reads, ST64 offset scaling, ADDTID, APPEND/CONSUME, and
  the non-LDS-memory nature of SWIZZLE/PERMUTE/BPERMUTE. The DS gaps above are
  about opcode-table drift, formula, side-effect, address, and mode metadata
  rather than total absence of instruction records.
- The CDNA3 Chapter 12.13 MUBUF and Chapter 12.14 MTBUF opcode inventories
  match the checked-in XML exactly: 73 MUBUF rows and 16 MTBUF rows, with no
  manual-only rows, XML-only rows, opcode mismatches, or duplicate opcode keys.
- The CDNA3 Chapter 13.5.1/13.5.2 MTBUF/MUBUF opcode tables also match the
  checked-in XML opcode inventory. Chapter 12.14's introductory `LDS` line
  conflicts with the Chapter 9 shared buffer microcode table's `MUBUF-ONLY`
  `LDS` row and the XML MTBUF encoding, which has no `LDS` bit; this audit
  treats that as manual preamble/table inconsistency rather than an XML
  omission.
- The CDNA3 XML records implicit SCC inputs and outputs on the audited scalar
  ALU instructions, including arithmetic writers, compare writers, conditional
  SCC consumers, and no-SCC operations such as `S_BFM_{B32,B64}`. The SCC gap
  above is about the missing max tie predicate, not total absence of SCC
  operand metadata.
- The CDNA3 XML does carry enough HWREG structure to parse the raw
  `{ID, OFFSET, SIZE}` fields for `S_GETREG_B32`, `S_SETREG_B32`, and
  `S_SETREG_IMM32_B32`. The access-instruction gaps above are about missing
  register semantics and contradictions, not total absence of the field
  partition.
- The CDNA3 XML includes useful coarse special-register operands such as
  `OPR_FLAT_SCRATCH`, `XNACK_MASK_LO`/`HI`, `OPR_VCC`, `SRC_EXECZ`,
  `SRC_VCCZ`, and `SRC_SCC`, plus generic SGPR/VGPR/AccVGPR names and
  wide-value alignment prose. The Chapter 3 gaps above are about missing
  state-field, allocation, aliasing, update, and out-of-range semantics, not
  total absence of special-register names.
