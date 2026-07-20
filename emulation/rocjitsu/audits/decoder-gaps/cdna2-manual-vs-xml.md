# CDNA2 Manual vs XML Gaps

Architecture: CDNA2

Manual source: `workspace_docs/amdgpu-isa-manuals/cdna2/README.md`

XML source: `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml`

## Coverage

| Manual section | Status | Notes |
| --- | --- | --- |
| Feature changes plus 1.1 terminology and 2.1 through 2.3 Program Organization | Audited statically | Checked the MI200 feature-change summary, host/command-processor/memory-controller overview, dispatch/workgroup/wavefront/work-item terms, 64-lane wavefront model, SALU/VALU roles, EXEC-masked vector execution, dispatch grid/work-item indexing, LDS/GWS topology, and Section 2.3 device-memory consistency/acknowledgment prose against XML global metadata and instruction descriptions. |
| 3.1 through 3.6 Kernel State | Audited for kernel-state opening slice | Checked state inventory, PC/EXEC/STATUS/MODE prose, status and mode bitfields, GPR/LDS allocation and aliasing rules, out-of-range behavior, and LDS allocation/clamping. |
| 3.7 through 3.12 Kernel State | Audited for special-state remainder | Checked M0 descriptor roles, SCC summary, VCC/VCCZ update and alias-hazard prose, trap/exception/TRAPSTS/memory-violation state, HW_ID bitfields, and compute GPR initialization. |
| 4.1 through 4.2 Program Flow Control | Audited statically | Checked control/branch instruction inventory, direct PC operations, trap return, debug conditional branches, calls, VSKIP, and fork/join branch entries against XML opcode, operand, flag, and description metadata. |
| 4.3 Workgroups | Audited for workgroup/barrier slice | Checked workgroup wavefront limits, `S_BARRIER` release behavior, early-terminated wave handling, and `STATUS.IN_BARRIER` against XML opcode metadata. |
| 4.4 Data Dependency Resolution | Audited for wait-counter slice | Checked `VM_CNT`, `LGKM_CNT`, and `EXP_CNT` producer/decrement/ordering prose against `S_WAITCNT` and `OPR_WAITCNT` XML metadata. |
| 4.5 Manually Inserted Wait States (NOPs) | Audited for wait-state hazard slice | Checked the required software-inserted wait-state table against XML opcode, operand, and description metadata. |
| 4.6 Arbitrary Divergent Control Flow | Audited statically | Checked `S_CBRANCH_{G,I}_FORK` and `S_CBRANCH_JOIN`, compiler-emitted fork/join block structure, CSP/branch-stack requirements, pass/fail mask selection, EXEC/PC update pseudocode, and XML branch/operand metadata. |
| 5.1 through 5.2 Scalar ALU Formats and Operands | Audited statically | Checked SALU format fields, scalar operand selector table, literal availability/exceptions, source/destination out-of-range rules, and 64-bit SGPR alignment against XML format and operand definitions. |
| 5.3 through 5.7 Scalar ALU Data Operations | Audited statically | Checked SCC producer/consumer prose, arithmetic, conditional, compare, and bitwise instruction descriptions, implicit SCC operands, and `S_MAX_{I32,U32}` tie behavior against representative XML instruction entries. |
| 5.8 Access Instructions | Audited statically | Checked HWREG layout, register ID table, access permissions, `S_SETREG` spacing prose, `S_SETREG_IMM32_B32` literal form, and access-instruction operand metadata. |
| 8.1 through 8.4 Scalar Memory Operations | Audited statically | Checked SMEM field layout, IMM/SOE offset forms, scalar/scratch/buffer addressing, source-overwrite and clause prose, atomics, cache/time/probe/discard behavior, LGKM counter accounting, alignment, and bounds behavior against XML SMEM fields, operand classes, and representative instruction entries. |
| 9.1 Vector Memory Buffer Instructions | Audited statically | Checked MTBUF/MUBUF fields, VGPR address/data usage, data-format routing, buffer resource fields, range checking, swizzled addressing, D16 behavior, Dword alignment, LDS-destination buffer loads, GLC/SLC/cache behavior, and buffer atomics against XML MUBUF/MTBUF fields, resource data formats, and representative instruction entries. |
| 9.2 through 9.4.5 Vector Memory Image Instructions | Audited statically | Checked MIMG fields, no-sampler and sampler address tables, VGPR data/DMASK usage, image atomic DMASK rules, D16/A16 behavior, resource and sampler descriptors, data-format conversion, cache/status bits, and VMEM dependency prose against XML MIMG fields, image/sampler descriptor formats, and representative instruction entries. |
| 9.5 Float Memory Atomics | Audited statically | Checked float atomic issuer coverage, RNE add rounding, LDS/L2 denormal rules, NaN quieting/selection, signed-zero ordering, compare-store equality, and packed-F16 lane behavior against XML DS, MUBUF, FLAT, and GLOBAL float-atomic entries. |
| 10.1 through 10.8 Flat Memory Instructions | Audited statically | Checked FLAT/GLOBAL/SCRATCH field behavior, segment naming, flat/private/shared aperture routing prose, scratch and global address forms, direct LDS movement, atomics, ordering, wait-counter behavior, memory-error policy, and data movement against XML flat/global/scratch fields and representative opcode entries. |
| 11.1 through 11.4 Data Share Operations | Audited statically | Checked LDS geometry/dataflow, direct LDS reads, indexed load/store and atomic address rules, M0 clamp prose, two-offset behavior, permute/bpermute lane rules, ADDTID, append/consume, GDS/GWS restrictions, and DS field metadata against XML DS fields, operand classes, and representative opcode entries. |
| 6.1 through 6.6 Vector ALU Operations | Audited statically | Checked VOP encoding roles, VALU operand/source restrictions, literal expansion, EXEC-gated writeback, FP round/denorm MODE behavior, ALU clamp/OMOD rules, VALU inventory, and M0-based VGPR indexing against XML VOP fields, operand classes, and `S_SET_GPR_IDX_*` entries. |
| Introduction FP32 Packed Math | Audited for opcode inventory only | Checked the manual's FP32 packed-math feature list. |
| 6.7 Packed Math | Audited for VOP3P packed-math slice only | Checked packed opcode inventory, MIX wording, packed 32-bit alignment/selector prose, and output-modifier support. |
| 7.1 Matrix Arithmetic Opcodes | Audited statically | Checked the MAI intro, miSIMD AccVGPR file and transfer model, A/B/C/D bank roles, VOP3P-MAI opcode families, detailed MFMA row C/D bank and pass-count metadata, denorm/MODE prose, inline-constant restrictions, and exception wording against XML metadata. |
| 7.2 Dependency Resolution: Required NOPs | Audited statically | Checked the full MAI producer/consumer wait table, XDL/DGEMM class definitions, pass-count dependent distances, forwarding exceptions, VALU/VMEM/LDS/FLAT/export overlap rules, and `V_CMPX` dependency prose against XML metadata. |
| 12 SOP1/SOP2/SOPC/SOPK Instruction Definitions | Audited for SALU/access slice | Checked representative scalar ALU and access instruction definitions, literal variants, and implicit SCC operands for the Chapter 5 slice. |
| 12.1 SOP2 Instruction Definitions | Audited statically | Checked the full SOP2 opcode inventory, literal variants, implicit SCC operands, arithmetic/carry/borrow/overflow equations, bitwise and shift rows, bitfield mask/extract rows, fork/RFE compatibility rows, absolute-difference edge examples, shifted-add carry rows, and scalar pack half-selection rows against XML instruction entries. |
| 12.2 SOPK Instruction Definitions | Audited statically | Checked the full SOPK opcode inventory including the opcode 19 hole and literal-only opcode 20 form, SIMM16 sign/zero extension rules, old-destination ADDK/MULK dataflow, fork/call PC formulas, and HWREG access rows against XML instruction entries. |
| 12.3 SOP1 Instruction Definitions | Audited statically | Checked the full SOP1 opcode inventory, literal variants, SCC-producing unary rows, bit-count/bit-scan edge examples, bitset old-destination rows, direct PC/RFE rows, saveexec/wrexec formulas, quad-mask/bitreplicate rows, relative SGPR addressing, fork/join compatibility, and GPR-index control row against XML instruction entries. |
| 12.4 SOPC Instruction Definitions | Audited statically | Checked the full SOPC opcode inventory, literal variants, compare and bit-compare formulas, VSKIP state row, `S_SET_GPR_IDX_ON` raw `SIMM4` semantics, and implicit M0/SCC metadata against XML instruction entries. |
| 12.5 SOPP Instruction Definitions | Audited statically | Checked the full SOPP opcode inventory, PC-relative branch formulas, wait-count field layout, status/sleep/priority/message/trap/cache/perf rows, GPR-index control rows, `S_ENDPGM*` variants, send-message payload table, and SOPP opcode table against XML instruction entries. |
| 12.6 SMEM Instruction Definitions | Audited statically | Checked the full SMEM opcode inventory, scalar/scratch/buffer load and store rows, offset-form details, cache/time/probe/discard rows, scalar and buffer atomic pseudocode, 64-bit atomic forms, and opcode-table coverage against XML instruction entries. |
| 12.7 through 12.9 VOP2/VOP1/VOPC Instruction Definitions | Audited statically | Checked VOP2 FP min/max edge rules, carry forms, literal/DPP/SDWA/VOP3 availability, VOP1 readfirstlane/lookup/conversion/exception-state rows, VOPC predicate/class-mask tables, `CMPX` side effects, and VOP3A compare `CLAMP` behavior against XML opcode, operand, and field metadata. |
| 12.13 MUBUF Instruction Definitions | Audited statically | Checked the full MUBUF opcode inventory, formatted/unformatted load/store rows, D16 and D16_HI rows, cache maintenance, LDS store, integer atomic, floating atomic, 64-bit atomic definitions, and Chapter 13.5 opcode-table cross-check. |
| 12.14 MTBUF Instruction Definitions | Audited statically | Checked the full MTBUF opcode inventory, typed load/store rows, D16 typed load/store rows, field descriptions, and Chapter 13.5 field/opcode-table cross-check. |
| 12.15 MIMG Instruction Definitions | Audited statically | Checked the full MIMG opcode inventory, bitfield-map aliases, MIP opcode miplevel restrictions, load/store/PCK/SGN semantics, `IMAGE_GET_RESINFO` return shape, integer atomic pseudocode, and `IMAGE_SAMPLE` row against XML instruction entries. |
| 12.16 FLAT/GLOBAL Instruction Definitions | Audited statically | Checked the full FLAT, Scratch, and Global opcode inventories, field summary, load/store rows, D16 and D16_HI rows, integer atomic rows, float atomic rows, 64-bit atomic rows, and Chapter 13.7 field/opcode-table cross-check. |
| 12.17 Instruction Limitations | Audited statically | Checked the DPP and SDWA exclusion lists against XML per-instruction DPP/SDWA encodings and wildcard compare-family coverage. |
| 12.12 LDS/GDS Instruction Definitions | Audited statically | Checked the full LDS/GDS opcode inventory, field map, RTN old-value rule, load/store/atomic rows, READ2/WRITE2, ADDTID, append/consume, GWS, permute/bpermute/swizzle including rotate/FFT/basic modes, B96/B128 rows, and opcode-table coverage against XML instruction entries. |
| 12.10 VOP3P Instructions | Audited statically | Checked MIX, DOT, packed 16-bit arithmetic, packed F32 arithmetic, `V_PK_MOV_B32`, and the full MFMA/ACCVGPR opcode-table inventory against XML opcode, operand, field, and description metadata. |
| 12.11 VOP3A and VOP3B Instructions | Audited for VOP3A/VOP3B definition slice | Checked VOP3A/VOP3B opcode inventory, VOP3B scalar-destination forms, division helper numerics, native and legacy F16 OPSEL destination rules, `V_PERM_B32`, packed conversions, F64 min/max, `V_READLANE_B32`, `V_WRITELANE_B32`, and `V_TRIG_PREOP_F64`. |
| 13.1 Scalar ALU and Control Formats | Audited for SALU/access slice | Checked SOP1/SOP2/SOPK/SOPC/SOPP field maps, scalar literal encodings, `OPR_SDST`, `OPR_SSRC`, `OPR_HWREG`, and `OPR_WAITCNT` metadata for the Chapter 5 and 12.5 slices. |
| 13.2 Scalar Memory Format | Audited for scalar-memory slice | Checked the SMEM field map, SBASE/SDATA/OFFSET/SOFFSET operand roles, and opcode table for the Chapter 8 slice and Chapter 12.6 SMEM definition pass. |
| 13.3 Vector ALU Formats | Audited for VOP1/VOP2/VOPC/VOP3/VOP3P/VOP3P-MAI field inventory | Checked VOP1, VOP2, VOPC, and VOP3 base format metadata, VOP3A `ABS`/`OPSEL`/`CLMP`/`OMOD`/`NEG`, VOP3B `SDST`, `OPSEL_HI2`, `OPSEL_HI`, `CLMP`, VOP3P-MAI `CBSZ`/`ABID`/`ACC`/`ACC_CD`/`BLGP`, and opcode table coverage. |
| 13.3.7 through 13.3.9 SDWA, SDWAB, and DPP Formats | Audited statically | Checked SDWA/SDWAB/DPP second-dword field maps, selector value tables, scalar-destination controls, `DPP_CTRL` enumeration, row/bank mask prose, bounds-control wording, and VOPC DPP availability against checked-in XML encoding records. |
| 13.5 MTBUF/MUBUF Formats | Audited statically | Checked MTBUF/MUBUF field maps, opcode tables, and cache/LDS field descriptions for the Chapter 9 slice and the Chapter 12.13/12.14 full definition pass. |
| 13.6 Vector Memory Image Format | Audited for image slice | Checked the MIMG field map and opcode table for the Chapter 9 image-instruction slice and Chapter 12.15 MIMG definition pass. |
| 13.7 Flat Memory Format | Audited statically | Checked FLAT/GLOBAL/SCRATCH field maps, `OFFSET`, `LDS`, `SEG`, `GLC`, `SLC`, reserved bit, `ADDR`, `SADDR`, `ACC`, and `VDST` descriptions plus opcode-table coverage for the Chapter 10 slice and the Chapter 12.16 full definition pass. |
| 13.4 LDS and GDS Format | Audited statically | Checked DS field map, `OFFSET0`/`OFFSET1`, `GDS`, `OP`, `ACC`, `ADDR`, `DATA0`, `DATA1`, and `VDST` operand roles plus opcode-table coverage for the Chapter 11 and Chapter 12.12 data-share passes. |
| Remaining CDNA2 manual sections | Not started | Full chapter-by-chapter audit remains. |

## Gaps

### CDNA2-XML-001: Packed F32 VOP3P operand widths contradict the manual

Manual evidence:

- CDNA2 section 6.7 says packed 32-bit instructions operate on two dwords at a
  time, require two-dword alignment, do not support output modifiers, and use
  `OPSEL`/`OPSEL_HI` to select the first or second dword for each source at
  `cdna2/README.md:1504` through `:1526`.
- `V_PK_FMA_F32`, `V_PK_MUL_F32`, and `V_PK_ADD_F32` read source dwords
  `[31:0]` and `[63:32]` in the detailed instruction table at
  `cdna2/README.md:4150` through `:4156`.

XML evidence:

- The CDNA2 XML entries for `V_PK_FMA_F32`, `V_PK_MUL_F32`, and
  `V_PK_ADD_F32` use 32-bit operands at `amdgpu_isa_cdna2.xml:65825`,
  `:65872`, and `:65913`.
- `V_PK_MOV_B32` is encoded with 64-bit source and destination operands at
  `amdgpu_isa_cdna2.xml:65954`, showing the XML can represent 64-bit packed
  dword operands when the metadata includes them.
- The generic VOP3P `OP_SEL`/`OP_SEL_HI` field descriptions still describe
  16-bit half selection at `amdgpu_isa_cdna2.xml:2035` through `:2055`.

Impact:

XML-only consumers see ordinary 32-bit operands for the packed F32 arithmetic
opcodes even though the manual says those opcodes consume and produce two-dword
pairs and use selector bits at dword granularity.

### CDNA2-XML-002: MIX-specific VOP3P selector and modifier overloads are prose-only

Manual evidence:

- Section 6.7 says `V_MAD_MIX_*` uses VOP3P but is not packed math at
  `cdna2/README.md:1519` through `:1524`.
- Each MIX instruction says source size/location is controlled by `OPSEL`,
  choosing between full FP32, low FP16, and high FP16 inputs, and that `NEG_HI`
  acts as an absolute-value modifier at `cdna2/README.md:4135` through `:4137`.

XML evidence:

- Generic VOP3P says `NEG_HI` negates the high operation at
  `amdgpu_isa_cdna2.xml:2015`, and says `OP_SEL`/`OP_SEL_HI` choose lower or
  upper 16-bit inputs at `:2035` through `:2055`.
- `V_MAD_MIX_F32`, `V_MAD_MIXLO_F16`, and `V_MAD_MIXHI_F16` are present at
  `amdgpu_isa_cdna2.xml:65355`, `:65402`, and `:65449`, but their entries do
  not encode the MIX-only selector mapping or `NEG_HI` absolute-value behavior.

Impact:

The generic XML field descriptions imply the wrong interpretation for MIX. XML
consumers need a hard-coded MIX override or manual-derived metadata.

### CDNA2-XML-003: MFMA C/D register-bank requirements are not represented per opcode

Manual evidence:

- Chapter 7 says A and B sources can come from Arch or AccVGPR, while the
  accumulate C input and D output use AccVGPRs, at `cdna2/README.md:1541`
  through `:1545`.
- The detailed MFMA opcode table marks ordinary F32, F16, I8, and BF16 rows as
  using AccVGPR C/D operands at `cdna2/README.md:4158` through `:4189`.
- The same table marks the BF16_1K and F64 rows as using ordinary VGPR C/D
  operands at `cdna2/README.md:4180` through `:4184` and `:4190` through
  `:4191`.
- The VOP3P-MAI field table only says `ACC_CD` indicates that SRC-C and VDST
  use AccVGPRs at `cdna2/README.md:6635`; it does not make the per-opcode
  C/D bank requirements machine-readable.

XML evidence:

- The VOP3P-MFMA encoding describes `ACC_CD` generically as selecting
  accumulator VGPR use for SRC-C and VDST at
  `amdgpu_isa_cdna2.xml:7365` through `:7370`.
- Representative ordinary F32, BF16_1K, and F64 entries all use
  `OPR_VGPR_OR_ACCVGPR` for VDST and `OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST` for
  SRC2 at `amdgpu_isa_cdna2.xml:65995` through `:66028`,
  `:66776` through `:66808`, and `:67246` through `:67280`.

Impact:

XML-only consumers cannot distinguish ordinary MFMA rows whose detailed table
requires AccVGPR C/D from BF16_1K and F64 rows whose detailed table requires
ordinary VGPR C/D. That makes `ACC_CD` legality/defaulting a prose-only rule.

### CDNA2-XML-004: MFMA floating-point mode, denorm, exception, and inline-constant rules are prose-only

Manual evidence:

- The MFMA opcode-family table says F32/F32 supports denorm allow/flush from
  `MODE.denorm`, F16/BF16/BF16_1K flush input and output denorms, and F64
  ignores `MODE` while forcing round-to-nearest-even and allowing input/output
  denorms at `cdna2/README.md:1574` through `:1582`.
- The manual lists unsupported MFMA source forms and restricts the accepted
  inline constants for all `V_MFMA` and `V_ACCVGPR` instructions to FP32
  `0.5`, `-0.5`, `1.0`, `-1.0`, `2.0`, `-2.0`, `4.0`, and `-4.0` at
  `cdna2/README.md:1584` through `:1589`.
- The later generic VOP3P-MAI field table lists the broader source-encoding
  options, including integer constants and `1/(2*PI)`, at
  `cdna2/README.md:6638` through `:6640`; the narrower Chapter 7 MFMA rule is
  not carried as per-instruction metadata.
- The manual states that miSIMD does not support arithmetic exceptions at
  `cdna2/README.md:1591`.

XML evidence:

- Representative MFMA entries carry only operand data format, operand type, and
  size metadata at `amdgpu_isa_cdna2.xml:65995` through `:66028`,
  `:66776` through `:66808`, and `:67246` through `:67280`; they do not encode
  per-family `MODE.denorm`, forced-denorm, rounding, or exception behavior.
- SRC2 uses the broad `OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST` class, whose generic
  predefined values include the usual scalar inline constants at
  `amdgpu_isa_cdna2.xml:108772` and following, rather than the MFMA-specific
  subset from Chapter 7. This follows the generic VOP3P-MAI field table, not
  the more specific MFMA prose.

Impact:

Generated decoders cannot derive MFMA denorm, rounding, exception, or
MFMA-only inline-constant legality from the XML. Those rules require a
manual-derived override.

### CDNA2-XML-005: MFMA dependency-wait rules and pass counts are prose-only

Manual evidence:

- Section 7.2 gives required NOP or independent VALU distances for VALU,
  XDL, DGEMM, VMEM/LDS/FLAT/Export, `V_CMPX`, and MFMA producer/consumer
  combinations at `cdna2/README.md:1593` through `:1648`.
- The detailed opcode table includes block counts and pass counts for each MFMA
  shape at `cdna2/README.md:4158` through `:4191`; the dependency table uses
  those pass counts to choose required wait distances.

XML evidence:

- The CDNA2 XML VOP3P-MFMA encoding records the opcode identifiers and field
  layout at `amdgpu_isa_cdna2.xml:7295` through `:7392`, but does not attach
  the manual's producer/consumer classes, pass counts, forwarding exceptions,
  or required-wait distances to the opcodes.

Impact:

XML-only consumers can decode an MFMA instruction, but they cannot derive the
software scheduling contract that the manual assigns to MFMA producer/consumer
sequences.

### CDNA2-XML-006: VOP3P-MAI `CBSZ` legality is under-described

Manual evidence:

- The VOP3P-MAI field table says `CBSZ` defines the number of blocks that can
  broadcast within a group, and that legal values are `0-4`, at
  `cdna2/README.md:6633`.
- The same table describes `ABID` as the block ID to broadcast during MFMA and
  `BLGP` as the B-matrix lane-group pattern at `cdna2/README.md:6634` through
  `:6642`.

XML evidence:

- The XML field descriptions for `ABID`, `CBSZ`, and `BLGP` describe the broad
  swizzle controls at `amdgpu_isa_cdna2.xml:7345` through `:7390`, but the
  `CBSZ` field is only a 3-bit field with no encoded legal-value restriction.

Impact:

Consumers generated from XML alone accept all 3-bit `CBSZ` values, while the
manual only defines `0-4` for CDNA2 VOP3P-MAI.

### CDNA2-XML-007: Program organization, GDS/GWS retention, and device-memory consistency are prose-only

Manual evidence:

- The MI200 feature-change summary says CDNA2 removes GDS operations while
  retaining GWS operations, and merges compute shader thread indices into a
  single VGPR, at `cdna2/README.md:295` through `:303`.
- Chapter 1 describes the CDNA command processor, memory controller, host/kernel
  split, cache invalidation/flush commands, DMA-style memory-controller
  behavior, hardware interrupts, floating-point exception detection, automatic
  instruction fetch, and latency hiding at `cdna2/README.md:350` through
  `:378`.
- Section 1.1 defines dispatch, workgroup, 64-work-item wavefronts, work-items,
  literal constants, SALU, VALU, and microcode-format terminology at
  `cdna2/README.md:380` through `:404`.
- Chapter 2 says kernels are grouped into 64-work-item wavefronts, control flow
  is handled by SALU instructions, vector ALU and vector-memory work is gated
  by `EXEC`, and vector compare/carry-out results return bit-per-work-item
  masks to SGPRs at `cdna2/README.md:410` through `:426`.
- Sections 2.1 through 2.3 describe 1D/2D/3D dispatch grids, per-work-item grid
  indices, LDS topology, the global wave synchronization unit, cache-less loads,
  load-clause overlap behavior, atomic return acknowledgments, relaxed
  consistency, and per-PE/per-channel scatter-write ordering at
  `cdna2/README.md:428` through `:463`.

XML evidence:

- The top-level XML architecture record only carries the architecture name and
  numeric ID at `amdgpu_isa_cdna2.xml:1` through `:13`.
- XML has useful local descriptions, such as VOPC compare `EXEC` wording at
  `amdgpu_isa_cdna2.xml:1602`, DS `GDS` field text at `:2868` through `:2869`,
  `OPR_DSMEM` at `:92254` through `:92255`, and functional-group descriptions
  at `:121498` through `:121513`.
- Those records still do not provide global metadata for wavefront size,
  dispatch-grid dimensionality, SALU/VALU control roles, EXEC-mask scope,
  command-processor host interaction, floating-exception recording, or the
  Section 2.3 memory consistency/acknowledgment/order model.
- The CDNA2-specific "remove GDS operations, retain GWS operations" caveat is
  not machine-readable. XML still exposes the generic DS `GDS` bit,
  `OPR_DSMEM` reads/writes-GDS-or-LDS description, and GDS/GWS instruction
  wording such as `DS_GWS_*` and `DS_CONSUME`/`DS_APPEND` at
  `amdgpu_isa_cdna2.xml:13460` through `:13720`.

Impact:

XML consumers can enumerate instructions, functional groups, and some operands,
but still need manual prose for processor-organization assumptions, the CDNA2
GDS-removal versus retained-GWS distinction, and the top-level memory model that
determine how decoded instructions execute together.

### CDNA2-XML-008: Kernel state registers and helper-bit semantics are mostly prose-only

Manual evidence:

- The Chapter 3 state table defines the PC, EXEC/EXECZ, VCC/VCCZ, SCC,
  FLAT_SCRATCH, XNACK_MASK, STATUS, MODE, M0, TRAPSTS, TBA/TMA, TTMPs, and
  wait-counter state at `cdna2/README.md:471` through `:505`.
- Section 3.2 describes PC initialization, PC-relative branch accounting,
  `S_GET_PC`/`S_SET_PC`/`S_SWAP_PC` even-SGPR-pair transfer, and `S_TRAP` saved
  PC behavior at `cdna2/README.md:507` through `:513`.
- Sections 3.3 through 3.5 define EXEC scope, `EXECZ`, the full STATUS field
  table, and the MODE field table including `FP16_OVFL`, `GPR_IDX_EN`,
  `VSKIP`, and `CSP` at `cdna2/README.md:515` through `:579`.

XML evidence:

- `OPR_HWREG` encodes `ID`, `OFFSET`, and `SIZE`, but its operand and field
  descriptions are `N/A` and only the register IDs are enumerated at
  `amdgpu_isa_cdna2.xml:99170` through `:99303`.
- `S_GETREG_B32`, `S_SETREG_B32`, and `S_SETREG_IMM32_B32` carry HWREG
  operands and short descriptions at `amdgpu_isa_cdna2.xml:44148` through
  `:44243`, but do not encode STATUS/MODE bit meanings, reset/init rules,
  read-only versus writable fields, helper-bit update rules, or PC side
  effects.
- The XML includes local `SRC_VCCZ`, `SRC_EXECZ`, and `SRC_SCC` descriptions at
  `amdgpu_isa_cdna2.xml:101953` through `:101964`, but not the manual's
  producer/update contracts for those bits.

Impact:

XML-only consumers can parse HWREG selectors and special source names, but need
manual-derived metadata for the wave-state bit layouts, update rules,
initialization, writability, and PC/control-flow side effects.

### CDNA2-XML-009: GPR and LDS allocation, aliasing, and out-of-range behavior are prose-only

Manual evidence:

- Section 3.6.1 defines SGPR, VGPR, LDS, memory-return, and multiple-destination
  out-of-range behavior, including SGPR0/VGPR0 fallback, destination NOP or
  nullification, and all-returned-VGPR checks at `cdna2/README.md:581` through
  `:623`.
- Section 3.6.2 defines SGPR allocation granularity, logical SGPR limits, VCC
  physical aliasing at SGPR 106/107, and privileged trap-handler SGPR reserves
  at `cdna2/README.md:625` through `:627`.
- Sections 3.6.3 through 3.6.5 define SGPR alignment, VGPR allocation pools,
  the manual's VGPR-pair alignment statements, packed VGPR0 work-item IDs, LDS
  allocation as 128-dword aligned blocks, and LDS clamping by the smaller of
  SPI allocation size and M0 at `cdna2/README.md:629` through `:656`.

XML evidence:

- The XML gives generic SGPR, AccVGPR, and VGPR operand names and alignment
  notes, for example `OPR_SDST` at `amdgpu_isa_cdna2.xml:99335` through
  `:99349`, `OPR_ACCVGPR` at `:97850` through `:97857`, and `OPR_VGPR` at
  `:123288` through `:123296`.
- Those operand definitions do not encode per-wave allocation counts,
  SGPR/VCC/TTMP aliasing and privilege, SGPR0/VGPR0 out-of-range fallback,
  destination nullification, all-returned-VGPR checks, LDS allocation
  granularity, or the M0/SPI LDS clamp rule.

Impact:

The XML describes register selector spaces but not the runtime allocation and
out-of-range contract that determines how decoded operands execute for a
particular wavefront.

### CDNA2-XML-010: M0, SCC, and VCC/VCCZ behavior is prose-only

Manual evidence:

- Section 3.7 lists the distinct M0 payload layouts for LDS interpolation,
  direct LDS reads, memory-to-LDS, GWS, indirect GPR indexing, and send-message
  data at `cdna2/README.md:658` through `:669`.
- Section 3.8 summarizes SCC producer classes, says moves do not alter SCC, and
  describes SCC as carry-in and conditional selector state at
  `cdna2/README.md:671` through `:681`.
- Section 3.9 says vector compares can write VCC or EXEC, VCCZ updates when
  VCC updates including scalar writes to VCC, VCC is fully written, VCC aliases
  the highest two user SGPRs, and scalar writes to physical VCC have a VCCZ
  branch hazard at `cdna2/README.md:683` through `:699`.

XML evidence:

- The XML names `SRC_SCC` as a scalar condition code at
  `amdgpu_isa_cdna2.xml:123266` through `:123272`, `vcc` as `VCC[63:0]` at
  `:123277` through `:123283`, and `SRC_VCCZ`/`SRC_EXECZ` at `:101953`
  through `:101959`.
- `S_SENDMSG` records an implicit M0 operand at `amdgpu_isa_cdna2.xml:44831`
  through `:44849`, but the XML does not encode the section 3.7 M0 payload
  layouts or the SCC/VCC producer, aliasing, full-write, helper-bit update, and
  hazard rules.

Impact:

XML-derived operand metadata can name the special registers, but it cannot
derive the state-transition rules needed for execution, validation, or hazard
diagnostics.

### CDNA2-XML-011: Trap, exception, memory-violation, and HW_ID state is not machine-readable

Manual evidence:

- Section 3.10 describes TTMP write privilege, TMA/TBA read-only access, trap
  payload construction in TTMP1/TTMP0, `STATUS.TRAP_EN`, and
  `MODE.EXCP_EN[8:0]` at `cdna2/README.md:701` through `:728`.
- Section 3.10.1 defines TRAPSTS sticky exception fields, `SAVECTX`,
  `ILLEGAL_INST`, address-watch bits, `EXCP_CYCLE`, and `DP_RATE` at
  `cdna2/README.md:730` through `:743`.
- Sections 3.11 and 3.12 define memory-violation sources, sticky
  `TRAPSTS.mem_viol`, trap enable behavior, imprecise saved PC, and HW_ID field
  packing at `cdna2/README.md:745` through `:773`.

XML evidence:

- `OPR_HWREG` enumerates `HW_REG_TRAPSTS`, `HW_REG_HW_ID`,
  `HW_REG_SQ_SHADER_TBA_*`, and `HW_REG_SQ_SHADER_TMA_*` IDs with `N/A`
  descriptions at `amdgpu_isa_cdna2.xml:99185` through `:99278`.
- `S_RFE_B64`, `S_TRAP`, and `V_CLREXCP` have short instruction descriptions at
  `amdgpu_isa_cdna2.xml:32316` through `:32328`, `:44905` through `:44917`, and
  `:50709` through `:50716`, but the XML does not encode TRAPSTS bitfields,
  TTMP privilege, TMA/TBA access restrictions, memory-violation sources, sticky
  behavior, imprecise PC reporting, or HW_ID field packing.
- Searching the CDNA2 XML for manual field names such as `SAVECTX`,
  `MEM_VIOL`, `EXCP_CYCLE`, and `DP_RATE` finds no matching machine-readable
  field records.

Impact:

An XML consumer can identify the trap/control opcodes and HWREG IDs, but cannot
implement the Chapter 3 trap, exception, memory-violation, or hardware-ID state
contract without manual prose.

### CDNA2-XML-012: Program-control branch-stack and status side effects are under-described

Manual evidence:

- Chapter 4.1 lists control instructions that terminate, trap/return, alter
  wave priority, sleep, send messages, and wake sleeping waves at
  `cdna2/README.md:781` through `:796`.
- Chapter 4.2 lists ordinary predicate branches, debug-status conditional
  branches, PC get/set/swap, fork/join divergent-control branches, VSKIP, and
  calls at `cdna2/README.md:798` through `:823`.
- Chapter 4.6 defines compiler-emitted fork/join blocks, a six-deep CSP stack,
  stack entries containing `{EXEC, PC}`, pass/fail mask selection by population
  count, and JOIN restore behavior at `cdna2/README.md:887` through `:936`.

XML evidence:

- The XML carries useful opcode and operand entries for this slice, including
  `S_SETPC_B64`, `S_SWAPPC_B64`, and `S_RFE_B64` at
  `amdgpu_isa_cdna2.xml:32231` through `:32335`,
  `S_CBRANCH_JOIN` at `:33403` through `:33416`,
  `S_CBRANCH_G_FORK` at `:39684` through `:39709`, and
  `S_CBRANCH_I_FORK` at `:44110` through `:44130`.
- The fork/join XML descriptions only say "Conditional branch using
  branch-stack" or "Conditional branch join point"; the entries do not expose
  CSP storage, the six stack entries, SGPR stack read/write side effects,
  pass/fail mask selection, or the conditions that decide when PC/EXEC are
  restored.
- `S_RFE_B64`'s text mentions clearing PRIV and `S_SENDMSGHALT`'s text
  mentions HALT at `amdgpu_isa_cdna2.xml:32316` through `:32335` and
  `:44868` through `:44886`, but those status writes are not represented as
  structured implicit operands or state effects.

Impact:

XML-only code can decode the Chapter 4.1/4.2 opcodes, but cannot build an
accurate control-flow emulator or side-effect verifier for fork/join, debug
branch predicates, trap return, message-halt, priority, or wake/sleep behavior
without manual semantics.

### CDNA2-XML-013: Workgroup barrier membership and wait-counter semantics are prose-only

Manual evidence:

- Chapter 3 defines `STATUS.IN_BARRIER` as the bit indicating a wavefront is
  waiting at a barrier at `cdna2/README.md:525` through `:541`.
- Chapter 4.3 says workgroups may contain up to 16 wavefronts or 1024
  work-items, that `S_BARRIER` waits until all other wavefronts reach the same
  instruction, and that early `S_ENDPGM` termination leaves only the remaining
  live waves to satisfy the barrier at `cdna2/README.md:825` through `:827`.

XML evidence:

- The checked-in XML `S_BARRIER` entry only carries the SOPP opcode and the
  short description "Synchronize waves within a threadgroup" at
  `amdgpu_isa_cdna2.xml:44646` through `:44653`.
- The entry has no operands, implicit status effects, workgroup-membership
  metadata, live-wave membership rule, workgroup-size limit, or wait-counter
  interaction rule.

Impact:

The XML can identify `S_BARRIER`, but it cannot drive barrier scheduling,
status-bit reporting, workgroup-size validation, or exact barrier release
behavior from machine-readable metadata alone.

### CDNA2-XML-014: Wait-counter producer, decrement, and ordering semantics are prose-only

Manual evidence:

- Chapter 4.4 says `S_WAITCNT` waits for `VM_CNT`, `LGKM_CNT`, and `EXP_CNT`
  to be at or below specified thresholds, and that same-type operations return
  in order while different instruction types can return out of order at
  `cdna2/README.md:829` through `:837`.
- The same section defines `VM_CNT` producer/decrement rules for MIMG, MUBUF,
  and MTBUF vector-memory reads and writes at `cdna2/README.md:839` through
  `:844`.
- It defines `LGKM_CNT` producer/decrement rules for LDS, GDS, scalar-memory,
  `S_SENDMSG`, and SMEM return dwords, including scalar-memory out-of-order
  return restrictions, at `cdna2/README.md:845` through `:852`.
- It defines `EXP_CNT` as the GDS VGPR-export count, including GDS issue and
  grant/decrement points at `cdna2/README.md:853` through `:855`.

XML evidence:

- The checked-in XML `S_WAITCNT` entry only gives a generic description and the
  SOPP `SIMM16`/`OPR_WAITCNT` operand at `amdgpu_isa_cdna2.xml:44703` through
  `:44715`.
- The `OPR_WAITCNT` partition records the `EXP`, `LGKM`, and `VM` threshold
  bit fields and "do not wait" encodings at `amdgpu_isa_cdna2.xml:128441`
  through `:128480`.
- The XML does not record which instruction families increment or decrement
  each counter, scalar-memory dword-count behavior, GDS/GWS `EXP_CNT`
  behavior, `S_SENDMSG` accounting, or the scalar-memory `S_WAITCNT 0`
  ordering restriction.

Impact:

The XML is sufficient to decode and print a wait-count operand, but it cannot
drive architectural wait-counter accounting, hazard modeling, or tests for the
producer-specific ordering rules without manual-specific policy.

### CDNA2-XML-015: Software-inserted wait-state hazard table is prose-only

Manual evidence:

- Chapter 4.5 says hardware does not check several dependency classes, so the
  shader must resolve them by inserting NOPs or independent instructions at
  `cdna2/README.md:857` through `:859`.
- Table 9 lists required waits for `S_SETREG`/`S_GETREG`, `MODE.VSKIP`,
  VALU-produced VCC/EXEC/SGPR/VGPR values, lane-select consumers,
  `V_DIV_FMAS`, wide store/atomic write-data hazards, M0 consumers,
  TRAPSTS/RFE, DPP, mixed VCC aliasing, and `S_MOVEREL` at
  `cdna2/README.md:861` through `:883`.

XML evidence:

- A targeted search of the checked-in CDNA2 XML for Table 9 phrases such as
  `Required Software`, `wait state`, `VCC can be accessed`, and `Trans op`
  returned no matches.
- Representative XML instruction entries carry opcode, operand, and short
  description metadata only: `S_GETREG_B32`/`S_SETREG_B32` at
  `amdgpu_isa_cdna2.xml:44148` through `:44243`, `S_NOP` at
  `:44310` through `:44324`, and `S_WAITCNT` at `:44703` through `:44715`.

Impact:

The XML can decode the individual instructions and `S_NOP`, but it cannot
drive a validator, scheduler, or emulator hazard model for CDNA2's manually
inserted wait-state requirements without manual-specific tables.

### CDNA2-XML-016: SALU allocation fallback and runtime SGPR-pair rules are prose-only

Manual evidence:

- Chapter 5.2 defines which SALU selectors can be used as destinations versus
  sources, gives the full scalar source table, places `POPS_EXITING_WAVE_ID` at
  selector 239, reserves 249-250, and uses selector 255 for a following 32-bit
  literal at `cdna2/README.md:980` through `:1024`.
- The same section says SALU literals are available to all SALU formats except
  SOPP and SOPK, says out-of-range source SGPRs read SGPR0, says
  out-of-range destination SGPRs suppress the SGPR write while still allowing
  SCC and saveexec side effects, and requires 64-bit SGPR data to start at an
  even SGPR at `cdna2/README.md:1026` through `:1032`.

XML evidence:

- The CDNA2 XML exposes the base literal encodings for SOP1, SOP2, SOPC, and
  SOPK-literal forms at `amdgpu_isa_cdna2.xml:3560`, `:3693`, `:3945`, and
  `:4155`.
- `OPR_SDST` and `OPR_SSRC` include useful generic 64-bit SGPR alignment prose
  at `amdgpu_isa_cdna2.xml:99335` through `:99349` and `:120008` through
  `:120026`, and `OPR_SSRC` correctly lists `SRC_POPS_EXITING_WAVE_ID` at
  selector 239 at `:123258` through `:123261`.
- The XML does not encode per-wave allocation-count fallback, destination
  write suppression, the SCC/saveexec side-effect exception for out-of-range
  destinations, or runtime validation hooks for odd 64-bit SGPR pairs.

Impact:

XML-only consumers can decode SALU operands and many literal variants, but they
still need manual-derived runtime policy for allocation-bound execution and
SGPR-pair validation.

### CDNA2-XML-017: HWREG access metadata is incomplete or partially contradictory

Manual evidence:

- Chapter 5.8 defines `S_GETREG_B32`, `S_SETREG_B32`, and
  `S_SETREG_IMM32_B32`, says consecutive `S_SETREG` writes to the same
  register require an intervening `S_NOP`, and defines the
  `{size, offset, hwRegId}` SIMM16 layout at `cdna2/README.md:1141` through
  `:1153`.
- The HWREG table marks MODE as read/write, STATUS as read-only, TRAPSTS as
  read/write, HW_ID/GPR_ALLOC/LDS_ALLOC/IB_STS as read-only, reserves IDs 8-15,
  and defines TBA/TMA IDs at `cdna2/README.md:1155` through `:1177`.
- The following tables define the `IB_STS`, `GPR_ALLOC`, and `LDS_ALLOC`
  bitfields at `cdna2/README.md:1178` through `:1201`.

XML evidence:

- `OPR_HWREG` carries the `ID`, `OFFSET`, and `SIZE` partitions and IDs 1-19,
  but the operand and field descriptions are `N/A` at
  `amdgpu_isa_cdna2.xml:99170` through `:99303`.
- The XML assigns IDs 8-15 to `PC_LO`, `PC_HI`, `INST_DW0`, `INST_DW1`,
  `IB_DBG0`, `IB_DBG1`, `FLUSH_IB`, and `SH_MEM_BASES` at
  `amdgpu_isa_cdna2.xml:99220` through `:99258`, while the checked CDNA2
  manual reserves those IDs.
- `S_GETREG_B32`, `S_SETREG_B32`, and `S_SETREG_IMM32_B32` carry the correct
  explicit operands, including the 32-bit literal operand for
  `S_SETREG_IMM32_B32`, at `amdgpu_isa_cdna2.xml:44148` through `:44243`, but
  do not encode read/write permissions, subfield layouts, access side effects,
  or the consecutive-`S_SETREG` spacing rule.

Impact:

XML consumers can parse HWREG operands and access opcodes, but need the manual
for register semantics, permissions, subfields, and scheduling restrictions;
they must also resolve the ID 8-15 discrepancy explicitly.

### CDNA2-XML-018: SMEM address post-processing rules are prose-only

Manual evidence:

- Section 8.2.1 defines the four `IMM`/`SOFFSET_EN` address forms, two-bit
  address masking, six-bit `S_DCACHE_DISCARD` masking, and the negative
  immediate-plus-offset undefined case at `cdna2/README.md:1700` through
  `:1711`.
- The buffer-constant prose says scalar buffer memory uses `base_address`,
  `stride`, `num_records`, and `NV`, uses stride only for bounds checking, and
  ignores the two low bits of the resource base and final address at
  `cdna2/README.md:1727` through `:1748`.
- The detailed SMEM table gives per-opcode offset rules, including scratch
  SGPR offsets as unsigned 64-byte offsets and discard opcodes as cache-line
  operations, at `cdna2/README.md:3199` through `:3240`.

XML evidence:

- The SMEM field map records the raw `IMM`, `OFFSET`, `SBASE`, `SOFFSET`, and
  `SOFFSET_EN` bitfields and short field descriptions at
  `amdgpu_isa_cdna2.xml:701` through `:777`.
- Representative scalar, scratch, buffer, probe, and discard entries expose
  `OPR_SREG`, `OPR_SMEM_OFFSET`, and `OPR_GPUMEM` operands at
  `amdgpu_isa_cdna2.xml:26804` through `:28196`, but the entries do not attach
  the manual's post-decode address transformations or buffer-resource bounds
  model.
- `OPR_SMEM_OFFSET` is an operand selector space, but its description fields are
  `N/A` at `amdgpu_isa_cdna2.xml:100204` through `:100216`.

Impact:

XML-only consumers can parse the SMEM instruction word, but they still need
manual-derived rules to apply low-bit masking, scratch-offset scaling, discard
cache-line alignment, buffer-resource base/stride/record bounds, `NV`, and the
negative-offset undefined case.

### CDNA2-XML-019: SMEM clause/XNACK and atomic return rules are prose-only

Manual evidence:

- Section 8.2.1 says scalar memory instructions must not overwrite their own
  sources because ATC XNACK replay is possible, and that a scalar memory clause
  is a string of same-type memory instructions broken by a nonmemory
  instruction at `cdna2/README.md:1723` through `:1725`.
- Section 8.2.2 says scalar atomics use the same addressing as scalar memory
  loads/stores and return the pre-operation value to `SDATA` when `GLC` is set
  at `cdna2/README.md:1750` through `:1756`.

XML evidence:

- The XML marks scalar and buffer atomics as SMEM/ATOMIC instructions and
  exposes their `SDATA`, `SBASE`, and `SOFFSET` operands, for example
  `S_BUFFER_ATOMIC_SWAP` at `amdgpu_isa_cdna2.xml:28210` through `:28243`.
- The generic `GLC` field description says the operation is globally coherent at
  `amdgpu_isa_cdna2.xml:691` through `:699`, but the XML entries do not encode
  the scalar-atomic pre-operation return contract, single-instruction atomic
  clause rule, or XNACK source-overwrite restrictions.

Impact:

Generated metadata can identify SMEM atomics, but it cannot derive the clause,
replay, source-overwrite, or `GLC` return-data contract without manual prose.

### CDNA2-XML-020: SMEM dependency-counter accounting is prose-only

Manual evidence:

- Section 8.3 says scalar memory reads/writes can return out of order or in
  partial cache-line results, increments `LGKM_CNT` by one for one-dword fetches
  and by two for two-or-more-dword fetches, and decrements by an equal amount on
  completion at `cdna2/README.md:1770` through `:1778`.
- `S_DCACHE_DISCARD` and `S_DCACHE_DISCARD_X2` increment `LGKM_CNT` by one and
  two respectively in the detailed opcode table at `cdna2/README.md:3239`
  through `:3240`.

XML evidence:

- The XML exposes the `S_WAITCNT` operand layout and the `VM`/`EXP`/`LGKM`
  threshold fields, but not producer-side increments or completion ordering, at
  `amdgpu_isa_cdna2.xml:128441` through `:128480`.
- SMEM instruction entries identify functional group and operands, but do not
  carry the per-dword or per-discard wait-counter contribution.

Impact:

An XML consumer can decode `S_WAITCNT` thresholds and SMEM opcodes, but cannot
derive the CDNA2 scalar-memory outstanding-count model from XML alone.

### CDNA2-XML-021: SMEM time/cache/probe details are not structured

Manual evidence:

- Sections 8.2.3 through 8.2.5 describe whole-cache invalidation/writeback,
  `S_MEMTIME`, and `S_MEMREALTIME`; `S_MEMREALTIME` uses a constant 25MHz clock
  unaffected by power modes or core clock changes at `cdna2/README.md:1758`
  through `:1768`.
- The detailed SMEM table defines `S_ATC_PROBE`, `S_ATC_PROBE_BUFFER`,
  `S_DCACHE_DISCARD`, and `S_DCACHE_DISCARD_X2` behavior at
  `cdna2/README.md:3235` through `:3240`.

XML evidence:

- The XML contains entries for cache invalidate/writeback, time, ATC probe, and
  discard opcodes at `amdgpu_isa_cdna2.xml:27882` through `:28196`.
- Those entries provide opcode and operand metadata, but not the real-time clock
  frequency/invariance, whole-cache versus volatile-line cache effects, ATC
  prefetch behavior, or discard writeback suppression and cache-line semantics
  as structured attributes.

Impact:

XML consumers need manual-derived semantics for cache maintenance, probes, and
time-counter behavior; the XML is sufficient for decode inventory but not for an
emulation contract.

### CDNA2-XML-022: MUBUF/MTBUF `OFFEN` descriptions contradict instruction-offset use

Manual evidence:

- Chapter 9.1.5 says `inst_offset` is present regardless of `OFFEN`, and
  `inst_offen` only controls whether the VGPR offset participates at
  `cdna2/README.md:1922` through `:1930`.
- The manual's buffer-addressing prose then composes the final address from the
  resource base, SGPR offset, and a buffer-offset term at
  `cdna2/README.md:1958` through `:1962`; the swizzled formula explicitly adds
  `inst_offset` to the optional VGPR offset at `:1997` through `:2008`.

XML evidence:

- The CDNA2 XML `ENC_MUBUF` `OFFEN` field says "If set, send VADDR as an
  offset. If unset, send the instruction offset stored in OFFSET. Only one of
  these offsets may be sent" at `amdgpu_isa_cdna2.xml:2481` through `:2489`.
- `ENC_MTBUF` carries the same wording at `amdgpu_isa_cdna2.xml:2671` through
  `:2679`.

Impact:

XML-derived address builders can treat the VGPR offset and instruction offset as
mutually exclusive, while the manual says the instruction offset is always part
of the buffer-offset term and `OFFEN` only adds the VGPR contribution.

### CDNA2-XML-023: MUBUF/MTBUF reserved bits are labeled `SCC`

Manual evidence:

- Chapter 9.1.2 says the MUBUF reserved bit must be zero and can return a NACK
  if it is not, at `cdna2/README.md:1860` through `:1866`.
- Chapter 13.5 marks MTBUF bit 53 as reserved and MUBUF bit 15 as reserved at
  `cdna2/README.md:7044` through `:7047` and `:7089` through `:7095`.

XML evidence:

- `ENC_MUBUF` names bit 15 `SCC` and describes it as "System level Cache
  Coherent" at `amdgpu_isa_cdna2.xml:2511` through `:2519`.
- `ENC_MTBUF` names bit 53 `SCC` with the same description at
  `amdgpu_isa_cdna2.xml:2701` through `:2709`.

Impact:

The XML exposes a reserved/NACK-causing bit as a coherent-cache field. Consumers
generated from XML alone can accept, preserve, or surface a nonexistent `SCC`
modifier instead of enforcing the reserved-zero contract.

### CDNA2-XML-024: Buffer resource descriptor, range checking, and swizzle behavior are prose-only

Manual evidence:

- Chapter 9.1.5 defines resource fields including base, stride, num-records,
  add-TID, swizzle enable, element size, and index stride at
  `cdna2/README.md:1936` through `:1949`.
- Range checking differs for private/scratch, raw, and structured buffers, with
  specific out-of-range load/store/atomic behavior at `cdna2/README.md:1968`
  through `:1992`.
- Swizzled addressing has separate index/offset decomposition, dword alignment,
  element-size constraints, and an explicit final-address formula at
  `cdna2/README.md:1993` through `:2008`.
- Chapter 9.1.8 defines the full 128-bit buffer resource descriptor and says an
  all-zero resource acts as an unbound buffer returning zeros or dropping writes
  at `cdna2/README.md:2054` through `:2084`.

XML evidence:

- XML resource data formats such as `FMT_RSRC`, `FMT_RSRC_SCRATCH`, and
  `FMT_RSRC_TYPED` are all opaque 128-bit `Descriptor` fields at
  `amdgpu_isa_cdna2.xml:97608` through `:97726`.
- Representative MUBUF/MTBUF instruction entries attach 128-bit resource
  operands, for example `TBUFFER_LOAD_FORMAT_X` and `BUFFER_LOAD_FORMAT_X` at
  `amdgpu_isa_cdna2.xml:21661` through `:21698` and `:22557` through `:22594`,
  but do not encode descriptor subfields, range-check modes, swizzle formulas,
  unbound-resource behavior, or add-TID/stride-extension rules.

Impact:

XML-only consumers can identify a resource descriptor operand, but cannot derive
the actual buffer addressing and bounds contract required for emulation,
validation, or test generation.

### CDNA2-XML-025: Buffer data-format conversion and D16 ECC details are prose-only

Manual evidence:

- Chapter 9.1.4 says MTBUF takes data and numeric formats from instruction
  fields, MUBUF format instructions use resource format and `dst_sel`, other
  MUBUF instructions derive format from opcode, and invalid resource format has
  unbound-resource meaning at `cdna2/README.md:1888` through `:1912`.
- The same section says D16 loads pack pairs of converted 16-bit values into
  32-bit VGPRs, and D16 stores take two 16-bit elements from each 32-bit VGPR at
  `cdna2/README.md:1914` through `:1916`; section 9.1.6 adds low/high half
  behavior and ECC full-register writes at `:2039` through `:2048`.

XML evidence:

- The XML instruction entries distinguish formatted, D16, D16_HI, typed, and
  raw opcodes and attach operand widths, for example
  `TBUFFER_LOAD_FORMAT_X` at `amdgpu_isa_cdna2.xml:21661` through `:21698` and
  `BUFFER_LOAD_FORMAT_X` at `:22557` through `:22594`.
- The XML does not provide structured data-format conversion tables, resource
  `dst_sel` behavior, invalid-format/unbound behavior, or the D16 ECC overwrite
  rule.

Impact:

Opcode and operand inventory alone is not enough to implement CDNA2 buffer data
conversion. Consumers need manual-derived tables for typed/resource formats,
component selection, D16 packing, and ECC-dependent writeback.

### CDNA2-XML-026: Buffer cache, LDS, and atomic-return side effects are under-described

Manual evidence:

- Chapter 9.1.2 overloads `GLC` by operation class: load L1 policy, store
  write-combine/persistence policy, and atomic pre-operation return behavior at
  `cdna2/README.md:1860` through `:1866`; section 9.1.10 expands those rules at
  `:2111` through `:2135`.
- Chapter 9.1.9 defines MUBUF load-to-LDS addressing and clamping rules,
  including M0-derived LDS offset, per-lane LDS writes, and LDS allocation
  clamping at `cdna2/README.md:2086` through `:2110`.
- The detailed MUBUF table defines cache maintenance and LDS store opcodes,
  including ACK-returning `BUFFER_WBL2`, `BUFFER_INVL2`, `BUFFER_WBINVL1`, and
  `BUFFER_WBINVL1_VOL`, at `cdna2/README.md:4819` through `:4828`.

XML evidence:

- XML field descriptions for `GLC`, `SLC`, and `LDS` are generic in the
  MUBUF/MTBUF field maps at `amdgpu_isa_cdna2.xml:2451` through `:2479` and
  `:2641` through `:2650`.
- `BUFFER_STORE_LDS_DWORD` and cache-maintenance opcodes are present in the XML,
  for example `BUFFER_STORE_LDS_DWORD` at `amdgpu_isa_cdna2.xml:24825` through
  `:24854`, but the entries do not encode ACK return, LDS allocation clamping,
  or operation-specific cache policy.

Impact:

XML consumers can decode the bits and opcodes, but need the manual to model
cache side effects, atomics' `GLC` return behavior, and LDS-destination buffer
loads/stores.

### CDNA2-XML-027: MIMG field descriptions drift from the manual

Manual evidence:

- Chapter 9.2.1 says the MIMG reserved bit must be zero and can cause a NACK
  that writes a status VGPR after the fetched data at
  `cdna2/README.md:2177` through `:2185`.
- The same table describes operation-specific `GLC` behavior for reads, writes,
  and atomics, `SLC` L2 streaming mode, `LWE` status returns, `A16` address
  packing, and `D16` load/store conversion restrictions at
  `cdna2/README.md:2177` through `:2185`.
- Chapter 13.6 marks bit 7 as `reserved`, names bit 12 `UNRM`, and places
  `D16` at bit 63 in the MIMG field map at `cdna2/README.md:7199` through
  `:7221`.

XML evidence:

- The CDNA2 XML `ENC_MIMG` field map names bit 7 `SCC` and describes it as
  "System level Cache Coherent" at `amdgpu_isa_cdna2.xml:2904` through
  `:2912`.
- `ENC_MIMG` describes `D16` only as converting 32-bit floating point texture
  return data to 16-bit at `amdgpu_isa_cdna2.xml:2831` through `:2838`, omitting
  store behavior, opcode restrictions, integer-vs-float interpretation through
  `NFMT`, and D16 packing.
- `ENC_MIMG` uses generic `GLC`, `SLC`, `LWE`, and `UNORM` descriptions at
  `amdgpu_isa_cdna2.xml:2870` through `:2888`, `:2914` through `:2922`, and
  `:2952` through `:2960`, rather than the manual's operation-specific cache
  and status-return rules.

Impact:

XML consumers can misclassify the reserved/NACK bit as a coherent-cache bit and
cannot derive the full MIMG status, cache, D16, and unnormalized-address
contract from field metadata alone.

### CDNA2-XML-028: Image address, `DMASK`, and atomic data-count rules are prose-only

Manual evidence:

- Chapter 9.3 gives no-sampler address component layouts by opcode, dimension,
  and array declaration, including mip-level and cube-face placement at
  `cdna2/README.md:2187` through `:2215`.
- Chapter 9.4 and 9.4.1 give sampler address typing, cubemap `face_id`,
  coordinate/derivative component ordering, and VGPR data rules at
  `cdna2/README.md:2216` through `:2248`.
- Chapter 9.4.1 defines `DMASK` read component selection, full-element write
  behavior, image atomic 32/64-bit surface restrictions, legal atomic `DMASK`
  values, atomic-return data placement, and D16 load/store packing at
  `cdna2/README.md:2244` through `:2259`.

XML evidence:

- Representative no-sampler image loads, stores, and atomics use fixed 128-bit
  `VDATA` and `VADDR` operands, with `IMAGE_GET_RESINFO` using a fixed 32-bit
  `VADDR`; for example `IMAGE_LOAD`, `IMAGE_STORE`, and `IMAGE_ATOMIC_SWAP`
  appear at `amdgpu_isa_cdna2.xml:20411` through `:20438`, `:20675` through
  `:20702`, and `:20895` through `:20937`.
- `IMAGE_SAMPLE` uses fixed 128-bit `VDATA` and 96-bit F32 `VADDR` operands at
  `amdgpu_isa_cdna2.xml:21610` through `:21642`.
- The `ENC_MIMG` `DMASK` field description says only that at least one bit must
  be set and data is packed into consecutive VGPRs at
  `amdgpu_isa_cdna2.xml:2851` through `:2858`.

Impact:

XML-derived tools can decode the operand fields but cannot determine the actual
number, type, or placement of address and data VGPRs for a given image opcode,
dimension, `DA`, `A16`, `D16`, or atomic `DMASK` mode.

### CDNA2-XML-029: Image resource and sampler descriptor bitfields are opaque

Manual evidence:

- Chapter 9.2 says image operations send a 256-bit image resource constant that
  defines address, data format, and surface characteristics, and sampler
  operations also send a 128-bit sampler constant at
  `cdna2/README.md:2143` through `:2149`.
- Chapter 9.4.2 defines image resource fields for base address, dimensions,
  data/numeric format, destination selectors, mip levels, tiling, type, depth,
  pitch, array pitch, metadata, compression, and partially resident texture
  fields at `cdna2/README.md:2261` through `:2309`.
- Chapter 9.4.3 defines sampler descriptor fields for clamp/wrap, anisotropy,
  depth compare, force unnormalized, filtering, LOD limits/biases, border
  color, and border color type at `cdna2/README.md:2311` through `:2352`.

XML evidence:

- `FMT_IMG` is a single opaque 256-bit `Descriptor` field at
  `amdgpu_isa_cdna2.xml:93511` through `:93529`.
- `FMT_SAMP` is a single opaque 128-bit `Descriptor` field at
  `amdgpu_isa_cdna2.xml:97827` through `:97845`.
- Representative image and sample instruction entries attach `SRSRC` and
  `SSAMP` operands with those descriptor formats but do not expose descriptor
  subfields; `IMAGE_LOAD` and `IMAGE_SAMPLE` show the operand forms at
  `amdgpu_isa_cdna2.xml:20431` through `:20436` and `:21630` through `:21640`.

Impact:

XML consumers can identify resource and sampler operands, but cannot derive
image dimensions, format conversion, destination selection, filtering, clamp,
mip/LOD, metadata, compression, or PRT behavior from descriptor metadata.

### CDNA2-XML-030: Image data-format conversion tables are not structured

Manual evidence:

- Chapter 9.2 says the image resource constant defines the data format and
  surface characteristics used by MIMG operations at `cdna2/README.md:2143`
  through `:2145`.
- Chapter 9.4.1 says loads expand memory data to canonical RGBA using the
  resource format before `DMASK` selection, while writes fill missing stored
  components with zero and ignore values outside the stored data format at
  `cdna2/README.md:2244` through `:2248`.
- Chapter 9.4.4 enumerates buffer/image `DATA_FORMAT` and `NUM_FORMAT` values
  at `cdna2/README.md:2354` through `:2396`.

XML evidence:

- XML instruction descriptions state that image loads and stores perform format
  conversion specified by the resource descriptor, for example `IMAGE_LOAD` and
  `IMAGE_STORE` at `amdgpu_isa_cdna2.xml:20411` through `:20412` and `:20675`
  through `:20676`.
- The resource descriptor itself is opaque as recorded in `CDNA2-XML-029`, and
  representative image data operands use `FMT_ANY` rather than a structured
  image-format table, for example at `amdgpu_isa_cdna2.xml:20419` through
  `:20429`.

Impact:

The XML records that conversion exists but does not encode the data-format and
numeric-format enumeration or the RGBA fill/selection rules needed for an image
load/store emulator.

### CDNA2-XML-031: Image VMEM dependency timing is prose-only

Manual evidence:

- Chapter 9.4.5 says an issued VMEM image instruction immediately reads VGPR
  addresses plus texture resources and samplers, while write data is not sent to
  the texture cache immediately at `cdna2/README.md:2398` through `:2404`.
- The same paragraph points developers to `VMCNT` waits before consuming data
  returned from the texture cache at `cdna2/README.md:2400` through `:2404`.

XML evidence:

- Image instruction entries mark the functional group as `VMEM` and subgroup as
  `TEXTURE`, with atomics also marked `ATOMIC`, for example
  `IMAGE_LOAD`, `IMAGE_ATOMIC_ADD`, and `IMAGE_SAMPLE` at
  `amdgpu_isa_cdna2.xml:20440` through `:20445`, `:21045` through `:21050`, and
  `:21645` through `:21651`.
- The XML entries do not encode the immediate address/resource/sampler read
  timing, delayed write-data timing, or `VMCNT` producer/decrement behavior for
  image operations.

Impact:

XML-derived schedulers, dependency analyzers, and emulators need manual-derived
rules for image issue timing and wait-counter behavior, even though the opcode
inventory is present.

### CDNA2-XML-032: Floating-memory atomic numeric rules are prose-only

Manual evidence:

- Chapter 9.5 says floating memory atomics execute in LDS and L2, can be issued
  as LDS, GDS, Buffer, Flat, Global, and Scratch instructions, and that the
  chapter defines rounding, denormal, and NaN rules at `cdna2/README.md:2406`
  through `:2410`.
- Float atomic ADD opcodes use round-to-nearest-even at
  `cdna2/README.md:2412` through `:2415`.
- Table 43 assigns separate LDS and L2 denormal behavior for `PK_ADD_F16`,
  `ADD_F32`, F32/F64 min/max, compare-store, and `ADD_F64` at
  `cdna2/README.md:2416` through `:2447`.
- Chapter 9.5.3 defines SNaN quieting, min/max total-order-style selection,
  `+0`/`-0` ordering, compare-store equality, and add edge cases at
  `cdna2/README.md:2449` through `:2492`.

XML evidence:

- DS float atomic entries expose opcode, operand, and short operation
  descriptions, for example `DS_CMPST_F32`, `DS_MIN_F32`, and
  `DS_MAX_F32` at `amdgpu_isa_cdna2.xml:8932` through `:9058`, plus return
  forms around `:10175` through `:10320` and F64 return forms around `:13071`
  through `:13210`.
- MUBUF and GLOBAL float atomic entries similarly expose operand widths and
  generic descriptions at `amdgpu_isa_cdna2.xml:24740` through `:24971` and
  `:18146` through `:18395`.
- Those records do not encode the manual's RNE requirement, per-atomic
  denormal policy, NaN quieting/selection, signed-zero ordering, compare-store
  `+0 == -0` rule, or add edge-case results.

Impact:

XML-only consumers can enumerate the float atomic opcodes and basic operand
widths, but cannot derive the numeric contract needed for faithful emulation,
validation, or edge-case test generation.

### CDNA2-XML-033: Flat/global float atomic return-mode rules are not structured

Manual evidence:

- Chapter 10.3.1 says float atomics must set `GLC=0`, with no return value, at
  `cdna2/README.md:2596` through `:2599`.
- The same subsection says FP32 atomics flush denormals to zero, FP64 and FP16
  atomics never flush denormals, and rounding is fixed RNE at
  `cdna2/README.md:2600` through `:2602`.

XML evidence:

- `FLAT_ATOMIC_ADD_F64` says the original flat-aperture value is stored iff
  `GLC` is set at `amdgpu_isa_cdna2.xml:15591` through `:15627`.
- `GLOBAL_ATOMIC_ADD_F32`, `GLOBAL_ATOMIC_PK_ADD_F16`, and
  `GLOBAL_ATOMIC_ADD_F64` carry the same generic `GLC` return wording at
  `amdgpu_isa_cdna2.xml:18146` through `:18289`.
- The XML has no structured per-opcode legality bit that marks float atomic
  return mode as unavailable or records the Chapter 10 L2 FP denormal/RNE
  policy.

Impact:

XML-derived decoders or disassemblers can surface return-data mode for CDNA2
flat/global float atomics even though the manual's Chapter 10 prose requires
the no-return form.

### CDNA2-XML-034: F64 min/max float atomic descriptions say signed integer

Manual evidence:

- Chapter 9.5 defines FP max/min selection rules for floating memory atomics at
  `cdna2/README.md:2464` through `:2482`.
- The detailed FLAT and GLOBAL tables name `*_MIN_F64` and `*_MAX_F64` and
  operate on `D.f64`/`DATA.f64` at `cdna2/README.md:5060` through `:5062` and
  `:5173` through `:5175`.
- The MUBUF opcode table lists `BUFFER_ATOMIC_MIN_F64` and
  `BUFFER_ATOMIC_MAX_F64` at `cdna2/README.md:7172` through `:7176`.

XML evidence:

- `FLAT_ATOMIC_MIN_F64` and `FLAT_ATOMIC_MAX_F64` describe selecting the
  minimum or maximum signed integer value at `amdgpu_isa_cdna2.xml:15643` and
  `:15695`.
- `GLOBAL_ATOMIC_MIN_F64` and `GLOBAL_ATOMIC_MAX_F64` use the same signed
  integer wording at `amdgpu_isa_cdna2.xml:18305` and `:18358`.
- `BUFFER_ATOMIC_MIN_F64` and `BUFFER_ATOMIC_MAX_F64` also use signed integer
  wording at `amdgpu_isa_cdna2.xml:24884` and `:24932`.

Impact:

Metadata consumers that trust XML descriptions can classify F64 float min/max
atomics as integer operations and miss the manual's FP NaN, signed-zero, and
denormal comparison rules.

### CDNA2-XML-035: Flat/global/scratch bit 25 is labeled as `SCC` instead of reserved

Manual evidence:

- The Chapter 10 FLAT format summary marks the bit between `OP` and `SEG` as
  reserved and says it must be zero at `cdna2/README.md:2502`.
- The Chapter 13.7 flat-memory field table repeats that bit 25 is reserved and
  must be zero at `cdna2/README.md:7277`.

XML evidence:

- `ENC_FLAT`, `ENC_FLAT_GLBL`, and `ENC_FLAT_SCRATCH` all expose bit 25 as a
  field named `SCC` with a system-level cache-coherent description at
  `amdgpu_isa_cdna2.xml:3779` through `:3787`, `:3975` through `:3983`, and
  `:4140` through `:4148`.

Impact:

XML consumers can treat a reserved-zero CDNA2 flat-memory bit as an active cache
coherency control, which can lead generated decoders, disassemblers, or fuzzers
to accept and describe encodings the manual reserves.

### CDNA2-XML-036: Segment-specific flat address and LDS rules are prose-only

Manual evidence:

- Chapter 10.1 says `LDS` enables direct data movement between LDS and memory
  for Global/Scratch and must be zero for Flat, while the `OFFSET` field is
  13-bit signed for Scratch/Global and 12-bit unsigned for Flat at
  `cdna2/README.md:2502` through `:2504`.
- Table 44 gives segment-specific `ADDR`, `SADDR`, and `M0` roles: Flat uses a
  VGPR address, Scratch can use an SGPR instead of a VGPR address, Global can
  use an SGPR base plus a 32-bit VGPR offset, and `M0` is used only for
  Scratch/Global direct-LDS forms at `cdna2/README.md:2510` through `:2511`.
- Sections 10.3 through 10.5 define PTR32/ADDRESS_MODE effects plus flat,
  global, and scratch address formulas at `cdna2/README.md:2582` through
  `:2638`.

XML evidence:

- The flat encodings expose broad field names and descriptions such as
  `ADDR` as a flat address VGPR, `SADDR` as an address/offset SGPR, `LDS` as
  read/write data from/to LDS, and segment-specific `OFFSET` widths at
  `amdgpu_isa_cdna2.xml:3699` through `:3757`, `:3769` through `:3777`,
  `:3889` through `:3953`, and `:4050` through `:4139`.
- The XML field records do not encode the Flat-only `LDS=0` restriction, the
  `M0` direct-LDS role, the `SADDR==0x7f` address-form switch, PTR32 or
  ADDRESS_MODE behavior, or the segment-specific address formulas.

Impact:

XML-derived code needs manual overrides to choose the legal address source,
direct-LDS behavior, and offset interpretation for each flat-memory segment.
The base field widths are present, but the architectural contract is not
machine-readable.

### CDNA2-XML-037: Flat ordering, wait-counter, aperture, and fault behavior is prose-only

Manual evidence:

- Chapter 10.2 says Flat instructions internally execute as both LDS and Buffer
  requests, increment both `VM_CNT` and `LGKM_CNT`, and are only complete when
  both counters decrement; the same section says the only sensible wait after a
  Flat instruction is `S_WAITCNT 0` because LDS and TA return paths can race at
  `cdna2/README.md:2566` through `:2578`.
- Chapter 10.4 says Global instructions use only `VM_CNT` and that a Global
  instruction targeting LDS returns a memory violation at
  `cdna2/README.md:2621`.
- Chapter 10.6 defines invalid-address, read-only, misaligned, and out-of-range
  memory errors, including MEM_VIOL trap status, dropped writes, and zeroed
  reads at `cdna2/README.md:2640` through `:2657`.

XML evidence:

- The XML flat/global/scratch encodings and representative instruction entries
  expose opcode, operand, and broad memory-field metadata, but they do not
  attach wait-counter producer behavior, Flat dual-path completion, per-lane
  aperture routing, Global-to-LDS memory-violation behavior, scratch no-check
  behavior, or MEM_VIOL/trap-status side effects.

Impact:

XML-only consumers can decode the instructions but cannot derive the ordering,
counter, aperture, or fault behavior required for a faithful scheduler, emulator,
or negative-test generator.

### CDNA2-XML-038: LDS direct-read source selection is not represented

Manual evidence:

- Section 6.2 lists "LDS direct data read" as a vector source operand kind at
  `cdna2/README.md:1261` and says only `SRC0` can use `LDS_DIRECT` at
  `:1271`.
- Section 11.3.1 says VALU instructions request an LDS direct read by using
  `LDS_DIRECT` for the `SRC0` field, and that `M0[15:0]` supplies the
  dword-aligned LDS byte address while `M0[18:16]` selects unsigned/signed
  byte, short, or dword data at `cdna2/README.md:2716` through `:2731`.

XML evidence:

- Searches of `amdgpu_isa_cdna2.xml` found no `LDS_DIRECT` operand value.
- CDNA2 source operand types omit selector value 254: `OPR_SRC_NOLDS` runs from
  SGPRs and inline constants to literal selector 255 but has no LDS-direct
  entry at `amdgpu_isa_cdna2.xml:97652` onward. The generated CDNA2 operand
  table mirrors that omission in `OPR_SRC` and `OPR_SRC_NOLDS`.

Impact:

XML-derived decoders cannot name or validate the manual's CDNA2 LDS direct-read
selector, so VALU `SRC0=LDS_DIRECT` encodings require manual overrides to decode
and execute correctly.

### CDNA2-XML-039: LDS M0 clamp, bank/conflict, and direct-access rules are prose-only

Manual evidence:

- Section 11.1 defines LDS as a 64 KiB per-CU, 32-bank memory where indexed and
  atomic operations can serialize on bank conflicts at `cdna2/README.md:2685`
  through `:2693`.
- Section 11.2 describes dataflow through VGPRs and LDS and notes buffer
  instructions can load directly into LDS at `cdna2/README.md:2701` through
  `:2703`.
- Section 11.3 says all LDS operations require `M0` to be initialized and that
  `M0` contains a size value that can restrict accesses to a subset of the LDS
  range; `0xffffffff` disables clamping at `cdna2/README.md:2760`.

XML evidence:

- `ENC_DS` exposes the DS field map and broad "Local and global data share
  operations" description at `amdgpu_isa_cdna2.xml:2814` through `:2916`.
- The DS field and instruction records do not encode LDS bank geometry,
  bank-conflict serialization, LDS-direct M0 address/datatype payloads, or the
  per-wave `M0` clamp/size contract for indexed and atomic LDS operations.

Impact:

The XML is enough to decode DS opcodes, but not enough for an emulator,
scheduler, or negative-test generator to derive LDS timing/serialization,
direct-read operand semantics, or M0-bounded LDS addressing.

### CDNA2-XML-040: DS double-address, atomic, and ADDTID address rules are prose-only

Manual evidence:

- Section 11.3 defines single-address DS operations as
  `LDS_BASE + VGPR[ADDR] + {InstrOffset1,InstrOffset0}` and READ2/WRITE2
  double-address operations as `VGPR[ADDR] + offset*ADJ`, with `ADJ=4` for
  8/16/32-bit data and `ADJ=8` for 64-bit data, at
  `cdna2/README.md:2777` through `:2797`.
- The same section says equal READ2/WRITE2 offsets mean only one read or write
  occurs and only `DATA0` is used at `cdna2/README.md:2799`.
- LDS atomics use a concatenated 16-bit offset, take `ADDR` as a dword address,
  use VGPR data sources or constants rather than SGPRs, and take denorm mode
  from `MODE.FP_DENORM` at `cdna2/README.md:2801` through `:2815`.
- The DS opcode table defines ADDTID as
  `ADDR_BASE + OFFSET + M0.OFFSET + TID*4` for write and read forms at
  `cdna2/README.md:4464` and `:4626`.

XML evidence:

- `OFFSET0`, `OFFSET1`, `ADDR`, `DATA0`, `DATA1`, and `VDST` are present in
  the DS encoding at `amdgpu_isa_cdna2.xml:2877` through `:2916`, and
  representative DS entries expose operands such as `ADDR` and `DATA0`.
- The XML records do not encode the per-instruction address formulas, ST64 and
  element-size scale factors, equal-offset single-access rule, atomic dword
  address interpretation, ADDTID `M0.OFFSET + TID*4` formula, or the manual's
  constant-but-not-SGPR data-source rule.

Impact:

XML consumers need manual-derived DS semantic tables to distinguish concatenated
offsets from dual offsets, collapse duplicate READ2/WRITE2 addresses, and avoid
generating incorrect ADDTID or atomic address/data behavior.

### CDNA2-XML-041: DS source-ACC operand classes contradict the DS field table

Manual evidence:

- The Chapter 13 DS field table says `ACC` means "`VDST` is Accumulation VGPR",
  while `DATA0` and `DATA1` are "First data VGPR" and "Second data VGPR" at
  `cdna2/README.md:6852` through `:6865`.
- The Chapter 11 atomic prose says VGPR data sources can only be VGPRs or
  constant values, not SGPRs, at `cdna2/README.md:2811` through `:2813`.

XML evidence:

- The XML `ACC` field description also says it specifies whether `VDST` uses an
  accumulator VGPR at `amdgpu_isa_cdna2.xml:2817` through `:2826`.
- Individual DS store and atomic entries nevertheless type `DATA0` as
  `OPR_VGPR_OR_ACCVGPR`, for example `DS_WRITE_B32` at
  `amdgpu_isa_cdna2.xml:8759` through `:8764` and `DS_ADD_U32` at
  `:8155` through `:8160`.

Impact:

XML-derived constructors can interpret the `ACC` bit as selecting accumulator
source registers for DS data operands even though both the manual field table
and XML field description scope `ACC` to `VDST`. This needs an architecture
check or manual-derived operand override before source-ACC DS forms are treated
as valid.

### CDNA2-XML-042: DS permute, bpermute, swizzle, and GWS details are prose-only

Manual evidence:

- `DS_PERMUTE_B32` and `DS_BPERMUTE_B32` do not access LDS memory, may be used
  without an LDS allocation, divide address bytes by four, ignore high address
  bits, apply `EXEC` to source and/or destination lanes, and define disabled
  source and conflict behavior at `cdna2/README.md:4507` through `:4511`.
- Section 11.4 says every GWS instruction must be immediately followed by
  `s_waitcnt 0`, and VGPRs used by any GWS instruction must be even at
  `cdna2/README.md:2817` through `:2825`.
- The detailed GWS opcode table carries queue/counter/barrier/semaphore
  pseudocode at `cdna2/README.md:4590` through `:4625`.

XML evidence:

- XML entries for `DS_PERMUTE_B32`, `DS_BPERMUTE_B32`, and `DS_SWIZZLE_B32`
  include short descriptions and operands but do not structure the no-LDS,
  byte-address-to-lane, high-bit, `EXEC`, disabled-source, or conflict rules.
- XML GWS entries are present as ordinary DS opcodes, but they do not encode
  the mandatory following `s_waitcnt 0`, even-VGPR restriction, or GWS resource
  state-machine behavior.

Impact:

XML-derived tests can enumerate the opcodes but cannot derive the lane-level
permute/swizzle contract or GWS scheduling restrictions without manual prose.

### CDNA2-XML-043: VOP2 FP min/max edge-selection rules are prose-only

Manual evidence:

- `V_MIN_F32` and `V_MAX_F32` define signaling-NaN quieting under `IEEE_MODE`,
  NaN operand selection, signed-zero tie selection, and the `V_MAX_F32`
  IEEE-mode equality case at `cdna2/README.md:3342` through `:3343`.
- `V_MAX_F16` and `V_MIN_F16` repeat the same NaN, signed-zero, and
  IEEE-mode selection contract for half precision at `cdna2/README.md:3401`
  through `:3402`.

XML evidence:

- XML entries for `V_MIN_F32` and `V_MAX_F32` describe generic min/max
  selection and operands at `amdgpu_isa_cdna2.xml:52714` through `:52841` and
  `:52855` through `:52935`.
- XML entries for `V_MAX_F16` and `V_MIN_F16` similarly expose the opcode,
  operand widths, literal/DPP/SDWA/VOP3 forms, and generic descriptions at
  `amdgpu_isa_cdna2.xml:57587` through `:57714` and `:57728` through `:57850`.
- Those XML records do not encode signaling-NaN quieting, NaN operand
  preference, signed-zero preference, or the `IEEE_MODE`-dependent equality
  case.

Impact:

XML consumers can decode the VOP2/VOP3 min/max forms, but need manual-derived
numeric rules to generate hardware-faithful NaN, signed-zero, and MODE-sensitive
tests or emulation.

### CDNA2-XML-044: `V_READFIRSTLANE_B32` EXEC and no-modifier rules are prose-only

Manual evidence:

- `V_READFIRSTLANE_B32` copies a VGPR value to an SGPR, selects the lowest
  active lane, forces lane 0 when `EXEC` is zero, ignores `EXEC` for the VGPR
  access, and says input/output modifiers are unsupported because the operation
  is untyped at `cdna2/README.md:3448`.

XML evidence:

- XML records the VOP1 and VOP3 encodings, destination, and source operand for
  `V_READFIRSTLANE_B32` at `amdgpu_isa_cdna2.xml:43549` through `:43608`.
- The XML description says only that the lowest active lane is read; it does
  not encode the all-disabled `EXEC` lane-0 rule, the access-mask override, or
  the no-modifier rule.

Impact:

The XML is enough to construct the opcode, but not enough to emulate the
all-disabled wave case or to validate that generic modifier/extension handling
must not be applied to this untyped scalarizing instruction.

### CDNA2-XML-045: VOP1 lookup, conversion-exception, and special-state semantics are prose-only

Manual evidence:

- `V_CVT_I32_F64`, `V_CVT_U16_F16`, and `V_CVT_I16_F16` tie out-of-range,
  NaN, saturation, truncation, and `CLAMP`-controlled `INEXACT` exception
  behavior to conversion semantics at `cdna2/README.md:3449` and
  `:3592` through `:3597`.
- `V_CLREXCP` clears the wave's exception state at `cdna2/README.md:3546`.
- `V_SCREEN_PARTITION_4SE_B32` carries a 256-entry lookup table and rectangle
  interpretation at `cdna2/README.md:3552` through `:3582`.
- The F16 reciprocal, square-root, rsq, log, exp, frexp, rounding, trig, and
  legacy exp/log rows carry special examples and denorm/rounding prose at
  `cdna2/README.md:3598` through `:3686`, and `V_SAT_PK_U8_I16` defines packed
  signed-16 to unsigned-8 saturation at `:3691`.

XML evidence:

- XML has the affected opcode records, including representative entries for
  `V_CVT_I32_F64`, `V_RCP_IFLAG_F32`, `V_CLREXCP`,
  `V_SCREEN_PARTITION_4SE_B32`, `V_CVT_U16_F16`, `V_CVT_I16_F16`, and
  `V_SAT_PK_U8_I16` at `amdgpu_isa_cdna2.xml:43622`, `:46835`, `:48662`,
  `:48696`, `:49029`, `:49140`, and `:51027`.
- Those records identify opcode and operand metadata, but do not structure the
  lookup table, exception-state effects, `CLAMP`-as-`INEXACT` behavior, F16
  special-result examples, denorm notes, legacy semantics, or packed saturation
  formula.

Impact:

An XML-only implementation can enumerate these VOP1 rows but cannot derive the
manual's bit-exact lookup, exception, conversion, and special numeric behavior.

### CDNA2-XML-046: VOPC predicate, class-mask, and `CLAMP` signaling rules are prose-only

Manual evidence:

- Chapter 12.9 defines the OP16 floating compare truth table, OP8 integer
  compare truth table, and opcode-family offsets at `cdna2/README.md:3720`
  through `:3790`.
- `V_CMP_CLASS_*` and `V_CMPX_CLASS_*` define the ten class-mask bits for
  signaling NaN, quiet NaN, infinities, normals, denormals, and signed zeros at
  `cdna2/README.md:3796` through `:3856`.
- The VOP3A compare promotion prose overloads `CLAMP`: when set, NaN inputs
  signal an exception; when clear, NaNs do not signal at
  `cdna2/README.md:4081` through `:4087`.

XML evidence:

- XML entries for class compares describe a generic 10-bit mask and expose the
  VOP3/VOPC/literal/SDWA operands, for example `V_CMP_CLASS_F32` at
  `amdgpu_isa_cdna2.xml:67340` through `:67440` and `V_CMP_CLASS_F16` at
  `:67781` through `:67880`.
- Ordinary compare entries such as `V_CMP_F_F32`, `V_CMP_NGE_F32`,
  `V_CMP_F_I32`, and `V_CMP_LT_U32` are present at
  `amdgpu_isa_cdna2.xml:71997`, `:73023`, `:83021`, and `:84047`.
- The generic VOP3 `CLAMP` field says only that output is clamped at
  `amdgpu_isa_cdna2.xml:6708` through `:6715`, not that VOPC compares use it
  as a NaN exception-signaling control.

Impact:

XML consumers can enumerate compare opcodes and operand shapes, but need manual
prose to implement ordered/unordered predicate truth tables, class-mask bit
meaning, CMPX `EXEC` behavior details, and VOP3A compare exception signaling.

### CDNA2-XML-047: VOP3B `SDST` semantics are under-described

Manual evidence:

- Chapter 12.11 says VOP3B has a unique scalar destination and is used by the
  carry-out arithmetic opcodes, `V_DIV_SCALE_F32/F64`, and
  `V_MAD_{U64,I64}_{U32,I32}` at `cdna2/README.md:4200` through `:4220`.
- `V_DIV_SCALE_F32/F64` define a per-lane `VCC` post-scale flag, require `S0`
  to match either the denominator or numerator, and provide detailed scaling
  pseudocode at `cdna2/README.md:4300` through `:4305`.
- `V_MAD_U64_U32` and `V_MAD_I64_I32` explicitly produce `{vcc_out,D}` at
  `cdna2/README.md:4315` through `:4316`.

XML evidence:

- The XML records VOP3B operands and opcodes for `V_DIV_SCALE_F32/F64` and
  `V_MAD_{U64,I64}_{U32,I32}` at `amdgpu_isa_cdna2.xml:61338` through
  `:61430` and `:61736` through `:61827`.
- The generic `VOP3_SDST_ENC` field description calls `SDST` the "Destination
  for compare result" at `amdgpu_isa_cdna2.xml:7242` through `:7248`, even
  though the VOP3B users here write carry, borrow, post-scale, or wide-MAD
  carry-out masks.
- The XML descriptions summarize the broad operation, but do not structure the
  manual's per-case divide-scale pseudocode or the `S0 == S1 || S0 == S2`
  operand contract.

Impact:

XML consumers can decode the extra scalar destination, but cannot derive the
exact mask meaning or division-scale legality and special-case behavior from
structured metadata alone.

### CDNA2-XML-048: Native and legacy F16 VOP3A destination-half rules are prose-only

Manual evidence:

- `V_MAD_LEGACY_F16`, `V_MAD_LEGACY_U16`, and `V_MAD_LEGACY_I16` say
  `op_sel[3] == 0` writes the low destination half and writes the high half as
  zero, while `op_sel[3] == 1` writes the high half and preserves the low half
  at `cdna2/README.md:4317` through `:4322`.
- The non-legacy `V_MAD_F16`, `V_MAD_U16`, `V_MAD_I16`, `V_FMA_F16`, and
  `V_DIV_FIXUP_F16` rows instead say both destination-half choices preserve the
  untouched half at `cdna2/README.md:4351` through `:4360`.

XML evidence:

- The XML descriptions for the legacy rows only say they implement a
  "non-standard rule for OPSEL" at `amdgpu_isa_cdna2.xml:61842` through
  `:61937`; the exact zero-versus-preserve behavior is not encoded.
- The non-legacy F16 rows expose 16-bit operands and generic descriptions at
  `amdgpu_isa_cdna2.xml:63011` through `:63231`, but do not encode the
  destination-half preservation rule.
- The generic VOP3A `OPSEL` field identifies the destination selector bit at
  `cdna2/README.md:6396`, but the XML cannot distinguish the legacy zero-high
  exception from the native preserve behavior.

Impact:

An XML-only generator has no structured way to choose between the legacy
zero-high writeback and the native preserve-other-half writeback.

### CDNA2-XML-049: `V_PERM_B32` selector-table semantics are prose-only

Manual evidence:

- `V_PERM_B32` defines selector bytes `0..7` as source-byte picks, selector 12
  as zero, selectors 8 through 11 as sign-fill from source bytes 1, 3, 5, and 7,
  and selectors `>= 13` as `0xff` at `cdna2/README.md:4323`.

XML evidence:

- The XML entry for `V_PERM_B32` says the selector can choose bytes, perform
  sign extension, or pad with 0/1 bits at `amdgpu_isa_cdna2.xml:61983` through
  `:62014`.
- The selector-to-result table itself is not structured in the XML.

Impact:

The XML describes the existence of special selectors, but consumers need the
manual's prose table to implement or test the exact selector values.

### CDNA2-XML-050: VOP3A division-helper edge numerics are prose-only

Manual evidence:

- `V_DIV_FIXUP_F32`, `V_DIV_FIXUP_F64`, and `V_DIV_FIXUP_F16` enumerate NaN,
  zero, infinity, sign, overflow, underflow, and exception-generation cases at
  `cdna2/README.md:4267` through `:4296` and `:4360`.
- `V_DIV_FMAS_F32/F64` conditionally scale the fused multiply-add result from
  `VCC[threadId]` at `cdna2/README.md:4306` through `:4310`.

XML evidence:

- XML descriptions for the divide-fixup rows identify the broad operation and
  operands at `amdgpu_isa_cdna2.xml:61244` through `:61325`, and the
  divide-FMAS rows identify the broad operation and operands at
  `amdgpu_isa_cdna2.xml:61444` through `:61577`.
- The XML does not structure the detailed NaN/zero/infinity result selection,
  exception side effects, or VCC-driven post-scale formula.

Impact:

Generated tests or emulators need manual-derived tables for divide helper edge
cases; opcode and operand metadata are not enough.

### CDNA2-XML-051: VOP3A F64 min/max edge-selection rules are prose-only

Manual evidence:

- `V_MIN_F64` and `V_MAX_F64` define signaling-NaN quieting under `IEEE_MODE`,
  NaN operand preference, signed-zero result selection, and the `V_MAX_F64`
  `IEEE_MODE` equality rule at `cdna2/README.md:4367` through `:4368`.

XML evidence:

- XML entries for `V_MIN_F64` and `V_MAX_F64` describe generic min/max
  selection and expose VOP3 operands at `amdgpu_isa_cdna2.xml:63328` through
  `:63394`.
- Those XML records do not encode signaling-NaN quieting, signed-zero
  preference, NaN operand preference, or `IEEE_MODE` equality behavior.

Impact:

The F64 VOP3A min/max rows require the same manual-derived numeric edge rules
as the F32/F16 min/max rows recorded earlier.

### CDNA2-XML-052: `V_READLANE_B32` and `V_WRITELANE_B32` EXEC and modifier rules are prose-only

Manual evidence:

- `V_READLANE_B32` copies one VGPR lane to one SGPR, ignores the `EXEC` mask,
  and says input/output modifiers are unsupported at `cdna2/README.md:4374`
  through `:4375`.
- `V_WRITELANE_B32` writes one VGPR lane, ignores the `EXEC` mask, and also
  says modifiers are unsupported at `cdna2/README.md:4379`.

XML evidence:

- XML records the VOP3 operands for `V_READLANE_B32` and `V_WRITELANE_B32` at
  `amdgpu_isa_cdna2.xml:63615` through `:63682`.
- The XML descriptions do not encode the `EXEC`-mask override or the
  instruction-specific no-modifier rule.

Impact:

XML consumers can decode the lane-transfer forms, but need manual prose to
avoid applying generic VOP3 modifier handling and to emulate disabled-lane
behavior correctly.

### CDNA2-XML-053: `V_TRIG_PREOP_F64` range-reduction behavior is prose-only

Manual evidence:

- `V_TRIG_PREOP_F64` defines the `2/pi` segment lookup, exponent-dependent
  shift and scale, round-toward-zero behavior, and large-input scaling at
  `cdna2/README.md:4389`.

XML evidence:

- XML describes the row as a 53-bit `2/pi` segment lookup scaled by the source
  exponent at `amdgpu_isa_cdna2.xml:63943` through `:63969`.
- The XML does not encode the segment extraction expression, RTZ rule, exponent
  thresholds, or large-input scaling adjustment.

Impact:

The XML is enough for decode and broad identification, but not enough for a
bit-exact range-reduction implementation.

### CDNA2-XML-054: VGPR indexing mode and operand remaps are prose-only

Manual evidence:

- Chapter 6.6 says VALU VGPR indexing uses M0 as an index for source or
  destination VGPRs at `cdna2/README.md:1450`.
- The indexing control table specifies that `S_SET_GPR_IDX_ON` and
  `S_SET_GPR_IDX_MODE` store the mode in `M0[15:12]`, while
  `S_SET_GPR_IDX_IDX` stores the unsigned offset in `M0[7:0]`, at
  `cdna2/README.md:1458` through `:1461`.
- The prose assigns `M0[15:12]` to destination, src2, src1, and src0 enables,
  excludes inline constants and SGPRs, makes out-of-range indexed VGPR access
  illegal, and gives special operand remaps for `v_readlane`, `v_writelane`,
  `v_mac_*`, `v_madak`, `v_madmk`, reverse shifts, `v_cvt_pkaccum`, and SDWA at
  `cdna2/README.md:1465` through `:1492`.

XML evidence:

- XML entries exist for `S_SET_GPR_IDX_IDX`, `S_SET_GPR_IDX_ON`,
  `S_SET_GPR_IDX_OFF`, and `S_SET_GPR_IDX_MODE` at
  `amdgpu_isa_cdna2.xml:31947`, `:41323`, `:43309`, and `:43331`.
- Those entries expose opcode and operand metadata but do not encode the
  architectural `M0[15:12]` bit layout, the dependency on `MODE.gpr_idx_en`, the
  per-source/destination mask semantics, the SGPR/inline exclusion, the
  out-of-range illegal case, or the special remap table.

Impact:

XML-only generators can decode the control instructions, but cannot derive the
runtime VGPR address remapping rules required for architectural indexing.

### CDNA2-XML-055: VALU rounding, denormal, and output-modifier MODE interactions are prose-only

Manual evidence:

- Chapter 6.2.2 says VOP3 floating-point results can apply OMOD and CLAMP, but
  OMOD is ignored for integer/bit results, ignored when output denormals are
  enabled, flushes denormals and `-0` when applied, and is ignored when
  `IEEE_MODE` is set at `cdna2/README.md:1295` through `:1297`.
- Chapter 6.4 defines `MODE.FP_ROUND` and `MODE.FP_DENORM` bits for
  single-precision and double/half-precision floating-point VALU behavior, and
  notes that floating `V_DOT2` instructions always flush input and output
  denormals at `cdna2/README.md:1431` through `:1442`.
- Chapter 6.5 gives the ALU clamp behavior for compare, integer, and
  floating-point instructions at `cdna2/README.md:1444` through `:1446`.

XML evidence:

- The XML exposes VOP3 `CLAMP` and `OMOD` fields at
  `amdgpu_isa_cdna2.xml:2568` through `:2599`, and equivalent VOP3B output
  fields exist in `VOP3_SDST_ENC`.
- The XML does not expose the `FP_ROUND`, `FP_DENORM`, or `IEEE_MODE`
  result-policy metadata for VALU instructions, and the field records do not
  encode the OMOD suppression, denormal-flush, or `-0` behavior.

Impact:

XML-only implementations can see the output-modifier fields, but need manual
prose to decide when OMOD is active, which rounding mode to use, and how to
handle denormal inputs and results.

### CDNA2-XML-056: SOP2 detailed formulas and edge examples are not machine-readable

Manual evidence:

- Chapter 12.1 gives the full SOP2 opcode table at `cdna2/README.md:2860`
  through `:2942`.
- `S_BFM_B32/B64` define exact width and offset bit slices from `S0` and `S1`
  at `cdna2/README.md:2912` through `:2914`.
- `S_BFE_U32/I32/U64/I64` define `S1[4:0]` or `S1[5:0]` as the offset and
  `S1[22:16]` as the field width at `cdna2/README.md:2922` through `:2925`.
- `S_ABSDIFF_I32` defines a wrapped signed subtract followed by
  negate-if-negative, and gives edge examples for `0x80000000` inputs at
  `cdna2/README.md:2932`.
- `S_PACK_LL_B32_B16`, `S_PACK_LH_B32_B16`, and `S_PACK_HH_B32_B16` define the
  exact low/high half selections at `cdna2/README.md:2940` through `:2942`.

XML evidence:

- The XML has matching `ENC_SOP2` opcode entries from `S_ADD_U32` through
  `S_PACK_HH_B32_B16` at `amdgpu_isa_cdna2.xml:32399` through `:39323`.
- The XML descriptions for `S_BFM_B32`, `S_BFE_U32`, `S_ABSDIFF_I32`, and the
  pack rows are short summaries at `amdgpu_isa_cdna2.xml:37063` through
  `:37064`, `:37411` through `:37412`, `:38067` through `:38068`, and
  `:39091` through `:39324`.
- Those entries expose operands, sizes, and implicit SCC where applicable, but
  do not structure the bit-slice formulas, the `S_ABSDIFF_I32` wraparound edge
  examples, or the scalar-pack source-half and destination-half placement rules.

Impact:

XML-only generators can build the SOP2 opcode inventory and operands, but need
manual-specific overrides or tests for exact bitfield, absolute-difference, and
pack semantics.

### CDNA2-XML-057: SOP1 relative-addressing rules and edge examples are prose-only

Manual evidence:

- Chapter 12.3 gives detailed SOP1 formulas and examples for WQM, bit count,
  first-bit scans, leading-bit scans, `S_ABS_I32`, quad-mask, and
  bit-replicate rows at `cdna2/README.md:3000` through `:3088`.
- `S_MOVRELS_B32/B64` and `S_MOVRELD_B32/B64` define the effective scalar
  register address as the instruction source/destination SGPR address plus
  `M0.u`, and the 64-bit forms require an even `M0.u`, at
  `cdna2/README.md:3072` through `:3075`.
- `S_CBRANCH_JOIN` gives branch-stack/CSP pseudocode at `cdna2/README.md:3076`;
  the broader fork/join XML gap is tracked separately in `CDNA2-XML-012`.

XML evidence:

- The XML has matching `ENC_SOP1` opcode entries for all Chapter 12.3 rows at
  `amdgpu_isa_cdna2.xml:28988` through `:32346`, including implicit SCC, EXEC,
  PC, and M0 operands where applicable.
- The bit-count, bit-scan, `S_ABS_I32`, and WQM entries use short descriptions
  at `amdgpu_isa_cdna2.xml:29352` through `:30223` and `:31883` through
  `:31934`, but do not structure the manual's edge examples, such as the
  negative `S_ABS_I32(0x80000000)` result.
- The `S_MOVRELS_*` and `S_MOVRELD_*` entries expose M0 as an implicit operand
  at `amdgpu_isa_cdna2.xml:31565` through `:31805`, but do not encode the
  address formula `addr += M0.u` or the even-`M0` requirement for 64-bit forms.

Impact:

XML-only generators can build the SOP1 opcode inventory and basic operands, but
need manual-derived rules for relative SGPR addressing, 64-bit relative-move
alignment, branch-stack join behavior, and edge-sensitive unary tests.

### CDNA2-XML-058: SOPC state and raw-field details are not represented precisely

Manual evidence:

- Chapter 12.4 defines `S_BITCMP0_B32`/`S_BITCMP1_B32` as indexing with
  `S1.u[4:0]`, and the B64 forms as indexing with `S1.u[5:0]`, at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:3114` through `:3117`.
- The `S_SETVSKIP` row says `VSKIP = S0.u[S1.u[4:0]]`, lists the instruction
  classes skipped by VSKIP, and states that VSKIPped memory instructions do not
  manipulate waitcnt counters at `cdna2/README.md:3118`.
- The `S_SET_GPR_IDX_ON` row says `MODE.gpr_idx_en = 1`, writes `M0[7:0]` from
  `S0.u[7:0]`, writes `M0[15:12] = SIMM4`, leaves the other M0 bits
  unchanged, and clarifies that the SIMM4 value is the direct raw content of
  the S1 field at `cdna2/README.md:3124`.

XML evidence:

- XML describes `S_SETVSKIP` only as enabling or disabling VSKIP mode and gives
  it two ordinary `OPR_SSRC` operands plus generic SOPC literal variants at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:41231` through
  `:41310`; it does not encode the bit-index mask, VSKIP state output, skipped
  instruction classes, or waitcnt policy.
- XML gives `S_SET_GPR_IDX_ON` a default `OPR_SIMM4` second operand and
  implicit M0 input/output operands at `amdgpu_isa_cdna2.xml:41323` through
  `:41352`, but generic literal variants also expose `SSRC1` as `SIMM32` with
  `OPR_SIMM4` at `:41384` through `:41428`. The broader GPR-index mode layout
  and remap omissions are tracked in `CDNA2-XML-054`.

Impact:

XML-only generators can decode the SOPC rows, but cannot derive the full VSKIP
state contract, bit-index masking, or the special raw-field treatment for
`S_SET_GPR_IDX_ON`. Generic literal expansion can assign an extension word to a
field the manual defines as direct SIMM4 bits.

### CDNA2-XML-059: SOPP opcode tables omit XML-only `S_TTRACEDATA`

Manual evidence:

- The detailed Chapter 12.5 SOPP table lists opcodes 0 through 21 and then
  skips to opcode 23 through 30 at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:3134` through `:3179`.
- The Chapter 13.1.5 SOPP opcode table likewise lists opcodes 0 through 21 and
  then 23 through 30, ending at `S_ENDPGM_ORDERED_PS_DONE`, at
  `cdna2/README.md:5677` through `:5708`.

XML evidence:

- The checked-in CDNA2 XML defines `S_TTRACEDATA` as `ENC_SOPP` opcode 22 and
  describes it as sending M0 as user data to the thread-trace stream at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:43143` through
  `:43149`.
- The same XML row gives the instruction an implicit 32-bit M0 input operand at
  `amdgpu_isa_cdna2.xml:43150` through `:43155`.

Impact:

Manual-derived SOPP inventories can classify opcode 22 as absent or reserved,
while XML-derived decoders expose `S_TTRACEDATA` with a thread-trace M0 input.

### CDNA2-XML-060: SOPP message, termination, and cache details are prose-only

Manual evidence:

- `S_ENDPGM` and `S_ENDPGM_SAVED` implicitly execute `S_WAITCNT 0` before
  terminating the wave at `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:3135`
  and `:3171`.
- `S_ENDPGM_ORDERED_PS_DONE` implicitly executes `S_WAITCNT 0` and combines
  `S_SENDMSG(MSG_ORDERED_PS_DONE)` with `S_ENDPGM` at `cdna2/README.md:3179`.
- The `S_SENDMSG` section defines the message/payload table, including illegal
  message 0, interrupt payload in `M0[23:0]`, save-wave, stall-wave-gen, and
  halt-waves encodings, at `cdna2/README.md:3183` through `:3191`.
- `S_ICACHE_INV` requires 16 following `S_NOP` instructions or a jump/branch to
  purge the SQ instruction buffer at `cdna2/README.md:3164`.

XML evidence:

- XML gives `S_ENDPGM` only a program-terminator flag and a short description at
  `amdgpu_isa_cdna2.xml:42516` through `:42529`.
- XML gives `S_SENDMSG` and `S_SENDMSGHALT` a generic `OPR_SENDMSG` immediate
  plus implicit M0 operand at `amdgpu_isa_cdna2.xml:42966` through `:43018`,
  but does not encode the message table, illegal message 0, or payload sources.
- XML gives `S_ICACHE_INV` only the cache-invalidation row at
  `amdgpu_isa_cdna2.xml:43063` through `:43070`, without the post-invalidation
  purge requirement.
- XML describes `S_ENDPGM_ORDERED_PS_DONE` as a program terminator at
  `amdgpu_isa_cdna2.xml:43370` through `:43377`, but does not structure the
  implicit wait or ordered-PS message side effect.

Impact:

XML consumers can decode and classify these SOPP rows, but still need manual
prose for termination ordering, send-message payload validation, ordered-PS
completion signaling, and the instruction-cache purge hazard.

### CDNA2-XML-061: MIMG MIP opcode miplevel-0 legality is prose-only

Manual evidence:

- `IMAGE_LOAD_MIP` is the user-supplied-mip load form, but the manual says it is
  only allowed for miplevel 0 and must be enforced by software at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:4948`.
- The same miplevel-0/software-enforced restriction is repeated for
  `IMAGE_LOAD_MIP_PCK`, `IMAGE_LOAD_MIP_PCK_SGN`, `IMAGE_STORE_MIP`, and
  `IMAGE_STORE_MIP_PCK` at `cdna2/README.md:4951` through `:4965`.

XML evidence:

- XML describes `IMAGE_LOAD_MIP`, `IMAGE_LOAD_MIP_PCK`, and
  `IMAGE_LOAD_MIP_PCK_SGN` as user-specified-miplevel load forms at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:20286` through
  `:20486`, but does not encode the miplevel-0 legality limit.
- XML likewise describes `IMAGE_STORE_MIP` and `IMAGE_STORE_MIP_PCK` as
  user-specified-miplevel store forms at `amdgpu_isa_cdna2.xml:20538` through
  `:20623`, without the software-enforced miplevel-0 restriction.

Impact:

XML-derived validators, assemblers, and emulators can identify the MIP opcodes
but cannot derive the CDNA2-specific legality rule that those forms are only
allowed for miplevel 0.

### CDNA2-XML-062: Atomic decrement threshold wrap is under-described

Manual evidence:

- The CDNA2 manual's integer atomic decrement pseudocode resets the memory value
  to the data operand when the old value is zero or greater than the data
  operand; the repeated OCR text omits the visible logical operator, but the
  same threshold pattern appears for SMEM, DS, MUBUF, MIMG, FLAT, and GLOBAL
  forms at `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:3258`,
  `:4439`, `:4846`, `:4984`, `:5059`, and `:5170`.
- The 64-bit decrement rows use the same threshold rule with the 64-bit data
  pair at `cdna2/README.md:3276`, `:4519`, `:4869`, `:5078`, and `:5191`.

XML evidence:

- XML descriptions for representative 32-bit decrement atomics say only that
  decrement wraps to the data register value if the decrement yields a negative
  value: `DS_DEC_U32` at `amdgpu_isa_cdna2.xml:8325` through `:8326`,
  `IMAGE_ATOMIC_DEC` at `:21210` through `:21211`,
  `BUFFER_ATOMIC_DEC` at `:24692` through `:24693`,
  `FLAT_ATOMIC_DEC` at `:15539` through `:15540`,
  `GLOBAL_ATOMIC_DEC` at `:18093` through `:18094`,
  `S_BUFFER_ATOMIC_DEC` at `:27308` through `:27309`, and
  `S_ATOMIC_DEC` at `:28400` through `:28401`.

Impact:

XML-derived implementations can infer an underflow-only decrement rule and miss
the manual's threshold case where an old value greater than the data operand
also resets to the data operand.

### CDNA2-XML-063: FLAT `ACC` field description names `VDST` instead of `VDATA`

Manual evidence:

- Chapter 13.7 defines the FLAT-family `ACC` field at bit 55 as "`VDATA` is
  Accumulation VGPR" at `cdna2/README.md:7287`.
- The same field table separately defines `VDST` as the data returned from
  memory to VGPRs at `cdna2/README.md:7288`, so the manual distinguishes the
  DATA/VDATA field from the return-data destination field.

XML evidence:

- The checked-in XML generic format records for `ENC_FLAT`, `ENC_FLAT_GLBL`,
  and `ENC_FLAT_SCRATCH` all describe `ACC` as specifying whether `VDST` uses
  an accumulator VGPR at `amdgpu_isa_cdna2.xml:3690` through `:3695`,
  `:3885` through `:3890`, and `:4050` through `:4055`.
- Per-instruction operand metadata partly recovers the actual data-field role:
  for example, `FLAT_STORE_DWORD` marks `DATA` as `OPR_VGPR_OR_ACCVGPR` at
  `amdgpu_isa_cdna2.xml:14455` through `:14474`, and `FLAT_ATOMIC_ADD` marks
  both `VDST` and `DATA` as `OPR_VGPR_OR_ACCVGPR` at `:15019` through `:15044`.

Impact:

Consumers that rely on the generic XML field map rather than every
instruction-specific operand list can attribute CDNA2 flat-memory `ACC` to the
return destination only, even though the manual's field definition names the
VDATA/DATA side of the instruction.

### CDNA2-XML-064: Chapter 12.17 DPP exclusions leak into F64 VOP1 metadata

Manual evidence:

- Chapter 12.17.1 says the listed instructions cannot use DPP at
  `cdna2/README.md:5197` through `:5229`. The list includes
  `V_CVT_I32_F64`, `V_CVT_F64_I32`, `V_CVT_F32_F64`, `V_CVT_U32_F64`,
  `V_CVT_F64_U32`, `V_TRUNC_F64`, `V_CEIL_F64`, `V_FLOOR_F64`,
  `V_FREXP_EXP_I32_F64`, `V_FREXP_MANT_F64`, and `V_FRACT_F64`.

XML evidence:

- XML still gives those listed rows `VOP1_VOP_DPP` encodings. Representative
  entries include `V_CVT_I32_F64` with `VOP1_VOP_DPP` at
  `amdgpu_isa_cdna2.xml:43622` through `:43666`, and
  `V_FREXP_EXP_I32_F64` with `VOP1_VOP_DPP` at `:48164` through `:48208`.
- The other affected DPP entries are `V_CVT_F64_I32` at
  `amdgpu_isa_cdna2.xml:43756`, `V_CVT_F32_F64` at `:44847`,
  `V_CVT_U32_F64` at `:45456`, `V_CVT_F64_U32` at `:45548`,
  `V_TRUNC_F64` at `:45640`, `V_CEIL_F64` at `:45732`,
  `V_FLOOR_F64` at `:45897`, `V_FREXP_MANT_F64` at `:48298`, and
  `V_FRACT_F64` at `:48390`.

Impact:

XML-derived decoders, generators, and fuzzers can synthesize DPP forms for
F64 VOP1 instructions that Chapter 12.17 explicitly excludes.

### CDNA2-XML-065: VOPC DPP format support is missing from XML

Manual evidence:

- Chapter 13.3.9 describes DPP as a second dword that can follow VOP1, VOP2, or
  VOPC instructions in place of a literal constant at `cdna2/README.md:6791`
  through `:6797`.
- The source selector table marks selector 250 as DPP and says it is valid only
  as source 0 at `cdna2/README.md:1342` through `:1343`.

XML evidence:

- The checked-in XML declares `VOP1_VOP_DPP` and `VOP2_VOP_DPP` format records
  at `amdgpu_isa_cdna2.xml:4386` and `:5085`.
- The same XML declares a VOPC SDWA/SDWAB format record,
  `VOPC_VOP_SDWA_SDST_ENC`, at `amdgpu_isa_cdna2.xml:6387`, but there is no
  `VOPC_VOP_DPP` encoding name in the CDNA2 XML.

Impact:

The XML cannot independently express the manual's legal VOPC DPP second-word
format. XML-derived decoders, fuzzers, and legality checks must either miss VOPC
DPP entirely or recover it through handwritten special cases rather than the
machine-readable ISA contract.

### CDNA2-XML-066: SDWA selector values and DPP control semantics are prose-only

Manual evidence:

- Table 80 gives the SDWA selector values: `DST_SEL` values 0-3 and 7 are
  reserved, 4 selects the low word, 5 selects the high word, 6 selects the full
  dword, `DST_U=3` is reserved, and `SRC0_SEL`/`SRC1_SEL` use the same options
  at `cdna2/README.md:6737` through `:6753`.
- Table 83 gives the `DPP_CTRL` enumeration, including `DPP_UNUSED` 0x100 as
  undefined/reserved and `DPP_ROW` 0x150 through 0x165 as row broadcast at
  `cdna2/README.md:6818` through `:6840`.

XML evidence:

- The XML records the SDWA field widths and offsets but only describes
  `DST_SEL`, `SRC0_SEL`, and `SRC1_SEL` generically as destination/source data
  selects at `amdgpu_isa_cdna2.xml:4720` through `:4825`,
  `:5403` through `:5508`, and `:6641` through `:6686`.
- The XML records `DPP_CTRL` as a 9-bit "Data-parallel primitive control" field
  at `amdgpu_isa_cdna2.xml:4507` through `:4513` and `:5195` through `:5201`,
  but does not include the manual enumeration or reserved ranges.

Impact:

The XML exposes raw control bits but not the values needed to decide which SDWA
selectors are reserved, how destination-unused modes behave, which `DPP_CTRL`
values are undefined, or how the CDNA2 row-broadcast range should be decoded.

### CDNA2-XML-067: SDWAB `SD` bit prose drifts from the manual

Manual evidence:

- Chapter 13.3.8 defines SDWAB for VOPC and says bit 47 is `SD`, selecting the
  scalar destination type: `0 = VCC, 1 = normal SGPR` at
  `cdna2/README.md:6759` through `:6773`.

XML evidence:

- The XML field layout keeps `SD` at bit 47 for both
  `VOP2_VOP_SDWA_SDST_ENC` and `VOPC_VOP_SDWA_SDST_ENC`, but its description
  says "If 1 SDST used. If 0 bits 8:15 used as SDWA destination control" at
  `amdgpu_isa_cdna2.xml:5674` through `:5675` and `:6591` through `:6592`.

Impact:

The XML does not preserve the VOPC SDWAB scalar-destination rule in the same
terms as the manual. In particular, a consumer cannot infer from the XML prose
that `SD=0` means the compare result targets VCC rather than a normal SGPR.

### CDNA2-XML-068: DPP mask prose loses the destination-write/source-fetch distinction

Manual evidence:

- Table 82 says `BANK_MASK` and `ROW_MASK` apply only to the VGPR destination
  write and do not affect the source VGPR fetch mask at `cdna2/README.md:6815`
  through `:6816`.
- The same table describes `BC` as bounds control for out-of-range source lanes
  at `cdna2/README.md:6803` through `:6809`.

XML evidence:

- The XML records the bank and row fields as lane enable masks at
  `amdgpu_isa_cdna2.xml:4487` through `:4493`, `:4537` through `:4543`,
  `:5175` through `:5181`, and `:5225` through `:5231`.
- The XML records the bit-51 behavior as `BOUND_CTRL`, saying 0 sets
  write-enable to zero and 1 sets source data to zero at
  `amdgpu_isa_cdna2.xml:4497` through `:4503` and `:5185` through `:5191`.

Impact:

The XML captures the field locations and some bounds behavior, but not the
manual's explicit warning that row and bank masks gate only destination writes.
That warning matters for emulators and dataflow tools because masked-off
destination lanes still do not imply a masked source VGPR fetch.

## No-Gap Notes

- The XML does include the raw VOP1/VOP2 DPP and SDWA field offsets checked in
  this slice: representative `VOP1_VOP_DPP`, `VOP1_VOP_SDWA`,
  `VOP2_VOP_DPP`, and `VOP2_VOP_SDWA` records are present at
  `amdgpu_isa_cdna2.xml:4386`, `:4620`, `:5085`, and `:5318`. The new gaps are
  about value semantics, VOPC DPP availability, and prose lost on top of those
  base field maps.
- The XML captures the CDNA2 SDWA and DPP source-zero marker conditions:
  `has_sdwa` checks selector 249 at `amdgpu_isa_cdna2.xml:4683` through
  `:4704`, `:5366` through `:5387`, and `:6524` through `:6545`; `has_dpp`
  checks selector 250 at `:4460` through `:4481` and `:5148` through `:5169`.
- The XML `BOUND_CTRL` wording includes the source-zero behavior for
  out-of-range DPP reads at `amdgpu_isa_cdna2.xml:4497` through `:4503` and
  `:5185` through `:5191`. `CDNA2-XML-068` is about the missing manual warning
  that row and bank masks are destination-write masks, not source-fetch masks.
- The XML includes the base CDNA2 VOP1, VOPC, VOP2, and VOP3 format records at
  `amdgpu_isa_cdna2.xml:1194`, `:1602`, `:1870`, and `:2554`; the Chapter 6
  gaps above are about stateful behavior layered on top of those encodings.
- The Chapter 12.17 SDWA exclusion list matches XML omissions: the listed
  `V_MAC_F32`, `V_MAC_F16`, and `V_FMAC_F32` rows have DPP but no SDWA
  encodings, `V_MADMK/MADAK` F32/F16 are literal-only, and
  `V_READFIRSTLANE_B32`, `V_CLREXCP`, and `V_SWAP_B32` omit DPP/SDWA forms.
- The Chapter 12.17 DPP exclusion gap is per opcode, not a blanket F64 rule:
  the CDNA2 manual's feature summary says MI200 supports DPP for 64-bit data
  types at `cdna2/README.md:297`, and XML has a DPP form for `V_FMAC_F64` at
  `amdgpu_isa_cdna2.xml:51947`.
- DPP exclusions for `V_MADMK/MADAK` F32/F16, `V_READFIRSTLANE_B32`,
  `V_CVT_F64_F32`, `V_RNDNE_F64`, `V_RCP_F64`, `V_RSQ_F64`, `V_SQRT_F64`,
  `V_CLREXCP`, `V_SWAP_B32`, and the listed 64-bit/class compare wildcard
  families are represented by missing DPP encodings in XML.
- Chapter 6.2.3 repeats the out-of-range VGPR source/destination contract
  already tracked in `CDNA2-XML-009`, so this slice did not add a duplicate XML
  gap for that prose.
- The CDNA2 Chapter 12.1 SOP2 opcode inventory matches the checked-in XML after
  normalizing the manual OCR wrap on `S_PACK_LL_B32_B16`: both list 53 opcodes,
  0 through 52, from `S_ADD_U32` to `S_PACK_HH_B32_B16`.
- The CDNA2 Chapter 12.2 SOPK opcode inventory matches the checked-in XML after
  accounting for the literal-only `S_SETREG_IMM32_B32` encoding: the manual and
  XML both omit opcode 19, both expose opcode 20 as a 32-bit-literal HWREG form,
  and both list opcode 21 as a PC-relative `S_CALL_B64` row at
  `cdna2/README.md:2954` through `:2980` and
  `amdgpu_isa_cdna2.xml:41888` through `:42480`.
- The XML records the SOPK old-destination dataflow needed by `S_ADDK_I32` and
  `S_MULK_I32`: `SDST` is both input and output for both rows, and only
  `S_ADDK_I32` has an implicit SCC output at `amdgpu_isa_cdna2.xml:42234`
  through `:42296`.
- The CDNA2 Chapter 12.3 SOP1 opcode inventory matches the checked-in XML after
  accounting for the markdown extraction gap around opcode 27: the detailed
  table jumps from `S_BITSET1_B32` to `S_GETPC_B64` at
  `cdna2/README.md:3041` through `:3048`, but the later CDNA2 opcode list
  includes `S_BITSET1_B64` at `cdna2/README.md:5542` through `:5544`, and the
  XML has opcode 27 at `amdgpu_isa_cdna2.xml:30546` through `:30572`.
- The CDNA2 Chapter 12.4 SOPC opcode inventory matches the checked-in XML:
  both list 20 opcodes, 0 through 19, from `S_CMP_EQ_I32` to `S_CMP_LG_U64` at
  `cdna2/README.md:3100` through `:3126` and
  `amdgpu_isa_cdna2.xml:39439` through `:41670`.
- Apart from the XML-only `S_TTRACEDATA` row recorded in `CDNA2-XML-059`, the
  CDNA2 Chapter 12.5/13.1.5 SOPP opcode inventory matches the checked-in XML:
  both sources list opcodes 0 through 21 and 23 through 30 from `S_NOP` to
  `S_ENDPGM_ORDERED_PS_DONE`.
- The CDNA2 Chapter 12.15/13.6.1 MIMG opcode inventory matches the checked-in
  XML: both list the same 25 opcodes, 0 through 5, 8 through 11, 14, 16
  through 28, and 32, from `IMAGE_LOAD` to `IMAGE_SAMPLE`.
- The manual markdown for `IMAGE_ATOMIC_OR` appears to have an OCR/extraction
  typo in the pseudocode at `cdna2/README.md:4981`; the XML description
  correctly records a bitwise OR operation at
  `amdgpu_isa_cdna2.xml:21084` through `:21085`, matching the surrounding
  atomic OR instruction pattern.
- The CDNA2 XML has a distinct `OP_SEL_HI_2` field at
  `amdgpu_isa_cdna2.xml:2055`, matching the VOP3P split high-selector field in
  the manual at `cdna2/README.md:6606` and `:6619`.
- `V_PK_MOV_B32` uses 64-bit XML operands at `amdgpu_isa_cdna2.xml:65954`,
  which matches the two-dword destination/source shape implied by section 6.7.
- CDNA2 MIX detailed instruction text and XML descriptions say multiply-add,
  while section 6.7 calls the operation fused. This is a manual-internal
  ambiguity, so it is not classified as an XML-only omission in this slice.
- The XML does include the CDNA2 Chapter 12.10 VOP3P MFMA/ACCVGPR opcode
  inventory: the 29 manual rows from `V_MFMA_F32_32X32X1F32` through
  `V_MFMA_F64_4X4X4F64` and `V_ACCVGPR_READ`/`V_ACCVGPR_WRITE` match XML
  opcode names, opcode numbers, operand classes, and operand sizes at
  `amdgpu_isa_cdna2.xml:65995` through `:67293`. The XML also includes the
  VOP3P-MFMA `ACC`/`ACC_CD`/`CBSZ`/`ABID`/`BLGP` fields at `:7295` through
  `:7392`; the gaps above are about missing per-opcode legality and semantic
  details, not absence of the base encoding.
- The XML does include useful direct PC and wait-count operand records:
  `S_SETPC_B64`/`S_SWAPPC_B64`/`S_RFE_B64` expose PC implicit operands at
  `amdgpu_isa_cdna2.xml:32231` through `:32335`, and `OPR_WAITCNT` exposes the
  `EXP`/`LGKM`/`VM` threshold fields at `:128441` through `:128480`. The
  Chapter 4 gaps above are about missing stateful semantics, not absence of
  base decode metadata.
- The scalar selector table itself is correctly represented for
  `POPS_EXITING_WAVE_ID`: the manual places it at selector 239 at
  `cdna2/README.md:1009`, and XML `OPR_SSRC` does the same at
  `amdgpu_isa_cdna2.xml:123258` through `:123261`.
- CDNA2 `S_MAX_I32` and `S_MAX_U32` tie behavior is consistent across the
  detailed manual and XML: the manual uses strict `S0 > S1` for both the
  selected value and SCC at `cdna2/README.md:2881` through `:2882`, and the XML
  says SCC is set iff the first value is selected at
  `amdgpu_isa_cdna2.xml:35157` through `:35297`.
- The XML does include the base CDNA2 SMEM encoding and representative operand
  widths: the `ENC_SMEM` field map is present at `amdgpu_isa_cdna2.xml:620`
  through `:785`, scalar load operands scale `SDATA` from 32 through 512 bits at
  `:26804` through `:27040`, and buffer/atomic entries use 128-bit scalar
  resource operands where expected. The Chapter 8 gaps above are about semantics
  layered on top of that decode metadata.
- The CDNA2 Chapter 12.6/13.2.1 SMEM opcode inventory matches the checked-in
  XML: both list the same 84 opcodes, including `S_MEMREALTIME` at opcode 37,
  from scalar loads/stores through buffer/scalar atomic forms at
  `cdna2/README.md:3199` through `:3317` and `:5739` through `:5831`.
- Several SMEM atomic pseudocode irregularities appear to be manual markdown/OCR
  extraction artifacts rather than XML drift: `S_BUFFER_ATOMIC_OR*` and
  `S_ATOMIC_OR*` rows omit the visible OR operator at `cdna2/README.md:3255`,
  `:3273`, `:3291`, and `:3314`, while the XML records bitwise OR entries; the
  64-bit SMEM min/max rows show `-=` at `cdna2/README.md:3268` through `:3271`
  and `:3304` through `:3307`, while XML describes selecting minimum/maximum
  values at `amdgpu_isa_cdna2.xml:27518` through `:27603` and `:28610` through
  `:28695`.
- The XML does include the complete CDNA2 MUBUF/MTBUF base encodings and opcode
  inventory checked for Chapters 12.13/12.14 and 13.5: `ENC_MUBUF` is present at
  `amdgpu_isa_cdna2.xml:2339` through `:2577`, `ENC_MTBUF` is present at
  `:2579` through `:2767`, and the manual/XML inventories both cover 76 MUBUF
  opcodes plus the 16 MTBUF opcodes from formatted/typed loads through D16,
  cache, LDS, integer atomic, floating atomic, and 64-bit atomic forms. The
  Chapter 9 gaps above are about contradictory field prose or missing semantic
  rules, not absence of the base decode surface.
- Several CDNA2 MUBUF table irregularities appear to be manual markdown/OCR
  extraction artifacts rather than XML drift: the Chapter 12.13 table has blank
  name cells for the D16 opcodes 9 and 12 at `cdna2/README.md:4776` and `:4779`,
  Chapter 13.5 collapses part of the MUBUF opcode table around
  `cdna2/README.md:7104` through `:7160`, the signed `BUFFER_LOAD_SBYTE_D16*`
  rows show zero-filled formulas at `cdna2/README.md:4808` and `:4810` while the
  XML sign-extends them at `amdgpu_isa_cdna2.xml:23700` through `:23749`, and
  the `BUFFER_ATOMIC_SUB_X2` row drops the subtract operator at
  `cdna2/README.md:4860` while XML records subtraction at
  `amdgpu_isa_cdna2.xml:25124` through `:25125`.
- The XML does include the CDNA2 DS, MUBUF, FLAT, and GLOBAL float-atomic opcode
  inventory checked in the Chapter 9.5 slice. The new float-atomic gaps are
  about missing numeric policy, stale return-mode wording, and wrong
  descriptions, not absence of the base instruction entries.
- The XML does include distinct CDNA2 `ENC_FLAT`, `ENC_FLAT_GLBL`, and
  `ENC_FLAT_SCRATCH` encodings plus the full Chapter 12.16/13.7 opcode
  inventory: 51 `ENC_FLAT` instructions, 53 `ENC_FLAT_GLBL` instructions, and
  22 `ENC_FLAT_SCRATCH` instructions, matching the manual's FLAT, Global, and
  Scratch opcode tables at `cdna2/README.md:5008` through `:5191` and
  `:7290` through `:7454`. The Chapter 10 gaps above are about reserved-bit
  drift and missing segment-specific semantic rules, not absence of the base
  flat-memory decode surface.
- Several CDNA2 FLAT/GLOBAL/SCRATCH table irregularities appear to be manual
  markdown/OCR extraction artifacts rather than XML drift: the signed
  `*_LOAD_SBYTE_D16*` rows show zero-filled formulas at
  `cdna2/README.md:5036` through `:5038`, `:5106` through `:5108`, and
  `:5144` through `:5146`, while the XML sign-extends those rows, for example
  at `amdgpu_isa_cdna2.xml:14731` through `:14778`, `:17269` through `:17317`,
  and `:20036` through `:20089`; several atomic OR/DEC and 64-bit min/max rows
  also drop visible operators in the markdown around `cdna2/README.md:5056`,
  `:5059`, `:5070` through `:5075`, `:5164`, `:5170`, and `:5183` through
  `:5188`.
- The Chapter 12.16 local legend lists `NV = Access to non-volatile memory` at
  `cdna2/README.md:5001`, but the detailed Chapter 13.7 FLAT-family field table
  has no `NV` field and does include `ACC` at `cdna2/README.md:7271` through
  `:7288`. XML likewise has no FLAT-family `NV` field, so the stray Chapter
  12.16 `NV` line is treated as a manual markdown/OCR artifact rather than an
  XML omission.
- The XML DS `OP` field uses an 8-bit width at `amdgpu_isa_cdna2.xml:2897`
  through `:2905`, matching the Chapter 13 DS format table at
  `cdna2/README.md:6858` through `:6860` and the opcode table values up to
  `DS_READ_B128` opcode 255. The Chapter 11 local field table's 7-bit `OP`
  size is treated as a manual-internal typo rather than an XML gap.
- The CDNA2 Chapter 12.12/13.4 DS opcode inventory matches the checked-in XML:
  both list the same 124 `ENC_DS` opcodes, including the holes and the
  high-numbered `DS_GWS_*`, `DS_CONSUME`, `DS_APPEND`, `DS_WRITE_B96`,
  `DS_WRITE_B128`, `DS_READ_B96`, and `DS_READ_B128` rows. The Chapter 12.12
  detailed markdown table ends at `DS_READ_B96` at `cdna2/README.md:4633`,
  but the Chapter 13.4 opcode table includes `DS_READ_B128` at
  `cdna2/README.md:7007` through `:7008`, and XML records the same opcode at
  `amdgpu_isa_cdna2.xml:13862`.
- The XML does include the base CDNA2 DS encoding and opcode inventory:
  `ENC_DS` exposes `OFFSET0`, `OFFSET1`, `GDS`, `OP`, `ACC`, `ADDR`, `DATA0`,
  `DATA1`, and `VDST`, and representative DS load/store/atomic/GWS entries are
  present. The Chapter 11 gaps above are about selector omissions,
  source-class ambiguity, and missing semantic rules, not absence of DS decode
  metadata.
- CDNA2 `V_SWAP_B32` XML already marks both operands as read-write in both
  VOP1 and VOP3 forms at `amdgpu_isa_cdna2.xml:51138` through `:51178`; the
  CDNA4 output-only XML issue does not carry over to this architecture.
- The checked CDNA2 VOP2 `V_MAC_F16` definition at `cdna2/README.md:3386` does
  not include the later CDNA4 OPSEL destination-half preservation text, so that
  CDNA4-specific native-F16 gap is not recorded for CDNA2 in this slice.
- CDNA2 VOP1-as-VOP3 opcode records match the generated `+0x140` promotion
  numbers, for example `V_READFIRSTLANE_B32` opcode 2 maps to VOP3 opcode 322
  at `amdgpu_isa_cdna2.xml:43591` through `:43599`; the manual sentence at
  `cdna2/README.md:3702` appears to say "VOP2 opcode" where it means VOP1
  opcode.
- The CDNA2 VOP3A/VOP3B opcode inventory for Chapter 12.11 matches the checked
  XML after accounting for the four detailed table rows that use
  `VOP3_SDST_ENC` instead of `ENC_VOP3`: no manual opcode/name row from 448
  through 673 is missing from the XML, and the XML has no extra opcode/name in
  that slice.
- XML marks `V_CVT_PKACCUM_U8_F32`'s `VDST` operand as both input and output at
  `amdgpu_isa_cdna2.xml:62124` through `:62137`, matching the manual's note at
  `cdna2/README.md:4329` that the destination is passed in as a source.
- XML explicitly says `V_CVT_PKRTZ_F16_F32` uses round-toward-zero semantics
  and ignores the current rounding mode at `amdgpu_isa_cdna2.xml:64107`
  through `:64108`, matching the manual at `cdna2/README.md:4393`.
