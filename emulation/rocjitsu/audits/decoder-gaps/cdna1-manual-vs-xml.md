# CDNA1 Manual vs XML Gaps

Architecture: CDNA1

Manual source: `workspace_docs/amdgpu-isa-manuals/cdna1/README.md`

XML source: `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml`

## Coverage

| Manual section | Status | Notes |
| --- | --- | --- |
| 1. Introduction and 1.1 Terminology | Audited statically | Checked processor overview, host/command-processor/memory-controller roles, dispatch/workgroup/wavefront/work-item terms, 64-lane wavefront model, literal/microcode terminology, SALU/VALU roles, texture/buffer resource terms, FP-exception/interrupt prose, automatic instruction fetch, and latency-hiding overview against XML top-level and functional-group metadata. |
| 2.1 through 2.3 Program Organization | Audited statically | Checked kernel grouping into 64-lane wavefronts, SALU/VALU/vector-memory roles, `EXEC`-masked vector execution, 1D/2D/3D dispatch indexing, LDS topology, GDS topology/preload/writeback/append-consume prose, and device-memory cache/acknowledgment/relaxed-consistency model against XML global metadata, DS fields, and functional-group descriptions. |
| 3.1 through 3.6 Kernel State | Audited statically | Checked state inventory, PC/EXEC/STATUS/MODE prose, status and mode bitfields, GPR/LDS allocation and aliasing rules, out-of-range behavior, and LDS allocation/clamping. |
| 3.7 through 3.11 Kernel State | Audited statically | Checked M0 descriptor roles, SCC summary, VCC/VCCZ update and alias-hazard prose, trap/exception/TRAPSTS state, and memory-violation state. |
| 4.1 through 4.6 Program Flow Control | Audited statically | Checked program-control instruction inventory, ordinary and debug branches, direct PC operations, calls, fork/join branch-stack control flow, workgroup barriers, wait-counter dependency rules, manually inserted wait-state hazards, and adjacent SOP instruction-definition rows against XML instruction flags, operand metadata, and message/wait operands. |
| 5.1 through 5.2 Scalar ALU Formats and Operands | Audited statically | Checked SALU format fields, scalar operand selector table, literal availability/exceptions, source/destination out-of-range rules, `POPS_EXITING_WAVE_ID`, and 64-bit SGPR alignment against XML format and operand definitions. |
| 5.3 through 5.7 Scalar ALU Data Operations | Audited statically | Checked SCC producer/consumer prose, signed overflow versus carry behavior, comparison/conditional/bitwise SCC effects, strict CDNA1 `S_MAX_{I32,U32}` tie behavior, `S_ABSDIFF_I32` edge examples, and `S_MOVRELS`/`S_MOVRELD` M0 indexing against representative XML instruction entries. |
| 5.8 Access Instructions | Audited statically | Checked HWREG layout, register ID table, allocation/status subfields, access permissions, `S_SETREG` spacing prose, `S_SETREG_IMM32_B32` literal form, and access-instruction operand metadata. |
| 6.1 through 6.6 Core Vector ALU Operations | Audited statically | Checked VOP1/VOP2/VOPC/VOP3 format roles, VALU source and literal restrictions, EXEC-masked writes, compare/carry destinations, VOP3 output modifiers, FP round/denorm MODE behavior, ALU clamp semantics, out-of-range VGPR behavior overlap, and VGPR indexing rules against XML encodings, operand classes, and representative instruction rows. |
| 6.7 Packed Math | Audited for VOP3P packed-math slice only | Checked packed opcode inventory, MIX wording, DPP reachability, and output-modifier/clamp fields. |
| 7.1 Matrix Arithmetic Opcodes | Audited statically | Checked miSIMD AccVGPR state, Arch/AccVGPR data movement, MFMA opcode families, C/D and A/B format naming, source/register-bank rules, inline-constant restrictions, and exception wording against XML metadata. |
| 7.2 Dependency Resolution: Required NOPs | Audited statically | Checked MAI producer/consumer wait prose and MFMA pass-count dependencies against XML metadata. |
| 8.1 through 8.4 Scalar Memory Operations | Audited statically | Checked SMEM field layout, IMM/SOE/M0 offset forms, scalar/scratch/buffer addressing, source-overwrite and clause rules, atomics/GLC return, cache/time/probe/discard operations, LGKM dependency accounting, alignment, bounds, and Chapter 12/13 SMEM definitions against XML fields and representative instruction rows. |
| 9.1 Vector Memory Buffer Instructions | Audited statically | Checked MUBUF/MTBUF field meanings, VGPR address/data usage, data-format/number-format/D16/`dst_sel` routing, buffer resource descriptors, range checking, swizzled addressing, buffer-to-LDS behavior, and GLC/SLC/TFE side effects against XML encodings, resource operands, data formats, and instruction rows. |
| 9.2 through 9.4.5 Vector Memory Image Instructions | Audited statically | Checked MIMG field meanings, no-sampler and sampler address tables, VGPR data/`DMASK` usage, image atomic `DMASK` rules, D16/A16 behavior, image resource/sampler descriptors, image data formats, cache/status fields, and VMEM dependency prose against XML MIMG fields, descriptor formats, and representative instruction entries. |
| 10.1 through 10.8 Flat Memory Instructions | Audited statically | Checked FLAT/GLOBAL/SCRATCH field behavior, segment naming, flat/private/shared aperture routing prose, scratch and global address forms, direct LDS movement, atomics, ordering, wait-counter behavior, memory-error policy, data movement, scratch-space state, and the corresponding flat/global/scratch field and instruction metadata. |
| 11.1 through 11.4 Data Share Operations | Audited statically | Checked LDS geometry/dataflow, direct and parameter reads, indexed load/store/atomic addressing, M0 clamping, READ2/WRITE2 duplicate-offset behavior, SRC2 opcodes, FP atomic mode rules, DS lane-routing operations, append/consume, and GWS restrictions against XML DS fields and representative instruction rows. |
| 12.1 SOP2 Instructions | Audited statically | Checked the full SOP2 opcode inventory, literal-extension forms, arithmetic/carry/overflow equations, bitfield rows, fork/RFE compatibility rows, absolute-difference examples, shifted-add carry rows, and pack half-selection. |
| 12.2 SOPK Instructions | Audited statically | Checked the full SOPK opcode inventory, opcode-19 hole, opcode-20 literal-only form, SIMM16 signed/unsigned extension rules, ADDK/MULK destination dataflow, fork/call PC formulas, and HWREG rows. |
| 12.3 SOP1 Instructions | Audited statically | Checked the full SOP1 opcode inventory, literal variants, unary SCC/bit-count/bit-scan/bitset rows, direct PC/RFE rows, saveexec/wrexec formulas, quad-mask/bitreplicate, relative SGPR addressing, join, and GPR-index control. |
| 12.4 SOPC Instructions | Audited statically | Checked the full SOPC opcode inventory, compare and bit-compare SCC formulas, VSKIP row, `S_SET_GPR_IDX_ON` raw `SIMM4` field behavior, literal variants, and adjacent U64 compare rows. |
| 12.5 SOPP Instructions | Audited statically | Checked the full SOPP opcode inventory, PC-relative branch formulas, wait-count field layout, barrier/status/sleep/priority/message/trap/cache/perf rows, GPR-index control rows, `S_ENDPGM*` variants, send-message payload table, and SOPP opcode table against XML instruction entries. |
| 12.6 SMEM Instructions | Audited statically | Checked the full SMEM opcode inventory, scalar/scratch/buffer load-store rows, cache/time/probe/discard rows, scalar-buffer and scalar atomic rows, markdown formatting artifacts in atomic pseudocode, and adjacent Chapter 8 SMEM semantic gaps. |
| 12.7 VOP2 Instructions | Audited statically | Checked the full VOP2 opcode inventory, literal-only FMA rows, promoted VOP3 forms, bitwise no-modifier rows, accumulator old-destination dataflow, packed F16 accumulator row, and adjacent Chapter 6 VALU semantic gaps. |
| 12.8 VOP1 Instructions | Audited statically | Checked the full VOP1 opcode inventory, literal/DPP/SDWA availability, readfirstlane lane-selection and LDS-direct prose, special screen-partition and packed-saturate rows, legacy exp/log rows, swap source/destination dataflow, and promoted VOP3 wording. |
| 12.9 VOPC Instructions | Audited statically | Checked COMPF/COMPI opcode families, class-mask bit meanings, VCC/EXEC result behavior, literal support, promoted VOP3 compare wording, `CLAMP` exception semantics, and summary-table description drift against detailed rows and XML metadata. |
| 12.10 VOP3P Instructions | Audited statically | Checked the full VOP3P instruction table: packed 16-bit arithmetic, MIX, DOT, full MFMA inventory, `V_ACCVGPR_READ`/`V_ACCVGPR_WRITE`, absence of packed F32 / `V_PK_MOV_B32`, and existing opcode-table drift for `V_ACCVGPR_WRFED` against XML metadata. |
| 12.11 VINTERP Instructions | Audited statically | Checked compact F32 interpolation definitions, promoted VOP3 forms, VOP3 F16 interpolation rows, textual-versus-encoding source order, parameter selector metadata, and F16 LDS/half-selection behavior. |
| 12.12 VOP3A & VOP3B Instructions | Audited statically | Checked native VOP3A opcode inventory, VOP3B scalar-destination opcode set, carry/div-scale/wide-MAD scalar-mask semantics, VOP3A/VOP3B field maps, and adjacent generic VALU semantic gaps. |
| 12.13 LDS & GDS Instructions | Audited statically | Checked DS field prose, detailed DS opcode definitions, ADDTID/READ2/WRITE2/SRC2/FP atomic/lane-routing/GWS/B96/B128 rows, swizzle-mode details, and GDS-only limitations against XML metadata. |
| 12.14 MUBUF Instructions | Audited statically | Checked MUBUF field prose, formatted/raw/D16/D16_HI opcodes, load-to-LDS/cache-maintenance rows, integer and floating atomic rows, and sparse opcode holes against XML `ENC_MUBUF` rows and existing Chapter 9.1 semantic gaps. |
| 12.15 MTBUF Instructions | Audited statically | Checked MTBUF field prose and the typed formatted/D16 load-store opcode rows against XML `ENC_MTBUF` rows, including the Chapter 12 prose mention of LDS versus the Chapter 13/XML MTBUF field map. |
| 12.16 MIMG Instructions | Audited statically | Checked MIMG field prose, no-sampler load/store/query rows, image atomics, sampler/sample/gather families, sparse opcode holes, and CD derivative variants against XML `ENC_MIMG` rows and existing Chapter 9 image semantic gaps. |
| 12.17 FLAT, Scratch and Global Instructions | Audited statically | Checked FLAT, SCRATCH, and GLOBAL load/store/D16 rows, integer atomics, GLOBAL-only floating atomics, and sparse opcode holes against XML `ENC_FLAT`, `ENC_FLAT_GLBL`, and `ENC_FLAT_SCRATCH` rows and existing Chapter 10 semantic gaps. |
| 12.18 Instruction Limitations | Audited statically | Checked DPP and SDWA exclusion lists against XML extension availability, LLVM assembler behavior for representative rows, and adjacent generated rocjitsu constructor legality. |
| 13.1.1 SOP2 | Audited for SOP2 field/opcode slice | Checked SOP2 field map, source literal selector, opcode table, and source/destination operand metadata against the Chapter 12.1 definition rows. |
| 13.1.2 SOPK | Audited for SOPK field/opcode slice | Checked SOPK field map, destination-as-source wording, 32-bit versus 64-bit SOPK encodings, opcode table, label operands, implicit SCC/PC operands, and `S_SETREG_IMM32_B32` literal metadata. |
| 13.1.3 SOP1 | Audited for SOP1 field/opcode slice | Checked SOP1 field map, source literal selector, opcode table including the OCR-mangled middle rows and opcode holes 47/49, literal variants, and implicit SCC/EXEC/PC/M0 metadata. |
| 13.1.4 SOPC | Audited for SOPC field/opcode slice | Checked SOPC field map, source selector/literal fields, opcode table 0-19, SCC/MODE/M0 metadata, bit-index operand sizes, and the `S_SET_GPR_IDX_ON` raw `SIMM4` special case. |
| 13.1.5 SOPP | Audited for SOPP field/opcode slice | Checked SOPP field map, opcode table 0-30, label/wait/message/immediate operand classes, implicit SCC/VCC/EXEC/M0 metadata, and opcode-hole behavior past opcode 30. |
| 13.2.1 SMEM | Audited for SMEM field/opcode slice | Checked SMEM field map, `SBASE`/`SDATA`/`SOFFSET` operand widths, `IMM`/`SOE`/`GLC`/`NV` fields, opcode table 0-172 with sparse holes, scalar-buffer resource operands, ATC payload operands, and cache/time/discard rows. |
| 13.3.1 VOP2 | Audited for VOP2 field/opcode slice | Checked compact VOP2 field map, opcode table 0-61, literal-extension rows, promoted VOP3 metadata, implicit VCC operands, and old-destination accumulator metadata against Chapter 12.7 definitions. |
| 13.3.2 VOP1 | Audited for VOP1 field/opcode slice | Checked compact VOP1 field map, opcode table 0-81 with holes 9/54/56/80, XML opcode rows, literal/DPP/SDWA variants, promoted VOP3 metadata, and special operand classes against Chapter 12.8 definitions. |
| 13.3.3 VOPC | Audited for VOPC field/opcode slice | Checked compact VOPC field map, opcode table 16-255 with holes 0-15/22-31/128-159, literal and SDWA variants, promoted VOP3 metadata, VCC/EXEC operand metadata, and class-mask rows against Chapter 12.9 definitions. |
| 13.3.4 VOP3A and 13.3.5 VOP3B | Audited for VOP3A/VOP3B field/opcode slice | Checked VOP3A `VDST`/`ABS`/`OPSEL`/`CLMP`/`OMOD`/`NEG` field metadata, VOP3B `SDST` field metadata, native opcode tables including VOP3 F16 interpolation, VOP2/VINTRP-promoted opcodes, and XML encoding split between `ENC_VOP3` and `VOP3_SDST_ENC`. |
| 13.3.6 VOP3P and VOP3P-MAI | Audited statically | Checked `OPSEL_HI2`, `OPSEL_HI`, `CLMP`, ordinary VOP3P source-selector fields, VOP3P-MAI `CBSZ`/`ABID`/`ACC`/`BLGP`, full MFMA opcode rows, AccVGPR move opcode table coverage, and the opcode-92 `V_ACCVGPR_WRFED` manual/XML drift. |
| 13.3.7 through 13.3.9 SDWA/SDWAB/DPP | Audited statically | Checked SDWA, SDWAB, and DPP second-word field maps, selector value meanings, DPP control enumeration, and extension-family availability against XML format records. |
| 13.4.1 VINTRP | Audited statically | Checked compact VINTRP field map, opcode table, promoted VOP3 opcodes, `VDST` accumulator metadata, `OPR_PARAM` values, and adjacent interpolation M0 restrictions. |
| 13.5.1 DS | Audited statically | Checked DS field map and opcode table against Chapter 12.13 definitions, including opcode holes, SRC2 absence from XML, GWS/GDS-only rows, `DS_CONDXCHG32_RTN_B64`, and D16 high-half rows. |
| 13.6.1 MTBUF | Audited statically | Checked MTBUF field bit ranges, `DFMT`/`NFMT`, `SRSRC` alignment padding, `TFE`, absence of an LDS bit, and opcode table rows 0 through 15 against XML encoding metadata. |
| 13.6.2 MUBUF | Audited statically | Checked MUBUF field bit ranges, `LDS`/`SLC`/`TFE`, `SRSRC` alignment padding, sparse opcode table rows 0 through 39, 61 through 78, and 96 through 108 against XML encoding metadata. |
| 13.7.1 MIMG | Audited statically | Checked MIMG split opcode field, `DMASK`/`UNRM`/`GLC`/`DA`/`A16`/`TFE`/`LWE`/`SLC`/`D16` fields, descriptor base padding, and sparse opcode table rows 0 through 111 against XML encoding metadata. |
| 13.8 Flat Formats | Audited statically | Checked FLAT, GLOBAL, and SCRATCH field maps, offset-width and signedness split, `SEG` encodings, `SADDR`/`LDS`/`NV` fields, and opcode tables against XML encoding metadata. |
| Remaining CDNA1 manual sections | None recorded | Chapter 1 through Chapter 13 now have section-level static coverage entries; future work is deeper oracle/fuzz validation rather than unrecorded manual-section discovery. |

## Gaps

### CDNA1-XML-001: MIX-specific VOP3P selector and modifier overloads are prose-only

Manual evidence:

- Section 6.7 says `V_MAD_MIX_*` uses VOP3P but is not packed math at
  `cdna1/README.md:1517` through `:1521`.
- Each MIX instruction says source size/location is controlled by `OPSEL`,
  choosing between full FP32, low FP16, and high FP16 inputs, and that `NEG_HI`
  acts as an absolute-value modifier at `cdna1/README.md:4140` through `:4142`.

XML evidence:

- Generic VOP3P says `NEG_HI` negates high inputs at
  `amdgpu_isa_cdna1.xml:2099`, and says `OP_SEL`/`OP_SEL_HI` choose lower or
  upper 16-bit inputs at `:2119` through `:2139`.
- `V_MAD_MIX_F32`, `V_MAD_MIXLO_F16`, and `V_MAD_MIXHI_F16` are present at
  `amdgpu_isa_cdna1.xml:68312`, `:68359`, and `:68406`, but their entries do
  not encode the MIX-only selector mapping or `NEG_HI` absolute-value behavior.

Impact:

The generic XML field descriptions imply the wrong interpretation for MIX. XML
consumers need a hard-coded MIX override or manual-derived metadata.

### CDNA1-XML-002: MFMA inline-constant and exception rules are prose-only

Manual evidence:

- Chapter 7 says MFMA instructions do not support SGPRs, shared/private
  sources, DPP, SDWA, VCCZ, EXECZ, SCC, LDS_DIRECT, or LITERAL, and that the
  only inline constants interpreted as FP32 for `V_MFMA` and `V_ACCVGPR` are
  `0.5`, `-0.5`, `1.0`, `-1.0`, `2.0`, `-2.0`, `4.0`, and `-4.0` at
  `cdna1/README.md:1574` through `:1581`.
- The later generic VOP3P-MAI field table lists broader source-encoding
  options, including integer constants and `1/(2*PI)`, at
  `cdna1/README.md:6755` through `:6757`; the narrower Chapter 7 MFMA rule is
  not carried as per-instruction metadata.
- The manual states that miSIMD does not support arithmetic exceptions at
  `cdna1/README.md:1581`.

XML evidence:

- Representative MFMA entries encode SRC0/SRC1 as
  `OPR_SRC_VGPR_OR_ACCVGPR` and SRC2 as `OPR_SRC_ACCVGPR_OR_CONST` at
  `amdgpu_isa_cdna1.xml:68782` through `:68818` and `:69563` through `:69602`,
  but do not attach the Chapter 7 source-form or exception restrictions to the
  instructions.
- The `OPR_SRC_ACCVGPR_OR_CONST` operand class includes generic integer
  constants and `0.15915494` in addition to the MFMA-specific FP32 constant
  subset at `amdgpu_isa_cdna1.xml:100301` through `:100340` and
  `:101980` through `:102035`.

Impact:

XML-only consumers cannot derive MFMA-only source legality, inline-constant
legality, or miSIMD arithmetic-exception behavior without manual-derived
overrides.

### CDNA1-XML-003: MFMA dependency-wait rules and pass counts are prose-only

Manual evidence:

- Section 7.2 gives required NOP or independent-instruction distances for VALU,
  MFMA, `V_ACCVGPR_READ`, `V_ACCVGPR_WRITE`, and `V_CMPX` producer/consumer
  combinations at `cdna1/README.md:1583` through `:1616`.
- The detailed opcode table includes block counts and pass counts for each MFMA
  shape at `cdna1/README.md:4155` through `:4181`; the dependency table uses
  those pass counts to choose required wait distances.

XML evidence:

- The CDNA1 XML VOP3P-MFMA encoding records opcode identifiers and field layout
  at `amdgpu_isa_cdna1.xml:7483` through `:7564`, but does not attach
  producer/consumer classes, pass counts, forwarding exceptions, or required
  wait distances to the opcodes.

Impact:

XML-only consumers can decode CDNA1 MFMA instructions, but cannot derive the
software scheduling contract that the manual assigns to MAI sequences.

### CDNA1-XML-004: VOP3P-MAI `CBSZ` legality is under-described

Manual evidence:

- The VOP3P-MAI field table says `CBSZ` defines the number of blocks that can
  broadcast within a group, and that legal values are `0-4`, at
  `cdna1/README.md:6751`.

XML evidence:

- The XML field descriptions for `ABID`, `CBSZ`, and `BLGP` describe the broad
  swizzle controls at `amdgpu_isa_cdna1.xml:7527` through `:7564`, but `CBSZ`
  is only represented as a 3-bit field with no encoded legal-value
  restriction.

Impact:

Consumers generated from XML alone accept all 3-bit `CBSZ` values, while the
manual only defines `0-4` for CDNA1 VOP3P-MAI.

### CDNA1-XML-005: SALU out-of-range source and destination behavior is prose-only

Manual evidence:

- Section 5.2 says SALU source selector 255 consumes the following instruction
  dword as a literal, while SOPP and SOPK are the ordinary exceptions, at
  `cdna1/README.md:1021`.
- The same section says an out-of-range source SGPR reads SGPR0, an
  out-of-range destination SGPR suppresses the SGPR write while still allowing
  SCC and saveexec EXEC side effects, and 64-bit SGPR data must use an
  even-aligned pair at `cdna1/README.md:1023` through `:1027`.

XML evidence:

- The XML scalar source operand table records special selectors such as
  `src_vccz`, `src_execz`, `src_scc`, aperture selectors, and
  `src_pops_exiting_wave_id` at
  `amdgpu_isa_cdna1.xml:97682` through `:97720`.
- Representative SALU entries expose literal alternatives and implicit SCC
  outputs, for example `S_ADD_I32` at `amdgpu_isa_cdna1.xml:35554` through
  `:35640`.
- Static XML search did not find metadata for the runtime SGPR0 fallback, the
  out-of-range destination no-write rule, or the SCC/EXEC side effects that
  survive a suppressed destination write.

Impact:

XML consumers can recover SALU operand classes, literals, and many SCC
read/write operands, but need manual knowledge to implement allocation-bound
source fallback and destination suppression correctly.

### CDNA1-XML-006: Signed SALU add/sub SCC descriptions contradict overflow prose

Manual evidence:

- Section 5.3 defines signed add/sub SCC as overflow, distinct from unsigned
  carry-out, at `cdna1/README.md:1033` through `:1038`.
- The detailed `S_ADD_I32` and `S_SUB_I32` rows use signed overflow predicates
  at `cdna1/README.md:2862` through `:2864`.

XML evidence:

- The XML entry for `S_ADD_I32` describes storing the carry-out bit into SCC at
  `amdgpu_isa_cdna1.xml:35554` through `:35555`, even though the operand
  metadata only records that SCC is written.

Impact:

Codegen or tests derived from the XML description can choose an unsigned
carry-out predicate for signed add/sub unless they cross-reference the manual
or detailed pseudocode.

### CDNA1-XML-007: `S_ABSDIFF_I32` wraparound examples are not machine-readable

Manual evidence:

- The detailed `S_ABSDIFF_I32` row performs a 32-bit signed subtract, then
  conditionally negates the 32-bit result, and gives edge examples such as
  `S_ABSDIFF_I32(0x80000000, 0x00000000) => 0x80000000` and
  `S_ABSDIFF_I32(0x80000000, 0x00000001) => 0x7fffffff` at
  `cdna1/README.md:2920`.

XML evidence:

- The XML entry says only that `S_ABSDIFF_I32` calculates the absolute value of
  the difference and sets SCC iff the result is nonzero, at
  `amdgpu_isa_cdna1.xml:40950` through `:40978`.

Impact:

XML-only consumers can reasonably implement a mathematical absolute difference,
but that disagrees with the manual's wrapped 32-bit edge behavior.

### CDNA1-XML-008: `S_MOVRELS` and `S_MOVRELD` M0 address formulas are prose-only

Manual evidence:

- Section 5.7 summarizes `S_MOVRELS_{B32,B64}` and
  `S_MOVRELD_{B32,B64}` as indexing `SGPR[S0+M0]` or `SGPR[D+M0]`, with M0 as
  an unsigned index and the 64-bit index required to be even, at
  `cdna1/README.md:1132`.
- The detailed SOP1 rows repeat the raw address formulas `addr += M0.u` for
  both source-relative and destination-relative moves at
  `cdna1/README.md:3061` through `:3064`.

XML evidence:

- The XML exposes M0 as an implicit operand on `S_MOVRELS_B32`,
  `S_MOVRELS_B64`, `S_MOVRELD_B32`, and `S_MOVRELD_B64` at
  `amdgpu_isa_cdna1.xml:34448` through `:34688`.
- Those entries do not encode the effective-address formula, full-width M0
  addition, or the even-M0 requirement for the 64-bit forms.

Impact:

XML-derived decoders can see that M0 participates in relative scalar moves, but
cannot derive the exact indexing rule or undefined odd-M0 64-bit case.

### CDNA1-XML-009: HWREG map and operand metadata are incomplete or inconsistent

Manual evidence:

- Section 5.8 defines `SIMM16 = {size[4:0], offset[4:0], hwRegId[5:0]}`,
  with offset 0 through 31 and size 1 through 32, at `cdna1/README.md:1146`
  through `:1148`.
- Table 19 defines CDNA1 hardware register IDs, including `MODE` at 1,
  read-only `STATUS` at 2, `TRAPSTS` at 3, `HW_ID` at 4, `GPR_ALLOC` at 5,
  `LDS_ALLOC` at 6, `IB_STS` at 7, reserved IDs 8 through 15, and TBA/TMA IDs
  16 through 19 at `cdna1/README.md:1150` through `:1171`.
- Tables 20 and 21 define bit layouts for `IB_STS` and `GPR_ALLOC` at
  `cdna1/README.md:1173` through `:1189`.

XML evidence:

- `OPR_HWREG` records the raw `ID`, `OFFSET`, and `SIZE` fields, but leaves
  field and predefined-value descriptions empty at
  `amdgpu_isa_cdna1.xml:94903` through `:95036`.
- The XML assigns IDs 8 through 15 to `hw_reg_pc_lo`, `hw_reg_pc_hi`,
  `hw_reg_inst_dw0`, `hw_reg_inst_dw1`, `hw_reg_ib_dbg0`, `hw_reg_ib_dbg1`,
  `hw_reg_flush_ib`, and `hw_reg_sh_mem_bases` at
  `amdgpu_isa_cdna1.xml:94952` through `:94990`, while the CDNA1 manual
  reserves those IDs.
- The XML instruction entries for `S_GETREG_B32`, `S_SETREG_B32`, and
  `S_SETREG_IMM32_B32` mark the `SIMM16`/`OPR_HWREG` operand with
  `OperandSize` 32 despite `FMT_NUM_B16` at
  `amdgpu_isa_cdna1.xml:45240` through `:45309`.

Impact:

XML-only consumers can parse the HWREG operand partition, but not the CDNA1
register semantics, access permissions, or allocation/status subfields. They
also see an ID map that contradicts the manual for IDs 8 through 15 and an
operand-size inconsistency that can leak into generated operand metadata.

### CDNA1-XML-010: SETREG permissions and spacing are prose-only

Manual evidence:

- Section 5.8 marks `MODE` and `TRAPSTS` as read/write, `STATUS`, `HW_ID`,
  `GPR_ALLOC`, `LDS_ALLOC`, and `IB_STS` as read-only, and requires an
  `S_NOP` between consecutive `S_SETREG` writes to the same register at
  `cdna1/README.md:1142` through `:1167`.
- The detailed SOPK rows describe `S_SETREG_B32` and `S_SETREG_IMM32_B32` as
  writing selected HWREG bits from an SGPR or literal at `cdna1/README.md:2965`
  through `:2967`.

XML evidence:

- The XML exposes `S_GETREG_B32`, `S_SETREG_B32`, and
  `S_SETREG_IMM32_B32` as HWREG access instructions at
  `amdgpu_isa_cdna1.xml:45240` through `:45315`, but does not encode
  per-register read/write permissions or the consecutive-SETREG spacing rule.

Impact:

An XML-derived emulator or validator cannot tell which CDNA1 HWREGs are
writable, which are read-only, or which instruction sequences require an
intervening scheduling NOP.

### CDNA1-XML-011: PC, EXEC, STATUS, and MODE state details are prose-only

Manual evidence:

- The Chapter 3 state table defines shader-visible PC, EXEC/EXECZ, VCC/VCCZ,
  SCC, FLAT_SCRATCH, XNACK_MASK, STATUS, MODE, M0, TRAPSTS, TBA/TMA, TTMPs, and
  wait counters at `cdna1/README.md:473` through `:506`.
- Section 3.2 defines PC initialization, even-SGPR-pair PC transfer, branch
  accounting from the instruction after the branch, and the special `S_TRAP`
  saved-PC rule at `cdna1/README.md:512` through `:516`.
- Sections 3.3 through 3.5 define EXEC masking, `EXECZ`, zero-EXEC/VSKIP
  guidance, read-only STATUS initialization, and MODE fields such as
  `FP_ROUND`, `FP_DENORM`, `FP16_OVFL`, `GPR_IDX_EN`, and `VSKIP` at
  `cdna1/README.md:520` through `:588`.

XML evidence:

- The XML contains instruction rows for direct PC operations such as
  `S_GETPC_B64`, `S_SETPC_B64`, `S_SWAPPC_B64`, and `S_RFE_B64` at
  `amdgpu_isa_cdna1.xml:33483` through `:33605`, plus branch predicate rows
  such as `S_CBRANCH_VCCZ` and `S_CBRANCH_EXECZ` at `:45546` through `:45630`.
- `OPR_HWREG` exposes raw `{ID, OFFSET, SIZE}` fields and HWREG names at
  `amdgpu_isa_cdna1.xml:94903` through `:94943`, but not the Chapter 3 STATUS
  or MODE bitfields and update rules.
- Searching the XML for representative field names including `FP_ROUND`,
  `FP_DENORM`, `FP16_OVFL`, `TRAP_EN`, `GPR_IDX_EN`, and `VSKIP` finds only
  the short `S_SETVSKIP` instruction description at
  `amdgpu_isa_cdna1.xml:44114` through `:44115`.

Impact:

XML-derived metadata can name the control-flow instructions and HWREG operand
shape, but cannot derive the wave-state initialization, helper-bit updates,
field writability, or MODE-controlled execution behavior without manual prose.

### CDNA1-XML-012: GPR/LDS allocation, aliasing, and non-SALU out-of-range behavior are prose-only

Manual evidence:

- Section 3.6.1 defines out-of-range behavior for SGPR, VGPR, LDS, memory/LDS/GDS
  return operations, PRT return VGPRs, and multi-destination instructions at
  `cdna1/README.md:595` through `:634`.
- Section 3.6.2 says SGPR allocation is 16 to 102 dwords in 16-dword units,
  VCC is physically aliased in the highest user SGPRs, and a trap handler
  reserves 16 privileged SGPRs after VCC at `cdna1/README.md:636`.
- Sections 3.6.3 through 3.6.5 define SGPR alignment, VGPR/AccVGPR allocation,
  LDS 128-dword allocation blocks, no-wrap LDS placement, and the
  `min(lds_size, M0)` clamp rule at `cdna1/README.md:640` through `:659`.

XML evidence:

- The XML gives generic SGPR, AccVGPR, and VGPR operand names and limited
  alignment text, for example `OPR_SDST` at `amdgpu_isa_cdna1.xml:95088`
  through `:95120`, `OPR_SREG` at `:113150` through `:113190`, `OPR_ACCVGPR` at
  `:93424` through `:93445`, and `OPR_VGPR` at `:117898` through `:117925`.
- `OPR_VGPR_OR_LDS` describes the LDS-direct M0 address form at
  `amdgpu_isa_cdna1.xml:119184` through `:119215`, but does not encode
  allocation bounds or the Chapter 3 clamp/nullification rules.
- The SALU-specific out-of-range omission is also tracked in `CDNA1-XML-005`;
  this entry covers the broader Chapter 3 allocation, aliasing, LDS, and
  memory-return contract.

Impact:

An XML-only validator or emulator cannot enforce CDNA1 register allocation
units, VCC physical aliasing, trap SGPR reserves, LDS allocation/clamping, or
the specified non-SALU out-of-range and nullification behavior.

### CDNA1-XML-013: M0, SCC, and VCC/VCCZ helper semantics are mostly prose-only

Manual evidence:

- Section 3.7 lists the distinct M0 payload layouts for LDS interpolation,
  LDS direct reads, memory/VFetch-to-LDS, GDS, indirect GPR indexing, and
  send-message data at `cdna1/README.md:663` through `:673`.
- Section 3.8 classifies SCC producer and consumer behavior at
  `cdna1/README.md:677` through `:683`.
- Section 3.9 says VCCZ updates every time VCC updates, including scalar writes
  to VCC; vector compares fully write VCC under EXEC; VCC physically aliases
  the highest user SGPRs; and scalar writes to that physical alias have a VCCZ
  branch hazard at `cdna1/README.md:687` through `:703`.

XML evidence:

- The XML names `src_vccz`, `src_execz`, and `src_scc` as scalar source values
  at `amdgpu_isa_cdna1.xml:97683` through `:97693`, and `OPR_VCC` describes
  only `VCC[63:0]` at `:117887` through `:117894`.
- `S_SET_GPR_IDX_MODE` exposes an implicit M0 output at
  `amdgpu_isa_cdna1.xml:46214` through `:46232`, and `OPR_VGPR_OR_LDS`
  includes one LDS-direct M0 layout at `:119184` through `:119215`.
- The XML does not encode the full Chapter 3 M0 role table, the SCC producer
  taxonomy, VCC full-write behavior, VCCZ update-on-scalar-write rule, or the
  physical VCC alias hazard.

Impact:

XML-derived instruction rows can mention the special registers, but cannot
derive their Chapter 3 state-transition rules or aliasing hazards.

### CDNA1-XML-014: Trap, exception, TRAPSTS, and memory-violation state is not machine-readable

Manual evidence:

- Section 3.10 defines TTMP write privilege, TBA/TMA read-only access, trap
  payload construction in TTMP1/TTMP0, `STATUS.TRAP_EN`, and `MODE.EXCP_EN` at
  `cdna1/README.md:707` through `:733`.
- Section 3.10.1 defines TRAPSTS sticky exception fields, `SAVECTX`,
  `ILLEGAL_INST`, address-watch bits, `EXCP_CYCLE`, and `DP_RATE` at
  `cdna1/README.md:735` through `:749`.
- Section 3.11 defines memory-violation sources, non-sources, buffer-to-LDS
  EXEC masking, sticky `TRAPSTS.mem_viol`, trap enable behavior, and imprecise
  saved-PC reporting at `cdna1/README.md:752` through `:771`.

XML evidence:

- `OPR_HWREG` names `hw_reg_trapsts` at `amdgpu_isa_cdna1.xml:94928`, but the
  HWREG descriptors in that table are otherwise empty.
- `S_RFE_B64` and `S_TRAP` have short instruction descriptions at
  `amdgpu_isa_cdna1.xml:33596` through `:33605` and `:45917` through `:45935`.
- Searching the CDNA1 XML for `SAVECTX`, `MEM_VIOL`, `ILLEGAL_INST`,
  `ADDR_WATCH`, `EXCP_CYCLE`, and `DP_RATE` finds no machine-readable state
  fields.

Impact:

XML consumers can identify trap-control opcodes and the TRAPSTS HWREG selector,
but cannot model trap entry, trap return, sticky exception state, memory
violation reporting, or TTMP/TBA/TMA access rules without manual-derived
metadata.

### CDNA1-XML-015: Processor overview and shader terminology are prose-only

Manual evidence:

- Chapter 1 describes the CDNA device as a DPP array, command processor, memory
  controller, and other logic, then explains the host/kernel split,
  command-driven launch, cache invalidation/flush commands, DMA-style
  memory-controller behavior, hardware interrupts, floating-point exception
  recording, automatic instruction fetch, and latency hiding at
  `cdna1/README.md:350` through `:379`.
- Section 1.1 defines dispatch, workgroup, 64-work-item wavefronts,
  work-items, literal constants, SALU, VALU, microcode format, instructions,
  quads, texture samplers/resources, and buffer resources at
  `cdna1/README.md:382` through `:409`.

XML evidence:

- The top-level CDNA1 XML architecture record only carries architecture name and
  numeric ID at `amdgpu_isa_cdna1.xml:1` through `:13`.
- XML does include useful local metadata, such as encoding descriptions for
  scalar/vector ALU families, VOPC `EXEC`-mask wording at
  `amdgpu_isa_cdna1.xml:1601`, literal-operand descriptions such as
  `:4526`, and functional-group descriptions for `SALU`, `SMEM`, `VALU`, and
  `VMEM` at `:120523` through `:120538`.
- Those records do not provide global metadata for wavefront size,
  dispatch-grid dimensionality, host/command-processor interaction,
  memory-controller/DMA behavior, FP-exception interrupt/recording behavior,
  automatic instruction fetch, latency hiding, or the resource terminology in
  Section 1.1.

Impact:

XML consumers can enumerate instructions, functional groups, and local operand
forms, but they still need manual prose for the architecture-level assumptions
that frame dispatch and execution.

### CDNA1-XML-016: Program organization, GDS topology, and device-memory consistency are prose-only

Manual evidence:

- Chapter 2 says kernels are grouped into 64-work-item wavefronts, control flow
  is handled by SALU instructions, vector ALU and vector-memory work is gated
  by `EXEC`, and vector compare/carry-out results return bit-per-work-item
  masks to SGPRs at `cdna1/README.md:410` through `:426`.
- Section 2.1 says dispatches cover 1D/2D/3D grids, generate wavefronts, and
  initialize each work-item with a unique grid index at `cdna1/README.md:430`
  through `:434`.
- Sections 2.2.1 and 2.2.2 describe a 64 KiB / 32-bank / 32-atomic-unit LDS
  and a 4 KiB GDS that is shared across CUs, supports preload/writeback,
  unordered append/consume, and ordered append/consume at
  `cdna1/README.md:446` through `:454`.
- Section 2.3 describes cache-less loads, load-clause overlap caching,
  write-combining, atomic pre-op return storage, write-confirmation
  acknowledgments, relaxed consistency, and per-PE/per-channel scatter-write
  ordering at `cdna1/README.md:458` through `:465`.

XML evidence:

- The top-level CDNA1 XML architecture record only carries architecture name and
  numeric ID at `amdgpu_isa_cdna1.xml:1` through `:13`.
- XML has useful local descriptions, such as VOPC compare `EXEC` wording at
  `amdgpu_isa_cdna1.xml:1601`, the DS encoding's broad "Local and global data
  share operations" description and `GDS` bit at `:2902` and `:2946` through
  `:2947`, DS/GWS opcode records such as `DS_GWS_*`, `DS_CONSUME`,
  `DS_APPEND`, and `DS_ORDERED_COUNT` at `:13533` through `:13812`, and
  functional-group descriptions at `:120523` through `:120538`.
- Those records still do not provide global metadata for wavefront size,
  dispatch-grid dimensionality, SALU/VALU control roles, `EXEC`-mask scope,
  LDS bank/atomic topology, GDS size/preload/writeback/append-consume
  machinery, or the Section 2.3 memory consistency/acknowledgment/order model.
- Existing narrower entries cover some downstream pieces: `CDNA1-XML-012` for
  GPR/LDS allocation and clamping, and `CDNA1-XML-013` for M0/SCC/VCC helper
  behavior.

Impact:

XML consumers can enumerate instructions, fields, and functional groups, but
still need manual prose for the processor-organization assumptions, GDS
topology, and top-level memory model that determine how decoded instructions
execute together.

### CDNA1-XML-017: MUBUF/MTBUF `OFFEN` descriptions contradict instruction-offset use

Manual evidence:

- Chapter 9.1.5 says buffer offsets can come from an SGPR, an optional VGPR,
  and the instruction itself at `cdna1/README.md:1878`; the instruction-field
  table says `inst_offset` is always present, regardless of `inst_offen`, at
  `:1880` through `:1886`.
- The swizzled-address formula explicitly adds `inst_offset` to the optional
  VGPR offset at `cdna1/README.md:1955` through `:1967`.

XML evidence:

- The CDNA1 XML `ENC_MUBUF` `OFFEN` field says "If set, send VADDR as an
  offset. If unset, send the instruction offset stored in OFFSET. Only one of
  these offsets may be sent" at `amdgpu_isa_cdna1.xml:3133` through `:3134`.
- `ENC_MTBUF` carries the same wording at `amdgpu_isa_cdna1.xml:3319` through
  `:3320`.

Impact:

XML-derived address builders can treat the VGPR offset and instruction offset
as mutually exclusive, while the manual says the instruction offset is always
part of the buffer-offset term and `OFFEN` only adds the VGPR contribution.

### CDNA1-XML-018: Buffer resource descriptor, range checking, and swizzle behavior are prose-only

Manual evidence:

- Chapter 9.1.5 defines buffer descriptor fields including base, stride,
  num-records, add-TID, swizzle enable, element size, and index stride at
  `cdna1/README.md:1894` through `:1906`.
- The same section defines private/scratch, raw, and structured range-check
  modes, including zero/drop behavior and per-component versus all-or-nothing
  checks, at `cdna1/README.md:1926` through `:1949`.
- Swizzled addressing has dword-alignment constraints, element-size/stride
  constraints, and an explicit final-address formula at
  `cdna1/README.md:1951` through `:1969`.
- Chapter 9.1.8 defines the full 128-bit buffer resource descriptor and says an
  all-zero resource acts as an unbound buffer at `cdna1/README.md:2010`
  through `:2040`.

XML evidence:

- XML resource data formats such as `FMT_RSRC`, `FMT_RSRC_SCRATCH`, and
  `FMT_RSRC_TYPED` are all opaque 128-bit `Descriptor` fields at
  `amdgpu_isa_cdna1.xml:93182` through `:93395`.
- Representative MTBUF and MUBUF instruction entries attach 128-bit resource
  operands, for example `TBUFFER_LOAD_FORMAT_X` at
  `amdgpu_isa_cdna1.xml:24393` through `:24397` and
  `BUFFER_LOAD_FORMAT_X` at `:25161` through `:25165`, but do not encode the
  descriptor subfields, swizzle formulas, range-check modes,
  unbound-resource behavior, or add-TID/stride-extension rules.

Impact:

XML-only consumers can identify that a 128-bit resource descriptor operand is
present, but cannot derive the CDNA1 buffer addressing, bounds, swizzle, and
unbound-resource contract required for emulation or validation.

### CDNA1-XML-019: Buffer data-format conversion, `dst_sel`, and D16 ECC details are prose-only

Manual evidence:

- Chapter 9.1.2 gives the `DFMT` and `NFMT` value tables at
  `cdna1/README.md:1811` through `:1812`.
- Chapter 9.1.3 says returned data is 32-bit, with float/normalized values
  returned as floats and integer values returned signed or unsigned based on
  memory format, at `cdna1/README.md:1836` through `:1844`.
- Chapter 9.1.4 says MTBUF uses instruction formats, MUBUF format operations
  use resource formats and `dst_sel`, other MUBUF operations derive formats
  from the opcode, invalid resource format means unbound resource, and D16
  packs converted 16-bit values into 32-bit VGPRs at
  `cdna1/README.md:1848` through `:1872`.
- Chapter 9.1.6 defines D16 low/high-half behavior and says ECC-enabled
  16-bit loads write the full 32-bit VGPR with unused bits zeroed at
  `cdna1/README.md:1995` through `:2004`.

XML evidence:

- `ENC_MTBUF` has field labels for `DFMT` and `NFMT` at
  `amdgpu_isa_cdna1.xml:3269` through `:3270` and `:3309` through `:3310`.
- Instruction descriptions identify formatted conversion for representative
  MTBUF and MUBUF rows at `amdgpu_isa_cdna1.xml:24373` through `:24374` and
  `:25141` through `:25142`, and D16 low/high-half rows at `:26677` through
  `:26726`.
- The XML does not provide structured conversion tables, resource `dst_sel`
  behavior, invalid-format/unbound interaction, or the D16 ECC overwrite rule.

Impact:

Opcode and operand inventory alone is not enough to implement CDNA1 typed and
resource-format buffer data conversion. Consumers need manual-derived tables
for component selection, numeric conversion, D16 packing, and ECC-dependent
writeback.

### CDNA1-XML-020: Buffer cache, TFE, LDS, and atomic-return side effects are under-described

Manual evidence:

- Chapter 9.1.2 overloads `GLC` by operation class, describes `SLC` as L2
  streaming mode, says `TFE` writes status to `DST+1`, and defines MUBUF `LDS`
  as returning read data to LDS instead of VGPRs at `cdna1/README.md:1819`
  through `:1822`.
- Chapter 9.1.9 limits load-to-LDS to a subset of MUBUF loads, makes `TFE`
  illegal for loads to LDS, derives the LDS offset from `M0[15:0]`, and
  requires active-mask clamping so writes stay within the wave's allocated LDS
  space at `cdna1/README.md:2042` through `:2066`.
- Chapter 9.1.10 expands `GLC` semantics separately for loads, stores, and
  atomics, including the floating-point atomic restriction at
  `cdna1/README.md:2068` through `:2094`.

XML evidence:

- `ENC_MUBUF` has one-line field descriptions for `GLC`, `LDS`, `SLC`, and
  `TFE` at `amdgpu_isa_cdna1.xml:3103` through `:3124` and `:3163` through
  `:3198`; `ENC_MTBUF` similarly has one-line descriptions for `GLC`, `SLC`,
  and `TFE` at `:3289` through `:3290` and `:3349` through `:3384`.
- The XML includes `BUFFER_STORE_LDS_DWORD` and cache-maintenance rows, for
  example at `amdgpu_isa_cdna1.xml:27061` through `:27121`, but it does not
  encode the load-to-LDS legal subset, `TFE` illegality, M0/LDS clamping rules,
  operation-specific `GLC` behavior, or floating-point atomic `GLC` restriction.

Impact:

XML consumers can see the relevant bits and opcodes, but need manual-derived
semantics for cache behavior, atomic return behavior, TFE status writes, and
the legal/clamped LDS data path.

### CDNA1-XML-021: MIMG cache/status and D16 field semantics are under-described

Manual evidence:

- Chapter 9.2.1 describes operation-specific `GLC` behavior for reads, writes,
  and atomics, `SLC` L2 streaming mode, `TFE` status returns, `LWE` status
  returns, `A16` address packing, and D16 load/store conversion restrictions at
  `cdna1/README.md:2136` through `:2141`.
- Chapter 13.6 repeats the MIMG field map for `DMASK`, `UNRM`, `GLC`, `DA`,
  `A16`, `TFE`, `LWE`, `SLC`, `SRSRC`, `SSAMP`, and `D16` at
  `cdna1/README.md:7348` through `:7364`.

XML evidence:

- The CDNA1 XML `ENC_MIMG` describes `D16` only as converting 32-bit floating
  point texture return data to 16-bit at `amdgpu_isa_cdna1.xml:3541` through
  `:3542`, omitting store behavior, opcode restrictions, integer-vs-float
  interpretation through `NFMT`, and D16 packing.
- `ENC_MIMG` uses generic `GLC`, `SLC`, `LWE`, `TFE`, and `UNORM`
  descriptions at `amdgpu_isa_cdna1.xml:3580` through `:3592`, `:3620`
  through `:3622`, and `:3659` through `:3670`, rather than the manual's
  operation-specific cache and status-return rules.

Impact:

XML consumers can decode the MIMG bits, but cannot derive the full CDNA1 image
status, cache, D16, and unnormalized-address contract from field metadata
alone.

### CDNA1-XML-022: Image address, `DMASK`, and atomic data-count rules are prose-only

Manual evidence:

- Chapter 9.3 gives no-sampler address component layouts by opcode, dimension,
  and array declaration, including mip-level and cube-face placement at
  `cdna1/README.md:2143` through `:2170`.
- Chapter 9.4 and 9.4.1 give sampler address typing, cubemap `face_id`,
  coordinate/derivative component ordering, and VGPR data rules at
  `cdna1/README.md:2172` through `:2276`.
- Chapter 9.4.1 defines `DMASK` read component selection, full-element write
  behavior, image atomic 32/64-bit surface restrictions, legal atomic `DMASK`
  values, atomic-return data placement, and D16 load/store packing at
  `cdna1/README.md:2260` through `:2276`.

XML evidence:

- Representative no-sampler image loads, stores, and atomics use fixed 128-bit
  `VDATA` and `VADDR` operands, with `IMAGE_GET_RESINFO` using a fixed 32-bit
  `VADDR`; for example `IMAGE_LOAD`, `IMAGE_STORE`, and `IMAGE_ATOMIC_SWAP`
  appear at `amdgpu_isa_cdna1.xml:20101` through `:20127`,
  `:20353` through `:20379`, and `:20563` through `:20589`.
- `IMAGE_SAMPLE` uses fixed 128-bit `VDATA` and 96-bit F32 `VADDR` operands at
  `amdgpu_isa_cdna1.xml:21109` through `:21140`.
- The `ENC_MIMG` `DMASK` field description says only that at least one bit must
  be set and data is packed into consecutive VGPRs at
  `amdgpu_isa_cdna1.xml:3561` through `:3562`.

Impact:

XML-derived tools can decode the operand fields but cannot determine the actual
number, type, or placement of address and data VGPRs for a given image opcode,
dimension, `DA`, `A16`, `D16`, or atomic `DMASK` mode.

### CDNA1-XML-023: Image resource and sampler descriptor bitfields are opaque

Manual evidence:

- Chapter 9.2 says image operations send a 256-bit image resource constant that
  defines address, data format, and surface characteristics, and sampler
  operations also send a 128-bit sampler constant at `cdna1/README.md:2102`
  through `:2104`.
- Chapter 9.4.2 defines image resource fields for base address, dimensions,
  data/numeric format, destination selectors, mip levels, tiling, type, depth,
  pitch, array pitch, metadata, compression, and partially resident texture
  fields at `cdna1/README.md:2278` through `:2329`.
- Chapter 9.4.3 defines sampler descriptor fields for clamp/wrap, anisotropy,
  depth compare, force unnormalized, filtering, LOD limits/biases, border
  color, and border color type at `cdna1/README.md:2331` through `:2372`.

XML evidence:

- `FMT_IMG` is a single opaque 256-bit `Descriptor` field at
  `amdgpu_isa_cdna1.xml:92570` through `:92585`.
- `FMT_SAMP` is a single opaque 128-bit `Descriptor` field at
  `amdgpu_isa_cdna1.xml:93402` through `:93418`.
- Representative image and sample instruction entries attach `SRSRC` and
  `SSAMP` operands with those descriptor formats but do not expose descriptor
  subfields; `IMAGE_LOAD` and `IMAGE_SAMPLE` show the operand forms at
  `amdgpu_isa_cdna1.xml:20121` through `:20126` and `:21129` through `:21140`.

Impact:

XML consumers can identify resource and sampler operands, but cannot derive
image dimensions, format conversion, destination selection, filtering, clamp,
mip/LOD, metadata, compression, or PRT behavior from descriptor metadata.

### CDNA1-XML-024: Image data-format conversion tables are not structured

Manual evidence:

- Chapter 9.2 says the image resource constant defines the data format and
  surface characteristics used by MIMG operations at `cdna1/README.md:2102`
  through `:2104`.
- Chapter 9.4.1 says loads expand memory data to canonical RGBA using the
  resource format before `DMASK` selection, while writes fill missing stored
  components with zero and ignore values outside the stored data format at
  `cdna1/README.md:2263` through `:2264`.
- Chapter 9.4.4 enumerates buffer/image `DATA_FORMAT` and `NUM_FORMAT` values
  at `cdna1/README.md:2374` through `:2416`.

XML evidence:

- XML instruction descriptions state that image loads and stores perform format
  conversion specified by the resource descriptor, for example `IMAGE_LOAD` and
  `IMAGE_STORE` at `amdgpu_isa_cdna1.xml:20101` through `:20102` and
  `:20353` through `:20354`.
- The resource descriptor itself is opaque as recorded in `CDNA1-XML-023`, and
  representative image data operands use `FMT_ANY` rather than a structured
  image-format table, for example at `amdgpu_isa_cdna1.xml:20109` through
  `:20125`.

Impact:

The XML records that conversion exists but does not encode the data-format and
numeric-format enumeration or the RGBA fill/selection rules needed for an image
load/store emulator.

### CDNA1-XML-025: Image VMEM dependency timing is prose-only

Manual evidence:

- Chapter 9.4.5 says an issued VMEM image instruction immediately reads VGPR
  addresses plus texture resources and samplers, while write data is not sent
  to the texture cache immediately at `cdna1/README.md:2418` through `:2424`.
- The same paragraph points developers to `VMCNT` waits before consuming data
  returned from the texture cache at `cdna1/README.md:2420` through `:2424`.

XML evidence:

- Image instruction entries mark the functional group as `VMEM` and subgroups
  such as `TEXTURE`, `ATOMIC`, and `SAMPLE`; representative `IMAGE_LOAD`,
  `IMAGE_ATOMIC_SWAP`, and `IMAGE_SAMPLE` rows carry that metadata at
  `amdgpu_isa_cdna1.xml:20130` through `:20133`, `:20592` through `:20595`,
  and `:21144` through `:21147`.
- The XML entries do not encode the immediate address/resource/sampler read
  timing, delayed write-data timing, or `VMCNT` producer/decrement behavior for
  image operations.

Impact:

XML consumers can classify image instructions as VMEM, but need manual prose to
model ordering and wait-counter behavior.

### CDNA1-XML-026: FLAT/GLOBAL/SCRATCH addressing and aperture rules are prose-only

Manual evidence:

- Chapter 10 says FLAT instructions address a single flat memory space covering
  video, system, LDS, and scratch memory, with aperture registers controlling
  the mapping and unmapped regions producing memory violations at
  `cdna1/README.md:2428`.
- Section 10.1 defines segment selection, segment-specific offset signedness,
  and `SADDR` meaning for Flat, Scratch, and Global at `cdna1/README.md:2434`
  through `:2452`.
- Section 10.3 describes 32-bit versus 64-bit addressing and the flat scratch
  address conversion at `cdna1/README.md:2523` through `:2535`; sections 10.4
  and 10.5 define Global and Scratch address forms at `:2547` through `:2579`.
- Sections 10.6 and 10.8 define memory-violation policy and the automatic
  `FLAT_SCRATCH` state sent with every FLAT request at `cdna1/README.md:2581`
  through `:2608`.

XML evidence:

- `ENC_FLAT`, `ENC_FLAT_GLBL`, and `ENC_FLAT_SCRATCH` expose field positions,
  segment constants, offset widths, and operand fields at
  `amdgpu_isa_cdna1.xml:3211` through `:3408`, `:3411` through `:3589`, and
  `:3594` through `:3744`.
- Representative flat/global/scratch rows expose the VGPR/SGPR operands and
  implicit `FLAT_SCRATCH`/memory operands, for example `FLAT_LOAD_*`,
  `GLOBAL_ATOMIC_*`, and `SCRATCH_LOAD_*` forms at
  `amdgpu_isa_cdna1.xml:14070` through `:14090`, `:17041` through `:17076`,
  and `:18889` through `:18925`.
- `OPR_FLAT_SCRATCH` is only a predefined 64-bit operand named
  `flat_scratch[63:0]` at `amdgpu_isa_cdna1.xml:101518` through `:101524`;
  the XML does not encode aperture base/limit routing, `PTR32` or
  `ADDRESS_MODE`, scratch swizzling, scratch-size checks, or memory-violation
  side effects.

Impact:

The XML is sufficient for raw field decode, but an emulator or validator needs
manual-derived rules for address-space routing, scratch/global address
calculation, private/shared aperture behavior, and fault policy.

### CDNA1-XML-027: Direct-LDS, M0, and FLAT wait-counter rules are prose-only

Manual evidence:

- Section 10.1 says `LDS` moves data between LDS and memory instead of VGPRs
  and memory for Global and Scratch only, while Flat must keep the bit zero; it
  also says `M0` is implied for Scratch and Global only when `LDS=1` at
  `cdna1/README.md:2443` and `:2452`.
- Section 10.2 says FLAT instructions cannot return data directly to LDS, and
  that internally they execute as both LDS and Buffer requests, incrementing
  both `VM_CNT` and `LGKM_CNT` until both complete at `cdna1/README.md:2503`
  through `:2507`.
- Sections 10.2.1 and 10.2.2 describe FLAT out-of-order completion and the
  resulting `S_WAITCNT 0` requirement at `cdna1/README.md:2511` through
  `:2519`.
- Sections 10.4 and 10.5 say Global and Scratch direct LDS movement is allowed
  and uses only `VM_CNT`, not `LGKM_CNT`, at `cdna1/README.md:2560` through
  `:2562` and `:2577` through `:2579`.

XML evidence:

- `ENC_FLAT`, `ENC_FLAT_GLBL`, and `ENC_FLAT_SCRATCH` use the same generic
  `LDS` field description, "Read/Write Data from/to LDS", at
  `amdgpu_isa_cdna1.xml:3328` through `:3329`, `:3511` through `:3512`, and
  `:3666` through `:3667`.
- Representative FLAT instruction rows list implicit `OPR_SDST_M0` operands,
  for example at `amdgpu_isa_cdna1.xml:14087` through `:14090` and
  `:14334` through `:14336`, without segment- or `LDS`-qualified metadata.
- The XML functional-group metadata classifies these rows as `VMEM`, but does
  not encode dual FLAT `VM_CNT`/`LGKM_CNT` increments, direct-LDS addressing
  formulas, or the ordering/timing hazards described in Chapter 10.

Impact:

XML consumers can see the `LDS` bit and M0 operand, but cannot derive when they
are legal, how LDS addresses are formed, which wait counters are incremented,
or why FLAT consumers require a full wait.

### CDNA1-XML-028: FLAT/GLOBAL atomic descriptions miss floating-point atomic restrictions

Manual evidence:

- Section 10.1 says `GLC=1` returns the pre-op value for atomics, but Section
  10.3.1 further says floating-point atomics must set `GLC=0`, host
  non-cacheable memory atomics can be silently dropped, FP32 atomics flush
  denormals to zero, FP16 atomics never flush denormals, and rounding is fixed
  round-to-nearest-even at `cdna1/README.md:2441` and `:2539` through `:2543`.
- The opcode tables make `GLOBAL_ATOMIC_ADD_F32` and
  `GLOBAL_ATOMIC_PK_ADD_F16` Global-only opcodes at `cdna1/README.md:2497`
  through `:2498` and `:7609` through `:7610`.
- The detailed instruction rows describe the F32 and packed-F16 memory updates
  at `cdna1/README.md:5283` through `:5284`.

XML evidence:

- Integer flat atomics use descriptions such as "Store the original value ...
  iff the GLC bit is set"; representative `FLAT_ATOMIC_ADD` metadata appears
  at `amdgpu_isa_cdna1.xml:14293` through `:14336`.
- `GLOBAL_ATOMIC_ADD_F32` and `GLOBAL_ATOMIC_PK_ADD_F16` are present at
  `amdgpu_isa_cdna1.xml:17899` through `:18010`, but the F32 row still says
  the original value is returned iff `GLC` is set and neither row encodes the
  `GLC=0` legality restriction, denormal policy, fixed rounding, or
  non-cacheable-host-memory drop rule.

Impact:

XML-only consumers can discover the Global FP atomic opcodes and broad operand
types, but they need manual-derived policy for FP atomic legality and numeric
edge behavior.

### CDNA1-XML-029: LDS geometry, direct-read, and M0 clamp rules are prose-only

Manual evidence:

- Chapter 11.1 describes LDS as 64 KiB per compute unit, split into 32 banks of
  512 dwords, and says bank conflicts serialize indexed and atomic operations at
  `cdna1/README.md:2624` through `:2633`.
- Section 11.3.1 says LDS direct reads are VALU SRC0 broadcasts from an address
  and data type encoded in `M0[18:0]` at `cdna1/README.md:2655` through
  `:2672`.
- Section 11.3.2 says parameter interpolation reads use an automatic M0 layout
  containing `new_prim_mask` and `lds_param_offset`, and gives the
  same-register restriction for `VSRC`/`VDST`, at `cdna1/README.md:2696`
  through `:2707`.
- Section 11.3.3 requires M0 initialization before indexed and atomic LDS
  operations and says M0 can restrict the accessible LDS range at
  `cdna1/README.md:2736`.

XML evidence:

- `ENC_DS` exposes raw DS fields such as `ADDR`, `DATA0`, `DATA1`, `GDS`,
  `OFFSET0`, `OFFSET1`, and `VDST` at `amdgpu_isa_cdna1.xml:2178` through
  `:2406`.
- Representative DS rows expose instruction operands, but the XML field and
  instruction metadata do not encode LDS bank topology, bank-conflict
  serialization, M0 clamp/size semantics, LDS-direct M0 address and data-type
  layout, or interpolation M0 layout and `VSRC != VDST` restrictions.

Impact:

XML consumers can decode DS bit fields and operands, but must consult the manual
for LDS timing/topology, clamping, and non-DS LDS read semantics.

### CDNA1-XML-030: DS address formulas and duplicate-offset behavior are prose-only

Manual evidence:

- Section 11.3.3 defines single-address DS operations as
  `LDS_BASE + VGPR[ADDR] + {InstrOffset1,InstrOffset0}` and double-address
  operations as two separately scaled 8-bit offsets at
  `cdna1/README.md:2753` through `:2769`.
- The manual says equal READ2/WRITE2 offsets cause only one read or write and
  only the first data source is used at `cdna1/README.md:2775`.
- `DS_WRITE_ADDTID_B32` and `DS_READ_ADDTID_B32` use
  `ADDR_BASE + OFFSET + M0.OFFSET + TID*4` at `cdna1/README.md:4516` and
  `:4668`.

XML evidence:

- The DS field descriptions only say `OFFSET0` and `OFFSET1` are DS offsets and
  to see individual instructions for use at `amdgpu_isa_cdna1.xml:2368`
  through `:2379`.
- Representative `DS_WRITE2_B32`, `DS_READ2_B32`, and
  `DS_READ_ADDTID_B32` entries are present at `amdgpu_isa_cdna1.xml:7578`,
  `:9375`, and `:12583`, but do not carry the formula, scaling, equal-offset
  collapse, or `M0.OFFSET + TID*4` contract as machine-readable metadata.

Impact:

An XML-only implementation can generate the fields and operands, but cannot
derive the address math or the equal-offset single-access rule.

### CDNA1-XML-031: CDNA1 DS SRC2 opcode family is missing from XML

Manual evidence:

- Section 11.3.3 says `DS_*_SRC2_*` operations use two LDS operands and treat
  the offset as a 16-bit signed dword offset, with an alternate per-thread mode
  selected by `offset1[7]`, at `cdna1/README.md:2777` through `:2789`.
- The detailed instruction table lists `DS_ADD_SRC2_U32` through
  `DS_ADD_SRC2_F32` at `cdna1/README.md:4634` through `:4655` and 64-bit SRC2
  forms through `DS_MAX_SRC2_F64` at `:4672` through `:4692`.

XML evidence:

- Searches of `amdgpu_isa_cdna1.xml` for `DS_*_SRC2_*` names returned no
  instruction rows.
- The `ENC_DS` identifier list and representative DS opcode rows are otherwise
  populated at `amdgpu_isa_cdna1.xml:2178` through `:2406` and
  `:6892` onward, so this is a targeted inventory gap rather than an absent DS
  encoding description.

Impact:

Code generated from the XML will treat CDNA1 SRC2 DS encodings as invalid or
unimplemented even though the manual lists them as DS opcodes with distinct
addressing semantics.

### CDNA1-XML-032: LDS floating atomic numeric rules are prose-only

Manual evidence:

- Section 11.3.3 says LDS floating atomics use `MODE.FP_DENORM` for denormal
  behavior and fixed round-to-nearest-even rounding at `cdna1/README.md:2805`
  through `:2807`.
- The detailed rows for `DS_CMPST_F32`, `DS_MIN_F32`, `DS_MAX_F32`,
  `DS_ADD_F32`, and their return variants describe NaN/Inf/denormal handling
  and the compare/source ordering at `cdna1/README.md:4511` through `:4515` and
  `:4543` through `:4550`.
- SRC2 floating rows add separate memory-source formulas and FP handling at
  `cdna1/README.md:4653` through `:4655` and `:4691` through `:4692`.

XML evidence:

- Floating DS rows such as `DS_CMPST_F32` and `DS_ADD_F32` are present at
  `amdgpu_isa_cdna1.xml:7731` and `:7909`, but their metadata does not encode
  `MODE.FP_DENORM`, fixed rounding, NaN/Inf/denormal policy, or the SRC2
  floating opcode rows.

Impact:

XML consumers can classify a DS instruction as floating-point, but need
manual-derived semantics for exact numeric behavior and for the SRC2 FP forms.

### CDNA1-XML-033: DS lane-routing and GWS execution details are prose-only

Manual evidence:

- Section 11.3.3 summarizes `DS_PERMUTE_B32` and `DS_BPERMUTE_B32` as
  wavefront lane-routing operations that do not access LDS memory at
  `cdna1/README.md:2750` through `:2751`.
- The detailed instruction rows give the permute/bpermute lane formulas, EXEC
  mask behavior, high-bit address truncation, and examples at
  `cdna1/README.md:4562` through `:4566`.
- Chapter 11.4 says every GWS instruction must be followed immediately by
  `s_waitcnt 0` at `cdna1/README.md:2809` through `:2813`; the detailed GWS
  rows define resource-id calculation and queue/counter/barrier state-machine
  behavior at `cdna1/README.md:4656` through `:4667`.

XML evidence:

- XML rows for `DS_SWIZZLE_B32`, `DS_PERMUTE_B32`, and `DS_BPERMUTE_B32` are
  present at `amdgpu_isa_cdna1.xml:9633` through `:9738`, but only include
  compact descriptions and operands.
- XML rows for `DS_GWS_SEMA_RELEASE_ALL`, `DS_GWS_INIT`, `DS_GWS_SEMA_V`,
  `DS_GWS_SEMA_BR`, `DS_GWS_SEMA_P`, and `DS_GWS_BARRIER` are present at
  `amdgpu_isa_cdna1.xml:12412` through `:12568`, but do not encode the required
  following `s_waitcnt 0` or the GWS state-machine details.

Impact:

XML-only consumers can decode lane-routing and GWS opcodes, but cannot implement
their full wave-lane, EXEC, sequencing, or resource-state contracts without
manual prose.

### CDNA1-XML-034: SMEM offset-mode and alignment rules are under-described

Manual evidence:

- Chapter 8 gives the SMEM `IMM`, `OFFSET`, and `SOE` address matrix, including
  M0 selection, store/atomic restrictions, scratch 64-byte offset scaling,
  low-bit address masking, and negative immediate legality at
  `cdna1/README.md:1635` through `:1641` and `:1651` through `:1672`.
- Chapter 12 and Chapter 13 describe the immediate form as a 21-bit signed byte
  offset, and the register form as an unsigned byte offset, at
  `cdna1/README.md:3195` through `:3211` and `:5830` through `:5843`.
- The same manual has an internal Chapter 8 wording conflict: the 8.1 field
  table says `IMM=1` is a 20-bit unsigned byte offset at
  `cdna1/README.md:1635`, while the detailed instruction and microcode-format
  chapters use 21-bit signed immediate offsets.

XML evidence:

- The XML `ENC_SMEM` fields describe `IMM`, `OFFSET`, `SOFFSET`, and
  `SOFFSET_EN` at `amdgpu_isa_cdna1.xml:702` through `:723` and
  `:766` through `:777`.
- The XML follows the 21-bit signed immediate width at
  `amdgpu_isa_cdna1.xml:722` through `:727`, but does not encode the full
  IMM/SOE address matrix, M0 selector behavior, store/atomic SGPR-offset ban,
  scratch 64-byte scaling, low-bit address masking, discard 64-byte alignment,
  or the negative-offset undefined case.

Impact:

XML-only consumers can recover the SMEM bitfields and many operand shapes, but
need manual-derived rules to compute CDNA1 SMEM addresses, validate legal
offset selectors, and handle the immediate-width wording conflict.

### CDNA1-XML-035: Scalar-buffer descriptor and bounds semantics are prose-only

Manual evidence:

- Chapter 8 says buffer SMEM uses a 4-SGPR resource constant, lists the only
  consumed fields as base, stride, `num_records`, and `NV`, forbids swizzled
  scalar buffers, and says stride is only used for bounds checking at
  `cdna1/README.md:1638` and `:1686` through `:1707`.
- The alignment section says buffer `SBASE` must identify an aligned SGPR quad,
  out-of-range `SBASE` uses SGPR0, and clamped memory-address dwords are not
  performed at `cdna1/README.md:1743` through `:1751`.

XML evidence:

- Representative buffer rows such as `S_BUFFER_LOAD_DWORD` mark `SBASE` as
  `FMT_RSRC_SCALAR` and 128 bits at `amdgpu_isa_cdna1.xml:30156` through
  `:30187`.
- The generic XML `SBASE` field description instead describes a pair carrying
  `{size[16], base[48]}` at `amdgpu_isa_cdna1.xml:742` through `:753`, and
  the XML does not encode the descriptor field layout, the no-swizzle rule,
  stride-as-bounds-only behavior, SGPR0 fallback, or per-dword bounds policy.

Impact:

XML operand metadata distinguishes scalar-buffer operands from ordinary base
pairs, but a consumer cannot implement CDNA1 scalar-buffer addressing or bounds
behavior from XML alone.

### CDNA1-XML-036: SMEM source-overwrite, clause, atomic-return, and LGKM rules are prose-only

Manual evidence:

- Chapter 8 forbids scalar memory instructions from overwriting their own
  sources, extends that rule across scalar memory clauses, and requires scalar
  atomics to be naturally aligned and single-instruction clauses at
  `cdna1/README.md:1682` through `:1684`.
- Scalar atomics use the same addressing as loads/stores and return the
  pre-operation value only when `GLC=1` at `cdna1/README.md:1711`.
- The dependency section says LGKM_CNT increments by 1 for a single-dword fetch
  and by 2 for two or more dwords, with out-of-order and partial returns, at
  `cdna1/README.md:1727` through `:1735`.

XML evidence:

- XML instruction rows cover loads, stores, and atomics under `ENC_SMEM`, for
  example `S_LOAD_DWORD` at `amdgpu_isa_cdna1.xml:29764` through `:29804` and
  scalar-buffer atomic rows beginning at `amdgpu_isa_cdna1.xml:31170`.
- The generic `GLC` field says only that the operation is globally coherent at
  `amdgpu_isa_cdna1.xml:691` through `:699`.
- Static XML search did not find metadata for scalar memory clause formation,
  source-overwrite replay hazards, single-instruction atomic clauses, atomic
  return/no-return operand effects, or the 1-versus-2 LGKM counter accounting.

Impact:

XML-only decode can identify SMEM opcodes and operands, but cannot derive the
runtime dependency, replay, or atomic-return contracts that software schedulers
and emulators need.

### CDNA1-XML-037: SMEM cache, discard, probe, and time-counter details are under-described

Manual evidence:

- Chapter 8 says `S_DCACHE_INV` and `S_DCACHE_WB` affect the entire scalar data
  cache and return no `SDST` at `cdna1/README.md:1715` through `:1717`.
- `S_MEMTIME` returns a 64-bit clock counter, while `S_MEMREALTIME` returns a
  64-bit real-time counter from a constant 25MHz clock, at
  `cdna1/README.md:1719` through `:1725`.
- The detailed `S_DCACHE_DISCARD` rows say discard operates on 64-byte aligned
  cache lines and increments LGKM by 1 or 2 at `cdna1/README.md:3230` through
  `:3231`.

XML evidence:

- XML rows exist for cache invalidation/writeback, time, ATC probe, and discard
  instructions at `amdgpu_isa_cdna1.xml:30842` through `:31140`.
- Those rows give opcode names, operand shapes, and short descriptions, but do
  not encode the 25MHz real-time clock source, discard address alignment and
  LGKM increments, volatile-line distinctions beyond the mnemonic text, or ATC
  probe/prefetch behavioral payloads.

Impact:

XML consumers can disassemble the auxiliary SMEM opcodes, but need manual prose
for precise cache, counter, and probe behavior.

### CDNA1-XML-038: Program-flow scheduling and wait-counter semantics are prose-only

Manual evidence:

- Chapter 4 says `S_ENDPGM`, `S_ENDPGM_SAVED`, `S_NOP`, `S_TRAP`, and
  `S_RFE` are program-control instructions at `cdna1/README.md:779` through
  `:783`.
- Section 4.3 defines same-workgroup `S_BARRIER` behavior, including the rule
  that early-terminated waves are no longer part of the barrier at
  `cdna1/README.md:818` through `:820`.
- Section 4.4 defines the VM/LGKM/EXP counters, says `S_WAITCNT` waits until
  counters are at or below the encoded targets, gives scalar-memory dword-count
  and `S_SENDMSG` LGKM increments, and describes in-order/out-of-order return
  rules at `cdna1/README.md:822` through `:849`.
- Section 4.5 lists required manually inserted wait states, including
  `S_SETREG`, VSKIP, VCC/EXEC, VMEM-SGPR, M0, DPP, TRAPSTS/RFE, and
  `S_MOVEREL` hazards at `cdna1/README.md:850` through `:877`.

XML evidence:

- XML rows exist for `S_NOP`, `S_ENDPGM`, ordinary branches, `S_BARRIER`,
  `S_WAITCNT`, `S_SENDMSG`, `S_SENDMSGHALT`, `S_TRAP`, and `S_ICACHE_INV` at
  `amdgpu_isa_cdna1.xml:47270` through `:47913`.
- The XML also exposes wait-count and message operands through `OPR_WAITCNT`
  and `OPR_SENDMSG`, but static inspection found no machine-readable
  representation of the barrier-live-wave rule, per-family wait-counter
  accounting, scalar-memory dword-count increments, `S_SENDMSG` LGKM behavior,
  scalar-memory out-of-order restrictions, or the required wait-state hazard
  table.

Impact:

XML-only consumers can identify the control-flow opcodes and immediate fields,
but cannot derive the Chapter 4 scheduling and synchronization contract without
manual-derived metadata.

### CDNA1-XML-039: Fork/join branch-stack semantics are not machine-readable

Manual evidence:

- Section 4.6 says `S_CBRANCH_I/G_FORK` and `S_CBRANCH_JOIN` use a six-deep
  stack, require three SGPRs per fork/join block, store `{exec[63:0],
  PC[47:2]}` entries, choose the smaller pass/fail path first, and update
  `EXEC`, PC, and CSP according to the listed pseudocode at
  `cdna1/README.md:878` through `:933`.

XML evidence:

- XML rows exist for `S_CBRANCH_JOIN`, `S_CBRANCH_G_FORK`, and
  `S_CBRANCH_I_FORK` at `amdgpu_isa_cdna1.xml:36363` through `:36405`,
  `:42644` through `:42750`, and `:47070` through `:47090`, but the entries
  only provide compact descriptions and operand lists.
- `S_CBRANCH_G_FORK` uses `OPR_SSRC_NOLIT` operands in the base row, while the
  XML also emits `SOP2_INST_LITERAL` variants for literal operands at
  `amdgpu_isa_cdna1.xml:42671` through `:42742`, leaving source-form legality
  ambiguous.

Impact:

An XML consumer can decode the fork/join mnemonics, but cannot implement or
validate the branch-stack algorithm, CSP storage contract, path-selection rule,
or literal legality without manual-derived overrides.

### CDNA1-XML-040: Auxiliary control, message, and debug branch side effects are under-described

Manual evidence:

- Chapter 4 lists debug conditional branches, direct PC operations,
  `S_SETVSKIP`, `S_SLEEP`, `S_SETPRIO`, and `S_SENDMSG` in the program-flow
  inventory at `cdna1/README.md:787` through `:816`.
- The detailed SOPP rows define `S_WAKEUP`, `S_SETHALT`, `S_SLEEP`,
  `S_SETPRIO`, `S_SENDMSG`, `S_SENDMSGHALT`, `S_TRAP`, `S_ICACHE_INV`,
  perf-level instructions, `S_TTRACEDATA`, debug branches,
  `S_ENDPGM_SAVED`, `S_SET_GPR_IDX_OFF`, `S_SET_GPR_IDX_MODE`, and
  `S_ENDPGM_ORDERED_PS_DONE` at `cdna1/README.md:3123` through `:3169`.

XML evidence:

- XML rows exist for the same SOPP surface, including debug branches and
  control/status/message/cache/perf/thread-trace rows, at
  `amdgpu_isa_cdna1.xml:47359` through `:48237`; message encodings are listed
  separately under `OPR_SENDMSG` at `:102452` through `:102502`.
- The XML rows do not encode sleep-cycle semantics, priority/status
  transitions, halt/kill behavior, message side effects or LGKM completion,
  thread-trace payload behavior, cache-invalidation spacing requirements, or
  debug-status predicate sources.

Impact:

XML-only decode preserves mnemonics and operands for the auxiliary program-flow
surface, but runtime or validation consumers need manual-derived semantics for
the side effects.

### CDNA1-XML-041: VALU scalar-source budget restrictions are prose-only

Manual evidence:

- Chapter 6.2.1 lists VALU inputs as VGPRs, SGPRs, inline constants, literal
  constants, LDS direct, M0, and EXEC, then restricts each instruction to at
  most one SGPR source and at most one literal, with literals disallowed when an
  SGPR or M0 is used, at `cdna1/README.md:1252` through `:1269`.
- The same section says only SRC0 can use LDS_DIRECT and that `ADDC`, `SUBB`,
  and `CNDMASK` implicitly use VCC, so those instructions cannot use an
  additional SGPR or literal source at `cdna1/README.md:1267` through `:1275`.
- The detailed definitions show the same implicit/explicit carry-mask
  interaction: `V_CNDMASK_B32` selects through VCC at `cdna1/README.md:3318`,
  and VOP3 `V_ADDC_CO_U32` / `V_SUBB_CO_U32` take their carry-in from the
  SGPR-pair at `S2.u` at `:3359` through `:3360`.

XML evidence:

- XML records operand-local classes for representative VOP3 `V_ADD_F32`: `SRC0`
  is `OPR_SRC_NOLIT` and `SRC1` is `OPR_SRC_SIMPLE` at
  `amdgpu_isa_cdna1.xml:54365` through `:54387`.
- Those operand classes enumerate scalar-register and special scalar selector
  values locally: `OPR_SRC_NOLIT` starts at `amdgpu_isa_cdna1.xml:104459`, and
  `OPR_SRC_SIMPLE` starts at `:106881`.
- VOP3 `V_ADDC_CO_U32` records broad `SRC0`/`SRC1` classes plus an explicit
  `OPR_SREG` `SRC2` carry-in at `amdgpu_isa_cdna1.xml:58143` through `:58177`,
  but the row does not encode that this scalar source consumes the instruction's
  scalar-source budget.

Impact:

XML-only validators can infer operand-local selector ranges, but cannot tell
which combinations of VALU sources exceed CDNA1's one-scalar-source and
literal/M0 restrictions.

### CDNA1-XML-042: VOP3 output-modifier MODE and denormal rules are prose-only

Manual evidence:

- Chapter 6.2.2 says VOP3 floating-point results can apply OMOD and CLAMP, but
  output modifiers apply only to floating-point results, are ignored for integer
  or bit results, are ignored when output denormals are enabled, flush denormals
  and `-0` when output denormals are disabled, and are ignored when the IEEE
  mode bit is set at `cdna1/README.md:1293` through `:1299`.

XML evidence:

- Generic `ENC_VOP3` exposes `CLAMP` and `OMOD` fields at
  `amdgpu_isa_cdna1.xml:2657` through `:2688`.
- The audited CDNA1 XML exposes the raw fields, but not the `FP_DENORM` or
  IEEE-mode conditions, denormal flush behavior, or `-0` to `+0` behavior.
  Targeted XML search found no `FP_ROUND`, `FP_DENORM`, or `IEEE_MODE` records.

Impact:

XML-derived execution code can see that VOP3 carries output-modifier fields, but
cannot derive when CDNA1 requires those fields to be ignored or when additional
floating-point result flushing is required.

### CDNA1-XML-043: ALU clamp-bit semantic overloads are prose-only

Manual evidence:

- Chapter 6.5 says the VALU clamp bit is overloaded by opcode family: `V_CMP`
  clamp requests signaling behavior for floating-point exceptions, integer
  operations clamp to the largest or smallest representable value, and
  floating-point operations clamp to `[0.0, 1.0]` at `cdna1/README.md:1442`
  through `:1444`.
- Chapter 12.9.1 repeats the compare-specific rule: when `CLAMP` is set, compare
  instructions signal an exception if either input is NaN; when clear, NaN does
  not signal an exception, at `cdna1/README.md:4092`.

XML evidence:

- Generic `ENC_VOP3` describes `CLAMP` only as clamping output to `[0.0, 1.0]`
  at `amdgpu_isa_cdna1.xml:2657` through `:2658`.
- Representative compare rows such as `V_CMP_F_F32` and `V_CMP_LG_F32` expose
  compare operands and destinations at `amdgpu_isa_cdna1.xml:74455` through
  `:74481` and `:75025` through `:75051`, but do not encode the
  signaling-compare meaning of `CLAMP`.
- Representative integer VOP3 rows such as `V_ADDC_CO_U32` expose operands and
  the VOP3/VOP3B encoding at `amdgpu_isa_cdna1.xml:58143` through `:58177`, but
  do not distinguish clamp-as-saturation semantics.

Impact:

The XML can identify the raw clamp bit, but not the opcode-family-dependent
contract needed to implement compare exception signaling or integer saturation.

### CDNA1-XML-044: VGPR indexing bit layout and operand-role mapping are prose-only

Manual evidence:

- Chapter 6.6 defines VGPR indexing as MODE-gated M0 indexing for selected VALU
  VGPR sources or destinations at `cdna1/README.md:1448` through `:1452`.
- Table 27 says `S_SET_GPR_IDX_ON` and `S_SET_GPR_IDX_MODE` store the mode in
  `M0[15:12]`, while `S_SET_GPR_IDX_IDX` stores the index in `M0[7:0]`, at
  `cdna1/README.md:1458` through `:1461`.
- The prose assigns `M0[15]` to destination, `M0[14]` to source 2, `M0[13]` to
  source 1, and `M0[12]` to source 0, limits indexing to VGPR operands, makes
  indexed out-of-range VGPR access illegal, and gives special role remaps for
  readlane, writelane, MAC/MAD, reverse shifts, `v_cvt_pkaccum`, and SDWA at
  `cdna1/README.md:1463` through `:1490`.

XML evidence:

- XML entries exist for `S_SET_GPR_IDX_IDX`, `S_SET_GPR_IDX_ON`,
  `S_SET_GPR_IDX_OFF`, and `S_SET_GPR_IDX_MODE` at
  `amdgpu_isa_cdna1.xml:34830`, `:44206`, `:46192`, and `:46214`.
- Those rows expose opcode and operand metadata, including `OPR_SIMM4` for
  `S_SET_GPR_IDX_ON`, but do not encode `MODE.gpr_idx_en`, the `M0[7:0]` versus
  `M0[15:12]` layout, the source/destination enable-bit meanings, the VGPR-only
  restriction, out-of-range illegality, or the unusual instruction role table.

Impact:

XML-derived operand readers can know that the control instructions touch M0, but
cannot derive which VALU operand slots should be indexed or which M0 bits make a
given indexed VGPR access legal.

### CDNA1-XML-045: MAI AccVGPR state and Arch/Acc dataflow are prose-only

Manual evidence:

- The Chapter 7 MAI introduction says miSIMD has its own Accumulation GPR file,
  separate from architectural VGPRs, and that shader I/O can only use ArchVGPRs
  at `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1525` through `:1531`.
- The same section says A and B matrices may come from ArchVGPRs or AccVGPRs,
  C always comes from AccVGPRs, D always uses AccVGPRs, and data moves between
  ArchVGPRs and AccVGPRs via `V_ACCVGPR_READ`/`V_ACCVGPR_WRITE` at
  `cdna1/README.md:1538` through `:1543`.

XML evidence:

- The XML exposes operand-local AccVGPR classes on representative MFMA rows,
  including `OPR_ACCVGPR`, `OPR_SRC_VGPR_OR_ACCVGPR`, and
  `OPR_SRC_ACCVGPR_OR_CONST`, at `amdgpu_isa_cdna1.xml:68782` through `:68818`.
- The `VOP3P_MFMA` encoding has an `ACC` field description saying it selects
  whether srcA/srcB use an accumulator VGPR at `amdgpu_isa_cdna1.xml:7536`.
- The XML operand tables name AccVGPR selector spaces, for example
  `OPR_ACCVGPR` at `amdgpu_isa_cdna1.xml:93424` and
  `OPR_SRC_VGPR_OR_ACCVGPR` at `:110584`, but they do not encode the separate
  MAI register-file state, shader-I/O boundary, or `V_ACCVGPR_READ`/`WRITE` as
  the only Arch/Acc movement path.

Impact:

XML consumers can see AccVGPR-looking operands, but cannot derive the Chapter 7
state model or dataflow contract without manual-derived architecture metadata.

### CDNA1-XML-046: `V_ACCVGPR_WRFED` appears in the manual opcode table but not XML

Manual evidence:

- The detailed Chapter 12.10 VOP3P instruction table lists
  `V_ACCVGPR_READ` and `V_ACCVGPR_WRITE` at opcodes 88 and 89, but does not
  include `V_ACCVGPR_WRFED`, at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:4178` through `:4180`.
- The Chapter 13.3 VOP3P opcode table lists opcode 92 as
  `V_ACCVGPR_WRFED` at `cdna1/README.md:6821`.

XML evidence:

- The checked-in CDNA1 XML includes `V_ACCVGPR_READ` and
  `V_ACCVGPR_WRITE` rows at `amdgpu_isa_cdna1.xml:69487` and `:69525`.
- Searching the checked-in CDNA1 XML finds no `V_ACCVGPR_WRFED`,
  `ACCVGPR_WRFED`, or `WRFED` instruction row.

Impact:

The manual is internally under-described, but XML-only opcode inventories miss
the opcode-92 table entry entirely if `V_ACCVGPR_WRFED` is architecturally
valid on CDNA1.

### CDNA1-XML-047: SOP2 literal variants do not encode the single-extension-word constraint

Manual/oracle evidence:

- Chapter 12.1 says SOP2 instructions may use a 32-bit literal constant
  immediately after the instruction at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2852` through `:2856`.
- Chapter 13.1.1 similarly says the SOP2 format can be followed by a 32-bit
  literal constant at `cdna1/README.md:5434` through `:5437`, and its source
  selector table gives selector 255 as the literal constant at
  `cdna1/README.md:5474`.
- As a checked assembler oracle for `gfx908`, LLVM accepts
  `s_add_u32 s0, 0x12345678, 0x12345678` and emits both source selectors as
  255 plus one extension word, but rejects
  `s_add_u32 s0, 0x12345678, 0x9abcdef0` with "only one unique literal operand
  is allowed".

XML evidence:

- The XML represents representative SOP2 rows such as `S_ADD_U32` with
  `SOP2_INST_LITERAL` alternatives for `has_lit_0`, `has_lit_1`, and
  `has_lit_0_has_lit_1` at `amdgpu_isa_cdna1.xml:35282` through `:35415`.
- In the `has_lit_0_has_lit_1` alternative both explicit source operands are
  represented as separate `SIMM32` operands at `amdgpu_isa_cdna1.xml:35385`
  through `:35407`, but the XML does not state that these two operand slots
  alias the same single extension word and therefore must have the same literal
  value.

Impact:

XML consumers can see that either or both source selectors may be 255, but not
the one-unique-literal constraint imposed by the single SOP2 extension dword.
An assembler, disassembler round-trip checker, or fuzzer generated directly
from XML can incorrectly treat the two `SIMM32` operands as independent
literals.

### CDNA1-XML-048: SOP1 edge formulas and examples are prose-only

Manual evidence:

- Chapter 12.3 gives executable formulas and edge examples for SOP1 whole-quad,
  bit count, bit scan, sign-run, bitset, saveexec/wrexec, quad-mask, absolute
  value, and bitreplicate rows at `cdna1/README.md:2988` through `:3077`.
- The sensitive cases include no-bit-found sentinel results for
  `S_FF*`/`S_FLBIT*`, sign-run results for all-zero/all-one inputs, the
  `S_ABS_I32(0x80000000)` negative fixed point, and the bitreplicate
  low-32-to-64 expansion examples at `cdna1/README.md:2998` through `:3077`.

XML evidence:

- The checked-in XML carries compact descriptions, operand sizes, literal
  variants, and implicit SCC/EXEC/PC/M0 operands for representative SOP1 rows
  such as `S_FF0_I32_B32`, `S_ABS_I32`, and `S_BITREPLICATE_B64_B32` at
  `amdgpu_isa_cdna1.xml:32727`, `:34766`, and `:35228`.
- Those rows do not structure the manual's pseudocode loops or edge examples as
  machine-readable result tables. For example, `S_ABS_I32` only says to compute
  the absolute value and set SCC at `amdgpu_isa_cdna1.xml:34766` through
  `:34817`, without the signed-minimum fixed-point example from the manual.

Impact:

XML-only generators can build the right opcode and operand shapes but still miss
edge-sensitive SOP1 tests or helper semantics unless they cross-reference the
manual prose.

### CDNA1-XML-049: `S_SET_GPR_IDX_ON` exposes literal variants for a raw 4-bit mode field

Manual/oracle evidence:

- Chapter 12.4 says `S_SET_GPR_IDX_ON` writes `M0[15:12] = SIMM4`, clarifies
  that this is the direct raw content of the S1 field, and maps the raw bits to
  `VSRC0_REL`, `VSRC1_REL`, `VSRC2_REL`, and `VDST_REL` at
  `cdna1/README.md:3113`.
- `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908` accepts
  `s_set_gpr_idx_on s4, 15` and `s_set_gpr_idx_on 0x12345678, 15`, but rejects
  `s_set_gpr_idx_on s4, 255` as an invalid 4-bit immediate. Raw bytes with
  `SSRC1=0xff` disassemble as `s_set_gpr_idx_on s0, 0xff` without consuming a
  following extension word.

XML evidence:

- XML records the default `S_SET_GPR_IDX_ON` second operand as field `SSRC1`
  with `OPR_SIMM4`, which matches the raw-mode intent at
  `amdgpu_isa_cdna1.xml:44220` through `:44224`.
- The same row also has generic `SOPC_INST_LITERAL` alternatives for
  `has_lit_1` and `has_lit_0_has_lit_1`, exposing `SSRC1` as `SIMM32` while
  retaining `OPR_SIMM4` at `amdgpu_isa_cdna1.xml:44268` through `:44323`.

Impact:

XML consumers need an instruction-specific override for `S_SET_GPR_IDX_ON`.
Generic SOPC literal expansion can assign an extension word to operand 1 even
though the manual and LLVM assembler define that operand as direct 4-bit
instruction-field content.

### CDNA1-XML-050: SOPP send-message enum and payload rules are not self-contained

Manual/oracle evidence:

- Chapter 12.5.1 says `S_SENDMSG` encodes a message type and can send data from
  the `SIMM16` field and sometimes `EXEC` at `cdna1/README.md:3171` through
  `:3175`.
- The CDNA1 send-message table lists message 0 as illegal, graphics messages 2
  and 3, save/stall/halt messages 4 through 6, ordered-PS done at 7, early
  primitive deallocation at 8 with an `EXEC` payload, and GS allocation request
  at 9 with an M0 payload at `cdna1/README.md:3177` through `:3185`.
- As a `gfx908` oracle, `llvm-mc` accepts the manual graphics form
  `sendmsg(MSG_GS, GS_OP_CUT, 1)` and also accepts `MSG_INTERRUPT` and
  `MSG_GET_DOORBELL`, which are not present in the CDNA1 markdown table.

XML evidence:

- XML gives `S_SENDMSG` and `S_SENDMSGHALT` an explicit `OPR_SENDMSG` immediate
  plus an implicit M0 input at `amdgpu_isa_cdna1.xml:45849` through `:45902`.
- `OPR_SENDMSG` structures `GSOP`, `MSG`, and `STREAMID` fields and enumerates
  messages 1 through 10 and 15 at `amdgpu_isa_cdna1.xml:95762` through
  `:95877`; this includes `msg_interrupt`, `msg_get_doorbell`, and
  `msg_sysmsg`, while the CDNA1 manual table lists only message 0 and 2 through
  9.
- The XML predefined-value rows have empty descriptions and do not encode the
  manual payload rules for GS, early-primitive deallocation, GS allocation,
  message-0 illegality, or which messages consume `EXEC` or M0 payload fields.

Impact:

XML-only consumers can decode a partitioned send-message immediate, but cannot
derive CDNA1 message legality or payload validation from the XML alone. The
manual table and LLVM oracle also expose a drift in message availability that
needs manual/oracle cross-checking rather than blind reuse of the enum.

### CDNA1-XML-051: VOP2 bitwise VOP3 promotions do not encode the no-modifier contract

Manual/oracle evidence:

- Chapter 12.7 explicitly says `V_AND_B32`, `V_OR_B32`, and `V_XOR_B32` do not
  support input or output modifiers at `cdna1/README.md:3350` through `:3352`.
- Chapter 12.7.1 still allows VOP2 rows to use the VOP3 encoding, but says the
  promoted form cannot use a literal constant at `cdna1/README.md:3427`
  through `:3429`; the no-modifier restriction remains attached to the
  bitwise instruction rows.
- As a `gfx908` oracle, `llvm-mc` accepts ordinary
  `v_and_b32_e64 v0, v1, v2` but rejects `-v1`, `abs(v1)`, and `clamp` forms
  for the bitwise promoted rows.

XML evidence:

- XML includes promoted `ENC_VOP3` rows for the bitwise instructions. For
  example, `V_AND_B32` has a VOP3 opcode 275 row with `OPR_SRC_NOLIT` and
  `OPR_SRC_SIMPLE` source operands at `amdgpu_isa_cdna1.xml:59061` through
  `:59188`; `V_OR_B32` and `V_XOR_B32` have the same shape beginning at
  `amdgpu_isa_cdna1.xml:59205` and `:59349`.
- The generic `ENC_VOP3` field map describes `ABS`, `NEG`, `OMOD`, and `CLAMP`
  fields and says almost all VOP1/VOP2/VOPC/VINTRP opcodes can be promoted to
  use input and output modifiers at `amdgpu_isa_cdna1.xml:1616` through
  `:2116`.
- The operand types used by the bitwise VOP3 rows do not encode the
  per-instruction modifier prohibition: `OPR_SRC_NOLIT` and `OPR_SRC_SIMPLE`
  have `N/A` descriptions at `amdgpu_isa_cdna1.xml:111137` through `:111150`
  and `amdgpu_isa_cdna1.xml:113570` through `:113582`.

Impact:

XML-only consumers can discover the promoted bitwise VOP3 opcode rows and the
generic modifier fields, but cannot derive the manual/assembler rule that those
modifier bits are illegal for these instructions. This is an XML semantic
expressiveness gap, not an opcode-inventory gap.

### CDNA1-XML-052: VOP2 accumulator rows omit old-destination input metadata

Manual/oracle evidence:

- Chapter 12.7 defines `V_MAC_F32` and `V_MAC_F16` as accumulating into the old
  destination value at `cdna1/README.md:3353` and `:3376`.
- The dot-accumulate and FMAC rows explicitly state that the VOP2 version uses
  the `vDst` VGPR address as the third source at `cdna1/README.md:3413`
  through `:3424`.
- Chapter 12.7.1 says these VOP2 rows can also use promoted VOP3 opcodes at
  `cdna1/README.md:3427` through `:3429`, so the old-destination source
  contract applies to both encodings.

XML evidence:

- XML includes the accumulator instructions, but marks `VDST` as output-only
  (`Input="false"`, `Output="true"`) in both `ENC_VOP2` and promoted
  `ENC_VOP3` rows for `V_MAC_F32`, `V_MAC_F16`, `V_DOT2C_F32_F16`,
  `V_DOT2C_I32_I16`, `V_DOT4C_I32_I8`, `V_DOT8C_I32_I4`, `V_FMAC_F32`, and
  `V_PK_FMAC_F16`.
- Representative rows begin at `amdgpu_isa_cdna1.xml:59493`,
  `:61454`, `:64183`, `:64302`, `:64421`, `:64540`, `:64659`, and
  `:64778`.

Impact:

XML-only consumers can decode these opcodes but cannot infer that `VDST` is a
read dependency as well as the write destination. That can produce incorrect
def-use information, scheduling dependencies, or emulator operand wiring for the
VOP2 accumulator family unless the manual prose is cross-referenced.

### CDNA1-XML-053: `V_EXP_LEGACY_F32` and `V_LOG_LEGACY_F32` are missing from CDNA1 XML

Manual/oracle evidence:

- Chapter 12.8 defines `V_EXP_LEGACY_F32` at opcode 75 and
  `V_LOG_LEGACY_F32` at opcode 76, with legacy base-2 exponential/logarithm
  semantics, at `cdna1/README.md:3690` through `:3692`.
- Chapter 13.3.2 repeats those VOP1 opcode-table rows at
  `cdna1/README.md:6154` through `:6155`.
- As a `gfx908` oracle, `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908
  -show-encoding` assembles both `v_exp_legacy_f32_e32 v0, v1` and
  `v_log_legacy_f32_e32 v0, v1`, and also accepts their promoted E64 spellings.

XML evidence:

- Searches of `amdgpu_isa_cdna1.xml` find no `V_EXP_LEGACY_F32` or
  `V_LOG_LEGACY_F32` instruction rows.
- The generated XML VOP1 opcode set has holes at 75 and 76, even though the
  manual VOP1 opcode table reserves those numbers for the two legacy rows.

Impact:

XML-only generators treat legal CDNA1 legacy transcendental opcodes as absent.
That loses both compact VOP1 decode coverage and any promoted form derived from
the VOP1 opcode.

### CDNA1-XML-054: `V_READFIRSTLANE_B32` zero-EXEC and LDS-direct dataflow are under-described

Manual evidence:

- Chapter 12.8 says `V_READFIRSTLANE_B32` copies a VGPR value to an SGPR, allows
  source data from a VGPR or M0 for LDS direct access, selects
  `FindFirst1fromLSB(exec)`, uses lane 0 when `EXEC` is zero, and ignores the
  exec mask for the source access at `cdna1/README.md:3445`.

XML evidence:

- The XML instruction row describes reading "the lowest active lane" and stores
  into an SGPR at `amdgpu_isa_cdna1.xml:48662` through `:48664`, but does not
  encode the zero-`EXEC` lane-0 fallback or the source-read mask bypass.
- The XML uses `OPR_VGPR_OR_LDS` for `SRC0` at
  `amdgpu_isa_cdna1.xml:48676` through `:48680`, and that operand type has a
  `SRC_LDS_DIRECT` predefined value at selector 254 at
  `amdgpu_isa_cdna1.xml:125913` through `:125923`; however, the instruction
  operands do not surface an implicit M0 dependency for the LDS-direct form.

Impact:

XML-only consumers can discover the direct-LDS selector value, but cannot fully
derive the instruction's lane-selection and M0/dataflow contract. Emulators and
def-use tools need manual-derived special handling for `EXEC==0`, mask-bypassed
source reads, and `src_lds_direct`.

### CDNA1-XML-055: `V_SWAP_B32` operands are write-only in XML metadata

Manual evidence:

- Chapter 12.8 defines `V_SWAP_B32` as reading both operands, then writing the
  exchanged values: `tmp = D.u; D.u = S0.u; S0.u = tmp` at
  `cdna1/README.md:3703`.

XML evidence:

- XML represents `V_SWAP_B32` at opcode 81, but marks both `VDST` and `SRC0` as
  `Input="false"` and `Output="true"` in the compact VOP1 row at
  `amdgpu_isa_cdna1.xml:56246` through `:56261`.
- The promoted `ENC_VOP3` row repeats the same write-only metadata at
  `amdgpu_isa_cdna1.xml:56265` through `:56280`.

Impact:

XML consumers can decode the two destination registers, but cannot infer that
both registers are also read before being overwritten. That can produce
incorrect def-use, hazard, or emulator operand metadata for a read-modify-write
VALU instruction.

### CDNA1-XML-056: `V_CMP_CLASS_*` numeric-class mask bit meanings are prose-only

Manual evidence:

- Chapter 12.9 defines `V_CMP_CLASS_F32`, `V_CMPX_CLASS_F32`,
  `V_CMP_CLASS_F64`, `V_CMPX_CLASS_F64`, `V_CMP_CLASS_F16`, and
  `V_CMPX_CLASS_F16` as IEEE numeric-class tests, then enumerates the ten
  `S1.u` mask bits: signaling NaN, quiet NaN, negative infinity, negative
  normal, negative denormal, negative zero, positive zero, positive denormal,
  positive normal, and positive infinity at `cdna1/README.md:3801` through
  `:3861`.

XML evidence:

- XML descriptions for representative class rows say only that the second input
  is a 10-bit mask, for example `V_CMP_CLASS_F32` at
  `amdgpu_isa_cdna1.xml:69798` through `:69799` and `V_CMP_CLASS_F16` at
  `amdgpu_isa_cdna1.xml:70239` through `:70240`.
- The XML operand metadata preserves source sizes and result destinations:
  representative F32 VOPC rows expose `OPR_VCC`, `SRC0`, and `VSRC1` at
  `amdgpu_isa_cdna1.xml:69826` through `:69848`; representative F16 rows use
  16-bit `SRC0`/`VSRC1`/literal operands at `:70267` through `:70313`; and
  CMPX rows expose implicit `OPR_SDST_EXEC` at `:69945` through `:70030` and
  `:70386` through `:70472`. None of these rows encode the per-bit class map.

Impact:

XML-only consumers can decode the instruction shape and the existence of a
10-bit class mask, but cannot infer which mask bit corresponds to which IEEE
numeric class without manual-derived semantics.

### CDNA1-XML-057: VOP3B `SDST` field description is compare-specific

Manual evidence:

- Chapter 12.12 says VOP3B allows a unique scalar destination and is used only
  by `V_ADD_CO_U32`, `V_SUB_CO_U32`, `V_SUBREV_CO_U32`, `V_ADDC_CO_U32`,
  `V_SUBB_CO_U32`, `V_SUBBREV_CO_U32`, `V_DIV_SCALE_F32`, `V_DIV_SCALE_F64`,
  `V_MAD_U64_U32`, and `V_MAD_I64_I32` at `cdna1/README.md:4210` through
  `:4230`.
- Chapter 13.3.5 repeats the ten-opcode VOP3B list and defines `SDST [14:8]`
  as a scalar destination at `cdna1/README.md:6653` through `:6693`.

XML evidence:

- The generic `VOP3_SDST_ENC` field map describes `SDST` as "Destination for
  compare result" at `amdgpu_isa_cdna1.xml:7430` through `:7436`.
- Per-instruction operands show that VOP3B `SDST` is broader than a compare
  result: `V_ADD_CO_U32` writes an `OPR_SREG` carry mask at
  `amdgpu_isa_cdna1.xml:57619` through `:57647`, `V_DIV_SCALE_F32` writes an
  `OPR_SDST` post-scale mask at `:64154` through `:64192`, and
  `V_MAD_U64_U32` writes an `OPR_SREG` carry/overflow mask at `:64552`
  through `:64590`.

Impact:

Consumers that read the generic field description can misclassify VOP3B `SDST`
as compare-only even though the encoding's scalar destination carries carry,
borrow, post-scale, and wide-MAD mask results.

### CDNA1-XML-058: VOP3B scalar-result semantics are only partly described

Manual evidence:

- The VOP3B add/sub carry rows define unsigned carry/borrow predicates, allow an
  arbitrary SGPR-pair carry-out destination in VOP3, and say carry-in forms read
  the carry source from the SGPR pair at `S2.u` at `cdna1/README.md:3356`
  through `:3366`.
- `V_DIV_SCALE_F32` and `V_DIV_SCALE_F64` provide exponent/denormal branch
  pseudocode, zero numerator/denominator NaN behavior, `S0 == S1 || S0 == S2`
  usage, and the post-scale mask at `cdna1/README.md:4344` through `:4351`.
- `V_MAD_U64_U32` and `V_MAD_I64_I32` define `{vcc_out,D}` 65-bit-style
  multiply-add results at `cdna1/README.md:4361` through `:4362`.

XML evidence:

- XML records VOP3B operands and broad descriptions for carry rows such as
  `V_ADD_CO_U32` and `V_ADDC_CO_U32` at `amdgpu_isa_cdna1.xml:57494` through
  `:57647` and `:58143` through `:58177`, for `V_DIV_SCALE_F32/F64` at
  `:64154` through `:64232`, and for `V_MAD_U64_U32` / `V_MAD_I64_I32` at
  `:64552` through `:64632`.
- Those rows do not machine-encode the carry/borrow predicates, carry-in
  source-bit mapping, div-scale branch conditions and zero-case NaN behavior, or
  the precise 64-bit result plus scalar-mask split for the wide-MAD rows.

Impact:

XML consumers can decode the VOP3B operand shape, but need manual-derived
semantics to emulate or validate the scalar mask result for the full VOP3B
opcode family.

### CDNA1-XML-059: VOP3 F16 interpolation half-selection rules are prose-only

Manual evidence:

- `V_INTERP_P1LL_F16` says `attr_word` selects the high or low 16 bits of each
  LDS dword and is available only for 32-bank LDS at `cdna1/README.md:4408`
  through `:4410`.
- `V_INTERP_P1LV_F16` says `attr_word` selects the high or low half of both the
  VGPR parameter and LDS data, and is intended for 16-bank LDS at
  `cdna1/README.md:4411` through `:4413`.
- `V_INTERP_P2_LEGACY_F16` always writes the result to the low 16 LSBs of the
  destination, while `V_INTERP_P2_F16` uses `op_sel[3]` to choose the
  destination half and preserve the other half, at `cdna1/README.md:4414`
  through `:4420`.

XML evidence:

- The generic `ENC_VOP3` `OP_SEL` field says destination `OP_SEL[3] == 0`
  writes the low half and sets the high half to zero, while `OP_SEL[3] == 1`
  writes the high half and preserves the low half, at
  `amdgpu_isa_cdna1.xml:2707` through `:2708`.
- The F16 interpolation rows exist at `amdgpu_isa_cdna1.xml:66062` through
  `:66230`, but they do not encode the `attr_word` LDS/VGPR half-selection
  rules, the 16-bank versus 32-bank LDS restrictions, the legacy low-half-only
  writeback, or the `V_INTERP_P2_F16` preserve-the-other-half override.

Impact:

XML consumers can decode the VOP3 F16 interpolation rows, but must consult the
manual to interpret `attr_word`, LDS-bank legality, and destination-half
writeback accurately.

### CDNA1-XML-060: SDWA selectors and DPP control enumeration are under-structured

Manual evidence:

- The SDWA table assigns concrete selector values: `DST_SEL`/`SRC*_SEL` values
  `0-3` and `7` are reserved, `4` selects `data[15:0]`, `5` selects
  `data[31:16]`, and `6` selects `data[31:0]`; `DST_U` values select pad,
  sign/zero extend, preserve, or reserved behavior at `cdna1/README.md:6840`
  through `:6859`.
- The DPP table maps `DPP_CTRL` values to lane-routing functions, including
  quad permutes, row shifts/rotates, wave shifts/rotates, mirrors, and
  broadcasts, and marks `0x100` reserved at `cdna1/README.md:6908` through
  `:6936`.

XML evidence:

- The XML field records preserve bit positions but describe `DPP_CTRL` only as
  "Data-parallel primitive control" at `amdgpu_isa_cdna1.xml:5771` through
  `:5777`.
- The SDWA XML fields similarly describe `DST_SEL` as "Destination data
  select" and `DST_UNUSED` as "Format for unused destination bits" without the
  selector values or reserved-value map at `amdgpu_isa_cdna1.xml:6010` through
  `:6025`.

Impact:

XML-only consumers can locate the CDNA1 SDWA and DPP control fields, but cannot
derive the legal selector values, destination-preserve behavior, or DPP lane
mapping without the manual prose.

### CDNA1-XML-061: CDNA compact VOPC DPP availability conflicts between manual prose and XML/backends

Manual evidence:

- The DPP format description says the second dword can follow VOP1, VOP2, or
  VOPC instructions at `cdna1/README.md:6898` through `:6902`.
- The Chapter 12.18 DPP exclusion list prohibits F64/I64/U64/class compare
  families, but not ordinary F16/F32/I16/U16/I32/U32 compares, at
  `cdna1/README.md:5309` through `:5342`.

XML and oracle evidence:

- The CDNA1 XML defines `VOPC_VOP_SDWA_SDST_ENC` at
  `amdgpu_isa_cdna1.xml:7939`, but static XML search found no
  `VOPC_VOP_DPP` encoding name for CDNA1. The same scripted search found zero
  VOPC DPP rows for CDNA1 through CDNA4, while RDNA3/RDNA3.5/RDNA4 XML carries
  VOPC DPP encodings.
- `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908` accepts representative
  `v_mov_b32_dpp` and `v_mac_f32_dpp` rows, but rejects
  `v_cmp_eq_u32_dpp` and `v_cmp_eq_f64_dpp` with "dpp variant of this
  instruction is not supported"; the same `v_cmp_eq_u32_dpp` probe was also
  rejected for `gfx90a`, `gfx942`, and `gfx950`.

Impact:

The manual prose and the machine-readable/backend sources disagree on compact
CDNA VOPC DPP availability. Consumers need an architecture-specific oracle or
manual erratum rather than inferring legality from only the generic DPP format
text.

## No-Gap Notes

- The CDNA1 XML does include base instruction rows for `S_GETPC_B64`,
  `S_SETPC_B64`, `S_SWAPPC_B64`, `S_RFE_B64`, `S_BRANCH`, and the
  VCC/EXEC/SCC conditional branch families. The Chapter 3 gaps above are about
  missing state semantics, not missing base opcode rows.
- The CDNA1 XML's generic source-selector records do describe literal expansion
  for 16-bit operands and 64-bit unsigned, signed, and floating-point operands:
  representative `src_literal` text appears at
  `amdgpu_isa_cdna1.xml:104452` through `:104453`. The Chapter 6 literal gap
  above is therefore about cross-operand source-combination legality, not the
  literal value-extension rule itself.
- The CDNA1 XML has a distinct `OP_SEL_HI_2` field at
  `amdgpu_isa_cdna1.xml:2139`, matching the VOP3P split high-selector field in
  the manual at `cdna1/README.md:6724` and `:6737`.
- CDNA1 does not list `V_PK_FMA_F32`, `V_PK_MUL_F32`, `V_PK_ADD_F32`, or
  `V_PK_MOV_B32` in the manual, XML, or generated rocjitsu instruction set for
  this slice. The packed-F32 dword-pair issues are CDNA2+ for this audit.
- `V_PK_FMAC_F16` is a VOP2 instruction in CDNA1, not VOP3P, at
  `cdna1/README.md:3423` and `amdgpu_isa_cdna1.xml:62393`.
- CDNA1 MIX detailed instruction text and XML descriptions say multiply-add,
  while section 6.7 calls the operation fused. This is a manual-internal
  ambiguity, so it is not classified as an XML-only omission in this slice.
- The DPP section limits DPP compatibility to VOP1/VOP2 at
  `cdna1/README.md:1683` through `:1695`; no VOP3P DPP legality gap was found
  for the audited ordinary packed-math VOP3P entries.
- CDNA1 has no BF16_1K or F64 MFMA rows in the manual, XML, or generated
  rocjitsu instruction set for this slice.
- The CDNA1 XML has fixed AccVGPR C/D operand classes for MFMA: representative
  entries use `OPR_ACCVGPR` for VDST and `OPR_SRC_ACCVGPR_OR_CONST` for SRC2 at
  `amdgpu_isa_cdna1.xml:68782` through `:68818`, matching the Chapter 7 rule
  that C and D always use AccVGPRs.
- The CDNA1 XML opcode inventory matches the detailed Chapter 12.10 MFMA and
  `V_ACCVGPR_READ`/`V_ACCVGPR_WRITE` rows for the audited MAI slice; the
  `V_ACCVGPR_WRFED` issue above is specific to the separate Chapter 13.3 opcode
  table entry.
- The CDNA1 Chapter 12.1 SOP2 opcode inventory matches the checked-in XML after
  normalizing the manual OCR wrap on `S_PACK_LL_B32_B16`: both list 53 opcodes,
  0 through 52, from `S_ADD_U32` to `S_PACK_HH_B32_B16`.
- CDNA1 `S_RFE_RESTORE_B64` is present in both the detailed Chapter 12.1 manual
  table and XML as SOP2 opcode 43; the XML-only `S_RFE_RESTORE_B64` drift
  recorded for CDNA3 does not apply to CDNA1.
- The CDNA1 XML carries implicit SCC outputs and inputs for representative SOP2
  rows such as `S_ADD_U32`, `S_ADDC_U32`, `S_CSELECT_B32`, bitwise operations,
  shifts, BFE, and shifted-add rows. The Chapter 12.1 gaps above are about
  formula-level prose and literal-extension constraints, not total absence of
  SCC operand metadata.
- The CDNA1 Chapter 12.2 SOPK opcode inventory matches the checked-in XML:
  both list opcodes 0 through 18, omit opcode 19, place the literal-only
  `S_SETREG_IMM32_B32` at opcode 20, and place `S_CALL_B64` at opcode 21. The
  manual rows are at `cdna1/README.md:2942` through `:2968`, the Chapter 13
  opcode table is at `cdna1/README.md:5572` through `:5595`, and the XML rows
  span `amdgpu_isa_cdna1.xml:44562` through `:45367`.
- CDNA1 XML distinguishes the ordinary 32-bit SOPK encoding from the 64-bit
  literal form: `ENC_SOPK` is defined as 32 bits at
  `amdgpu_isa_cdna1.xml:435` through `:448`, while `SOPK_INST_LITERAL` is
  defined as 64 bits at `amdgpu_isa_cdna1.xml:5371` through `:5384` and is used
  by `S_SETREG_IMM32_B32` at `amdgpu_isa_cdna1.xml:45297` through `:45315`.
- The CDNA1 XML carries the key SOPK implicit dataflow metadata for this slice:
  `S_CMOVK_I32` has an implicit SCC input, the `S_CMPK_*` rows and
  `S_ADDK_I32` have implicit SCC outputs, `S_MULK_I32` has no SCC output, and
  `S_CALL_B64` has implicit PC input/output plus a label operand at
  `amdgpu_isa_cdna1.xml:44597` through `:45367`. The remaining HWREG
  permission/spacing omissions are already tracked by `CDNA1-XML-009` and
  `CDNA1-XML-010`.
- The CDNA1 SOP1 opcode inventory matches the Chapter 12.3 definition rows and
  the Chapter 13.1.3 opcode table after normalizing the markdown/OCR wrap:
  opcodes 0 through 46, 48, and 50 through 55 are present, while opcodes 47 and
  49 are holes, at `cdna1/README.md:2988` through `:3077` and
  `cdna1/README.md:5619` through `:5679`. XML rows span `S_MOV_B32` through
  `S_BITREPLICATE_B64_B32` at `amdgpu_isa_cdna1.xml:31871` through `:35273`.
  As a `gfx908` oracle, `llvm-mc` assembles representative SOP1 rows with those
  opcodes and warns that raw opcode-47 and opcode-49 encodings are invalid.
- The CDNA1 XML preserves important SOP1 operand metadata despite the
  prose-only formula gap above. For example, bitset rows mark `SDST` as both
  input and output at `amdgpu_isa_cdna1.xml:33375` through `:33467`, and the
  saveexec/wrexec rows expose implicit `EXEC` and `SCC` operands around
  `amdgpu_isa_cdna1.xml:35060` through `:35213`.
- The CDNA1 SOPC opcode inventory matches the Chapter 12.4 definition rows and
  Chapter 13.1.4 opcode table: opcodes 0 through 19 are present from
  `S_CMP_EQ_I32` through `S_CMP_LG_U64`, at `cdna1/README.md:3091` through
  `:3115` and `cdna1/README.md:5737` through `:5763`. XML rows span the same
  set at `amdgpu_isa_cdna1.xml:42322` through `:44545`, and a `gfx908`
  `llvm-mc` oracle assembled representative compare, bit-compare, VSKIP,
  GPR-index, and U64 compare rows with the same opcode positions.
- The CDNA1 XML carries the key SOPC implicit and operand-width metadata for
  ordinary compare and bit-compare rows. Representative `S_CMP_*` rows expose
  implicit SCC outputs, B32 bit-compare rows use a 32-bit bit index, and B64
  bit-compare rows use a 64-bit source with a 32-bit index at
  `amdgpu_isa_cdna1.xml:43666` through `:43988` and `:44338` through `:44545`.
  The VSKIP state contract and `S_SET_GPR_IDX_ON` raw-mode exception are the
  prose-only pieces tracked by the gaps above.
- The CDNA1 Chapter 12.5/13.1.5 SOPP opcode inventory matches the checked-in
  XML: both list opcodes 0 through 30 from `S_NOP` through
  `S_ENDPGM_ORDERED_PS_DONE` at `cdna1/README.md:3123` through `:3169` and
  `cdna1/README.md:5783` through `:5818`, while XML rows span
  `S_NOP` through `S_ENDPGM_ORDERED_PS_DONE` at
  `amdgpu_isa_cdna1.xml:45376` through `:46265`. A `gfx908` `llvm-mc` oracle
  assembled representative SOPP rows with the same opcode positions and
  rejected raw opcode-31 and opcode-32 SOPP encodings as invalid.
- The CDNA1 XML carries useful SOPP operand metadata despite the semantic gaps
  above: ordinary branches use `OPR_LABEL`, SCC/VCC/EXEC conditional branches
  include implicit condition operands, `S_WAITCNT` uses `OPR_WAITCNT`, and
  `S_SET_GPR_IDX_MODE` records implicit M0 input/output operands at
  `amdgpu_isa_cdna1.xml:45427` through `:45745` and `:46214` through `:46238`.
  `OPR_WAITCNT` also structures the VM, VM_HI, EXP, and LGKM bitfields at
  `amdgpu_isa_cdna1.xml:120475` through `:120519`; the remaining wait-counter
  gaps are about producer accounting and scheduling prose, not this immediate
  layout.
- The CDNA1 Chapter 12.6/13.2.1 SMEM opcode inventory matches the checked-in
  XML after normalizing markdown table wrapping and formatting loss in the
  atomic pseudocode. Manual rows list the load/store, cache/time/probe/discard,
  scalar-buffer atomic, and scalar atomic opcodes at `cdna1/README.md:3195`
  through `:3308`, and the Chapter 13 opcode table covers the same sparse
  range at `cdna1/README.md:5847` through `:5942`. XML rows span
  `S_LOAD_DWORD` through `S_ATOMIC_DEC_X2` at
  `amdgpu_isa_cdna1.xml:28487` through `:31829`; a scripted comparison found
  84 SMEM XML rows and no missing manual opcodes, and a `gfx908` `llvm-mc`
  oracle assembled representative load/store, cache, time, ATC probe, discard,
  scalar atomic, and scalar-buffer atomic rows.
- The CDNA1 XML carries the key SMEM operand-shape metadata despite the
  prose-only semantic gaps above. Representative rows distinguish 64-bit
  ordinary `SBASE` from 128-bit scalar-buffer resources, mark cache
  invalidation/writeback rows as operandless, expose 64-bit `S_MEMTIME` and
  `S_MEMREALTIME` destinations, use `OPR_SIMM8` for ATC probe payloads, and
  model `SDATA` widths for 32-bit, 64-bit, 128-bit, 256-bit, and 512-bit
  transfers at `amdgpu_isa_cdna1.xml:28487` through `:31829`. The remaining
  SMEM issues are the address, descriptor, atomic-return, cache/probe/time, and
  LGKM/clause semantics already tracked by `CDNA1-XML-034` through
  `CDNA1-XML-037`.
- The scalar selector table itself is represented for `POPS_EXITING_WAVE_ID`:
  the manual places it at selector 239 at `cdna1/README.md:1004`, and XML
  `OPR_SSRC` does the same at `amdgpu_isa_cdna1.xml:97717` through `:97720`.
- CDNA1 `S_MAX_I32` and `S_MAX_U32` tie behavior is consistent across the
  detailed manual and XML: the manual uses strict `S0 > S1` for both the
  selected value and SCC at `cdna1/README.md:2869` through `:2870`, and the XML
  says SCC is set iff the first value is selected at
  `amdgpu_isa_cdna1.xml:36410` through `:36547`.
- The CDNA1 Chapter 12.7/13.3.1 VOP2 opcode inventory matches the checked-in
  XML when `ENC_VOP2` rows are unioned with the VOP2 literal-extension rows:
  both list opcodes 0 through 61 from `V_CNDMASK_B32` through `V_XNOR_B32` at
  `cdna1/README.md:3318` through `:3425` and `cdna1/README.md:5969` through
  `:6044`. XML rows span `V_CNDMASK_B32` through `V_XNOR_B32` at
  `amdgpu_isa_cdna1.xml:56299` through `:65078`; a scripted comparison found
  62 VOP2 opcodes and no missing manual rows.
- CDNA1 XML represents the VOP2 literal-only FMA rows explicitly rather than
  dropping them from the base `ENC_VOP2` set. The F32 `V_MADMK_F32` and
  `V_MADAK_F32` rows use `VOP2_INST_LITERAL` at
  `amdgpu_isa_cdna1.xml:59612` through `:59755`, and the F16
  `V_MADMK_F16`/`V_MADAK_F16` rows carry 16-bit `SIMM32` operands at
  `amdgpu_isa_cdna1.xml:61573` through `:61735`.
- The CDNA1 XML carries the key VOP2 implicit VCC metadata for this slice:
  `V_CNDMASK_B32` exposes implicit `VCC`, and carry/borrow rows expose implicit
  `VCC` inputs/outputs. The old-destination accumulator omission is tracked
  separately in `CDNA1-XML-052`.
- The CDNA1 Chapter 12.8/13.3.2 VOP1 opcode inventory otherwise agrees with
  the checked-in XML once the missing legacy rows in `CDNA1-XML-053` are
  excluded: both cover `V_NOP` through `V_SWAP_B32`, include `V_CLREXCP` at
  opcode 53 and `V_SCREEN_PARTITION_4SE_B32` at opcode 55, and leave opcode
  holes 9, 54, 56, and 80 unused. The manual rows are at
  `cdna1/README.md:3437` through `:3707` and `cdna1/README.md:6055` through
  `:6159`; representative XML rows span `V_MOV_B32` through `V_SWAP_B32` at
  `amdgpu_isa_cdna1.xml:48548` through `:56290`.
- The CDNA1 XML does carry the basic direct-LDS source selector for
  `V_READFIRSTLANE_B32`: the instruction uses `OPR_VGPR_OR_LDS` at
  `amdgpu_isa_cdna1.xml:48676` through `:48680`, and that operand class defines
  selector 254 as `SRC_LDS_DIRECT` at `amdgpu_isa_cdna1.xml:125913` through
  `:125923`. `CDNA1-XML-054` is about the missing instruction-level
  zero-`EXEC`, mask-bypass, and implicit-M0 details, not absence of the selector
  itself.
- The CDNA1 Chapter 12.9/13.3.3 VOPC opcode inventory matches the checked-in
  XML after normalizing the manual range-table wording drift. The detailed
  manual rows and Chapter 13 opcode table list 198 opcodes from
  `V_CMP_CLASS_F32` through `V_CMPX_T_U64`, with holes at 0-15, 22-31, and
  128-159, at `cdna1/README.md:3797` through `:3861` and
  `cdna1/README.md:6263` through `:6470`. XML compact and literal rows cover
  the same opcode set from `V_CMP_CLASS_F32` through `V_CMPX_T_U64`, and a
  scripted comparison found no missing or extra VOPC opcodes.
- The CDNA1 XML carries the key VOPC result metadata for ordinary compares:
  compact `V_CMP_EQ_U32` exposes an `OPR_VCC` output at
  `amdgpu_isa_cdna1.xml:86647` through `:86669`, while compact
  `V_CMPX_EQ_U32` exposes both `OPR_VCC` output and implicit `OPR_SDST_EXEC`
  output at `amdgpu_isa_cdna1.xml:88677` through `:88703`. `CDNA1-XML-056` is
  specifically about class-mask bit meanings, not missing VCC/EXEC metadata.
- The CDNA1 Chapter 12.12/13.3.4 native VOP3A opcode inventory matches
  `ENC_VOP3` after excluding promoted VOP1/VOP2/VOPC/VINTRP aliases: the
  manual lists 103 native opcodes from 448 through 672 at
  `cdna1/README.md:4234` through `:4419` and `:6528` through `:6651`, and a
  scripted XML comparison found no missing native rows.
- The CDNA1 VOP3B opcode inventory is present in XML: `VOP3_SDST_ENC` carries
  all ten scalar-destination forms, including the VOP2-promoted carry opcodes
  281 through 286 and native opcodes 480, 481, 488, and 489. `CDNA1-XML-057`
  and `CDNA1-XML-058` are about field prose and structured semantics, not
  missing opcode rows.
- The CDNA1 compact VINTRP and promoted/native VOP3 interpolation opcode
  inventory matches the manual slice. XML lists compact `V_INTERP_P1_F32`,
  `V_INTERP_P2_F32`, and `V_INTERP_MOV_F32` as `ENC_VINTRP` opcodes 0, 1, and
  2, plus promoted `ENC_VOP3` opcodes 624, 625, and 626 at
  `amdgpu_isa_cdna1.xml:46275` through `:46490`; XML also lists the VOP3 F16
  interpolation opcodes 628 through 631 at `:66062` through `:66230`.
- The CDNA1 XML preserves the important compact VINTRP operand metadata for this
  slice: `V_INTERP_P2_F32` marks `VDST` as both input and output at
  `amdgpu_isa_cdna1.xml:46359` through `:46364`, promoted VOP3 F32 rows store
  the attribute in `SRC0` and the VGPR/parameter selector in `SRC1` at
  `:46319` through `:46330` and `:46395` through `:46405`, and `OPR_PARAM`
  defines `p10`, `p20`, and `p0` values at `:95056` through `:95074`.
- The CDNA1 XML opcode inventory distinguishes the Global-only floating-point
  atomics from the Flat and Scratch opcode sets: `GLOBAL_ATOMIC_ADD_F32` and
  `GLOBAL_ATOMIC_PK_ADD_F16` are present at `amdgpu_isa_cdna1.xml:17899` and
  `:17965`, while searches found no `FLAT_ATOMIC_ADD_F32`,
  `FLAT_ATOMIC_PK_ADD_F16`, or `SCRATCH_ATOMIC_*` rows for CDNA1.
- The CDNA1 XML does include the ordinary DS encoding fields and many base
  load/store/atomic/GWS rows. The Chapter 11 XML gaps above are about missing
  semantic rules and the targeted absence of the SRC2 opcode family, not a
  wholesale absence of DS metadata.
- The CDNA1 Chapter 12.13/13.5.1 DS opcode inventory matches XML once the
  targeted SRC2 omission in `CDNA1-XML-031` is excluded. A scripted comparison
  over the manual's 154 DS opcode/name pairs found 123 XML `ENC_DS` rows, with
  the 31 missing rows exactly equal to `DS_*_SRC2_*` opcodes 128 through 149 and
  192 through 211. The XML includes less visually obvious manual rows such as
  `DS_WRITE_B16_D16_HI` at `amdgpu_isa_cdna1.xml:12001` through `:12033` and
  `DS_CONDXCHG32_RTN_B64` at `:13481` through `:13524`.
- The CDNA1 Chapter 12.14/12.15 and 13.6.1/13.6.2 buffer opcode inventory
  matches XML after normalizing markdown/OCR table wrapping. A scripted
  comparison found all 16 MTBUF rows and all 71 sparse MUBUF rows present with
  no extras: XML `ENC_MTBUF` covers `TBUFFER_LOAD_FORMAT_X` through
  `TBUFFER_STORE_FORMAT_D16_XYZW` at `amdgpu_isa_cdna1.xml:24373` through
  `:25093`, and XML `ENC_MUBUF` covers `BUFFER_LOAD_FORMAT_X` through
  `BUFFER_ATOMIC_DEC_X2` at `:25141` through `:28439`. Existing gaps
  `CDNA1-XML-017` through `CDNA1-XML-020` cover the remaining buffer semantic
  prose that is not represented as structured XML metadata.
- The CDNA1 Chapter 12.16/13.7.1 MIMG opcode inventory matches XML after
  normalizing sparse opcode holes and manual table wrapping. A scripted
  comparison found all 92 manual `IMAGE_*` rows present in XML `ENC_MIMG`, with
  no extras, spanning `IMAGE_LOAD` at `amdgpu_isa_cdna1.xml:20101` through
  `IMAGE_SAMPLE_C_CD_CL_O` at `:24325`. Existing gaps `CDNA1-XML-021` through
  `CDNA1-XML-025` cover the image semantic prose that remains under-structured
  in XML.
- The CDNA1 Chapter 12.17/13.8 flat-family opcode inventory matches XML after
  normalizing the separate segment tables. A scripted comparison found all 48
  `FLAT_*` rows present in `ENC_FLAT` from `FLAT_LOAD_UBYTE` through
  `FLAT_ATOMIC_DEC_X2`, all 50 `GLOBAL_*` rows present in `ENC_FLAT_GLBL`
  including `GLOBAL_ATOMIC_ADD_F32` and `GLOBAL_ATOMIC_PK_ADD_F16`, and all 22
  `SCRATCH_*` rows present in `ENC_FLAT_SCRATCH`. The corresponding encoding
  field records preserve the manual's offset split: 12-bit unsigned offset for
  FLAT and 13-bit signed offset for GLOBAL/SCRATCH. Existing gaps
  `CDNA1-XML-026` through `CDNA1-XML-028` cover the flat-family semantic prose
  that remains under-structured in XML.
- The CDNA1 XML extension availability matches the Chapter 12.18 exclusion
  lists for representative VOP1/VOP2 DPP and SDWA cases: XML omits DPP rows for
  `V_READFIRSTLANE_B32`, `V_CVT_I32_F64`, `V_RCP_F64`, `V_CLREXCP`, and
  `V_SWAP_B32`, while retaining legal DPP for `V_MAC_F32`; XML also omits SDWA
  rows for `V_MAC_F32`, `V_MAC_F16`, `V_FMAC_F32`, the `V_MADMK/MADAK_*`
  literal rows, `V_READFIRSTLANE_B32`, `V_CLREXCP`, and `V_SWAP_B32`.
