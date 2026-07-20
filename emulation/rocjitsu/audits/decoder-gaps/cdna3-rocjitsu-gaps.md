# CDNA3 Rocjitsu Gaps

Architecture: CDNA3 / MI300

Manual source: `workspace_docs/amdgpu-isa-manuals/cdna3/README.md`

Rocjitsu source: `emulation/rocjitsu`

## Coverage

| Area | Status | Notes |
| --- | --- | --- |
| CDNA3 introduction and program organization | Audited statically | Checked Chapters 1-2 wave/workgroup/dispatch/SALU/VALU/EXEC/LDS/GWS/device-memory prose against ISA profiles, dispatch entry/command processor paths, wavefront/CU initialization, LDS backing, memory pipeline, and existing gap/no-gap notes. |
| CDNA3 Chapter 12.10 VOP3P definitions | Audited statically | Checked generated decode/autogen/runtime for the full VOP3P opcode set across packed 16-bit arithmetic, MIX, DOT, packed F32 arithmetic, `V_PK_MOV_B32`, `V_ACCVGPR_READ`/`WRITE`, dense MFMA, and sparse SMFMAC rows. |
| CDNA3 Chapter 12.11 VOP3A/VOP3B definitions | Audited statically | Checked full generated decode/autogen/runtime inventory for VOP3A/VOP3B rows, VOP3B scalar-destination execution, VOP3 no-literal handling, hard-stubbed VOP3A rows, `V_LSHL_ADD_U64`, packed F32 conversions, FP8/BF8 packed destination preservation, stochastic seed helper coverage, and VOP3B `OPR_SREG` def-use metadata. |
| CDNA3 `V_CVT_{PK,SR}_{FP8,BF8}_F32` execution | Audited statically | Focused on helper selection, destination preservation, source operand classes, and runtime state dependencies. |
| CDNA3 FP8/BF8 helper selection codegen | Audited statically | Checked architecture selection between FNUZ and OCP helpers. |
| CDNA3 FP8/BF8 tests | Audited statically | Checked helper and HIP conversion tests for overflow-mode coverage, CDNA3 packed low-half destination preservation, and stochastic seed-formula coverage. |
| CDNA3 dense MFMA layout and execution | Audited statically | Checked generated dense F32/F16/BF16/I8/XF32/F64 MFMA classes, ACC/ACC_CD operand rewriting, shared layout formulas, broadcast helpers, full-wave access, and clamp/state handling. |
| CDNA3 MFMA broadcast field handling | Audited statically | Checked `CBSZ`/`ABID` A-lane broadcast, `BLGP` B-lane permutation, generated dense MFMA field plumbing, and F64 `BLGP`-as-negation override. |
| CDNA3 MFMA floating-point mode and exception behavior | Audited statically | Checked section 7.3 denorm/MODE/rounding and DGEMM exception rules against generated dense MFMA execution and shared helpers. |
| CDNA3 sparse MFMA execution | Audited statically | Checked generated SMFMAC constructors/execution stubs, shared sparse helpers, `SRC2`/`ACC_CD` handling, `CBSZ`/`ABID` sparse-index selection, index-pair/alignment validation, generated DBT legalization rows, and adjacent tests. |
| CDNA3 MAI dependency waits | Audited statically | Checked section 7.5 required-wait rules against generated instruction flags, dispatch, and wait-counter state. |
| CDNA3 scalar memory execution | Audited statically | Checked generated SMEM loads/stores/cache/time/discard/atomics, scalar address helpers, scalar memory pipeline, wait counters, `GLC`/`NV` handling, and adjacent tests. |
| CDNA3 SMEM instruction definitions | Audited statically | Checked full Chapter 12.6 opcode inventory, generated SMEM constructors and decoder table, load/store/cache/time/discard/atomic operand widths, XML-only ATC probe decode stubs, existing SMEM runtime gaps, and adjacent smoke/test coverage. |
| CDNA3 vector buffer execution | Audited statically | Checked generated MUBUF/MTBUF constructors and execution for the Chapter 12.13/12.14 slice, shared buffer address helpers, vector-memory pipeline completion, cache flag helpers, semantic derivation, no-return atomic metadata, cache-maintenance ACK behavior, MTBUF D16 load/store packing, and adjacent tests. |
| CDNA3 float memory atomics | Audited statically | Checked Chapter 9.2 LDS/L2 float atomic numeric rules, generated MUBUF/FLAT/DS execute bodies, shared atomic RMW pipeline behavior, packed-lane lowering, and adjacent coverage. |
| CDNA3 flat/global/scratch memory execution | Audited statically | Checked generated FLAT classes, shared flat address helper, segment-specific operand shaping, private/shared aperture routing, direct-LDS hooks, segment-specific decode validity including reserved `SEG`, wait-counter behavior, atomic execution/metadata, semantic derivation, disassembly modifiers, and adjacent tests. |
| CDNA3 data-share LDS execution | Audited statically | Checked generated DS load/store/atomic bodies, shared DS address helpers, local-memory pipeline completion, WRAP, ADDTID, CONDXCHG32, APPEND/CONSUME, permute/swizzle helpers, ACC handling, GDS handling, packed/scalar floating LDS atomics, and adjacent tests. |
| CDNA3 data-share GWS decode/autogen/runtime | Audited statically | Checked generated GWS constructors, opcode table entries, generated encoding fixtures, LLVM `llvm-mc` syntax/encoding oracle, codegen semantic classification, and runtime stubs. |
| CDNA3 SALU scalar operand execution | Audited statically | Checked generated SOP1/SOP2/SOPK/SOPC/SOPP constructors, scalar operand read/write helpers, dynamic literal handling, out-of-range and 64-bit pair prose, and SCC side effects. |
| CDNA3 SALU def-use metadata | Audited statically | Checked representative XML implicit SCC operands against generated instruction operand arrays, `implicit_uses`/`implicit_defs`, and `RegisterSet` special-register handling. |
| CDNA3 SALU SCC execution | Audited statically | Checked Chapter 5.3-5.7 SCC-producing and SCC-consuming arithmetic, conditional, compare, bitwise, shift, bitfield, abs, and absdiff behavior against shared execution helpers and adjacent scalar SCC tests. |
| CDNA3 SOP1 instruction definitions | Audited statically | Checked Chapter 12.3 full SOP1 opcode inventory, generated constructors and decode table, literal handling, direct PC forms, shared unary/EXEC helpers, relative SGPR addressing, existing RFE/fork/join/GPR-index gaps, and adjacent smoke/test coverage. |
| CDNA3 SOP2 instruction definitions | Audited statically | Checked Chapter 12.1 full SOP2 opcode inventory, generated constructors and decode table, literal handling, shared scalar execution helpers, bitfield and pack rows, `S_ABSDIFF_I32` edge examples, shifted-add carry rows, existing fork/RFE gaps, and adjacent tests. |
| CDNA3 SOPK instruction definitions | Audited statically | Checked Chapter 12.2 full SOPK opcode inventory, generated constructors and decode table, literal-only opcode 20 handling, signed and unsigned SIMM16 helpers, ADDK/MULK old-destination dataflow, SCC effects, direct-call metadata, existing fork/HWREG gaps, and adjacent tests. |
| CDNA3 SOPC instruction definitions | Audited statically | Checked Chapter 12.4 full SOPC opcode inventory, generated constructors and literal handling, shared compare/bit-compare/GPR-index helpers, existing VSKIP/GPR-index gaps, LLVM raw-mode oracle behavior, and adjacent SCC tests. |
| CDNA3 SOPP instruction definitions | Audited statically | Checked Chapter 12.5 full SOPP opcode inventory, generated constructors and decode table, branch/waitcnt/barrier/endpgm behavior, XML-only ordered-PS/co-execution rows, existing trap/debug/status/message/GPR-index gaps, and adjacent branch/barrier tests. |
| CDNA3 VOP2 instruction definitions | Audited statically | Checked full Chapter 12.7 generated VOP2/VOP3 inventory, literal/DPP/SDWA extension dispatch, CNDMASK S2, carry-out SDST/SRC2 forms, FP min/max helpers, F16 MAC `OPSEL[3]`, LDEXP_F16 exponent sizing, dot accumulators, packed FMAC, literal-only `_MK`/`_AK` handling, and adjacent SIMD tests. |
| CDNA3 Chapter 12.8 VOP1 definitions | Audited statically | Checked generated VOP1 opcode inventory, VOP3 aliases, XML-only opcode 55, `V_READFIRSTLANE_B32` EXEC handling, move modifiers, saturation/fract/CLREXCP helpers, swap dataflow, DPP/SDWA exclusions, and adjacent tests. |
| CDNA3 Chapter 12.9 VOPC definitions | Audited statically | Checked generated VOPC opcode inventory, VOP3A aliases, compare/class scalar and SIMD helpers, CMPX VCC/EXEC/SDST writeback, literal/DPP/SDWA extension dispatch, DPP write-mask merge, semantic derivation, and adjacent generator/runtime tests. |
| CDNA3 Chapter 12.16 DPP/SDWA limitations | Audited statically | Checked generated VOP1/VOP2/VOPC extension dispatch against the DPP/SDWA deny lists, XML extension-row availability, 64-bit `DPP_ROW*` control legality, and LLVM assembler behavior for representative 64-bit DPP forms. |
| CDNA3 SALU access instructions | Audited statically | Checked `S_GETREG_B32`, `S_SETREG_B32`, and `S_SETREG_IMM32_B32` constructors, HWREG operand display, literal extension handling, generated execution helpers, wavefront state storage, and adjacent HWREG tests. |
| CDNA3 wave/kernel state | Audited statically | Checked Chapter 3.1-3.6 PC/EXEC/STATUS/MODE state, raw STATUS launch/reuse behavior, `XNACK_MASK`, helper bits, VSKIP, GPR/LDS allocation and aliasing, out-of-range behavior, and LDS allocation/clamping against `Wavefront`, generated SALU accessors, branch/source helpers, and CU/SPI allocation code. |
| CDNA3 special wave/kernel state | Audited statically | Checked Chapter 3.7-3.13 M0, SCC, VCC/VCCZ, trap/exception/TRAPSTS/memory-violation state, HW_ID/XCC_ID, TTMP privilege/init, system SGPR launch payloads, and packed VGPR0 workitem ID against operands, SOPK/SOPP execution, and command-processor initialization. |
| CDNA3 program control and branching | Audited statically | Checked Chapter 4.1-4.2 control/branch tables, direct PC operations including the zero-target path, trap return, debug conditional branches, control-status/message/perf/trace helpers, and adjacent decode-only coverage. |
| CDNA3 workgroups and `S_BARRIER` | Audited statically | Checked Chapter 4.3 workgroup size and barrier semantics, generated `S_BARRIER` class plumbing, `WfState::BARRIER` scheduler release, workgroup dispatch bookkeeping, `STATUS.IN_BARRIER`, and adjacent tests. |
| CDNA3 wait-counter dependency resolution | Audited statically | Checked Chapter 4.4 `S_WAITCNT` threshold decoding, runtime wait-counter state, memory-pipeline producer accounting, scalar-memory dword counts, `S_SENDMSG`, FLAT counter notes, GWS `EXP_CNT`, and adjacent tests. |
| CDNA3 manually inserted wait states | Audited statically | Checked Chapter 4.5 Table 11/12 NOP hazard rows, generated `S_NOP`, adjacent `S_ICACHE_INV` spacing, runtime timing/scoreboard hooks, GFX12 DBT hazard insertion, and adjacent decode/execution tests. |
| CDNA3 arbitrary divergent control flow | Audited statically | Checked Chapter 4.6 fork/join branch-stack prose, generated `S_CBRANCH_{G,I}_FORK` and `S_CBRANCH_JOIN` constructors/execution, branch-stack state storage, generated encoding fixtures, and adjacent tests. |
| CDNA3 VALU formats and operands | Audited statically | Checked Chapter 6.1-6.2 VOP1/VOP2/VOP3/VOPC overview, generated source/destination operand classes, literal handling, scalar-source restrictions, VOP3 output modifiers, MODE storage hooks, EXEC-masked writes, carry/compare destinations, and adjacent tests. |
| CDNA3 dense MFMA tests | Audited statically | Checked CTS skip list, generated coverage exceptions, and expensive SIMD exact-test architecture coverage. |
| CDNA3 scalar memory tests | Audited statically | Checked the SBASE operand regression and adjacent RDNA SMEM address tests; uncovered CDNA3 selector, M0, scratch, buffer-resource, atomic, cache-policy, and counter cases are recorded below. |
| CDNA3 vector buffer tests | Audited statically | Checked operand-width/SRSRC and LDS-disassembly tests; uncovered descriptor addressing, data-format conversion, LDS clamping/M0, and cache-policy cases are recorded below. |
| CDNA3 flat/global/scratch tests | Audited statically | Checked shared address-calculation and decode smoke coverage; uncovered CDNA3-specific PTR32, reserved `SEG=3`, unsigned global VGPR offsets, signed immediate-offset disassembly, direct LDS, mixed-lane aperture routing, FP atomics, wait-counter, and error-policy cases are recorded below. |
| CDNA3 data-share tests | Audited statically | Checked decode smoke, VM atomic stress, ACC DS transpose, and shared-infra coverage adjacent to DS; uncovered M0 clamp, equal-offset READ2/WRITE2, no-return atomic destination preservation, packed LDS atomics, GDS/GWS, and FP-mode cases are recorded below. |
| CDNA3 rocjitsu surface through Chapter 13 | Complete | Coverage rows above cover the CDNA3 manual chapters and instruction-definition families; detailed entries below record the remaining decode, execution, metadata, disassembly, DBT, and test gaps found in this pass. |

## Gaps

### CDNA3-RJ-001: FP8/BF8 conversions hard-wire FNUZ helpers and cannot model runtime state

Manual evidence:

- `cdna3/README.md:1985` through `:1989` defines CDNA3 FNUZ-style FP8/BF8
  numeric encodings.
- `cdna3/README.md:2034` through `:2048` defines `FP16_OVFL` conversion
  behavior and the `SH_MEM_CONFIG[8]` prerequisite.

Rocjitsu evidence:

- Codegen selects FNUZ helpers only by architecture name at
  `lib/python/amdisa/codegen/execute/fp8_formats.py:8` through `:28`.
- Generated CDNA3 execution calls FNUZ helpers directly for
  `V_CVT_PK_FP8_F32`, `V_CVT_PK_BF8_F32`, `V_CVT_SR_FP8_F32`, and
  `V_CVT_SR_BF8_F32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3.cpp:6084` through
  `:6089`, `:6118` through `:6123`, `:6152` through `:6160`, and `:6189`
  through `:6197`.
- None of these paths reads MODE state or `SH_MEM_CONFIG`.

Impact:

The CDNA3 helper selection handles the architectural FNUZ numeric format, but
the conversion execution still has only one behavior for Inf/overflow and no
configuration prerequisite check.

### CDNA3-RJ-002: Generated source operands allow forms the manual says are illegal

Manual evidence:

- `cdna3/README.md:2038` says `CVT_SR_*` and `CVT_PK_*` support only VGPR
  inputs, not SGPRs, literal constants, or inline constants.

Rocjitsu evidence:

- Generated CDNA3 constructors use `OPR_SRC_NOLIT` and `OPR_SRC_SIMPLE` for
  `V_CVT_PK_FP8_F32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3.cpp:6059` through
  `:6064`.
- The stochastic forms use the same broad operand classes at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3.cpp:6127` through
  `:6132` and `:6164` through `:6169`.
- These generated classes come from XML classes that include scalar-register
  subtypes.

Impact:

Rocjitsu inherits the XML/manual legality mismatch. Hardware and LLVM oracle
behavior still need a focused check before deciding whether to enforce the
manual-only VGPR rule or update the audit classification.

### CDNA3-RJ-003: FP8/BF8 tests do not cover alternate mode/configuration behavior

Rocjitsu evidence:

- Shared helper tests cover fixed helper outcomes, including OCP FP8/BF8 cases
  at `tests/data_types_test.cpp:274` through `:331` and `:493` through `:504`.
- HIP conversion tests use finite values and helper-derived expectations for
  non-scaled FP8/BF8 at `tests/hip_cvt_narrow_test.cpp:216` through `:309`.

Impact:

The tests cover finite conversion plumbing and helper-specific behavior, but
not CDNA3 `FP16_OVFL` alternatives or `SH_MEM_CONFIG[8]` effects.

### CDNA3-RJ-004: FP8/BF8 forwarding hazard is not modeled

Manual evidence:

- `cdna3/README.md:2009` requires spacing between `CVT_*_F32` conversions that
  write low/high halves or bytes of the same destination register.

Rocjitsu evidence:

- The generated execution bodies are functional per-instruction operations and
  do not track a 4-cycle forwarding hazard for these conversions.

Impact:

Rocjitsu can execute sequences that the manual requires software to space on
hardware. This is a timing/scheduling gap rather than a single-instruction
decode bug.

### CDNA3-RJ-005: Packed F32 arithmetic broadcasts scalar sources instead of reading 64-bit pairs

Manual/XML evidence:

- CDNA3 section 6.7 says packed 32-bit instructions operate on two dwords at a
  time and that `OPSEL`/`OPSEL_HI` select the first or second dword for each
  source at `cdna3/README.md:1527`.
- `V_PK_FMA_F32`, `V_PK_MUL_F32`, and `V_PK_ADD_F32` read source dwords
  `[31:0]` and `[63:32]` in the detailed pseudocode at
  `cdna3/README.md:11495` through `:11529`.
- The CDNA3 XML encodes these sources as 64-bit operands at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna3.xml:65983` through
  `:66095`.

Rocjitsu evidence:

- The generated CDNA3 constructors preserve the 64-bit operand sizes for
  `V_PK_FMA_F32`, `V_PK_MUL_F32`, and `V_PK_ADD_F32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3p.cpp:623` through
  `:673`.
- The shared generator reads a 64-bit pair only when the encoding value is a
  VGPR, otherwise it initializes the high word from the low 32-bit read at
  `lib/python/amdisa/codegen/execute/packed.py:19` through `:38`.
- The generated helpers follow that pattern for `V_PK_ADD_F32`,
  `V_PK_FMA_F32`, and `V_PK_MUL_F32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:16709`
  through `:16722`, `:16897` through `:16917`, and `:17361` through `:17373`.
- The operand layer already has scalar-pair support through `read_lane64()` and
  `resolve_src_scalar64()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/operand.cpp:1121` through
  `:1150` and `:1368` through `:1381`, and `V_PK_MOV_B32` uses that path
  unconditionally.

Impact:

A scalar-pair source such as `s6:s7` with different low/high dwords feeds the
low dword to both packed F32 components in rocjitsu. The manual and XML both
describe these operands as 64-bit pairs, so the high component should come from
the second dword of the pair before `OPSEL_HI` selection.

### CDNA3-RJ-006: Packed 32-bit VOP3P VGPR pair alignment is not validated

Manual evidence:

- CDNA3 section 6.7 says packed 32-bit operands must be two-dword aligned, with
  an even VGPR address, at `cdna3/README.md:1527`.
- The `V_PK_MOV_B32` notes repeat that sources are 64-bit operands subject to
  alignment restrictions for both SGPR and VGPR at `cdna3/README.md:11545`
  through `:11558`.

Rocjitsu evidence:

- The packed F32 and `V_PK_MOV_B32` constructors use raw 64-bit VGPR/source
  operands at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3p.cpp:623`
  through `:690`.
- `Isa::resolved_vgpr_offset()` accepts any source VGPR encoding from 256
  through 511 and returns the unadjusted VGPR index at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/operand.cpp:1305` through
  `:1311`.
- `Operand::read_lane64()` and `write_lane64()` then read or write the pair
  starting at that index at `operand.cpp:1368` through `:1392`.

Impact:

Odd VGPR encodings are treated as valid pairs rooted at the odd register,
whereas the manual restricts packed 32-bit operands to even-aligned pairs.

### CDNA3-RJ-007: Packed F16 VOP3P arithmetic ignores `CLMP`

Manual/XML evidence:

- The CDNA3 VOP3P format includes `CLMP` as "1 = clamp result" at
  `cdna3/README.md:23958` through `:23960`.
- The CDNA3 XML describes generic VOP3P `CLAMP` as clamping output to
  `[0.0, 1.0]` at `shared/machine-readable-isa/isa/amdgpu_isa_cdna3.xml:1990`
  through `:1991`.

Rocjitsu evidence:

- `V_PK_ADD_F16`, `V_PK_FMA_F16`, `V_PK_MAX_F16`, `V_PK_MIN_F16`, and
  `V_PK_MUL_F16` execute without applying `inst.inst_.clamp` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:16664`
  through `:16698`, `:16838` through `:16884`, `:17089` through `:17127`,
  `:17192` through `:17230`, and `:17316` through `:17350`.
- MIX helpers do apply `std::clamp(result, 0.0f, 1.0f)` when
  `inst.inst_.clamp` is set at `execute_shared.h:13290` through `:13411`, so
  the generated code already has a working pattern for floating VOP3P clamp.

Impact:

Packed F16 results outside `[0, 1]` remain unclamped for the ordinary packed
F16 VOP3P arithmetic helpers even when the VOP3P `CLMP` bit is set.

### CDNA3-RJ-008: XF32 MFMA executes as ordinary F32

Manual evidence:

- The CDNA3 manual says XF32 MFMA takes 32-bit floats but rounds the mantissa
  to 10 bits for reduced-precision multiplication at `cdna3/README.md:1640`
  through `:1642`.

Rocjitsu evidence:

- The matrix execute generator maps `XF32` into the ordinary F32 specialized
  family and `amdgpu::extract_f32` extractor at
  `lib/python/amdisa/codegen/execute/matrix.py:21` and `:220` through `:224`.
- `VMfmaF3216x16x8Xf32Vop3pMfma::execute_impl()` and
  `VMfmaF3232x32x4Xf32Vop3pMfma::execute_impl()` both call
  `exec_f32_mfma_f32_spec` with ordinary F32 extractors at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3p.cpp:767` through
  `:820`.
- The shared F32 MFMA path multiplies the extracted float values directly at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:1179` through
  `:1190`.
- Searching rocjitsu for `xf32`, `mantissa`, and reduced-precision wording
  finds only the generated XF32 class names and unrelated comments, not an XF32
  rounding helper.

Impact:

CDNA3 XF32 MFMA results match ordinary F32 MFMA in rocjitsu, so inputs whose
low mantissa bits affect the product will diverge from the manual's reduced
precision rule.

### CDNA3-RJ-009: Dense MFMA register-block alignment is not validated

Manual evidence:

- MFMA input and output register blocks must be contiguous, and the first
  register must be aligned to the number of registers required by the operand,
  at `cdna3/README.md:1562`.
- The dense MFMA rule table also says `SRC0`, `SRC1`, `SRC2`, and `VDST` need
  even alignment when encoded as VGPR operands at `cdna3/README.md:1637`
  through `:1638`.

Rocjitsu evidence:

- Generated dense MFMA constructors expose raw register encodings and operand
  sizes without checking block-width alignment; representative XF32
  constructors are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3p.cpp:735` through
  `:820`, and I8/F64 constructors follow the same generated pattern around
  `:1307` through `:1393` and `:1984` through `:2025`.
- `dst_base()` and `src_base()` map the encoded register number to a physical
  VGPR/AccVGPR base but do not validate alignment at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:72` through
  `:92`.
- The shared layout helpers then assume a contiguous block by adding computed
  register offsets from the base at `mma_exec.h:172` through `:197` and
  `:204` through `:225`.

Impact:

Illegal dense MFMA encodings with odd or under-aligned source, accumulator, or
destination bases execute as if valid. This is broader than the packed-VOP3P
two-register alignment gap because dense MFMA operands can require much larger
contiguous register blocks.

### CDNA3-RJ-010: Dense MFMA clamp and overflow state is not modeled

Manual evidence:

- The dense MFMA rule table says clamp is supported, uses `FP16_OVFL` from
  MODE, clamps F32 overflow to `+/-MAX` when set and otherwise produces
  `+/-INF`, and clamps I32 overflow/underflow to `+/-MAX` when set at
  `cdna3/README.md:1628` through `:1633`.

Rocjitsu evidence:

- Generated dense F32/XF32/F16/BF16 MFMA execution passes only source bases,
  destination base, constant-accumulator state, `CBSZ`, `ABID`, and `BLGP` into
  shared helpers; representative calls are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3p.cpp:774` through
  `:820`, `:1082` through `:1126`, and `:1522` through `:1566`.
- Dense I8 MFMA generated paths call `exec_i32_i8()` without a clamp argument
  at `cdna3/vop3p.cpp:1302` through `:1390`.
- The generic dense I8 helper has no clamp parameter and accumulates through a
  32-bit path at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:3074` through
  `:3098`; separate helpers do have clamp-capable integer primitives at
  `mma_exec.h:1424` through `:1440`, but the generated CDNA3 dense I8 MFMA
  paths do not use them.

Impact:

If the CDNA3 dense MFMA clamp/`FP16_OVFL` row is architecturally meaningful,
rocjitsu currently executes those cases as unclamped, mode-independent MFMA.

### CDNA3-RJ-011: MFMA broadcast field legality is not validated

Manual evidence:

- Section 7.1.5 says `CBSZ` values where `(1 << CBSZ)` exceeds the number of
  blocks are undefined, the largest legal `CBSZ` is 4, and `ABID` is illegal
  when `ABID >= (1 << CBSZ)` at `cdna3/README.md:1922` through `:1936`.

Rocjitsu evidence:

- Generated dense MFMA execution passes raw `inst_.cbsz`, `inst_.abid`, and
  `inst_.blgp` into shared helpers; representative XF32 calls are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3p.cpp:774` through
  `:820`.
- `permute_a_lane()` implements `S = 64 >> cbsz` and
  `(lane % S) + S * abid` without validating `CBSZ` or `ABID` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:568` through
  `:577`.

Impact:

Illegal or undefined MFMA broadcast encodings execute with a computed lane
permutation instead of being rejected or specially classified.

### CDNA3-RJ-012: Dense MFMA tests miss CDNA3-specific semantics

Rocjitsu evidence:

- The gfx942 CTS skip list excludes CDNA3 MFMA fpsan coverage, including F16,
  BF16, FP8/BF8, XF32, and F32 MFMA cases at
  `emulation/rocjitsu/tests/corpus/gfx942_skip_tests.json:13` through `:25`.
- The expensive bit-exact MFMA SIMD suite is documented and instantiated as
  CDNA4-only at
  `emulation/rocjitsu/tests/simd_correctness/mfma_simd_exact_test.cpp:4`
  through `:12` and `:26` through `:28`.
- CDNA3 vector instruction execute coverage is broadly excepted from the
  generated harness and delegated to kernel simulation at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/coverage_exceptions_cdna3.txt:7`
  through `:12`.

Impact:

The existing test surface can miss CDNA3-only dense MFMA behavior such as XF32
mantissa reduction, CDNA3 MODE-denorm distinctions, DGEMM exception behavior,
MAI dependency-wait rules, and CDNA3 field legality cases even when generic
CDNA4 MFMA SIMD checks pass.

### CDNA3-RJ-013: MFMA denorm, rounding, and DGEMM exception behavior are not modeled

Manual evidence:

- The dense MFMA rule table gives denorm, clamp, forced-RNE round mode,
  exception, and execution-mask behavior rows at `cdna3/README.md:1628`
  through `:1636`.
- Section 7.3 says ordinary `V_MFMA_F32_*_F32` honors MODE denormal flags,
  XF32 ignores `MODE.denorm`, matrix C input and D output do not flush
  denormals, sub-32-bit floating MFMA ignores MODE denorms, F64 MFMA ignores
  MODE and rounds to nearest even while allowing denorms, and DGEMM matrix
  operations support arithmetic exceptions at `cdna3/README.md:2052` through
  `:2061`.

Rocjitsu evidence:

- The generated dense MFMA paths pass only register bases, constant-accumulator
  state, and broadcast/negation fields into shared helpers; representative
  F32/XF32/F16/BF16/F64 calls are in
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3p.cpp:767` through
  `:820`, `:1082` through `:1126`, `:1522` through `:1566`, and `:1973`
  through `:2024`.
- Shared extractors convert raw F16/BF16 input halves directly to host `float`
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:636`
  through `:650`.
- The shared F32 MFMA helper accumulates with host `float` multiply-add and no
  shader MODE, denorm policy, or exception-state plumbing at `mma_exec.h:1154`
  through `:1192`.
- The shared F64 MFMA helper accumulates with host `double` arithmetic and
  `BLGP` negation bits, but has no DGEMM arithmetic-exception model or shader
  rounding/denorm state at `mma_exec.h:3492` through `:3532`.

Impact:

CDNA3 MFMA execution is currently a functional host-arithmetic model rather
than a model of the manual's per-family floating-point state. This is separate
from `CDNA3-RJ-010`, which records the missing clamp/`FP16_OVFL` behavior.

### CDNA3-RJ-014: MAI dependency-wait rules are not modeled

Manual evidence:

- Section 7.5 says users must insert NOPs or independent VALU instructions for
  specific MAI dependency cases and defines `DLop`, `XDLOP`, `DGEMM`, and
  `PASS` at `cdna3/README.md:2191` through `:2203`.
- Table 37 gives required waits for VALU-to-MFMA reads, DLop forwarding,
  XDL/SMFMA overlap cases, SGEMM overlap cases, DGEMM/F64 cases, and
  `V_CMPX*` execution-mask forwarding at `cdna3/README.md:2205` through
  `:2302`.

Rocjitsu evidence:

- Codegen only marks matrix instructions with a broad `MFMA` flag when the
  instruction name starts with `V_MFMA_` or `V_SMFMAC_` at
  `lib/python/amdisa/codegen/_generator.py:6539` through `:6542`.
- `ComputeUnit::issue_instruction()` executes non-memory instructions directly
  and deletes them after execution; only memory operations are routed through
  the memory wait-counter path at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:411` through `:437`.
- `WaitCounters` tracks memory/export counter families such as VMCNT, LGKMCNT,
  EXPCNT, VSCNT, LOADCNT, STORECNT, DSCNT, KMCNT, TENSORCNT, and ASYNCCNT, but
  has no MAI pass/forwarding wait state at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wait_counters.h:14` through `:64`.

Impact:

Rocjitsu can execute instruction sequences that the CDNA3 manual requires
software to separate with independent instructions. This is a timing/scheduling
model gap rather than a single-instruction arithmetic mismatch.

### CDNA3-RJ-015: SMFMAC ignores sparse index-set selection from `CBSZ`/`ABID`

Manual evidence:

- For 16-bit sparse source data, `CBSZ[1:0] == 0` lets `ABID[1:0]` select one
  of four 8-bit sparse-index sets within the `SRC2` VGPR; if `CBSZ[1:0] != 0`,
  the first set is selected at `cdna3/README.md:2088` through `:2090`.
- For 8-bit sparse source data, `CBSZ[1:0] == 0` lets `ABID[0]` select one of
  two 16-bit sparse-index sets; if `CBSZ[1:0] != 0`, the first set is selected
  at `cdna3/README.md:2092` through `:2094`.
- The sparse restrictions say `CBSZ` and `ABID` only pick the sparse index from
  the VGPR read and do not perform ordinary MFMA A-matrix broadcast at
  `cdna3/README.md:2102`.

Rocjitsu evidence:

- The code generator emits F32-result SMFMAC calls that pass only `dst`, source
  bases, and an `idx` base to the sparse helpers; it does not pass `inst_.cbsz`
  or `inst_.abid` at `lib/python/amdisa/codegen/execute/matrix.py:140`
  through `:179`.
- Representative generated CDNA3 F16/BF16 paths call sparse helpers with only
  `idx` at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3p.cpp:1734`
  through `:1741`, `:1775` through `:1782`, `:1816` through `:1823`, and
  `:1857` through `:1864`.
- Representative generated CDNA3 BF8/FP8 paths use the same index-base-only
  call shape at `cdna3/vop3p.cpp:2418` through `:2426` and `:2586` through
  `:2594`.
- The shared CDNA3 sparse helpers accept only `idx_base` and extract fixed
  low sparse-index fields from the `SRC2` VGPR at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:3696` through
  `:3718`, `:3731` through `:3756`, `:3846` through `:3875`, and `:3888`
  through `:3919`.

Impact:

SMFMAC encodings that should select a nonzero sparse-index set through
`ABID` execute with the helper's fixed/default index extraction. The helper
call shape correctly avoids ordinary dense-MFMA A-lane broadcast for SMFMAC, but
it drops the same fields before their sparse-specific selector meaning can be
applied.

### CDNA3-RJ-016: I32 SMFMAC variants are generated as unimplemented stubs

Manual evidence:

- The CDNA3 sparse matrix table includes `V_SMFMAC_I32_*_I8` variants at
  `cdna3/README.md:2079` through `:2080`.
- The detailed `V_SMFMAC_I32_16X16X64_I8` and
  `V_SMFMAC_I32_32X32X32_I8` definitions describe legal sparse I8-to-I32
  matrix multiply-accumulate instructions at `cdna3/README.md:12025` through
  `:12063`.

Rocjitsu evidence:

- The code generator intentionally emits a stub for non-F32-result SMFMAC
  variants at `lib/python/amdisa/codegen/execute/matrix.py:181` through `:186`.
- Generated `VSmfmacI3216x16x64I8Vop3pMfma::execute_impl()` throws
  `UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3p.cpp:1898` through
  `:1901`.
- Generated `VSmfmacI3232x32x32I8Vop3pMfma::execute_impl()` does the same at
  `cdna3/vop3p.cpp:1935` through `:1938`.

Impact:

Legal CDNA3 sparse I8 matrix instructions decode and construct, but they throw
instead of executing.

### CDNA3-RJ-017: SMFMAC index-pair legality and VGPR alignment are not validated

Manual evidence:

- Each sparse index pair must satisfy `index0 < index1` and `index0 != index1`
  at `cdna3/README.md:2100`.
- `SRC0`, `SRC1`, and `VDST` VGPR addresses must be even-aligned at
  `cdna3/README.md:2101`.

Rocjitsu evidence:

- Shared SMFMAC helpers split each sparse-index nibble into two selectors and
  use both without checking ordering or distinctness at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:3713` through
  `:3718`, `:3751` through `:3756`, `:3868` through `:3875`, and `:3912`
  through `:3919`.
- Generated SMFMAC execution maps `SRC0`, `SRC1`, `VDST`, and `SRC2` through
  `src_base()` / `dst_base()` without sparse-specific validation; representative
  paths are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3p.cpp:1734` through
  `:1741` and `:2418` through `:2426`.
- `Isa::resolved_vgpr_offset()` accepts any encoded VGPR index and returns the
  unadjusted VGPR offset at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/operand.cpp:1305` through
  `:1311`.

Impact:

Invalid sparse-index pairs execute as ordinary selector pairs, and under-aligned
SMFMAC source or destination bases are accepted. The index-pair rule is unique
to SMFMAC; the alignment issue is the sparse instance of the broader dense-MFMA
register-block validation gap.

### CDNA3-RJ-018: Sparse SMFMAC tests miss selector and I8 execution contracts

Rocjitsu evidence:

- The gfx942 CTS skip list excludes the CDNA3 sparse MFMA fpsan test at
  `emulation/rocjitsu/tests/corpus/gfx942_skip_tests.json:5` through `:7`.
- The generated-code tests only assert that CDNA3 FP8/BF8 SMFMAC generation
  uses FNUZ readers at
  `lib/python/amdisa/tests/test_generator_profile_gates.py:970` through
  `:975` and `:1678` through `:1679`.
- Searching `emulation/rocjitsu/tests/simd_correctness` for `V_SMFMAC` finds no
  sparse CDNA3 SMFMAC execution cases; the sparse exact SIMD tests in that
  directory cover RDNA4 `SWMMAC`, not CDNA3 `SMFMAC`.

Impact:

Current local tests can miss missing `CBSZ`/`ABID` sparse-index selection,
invalid index-pair handling, and the I8 SMFMAC stubs even when helper-selection
or unrelated sparse-WMMA tests pass.

### CDNA3-RJ-019: Generated CDNA3 DBT legalization rows duplicate sparse SMFMAC keys

Rocjitsu evidence:

- The generated legalization lookup key compares only `src_encoding_id` and
  `src_opcode`; `lookup()` uses `std::lower_bound()` and returns the first entry
  whose key matches at
  `lib/rocjitsu/src/rocjitsu/code/dbt/generated/legalization_types.h:23`
  through `:49`.
- CDNA3 VOP3P-MFMA opcode table entries identify sparse SMFMAC opcodes 98,
  100, 102, 104, 106, 108, and 120 through 127 at
  `cdna3/README.md:24059` through `:24082`.
- The generated CDNA3-to-CDNA4 table contains duplicate rows for those sparse
  opcodes under encoding id 423, with `Action::Identity` preceding
  `Action::Lower`, for example opcodes 98, 100, 102, 104, 106, 108, and
  120-127 at
  `lib/rocjitsu/src/rocjitsu/code/dbt/generated/legalization_cdna3_to_cdna4.h:14111`
  through `:14138`.
- The generated CDNA3-to-RDNA3 and CDNA3-to-RDNA4 tables contain duplicate rows
  for the same sparse opcode keys, usually with `Action::Lower` preceding
  `Action::Expand`, at
  `lib/rocjitsu/src/rocjitsu/code/dbt/generated/legalization_cdna3_to_rdna3.h:14111`
  through `:14138` and
  `lib/rocjitsu/src/rocjitsu/code/dbt/generated/legalization_cdna3_to_rdna4.h:14111`
  through `:14138`.
- `select_legalization()` currently wires only CDNA4 guest legalization tables,
  so these CDNA3 guest tables are not active in the current DBT dispatch path at
  `lib/rocjitsu/src/rocjitsu/code/dbt/binary_translator.cpp:53` through `:69`.

Impact:

If CDNA3 guest DBT legalization is enabled or used by tooling, the second row
for a duplicate sparse SMFMAC key is shadowed by the first matching row. For
CDNA3-to-CDNA4 that can turn a sparse opcode that also has a `Lower` entry into
an apparent identity mapping; for CDNA3-to-RDNA targets it can similarly hide
the alternate expansion/lowering action. Today this is generated-table
integrity debt rather than an active binary-translator runtime path.

### CDNA3-RJ-020: CDNA3 SMEM address calculation misses selector and alignment rules

Manual evidence:

- Chapter 8.1.1 gives scalar/global `S_LOAD`, `S_STORE`, and
  `S_DCACHE_DISCARD` addressing as `SBASE` plus an instruction offset plus M0,
  an SGPR offset, or zero, depending on `IMM` and `SOE`, at
  `cdna3/README.md:2348` through `:2374`.
- The same section says scratch SMEM uses the selected scalar offset multiplied
  by 64, and that all address components are byte quantities whose two low bits
  are ignored or forced to zero, at `cdna3/README.md:2375` through `:2394`.

Rocjitsu evidence:

- CDNA3 delegates SMEM address calculation to the shared helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/addr_calc.cpp:1` through
  `:18`.
- The shared helper only adds `SOFFSET` when `SOFFSET_EN` is set and only adds
  `OFFSET` when `IMM` is set at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:25`
  through `:48`.
- That path does not implement the `IMM=0, SOE=0` `OFFSET[6:0]` selector, does
  not treat offset value 124 as M0, does not mask the two low address bits, and
  is also used by generated CDNA3 scratch loads/stores.

Impact:

Legal CDNA3 SMEM encodings using the non-SOE offset selector, M0, unaligned
byte addresses, or scratch 64-byte offset units will execute against the wrong
address in rocjitsu.

### CDNA3-RJ-021: `S_BUFFER_*` SMEM ignores buffer resource descriptor and bounds semantics

Manual evidence:

- Chapter 8.1.1 says scalar buffer memory uses a four-SGPR resource descriptor
  containing base address, stride, `num_records`, and `NV`; the stride is used
  only for bounds checking and not for address calculation at
  `cdna3/README.md:2395` through `:2407`.
- Chapter 8.4 says an even `SBASE` is required for buffer loads and that
  out-of-bounds dwords are clamped by not performing those memory operations at
  `cdna3/README.md:2442` through `:2456`.

Rocjitsu evidence:

- Generated CDNA3 scalar buffer loads and stores construct a 128-bit `SBASE`
  operand, but still call `smem_calculate_address()` like raw-pointer scalar
  memory operations in
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/smem.cpp`.
- The shared scalar address helper reads only an SGPR pair as a 64-bit base and
  returns base plus offset at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:25`
  through `:48`.
- `ScalarMemState` carries only an address, width, data buffer, memory type,
  wait-counter type, and load/store flag, with no buffer descriptor or
  per-dword bounds state at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:71` through `:84`.

Impact:

CDNA3 `S_BUFFER_*` instructions use raw-pointer-style addressing instead of the
manual's scalar-buffer descriptor semantics, and rocjitsu cannot suppress only
the out-of-bounds dwords of a buffer scalar-memory access.

### CDNA3-RJ-022: SMEM dependency counter, clause, and legality rules are not modeled

Manual evidence:

- Section 8.2 says scalar memory reads and writes can return out of order, can
  return partial results, and increment `LGKMCNT` by one for one dword or by two
  for two or more dwords at `cdna3/README.md:2432` through `:2440`.
- Sections 8.1.1, 8.3, and 8.4 describe source-overwrite hazards, scalar-memory
  clauses, SDATA/SBASE alignment restrictions, and out-of-range source/dest
  behavior at `cdna3/README.md:2383` through `:2456`.

Rocjitsu evidence:

- The scalar memory pipeline increments exactly one wait-counter slot per issued
  memory instruction and completes by writing the full data payload at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/memory_pipeline.h:60`
  through `:78` and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/memory_pipeline.cpp:204`
  through `:228`.
- Generated CDNA3 SMEM constructors and execution paths do not validate the
  source-overwrite, clause, SDATA alignment, SBASE alignment, or out-of-range
  source/destination rules.

Impact:

Multi-dword SMEM instructions undercount `LGKMCNT`, and rocjitsu accepts or
executes scalar-memory sequences and operands that the manual marks as
restricted or undefined.

### CDNA3-RJ-023: Scalar atomics decode but do not execute

Manual evidence:

- Section 8.1.2 says scalar atomics support the same operations as vector memory
  atomics, use the same address calculations, and return the pre-atomic value
  when `GLC` is set at `cdna3/README.md:2414` through `:2416`.

Rocjitsu evidence:

- Generated CDNA3 scalar atomic classes construct operands, but their
  `execute_impl()` bodies throw `UnimplementedInst` in
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/smem.cpp`.
- Python semantic derivation classifies SMEM atomics as `nop` at
  `lib/python/amdisa/semantics.py:1687` through `:1725`.

Impact:

Legal CDNA3 scalar atomic instructions cannot execute in rocjitsu, and Python
tooling can silently classify those instructions as no-ops rather than
unsupported read-modify-write operations.

### CDNA3-RJ-024: SMEM `GLC`/`NV` cache policy and discard are incomplete

Manual evidence:

- Chapter 8.1 describes `GLC` load/store/atomic policy and `NV` at
  `cdna3/README.md:2325` through `:2342`.
- The cache-policy table says `GLC=0` and `GLC=1` change read, write, and
  atomic persistence/return behavior at `cdna3/README.md:2795` through `:2816`.
- Section 8.1.3 defines scalar cache invalidate/write-back operations, and
  section 8.1.4 defines `S_MEMTIME`, `S_MEMREALTIME`, and discard operations at
  `cdna3/README.md:2418` through `:2428`.

Rocjitsu evidence:

- Generated CDNA3 SMEM load/store execution sets `ScalarMemState::mtype` from
  `GLC`, but `ScalarMemPipeline::initiate_access()` calls scalar L1 load/store
  helpers without passing that instruction memory type.
- The scalar L1 cache derives memory type from the page-table entry rather than
  from the SMEM instruction flags in
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/l1_scalar_cache.cpp`.
- Generated CDNA3 `S_DCACHE_DISCARD` and `S_DCACHE_DISCARD_X2` execute paths
  throw `UnimplementedInst`, and `NV` is not consumed by the scalar-memory
  pipeline.

Impact:

Scalar-memory `GLC`/`NV` policy bits are decoded but mostly inert for execution,
and legal discard operations remain unimplemented.

### CDNA3-RJ-025: `S_MEMTIME` and `S_MEMREALTIME` use indistinguishable placeholder counters

Manual evidence:

- Section 8.1.4 describes `S_MEMTIME` as a scalar-memory timer query and
  `S_MEMREALTIME` as reading the 100 MHz real-time clock at
  `cdna3/README.md:2422` through `:2428`.

Rocjitsu evidence:

- Shared execution helpers for `S_MEMTIME` and `S_MEMREALTIME` each use a
  separate `thread_local` counter incremented by 100 per call at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1934`
  through `:1951`.

Impact:

Both timer instructions currently provide deterministic placeholder values, but
rocjitsu does not model the architectural distinction between the memory timer
and the 100 MHz real-time clock.

### CDNA3-RJ-026: Buffer descriptor addressing and range checking are incomplete

Manual evidence:

- Section 9.1.5 defines buffer addresses from resource base, SGPR offset, VGPR
  offset/index, stride, element size, `ADD_TID`, swizzle state, and `NumRecords`
  at `cdna3/README.md:2606` through `:2648`.
- Section 9.1.5.1 defines private, raw, and structured range-checking modes and
  the `dst_sel = SEL_1` OOB read exception at `cdna3/README.md:2650` through
  `:2673`.
- Section 9.1.8 defines the full 128-bit descriptor layout and says an all-zero
  resource acts as an unbound buffer returning zero and dropping writes at
  `cdna3/README.md:2735` through `:2765`.

Rocjitsu evidence:

- `mubuf_calculate_addresses()` reads base, 14-bit stride, `NumRecords`, and
  a single `oob_raw` bit from the descriptor at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h:57`
  through `:97`.
- Its range checks use `oob_raw`, `stride`, `index`, and `offset_part`, but do
  not model `ADD_TID`, swizzle enable, 18-bit stride extension, private-scratch
  no-range-check mode, descriptor type/user-VM/NV/reserved bits, unbound
  all-zero behavior, `dst_sel = SEL_1`, or all-or-nothing versus per-component
  distinctions at `addr_calc_buffer.h:118` through `:133`.
- `mtbuf_calculate_addresses()` follows the same reduced descriptor model at
  `addr_calc_buffer.h:169` through `:216`.

Impact:

Basic linear buffer cases can execute, but descriptor-driven scratch/private,
swizzled, unbound, and several OOB/read-channel cases can diverge from CDNA3
hardware semantics.

### CDNA3-RJ-027: Buffer format conversion and `dst_sel` semantics are missing

Manual evidence:

- Section 9.1.3 describes read/write data VGPR counts and buffer data-format
  conversion at `cdna3/README.md:2568` through `:2574`.
- Section 9.1.4 says MTBUF takes format from the instruction, formatted MUBUF
  takes format and `dst_sel` from the resource, raw MUBUF derives size/type
  from the opcode, INVALID resource format remains unbound, and D16 variants
  pack/load/store 16-bit values at `cdna3/README.md:2576` through `:2605`.
- Section 9.1.11 begins the buffer data-format enum table at
  `cdna3/README.md:2883` through `:2910`.

Rocjitsu evidence:

- Generated CDNA3 MUBUF formatted load/store bodies throw `UnimplementedInst`
  for representative non-D16 formatted operations at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/mubuf.cpp:38` through
  `:140` and for representative D16 formatted operations at `:1516` through
  `:1565`.
- Generated CDNA3 MTBUF formatted loads and stores use fixed 4-byte element
  sizes and raw VGPR payload transfers rather than `DFMT`/`NFMT` conversion;
  representative load/store bodies are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/mtbuf.cpp:59` through
  `:70` and `:199` through `:219`.
- Implemented CDNA3 `TBUFFER_STORE_FORMAT_D16_XY`, `_XYZ`, and `_XYZW` bodies
  repeatedly read the low half of `VDATA` for every component at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/mtbuf.cpp:575` through
  `:691`, while Chapter 12.14 stores X/Y/Z/W from consecutive 16-bit component
  halves at `cdna3/README.md:20985` through `:21030`.
- Python semantic derivation maps MTBUF format mnemonics to fixed element
  counts and classifies unmatched MUBUF format mnemonics as `nop` at
  `lib/python/amdisa/semantics.py:2029` through `:2126`.

Impact:

Rocjitsu does not yet model CDNA3 typed-buffer conversion, resource-derived
formatted MUBUF conversion, destination-channel selection, INVALID/unbound
format behavior, or D16 formatted packing.

### CDNA3-RJ-131: MTBUF D16 formatted loads drop packed components

Manual evidence:

- Section 9.1.4 says D16 buffer loads convert returned data to 16 bits and pack
  pairs of data into each 32-bit VGPR, LSBs first and then MSBs, at
  `cdna3/README.md:2600` through `:2605`.
- Chapter 12.14 spells out that `TBUFFER_LOAD_FORMAT_D16_XY` writes X into
  `VDATA[15:0]` and Y into `VDATA[31:16]`, `TBUFFER_LOAD_FORMAT_D16_XYZ`
  writes Z into `VDATA[47:32]` while preserving `VDATA[63:48]`, and
  `TBUFFER_LOAD_FORMAT_D16_XYZW` writes all four 16-bit components at
  `cdna3/README.md:20946` through `:20983`.

Rocjitsu evidence:

- Generated CDNA3 `TBUFFER_LOAD_FORMAT_D16_XY`, `_XYZ`, and `_XYZW` bodies set
  `elem_size = 2`, `num_elems = 2`, `3`, or `4`, and a single `d16_lo = true`
  flag before issuing the vector-memory operation at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/mtbuf.cpp:423` through
  `:506`.
- The shared vector-memory completion path computes `vgpr_count` as
  `std::max(1u, total_bytes / 4)` and, for non-atomic sub-dword loads, copies
  only `elem_size` bytes at offsets `i * 4` before merging them as either the
  low or high half at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:131` through `:195`.
  For D16 XY only X is written; for XYZ only X is written; for XYZW, X and Z are
  written into the low halves of successive VGPRs while Y and W are skipped.
- The MTBUF semantic derivation only records `(elem_size, num_elems, d16_lo)`;
  it has no semantic shape for packing successive 16-bit components into low and
  high halves of consecutive VGPRs at `lib/python/amdisa/semantics.py:2094`
  through `:2125`.

Impact:

End-to-end CDNA3 typed D16 buffer loads with two, three, or four components can
return only a subset of the components and place them in the wrong halves of the
destination VGPRs, even when address calculation and memory bytes are otherwise
valid. `CDNA3-RJ-027` covers the broader missing format conversion and D16 store
packing; this entry covers the separate implemented-load packing bug.

### CDNA3-RJ-028: Buffer `SOFFSET` and dword-alignment edge cases are not modeled

Manual evidence:

- The MUBUF/MTBUF field table says `SOFFSET` must be an SGPR, M0, or inline
  constant at `cdna3/README.md:2506` through `:2513`.
- Section 9.1.5 says the SGPR offset can come from an SGPR or M0 at
  `cdna3/README.md:2632` through `:2638`.
- Section 9.1.7 says dword-or-larger reads and writes ignore the two address
  LSBs, forcing dword alignment, at `cdna3/README.md:2731` through `:2733`.

Rocjitsu evidence:

- The MUBUF helper treats `SOFFSET == 0x80` as inline constant zero and all
  other values as SGPR indices at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h:63`
  through `:67`; MTBUF does the same at `:175` through `:178`.
- The final MUBUF and MTBUF addresses are written directly from
  `base_addr + total_offset` at `addr_calc_buffer.h:133` and `:216`; there is
  no element-size-dependent clearing of low address bits.

Impact:

M0 `SOFFSET`, nonzero inline constants, and misaligned dword-or-larger buffer
addresses can execute differently from the manual.

### CDNA3-RJ-029: Buffer-to-LDS subset, M0 offset, and clamping are incomplete

Manual evidence:

- Section 9.1.9 says load-to-LDS is supported only for
  `BUFFER_LOAD_{ubyte,sbyte,ushort,sshort,dword,format_x}`, defines
  `LDS_offset = M0[15:0]`, and requires active-mask clamping so return data is
  not written outside the LDS allocation for the wave at
  `cdna3/README.md:2767` through `:2790`.

Rocjitsu evidence:

- Allowed raw byte/short/dword loads do implement an `inst_.lds` path, but use
  `wf.m0() + wf.lds_base()` as the base at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/mubuf.cpp:476` through
  `:687`.
- The same `inst_.lds` pattern is also generated for `buffer_load_dwordx2`,
  `dwordx3`, and `dwordx4` at `mubuf.cpp:723` through `:846`, and for raw D16
  loads at `:1236` through `:1514`, even though those forms are outside the
  manual's listed LDS subset.
- The vector-memory completion path writes LDS-destination loads at
  `lds_base + lane * per_lane_bytes` when no per-lane LDS address is present at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:60` through `:72`.
  The LDS backing drops writes beyond total LDS size at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/lds.h:102` through `:106`, but the
  buffer path does not derive the manual's allocation-aware active mask before
  the memory read.

Impact:

Rocjitsu can accept LDS forms the manual does not list, use high bits of M0 in
the LDS offset, and issue reads for lanes whose return data should be masked by
LDS allocation clamping.

### CDNA3-RJ-030: Buffer cache-control and cache-maintenance policies are coarse

Manual evidence:

- Section 9.1.10 gives detailed vector-memory load, store, atomic, invalidate,
  and writeback cache-policy tables for `SC[1:0]` and `NT`, including
  `TG_SPLIT` behavior and SC-dependent `BUFFER_WBL2`/`BUFFER_INV` effects, at
  `cdna3/README.md:2791` through `:2882`.

Rocjitsu evidence:

- `mtype_from_flags_gfx940()` collapses `SC0`, `SC1`, and `NT` into a coarse
  `Mtype` value at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/gfx940_cache_flags.h:29`
  through `:50`.
- Shared `BUFFER_INV`/`BUFFER_WBL2` helpers invalidate or flush broad cache
  levels without consulting the SC table or `TG_SPLIT` refinements at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:29`
  through `:56`, and generated CDNA3 `BUFFER_WBL2`/`BUFFER_INV` dispatches
  simply call those helpers at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/mubuf.cpp:1568` through
  `:1588`.

Impact:

The current model has useful broad cache-behavior hooks, but not the
instruction- and scope-specific CDNA3 policy required for precise cache-control
validation.

### CDNA3-RJ-031: Vector-buffer tests miss descriptor and format edge cases

Rocjitsu evidence:

- `tests/buffer_operand_test.cpp:33` through `:139` covers dynamic VADDR width,
  SRSRC scaling, and CDNA ACC bank folding for MUBUF/MTBUF operands.
- `tests/decode_smoke_test.cpp:725` through `:781` covers the MUBUF `lds`
  modifier in disassembly.
- `lib/python/amdisa/tests/test_semantic_operand_codegen.py:9` through `:47`
  covers generator helpers for legacy buffer VADDR width and SRSRC scaling, and
  `lib/python/amdisa/tests/test_sema_derive.py:1644` through `:1657` covers a
  derived `BUFFER_LOAD_B32` semantic shape.

Impact:

The existing tests cover important operand/codegen regressions, but not the
manual-derived descriptor modes, format conversion, M0/inline `SOFFSET`,
alignment, LDS clamping/subset, unbound resource, or cache-policy cases
identified in the vector-buffer audit slice.

### CDNA3-RJ-032: FLAT/GLOBAL/SCRATCH address-mode and fault behavior is incomplete

Manual evidence:

- Chapter 10 says flat memory uses aperture registers to distinguish
  video/system, LDS, and scratch spaces and that unmapped regions generate a
  memory violation at `cdna3/README.md:3019` through `:3021`.
- Section 10.3 says FLAT supports 32-bit and 64-bit addressing according to the
  `PTR32` mode register, and sections 10.4 and 10.5 say global/scratch address
  component size depends on `ADDRESS_MODE` at `cdna3/README.md:3121` through
  `:3142`, `:3154` through `:3167`, and `:3169` through `:3184`.

Rocjitsu evidence:

- Generated CDNA3 FLAT constructors fix the ordinary flat address operand as a
  64-bit VGPR pair, narrow scratch to a 32-bit VGPR offset, and narrow global
  to a 32-bit VGPR offset only when `saddr != 0x7F`; representative code is in
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/flat.cpp:27` through `:53`.
- The shared helper uses fixed segment formulas: FLAT always reads a 64-bit VGPR
  pair, GLOBAL chooses either 64-bit VGPR pair or 64-bit SGPR base plus
  sign-extended 32-bit VGPR offset, and SCRATCH uses scratch base plus lane
  stride plus optional VGPR/SADDR offsets at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h:46`
  through `:132`.
- The helper maps private-aperture FLAT addresses into scratch backing memory
  and the compute unit maps shared-aperture addresses into LDS, but there is no
  visible `PTR32`/`ADDRESS_MODE` state or memory-violation path for aperture
  holes at `addr_calc_flat.h:113` through `:132` and
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:263` through `:281`.

Impact:

Rocjitsu has useful segment and aperture routing, but it cannot model
mode-dependent address-size behavior or flat aperture fault policy from the
CDNA3 manual.

### CDNA3-RJ-033: Global/scratch direct-to-LDS flat-memory forms are not modeled

Manual evidence:

- Sections 10.4 and 10.5 say GLOBAL and SCRATCH instructions can transfer data
  directly between LDS and memory at `cdna3/README.md:3154` through `:3167` and
  `:3169` through `:3184`.
- Section 10.3 gives LDS destination formulas using the hardware LDS base,
  `M0[17:2] * 4`, instruction offset, and `ThreadID` scaling at
  `cdna3/README.md:3137` through `:3142`.

Rocjitsu evidence:

- The generated flat-load body only sets a VGPR destination and issues a
  `VectorMemState(GLOBAL_MEM)` through `flat_calculate_addresses()`; it never
  sets `lds_dst`, `lds_base`, or per-lane LDS addresses at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/flat.cpp:55` through `:65`
  and the generator template at
  `lib/python/amdisa/codegen/_generator.py:4620` through `:4648`.
- A targeted search of CDNA3 flat generated code found no `LOAD_LDS`,
  `lds_dst`, or `wf.m0()` handling in the flat instruction family.
- The CDNA3 subdecode table marks FLAT opcodes 38 through 42 invalid at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/decoder.cpp:8835` through
  `:8839`, even though the manual and XML reserve those opcode numbers for
  `GLOBAL_LOAD_LDS_*` and `SCRATCH_LOAD_LDS_*`.
- The profile generator skips segment-specific flat encodings, including
  `ENC_FLAT_GLBL` and `ENC_FLAT_SCRATCH`, at
  `lib/python/amdisa/isa_profile.py:758` through `:769`. The generic FLAT
  class sharing therefore drops segment-only opcodes 38 through 42.
- Semantic derivation strips `GLOBAL_LOAD_`/`SCRATCH_LOAD_` and looks up the
  remaining suffix in `_FLAT_DATA_MAP` at
  `lib/python/amdisa/semantics.py:1990` through `:2022`; suffixes such as
  `LDS_DWORD` miss the data map and fall through as non-modeled rows.
- The generated `FlatMachineInst` exposes bit 13 as `lds`, while the
  specialized global/scratch structs expose it as `sve`, and the builder calls
  the bit `lds`; this preserves the bit but does not create the direct-LDS
  destination path at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/machine_insts.h:104`
  through `:139` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/builders.h:372`
  through `:395`.

Impact:

Ordinary VGPR-return global/scratch memory can execute, but the manual's
GLOBAL/SCRATCH-to-LDS forms lack the destination and `M0` address semantics
needed for end-to-end execution.

### CDNA3-RJ-034: FLAT wait-counter and ordering contract is simplified

Manual evidence:

- Section 10.2 says FLAT instructions execute internally as both LDS and Buffer,
  increment both `VM_CNT` and `LGKM_CNT`, and complete only when both counters
  decrement at `cdna3/README.md:3105` through `:3111`.
- Sections 10.2.1 and 10.2.2 describe out-of-order completion, same-VGPR return
  hazards, and the `S_WAITCNT 0` restriction at `cdna3/README.md:3113` through
  `:3119`.

Rocjitsu evidence:

- Generated CDNA3 flat loads, stores, and atomics set
  `wait_counter_type = WaitCounterType::VMCNT` before issuing; representative
  bodies are at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/flat.cpp:55`
  through `:65`, `:385` through `:393`, and `:1059` through `:1069`.
- The compute-unit router changes shared-aperture FLAT operations to the local
  pipeline and `LGKMCNT`, but this is an either/or counter choice rather than a
  dual VM+LGKM issue/retire model at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:263` through `:281`.

Impact:

The current model tracks flat memory enough for functional completion, but not
the dual-counter and ordering behavior needed for wait-counter exactness.

### CDNA3-RJ-035: Floating flat atomics miss packed and FP-mode special cases

Manual evidence:

- Section 10.3.1 says floating-point atomics must use `SC[0]=0`, FP32 atomics
  flush denormals to zero, FP64 and FP16 atomics do not flush denormals, and
  rounding is fixed RNE at `cdna3/README.md:3146` through `:3152`.
- The FLAT instruction definitions include `FLAT_ATOMIC_ADD_F32`,
  `FLAT_ATOMIC_PK_ADD_F16`, `FLAT_ATOMIC_ADD_F64`, `FLAT_ATOMIC_MIN_F64`,
  `FLAT_ATOMIC_MAX_F64`, and `FLAT_ATOMIC_PK_ADD_BF16` at
  `cdna3/README.md:21441` through `:21527`.

Rocjitsu evidence:

- Generated floating flat atomics use `d->is_load = (inst_.sc0 != 0)`, so
  return-data mode is still allowed for FP atomics; representative F32/F64
  bodies are at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/flat.cpp:1815`
  through `:1825` and `:1931` through `:1941`.
- `FLAT_ATOMIC_PK_ADD_F16` and `FLAT_ATOMIC_PK_ADD_BF16` both set
  `elem_size = 4` and `atomic_op = AtomicOp::FADD`, the same path used for
  scalar F32 atomics, at `flat.cpp:1873` through `:1883` and `:2111` through
  `:2121`.
- The memory pipeline treats 4-byte floating atomics as a single `float` and
  8-byte floating atomics as a single `double`; `apply_fp_atomic()` uses
  ordinary `+`, `std::fmin`, and `std::fmax` without FP32 denormal flushing or
  packed F16/BF16 lane handling at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:285` through `:364`.

Impact:

F32/F64 atomic add/min/max have a coarse functional model, but packed F16/BF16
atomics are not type-correct and the CDNA3 FP atomic `SC0`, denormal, and
rounding rules are not enforced.

### CDNA3-RJ-036: Flat/global/scratch tests miss CDNA3-specific edge cases

Rocjitsu evidence:

- Shared address tests cover several scratch/global formulas through hand-built
  machine instructions, including CDNA4 and RDNA3 scratch/global cases at
  `tests/shared_infra_test.cpp:2780` through `:3140`.
- Existing decode smoke coverage exercises ACC on global/scratch forms and
  adjacent memory tests exercise generic flat/global behavior, but the static
  pass did not find CDNA3 end-to-end cases for `PTR32`/`ADDRESS_MODE`,
  direct-to-LDS global/scratch forms, scratch atomic decode rejection, FLAT
  dual-counter behavior, aperture-hole faults, no-return atomic def-use
  metadata, or packed FP atomic execution.

Impact:

The tests cover useful address-helper regressions, but the manual-derived CDNA3
contracts identified in this slice can regress without focused coverage.

### CDNA3-RJ-037: LDS M0 clamping and bank-conflict behavior are not modeled

Manual evidence:

- Section 11.1 describes LDS as 32 banks and says bank conflicts are serialized
  for indexed and atomic operations at `cdna3/README.md:3212` through `:3222`.
- Section 11.3.1 says all LDS operations require `M0` initialization, and that
  `M0[16:0]` contains the LDS segment byte-size and clamps the final address at
  `cdna3/README.md:3240` through `:3265`.

Rocjitsu evidence:

- CDNA3 DS address calculation delegates to the shared DS helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/addr_calc.cpp:36` through
  `:38`.
- The shared helper computes `VGPR[ADDR] + concatenated offset + lds_base` and
  never reads `wf.m0()` for normal DS operations at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:53`
  through `:76`.
- Representative DS2 bodies compute the two addresses directly from `ADDR`,
  scaled offsets, and `wf.lds_base()` without M0 clamping at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/ds.cpp:2177` through
  `:2200`.
- The local-memory pipeline performs functional vector loads/stores and atomics
  at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:506` through
  `:596`, with no LDS bank or conflict-timing model.

Impact:

Out-of-range LDS accesses can execute without the manual's `M0[16:0]` clamp,
and bank-conflict behavior is not visible to timing or hazard-sensitive tests.

### CDNA3-RJ-038: DS READ2/WRITE2 duplicate-offset collapse is not modeled

Manual evidence:

- Section 11.3.1 says a two-address operation can specify only one address by
  setting both offsets to the same value; this causes only one read/write and
  uses only the first `DATA0` field at `cdna3/README.md:3300` through `:3302`.

Rocjitsu evidence:

- `DsRead2B32Ds::execute_impl()` always sets `ds2_active = true`, computes both
  addresses from `offset0` and `offset1`, and sets a second destination register
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/ds.cpp:2177` through
  `:2200`.
- The local-memory pipeline always issues the second DS2 load when
  `ds2_active` is true and writes the second response during completion at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:520` through `:528`
  and `:606` through `:637`.
- Generated WRITE2 bodies follow the same unconditional second-address pattern;
  representative B32 code computes both addresses and fills `ds2_store_data` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/ds.cpp:580` through `:632`.

Impact:

Equal-offset READ2/WRITE2 encodings perform two modeled accesses in rocjitsu
instead of the manual's one-access form. Stores can use `DATA1` where hardware
should ignore it, and reads can write a second destination value where only one
access should occur.

### CDNA3-RJ-039: No-return DS atomics can clobber `VDST`

Manual evidence:

- Section 11.3.1 distinguishes LDS atomic updates from the optional return of
  the pre-operation value to VGPRs at `cdna3/README.md:3238` through `:3261`.

Rocjitsu evidence:

- The no-return `ds_add_u32` constructor has `num_dst_ = 0`, but its execute
  path still sets `d->dst_reg_base = ... + inst_.vdst`, `d->is_load = true`,
  and `d->atomic_op = AtomicOp::ADD` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/ds.cpp:28` through `:65`.
- `MemoryPipeline::issue()` always calls `complete_access()` after initiating a
  memory operation at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.h:62`
  through `:88`.
- Local memory atomics execute `execute_lds_atomic_rmw()` and then call
  `vector_complete()` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:515` through `:517`
  and `:599` through `:605`.
- `vector_complete()` writes response data to `dst_reg_base` whenever
  `d.is_load` is true, regardless of the instruction's `num_dst_`, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:119` through `:198`.

Impact:

No-return DS atomics can write the returned old LDS value into the encoded
`VDST` register even though the instruction has no architectural destination.
This can silently corrupt a VGPR chosen as the unused `VDST` field.

### CDNA3-RJ-040: Packed F16/BF16 LDS atomics execute through the scalar F32 atomic path

Manual evidence:

- The DS opcode table includes packed F16/BF16 LDS atomic add forms at
  `cdna3/README.md:24263` through `:24297`.
- Section 11.3.1 says LDS floating atomics follow `MODE.FP_DENORM` denormal
  behavior and fixed round-to-nearest-even rounding at `cdna3/README.md:3314`
  through `:3318`.

Rocjitsu evidence:

- `DS_PK_ADD_F16` and `DS_PK_ADD_BF16` set `elem_size = 4` and
  `atomic_op = AtomicOp::FADD`, the same memory-pipeline operation used for
  scalar F32 atomics, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/ds.cpp:909` through `:980`.
- The return forms `DS_PK_ADD_RTN_F16` and `DS_PK_ADD_RTN_BF16` do the same at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/ds.cpp:4948` through
  `:5037`.
- `execute_lds_atomic_rmw()` treats `elem_size == 4` floating atomics as one
  `float`, and `apply_fp_atomic()` uses ordinary C++ add/min/max without packed
  half or BF16 lane handling, denormal-mode logic, or explicit RNE behavior at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:285` through
  `:297` and `:384` through `:471`.

Impact:

Packed LDS F16/BF16 atomics are type-incorrect: rocjitsu interprets the 32-bit
word as a scalar F32 value instead of two packed 16-bit lanes, and it cannot
model the manual's FP-mode requirements.

### CDNA3-RJ-041: GDS/GWS forms are decoded but not executed

Manual evidence:

- The DS format includes the `GDS` bit, and Chapter 11.4 adds GWS restrictions:
  all GWS instructions must be followed immediately by `s_waitcnt 0`, and VGPRs
  used by GWS instructions must be even at `cdna3/README.md:3320` through
  `:3328`.
- Section 12.12 says GWS instructions operate only on the first active lane in
  the EXEC mask at `cdna3/README.md:18086` through `:18108`.
- The detailed GWS definitions describe resource-id calculation and semaphore
  or barrier state-machine behavior at `cdna3/README.md:19711` through
  `:19832`.

Rocjitsu evidence:

- Generated CDNA3 GWS classes decode `DS_GWS_SEMA_RELEASE_ALL`,
  `DS_GWS_INIT`, `DS_GWS_SEMA_V`, `DS_GWS_SEMA_BR`, `DS_GWS_SEMA_P`, and
  `DS_GWS_BARRIER`, but every execute body throws `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/ds.cpp:4829` through
  `:4904`.
- Generated CDNA3 DS execute bodies throw `util::UnimplementedInst` whenever
  `inst_.gds` is set; representative load, store, atomic, and ADDTID paths are
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/ds.cpp:43` through
  `:65`, `:543` through `:578`, `:2149` through `:2159`, and `:4920` through
  `:4945`.
- Codegen classifies `DS_GWS_*` semantics as `nop` because they are hardware
  scheduling primitives at `lib/python/amdisa/semantics.py:2398` through
  `:2401`.
- The static pass did not find a GDS/GWS execution path or validation for
  first-active-lane execution, resource state, immediate-following
  `s_waitcnt 0`, or even-VGPR restrictions.

Impact:

LDS forms execute functionally, but GDS/GWS encodings cannot execute and their
manual lane, state, sequence, and register constraints are not represented.

### CDNA3-RJ-042: Data-share tests miss CDNA3 LDS edge cases

Rocjitsu evidence:

- Existing VM atomic stress tests include CDNA4 `ds_add_rtn_u32` and a
  no-return `ds_add_u32` memory-result case at `tests/amdgpu_vm_test.cpp:1880`
  through `:1966`, but the no-return test only checks final LDS contents and
  does not assert that the unused `VDST` register remains unchanged.
- Decode smoke coverage exercises a DS ACC destination case at
  `tests/decode_smoke_test.cpp:783` through `:794`, and the HIP ACC transpose
  test checks one DS transpose/ACC behavior at `tests/hip_ds_transpose_acc_test.cpp:59`
  through `:87`.
- The static pass did not find focused CDNA3 tests for `M0[16:0]` LDS clamping,
  READ2/WRITE2 equal-offset collapse, packed F16/BF16 LDS atomics, GWS
  `GDS=1` encoding/disassembly, GDS/GWS runtime behavior, or LDS FP atomic MODE
  behavior.

Impact:

The current tests cover useful ordinary DS plumbing, but the manual-derived LDS
contracts identified in this slice can regress without detection.

### CDNA3-RJ-043: Generated GWS encodings and decode metadata miss required `GDS=1`

Manual/oracle evidence:

- Section 12.12 says `GDS` is set for GWS and clear for LDS at
  `cdna3/README.md:18094` through `:18105`.
- LLVM's assembler accepts GWS forms only with the explicit `gds` modifier:
  `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx942 -show-encoding` encodes
  `ds_gws_init v2 gds` as bytes `[0x00,0x00,0x33,0xd9,0x02,0x00,0x00,0x00]`,
  where DS bit 16 is set.
- The same LLVM oracle rejects missing-`gds` assembly as "too few operands" and
  rejects odd data operands such as `v3` as "vgpr must be even aligned".
- LLVM disassembly treats the corresponding `GDS=0` byte sequence
  `0x00 0x00 0x32 0xd9 0x02 0x00 0x00 0x00` as an invalid instruction
  encoding rather than as `ds_gws_init`.

Rocjitsu evidence:

- Generated CDNA3 test encodings list the GWS opcodes with dword0 values
  `0xD9300000`, `0xD9320000`, `0xD9340000`, `0xD9360000`, `0xD9380000`, and
  `0xD93A0000`, all with the `GDS` bit clear, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/test_encodings.h:1204`
  through `:1209`.
- The generated DS decoder table selects GWS decoders by opcode slots 152
  through 157 without a `GDS=1` condition at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/decoder.cpp:8688` through
  `:8693`.
- `DsMachineInst` carries the `gds` bit, but the CDNA3 `Ds` encoding base has
  no `build_modifiers()` override to print a required `gds` modifier at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/machine_insts.h:54` through
  `:64` and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/encodings.h:550` through
  `:555`.

Impact:

Rocjitsu's generated GWS fixture encodings and decode metadata can treat
non-GDS byte patterns as valid GWS instructions, while LLVM and the manual
require the `GDS` bit for these opcodes. Disassembly also lacks the required
`gds` modifier, so textual output can round-trip to a form LLVM rejects.

### CDNA3-RJ-044: SALU allocation-bound fallback and out-of-range destination side effects are not modeled

Manual evidence:

- Section 5.2 says SALU instructions cannot use VGPRs or LDS, source selector
  255 consumes the following 32-bit literal dword for all SALU formats except
  SOPP and SOPK, out-of-range source SGPRs read SGPR0, and out-of-range
  destination SGPRs suppress the SGPR write while still writing SCC and EXEC
  saveexec side effects at `cdna3/README.md:983` through `:989`.

Rocjitsu evidence:

- `Wavefront` tracks a per-dispatch SGPR allocation count separately from the
  physical allocation base at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:87` through `:89` and
  `:40` through `:44`.
- The CDNA3 scalar source resolver reads SGPR selectors by adding the raw
  selector to `wf.sgpr_alloc().base` and has no `wf.num_sgprs()` or
  `sgpr_alloc().count` bound check before reading at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/operand.cpp:1016` through
  `:1028`.
- Unsupported scalar source selectors throw at `operand.cpp:1081` through
  `:1083`, and unsupported scalar destination selectors throw at
  `operand.cpp:1179` through `:1218` and `:1221` through `:1250`.
- Some scalar helpers perform the destination write before the SCC side effect;
  for example `execute_s_add_u32_sop2()` writes `SDST` before `wf.write_scc()`
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:433`
  through `:440`.

Impact:

Rocjitsu can read or write outside the wavefront's declared SGPR allocation, or
throw for raw unsupported destination selectors before completing side effects
that the manual says survive an out-of-range destination. That makes raw-SALU
edge encodings and low-SGPR-count dispatches diverge from the manual's fallback
contract.

### CDNA3-RJ-045: SALU 64-bit SGPR pair alignment is not enforced

Manual evidence:

- Section 5.2 says any instruction using 64-bit data in SGPRs must use an
  even-aligned SGPR pair, giving `s[2:3]` and `s[8:9]` as legal examples and
  `s[11:12]` as illegal at `cdna3/README.md:989`.
- The SOP1, SOP2, and SOPC format tables expose 64-bit scalar operations that
  use the same SSRC/SDST selector space, for example `S_MOV_B64` in the SOP1
  opcode table at `cdna3/README.md:22895` through `:22904`.

Rocjitsu evidence:

- Generated 64-bit scalar instructions keep raw 64-bit `OPR_SDST`/`OPR_SSRC`
  operands; representative `S_MOV_B64` construction is at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sop1.cpp:39` through
  `:50`.
- `resolve_src_scalar64()` reads SGPR selector `ev` and `ev + 1` for ordinary
  scalar and TTMP pairs without checking that `ev` is even at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/operand.cpp:1121` through
  `:1138`.
- `resolve_dst_write64()` writes SGPR selector `ev` and `ev + 1` with the same
  missing even-boundary check at `operand.cpp:1221` through `:1241`.

Impact:

Odd 64-bit SGPR pairs such as `s[11:12]` execute as ordinary consecutive
register pairs in rocjitsu even though the manual declares them illegal.

### CDNA3-RJ-046: SALU implicit SCC operands are not surfaced in C++ def-use metadata

Manual/XML evidence:

- Chapter 5 says many SALU operations set SCC for comparisons, carry-out, or
  zero-result testing at `cdna3/README.md:895`.
- The XML includes implicit SCC output operands on representative writers such
  as `S_ADD_U32` and `S_CMP_EQ_I32` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna3.xml:31954` through
  `:31984` and `:38994` through `:39018`.
- The XML includes an implicit SCC input operand on `S_CBRANCH_SCC0` at
  `amdgpu_isa_cdna3.xml:42150` through `:42168`.

Rocjitsu evidence:

- Execution does read and write SCC directly; representative helpers call
  `wf.write_scc()` and `wf.read_scc()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:433`
  through `:459` and `:1080` through `:1124`.
- Generated CDNA3 constructors omit the XML implicit SCC operands from the
  C++ operand arrays: `S_ADD_U32` exposes only `SDST`, `SSRC0`, and `SSRC1` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sop2.cpp:20` through `:35`;
  `S_CMP_EQ_I32` exposes only its two explicit sources at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopc.cpp:20` through `:28`;
  and `S_CBRANCH_SCC0/SCC1` expose only the label operand at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopp.cpp:66` through
  `:108`.
- `InstDefUse` derives metadata from explicit operands plus
  `implicit_defs()`/`implicit_uses()` at
  `lib/rocjitsu/src/rocjitsu/analysis/def_use_chain.cpp:25` through `:45`,
  but `Instruction`'s default implicit hooks are no-ops at
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:215` through `:222`.
- Even if an implicit SCC `RegisterRef` were produced, `RegisterSet` currently
  documents SCC as untracked and ignores non-SGPR/VGPR/ACC classes at
  `lib/rocjitsu/src/rocjitsu/isa/register_set.h:54` through `:60` and
  `lib/rocjitsu/src/rocjitsu/isa/register_set.cpp:37` through `:51`.

Impact:

Scalar execution has SCC behavior, but C++ def-use and liveness consumers do
not see the XML's implicit SCC reads/writes. Any analysis or patch planning
that depends on decoded `RegisterSet` metadata must handle SCC separately.

### CDNA3-RJ-047: `S_MAX_{I32,U32}` clears SCC for equal operands

Manual evidence:

- CDNA3 detailed `S_MAX_I32` and `S_MAX_U32` definitions set SCC with inclusive
  predicates, `SCC = S0.i32 >= S1.i32` and `SCC = S0.u32 >= S1.u32`, then
  select `D0 = SCC ? S0 : S1` at `cdna3/README.md:3469` through `:3482`.
- CDNA4 and RDNA4 detailed manuals use the same inclusive max predicate at
  `cdna4/README.md:3940` through `:3953` and `rdna4/README.md:8447` through
  `:8460`.

Rocjitsu evidence:

- The shared `S_MAX_I32` and `S_MAX_U32` helpers compute the correct numeric
  max but write SCC with strict `s0 > s1` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1866`
  through `:1871` and `:1893` through `:1898`.
- The semantic-derivation regression explicitly requires strict `>` and rejects
  `>=` at `lib/python/amdisa/tests/test_sema_derive.py:300` through `:305`.
- The cross-architecture scalar SCC runtime test expects `false` for equal
  `s_max_i32` and `s_max_u32` inputs at
  `tests/scalar_scc_test.cpp:733` through `:738`.

Impact:

Equal operands produce the same destination value under either predicate, so
ordinary destination-only tests pass. Any subsequent SCC consumer, such as
`S_CSELECT`, `S_CBRANCH_SCC*`, or a carry-chain probe using SCC state, can
observe the divergence.

### CDNA3-RJ-048: HWREG get/set uses the wrong CDNA3 register map

Manual/XML evidence:

- CDNA3 Table 19 defines HWREG ID 1 as `MODE`, ID 2 as read-only `STATUS`, ID 3
  as `TRAPSTS`, ID 4 as `HW_ID`, ID 5 as packed `GPR_ALLOC`, ID 6 as
  `LDS_ALLOC`, and ID 7 as `IB_STS` at `cdna3/README.md:1121` through `:1140`.
- The CDNA3 XML `OPR_HWREG` table agrees on the low IDs, naming
  `hw_reg_mode` at 1, `hw_reg_status` at 2, `hw_reg_trapsts` at 3,
  `hw_reg_hw_id` at 4, `hw_reg_gpr_alloc` at 5, `hw_reg_lds_alloc` at 6, and
  `hw_reg_ib_sts` at 7 at `amdgpu_isa_cdna3.xml:93610` through `:93659`.

Rocjitsu evidence:

- `SGetregB32Sopk::execute_impl()` reads HWREG ID 1 from `wf.status_raw()`, ID
  4 from the low CU ID, ID 5 from the high CU ID, ID 6 from a two-field SGPR
  allocation value, and ID 7 from a two-field VGPR allocation value at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopk.cpp:291` through
  `:318`.
- `SSetregB32Sopk::execute_impl()` and `SSetregImm32B32Sopk::execute_impl()`
  write only HWREG ID 1 and route it to `wf.status_raw()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopk.cpp:334` through
  `:383`.
- The only adjacent HWREG execution tests use CDNA4 and explicitly describe
  `hwreg=1` as STATUS at `tests/instruction_execution_harness_test.cpp:3595`
  through `:3669`, matching the generated rocjitsu mapping rather than the
  CDNA3/CDNA4 manual map.

Impact:

`S_GETREG_B32` and `S_SETREG*` read or update the wrong wave state for common
CDNA3 selectors. MODE cannot be read or written through HWREG ID 1, STATUS ID 2
is unhandled, `GPR_ALLOC` is not returned in the manual's packed layout, and
`LDS_ALLOC`/`IB_STS` are not modeled by their documented IDs.

### CDNA3-RJ-049: SETREG write permissions, side effects, and spacing are not modeled

Manual evidence:

- Section 5.8 requires an `S_NOP` between consecutive `S_SETREG` writes to the
  same register at `cdna3/README.md:1108` through `:1113`.
- Detailed `S_SETREG_B32` and `S_SETREG_IMM32_B32` pseudocode applies
  `HwRegWriteMask(hwRegId, WAVE_STATUS.PRIV)`, preserves non-writable bits,
  and notes that some bit updates may trigger side effects at
  `cdna3/README.md:4074` through `:4119`.

Rocjitsu evidence:

- The CDNA3 `S_SETREG_B32` and `S_SETREG_IMM32_B32` helpers only build a mask
  from requested `{offset,size}` and directly splice bits into `wf.status_raw()`
  for their one handled register at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopk.cpp:334` through
  `:383`.
- No audited CDNA3 path applies a per-register write mask, checks
  `WAVE_STATUS.PRIV`, models side effects from modified HWREG bits, or tracks a
  dependency/hazard between consecutive `S_SETREG` writes to the same register.

Impact:

Even after the HWREG ID map is corrected, SETREG can still accept writes that
the manual treats as non-writable or privileged, and instruction sequences that
hardware requires software to space with an `S_NOP` execute without any warning
or timing/hazard model.

### CDNA3-RJ-050: `S_SETREG_IMM32_B32` hides its literal operand

Manual/XML evidence:

- The manual defines `S_SETREG_IMM32_B32` as a 64-bit SOPK form whose 32-bit
  data comes from a literal constant at `cdna3/README.md:1108` through `:1113`
  and in the detailed definition at `cdna3/README.md:4133` through `:4150`.
- The CDNA3 XML exposes an explicit `OPR_SIMM32` input operand on
  `S_SETREG_IMM32_B32` at `amdgpu_isa_cdna3.xml:41969` through `:41992`.

Rocjitsu evidence:

- `Sopk::hasImpliedLiteral()` correctly makes opcode 20 consume the extension
  dword and store it in `literal_` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/encodings.cpp:68` through
  `:82`.
- `SSetregImm32B32Sopk::execute_impl()` reads `literal_`, but the CDNA3
  constructor exposes only the HWREG operand as a destination and sets
  `num_src_ = 0` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopk.cpp:356` through
  `:376`.
- Generic operand access and disassembly are driven by `num_src_`,
  `num_dst_`, `src_operands_`, and `dst_operands_` at
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:152` through `:173` and
  `:239` through `:252`.
- The newer gfx1250 generated constructor includes
  `literal(32, OperandType::OPR_SIMM32, 0)`, exposes it as a source, and sets
  `num_src_ = 1` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/gfx1250/sopk.cpp:224` through
  `:233`.

Impact:

Runtime uses the extension word, but CDNA3 disassembly and operand metadata
hide the 32-bit literal. Oracle/disassembly fuzzing and operand consumers can
miss or miscompare the literal even when execution happens to use it.

### CDNA3-RJ-051: Raw STATUS `EXECZ`/`VCCZ` bits can drift from EXEC/VCC

Manual evidence:

- The STATUS table defines `EXECZ` as "Exec mask is zero" and `VCCZ` as
  "Vector condition code is zero" at `cdna3/README.md:434` through `:435`.
- EXEC itself is a 64-bit execution mask, and `EXECZ` is a helper bit usable as a
  branch condition at `cdna3/README.md:408` through `:416`.
- STATUS fields are readable by shader code, initialized at wavefront creation,
  and some are set by shader instructions at `cdna3/README.md:418` through
  `:420`.

Rocjitsu evidence:

- `Wavefront` stores EXEC and VCC as separate `exec_` and `vcc_` members, and
  `set_exec()`/`set_vcc()` update only those members at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:220` through `:238`.
- Raw STATUS is a separate ISA-specific `StatusType status{0}` exposed through
  `status_raw()` and `set_status_raw()` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:608` through `:617`.
- CDNA status bit wrappers define `EXECZ` and `VCCZ` as bits 9 and 10 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/cdna_isa_base.h:55`
  through `:58`, but the EXEC/VCC setters do not update those bits.
- Branch and scalar-source paths compute `VCCZ`/`EXECZ` directly from
  `wf.vcc()` and `wf.exec()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopp.cpp:125` through
  `:152` and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/operand.cpp:1073`
  through `:1079`.
- The generated CDNA3 HWREG path reads raw STATUS with `wf.status_raw()` for
  its handled status-like register at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopk.cpp:291` through
  `:300`.

Impact:

Conditional branches and source operands can see the correct derived EXEC/VCC
zero state while raw STATUS readers see stale bits. Any code path that observes
STATUS rather than recomputing helper values can disagree with the architectural
helper-bit contract.

### CDNA3-RJ-052: `S_SETVSKIP` is decoded but unimplemented

Manual evidence:

- MODE bit 28 is `VSKIP`; when set, vector, VMEM, LDS, and GDS instructions are
  skipped rather than issued at `cdna3/README.md:472`.
- The instruction overview describes `S_SETVSKIP` as setting a bit that causes
  vector instructions to be ignored at `cdna3/README.md:736`.
- The detailed `S_SETVSKIP` definition says it enables or disables VSKIP and
  that VSKIPped memory instructions do not manipulate wait counters at
  `cdna3/README.md:5188` through `:5195`.

Rocjitsu evidence:

- CDNA3 decodes `S_SETVSKIP` and constructs two scalar source operands at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopc.cpp:388` through
  `:404`.
- `SSetvskipSopc::execute_impl()` immediately throws `UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopc.cpp:406` through
  `:409`.
- The CU execution path directly dispatches each decoded instruction through
  `execute_instruction()` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:400` through `:412`,
  and the concrete execute hook simply calls `inst->execute()` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.h:753` through `:756`;
  there is no VSKIP-mode gate in the audited dispatch path.

Impact:

Programs that use `S_SETVSKIP` trap in the emulator instead of toggling
MODE.VSKIP. Even if the bit were set through another path, vector/memory/LDS
instruction issue is not suppressed by the execution dispatcher.

### CDNA3-RJ-053: LDS workgroup allocation uses 256-byte granularity instead of 512-byte blocks

Manual evidence:

- Chapter 3 says LDS is allocated per work-group or wavefront in contiguous
  512-byte blocks on 512-byte alignment, with no wrap, and LDS clamping uses the
  smaller of the SPI allocation size and M0 at `cdna3/README.md:540` through
  `:544`.

Rocjitsu evidence:

- The command-processor accounting helper aligns per-workgroup LDS to 256 bytes
  at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:99` through
  `:101`.
- CU admission checks use `util::align_up(lds_bytes, 256u)` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:243` through `:245`.
- CU-local allocation also advances `next_lds_alloc_` by a 256-byte-aligned
  size at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.h:239` through
  `:245`.
- The WGP placement path uses the same 256-byte alignment at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/spi.h:205` through `:235`, and dispatch
  validation reports the 256-byte-aligned value at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:988` through
  `:998`.

Impact:

Rocjitsu can pack workgroups whose LDS allocations would occupy distinct
512-byte blocks on CDNA3 hardware. That can expose different workgroup
residency, LDS base addresses, and allocation-limit behavior for kernels with
nonzero LDS sizes below or not aligned to 512 bytes.

### CDNA3-RJ-054: Trap and exception state is not modeled

Manual evidence:

- Chapter 3.10 says traps load a hardware-generated `S_TRAP` payload into
  TTMP0/1, obey `STATUS.TRAP_EN`, reserve extra trap-handler SGPRs, and use
  `MODE.EXCP_EN` exception enables at `cdna3/README.md:590` through `:618`.
- Chapter 3.10.1 and 3.11 define sticky TRAPSTS fields, memory-violation
  sources, `TRAPSTS.mem_viol`, EXEC masking for buffer-to-LDS LDS address
  violations, and imprecise memory-violation saved-PC behavior at
  `cdna3/README.md:619` through `:654`.

Rocjitsu evidence:

- CDNA3 `S_TRAP` is marked as a program terminator and its `execute_impl()`
  throws `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopp.cpp:301` through
  `:313`.
- CDNA3 `S_GETREG_B32`/`S_SETREG_B32` only handle a small subset of HWREG IDs
  and default-log the rest; `S_GETREG_B32` handles ids 1, 4, 5, 6, and 7, and
  `S_SETREG_B32`/`S_SETREG_IMM32_B32` only write id 1 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopk.cpp:291` through
  `:385`.
- CDNA3 `V_CLREXCP` VOP1 and VOP3 bodies call shared helpers, but those
  helpers are empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:3796`
  through `:3799`, and semantic derivation maps `V_CLREXCP` to `true_nop` at
  `lib/python/amdisa/semantics.py:730`.
- Searching rocjitsu source and tests for `trapsts`, `TRAPSTS`, `mem_viol`,
  `MEM_VIOL`, `SAVECTX`, `ILLEGAL_INST`, `ADDR_WATCH`, `EXCP_CYCLE`, and
  `DP_RATE` finds no runtime trap-status or memory-violation model outside a
  semantic-fingerprint word list.

Impact:

Kernels that depend on trap enable state, trap status accumulation, memory
violation reporting, or `S_TRAP` NOP/trap behavior cannot be represented by the
current CDNA3 emulator path.

### CDNA3-RJ-055: TTMP privilege and CDNA3 TTMP launch initialization are missing

Manual evidence:

- Chapter 3.10 says all TTMP writes are privileged; outside a trap handler,
  writes are ignored and reads return zero at `cdna3/README.md:594`.
- Chapter 3.13 says TTMP4/5 are initialized to zero, TTMP6/7 hold the dispatch
  packet address, TTMP8/9/10 hold dispatch grid dimensions, TTMP11 holds
  `wave_id_in_workgroup`, and other TTMPs are not initialized at
  `cdna3/README.md:685` through `:693`.

Rocjitsu evidence:

- CDNA3 scalar operand reads for encodings 108 through 123 read the wavefront
  SGPR file directly with no privilege check at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/operand.cpp:1016` through
  `:1034`.
- CDNA3 scalar destination writes for encodings 108 through 123 similarly write
  the wavefront SGPR file directly, and 64-bit TTMP-pair writes do the same for
  encodings 108 through 122, at `operand.cpp:1180` through `:1230`.
- `CommandProcessor::init_wavefront_regs()` initializes user SGPRs, enabled
  workgroup-id SGPRs, an RDNA4/gfx1250-only TTMP7/TTMP9 launch payload, and
  packed VGPR0 workitem IDs, but has no CDNA3 branch for TTMP4 through TTMP11
  at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:226` through
  `:324`.

Impact:

Unprivileged CDNA3 code can observe and modify TTMP storage in rocjitsu instead
of seeing zero/ignored accesses, and kernels that use the CDNA3 TTMP launch
payload observe uninitialized simulator state.

### CDNA3-RJ-056: CDNA3 HW_ID and XCC_ID contents are incomplete

Manual evidence:

- Chapter 3.12 defines `HW_ID` bitfields for wave, SIMD, pipe, CU, shader
  engine, thread-group, VM, queue, state, and ME IDs, and defines `XCC_ID` bits
  3:0 at `cdna3/README.md:656` through `:668`.

Rocjitsu evidence:

- CDNA3 `S_GETREG_B32` treats HWREG id 4 as the low half of `wf.cu().id()` and
  id 5 as the high half, treats ids 6 and 7 as SGPR/VGPR allocation fields, and
  logs all other ids as unhandled at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopk.cpp:291` through
  `:320`.
- There is no CDNA3 `S_GETREG_B32` case for the XML/manual `XCC_ID` register
  id 20, and the existing id-4 path does not pack the manual's HW_ID fields.

Impact:

`S_GETREG_B32` consumers see a CU-id placeholder instead of the CDNA3 `HW_ID`
layout and cannot read `XCC_ID`, which affects kernels or tests that inspect
placement identity.

### CDNA3-RJ-057: The TG_SIZE system SGPR launch payload is not initialized

Manual evidence:

- Chapter 3.13 says that when `tg_size_en` is enabled, SPI initializes a system
  SGPR containing `{first_wave, 6'h00, wave_id_in_group[4:0], 2'h0, 14'h0,
  work-group_size_in_waves[5:0]}` after the enabled workgroup-id SGPRs at
  `cdna3/README.md:680` through `:684`.

Rocjitsu evidence:

- `CommandProcessor::init_wavefront_regs()` writes user SGPRs, optional
  kernarg preload SGPRs, and enabled workgroup-id SGPRs, then proceeds to an
  RDNA4/gfx1250 TTMP payload and VGPR workitem IDs; there is no TG_SIZE system
  SGPR write in the CDNA3 path at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:226` through
  `:324`.
- Searching rocjitsu for `TG_SIZE`, `tg_size`, `first_wave`,
  `wave_id_in_group`, and `workgroup_size_in_waves` finds no implementation;
  the only relevant descriptor plumbing found is workgroup-id enable handling.

Impact:

CDNA3 kernels compiled to consume the TG_SIZE system SGPR receive an
uninitialized or stale value from rocjitsu even though the manual defines a
precise launch-time payload.

### CDNA3-RJ-058: `S_RFE_B64` does not clear PRIV or branch to the return address

Manual evidence:

- Chapter 4.1 lists `S_RFE` as the trap-handler return instruction at
  `cdna3/README.md:712` through `:713`.
- The detailed `S_RFE_B64` definition says it may only be used within a trap
  handler, clears `WAVE_STATUS.PRIV`, and sets PC to the scalar source address
  at `cdna3/README.md:4654` through `:4664`.

Rocjitsu evidence:

- The CDNA3 generated `SRfeB64Sop1` constructor decodes the 64-bit source, but
  its execution delegates to `execute_s_rfe_b64_sop1()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sop1.cpp:596` through
  `:608`.
- The shared helper body is empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2405`
  through `:2406`.
- The CDNA3 `S_RFE_RESTORE_B64` form is decoded but throws
  `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sop2.cpp:1011` through
  `:1033`.

Impact:

Trap-handler return code cannot clear privileged mode or resume at the saved
PC in rocjitsu; `S_RFE_B64` behaves as a no-op and `S_RFE_RESTORE_B64` traps in
the emulator.

### CDNA3-RJ-059: Debug conditional branches never branch

Manual evidence:

- Chapter 4.2 says `S_CBRANCH_CDBGSYS`, `S_CBRANCH_CDBGUSER`, and
  `S_CBRANCH_CDBGSYS_AND_USER` branch based on the corresponding
  `COND_DBG_SYS` and `COND_DBG_USER` STATUS bits at `cdna3/README.md:727`
  through `:731`.

Rocjitsu evidence:

- CDNA3 constructors for the four debug conditional branches decode the label
  operand but do not set branch metadata flags, and each `execute_impl()`
  delegates to a shared helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopp.cpp:363` through
  `:413`.
- The shared helpers for `S_CBRANCH_CDBGSYS`,
  `S_CBRANCH_CDBGSYS_AND_USER`, `S_CBRANCH_CDBGSYS_OR_USER`, and
  `S_CBRANCH_CDBGUSER` are empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:997`
  through `:1011`.
- Rocjitsu defines CDNA3/4 `COND_DBG_USER` and `COND_DBG_SYS` status accessors
  in `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/cdna_isa_base.h:73`
  through `:78`, but the branch helpers do not read them.

Impact:

Debug-status conditional branches fall through regardless of STATUS, and
static branch analysis also lacks normal conditional-branch metadata for these
generated CDNA3 classes.

### CDNA3-RJ-060: Fork/join divergent control flow is not implemented

Manual evidence:

- Chapter 4.2 lists `S_CBRANCH_{G,I}_FORK` and `S_CBRANCH_JOIN` as conditional
  branch instructions for complex branching at `cdna3/README.md:727` through
  `:735`.
- The arbitrary divergent-control-flow section defines a six-deep CSP stack,
  `{EXEC, PC}` stack entries in SGPRs, pass/fail mask selection by bitcount,
  EXEC updates, branch target selection, and JOIN restoration at
  `cdna3/README.md:842` through `:890`.

Rocjitsu evidence:

- CDNA3 `S_CBRANCH_I_FORK` decodes the mask SGPR-pair and label but throws
  `util::UnimplementedInst` in `execute_impl()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopk.cpp:264` through
  `:278`.
- CDNA3 `S_CBRANCH_G_FORK` decodes its two 64-bit sources and also throws
  `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sop2.cpp:962` through
  `:984`.
- CDNA3 `S_CBRANCH_JOIN` decodes the source but dispatches to an empty shared
  helper at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sop1.cpp:916`
  through `:930` and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1013`
  through `:1015`.
- Searching rocjitsu for `CSP`, `control stack`, and fork/join helpers finds
  only generated decode fixtures and the unimplemented/empty execution paths,
  not a branch-stack state model.

Impact:

CDNA3 kernels that use compiler-emitted fork/join divergent control flow either
throw on the FORK instruction or fail to restore EXEC/PC at JOIN.

### CDNA3-RJ-061: Program-control status, priority, perf, trace, message, and wakeup side effects are mostly stubs

Manual evidence:

- Chapter 4.1 says `S_SETPRIO` modifies wave priority, `S_SLEEP` sleeps the
  wave, `S_SENDMSG` sends a host/upstream message, and `S_WAKEUP` wakes sleeping
  waves in the workgroup at `cdna3/README.md:714` through `:717`.
- Detailed SOPP definitions say `S_SETKILL` kills the wave when bit 0 is set,
  `S_SETHALT` sets or clears STATUS.HALT, `S_SETPRIO` updates the user priority
  bits, and `S_SENDMSGHALT` sends a message and halts at
  `cdna3/README.md:5408` through `:5474`.
- Detailed SOPP definitions also say `S_INCPERFLEVEL` and `S_DECPERFLEVEL`
  update performance counters and `S_TTRACEDATA` sends M0 to the thread-trace
  stream at `cdna3/README.md:5503` through `:5513`.
- Chapter 4.4 says `LGKM_CNT` increments by one for each `S_SENDMSG` and
  decrements when the message is sent out at `cdna3/README.md:771` through
  `:772`.

Rocjitsu evidence:

- CDNA3 `S_SETHALT`, `S_SETKILL`, `S_SETPRIO`, `S_SENDMSG`,
  `S_SENDMSGHALT`, `S_WAKEUP`, `S_INCPERFLEVEL`, `S_DECPERFLEVEL`, and
  `S_TTRACEDATA` dispatch to shared helpers, while `S_SLEEP` dispatches to a
  helper that only requests a functional yield. Representative generated calls
  are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopp.cpp:213` through
  `:298` and `:326` through `:360`.
- The shared helpers for `S_SENDMSG`, `S_SENDMSGHALT`, `S_SETHALT`,
  `S_SETKILL`, `S_SETPRIO`, and `S_WAKEUP` are empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2427`
  through `:2428`, `:2490` through `:2492`, `:2526` through `:2533`, and
  `:2689` through `:2690`.
- The shared helpers for `S_DECPERFLEVEL`, `S_INCPERFLEVEL`, and
  `S_TTRACEDATA` are also empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1637`
  through `:1638`, `:1776` through `:1777`, and `:2663` through `:2669`.
- `S_SLEEP` only calls `wf.cu().request_functional_yield()` and does not model
  the immediate value's sleep duration or `S_WAKEUP` interaction at
  `execute_shared.h:2555` through `:2563`.

Impact:

Control/status instructions used for halt, kill, priority, performance
counters, thread trace, host messages, message wait-counter accounting, and
sleep/wakeup scheduling have decode coverage but do not update the wave state,
side-channel state, wait-counter state, or scheduler behavior described by the
manual.

### CDNA3-RJ-128: Zero-target `S_SETPC_B64` and `S_SWAPPC_B64` hard-halt instead of branching

Manual evidence:

- `S_SETPC_B64` jumps to the byte address specified by the scalar source pair,
  with pseudocode `PC = S0.i64`, at `cdna3/README.md:4626` through `:4636`.
- `S_SWAPPC_B64` saves `PC + 4` and then jumps to the scalar input address,
  with pseudocode `PC = jump_addr.i64`, at `cdna3/README.md:4638` through
  `:4648`.
- The manual text for these default-form SOP1 PC instructions does not define a
  zero-address termination special case.

Rocjitsu evidence:

- Before the generated instruction body executes, `ComputeUnitCore` scans the
  mnemonic for `s_setpc` or `s_swappc`, reads the raw `SSRC0` scalar pair, and
  calls `active->halt()` when that target value is zero at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:392` through `:405`.
- The generated `S_SETPC_B64` and `S_SWAPPC_B64` execute bodies otherwise branch
  through the scalar source pair, and `S_SWAPPC_B64` writes the link value, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sop1.cpp:556` through
  `:594`.

Impact:

CDNA3 code that intentionally branches or calls through address zero is treated
as wave termination before the instruction semantics run. This changes control
flow and suppresses the `S_SWAPPC_B64` link write that the manual defines for a
zero target.

### CDNA3-RJ-062: Program-control tests are mostly decode fixtures

Rocjitsu evidence:

- Generated CDNA3 encoding fixtures include `s_rfe_b64`,
  `s_cbranch_join`, debug conditional branches, fork forms, `s_sethalt`,
  `s_setkill`, `s_setprio`, `s_sendmsg`, `s_sendmsghalt`, and `s_wakeup` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/test_encodings.h:53`,
  `:68`, `:99` through `:122`, `:144`, and `:189` through `:191`.
- The instruction execution harness explicitly excludes several control
  instructions, including `s_sethalt`, `s_sendmsg`, `s_sendmsghalt`, and
  `s_rfe`, at `emulation/rocjitsu/tests/instruction_execution_harness_test.cpp:169`
  through `:177`.
- Searching `emulation/rocjitsu/tests` found no behavior tests for CDNA3 debug
  conditional branch predicates, fork/join stack effects, `S_RFE` return
  behavior, or HALT/kill/message/wakeup side effects.

Impact:

The current tests can confirm these opcodes decode, but they would not catch
the no-op/unimplemented execution paths recorded above.

### CDNA3-RJ-063: `S_BARRIER` does not expose `STATUS.IN_BARRIER`

Manual evidence:

- Chapter 3 defines `STATUS.IN_BARRIER` bit 12 as "Wavefront is waiting at a
  barrier" at `cdna3/README.md:430` through `:437`.
- The detailed `S_BARRIER` definition says the wave waits at a threadgroup
  barrier until the release conditions are satisfied at `cdna3/README.md:5400`
  through `:5406`.

Rocjitsu evidence:

- The CDNA status wrapper defines `IN_BARRIER` as bit 12 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/cdna_isa_base.h:59`
  through `:62`.
- CDNA3 `S_GETREG_B32` reads raw STATUS through `wf.status_raw()` for HWREG id
  1 at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopk.cpp:291` through
  `:300`, and the wavefront stores raw status independently of scheduler state
  at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:608` through `:614`.
- `execute_s_barrier_sopp()` only sets `WfState::BARRIER` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:745`
  through `:747`; it does not set bit 12 while the wave is waiting or clear it
  when the CU releases the barrier.
- The CU release path changes only scheduler state from `BARRIER` back to
  `RUNNING` at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:313`
  through `:335`.

Impact:

Barrier scheduling can block and release waves, but shader/debug code that
observes raw STATUS through HWREG cannot see the architectural
`IN_BARRIER` bit while a wave is waiting.

### CDNA3-RJ-064: Workgroup size limits are not validated

Manual evidence:

- Chapter 4.3 says up to 16 wavefronts or 1024 work-items can be combined into
  a workgroup at `cdna3/README.md:746` through `:748`.

Rocjitsu evidence:

- The AQL dispatch path computes `wg_size` directly from packet workgroup
  dimensions, derives `wfs_per_wg = (wg_size + wave_size - 1) / wave_size`, and
  stores it in the dispatch entry at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:941` through
  `:965`.
- The workgroup admission check only tests available wavefront slots, SGPR
  blocks, VGPR blocks, and LDS capacity at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:215` through `:251`.
- A focused search found no architectural validation rejecting workgroups above
  16 waves or 1024 work-items.

Impact:

Rocjitsu can accept a dispatch shape that exceeds CDNA3's documented per
workgroup size limit if enough simulator resources are configured, which can
change residency, barrier membership, and launch-state behavior for malformed
or stress dispatches.

### CDNA3-RJ-065: Barrier tests cover only the basic all-live release case

Rocjitsu evidence:

- `HookOrderingTest.BarrierTwoWaves` runs `{S_BARRIER, S_ENDPGM}` with two
  waves in one workgroup and asserts one `BARRIER_RESOLVED` event plus two
  dispatched and halted wavefronts at
  `emulation/rocjitsu/tests/execution_plugin_test.cpp:870` through `:886`.
- The instruction execution harness excludes `s_barrier` from its generic
  execution coverage at
  `emulation/rocjitsu/tests/instruction_execution_harness_test.cpp:138`
  through `:166`.
- Searching the tests found no behavior cases for early-terminated peers,
  `STATUS.IN_BARRIER` observation, wait counters remaining outstanding across
  barrier issue, single-wave immediate release, or oversized workgroup
  rejection.

Impact:

The current behavior test would catch a disconnected baseline barrier-release
hook, but it would not catch the `IN_BARRIER` state gap, size-limit gap, or
several manual-only release/issuing conditions.

### CDNA3-RJ-066: Scalar-memory `LGKM_CNT` accounting ignores dword counts

Manual evidence:

- Chapter 4.4 says `LGKM_CNT` is incremented by the dword count for scalar
  memory reads, with one count for one-dword loads and two counts for
  two-dword-or-larger loads; `S_MEMTIME` counts like `S_LOAD_DWORDX2` at
  `cdna3/README.md:764` through `:765`.
- The same section says `LGKM_CNT` decrements once for each dword returned from
  the data cache for SMEM at `cdna3/README.md:771` through `:773`.
- Chapter 4.4 also states scalar-memory reads can return out of order, so only
  `S_WAITCNT 0` is legitimate for such dependencies at
  `cdna3/README.md:775` through `:778`.

Rocjitsu evidence:

- `ScalarMemState` carries `num_dwords`, but only one
  `wait_counter_type = WaitCounterType::LGKMCNT` value at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:70` through `:83`.
- The generated CDNA3 SMEM load bodies do set `num_dwords`, for example
  `S_LOAD_DWORD` uses `1` and `S_LOAD_DWORDX2` uses `2` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/smem.cpp:57` through
  `:94`.
- `MemoryPipeline::issue()` increments exactly one wait counter for the
  instruction at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.h:62`
  through `:88`.
- `ScalarMemPipeline::complete_access()` writes every returned dword to SGPRs,
  but completion returns once and the deferred callback releases that same
  single counter at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:203` through
  `:241`.

Impact:

Functional execution writes the right SGPR payload, but outstanding
`LGKM_CNT` depth does not match CDNA3 for multi-dword SMEM loads or
`S_MEMTIME`. Thresholds such as `lgkmcnt(1)` can therefore unblock differently
from hardware in timing/deferred paths, and the model cannot represent the
manual's per-dword SMEM return accounting.

### CDNA3-RJ-067: GWS `EXP_CNT` producer behavior is not modeled in production

Manual evidence:

- Chapter 4.4 says `LGKM_CNT` is incremented by one for each GWS instruction at
  `cdna3/README.md:764` through `:767`.
- The same section defines `EXP_CNT` as the VGPR-export count for GWS,
  incremented when a GWS instruction issues from the wavefront buffer and
  decremented when the last GWS cycle is granted/executed and VGPRs have been
  read out at `cdna3/README.md:779` through `:784`.

Rocjitsu evidence:

- The generated CDNA3 GWS instruction classes decode the GWS opcodes, but all
  six `execute_impl()` bodies immediately throw `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/ds.cpp:4829` through
  `:4904`.
- A targeted search for `WaitCounterType::EXPCNT` found only the wait-counter
  primitive and unit/infrastructure tests, not a production instruction path
  that issues or retires `EXP_CNT`.
- Existing broader GWS behavior is already tracked in `CDNA3-RJ-041`; this
  item records the specific Chapter 4.4 wait-counter consequence.

Impact:

Rocjitsu can decode GWS instructions, but it cannot model the CDNA3
`LGKM_CNT`/`EXP_CNT` effects that make GWS VGPR sources safe to overwrite only
after the export count retires.

### CDNA3-RJ-068: Wait-counter tests miss CDNA3 producer-accounting edge cases

Rocjitsu evidence:

- `tests/waitcnt_counter_test.cpp` covers primitive increments/decrements and
  saturation for individual counter types, including `EXPCNT`, at
  `emulation/rocjitsu/tests/waitcnt_counter_test.cpp:19` through `:117`.
- `tests/shared_infra_test.cpp` manually seeds an RDNA3 `EXPCNT` value and
  checks `S_WAITCNT_EXPCNT` threshold behavior at
  `emulation/rocjitsu/tests/shared_infra_test.cpp:199` through `:207`.
- Searching the adjacent tests found no focused CDNA3 producer cases for
  multi-dword SMEM `LGKM_CNT` accounting, `S_MEMTIME` counting like
  `S_LOAD_DWORDX2`, `S_SENDMSG` message-counter accounting, GWS `EXP_CNT`
  issue/retire behavior, or producer-driven FLAT dual-counter accounting beyond
  the already-recorded FLAT gap
  `CDNA3-RJ-034`.

Impact:

The tests prove that counter storage and manual threshold waits work in
isolation, but not that CDNA3 instructions feed those counters according to the
manual's producer-specific rules.

### CDNA3-RJ-069: Manual NOP wait-state hazards are not modeled or diagnosed

Manual evidence:

- Chapter 4.5 says hardware does not check several dependency classes, so the
  shader must resolve them by inserting NOPs or independent instructions at
  `cdna3/README.md:786` through `:788`.
- Table 11 lists required waits for `S_SETREG`/`S_GETREG`, `MODE.VSKIP`,
  VALU-produced VCC/EXEC/SGPR/VGPR values, lane-select consumers,
  `V_DIV_FMAS`, wide store/atomic write-data hazards, M0 consumers,
  TRAPSTS/RFE, DPP, OPSEL/SDWA bit-position changes, and trans-op consumers at
  `cdna3/README.md:790` through `:822`.
- Table 12 defines the trans-op set used by the Table 11 trans-op wait row at
  `cdna3/README.md:824` through `:831`.
- The detailed `S_ICACHE_INV` row says 16 separate `S_NOP` instructions or a
  jump/branch must follow instruction-cache invalidation at
  `cdna3/README.md:5498` through `:5504`.

Rocjitsu evidence:

- Generated CDNA3 `S_NOP` constructs a SOPP instruction and delegates to the
  shared helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopp.cpp:20` through
  `:28`, but the shared helper is an empty function at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2137`
  through `:2138`.
- The functional memory pipeline documents synchronous completion and has a
  no-op `tick()` in functional mode at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.h:35` through `:91`;
  `ExecComputeUnit` also leaves CLOCKED mode as a TODO at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.h:622` through `:643`.
- The existing DBT `HazardTracker` is explicitly a GFX12 `s_delay_alu`
  auto-insertion helper at
  `lib/rocjitsu/src/rocjitsu/code/dbt/hazard_tracker.h:4` through `:21`; it
  emits RDNA4 `S_DELAY_ALU` words at
  `lib/rocjitsu/src/rocjitsu/code/dbt/hazard_tracker.cpp:11` through `:31`,
  not CDNA3 Table 11 `S_NOP` timing or runtime diagnostics.
- `S_ICACHE_INV` dispatches to an empty shared helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1769`
  through `:1775`, with no branch-or-16-`S_NOP` spacing validation.
- Some rows overlap existing behavior gaps, such as consecutive `S_SETREG`
  hazard handling in `CDNA3-RJ-049` and unimplemented `S_SETVSKIP` behavior in
  `CDNA3-RJ-052`; this finding records the broader absence of a CDNA3 Table 11
  wait-state/scoreboard model.

Impact:

Rocjitsu executes most producer/consumer pairs with fully updated architectural
state at instruction boundaries. A kernel that omits required CDNA3 NOPs,
including instruction-cache invalidation spacing, can therefore appear correct
under emulation even when hardware requires padding or independent instructions
to avoid races.

### CDNA3-RJ-070: Tests do not cover CDNA3 Table 11 wait-state hazards

Rocjitsu evidence:

- Decode smoke tests only assert that the shared `S_NOP` encoding decodes to
  `s_nop` for CDNA3 and other ISAs at
  `emulation/rocjitsu/tests/decode_smoke_test.cpp:54` through `:125`.
- The generic instruction execution harness skips `s_nop`, `s_getreg`,
  `s_setreg`, memory, wait, barrier, branch, and message/control instructions
  at `emulation/rocjitsu/tests/instruction_execution_harness_test.cpp:138`
  through `:176`.
- DBT tests cover GFX12/RDNA4 `s_delay_alu` insertion paths, but the audited
  search found no CDNA3 tests that compare programs with and without the Table
  11 NOP counts or assert warnings/diagnostics for missing `S_NOP` hazards.

Impact:

Current tests cover `S_NOP` decoding and several unrelated functional
behaviors, but they would not catch a missing CDNA3 Table 11 hazard model or an
emulation path that silently accepts hardware-racy instruction sequences.

### CDNA3-RJ-071: VOP3 floating output modifiers ignore MODE denorm/IEEE gating

Manual evidence:

- Chapter 6.2.2 says VOP3 floating-point output modifiers are ignored when output
  denormals are enabled or when `MODE.IEEE` is 1; when output denormals are
  disabled, applying an output modifier also flushes denormals to zero and
  flushes `-0` to `+0` at `cdna3/README.md:1261` through `:1263`.

Rocjitsu evidence:

- `Wavefront` stores raw MODE state through `mode_raw()` and `set_mode_raw()` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:103` through `:111`.
- Representative generated VOP3 FP helpers apply `OMOD` and `CLAMP`
  unconditionally: `execute_v_add_f32_vop3()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:3018`
  through `:3056`, `execute_v_add_f64_vop3()` at `:3076` through `:3114`,
  `execute_v_mul_f32_vop3()` at `:16124` through `:16162`, and
  `execute_v_mul_f64_vop3()` at `:16182` through `:16220`.
- The shared `S_DENORM_MODE` helper is currently empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1644`
  through `:1645`, and the audited VOP3 helper bodies do not read `wf.mode_raw()`
  before applying the modifier or clamp.
- Existing SIMD correctness tests sweep modifier bits against the generated
  direct modifier model, for example
  `tests/simd_correctness/vop3_binary_simd_correctness_test.cpp:290` through
  `:296` and `tests/simd_correctness/vop3_modifier_simd_test.cpp:187` through
  `:217`, but they do not vary `MODE.IEEE` or `MODE.FP_DENORM`.

Impact:

CDNA3 VOP3 FP instructions can produce modified, clamped, denormal, or `-0`
results in rocjitsu even in MODE states where the manual says the modifier must
be ignored or the result must be flushed.

### CDNA3-RJ-072: VALU source validation allows manual-disallowed extra scalar sources

Manual evidence:

- Chapter 6.2.1 says VALU instructions can read at most one SGPR per instruction
  and can use at most one literal, only when neither SGPR nor M0 is used. It also
  says `ADDC`, `SUBB`, and `CNDMASK` implicitly use VCC and therefore cannot use
  an additional SGPR or literal at `cdna3/README.md:1226` through `:1233`.
- The detailed definitions clarify that VOP3 `V_CNDMASK_B32` may take the VCC
  source from scalar GPR `S2`, and VOP3 `V_ADDC_CO_U32`/`V_SUBB_CO_U32` may take
  the VCC source from the SGPR pair at `S2.u`, at `cdna3/README.md:6727` through
  `:6738`, `:7072` through `:7085`, `:13652` through `:13663`, and `:13960`
  through `:13973`. That explicit S2 source still consumes the scalar-source
  budget, so `S0`/`S1` cannot also be additional scalar or literal sources.

Rocjitsu evidence:

- Representative generated VOP3 `V_ADD_F32` independently constructs `src0` as
  `OPR_SRC_NOLIT` and `src1` as `OPR_SRC_SIMPLE` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3.cpp:2017` through
  `:2025`; both operand classes include scalar-source encodings.
- `Operand::read_lane()` resolves each non-VGPR VALU source independently
  through `resolve_src_scalar()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/operand.cpp:1340` through
  `:1352`, and `resolve_src_scalar()` can read SGPRs, VCC, M0, EXEC, inline
  constants, and helper predicates at `operand.cpp:1016` through `:1083`.
- Carry-consuming VOP3 forms such as `V_ADDC_CO_U32` construct broad
  `SRC0`/`SRC1` operands plus an explicit scalar-pair `SRC2` carry-in at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3.cpp:12247` through
  `:12260`, with no adjacent validation that the explicit `SRC2` scalar source
  disallows extra scalar/literal sources in `SRC0` or `SRC1`.

Impact:

This is a legality/diagnostic gap rather than proof that a legal encoding
executes incorrectly: rocjitsu can decode and execute source combinations that
the CDNA3 manual says the hardware contract disallows.

### CDNA3-RJ-073: VALU FP round/denorm modes and V_DOT2 denorm flushing are not modeled

Manual evidence:

- Chapter 6.4 says the shader program controls floating-point rounding and
  denormal input/result handling through MODE fields set by `S_SETREG`, with
  separate single-precision and double/half-precision fields at
  `cdna3/README.md:1435` through `:1446`.
- The same section says floating-point `V_DOT2` instructions do not support
  denormal or rounding modes and flush input and output denormals at
  `cdna3/README.md:1439`.

Rocjitsu evidence:

- `Wavefront` can store raw MODE state at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:103` through `:111`, but
  CDNA3 `S_SETREG_B32` and `S_SETREG_IMM32_B32` only handle HWREG id 1
  (`STATUS`) in the audited helpers at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopk.cpp:334` through
  `:352` and `:365` through `:384`, so the Chapter 6.4 `S_SETREG` path cannot
  write MODE.
- The only `mode_raw()` consumers found in the audited rocjitsu tree are raw
  storage/GPR-indexing paths, not VALU FP arithmetic. Representative FP VOP3
  helpers compute with host arithmetic and direct modifier handling, for
  example `execute_v_add_f32_vop3()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:3018`
  through `:3056` and `execute_v_add_f64_vop3()` at `:3076` through `:3114`.
- Floating `V_DOT2` helpers widen FP16 inputs with generic `util::f16_to_f32`,
  use ordinary host multiply/add, and optionally clamp; no explicit input/output
  denormal flush or MODE-independent V_DOT2 rule is visible in
  `execute_v_dot2_f32_f16_vop3p()` at `execute_shared.h:10450` through
  `:10481`, the `V_DOT2C` paths at `:10562` through `:10601`, or the SIMD path
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/simd_glue.h:3465`
  through `:3527`.
- The shared `S_DENORM_MODE` and `S_ROUND_MODE` execution helpers are empty at
  `execute_shared.h:1644` through `:1645` and `:2423` through `:2425`; those are
  not the CDNA3 Chapter 6.4 route, but they show there is no alternate shared
  MODE update path for these fields.

Impact:

Ordinary CDNA3 VALU FP arithmetic is evaluated under host default rounding and
denormal behavior instead of the MODE fields, and floating `V_DOT2` does not
statically show the manual's required flush-in/flush-out denormal behavior.

### CDNA3-RJ-074: ALU clamp non-FP semantics are incomplete

Manual evidence:

- Chapter 6.5 says `V_CMP` clamp requests signaling compare behavior on FP
  exceptions, integer operations clamp to the largest/smallest representable
  value, and floating-point operations clamp to `[0.0, 1.0]` at
  `cdna3/README.md:1448` through `:1450`.
- Chapter 12.9.1 repeats the VOPC/VOP3A compare-specific rule: `CLAMP=1`
  signals an exception when either compare input is NaN, while `CLAMP=0` does
  not, at `cdna3/README.md:11084` through `:11105`.
- CDNA3 VOP3/VOP3B format tables expose `CLMP` as a result/output clamp at
  `cdna3/README.md:23545` through `:23552` and `:23881` through `:23886`, and
  the opcode tables include integer forms such as `V_ADD_U32` and `V_ADD_I32`
  at `cdna3/README.md:23748` and `:23622`.

Rocjitsu evidence:

- Representative integer VOP3 helpers ignore `inst.inst_.clamp` and execute
  wrapping arithmetic directly: `execute_v_add_i32_vop3()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:3119`
  through `:3128`, `execute_v_add_u32_vop3()` at `:3244` through `:3253`, and
  `execute_v_sub_i32_vop3()` at `:18689` through `:18698`.
- The SIMD VOP3 integer fast-path comment acknowledges that clamp on an integer
  op means saturation, but says the wrap-around/bitwise twins do not request it
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/simd_glue.h:1367`
  through `:1372`; the VOP3 integer-compare path similarly says clamp is unused
  at `simd_glue.h:1546` through `:1552`.
- Representative VOP3 compare helpers compute and store the condition result
  without inspecting `inst.inst_.clamp` or setting exception/signaling state, for
  example `execute_v_cmp_eq_f32_vop3()` at `execute_shared.h:4358` through
  `:4387` and `execute_v_cmp_lg_f32_vop3()` at `:6009` through `:6040`.

Impact:

Programs that encode `CLMP=1` on ordinary integer VOP3 arithmetic will see
wrap-around results rather than CDNA3's documented saturation, and `V_CMP`
encodings cannot request the documented signaling-compare behavior through the
clamp bit.

### CDNA3-RJ-075: VGPR indexing uses the wrong M0 layout and cannot honor source-role masks

Manual evidence:

- Chapter 6.6 defines VGPR indexing as a MODE-enabled M0 index for selected VALU
  VGPR sources or destinations at `cdna3/README.md:1452` through `:1455`.
- Table 27 says `S_SET_GPR_IDX_ON` and `S_SET_GPR_IDX_MODE` store the mode in
  `M0[15:12]`, while `S_SET_GPR_IDX_IDX` stores the index in `M0[7:0]`, at
  `cdna3/README.md:1460` through `:1465`.
- The prose defines `M0[15]` as destination enable, `M0[14]` as source-2
  enable, `M0[13]` as source-1 enable, and `M0[12]` as source-0 enable, limits
  indexing to VGPR operands, and makes out-of-range indexed VGPRs illegal at
  `cdna3/README.md:1469` through `:1479`.
- Section 6.6.2 gives instruction-specific source/destination role remapping for
  readlane, writelane, MAC/MAD, shift, `v_cvt_pkaccum`, and SDWA preserve forms
  at `cdna3/README.md:1481` through `:1495`.

Rocjitsu evidence:

- The shared `S_SET_GPR_IDX_MODE` and `S_SET_GPR_IDX_ON` helpers store the mode
  with `<< 8` and masks around bits `[11:8]`, not `M0[15:12]`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2501`
  through `:2519`.
- `Wavefront::gpr_idx_mode()` likewise reads `(m0_ >> 8) & 0xF`, and
  `apply_gpr_idx()` treats any low three source bits as applying to every source
  operand rather than src0/src1/src2 separately at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:248` through `:251` and
  `:584` through `:589`.
- CDNA3 scalar fallback operand reads/writes pass only an `is_dst` boolean into
  `apply_gpr_idx()`, so the operand layer has no information about whether a
  VGPR read is src0, src1, or src2; representative paths are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/operand.cpp:1340` through
  `:1352` and `:1359` through `:1388`.
- The generic SIMD operand helpers use the same `apply_gpr_idx()` helper with
  only source-versus-destination information at
  `lib/rocjitsu/src/rocjitsu/isa/isa_operand_simd_inl.h:53` through `:60`,
  `:76` through `:77`, and `:105` through `:108`; true16 write glue does the
  same at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/simd_glue.h:305`
  through `:311`.

Impact:

CDNA3 programs that use `S_SET_GPR_IDX_*` according to the manual will write the
selector bits where rocjitsu does not read them, while programs that happen to
populate `M0[11:8]` can index the wrong source operands because rocjitsu lacks
per-source role mapping.

### CDNA3-RJ-076: Packed FP8/BF8-to-F32 converts do not validate even destination bases

Manual evidence:

- Table 31 says `CVT_PK_F32_FP8` and `CVT_PK_F32_BF8` write `dst,dst+1` and
  require an even destination VGPR at `cdna3/README.md:2002` through `:2003`.

Rocjitsu evidence:

- Generated CDNA3 VOP1 constructors model `VCvtPkF32Fp8Vop1` and
  `VCvtPkF32Bf8Vop1` with 64-bit `OPR_VGPR` destinations at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop1.cpp:10199` through
  `:10339`.
- Their execution writes through `RegisterAccess::write_lane64()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop1.cpp:10238` through
  `:10441`.
- The CDNA3 operand and register-access layers turn a 64-bit VGPR operand into a
  two-register region but do not validate even alignment:
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/operand.cpp:1368` through
  `:1389` and
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/register_access.h:731` through `:740`.

Impact:

An odd-`VDST` encoding for these packed widening converts will execute as a
write to the odd/even+1 pair in rocjitsu instead of being rejected or diagnosed
as an illegal destination base.

### CDNA3-RJ-077: F32-to-FP8/BF8 VOP3 converts ignore supported source modifiers

Manual evidence:

- Table 31 says `CVT_PK_FP8_F32`, `CVT_PK_BF8_F32`, `CVT_SR_FP8_F32`, and
  `CVT_SR_BF8_F32` support `NEG` and `ABS` source modifiers while ignoring
  `CLAMP` and `OMOD` at `cdna3/README.md:1996` through `:2001`.

Rocjitsu evidence:

- `VCvtPkFp8F32Vop3::execute_impl()` and `VCvtPkBf8F32Vop3::execute_impl()`
  read `src0` and `src1` as raw F32 values and convert them directly, with no
  `inst_.abs` or `inst_.neg` handling, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3.cpp:6068` through
  `:6123`.
- `VCvtSrFp8F32Vop3::execute_impl()` and `VCvtSrBf8F32Vop3::execute_impl()`
  likewise read raw `src0`, use `src1` as the seed, and merge the result byte
  without applying source `ABS` or `NEG` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3.cpp:6127` through
  `:6197`.
- Other generated VOP3 conversion bodies do apply source modifiers locally, for
  example `VCvtF32F16Vop3::execute_impl()` applies `inst_.abs` and `inst_.neg`
  after reading the selected half at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3.cpp:199` through
  `:216`.

Impact:

CDNA3 FP8/BF8 narrow conversions with negative inputs and `ABS`, or positive
inputs and `NEG`, will produce results for the unmodified source value instead
of the modifier-adjusted value required by the manual.

### CDNA3-RJ-129: `V_CVT_PK_{FP8,BF8}_F32` low-half writes zero the preserved half

Manual evidence:

- Chapter 12.11 definitions 674 and 675 say `V_CVT_PK_FP8_F32` and
  `V_CVT_PK_BF8_F32` write the packed two-byte result into the selected
  16-bit half using `OPSEL[3]`, while preserving the other half of `D0`, at
  `workspace_docs/amdgpu-isa-manuals/cdna3/README.md:15916` through `:15956`.

Rocjitsu evidence:

- Generated CDNA3 execution for both packed F32-to-FP8/BF8 forms reads the old
  destination through `implicit_uses`, but then calls
  `write_vop3_true16_dst(..., true)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3.cpp:6089` and
  `:6123`.
- The shared true16 destination helper interprets that final `true` as the
  CDNA low-half zero-high rule for CDNA1 through CDNA4 and, for low-half
  writes, returns `src_half` instead of merging with the old high 16 bits at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/simd_glue.h:285` through
  `:317`.
- Existing gfx1250 tests assert the analogous packed FP8 low/high destination
  preservation behavior at `tests/instruction_execution_harness_test.cpp:3216`
  through `:3283`, but no CDNA3 regression pins the Chapter 12.11 preservation
  rule for definitions 674 and 675.

Impact:

On CDNA3, a low-half `V_CVT_PK_FP8_F32` or `V_CVT_PK_BF8_F32` write can clear
destination bits 31:16 instead of preserving them, corrupting adjacent packed
data. The stochastic byte-write forms do their own read/mask/write merge and
are not covered by this specific bug.

### CDNA3-RJ-130: Stochastic FP8/BF8 helpers are not covered against the Chapter 12.11 seed-bit formula

Manual evidence:

- Chapter 12.11 definitions 676 and 677 say `V_CVT_SR_FP8_F32` adds
  `S1[31:12]` into the F32 mantissa before conversion, while
  `V_CVT_SR_BF8_F32` uses `S1[31:11]`, at
  `workspace_docs/amdgpu-isa-manuals/cdna3/README.md:15958` through `:16010`.
- Chapter 7 describes the same stochastic conversion family as adding the
  stochastic value and then truncating at
  `workspace_docs/amdgpu-isa-manuals/cdna3/README.md:2014` through `:2032`.

Rocjitsu evidence:

- Generated CDNA3 stochastic execution delegates the conversion to
  `util::f32_to_fp8_e4m3_fnuz_sr()` and
  `util::f32_to_bf8_e5m2_fnuz_sr()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3.cpp:6152` and
  `:6189`, then masks the selected destination byte manually.
- The helper implementations add seed-derived bits inside the conversion
  helper, including `seed >> 12` for normal FP8 values and `seed >> 11` for
  normal BF8 values, with separate subnormal paths at
  `lib/util/include/util/data_types.h:412` and `:734`.
- Existing helper/HIP conversion tests cover selected fixed outputs and finite
  conversion plumbing, but no CDNA3 end-to-end case asserts the Chapter 12.11
  FP8 `S1[31:12]` and BF8 `S1[31:11]` seed formulas across normal and
  subnormal inputs.

Impact:

The current helper structure may be equivalent to the manual's
add-random-then-truncate description for ordinary cases, but that equivalence
is not captured by a regression or hardware/oracle case. A future helper change
could use the wrong seed-bit range while the existing CDNA3 tests still pass.

### CDNA3-RJ-078: FP8/BF8 widening VOP1 SDWA converts honor ignored destination controls

Manual evidence:

- Table 31 says `CVT_F32_FP8`, `CVT_F32_BF8`, `CVT_PK_F32_FP8`, and
  `CVT_PK_F32_BF8` use SDWA source selection and ignore `ABS`, `NEG`, and
  `SEXT` at `cdna3/README.md:2002` through `:2005`.
- The section 7.2 prose says VOP1 8-bit-format converts use the SDWA word only
  for the `SRC0` VGPR and `SRC0_SELECT`, and that the other SDWA fields are
  ignored at `cdna3/README.md:2011` through `:2012`.

Rocjitsu evidence:

- The generated CDNA3 `VCvtF32Fp8Vop1` and `VCvtF32Bf8Vop1` SDWA paths copy
  generic SDWA source and destination fields, then merge the computed F32 result
  according to `sdwa_dst_sel_`/`sdwa_dst_unused_` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop1.cpp:9979` through
  `:10194`.
- The packed widening forms use the same generic SDWA source handling and
  destination merge controls around their 64-bit `VDST` writes at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop1.cpp:10238` through
  `:10464`.

Impact:

SDWA encodings that set `DST_SEL` or `DST_UNUSED` on these widening converts
can partially merge or preserve destination bits in rocjitsu, even though the
manual says those SDWA controls are ignored and the conversion should be
governed only by `SRC0_SELECT`.

### CDNA3-RJ-079: Device-memory consistency and acknowledgment behavior is not represented

Manual evidence:

- Section 2.3 describes the CDNA device-memory hierarchy, cache-less loads,
  load-clause overlap caching, write-combining, atomic pre-op return storage,
  write-confirmation acknowledgments, relaxed consistency, per-PE/per-channel
  scatter-write ordering, and acknowledgment/fence use at
  `cdna3/README.md:341` through `:354`.

Rocjitsu evidence:

- Ordinary scalar memory accesses issue direct L1 load/store operations and
  complete load writeback with no modeled acknowledgment, per-channel order
  state, or relaxed-consistency fence state at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:203` through
  `:241`.
- Vector atomics perform lane-by-lane L2 read-modify-write and store old values
  in per-lane response data for returned atomics, but do not model the manual's
  return-address write-confirmation acknowledgment path at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:310` through
  `:374`.
- `L2Cache` documents a functional cache model: ordinary read/write paths are
  not thread-safe, only `atomic_rmw()` is mutex-protected, and atomics serialize
  under cache-line stripes rather than PE/channel ordering state at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/l2_cache.h:49` through `:56` and
  `:126` through `:150`.
- AQL acquire-fence handling only invalidates modeled caches for agent/system
  scope at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:1052`
  through `:1059`.

Impact:

rocjitsu can still serve as a functional memory-value oracle for many kernels,
but it cannot validate the Chapter 2.3 relaxed-consistency, acknowledgment, or
per-PE/per-channel ordering contracts.

### CDNA3-RJ-080: Buffer floating atomics miss L2 FP numeric and packed-lane rules

Manual evidence:

- Chapter 9.2 says floating memory atomics execute in LDS and L2 and can be
  issued as LDS, Buffer, Flat, Global, and Scratch instructions at
  `workspace_docs/amdgpu-isa-manuals/cdna3/README.md:2927` through `:2931`.
- Float atomic ADD opcodes use RNE, and Table 52 defines L2 denormal behavior:
  packed F16/BF16 and F64 add/min/max do not flush denorms while F32 add
  flushes denorms at `cdna3/README.md:2935` through `:2968`.
- Chapter 9.2.3 defines SNaN quieting, NaN propagation/selection, signed-zero
  ordering, compare-store equality, and add edge cases at `cdna3/README.md:2972`
  through `:3017`.

Rocjitsu evidence:

- Generated CDNA3 MUBUF floating atomics lower to the generic memory-pipeline
  `AtomicOp::FADD`, `FMIN`, and `FMAX` operations: representative F32, packed
  F16, F64 add, and F64 min/max paths are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/mubuf.cpp:2213` through
  `:2408`.
- `BUFFER_ATOMIC_PK_ADD_F16` sets `elem_size = 4` and `AtomicOp::FADD`, the
  same scalar 32-bit path as `BUFFER_ATOMIC_ADD_F32`, at `mubuf.cpp:2259`
  through `:2266`.
- The semantic derivation table also labels packed FP atomics as "treated as
  32-bit fadd for now" at `lib/python/amdisa/semantics.py:1828` through
  `:1856`.
- The shared L2 atomic executor treats 4-byte floating atomics as one host
  `float` and 8-byte floating atomics as one host `double`, using ordinary
  addition plus `std::fmin`/`std::fmax` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:285` through
  `:364`.
- `VectorMemState` carries the `AtomicOp` enum, element size, and dataflow
  fields, but not the floating-point subtype, packed-lane mode, or
  denormal/rounding policy needed by Chapter 9.2, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:87` through `:112`.

Impact:

F32/F64 buffer atomics have a coarse functional model, but rocjitsu does not
enforce CDNA3's L2 floating-atomic denormal, RNE, or NaN/signed-zero rules.
Packed F16 buffer atomics are not type-correct because their two lanes execute
as one scalar F32 operation. Existing `CDNA3-RJ-035` and `CDNA3-RJ-040` cover
the related flat/global/scratch and LDS floating-atomic paths.

### CDNA3-RJ-081: `S_ABSDIFF_I32` uses mathematical absolute difference instead of the wrapped SOP2 definition

Manual/XML evidence:

- Chapter 12.1 defines `S_ABSDIFF_I32` as a 32-bit `D0.i32 = S0.i32 - S1.i32`,
  followed by `D0.i32 = -D0.i32` only when that 32-bit result is negative, and
  gives overflow examples such as
  `S_ABSDIFF_I32(0x80000000, 0x00000001) => 0x7fffffff` at
  `cdna3/README.md:3809` through `:3832`.
- The XML only summarizes the operation as "absolute value of difference" at
  `amdgpu_isa_cdna3.xml:37622` through `:37623`; the wrapped-subtract edge
  examples are recorded as `CDNA3-XML-055`.

Rocjitsu evidence:

- Generated `SAbsdiffI32Sop2::execute_impl()` delegates to the shared helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sop2.cpp:986` through
  `:1008`.
- The shared helper promotes both signed inputs to `int64_t` and computes
  `a > b ? a - b : b - a` before truncating to 32 bits at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:358`
  through `:368`.
- The generator lowering emits the same wide absolute-difference algorithm at
  `lib/python/amdisa/codegen/execute/scalar.py:480` through `:489`, so
  regeneration preserves this behavior.

Impact:

The manual's overflow examples do not all execute correctly. For example,
`S_ABSDIFF_I32(0x80000000, 0x00000001)` should produce `0x7fffffff`, but the
wide absolute-difference implementation produces `0x80000001`.

### CDNA3-RJ-082: Chapter 12.1 SOP2 tests miss detailed semantic edge contracts

Evidence:

- The generated CDNA3 encoding fixture includes a single `s_absdiff_i32`
  encoding smoke entry at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/test_encodings.h:190`.
- The Python semantic-derivation test for `S_ABSDIFF_I32` only checks that the
  derived block contains an `ABSDIFF` call at
  `lib/python/amdisa/tests/test_sema_derive.py:371` through `:377`.
- The Python `S_BFE_U32` test only checks for a BFE call and SCC write at
  `lib/python/amdisa/tests/test_sema_derive.py:521` through `:531`.
- This slice found no targeted runtime golden tests for the `S_ABSDIFF_I32`
  overflow examples, BFM/BFE width and offset edge cases, shifted-add carry
  cases, or scalar-pack literal/high-half selection.

Impact:

Full-suite passing can miss the concrete `S_ABSDIFF_I32` divergence above and
similar SOP2 definition-level regressions in bitfield, shifted-add, and pack
rows.

### CDNA3-RJ-083: `S_CALL_B64` direct calls are flagged as `INDIRECT_CALL`

Manual/XML evidence:

- Chapter 12.2 defines `S_CALL_B64` as `D0.i64 = PC + 4` and
  `PC = PC + signext(SIMM16.i16 * 4) + 4`, and says the instruction implements
  a short subroutine call that must be 4 bytes at `cdna3/README.md:4152`
  through `:4169`.
- The XML describes the target as a constant offset relative to the current PC
  and marks `IsIndirectBranch` false at `amdgpu_isa_cdna3.xml:41996` through
  `:42039`.

Rocjitsu evidence:

- CDNA3 `SCallB64Sopk` exposes a PC-relative `branch_offset_bytes()` and
  executes by writing `PC + size_` to `SDST` and adding the signed SIMM16
  instruction-count offset to `wf.pc` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopk.cpp:387` through
  `:407`.
- The same constructor sets `flags_ |= INDIRECT_CALL` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopk.cpp:395`, even though
  `Instruction` documents that flag as an indirect call whose target comes from
  a register at `lib/rocjitsu/src/rocjitsu/isa/instruction.h:30` through `:33`.
- Some internal clients compensate by treating `INDIRECT_CALL` plus a present
  `branch_offset_bytes()` as a direct call at
  `lib/rocjitsu/src/rocjitsu/analysis/indirect_branch_discovery.cpp:596`
  through `:598`, but flag-only clients such as the probe-callable helper still
  classify calls from the flag alone at
  `lib/rocjitsu/src/rocjitsu/code/patch/probe_callable.cpp:105`.

Impact:

Execution and relocation have a direct PC-relative offset available, but the
public instruction flag conflates direct PC-relative calls with register-target
indirect calls. Flag-only analysis, instrumentation, or reporting clients can
misclassify `S_CALL_B64` relative to the manual/XML branch metadata.

### CDNA3-RJ-084: M0-relative scalar moves truncate M0 and B64 forms scale the offset

Manual/XML evidence:

- Chapter 12.3 defines `S_MOVRELS_B32/B64` as `addr = SRC0.u32; addr +=
  M0.u32[31:0]`, then reads from `SGPR[addr]`; the B64 form also says both the
  index in M0 and the source address must be even at `cdna3/README.md:4808`
  through `:4842`.
- `S_MOVRELD_B32/B64` uses the same full-32-bit `addr += M0.u32[31:0]` rule for
  the instruction destination field, with even M0 plus an even destination
  address required for B64 at `cdna3/README.md:4844` through `:4877`.
- The XML exposes M0 as an implicit operand on `S_MOVRELS_*` and
  `S_MOVRELD_*` at `amdgpu_isa_cdna3.xml:31120` through `:31363`, while the
  detailed formula omission is recorded in `CDNA3-XML-057`.

Rocjitsu evidence:

- All four relative-move helpers first truncate the index with
  `index = wf.m0() & 0xFFu`, so values outside the low byte are ignored before
  address formation at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sop1.cpp:815` through `:818`,
  `:844` through `:847`, `:873` through `:876`, and `:902` through `:905`.
- The B64 helpers then compute `width_words = size_bits / 32` and use
  `base + index * width_words`; for the 64-bit forms this is `base + 2 * M0` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sop1.cpp:844` through `:848`
  and `:902` through `:906`.
- Neither path checks the manual's even-M0 requirement before forming the
  64-bit SGPR pair.

Impact:

For any relative move, an M0 value such as `0x100` should participate in the
manual's full-width `addr += M0.u32[31:0]` calculation, but rocjitsu uses zero.
For B64, a legal even M0 value such as 2 means `s_movrels_b64 s[10:11],
s[2:3]` should read from `s[4:5]`, but rocjitsu reads from `s[6:7]`. Odd M0
values are also silently remapped by the doubled formula instead of being
rejected or diagnosed as invalid 64-bit relative moves.

### CDNA3-RJ-085: Default-only SOP1 PC forms are accepted as 8-byte literal encodings

Manual/XML evidence:

- `S_GETPC_B64` saves `PC + 4` and its note says the instruction must be 4 bytes
  at `cdna3/README.md:4612` through `:4624`.
- `S_SETPC_B64` jumps to an address specified in a scalar register at
  `cdna3/README.md:4626` through `:4636`.
- `S_SWAPPC_B64` saves `PC + 4`, jumps to the scalar input, and its note says
  the instruction must be 4 bytes at `cdna3/README.md:4638` through `:4652`.
- The CDNA3 XML provides only default `ENC_SOP1` encodings for
  `S_GETPC_B64`, `S_SETPC_B64`, and `S_SWAPPC_B64`, with no
  `SOP1_INST_LITERAL` alternatives, at `amdgpu_isa_cdna3.xml:30155` through
  `:30255`.

Rocjitsu evidence:

- The generated CDNA3 `Sop1` base class treats every SOP1 instruction with
  `SSRC0 == 255` as a 64-bit instruction by adding an extension dword to
  `size_` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/encodings.cpp:24` through
  `:36`.
- `SGetpcB64Sop1::execute_impl()` writes `wf.pc + size_`, so a reserved
  `SSRC0 == 255` encoding would report `PC + 8` instead of the manual's
  required `PC + 4` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sop1.cpp:543` through `:554`.
- `SSetpcB64Sop1` and `SSwappcB64Sop1` rewrite `SSRC0 == 255` into an
  `OPR_SIMM32` operand and then branch through it at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sop1.cpp:556` through `:594`,
  even though the XML/manual form is a scalar-register target.

Impact:

Reserved/default-only SOP1 PC encodings can consume the following dword as a
literal extension, change instruction length, and alter PC/link values or branch
targets. This can desynchronize decode streams and hide illegal encodings behind
apparently valid `S_GETPC_B64`, `S_SETPC_B64`, or `S_SWAPPC_B64` instructions.

### CDNA3-RJ-086: `S_SET_GPR_IDX_ON` treats operand 1 as a literal-capable scalar source

Manual/oracle evidence:

- The detailed `S_SET_GPR_IDX_ON` definition says vector operations use M0 for
  relative GPR addressing, source 0 supplies the index, and the raw bits of the
  `SRC1` field set the enable bits; the pseudocode writes `M0[15:12]` from
  `SRC1[3:0]` and says this is direct raw-field content at
  `cdna3/README.md:5207` through `:5223`.
- `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx942` accepts a 4-bit mode operand
  and a source-0 literal, producing `s_set_gpr_idx_on s0, 15` as a one-dword
  encoding and `s_set_gpr_idx_on 0x12345678, 15` as an 8-byte source-0 literal
  encoding, but rejects `s_set_gpr_idx_on s0, 16` and
  `s_set_gpr_idx_on s0, 0x12345678`.

Rocjitsu evidence:

- The generic CDNA3 `Sopc` base treats any `ssrc1 == 255` as a literal form and
  increases instruction size at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/encodings.cpp:38` through
  `:54`.
- `SSetGprIdxOnSopc` initially declares operand 1 as `OPR_SIMM4`, but still
  replaces it with an `OPR_SIMM32` operand when the raw field is 255 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopc.cpp:411` through `:427`.
- The shared executor then reads operand 1 through `RegisterAccess` and masks the
  resulting value to four bits before writing M0 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2514`
  through `:2519`. The separate M0 bit-position bug is tracked in
  `CDNA3-RJ-075`.

Impact:

rocjitsu accepts and sizes an operand-1 literal form that the ISA text and LLVM
assembler treat as invalid. If a raw `SSRC1=255` word is encountered, rocjitsu
consumes the next dword as a literal and derives the mode from that extension
word instead of treating operand 1 as a raw 4-bit mode field.

### CDNA3-RJ-087: XML-only SOPP opcodes are generated on CDNA3 with incomplete execution

Manual/XML evidence:

- The CDNA3 manual's detailed Chapter 12.5 SOPP definitions and Chapter 13.1.5
  SOPP opcode table stop at `S_SET_GPR_IDX_MODE` opcode 29, and a direct manual
  search finds no `S_ENDPGM_ORDERED_PS_DONE` or `S_SET_VALU_COEXEC_MODE` entry;
  this source drift is recorded in `CDNA3-XML-059`.
- CDNA3 XML nevertheless records `S_ENDPGM_ORDERED_PS_DONE` as `ENC_SOPP`
  opcode 30 with program-terminator metadata and records
  `S_SET_VALU_COEXEC_MODE` as opcode 31 with a `SIMM16` operand and text saying
  the value in `SIMM16[1:0]` controls vector ALU co-execution mode for the next
  VALU instruction at `amdgpu_isa_cdna3.xml:42925` through `:42966`.

Rocjitsu evidence:

- Generated CDNA3 decoding maps SOPP opcodes 30 and 31 to
  `decodeSEndpgmOrderedPsDoneSopp` and `decodeSSetValuCoexecModeSopp` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/decoder.cpp:7012` through
  `:7013`.
- Generated CDNA3 smoke encodings include `s_endpgm_ordered_ps_done` and
  `s_set_valu_coexec_mode` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/test_encodings.h:126`
  through `:127`.
- `SEndpgmOrderedPsDoneSopp::execute_impl()` only calls `wf.end()`, while
  `SSetValuCoexecModeSopp::execute_impl()` is a no-op at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopp.cpp:449` through
  `:468`.
- Static source search found no CDNA3 wavefront or VALU issue state for the
  one-instruction co-execution mode described by the XML.

Impact:

Manual-based CDNA3 tools may reject SOPP opcodes 30 and 31 as absent, while
rocjitsu accepts and disassembles them. If the XML rows are treated as
authoritative, the generated execution still reduces the ordered-PS row to a
plain wave termination and ignores the one-instruction VALU co-execution mode.

### CDNA3-RJ-088: FP min/max helpers do not model NaN, signed-zero, or IEEE/MODE tie rules

Manual evidence:

- `V_MIN_F32` and `V_MAX_F32` define SNaN quieting, NaN operand selection,
  signed-zero selection, and IEEE/non-IEEE equality behavior at
  `cdna3/README.md:6845` through `:6895`.
- `V_MAX_F16` and `V_MIN_F16` define the corresponding F16 edge behavior at
  `cdna3/README.md:7301` through `:7357`.

Rocjitsu evidence:

- The shared F16 and F32 VOP2/VOP3 max helpers call `util::stdx::fmax` in the
  SIMD path and `std::fmax` in the scalar path at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:13810`
  through `:13894`.
- The corresponding min helpers call `util::stdx::fmin` and `std::fmin` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:15117`
  through `:15201`.
- The adjacent SIMD correctness test explicitly excludes NaN-input lanes and
  signed-zero ties from comparison at
  `tests/simd_correctness/vop2_minmax_simd_correctness_test.cpp:4` through
  `:19`.

Impact:

The helpers may be adequate for ordinary finite lanes, but they do not encode
the ISA's operand-selection contract for NaNs, SNaNs, signed-zero ties, or
IEEE-mode equality differences. Existing SIMD/scalar parity coverage
intentionally avoids those lanes, so this gap can survive local correctness
tests.

### CDNA3-RJ-089: `V_PK_FMAC_F16` decodes but throws in VOP2 and VOP3 forms

Manual/XML evidence:

- The CDNA3 manual defines `V_PK_FMAC_F16` as a packed F16 accumulate operation
  at `cdna3/README.md:7506` through `:7513`.
- The checked-in XML exposes `V_PK_FMAC_F16` VOP2 literal/DPP alternatives and
  the promoted VOP3 form at `amdgpu_isa_cdna3.xml:59618` through `:59695`.

Rocjitsu evidence:

- The generated VOP2 constructor records `VDST` as both the accumulator source
  and destination, but `VPkFmacF16Vop2::execute_impl()` throws
  `UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop2.cpp:8020` through
  `:8066`.
- The generated VOP3 constructor has the same accumulator/destination operand
  shape, but `VPkFmacF16Vop3::execute_impl()` also throws
  `UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3.cpp:3393` through
  `:3410`.

Impact:

Programs using this manual-defined packed accumulate opcode can decode and
disassemble, but execution terminates with an unimplemented-instruction
exception instead of performing the two packed F16 FMAs.

### CDNA3-RJ-090: VOP2 accumulator DPP forms permute `VDST` instead of real `SRC0`

Manual/XML evidence:

- Chapter 12.7 says VOP2 instructions may use DPP immediately after the
  instruction at `cdna3/README.md:6721` through `:6725`, and Chapter 12.16
  defines the DPP extension's `SRC0` field as the source-0 VGPR at
  `cdna3/README.md:24150` through `:24170`.
- The DPP exclusion list at `cdna3/README.md:22581` onward does not exclude
  accumulator-style VOP2 rows such as `V_FMAC_F64`, `V_MAC_F16`, `V_DOT*C`, or
  `V_FMAC_F32`, and the XML records DPP alternatives for these rows, for
  example `V_FMAC_F64` at `amdgpu_isa_cdna3.xml:52102` through `:52135` and
  `V_MAC_F16` at `:56423` through `:56446`.

Rocjitsu evidence:

- Generated accumulator-style VOP2 constructors place `VDST` in
  `src_operands_[0]` and the real source 0 in a later slot; `VFmacF64Vop2` is a
  representative example at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop2.cpp:571` through
  `:580`.
- The generic DPP preamble still applies DPP to `src_operands_[0]`, then
  delegates `VDST` to the DPP operand before executing the helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop2.cpp:616` through
  `:674`.
- The same pattern is present in `VMacF16Vop2` at `vop2.cpp:4587` through
  `:4695`, dot accumulators at `:7351` through `:7593` and `:7621` through
  `:7851`, and `VFmacF32Vop2` at `:7879` through `:7980`.

Impact:

Legal DPP forms for accumulator-style VOP2 instructions use the DPP-selected
old destination as the accumulator while leaving the real `SRC0` unpermuted.
That reverses the intended source role for cross-lane DPP selection and can
produce wrong results for any non-identity DPP control.

### CDNA3-RJ-091: Literal-only `_MK`/`_AK` VOP2 forms accept modifier-shaped encodings

Manual/XML evidence:

- `V_FMAMK_F32`, `V_FMAAK_F32`, `V_MADMK_F16`, and `V_MADAK_F16` are
  literal-only VOP2 rows, and the manual says they do not support VOP3, input
  modifiers, or output modifiers at `cdna3/README.md:6993` through `:7017` and
  `:7194` through `:7221`.
- The XML rows expose only the literal VOP2 forms for these instructions at
  `amdgpu_isa_cdna3.xml:54563` through `:54718` and `:56485` through `:56632`.

Rocjitsu evidence:

- The generic VOP2 encoding helper still classifies `src0 == 250` as DPP and
  `src0 == 249` as SDWA for all VOP2 opcodes, while separately marking opcodes
  23, 24, 36, and 37 as implied-literal forms at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/encodings.cpp:199` through
  `:210`.
- Generated `V_FMAMK_F32` and `V_FMAAK_F32` constructors/executors parse DPP
  and SDWA sentinels while also reading the same extension dword as `SIMM32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop2.cpp:2925` through
  `:3145`.
- The F16 literal forms follow the same DPP/SDWA parsing pattern at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop2.cpp:4735` through
  `:4955`.

Impact:

rocjitsu can treat extension words for manual-defined literal-only opcodes as
DPP or SDWA modifier payloads, even though the ISA defines those rows as
literal-only and explicitly forbids input/output modifiers. That can admit
encodings the hardware/assembler contract rejects, or decode a required literal
extension through the wrong extension format.

### CDNA3-RJ-092: `V_READFIRSTLANE_B32` returns zero instead of lane 0 when `EXEC` is disabled

Manual evidence:

- `V_READFIRSTLANE_B32` says an all-disabled `EXEC` mask selects lane 0, while
  nonzero `EXEC` selects the lowest active lane; it also says the VGPR read
  overrides the `EXEC` mask at `cdna3/README.md:7574` through `:7590`.

Rocjitsu evidence:

- Generated VOP1 execution initializes `val` to zero, iterates only active
  `EXEC` lanes, and writes the untouched zero when no active lane exists at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop1.cpp:324` through
  `:332`.
- The promoted VOP3 alias follows the same pattern at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3.cpp:54` through `:64`.
- Existing shared-infra coverage exercises an active-lane case for
  `v_readfirstlane_b32_e32 s24, v1` at
  `emulation/rocjitsu/tests/shared_infra_test.cpp:3486`, but does not seed
  `EXEC == 0`.

Impact:

An all-disabled wave reads zero instead of lane 0. Kernels using
`V_READFIRSTLANE_B32` as an EXEC-independent scalarization primitive can observe
incorrect results when no lanes are active.

### CDNA3-RJ-093: `V_SAT_PK_U8_I16` decodes but throws in both VOP1 and VOP3 forms

Manual/XML evidence:

- `V_SAT_PK_U8_I16` saturates two signed 16-bit integer inputs over an unsigned
  8-bit range and packs the results at `cdna3/README.md:8725` through `:8746`.
- The checked-in XML exposes base VOP1 and promoted VOP3 encodings at
  `amdgpu_isa_cdna3.xml:50740` through `:50838`.

Rocjitsu evidence:

- The generated VOP1 decoder and constructor cover opcode 79, including
  `implicit_uses()` expansion for the packed destination, but
  `execute_impl()` throws `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop1.cpp:9636` through
  `:9683`.
- The promoted VOP3 form has the same generated dataflow and throwing executor
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3.cpp:1819` through
  `:1839`.
- Semantic derivation maps `V_SAT_PK_U8_I16` to `nop` at
  `lib/python/amdisa/semantics.py:731`.

Impact:

Legal CDNA3 VOP1/VOP3 encodings decode but cannot execute, and semantic
fingerprints can understate the missing saturation/packing behavior.

### CDNA3-RJ-094: XML-only `V_SCREEN_PARTITION_4SE_B32` decodes but cannot execute

Manual/XML evidence:

- The CDNA3 manual VOP1 tables skip opcode 55, while the checked-in XML defines
  `V_SCREEN_PARTITION_4SE_B32` at `amdgpu_isa_cdna3.xml:48291` through
  `:48400`; `CDNA3-XML-065` records the source drift.

Rocjitsu evidence:

- The generated decoder maps VOP1 opcode 55 and promoted VOP3 opcode 375 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/decoder.cpp:6444` and
  `:7749`.
- Generated VOP1 construction accepts default, literal, DPP, SDWA, and VOP3
  forms, but both VOP1 and VOP3 executors throw `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop1.cpp:6720` through
  `:6761` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3.cpp:808`
  through `:822`.

Impact:

rocjitsu exposes an XML-only CDNA3 opcode but has no executable semantics for
it. Depending on the source-of-truth decision, this either needs execution
support or architecture-specific removal/invalid-op handling.

### CDNA3-RJ-095: `V_MOV_B32` and `V_MOV_B64` VOP3 aliases ignore allowed floating modifiers

Manual evidence:

- `V_MOV_B32` allows input modifiers when the source is treated as F32, and
  `V_MOV_B64` allows input modifiers when the source is treated as F64, at
  `cdna3/README.md:7549` through `:7572` and `:8378` through `:8388`.

Rocjitsu evidence:

- The VOP3 aliases construct modifier fields, but shared execution helpers
  perform raw lane copies and ignore `inst_.abs`/`inst_.neg` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:15912`
  through `:15920` for B32 and `:15937` through `:15944` for B64.
- The adjacent SIMD test currently documents ignored `v_mov_b32` modifiers as
  expected raw-copy behavior at
  `emulation/rocjitsu/tests/simd_correctness/vop3_unary_simd_correctness_test.cpp:251`
  through `:287`.

Impact:

VOP3 MOV encodings with legal floating input modifiers can produce the
unmodified bit pattern instead of the manual-defined F32/F64 modified value.

### CDNA3-RJ-096: `V_FRACT_F32/F64` helpers omit the ISA max-below-one clamp

Manual evidence:

- `V_FRACT_F32` and `V_FRACT_F64` define DX-style negative behavior, obey the
  selected rounding mode, and clamp results to the largest representable value
  below 1.0 at `cdna3/README.md:7896` through `:7912` and
  `:8324` through `:8339`.

Rocjitsu evidence:

- VOP1 and VOP3 F32 helpers compute `v - std::floor(v)` without the manual
  max-below-one clamp at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:11980`
  through `:12029`.
- VOP1 and VOP3 F64 helpers use the same unclamped formula at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:12034`
  through `:12081`.
- Existing `fract` coverage in
  `emulation/rocjitsu/tests/simd_correctness/vop1_simd_correctness_test.cpp:156`
  through `:160` and
  `emulation/rocjitsu/tests/simd_correctness/vop1_f64_simd_correctness_test.cpp:80`
  through `:84` compares scalar and SIMD execution paths, so it does not catch
  a shared manual-oracle mismatch.

Impact:

Boundary inputs just below a negative integer can round to exactly 1.0 in the
emulator instead of the ISA's max-below-one value.

### CDNA3-RJ-097: Chapter 12.16 VOP1 DPP/SDWA exclusions are not enforced

Manual/XML evidence:

- Chapter 12.16.1 says DPP cannot be used with `V_READFIRSTLANE_B32`, the F64
  VOP1 conversion/unary block from `V_CVT_I32_F64` through `V_FRACT_F64`,
  `V_CLREXCP`, and `V_SWAP_B32` at `cdna3/README.md:22583` through `:22616`.
- Chapter 12.16.2 similarly excludes SDWA for `V_READFIRSTLANE_B32`,
  `V_CLREXCP`, and `V_SWAP_B32` at `cdna3/README.md:22619` through `:22638`.
- `CDNA3-XML-070` records the F64 VOP1 DPP rows that also leak through the
  checked-in XML.

Rocjitsu evidence:

- The generic VOP1 encoding helper treats source selectors 250 and 249 as DPP
  and SDWA extension formats without instruction-local legality checks at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/encodings.cpp:121` through
  `:153`.
- Generated F64 VOP1 constructors accept and execute DPP forms for prohibited
  rows, including representative paths for `V_CVT_I32_F64`,
  `V_CVT_F32_F64`, `V_TRUNC_F64`, `V_FREXP_EXP_I32_F64`, and `V_FRACT_F64` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop1.cpp:359` through
  `:410`, `:1711` through `:1762`, `:2683` through `:2734`, `:5962` through
  `:6013`, and `:6232` through `:6283`.
- `V_READFIRSTLANE_B32` parses DPP/SDWA extension words despite the manual
  exclusion at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop1.cpp:239`
  through `:280`.
- `V_SWAP_B32` accepts literal, DPP, and SDWA extension words and applies the
  generic DPP/SDWA machinery before swapping at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop1.cpp:9696` through
  `:9790`.

Impact:

rocjitsu can admit and execute modifier-extension encodings that the CDNA3 ISA
declares illegal. The literal-only `_MK`/`_AK` VOP2 overlap is tracked
separately in `CDNA3-RJ-091`; this entry is limited to the Chapter 12.16 VOP1
exclusions.

### CDNA3-RJ-098: VOP3 `V_CMPX_CLASS_*` writes VCC instead of explicit SDST

Manual/XML evidence:

- `V_CMPX_CLASS_F32` stores the class result into both `EXEC` and `D0`, with
  `D0 = VCC` only for VOPC encoding, at `cdna3/README.md:8970` through
  `:9022`.
- Chapter 13.3.3 says compare results target VCC in VOPC encoding and an
  arbitrary SGPR in VOP3 encoding at `cdna3/README.md:23320` through `:23340`.
- The checked-in XML matches that split: `V_CMPX_CLASS_F32` `ENC_VOP3`
  exposes an explicit `VDST` `OPR_SDST` plus implicit `OPR_SDST_EXEC`, while
  `ENC_VOPC` exposes VCC plus implicit EXEC at `amdgpu_isa_cdna3.xml:68693`
  through `:68840`.

Rocjitsu evidence:

- Generated CDNA3 VOP3 class-X constructors define `vdst` for
  `VCmpxClassF32Vop3`, `VCmpxClassF64Vop3`, and `VCmpxClassF16Vop3`, but their
  execute bodies end with `wf.set_vcc(result); wf.set_exec(result);` and never
  write the explicit scalar destination at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3.cpp:6218` through
  `:6271`, `:6290` through `:6345`, and `:6438` through `:6499`.
- The class-compare generator has the same pattern: `gen_vector_cmp_class()`
  writes VCC for `is_cmpx` whenever `cmpx_writes_vcc` is true and has no
  `dst`/VOP3 SDST branch in that path at
  `lib/python/amdisa/codegen/execute/vector_cmp.py:27` through `:151`.
- Ordinary non-class `gen_vector_cmpx()` already handles the distinction by
  writing `dst[0]` for VOP3 and VCC for VOPC at
  `lib/python/amdisa/codegen/execute/vector_cmp.py:384` through `:421`, so the
  bug is specific to the class-X generator path.

Impact:

`v_cmpx_class_{f16,f32,f64}_e64` clobbers fixed VCC and leaves the requested
SDST stale. Programs that use a non-VCC scalar destination for a VOP3 compare
mask see incorrect mask data and incorrect VCC liveness.

### CDNA3-RJ-099: VOPC SDWA sub-dword sources ignore scalar-source mode

Manual/XML evidence:

- Chapter 13.3.8 defines SDWAB as the VOPC SDWA extension word and includes
  `S0`/`S1` bits selecting whether source 0/source 1 are VGPR or SGPR sources
  at `cdna3/README.md:24120` through `:24148`.
- Chapter 12.16.2 excludes SDWA only for `V_MAC_F32`, `V_MAD*`, `V_FMAC_F32`,
  `V_READFIRSTLANE_B32`, `V_CLREXCP`, and `V_SWAP_B32`, leaving VOPC compare
  rows in scope at `cdna3/README.md:22618` through `:22638`.

Rocjitsu evidence:

- Generated VOPC constructors preserve the SDWA scalar-source mode initially:
  for representative `VCmpClassF32Vopc`, `sw->s0` creates `src0` as
  `OPR_SRC`, and `sw->s1` creates `vsrc1` as `OPR_SRC`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vopc.cpp:44` through
  `:61`.
- The shared generated SDWA preamble then ignores that operand type whenever
  `SRC*_SEL != DWORD`: it builds a VGPR address from `encoding_value_`, reads
  `read_vgpr()`, and creates a `DppOperand` delegate at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vopc.cpp:75` through
  `:104`.
- The same template also applies ABS/NEG by bit-casting selected dwords to
  `float` regardless of whether the VOPC row is F32, F16, integer, or a class
  mask source. The code generator emits this generic template for all
  SDWA-capable VOPC operations at
  `lib/python/amdisa/codegen/_generator.py:6724` through `:6772`.

Impact:

VOPC SDWA encodings with scalar sources and byte/word source selectors can
execute with VGPR lane data instead of the selected SGPR value. Integer and F16
compare SDWA modifier cases can also see F32-style ABS/NEG transformations
instead of type-appropriate source handling.

### CDNA3-RJ-100: Illegal 64-bit VOPC DPP forms are accepted

Manual/XML evidence:

- Chapter 12.16.1 explicitly says DPP cannot be used with
  `V_CMP_CLASS_F64`, `V_CMPX_CLASS_F64`, `V_CMP*_F64`, `V_CMP*_I64`,
  `V_CMP*_U64`, and their CMPX forms at `cdna3/README.md:22583` through
  `:22616`.
- Chapter 13.3.9's generic 64-bit DPP row-only note at
  `cdna3/README.md:24155` through `:24199` does not remove those
  instruction-specific exclusions.

Rocjitsu evidence:

- The CDNA3 generator treats `ENC_VOPC` as DPP-capable on non-RDNA DPP16
  targets by returning true from `_supports_dpp_for_encoding('ENC_VOPC')`, and
  maps `ENC_VOPC` DPP through the VOP1 DPP machine-inst layout at
  `lib/python/amdisa/codegen/_generator.py:740` through `:815` and `:6333`
  through `:6390`.
- Representative generated 64-bit VOPC constructors accept `SRC_DPP`, read the
  DPP extension, and execute `apply_dpp()` without instruction-local legality
  checks. `VCmpClassF64Vopc` does this at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vopc.cpp:302` through
  `:331`.
- `Vopc::default_encoding()` also treats `SRC_DPP` as an extension-word form
  for all VOPC rows, with no VOPC-specific legality filter, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/encodings.cpp:157`
  through `:171`.

Impact:

Rocjitsu can decode and execute 64-bit VOPC compare DPP forms that the CDNA3
manual marks illegal, instead of rejecting or diagnosing those encodings.

### CDNA3-RJ-101: CMPX semantic derivation drops the VCC/SDST result

Manual/XML evidence:

- Chapter 13.3.3 says every compare writes VCC for VOPC or an SGPR for VOP3,
  and CMPX variants additionally write EXEC, at `cdna3/README.md:23320`
  through `:23340`.
- `V_CMPX_CLASS_F32` spells this out as `EXEC.u64[laneId] = D0.u64[laneId] =
  result` at `cdna3/README.md:8970` through `:9022`.

Rocjitsu evidence:

- `_VectorCmpx.derive()` contains a TODO noting that it writes only EXEC even
  though GFX9/CDNA CMPX also writes VCC, then emits only an EXEC lane
  assignment at `lib/python/amdisa/sema_derive.py:1321` through `:1340`.
- `_VectorCmpxClass.derive()` has the same shape for class compares: it builds
  the class-test result and assigns only `EXEC[laneId]` at
  `lib/python/amdisa/sema_derive.py:1374` through `:1401`.
- Existing semantic-derivation coverage checks only that CMPX mentions EXEC,
  not that it also preserves the CDNA VCC/SDST result contract, at
  `lib/python/amdisa/tests/test_sema_derive.py:1451` through `:1462`.

Impact:

Any semantic/DBT consumer built from `vector_cmpx` or `vector_cmpx_class`
metadata can lose the architectural compare mask result even when the generated
C++ execute generator models some ordinary CMPX writeback cases correctly.

### CDNA3-RJ-102: Packed 16-bit VOP3P operand metadata mixes packed-width and element-width

Manual/XML evidence:

- CDNA3 Chapter 12.10 packed 16-bit rows produce both low and high 16-bit
  components and write the packed result to `D0.b32`; representative rows are
  `V_PK_MAD_I16`, `V_PK_MUL_LO_U16`, `V_PK_ADD_F16`, and `V_PK_MIN_F16` at
  `cdna3/README.md:11115` through `:11130` and `:11272` through `:11322`.
- `CDNA3-XML-074` records that the checked-in XML uses 16-bit operands for
  some packed rows and 32-bit operands for adjacent rows with the same packed
  one-dword dataflow.

Rocjitsu evidence:

- Generated CDNA3 constructors inherit that mixed metadata: `V_PK_MAD_I16`,
  `V_PK_MUL_LO_U16`, `V_PK_MAX_I16`, `V_PK_MAD_U16`, `V_PK_MAX_U16`,
  `V_PK_MIN_U16`, `V_PK_MIN_F16`, and `V_PK_MAX_F16` construct 16-bit operands
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3p.cpp:36` through
  `:66`, `:170` through `:221`, `:275` through `:302`, and `:373` through
  `:401`.
- Adjacent packed F16 rows construct 32-bit source operands at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3p.cpp:320` through
  `:361`.
- The operand resolver switches inline floating constants between f16 bit
  patterns and f32 bit patterns solely from `size_bits_`: 16-bit operands use
  `resolve_src_scalar16()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/operand.cpp:1086` through
  `:1107`, while other widths use `resolve_src_scalar()` at `:1043` through
  `:1060` and `:1348` through `:1352`.

Impact:

Register-sourced scalar execution still reads the full 32-bit lane and writes a
packed 32-bit result, but scalar/inline-constant sources and generated operand
metadata are inconsistent across the packed 16-bit family. For example, packed
F16 min/max use half-precision inline constants while packed F16 add/mul/fma
use the 32-bit floating constant pattern before splitting the value into two
halves.

### CDNA3-RJ-103: `V_ACCVGPR_READ` and `V_ACCVGPR_WRITE` miss the `ACCVGPR` instruction flag

Manual/XML evidence:

- CDNA3 Chapter 12.10 defines `V_ACCVGPR_READ` and `V_ACCVGPR_WRITE` as
  accumulator VGPR move operations at `cdna3/README.md:11851` through `:11857`.
- The checked-in XML canonical names are `V_ACCVGPR_READ` and
  `V_ACCVGPR_WRITE`, with `_B32` only as aliases, at
  `amdgpu_isa_cdna3.xml:67020` through `:67079`.

Rocjitsu evidence:

- Rocjitsu documents `InstFlags::ACCVGPR` as covering `v_accvgpr_write`,
  `v_accvgpr_read`, and `v_accvgpr_mov`, and `Instruction::is_accvgpr()` reads
  that flag at `lib/rocjitsu/src/rocjitsu/isa/instruction.h:46` and `:206`.
- The generator only sets the flag for `_B32` canonical names
  `V_ACCVGPR_WRITE_B32`, `V_ACCVGPR_READ_B32`, and `V_ACCVGPR_MOV_B32` at
  `lib/python/amdisa/codegen/_generator.py:6544` through `:6549`.
- Generated CDNA3 `V_ACCVGPR_READ` and `V_ACCVGPR_WRITE` constructors do not
  set `flags_ |= ACCVGPR` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3p.cpp:693` through
  `:718`, while the generated `V_ACCVGPR_MOV_B32` VOP1 constructor does set
  the flag at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop1.cpp:9855`.

Impact:

The lane-wise copy execution bodies are present, so this is a classification
and analysis gap rather than a base decode/execute gap. Any path using
`is_accvgpr()` to recognize accumulator move instructions will miss the
read/write VOP3P forms.

### CDNA3-RJ-104: Native VOP3A rows decode but throw at execution

Manual/XML evidence:

- CDNA3 Chapter 12.11 defines `V_QSAD_PK_U16_U8`, `V_MQSAD_PK_U16_U8`, and
  `V_MQSAD_U32_U8` at `cdna3/README.md:14954` through `:14989`.
- The same chapter defines `V_TRIG_PREOP_F64` at `cdna3/README.md:15727`
  through `:15762` and `V_CVT_PKNORM_I16_F16` /
  `V_CVT_PKNORM_U16_F16` at `:15832` through `:15863`.
- The checked-in XML exposes these rows as `ENC_VOP3` instructions with
  ordinary operands.

Rocjitsu evidence:

- Generated CDNA3 constructors and decoder entries exist for the QSAD/MQSAD
  group, but each `execute_impl()` throws `UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3.cpp:4032`,
  `:4052`, and `:4072`.
- `VTrigPreopF64Vop3::execute_impl()` throws at `cdna3/vop3.cpp:5781`.
- `VCvtPknormI16F16Vop3::execute_impl()` and
  `VCvtPknormU16F16Vop3::execute_impl()` throw at `cdna3/vop3.cpp:5898`
  and `:5916`.
- Other `UnimplementedInst` throws in `cdna3/vop3.cpp` are already covered by
  older entries such as `CDNA3-RJ-089`, `CDNA3-RJ-093`, and `CDNA3-RJ-094`.

Impact:

Legal CDNA3 VOP3A instructions pass decode and constructor coverage but abort
when executed. This is a runtime support gap, not an opcode-inventory gap.

### CDNA3-RJ-105: `V_LSHL_ADD_U64` uses the ordinary masked shift rule

Manual evidence:

- `V_LSHL_ADD_U64` says the shift count must be 0 through 4, higher counts are
  unsupported, and unsupported counts behave as a shift of zero at
  `cdna3/README.md:15462` through `:15475`.

Rocjitsu evidence:

- Generated CDNA3 execution dispatches `V_LSHL_ADD_U64` to the shared helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:12745`
  through `:12761`.
- That helper calls `lshl_masked()` for the 64-bit shift, and the shared
  64-bit implementation masks the count with `count & 63` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/simd_glue.h:58`.

Impact:

Shift counts 5 through 63 produce real shifted results in rocjitsu instead of
the manual's zero-shift behavior. Counts outside 0 through 4 therefore diverge
from the CDNA3 instruction contract.

### CDNA3-RJ-106: F32 packed conversion VOP3A helpers ignore source modifiers and `PKRTZ` rounding

Manual evidence:

- Chapter 6.2.1 says VOP3 instructions with floating-point inputs may apply
  `ABS` and `NEG` to input operands at `cdna3/README.md:1235`.
- Chapter 12.11 defines F32-input packed conversions including
  `V_CVT_PK_U8_F32`, `V_CVT_PKACCUM_U8_F32`,
  `V_CVT_PKNORM_I16_F32`, `V_CVT_PKNORM_U16_F32`, and
  `V_CVT_PKRTZ_F16_F32` at `cdna3/README.md:14718` through `:14729`,
  `:15170` through `:15183`, and `:15763` through `:15808`.
- `V_CVT_PKRTZ_F16_F32` explicitly uses round-toward-zero and ignores the
  current rounding mode at `cdna3/README.md:15783` through `:15808`.

Rocjitsu evidence:

- `execute_v_cvt_pk_u8_f32_vop3()` and
  `execute_v_cvt_pkaccum_u8_f32_vop3()` read raw `SRC0` as `float` without
  applying `inst.inst_.abs` or `inst.inst_.neg` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:9718`
  through `:9745`.
- `execute_v_cvt_pknorm_i16_f32_vop3()` and
  `execute_v_cvt_pknorm_u16_f32_vop3()` use
  `ROCJITSU_TRY_SIMD_VOP3_BINARY_INT`, whose comment says it reads
  `src0/src1` with no modifiers at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/simd_glue.h:3901`
  through `:3904`, and their scalar fallbacks read raw `SRC0`/`SRC1` at
  `execute_shared.h:9751` through `:9814`.
- `execute_v_cvt_pkrtz_f16_f32_vop3()` uses the same no-modifier SIMD macro,
  reads raw scalar fallback sources, and calls `util::f32_to_f16()` rather than
  the available `util::f32_to_f16_rtz()` helper at
  `execute_shared.h:9825` through `:9843` and
  `lib/util/include/util/data_types.h:135` through `:155`.

Impact:

Encoded `ABS`/`NEG` bits are ignored for these F32-input VOP3A packed
conversions, and `V_CVT_PKRTZ_F16_F32` rounds like the ordinary F32-to-F16
helper instead of using round-toward-zero semantics.

### CDNA3-RJ-107: VOP3B VCC selector operands are invisible to def-use metadata

Manual/XML evidence:

- VOP3B is the scalar-destination encoding for carry/div-scale/wide-MAD
  opcodes at `cdna3/README.md:12393` through `:12413`.
- Chapter 13.3.5 lists VCC selector values in the VOP3B source selector table
  at `cdna3/README.md:23879` through `:23924`.
- The checked-in XML exposes `V_ADD_CO_U32` and carry-consuming VOP3B rows with
  `SDST` or `SRC2` operands typed as `OPR_SREG`, for example
  `amdgpu_isa_cdna3.xml:54854` through `:54872` and `:55378` through `:55397`.

Rocjitsu evidence:

- Generated VOP3B constructors preserve those operands as `OPR_SREG`; for
  example `VAddCoU32Vop3SdstEnc` uses `sdst(64, OperandType::OPR_SREG, ...)`
  and `VAddcCoU32Vop3SdstEnc` uses both `sdst` and `src2` as `OPR_SREG` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3.cpp:12190` through
  `:12260`.
- Runtime scalar reads and writes special-case selector value `106` as VCC at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/operand.cpp:1023`,
  `:1131`, and `:1233`, so execution can use the VCC selector.
- `Operand::to_register_ref()` maps `OPR_SREG` only for SGPR values 0 through
  101 and drops VCC selector values 106/107 at `operand.cpp:927` through
  `:934`.
- `InstDefUse` relies on `to_register_ref()` for explicit source and
  destination operands at
  `lib/rocjitsu/src/rocjitsu/analysis/def_use_chain.cpp:28` through `:42`.

Impact:

Execution can read or write VCC through legal VOP3B selector operands, but
def-use/liveness analysis misses those explicit VCC uses and defs. Carry chains
using VCC-form VOP3B operands can therefore be misrepresented to downstream
analysis even though scalar-pair SGPR operands are tracked.

### CDNA3-RJ-108: `DS_WRAP_RTN_B32` decodes but always throws

Manual/XML evidence:

- Chapter 12.12 defines `DS_WRAP_RTN_B32` opcode 52 as a ring-buffer-oriented
  wraparound subtract/add RMW at `cdna3/README.md:18702` through `:18717`.
- The checked-in XML has the same row at `amdgpu_isa_cdna3.xml:10239` through
  `:10288`.

Rocjitsu evidence:

- The CDNA3 decoder table maps DS opcode 52 to `decodeDsWrapRtnB32Ds` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/decoder.cpp:8588`.
- `DsWrapRtnB32Ds::execute_impl()` throws `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/ds.cpp:2083` through
  `:2086`.

Impact:

Legal CDNA3 code using `DS_WRAP_RTN_B32` aborts in rocjitsu even though the
opcode decodes.

### CDNA3-RJ-109: `DS_WRITE_ADDTID_B32` and `DS_READ_ADDTID_B32` use the wrong address contract

Manual/XML evidence:

- Chapter 12.12 defines both ADDTID forms as using
  `{OFFSET1, OFFSET0} + M0[15:0] + laneID * 4`, with no `ADDR` VGPR operand, at
  `cdna3/README.md:18410` through `:18418` and `:19835` through `:19844`.
- The XML rows expose `DATA0` plus implicit DSMEM/M0 for the write form and
  `VDST` plus implicit DSMEM/M0 for the read form at
  `amdgpu_isa_cdna3.xml:9047` through `:9071` and `:13521` through `:13545`.

Rocjitsu evidence:

- `DsWriteAddtidB32Ds::execute_impl()` calls ordinary
  `ds_calculate_addresses()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/ds.cpp:1002` through
  `:1010`.
- The shared DS address helper reads the encoded `ADDR` VGPR and computes
  `ADDR + {OFFSET1, OFFSET0} + lds_base` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:60`
  through `:76`, even though ADDTID has no architectural `ADDR` source.
- `DsReadAddtidB32Ds::execute_impl()` computes a per-lane stride from
  `M0[24:16] * 4` and uses `lane * stride + offset + lds_base` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/ds.cpp:4920` through
  `:4945`.
- The semantic derivation comment records the same stride model at
  `lib/python/amdisa/semantics.py:2241` through `:2250`.

Impact:

ADDTID accesses are wrong whenever `M0[15:0]` is nonzero, and the write form can
depend on an encoded `ADDR` field that the manual and XML operand list do not
expose. The older no-gap note for ADDTID has been retired in favor of this
specific gap.

### CDNA3-RJ-110: `DS_CONDXCHG32_RTN_B64` is modeled as generic compare-swap

Manual/XML evidence:

- Chapter 12.12 defines `DS_CONDXCHG32_RTN_B64` as two independent conditional
  dword writes: align `(ADDR + offset)` with `& 0xfff8`, return both old dwords,
  and write each dword only when the corresponding source MSB is set, clearing
  that MSB on write, at `cdna3/README.md:19688` through `:19709`.
- The XML row has one 64-bit `DATA0` source and generic 64-bit DSMEM
  input/output operands at `amdgpu_isa_cdna3.xml:13310` through `:13352`.

Rocjitsu evidence:

- Generated execution sets `d->atomic_op = amdgpu::AtomicOp::CMPSWAP`,
  `elem_size = 8`, and `num_elems = 1` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/ds.cpp:4798` through
  `:4808`.
- The same body reads both `DATA0` and `DATA1` register-pair fields into a
  four-dword compare-swap source payload at `ds.cpp:4811` through `:4824`.
- Semantic derivation classifies `_CONDXCHG32_RTN_B64` as `cmpswap` at
  `lib/python/amdisa/semantics.py:2335` through `:2341`.

Impact:

rocjitsu can use the wrong source operands, skip the manual address alignment,
and perform ordinary compare-swap behavior instead of the manual's per-half
conditional write exchange.

### CDNA3-RJ-111: `DS_SWIZZLE_B32` omits FFT and rotate modes

Manual/XML evidence:

- Chapter 12.12 defines `DS_SWIZZLE_B32` offset `>= 0xe000` as FFT mode,
  `0xc000 <= offset < 0xe000` as rotate mode, and lower offsets as two basic
  modes at `cdna3/README.md:18799` through `:18913`.
- The XML row identifies the opcode and operands at `amdgpu_isa_cdna3.xml:10636`
  through `:10658`.

Rocjitsu evidence:

- CDNA3 `DsSwizzleB32Ds::execute_impl()` dispatches to
  `execute_ds_swizzle_b32_ds()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/ds.cpp:2374` through
  `:2376`.
- The shared helper only branches on `offset & 0x8000` for quad mode versus
  bit mode and never checks the manual's `0xc000`/`0xe000` rotate/FFT ranges at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:265`
  through `:295`.
- Adjacent tests cover broadcast and quad-permutation offsets, not rotate or
  FFT offsets, at `tests/instruction_execution_harness_test.cpp:3393` through
  `:3415`.

Impact:

High-offset swizzle encodings execute with the wrong lane mapping.

### CDNA3-RJ-112: DS lane-routing helpers ignore the `ACC` register bank

Manual/XML evidence:

- Chapter 13.4.1 defines DS bit 25 `ACC` as selecting an AccVGPR destination at
  `cdna3/README.md:24216` through `:24221`.
- The XML rows for `DS_SWIZZLE_B32`, `DS_PERMUTE_B32`, and
  `DS_BPERMUTE_B32` expose `VDST` as `OPR_VGPR_OR_ACCVGPR`, and the permute
  rows also expose `DATA0` as `OPR_VGPR_OR_ACCVGPR`, at
  `amdgpu_isa_cdna3.xml:10636` through `:10747`.

Rocjitsu evidence:

- The CDNA3 constructors adjust visible operands by `ACC_MIN` when `inst_.acc`
  is set at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/ds.cpp:2359`
  through `:2425`.
- The shared swizzle, permute, and bpermute execution helpers read and write
  `vb + inst.inst_.data0` or `vb + inst.inst_.vdst` directly, ignoring
  `inst.inst_.acc`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:68`
  through `:70`, `:204` through `:206`, and `:271` through `:272`.

Impact:

Encodings with `ACC=1` can display as AccVGPR operands but execute against the
ordinary VGPR bank. At minimum, accumulator destinations for DS lane-routing
forms write the wrong register file.

### CDNA3-RJ-113: `DS_APPEND` and `DS_CONSUME` add `M0` in the LDS path

Manual/XML evidence:

- Chapter 12.12 says LDS APPEND/CONSUME use `instr_offset`, while GDS uses
  `M0.base + instr_offset`, at `cdna3/README.md:19882` through `:19888`.

Rocjitsu evidence:

- `DsConsumeDs::execute_impl()` and `DsAppendDs::execute_impl()` both reject
  `inst_.gds`, then compute `wf.lds_base() + wf.m0() + offset` for the LDS
  access at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/ds.cpp:5053`
  through `:5113`.
- The memory pipeline has special active-count handling for APPEND/CONSUME at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:393` through
  `:419`, so the gap is the LDS address base rather than the count update.

Impact:

LDS APPEND/CONSUME can access the wrong counter whenever `M0` is nonzero.

### CDNA3-RJ-114: Scalar DS floating atomics use the generic host FP path

Manual evidence:

- Chapter 11 says LDS floating atomics use `MODE.FP_DENORM` denormal behavior
  and fixed round-to-nearest-even rounding at `cdna3/README.md:3314` through
  `:3318`.
- Chapter 12.12 marks DS floating add/min/max/compare-store forms as handling
  NaN, Inf, and denorms, for example at `cdna3/README.md:18683` and `:19686`.

Rocjitsu evidence:

- Generated scalar DS floating atomics lower to the same memory-pipeline
  `AtomicOp::FADD`, `FMIN`, `FMAX`, or `CMPSWAP` machinery as other atomics.
- `apply_fp_atomic()` uses host `+`, `std::fmin`, and `std::fmax` without LDS
  denormal-mode policy, fixed RNE handling, SNaN quieting, or signed-zero/NaN
  selection rules at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:285` through
  `:297`.

Impact:

Scalar `DS_ADD/MIN/MAX/CMPST_F32/F64` can differ from CDNA3 hardware on denorm
mode, NaN propagation/quieting, signed zero, and rounding edge cases. Existing
`CDNA3-RJ-040` covers packed LDS F16/BF16 atomics; `CDNA3-RJ-080` covers the
related L2/buffer floating-atomic path.

### CDNA3-RJ-115: MUBUF no-return atomics still advertise `VDATA` as a destination

Manual evidence:

- Chapter 12.13 atomic rows say the original memory value is returned only when
  `SC0` is set, for example `BUFFER_ATOMIC_ADD` at
  `cdna3/README.md:20443`.
- Chapter 9.1.10.2 says `SC0=1` means return the pre-op value and `SC0=0`
  means no return for buffer atomics at `cdna3/README.md:2856` through `:2862`.

Rocjitsu evidence:

- Generated MUBUF atomic execute bodies correctly condition memory readback on
  `SC0`; representative `BUFFER_ATOMIC_ADD` code sets
  `d->is_load = (inst_.sc0 != 0)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/mubuf.cpp:1709` through
  `:1714`.
- The generated instruction metadata still unconditionally exposes `VDATA` as
  both a source and a destination with `dst_operands_[0] = &vdata` and
  `num_dst_ = 1` for the same row at `mubuf.cpp:1685` through `:1702`. The
  same constructor pattern is used across swap, compare-swap, integer,
  floating, packed, and X2 buffer atomics.

Impact:

Execution avoids the actual `VDATA` write when `SC0=0`, but public def-use,
liveness, and race/metadata consumers see a false destination write for legal
no-return buffer atomics.

### CDNA3-RJ-116: `BUFFER_WBL2` and `BUFFER_INV` do not model returned ACK/wait behavior

Manual evidence:

- Chapter 12.13 says both `BUFFER_WBL2` and `BUFFER_INV` return an ACK to the
  shader at `cdna3/README.md:20411` through `:20415`.
- Chapter 9.1.10 gives scope-specific writeback/invalidate behavior for these
  operations at `cdna3/README.md:2838` through `:2882`.

Rocjitsu evidence:

- Generated `BufferWbl2Mubuf` and `BufferInvMubuf` constructors have no memory
  operation flag, no destination, and no wait-counter state at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/mubuf.cpp:1568` through
  `:1588`.
- Shared helpers synchronously flush or invalidate broad cache levels and
  return immediately at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:45`
  through `:56`.
- `CDNA3-RJ-030` tracks the coarse cache-policy behavior; this entry covers the
  separate returned-ACK/dependency-accounting contract.

Impact:

Cache maintenance side effects are represented only as immediate helper calls.
Workloads or tests that depend on the architectural ACK/wait behavior cannot be
validated against rocjitsu.

### CDNA3-RJ-117: The FLAT subdecoder over-accepts scratch atomics

Manual/XML evidence:

- Chapter 13.6.3 lists SCRATCH opcodes only from 16 through 42, ending with
  `SCRATCH_LOAD_LDS_DWORD`, at `cdna3/README.md:24535` through `:24556`.
- The CDNA3 XML likewise has `ENC_FLAT_SCRATCH` instruction rows for opcodes
  16 through 42 only, beginning at `amdgpu_isa_cdna3.xml:19530`; no
  `SCRATCH_ATOMIC_*` rows are present.
- FLAT and GLOBAL tables, by contrast, include atomic opcodes 64 through 82 and
  96 through 108 at `cdna3/README.md:24460` through `:24531`.

Rocjitsu evidence:

- `Decoder::subDecodeFlat()` indexes `sub_decode_flat` only by `op.op` and does
  not validate the `SEG` field at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/decoder.cpp:5141` through
  `:5143`.
- The same table maps opcodes 64 through 82 and 96 through 108 to
  `FlatAtomic*` constructors unconditionally at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/decoder.cpp:8861` through
  `:8878` and later.
- `flat_mnemonic()` rewrites any `flat_*` mnemonic to `scratch_*` when
  `SEG==1` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/encodings.cpp:14` through
  `:18`, and the generated atomic constructors contain scratch
  `inst_.seg == 1` operand shaping.

Impact:

Illegal `SEG=SCRATCH` encodings with atomic opcode numbers decode and execute
as if scratch atomics existed on CDNA3, instead of rejecting the encoding.

### CDNA3-RJ-118: FLAT/GLOBAL no-return atomics still advertise `VDST` as a destination

Manual evidence:

- Chapter 12.15 atomic rows say the original memory value is returned only when
  `SC0` is set, for example `FLAT_ATOMIC_ADD` at
  `cdna3/README.md:21315` and `GLOBAL_ATOMIC_ADD` at
  `cdna3/README.md:22204`.
- Chapter 13.6 defines `SC0` as the scope/atomic-return bit and `VDST` as the
  destination for memory return data at `cdna3/README.md:24447` through
  `:24458`.

Rocjitsu evidence:

- Generated atomic constructors unconditionally expose `VDST` as a destination
  with `dst_operands_[0] = &vdst` and `num_dst_ = 1`; representative
  `FlatAtomicAddFlat` code is at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/flat.cpp:1143` through
  `:1162`, and X2 compare-swap uses the same pattern at `:2195` through
  `:2214`.
- The corresponding execute bodies correctly condition memory readback on
  `SC0` with `d->is_load = (inst_.sc0 != 0)`, for example
  `flat.cpp:1177` through `:1183` and `:2229` through `:2235`.
- The same generated classes are reused for GLOBAL via `SEG==2` mnemonic and
  operand shaping, so this metadata issue applies to both FLAT and GLOBAL
  atomics.

Impact:

Execution avoids the actual `VDST` write when `SC0=0`, but public def-use,
liveness, and race/metadata consumers see a false destination write for legal
no-return flat/global atomics.

### CDNA3-RJ-119: `V_MAC_F16` and `V_FMAC_F32` prohibited SDWA forms are accepted

Manual/XML evidence:

- Chapter 12.16.2 says SDWA cannot be used with `V_MAC_F16` and `V_FMAC_F32`
  at `cdna3/README.md:22619` through `:22638`.
- The checked-in XML matches that restriction for the listed rows: `V_MAC_F16`
  and `V_FMAC_F32` expose default, literal, DPP, and promoted VOP3 encodings,
  but no SDWA alternative, at `amdgpu_isa_cdna3.xml:56369` through `:56455`
  and `:59502` through `:59587`.
- `CDNA3-RJ-091` separately tracks literal-only `_MK`/`_AK` modifier
  over-acceptance, and `CDNA3-RJ-097` is limited to Chapter 12.16 VOP1
  exclusions.

Rocjitsu evidence:

- The generic CDNA3 VOP2 encoding helper treats `src0 == 249` as SDWA for every
  VOP2 opcode through `Vop2::has_sdwa()` and removes that selector from the
  default encoding at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/encodings.cpp:199` through
  `:207`.
- Generated `VMacF16Vop2` accepts the SDWA sentinel in its constructor and runs
  the shared SDWA preamble in execution at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop2.cpp:4587` through
  `:4649`.
- Generated `VFmacF32Vop2` has the same SDWA constructor and execution path at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop2.cpp:7879` through
  `:7934`.

Impact:

rocjitsu can decode and execute SDWA extension encodings for VOP2 rows that
the CDNA3 manual and XML both classify as non-SDWA instructions.

### CDNA3-RJ-120: 64-bit DPP controls are not restricted to `DPP_ROW*`

Manual/XML evidence:

- Chapter 13.3.9 says DPP can follow VOP1, VOP2, or VOPC instructions, but
  states that for 64-bit input data the only legal DPP type is `DPP_ROW*` at
  `cdna3/README.md:24155` through `:24199`.
- `CDNA3-XML-092` records that the checked-in XML exposes 64-bit DPP rows such
  as `V_MOV_B64` and `V_FMAC_F64`, but does not carry the `DPP_CTRL` subset
  restriction.
- LLVM agrees with the manual restriction on `gfx942`: `row_newbcast:1`
  assembles for `v_mov_b64` and `v_fmac_f64`, while `row_shr:1` and
  `quad_perm:[0,1,2,3]` are rejected with "DP ALU dpp only supports
  row_newbcast".

Rocjitsu evidence:

- Generated 64-bit DPP constructors copy `dp->dpp_ctrl` without classifying the
  control value, for example `VMovB64Vop1` and `VFmacF64Vop2` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop1.cpp:6763` through
  `:6788` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop2.cpp:571`
  through `:598`.
- The execute paths call the generic `amdgpu::dpp::apply_dpp()` without an
  operand-width or `DPP_CTRL` legality check at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop1.cpp:6801` through
  `:6817` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop2.cpp:616`
  through `:626`.
- The shared helper implements non-`DPP_ROW*` controls such as quad permute,
  row shifts, wave shifts, mirrors, and broadcasts, and unknown controls fall
  through to identity at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h:85`
  through `:199`.

Impact:

Invalid 64-bit DPP encodings can be decoded and routed into normal execution
instead of being rejected at decode or construction time. This is separate from
the `CDNA3-RJ-100` fully-illegal 64-bit VOPC DPP forms and from the
`CDNA3-RJ-121` valid-control data handling issue.

### CDNA3-RJ-121: 64-bit DPP source substitution only carries one dword

Manual/XML evidence:

- Chapter 13.3.9 allows `DPP_ROW*` for 64-bit input data at
  `cdna3/README.md:24196` through `:24199`.
- `V_MOV_B64` is a representative 64-bit VOP1 row with a DPP encoding and
  64-bit source/destination operands in the checked-in XML at
  `amdgpu_isa_cdna3.xml:48402` through `:48459`; LLVM accepts the legal
  `row_newbcast` form on `gfx942`.

Rocjitsu evidence:

- `VMovB64Vop1::execute_impl()` applies DPP, delegates `src0` to the resulting
  `DppOperand`, and then calls the shared 64-bit move helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop1.cpp:6801` through
  `:6857`.
- `apply_dpp()` reads only one VGPR register, stores one `uint32_t` per lane,
  and constructs `DppOperand` from that 32-bit lane array at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h:324`
  through `:348`.
- `DppOperand` implements 32-bit lane reads and a 32-bit SIMD storage view, but
  does not override `read_lane64()` or `simd_vgpr_storage64_impl()` at
  `lib/rocjitsu/src/rocjitsu/isa/operand.h:438` through `:485`; the base
  64-bit read throws at `lib/rocjitsu/src/rocjitsu/isa/operand.cpp:28`, and
  CDNA3 operand delegation forwards `read_lane64()` to that delegate at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/operand.cpp:1368` through
  `:1370`.

Impact:

Even a legal 64-bit DPP control such as `row_newbcast` cannot provide the full
64-bit source lane value through the current DPP proxy. Forced-scalar execution
falls into the missing `read_lane64()` path, and the SIMD path falls back to
the same delegate read when no 64-bit storage pair is available.

### CDNA3-RJ-122: Reserved FLAT `SEG=3` encodings are accepted

Manual/XML evidence:

- Chapter 10.1 defines `SEG` as memory segment `0=FLAT`, `1=SCRATCH`,
  `2=GLOBAL`, and `3=reserved` at `cdna3/README.md:3023` through `:3042`.
- Chapter 13.6 repeats the same segment set for the FLAT format at
  `cdna3/README.md:24447` through `:24448`.
- `CDNA3-RJ-117` covers illegal scratch atomic opcodes; this entry covers the
  separate reserved segment value.

Rocjitsu evidence:

- `Decoder::subDecodeFlat()` indexes the flat subdecode table only by `op.op`
  and does not validate `op.seg` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/decoder.cpp:5141` through
  `:5143`.
- `flat_mnemonic()` rewrites only `SEG==1` and `SEG==2`; other segment values
  keep the flat mnemonic at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/encodings.cpp:14` through
  `:20`.
- The shared flat address helper treats non-scratch and non-global segment
  values as the ordinary FLAT path at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h:113`
  through `:132`.

Impact:

Reserved `SEG=3` flat-family encodings can decode and execute as ordinary FLAT
instructions instead of being rejected.

### CDNA3-RJ-123: Global SGPR-base VGPR offsets are sign-extended

Manual/XML evidence:

- Section 10.4 says GLOBAL instructions may use `SGPR-address + VGPR-offset +
  instruction offset` and that the VGPR offset is 32 bits at
  `cdna3/README.md:3154` through `:3167`.
- Chapter 13.6 says the single `ADDR` VGPR is a 32-bit unsigned offset when
  `SADDR` is not `0x7f` for GLOBAL and SCRATCH forms at
  `cdna3/README.md:24452` through `:24456`.

Rocjitsu evidence:

- `flat_calculate_addresses()` reads one VGPR when `SADDR != 0x7f`, casts that
  value to `int32_t`, then sign-extends it to `uint64_t` before adding the
  SGPR base and signed instruction offset at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h:94`
  through `:106`.
- The no-gap note below previously described this sign extension as modeled
  segment behavior; the manual's field table makes the global VGPR offset
  unsigned.

Impact:

GLOBAL instructions that use an SGPR base and a VGPR offset with bit 31 set can
address below the base in rocjitsu, while the CDNA3 manual describes that VGPR
component as an unsigned 32-bit byte offset.

### CDNA3-RJ-124: Signed GLOBAL/SCRATCH immediate offsets disassemble as unsigned

Manual/XML evidence:

- Chapter 10.1 says scratch and global use a 13-bit signed byte offset, while
  flat uses a 12-bit unsigned offset with the MSB ignored, at
  `cdna3/README.md:3039`.
- Chapter 13.6 repeats the same offset signedness split at
  `cdna3/README.md:24435` through `:24439`.

Rocjitsu evidence:

- Runtime address calculation sign-extends the flat-family immediate through
  `sign_extend(inst.offset | (inst.pad_12 << 12), 13)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h:46`.
- `Flat::build_modifiers()` instead reconstructs the same 13 bits as an
  integer and prints it directly in the `offset:` modifier, without
  sign-extending global/scratch forms, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/encodings.cpp:296` through
  `:300`.

Impact:

Negative GLOBAL/SCRATCH immediate offsets can execute with the intended signed
address calculation but disassemble as large positive offsets, so text output
does not round-trip the encoded instruction semantics.

### CDNA3-RJ-125: Flat aperture routing samples only the first active lane

Manual/XML evidence:

- Chapter 10 says FLAT addresses can map to video/system memory, LDS, or
  scratch and that unmapped regions generate memory violations at
  `cdna3/README.md:3019` through `:3021`.
- Section 10.2 says FLAT internally uses both LDS and buffer paths and
  increments both `VM_CNT` and `LGKM_CNT` at `cdna3/README.md:3105` through
  `:3111`.
- Section 10.4 says GLOBAL instructions must not access LDS and return
  `MEM_VIOL` if they do at `cdna3/README.md:3154` through `:3167`.

Rocjitsu evidence:

- `ComputeUnitCore::route_memory_inst()` chooses a single `probe` address from
  the first active lane of a `GLOBAL_MEM` vector-memory state, then routes the
  entire instruction to LDS and rewrites all active lane addresses if that
  first address is in the shared aperture at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:260` through `:281`.
- The same routing path changes the instruction tag to `LOCAL_MEM` and changes
  the wait counter to `LGKMCNT`, which overlaps but does not fully model the
  dual-counter FLAT behavior recorded in `CDNA3-RJ-034`.

Impact:

Mixed-lane FLAT operations can be routed as all-LDS or all-global based on the
first active lane instead of per-lane aperture behavior. GLOBAL instructions
that hit the LDS aperture can be silently converted into LDS operations rather
than reporting the manual's memory-violation case.

### CDNA3-RJ-126: `XNACK_MASK_LO/HI` use ordinary SGPR storage

Manual evidence:

- The Chapter 3.1 state table defines `XNACK_MASK` as a 64-bit bit mask of
  threads that have failed address translation at `cdna3/README.md:362`
  through `:389`.

Rocjitsu evidence:

- Generated CDNA3 operand selector enums name `XNACK_MASK_LO` and
  `XNACK_MASK_HI` for scalar source and destination selector values 104 and
  105 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/operand_types.h:70`
  through `:71` and `:109` through `:110`.
- `resolve_src_scalar()` special-cases `FLAT_SCRATCH_LO/HI` at selector values
  102 and 103, but then sends all selector values `<= 105`, including 104/105,
  to ordinary SGPR storage at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/operand.cpp:1016` through
  `:1022`.
- `resolve_dst_write()` mirrors the same behavior: 102/103 update the
  dedicated scratch-base field, while selector values `<= 105` write ordinary
  SGPR storage at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/operand.cpp:1179` through
  `:1192`.
- A targeted search of `Wavefront`, register access, command-processor, CU, and
  memory-pipeline state found no CDNA3 `XNACK_MASK` state or address-translation
  failure mask updates.

Impact:

Kernels that read or write `XNACK_MASK_LO/HI` observe simulator SGPR contents
rather than the architectural per-lane address-translation-failure mask. Memory
paths also cannot set the mask on replayable address-translation failures, so
stateful XNACK behavior is absent even though the selector names decode.

### CDNA3-RJ-127: Raw STATUS launch and reuse initialization is incomplete

Manual evidence:

- Chapter 3.4 says STATUS fields are initialized when a wavefront is created,
  are readable by shader code, and include bits such as `EXECZ`, `VCCZ`,
  `IN_TG`, `IN_BARRIER`, `HALT`, `TRAP`, `VALID`, `SCRATCH_EN`, and `IDLE` at
  `cdna3/README.md:418` through `:448`.
- Chapter 3.3 defines `EXECZ` as the helper bit for zero EXEC, and Chapter
  3.9 says `VCCZ` updates whenever VCC updates at `cdna3/README.md:408`
  through `:416` and `:570` through `:588`.

Rocjitsu evidence:

- The shared CDNA STATUS wrapper exposes only part of the CDNA3 STATUS table,
  from `SCC` through `ALLOW_REPLAY`, and has no helpers for later manual bits
  such as `SCRATCH_EN` or `IDLE` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/cdna_isa_base.h:45`
  through `:81`.
- `ComputeUnitCore::allocate_wf()` initializes the wavefront PC, allocation
  records, EXEC, VCC, M0, apertures, scheduler state, and instruction count at
  launch, but does not initialize or rebuild raw STATUS at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:136` through `:150`.
- `Wavefront::reset()` explicitly says it does not change the status register,
  then resets dispatch state, EXEC, VCC, M0, MODE, apertures, counters, and
  scheduler state while leaving raw STATUS intact at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:459` through `:491`.
- The ISA-specific wavefront still exposes raw STATUS through `status_raw()`
  and initializes the stored value to zero only at object construction at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:608` through `:617`.
- The current CDNA3 HWREG read path can return that raw STATUS value for its
  handled status-like register path at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopk.cpp:291` through
  `:300`, separate from the HWREG selector-map mismatch in `CDNA3-RJ-048`.

Impact:

Fresh or reused wavefronts can expose raw STATUS bits that do not match the
manual's wave-creation state. The first launch sees zero for bits that should
reflect the created/running wave, and reused slots can preserve stale raw STATUS
bits across dispatches even as the execution state, EXEC, VCC, M0, and MODE are
reset. This is distinct from `CDNA3-RJ-051`, which tracks later
`EXECZ`/`VCCZ` drift after EXEC/VCC mutations.

## No-Gap Notes

- The generated CDNA3 VOP3A/VOP3B decoder table and constructors cover every
  named Chapter 12.11 instruction found in the checked-in XML. The new VOP3A
  runtime gaps above are about hard-stubbed execute bodies or detailed
  semantics, not missing decode entries.
- VOP3B opcode inventory and production execution cover the basic
  scalar-destination dataflow for the ten manual VOP3B opcodes: add/sub carry
  helpers write arbitrary `SDST`, carry-consuming helpers read source 2 as the
  carry-in mask, division-scale helpers write the scalar mask, and wide-MAD
  helpers write the explicit scalar carry destination. The remaining VOP3B
  issue in this slice is the VCC selector's analysis metadata in
  `CDNA3-RJ-107`.
- Literal extension handling matches the manual's "not VOP3" prose in the
  audited generated path: `Vop3` and `Vop3SdstEnc` keep size to one 64-bit
  instruction and do not consume a following literal extension word.
- The generated CDNA3 VOP1 decoder covers all manual Chapter 12.8 rows and
  leaves the manual opcode holes invalid, except for XML-only opcode 55 tracked
  in `CDNA3-RJ-094`. The VOP1 gaps above are about execution semantics,
  source drift, or illegal extension forms rather than missing manual-listed
  base opcodes.
- The generated CDNA3 VOPC decoder covers the full Chapter 12.9 opcode
  inventory: class opcodes 16-21, FP compare ranges 32-127, and integer
  compare ranges 160-255 are decoded, while the documented holes remain
  invalid in `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/decoder.cpp`.
- Ordinary non-class VOP3 `V_CMPX_*` rows already use the explicit SDST path in
  the generator and generated code; `CDNA3-RJ-098` is limited to
  `V_CMPX_CLASS_*` rows.
- VOPC E32 `CMPX` rows write VCC and EXEC in the generated execution path, and
  the generic DPP cleanup merges VCC/EXEC with the DPP write mask for masked
  lanes. The VOPC DPP findings above are about missing XML availability and
  64-bit legality, not absence of all DPP execution machinery.
- VOPC literal handling follows the expected format split in this static pass:
  E32 VOPC extends size for literal/DPP/SDWA sentinel encodings, while VOP3A
  compare constructors use no literal fixup and source operand classes exclude
  literal extension words.
- Outside the writeback/SDWA issues above, the audited generated class and
  relational predicate helpers distinguish sNaN/qNaN, infinities, normals,
  denormals, signed zero, unordered predicates, and negated-NaN compare cases
  in line with the detailed Chapter 12.9 formulas.
- Plain `V_SWAP_B32` VOP1/VOP3 dataflow is represented: generated classes mark
  both `VDST` and `SRC0` as read-write operands and the plain VOP3 executor
  performs the swap at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3.cpp:1841` through
  `:1863`. `CDNA3-RJ-097` is limited to illegal literal/DPP/SDWA extension
  forms on the VOP1 path.
- `V_CVT_OFF_F32_I4` is not currently a semantic-fallback gap: semantic
  derivation has a specific `vector_unary/cvt_off_f32_i4` override at
  `lib/python/amdisa/semantics.py:832`.
- FP8/BF8 VOP1 widening behavior for rows 84-87 remains covered by
  `CDNA3-RJ-076` and `CDNA3-RJ-078`; this VOP1 pass did not find an additional
  opcode-inventory miss for those rows.
- `V_CLREXCP` no-op execution is recorded under the broader trap/exception
  state gap `CDNA3-RJ-054`, not duplicated as a separate VOP1 finding.
- The generated CDNA3 VOP2 inventory covers the Chapter 12.7 opcode table at
  static-audit level: the decoder maps manual opcodes 0-21 and 23-61 and leaves
  the opcode 22 hole invalid, while the literal-only `_MK`/`_AK` VOP2 rows are
  not promoted to VOP3. The VOP2 gaps above concern execution semantics or
  illegal extension forms, not a missing base opcode inventory.
- VOP3 `V_MAC_F16` preserves the manual `OPSEL[3]` read/modify/write contract:
  generated execution reads the selected old destination half and writes back
  through `write_vop3_true16_dst(..., true)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3.cpp:2603` through
  `:2671`. The remaining F16 destination-half concerns are about XML
  recoverability and VOP2 literal/DPP legality, not this promoted true16 path.
- CDNA3 VOP2 carry/borrow helpers model the basic VOP2/VOP3 carry-out dataflow:
  `V_ADD_CO_U32` and `V_ADDC_CO_U32` write VCC or arbitrary VOP3 `SDST` forms,
  and the VOP3 add-with-carry path reads source 2 as the carry-in at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2889`
  through `:2935` and `:3257` through `:3308`. The existing clamp/saturation
  limitations remain covered by `CDNA3-RJ-074`.
- `V_LDEXP_F16` sign-extends the 16-bit exponent source before calling
  `std::ldexp`, matching the manual's signed-integer second operand at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:12460`
  through `:12482`.
- CDNA3 Chapter 1-2 dispatch, 64-lane wavefront, initial `EXEC`, and packed
  work-item-ID basics are represented by the production dispatch path and
  existing Chapter 3 no-gap notes below. The new Chapter 2 issue above is
  limited to the device-memory consistency and acknowledgment model; LDS
  clamping/bank behavior, GWS execution, launch TTMP/TG_SIZE state, barrier
  behavior, and workgroup-size validation remain covered by their existing
  narrower entries.
- CDNA3 currently does use FNUZ-specific helper names for FP8/BF8 conversion
  execution, so the main format-family split from OCP RDNA4/CDNA4 is present.
- CDNA3 generated SOP literal handling matches the broad audited format split
  for ordinary literal-capable SOP1/SOP2/SOPC rows, while SOPK's
  `S_SETREG_IMM32_B32` opcode 20 is the only SOPK path that consumes an
  extension dword. Runtime stores that extension word in `Sopk::literal_` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/encodings.cpp:68` through
  `:82`. `CDNA3-RJ-085` records the separate default-only SOP1 PC-form
  legality/size issue; the `S_SETREG_IMM32_B32` gap above is specifically about
  missing literal operand metadata/disassembly, not extension-word fetch.
- The CDNA3 generated SOP1 inventory matches the Chapter 12.3 manual/XML opcode
  inventory: constructors span the generated SOP1 file, the decode table maps
  opcode holes 47 and 49 to invalid, and generated smoke encodings cover the
  ordinary 32-bit rows at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sop1.cpp:23` through `:1058`,
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/decoder.cpp:6648` through
  `:6705`, and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/test_encodings.h:21`
  through `:75`. `CDNA3-RJ-085` records the separate legality issue for selected
  default-only PC encodings.
- Shared SOP1 helpers implement representative Chapter 12.3 unary and EXEC
  formulas at static-audit level: bit-count and bit-scan helpers are present at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:764`
  through `:791` and `:1648` through `:1729`, `S_ABS_I32` preserves the
  `0x80000000` wrap result and writes SCC at `:346` through `:354`,
  bitset/bitreplicate helpers cover the manual bit-index behavior at `:929`
  through `:970`, and saveexec/wrexec/quadmask helpers update EXEC, SDST, and
  SCC in the reviewed ranges at `:593` through `:651`, `:2267` through `:2304`,
  and `:2376` through `:2402`.
- The CDNA3 generated SOPC inventory matches the Chapter 12.4 manual/XML opcode
  inventory: constructors span `SCmpEqI32Sopc` through `SCmpLgU64Sopc`, the
  decode table maps opcodes 0 through 19 to those constructors, and generated
  smoke encodings cover the ordinary 32-bit rows at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopc.cpp:20` through `:478`,
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/decoder.cpp:6909` through
  `:6920`, and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/test_encodings.h:76`
  through `:95`.
- Shared SOPC compare and bit-compare helpers implement the audited Chapter
  12.4 integer formulas at static-audit level: bit compare masks source 1 to
  five bits for B32 and six bits for B64 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:897`
  through `:925`, while signed, unsigned, EQ, LG, GT, GE, LT, and LE compare
  helpers use the expected integer comparisons at `:1116` through `:1129`,
  `:1147` through `:1155`, `:1172` through `:1180`, `:1197` through `:1205`,
  `:1229` through `:1252`, and `:1269` through `:1277`.
- `S_SETVSKIP` is a SOPC row, but its missing execution semantics are already
  tracked under the program-control finding `CDNA3-RJ-052`; `S_SET_GPR_IDX_ON`
  has its broad M0 layout and role-mapping gap tracked in `CDNA3-RJ-075`, with
  the raw source-1 literal exception recorded separately in `CDNA3-RJ-086`.
- The CDNA3 generated SOPK inventory matches the Chapter 12.2 manual/XML opcode
  inventory after accounting for the literal-only opcode 20 form: constructors
  span `SMovkI32Sopk` through `SCallB64Sopk`, the decode table keeps opcode 19
  invalid and opcode 20 mapped to `SSetregImm32B32Sopk`, and generated smoke
  encodings cover the ordinary 32-bit SOPK rows at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopk.cpp:23` through `:407`,
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/decoder.cpp:6091` through
  `:6112`, and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/test_encodings.h:128`
  through `:147`.
- The shared SOPK helpers implement the Chapter 12.2 signed/unsigned SIMM16
  rules for `S_MOVK_I32`, `S_CMOVK_I32`, and `S_CMPK_*`: signed forms cast
  through `int16_t`, while unsigned compare forms zero-extend the 16-bit
  immediate at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1095`
  through `:1099`, `:1412` through `:1504`, and `:2033` through `:2037`.
- `S_ADDK_I32` and `S_MULK_I32` are not missing old-destination dataflow:
  generated constructors expose the destination as both source and destination,
  the shared helpers read the old scalar value before writeback, and adjacent
  tests assert def-use plus SCC behavior at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopk.cpp:234` through `:261`,
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:455`
  through `:465` and `:2093` through `:2098`, and
  `tests/scalar_scc_test.cpp:622` through `:695`.
- `S_CBRANCH_I_FORK` is a SOPK row, but its missing execution semantics are
  already tracked under the program-control finding `CDNA3-RJ-060`.
- `S_RFE_RESTORE_B64` is a CDNA3 SOP2 row in XML but not in the CDNA3 manual;
  the source drift is recorded as `CDNA3-XML-056`, and the generated runtime
  throw is already covered by `CDNA3-RJ-058`.
- Generated stochastic narrow converts read the old destination and merge the
  selected byte manually, matching the manual's unwritten-byte preservation rule
  for the audited forms. The packed 16-bit low-half preservation divergence is
  recorded separately as `CDNA3-RJ-129`.
- Generated packed shift helpers mask the selected shift-count half with
  `& 15`, matching the Chapter 12.10 use of `S0[3:0]` and `S0[19:16]` for the
  low and high components. Representative helpers are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:16805`
  through `:16834` and analogous LSHL/LSHR helpers.
- `V_PK_MOV_B32` uses `read_lane64()` for both sources and selects output dwords
  with `OPSEL[0]` and `OPSEL[1]` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:17295`
  through `:17312`, matching the CDNA3 manual's special `V_PK_MOV_B32`
  selector behavior for scalar pairs and VGPR gather.
- `V_ACCVGPR_READ` and `V_ACCVGPR_WRITE` have generated 32-bit lane-copy
  execution bodies under `EXEC` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3p.cpp:704` through
  `:731`. `CDNA3-RJ-103` is limited to the missing public instruction flag.
- `V_ACCVGPR_WRITE` uses the broad `OPR_SRC_NOLIT` source class in generated
  code, matching the checked-in XML and LLVM-accepted scalar/inline-constant
  source forms used by real kernels; this static pass did not record that as a
  runtime legality gap.
- MIX helpers implement the MIX-specific selector mapping, treat `NEG_HI` as
  absolute value, and apply `CLMP` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:13290`
  through `:13411`. They use multiply-add rather than fused FMA; the detailed
  instruction pseudocode and XML descriptions support multiply-add, while
  section 6.7 contains conflicting fused wording.
- Packed 32-bit helpers do not apply clamp or other output modifiers, matching
  the packed 32-bit statement in `cdna3/README.md:1527` that output modifiers
  are not supported for those instructions.
- Generated dense MFMA constructors implement the manual's register-bank
  controls by applying `ACC` to source A/B operands and `ACC_CD` to destination
  and source C operands; representative generated code is at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3p.cpp:738` through
  `:757`, and `mfma_src2_encoding()` applies the `ACC_CD` source-C rewrite at
  `:24` through `:33`.
- Shared dense MFMA layout helpers implement the manual's input and output
  lane/register formulas for 8/16/32/64-bit data at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:172` through
  `:225`.
- Dense MFMA helper paths apply ordinary `CBSZ`/`ABID` A-lane broadcast and
  `BLGP` B-lane permutation through `permute_a_lane()` and `permute_b_lane()` at
  `mma_exec.h:568` through `:610`; the gap above is only about missing legality
  validation.
- F64 dense MFMA repurposes `BLGP` as A/B/C negation in rocjitsu: generated F64
  calls pass `inst_.blgp` as the `neg` parameter at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3p.cpp:1973` through
  `:2024`, and `exec_f64()` applies bits `0x1`, `0x2`, and `0x4` to A, B, and C
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:3492`
  through `:3528`.
- Dense MFMA execution writes full-wave results through `RegisterAccess(cu)`
  rather than `Wavefront` EXEC-filtered writes, matching the manual's statement
  that MFMA ignores the execution mask. Representative F32/I32/F64 writes are
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:1488`
  through `:1489`, `:3070` through `:3071`, and `:3617` through `:3619`.
- Generated SMFMAC constructors model `SRC2` as an Arch VGPR operand and do
  not apply `ACC_CD` to it; `ACC_CD` only adjusts the generated destination
  operand. Representative constructors are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3p.cpp:1703` through
  `:1724` and `:2387` through `:2408`.
- F32-result SMFMAC helpers read the old destination block as the accumulator
  and write the destination block after buffering results, matching the
  accumulate-only C/D aliasing contract. Representative helper bodies are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:3696` through
  `:3726` and `:3846` through `:3883`.
- CDNA3 FP8/BF8 SMFMAC generated execution uses FNUZ readers, matching CDNA3's
  FP8/BF8 numeric-family split; representative generated calls are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3p.cpp:2425` through
  `:2426` and `:2509` through `:2510`.
- CDNA3 SMEM SBASE operand display and def-use scale the encoded SBASE field to
  an even SGPR pair, and `tests/smem_sbase_operand_test.cpp` covers that
  operand-level behavior.
- Generated CDNA3 SMEM disassembly preserves the immediate offset as an
  `offset:` modifier when both `IMM` and `SOFFSET_EN` are set, so the paired
  immediate is not lost by the disassembly path.
- Generated CDNA3 SMEM class and decoder inventory matches the XML `ENC_SMEM`
  records exactly: 84 generated constructors and 84 non-invalid
  `sub_decode_smem` entries cover the 82 manual Chapter 12.6 instruction
  definitions plus the two XML-only `S_ATC_PROBE*` opcodes recorded in
  `CDNA3-XML-060`. The decoder table places the manual
  load/store/cache/time/discard/atomic opcodes in their documented slots and
  leaves the holes invalid at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/decoder.cpp:7112` through
  `:7285`.
- Generated CDNA3 `S_ATC_PROBE` and `S_ATC_PROBE_BUFFER` decode from the XML,
  appear in the generated smoke encodings, and no-op in execution at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/smem.cpp:713` through
  `:744`, `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/test_encodings.h:229`
  through `:230`, and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:738`
  through `:743`. Because the CDNA3 manual does not list those opcodes, this
  audit records the mismatch as `CDNA3-XML-060` rather than a manual-derived
  runtime gap.
- Generated CDNA3 SMEM operand widths match the audited Chapter 12.6 data-width
  rows at static-audit level: load/store widths scale from dword through x16,
  buffer-resource `SBASE` operands are 128-bit while scalar/scratch bases are
  64-bit, and compare-swap atomics expose double-width `SDATA` operands in both
  32-bit and 64-bit families. The remaining SMEM gaps above are about address,
  descriptor, dependency, cache, and atomic execution behavior.
- Generated CDNA3 MUBUF/MTBUF constructors derive VADDR operand width from
  `IDXEN`/`OFFEN` with zero-, one-, or two-VGPR widths at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/mubuf.cpp:28` through
  `:35` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/mtbuf.cpp:27`
  through `:34`, matching the manual's address-VGPR table rather than the
  fixed-width XML operand entries.
- The shared MUBUF/MTBUF address helpers combine instruction offset and VGPR
  offset before adding the SGPR offset at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h:28`
  through `:37`, `:118` through `:120`, and `:202` through `:203`, so rocjitsu
  does not inherit the XML `OFFEN` wording that implies only one offset source.
- Basic raw CDNA3 MUBUF byte/short/dword load paths set element size, sign
  extension, D16 half selection, and SC/NT-derived memory type before issuing
  vector-memory operations; representative bodies are in
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/mubuf.cpp:476` through
  `:699` and `:1236` through `:1514`.
- For single-half raw D16 buffer loads, the vector-memory completion path models
  the CDNA3 D16 ECC note by preserving the untouched half when SRAM ECC is
  disabled and zeroing unused bits when it is enabled at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:181` through
  `:195`. Multi-component MTBUF D16 load packing is tracked separately in
  `CDNA3-RJ-131`.
- MUBUF integer atomic update formulas are implemented in the shared memory
  pipeline: compare-swap source/compare ordering and `INC`/`DEC` wrap behavior
  are handled by `apply_int_atomic()` and `execute_atomic_rmw()` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:246` through
  `:379`. The MUBUF runtime gaps above are about metadata and ACK/wait
  behavior, not the basic integer RMW formulas.
- The shared flat address helper models part of the main decoded segment split:
  unsigned 12-bit FLAT offsets, signed 13-bit GLOBAL/SCRATCH immediate
  offsets, 64-bit SGPR global bases, `saddr == 0x7F` VGPR-pair global
  addressing, scratch base plus lane stride, and optional scratch VGPR/SADDR
  offsets at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h:46`
  through `:132`. `CDNA3-RJ-123` records the separate unsigned global
  VGPR-offset issue.
- Rocjitsu partially models aperture routing: private-aperture FLAT addresses
  are remapped to scratch backing memory in the shared helper, and shared
  aperture addresses are routed to the LDS pipeline in the compute-unit router.
- Generated CDNA3 flat memory classes preserve D16 low/high-half load behavior,
  sign extension, ACC bank selection, `SC0` return selection for integer
  atomics, and `SC`/`NT` cache-flag plumbing in the audited representative
  bodies.
- FLAT/GLOBAL integer atomic update formulas are implemented in the shared
  memory pipeline: compare-swap source/compare ordering, signed/unsigned
  min/max, and `INC`/`DEC` wrap behavior are handled by `apply_int_atomic()`
  and `execute_atomic_rmw()` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:246` through
  `:379`. The flat-family runtime gaps above are about decode validity,
  metadata, wait counters, direct-LDS movement, and FP atomic edge behavior.
- Shared CDNA3 DS address paths cover the ordinary concatenated-offset form for
  single-address operations and the expected width/ST64 scaling in representative
  READ2/WRITE2 bodies. The remaining gaps are about M0 clamp, duplicate-offset,
  ADDTID, APPEND/CONSUME, and timing details.
- Generated narrow DS reads apply signed or zero extension for I8/U8/I16/U16
  forms in the audited representative bodies.
- DS_SWIZZLE, DS_PERMUTE, and DS_BPERMUTE execute through direct cross-lane
  helpers rather than the memory pipeline, matching the manual/XML statement
  that these forms do not write actual LDS memory. `CDNA3-RJ-111` and
  `CDNA3-RJ-112` record the remaining swizzle-mode and ACC-bank issues.
- APPEND/CONSUME have special active-count handling in the LDS atomic helper.
  `CDNA3-RJ-113` records the separate LDS address-base issue.
- Adjacent scalar SCC helpers reviewed in this slice match the CDNA3 manual at
  static-audit level: `S_MIN_{I32,U32}` uses strict `<`; `S_ADD*`/`S_SUB*`,
  `S_ADDC_U32`, `S_SUBB_U32`, and `S_LSHL{1,2,3,4}_ADD_U32` write carry or
  overflow through SCC; logical, shift, BFE, ABS, and ABSDIFF helpers write
  result-nonzero SCC; and `S_BFM_{B32,B64}` does not write SCC.
- CDNA3 access-instruction constructors and decoders do recognize
  `S_GETREG_B32`, `S_SETREG_B32`, and the 64-bit `S_SETREG_IMM32_B32` opcode,
  and the execute helpers implement the generic `{offset,size}` extraction or
  insertion around the handled raw register value. The access gaps above are
  about the register map, write legality/side effects, and literal operand
  visibility.
- Ordinary CDNA3 `EXECZ`/`VCCZ` branch and scalar-source consumers compute from
  the current `wf.exec()` and `wf.vcc()` values, so the raw STATUS gap above is
  not a claim that conditional branches themselves use stale helper bits.
- Logical CDNA3 scalar operands special-case VCC encodings 106/107 to
  `wf.vcc()`, and EXEC encodings 126/127 to `wf.exec()`. The remaining Chapter 3
  register-state gaps are about raw/physical state exposure, `XNACK_MASK`,
  allocation granularity, and out-of-range semantics rather than absence of all
  logical VCC/EXEC operand handling.
- CDNA3 M0 has a dedicated wavefront field and scalar operand handling:
  encoding 124 reads and writes `wf.m0()` in the CDNA3 operand helpers, and
  `S_SET_GPR_IDX_*` updates M0 and MODE state in the shared SOPP helper. The
  current M0 gaps are about DS/LDS/GWS semantics and selector side effects, not
  total absence of M0 storage.
- CDNA3 packed workitem ID initialization for VGPR0 is present in the
  command-processor launch path on packed-TID targets: `pack_workitem_id()`
  writes X/Y/Z into v0 bitfields for CDNA3/4 and GFX11+ at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:300` through
  `:321`. The launch-initialization gaps above are limited to TTMP4-11 and the
  TG_SIZE system SGPR payload.
- Ordinary CDNA3 PC-relative branches are modeled: `S_BRANCH` and the SCC,
  VCC, and EXEC conditional branches sign-extend the 16-bit label, scale it by
  four bytes, and update `wf.pc`; `S_CBRANCH_VCC*` masks VCC to wave size, and
  `S_CBRANCH_EXEC*` reads live EXEC at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopp.cpp:39` through
  `:200`.
- For default encodings, CDNA3 direct PC helper instructions are mostly modeled:
  `S_GETPC_B64` writes the next PC, `S_SETPC_B64` jumps to the scalar-pair
  address, and `S_SWAPPC_B64` writes the next PC to SDST while jumping to SSRC0
  at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sop1.cpp:543` through
  `:594`. `CDNA3-RJ-085` records the separate default-only literal-size
  acceptance issue, and `CDNA3-RJ-128` records the zero-target pre-execution
  halt special case.
- `S_ENDPGM` and `S_ENDPGM_SAVED` terminate the wave in rocjitsu through
  `wf.end()`, matching the base Chapter 4.1 termination behavior. The
  remaining saved-context signal side effects overlap with the broader
  trap/context-save gaps.
- Rocjitsu's classic `S_BARRIER` path does implement the core multi-wave
  scheduler release: the direct command-processor path creates every wave in a
  workgroup together and registers the expected count at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:700` through
  `:724`, the SPI path mirrors the same all-waves placement at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/spi.h:104` through `:116`,
  `Wavefront::halt()` decrements the workgroup refcount at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.cpp:15` through `:19`; and
  `ComputeUnitCore::update_wf_states()` releases barrier waves only when every
  non-halted wavefront in the same dispatch/workgroup is also at `BARRIER` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:313` through `:335`.
- `S_BARRIER` does not alter wait-counter targets or wait for counters to
  drain in the audited helper; this matches the CDNA3 manual's statement that
  barriers do not wait for counters to become zero before issuing. Explicit
  `S_WAITCNT` behavior remains separate in the generated SOPP wait-counter
  helper at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopp.cpp:234`
  through `:240`.
- The CDNA3 `S_WAITCNT` immediate decode matches the XML wait-count bit layout:
  rocjitsu combines `VM` low bits with `VM_HI`, extracts `EXP` and `LGKM`, and
  calls `wf.set_wait_target(vm, lgkm, exp)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopp.cpp:234` through
  `:240`. The Chapter 4.4 gaps above are about producer accounting, not this
  threshold decode.
- Generated CDNA3 SOPP constructors and the decoder table cover the manual
  opcode inventory for opcodes 0 through 29 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/sopp.cpp:20` through `:446`
  and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/decoder.cpp:6978`
  through `:7011`. `CDNA3-RJ-087` records the separate XML-only opcode 30 and
  31 exposure; `CDNA3-RJ-054`, `CDNA3-RJ-059`, `CDNA3-RJ-061`,
  `CDNA3-RJ-063`, `CDNA3-RJ-075`, and `CDNA3-RJ-087` cover the remaining
  trap/debug/status/perf/trace/barrier/GPR-index/XML-only SOPP behavior gaps.
- `HookOrderingTest.BarrierTwoWaves` gives a basic end-to-end check that a
  two-wave workgroup reaches and resolves one barrier event before both waves
  halt.
