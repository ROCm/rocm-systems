# CDNA4 Rocjitsu Gaps

Architecture: CDNA4

Manual source: `workspace_docs/amdgpu-isa-manuals/cdna4/README.md`

Rocjitsu source: `emulation/rocjitsu`

## Coverage

| Area | Status | Notes |
| --- | --- | --- |
| CDNA4 introduction and program organization | Audited statically | Checked Chapters 1-2 wave/workgroup/dispatch/SALU/VALU/EXEC/LDS/device-memory prose against ISA profiles, dispatch entry/command processor paths, wavefront/CU initialization, LDS backing, memory pipeline, and existing gap/no-gap notes. |
| CDNA4 kernel/wave state | Audited statically | Checked Chapter 3.1-3.6 PC/EXEC/STATUS/MODE state, helper bits, GPR/LDS allocation and aliasing, out-of-range behavior, and LDS allocation/clamping against `Wavefront`, generated HWREG accessors, operand reads/writes, and dispatch initialization. |
| CDNA4 special kernel state | Audited statically | Checked Chapter 3.7-3.13 M0/SCC/VCC/VCCZ, trap/exception/TRAPSTS/memory-violation state, HW_ID/XCC_ID, TTMP privilege/init, TG_SIZE system SGPR, and packed VGPR0 workitem ID against operand, SOPK/SOPC/SOPP, and command-processor paths. |
| CDNA4 program control and branching | Audited statically | Checked Chapter 4.1-4.2 control/branch tables, ordinary PC-relative branches, direct PC operations, trap return, debug conditional branches, call, sleep/wakeup/message/status helpers, Chapter 12/13 SOPP opcode inventory, and adjacent decode/execution tests. |
| CDNA4 workgroups and barriers | Audited statically | Checked Chapter 4.3 workgroup size limits and `S_BARRIER` release behavior against dispatch sizing, CU barrier state, status helpers, plugin hooks, and adjacent tests. |
| CDNA4 wait counters and wait states | Audited statically | Checked Chapter 4.4-4.5 `S_WAITCNT` decode, wait-counter data structures, memory-pipeline accounting, `S_SENDMSG` accounting, manual NOP hazard tables, and adjacent tests. |
| CDNA4 arbitrary divergent control flow | Audited statically | Checked Chapter 4.6 `S_CBRANCH_{I,G}_FORK`, `S_CBRANCH_JOIN`, CSP/branch-stack prose, generated instruction bodies, and shared execution helpers. |
| CDNA4 SALU scalar operand execution | Audited statically | Checked Chapter 5.1-5.2 scalar selector handling, literals, out-of-range source/destination behavior, 64-bit SGPR pair alignment, full SOP1/SOP2/SOPK/SOPC opcode/decode inventory, instruction-specific SOP1/SOP2 literal exclusions, the SOPK literal exception, the `S_SET_GPR_IDX_ON` raw-mode operand exception, and M0-relative move indexing against generated operands and the scalar operand resolver. |
| CDNA4 SALU SCC/M0 execution and metadata | Audited statically | Checked Chapter 5.3-5.7 SCC producer/consumer rules, arithmetic/compare/bitwise/absdiff helpers, `S_MAX` tie behavior, saveexec/wrexec handling, GPR-indexing M0 updates, C++ def-use metadata, and adjacent tests. |
| CDNA4 SALU access instructions | Audited statically | Checked Chapter 5.8 GETREG/SETREG/SETREG_IMM32 constructors, HWREG map handling, literal extension handling, wavefront state storage, SETREG spacing, and adjacent tests. |
| CDNA4 VALU opening semantics | Audited statically | Checked Chapter 6.1-6.6 VOP2/VOP3/VOPC operand classes, scalar/literal source restrictions, output modifiers, FP round/denorm mode, clamp-bit overloads, VGPR indexing, SDWA/OPSEL hazards, generated constructors, shared helpers, and adjacent tests. |
| CDNA4 VOP2 definition execution | Audited statically | Checked all Chapter 12.7 VOP2 generated opcode classes and VOP3 aliases, CNDMASK S2 condition handling, carry-out SDST/SRC2 forms, `V_MAC_F16` OPSEL[3] writeback, `V_LDEXP_F16` source sizing, literal-only `_MK`/`_AK` modifier handling, DOT2C BF16 VOP3 modifiers, FP min/max edge semantics, and adjacent tests. |
| CDNA4 VOP1 definition execution | Audited statically | Checked all detailed Chapter 12.8 VOP1 generated opcode classes and VOP3 aliases, `V_READFIRSTLANE_B32` EXEC handling and modifier acceptance, VOP1/VOP3 move modifiers, exception-state clearing, untyped bit operations, `V_SAT_PK_U8_I16`, XML-only opcode rows, `V_PRNG_B32`, permlane swap loops/dataflow, and adjacent tests. |
| CDNA4 VOPC definition execution | Audited statically | Checked all 198 generated VOPC classes and VOP3 compare aliases, compare/class helpers, literal/SDWA/DPP extension paths, 64-bit DPP exclusions, implicit VCC/EXEC side effects, and adjacent compare/class/DPP tests. |
| CDNA4 VOP3P definition execution | Audited statically | Checked the Chapter 12.10 VOP3P opcode inventory, packed 16-bit/F32 helpers, DOT, MIX, `V_PK_MOV_B32`, `V_ACCVGPR_*`, and adjacent generated SIMD hooks against manual prose, XML operands, and generated execution helpers. MFMA/SMFMAC execution remains covered by the separate MFMA rows. |
| CDNA4 VOP3A/VOP3B definition execution | Audited statically | Checked generated VOP3B scalar-destination classes and shared carry/div-scale/MAD helpers, VOP3A/VOP3B machine fields, VOP3_SDST modifier handling, VOP3A stub coverage, FMAC accumulator metadata, F32/F64 min/max MODE behavior, native F16 OPSEL destination preservation, `V_BITOP3_*` truth-table field overloads, `V_LSHL_ADD_U64` shift-count behavior, packed F32 conversion source modifiers, promoted conversion modifiers, `V_CVT_PKRTZ_F16_F32` rounding behavior, and `V_CVT_PKACCUM_U8_F32` destination dataflow against Chapter 12.11/13.3.4/13.3.5 manual prose and XML operands. |
| CDNA4 VOP3A definitions 640-659 execution | Audited statically | Checked F64 arithmetic/ldexp/min/max, integer multiply, readlane/writelane lane selectors, `V_BCNT`/`V_MBCNT`, 64-bit shifts, `V_TRIG_PREOP_F64`, and `V_BFM_B32`; new gaps cover `V_BCNT` base-add and lane-selector masking, while existing `CDNA4-RJ-110` covers the `V_TRIG_PREOP_F64` stub. |
| CDNA4 VOP3A definitions 660-673 execution | Audited statically | Checked packed normalized conversions, packed F32-to-F16 RTZ conversion, packed integer narrowing conversions, signed add/sub saturation notes, `V_PACK_B32_F16`, and `V_MUL_LEGACY_F32`; existing `CDNA4-RJ-110`, `CDNA4-RJ-080`, and `CDNA4-RJ-113` cover the observed runtime gaps. |
| CDNA4 VOP3A definitions 680-681 execution | Audited statically | Checked `V_MINIMUM3_F32` and `V_MAXIMUM3_F32` constructors, helper dispatch, SIMD hooks, and MODE/DX10_CLAMP handling; runtime gap remains `CDNA4-RJ-112`. |
| CDNA4 DPP/SDWA instruction limitations | Audited statically | Checked Chapter 12.16 DPP/SDWA exclusion lists against generated VOP1/VOP2/VOPC constructors, sentinel-based encoding helpers, and representative execution paths. Existing overlap entries cover `_MK`/`_AK`, `V_READFIRSTLANE_B32`, and VOPC DPP exclusions; remaining VOP1/VOP2 exclusion drift is tracked in `CDNA4-RJ-118`. |
| CDNA4 `V_CVT_{PK,SR}_{FP8,BF8}_F32` execution | Audited statically | Focused on definitions 674-677, helper behavior, `OPSEL` destination selection, destination preservation, operand classes, and runtime state dependencies. |
| CDNA4 FP8/BF8 widening conversion execution | Audited statically | Checked definitions 84-87, VOP1 SDWA byte/word selection and ignored-field behavior, VOP3 aliases, packed 64-bit destination writes, and adjacent generator/tests. |
| CDNA4 scaled FP8/BF8/FP4/FP6/BF6 conversion execution | Audited statically | Checked definitions 565-611, representative generated paths, codegen helper selection, wide packed FP6/BF6 layouts, source/destination `OPSEL`, scale extraction, stochastic PRNG advancement, and source/operand classes. |
| CDNA4 non-FP8 packed and stochastic conversion definitions | Audited statically | Checked Chapter 12 definitions 613-618 and 678-679 against generated constructors, execute bodies, semantic derivation, and adjacent tests. |
| CDNA4 dense MFMA layout and execution | Audited statically | Checked generated dense F32/F16/BF16/I8/F64 MFMA classes, ACC/ACC_CD operand rewriting, source/C resolution, shared layout formulas, broadcast helpers, full-wave access, state/control handling, and adjacent tests. |
| CDNA4 MFMA broadcast field handling | Audited statically | Checked `CBSZ`/`ABID` A-lane broadcast, `BLGP` B-lane permutation, generated dense MFMA field plumbing, F64 `BLGP`-as-negation override, and adjacent helper tests. |
| CDNA4 common F8F6F4 MFMA execution | Audited statically | Checked generated non-scale instruction classes, A/B format dispatch, C operand handling, execution-mask behavior, operand classes, and adjacent test coverage. |
| CDNA4 block-scaled F8F6F4 MFMA execution | Audited statically | Checked VOP3PX2 special decode, generated instruction classes, raw prefix preservation, `ABID[0]`, scale source handling, and shared scaled-MFMA helper behavior. |
| CDNA4 sparse MFMA execution | Audited statically | Checked generated SMFMAC constructors/execution stubs, shared sparse helpers, SRC2/ACC_CD handling, `CBSZ`/`ABID` sparse-index selection, index-pair/state/validation behavior, and adjacent test coverage. |
| CDNA4 MAI floating-point handling | Audited statically | Checked Section 7.4 denorm/MODE/RNE/exception rules against generated MFMA calls, shared MFMA helpers, MODE storage, and exception stubs. |
| CDNA4 MAI dependency waits | Audited statically | Checked Section 7.6 required NOP/independent-VALU table against generated instruction flags, wait-counter plumbing, and the instruction issue/execute path. |
| CDNA4 scalar memory execution | Audited statically | Checked generated SMEM loads/stores/cache/time/discard/atomics, XML-only ATC probe decode stubs, scalar address helpers including the CDNA4 scratch override, scalar memory pipeline, wait counters, `GLC`/`NV` handling, and adjacent tests. |
| CDNA4 vector buffer execution | Audited statically | Checked generated MUBUF/MTBUF opcode-table class inventory, constructors and execution paths, shared buffer address helpers, vector memory pipeline writeback, cache helpers, buffer atomics, Chapter 9.2 floating-atomic numeric rules for MUBUF/L2 forms, and adjacent tests. Chapter 9 prose sections 9.1 and 9.2 are fully covered by this row; open rocjitsu and test gaps are recorded below. |
| CDNA4 flat/global/scratch execution | Audited statically | Checked generated FLAT/GLOBAL/SCRATCH opcode-table coverage through the shared `SEG`-driven FLAT class set, constructors and execution paths, shared flat address helper, aperture routing, direct-to-LDS decode gaps, flat atomics, floating atomics, wait counters, and adjacent tests. |
| CDNA4 data-share LDS execution | Audited statically | Checked generated DS opcode-table class/decode inventory, load/store/atomic bodies, shared DS address helpers, local-memory pipeline completion, ADDTID, READ2/WRITE2, swizzle/permute helpers, GDS handling, packed LDS atomics, transpose loads, and adjacent tests. |
| CDNA4 small-format conversion tests | Audited statically | Checked helper and HIP conversion tests plus gfx950 fpsan skip metadata; uncovered overflow-mode, scale, and FP4/FP6/BF6 edge cases are recorded below. |
| CDNA4 scalar memory tests | Audited statically | Checked the SBASE operand regression and adjacent RDNA SMEM address tests; uncovered CDNA4 selector, M0, scratch, buffer-resource, atomic, cache-policy, and counter cases are recorded below. |
| CDNA4 vector buffer tests | Audited statically | Checked operand/codegen regressions, LDS disassembly coverage, and real CDNA4 buffer-to-LDS kernel fixtures; uncovered descriptor, format, SOFFSET, alignment, LDS clamping, atomic dataflow, floating-atomic numeric, and cache-policy cases are recorded below. |
| CDNA4 flat/global/scratch tests | Audited statically | Checked shared CDNA4 address-form tests and generic flat VM fixtures; uncovered `PTR32`/`ADDRESS_MODE`, direct-to-LDS, FLAT dual-counter ordering, aperture-hole, floating-atomic, and `NV` ambiguity cases are recorded below. |
| CDNA4 data-share tests | Audited statically | Checked DS atomic stress, DS transpose ACC routing, swizzle codegen, and semantics-derivation coverage; uncovered M0 clamping, ADDTID, duplicate-offset READ2/WRITE2, swizzle FFT/rotate, packed LDS atomics, transpose constraints/layouts, and GDS ambiguity cases are recorded below. |
| CDNA4 rocjitsu surface through Chapter 13 | Complete | Coverage rows above cover the CDNA4 manual chapters and instruction-definition families; detailed entries below record the remaining decode, execution, metadata, disassembly, DBT, and test gaps found in this pass. |

## Gaps

### CDNA4-RJ-001: Non-scaled FP8/BF8 converts cannot model `FP16_OVFL=1`

Manual evidence:

- `cdna4/README.md:2402` through `:2416` defines separate F32 to FP8/BF8
  results for `FP16_OVFL=0` and `FP16_OVFL=1`.

Rocjitsu evidence:

- Generated `V_CVT_PK_FP8_F32` and `V_CVT_PK_BF8_F32` call fixed OCP helpers at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:11327` through
  `:11332` and `:11360` through `:11365`.
- Generated stochastic FP8/BF8 converts call fixed OCP helpers at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:11393` through
  `:11401` and `:11429` through `:11437`.
- None of these paths reads MODE state or passes an `FP16_OVFL` flag.

Impact:

CDNA4 emulation has the same state-dependent overflow gap as the RDNA4
non-scaled FP8/BF8 conversion paths.

### CDNA4-RJ-002: Scaled conversion-to-F8 paths also ignore `FP16_OVFL`

Manual evidence:

- `cdna4/README.md:2378` through `:2385` says conversion to F8 supports
  `FP16_OVFL` and that scale is an E8M0 exponent with bias 127.

Rocjitsu evidence:

- Generated scaled F32-to-FP8/BF8 paths extract the scale exponent at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:5790` through
  `:5797` and `:5842` through `:5849`, then call fixed OCP helpers at `:5804`
  through `:5805` and `:5856` through `:5857`.
- Generated scaled F32 stochastic paths call fixed OCP helpers at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:5894` through
  `:5911`.
- Generated F16-to-FP8/BF8 scaled paths call fixed OCP helpers at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:6268` through
  `:6295` and `:6318` through `:6345`.
- Codegen chooses these helper names through the unconditional scaled mapping at
  `lib/python/amdisa/codegen/execute/vector_special.py:1735` through `:1758`
  and emits the calls in `_gen_narrow_scalef32` beginning at `:1861`.

Impact:

Scaled CDNA4 conversion-to-F8 execution cannot produce the manual's
`FP16_OVFL=1` max-value outcomes.

### CDNA4-RJ-003: `SH_MEM_CONFIG[8]` is not modeled for BF8/FP8 operations

Manual evidence:

- `cdna4/README.md:2418` says `SH_MEM_CONFIG` bit 8 must be set for correct
  BF8/FP8 operation results.

Rocjitsu evidence:

- The audited generated conversion bodies read operands and destination state,
  but no scalar memory configuration state, before invoking FP8/BF8 helpers.

Impact:

Rocjitsu currently cannot distinguish configured and misconfigured BF8/FP8
operation behavior for this slice.

### CDNA4-RJ-004: Generated source operands allow forms the manual says are illegal

Manual evidence:

- `cdna4/README.md:2406` says `CVT_SR_*` and `CVT_PK_*` support only VGPR
  inputs, not SGPRs, literal constants, or inline constants.

Rocjitsu evidence:

- Generated CDNA4 constructors use `OPR_SRC_NOLIT` and `OPR_SRC_SIMPLE` for
  non-scaled `V_CVT_PK_FP8_F32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:11303` through
  `:11308`.
- The same broad operand classes appear in scaled FP8/BF8 constructors, for
  example `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:5764`
  through `:5770` and `:5868` through `:5874`.
- The underlying CDNA4 operand classes include scalar-register and inline-source
  subtypes in the generated operand metadata.

Impact:

Rocjitsu inherits the XML/manual legality mismatch for both non-scaled and
scaled conversion sources. Hardware and LLVM oracle behavior still need a
focused decision pass before turning this into an enforcement change.

### CDNA4-RJ-005: Tests lock in fixed helper behavior, not state-dependent behavior

Rocjitsu evidence:

- Shared helper tests assert OCP FP8 overflow/Inf behavior at
  `tests/data_types_test.cpp:274` through `:331` and BF8 Inf behavior at
  `:493` through `:504`.
- HIP conversion tests use finite values and helper-derived expectations for
  non-scaled FP8/BF8 at `tests/hip_cvt_narrow_test.cpp:216` through `:309` and
  scaled FP8/BF8 at `:314` through `:445`.
- The NaN/Inf scaled edge test in that file covers FP4, not FP8/BF8, at
  `tests/hip_cvt_narrow_test.cpp:614` through `:630`.
- The scaled FP8/BF8 tests cover default finite cases and one FP8 byte-select
  path at `tests/hip_cvt_narrow_test.cpp:314` through `:445`, but not BF8
  byte selection, high-half destination preservation, ignored SDWA fields, or
  64-bit destination-pair alignment.

Impact:

Existing tests exercise finite conversion plumbing and the current fixed helper
mode, but would not fail if CDNA4 continued to ignore `FP16_OVFL` for FP8/BF8
edge cases or regressed the less common selector, preservation, and legality
contracts.

### CDNA4-RJ-006: FP8/BF8 forwarding hazard is not modeled

Manual evidence:

- `cdna4/README.md:2360` requires spacing between `CVT_*_F32` conversions that
  write low/high halves or bytes of the same destination register.

Rocjitsu evidence:

- The generated execution bodies are functional per-instruction operations and
  do not track a 4-cycle forwarding hazard for these conversions.

Impact:

Rocjitsu can execute sequences that the manual requires software to space on
hardware. This is a timing/scheduling gap rather than a single-instruction
decode bug.

### CDNA4-RJ-007: Packed F32 arithmetic broadcasts scalar sources instead of reading 64-bit pairs

Manual/XML evidence:

- CDNA4 section 6.7 says packed 32-bit instructions operate on two dwords at a
  time and that `OPSEL`/`OPSEL_HI` select the first or second dword for each
  source at `cdna4/README.md:1571`.
- `V_PK_FMA_F32`, `V_PK_MUL_F32`, and `V_PK_ADD_F32` read source dwords
  `[31:0]` and `[63:32]` in the detailed pseudocode at
  `cdna4/README.md:12190` through `:12217`.
- The CDNA4 XML encodes these sources as 64-bit `FMT_NUM_PK2_F32` operands at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna4.xml:74332` through
  `:74449`.

Rocjitsu evidence:

- The generated CDNA4 constructors preserve the 64-bit operand sizes for
  `V_PK_FMA_F32`, `V_PK_MUL_F32`, and `V_PK_ADD_F32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:722` through
  `:771`.
- The shared generator reads a 64-bit pair only when the encoding value is a
  VGPR, otherwise it initializes the high word from the low 32-bit read at
  `lib/python/amdisa/codegen/execute/packed.py:19` through `:38`.
- The generated shared helpers follow that pattern for `V_PK_ADD_F32`,
  `V_PK_FMA_F32`, and `V_PK_MUL_F32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:16709`
  through `:16722`, `:16897` through `:16917`, and `:17361` through `:17373`.
- The CDNA4 operand layer already has scalar-pair support through
  `read_lane64()` and `resolve_src_scalar64()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/operand.cpp:1443` through
  `:1468` and `:1690` through `:1703`, and `V_PK_MOV_B32` uses that path
  unconditionally.

Impact:

A scalar-pair source such as `s6:s7` with different low/high dwords feeds the
low dword to both packed F32 components in rocjitsu. The manual and XML both
describe these operands as packed 64-bit pairs, so the high component should
come from the second dword of the pair before `OPSEL_HI` selection.

### CDNA4-RJ-008: Packed 32-bit VOP3P VGPR pair alignment is not validated

Manual evidence:

- CDNA4 section 6.7 says packed 32-bit operands must be two-dword aligned, with
  an even VGPR address, at `cdna4/README.md:1571`.
- The `V_PK_MOV_B32` notes repeat that sources are 64-bit operands subject to
  alignment restrictions for both SGPR and VGPR at `cdna4/README.md:12231`
  through `:12245`.

Rocjitsu evidence:

- The packed F32 and `V_PK_MOV_B32` constructors use raw 64-bit VGPR/source
  operands at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:722`
  through `:788`.
- `Isa::resolved_vgpr_offset()` accepts any source VGPR encoding from 256
  through 511 and returns the unadjusted VGPR index at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/operand.cpp:1627` through
  `:1633`.
- `Operand::read_lane64()` and `write_lane64()` then read or write the pair
  starting at that index at `operand.cpp:1690` through `:1713`.

Impact:

Odd VGPR encodings are treated as valid pairs rooted at the odd register,
whereas the manual restricts packed 32-bit operands to even-aligned pairs.

### CDNA4-RJ-009: Packed F16 VOP3P arithmetic ignores `CLMP`

Manual/XML evidence:

- The CDNA4 VOP3P format includes `CLMP` as "1 = clamp result" at
  `cdna4/README.md:25990` through `:25998`.
- The CDNA4 XML describes generic VOP3P `CLAMP` as clamping output to
  `[0.0, 1.0]` at `shared/machine-readable-isa/isa/amdgpu_isa_cdna4.xml:2212`
  through `:2213`.

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

### CDNA4-RJ-010: Scaled conversions ignore MODE-based F32 denormal control

Manual evidence:

- `cdna4/README.md:2381` says conversion from F32 supports MODE-based denormal
  control, while F4/F6/F8 allows denorms regardless of MODE.

Rocjitsu evidence:

- Generated scaled F32-to-FP4 code reads raw F32 sources and directly divides
  them by the decoded scale before calling the FP4 helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:6132` through
  `:6147`.
- Generated wide F32-to-FP6 code follows the same raw-F32 pattern at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:7117` through
  `:7127`.
- Generated FP6-to-F32 code multiplies the converted small-format value by the
  decoded scale and writes the result without consulting `wf.mode_raw()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:8001` through
  `:8017`.
- `Wavefront` stores raw MODE state at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:103` through `:110`, but
  the shared `s_denorm_mode` executor is currently a no-op at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1644`
  through `:1645`, and the audited CDNA4 scaled conversion bodies do not read
  MODE.

Impact:

F32 subnormal inputs and results follow host floating-point behavior regardless
of the shader MODE denormal settings the manual says should affect F32
conversion.

### CDNA4-RJ-011: Scaled conversion scale operands allow SGPR/literal forms the manual excludes

Manual evidence:

- `cdna4/README.md:2385` says the scale can come from a VGPR or an inline
  constant, using the float exponent portion.

Rocjitsu evidence:

- Generated scale operands use `OperandType::OPR_SRC_SIMPLE` in representative
  scaled conversion constructors, including FP4 F32 narrowing at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:6106` through
  `:6112`, wide FP6 narrowing at `:7089` through `:7095`, stochastic FP6
  narrowing at `:7500` through `:7505`, and FP6 widening at `:7975` through
  `:7980`.
- The CDNA4 operand layer resolves `OPR_SRC_SIMPLE` scalar-register encodings
  as SGPRs at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/operand.cpp:1189`
  through `:1195` and VGPR encodings at `:1196` through `:1201`; the XML
  source class also includes scalar-register source classes at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna4.xml:127127` through
  `:127138`.

Impact:

Rocjitsu inherits the XML source-class broadness for scale operands. A scale
encoded as an SGPR or literal can be decoded and executed even though the manual
limits the scale source to VGPR or inline constant forms.

### CDNA4-RJ-012: Wide scaled/packed conversion VGPR alignment is not validated

Manual/XML evidence:

- CDNA4 section 6.7.1 says convert opcodes operating on FP6/BF6/FP4 data must
  use VGPR sources for operand slots providing more than 32 bits of data at
  `cdna4/README.md:1575`.
- The CDNA4 XML's `OPR_SRC_VGPR` class says 64-bit and wider VGPR values must
  be even-aligned at `shared/machine-readable-isa/isa/amdgpu_isa_cdna4.xml:129556`
  through `:129564`.
- The CDNA4 XML's plain `OPR_VGPR` class repeats the same alignment rule for
  64-bit and wider VGPR destinations at `amdgpu_isa_cdna4.xml:140980` through
  `:140987`.
- Table 31 says `CVT_PK_F32_FP8` and `CVT_PK_F32_BF8` write `dst,dst+1` and
  require an even destination at `cdna4/README.md:2320` through `:2323`; their
  instruction definitions write `D0[31:0]` and `D0[63:32]` at
  `:9313` through `:9343`.
- The scaled FP8/BF8 packed widening definitions also produce 64-bit F32 pairs
  at `cdna4/README.md:16662` through `:16688`.

Rocjitsu evidence:

- Generated scaled conversion constructors preserve 64-bit and wider VGPR data
  operands, for example `V_CVT_SCALEF32_SR_PK_FP4_F32` uses a 64-bit VGPR
  source at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:6156`
  through `:6162`, `V_CVT_SCALEF32_2XPK16_FP6_F32` uses 192/512-bit VGPR
  operands at `:7089` through `:7095`, and `V_CVT_SCALEF32_PK32_F32_FP6` uses
  1024/192-bit VGPR operands at `:7975` through `:7980`.
- Generated `V_CVT_PK_F32_FP8/BF8` VOP1/VOP3 forms and
  `V_CVT_SCALEF32_PK_F32_FP8/BF8` construct 64-bit `VDST` operands and write via
  `write_lane64()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop1.cpp:10474` through
  `:10582`, `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:2025`
  through `:2078`, and `:5962` through `:6038`.
- `Isa::resolved_vgpr_offset()` accepts any VGPR encoding from 256 through 511
  and returns the unadjusted register index at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/operand.cpp:1627` through
  `:1633`.
- `Operand::read_lane64()` and `write_lane64()` then read or write pairs rooted
  at that returned index at `operand.cpp:1690` through `:1713`; the wide
  generated conversion paths use the same unvalidated root index in their
  multi-dword VGPR access lambdas.

Impact:

Odd VGPR encodings can root 64-bit packed F32 results or 64-bit, 192-bit,
512-bit, and 1024-bit scaled conversion operands in rocjitsu, despite the XML
operand-class alignment rule for 64-bit and wider VGPR values.

### CDNA4-RJ-013: FP4/FP6/BF6 scaled-conversion tests miss state and legality edges

Rocjitsu evidence:

- HIP FP4 tests round-trip exact FP4 values with scale 1.0 at
  `tests/hip_cvt_narrow_test.cpp:452` through `:518`.
- HIP FP6/BF6 tests round-trip exact representable values with scale 1.0 at
  `tests/hip_cvt_narrow_test.cpp:525` through `:592`.
- The explicit scale edge test covers scaled FP8 only at
  `tests/hip_cvt_narrow_test.cpp:597` through `:611`; the NaN/Inf edge test
  covers FP4 finite helper behavior at `:614` through `:630`.
- The gfx950 corpus skip list still skips the fpsan scaled FP4/FP6/SR tests at
  `tests/corpus/gfx950_skip_tests.json:8` through `:10`.

Impact:

The current tests cover useful finite plumbing and helper behavior, but would
not catch the MODE-denorm, scale-source legality, wide-VGPR alignment, E8M0
`0xff`, or multi-pass stochastic edge cases recorded above.

### CDNA4-RJ-014: `V_MFMA_SCALE_*_F8F6F4` decodes as non-scale MFMA operands

Manual/XML evidence:

- The CDNA4 manual lists
  `V_MFMA_SCALE_F32_16X16X128_F8F6F4` and
  `V_MFMA_SCALE_F32_32X32X64_F8F6F4` as distinct four-dword scale MFMA
  instructions at `cdna4/README.md:2246` through `:2254`.
- The XML entries expose `SCALE_SRC0` and `SCALE_SRC1` as explicit operands for
  those opcodes at `shared/machine-readable-isa/isa/amdgpu_isa_cdna4.xml:77932`
  through `:78047`.

Rocjitsu evidence:

- The generated CDNA4 decoder recognizes the four-word VOP3PX2 scale form, but
  returns the non-scale `VMfmaF3216x16x128F8f6f4Vop3pMfma` or
  `VMfmaF3232x32x64F8f6f4Vop3pMfma` class with `opcode + 2` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/decoder.cpp:18` through
  `:37`.
- The code generator emits the same special-case mapping at
  `lib/python/amdisa/codegen/_generator.py:2353` through `:2374` and
  `:9336` through `:9348`.
- The generated classes only store `vdst`, `src0`, `src1`, `src2`, and
  `raw_words_` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.h:377` through
  `:397`.
- Constructors use the non-scale mnemonics and publish only three source
  operands at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:834`
  through `:867` and `:898` through `:930`.

Impact:

Execution can still consult raw prefix words, but instruction metadata,
disassembly, operand iteration, and tooling do not see the scale mnemonic or
explicit `SCALE_SRC0`/`SCALE_SRC1` dependencies.

### CDNA4-RJ-015: Block-scale byte selectors are ignored

Manual evidence:

- The load-scale prefix uses `{OP_SEL_HI[0], OP_SEL[0]}` for matrix A scale and
  `{OP_SEL_HI[1], OP_SEL[1]}` for matrix B scale, selecting one of the four
  source bytes, at `cdna4/README.md:2265` through `:2275`.

Rocjitsu evidence:

- The 16x16 scale path extracts only the raw scale source encodings from the
  prefix and passes scale register bases to the shared helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:886` through
  `:891`.
- The 32x32 scale path does the same at `vop3p.cpp:950` through `:955`.
- `exec_f32_scaled_mixed` accepts only `scale_a_base` and `scale_b_base`, then
  reads the low byte of those VGPR values at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:3028` through
  `:3064`. It has no selector arguments and does not inspect the VOP3PX2 prefix
  `OP_SEL`/`OP_SEL_HI` bits.

Impact:

Scale encodings selecting byte 1, 2, or 3 are emulated as byte 0.

### CDNA4-RJ-016: Inline-constant scale sources are not modeled for block-scale MFMA

Manual evidence:

- Block-scale MFMA scale values can be VGPRs or inline constants, using only
  the exponent portion, at `cdna4/README.md:2276`.

Rocjitsu evidence:

- The scaled execution paths derive `sa_base` and `sb_base` by passing the raw
  9-bit prefix source encodings to `amdgpu::src_base` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:886` through
  `:887` and `:950` through `:951`.
- `src_base` is an MFMA VGPR/AccVGPR base resolver; it maps encodings to
  VGPR-bank offsets and has no inline-constant decoding path at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:86` through
  `:92`.
- The shared scaled helper then reads VGPR state for the scale values at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:3059` through
  `:3062`.

Impact:

An inline-constant scale encoding is treated as a VGPR/AccVGPR base rather than
as a constant E8M0 exponent source.

### CDNA4-RJ-017: F8F6F4 MFMA C modifiers are ignored

Manual evidence:

- The detailed definitions for `V_MFMA_F32_16X16X128_F8F6F4` and
  `V_MFMA_F32_32X32X64_F8F6F4` say `NEG[1:0]` and `ABS[1:0]` must be zero,
  while `NEG[2]` and `ABS[2]` may control matrix C, at
  `cdna4/README.md:12121` through `:12134` and `:12154` through `:12172`.

Rocjitsu evidence:

- The generated constructors expose only `vdst`, `src0`, `src1`, and `src2`
  operands for the non-scale F8F6F4 MFMA classes at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:834` through
  `:867` and `:898` through `:930`.
- Their execution paths read `SRC2` or a constant accumulator and dispatch to
  `exec_f32_mixed` / `exec_f32_scaled_mixed` without inspecting `inst_.neg` or
  `inst_.neg_hi` and without passing a C modifier at `vop3p.cpp:873` through
  `:891` and `:937` through `:955`.
- `exec_f32_mixed` seeds the accumulator directly from `SRC2` or `const_acc`
  and has no `c_modifier` parameter at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:1154` through
  `:1192`.
- The shared WMMA F32 path has an explicit `apply_wmma_c_modifier` pattern at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:100` through
  `:107` and applies it in `exec_wmma_f32_mixed` at `:1492` through `:1545`,
  so this is not a missing primitive.

Impact:

Encodings that request negation or absolute value of matrix C execute as if the
modifier bits were clear.

### CDNA4-RJ-018: F8F6F4 MFMA tests miss the direct instruction surface

Rocjitsu evidence:

- Searching the checked-in tests for `V_MFMA_F32_16X16X128_F8F6F4`,
  `V_MFMA_F32_32X32X64_F8F6F4`, and their lower-case mnemonics returns no
  direct test hits.
- Existing CDNA FP8 helper coverage checks CDNA3 FP8/BF8 extract consistency at
  `tests/shared_infra_test.cpp:442` through `:463`, and codegen tests check
  CDNA3/CDNA4 FP8 MFMA helper variant selection for older FP8/BF8 shapes at
  `lib/python/amdisa/tests/test_generator_profile_gates.py:963` through
  `:983`.
- The F8F6F4 decode tests in this area are for gfx1250 WMMA, not CDNA4 MFMA, at
  `tests/gfx1250_sim_test.cpp:3219` through `:3261`.
- The only explicit scaled MFMA benchmark calls the shared helper directly
  rather than decoding/executing a CDNA4 instruction at
  `tests/mfma_simd_benchmark.cpp:502` through `:521`.
- The gfx950 corpus still skips MFMA fpsan suites at
  `tests/corpus/gfx950_skip_tests.json:6` and `:17` through `:18`.

Impact:

The current suite would not catch the C-modifier gap above, `CLAMP`/manual
ambiguity decisions, direct decode/disassembly issues for non-scale F8F6F4
MFMA, or illegal/edge format-control encodings.

### CDNA4-RJ-019: Dense MFMA register-block alignment is not validated

Manual evidence:

- Section 7.1 says MFMA input/output register blocks must be contiguous and the
  first register must be aligned to the number of registers required by that
  operand at `cdna4/README.md:1647` through `:1650`.
- The dense MFMA rule table also says `SRC0`, `SRC1`, `SRC2`, and `VDST` need
  VGPR alignment at `cdna4/README.md:1721` through `:1738`.

Rocjitsu evidence:

- Generated dense MFMA constructors expose large register ranges from the raw
  encoded base without legality checks, for example the 32-register F32
  destination and accumulator for `V_MFMA_F32_32X32X1_2B_F32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:1338` through
  `:1371`, and the 64-bit F16 source ranges for
  `V_MFMA_F32_32X32X4_2B_F16` at `:1675` through `:1708`.
- Execution maps encoded bases through `dst_base()` / `src_base()` and passes
  them directly into shared helpers, for example the F32 path at
  `vop3p.cpp:1365` through `:1374` and the F16 path at `:1702` through
  `:1711`.
- `dst_base()` and `src_base()` only translate VGPR versus AccVGPR numbering at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:73` through
  `:92`; the layout helpers compute offsets from that base at `:172` through
  `:229`; and the contiguous-region read/write helpers size regions but do not
  validate base alignment at `:1013` through `:1063`.

Impact:

Illegal dense MFMA encodings with odd or under-aligned source, accumulator, or
destination bases execute against shifted register blocks instead of being
rejected or diagnosed. This is broader than the packed 32-bit VOP3P alignment
gap because dense MFMA operands can require 4, 16, or 32 contiguous registers.

### CDNA4-RJ-020: Dense MFMA clamp and overflow state is not modeled

Manual evidence:

- The dense MFMA rule table, excluding F8F6F4, says clamp is supported, uses
  `FP16_OVFL` from MODE, clamps F32 overflow to `+/-MAX` when set and otherwise
  produces `+/-INF`, and clamps I32 overflow/underflow to `+/-MAX` when set
  and otherwise drops carry-out bits at `cdna4/README.md:1721` through `:1738`.

Rocjitsu evidence:

- Generated dense MFMA execution paths pass source/destination bases,
  `const_acc`, `CBSZ`, `ABID`, and `BLGP` into shared helpers, but do not read
  MODE, `FP16_OVFL`, or any clamp-control bit; representative F32/F16 calls are
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:1365` through
  `:1374` and `:1880` through `:1890`.
- Dense I8 MFMA generated paths call `exec_i32_i8()` without a clamp argument,
  for example at `vop3p.cpp:2056`, `:2100`, `:2144`, `:2317`, and `:2361`.
- The shared integer helper has a clamp-capable primitive
  `exec_i32_mixed(..., bool clamp = false, ...)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:1424` through
  `:1487`, but the dense I8 MFMA specialization documents that there is no
  clamp on its path and wraps in 32 bits at `:4377` through `:4392`.

Impact:

If the manual's dense MFMA clamp/`FP16_OVFL` row is architecturally meaningful,
rocjitsu currently executes those cases as unclamped, mode-independent MFMA.
The VOP3P-MAI field table does not expose a `CLAMP` bit, so this likely needs
manual/assembler reconciliation before a precise emulator contract can be
implemented.

### CDNA4-RJ-021: MFMA broadcast field legality is not validated

Manual evidence:

- Section 7.1.6.1 says the largest legal `CBSZ` value is 4, `(1 << CBSZ)` must
  not exceed the number of blocks the MFMA instruction processes, and `ABID`
  must be less than `(1 << CBSZ)` at `cdna4/README.md:2164` through `:2184`.

Rocjitsu evidence:

- Generated dense MFMA execution passes raw `inst_.cbsz`, `inst_.abid`, and
  `inst_.blgp` into shared helpers for representative F32/F16/BF16/I8 paths,
  for example `V_MFMA_F32_32X32X1_2B_F32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:1376` through
  `:1378`, `V_MFMA_F32_32X32X4_2B_F16` at `:1713` through `:1715`, and I8
  MFMA at `:2056` through `:2058`.
- `permute_a_lane()` computes `S = 64 >> cbsz` and returns
  `(lane % S) + S * abid` without checking `cbsz <= 4`, whether
  `(1 << cbsz)` fits the instruction's block count, or whether
  `abid < (1 << cbsz)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:568` through
  `:577`.
- The scalar/SIMD MFMA helpers only gate whether the raw fields are non-zero
  before applying the permutation; representative F32 and I32 call sites are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:1182` through
  `:1187`, `:1243` through `:1251`, and `:3098` through `:3101`.

Impact:

Illegal or undefined MFMA broadcast encodings execute with a computed lane
mapping instead of being rejected or diagnosed. Extreme illegal values can also
produce nonsensical helper behavior, such as `CBSZ=7` yielding `S=0` before the
modulus in `permute_a_lane()`.

### CDNA4-RJ-022: SMFMAC ignores sparse index-set selection from `CBSZ`/`ABID`

Manual evidence:

- For 16-bit sparse source data, `CBSZ[1:0] == 0` lets `ABID[1:0]` select one
  of four 8-bit sparse-index sets within the `SRC2` VGPR; if `CBSZ[1:0] != 0`,
  the first set is selected at `cdna4/README.md:2464` through `:2466`.
- For the later 16-bit large forms, the sparse index table narrows this to two
  sets selected by `ABID[0]`, while the 8-bit/IU8/F8 large forms ignore
  `CBSZ[1:0]` and `ABID[1:0]`, at `cdna4/README.md:2497` through `:2500`.
- The manual also says `CBSZ` and `ABID` only select the index from the VGPR
  read and do not affect source-A matrix broadcast for sparse MFMA at
  `cdna4/README.md:2478`.

Rocjitsu evidence:

- Generated SMFMAC execution derives an `idx` base from `SRC2` and calls sparse
  helpers without passing `inst_.cbsz` or `inst_.abid`; representative BF16 and
  F16 generated paths are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:1168` through
  `:1175`, `:1628` through `:1635`, `:2398` through `:2403`, and `:2436`
  through `:2443`.
- The code generator emits the same index-base-only call shape for F32-result
  SMFMAC variants at `lib/python/amdisa/codegen/execute/matrix.py:140` through
  `:179`.
- The shared sparse helpers accept only `idx_base` and use fixed bit extraction
  from the `SRC2` VGPR, with no selector argument, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:3768` through
  `:3801`, `:3806` through `:3841`, `:3932` through `:3968`, and `:3974`
  through `:4008`.

Impact:

16-bit SMFMAC encodings that should select an alternate sparse-index set execute
with the helper's fixed/default index extraction. The large 8-bit/IU8/F8
selector-ignore rule may match the fixed extraction shape for those forms, but
the 16-bit selector contract is not represented.

### CDNA4-RJ-023: I32 SMFMAC variants are generated as unimplemented stubs

Manual evidence:

- The CDNA4 sparse MFMA inventory includes I8-to-I32 sparse forms, including
  `16x16x128_I8`, `32x32x64_I8`, and older/smaller I8 forms, at
  `cdna4/README.md:2446` through `:2459`.

Rocjitsu evidence:

- Generated `V_SMFMAC_I32_16X16X128_I8` and `V_SMFMAC_I32_32X32X64_I8`
  execution bodies throw `UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:1178` through
  `:1211` and `:1638` through `:1671`.
- The smaller `V_SMFMAC_I32_16X16X64_I8` and
  `V_SMFMAC_I32_32X32X32_I8` classes are also generated as stubs at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:2826` through
  `:2895`.
- The code generator emits stubs for non-F32 SMFMAC variants at
  `lib/python/amdisa/codegen/execute/matrix.py:181` through `:186`.

Impact:

Legal CDNA4 I8 sparse matrix instructions decode, but throw instead of
executing.

### CDNA4-RJ-024: SMFMAC index legality, alignment, and floating-state controls are not modeled

Manual evidence:

- Sparse index pairs require `index0 < index1` and `index0 != index1` at
  `cdna4/README.md:2476`.
- The sparse MFMA state table says denorm handling ignores MODE and keeps
  denorms, clamp uses `FP16_OVFL`, rounding is forced RNE, exceptions are not
  supported, `SRC0`/`SRC1`/`VDST` VGPRs must be even-aligned, and `SRC2` is
  VGPR-only with no even-alignment requirement at `cdna4/README.md:2484`
  through `:2496`.

Rocjitsu evidence:

- Shared SMFMAC helpers split the raw sparse-index field into two selectors and
  use both without validating order or distinctness at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:3788` through
  `:3793`, `:3828` through `:3833`, `:3952` through `:3961`, and `:3996`
  through `:4005`.
- Generated SMFMAC execution maps `SRC0`, `SRC1`, `VDST`, and `SRC2` bases
  directly through `src_base()` / `dst_base()` without a sparse-specific
  alignment or selector-legality check; representative generated paths are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:1168` through
  `:1175` and `:2436` through `:2443`.
- The executed F32-result SMFMAC paths do not read MODE or `FP16_OVFL`, and the
  helpers perform host floating accumulation without a clamp/overflow policy.
  The I32-result SMFMAC paths are currently stubs, so their clamp behavior is
  not reachable yet.

Impact:

Invalid sparse-index pairs execute as ordinary selector pairs, under-aligned
SMFMAC source/destination bases are accepted, and sparse F32 overflow behavior
is independent of the manual's MODE/`FP16_OVFL` contract. These are
sparse-specific instances of broader MFMA legality/state gaps, with the
additional index-pair legality rule unique to SMFMAC.

### CDNA4-RJ-025: Section 7.6 MAI dependency-wait rules are not modeled

Manual evidence:

- Section 7.6 states that the VOP3P-matrix table gives timing conditions where
  users must insert NOPs or independent VALU instructions at
  `cdna4/README.md:2697` through `:2709`.
- The table covers DLop, XDL, SGEMM, DGEMM, and SMFMA producer/consumer
  combinations with exact-opcode Source C forwarding exceptions, Source C
  overlap wait tiers, SrcA/SrcB waits, VM/LDS/FLAT/Export overlap waits, VALU
  RAW+WAW waits, `V_CMPX*` EXEC-mask forwarding waits, and an XDL/SMFMA Source C
  read versus VALU write WAR anti-dependency at `cdna4/README.md:2711` through
  `:2787`.

Rocjitsu evidence:

- The generic `MFMA` instruction flag only marks matrix FMA instructions, and
  `is_mfma()` only exposes that broad classification at
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:24` through `:45` and `:196`
  through `:204`.
- Codegen sets `flags_ |= MFMA` for names starting with `V_MFMA_` or
  `V_SMFMAC_`, but does not attach Section 7.6 producer/consumer class, pass
  count, overlap, forwarding, or required-wait metadata at
  `lib/python/amdisa/codegen/_generator.py:6539` through `:6542`.
- `ComputeUnit::issue_instruction()` executes the decoded instruction directly
  and only routes memory operations through the wait-counter memory pipeline at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:411` through `:437`.
- `WaitCounters` are documented as outstanding memory-operation counters for
  `vmcnt`, `lgkmcnt`, `expcnt`, vector-store, DS/K, tensor, and async memory
  forms, not MAI producer/consumer timing hazards, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wait_counters.h:36` through `:63`.
- The shared MFMA helpers do buffer inputs before writes to avoid hazards inside
  a single helper call, but that is not an inter-instruction Section 7.6
  scheduling model.

Impact:

Rocjitsu functional execution can run MAI producer/consumer sequences that the
manual requires software to separate with independent instructions, and it
cannot diagnose or model the Section 7.6 forwarding and overlap contract. If
rocjitsu remains intentionally non-cycle-accurate, this is still a missing
metadata/diagnostic surface for scheduler-sensitive MAI rules rather than a
basic arithmetic helper mismatch.

### CDNA4-RJ-026: Section 7.4 MFMA floating-point mode and DGEMM exceptions are not modeled

Manual evidence:

- Section 7.4 says MAI denormal handling varies by datatype and sometimes by
  MODE at `cdna4/README.md:2420` through `:2423`.
- `V_MFMA_F32_*_F32` honors MODE denormal flags for 32-bit inputs, while
  matrix-C input and result output ignore `MODE.denorm` and preserve denormals
  at `cdna4/README.md:2424` through `:2425`.
- Sub-32-bit float MFMA inputs ignore `MODE.denorm` and preserve denormals;
  `V_MFMA_F64_*_F64` ignores MODE, rounds RNE, and allows input/output denorms;
  and `V_MFMA_I32_*_I8` ignores MODE because it is integer arithmetic with
  sign-extended intermediate values at `cdna4/README.md:2426` through `:2428`.
- The matrix core does not support arithmetic exceptions except for DGEMM matrix
  operations, which do support exceptions, at `cdna4/README.md:2430`.

Rocjitsu evidence:

- `Wavefront` stores raw MODE state, but the generated CDNA4 MFMA call sites do
  not read it or pass it into helpers. Representative F32-input, I8, and F64
  generated paths pass only register bases, constant-accumulator state, and
  MFMA encoding fields at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:1369` through
  `:1378`, `:2049` through `:2058`, `:2930` through `:2938`, and `:2973`
  through `:2981`.
- The shared `exec_f32()` wrapper and F32 fast path accept no MODE or
  per-operand denorm policy, and use host/SIMD floating arithmetic through the
  generic MFMA helpers at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:922`
  through `:969`, `:1311` through `:1318`, and `:4038` through `:4050`.
- The F64 helper uses host floating arithmetic, with a scalar multiply/add loop
  and a SIMD fused-FMA path, and accepts only register bases,
  constant-accumulator state, and F64 negation bits. It does not explicitly
  force GPU RNE or record arithmetic exceptions at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:3492` through
  `:3600`.
- `Wavefront` exposes raw MODE storage at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:103` through `:111`, but
  shared `s_denorm_mode`, `s_round_mode`, and `v_clrexcp` execution helpers are
  empty stubs at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1640`
  through `:1645`, `:2422` through `:2425`, and `:3795` through `:3799`.

Impact:

F32-input MFMA currently cannot vary A/B denormal handling with GPU MODE while
preserving matrix-C/result denormals independently. DGEMM arithmetic exception
support is also absent: F64 MFMA can produce numeric results through host FMA,
but rocjitsu has no visible exception flagging or trap-facing state for the
manual's DGEMM exception contract. The I8 path naturally ignores MODE because
it is integer arithmetic; the missing pieces are the floating-point state and
exception semantics.

### CDNA4-RJ-027: CDNA4 SMEM address calculation misses selector and alignment rules

Manual evidence:

- Section 8.2.1.1 gives scalar/global `S_LOAD`, `S_STORE`, and
  `S_DCACHE_DISCARD` addressing as `SBASE` plus an instruction offset plus M0,
  an SGPR offset, or zero, depending on `IMM` and `SOE`, at
  `cdna4/README.md:2826` through `:2845`.
- The same section says scratch SMEM uses the selected scalar offset multiplied
  by 64, all address components are byte quantities whose two low bits are
  ignored or forced to zero, and `S_DCACHE_DISCARD` ignores six low bits at
  `cdna4/README.md:2846` through `:2850`.

Rocjitsu evidence:

- Generated CDNA4 SMEM disassembly/operand shaping uses `make_smem_offset()`;
  when `SOFFSET_EN=0` and `IMM=0`, it returns an immediate zero instead of the
  `OFFSET[6:0]` SGPR/M0 selector at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/smem.cpp:29` through
  `:39`. Codegen emits that helper from
  `lib/python/amdisa/codegen/_generator.py:7358` through `:7396`.
- The shared scalar-memory execution helper, used by non-scratch CDNA4 SMEM,
  only adds `SOFFSET` when `SOFFSET_EN` is set and only adds `OFFSET` when
  `IMM` is set at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:31`
  through `:48`.
- CDNA4 now routes `S_SCRATCH_LOAD_*` and `S_SCRATCH_STORE_*` through a
  scratch-specific helper that applies the manual's 64-byte scaling for
  `SOFFSET_EN` offsets at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/addr_calc.cpp:24` through
  `:47`, but that helper still ignores the `IMM=0, SOE=0` `OFFSET[6:0]`
  selector, reads `SOFFSET` selector value 124 as `sgpr_base + 124` rather than
  `wf.m0()`, and does not mask the two low address bits.
- `Operand::register_ref()` maps `OPR_SMEM_OFFSET` selector values only when
  they are ordinary SGPRs, so special offset selectors such as M0 also disappear
  from def-use metadata at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/operand.cpp:1129` through
  `:1134`.

Impact:

Legal CDNA4 SMEM encodings using the non-SOE offset selector, M0, unaligned
byte addresses, or special offset selectors will disassemble, analyze, or
execute against the wrong address or an incomplete dependency set in rocjitsu.

### CDNA4-RJ-028: `S_BUFFER_*` SMEM ignores buffer resource descriptor and bounds semantics

Manual evidence:

- Section 8.2.1.1 says scalar buffer memory uses a four-SGPR resource
  descriptor containing base address, stride, `num_records`, and `NV`; stride is
  used only for bounds checking and not for address calculation at
  `cdna4/README.md:2872` through `:2890`.
- Section 8.4 says `SBASE` must be even for `S_BUFFER_LOAD` and out-of-bounds
  dwords are clamped by not performing those memory operations at
  `cdna4/README.md:2930` through `:2938`.

Rocjitsu evidence:

- Generated CDNA4 scalar-buffer loads and stores construct a 128-bit `SBASE`
  operand, but still call `smem_calculate_address()` like raw-pointer scalar
  memory operations at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/smem.cpp:260` through
  `:389` and `:564` through `:637`.
- The shared scalar address helper reads only an SGPR pair as a 64-bit base and
  returns base plus offset at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:31`
  through `:48`.
- `ScalarMemState` carries only an address, width, data buffer, memory type,
  wait-counter type, and load/store flag, with no buffer descriptor or
  per-dword bounds state at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:70` through `:84`.

Impact:

CDNA4 `S_BUFFER_*` instructions use raw-pointer-style addressing instead of the
manual's scalar-buffer descriptor semantics, and rocjitsu cannot suppress only
the out-of-bounds dwords of a buffer scalar-memory access.

### CDNA4-RJ-029: SMEM dependency counter, time, clause, and legality rules are not modeled

Manual evidence:

- Section 8.2.1.1 describes scalar-memory source-overwrite and clause hazards,
  including the single-dword single-instruction exception and atomic
  single-instruction-clause requirement, at `cdna4/README.md:2861` through
  `:2870`.
- Section 8.3 says scalar memory reads and writes can return out of order, can
  return partial results, and increment `LGKM_CNT` by one for one dword or by
  two for two or more dwords at `cdna4/README.md:2914` through `:2925`.
- Section 8.4 describes SDATA/SBASE alignment restrictions and out-of-range
  behavior at `cdna4/README.md:2930` through `:2938`, and sections 8.2.4/8.2.5
  define 64-bit `S_MEMTIME`/`S_MEMREALTIME` writes at `:2908` through `:2912`.

Rocjitsu evidence:

- `ScalarMemState` carries `num_dwords`, but only one
  `wait_counter_type = WaitCounterType::LGKMCNT` value at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:70` through `:84`.
- Generated CDNA4 SMEM load bodies set `num_dwords` and
  `wait_counter_type`, for example `S_LOAD_DWORD` and `S_LOAD_DWORDX2` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/smem.cpp:57` through
  `:94`.
- `MemoryPipeline::issue()` increments exactly one wait counter for the issued
  instruction at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.h:62`
  through `:88`; `ScalarMemPipeline::complete_access()` writes every returned
  dword and then completes once at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:219` through
  `:245`.
- `S_MEMTIME` and `S_MEMREALTIME` execute as immediate SGPR writes outside the
  scalar memory pipeline at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1934`
  through `:1952`.
- Generated CDNA4 SMEM constructors and execution paths do not validate the
  source-overwrite, clause, SDATA alignment, SBASE alignment, or out-of-range
  source/destination rules.

Impact:

Multi-dword SMEM instructions undercount `LGKM_CNT`, time reads do not
participate in the scalar-memory dependency model, and rocjitsu accepts or
executes scalar-memory sequences and operands that the manual marks as
restricted or undefined.

### CDNA4-RJ-030: Scalar atomics decode with the wrong data operand contract and do not execute

Manual evidence:

- Table 39 says `GLC=1` makes scalar atomics return the pre-op value at
  `cdna4/README.md:2811`.
- Section 8.2.2 says scalar atomics use scalar-memory addressing and return the
  pre-operation value to `SDATA` only when `GLC` is set at
  `cdna4/README.md:2892` through `:2895`.

Rocjitsu evidence:

- Generated CDNA4 scalar atomic constructors model `SDATA` as a destination
  only; for example `S_ATOMIC_ADD` sets `dst_operands_[0] = &sdata`, uses only
  `sbase` and `soffset` as sources, and reports one destination at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/smem.cpp:1279` through
  `:1290`. Buffer atomics follow the same pattern, for example
  `S_BUFFER_ATOMIC_ADD` at `:811` through `:822`.
- The generated atomic `execute_impl()` bodies throw `UnimplementedInst`, with
  representative cases at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/smem.cpp:788` through
  `:824` and `:1256` through `:1292`.
- Python semantic derivation classifies SMEM atomics as `nop` at
  `lib/python/amdisa/semantics.py:1712` through `:1714`.

Impact:

Legal CDNA4 scalar atomic instructions cannot execute in rocjitsu, and the
decoded dataflow contract already loses the atomic input data source while
unconditionally claiming an `SDATA` write even when `GLC=0`.

### CDNA4-RJ-031: SMEM `GLC`/`NV` cache policy and discard behavior are incomplete

Manual evidence:

- Table 39 describes `GLC` load, store, and atomic policy plus `NV` at
  `cdna4/README.md:2811` through `:2821`.
- Chapter 9.1 details scalar-memory `GLC=0`/`GLC=1` read, write, and atomic
  cache-persistence/return behavior at `cdna4/README.md:3250` through `:3275`.
- Section 8.2.3 defines scalar cache invalidate/write-back operations, and
  section 8.2.1.1 says `S_DCACHE_DISCARD` ignores six low address bits at
  `cdna4/README.md:2846` through `:2850` and `:2902` through `:2908`.

Rocjitsu evidence:

- Generated CDNA4 SMEM load/store execution sets `ScalarMemState::mtype` from
  `GLC`, for example at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/smem.cpp:65` through
  `:92`, using `mtype_from_flags_gfx9()` from
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/gfx9_cache_flags.h:21`
  through `:31`.
- `ScalarMemPipeline::initiate_access()` calls scalar L1 load/store helpers
  without passing that instruction memory type at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:203` through
  `:214`.
- The scalar L1 cache derives memory type from the page-table entry rather than
  from the SMEM instruction flags in
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/l1_scalar_cache.cpp:49` through `:63`
  and `:118` through `:128`.
- The manual's `S_DCACHE_INV_VOL` and `S_DCACHE_WB_VOL` definitions target
  volatile scalar-cache lines at `cdna4/README.md:6526` through `:6532`, but
  generated CDNA4 execution calls the same full-cache invalidate/writeback
  helpers as the non-volatile forms at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/smem.cpp:673` through
  `:685`.
- Generated CDNA4 `S_DCACHE_DISCARD` and `S_DCACHE_DISCARD_X2` execute paths
  throw `UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/smem.cpp:743` through
  `:773`, and `NV` is not consumed by the scalar-memory pipeline.

Impact:

Scalar-memory `GLC`/`NV` policy bits are decoded but mostly inert for
execution, volatile-line cache operations collapse to full-cache operations, and
legal discard operations remain unimplemented.

### CDNA4-RJ-032: Buffer descriptor addressing and range checking are incomplete

Manual evidence:

- Section 9.1.5 defines buffer addresses from resource base, SGPR offset, VGPR
  offset/index, stride, element size, `ADD_TID`, swizzle state, and `NumRecords`
  at `cdna4/README.md:3061` through `:3100`.
- Section 9.1.5.1 defines private, raw, and structured range-checking modes and
  the `dst_sel = SEL_1` OOB read exception at `cdna4/README.md:3105` through
  `:3128`.
- Section 9.1.8 defines the full 128-bit descriptor layout and says an all-zero
  resource acts as an unbound buffer returning zero and dropping writes at
  `cdna4/README.md:3190` through `:3220`.

Rocjitsu evidence:

- `mubuf_calculate_addresses()` reads base, 14-bit stride, `NumRecords`, and a
  single `oob_raw` bit from the descriptor at
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
swizzled, unbound, and several OOB/read-channel cases can diverge from CDNA4
hardware semantics.

### CDNA4-RJ-033: Buffer format conversion and `dst_sel` semantics are missing

Manual evidence:

- Section 9.1.3 describes read/write data VGPR counts and buffer data-format
  conversion at `cdna4/README.md:3023` through `:3029`.
- Section 9.1.4 says MTBUF takes format from the instruction, formatted MUBUF
  takes format and `dst_sel` from the resource, raw MUBUF derives size/type
  from the opcode, INVALID resource format remains unbound, and D16 variants
  pack/load/store 16-bit values at `cdna4/README.md:3031` through `:3059`.
- Chapter 13.5 lists MTBUF `DFMT`/`NFMT` values and MUBUF format/D16 opcodes at
  `cdna4/README.md:26364` through `:26444`.

Rocjitsu evidence:

- Generated CDNA4 MUBUF formatted load/store bodies throw `UnimplementedInst`
  for representative non-D16 formatted operations at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/mubuf.cpp:38` through
  `:140` and for representative D16 formatted operations at `:247` through
  `:451`.
- Generated CDNA4 MTBUF formatted loads and stores use fixed 4-byte element
  sizes and raw VGPR payload transfers rather than `DFMT`/`NFMT` conversion;
  representative load/store bodies are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/mtbuf.cpp:59` through
  `:104`.
- Generated CDNA4 MTBUF D16 paths set fixed two-byte elements and D16
  writeback flags, but still do not derive typed conversion behavior from
  `DFMT`/`NFMT` at `mtbuf.cpp:387` through `:505`.
- Python semantic derivation maps MTBUF format mnemonics to fixed element
  counts and classifies unmatched MUBUF format mnemonics as `nop` at
  `lib/python/amdisa/semantics.py:2029` through `:2126`.

Impact:

Rocjitsu does not yet model CDNA4 typed-buffer conversion, resource-derived
formatted MUBUF conversion, destination-channel selection, INVALID/unbound
format behavior, or D16 formatted packing.

### CDNA4-RJ-034: Buffer `SOFFSET` and dword-alignment edge cases are not modeled

Manual evidence:

- The MUBUF/MTBUF field table says `SOFFSET` must be an SGPR, M0, or inline
  constant at `cdna4/README.md:2988` through `:3004`.
- Section 9.1.5 says the SGPR offset can come from an SGPR or M0 at
  `cdna4/README.md:3087` through `:3099`.
- Section 9.1.7 says dword-or-larger reads and writes ignore the two address
  LSBs, forcing dword alignment, at `cdna4/README.md:3186` through `:3188`.

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

### CDNA4-RJ-035: Buffer-to-LDS subset, M0 offset, and clamping are incomplete

Manual evidence:

- Section 9.1.9 says load-to-LDS is supported only for
  `BUFFER_LOAD_{ubyte,sbyte,ushort,sshort,dword,dwordX3,dwordX4,format_x}`,
  defines `LDS_offset = M0[17:0]`, uses `TIDinWave * 16` for 3- and 4-dword
  loads, and requires active-mask clamping so return data is not written
  outside the LDS allocation for the wave at `cdna4/README.md:3222` through
  `:3247`.

Rocjitsu evidence:

- Allowed raw byte/short/dword loads do implement an `inst_.lds` path, but use
  `wf.m0() + wf.lds_base()` as the base at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/mubuf.cpp:476` through
  `:687`.
- The same `inst_.lds` pattern is also generated for `buffer_load_dwordx2` even
  though that form is outside the manual's listed LDS subset at
  `mubuf.cpp:723` through `:747`.
- The D16 raw load paths also accept `inst_.lds` and use the full `M0` value
  for the LDS base at `mubuf.cpp:1236` through `:1514`.
- The vector-memory completion path writes LDS-destination loads at
  `lds_base + lane * per_lane_bytes` when no per-lane LDS address is present at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:119` through
  `:127`. The buffer path does not derive the manual's allocation-aware active
  mask before issuing the memory read.

Impact:

Rocjitsu can accept LDS forms the manual does not list, use high bits of M0 in
the LDS offset, use a simplified lane-to-LDS address formula, and issue reads
for lanes whose return data should be masked by LDS allocation clamping.

### CDNA4-RJ-036: Buffer cache-control and cache-maintenance policies are coarse

Manual evidence:

- Section 9.1.10 gives detailed vector-memory load, store, atomic, invalidate,
  and writeback cache-policy tables for `SC[1:0]` and `NT`, including
  `TG_SPLIT` behavior and SC-dependent `BUFFER_WBL2`/`BUFFER_INV` effects,
  beginning at `cdna4/README.md:3274`.

Rocjitsu evidence:

- `mtype_from_flags_gfx940()` collapses `SC0`, `SC1`, and `NT` into a coarse
  `Mtype` value at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/gfx940_cache_flags.h:29`
  through `:50`.
- Shared `BUFFER_INV`/`BUFFER_WBL2` helpers invalidate or flush broad cache
  levels without consulting the SC table or `TG_SPLIT` refinements at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:29`
  through `:56`.
- Generated CDNA4 `BUFFER_WBL2`/`BUFFER_INV` dispatches simply call those
  helpers at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/mubuf.cpp:1568`
  through `:1588`.

Impact:

The current model has useful broad cache-behavior hooks, but not the
instruction- and scope-specific CDNA4 policy required for precise cache-control
validation.

### CDNA4-RJ-037: Buffer atomics expose the wrong `VDATA` dataflow contract

Manual evidence:

- Section 9.1 says buffer atomics take data from VGPRs and optionally return
  the pre-operation value at `cdna4/README.md:2964`.
- Section 9.1.3 says atomics read data from the VGPRs starting at `VDATA`; if
  the atomic returns a value, it is returned to those same VGPRs at
  `cdna4/README.md:3029`.

Rocjitsu evidence:

- Generated CDNA4 buffer atomic constructors model `VDATA` as a destination
  only; `BUFFER_ATOMIC_ADD` sets `dst_operands_[0] = &vdata`, exposes only
  `vaddr`, `srsrc`, and `soffset` as sources, and reports one destination at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/mubuf.cpp:1682` through
  `:1701`.
- The execute path still reads `VDATA` internally to populate `store_data`, and
  gates the return write with `inst_.sc0`, at `mubuf.cpp:1704` through `:1724`;
  the public operand contract therefore diverges from the runtime behavior.
- Compare-and-swap forms need two input dwords from `VDATA`, and the generated
  execution reads both at `mubuf.cpp:1657` through `:1678`, but the constructor
  exposes only a destination operand at `:1640` through `:1653`.

Impact:

Execution can perform the RMW source read, but decode/dataflow users can miss
the VGPR input dependency and can report a destination write even for no-return
atomics where `SC0=0`.

### CDNA4-RJ-090: Buffer floating atomics miss L2 FP numeric and packed-lane rules

Manual evidence:

- Chapter 9.2 says floating memory atomics execute in LDS and L2 and can be
  issued as LDS, Buffer, Flat, Global, and Scratch instructions at
  `cdna4/README.md:3385` through `:3389`.
- Chapter 9.2 defines fixed RNE for float atomic adds, L2-specific denormal
  behavior for F32/F64/packed F16/BF16 forms, and NaN/signed-zero min/max,
  compare-swap, and add edge cases at `cdna4/README.md:3393` through `:3475`.

Rocjitsu evidence:

- Generated CDNA4 MUBUF floating atomics lower to the generic memory-pipeline
  `AtomicOp::FADD`, `FMIN`, and `FMAX` operations: representative F32, packed
  F16, F64 add, F64 min/max, and packed BF16 paths are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/mubuf.cpp:2199` through
  `:2450`.
- `BUFFER_ATOMIC_PK_ADD_F16` and `BUFFER_ATOMIC_PK_ADD_BF16` set
  `elem_size = 4` and `AtomicOp::FADD`, the same scalar 32-bit path as
  `BUFFER_ATOMIC_ADD_F32`, at `mubuf.cpp:2244` through `:2253` and
  `:2430` through `:2439`.
- The shared L2 atomic executor treats 4-byte floating atomics as one host
  `float` and 8-byte floating atomics as one host `double`, using ordinary
  addition plus `std::fmin`/`std::fmax` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:285` through
  `:363`.
- `VectorMemState` carries the `AtomicOp` enum, element size, and dataflow
  fields, but not the floating-point subtype, packed-lane mode, or
  denormal/rounding policy needed by Chapter 9.2, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:87` through `:112`.

Impact:

F32/F64 buffer atomics have a coarse functional model, but rocjitsu does not
enforce CDNA4's L2 floating-atomic denormal, RNE, or NaN/signed-zero rules.
Packed F16/BF16 buffer atomics are not type-correct because their two lanes
execute as one scalar F32 operation. `CDNA4-RJ-037` covers the separate buffer
atomic `VDATA` input/conditional-return metadata gap.

### CDNA4-RJ-038: Vector-buffer tests miss descriptor and format edge cases

Rocjitsu evidence:

- `tests/buffer_operand_test.cpp:33` through `:139` covers dynamic VADDR width,
  SRSRC scaling, and CDNA ACC bank folding for MUBUF/MTBUF operands.
- `tests/decode_smoke_test.cpp:726` through `:781` covers the MUBUF `lds`
  modifier in disassembly.
- `lib/python/amdisa/tests/test_semantic_operand_codegen.py:9` through `:47`
  covers generator helpers for legacy buffer VADDR width and SRSRC scaling, and
  `lib/python/amdisa/tests/test_sema_derive.py:1645` through `:1657` covers a
  derived buffer-load semantic shape.

Impact:

The existing tests cover important operand/codegen regressions, but not the
manual-derived descriptor modes, format conversion, M0/inline `SOFFSET`,
alignment, LDS clamping/subset, unbound resource, atomic dataflow, or
cache-policy cases identified in the vector-buffer audit slice.

### CDNA4-RJ-039: FLAT/GLOBAL/SCRATCH address-mode and aperture behavior is incomplete

Manual evidence:

- Chapter 10 says flat addresses are routed by aperture registers across
  video/system memory, LDS, and scratch, and unmapped regions generate memory
  violations at `cdna4/README.md:3477` through `:3479`.
- Section 10.3 says FLAT supports 32-bit and 64-bit addressing selected by
  `PTR32`, and defines private-aperture scratch address conversion at
  `cdna4/README.md:3586` through `:3600`.
- Sections 10.4 and 10.5 say GLOBAL and SCRATCH address component size depends
  on `ADDRESS_MODE`, GLOBAL must not access LDS, and SCRATCH uses swizzled
  `FLAT_SCRATCH` addressing with unsigned byte offsets at
  `cdna4/README.md:3619` through `:3651`.

Rocjitsu evidence:

- Generated CDNA4 FLAT constructors fix ordinary flat addresses as 64-bit VGPR
  pairs, narrow scratch to a 32-bit VGPR offset, and narrow global to a 32-bit
  VGPR offset only when `SADDR != 0x7f`; representative constructor code is at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/flat.cpp:27` through `:52`.
- The shared flat address helper uses fixed address-size formulas for FLAT,
  GLOBAL, and SCRATCH and does not consult visible `PTR32` or `ADDRESS_MODE`
  state at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h:27`
  through `:112`.
- The helper maps private-aperture FLAT addresses using a high-32-bit match and
  scratch lane stride, but does not model aperture limits, aperture holes, or
  the manual's scratch swizzle formula at `addr_calc_flat.h:113` through
  `:132`.
- The compute-unit router converts the first active lane's shared-aperture
  address to LDS and `LGKMCNT`, but it has no memory-violation path for GLOBAL
  accesses that resolve to LDS or for unmapped aperture ranges at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:263` through `:285`.

Impact:

Rocjitsu can run common flat/global/scratch addressing cases, but
mode-dependent address widths, scratch swizzling, aperture bounds, and
memory-violation behavior can diverge from CDNA4.

### CDNA4-RJ-040: Global/scratch direct-to-LDS flat-memory forms are not decoded or generated

Manual evidence:

- Sections 10.4 and 10.5 say GLOBAL and SCRATCH instructions can move data
  directly between LDS and memory at `cdna4/README.md:3630` and `:3649`.
- Section 10.3 gives direct-LDS destination formulas using the hardware LDS
  base, `M0[17:2] * 4`, instruction offset, and thread ID at
  `cdna4/README.md:3604` through `:3608`.
- Chapter 13.6 lists `GLOBAL_LOAD_LDS_*` and `SCRATCH_LOAD_LDS_*` opcode
  entries at `cdna4/README.md:26566` through `:26578` and `:26595` through
  `:26600`.

Rocjitsu evidence:

- A targeted search of generated CDNA4 and CDNA3 flat code found no
  `GLOBAL_LOAD_LDS`, `SCRATCH_LOAD_LDS`, `GlobalLoadLds`, or `ScratchLoadLds`
  instruction classes.
- CDNA4 generated flat decoding shares `FLAT_*` classes across `SEG` values for
  overlapping opcodes, with `flat_mnemonic()` rewriting `seg=1` to
  `scratch_*` and `seg=2` to `global_*` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/encodings.cpp:14` through
  `:20` and `:281` through `:284`. That only covers opcodes that have a
  `FLAT_*` base entry.
- The generated CDNA4 flat sub-decode table marks opcodes 38-42 invalid at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/decoder.cpp:9258` through
  `:9262`, even though those are `GLOBAL_LOAD_LDS_{UBYTE,SBYTE,USHORT,SSHORT,DWORD}`
  and `SCRATCH_LOAD_LDS_{UBYTE,SBYTE,USHORT,SSHORT,DWORD}` in the manual/XML.
  It also leaves opcode slots 125-126 invalid at `decoder.cpp:9345` through
  `:9346`, matching the missing `GLOBAL_LOAD_LDS_DWORDX4` and
  `GLOBAL_LOAD_LDS_DWORDX3` forms.
- The generic flat-load generator only sets a VGPR destination and never sets
  `lds_dst`, `lds_base`, or per-lane LDS addresses at
  `lib/python/amdisa/codegen/_generator.py:4620` through `:4648`.
- The vector memory pipeline has an LDS-destination completion path, but it is
  only reached when an instruction sets `d.lds_dst` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:76` through
  `:127`.

Impact:

The CDNA4 decoder/runtime cannot decode or execute the manual's
GLOBAL/SCRATCH-to-LDS flat-memory forms end to end, even though those opcodes
are present in the XML. Overlapping GLOBAL/SCRATCH load/store/atomic opcodes
can still reach the shared FLAT classes through `SEG`-based mnemonic rewriting;
the missing part is the direct-to-LDS-only opcode subset.

### CDNA4-RJ-041: FLAT wait-counter and ordering contract is simplified

Manual evidence:

- Section 10.2 says FLAT instructions execute internally as both LDS and Buffer
  operations, increment both `VM_CNT` and `LGKM_CNT`, and complete only when
  both have decremented at `cdna4/README.md:3570` through `:3576`.
- Sections 10.2.1 and 10.2.2 describe out-of-order completion, same-VGPR
  return hazards, and the `S_WAITCNT 0` restriction at
  `cdna4/README.md:3578` through `:3584`.
- Sections 10.4 and 10.5 say GLOBAL and SCRATCH use only `VM_CNT`, not
  `LGKM_CNT`, at `cdna4/README.md:3630` through `:3651`.

Rocjitsu evidence:

- Generated CDNA4 flat loads and atomics initially set
  `wait_counter_type = WaitCounterType::VMCNT`; representative bodies are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/flat.cpp:55` through
  `:65` and `:1059` through `:1080`.
- The compute-unit router changes shared-aperture FLAT operations to
  `LOCAL_MEM` and `LGKMCNT`, which is an either/or counter choice rather than a
  dual VM+LGKM issue/retire model at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:263` through `:285`.

Impact:

Functional LDS/global routing can work for simple cases, but rocjitsu does not
model the CDNA4 FLAT dual-counter and ordering hazards needed for wait-counter
exactness.

### CDNA4-RJ-042: Floating flat atomics miss packed and FP-mode special cases

Manual evidence:

- Section 10.3.1 says floating-point atomics must set `SC[0]=0`, FP32 atomics
  flush denormals to zero, FP64 and FP16 atomics do not flush denormals, and
  rounding is fixed RNE at `cdna4/README.md:3611` through `:3617`.
- The FLAT/GLOBAL opcode inventory includes F32, packed F16/BF16, and F64
  floating atomics at `cdna4/README.md:3546` through `:3551` and
  `:26513` through `:26523`.

Rocjitsu evidence:

- Generated CDNA4 floating flat atomics use `d->is_load = (inst_.sc0 != 0)`,
  so return-data mode remains executable for FP atomics; representative F32,
  packed F16, F64, and packed BF16 bodies are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/flat.cpp:1815` through
  `:1836`, `:1873` through `:1895`, `:1931` through `:1955`, and
  `:2111` through `:2133`.
- `FLAT_ATOMIC_PK_ADD_F16` and `FLAT_ATOMIC_PK_ADD_BF16` set
  `elem_size = 4` and use `AtomicOp::FADD`, the same scalar 32-bit floating
  atomic operation used by `FLAT_ATOMIC_ADD_F32`, at `flat.cpp:1873` through
  `:1880` and `:2111` through `:2118`.
- The memory pipeline applies 4-byte FP atomics as scalar `float` and 8-byte
  FP atomics as scalar `double`, using ordinary `+`, `std::fmin`, and
  `std::fmax` without FP32 denormal flushing, packed F16/BF16 lanes, or
  explicit RNE handling at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:285` through
  `:365`.

Impact:

F32/F64 atomic add/min/max have a coarse functional model, but packed F16/BF16
atomics are not type-correct and CDNA4's FP atomic return-mode, denormal, and
rounding rules are not enforced.

### CDNA4-RJ-043: Flat-memory `NV` is not represented

Manual evidence:

- The Section 10.1 field table lists an `NV` field and describes it as
  non-volatile memory access at `cdna4/README.md:3495` through `:3497`, while
  Chapter 13.6's FLAT field map has no corresponding bit at
  `cdna4/README.md:26488` through `:26499`.

Rocjitsu evidence:

- Generated CDNA4 flat machine instruction layouts contain `nt`, `sve`/`lds`,
  `seg`, `sc0`, and `sc1`, but no `nv` field at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h:104`
  through `:148`.
- Generated CDNA4 flat execution forwards `NT` and cache scope to
  `mtype_from_flags_gfx940()`, but has no `NV` state to forward; representative
  load code is at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/flat.cpp:55` through `:65`.

Impact:

If the Chapter 10 `NV` prose reflects real CDNA4 behavior, rocjitsu cannot
decode or model it. Because Chapter 13 also omits the bit, this should remain
an ambiguity rather than a confirmed execution bug until resolved.

### CDNA4-RJ-044: Flat/global/scratch tests miss CDNA4-specific edge cases

Rocjitsu evidence:

- Shared tests cover CDNA4 scratch-base and global-address basics at
  `tests/shared_infra_test.cpp:2762` through `:2842`.
- Generic flat VM tests include a CDNA3/CDNA4 encoding helper at
  `tests/amdgpu_vm_test.cpp:1203` through `:1221` and exercise some global
  atomic behavior later in that file.
- The static pass did not find CDNA4 end-to-end cases for `PTR32` or
  `ADDRESS_MODE`, direct-to-LDS global/scratch flat forms, FLAT dual-counter
  behavior, aperture-hole or GLOBAL-to-LDS memory violations, packed
  F16/BF16 atomics, FP atomic `SC0=0`, or the flat `NV` manual ambiguity.

Impact:

Existing tests protect useful core addressing behavior, but not the
manual-derived edge cases identified by the CDNA4 Chapter 10 slice.

### CDNA4-RJ-045: LDS M0 clamping, allocation granularity, and bank behavior are not modeled

Manual evidence:

- Section 11.1 describes CDNA4 LDS as 64 banks and says bank conflicts serialize
  indexed and atomic operations at `cdna4/README.md:3679` through `:3687`.
- Section 11.3.1 says all LDS operations require `M0` initialization and that
  `M0[16:0]` contains the LDS segment byte-size used to clamp final addresses at
  `cdna4/README.md:3728` through `:3730`.
- Section 3.6.5 says CDNA4 LDS allocation uses contiguous 1280-byte blocks on
  1280-byte alignment, and clamping uses the smaller of the SPI allocation size
  and `M0` at `cdna4/README.md:577` through `:581`.

Rocjitsu evidence:

- The shared DS address helper computes `VGPR[ADDR] + offset + lds_base` and
  never reads `wf.m0()` for normal DS operations at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:60`
  through `:76`.
- Representative CDNA4 DS generated bodies delegate to that helper for ordinary
  loads/stores/atomics, for example `DS_ADD_U32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/ds.cpp:43` through `:64`.
- The CU LDS allocator aligns allocations to 256 bytes, not CDNA4's documented
  1280-byte granularity, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.h:239` through `:245`.
- The LDS backing and local-memory pipeline implement functional OOB read-zero
  and write-drop behavior against total backing size at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/lds.h:31` through `:107` and vector
  loads/stores at `:136` through `:168`, but do not model per-wave
  `min(lds_size, M0)` clamping or bank-conflict serialization.

Impact:

CDNA4 LDS accesses can address outside the wave/workgroup allocation and still
hit another modeled LDS region if they remain inside the CU backing. Residency,
OOB behavior, and timing-sensitive bank-conflict behavior can differ from the
manual.

### CDNA4-RJ-046: `DS_*_ADDTID_B32` uses the wrong address formula

Manual evidence:

- `DS_WRITE_ADDTID_B32` stores to
  `{OFFSET1,OFFSET0} + M0[15:0] + laneID * 4` at
  `cdna4/README.md:20450` through `:20455`.
- `DS_READ_ADDTID_B32` uses the same formula for the load address at
  `cdna4/README.md:21799` through `:21807`.

Rocjitsu evidence:

- Current generated CDNA4 `DS_WRITE_ADDTID_B32` calls the generic
  `ds_calculate_addresses()` helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/ds.cpp:1002` through
  `:1020`, so it reads the encoded `ADDR` VGPR instead of using ADDTID's
  lane-based formula.
- Current generated CDNA4 `DS_READ_ADDTID_B32` does use a special path, but it
  computes `lane * (((M0 >> 16) & 0x1ff) * 4) + offset + lds_base` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/ds.cpp:4843` through
  `:4868`, not `{OFFSET1,OFFSET0} + M0[15:0] + laneID * 4`.
- The generator's ADDTID templates contain the same high-M0 stride formula for
  both read and write at `lib/python/amdisa/codegen/_generator.py:5298` through
  `:5357`, so a regeneration would not recover the CDNA4 manual formula.

Impact:

CDNA4 ADDTID reads and writes can access completely different LDS addresses from
hardware. The write form is currently generic-DS addressing, while the generated
special form would still use the wrong M0 bit slice and lane scaling.

### CDNA4-RJ-047: DS READ2/WRITE2 duplicate-offset collapse is not modeled

Manual evidence:

- Section 11.3.1 says setting both two-address offsets to the same value
  specifies only one address, causes only one read/write, and uses only `DATA0`
  at `cdna4/README.md:3761` through `:3767`.

Rocjitsu evidence:

- `DsRead2B32Ds::execute_impl()` always sets `ds2_active = true`, computes both
  addresses from `offset0` and `offset1`, and sets a second destination register
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/ds.cpp:2177` through
  `:2200`.
- The local-memory pipeline always issues the second DS2 load when
  `ds2_active` is true and writes the second response during completion at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:520` through `:528`
  and `:606` through `:637`.
- Generated WRITE2 bodies follow the same unconditional second-access pattern.

Impact:

Equal-offset READ2/WRITE2 encodings perform two modeled accesses in rocjitsu
instead of the manual's one-access form. Stores can use `DATA1` where hardware
should ignore it, and reads can write a second destination value where only one
access should occur.

### CDNA4-RJ-048: `DS_SWIZZLE_B32` misses FFT and rotate modes

Manual evidence:

- `DS_SWIZZLE_B32` supports FFT mode for offsets `>= 0xe000`, rotate mode for
  offsets `>= 0xc000 && < 0xe000`, quad mode, and 32-lane bit-mask mode, with
  invalid-thread reads returning zero at `cdna4/README.md:20852` through
  `:20973`.

Rocjitsu evidence:

- The shared generated helper only branches on `offset & 0x8000`: bit set uses
  quad selectors, bit clear uses the 32-lane bit-mask formula at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:265`
  through `:296`.
- CDNA4 `DsSwizzleB32Ds::execute_impl()` calls that helper directly at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/ds.cpp:2374` through
  `:2376`.
- The generator profile test only asserts source selection and the quad
  selector expression, not FFT or rotate behavior, at
  `lib/python/amdisa/tests/test_generator_profile_gates.py:1116` through
  `:1157`.

Impact:

CDNA4 swizzle offsets in the FFT and rotate ranges execute as quad-mode
swizzles in rocjitsu, so any shader using those documented forms gets incorrect
cross-lane values.

### CDNA4-RJ-049: Packed F16/BF16 LDS atomics execute through the scalar F32 atomic path

Manual evidence:

- Chapter 9.2 says LDS packed F16/BF16 atomics have MODE-dependent denormal
  handling and RNE rounding at `cdna4/README.md:3387` through `:3428`.
- Section 12.12 defines `DS_PK_ADD_F16`, `DS_PK_ADD_BF16`, and their return
  variants as packed two-component 16-bit additions at `cdna4/README.md:20407`
  through `:20442` and `:21809` through `:21844`.

Rocjitsu evidence:

- CDNA4 packed no-return and return forms set `elem_size = 4` and
  `atomic_op = AtomicOp::FADD`, the same operation used for scalar F32 atomics,
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/ds.cpp:924` through
  `:982` and `:4893` through `:4960`.
- `execute_lds_atomic_rmw()` treats `elem_size == 4` floating atomics as one
  `float` and applies ordinary C++ `+`, `fmin`, or `fmax` without packed
  half/BF16 lane handling or MODE denormal policy at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:285` through `:297`
  and `:423` through `:445`.

Impact:

Packed LDS F16/BF16 atomics are type-incorrect: rocjitsu interprets the 32-bit
word as scalar F32 instead of two packed 16-bit lanes and cannot model the
manual's FP-mode requirements.

### CDNA4-RJ-050: LDS transpose-load preconditions are not validated

Manual evidence:

- Section 11.4 says MFMA transpose loads require `EXEC` all ones, LDS address
  alignment to data size, and even-aligned VGPRs for 64-bit-or-larger data
  except `DS_READ_B96_TR_B6` at `cdna4/README.md:3785` through `:3790`.

Rocjitsu evidence:

- CDNA4 transpose load execute bodies set `d->transpose` and call generic DS
  address calculation, with no checks for all-ones `EXEC`, LDS address
  alignment, or even VGPR alignment at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/ds.cpp:5140` through
  `:5241`.
- The shared transpose helpers implement data shuffles for TR_B4/B6/B8/B16 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/ds_transpose.h:26` through
  `:250`, but they do not validate instruction preconditions.
- Existing transpose tests check ACC destination routing for `DS_READ_B64_TR_B16`
  at `tests/amdgpu_vm_test.cpp:2039` through `:2090` and
  `tests/hip_ds_transpose_acc_test.cpp:59` through `:87`, not precondition
  enforcement or all layout variants.

Impact:

Illegal or underspecified CDNA4 transpose loads can execute in rocjitsu instead
of being rejected/nullified according to the manual's constraints. Layout bugs
in TR_B4/B6/B8/B16 can also escape because adjacent tests mostly protect ACC
routing.

### CDNA4-RJ-051: Data-share tests miss CDNA4 LDS edge cases

Rocjitsu evidence:

- Existing runtime coverage includes CDNA4 `ds_add_rtn_u32` and no-return
  `ds_add_u32` stress cases at `tests/amdgpu_vm_test.cpp:1877` through `:1966`.
- Existing transpose coverage checks ACC routing at
  `tests/amdgpu_vm_test.cpp:2039` through `:2090` and
  `tests/hip_ds_transpose_acc_test.cpp:59` through `:87`.
- Python generator/semantics tests cover DS swizzle source selection,
  READ2/atomic semantic-class derivation, and transpose-kind derivation at
  `lib/python/amdisa/tests/test_generator_profile_gates.py:1116` through
  `:1157` and `lib/python/amdisa/tests/test_sema_derive.py:1880` through
  `:1995`.
- The static pass did not find focused CDNA4 tests for `M0[16:0]` LDS clamping,
  1280-byte allocation granularity, ADDTID addressing, duplicate-offset
  READ2/WRITE2 collapse, `DS_SWIZZLE_B32` FFT/rotate modes, packed F16/BF16 LDS
  atomics, transpose preconditions/layout variants, or the CDNA4 `GDS` reserved
  versus GDS wording ambiguity.

Impact:

The current tests protect useful DS plumbing and ACC routing, but the CDNA4
manual-derived LDS edge cases identified in this slice can regress without
focused coverage.

### CDNA4-RJ-052: CDNA4 HWREG get/set uses the wrong register map

Manual/XML evidence:

- Chapter 3 lists separate `STATUS`, `MODE`, `HW_ID`, `XCC_ID`, `TRAPSTS`, GPR
  allocation, and LDS allocation state at `cdna4/README.md:401` through `:437`
  and details `STATUS`/`MODE` fields at `cdna4/README.md:457` through `:512`.
- CDNA4 XML maps `HW_REG_MODE=1`, `HW_REG_STATUS=2`, `HW_REG_TRAPSTS=3`,
  `HW_REG_HW_ID=4`, `HW_REG_GPR_ALLOC=5`, `HW_REG_LDS_ALLOC=6`,
  `HW_REG_IB_STS=7`, and `HW_REG_XCC_ID=20` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna4.xml:116837` through
  `:116970`.

Rocjitsu evidence:

- CDNA4 `S_GETREG_B32` treats `reg_id=1` as `wf.status_raw()`, `reg_id=4` and
  `reg_id=5` as low/high CU ID, `reg_id=6` as SGPR allocation, and `reg_id=7`
  as VGPR allocation at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopk.cpp:298` through
  `:327`.
- CDNA4 `S_SETREG_B32` and `S_SETREG_IMM32_B32` only handle `reg_id=1`, again
  as raw `STATUS`, at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopk.cpp:347`
  through `:404`.
- Existing harness tests encode `hwreg=1` as `STATUS` and assert raw status
  writes at `tests/instruction_execution_harness_test.cpp:3592` through `:3675`,
  so the test suite locks in the wrong CDNA4 map.

Impact:

CDNA4 HWREG accesses to `MODE`, `STATUS`, `TRAPSTS`, `GPR_ALLOC`, `LDS_ALLOC`,
`IB_STS`, and `XCC_ID` are decoded or executed against the wrong state, while
some real register IDs are unreachable.

### CDNA4-RJ-053: Raw `STATUS` helper bits can drift from `EXEC` and `VCC`

Manual evidence:

- Chapter 3 defines `EXECZ` as the summary bit for zero `EXEC` at
  `cdna4/README.md:447` through `:455`.
- The `STATUS` table includes `EXECZ` and `VCCZ` bits at
  `cdna4/README.md:457` through `:487`.
- Section 3.9 says `VCCZ` is updated every time VCC is updated, including
  scalar writes to VCC, and that VCC is fully written at `cdna4/README.md:606`
  through `:624`.

Rocjitsu evidence:

- `Wavefront` stores `EXEC`, `VCC`, raw `MODE`, and raw `STATUS` as separate
  state; `set_exec()` and `set_vcc()` do not update raw `STATUS` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:211` through `:238`, while
  `IsaWavefront::status` is a separate raw register at `:608` through `:617`.
- CDNA status accessors expose `EXECZ` and `VCCZ` bits in the raw status layout
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/cdna_isa_base.h:45`
  through `:81`.
- Normal CDNA4 branch operands compute `VCCZ` and `EXECZ` from live `VCC` and
  `EXEC` at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopp.cpp:126`
  through `:178`, and scalar special-source reads do the same at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/operand.cpp:1395` through
  `:1404`.
- `S_GETREG_B32`, however, reads the raw status register for its current
  `reg_id=1` mapping at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopk.cpp:298` through
  `:327`.

Impact:

Control-flow instructions and scalar special-source operands can observe correct
live helper values while HWREG-visible `STATUS` reports stale `EXECZ`/`VCCZ`
bits. After the HWREG ID-map bug above is fixed, this still needs an explicit
status synchronization or computed-read policy.

### CDNA4-RJ-054: `S_SETVSKIP` is decoded but unimplemented

Manual/XML evidence:

- The CDNA4 `EXEC` section says hardware does not optimize `EXEC=0` and points
  software to `CBRANCH` or `VSKIP` for fast skipping at `cdna4/README.md:447`
  through `:455`.
- The `MODE` table includes `VSKIP` bit 28 at `cdna4/README.md:489` through
  `:512`.
- CDNA4 XML names `S_SETVSKIP` and describes it as enabling or disabling VSKIP
  mode at `shared/machine-readable-isa/isa/amdgpu_isa_cdna4.xml:44334` through
  `:44335`.

Rocjitsu evidence:

- CDNA4 decodes `S_SETVSKIP`, but its executor throws `UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopc.cpp:388` through
  `:408`.
- The expected-CDNA-unimplemented list explicitly includes `s_setvskip` at
  `tests/instruction_execution_harness_test.cpp:185`.

Impact:

Shaders that rely on the documented VSKIP mechanism cannot execute through
rocjitsu even though the instruction is present in XML and generated decode.

### CDNA4-RJ-055: Trap, exception, and TRAPSTS state is not modeled

Manual evidence:

- Section 3.10 defines TTMP privilege, trap payload packing into
  `TTMP0`/`TTMP1`, `STATUS.TRAP_EN`, and `MODE.EXCP_EN` at
  `cdna4/README.md:626` through `:654`.
- Section 3.10.1 defines sticky `TRAPSTS` fields including `EXCP`, `SAVECTX`,
  `ILLEGAL_INST`, address-watch bits, `EXCP_CYCLE`, and `DP_RATE` at
  `cdna4/README.md:655` through `:668`.
- Section 3.11 defines sticky `TRAPSTS.mem_viol`, memory-violation trap enable,
  and imprecise saved PC behavior at `cdna4/README.md:670` through `:689`.

Rocjitsu evidence:

- A static source search finds no modeled `TRAPSTS`, `mem_viol`, `SAVECTX`,
  `ILLEGAL_INST`, `EXCP_CYCLE`, or `DP_RATE` state in rocjitsu runtime code.
- CDNA4 `S_TRAP` is decoded as a program terminator but throws
  `UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopp.cpp:302` through
  `:314`.
- The shared `S_RFE_B64` executor is empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2405`
  through `:2406`, and `S_SENDMSG` is also a stub at `:2427` through `:2428`.
- The shared `V_CLREXCP` VOP1/VOP3 helpers are empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:3796`
  through `:3799`, even though the CDNA4 VOP1 definition clears this wave's
  vector-ALU exception state at `cdna4/README.md:8878` through `:8880`.

Impact:

CDNA4 trap enable state, trap entry payloads, trap returns, sticky exception
status, and memory-violation reporting cannot be emulated or observed through
HWREG state.

### CDNA4-RJ-056: TTMP privilege and CDNA4 TTMP launch payloads are missing

Manual evidence:

- Section 3.10 says TTMP writes are privileged; when not privileged, TTMP writes
  are ignored and reads return zero at `cdna4/README.md:626` through `:640`.
- Section 3.13 initializes CDNA4 `TTMP4`/`TTMP5` to zero, `TTMP6`/`TTMP7` to
  the dispatch packet address, `TTMP8`/`TTMP9`/`TTMP10` to dispatch grid X/Y/Z,
  and `TTMP11` to wave ID in workgroup at `cdna4/README.md:705` through `:728`.

Rocjitsu evidence:

- CDNA4 scalar reads map TTMP encodings 108 through 123 directly into the SGPR
  file with no privilege check at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/operand.cpp:1338` through
  `:1350`.
- CDNA4 scalar writes map TTMP encodings 108 through 123 directly into the SGPR
  file with no privilege check at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/operand.cpp:1501` through
  `:1526`.
- Dispatch initialization writes workgroup-id system SGPRs and has an
  RDNA4/gfx1250-only TTMP payload for `TTMP7`/`TTMP9`, but no CDNA4
  `TTMP4` through `TTMP11` branch, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:265` through
  `:298`.

Impact:

Unprivileged CDNA4 shader code can read or write TTMP storage through rocjitsu,
and CDNA4 kernels do not receive the documented dispatch-packet, grid-size, or
wave-id launch payload in TTMPs.

### CDNA4-RJ-057: CDNA4 `HW_ID` and `XCC_ID` contents are incomplete

Manual/XML evidence:

- Section 3.12 defines `HW_ID` fields for wave slot, SIMD, pipeline, CU, shader
  array, shader engine, thread-group ID, VM ID, queue ID, state ID, and
  micro-engine ID, and defines `XCC_ID[3:0]` at `cdna4/README.md:691` through
  `:703`.
- CDNA4 XML maps `HW_REG_HW_ID=4` and `HW_REG_XCC_ID=20` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna4.xml:116867` through
  `:116950`.

Rocjitsu evidence:

- CDNA4 `S_GETREG_B32` returns only `wf.cu().id()` for `reg_id=4`, uses
  `reg_id=5` as a high half of CU ID even though XML maps it to
  `GPR_ALLOC`, and never handles `reg_id=20` for `XCC_ID` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopk.cpp:304` through
  `:323`.
- `Wavefront` tracks useful dispatch state such as `wf_id`, `wg_id`,
  `dispatch_id`, and `process_id` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:147` through `:165`, but
  the CDNA4 HWREG path does not pack these into the manual's `HW_ID` layout.

Impact:

Code reading CDNA4 `HW_ID` or `XCC_ID` through HWREG gets an incomplete CU-only
value or an unhandled register instead of the documented packed hardware-ID
state.

### CDNA4-RJ-058: Optional `TG_SIZE` system SGPR launch payload is not initialized

Manual evidence:

- Section 3.13 says compute launch may append a `TG_SIZE` system SGPR containing
  `{first_wave, 6'h00, wave_id_in_group[4:0], 2'h0, 14'h0,
  work-group_size_in_waves[5:0]}` when `COMPUTE_PGM_RSRC2.tg_size_en` is set at
  `cdna4/README.md:705` through `:720`.

Rocjitsu evidence:

- Dispatch initialization writes enabled workgroup-id system SGPRs after user
  SGPRs, then proceeds to the RDNA4/gfx1250 TTMP payload and packed workitem-ID
  initialization; it has no `TG_SIZE` write path at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:272` through
  `:321`.
- A static source search for `TG_SIZE`, `tg_size`, `wave_id_in_group`,
  `first_wave`, and `work_group_size_in_waves` found no rocjitsu launch payload
  implementation.

Impact:

CDNA4 kernels that request the `TG_SIZE` system SGPR observe zero or stale SGPR
state instead of the documented wave/workgroup-size payload.

### CDNA4-RJ-059: `S_RFE_B64` and `S_RFE_RESTORE_B64` do not model trap return

Manual evidence:

- Section 4.1 describes `S_RFE` as returning from the trap handler at
  `cdna4/README.md:736` through `:752`.
- Section 4.5 includes a required delay from `S_SETREG_B32` writing `TRAPSTS`
  to `S_RFE` / `S_RFE_restore`, tying trap-return behavior to trap status at
  `cdna4/README.md:819` through `:867`.

Rocjitsu evidence:

- CDNA4 `S_RFE_B64` constructs an operand and dispatches to the shared helper
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sop1.cpp:593` through
  `:605`.
- The shared `execute_s_rfe_b64_sop1` helper is empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2405`
  through `:2406`.
- CDNA4 `S_RFE_RESTORE_B64` throws `UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sop2.cpp:1011` through
  `:1033`.

Impact:

Trap-handler return does not restore PC/privilege/trap state, and the restore
variant cannot execute at all.

### CDNA4-RJ-060: Debug conditional branches never branch

Manual evidence:

- Section 4.2 lists `S_CBRANCH_CDBGSYS`, `S_CBRANCH_CDBGUSER`,
  `S_CBRANCH_CDBGSYS_OR_USER`, and `S_CBRANCH_CDBGSYS_AND_USER` as debug-flag
  conditional branches at `cdna4/README.md:754` through `:779`.

Rocjitsu evidence:

- The CDNA4 constructors for these four SOPP branch instructions decode the
  label operand but do not set `COND_BRANCH` flags or implement a branch body at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopp.cpp:364` through
  `:414`.
- The shared debug-branch helpers are empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:997`
  through `:1011`.

Impact:

Programs using debug conditional branches always fall through in rocjitsu,
regardless of debug flag state.

### CDNA4-RJ-061: Fork/join divergent control flow is not implemented

Manual evidence:

- Section 4.6 describes arbitrary divergent control flow with
  `S_CBRANCH_{I,G}_FORK` and `S_CBRANCH_JOIN`, including a six-deep branch
  stack, CSP mode bits, EXEC/PC stack entries, path selection, and pseudocode at
  `cdna4/README.md:868` through `:924`.

Rocjitsu evidence:

- CDNA4 `S_CBRANCH_G_FORK` throws `UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sop2.cpp:962` through
  `:984`.
- CDNA4 `S_CBRANCH_I_FORK` throws `UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopk.cpp:271` through
  `:285`.
- CDNA4 `S_CBRANCH_JOIN` calls the shared helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sop1.cpp:913` through
  `:928`, and that helper is empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1013`
  through `:1015`.
- Static source search found no rocjitsu model for the CDNA4 CSP branch stack
  described by the manual.

Impact:

Compiler-generated fork/join divergent-control sequences either fail at fork or
fall through at join without restoring `EXEC`/PC from the branch stack.

### CDNA4-RJ-062: Sleep, wakeup, priority, and message control side effects are mostly stubs

Manual evidence:

- Section 4.1 lists `S_NOP`, `S_SLEEP`, `S_WAKEUP`, `S_SETPRIO`,
  `S_SENDMSG`, and `S_SENDMSGHALT` control behavior at `cdna4/README.md:736`
  through `:752`.
- Section 4.4 says `S_SENDMSG` increments `LGKM_CNT` and decrements after the
  message is sent at `cdna4/README.md:785` through `:817`.

Rocjitsu evidence:

- `S_WAKEUP`, `S_SLEEP`, `S_SETPRIO`, `S_SENDMSG`, and `S_SENDMSGHALT`
  constructors dispatch to shared helpers at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopp.cpp:59` through `:65`
  and `:255` through `:300`.
- Shared helpers for `S_WAKEUP`, `S_SETPRIO`, `S_SENDMSG`, and
  `S_SENDMSGHALT` are empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2690`,
  `:2532` through `:2533`, `:2427` through `:2428`, and `:2490` through
  `:2492`.
- `S_SLEEP` only requests a functional yield and does not model the
  immediate-controlled 64-to-8128 cycle sleep duration or interaction with
  `S_WAKEUP` at `execute_shared.h:2555` through `:2563`.

Impact:

These instructions decode, but their architectural scheduling, priority,
message, halt, wakeup, and wait-counter side effects are absent or only modeled
as a generic functional yield.

### CDNA4-RJ-063: Scalar-memory and message `LGKM_CNT` accounting is simplified

Manual evidence:

- Section 4.4 says scalar-memory reads increment `LGKM_CNT` by dword count
  (one for one dword, two for two or more dwords), `S_MEMTIME` counts like
  `s_load_dwordx2`, `S_SENDMSG` increments by one, and decrements occur for
  each scalar-memory dword returned and each message sent at
  `cdna4/README.md:785` through `:817`.

Rocjitsu evidence:

- The shared wait-counter model increments a selected counter by one per issue
  and decrements by one per completion at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wait_counters.h:76` through `:150`.
- `MemoryPipeline::issue()` calls `wf.wait_counters().increment(issue_counter)`
  once and releases the same counter once on completion at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.h:62` through `:87`.
- Scalar memory uses the `LGKMCNT` pipeline by default at
  `memory_pipeline.h:118` through `:122`.
- `S_SENDMSG` uses an empty shared helper and therefore does not participate in
  `LGKM_CNT` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2427`
  through `:2428`.

Impact:

Functional waiting works for many simple memory dependencies, but CDNA4 scalar
memory dword-count behavior and message-count behavior are not represented.

### CDNA4-RJ-064: Wait-counter ordering restrictions are not modeled

Manual evidence:

- Section 4.4 states that instructions of the same type return in order, that
  mixed reads/writes return in order for a given memory type, and that scalar
  memory reads can return out of order so only `S_WAITCNT 0` is legitimate at
  `cdna4/README.md:785` through `:817`.

Rocjitsu evidence:

- The wait target is a pure threshold over the current counter values at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:299` through `:305` and
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wait_counters.h:153` through `:172`.
- The memory pipeline tracks a single counter type and releases that counter
  when an access completes at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.h:62` through `:87`,
  with no per-instruction-class ordering metadata.

Impact:

Rocjitsu does not validate illegal partial scalar-memory waits or model the
manual's same-type ordering guarantees beyond simple outstanding-count
thresholds.

### CDNA4-RJ-065: Manual wait-state hazards are not modeled or diagnosed

Manual evidence:

- Section 4.5 lists required manually inserted wait states for control/status,
  VALU helper-bit, readlane/lane-select, store data overwrite, SALU-to-VMEM,
  SALU-to-message, DPP, mixed VCC aliasing, RFE/TRAPSTS, M0-to-LDS/addressed
  operations, `S_MOVEREL`, `V_CMPX`, and transcendental hazards at
  `cdna4/README.md:819` through `:867`.

Rocjitsu evidence:

- `S_NOP` dispatches to an empty shared helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopp.cpp:21` through `:29`
  and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2138`.
- The instruction issue path executes functional instruction bodies and wait
  counters, but no static or dynamic check was found for the Chapter 4.5
  hazard table.

Impact:

Rocjitsu can execute instruction sequences that the CDNA4 manual requires
software to separate with NOPs or independent instructions. This is primarily a
timing/diagnostic gap, but it can also hide hazards when rocjitsu is used as an
oracle for hand-written ISA.

### CDNA4-RJ-066: `S_BARRIER` does not expose `STATUS.IN_BARRIER`

Manual evidence:

- Chapter 3 lists `STATUS.IN_BARRIER` among the hardware status bits at
  `cdna4/README.md:467` through `:487`.
- Section 4.3 describes `S_BARRIER` as waiting until other workgroup
  wavefronts reach the same barrier, with early-terminated waves satisfying the
  barrier, at `cdna4/README.md:781` through `:783`.

Rocjitsu evidence:

- `S_BARRIER` dispatches to `execute_s_barrier_sopp`, which sets only the
  internal `WfState::BARRIER` state at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopp.cpp:203` through
  `:212` and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:745`
  through `:747`.
- The CU releases barriers when all non-halted wavefronts in the same
  dispatch/workgroup are either halted or at the barrier at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:313` through `:335`.
- No corresponding update to raw `STATUS.IN_BARRIER` was found.

Impact:

The core multi-wave barrier release can work, but code that reads `STATUS`
during or around a barrier cannot observe the documented barrier status bit.

### CDNA4-RJ-067: Workgroup size limits are not validated against the ISA cap

Manual evidence:

- Section 4.3 says a workgroup can contain up to 16 wavefronts, or 1024
  work-items, at `cdna4/README.md:781` through `:783`.

Rocjitsu evidence:

- Dispatch computes `wfs_per_workgroup` from packet workgroup size and wave
  size at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:941`
  through `:964`.
- Placement checks whether a CU has enough free wavefront slots and LDS at
  `command_processor.cpp:775` through `:797`.
- `ComputeUnitCore::can_accept_workgroup()` compares requested wavefronts
  against free wavefront slots at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:215` through `:225`,
  but no static check was found for the ISA's 16-wavefront / 1024-work-item
  workgroup limit.

Impact:

Oversized CDNA4 workgroups may be accepted or rejected based on emulator
resource configuration rather than the architectural limit.

### CDNA4-RJ-068: Chapter 4 program-flow tests miss side-effect-heavy cases

Rocjitsu evidence:

- The generic instruction execution harness lists `s_waitcnt`, `s_barrier`,
  `s_sleep`, `s_sendmsg`, `s_sendmsghalt`, and `s_rfe` as decode/execute
  surface mnemonics at `tests/instruction_execution_harness_test.cpp:164`
  through `:177`.
- The adjacent plugin barrier test covers the basic two-wave barrier-resolved
  hook path at `tests/execution_plugin_test.cpp:870` through `:886`.
- Static test search found ordinary waitcnt decode/execution fixtures and
  kernel fixtures containing `s_waitcnt` / `s_barrier`, but no focused CDNA4
  coverage for debug branch predicates, RFE/RFE_RESTORE, fork/join stack
  behavior, `S_SENDMSG` `LGKM_CNT`, scalar-memory dword-count waits,
  `STATUS.IN_BARRIER`, or the Chapter 4.5 wait-state hazard table.

Impact:

The current tests can catch basic decode, ordinary waiting, and a simple
barrier release, but they would not fail for most of the Chapter 4 semantic
gaps above.

### CDNA4-RJ-069: SALU allocation-bound fallback and out-of-range destination side effects are not modeled

Manual evidence:

- `cdna4/README.md:948` through `:1022` defines scalar source/destination
  selector behavior. Out-of-range source SGPRs read SGPR0, and out-of-range
  destination SGPRs suppress the SGPR write while still writing SCC and the EXEC
  result of saveexec.

Rocjitsu evidence:

- CDNA4 scalar reads directly read `wf.sgpr_alloc().base + ev` for ordinary
  sources at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/operand.cpp:1443`
  through `:1460`, and scalar writes directly write `base + ev` at `:1501`
  through `:1540`.
- 64-bit writes likewise write both raw physical SGPR slots at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/operand.cpp:1543` through
  `:1572`.
- SCC-producing helpers write the scalar destination through these operands
  before or alongside SCC updates, for example `S_ADD_I32` and `S_ADD_U32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:417`
  through `:440`.

Impact:

An out-of-allocation scalar operand can read/write emulator storage instead of
using the manual's SGPR0/no-write fallback, and destination failures can prevent
the architectural SCC/EXEC side effects that should still occur.

### CDNA4-RJ-070: SALU 64-bit SGPR pair alignment is not enforced

Manual evidence:

- Chapter 5.2 says 64-bit SGPR operands must be aligned to an even SGPR index
  at `cdna4/README.md:1021`.

Rocjitsu evidence:

- `resolve_src_scalar64()` accepts raw 64-bit scalar sources such as `ev <= 105`
  and `108 <= ev <= 122` without checking that `ev` is even at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/operand.cpp:1443` through
  `:1498`.
- `resolve_dst_write64()` similarly writes 64-bit destinations without an even
  alignment check at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/operand.cpp:1543`
  through `:1572`.

Impact:

Odd 64-bit scalar pairs can execute silently in rocjitsu even though the CDNA4
ISA treats them as illegal or constrained.

### CDNA4-RJ-071: SALU implicit SCC operands are not surfaced in C++ def-use metadata

Manual evidence:

- Chapter 5.3 through 5.7 defines many SALU instructions that read or write SCC:
  arithmetic carry/overflow, comparisons, conditional moves/selects, and
  bitwise/nonzero predicates.

Rocjitsu evidence:

- Generated CDNA4 SOP2 constructors expose only explicit operands; for example
  `S_ADD_U32` and `S_ADD_I32` set one destination and two source operands at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sop2.cpp:20` through `:84`,
  but do not override `implicit_defs()` for SCC.
- Def-use analysis depends on explicit operands plus `implicit_defs()` and
  `implicit_uses()` at `lib/rocjitsu/src/rocjitsu/analysis/def_use_chain.cpp:25`
  through `:45`.
- The base instruction hooks are no-ops at
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:215` through `:222`.
- `RegisterSet` declares SCC as a special register class but says it is not
  currently tracked at `lib/rocjitsu/src/rocjitsu/isa/register_set.h:50`
  through `:61`.

Impact:

Analyses, DBT passes, and schedulers using rocjitsu def-use metadata cannot see
SALU SCC producers or consumers even when execution itself reads or writes SCC.

### CDNA4-RJ-072: `S_MAX_{I32,U32}` clears SCC for equal operands

Manual evidence:

- The detailed CDNA4 `S_MAX_I32` and `S_MAX_U32` definitions set SCC with
  inclusive predicates, `S0 >= S1`, at `cdna4/README.md:3940` through `:3953`.

Rocjitsu evidence:

- Shared execution for `S_MAX_I32` and `S_MAX_U32` writes SCC with strict
  `s0 > s1` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1865`
  through `:1898`.
- The scalar SCC test currently locks in SCC=false for equal signed and
  unsigned max operands at `tests/scalar_scc_test.cpp:733` through `:738`.
- The semantic-derivation test likewise requires strict `s0 > s1` and rejects
  `s0 >= s1` at `lib/python/amdisa/tests/test_sema_derive.py:300` through
  `:305`.

Impact:

rocjitsu produces the right max value for equal inputs but the wrong SCC value,
and current tests protect that behavior.

### CDNA4-RJ-073: `S_MOVRELS` / `S_MOVRELD` scale M0 by operand width

Manual evidence:

- The Chapter 5 summary defines `MOVERELS: D = SGPR[S0+M0]` and
  `MOVERELD: SGPR[D+M0] = S0`, says M0 is an unsigned index, and requires an
  even index for 64-bit forms at `cdna4/README.md:1128` through `:1130`.
- The detailed B64 definitions repeat the raw `addr += M0.u32[31:0]` formula
  and evenness requirement at `cdna4/README.md:5298` through `:5355`.

Rocjitsu evidence:

- CDNA4 `S_MOVRELS_B32` / `S_MOVRELS_B64` compute `src_reg` as
  `ssrc0.encoding_value() + index * width_words`, so B64 doubles the M0 index,
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sop1.cpp:812` through
  `:852`.
- `S_MOVRELD_B32` / `S_MOVRELD_B64` compute `dst_reg` with the same
  `index * width_words` pattern at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sop1.cpp:870` through
  `:910`.
- These paths mask M0 to 8 bits and do not validate the manual's evenness
  requirement for the 64-bit source/destination index.

Impact:

CDNA4 B64 relative scalar moves address different SGPR pairs than the manual
pseudocode for nonzero M0 values, and invalid odd-index cases are not caught.

### CDNA4-RJ-074: `S_SETREG_IMM32_B32` hides its literal operand

Manual evidence:

- Chapter 5.8 describes `S_SETREG_IMM32_B32` as using a 32-bit literal constant
  and therefore being a 64-bit instruction at `cdna4/README.md:1142`.
- The detailed instruction definition says the data source is `SIMM32.u32` at
  `cdna4/README.md:4596` through `:4625`.

Rocjitsu evidence:

- CDNA4 `SSetregImm32B32Sopk` stores only the HWREG operand in
  `dst_operands_`, sets `num_src_ = 0`, and does not expose an `OPR_SIMM32`
  source operand at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopk.cpp:369` through
  `:375`; execution later consumes `literal_` at `:384` through `:404`.
- The gfx1250 version exposes a `literal(32, OPR_SIMM32, 0)` source and sets
  `num_src_ = 1` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/gfx1250/sopk.cpp:224` through
  `:233`.

Impact:

Disassembly, operand introspection, def-use, and tests can miss the extension
word even though execution uses it internally.

### CDNA4-RJ-075: SETREG write masks, side effects, and spacing rules are not modeled

Manual evidence:

- Chapter 5.8 requires an `S_NOP` between consecutive `S_SETREG` writes to the
  same hardware register at `cdna4/README.md:1141`.
- Detailed `S_SETREG_B32` and `S_SETREG_IMM32_B32` pseudocode applies
  `HwRegWriteMask(hwRegId, WAVE_STATUS.PRIV)` and notes register-specific side
  effects at `cdna4/README.md:4580` through `:4625`.

Rocjitsu evidence:

- CDNA4 `S_GETREG_B32` / `S_SETREG_B32` still use the Chapter 3 HWREG mapping
  issue recorded as `CDNA4-RJ-052`: reg_id 1 accesses `status_raw()` and only a
  small set of IDs are handled at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopk.cpp:298` through
  `:367`.
- `S_SETREG_B32` and `S_SETREG_IMM32_B32` apply a local bit mask from offset
  and size, but not `HwRegWriteMask`, privilege filtering, register-specific
  side effects, or spacing validation at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopk.cpp:347` through
  `:404`.

Impact:

Even once the HWREG ID map is corrected, SETREG can still permit writes the ISA
would mask or reject, omit side effects, and miss required scheduling
diagnostics.

### CDNA4-RJ-076: VALU source validation allows manual-disallowed extra scalar sources

Manual evidence:

- Chapter 6.2.1 limits VALU to at most one SGPR source and at most one literal,
  with literals disallowed when an SGPR or M0 is used, at
  `cdna4/README.md:1241` through `:1256`.
- The same section says `ADDC`, `SUBB`, and `CNDMASK` implicitly use VCC and
  therefore cannot use an additional SGPR or literal at `cdna4/README.md:1257`
  through `:1259`.

Rocjitsu evidence:

- Generated CDNA4 VOP3 constructors expose broad operand classes independently:
  `V_CNDMASK_B32` uses `OPR_SRC_NOLIT`, `OPR_SRC_SIMPLE`, and an explicit
  `OPR_SREG` third source at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:2167` through
  `:2179`, while `V_ADD_F32` uses the two broad source classes at `:2186`
  through `:2198`.
- Carry-in forms likewise expose broad `SRC0`/`SRC1` classes plus an explicit
  scalar-pair `SRC2` for `V_ADDC_CO_U32` and `V_SUBB_CO_U32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:17583` through
  `:17622`.
- `Operand::read_lane()` resolves non-VGPR operands through scalar/immediate
  fallback independently for each operand at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/operand.cpp:1662` through
  `:1674`; this path does not enforce a per-instruction scalar-source budget.

Impact:

rocjitsu can decode and execute VALU operand combinations that the manual
forbids, especially implicit-VCC instructions with an extra SGPR or literal.

### CDNA4-RJ-077: SDWA/OPSEL next-VALU hazards are not modeled or diagnosed

Manual evidence:

- Chapter 6.2.1 says DOT instructions must not use SDWA or OPSEL, and that a
  VALU instruction using SDWA or OPSEL must not have its result consumed by the
  next VALU instruction; an independent instruction or `V_NOP` is required at
  `cdna4/README.md:1263` through `:1267`.

Rocjitsu evidence:

- rocjitsu has instruction flags for branches, wait counters, barriers, MFMA,
  AccVGPR, and predicated defs, but no generic SDWA/OPSEL hazard flag at
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:24` through `:50`.
- Generated CDNA4 SDWA/OPSEL-bearing instructions execute through normal
  instruction helpers and operand delegates; the instruction metadata has no
  producer/consumer hazard state analogous to the manual's next-VALU rule.

Impact:

Execution and scheduling analyses can accept instruction streams that require a
manual-inserted independent VALU separation on hardware.

### CDNA4-RJ-078: VOP3 floating output modifiers ignore MODE denorm/IEEE gating

Manual evidence:

- Chapter 6.2.2 says VOP3 output modifiers are floating-point-only, are ignored
  for integer or bit results, are ignored when output denormals are enabled,
  flush denormals when output denormals are disabled, flush `-0` to `+0`, and
  are ignored when IEEE mode is set at `cdna4/README.md:1279` through `:1289`.

Rocjitsu evidence:

- Shared `execute_v_add_f32_vop3()` applies `OMOD` and `CLAMP`
  unconditionally from the instruction bits and does not read MODE state at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:3018`
  through `:3056`.
- `Wavefront` stores raw MODE state, but CDNA4/shared VALU execution does not
  read `mode_raw()` except for GPR indexing; static search of the CDNA4 and
  shared ISA paths found only GPR-index uses at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2510`
  through `:2519`.
- CDNA4 `S_SETREG_B32` / `S_SETREG_IMM32_B32` currently handle only reg_id 1
  as `status_raw()`, so even SETREG writes to MODE do not populate the state
  that VALU would need at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopk.cpp:347` through
  `:404`.

Impact:

rocjitsu can apply VOP3 output modifiers when hardware would ignore them, and
it misses denormal flushing and `-0` canonicalization tied to MODE.

### CDNA4-RJ-079: VALU FP round/denorm modes and V_DOT2 denorm flushing are not modeled

Manual evidence:

- Chapter 6.4 defines MODE-controlled VALU FP_ROUND and FP_DENORM behavior and
  notes that floating-point `V_DOT2` ignores those modes and flushes input and
  output denormals at `cdna4/README.md:1479` through `:1487`.

Rocjitsu evidence:

- Shared `S_DENORM_MODE` and `S_ROUND_MODE` helpers are empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1643`
  through `:1645` and `:2422` through `:2425`.
- Floating DOT2 helpers convert F16/BF16 payloads with utility conversions and
  host arithmetic but do not flush input/output denormals explicitly; the
  representative `V_DOT2_F32_F16` and `V_DOT2C_F32_F16` helpers are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:10450`
  through `:10483` and `:10562` through `:10603`.
- As above, CDNA4 SETREG does not write MODE state for CDNA4 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopk.cpp:347` through
  `:404`.

Impact:

rocjitsu's VALU FP results are not sensitive to the documented round/denorm
mode state, and DOT2 can preserve or produce denormals that the manual says are
flushed.

### CDNA4-RJ-080: ALU clamp non-FP semantics are incomplete

Manual evidence:

- Chapter 6.5 says the VOP3 clamp bit signals FP exceptions for `V_CMP`,
  saturates integer results to the representable extrema, and clamps
  floating-point results to `[0.0, 1.0]` at `cdna4/README.md:1489` through
  `:1491`.

Rocjitsu evidence:

- Shared `execute_v_add_u32_vop3()` ignores `inst.inst_.clamp` and performs
  ordinary wrapping integer addition at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:3244`
  through `:3253`.
- The same wrapping behavior is visible in the Chapter 12.11 signed add/sub
  forms: `execute_v_add_i32_vop3()` and `execute_v_sub_i32_vop3()` ignore
  `inst.inst_.clamp` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:3119`
  through `:3130` and `:18689` through `:18700`, while generated
  `VAddI16Vop3` and `VSubI16Vop3` write wrapped true16 results at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:11209` through
  `:11257`.
- Shared compare helpers such as `execute_v_cmp_eq_f32_vop3()` also ignore
  `inst.inst_.clamp`, so they cannot model clamp-as-FP-exception signaling at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:4358`
  through `:4387`.
- The floating helper cited in `CDNA4-RJ-078` implements only the
  floating-point `[0,1]` clamp case.

Impact:

rocjitsu handles ordinary floating-point clamp but not the manual's integer
saturation or compare exception-signaling overloads.

### CDNA4-RJ-091: Literal-only `_MK`/`_AK` VOP2 forms accept modifier-shaped encodings

Manual evidence:

- `V_FMAMK_F32` and `V_FMAAK_F32` say they cannot use VOP3 and cannot use
  input/output modifiers at `cdna4/README.md:7501` through `:7523`.
- `V_MADMK_F16` and `V_MADAK_F16` carry the same no-VOP3/no-modifier rule at
  `cdna4/README.md:7709` through `:7733`.

Rocjitsu evidence:

- XML lists only `VOP2_INST_LITERAL` forms for these four opcodes at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna4.xml:59246` through `:59395`
  and `:61203` through `:61352`, matching the manual's promoted-encoding
  exclusion.
- CDNA4 `Vop2` sizing treats `src0 == SRC_DPP` and `src0 == SRC_SDWA` as
  non-default extension encodings, while also treating opcodes 23, 24, 36, and
  37 as implied-literal instructions at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/encodings.cpp:193` through
  `:205`.
- The generated constructors for `V_FMAMK_F32`, `V_FMAAK_F32`, `V_MADMK_F16`,
  and `V_MADAK_F16` all parse `SRC_DPP`/`SRC_SDWA` extension fields and then
  unconditionally read the same extension word as `simm32_` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop2.cpp:3066` through
  `:3108`, `:3208` through `:3249`, `:4854` through `:4896`, and `:5013`
  through `:5055`.
- Their execute paths apply DPP/SDWA source transforms, partial-write restore,
  destination selection, and SDWA clamp around the literal computation; examples
  are `vop2.cpp:3110` through `:3205`, `:3252` through `:3310`, `:4905`
  through `:5010`, and `:5064` through `:5168`.

Impact:

rocjitsu can decode and execute illegal modifier-shaped encodings for the
literal-only VOP2 FMA/MAD forms, and the modifier control word is also
interpreted as the instruction literal. This can make invalid encodings look
well-formed and produce behavior unrelated to hardware/assembler legality.

### CDNA4-RJ-092: `V_DOT2C_F32_BF16` VOP3 ignores ABS-as-NEG_HI

Manual evidence:

- `V_DOT2C_F32_BF16` says `ABS[1:0]` are used as `NEG_HI[1:0]` during
  translation and that `NEG`/`ABS` input modifiers do not affect the accumulator
  source at `cdna4/README.md:7482` through `:7499`.

Rocjitsu evidence:

- The generated CDNA4 VOP3 body reads packed BF16 halves from `src0` and `src1`,
  accumulates into `vdst`, and never consults `inst_.abs` or `inst_.neg` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:2507` through
  `:2534`.
- Existing `CDNA4-RJ-079` covers DOT denormal flushing, but not this
  instruction-specific modifier remapping.

Impact:

VOP3 encodings that rely on the manual's high-half negation controls execute as
plain positive dot products in rocjitsu, so generated or decoded kernels can be
wrong even when the packed BF16 dataflow and accumulator are otherwise correct.

### CDNA4-RJ-093: FP min/max helpers do not model NaN, signed-zero, or IEEE/MODE tie rules

Manual evidence:

- `V_MIN_F32` and `V_MAX_F32` define signaling-NaN quieting under
  `WAVE_MODE.IEEE`, NaN operand selection, signed-zero tie selection, and
  `V_MAX_F32` IEEE versus non-IEEE equality behavior at `cdna4/README.md:7334`
  through `:7383`.
- `V_MAX_F16` and `V_MIN_F16` define corresponding F16 NaN and signed-zero rules
  at `cdna4/README.md:7809` through `:7864`.
- `V_MIN_F64` and `V_MAX_F64` repeat the same style of signaling-NaN, NaN
  operand-selection, signed-zero, and IEEE-mode equality rules for F64 at
  `cdna4/README.md:17430` through `:17478`.

Rocjitsu evidence:

- Shared F32 VOP2/VOP3 min/max helpers use `util::stdx::fmax`/`fmin` in the SIMD
  path and host `std::fmax`/`std::fmin` in the scalar path, with no
  `WAVE_MODE.IEEE` handling or manual operand-selection logic, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:13877`
  through `:13918` and `:15184` through `:15225`.
- Shared F16 VOP2/VOP3 min/max helpers convert through F32 and likewise use
  `std::fmax`/`std::fmin` at `execute_shared.h:13811` through `:13870` and
  `:15118` through `:15180`.
- Shared F64 VOP3 min/max helpers use `util::stdx::fmax`/`fmin` in the SIMD
  path and host `std::fmax`/`std::fmin` in the scalar path, with no
  `WAVE_MODE.IEEE` handling or manual operand-selection logic, at
  `execute_shared.h:13937` through `:13980` and `:15244` through `:15287`.
- The SIMD correctness test documents that NaN-input lanes and signed-zero ties
  are excluded from comparison because scalar and SIMD paths may diverge there at
  `emulation/rocjitsu/tests/simd_correctness/vop2_minmax_simd_correctness_test.cpp:4`
  through `:19`.

Impact:

Finite non-tie min/max cases have coverage, but rocjitsu does not pin or emulate
the ISA's edge selection rules for NaNs, signaling NaNs, signed-zero ties, or
IEEE-mode equality predicates.

### CDNA4-RJ-094: `V_READFIRSTLANE_B32` returns zero instead of lane 0 when EXEC is disabled

Manual evidence:

- `V_READFIRSTLANE_B32` says `EXEC == 0` forces lane 0 and then reads
  `VGPR[lane][SRC0.u32]`; its notes also say the VGPR read overrides the EXEC
  mask and modifiers are unsupported at `cdna4/README.md:8078` through `:8096`.

Rocjitsu evidence:

- The generated VOP1 body initializes `val = 0`, searches only active lanes, and
  writes the unchanged zero when `EXEC` has no active bits at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop1.cpp:324` through
  `:332`.
- The VOP3 alias repeats the same active-lane-only loop at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:54` through `:64`.
- The VOP1 constructor also accepts literal, DPP, and SDWA extension sentinels at
  `vop1.cpp:239` through `:265`, although XML lists only default VOP1 and VOP3
  encodings for this instruction.
- Existing shared-infra coverage exercises the ordinary active-lane case, not
  the all-disabled EXEC path, at `emulation/rocjitsu/tests/shared_infra_test.cpp:3486`.

Impact:

All-disabled waves can scalarize zero instead of the documented lane-0 VGPR
value, and illegal modifier-shaped encodings can reach a path whose manual
contract is explicitly untyped and no-modifier.

### CDNA4-RJ-095: `V_SAT_PK_U8_I16` decodes but throws in both VOP1 and VOP3 forms

Manual evidence:

- `V_SAT_PK_U8_I16` defines signed 16-bit saturation into two packed unsigned
  8-bit results at `cdna4/README.md:9227` through `:9248`.

Rocjitsu evidence:

- The generated VOP1 executor throws `UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop1.cpp:9962` through
  `:9965`.
- The generated VOP3 executor also throws at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:1920` through
  `:1923`.

Impact:

Kernels using this documented packed saturation instruction cannot execute
through rocjitsu despite generated decode and operand metadata being present.

### CDNA4-RJ-096: `V_PRNG_B32` execute bodies are no-ops

Manual evidence:

- `V_PRNG_B32` defines `D0.u32 = ((in << 1U) ^ (in[31] ? 197U : 0U))` and gives
  the nonzero period at `cdna4/README.md:9345` through `:9356`.

Rocjitsu evidence:

- The generated VOP1 body applies generic extension scaffolding, then never
  writes `VDST` before returning at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop1.cpp:10782` through
  `:10863`.
- The generated VOP3 alias is a one-line no-op at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:2091`.
- Existing data-type tests cover the PRNG helper itself, not the VOP1/VOP3
  instruction bodies.

Impact:

The decoded PRNG instruction leaves the destination unchanged instead of
advancing the documented LFSR sequence.

### CDNA4-RJ-097: Swap-style VOP1 dataflow misses source reads and `V_PERMLANE16_SWAP_B32` misses the second pass

Manual evidence:

- `V_SWAP_B32` reads `D0` and `S0` and writes both operands at
  `cdna4/README.md:9250` through `:9258`.
- `V_PERMLANE16_SWAP_B32` loops `pass in 0 : 1`, swapping lanes 0-15 with 16-31
  and lanes 32-47 with 48-63 at `cdna4/README.md:9358` through `:9370`.
- `V_PERMLANE32_SWAP_B32` similarly reads and writes both operands for lanes
  0-31 and 32-63 at `cdna4/README.md:9380` through `:9389`.

Rocjitsu evidence:

- `VSwapB32Vop1` exposes only `VDST` as a source operand and treats `SRC0` only
  as a destination in public def-use metadata at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop1.cpp:9967` through
  `:9976`, even though its execute body reads `src0`.
- `VPermlane16SwapB32Vop1` and `VPermlane32SwapB32Vop1` set `num_src_ = 0` and
  expose both operands only as destinations at `vop1.cpp:10865` through `:10873`
  and `:10993` through `:11001`; their VOP3 aliases do the same at
  `vop3.cpp:2093` through `:2101` and `:2118` through `:2126`.
- The VOP1 and VOP3 `V_PERMLANE16_SWAP_B32` bodies loop only `lane < 16`, so
  they implement the first row-pair swap but never process the manual's second
  pass for lanes 32-47 and 48-63 at `vop1.cpp:10957` through `:10967` and
  `vop3.cpp:2104` through `:2116`.
- The VOP1 permlane constructors also accept literal, DPP, and SDWA extension
  sentinel forms at `vop1.cpp:10874` through `:10900` and `:11002` through
  `:11028`, even though XML lists only default VOP1 and VOP3 encodings.

Impact:

Schedulers, DBT passes, and dependency analyses can miss real read dependencies
on swap-style source operands, and wave64 execution of `V_PERMLANE16_SWAP_B32`
updates only half of the lanes required by the manual.

### CDNA4-RJ-098: XML-only VOP1 rows generate incomplete or ambiguous runtime behavior

Manual/XML evidence:

- `V_SCREEN_PARTITION_4SE_B32`, `V_EXP_LEGACY_F32`, and `V_LOG_LEGACY_F32` are
  not present in the detailed Chapter 12.8 VOP1 manual definitions or Chapter 13
  VOP1 opcode table; `CDNA4-XML-068` records the source drift.
- XML nevertheless lists `V_SCREEN_PARTITION_4SE_B32` at
  `amdgpu_isa_cdna4.xml:52139`, `V_EXP_LEGACY_F32` at `:54419`, and
  `V_LOG_LEGACY_F32` at `:54533`.

Rocjitsu evidence:

- Generated `V_SCREEN_PARTITION_4SE_B32` VOP1 and VOP3 executors throw
  `UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop1.cpp:6758` through
  `:6761` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:819`
  through `:822`.
- Generated legacy EXP/LOG forms use the ordinary transcendental helper pattern;
  for example `V_EXP_LEGACY_F32` executes through the normal `exp_f32` helper at
  `vop1.cpp:9454` and has only XML's brief "legacy semantics" description as an
  oracle.

Impact:

rocjitsu inherits XML-visible VOP1 surface area that the detailed manual table
does not describe. One such opcode decodes but cannot execute; the legacy EXP/LOG
forms execute through ordinary helper names without enough manual detail in this
slice to validate the intended legacy edge semantics.

### CDNA4-RJ-099: `V_MOV_B32` and `V_MOV_B64` VOP3 aliases ignore allowed floating modifiers

Manual evidence:

- `V_MOV_B32` says floating-point modifiers are valid when the source is a
  32-bit floating-point value, and shows negation/absolute-value examples at
  `cdna4/README.md:8053` through `:8076`.
- `V_MOV_B64` carries the same floating-modifier note for 64-bit floating-point
  values at `cdna4/README.md:8882` through `:8892`.

Rocjitsu evidence:

- Shared `execute_v_mov_b32_vop3()` and `execute_v_mov_b64_vop3()` raw-copy the
  source bits to the destination without consulting `inst_.abs` or `inst_.neg`
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:15912`
  through `:15920` and `:15936` through `:15944`.
- Adjacent VOP3 FP helper bodies explicitly apply `inst_.abs`/`inst_.neg` before
  arithmetic, for example at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:1337` through
  `:1347`, showing the MOV aliases bypass the usual modifier path.
- The current VOP3 unary SIMD test locks in ignored `abs`/`neg` behavior for
  `v_mov_b32`, rather than the manual's modifier behavior.

Impact:

Promoted MOV encodings used as floating negation or absolute-value operations
execute as plain bit copies in rocjitsu.

### CDNA4-RJ-100: VOP1 SDWA wrappers apply floating modifiers before untyped bit operations

Manual evidence:

- `V_NOT_B32` and `V_BFREV_B32` say input and output modifiers are not supported
  because they are untyped operations at `cdna4/README.md:8676` through `:8688`.

Rocjitsu evidence:

- The generated `V_NOT_B32` VOP1 SDWA wrapper feeds `sdwa_src0_sext_` into the
  generic SDWA source selector and applies `sdwa_src0_abs_`/`sdwa_src0_neg_`
  as floating operations before invoking the bitwise helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop1.cpp:5408` through
  `:5451`.
- `V_BFREV_B32` repeats the same pre-wrapper behavior at `vop1.cpp:5528`
  through `:5549`.
- Existing VOP3 modifier sweeps cover ordinary bit-operation helpers, but not
  these VOP1 SDWA modifier-shaped encodings.

Impact:

An SDWA encoding that should be rejected or treated as no-modifier can transform
the integer bit pattern as if it were a floating-point value before `NOT` or
bit-reverse execution.

### CDNA4-RJ-101: E32 VOPC VCC/EXEC writes are invisible to C++ def-use metadata

Manual evidence:

- Chapter 12.9 says compare instructions produce one bit per lane into VCC or
  EXEC at `cdna4/README.md:9414` through `:9422`.
- The class-compare definitions say ordinary `V_CMP_CLASS_*` stores into VCC or
  a scalar register, while `V_CMPX_CLASS_*` stores into EXEC and VCC or a scalar
  register; representative F32 rows are at `cdna4/README.md:9502` through
  `:9581`.

Rocjitsu evidence:

- Generated E32 VOPC classes expose no destination operands: for example
  `VCmpClassF32Vopc` sets `num_dst_ = 0` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vopc.cpp:23` through `:34`.
- The execute helpers write the special state directly, for example
  `execute_v_cmp_class_f32_vopc()` calls `wf.set_vcc(vcc)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:4101`
  through `:4160`, and `VCmpxClassF32Vopc::execute_impl()` writes VCC/EXEC at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vopc.cpp:175` through
  `:270`.
- `InstDefUse` collects explicit destination operands and then
  `implicit_defs()`, but the base `Instruction::implicit_defs()` is empty at
  `lib/rocjitsu/src/rocjitsu/analysis/def_use_chain.cpp:25` through `:39` and
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:219` through `:222`.
- `probe_clobber.cpp` explicitly notes that implicit VCC/EXEC writes from
  `v_cmp` and `v_cmpx` are not modeled by the decoder at
  `lib/rocjitsu/src/rocjitsu/code/patch/probe_clobber.cpp:57` through `:60`.

Impact:

Schedulers, probe insertion, and dependency analysis built from C++ instruction
metadata can miss real VCC and EXEC clobbers for E32 VOPC compares, even though
the runtime mutates those wavefront registers.

### CDNA4-RJ-102: VOP3 `V_CMPX_CLASS_*` writes VCC instead of the explicit SDST

Manual evidence:

- `V_CMPX_CLASS_F32` says the result is stored into EXEC and to VCC or a scalar
  register at `cdna4/README.md:9542` through `:9581`; F64 and F16 class-CMPX
  definitions repeat the same contract at `:9626` through `:9668` and `:9714`
  through `:9755`.

Rocjitsu evidence:

- The generated VOP3 class-CMPX constructor exposes an explicit `OPR_SDST`
  destination, for example `VCmpxClassF32Vop3` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:11554` through
  `:11567`.
- Its execute body computes a result mask, then calls `wf.set_vcc(result)` and
  `wf.set_exec(result)` without writing `vdst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:11569` through
  `:11615`.
- Ordinary relational VOP3 CMPX helpers write the explicit scalar destination
  before updating EXEC, such as `VCmpxEqF32Vop3::execute_impl()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:13556` through
  `:13593`.

Impact:

VOP3 class-CMPX encodings with an explicit scalar destination can leave that
destination stale and clobber VCC instead, diverging from the VOP3 destination
contract and from neighboring CMPX compare helpers.

### CDNA4-RJ-103: VOPC DPP legality and 64-bit exclusions are not enforced

Manual evidence:

- Chapter 13 says a DPP second dword can follow VOP1, VOP2, or VOPC
  instructions at `cdna4/README.md:26198` through `:26204`.
- Chapter 12.16.1 forbids DPP for `V_CMP_CLASS_F64`,
  `V_CMPX_CLASS_F64`, all F64 compare families, and all I64/U64 compare
  families at `cdna4/README.md:24581` through `:24614`.
- The DPP format also says 64-bit input data is limited to `DPP_ROW*` controls
  at `cdna4/README.md:26244`.

Rocjitsu evidence:

- The generator enables DPP for all non-RDNA VOPC encodings by returning true
  for `ENC_VOPC` unless RDNA DPP16 availability needs a special XML check at
  `lib/python/amdisa/codegen/_generator.py:790` through `:813`.
- Generated constructors accept `SRC_DPP` in 64-bit-forbidden VOPC classes,
  including `VCmpClassF64Vopc` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vopc.cpp:287` through
  `:309`, `VCmpLtF64Vopc` at `:8704` through `:8726`, and
  `VCmpxLtI64Vopc` at `:22156` through `:22176`.
- The constructors record DPP control fields but do not reject non-row DPP
  controls for 64-bit sources or reject the instruction-family exclusions.

Impact:

rocjitsu can decode and execute CDNA4 VOPC DPP forms that the manual forbids for
F64 and I64/U64 compares, and 64-bit source paths can accept DPP controls beyond
the manual's row-only restriction.

### CDNA4-RJ-104: VOPC DPP source modifiers are ignored

Manual evidence:

- The DPP second word has `SRC0_NEG`, `SRC0_ABS`, `SRC1_NEG`, and `SRC1_ABS`
  fields at `cdna4/README.md:26206` through `:26217`.

Rocjitsu evidence:

- CDNA4 DPP machine-instruction structs contain these fields, including
  `Vop1VopDppMachineInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h:163`
  through `:176`.
- Generated VOPC constructors load only `vsrc0`, `dpp_ctrl`, row/bank masks,
  and `bound_ctrl`; `VCmpClassF32Vopc` is representative at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vopc.cpp:36` through `:43`.
- The generated DPP preamble calls `amdgpu::dpp::apply_dpp()` with only lane
  permutation and mask fields at `lib/python/amdisa/codegen/_generator.py:6710`
  through `:6720`, and `apply_dpp()` itself reads/permutates raw VGPR data at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h:291`
  through `:335`.

Impact:

DPP encodings that use source negation or absolute-value controls for floating
VOPC compares execute as if those bits were clear.

### CDNA4-RJ-105: VOPC SDWA scalar sources and modifiers are handled with VGPR/float-only paths

Manual evidence:

- The SDWAB second word for VOPC carries `S0` and `S1` scalar-source flags,
  source select/sign-extension fields, and source neg/abs bits at
  `cdna4/README.md:26170` through `:26188`.

Rocjitsu evidence:

- Generated constructors honor the `S0`/`S1` flags in operand type selection,
  for example `VCmpClassF32Vopc` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vopc.cpp:44` through `:59`.
- When a sub-dword selector is active, execution reads
  `wf.vgpr_alloc().base + src_operands_[N]->encoding_value_` directly for both
  sources, so an operand retyped as scalar still uses the VGPR file at
  `vopc.cpp:73` through `:79` and `:93` through `:99`.
- Source neg/abs is applied only inside the sub-dword selector path and is
  implemented by bit-casting the selected 32-bit value to `float`, for example
  at `vopc.cpp:81` through `:89` and `:100` through `:108`, even for integer,
  bit-mask, and F16 VOPC forms.

Impact:

Legal SDWA scalar-source forms can read the wrong register file when source
selection is used, and SDWA modifier-shaped encodings can apply type-blind F32
transforms before integer or F16 compare helpers.

### CDNA4-RJ-106: `V_DOT2_F32_BF16` executes through the F16 dot path and modifies `S2`

Manual/XML evidence:

- `V_DOT2_F32_BF16` multiplies BF16 halves from `S0` and `S1`, adds `S2.f32`,
  and states that `NEG` and `ABS` input modifiers do not affect `S2` at
  `cdna4/README.md:11890` through `:11903`.
- The XML record preserves this operand shape, with `SRC0` and `SRC1` as
  `FMT_NUM_PK2_BF16` and `SRC2` as `FMT_NUM_F32` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna4.xml:73582` through
  `:73613`.

Rocjitsu evidence:

- The semantic derivation maps `V_DOT2_F32_BF16` to `dot2_f32_f16` at
  `lib/python/amdisa/semantics.py:1615` through `:1617`.
- The generated CDNA4 helper named `execute_v_dot2_f32_bf16_vop3p()` invokes the
  F16 SIMD hook, converts BF16 source halves with `util::f16_to_f32()`, and
  negates the accumulator when `inst.inst_.neg & 4` is set at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:10414`
  through `:10444`.
- The SIMD table comments explicitly say the BF16 body is byte-identical to the
  F16 body and widens via `util::f16_to_f32` at
  `lib/python/amdisa/codegen/execute/simd_codegen.py:1278` through `:1288`.

Impact:

BF16 payloads are decoded as IEEE half-precision values rather than BF16, and
`NEG[2]` can change the F32 accumulator source even though the manual says
source modifiers do not affect `S2`.

### CDNA4-RJ-107: Packed F16 min/max3 ignore `DX10_CLAMP` NaN-to-zero MODE behavior

Manual evidence:

- The MODE register defines `DX10_CLAMP` as vector-ALU NaN-to-zero behavior and
  `IEEE` as signaling-NaN propagation/quieting state at `cdna4/README.md:491`
  through `:500`.
- `V_PK_MINIMUM3_F16` and `V_PK_MAXIMUM3_F16` say signaling NaNs propagate, then
  add that `DX10_CLAMP` forces NaNs to zero and `IEEE` is forced to 1 for these
  operations at `cdna4/README.md:11905` through `:11937`.

Rocjitsu evidence:

- `VPkMinimum3F16Vop3p::execute_impl()` hard-codes an IEEE-style pairwise
  minimum helper that returns quiet NaN for any NaN input, then writes the F16
  result without inspecting `wf.mode_raw()` or zeroing NaNs at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:399` through
  `:447`.
- `VPkMaximum3F16Vop3p::execute_impl()` has the same hard-coded IEEE-style NaN
  propagation and no MODE/DX10 handling at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:466` through
  `:514`.

Impact:

When MODE.DX10_CLAMP is set, rocjitsu can preserve or produce NaN results where
hardware is documented to force NaN to zero for these packed min/max3
instructions.

### CDNA4-RJ-108: `V_ACCVGPR_READ/WRITE` do not set the ACCVGPR instruction flag

Manual/XML evidence:

- The CDNA4 VOP3P opcode table lists `V_ACCVGPR_READ` and `V_ACCVGPR_WRITE` at
  opcodes 88 and 89 at `cdna4/README.md:26101` through `:26102`.
- XML gives those records `_B32` aliases and the expected accumulator operand
  classes at `shared/machine-readable-isa/isa/amdgpu_isa_cdna4.xml:76209`
  through `:76272`.

Rocjitsu evidence:

- `InstFlags::ACCVGPR` is documented as the flag for `v_accvgpr_write`,
  `v_accvgpr_read`, and `v_accvgpr_mov`, and `Instruction::is_accvgpr()` exposes
  it at `lib/rocjitsu/src/rocjitsu/isa/instruction.h:44` through `:47` and
  `:204` through `:206`.
- The generator sets this flag only when `inst.name` is one of the `_B32`
  aliases at `lib/python/amdisa/codegen/_generator.py:6545` through `:6549`.
- Generated CDNA4 constructors use the primary XML names
  `V_ACCVGPR_READ`/`V_ACCVGPR_WRITE`, construct the correct accumulator
  operands, and execute lane copies, but they never set `flags_ |= ACCVGPR` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:792` through
  `:832`.

Impact:

Execution dataflow is present, but instruction-metadata consumers using
`Instruction::is_accvgpr()` can miss the explicit accumulator move opcodes.

### CDNA4-RJ-109: `V_PK_FMAC_F16` VOP3A is generated as a stub and misses the accumulator read

Manual/XML evidence:

- `V_PK_FMAC_F16` multiplies two packed half-precision inputs and accumulates
  both half results into the destination register at `cdna4/README.md:15409`
  through `:15415`.
- XML records the VOP3A form as opcode 316 with `VDST`, `SRC0`, and `SRC1`
  operands at `shared/machine-readable-isa/isa/amdgpu_isa_cdna4.xml:64404`
  through `:64506`. `CDNA4-XML-078` records the separate XML dataflow gap where
  `VDST` is not marked as an input.

Rocjitsu evidence:

- Semantic derivation maps `V_PK_FMAC_F16` to `nop` at
  `lib/python/amdisa/semantics.py:971` through `:972`.
- Generated `VPkFmacF16Vop3` exposes only `SRC0` and `SRC1` as source
  operands, does not add `VDST` to `src_operands_`, and throws
  `UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:3592` through
  `:3608`.
- The neighboring `VFmacF32Vop3` constructor has the expected accumulator
  pattern, putting `vdst` in both `src_operands_` and `dst_operands_`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:3575` through
  `:3589`.

Impact:

The valid CDNA4 VOP3A opcode decodes but cannot execute, and static dependency
metadata misses the destination accumulator read that the manual requires.

### CDNA4-RJ-110: Several valid native VOP3A opcodes decode to hard stubs

Manual/XML evidence:

- `V_QSAD_PK_U16_U8`, `V_MQSAD_PK_U16_U8`, and `V_MQSAD_U32_U8` have detailed
  SAD/MSAD pseudocode at `cdna4/README.md:16015` through `:16051`, and XML
  records the matching VOP3A operand shapes at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna4.xml:66439` through
  `:66572`.
- `V_TRIG_PREOP_F64` has detailed argument-reduction pseudocode at
  `cdna4/README.md:17668` through `:17690`, with XML metadata at
  `amdgpu_isa_cdna4.xml:70084` through `:70108`.
- `V_CVT_PKNORM_I16_F16` and `V_CVT_PKNORM_U16_F16` have packed normalized
  conversion pseudocode at `cdna4/README.md:17766` through `:17785`, with XML
  metadata at `amdgpu_isa_cdna4.xml:70392` through `:70471`.

Rocjitsu evidence:

- Generated `VQsadPkU16U8Vop3`, `VMqsadPkU16U8Vop3`, and `VMqsadU32U8Vop3`
  constructors exist, but each `execute_impl()` throws `UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:4213` through
  `:4270`.
- `VTrigPreopF64Vop3` likewise decodes to a generated class whose
  `execute_impl()` throws at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:11010` through
  `:11025`.
- `VCvtPknormI16F16Vop3` and `VCvtPknormU16F16Vop3` also throw at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:11127` through
  `:11160`.

Impact:

These native CDNA4 VOP3A instructions have manual and XML semantic records and
decode as concrete rocjitsu classes, but runtime execution fails immediately.

### CDNA4-RJ-111: VOP3B `SDST=VCC` writes are invisible to register metadata

Manual/XML evidence:

- Chapter 13.3.5 defines VOP3B as the encoding with a unique scalar
  destination and lists the ten scalar-destination opcodes at
  `cdna4/README.md:25895` through `:25978`.
- XML records `SDST` as a scalar mask destination for the VOP3B records, using
  `OPR_SREG` for carry/MAD forms and `OPR_SDST` for div-scale forms, for
  example at `shared/machine-readable-isa/isa/amdgpu_isa_cdna4.xml:59539`
  through `:59543`, `:66181` through `:66185`, and `:66603` through `:66607`.

Rocjitsu evidence:

- Generated VOP3B constructors mirror those operand classes: carry and MAD
  `sdst` operands use `OPR_SREG`, while `V_DIV_SCALE_*` use `OPR_SDST`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:17530` through
  `:17671` and `:17692` through `:17713`.
- Runtime writes go through `inst.sdst`, and the scalar-write resolver handles
  selector 106 as VCC at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2883`
  through `:2885` and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/operand.cpp:1517` through
  `:1556`.
- `Operand::to_register_ref()` only returns `RegisterRef` values for the SGPR
  numeric ranges of `OPR_SDST` and `OPR_SREG`, and drops special scalar
  selectors such as VCC, M0, and EXEC at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/operand.cpp:1088` through
  `:1122` and `:1249` through `:1255`.

Impact:

An encoding with explicit `SDST=VCC` can update VCC during execution, but
liveness, def-use, and probe-clobber analysis built from `RegisterRef` metadata
can miss the explicit VCC clobber.

### CDNA4-RJ-112: VOP3A F32 min/max3 ignore `DX10_CLAMP` NaN-to-zero MODE behavior

Manual/XML evidence:

- `V_MINIMUM3_F32` and `V_MAXIMUM3_F32` say signaling NaNs propagate, then add
  that `DX10_CLAMP` forces NaNs to zero and `IEEE` is forced to 1 for these
  operations at `cdna4/README.md:18004` through `:18030`.
- XML records the opcodes and F32 operand shapes for these two VOP3A
  instructions at `amdgpu_isa_cdna4.xml:71008` through `:71099`, but the MODE
  behavior is prose-only and recorded as
  `CDNA4-XML-081`.

Rocjitsu evidence:

- Generated CDNA4 `VMinimum3F32Vop3::execute_impl()` and
  `VMaximum3F32Vop3::execute_impl()` dispatch directly into the shared helpers
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:11514`
  through `:11534`.
- `execute_v_maximum3_f32_vop3()` returns quiet NaN if any input is NaN and
  later applies only the VOP3 `clamp` bit at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:14230`
  through `:14284`.
- `execute_v_minimum3_f32_vop3()` has the same NaN propagation and ordinary
  VOP3 `clamp` handling at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:15537`
  through `:15591`.

Impact:

When MODE.DX10_CLAMP is set, rocjitsu can preserve or produce NaN results for
the scalar F32 min/max3 VOP3A instructions even though the CDNA4 manual says
NaNs are forced to zero for these operations.

### CDNA4-RJ-113: `V_CVT_PKRTZ_F16_F32` uses round-to-nearest conversion helpers

Manual/XML evidence:

- `V_CVT_PKRTZ_F16_F32` says the packed F32-to-F16 conversion uses
  round-toward-zero semantics, ignores the current rounding mode, and
  temporarily sets `ROUND_TOWARD_ZERO` in the pseudocode at
  `cdna4/README.md:17726` through `:17737`.
- XML preserves the same RTZ description and VOP3 operand shape for opcode 662
  at `amdgpu_isa_cdna4.xml:70260` through `:70295`.

Rocjitsu evidence:

- Generated `execute_v_cvt_pkrtz_f16_f32_vop3()` calls
  `util::f32_to_f16_simd()` in the SIMD path and `util::f32_to_f16()` in the
  scalar fallback at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:9825`
  through `:9840`.
- `util::f32_to_f16()` implements round-to-nearest-even by adding the
  tie/sticky round bit at `lib/util/include/util/data_types.h:93` through
  `:132`, while a separate `util::f32_to_f16_rtz()` helper exists at
  `:135` through `:155`.
- The SIMD helper is documented as a bit-exact port of `f32_to_f16` at
  `lib/util/include/util/simd.h:470` through `:476`.
- The generator source for `vector_cvt_pkrtz_f16_f32` emits
  `util::f32_to_f16()` calls at
  `lib/python/amdisa/codegen/execute/vector_special.py:1006` through `:1016`.

Impact:

Halfway and otherwise non-exact F32 inputs can round to the nearest half value
instead of truncating toward zero, so both scalar execution and SIMD dispatch
can disagree with the CDNA4 `V_CVT_PKRTZ_F16_F32` contract.

### CDNA4-RJ-114: `V_CVT_PKACCUM_U8_F32` old-destination read is hidden from metadata

Manual/XML evidence:

- `V_CVT_PKACCUM_U8_F32` writes one selected destination byte and explicitly
  says the opcode uses `src_c` to pass the destination in as a source at
  `cdna4/README.md:16236` through `:16246`.
- XML describes the instruction as storing one selected byte into the
  destination register, but marks `VDST` output-only and lists only `SRC0` and
  `SRC1` as inputs at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna4.xml:67001` through
  `:67033`.

Rocjitsu evidence:

- Generated `VCvtPkaccumU8F32Vop3` exposes only `SRC0` and `SRC1` as source
  operands and does not put `VDST` in `src_operands_` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:4610` through
  `:4621`.
- The class declaration has only constructor and `execute_impl()` methods, with
  no `implicit_uses()` override, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.h:1652` through
  `:1658`.
- Runtime execution reads the old destination, updates only the selected byte,
  and writes the merged result at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:4623` through
  `:4633`.

Impact:

The value path preserves the other destination bytes, but liveness, def-use,
and probe-clobber analysis can treat `V_CVT_PKACCUM_U8_F32` as a pure write and
miss the old-`VDST` dependency that the generated execution body actually
consumes.

### CDNA4-RJ-115: `V_LSHL_ADD_U64` treats unsupported shift counts as masked counts

Manual/XML evidence:

- `V_LSHL_ADD_U64` says the shift count must be between 0 and 4, and the notes
  say unsupported shift counts are treated as a shift of zero at
  `cdna4/README.md:16524` through `:16536`.
- XML records the VOP3 operand shape for opcode 520 but has no operand or
  semantic field that captures the unsupported-count-as-zero rule at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna4.xml:68195` through
  `:68227`; see `CDNA4-XML-085`.

Rocjitsu evidence:

- Generated `execute_v_lshl_add_u64_vop3()` calls `lshl_masked()` on `SRC0`
  and `SRC1` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:12745`
  through `:12757`.
- The shared 64-bit helper masks the count with `& 63u`, not with a CDNA4
  `0..4` legality check, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/simd_glue.h:56` through
  `:58`.
- The SIMD codegen comment pins the same `(src0 << (src1 & 63)) + src2`
  behavior for `v_lshl_add_u64_vop3` at
  `lib/python/amdisa/codegen/execute/simd_codegen.py:2483` through `:2485`.

Impact:

Shift counts 5 through 63 produce real left shifts in rocjitsu, while the CDNA4
manual says unsupported counts should behave as shift zero. Kernels that rely
on the hardware fallback value can get different 64-bit results in both scalar
fallback and SIMD dispatch.

### CDNA4-RJ-116: Packed F32-input VOP3A conversions ignore source modifiers

Manual/XML evidence:

- Chapter 6.2.1 says instructions using the VOP3 form with floating-point
  inputs can apply `ABS` and `NEG` to input operands at `cdna4/README.md:1261`,
  and the VOP3A field map carries per-source `ABS` and `NEG` fields at
  `cdna4/README.md:25550` and `:25601` through `:25603`.
- XML records native VOP3A packed conversion forms with F32 inputs for
  `V_CVT_PK_U8_F32`, `V_CVT_PKACCUM_U8_F32`, `V_CVT_PK_F16_F32`,
  `V_CVT_PK_BF16_F32`, `V_CVT_PKNORM_I16_F32`, `V_CVT_PKNORM_U16_F32`, and
  `V_CVT_PKRTZ_F16_F32` at `amdgpu_isa_cdna4.xml:64397`, `:65381`,
  `:69154`, `:69198`, `:70172`, `:70216`, and `:70260`.

Rocjitsu evidence:

- Shared execution for `V_CVT_PK_U8_F32`, `V_CVT_PKACCUM_U8_F32`,
  `V_CVT_PKNORM_I16_F32`, `V_CVT_PKNORM_U16_F32`, and
  `V_CVT_PKRTZ_F16_F32` bit-casts raw source reads directly to F32 and never
  consults `inst_.abs` or `inst_.neg`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:9719`
  through `:9840`.
- Generated CDNA4 bodies for `V_CVT_PK_F16_F32` and `V_CVT_PK_BF16_F32` do the
  same raw source reads before conversion at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:10598` through
  `:10648`.
- Existing `CDNA4-RJ-113` covers the separate `V_CVT_PKRTZ_F16_F32` rounding
  issue; this finding is about legal source modifier bits.

Impact:

Legal VOP3A encodings that negate or take the absolute value of the F32
conversion inputs execute as if the modifier bits were clear. That affects
integer-packed, normalized-packed, and F16/BF16 packed conversion results.

### CDNA4-RJ-117: Promoted VOP3 conversion aliases drop conversion modifiers

Manual/XML evidence:

- Chapter 12.8 says VOP1 instructions may also be encoded as VOP3 to access
  extra control bits such as `ABS` and `OMOD` at `cdna4/README.md:8035`; the
  later VOP3A section repeats the same promoted VOP1 conversion rows in the
  VOP3 opcode table at `cdna4/README.md:25276` through `:25290` and
  `:25614` through `:25628`.
- Chapter 6.2.1 says VOP3 floating-point inputs can use `ABS`/`NEG`, and
  Chapter 6.2.2 says VOP3 instructions with floating-point results can use
  `OMOD` and `CLAMP`, subject to MODE restrictions, at `cdna4/README.md:1261`
  and `:1287` through `:1289`.
- XML records promoted `ENC_VOP3` conversion forms with floating inputs or
  floating results, for example `V_CVT_F32_I32`, `V_CVT_U32_F32`,
  `V_CVT_I32_F32`, `V_CVT_F16_F32`, `V_CVT_F32_F16`, and `V_CVT_F32_F64`, at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna4.xml:47105`, `:47333`,
  `:47447`, `:47561`, `:47675`, and `:48131`.

Rocjitsu evidence:

- Some promoted conversion helpers with floating inputs read the raw source and
  ignore `ABS`/`NEG`, such as `execute_v_cvt_i32_f32_vop3()` and
  `execute_v_cvt_u32_f32_vop3()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:9307`
  through `:9325` and `:9964` through `:9980`.
- Promoted conversion helpers with floating results write the converted value
  directly and do not apply `OMOD` or `CLAMP`, including
  `execute_v_cvt_f32_f64_vop3()`, `execute_v_cvt_f32_i32_vop3()`,
  `execute_v_cvt_f32_u32_vop3()`, `VCvtF16F32Vop3::execute_impl()`, and
  `VCvtF32F16Vop3::execute_impl()` at
  `execute_shared.h:8813` through `:8895` and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:173` through
  `:214`.
- `execute_v_cvt_off_f32_i4_vop3()` does apply VOP3 `OMOD` and `CLAMP` at
  `execute_shared.h:9540` through `:9566`, showing the omission is not an
  intentional blanket rule for all conversion-style helpers. Existing
  `CDNA4-RJ-078` covers MODE gating when modifiers are applied; this finding
  covers promoted conversion paths where the modifiers are not applied at all.

Impact:

Promoted VOP3 conversion encodings can ignore legal source modifiers for
floating inputs, and can ignore legal output modifiers for floating results.
That makes the VOP3 aliases semantically weaker than the manual's advertised
extra-control-bit form.

### CDNA4-RJ-118: Chapter 12.16 DPP/SDWA exclusions are not enforced for generated VOP1/VOP2 paths

Manual evidence:

- Chapter 12.16.1 lists instructions that cannot use DPP, including F64 VOP1
  conversions and unary operations, `V_CLREXCP`, `V_SWAP_B32`, the literal-only
  `_MK`/`_AK` forms, `V_READFIRSTLANE_B32`, and 64-bit VOPC compare families at
  `cdna4/README.md:24581` through `:24614`.
- Chapter 12.16.2 lists instructions that cannot use SDWA, including
  `V_MAC_F16`, `V_FMAC_F32`, `V_CLREXCP`, `V_SWAP_B32`, the literal-only
  `_MK`/`_AK` forms, and `V_READFIRSTLANE_B32` at `cdna4/README.md:24615`
  through `:24629`.

Rocjitsu evidence:

- CDNA4 `Vop1` and `Vop2` helper methods classify `SRC_DPP` and `SRC_SDWA` by
  source-selector sentinel only, without an instruction-local Chapter 12.16
  legality check, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/encodings.cpp:139` through
  `:147` and `:193` through `:201`.
- Generated F64 VOP1 constructors accept the manual-forbidden DPP sentinel and
  execution applies DPP; `VCvtI32F64Vop1` is representative at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop1.cpp:359` through
  `:410`. The same generated pattern appears in the other DPP-forbidden F64
  VOP1 rows, including `VCvtF64I32Vop1` at `:482`,
  `VCvtF32F64Vop1` at `:1711`, `VTruncF64Vop1` at `:2683`,
  `VSqrtF64Vop1` at `:4956`, `VFrexpExpI32F64Vop1` at `:5962`, and
  `VFractF64Vop1` at `:6232`.
- `VSwapB32Vop1` parses both `SRC_DPP` and `SRC_SDWA`, then applies the generic
  DPP/SDWA preamble during execution at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop1.cpp:9967` through
  `:10018`.
- `VClrexcpVop1` has no declared source operands, but its generated execute body
  still branches on `SRC_DPP`/`SRC_SDWA` and enters the generic DPP/SDWA source
  handling at `vop1.cpp:6633` through `:6652`.
- `VMacF16Vop2` and `VFmacF32Vop2` parse `SRC_SDWA` and execute the generic SDWA
  preamble despite the Chapter 12.16.2 ban, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop2.cpp:4706` through
  `:4768` and `:8010` through `:8068`.
- Existing findings cover overlapping rows: `CDNA4-RJ-091` tracks the
  literal-only `_MK`/`_AK` forms, `CDNA4-RJ-094` tracks
  `V_READFIRSTLANE_B32`, and `CDNA4-RJ-103` tracks VOPC DPP legality and
  64-bit compare exclusions.

Impact:

rocjitsu can decode and execute, or in the operand-less `V_CLREXCP` case enter
generic extension handling for, DPP/SDWA encodings that Chapter 12.16 marks
illegal. That can make invalid modifier-extension words look like real
instruction behavior instead of decode-time rejections.

### CDNA4-RJ-119: Device-memory consistency and acknowledgment behavior is not represented

Manual evidence:

- Section 2.3 describes the CDNA device-memory hierarchy, cache-less loads,
  load-clause overlap caching, write-combining, atomic pre-op return storage,
  write-confirmation acknowledgments, relaxed consistency, per-PE/per-channel
  scatter-write ordering, and acknowledgment/fence use at
  `cdna4/README.md:382` through `:393`.

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

### CDNA4-RJ-081: VGPR indexing uses the wrong M0 layout and cannot honor source-role masks

Manual evidence:

- Chapter 6.6 defines `M0[7:0]` as the index and `M0[15:12]` as
  dest/src2/src1/src0 enable bits, with indexing applying only to VGPR
  operands and indexed out-of-range VGPRs illegal at `cdna4/README.md:1493`
  through `:1522`.
- Chapter 6.6.2 gives instruction-specific role remapping for readlane,
  writelane, MAC/MAD, reverse shifts, `v_cvt_pkaccum`, and SDWA
  read-modify-write at `cdna4/README.md:1524` through `:1538`.

Rocjitsu evidence:

- `Wavefront::gpr_idx_mode()` reads `(m0 >> 8) & 0xF`, and
  `execute_s_set_gpr_idx_mode_sopp()` / `execute_s_set_gpr_idx_on_sopc()` write
  the mode nibble at `M0[11:8]`, not `M0[15:12]`, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:248` through `:251` and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2500`
  through `:2519`.
- `apply_gpr_idx()` applies any low source-enable bit to all source operands and
  only distinguishes destination versus non-destination at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:584` through `:589`.
- CDNA4 operand reads/writes pass only a boolean source/destination role into
  that helper at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/operand.cpp:1662`
  through `:1710`, so they cannot distinguish src0, src1, src2, or the manual's
  instruction-specific remaps.

Impact:

CDNA4 indexed VGPR accesses use different M0 bits from the manual and can index
the wrong operands, while illegal indexed out-of-range accesses are not
diagnosed.

### CDNA4-RJ-082: `V_CVT_SR_{F16,BF16}_F32` ignores stochastic seed and destination `OPSEL`

Manual/XML evidence:

- `V_CVT_SR_F16_F32` and `V_CVT_SR_BF16_F32` use `SRC1` as stochastic rounding
  seed data, use `OPSEL[3]` to select which 16-bit destination word is written,
  preserve the unwritten bytes, force round-nearest-even, and ignore OMOD/clamp
  at `cdna4/README.md:17958` through `:18002`.
- The CDNA4 XML records `SRC1` as an input and describes the OPSEL-selected
  16-bit destination write at `shared/machine-readable-isa/isa/amdgpu_isa_cdna4.xml:72540`
  through `:72612`.

Rocjitsu evidence:

- The generated CDNA4 constructors carry `src1`, but `execute_impl()` for
  `VCvtSrF16F32Vop3` and `VCvtSrBf16F32Vop3` only read `src0` and then write
  `util::f32_to_f16(s0)` / `util::f32_to_bf16(s0)` directly to `vdst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:11441` through
  `:11496`.
- The generator has the same issue: the `vector_cvt_sr_f16_f32` comment says to
  use `src1` as random bits, but the emitted code never reads it; the BF16 form
  also lowers to ordinary truncating `f32_to_bf16` at
  `lib/python/amdisa/codegen/execute/vector_special.py:1214` through `:1228`.
- Stochastic helpers for these formats do exist at
  `lib/util/include/util/data_types.h:991` through `:998`, and packed
  stochastic F16/BF16 codegen already uses seed operands at
  `lib/python/amdisa/codegen/execute/vector_special.py:1168` through `:1199`.
- The adjacent semantic test only proves a derivation block exists for
  `V_CVT_SR_F16_F32` at `lib/python/amdisa/tests/test_sema_derive.py:2484`
  through `:2487`; no checked output asserts the seed, destination-half
  selection, or preservation contract.

Impact:

Seed-dependent F16/BF16 stochastic conversion results are deterministic ordinary
conversions in rocjitsu, and high-half `OPSEL` cases can overwrite or clear the
wrong 16-bit destination word instead of preserving the other half.

### CDNA4-RJ-083: VOP1 FP8/BF8 widening converts apply SDWA fields that should be ignored

Manual/XML evidence:

- Table 31 says `CVT_PK_F32_FP8`, `CVT_PK_F32_BF8`, `CVT_F32_FP8`, and
  `CVT_F32_BF8` ignore `abs`, `neg`, and `sext` at `cdna4/README.md:2320`
  through `:2325`.
- Section 7.3 says converts from 8-bit formats use SDWA only for the source
  VGPR and `SRC0_SELECT`, and that other SDWA fields are ignored at
  `cdna4/README.md:2374` through `:2376`.
- The instruction definitions limit the meaningful control to BYTE or WORD
  source selection and non-SDWA BYTE0/WORD0 defaults at `cdna4/README.md:9268`
  through `:9343`.

Rocjitsu evidence:

- Generated VOP1 constructors record the generic SDWA source modifiers,
  destination select, destination-unused, and clamp fields for these converts,
  for example `VCvtF32Fp8Vop1` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop1.cpp:10237` through
  `:10250` and `VCvtPkF32Fp8Vop1` at `:10496` through `:10510`.
- The generated execute bodies feed `sdwa_src0_sext_` into generic
  `sdwa_src_select()`, apply `sdwa_src0_abs_`/`sdwa_src0_neg_`, and then install
  the selected temporary source at `vop1.cpp:10266` through `:10286` and
  `:10525` through `:10545`.
- After writing the converted F32 result, the same bodies apply generic
  `sdwa_dst_merge()` when `sdwa_dst_sel_` is not DWORD at `vop1.cpp:10332`
  through `:10340`, `:10595` through `:10605`, and `:10730` through `:10740`.
- The adjacent HIP conversion tests cover default finite conversion paths, not
  SDWA modifier/destination ignored-field cases.

Impact:

SDWA encodings that should only choose a source byte or word can change the
converted value or merge only part of the F32 destination in rocjitsu, diverging
from the manual's ignored-field contract for VOP1 FP8/BF8 widening converts.

### CDNA4-RJ-084: `S_SET_VALU_COEXEC_MODE` decodes but does not update co-execution state

XML evidence:

- CDNA4 XML records `S_SET_VALU_COEXEC_MODE` as `ENC_SOPP` opcode 31 with a
  `SIMM16` operand at `amdgpu_isa_cdna4.xml:46663` through `:46675`.
- The XML description says the instruction sets vector ALU co-execution mode
  from `SIMM16[1:0]` for the next VALU instruction and clears that mode after
  the next VALU instruction issues at `amdgpu_isa_cdna4.xml:46663` through
  `:46664`.

Rocjitsu evidence:

- Generated CDNA4 decoding includes opcode 31 in `sub_decode_sopp` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/decoder.cpp:7433` through
  `:7434`.
- The generated constructor exposes the `SIMM16` operand, but
  `SSetValuCoexecModeSopp::execute_impl()` is a no-op at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopp.cpp:461` through
  `:470`.
- Static source search found no CDNA4 wavefront or VALU issue state for this
  one-instruction co-execution mode.

Impact:

Code using this XML-described SOPP encoding decodes and disassembles, but it
has no effect on the following VALU instruction in rocjitsu.

### CDNA4-RJ-085: SOP1 no-literal forms are accepted as 8-byte literal encodings

Manual/XML evidence:

- `S_GETPC_B64` stores `PC + 4` and the manual says the instruction must be 4
  bytes at `cdna4/README.md:5083` through `:5095`.
- `S_SWAPPC_B64` saves `PC + 4` and the manual says the instruction must be 4
  bytes at `cdna4/README.md:5109` through `:5123`.
- `S_SETPC_B64` and `S_RFE_B64` take scalar-register byte addresses at
  `cdna4/README.md:5097` through `:5107` and `:5125` through `:5135`.
- `S_MOVRELS_B32/B64` use the raw `SRC0` register index from the instruction
  plus M0 at `cdna4/README.md:5284` through `:5318`.
- CDNA4 XML provides only the default `ENC_SOP1` encoding for these six
  opcodes, with no `SOP1_INST_LITERAL` alternative. The PC forms use
  `OPR_SREG` or implicit `OPR_PC` at `amdgpu_isa_cdna4.xml:33555` through
  `:33695`; `S_MOVRELS_B32/B64` use `OPR_SREG` plus implicit M0 at
  `amdgpu_isa_cdna4.xml:34545` through `:34614`.

Rocjitsu evidence:

- The shared CDNA4 `Sop1` base class sets `size_ += sizeof(MachineInst)` for
  every SOP1 instruction when `SSRC0 == 255`, and exposes `has_lit_0()` with no
  opcode-specific filter at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/encodings.cpp:24` through
  `:36`.
- `S_GETPC_B64` does not consume `SSRC0`, but writes `wf.pc + size_`; an
  encoding with `SSRC0 == 255` is therefore treated as 8 bytes and stores
  `PC + 8` instead of the manual's required `PC + 4` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sop1.cpp:545` through
  `:550`.
- `S_SETPC_B64`, `S_SWAPPC_B64`, and `S_RFE_B64` rewrite `SSRC0 == 255` into
  an `OPR_SIMM32` operand even though XML only allows scalar registers at
  `sop1.cpp:553` through `:603`; `S_SWAPPC_B64` also saves `wf.pc + size_` at
  `:587` through `:590`.
- `S_MOVRELS_B32/B64` similarly rewrite `SSRC0 == 255` into a literal operand,
  then derive the indexed SGPR from `ssrc0.encoding_value()` at
  `sop1.cpp:797` through `:842`, rather than rejecting the no-literal form.

Impact:

Illegal or reserved SOP1 encodings with `SSRC0 == 255` can consume an extra
dword, desynchronize subsequent decoding, and produce wrong `PC + 4` values or
literal-based branch/index behavior for instructions whose XML/manual contract
has no literal form.

### CDNA4-RJ-086: `S_RFE_RESTORE_B64` accepts literal selectors despite having no literal encoding

Manual/XML evidence:

- CDNA4 XML records `S_RFE_RESTORE_B64` as `ENC_SOP2` opcode 43 with 64-bit
  `SSRC0`, 32-bit `SSRC1`, and implicit PC output at
  `amdgpu_isa_cdna4.xml:41300` through `:41327`.
- The same XML instruction has no `SOP2_INST_LITERAL` alternative; in the full
  audited SOP2 inventory, opcode 43 was the only XML `ENC_SOP2` record without
  any literal-encoding variant.
- The detailed CDNA4 SOP2 manual inventory omits this opcode entirely; the
  manual/XML source drift is recorded separately as `CDNA4-XML-060`.

Rocjitsu evidence:

- The shared CDNA4 `Sop2` base class treats every SOP2 instruction with
  `SSRC0 == 255` or `SSRC1 == 255` as an 8-byte literal form, with no
  opcode-specific filter, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/encodings.cpp:78` through
  `:94`.
- `SRfeRestoreB64Sop2` rewrites either source selector 255 into `OPR_SIMM32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sop2.cpp:1011` through
  `:1028`, even though the XML record has only the default encoding.

Impact:

Reserved or unsupported `S_RFE_RESTORE_B64` selector-255 encodings decode as
literal forms and consume a following dword, which can desynchronize
instruction-stream decoding and produce operand metadata that does not match the
XML encoding contract.

### CDNA4-RJ-087: `S_ABSDIFF_I32` uses 64-bit absolute difference instead of 32-bit wraparound

Manual evidence:

- The detailed CDNA4 `S_ABSDIFF_I32` definition first stores the 32-bit signed
  subtraction result and then negates that 32-bit result if it is negative at
  `cdna4/README.md:4280` through `:4289`.
- The manual examples pin the overflow edge cases, including
  `S_ABSDIFF_I32(0x80000000, 0x00000000) => 0x80000000` and
  `S_ABSDIFF_I32(0x80000000, 0x00000001) => 0x7fffffff`, at
  `cdna4/README.md:4292` through `:4302`.

Rocjitsu evidence:

- The shared `execute_s_absdiff_i32_sop2()` helper widens both signed inputs to
  `int64_t` and returns the mathematical absolute difference before truncating
  to `uint32_t` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:358`
  through `:368`.
- For the manual's `0x80000000, 0x00000001` example, that formula computes
  `2147483649` and truncates to `0x80000001`, while the documented 32-bit
  subtract-then-negate behavior produces `0x7fffffff`.

Impact:

rocjitsu can return the wrong scalar result when the signed 32-bit subtraction
overflows before the absolute-value step. The audited overflow examples remain
nonzero, so this is primarily a destination-value mismatch rather than an SCC
predicate mismatch for those cases.

### CDNA4-RJ-088: `S_SET_GPR_IDX_ON` treats operand 1 as a literal-capable scalar source

Manual/oracle evidence:

- The detailed `S_SET_GPR_IDX_ON` definition says vector operations use M0 for
  relative GPR addressing, source 0 supplies the index, and the raw bits of the
  `SRC1` field set the enable bits; the pseudocode writes `M0[15:12]` from
  `SRC1[3:0]` and says this is direct raw-field content at
  `cdna4/README.md:5682` through `:5698`.
- `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx950` accepts a 4-bit mode operand
  and a source-0 literal, producing `s_set_gpr_idx_on s0, 15` as a one-dword
  encoding and `s_set_gpr_idx_on 0x12345678, 15` as an 8-byte source-0 literal
  encoding, but rejects `s_set_gpr_idx_on s0, 16` and
  `s_set_gpr_idx_on s0, 0x12345678`.

Rocjitsu evidence:

- The generic CDNA4 `Sopc` base treats any `ssrc1 == 255` as a literal form and
  increases instruction size at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/encodings.cpp:38` through
  `:54`.
- `SSetGprIdxOnSopc` initially declares operand 1 as `OPR_SIMM4`, but still
  replaces it with an `OPR_SIMM32` operand when the raw field is 255 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopc.cpp:411` through `:427`.
- The shared executor then reads operand 1 through `RegisterAccess` and masks the
  resulting value to four bits before writing M0 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2514`
  through `:2519`.

Impact:

rocjitsu accepts and sizes an operand-1 literal form that the ISA text and LLVM
assembler treat as invalid. If a raw `SSRC1=255` word is encountered, rocjitsu
consumes the next dword as a literal and derives the mode from that extension
word instead of treating operand 1 as a raw 4-bit mode field.

### CDNA4-RJ-089: GPR-indexing instructions drop implicit M0 def-use metadata

Manual/XML evidence:

- `S_SET_GPR_IDX_ON` updates GPR-indexing mode and M0 state, preserving the
  remaining M0 bits, at `cdna4/README.md:5682` through `:5698`.
- XML records implicit M0 input and output operands for `S_SET_GPR_IDX_IDX`,
  `S_SET_GPR_IDX_ON`, and `S_SET_GPR_IDX_MODE` at
  `amdgpu_isa_cdna4.xml:34897` through `:34922`,
  `:44449` through `:44457`, and `:46610` through `:46620`.

Rocjitsu evidence:

- Generated CDNA4 constructors for the GPR-indexing state instructions expose
  only printed operands and set `num_dst_ = 0`: `S_SET_GPR_IDX_IDX` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sop1.cpp:947` through `:957`,
  `S_SET_GPR_IDX_ON` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopc.cpp:411` through `:428`,
  and `S_SET_GPR_IDX_MODE` / `S_SET_GPR_IDX_OFF` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopp.cpp:426` through `:447`.
- No CDNA4 `SSetGprIdx*` class overrides `implicit_uses()` or
  `implicit_defs()`, so def-use analysis only sees explicit operands through
  `InstDefUse` at `lib/rocjitsu/src/rocjitsu/analysis/def_use_chain.cpp:25`
  through `:45`.
- `RegisterSet` defines `RegClass::M0`, but documents it as untracked and
  ignores non-SGPR/VGPR/ACC classes in `expand()` at
  `lib/rocjitsu/src/rocjitsu/isa/register_set.h:50` through `:61` and
  `lib/rocjitsu/src/rocjitsu/isa/register_set.cpp:37` through `:51`.

Impact:

Analyses, schedulers, and DBT passes that depend on rocjitsu def-use metadata
cannot see that GPR-indexing setup reads and writes M0. That can let M0-dependent
indexing state move across or be reasoned about independently of the vector
instructions whose operands it controls.

### CDNA4-RJ-120: `V_BCNT_U32_B32` ignores `SRC1`

Manual/XML evidence:

- `V_BCNT_U32_B32` initializes the result with `S1.u32` and adds the set-bit
  count from `S0` at `cdna4/README.md:17569` through `:17580`.
- XML records `SRC1` as an input operand for opcode 651 at
  `amdgpu_isa_cdna4.xml:69840` through `:69844`.

Rocjitsu evidence:

- The generated CDNA4 class retains `src1`, but its executor delegates directly
  to the shared helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:10908` through
  `:10923`.
- `execute_v_bcnt_u32_b32_vop3` writes only
  `std::popcount(read_lane(inst.src0, lane))` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:3507`
  through `:3519`.
- Codegen classifies `V_BCNT_U32` as a unary operation at
  `lib/python/amdisa/semantics.py:745` and maps it to `std::popcount(s)` in
  `lib/python/amdisa/codegen/execute/vector_alu.py:252` through `:269`; the
  SIMD generator has the same scalar body at
  `lib/python/amdisa/codegen/execute/simd_codegen.py:1908` through `:1915`.
- The adjacent SIMD correctness table treats `v_bcnt_u32_b32` as unary popcount
  at `tests/simd_correctness/vop3_unary_simd_correctness_test.cpp:113` through
  `:118`.

Impact:

Any `V_BCNT_U32_B32` case with nonzero `SRC1` produces a result that is too
small by the base addend.

### CDNA4-RJ-121: `V_READLANE_B32` and `V_WRITELANE_B32` use unmasked lane selectors

Manual/XML evidence:

- `V_READLANE_B32` and `V_WRITELANE_B32` select lanes with `S1.u32[5:0]` at
  `cdna4/README.md:17540` through `:17567`.
- XML records the second source as `OPR_SSRC_LANESEL` for opcodes 649 and 650
  at `amdgpu_isa_cdna4.xml:69750` through `:69800`.

Rocjitsu evidence:

- Generated CDNA4 executors read the scalar lane selector and pass it unmasked
  to `read_lane`/`write_lane` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:10884` through
  `:10905`.
- The operand and register-access paths forward the raw lane value into VGPR
  access; the VGPR regions assert that the lane is below wave size at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/operand.cpp:1662` through
  `:1685` and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/vm/amdgpu/register_access.h:480`
  through `:517`.
- Nearby shared-infra tests use only small selectors such as lanes 2 and 31 at
  `tests/shared_infra_test.cpp:3490` through `:3504`.

Impact:

High-bit lane selectors can assert, access the wrong lane path, or fail instead
of aliasing to the low six bits as the manual specifies.

## No-Gap Notes

- CDNA4 Chapter 1-2 dispatch, 64-lane wavefront, initial `EXEC`, and packed
  work-item-ID basics are represented by the production dispatch path and
  existing Chapter 3 no-gap notes below. The new Chapter 2 issue above is
  limited to the device-memory consistency and acknowledgment model; LDS
  allocation/bank behavior, launch TTMP/TG_SIZE state, barrier behavior, and
  workgroup-size validation remain covered by their existing narrower entries.
- The CDNA4 VOP3P opcode inventory is generated for all 104 Chapter 12.10
  detailed opcode headings, with decode-table entries for the checked DOT,
  packed, ACCVGPR, and MFMA families. The only additional XML VOP3P-family
  entries are the two scaled `ENC_VOP3PX2` forms already covered by
  `CDNA4-RJ-014` through `CDNA4-RJ-018`.
- Generated CDNA4 VOP3A/VOP3B class inventory matches the XML decode split for
  this slice: 500 `ENC_VOP3` classes and 10 `VOP3_SDST_ENC` classes. The gaps
  above are semantic/runtime and metadata gaps, not missing decoder entries.
- CDNA4 Chapter 12.11 definitions 640-659 have generated constructors and
  encoding tests for the manual/XML opcodes 640-653 and 655-659; opcode 654 is
  absent from both sources. For this slice, `V_MBCNT_*` adds `SRC1`, 64-bit
  shift helpers mask counts with `&63`, and `V_BFM_B32` masks width/offset with
  `&31`. The new gaps are limited to `V_BCNT_U32_B32`'s missing base addend and
  readlane/writelane lane-selector masking; the `V_TRIG_PREOP_F64` hard stub is
  already covered by `CDNA4-RJ-110`.
- CDNA4 Chapter 12.11 definitions 660-673 have generated constructors and
  encoding tests for the manual/XML opcodes 660-666 and 668-673; opcode 667 is
  absent from both sources. The F32 normalized converts, packed integer
  narrowing converts, `V_PACK_B32_F16`, and `V_MUL_LEGACY_F32` have generated
  execution bodies matching the audited source-selection and pack/legacy-zero
  dataflow. Existing gaps cover the two F16 normalized-convert hard stubs
  (`CDNA4-RJ-110`), integer clamp/saturation overloads (`CDNA4-RJ-080`), and
  packed F32-to-F16 RTZ rounding (`CDNA4-RJ-113`).
- CDNA4 Chapter 12.11 definitions 680-681 have generated constructors and
  encoding tests for `V_MINIMUM3_F32` and `V_MAXIMUM3_F32`, and the generated
  bodies call the expected IEEE min/max helper composition. The remaining
  runtime issue for this pair is the MODE.DX10_CLAMP NaN-to-zero behavior
  recorded as `CDNA4-RJ-112`.
- Generated CDNA4 VOP3B class coverage exists for all ten Chapter 12.11/13.3.5
  VOP3B opcodes, and the shared carry/div-scale/MAD helpers write their
  explicit `inst.sdst` scalar destinations. `CDNA4-RJ-111` is limited to the
  public register-metadata view for special scalar destinations such as VCC.
- Ordinary `V_FMAC_F64` and `V_FMAC_F32` accumulator metadata is present: the
  generated constructors put `VDST` in both `src_operands_` and
  `dst_operands_` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:2231` through
  `:2241` and `:3575` through `:3585`. `CDNA4-RJ-109` remains limited to the
  hard-stubbed `V_PK_FMAC_F16` path.
- XML carries the `V_CVT_PKRTZ_F16_F32` round-toward-zero description and VOP3
  operand shape. `CDNA4-RJ-113` is a generated rocjitsu execution issue, not a
  missing XML semantic description for this opcode.
- Native F16 VOP3A destination-half preservation is implemented for
  `V_MAD_F16`, `V_MAD_U16`, `V_MAD_I16`, `V_FMA_F16`, and
  `V_DIV_FIXUP_F16`: the generated classes add old `VDST` to implicit uses and
  call `write_vop3_true16_dst(..., true)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:5378` through
  `:5670`. The corresponding XML gap is `CDNA4-XML-083`.
- `V_BITOP3_B16` and `V_BITOP3_B32` are implemented as truth-table operations
  over the overloaded `{OMOD, ABS, NEG}` bits rather than ordinary modifiers,
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:5692` through
  `:5761`; the generator documents the same packing rule at
  `lib/python/amdisa/codegen/execute/vector_special.py:700` through `:718`.
  The corresponding XML gap is `CDNA4-XML-084`.
- A fresh CDNA4 VOP3 hard-stub scan found only the VOP3A stubs already covered
  by `CDNA4-RJ-095`, `CDNA4-RJ-098`, `CDNA4-RJ-109`, and `CDNA4-RJ-110`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:821`, `:1922`,
  `:3607`, `:4230`, `:4250`, `:4270`, `:11025`, `:11142`, and `:11160`.
- The shared CDNA VOP3B machine-instruction struct has no `ABS` field, and the
  generator's `has_abs_modifier()` rule excludes `VOP3_SDST_ENC`, matching the
  manual's VOP3B field map rather than leaking generic VOP3A `ABS` handling
  into VOP3B.
- Generated packed shift helpers mask the selected shift-count half with
  `& 15`, matching the manual's use of `S0[3:0]` and `S0[19:16]` for the low
  and high components. The generated constructors and shared helpers for
  `V_PK_LSHLREV_B16`, `V_PK_LSHRREV_B16`, and `V_PK_ASHRREV_I16` dispatch
  through `execute_shared.h:16805`, `:16950`, and `:16979`.
- `V_ACCVGPR_READ` and `V_ACCVGPR_WRITE` are not missing execution bodies:
  generated CDNA4 code copies one 32-bit lane between the VGPR and ACCVGPR
  register classes under `EXEC` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:792` through
  `:832`. `CDNA4-RJ-108` is limited to the public instruction flag metadata.
- CDNA4 generated VALU instruction coverage is not broadly absent for the
  audited Chapter 6.3 inventory. Representative VOP2/VOP3/VOPC/VOP3P DOT,
  compare, carry-out, CNDMASK, `_MK`/`_AK`, and GPR-indexing instruction
  classes exist; the Chapter 6 VALU gaps above are about legality, stateful
  semantics, and hazard modeling.
- CDNA4 VALU execution honors the EXEC mask for ordinary per-lane VGPR writes in
  shared helpers such as `execute_v_add_f32_vop3()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:3020`
  through `:3024`, matching the basic Chapter 6.2.2 write-mask contract.
- Generated CDNA4 VOPC class/decode inventory exists for all 198 XML VOPC
  opcode rows; the VOPC gaps above are about special-state metadata and
  extension semantics, not missing base opcode classes.
- CDNA4 relational VOP3 CMPX helpers write the explicit scalar destination and
  then update EXEC, matching the VOP3 destination contract. `CDNA4-RJ-102`
  records the separate class-CMPX exception, while `CDNA4-RJ-101` covers E32
  VOPC implicit VCC/EXEC metadata.
- Chapter 6.2.3's ordinary out-of-range VGPR behavior overlaps the broader GPR
  allocation/out-of-range execution gap already recorded in the Chapter 3
  rocjitsu pass; this Chapter 6 slice adds the distinct indexed-VGPR
  out-of-range legality issue under `CDNA4-RJ-081`.
- Generated CDNA4 SOP1 class and decoder inventory matches XML `ENC_SOP1`
  records exactly: all 54 manual/XML opcode rows have constructors and
  non-invalid `sub_decode_sop1` entries, while opcode holes 47 and 49 decode as
  invalid. `CDNA4-RJ-085` records the separate no-literal legality and size
  issue for selected SOP1 opcodes.
- Generated CDNA4 SOP2 class and decoder inventory matches XML `ENC_SOP2`
  records exactly: all 53 XML opcode rows, including XML-only opcode 43, have
  constructors and non-invalid primary decode-table entries. `CDNA4-RJ-086`
  records the separate no-literal legality and size issue for
  `S_RFE_RESTORE_B64`, and `CDNA4-XML-060` records the manual-source drift for
  opcode 43.
- Generated CDNA4 SOPK class and decoder inventory matches XML `ENC_SOPK`
  records exactly: all 21 XML opcode rows have constructors and non-invalid
  primary decode-table entries, opcode 19 decodes invalid, and only opcode 20
  uses an instruction extension word as an operand. Existing gaps cover the
  semantic issues found in this inventory: `CDNA4-RJ-061` for
  `S_CBRANCH_I_FORK`, `CDNA4-RJ-052`/`CDNA4-RJ-075` for HWREG/SETREG behavior,
  `CDNA4-RJ-074` for `S_SETREG_IMM32_B32` literal visibility, and
  `CDNA4-RJ-071` for implicit SCC metadata.
- Generated CDNA4 SOPC class and decoder inventory matches XML `ENC_SOPC`
  records exactly: all 20 XML opcode rows have constructors and non-invalid
  `sub_decode_sopc` entries, while the remaining slots 20 through 127 decode
  invalid. `CDNA4-RJ-054` records the existing `S_SETVSKIP` execution gap,
  `CDNA4-RJ-071` records the SOPC compare/bitcmp SCC metadata gap, and
  `CDNA4-RJ-088` / `CDNA4-RJ-089` record the distinct GPR-indexing operand and
  M0 metadata issues found in this full SOPC pass.
- CDNA4 normal SALU literal replacement exists for SOP1/SOP2/SOPC instructions
  that use selector 255; representative constructors replace the source operand
  with `OPR_SIMM32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sop2.cpp:20` through `:84`.
  The SETREG_IMM and `S_SET_GPR_IDX_ON` gaps above are limited to their
  instruction-specific operand contracts.
- CDNA4 shared SALU helpers implement the audited signed/unsigned add/sub SCC
  behavior for ordinary arithmetic: signed add/sub use overflow, unsigned
  add/sub and addc/subb use carry/borrow, and ADDK uses signed overflow. The
  scalar SCC predicate gap from this slice is the max-equality predicate;
  `CDNA4-RJ-087` records the separate `S_ABSDIFF_I32` destination-value issue.
- CDNA4 special scalar selectors for VCCZ, EXECZ, and SCC read live wavefront
  state in `resolve_src_scalar()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/operand.cpp:1395` through
  `:1404`. The remaining STATUS/raw-state drift is covered by the earlier
  `CDNA4-RJ-053` finding.
- Chapter 5 reconfirmed the wrong CDNA4 HWREG map already recorded as
  `CDNA4-RJ-052`; this slice adds the access-instruction fallout for
  `S_SETREG_IMM32_B32` operand metadata and SETREG write-mask/spacing rules
  rather than duplicating that map finding.
- CDNA4 ordinary PC-relative branch behavior is implemented for `S_BRANCH` and
  SCC/VCC/EXEC conditional branches. The constructors set branch flags and the
  execute bodies apply the `pc + 4 + simm16*4 - size_` formula at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopp.cpp:40` through
  `:201`.
- Generated CDNA4 SOPP class and decoder inventory matches XML `ENC_SOPP`
  records exactly: 32 constructors and 32 non-invalid `sub_decode_sopp` entries
  cover opcodes 0 through 31. The SOPP gaps above are semantic/runtime gaps,
  plus the separate manual/XML source drift recorded as `CDNA4-XML-059`.
- CDNA4 direct PC operations have concrete behavior: `S_GETPC_B64` writes
  `PC+size`, `S_SETPC_B64` sets PC from the scalar source, and `S_SWAPPC_B64`
  writes the next PC while branching through the scalar source at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sop1.cpp:540` through
  `:590`.
- CDNA4 `S_CALL_B64` writes `PC+size` to the destination SGPR pair and uses the
  documented PC-relative branch formula at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopk.cpp:406` through
  `:426`. Although the constructor uses the generic `INDIRECT_CALL` call marker,
  it also exposes `branch_offset_bytes()`, and CFG recovery distinguishes this
  branch-offset-bearing case as a direct call edge.
- CDNA4 `S_WAITCNT` immediate decoding matches the XML bit layout for the
  audited fields: low VM bits, EXP bits, LGKM bits, and high VM bits are
  extracted at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/sopp.cpp:226` through
  `:241`.
- The core CDNA4 `S_BARRIER` release path is not absent: CU state release waits
  until all non-halted wavefronts in the same dispatch/workgroup are at the
  barrier and then releases the blocked waves together at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:313` through `:335`.
  The remaining barrier gaps are architectural status exposure, ISA-limit
  validation, and edge-case tests.
- The CDNA4 generated SMEM SBASE operand scaling regression is covered by
  `tests/smem_sbase_operand_test.cpp:64` through `:139`, including CDNA4
  64-bit loads, 128-bit scalar-buffer descriptors, and store-family source
  ordering. The SMEM gaps above are about later address semantics, dependency
  behavior, and cache/atomic side effects rather than that operand-width fix.
- Generated CDNA4 SMEM class and decoder inventory matches the XML
  `ENC_SMEM` records exactly: 84 generated constructors and 84 non-invalid
  `sub_decode_smem` entries cover the 82 manual opcode-table entries plus the
  two XML-only `S_ATC_PROBE*` opcodes. The decoder table places the manual
  load/store/cache/time/discard/atomic opcodes in their documented slots and
  leaves the holes invalid at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/decoder.cpp:7534` through
  `:7710`.
- Generated CDNA4 `S_ATC_PROBE` and `S_ATC_PROBE_BUFFER` decode from the XML and
  no-op in execution at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/smem.cpp:713` through
  `:741`. Because the CDNA4 manual does not list those opcodes, this audit
  records the mismatch as `CDNA4-XML-058` rather than a manual-derived runtime
  gap.
- Production dispatch sets the initial `EXEC` mask from
  `initial_exec_mask_for_wave()` before register initialization at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:693` through
  `:720`, and that helper accounts for partial workgroups and grid bounds at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/dispatch_entry.h:206` through `:239`.
  The Chapter 3 EXEC gap above is therefore about HWREG-visible raw `STATUS`,
  not missing active-lane initialization.
- CDNA4 packed `VGPR0` workitem IDs are initialized by the dispatch path using
  the Chapter 3 `{Z,Y,X}` packing at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:300` through
  `:321` and `lib/rocjitsu/src/rocjitsu/vm/amdgpu/dispatch_entry.h:182`
  through `:193`. The remaining launch-initialization gaps are TTMP payloads
  and the optional `TG_SIZE` system SGPR.
- CDNA4 generated scaled code does extract the scale exponent with
  `(scale_bits >> 23) & 0xFFu`, matching the manual's E8M0/exponent-only
  direction at a high level. The remaining gap is the missing mode/configuration
  behavior around that conversion.
- Generated scaled FP8/BF8 conversion definitions 565-572 implement the
  high-level `OPSEL` and scale plumbing checked in this slice: packed and
  stochastic narrow forms preserve the old destination, packed widening forms
  use `OPSEL[0]`, single widening forms use `OPSEL[1:0]`, and the scale source
  is reduced to the F32 exponent field at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:5764` through
  `:6104`. The remaining issues are the recorded state, source-legality, and
  alignment gaps.
- Generated scaled FP4 code preserves the old destination for byte writes via
  implicit destination uses and read-modify-write execution, for example
  `V_CVT_SCALEF32_PK_FP4_F32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:6121` through
  `:6152` and `V_CVT_SCALEF32_SR_PK_FP4_F32` at `:6171` through `:6206`.
- Generated scaled FP6/BF6 wide paths exist for the CDNA4 `PK32` and `2XPK16`
  forms and use 32-element 6-bit pack/unpack helpers, for example
  `V_CVT_SCALEF32_2XPK16_FP6_F32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:7104` through
  `:7127` and `V_CVT_SCALEF32_PK32_F32_FP6` at `:7988` through `:8017`.
- Generated stochastic FP4/FP6/BF6 scaled conversions advance the PRNG between
  elements with `util::prng_advance`, matching the manual's "internal
  V_PRNG_B32 but not written" direction at a high level; representative
  generated calls appear at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:6200` and
  `:7538` through `:7544`.
- `util::fp4_e2m1_to_f32`, `util::fp6_e2m3_to_f32`, and
  `util::bf6_e3m2_to_f32` plus their RNE/SR narrow helpers encode the CDNA4
  FP4/FP6/BF6 range shape, no-Inf/no-NaN destinations, saturation, and tiny
  underflow-to-zero behavior at
  `lib/util/include/util/data_types.h:1047` through `:1325`.
- CDNA4 generated MUBUF and MTBUF constructors do compensate for XML's fixed
  64-bit `VADDR` metadata by deriving the public address operand width from
  `IDXEN`/`OFFEN`, and scale `SRSRC` to the aligned SGPR descriptor base at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/mubuf.cpp:28` through
  `:50` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/mtbuf.cpp:27`
  through `:49`.
- CDNA4 generated MUBUF/MTBUF class inventory covers the audited manual opcode
  tables: 16 MTBUF declarations at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/mtbuf.h:17` through `:175`
  and 74 MUBUF declarations at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/mubuf.h:17` through `:747`,
  with generated constructor/execution bodies running through
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/mtbuf.cpp:37` through `:695`
  and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/mubuf.cpp:38` through
  `:3066`.
- CDNA4 raw MUBUF byte/short/dword loads and stores are generated with useful
  basic element sizes, sign extension, D16 low/high writeback flags, and global
  memory-pipeline routing. The vector-buffer gaps above are about descriptor,
  formatted conversion, address edge cases, LDS, atomic dataflow metadata,
  floating-atomic numeric behavior, and precise cache policy rather than
  absence of all raw-buffer execution.
- The vector memory pipeline does preserve/zero D16 results according to the
  emulator's SRAM ECC flag, matching the high-level direction of the CDNA4 D16
  ECC note for implemented raw D16 paths at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:157` through
  `:195`.
- CDNA4 special decode does consume the four-word VOP3PX2 scale form and stores
  the two prefix words in `raw_words_`; the generated constructors set
  `size_ = 16` and `raw_encoding_` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:863` through
  `:866` and `:927` through `:930`.
- `ABID[0]=0` dispatches the unscaled mixed-format MFMA helper, matching the
  manual's "runs without scale source" behavior for that bit at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:879` through
  `:884` and `:943` through `:948`.
- Dense MFMA generated classes do use `ACC[0]`/`ACC[1]` to select Arch VGPR
  versus AccVGPR sources and `ACC_CD` for C/D register-bank selection. The
  constructor rewrites representative dense operands at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:1338` through
  `:1361`, and execution resolves physical bases with `dst_base()`,
  `src_base()`, and `resolve_acc()` at `:1365` through `:1374`.
- Generated MFMA and SMFMAC classes are tagged with the broad `MFMA`
  instruction flag, and rocjitsu has generated `WAITCNT` flags plus memory wait
  counters for ordinary memory dependencies. The Section 7.6 MAI gap above is
  about missing producer/consumer hazard predicates and wait-count rules, not a
  total absence of MFMA or wait-counter classification.
- CDNA4 dense I8 MFMA paths execute integer multiply-add helpers and therefore
  naturally ignore floating-point MODE state. The Section 7.4 MAI gap above is
  about missing floating-point denorm/RNE/exception modeling, not a claim that
  the integer I8 helper should consult MODE.
- Shared dense MFMA layout helpers implement the manual's lane/item formulas:
  `input_loc()` computes input item/register/lane placement, `output_loc_32()`
  implements the 4-row F32/I32 output grouping, and `output_loc_64()`
  implements the paired-register F64 output layout at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:172` through
  `:229`.
- Dense MFMA helper paths apply `CBSZ`/`ABID` A-lane broadcast and `BLGP`
  B-lane permutation in the scalar path and route non-zero broadcast fields out
  of the SIMD fast paths, for example `permute_a_lane()` / `permute_b_lane()`
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:565` through
  `:615` and `exec_f32_mixed()` at `:1154` through `:1259`.
- `permute_b_lane()` implements all documented ordinary `BLGP` values 0
  through 7, including the rotate-16 case, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:580` through
  `:613`.
- F64 MFMA generated code passes `inst_.blgp` as the `exec_f64()` negation mask
  and does not pass `CBSZ`/`ABID`, matching the manual's alternate F64 field
  meaning. Representative generated calls are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:2937` through
  `:2938` and `:2980` through `:2981`, and `exec_f64()` applies bits 0/1/2 to
  A/B/C at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:3493`
  through `:3530`.
- Generated SMFMAC constructors model `SRC2` as an Arch VGPR operand and do
  not let `ACC_CD` redirect it to AccVGPR, while destination/C selection still
  uses `ACC_CD`. Representative constructors and execution paths are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:1144` through
  `:1175`, `:2406` through `:2443`, and `:3538` through `:3669`.
- Sparse MFMA helper paths ignore `EXEC` by construction and write the full
  matrix footprint from `ComputeUnit` state, matching the manual's
  execution-mask ignore rule at a high level. Representative helpers are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:3693` through
  `:3725`, `:3768` through `:3801`, and `:3932` through `:3968`.
- Existing tests cover the F64 `BLGP`-as-negation path, including generated
  CDNA4 instruction execution, at `emulation/rocjitsu/tests/amdgpu_vm_test.cpp:1793`
  through `:1869`. SIMD exact tests also cover representative non-default
  `CBSZ` and `BLGP` helper cases at
  `emulation/rocjitsu/tests/simd_correctness/mfma_simd_exact_test.cpp:109`
  through `:115`.
- Dense MFMA reads and writes use full-wave masks, matching the manual's
  execution-mask ignore rule at a register-access level:
  `mfma_full_lane_mask()` and the MFMA region helpers are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:1013` through
  `:1063`.
- The generated F8F6F4 MFMA execution dispatches by `CBSZ`/`BLGP` matrix format
  through `dispatch_matrix_fmt_pair`, matching the manual's use of those fields
  as A/B format selectors for this instruction family. The dispatch table covers
  all 25 documented A/B combinations for `FP8`, `BF8`, `FP6`, `BF6`, and `FP4`
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:816` through
  `:916`.
- F8F6F4 MFMA execution ignores `EXEC` and writes the full wave, matching the
  manual's "forces it to 1 for all threads" rule: MFMA region helpers use
  `mfma_full_lane_mask` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:1013` through
  `:1014`, `:1057` through `:1063`, and the F8/F6/F4 paths use those full-lane
  region writes at `:4309` through `:4313` and `:4354` through `:4360`.
- The generated F8F6F4 MFMA operand classes match the manual's source legality
  at decode/disassembly level: `SRC0` and `SRC1` are
  `OPR_SRC_VGPR_OR_ACCVGPR`, while `SRC2` is
  `OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST`, for example
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3p.cpp:842` through
  `:855` and `:906` through `:919`.
- Generated FP8/BF8 packed and stochastic narrow converts plus scaled FP4 byte
  converts read the old destination before partial writes through
  `implicit_uses` and read-modify-write code, even where XML marks `VDST` as
  output-only. `CDNA4-RJ-082` records the separate F16/BF16 stochastic
  conversion exception found in the Chapter 12 definition sweep, and
  `CDNA4-RJ-114` records the same public-metadata problem for
  `V_CVT_PKACCUM_U8_F32`.
- Generated VOP3 FP8/BF8 widening aliases for definitions 84-87 select source
  bytes through `OPSEL[1:0]` or packed source words through `OPSEL[0]`, matching
  the definition-level VOP3 behavior at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:1964` through
  `:2078`. `CDNA4-RJ-012` covers the separate 64-bit destination-alignment
  issue, and `CDNA4-RJ-083` covers VOP1 SDWA ignored fields.
- Generated `V_ASHR_PK_{I8,U8}_I32` masks the shift count to five bits, performs
  signed arithmetic shifts, clamps to the documented signed/unsigned 8-bit
  intervals, and packs the two bytes before the true16 destination write at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:10509` through
  `:10596`.
- Generated `V_CVT_PK_F16_F32` and `V_CVT_PK_BF16_F32` write full 32-bit
  packed destinations and use the RNE F16/BF16 conversion helpers at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:10598` through
  `:10648`; the Python semantic tests also pin BF16 RNE lowering at
  `lib/python/amdisa/tests/test_sema_derive.py:1073` through `:1083`.
- Generated `V_CVT_SCALEF32_PK_BF16_{FP8,BF8}` extracts `OPSEL[0]` source-word
  selection and the exponent-only scale field at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:10650` through
  `:10728`. The remaining scaled-conversion gaps are the previously recorded
  MODE/source-legality/alignment issues, not absence of these two generated
  BF16 widening forms.
- Generated CDNA4 VOP3P machine state keeps bit 14 as a separate
  `op_sel_hi_2` field at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h:37`
  through `:49`, and the generator has an explicit CDNA4 field-rename override
  at `lib/python/amdisa/isa_profile.py:967` through `:970`. That preserves
  source-2 high-half selection despite the CDNA4 XML folding bit 14 into
  `OP_SEL_HI`.
- `V_PK_MOV_B32` uses `read_lane64()` for both sources and selects output dwords
  with `OPSEL[0]` and `OPSEL[1]` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:17295`
  through `:17312`, matching the CDNA4 manual's special `V_PK_MOV_B32`
  selector behavior for scalar pairs and VGPR gather.
- MIX helpers implement the MIX-specific selector mapping, treat `NEG_HI` as
  absolute value, and apply `CLMP` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:13290`
  through `:13411`. They use multiply-add rather than fused FMA; the detailed
  instruction pseudocode and XML descriptions support multiply-add, while
  section 6.7 contains conflicting fused wording.
- Packed 32-bit helpers do not apply clamp or other output modifiers, matching
  the packed 32-bit statement in `cdna4/README.md:1571` that output modifiers
  are not supported for those instructions.
- CDNA4 generated VOP2/VOP3 inventory covers all Chapter 12.7 opcodes 0 through
  61. The expected VOP3 holes for the four literal-only `_MK`/`_AK` opcodes are
  invalid in the VOP3 decode table, while the VOP2 decode table dispatches the
  literal forms through their generated classes.
- `V_CNDMASK_B32` VOP3 correctly uses source 2 as the scalar condition source
  and applies B32 source modifiers only to sources 0 and 1. `CDNA4-RJ-076`
  records the separate scalar-source-combination validation issue.
- Generated VOP3 carry-out forms use arbitrary SGPR-pair `SDST` destinations
  and read carry-in from `SRC2` for ADDC/SUBB/SUBBREV. `CDNA4-RJ-080` records
  the separate integer saturation/clamp gap.
- Generated `V_MAC_F16` VOP3 handles the manual's non-standard `OPSEL[3]`
  destination-half rule by selecting the old destination half and passing the
  OPSEL-derived write location into the true16 destination helper.
- Generated `V_LDEXP_F16` sign-extends the 16-bit exponent source, matching the
  instruction-local source-size rule.
- Generated CDNA4 VOP1 and VOP3 class inventory covers all 85 detailed Chapter
  12.8 VOP1 opcode definitions. `CDNA4-RJ-098` records the separate runtime
  ambiguity for the three XML-only VOP1 rows.
- Generated VOP3 FP8/BF8 widening aliases for definitions 84-87 select source
  bytes through `OPSEL[1:0]` or packed source words through `OPSEL[0]`, matching
  the definition-level VOP3 behavior. `CDNA4-RJ-083` records only the VOP1 SDWA
  ignored-field issue for these widening converts.
- `V_PERMLANE32_SWAP_B32` implements the manual's lane-0-through-31 row-pair
  swap shape in both generated VOP1 and VOP3 execute bodies at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop1.cpp:11085` through
  `:11095` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:2129`
  through `:2138`; `CDNA4-RJ-097` records the separate public dataflow issue and
  the `V_PERMLANE16_SWAP_B32` second-pass execution gap.
- Generated CDNA4 FLAT classes intentionally cover the 54 FLAT opcode-table
  entries and use `SEG`-based mnemonic rewriting to present overlapping
  scratch/global forms; the 54 declarations run from
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/flat.h:17` through `:533`,
  with constructor/execution bodies from
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/flat.cpp:28` through
  `:2917`. `CDNA4-RJ-040` records the separate missing direct-to-LDS-only
  `GLOBAL_LOAD_LDS_*`/`SCRATCH_LOAD_LDS_*` opcodes. These constructors adjust
  public address operand width based on `SEG` and `SADDR`, and preserve
  ACC-bank addressing for data/destination operands.
- The shared flat address helper implements the common 64-bit FLAT,
  SGPR-base-plus-VGPR-offset GLOBAL, VGPR-pair GLOBAL, and basic
  `FLAT_SCRATCH + lane_stride + offset` SCRATCH formulas at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h:59`
  through `:132`.
- Generated CDNA4 FLAT load/store paths implement raw byte/short/dword element
  sizes, sign extension, D16 low/high writeback flags, ACC addressing, and
  vector-memory routing for ordinary VGPR-return memory operations.
- CDNA4 `DS_PERMUTE_B32` and `DS_BPERMUTE_B32` execute through shared helpers
  that use 64-lane grouping on CDNA4, apply `EXEC` to source/writeback as
  expected, and use ascending lane order so the highest-numbered active
  `DS_PERMUTE_B32` source wins collisions at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:59`
  through `:90` and `:196` through `:227`. The swizzle gap above is limited to
  `DS_SWIZZLE_B32`'s missing FFT/rotate modes.
- Generated CDNA4 DS class and mnemonic inventory covers the audited opcode
  table: 126 declarations at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/ds.h:17` through `:1086`,
  constructor/execution bodies from
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/ds.cpp:28` through `:5298`,
  and `sub_decode_ds` dispatch entries at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/decoder.cpp:8957` through
  `:9216`. The data-share gaps above are semantic/runtime gaps rather than
  missing generated DS instruction records.
- CDNA4 transpose-load execution is not absent: generated DS bodies set
  `d->transpose`, and local-memory completion invokes `transpose_response()` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:599` through `:604`.
  The remaining transpose gap is validation and focused layout coverage.
