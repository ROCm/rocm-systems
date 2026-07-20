# CDNA2 Rocjitsu Gaps

Architecture: CDNA2

Manual source: `workspace_docs/amdgpu-isa-manuals/cdna2/README.md`

Rocjitsu source: `emulation/rocjitsu`

## Coverage

| Area | Status | Notes |
| --- | --- | --- |
| CDNA2 introduction and program organization | Audited statically, partial | Checked Chapters 1-2 plus the MI200 feature-change wave/workgroup/dispatch/SALU/VALU/EXEC/LDS/GWS/device-memory prose against ISA profiles, dispatch entry/command processor paths, wavefront/CU initialization, LDS backing, generated DS/GWS stubs, memory pipeline, and existing gap/no-gap notes. |
| CDNA2 VOP3P packed math | Audited statically | Checked generated packed F16/I16/U16 helpers, MIX helpers, packed F32 helpers, `V_PK_MOV_B32`, selector handling, clamp behavior, and pair alignment for this slice. |
| CDNA2 MFMA/VOP3P-MAI | Audited statically | Checked the CDNA2 Chapter 7 MAI intro plus sections 7.1-7.2 and the full Chapter 12.10 MFMA/ACCVGPR opcode table: generated MFMA and AccVGPR move inventory, operand classes and sizes, AccVGPR/ArchVGPR transfer and bank routing, C/D bank handling, denorm/MODE prose, inline constants, exception suppression, dependency waits, pass-count metadata, `CBSZ` legality, F64 field handling, 64-bit pair alignment, and targeted test coverage. |
| CDNA2 wave/kernel state | Audited statically, partial | Checked Chapter 3.1-3.6 PC/EXEC/STATUS/MODE state, helper bits, VSKIP, GPR/LDS allocation and aliasing, out-of-range behavior, and LDS allocation/clamping against `Wavefront`, generated SALU accessors, branch/source helpers, and CU/SPI allocation code. |
| CDNA2 special wave/kernel state | Audited statically, partial | Checked Chapter 3.7-3.12 M0, SCC, VCC/VCCZ, trap/exception/TRAPSTS/memory-violation state, HW_ID, TTMP privilege, and packed VGPR0 workitem ID against operands, SOPK/SOPP execution, and command-processor initialization. |
| CDNA2 program control and branching | Audited statically, partial | Checked Chapter 4.1-4.2 control/branch tables, ordinary PC-relative branches, direct PC operations, trap return, debug conditional branches, call, sleep/wakeup/message/status helpers, Chapter 12/13 SOPP opcode inventory, and adjacent decode/execution tests. |
| CDNA2 workgroups and `S_BARRIER` | Audited statically, partial | Checked Chapter 4.3 workgroup size limits and barrier semantics against generated `S_BARRIER` plumbing, `WfState::BARRIER` scheduler release, dispatch bookkeeping, `STATUS.IN_BARRIER`, plugin hooks, and adjacent tests. |
| CDNA2 wait-counter dependency resolution | Audited statically, partial | Checked Chapter 4.4 `S_WAITCNT` threshold decoding, runtime wait-counter state, memory-pipeline producer accounting, scalar-memory dword counts, GDS/GWS `EXP_CNT`, and adjacent tests. |
| CDNA2 manually inserted wait states | Audited statically, partial | Checked Chapter 4.5 required NOP/independent-instruction hazards against generated `S_NOP`, functional issue/execute behavior, wait-counter plumbing, and existing hazard-tracker/test coverage. |
| CDNA2 arbitrary divergent control flow | Audited statically, partial | Checked Chapter 4.6 fork/join branch-stack prose, generated `S_CBRANCH_{G,I}_FORK` and `S_CBRANCH_JOIN` constructors/execution, branch-stack state storage, generated encoding fixtures, and adjacent tests. |
| CDNA2 SALU scalar operands and SCC behavior | Audited statically, partial | Checked Chapter 5.1-5.7 SALU field formats, scalar selector table, literals, source/destination out-of-range rules, 64-bit SGPR-pair alignment, arithmetic/compare/bitwise SCC effects, generated operands, scalar helpers, and adjacent SCC tests. |
| CDNA2 SALU def-use metadata | Audited statically, partial | Checked representative generated SOP1/SOP2/SOPC/SOPP constructors, XML implicit SCC operands, generic def-use hooks, and tracked register classes. |
| CDNA2 HWREG access and SOPK literals | Audited statically, partial | Checked Chapter 5.8 access-instruction prose, HWREG ID/field tables, `S_SETREG` spacing, `S_SETREG_IMM32_B32` literal form, generated CDNA2 SOPK constructors/execution, and gfx1250 contrast code. |
| CDNA2 SOP2 instruction definitions | Audited statically, partial | Checked Chapter 12.1 full SOP2 opcode inventory, generated constructors, literal handling, shared scalar execution helpers, bitfield and pack rows, `S_ABSDIFF_I32` edge examples, shifted-add carry rows, existing fork/RFE gaps, and adjacent tests. |
| CDNA2 SOPK instruction definitions | Audited statically, partial | Checked Chapter 12.2 full SOPK opcode inventory, generated constructors, literal-only opcode 20 handling, signed and unsigned SIMM16 helpers, ADDK/MULK old-destination dataflow, SCC effects, direct-call metadata, existing fork/HWREG gaps, and adjacent tests. |
| CDNA2 SOP1 instruction definitions | Audited statically, partial | Checked Chapter 12.3 full SOP1 opcode inventory, generated constructors, literal handling, SCC-producing unary rows, bit-count/bit-scan and bitset helpers, direct PC/RFE rows, saveexec/wrexec formulas, quad-mask/bitreplicate helpers, relative SGPR addressing, existing fork/join and GPR-index gaps, and adjacent tests. |
| CDNA2 SOPC instruction definitions | Audited statically, partial | Checked Chapter 12.4 full SOPC opcode inventory, generated constructors and literal handling, shared compare/bit-compare/GPR-index helpers, existing VSKIP/GPR-index gaps, and adjacent SCC tests. |
| CDNA2 SOPP instruction definitions | Audited statically, partial | Checked Chapter 12.5 full SOPP opcode inventory, generated constructors and decoder table, ordinary branch/waitcnt/barrier/endpgm behavior, XML-only `S_TTRACEDATA`, trap metadata, message/cache/perf/ordered-PS side effects, existing status/debug/GPR-index gaps, and adjacent branch/barrier tests. |
| CDNA2 SMEM addressing and descriptors | Audited statically, partial | Checked Chapter 8.1-8.2 and Chapter 12.6 scalar, scratch, and buffer SMEM addressing/definition rows against generated SMEM constructors, `OPR_SMEM_OFFSET`, shared scalar address calculation, scalar memory state, and memory pipeline behavior. |
| CDNA2 SMEM cache/time/atomics and dependencies | Audited statically, partial | Checked Chapter 8.2-8.4 and Chapter 12.6 atomics, cache maintenance, ATC probe, dcache discard, time reads including `S_MEMREALTIME`, LGKM counter accounting, alignment, bounds, source-overwrite/clause prose, generated fixture coverage, and full SMEM opcode inventory against generated execute bodies, shared helpers, wait counters, and adjacent tests. |
| CDNA2 vector-buffer addressing and data movement | Audited statically, partial | Checked Chapter 9.1 plus Chapter 12.13/12.14 MTBUF/MUBUF fields and full opcode inventory, VGPR usage, address formation, resource descriptors, range checking, swizzle, D16, LDS-destination loads, generated buffer constructors, shared buffer address calculation, vector memory state, and memory pipeline writeback. |
| CDNA2 vector-buffer cache and atomics | Audited statically, partial | Checked Chapter 9.1 plus Chapter 12.13 MUBUF cache, LDS-store, integer-atomic, floating-atomic, and 64-bit-atomic rows against generated execute bodies, cache-flag mapping, and vector atomic RMW pipeline behavior. |
| CDNA2 vector-image decode and operands | Audited statically, partial | Checked Chapter 9.2-9.4.5 and Chapter 12.15 MIMG fields, image opcode inventory, no-sampler/sampler address tables, MIP form restrictions, DMASK/data rules, D16/A16, resource and sampler operands, generated MIMG constructors, disassembly modifiers, generated encoding fixtures, and decode smoke coverage. |
| CDNA2 vector-image execution and dependencies | Audited statically, partial | Checked image load/store/query/sample/atomic execute bodies, image resource/sampler descriptor handling, VMEM wait-counter routing, generator image stubs, and image coverage exceptions. |
| CDNA2 float memory atomics | Audited statically, partial | Checked Chapter 9.5 LDS/L2 float atomic numeric rules, generated MUBUF/FLAT/DS execute bodies, missing GLOBAL float-atomic decode surface, shared atomic RMW pipeline behavior, and adjacent tests. |
| CDNA2 flat/global/scratch memory | Audited statically, partial | Checked Chapter 10 and Chapter 12.16 FLAT/GLOBAL/SCRATCH field behavior, cache/coherency flags, full opcode inventories, segment naming and legality, flat/private/shared aperture routing, scratch and global address forms, direct LDS movement, atomics, ordering, wait-counter behavior, memory-error policy, generated flat constructors/execution, shared address calculation, CU memory routing, disassembly, and adjacent tests. |
| CDNA2 data share operations | Audited statically, partial | Checked Chapter 11 LDS overview/dataflow and Chapter 12.12 full LDS/GDS instruction definitions, direct LDS reads, indexed load/store and atomic address rules, M0 clamp prose, READ2/WRITE2 duplicate-offset behavior, ADDTID, append/consume, GWS restrictions, DS permute/bpermute/swizzle including rotate/FFT/basic modes, B96/B128 decode and transfer paths, generated DS constructors/execution, shared LDS memory pipeline paths, and adjacent tests. |
| CDNA2 VOP2/VOP1/VOPC instruction definitions | Audited statically, partial | Checked Chapter 12.7-12.9 VOP2, VOP1, and VOPC definition semantics against generated constructors, promoted VOP3 paths, shared VALU helpers, compare/class execution, special-state helpers, DPP/SDWA/literal extension handling, and adjacent tests. |
| CDNA2 VOP3A/VOP3B instruction definitions | Audited statically, partial | Checked Chapter 12.11 VOP3A/VOP3B opcode inventory, VOP3B scalar destinations, division helper edge cases, native and legacy F16 OPSEL destination behavior, `V_PERM_B32`, packed F32-input conversions, F64 min/max, readlane/writelane, `V_TRIG_PREOP_F64`, and adjacent test coverage. |
| CDNA2 DPP/SDWA instruction limitations | Audited statically, partial | Checked Chapter 12.17 DPP and SDWA exclusion lists against generated CDNA2 VOP1/VOP2/VOPC constructors, decoder tables, DPP/SDWA extension handling, and adjacent tests. |
| CDNA2 VALU format/operand semantics | Audited statically, partial | Checked Chapter 6.1-6.6 VOP encoding roles, VALU operands, EXEC-gated writes, MODE round/denorm and OMOD/CLAMP behavior, VGPR indexing, generated operands, shared VALU helpers, and adjacent tests. |
| CDNA2 SDWA/SDWAB/DPP format semantics | Audited statically, partial | Checked Chapter 13.3.7-13.3.9 selector/control value tables, VOP1/VOP2/VOPC generated constructors, shared DPP/SDWA helpers, scalar-destination paths, disassembly modifiers, liveness hooks, and adjacent tests. |
| Remaining CDNA2 rocjitsu surface | Not started | Full decoder/autogen/runtime fuzzing remains. |

## Gaps

### CDNA2-RJ-001: Packed F32 arithmetic is generated with 32-bit operand metadata and broadcasts scalar high words

Manual/XML evidence:

- CDNA2 section 6.7 says packed 32-bit instructions operate on two dwords at a
  time and that `OPSEL`/`OPSEL_HI` select the first or second dword for each
  source at `cdna2/README.md:1504` through `:1526`.
- `V_PK_FMA_F32`, `V_PK_MUL_F32`, and `V_PK_ADD_F32` read source dwords
  `[31:0]` and `[63:32]` at `cdna2/README.md:4150` through `:4156`.
- The CDNA2 XML under-sizes these operands as 32-bit at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:65825`, `:65872`, and
  `:65913`.

Rocjitsu evidence:

- Generated CDNA2 constructors expose `V_PK_FMA_F32`, `V_PK_MUL_F32`, and
  `V_PK_ADD_F32` destination and source operands as 32-bit at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop3p.cpp:665` through
  `:714`.
- The shared execution helper tries to recover VGPR pairs by calling
  `read_lane64()` only when the source encoding value is a VGPR. Otherwise it
  initializes the high word from the low 32-bit read at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:16702`,
  `:16889`, and `:17354` for the ADD, FMA, and MUL helpers respectively.
- CDNA2 already has scalar-pair support through `Operand::read_lane64()` and
  `resolve_src_scalar64()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/operand.cpp:1368` through
  `:1381`.

Impact:

The generated operand metadata misses the two-dword source/destination contract,
and scalar or other non-VGPR sources feed the low dword to both packed F32
components instead of reading a two-dword pair.

### CDNA2-RJ-002: Packed 32-bit VOP3P VGPR pair alignment is not validated

Manual evidence:

- CDNA2 section 6.7 says packed 32-bit operands must be two-dword aligned, with
  an even VGPR address, at `cdna2/README.md:1524` through `:1526`.

Rocjitsu evidence:

- `V_PK_FMA_F32`, `V_PK_MUL_F32`, `V_PK_ADD_F32`, and `V_PK_MOV_B32` execute by
  reading or writing 64-bit VGPR pairs at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:16702`,
  `:16889`, `:17354`, and `:17295` through `:17312`.
- `Isa::resolved_vgpr_offset()` accepts any source VGPR encoding and
  `Operand::read_lane64()` / `write_lane64()` use the unadjusted starting index
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/operand.cpp:1305`
  through `:1388`.

Impact:

Odd VGPR encodings are treated as valid packed 32-bit pairs rooted at the odd
register, whereas the manual restricts these operands to even-aligned pairs.

### CDNA2-RJ-003: Packed F16 VOP3P arithmetic ignores `CLMP`

Manual/XML evidence:

- The CDNA2 VOP3P field table defines `CLMP` as "1 = clamp result" at
  `cdna2/README.md:6606` through `:6607`.
- The CDNA2 XML describes VOP3P `CLAMP` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:1985`.

Rocjitsu evidence:

- `V_PK_ADD_F16`, `V_PK_FMA_F16`, `V_PK_MAX_F16`, `V_PK_MIN_F16`, and
  `V_PK_MUL_F16` execute without applying `inst.inst_.clamp` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:16664`
  through `:16698`, `:16839` through `:16884`, `:17089` through `:17127`,
  `:17192` through `:17230`, and `:17316` through `:17350`.
- MIX helpers do apply `std::clamp(result, 0.0f, 1.0f)` when
  `inst.inst_.clamp` is set at `execute_shared.h:13290` through `:13415`.

Impact:

Packed F16 results outside the clamp range remain unclamped when the VOP3P
`CLMP` bit is set.

### CDNA2-RJ-004: MFMA C/D register-bank requirements are not enforced

Manual/XML evidence:

- Chapter 7 says A and B sources can come from Arch or AccVGPR, while C and D
  use AccVGPRs, at `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:1541`
  through `:1545`.
- The detailed MFMA opcode table marks ordinary F32, F16, I8, and BF16 rows as
  using AccVGPR C/D operands at `cdna2/README.md:4158` through `:4189`, but
  marks BF16_1K and F64 rows as using ordinary VGPR C/D operands at
  `cdna2/README.md:4180` through `:4184` and `:4190` through `:4191`.
- The XML uses generic `OPR_VGPR_OR_ACCVGPR` / `OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST`
  C/D classes for representative ordinary F32, BF16_1K, and F64 entries at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:65995` through
  `:66028`, `:66776` through `:66808`, and `:67246` through `:67280`.

Rocjitsu evidence:

- Generated CDNA2 MFMA constructors apply `acc_cd` to VDST and SRC2 for
  ordinary F32 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop3p.cpp:777` through
  `:819`.
- The same generated pattern is used for BF16_1K at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop3p.cpp:1437` through
  `:1479` and F64 at `:1877` through `:1918`, even though the manual's detailed
  table lists those C/D operands as ordinary VGPRs.
- `mfma_src2_encoding()` rewrites SRC2 into the accumulator range whenever
  `acc_cd` is set at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop3p.cpp:25` through
  `:32`, and CDNA2 `resolve_acc()` routes through the separate AccVGPR file at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mma_exec.h:9` through
  `:24`.

Impact:

Rocjitsu accepts and executes C/D bank combinations that the detailed CDNA2
manual table does not describe as legal. In particular, BF16_1K and F64 can be
decoded/executed with AccVGPR C/D despite their rows naming VGPR C/D, and
ordinary MFMA rows can be executed without the AccVGPR C/D contract if
`ACC_CD` is clear.

### CDNA2-RJ-005: MFMA fused, denorm, rounding, and arithmetic-exception behavior is not fully modeled

Manual evidence:

- The MFMA family table says F32/F32 supports denorm allow/flush from
  `MODE.denorm`, F16/BF16/BF16_1K flush input and output denorms, and F64
  ignores `MODE` while forcing round-to-nearest-even and allowing input/output
  denorms at `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:1574` through
  `:1582`.
- The Chapter 12.10 MFMA rows describe multiply-add using fused multiply-add
  semantics across the MFMA table at `cdna2/README.md:4158` through `:4191`.
- The manual states that miSIMD does not support arithmetic exceptions at
  `cdna2/README.md:1591`.

Rocjitsu evidence:

- Representative CDNA2 F32, BF16_1K, and F64 generated execute paths call the
  shared MFMA helpers directly at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop3p.cpp:816` through
  `:819`, `:1476` through `:1479`, and `:1909` through `:1918`.
- The shared F16/BF16 extraction helpers convert raw halves to host `float`
  without an instruction-family denorm/MODE policy at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:636` through
  `:650`.
- `exec_f32_mixed()` accumulates with host `float` arithmetic and stores the
  resulting bit pattern without reading `MODE.denorm`, applying forced
  input/output denorm flushing, or modeling miSIMD exception suppression at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:1154` through
  `:1192`.
- The same helper documents its scalar path as non-fused multiply-add and uses
  `acc += a_val * b_val` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:1168` through
  `:1190`; the optimized SIMD path documents fused single-rounding behavior
  separately at `:1203` through `:1204`, so force-scalar or generic fallback
  execution can diverge from the fused MFMA contract.

Impact:

Fused multiply-add, denormal inputs/outputs, and rounding-sensitive cases can
diverge from CDNA2 hardware rules, especially for fallback/scalar execution, the
F16/BF16 forced-flush families, and F64's forced RNE/allow-denorm behavior.

### CDNA2-RJ-006: MFMA dependency-wait rules are not modeled

Manual evidence:

- Section 7.2 requires specific NOP or independent-VALU distances for VALU,
  XDL, DGEMM, VMEM/LDS/FLAT/Export, `V_CMPX`, and MFMA producer/consumer
  combinations at `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:1593`
  through `:1648`.
- The detailed opcode table gives block and pass counts at `cdna2/README.md:4158`
  through `:4191`; the dependency table depends on those pass counts.

Rocjitsu evidence:

- Codegen only marks MFMA instructions with the broad `MFMA` flag at
  `lib/python/amdisa/codegen/_generator.py:6539` through `:6542`.
- `ComputeUnit::issue_instruction()` executes the decoded instruction directly
  and routes only memory operations through the wait-counter pipeline at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:411` through `:437`.
- `WaitCounters` are documented as outstanding memory-operation counters for
  VMEM/LDS/GDS/K/M/export/vector-store/tensor/async operations, not MAI
  producer/consumer timing hazards, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wait_counters.h:36` through `:65`.

Impact:

Rocjitsu functional execution can run MFMA producer/consumer sequences that the
CDNA2 manual requires software to separate. If rocjitsu intentionally remains
non-cycle-accurate, this is still a missing diagnostic/metadata surface for the
manual scheduling contract.

### CDNA2-RJ-007: VOP3P-MAI `CBSZ` legal values are not validated

Manual evidence:

- The VOP3P-MAI field table says `CBSZ` legal values are `0-4` at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:6633`.

Rocjitsu evidence:

- Generated CDNA2 MFMA execute paths pass raw `inst_.cbsz` to the shared MFMA
  helpers, for example at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop3p.cpp:816` through
  `:819` and `:1476` through `:1479`.
- `permute_a_lane()` accepts the raw value and computes `64 >> cbsz` whenever
  `cbsz != 0`, with no CDNA2 legal-value check, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:568` through
  `:579`.

Impact:

Encodings with undefined `CBSZ` values `5-7` are decoded and executed using the
helper's shift formula instead of being rejected or reported as undefined.

### CDNA2-RJ-008: MFMA SRC2 accepts generic constants beyond the MFMA subset

Manual evidence:

- Chapter 7 lists unsupported MFMA source forms and says the only inline
  constants interpreted as FP32 for all `V_MFMA` and `V_ACCVGPR` instructions
  are `0.5`, `-0.5`, `1.0`, `-1.0`, `2.0`, `-2.0`, `4.0`, and `-4.0` at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:1584` through `:1589`.
- The generic VOP3P-MAI field table still lists broader source-encoding
  options, including integer constants and `1/(2*PI)`, at `cdna2/README.md:6638`
  through `:6640`; this gap follows the narrower Chapter 7 MFMA-specific rule.

Rocjitsu evidence:

- Representative CDNA2 MFMA constructors use
  `OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST` for SRC2 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop3p.cpp:777` through
  `:819`, `:1437` through `:1479`, and `:1877` through `:1918`.
- The CDNA2 operand printer and scalar resolution path for that class accept
  generic integer inline constants and `0.15915494` in addition to the MFMA
  subset at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/operand.cpp:456`
  through `:507`.

Impact:

Rocjitsu can disassemble and execute MFMA SRC2 encodings using inline constants
that the CDNA2 Chapter 7 MFMA-specific prose does not list as legal for MFMA.

### CDNA2-RJ-009: F64 MFMA ignores VOP3P-MAI broadcast/swizzle fields

Manual evidence:

- The CDNA2 VOP3P-MAI table describes `CBSZ` as the number of blocks that can
  broadcast, `ABID` as the block ID to broadcast during MFMA, and `BLGP` as the
  B-matrix lane-group pattern at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:6633` through `:6642`.
- The detailed F64 MFMA rows are ordinary MFMA entries in the same opcode table
  at `cdna2/README.md:4190` through `:4191`. This slice found no CDNA2 manual
  exception that repurposes or disables the VOP3P-MAI broadcast/swizzle fields
  for F64.

Rocjitsu evidence:

- CDNA2 `V_MFMA_F64_16X16X4F64` and `V_MFMA_F64_4X4X4F64` call `exec_f64()`
  without passing `inst_.cbsz`, `inst_.abid`, or `inst_.blgp` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop3p.cpp:1909` through
  `:1918` and `:1953` through `:1962`.
- The shared `exec_f64()` helper only accepts a `neg` field and does not apply
  A/B broadcast or B-lane-group swizzling at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:3493` through
  `:3530`.

Impact:

CDNA2 F64 MFMA encodings with nonzero VOP3P-MAI broadcast/swizzle fields execute
as if those fields were zero. If later hardware or assembler probes show these
fields are ignored for CDNA2 F64, this should be downgraded to a manual/XML
ambiguity; the checked manual slice does not state that exception.

### CDNA2-RJ-010: 64-bit MFMA VGPR/AccVGPR pair alignment is not validated

Manual/XML evidence:

- The CDNA2 XML operand definitions state that 64-bit and wider VGPR values
  must be even-aligned unless otherwise noted at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:106209` and `:108779`.
- The XML also states the same even-alignment rule for 64-bit and wider
  AccVGPR values at `amdgpu_isa_cdna2.xml:107491` and `:110061`.
- Representative BF16_1K and F64 MFMA entries use 64-bit source operands at
  `amdgpu_isa_cdna2.xml:66776` through `:66808` and `:67246` through `:67280`.

Rocjitsu evidence:

- `Isa::resolved_vgpr_offset()` returns the encoded VGPR/AccVGPR offset without
  checking even alignment at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/operand.cpp:1305` through
  `:1318`.
- `Operand::read_lane64()` then reads a 64-bit pair from that unvalidated base
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/operand.cpp:1368`
  through `:1381`.
- Generated CDNA2 BF16_1K and F64 MFMA constructors accept 64-bit source
  operands without adding an alignment check at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop3p.cpp:1437` through
  `:1479` and `:1877` through `:1918`.

Impact:

Odd VGPR or AccVGPR bases for 64-bit MFMA operands are treated as valid pairs
rooted at the odd register, even though the operand metadata defines those
64-bit register values as even-aligned.

### CDNA2-RJ-011: CDNA2-specific MFMA semantic regressions are not targeted by tests

Evidence:

- The instruction execution coverage exception file excludes vector execution
  from the scalar harness and relies on kernel simulation coverage at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/coverage_exceptions_cdna2.txt:4`
  through `:13`.
- Existing generic VM coverage includes a basic MFMA F16 accumulation test at
  `tests/amdgpu_vm_test.cpp:1642` and following.
- The bit-exact SIMD-vs-scalar MFMA test fixture is explicitly CDNA4-only at
  `tests/simd_correctness/mfma_simd_exact_test.cpp:4` through `:26`.
- The codegen unit test covers generic `ACC_CD` expression generation at
  `lib/python/amdisa/tests/test_semantic_operand_codegen.py:108` and
  following, but not CDNA2 per-opcode C/D bank legality.

Impact:

The current tests can miss CDNA2-specific MFMA issues such as invalid `CBSZ`,
F64 broadcast/swizzle handling, MFMA-only inline constants, 64-bit pair
alignment, denorm/MODE behavior, and BF16_1K/F64 C/D bank restrictions.

### CDNA2-RJ-012: Device-memory consistency and acknowledgment behavior is not represented

Manual evidence:

- Section 2.3 describes the CDNA device-memory hierarchy, cache-less loads,
  load-clause overlap caching, write-combining, atomic pre-op return storage,
  write-confirmation acknowledgments, relaxed consistency, per-PE/per-channel
  scatter-write ordering, and acknowledgment/fence use at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:454` through `:463`.

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

### CDNA2-RJ-013: Retained GWS instructions decode but cannot execute

Manual evidence:

- The MI200 feature-change summary says GDS operations are removed while GWS
  operations are retained at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:295` through `:303`.
- Section 2.2.2 says CDNA devices contain a global synchronization unit capable
  of synchronizing workgroups across the device at `cdna2/README.md:450`
  through `:452`.
- Chapter 11.4 says all GWS instructions must be followed immediately by
  `s_waitcnt 0` and that VGPRs used by GWS instructions must be even at
  `cdna2/README.md:2817` through `:2825`.
- The detailed GWS definitions describe resource-id calculation and semaphore
  or barrier state-machine behavior at `cdna2/README.md:4591` through `:4625`.

Rocjitsu evidence:

- Generated CDNA2 GWS classes decode `DS_GWS_SEMA_RELEASE_ALL`,
  `DS_GWS_INIT`, `DS_GWS_SEMA_V`, `DS_GWS_SEMA_BR`, `DS_GWS_SEMA_P`, and
  `DS_GWS_BARRIER`, but every execute body throws `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/ds.cpp:4749` through
  `:4823`.
- Generated ordinary CDNA2 DS execute bodies also throw
  `util::UnimplementedInst` whenever `inst_.gds` is set; a representative
  atomic path is `DsAddU32Ds` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/ds.cpp:42` through `:64`.
  That is plausible for removed GDS data-share operations, but it leaves no
  retained GWS execution path.
- Codegen classifies `DS_GWS_*` semantics as `nop` because they are hardware
  scheduling primitives at `lib/python/amdisa/semantics.py:2398` through
  `:2401`.
- The static pass did not find a CDNA2 GWS execution path or validation for
  resource state, immediate-following `s_waitcnt 0`, or even-VGPR restrictions.

Impact:

LDS forms execute functionally, and removed CDNA2 GDS data-share operations may
reasonably remain unimplemented, but the retained GWS synchronization opcodes
decode to classes that cannot execute or validate the manual's state, sequence,
and register constraints.

### CDNA2-RJ-014: Allocation-bound GPR out-of-range behavior is not modeled

Manual evidence:

- Section 3.6.1 says out-of-range SGPR sources return SGPR0, out-of-range SGPR
  destinations write no result, out-of-range VGPR sources use VGPR0,
  out-of-range VGPR destinations are treated as NOPs, memory-return
  destination VGPR checks nullify the operation, and multiple-destination
  instructions write no results if any destination is out of range at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:585` through `:623`.
- Section 3.6.2 says wavefront SGPR allocations are 16 to 102 dwords, while
  user-visible logical SGPRs and VCC aliases extend beyond some allocation
  sizes at `cdna2/README.md:625` through `:627`.
- Section 3.6.4 says VGPRs are allocated in groups of eight dwords and a wave
  may have fewer than the maximum architectural VGPRs at `cdna2/README.md:638`
  through `:648`.

Rocjitsu evidence:

- CDNA2 scalar operand reads and writes directly add the encoded SGPR selector
  to `wf.sgpr_alloc().base` for SGPR and TTMP ranges, without comparing against
  `wf.sgpr_alloc().count`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/operand.cpp:1016` through
  `:1028`, `:1179` through `:1204`, and `:1221` through `:1241`.
- CDNA2 VGPR lane reads and writes resolve the encoded VGPR offset and access
  `wf.vgpr_alloc().base + voff` without checking `wf.vgpr_alloc().count` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/operand.cpp:1305` through
  `:1347` and `:1359` through `:1388`.
- `ComputeUnitCore::alloc_wf()` stores the requested allocation counts in the
  wavefront at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:136` through `:153`,
  but the generated CDNA2 operand accessors above do not use those counts to
  implement SGPR0/VGPR0 fallback or destination nullification.

Impact:

CDNA2 kernels can read or write physical registers outside the wavefront's
declared allocation instead of receiving the manual's SGPR0/VGPR0 fallback,
write suppression, or nullification behavior. This affects both functional
execution and diagnostics for bad GPR indexing.

### CDNA2-RJ-015: CDNA2 HWREG get/set uses the wrong register map

Manual/XML evidence:

- The manual state table identifies STATUS as read-only shader status and MODE
  as writable shader mode at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:491` through `:492`, with
  detailed STATUS and MODE fields at `cdna2/README.md:525` through `:579`.
- The CDNA2 XML maps `HW_REG_MODE` to ID 1, `HW_REG_STATUS` to ID 2,
  `HW_REG_TRAPSTS` to ID 3, `HW_REG_HW_ID` to ID 4, `HW_REG_GPR_ALLOC` to ID 5,
  `HW_REG_LDS_ALLOC` to ID 6, and `HW_REG_IB_STS` to ID 7 at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:99185` through
  `:99218`.
- `S_GETREG_B32`, `S_SETREG_B32`, and `S_SETREG_IMM32_B32` are represented as
  HWREG access instructions at `amdgpu_isa_cdna2.xml:44148` through `:44243`.

Rocjitsu evidence:

- CDNA2 `SGetregB32Sopk::execute_impl()` returns `wf.status_raw()` for ID 1,
  returns CU ID fragments for IDs 4 and 5, and returns SGPR/VGPR allocation
  values for IDs 6 and 7 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopk.cpp:291` through
  `:320`.
- CDNA2 `SSetregB32Sopk` and `SSetregImm32B32Sopk` only handle ID 1 and splice
  into `wf.status_raw()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopk.cpp:334` through
  `:385`.
- `Wavefront` stores raw STATUS and raw MODE separately at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:95` through `:110`.

Impact:

CDNA2 HWREG ID 1 acts like STATUS in rocjitsu even though XML/manual identify
it as MODE. STATUS ID 2, TRAPSTS ID 3, GPR_ALLOC ID 5, LDS_ALLOC ID 6, and
IB_STS ID 7 are either unreachable or return unrelated data, so HWREG execution
can diverge from both disassembly metadata and hardware state.

### CDNA2-RJ-016: Raw STATUS EXECZ/VCCZ bits can drift from live EXEC/VCC state

Manual evidence:

- Section 3.3 defines `EXECZ` as the helper bit for zero EXEC branch tests at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:515` through `:523`.
- The STATUS table defines `EXECZ` and `VCCZ` bits at `cdna2/README.md:525`
  through `:539`.
- Section 3.9 says VCCZ is updated every time VCC is updated, including vector
  compares and scalar writes to VCC, at `cdna2/README.md:683` through `:690`.

Rocjitsu evidence:

- `Wavefront::set_exec()` and `Wavefront::set_vcc()` update separate EXEC and
  VCC storage but do not update raw STATUS bits at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:211` through `:238`.
- CDNA2 special scalar source resolution computes VCCZ and EXECZ from live
  `wf.vcc()` and `wf.exec()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/operand.cpp:1073` through
  `:1082`.
- CDNA2 VCCZ/EXECZ branches also test live VCC or EXEC at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopp.cpp:125` through
  `:155` and `:173` through `:199`.
- Raw STATUS is still separately exposed by `status_raw()` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:608` through `:617` and by
  the generated HWREG path above.

Impact:

Branches and special scalar sources can observe the current EXEC/VCC value while
`S_GETREG_B32` or raw status inspection observes stale or independently written
STATUS helper bits. That splits one architectural state contract into two
different rocjitsu views.

### CDNA2-RJ-017: `S_SETVSKIP` decodes but VSKIP execution is absent

Manual/XML evidence:

- Section 3.3 recommends VSKIP to skip code quickly when EXEC is zero at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:521` through `:523`.
- The MODE table defines `VSKIP` bit 28 as skipping VALU, VMEM, LDS, and GDS
  instructions at `cdna2/README.md:575` through `:579`.
- The XML describes `S_SETVSKIP` as enabling or disabling VSKIP mode at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:42972` through
  `:42984`.

Rocjitsu evidence:

- CDNA2 `SSetvskipSopc::execute_impl()` throws `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopc.cpp:388` through
  `:409`.
- `ComputeUnitCore::issue_instruction()` calls `execute_instruction()` and then
  routes memory operations without a VSKIP gate at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:393` through `:437`.

Impact:

Code that uses CDNA2 VSKIP cannot execute, and there is no runtime path that
skips the manual's vector instruction classes when the MODE VSKIP bit is set.

### CDNA2-RJ-018: LDS workgroup allocation uses 256-byte granularity instead of 512-byte blocks

Manual evidence:

- Section 3.6.5 says LDS is allocated in contiguous blocks of 128 dwords on
  128-dword alignment at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:652` through `:654`.
- The same section says LDS accesses are clamped by the smaller of the SPI
  allocation size and the instruction M0 size at `cdna2/README.md:654` through
  `:656`.

Rocjitsu evidence:

- The command-processor planning helper aligns LDS bytes per workgroup to 256
  bytes at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:99` through
  `:102`.
- `ComputeUnitCore::can_accept_workgroup()` also aligns requested LDS bytes to
  256 bytes before checking capacity at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:243` through `:249`.

Impact:

rocjitsu can pack CDNA2 workgroups using half the manual allocation alignment.
That can admit workgroup placements or overlap conditions that hardware's
128-dword block allocation would not.

### CDNA2-RJ-019: Trap, exception, TRAPSTS, and memory-violation state is not modeled

Manual/XML evidence:

- Section 3.10 defines TTMP trap payload state, `STATUS.TRAP_EN`, and
  `MODE.EXCP_EN` trap gating at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:701` through `:728`.
- Section 3.10.1 defines sticky TRAPSTS fields at `cdna2/README.md:730`
  through `:743`.
- Section 3.11 defines memory-violation sources, sticky
  `TRAPSTS.mem_viol`, exception enable behavior, and imprecise saved PC at
  `cdna2/README.md:745` through `:767`.
- XML identifies `S_RFE_B64`, `S_TRAP`, and `V_CLREXCP` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:32316` through
  `:32328`, `:44905` through `:44917`, and `:50709` through `:50716`.

Rocjitsu evidence:

- CDNA2 `STrapSopp::execute_impl()` throws `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopp.cpp:301` through
  `:313`.
- Shared `S_RFE_B64`, `S_SENDMSG`, and `V_CLREXCP` helpers are empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2405`
  through `:2428` and `:3795` through `:3799`.
- The CDNA2 HWREG implementation above does not expose TRAPSTS ID 3, and a
  source/test search found no modeled `trapsts`, `mem_viol`, `SAVECTX`,
  `EXCP_CYCLE`, or `DP_RATE` state.

Impact:

rocjitsu cannot model CDNA2 trap entry, trap return, trap-status stickiness,
memory-violation reporting, or exception clear behavior. Trap-control opcodes
either throw or act as empty functional stubs.

### CDNA2-RJ-020: TTMP privilege is not modeled

Manual evidence:

- The state table says `TTMP0-TTMP15` are SGPRs available only to the trap
  handler at `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:499` through
  `:505`.
- Section 3.6.2 says the trap-handler SGPR reserve is privileged at
  `cdna2/README.md:625` through `:627`.
- Section 3.10 says TTMP writes are only allowed in the trap handler when
  `status.priv = 1`, and unprivileged writes are ignored, at
  `cdna2/README.md:701` through `:705`.

Rocjitsu evidence:

- CDNA2 scalar source resolution reads TTMP selector encodings 108 through 123
  directly from the wavefront SGPR block at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/operand.cpp:1027` through
  `:1028`.
- CDNA2 scalar destination writes and 64-bit destination writes also write
  TTMP selectors directly at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/operand.cpp:1202` through
  `:1204` and `:1237` through `:1241`.
- `RegisterSet` documents TTMP as a special class that is not tracked by its
  def/use set at `lib/rocjitsu/src/rocjitsu/isa/register_set.h:52` through
  `:60`, and this slice found no CDNA2 privilege gate in the operand write
  path.

Impact:

Unprivileged CDNA2 code can read and overwrite TTMP storage in rocjitsu instead
of having trap-handler-only write access with ignored unprivileged writes.

### CDNA2-RJ-021: `HW_ID` contents are incomplete and shifted by the HWREG map mismatch

Manual evidence:

- Section 3.12 defines `HW_ID` field packing for WAVE_ID, SIMD_ID, PIPE_ID,
  CU_ID, SH_ID, SE_ID, TG_ID, VM_ID, QUEUE_ID, STATE_ID, and ME_ID at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:769` through `:773`.
- The CDNA2 XML assigns `HW_REG_HW_ID` to HWREG ID 4 at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:99200` through
  `:99203`.

Rocjitsu evidence:

- CDNA2 `SGetregB32Sopk::execute_impl()` returns only `wf.cu().id()` for ID 4
  and uses ID 5 for the high CU-ID fragment, even though XML ID 5 is
  `HW_REG_GPR_ALLOC`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopk.cpp:291` through
  `:312`.
- `Wavefront` tracks state that could populate some HW_ID fields, such as
  wavefront slot, workgroup ID, dispatch ID, and process ID, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:147` through `:165`, but
  the CDNA2 HWREG path does not pack those fields.

Impact:

Even apart from the HWREG ID mismatch, `s_getreg_b32 hwreg(HW_ID,...)` cannot
return the manual's CDNA2 hardware-ID layout. Most fields are absent, and ID 5
collides with the XML/manual GPR allocation register.

### CDNA2-RJ-022: `S_RFE_B64` does not clear PRIV or branch to the return address

Manual/XML evidence:

- Chapter 4.1 lists `S_RFE` as the trap-handler return instruction at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:787` through `:793`.
- The XML `S_RFE_B64` description says the instruction returns from the
  exception handler, clears the wave's PRIV bit, and jumps to the scalar input
  address at `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:32316`
  through `:32335`.

Rocjitsu evidence:

- The CDNA2 generated `SRfeB64Sop1` constructor decodes the 64-bit source, but
  its execution delegates to `execute_s_rfe_b64_sop1()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sop1.cpp:596` through
  `:608`.
- The shared helper body is empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2405`
  through `:2406`.
- The CDNA2 `S_RFE_RESTORE_B64` form is decoded but throws
  `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sop2.cpp:1011` through
  `:1033`.

Impact:

Trap-handler return code cannot clear privileged mode or resume at the saved
PC in rocjitsu; `S_RFE_B64` behaves as a no-op and `S_RFE_RESTORE_B64` traps in
the emulator.

### CDNA2-RJ-023: Debug conditional branches never branch

Manual evidence:

- Chapter 4.2 lists `S_CBRANCH_CDBGSYS`, `S_CBRANCH_CDBGUSER`, and
  `S_CBRANCH_CDBGSYS_AND_USER` as branches based on the corresponding
  `COND_DBG_SYS` and `COND_DBG_USER` STATUS bits at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:804` through `:810`.

Rocjitsu evidence:

- CDNA2 constructors for the four debug conditional branches decode the label
  operand but do not set branch metadata flags, and each `execute_impl()`
  delegates to a shared helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopp.cpp:363` through
  `:413`.
- The shared helpers for `S_CBRANCH_CDBGSYS`,
  `S_CBRANCH_CDBGSYS_AND_USER`, `S_CBRANCH_CDBGSYS_OR_USER`, and
  `S_CBRANCH_CDBGUSER` are empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:997`
  through `:1011`.
- The shared CDNA status wrapper exposes `COND_DBG_USER` and `COND_DBG_SYS`
  accessors but comments that they are inert on CDNA1/2 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/cdna_isa_base.h:73`
  through `:78`; this finding follows the checked CDNA2 manual table, which
  still lists the branch opcodes.

Impact:

Debug-status conditional branches fall through regardless of STATUS, and
static branch analysis also lacks normal conditional-branch metadata for these
generated CDNA2 classes.

### CDNA2-RJ-024: Fork/join divergent control flow is not implemented

Manual evidence:

- Chapter 4.2 lists `S_CBRANCH_{G,I}_FORK` and `S_CBRANCH_JOIN` as conditional
  branch instructions for complex branching at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:815` through `:823`.
- Chapter 4.6 defines a six-deep CSP stack, `{EXEC, PC}` stack entries in
  SGPRs, pass/fail mask selection by bitcount, EXEC updates, branch target
  selection, and JOIN restoration at `cdna2/README.md:887` through `:936`.

Rocjitsu evidence:

- CDNA2 `S_CBRANCH_I_FORK` decodes the mask SGPR-pair and label but throws
  `util::UnimplementedInst` in `execute_impl()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopk.cpp:264` through
  `:278`.
- CDNA2 `S_CBRANCH_G_FORK` decodes its two 64-bit sources and also throws
  `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sop2.cpp:962` through
  `:984`.
- CDNA2 `S_CBRANCH_JOIN` decodes the source but dispatches to an empty shared
  helper at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sop1.cpp:916`
  through `:930` and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1013`
  through `:1015`.
- Searching rocjitsu for `CSP`, `control stack`, and fork/join helpers finds
  only generated decode fixtures and the unimplemented/empty execution paths,
  not a branch-stack state model.

Impact:

CDNA2 kernels that use compiler-emitted fork/join divergent control flow either
throw on the FORK instruction or fail to restore EXEC/PC at JOIN.

### CDNA2-RJ-025: Program-control status, priority, message, and wakeup side effects are mostly stubs

Manual evidence:

- Chapter 4.1 says `S_SETPRIO` modifies wave priority, `S_SLEEP` sleeps the
  wave, and `S_SENDMSG` sends a host message at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:787` through `:796`.
- XML entries say `S_WAKEUP` wakes other waves in the threadgroup,
  `S_SETHALT` sets or clears HALT, `S_SETKILL` kills the wave when the low bit
  is set, and `S_SENDMSGHALT` sends a message and halts at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:44399` through
  `:44406`, `:44671` through `:44684`, `:44735` through `:44748`, and
  `:44868` through `:44886`.

Rocjitsu evidence:

- CDNA2 `S_SETHALT`, `S_SETKILL`, `S_SETPRIO`, `S_SENDMSG`,
  `S_SENDMSGHALT`, and `S_WAKEUP` dispatch to shared helpers, while `S_SLEEP`
  dispatches to a helper that only requests a functional yield. Representative
  generated calls are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopp.cpp:58` through
  `:64`, `:213` through `:298`, and `:363` through `:423`.
- The shared helpers for `S_SENDMSG`, `S_SENDMSGHALT`, `S_SETHALT`,
  `S_SETKILL`, `S_SETPRIO`, and `S_WAKEUP` are empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2427`
  through `:2428`, `:2490` through `:2492`, `:2526` through `:2533`, and
  `:2689` through `:2690`.
- `S_SLEEP` only calls `wf.cu().request_functional_yield()` and does not model
  the immediate value's sleep duration or `S_WAKEUP` interaction at
  `execute_shared.h:2555` through `:2558`.

Impact:

Control/status instructions used for halt, kill, priority, host messages, and
sleep/wakeup scheduling have decode coverage but do not update the wave state
or scheduler behavior described by the manual/XML descriptions.

### CDNA2-RJ-026: `S_BARRIER` does not expose `STATUS.IN_BARRIER`

Manual evidence:

- Chapter 3 defines `STATUS.IN_BARRIER` bit 12 as "Wavefront is waiting at a
  barrier" at `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:525` through
  `:541`.
- Chapter 4.3 says `S_BARRIER` waits until the live workgroup waves reach the
  same instruction at `cdna2/README.md:825` through `:827`.

Rocjitsu evidence:

- The CDNA status wrapper defines `IN_BARRIER` as bit 12 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/cdna_isa_base.h:59`
  through `:62`.
- CDNA2 `S_GETREG_B32` reads raw STATUS through `wf.status_raw()` for HWREG id
  1 at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopk.cpp:291`
  through `:300`, and the wavefront stores raw status independently of
  scheduler state at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:608`
  through `:617`.
- `execute_s_barrier_sopp()` only sets `WfState::BARRIER` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:745`
  through `:747`; it does not set bit 12 while the wave is waiting or clear it
  when the CU releases the barrier.
- The CU release path changes only scheduler state from `BARRIER` back to
  `RUNNING` at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:313`
  through `:335`.

Impact:

Barrier scheduling can block and release waves, but shader/debug code that
observes raw STATUS through HWREG cannot see the architectural `IN_BARRIER`
bit while a wave is waiting.

### CDNA2-RJ-027: Workgroup size limits are not validated

Manual evidence:

- Chapter 4.3 says up to 16 wavefronts or 1024 work-items can be combined into
  a workgroup at `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:825`
  through `:827`.

Rocjitsu evidence:

- The AQL dispatch path computes `wg_size` directly from packet workgroup
  dimensions, derives `wfs_per_wg = (wg_size + wave_size - 1) / wave_size`,
  and stores it in the dispatch entry at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:941` through
  `:965`.
- The workgroup admission check only tests available wavefront slots, SGPR
  blocks, VGPR blocks, and LDS capacity at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:215` through `:251`.
- A focused search found no architectural validation rejecting workgroups above
  16 waves or 1024 work-items.

Impact:

Rocjitsu can accept a dispatch shape that exceeds CDNA2's documented per
workgroup size limit if enough simulator resources are configured, which can
change residency, barrier membership, and launch-state behavior for malformed
or stress dispatches.

### CDNA2-RJ-028: Scalar-memory `LGKM_CNT` accounting ignores dword counts

Manual evidence:

- Chapter 4.4 says `LGKM_CNT` is incremented by the dword count for scalar
  memory reads, with one count for one-dword loads and two counts for
  two-dword-or-larger loads; `S_MEMTIME` counts like `S_LOAD_DWORDX2` at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:845` through `:846`.
- The same section says `LGKM_CNT` decrements once for each dword returned
  from the data cache for SMEM, and that scalar-memory reads can return out of
  order so only `S_WAITCNT 0` is legitimate for such dependencies, at
  `cdna2/README.md:850` through `:852`.
- Chapter 8.3 repeats the SMEM-specific rule that `LGKM_CNT` increments by one
  for one-dword fetches, by two for two-or-more-dword fetches, and decrements by
  the same amount when the instruction completes at `cdna2/README.md:1770`
  through `:1778`; `S_DCACHE_DISCARD_X2` likewise counts as two in the detailed
  opcode table at `cdna2/README.md:3239` through `:3240`.

Rocjitsu evidence:

- `ScalarMemState` carries `num_dwords`, but only one
  `wait_counter_type = WaitCounterType::LGKMCNT` value at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:70` through `:83`.
- The generated CDNA2 SMEM load bodies do set `num_dwords`, for example
  `S_LOAD_DWORD` uses `1` and `S_LOAD_DWORDX2` uses `2` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/smem.cpp:56` through
  `:94`.
- `MemoryPipeline::issue()` increments exactly one wait counter for the
  instruction at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.h:62`
  through `:88`.
- `ScalarMemPipeline::complete_access()` writes every returned dword to SGPRs,
  but completion returns once and the deferred callback releases that same
  single counter at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:203` through
  `:241`.
- `S_MEMTIME` and `S_MEMREALTIME` are not issued through the scalar memory
  pipeline at all: the generated constructors omit `MEMORY_OP`, and the helpers
  write synthetic counters immediately at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/smem.cpp:687` through `:710`
  and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1934`
  through `:1950`.

Impact:

Functional execution writes the right SGPR payload, but outstanding
`LGKM_CNT` depth does not match CDNA2 for multi-dword SMEM loads or
`S_MEMTIME`; discard instructions are unimplemented and therefore cannot
contribute their documented counts. Thresholds such as `lgkmcnt(1)` can
therefore unblock differently from hardware in timing/deferred paths, and the
model cannot represent the manual's per-dword SMEM return accounting.

### CDNA2-RJ-029: GWS/GDS `EXP_CNT` producer behavior is not modeled in production

Manual evidence:

- Chapter 4.4 defines `EXP_CNT` as the VGPR-export count for GDS, incremented
  when a GDS instruction issues from the wavefront buffer and decremented when
  the last GDS cycle is granted/executed and VGPRs have been read out at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:853` through `:855`.
- The CDNA2 feature-change summary says GDS operations are removed while GWS
  operations are retained at `cdna2/README.md:295` through `:303`; retained GWS
  decode/execution is already tracked in `CDNA2-RJ-013`.

Rocjitsu evidence:

- The generated CDNA2 GWS instruction classes decode the GWS opcodes, but all
  six `execute_impl()` bodies immediately throw `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/ds.cpp:4749` through
  `:4823`.
- A targeted search for `WaitCounterType::EXPCNT` found only the wait-counter
  primitive and unit/infrastructure tests, not a production instruction path
  that issues or retires `EXP_CNT`.
- Existing broader GWS behavior is already tracked in `CDNA2-RJ-013`; this
  item records the specific Chapter 4.4 wait-counter consequence.

Impact:

Rocjitsu can decode GWS instructions, but it cannot model the CDNA2
`LGKM_CNT`/`EXP_CNT` effects that make GWS VGPR sources safe to overwrite only
after the export count retires.

### CDNA2-RJ-030: Manual NOP wait-state hazards are not modeled or diagnosed

Manual evidence:

- Chapter 4.5 says hardware does not check several dependency classes, so the
  shader must resolve them by inserting NOPs or independent instructions at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:857` through `:859`.
- Table 9 lists required waits for `S_SETREG`/`S_GETREG`, `MODE.VSKIP`,
  VALU-produced VCC/EXEC/SGPR/VGPR values, lane-select consumers,
  `V_DIV_FMAS`, wide store/atomic write-data hazards, M0 consumers,
  TRAPSTS/RFE, DPP, mixed VCC aliasing, and `S_MOVEREL` at
  `cdna2/README.md:861` through `:883`.

Rocjitsu evidence:

- Generated CDNA2 `S_NOP` constructs a SOPP instruction and delegates to the
  shared helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopp.cpp:20` through
  `:28`, but the shared helper is an empty function at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2137`
  through `:2138`.
- The functional memory pipeline documents synchronous completion and has a
  no-op `tick()` in functional mode at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.h:55` through `:91`.
- The existing DBT `HazardTracker` is explicitly a GFX12 `s_delay_alu`
  auto-insertion helper at
  `lib/rocjitsu/src/rocjitsu/code/dbt/hazard_tracker.h:4` through `:21`, not a
  CDNA2 Table 9 `S_NOP` timing or runtime diagnostic model.
- Some rows overlap existing behavior gaps, such as unimplemented `S_SETVSKIP`
  behavior in `CDNA2-RJ-017` and trap/TRAPSTS/RFE behavior in
  `CDNA2-RJ-019`/`CDNA2-RJ-022`; this finding records the broader absence of a
  CDNA2 Table 9 wait-state/scoreboard model.

Impact:

Rocjitsu executes most producer/consumer pairs with fully updated
architectural state at instruction boundaries. A kernel that omits required
CDNA2 NOPs can therefore appear correct under emulation even when hardware
requires padding or independent instructions to avoid races.

### CDNA2-RJ-031: Program-flow tests are mostly decode fixtures or basic smoke cases

Rocjitsu evidence:

- Generated CDNA2 encoding fixtures include `s_rfe_b64`,
  `s_cbranch_join`, debug conditional branches, fork forms, `s_sethalt`,
  `s_setkill`, `s_setprio`, `s_sendmsg`, `s_sendmsghalt`, and `s_wakeup` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/test_encodings.h:53`,
  `:68`, `:99` through `:122`, `:143`, and `:188` through `:190`.
- The instruction execution harness explicitly excludes several control
  instructions, including branches, waitcnt, barrier, trap, sleep, sethalt,
  sendmsg, sendmsghalt, nop, getpc/getreg/setreg, and rfe at
  `emulation/rocjitsu/tests/instruction_execution_harness_test.cpp:138`
  through `:177`.
- `HookOrderingTest.BarrierTwoWaves` runs `{S_BARRIER, S_ENDPGM}` with two
  waves in one workgroup and asserts one `BARRIER_RESOLVED` event plus two
  dispatched and halted wavefronts at
  `emulation/rocjitsu/tests/execution_plugin_test.cpp:870` through `:886`.
- Searching the adjacent tests found no behavior cases for CDNA2 debug branch
  predicates, RFE/RFE_RESTORE, fork/join stack effects, `S_SENDMSG`
  `LGKM_CNT`, scalar-memory dword-count waits, `STATUS.IN_BARRIER`,
  early-terminated barrier peers, single-wave immediate barrier release,
  oversized workgroup rejection, or the Chapter 4.5 wait-state hazard table.

Impact:

The current tests confirm useful decode coverage and one baseline barrier hook
case, but would not catch most of the no-op, unimplemented, or under-modeled
program-flow behavior recorded in the Chapter 4 audit entries above.

### CDNA2-RJ-032: SALU `POPS_EXITING_WAVE_ID` and reserved source selectors are shifted

Manual/XML evidence:

- Chapter 5.2 places `POPS_EXITING_WAVE_ID` at scalar selector 239, reserves
  selectors 249-250, and uses 251-253 for VCCZ, EXECZ, and SCC at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:1003` through `:1024`.
- The CDNA2 XML agrees: `OPR_SSRC` lists `SRC_POPS_EXITING_WAVE_ID` with
  value 239 at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:123258` through
  `:123261`.

Rocjitsu evidence:

- CDNA2 `resolve_src_scalar()` handles selectors 235-238, then maps selector
  249 to `SRC_POPS_EXITING_WAVE_ID` and selector 250 to `NULL`, while selector
  239 falls through to an unsupported scalar-read exception at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/operand.cpp:1061` through
  `:1083`.
- `can_resolve_src_scalar()` likewise returns true for 240-253, but excludes
  selector 239 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/operand.cpp:1115` through
  `:1118`.

Impact:

Legal selector 239 cannot be resolved by CDNA2 scalar operand execution or SIMD
capability checks, while reserved selectors 249-250 are accepted and resolved
as if they were defined.

### CDNA2-RJ-033: 64-bit SALU SGPR pair alignment is not enforced

Manual/XML evidence:

- Chapter 5.2 requires 64-bit SGPR data to start on an even SGPR boundary at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:1032`.
- The XML operand metadata carries the same generic alignment note for
  `OPR_SDST` and `OPR_SSRC` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:99335` through `:99349`
  and `:120008` through `:120026`.

Rocjitsu evidence:

- Generated CDNA2 `S_MOV_B64` exposes 64-bit `OPR_SDST` and `OPR_SSRC`
  operands but adds no alignment check at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sop1.cpp:39` through `:53`.
- `resolve_src_scalar64()` reads `ev` and `ev + 1` directly for SGPR and TTMP
  pairs, and `resolve_dst_write64()` writes `ev` and `ev + 1` directly, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/operand.cpp:1121` through
  `:1138` and `:1221` through `:1241`.

Impact:

Odd SGPR encodings for 64-bit SALU operands are treated as valid pairs rooted
at the odd register instead of being rejected or diagnosed as undefined.

### CDNA2-RJ-034: SALU implicit SCC dependencies are not surfaced in C++ def-use metadata

Manual/XML evidence:

- Chapter 5 says many SALU operations set SCC to report comparison, carry-out,
  overflow, or nonzero-result state at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:1034` through `:1043`.
- The XML represents those SCC effects as implicit operands: `S_ADD_U32` has
  an implicit SCC output at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:34005` through
  `:34035`, `S_CMP_EQ_I32` has an implicit SCC output at `:41132` through
  `:41155`, and `S_CBRANCH_SCC0` has an implicit SCC input at `:44424`
  through `:44441`.

Rocjitsu evidence:

- Representative generated CDNA2 constructors expose only explicit operands:
  `S_ADD_U32` has one destination and two sources at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sop2.cpp:20` through `:40`,
  and `S_CMP_EQ_I32` has two sources and no destination at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopc.cpp:20` through `:40`.
- Generic def-use construction only records explicit operands plus subclass
  `implicit_defs()` and `implicit_uses()` hooks at
  `lib/rocjitsu/src/rocjitsu/analysis/def_use_chain.cpp:25` through `:45`;
  `Instruction`'s default hooks are empty at
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:215` through `:222`.
- `RegisterSet` documents SCC as not currently tracked and ignores non-SGPR,
  non-VGPR, and non-AccVGPR classes in `expand()` at
  `lib/rocjitsu/src/rocjitsu/isa/register_set.h:50` through `:60` and
  `lib/rocjitsu/src/rocjitsu/isa/register_set.cpp:37` through `:51`.

Impact:

Execution can update or read SCC, but C++ def-use clients cannot see the scalar
condition-code dependency carried by Chapter 5 SALU producers and SCC-consuming
branches.

### CDNA2-RJ-035: HWREG write permissions, side effects, and spacing are not modeled

Manual/XML evidence:

- Chapter 5.8 marks MODE as read/write, STATUS as read-only, TRAPSTS as
  read/write, HW_ID/GPR_ALLOC/LDS_ALLOC/IB_STS as read-only, and requires an
  `S_NOP` between two consecutive `S_SETREG` writes to the same register at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:1147` through `:1177`.
- The XML exposes `S_GETREG_B32`, `S_SETREG_B32`, and
  `S_SETREG_IMM32_B32` as HWREG access instructions at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:44148` through
  `:44243`, but does not encode those permissions or spacing rules.

Rocjitsu evidence:

- CDNA2 `S_GETREG_B32` returns a few ad hoc values based on `reg_id`, and
  `S_SETREG_B32`/`S_SETREG_IMM32_B32` only handle ID 1 by splicing bits into
  `wf.status_raw()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopk.cpp:291` through
  `:385`.
- The same generated execution paths do not apply read-only rejection, MODE
  versus STATUS write policy, TRAPSTS side effects, IB_STS live counter reads,
  or a consecutive-`S_SETREG` spacing diagnostic. The wrong CDNA2 HWREG ID map
  is tracked separately in `CDNA2-RJ-015`.

Impact:

Even if the ID map is fixed, CDNA2 HWREG execution would still not implement
the manual's per-register access contract or the `S_SETREG` scheduling hazard.

### CDNA2-RJ-036: CDNA2 HWREG operand metadata over-sizes SIMM16 and hides the SETREG literal

Manual/XML evidence:

- Chapter 5.8 says the HWREG operand is a SIMM16 bitfield, and that
  `S_SETREG_IMM32_B32` is a 64-bit instruction whose data comes from a 32-bit
  literal constant at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:1147` through `:1153`.
- The XML records 16-bit `OPR_HWREG` operands and a 32-bit `OPR_SIMM32`
  literal operand for `S_SETREG_IMM32_B32` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:44148` through
  `:44243`.

Rocjitsu evidence:

- CDNA2 generated access constructors build the HWREG operand with size 32,
  not 16, for `S_GETREG_B32`, `S_SETREG_B32`, and `S_SETREG_IMM32_B32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopk.cpp:280` through
  `:284`, `:323` through `:327`, and `:356` through `:359`.
- `S_SETREG_IMM32_B32` reads the extension word into `literal_` through
  `Sopk::hasImpliedLiteral()`, but its constructor exposes only the HWREG
  destination and sets `num_src_ = 0` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/encodings.cpp:68` through
  `:82` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopk.cpp:356`
  through `:363`.
- A newer generated target exposes the same instruction with a 16-bit HWREG
  operand and explicit 32-bit literal source at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/gfx1250/sopk.cpp:224` through
  `:233`.

Impact:

CDNA2 disassembly and analysis see the HWREG field as a 32-bit operand and do
not see the literal source for `S_SETREG_IMM32_B32`, even though execution does
consume the extension word.

### CDNA2-RJ-037: Chapter 5 scalar edge cases are not targeted by tests

Evidence:

- Existing scalar SCC coverage exercises representative arithmetic and bitwise
  SCC behavior, including strict false SCC for equal `S_MAX_I32` and
  `S_MAX_U32` operands at `tests/scalar_scc_test.cpp:720` through `:739`.
- This slice did not find adjacent CDNA2 tests for selector 239 versus reserved
  249-250 behavior, odd 64-bit SGPR pairs, HWREG read/write permissions,
  `S_SETREG` spacing, HWREG operand width, or the explicit literal source on
  `S_SETREG_IMM32_B32`.

Impact:

The existing tests cover common scalar SCC execution, but they can miss the
Chapter 5 selector-map, alignment, HWREG, and operand-metadata gaps above.

### CDNA2-RJ-038: SMEM address calculation ignores `OFFSET` selector and `m0` forms

Manual/XML evidence:

- Chapter 8.2.1 says the `IMM=0,SOFFSET_EN=0` form computes
  `SGPR[base] + (SGPR[offset] or M0)`, and `SOFFSET_EN=1` forms use
  `SGPR[soffset] or M0` at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:1700` through `:1708`.
- The CDNA2 operand selector table includes `OPR_SMEM_OFFSET_M0 = 124` in
  rocjitsu's generated operand metadata at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/operand_types.h:93` through
  `:103`, matching the XML operand class.

Rocjitsu evidence:

- `make_smem_offset()` exposes `OPR_SMEM_OFFSET` only when `soffset_en` is set,
  exposes a signed immediate when `imm` is set, and otherwise builds a constant
  zero operand at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/smem.cpp:28` through `:40`.
- The shared address helper reads `inst.soffset` only when `soffset_en` is set
  and reads `inst.offset` only when `imm` is set; it never reads the `OFFSET`
  selector for the `IMM=0,SOFFSET_EN=0` form at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:32`
  through `:50`.
- For `m0` encoded in `SOFFSET`, the same helper reads
  `wf.sgpr_alloc().base + inst.soffset` as an SGPR index, so selector 124 is
  treated as an allocated SGPR rather than the architectural `M0` register at
  `addr_calc_scalar.h:38` through `:41`.

Impact:

SMEM instructions that use the legal register/M0 offset forms can access the
wrong address: `IMM=0,SOFFSET_EN=0` loses the offset entirely, and `m0` offsets
read an unrelated SGPR slot.

### CDNA2-RJ-039: Scratch SMEM SGPR/M0 offsets are not scaled by 64 bytes

Manual evidence:

- The detailed SMEM table says `S_SCRATCH_LOAD_*` and `S_SCRATCH_STORE_*` use an
  unsigned 64-byte offset when the offset is specified as an SGPR, while the
  immediate form remains a signed byte offset, at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:3209` through `:3222`.

Rocjitsu evidence:

- Generated `S_SCRATCH_LOAD_*` and `S_SCRATCH_STORE_*` bodies use the same
  `smem_calculate_address()` helper as ordinary scalar loads and stores at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/smem.cpp:177` through `:254`
  and `:477` through `:559`.
- That helper adds the register offset value directly to the base and only
  sign-extends the immediate form; it has no scratch-specific 64-byte scaling
  branch at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:32`
  through `:50`.

Impact:

Scratch SMEM instructions with SGPR or `m0` offsets address byte offsets in
rocjitsu instead of the manual's 64-byte scratch units.

### CDNA2-RJ-040: Buffer SMEM ignores resource descriptor fields and bounds

Manual evidence:

- Chapter 8.2.1 says scalar buffer memory uses `base_address`, `stride`,
  `num_records`, and `NV`, uses stride only for bounds checking, ignores other
  descriptor fields, and masks the resource-base and final-address low bits at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:1727` through `:1748`.

Rocjitsu evidence:

- Generated `S_BUFFER_LOAD_*` constructors expose a 128-bit `SBASE` operand, but
  each execute body still calls the generic `smem_calculate_address()` helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/smem.cpp:258` through `:389`.
- Generated `S_BUFFER_STORE_*` follows the same pattern at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/smem.cpp:561` through `:642`.
- The shared helper reads only `sbase` and `sbase+1` as a 64-bit base address
  and never reads descriptor `stride`, `num_records`, or `NV`, nor performs
  buffer bounds checks at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:32`
  through `:50`.

Impact:

Scalar buffer loads/stores can decode and move data, but their address and
range-check behavior does not match CDNA2 resource descriptors. Descriptor
`NV`, record bounds, and stride-as-bounds semantics are ignored.

### CDNA2-RJ-041: SMEM alignment, masking, and out-of-range rules are not modeled

Manual evidence:

- Chapter 8.2.1 says all scalar-memory address components are byte quantities
  but the low two bits are ignored, while `S_DCACHE_DISCARD` ignores six low
  bits at `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:1709`.
- Chapter 8.4 requires even `SDST` for two-dword fetches, multiples of four for
  larger fetches, aligned `SBASE` resource pairs, and no execution when `SDST`
  is out of range at `cdna2/README.md:1780` through `:1792`.

Rocjitsu evidence:

- `smem_calculate_address()` returns `base + off` without low-bit masking or
  the negative-offset undefined case at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:32`
  through `:50`.
- Generated SMEM load constructors assign `sdata` directly from the encoded
  field and set `dst_reg_base = wf.sgpr_alloc().base + inst_.sdata` in execute
  bodies, for example at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/smem.cpp:42` through `:94`.
- `ScalarMemPipeline::complete_access()` writes each returned dword to
  `dst_reg_base + i` with no `SDST` alignment or allocation-range check at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:218` through `:241`.

Impact:

Misaligned addresses and invalid `SDATA`/`SBASE` encodings can produce ordinary
loads/stores in rocjitsu instead of the manual's masked-address, invalid-data,
or not-executed behavior.

### CDNA2-RJ-042: `S_DCACHE_DISCARD` decodes but throws instead of discarding cache lines

Manual evidence:

- The detailed opcode table says `S_DCACHE_DISCARD` discards one dirty 64-byte
  scalar cache line and increments `LGKM_CNT` by one, while
  `S_DCACHE_DISCARD_X2` discards two consecutive lines and increments
  `LGKM_CNT` by two at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:3239` through `:3240`.

Rocjitsu evidence:

- Generated CDNA2 constructors decode `S_DCACHE_DISCARD` and
  `S_DCACHE_DISCARD_X2` operands, but both execute bodies throw
  `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/smem.cpp:746` through `:775`.

Impact:

Kernels using scalar cache-line discard cannot execute under rocjitsu, and the
documented cache-line invalidation plus LGKM contribution is absent.

### CDNA2-RJ-043: `S_ATC_PROBE*` helpers are empty

Manual evidence:

- The detailed SMEM table defines `S_ATC_PROBE` and `S_ATC_PROBE_BUFFER` as
  probe or prefetch operations into the SQC data cache at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:3237` through `:3238`.

Rocjitsu evidence:

- Generated CDNA2 `S_ATC_PROBE` and `S_ATC_PROBE_BUFFER` decode their operands
  and delegate to shared helpers at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/smem.cpp:712` through `:744`.
- The shared helpers are empty functions at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:737`
  through `:742`.

Impact:

ATC probe and prefetch instructions execute as no-ops, so rocjitsu cannot model
their cache-warming or address-probe behavior.

### CDNA2-RJ-044: Scalar and buffer SMEM atomics are unimplemented

Manual evidence:

- Section 8.2.2 says the scalar memory unit supports the same atomics as vector
  memory, uses scalar-memory addressing, and can return the pre-operation value
  to `SDATA` when `GLC` is set at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:1750` through `:1756`.
- The detailed opcode table enumerates scalar and buffer atomic operations and
  their pseudocode at `cdna2/README.md:3241` through `:3317`.

Rocjitsu evidence:

- Generated buffer atomic constructors expose `SDATA`, 128-bit `SBASE`, and
  `SOFFSET`, but representative execute bodies immediately throw
  `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/smem.cpp:778` through `:825`;
  the buffer atomic block continues through `:1270`.
- Generated scalar atomic constructors expose `SDATA`, 64-bit `SBASE`, and
  `SOFFSET`, but representative execute bodies likewise throw from
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/smem.cpp:1272` through
  `:1320`; the scalar atomic block continues through `:1763`.

Impact:

CDNA2 SMEM atomics are part of the decode surface but cannot execute, and the
`GLC` pre-operation return behavior is absent.

### CDNA2-RJ-045: SMEM time reads use synthetic immediate counters

Manual evidence:

- Chapter 8.2.4 says `S_MEMTIME` reads a 64-bit clock counter into `SDST:SDST+1`,
  and Chapter 8.2.5 says `S_MEMREALTIME` reads a constant 25MHz real-time clock
  unaffected by power modes or core clock changes at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:1762` through `:1768`.

Rocjitsu evidence:

- Generated CDNA2 `S_MEMTIME` and `S_MEMREALTIME` constructors expose only
  `SDATA`, do not set `MEMORY_OP`, and call shared helpers directly at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/smem.cpp:687` through `:710`.
- The shared helpers use independent `thread_local` counters that increase by
  100 on each call, then write the pair immediately at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1934`
  through `:1950`.

Impact:

The time instructions provide deterministic synthetic values rather than the
manual's architectural time sources. Their missing scalar-memory wait-counter
behavior is recorded in `CDNA2-RJ-028`.

### CDNA2-RJ-046: SMEM source-overwrite and clause/XNACK hazards are not modeled

Manual evidence:

- Chapter 8.2.1 says a scalar memory instruction must not overwrite its own
  source registers because ATC XNACK replay is possible, and that instructions
  in a scalar memory clause must not overwrite sources of any instruction in the
  clause; a clause is a string of same-type memory instructions broken by a
  nonmemory instruction at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:1723` through `:1725`.

Rocjitsu evidence:

- SMEM load/store execute bodies immediately build a `ScalarMemState` and route
  through the functional memory pipeline; the shared pipeline issue path
  initiates and completes an access with only one outstanding counter callback
  at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.h:55` through `:88`.
- `ScalarMemState` carries address, destination, payload, and wait-counter type
  fields, but no source snapshot, clause identity, or replay metadata at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:70` through `:83`.
- Generated constructors expose ordinary operand lists, but do not add any
  validation or diagnostic for source overwrite across adjacent SMEM
  instructions.

Impact:

Rocjitsu cannot diagnose kernels that violate the CDNA2 scalar-memory replay
contract, and functional execution can appear correct for source-overwrite
patterns that hardware marks unsafe.

### CDNA2-RJ-047: CDNA2 SMEM behavior tests are limited to metadata smoke coverage

Evidence:

- The instruction execution harness intentionally skips memory instructions,
  including `s_load_`, `s_store_`, `s_buffer_`, `s_dcache_`, and `s_atomic_`, at
  `emulation/rocjitsu/tests/instruction_execution_harness_test.cpp:135` through
  `:165`.
- Existing `smem_sbase_operand_test.cpp` checks SBASE operand scaling across
  architectures, including CDNA2, but does not cover address calculation,
  scratch scaling, buffer descriptors, cache/discard/probe behavior, atomics,
  time-counter LGKM behavior, or source-overwrite/clause rules.

Impact:

The current tests can catch selected generated operand metadata regressions, but
would not catch the Chapter 8 execution and address-semantics gaps above.

### CDNA2-RJ-048: MUBUF/MTBUF `SOFFSET` special selectors are read as SGPRs

Manual/XML evidence:

- Chapter 9.1.2 says `SOFFSET` supplies an unsigned byte offset and must be an
  SGPR, `M0`, or inline constant at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:1847` through `:1853`.
- Generated CDNA2 operand metadata exposes buffer `SOFFSET` as `OPR_SSRC_NOLIT`,
  whose selector class includes `M0`, integer inline constants, and floating
  inline constants at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/operand_types.h:378`
  through `:408`.

Rocjitsu evidence:

- Generated MUBUF and MTBUF constructors expose `SOFFSET` as `OPR_SSRC_NOLIT`,
  for example `BUFFER_LOAD_DWORD` and `TBUFFER_LOAD_FORMAT_X` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mubuf.cpp:650` through
  `:662` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mtbuf.cpp:36`
  through `:48`.
- The shared MUBUF address helper special-cases only selector `0x80` as zero,
  and otherwise reads `wf.sgpr_alloc().base + inst.soffset` as an SGPR at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h:57`
  through `:67`.
- The MTBUF helper uses the same selector logic at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h:169`
  through `:178`.

Impact:

Legal buffer offsets using `m0` or inline constants compute addresses from an
unrelated SGPR index instead of the architectural scalar value.

### CDNA2-RJ-049: MUBUF formatted load/store opcodes decode but throw

Manual/XML evidence:

- Chapter 9.1.4 says MUBUF `BUFFER_LOAD_FORMAT_*` and
  `BUFFER_STORE_FORMAT_*` use the resource data format and `dst_sel`, while
  other MUBUF instructions derive format from the opcode at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:1888` through `:1912`.
- The detailed MUBUF table defines formatted load/store and D16 formatted
  opcodes at `cdna2/README.md:4765` through `:4782`, plus D16_HI formatted
  forms at `:4816` through `:4818`.

Rocjitsu evidence:

- Generated `BUFFER_LOAD_FORMAT_{X,XY,XYZ,XYZW}` and
  `BUFFER_STORE_FORMAT_{X,XY,XYZ,XYZW}` constructors expose operands, but their
  execute bodies throw `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mubuf.cpp:36` through
  `:163`.
- The formatted D16 and D16_HI MUBUF variants likewise throw in the generated
  block at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mubuf.cpp:244`
  through `:371` and `:1514` through `:1564`.

Impact:

Untyped raw MUBUF load/store opcodes can execute, but the resource-format MUBUF
surface described by Chapter 9 cannot.

### CDNA2-RJ-050: MTBUF typed format conversion ignores `DFMT` and `NFMT`

Manual/XML evidence:

- Chapter 9.1 says MTBUF data format is specified in the instruction and MTBUF
  only provides load/store operations with data format conversion at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:1816` through `:1823`.
- Chapter 9.1.4 says MTBUF derives data and numeric format from the instruction
  fields at `cdna2/README.md:1888` through `:1904`.
- The MTBUF field map carries `DFMT` and `NFMT` fields at
  `cdna2/README.md:7032` through `:7040`.

Rocjitsu evidence:

- Generated `TBUFFER_LOAD_FORMAT_X` sets `elem_size = 4` and `num_elems = 1`,
  routes through the generic memory pipeline, and does not consult `inst_.dfmt`
  or `inst_.nfmt` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mtbuf.cpp:58` through `:68`.
- Generated `TBUFFER_STORE_FORMAT_X` similarly stores raw 32-bit VGPR payloads
  with fixed `elem_size = 4`, without format conversion, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mtbuf.cpp:198` through
  `:215`.
- D16 MTBUF loads set `elem_size = 2` and the generic D16 half-writeback flag,
  but still do not use `DFMT`/`NFMT` conversion metadata, for example at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mtbuf.cpp:386` through
  `:505`.

Impact:

MTBUF instructions perform raw fixed-width transfers instead of the typed
texture-format conversion required by the instruction's `DFMT`/`NFMT` fields.

### CDNA2-RJ-051: Buffer resource swizzle, add-TID, and descriptor modes are not modeled

Manual evidence:

- Chapter 9.1.5 defines descriptor fields for base, stride, num-records,
  add-TID, swizzle enable, element size, and index stride at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:1936` through `:1949`.
- The same section defines separate range-check behavior for private/scratch,
  raw, and structured buffers at `cdna2/README.md:1968` through `:1992`, and a
  swizzled-address formula at `:1993` through `:2008`.
- Chapter 9.1.8 defines resource descriptor fields including `dst_sel`,
  numeric/data format, user-VM behavior, add-TID, `NV`, type, and all-zero
  unbound behavior at `cdna2/README.md:2054` through `:2084`.

Rocjitsu evidence:

- The shared buffer helper reads `base`, `stride`, `num_records`, and `srd3`,
  but only uses `stride`, `num_records`, and `srd3 >> 31` to choose a simplified
  raw/structured OOB mode at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h:57`
  through `:133`.
- The same helper always computes a linear
  `index * stride + voffset + inst_offset + soffset` style address and does not
  implement the manual's add-TID, stride-extension, swizzled decomposition,
  element-size/index-stride constraints, `dst_sel`, user-VM/unbound, `NV`, or
  descriptor-type rules at `addr_calc_buffer.h:84` through `:133`.
- MTBUF shares the same simplified address and OOB logic at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h:160`
  through `:219`.

Impact:

Common linear raw/structured cases have partial coverage, but descriptor-driven
buffer modes can produce wrong addresses, wrong OOB masks, or ordinary memory
traffic where the manual requires zero/drop/unbound behavior.

### CDNA2-RJ-052: Vector-buffer cache policy collapses `GLC`/`SLC` behavior

Manual evidence:

- Chapter 9.1.2 says `GLC` controls load L1 persistence, store
  write-combine/L1 persistence, and atomic pre-operation return behavior, and
  `SLC` sets L2 streaming/non-temporal mode at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:1860` through `:1866`.
- Section 9.1.10 expands the operation-specific `GLC` behavior at
  `cdna2/README.md:2111` through `:2135`.

Rocjitsu evidence:

- CDNA2 MUBUF/MTBUF execute bodies use `mtype_from_flags_gfx9(inst_.glc)` and
  then set `non_temporal = 0`, for example at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mubuf.cpp:672` through
  `:695` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mtbuf.cpp:58`
  through `:68`.
- `mtype_from_flags_gfx9()` maps only `GLC` to `Mtype::CC` versus `Mtype::RW`
  and explicitly treats `SLC` as not changing simulator `Mtype` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/gfx9_cache_flags.h:7`
  through `:29`.
- The global memory pipeline passes `d.non_temporal` into L1 load/store, but the
  CDNA2 buffer constructors leave it false for all checked MUBUF/MTBUF paths at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:488` through `:495`.

Impact:

Rocjitsu does not model CDNA2's operation-specific L1 persistence,
write-combine, L2 streaming, or cache-bypass distinctions for vector-buffer
memory. Atomic return selection via `GLC` is present separately through
`d->is_load = (inst_.glc != 0)`.

### CDNA2-RJ-053: Buffer cache maintenance and LDS-store opcodes are incomplete

Manual evidence:

- The MUBUF table defines ACK-returning `BUFFER_WBL2`, `BUFFER_INVL2`,
  `BUFFER_WBINVL1`, `BUFFER_WBINVL1_VOL`, and `BUFFER_STORE_LDS_DWORD` at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:4819` through `:4828`.

Rocjitsu evidence:

- `BUFFER_WBL2` flushes L2, but `BUFFER_INVL2` throws
  `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mubuf.cpp:1566` through
  `:1588`.
- `BUFFER_STORE_LDS_DWORD` decodes `SRSRC` and `SOFFSET`, but its execute body
  throws at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mubuf.cpp:1590`
  through `:1604`.
- `BUFFER_WBINVL1` and `BUFFER_WBINVL1_VOL` both invalidate the entire vector
  L1 and flush L2 at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mubuf.cpp:1607`
  through `:1630`, while the manual distinguishes L1-only and volatile-line
  variants.

Impact:

Some buffer cache-maintenance opcodes are unavailable, and implemented L1
writeback/invalidate variants do not preserve the manual's ACK and volatile-line
distinctions. LDS-to-memory MUBUF store cannot execute.

### CDNA2-RJ-054: Chapter 9 vector-buffer tests do not cover most semantic gaps

Evidence:

- `decode_smoke_test.cpp` has a MUBUF `lds` decode/modifier smoke test and a
  CDNA1-style `buffer_load_dword` decode check at
  `emulation/rocjitsu/tests/decode_smoke_test.cpp:726` through `:779`.
- Existing code paths include useful generic memory-pipeline behavior tests, but
  this slice did not find CDNA2-targeted tests for `SOFFSET` `m0`/inline
  selectors, MUBUF formatted opcodes, MTBUF `DFMT`/`NFMT`, descriptor
  add-TID/swizzle/unbound modes, `SLC` cache behavior, `BUFFER_INVL2`,
  `BUFFER_WBINVL1_VOL`, or `BUFFER_STORE_LDS_DWORD`.

Impact:

The present tests can catch some decode regressions and broad memory-pipeline
behavior, but they would not catch the CDNA2 Chapter 9 buffer-specific gaps
above.

### CDNA2-RJ-055: CDNA2 MIMG image instructions decode as no-op execution stubs

Manual/XML evidence:

- Chapter 9.2 says MIMG `IMAGE_*` and `SAMPLE_*` instructions read, write,
  sample, or atomically update image objects through the texture cache at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:2137` through `:2149`.
- Chapter 9.2.1 lists CDNA2 `IMAGE_LOAD*`, `IMAGE_STORE*`,
  `IMAGE_GET_RESINFO`, `IMAGE_ATOMIC_*`, and `IMAGE_SAMPLE` opcodes at
  `cdna2/README.md:2151` through `:2163`; Chapter 13.6 repeats the MIMG opcode
  table at `cdna2/README.md:7223` through `:7252`.
- The XML has corresponding `ENC_MIMG` instruction entries, for example
  `IMAGE_LOAD`, `IMAGE_STORE`, `IMAGE_ATOMIC_SWAP`, and `IMAGE_SAMPLE` at
  `amdgpu_isa_cdna2.xml:20411`, `:20675`, `:20895`, and `:21610`.

Rocjitsu evidence:

- Generated `IMAGE_LOAD*` constructors expose operands, but their execute
  bodies only ignore the wavefront, for example `IMAGE_LOAD` and
  `IMAGE_LOAD_MIP` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mimg.cpp:19` through `:60`.
- Generated `IMAGE_STORE*` forms similarly return from placeholder bodies, for
  example `IMAGE_STORE` through `IMAGE_STORE_MIP_PCK` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mimg.cpp:150` through
  `:236`.
- `IMAGE_GET_RESINFO`, image atomics, and `IMAGE_SAMPLE` are also image-pipeline
  no-op stubs at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mimg.cpp:238` through
  `:278`, `:281` through `:542`, and `:545` through `:565`.
- The code generator explicitly emits image-pipeline stubs for `image_load`,
  `image_store`, `image_atomic`, `image_sample`, and `image_query` classes at
  `lib/python/amdisa/codegen/_generator.py:4241` through `:4256`.

Impact:

Decoded CDNA2 image instructions do not read texture memory, write image memory,
return texel/resource data, perform sampler filtering, apply image atomics, or
produce architectural data/status VGPR results.

### CDNA2-RJ-056: MIMG operand widths are fixed instead of derived from `DMASK`, opcode, and resource shape

Manual evidence:

- Chapter 9.3 defines no-sampler address VGPR layouts by opcode, resource
  dimension, array declaration, mip level, and MSAA/cube fields at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:2187` through `:2215`.
- Chapter 9.4 and 9.4.1 define sampler address component ordering, derivative
  address groups, data VGPR use, read/write `DMASK` behavior, atomic legal
  `DMASK` values, atomic return placement, and D16 packing at
  `cdna2/README.md:2216` through `:2259`.

Rocjitsu evidence:

- Generated no-sampler image load/store constructors use fixed 128-bit `VDATA`
  and `VADDR` operands for representative image loads and stores, for example
  `IMAGE_LOAD` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mimg.cpp:19` through `:33`
  and `IMAGE_STORE` at `:150` through `:165`.
- Generated image atomics expose fixed 128-bit `VDATA` and `VADDR` operands,
  even for forms whose data count is controlled by atomic width and `DMASK`;
  `IMAGE_ATOMIC_SWAP` and `IMAGE_ATOMIC_CMPSWAP` are representative at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mimg.cpp:259` through
  `:319`.
- `IMAGE_SAMPLE` exposes a fixed 96-bit F32 `VADDR` and fixed 128-bit `VDATA`
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mimg.cpp:545` through
  `:561`.

Impact:

Def-use, disassembly, dependency tools, and any future execution path will see
coarse fixed VGPR footprints rather than the image opcode's actual address and
data footprint. That hides `DMASK`, D16, A16, atomic width, and resource
dimension effects.

### CDNA2-RJ-057: MIMG disassembly omits semantic modifier fields

Manual/XML evidence:

- Chapter 9.2.1 lists user-visible MIMG fields including `DMASK`, `UNRM`,
  `GLC`, `SLC`, `DA`, `A16`, `D16`, `LWE`, and `ACC` at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:2164` through `:2185`.
- Chapter 13.6 records the raw MIMG field map, including reserved bit 7,
  `DMASK`, `UNRM`, `GLC`, `DA`, `A16`, `ACC`, `LWE`, `OP`, `SLC`, `SRSRC`,
  `SSAMP`, and `D16`, at `cdna2/README.md:7199` through `:7221`.

Rocjitsu evidence:

- `MimgMachineInst` stores the raw fields, including `scc`, `dmask`, `unorm`,
  `glc`, `da`, `a16`, `acc`, `lwe`, `slc`, `ssamp`, and `d16`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/machine_insts.h:104`
  through `:123`.
- `Mimg` only declares the constructor and stored encoding and does not
  override `build_modifiers` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/encodings.h:585` through
  `:590`; its constructor only records raw encoding metadata at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/encodings.cpp:283` through
  `:289`.
- The base disassembler appends modifiers only through `build_modifiers()`, whose
  default implementation is empty at
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:239` through `:274`.

Impact:

Distinct MIMG encodings that differ in `DMASK`, cache/status/address/data
modifiers, or reserved-bit state can disassemble identically, which obscures
the semantic contract and makes decode-oriented tests too weak.

### CDNA2-RJ-058: Image resource and sampler descriptors are opaque and unused

Manual evidence:

- Chapter 9.2 says each image operation sends a 256-bit resource constant that
  defines address, data format, and surface characteristics, and sample
  operations also send a 128-bit sampler constant at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:2143` through `:2149`.
- Chapter 9.4.2 and 9.4.3 define image resource and sampler descriptor
  bitfields at `cdna2/README.md:2261` through `:2352`.

Rocjitsu evidence:

- Generated MIMG constructors expose `SRSRC` as a fixed 256-bit `OPR_SREG`
  operand, for example `IMAGE_LOAD` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mimg.cpp:19` through `:33`.
- `IMAGE_SAMPLE` exposes both `SRSRC` and `SSAMP`, with `SSAMP` as a fixed
  128-bit `OPR_SREG`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mimg.cpp:545` through
  `:561`.
- The generated image execute bodies are stubs and never read or parse resource
  or sampler descriptor fields, as recorded in `CDNA2-RJ-055`.

Impact:

Rocjitsu cannot model resource dimensions, formats, destination selectors,
tiling, mip/LOD, metadata/compression, PRT behavior, sampler clamp/filter/depth
compare, or force-unnormalized address behavior for CDNA2 image instructions.

### CDNA2-RJ-059: Image instructions do not issue countered VMEM work

Manual evidence:

- Chapter 9.4.5 says VMEM image instructions immediately read address VGPRs and
  image/sampler resources, while write data is sent later, and that consumers
  must use `VMCNT` waits before reading texture-cache results at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:2398` through `:2404`.

Rocjitsu evidence:

- Rocjitsu's memory pipeline increments a wait counter when memory work is
  issued and releases it on completion at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.h:59` through `:80` and
  `:100` through `:105`.
- CDNA2 image execute bodies return from stubs and never create vector-memory
  work or call the global/vector memory pipeline, as shown by representative
  load, store, atomic, query, and sample bodies in `CDNA2-RJ-055`.
- `coverage_exceptions_cdna2.txt` records `image_*` as execution-incomplete
  because image sampling/query is not simulated at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/coverage_exceptions_cdna2.txt:26`
  through `:27`.

Impact:

`S_WAITCNT` can observe no outstanding image work because image instructions do
not issue countered VMEM operations. This misses both texture-result
availability and the manual's address/resource/read-data timing distinction.

### CDNA2-RJ-060: Chapter 9 image tests are decode-only

Evidence:

- The only CDNA2 image-specific test found in this slice is the ACC destination
  selection smoke case for `image_load` at
  `emulation/rocjitsu/tests/decode_smoke_test.cpp:864` through `:895`.
- The codegen coverage exception marks `image_*` execution incomplete at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/coverage_exceptions_cdna2.txt:26`
  through `:27`.
- Searches of `emulation/rocjitsu/tests` and `lib/python/amdisa/tests` found
  semantic derivation/unit tests for image instruction classification, but no
  CDNA2 image execution tests for `DMASK`, D16/A16, resource/sampler
  descriptors, image atomics, cache/status bits, or `VMCNT` behavior.

Impact:

The suite can catch selected MIMG decode regressions, but it would not catch the
execution, descriptor, data-format, dependency, or disassembly gaps above.

### CDNA2-RJ-061: Buffer floating atomics miss L2 FP numeric and packed-lane rules

Manual evidence:

- Chapter 9.5 says floating memory atomics execute in LDS and L2 and can be
  issued as Buffer, Flat, Global, Scratch, LDS, and GDS instructions at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:2406` through `:2410`.
- Float atomic ADD opcodes use RNE, and Table 43 defines L2 denormal behavior:
  `PK_ADD_F16` and `ADD_F64` never flush denorms while `ADD_F32` always flushes
  denorms at `cdna2/README.md:2412` through `:2440`.
- Chapter 9.5.3 defines SNaN quieting, NaN propagation, signed-zero ordering,
  min/max selection, compare-store equality, and add edge cases at
  `cdna2/README.md:2449` through `:2492`.
- The MUBUF opcode table includes `BUFFER_ATOMIC_ADD_F32`,
  `BUFFER_ATOMIC_PK_ADD_F16`, `BUFFER_ATOMIC_ADD_F64`,
  `BUFFER_ATOMIC_MIN_F64`, and `BUFFER_ATOMIC_MAX_F64` at
  `cdna2/README.md:7172` through `:7176`; the global table shows packed F16 as
  two independent 16-bit lane additions at `cdna2/README.md:5171` through
  `:5172`.

Rocjitsu evidence:

- Generated CDNA2 MUBUF floating atomics lower to the generic memory-pipeline
  `AtomicOp::FADD`, `FMIN`, and `FMAX` operations. Representative F32, packed
  F16, F64 add, and F64 min/max paths are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mubuf.cpp:2233` through
  `:2450`.
- `BUFFER_ATOMIC_PK_ADD_F16` sets `elem_size = 4` and
  `AtomicOp::FADD`, the same scalar 32-bit operation used by
  `BUFFER_ATOMIC_ADD_F32`, at `mubuf.cpp:2302` through `:2322`.
- The semantic derivation table also labels packed FP atomics as "treated as
  32-bit fadd for now" at `lib/python/amdisa/semantics.py:1828` through
  `:1856`.
- The shared L2 atomic executor treats 4-byte floating atomics as one host
  `float` and 8-byte floating atomics as one host `double`, using ordinary
  addition plus `std::fmin`/`std::fmax` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:285` through `:363`.

Impact:

F32/F64 buffer atomics have a coarse functional model, but rocjitsu does not
enforce CDNA2's L2 floating-atomic denormal, RNE, NaN, or signed-zero rules.
Packed F16 buffer atomics are not type-correct because their two lanes execute
as one scalar F32 operation.

### CDNA2-RJ-062: CDNA2 global float atomics are not generated as global opcodes

Manual/XML evidence:

- Chapter 9.5 says floating memory atomics can be issued as Global instructions
  at `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:2406` through `:2408`.
- The Flat/Global opcode inventory lists `GLOBAL_ATOMIC_ADD_F32` and
  `GLOBAL_ATOMIC_PK_ADD_F16`, with no matching flat forms, at
  `cdna2/README.md:2540` through `:2558`.
- The detailed GLOBAL table lists `GLOBAL_ATOMIC_ADD_F32`,
  `GLOBAL_ATOMIC_PK_ADD_F16`, `GLOBAL_ATOMIC_ADD_F64`,
  `GLOBAL_ATOMIC_MIN_F64`, and `GLOBAL_ATOMIC_MAX_F64` at
  `cdna2/README.md:5171` through `:5175`.
- The checked-in XML has matching `GLOBAL_ATOMIC_*` entries at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:18146` through
  `:18395`.

Rocjitsu evidence:

- The CDNA2 build only includes `flat.cpp`; it has no `vglobal.cpp` or
  generated global-specific source file at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/CMakeLists.txt:4` through
  `:24`.
- The generated CDNA2 opcode constants expose only `kFlatAtomic*` names for the
  overlapping flat/global opcode space at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/opcodes.h:2272` through
  `:2300`.
- The flat subdecoder marks opcode slots 77 and 78 invalid, even though those
  are the manual/XML `GLOBAL_ATOMIC_ADD_F32` and `GLOBAL_ATOMIC_PK_ADD_F16`
  opcodes, and maps slots 79-81 to `FlatAtomic*F64` classes at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/decoder.cpp:8798` through
  `:8815`.
- The generated test-encoding fixture lists only `flat_atomic_add_f64`,
  `flat_atomic_min_f64`, and `flat_atomic_max_f64`, not the CDNA2 global
  float-atomic mnemonics, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/test_encodings.h:1346`
  through `:1357`.

Impact:

CDNA2 `GLOBAL_ATOMIC_ADD_F32` and `GLOBAL_ATOMIC_PK_ADD_F16` encodings cannot
decode through the generated flat subdecoder. F64 global forms can reach the
flat F64 classes through the shared opcode slots, but their generated class and
opcode-constant surface remains flat-only rather than carrying distinct
manual/XML `GLOBAL_ATOMIC_*` entries.

### CDNA2-RJ-063: Flat F64 float atomics miss return-mode and FP-mode rules

Manual evidence:

- Chapter 10.3.1 says float atomics must set `GLC=0`, with no return value, and
  that FP32 flushes denorms while FP64/FP16 never flush denorms with fixed RNE
  rounding at `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:2596` through
  `:2602`.
- The detailed FLAT table lists `FLAT_ATOMIC_ADD_F64`,
  `FLAT_ATOMIC_MIN_F64`, and `FLAT_ATOMIC_MAX_F64` at
  `cdna2/README.md:5060` through `:5062`.
- Chapter 9.5.3 defines the FP min/max and add edge cases at
  `cdna2/README.md:2449` through `:2492`.

Rocjitsu evidence:

- Generated CDNA2 `FlatAtomicAddF64Flat`, `FlatAtomicMinF64Flat`, and
  `FlatAtomicMaxF64Flat` set `d->is_load = (inst_.glc != 0)`, allowing
  return-data mode for FP atomics, and lower to generic `AtomicOp::FADD`,
  `FMIN`, and `FMAX` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/flat.cpp:1780` through
  `:1948`.
- The shared L2 atomic executor uses host `double` addition plus
  `std::fmin`/`std::fmax`, with no per-instruction return-mode, denormal, RNE,
  NaN, or signed-zero policy at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:285` through `:363`.

Impact:

CDNA2 flat F64 float atomics can execute return mode that Chapter 10 forbids,
and their numeric behavior follows host floating-point helpers rather than the
manual's CDNA2 L2 float-atomic edge rules.

### CDNA2-RJ-064: LDS float atomics use generic host FP and integer compare-swap

Manual evidence:

- Chapter 9.5 says LDS float atomics use `MODE.denorm` controls for F16/F32
  add, F32/F64 min/max, and compare-store, while F64 add never flushes denorms,
  at `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:2416` through `:2447`.
- Chapter 9.5.3 defines LDS NaN behavior for `DS_ADD_F32`, FP min/max SNaN and
  signed-zero ordering, and compare-store equality where `+0` and `-0` compare
  equal and NaNs do not swap at `cdna2/README.md:2449` through `:2484`.

Rocjitsu evidence:

- Generated CDNA2 DS F32 min/max/add lower to generic `AtomicOp::FMIN`,
  `FMAX`, and `FADD` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/ds.cpp:783` through `:900`;
  F64 min/max/add use the same generic operations at `ds.cpp:3150` through
  `:3222` and `:3492` through `:3522`.
- `DS_CMPST_F32` and `DS_CMPST_F64` lower to generic `AtomicOp::CMPSWAP` at
  `ds.cpp:736` through `:780` and `:3098` through `:3147`.
- `AtomicOp` has no separate FP compare-store or denormal-policy variant at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:44` through `:68`.
- The LDS atomic executor treats `FADD`/`FMIN`/`FMAX` as host `float`/`double`
  operations and routes `CMPSWAP` through integer bit-equality at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:247` through
  `:293` and `:383` through `:468`.

Impact:

LDS F32/F64 add/min/max do not apply CDNA2's `MODE.denorm`, NaN, or signed-zero
selection rules. FP compare-store uses raw integer equality, so `+0` and `-0`
do not compare equal and equal NaN bit patterns can swap even though the manual
forbids NaN compare-store swaps.

### CDNA2-RJ-065: Float-atomic edge tests are absent for CDNA2

Evidence:

- Existing atomic stress tests cover integer LDS/global atomic add behavior in
  `tests/amdgpu_vm_test.cpp:1877` through `:2036`.
- Existing float-atomic execution tests found in this slice target GFX1250
  `global_atomic_add_f32` ordinary add/return behavior at
  `tests/instruction_execution_harness_test.cpp:3423` through `:3505`.
- Searches of `emulation/rocjitsu/tests` and `lib/python/amdisa/tests` found
  semantic/codegen tests for atomic classification and payload width, but no
  CDNA2 execution tests for packed-F16 buffer/global atomics, F32/F64 denormal
  modes, SNaN quieting, `+0`/`-0` min/max ordering, FP compare-store equality,
  or the Chapter 10 float-atomic no-return rule.

Impact:

The current test surface can catch broad integer atomic and newer-architecture
global F32 regressions, but it would not catch the CDNA2 float-atomic decode,
packed-lane, return-mode, denormal, NaN, signed-zero, or FP compare-store gaps
above.

### CDNA2-RJ-066: Flat aperture routing and wait-counter accounting collapse the dual-path contract

Manual evidence:

- Chapter 10.2 says Flat instructions internally execute as both LDS and Buffer
  requests and increment both `VM_CNT` and `LGKM_CNT`; completion requires both
  counters, and the only sensible wait after Flat is `S_WAITCNT 0` at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:2566` through `:2578`.
- Chapter 10.4 says Global instructions use only `VM_CNT` and that a Global
  instruction attempting LDS access returns a memory violation at
  `cdna2/README.md:2621`.

Rocjitsu evidence:

- Every generated CDNA2 flat load/store/atomic execute path sets
  `d->wait_counter_type = WaitCounterType::VMCNT`; representative load code is
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/flat.cpp:220` through
  `:228`, and the same assignment appears throughout the file.
- `ComputeUnitCore::route_memory_inst()` samples only the first active lane's
  address to decide whether a `GLOBAL_MEM` instruction targets the shared
  aperture, rewrites every active lane as LDS when that one probe hits, changes
  the tag to `LOCAL_MEM`, and switches the wait counter to `LGKMCNT` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:260` through `:282`.
- `MemoryPipeline::issue()` increments and later releases exactly one wait
  counter chosen from the state at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.h:62` through `:88`.

Impact:

Flat requests are modeled as either VM or LDS, not the manual's dual VM/LGKM
request. Mixed-lane flat accesses can be routed incorrectly because only the
first active lane selects the path, and Global instructions that land in the
shared aperture are routed to LDS instead of producing the manual's MEM_VIOL.

### CDNA2-RJ-067: Scratch address calculation gates the VGPR offset on `LDS`

Manual evidence:

- Table 44 says Scratch uses `SADDR` to select an SGPR address instead of the
  VGPR address, while `M0` is only implied when `LDS=1` for direct LDS movement
  at `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:2510` through `:2511`.
- Section 10.5 says Scratch can take its offset from either an SGPR or VGPR,
  and that the offset is 32-bit unsigned at `cdna2/README.md:2629` through
  `:2634`.

Rocjitsu evidence:

- Generated CDNA2 scratch constructors expose a 32-bit VGPR address operand and
  optionally an SGPR offset operand when `inst_.seg == 1`, as shown for
  `FlatLoadDwordFlat` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/flat.cpp:206` through
  `:211`.
- The shared flat address helper sets `has_vaddr = (inst.lds == 1)` for
  scratch encodings that do not have an `SVE` field, and only reads the VGPR
  offset when `has_vaddr` is true at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h:59`
  through `:89`.

Impact:

Ordinary CDNA2 scratch forms with `LDS=0` can ignore the VGPR offset even though
the constructor and manual describe it as the address operand. The `LDS` bit is
a direct-LDS data-movement control, not the Scratch VGPR-offset enable in the
CDNA2 manual.

### CDNA2-RJ-068: Global SGPR-base addressing sign-extends the VGPR offset

Manual evidence:

- Table 44 says the Global `SADDR` form uses an SGPR base address and a
  32-bit VGPR offset at `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:2510`.
- Section 10.4 defines the Global address form as SGPR address plus VGPR offset
  plus instruction offset, and says the VGPR offset is 32 bits at
  `cdna2/README.md:2608` through `:2617`.
- Chapter 13.7 describes the single-VGPR `ADDR` offset as unsigned and the
  `SADDR` value as providing an unsigned address or offset at
  `cdna2/README.md:7279` and `:7286`.

Rocjitsu evidence:

- The shared flat address helper handles Global `saddr != 0x7f` by reading one
  VGPR dword and sign-extending it through `int32_t` before adding it to the
  SGPR base at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h:90`
  through `:111`.

Impact:

Global SGPR-base accesses whose VGPR offset has bit 31 set are interpreted as
negative offsets. The manual describes a 32-bit offset and assigns signedness to
the 13-bit instruction `OFFSET`, so this can send high unsigned VGPR offsets to
the wrong address.

### CDNA2-RJ-069: Global/scratch direct-LDS forms are not generated for CDNA2 flat memory

Manual evidence:

- Chapter 10.1 says `LDS` enables direct data movement between LDS and memory
  for Global and Scratch instructions and must be zero for Flat instructions at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:2502` through `:2504`.
- Table 44 says `M0` supplies the LDS offset for Scratch/Global when `LDS=1`
  at `cdna2/README.md:2511`.
- Sections 10.4 and 10.5 repeat that Global and Scratch can move data directly
  between LDS and memory at `cdna2/README.md:2619` and `:2636`.

Rocjitsu evidence:

- Representative generated CDNA2 flat load/store code uses VGPR destinations or
  VGPR data operands and constructs a `VectorMemState(GLOBAL_MEM)`, for example
  `FlatLoadDwordFlat` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/flat.cpp:192` through `:228`.
- Searches of generated CDNA2 `flat.cpp` and `flat.h` found no
  `lds_dst`, `lds_per_lane_addr`, `global_load_async_to_lds`, or
  `global_store_async_from_lds` path.

Impact:

CDNA2 Global/Scratch encodings with `LDS=1` do not get the manual's
M0-relative direct LDS transfer semantics. They execute through the ordinary
VGPR memory path, so tests cannot currently exercise the direct-LDS contract for
Chapter 10 flat-memory instructions.

### CDNA2-RJ-070: Memory-violation and flat-memory trap behavior is absent

Manual evidence:

- Chapter 10.6 says invalid addresses, read-only writes, misalignment, and
  out-of-range LDS or scratch accesses report memory errors; writes are
  dropped, reads return zero, and the wave MEM_VIOL trap-status bit is set at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:2640` through `:2657`.
- Chapter 10.4 says Global instructions attempting LDS access return MEM_VIOL
  at `cdna2/README.md:2621`, while Chapter 10.5 says Scratch performs no
  aperture checking and no error reporting at `:2638`.

Rocjitsu evidence:

- The shared flat address helper maps private-aperture Flat addresses into the
  scratch backing buffer but has no scratch-size or memory-error check at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h:113`
  through `:132`.
- `ComputeUnitCore::route_memory_inst()` maps shared-aperture `GLOBAL_MEM`
  requests into LDS instead of producing a trap or zeroed read at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:260` through `:282`.
- Searches of `lib/rocjitsu/src/rocjitsu` found no `MEM_VIOL`, `TrapStatus`,
  `TRAPSTS`, `EXCPEN`, or `MEMORY_VIOL` implementation hooks.

Impact:

Negative flat/global/scratch memory cases execute as ordinary memory traffic
instead of applying CDNA2's defined drop/zero/trap-status behavior. This also
makes Global-to-LDS and out-of-range LDS/scratch cases untestable against the
manual contract.

### CDNA2-RJ-071: Chapter 10 tests miss flat-memory routing and direct-LDS edge cases

Evidence:

- CDNA2 decode smoke tests include `global_load_dword` and
  `scratch_load_dword` accumulator-destination decode coverage at
  `tests/decode_smoke_test.cpp:864` through `:893`.
- Existing atomic stress tests cover integer LDS atomics and global atomic add
  through the L2 path at `tests/amdgpu_vm_test.cpp:1877` through `:2036`.
- Searches of `emulation/rocjitsu/tests` found no CDNA2 execution tests for
  Flat dual `VM_CNT`/`LGKM_CNT` completion, mixed-lane flat aperture routing,
  Global-to-LDS MEM_VIOL, Scratch `LDS=0` VGPR offsets, high-bit Global VGPR
  offsets, Scratch/Global direct-LDS `LDS=1` forms, or Chapter 10 memory-error
  read-zero/drop/trap behavior.

Impact:

The current test surface verifies some segment-specific decode and broad global
atomic execution, but it would not catch the Chapter 10 behavioral gaps above.

### CDNA2-RJ-072: CDNA2 VALU source operands cannot select LDS direct reads

Manual/XML evidence:

- CDNA2 section 6.2 says only `SRC0` can use `LDS_DIRECT`, and section 11.3.1
  says the `SRC0=LDS_DIRECT` selector reads a single LDS value using the address
  and data type encoded in `M0[18:0]` at `cdna2/README.md:1271` and
  `:2716` through `:2731`.
- The checked-in CDNA2 XML has no `LDS_DIRECT` operand value; this is tracked
  separately as `CDNA2-XML-038`.

Rocjitsu evidence:

- CDNA2 generated operand selector enums have no `OPR_SRC_SRC_LDS_DIRECT` or
  `OPR_SRC_NOLDS_SRC_LDS_DIRECT` value; `OPR_SRC` skips from SCC selector 253
  to literal selector 255, and `OPR_SRC_NOLDS` likewise omits selector 254 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/operand_types.h:118`
  through `:143` and `:151` through `:188`.
- Representative CDNA2 VOP2 constructors use `OPR_SRC_NOLDS` for source 0 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop2.cpp:2162`,
  `:2290`, and `:2418`, and searches found no CDNA2 execution path that reads
  an LDS direct value from `M0` for a VALU source.

Impact:

CDNA2 encodings that use the manual-defined LDS direct source selector cannot
be named or executed by rocjitsu. They decode as unsupported/unknown selector
values instead of broadcasting the M0-selected LDS byte/short/dword source.

### CDNA2-RJ-073: Indexed and atomic LDS address calculation ignores M0 clamp/size

Manual/XML evidence:

- Chapter 11 says all LDS operations require `M0` to be initialized and that
  `M0` can restrict accesses to a subset of allocated LDS; `0xffffffff`
  disables clamping at `cdna2/README.md:2760`.
- Single-address DS operations are based on `LDS_BASE + VGPR[ADDR] +
  {InstrOffset1,InstrOffset0}` and atomics use their own concatenated offset
  form at `cdna2/README.md:2777` through `:2781` and `:2801` through `:2809`.

Rocjitsu evidence:

- The shared DS address helper computes only `VGPR[ADDR] + offset +
  wf.lds_base()` and never reads `wf.m0()` or applies an M0 clamp at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:60`
  through `:76`.
- Generated CDNA2 DS load/store/atomic execute paths call that helper for
  ordinary indexed operations, for example `DS_ADD_U32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/ds.cpp:42` through `:52`.

Impact:

Out-of-range LDS accesses under a restrictive M0 execute as ordinary LDS
traffic, so kernels or negative tests that depend on CDNA2's M0-bounded LDS
range are modeled incorrectly.

### CDNA2-RJ-074: READ2/WRITE2 duplicate offsets perform two accesses instead of one

Manual evidence:

- Chapter 11 says a READ2/WRITE2 instruction can specify only one address by
  setting both offsets to the same value, causing only one read or write to
  occur and using only `DATA0` at `cdna2/README.md:2799`.

Rocjitsu evidence:

- Generated `DS_WRITE2_B32` unconditionally sets `ds2_active = true`, computes
  both per-lane addresses, reads both `DATA0` and `DATA1`, and fills both store
  buffers at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/ds.cpp:601`
  through `:630`. The same generated pattern is used by other READ2/WRITE2 and
  ST64 variants.
- The LDS memory pipeline unconditionally issues the second load or store when
  `ds2_active` is set at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:520` through `:528`
  and `:590` through `:595`, and writes second-load results to the second
  destination at `:606` through `:617`.

Impact:

Equal-offset READ2/WRITE2 forms can read twice, write twice, and let `DATA1`
overwrite the single intended address. Hardware-compatible single-address
cases and duplicate-offset regressions are therefore not represented.

### CDNA2-RJ-075: ADDTID DS address formulas do not use `M0.OFFSET + TID*4`

Manual evidence:

- The detailed DS opcode table defines `DS_WRITE_ADDTID_B32` as
  `MEM[ADDR_BASE + OFFSET + M0.OFFSET + TID*4] = DATA` at
  `cdna2/README.md:4464`.
- `DS_READ_ADDTID_B32` returns
  `MEM[ADDR_BASE + OFFSET + M0.OFFSET + TID*4]` at
  `cdna2/README.md:4626`.

Rocjitsu evidence:

- `DsWriteAddtidB32Ds::execute_impl()` calls the generic
  `ds_calculate_addresses()` helper, which uses an encoded VGPR `ADDR` field
  and does not add `M0.OFFSET` or `TID*4`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/ds.cpp:922` through `:940`.
- `DsReadAddtidB32Ds::execute_impl()` does have bespoke address code, but it
  derives `ds_stride_bytes = ((M0 >> 16) & 0x1ff) * 4` and computes
  `lane * ds_stride_bytes + offset + lds_base`, never adding the manual's low
  M0 offset or fixed `TID*4` stride at `ds.cpp:4840` through `:4865`.

Impact:

ADDTID reads and writes address the wrong LDS locations. With a default or
low-offset-only `M0`, rocjitsu can collapse all lanes to the same address or
ignore the required per-lane dword stride.

### CDNA2-RJ-076: DS `ACC` is applied to source data operands even though the DS field scopes it to `VDST`

Manual/XML evidence:

- The CDNA2 DS field table says `ACC` means "`VDST` is Accumulation VGPR" and
  defines `DATA0`/`DATA1` as ordinary data VGPR fields at
  `cdna2/README.md:6852` through `:6865`.
- The checked-in XML field description agrees that `ACC` controls `VDST`, while
  individual instruction entries type `DATA0` as `OPR_VGPR_OR_ACCVGPR`; this
  ambiguity is tracked as `CDNA2-XML-041`.

Rocjitsu evidence:

- Generated CDNA2 DS store constructors apply the `acc` bit to data source
  operands. For example `DS_WRITE2_B32` adds
  `OPR_VGPR_OR_ACCVGPR_ACC_MIN` to both `DATA0` and `DATA1` when `inst_.acc`
  is set at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/ds.cpp:580`
  through `:592`, and the execute body reads from `vgpr_base + 256 + dataN`
  at `:616` through `:628`.
- `DS_WRITE_ADDTID_B32` applies the same source-ACC offset to `DATA0` at
  `ds.cpp:908` through `:915` and `:931` through `:939`.

Impact:

Until the source-ACC form is hardware-confirmed, rocjitsu may read AccVGPR
source data for DS store/atomic forms that the manual describes as ordinary
VGPR data operands. This can hide illegal encodings or mis-execute programs
that set the reserved/VDST-only `ACC` bit on source-only DS operations.

### CDNA2-RJ-077: GWS DS instructions remain nonsimulated and restrictions are not enforced

Manual evidence:

- Section 11.4 says every GWS instruction must be immediately followed by
  `s_waitcnt 0`, and VGPRs used by any GWS instruction must be even at
  `cdna2/README.md:2817` through `:2825`.
- The detailed DS table defines GWS semaphore/barrier state-machine behavior at
  `cdna2/README.md:4590` through `:4625`.

Rocjitsu evidence:

- The semantics classifier explicitly maps `DS_GWS_*` instructions to `nop`
  because GWS is not simulated at `lib/python/amdisa/semantics.py:2393`
  through `:2401`.
- Generated support code throws `util::UnimplementedInst` for GDS/GWS-like DS
  paths when the `gds` field is set, including atomic, append/consume, mskor,
  and barrier-arrive classes at
  `lib/python/amdisa/codegen/_generator.py:4083` through `:4103`.

Impact:

This is a known unsupported hardware-scheduling primitive rather than a silent
numeric bug, but the CDNA2 GWS contract is still not executable or validated:
mandatory following waits, even-VGPR operands, and semaphore/barrier behavior
are not modeled.

### CDNA2-RJ-078: Chapter 11 tests miss LDS direct, M0 clamp, duplicate-offset, and ADDTID cases

Evidence:

- Existing tests cover basic DS store/load routing, representative LDS atomics,
  and `ds_read_b64_tr_b16` AccVGPR destination behavior at
  `tests/amdgpu_vm_test.cpp:1219` through `:1269`, `:1880` through `:1966`,
  and `:2039` through `:2089`.
- Existing instruction-harness tests cover RDNA4/GFX1250 DS swizzle decode and
  execution at `tests/instruction_execution_harness_test.cpp:2968` through
  `:3411`.
- Searches of `emulation/rocjitsu/tests` found no CDNA2 execution tests for
  `SRC0=LDS_DIRECT`, restrictive `M0` LDS clamps, equal-offset READ2/WRITE2
  collapse, ADDTID `M0.OFFSET + TID*4` addressing, source-only DS `ACC`
  rejection/behavior, or GWS wait/even-VGPR restrictions.

Impact:

The current suite exercises useful DS basics, but it would not catch the
Chapter 11 behavioral gaps above or prevent regressions in the manual's
selector/addressing corner cases.

### CDNA2-RJ-079: FP min/max helpers do not model NaN, signed-zero, or `IEEE_MODE` tie rules

Manual evidence:

- `V_MIN_F32` and `V_MAX_F32` define signaling-NaN quieting under `IEEE_MODE`,
  NaN operand selection, signed-zero tie selection, and `V_MAX_F32`
  IEEE-mode equality behavior at `cdna2/README.md:3342` through `:3343`.
- `V_MAX_F16` and `V_MIN_F16` repeat the NaN and signed-zero selection rules
  for half precision at `cdna2/README.md:3401` through `:3402`.

Rocjitsu evidence:

- Shared F16 min/max helpers convert through F32 and use
  `util::stdx::fmax`/`fmin` in SIMD or host `std::fmax`/`std::fmin` in the
  scalar path at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:13811`
  through `:13874` and `:15118` through `:15180`.
- Shared F32 helpers likewise use `util::stdx::fmax`/`fmin` or host
  `std::fmax`/`std::fmin` at `execute_shared.h:13877` through `:13934` and
  `:15184` through `:15241`.
- The SIMD correctness test explicitly excludes NaN-input lanes and signed-zero
  tie lanes from comparison, because scalar and SIMD paths may diverge there,
  at `tests/simd_correctness/vop2_minmax_simd_correctness_test.cpp:4`
  through `:19`.

Impact:

Finite non-tie cases are covered, but rocjitsu does not pin or emulate the
ISA's specified result selection for NaNs, signaling NaNs, signed-zero ties, or
the `IEEE_MODE`-dependent equality case.

### CDNA2-RJ-080: `V_READFIRSTLANE_B32` returns zero instead of lane 0 when `EXEC` is disabled

Manual evidence:

- `V_READFIRSTLANE_B32` says `Lane# = FindFirst1fromLSB(exec)`, with
  `Lane# = 0 if exec is zero`, and ignores the `EXEC` mask for the VGPR access
  at `cdna2/README.md:3448`.

Rocjitsu evidence:

- The generated VOP1 body initializes `val = 0`, searches only active lanes,
  and writes the unchanged zero when `EXEC` has no active bits at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop1.cpp:324` through
  `:332`.
- The VOP3 alias repeats the same active-lane-only loop at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop3.cpp:54` through
  `:64`.
- Existing shared-infra coverage exercises an ordinary active-lane
  `v_readfirstlane_b32_e32` case, not the all-disabled `EXEC` path, at
  `tests/shared_infra_test.cpp:3483` through `:3489`.

Impact:

All-disabled waves can scalarize zero instead of the documented lane-0 VGPR
value.

### CDNA2-RJ-081: `V_SAT_PK_U8_I16` decodes but throws in both VOP1 and VOP3 forms

Manual evidence:

- `V_SAT_PK_U8_I16` saturates both signed 16-bit source halves to unsigned
  8-bit results and zero-extends the high 16 destination bits at
  `cdna2/README.md:3691`.

Rocjitsu evidence:

- Semantic derivation maps `V_SAT_PK_U8_I16` to `nop` at
  `lib/python/amdisa/semantics.py:726` through `:733`.
- The generated VOP1 executor throws `UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop1.cpp:9559` through
  `:9562`.
- The generated VOP3 executor also throws at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop3.cpp:1824` through
  `:1827`.

Impact:

Kernels using this documented packed saturation instruction cannot execute
through rocjitsu despite generated decode and operand metadata being present.

### CDNA2-RJ-082: VOP1/VOP3 exception-state semantics are no-ops

Manual evidence:

- `V_CLREXCP` clears the wave's exception state in SIMD at
  `cdna2/README.md:3546`.
- VOP3A compare promotion says setting `CLAMP` makes compare instructions
  signal an exception when either input is NaN, while `CLAMP=0` avoids the NaN
  signal at `cdna2/README.md:4081` through `:4087`.

Rocjitsu evidence:

- Semantic derivation maps `V_CLREXCP` to `true_nop` at
  `lib/python/amdisa/semantics.py:726` through `:733`.
- Shared `execute_v_clrexcp_vop1()` and `execute_v_clrexcp_vop3()` are empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:3795`
  through `:3799`.
- Representative generated VOP3 compare bodies compute and write the lane mask
  but do not consult `inst_.clamp` or update exception state; `VCmpxOF16Vop3`
  is representative at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop3.cpp:7293` through
  `:7320`.

Impact:

rocjitsu can execute compare result dataflow, but exception-state clearing and
the VOP3A compare `CLAMP` exception-signaling contract are not observable.

### CDNA2-RJ-083: Chapter 12.7-12.9 tests miss VALU definition edge contracts

Evidence:

- The min/max SIMD correctness test explicitly excludes NaN-input and
  signed-zero-tie lanes from result comparison at
  `tests/simd_correctness/vop2_minmax_simd_correctness_test.cpp:4` through
  `:19`.
- Existing `V_READFIRSTLANE_B32` coverage uses an active-lane case at
  `tests/shared_infra_test.cpp:3483` through `:3489`, and searches found no
  all-disabled `EXEC` regression.
- Existing VOPC class tests cover class-mask dataflow, including scalar and
  SIMD compare-class suites, but not the VOP3A `CLAMP` exception-signaling
  contract.
- Searches found no execution test for `V_SAT_PK_U8_I16` or observable
  `V_CLREXCP` exception-state clearing.

Impact:

Full-suite passing does not currently protect the manual's min/max edge
selection, readfirstlane disabled-wave behavior, packed saturation execution, or
exception-state semantics.

### CDNA2-RJ-084: `V_DIV_SCALE_F32/F64` zero numerator or denominator cases return `S0` instead of NaN

Manual/XML evidence:

- CDNA2 `V_DIV_SCALE_F32` and `V_DIV_SCALE_F64` set `D` to NaN when either the
  numerator or denominator is zero, then use later divide-fixup steps to handle
  the final special case at `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:4300`
  through `:4305`.
- The XML identifies these as VOP3B operations that scale the first input and
  set a vector condition-code mask at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:61338` through
  `:61430`, but does not carry the detailed zero-case pseudocode.

Rocjitsu evidence:

- `execute_v_div_scale_f32_vop3()` initializes `result = s0`, then leaves that
  result unchanged when `s2 == 0.0f || s1 == 0.0f` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:10312`
  through `:10317`.
- `execute_v_div_scale_f64_vop3()` has the same pass-through zero-case behavior
  for double precision at `execute_shared.h:10371` through `:10375`.

Impact:

Zero numerator or denominator inputs produce the unscaled input operand rather
than the manual's NaN pre-scale result, so the state entering the later
`V_DIV_FIXUP_*` sequence can diverge from hardware.

### CDNA2-RJ-085: Native F16 VOP3A writes use the legacy low-half zeroing rule

Manual/XML evidence:

- The legacy F16 rows say `op_sel[3] == 0` writes the low destination half and
  writes the high half as zero, while `op_sel[3] == 1` preserves the low half at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:4317` through `:4322`.
- The non-legacy `V_MAD_F16`, `V_MAD_U16`, `V_MAD_I16`, `V_FMA_F16`, and
  `V_DIV_FIXUP_F16` rows instead say both low and high destination writes
  preserve the untouched half at `cdna2/README.md:4351` through `:4360`.
- CDNA3 and CDNA4 repeat the same split: legacy rows zero the high half on low
  writes, while non-legacy native F16 rows preserve the untouched half at
  `workspace_docs/amdgpu-isa-manuals/cdna3/README.md:15013` through `:15074`
  and `:15375` through `:15460`, and
  `workspace_docs/amdgpu-isa-manuals/cdna4/README.md:16187` through `:16233`
  and `:16434` through `:16522`.

Rocjitsu evidence:

- `write_vop3_true16_dst()` zeroes the high half for CDNA low-half writes when
  its `cdna_low_dst_zeroes_high` argument is true at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/simd_glue.h:285` through
  `:317`.
- Generated CDNA2 non-legacy bodies pass that argument as true for
  `V_MAD_F16`, `V_MAD_U16`, `V_MAD_I16`, `V_FMA_F16`, and `V_DIV_FIXUP_F16` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop3.cpp:5148`, `:5183`,
  `:5220`, `:5296`, and `:5383`.
- The current SIMD correctness test pins the zero-high result for both
  `v_div_fixup_f16_vop3` and `v_div_fixup_legacy_f16_vop3`, including the
  non-legacy opcode, at `tests/simd_correctness/vop3_div_helpers_simd_correctness_test.cpp:526`
  through `:540`.

Impact:

Low-half native F16 VOP3A writes can clobber the high half even though the
manual says those rows preserve it. Existing tests currently protect the
implementation's behavior rather than the manual distinction.

### CDNA2-RJ-086: `V_PERM_B32` implements an incomplete selector table

Manual/XML evidence:

- `V_PERM_B32` selector bytes `0..7` select source bytes, selector 12 returns
  zero, selectors 8 through 11 sign-fill from source bytes 1, 3, 5, and 7, and
  selectors `>= 13` return `0xff` at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:4323`.
- XML describes the broad ability to choose bytes, sign-extend, or pad with 0/1
  bits at `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:61983` through
  `:62014`, but does not structure the selector table.

Rocjitsu evidence:

- The SIMD helper documents and implements only selectors `0..7`, `0xC`, and
  `0xD`; every other selector yields zero at
  `lib/util/include/util/simd.h:1118` through `:1136`.
- The scalar fallback in `execute_v_perm_b32_vop3()` likewise returns source
  bytes for selectors `0..7`, zero for `0xC`, `0xff` only for `0xD`, and zero
  otherwise at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:16628`
  through `:16633`.

Impact:

Selectors 8 through 11 miss the sign-extension cases, and selectors 14 through
255 produce zero instead of `0xff`. SIMD and scalar paths agree with each
other, so scalar-vs-SIMD tests cannot catch the hardware mismatch.

### CDNA2-RJ-087: `V_CVT_PKRTZ_F16_F32` uses round-to-nearest-even helpers

Manual/XML evidence:

- The manual says `V_CVT_PKRTZ_F16_F32` rounds toward zero regardless of the
  current hardware round mode at `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:4393`.
- XML repeats the round-toward-zero rule at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:64107` through `:64108`.

Rocjitsu evidence:

- `execute_v_cvt_pkrtz_f16_f32_vop3()` calls `util::f32_to_f16_simd()` in the
  SIMD path and `util::f32_to_f16()` in the scalar path at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:9827`
  through `:9840`.
- `util::f32_to_f16()` performs round-to-nearest-even by using guard/sticky
  bits and LSB tie logic at `lib/util/include/util/data_types.h:112` through
  `:125`.
- `util::f32_to_f16_simd()` is documented as a bit-exact port of that helper
  and comments its denormal path as round-to-nearest-even at
  `lib/util/include/util/simd.h:470` through `:497`.
- A separate `util::f32_to_f16_rtz()` helper exists at
  `lib/util/include/util/data_types.h:135` through `:159`, but this VOP3 helper
  does not use it.

Impact:

Inputs that are not exactly representable in FP16 can round to the nearest-even
half instead of truncating toward zero.

### CDNA2-RJ-088: Packed F32-input VOP3A conversions ignore source modifiers

Manual/XML evidence:

- The VOP3A field table provides `ABS`, `OPSEL`, `CLMP`, `OMOD`, and `NEG`
  fields at `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:6394` through
  `:6409`.
- `V_READLANE_B32` and `V_WRITELANE_B32` explicitly say modifiers are not
  supported at `cdna2/README.md:4374` through `:4379`; the packed F32-input
  conversion rows do not carry a comparable no-modifier exception at
  `cdna2/README.md:4266`, `:4329`, and `:4391` through `:4393`.
- XML models the conversion rows as VOP3A operands, including
  `V_CVT_PK_U8_F32` at `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:61197`,
  `V_CVT_PKACCUM_U8_F32` at `:62124`, and
  `V_CVT_PKNORM_I16_F32`, `V_CVT_PKNORM_U16_F32`, and
  `V_CVT_PKRTZ_F16_F32` at `:64025` through `:64135`.

Rocjitsu evidence:

- `V_CVT_PK_U8_F32`, `V_CVT_PKACCUM_U8_F32`,
  `V_CVT_PKNORM_I16_F32`, `V_CVT_PKNORM_U16_F32`, and
  `V_CVT_PKRTZ_F16_F32` read raw source bits and do not apply `inst_.abs` or
  `inst_.neg` in their scalar paths at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:9719`
  through `:9840`.
- Their SIMD probes use `ROCJITSU_TRY_SIMD_VOP3_BINARY_INT`, whose comment says
  it reads `src0`/`src1` with no modifiers at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/simd_glue.h:3899` through
  `:3903`.
- The SIMD code generator separately documents that the plain integer VOP3
  glue does not apply modifier fields at
  `lib/python/amdisa/codegen/execute/simd_codegen.py:2843` through `:2850`.

Impact:

VOP3A encodings with ABS/NEG modifier bits set on these conversion sources can
execute as if the bits were clear.

### CDNA2-RJ-089: `V_TRIG_PREOP_F64` decodes but throws

Manual/XML evidence:

- `V_TRIG_PREOP_F64` defines a `2/pi` segment lookup, exponent-dependent shift
  and scale, RTZ behavior, and large-input scaling at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:4389`.
- XML exposes the VOP3 opcode and operands at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:63943` through `:63969`.

Rocjitsu evidence:

- The generated CDNA2 `VTrigPreopF64Vop3::execute_impl()` body throws
  `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop3.cpp:5681` through
  `:5683`.

Impact:

Any CDNA2 shader using the double-precision trig range-reduction helper decodes
but cannot execute in rocjitsu.

### CDNA2-RJ-090: VOP3A F64 min/max use host `std::fmin`/`std::fmax`

Manual/XML evidence:

- `V_MIN_F64` and `V_MAX_F64` define signaling-NaN quieting under `IEEE_MODE`,
  NaN operand selection, signed-zero tie selection, and an `IEEE_MODE`
  equality branch for `V_MAX_F64` at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:4367` through `:4368`.
- XML provides only generic min/max descriptions at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:63328` through
  `:63394`.

Rocjitsu evidence:

- `execute_v_max_f64_vop3()` uses `util::stdx::fmax` in SIMD and host
  `std::fmax` in the scalar body at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:13937`
  through `:13976`.
- `execute_v_min_f64_vop3()` uses `util::stdx::fmin` in SIMD and host
  `std::fmin` in the scalar body at `execute_shared.h:15244` through `:15283`.

Impact:

F64 NaN, signaling-NaN, signed-zero, and `IEEE_MODE` equality cases follow host
library behavior instead of the manual's explicit selection rules.

### CDNA2-RJ-091: Chapter 12.11 tests miss VOP3A/VOP3B definition edge contracts

Manual evidence:

- The Chapter 12.11 slice includes manual-only edge contracts for VOP3B
  divide-scale zero behavior, native versus legacy F16 OPSEL destination
  writeback, the full `V_PERM_B32` selector table, `V_CVT_PKRTZ_F16_F32` RTZ
  rounding, packed conversion modifiers, F64 min/max, and `V_TRIG_PREOP_F64`.

Rocjitsu evidence:

- Existing SIMD tests include generic entries for `v_perm_b32_vop3`, F64
  min/max, and F16/div-fixup paths, but the visible test coverage compares
  rocjitsu execution paths to each other or excludes/manual-pins edge behavior
  rather than checking the manual contracts. For example, the
  `v_div_fixup_f16_vop3` test pins low-destination high-half zeroing for the
  non-legacy opcode at
  `tests/simd_correctness/vop3_div_helpers_simd_correctness_test.cpp:526`
  through `:540`.
- Searches found no targeted golden tests for `V_DIV_SCALE_F32/F64` zero
  numerator/denominator NaN output, `V_PERM_B32` selectors 8 through 11 and
  14+, `V_CVT_PKRTZ_F16_F32` RTZ tie cases, packed-conversion source modifiers,
  or `V_TRIG_PREOP_F64` execution.

Impact:

Full-suite passing does not currently protect the Chapter 12.11 edge semantics
identified above.

### CDNA2-RJ-092: VGPR indexing uses the wrong M0 bits and cannot select individual sources

Manual/XML evidence:

- CDNA2 Chapter 6.6 says `S_SET_GPR_IDX_ON` and `S_SET_GPR_IDX_MODE` store the
  mode in `M0[15:12]`, and assigns those four bits to destination, src2, src1,
  and src0 enables at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:1458` through `:1472`.
- The same section says indexing applies only to VGPR operands, makes
  out-of-range indexed VGPR access illegal, and gives special remaps for
  `v_readlane`, `v_writelane`, `v_mac_*`, `v_madak`, `v_madmk`, reverse shifts,
  `v_cvt_pkaccum`, and SDWA at `cdna2/README.md:1474` through `:1492`.
- XML exposes the control instructions but leaves the mode layout and remap
  semantics prose-only, as recorded in `CDNA2-XML-054`.

Rocjitsu evidence:

- `Wavefront::gpr_idx_mode()` reads `(m0_ >> 8) & 0xF` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:248` through `:251`, while
  the manual says the mode lives in `M0[15:12]`.
- Generated `S_SET_GPR_IDX_MODE` and `S_SET_GPR_IDX_ON` write `(mode << 8)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2500`
  through `:2519`.
- `apply_gpr_idx()` only receives `bool is_dst` and applies any source-mode bit
  to every source operand at `wavefront.h:584` through `:588`.
- CDNA2 operand reads and writes call that helper with only source/destination
  polarity at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/operand.cpp:1340`
  through `:1388`, and the SIMD operand path does the same at
  `lib/rocjitsu/src/rocjitsu/isa/isa_operand_simd_inl.h:47` through `:87`.

Impact:

Architectural `M0[15:12]` settings do not drive indexing in rocjitsu, generated
indexing-control instructions produce non-architectural M0 contents, and
instructions requiring separate src0/src1/src2 or special destination-source
remaps can index the wrong VGPR operands.

### CDNA2-RJ-093: Ordinary VALU ignores MODE-driven rounding, denormal, and OMOD suppression rules

Manual/XML evidence:

- Chapter 6.2.2 says VOP3 OMOD is ignored for integer/bit results, ignored when
  output denormals are enabled, flushes denormals and `-0` when applied, and is
  ignored when `IEEE_MODE` is set at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:1295` through `:1297`.
- Chapter 6.4 defines `MODE.FP_ROUND` and `MODE.FP_DENORM` controls for
  single-precision and double/half-precision floating-point VALU behavior at
  `cdna2/README.md:1431` through `:1442`.
- XML exposes `CLAMP` and `OMOD` fields but leaves those MODE interactions
  prose-only, as recorded in `CDNA2-XML-055`.

Rocjitsu evidence:

- The VOP3 modifier generator emits unconditional OMOD and CLAMP code for F32
  and F64 results at
  `lib/python/amdisa/codegen/execute/vop3_modifiers.py:37` through `:53`.
- Representative generated VALU helpers apply input ABS/NEG, then OMOD, then
  CLAMP without reading `wf.mode_raw()`; for example
  `execute_v_maxmin_num_f32_vop3()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:14523`
  through `:14572`.
- Static search found no ordinary VALU execution path that consumes
  `FP_ROUND`, `FP_DENORM`, or `IEEE_MODE`; nearby tests pin unconditional
  VOP3 modifier codegen rather than MODE-dependent behavior.

Impact:

Rounding-sensitive and denormal-sensitive FP VALU cases, output-denormal
suppression of OMOD, IEEE-mode suppression of OMOD, and OMOD's `-0` to `+0`
behavior can diverge from CDNA2 hardware.

### CDNA2-RJ-094: `S_ABSDIFF_I32` uses mathematical absolute difference instead of the wrapped SOP2 definition

Manual/XML evidence:

- Chapter 12.1 defines `S_ABSDIFF_I32` as `D.i = S0.i - S1.i`, followed by
  `D.i = -D.i` only when that 32-bit signed result is negative, and gives edge
  examples such as `S_ABSDIFF_I32(0x80000000, 0x00000001) => 0x7fffffff` at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:2932`.
- The XML only summarizes the operation as "absolute value of difference" at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:38067` through
  `:38068`; the wrapped-subtract edge examples are recorded as
  `CDNA2-XML-056`.

Rocjitsu evidence:

- Generated `SAbsdiffI32Sop2::execute_impl()` delegates to the shared helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sop2.cpp:1007` through
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

### CDNA2-RJ-095: Chapter 12.1 SOP2 tests miss detailed semantic edge contracts

Evidence:

- The generated CDNA2 encoding fixture includes a single `s_absdiff_i32`
  encoding smoke entry at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/test_encodings.h:189`.
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

### CDNA2-RJ-096: `S_CALL_B64` direct calls are flagged as `INDIRECT_CALL`

Manual/XML evidence:

- Chapter 12.2 defines `S_CALL_B64` as `D.u64 = PC + 4` and
  `PC = PC + signext(SIMM16 * 4) + 4`, and says it is always a 4-byte short
  call at `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:2980`.
- The XML describes the target as a constant offset relative to the current PC
  and marks `IsIndirectBranch` false at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:42440` through
  `:42480`.

Rocjitsu evidence:

- CDNA2 `SCallB64Sopk` exposes a PC-relative `branch_offset_bytes()` and
  executes by writing `PC + size_` to `SDST` and adding the signed SIMM16
  instruction-count offset to `wf.pc` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopk.cpp:387` through
  `:407`.
- The same constructor sets `flags_ |= INDIRECT_CALL` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopk.cpp:395`, even though
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

### CDNA2-RJ-097: 64-bit scalar relative moves multiply the M0 offset by two

Manual/XML evidence:

- Chapter 12.3 defines `S_MOVRELS_B64` as `addr = SGPR address appearing in
  instruction SRC0 field; addr += M0.u; D.u64 = SGPR[addr].u64`, and says the
  index in M0 must be even at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:3073`.
- `S_MOVRELD_B64` uses the same `addr += M0.u` rule for the instruction
  destination field and also requires an even M0 value at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:3075`.
- The XML exposes M0 as an implicit operand on `S_MOVRELS_B64` and
  `S_MOVRELD_B64` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:31629` through
  `:31805`, while the detailed formula omission is recorded in
  `CDNA2-XML-057`.

Rocjitsu evidence:

- CDNA2 `SMovrelsB64Sop1::execute_impl()` reads `index = wf.m0() & 0xFFu`,
  computes `width_words = ssrc0.size_bits() / 32`, and uses
  `src_reg = ssrc0.encoding_value() + index * width_words`; for the 64-bit form
  this is `base + 2 * M0` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sop1.cpp:844` through
  `:848`.
- `SMovreldB64Sop1::execute_impl()` applies the same scaled formula to the
  destination register at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sop1.cpp:902` through
  `:906`.
- Neither path checks the manual's even-M0 requirement before forming the
  64-bit SGPR pair.

Impact:

For a legal even M0 value such as 2, `s_movrels_b64 s[10:11], s[2:3]` should
read from `s[4:5]`, but rocjitsu reads from `s[6:7]`. Odd M0 values are also
silently remapped by the doubled formula instead of being rejected or diagnosed
as invalid 64-bit relative moves.

### CDNA2-RJ-098: `S_SET_GPR_IDX_ON` treats the raw SIMM4 field as a literal source

Manual/XML evidence:

- CDNA2 Chapter 12.4 says `S_SET_GPR_IDX_ON` writes `M0[15:12] = SIMM4`, and
  clarifies that this is the direct raw content of the S1 field; the raw S1
  bits select `VSRC0_REL`, `VSRC1_REL`, `VSRC2_REL`, and `VDST_REL` at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:3124`.
- XML's default `S_SET_GPR_IDX_ON` row records the second operand as
  `OPR_SIMM4`, but the generic literal variants for this row are imprecise as
  recorded in `CDNA2-XML-058`.

Rocjitsu evidence:

- Generated CDNA2 `SSetGprIdxOnSopc` initially constructs `ssrc1` as
  `OperandType::OPR_SIMM4` from the raw `ssrc1` field at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopc.cpp:411` through
  `:415`.
- The same constructor rewrites `ssrc1` to `OPR_SIMM32` when the raw field is
  255 at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopc.cpp:424`
  through `:427`.
- The generic CDNA2 SOPC encoding marks any instruction with either source
  field equal to 255 as having an extension literal at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/encodings.cpp:44`
  through `:54`.
- `execute_s_set_gpr_idx_on_sopc()` then reads `inst.ssrc1` through
  `RegisterAccess` and masks the low four bits at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2514`
  through `:2519`. The separate M0 bit-position bug is tracked in
  `CDNA2-RJ-092`.

Impact:

An encoding with raw S1 field `0xff` should use the raw low four bits as the
SIMM4 mode mask, but rocjitsu consumes the following extension word and uses the
literal's low four bits instead. Decode and disassembly can therefore assign an
extension literal to a direct field that the manual defines as raw instruction
bits.

### CDNA2-RJ-099: XML-only `S_TTRACEDATA` drops M0 data and has empty execution

Manual/XML evidence:

- The CDNA2 manual does not list SOPP opcode 22 in the detailed Chapter 12.5
  table or Chapter 13.1.5 opcode table, as recorded in `CDNA2-XML-059`.
- The checked-in CDNA2 XML nevertheless defines `S_TTRACEDATA` as `ENC_SOPP`
  opcode 22, describes it as sending M0 as user data to the thread-trace
  stream, and gives it an implicit 32-bit M0 input at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:43143` through
  `:43155`.

Rocjitsu evidence:

- The CDNA2 decoder maps SOPP opcode 22 to `decodeSTtracedataSopp` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/decoder.cpp:6918` through
  `:6939`.
- Generated `STtracedataSopp` exposes zero source and destination operands at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopp.cpp:352` through
  `:357`.
- `STtracedataSopp::execute_impl()` dispatches to the shared helper at
  `cdna2/sopp.cpp:359` through `:360`, and that helper body is empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2662`
  through `:2665`.

Impact:

An XML-described instruction decodes and disassembles, but rocjitsu neither
surfaces nor consumes the M0 payload and does not emit any thread-trace side
effect. Manual-based tools may reject the same opcode as absent, while
rocjitsu treats it as an executable no-op.

### CDNA2-RJ-100: SOPP ordered-PS, instruction-cache, and perf-level side effects are stubs

Manual/XML evidence:

- `S_ENDPGM_ORDERED_PS_DONE` is a combined
  `S_SENDMSG(MSG_ORDERED_PS_DONE)` plus `S_ENDPGM` with an implicit
  `S_WAITCNT 0` at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:3179`.
- `S_ICACHE_INV` invalidates the L1 instruction cache and requires 16 following
  `S_NOP` instructions or a jump/branch to purge the SQ instruction buffer at
  `cdna2/README.md:3164`.
- `S_INCPERFLEVEL` and `S_DECPERFLEVEL` increment or decrement the performance
  counter selected by `SIMM16[3:0]` at `cdna2/README.md:3165` through `:3166`.
- XML carries short entries for these SOPP rows at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:43063` through
  `:43133` and `:43370` through `:43377`.

Rocjitsu evidence:

- Generated `SEndpgmOrderedPsDoneSopp::execute_impl()` only calls `wf.end()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopp.cpp:449` through
  `:457`.
- Generated `SIcacheInvSopp`, `SIncperflevelSopp`, and `SDecperflevelSopp`
  dispatch to shared helpers at `cdna2/sopp.cpp:315` through `:350`.
- The shared `execute_s_icache_inv_sopp()` helper is empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1770`
  through `:1773`.
- The shared `execute_s_incperflevel_sopp()` and
  `execute_s_decperflevel_sopp()` helpers are empty at `execute_shared.h:1775`
  through `:1777` and `:1636` through `:1638`.

Impact:

rocjitsu can decode and execute these SOPP instructions without throwing, but
ordered-PS completion, instruction-cache invalidation/purge behavior, and
performance-level counter updates have no observable effect. The generic
message/status stubs are tracked in `CDNA2-RJ-025`; this entry records the
additional detailed Chapter 12.5 SOPP side effects.

### CDNA2-RJ-101: `S_TRAP` metadata is marked as a program terminator instead of an indirect branch

Manual/XML evidence:

- The detailed `S_TRAP` row says the instruction waits for all prior
  instructions, saves trap payload state in TTMPs, sets `PC = TBA`, sets
  `PRIV = 1`, and enters the trap handler at
  `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:3163`.
- XML marks `S_TRAP` as a branch and indirect branch, and explicitly does not
  mark it as a program terminator, at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:43027` through
  `:43031`.

Rocjitsu evidence:

- Generated `STrapSopp` constructs the SIMM16 operand but sets only
  `PROGRAM_TERMINATOR` metadata at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopp.cpp:301` through
  `:307`.
- The execution path throws `util::UnimplementedInst` at `cdna2/sopp.cpp:310`
  through `:312`; the trap-state execution gap is already tracked in
  `CDNA2-RJ-019`.

Impact:

Even before trap execution is implemented, static branch analysis and DBT or
instrumentation clients can classify `S_TRAP` as terminating the program rather
than as an indirect control transfer to the trap handler.

### CDNA2-RJ-102: `DS_SWIZZLE_B32` treats rotate and FFT offsets as basic quad swizzles

Manual/XML evidence:

- The detailed `DS_SWIZZLE_B32` row says the instruction writes no LDS memory and
  points to the next section for details at `cdna2/README.md:4503`.
- Section 12.12.1 says invalid source threads produce zero, defines FFT mode for
  offsets `>= 0xe000`, rotate mode for offsets `>= 0xc000` and `< 0xe000`,
  and then defines the two basic group-of-4/group-of-32 modes at
  `cdna2/README.md:4635` through `:4736`.
- XML exposes only the opcode and short description, so the manual-derived
  selector details remain tracked as `CDNA2-XML-042`.

Rocjitsu evidence:

- The generated CDNA2 instruction body delegates to
  `execute_ds_swizzle_b32_ds()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/ds.cpp:2279` through
  `:2295`.
- The shared generator emits only two cases for `ds_swizzle`: `if (offset &
  0x8000)` it interprets the bits as packed quad selectors, otherwise it uses
  the 32-lane and/or/xor mapping, at
  `lib/python/amdisa/codegen/_generator.py:4194` through `:4238`. There is no
  `offset >= 0xe000` FFT path and no `0xc000 <= offset < 0xe000` rotate path.
- Adjacent tests found by static search cover only broadcast/quad-style forms
  for `ds_swizzle_b32` at `tests/instruction_execution_harness_test.cpp:3364`
  through `:3411`; no regression exercises the rotate or FFT offset ranges.

Impact:

Any CDNA2 `DS_SWIZZLE_B32` using the rotate or FFT ranges has bit 15 set and is
therefore executed as a basic quad selector in rocjitsu. That produces the wrong
source lane mapping while still looking like a supported instruction.

### CDNA2-RJ-103: `DS_APPEND` and `DS_CONSUME` return per-lane ranks instead of broadcasting the pre-op counter

Manual/XML evidence:

- The CDNA2 detailed rows say `DS_CONSUME` subtracts `count_bits(exec_mask)` and
  `DS_APPEND` adds `count_bits(exec_mask)`, and both return the pre-operation
  value to VGPRs at `cdna2/README.md:4628` through `:4629`.
- The CDNA2 XML repeats the same pre-operation-value return wording for
  `DS_CONSUME` and `DS_APPEND` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:13659` through
  `:13700`.
- RDNA4's manual prose states the same contract more explicitly: DS returns the
  pre-op value to all valid lanes and broadcasts the single atomic result at
  `workspace_docs/amdgpu-isa-manuals/rdna4/README.md:25987` through `:26020`.

Rocjitsu evidence:

- Generated CDNA2 `DsConsumeDs` and `DsAppendDs` route to shared LDS atomics via
  `AtomicOp::CONSUME` and `AtomicOp::APPEND` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/ds.cpp:4881` through
  `:4941`.
- The shared LDS atomic RMW path reads the counter once and updates memory by
  the active-lane count, but writes per-lane response values as
  `old_val + active_rank` for append and `old_val - active_rank - 1` for
  consume at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:393` through
  `:419`.
- Static search found no `ds_append` or `ds_consume` execution regression; only
  semantic-classification tests mention these opcodes.

Impact:

Active lanes see distinct ranked values instead of the single pre-operation
counter value. For example, with an old counter of 10 and two active lanes,
manual-compatible append returns 10 to both lanes and stores 12, while rocjitsu
returns 10 and 11. Consume similarly returns decremented lane ranks instead of
the pre-op counter value.

### CDNA2-RJ-104: Flat decoder accepts reserved `SEG` and scratch atomic opcode combinations

Manual/XML evidence:

- Chapter 12.16 defines Scratch instructions as load/store plus D16 load rows
  only, with opcodes 16 through 37 at `cdna2/README.md:5080` through `:5115`.
  The Chapter 13.7 Scratch opcode table repeats only opcodes 16 through 37 at
  `cdna2/README.md:7422` through `:7454`.
- The Chapter 10 flat-format summary says `SEG` value 3 is reserved at
  `cdna2/README.md:2502`, and the Chapter 13.7 field table defines only
  `SEG` values 0 = Flat, 1 = Scratch, and 2 = Global at
  `cdna2/README.md:7271` through `:7278`.
- The checked-in XML has distinct `ENC_FLAT`, `ENC_FLAT_GLBL`, and
  `ENC_FLAT_SCRATCH` encodings. `ENC_FLAT_SCRATCH` carries only 22 encoding
  identifiers at `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:4019`
  through `:4045`, and the scratch instruction entries run from
  `SCRATCH_LOAD_UBYTE` through `SCRATCH_LOAD_SHORT_D16_HI` at
  `amdgpu_isa_cdna2.xml:19100` through `:20196`.

Rocjitsu evidence:

- CDNA2 `Decoder::subDecodeFlat()` reinterprets every FLAT-format word as
  `Flat::OpEncoding` and indexes `sub_decode_flat` by `op` only at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/decoder.cpp:4973` through
  `:4975`.
- The shared flat opcode table maps atomic opcodes 64 through 76, 79 through
  81, and 96 through 108 to `FlatAtomic*` constructors regardless of `SEG` at
  `decoder.cpp:8798` through `:8842`.
- The generated `Flat` constructor rewrites any `flat_*` mnemonic to
  `scratch_*` when `SEG == 1` and to `global_*` when `SEG == 2` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/encodings.cpp:14` through
  `:21` and `:291` through `:294`.
- The shared address helper treats only `SEG == 1` as Scratch and `SEG == 2` as
  Global; any other segment, including reserved `SEG == 3`, falls into the Flat
  path at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h:59`
  through `:132`.

Impact:

Encodings with `SEG=1` and atomic opcodes decode as unsupported
`scratch_atomic_*` instructions even though Scratch has no atomic opcode rows in
the manual or XML. Encodings with reserved `SEG=3` are accepted and executed as
Flat-like operations instead of being rejected as an invalid encoding.

### CDNA2-RJ-105: Global and scratch signed instruction offsets disassemble as unsigned raw values

Manual/XML evidence:

- Chapter 13.7 defines `OFFSET` as a 12-bit unsigned offset for FLAT but a
  13-bit signed byte offset for Scratch and Global at
  `cdna2/README.md:7271`.
- The XML repeats the segment-specific signedness: `ENC_FLAT_GLBL` describes
  `OFFSET` as a 13-bit signed offset at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:3945` through `:3951`,
  and `ENC_FLAT_SCRATCH` repeats the same at `:4110` through `:4117`.

Rocjitsu evidence:

- The shared flat address helper sign-extends the 13-bit Scratch/Global offset
  for execution and keeps Flat as unsigned at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h:46`
  through `:57`.
- CDNA2 `Flat::build_modifiers()` reconstructs the Scratch/Global offset as a
  raw 13-bit integer and prints it with `std::to_string(flat_offset)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/encodings.cpp:301` through
  `:306`; it does not sign-extend the display value.
- The adjacent CDNA2 decode smoke coverage exercises `global_load_dword` and
  `scratch_load_dword` only with zero offset at `tests/decode_smoke_test.cpp:864`
  through `:893`.

Impact:

Execution can use the correct signed address offset while disassembly still
prints the wrong modifier. A Global or Scratch encoding with raw `OFFSET=0x1000`
should disassemble with `offset:-4096`, but rocjitsu prints `offset:4096`,
which breaks decoder-oracle fuzzing and any tool that round-trips
segment-specific flat-memory offsets.

### CDNA2-RJ-106: Flat-memory cache and bit-25 controls are only partially surfaced

Manual/XML evidence:

- Chapter 10 says `SLC` is used with `GLC` to determine flat-memory cache
  policies, and it also names `NV` as a non-volatile memory control at
  `cdna2/README.md:2502`.
- The Chapter 12.16 field legend names `GLC`, `SLC`, and `NV` for
  FLAT/GLOBAL/SCRATCH definitions at `cdna2/README.md:4995` through `:5005`.
- The detailed Chapter 13.7 field table says `GLC` is globally coherent,
  `SLC` bypasses L2, and bit 25 is reserved-zero at `cdna2/README.md:7274`
  through `:7277`. The bit-25 manual/XML drift is tracked separately as
  `CDNA2-XML-035`.
- The XML still exposes bit 25 for `ENC_FLAT`, `ENC_FLAT_GLBL`, and
  `ENC_FLAT_SCRATCH` as `SCC`, described as system-level cache coherent, at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna2.xml:3779` through `:3787`,
  `:3975` through `:3983`, and `:4140` through `:4148`.

Rocjitsu evidence:

- Generated CDNA2 flat machine-instruction structs store `glc`, `slc`, and the
  bit-25 `scc` field for flat, global, and scratch encodings at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/machine_insts.h:126`
  through `:172`.
- CDNA2 flat execution maps only `inst_.glc` into `mtype` and sets
  `d->non_temporal = 0` in every generated flat/global/scratch operation, for
  example `FlatLoadUbyteFlat` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/flat.cpp:56` through `:63`
  and `FlatAtomicSwapFlat` at `:1059` through `:1068`.
- The shared global-memory pipeline does consume `d.non_temporal` for loads and
  stores at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:488`
  through `:494`, so flat `SLC` never reaches that modeled cache-policy path.
- `Flat::build_modifiers()` prints `glc` and `slc`, but it never reports bit 25
  as either a reserved-bit violation or a cache/non-volatile modifier at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/encodings.cpp:301`
  through `:310`.

Impact:

Flat/global/scratch encodings that differ in `SLC` or bit 25 collapse to the
same runtime behavior except for `SLC` text in disassembly. Depending on which
manual field summary is treated as authoritative, bit 25 should either be
rejected as reserved-zero or modeled/surfaced as the non-volatile/system-cache
control; rocjitsu currently accepts it and drops it.

### CDNA2-RJ-107: DPP/SDWA forms are accepted for Chapter 12.17 excluded instructions

Manual/XML evidence:

- Chapter 12.17.1 lists instructions and wildcard compare families that cannot
  use DPP at `workspace_docs/amdgpu-isa-manuals/cdna2/README.md:5197` through
  `:5229`.
- Chapter 12.17.2 lists instructions that cannot use SDWA at
  `cdna2/README.md:5231` through `:5244`.
- Chapter 13.3.7 and 13.3.9 say SDWA/DPP occupy the second dword in place of a
  literal constant at `cdna2/README.md:6725` through `:6731` and `:6791`
  through `:6797`.
- The checked-in XML incorrectly exposes DPP encodings for 11 listed F64 VOP1
  rows, as recorded in `CDNA2-XML-064`. For the other DPP exclusions and the
  named SDWA exclusions, XML generally represents the limitation by omitting
  the corresponding DPP/SDWA instruction encodings.

Rocjitsu evidence:

- CDNA2 `Decoder::subDecodeVop1()` dispatches by `op` only, with no
  instruction-specific DPP/SDWA legality check, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/decoder.cpp:1071`
  through `:1074`. The VOP1 table maps the excluded F64 opcodes to ordinary
  constructors at `:6329` through `:6376`.
- The generated base `Vop1`, `Vop2`, and `Vopc` classes all treat source
  markers 250 and 249 as non-default extension forms at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/encodings.cpp:145` through
  `:171` and `:173` through `:201`.
- Generated VOP1 constructors accept DPP/SDWA markers even for rows that
  Chapter 12.17 excludes. `VReadfirstlaneB32Vop1` reads `Vop1VopDppMachineInst`
  and `Vop1VopSdwaMachineInst` fields at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop1.cpp:230` through
  `:260`; `VRcpF64Vop1` does the same at `:4553` through `:4585`; and
  `VSwapB32Vop1` does the same at `:9564` through `:9600`.
- The zero-source `VClrexcpVop1` constructor has no DPP/SDWA operands, but its
  generated execute body still enters generic DPP/SDWA paths when `src0` is
  250 or 249 at `vop1.cpp:6633` through `:6655`.
- Generated VOP2 constructors also accept excluded SDWA forms: `VMacF32Vop2`
  handles `SRC_SDWA` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop2.cpp:2925` through
  `:2958`, `VMadmkF32Vop2` does the same at `:3066` through `:3105`, and
  `VFmacF32Vop2` does the same at `:8004` through `:8045`.
- The `MADMK`/`MADAK` case additionally collides with implied-literal handling:
  generated `Vop2` always increases instruction size and reads word 1 as a
  literal when `hasImpliedLiteral()` is true at `encodings.cpp:173` through
  `:182`, while `VMadmkF32Vop2` can also reinterpret the same second dword as
  DPP/SDWA extension fields when `SRC0` is 250 or 249.
- Generated VOPC constructors accept DPP/SDWA markers for excluded 64-bit and
  class compare families, for example `VCmpClassF64Vopc` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vopc.cpp:287` through
  `:328`, `VCmpEqF64Vopc` at `:8819` through `:8860`, `VCmpxEqF64Vopc` at
  `:10686` through `:10727`, and `VCmpEqI64Vopc` at `:20449` through `:20488`.
- CDNA2 VOP1/VOP2/VOPC classes store DPP/SDWA fields but do not override
  `build_modifiers()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/encodings.h:459` through
  `:470`; disassembly is built from operands and `build_modifiers()` at
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:239` through `:244`, so accepted
  illegal suffix details are silently normalized away after constructors replace
  marker `SRC0` with the actual VGPR operand.
- Static search found no CDNA2 test asserting that Chapter 12.17 illegal
  DPP/SDWA encodings reject or disassemble differently; adjacent DPP/SDWA
  tests exercise legal dataflow only. The Python profile/parser also filters
  non-default DPP/SDWA instruction rows before generated test encodings at
  `lib/python/amdisa/isa_profile.py:758` through `:762` and
  `lib/python/amdisa/parser.py:875` through `:884`, and CDNA2 coverage
  exceptions explicitly skip `*_dpp` and `*_sdwa` rows at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/coverage_exceptions_cdna2.txt:29`
  through `:31`.

Impact:

rocjitsu can decode and execute DPP/SDWA forms that the CDNA2 manual marks
illegal, including cases where XML omits the corresponding encoding. Some of
those accepted encodings also disassemble without the suffix that made them
illegal. Decoder-oracle fuzzing will therefore accept invalid encodings unless
it layers the Chapter 12.17 exclusion list on top of generated metadata,
constructor behavior, and disassembly.

### CDNA2-RJ-108: CDNA2 `DPP_CTRL` 0x160-0x165 is treated as row-xmask

Manual/XML evidence:

- CDNA2 Table 83 defines `DPP_UNUSED` 0x100 as undefined/reserved and defines
  `DPP_ROW` 0x150 through 0x165 as broadcasting one source lane within the row
  at `cdna2/README.md:6818` through `:6840`.
- The XML does not carry the `DPP_CTRL` enumeration, as recorded in
  `CDNA2-XML-066`.

Rocjitsu evidence:

- The shared DPP helper defines row-share as 0x150 through 0x15F and row-xmask
  as 0x160 through 0x16F at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h:67`
  through `:70`.
- `dpp_permute()` implements row-share only through 0x15F, implements row-xmask
  for 0x160 through 0x16F, and returns identity for unknown controls at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h:180`
  through `:199`.
- The shared helper test asserts 0x161 as row-xmask behavior at
  `tests/shared_infra_test.cpp:1617` through `:1633`.

Impact:

For CDNA2, valid manual row-broadcast controls 0x160 through 0x165 are decoded
as row-xmask, and controls 0x166 through 0x16F are accepted as row-xmask even
though the CDNA2 manual does not list that range. Reserved 0x100 also falls into
the helper's unknown-control identity behavior.

### CDNA2-RJ-109: DPP source modifiers are decoded in the layout but ignored

Manual/XML evidence:

- CDNA2 Table 82 defines `SRC0_NEG`, `SRC0_ABS`, `SRC1_NEG`, and `SRC1_ABS` in
  the DPP extension word at `cdna2/README.md:6803` through `:6809`.
- The generated CDNA2 machine structs include those fields for VOP1 and VOP2 DPP
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/machine_insts.h:185`
  through `:199` and `:228` through `:244`.

Rocjitsu evidence:

- The generator's DPP constructor path only copies `vsrc0`, `dpp_ctrl`,
  `row_mask`, `bank_mask`, and `bound_ctrl` from the extension word at
  `lib/python/amdisa/codegen/_generator.py:6380` through `:6390`.
- A generated representative VOP2 constructor mirrors that omission at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop2.cpp:166` through
  `:173`.
- `apply_dpp()` only reads and permutes raw VGPR source data; it receives no DPP
  source modifier state at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h:324`
  through `:340`.

Impact:

Non-zero DPP source modifier bits do not affect execution. DPP forms of
floating-point VOP1/VOP2/VOPC instructions can therefore compute as if
`SRC*_NEG` and `SRC*_ABS` were clear.

### CDNA2-RJ-110: CDNA2 SDWA reserved selector values execute as byte selectors

Manual/XML evidence:

- CDNA2 Table 80 says SDWA `DST_SEL` values 0-3 and 7 are reserved, values 4 and
  5 select the low and high 16-bit words, value 6 selects the full dword, and
  `SRC0_SEL`/`SRC1_SEL` use the same options at `cdna2/README.md:6737` through
  `:6753`.
- The XML exposes those selector fields without the reserved-value table, as
  recorded in `CDNA2-XML-066`.

Rocjitsu evidence:

- The shared SDWA helper defines selector values 0-3 as `BYTE_0` through
  `BYTE_3`, 4 as `WORD_0`, 5 as `WORD_1`, and 6 as `DWORD` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h:381`
  through `:390`.
- `sdwa_src_select()` and `sdwa_dst_merge()` implement byte extraction and byte
  destination merging for selectors 0-3 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h:405`
  through `:423` and `:432` through `:465`.
- The shared SDWA tests assert byte selector behavior at
  `tests/shared_infra_test.cpp:2724` through `:2755`.

Impact:

CDNA2 SDWA encodings that the manual marks reserved are treated as defined byte
operations. This can hide illegal encodings and produce concrete byte-level
results where a CDNA2 oracle should reject, trap, or treat the behavior as
undefined.

### CDNA2-RJ-111: SDWA scalar and inline sources bypass operand reads during sub-dword selection

Manual/XML evidence:

- CDNA2 Table 80 and Table 81 define `S0`/`S1` bits selecting VGPR versus SGPR
  sources for SDWA/SDWAB at `cdna2/README.md:6748` through `:6753` and
  `:6778` through `:6783`.
- The XML says `S0`/`S1` select VGPR versus SGPR or built-in inline constants at
  `amdgpu_isa_cdna2.xml:5453` through `:5469` and `:6571` through `:6587`.

Rocjitsu evidence:

- Generated constructors classify SDWA sources as `OPR_SRC` when `S0` or `S1`
  are set, for example at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop2.cpp:174` through
  `:190` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vopc.cpp:44`
  through `:58`.
- The execute paths then bypass the operand read helpers whenever
  `SRC*_SEL != DWORD`: they compute `wf.vgpr_alloc().base +
  encoding_value_` and call `read_vgpr()` directly at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop2.cpp:208` through
  `:244` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vopc.cpp:75`
  through `:111`.
- The SDWA `NEG`/`ABS` logic also lives only inside those non-DWORD selector
  branches, so `SRC*_SEL=DWORD` with source modifiers set skips the modifier
  path in the same generated representative code.

Impact:

Sub-dword SDWA sources marked as SGPR or inline constants are read from the VGPR
file using the raw encoding value. Separately, a full-dword SDWA source with
non-zero `NEG`/`ABS` bits can skip the modifier entirely because the modifier
application is tied to sub-dword extraction.

### CDNA2-RJ-112: SDWA `OMOD` is decoded in structs but not modeled

Manual/XML evidence:

- CDNA2 Table 80 defines the SDWA `OMOD` field at bits 47:46 as output
  modifiers that refer to VOP3 behavior at `cdna2/README.md:6740` through
  `:6743`.
- The generated CDNA2 VOP1 and VOP2 SDWA machine structs include `omod` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/machine_insts.h:202`
  through `:223` and `:246` through `:268`.

Rocjitsu evidence:

- The generated VOP1/VOP2 encoding classes keep SDWA source selectors,
  destination selectors, destination-unused mode, and clamp state, but no
  `sdwa_omod_` field, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/encodings.h:459` through
  `:546`.
- The generator emits the same set of SDWA fields and omits OMOD storage at
  `lib/python/amdisa/codegen/_generator.py:2074` through `:2099`.
- A representative generated execute postamble applies destination merge and
  clamp but no OMOD at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop2.cpp:266` through
  `:285`.

Impact:

SDWA encodings with non-zero `OMOD` execute as if the output modifier were zero.
This affects any legal SDWA instruction where the manual permits VOP3-style
output scaling through the SDWA extension word.

### CDNA2-RJ-113: VOP2 SDWA-SDST carry forms are not wired as scalar-destination instructions

Manual/XML evidence:

- The XML defines a `VOP2_VOP_SDWA_SDST_ENC` format at
  `amdgpu_isa_cdna2.xml:5596` through `:5690`.
- Carry-out VOP2 rows such as `V_ADD_CO_U32` and `V_SUB_CO_U32` use that format
  with explicit `VDST` and `SDST` outputs at `amdgpu_isa_cdna2.xml:56969`
  through `:56997` and `:57140` through `:57168`.

Rocjitsu evidence:

- The CDNA2 machine struct for `Vop2VopSdwaSdstEncMachineInst` exists at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/machine_insts.h:271`
  through `:292`.
- The generator chooses `VopcVopSdwaSdstEncMachineInst` only for `ENC_VOPC`;
  `ENC_VOP2` falls back to `Vop2VopSdwaMachineInst` at
  `lib/python/amdisa/codegen/_generator.py:6408` through `:6411` and copies
  destination-selector fields rather than `sdst`/`sd` at `:6431` through
  `:6439`.
- Generated carry-out VOP2 classes expose only `vdst`, `src0`, and `vsrc1` in
  the C++ wrapper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop2.h:244` through `:251`,
  and representative construction parses SDWA with `Vop2VopSdwaMachineInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop2.cpp:3379` through
  `:3395`.

Impact:

For XML-defined VOP2 SDWA-SDST encodings, rocjitsu interprets bits 46:40 as the
ordinary SDWA destination-control fields instead of the scalar destination.
The carry-out destination is therefore not surfaced as an explicit `SDST`
operand and can be routed through the wrong contract.

### CDNA2-RJ-114: Legal DPP/SDWA/SDWAB suffix fields do not round-trip through disassembly

Manual/XML evidence:

- Chapter 13.3.7 through 13.3.9 define non-operand suffix/control fields for
  SDWA, SDWAB, and DPP, including selectors, masks, bounds control, source
  modifiers, scalar destination controls, clamp, and OMOD at
  `cdna2/README.md:6721` through `:6840`.

Rocjitsu evidence:

- Generic instruction disassembly prints the mnemonic, operands, and
  `build_modifiers()` output at `lib/rocjitsu/src/rocjitsu/isa/instruction.h:239`
  through `:255`.
- CDNA2 `Vop1`, `Vopc`, and `Vop2` store DPP/SDWA state, but do not override
  `build_modifiers()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/encodings.h:459` through
  `:546`.
- The CDNA2 generated coverage exception still skips `*_dpp` and `*_sdwa`
  modifier encodings at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/coverage_exceptions_cdna2.txt:29`
  through `:31`.

Impact:

Even for legal DPP/SDWA/SDWAB forms, disassembly hides `dpp_ctrl`, row/bank
masks, `bound_ctrl`, source modifiers, SDWA selectors, `S0`/`S1`, `OMOD`,
`CLAMP`, and SDWAB scalar-destination controls. Round-trip disassembly and
debug output can collapse distinct encodings to the same textual instruction.

## No-Gap Notes

- The generated CDNA2 machine structs preserve the raw bit positions for the
  VOP1/VOP2 DPP and SDWA extension words and the VOPC SDWAB extension word at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/machine_insts.h:185`
  through `:292` and `:312` through `:330`.
- VOPC SDWAB scalar-destination handling is present: generated constructors
  copy `sdst`/`sd` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vopc.cpp:44` through `:58`,
  and execution writes the compare mask to the selected SGPR pair while
  restoring VCC when `SD=1` at `:124` through `:130`.
- DPP bounds, row-mask, and bank-mask output behavior is substantially modeled.
  The shared helper computes lane write enables and output masks at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h:239`
  through `:261`; representative generated VOP2 and VOPC paths merge old
  destination/VCC lanes at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop2.cpp:252` through
  `:260` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vopc.cpp:119`
  through `:123`.
- No missing DPP real-`SRC0` high bit was found for CDNA2. The manual's real DPP
  `SRC0` is bits [39:32] at `cdna2/README.md:6803`, the machine structs expose
  an 8-bit `vsrc0` field, and the shared helper reads from that VGPR index.
- DPP permutation/write-mask test coverage exists for generated CDNA2
  instantiations at `tests/shared_infra_test.cpp:2600` through `:2624`. The new
  DPP findings are about CDNA2-specific control-value legality, source
  modifiers, and disassembly, not absence of every DPP execution test.
- Base EXEC-masked VGPR writeback is broadly represented by the generated
  per-lane execution loops; the new Chapter 6 gaps above are about VGPR-index
  addressing and MODE-driven FP result behavior.
- `CDNA2-RJ-107` is not a blanket F64-DPP rejection: CDNA2 documents MI200 DPP
  support for 64-bit data types at `cdna2/README.md:297`, and generated
  `V_FMAC_F64` DPP handling follows an XML DPP encoding that is not listed in
  Chapter 12.17's exclusions.
- Chapter 6.2.3 repeats the out-of-range VGPR source/destination contract
  already tracked in `CDNA2-RJ-014`, so this slice did not add a duplicate
  rocjitsu gap for that prose.
- The CDNA2 generated SOP2 inventory matches the Chapter 12.1 manual/XML opcode
  inventory after normalizing the manual OCR wrap on `S_PACK_LL_B32_B16`: 53
  generated constructors span `SAddU32Sop2` through `SPackHhB32B16Sop2` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sop2.cpp:20` through
  `:1260`.
- The CDNA2 generated SOPK inventory matches the Chapter 12.2 manual/XML opcode
  inventory after accounting for the literal-only opcode 20 form: generated
  constructors span `SMovkI32Sopk` through `SCallB64Sopk`, include
  `SSetregImm32B32Sopk`, and keep opcode 19 absent at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopk.cpp:23` through
  `:407`.
- `S_CBRANCH_G_FORK` and `S_RFE_RESTORE_B64` are SOP2 rows, but their missing
  execution semantics are already tracked under the program-control findings
  `CDNA2-RJ-024` and `CDNA2-RJ-022`, respectively.
- `S_CBRANCH_I_FORK` is a SOPK row, but its missing execution semantics are
  already tracked under the program-control finding `CDNA2-RJ-024`.
- The shared SOPK helpers implement the Chapter 12.2 signed/unsigned SIMM16
  rules for `S_MOVK_I32` and `S_CMPK_*`: signed forms cast through `int16_t`,
  while unsigned compare forms zero-extend the 16-bit immediate at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1412`
  through `:1504` and `:2033` through `:2037`.
- `S_ADDK_I32` and `S_MULK_I32` are not missing old-destination dataflow:
  generated constructors expose the destination as both source and destination,
  the shared helpers read the old scalar value before writeback, and adjacent
  tests assert def-use plus SCC behavior at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopk.cpp:235` through
  `:261`, `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:455`
  through `:465` and `:2093` through `:2098`, and
  `tests/scalar_scc_test.cpp:627` through `:690`.
- The CDNA2 generated SOP1 inventory matches the Chapter 12.3 XML opcode
  inventory, including `SBitset1B64Sop1` for opcode 27 and the later
  `SAndn1*`/`SOrn1*`/`SBitreplicate*` rows at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sop1.cpp:23` through
  `:1058`.
- The CDNA2 generated SOPC inventory matches the Chapter 12.4 manual/XML opcode
  inventory: constructors span `SCmpEqI32Sopc` through `SCmpLgU64Sopc` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopc.cpp:20` through
  `:477`.
- The shared SOPC compare and bit-compare helpers implement the Chapter 12.4
  signed/unsigned comparisons and B32/B64 bit-index masking at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:897`
  through `:925` and `:1116` through `:1278`; adjacent tests cover
  representative SOPC SCC producer behavior at
  `tests/scalar_scc_test.cpp:816` through `:875`.
- `S_RFE_B64` and `S_CBRANCH_JOIN` are SOP1 rows, but their missing execution
  semantics are already tracked under the program-control findings
  `CDNA2-RJ-022` and `CDNA2-RJ-024`, respectively.
- `S_SETVSKIP` is a SOPC row, and its missing execution semantics are already
  tracked under the program-control finding `CDNA2-RJ-017`; `S_SET_GPR_IDX_ON`
  M0 bit-position and per-source indexing behavior remain tracked under
  `CDNA2-RJ-092`.
- The shared SOP1 helpers implement representative Chapter 12.3 unary and EXEC
  formulas: bit-count helpers write SCC, WQM/quad-mask helpers reduce four-lane
  groups, `S_ABS_I32` preserves the `0x80000000` bit pattern, and the
  saveexec/wrexec helpers save or overwrite EXEC as specified at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:346`
  through `:354`, `:592` through `:710`, `:764` through `:797`, `:2376`
  through `:2402`, and `:2693` through `:2718`.
- The shared scalar-pack helpers implement the Chapter 12.1 half-selection rows
  for `S_PACK_LL_B32_B16`, `S_PACK_LH_B32_B16`, and `S_PACK_HH_B32_B16` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2344`
  through `:2372`; the remaining pack risk in this slice is missing targeted
  regression coverage.
- CDNA2 Chapter 1-2 dispatch, 64-lane wavefront, initial `EXEC`, and packed
  work-item-ID basics are represented by the production dispatch path. The
  ISA profile defaults CDNA to Wave64 at
  `lib/python/amdisa/isa_profile.py:790` through `:799`, config enables packed
  TID for CDNA2 at `lib/rocjitsu/src/rocjitsu/config/config_loader.cpp:441`
  through `:447`, work-item coordinates and packed IDs are computed in
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/dispatch_entry.h:186` through `:203`,
  initial `EXEC` is built from grid/workgroup bounds at
  `dispatch_entry.h:206` through `:235`, and production dispatch installs that
  mask before register initialization at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:710` through
  `:720`. The new Chapter 2 issue above is limited to the device-memory
  consistency and acknowledgment model; detailed LDS/GWS execution gaps remain
  tracked separately.
- `V_PK_MOV_B32` is generated with 64-bit source and destination operands at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop3p.cpp:718` through
  `:731`, and the shared helper reads both sources with `read_lane64()` and
  selects output dwords with `OPSEL[0]` and `OPSEL[1]` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:17295`
  through `:17312`.
- MIX helpers implement the MIX-specific selector mapping, treat `NEG_HI` as
  absolute value, and apply `CLMP` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:13290`
  through `:13415`. They use multiply-add rather than fused FMA; the detailed
  CDNA2 instruction table and XML descriptions support multiply-add, while
  section 6.7 contains conflicting fused wording.
- Packed 32-bit helpers do not apply clamp or other output modifiers, matching
  the packed 32-bit statement in `cdna2/README.md:1524` through `:1526` that
  output modifiers are not supported for those instructions.
- CDNA2 has explicit AccVGPR plumbing for MFMA A/B and C/D selection:
  generated constructors apply the `ACC` and `ACC_CD` fields, CDNA2
  `resolve_acc()` uses `AccMode::Separate`, and the shared destination/source
  base helpers map encoded AccVGPR operands at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop3p.cpp:777` through
  `:819`, `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mma_exec.h:9`
  through `:24`, and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:78` through
  `:150`.
- CDNA2 generated VOP3P MFMA/ACCVGPR decode inventory is not absent. A static
  comparison of the 29 Chapter 12.10 MFMA/ACCVGPR rows found matching
  manual/XML/generated opcode names, opcode numbers, operand classes, and
  operand sizes across the manual tables at `cdna2/README.md:4158` through
  `:4191` and `:6686` through `:6719`, XML entries at
  `amdgpu_isa_cdna2.xml:65995` through `:67293`, generated constructors at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop3p.cpp:740` through
  `:1959`, and decoder table entries at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/decoder.cpp:8404` through
  `:8451`. The MFMA findings above are semantic or legality gaps layered on
  top of that decode surface.
- CDNA2 F64 MFMA does not appear to have the later CDNA3/CDNA4
  BLGP-as-negation override in the manual. The missing CDNA3/CDNA4-style
  negation behavior is therefore not recorded as a CDNA2 gap; CDNA2 F64
  broadcast/swizzle field handling is tracked separately in `CDNA2-RJ-009`.
- The CDNA2 generated SOPP class and decoder inventory matches the checked-in
  XML inventory, including the XML-only `S_TTRACEDATA` opcode recorded in
  `CDNA2-XML-059`: constructors span `SNopSopp` through
  `SEndpgmOrderedPsDoneSopp` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopp.cpp:20` through
  `:457`, and `sub_decode_sopp` maps opcodes 0 through 30 before invalid
  entries at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/decoder.cpp:6918`
  through `:6948`.
- Ordinary PC-relative predicate branches are implemented for CDNA2: `S_BRANCH`
  and the SCC/VCC/EXEC conditional forms update `wf.pc` from signed
  instruction-count offsets and set branch metadata at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopp.cpp:39` through
  `:200`.
- The normal `S_ENDPGM`/`S_ENDPGM_SAVED` drain-before-halt behavior is
  represented at the simulator wait-counter level: the generated SOPP bodies
  call `wf.end()`, and `Wavefront::end()` transitions to `ENDING` until
  `wait_counters_.empty()` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:446` through `:453`; the
  memory pipeline and wait-counter release path retire `ENDING` wavefronts only
  after outstanding counters drain at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.h:77` through `:83`,
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.cpp:21` through `:26`, and
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:301` through `:310`.
- Direct PC and call forms are not absent: `S_GETPC_B64`, `S_SETPC_B64`,
  `S_SWAPPC_B64`, and `S_CALL_B64` read/write scalar pairs and update `wf.pc`
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sop1.cpp:543` through
  `:594` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopk.cpp:387`
  through `:407`; `CDNA2-RJ-096` is limited to call metadata classification.
- The core classic `S_BARRIER` release path is represented: the generated
  instruction sets `WfState::BARRIER`, and the CU releases all non-halted
  waves in the same dispatch/workgroup once they have all reached the barrier
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:745`
  through `:747` and
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:313` through `:335`.
- CDNA2 `S_WAITCNT` threshold decode itself is present: the SOPP executor
  decodes VM, EXP, and LGKM fields from the immediate and calls
  `wf.set_wait_target()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/sopp.cpp:225` through
  `:240`. The wait-counter gaps above are about producer accounting and
  architecture-specific hazard rules, not absence of the threshold instruction.
- Representative CDNA2 scalar ALU execution does write SCC for ordinary
  arithmetic and bitwise operations. For example, `S_ADD_U32` writes carry-out
  SCC at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:433`
  through `:440`. The def-use gap above is about analysis metadata, not the
  basic execution side effect.
- CDNA2 `S_MAX_I32` and `S_MAX_U32` match the checked CDNA2 strict tie rule:
  the shared helpers write SCC with `s0 > s1` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1866`
  through `:1898`, and existing tests assert false SCC for equal signed and
  unsigned operands at `tests/scalar_scc_test.cpp:733` through `:738`.
- CDNA2 `S_SETREG_IMM32_B32` is not missing extension-word fetch for execution:
  `Sopk::hasImpliedLiteral()` adds and stores the extension word for opcode 20
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/encodings.cpp:68`
  through `:82`. The gap above is that the literal is hidden from operand
  metadata and def-use/disassembly clients.
- CDNA2 raw MUBUF load/store execution is not absent. The generated
  `BUFFER_LOAD_DWORD` and `BUFFER_STORE_DWORD` paths call the shared buffer
  address helper, issue vector-memory operations, and route lane masks through
  the memory pipeline at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mubuf.cpp:650` through
  `:697` and `:1024` through `:1066`. The Chapter 9 gaps above are about
  special `SOFFSET` selectors, formatted/typed conversion, descriptor modes,
  and cache details.
- CDNA2 generated MUBUF/MTBUF decode inventory is not absent. The CDNA2 decoder
  maps the 76 defined MUBUF opcode rows, including the high atomic range and the
  invalid holes, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/decoder.cpp:8865` through
  `:8975`, maps all 16 MTBUF opcodes at `:8998` through `:9008`, and generated
  encoding fixtures include the same MUBUF/MTBUF rows at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/test_encodings.h:1203`
  through `:1294`. The MUBUF/MTBUF gaps above are semantic gaps layered on that
  generated decode surface.
- CDNA2 signed MUBUF D16 load sign extension is represented for implemented raw
  paths. The generated `BUFFER_LOAD_SBYTE_D16*` execute bodies set
  `d->sign_extend = true` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mubuf.cpp:1334` through
  `:1406`, and memory-pipeline writeback applies 8-bit or 16-bit sign extension
  before D16 lane writeback at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:175` through `:185`.
- CDNA2 MUBUF atomics are not wholly missing. Atomic execute bodies set
  `d->is_load = (inst_.glc != 0)`, fill the atomic opcode, and pass store data
  to the vector-memory pipeline at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mubuf.cpp:2140` through
  `:2225`; the pipeline performs the read-modify-write and returns old data
  when requested at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:286` through
  `:370` and `:476` through `:485`.
- CDNA2 MUBUF and DS float atomics are not wholly absent. The generated MUBUF
  and DS classes route representative float atomics into the shared atomic RMW
  pipeline; `CDNA2-RJ-061` and `CDNA2-RJ-064` are about the missing
  floating-point subtype, packed-lane, and edge-case semantics layered on that
  base execution path.
- CDNA2 base flat-memory segment naming is not absent. The generated
  `flat_mnemonic()` helper rewrites `flat_*` mnemonics to `scratch_*` or
  `global_*` when `SEG` is 1 or 2 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/encodings.cpp:14` through
  `:21`, and decode smoke tests assert `global_load_dword` and
  `scratch_load_dword` names at `tests/decode_smoke_test.cpp:864` through
  `:893`. The Chapter 10 and Chapter 12.16 gaps above are about segment/opcode
  legality, execution semantics, signed-offset display, and edge-case coverage,
  not the basic load/store mnemonic rewrite.
- CDNA2 shared flat address calculation is not wholly absent. The helper
  distinguishes Flat, Scratch, and Global base address formulas and applies the
  12-bit Flat versus 13-bit Global/Scratch instruction-offset widths at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h:27`
  through `:57`; the new gaps are about the remaining `LDS`, wait-counter,
  aperture, fault, segment-legality, and disassembly details.
- CDNA2 non-format D16 half writeback is represented in the shared memory
  pipeline. The completion path updates the selected half and preserves the
  other half when ECC is disabled, while ECC mode writes a full register value
  with the unused half cleared/shifted, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:120` through
  `:198`.
- CDNA2 ordinary LDS load/store/atomic execution is not wholly absent. Generated
  DS paths create `LOCAL_MEM` vector-memory states, assign `LGKMCNT`, compute
  per-lane LDS addresses, and route loads/stores/atomics through the shared
  LDS memory pipeline. The Chapter 11 gaps above are about M0 clamp,
  duplicate-offset, ADDTID, selector, GWS, swizzle-mode, append/consume return,
  and operand-role edge cases layered on top of that base execution path.
- CDNA2 generated DS decode inventory is not absent. The decoder maps the 124
  Chapter 12.12/13.4 opcode rows, including `DS_SWIZZLE_B32`,
  `DS_CONSUME`, `DS_APPEND`, `DS_WRITE_B128`, and `DS_READ_B128`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/decoder.cpp:8471` through
  `:8728`; generated encoding fixtures include the same DS rows at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/test_encodings.h:1079`
  through `:1202`. The current DS findings are semantic and coverage gaps, not
  absence of base decode entries.
- CDNA2 `DS_PERMUTE_B32`/`DS_BPERMUTE_B32` lane movement is substantially
  represented: generated helpers divide byte addresses by four, apply the
  wave64 lane group for CDNA2, honor `EXEC` source/destination behavior, and
  let later active lanes overwrite earlier conflicting permute destinations at
  `lib/python/amdisa/codegen/_generator.py:4170` through `:4191`. The XML gap
  records that these rules are prose-only; this slice did not identify a clear
  CDNA2 rocjitsu execution mismatch for the basic permute/bpermute contract.
- CDNA2 `V_SWAP_B32` ordinary VOP1/VOP3 read-write dataflow is represented:
  generated constructors expose both operands as sources and destinations, and
  the execute bodies read `VDST`, write `VDST` from `SRC0`, and then write the
  old destination back to `SRC0` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop1.cpp:9564` through
  `:9574`, `:9662` through `:9670`, and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop3.cpp:1829` through
  `:1851`.
- CDNA2 `V_CMP_CLASS_*` result classification is substantially represented in
  both scalar and SIMD paths: the generated/shared helpers classify raw
  f16/f32/f64 sign, exponent, mantissa, signaling-NaN, quiet-NaN, infinity,
  normal, denormal, and signed-zero bits at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:3882`
  through `:3958` and
  `lib/python/amdisa/codegen/execute/simd_codegen.py:1451` through `:1582`.
  The new compare gap above is limited to exception signaling, not ordinary
  class-mask result bits.
- CDNA2 VOP3B scalar-destination dataflow is not absent: generated constructors
  expose both the VGPR destination and scalar destination for carry-out,
  divide-scale, and wide-MAD forms at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop3.cpp:11950` through
  `:12151`, and the shared helpers write `inst.sdst` for the VOP3B mask/carry
  result.
- `V_CVT_PKACCUM_U8_F32` old-destination dataflow is represented: the
  generated constructor lists `vdst` as a source and destination at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop3.cpp:4333` through
  `:4344`, and the helper reads the old destination byte container before
  merging the selected byte at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:9735`
  through `:9746`.
- CDNA2 `V_READLANE_B32` and `V_WRITELANE_B32` basic EXEC-ignoring dataflow is
  represented: the readlane body reads the selected VGPR lane and writes an
  SGPR without checking `EXEC`, and the writelane body writes the selected VGPR
  lane without checking `EXEC`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/vop3.cpp:5541` through
  `:5563`.
- CDNA2 MUBUF load-to-LDS routing exists for implemented load forms. The
  generated MUBUF paths set `lds_dst` and derive the LDS base from `M0` plus
  the wave LDS base at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mubuf.cpp:470` through
  `:499`, `:620` through `:697`, and `:1228` through `:1298`; the memory
  pipeline writes per-lane data to LDS at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:55` through `:112`.
- CDNA2 SMEM decode inventory is not absent. The generated decoder maps the
  Chapter 12.6/13.2.1 opcode inventory, including `S_MEMREALTIME`, cache/probe
  rows, and buffer/scalar atomics, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/decoder.cpp:7048` through
  `:7222`; generated encoding fixtures include all 84 SMEM rows at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/test_encodings.h:200`
  through `:283`. The SMEM gaps above are about address semantics, descriptor
  behavior, counter accounting, cache/probe/time side effects, atomics, and
  tests, not absence of the base decode surface.
- CDNA2 base MIMG decode is not absent. The generated decode table maps the
  Chapter 13.6 opcode inventory to `IMAGE_LOAD*`, `IMAGE_STORE*`,
  `IMAGE_GET_RESINFO`, `IMAGE_ATOMIC_*`, and `IMAGE_SAMPLE` constructors at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/decoder.cpp:9011` through
  `:9045`, and the generated encoding fixtures include all 25 MIMG rows from
  `IMAGE_LOAD` through `IMAGE_SAMPLE` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/test_encodings.h:1295`
  through `:1319`. `IMAGE_SAMPLE` exposes the sampler operand at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mimg.cpp:545` through
  `:561`. The Chapter 9 and Chapter 12.15 image gaps above are about
  execution, dynamic operand counts, descriptor semantics, modifiers, and
  dependency behavior.
- CDNA2 integer atomic decrement behavior is not missing for the implemented
  DS, MUBUF, and FLAT paths. Those generated instructions select
  `amdgpu::AtomicOp::DEC`, for example at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/ds.cpp:206`,
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/mubuf.cpp:2216`, and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna2/flat.cpp:1762`; the shared
  RMW helper implements the manual threshold rule as
  `old_val == 0 || old_val > src_val` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:276` through
  `:279`. MIMG atomics remain covered by the image execution stub gap in
  `CDNA2-RJ-055`.
