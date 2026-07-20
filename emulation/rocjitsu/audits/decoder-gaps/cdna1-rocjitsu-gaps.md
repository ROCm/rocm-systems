# CDNA1 Rocjitsu Gaps

Architecture: CDNA1

Manual source: `workspace_docs/amdgpu-isa-manuals/cdna1/README.md`

Rocjitsu source: `emulation/rocjitsu`

## Coverage

| Area | Status | Notes |
| --- | --- | --- |
| CDNA1 introduction and terminology | Audited statically, partial | Checked Chapter 1 architecture overview, dispatch/workgroup/wavefront/work-item terms, Wave64 model, SALU/VALU terminology, literal/microcode terms, and host/kernel launch context against ISA profiles, dispatch entry helpers, command-processor register initialization, CU wavefront allocation, and wavefront state. |
| CDNA1 program organization and memory overview | Audited statically, partial | Checked Chapter 2 SALU/VALU/vector-memory roles, `EXEC`-masked vector execution, dispatch-grid indexing, LDS/GDS topology, and device-memory consistency/acknowledgment prose against dispatch, wavefront, LDS, DS/GDS stubs, memory-pipeline, and cache code. |
| CDNA1 wave/kernel state | Audited statically, partial | Checked Chapter 3.1-3.6 PC/EXEC/STATUS/MODE state, helper bits, VSKIP, GPR/LDS allocation and aliasing, out-of-range behavior, and LDS allocation/clamping against `Wavefront`, generated SALU accessors, branch/source helpers, and CU/SPI allocation code. |
| CDNA1 special wave/kernel state | Audited statically, partial | Checked Chapter 3.7-3.11 M0, SCC, VCC/VCCZ, trap/exception/TRAPSTS/memory-violation state, and TTMP privilege against operands, SOPK/SOPP execution, memory pipelines, and command-processor initialization. |
| CDNA1 program flow control | Audited statically, partial | Checked Chapter 4.1-4.6 branch/direct-PC/call/fork/join/barrier/waitcnt/wait-state/message/trap/sleep/wakeup/debug-control behavior against generated SOPP/SOP1/SOP2/SOPK code, shared helpers, CU wavefront state transitions, and adjacent tests. |
| CDNA1 core VALU | Audited statically, partial | Checked Chapter 6.1-6.6 VOP1/VOP2/VOPC/VOP3 format roles, source and literal restrictions, EXEC-masked writes, compare/carry destinations, VOP3 output modifiers, FP round/denorm MODE behavior, ALU clamp semantics, VGPR indexing, generated operand/runtime helpers, and adjacent tests. |
| CDNA1 VOP3P packed math | Audited statically | Checked generated packed F16/I16/U16 helpers, MIX helpers, selector handling, clamp behavior, DPP reachability, and absence of packed F32 / `V_PK_MOV_B32` classes for this slice. |
| CDNA1 MFMA/VOP3P-MAI | Audited statically | Checked generated MFMA variants, fixed AccVGPR C/D routing, `ACC` source selection, AccVGPR allocation and Arch/Acc movement instructions, inline constants, dependency waits, `CBSZ` legality, opcode inventory drift, and targeted test coverage. |
| CDNA1 SALU scalar operands and SCC behavior | Audited statically, partial | Checked Chapter 5.1-5.7 scalar selector handling, literals, out-of-range source/destination behavior, 64-bit SGPR pair alignment, arithmetic/compare/bitwise SCC effects, `S_ABSDIFF_I32` edge behavior, implicit SCC metadata, and adjacent scalar helpers. |
| CDNA1 SOP2 instruction definitions | Audited statically | Checked Chapter 12.1 generated SOP2 opcode inventory, literal fixups, arithmetic/carry/overflow helpers, bitfield helpers, fork/RFE rows, absolute-difference behavior, shifted-add helpers, pack half-selection, and adjacent tests. |
| CDNA1 SOPK instruction definitions | Audited statically | Checked Chapter 12.2 generated SOPK opcode inventory, opcode-19 invalid slot, opcode-20 literal-only form, SIMM16 extension helpers, ADDK/MULK destination dataflow, SCC behavior, fork/call PC formulas, HWREG constructors/execution, and adjacent tests/oracle encodings. |
| CDNA1 SOP1 instruction definitions | Audited statically | Checked Chapter 12.3 generated SOP1 opcode inventory, literal constructors, unary helpers, bit-count/bit-scan/bitset behavior, direct PC/RFE paths, saveexec/wrexec helpers, quad-mask/bitreplicate, relative SGPR addressing, join, GPR-index control, adjacent tests, and oracle encodings. |
| CDNA1 SOPC instruction definitions | Audited statically | Checked Chapter 12.4 generated SOPC opcode inventory, literal constructors, compare and bit-compare helpers, VSKIP/GPR-index rows, U64 compare rows, adjacent SCC tests, and oracle encodings. |
| CDNA1 SOPP instruction definitions | Audited statically | Checked Chapter 12.5 generated SOPP opcode inventory, constructors and decoder table, branch/waitcnt/barrier/endpgm behavior, trap metadata, message/cache/perf/ordered-PS side effects, GPR-index/thread-trace M0 metadata, existing status/debug/control gaps, and adjacent branch/barrier tests. |
| CDNA1 SMEM instruction definitions | Audited statically | Checked Chapter 12.6 and 13.2.1 generated SMEM opcode constants, test encodings, constructors, decoder table, operand metadata, shared address/runtime helpers, auxiliary cache/time/probe/discard paths, atomics, adjacent tests, and oracle encodings. |
| CDNA1 VOP2 instruction definitions | Audited statically | Checked Chapter 12.7 and 13.3.1 generated VOP2 opcode constants, test encodings, decoder table, literal-only FMA rows, promoted VOP3 forms, bitwise modifier restrictions, accumulator metadata, packed F16 accumulator execution, adjacent tests, and oracle encodings. |
| CDNA1 VOP1 instruction definitions | Audited statically | Checked Chapter 12.8 and 13.3.2 generated VOP1 opcode constants, test encodings, decoder table, literal/DPP/SDWA constructors, readfirstlane lane-selection and LDS-direct behavior, special VOP1 runtime rows, legacy exp/log availability, swap operand metadata, adjacent tests, and oracle encodings. |
| CDNA1 VOPC instruction definitions | Audited statically | Checked Chapter 12.9 and 13.3.3 generated VOPC opcode constants, test encodings, decoder table, literal/DPP/SDWA constructors, class-mask helpers, signed/unsigned compare width handling, CMPX VCC/EXEC writes, promoted VOP3 compare behavior, implicit special-state metadata, and adjacent tests. |
| CDNA1 VOP3A/VOP3B instruction definitions | Audited statically | Checked Chapter 12.12 and 13.3.4-13.3.5 generated VOP3A/VOP3B opcode constants, decoder tables, constructor operand shapes, VOP3B SDST writeback, carry/div-scale/wide-MAD runtime helpers, and adjacent scalar-vs-SIMD tests. |
| CDNA1 VINTERP instruction definitions | Audited statically | Checked Chapter 12.11 and 13.4.1 generated compact VINTRP and promoted/native VOP3 interpolation opcode constants, decoder tables, constructor operand shapes, runtime stubs, and test coverage. |
| CDNA1 LDS/GDS instruction definitions | Audited statically | Checked Chapter 12.13 and 13.5.1 generated DS opcode constants, test encodings, decoder tables, representative constructors/runtime paths, existing LDS/GDS semantic gaps, and `DS_CONDXCHG32_RTN_B64` behavior. |
| CDNA1 MUBUF/MTBUF instruction definitions | Audited statically, partial | Checked Chapter 12.14/12.15 and 13.6.1/13.6.2 generated MUBUF/MTBUF opcode constants, test encodings, decoder tables, representative constructors/runtime paths, existing vector-buffer semantic gaps, and packed-F16 buffer atomic behavior. |
| CDNA1 MIMG instruction definitions | Audited statically, partial | Checked Chapter 12.16 and 13.7.1 generated MIMG opcode constants, test encodings, decoder table, representative constructors/runtime stubs, existing image semantic gaps, and CD derivative/sample/gather opcode coverage. |
| CDNA1 FLAT/Scratch/Global instruction definitions | Audited statically, partial | Checked Chapter 12.17 and 13.8 generated flat-family opcode constants, test encodings, decoder table, representative constructors/runtime paths, existing flat/global/scratch semantic gaps, and the global-only FP atomic opcode hole. |
| CDNA1 DPP/SDWA instruction limitations and formats | Audited statically, partial | Checked Chapter 12.18 DPP/SDWA exclusion lists, Chapter 13.3.7-13.3.9 extension fields, generated compact VOP1/VOP2/VOPC constructors, representative LLVM assembler legality, DPP shared tests, and overlap with existing `V_SWAP_B32` and DPP write-mask findings. |
| CDNA1 HWREG access and SOPK literals | Audited statically, partial | Checked Chapter 5.8 access-instruction prose, HWREG ID/subfield tables, access permissions, `S_SETREG` spacing, `S_SETREG_IMM32_B32` literal form, generated SOPK constructors/execution, and operand metadata. |
| CDNA1 scalar relative moves | Audited statically, partial | Checked `S_MOVRELS_{B32,B64}` and `S_MOVRELD_{B32,B64}` M0-index formulas, M0 width, even-M0 64-bit requirement, XML implicit M0 operands, and generated SOP1 execution. |
| CDNA1 scalar memory operations | Audited statically, partial | Checked Chapter 8.1-8.4 SMEM field/operand decoding, IMM/SOE/M0 offset forms, ordinary/scratch/buffer address helpers, scalar-buffer resources, stores, cache/discard/probe/time ops, atomics, LGKM accounting, alignment/OOR rules, disassembly modifiers, and adjacent tests. |
| CDNA1 vector-buffer operations | Audited statically, partial | Checked Chapter 9.1 MUBUF/MTBUF field behavior, `SOFFSET` selector handling, shared address helper behavior, typed/resource format conversion, raw D16 half-writeback, raw buffer loads/stores/atomics, buffer-to-LDS, cache flags, cache-maintenance opcodes, and LDS-store opcodes. |
| CDNA1 vector-image operations | Audited statically, partial | Checked Chapter 9.2-9.4.5 MIMG fields, image opcode inventory, no-sampler/sampler address tables, `DMASK`/D16/A16 behavior, resource and sampler operands, generated MIMG constructors/stubs, disassembly modifiers, wait-counter behavior, coverage exceptions, and adjacent decode tests. |
| CDNA1 flat/global/scratch memory operations | Audited statically, partial | Checked Chapter 10.1-10.8 FLAT/GLOBAL/SCRATCH field behavior, aperture routing, scratch/global address forms, direct-LDS behavior, atomics including global-only FP atomics, segment/opcode validity, signed offsets, cache/NV fields, memory-error policy, and adjacent tests. |
| CDNA1 data-share operations | Audited statically, partial | Checked Chapter 11.1-11.4 LDS geometry/dataflow, M0 clamp prose, indexed load/store/atomic address helpers, READ2/WRITE2 duplicate-offset behavior, SRC2 opcode inventory, ADDTID, FP atomics, swizzle/permute/bpermute helpers, append/consume, GWS stubs, DS disassembly, and adjacent tests. |
| Remaining CDNA1 rocjitsu surface | Not started | Full decoder/autogen/runtime fuzzing remains. |

## Gaps

### CDNA1-RJ-001: Packed F16 VOP3P arithmetic ignores `CLMP`

Manual/XML evidence:

- The CDNA1 VOP3P field table defines `CLMP` as "1 = clamp result" at
  `cdna1/README.md:6724` through `:6725`.
- The CDNA1 XML describes VOP3P `CLAMP` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:2069`.

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

### CDNA1-RJ-002: MFMA ignores the `ACC` source-bank selector for SRC0/SRC1

Manual/XML evidence:

- The CDNA1 manual says A and B matrix sources can come from either ArchVGPRs
  or AccVGPRs at `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1539`
  through `:1543`.
- The VOP3P-MAI field table defines `ACC[0]` and `ACC[1]` as selecting whether
  SRC-A and SRC-B read ArchVGPR or AccVGPR at `cdna1/README.md:6758`.
- The XML exposes SRC0 and SRC1 as `OPR_SRC_VGPR_OR_ACCVGPR` on representative
  MFMA entries at `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:68782`
  through `:68818`.

Rocjitsu evidence:

- Generated CDNA1 MFMA constructors store SRC0 and SRC1 directly from the raw
  encoding fields and do not add the AccVGPR encoding offset when `inst_.acc`
  selects AccVGPR, for example at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop3p.cpp:599` through
  `:617`, `:754` through `:772`, `:909` through `:927`, and `:1064` through
  `:1082`.
- The shared `src_base()` helper only selects the AccVGPR bank when the operand
  encoding value is in the AccVGPR range (`>= 768`); otherwise raw values
  `256-511` are read as ArchVGPRs at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:88` through
  `:92`.

Impact:

CDNA1 MFMA encodings with `ACC[0]` or `ACC[1]` set still read A/B operands from
the ArchVGPR bank, so AccVGPR A/B source forms decode and execute incorrectly.

### CDNA1-RJ-003: MFMA SRC2 reads the ArchVGPR bank instead of the AccVGPR C source

Manual/XML evidence:

- The manual states that C always comes from AccVGPRs and D always uses
  AccVGPRs at `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1539` through
  `:1543`.
- The XML uses fixed `OPR_ACCVGPR` for VDST and
  `OPR_SRC_ACCVGPR_OR_CONST` for SRC2 on representative MFMA entries at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:68782` through
  `:68818` and `:69563` through `:69602`.

Rocjitsu evidence:

- Generated CDNA1 MFMA constructors store SRC2 directly from the raw encoding
  field as `OPR_SRC_ACCVGPR_OR_CONST`, for example at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop3p.cpp:599` through
  `:617`.
- CDNA1 `resolve_acc()` uses `AccMode::VgprOnly`, documented as having no
  dedicated AccVGPR file, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mma_exec.h:9` through
  `:24`.
- In the shared `AccMode::VgprOnly` path, raw SRC2 values `256-511` resolve to
  `vb + (src2_ev - 256)`, i.e. the ArchVGPR bank, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:151` through
  `:157`.
- Other fixed AccVGPR operations map raw AccVGPR operands to the physical
  AccVGPR region: `v_accvgpr_write` constructs an `OPR_ACCVGPR` destination at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop3p.cpp:578` through
  `:594`, and `Operand::resolved_vgpr_offset()` maps raw `OPR_ACCVGPR` values
  to offset `+256` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/operand.cpp:1399` through
  `:1407`.

Impact:

MFMA accumulates from the wrong C matrix bank when SRC2 names an AccVGPR. A
program that initializes C with `v_accvgpr_write` writes the physical AccVGPR
region, but MFMA SRC2 reads the ArchVGPR region for the same encoded register.

### CDNA1-RJ-004: Fixed AccVGPR operands lose disassembly and register-reference metadata

Manual/XML evidence:

- CDNA1 MFMA and `V_ACCVGPR_READ`/`V_ACCVGPR_WRITE` expose fixed AccVGPR
  operands in the manual and XML; representative MFMA XML entries use
  `OPR_ACCVGPR` and `OPR_SRC_ACCVGPR_OR_CONST` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:68782` through
  `:68818`.

Rocjitsu evidence:

- Generated CDNA1 fixed-AccVGPR operands are constructed from raw 8- or 9-bit
  fields, for example `v_accvgpr_read`/`v_accvgpr_write` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop3p.cpp:557` through
  `:594` and MFMA VDST/SRC2 at `:599` through `:617`.
- `Operand::name()` only prints `accN` for `OPR_ACCVGPR`,
  `OPR_SRC_ACCVGPR`, and `OPR_SRC_ACCVGPR_OR_CONST` when the internal encoding
  value is already in the shifted AccVGPR ranges, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/operand.cpp:61` through
  `:64` and `:297` through `:335`.
- `Operand::to_register_ref()` has the same shifted-range requirement for
  fixed AccVGPR operand classes at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/operand.cpp:922` through
  `:996`.
- The execution offset helper does recognize raw fixed AccVGPR values by
  mapping them to physical offset `+256` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/operand.cpp:1399` through
  `:1407`, so execution and metadata use different normal forms.

Impact:

Decoded CDNA1 AccVGPR operands can fall back to numeric display text and may be
missing from register-reference/liveness metadata even when execution maps the
same raw value to the AccVGPR physical region.

### CDNA1-RJ-005: MFMA dependency-wait rules are not modeled

Manual evidence:

- Section 7.2 requires NOPs or independent instructions for VALU, MFMA,
  `V_ACCVGPR_READ`, `V_ACCVGPR_WRITE`, and `V_CMPX` producer/consumer
  combinations at `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1583`
  through `:1616`.
- The detailed opcode table gives block and pass counts at `cdna1/README.md:4155`
  through `:4181`; the dependency table depends on those pass counts.

Rocjitsu evidence:

- Codegen only marks MFMA instructions with the broad `MFMA` flag at
  `lib/python/amdisa/codegen/_generator.py:6539` through `:6542`.
- `ComputeUnit::issue_instruction()` executes the decoded instruction directly
  and routes only memory operations through the wait-counter pipeline at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:411` through `:437`.
- `WaitCounters` are documented as memory-operation counters, not MAI
  producer/consumer timing hazards, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wait_counters.h:36` through `:65`.

Impact:

Rocjitsu functional execution can run CDNA1 MAI producer/consumer sequences
that the manual requires software to separate.

### CDNA1-RJ-006: VOP3P-MAI `CBSZ` legal values are not validated

Manual evidence:

- The VOP3P-MAI field table says `CBSZ` legal values are `0-4` at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:6751`.

Rocjitsu evidence:

- Generated CDNA1 MFMA execute paths pass raw `inst_.cbsz` to the shared MFMA
  helpers, for example at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop3p.cpp:618` through
  `:627`, `:773` through `:782`, `:928` through `:937`, and `:1083` through
  `:1092`.
- `permute_a_lane()` accepts the raw value and computes `64 >> cbsz` whenever
  `cbsz != 0`, with no CDNA1 legal-value check, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:568` through
  `:579`.

Impact:

Encodings with undefined `CBSZ` values `5-7` are decoded and executed using the
helper's shift formula instead of being rejected or reported as undefined.

### CDNA1-RJ-007: MFMA SRC2 accepts generic constants beyond the MFMA subset

Manual evidence:

- Chapter 7 lists unsupported MFMA source forms and says the only inline
  constants interpreted as FP32 for all `V_MFMA` and `V_ACCVGPR` instructions
  are `0.5`, `-0.5`, `1.0`, `-1.0`, `2.0`, `-2.0`, `4.0`, and `-4.0` at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1574` through `:1581`.
- The generic VOP3P-MAI field table still lists broader source-encoding
  options, including integer constants and `1/(2*PI)`, at `cdna1/README.md:6755`
  through `:6757`; this gap follows the narrower Chapter 7 MFMA-specific rule.

Rocjitsu evidence:

- Representative CDNA1 MFMA constructors use `OPR_SRC_ACCVGPR_OR_CONST` for
  SRC2 at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop3p.cpp:599`
  through `:617`, `:754` through `:772`, `:909` through `:927`, and `:1064`
  through `:1082`.
- The CDNA1 operand printer for that class accepts generic integer inline
  constants and `0.15915494` in addition to the MFMA subset at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/operand.cpp:304` through
  `:335`.

Impact:

Rocjitsu can disassemble and execute MFMA SRC2 encodings using inline constants
that the CDNA1 Chapter 7 MFMA-specific prose does not list as legal for MFMA.

### CDNA1-RJ-008: CDNA1-specific MFMA semantic regressions are not targeted by tests

Evidence:

- The instruction execution coverage exception file excludes vector execution
  from the scalar harness and relies on kernel simulation coverage at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/coverage_exceptions_cdna1.txt:4`
  through `:13`.
- Existing generic VM coverage includes a basic MFMA F16 accumulation test at
  `tests/amdgpu_vm_test.cpp:1642` and following.
- The bit-exact SIMD-vs-scalar MFMA test fixture is explicitly CDNA4-only at
  `tests/simd_correctness/mfma_simd_exact_test.cpp:4` through `:26`.
- AccVGPR allocation/no-alias infrastructure coverage currently asserts that
  CDNA1 has zero AccVGPRs at `tests/shared_infra_test.cpp:123` through `:129`,
  while the runtime no-alias and redispatch-clearing tests cover only CDNA2/3/4
  at `tests/shared_infra_test.cpp:1424` through `:1497`.
- MFMA VM tests instantiate CDNA3 and CDNA4 only at
  `tests/amdgpu_vm_test.cpp:1635`, and targeted AccVGPR decode tests do not
  exercise CDNA1 AccVGPR operands.

Impact:

The current tests can miss CDNA1-specific MFMA issues such as `ACC` A/B source
selection, SRC2 AccVGPR C-source routing, invalid `CBSZ`, MFMA-only inline
constants, fixed AccVGPR operand disassembly/register metadata, and the
CDNA1-specific AccVGPR allocation/data-movement contract.

### CDNA1-RJ-009: SALU `POPS_EXITING_WAVE_ID` and reserved source selectors are shifted

Manual/XML evidence:

- Chapter 5.2 places `POPS_EXITING_WAVE_ID` at scalar selector 239, reserves
  selectors 249-250, and uses 251-253 for VCCZ, EXECZ, and SCC at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1004` through `:1017`.
- The CDNA1 XML agrees: `OPR_SSRC` lists `src_pops_exiting_wave_id` with value
  239 at `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:97717` through
  `:97720`.

Rocjitsu evidence:

- CDNA1 `resolve_src_scalar()` handles selectors 235-238, then maps selector
  249 to `SRC_POPS_EXITING_WAVE_ID` and selector 250 to `NULL`, while selector
  239 falls through to an unsupported scalar-read exception at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/operand.cpp:1192` through
  `:1214`.
- `can_resolve_src_scalar()` likewise returns true for 240-253, but excludes
  selector 239 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/operand.cpp:1246` through
  `:1249`.

Impact:

Legal selector 239 cannot be resolved by CDNA1 scalar operand execution or SIMD
capability checks, while reserved selectors 249-250 are accepted and resolved
as if they were defined.

### CDNA1-RJ-010: SALU allocation-bound fallback and out-of-range destination side effects are not modeled

Manual/XML evidence:

- Chapter 5.2 says out-of-range source SGPRs read SGPR0, and out-of-range
  destination SGPRs suppress the SGPR write while still writing SCC and
  saveexec EXEC side effects at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1023` through `:1025`.
- The XML records the relevant SALU operand classes and implicit SCC operands,
  but the manual-vs-XML limitation is tracked separately in `CDNA1-XML-005`.

Rocjitsu evidence:

- CDNA1 scalar source reads index `wf.sgpr_alloc().base + ev` directly for
  SGPR selectors, and 64-bit reads use `ev` and `ev + 1` directly at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/operand.cpp:1255` through
  `:1266`.
- Destination writes similarly index `wf.sgpr_alloc().base + ev` and throw for
  unsupported destination values rather than suppressing only the SGPR write at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/operand.cpp:1321`
  through `:1349`.
- SCC-producing helpers perform ordinary destination writes and SCC updates
  without an out-of-range destination suppression path, for example
  `S_ADD_I32` and `S_ADDC_U32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:418`
  through `:451`.

Impact:

Raw encodings that target allocation-out-of-range SGPRs can read/write outside
the architectural fallback contract or throw before preserving SCC/EXEC side
effects that the manual says still occur.

### CDNA1-RJ-011: 64-bit SALU SGPR pair alignment is not enforced

Manual/XML evidence:

- Chapter 5.2 requires 64-bit SGPR data to start on an even SGPR boundary at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1027`.
- The XML operand metadata carries generic wide-SGPR alignment prose on scalar
  operand definitions, and `S_MOV_B64`/other 64-bit SALU rows expose 64-bit
  scalar operands.

Rocjitsu evidence:

- CDNA1 `resolve_src_scalar64()` reads SGPR pairs rooted at the encoded `ev`
  and `ev + 1` without checking even alignment at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/operand.cpp:1255` through
  `:1268`.
- `resolve_dst_write64()` similarly writes pairs rooted at `ev` and `ev + 1`
  without checking even alignment at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/operand.cpp:1357` through
  `:1373`.

Impact:

Odd SGPR encodings for 64-bit SALU operands are treated as valid odd-rooted
pairs instead of being rejected or diagnosed as undefined.

### CDNA1-RJ-012: SALU implicit SCC dependencies are not surfaced in C++ def-use metadata

Manual/XML evidence:

- Chapter 5.3 through 5.7 defines SALU operations that write SCC for
  comparisons, carry-out, signed overflow, or nonzero results, and consume SCC
  for conditional moves/selects and carry chains at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1031` through `:1088`.
- The XML includes implicit SCC operands on representative SALU instructions:
  `S_ADD_I32` has an implicit SCC output at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:35580` through
  `:35584`, `S_CMP_EQ_I32` has implicit SCC outputs in its encodings starting
  at `:42322`, and `S_CBRANCH_SCC0` has an implicit SCC input at `:45478`
  through `:45494`.

Rocjitsu evidence:

- Generated CDNA1 constructors expose only explicit operands: `S_ADD_U32` has
  one destination and two sources at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sop2.cpp:20` through `:40`,
  and `S_CMP_EQ_I32` has two sources and no destination at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopc.cpp:20` through `:40`.
- Generic def-use construction records explicit operands plus subclass
  `implicit_defs()` and `implicit_uses()` hooks at
  `lib/rocjitsu/src/rocjitsu/analysis/def_use_chain.cpp:25` through `:44`;
  `Instruction`'s default hooks are empty at
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:215` through `:222`.
- `RegisterSet` declares SCC as a special register class but says it is not
  currently tracked and ignores non-SGPR/VGPR/AccVGPR classes at
  `lib/rocjitsu/src/rocjitsu/isa/register_set.h:54` through `:57` and
  `lib/rocjitsu/src/rocjitsu/isa/register_set.cpp:37` through `:51`.

Impact:

Execution can update or read SCC, but C++ def-use clients cannot see the scalar
condition-code dependency carried by Chapter 5 SALU producers and SCC-consuming
instructions.

### CDNA1-RJ-013: `S_ABSDIFF_I32` uses a mathematical absolute difference instead of wrapped 32-bit subtract

Manual/XML evidence:

- The detailed CDNA1 `S_ABSDIFF_I32` row first computes `D.i = S0.i - S1.i`,
  then negates the 32-bit result if it is negative, and gives edge examples at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2920`.
- In particular, the manual gives
  `S_ABSDIFF_I32(0x80000000, 0x00000001) => 0x7fffffff`, which depends on the
  wrapped 32-bit subtraction result.
- The XML records only the high-level absolute-difference description, as
  tracked in `CDNA1-XML-007`.

Rocjitsu evidence:

- The shared helper casts both operands to signed 32-bit values, promotes them
  to `int64_t`, computes `a > b ? a - b : b - a`, and casts the mathematical
  absolute difference back to 32 bits at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:358`
  through `:368`.
- For `S0=0x80000000` and `S1=0x00000001`, that computes `2147483649`, which
  wraps to `0x80000001`, not the manual's `0x7fffffff`.

Impact:

Most ordinary absolute differences match, but signed-overflow edge cases around
`INT_MIN` produce the wrong 32-bit result while still setting SCC nonzero.

### CDNA1-RJ-014: HWREG get/set uses the wrong CDNA1 register map

Manual/XML evidence:

- CDNA1 Table 19 defines HWREG ID 1 as `MODE`, ID 2 as read-only `STATUS`, ID
  3 as `TRAPSTS`, ID 4 as `HW_ID`, ID 5 as `GPR_ALLOC`, ID 6 as `LDS_ALLOC`,
  and ID 7 as `IB_STS` at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1150` through `:1167`.
- The CDNA1 XML `OPR_HWREG` low-ID names agree at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:94917` through
  `:94950`.

Rocjitsu evidence:

- CDNA1 `SGetregB32Sopk::execute_impl()` reads HWREG ID 1 from
  `wf.status_raw()`, not `MODE`, and maps IDs 5/6/7 to CU-ID halves and
  simplified allocation pairs that do not match `GPR_ALLOC`, `LDS_ALLOC`, or
  `IB_STS`, at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopk.cpp:291`
  through `:320`.
- `S_SETREG_B32` and `S_SETREG_IMM32_B32` only handle ID 1 and splice bits into
  `wf.status_raw()`, again treating ID 1 as STATUS instead of MODE, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopk.cpp:334` through
  `:384`.

Impact:

CDNA1 HWREG ID 1 acts like STATUS in rocjitsu even though the manual/XML
identify it as MODE. STATUS ID 2, TRAPSTS ID 3, allocation registers, and
IB_STS ID 7 are either unreachable or return unrelated data.

### CDNA1-RJ-015: HWREG permissions, subfields, side effects, and SETREG spacing are not modeled

Manual/XML evidence:

- Chapter 5.8 marks MODE and TRAPSTS as read/write, STATUS and allocation/status
  registers as read-only, defines `IB_STS`, `GPR_ALLOC`, and `LDS_ALLOC`
  subfields, and requires an `S_NOP` between consecutive `S_SETREG` writes to
  the same register at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1142` through `:1189`.
- The XML exposes the HWREG access instructions but does not encode those
  permissions or spacing rules, as tracked in `CDNA1-XML-010`.

Rocjitsu evidence:

- CDNA1 `S_GETREG_B32` returns a few ad hoc values based on `reg_id`, and
  `S_SETREG_B32`/`S_SETREG_IMM32_B32` only handle ID 1 by splicing bits into
  `wf.status_raw()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopk.cpp:291` through
  `:384`.
- The same execution paths do not apply read-only rejection, MODE versus STATUS
  write policy, TRAPSTS side effects, live `IB_STS` counter reads, allocation
  subfield packing, or a consecutive-`S_SETREG` spacing diagnostic. The wrong
  ID map is tracked separately in `CDNA1-RJ-014`.

Impact:

Even if the ID map is fixed, CDNA1 HWREG execution would still not implement
the manual's per-register access contract, subfield contents, side effects, or
`S_SETREG` scheduling hazard.

### CDNA1-RJ-016: CDNA1 HWREG operand metadata over-sizes SIMM16 and hides the SETREG literal

Manual/XML evidence:

- Chapter 5.8 says the HWREG operand is a SIMM16 bitfield, and that
  `S_SETREG_IMM32_B32` is a 64-bit instruction whose data comes from a 32-bit
  literal constant at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1142` through `:1148`.
- The XML records `S_SETREG_IMM32_B32` with a 32-bit `OPR_SIMM32` literal
  operand at `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:45297`
  through `:45315`, but the HWREG operand-size inconsistency is tracked in
  `CDNA1-XML-009`.

Rocjitsu evidence:

- CDNA1 generated access constructors build the HWREG operand with size 32, not
  16, for `S_GETREG_B32`, `S_SETREG_B32`, and `S_SETREG_IMM32_B32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopk.cpp:280` through
  `:284`, `:323` through `:327`, and `:356` through `:359`.
- `S_SETREG_IMM32_B32` reads the extension word into `literal_` through
  `Sopk::hasImpliedLiteral()`, but its constructor exposes only the HWREG
  destination and sets `num_src_ = 0` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/encodings.cpp:68` through
  `:82` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopk.cpp:356`
  through `:363`.

Impact:

CDNA1 disassembly and analysis see the HWREG field as a 32-bit operand and do
not see the literal source for `S_SETREG_IMM32_B32`, even though execution does
consume the extension word.

### CDNA1-RJ-017: `S_MOVRELS` and `S_MOVRELD` truncate and scale the M0 scalar-register index

Manual/XML evidence:

- Chapter 5.7 summarizes `S_MOVRELS_{B32,B64}` and
  `S_MOVRELD_{B32,B64}` as indexing through `SGPR[S0+M0]` or `SGPR[D+M0]`, with
  M0 as an unsigned index and the 64-bit index required to be even, at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1132`.
- The detailed SOP1 rows use the raw formulas `addr += M0.u` for both source
  and destination relative moves at `cdna1/README.md:3061` through `:3064`.
- The XML exposes M0 as an implicit operand on the relative-move rows, as
  tracked in `CDNA1-XML-008`.

Rocjitsu evidence:

- CDNA1 `SMovrelsB32Sop1::execute_impl()` and `SMovrelsB64Sop1::execute_impl()`
  truncate M0 to eight bits with `wf.m0() & 0xFFu`, then compute
  `src_reg = ssrc0.encoding_value() + index * width_words`; for B64 this uses
  `base + 2 * M0` instead of `base + M0` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sop1.cpp:815` through
  `:856`.
- `SMovreldB32Sop1::execute_impl()` and `SMovreldB64Sop1::execute_impl()`
  apply the same eight-bit truncation and width scaling to the destination
  register at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sop1.cpp:873`
  through `:914`.
- Neither 64-bit path checks the manual's even-M0 requirement before forming
  the SGPR pair.

Impact:

Relative scalar moves with M0 values above 255 wrap locally, and 64-bit forms
address every other SGPR pair instead of using the manual's raw scalar-register
index. Odd-M0 64-bit forms are accepted rather than diagnosed as undefined.

### CDNA1-RJ-018: `S_SETPC_B64` and `S_SWAPPC_B64` zero targets halt instead of setting `PC`

Manual/XML evidence:

- Section 3.2 says `S_SET_PC` and `S_SWAP_PC` transfer PC to and from an
  even-aligned SGPR pair, and that PC operations are relative to the next
  instruction at `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:512`
  through `:516`.
- The XML describes `S_SETPC_B64` as jumping to the address specified in a
  scalar register, and `S_SWAPPC_B64` as storing the next-instruction address
  and then jumping to the scalar input, at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:33517` through `:33586`.

Rocjitsu evidence:

- The generated CDNA1 direct-PC execute bodies otherwise implement the normal
  operation: `S_GETPC_B64` writes `wf.pc + size_`, `S_SETPC_B64` assigns the
  scalar source minus `size_`, and `S_SWAPPC_B64` writes the next PC and then
  assigns the scalar source minus `size_` at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sop1.cpp:543`
  through `:594`.
- Before the instruction executes, `ComputeUnitCore::issue_instruction()` looks
  for any mnemonic containing `s_setpc` or `s_swappc`, reads the raw source
  SGPR pair, and halts/deletes the instruction when the target is zero at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:393`
  through `:407`.

Impact:

CDNA1 code that intentionally sets or swaps PC to address zero, or tests that
expect the architectural PC update before fault/termination handling, observes
a wave halt instead of the documented PC transfer.

### CDNA1-RJ-019: Raw STATUS helper bits are not maintained with EXEC/VCC state

Manual/XML evidence:

- Section 3.3 defines `EXECZ` as a helper bit for branch conditions at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:520` through `:526`.
- Section 3.4 says STATUS is initialized at wavefront creation and includes
  `SCC`, `EXECZ`, `VCCZ`, `HALT`, `TRAP`, `VALID`, and other read-only fields
  at `cdna1/README.md:530` through `:562`.
- Section 3.9 says VCCZ is updated every time VCC updates, including vector
  compares and scalar writes to VCC at `cdna1/README.md:687` through `:703`.

Rocjitsu evidence:

- `Wavefront::set_exec()` only updates `exec_`, and `Wavefront::set_vcc()` only
  updates `vcc_`, at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:227`
  and `:238`.
- The raw STATUS storage is separate: `IsaWavefront::status_raw()` and
  `set_status_raw()` access the `status` bitfield at `wavefront.h:610` through
  `:614`.
- Dispatch initializes `exec_`, `vcc_`, and `m0_` directly at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:139`
  through `:146`, while `Wavefront::reset()` does not reset the ISA-specific
  `status` member at `wavefront.h:464` through `:484`.
- CDNA1 scalar VCC/EXEC writes go through `set_vcc()`/`set_exec()` without
  synchronizing the raw helper bits at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/operand.cpp:1325`
  through `:1347` and `:1364` through `:1378`.

Impact:

Branch helpers and scalar special sources often use live EXEC/VCC directly, but
any fixed HWREG/status access or plugin/diagnostic consumer of `status_raw()`
can see stale `EXECZ`, `VCCZ`, `VALID`, `HALT`, or trap-related state even after
the architectural state changes.

### CDNA1-RJ-020: `S_SETVSKIP` decodes but VSKIP execution is absent

Manual/XML evidence:

- Section 3.3 says hardware does not optimize `EXEC=0` and points software to
  `CBRANCH` or `VSKIP` for fast skipping at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:520` through `:526`.
- The MODE table defines `VSKIP` bit 28 as skipping VALU, VMEM, LDS, and GDS
  instructions at `cdna1/README.md:565` through `:588`.
- The detailed `S_SETVSKIP` row says the instruction sets
  `VSKIP = S0.u[S1.u[4:0]]`, lists the skipped instruction classes, and notes
  that skipped memory instructions do not manipulate wait counters at
  `cdna1/README.md:3107`.
- The XML describes `S_SETVSKIP` as enabling or disabling VSKIP mode at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:44114` through `:44135`.

Rocjitsu evidence:

- CDNA1 `SSetvskipSopc::execute_impl()` throws `util::UnimplementedInst` at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopc.cpp:388`
  through `:409`.
- `ComputeUnitCore::issue_instruction()` executes the decoded instruction and
  then routes memory operations without checking MODE.VSKIP at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:393`
  through `:437`.

Impact:

CDNA1 shaders that rely on VSKIP cannot execute, and there is no fallback path
that skips the manual's vector instruction classes or suppresses skipped-memory
wait-counter changes.

### CDNA1-RJ-021: LDS allocation granularity and per-workgroup/M0 bounds are incomplete

Manual evidence:

- Section 3.6.1 says LDS out-of-range reads return zero, writes are discarded,
  destination VGPR out-of-range nullifies the instruction, and memory/LDS/GDS
  return operations must check all returned VGPRs at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:595` through `:634`.
- Section 3.6.5 says LDS is allocated in contiguous 128-dword blocks on
  128-dword alignment, does not wrap, all LDS accesses are restricted to the
  allocated space, and LDS address clamping uses the smaller of SPI LDS size
  and M0 at `cdna1/README.md:657` through `:659`.

Rocjitsu evidence:

- The command-processor planning helper aligns group LDS size to 256 bytes at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:99`
  through `:102`.
- CU admission and allocation also use 256-byte alignment at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:243`
  through `:249` and
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.h:239`
  through `:245`; the WGP placement path uses the same alignment at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/vm/amdgpu/spi.h:205` through
  `:235`.
- Generic DS address calculation adds the lane address and offset to
  `wf.lds_base()` without checking the per-workgroup allocation size or M0
  clamp at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:65`
  through `:83`.
- The local memory pipeline forwards those addresses to `Lds::vector_load()` and
  `Lds::vector_store()` at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:522`
  through `:593`; `Lds` bounds them only against the whole backing store at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/vm/amdgpu/lds.h:136` through
  `:169`.

Impact:

Rocjitsu can place CDNA1 workgroups at addresses that differ from the manual
512-byte LDS block allocation. LDS accesses can also reach beyond the
workgroup's requested LDS window, provided they remain inside the backing LDS
object, and they do not apply the documented `min(lds_size, M0)` clamp.

### CDNA1-RJ-022: Trap, exception, TRAPSTS, and memory-violation state is not modeled

Manual/XML evidence:

- Section 3.10 defines TTMP trap payload state, `STATUS.TRAP_EN`, and
  `MODE.EXCP_EN` trap gating at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:707` through `:733`.
- Section 3.10.1 defines sticky TRAPSTS fields at `cdna1/README.md:735`
  through `:749`.
- Section 3.11 defines memory-violation sources, sticky `TRAPSTS.mem_viol`,
  exception-enable behavior, and imprecise saved-PC reporting at
  `cdna1/README.md:752` through `:771`.
- XML identifies `S_RFE_B64`, `S_TRAP`, and the `TRAPSTS` HWREG selector at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:33596` through
  `:33605`, `:45917` through `:45935`, and `:94928`.

Rocjitsu evidence:

- CDNA1 `STrapSopp::execute_impl()` throws `util::UnimplementedInst` at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopp.cpp:301`
  through `:313`.
- CDNA1 `S_RFE_B64` dispatches to a shared helper at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sop1.cpp:608`,
  and that shared helper is empty at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2405`
  through `:2406`.
- CDNA1 `V_CLREXCP` decodes in both VOP1 and VOP3 forms, but both shared
  helpers are empty at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:3796`
  through `:3799`.
- A source search found no modeled `trapsts`, `mem_viol`, `SAVECTX`,
  `ILLEGAL_INST`, `ADDR_WATCH`, `EXCP_CYCLE`, or `DP_RATE` state outside names
  in generated operand/test tables.

Impact:

CDNA1 trap entry, trap return, sticky exception status, memory-violation
reporting, and exception clearing cannot be emulated or observed through
architectural state.

### CDNA1-RJ-023: TTMP privilege is not modeled

Manual evidence:

- Section 3.6.2 says a trap handler reserves 16 additional privileged SGPRs
  after VCC at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:636`.
- Section 3.10 says TTMP writes are privileged, writes are ignored when
  `status.priv` is clear, and TMA/TBA are read-only at `cdna1/README.md:707`
  through `:709`.

Rocjitsu evidence:

- CDNA1 scalar source resolution reads TTMP selector encodings 108 through 123
  directly from the wavefront SGPR block at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/operand.cpp:1158`
  through `:1159`.
- CDNA1 scalar destination writes and 64-bit destination writes also write TTMP
  selectors directly at `operand.cpp:1333` through `:1335` and `:1368` through
  `:1373`.
- The only architecture-specific launch payload initialization in
  `CommandProcessor::init_wavefront_regs()` is for RDNA4/gfx1250 TTMP7/TTMP9
  at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:288`
  through `:297`; this slice found no CDNA1 TTMP privilege gate.

Impact:

Unprivileged CDNA1 code can read and overwrite TTMP storage in rocjitsu instead
of observing trap-handler-only write access with ignored unprivileged writes.

### CDNA1-RJ-024: GDS backing, preload/writeback, and GDS operations are absent

Manual/XML evidence:

- Section 2.2.2 describes a 4 KiB GDS shared by wavefronts across all CUs, with
  integer atomics, preload before kernel launch, writeback after completion,
  unordered append/consume, and ordered append/consume support at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:452` through `:454`.
- CDNA1 XML records the DS `GDS` bit as `1=GDS, 0=LDS` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:2946` through
  `:2947`, and includes `DS_GWS_*`, `DS_CONSUME`, `DS_APPEND`, and
  `DS_ORDERED_COUNT` rows at `:13533` through `:13812`.

Rocjitsu evidence:

- Static source search found no modeled CDNA1 GDS backing store, preload, or
  completion writeback state under the rocjitsu VM.
- Generated CDNA1 DS execution rejects ordinary LDS/GDS dual-use operations
  when `inst_.gds` is set, for example `ds_add_u32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/ds.cpp:38` through `:40`
  and `ds_consume` / `ds_append` at `:4088` through `:4124`.
- Explicit GWS classes `DS_GWS_SEMA_RELEASE_ALL` through `DS_GWS_BARRIER`
  decode but throw `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/ds.cpp:3971` through
  `:4038`, and `DS_ORDERED_COUNT` also throws at `:4158` through `:4160`.

Impact:

CDNA1 programs that use GDS, GWS, or GDS-backed append/consume cannot execute
with the documented shared 4 KiB GDS state, preload/writeback behavior, or
cross-CU synchronization semantics in rocjitsu.

### CDNA1-RJ-025: Device-memory consistency and acknowledgment behavior is not represented

Manual evidence:

- Section 2.3 describes the CDNA device-memory hierarchy, cache-less loads,
  load-clause overlap caching, write-combining, atomic pre-op return storage,
  write-confirmation acknowledgments, relaxed consistency, per-PE/per-channel
  scatter-write ordering, and acknowledgment/fence use at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:458` through `:465`.

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
  `:378`.
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

### CDNA1-RJ-026: MUBUF/MTBUF `SOFFSET` special selectors are read as SGPRs

Manual/XML evidence:

- Chapter 9.1.2 says `SOFFSET` supplies an unsigned byte offset and must be an
  SGPR, `M0`, or inline constant at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1809`.
- Chapter 9.1.5 repeats that the SGPR offset comes from an SGPR or M0 at
  `cdna1/README.md:1910` through `:1914`.
- Generated CDNA1 operand metadata exposes buffer `SOFFSET` as
  `OPR_SSRC_NOLIT`, whose selector class includes `M0`, integer inline
  constants, and floating inline constants at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/operand_types.h:423`
  through `:456`.

Rocjitsu evidence:

- Generated MUBUF and MTBUF constructors expose `SOFFSET` as `OPR_SSRC_NOLIT`,
  for example `BUFFER_LOAD_FORMAT_X` and `TBUFFER_LOAD_FORMAT_X` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mubuf.cpp:36` through
  `:44` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mtbuf.cpp:36`
  through `:44`.
- The shared MUBUF address helper special-cases only selector `0x80` as zero,
  and otherwise reads `wf.sgpr_alloc().base + inst.soffset` as an SGPR at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h:63`
  through `:67`.
- MTBUF uses the same selector logic at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h:175`
  through `:178`.

Impact:

Legal buffer offsets using `m0` or inline constants compute addresses from an
unrelated SGPR index instead of the architectural scalar value.

### CDNA1-RJ-027: MUBUF formatted load/store opcodes decode but throw

Manual/XML evidence:

- Chapter 9.1.4 says MUBUF `BUFFER_LOAD_FORMAT_*` and
  `BUFFER_STORE_FORMAT_*` use the resource data format and `dst_sel`, while
  other MUBUF instructions derive format from the opcode at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1848` through `:1870`.
- The detailed MUBUF table defines formatted D16 and D16_HI opcodes at
  `cdna1/README.md:4847` through `:4859` and `:4892` through `:4894`.

Rocjitsu evidence:

- Generated `BUFFER_LOAD_FORMAT_{X,XY,XYZ,XYZW}` and
  `BUFFER_STORE_FORMAT_{X,XY,XYZ,XYZW}` constructors expose operands, but their
  execute bodies throw `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mubuf.cpp:36` through
  `:165`.
- The formatted D16 and D16_HI MUBUF variants likewise throw, for example at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mubuf.cpp:236` through
  `:253` and `:1362` through `:1403`.

Impact:

Raw untyped MUBUF load/store opcodes have partial execution coverage, but the
resource-format MUBUF surface described by Chapter 9 cannot execute.

### CDNA1-RJ-028: MTBUF typed format conversion ignores `DFMT` and `NFMT`

Manual/XML evidence:

- Chapter 9.1 says MTBUF data format is specified in the instruction and MTBUF
  only provides load/store operations with data format conversion at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1775` through `:1778`.
- Chapter 9.1.4 says MTBUF derives data and numeric formats from the
  instruction fields at `cdna1/README.md:1848` through `:1870`.
- The XML `ENC_MTBUF` field map carries `DFMT` and `NFMT` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:3269` through `:3310`.

Rocjitsu evidence:

- Generated `TBUFFER_LOAD_FORMAT_X` sets `elem_size = 4` and `num_elems = 1`,
  routes through the generic memory pipeline, and does not consult `inst_.dfmt`
  or `inst_.nfmt` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mtbuf.cpp:54` through
  `:64`.
- Generated `TBUFFER_STORE_FORMAT_X` stores raw 32-bit VGPR payloads with fixed
  `elem_size = 4`, without typed conversion, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mtbuf.cpp:178` through
  `:197`.
- D16 MTBUF loads set a fixed 2-byte element size and `d16_lo`, but still do
  not use `DFMT`/`NFMT` conversion metadata, for example at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mtbuf.cpp:350` through
  `:390`.

Impact:

MTBUF instructions perform raw fixed-width transfers instead of the typed
texture-format conversion required by the instruction's `DFMT`/`NFMT` fields.

### CDNA1-RJ-029: Buffer resource swizzle, add-TID, and descriptor modes are not modeled

Manual evidence:

- Chapter 9.1.5 defines descriptor fields for base, stride, num-records,
  add-TID, swizzle enable, element size, and index stride at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1894` through `:1906`.
- The same section defines private/scratch, raw, and structured range-check
  behavior at `cdna1/README.md:1926` through `:1949`, and a swizzled-address
  formula at `:1951` through `:1969`.
- Chapter 9.1.8 defines descriptor fields including `dst_sel`, numeric/data
  format, user-VM behavior, add-TID, `NV`, type, reserved bits, and all-zero
  unbound behavior at `cdna1/README.md:2010` through `:2040`.

Rocjitsu evidence:

- The shared MUBUF helper reads `base`, `stride`, `num_records`, and `srd3`,
  but only uses `stride`, `num_records`, and `srd3 >> 31` to choose a
  simplified raw/structured OOB mode at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h:57`
  through `:133`.
- The same helper always computes a linear
  `index * stride + voffset + inst_offset + soffset` address and does not
  implement add-TID, stride-extension, swizzled decomposition,
  element-size/index-stride constraints, `dst_sel`, user-VM/unbound, `NV`, or
  descriptor-type rules at `addr_calc_buffer.h:84` through `:133`.
- MTBUF shares the same simplified address and OOB logic at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h:169`
  through `:219`.

Impact:

Common linear raw/structured cases have partial coverage, but descriptor-driven
buffer modes can produce wrong addresses, wrong OOB masks, or ordinary memory
traffic where the manual requires zero/drop/unbound behavior.

### CDNA1-RJ-030: Vector-buffer cache policy collapses `GLC`/`SLC` behavior

Manual evidence:

- Chapter 9.1.2 says `GLC` controls load L1 persistence, store
  write-combine/L1 persistence, and atomic pre-operation return behavior, and
  `SLC` sets L2 streaming/non-temporal mode at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1819` through `:1820`.
- Section 9.1.10 expands operation-specific `GLC` behavior and says floating
  point atomics must use `GLC == 0` at `cdna1/README.md:2068` through `:2094`.

Rocjitsu evidence:

- CDNA1 MUBUF/MTBUF execute bodies use `mtype_from_flags_gfx9(inst_.glc)` and
  set `non_temporal = 0`, for example at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mubuf.cpp:640` through
  `:655` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mtbuf.cpp:54`
  through `:64`.
- `mtype_from_flags_gfx9()` maps only `GLC` to `Mtype::CC` versus `Mtype::RW`
  and explicitly treats `SLC` as not changing simulator `Mtype` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/gfx9_cache_flags.h:7`
  through `:29`.
- Atomic return selection via `GLC` is present separately, for example
  `BUFFER_ATOMIC_ADD` sets `d->is_load = (inst_.glc != 0)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mubuf.cpp:1556` through
  `:1566`; this gap is about the broader load/store cache policy and `SLC`
  behavior.

Impact:

Rocjitsu does not model CDNA1's operation-specific L1 persistence,
write-combine, L2 streaming, cache-bypass distinctions, or the documented
floating-point atomic `GLC` restriction.

### CDNA1-RJ-031: Buffer-to-LDS and LDS-store semantics are incomplete

Manual/XML evidence:

- Chapter 9.1.9 limits load-to-LDS to
  `BUFFER_LOAD_{ubyte,sbyte,ushort,sshort,dword,format_x}`, says `TFE` is
  illegal for loads to LDS, derives the LDS offset from `M0[15:0]`, and
  requires active-mask clamping so return data is not written outside the
  wave's allocated LDS space at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2042` through `:2066`.
- The detailed MUBUF table defines `BUFFER_STORE_LDS_DWORD`, `BUFFER_WBINVL1`,
  and `BUFFER_WBINVL1_VOL` at `cdna1/README.md:4895` through `:4897`.

Rocjitsu evidence:

- Generated CDNA1 `BUFFER_LOAD_DWORDX2` and `BUFFER_LOAD_DWORDX3` accept
  `inst_.lds` and route directly to LDS, even though Chapter 9.1.9 lists only
  the scalar `dword` form from the Dword family, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mubuf.cpp:640` through
  `:645` and `:678` through `:690`.
- D16 raw load forms also route `inst_.lds` to LDS without the D16 low/high
  VGPR semantics or an explicit Chapter 9.1.9 legality check, for example at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mubuf.cpp:1102` through
  `:1115` and `:1148` through `:1161`.
- The LDS destination writer uses `d.lds_base + lane * per_lane_bytes` or a
  precomputed per-lane LDS address, but this CDNA1 buffer path sets
  `d->lds_base = wf.m0() + wf.lds_base()` rather than masking `M0[15:0]`, and
  this slice did not find an active-mask clamp to the allocated LDS range at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:60` through `:72`
  and `:124` through `:127`.
- `BUFFER_STORE_LDS_DWORD` decodes `SRSRC` and `SOFFSET`, but its execute body
  throws `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mubuf.cpp:1406` through
  `:1420`.
- `BUFFER_WBINVL1` and `BUFFER_WBINVL1_VOL` both invalidate the entire vector
  L1 and flush L2 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mubuf.cpp:1423` through
  `:1447`, while the manual distinguishes volatile-line behavior and ACK
  semantics.

Impact:

The load-to-LDS path is useful for common raw loads, but it accepts forms that
the manual does not list, misses `TFE`/M0/clamping details, cannot execute the
LDS-to-memory store opcode, and collapses cache-maintenance variants.

### CDNA1-RJ-032: Chapter 9 vector-buffer tests are mostly decode/operand smoke coverage

Evidence:

- `decode_smoke_test.cpp` has a MUBUF `lds` decode/modifier smoke test at
  `emulation/rocjitsu/tests/decode_smoke_test.cpp:726` through `:779`.
- `buffer_operand_test.cpp` checks descriptor/register operand rendering for
  representative CDNA1 MUBUF and MTBUF instructions at
  `emulation/rocjitsu/tests/buffer_operand_test.cpp:105` through `:122`.
- This slice did not find CDNA1-targeted execution tests for `SOFFSET` `m0` or
  inline selectors, MUBUF formatted opcodes, MTBUF `DFMT`/`NFMT`, descriptor
  add-TID/swizzle/unbound modes, `SLC` cache behavior, or the Chapter 9.1.9
  load-to-LDS legal subset and clamping rules.

Impact:

Existing tests can catch decode/regression drift for representative buffer
forms, but most CDNA1 vector-buffer semantics above can regress without a
targeted CDNA1 execution failure.

### CDNA1-RJ-033: CDNA1 MIMG image instructions decode as no-op execution stubs

Manual/XML evidence:

- Chapter 9.2 says MIMG `IMAGE_*` and `SAMPLE_*` instructions read, write,
  sample, or atomically update image objects through the texture cache at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2098` through `:2106`.
- Chapter 9.2.1 lists CDNA1 `IMAGE_LOAD*`, `IMAGE_STORE*`,
  `IMAGE_ATOMIC_*`, and `IMAGE_SAMPLE` opcodes at `cdna1/README.md:2112`
  through `:2119`; Chapter 13.6 repeats the MIMG opcode table starting at
  `cdna1/README.md:7366`.
- The XML has corresponding `ENC_MIMG` instruction entries, for example
  `IMAGE_LOAD`, `IMAGE_STORE`, `IMAGE_ATOMIC_SWAP`, and `IMAGE_SAMPLE` at
  `amdgpu_isa_cdna1.xml:20101`, `:20353`, `:20563`, and `:21109`.

Rocjitsu evidence:

- Generated `IMAGE_LOAD*` constructors expose operands, but their execute
  bodies only ignore the wavefront, for example `IMAGE_LOAD` through
  `IMAGE_LOAD_MIP_PCK_SGN` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mimg.cpp:19` through
  `:123`.
- Generated `IMAGE_STORE*` forms similarly return from placeholder bodies, for
  example `IMAGE_STORE` through `IMAGE_STORE_MIP_PCK` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mimg.cpp:126` through
  `:195`.
- `IMAGE_GET_RESINFO`, image atomics, and `IMAGE_SAMPLE` are also
  image-pipeline no-op stubs at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mimg.cpp:198` through
  `:230`, `:215` through `:446`, and `:449` through `:465`.
- The code generator explicitly emits image-pipeline stubs for `image_load`,
  `image_store`, `image_atomic`, `image_sample`, and `image_query` classes at
  `lib/python/amdisa/codegen/_generator.py:4241` through `:4256`.

Impact:

Decoded CDNA1 image instructions do not read texture memory, write image
memory, return texel/resource data, perform sampler filtering, apply image
atomics, or produce architectural data/status VGPR results.

### CDNA1-RJ-034: MIMG operand widths are fixed instead of derived from `DMASK`, opcode, and resource shape

Manual evidence:

- Chapter 9.3 defines no-sampler address VGPR layouts by opcode, resource
  dimension, array declaration, mip level, and MSAA/cube fields at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2143` through `:2170`.
- Chapter 9.4 and 9.4.1 define sampler address component ordering, derivative
  address groups, data VGPR use, read/write `DMASK` behavior, atomic legal
  `DMASK` values, atomic return placement, and D16 packing at
  `cdna1/README.md:2172` through `:2276`.

Rocjitsu evidence:

- Generated no-sampler image load/store constructors use fixed 128-bit `VDATA`
  and `VADDR` operands for representative image loads and stores, for example
  `IMAGE_LOAD` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mimg.cpp:19` through `:33`
  and `IMAGE_STORE` at `:126` through `:140`.
- Generated image atomics expose fixed 128-bit `VDATA` and `VADDR` operands,
  even for forms whose data count is controlled by atomic width and `DMASK`;
  `IMAGE_ATOMIC_SWAP` and `IMAGE_ATOMIC_CMPSWAP` are representative at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mimg.cpp:215` through
  `:248`.
- `IMAGE_SAMPLE` exposes a fixed 96-bit `VADDR` and fixed 128-bit `VDATA` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mimg.cpp:449` through
  `:465`, while other generated sample variants use fixed widened address
  spans such as 288 and 320 bits at `:487` through `:522`.

Impact:

Def-use, disassembly, dependency tools, and any future execution path will see
coarse fixed VGPR footprints rather than the image opcode's actual address and
data footprint. That hides `DMASK`, D16, A16, atomic width, and resource
dimension effects.

### CDNA1-RJ-035: MIMG resource and sampler operands are not scaled to aligned SGPR bases

Manual/XML evidence:

- Chapter 9.2.1 says `SSAMP` and `SRSRC` omit the low two SGPR-address bits
  because the sampler and resource constants are aligned to multiples of four
  SGPRs at `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2128` through
  `:2129`.
- Chapter 13.6 repeats that `SRSRC` and `SSAMP` are missing two low SGPR
  address bits at `cdna1/README.md:7362` through `:7363`.
- The XML MIMG field map likewise describes `SRSRC` and `SSAMP` as SGPRs in
  units of four and includes two padding bits for each field at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:3631` through `:3655`.

Rocjitsu evidence:

- Generated MIMG constructors pass the raw 5-bit field directly as an
  `OPR_SREG`, for example `IMAGE_LOAD` uses `inst->srsrc` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mimg.cpp:19` through `:23`
  and `IMAGE_SAMPLE` uses raw `inst->srsrc` and `inst->ssamp` at `:449`
  through `:455`.
- `Operand::name()` treats an `OPR_SREG` encoding value as the actual SGPR
  number at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/operand.cpp:572`
  through `:575`.
- Other CDNA1 resource-base encodings do apply the missing-bit scale: MUBUF and
  MTBUF construct `srsrc` from `inst->srsrc * 4` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mubuf.cpp:42` and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mtbuf.cpp:42`.

Impact:

MIMG disassembly, register-reference metadata, and any future descriptor read
can point at `sN` instead of the architected `s[4*N:4*N+...]` base. A raw field
value of 2 should name an aligned descriptor starting at `s8`, not `s2`.

### CDNA1-RJ-036: MIMG disassembly omits semantic modifier fields

Manual/XML evidence:

- Chapter 9.2.1 lists user-visible MIMG fields including `DMASK`, `UNRM`,
  `GLC`, `SLC`, `TFE`, `LWE`, `DA`, `A16`, and `D16` at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2121` through `:2141`.
- Chapter 13.6 records the raw MIMG field map, including `DMASK`, `UNRM`,
  `GLC`, `DA`, `A16`, `TFE`, `LWE`, `SLC`, `SRSRC`, `SSAMP`, and `D16`, at
  `cdna1/README.md:7348` through `:7364`.

Rocjitsu evidence:

- `MimgMachineInst` stores the raw fields, including `dmask`, `unorm`, `glc`,
  `da`, `a16`, `tfe`, `lwe`, `slc`, `ssamp`, and `d16`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/machine_insts.h:113`
  through `:131`.
- `Mimg` only declares the constructor and stored encoding and does not
  override `build_modifiers` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/encodings.h:609` through
  `:614`.
- The base disassembler appends modifiers only through `build_modifiers()`,
  whose default implementation is empty at
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:239` through `:274`.

Impact:

Distinct MIMG encodings that differ in `DMASK`, cache/status/address/data
modifiers, or D16 state can disassemble identically, which obscures the
semantic contract and makes decode-oriented tests too weak.

### CDNA1-RJ-037: Image resource and sampler descriptors are opaque and unused

Manual evidence:

- Chapter 9.2 says each image operation sends a 256-bit resource constant that
  defines address, data format, and surface characteristics, and sample
  operations also send a 128-bit sampler constant at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2102` through `:2104`.
- Chapter 9.4.2 and 9.4.3 define image resource and sampler descriptor
  bitfields at `cdna1/README.md:2278` through `:2372`.

Rocjitsu evidence:

- Generated MIMG constructors expose `SRSRC` as a fixed 256-bit `OPR_SREG`
  operand, for example `IMAGE_LOAD` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mimg.cpp:19` through `:33`.
- `IMAGE_SAMPLE` exposes both `SRSRC` and `SSAMP`, with `SSAMP` as a fixed
  128-bit `OPR_SREG`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mimg.cpp:449` through
  `:465`.
- The generated image execute bodies are stubs and never read or parse resource
  or sampler descriptor fields, as recorded in `CDNA1-RJ-033`.

Impact:

Rocjitsu cannot model resource dimensions, formats, destination selectors,
tiling, mip/LOD, metadata/compression, PRT behavior, sampler clamp/filter/depth
compare, or force-unnormalized address behavior for CDNA1 image instructions.

### CDNA1-RJ-038: Image instructions do not issue countered VMEM work

Manual evidence:

- Chapter 9.4.5 says VMEM image instructions immediately read address VGPRs and
  image/sampler resources, while write data is sent later, and that consumers
  must use `VMCNT` waits before reading texture-cache results at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2418` through `:2424`.

Rocjitsu evidence:

- Rocjitsu's memory pipeline increments a wait counter when memory work is
  issued and releases it on completion at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.h:55` through `:88`
  and `:104` through `:107`.
- CDNA1 image execute bodies return from stubs and never create vector-memory
  work or call the global/vector memory pipeline, as shown by representative
  load, store, atomic, query, and sample bodies in `CDNA1-RJ-033`.
- `coverage_exceptions_cdna1.txt` records `image_*` as execution-incomplete
  because image sampling/query is not simulated at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/coverage_exceptions_cdna1.txt:27`.

Impact:

`S_WAITCNT` can observe no outstanding image work because image instructions do
not issue countered VMEM operations. This misses both texture-result
availability and the manual's address/resource/read-data timing distinction.

### CDNA1-RJ-039: Chapter 9 image tests are decode-only or absent

Evidence:

- The only image-specific decode smoke case found in this slice is a CDNA2 ACC
  destination selection test for `image_load` at
  `emulation/rocjitsu/tests/decode_smoke_test.cpp:864` through `:895`.
- The CDNA1 coverage exception marks `image_*` execution incomplete at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/coverage_exceptions_cdna1.txt:27`.
- Searches of `emulation/rocjitsu/tests` and `lib/python/amdisa/tests` found
  semantic derivation/unit tests for image instruction classification, but no
  CDNA1 image execution tests for `DMASK`, D16/A16, resource/sampler
  descriptors, image atomics, cache/status bits, SGPR descriptor-base scaling,
  or `VMCNT` behavior.

Impact:

The suite can catch selected MIMG classification or non-CDNA1 decode drift, but
it would not catch the CDNA1 execution, descriptor, data-format, dependency,
SGPR-base, or disassembly gaps above.

### CDNA1-RJ-040: Flat aperture routing and wait-counter accounting collapse the dual-path contract

Manual evidence:

- Chapter 10.2 says FLAT instructions internally execute as both LDS and Buffer
  requests, increment both `VM_CNT` and `LGKM_CNT`, and are not complete until
  both counters decrement; the only sensible wait after FLAT is `S_WAITCNT 0`
  at `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2507` through `:2519`.
- Chapter 10.4 says Global instructions use only `VM_CNT`, and that a Global
  instruction accessing LDS returns `MEM_VIOL`, at `cdna1/README.md:2560`
  through `:2562`.

Rocjitsu evidence:

- Every generated CDNA1 flat/global/scratch load/store/atomic path constructs
  a `VectorMemState(GLOBAL_MEM)` and sets `d->wait_counter_type =
  WaitCounterType::VMCNT`; representative load and D16 load paths are at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/flat.cpp:50` through
  `:60` and `:768` through `:779`.
- `ComputeUnitCore::route_memory_inst()` samples only the first active lane to
  decide whether a `GLOBAL_MEM` instruction targets the shared aperture, then
  rewrites every active lane as LDS, changes the tag to `LOCAL_MEM`, and
  switches the single wait counter to `LGKMCNT` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:260` through `:282`.
- The memory issue path routes one memory state to one pipeline after
  execution at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:430`
  through `:435`.

Impact:

FLAT requests are modeled as either VM or LDS, not the manual's dual
VM/LGKM request. Mixed-lane flat accesses can route all lanes according to the
first active lane, and Global instructions that land in the shared aperture are
routed to LDS instead of reporting the manual's memory violation.

### CDNA1-RJ-041: Scratch address calculation gates the VGPR offset on `LDS`

Manual evidence:

- Table 47 says the `LDS` bit controls direct LDS-memory data movement and is
  only valid for Global and Scratch; Table 48 says `SADDR` selects an SGPR
  address instead of a VGPR address for Scratch at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2443` through `:2452`.
- Section 10.5 says Scratch addresses use
  `flat_scratch.addr + swizzle(V/SGPR_offset + inst_offset, threadID)`, and
  that the offset can come from either an SGPR or a VGPR and is a 32-bit
  unsigned byte offset at `cdna1/README.md:2570` through `:2574`.

Rocjitsu evidence:

- Generated CDNA1 scratch constructors expose a 32-bit VGPR address operand
  and optionally a 32-bit SGPR address operand when `inst_.seg == 1`; the
  representative `FlatLoadUbyteFlat` constructor does this at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/flat.cpp:26` through
  `:45`.
- The shared flat address helper sets `has_vaddr = (inst.lds == 1)` for
  scratch encodings that do not have an `SVE` field, and only reads the VGPR
  offset when `has_vaddr` is true at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h:59`
  through `:89`.

Impact:

Ordinary CDNA1 Scratch encodings with `LDS=0` can ignore the VGPR offset even
though the manual describes the offset source as VGPR or SGPR independent of
the direct-LDS data-movement bit.

### CDNA1-RJ-042: Global SGPR-base addressing sign-extends the VGPR offset

Manual evidence:

- Section 10.4 defines the Global `SADDR` form as SGPR address plus VGPR offset
  plus instruction offset, and says the VGPR offset is always 32 bits at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2549` through `:2558`.
- Chapter 13.8.1 describes the single-VGPR `ADDR` offset as a 32-bit unsigned
  offset and the `SADDR` value as an unsigned address or offset at
  `cdna1/README.md:7496` through `:7498`.

Rocjitsu evidence:

- The shared flat address helper handles Global `saddr != 0x7f` by reading one
  VGPR dword and sign-extending it through `int32_t` before adding it to the
  SGPR base at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h:90`
  through `:111`.

Impact:

Global SGPR-base accesses whose VGPR offset has bit 31 set are interpreted as
negative offsets. The manual assigns signedness to the 13-bit instruction
`OFFSET`, not to the 32-bit VGPR offset, so high unsigned offsets can target
the wrong address.

### CDNA1-RJ-043: Global/scratch direct-LDS forms are not generated for CDNA1 flat memory

Manual evidence:

- Chapter 10.1 says `LDS` enables direct data movement between LDS and memory
  for Global and Scratch instructions and must be zero for Flat instructions at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2443`.
- Table 47 says `M0` supplies the LDS offset for Scratch/Global when `LDS=1`
  at `cdna1/README.md:2452`.
- Sections 10.4 and 10.5 repeat that Global and Scratch can move data directly
  between LDS and memory at `cdna1/README.md:2560` and `:2577`.

Rocjitsu evidence:

- Representative generated CDNA1 flat load/store code uses VGPR destinations or
  VGPR data operands and constructs a `VectorMemState(GLOBAL_MEM)`, for
  example `FlatLoadUbyteFlat` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/flat.cpp:26` through `:60`.
- The vector-memory state has explicit LDS-destination fields such as
  `lds_dst`, `lds_base`, and `lds_per_lane_addr` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:112` through `:119`, but
  searches of generated CDNA1 `flat.cpp` and `flat.h` found no `lds_dst`,
  `lds_per_lane_addr`, `GLOBAL_LOAD_LDS`, or `SCRATCH_LOAD_LDS` path.

Impact:

CDNA1 Global/Scratch encodings with `LDS=1` do not get the manual's
M0-relative direct LDS transfer semantics. They execute through the ordinary
VGPR memory path, so direct-LDS flat-family behavior is not currently
emulated.

### CDNA1-RJ-044: Memory-violation and flat-memory trap behavior is absent

Manual evidence:

- Chapter 10.6 says invalid addresses, read-only writes, misalignment, and
  out-of-range LDS or scratch accesses report memory errors; writes are
  dropped, reads return zero, and the wave MEM_VIOL trap-status bit is set at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2581` through `:2597`.
- Chapter 10.4 says Global instructions attempting LDS access return
  `MEM_VIOL` at `cdna1/README.md:2562`, while Chapter 10.5 says Scratch
  performs no aperture checking and no error reporting at `:2579`.

Rocjitsu evidence:

- The shared flat address helper maps private-aperture Flat addresses into the
  scratch backing buffer but has no scratch-size or memory-error check at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h:113`
  through `:132`.
- `ComputeUnitCore::route_memory_inst()` maps shared-aperture `GLOBAL_MEM`
  requests into LDS instead of producing a trap-status side effect or zeroed
  read at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:260` through
  `:282`.
- Searches of `lib/rocjitsu/src/rocjitsu` found no `MEM_VIOL`, `TrapStatus`,
  `TRAPSTS`, or `EXCPEN` implementation hooks for flat-memory errors.

Impact:

Negative flat/global/scratch memory cases execute as ordinary memory traffic
instead of applying CDNA1's defined drop/zero/trap-status behavior. This also
makes Global-to-LDS and out-of-range LDS/scratch cases untestable against the
manual contract.

### CDNA1-RJ-045: Flat decoder accepts reserved `SEG` and scratch atomic opcode combinations

Manual/XML evidence:

- The Chapter 10 field table defines `SEG` value 3 as reserved at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2442`.
- Chapter 10 and Chapter 13.8 define Scratch instructions as load/store plus
  D16 load rows only; Scratch has no atomic opcodes at
  `cdna1/README.md:2481` through `:2499` and `:7630` through `:7658`.
- The XML has a distinct `ENC_FLAT_SCRATCH` encoding with only scratch
  load/store rows; representative scratch rows start at
  `amdgpu_isa_cdna1.xml:18889`.

Rocjitsu evidence:

- CDNA1 `Decoder::subDecodeFlat()` reinterprets every FLAT-format word as
  `Flat::OpEncoding` and indexes `sub_decode_flat` by `op` only at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/decoder.cpp:4957`
  through `:4959`.
- The shared flat opcode table maps atomic opcodes 64 through 76 and 96
  through 108 to `FlatAtomic*` constructors regardless of `SEG` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/decoder.cpp:9027`
  through `:9071`.
- The generated `Flat` constructor rewrites any `flat_*` mnemonic to
  `scratch_*` when `SEG == 1` and `global_*` when `SEG == 2` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/encodings.cpp:14`
  through `:20` and `:310` through `:312`.
- The shared address helper treats only `SEG == 1` as Scratch and `SEG == 2`
  as Global; any other segment, including reserved `SEG == 3`, falls into the
  Flat path at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h:59`
  through `:132`.

Impact:

Encodings with `SEG=1` and atomic opcodes decode as unsupported
`scratch_atomic_*` instructions even though Scratch has no atomic rows in the
manual or XML. Encodings with reserved `SEG=3` are accepted and executed as
Flat-like operations instead of being rejected as invalid.

### CDNA1-RJ-046: CDNA1 Global floating-point atomics decode as invalid

Manual/XML evidence:

- The Chapter 10 opcode table lists `GLOBAL_ATOMIC_ADD_F32` and
  `GLOBAL_ATOMIC_PK_ADD_F16` as Global-only opcodes at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2497` through `:2498`;
  Chapter 13.8 repeats them at `cdna1/README.md:7609` through `:7610`.
- The detailed instruction rows describe the F32 and packed-F16 memory updates
  at `cdna1/README.md:5283` through `:5284`.
- The XML exposes `GLOBAL_ATOMIC_ADD_F32` and
  `GLOBAL_ATOMIC_PK_ADD_F16` in `ENC_FLAT_GLBL` at
  `amdgpu_isa_cdna1.xml:17899` through `:18010`.

Rocjitsu evidence:

- CDNA1 `sub_decode_flat` maps opcode 77 and opcode 78 to `decodeInvalid` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/decoder.cpp:9039`
  through `:9041`, so the legal Global FP atomic encodings are not decoded
  when `SEG == 2`.
- The generated CDNA1 flat opcode constants include integer flat atomics 64
  through 76 and 96 through 108, but no flat/global FP atomic constants, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/opcodes.h:1406` through
  `:1431`.
- Searches of generated CDNA1 `flat.cpp`, `flat.h`, and `decoder.cpp` found no
  `GLOBAL_ATOMIC_ADD_F32` or `GLOBAL_ATOMIC_PK_ADD_F16` constructors or execute
  paths, while CDNA1 MUBUF does generate buffer FP atomic constants at
  `opcodes.h:1260` through `:1261`.

Impact:

Legal CDNA1 Global F32 and packed-F16 atomic instructions decode as invalid,
so rocjitsu cannot emulate their memory update, no-return `GLC=0` contract,
or FP denormal/rounding behavior.

### CDNA1-RJ-047: Global and scratch signed instruction offsets disassemble as unsigned raw values

Manual/XML evidence:

- Chapter 10.1 defines `OFFSET` as a 13-bit signed byte offset for Scratch and
  Global but a 12-bit unsigned offset for Flat at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2445`.
- Chapter 13.8.1 repeats the segment-specific signedness at
  `cdna1/README.md:7489`.
- The XML repeats `ENC_FLAT_GLBL` and `ENC_FLAT_SCRATCH` `OFFSET` as a
  13-bit signed offset at `amdgpu_isa_cdna1.xml:3531` through `:3537` and
  `:3686` through `:3691`.

Rocjitsu evidence:

- The shared flat address helper sign-extends the 13-bit Scratch/Global offset
  for execution and keeps Flat unsigned at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h:46`
  through `:57`.
- CDNA1 `Flat::build_modifiers()` reconstructs the Scratch/Global offset as a
  raw 13-bit integer and prints it with `std::to_string(flat_offset)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/encodings.cpp:320`
  through `:325`; it does not sign-extend the display value.

Impact:

Execution can use the correct signed address offset while disassembly still
prints the wrong modifier. A Global or Scratch encoding with raw
`OFFSET=0x1000` should display `offset:-4096`, but rocjitsu prints
`offset:4096`.

### CDNA1-RJ-048: Flat-memory cache, LDS, and NV modifiers are only partially surfaced

Manual/XML evidence:

- Chapter 10.1 says `SLC` works with `GLC` to determine cache policy, `GLC`
  also controls atomic return data, `LDS` selects LDS-memory transfer for
  Global/Scratch, and `NV` marks non-volatile memory operations at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2440` through `:2444`.
- Chapter 13.8.1 repeats `GLC`, `SLC`, and `NV` field meanings at
  `cdna1/README.md:7492` through `:7499`.
- The generated CDNA1 machine-instruction struct stores `glc`, `slc`, `lds`,
  and `nv` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/machine_insts.h:148`
  through `:162`.

Rocjitsu evidence:

- Generated CDNA1 flat execution maps only `inst_.glc` into `mtype` and sets
  `d->non_temporal = 0` in representative load and atomic paths at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/flat.cpp:50` through
  `:60` and `:2242` through `:2252`.
- The shared CDNA1/CDNA2 cache helper documents that `SLC` does not change
  modeled `Mtype` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/gfx9_cache_flags.h:7`
  through `:29`.
- `Flat::build_modifiers()` prints only `offset`, `glc`, and `slc`; it never
  reports `LDS` or `NV` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/encodings.cpp:320`
  through `:330`.

Impact:

Flat/global/scratch encodings that differ in `SLC`, `LDS`, or `NV` largely
collapse to the same runtime behavior and, for `LDS`/`NV`, the same
disassembly. This hides direct-LDS and non-volatile controls from decode tests
and from the memory model.

### CDNA1-RJ-049: Chapter 10 flat-memory tests miss routing, legality, and FP atomic edge cases

Evidence:

- The adjacent decode smoke coverage for `global_load_dword` and
  `scratch_load_dword` is a CDNA2 ACC-destination test, not a CDNA1 Chapter 10
  execution test, at `emulation/rocjitsu/tests/decode_smoke_test.cpp:864`
  through `:880`.
- Shared flat address helper tests cover CDNA4 scratch/global cases and RDNA3
  address cases at `emulation/rocjitsu/tests/shared_infra_test.cpp:2761`
  through `:2842` and `:2867` through `:2933`, but this slice did not find
  equivalent CDNA1 flat/global/scratch execution coverage.
- Searches of `emulation/rocjitsu/tests` and `lib/python/amdisa/tests` found no
  CDNA1 execution tests for FLAT dual `VM_CNT`/`LGKM_CNT` completion,
  mixed-lane aperture routing, Global-to-LDS `MEM_VIOL`, Scratch `LDS=0` VGPR
  offsets, high-bit Global VGPR offsets, Scratch/Global direct-LDS forms,
  reserved `SEG`, scratch atomic rejection, Global FP atomic decode/execution,
  signed-offset disassembly, or Chapter 10 memory-error read-zero/drop/trap
  behavior.

Impact:

The current test surface can catch broad helper drift and non-CDNA1 decode
regressions, but it would not catch the CDNA1 Chapter 10 behavioral and
legality gaps above.

### CDNA1-RJ-050: Indexed and atomic LDS address calculation ignores M0 clamp and bank behavior

Manual/XML evidence:

- Chapter 11.1 defines the CDNA1 LDS as 32 banks and says indexed/atomic bank
  conflicts are serialized at `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2624`
  through `:2633`.
- Section 11.3.3 requires M0 to be initialized and says it can restrict the LDS
  range at `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2736`.
- XML exposes the DS fields but not the clamp or bank timing rules at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:2178` through `:2406`.

Rocjitsu evidence:

- `addr_calc::ds_calculate_addresses()` computes
  `VGPR[ADDR] + ((offset1 << 8) | offset0) + wf.lds_base()` and never reads
  `wf.m0()` or clamps against an M0 LDS-size field at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:60`
  through `:76`.
- `LocalMemPipeline::initiate_access()` issues LDS vector loads, stores, and
  atomics through the backing LDS object without an LDS bank/conflict timing
  model at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:506`
  through `:596`.

Impact:

Out-of-range LDS accesses can execute without the manual's M0 clamp, and
bank-conflict behavior is invisible to timing or hazard-sensitive tests.

### CDNA1-RJ-051: CDNA1 DS SRC2 opcodes are absent from generated decode and execution

Manual/XML evidence:

- Section 11.3.3 defines `DS_*_SRC2_*` addressing as a special two-LDS-operand
  atomic form with signed dword offsets and an `offset1[7]` per-thread mode at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2777` through `:2789`.
- The detailed table lists U32, B32, F32, U64, B64, and F64 SRC2 forms at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:4634` through `:4692`.
- Searches found no `DS_*_SRC2_*` rows in the CDNA1 XML.

Rocjitsu evidence:

- Searches of generated CDNA1 `ds.cpp`, `decoder.cpp`, `opcodes.h`, and
  `test_encodings.h` found no SRC2 instruction classes, decode methods, opcode
  constants, or fixtures.
- The generated DS decoder table contains ordinary opcodes, GWS, ADDTID,
  append/consume, B96/B128, and related rows around
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/decoder.cpp:8702` through
  `:8957`, but no SRC2 slots.

Impact:

Legal CDNA1 SRC2 DS encodings cannot decode or execute in rocjitsu, and the
special signed/per-thread LDS address contract is not represented.

### CDNA1-RJ-052: `DS_WRITE_ADDTID_B32` and `DS_READ_ADDTID_B32` use the wrong address contract

Manual/XML evidence:

- The CDNA1 detailed rows define both ADDTID forms as
  `ADDR_BASE + OFFSET + M0.OFFSET + TID*4` at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:4516` and `:4668`.
- `DS_READ_ADDTID_B32` exposes an implicit M0 operand in the XML at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:12583` through
  `:12607`.

Rocjitsu evidence:

- `DS_WRITE_ADDTID_B32` calls the generic DS address helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/ds.cpp:814` through `:833`,
  so it uses the encoded `ADDR` VGPR and the combined offset rather than
  `M0.OFFSET + TID*4`.
- `DS_READ_ADDTID_B32` has a special path, but derives a per-lane stride from
  `M0[24:16]` and computes `lane * stride + offset + lds_base` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/ds.cpp:4051` through
  `:4076`.

Impact:

ADDTID loads and stores address different LDS locations from the manual formula
when M0 contains a nonzero base offset or when the high M0 bits do not encode
the expected lane stride.

### CDNA1-RJ-053: DS READ2/WRITE2 duplicate-offset collapse is not modeled

Manual/XML evidence:

- Chapter 11 says READ2/WRITE2 use separately scaled 8-bit offsets and that
  setting both offsets to the same value causes only one read or write and only
  the first data source is used at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2761` through `:2775`.
- The detailed table repeats the separate-address formulas for B32 and B64
  READ2/WRITE2 forms at `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:4505`
  through `:4506`, `:4552` through `:4553`, `:4587` through `:4588`, and
  `:4631` through `:4632`.

Rocjitsu evidence:

- `DS_WRITE2_B32` unconditionally sets `ds2_active = true`, computes both
  addresses, and writes both `DATA0` and `DATA1` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/ds.cpp:533` through `:563`.
- `DS_READ2_B32` unconditionally sets `ds2_active = true`, computes both
  addresses, and writes the second result to `VDST+1` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/ds.cpp:1773` through `:1797`;
  the local-memory completion path then writes that second result when
  `ds2_active` is set at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:599` through `:637`.

Impact:

Equal-offset READ2/WRITE2 encodings perform two modeled accesses in rocjitsu.
For WRITE2, `DATA1` can overwrite the value that the manual says should come
only from `DATA0`.

### CDNA1-RJ-054: Non-returning DS atomics write the raw `VDST` field

Manual/XML evidence:

- The CDNA1 XML `DS_ADD_U32` row has explicit `ADDR` and `DATA0` operands plus
  implicit DS memory operands, but no explicit `VDST` output, at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:6892` through `:6922`.
- The return form `DS_ADD_RTN_U32` has an explicit `VDST` output and says it
  stores the original data-share value into a vector register at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:8086` through `:8122`.

Rocjitsu evidence:

- `DsAddU32Ds` has `num_dst_ = 0`, but its execute body still sets
  `d->dst_reg_base = ... + inst_.vdst`, marks the access as a load, and assigns
  `AtomicOp::ADD` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/ds.cpp:27` through `:59`.
  The same generated pattern appears across the base DS atomic family.
- `vector_complete()` writes any load response to `d.dst_reg_base` for all
  active lanes at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:119`
  through `:198`.

Impact:

The non-RTN DS atomic forms can clobber an architectural VGPR selected by raw
encoding bits even though the instruction has no explicit VGPR destination. The
existing adjacent `DsAddU32_NoReturn` test checks only the final LDS value at
`emulation/rocjitsu/tests/amdgpu_vm_test.cpp:1942` through `:1965`.

### CDNA1-RJ-055: LDS floating atomics ignore MODE denormal and fixed-rounding policy

Manual/XML evidence:

- Section 11.3.3 says floating-point LDS atomics are controlled by
  `MODE.FP_DENORM` and use fixed round-to-nearest-even at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2805` through `:2807`.
- Detailed rows for `DS_CMPST_F32`, `DS_MIN_F32`, `DS_MAX_F32`, `DS_ADD_F32`,
  and return variants describe NaN/Inf/denormal handling at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:4511` through `:4515` and
  `:4543` through `:4550`.

Rocjitsu evidence:

- Generated CDNA1 floating DS atomic bodies lower to generic `AtomicOp::FMIN`,
  `FMAX`, or `FADD` without reading MODE at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/ds.cpp:702` through `:790`.
- `apply_fp_atomic()` uses host `+`, `std::fmin`, and `std::fmax`, with no
  MODE-denormal or explicit fixed-rounding policy, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:285` through `:297`.

Impact:

FP LDS atomics can diverge from hardware for denormals, NaNs, and rounding-mode
edge cases even when the address and data operands are otherwise decoded
correctly.

### CDNA1-RJ-056: GWS DS instructions are not simulated and restrictions are not enforced

Manual/XML evidence:

- Chapter 11.4 requires every GWS instruction to be immediately followed by
  `s_waitcnt 0` at `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2809`
  through `:2813`.
- The detailed GWS rows define resource-id calculation and semaphore/barrier
  state-machine behavior at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:4656` through `:4667`.
- XML rows for the six GWS opcodes are present at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:12412` through `:12568`.

Rocjitsu evidence:

- Generated CDNA1 GWS classes decode `DS_GWS_SEMA_RELEASE_ALL`,
  `DS_GWS_INIT`, `DS_GWS_SEMA_V`, `DS_GWS_SEMA_BR`, `DS_GWS_SEMA_P`, and
  `DS_GWS_BARRIER`, but every execute body throws `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/ds.cpp:3964` through
  `:4039`.
- The constructors do not enforce `GDS=1`, mandatory following `s_waitcnt 0`,
  or the GWS resource-state side effects.

Impact:

CDNA1 GWS encodings can decode but cannot execute, and their sequencing and
state-machine restrictions are not validated.

### CDNA1-RJ-057: `DS_APPEND` and `DS_CONSUME` return per-lane ranks instead of the pre-op counter

Manual/XML evidence:

- The detailed rows say `DS_CONSUME` subtracts `count_bits(exec_mask)` and
  `DS_APPEND` adds `count_bits(exec_mask)`, and both return the pre-operation
  value to VGPRs at `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:4669`
  through `:4670`.
- XML descriptions repeat that the pre-operation value is returned for
  `DS_CONSUME` and `DS_APPEND` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:12625` through
  `:12690`.

Rocjitsu evidence:

- `DsConsumeDs` and `DsAppendDs` set `AtomicOp::CONSUME`/`APPEND` and route
  through the LDS atomic helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/ds.cpp:4079` through
  `:4145`.
- `execute_lds_atomic_rmw()` reads the old counter, but writes
  `old_val + active_rank` for append and `old_val - active_rank - 1` for
  consume into each lane response at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:393` through `:420`.

Impact:

Kernels that use the returned append/consume counter see per-lane allocated
indices rather than the manual/XML pre-operation counter value.

### CDNA1-RJ-058: DS offsets and `GDS` are hidden in disassembly

Manual/XML evidence:

- The DS format exposes `GDS`, `OFFSET0`, and `OFFSET1` as instruction fields at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2723` through `:2735`.
- XML field metadata likewise exposes `GDS`, `OFFSET0`, and `OFFSET1` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:2358` through `:2379`.

Rocjitsu evidence:

- The generated `Ds` instruction base stores the raw `DsMachineInst`, but unlike
  MUBUF/MTBUF/FLAT it does not override `build_modifiers()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/encodings.h:586` through
  `:591`.
- The adjacent encoding constructors show modifier support for other memory
  formats, for example MUBUF/MTBUF/FLAT at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/encodings.cpp:257`
  through `:330`.

Impact:

Different DS encodings can disassemble identically when they differ only by
offsets or `GDS`, hiding address and memory-space differences from decode smoke
tests and diagnostics.

### CDNA1-RJ-059: Chapter 11 data-share tests miss core LDS/GWS edge cases

Evidence:

- Existing adjacent tests include a cross-architecture DS atomic memory check
  and a CDNA3 ACC transpose decode test, but the `DsAddU32_NoReturn` test only
  checks final LDS state at `emulation/rocjitsu/tests/amdgpu_vm_test.cpp:1942`
  through `:1965`, and `decode_smoke_test.cpp` coverage found in this slice is
  not a CDNA1 Chapter 11 execution suite.
- Searches of `emulation/rocjitsu/tests` and `lib/python/amdisa/tests` did not
  find focused CDNA1 coverage for M0 LDS clamping, READ2/WRITE2 equal-offset
  collapse, ADDTID address formulas, SRC2 opcode decode/execution, DS
  no-return destination preservation, FP atomic MODE/NaN/denormal behavior,
  append/consume return values, GWS sequencing, or DS offset/GDS disassembly.

Impact:

The current test surface can catch broad DS memory functionality regressions,
but it would not catch the CDNA1 Chapter 11 semantic and decode gaps above.

### CDNA1-RJ-060: SMEM offset selection ignores `OFFSET` register, M0, and low-bit masking

Manual/XML evidence:

- The CDNA1 manual address table allows `IMM=0/SOE=0` to use
  `SGPR[offset]` or M0, `IMM=0/SOE=1` to use `SGPR[soffset]` or M0, and all
  components are byte offsets with the low two address bits ignored at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1651` through `:1672`.
- The microcode-format table says `SOFFSET` may specify an SGPR or M0 at
  `cdna1/README.md:5842` through `:5843`.
- The XML field metadata exposes `OFFSET`, `SOFFSET`, `IMM`, and
  `SOFFSET_EN` at `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:702`
  through `:723` and `:766` through `:777`.

Rocjitsu evidence:

- CDNA1 `make_smem_offset()` returns the `SOFFSET` operand when `SOE` is set,
  returns a literal when `IMM` is set, and otherwise returns a literal zero, so
  the decoded operand never surfaces the `OFFSET` register path for
  `IMM=0/SOE=0` at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/smem.cpp:28`
  through `:38`.
- The shared SMEM address helper adds only `inst.soffset` when
  `inst.soffset_en` is set and the sign-extended immediate when `inst.imm` is
  set, without reading `OFFSET` as a register, recognizing M0/null selector
  values, or masking the computed address to dword alignment at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:32`
  through `:50`.
- CDNA1 operand metadata does know the SMEM M0 selector for presentation, while
  RDNA3/RDNA4 address helpers implement explicit null/M0 selector handling at
  `emulation/rocjitsu/tests/shared_infra_test.cpp:2999` through `:3019` and
  `:3217` through `:3237`.

Impact:

CDNA1 SMEM instructions using the legal register-offset form, M0 offset form,
or unaligned byte offsets compute and display the wrong address/operand.

### CDNA1-RJ-061: `S_SCRATCH_*` SMEM uses ordinary byte-offset addressing

Manual/XML evidence:

- Chapter 8 says `S_SCRATCH_LOAD` and `S_SCRATCH_STORE` add
  `{ M0 or SGPR[offset] or zero } * 64` while immediate offsets remain byte
  offsets at `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1657` through
  `:1668`.
- The detailed instruction rows repeat that SGPR scratch offsets are unsigned
  64-byte offsets at `cdna1/README.md:3200` through `:3202` and
  `:3211` through `:3218`.

Rocjitsu evidence:

- Representative CDNA1 scratch load constructors and execute methods route to
  the ordinary `smem_calculate_address()` helper at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/smem.cpp:177`
  through `:255`.
- The CDNA1 architecture-specific helper forwards all SMEM opcodes to the
  shared helper, which has no scratch-op branch or 64-byte scaling at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:32`
  through `:50`.
- CDNA4 has an explicit scratch-op path that applies the `* 64` rule at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/addr_calc.cpp:24`
  through `:40`, showing the missing specialization on the CDNA1 path.

Impact:

CDNA1 scalar scratch loads/stores using SGPR or M0 offsets access byte-scaled
addresses instead of scratch-slot-scaled addresses.

### CDNA1-RJ-062: `S_BUFFER_*` SMEM ignores scalar resource descriptors and bounds

Manual/XML evidence:

- The CDNA1 manual says scalar-buffer SMEM uses a 4-SGPR resource constant and
  only the base, stride, `num_records`, and `NV` fields are consumed at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1638` and
  `:1686` through `:1707`.
- The alignment section says buffer `SBASE` must select an aligned SGPR quad,
  out-of-range `SBASE` falls back to SGPR0, and out-of-range memory-address
  dwords are not performed at `cdna1/README.md:1743` through `:1751`.
- XML marks representative scalar-buffer operands as `FMT_RSRC_SCALAR` and
  128-bit `SBASE` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:30156` through `:30187`.

Rocjitsu evidence:

- Generated CDNA1 buffer SMEM constructors model `SBASE` as a 128-bit source,
  but execution still calls the same base-pair address helper, for example
  `S_BUFFER_LOAD_DWORD` at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/smem.cpp:258`
  through `:282`.
- The shared helper reads only `sbase*2` and `sbase*2+1` as a 64-bit base and
  has no descriptor decode, stride/`num_records` bounds check, SGPR0 fallback,
  or per-dword suppression at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:32`
  through `:50`.
- `ScalarMemState` only carries an address, transfer size, cache type, and
  data buffers, with no scalar resource descriptor or bounds metadata, at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:70`
  through `:83`.

Impact:

CDNA1 scalar-buffer loads/stores/probes/atomics address memory as ordinary
64-bit-base SMEM operations, so descriptor-base, stride/size bounds, and
alignment fallback behavior are not represented.

### CDNA1-RJ-063: SMEM atomics decode but all execute paths throw

Manual/XML evidence:

- The manual says scalar atomics share scalar-memory addressing and can return
  the pre-operation value to `SDATA` when `GLC=1` at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1711`.
- Chapter 12 lists the scalar-buffer atomic rows at `cdna1/README.md:3232`
  through `:3272` and the scalar atomic rows at `:3273` through `:3308`.

Rocjitsu evidence:

- Generated CDNA1 atomic constructors are present, but representative bodies
  immediately throw `util::UnimplementedInst`, for example
  `S_BUFFER_ATOMIC_SWAP` at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/smem.cpp:778`
  through `:795`.
- Static search found every scalar-buffer and scalar atomic execute body in
  `cdna1/smem.cpp` throwing at lines `794` through `1763`.
- The atomic constructors always expose `SDATA` as both a source and
  destination in the generated metadata, for example
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/smem.cpp:781`
  through `:789`, so the `GLC=0` no-return shape is not modeled.

Impact:

Any CDNA1 scalar atomic reaches an unimplemented path, and even decode/def-use
metadata overstates the destination side effect for non-returning atomics.

### CDNA1-RJ-064: SMEM cache, discard, probe, and time operations are incomplete

Manual/XML evidence:

- `S_DCACHE_INV/WB` affect the scalar data cache and return no `SDST`,
  `S_MEMREALTIME` uses a constant 25MHz clock, and discard operates on
  64-byte-aligned cache lines with LGKM increments of 1 or 2 at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1715` through `:1725` and
  `:3230` through `:3231`.
- XML rows exist for cache invalidate/writeback, time, ATC probe, and discard
  opcodes at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:30842` through
  `:31140`.

Rocjitsu evidence:

- `S_DCACHE_INV_VOL` and `S_DCACHE_WB_VOL` execute the same whole-cache helpers
  as the non-volatile forms at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/smem.cpp:665`
  through `:685`.
- `S_DCACHE_DISCARD` and `S_DCACHE_DISCARD_X2` throw unimplemented at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/smem.cpp:746`
  through `:776`.
- ATC probe helpers are empty no-ops at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:737`
  through `:742`.
- `S_MEMTIME` and `S_MEMREALTIME` both use thread-local synthetic counters that
  add 100 per execution, rather than distinct timestamp/25MHz real-time
  sources, at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1933`
  through `:1950`.

Impact:

CDNA1 auxiliary SMEM opcodes decode, but cache-line granularity, volatile-line
selection, probe/prefetch effects, discard wait accounting, and real timer
semantics are absent or approximated.

### CDNA1-RJ-065: SMEM LGKM accounting, alignment, bounds, and clause legality are not modeled

Manual/XML evidence:

- Chapter 8 says SMEM increments LGKM by 1 for single-dword fetches and by 2
  for two-or-more-dword fetches, and describes out-of-order/partial returns at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1727` through `:1735`.
- The manual also gives source-overwrite, clause-break, atomic single-clause,
  `SDATA`, `SBASE`, and memory-address legality rules at
  `cdna1/README.md:1682` through `:1684` and `:1737` through `:1751`.

Rocjitsu evidence:

- The memory pipeline increments the selected wait counter exactly once per
  issued memory instruction and releases it exactly once after completion at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.h:62`
  through `:88`.
- Generated CDNA1 SMEM execute paths set `wait_counter_type = LGKMCNT`, but do
  not carry a per-instruction LGKM increment count, representative load shown at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/smem.cpp:56`
  through `:66`.
- Static inspection found no clause tracking, source-overwrite validation,
  `SDATA`/`SBASE` alignment enforcement, out-of-range `SBASE` fallback, or
  out-of-range memory dword suppression on the CDNA1 SMEM path.

Impact:

Programs that rely on precise LGKM counts, replay/clause legality, or SMEM
alignment and bounds behavior can execute with hardware-inaccurate ordering and
side effects.

### CDNA1-RJ-066: Chapter 8 scalar-memory tests cover SBASE decode only

Evidence:

- The adjacent SMEM test is an operand-model test for SBASE scaling across
  architectures at `emulation/rocjitsu/tests/smem_sbase_operand_test.cpp:1`
  through `:144`; it does not execute CDNA1 SMEM addressing, descriptors,
  cache/discard/probe/time, atomics, or LGKM semantics.
- The generic instruction execution harness explicitly skips scalar memory
  prefixes including `s_load_`, `s_store_`, `s_buffer_`, `s_dcache_`, and
  `s_atomic_` at
  `emulation/rocjitsu/tests/instruction_execution_harness_test.cpp:135`
  through `:156`.
- Existing RDNA3/RDNA4 shared-infra tests cover SMEM null/M0/SGPR offset
  selectors at `emulation/rocjitsu/tests/shared_infra_test.cpp:2999` through
  `:3019` and `:3217` through `:3237`, but there is no matching CDNA1
  regression coverage for the Chapter 8 selector and address rules.

Impact:

The current tests would catch an SBASE register-reference regression, but not
the CDNA1 scalar-memory semantic gaps above.

### CDNA1-RJ-067: Fork/join branch-stack control flow is unimplemented

Manual/XML evidence:

- Section 4.6 says `S_CBRANCH_I/G_FORK` and `S_CBRANCH_JOIN` use a six-deep
  control stack, store `{exec[63:0], PC[47:2]}` entries in SGPRs, choose the
  smaller pass/fail path first, and update `EXEC`, PC, and CSP according to the
  listed pseudocode at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:878` through `:933`.
- XML rows exist for `S_CBRANCH_G_FORK`, `S_CBRANCH_I_FORK`, and
  `S_CBRANCH_JOIN` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:42644` through
  `:42750`, `:47070` through `:47090`, and `:36363` through `:36405`.

Rocjitsu evidence:

- Generated CDNA1 `S_CBRANCH_G_FORK` decodes operands, including literal
  extension handling, but its execute body throws `util::UnimplementedInst` at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sop2.cpp:962`
  through `:984`.
- Generated `S_CBRANCH_I_FORK` also throws at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopk.cpp:264`
  through `:278`.
- Generated `S_CBRANCH_JOIN` calls `execute_s_cbranch_join_sop1()` at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sop1.cpp:916`
  through `:930`, but the shared helper is an empty no-op at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1013`
  through `:1015`.

Impact:

CDNA1 irreducible or complex divergent-control sequences using fork/join cannot
execute with the documented control-stack behavior.

### CDNA1-RJ-068: Debug conditional branches decode as no-op and lack branch metadata

Manual/XML evidence:

- Chapter 4 lists `S_CBRANCH_CDBGSYS`, `S_CBRANCH_CDBGUSER`, and combined
  debug predicates as conditional branches at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:797` through `:803`.
- Detailed SOPP rows include the debug conditional branch forms at
  `cdna1/README.md:3157` through `:3160`, and XML rows for the same mnemonics
  start at `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:48017`.

Rocjitsu evidence:

- Generated CDNA1 debug branch constructors expose only the label operand and
  do not set `BRANCH` or `COND_BRANCH` flags at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopp.cpp:363`
  through `:413`.
- The shared debug branch helpers are empty no-ops at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:998`
  through `:1011`.

Impact:

A taken CDNA1 debug conditional branch falls through in rocjitsu, and CFG,
patching, and indirect-branch analysis do not see those instructions as branch
terminators.

### CDNA1-RJ-069: Auxiliary SOPP controls are no-ops or functional-yield approximations

Manual/XML evidence:

- Chapter 4 and the detailed SOPP table describe `S_WAKEUP`, `S_SETHALT`,
  `S_SETKILL`, `S_SLEEP`, `S_SETPRIO`, `S_SENDMSG`, `S_SENDMSGHALT`,
  `S_ICACHE_INV`, perf-level operations, `S_TTRACEDATA`,
  `S_ENDPGM_SAVED`, and `S_ENDPGM_ORDERED_PS_DONE` at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:779` through `:816` and
  `:3123` through `:3169`.
- The XML carries rows for these SOPP forms at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:47270` through
  `:48237`.

Rocjitsu evidence:

- Shared helpers for `S_ICACHE_INV`, perf-level controls, `S_SENDMSG`,
  `S_SENDMSGHALT`, `S_SETHALT`, `S_SETKILL`, `S_SETPRIO`, `S_TTRACEDATA`, and
  `S_WAKEUP` are empty at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1770`
  through `:1777`, `:2427` through `:2492`, `:2526` through `:2533`, and
  `:2662` through `:2690`.
- `S_SLEEP` only requests a functional yield and does not model the
  documented sleep duration or wakeup interaction at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2555`
  through `:2558`.
- Generated CDNA1 `S_ENDPGM_SAVED` and `S_ENDPGM_ORDERED_PS_DONE` both just
  end the wavefront, so context-save and ordered-pixel-shader completion
  semantics are not distinguished at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopp.cpp:415`
  through `:457`.

Impact:

CDNA1 code that relies on message delivery, halt/kill/priority state, cache
invalidation side effects, trace data, wake/sleep timing, or the specialized
end-program variants executes as a functional no-op or plain wavefront halt.

### CDNA1-RJ-070: Required software wait-state hazards are not modeled or validated

Manual/XML evidence:

- Section 4.5 says hardware does not check several dependencies and requires
  manually inserted NOPs or independent instructions for `S_SETREG`, VSKIP,
  VCC/EXEC, lane select, `V_DIV_FMAS`, wide stores/atomics, VMEM SGPR reads,
  M0 consumers, DPP, TRAPSTS/RFE, and `S_MOVEREL` cases at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:850` through `:877`.

Rocjitsu evidence:

- `S_NOP` executes as a simple no-op at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopp.cpp:20`
  through `:28`, and the CU issue loop executes each decoded instruction
  directly before advancing PC at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:393`
  through `:439`.
- Static inspection found no scoreboard, hazard validator, or instruction
  spacing model for the Chapter 4.5 required wait-state table; related helper
  bodies such as `S_ICACHE_INV` are also empty no-ops at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1770`
  through `:1773`.

Impact:

rocjitsu cannot detect missing NOPs or model the hardware-visible stale-read
behavior for CDNA1 programs that violate the manual's software scheduling
contract.

### CDNA1-RJ-071: Chapter 4 tests cover only ordinary branches, endpgm, and a barrier event

Evidence:

- The adjacent VM tests cover `S_BRANCH`, SCC conditional branches, and
  `S_ENDPGM` at
  `emulation/rocjitsu/tests/amdgpu_vm_test.cpp:1442` through `:1515`.
- A plugin-ordering test covers two-wave `S_BARRIER` event reporting at
  `emulation/rocjitsu/tests/execution_plugin_test.cpp:870` through `:886`.
- The generic instruction execution harness skips branch, call, wait,
  barrier, trap, sleep, halt, send-message, cache-invalidate, NOP, direct-PC,
  HWREG, and RFE instructions at
  `emulation/rocjitsu/tests/instruction_execution_harness_test.cpp:135`
  through `:178`.
- Static search found no focused CDNA1 coverage for fork/join, debug
  branches, direct-PC zero-target behavior, RFE/trap flow, `S_SENDMSG`
  LGKM behavior, wake/sleep, halt/kill/priority, perf/thread-trace controls,
  instruction-cache spacing, or the required wait-state hazards.

Impact:

The existing tests exercise the basic branch/barrier happy path, but would not
catch the Chapter 4 control-flow and auxiliary-control gaps listed above.

### CDNA1-RJ-072: `S_CALL_B64` is tagged as an indirect call even though it is PC-relative

Manual/XML evidence:

- Chapter 4 defines `S_CALL_B64` as saving `PC+4` to an SGPR pair and setting
  `PC = PC+4+SIMM16*4` at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:814`.
- The CDNA1 XML marks `S_CALL_B64` as `IsBranch=TRUE` and
  `IsIndirectBranch=FALSE`, describes a constant offset relative to the current
  PC, and exposes the second operand as `OPR_LABEL` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:47214` through
  `:47250`.

Rocjitsu evidence:

- The generated CDNA1 `SCallB64Sopk` constructor sets `flags_ |=
  INDIRECT_CALL`, even though `branch_offset_bytes()` is the signed
  `SIMM16 * 4` PC-relative offset and execution applies the same immediate
  offset at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopk.cpp:387`
  through `:407`.
- Downstream code treats `INDIRECT_CALL` as block-terminator/call metadata in
  basic block construction and patching at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/code/basic_block.cpp:25`
  through `:57` and
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/code/patch/instruction_builder.cpp:103`
  through `:108`; indirect-branch discovery also classifies direct calls via
  the `INDIRECT_CALL` bit plus a branch offset at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/analysis/indirect_branch_discovery.cpp:592`
  through `:598`.

Impact:

The runtime target is PC-relative, but analysis and patch metadata describe the
instruction as an indirect call. Consumers that distinguish direct and indirect
control flow cannot rely on the CDNA1 `S_CALL_B64` flags alone.

### CDNA1-RJ-073: VOP3 floating output modifiers ignore MODE denorm and IEEE gating

Manual evidence:

- Chapter 6.2.2 says VOP3 output modifiers apply only to floating-point results,
  are ignored when output denormals are enabled or when the IEEE mode bit is set,
  and otherwise flush denormals and `-0` when output denormals are disabled at
  `cdna1/README.md:1293` through `:1299`.

Rocjitsu evidence:

- `Wavefront` stores raw MODE state through `mode_raw()` and `set_mode_raw()` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:103` through `:111`.
- Representative shared VOP3 FP execution applies `inst.inst_.omod` and
  `inst.inst_.clamp` directly in `execute_v_add_f32_vop3()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:3018`
  through `:3056`; the audited body does not read MODE before applying those
  modifiers.
- CDNA1 `S_SETREG_B32` and `S_SETREG_IMM32_B32` update only HWREG id 1
  (`STATUS`) and warn on other IDs at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopk.cpp:334` through
  `:382`, so the Chapter 6 MODE route cannot currently update FP_DENORM/IEEE
  state for VALU execution.

Impact:

CDNA1 VOP3 FP instructions can produce modified, clamped, denormal, or `-0`
results in rocjitsu even in MODE states where the manual says the modifier must
be ignored or the result must be flushed.

### CDNA1-RJ-074: VALU source validation allows manual-disallowed extra scalar sources

Manual evidence:

- Chapter 6.2.1 says VALU instructions can read at most one SGPR per instruction
  and can use at most one literal, only when no SGPR or M0 source is used, at
  `cdna1/README.md:1265` through `:1269`.
- The same section says `ADDC`, `SUBB`, and `CNDMASK` implicitly use VCC, so
  those instructions cannot use an additional SGPR or literal at
  `cdna1/README.md:1271` through `:1275`.

Rocjitsu evidence:

- Representative generated CDNA1 VOP3 `V_ADD_F32` independently constructs
  `src0` as `OPR_SRC_NOLIT` and `src1` as `OPR_SRC_SIMPLE` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop3.cpp:1924` through
  `:1933`; both operand paths can resolve scalar-source selector values.
- `V_CNDMASK_B32` adds an explicit scalar-pair `src2` for VCC at
  `vop3.cpp:1905` through `:1916`, and VOP3B carry-consuming forms add explicit
  `OPR_SREG` `src2` operands at `vop3.cpp:12104` through `:12118`, with no
  adjacent validation that `SRC0`/`SRC1` avoid extra scalar or literal sources.
- `Operand::read_lane()` resolves each non-VGPR VALU source independently at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/operand.cpp:1461` through
  `:1473`, and `resolve_src_scalar()` can read SGPRs, VCC, M0, EXEC, inline
  constants, and helper predicates at `operand.cpp:1147` through `:1214`.

Impact:

This is a legality/diagnostic gap rather than proof that a legal encoding
executes incorrectly: rocjitsu can decode and execute VALU source combinations
that the CDNA1 manual says the hardware contract disallows.

### CDNA1-RJ-075: VALU FP round and denormal modes are not modeled

Manual evidence:

- Chapter 6.4 says the shader program controls floating-point rounding and
  denormal input/result handling through MODE fields set by `S_SETREG`, with
  separate single-precision and double/half-precision fields at
  `cdna1/README.md:1431` through `:1440`.

Rocjitsu evidence:

- CDNA1 `S_SETREG_B32` and `S_SETREG_IMM32_B32` only implement writes to
  `STATUS`; other HWREG IDs log warnings at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopk.cpp:334` through
  `:382`.
- Shared `S_DENORM_MODE` and `S_ROUND_MODE` helpers are empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1644`
  through `:1645` and `:2423` through `:2425`.
- Representative VOP3 FP arithmetic uses host arithmetic plus direct
  modifier/clamp handling, for example `execute_v_add_f32_vop3()` at
  `execute_shared.h:3018` through `:3056`.

Impact:

Ordinary CDNA1 VALU FP arithmetic is evaluated under host default behavior
instead of the MODE-controlled rounding and denormal policy described by the ISA
manual.

### CDNA1-RJ-076: ALU clamp non-FP semantics are incomplete

Manual evidence:

- Chapter 6.5 says `V_CMP` clamp requests signaling compare behavior on
  floating-point exceptions, integer operations clamp to the largest/smallest
  representable value, and floating-point operations clamp to `[0.0, 1.0]` at
  `cdna1/README.md:1442` through `:1444`.
- Chapter 12.9.1 repeats the compare-specific rule that `CLAMP=1` signals an
  exception when either input is NaN at `cdna1/README.md:4092`.

Rocjitsu evidence:

- Representative integer VOP3 helpers ignore `inst.inst_.clamp` and execute
  wrapping arithmetic directly: `execute_v_add_i32_vop3()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:3119`
  through `:3128` and `execute_v_add_u32_vop3()` at `:3244` through `:3253`.
- Representative VOP3 compare helpers compute and store the condition result
  without inspecting `inst.inst_.clamp` or setting exception/signaling state, for
  example `execute_v_cmp_eq_f32_vop3()` at `execute_shared.h:4358` through
  `:4387`.
- The adjacent VOPC/VOP3 FP32 SIMD test encoder comments that compare `clamp`
  is irrelevant and leaves the bit zero at
  `tests/simd_correctness/vopc_vop3_fp32_simd_correctness_test.cpp:45` through
  `:51`.

Impact:

Programs that encode `CLAMP=1` on ordinary integer VOP3 arithmetic see
wrap-around results rather than CDNA1's documented saturation, and `V_CMP`
encodings cannot request the documented signaling-compare behavior through the
clamp bit.

### CDNA1-RJ-077: VGPR indexing uses the wrong M0 layout and cannot honor source-role masks

Manual evidence:

- Chapter 6.6 defines `M0[7:0]` as the index and `M0[15:12]` as
  dest/src2/src1/src0 enable bits, with indexing applying only to VGPR operands
  and indexed out-of-range VGPRs illegal at `cdna1/README.md:1458` through
  `:1474`.
- Chapter 6.6.2 gives instruction-specific role remapping for readlane,
  writelane, MAC/MAD, reverse shifts, `v_cvt_pkaccum`, and SDWA read-modify-write
  at `cdna1/README.md:1476` through `:1490`.

Rocjitsu evidence:

- `Wavefront::gpr_idx_mode()` reads `(m0 >> 8) & 0xF`, and the shared
  `S_SET_GPR_IDX_MODE` / `S_SET_GPR_IDX_ON` helpers write the mode nibble at
  `M0[11:8]`, not `M0[15:12]`, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:248` through `:251` and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2495`
  through `:2519`.
- `apply_gpr_idx()` applies any low source-enable bit to all source operands and
  only distinguishes destination versus non-destination at
  `wavefront.h:584` through `:589`.
- CDNA1 operand reads/writes pass only a boolean source/destination role into
  that helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/operand.cpp:1461` through
  `:1510`, so they cannot distinguish src0, src1, src2, or the manual's
  instruction-specific remaps.

Impact:

CDNA1 indexed VGPR accesses use different M0 bits from the manual and can index
the wrong operands, while illegal indexed out-of-range accesses are not
diagnosed.

### CDNA1-RJ-078: Chapter 6 tests miss MODE, source-legality, clamp, and indexing edges

Manual evidence:

- Chapter 6.2 through 6.6 define the VALU scalar/literal/M0 source budget,
  EXEC-masked writes, VOP3 output modifier and MODE interactions, ALU clamp
  overloads, and VGPR indexing rules at `cdna1/README.md:1252` through `:1490`.

Rocjitsu evidence:

- The shared ISA harness has basic VALU smoke tests for `V_MOV_B32`,
  `V_ADD_F32`, `V_MUL_F32`, `V_ADD_U32`, `V_CMP_EQ_F32`, and `V_CNDMASK_B32` at
  `tests/amdgpu_vm_test.cpp:1518` through `:1588`, but those use ordinary
  sources and default MODE state.
- SIMD correctness tests sweep raw VOP3 modifier combinations against the
  generated scalar model, for example
  `tests/simd_correctness/vop3_binary_simd_correctness_test.cpp:170` through
  `:207` and `:290` through `:296`, and
  `tests/simd_correctness/vop3_modifier_simd_test.cpp:187` through `:217`.
- Static test search found `s_setreg_imm32_b32` coverage only for STATUS writes
  at `tests/instruction_execution_harness_test.cpp:3592` through `:3638`, and
  found no focused CDNA1 `S_SET_GPR_IDX`, `FP_ROUND`, or `FP_DENORM` coverage.

Impact:

The existing tests cover ordinary VALU execution and scalar/SIMD parity, but
would not catch the Chapter 6 stateful MODE behavior, illegal source
combinations, clamp overloads, or VGPR-index role-mask gaps listed above.

### CDNA1-RJ-079: CDNA1 AccVGPR instructions address a register file that rocjitsu does not allocate

Manual evidence:

- The CDNA1 Chapter 7 MAI introduction says miSIMD has its own AccVGPR file,
  separate from ArchVGPRs, and that shader I/O can only use ArchVGPRs at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:1525` through `:1531`.
- The same section says A and B may come from ArchVGPRs or AccVGPRs, C always
  comes from AccVGPRs, D always uses AccVGPRs, and
  `V_ACCVGPR_READ`/`V_ACCVGPR_WRITE` move data between the two files at
  `cdna1/README.md:1538` through `:1543`.
- The detailed VOP3P table lists `V_ACCVGPR_READ` and `V_ACCVGPR_WRITE` as
  moves between AccVGPR and ArchVGPR at `cdna1/README.md:4178` through `:4180`.

Rocjitsu evidence:

- `cdna1::Isa` documents `MAX_ACC_VGPRS = 0`, saying CDNA1 has no AccVGPR
  file and MFMA writes to VGPRs, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/isa.h:25`.
- The shared CDNA ISA base defaults `MAX_ACC_VGPRS_PER_WF` to zero at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/cdna_isa_base.h:97`, and
  `tests/shared_infra_test.cpp:123` through `:129` asserts that CDNA1 has zero
  AccVGPRs while CDNA2/3/4 have 256.
- The compute-unit allocator reserves the physical AccVGPR region only when
  `Isa::MAX_ACC_VGPRS_PER_WF != 0`; otherwise the per-wave VGPR block does not
  include `base + ACC_VGPR_OFFSET` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.h:686` through `:692`.
- Generated CDNA1 `V_ACCVGPR_READ` and `V_ACCVGPR_WRITE` decode and execute
  paths are present at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop3p.cpp:557` through
  `:594`, and fixed `OPR_ACCVGPR` execution maps raw fields to physical offset
  `+256` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/operand.cpp:1399` through
  `:1407`.

Impact:

CDNA1 AccVGPR move and MFMA C/D paths can address `base + 256` even though the
CDNA1 wave allocation does not reserve a separate AccVGPR file. With one wave
slot this can go out of range; with adjacent wave slots it can alias the next
wave's ArchVGPR allocation. This is the broader allocation/state issue behind
the narrower CDNA1 MFMA bank-selection gaps above.

### CDNA1-RJ-080: `V_ACCVGPR_WRFED` opcode 92 is not generated

Manual/XML evidence:

- The CDNA1 Chapter 13.3 VOP3P opcode table lists opcode 92 as
  `V_ACCVGPR_WRFED` at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:6821`.
- The checked-in CDNA1 XML has `V_ACCVGPR_READ` and `V_ACCVGPR_WRITE` rows at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:69487` and `:69525`,
  but no `V_ACCVGPR_WRFED` row.

Rocjitsu evidence:

- Generated CDNA1 opcode constants include only `kVAccvgprReadVop3p = 88` and
  `kVAccvgprWriteVop3p = 89` for this AccVGPR move group at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/opcodes.h:647` through
  `:648` and `:1958` through `:1959`.
- Generated decoder dispatch includes `decodeVAccvgprReadVop3p` and
  `decodeVAccvgprWriteVop3p`, but no WRFED decoder, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/decoder.cpp:4411` through
  `:4416` and `:8648` through `:8649`.

Impact:

If the manual opcode-table entry is valid for CDNA1, rocjitsu cannot decode,
disassemble, or execute opcode 92 because the checked-in XML omits it and the
generated CDNA1 instruction set has no corresponding class.

### CDNA1-RJ-081: `S_CBRANCH_G_FORK` literal selectors decode despite no-literal operands

Manual/XML/oracle evidence:

- The CDNA1 Chapter 12.1 `S_CBRANCH_G_FORK` row defines the operands as a
  64-bit compare mask and 64-bit target byte address at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:2906` through `:2916`.
- XML gives the default `S_CBRANCH_G_FORK` operands type `OPR_SSRC_NOLIT`, but
  also emits `SOP2_INST_LITERAL` alternatives for `has_lit_0`, `has_lit_1`,
  and `has_lit_0_has_lit_1` at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:40838` through
  `:40932`; the corresponding manual-vs-XML ambiguity is tracked in
  `CDNA1-XML-039`.
- As an oracle check, `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908`
  accepts `s_cbranch_g_fork s[0:1], s[2:3]` but rejects
  `s_cbranch_g_fork s[0:1], 0x12345678` as an invalid operand.

Rocjitsu evidence:

- Generated CDNA1 `SCbranchGForkSop2` constructs both operands as
  `OPR_SSRC_NOLIT`, but then rewrites raw selector 255 for either operand into
  `OPR_SIMM32` from the SOP2 literal extension at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sop2.cpp:962` through
  `:980`.
- The execute body then throws `UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sop2.cpp:981` through
  `:984`, as already tracked by `CDNA1-RJ-067`, but the invalid literal form is
  already accepted and surfaced by decode/operand construction before
  execution.

Impact:

Raw selector-255 `S_CBRANCH_G_FORK` encodings can decode and disassemble as
literal branch operands even though LLVM rejects those operands and the XML base
operand type says no literal. This is a decoder/disassembly legality gap in
addition to the already-known missing fork-stack execution semantics.

### CDNA1-RJ-082: `S_SET_GPR_IDX_ON` treats the raw SIMM4 field as a literal source

Manual/XML/oracle evidence:

- CDNA1 Chapter 12.4 says `S_SET_GPR_IDX_ON` writes `M0[15:12] = SIMM4`, and
  clarifies that this is the direct raw content of the S1 field; the raw S1
  bits select `VSRC0_REL`, `VSRC1_REL`, `VSRC2_REL`, and `VDST_REL` at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:3113`.
- XML's default `S_SET_GPR_IDX_ON` row records the second operand as
  `OPR_SIMM4`, but the generic literal variants for this row are imprecise as
  recorded in `CDNA1-XML-049`.
- As an oracle check, `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908`
  accepts `s_set_gpr_idx_on s4, 15` and `s_set_gpr_idx_on 0x12345678, 15`, but
  rejects `s_set_gpr_idx_on s4, 255` as an invalid 4-bit immediate. Raw bytes
  with `SSRC1=0xff` disassemble as `s_set_gpr_idx_on s0, 0xff` without a
  following extension word.

Rocjitsu evidence:

- Generated CDNA1 `SSetGprIdxOnSopc` initially constructs `ssrc1` as
  `OperandType::OPR_SIMM4` from the raw `ssrc1` field at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopc.cpp:411` through
  `:415`.
- The same constructor rewrites `ssrc1` to `OPR_SIMM32` when the raw field is
  255 at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopc.cpp:424`
  through `:427`.
- The generic CDNA1 SOPC encoding marks any instruction with either source
  field equal to 255 as having an extension literal at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/encodings.cpp:44` through
  `:54`.
- `execute_s_set_gpr_idx_on_sopc()` then reads `inst.ssrc1` through
  `RegisterAccess` and masks the low four bits at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2514`
  through `:2519`. The separate M0 bit-position bug is tracked in
  `CDNA1-RJ-077`.

Impact:

An encoding with raw S1 field `0xff` should use the raw low four bits as the
SIMM4 mode mask, but rocjitsu consumes the following extension word and uses the
literal's low four bits instead. Decode and disassembly can therefore assign an
extension literal to a direct field that the manual defines as raw instruction
bits.

### CDNA1-RJ-083: GPR-indexing and thread-trace rows drop implicit M0 metadata

Manual/XML evidence:

- CDNA1 `S_SET_GPR_IDX_IDX`, `S_SET_GPR_IDX_ON`, and `S_SET_GPR_IDX_MODE` update
  GPR-indexing mode or M0 state, preserving unrelated M0 bits, at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:3072`,
  `:3113`, and `:3168`.
- CDNA1 `S_TTRACEDATA` sends M0 as user data to the thread-trace stream at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:3156`.
- XML records implicit M0 input/output operands for `S_SET_GPR_IDX_IDX`,
  `S_SET_GPR_IDX_ON`, and `S_SET_GPR_IDX_MODE`, and an implicit M0 input for
  `S_TTRACEDATA`, at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:34844` through `:34852`,
  `:44226` through `:44234`, `:46034` through `:46037`, and `:46228` through
  `:46236`.

Rocjitsu evidence:

- Generated CDNA1 constructors for the GPR-indexing state instructions expose
  only their printed operands and set `num_dst_ = 0`: `S_SET_GPR_IDX_IDX` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sop1.cpp:949` through
  `:959`, `S_SET_GPR_IDX_ON` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopc.cpp:411` through
  `:428`, and `S_SET_GPR_IDX_MODE` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopp.cpp:436` through
  `:447`.
- Generated `STtracedataSopp` exposes no source operands even though the XML row
  carries an implicit M0 input, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopp.cpp:352` through
  `:360`.
- No CDNA1 `SSetGprIdx*` or `STtracedataSopp` class overrides
  `implicit_uses()` or `implicit_defs()`, so def-use analysis only sees explicit
  operands through `InstDefUse` at
  `lib/rocjitsu/src/rocjitsu/analysis/def_use_chain.cpp:25` through `:45`;
  the default hooks are empty at `lib/rocjitsu/src/rocjitsu/isa/instruction.h:215`
  through `:222`.
- `RegisterSet` declares `RegClass::M0` but documents it as untracked and drops
  non-SGPR/VGPR/ACC classes in `expand()` at
  `lib/rocjitsu/src/rocjitsu/isa/register_set.h:50` through `:61` and
  `lib/rocjitsu/src/rocjitsu/isa/register_set.cpp:37` through `:51`.

Impact:

Analyses, schedulers, and DBT passes that depend on rocjitsu operand or def-use
metadata cannot see that these instructions read or write M0. Runtime GPR-index
layout issues remain tracked in `CDNA1-RJ-077`, and the thread-trace execution
stub remains tracked in `CDNA1-RJ-069`; this gap is specifically the missing
implicit M0 dataflow surface.

### CDNA1-RJ-084: `V_PK_FMAC_F16` decodes but both execute paths throw

Manual/XML/oracle evidence:

- Chapter 12.7 defines `V_PK_FMAC_F16` as VOP2 opcode 60: both packed F16 halves
  multiply their corresponding sources and accumulate into the old destination
  halves at `cdna1/README.md:3423`.
- Chapter 12.7.1 says VOP2 instructions can use a VOP3 encoding with opcode
  `VOP2+0x100` at `cdna1/README.md:3427` through `:3429`.
- XML represents `V_PK_FMAC_F16` as VOP2 opcode 60 and promoted VOP3 opcode 316
  at `amdgpu_isa_cdna1.xml:64778` through `:64884`.
- As a `gfx908` oracle, `llvm-mc` assembles
  `v_pk_fmac_f16_e32 v0, v1, v2`.

rocjitsu evidence:

- Generated decode metadata includes `kVPkFmacF16Vop2 = 60` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/opcodes.h:613`, the E32
  test-encoding row at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/test_encodings.h:614`,
  and primary decode table slots for `decodeVPkFmacF16Vop2` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/decoder.cpp:6136` through
  `:6139`.
- The VOP2 constructor records `VDST` as both source and destination, matching
  the accumulator operand shape, but `VPkFmacF16Vop2::execute_impl()` throws
  `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop2.cpp:8142` through
  `:8188`.
- The promoted VOP3 constructor also records `VDST` as an input/output
  accumulator, but `VPkFmacF16Vop3::execute_impl()` throws
  `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop3.cpp:3315` through
  `:3332`.

Impact:

Any CDNA1 kernel that reaches `V_PK_FMAC_F16` traps in rocjitsu even though the
instruction is present in the manual, XML, assembler, generated opcode table,
and decoder. This is narrower than a decode gap: the generated operand metadata
is mostly correct, but the packed F16 accumulator semantics are absent.

### CDNA1-RJ-085: CDNA1 legacy VOP1 exp/log opcodes decode as invalid

Manual/oracle evidence:

- Chapter 12.8 defines `V_EXP_LEGACY_F32` at VOP1 opcode 75 and
  `V_LOG_LEGACY_F32` at opcode 76 at `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:3690`
  through `:3692`.
- Chapter 13.3.2 repeats those opcode-table entries at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:6154` through `:6155`.
- As a `gfx908` oracle, `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908
  -show-encoding` assembles `v_exp_legacy_f32_e32 v0, v1`,
  `v_log_legacy_f32_e32 v0, v1`, and their E64 spellings.

Rocjitsu evidence:

- The generated CDNA1 opcode constants skip VOP1 opcodes 75 and 76:
  `kVCosF16Vop1 = 74` is followed by `kVCvtNormI16F16Vop1 = 77` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/opcodes.h:349` through
  `:351`.
- The VOP1 decoder table routes slots 75 and 76 to `decodeInvalid` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/decoder.cpp:6619` through
  `:6623`.
- Searches of generated CDNA1 source find no `VExpLegacyF32` or
  `VLogLegacyF32` classes, decode functions, or test-encoding rows. CDNA4 does
  generate these classes and opcodes, which points back to the CDNA1 input XML
  omission tracked by `CDNA1-XML-053`.

Impact:

Legal CDNA1 legacy transcendental instructions are rejected by rocjitsu before
execution. This is an XML-fed generated decode gap, not just a missing runtime
helper.

### CDNA1-RJ-086: `V_READFIRSTLANE_B32` drops direct-LDS and zero-EXEC behavior

Manual/XML/oracle evidence:

- Chapter 12.8 says `V_READFIRSTLANE_B32` reads a VGPR source or M0-backed LDS
  direct source, selects the first active lane, uses lane 0 when `EXEC` is zero,
  and ignores the exec mask for the source access at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:3445`.
- XML assigns `SRC0` the `OPR_VGPR_OR_LDS` operand class at
  `workspace_docs/amdgpu_isa_cdna1.xml:48676` through `:48680`; that class
  defines selector 254 as `SRC_LDS_DIRECT` at
  `workspace_docs/amdgpu_isa_cdna1.xml:125913` through `:125923`.
- As a `gfx908` oracle, `llvm-mc` accepts
  `v_readfirstlane_b32 s0, src_lds_direct`.

Rocjitsu evidence:

- The generated compact VOP1 constructor uses `OPR_SRC_VGPR` for `SRC0`, not
  `OPR_VGPR_OR_LDS`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop1.cpp:230` through
  `:234`. The generated promoted VOP3 constructor does the same at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop3.cpp:95` through `:104`.
- The generic operand resolver only maps `OPR_VGPR_OR_LDS` values in the VGPR
  range and has no LDS-direct read path for selector 254 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/operand.cpp:1130` through
  `:1138` and `:1461` through `:1473`.
- Both VOP1 and VOP3 execute paths initialize the result to zero and only read a
  lane after finding a set bit in `EXEC` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop1.cpp:324` through
  `:332` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop3.cpp:106`
  through `:115`.

Impact:

The legal `src_lds_direct` source form is not executable, and the all-inactive
`EXEC==0` case returns zero instead of reading source lane 0. Kernels that use
readfirstlane for scalarization can silently get the wrong scalar when the
active mask is empty, and direct-LDS encodings cannot follow the hardware
contract.

### CDNA1-RJ-087: Special VOP1 rows decode but throw at execution

Manual/XML/oracle evidence:

- Chapter 12.8 defines `V_SCREEN_PARTITION_4SE_B32` as a lookup-table operation
  on `S0.u[7:0]` at `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:3555`
  and repeats opcode 55 in the Chapter 13.3.2 VOP1 table at `:6130`.
- Chapter 12.8 defines `V_SAT_PK_U8_I16` as packing saturated 16-bit halves
  into a 32-bit result at `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:3697`
  and repeats opcode 79 at `:6158`.
- XML includes compact, literal/DPP/SDWA, and promoted VOP3 rows for
  `V_SCREEN_PARTITION_4SE_B32` at `workspace_docs/amdgpu_isa_cdna1.xml:53734`
  through `:53830`, and corresponding `V_SAT_PK_U8_I16` rows at
  `workspace_docs/amdgpu_isa_cdna1.xml:56128` through `:56225`.
- As a `gfx908` oracle, `llvm-mc` assembles both E32 and E64 forms for these
  two instructions.

Rocjitsu evidence:

- Generated CDNA1 opcode constants and test-encoding rows include
  `V_SCREEN_PARTITION_4SE_B32` and `V_SAT_PK_U8_I16` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/opcodes.h:331` through
  `:353` and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/test_encodings.h:337`
  through `:358`, with promoted VOP3 rows at `test_encodings.h:706` through
  `:727`.
- The VOP1 `V_SCREEN_PARTITION_4SE_B32` execute path throws
  `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop1.cpp:6758` through
  `:6760`; the promoted VOP3 path throws at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop3.cpp:871` through
  `:873`.
- The VOP1 `V_SAT_PK_U8_I16` execute path throws at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop1.cpp:9559` through
  `:9561`; the promoted VOP3 path throws at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop3.cpp:1876` through
  `:1878`.

Impact:

These legal CDNA1 VOP1 instructions are decodable but cannot be emulated. The
generated smoke encodings catch their presence in the decoder table, but not
the missing semantics.

### CDNA1-RJ-088: `V_SWAP_B32` accepts illegal source-extension forms before failing later

Manual/XML/oracle evidence:

- Chapter 12.8 says `V_SWAP_B32` swaps `D` and `S0`, does not support input or
  output modifiers, and is an untyped operation at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:3703`.
- XML lists only compact `ENC_VOP1` and promoted `ENC_VOP3` rows for
  `V_SWAP_B32`; it does not list `VOP1_INST_LITERAL`, DPP, or SDWA variants at
  `workspace_docs/amdgpu_isa_cdna1.xml:56242` through `:56283`.
- As a `gfx908` oracle, `llvm-mc` accepts `v_swap_b32 v0, v1`, but rejects
  literal, SGPR, DPP, and `clamp` forms.

Rocjitsu evidence:

- The generated compact VOP1 constructor starts with the correct
  `OPR_SRC_VGPR` operand class, but then still handles raw `src0 == 255`,
  `SRC_DPP`, and `SRC_SDWA` extension selectors at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop1.cpp:9564` through
  `:9601`.
- The execute path later writes through both operands at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop1.cpp:9662` through
  `:9669`, so a literal or scalar-like decoded source is not rejected at decode
  time and can fail as a non-VGPR write target instead of being reported as an
  illegal instruction form.

Impact:

Raw invalid `V_SWAP_B32` encodings can pass the generated constructor's
legality surface and fail later with operand/runtime behavior that does not
match the ISA or assembler contract. This is narrower than the general VALU
scalar-source budget issue in `CDNA1-RJ-074`: `V_SWAP_B32` has a VGPR-only
read/write source operand and no literal/DPP/SDWA forms in XML.

### CDNA1-RJ-089: Compact VOPC VCC/EXEC writes are invisible to generic def-use and probe-clobber analysis

Manual/XML evidence:

- Chapter 12.9 says VOPC compares produce one result bit per lane into VCC or
  EXEC at `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:3717`, and the
  detailed class rows describe CMPX forms as writing `EXEC = VCC` at
  `:3818`, `:3842`, and `:3861`.
- XML exposes compact VOPC result operands: representative `V_CMP_EQ_U32`
  encodings include an `OPR_VCC` output at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:86647` through
  `:86669`, and representative `V_CMPX_EQ_U32` encodings include both
  `OPR_VCC` and implicit `OPR_SDST_EXEC` outputs at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:88677` through
  `:88703`.

Rocjitsu evidence:

- Generated compact VOPC constructors expose only source operands and set
  `num_dst_ = 0`; representative `VCmpClassF32Vopc` does this at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vopc.cpp:23` through
  `:31`, and representative `VCmpxEqU32Vopc` does the same at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vopc.cpp:19458` through
  `:19466`.
- The CDNA1 `Vopc` base class declares no `implicit_defs()` override at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/encodings.h:505` through
  `:530`, while the default `Instruction::implicit_defs()` is a no-op at
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:215` through `:222`.
- The runtime side effects are present: representative compact compare helpers
  call `wf.set_vcc(vcc)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:4507`
  through `:4519`, and representative compact CMPX execution writes both VCC
  and EXEC at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vopc.cpp:19554` through
  `:19565`.
- `probe_clobber.cpp` explicitly notes that implicit VCC/EXEC writes from
  `v_cmp`/`v_cmpx` have no operand and remain invisible to its special-state
  scan at `lib/rocjitsu/src/rocjitsu/code/patch/probe_clobber.cpp:48` through
  `:61`; generic def-use similarly relies on explicit destination operands and
  `implicit_defs()` at `lib/rocjitsu/src/rocjitsu/analysis/def_use_chain.cpp:25`
  through `:45`.

Impact:

Instruction execution updates VCC/EXEC, but analysis and probe-clobber
summaries can miss compact VOPC special-state clobbers. A probe or patch body
containing `v_cmp*`/`v_cmpx*` can therefore look less stateful than it is, which
can under-drive save/restore or rejection policy for VCC/EXEC-sensitive code.

### CDNA1-RJ-090: `V_DIV_SCALE_F32/F64` zero numerator or denominator cases return `S0` instead of NaN

Manual/XML evidence:

- CDNA1 `V_DIV_SCALE_F32` and `V_DIV_SCALE_F64` initialize the post-scale mask to
  zero and set `D` to NaN when either the numerator or denominator is zero at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:4344` through `:4351`.
- XML identifies both as VOP3B operations with explicit `VDST` and `SDST`
  outputs and source operands at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:64154` through
  `:64232`, but it does not carry the detailed zero-case pseudocode.

Rocjitsu evidence:

- `execute_v_div_scale_f32_vop3()` initializes `result = s0`, then leaves that
  result unchanged when `s2 == 0.0f || s1 == 0.0f` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:10312`
  through `:10317`.
- `execute_v_div_scale_f64_vop3()` has the same pass-through zero-case behavior
  for double precision at `execute_shared.h:10371` through `:10375`.
- Searches found no targeted CDNA1 golden test for the zero numerator or
  denominator `V_DIV_SCALE_F32/F64` cases; nearby `V_DIV_SCALE_F32` coverage in
  `tests/gfx1250_sim_test.cpp:1100` through `:1141` exercises nonzero
  post-scale behavior and SDST selector routing on gfx1250.

Impact:

Zero numerator or denominator inputs produce the unscaled input operand rather
than the manual's NaN pre-scale result, so the state entering the later
`V_DIV_FIXUP_*` sequence can diverge from CDNA1 hardware behavior.

### CDNA1-RJ-091: VINTERP compact and VOP3 interpolation execution is missing

Manual/XML evidence:

- CDNA1 compact `V_INTERP_P1_F32`, `V_INTERP_P2_F32`, and `V_INTERP_MOV_F32`
  define parameter interpolation/load formulas at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:4183` through `:4198`, and
  Chapter 13.4.1 repeats the compact VINTRP field formulas at `:6940` through
  `:6963`.
- CDNA1 VOP3 F16 interpolation rows define LDS/VGPR half-selection, LDS-bank
  restrictions, and destination-half writeback at `cdna1/README.md:4408`
  through `:4420`.
- XML contains the corresponding compact and promoted F32 rows at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:46275` through
  `:46490`, and the VOP3 F16 rows at `:66062` through `:66230`.

Rocjitsu evidence:

- Generated compact VINTRP `V_INTERP_P1_F32`, `V_INTERP_P2_F32`, and
  `V_INTERP_MOV_F32` constructors are present, but their `execute_impl()` bodies
  are no-ops at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vintrp.cpp:19` through
  `:68`.
- The promoted VOP3 F32 interpolation constructors are likewise present, but
  their execution bodies are no-ops at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop3.cpp:23` through `:72`.
- The VOP3 F16 interpolation constructors are generated, but
  `V_INTERP_P1LL_F16`, `V_INTERP_P1LV_F16`, `V_INTERP_P2_LEGACY_F16`, and
  `V_INTERP_P2_F16` all throw `util::UnimplementedInst` at `vop3.cpp:5423`
  through `:5498`.
- Searches found no dedicated `v_interp` coverage under `emulation/rocjitsu/tests`;
  generated oracle encodings only cover decode/disassembly smoke vectors in
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/test_encodings.h:616`
  through `:618`, `:650` through `:652`, and `:849` through `:852`.

Impact:

Any CDNA1 shader that depends on interpolation parameter loads/computation will
either leave the destination unchanged for the F32 forms or trap as unimplemented
for the F16 forms, even though the opcodes decode.

### CDNA1-RJ-092: `DS_CONDXCHG32_RTN_B64` executes as ordinary 64-bit compare-swap

Manual/XML evidence:

- CDNA1 lists `DS_CONDXCHG32_RTN_B64` as opcode 126 and describes it as a
  conditional write exchange at
  `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:4633` and `:7112`.
- XML gives the executable contract: the instruction performs two conditional
  32-bit write exchanges, each writing a data-register value to data-share
  memory iff that data value's most-significant bit is set, at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:13481` through
  `:13517`.

Rocjitsu evidence:

- Generated `DsCondxchg32RtnB64Ds` exposes `VDST`, `ADDR`, and a 64-bit `DATA0`
  source, but its execute path sets `elem_size = 8`, `atomic_op =
  AtomicOp::CMPSWAP`, and then reads both the `DATA0` and raw `DATA1` VGPR
  pairs into a 16-byte per-lane atomic payload at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/ds.cpp:3919` through
  `:3961`.
- The LDS atomic helper handles `AtomicOp::CMPSWAP` as a single 64-bit compare
  against the second source and writes the first source only when the old 64-bit
  memory value equals that compare value, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:247` through `:253`
  and `:446` through `:468`.
- Static searches found no dedicated `ds_condxchg32_rtn_b64` correctness test;
  generated test encodings only include a decode/disassembly smoke vector at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/test_encodings.h:1190`.

Impact:

The instruction consumes a hidden `DATA1` pair and conditionally writes based on
memory equality instead of per-dword source MSB bits. Programs using the
conditional-exchange contract can update neither the correct dword lanes nor the
correct memory values.

### CDNA1-RJ-093: `BUFFER_ATOMIC_PK_ADD_F16` executes as a single F32 atomic add

Manual/XML evidence:

- CDNA1 `BUFFER_ATOMIC_ADD_F32` adds a single 32-bit floating value, while
  `BUFFER_ATOMIC_PK_ADD_F16` independently adds the high and low packed F16
  components at `workspace_docs/amdgpu-isa-manuals/cdna1/README.md:4916`
  through `:4917`.
- XML preserves the same distinction: `BUFFER_ATOMIC_ADD_F32` is described as a
  single-precision float add, and `BUFFER_ATOMIC_PK_ADD_F16` as a packed
  two-component half-precision add at
  `shared/machine-readable-isa/isa/amdgpu_isa_cdna1.xml:27767` through
  `:27816`.

Rocjitsu evidence:

- Generated CDNA1 `BufferAtomicPkAddF16Mubuf` decodes opcode 78 and reads one
  32-bit `VDATA` word, but sets `d->elem_size = 4` and `d->atomic_op =
  AtomicOp::FADD`, exactly like `BufferAtomicAddF32Mubuf`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mubuf.cpp:1997` through
  `:2064`.
- The shared memory pipeline treats any 4-byte `AtomicOp::FADD` payload as one
  IEEE F32 value by bit-casting the whole dword to `float`, adding once, and
  writing the whole dword back at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:321` through `:343`.
- Static searches found no dedicated `buffer_atomic_pk_add_f16` correctness
  test under `emulation/rocjitsu/tests`; the generated oracle fixture is only a
  decode/disassembly smoke row at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/test_encodings.h:1262`.

Impact:

Any packed half inputs whose bit pattern is not equivalent to the desired F32
operation will update memory with a single 32-bit float-add result instead of
two independent F16 additions. This also reuses the ordinary FP atomic path's
MODE/rounding behavior, so it does not model packed-half atomic edge cases.

### CDNA1-RJ-094: Compact DPP/SDWA constructors accept Chapter 12.18-prohibited forms

Manual/XML/oracle evidence:

- CDNA1 Chapter 12.18 says `V_MADMK_F32`, `V_MADAK_F32`,
  `V_MADMK_F16`, `V_MADAK_F16`, `V_READFIRSTLANE_B32`, F64 VOP1
  conversion/transcendental rows, `V_CLREXCP`, `V_SWAP_B32`, and F64/I64/U64
  compare/class families cannot use DPP at `cdna1/README.md:5309` through
  `:5342`.
- The same section says `V_MAC_F32`, `V_MAC_F16`, `V_FMAC_F32`, the
  `V_MADMK/MADAK_*` rows, `V_READFIRSTLANE_B32`, `V_CLREXCP`, and
  `V_SWAP_B32` cannot use SDWA at `cdna1/README.md:5344` through `:5357`.
- The CDNA1 XML omits representative prohibited extension rows such as DPP for
  `V_READFIRSTLANE_B32` and `V_RCP_F64`, and SDWA for `V_MAC_F32`; `llvm-mc`
  for `gfx908` likewise rejects `v_readfirstlane_b32_dpp`,
  `v_cmp_eq_f64_dpp`, and `v_mac_f32_sdwa`.

Rocjitsu evidence:

- Generated CDNA1 `V_READFIRSTLANE_B32` accepts both `SRC_DPP` and `SRC_SDWA`
  markers in its compact constructor at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop1.cpp:243` through
  `:265`.
- Generated CDNA1 `V_RCP_F64` accepts both markers at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop1.cpp:4568` through
  `:4588`, despite `V_RCP_F64` being listed in the DPP-prohibited set.
- Generated CDNA1 `V_MAC_F32` accepts `SRC_SDWA` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop2.cpp:2946` through
  `:2962`, despite `V_MAC_F32` being listed in the SDWA-prohibited set.
- Generated CDNA1 `V_CMP_CLASS_F64`, `V_CMP_EQ_F64`, and ordinary
  `V_CMP_EQ_U32` all accept `SRC_DPP` by casting the word to
  `Vop1VopDppMachineInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vopc.cpp:302` through
  `:309`, `:8834` through `:8841`, and `:17522` through `:17529`. The first
  two are explicitly DPP-prohibited; the ordinary U32 case also conflicts with
  CDNA XML/LLVM, which expose no compact VOPC DPP rows for CDNA1 through CDNA4.
- Existing tests synthesize CDNA VOPC DPP directly with
  `Vop1VopDppMachineInst` at `tests/shared_infra_test.cpp:1989` through
  `:2040`, and a CDNA4 liveness test real-decodes `v_rcp_f64` with
  `src0=SRC_DPP` at `tests/analysis/liveness_test.cpp:939` through `:952`.

Impact:

Rocjitsu can decode and execute illegal compact DPP/SDWA encodings instead of
rejecting them or reporting an unsupported instruction. The existing shared
DPP tests also make some illegal CDNA forms look intentional, which can hide
future legality regressions around the Chapter 12.18 contract.

## No-Gap Notes

- CDNA1 Chapter 1-2 Wave64, wavefront/workgroup/dispatch, initial `EXEC`, and
  work-item-ID basics are represented by the CDNA ISA profile and production
  dispatch path: `CdnaIsaBase` fixes `WF_SIZE` and `WF_SIZE_MAX` at 64 in
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/cdna_isa_base.h:83`
  through `:99`; the Python profile mirrors Wave64 at
  `lib/python/amdisa/isa_profile.py:790` through `:799`;
  `config_loader.cpp` enables packed work-item IDs for CDNA2+ but not CDNA1 at
  `lib/rocjitsu/src/rocjitsu/config/config_loader.cpp:441` through `:447`;
  `ComputeUnitCore::dispatch_wf()` records workgroup and allocation state and
  initializes `EXEC`, `VCC`, and `M0` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:95` through `:146`;
  `DispatchEntry` computes local work-item coordinates and initial active-lane
  masks at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/dispatch_entry.h:196` through
  `:239`; and `CommandProcessor::init_wavefront_regs()` writes workgroup IDs
  and work-item IDs at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:265` through
  `:321`. Broader memory hierarchy, exception, and cache-ordering behavior
  remains covered by the narrower memory/state gaps above.
- CDNA1 has no generated packed-F32 arithmetic or `V_PK_MOV_B32` VOP3P classes,
  matching their absence from the CDNA1 manual/XML slice.
- CDNA1 branch consumers use live EXEC/VCC state for `S_CBRANCH_EXECZ`,
  `S_CBRANCH_EXECNZ`, `S_CBRANCH_VCCZ`, and `S_CBRANCH_VCCNZ` at
  `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopp.cpp:111`
  through `:205`; `CDNA1-RJ-019` is about stale raw STATUS helper bits, not
  those branch decisions.
- The normal CDNA1 direct-PC helper bodies write the next PC and apply the
  scalar target as described in Chapter 3; `CDNA1-RJ-018` is specifically the
  pre-execute zero-target halt in the CU issue loop.
- The generated CDNA1 VOP2 opcode and primary-decode inventories match the
  manual/XML opcode list for Chapter 12.7/13.3.1: `opcodes.h` spans
  `kVCndmaskB32Vop2 = 0` through `kVXnorB32Vop2 = 61` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/opcodes.h:553` through
  `:614`, and a scripted count of the primary decode table found 248 entries,
  four slots for each of the 62 VOP2 opcodes, with no holes from
  `decodeVCndmaskB32Vop2` through `decodeVXnorB32Vop2` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/decoder.cpp:5896`
  through `:6143`.
- CDNA1 declared 16-bit VOP2 literal rows keep the low-half literal contract:
  generated `V_MADMK_F16` and `V_MADAK_F16` constructors mask `SIMM32` with
  `0xFFFFu` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop2.cpp:4857` through
  `:4885` and `:5010` through `:5038`, shared execution reads
  `static_cast<uint16_t>(inst.simm32_)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:11888`
  through `:11905`, and tests cover declared-literal decode and VOP2 FMA
  execution at `tests/decode_smoke_test.cpp:146` through `:165` and
  `tests/simd_correctness/vop2_fma_simd_correctness_test.cpp:161` through
  `:189`.
- MIX helpers implement the MIX-specific selector mapping, treat `NEG_HI` as
  absolute value, and apply `CLMP` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:13290`
  through `:13415`. They use multiply-add rather than fused FMA; the detailed
  CDNA1 instruction table and XML descriptions support multiply-add, while
  section 6.7 contains conflicting fused wording.
- CDNA1 `S_MAX_I32` and `S_MAX_U32` use the strict CDNA1 manual predicate:
  detailed rows set SCC with `S0 > S1` at `cdna1/README.md:2869` through
  `:2870`, and the shared helpers write SCC with the same strict comparison at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1866`
  through `:1898`. The inclusive max-equality gap recorded for CDNA3/CDNA4 does
  not apply to CDNA1.
- Representative CDNA1 signed/unsigned add and carry helpers match the detailed
  SCC formulas: signed add uses overflow and `S_ADDC_U32` uses unsigned
  carry-out at `cdna1/README.md:2862` through `:2865`, matching
  `execute_shared.h:418` through `:451`.
- The generated CDNA1 SOP2 opcode inventory matches the Chapter 12.1 manual/XML
  opcode list after normalizing the manual OCR wrap on `S_PACK_LL_B32_B16`:
  generated test encodings span the SOP2 rows from `s_add_u32` through
  `s_pack_hh_b32_b16` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/test_encodings.h:147`
  through `:199`, and generated constructors span `SAddU32Sop2` through
  `SPackHhB32B16Sop2` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sop2.cpp:20` through
  `:1260`.
- Ordinary CDNA1 SOP2 double-literal decoding uses the single extension word for
  both source operands when both raw selectors are 255, matching the LLVM
  `gfx908` behavior for two identical literals: representative `SAddU32Sop2`
  reads the same `Sop2InstLiteralMachineInst::simm32` for both operands at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sop2.cpp:20` through `:40`.
- The shared scalar-pack helpers implement the Chapter 12.1 half-selection rows
  for `S_PACK_LL_B32_B16`, `S_PACK_LH_B32_B16`, and `S_PACK_HH_B32_B16` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2344`
  through `:2372`; no pack half-selection runtime gap was found in this pass.
- The shared SOP2 bitfield, shift, high-multiply, and shifted-add helpers match
  the Chapter 12.1 masking/carry patterns for representative rows: BFM masks
  width and offset in `simd_glue.h:60` through `:64` and
  `execute_shared.h:880` through `:892`, BFE clamps extracted width in
  `execute_shared.h:800` through `:871`, shifts mask the count in
  `execute_shared.h:1832` through `:1864`, high-multiply helpers use 64-bit
  products at `:2055` through `:2077`, and shifted-add helpers set SCC on
  unsigned carry at `:1784` through `:1830`.
- The generated CDNA1 SOPK opcode inventory matches the Chapter 12.2 manual/XML
  opcode list: generated encodings span `s_movk_i32` through `s_setreg_b32`,
  skip opcode 19, and include `s_call_b64` at opcode 21 in
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/test_encodings.h:127`
  through `:146`; generated opcode constants likewise omit opcode 19 and place
  `kSSetregImm32B32Sopk` at 20 and `kSCallB64Sopk` at 21 in
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/opcodes.h:128` through
  `:141`. The CDNA1 decoder table keeps opcode 19 invalid and routes opcodes
  20 and 21 to the generated literal/call decoders at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/decoder.cpp:6248` through
  `:6269`.
- CDNA1 SOPK literal sizing matches the ISA/LLVM oracle for the opcode-20 form:
  `Sopk::hasImpliedLiteral()` returns true only for opcode 20 and loads the
  following dword at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/encodings.cpp:68` through
  `:82`. `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908 --show-encoding`
  encoded `s_setreg_imm32_b32 hwreg(HW_REG_MODE), 0x12345678` as
  `[0x01,0xf8,0x00,0xba,0x78,0x56,0x34,0x12]`, and
  `llvm-mc --disassemble` rejected raw opcode 19 bytes while decoding opcode
  20 with the extension word.
- The shared SOPK helpers match the audited immediate and SCC rules for ordinary
  scalar arithmetic/compare rows: `S_MOVK_I32` and `S_CMOVK_I32` sign-extend
  `SIMM16`, `S_CMPK_*_I32` sign-extend while `S_CMPK_*_U32` zero-extend,
  `S_ADDK_I32` writes SCC on signed overflow, and `S_MULK_I32` reads the old
  destination without writing SCC at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:455`
  through `:466`, `:1095` through `:1100`, `:1412` through `:1504`, and
  `:2033` through `:2101`. Adjacent regression coverage includes ADDK, MULK,
  register dataflow, and SOPK compare SCC cases at
  `tests/scalar_scc_test.cpp:622`, `:660`, `:665`, and `:872`.
- The remaining CDNA1 SOPK semantic gaps are covered by existing findings:
  HWREG ID/permission/literal-operand metadata by `CDNA1-RJ-014` through
  `CDNA1-RJ-016`, branch-stack execution by `CDNA1-RJ-067`, and
  `S_CALL_B64`'s direct-PC-relative instruction being tagged `INDIRECT_CALL` by
  `CDNA1-RJ-072`.
- The generated CDNA1 SOP1 opcode inventory matches the Chapter 12.3 and
  Chapter 13.1.3 manual/XML opcode set: generated encodings span `s_mov_b32`
  through `s_bitreplicate_b64_b32`, with opcode holes 47 and 49 preserved, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/test_encodings.h:22`
  through `:75`; opcode constants likewise skip 47 and 49 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/opcodes.h:16` through
  `:69`; and the decoder table routes those holes to `decodeInvalid` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/decoder.cpp:6807` through
  `:6862`. `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908 --show-encoding`
  also assembled representative SOP1 rows with the same opcodes, while
  `llvm-mc --disassemble` rejected raw opcode-47 and opcode-49 bytes.
- The audited CDNA1 SOP1 runtime helpers did not expose a new helper-level gap
  for the unary/bit-manipulation rows. Representative shared helpers implement
  unsigned fixed-point `S_ABS_I32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:346`
  through `:354`, bit counts at `:764` through `:797`, bitreplicate and
  destination-preserving bitset at `:929` through `:970`, bit scans/sign-run at
  `:1648` through `:1728`, and quad-mask reduction at `:2376` through `:2402`.
- Existing test coverage exercises the highest-risk SOP1 scalar-state cases:
  wrexec availability/def-use and execution at `tests/scalar_scc_test.cpp:366`
  through `:383`, saveexec boolean/SCC behavior at `:407` through `:583`, and
  scalar scan SCC preservation at `:957` through `:980`. A gfx1250 simulation
  test covers bitreplicate expansion at `tests/gfx1250_sim_test.cpp:3804`
  through `:3815`; this is adjacent shared-helper coverage, not CDNA1-specific
  full validation.
- The remaining CDNA1 SOP1 semantic misses are already tracked by existing
  findings: relative SGPR addressing by `CDNA1-RJ-017`, direct-PC zero-target
  behavior by `CDNA1-RJ-018`, trap/RFE state by `CDNA1-RJ-022`, fork/join
  execution by `CDNA1-RJ-067`, and GPR-index M0 layout by `CDNA1-RJ-077`.
- The generated CDNA1 SOPC opcode inventory matches the Chapter 12.4 manual/XML
  opcode list: generated encodings span `s_cmp_eq_i32` through
  `s_cmp_lg_u64` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/test_encodings.h:76`
  through `:95`; opcode constants likewise place `kSCmpEqI32Sopc` at 0 through
  `kSCmpLgU64Sopc` at 19 in
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/opcodes.h:70` through
  `:89`; and the decoder table routes opcodes 0 through 19 before invalid rows
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/decoder.cpp:7066`
  through `:7078`.
- The audited CDNA1 SOPC compare and bit-compare helpers did not expose a new
  helper-level gap. Shared bit-compare helpers mask B32 indices to five bits
  and B64 indices to six bits, and shared compare helpers implement the signed,
  unsigned, and U64 SCC predicates at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:897`
  through `:925` and `:1116` through `:1277`.
- Existing adjacent tests cover representative SOPC SCC producers and consumers:
  `SopcAndSopkCompareFeedSccConsumer` exercises `s_cmp_eq_i32` and
  `s_bitcmp0_b32` at `tests/scalar_scc_test.cpp:816` through `:875`, and the
  VM smoke tests cover representative SOPC compare execution at
  `tests/amdgpu_vm_test.cpp:1416` through `:1432`. The VSKIP execution miss is
  already tracked by `CDNA1-RJ-020`, and the GPR-index layout miss is tracked by
  `CDNA1-RJ-077`.
- The generated CDNA1 SOPP class and decoder inventory matches the manual/XML
  opcode inventory: generated encodings span `s_nop` through
  `s_endpgm_ordered_ps_done` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/test_encodings.h:96`
  through `:126`, opcode constants place `kSNopSopp` at 0 through
  `kSEndpgmOrderedPsDoneSopp` at 30 in
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/opcodes.h:90` through
  `:120`, and `sub_decode_sopp` maps opcodes 0 through 30 before invalid
  entries at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/decoder.cpp:7137`
  through `:7168`. `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908` assembled
  representative SOPP rows with the same opcode positions and rejected raw
  opcode-31 and opcode-32 SOPP encodings.
- Ordinary CDNA1 PC-relative SOPP branches are implemented: `S_BRANCH` and the
  SCC/VCC/EXEC conditional forms sign-extend the 16-bit instruction-count
  offset, scale it by four bytes, update `wf.pc`, and expose branch metadata at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopp.cpp:39` through
  `:200`. Adjacent VM tests cover `S_BRANCH` and SCC conditional branch cases
  at `tests/amdgpu_vm_test.cpp:1442` through `:1505`.
- CDNA1 `S_WAITCNT` threshold decode itself is present: the generated SOPP
  executor combines VM low/high bits, extracts EXP and LGKM, and calls
  `wf.set_wait_target(vm, lgkm, exp)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopp.cpp:225` through
  `:240`. The wait-counter gaps above are about producer accounting and
  architecture-specific hazard rules, not absence of the threshold instruction.
- The normal `S_ENDPGM` and `S_ENDPGM_SAVED` drain-before-halt behavior is
  represented at the simulator wait-counter level: the generated SOPP bodies
  call `wf.end()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/sopp.cpp:30` through `:37`
  and `:415` through `:423`, `Wavefront::end()` waits in `ENDING` when counters
  remain at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:446` through
  `:454`, and `ComputeUnitCore::update_wf_states()` halts `ENDING` waves only
  after counters drain at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:301`
  through `:310`. The ordered-PS message side effect remains covered by
  `CDNA1-RJ-069`.
- The core classic `S_BARRIER` release path is represented: the generated
  instruction sets `WfState::BARRIER`, and the CU releases all non-halted waves
  in the same dispatch/workgroup once they have all reached the barrier at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:745`
  through `:747` and
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:313` through `:335`.
- The generated CDNA1 SMEM opcode inventory matches the Chapter 12.6/13.2.1
  manual/XML opcode list: generated test encodings span `s_load_dword` through
  `s_atomic_dec_x2` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/test_encodings.h:200`
  through `:283`, opcode constants place `kSLoadDwordSmem` at 0 through
  `kSAtomicDecX2Smem` at 172 in
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/opcodes.h:195` through
  `:278`, and `sub_decode_smem` maps the same 84 non-invalid sparse slots at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/decoder.cpp:7270`
  through `:7442`. A scripted comparison found 84 XML SMEM opcodes and 84
  generated decoder entries with no missing or extra slots, and `llvm-mc`
  assembled representative gfx908 forms for loads/stores, cache/time, ATC
  probe, discard, scalar atomics, and scalar-buffer atomics.
- The generated CDNA1 SMEM constructors preserve the main operand-shape
  distinctions from XML/manual inventory: ordinary scalar memory rows use
  64-bit `SBASE`, scalar-buffer rows use 128-bit `SBASE`, time rows have only a
  64-bit `SDATA` destination, cache invalidation/writeback rows have no
  operands, ATC probe rows use a `SIMM8` payload plus address operands, and
  discard rows omit `SDATA`, as shown across
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/smem.h:17` through `:748`.
  The remaining SMEM misses are semantic/runtime issues already tracked by
  `CDNA1-RJ-060` through `CDNA1-RJ-066`, not a separate decode-inventory gap.
- CDNA1 has basic end-to-end VALU smoke coverage for VOP1/VOP2/VOPC-style
  execution in the shared ISA harness: `V_MOV_B32`, `V_ADD_F32`, `V_MUL_F32`,
  `V_ADD_U32`, `V_CMP_EQ_F32`, and `V_CNDMASK_B32` are exercised at
  `tests/amdgpu_vm_test.cpp:1518` through `:1588`. The Chapter 6 test gap above
  is about the stateful and legality edge cases, not total absence of VALU
  tests.
- The generated CDNA1 VOP1 opcode inventory matches the XML-fed subset: opcode
  constants span `V_NOP` through `V_SWAP_B32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/opcodes.h:279` through
  `:354`, and the decoder table maps opcodes 0 through 81 with invalid slots at
  9, 54, 56, 75, 76, and 80 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/decoder.cpp:6544` through
  `:6627`. `CDNA1-RJ-085` covers the two manual/LLVM legacy rows missing from
  that generated subset.
- The generated `V_SWAP_B32` def-use metadata correctly treats both operands as
  source and destination operands for the ordinary VGPR form at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop1.cpp:9564` through
  `:9574` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop3.cpp:1881`
  through `:1890`. `CDNA1-RJ-088` is about illegal extension/source forms in the
  compact constructor, and `CDNA1-XML-055` tracks the upstream XML metadata
  omission.
- The generated CDNA1 VOPC opcode inventory matches the manual/XML opcode set:
  generated opcode constants cover the 198 compact VOPC rows from
  `kVCmpClassF32Vopc` through `kVCmpxTU64Vopc`, and a scripted comparison found
  the same holes at 0-15, 22-31, and 128-159 as the Chapter 13.3.3 opcode
  table. Generated test encodings cover compact rows at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/test_encodings.h:360`
  through `:557` and promoted VOP3 compare rows beginning at `:884`.
- The audited CDNA1 VOPC runtime helpers did not expose a new compare/class
  semantics gap. Representative signed 16-bit helpers sign-extend the low half
  and unsigned 16-bit helpers mask to 16 bits at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:4471`
  through `:4584` and `:6282` through `:6395`; class helpers implement the ten
  IEEE class bits for F16/F32/F64 at `:3961` through `:4028`; compact CMPX
  paths update VCC and EXEC at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vopc.cpp:19554` through
  `:19565`; and promoted VOP3 CMPX paths write the scalar destination and EXEC
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop3.cpp:8038` through
  `:8075`.
- Existing adjacent tests cover important VOPC runtime edges found in this
  slice: DPP write-mask and CMPX EXEC preservation mechanics across CDNA/RDNA
  targets at `tests/shared_infra_test.cpp:1989` through `:2100` and `:2450`
  through `:2709`, scalar class-mask behavior at
  `tests/simd_correctness/vopc_cmp_class_scalar_correctness_test.cpp:47`
  through `:155`, and scalar-versus-SIMD class-mask equivalence at
  `tests/simd_correctness/vopc_cmp_class_simd_correctness_test.cpp:48`
  through `:319`. `CDNA1-RJ-094` tracks the separate legality issue for the
  synthesized CDNA compact DPP forms used by some of those shared tests.
- The CDNA1 generated VOP3A/VOP3B opcode inventory matches the manual/XML slice
  for native VOP3A rows and VOP3B scalar-destination rows: representative
  decoder slots for native VOP3A opcodes start at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/decoder.cpp:7979`, and
  VOP3B carry/div-scale/wide-MAD constructors are present at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop3.cpp:12047` through
  `:12248`.
- Generated VOP3B carry and wide-MAD scalar-output paths are not newly flagged
  by this slice: carry helpers write `inst.sdst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2913`
  through `:2937`, `:3286` through `:3316`, and `:18839` through `:18868`,
  and wide-MAD helpers write `inst.sdst` at `execute_shared.h:13201` through
  `:13232` and `:13467` through `:13492`. Adjacent SIMD tests compare the
  scalar destination for VOP3 carry and wide-MAD cases at
  `tests/simd_correctness/vop3_carry_simd_correctness_test.cpp:232` through
  `:248` and
  `tests/simd_correctness/vop3_shift64_simd_correctness_test.cpp:291` through
  `:303`.
- The generated CDNA1 interpolation decode inventory matches the manual/XML
  opcode slice: compact VINTRP decodes opcodes 0, 1, and 2 through
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/decoder.cpp:4443` through
  `:4457` and `:8691` through `:8696`; promoted/native VOP3 interpolation
  decodes opcodes 624, 625, 626, 628, 629, 630, and 631 at `decoder.cpp:4077`
  through `:4102` and `:8155` through `:8162`. `CDNA1-RJ-091` is about missing
  execution semantics, not missing opcode decode.
- The generated interpolation constructors preserve key def-use/source-order
  metadata despite the missing runtime behavior: compact and promoted
  `V_INTERP_P2_F32` treat `VDST` as both input and output at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vintrp.cpp:36` through
  `:48` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop3.cpp:40`
  through `:52`, while promoted VOP3 F32 rows expose `SRC1` as the
  VGPR/parameter source and `SRC0` as the attribute at `vop3.cpp:23` through
  `:67`.
- The generated CDNA1 DS decode inventory matches the XML-fed opcode set for
  Chapter 12.13/13.5.1. A scripted comparison against the manual's 154
  opcode/name pairs found missing constants and generated test encodings only
  for the 31 `DS_*_SRC2_*` rows already tracked by `CDNA1-RJ-051`; the generated
  set includes opcode 85 `DS_WRITE_B16_D16_HI` and opcode 126
  `DS_CONDXCHG32_RTN_B64` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/opcodes.h:1159` and
  `:1189`, with decode slots at `decoder.cpp:8787` and `:8828`.
- `DS_CONDXCHG32_RTN_B64` is decode-present and has a generated smoke encoding
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/test_encodings.h:1190`;
  `CDNA1-RJ-092` is about the executed conditional-exchange semantics, not a
  missing opcode row.
- The generated CDNA1 MUBUF/MTBUF decode inventory matches the Chapter
  12.14/12.15 and 13.6.1/13.6.2 manual/XML opcode set. A scripted comparison
  found all 16 MTBUF rows and all 71 sparse MUBUF rows present in generated
  opcode constants, decoder slots, and oracle encodings. Representative
  generated entries include `kBufferLoadFormatD16HiXMubuf = 38`,
  `kBufferStoreLdsDwordMubuf = 61`, `kBufferAtomicAddF32Mubuf = 77`,
  `kBufferAtomicPkAddF16Mubuf = 78`, and `kTbufferLoadFormatXMtbuf = 0` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/opcodes.h:1242` through
  `:1275`, with decoder routes at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/decoder.cpp:5314` through
  `:5453` and `:9134` through `:9229`. `CDNA1-RJ-093` is about the executed
  packed-half atomic semantics, not missing decode coverage.
- The generated CDNA1 MIMG decode inventory matches the Chapter 12.16/13.7.1
  manual/XML opcode set. A scripted comparison found all 92 sparse `IMAGE_*`
  rows present in generated opcode constants, decoder slots, and oracle
  encodings. Representative generated ranges span `kImageLoadMimg = 0` through
  `kImageSampleCCdClOMimg = 111` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/opcodes.h:1291` through
  `:1382`, fixture rows at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/test_encodings.h:1292`
  through `:1383`, and decoder slots from
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/decoder.cpp:9240` through
  `:9353`. Existing gaps `CDNA1-RJ-033` through `CDNA1-RJ-039` cover the image
  execution, operand-footprint, modifier, descriptor, wait-counter, and test
  coverage gaps rather than a missing opcode row.
- CDNA1 has no generated BF16_1K or F64 MFMA classes, matching their absence
  from the CDNA1 manual/XML slice.
- Valid non-F64 broadcast/swizzle fields are wired into the CDNA1 MFMA arithmetic
  helpers for F32/F16/I8/BF16 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/vop3p.cpp:618` through
  `:627`, `:773` through `:782`, `:928` through `:937`, and `:1083` through
  `:1092`; `CDNA1-RJ-006` is about missing legality validation for undefined
  `CBSZ` values.
- Raw CDNA1 D16 MUBUF low/high load writeback has explicit execution state:
  representative low/high loads set `d16_lo` and `d16_hi` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/mubuf.cpp:1123` and `:1169`,
  and the memory pipeline preserves or zeroes the other half based on ECC at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:181` through `:195`.
  The Chapter 9.1 gaps above are about formatted/resource conversion,
  descriptor addressing, LDS legality, and cache policy rather than every raw
  D16 half-writeback path.
- CDNA1 raw flat/global/scratch load/store and integer atomic opcodes are not
  wholly absent: generated constructors/execution paths exist for load/store
  opcodes 16 through 37 and integer atomic opcodes 64 through 76 and 96 through
  108 in `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/flat.cpp`. The
  Chapter 10 gaps above are about segment-specific legality, address
  calculation edge cases, direct-LDS, wait-counter/fault/cache behavior, and
  the legal Global FP atomics that are missing from the generated table.
- The generated CDNA1 flat-family decode inventory matches the shared
  `ENC_FLAT` opcode set used by rocjitsu's segment-normalized constructors: a
  scripted comparison found 48 generated opcode constants, 48 oracle encodings,
  and 48 decoder-table entries for opcodes 16 through 37, 64 through 76, and
  96 through 108. Representative generated rows span
  `kFlatLoadUbyteFlat = 16` through `kFlatAtomicDecX2Flat = 108` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/opcodes.h:1384` through
  `:1431`, fixture rows at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/test_encodings.h:1384`
  through `:1431`, and decoder slots at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna1/decoder.cpp:8979` through
  `:9071`. `CDNA1-RJ-046` tracks the two legal `ENC_FLAT_GLBL` opcode rows
  absent from that generated table: `GLOBAL_ATOMIC_ADD_F32` and
  `GLOBAL_ATOMIC_PK_ADD_F16` at opcodes 77 and 78.
