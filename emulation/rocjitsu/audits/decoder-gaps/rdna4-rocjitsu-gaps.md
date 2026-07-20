# RDNA4 Rocjitsu Gaps

Architecture: RDNA4 / gfx12

Manual source: `workspace_docs/amdgpu-isa-manuals/rdna4/README.md`

Rocjitsu source: `emulation/rocjitsu`

## Coverage

| Area | Status | Notes |
| --- | --- | --- |
| RDNA4 Chapter 1 introduction / hardware overview | Audited statically | Checked RDNA4 config/topology, ISA wave-size traits, command-processor dispatch setup, CU/WGP/LDS placement, L1/L2 cache surfaces, completion interrupts, and existing Chapter 1-overlap gaps. Detailed operational gaps remain tracked under later wave, LDS, cache, memory, and exception rows. |
| RDNA4 Chapter 2 shader concepts / wave modes / work-groups / padding / WQM | Audited statically | Checked compute dispatch and work-item initialization, wave-size traits, workgroup scheduling, CU/WGP placement, shader-padding/S_CODE_END handling, explicit S_WQM helpers, and existing detailed wave/barrier/LDS/image/export gaps. |
| RDNA4 Chapter 3 PC/EXEC wave state | Audited statically | Checked state storage, operand paths, branch/direct-PC behavior, `EXEC` masking/skipping boundaries, and related timing/issue gaps. |
| RDNA4 Chapter 3 SGPR/VGPR/LDS storage rules | Audited statically | Checked common operand and LDS storage paths, allocation state, out-of-range behavior, memory-family interactions, and related runtime gaps. |
| RDNA4 Chapter 3 dynamic VGPR | Audited statically | `S_ALLOC_VGPR` execution body checked. |
| RDNA4 Chapter 3 HWREG/state-register access | Audited statically | RDNA4 SOPK HWREG mapping and wavefront status/mode storage checked. |
| RDNA4 Chapter 3 compute launch state | Audited statically | Checked compute SGPR/VGPR/TTMP initialization, active-lane setup, scratch state, and the graphics-stage launch boundary recorded in `RDNA4-RJ-203`. |
| RDNA4 Chapter 4.1 common instruction fields | Audited statically | Checked literal and common selector behavior; DPP selector legality/details are tracked under Chapter 7.9 and aperture selectors under Chapter 6.10/11. |
| RDNA4 Chapter 4.1.1 cache controls | Audited statically | Checked generated `SCOPE`/`TH` decode, cache flag helpers, and scalar/vector memory pipelines. |
| RDNA4 SOPP wait/cache-maintenance definitions | Audited statically | Folded into the full Chapter 16.5 SOPP pass; existing wait/cache gaps retained. |
| RDNA4 Chapter 5.1/5.2 program flow and control instructions | Audited statically | Checked generated control instruction decode, obvious runtime helpers, and overlap with detailed clause/message/branch/barrier/dependency sections. |
| RDNA4 Chapter 5.3 instruction clauses | Audited statically | Checked generated `S_CLAUSE`, `S_DELAY_ALU`, `S_WAIT_ALU`, and runtime state. |
| RDNA4 Chapter 5.4 send message types | Audited statically | Checked message decode, operand classes, `M0`, return helpers, and KMCNT plumbing. |
| RDNA4 Chapter 5.5 branching | Audited statically | Checked generated branch decode, execute bodies, VM step behavior, and LLVM legality for selected operands. |
| RDNA4 Chapter 5.6 work-groups/barriers | Audited statically | Checked split-barrier decode, generated metadata, generic barrier rendezvous, and LLVM legality for selected operand forms. |
| RDNA4 Chapter 5.7 intro / 5.7.1 memory dependency counters | Audited statically | Checked split-wait decode, wait-counter storage, generic memory-pipeline counter release, producer classification, Flat routing, cache/message/export gaps, and DBT wait lowering. |
| RDNA4 Chapter 5.7.2 specific dependency cases | Audited statically | Checked `S_SETHALT`/`S_SETPRIO` to `S_GETREG` state-register delay cases and overlap with existing state-register/control-op gaps. |
| RDNA4 Chapter 5.8 ALU instruction software scheduling | Audited statically | Checked `S_DELAY_ALU` decode/runtime, `OPR_DELAY` display, DBT hazard-tracker emission, and current prologue-delay tests. |
| RDNA4 Chapter 6.1/6.2 SALU formats and operands | Audited statically | Checked SALU format/decode shape, scalar operand selectors, literal availability, out-of-range/alignment overlap, and `S_SETREG_IMM32_B32` literal decode, execution, and disassembly metadata. |
| RDNA4 Chapter 6.3-6.7 SALU SCC/arithmetic/conditional/compare/bit-wise behavior | Audited statically | Checked representative execute bodies, implicit SCC metadata, WREXEC destination restrictions, relative SGPR moves, bit-compare masks, and detailed shift-add/bitwise behavior. |
| RDNA4 Chapter 6.8-6.10 SALU floating point, state access, and memory aperture query | Audited statically | Checked scalar FP execution helpers, scalar F16 literal/destination metadata, `S_CVT_HI_F32_F16`, state-register instructions, and aperture selector resolution. |
| RDNA4 `V_CVT_{PK,SR}_{FP8,BF8}_F32` execution | Audited statically | No hardware run. |
| RDNA4 MODE/dispatch plumbing for `FP16_OVFL` | Audited statically | Command processor and wavefront state only. |
| Shared FP8/BF8 helpers | Audited statically for OCP overflow behavior | Checked the helper behavior used by the RDNA4 FP8/BF8 conversion rows; non-RDNA FNUZ behavior is outside this RDNA4 report. |
| FP8/BF8 conversion codegen paths | Audited statically for named non-scaled/scaled helper emission | Checked the RDNA4 generated conversion paths relevant to Chapter 7.6 and Chapter 16 data-conversion rows. |
| RDNA4 `S_GETREG_B32` / `S_SETREG_B32` MODE access | Audited statically | No hardware run. |
| RDNA4 Chapter 7.1 microcode encodings | Audited statically | Checked generated VOP3/VOP3P/VOP3SD fields, literal sizing, non-promotable instruction generation, generic modifier legality, disassembly, and scalar destination validation paths. |
| RDNA4 Chapter 7.2 operands | Audited statically | Checked generated VALU operand paths, non-standard readlane/writelane/carry/mask cases, source/modifier legality, output clamp behavior, round/denorm mode hooks, wave64 SGPR notes, OOR behavior, and PERMLANE adjacency state. |
| RDNA4 Chapter 7.3 VALU instruction inventory | Audited statically | Checked generated mnemonic coverage, VOP1/VOP2/VOP3/VOP3SD classification, compare-family expansion, baseline encodings, and unconditional execute stubs for inventory-listed instructions. |
| RDNA4 Chapter 7.4 16-bit Math and VGPRs | Audited statically | Checked true16 VGPR half-addressing in compact VOP1/VOP2/VOPC operands, OPSEL-based VOP3 helpers, VINTERP decode/runtime bodies, and existing true16 regression coverage. |
| RDNA4 Chapter 7.5/7.7 packed math | Audited statically | Checked generated packed I16/U16/F16, MIX, DOT execution, VOP3P modifiers, scalar/inline constants, DPP reachability, and SIMD fallback boundaries; WMMA/SWMMAC details are tracked under 7.12. |
| RDNA4 Chapter 7.6 data conversion overview | Audited statically | Checked generated conversion categories, FP8/BF8 special cases, true16 conversion helpers, and modifier handling against Chapter 7.6 common restrictions. |
| RDNA4 Chapter 7.8 VOPD dual issue | Audited statically | Checked generated VOPD decode, disassembly, operand collection, slot execution, source legality, wave-size assumptions, and paired-operation behavior. |
| RDNA4 Chapter 7.9 DPP cross-lane behavior | Audited statically | Checked generated DPP decode sizing, DPP8/DPP16 suffix handling, DPP_CTRL helper semantics, VOPC/CMPX masking, OPSEL legality, disassembly, tests, and DBT translation. |
| RDNA4 Chapter 7.10/7.11 pseudo-scalar transcendental and VGPR indexing | Audited statically | Checked generated pseudo-scalar execution and relative VGPR-indexing implementation. |
| RDNA4 Chapter 7.12 WMMA/SWMMAC layout, modifiers, and hazards | Audited statically | Checked dense/sparse generated bodies, shared matrix helpers, sparse index OPSEL handling, disassembly modifiers, hazard modeling, and LLVM/rocjitsu disassembly spot checks. The layout, legality, modifier, disassembly, and hazard mismatches are recorded below. |
| RDNA4 Chapter 8.1-8.5 scalar memory encoding, addressing, buffer resources, narrow loads, S_DCACHE_INV, dependency checking, SMEM groups, alignment, bounds, and prefetch | Audited statically | Checked generated SMEM decode/runtime, scalar memory address helpers, buffer-load treatment, narrow-load paths, destination handling, cache invalidation, `KMcnt` dependency handling, scalar-memory clause/group behavior, alignment/range-check behavior, and scalar prefetch handling. Detailed SMEM instruction-definition coverage is tracked in Chapter 16.6. |
| RDNA4 Chapter 9 intro / 9.1-9.6 vector memory buffer instructions, data, addressing, alignment, and resource descriptors | Audited statically | Checked generated VBUFFER decode/runtime, descriptor addressing helpers, TFE plumbing, format/D16/TBUFFER execute bodies, atomics, disassembly modifiers, and existing address-calculation tests. Detailed VBUFFER instruction-definition coverage is tracked in Chapter 16.16. |
| RDNA4 Chapter 10 intro / 10.1-10.10 vector memory image instructions, TFE/LWE, D16/A16/G16, no-sampler/sampler address tables, VGPR usage, image resources, samplers, data formats, data dependencies, ray tracing, and PRT | Audited statically | Checked generated VIMAGE/VSAMPLE field storage, operand exposure, execute stubs, disassembly modifiers, coverage exceptions, TFE/LWE/data operand modeling, opcode-family decode coverage, image-resource descriptor handling, sampler descriptor handling, buffer/image data-format modeling, VMEM dependency-counter plumbing, BVH ray-tracing VADDR/descriptor/return rules, and partially resident texture enable/status handling. LDS BVH stack operations referenced by Chapter 10.9 are covered by the Chapter 12 LDS stack slice. |
| RDNA4 Chapter 11 intro / 11.1-11.6 global, scratch, and flat address-space operations | Audited statically | Checked generated VFLAT/VGLOBAL/VSCRATCH field storage, operand exposure, disassembly modifiers, coverage exceptions, ADDTID execution, scratch-base plumbing, flat aperture routing, scratch/global/flat address helpers, special atomic execute bodies, vector-memory pipeline behavior, memory-error handling, data VGPR layout, atomic return-data shape, D16/D16_HI behavior, block VGPR load/store mask/address/OOR rules, and WMMA transpose-load EXEC/wave/layout rules. |
| RDNA4 Chapter 12 intro / 12.1/12.1.1/12.1.2 local data share overview, CU/WGP modes, and access methods | Audited statically | Checked LDS allocation/backing, DS address calculation, local-memory pipeline, WGP-mode placement, VDSDIR generated bodies, and existing LDS tests. Instruction-subsection details are tracked in separate rows. |
| RDNA4 Chapter 12.2/12.2.1 LDS parameter loads | Audited statically | Checked `DS_PARAM_LOAD` decode, operand exposure, M0 dependence, quad/lane semantics, 16-bit packing, `LDS_PARAM_RDY`, VDSDIR wait fields, EXPcnt plumbing, and local tests. DS_DIRECT was checked only where Chapter 12.2 shares hazards with parameter loads. |
| RDNA4 Chapter 12.3/12.3.1 VINTERP parameter interpolation | Audited statically | Checked VINTERP decode, operand widths, raw fields, execution stubs, built-in `WAIT_EXP`, fixed DPP/FMA semantics, OPSEL/destination preservation, clamp/RTZ behavior, and local fixture coverage. |
| RDNA4 Chapter 12.4 LDS direct load | Audited statically | Checked `DS_DIRECT_LOAD` decode, operand exposure, M0/datatype dependence, whole-quad broadcast, sign/zero extension, CU-only restriction, `LDS_PARAM_RDY`, EXPcnt/shared VDSDIR hazards, and local tests. |
| RDNA4 Chapter 12.5/12.5.1 indexed and atomic LDS | Audited statically | Checked VDS decode/build/disassembly, single and 2-address load/store address paths, ADDTID M0 semantics, DScnt wait plumbing, atomic return/no-return execution, STOREXCHG 2-address variants, FP and packed FP atomics, and local tests. Lane-permute and stack sections are audited in separate rows. |
| RDNA4 Chapter 12.5.2 LDS lane-permute ops | Audited statically | Checked generated wrappers and shared helpers for `DS_SWIZZLE_B32`, `DS_PERMUTE_B32`, `DS_BPERMUTE_B32`, and `DS_BPERMUTE_FI_B32`, including no-LDS-storage behavior, EXEC/source-read rules, byte-index wrapping, DS_SWIZZLE modes, semantic IR, read notifications, and local tests. DS stack/BVH is audited separately in the 12.5.3 row. |
| RDNA4 Chapter 12.5.3 DS stack operations for ray tracing | Audited statically | Checked generated wrappers, opcode/operand metadata, VDS raw field storage, execute bodies, memory-op flags, generator semantic classification, and RDNA3/RDNA3.5 stack-op boundary for `DS_BVH_STACK_PUSH4_POP1_RTN_B32`, `DS_BVH_STACK_PUSH8_POP1_RTN_B32`, and `DS_BVH_STACK_PUSH8_POP2_RTN_B64`. |
| RDNA4 Chapter 13 float memory atomics | Audited statically | Checked generated DS, buffer, flat, global, and image float atomic bodies against Chapter 13 rounding, denormal, NaN/minNum/maxNum, and float-add rules. DS-specific denormal and packed-lane gaps are already recorded in `RDNA4-RJ-170` and `RDNA4-RJ-171`; Chapter 13-wide and VMEM-specific gaps are recorded below. |
| RDNA4 Chapter 14 export | Audited statically | Checked generated `VEXPORT` decode, operand metadata, disassembly, execute body, status-register layout, and existing wait-counter gaps against Chapter 14 export field, packing, pixel/primitive, dual-source blend, status, and dependency rules. |
| RDNA4 Chapter 15 microcode formats | Audited statically | Checked generated scalar, VALU, VINTERP, VDSDIR, VDS, VBUFFER, VIMAGE, VSAMPLE, VFLAT/VGLOBAL/VSCRATCH, and VEXPORT raw field storage/build/disassembly hooks. Opcode-table semantics are tracked in the corresponding instruction-family slices. |
| RDNA4 Chapter 16.1 SOP2 definitions | Audited statically | Checked generated SOP2 mnemonic/decoder coverage, invalid opcode holes, scalar pack half-selection helpers, literal-only FMAAK/FMAMK literal placement, and overlap with existing SALU literal/SCC/FP/F16 gaps. |
| RDNA4 Chapter 16.2 SOPK definitions | Audited statically | Checked generated SOPK mnemonic/decoder coverage, invalid opcode holes, ordinary no-literal forms, `S_SETREG_IMM32_B32` implied literal handling, `S_VERSION`, `S_CALL_B64`, and overlap with existing HWREG/SCC/control-flow gaps. |
| RDNA4 Chapter 16.3 SOP1 definitions | Audited statically | Checked generated SOP1 mnemonic/decoder coverage, invalid opcode holes, markdown conversion artifact around `S_WQM_B32`, relative SGPR indexing, direct-PC and control instructions, and overlap with existing barrier/message/dynamic-VGPR/sleep/scalar-FP/F16 gaps. |
| RDNA4 Chapter 16.4 SOPC definitions | Audited statically | Checked generated SOPC mnemonic/decoder coverage, invalid opcode holes, literal handling, integer/bit/floating compare execution helpers, and overlap with existing SCC/scalar-FP/F16 gaps. |
| RDNA4 Chapter 16.5 SOPP definitions | Audited statically | Checked generated SOPP mnemonic/decoder coverage, invalid opcode holes, reserved/compatibility opcode drift, branch execute bodies, wait-counter plumbing, and overlap with existing trap/halt/sleep/clause/barrier/message/cache gaps. |
| RDNA4 Chapter 16.6 SMEM definitions | Audited statically | Checked generated SMEM mnemonic/decoder coverage, load width/sign-extension setup, scalar-buffer and scalar-prefetch constructors, `S_DCACHE_INV`, overlap with existing Chapter 8 SMEM gaps, and XML-only ATC probe drift. |
| RDNA4 Chapter 16.7 VOP2 definitions | Audited statically | Checked generated VOP2 mnemonic/decoder coverage, implied-literal forms, DPP/literal sizing, compact VCC dataflow, packed-FMAC classification, and overlap with existing Chapter 7 VALU gaps. |
| RDNA4 Chapter 16.8 VOP1 definitions | Audited statically | Checked generated VOP1 mnemonic/decoder coverage, VOP3 aliases, true16/FP8 selector plumbing, readfirstlane, pipeflush, swap/permlane, relative-indexed forms, and overlap with existing Chapter 7 VALU gaps. |
| RDNA4 Chapter 16.9 VOPC definitions | Audited statically | Checked generated VOPC mnemonic/decoder coverage, compact and VOP3 compare-result operands, compare-family expansion, class-compare masks, true16/DPP/literal variants, and overlap with existing Chapter 7 VALU gaps. |
| RDNA4 Chapter 16.10 VOP3P definitions | Audited statically | Checked generated VOP3P mnemonic/decoder coverage, opcode holes, DOT/MIX/packed-F16/WMMA/SWMMAC rows, DPP alternatives, VOP3P field metadata, and overlap with existing Chapter 7 packed-math and WMMA gaps. |
| RDNA4 Chapter 16.11 VOPD definitions | Audited statically | Checked generated VOPD slot opcode inventory, OPX/OPY decode shape, Y-only opcode handling, literal sizing, ordinary slot execution, disassembly, operand metadata, and overlap with existing VOPD/VALU gaps. |
| RDNA4 Chapter 16.12 VOP3/VOP3SD definitions | Audited statically | Checked generated VOP3/VOP3SD mnemonic/decoder coverage, opcode inventory, VOP3SD restriction, compare result operands, DPP/literal handling, true16 literal selection, generic modifiers, and overlap with existing Chapter 7 VALU gaps. |
| RDNA4 Chapter 16.13 VINTERP definitions | Audited statically | Checked generated VINTERP mnemonic/decoder coverage, opcode inventory, raw field preservation, operand widths, no-op execution stubs, `WAIT_EXP`, F16 destination-half preservation metadata, and overlap with existing Chapter 12.3 VINTERP gaps. |
| RDNA4 Chapter 16.14 parameter/direct LDS load definitions | Audited statically | Checked generated VDSDIR mnemonic/decoder coverage, opcode inventory, raw field preservation, operand exposure, no-op execution stubs, aliases/fixtures, and overlap with existing Chapter 12 VDSDIR gaps. |
| RDNA4 Chapter 16.15 VDS definitions | Audited statically | Checked generated VDS mnemonic/decoder/class coverage, opcode inventory, raw field preservation, load/store/atomic execution paths, D16/B96/B128 data movement, special atomic lowering, lane-permute/swizzle/BVH rows, and overlap with existing Chapter 12/13 DS gaps. |
| RDNA4 Chapter 16.16 VBUFFER definitions | Audited statically | Checked generated VBUFFER mnemonic/decoder/class coverage, opcode inventory, raw field preservation, formatted/typed/TBUFFER stubs, plain load/store paths, integer/FP/packed atomic lowering, and overlap with existing Chapter 9/13 buffer gaps. |
| RDNA4 Chapter 16.17 VIMAGE definitions | Audited statically | Checked generated VIMAGE mnemonic/decoder/class coverage, opcode inventory, raw field preservation, load/store/atomic/BVH stubs, GET_RESINFO, packed/FP atomic rows, and overlap with existing Chapter 10/13 image gaps. |
| RDNA4 Chapter 16.18 VSAMPLE definitions | Audited statically | Checked generated VSAMPLE mnemonic/decoder/class coverage, opcode inventory, raw field preservation, MSAA/sample/gather/GET_LOD stubs, G16 rows, sampler operands, and overlap with existing Chapter 10 sample/gather gaps. |
| RDNA4 Chapter 16.19 VEXPORT definitions | Audited statically | Checked generated VEXPORT singleton mnemonic/decoder/class coverage, raw field preservation, no-op export body, target/source operands, and overlap with existing Chapter 14 export gaps. |
| RDNA4 Chapter 16.20 FLAT/Scratch/Global definitions | Audited statically | Checked generated VFLAT/VSCRATCH/VGLOBAL mnemonic/decoder/class coverage, opcode inventory, raw field preservation, address helpers, D16 forms, block loads/stores, transpose loads, integer/FP/packed atomics, special atomic lowering, and overlap with existing Chapter 11/13 gaps. |
| RDNA4 decoder/autogen/disassembly surface through Chapter 16 | Complete | Coverage rows above cover the RDNA4 manual chapters and instruction-definition families; detailed entries below record the remaining decode, execution, metadata, disassembly, and DBT gaps found in this pass. |

## Gaps

### RDNA4-RJ-001: FP8/BF8 converts cannot model `FP16_OVFL=1`

Manual evidence:

- `rdna4/README.md:3166` through `:3174` defines separate F32 to FP8/BF8
  results for `FP16_OVFL=0` and `FP16_OVFL=1`.

Rocjitsu evidence:

- `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vop3.cpp:22780` and
  `:22781` call `util::f32_to_fp8_e4m3_rne` for `V_CVT_PK_FP8_F32`.
- `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vop3.cpp:22869` and
  `:22870` call `util::f32_to_bf8_e5m2_rne` for `V_CVT_PK_BF8_F32`.
- `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vop3.cpp:22960` calls
  `util::f32_to_fp8_e4m3_sr` for `V_CVT_SR_FP8_F32`.
- `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vop3.cpp:23053` calls
  `util::f32_to_bf8_e5m2_sr` for `V_CVT_SR_BF8_F32`.
- None of these calls passes MODE state or an `FP16_OVFL` boolean.
- The OCP helpers hard-code the `FP16_OVFL=0` outcomes: FP8 Inf/overflow maps
  to signed NaN at `lib/util/include/util/data_types.h:446` through `:447`,
  `:465` through `:478`, `:489` through `:490`, and `:507` through `:520`;
  BF8 Inf/overflow maps to signed Inf at `:768` through `:769`, `:786` through
  `:797`, `:808` through `:809`, and `:826` through `:837`.

Impact:

RDNA4 emulation currently cannot produce `max_E4M3` / `max_E5M2` for the
manual's `FP16_OVFL=1` mode on these instructions.

### RDNA4-RJ-002: Dispatch does not initialize `Wavefront::mode_raw()` from kernel descriptor float mode bits

Manual and ABI evidence:

- RDNA4 manual defines `MODE.FP16_OVFL` at `rdna4/README.md:1074`.
- Rocjitsu's imported HSA descriptor header defines
  `COMPUTE_PGM_RSRC1_FP16_OVFL` at
  `lib/rocjitsu/external_headers/hsa_headers/hsa/AMDHSAKernelDescriptor.h:118`.

Rocjitsu evidence:

- `Wavefront` stores raw MODE state and exposes `mode_raw()` /
  `set_mode_raw()` at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:103`
  through `:111`.
- Reset initializes MODE to zero at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:480`.
- `CommandProcessor::process_aql_packet` reads VGPR, SGPR, user-SGPR, and WGP
  fields from `compute_pgm_rsrc1` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:929` through
  `:977`, but does not read `COMPUTE_PGM_RSRC1_FP16_OVFL`,
  `FLOAT_ROUND_MODE_*`, or `FLOAT_DENORM_MODE_*`.
- `DispatchEntry` stores `wgp_mode` but has no `compute_pgm_rsrc1` or
  FP16-overflow field at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/dispatch_entry.h:43` through `:84`.
- `ComputeUnitCore::dispatch_wf` sets wavefront identity, PC, register
  allocation, `exec`, `vcc`, `m0`, and apertures at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:136` through `:148`,
  but does not initialize `mode_raw()` from dispatch state.

Impact:

Even after helper-level support exists, production dispatch needs a path from
the kernel descriptor or shader-written MODE state to the conversion helpers.

### RDNA4-RJ-003: Existing FP8/BF8 tests assert only one overflow mode

Rocjitsu evidence:

- `tests/data_types_test.cpp:274` through `:331` asserts FP8 OCP RNE/SR
  overflow maps to signed NaN.
- `tests/data_types_test.cpp:493` through `:504` asserts BF8 OCP Inf maps to
  Inf.
- `tests/hip_cvt_narrow_test.cpp:216` through `:309` and `:314` through `:448`
  use finite values (`1.0f`, `2.0f`) for FP8/BF8 packed and stochastic HIP
  checks.

Impact:

The tests cover the current `FP16_OVFL=0` helper behavior and finite-value
plumbing, but do not seed or validate `FP16_OVFL=1` dispatch/execution behavior.

### RDNA4-RJ-004: Codegen emits unconditional FP8/BF8 helpers and a generated helper template appears stale

Rocjitsu evidence:

- `lib/python/amdisa/semantics.py:1335` through `:1344` maps
  `V_CVT_PK_{FP8,BF8}_{F32,F16}` to the `vector_cvt_pk` semantic class.
- `lib/python/amdisa/codegen/execute/vector_special.py:1018` through `:1039`
  emits direct calls to `util::f32_to_fp8_e4m3_rne` or
  `util::f32_to_bf8_e5m2_rne` for packed FP8/BF8 conversions, without a
  MODE/`FP16_OVFL` parameter.
- `lib/python/amdisa/semantics.py:1416` through `:1422` maps
  `V_CVT_SR_FP8_F32` and `V_CVT_SR_BF8_F32` to the `cvt_fp8` semantic class.
- Scaled conversion code uses `_scale_encode_call`, which emits the same
  unconditional FP8/BF8 helper names at
  `lib/python/amdisa/codegen/execute/vector_special.py:1288` through `:1303`.
- `lib/python/amdisa/codegen/_generator.py:7943` through `:8001` contains the
  generated text for `f32_to_fp8_e4m3_{rne,sr}`.
- That template returns `max normal` (`sign | 0x7E`) for FP32 Inf and many
  overflow cases at `:7949`, `:7963` through `:7967`, `:7976` through `:7978`,
  `:7990`, `:7993`, and `:7999`.
- The checked-in runtime helper in `lib/util/include/util/data_types.h:439`
  through `:521` now returns signed NaN for OCP FP8 overflow in the
  `FP16_OVFL=0` mode.

Impact:

Regenerated instruction execution paths will continue to hard-code one
`FP16_OVFL` mode unless the semantic/codegen API grows an explicit mode input.
If the `_generator.py` helper text is still used for emitted utility code,
regeneration can also reintroduce the old FP8 overflow behavior. The stale
helper-template risk needs confirmation in the codegen path before
classification as a live runtime bug.

### RDNA4-RJ-005: RDNA4 `S_GETREG_B32` / `S_SETREG_B32` do not expose MODE

Manual/XML evidence:

- The RDNA4 manual says MODE fields can be read and written by shader scalar
  instructions at `rdna4/README.md:1064` through `:1067`.
- The manual lists `S_GETREG_B32` and `S_SETREG_B32` as hardware-register
  access instructions at `rdna4/README.md:2558` through `:2565`.
- The checked-in RDNA4 XML defines `OPR_HWREG` ID 1 as `hw_reg_wave_mode` and
  ID 2 as `hw_reg_wave_status` at `amdgpu_isa_rdna4.xml:181767` through
  `:181790`.

Rocjitsu evidence:

- RDNA4 `SGetregB32Sopk::execute_impl` maps `reg_id == 1` to
  `wf.status_raw()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/sopk.cpp:116` through
  `:125`, and has no MODE case.
- RDNA4 `SSetregB32Sopk::execute_impl` maps `reg_id == 1` to
  `wf.set_status_raw(...)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/sopk.cpp:165` through
  `:180`.
- RDNA4 `SSetregImm32B32Sopk::execute_impl` also maps `reg_id == 1` to status
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/sopk.cpp:202` through
  `:217`.
- GFX1250 has explicit `HW_REG_MODE = 1` and `HW_REG_STATUS = 2` support at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/gfx1250/sopk.cpp:23` through
  `:83`, which matches the RDNA4 XML numbering.

Impact:

RDNA4 shader code cannot use rocjitsu's SOPK path to read or update
`MODE.FP16_OVFL`; even if the conversion helpers became mode-sensitive, shader
mode changes would not affect them through this path. This is a static finding;
hardware behavior was not probed.

### RDNA4-RJ-006: RDNA4 SCC/PRIV are modeled in `STATUS`, but the manual puts SCC in `STATE_PRIV`

Manual evidence:

- `rdna4/README.md:1010` through `:1042` defines the RDNA4 `STATUS` bitfield.
- `rdna4/README.md:1042` through `:1053` defines `STATE_PRIV`, including
  `SCC` at bit 9.

Rocjitsu evidence:

- RDNA4 `StatusReg` in `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/isa.h:27`
  through `:47` defines `SCC` at bit 0, `PRIV` at bit 5, `EXECZ` at bit 9, and
  `VCCZ` at bit 10.
- `Wavefront::read_scc()` and `write_scc()` read/write bit 0 of
  `status_raw()` at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:417`
  through `:425`.
- There is no separate raw `STATE_PRIV` register in `Wavefront`; only raw
  `STATUS`, raw `MODE`, and raw `WAVE_SCHED_MODE` are stored at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:97` through `:128` and
  `:535` through `:537`.

Impact:

Scalar instruction control flow can still use rocjitsu's internal SCC helper,
but debugger-visible `S_GETREG_B32 state_priv[9]`, `S_SETREG_B32 state_priv`,
and privilege-gated behavior cannot match RDNA4 Chapter 3 with the current
state split.

### RDNA4-RJ-007: `S_ALLOC_VGPR` is decoded but executes as a no-op

Manual evidence:

- `rdna4/README.md:860` through `:899` defines dynamic VGPR launch mode,
  block-size limits, all-or-nothing reallocation, and SCC success/failure.
- `rdna4/README.md:10384` through `:10400` gives `S_ALLOC_VGPR` pseudocode:
  call `ReallocVgprs`, update `NUM_VGPRS` on success, and write SCC.

Rocjitsu evidence:

- RDNA4 `SAllocVgprSop1::execute_impl` dispatches to
  `amdgpu::execute_s_alloc_vgpr_sop1` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/sop1.cpp:1329` through
  `:1330`.
- The shared generated helper is empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:469`
  through `:472`.
- `Wavefront` stores fixed per-dispatch `num_vgprs_` / `vgpr_alloc_`, and
  `ComputeUnitCore::dispatch_wf` initializes those once at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:138` through `:142`.

Impact:

Any shader that relies on dynamic VGPR allocation observes no allocation change
and no SCC success/failure result in rocjitsu.

### RDNA4-RJ-008: Compute launch has incomplete RDNA4 `TTMP7`/`TTMP8` dispatch payloads

Manual evidence:

- `rdna4/README.md:1453` through `:1466` says compute waves initialize
  `TTMP7`, `TTMP8`, and `TTMP9`; `TTMP8` contains
  `{DebugMark, GridYZvalid, waveIDinGroup[4:0], dispatchIndex[24:0]}` and is
  initialized for all CS waves, while `TTMP7` is loaded only when
  `GridYZvalid==1`.
- `rdna4/README.md:1476` through `:1486` defines `GridYZvalid`,
  wave-id-in-workgroup, and dispatch-index details.

Rocjitsu evidence:

- The RDNA4/gfx1250 launch path in
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:288` through
  `:297` always writes `TTMP7` and `TTMP9`, but has no `TTMP8` write and no
  shader-usage-derived `GridYZvalid` gate for `TTMP7`.
- `DispatchEntry` stores a rocjitsu dispatch ID, but no AQL queue
  dispatch-index payload field for `TTMP8`, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/dispatch_entry.h:43` through `:84`.

Impact:

RDNA4 compute code or trap/debug code that reads `TTMP8` gets zero/stale data
instead of the architected dispatch metadata, and code that treats `TTMP7`
validity as controlled by `GridYZvalid` can observe Y/Z grid data in cases
where the manual leaves it uninitialized.

### RDNA4-RJ-009: Operand reads/writes do not apply per-wave SGPR/VGPR out-of-range rules

Manual evidence:

- `rdna4/README.md:778` through `:816` defines SGPR OOR behavior: sources
  return zero, writes are ignored, TTMP writes outside trap mode are ignored,
  and failed TTMP writes do not update SCC.
- `rdna4/README.md:826` through `:858` defines VGPR OOR behavior: dest OOR
  nullifies the instruction, source OOR uses VGPR0, and multiple-destination
  instructions suppress all writes if any destination is OOR.

Rocjitsu evidence:

- RDNA4 scalar source resolution directly reads physical
  `wf.sgpr_alloc().base + ev` for SGPR/TTMP encodings at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/operand.cpp:749` through
  `:761`, without checking `wf.num_sgprs()` or trap privilege.
- RDNA4 scalar writes directly write physical `wf.sgpr_alloc().base + ev` for
  SGPR/TTMP encodings at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/operand.cpp:927` through
  `:950`, without TTMP privilege or out-of-range suppression.
- RDNA4 VGPR reads/writes compute `wf.vgpr_alloc().base + voff` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/operand.cpp:1095` through
  `:1125`, without comparing the logical operand range against
  `wf.num_vgprs()`.
- The physical CU accessors index the backing register files directly at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.h:379` through `:392` and
  `:702` through `:719`.

Impact:

Small-register-allocation shaders and dynamic-VGPR shaders can read/write
physical registers that should architecturally behave as zero, ignored writes,
VGPR0 substitution, or full instruction nullification.

### RDNA4-RJ-010: LDS allocation granularity and OOR access rules are partial

Manual evidence:

- `rdna4/README.md:929` through `:935` says LDS is allocated in 1024-byte
  blocks and allocations do not wrap across the internal CU-side boundary.
- `rdna4/README.md:945` through `:958` says LDS loads return zero for OOR
  dwords, atomics report `MEMVIOL` on alignment/OOR failures, and any portion
  of a store operation that is in range is written while stored bytes out of
  range are discarded.

Rocjitsu evidence:

- `ComputeUnitCore::allocate_lds` aligns group LDS allocations to 256 bytes at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.h:239` through `:245`.
- The WGP allocator also aligns LDS requests to 256 bytes at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/spi.h:203` through `:206`.
- `Lds::vector_store` drops an element whenever `ea + elem_size > data_.size()`
  at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/lds.h:151` through `:164`.
- `LocalMemPipeline::issue_access` uses `Lds::vector_store` for LDS stores at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:596` through `:601`.

Impact:

An LDS store whose first byte(s) are in range and last byte(s) cross the LDS
boundary should leave the in-range bytes visible, but rocjitsu currently drops
that whole per-lane element. LDS allocation placement can also differ from the
manual's 1024-byte granularity.

### RDNA4-RJ-011: RDNA4 HWREG access is much narrower than the Chapter 3 register set

Manual/XML evidence:

- The RDNA4 manual HWREG table at `rdna4/README.md:983` through `:1008`
  includes MODE, STATUS, STATE_PRIV, perf snapshot registers, exception flags,
  TRAP_CTRL, SCRATCH_BASE, HW_ID1/2, `IB_STS2`, and shader cycles.
- `OPR_HWREG` in XML enumerates most of those IDs at
  `amdgpu_isa_rdna4.xml:181783` through `:181877`.

Rocjitsu evidence:

- RDNA4 `SGetregB32Sopk::execute_impl` handles only IDs 1, 4, 5, 6, and 7 at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/sopk.cpp:116` through
  `:137`.
- RDNA4 `SSetregB32Sopk::execute_impl` and `SSetregImm32B32Sopk::execute_impl`
  handle only ID 1 at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/sopk.cpp:165`
  through `:217`.
- Those IDs do not match RDNA4 XML/manual for several registers: ID 1 is MODE,
  not STATUS; IDs 23 and 24 are HW_ID1/HW_ID2, not IDs 4 and 5; ID 6 is
  LDS_ALLOC in XML, not VGPR allocation.

Impact:

Beyond the `MODE.FP16_OVFL` issue, most RDNA4 HWREG reads/writes either warn
and return zero or read/write the wrong underlying emulator state.

### RDNA4-RJ-012: Scalar selectors 102 and 103 shadow real `s102` / `s103`

Manual/XML evidence:

- The RDNA4 manual says scalar encodings `0-105` are SGPRs at
  `rdna4/README.md:778` through `:782`.
- The generated RDNA4 operand table also classifies `OPR_SRC` SGPRs as
  `0-105` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/operand_types.h:152`
  through `:155`.
- The checked-in XML names value 102 as `s102`, for example at
  `amdgpu_isa_rdna4.xml:182459` through `:182461`.

Rocjitsu evidence:

- RDNA4 scalar source resolution returns `wf.scratch_base()` for encoding 102
  and `wf.scratch_base() >> 32` for encoding 103 before the normal `ev <= 105`
  SGPR case at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/operand.cpp:749` through
  `:755`.
- RDNA4 scalar destination resolution writes scratch-base halves for encodings
  102 and 103 before the normal SGPR case at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/operand.cpp:916` through
  `:928`.

Impact:

Instructions that name `s102` or `s103` as ordinary scalar operands read/write
`SCRATCH_BASE` instead of the allocated SGPRs.

### RDNA4-RJ-013: `S_ROUND_MODE` and `S_DENORM_MODE` do not update MODE

Manual evidence:

- `rdna4/README.md:1064` through `:1073` defines `MODE.FP_ROUND` and
  `MODE.FP_DENORM`.
- `rdna4/README.md:1896` through `:1897` lists `S_ROUND_MODE` and
  `S_DENORM_MODE` as setting those modes from immediate bits.
- `rdna4/README.md:2894` through `:2896` says round and denormal modes can be
  set through those instructions as well as `S_SETREG`.

Rocjitsu evidence:

- RDNA4 `S_ROUND_MODE` dispatches to `execute_s_round_mode_sopp` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/sopp.cpp:157` through
  `:167`; that shared helper is empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2422`
  through `:2425`.
- RDNA4 `S_DENORM_MODE` dispatches to `execute_s_denorm_mode_sopp` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/sopp.cpp:170` through
  `:180`; that shared helper is empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1643`
  through `:1645`.

Impact:

Shaders cannot change rocjitsu's MODE rounding or denormal fields through the
dedicated RDNA4 scalar instructions, and downstream float semantics cannot
observe those state changes.

### RDNA4-RJ-014: Trap, exception, and trap-message state is largely absent

Manual evidence:

- `rdna4/README.md:1253` through `:1327` defines TTMP privilege, trap entry,
  TBA/TMA access, trap enable/control behavior, and exception flag registers.
- `rdna4/README.md:1841` through `:1854` defines `S_TRAP` / `S_RFE_B64` trap
  behavior.
- `rdna4/README.md:1887` through `:1888` defines `S_SENDMSG` and
  `S_SENDMSG_RTN` behavior, and `rdna4/README.md:1985` through `:1988`
  includes TMA/TBA return-message IDs.

Rocjitsu evidence:

- RDNA4 `S_TRAP` throws `UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/sopp.cpp:152` through
  `:154`.
- The shared `S_SENDMSG` helper is a no-op at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2427`
  through `:2428`.
- `S_SENDMSG_RTN_B32` only returns nonzero data for message `0x83`; TMA/TBA
  and other message IDs fall through to zero at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2431`
  through `:2457`.
- TTMP writes are not privilege-gated in the scalar operand resolver, as noted
  in RDNA4-RJ-009.

Impact:

Trap handlers, context-save flows, TBA/TMA queries, exception flags, and
privilege-restricted trap state cannot be modeled correctly for RDNA4.

### RDNA4-RJ-015: Unlisted launch state is zero-filled instead of uninitialized

Manual evidence:

- `rdna4/README.md:1353` through `:1359` says shader state not explicitly
  listed as initialized in the launch-state section is not initialized,
  including unlisted LDS, VGPRs, and SGPRs.

Rocjitsu evidence:

- `ComputeUnitCore::dispatch_wf` zeroes the full SGPR block and the whole VGPR
  allocation block before launch at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:125` through `:129`.
- `ComputeUnitCore::allocate_lds` zeroes each allocated LDS range at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.h:239` through `:245`.

Impact:

Rocjitsu deterministically supplies zero for uninitialized launch state. That
may be useful for reproducibility, but it can hide shader bugs or differ from
hardware when code consumes unspecified SGPR/VGPR/LDS contents.

### RDNA4-RJ-016: RDNA4 Wave64 dispatch is not selectable

Manual evidence:

- `rdna4/README.md:567` through `:573` says RDNA4 supports both wave32 and
  wave64, and shader programs are compiled for a particular wave size.
- `rdna4/README.md:713` says wave64 uses all 64 bits of `EXEC`, while wave32
  uses only low 32 bits.

Rocjitsu evidence:

- `RdnaIsaBase` sets `WF_SIZE = 32` and `WF_SIZE_MAX = 64` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/rdna_isa_base.h:50`
  through `:53`.
- `IsaExecComputeUnit` constructs the execution CU with `Isa::WF_SIZE` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.h:683` through `:685`.
- `CommandProcessor::process_aql_packet` computes waves per workgroup from
  `cus_[0]->wf_size()` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:941` through
  `:944`; the AMDHSA `ENABLE_WAVEFRONT_SIZE32` bit is defined in the imported
  descriptor header at
  `lib/rocjitsu/external_headers/hsa_headers/hsa/AMDHSAKernelDescriptor.h:190`,
  but is not consumed there.

Impact:

RDNA4 wave64 kernels are dispatched and executed as wave32 in rocjitsu, so
wave64-specific EXEC/VCC behavior, issue passes, VGPR allocation granularity,
and instruction restrictions are not modeled.

### RDNA4-RJ-017: Indirect PC writes do not force DWORD alignment

Manual evidence:

- `rdna4/README.md:699` through `:705` says the PC is DWORD-aligned and the
  low two bits are forced to zero.

Rocjitsu evidence:

- `S_SETPC_B64` assigns `wf.pc` from the scalar source minus instruction size
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/sop1.cpp:1201` through
  `:1202`.
- `S_SWAPPC_B64` does the same at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/sop1.cpp:1221` through
  `:1223`.
- The instruction fetch loop reads from `active->pc + i * 4` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:340` through `:345`,
  and the main loop advances the raw stored PC at `:432` through `:439`.

Impact:

An indirect branch or return to an unaligned address can make rocjitsu fetch
from an unaligned PC instead of applying the architectural low-bit masking.
Practical impact needs a targeted repro because ordinary code-object entry
points and assembler-generated branch targets are normally aligned.

### RDNA4-RJ-018: GFX12 `SCOPE` mapping reuses the wrong scope order

Manual evidence:

- `rdna4/README.md:1715` through `:1720` defines RDNA4 `SCOPE` values as
  `0=CU`, `1=SE`, `2=DEV`, and `3=SYS`.

Rocjitsu evidence:

- `shared/gfx12_cache_flags.h:12` through `:24` documents the same RDNA4 order
  and says `SCOPE=1,TH=0` maps to `RW`, `SCOPE=2,TH=0` maps to `CC`, and
  `SCOPE=3,TH=0` maps to `UC`.
- `mtype_from_flags_gfx12` calls `mtype_from_scope_nt(scope_val, ...)` at
  `shared/gfx12_cache_flags.h:45` through `:47`.
- That shared helper is documented for `0=WAVE/CU`, `1=DEVICE`, `2+=SYSTEM` at
  `shared/gfx940_cache_flags.h:29` through `:43`, so it maps RDNA4
  `SCOPE=1` to `CC` and `SCOPE>=2` to `UC`.

Impact:

RDNA4 SE-scope accesses are treated like device-coherent accesses, and
device-scope accesses are treated like system/uncached accesses.

### RDNA4-RJ-019: GFX12 temporal hints are collapsed, and NT does not bypass vector L1

Manual evidence:

- `rdna4/README.md:1758` through `:1782` defines distinct load/store policies
  for `RT`, `NT`, `HT`, `LU`, `NT_RT`, `RT_NT`, `NT_HT`, store `WB`, and
  `NT_WB`.

Rocjitsu evidence:

- `shared/gfx12_cache_flags.h:36` through `:47` only names
  `GFX12_TH_NT = 1`, then derives non-temporal status from `th == 1`.
- Generated RDNA4 vector memory instructions set `d->mtype` from
  `mtype_from_flags_gfx12(inst_.scope, inst_.th)`, but set
  `d->non_temporal = 0`, for example `global_load_u8` at
  `rdna4/vglobal.cpp:40` through `:49`.
- The global memory pipeline only passes `d.mtype` and `d.non_temporal` to the
  vector L1 at `vm/amdgpu/memory_pipeline.cpp:488` through `:494`.
- `Mtype::NT` is documented as `Bypass L1` at `vm/amdgpu/mtype.h:24` through
  `:30`, but `L1VectorCache` only bypasses on `Mtype::UC`, the separate
  `non_temporal` flag, or forced bypass at `vm/amdgpu/l1_vector_cache.cpp:92`
  through `:101`; stores have the same `Mtype::UC || non_temporal` check at
  `:144` through `:152`.

Impact:

All TH values other than exactly `1` lose their near/far cache-policy meaning,
including last-use, high-priority, mixed near/far NT, and store writeback
variants. Even the exact `TH=1` NT case can still allocate/use vector L1
because `Mtype::NT` is not treated as an L1 bypass in the vector cache path.

### RDNA4-RJ-020: Atomic return selection ignores TH bit semantics

Manual evidence:

- `rdna4/README.md:1784` through `:1795` says atomic TH is split into
  independent bits: `TH[0]` selects returning versus non-returning atomics,
  `TH[1]` selects temporal versus non-temporal, and `TH[2]` controls cascading
  deferred scope for non-returning atomics.

Rocjitsu evidence:

- `gfx12_atomic_returns(th)` returns true only when `th == 1` at
  `shared/gfx12_cache_flags.h:49` through `:52`.
- Generated RDNA4 atomics use that helper for the load/return decision, for
  example `is_load = amdgpu::gfx12_atomic_returns(inst_.th)` in generated
  VBUFFER/VGLOBAL/VFLAT atomic paths.

Impact:

Returning atomics with `TH[0]=1` and other bits set, such as `TH=3`, `5`, or
`7`, are treated as non-returning. The NT and cascading-scope bits are also not
modeled independently.

### RDNA4-RJ-021: SMEM scope/TH is decoded but not used by the scalar memory pipeline

Manual evidence:

- `rdna4/README.md:1805` through `:1811` defines SMEM `SCOPE`/`TH` semantics,
  including the scalar-only CU coherence caveat and non-coherence with VMEM
  stores/atomics at `scope==CU`.

Rocjitsu evidence:

- Generated SMEM loads set `d->mtype = mtype_from_flags_gfx12(inst_.scope,
  inst_.th)`, for example `s_load_b32` at
  `rdna4/smem.cpp:44` through `:53`.
- `ScalarMemPipeline::initiate_access` calls the scalar L1 `load`,
  `load_bytes`, and `store` APIs without passing `d.mtype` at
  `vm/amdgpu/memory_pipeline.cpp:203` through `:215`.
- `S_DCACHE_INV` has no operands and unconditionally invalidates the whole
  scalar L1 at `rdna4/smem.cpp:564` through `:571`.

Impact:

SMEM scope and temporal hints do not affect scalar memory execution today, and
cache invalidation is all-or-nothing rather than following the scoped
acquire/invalidate policy.

### RDNA4-RJ-022: Signed 64-bit literal operands are zero-extended

Manual evidence:

- `rdna4/README.md:1654` through `:1661` says a 32-bit literal used by a
  64-bit signed integer operation is sign-extended to 64 bits.

Rocjitsu evidence:

- `read_immediate64` zero-extends every `OPR_SIMM32` value at
  `rdna4/operand.cpp:1008` through `:1011`.
- `Operand::read_lane64` and `Operand::read_scalar64` both call
  `read_immediate64` for immediate operands at `rdna4/operand.cpp:1132`
  through `:1145` and `:1158` through `:1163`.
- `V_MAD_CO_I64_I32` is a signed integer instruction whose XML describes
  `SRC2` as `FMT_NUM_I64` and whose literal encoding permits `SIMM32` in that
  source position at `amdgpu_isa_rdna4.xml:111175` through `:111214` and
  `:111329` through `:111363`.
- The generated constructor builds that literal as `Operand(64,
  OperandType::OPR_SIMM32, ...)` at `rdna4/vop3.cpp:38028` through `:38031`,
  and execution reads it with `read_lane64(inst.src2, lane)` at
  `shared/execute_shared.h:13051` through `:13055`.

Impact:

For a literal like `0xffffffff` in a signed 64-bit source, rocjitsu supplies
`0x00000000ffffffff` instead of the architectural
`0xffffffffffffffff`.

### RDNA4-RJ-023: Cache-maintenance operations ignore scope and wait-counter behavior

Manual evidence:

- `rdna4/README.md:1733` through `:1752` defines acquire/release WB/INV rules
  based on `ISA.SCOPE > CACHE_SCOPE`.
- `rdna4/README.md:1796` through `:1803` defines which CU/L2 cache levels are
  affected for each VMEM scope.
- `rdna4/README.md:2181` and `:2186` include `global_inv` in `LOADcnt` and
  `global_wb/wbinv` in `STOREcnt`; `rdna4/README.md:2213` through `:2215` says
  scalar cache invalidates are tracked by `KMcnt` and require a wait for
  completion.
- `rdna4/README.md:3964` and `:3970` repeat the Chapter 8.2 scalar-memory
  rule: cache invalidates increment `KMcnt` and are not known complete until
  the shader waits for `KMcnt == 0`.

Rocjitsu evidence:

- `global_inv` invalidates all vector L1 state and flushes all L2 state at
  `rdna4/vglobal.cpp:807` through `:818`.
- `global_wb` is a no-op at `rdna4/vglobal.cpp:820` through `:827`.
- `global_wbinv` flushes only vector L1 at `rdna4/vglobal.cpp:1941` through
  `:1948`.
- `s_dcache_inv` directly invalidates the scalar L1 with no memory state or
  wait-counter participation at `rdna4/smem.cpp:564` through `:571`.
- `S_DCACHE_INV` is classified as `dcache_inv` at
  `lib/python/amdisa/semantics.py:1699` through `:1701`; the generator emits a
  direct `wf.cu().l1_scalar().invalidate_all()` body at
  `lib/python/amdisa/codegen/_generator.py:4047` through `:4048`.
- `dcache_inv` is not in the constructor-side `_MEM_CLASSES` set that adds
  `flags_ |= MEMORY_OP` at `lib/python/amdisa/codegen/_generator.py:6210`
  through `:6240` and `:6471` through `:6472`.
- These helpers execute directly rather than routing through
  `MemoryPipeline::issue`, so they do not increment or release the documented
  `LOADcnt`, `STOREcnt`, or `KMcnt` producer counters.

Impact:

Acquire/release ordering can be too weak or too strong, and waits cannot
observe these cache-maintenance operations with the manual's counters.

### RDNA4-RJ-024: Reserved common source selectors still execute with legacy meanings

Manual evidence:

- `rdna4/README.md:1676` through `:1701` marks source selector values
  `209-232`, `239`, `249`, `251`, `252`, and `254` as reserved, while `233`,
  `234`, and `250` are DPP selectors only valid as `SRC0`.

Rocjitsu evidence:

- The RDNA4 source selector enum omits the manual-reserved values at
  `rdna4/operand_types.h:152` through `:184`.
- Scalar source resolution still assigns meanings to values `230`, `231`,
  `249`, `250`, `251`, and `252` at `rdna4/operand.cpp:774` through `:815`.
- `can_resolve_src_scalar` reports the same values as resolvable at
  `rdna4/operand.cpp:846` through `:854`.

Impact:

Manual-invalid RDNA4 encodings can execute with scratch, null, VCCZ, EXECZ, or
other legacy-style values instead of being rejected or trapped as invalid.

### RDNA4-RJ-025: Wait-counter field widths, distinct counters, and DBT translation are partial

Manual evidence:

- `rdna4/README.md:11527` through `:11537` gives separate wait field widths:
  6-bit load/store/sample/DS waits, 3-bit BVH/export waits, and 5-bit KM waits.
- `rdna4/README.md:11533` through `:11537` specifically defines
  `S_WAIT_KMCNT` as waiting until `KMcnt <= SIMM16[4:0]`.
- `rdna4/README.md:11539` through `:11549` says combined load/store-plus-DS
  waits encode the memory counter in `SIMM16[13:8]` and DS in `SIMM16[5:0]`.
- `rdna4/README.md:2184` through `:2185` gives separate `SAMPLEcnt` and
  `BVHcnt` counters with their own ordering domains.

Rocjitsu evidence:

- `Wavefront::set_wait_counter` truncates the generic threshold to `uint8_t`
  and maps `wait_samplecnt` and `wait_bvhcnt` to `vmcnt` at
  `vm/amdgpu/wavefront.h:376` through `:399`; most direct wait instructions
  pass the full `SIMM16` value through this path, and only `wait_expcnt`
  explicitly masks to 3 bits at `:392` through `:395`.
- Generated `S_WAIT_KMCNT` passes the full 16-bit operand to
  `wf.set_wait_counter("wait_kmcnt", cnt)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/sopp.cpp:587` through
  `:589`; the shared path then narrows to 8 bits at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:376` through `:387`,
  without applying the documented `& 0x1f` mask.
- The wait-counter enum has no separate sample or BVH counter at
  `vm/amdgpu/wait_counters.h:23` through `:34`.
- The DBT wait translator emits combined store/DS waits as `(sc << 4) | dc`,
  with 4-bit clamping for both pieces, at
  `code/dbt/waitcnt_translator.cpp:47` through `:50`.

Impact:

Translated waits can under-wait or wait on the wrong counter field. For
`S_WAIT_KMCNT`, encodings with set bits above `SIMM16[4:0]` can become larger
thresholds in rocjitsu instead of aliases of the low 5-bit value.

### RDNA4-RJ-026: SOPP reserved and compatibility encodings differ from the manual

Manual evidence:

- `rdna4/README.md:6658` through `:6659` marks SOPP opcodes 58 and 59 as
  reserved.
- `rdna4/README.md:11260` through `:11262` says opcode 9 `S_WAITCNT` is
  equivalent to `S_WAIT_IDLE` and ignores its operand for compatibility.

Rocjitsu evidence:

- The RDNA4 decoder maps opcode 58 to `S_TTRACEDATA` and opcode 59 to
  `S_TTRACEDATA_IMM` at `rdna4/decoder.cpp:7495` through `:7496`.
- `S_WAITCNT` decodes the operand as legacy VM/LGKM/EXP counter fields and
  calls `wf.set_wait_target(vm, lgkm, exp)` at `rdna4/sopp.cpp:100` through
  `:115`.

Impact:

Manual-reserved SOPP encodings can decode as executable instructions, and
opcode 9 can under-wait relative to the RDNA4 manual's wait-idle compatibility
rule.

### RDNA4-RJ-027: Reserved padding fields are accepted silently

Manual evidence:

- `rdna4/README.md:1642` says unused fields that are not SGPR source/dest
  fields are typically set to zero.

Rocjitsu evidence:

- RDNA4 machine instruction structs expose padding fields such as
  `pad_23`, `pad_21_23`, and `pad_59_60` at `rdna4/machine_insts.h:80`
  through `:108`.
- Encoding constructors copy the bitfield struct and derive opcode/encoding
  without validating padding, for example `Smem::Smem` at
  `rdna4/encodings.cpp:95` through `:101`.

Impact:

Non-canonical or reserved encodings with nonzero padding are accepted silently.
The manual wording is `typically`, so this is a canonicalization/legality gap
rather than a proven execution-semantics mismatch.

### RDNA4-RJ-028: `S_GET_BARRIER_STATE` is missing from generated RDNA4

Reported by: Nash subreviewer; local audit.

Manual evidence:

- `rdna4/README.md:2086` says `S_GET_BARRIER_STATE` returns barrier state to
  an SGPR as
  `{ 0, signalCnt[6:0], 5'b0, memberCnt[6:0], 3'b0, valid }` and uses
  `KMcnt` to track completion.
- The SOP1 opcode table lists opcode `80` as `S_GET_BARRIER_STATE` at
  `rdna4/README.md:6522`.
- The detailed instruction definition at `rdna4/README.md:10361` through
  `:10376` repeats that `S_GET_BARRIER_STATE` increments/decrements `KMCNT`,
  packs `signalCnt`, `memberCnt`, and `valid`, and requires `S_WAIT_KMCNT`
  before the destination can be read.

Rocjitsu evidence:

- Searches for `S_GET_BARRIER_STATE`, `s_get_barrier_state`, and
  `GET_BARRIER_STATE` found no RDNA4 opcode, decoder, or instruction class
  under `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4`.
- RDNA4 generated opcodes jump from `S_BARRIER_SIGNAL_ISFIRST = 79` at
  `rdna4/opcodes.h:82` to `S_ALLOC_VGPR = 83` at `:83`; opcode `80` is absent.
- The RDNA4 SOP1 decoder table has valid slots for `S_BARRIER_SIGNAL` and
  `S_BARRIER_SIGNAL_ISFIRST` at `rdna4/decoder.cpp:7185` through `:7186`,
  followed by invalid slots where opcode `80` would be.
- The semantic rule exists at `lib/python/amdisa/semantics.py:424`, and the
  generator has a `scalar_barrier_state` emission path at
  `lib/python/amdisa/codegen/_generator.py:3603` through `:3606`.
- GFX1250 generates `SGetBarrierStateSop1` at `gfx1250/sop1.cpp:1816` through
  `:1839`, so the gap follows the generic RDNA4 XML/input set rather than a
  generator-wide inability to emit the class.

Oracle evidence:

- A local `llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding` check accepted
  `s_get_barrier_state s0, -1` and `s_get_barrier_state s0, m0`, producing
  SOP1 opcode `80` encodings.

Impact:

RDNA4 binaries using the manual-listed barrier-state instruction cannot decode
or execute through rocjitsu, despite LLVM accepting the instruction for
`gfx1200` and rocjitsu having a generated path for the nearby `gfx1250`
architecture.

### RDNA4-RJ-029: Trap entry, trap return, and saved-end program control are incomplete

Manual evidence:

- `rdna4/README.md:1846` says `S_ENDPGM_SAVED` terminates because of context
  save and is intended only within a trap handler.
- `rdna4/README.md:1850` through `:1852` says `S_TRAP` waits for outstanding
  instructions, saves `{TrapID, PC}` in `{TTMP1, TTMP0}`, sets `PC = TBA`, and
  sets `PRIV = 1`.
- `rdna4/README.md:1853` through `:1854` says `S_RFE_B64` resumes from the
  source PC value, clears `STATUS.PRIV`, and may be used only within a trap
  handler.

Rocjitsu evidence:

- RDNA4 `S_TRAP` is marked `PROGRAM_TERMINATOR` and throws
  `util::UnimplementedInst` at `rdna4/sopp.cpp:143` through `:155`.
- RDNA4 `S_RFE_B64` decodes at `rdna4/sop1.cpp:1227` through `:1239`, but its
  shared execute helper is empty at
  `shared/execute_shared.h:2405`.
- RDNA4 `S_ENDPGM_SAVED` executes the same `wf.end()` path as `S_ENDPGM` at
  `rdna4/sopp.cpp:371` through `:379`; no context-save or trap-handler
  restriction is modeled.

Impact:

Trap handlers, host-generated traps, saved-wave context-switch paths, and
privilege transitions cannot be modeled from these instructions.

### RDNA4-RJ-030: Halt instructions do not halt or model fatal halt

Manual evidence:

- `rdna4/README.md:1855` through `:1862` defines `S_SETHALT` bit meanings,
  user-vs-trap privilege restrictions, `HALT`, `FATAL_HALT`, delayed halt
  after `S_RFE`, and host-only fatal unhalt behavior.
- `rdna4/README.md:1889` says `S_SENDMSGHALT` performs `S_SENDMSG` and then
  `HALT`.

Rocjitsu evidence:

- RDNA4 `S_SETHALT` dispatches through `rdna4/sopp.cpp:42` through `:51`, but
  `execute_s_sethalt_sopp` is empty at `shared/execute_shared.h:2527`.
- RDNA4 `S_SENDMSGHALT` dispatches through `rdna4/sopp.cpp:413` through `:423`,
  but `execute_s_sendmsghalt_sopp` is empty at
  `shared/execute_shared.h:2491` through `:2492`.

Impact:

Programs that deliberately halt, fatal-halt, or send-and-halt continue running
in rocjitsu.

### RDNA4-RJ-031: Sleep and wakeup state is not modeled

Manual evidence:

- `rdna4/README.md:1868` through `:1871` defines `S_NOP` repeat timing,
  `S_SLEEP` duration and sleep-forever behavior, `S_SLEEP_VAR` duration from
  `SGPR_value[6:0]`, and `S_WAKEUP` for early wakeup of sleeping waves in the
  same work-group.

Rocjitsu evidence:

- `S_SLEEP` dispatches through `rdna4/sopp.cpp:54` through `:62`, but
  `execute_s_sleep_sopp` only calls `request_functional_yield()` at
  `shared/execute_shared.h:2556` through `:2558`.
- `S_SLEEP_VAR` dispatches through `rdna4/sop1.cpp:1333` through `:1347`, but
  `execute_s_sleep_var_sop1` only calls `request_functional_yield()` at
  `shared/execute_shared.h:2561` through `:2563`.
- `execute_s_wakeup_sopp` is empty at `shared/execute_shared.h:2690`.

Impact:

Functional scheduling has no persistent sleep state, no sleep-forever state,
and no observable wakeup behavior.

### RDNA4-RJ-032: `S_CLAUSE` does not validate or record clause state

Manual evidence:

- `rdna4/README.md:1873` says `S_CLAUSE` starts a clause whose type is the next
  instruction's type, whose length is `SIMM16[5:0] + 1`, and whose low six bits
  must be `1-32`, not `0` or `63`.
- `rdna4/README.md:1901` through `:1905` repeats that the clause type is
  implicitly defined by the instruction immediately after `S_CLAUSE`.
- `rdna4/README.md:1907` through `:1917` lists legal clause types, and
  `rdna4/README.md:1921` through `:1932` lists instruction classes that are
  illegal in a clause or legal only after the first clause instruction.

Rocjitsu evidence:

- RDNA4 `S_CLAUSE` accepts `OPR_CLAUSE` at `rdna4/sopp.cpp:64` through `:72`.
- The shared execute helper is empty at `shared/execute_shared.h:1031` through
  `:1032`.
- The generic instruction flags include branch, terminator, memory, wait,
  barrier, MFMA, and acc-VGPR categories at `isa/instruction.h:25` through
  `:50`, but no clause type/category flag.
- Generated semantic properties likewise expose execution mask, matrix,
  barrier, wait, SCC, and related flags at `lib/python/amdisa/sema_properties.py:25`
  through `:36`, while the C++ generator emits broad flags such as
  `MEMORY_OP`, branch, terminator, wait, and barrier at
  `lib/python/amdisa/codegen/_generator.py:6471` and following; no clause
  property is emitted.
- Manual-illegal clause members such as `DS_PARAM_LOAD` and `EXPORT` still have
  normal generated RDNA4 classes at `rdna4/vdsdir.cpp:19` and
  `rdna4/vexport.cpp:19`.

Impact:

Invalid clause immediates and invalid instruction-class membership are accepted,
and no clause boundaries are available to later scheduling, validation, or
instrumentation.

### RDNA4-RJ-033: Split barrier signal/wait semantics are incomplete

Reported by: Nash subreviewer; local audit.

Manual evidence:

- `rdna4/README.md:2021` says split barriers are Signal/Wait: a wave first
  issues `S_BARRIER_SIGNAL`, later issues `S_BARRIER_WAIT`, and wait releases
  after every wave in the work-group has signaled.
- `rdna4/README.md:2025` through `:2033` defines barrier validity after all
  waves have been created, completion after all waves have signaled or
  terminated, and the single-wave/no-work-group `S_NOP` rule.
- `rdna4/README.md:2043` through `:2045` defines the work-group barrier and
  trap barrier; trap-barrier operations from user shaders are ignored.
- `rdna4/README.md:2051` through `:2061` defines `memberCount`,
  `signaledCount`, `barrierComplete`, `trapBarrierComplete`, and
  reset-on-completion/reset-on-wait behavior.
- `rdna4/README.md:2083` through `:2086` defines `S_BARRIER_SIGNAL`,
  `S_BARRIER_SIGNAL_ISFIRST`, `S_BARRIER_WAIT`, and `S_GET_BARRIER_STATE`;
  `ISFIRST` returns `SCC`, and `ISFIRST` / `GET_BARRIER_STATE` use `KMcnt`.
- The detailed definitions increment `signalCnt` and call
  `CheckBarrierComplete` for `S_BARRIER_SIGNAL` at
  `rdna4/README.md:10328` and `:10331`, set `SCC` from whether `signalCnt`
  was zero for `S_BARRIER_SIGNAL_ISFIRST` at `:10355`, and make
  `S_BARRIER_WAIT` wait on and clear `WAVE_BARRIER_COMPLETE[barrierBit]` at
  `:11320` through `:11325`.

Rocjitsu evidence:

- `S_BARRIER_SIGNAL` and `S_BARRIER_SIGNAL_ISFIRST` decode at
  `rdna4/sop1.cpp:1279` through `:1313`, but their shared execute helpers are
  empty at `shared/execute_shared.h:750` through `:755`.
- The Python semantics comment explicitly says arrival accounting and
  named-barrier IDs are not modeled at `lib/python/amdisa/semantics.py:355`
  through `:361`, and the mnemonic map classifies both signal forms as
  `true_nop` at `:422` through `:423`.
- `S_BARRIER_WAIT` accepts the immediate at `rdna4/sopp.cpp:183` through
  `:190`, but execution only sets the wave state to `WfState::BARRIER` at
  `shared/execute_shared.h:758` through `:761`.
- `ComputeUnitCore::update_wf_states` resolves barriers only when all
  non-halted sibling waves in the same dispatch/work-group are also in
  `WfState::BARRIER` at `vm/amdgpu/compute_unit.cpp:318` through `:333`.
- `Wavefront` exposes only a coarse `WfState::BARRIER` at
  `vm/amdgpu/wavefront.h:31` through `:36`; searches found no storage for
  `barrierComplete`, `trapBarrierComplete`, `memberCount`, or `signaledCount`.

Impact:

Rocjitsu can approximate a simple old-style rendezvous where every live wave
parks at a wait instruction, but it cannot model named split barriers,
pre-signaled barriers, trap/work-group barrier special cases, first-arrival
`SCC`, `KMcnt` completion, per-wave completion bits, not-yet-created waves, or
the single-wave/no-work-group immediate `S_NOP` contract.

### RDNA4-RJ-034: `S_CODE_END` executes as a no-op

Manual evidence:

- `rdna4/README.md:1886` says `S_CODE_END` is treated as an illegal
  instruction and is used to pad past the end of shaders.

Rocjitsu evidence:

- RDNA4 decodes `S_CODE_END` at `rdna4/sopp.cpp:197` through `:205`.
- `execute_s_code_end_sopp` is empty at `shared/execute_shared.h:1507`.

Impact:

Falling through past shader code silently continues through padding instead of
raising the manual's illegal-instruction behavior.

### RDNA4-RJ-035: Send-message execution and `KMcnt` behavior are incomplete

Reported by: local audit; Rawls subreviewer.

Manual evidence:

- `rdna4/README.md:1887` says `S_SENDMSG` sends a message upstream and has no
  enforced `S_WAIT_*CNT` before it.
- `rdna4/README.md:1888` says `S_SENDMSG_RTN_B32/B64` uses `KMcnt`, returns
  data to an SGPR or aligned SGPR pair, treats `SSRC0` as an enum, and leaves
  `VCCZ` undefined if writing `VCC`.
- `rdna4/README.md:1889` says `S_SENDMSGHALT` sends a message and then halts.
- `rdna4/README.md:1959` says `S_SENDMSG` and `S_SENDMSG_RTN_B*` carry message
  payloads in `M0`.
- `rdna4/README.md:1963` through `:1965` says `S_SENDMSG_RTN_B*` increments
  `KMcnt` by 2, decrements once when the message goes out and once when data
  returns, and can be waited for with `S_WAIT_KMCNT==0`.
- `rdna4/README.md:1972` through `:1975` defines non-returning message effects
  and payloads for interrupt, HS tessfactor, VGPR/scratch deallocation, and GS
  allocation request.
- `rdna4/README.md:1983` through `:1990` defines return-message values,
  including doorbell, draw/dispatch ID, TMA, realtime, save-wave, TBA,
  `SE_ID`/`AID_ID`, and illegal-return-message cases.

Rocjitsu evidence:

- `S_SENDMSG` dispatches through `rdna4/sopp.cpp:401` through `:410`, but
  `execute_s_sendmsg_sopp` is empty at `shared/execute_shared.h:2428`.
- `S_SENDMSG_RTN_B32/B64` decode enum operands at `rdna4/sop1.cpp:1241`
  through `:1277`, but the shared helpers only synthesize a value for message
  `0x83` and otherwise write zero at `shared/execute_shared.h:2431` through
  `:2488`.
- `RTN_GET_REALTIME` (`0x83`) returns `engine->global_time()` through
  `shared/execute_shared.h:2436` through `:2439` and `:2466` through `:2469`,
  so it is only partial because it still bypasses the return-message wait
  protocol.
- `RTN_SAVE_WAVE` (`0x84`) is accepted as a zero-return case at
  `shared/execute_shared.h:2444` and `:2474`, with no trap-handler privilege
  check or conversion to the illegal-return message.
- `RTN_GET_SE_HW_ID` (`0x87`) is also a zero-return case at
  `shared/execute_shared.h:2447` and `:2477`, so rocjitsu does not pack
  `SE_ID` into `data[3:0]` or `AID_ID` into `data[11:8]`.
- `S_WAIT_KMCNT` does set a `wait_kmcnt` threshold at `rdna4/sopp.cpp:587`
  through `:589`, and `WaitCounters` has a `KMCNT_MAX` and increment support at
  `vm/amdgpu/wait_counters.h:72` through `:76`; the send-message return helpers
  do not use that infrastructure.
- `S_SENDMSG` is empty at `shared/execute_shared.h:2428`, so non-returning
  messages do not increment `KMcnt` until the message is sent out of the WGP as
  required by `rdna4/README.md:2259`.
- `S_SENDMSGHALT` dispatches through `rdna4/sopp.cpp:413` through `:423`, but
  `execute_s_sendmsghalt_sopp` is empty at `shared/execute_shared.h:2491`
  through `:2492`.

Impact:

Message side effects, `M0` payload consumption, returned values other than the
modeled realtime counter, return-message wait semantics, trap-only save-wave
handling, reserved/illegal handling, and send-and-halt behavior do not match the
manual.

### RDNA4-RJ-036: `S_SETPRIO` does not update priority state

Manual evidence:

- `rdna4/README.md:1872` defines `S_SETPRIO` as setting 2 bits of `USER_PRIO`
  and gives the resulting scheduler priority formula.

Rocjitsu evidence:

- RDNA4 `S_SETPRIO` dispatches through `rdna4/sopp.cpp:389` through `:398`.
- `execute_s_setprio_sopp` is empty at `shared/execute_shared.h:2533`.

Impact:

Priority-sensitive scheduling or diagnostics cannot observe `S_SETPRIO`.

### RDNA4-RJ-037: `S_ICACHE_INV` has no instruction-cache effect

Manual evidence:

- `rdna4/README.md:1890` says `S_ICACHE_INV` invalidates the first-level shader
  instruction cache for the WGP associated with the issuing wave.

Rocjitsu evidence:

- RDNA4 `S_ICACHE_INV` dispatches through `rdna4/sopp.cpp:476` through `:484`.
- `execute_s_icache_inv_sopp` is empty at `shared/execute_shared.h:1771`.

Impact:

Self-modifying or instruction-cache-sensitive scenarios cannot observe the
manual's invalidation operation in rocjitsu.

### RDNA4-RJ-038: Clause start, skip, and break behavior is absent

Manual evidence:

- `rdna4/README.md:1933` says `S_TRAP` is legal inside a clause, even as the
  first instruction after `S_CLAUSE`, and ends the clause.
- `rdna4/README.md:1935` says pseudo-scalar `V_S_*` instructions are VALU ops
  that may be used in VALU clauses.
- `rdna4/README.md:1937` defines VALU-clause behavior when `EXEC==0` at the
  first instruction versus becoming zero in the middle of the clause.
- `rdna4/README.md:1939` through `:1947` requires `S_DELAY_ALU` to come before
  `S_CLAUSE` when delaying before a clause, and says a skipped first
  instruction after `S_CLAUSE` prevents the clause from starting.
- `rdna4/README.md:1951` through `:1955` lists clause-break causes from VALU
  exceptions, host commands, context save, halt/kill, and trap-handler entry.

Rocjitsu evidence:

- Searches for `CLAUSE` or `clause` in `isa/instruction.h`,
  `vm/amdgpu/wavefront.h`, `vm/amdgpu/compute_unit.cpp`,
  `code/basic_block.cpp`, and `code/dbt` found no clause state or stream-rule
  implementation.
- `WfState` has `HALTED`, `RUNNING`, `WAITCNT`, `BARRIER`, and `ENDING` states
  at `vm/amdgpu/wavefront.h:31` through `:38`, but no active-clause state.
- `ComputeUnitCore::step` iterates all `RUNNING` wavefronts and issues one
  instruction from each at `vm/amdgpu/compute_unit.cpp:442` through `:450`,
  so no one-wave uninterrupted clause arbitration is visible.
- RDNA4 `S_DELAY_ALU` decodes at `rdna4/sopp.cpp:76` through `:86`, but
  `execute_s_delay_alu_sopp` is empty at `shared/execute_shared.h:1643`.
- RDNA4 `S_WAIT_ALU` decodes and carries the `WAITCNT` flag at
  `rdna4/sopp.cpp:88` through `:100`, but `execute_s_wait_alu_sopp` is empty
  at `shared/execute_shared.h:2675`.
- `BasicBlock` terminator logic keys off branch and program-terminator flags
  at `code/basic_block.cpp:24` through `:34`, with no clause-break metadata.

Impact:

Rocjitsu cannot model whether a clause starts, which instruction type it binds
to, whether it survives skipped/disabled first instructions, or when hardware
would break the clause.

### RDNA4-RJ-039: `RTN_ILLEGAL_MSG` collides with the generic SOP1 literal path

Reported by: Rawls subreviewer; local audit.

Manual/XML evidence:

- `rdna4/README.md:1959` says `S_SENDMSG_RTN_B*` encodes the message type in
  the `SSRC0` instruction field and does not read an SGPR.
- `rdna4/README.md:1990` lists `RTN_ILLEGAL_MSG` at code `0xFF`.
- The XML `OPR_SENDMSG_RTN` enum lists `msg_rtn_illegal_msg` value `255` at
  `amdgpu_isa_rdna4.xml:182748` through `:182751`.
- A local LLVM assembler sanity check accepted
  `s_sendmsg_rtn_b32 s0, 0xff` as a single SOP1 word with encoding
  `[0xff,0x4c,0x80,0xbe]`, not as a literal-extension instruction.

Rocjitsu evidence:

- RDNA4 `SSendmsgRtnB32Sop1` first constructs `ssrc0` as `OPR_SENDMSG_RTN`, but
  then rewrites it to `OPR_SIMM32` whenever the raw `ssrc0` field is `255` at
  `rdna4/sop1.cpp:1241` through `:1254`.
- RDNA4 `SSendmsgRtnB64Sop1` has the same rewrite at `rdna4/sop1.cpp:1260`
  through `:1273`.
- The generator maps all `ENC_SOP1` forms to `Sop1InstLiteralMachineInst`
  literal handling at `lib/python/amdisa/codegen/_generator.py:96`, even though
  the `S_SENDMSG_RTN_B32/B64` XML entries have only the default `ENC_SOP1`
  encoding with no literal-extension form.

Impact:

The architectural illegal-return-message code can be decoded as a 32-bit literal
read from the following word. That can corrupt disassembly and execution of
`S_SENDMSG_RTN_* 0xff`, and can also consume the next instruction word as data.

### RDNA4-RJ-040: `S_SENDMSG_RTN_B64` does not enforce aligned SGPR-pair destinations

Reported by: Rawls subreviewer; local audit.

Manual/XML evidence:

- `rdna4/README.md:1888` says `S_SENDMSG_RTN_B64` returns data to an aligned
  SGPR pair.
- The XML `OPR_SDST` description says 64-bit SGPR values must be even-aligned at
  `amdgpu_isa_rdna4.xml:181946` through `:181950`.

Rocjitsu evidence:

- `SSendmsgRtnB64Sop1` constructs `sdst` as a 64-bit `OPR_SDST` at
  `rdna4/sop1.cpp:1260` through `:1264`.
- The shared return-message helper writes through `write_scalar64` at
  `shared/execute_shared.h:2484` through `:2485`.
- RDNA4 `resolve_dst_write64` accepts `ev <= 105` and writes consecutive SGPRs
  at `rdna4/operand.cpp:960` through `:970`, and accepts `ev >= 108 && ev <=
  122` similarly at `:976` through `:981`, with no even-address check.

Impact:

Odd or otherwise invalid 64-bit scalar destinations are accepted instead of
being rejected or modeled according to the ISA alignment rule. This appears to
be inherited generic `OPR_SDST` behavior, but `S_SENDMSG_RTN_B64` is one manual
site where the aligned-pair contract is explicit.

### RDNA4-RJ-041: `S_SETPC_B64` / `S_SWAPPC_B64` branch-to-zero halts instead of setting `PC`

Reported by: Arendt subreviewer.

Manual evidence:

- `rdna4/README.md:2010` says `S_SETPC_B64` directly sets `PC` from an SGPR pair.
- `rdna4/README.md:2011` says `S_SWAPPC_B64` swaps the next-instruction `PC`
  with an address in an SGPR pair.
- The scoped Chapter 5.5 text does not describe a special halt behavior for a
  zero target.

Rocjitsu evidence:

- Before executing an instruction, `ComputeUnitCore::step` checks mnemonics
  containing `s_setpc` or `s_swappc`, reads a target from the raw `ssrc0` low
  seven bits as an SGPR pair, and halts the wave when that target is zero at
  `vm/amdgpu/compute_unit.cpp:395` through `:407`.
- The generated execute bodies otherwise implement the manual set/swap behavior:
  `SSetpcB64Sop1::execute_impl` sets `wf.pc` from `ssrc0` at
  `rdna4/sop1.cpp:1201` through `:1202`, while
  `SSwappcB64Sop1::execute_impl` writes next PC and then sets `wf.pc` at
  `rdna4/sop1.cpp:1221` through `:1224`.

Impact:

Production wave execution can terminate instead of transferring control to
address zero. Focused instruction execution that bypasses `ComputeUnitCore::step`
would not catch this behavior.

### RDNA4-RJ-042: Branch SGPR-pair operands are not checked for even alignment

Reported by: Arendt subreviewer; local audit.

Manual/XML evidence:

- `rdna4/README.md:2010` through `:2013` describes `S_SETPC_B64`,
  `S_SWAPPC_B64`, `S_GETPC_B64`, and `S_CALL_B64` as operating on SGPR pairs.
- The XML `OPR_SREG` description says 64-bit SGPR values must be even-aligned at
  `amdgpu_isa_rdna4.xml:189669` through `:189672`, and `OPR_SDST` says the same
  at `amdgpu_isa_rdna4.xml:181946` through `:181950`.
- A local LLVM assembler sanity check rejected odd pairs such as
  `s_getpc_b64 s[1:2]`, `s_call_b64 s[1:2], 0`, `s_setpc_b64 s[1:2]`, and
  `s_swappc_b64 s[1:2], s[3:4]` with `invalid register alignment`.

Rocjitsu evidence:

- `SGetpcB64Sop1`, `SSwappcB64Sop1`, and `SCallB64Sopk` construct 64-bit
  `OPR_SDST` destinations at `rdna4/sop1.cpp:1174` through `:1178`,
  `rdna4/sop1.cpp:1205` through `:1209`, and `rdna4/sopk.cpp:224` through
  `:227`.
- `SSetpcB64Sop1` and `SSwappcB64Sop1` construct 64-bit `OPR_SREG` sources at
  `rdna4/sop1.cpp:1187` through `:1191` and `:1205` through `:1209`.
- RDNA4 64-bit scalar reads and writes use the encoded low register and the next
  register directly at `rdna4/operand.cpp:856` through `:864` and `:960`
  through `:981`, with no even-low-register check.

Impact:

Odd SGPR-pair encodings can be decoded and executed as consecutive scalar
registers even though the assembler and operand descriptions require aligned
pairs. This is inherited generic scalar-pair behavior, but Chapter 5.5 gives
several PC-control sites where the low-register legality matters.

### RDNA4-RJ-043: `S_SETPC_B64` accepts an invalid literal source encoding

Reported by: local audit; Arendt subreviewer.

Manual/XML evidence:

- `rdna4/README.md:2010` says `S_SETPC_B64` sets `PC` from an SGPR pair.
- The XML `S_SETPC_B64` source operand is `SSRC0` with `OPR_SREG`, not a
  literal-capable operand class, at `amdgpu_isa_rdna4.xml:21356` through
  `:21360`.
- A local LLVM assembler sanity check rejected `s_setpc_b64 0`,
  `s_setpc_b64 1`, `s_setpc_b64 -1`, and `s_setpc_b64 0xffffffff` as invalid
  operands.

Rocjitsu evidence:

- `SSetpcB64Sop1` constructs `ssrc0` as `OPR_SREG`, but rewrites raw
  `ssrc0 == 255` to a 64-bit `OPR_SIMM32` using the following literal word at
  `rdna4/sop1.cpp:1187` through `:1198`.
- The runtime then reads the rewritten operand through `read_scalar64`, which
  returns immediate values for immediate operand types at `rdna4/operand.cpp:1158`
  through `:1163`.
- `S_SWAPPC_B64` has similar constructor literal handling at `rdna4/sop1.cpp:1214`
  through `:1217`, but LLVM accepts a literal source for `s_swappc_b64`; the
  confirmed over-acceptance in this slice is therefore `S_SETPC_B64`.

Impact:

An invalid raw `S_SETPC_B64` encoding with `ssrc0 == 255` can consume the next
instruction word as a 32-bit literal and branch through that value instead of
being rejected or handled as an invalid SGPR-pair source.

### RDNA4-RJ-044: `S_CALL_B64` uses the `INDIRECT_CALL` flag despite having a direct target

Reported by: Arendt subreviewer; local audit.

Manual/XML evidence:

- `rdna4/README.md:2013` defines `S_CALL_B64` as a PC-relative call:
  `PC = PC + SIMM16*4`.
- The XML marks `S_CALL_B64` as a branch with `IsIndirectBranch` false at
  `amdgpu_isa_rdna4.xml:37614` through `:37622`.

Rocjitsu evidence:

- RDNA4 `SCallB64Sopk` sets `flags_ |= INDIRECT_CALL` at `rdna4/sopk.cpp:224`
  through `:233`, while its `branch_offset_bytes()` and execute body use the
  signed immediate at `rdna4/sopk.cpp:235` through `:243`.
- `INDIRECT_CALL` is documented as "target from register, returns to
  fallthrough" in `isa/instruction.h:30` through `:33`.
- CFG/DBT code compensates by checking `branch_offset_bytes()` for direct-call
  handling, for example `code/basic_block.cpp:51` through `:56` and
  `code/dbt/binary_translator.cpp:787` through `:793`.

Impact:

Runtime PC behavior matches the manual, but consumers that interpret the public
instruction flag literally can classify a direct PC-relative call as an indirect
target-from-register call. The existing CFG/DBT paths already account for this
with `branch_offset_bytes()`, so this is a metadata precision gap rather than a
known execution mismatch.

### RDNA4-RJ-045: Split-barrier signal instructions accept invalid literal sources

Reported by: local audit; Ptolemy the 2nd subagent.

Manual/XML/oracle evidence:

- Chapter 5.6 says barrier instructions can use `M0` or an inline constant for
  the barrier number, except `S_BARRIER_WAIT` can only take an inline constant
  at `rdna4/README.md:2065` through `:2067`.
- The detailed `S_BARRIER_SIGNAL` and `S_BARRIER_SIGNAL_ISFIRST` definitions
  say `M0` support is reserved for other architectures at
  `rdna4/README.md:10307` and `:10338`, but neither detailed definition
  permits an arbitrary 32-bit literal extension.
- RDNA4 XML nevertheless declares `SOP1_INST_LITERAL` alternatives for
  `S_BARRIER_SIGNAL` and `S_BARRIER_SIGNAL_ISFIRST` at
  `amdgpu_isa_rdna4.xml:21566` through `:21573` and `:21613` through `:21620`.
- A local `llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding` check rejected
  `s_barrier_signal 0x12345678`,
  `s_barrier_signal_isfirst 0x12345678`, and
  `s_get_barrier_state s0, 0x12345678`.

Rocjitsu evidence:

- `SBarrierSignalSop1` rewrites raw `ssrc0 == 255` to `OPR_SIMM32` from the
  following literal word at `rdna4/sop1.cpp:1287` through `:1290`.
- `SBarrierSignalIsfirstSop1` does the same rewrite at
  `rdna4/sop1.cpp:1306` through `:1309`.
- The generic generator keeps the RDNA4 `SOP1_INST_LITERAL` path because the
  gfx1250-only profile override that skips `SOP1_INST_LITERAL` does not apply
  to generic RDNA4; see `lib/python/amdisa/isa_profile.py:1559` through
  `:1562` and the gfx1250 regression at
  `lib/python/amdisa/tests/test_profile_properties.py:306` through `:307`.

Impact:

Rocjitsu can decode and construct raw split-barrier signal encodings with
`ssrc0 == 255` as literal-extension instructions, while LLVM rejects those
source forms for `gfx1200`. The execute helpers are currently no-ops, so the
present runtime effect is mostly decode/disassembly legality; it becomes a
semantic issue if split-barrier execution is implemented on top of the accepted
literal operand.

### RDNA4-RJ-046: `S_BARRIER_SIGNAL_ISFIRST` is missing barrier metadata

Reported by: Nash subreviewer; local audit.

Manual/XML evidence:

- `rdna4/README.md:2083` says `S_BARRIER_SIGNAL_ISFIRST` is the same barrier
  signal operation plus an `SCC` result.
- RDNA4 XML records an implicit `SCC` output for
  `S_BARRIER_SIGNAL_ISFIRST` at `amdgpu_isa_rdna4.xml:21605` through `:21609`.

Rocjitsu evidence:

- The C++ generator's `_barrier_names` set includes `S_BARRIER`,
  `S_BARRIER_SIGNAL`, and `S_BARRIER_WAIT`, but not
  `S_BARRIER_SIGNAL_ISFIRST`, at
  `lib/python/amdisa/codegen/_generator.py:6529` through `:6537`.
- Generated RDNA4 `SBarrierSignalSop1` sets `flags_ |= BARRIER` at
  `rdna4/sop1.cpp:1279` through `:1291`.
- Generated RDNA4 `SBarrierSignalIsfirstSop1` decodes at
  `rdna4/sop1.cpp:1298` through `:1313`, but has no corresponding
  `flags_ |= BARRIER`.
- The profile-gate test explicitly excludes RDNA4
  `S_BARRIER_SIGNAL_ISFIRST` from implicit-SCC access checking at
  `lib/python/amdisa/tests/test_generator_profile_gates.py:156` through
  `:159`, matching the current generated omission.

Impact:

Any rocjitsu consumer that relies on `Instruction::is_barrier()` can see
`S_BARRIER_SIGNAL` and `S_BARRIER_WAIT` as barrier instructions while missing
the `ISFIRST` variant, even though the manual defines it as a barrier signal
with an additional `SCC` result.

### RDNA4-RJ-047: `S_WAIT_IDLE` and `S_WAIT_EVENT` execute as no-ops

Reported by: Darwin subreviewer; local audit.

Manual evidence:

- `rdna4/README.md:11264` through `:11266` says `S_WAIT_IDLE` waits for all
  wave activity to complete, including all dependency and memory counters.
- `rdna4/README.md:11268` through `:11270` says `S_WAIT_EVENT` waits for an
  event or condition specified by `SIMM16`.
- `rdna4/README.md:2148` says `S_WAIT_EVENT` bit 1 waits for export-ready,
  other bits are reserved, and exceptions wait for the event before processing.

Rocjitsu evidence:

- RDNA4 `S_WAIT_IDLE` and `S_WAIT_EVENT` decode and set `flags_ |= WAITCNT` at
  `rdna4/sopp.cpp:117` through `:141`.
- The shared execute helpers are empty at
  `shared/execute_shared.h:2678` through `:2683`.

Impact:

Code that relies on `S_WAIT_IDLE` to drain outstanding work or on
`S_WAIT_EVENT` to observe export-ready/event state continues immediately in
rocjitsu.

### RDNA4-RJ-048: Flat instructions do not model dual `DScnt` plus `LOADcnt`/`STOREcnt` participation

Reported by: Darwin subreviewer; local audit.

Manual evidence:

- `rdna4/README.md:2231` says every Flat instruction increments both `DScnt`
  and either `LOADcnt` or `STOREcnt`, and decrements the two counters at
  different completion points.
- `rdna4/README.md:2233` through `:2243` says Flat behaves as simultaneous
  VMEM and LDS halves with complementary `EXEC` masks, independent completion,
  and ordered LDS-side interaction with LDS instructions.

Rocjitsu evidence:

- The generated producer map assigns `flat_load` to `LOADCNT`, `flat_store` to
  `STORECNT`, and `flat_atomic` to `LOADCNT` for GFX11+ at
  `lib/python/amdisa/codegen/_generator.py:4435` through `:4452`.
- Generated RDNA4 flat loads set a single `LOADCNT` producer, for example
  `FlatLoadU8Vflat::execute_impl` at `rdna4/vflat.cpp:38` through `:48`.
- Generated RDNA4 flat stores set a single `STORECNT` producer, for example
  `FlatStoreB8Vflat::execute_impl` at `rdna4/vflat.cpp:240` through `:245`.
- `MemoryPipeline::issue` increments and later releases exactly one
  `WaitCounterType` for each memory instruction at `vm/amdgpu/memory_pipeline.h:62`
  through `:87`.
- Shared-aperture Flat routing retags the instruction to `LOCAL_MEM` and
  overwrites the counter with legacy `LGKMCNT` rather than adding `DSCNT` at
  `vm/amdgpu/compute_unit.cpp:272` through `:281`.

Impact:

Flat waits can miss the required DS-side or VMEM-side dependency, and
shared-aperture Flat accesses can wait on the legacy aggregate counter instead
of the RDNA4 split counters.

### RDNA4-RJ-049: Multi-dword SMEM operations undercount `KMcnt`

Reported by: Darwin subreviewer.

Manual evidence:

- `rdna4/README.md:2194` through `:2198` says `KMcnt` increments by one for
  32-bit-and-smaller SMEM loads, by two for larger loads, decrements for
  returned SMEM data, and scalar-memory loads can return out of order.
- `rdna4/README.md:2213` repeats the same SMEM increment rule and notes that
  only `S_WAIT_KMCNT <= 0` is sensible for out-of-order scalar loads.
- `rdna4/README.md:3964` through `:3968` restates the Chapter 8.2
  scalar-memory rule: `KMcnt` increments by one for single-dword fetches, by
  two for two-or-more-dword fetches, and decrements by an equal amount when the
  instruction completes.

Rocjitsu evidence:

- `S_LOAD_B64` records `num_dwords = 2` and `wait_counter_type = KMCNT` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/smem.cpp:70` through `:77`;
  `S_LOAD_B512` similarly records 16 dwords with one `KMCNT` counter type at
  `:151` through `:158`.
- `MemoryPipeline::issue` increments one selected counter once per issued
  instruction at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.h:62`
  through `:77`, and `finish_completed_access` releases that counter once at
  `:104` through `:106`.
- `WaitCounters::increment(WaitCounterType::KMCNT)` increments `kmcnt` and
  `lgkmcnt` by one at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wait_counters.h:96` through `:99`.

Impact:

Wide scalar memory operations can become visible to `S_WAIT_KMCNT` one count
too early relative to the manual's two-count accounting for wider SMEM fetches.

### RDNA4-RJ-050: Atomic return/no-return wait-counter selection is incomplete

Reported by: Darwin subreviewer.

Manual evidence:

- `rdna4/README.md:2181` through `:2183` says vector-memory loads and
  atomic-with-return instructions increment `LOADcnt`.
- `rdna4/README.md:2186` through `:2188` says vector-memory stores and
  atomic-without-return instructions increment `STOREcnt`.

Rocjitsu evidence:

- The generator maps `flat_atomic` to `LOADCNT` unconditionally for GFX11+ at
  `lib/python/amdisa/codegen/_generator.py:4449` through `:4452`.
- Generated RDNA4 global atomic code computes whether the atomic returns from
  `gfx12_atomic_returns(inst_.th)`, but still assigns `LOADCNT` regardless of
  the no-return case at `rdna4/vglobal.cpp:845` through `:853`.
- Generated RDNA4 buffer atomic code also computes whether the atomic returns
  from `gfx12_atomic_returns(inst_.th)`, but does not assign
  `wait_counter_type`; for example `BUFFER_ATOMIC_SWAP_B32` sets `is_load`,
  `atomic_op`, and `mtype` at `rdna4/vbuffer.cpp:1188` through `:1201` without
  selecting `LOADCNT` or `STORECNT`.
- The generic `VectorMemState` constructor defaults non-local vector memory to
  `VMCNT` at `vm/amdgpu/mem_state.h:87` through `:102`, while ordinary RDNA4
  buffer loads and stores explicitly set `LOADCNT` and `STORECNT` respectively
  at `rdna4/vbuffer.cpp:511` through `:522` and `:664` through `:675`.

Impact:

No-return atomics can be waited on as loads instead of stores, so
`S_WAIT_STORECNT` may fail to cover outstanding atomic writes while
`S_WAIT_LOADCNT` can over-wait on them.

### RDNA4-RJ-051: Export and LDS-direct/parameter load `EXPcnt` producers are not modeled

Reported by: Darwin subreviewer.

Manual evidence:

- `rdna4/README.md:2202` through `:2209` says `EXPcnt` tracks VGPR exports,
  `DS_PARAM_LOAD`, and `DS_DIRECT_LOAD`, with export-family ordering and
  DS-direct/parameter ordering separate from exports.
- `rdna4/README.md:2249` says LDS parameter and direct loads use `EXPcnt` to
  track when VGPR writes have completed.
- `rdna4/README.md:2253` through `:2255` says exports use `EXPcnt` to prevent
  write-after-read hazards on source VGPRs and that `EXEC` is read only after
  the export is granted.

Rocjitsu evidence:

- RDNA4 `ExportVexport::execute_impl` is a compute-simulation no-op at
  `rdna4/vexport.cpp:35` through `:36`.
- RDNA4 `DsParamLoadVdsdir::execute_impl` and
  `DsDirectLoadVdsdir::execute_impl` are no-ops at `rdna4/vdsdir.cpp:30`
  through `:44`.
- These instruction classes do not attach memory state or increment `EXPCNT`,
  so `S_WAIT_EXPCNT` has no producer-side work for them to observe.

Impact:

Shaders can overwrite export source VGPRs or `EXEC` without an effective
`EXPcnt` wait, and LDS-direct/parameter load dependencies are invisible to the
wait-counter model.

### RDNA4-RJ-052: Counter overflow and ordered decrement behavior are simplified

Reported by: Darwin subreviewer; local audit.

Manual evidence:

- `rdna4/README.md:2169` says hardware prevents counter overflow by stalling
  issue of any instruction whose increment would overflow the counter.
- `rdna4/README.md:2171` through `:2175` says load data can write VGPRs out of
  order but counter decrement still reflects in-order completion, and counters
  count instructions rather than threads.
- `rdna4/README.md:2133` and `:2167` describe same-type ordering, SMEM
  out-of-order behavior, and different-type out-of-order completion.
- `rdna4/README.md:4088` repeats the vector-buffer case: different types of
  vector-memory buffer operations, such as loads and stores, can complete out
  of order with respect to each other.

Rocjitsu evidence:

- `WaitCounters::increment` saturates counters with `std::min` rather than
  stalling issue at `vm/amdgpu/wait_counters.h:76` through `:107`.
- `MemoryPipeline::issue` chooses one counter and later releases that same
  counter directly when the memory access completes at
  `vm/amdgpu/memory_pipeline.h:62` through `:87`.
- `Wavefront::release_wait_counter` decrements the counter immediately and
  wakes the wave if thresholds are satisfied at `vm/amdgpu/wavefront.cpp:21`
  through `:27`.

Impact:

The functional model can represent simple outstanding-work waits, but it cannot
model issue backpressure at counter capacity or the manual's per-counter
ordered decrement rules for out-of-order memory completion.

### RDNA4-RJ-053: RDNA4 DOT4 BF8 variants decode BF8 operands as FP8

Reported by: Archimedes subreviewer; local audit.

Manual/XML evidence:

- RDNA4 Chapter 7.7 lists four FP8/BF8 DOT4 variants in the packed math opcode
  set at `rdna4/README.md:3180` through `:3182`.
- The checked-in XML distinguishes the operand formats:
  `V_DOT4_F32_FP8_BF8` has `SRC0=PK4_FP8` and `SRC1=PK4_BF8` at
  `amdgpu_isa_rdna4.xml:125607` through `:125616`;
  `V_DOT4_F32_BF8_FP8` reverses those formats at `:125871` through `:125880`;
  `V_DOT4_F32_BF8_BF8` uses BF8 for both at `:126399` through `:126408`.

Rocjitsu evidence:

- `lib/python/amdisa/semantics.py:1596` through `:1603` maps all four
  `V_DOT4_F32_{FP8,BF8}_{FP8,BF8}` mnemonics to one semantic class,
  `dot4_f32_fp8`.
- `lib/python/amdisa/codegen/execute/packed.py:961` through `:971` emits
  `util::fp8_e4m3_to_f32` for both packed sources in that class.
- The generated `V_DOT4_F32_FP8_BF8`, `V_DOT4_F32_BF8_FP8`, and
  `V_DOT4_F32_BF8_BF8` bodies call `util::fp8_e4m3_to_f32` for both sources at
  `rdna4/vop3p.cpp:3090` through `:3098`, `:3182` through `:3190`, and
  `:3366` through `:3374`.
- Rocjitsu has distinct OCP decode helpers for FP8 and BF8 at
  `lib/util/include/util/data_types.h:321` and `:643`.

Impact:

Any RDNA4 BF8 DOT4 source is interpreted as FP8 E4M3, so mixed and BF8-only
DOT4 results can be wrong even when the instruction is otherwise decoded.

### RDNA4-RJ-054: DOT4 FP8/BF8 source-2 modifiers and legality are not modeled

Reported by: Archimedes subreviewer; local audit.

Manual evidence:

- `DOT4_F32_{FP8,BF8}_{FP8,BF8}` requires `OPSEL=0`, `OPSEL_HI=7`, permits
  `NEG`/`NEG_HI` only on `SRC2`, treats `NEG_HI` as `ABS`, disallows
  `OMOD`/`CLAMP`, uses round-to-nearest-even, and reports no exceptions at
  `rdna4/README.md:3228` through `:3230`.
- Chapter 7.7.2 also says packed math instructions, excluding WMMA, with source
  float data sizes smaller than 16 bits do not work with inline constants at
  `rdna4/README.md:3257` through `:3259`.

Rocjitsu evidence:

- The DOT4 FP8/BF8 generator path reads `SRC2` directly as an accumulator and
  never applies `inst_.neg`, `inst_.neg_hi`, `inst_.opsel`, or
  `inst_.opsel_hi` to it at
  `lib/python/amdisa/codegen/execute/packed.py:961` through `:976`.
- The generated BF8/FP8 DOT4 bodies follow the same pattern, reading `SRC2`,
  accumulating, and writing `VDST` without enforcing the selector, modifier,
  or clamp restrictions at `rdna4/vop3p.cpp:3090` through `:3098`,
  `:3182` through `:3190`, and `:3366` through `:3374`.
- Generated constructors are driven by the XML `OPR_SRC` class for DOT4
  operands, which includes inline constants and the literal selector; no
  DOT4-specific constructor legality check overrides the broad XML source
  class.

Impact:

Rocjitsu cannot emulate the manual's `SRC2` negate/absolute-value behavior for
FP8/BF8 DOT4, and it does not reject or canonicalize selector, clamp, or
inline-constant combinations that the manual marks unsupported.

### RDNA4-RJ-055: VOP3P disassembly hides DOT signedness and selector/modifier fields

Reported by: Archimedes subreviewer; local audit.

Manual evidence:

- `V_DOT4...IU...` uses `NEG[1:0]` to encode source signedness rather than
  floating-point negation at `rdna4/README.md:3224` through `:3227`.
- `DOT4_F32_{FP8,BF8}_{FP8,BF8}` has required selector and modifier state at
  `rdna4/README.md:3228` through `:3230`.

Rocjitsu evidence:

- Generic disassembly appends encoding-specific modifiers only through
  `build_modifiers()` at `lib/rocjitsu/src/rocjitsu/isa/instruction.h:247`
  through `:255`; the default implementation emits nothing at `:271` through
  `:274`.
- The RDNA4 `Vop3p` base in
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/encodings.h:540` through
  `:561` does not override `build_modifiers()`, while other encoding bases do.
- Generated integer DOT execution consumes `inst_.neg` as signedness at
  `rdna4/vop3p.cpp:1818` through `:1830`, but that state is not surfaced by
  VOP3P disassembly.

Impact:

Disassembling VOP3P DOT instructions can omit semantically relevant `NEG`,
`NEG_HI`, `OPSEL`, `OPSEL_HI`, and `CLAMP` state. That makes decoder or corpus
round-tripping unable to catch signedness/selector/modifier mismatches in this
instruction family.

### RDNA4-RJ-056: SWMMAC wave64 sparse index selection drops `OPSEL[1]`

Reported by: Russell subreviewer; local audit.

Manual evidence:

- Chapter 7.12.3 says sparse index VGPRs hold two index sets in wave32 selected
  by `OPSEL[0]`, and four index sets in wave64 selected by `OPSEL[1:0]`, at
  `rdna4/README.md:3822`.

Rocjitsu evidence:

- The matrix execution generator always emits `uint32_t index_key =
  inst_.opsel & 0x1u` for RDNA4 SWMMAC at
  `lib/python/amdisa/codegen/execute/matrix.py:310` through `:316`.
- Generated RDNA4 SWMMAC bodies follow that one-bit mask, for example
  `V_SWMMAC_F32_16X16X32_F16` at `rdna4/vop3p.cpp:4455` through `:4462` and
  `V_SWMMAC_I32_16X16X64_IU4` at `rdna4/vop3p.cpp:5004` through `:5021`.
- `read_swmmac_index_set` branches only on `index_key & 0x1u` for 16- and
  32-entry sparse index data at `shared/mma_exec.h:428` through `:449`.

Impact:

In wave64 mode, index sets selected by OPSEL values 2 and 3 alias the OPSEL
values 0 and 1 paths. SWMMAC programs that rely on the manual's four wave64
index sets can read the wrong sparse expansion data.

### RDNA4-RJ-057: `V_SWMMAC_I32_16X16X64_IU4` wave64 layout is generated but unsupported

Reported by: Russell subreviewer; local audit.

Manual evidence:

- Chapter 7.12 lists `V_SWMMAC_I32_16X16X64_IU4` in the SWMMAC table at
  `rdna4/README.md:3571`.
- The VGPR usage table includes the 4-bit wave64 16x32 packed / 16x64 expanded
  sparse case at `rdna4/README.md:3688` through `:3689`.
- Chapter 7.12.3 says matrices with larger K values operate similarly to the
  K=32 sparse-index selection model at `rdna4/README.md:3822`.

Rocjitsu evidence:

- The generated RDNA4 body calls `exec_swmmac_i32` with `K=64`,
  `index_entries=16`, and `wf.wf_size()` at `rdna4/vop3p.cpp:5004` through
  `:5021`.
- The wave64 sparse index helper handles only `M==16`, `K==32`, and
  `index_entries==16`, otherwise throwing `unsupported gfx12 wave64 SWMMAC
  index layout` at `shared/mma_exec.h:482` through `:495`.
- The wave64 sparse A helper similarly only handles `M==16` and `K==32`, then
  throws at `shared/mma_exec.h:515` through `:530`.
- The wave64 sparse B helper only handles `N==16` and `K==32`, then throws at
  `shared/mma_exec.h:547` through `:561`.

Impact:

The instruction is decodable and has a generated execution body, but a wave64
execution of the manual-listed K=64 IU4 SWMMAC shape reaches unsupported layout
paths instead of emulating the operation.

### RDNA4-RJ-058: Dense IU4 wave64 WMMA layouts do not match the manual special cases

Reported by: Russell subreviewer; local audit.

Manual evidence:

- Chapter 7.12 lists dense `V_WMMA_I32_16X16X16_IU4` and
  `V_WMMA_I32_16X16X32_IU4` at `rdna4/README.md:3557` through `:3558`.
- The layout section says `WMMA_16X16X16_IU4` works in wave64 but uses only
  lanes 0 through 31, and `WMMA_16X16X32_IU4` works in wave32 but uses two
  VGPRs, at `rdna4/README.md:3661` through `:3662`.
- The VGPR usage table includes the 4-bit wave64 16x32 dense case at
  `rdna4/README.md:3688` through `:3689`, with the "first subv" note naming
  lanes 0 through 31 at `rdna4/README.md:3699`.

Rocjitsu evidence:

- Generated RDNA4 dense IU4 execution bodies pass the live wave size into the
  generic helpers: K=16 at `rdna4/vop3p.cpp:3908` through `:3921`, and K=32 at
  `rdna4/vop3p.cpp:4360` through `:4373`.
- The shared wave64 helper accepts K=16 but maps K groups across all four
  16-lane groups, then throws for other shapes at `shared/mma_exec.h:295`
  through `:305`.
- Existing helper coverage asserts the all-four-groups K=16 mapping, including
  lanes 35 and 51, at `tests/shared_infra_test.cpp:357` through `:376`.

Impact:

The K=16 IU4 wave64 path consumes lanes the manual says are unused, while the
K=32 dense IU4 path can reach the unsupported wave64 helper case instead of
applying the manual's special wave/layout rule.

### RDNA4-RJ-059: WMMA/SWMMAC DPP and literal legality is too permissive in generated classes

Reported by: Russell subreviewer; local audit.

Manual evidence:

- Chapter 7.7.4 marks the VOP3P `WMMA` class as `NO DPP` at
  `rdna4/README.md:3388` through `:3400`.
- The FP8/BF8 WMMA subsection says A and B must come from VGPRs, C may be VGPR
  or inline, and these forms do not support OPSEL, ABS, NEG, OMOD, DPP,
  `FP16_OVFL`, or clamp at `rdna4/README.md:3232` through `:3234`.

Rocjitsu evidence:

- A representative dense FP8 WMMA constructor still rewrites source selector
  255 as `OPR_SIMM32` for `SRC0`, `SRC1`, and `SRC2`, and accepts DPP/DPP8
  source-0 encodings at `rdna4/vop3p.cpp:3952` through `:3980`; its execute
  body applies DPP when those fields are present at `:3983` through `:3998`.
- Representative SWMMAC constructors do the same literal and DPP/DPP8 handling
  at `rdna4/vop3p.cpp:4404` through `:4448` and `:4955` through `:4983`.

Impact:

The generated classes contain permissive literal/DPP paths for instruction
forms the manual marks unsupported. The current XML-driven decoder may make
some DPP alternatives unreachable, so this still needs negative decoder tests,
but the generated legality surface is broader than the manual contract.

### RDNA4-RJ-060: WMMA/SWMMAC modifier restrictions are not enforced or fully emulated

Reported by: Russell subreviewer; local audit.

Manual evidence:

- For IU4/IU8 WMMA, `NEG[1:0]` selects A/B signedness and `NEG[2]` plus
  `NEG_HI[2:0]` must be zero at `rdna4/README.md:3577` through `:3579`.
- For F16/BF16 WMMA, the manual applies `NEG[1:0]` and `NEG_HI[1:0]` to A/B
  low/high halves, and applies `{NEG_HI[2], NEG[2]}` as `{ABS, NEG}` on `SRC2`
  at `rdna4/README.md:3581`.
- For FP8/BF8 WMMA, `NEG` must be zero for A and B at
  `rdna4/README.md:3583`.

Rocjitsu evidence:

- Generated F16/BF16/FP8 dense WMMA paths pass only the C modifier derived from
  `NEG[2]` and `NEG_HI[2]`, for example at `rdna4/vop3p.cpp:3454` through
  `:3464` and `:3998` through `:4010`.
- `wmma_c_modifier` extracts only those C bits at `shared/mma_exec.h:100`
  through `:109`.
- Generated IU4 paths use `NEG[0]` and `NEG[1]` for signedness, but do not
  reject forbidden `NEG[2]` or `NEG_HI` bits, for example at
  `rdna4/vop3p.cpp:3908` through `:3921` and `:5004` through `:5021`.

Impact:

Some WMMA source modifier behavior is missing for F16/BF16 A/B halves, and
forbidden modifier encodings are accepted for IU and FP8/BF8 forms instead of
being validated or canonicalized according to the manual.

### RDNA4-RJ-061: RDNA4 SWMMAC disassembly omits sparse index-set selection and duplicates the accumulator operand

Reported by: Russell subreviewer; local audit.

Manual evidence:

- Chapter 7.12.3 makes OPSEL part of sparse index-set selection for SWMMAC at
  `rdna4/README.md:3822`.
- Table 42 says `SRC0` is A, `SRC1` is B, `SRC2` is sparse index data, and
  `VDST` is both C and D because SWMMAC reads and accumulates into it at
  `rdna4/README.md:3668` through `:3673`.

Rocjitsu evidence:

- LLVM 23 for `gfx1200` assembles
  `v_swmmac_f32_16x16x32_f16 v[0:7], v[8:11], v[12:19], v20` as
  `{0xcc504000, 0x1c521908}` and the same instruction with `index_key:1` as
  `{0xcc504800, 0x1c521908}`. LLVM disassembles the latter with the
  `index_key:1` modifier.
- A rocjitsu C API decode/disassembly spot check on the same two encodings
  emits the same text for both encodings, with no `index_key:1` suffix.
- The same spot check prints SWMMAC as
  `v_swmmac_f32_16x16x32_f16 v[0:7], v[0:7], v[8:11], v[12:19], v20`, while
  LLVM prints the four-operand syntax
  `v_swmmac_f32_16x16x32_f16 v[0:7], v[8:11], v[12:19], v20`.
- The generator has SWMMAC modifier printing only for the `gfx1250` architecture
  at `lib/python/amdisa/codegen/_generator.py:2389` through `:2392`, and emits
  `index_key:1` only for that path at `:7008` through `:7018`.
- RDNA4 SWMMAC generated classes declare only operands and `execute_impl`, with
  no `build_modifiers` overrides, for example `rdna4/vop3p.h:451` through
  `:519`.
- `VSwmmacF3216x16x32F16Vop3p` registers `vdst` in both `src_operands_` and
  `dst_operands_` at `rdna4/vop3p.cpp:4393` through `:4404`, and other RDNA4
  SWMMAC constructors follow the same generated pattern.
- Generic disassembly prints destination operands and then every source operand
  at `isa/instruction.h:246` through `:266`, so an input/output `VDST` appears
  twice unless the instruction overrides the display syntax.
- Generic disassembly calls `build_modifiers()`, whose default implementation
  emits nothing, at `isa/instruction.h:266` through `:274`; the RDNA4 VOP3P
  base stores OPSEL fields but does not override modifier printing at
  `rdna4/encodings.h:540` through `:561`.

Impact:

Disassembly of RDNA4 SWMMAC can omit the OPSEL/index-set selector even though
that selector changes the sparse matrix expansion, and can print an assembly
form that LLVM does not accept for the same encoding. Round-trip tests and
corpus inspection can miss distinct encodings or preserve misleading SWMMAC
text.

### RDNA4-RJ-062: WMMA/SWMMAC data-hazard requirements are not generally modeled

Reported by: Russell subreviewer; local audit.

Manual evidence:

- Chapter 7.12.1 says the table's `WMMA` term includes both WMMA and SWMMAC,
  and requires a `V_NOP` or independent VALU when a later WMMA/SWMMAC A/B/index
  operand overlaps a previous D matrix, at `rdna4/README.md:3585` through
  `:3599`.

Rocjitsu evidence:

- Generated RDNA4 WMMA/SWMMAC execution bodies directly call shared matrix
  helpers, for example dense F32 WMMA at `rdna4/vop3p.cpp:3454` through
  `:3464` and SWMMAC at `rdna4/vop3p.cpp:4455` through `:4462`.
- The shared matrix helpers compute and write results directly; for example
  `exec_wmma_f32_mixed` buffers results and writes them at
  `shared/mma_exec.h:1493` through `:1568`.
- The only hazard machinery found in this slice was generic DBT delay handling
  and one CDNA4-to-RDNA4 lowering wait, not a general RDNA4 WMMA/SWMMAC
  overlap rule for decoder/runtime execution.

Impact:

Rocjitsu's functional model can produce ideal matrix results for adjacent
instruction sequences that the manual says require a scheduling gap for correct
hardware behavior. There is no general diagnostic or hazard-state model for
these WMMA/SWMMAC dependencies.

### RDNA4-RJ-063: VOPD `DOT2ACC` opcodes decode but are not executable

Reported by: Faraday subreviewer; local audit.

Manual evidence:

- Chapter 7.8 lists `V_DUAL_DOT2ACC_F32_F16` and
  `V_DUAL_DOT2ACC_F32_BF16` in both OPX and OPY opcode tables at
  `rdna4/README.md:3331` and `:3341`.
- The VOPD source-cache rules say `DOT2ACC_F32_F16` and
  `DOT2ACC_F32_BF16` use the destination operand as `SRC2`, at
  `rdna4/README.md:3297`.

Rocjitsu evidence:

- RDNA4 `Vopd::op_name` names both `DOT2ACC` opcodes at
  `rdna4/vopd.cpp:87` through `:90`, and decode smoke coverage includes them
  at `tests/decode_smoke_test.cpp:306` through `:310`.
- `Vopd::execute_slot` has no `DOT2ACC` case and falls through to
  `UnimplementedInst` at `rdna4/vopd.cpp:145` through `:207`.
- VOPD source collection special-cases `FMAC`, `MOV`, and `CNDMASK`, but not
  `DOT2ACC`, so the destination-as-`SRC2` accumulator read is not modeled at
  `rdna4/vopd.cpp:290` through `:309`.

Impact:

Valid RDNA4 VOPD `DOT2ACC` instructions can decode and disassemble, but fail at
execution and expose incomplete source/dependency information.

### RDNA4-RJ-064: VOPD decoder accepts reserved or invalid opcode pairs

Reported by: Faraday subreviewer; local audit.

Manual evidence:

- Chapter 7.8 gives finite OPX opcode values `0` through `13` and OPY opcode
  values `0` through `13`, `16`, `17`, and `18` at
  `rdna4/README.md:3324` through `:3346`.

Rocjitsu evidence:

- RDNA4 `Decoder::decode` routes any instruction whose top six bits are
  `0x32` to the VOPD constructor at `rdna4/decoder.cpp:16` through `:20`.
- `Vopd::is_vopd` checks only `(word0 >> 26) == 0x32` at
  `rdna4/vopd.cpp:56` through `:59`.
- Unknown slot opcodes format as `v_dual_unknown` at `rdna4/vopd.cpp:97`
  through `:99`, and only throw if execution reaches the default case at
  `rdna4/vopd.cpp:207`.

Impact:

Reserved VOPD encodings can be represented and disassembled as VOPD instead of
being rejected at decode time, and failures are deferred until slot evaluation
rather than surfaced as decoder legality errors.

### RDNA4-RJ-065: VOPD pair legality restrictions are not enforced

Reported by: Faraday subreviewer; local audit.

Manual evidence:

- Chapter 7.8 says VOPD is legal only for wave32 and hardware does not function
  correctly unless all restrictions are met, at `rdna4/README.md:3274`.
- The restrictions include per-slot VGPR count, scalar/literal combinations,
  `VSRC1` VGPR-only use, source-cache bank and port limits, `SRC2` even/odd
  pairing, one literal, one even and one odd destination, asymmetric
  independence, no DPP, and wave32-only, at `rdna4/README.md:3283` through
  `:3304`.

Rocjitsu evidence:

- The RDNA4 VOPD constructor extracts fields, builds operands, infers the
  literal size, and initializes slot operands at `rdna4/vopd.cpp:211` through
  `:315`.
- No validation in that path checks the manual's SGPR/literal count,
  source-cache bank/port restrictions, `SRC2` parity rule, asymmetric
  dependency rule, or wave32-only rule.

Impact:

The functional model can accept and often execute deterministic software results
for encodings the manual says are illegal and may not function correctly on
hardware, which can hide compiler or assembler bugs.

### RDNA4-RJ-066: Illegal VOPD DPP selectors can be treated as ordinary sources

Reported by: Faraday subreviewer; local audit.

Manual evidence:

- Chapter 7.8 says VOPD must not use DPP at `rdna4/README.md:3303`.
- Chapter 15.3.7 says VOPD may be followed by a literal constant, but not a DPP
  control DWORD, at `rdna4/README.md:7535`.
- The source selector table still assigns values `233`, `234`, and `250` to DPP
  selectors at `rdna4/README.md:7558`.

Rocjitsu evidence:

- RDNA4 VOPD routes non-literal source-0 selectors through generic `OPR_SRC` at
  `rdna4/vopd.cpp:40` through `:48`.
- The RDNA4 `OPR_SRC` enum does not contain DPP selector names at
  `rdna4/operand_types.h:152` through `:184`.
- Generic scalar source reading treats encoding value `250` as zero/`NULL` at
  `rdna4/operand.cpp:806`, not as an illegal VOPD DPP16 selector.

Impact:

Illegal VOPD encodings using DPP selector values can be decoded as ordinary
source operands and may either mis-execute or fail later with a generic operand
error instead of a VOPD legality diagnostic.

### RDNA4-RJ-067: VOPD implicit VCC source accounting is incomplete

Reported by: Faraday subreviewer; local audit.

Manual evidence:

- Chapter 7.8 says `V_CNDMASK_B32` is the VOP2 form that uses VCC, and VCC
  counts as one SGPR read, at `rdna4/README.md:3348`.

Rocjitsu evidence:

- RDNA4 VOPD marks CNDMASK slots with `uses_vcc` at `rdna4/vopd.cpp:270`
  through `:271`, and execution reads `wf.vcc()` for those slots at
  `rdna4/vopd.cpp:187` through `:189`.
- `add_slot_sources` omits the implicit VCC operand whenever `uses_vcc` is true
  at `rdna4/vopd.cpp:300` through `:306`.
- `Operand::to_register_ref()` maps only `OPR_SRC` SGPR and VGPR ranges for
  dependency references at `rdna4/operand.cpp:648` through `:660`.

Impact:

Downstream dependency or legality logic cannot see the implicit VCC read needed
to enforce VOPD scalar-source limits or reason about CNDMASK dependencies.

### RDNA4-RJ-068: VOPD wave32-only legality is not checked on decode or DBT paths

Reported by: Faraday subreviewer; local audit.

Manual evidence:

- Chapter 7.8 says VOPD is legal only for wave32 and must not be used by
  wave64 shaders at `rdna4/README.md:3274` and `:3304`.

Rocjitsu evidence:

- `Vopd::execute_impl` iterates over `wf.wf_size()` without asserting wave32 at
  `rdna4/vopd.cpp:345` through `:355`.
- Direct RDNA4 compute-unit construction currently defaults to
  `RdnaIsaBase::WF_SIZE == 32`, but the same trait advertises
  `WF_SIZE_MAX == 64` at `shared/rdna_isa_base.h:51` through `:52`.
- DBT kernel-descriptor translation treats RDNA descriptors with
  `ENABLE_WAVEFRONT_SIZE32` clear as wave64 at
  `code/dbt/kernel_descriptor_translator.cpp:321` through `:330`.

Impact:

Decode/disassembly and DBT descriptor paths can accept VOPD inside wave64 code
objects without surfacing the manual violation; if a wave64 VOPD reaches a
non-default execution context, the loop shape is not guarded.

### RDNA4-RJ-069: VOPD `MOV` reads the ignored `VSRC1` field during execution

Reported by: local audit.

Manual evidence:

- Chapter 7.8 says `vsrc1X` and `vsrc1Y` are ignored for `V_MOV_B32` at
  `rdna4/README.md:3314` through `:3315`.

Rocjitsu evidence:

- `Vopd::execute_slot` unconditionally reads `slot.src1` before switching on
  the slot opcode at `rdna4/vopd.cpp:134` through `:136`.
- The `V_MOV_B32` case later returns only `src0` at `rdna4/vopd.cpp:185`
  through `:186`.
- Source collection does correctly omit `src1` for MOV at
  `rdna4/vopd.cpp:294` through `:296`, so the execution-time read disagrees
  with both the manual and the instruction's advertised source list.

Impact:

MOV slots can perform a spurious VGPR read that hardware ignores. This can
distort register dependency instrumentation and out-of-range/source-side-effect
modeling even when the final data result is unchanged.

### RDNA4-RJ-070: VOPD paired-exception coalescing is not modeled

Reported by: Faraday subreviewer; local audit.

Manual evidence:

- Chapter 7.8 says VOPD instruction pairs generate only a single exception if
  either or both operations raise an exception, at `rdna4/README.md:3350`.

Rocjitsu evidence:

- `Vopd::execute_slot` directly performs the individual slot arithmetic at
  `rdna4/vopd.cpp:134` through `:209`.
- `Vopd::execute_impl` evaluates X then Y and writes both results at
  `rdna4/vopd.cpp:350` through `:354`; no VOPD-level exception aggregation or
  coalescing hook was found in this path.

Impact:

If rocjitsu grows or uses FP exception/trap fidelity for VOPD, the current slot
execution structure does not encode the manual's single-exception rule.

### RDNA4-RJ-071: VOP3-family DPP decode sizes are too small

Reported by: Avicenna subreviewer; local audit.

Manual evidence:

- Chapter 15.3.8 says DPP16 can follow VOP1, VOP2, VOPC, VOP3, or VOP3P in
  place of a literal at `rdna4/README.md:7618`.
- Chapter 15.3.9 says DPP8 can likewise follow those encodings at
  `rdna4/README.md:7657`.
- The XML marks VOP3, VOP3_SDST, and VOP3P DPP16 encodings as 96-bit at
  `amdgpu_isa_rdna4.xml:8534`, `:12148`, and `:13971`.

Rocjitsu evidence:

- Generated VOP3 constructors read DPP8/DPP16 suffix words when `SRC0` is a DPP
  marker; `VMovB32Vop3` does this at `rdna4/vop3.cpp:71` through `:86`.
- The `Vop3`, `Vop3p`, and `Vop3SdstEnc` base constructors start at
  `sizeof(OpEncoding)` and add one word only for literal predicates, not DPP
  markers, at `rdna4/encodings.cpp:192` through `:200`, `:244` through `:252`,
  and `:413` through `:422`.

Impact:

Decoded streams can advance by 8 bytes for a 12-byte VOP3-family DPP
instruction, allowing the DPP suffix to be treated as the next instruction and
desynchronizing DBT or basic-block boundaries.

### RDNA4-RJ-072: DPP16 suffix ABS/NEG fields are ignored for VOP1/VOP2/VOPC

Reported by: Avicenna subreviewer; local audit.

Manual evidence:

- Chapter 7.9.1 says DPP16 has source ABS/NEG controls and that only VOP3 and
  VOP3P ignore those DPP16 fields in favor of their own modifier fields at
  `rdna4/README.md:3425` through `:3428`.
- Chapter 15.3.8 lists `SRC0_NEG`, `SRC0_ABS`, `SRC1_NEG`, and `SRC1_ABS` at
  `rdna4/README.md:7628` through `:7631`.

Rocjitsu evidence:

- RDNA4 DPP16 structs expose those bits for VOP1, VOP2, and VOPC at
  `rdna4/machine_insts.h:279` through `:295`, `:387` through `:403`, and
  `:574` through `:589`.
- Generated constructors load `vsrc0`, `dpp_ctrl`, masks, `bound_ctrl`, and
  `fi`, but not the DPP16 suffix source modifiers; see `rdna4/vop1.cpp:132`
  through `:139`, `rdna4/vop2.cpp:170` through `:177`, and
  `rdna4/vopc.cpp:47` through `:54`.
- The generator emits the same limited fixup at
  `lib/python/amdisa/codegen/_generator.py:6382` through `:6390`.

Impact:

Legal VOP1/VOP2/VOPC DPP16 source modifiers execute as unmodified source
operands.

### RDNA4-RJ-073: RDNA4 `NO DPP` opcode rules are not enforced

Reported by: Avicenna subreviewer; local audit.

Manual evidence:

- Chapter 7.9 says DPP may be used only with VOP1, VOP2, VOPC, VOP3, and VOP3P
  but not packed math ops at `rdna4/README.md:3384`.
- Table 38 rejects all 64-bit VOP1/VOP2/VOPC opcodes and multiple VOP3/VOP3P
  opcode classes, including `V_PK_*` and WMMA, at `rdna4/README.md:3390`
  through `:3417`.

Rocjitsu evidence:

- The generator's DPP support check returns true for every encoding except the
  special VOPC availability case at
  `lib/python/amdisa/codegen/_generator.py:795` through `:813`, and maps
  generic VOP bases to DPP structs at `:6341` through `:6347`.
- `V_ADD_F64` has only base and literal encodings in XML at
  `amdgpu_isa_rdna4.xml:71464` through `:71505`, but its generated VOP2
  constructor still accepts DPP8/DPP16 marker paths at `rdna4/vop2.cpp:145`
  through `:177`.
- The primary decode table routes the relevant `V_ADD_F64` entries directly to
  `decodeVAddF64Vop2` at `rdna4/decoder.cpp:6332` through `:6335`.
- A forbidden packed VOP3P example, `VPkMadI16Vop3p`, also accepts DPP8/DPP16
  marker paths at `rdna4/vop3p.cpp:24` through `:65`.

Impact:

Invalid RDNA4 DPP encodings can decode and attempt execution. Some 64-bit paths
also route through 32-bit DPP operand storage, which can fail later with a
generic operand error rather than a DPP legality diagnostic.

### RDNA4-RJ-074: VOPC DPP masked lanes preserve old compare state instead of zeroing

Reported by: Newton subreviewer; local audit.

Manual evidence:

- Chapter 7.9.1 says row/bank-masked VOPC lanes receive zero bits at
  `rdna4/README.md:3434` through `:3435`.
- It also says DPP with `V_CMP` or `V_CMPX` and `bound_ctrl=0` writes zero for
  lanes whose EXEC mask bit is zero, and `FI=1` does not turn on inactive lanes
  for `V_CMPX`, at `rdna4/README.md:3467`.

Rocjitsu evidence:

- `VCmpLtF16Vopc::execute_impl` snapshots old VCC, computes a DPP write mask,
  executes the compare, then merges old VCC bits back outside the write mask at
  `rdna4/vopc.cpp:58` through `:120`.
- `VCmpxLtI64Vopc::execute_impl` does the same for both VCC and EXEC at
  `rdna4/vopc.cpp:17100` through `:17176`.
- The generator emits this old-bit merge at
  `lib/python/amdisa/codegen/_generator.py:6808` through `:6823`, and
  generator tests pin the CMPX EXEC preservation at
  `lib/python/amdisa/tests/test_generator_profile_gates.py:1578` through
  `:1599`.

Impact:

Masked or invalid compare-DPP lanes can keep stale VCC/EXEC bits where the
manual says the result bit should be zero.

### RDNA4-RJ-075: RDNA4 reserved DPP_CTRL values execute as legacy behavior or identity

Reported by: Avicenna subreviewer; local audit.

Manual evidence:

- Chapter 7.9.1 lists RDNA4 DPP16 controls through `DPP_ROW_XMASK{0:15}` and
  marks `DPP_UNUSED` as reserved at `rdna4/README.md:3444` through `:3454`.
- Chapter 15.3.8 repeats that enumeration at `rdna4/README.md:7639` through
  `:7649`.

Rocjitsu evidence:

- The shared DPP helper includes legacy wave shifts, wave rotates, and row
  broadcasts not listed in the RDNA4 manual at `shared/dpp_sdwa_ops.h:59`
  through `:66`, with behavior implemented at `:125` through `:178`.
- Unknown controls fall back to identity at `shared/dpp_sdwa_ops.h:198` through
  `:199`.
- Existing helper tests codify wave shift/rotate and row broadcast behavior at
  `tests/shared_infra_test.cpp:1569` through `:1598` and `:1635` through
  `:1645`.

Impact:

Malformed RDNA4 DPP controls can silently produce deterministic non-RDNA4
behavior instead of being rejected or treated as undefined.

### RDNA4-RJ-076: DPP disassembly drops all cross-lane detail

Reported by: Avicenna subreviewer; local audit.

Manual evidence:

- Chapter 7.9 says DPP is indicated by `DPP8`, `DPP8FI`, or `DPP16` in `SRC0`
  and the actual VGPR source comes from the DPP DWORD at
  `rdna4/README.md:3374`.
- Chapter 7.9.2 says the DPP8 source field supplies the real VGPR after the
  source slot is filled by the DPP marker at `rdna4/README.md:3479` through
  `:3480`.

Rocjitsu evidence:

- Generated constructors replace the DPP marker source with the real VGPR
  operand; `VMovB32Vop1` does this at `rdna4/vop1.cpp:124` through `:139`.
- `Instruction::disassemble()` prints operand names and then calls
  `build_modifiers()` at `isa/instruction.h:239` through `:255`.
- RDNA4 VOP1, VOP2, VOP3, and VOP3P bases store DPP fields but do not override
  `build_modifiers()` at `rdna4/encodings.h:430` through `:560`.

Impact:

DPP and non-DPP instructions can disassemble identically; DPP8 lane selectors,
DPP16 `DPP_CTRL`, masks, `BC`, and `FI` are invisible in traces.

### RDNA4-RJ-077: DPP-specific OPSEL legality is not enforced

Reported by: local audit.

Manual evidence:

- Chapter 7.9.1 says DPP with VOP3/VOP3P requires OPSEL to keep low results
  using low inputs and high results using high inputs at `rdna4/README.md:3423`.
- Chapter 7.9.2 repeats the rule for DPP8 at `rdna4/README.md:3473`.

Rocjitsu evidence:

- Allowed VOP3P DPP instructions such as `VFmaMixF32Vop3p` accept DPP8 and
  DPP16 marker paths at `rdna4/vop3p.cpp:2691` through `:2707`.
- The execution body then consumes `inst_.opsel`, `inst_.opsel_hi`, and
  `inst_.opsel_hi_2` directly for source half selection at
  `rdna4/vop3p.cpp:2732` through `:2746`.
- No constructor or execute-path validation rejects DPP encodings whose OPSEL
  fields cross low/high halves.

Impact:

Rocjitsu can execute illegal DPP half-selection combinations as if they were
ordinary VOP3/VOP3P selectors, hiding assembler or compiler legality bugs.

### RDNA4-RJ-078: DBT is not DPP-format aware

Reported by: Avicenna subreviewer; local audit.

Manual evidence:

- RDNA4 has split DPP16 and DPP8 extension formats at `rdna4/README.md:7618`
  through `:7671`.

Rocjitsu evidence:

- The generated CDNA4-to-RDNA4 translator decodes base VOP fields only for
  VOP1/VOP2/VOPC at `code/dbt/generated/encoding_cdna4_to_rdna4.h:71`
  through `:87`, then encodes base VOP forms at `:323` through `:335`.
- The translation switch passes only base words into those encoders at
  `code/dbt/generated/encoding_cdna4_to_rdna4.h:547` through `:555`.
- The runtime translator appends one trailing word using a literal-size
  heuristic at `code/dbt/binary_translator.cpp:1241` through `:1253`.

Impact:

DPP suffix fields are not translated, checked, or reliably preserved across the
CDNA4-to-RDNA4 DBT path. The VOP3-family decode-size gap makes those cases even
more vulnerable to stream desynchronization.

### RDNA4-RJ-079: Pseudo-scalar `V_S_*` instructions skip execution when `EXEC==0`

Reported by: Bohr subreviewer; local audit.

Manual evidence:

- Chapter 7.10 says pseudo-scalar transcendental instructions operate on a
  single lane with SGPR source/destination operands at `rdna4/README.md:3492`.
- The notes state that `EXEC` is ignored and these instructions execute even
  when `EXEC==0`, at `rdna4/README.md:3507`.

Rocjitsu evidence:

- The shared F32 helpers read `wf.exec()` and guard the scalar write with
  `if (exec != 0)`, for example `execute_v_s_exp_f32_vop3` at
  `shared/execute_shared.h:17979` through `:18005`; the same pattern appears
  for log, rcp, rsq, and sqrt at `:18009`, `:18039`, `:18069`, and `:18099`.
- The generated RDNA4 F16 bodies also guard their scalar writes with
  `if (exec != 0)`, for example `VSExpF16Vop3::execute_impl` at
  `rdna4/vop3.cpp:17861` through `:17905`, with the same pattern at
  `:18019`, `:18177`, `:18335`, and `:18495`.

Impact:

When `EXEC` is zero, rocjitsu leaves the destination unchanged for all ten
`V_S_*` pseudo-scalar transcendental ops, while RDNA4 specifies that they still
execute.

### RDNA4-RJ-080: Pseudo-scalar `VCC` destination and `OPSEL[3]` legality are not enforced

Reported by: Bohr subreviewer; local audit.

Manual evidence:

- Chapter 7.10 says `VCC` may not be used as a destination and `OPSEL[3]` must
  be zero, at `rdna4/README.md:3508` through `:3509`.
- For F16 pseudo-scalar results, the manual says half-SGPRs are not supported
  and the full 32-bit destination is written with the upper half zeroed at
  `rdna4/README.md:3504`.

Rocjitsu evidence:

- Generated constructors use the broad `OPR_SREG` destination class, for
  example `VSExpF32Vop3` at `rdna4/vop3.cpp:17763` through `:17766`.
- RDNA4 `OPR_SREG` includes `VCC_LO` and `VCC_HI` selectors in generated
  operand metadata at `rdna4/operand_types.h:209` through `:216`, and
  disassembly prints those names at `rdna4/operand.cpp:363` through `:375`.
- Scalar writes route selector 106 and 107 into `wf.set_vcc(...)` at
  `rdna4/operand.cpp:916` through `:938`.
- The F16 bodies read only source `OPSEL[0]`, for example at
  `rdna4/vop3.cpp:17878` through `:17886`, and no constructor or execute path
  rejects `OPSEL[3] == 1`.

Impact:

Rocjitsu accepts and can execute pseudo-scalar forms that the manual marks
illegal or undefined. For F16, an invalid high-half destination selector gets a
deterministic lower-half scalar write instead of being rejected or surfaced.

### RDNA4-RJ-081: `V_MOVRELD_B32` and `V_MOVRELS_B32` use signed `M0` and throw on indexed OOR

Reported by: Bohr subreviewer; local audit.

Manual evidence:

- Chapter 7.11 says VGPR indexing uses `M0` and that indices are unsigned at
  `rdna4/README.md:3514`.
- The table defines `V_MOVRELD_B32` and `V_MOVRELS_B32` with `M0[31:0]` at
  `rdna4/README.md:3520` through `:3521`.
- Chapter 3.3.2.2 says a destination VGPR out of range treats the instruction
  as a NOP, while a source VGPR out of range in a VALU instruction acts as
  source VGPR0, at `rdna4/README.md:850` through `:855`.

Rocjitsu evidence:

- RDNA4 VOP1 `VMovreldB32Vop1` computes the relative destination with
  `static_cast<int32_t>(wf.m0())`, then throws `UnimplementedInst` for negative
  or out-of-allocation values at `rdna4/vop1.cpp:6778` through `:6783`.
- RDNA4 VOP1 `VMovrelsB32Vop1` uses the same signed cast and throws on source
  OOR at `rdna4/vop1.cpp:6908` through `:6913`.
- The VOP3 forms repeat the same behavior at `rdna4/vop3.cpp:3548` through
  `:3553` and `:3625` through `:3630`.
- The generator source emits the same signed pattern from
  `lib/python/amdisa/codegen/execute/vector_special.py:84` through `:102`.

Impact:

Large unsigned `M0` values can be treated as negative, and architectural NOP or
VGPR0 fallback cases become exceptions in rocjitsu.

### RDNA4-RJ-082: `V_MOVRELSD*` and `V_SWAPREL_B32` decode but are not executable

Reported by: Bohr subreviewer; local audit.

Manual evidence:

- Chapter 7.11 defines `V_MOVRELSD_B32`, `V_MOVRELSD_2_B32`, and
  `V_SWAPREL_B32`, including their source/destination index formulas and swap
  order, at `rdna4/README.md:3522` through `:3524`.
- The instruction opcode tables list these opcodes in VOP1/VOP3 slots for
  MOVRELSD forms and VOP1 slot 104 for `V_SWAPREL_B32`, at
  `rdna4/README.md:6857`, `:6887`, and `:7196`.

Rocjitsu evidence:

- Generated VOP1 `VMovrelsdB32Vop1` and `VMovrelsd2B32Vop1` constructors
  decode the instructions, but their execute bodies throw `UnimplementedInst`
  at `rdna4/vop1.cpp:6980` through `:6982` and `:7017` through `:7019`.
- Generated VOP3 `VMovrelsdB32Vop3` and `VMovrelsd2B32Vop3` likewise throw at
  `rdna4/vop3.cpp:3686` through `:3688` and `:3723` through `:3725`.
- Generated VOP1 `VSwaprelB32Vop1` decodes both input/output operands at
  `rdna4/vop1.cpp:10549` through `:10559`, but its execute body throws at
  `:10583` through `:10585`.
- The generator semantic map classifies `V_SWAPREL` as `nop` at
  `lib/python/amdisa/semantics.py:768`.

Impact:

Three of the five Chapter 7.11 relative VGPR-indexing operations are
unavailable in functional emulation even though rocjitsu can decode them.

### RDNA4-RJ-083: Pseudo-scalar `V_S_*` constructors accept DPP VGPR sources

Reported by: Bohr subreviewer; local audit.

Manual evidence:

- Chapter 7.10 says pseudo-scalar transcendental ops operate on a single lane
  of data where both the source and destination are SGPRs, at
  `rdna4/README.md:3492`.

Rocjitsu evidence:

- Generated `VSExpF32Vop3` starts with `src0` as `OPR_SSRC`, but the same
  constructor rewrites DPP8 and DPP16 source-marker forms to `OPR_VGPR` at
  `rdna4/vop3.cpp:17763` through `:17786`.
- Generated `VSExpF16Vop3` similarly rewrites DPP8 and DPP16 source-marker
  forms to `OPR_VGPR` at `rdna4/vop3.cpp:17842` through `:17852`.
- This is the pseudo-scalar-specific instance of the broader RDNA4 DPP
  allow-list enforcement issue recorded in `RDNA4-RJ-073`.

Impact:

Rocjitsu can decode pseudo-scalar transcendental forms whose source operand is
taken from a DPP VGPR suffix instead of an SGPR source, obscuring assembler or
decoder legality bugs for these scalar-source-only operations.

### RDNA4-RJ-084: VOP3-family modifier and `OPSEL` legality is not enforced generically

Reported by: Averroes subreviewer; local audit.

Manual evidence:

- Chapter 7.1 says `NEG`, `ABS`, and `OMOD` are floating-point only at
  `rdna4/README.md:2631` through `:2636`.
- The field table repeats that `OMOD`, `NEG`, and `ABS` apply to floating
  results/inputs and says `OPSEL` may only be used for 16-bit
  operands/results and must be zero otherwise, at `rdna4/README.md:2666`
  through `:2673`.
- Chapter 7.2.2.1 lists the VOP instructions where input modifiers are not
  supported, and Chapter 7.2.2.3 lists the limited instruction set where
  `OPSEL` is meaningful, at `rdna4/README.md:2764` through `:2830`.

Rocjitsu evidence:

- RDNA4 stores the raw VOP3 fields in `Vop3MachineInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h:66`
  through `:78`, and VOP3P selector/modifier fields at `:80` through `:94`.
- Generic VOP3/VOP3P/VOP3SD constructors copy the instruction bits and set
  size/literal state without opcode-level legality checks, for example
  `rdna4/encodings.cpp:192` through `:201` and `:244` through `:253`.
- A concrete 32-bit integer example, `VAddNcU32Vop3`, constructs ordinary
  operands without rejecting nonzero `opsel`, `abs`, `neg`, or `omod`, at
  `rdna4/vop3.cpp:8448` through `:8484`; its execute helper uses only the
  arithmetic operands at `shared/execute_shared.h:3203` through `:3211`.

Impact:

Illegal or nonsensical VOP3-family selector/modifier encodings can decode and
execute as if the ignored bits were not present. That can hide assembler,
compiler, and decoder bugs where nonzero reserved/illegal modifier bits should
be rejected or at least reported.

### RDNA4-RJ-085: Generic VOP3-family modifier fields are hidden in disassembly

Reported by: Averroes subreviewer; local audit.

Manual evidence:

- Chapter 7.1 defines visible VOP3/VOP3SD/VOP3P modifier fields in the format
  table and field tables at `rdna4/README.md:2622` through `:2624` and
  `:2666` through `:2673`.

Rocjitsu evidence:

- The generic disassembly path prints operands and then calls
  `build_modifiers()`, at `lib/rocjitsu/src/rocjitsu/isa/instruction.h:239`
  through `:255`.
- The base `build_modifiers()` implementation is empty at
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:271` through `:274`.
- RDNA4 `Vop3`, `Vop3p`, and `Vop3SdstEnc` do not override
  `build_modifiers()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/encodings.h:517`
  through `:560` and `:638` through `:658`.

Impact:

Disassembly and trace output can omit semantically relevant `abs`, `neg`,
`omod`, `clamp`, `opsel`, `opsel_hi`, and `neg_hi` state. This generic 7.1
visibility gap is broader than the narrower DOT, SWMMAC, and DPP disassembly
issues recorded elsewhere.

### RDNA4-RJ-086: Forbidden scalar destination selectors can mutate special registers

Reported by: Averroes subreviewer; local audit.

Manual evidence:

- Chapter 7.1 says `VDST` used as an SGPR result for `V_READLANE` and `V_CMP`
  cannot be `M0` or `EXEC`, at `rdna4/README.md:2664`.
- The same table says VOP3SD `SDST` cannot be `M0` or `EXEC`, and supports
  `NULL`, at `rdna4/README.md:2665`.

Rocjitsu evidence:

- The VOP3SD machine-instruction struct stores raw `sdst` as a 7-bit field at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h:432`
  through `:442`.
- Generated VOP3SD constructors bind that raw value as `OPR_SREG` without a
  reject path, for example `VAddCoCiU32Vop3SdstEnc` at
  `rdna4/vop3.cpp:37513` through `:37520`.
- `VReadlaneB32Vop3` and VOP3 `V_CMP*` paths similarly bind raw `vdst` as
  `OPR_SREG`, for example `rdna4/vop3.cpp:21851` through `:21856` and
  `:23076` through `:23081`.
- Scalar writes ignore the operand enum and route raw selector values `125`,
  `126`, and `127` to `M0`, `EXEC_LO`, and `EXEC_HI`/raw `EXEC`, at
  `rdna4/operand.cpp:916` through `:989`.
- The carry-out helpers write through those scalar destinations, for example
  `shared/execute_shared.h:2865` through `:2885` and `:2913` through `:2936`,
  and VOP3 compare writes use the same scalar write path at
  `rdna4/vop3.cpp:23170` through `:23173`.

Impact:

Raw encodings that the manual forbids can be constructed and can update special
wave state instead of being rejected. This can corrupt `M0` or `EXEC` in
negative tests and masks missing legality checks for scalar-destination VOP3
forms.

### RDNA4-RJ-087: VOP3 floating-point clamp preserves negative zero

Reported by: local audit.

Manual evidence:

- Chapter 7.1 defines `CM`/`CLAMP` for floating-point arithmetic as clamping to
  `[0, 1.0]` and says `-0` is clamped to `+0`, at `rdna4/README.md:2672`.

Rocjitsu evidence:

- Scalar generated VOP3 floating-point tails commonly use `std::clamp`, for
  example `execute_v_add_f32_vop3` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:3045`
  through `:3055`.
- Generated RDNA4 VOP3/VOP3P bodies repeat the same pattern, for example
  `rdna4/vop3p.cpp:1733` through `:1736`.
- The SIMD helper uses ordered comparisons `v < 0` and `v > 1` at
  `shared/simd_glue.h:371` through `:388`, matching `std::clamp` saturation
  but not canonicalizing exact `-0.0` to `+0.0`.

Impact:

For a clamped floating-point operation whose computed result is exactly
negative zero, rocjitsu can preserve the sign bit where RDNA4 specifies a
positive-zero result.

### RDNA4-RJ-088: `READLANE` and `WRITELANE` use unmasked lane selectors

Reported by: local audit.

Manual evidence:

- Chapter 7.2.1 says `V_READLANE` ignores upper bits of the lane number so the
  selector is limited to `0..31` for wave32 and `0..63` for wave64, at
  `rdna4/README.md:2747`.
- The instruction definitions make the same low-bit selection explicit for both
  `V_READLANE_B32` and `V_WRITELANE_B32`, at `rdna4/README.md:22496` through
  `:22531`.

Rocjitsu evidence:

- `VReadlaneB32Vop3::execute_impl` reads `src1` with `read_scalar` and passes
  the raw value directly to `read_lane(src0, lane)`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vop3.cpp:21907` through
  `:21908`.
- `VWritelaneB32Vop3::execute_impl` likewise passes the raw scalar lane to
  `write_lane(vdst, lane, val)`, at `rdna4/vop3.cpp:21982` through `:21984`.
- The RDNA4 operand and register-access paths read/write the requested lane
  directly and do not mask the lane number first, at `rdna4/operand.cpp:1088`
  through `:1100` and `vm/amdgpu/register_access.h:600` through `:616`.

Impact:

Lane selectors with high bits set can read or write an out-of-range software
lane instead of aliasing to the architecturally selected low lane. DPP delegates
return zero for out-of-range lane numbers, so DPP-form readlane can diverge even
without invoking undefined host memory behavior.

### RDNA4-RJ-089: VOP3 clamp leaves NaN results unchanged

Reported by: local audit.

Manual evidence:

- Chapter 7.2.3.1 says that when `CLAMP==1`, any NaN result is clamped to zero
  and exceptions are reported before clamp, at `rdna4/README.md:2864` through
  `:2868`.

Rocjitsu evidence:

- Scalar generated floating-point VOP3 paths use `std::clamp`, for example
  `execute_v_add_f32_vop3` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:3053`
  through `:3055`.
- The SIMD helper explicitly documents that ordered compares make NaN pass
  through unchanged, at `shared/simd_glue.h:357` through `:388`.
- Generated VOP3P DOT2 paths also use `std::clamp` after computing the result,
  for example `execute_v_dot2_f32_f16_vop3p` at
  `shared/execute_shared.h:10475` through `:10480`.

Impact:

Clamped FP operations that produce NaN can leave a NaN in the destination rather
than writing zero as RDNA4 specifies. This is broader than `RDNA4-RJ-087`, which
tracks the separate negative-zero canonicalization issue.

### RDNA4-RJ-090: DOT2 inline constants are not replicated for packed halves

Reported by: local audit.

Manual evidence:

- Chapter 7.2.2 says `V_DOT2_F32_BF16`, `V_DOT2_F32_F16`,
  `V_DOT2_F16_F16`, and `V_DOT2_BF16_BF16` replicate inline constants for
  sources 0 and 1 into bits `[31:16]`, and source 2 uses `OPSEL` to select
  replicate-versus-zero behavior, at `rdna4/README.md:2760` through `:2763`.

Rocjitsu evidence:

- RDNA4 scalar inline constants resolve to ordinary 32-bit F32 encodings in
  `resolve_src_scalar`, while the 16-bit values are exposed only through
  `resolve_src_scalar16`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/operand.cpp:778` through
  `:843`.
- `VDot2F32F16Vop3p` constructs `src0`, `src1`, and `src2` as 32-bit
  `OPR_SRC` operands and rewrites literal selector `255` to 32-bit `OPR_SIMM32`
  without DOT2-specific inline replication, at `rdna4/vop3p.cpp:1646` through
  `:1670`.
- `execute_v_dot2_f32_f16_vop3p` reads `raw0` and `raw1` as 32-bit lane values
  and slices the low/high halves directly according to `op_sel` and
  `op_sel_hi`, at `shared/execute_shared.h:10457` through `:10466`.
- `execute_v_dot2_f32_bf16_vop3p` follows the same raw read and half-slice
  pattern, at `shared/execute_shared.h:10421` through `:10430`.
- The VOP3 `V_DOT2_F16_F16` and `V_DOT2_BF16_BF16` bodies do the same raw
  32-bit `read_lane` for `src0/src1` and then split low/high halves in the
  executor, at `rdna4/vop3.cpp:16631` through `:16642` and `:16764` through
  `:16775`.

Impact:

For inline constants on DOT2 packed sources, rocjitsu can feed the high half
from the high 16 bits of the ordinary 32-bit inline encoding instead of the
manual's replicated half value. Source-2 OPSEL replication/zero behavior is also
not modeled as an instruction-specific rule.

### RDNA4-RJ-091: `PERMLANE`-after-`CMPX` stream hazard is not tracked

Reported by: local audit.

Manual evidence:

- Chapter 7.2.8 says `V_PERMLANE*` may not occur immediately after `V_CMPX`;
  another VALU instruction such as `V_NOP` must be inserted, at
  `rdna4/README.md:2949` through `:2951`.

Rocjitsu evidence:

- Generated `VPermlane16B32Vop3`, `VPermlanex16B32Vop3`,
  `VPermlane16VarB32Vop3`, and `VPermlanex16VarB32Vop3` constructors and
  execute paths perform only local operand setup/execution, for example
  `rdna4/vop3.cpp:15929` through `:16014` and `:19622` through `:19792`.
- No searched RDNA4 VOP execution path records the previously executed VALU
  opcode or checks whether a `V_PERMLANE*` immediately follows a `V_CMPX`.

Impact:

Rocjitsu can execute an instruction stream that RDNA4 requires software to
avoid or pad, so decoder/runtime tests will not catch missing compiler or
assembler scheduling around this hazard.

### RDNA4-RJ-092: VOP3-family source-combination legality is not validated

Reported by: Parfit subreviewer; local audit.

Manual evidence:

- Chapter 7.2.2.2 says not every source combination expressible in the
  instruction format is legal, defines scalar-value accounting, caps each VALU
  instruction at two scalar values, permits only one literal constant, forbids
  literals with DPP, counts implicit `VCC` uses for selected opcodes, and gives
  same-SGPR/same-literal accounting rules, at `rdna4/README.md:2783` through
  `:2801`.

Rocjitsu evidence:

- The generic RDNA4 `Vop3`, `Vop3p`, and `Vop3SdstEnc` constructors increase
  instruction size for all detected literal-source combinations, including
  `has_lit_0_has_lit_1`, `has_lit_0_has_lit_2`, `has_lit_1_has_lit_2`, and
  `has_lit_0_has_lit_1_has_lit_2`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/encodings.cpp:192` through
  `:294` and `:413` through `:466`.
- Representative generated constructors bind all source fields independently
  as source operands and rewrite each raw selector `255` to the same extension
  word, for example `VFmaF32Vop3` at `rdna4/vop3.cpp:10666` through
  `:10689` and `VAddCoCiU32Vop3SdstEnc` at `rdna4/vop3.cpp:37513` through
  `:37536`.
- `VDivFmasF32Vop3` has the implicit-`VCC` source case called out by the
  manual, but the generated constructor still accepts three generic `OPR_SRC`
  fields without scalar-count validation at `rdna4/vop3.cpp:13356` through
  `:13380`.

Impact:

Rocjitsu can decode and execute source tuples that RDNA4 says are illegal,
including multiple literal selectors or too many scalar values once implicit
`VCC` is counted. This is broader than the existing DPP-specific legality
entries because it applies to ordinary VOP3, VOP3P, and VOP3SD forms.

### RDNA4-RJ-093: Wave64 SGPR mask/carry restrictions are not validated

Reported by: Parfit subreviewer; local audit.

Manual evidence:

- Chapter 7.2.2.2 says the source rules must be met for both wave64 passes and
  mask sources may not be shared with another SGPR source, while Chapter
  7.2.3.2 forbids wave64 VALU instructions from reading and writing the same
  SGPR value, at `rdna4/README.md:2801` and `:2888` through `:2890`.
- Chapter 7.2.6 says wave64 mask/carry inputs read two consecutive SGPRs, with
  the second pass using the incremented SGPR address, at `rdna4/README.md:2937`
  through `:2939`.

Rocjitsu evidence:

- RDNA4 wave64 execution is already not selectable, as tracked by
  `RDNA4-RJ-016`, but decode/construction still accepts the affected VALU
  forms without same-SGPR or mask-sharing checks.
- `VAddCoCiU32Vop3SdstEnc` binds `sdst` as a 64-bit `OPR_SREG` destination and
  `src2` as a 64-bit `OPR_SREG` carry input at `rdna4/vop3.cpp:37513` through
  `:37520`.
- `VCndmaskB32Vop3` binds the select mask as a 64-bit scalar source at
  `rdna4/vop3.cpp:6395` through `:6401`.
- The shared carry and conditional-mask executors simply read or write the
  scalar64 operands at `shared/execute_shared.h:2865` through `:2885` and
  `:8272` through `:8286`; the RDNA4 decoder is per-instruction/stateless at
  `rdna4/decoder.cpp:16` through `:20`.

Impact:

Even before wave64 execution is fully modeled, rocjitsu lacks validation for
the Chapter 7.2 SGPR hazards that would make some mask/carry encodings illegal
in wave64 code. Decoder or trace tests can therefore accept streams that a
wave64-aware validator should reject.

### RDNA4-RJ-094: F64 inline constant selector `248` is one ULP high

Reported by: Parfit subreviewer.

Manual evidence:

- The RDNA4 inline-constant table lists selector `248` as `1/(2*PI)` and gives
  the 64-bit value `0x3fc45f306dc9c882`, at `rdna4/README.md:2713` through
  `:2716`.

Rocjitsu evidence:

- The checked-in RDNA4 XML also describes selector `248` for all 64-bit formats
  as `0x3fc45f30_6dc9c882`, at `amdgpu_isa_rdna4.xml:185320`.
- `resolve_src_scalar64` returns `0x3FC45F306DC9C883ULL` for selector `248`,
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/operand.cpp:903`
  through `:904`.
- The 32-bit and 16-bit selector values match the documented table at
  `rdna4/operand.cpp:794` through `:795` and `:839` through `:840`.

Impact:

F64 consumers of the inline `1/(2*PI)` selector get the next representable
double above the value documented by both the RDNA4 manual and XML, while F16
and F32 consumers use the expected encodings.

### RDNA4-RJ-095: Several Chapter 7.3 VALU instructions decode but always throw

Reported by: Turing subreviewer; local audit.

Manual evidence:

- Chapter 7.3 lists `V_PK_FMAC_F16`, `V_TRIG_PREOP_F64`,
  `V_MQSAD_PK_U16_U8`, `V_MQSAD_U32_U8`, `V_MULLIT_F32`,
  `V_QSAD_PK_U16_U8`, and `V_SAT_PK_U8_I16` in the non-VOP3P VALU inventory,
  at `rdna4/README.md:3002`, `:3017`, `:3037`, `:3043`, `:3045`, and `:3050`.
- The detailed opcode tables assign these instructions concrete VOP1, VOP2, or
  VOP3 opcode slots, for example `V_PK_FMAC_F16` at `rdna4/README.md:6789`,
  `V_SAT_PK_U8_I16` at `:6881` and `:7220`, `V_MULLIT_F32` at `:7247`,
  the QSAD/MQSAD forms at `:7280` through `:7282`, and
  `V_TRIG_PREOP_F64` at `:7368`.

Rocjitsu evidence:

- The checked-in RDNA4 XML contains these affected instructions as concrete
  VALU entries, for example `V_SAT_PK_U8_I16` at
  `amdgpu_isa_rdna4.xml:69450`, `V_PK_FMAC_F16` at `:83066`,
  `V_MULLIT_F32` at `:87963`, QSAD/MQSAD at `:96615` through `:97143`, and
  `V_TRIG_PREOP_F64` at `:116954`.
- Generated baseline encodings exist for the affected names in
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/test_encodings.h:378`,
  `:598`, `:675`, `:741`, `:768` through `:770`, and `:850`.
- `VPkFmacF16Vop2::execute_impl` unconditionally throws
  `util::UnimplementedInst` at `rdna4/vop2.cpp:6499` through `:6501`.
- `VSatPkU8I16Vop1::execute_impl` and `VSatPkU8I16Vop3::execute_impl`
  unconditionally throw at `rdna4/vop1.cpp:9822` through `:9824` and
  `rdna4/vop3.cpp:5653` through `:5655`.
- `VMullitF32Vop3::execute_impl`, `VQsadPkU16U8Vop3::execute_impl`,
  `VMqsadPkU16U8Vop3::execute_impl`, `VMqsadU32U8Vop3::execute_impl`, and
  `VTrigPreopF64Vop3::execute_impl` likewise throw at
  `rdna4/vop3.cpp:11162` through `:11164`, `:13642` through `:13742`, and
  `:21253` through `:21255`.
- The separate `V_MOVRELSD*` and `V_SWAPREL_B32` unimplemented bodies are
  already tracked by `RDNA4-RJ-082`.

Impact:

These inventory-listed VALU instructions can decode and disassemble, but any
functional execution path reaches an unconditional unimplemented exception.
Baseline encoding tests therefore prove table presence, not executable
semantics, for this subset of Chapter 7.3.

### RDNA4-RJ-096: `VINTERP` F16 instructions decode but execute as no-ops

Reported by: local audit.

Manual evidence:

- Chapter 7.4 groups VINTERP with VOP3 and VOP3P as an OPSEL-based 16-bit
  VGPR addressing form, with `SRC/DST[7:0]` as the 32-bit VGPR address and
  OPSEL selecting the high or low half at `rdna4/README.md:3095` through
  `:3097`.
- The detailed VINTERP instruction definitions give real F16 interpolation
  formulas and say OPSEL selects which half of `S0` and `S2` to read for
  `V_INTERP_P10_F16_F32` at `rdna4/README.md:25108` through `:25127`.
- `V_INTERP_P2_F16_F32` says OPSEL selects which half of `S0` to read and
  which half of `D0` to write at `rdna4/README.md:25129` through `:25149`;
  the RTZ variants repeat those source and destination selector rules at
  `rdna4/README.md:25151` through `:25190`.

Rocjitsu evidence:

- RDNA4 generates concrete decode/operand classes for the four F16 VINTERP
  instructions, with 16-bit `src0`/`src2` or 16-bit `vdst` as appropriate, at
  `rdna4/vinterp.cpp:57` through `:82` and `:101` through `:126`.
- Every generated F16 VINTERP `execute_impl` body is a compute-simulation
  no-op: `rdna4/vinterp.cpp:72` through `:74`, `:97` through `:99`,
  `:116` through `:118`, and `:141` through `:143`.

Impact:

Decoded VINTERP F16 instructions leave their destination unchanged instead of
performing the fixed-DPP interpolation, true16 source half selection, RTZ
rounding where applicable, and OPSEL-controlled destination half update.

### RDNA4-RJ-097: OPSEL-based true16 destinations `128..255` alias to `v0..v127`

Reported by: Plato subreviewer; local audit.

Manual evidence:

- Chapter 7.4 says VOP3, VOP3P, and VINTERP true16 encodings use
  `SRC/DST[7:0]` as the 32-bit VGPR address and OPSEL as the high/low half
  selector, so a wave can address 512 16-bit VGPRs at `rdna4/README.md:3095`
  through `:3097`.

Rocjitsu evidence:

- RDNA4 `packed_16bit_vgpr_dst` unconditionally treats any 16-bit
  `OPR_VGPR` destination value `128..255` as the high half of `v0..v127` at
  `rdna4/operand.cpp:50` through `:58`.
- `Operand::to_register_ref()` applies that compact-E32 destination collapse
  before normal VGPR resolution at `rdna4/operand.cpp:590` through `:599`.
- `write_vop3_true16_dst()` then uses the collapsed register reference and
  applies `OPSEL[3]` only as the destination half selector at
  `shared/simd_glue.h:297` through `:317`.
- Representative affected generated paths construct 16-bit `OPR_VGPR`
  destinations from the raw 8-bit VOP3/VOP3P/VINTERP `VDST` fields:
  `rdna4/vop3.cpp:9083` through `:9086`, `rdna4/vop3p.cpp:2780` through
  `:2784`, and `rdna4/vinterp.cpp:76` through `:82`.

Impact:

For OPSEL-based true16 instructions, raw `VDST=128` with `OPSEL[3]=0` should
write `v128.l`, but rocjitsu resolves the destination as `v0.l`. The same
collapse affects high-half writes and liveness register references for raw
destinations `128..255`.

### RDNA4-RJ-098: Packed F16 inline constants use FP32 bit patterns

Reported by: Fermat subreviewer; local audit.

Manual evidence:

- Chapter 7.7.1 says packed F16 inline constants are `{16'h0, const}` at
  `rdna4/README.md:3242` through `:3249`.
- Chapter 7.7.2 says inline constants used with float 16-bit sources produce an
  F16 constant value in only the low 16 bits, and OPSEL should select that low
  input for both halves when the same constant is needed in both results, at
  `rdna4/README.md:3251` through `:3257`.

Rocjitsu evidence:

- Representative generated packed F16 constructors, such as
  `VPkAddF16Vop3p`, construct sources as 32-bit `OPR_SRC` operands and rewrite
  literal selector `255` as a 32-bit `OPR_SIMM32` at
  `rdna4/vop3p.cpp:1440` through `:1458`.
- The execution body reads the raw 32-bit source values, splits the selected
  low/high 16 bits, and converts those halves with `util::f16_to_f32` at
  `rdna4/vop3p.cpp:1500` through `:1509`.
- `Operand::read_lane()` only calls the 16-bit inline constant table when
  `size_bits_ == 16`; otherwise it calls the 32-bit scalar table at
  `rdna4/operand.cpp:1102` through `:1106`.
- The 32-bit scalar table maps inline selector `242` to FP32 `1.0`
  (`0x3f800000`) at `rdna4/operand.cpp:778` through `:783`, while the
  16-bit table maps the same selector to F16 `1.0` (`0x3c00`) at
  `rdna4/operand.cpp:821` through `:828`.

Impact:

For `V_PK_*_F16` inline constants, rocjitsu exposes halves of the FP32 bit
pattern instead of `{16'h0, f16_const}`. For example, inline `1.0` presents
low half `0x0000` and high half `0x3f80` from `0x3f800000`, rather than low
half `0x3c00` with high half zero, so both default and OPSEL-replicated inline
constant cases can produce incorrect results.

### RDNA4-RJ-099: Ordinary packed I16/U16/F16 VOP3P arithmetic ignores `CLAMP`

Reported by: Fermat subreviewer; local audit.

Manual/XML evidence:

- Chapter 7.7 says `CM` clamps floating-point arithmetic results to `[0, 1.0]`
  with `-0` clamped to `+0`, signed integer arithmetic to
  `[INT_MIN, INT_MAX]`, and unsigned integer arithmetic to `[0, UINT_MAX]` at
  `rdna4/README.md:3203` through `:3208`.
- The XML generic VOP3P `CLAMP` field repeats the same float/signed/unsigned
  clamp description at `amdgpu_isa_rdna4.xml:13839`.

Rocjitsu evidence:

- The packed generator's F16 binary path computes `rlo`/`rhi` and writes them
  directly without any `inst_.clamp` branch at
  `lib/python/amdisa/codegen/execute/packed.py:71` through `:100`.
- The generated integer ternary path likewise writes wrapped 16-bit results
  directly at `lib/python/amdisa/codegen/execute/packed.py:313` through
  `:355`.
- Representative generated bodies, such as `VPkMadI16Vop3p` and
  `VPkAddF16Vop3p`, compute and write results without checking `inst_.clamp`
  at `rdna4/vop3p.cpp:93` through `:108` and `:1506` through `:1525`.
- Searching the generated RDNA4 `vop3p.cpp` shows `inst_.clamp` only in later
  DOT2, MIX, and WMMA call sites, not in the ordinary packed I16/U16/F16
  arithmetic bodies.

Impact:

With `CM=1`, ordinary packed F16 arithmetic can write unclamped results above
`1.0`, negative values, NaNs, or `-0`, while packed integer arithmetic can wrap
or truncate instead of saturating to the manual's signed or unsigned range.
This is separate from the existing scalar VOP3 clamp corner cases because these
packed paths do not apply clamp at all.

### RDNA4-RJ-100: `V_PK_MINIMUM_F16` and `V_PK_MAXIMUM_F16` collapse to minNum/maxNum behavior

Reported by: Fermat subreviewer; local audit.

Manual evidence:

- `V_PK_MIN_NUM_F16` and `V_PK_MAX_NUM_F16` are defined as IEEE
  `minimumNumber()` / `maximumNumber()` operations that favor a numeric
  argument over NaN at `rdna4/README.md:17128` through `:17154`.
- `V_PK_MINIMUM_F16` and `V_PK_MAXIMUM_F16` are separate IEEE
  `minimum()` / `maximum()` operations, and the manual explicitly says a
  signaling NaN in either argument is propagated, at `rdna4/README.md:17157`
  through `:17180`.

Rocjitsu evidence:

- The semantic table maps `V_PK_MIN_F16`, `V_PK_MIN_NUM_F16`, and
  `V_PK_MINIMUM_F16` all to the same `('pk_binop', 'min', 'f16')` key, and
  maps the max variants to the same `max` key at
  `lib/python/amdisa/semantics.py:1523` through `:1528`.
- The packed generator emits `std::fmin` and `std::fmax` for those keys at
  `lib/python/amdisa/codegen/execute/packed.py:90` through `:100`.
- The generated `VPkMinNumF16Vop3p` and `VPkMinimumF16Vop3p` bodies both call
  `std::fmin`, while `VPkMaxNumF16Vop3p` and `VPkMaximumF16Vop3p` both call
  `std::fmax`, at `rdna4/vop3p.cpp:2336`, `:2439`, `:2542`, and `:2645`.

Impact:

Rocjitsu cannot distinguish the manual's `minimumNumber`/`maximumNumber`
instructions from IEEE `minimum`/`maximum`. NaN propagation and signed-zero
corner cases for the `MINIMUM`/`MAXIMUM` variants can therefore follow the
wrong instruction contract.

### RDNA4-RJ-101: `V_FMA_MIX*` uses multiply-add instead of fused FMA

Reported by: Fermat subreviewer; local audit.

Manual/XML evidence:

- The RDNA4 manual describes `V_FMA_MIX_F32`, `V_FMA_MIXLO_F16`, and
  `V_FMA_MIXHI_F16` as fused multiply-add instructions and uses `fma(...)` in
  the pseudocode at `rdna4/README.md:17185` through `:17249`.
- The XML descriptions for those same instructions also say they use fused
  multiply add at `amdgpu_isa_rdna4.xml:124615`, `:124941`, and `:125267`.

Rocjitsu evidence:

- The RDNA4 MIX generator emits `a * b + c` unless the newer gfx1250 helper
  path is enabled at `lib/python/amdisa/codegen/execute/packed.py:599` and
  `:674`.
- The generated RDNA4 `V_FMA_MIX_F32`, `V_FMA_MIXLO_F16`, and
  `V_FMA_MIXHI_F16` execution bodies compute `float result = a * b + c` at
  `rdna4/vop3p.cpp:2759`, `:2880`, and `:3002`.

Impact:

Inputs that depend on single-rounding fused FMA behavior can produce different
rounded results, exception behavior, or half conversion results in rocjitsu
than on hardware.

### RDNA4-RJ-102: Packed math accepts `EXEC` scalar sources that Chapter 7.7 marks invalid

Reported by: Fermat subreviewer; local audit.

Manual evidence:

- Chapter 7.7.1 marks `EXEC` invalid for packed scalar constants in both the
  `PK_F16` and `PK_F32` columns at `rdna4/README.md:3242` through `:3249`.

Rocjitsu evidence:

- RDNA4 `OPR_SRC` includes `OPR_SRC_EXEC_LO` and `OPR_SRC_EXEC_HI` at
  `rdna4/operand_types.h:152` through `:162`.
- The operand resolver names those encodings as `exec_lo`/`exec_hi` at
  `rdna4/operand.cpp:279` through `:282` and reads live EXEC state for
  selectors `126` and `127` at `rdna4/operand.cpp:766` through `:769`.
- Generated packed constructors use generic `OPR_SRC` for packed operands, for
  example `VPkAddF16Vop3p` at `rdna4/vop3p.cpp:1440` through `:1458` and the
  MIX constructors at `rdna4/vop3p.cpp:2780` through `:2805`.

Impact:

Rocjitsu can disassemble and execute packed-math source forms with `exec_lo` or
`exec_hi` that Chapter 7.7 says are invalid. This is a concrete packed-math
instance of the broader VOP3-family source legality issue tracked by
`RDNA4-RJ-092`.

### RDNA4-RJ-103: `S_SETREG_IMM32_B32` drops its literal source operand from metadata

Manual/XML evidence:

- `rdna4/README.md:2375` says SOPK normally cannot use literal constants, but
  `S_SETREG_IMM32_B32` is the exception.
- `rdna4/README.md:2564` says `S_SETREG_IMM32_B32` is a 64-bit SOPK
  instruction whose 32-bit data comes from a literal constant.
- The RDNA4 XML entry for `S_SETREG_IMM32_B32` uses `SOPK_INST_LITERAL` at
  `amdgpu_isa_rdna4.xml:37587` through `:37591` and includes an input
  `OPR_SIMM32` operand at `:37603`.

Rocjitsu evidence:

- RDNA4 `Sopk::hasImpliedLiteral()` returns true only for opcode 19, and the
  encoding constructor reads the following word into `literal_` at
  `rdna4/encodings.cpp:63` through `:71`.
- `SSetregImm32B32Sopk` constructs only the `OPR_HWREG` destination operand and
  sets `num_src_ = 0` at `rdna4/sopk.cpp:187` through `:192`.
- The execute body uses `literal_` as the source data at `rdna4/sopk.cpp:202`
  through `:210`, so execution sees the extension word even though the
  instruction has no source operand.
- The common disassembler prints only `dst_operands_` and `src_operands_` at
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:239` through `:252`.
- The Python XML parser appends operands only when a `FieldName` is present at
  `lib/python/amdisa/parser.py:918` through `:923`; the XML `OPR_SIMM32`
  literal source is fieldless.
- Existing harness coverage checks CDNA4 execution of `S_SETREG_IMM32_B32` at
  `tests/instruction_execution_harness_test.cpp:3592` through `:3633`, but not
  RDNA4 operand metadata or disassembly.

Impact:

Rocjitsu executes the RDNA4 literal-extension word for this instruction, but
operand-based consumers and disassembly see only the HWREG selector. A decoded
`s_setreg_imm32_b32` instruction therefore hides the 32-bit literal that the
manual and XML model as the instruction's explicit input.

### RDNA4-RJ-104: WREXEC accepts forbidden `EXEC` explicit destinations

Manual/XML evidence:

- Chapter 6.7 says `S_AND_NOT{0,1}_WREXEC_B{32,64}` writes both `D` and
  `EXEC`, but `D cannot be EXEC` at `rdna4/README.md:2503`.
- The XML encodes the explicit `SDST` for these instructions as `OPR_SREG` at
  `amdgpu_isa_rdna4.xml:20718` through `:20732` and `:20892` through
  `:20906`.
- `OPR_SREG` includes SGPRs, TTMP, VCC, and `NULL`, but not `M0` or `EXEC`, as
  reflected in generated RDNA4 operand constants at `rdna4/operand_types.h:210`
  through `:216`.

Rocjitsu evidence:

- The generated WREXEC constructors build `sdst` as `OperandType::OPR_SREG`
  from the raw `SDST` field, but do not validate that the raw encoding is in
  the `OPR_SREG` value set: `rdna4/sop1.cpp:956` through `:1028`.
- The shared WREXEC execute bodies write `inst.sdst` and then update `EXEC` at
  `shared/execute_shared.h:510` through `:529` and `:571` through `:590`.
- RDNA4 scalar write resolution accepts raw destination selectors `126` and
  `127` as `EXEC_LO`/`EXEC_HI` for 32-bit writes, and raw selector `126` as
  full `EXEC` for 64-bit writes at `rdna4/operand.cpp:916` through `:986`.
- `Operand::write_scalar64()` dispatches to that raw destination resolver
  without checking the operand type at `rdna4/operand.cpp:1166` through
  `:1167`.

Impact:

Invalid WREXEC encodings with `SDST=exec_lo`, `exec_hi`, or `exec` can execute
and mutate `EXEC` through the explicit destination path before the instruction's
implicit EXEC write. Hardware should reject or otherwise not treat those raw
selectors as legal explicit destinations for this instruction class.

### RDNA4-RJ-105: SALU implicit SCC operands are missing from generated def-use metadata

Manual/XML evidence:

- Chapter 6.3 says most SALU instructions write SCC, signed arithmetic uses
  overflow, bit/logical operations use result-nonzero, and extended arithmetic
  uses SCC as carry/borrow input at `rdna4/README.md:2383` through `:2392`.
- Chapter 6.5 says scalar conditional moves use SCC but do not write it at
  `rdna4/README.md:2445` through `:2447`.
- Chapter 6.6 says SOPC comparisons and bit compares set SCC at
  `rdna4/README.md:2459` through `:2460`.
- The XML exposes these SCC dependencies structurally. For example,
  `S_ADD_CO_CI_U32` has implicit SCC output and input operands at
  `amdgpu_isa_rdna4.xml:23149` through `:23154`, and `S_CMOVK_I32` has an
  implicit SCC input at `amdgpu_isa_rdna4.xml:37421`.

Rocjitsu evidence:

- `DefUseChain` collects explicit destination/source operands and then calls
  `implicit_defs()` and `implicit_uses()` at
  `lib/rocjitsu/src/rocjitsu/analysis/def_use_chain.cpp:28` through `:44`.
- The base `Instruction` implementations of both hooks are empty at
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:219` through `:222`.
- Representative generated classes that read or write SCC do not override those
  hooks: `SAddCoI32Sop2`, `SAddCoCiU32Sop2`, and `SSubCoCiU32Sop2` are
  declared without SCC hooks at `rdna4/sop2.h:35` through `:64`, and their
  constructors add only explicit SDST/SSRC operands at `rdna4/sop2.cpp:71`
  through `:167`.
- Conditional move constructors likewise add only explicit operands, for
  example `SCmovB32Sop1` / `SCmovB64Sop1` at `rdna4/sop1.cpp:56` through
  `:92` and `SCselectB32Sop2` / `SCselectB64Sop2` at `rdna4/sop2.cpp:1104`
  through `:1150`.
- SOPC compare classes are declared without SCC def hooks at
  `rdna4/sopc.h:17` through `:377`, and representative constructors only set
  explicit sources in `rdna4/sopc.cpp:20` through `:39` and `:296` through
  `:384`.

Impact:

Runtime execution can still be correct, but operand-based analyses do not see
SCC as a dependency or clobber for ordinary SALU SCC producers and consumers.
That can hide ordering hazards in def-use based transforms, schedulers, or
metadata-driven tests even when `execute_impl()` reads or writes SCC correctly.

### RDNA4-RJ-106: Scalar F16 inline constants use 32-bit scalar constant values

Reported by: local audit.

Manual/XML evidence:

- Chapter 6.8 says scalar F16 instructions do not encode half SGPRs and operate
  on the low `bit[15:0]` part of the named SGPR at `rdna4/README.md:2552`.
- XML marks scalar F16 sources as `FMT_NUM_F16` / `OperandSize 16`, for example
  `S_ADD_F16` at `amdgpu_isa_rdna4.xml:30625` through `:30649` and
  `S_CMP_EQ_F16` at `amdgpu_isa_rdna4.xml:34510` through `:34532`.

Rocjitsu evidence:

- RDNA4 `resolve_src_scalar()` maps inline selector `242` to the F32 bit pattern
  `0x3F800000` at `rdna4/operand.cpp:783`.
- RDNA4 has a separate `resolve_src_scalar16()` path that maps the same selector
  to the F16 bit pattern `0x3C00` at `rdna4/operand.cpp:821` through `:828`.
- `Operand::read_scalar()` ignores `size_bits_` and always calls
  `resolve_src_scalar()` for non-immediate operands at `rdna4/operand.cpp:1078`
  through `:1085`; only `read_lane()` switches to `resolve_src_scalar16()` for
  16-bit operands at `rdna4/operand.cpp:1104` through `:1105`.
- Scalar F16 execute bodies call `RegisterAccess(wf).read_scalar()` and then
  cast to `uint16_t`, for example `S_ADD_F16` at
  `shared/execute_shared.h:403` through `:407` and `S_CMP_EQ_F16` at
  `shared/execute_shared.h:1103` through `:1111`.

Impact:

An inline F16 source such as selector `242` (`1.0`) is read as F32 `0x3F800000`
and then truncated to low 16 bits, producing `0x0000` instead of F16 `0x3C00`.
The same generated scalar-reader shape is present in several other architecture
directories, so the RDNA4 finding should be checked for cross-architecture
propagation rather than treated as an RDNA4-only defect.

### RDNA4-RJ-107: Scalar floating-point helpers ignore MODE and FP exception behavior

Reported by: local audit.

Manual evidence:

- Chapter 6.8 says scalar floating-point rounding and denormal handling follow
  `MODE.round` and `MODE.denorm` at `rdna4/README.md:2548`.
- The same section says scalar floating-point arithmetic instructions can
  trigger floating-point exceptions handled like VALU-pipe exceptions at
  `rdna4/README.md:2550`.

Rocjitsu evidence:

- `Wavefront` stores raw MODE state at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:104` through `:108`, and
  `S_ROUND_MODE` / `S_DENORM_MODE` missing updates are already tracked by
  `RDNA4-RJ-013`.
- Scalar FP helpers use host arithmetic and fixed conversion helpers without
  reading `wf.mode_raw()`: representative examples include `S_ADD_F16` /
  `S_ADD_F32` at `shared/execute_shared.h:403` through `:415`,
  `S_CVT_F16_F32` at `shared/execute_shared.h:1547` through `:1554`, and
  `S_FMAC_F16` / `S_FMAC_F32` at `shared/execute_shared.h:1747` through
  `:1760`.
- `util::f32_to_f16()` implements a fixed round-to-nearest-even conversion at
  `lib/util/include/util/data_types.h:93` through `:126`; `S_CVT_PK_RTZ_F16_F32`
  separately uses fixed RTZ as expected at `shared/execute_shared.h:1610`
  through `:1617`.

Impact:

Even if MODE is seeded externally or updated through a fixed state-register
path, scalar FP execution does not vary with MODE rounding or denormal bits and
does not record the exception behavior Chapter 6.8 assigns to these
instructions.

### RDNA4-RJ-108: Non-accumulating scalar F16 instructions report old-destination uses

Reported by: local audit.

Manual/XML evidence:

- Chapter 6.8 says scalar F16 destination high bits are set to zero at
  `rdna4/README.md:2552`.
- XML marks non-accumulating scalar F16 destinations as output-only, for example
  `S_CVT_F16_F32` at `amdgpu_isa_rdna4.xml:22177` through `:22208`,
  `S_CVT_PK_RTZ_F16_F32` at `amdgpu_isa_rdna4.xml:30508` through `:30525`,
  and `S_MINIMUM_F16` at `amdgpu_isa_rdna4.xml:31567` through `:31591`.
- By contrast, XML explicitly models `S_FMAC_F16` `SDST` as both input and
  output because it accumulates with the destination at
  `amdgpu_isa_rdna4.xml:31216` through `:31234`.

Rocjitsu evidence:

- Generated RDNA4 constructors add implicit `sdst` uses for output-only scalar
  F16 instructions such as `S_CVT_F16_F32` at `rdna4/sop1.cpp:1516`,
  `S_CVT_PK_RTZ_F16_F32` at `rdna4/sop2.cpp:1486`, `S_ADD_F16` at
  `rdna4/sop2.cpp:1518`, `S_SUB_F16` at `rdna4/sop2.cpp:1548`,
  `S_MIN_NUM_F16` at `rdna4/sop2.cpp:1579`, `S_MAX_NUM_F16` at
  `rdna4/sop2.cpp:1612`, `S_MUL_F16` at `rdna4/sop2.cpp:1644`,
  `S_MINIMUM_F16` at `rdna4/sop2.cpp:1758`, and `S_MAXIMUM_F16` at
  `rdna4/sop2.cpp:1791`.
- The unary F16 rounding helpers also add old-destination uses at
  `rdna4/sop1.cpp:1581`, `:1607`, `:1633`, and `:1659`.
- Execution writes the scalar destination directly through
  `Operand::write_scalar()` at `rdna4/operand.cpp:1109` and representative
  shared helpers write a fresh value, including full packed results for
  `S_CVT_PK_RTZ_F16_F32` at `shared/execute_shared.h:1610` through `:1617`.

Impact:

Runtime results for these helpers can still be correct, but def-use analysis
and metadata-driven tests see false dependencies on the previous destination
value. `S_FMAC_F16` is the accumulator exception and should remain an
old-destination use.

### RDNA4-RJ-109: `S_CVT_HI_F32_F16` literal sources lose the high half before execution

Reported by: local audit.

Manual/XML evidence:

- The detailed instruction definition says
  `D0.f32 = f16_to_f32(S0[31 : 16].f16)` at `rdna4/README.md:10518` through
  `:10525`.
- XML exposes a `SOP1_INST_LITERAL` form with a `SIMM32` source for
  `S_CVT_HI_F32_F16` at `amdgpu_isa_rdna4.xml:22287` through `:22324`.
- A local `llvm-mc -arch=amdgcn -mcpu=gfx1200` check accepts
  `s_cvt_hi_f32_f16 s0, 0x3c000000` and emits a 32-bit literal extension word.

Rocjitsu evidence:

- The generated RDNA4 constructor masks literal sources with `simm32 & 0xFFFF`
  before constructing the operand at `rdna4/sop1.cpp:1546` through `:1560`.
- The shared execute body then shifts the scalar source right by 16 before
  converting at `shared/execute_shared.h:1583` through `:1589`.
- Existing scalar conversion coverage exercises the register-source high-half
  case at `tests/scalar_scc_test.cpp:950`, but not the literal-source form.

Impact:

A literal such as `0x3c000000`, whose high half is F16 `1.0`, is first reduced
to `0x0000` and then shifted right by 16, so rocjitsu converts `+0` instead of
the high-half literal value.

### RDNA4-RJ-110: RDNA4 SMEM byte and halfword loads are dword-aligned

Reported by: local audit.

Manual/XML evidence:

- Chapter 8.1.1 says SMEM address components are byte quantities whose low
  bits are ignored according to access size: two low bits for dword loads, one
  low bit for 16-bit loads, and no low-bit masking for 8-bit loads at
  `rdna4/README.md:3903` through `:3911`.
- Chapter 8.1.3 says 8-bit and 16-bit scalar loads sign- or zero-extend to
  32-bit scalar results, and 16-bit loads must be 16-bit aligned, at
  `rdna4/README.md:3945` through `:3954`.
- Chapter 8.4.1 restates that dword-or-larger loads force dword alignment,
  16-bit loads force two-byte alignment, and byte loads have no forced
  alignment at `rdna4/README.md:4000` through `:4002`.
- XML carries the narrow implicit memory sizes for representative entries such
  as `S_LOAD_I8` at `amdgpu_isa_rdna4.xml:16260` through `:16290` and
  `S_LOAD_I16` at `amdgpu_isa_rdna4.xml:16352` through `:16382`.

Rocjitsu evidence:

- `rdna4/addr_calc.cpp:56` through `:65` always returns `(base + off) &
  ~0x3ULL`, regardless of instruction access size.
- Generated narrow-load bodies set `elem_size` to one or two bytes but still
  call `smem_calculate_address(inst_, wf)`, for example `SLoadI8Smem` at
  `rdna4/smem.cpp:203` through `:213`, `SBufferLoadI8Smem` at
  `rdna4/smem.cpp:470` through `:480`, and `SBufferLoadU16Smem` at
  `rdna4/smem.cpp:551` through `:561`.
- The generator has an access-size-aware call path guarded by
  `smem_address_uses_access_size` at `codegen/_generator.py:4285` through
  `:4289`, but the base profile default is false at `isa_profile.py:338`
  through `:340`; only `Gfx1250Profile` overrides it at
  `isa_profile.py:1626` through `:1627`.

Impact:

RDNA4 `S_LOAD_I8/U8` and `S_BUFFER_LOAD_I8/U8` read from the previous
dword-aligned address instead of the exact byte address. The 16-bit forms also
over-align to four bytes instead of two bytes. Dword forms can also differ
when ignored low bits should be discarded per component rather than after the
components are summed.

### RDNA4-RJ-111: RDNA4 S_BUFFER_LOAD uses raw-pointer addressing instead of buffer-resource fields

Reported by: local audit.

Manual/XML evidence:

- Chapter 8.1.2 says `S_BUFFER_LOAD` gets its base from the buffer resource,
  uses only `base_address`, `stride`, and `num_records`, does not support
  swizzled buffers, uses `stride` only for bounds checking, and treats negative
  `IOFFSET` as `MEMVIOL` at `rdna4/README.md:3915` through `:3942`.
- Chapter 8.4.1 defines buffer-load address out-of-range as
  `offset >= ((stride == 0 ? 1 : stride) * num_records)`, where `offset` is
  `IOFFSET + {M0 or sgpr-offset}`, and says out-of-range dwords in a
  multi-dword buffer load return zero at `rdna4/README.md:4004` through `:4010`.
- XML marks scalar-buffer load bases as 128-bit `FMT_RSRC_SCALAR`, for example
  `S_BUFFER_LOAD_B32` at `amdgpu_isa_rdna4.xml:16444` through `:16477` and
  `S_BUFFER_LOAD_I16` at `amdgpu_isa_rdna4.xml:16827` through `:16856`.

Rocjitsu evidence:

- Generated RDNA4 scalar-buffer constructors expose `sbase` as 128-bit
  `OPR_SREG`, for example `SBufferLoadB32Smem` at `rdna4/smem.cpp:294`
  through `:306` and `SBufferLoadI8Smem` at `rdna4/smem.cpp:456` through
  `:468`.
- The execute bodies call the same `smem_calculate_address(inst_, wf)` helper
  used by raw `S_LOAD`, for example `SBufferLoadB32Smem` at
  `rdna4/smem.cpp:308` through `:318` and `SBufferLoadU16Smem` at
  `rdna4/smem.cpp:551` through `:561`.
- That helper reads only the first two SGPRs as a raw 64-bit base and adds the
  signed immediate plus `SOFFSET` at `rdna4/addr_calc.cpp:56` through `:65`.
  It does not decode `resource[47:0]`, `resource[61:48]`,
  `resource[95:64]`, bounds size, `stride_scale`, per-dword out-of-range
  zeroing, or negative-`IOFFSET` `MEMVIOL`.

Impact:

Scalar-buffer loads can use the wrong address and never enforce the manual's
resource bounds, per-dword out-of-range zeroing, or negative-offset fault
behavior. Any descriptor whose high bits contain stride/size fields instead of a
raw pointer is at risk.

### RDNA4-RJ-112: SMEM source metadata drops SOFFSET register operands

Reported by: local audit.

Manual/XML evidence:

- Chapter 8.1 says SMEM addresses are composed from `SBASE`, `IOFFSET`, and
  `SOFFSET`, with `SOFFSET` selecting an SGPR, `M0`, or `NULL` at
  `rdna4/README.md:3867` through `:3899`.
- XML models the explicit source operand as `SOFFSET` with
  `OPR_SMEM_OFFSET`, while `IOFFSET` is the direct 24-bit field, for example
  `S_LOAD_B32` at `amdgpu_isa_rdna4.xml:15969` through `:16000`.

Rocjitsu evidence:

- `make_smem_offset()` constructs the second source operand from
  `enc->ioffset` as `OPR_SIMM32` at `rdna4/smem.cpp:26` through `:28`.
- Generated constructors then publish that immediate as `src_operands_[1]`,
  for example `SLoadB32Smem` at `rdna4/smem.cpp:31` through `:41` and
  `SBufferLoadB32Smem` at `rdna4/smem.cpp:294` through `:306`.
- Runtime address calculation still reads `inst.soffset` at
  `rdna4/addr_calc.cpp:36` through `:41` and adds it at `:63` through `:64`.

Impact:

Execution may use a live `SOFFSET` register while disassembly and operand-based
analyses report only the immediate `IOFFSET` value. A dependency on an SGPR or
`M0` offset can therefore be invisible to def-use metadata and generated
goldens.

### RDNA4-RJ-113: SMEM special SDATA destinations are written as plain SGPRs

Reported by: local audit.

Manual/XML evidence:

- Chapter 8.1 says `SDATA` may be SGPR, VCC, or `NULL`, not `EXEC` or `M0`, at
  `rdna4/README.md:3865`.
- Chapter 8.1 also says a `NULL` SMEM destination can be used for prefetch data
  with acknowledge at `rdna4/README.md:3897` through `:3899`.
- XML `OPR_SREG` includes VCC and `NULL`; the `null` predefined value says a
  destination write should be suppressed at `amdgpu_isa_rdna4.xml:189665`
  through `:190298`.

Rocjitsu evidence:

- Generated SMEM load bodies store the raw selector as a physical SGPR base,
  for example `SLoadB32Smem` sets `d->dst_reg_base = wf.sgpr_alloc().base +
  inst_.sdata` at `rdna4/smem.cpp:44` through `:54`, and buffer/narrow forms
  use the same pattern at `rdna4/smem.cpp:470` through `:480` and `:551`
  through `:561`.
- `ScalarMemPipeline::complete_access()` writes every response word with
  `cu.write_sgpr(d.dst_reg_base + i, ...)` at
  `vm/amdgpu/memory_pipeline.cpp:218` through `:227`.
- The writeback path has no check for `SDATA == null` and no route to
  `Wavefront::set_vcc()` for `SDATA == vcc_lo/vcc_hi`.

Impact:

Loads to `NULL` are not discarded as acknowledged prefetches, and loads to VCC
do not update the wavefront VCC state. They instead write raw SGPR-file slots
derived from the selector value.

### RDNA4-RJ-114: Derived SMEM semantics collapse RDNA4 addressing to a 32-bit global address

Reported by: local audit.

Manual/XML evidence:

- Chapter 8.1.1 gives raw-pointer SMEM addressing as a 64-bit base plus
  byte offsets at `rdna4/README.md:3901` through `:3913`.
- Chapter 8.1.2 gives a separate scalar-buffer resource formula at
  `rdna4/README.md:3915` through `:3942`.

Rocjitsu evidence:

- The autogenerated semantic deriver for `smem_load` calls
  `CalcScalarGlobalAddr` with source 0 cast to `U64` and source 1 cast to
  `U32`, then assigns the result into a `U32` `addr` variable at
  `lib/python/amdisa/sema_derive.py:1641` through `:1658`.
- This derived semantic form has no separate `IOFFSET`/`SOFFSET` fields, no
  access-size component masking, no buffer-resource descriptor path, and no
  64-bit address result.

Impact:

The hand-generated C++ execution path is the primary runtime path, but semantic
tools or future generators that rely on `sema_derive.py` cannot reconstruct the
RDNA4 SMEM contract from this derived form.

### RDNA4-RJ-115: SMEM dependency timing is collapsed to immediate completion

Reported by: local audit.

Manual evidence:

- `rdna4/README.md:3962` says scalar memory loads can return data out of issue
  order and can return partial results at different times when a load crosses
  two cache lines.
- `rdna4/README.md:3968` says the only sensible use is `S_WAIT_KMCNT 0`, which
  waits for all data from previous SMEMs before continuing.

Rocjitsu evidence:

- `MemoryPipeline::issue` documents the functional-mode behavior as completing
  memory accesses synchronously inside the call at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.h:55` through `:61`,
  then increments the counter, initiates the access, calls `complete_access`,
  and releases the counter immediately when the completion is synchronous at
  `:62` through `:88`.
- `ScalarMemPipeline::initiate_access` performs the scalar memory load directly
  at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:203` through
  `:212`.
- `ScalarMemPipeline::complete_access` writes every destination dword to SGPRs
  and returns `Complete` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:218` through
  `:241`.

Impact:

An SMEM load's data is written before later instructions can execute in the
functional backend, so a missing `S_WAIT_KMCNT 0` cannot expose the hardware
hazard described by Chapter 8.2. This also prevents rocjitsu from modeling
out-of-order or partial cache-line return behavior for scalar loads.

### RDNA4-RJ-116: SMEM group and `S_DCACHE_INV` clause restrictions are absent

Reported by: local audit.

Manual evidence:

- `rdna4/README.md:3974` defines scalar clauses as sequences that begin with
  `S_CLAUSE` and continue for 2-63 instructions.
- `rdna4/README.md:3976` defines scalar-memory groups as same-type instruction
  sequences that end at a non-SMEM instruction, says SMEM instructions are
  issued in groups, and notes that hardware does not force one wave to finish an
  entire group before another wave issues.
- `rdna4/README.md:3978` through `:3980` says `INV` must be in a group by
  itself and may not be in a clause.

Rocjitsu evidence:

- RDNA4 `S_CLAUSE` constructs only an `OPR_CLAUSE` source operand and records no
  active clause state at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/sopp.cpp:64` through `:72`.
- The shared `execute_s_clause_sopp` helper is empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1031`
  through `:1032`.
- The common instruction flags include memory, wait, barrier, branch, MFMA, and
  predicated-def categories but no clause or group membership at
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:24` through `:50`.
- `ComputeUnitCore::step` iterates all running wavefronts and issues one
  instruction from each at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:442`
  through `:450`, with no SMEM group state or group-end check.
- `S_DCACHE_INV` is generated as a normal operandless SMEM instruction that
  immediately invalidates the scalar cache at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/smem.cpp:564` through
  `:571`, with no validation that it is alone in a group or outside a clause.

Impact:

Rocjitsu can execute scalar-memory instructions one at a time, but it cannot
validate or model Chapter 8.3's scalar-memory group contract. In particular,
`S_DCACHE_INV` can appear after `S_CLAUSE` or adjacent to other SMEM operations
without any group-alone restriction being observed.

### RDNA4-RJ-117: SMEM SGPR range checks and SDST alignment effects are absent

Reported by: local audit.

Manual evidence:

- `rdna4/README.md:3984` through `:3986` says `SDST` must be even for two-dword
  fetches and a multiple of four for larger fetches; otherwise invalid data can
  result.
- `rdna4/README.md:3988` through `:3990` says `S_BUFFER_LOAD` `SBASE` must be
  even and that an out-of-range `SBASE` uses the value from `SGPR0`.
- `rdna4/README.md:3996` through `:3998` says hardware checks source and
  destination SGPRs for out-of-range conditions.
- `rdna4/README.md:4012` through `:4018` says out-of-range source SGPR data is
  replaced with zero, while an out-of-range destination SGPR suppresses writeback
  for the instruction.

Rocjitsu evidence:

- `dispatch_wf` stores both the physical SGPR allocation base/count and the
  declared `wf->num_sgprs_` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:115` and `:139` through
  `:142`, while the RDNA4 default CU configuration reserves 128 physical SGPRs
  per wave at `lib/rocjitsu/src/rocjitsu/config/config_loader.cpp:66` through
  `:68`.
- `smem_calculate_address` reads `SBASE` as `wf.sgpr_alloc().base +
  inst.sbase * 2` and reads both dwords directly from the physical SGPR file at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.cpp:56` through
  `:62`, with no `wf.num_sgprs()` check or fallback to `SGPR0`.
- `read_smem_offset` reads SGPR `SOFFSET` directly from
  `wf.sgpr_alloc().base + soffset` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.cpp:36` through
  `:41`, with no source-out-of-range zero substitution.
- Generated load bodies set `dst_reg_base = wf.sgpr_alloc().base + inst_.sdata`
  and a fixed dword count, for example `S_LOAD_B64` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/smem.cpp:70` through `:79`,
  `S_LOAD_B96` at `:177` through `:186`, and `S_BUFFER_LOAD_B512` at `:416`
  through `:425`, with no SDST alignment or range check.
- `ScalarMemPipeline::complete_access` writes every response dword with
  `cu.write_sgpr(d.dst_reg_base + i, ...)` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:218` through `:227`;
  the physical CU accessors index the backing SGPR file directly at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.h:379` through `:392`.

Impact:

SMEM operations can read or write physical SGPR slots that are outside the
shader's declared scalar-register range but still inside rocjitsu's physical
allocation block. Out-of-range `SBASE` does not fall back to `SGPR0`,
out-of-range `SOFFSET` does not become zero, out-of-range destinations are not
suppressed, and misaligned multi-dword `SDST` receives clean sequential data
instead of the manual's invalid-data behavior.

### RDNA4-RJ-118: Scalar prefetch instructions execute as unconditional no-ops

Reported by: local audit.

Manual evidence:

- `rdna4/README.md:4022` through `:4028` says scalar prefetch instructions
  request instruction/data prefetch into first-level caches, do not use `KMcnt`,
  use `SCOPE`/`TH` cache reuse policies for data-prefetch, and are skipped as
  `S_NOP` only when `MODE.SCALAR_PREFETCH_EN == 0`.
- `rdna4/README.md:4028` through `:4030` defines the length as
  `SOFFSET + SDATA`, modulo 32, representing 1-32 chunks of 128 bytes.
- `rdna4/README.md:4034` through `:4063` defines base, buffer, and PC-relative
  prefetch address formulas, 128-byte forced alignment, `PC + 8` for
  PC-relative forms, and the `S_BUFFER_PREFETCH_DATA` negative-`IOFFSET`
  drop/no-`MEMVIOL` rule.
- The detailed instruction definitions repeat the `MODE.SCALAR_PREFETCH_EN`
  guard, no completion/error status, length calculation, and cache target calls
  at `rdna4/README.md:11941` through `:12070`.

Rocjitsu evidence:

- The semantic deriver classifies `S_PREFETCH_INST`,
  `S_PREFETCH_INST_PC_REL`, `S_PREFETCH_DATA`, `S_PREFETCH_DATA_PC_REL`, and
  `S_BUFFER_PREFETCH_DATA` as `true_nop` at `lib/python/amdisa/semantics.py:1689`
  through `:1698`; `true_nop` derives to an empty scalar semantic block at
  `lib/python/amdisa/sema_derive.py:2678` through `:2687`.
- The generator lowers ordinary `true_nop` bodies to `(void)wf` at
  `lib/python/amdisa/codegen/_generator.py:3350` through `:3351`.
- The generated RDNA4 prefetch constructors preserve the decoded operands, but
  each execute body is empty: `S_PREFETCH_INST` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/smem.cpp:603` through `:618`,
  `S_PREFETCH_INST_PC_REL` at `:620` through `:633`,
  `S_PREFETCH_DATA` at `:635` through `:650`,
  `S_BUFFER_PREFETCH_DATA` at `:652` through `:667`, and
  `S_PREFETCH_DATA_PC_REL` at `:669` through `:682`.

Impact:

Rocjitsu decodes scalar prefetch instructions but never computes their address
or length, never checks `MODE.SCALAR_PREFETCH_EN`, never applies data-prefetch
`SCOPE`/`TH` cache policy, never drops negative buffer prefetches specifically,
and never warms instruction or scalar data cache state. This is functionally
benign for many kernels because prefetch has no completion status, but it loses
the cache-model and mode-gating behavior described by Chapter 8.5.

### RDNA4-RJ-119: RDNA4 VBUFFER address calculation ignores descriptor bounds, stride, swizzle, and `IDXEN`

Reported by: local audit.

Manual evidence:

- Table 47 says `VADDR` supplies an index, an offset, or both; `IOFFSET` is a
  signed 24-bit byte offset that must be non-negative; and `RSRC` selects four
  consecutive SGPRs aligned to a multiple of four at `rdna4/README.md:4128`
  through `:4145`.
- Chapter 9.4 defines buffer index/offset/stride addressing and the
  resource-controlled `const_add_tid_enable`, swizzle, and index-stride fields
  at `rdna4/README.md:4386` through `:4408`.
- Chapter 9.4.1 defines `OOB_SELECT`, zero-return/drop behavior,
  per-component versus all-or-nothing range checks, and structured/raw/scratch
  address formulas at `rdna4/README.md:4434` through `:4488`.
- Chapter 9.4.2 defines swizzled-buffer address calculation and restrictions at
  `rdna4/README.md:4509` through `:4530`.
- Chapter 9.6 defines the descriptor bit layout, including base, stride,
  `Num_records`, `Dst_sel`, format, stride scale, index stride, add-tid,
  `OOB_SELECT`, and type at `rdna4/README.md:4562` through `:4588`.

Rocjitsu evidence:

- RDNA4 `mubuf_calculate_addresses` reads only the descriptor base from
  `s[rsrc]` and low base bits from `s[rsrc+1]`, then calculates
  `base + buffer_offset_part(voffset, ioff) + soffset` at
  `rdna4/addr_calc.cpp:161` through `:190`.
- The same helper asserts `!inst.idxen` at `rdna4/addr_calc.cpp:177`, so a
  legal index-only or index-plus-offset VBUFFER form aborts debug builds and
  silently lacks index/stride behavior in non-assert builds.
- The helper sign-extends `IOFFSET` at `rdna4/addr_calc.cpp:176` but has no
  check for the manual's non-negative `IOFFSET` restriction.
- It does not read `s[rsrc+2]` or `s[rsrc+3]`, so descriptor `Num_records`,
  stride, swizzle, add-tid, index-stride, `OOB_SELECT`, `Dst_sel`, format, and
  type do not participate in address calculation or lane masking.
- Existing RDNA4 tests cover the decoded first-SGPR selector, optional
  `SOFFSET`, `M0`, `OFFEN`, and offset wrapping at
  `tests/shared_infra_test.cpp:3240` through `:3335`; the structured-stride
  and zero-records range tests exist only for `gfx1250` at
  `tests/shared_infra_test.cpp:3337` through `:3455`.

Impact:

Legal indexed VBUFFER instructions are not modeled correctly, descriptor-based
structured/raw/scratch/swizzled addressing is absent, and out-of-range
buffer loads/stores/atomics can hit memory instead of returning zero or being
dropped.

### RDNA4-RJ-120: RDNA4 VBUFFER `TFE` does not add or write the PRT status VGPR

Reported by: local audit.

Manual evidence:

- Table 47 says `TFE=1` enables PRT status reporting and writes status to the
  VGPR after the last fetch-destination VGPR when the fetch returns a NACK at
  `rdna4/README.md:4145`.
- Chapter 3.3.4 says destination-range checks include the extra PRT VGPR and
  nullify the fetch if that extra VGPR would be out of range, regardless of
  whether the texture system returns it at `rdna4/README.md:909` through `:910`.

Rocjitsu evidence:

- `VbufferMachineInst` stores the decoded `tfe` bit at
  `rdna4/machine_insts.h:134` through `:152`.
- `Vbuffer::build_modifiers` prints only `offen`, `idxen`, `offset`, and `nv`
  at `rdna4/encodings.cpp:332` through `:343`, so `tfe` is not preserved in
  disassembly.
- Implemented load constructors keep only the ordinary `VDATA` destination; for
  example `BUFFER_LOAD_B32` has one destination and sets `num_dst_ = 1` at
  `rdna4/vbuffer.cpp:494` through `:509`.
- The runtime vector-memory state has no TFE/status field or extra destination
  tracking at `vm/amdgpu/mem_state.h:87` through `:129`.

Impact:

`TFE=1` buffer loads lose the extra status result, miss the extra destination
range/nullification rule, and disassemble as if the bit were clear.

### RDNA4-RJ-121: RDNA4 formatted and typed buffer instructions decode but are not executable

Reported by: local audit.

Manual evidence:

- Table 46 lists untyped formatted buffer loads/stores, D16 formatted variants,
  D16_HI formatted variants, typed `TBUFFER_*` loads/stores, and typed D16
  loads/stores at `rdna4/README.md:4106` through `:4126`.
- Chapter 9.1 says typed buffer operations use the instruction's format field
  and expand results to 32-bit components except D16 forms, which expand to
  16-bit components, at `rdna4/README.md:4147` through `:4172`.

Rocjitsu evidence:

- Generated RDNA4 untyped formatted loads and stores construct operands, but
  `BUFFER_LOAD_FORMAT_X` and `BUFFER_STORE_FORMAT_X` throw
  `util::UnimplementedInst` at `rdna4/vbuffer.cpp:52` through `:55` and
  `:136` through `:139`; adjacent `_XY/_XYZ/_XYZW` formatted bodies follow the
  same pattern.
- Generated RDNA4 D16 formatted loads throw in the same way, for example
  `BUFFER_LOAD_D16_FORMAT_X` at `rdna4/vbuffer.cpp:220` through `:223`.
- Generated typed buffer operations also throw, for example
  `TBUFFER_LOAD_FORMAT_X` at `rdna4/vbuffer.cpp:2571` through `:2574`, with
  the typed store and D16 typed bodies through `rdna4/vbuffer.cpp:2571` through
  `:2891` following the same pattern.

Impact:

Rocjitsu can decode and print many Chapter 9 VBUFFER opcodes, but executable
coverage is limited to plain unformatted buffer loads/stores and atomics.

### RDNA4-RJ-122: RDNA4 VBUFFER descriptor data controls and unbound-resource rules are ignored

Reported by: local audit.

Manual evidence:

- Chapter 9.3 says buffer data is controlled by resource format, `dst_sel`, and
  opcode, and identifies which instruction families use instruction format,
  resource format, or an opcode-derived format at `rdna4/README.md:4335`
  through `:4353`.
- The same section says a resource `INVALID` data format is an unbound-resource
  marker and is not replaced by an opcode-implied data format at
  `rdna4/README.md:4355` through `:4357`.
- Chapter 9.6 says unbound resources are keyed off data-format zero and, for
  untyped buffers, `add_tid_en=false`; resource/instruction type mismatch is
  ignored, with loads returning nothing and stores not altering memory, at
  `rdna4/README.md:4590` through `:4598`.

Rocjitsu evidence:

- Executed plain VBUFFER loads derive `elem_size`, `num_elems`, and
  sign-extension directly from the opcode; for example `BUFFER_LOAD_B32` sets
  those fields at `rdna4/vbuffer.cpp:511` through `:522`.
- Executed plain VBUFFER stores likewise derive their payload directly from the
  opcode; for example `BUFFER_STORE_B16` sets `elem_size`, `num_elems`,
  `is_load`, and `STORECNT` at `rdna4/vbuffer.cpp:664` through `:675`.
- RDNA4 address calculation reads only descriptor base bits from `s[rsrc]` and
  `s[rsrc+1]` at `rdna4/addr_calc.cpp:166` through `:171`; descriptor format,
  `Dst_sel`, add-tid, and type are not read by the implemented VBUFFER path.

Impact:

Plain buffer operations can access memory even when the descriptor represents an
unbound or mismatched resource, and resource-driven component selection/format
state is unavailable for future formatted-buffer execution.

### RDNA4-RJ-123: RDNA4 VBUFFER alignment and `MEMVIOL` behavior is absent

Reported by: local audit.

Manual evidence:

- Chapter 9 introduces buffer addressing by saying memory instructions return
  `MEMVIOL` for misaligned accesses when the alignment mode does not allow them
  at `rdna4/README.md:4090` through `:4094`.
- Chapter 9.5 says formatted buffer ops have fixed alignment requirements,
  non-formatted ops are controlled by `SH_MEM_CONFIG.alignment_mode`, and
  atomics must be aligned to the data size or trigger `MEMVIOL` at
  `rdna4/README.md:4538` through `:4560`.

Rocjitsu evidence:

- RDNA4 VBUFFER address calculation has no alignment check or address
  adjustment beyond 32-bit offset wrapping at `rdna4/addr_calc.cpp:161`
  through `:190`.
- Searches for `alignment_mode`, `MEMVIOL`, `memviol`, and `SH_MEM_CONFIG` in
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4` and
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu` found no buffer-alignment or
  memory-violation model.
- `VectorMemState` carries addresses, masks, payload sizes, and D16 flags, but
  no `MEMVIOL` or alignment-mode state at `vm/amdgpu/mem_state.h:87` through
  `:129`.

Impact:

Misaligned VBUFFER accesses can execute as ordinary memory operations even when
the manual requires forced alignment or `MEMVIOL`, and atomics do not get their
stricter payload-alignment behavior.

### RDNA4-RJ-124: RDNA4 VBUFFER disassembly drops semantic fields

Reported by: local audit.

Manual evidence:

- Table 47 lists `FORMAT`, `SCOPE`, `TH`, and `TFE` as instruction fields, with
  `FORMAT` controlling typed-buffer memory data format, `SCOPE`/`TH`
  controlling memory/cache/atomic-return behavior, and `TFE` controlling PRT
  status writes at `rdna4/README.md:4128` through `:4145`.

Rocjitsu evidence:

- `VbufferMachineInst` captures `tfe`, `scope`, `th`, `format`, `offen`,
  `idxen`, `vaddr`, and `ioffset` at `rdna4/machine_insts.h:134` through
  `:152`.
- `Vbuffer::build_modifiers` emits only `offen`, `idxen`, nonzero `offset`,
  and `nv` at `rdna4/encodings.cpp:332` through `:343`.
- The printed `offset` uses the raw unsigned 24-bit field, while the runtime
  sign-extends `IOFFSET` at `rdna4/addr_calc.cpp:176`.

Impact:

Disassembly can hide cache/atomic-return, typed-format, and PRT-status behavior
and can print negative encoded `IOFFSET` values as large positive offsets.

### RDNA4-RJ-125: RDNA4 image and sample instructions decode as non-memory stubs

Reported by: local audit.

Manual evidence:

- `rdna4/README.md:4604` through `:4612` says VMEM image operations transfer
  data between VGPRs and memory through the texture cache, distinguishes
  direct image loads, sampler-filtered sample operations, image atomics, and
  out-of-order completion across loads, stores, and samples.
- `rdna4/README.md:4618` through `:4639` lists VIMAGE and VSAMPLE load, store,
  atomic, sample, gather, MSAA load, GET_LOD, GET_RESINFO, and BVH families.

Rocjitsu evidence:

- Generated `IMAGE_LOAD` exposes only VDATA/RSRC operands and its execute body
  is a no-op stub at `rdna4/vimage.cpp:19` through `:32`.
- Generated `IMAGE_SAMPLE` exposes VDATA/RSRC/SAMP operands and its execute
  body says the image pipeline is not implemented at `rdna4/vsample.cpp:35`
  through `:49`.
- Generated image atomics are also stubs; for example `IMAGE_ATOMIC_SWAP`
  returns from an empty execute body at `rdna4/vimage.cpp:179` through `:192`.
- Searching `rdna4/vimage.cpp` and `rdna4/vsample.cpp` for
  `flags_ |= MEMORY_OP`, `wait_counter_type`, or memory-pipeline issue calls
  returns no matches.
- `rdna4/coverage_exceptions_rdna4.txt:27` records `image_*` as decode-only
  because image sampling/query is not simulated.

Impact:

Decoded image/sample instructions do not read or write texture memory, do not
update data VGPRs from texture results, do not perform sampler filtering, and
do not produce the documented memory-counter effects.

### RDNA4-RJ-126: RDNA4 VIMAGE/VSAMPLE address VGPR operands are not exposed

Reported by: local audit.

Manual evidence:

- `rdna4/README.md:4673` through `:4678` defines VIMAGE `VADDR0-4` and VSAMPLE
  `VADDR0-3` as address VGPR fields.
- `rdna4/README.md:4718` through `:4777` and `:4781` through `:4785` define
  no-sampler versus sampler address typing, cubemap formulas, MSAA
  restrictions, opcode-by-`DIM` address tables, and `ACNT`.
- `rdna4/README.md:4853` through `:4903` defines sampler suffix address parts,
  packed offsets, derivative counts, `A16` packing, `G16` derivative packing,
  and VADDR gathering/sequential groups.

Rocjitsu evidence:

- `VimageMachineInst` stores raw `vaddr4`, `vaddr0`, `vaddr1`, `vaddr2`, and
  `vaddr3` fields at `rdna4/machine_insts.h:154` through `:176`, and
  `VsampleMachineInst` stores raw `vaddr0` through `vaddr3` at
  `rdna4/machine_insts.h:178` through `:200`.
- Generated constructors do not publish address operands. `IMAGE_LOAD`
  publishes only `vdata` as destination and `rsrc` as source at
  `rdna4/vimage.cpp:19` through `:27`; `IMAGE_SAMPLE` publishes `vdata`,
  `rsrc`, and `samp`, but no VADDR operand at `rdna4/vsample.cpp:35` through
  `:45`.
- The generic instruction API has `implicit_uses` / `implicit_defs` hooks for
  encoded register fields that are not printed operands at
  `isa/instruction.h:215` through `:222`, but `Vimage` and `Vsample` declare no
  override at `rdna4/encodings.h:593` through `:605`.

Impact:

Disassembly, dependency analysis, and any future generic execute path cannot
see the address VGPRs, `A16`/`G16` packing, non-sequential-address gathers, or
opcode/`DIM`-dependent address counts from the generated operand lists.

### RDNA4-RJ-127: RDNA4 image data and status VGPR counts are not derived from `DMASK`, atomics, `TFE`, or `LWE`

Reported by: local audit.

Manual evidence:

- `rdna4/README.md:4666` through `:4678` and `:4684` through `:4698` define
  optional extra status VGPR writes for `TFE`/`LWE`, the returned status
  payload, and the conditional no-status case.
- `rdna4/README.md:4909` through `:4927` says image data VGPR count is
  completely determined by `DMASK`, including D16 packing and one-component
  D16 low/high behavior.
- `rdna4/README.md:4931` through `:4939` lists legal image-atomic `DMASK`
  values and says returning atomics write back to the same VDATA VGPRs.

Rocjitsu evidence:

- Representative load/sample constructors use fixed 128-bit `vdata` operands
  and one destination regardless of `DMASK`, `D16`, `TFE`, or `LWE` at
  `rdna4/vimage.cpp:19` through `:27` and `rdna4/vsample.cpp:35` through `:45`.
- `IMAGE_ATOMIC_SWAP` uses fixed 128-bit VDATA and one destination at
  `rdna4/vimage.cpp:179` through `:188`.
- `IMAGE_ATOMIC_CMPSWAP` instead uses a fixed 32-bit VDATA operand at
  `rdna4/vimage.cpp:195` through `:204`, despite the manual's 32-bit and
  64-bit atomic `DMASK` forms.
- `IMAGE_GATHER4` and related gather forms use fixed 128-bit VDATA operands at
  `rdna4/vsample.cpp:375` through `:385`, `IMAGE_GET_LOD` uses a fixed 128-bit
  VDATA operand at `rdna4/vsample.cpp:528` through `:538`, and
  `IMAGE_MSAA_LOAD` uses fixed 128-bit VDATA at `rdna4/vsample.cpp:19` through
  `:27` even though the manual says MSAA/gather-like returns use four VGPRs, or
  two when `D16=1`.

Impact:

Rocjitsu cannot currently model which consecutive VDATA VGPRs are live, which
status VGPR is conditionally written, or which VDATA range is read and written
by returning atomics.

### RDNA4-RJ-128: RDNA4 VIMAGE/VSAMPLE disassembly omits semantic fields

Reported by: local audit.

Manual evidence:

- `rdna4/README.md:4647` through `:4678` lists user-visible VIMAGE/VSAMPLE
  fields including `DIM`, `DMASK`, `R128`, `D16`, `A16`, `VDATA`, `RSRC`,
  `SCOPE`, `TH`, `TFE`, `VADDR*`, `SAMP`, `UNRM`, and `LWE`.

Rocjitsu evidence:

- `Vimage` and `Vsample` store their raw machine instruction copies at
  `rdna4/encodings.cpp:345` through `:359`.
- Their classes do not override `build_modifiers` at `rdna4/encodings.h:593`
  through `:605`, unlike neighboring memory encodings such as `Vbuffer`, which
  overrides `build_modifiers` and prints selected fields at
  `rdna4/encodings.cpp:332` through `:343`.

Impact:

Trace and disassembly output drops fields that affect addressing, data count,
sampler behavior, cache policy, fault/status reporting, resource size, and D16
packing, making distinct image/sample encodings appear equivalent.

### RDNA4-RJ-129: RDNA4 image resource descriptors are opaque and fixed-width

Reported by: local audit.

Manual evidence:

- `rdna4/README.md:4963` says the image resource/T# defines memory location,
  dimensions, tiling, and data format; resources are stored in four or eight
  consecutive SGPRs and are read by image instructions.
- `rdna4/README.md:4969` through `:4984` defines base, mip-level, format,
  width/height, `dst_sel_*`, border-color swizzle, and type fields in the
  128-bit resource.
- `rdna4/README.md:4985` through `:4998` defines depth/pitch/base-array,
  `UAV3D`, LOD warning/clamp, corner-sample mode, and all-zero unbound-resource
  behavior.

Rocjitsu evidence:

- `VimageMachineInst` and `VsampleMachineInst` preserve the raw `r128` bit and
  `rsrc` selector at `rdna4/machine_insts.h:154` through `:181`; the test
  builders also expose `r128` at `rdna4/builders.h:380` through `:455`.
- Representative ordinary image/sample constructors create fixed 256-bit
  `rsrc` operands regardless of `r128`: `IMAGE_LOAD` at `rdna4/vimage.cpp:19`
  through `:27` and `IMAGE_SAMPLE` at `rdna4/vsample.cpp:35` through `:45`.
- BVH-specific image forms use fixed 128-bit `rsrc` operands, for example
  `IMAGE_BVH_INTERSECT_RAY` at `rdna4/vimage.cpp:402` through `:410`, but that
  does not model the ordinary image T# `R128` selector.
- Generated image/sample execute paths do not read or parse the resource
  descriptor; the code generator emits only image-pipeline stubs at
  `lib/python/amdisa/codegen/_generator.py:4241` through `:4256`.

Impact:

Even before full texture-memory execution exists, rocjitsu metadata cannot
distinguish a 4-SGPR ordinary image resource from an 8-SGPR one, expose the
descriptor fields used by disassembly/dependency tools, or implement base
addressing, data-format selection, `T#.DST_SEL`, type/dimension rules,
mip/depth/pitch interpretation, LOD warning/clamp behavior, or all-zero
unbound-resource semantics from the generated instruction objects.

### RDNA4-RJ-130: RDNA4 sampler descriptors are opaque and unused

Reported by: local audit.

Manual evidence:

- `rdna4/README.md:5002` says the sampler resource/S# defines operations
  applied by sample instructions, primarily address clamping and filtering, and
  is supplied to the texture cache with every sample instruction.
- `rdna4/README.md:5006` through `:5020` defines low-word sampler fields for
  clamp modes, anisotropy, depth compare, force unnormalized coordinates,
  degamma, coordinate truncation, cube wrapping, and filter mode.
- `rdna4/README.md:5024` through `:5036` defines skip-degamma, min/max LOD,
  LOD bias, XY/Z/mip filters, and border-color controls.
- `rdna4/README.md:4638` and `:4785` say `IMAGE_MSAA_LOAD` uses the VSAMPLE
  encoding but no sampler, so `SAMP` should be set to `NULL`.

Rocjitsu evidence:

- `VsampleMachineInst` preserves raw `unorm`, `lwe`, and `samp` fields at
  `rdna4/machine_insts.h:178` through `:199`; the test builder exposes the same
  fields at `rdna4/builders.h:425` through `:470`.
- Representative sample constructors expose `samp` only as an opaque 128-bit
  SGPR operand, for example `IMAGE_SAMPLE` at `rdna4/vsample.cpp:35` through
  `:45` and `IMAGE_GET_LOD` at `:528` through `:538`.
- `IMAGE_MSAA_LOAD` correctly omits a `samp` operand despite using VSAMPLE at
  `rdna4/vsample.cpp:19` through `:27`.
- `Vsample` stores the raw machine instruction but declares no
  `build_modifiers`, `implicit_uses`, or descriptor-field helpers at
  `rdna4/encodings.h:600` through `:605` and `rdna4/encodings.cpp:353` through
  `:359`.
- The code generator emits image/sample execution stubs at
  `lib/python/amdisa/codegen/_generator.py:4241` through `:4256`, so sampler
  fields are never read when executing `IMAGE_SAMPLE*`, `IMAGE_GATHER4*`, or
  `IMAGE_GET_LOD`.

Impact:

Rocjitsu can show that a sample instruction references four SGPRs of sampler
state, but cannot currently expose or apply clamp/wrap mode, filter selection,
anisotropy, depth comparison, degamma, coordinate rounding, LOD bias/clamps,
border color, or the OR between instruction `UNORM` and
`SAMP.FORCE_UNNORMALIZED`. This affects semantic execution and also hides
sampler-dependent behavior from disassembly and dependency-style tooling.

### RDNA4-RJ-131: Buffer/image data-format conversion is not modeled

Reported by: local audit.

Manual evidence:

- Chapter 10.7 says Table 62 details all data formats usable by image and
  buffer resources at `rdna4/README.md:5040`.
- Table 62 enumerates numeric surface-format values 0-89 at
  `rdna4/README.md:5046` through `:5084`, including normalized, scaled,
  integer, float, packed 10/11-bit, sRGB, depth/stencil-like, and channel-order
  forms.
- The same table enumerates compressed and special values 109-122, 205-206,
  and 227 at `rdna4/README.md:5080` through `:5097`, including BCn, YCBCR, and
  `6E4_FLOAT` formats.

Rocjitsu evidence:

- `VbufferMachineInst` preserves the raw instruction-side 7-bit `format` field
  at `rdna4/machine_insts.h:134` through `:152`, and the test builder writes it
  at `rdna4/builders.h:343` through `:377`.
- Searching rocjitsu source for representative Table 62 names such as `BC1`,
  `YCBCR`, `6E4`, `10_10_10_2`, `GB_GR`, `32_FLOAT_CLAMP`, and `X24` returns
  no format enum/table or conversion model.
- Formatted buffer loads and stores expose operands but throw
  `UnimplementedInst`, for example `BUFFER_LOAD_FORMAT_X` at
  `rdna4/vbuffer.cpp:36` through `:55` and `BUFFER_STORE_FORMAT_X` at `:120`
  through `:139`.
- Typed-buffer formatted loads and stores likewise throw, for example
  `TBUFFER_LOAD_FORMAT_X` at `rdna4/vbuffer.cpp:2555` through `:2574` and
  `TBUFFER_STORE_FORMAT_X` at `:2639` through `:2658`.
- Image loads and stores are still minimal stubs, for example `IMAGE_LOAD` at
  `rdna4/vimage.cpp:19` through `:33` and `IMAGE_STORE` at `:115` through
  `:129`.
- RDNA4 VBUFFER address calculation reads descriptor base-address fields but
  does not inspect data format, destination selection, or conversion metadata at
  `rdna4/addr_calc.cpp:161` through `:189`.

Impact:

Rocjitsu preserves raw typed-buffer format bits, but cannot execute or model
the data-format taxonomy in Chapter 10.7. Descriptor- and instruction-selected
conversion for normalized/scaled/integer/float, sRGB, depth/stencil-like,
compressed, YCBCR, and special packed formats is absent for buffer, typed-buffer,
and image paths. This extends the formatted-buffer execution gaps already noted
for Chapter 9 into the shared buffer/image data-format table from Chapter 10.7.

### RDNA4-RJ-132: SAMPLEcnt and BVHcnt are not independent counters

Reported by: local audit.

Manual evidence:

- Chapter 10.8 says shader authors must wait for VMEM read completion using
  `LOADcnt`/`STOREcnt` and that ray-tracing image BVH instructions are tracked
  with `BVHcnt` at `rdna4/README.md:5103` through `:5105`.
- The referenced dependency section says samples stay in order with samples but
  are unordered with loads, stores, and BVH ops at `rdna4/README.md:2133`.
- The memory-counter table defines separate `LOADcnt`, `SAMPLEcnt`, `BVHcnt`,
  and `STOREcnt` rows, with `SAMPLEcnt` incrementing for each sample instruction
  and `BVHcnt` incrementing/decrementing for each BVH instruction at
  `rdna4/README.md:2181` through `:2188`.

Rocjitsu evidence:

- `WaitCounterType` has `VMCNT`, `VSCNT`, `LOADCNT`, `STORECNT`, `DSCNT`,
  `KMCNT`, `TENSORCNT`, and `ASYNCCNT`, but no `SAMPLECNT` or `BVHCNT`, at
  `vm/amdgpu/wait_counters.h:23` through `:34`.
- `Wavefront::set_wait_counter` handles `wait_samplecnt` and `wait_bvhcnt` by
  assigning `wait_target_.vmcnt`, so both waits target the load/vmcnt alias at
  `vm/amdgpu/wavefront.h:396` through `:399`.
- Generated RDNA4 `S_WAIT_SAMPLECNT` and `S_WAIT_BVHCNT` execute paths call
  those names at `rdna4/sopp.cpp:517` through `:545`.

Impact:

`S_WAIT_SAMPLECNT` and `S_WAIT_BVHCNT` cannot model the manual's independent
sample and BVH queues. A wait for samples or BVH work can be satisfied or blocked
by ordinary load-count state instead, and the manual rule that samples are
ordered with samples but unordered with loads/stores/BVH cannot be represented.

### RDNA4-RJ-133: Image/sample/BVH instructions do not issue countered memory work

Reported by: local audit.

Manual evidence:

- Chapter 10.8 says VMEM read results require explicit waits before use and
  image BVH instructions are tracked with `BVHcnt` at `rdna4/README.md:5103`
  through `:5105`.
- The memory-counter table assigns image, buffer, flat, scratch, global loads
  and atomic-with-return to `LOADcnt`; sample/gather to `SAMPLEcnt`; BVH to
  `BVHcnt`; and stores/atomic-without-return to `STOREcnt` at
  `rdna4/README.md:2181` through `:2188` and `:2217` through `:2227`.

Rocjitsu evidence:

- The code generator returns image-pipeline stubs for `image_load`,
  `image_store`, `image_atomic`, `image_sample`, `image_query`, and `image_bvh`
  at `lib/python/amdisa/codegen/_generator.py:4241` through `:4256`, before any
  `VectorMemState` or wait-counter assignment is emitted.
- The generator's wait-counter map includes scalar, flat, buffer, typed-buffer,
  global, and DS classes but no image/sample/BVH classes at
  `lib/python/amdisa/codegen/_generator.py:4411` through `:4570`.
- Representative generated sample code exposes operands but does not create
  memory state: `IMAGE_MSAA_LOAD` and `IMAGE_SAMPLE` return from stubs at
  `rdna4/vsample.cpp:19` through `:50`.
- Representative generated BVH code either returns from an image-pipeline stub
  or throws `UnimplementedInst`, for example at `rdna4/vimage.cpp:402` through
  `:430`.
- By contrast, implemented buffer paths create `VectorMemState` and set
  `LOADCNT`/`STORECNT`, for example `BUFFER_LOAD_U8` at
  `rdna4/vbuffer.cpp:389` through `:399` and `BUFFER_STORE_B8` at `:631`
  through `:650`.

Impact:

Even aside from full image execution, image loads, image stores, image atomics,
sample/gather/MSAA loads, image queries, and BVH instructions do not currently
increment or retire the counters that Chapter 10.8 requires. Wait instructions
can be decoded and executed, but there is no outstanding image/sample/BVH work
for them to observe.

### RDNA4-RJ-134: VMEM source-read timing, counter overflow, and in-order retirement are not modeled

Reported by: local audit.

Manual evidence:

- Chapter 10.8 says VM issue schedules reads of address and store-data VGPRs to
  the texture unit, and an ALU instruction that attempts to overwrite this data
  before it has been sent is stalled at `rdna4/README.md:5101`.
- The memory-counter section says hardware prevents counter overflow by
  stalling issue at `rdna4/README.md:2169`.
- It also says load data can be written to VGPRs out of order while the counter
  decrement still reflects in-order completion, stores to different addresses
  from the same wave are not kept in order, and counters count instructions
  rather than threads at `rdna4/README.md:2171` through `:2175`.

Rocjitsu evidence:

- Implemented store paths read source VGPR data into `VectorMemState` during
  `execute_impl`, for example `BUFFER_STORE_B8` reads `vdata` with
  `RegisterAccess::read_vgpr` before `set_data` at `rdna4/vbuffer.cpp:631`
  through `:650`; there is no pending texture-unit source-read window for later
  ALU writes to stall against.
- `WaitCounters::increment` saturates each counter with `std::min` at
  `vm/amdgpu/wait_counters.h:76` through `:105`, rather than stalling issue
  before overflow.
- `MemoryPipeline::issue` increments one counter when an instruction is issued
  and releases that same counter from the completion callback at
  `vm/amdgpu/memory_pipeline.h:62` through `:88` and `:104` through `:105`;
  the counter implementation is a count, not a per-counter FIFO that enforces
  same-type in-order decrement while allowing data writeback to occur
  out-of-order.

Impact:

Rocjitsu can model simple outstanding-count waits for implemented memory
instructions, but it cannot reproduce Chapter 10.8's source-read stall timing,
hardware issue stall on counter overflow, or the exact in-order counter
retirement model for same-type VMEM operations. A timing backend that completes
deferred accesses out of order would release counters in callback order unless
it adds its own ordering layer.

### RDNA4-RJ-135: BVH address VGPR roles and A16 footprints are not represented

Reported by: local audit; Poincare subagent.

Manual evidence:

- Chapter 10.9 says the 32-bit and 64-bit BVH ray instructions support `A16`
  for `ray_dir` and `ray_inv_dir`, but `image_bvh_dual_intersect_ray` and
  `image_bvh8_intersect_ray` do not support `A16=1`, at
  `rdna4/README.md:5124`.
- The prototypes show `image_bvh_intersect_ray` using `vgpr_a[11]` or
  `vgpr_a[8]` with `A16=1`, `image_bvh64_intersect_ray` using `vgpr_a[12]` or
  `vgpr_a[9]` with `A16=1`, and the dual/BVH8 forms using `vgpr_a[12]` and
  `vgpr_a[11]` at `rdna4/README.md:5129` through `:5134`.
- The BVH VADDR tables define the ray/node field layout, packed f16 `A16`
  layout, dual/BVH8 input fields, and dual/BVH8 ray origin/direction
  input/output parameters at `rdna4/README.md:5154` through `:5173` and
  `:5189` through `:5197`.
- Section 10.9.2 says `VADDR0` through `VADDR4` are BVH component groups, and
  `ADDR4` is unused when `A16=1`, at `rdna4/README.md:5232` through `:5241`.

Rocjitsu evidence:

- `VimageMachineInst` preserves the raw `vaddr4`, `vaddr0`, `vaddr1`,
  `vaddr2`, and `vaddr3` fields at `rdna4/machine_insts.h:154` through
  `:176`, and the builder exposes those raw fields at `rdna4/builders.h:381`
  through `:420`.
- The generated BVH classes only own `vdata` and `rsrc` operands at
  `rdna4/vimage.h:209` through `:238`; no VADDR operand or implicit-use/def
  hook is present there.
- Representative BVH constructors publish only `vdata` and `rsrc` through
  `src_operands_`/`dst_operands_`: `IMAGE_BVH_INTERSECT_RAY` at
  `rdna4/vimage.cpp:402` through `:410`, `IMAGE_BVH64_INTERSECT_RAY` at
  `:417` through `:425`, `IMAGE_BVH_DUAL_INTERSECT_RAY` at `:433` through
  `:441`, and `IMAGE_BVH8_INTERSECT_RAY` at `:449` through `:457`.
- The generic disassembler prints only registered operands and optional
  `build_modifiers` output at `isa/instruction.h:240` through `:255`; `Vimage`
  stores the raw instruction but does not override `build_modifiers` at
  `rdna4/encodings.h:593` through `:600` and `rdna4/encodings.cpp:345`
  through `:351`.

Impact:

Raw BVH VADDR bits are retained, but rocjitsu operand metadata, disassembly,
dependency analysis, and future generic execution cannot see which VGPR groups
are read, which ray-origin/ray-direction VGPRs may be written back, how `A16`
changes the address footprint, or which BVH opcodes disallow `A16`.

### RDNA4-RJ-136: BVH descriptor, return data, and instance-node semantics are not modeled

Reported by: local audit; Poincare subagent.

Manual evidence:

- BVH instructions receive a BVH resource descriptor, fetch a BVH node, test the
  node, return intersection results, and update ray origin/direction if an
  instance node is tested at `rdna4/README.md:5137` through `:5148`.
- Result data depends on node type and descriptor mode: box nodes return sorted
  child pointers, triangle nodes return hit data or barycentrics, dual/BVH8
  forms can return instance-node data and ShapeID/GeoID payloads, and wide sort
  can intermix results at `rdna4/README.md:5175`, `:5199` through `:5207`, and
  `:5213` through `:5215`.
- The BVH texture descriptor defines sorting, growth, bounds, node-type,
  instance, pointer-flag, triangle-return-mode, and required type fields at
  `rdna4/README.md:5243` through `:5267`; the return-mode table is at
  `rdna4/README.md:5269` through `:5279`.

Rocjitsu evidence:

- BVH constructors expose a fixed 128-bit `rsrc` operand, for example
  `IMAGE_BVH_INTERSECT_RAY` at `rdna4/vimage.cpp:402` through `:410`, but they
  do not parse any BVH descriptor fields.
- `IMAGE_BVH_INTERSECT_RAY` currently executes as an image-pipeline no-op stub
  at `rdna4/vimage.cpp:413` through `:415`.
- The remaining RDNA4 BVH execute bodies throw `UnimplementedInst`:
  `IMAGE_BVH64_INTERSECT_RAY` at `rdna4/vimage.cpp:428` through `:430`,
  `IMAGE_BVH_DUAL_INTERSECT_RAY` at `:444` through `:446`, and
  `IMAGE_BVH8_INTERSECT_RAY` at `:460` through `:462`.
- The generator emits only image-pipeline stubs for the `image_bvh` semantic
  class at `lib/python/amdisa/codegen/_generator.py:4241` through `:4256`.

Impact:

Rocjitsu cannot derive BVH node addresses or bounds from the descriptor, cannot
interpret descriptor-controlled sorting, pointer flags, instance nodes, or
triangle return mode, cannot write the documented result payloads, and cannot
perform the conditional ray-origin/ray-direction updates described by Chapter
10.9.

### RDNA4-RJ-137: BVH-only encoding restrictions are not checked or surfaced

Reported by: Poincare subagent.

Manual evidence:

- Chapter 10.9 says the dual and BVH8 instructions do not support `A16=1` at
  `rdna4/README.md:5124`.
- The BVH restriction block requires `DMASK=0xf`, `D16=0`, `R128=1`,
  `UNRM=1`, `DIM=0`, `LWE=0`, `TFE=0`, and `SSAMP=0` at
  `rdna4/README.md:5219` through `:5226`.
- The instruction-definition chapter says these restrictions are a
  software/compiler contract; hardware is not required to enforce them and may
  ignore invalid values or produce undefined behavior at
  `rdna4/README.md:28612` through `:28620`.

Rocjitsu evidence:

- `VimageMachineInst` carries the raw fields needed for the check, including
  `dim`, `r128`, `d16`, `a16`, `dmask`, `tfe`, and `vaddr*` at
  `rdna4/machine_insts.h:154` through `:176`.
- `Decoder::subDecodeVimage` selects the instruction solely from `op.op` at
  `rdna4/decoder.cpp:2943` through `:2945`; the BVH decode constructors are
  simple `make_unique` wrappers at `:3043` through `:3058`.
- BVH constructors do not validate the BVH-only field values or expose them as
  modifiers, for example `IMAGE_BVH_INTERSECT_RAY` at `rdna4/vimage.cpp:402`
  through `:410` and `IMAGE_BVH_DUAL_INTERSECT_RAY` at `:433` through `:441`.
- `Vimage` does not override `build_modifiers`, so BVH disassembly also hides
  these restriction-bearing fields at `rdna4/encodings.h:593` through `:600`
  and `rdna4/encodings.cpp:345` through `:351`.

Impact:

Rocjitsu can retain the raw bits, but it cannot flag or expose BVH encodings
that violate the software/compiler contract, including `A16=1` on dual/BVH8 or
non-BVH field values that hardware treats as undefined. Distinct legal and
invalid BVH encodings can collapse to the same instruction-level view.

### RDNA4-RJ-138: PRT enablement and extra status VGPR behavior are not modeled

Reported by: local audit; Dalton subagent.

Manual evidence:

- Chapter 10.10 says a PRT texture fetch that accesses a missing MIP returns an
  extra DWORD status in VGPRs, NACK writes a nonzero value for each failing
  thread, the value may represent the requested LOD, and shaders must allocate,
  initialize, and check that extra VGPR at `rdna4/README.md:5287`.
- Chapter 10.10 says PRT is enabled by nonzero texture-resource
  `MIN_LOD_WARN`, normal textures cannot NACK, and PRT status is written to
  `DST_VGPR+Num_VGPRS` after the normal fetch return at
  `rdna4/README.md:5289`.
- The image-resource descriptor defines `min_lod_warn` in bits 177:165 and
  `min_lod` in bits 198:186 at `rdna4/README.md:4989` and `:4996`.
- Chapter 3.3.4 says destination out-of-range checks include the extra PRT VGPR
  and nullify the fetch if that extra VGPR would be out of range, even if the
  texture system does not ultimately return status, at `rdna4/README.md:907`
  through `:910`.
- The image `DMASK` field description says `DMASK==0` drops the fetch and TFE
  status is not generated at `rdna4/README.md:4656`.

Rocjitsu evidence:

- `VimageMachineInst` and `VsampleMachineInst` store the raw `dmask`, `tfe`,
  `lwe`, and `rsrc` selector bits at `rdna4/machine_insts.h:154` through
  `:196`, so decode preserves the instruction fields needed for PRT-related
  checks.
- Generated ordinary image/sample constructors expose fixed 256-bit `rsrc`
  operands and do not parse texture-resource descriptor fields; representative
  `IMAGE_LOAD` and `IMAGE_SAMPLE` constructors are at `rdna4/vimage.cpp:19`
  through `:27` and `rdna4/vsample.cpp:35` through `:45`.
- Those same constructors publish only the ordinary fixed-size `vdata`
  destination and no conditional extra PRT status destination, while their
  execute bodies are image-pipeline stubs at `rdna4/vimage.cpp:30` through
  `:32` and `rdna4/vsample.cpp:48` through `:49`.
- The code generator emits only image-pipeline stubs for `image_load`,
  `image_store`, `image_atomic`, `image_sample`, `image_query`, and
  `image_bvh` semantic classes at
  `lib/python/amdisa/codegen/_generator.py:4241` through `:4256`.
- The base implicit-use/implicit-def hooks are empty at
  `isa/instruction.h:215` through `:222`, and the generated VIMAGE/VSAMPLE
  classes do not add an implicit def for the status VGPR.
- `coverage_exceptions_rdna4.txt:27` marks `image_*` as decode-only because
  image sampling/query is not simulated.

Impact:

Rocjitsu can decode the raw PRT-related instruction bits, but it cannot decide
whether the referenced resource enables PRT, cannot model normal-texture versus
PRT NACK behavior, cannot expose or write the dynamic `DST+Num_VGPRS` status
VGPR, cannot enforce the extra destination out-of-range nullification rule, and
cannot model the `DMASK==0` TFE-status suppression. This is partly covered by
the broader image/sample findings `RDNA4-RJ-125`, `RDNA4-RJ-127`, and
`RDNA4-RJ-129`; this entry records the Chapter 10.10-specific contract.

### RDNA4-RJ-139: Flat aperture routing is whole-wave and uses post-offset addresses

Reported by: local audit; Wegener subagent.

Manual evidence:

- Chapter 11 says Flat per-thread addresses may resolve to global, scratch,
  LDS, or invalid memory, at `rdna4/README.md:5295` through `:5303`.
- The Flat aperture check is performed on the VGPR value before `IOFFSET` is
  added, and LDS-targeted lanes use a logical-address bounds/remap rule, at
  `rdna4/README.md:5412` through `:5418`.
- Flat scratch mapping uses the per-wave `SCRATCH_BASE` state at
  `rdna4/README.md:5408` through `:5410`.

Rocjitsu evidence:

- `flat_calculate_addresses(const VflatMachineInst&, ...)` sign-extends
  `IOFFSET`, adds it into `addr`, and then checks the private aperture at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.cpp:74` and
  `:96` through `:99`.
- `ComputeUnitCore::route_memory_inst` chooses the Flat LDS/global route from
  the first active lane only at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:263` through `:274`,
  then retags and remaps every active lane to LDS if that probe lane is in the
  shared aperture at `:275` through `:281`; otherwise the whole instruction
  remains on the global pipeline at `:286` through `:294`.
- The existing `RDNA4-RJ-048` entry covers the separate dual-counter issue for
  Flat instructions.

Impact:

Mixed Flat waves with some lanes in LDS, scratch, global, or invalid apertures
cannot be modeled correctly. Addresses that cross an aperture only because of
`IOFFSET` are classified differently from the manual's pre-offset rule, and LDS
logical-address remapping/OOB behavior is not represented.

### RDNA4-RJ-140: Scratch VGPR and SGPR address offsets are zero-extended

Reported by: Wegener subagent.

Manual evidence:

- The Chapter 11 field table says Scratch `VADDR` is a signed byte offset when
  `SVE=1`, and Scratch `SADDR` supplies a signed 32-bit address component, at
  `rdna4/README.md:5318` through `:5325`.

Rocjitsu evidence:

- The Scratch helper correctly sign-extends the 24-bit immediate offset at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.cpp:139`.
- The same helper stores `SADDR` in a `uint32_t`, stores the optional `VADDR`
  in a `uint32_t`, and adds both directly into the 64-bit scratch address at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.cpp:142` through
  `:157`.

Impact:

Negative Scratch VGPR or SGPR offsets are interpreted as large positive
offsets. Tests using only non-negative offsets can pass while signed addressing
near the start of a private scratch slice reads or writes the wrong address.

### RDNA4-RJ-141: VFLAT/VGLOBAL/VSCRATCH metadata collapses address-mode fields

Reported by: local audit.

Manual evidence:

- Chapter 11 gives `SADDR`, `SVE`, `VADDR`, `IOFFSET`, `SCOPE`, and `TH`
  instruction-field meanings for Flat, Global, and Scratch addressing at
  `rdna4/README.md:5318` through `:5325`.

Rocjitsu evidence:

- Generated machine-instruction structs preserve the raw RDNA4 VFLAT,
  VSCRATCH, and VGLOBAL fields at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h:217`
  through `:268`.
- Unlike `Vbuffer::build_modifiers`, which prints `offen`, `idxen`, `offset`,
  and `nv` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/encodings.cpp:332`
  through `:342`, `Vflat::build_modifiers`, `Vscratch::build_modifiers`, and
  `Vglobal::build_modifiers` print only `nv` at `:376` through `:410`.
- Representative generated operand metadata exposes `GLOBAL_LOAD_B32` as both
  a 64-bit `VADDR` and a 64-bit `SADDR` source unconditionally at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vglobal.cpp:136` through
  `:145`, and exposes `SCRATCH_LOAD_B32` as both `VADDR` and `SADDR`
  unconditionally at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vscratch.cpp:136` through
  `:145`.
- Runtime address calculation does use `SADDR` and `SVE` to choose the number
  of VGPRs read at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.cpp:118` and
  `:150`, so this gap is in operand/disassembly metadata rather than the basic
  runtime source-count selection.

Impact:

Disassembly and def-use users cannot see nonzero `IOFFSET`, `SVE`, `SCOPE`, or
`TH`, and operand metadata over-reports sources for encodings where the manual
makes source presence or width conditional. Distinct instruction encodings can
therefore display as the same instruction and expose misleading source lists.

### RDNA4-RJ-142: Special Flat/Global atomics collapse to ordinary RMW operations

Reported by: Wegener subagent.

Manual/XML evidence:

- Flat and Global sub-clamp atomics clamp underflow to zero in their manual
  pseudocode at `rdna4/README.md:29391` through `:29404` and `:30325` through
  `:30338`.
- XML descriptions for `FLAT_ATOMIC_COND_SUB_U32` and
  `GLOBAL_ATOMIC_COND_SUB_U32` say the subtraction happens only if the memory
  value is greater than or equal to the data value, at
  `amdgpu_isa_rdna4.xml:50122` through `:50123` and `:53554` through `:53555`.
- `GLOBAL_ATOMIC_ORDERED_ADD_B64` operates on an `(ID, value)` pair and only
  updates the value when the memory ID matches the shader ID, at
  `rdna4/README.md:30744` through `:30765`.
- Table 67 also marks the conditional-sub, global sub-clamp, and ordered-add
  forms as return-preOp-only at `rdna4/README.md:5388` through `:5394`.

Rocjitsu evidence:

- `FlatAtomicSubClampU32Vflat` and `FlatAtomicCondSubU32Vflat` both set
  `d->atomic_op = amdgpu::AtomicOp::SUB` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vflat.cpp:840` through
  `:850` and `:1744` through `:1754`.
- `GlobalAtomicSubClampU32Vglobal` and `GlobalAtomicCondSubU32Vglobal` also
  use `AtomicOp::SUB` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vglobal.cpp:1007` through
  `:1017` and `:1966` through `:1976`.
- `GlobalAtomicOrderedAddB64Vglobal` sets `AtomicOp::ADD` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vglobal.cpp:2387`
  through `:2397`.
- The shared integer atomic helper implements `AtomicOp::SUB` as
  `old_val - src_val`, and `AtomicOp` has no sub-clamp, conditional-sub, or
  ordered-add variants, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:247` through
  `:259` and `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:45` through
  `:68`.
- These generated execute bodies also derive `d->is_load` from
  `gfx12_atomic_returns(inst_.th)` instead of forcing the manual's
  return-preOp-only forms to return; the broader `TH` return-bit and
  wait-counter issues are covered separately by `RDNA4-RJ-020` and
  `RDNA4-RJ-050`.

Impact:

Sub-clamp can wrap instead of clamping to zero, conditional-sub can modify
memory when it should leave it unchanged, ordered-add ignores the ID-match
protocol and paired low/high 32-bit update semantics, and these return-only
forms are still allowed to behave as no-return atomics.

### RDNA4-RJ-143: 64-bit Flat atomics into scratch do not raise `MEMVIOL`

Reported by: Wegener subagent.

Manual evidence:

- Chapter 11 says Flat atomics that map into scratch support 4-byte atomics,
  while 8-byte atomics return `MEMVIOL`, at `rdna4/README.md:5408`.

Rocjitsu evidence:

- The Flat address helper maps private-aperture addresses to scratch backing
  addresses at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.cpp:96` through
  `:99`.
- `ComputeUnitCore::route_memory_inst` explicitly leaves scratch-targeting
  Flat instructions on the global path at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:272` through `:294`.
- Representative 64-bit Flat atomics set `elem_size = 8` and call
  `flat_calculate_addresses` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vflat.cpp:1220` through
  `:1230`, `:1260` through `:1270`, and `:1544` through `:1554`.
- `GlobalMemPipeline::initiate_access` sends any vector atomic to
  `execute_atomic_rmw` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:475` through
  `:485`, and that helper performs ordinary 8-byte RMW at `:354` through
  `:373`.

Impact:

A private-address `flat_atomic_*_b64` can update scratch backing memory and
return old data instead of reporting the manual's `MEMVIOL` result.

### RDNA4-RJ-144: Flat `SADDR` is treated as an active SGPR base

Reported by: local audit.

Manual evidence:

- The Chapter 11 field table says `VADDR` specifies the address for `FLAT_*`
  instructions, while `SADDR` is explicitly "Unused" for Flat at
  `rdna4/README.md:5318` and `:5324`.

Rocjitsu evidence:

- The shared GFX12 address helper treats non-null `SADDR` values as active at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.cpp:22` through
  `:25`.
- `flat_calculate_addresses(const VflatMachineInst&, ...)` reads a 64-bit SGPR
  base when `has_saddr(inst.saddr)` is true, then reads one VGPR instead of a
  64-bit VGPR pair and adds the SGPR base into the final Flat address at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.cpp:75` through
  `:86` and `:90` through `:96`.

Impact:

A Flat instruction with non-null bits in the raw `SADDR` field can address a
different location in rocjitsu than hardware, and the emulated source footprint
can shrink from a 64-bit `VADDR` pair to one VGPR.

### RDNA4-RJ-145: Scratch swizzle and ST legality rules are not modeled

Reported by: local audit.

Manual evidence:

- Chapter 11.2 defines Scratch `SV`, `SS`, `SVS`, and `ST` as
  `SCRATCH_BASE + SWIZZLE(offset, ThreadID)` forms at `rdna4/README.md:5470`
  through `:5475`.
- The combined offsets inside `SWIZZLE()` must be non-negative, and SGPR/VGPR
  offsets are signed 32-bit byte offsets at `rdna4/README.md:5477` through
  `:5479`.
- Scratch `ST` mode requires `IOFFSET` alignment by payload size: 4-byte
  aligned for one DWORD and 16-byte aligned for four DWORDs at
  `rdna4/README.md:5481`.

Rocjitsu evidence:

- The RDNA4 Scratch address helper uses a private lane-slice model:
  `scratch_base + lane * lane_stride + vaddr + saddr_val + offset`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.cpp:132` through
  `:158`.
- The helper has no `SWIZZLE(...)` transformation, no non-negative combined
  offset check, and no distinction for `ST` payload alignment; generated
  Scratch load/store bodies set `elem_size`, `num_elems`, and call the helper,
  for example `SCRATCH_LOAD_B32` at `rdna4/vscratch.cpp:150` through `:160`,
  `SCRATCH_LOAD_B128` at `:231` through `:240`, `SCRATCH_STORE_B32` at `:330`
  through `:338`, and block scratch stores at `:757` through `:765`.
- Existing RDNA4 address tests cover positive `SV`, `SVS`, and `SS` address
  cases at `tests/shared_infra_test.cpp:3114` through `:3143`, but do not
  exercise swizzle layout, negative combined offsets, or `ST` alignment.

Impact:

Rocjitsu can select which Scratch sources participate, but it does not model
the hardware's swizzled scratch layout or the legality constraints around
negative combined offsets and unaligned thread-private transfers. Programs
that alias scratch through Flat/private addressing or depend on edge-case
scratch legality can observe different behavior.

### RDNA4-RJ-146: Flat LDS addressing misses U17 wrap and allocated-size range checks

Reported by: local audit.

Manual evidence:

- Flat aperture classification uses only the base VGPR address, not `IOFFSET`,
  at `rdna4/README.md:5491`.
- For LDS-addressed flat lanes, Chapter 11.2 defines
  `LDS_ADDR.U17 = VGPR(addr)[16:0] + IOFFSET[16:0]`; LDS address math is
  truncated and may wrap without being detected as out-of-range, and the only
  range check is `LDS_ADDR.U17 < LDS_SIZE` with the address zero-extended for
  the check at `rdna4/README.md:5507` through `:5514`.

Rocjitsu evidence:

- `flat_calculate_addresses(const VflatMachineInst&, ...)` adds `IOFFSET` into
  the 64-bit address before any aperture routing at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.cpp:74` and
  `:96` through `:99`.
- `ComputeUnitCore::route_memory_inst` detects the shared aperture from the
  first active lane's already-computed address, then retags every active lane
  to LDS and remaps with `(addr - shared_aperture_base) + wf.lds_base()` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:263` through `:281`.
- The local-memory pipeline passes those remapped addresses directly to
  `Lds::vector_load` and `Lds::vector_store` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:520` through
  `:527` and `:590` through `:594`.
- `Lds::vector_load` and `vector_store` cast the address to `uint32_t` and
  check only against the backing LDS object size, returning zero or dropping
  stores on backing-memory OOB at `vm/amdgpu/lds.h:136` through `:168`.

Impact:

Flat LDS accesses are not classified before `IOFFSET`, are not reduced through
the manual's U17 add/truncate path, and are not range-checked against the
wave's allocated `LDS_SIZE`. A flat LDS address that should wrap within U17,
fault against the wave allocation, or remain classified as LDS despite an
offset crossing the aperture can be routed or addressed differently.

### RDNA4-RJ-147: Flat hole addresses fall through to global memory

Reported by: Harvey subreviewer; local audit.

Manual evidence:

- Chapter 11.2 defines `isHole` for Flat aperture classification and says
  `Hole` produces Memory Violation at `rdna4/README.md:5495` through `:5515`.
- The Memory Aperture Query section defines the hole as the illegal address
  range outside sign-extended segment 0/1 and outside shared/private
  apertures at `rdna4/README.md:2580` through `:2600`.

Rocjitsu evidence:

- `flat_calculate_addresses(const VflatMachineInst&, ...)` maps private
  aperture addresses into scratch backing memory but does not classify the
  hole range or record a per-lane violation state at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.cpp:96` through
  `:99`.
- `ComputeUnitCore::route_memory_inst` only special-cases the shared aperture;
  if the first active lane is not in the configured shared range, the
  instruction remains `GLOBAL_MEM` and is issued to the global pipeline at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:263` through `:294`.
- `GlobalMemPipeline::initiate_access` then performs ordinary global loads or
  stores for those addresses at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:488` through
  `:495`.
- `VectorMemState` has no memory-violation or per-lane aperture-result field
  at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:87` through `:112`.

Impact:

Flat addresses in the manual's hole range can be sent to the global-memory
pipeline and observe backing memory behavior instead of raising Memory
Violation. This is separate from `RDNA4-RJ-139`'s whole-wave/post-`IOFFSET`
routing issue because even a uniformly hole-targeted wave has no invalid
aperture path.

### RDNA4-RJ-148: Chapter 11.3 memory-error state and side effects are absent

Reported by: local audit.

Manual evidence:

- Chapter 11.3 says Cache and LDS can report bad-address errors for invalid
  addresses outside any aperture, writes to read-only global pages, misaligned
  data, and LDS out-of-range addresses outside `[0, LDS_SIZE-1]`, at
  `rdna4/README.md:5517` through `:5526`.
- The bad-address policy says stores outside the valid range do not store,
  reads return zero, invalid-address aperture checking happens before adding
  address offsets, and other checks happen after offsets are added at
  `rdna4/README.md:5527`.
- Addressing errors from either LDS or VMEM set the wave's `MEMVIOL` bit and
  cause an exception/trap at `rdna4/README.md:5529`.

Rocjitsu evidence:

- `VectorMemState` tracks per-lane addresses, lane masks, payload sizes,
  memory type, and atomic kind, but has no `MEMVIOL`, trap, bad-address cause,
  read-only-page, or per-lane memory-error field at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:87` through `:112`.
- RDNA4 `StatusReg` models `TRAP`, `TRAP_EN`, and `ECC_ERR` status bits, but
  no `MEMVIOL` or wave exception flag storage at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/isa.h:23` through `:60`;
  the generated `V_CLREXCP` helpers are no-ops at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:3795`
  through `:3799`.
- Global/Scratch/Flat memory operations that remain on the global pipeline use
  ordinary vector loads and stores for every lane left in `lane_mask` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:488` through
  `:495`; load completion only zeroes lanes that an address calculator removed
  from `lane_mask` at `:139` through `:169`.
- The process page-table entry stores only a host pointer and PTE `Mtype` at
  `lib/rocjitsu/src/rocjitsu/kmd/linux/kfd_process.h:207` through `:214`, so
  read-only page permissions are not represented in the VM state used by vector
  memory.
- The external KFD allocation flags include a `WRITABLE` bit at
  `lib/rocjitsu/external_headers/hsa_headers/linux/uapi/kfd_ioctl.h:1020`
  through `:1024`, but `pte_mtype_for_flags` derives only cache `Mtype` at
  `lib/rocjitsu/src/rocjitsu/kmd/linux/simulated_kfd.cpp:51` through `:61`.
- `GpuMemory::read_block` falls through to client memory or sparse backing
  memory for untranslated addresses, and `write_block` writes to translated
  host memory, client memory, or sparse backing memory, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/gpu_memory.h:115` through `:123` and
  `:132` through `:140`.
- The sparse backing itself zero-fills missing pages on reads, but writes
  allocate/update sparse pages at
  `lib/simdojo/include/simdojo/components/sparse_memory.h:208` through `:223`,
  so an unmapped invalid store is not naturally dropped.
- Searches in the RDNA4 generated code and AMDGPU VM runtime found no
  `alignment_mode`, `SH_MEM_CONFIG`, `memviol`, read-only-page, or
  write-protection handling for Chapter 11 Flat/Global/Scratch memory errors.

Impact:

Rocjitsu can model some buffer-style OOB load zeroing through `lane_mask`, and
the LDS backing object returns zero or drops stores when backing LDS object
bounds are exceeded, but Chapter 11.3's common error contract is not modeled for
Flat/Global/Scratch. Invalid global addresses can read/write backing memory,
read-only global writes are not rejected, misaligned accesses are not converted
to forced alignment or `MEMVIOL`, and memory errors do not set wave exception
state or enter a trap path. Address-specific pieces of this are tracked in
`RDNA4-RJ-145`, `RDNA4-RJ-146`, and `RDNA4-RJ-147`; this entry records the
missing common error-state and side-effect machinery.

### RDNA4-RJ-149: D16 memory-load partial writes are hidden from def-use metadata

Reported by: local audit.

Manual evidence:

- Chapter 11.4 says `"D16"` instructions use only 16 bits of the VGPR,
  `D16_HI` instructions read or write only the high 16 bits, and `D16`
  instructions use the low 16 bits, at `rdna4/README.md:5535`.

Rocjitsu evidence:

- Generated D16 memory-load execution does mark the partial-write intent. For
  example, `FlatLoadD16U8Vflat` sets `d16_lo` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vflat.cpp:456` through
  `:464`, while `FlatLoadD16HiU8Vflat` sets `d16_hi` at `:535` through `:543`.
  Global and Scratch D16 loads use the same flags at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vglobal.cpp:486` through
  `:494`, `:571` through `:579`,
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vscratch.cpp:486` through
  `:494`, and `:571` through `:579`.
- `vector_complete` preserves the old unselected half for D16 loads when SRAM
  ECC is not active, including OOB-zero completion, by reading the old
  destination VGPR and merging low/high halves at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:158` through `:160`
  and `:181` through `:193`.
- The generated instruction metadata still exposes D16 loads as ordinary
  32-bit output-only `VDST` operands. Representative constructors use
  `vdst(32, OPR_VGPR)` with `dst_operands_[0] = &vdst` and only address
  sources at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vflat.cpp:444`
  through `:454`, `:523` through `:533`,
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vglobal.cpp:472` through
  `:483`, and `:557` through `:568`; the same pattern exists for Scratch.
- Searches found no `implicit_uses` override in RDNA4 `vflat`, `vglobal`, or
  `vscratch`. `InstDefUse` collects uses from source operands and
  `implicit_uses` at `lib/rocjitsu/src/rocjitsu/analysis/def_use_chain.cpp:28`
  through `:44`, while the base `Instruction::implicit_uses` is a no-op at
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:211` through `:219`.
- The generic register-granularity contract says narrower-than-32 writes
  read-modify-write the register at
  `lib/rocjitsu/src/rocjitsu/isa/isa_traits.h:17` through `:23`, and
  liveness relies on uses remaining visible for read-modify-write instructions
  at `lib/rocjitsu/src/rocjitsu/analysis/liveness.cpp:196` through `:212`.

Impact:

Runtime execution preserves the untouched half, but `InstDefUse` and liveness
consumers can treat these instructions as full VGPR overwrites. A transformation
or analysis that depends on the old destination being live before a D16 memory
load will miss that dependency unless it has separate D16-specific knowledge.

### RDNA4-RJ-150: No-return atomics still expose `VDST` as an unconditional def

Reported by: Curie the 2nd subagent; local audit.

Manual evidence:

- The Chapter 11 field table says `VDST` is the destination for data returned
  by loads or atomics that return the pre-op value, and says `TH` controls
  whether atomics return the pre-op value, at `rdna4/README.md:5320` through
  `:5322`.
- The common temporal-hint prose defines atomic `TH[0]` as the return/no-return
  selector at `rdna4/README.md:1784` through `:1795`.

Rocjitsu evidence:

- Representative RDNA4 atomic constructors expose `VDST` as a destination
  operand unconditionally: `FlatAtomicAddU32Vflat` assigns
  `dst_operands_[0] = &vdst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vflat.cpp:750` through
  `:762`, and `GlobalAtomicAddU32Vglobal` does the same at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vglobal.cpp:911` through
  `:925`.
- The execute paths separately compute whether the atomic returns with
  `d->is_load = amdgpu::gfx12_atomic_returns(inst_.th)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vflat.cpp:764` through
  `:772` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vglobal.cpp:927`
  through `:935`.
- The TH bit decoding problem itself is already recorded in `RDNA4-RJ-020`,
  and wait-counter classification is recorded in `RDNA4-RJ-050`; this entry is
  limited to the unconditional def-use shape.

Impact:

Even when an atomic is encoded as no-return, rocjitsu instruction metadata can
mark `VDST` as defined. Analyses that use generated operands rather than
runtime `VectorMemState::is_load` can clobber liveness or dependency facts for
the destination register.

### RDNA4-RJ-151: Race detector records only one destination VGPR for returning B64 atomics

Reported by: Einstein the 2nd subagent.

Manual evidence:

- Chapter 11.4 says Flat instructions can use up to four consecutive DWORDs of
  data, and `VDST` VGPRs hold return data when present, at
  `rdna4/README.md:5533`.

Rocjitsu evidence:

- Generated returning B64 Flat/Global atomics use `elem_size = 8`,
  `num_elems = 1`, and `is_load = gfx12_atomic_returns(inst_.th)`, for example
  `FlatAtomicSwapB64Vflat` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vflat.cpp:1220` through
  `:1226`, and `GlobalAtomicSwapB64Vglobal` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vglobal.cpp:1407` through
  `:1414`.
- `vector_complete` treats atomics specially and computes
  `vgpr_count = d.elem_size / 4`, so an 8-byte returning atomic writes `VDST`
  and `VDST+1` at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:128`
  through `:137`.
- The atomic helper copies 8 response bytes for 64-bit atomics at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:354` through
  `:372`.
- The race detector's global-load registration path uses
  `std::vector<uint32_t> registers(d.num_elems)` and appends
  `logicalBase + i`, so B64 atomics with `num_elems = 1` register only one
  destination VGPR even though runtime completion writes two, at
  `lib/rocjitsu/src/rocjitsu/vm/plugins/race_detector/plugin.cpp:314` through
  `:321`.

Impact:

For returning 64-bit Flat/Global atomics, a later read of `VDST+1` can miss the
memory-to-VGPR dependency that the race detector should have recorded.

### RDNA4-RJ-152: Race-detector observation over-reports sub-DWORD and D16 source reads

Reported by: Einstein the 2nd subagent.

Manual evidence:

- Chapter 11.4 says `D16_HI` instructions read or write only the high 16 bits,
  while `D16` instructions use the low 16 bits, at `rdna4/README.md:5535`.
- The same section says there is no data-format conversion for Flat data at
  `rdna4/README.md:5533`; B8/B16 stores consume only their stored byte count.

Rocjitsu evidence:

- Generated B8/B16 and D16_HI stores functionally extract the right low or high
  bytes, but they read source VGPRs through `RegisterAccess(cu).read_vgpr(...)`
  without passing a byte mask. Representative examples are
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vflat.cpp:256` through
  `:258`, `:630` through `:633`,
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vglobal.cpp:274` through
  `:276`, and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vscratch.cpp:672` through
  `:674`.
- `RegisterAccess::read_vgpr` defaults to `byte_mask = 0xF` and forwards that
  mask to observers at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/register_access.h:726` through `:744`
  and `:782`.
- D16 load merge reads the old destination through `cu.read_vgpr` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:188`, and
  `ComputeUnit::read_vgpr` notifies a full-register read at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.h:702` through `:704`.
- The race detector does use byte masks in its overlap check at
  `lib/rocjitsu/src/rocjitsu/vm/plugins/race_detector/core/wave_race_state.cpp:182`
  through `:187`, so observing a full-register source read can create false
  conflicts against half-register memory events.

Impact:

Functional emulation stores the correct bytes, but race-detector instrumentation
can report false dependencies or races involving untouched bytes of the same
VGPR. This matches the false-positive class already noted in the GFX950 HIP
race test comments.

### RDNA4-RJ-153: Block VGPR load/store ignores `M0` and transfers all 32 DWORDs

Reported by: local audit.

Manual evidence:

- Chapter 11.5 says `M0` carries the per-VGPR load/store bitmask, with the LSB
  selecting the first VGPR, at `rdna4/README.md:5549`.
- It says skipped VGPRs also skip the corresponding memory location rather than
  compacting the block at `rdna4/README.md:5547`.
- It says the block address advances by 4 bytes for every VGPR position
  regardless of whether the `M0` bit transfers data, and `M0==0` transfers no
  data, at `rdna4/README.md:5553` and `:5564`.
- The instruction-definition pseudocode gates each DWORD on `M0[i]` for both
  scratch and global block operations at `rdna4/README.md:29999` through
  `:30021` and `:30670` through `:30692`.

Rocjitsu evidence:

- `semantics.py` lowers `GLOBAL_LOAD_BLOCK`/`SCRATCH_LOAD_BLOCK` to the generic
  `flat_load` semantic and `GLOBAL_STORE_BLOCK`/`SCRATCH_STORE_BLOCK` to the
  generic `flat_store` semantic with `elem_size=4` and `num_elems=32`, at
  `lib/python/amdisa/semantics.py:1976` through `:1979`.
- The generic flat load/store generator emits `num_elems=32`, calls the normal
  address calculator, and does not read `wf.m0()` or capture a per-DWORD mask at
  `lib/python/amdisa/codegen/_generator.py:4620` through `:4684`.
- The generated `GLOBAL_LOAD_BLOCK` and `SCRATCH_LOAD_BLOCK` execute bodies set
  `d->num_elems = 32`, mark one load counter event, call `flat_calculate_addresses`,
  and never read `wf.m0()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vglobal.cpp:2084` through
  `:2094` and `rdna4/vscratch.cpp:730` through `:740`.
- The generated block stores similarly read all 32 consecutive source VGPRs into
  a dense 128-byte per-lane payload at `rdna4/vglobal.cpp:2111` through `:2192`
  and `rdna4/vscratch.cpp:757` through `:838`.
- `VectorMemState` has scalar `elem_size` and `num_elems` fields but no
  per-element transfer mask at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:87`
  through `:128`.
- The vector L1 cache consumes `num_elems * elem_size` as one contiguous
  per-lane stride for loads and stores at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/l1_vector_cache.cpp:176` through `:199`.
- Race-detector memory events are derived from `d.num_elems` rather than from a
  per-DWORD block mask: vector load events register every destination VGPR in
  the block at `lib/rocjitsu/src/rocjitsu/vm/plugins/race_detector/plugin.cpp:314`
  through `:321`, while block stores read every source VGPR through the
  generated full-span store paths above.

Impact:

For any `M0` other than all ones, rocjitsu over-reads, over-writes, and
over-reports all 32 DWORD slots. The strongest observable cases are `M0==0`,
which should transfer nothing but currently transfers 128 bytes per active lane,
and sparse masks, where skipped memory locations should be left untouched rather
than read or written as part of a dense block.

### RDNA4-RJ-154: Block VGPR per-DWORD out-of-range rules are not modeled

Reported by: local audit; Ptolemy the 2nd subagent.

Manual evidence:

- Chapter 11.5.1 says block data VGPRs are checked per DWORD: out-of-range load
  destinations are ignored, and out-of-range store source DWORDs read from
  `VGPR0`, at `rdna4/README.md:5572` through `:5582`.

Rocjitsu evidence:

- Generated load constructors expose one 1024-bit `VDST` operand, and execute
  bodies store only the base destination register in `d->dst_reg_base` before
  entering the generic memory pipeline, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vglobal.cpp:2070` through
  `:2094` and `rdna4/vscratch.cpp:716` through `:740`.
- Generic vector load completion derives `vgpr_count` from `num_elems *
  elem_size` and writes every `dst_reg_base + i` register in the resulting span
  for active lanes at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:128`
  through `:195`.
- Generated block stores unconditionally read `data_base + 0` through
  `data_base + 31` with `RegisterAccess(cu).read_vgpr(...)` at
  `rdna4/vglobal.cpp:2120` through `:2192` and `rdna4/vscratch.cpp:766`
  through `:838`; no path substitutes `VGPR0` for individual out-of-range
  source DWORDs.

Impact:

Rocjitsu cannot reproduce partially in-range block transfers near the edge of a
wave's allocated VGPR range. Loads should preserve/ignore only the out-of-range
destination DWORDs, while stores should pull those individual source DWORDs from
`VGPR0`; the current dense-span path treats the 32-DWORD block as ordinary
consecutive registers.

### RDNA4-RJ-155: Scratch block load/store accepts the unsupported `ST` addressing form

Reported by: local audit; Ptolemy the 2nd subagent.

Manual evidence:

- Chapter 11.5 says scratch block operations support `SS`, `SV`, and `SVS`, but
  not `ST`, at `rdna4/README.md:5551`.

Rocjitsu evidence:

- `SCRATCH_LOAD_BLOCK` and `SCRATCH_STORE_BLOCK` are generated as ordinary
  `Vscratch` instructions with `SADDR`, `SVE`, `VADDR`, and `IOFFSET` fields at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h:235`
  through `:250`.
- The generated scratch block execute bodies call the generic scratch
  `flat_calculate_addresses(inst_, wf, *d)` helper at `rdna4/vscratch.cpp:730`
  through `:740` and `:757` through `:765`.
- The helper computes a scratch address even when `SVE` is unset and `SADDR` is
  null, using only `scratch_base + lane * lane_stride + IOFFSET`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.cpp:132` through
  `:158`. That is the ordinary `ST` shape, but Chapter 11.5 excludes it for
  block operations.

Impact:

An unsupported scratch-block encoding can execute instead of being rejected or
treated according to the hardware's illegal-encoding behavior. This is narrower
than the generic scratch swizzle/ST legality concern in `RDNA4-RJ-145`: even
without swizzle, Chapter 11.5 removes `ST` from the block instruction contract.

### RDNA4-RJ-156: Global transpose loads model wave64 as two independent wave32 transposes

Reported by: Tesla the 2nd subagent; local audit.

Manual evidence:

- Chapter 11.6.2 says `GLOBAL_LOAD_TR_B128` loads the same total amount of data
  in wave32 and wave64: wave32 loads into four consecutive VGPRs, while wave64
  loads into two consecutive VGPRs, uses only addresses from lanes 0-31, and
  ignores addresses in lanes 32-63, at `rdna4/README.md:5622`.
- The same section says `GLOBAL_LOAD_TR_B64` wave32 loads into two consecutive
  VGPRs, while wave64 loads into one VGPR and likewise ignores addresses from
  lanes 32-63, at `rdna4/README.md:5623`.
- The instruction selection table restates the wave64 destination width:
  `GLOBAL_LOAD_TR_B128` writes 64 bits per lane and `GLOBAL_LOAD_TR_B64` writes
  32 bits per lane in wave64 mode, at `rdna4/README.md:5631` through `:5636`.

Rocjitsu evidence:

- The generated RDNA4 execute bodies use fixed wave32-sized result widths:
  `GLOBAL_LOAD_TR_B128` sets `d->num_elems = 4`, and `GLOBAL_LOAD_TR_B64` sets
  `d->num_elems = 2`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vglobal.cpp:2249` through
  `:2257` and `:2277` through `:2285`.
- The RDNA4 global address helper copies `wf.exec()` directly into
  `d.lane_mask`, records `d.wf_size = wf.wf_size()`, and computes addresses for
  every active lane at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.cpp:44` through
  `:48` and `:103` through `:129`.
- The global memory pipeline issues a vector load for that full lane mask at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:488` through
  `:491`, and the vector L1 cache iterates all coalesced lanes in the mask at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/l1_vector_cache.cpp:176` through `:184`.
- The shared transpose helpers process two 32-lane halves when `wf_size > 32`,
  so the wave64 path sources lanes 32-63 instead of ignoring their addresses, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/ds_transpose.h:40` through
  `:63` for `TR_B8` and `:83` through `:106` for `TR_B16`.
- Generic vector load completion writes `num_elems * elem_size / 4` VGPRs per
  active lane at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:128`
  through `:195`.
- Broad RDNA4 wave64 dispatch is already tracked in `RDNA4-RJ-016`; this entry
  records the instruction-specific wave64 behavior that would still need to be
  fixed or guarded if wave64 execution reaches these helpers.

Impact:

For a wave64 execution context, rocjitsu would read memory addresses from lanes
32-63 even though the manual ignores those addresses, transpose the upper half
as a separate 32-lane tile, and write twice the manual's per-lane destination
width. This can corrupt both the matrix contents and the destination VGPR range.

### RDNA4-RJ-157: Global transpose loads do not model the Chapter 11.6 `EXEC` precondition

Reported by: Tesla the 2nd subagent; local audit.

Manual evidence:

- Chapter 11.6.2 says `GLOBAL_LOAD_TR_B128` and `GLOBAL_LOAD_TR_B64` act like
  `S_NOP` when `EXEC==0`; otherwise `EXEC` must be all ones or the operation is
  undefined, at `rdna4/README.md:5610`.

Rocjitsu evidence:

- The generated execute bodies have no opcode-specific `EXEC` check; they build
  ordinary global-load `VectorMemState`, call `flat_calculate_addresses`, and
  set the dynamic instruction state at `rdna4/vglobal.cpp:2249` through `:2259`
  and `:2277` through `:2287`.
- `flat_calculate_addresses` copies `wf.exec()` into `d.lane_mask` for ordinary
  predicated memory execution at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.cpp:44` through
  `:48` and `:103` through `:129`.
- Memory instructions are routed after execute through the normal memory
  pipeline at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:411` through
  `:437`; `MemoryPipeline::issue` increments the selected wait counter and
  initiates access without a transpose-load zero-`EXEC` `S_NOP` special case at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.h:60` through `:86`.

Impact:

Rocjitsu gives partial-`EXEC` transpose loads deterministic predicated global
load behavior instead of surfacing the manual's undefined/illegal contract, and
zero-`EXEC` transpose loads still flow through the memory-pipeline/plugin path
rather than being represented as an opcode-level `S_NOP`. This is mostly a
validation and observability issue in the current synchronous model, but it can
hide invalid kernels and matter for timing/plugins.

### RDNA4-RJ-158: WGP-mode dispatch allows a single workgroup to reserve 128 KiB of LDS

Reported by: Gauss the 2nd subagent; local audit.

Manual evidence:

- The hardware overview says a single workgroup may allocate up to 64kB of LDS
  space, at `rdna4/README.md:532`.
- Chapter 2.3 says waves in a workgroup may share an LDS allocation up to 64kB,
  at `rdna4/README.md:620`.
- Chapter 12.1 says a WGP physically has 128kB LDS but that one workgroup can
  request up to 64kB memory, at `rdna4/README.md:5654`.
- The WGP-mode section says WGP mode allocates from one large contiguous LDS up
  to the same maximum allocation size, at `rdna4/README.md:632`.

Rocjitsu evidence:

- `ShaderProcessorInput::WgpResource` constructs a WGP LDS backing by summing
  both sibling CU `lds_size_kb` values at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/spi.h:270` through `:273`.
- WGP placement validates a request against `wgp.lds.size_bytes()` and then
  advances `wgp.next_lds_alloc` by the aligned request, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/spi.h:203` through `:239`.
- AQL dispatch validation computes WGP `lds_capacity` from
  `spi->max_wgp_lds_bytes()` and rejects only requests larger than that value,
  at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:979` through
  `:998`.
- `RdnaDispatchTest.WgpModeCombinesSiblingCuLdsCapacity` dispatches a single
  WGP-mode workgroup with `kWgpLdsBytes = 128 * 1024` and expects it to run,
  at `tests/amdgpu_vm_test.cpp:236` through `:260`.
- `RdnaDispatchTest.WgpModeRejectsLdsRequestAboveSiblingPair` expects rejection
  only above `128 * 1024`, at `tests/amdgpu_vm_test.cpp:306` through `:322`.

Impact:

Rocjitsu can accept and execute RDNA4 kernels that request more LDS for one
workgroup than the manual allows. It also bakes this interpretation into tests,
so a future fix needs both runtime validation and updated WGP-mode coverage,
including a negative `64kB + 1` case.

### RDNA4-RJ-159: LDS accesses are bounded by the whole backing, not the wave/workgroup allocation

Reported by: Gauss the 2nd subagent; local audit.

Manual evidence:

- Chapter 12 says `LDSbase`/size allocation registers restrict all LDS accesses
  to the space owned by the workgroup or wave, at `rdna4/README.md:5650`.
- Chapter 3.3.5 says all LDS accesses are restricted to the allocated
  wave/workgroup space, allocations do not wrap, CU-mode accesses cannot cross
  to the other LDS side, and WGP-mode allocations may straddle the CU0/CU1
  boundary only within the allocation, at `rdna4/README.md:931` through `:941`.

Rocjitsu evidence:

- `Wavefront` stores only a per-workgroup `lds_base` offset and selected `Lds`
  backing, at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:167` through
  `:180`.
- `ComputeUnitCore::allocate_lds` returns only a base offset, advances
  `next_lds_alloc_`, and does not preserve a per-wave allocation size on the
  `Wavefront`, at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.h:239`
  through `:245`.
- RDNA4/WGP dispatch writes `wf->set_lds_base(lds_base)` and
  `wf->set_lds(placement.lds)` but no size limit, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:710` through
  `:720`.
- DS address calculation adds `wf.lds_base()` to each lane address without
  checking the original group-segment size, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:69`
  through `:75`.
- `Lds::vector_load` and `Lds::vector_store` compare effective addresses
  against `data_.size()`, which is the whole CU or WGP LDS backing, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/lds.h:136` through `:166`.

Impact:

An address beyond a workgroup's requested LDS allocation can still read or
write later bytes in the same CU/WGP LDS backing if those bytes are in range for
the backing store. That can hide out-of-range LDS violations and allow one
resident workgroup to access storage that belongs to another workgroup
allocation in the same backing.

### RDNA4-RJ-160: RDNA4 direct and parameter LDS loads decode but execute as no-ops

Reported by: Gauss the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.1.2 says direct LDS load reads a single DWORD from LDS and
  broadcasts it to a VGPR across all lanes, at `rdna4/README.md:5672` through
  `:5674`.
- Chapter 12.1.2 says parameter interpolation loads read pixel parameters from
  LDS per quad into one VGPR, with three parameters placed into three lanes and
  the fourth lane receiving zero, at `rdna4/README.md:5676` through `:5680`.

Rocjitsu evidence:

- RDNA4 decoder entries create `DsParamLoadVdsdir` and `DsDirectLoadVdsdir` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/decoder.cpp:2934` through
  `:2939`.
- The generated RDNA4 execute bodies for both instructions explicitly do only
  `(void)wf` with an "Interpolation/LDS-direct: no-op in compute simulation"
  comment at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vdsdir.cpp:30`
  through `:45`.
- The generator emits that no-op for semantic class `lds_direct` and
  interpolation at `lib/python/amdisa/codegen/_generator.py:4263` through
  `:4267`.

Impact:

Any RDNA4 shader that relies on `DS_DIRECT_LOAD` or `DS_PARAM_LOAD` will leave
the destination VGPR unchanged in rocjitsu instead of receiving LDS data. The
12.2/12.2.1 parameter-load-specific M0, EXPcnt, CU-only, and pixel-parameter
rules are refined in `RDNA4-RJ-162` through `RDNA4-RJ-164`; the
12.1.2 access-method contract is already functionally missing.

### RDNA4-RJ-161: LDS bank topology and bank-conflict serialization are not modeled

Reported by: Gauss the 2nd subagent; local audit.

Manual evidence:

- Chapter 12 says simultaneous LDS bank access provides bandwidth and that
  indexed and atomic same-bank conflicts are turned into serial accesses by
  hardware, at `rdna4/README.md:5646`.
- Chapter 12.1 says LDS has 64 DWORD-wide banks split into two 32-bank sets,
  and each bank is a 512x32 two-port RAM with one read and one write port per
  clock, at `rdna4/README.md:5654`.

Rocjitsu evidence:

- `Lds` stores LDS as one flat `std::vector<uint8_t>` backing, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/lds.h:17` through `:26` and `:184`
  through `:185`.
- `LocalMemPipeline::initiate_access` executes LDS atomics immediately through
  `execute_lds_atomic_rmw`, and otherwise calls `lds.vector_load` or
  `lds.vector_store` over the active lane mask with no bank metadata, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:506` through `:596`.
- The atomic helper iterates lanes and applies read-modify-write operations to
  the flat `Lds` object, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:383` through
  `:470`.

Impact:

This is primarily a timing and performance-fidelity gap: functional same-bank
loads/stores often produce the same final bytes, but rocjitsu cannot model
bank-conflict stalls, bank-affinity effects, or conflict-driven ordering and
latency differences for indexed LDS and LDS atomics.

### RDNA4-RJ-162: `DS_PARAM_LOAD` parameter addressing and quad semantics are absent behind the no-op

Reported by: Hubble the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.2 says `DS_PARAM_LOAD` reads vertex parameter data from LDS into
  VGPRs for interpolation, at `rdna4/README.md:5684` through `:5686`.
- Chapter 12.2.1 says a parameter load reads P0/P10/P20 for one 32-bit
  attribute or two 16-bit attributes and spreads the values into lanes 0, 1,
  and 2 of each quad, at `rdna4/README.md:5706` through `:5708`.
- Chapter 12.2.1 defines the implicit M0 layout, requires M0 to be initialized,
  and gives the LDS parameter address formula, at `rdna4/README.md:5720`
  through `:5736`.
- Chapter 12.2.1 defines destination-out-of-range, per-quad EXEC, whole-quad
  write, and FP16 packed-attribute behavior at `rdna4/README.md:5742` through
  `:5750`.

Rocjitsu evidence:

- `DsParamLoadVdsdir` exposes only a 32-bit VGPR destination and an `OPR_ATTR`
  source at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vdsdir.cpp:19`
  through `:28`.
- The generated execute body is an explicit no-op at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vdsdir.cpp:30` through
  `:32`.
- The generator emits no-op bodies for LDS-direct/interpolation semantic
  classes at `lib/python/amdisa/codegen/_generator.py:4263` through `:4267`;
  the semantic classifier groups VDSDIR with LDS-direct handling at
  `lib/python/amdisa/semantics.py:2466` through `:2508`.
- `RegisterSet` explicitly does not track M0, and M0 is stored on the
  wavefront as separate state, at
  `lib/rocjitsu/src/rocjitsu/isa/register_set.h:57` and
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:138` through `:145`.

Impact:

Rocjitsu can decode the parameter-load mnemonic, but execution ignores the LDS
parameter storage, M0-derived primitive/offset state, quad-wide write rules,
FP16 packing, and destination-out-of-range side effect.

### RDNA4-RJ-163: VDSDIR channel and wait sideband fields are decoded but not exposed or used

Reported by: Hubble the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.2.1 defines VDSDIR `ATTR_CHAN` as the X/Y/Z/W channel selector,
  `WAITVDST` as a prior-VALU source-read hazard wait, and `WAITVMVS` as a
  prior-VMEM source-read hazard wait, at `rdna4/README.md:5715` through
  `:5719`.
- Chapter 12.2.1 explains the same source-read WAR hazards for `wait_va_vdst`
  and `WAITVMVS`, at `rdna4/README.md:5768` through `:5772`.

Rocjitsu evidence:

- The generated machine-instruction struct preserves `attr_chan`,
  `wait_va_vdst`, and `wait_vm_vsrc` fields at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h:111`
  through `:119`.
- The RDNA4 builder can encode those fields at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/builders.h:295` through
  `:316`.
- `DsParamLoadVdsdir` constructs operands only from `vdst` and `attr`, dropping
  `attr_chan`, `wait_va_vdst`, and `wait_vm_vsrc` from the instruction operand
  surface at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vdsdir.cpp:19`
  through `:28`.
- Searching rocjitsu for `VA_VDST`, `VM_VSRC`, `wait_va_vdst`, and
  `wait_vm_vsrc` finds only generated field storage/building, not runtime
  hazard handling.

Impact:

Different encodings that vary only by channel or wait fields collapse to the
same high-level instruction shape, and the hazard fields cannot affect
execution or dependency modeling.

### RDNA4-RJ-164: Parameter-load readiness and EXPcnt behavior are not connected

Reported by: Hubble the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.2 says parameter loads use `EXPcnt` for outstanding reads and
  decrement it when data arrives in VGPRs, at `rdna4/README.md:5686`.
- Chapter 12.2 says pixel shader waves stall if `DS_DIRECT_LOAD` or
  `DS_PARAM_LOAD` issues before the LDS parameter data is ready, at
  `rdna4/README.md:5688`.
- The STATUS table defines `LDS_PARAM_RDY` at `rdna4/README.md:1035`.
- Chapter 12.2.1 says parameter/direct loads require `S_WAIT_EXPCNT` before
  use and can complete out of order with exports, at `rdna4/README.md:5756`
  through `:5766`.

Rocjitsu evidence:

- Rocjitsu has generic `EXPCNT` storage and wait handling at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wait_counters.h:47` through `:87` and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/sopp.cpp:547` through
  `:560`.
- Ordinary vector-memory paths set a wait-counter type and release it through
  the memory pipeline at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.h:62`
  through `:88`.
- `DS_PARAM_LOAD` does not create memory-pipeline state or increment/release
  `EXPCNT` because its execute body is a no-op at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vdsdir.cpp:30` through
  `:32`.
- RDNA4 `StatusReg` has no bit-25 `LDS_PARAM_RDY` accessor at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/isa.h:23` through `:62`,
  and searching rocjitsu for `LDS_PARAM_RDY` and `LDS_READY` finds no runtime
  handling.
- The broader missing `EXPcnt` producer modeling for exports and LDS
  direct/parameter loads is already recorded in `RDNA4-RJ-051`; this entry
  records the Chapter 12.2 parameter-load-specific readiness and producer path.

Impact:

`S_WAIT_EXPCNT`, VINTERP wait fields, and pixel-parameter readiness cannot
observe a pending `DS_PARAM_LOAD`. Programs that depend on the documented
parameter-load ordering contract execute as if the load never issued.

### RDNA4-RJ-165: `DS_DIRECT_LOAD` execution ignores LDS, M0 datatype, broadcast, and extension semantics

Reported by: Jason the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.4 says `DS_DIRECT_LOAD` reads a single LDS DWORD, broadcasts the
  result to active lanes, uses per-quad `EXEC`, and uses `EXPcnt`, at
  `rdna4/README.md:5871`.
- Chapter 12.4 defines `M0[15:0]` as the DWORD-aligned LDS byte address and
  `M0[18:16]` as the data type selector, at `rdna4/README.md:5876` through
  `:5886`.
- Chapter 12.4 says signed byte/short data is sign-extended and unsigned
  byte/short data is zero-extended to 32 bits, at `rdna4/README.md:5891`.
- The instruction definition repeats that direct load reads from
  `ADDR[M0[15:0]]`, uses M0 datatype values, and runs in whole-quad mode at
  `rdna4/README.md:25216` through `:25220`.

Rocjitsu evidence:

- `DsDirectLoadVdsdir` exposes only a 32-bit VGPR destination and sets
  `num_src_ = 0`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vdsdir.cpp:34` through
  `:41`.
- The generated execute body explicitly does only `(void)wf`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vdsdir.cpp:43` through
  `:45`.
- The code generator emits this no-op for LDS-direct/interpolation classes at
  `lib/python/amdisa/codegen/_generator.py:4258` through `:4267`.
- The semantic derivation layer has an abstract `lds_direct_load` call at
  `lib/python/amdisa/sema_derive.py:2766` through `:2780`, but that call does
  not reach the generated RDNA4 runtime body.
- `RegisterSet` explicitly does not track M0, while the actual M0 value is
  held on the wavefront at `lib/rocjitsu/src/rocjitsu/isa/register_set.h:57`
  and `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:138` through `:145`.

Impact:

Direct loads leave `VDST` unchanged in rocjitsu. The emulator does not read LDS
through M0, decode the direct-load datatype, sign/zero extend sub-DWORD data,
or write the result using the manual's whole-quad broadcast mask.

### RDNA4-RJ-166: `DS_DIRECT_LOAD` mode, readiness, and wait-counter hazards are not enforced

Reported by: Jason the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.4 says direct access is available only in CU mode, not WGP mode,
  at `rdna4/README.md:5869`.
- Chapter 12.2 says pixel shader waves stall if `DS_DIRECT_LOAD` or
  `DS_PARAM_LOAD` issues before LDS parameter data is ready, at
  `rdna4/README.md:5688`.
- Chapter 12.2.1 applies the data-dependency rules to both direct and
  parameter loads, including `EXPcnt`, `WAITVDST`, and `WAITVMVS`, at
  `rdna4/README.md:5715` through `:5719` and `:5754` through `:5772`.
- Chapter 5.7.1 says `DS_DIRECT_LOAD` increments `EXPcnt` on issue and
  decrements it when the data writes VGPRs, at `rdna4/README.md:2202` through
  `:2209`.

Rocjitsu evidence:

- RDNA4 decoder entries route VDSDIR opcode 1 to `DsDirectLoadVdsdir`
  unconditionally at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/decoder.cpp:2938`
  through `:2939` and `:7999` through `:8008`.
- The direct-load execute body is a no-op and performs no CU/WGP-mode check at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vdsdir.cpp:43` through
  `:45`.
- Rocjitsu has generic `EXPCNT` storage and `S_WAIT_EXPCNT` handling at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wait_counters.h:47` through `:87` and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/sopp.cpp:547` through
  `:560`, but direct load never creates memory-pipeline state or touches the
  counter.
- The VDSDIR machine instruction stores `wait_va_vdst` and `wait_vm_vsrc` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h:111`
  through `:119`, and builders can encode them at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/builders.h:295` through
  `:316`, but searching rocjitsu finds no runtime `VA_VDST` / `VM_VSRC`
  hazard implementation.
- RDNA4 `StatusReg` has no bit-25 `LDS_PARAM_RDY` accessor at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/isa.h:23` through `:62`,
  and searching rocjitsu for `LDS_READY` and `LDS_PARAM_RDY` finds no runtime
  handling. The shared parameter-load readiness side is also recorded in
  `RDNA4-RJ-164`.

Impact:

WGP-mode direct-load encodings are accepted, readiness does not stall issue,
and waits cannot observe a pending direct load. Dependency-sensitive shaders
therefore execute as if `DS_DIRECT_LOAD` has no memory, readiness, or
completion side effects.

### RDNA4-RJ-167: `DS_LOAD/STORE_ADDTID_B32` use stale or generic address formulas

Reported by: Avicenna the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.5 says ADDTID uses
  `LDS_BASE + {OFFSET1, OFFSET0} + TID(0..63)*4 + M0`, with no VGPR address
  component and DWORD-aligned M0, at `rdna4/README.md:5967` through `:5974`.
- The VDS field table says M0 is used only for `ds_load_addtid_b32` and
  `ds_write_addtid_b32`, and represents a byte address, at
  `rdna4/README.md:5920` through `:5922`.

Rocjitsu evidence:

- `DsStoreAddtidB32Vds` publishes only `DATA0` as a source, but its execute
  body calls generic `ds_calculate_addresses(inst_, wf, *d)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vds.cpp:3865` through
  `:3882`.
- The generic DS helper reads `inst.addr` as a VGPR address and computes
  `VGPR[ADDR] + {OFFSET1,OFFSET0} + LDS_BASE` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:60`
  through `:76`.
- `DsLoadAddtidB32Vds` has a special path, but it interprets `M0[24:16]` as a
  stride and computes `lane * ds_stride_bytes + offset + LDS_BASE`, omitting
  the low M0 byte offset and fixed `TID*4` term, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vds.cpp:3905` through
  `:3928`.
- `_derive_ds()` classifies generic writes before reaching the ADDTID branch,
  while the ADDTID comment still documents the stale M0-stride model, at
  `lib/python/amdisa/semantics.py:2219` through `:2249`.

Impact:

`DS_STORE_ADDTID_B32` can read an address field that the instruction does not
expose, and both ADDTID load/store forms compute the wrong LDS byte address
whenever M0 or the fixed thread-ID offset matters.

### RDNA4-RJ-168: `DS_STOREXCHG_2ADDR*_RTN` executes as a one-address, one-source swap

Reported by: Avicenna the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.5 lists `DS_STOREXCHG_2ADDR_RTN_{B32,B64}` and
  `DS_STOREXCHG_2ADDR_STRIDE64_RTN_{B32,B64}` as two-address store-exchange
  forms, at `rdna4/README.md:5934` through `:5936` and `:6010` through
  `:6012`.
- Chapter 12.5 says the double-address instructions include
  `STOREXCHG_2ADDR_*`, with separate `OFFSET0` and `OFFSET1` scaled by data
  size from the VGPR address base, at `rdna4/README.md:5960` through `:5965`.

Rocjitsu evidence:

- `DsStorexchg2addrRtnB32Vds` exposes `VDST`, `ADDR`, `DATA0`, and `DATA1`,
  but its execute body calls the single-address `ds_calculate_addresses`
  helper, never sets `ds2_active`, and reads only `DATA0`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vds.cpp:1301` through
  `:1336`.
- `DsStorexchg2addrStride64RtnB32Vds` follows the same one-address path instead
  of the stride64 two-address path, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vds.cpp:1339` through
  `:1375`.
- `DsStorexchg2addrRtnB64Vds` follows the same one-address path and reads only
  the `DATA0` pair, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vds.cpp:2999` through
  `:3036`.
- `DsStorexchg2addrStride64RtnB64Vds` also uses the generic single-address
  helper instead of the stride64 two-address path, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vds.cpp:3039` through
  `:3077`.

Impact:

The generated instruction shape suggests both exchange sources and both LDS
addresses are modeled, but execution swaps only the first source through a
single address. Any test using distinct `DATA0`/`DATA1` values or offsets will
observe the wrong LDS and return data.

### RDNA4-RJ-169: No-return DS atomics write old LDS data back to `VDST`

Reported by: Avicenna the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.5 says atomic operations have the option of returning the LDS
  pre-op value to VGPRs, at `rdna4/README.md:5901`.
- The VDS field table defines `VDST` as the destination for LDS loads or
  atomic return values, at `rdna4/README.md:5916`.
- Chapter 12.5.1 repeats that atomics optionally return the pre-op value, at
  `rdna4/README.md:5978`.

Rocjitsu evidence:

- `DsAddU32Vds` has `num_dst_ = 0`, but its execute body still sets
  `dst_reg_base` from `inst_.vdst` and marks the memory state as a load at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vds.cpp:27` through `:45`.
- The generator hard-codes `DS atomics always return the old value` and emits
  `d->is_load = true` for DS atomics at
  `lib/python/amdisa/codegen/_generator.py:4969` through `:4992`.
- `execute_lds_atomic_rmw()` always writes the old LDS value into
  `response_data`, and `vector_complete()` writes response data to VGPRs
  whenever `d.is_load` is true, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:383` through
  `:445` and `:119` through `:198`.
- The related Chapter 11 metadata-only no-return atomic finding is recorded in
  `RDNA4-RJ-150`; this entry is the DS runtime writeback form.

Impact:

No-return DS atomics can clobber a VGPR selected by the raw `VDST` bits even
though the instruction has no architectural return destination.

### RDNA4-RJ-170: LDS floating-point atomics ignore MODE denormal controls

Reported by: Avicenna the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.5.1 says floating-point atomic ops use MODE to control denormal
  flushing behavior, at `rdna4/README.md:5984`.
- Chapter 13.2 says LDS-indexed FP atomics use `MODE.denormal`, with separate
  input/output controls and different F32, F64, F16, min/max, and compare-store
  behavior, at `rdna4/README.md:6082` through `:6105`.

Rocjitsu evidence:

- `Wavefront` stores raw MODE state at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:103` through `:108`, but
  `VectorMemState` has no MODE or denorm-control fields at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:85` through `:125`.
- The LDS FP atomic path bit-casts the old and source values to `float` or
  `double`, calls plain `+`, `std::fmin`, or `std::fmax`, and writes the result
  without consulting MODE at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:285` through
  `:297` and `:423` through `:459`.

Impact:

Denormal-sensitive DS FP atomics can produce results for the host math default
instead of the wave's MODE-controlled input/output denormal policy.

### RDNA4-RJ-171: Packed LDS FP atomics are modeled as scalar F32 atomics

Reported by: Avicenna the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.5.1 lists `ds_pk_add_f16`, `ds_pk_add_rtn_f16`,
  `ds_pk_add_bf16`, and `ds_pk_add_rtn_bf16` as packed LDS atomic opcodes, at
  `rdna4/README.md:6007` through `:6009`.
- Chapter 13.2 gives separate denormal handling for `PK_ADD_F16 / _BF16`, at
  `rdna4/README.md:6095` through `:6100`.

Rocjitsu evidence:

- `_DS_ATOMIC_MAP` maps the packed F16 and BF16 suffixes to
  `('fadd', 4, 1)`, at `lib/python/amdisa/semantics.py:2378` through `:2381`.
- Generated `DsPkAddF16Vds` sets `elem_size = 4`, `num_elems = 1`, and
  `AtomicOp::FADD`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vds.cpp:3441` through
  `:3461`.
- The local LDS atomic implementation treats any 4-byte FP atomic as one
  `float`, not two packed F16 or BF16 lanes, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:423` through
  `:437`.

Impact:

Packed LDS FP atomics update the 32-bit word as scalar F32 data instead of
performing two independent half-width additions, so both stored LDS data and
returned pre-op values can be wrong.

### RDNA4-RJ-172: VDS disassembly drops nonzero offset fields

Reported by: Avicenna the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.5 defines `OFFSET0` and `OFFSET1` as VDS instruction fields with
  one-address, two-address, and stride64-specific meanings, at
  `rdna4/README.md:5913` through `:5915` and `:5951` through `:5965`.

Rocjitsu evidence:

- `VdsMachineInst` stores `offset0` and `offset1`, and the builder can encode
  both fields, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h:122`
  through `:128` and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/builders.h:328` through
  `:340`.
- The RDNA4 `Vds` base does not override `build_modifiers()`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/encodings.h:578` through
  `:583`.
- Base disassembly prints only destination/source operands and then calls the
  empty default modifier hook, at
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:239` through `:274`.

Impact:

Nonzero VDS offsets can be lost in disassembly, making round-trip decode
diagnostics and human inspection misleading for indexed and atomic LDS
instructions.

### RDNA4-RJ-173: `DS_SWIZZLE_B32` misses FFT and rotate modes

Reported by: Pascal the 2nd subagent; local audit.

Manual evidence:

- The Chapter 16.15 `DS_SWIZZLE_B32` definition says the opcode supports FFT,
  rotate, group-of-4, and group-of-32 modes, at `rdna4/README.md:25807`
  through `:25815`.
- It defines FFT mode for `offset >= 0xe000`, including mask and bit-reversal
  pseudocode, at `rdna4/README.md:25817` through `:25822` and `:25860`
  through `:25872`.
- It defines rotate mode for `0xc000 <= offset < 0xe000`, including direction,
  rotate amount, mask, and pseudocode, at `rdna4/README.md:25823` through
  `:25829` and `:25874` through `:25887`.
- Local assembler sanity checks accepted `ds_swizzle_b32 v0, v1 offset:0xc020`
  as `swizzle(ROTATE,0,1)` and `offset:0xe000` as `swizzle(FFT,0)` for
  `gfx1200`.

Rocjitsu evidence:

- `DsSwizzleB32Vds` routes directly to the shared swizzle helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vds.cpp:1490` through
  `:1502`.
- The shared VDS swizzle helper branches only on `offset & 0x8000`, treating
  every high-bit offset as group-of-4 QDMode and every other offset as
  group-of-32 bitmode, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:299`
  through `:331`.
- The generator mirrors that two-branch implementation at
  `lib/python/amdisa/codegen/_generator.py:4194` through `:4238`.
- Existing swizzle execution coverage exercises full-EXEC broadcast and QDMode
  cases only, at `tests/instruction_execution_harness_test.cpp:3384` through
  `:3418`.

Impact:

Encoded RDNA4 `DS_SWIZZLE_B32` rotate and FFT forms execute as the wrong basic
swizzle mode. Any shader relying on those offset ranges can receive unrelated
lane data.

### RDNA4-RJ-174: `DS_SWIZZLE_B32` ignores disabled-source zeroing

Reported by: Pascal the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.5.2 says disabled lane reads return zero for lane-permute
  operations, at `rdna4/README.md:6027`.
- The `DS_SWIZZLE_B32` instruction definition says invalid-thread reads return
  `0x0`, at `rdna4/README.md:25813`.
- The DS_SWIZZLE pseudocode applies `thread_valid[j] ? thread_in[j] : 0` in
  FFT, rotate, and group-of-32 modes, at `rdna4/README.md:25870`, `:25885`,
  and `:25914`.

Rocjitsu evidence:

- The shared VDS swizzle helper reads the source VGPR with `full_lane_mask`,
  copies every lane into `src_data`, and writes `src_data[src_lane]` for active
  destination lanes without checking whether `src_lane` is active in EXEC, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:299`
  through `:329`.
- The generator emits the same full-lane source read and unconditional
  `src_data[src_lane]` write at `lib/python/amdisa/codegen/_generator.py:4200`
  through `:4238`.
- Existing swizzle tests seed full EXEC before exercising broadcast and QDMode
  at `tests/instruction_execution_harness_test.cpp:3384` through `:3418`, so
  inactive-source zeroing is not covered there.

Impact:

`DS_SWIZZLE_B32` can copy values from EXEC-disabled source lanes instead of
returning zero for active destinations that select those lanes.

### RDNA4-RJ-175: Lane-permute semantic IR conflates scatter, gather, and FI

Reported by: Pascal the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.5.2 distinguishes forward scatter `DS_PERMUTE_B32`, backward
  gather `DS_BPERMUTE_B32`, and the `DS_BPERMUTE_FI_B32` source-read exception
  where all lanes are read and EXEC only gates writes, at `rdna4/README.md:6023`
  through `:6029`.

Rocjitsu evidence:

- `derive_semantics()` maps `DS_PERMUTE_B32`, `DS_BPERMUTE_B32`, and
  `DS_BPERMUTE_FI_B32` all to the same `ds_permute` semantic class with no
  operation tag, at `lib/python/amdisa/semantics.py:2388` through `:2392`.
- `_DsPermute.derive()` defaults that class to `ds_bpermute` when
  `sem.operation` is absent, at `lib/python/amdisa/sema_derive.py:1949`
  through `:1966`.
- Existing semantic tests assert only the BPERMUTE class and call-name shape;
  they do not assert that `DS_PERMUTE_B32` derives a scatter call or that
  `DS_BPERMUTE_FI_B32` preserves the FI source-read exception, at
  `lib/python/amdisa/tests/test_sema_derive.py:1968` through `:1984`.
- The generated C++ execute body branches by mnemonic, so this entry is limited
  to semantic-IR consumers rather than the generated C++ wrappers, at
  `lib/python/amdisa/codegen/_generator.py:4105` through `:4192`.

Impact:

Semantic-IR consumers can treat forward permute and FI as ordinary backward
permute. That can affect semantic comparison, DBT rule generation, and tests
that rely on derived IR rather than the generated C++ execute body.

### RDNA4-RJ-176: Lane-permute helpers over-report source-lane reads to observers

Reported by: Pascal the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.5.2 says EXEC is honored for source reads and destination writes
  for lane-permute operations, except that `DS_BPERMUTE_FI_B32` reads all lanes
  and EXEC gates only writes, at `rdna4/README.md:6027`.

Rocjitsu evidence:

- `DS_BPERMUTE_B32` reads both `DATA0` and `ADDR` regions with
  `full_lane_mask`, then later applies EXEC when selecting the source lane and
  destination write, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:93`
  through `:123`.
- `DS_PERMUTE_B32` reads `DATA0` with `full_lane_mask` while reading `ADDR`
  with EXEC, then later loops only over active source lanes, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:230`
  through `:261`.
- `DS_SWIZZLE_B32` reads its source VGPR with `full_lane_mask`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:299`
  through `:307`.
- `RegisterAccess::read_vgpr_region()` immediately forwards the provided
  lane mask to CU read observers, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/register_access.h:744` through `:787`.

Impact:

Architectural values are corrected later for ordinary `DS_PERMUTE_B32` and
`DS_BPERMUTE_B32`, but plugins, traces, race detectors, or liveness tools that
consume VGPR read observations can see reads from lanes that EXEC should have
masked off. `DS_BPERMUTE_FI_B32` is the exception where full source-lane reads
are expected.

### RDNA4-RJ-177: BVH stack DS instructions decode but are not executable

Reported by: Kant the 2nd subagent; local audit.

Manual evidence:

- Chapter 12.5.3 defines three ray-tracing short-stack instructions:
  `DS_BVH_STACK_PUSH4_POP1_RTN_B32`,
  `DS_BVH_STACK_PUSH8_POP1_RTN_B32`, and
  `DS_BVH_STACK_PUSH8_POP2_RTN_B64`, at `rdna4/README.md:6033` through
  `:6042`.
- The manual defines packed stack address state, stack-size fields,
  primitive/triangle optimization flags, and updated `ADDR` return at
  `rdna4/README.md:6048` through `:6069`.
- The instruction definitions specify `DATA_VALID` filtering, last-node
  termination, push order, pop fallback, LDS invalidation, return data, and the
  B64 second-pop case at `rdna4/README.md:27048` through `:27213`.

Rocjitsu evidence:

- RDNA4 constructors and operand metadata exist for all three stack forms at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vds.cpp:4056` through
  `:4112`.
- Each execute body immediately throws `util::UnimplementedInst(mnemonic())`
  at `rdna4/vds.cpp:4072` through `:4075`, `:4093` through `:4096`, and
  `:4114` through `:4117`.
- The Python semantic derivation falls unrecognized DS instructions through to
  `InstructionSemantics(name, 'nop')` at `lib/python/amdisa/semantics.py:2428`,
  and generated `nop` execution emits `UnimplementedInst` at
  `lib/python/amdisa/codegen/_generator.py:3377`.

Impact:

RDNA4 kernels containing these legal BVH stack instructions can decode but
cannot execute in rocjitsu. The emulator cannot model the short-stack LDS
updates or return data described by the ISA.

### RDNA4-RJ-178: BVH stack DS instructions are not marked as memory operations

Reported by: Kant the 2nd subagent; local audit.

Manual evidence:

- Chapter 5.3 groups `BVH_stack` with LDS indexed load/store/atomic clause
  types at `rdna4/README.md:1907`.
- The RDNA4 XML classifies the three stack opcodes as `VMEM` /
  `DATA_SHARE` at `amdgpu_isa_rdna4.xml:49736` through `:49739`, `:49784`
  through `:49787`, and `:49832` through `:49835`.

Rocjitsu evidence:

- The generated constructors for the three RDNA4 stack classes populate
  operands but do not set `flags_ |= MEMORY_OP`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vds.cpp:4056` through
  `:4112`.
- Nearby generated VDS memory instructions do set `flags_ |= MEMORY_OP`, for
  example `DsLoadAddtidB32Vds` at `rdna4/vds.cpp:3895` through `:3903`.
- The generator's memory-class allowlist includes DS read/write/atomic classes
  but no BVH stack semantic class, at `lib/python/amdisa/codegen/_generator.py:6208`
  through `:6220` and `:6471` through `:6474`.

Impact:

Even before execution semantics are implemented, generic instruction consumers
that rely on `MEMORY_OP` can treat these LDS stack operations as non-memory
instructions. That affects scheduling, dependency, tracing, and DBT analyses
that classify memory effects by instruction flags.

### RDNA4-RJ-179: VINTERP instructions decode but execute as no-ops

Manual/XML evidence:

- Chapter 12.3 describes VINTERP as FMA-based parameter interpolation with
  fixed DPP8 lane selection, implicit `fetch invalid = 1`, `WAIT_EXP`, OPSEL,
  clamp, no-exception behavior, and data-forwarding restrictions at
  `rdna4/README.md:5774` through `:5833`.
- Chapter 12.3.1 and the detailed instruction definitions specify F16/F32
  interpolation, RTZ variants, OPSEL source/destination roles, fixed DPP8
  selectors, and fused FMA formulas at `rdna4/README.md:5850` through `:5865`
  and `rdna4/README.md:25062` through `:25194`.
- The RDNA4 XML lists all six VINTERP opcodes with F32/F16 operand widths at
  `amdgpu_isa_rdna4.xml:57958` through `:58233`.

Rocjitsu evidence:

- Rocjitsu generates decode entries for all six VINTERP opcodes at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/decoder.cpp:2900` through
  `:2926` and `decoder.cpp:7962` through `:7970`.
- The generated constructors expose the expected operand widths, including
  F32 `P10/P2` forms and F16/RTZ mixed-width forms, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vinterp.cpp:19` through
  `:142`.
- Every generated VINTERP `execute_impl()` body is the same compute-simulation
  no-op at `rdna4/vinterp.cpp:34` through `:35`, `:53` through `:54`,
  `:72` through `:73`, `:97` through `:98`, `:116` through `:117`, and
  `:141` through `:142`. The generator emits this body for semantic class
  `interp` at `lib/python/amdisa/codegen/_generator.py:4263` through `:4267`.
- `RDNA4-RJ-096` already records the true16-specific F16 subset; this entry
  records the complete Chapter 12.3 interpolation contract, including F32
  forms and built-in DPP/wait behavior.

Impact:

Decoded VINTERP instructions leave their destination unchanged instead of
performing parameter interpolation. F32 shaders miss the basic interpolation
operation, and F16/RTZ shaders also miss OPSEL half selection, destination-half
preservation, RTZ rounding, and clamp behavior.

### RDNA4-RJ-180: VINTERP `WAIT_EXP` is decoded but not enforced

Manual/XML evidence:

- Chapter 12.2.1 says `DS_PARAM_LOAD`/`DS_DIRECT_LOAD` use `EXPcnt` and that
  VINTERP has a `wait_EXPcnt` field to avoid the load-use hazard, at
  `rdna4/README.md:5756` through `:5760`.
- Chapter 12.3 says VINTERP includes a built-in `S_WAIT_EXPCNT` and gives
  examples using `S_WAIT_EXPCNT <= N` on interpolation operations at
  `rdna4/README.md:5817` through `:5848`.
- XML carries the raw `WAIT_EXP` field and describes value 7 as no wait at
  `amdgpu_isa_rdna4.xml:3179` through `:3188`.

Rocjitsu evidence:

- The raw `VinterpMachineInst` preserves `wait_exp` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h:96`
  through `:109`, and the builder writes it at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/builders.h:278` through
  `:292`.
- The VINTERP constructors and no-op execute bodies never read `inst_.wait_exp`
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vinterp.cpp:19`
  through `:142`.
- Rocjitsu does have `S_WAIT_EXPCNT` handling at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/sopp.cpp:547` through
  `:560`, but no equivalent wait is performed by VINTERP before the no-op.

Impact:

Programs that rely on VINTERP's embedded wait field execute as if the field
were always "no wait." Even if VINTERP arithmetic is implemented later, the
load-use hazard remains incorrect unless `wait_exp` is wired into the wait
counter path.

### RDNA4-RJ-181: VDSDIR parameter/direct loads are not marked as memory operations

Manual/XML evidence:

- Chapter 12.2 says `DS_PARAM_LOAD` reads LDS into VGPRs using `EXPcnt`, at
  `rdna4/README.md:5684` through `:5686`; Chapter 12.4 says
  `DS_DIRECT_LOAD` reads LDS and broadcasts to VGPRs using `EXPcnt`, at
  `rdna4/README.md:5867` through `:5871`.
- XML classifies `DS_PARAM_LOAD` and `DS_DIRECT_LOAD` as `VMEM` /
  `DATA_SHARE` instructions with implicit DSMEM operands at
  `amdgpu_isa_rdna4.xml:49933` through `:50014`.

Rocjitsu evidence:

- The generated `DsParamLoadVdsdir` and `DsDirectLoadVdsdir` constructors set
  their explicit operands but do not set `flags_ |= MEMORY_OP`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vdsdir.cpp:19` through
  `:40`.
- The generator memory-class allowlist includes DS read/write/atomic classes
  but not the `lds_direct` class used for VDSDIR, at
  `lib/python/amdisa/codegen/_generator.py:6208` through `:6240` and
  `:6471` through `:6472`.
- `Instruction::is_memory_op()` is driven entirely by that flag at
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:193` through `:196`;
  exported code metadata and compute-unit memory routing consume it at
  `lib/rocjitsu/src/rocjitsu/code/rj_code.cpp:304` through `:313` and
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:430` through `:437`.

Impact:

Generic consumers classify VDSDIR loads as non-memory even though they read
LDS, write VGPRs, and participate in `EXPcnt`. This compounds the existing
no-op execution gaps in `RDNA4-RJ-160`, `RDNA4-RJ-162`, and `RDNA4-RJ-165`.

### RDNA4-RJ-182: Float atomics rely on host FP operations for Chapter 13 bit rules

Reported by: local audit.

Manual evidence:

- Chapter 13 says memory atomics suppress numeric exceptions, at
  `rdna4/README.md:6074` through `:6076`.
- Chapter 13.1 fixes float-atomic-add rounding to round-to-nearest-even and
  ignores `MODE.round`, at `rdna4/README.md:6078` through `:6080`.
- Chapter 13.3 defines NaN quieting/propagation, memory-atomic
  MINNUM/MAXNUM selection, `-0 < +0` signed-zero ordering, and float-add
  special cases, at `rdna4/README.md:6113` through `:6167`.

Rocjitsu evidence:

- Generated buffer, flat, and global F32 float atomics all collapse to the
  generic `AtomicOp::FMIN`, `AtomicOp::FMAX`, or `AtomicOp::FADD` selectors at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vbuffer.cpp:2368`
  through `:2456`, `rdna4/vflat.cpp:1782` through `:1864`, and
  `rdna4/vglobal.cpp:2006` through `:2217`.
- The shared atomic helper implements those selectors as plain host
  `old_val + src_val`, `std::fmin(old_val, src_val)`, and
  `std::fmax(old_val, src_val)`, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:286` through
  `:296`.
- The L2 RMW path bit-casts 32-bit and 64-bit memory values into host `float`
  or `double`, calls that helper, and writes the host result bits at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:321` through
  `:338` and `:353` through `:358`. The LDS RMW path does the same for local
  memory at `memory_pipeline.cpp:423` through `:459`; LDS denormal-control
  handling is tracked separately in `RDNA4-RJ-170`.

Impact:

Ordinary F32/F64 float atomics are not pinned to the manual's exact bit-level
policy for NaN payload/source selection, signed-zero ties, exception
suppression, and add special cases. Host `fmin`/`fmax` and `+` may agree for
common values, but they are not an explicit model of the Chapter 13 contract.

### RDNA4-RJ-183: Packed VMEM FP atomics are modeled as scalar F32 atomics

Reported by: local audit.

Manual/XML evidence:

- Chapter 13 says packed F16/BF16 float atomics are part of the same
  float-memory-atomic family and gives `PK_ADD_F16 / _BF16` its own denormal
  table row, at `rdna4/README.md:6074` and `:6095` through `:6105`.
- XML records buffer, flat, and global packed float atomics as
  two-component packed F16/BF16 data formats, for example
  `BUFFER_ATOMIC_PK_ADD_F16` / `_BF16` at `amdgpu_isa_rdna4.xml:40037` and
  `:40098`, `FLAT_ATOMIC_PK_ADD_F16` / `_BF16` at `:50350` and `:50405`, and
  `GLOBAL_ATOMIC_PK_ADD_F16` / `_BF16` at `:54020` and `:54082`.

Rocjitsu evidence:

- Generated buffer packed add bodies set `elem_size = 4`, `num_elems = 1`,
  and `AtomicOp::FADD`, then copy one 32-bit VGPR word per lane at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vbuffer.cpp:2473`
  through `:2538`.
- Generated flat and global packed add bodies use the same scalar 4-byte
  `AtomicOp::FADD` shape at `rdna4/vflat.cpp:1882` through `:1940` and
  `rdna4/vglobal.cpp:2291` through `:2353`.
- The shared L2 atomic path treats every 4-byte FP atomic as one host `float`
  value at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:321`
  through `:338`.
- RDNA4 image packed atomics are generated as image-pipeline stubs, already
  covered by the broader VIMAGE execution gaps `RDNA4-RJ-125` and
  `RDNA4-RJ-131`; this entry is limited to buffer/flat/global paths that
  actually enter the scalar RMW pipeline.

Impact:

Packed buffer, flat, and global F16/BF16 atomics update the 32-bit word as a
single F32 RMW instead of performing two independent packed-lane additions, so
both memory results and optional returned old values can be wrong.

### RDNA4-RJ-184: RDNA4 `EXPORT` executes as a no-op

Reported by: local audit.

Manual evidence:

- Chapter 14 says exports copy VGPR data to position, color, or Z export
  buffers and use `EXEC` to output only enabled pixels or vertices, at
  `rdna4/README.md:6171`.
- Sections 14.1 and 14.2 require pixel and vertex/position exports, `DONE`
  handling, per-target uniqueness, and stage-specific ordering at
  `rdna4/README.md:6206` through `:6249`.
- Section 14.3 says exports issue in two phases, increment `EXPcnt`, read
  `EXEC` and VGPR data when the export occurs, and then decrement `EXPcnt`, at
  `rdna4/README.md:6251` through `:6258`.

Rocjitsu evidence:

- RDNA4 generates `ExportVexport`, but its execute body is a compute-simulation
  no-op at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vexport.cpp:35` through
  `:36`.
- The constructor records ordinary operands only and does not attach any
  export-pipeline state or `MEMORY_OP`-like routing flag at `rdna4/vexport.cpp:19`
  through `:33`.
- The already-recorded wait-counter side of this is `RDNA4-RJ-051`; this entry
  is the Chapter 14 data/export side effect itself.

Impact:

Graphics export instructions decode but do not emit data, consume final `EXEC`
masks, enforce `DONE`/target contracts, or create an export operation that
later `S_WAIT_EXPCNT` can make visible.

### RDNA4-RJ-185: `VEXPORT` metadata drops `EN`, `DONE`, `ROW_EN`, and implicit M0/EXEC use

Reported by: local audit.

Manual/XML evidence:

- Chapter 14 defines `EN` as the component enable and packed-half selector,
  `DONE` as the last-export marker, and `ROW_EN` as the control that uses M0
  for mesh POS/PRIM export rows, at `rdna4/README.md:6181` through `:6204`.
- Chapter 14.3 says M0 is read when the export request is made and `EXEC` is
  read later when export is granted, at `rdna4/README.md:6253` through
  `:6258`.
- XML lists implicit `EXEC` and M0 operands for `EXPORT` at
  `amdgpu_isa_rdna4.xml:47513` through `:47522`.

Rocjitsu evidence:

- `VexportMachineInst` stores raw `en`, `tgt`, `done`, `row_en`, and
  `vsrc0..3` fields at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h:202`
  through `:214`.
- `ExportVexport` exposes `tgt` as one destination and all four VGPRs as
  unconditional sources, but exposes no `EN`, `DONE`, `ROW_EN`, M0, or EXEC
  operand/modifier at `rdna4/vexport.cpp:19` through `:33`.
- The `Vexport` base does not override `build_modifiers()`, so default
  disassembly prints only the operand lists and drops encoded modifier fields
  at `rdna4/encodings.h:607` through `:611` and
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:239` through `:255`.

Impact:

Liveness, race/probe metadata, and disassembly cannot tell which VGPR channels
are actually exported, whether the export is final, whether M0 is consumed for
row selection, or when EXEC is a late source. Disabled channels are
over-reported as VGPR reads while `ROW_EN` and status-sensitive behavior are
under-reported.

### RDNA4-RJ-186: RDNA4 status layout omits export-ready and skip-export bits

Reported by: local audit.

Manual evidence:

- Chapter 3.4 defines `STATUS.EXPORT_RDY` bit 8 as the export-buffer
  allocation gate for pixel shaders, at `rdna4/README.md:1020`.
- The same table defines `STATUS.SKIP_EXPORT` bit 18 and `MUST_EXPORT` bit 27,
  at `rdna4/README.md:1032` through `:1037`.
- Chapter 14.1 says pixel exports wait for export-buffer space and are ignored
  when `STATUS.SKIP_EXPORT==1`, at `rdna4/README.md:6224` through `:6226`;
  Chapter 14.3 repeats the skip behavior at `rdna4/README.md:6258`.

Rocjitsu evidence:

- RDNA4 `StatusReg` is explicitly documented as having no `EXPORT_RDY` or
  `SKIP_EXPORT` fields at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/isa.h:23` through `:26`.
- The exposed RDNA4 status members include `SCC`, priorities, `TRAP_EN`,
  `EXECZ`, `VCCZ`, `IN_TG`, `IN_BARRIER`, `HALT`, `TRAP`, `VALID`, `ECC_ERR`,
  and `ALLOW_REPLAY`, but no `EXPORT_RDY`, `SKIP_EXPORT`, `MUST_EXPORT`, or
  `LDS_PARAM_RDY`, at `rdna4/isa.h:34` through `:61`.
- `S_WAIT_EVENT` export-ready behavior is separately recorded in
  `RDNA4-RJ-047`; this entry is the missing wave-state plumbing that Chapter 14
  relies on.

Impact:

RDNA4 export execution cannot distinguish waiting for export-buffer allocation,
skip-as-NOP waves, or pixel-shader must-export state from ordinary no-op
export behavior.

### RDNA4-RJ-187: `S_VERSION` accepts and displays nonzero upper-byte metadata

Reported by: local audit.

Manual evidence:

- Chapter 16.2 says `S_VERSION` is a no-op whose argument is ignored by
  hardware, but also says tool-visible `SIMM16[15:8]` must be zero at
  `rdna4/README.md:9121` through `:9135`.
- The checked XML conflict is recorded in `RDNA4-XML-149`: it names bits 13
  through 15 as `W64`, `W32`, and `MDP`.

Rocjitsu evidence:

- `SVersionSopk` stores the full 16-bit field as an `OPR_VERSION` operand at
  `rdna4/sopk.cpp:37` through `:43`.
- `execute_s_version_sopk` is an empty no-op at
  `shared/execute_shared.h:2672`, matching the runtime no-op behavior but not
  validating the tool/header contract.
- RDNA4 operand printing renders `OPR_VERSION` as the raw decimal
  `encoding_value_` at `rdna4/operand.cpp:572` through `:573`; no
  architecture-side validation or canonicalization masks `SIMM16[15:8]`.

Impact:

Rocjitsu can decode and disassemble `S_VERSION` encodings with nonzero
upper-byte bits as ordinary version operands. Execution remains a correct
no-op, but decoder/disassembly legality does not enforce the RDNA4 manual's
zero-high-byte contract.

### RDNA4-RJ-188: `S_MOVREL*` uses truncated and scaled `M0` offsets

Reported by: local audit.

Manual evidence:

- Chapter 6.1 summarizes `S_MOVRELS_{B32,B64}` and `S_MOVRELD_{B32,B64}` as
  `D = SGPR[S0+M0]` and `SGPR[D+M0] = S0`, says the B64 index must be even, and
  says `M0` is an unsigned index at `rdna4/README.md:2504`.
- Chapter 16.3 gives `S_MOVRELS_B32/B64` and `S_MOVRELD_B32/B64` as adding raw
  `M0.u32[31:0]` to the source or destination scalar-register address at
  `rdna4/README.md:10135` through `:10160` and `:10168` through `:10196`.
- `S_MOVRELSD_2_B32` uses source index `M0[9:0]` and destination index
  `M0[25:16]` at `rdna4/README.md:10204` through `:10210`.

Rocjitsu evidence:

- RDNA4 `SMovrelsB32Sop1` and `SMovrelsB64Sop1` use `wf.m0() & 0xFFu`, then
  compute `src_reg = base + index * width_words` at `rdna4/sop1.cpp:1047`
  through `:1051` and `:1076` through `:1080`. This truncates B32 offsets and
  also doubles B64 offsets instead of adding the raw even SGPR index.
- RDNA4 `SMovreldB32Sop1` and `SMovreldB64Sop1` use the same truncated,
  width-scaled pattern for the destination address at `rdna4/sop1.cpp:1105`
  through `:1109` and `:1134` through `:1138`.
- RDNA4 `SMovrelsd2B32Sop1` uses `src_index = M0[7:0]` and
  `dst_index = M0[15:8]` at `rdna4/sop1.cpp:1163` through `:1167`, rather than
  the manual's `M0[9:0]` source slice and `M0[25:16]` destination slice.

Impact:

Scalar relative moves can read or write the wrong SGPR whenever the offset does
not fit the current 8-bit slices, and the B64 forms can select the wrong pair
even for small even offsets such as `M0 = 2`.

### RDNA4-RJ-189: `S_SETKILL` executes as a no-op

Reported by: local audit.

Manual evidence:

- Chapter 16.5 says `S_SETKILL 1` kills the wave if the least significant bit of
  the immediate constant is 1 at `rdna4/README.md:11012` through `:11016`.

Rocjitsu evidence:

- RDNA4 `S_SETKILL` decodes and dispatches through `rdna4/sopp.cpp:30` through
  `:40`.
- `execute_s_setkill_sopp` is empty at `shared/execute_shared.h:2530`.

Impact:

Programs that use `S_SETKILL` for debug or host-command kill behavior continue
running even when `SIMM16[0]` requests wave termination.

### RDNA4-RJ-190: SOPP performance-counter instructions have no counter side effects

Reported by: local audit.

Manual evidence:

- Chapter 16.5 says `S_INCPERFLEVEL` increments and `S_DECPERFLEVEL` decrements
  the performance counter selected by `SIMM16[3:0]` at
  `rdna4/README.md:11518` through `:11524`.

Rocjitsu evidence:

- RDNA4 `S_INCPERFLEVEL` and `S_DECPERFLEVEL` decode and dispatch through
  `rdna4/sopp.cpp:426` through `:450`.
- `execute_s_incperflevel_sopp` and `execute_s_decperflevel_sopp` are empty at
  `shared/execute_shared.h:1776` through `:1777` and `:1637` through `:1638`.

Impact:

Shader-visible or tool-visible performance-counter instrumentation cannot be
observed through rocjitsu; both instructions are treated as ordinary no-ops.

### RDNA4-RJ-191: SMEM decoder accepts manual-omitted ATC probe opcodes

Reported by: local audit.

Manual/XML evidence:

- Chapter 15.2 Table 85 lists the RDNA4 SMEM opcode inventory but has no entries
  for opcodes 34 or 35 at `rdna4/README.md:6690` through `:6706`.
- Chapter 16.6 defines scalar loads, scalar-buffer loads, `S_DCACHE_INV`, and
  scalar prefetch instructions, then proceeds to VOP2 without defining
  `S_ATC_PROBE` or `S_ATC_PROBE_BUFFER` at `rdna4/README.md:11569` through
  `:12075`.
- XML nevertheless defines `S_ATC_PROBE` opcode 34 and
  `S_ATC_PROBE_BUFFER` opcode 35 as `ENC_SMEM` instructions at
  `amdgpu_isa_rdna4.xml:13149` and `:13193`, as recorded in
  `RDNA4-XML-153`.

Rocjitsu evidence:

- The generated RDNA4 SMEM decoder table routes opcodes 34 and 35 to
  `decodeSAtcProbeSmem` and `decodeSAtcProbeBufferSmem` at
  `rdna4/decoder.cpp:10639` through `:10640`.
- The generated classes expose `s_atc_probe` and `s_atc_probe_buffer` operands
  at `rdna4/smem.cpp:573` through `:601`.
- Both execute bodies are no-ops at `rdna4/smem.cpp:586` and `:601`.

Impact:

Rocjitsu decodes and disassembles binaries containing two RDNA4 manual-hole SMEM
opcodes as legal instructions. If the XML is the intended hardware source here,
rocjitsu still does not model any ATC probe or scalar-cache prefetch side
effects; if the manual is authoritative, the decoder accepts encodings that
should remain invalid.

### RDNA4-RJ-192: Compact VOP2 VCC operands are missing from metadata

Reported by: local audit.

Manual/XML evidence:

- Chapter 16.7 defines `V_CNDMASK_B32` as selecting between `S0` and `S1` using
  `VCC.u64[laneId]` at `rdna4/README.md:12086`, and says the VOP3 form can
  source the condition from an arbitrary scalar GPR at `:12091`.
- Chapter 16.7 defines `V_ADD_CO_CI_U32`, `V_SUB_CO_CI_U32`, and
  `V_SUBREV_CO_CI_U32` as reading a carry-in bit from `VCC.u64[laneId]` and
  writing carry-out back to `VCC.u64[laneId]` at `rdna4/README.md:12498`
  through `:12540`.
- XML carries non-binary-microcode `OPR_VCC` operands for these compact VOP2
  forms: `V_CNDMASK_B32` has a VCC input at `amdgpu_isa_rdna4.xml:69306`
  through `:69344`, and `V_ADD_CO_CI_U32` has VCC output and input operands at
  `amdgpu_isa_rdna4.xml:76384` through `:76429`.

Rocjitsu evidence:

- Generated `VCndmaskB32Vop2` publishes only `VDST`, `SRC0`, and `VSRC1`
  operands with `num_src_ = 2` and `num_dst_ = 1` at `rdna4/vop2.cpp:23`
  through `:36`.
- Generated carry-in/out VOP2 constructors likewise publish only `VDST`, `SRC0`,
  and `VSRC1` for `V_ADD_CO_CI_U32`, `V_SUB_CO_CI_U32`, and
  `V_SUBREV_CO_CI_U32` at `rdna4/vop2.cpp:3578`, `:3700`, and `:3822`.
- The VOP2 base `implicit_uses()` only adds old-destination reads for SDWA/DPP
  preservation at `rdna4/encodings.cpp:168` through `:179`; there is no
  matching VCC implicit-use or implicit-def hook.
- Execution still reads and writes the live wavefront VCC: `V_CNDMASK_B32`
  tests `wf.vcc()` at `shared/execute_shared.h:8257` through `:8272`, while
  the three carry forms read `wf.vcc()` and call `wf.set_vcc()` at
  `shared/execute_shared.h:2834` through `:2852`, `:18468` through `:18486`,
  and `:18934` through `:18952`.

Impact:

Def-use, liveness, scheduling, and legality passes that rely on generated
operand metadata cannot see compact VOP2 dependencies on VCC. A consumer can
move or rewrite a VCC producer/consumer pair incorrectly even though the runtime
execute body uses the hidden wavefront VCC state.

### RDNA4-RJ-193: `V_PIPEFLUSH` executes as an empty helper

Reported by: local audit.

Manual/XML evidence:

- Chapter 16.8 defines compact `V_PIPEFLUSH` opcode 27 as flushing the vector
  ALU pipeline through the destination cache at `rdna4/README.md:13201`
  through `:13203`.
- The VOP3 alias has the same wording at `rdna4/README.md:18658` through
  `:18660`.
- XML carries the operandless `V_PIPEFLUSH` instruction and repeats the flush
  description at `amdgpu_isa_rdna4.xml:59612` through `:59613`.

Rocjitsu evidence:

- Generated RDNA4 VOP1 and VOP3 constructors expose `V_PIPEFLUSH` as
  operandless instructions at `rdna4/vop1.cpp:3138` through `:3147` and
  `rdna4/vop3.cpp:1717` through `:1724`.
- Both generated execute paths delegate to shared helpers at
  `rdna4/vop1.cpp:3206` and `rdna4/vop3.cpp:1738`.
- The shared helpers are empty no-ops at `shared/execute_shared.h:16658` and
  `:16661`.

Impact:

Rocjitsu accepts and executes `V_PIPEFLUSH`, but it does not model the manual's
pipeline or destination-cache flush side effect. Any program depending on that
ordering/cache behavior is treated like it executed an ordinary no-op.

### RDNA4-RJ-194: Compact VOPC VCC/EXEC result masks are missing from metadata

Reported by: local audit.

Manual/XML evidence:

- Chapter 16.9 says VOPC compare instructions produce one result bit per lane
  into `VCC` or `EXEC` at `rdna4/README.md:14364`.
- The detailed `V_CMP_LT_F16` definition stores to `D0.u64[laneId]` and says
  `D0 = VCC in VOPC encoding` at `rdna4/README.md:14443` through `:14449`;
  `V_CMPX_CLASS_F16` instead writes `EXEC.u64[laneId]` and says to write only
  `EXEC` at `rdna4/README.md:16641` through `:16682`.
- XML carries the compact non-`CMPX` result as an `OPR_VCC` output for
  `V_CMP_LT_F16` at `amdgpu_isa_rdna4.xml:125999` through `:126013`, and
  carries the compact `CMPX` result as an `OPR_EXEC` output plus implicit
  `OPR_SDST_EXEC` for `V_CMPX_LT_F16` at `amdgpu_isa_rdna4.xml:144846`
  through `:144872`.

Rocjitsu evidence:

- Generated compact `VCmpLtF16Vopc` publishes only `src0` and `vsrc1` and sets
  `num_dst_ = 0` at `rdna4/vopc.cpp:23` through `:36`.
- Generated compact `VCmpxLtF16Vopc` likewise publishes only `src0` and
  `vsrc1` and sets `num_dst_ = 0` at `rdna4/vopc.cpp:8882` through `:8895`.
- The `Vopc` base class declares no `implicit_defs()` override at
  `rdna4/encodings.h:459` through `:484`, and its constructor only records
  common encoding fields at `rdna4/encodings.cpp:140` through `:154`; the base
  `Instruction::implicit_defs()` hook is empty at `isa/instruction.h:222`.
- Execution still mutates the hidden masks: shared non-`CMPX` helpers write
  `wf.set_vcc(vcc)`, for example `execute_v_cmp_lt_f16_vopc` at
  `shared/execute_shared.h:6151` through `:6171`, while generated `CMPX`
  execute bodies write `wf.set_exec(result)`, for example
  `VCmpxLtF16Vopc::execute_impl` at `rdna4/vopc.cpp:8917` through `:8990`.

Impact:

Def-use, liveness, scheduling, and legality passes that rely on generated
operand metadata cannot see compact VOPC compare definitions of `VCC` or
`EXEC`. A compare result can therefore be moved, removed, or consumed out of
order even though the runtime execute path mutates the live wavefront masks.

### RDNA4-RJ-195: VALU min/max-number helpers inherit host NaN, zero-tie, and exception behavior

Reported by: local audit.

Manual evidence:

- Chapter 16.7 defines `V_MIN_NUM_F32` / `V_MAX_NUM_F32` with full
  `minimumNumber()` / `maximumNumber()` pseudocode, including signaling-NaN
  invalid flagging, NaN quieting/source selection, signed-zero ordering,
  denorm-mode notes, exception flags, and saturation support at
  `rdna4/README.md:12331` through `:12393`.
- Chapter 16.11 repeats that edge behavior for VOPD `V_DUAL_MAX_NUM_F32` and
  `V_DUAL_MIN_NUM_F32` at `rdna4/README.md:17946` through `:18011` and
  `:18168` through `:18231`.

Rocjitsu evidence:

- Generated RDNA4 VOP2/VOP3 helpers implement `V_MAX_NUM_F32` with
  `util::stdx::fmax` in the SIMD path and host `std::fmax` in scalar fallback
  at `shared/execute_shared.h:14047` through `:14058`, and implement
  `V_MIN_NUM_F32` with `util::stdx::fmin` / `std::fmin` at
  `shared/execute_shared.h:15354` through `:15365`.
- RDNA4 VOPD implements `V_DUAL_MAX_NUM_F32` / `V_DUAL_MIN_NUM_F32` with direct
  `std::fmax` / `std::fmin` calls at `rdna4/vopd.cpp:192` through `:198`.
- The SIMD generator documents accepted divergences for these helpers: NaN
  inputs may differ in payload and signed-zero ties are skipped by guard tests
  at `lib/python/amdisa/codegen/execute/simd_codegen.py:188` through `:223`.
  The shared SIMD glue repeats that every min/max carries the same accepted
  NaN-payload / signed-zero-tie carve-out at `shared/simd_glue.h:763` through
  `:768`.

Impact:

For ordinary numeric inputs these helpers match the arithmetic operation, but
edge cases that the ISA defines explicitly can follow host-library behavior
instead of the RDNA4 contract. The emulator does not model the specified
signaling-NaN invalid flag, exact quiet-NaN/source selection, signed-zero
tie-breaking, denorm-mode dependency, or saturation/exception metadata for
these VALU min/max-number forms.

### RDNA4-RJ-196: VOP3 compare-DPP paths do not zero masked scalar/EXEC result bits

Reported by: local audit.

Manual evidence:

- Chapter 7.9.1 says DPP16 row and bank masks apply to the destination write
  and that VOPC disabled-lane SGPR/VCC bits receive zero at
  `rdna4/README.md:3434` through `:3435`.
- The same section says DPP with `V_CMP` or `V_CMPX` and `bound_ctrl=0` writes
  zero for lanes whose `EXEC` bit is zero, and `FI=1` still does not turn on an
  inactive lane for `V_CMPX`, at `rdna4/README.md:3467`.
- Chapter 16.12 VOP3 compare definitions write a scalar result mask for
  `V_CMP*` and write only `EXEC` for `V_CMPX*`; for example
  `V_CMPX_LT_U32` stores `EXEC.u64[laneId]` and says the destination must be
  `EXEC_LO` at `rdna4/README.md:24702` through `:24712`.

Rocjitsu evidence:

- Generated VOP3 compare constructors accept DPP8/DPP16 source-0 marker forms,
  for example `VCmpLtF16Vop3` loads DPP suffix state at
  `rdna4/vop3.cpp:23103` through `:23118`.
- `VCmpLtF16Vop3::execute_impl` computes the scalar compare mask across the
  active `EXEC` lanes and writes it to the scalar destination before computing
  a DPP write mask at `rdna4/vop3.cpp:23144` through `:23178`; the later mask
  path restores a VGPR destination at `:23179` through `:23182`, not scalar
  compare-result bits.
- `VCmpxGeU32Vop3::execute_impl` similarly computes a result mask, calls
  `wf.set_exec(result)`, and only then computes a DPP write mask whose restore
  path writes a VGPR indexed by the encoded destination at
  `rdna4/vop3.cpp:36140` through `:36159`.
- The related compact VOPC issue is tracked separately in `RDNA4-RJ-074`; this
  finding covers the VOP3-encoded compare forms audited in Chapter 16.12.

Impact:

VOP3 `V_CMP*`/`V_CMPX*` encodings with DPP can leave row/bank-masked or
inactive-lane scalar/EXEC result bits set according to the comparison instead
of forcing those bits to zero. The generic VGPR old-destination restore cannot
repair scalar compare masks and can write an unrelated VGPR when the encoded
destination is a scalar/EXEC result selector.

### RDNA4-RJ-197: Special DS atomics collapse to compare-swap or wrapping subtract

Reported by: local audit.

Manual/XML evidence:

- Chapter 16.15 defines `DS_CONDXCHG32_RTN_B64` as two independent 32-bit
  conditional exchanges at aligned LDS addresses. Each write is gated by the
  corresponding source dword's high bit, and the written value clears that high
  bit, at `rdna4/README.md:26620` through `:26645`.
- `DS_COND_SUB_U32` and `DS_COND_SUB_RTN_U32` subtract only when old LDS data
  is greater than or equal to the source, at `rdna4/README.md:26648` through
  `:26657` and `:26778` through `:26787`.
- `DS_SUB_CLAMP_U32` and `DS_SUB_CLAMP_RTN_U32` clamp underflow to zero, at
  `rdna4/README.md:26660` through `:26673` and `:26790` through `:26803`.
- XML carries these opcodes in `ENC_VDS` at `amdgpu_isa_rdna4.xml:48605`
  through `:48740` and `:49193` through `:49282`.

Rocjitsu evidence:

- The DS semantic map classifies `DS_CONDXCHG32_RTN_B64` as `cmpswap`,
  `COND_SUB` as `sub`, and `DS_SUB_CLAMP*` as `sub` at
  `lib/python/amdisa/semantics.py:1843`, `:2341`, and `:2362` through
  `:2363`; the generic atomic-op enum maps those strings only to
  `AtomicOp::CMPSWAP` and `AtomicOp::SUB` at
  `lib/python/amdisa/codegen/_generator.py:4848` through `:4853`.
- `DsCondxchg32RtnB64Vds` exposes only a B64 `DATA0` operand, but its execute
  body sets `AtomicOp::CMPSWAP` and reads the raw `DATA1` field as a hidden
  compare source at `rdna4/vds.cpp:3330` through `:3368`.
- `DsCondSubU32Vds`, `DsSubClampU32Vds`, `DsCondSubRtnU32Vds`, and
  `DsSubClampRtnU32Vds` all set `d->atomic_op = amdgpu::AtomicOp::SUB` at
  `rdna4/vds.cpp:3385` through `:3392`, `:3419` through `:3426`, `:3735`
  through `:3742`, and `:3771` through `:3778`.
- `AtomicOp` has no conditional-subtract, saturating-subtract, or
  condxchg32-specific operation at `vm/amdgpu/mem_state.h:45` through `:68`;
  the shared integer atomic helper implements `CMPSWAP` as equality compare and
  `SUB` as wrapping `old_val - src_val` at
  `vm/amdgpu/memory_pipeline.cpp:247` through `:259`.
- Local coverage found only generated zero-encoding fixtures for these DS
  opcodes at `rdna4/test_encodings.h:1191` through `:1205`, not execution
  cases that distinguish the special predicates from generic compare-swap or
  subtract.

Impact:

`DS_CONDXCHG32_RTN_B64` can update neither the right dword granularity nor the
manual's high-bit-gated values, and it can consume a hidden raw `DATA1` source
that the instruction does not expose. `DS_COND_SUB*` wraps underflow instead of
leaving LDS unchanged, while `DS_SUB_CLAMP*` wraps underflow instead of
storing zero. No-return variants also inherit the existing no-return DS atomic
writeback issue in `RDNA4-RJ-169`.

### RDNA4-RJ-198: Special VBUFFER atomics collapse to wrapping subtract

Reported by: local audit.

Manual/XML evidence:

- `BUFFER_ATOMIC_SUB_CLAMP_U32` clamps unsigned underflow to zero in its manual
  pseudocode at `rdna4/README.md:27742` through `:27762`.
- `BUFFER_ATOMIC_COND_SUB_U32` subtracts only when the old buffer value is
  greater than or equal to the data value, otherwise leaving memory unchanged,
  at `rdna4/README.md:28033` through `:28047`.
- XML records both opcodes with only prose descriptions plus generic U32 memory
  operands at `amdgpu_isa_rdna4.xml:41334` through `:41410` and
  `:42738` through `:42800`; the structured XML metadata gap is
  `RDNA4-XML-156`.

Rocjitsu evidence:

- `BufferAtomicSubClampU32Vbuffer` sets `d->atomic_op =
  amdgpu::AtomicOp::SUB` at `rdna4/vbuffer.cpp:1354` through `:1361`.
- `BufferAtomicCondSubU32Vbuffer` also sets `AtomicOp::SUB` at
  `rdna4/vbuffer.cpp:2327` through `:2334`.
- `AtomicOp` has no saturating-subtract or conditional-subtract variants at
  `vm/amdgpu/mem_state.h:45` through `:68`.
- The shared integer atomic helper implements `AtomicOp::SUB` as
  `old_val - src_val`, at `vm/amdgpu/memory_pipeline.cpp:247` through `:259`.

Impact:

`BUFFER_ATOMIC_SUB_CLAMP_U32` can wrap underflow instead of clamping to zero,
and `BUFFER_ATOMIC_COND_SUB_U32` can update memory in cases where the manual
requires the old value to be preserved.

### RDNA4-RJ-199: `V_CVT_F32_F16` VOP3 applies unsupported input modifiers

Reported by: local audit.

Manual evidence:

- Chapter 7.6 lists common data-conversion restrictions and says input
  modifiers `neg` and `abs` are not supported, at `rdna4/README.md:3128`
  through `:3131`.

Rocjitsu evidence:

- The generated RDNA4 `VCvtF32F16Vop3` constructor exposes a true16 `src0`
  operand at `rdna4/vop3.cpp:661` through `:677`.
- Its execute body reads the selected true16 half with
  `read_vop3_true16_src`, converts it with `util::f16_to_f32`, then applies
  `inst_.abs` and `inst_.neg` before writing `VDST`, at
  `rdna4/vop3.cpp:717` through `:728`.
- The shared `execute_v_cvt_f32_f16_vop3` helper ignores source modifiers at
  `shared/execute_shared.h:8782` through `:8795`, but this RDNA4 generated
  true16 body bypasses that helper for the scalar fallback path.

Impact:

An RDNA4 `V_CVT_F32_F16` VOP3 encoding with nonzero ABS or NEG modifier bits
can produce a sign-modified result instead of preserving the manual's
unsupported-modifier contract. This can hide illegal encodings or make
rocjitsu disagree with hardware/toolchain behavior for modifier-bearing
conversion encodings.

### RDNA4-RJ-200: State-register read-after-update delay cases are not modeled

Reported by: local audit.

Manual evidence:

- Chapter 5.7.2 says `S_SETHALT` followed by `S_GETREG` of `STATE_PRIV` or
  `STATUS` requires an intervening `S_NOP` or any other instruction, at
  `rdna4/README.md:2261` through `:2265`.
- The same list says `S_SETPRIO` followed by `S_GETREG STATE_PRIV` requires
  the same separation, at `rdna4/README.md:2263` through `:2266`.

Rocjitsu evidence:

- RDNA4 `S_SETHALT` dispatches through `rdna4/sopp.cpp:42` through `:51`, but
  `execute_s_sethalt_sopp` is empty at `shared/execute_shared.h:2527`.
- RDNA4 `S_SETPRIO` dispatches through `rdna4/sopp.cpp:389` through `:398`,
  but `execute_s_setprio_sopp` is empty at `shared/execute_shared.h:2533`.
- RDNA4 `S_GETREG_B32` reads the selected hardware register state directly
  during execute at `rdna4/sopk.cpp:116` through `:145`.
- `S_NOP`, `S_DELAY_ALU`, and `S_WAIT_ALU` execute through empty helpers at
  `shared/execute_shared.h:2138`, `:1641`, and `:2675`.
- The DBT `HazardTracker` models a generic recent SALU/VALU/TRANS history at
  `code/dbt/hazard_tracker.h:16` through `:30` and inserts only
  `S_DELAY_ALU` based on that pipeline class history at
  `code/dbt/hazard_tracker.cpp:11` through `:31`; searches found no
  `S_GETREG`/`S_SETHALT`/`S_SETPRIO` state-register hazard handling in the DBT
  paths.

Impact:

The current data-value behavior is already blocked by existing
`S_SETHALT`/`S_SETPRIO`/`S_GETREG` state gaps (`RDNA4-RJ-030`,
`RDNA4-RJ-036`, `RDNA4-RJ-005`, and `RDNA4-RJ-006`). Separately, rocjitsu has
no instruction-stream validator or DBT repair path for the Chapter 5.7.2
one-instruction separation rule, so fixing the state updates alone would still
leave immediate producer-to-`S_GETREG` sequences deterministic and silently
accepted.

### RDNA4-RJ-201: DBT `HazardTracker` encodes `S_DELAY_ALU` INSTID as a bitfield

Reported by: local audit.

Manual evidence:

- Chapter 5.8 defines `S_DELAY_ALU` as a packed `InstID1[4], Skip[3],
  InstID0[4]` operand at `rdna4/README.md:2283`.
- The dependency-code table is an enumeration: `1..4` mean previous ordinary
  VALU instructions one through four back, `5..7` mean previous transcendental
  VALU instructions one through three back, `8` is reserved, and `9..11` mean
  one through three SALU-cycle waits, at `rdna4/README.md:2295` through
  `:2301`.

Rocjitsu evidence:

- The DBT hazard tracker defines pipeline classes as `VALU = 1`, `TRANS = 2`,
  and `SALU = 3`, at `code/dbt/hazard_tracker.h:23` through `:28`.
- `HazardTracker::maybe_insert_delay` documents and implements
  `instid = pipeline[1:0] | (distance[2:0] << 2)` at
  `code/dbt/hazard_tracker.cpp:15` through `:23`.
- That formula does not match the manual enum. For example, an ordinary VALU
  producer at tracker distance 1 encodes `5`, which the manual defines as a
  one-back transcendental VALU dependency rather than a two-back ordinary VALU
  dependency. Distance 2 encodes `9`, which the manual defines as a SALU
  one-cycle wait.
- The tracker allows `distance == 4` and does not mask the result before
  OR-ing it into the instruction word at `code/dbt/hazard_tracker.cpp:20`
  through `:26`, so an unencodable ordinary-VALU distance can spill into the
  `INSTSKIP` field instead of being dropped.
- Current DBT tests found in this slice assert only the explicit
  `kDelayAluSaluDep1 = 9` prologue delay at
  `code/patch/instruction_builder.h:64`, `tests/dbt/device_kernels_translate_tests.cpp:260`
  through `:273`, and `tests/dbt/translate_test.cpp:1545` through `:1553`;
  they do not exercise the `HazardTracker` distance/class encoding.

Impact:

DBT-generated `S_DELAY_ALU` hints can describe the wrong dependency class or
distance once the dependency is not the immediately preceding producer. Since
Chapter 5.8 says `S_DELAY_ALU` is optional and not required for architectural
correctness, this is primarily a hardware scheduling and stall-avoidance
fidelity issue rather than a functional emulator state bug.

### RDNA4-RJ-202: Workgroup work-item limit is not enforced

Reported by: local audit.

Manual evidence:

- Chapter 2.3 says a WGP supports up to 32 work-groups in flight and a maximum
  of 1024 work-items per workgroup at `rdna4/README.md:618`.

Rocjitsu evidence:

- The command processor derives `wg_size` by multiplying
  `workgroup_size_x`, `workgroup_size_y`, and `workgroup_size_z`, then computes
  `wfs_per_wg` from that product at `command_processor.cpp:941` through
  `:944`.
- `DispatchEntry` stores the workgroup dimensions directly from the AQL packet
  at `dispatch_entry.h:81` through `:83`; searches found no Chapter 2
  max-1024 validation in the dispatch-entry or command-processor paths.
- The SPI allocation path dispatches every wave from `0` through
  `wfs_per_workgroup - 1` at `spi.h:109` through `:118`.

Impact:

Rocjitsu can accept and schedule an RDNA4 dispatch whose workgroup has more
than 1024 work-items. That can exercise resource, barrier, and LDS behavior
outside the Chapter 2 hardware contract instead of rejecting or flagging the
invalid launch.

### RDNA4-RJ-203: Graphics shader launch modes are not modeled

Reported by: local audit.

Manual evidence:

- Chapter 2.2 describes pixel, geometry, and hull shader waves, says the normal
  geometry-engine launch initializes VGPRs with primitive-index and
  vertex-buffer data, and describes mesh shader and amplification shader launch
  modes at `rdna4/README.md:603` through `:614`.

Rocjitsu evidence:

- `DispatchEntry` is compute/AQL shaped, with kernel dispatch identifiers,
  grid sizes, workgroup sizes, workgroup IDs, and WGP mode, at
  `dispatch_entry.h:43` through `:87`.
- `CommandProcessor::init_wavefront_regs` initializes AMDHSA compute SGPRs,
  workgroup IDs, RDNA4 TTMP fields, and local work-item IDs at
  `command_processor.cpp:174` through `:325`.
- Searches found no PS/GS/HS, geometry-engine, mesh-shader, or amplification
  shader launch path beyond generic shader-engine topology names. Existing
  `RDNA4-RJ-008` covers incomplete compute launch payload details; this entry
  records the broader absence of Chapter 2 graphics shader launch support.

Impact:

Rocjitsu is compute-dispatch oriented for this layer. It cannot emulate or
validate Chapter 2 graphics-stage wave creation, geometry-engine VGPR payloads,
mesh shader launch conversion, or amplification shader control.

## No Gap Found In This Slice

- RDNA4 Chapter 2 compute dispatch narrow match: the command processor and
  dispatch-entry paths create waves over a 1D/2D/3D workgroup grid and
  initialize local work-item IDs, matching the broad compute-shader launch
  concept in Chapter 2. Existing `RDNA4-RJ-008` and `RDNA4-RJ-202` track the
  missing launch-payload and validation pieces.
- RDNA4 Chapter 2 explicit WQM narrow match: generated `S_WQM_B32` and
  `S_WQM_B64` execute through shared helpers that expand any enabled lane in a
  four-lane group to the whole quad and update SCC. Temporary WQM cases remain
  covered by existing VINTERP, VDSDIR, and VSAMPLE gaps because those
  instruction families are currently stubbed or missing their detailed
  per-instruction semantics.
- RDNA4 Chapter 2 shader-padding boundary: `S_CODE_END` no-op execution is
  already recorded under `RDNA4-RJ-034`. The Chapter 2 prefetch padding rule is
  a code-object validation/host safety requirement rather than an additional
  state mutation for the functional instruction executor.
- RDNA4 Chapter 1 architecture-model boundary: rocjitsu has an RDNA4 device
  config with shader-engine/CU/cache/topology fields, RDNA wave-size traits, a
  command-processor dispatch path, WGP-mode LDS placement, L1/L2 cache objects,
  and completion callbacks. The remaining differences are the detailed
  operational gaps already recorded under later sections, including
  `RDNA4-RJ-016`, `RDNA4-RJ-018` through `RDNA4-RJ-025`,
  `RDNA4-RJ-049`, `RDNA4-RJ-050`, `RDNA4-RJ-115`, `RDNA4-RJ-116`,
  `RDNA4-RJ-148`, and `RDNA4-RJ-158` through `RDNA4-RJ-161`.
- RDNA4 Chapter 5.8 runtime note: generated RDNA4 preserves and disassembles
  the raw `S_DELAY_ALU` operand through `OPR_DELAY`, and executing
  `S_DELAY_ALU` as a no-op is architecturally acceptable for functional
  emulation because the manual says the instruction is optional and not needed
  for correct operation. The DBT encoding issue is tracked separately in
  `RDNA4-RJ-201`.
- RDNA4 Chapter 5.7.2 overlap note: the observable state-register values for
  the named sequences are already covered by `RDNA4-RJ-030`, `RDNA4-RJ-036`,
  `RDNA4-RJ-005`, and `RDNA4-RJ-006`; `RDNA4-RJ-200` records only the missing
  cross-instruction delay/validation contract.
- RDNA4 Chapter 7.6 sampled conversion narrow match: the checked RDNA4
  conversion bodies for shared integer/F32 conversion helpers, packed integer
  converts, and FP8/BF8 packed/stochastic converts did not show the same
  source-modifier application. The live sampled exception is the
  `V_CVT_F32_F16` VOP3 path recorded in `RDNA4-RJ-199`.
- RDNA4 Chapter 15 broad no-gap: generated machine-instruction structs preserve
  the raw Chapter 15 field layouts for SMEM, VOP*, VINTERP, VDSDIR, VDS,
  VBUFFER, VIMAGE, VSAMPLE, VEXPORT, and VFLAT/VGLOBAL/VSCRATCH. The only new
  source drift found in this pass is the XML/manual `NV` field mismatch recorded
  in `RDNA4-XML-148`, not a rocjitsu decoder drop.
- RDNA4 Chapter 15 SMEM `NV` note: the generated SMEM machine instruction
  stores `nv` between `pad_19` and `scope`, and disassembly prints `nv` when
  set, at `rdna4/machine_insts.h:28` through `:34` and
  `rdna4/encodings.cpp:103` through `:107`.
- RDNA4 Chapter 16.1 SOP2 decoder narrow match: generated RDNA4 SOP2 mnemonic
  coverage matches the 74 normalized manual/XML definitions, with the expected
  invalid opcode holes preserved in the decoder table. The pack opcodes 50
  through 53 route to `decodeSPack*B32B16Sop2`, and the FP/F16/U64 tail routes
  through opcodes 64 through 85 at `rdna4/decoder.cpp:6578` through `:6665`.
- RDNA4 Chapter 16.1 scalar pack execution narrow match: generated pack
  constructors expose the manual half-width source shapes at
  `rdna4/sop2.cpp:1154` through `:1255`, and the shared helpers implement the
  Chapter 16 formulas for LL/LH/HH/HL half selection at
  `shared/execute_shared.h:2344` through `:2373`.
- RDNA4 Chapter 16.1 literal-only FMA narrow match: generated
  `SFmaakF32Sop2` always exposes the extension word as `src2` and executes
  `fma(S0, S1, SIMM32)`, while `SFmamkF32Sop2` executes
  `fma(S0, SIMM32, S1)`, at `rdna4/sop2.cpp:1374` through `:1437`. Signed
  64-bit scalar literal sign extension remains the preexisting
  `RDNA4-RJ-022`; SCC, scalar FP MODE, and scalar F16 metadata gaps remain
  `RDNA4-RJ-105`, `RDNA4-RJ-107`, and `RDNA4-RJ-108`.
- RDNA4 Chapter 16.2 SOPK decoder narrow match: generated RDNA4 SOPK mnemonic
  coverage matches all nine manual/XML definitions, with invalid opcode holes
  preserved between opcodes 2 and 15 and after opcode 20 at
  `rdna4/decoder.cpp:6676` through `:6702`.
- RDNA4 Chapter 16.2 SOPK literal narrow match: ordinary SOPK forms keep the
  32-bit instruction size, while opcode 19 `S_SETREG_IMM32_B32` is the only
  implied-literal SOPK form through `Sopk::hasImpliedLiteral()` at
  `rdna4/encodings.cpp:57` through `:72`; its execute body consumes
  `literal_` at `rdna4/sopk.cpp:187` through `:221`.
- RDNA4 Chapter 16.2 SOPK execution narrow match: `S_MOVK_I32` and
  `S_CMOVK_I32` sign-extend the 16-bit immediate, `S_MULK_I32` multiplies
  without writing SCC, and `S_VERSION` is a no-op at
  `shared/execute_shared.h:1095` through `:1100`, `:2033` through `:2037`,
  `:2093` through `:2098`, and `:2672`. `S_ADDK_CO_I32`, `S_GETREG_B32`, and
  `S_SETREG*` retain the existing SCC/HWREG gaps already recorded in
  `RDNA4-RJ-005`, `RDNA4-RJ-006`, and `RDNA4-RJ-011`.
- RDNA4 Chapter 16.2 `S_CALL_B64` branch narrow match: the generated
  instruction saves `PC + size_`, sign-extends `SIMM16`, scales by four bytes,
  and adjusts for the generic post-execute PC advance at `rdna4/sopk.cpp:224`
  through `:244`; branch metadata and broad control-flow behavior remain
  covered by the Chapter 5 entries.
- RDNA4 Chapter 16.3 SOP1 decoder boundary: generated RDNA4 SOP1 mnemonic
  coverage matches the 84 XML/generated entries after normalizing the manual
  `S_WQM_B32` markdown artifact; the missing manual opcode is the existing
  `S_GET_BARRIER_STATE` gap recorded in `RDNA4-RJ-028`. The generated decoder
  preserves invalid opcode holes through the SOP1 table, including the control
  cluster around `S_BARRIER_SIGNAL*`, `S_ALLOC_VGPR`, and `S_SLEEP_VAR` at
  `rdna4/decoder.cpp:7120` through `:7205`.
- RDNA4 Chapter 16.3 WQM/count/scan narrow match: generated `S_WQM_B32/B64`
  dispatch through `rdna4/decoder.cpp:7135` through `:7136`, and the shared
  helpers expand any set lane in each quad to the full quad at
  `shared/execute_shared.h:2693` through `:2718`; count/scan no-match cases
  remain covered by the earlier Chapter 6 narrow-match note.
- RDNA4 Chapter 16.3 direct-PC narrow match: generated `S_GETPC_B64`,
  `S_SETPC_B64`, and `S_SWAPPC_B64` store the next-instruction PC and apply the
  source PC through `rdna4/sop1.cpp:1174` through `:1225`; branch alignment,
  branch-to-zero, literal-source, and trap-return limitations remain in the
  existing Chapter 5 gaps.
- RDNA4 Chapter 16.3 semantic boundary: `S_SENDMSG_RTN*`, `S_BARRIER_SIGNAL*`,
  `S_ALLOC_VGPR`, `S_SLEEP_VAR`, scalar FP/F16 conversion, SAVEEXEC/WREXEC, and
  SCC metadata issues overlap with existing gaps; the new implementation issue
  found in this SOP1 pass is the scalar relative-index execution drift recorded
  in `RDNA4-RJ-188`.
- RDNA4 Chapter 16.4 SOPC decoder narrow match: generated RDNA4 SOPC mnemonic
  coverage matches all 46 manual/XML definitions, with invalid opcode holes
  preserved in the table at `rdna4/decoder.cpp:7366` through `:7420`.
- RDNA4 Chapter 16.4 SOPC literal narrow match: `Sopc::default_encoding()` keeps
  non-literal forms 32-bit and adds one extension word when either source uses
  selector 255 at `rdna4/encodings.cpp:27` through `:43`; generated constructors
  expose the same extension word through source 0, source 1, or both sources.
- RDNA4 Chapter 16.4 SOPC execution narrow match: integer comparisons, unsigned
  64-bit equality/inequality, floating ordered/unordered/negated predicates, and
  bit-compare index masks match the checked Chapter 16.4 formulas in the shared
  helpers. Representative bit-compare helpers mask with 31 or 63 at
  `shared/execute_shared.h:897` through `:925`, and representative floating
  ordered/unordered helpers use `std::isnan` at `:1382` through `:1408`.
- RDNA4 Chapter 16.4 semantic boundary: generated SOPC classes still inherit the
  existing SCC def-use metadata gap in `RDNA4-RJ-105`, scalar F16 inline-constant
  gap in `RDNA4-RJ-106`, and scalar FP MODE/exception gap in `RDNA4-RJ-107`; no
  new rocjitsu implementation gap was found in this SOPC pass.
- RDNA4 Chapter 16.5 SOPP decoder boundary: generated RDNA4 SOPP coverage has
  42 opcode entries, preserving the manual's 40 entries plus the XML-derived
  trace opcodes 58/59 that are already tracked in `RDNA4-RJ-026`. The generated
  decoder table also preserves the invalid opcode holes between the documented
  clusters at `rdna4/decoder.cpp:7435` through `:7510`.
- RDNA4 Chapter 16.5 SOPP branch narrow match: `S_BRANCH` and all six
  `S_CBRANCH_*` generated execute bodies sign-extend `SIMM16`, scale by four
  bytes, and use the manual's next-PC base; conditional branches test SCC, live
  VCC, and EXEC zero/nonzero through `rdna4/sopp.cpp:216` through `:360`.
- RDNA4 Chapter 16.5 SOPP semantic boundary: trap, halt, sleep/wakeup, clause,
  barrier, send-message, wait-counter, cache-maintenance, code-end, and wait
  event/idling behavior overlaps with existing Chapter 5 gaps. The new
  implementation gaps found in the full SOPP definition pass are `S_SETKILL` and
  the performance-counter instructions in `RDNA4-RJ-189` and `RDNA4-RJ-190`.
- RDNA4 Chapter 16.6 SMEM decoder boundary: generated RDNA4 SMEM coverage has
  28 opcode entries, preserving the manual's 26 definitions plus the XML-derived
  ATC probe opcodes 34/35 tracked in `RDNA4-RJ-191`. Other invalid opcode holes
  are preserved around the documented load, invalidation, and prefetch clusters
  at `rdna4/decoder.cpp:10608` through `:10645`.
- RDNA4 Chapter 16.6 SMEM load narrow match: generated scalar load and
  scalar-buffer load classes wire the checked access widths, signedness, and
  `KMcnt` memory-op setup for B32/B64/B96/B128/B256/B512 and I8/U8/I16/U16
  forms at `rdna4/smem.cpp:31` through `:561`; address, buffer-resource,
  `SOFFSET`, and range/alignment limitations remain the existing
  `RDNA4-RJ-110`, `RDNA4-RJ-111`, `RDNA4-RJ-112`, and `RDNA4-RJ-117`.
- RDNA4 Chapter 16.6 SMEM cache/prefetch semantic boundary: `S_DCACHE_INV`
  remains represented as operandless scalar-cache invalidation at
  `rdna4/smem.cpp:564` through `:571`; wait-counter, clause/group, and prefetch
  side-effect limitations remain `RDNA4-RJ-023`, `RDNA4-RJ-116`, and
  `RDNA4-RJ-118`. The new source-drift issue in this pass is the ATC probe
  decoder acceptance recorded in `RDNA4-RJ-191`.
- RDNA4 Chapter 16.7 VOP2 decoder boundary: generated RDNA4 VOP2 coverage
  matches the manual's 49 definitions after accounting for repeated compact
  opcode-table entries and the four implied-literal FMAMK/FMAAK opcodes; invalid
  opcode holes are preserved through the decoder table at
  `rdna4/decoder.cpp:6440` through `:6575`.
- RDNA4 Chapter 16.7 VOP2 literal/promoted-form boundary: generated VOP2 sizing
  treats opcodes 44, 45, 55, and 56 as implied-literal forms at
  `rdna4/encodings.cpp:156` through `:191`, and generated opcodes keep
  FMAMK/FMAAK/PK_FMAC as VOP2-only forms with no matching generated VOP3
  classes. The remaining DPP/source-combination legality issues stay covered by
  `RDNA4-RJ-073` and `RDNA4-RJ-092`.
- RDNA4 Chapter 16.7 VOP2 semantic boundary: most arithmetic, min/max, true16,
  DPP, source-legality, and unimplemented packed-FMAC behavior overlaps with
  existing Chapter 7 gaps, including scalar MODE/exception limitations,
  DPP legality/modifier/disassembly gaps, wave64 mask/carry validation, and the
  `V_PK_FMAC_F16` unimplemented body in `RDNA4-RJ-095`. The new implementation
  issue found in this VOP2 definition pass is the compact VOP2 VCC metadata gap
  recorded in `RDNA4-RJ-192`.
- RDNA4 Chapter 16.8 VOP1 decoder boundary: generated RDNA4 VOP1 coverage
  matches the manual's 90 definitions, preserving invalid opcode holes in the
  compact decoder table at `rdna4/decoder.cpp:6972` through `:7085`, and the
  corresponding VOP3 aliases are generated where the manual permits promotion.
- RDNA4 Chapter 16.8 VOP1 true16/FP8 narrow match: packed FP8/BF8 VOP1
  converts use 16-bit `OPR_SRC` operands in generated code at
  `rdna4/vop1.cpp:11217` through `:11447`, while the RDNA4 operand layer maps
  compact true16 source values 0-127 to `vN.l` and 128-255 to `vN.h` at
  `rdna4/operand.cpp:34` through `:47` and `:1091` through `:1096`.
  This covers the manual's VOP1 OPSEL16 source-half selection for these packed
  converts.
- RDNA4 Chapter 16.8 VOP1 special-form boundary: `V_READFIRSTLANE_B32`
  publishes an SGPR destination and reads the lowest active lane, `V_SWAP_B32`
  and `V_SWAP_B16` publish both operands as read/write, and
  `V_PERMLANE64_B32` snapshots the source and swaps lanes 0-31 with 32-63
  while performing no writes in wave32. Existing gaps cover `V_MOVRELD_B32` /
  `V_MOVRELS_B32` signed/OOR behavior (`RDNA4-RJ-081`), `V_MOVRELSD*` /
  `V_SWAPREL_B32` unimplemented bodies (`RDNA4-RJ-082`), and
  `V_SAT_PK_U8_I16` as part of the broader VALU unimplemented set
  (`RDNA4-RJ-095`); the new issue found in this pass is `V_PIPEFLUSH` in
  `RDNA4-RJ-193`.
- RDNA4 Chapter 16.9 VOPC decoder boundary: generated RDNA4 VOPC coverage
  matches the manual's 162 concrete compact definitions after normalizing
  markdown conversion artifacts around several detailed compare headings, and
  the compact decoder table preserves the opcode range through
  `rdna4/decoder.cpp:6839`. The `F`/`TRU` compare-operation summary rows remain
  non-concrete operations, matching the XML/generated absence recorded in the
  Chapter 7.3 ambiguity note.
- RDNA4 Chapter 16.9 VOPC compare/class semantics narrow match: ordinary
  compare helpers write only active-lane results to VCC, `CMPX` bodies write
  the result to EXEC, and class-compare helpers follow the manual's ten-bit
  mask order for F16/F32/F64 at `shared/execute_shared.h:3961` through `:4028`,
  `:4101` through `:4158`, and `:4238` through `:4298`. Existing gaps still
  cover DPP masked-lane zeroing (`RDNA4-RJ-074`), DPP modifier/legal-form
  handling (`RDNA4-RJ-072`/`RDNA4-RJ-073`), source-combination and wave64 mask
  restrictions (`RDNA4-RJ-092`/`RDNA4-RJ-093`), VOP3 scalar-destination
  legality (`RDNA4-RJ-086`), and scalar FP exception/MODE behavior
  (`RDNA4-RJ-107`); the new issue found in this pass is the compact VOPC
  result metadata gap in `RDNA4-RJ-194`.
- RDNA4 Chapter 16.10 VOP3P decoder boundary: generated RDNA4 VOP3P coverage
  matches the manual's 56 concrete definitions after normalizing markdown
  conversion artifacts around several detailed headings. Generated opcode
  constants cover opcodes 0 through 90 with the manual's holes at
  `rdna4/opcodes.h:1034` through `:1089`, and the compact decoder table routes
  the same 56 non-invalid entries at `rdna4/decoder.cpp:7829` through `:7921`.
- RDNA4 Chapter 16.10 VOP3P semantic boundary: the detailed DOT, MIX, packed
  F16, WMMA, and SWMMAC rows do not add a new gap beyond the existing VOP3P
  entries. DOT BF8 decoding/modifiers/disassembly remain `RDNA4-RJ-053`
  through `RDNA4-RJ-055`; SWMMAC and WMMA layout, legality, modifier,
  disassembly, and hazard issues remain `RDNA4-RJ-056` through `RDNA4-RJ-062`;
  packed inline constants, clamp, min/max-number distinction, MIX fusion, and
  source legality remain `RDNA4-RJ-098` through `RDNA4-RJ-102`; and DPP
  sizing/legal-selector/disassembly behavior remains `RDNA4-RJ-071`,
  `RDNA4-RJ-073`, `RDNA4-RJ-076`, and `RDNA4-RJ-077`.
- RDNA4 Chapter 16.11 VOPD decoder boundary: generated VOPD slot constants and
  names cover the manual's 17-entry union, while decode extracts `OPX` as
  4 bits and `OPY` as 5 bits at `rdna4/vopd.cpp:21` through `:38` and
  `:222` through `:225`. This preserves the manual's X-slot `0..13` and
  Y-slot `0..13,16,17,18` shape.
- RDNA4 Chapter 16.11 VOPD ordinary execution boundary: rocjitsu uses fused
  FMA for FMAC/FMAAK/FMAMK, implements DX9-zero multiply, CNDMASK over VCC,
  add/sub/subrev, add-no-carry, shift, and AND in the VOPD slot dispatcher at
  `rdna4/vopd.cpp:146` through `:205`. The remaining VOPD-specific defects are
  already recorded in `RDNA4-RJ-063` through `RDNA4-RJ-070`; the new
  family-wide min/max edge-fidelity issue is `RDNA4-RJ-195`.
- RDNA4 Chapter 16.11 VOPD literal and operand narrow match: rocjitsu sizes
  VOPD as 96 bits when either source-0 is literal or either slot is FMAAK/FMAMK,
  constructs `SRC0` as `OPR_SRC` and `VSRC1` as `OPR_VGPR`, formats literal
  FMA forms with the extension word, and keeps CNDMASK's VCC form implicit at
  `rdna4/vopd.cpp:233` through `:248`, `:270` through `:279`, and
  `:317` through `:341`.
- RDNA4 Chapter 16.12 VOP3/VOP3SD inventory narrow match: generated RDNA4 has
  444 VOP3-family classes and 444 VOP3-family opcode constants, matching the
  normalized manual/XML instruction records. The ten VOP3SD entries are the
  manual-listed add/sub carry, div-scale, and `MAD_CO` forms.
- RDNA4 Chapter 16.12 VOP3 decoder boundary: generated decode tables route the
  VOP3 opcode space through the generated `V*Vop3` classes and the VOP3SD
  subset through `V*Vop3SdstEnc` classes; invalid opcode holes remain table
  holes rather than aliasing to neighboring definitions.
- RDNA4 Chapter 16.12 VOP3 semantic boundary: ordinary literal, true16,
  VOP3SD scalar-destination, compare-result, and generic modifier paths are
  generated. The remaining issues are the earlier generic VOP3 legality,
  disassembly, DPP sizing/legalization, source-combination, clamp, CMPX hazard,
  and min/max edge-fidelity gaps, plus the new VOP3 compare-DPP result-mask
  issue in `RDNA4-RJ-196`.
- RDNA4 Chapter 16.13 VINTERP inventory narrow match: generated RDNA4 exposes
  six VINTERP opcode constants, six VINTERP decoder entries for opcodes 0
  through 5, and invalid entries for opcodes 6 through 31, matching the
  manual/XML instruction set at `rdna4/opcodes.h:1090` through `:1095` and
  `rdna4/decoder.cpp:7962` through `:7970`.
- RDNA4 Chapter 16.13 VINTERP operand-width boundary: generated constructors
  model the XML/manual F32 and F16/F32 source/destination widths, and the two
  F16 P2 forms add old-`VDST` implicit uses for destination-half preservation
  at `rdna4/vinterp.cpp:19` through `:142`. The execution and wait-counter
  gaps remain the existing `RDNA4-RJ-179` and `RDNA4-RJ-180`.
- RDNA4 Chapter 16.14 VDSDIR inventory narrow match: generated RDNA4 exposes
  `kDsParamLoadVdsdir = 0` and `kDsDirectLoadVdsdir = 1`, routes opcodes 0
  and 1 to `DsParamLoadVdsdir` and `DsDirectLoadVdsdir`, and leaves opcodes
  2/3 invalid, matching the manual/XML instruction set at `rdna4/opcodes.h:1096`
  through `:1097` and `rdna4/decoder.cpp:7999` through `:8005`.
- RDNA4 Chapter 16.14 VDSDIR raw-field and operand boundary: generated
  machine/build helpers preserve `VDST`, `ATTR_CHAN`, `ATTR`,
  `WAIT_VA_VDST`, `OP`, and `WAIT_VM_VSRC` at `rdna4/machine_insts.h:111`
  through `:119` and `rdna4/builders.h:296` through `:316`; constructors expose
  `DS_PARAM_LOAD` as `VDST` plus `ATTR` and `DS_DIRECT_LOAD` as only `VDST` at
  `rdna4/vdsdir.cpp:19` through `:43`. The missing M0/LDS/quad/readiness/wait
  behavior remains the earlier `RDNA4-RJ-162` through `RDNA4-RJ-166` plus
  `RDNA4-RJ-181`.
- RDNA4 Chapter 16.15 VDS inventory narrow match: generated RDNA4 has 123 VDS
  opcode constants, 123 VDS classes, and decoder entries matching the
  normalized manual/XML opcode set. Constants span `rdna4/opcodes.h:1098`
  through `:1220`, the decoder table spans `rdna4/decoder.cpp:9298` through
  `:9557`, and generated zero-encoding fixtures span
  `rdna4/test_encodings.h:1097` through `:1219`.
- RDNA4 Chapter 16.15 VDS raw-field and execution boundary: the generated
  machine instruction and builder preserve `OFFSET0`, `OFFSET1`, `OP`, `ADDR`,
  `DATA0`, `DATA1`, and `VDST` at `rdna4/machine_insts.h:122` through `:132`
  and `rdna4/builders.h:328` through `:340`. Ordinary load/store, 2-address,
  stride64, narrow D16, append/consume, packed-FP, lane-permute/swizzle, BVH,
  and wait-counter issues remain covered by `RDNA4-RJ-167` through
  `RDNA4-RJ-178`; the new special DS atomic execution issue is
  `RDNA4-RJ-197`.
- RDNA4 Chapter 16.16 VBUFFER inventory narrow match: generated RDNA4 has 89
  VBUFFER opcode constants, 89 VBUFFER classes, and decoder entries matching
  the normalized manual/XML opcode set. Constants span `rdna4/opcodes.h:1221`
  through `:1309`, and the decoder table includes the same opcode holes and
  entries from `rdna4/decoder.cpp:7590` through `:7713`.
- RDNA4 Chapter 16.16 VBUFFER raw-field and execution boundary: the generated
  machine instruction and builder preserve `SOFFSET`, `NV`, `OP`, `TFE`,
  `VDATA`, `RSRC`, `SCOPE`, `TH`, `FORMAT`, `OFFEN`, `IDXEN`, `VADDR`, and
  `IOFFSET` at `rdna4/machine_insts.h:134` through `:152` and
  `rdna4/builders.h:360` through `:381`. Addressing, TFE, formatted/typed
  execution, descriptor data controls, alignment, disassembly fields, and
  packed FP atomics remain covered by `RDNA4-RJ-119` through `RDNA4-RJ-124`
  plus `RDNA4-RJ-183`; the new VBUFFER special-atomic execution issue is
  `RDNA4-RJ-198`.
- RDNA4 Chapter 16.17 VIMAGE inventory narrow match: generated RDNA4 has 33
  VIMAGE opcode constants, 33 VIMAGE classes, and decoder entries matching the
  normalized manual/XML opcode set. Constants span `rdna4/opcodes.h:1310`
  through `:1342`, classes span `rdna4/vimage.h:17` through `:275`, and the
  decoder table includes the same opcode holes and entries from
  `rdna4/decoder.cpp:8008` through `:8145`.
- RDNA4 Chapter 16.17 VIMAGE raw-field and execution boundary: the generated
  machine instruction and builder preserve `DIM`, `R128`, `D16`, `A16`, `NV`,
  `OP`, `DMASK`, `VDATA`, `RSRC`, `SCOPE`, `TH`, `TFE`, and `VADDR0..4` at
  `rdna4/machine_insts.h:154` through `:176` and `rdna4/builders.h:381`
  through `:425`. The image pipeline remains decode-only/no-op or
  unimplemented at execution time, with representative stubs at
  `rdna4/vimage.cpp:19` through `:33`, `:179` through `:193`, and `:402`
  through `:462`; those semantics are already covered by `RDNA4-RJ-125`
  through `RDNA4-RJ-138`, and image packed atomics by `RDNA4-RJ-131`.
- RDNA4 Chapter 16.18 VSAMPLE inventory narrow match: generated RDNA4 has 58
  VSAMPLE opcode constants, 58 VSAMPLE classes, and decoder entries matching
  the normalized manual/XML opcode set. Constants span `rdna4/opcodes.h:1343`
  through `:1400`, classes span `rdna4/vsample.h:17` through `:531`, and the
  decoder table includes the same opcode holes and entries from
  `rdna4/decoder.cpp:9585` through `:9705`.
- RDNA4 Chapter 16.18 VSAMPLE raw-field and execution boundary: the generated
  machine instruction and builder preserve `DIM`, `TFE`, `R128`, `D16`, `A16`,
  `NV`, `UNORM`, `OP`, `DMASK`, `VDATA`, `LWE`, `RSRC`, `SCOPE`, `TH`, `SAMP`,
  and `VADDR0..3` at `rdna4/machine_insts.h:178` through `:200` and
  `rdna4/builders.h:425` through `:472`. All 58 generated execute bodies are
  image-pipeline no-op stubs, with representative MSAA/sample/gather/GET_LOD
  bodies at `rdna4/vsample.cpp:19` through `:50`, `:375` through `:402`, and
  `:528` through `:557`; those semantics are already covered by
  `RDNA4-RJ-125` through `RDNA4-RJ-138`.
- RDNA4 Chapter 16.19 VEXPORT singleton narrow match: generated RDNA4 exposes
  `kExportVexport = 0`, one `ExportVexport` class, and `decodeExportVexport`
  entries for the singleton `ENC_VEXPORT` path at `rdna4/opcodes.h:1401`,
  `rdna4/vexport.h:17`, and `rdna4/decoder.cpp:6818` through `:6827`.
- RDNA4 Chapter 16.19 VEXPORT raw-field and execution boundary: the generated
  machine instruction and builder preserve `EN`, `TGT`, `DONE`, `ROW_EN`, and
  `VSRC0..3` at `rdna4/machine_insts.h:202` through `:214` and
  `rdna4/builders.h:474` through `:501`. The generated constructor exposes
  `TGT` plus four unconditional VGPR sources and the execute body is a
  compute-simulation no-op at `rdna4/vexport.cpp:19` through `:38`; those
  semantics are already covered by `RDNA4-RJ-051`, `RDNA4-RJ-184`, and
  `RDNA4-RJ-185`.
- RDNA4 Chapter 16.20 VFLAT/VSCRATCH/VGLOBAL inventory narrow match: generated
  RDNA4 has 144 opcode constants, 144 instruction classes, and decoder entries
  matching the normalized manual/XML opcode set: 55 VFLAT, 24 VSCRATCH, and 65
  VGLOBAL definitions. Constants span `rdna4/opcodes.h:1402` through `:1545`,
  and the decoder functions/tables span `rdna4/decoder.cpp:5606` through
  `:6197`, `:9838` through `:9912`, `:10099` through `:10167`, and `:10360`
  through `:10459`.
- RDNA4 Chapter 16.20 raw-field and execution boundary: the generated machine
  instructions and builders preserve `SADDR`, `NV`, `OP`, `VDST`, `SVE`,
  `SCOPE`, `TH`, `VSRC`, `VADDR`, and `IOFFSET` for VFLAT, VSCRATCH, and
  VGLOBAL at `rdna4/machine_insts.h:217` through `:268` and
  `rdna4/builders.h:501` through `:592`. Ordinary byte/short/dword loads and
  stores, D16/D16_HI merge behavior, ADDTID globals, cache ops, block
  transfers, transpose loads, and integer/FP/packed atomics are all generated;
  the known semantic and metadata issues remain covered by Chapter 11/13
  entries `RDNA4-RJ-139` through `RDNA4-RJ-157`, `RDNA4-RJ-020`,
  `RDNA4-RJ-050`, and `RDNA4-RJ-183`.
- RDNA4 Chapter 14 narrow match: the generated RDNA4 decoder and builders
  preserve the raw `ENC_VEXPORT` field bits, including `EN`, `TGT`, `DONE`,
  `ROW_EN`, and `VSRC0..3`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h:202`
  through `:214` and `rdna4/builders.h:474` through `:501`. The generated
  operand layer also names the visible export targets at `rdna4/operand.cpp:500`
  through `:533`.
- RDNA4 Chapter 3.5.1 active-thread EXEC initialization is represented in the
  production dispatch paths: after allocating each wave, both
  `CommandProcessor::dispatch_workgroups()` and `ShaderProcessorInput::advance()`
  call `initial_exec_mask_for_wave()` before register initialization at
  `vm/amdgpu/command_processor.cpp:710` through `:719` and
  `vm/amdgpu/spi.h:109` through `:118`. The helper builds the mask from the
  workgroup dimensions, wave index, lane id, grid bounds, and workgroup-id
  offset at `vm/amdgpu/dispatch_entry.h:206` through `:239`. The all-lanes
  value assigned by low-level `ComputeUnitCore::dispatch_wf()` is a temporary
  allocation default for these launch paths, not the final compute-dispatch
  EXEC state.
- RDNA4 Chapter 12.3 decode narrow match: generated VINTERP decode routes
  opcodes 0 through 5 to the six VINTERP classes and opcodes 6 through 31 to
  invalid decode entries at `rdna4/decoder.cpp:7962` through `:7970`.
- RDNA4 Chapter 12.3 raw-field narrow match: the VINTERP machine instruction
  preserves `VDST`, `WAIT_EXP`, `OPSEL`, `CLAMP`, `OP`, `SRC0`, `SRC1`,
  `SRC2`, and `NEG`, and the builder writes those fields at
  `rdna4/machine_insts.h:96` through `:109` and `rdna4/builders.h:278`
  through `:292`.
- RDNA4 Chapter 12.3 operand-width narrow match: generated VINTERP constructors
  model F32 and F16/F32 operand widths according to the XML, and the two F16
  P2 destination forms add old-`VDST` implicit uses for destination-half
  preservation, at `rdna4/vinterp.cpp:19` through `:142`.
- RDNA4 Chapter 12.3 test note: generated zero-encoding fixtures cover all six
  VINTERP mnemonics at `rdna4/test_encodings.h:1089` through `:1094`, but no
  dedicated execution tests were found for interpolation arithmetic or
  `WAIT_EXP`.
- RDNA4 Chapter 12.2.1 decode narrow match: generated VDSDIR decode routes
  opcode 0 to `DS_PARAM_LOAD`, opcode 1 to `DS_DIRECT_LOAD`, and opcodes 2/3
  to invalid decode entries at `rdna4/decoder.cpp:7999` through `:8008`.
- RDNA4 Chapter 12.2.1 raw-field narrow match: the VDSDIR machine instruction
  preserves `VDST`, `ATTR_CHAN`, `ATTR`, `WAIT_VA_VDST`, `OP`, and
  `WAIT_VM_VSRC`, and the builder writes those bit positions at
  `rdna4/machine_insts.h:111` through `:119` and `rdna4/builders.h:295`
  through `:316`. The higher-level operand/runtime gaps are recorded in
  `RDNA4-RJ-163`.
- RDNA4 Chapter 12.2.1 operand-name narrow match: `attr0` through `attr32`
  operand names are generated in `rdna4/operand_types.h:56` through `:88` and
  `rdna4/operand.cpp:103` through `:137`.
- RDNA4 Chapter 12.2.1 test note: no direct C++ or Python execution tests were
  found for `ds_param_load`, `vdsdir`, or `lds_param`; the explicit RDNA4
  coverage found in this slice is a generated zero-encoding fixture at
  `rdna4/test_encodings.h:1095`.
- RDNA4 Chapter 12.4 decode narrow match: generated VDSDIR decode routes
  opcode 1 to `DS_DIRECT_LOAD`, with opcodes 2/3 invalid, at
  `rdna4/decoder.cpp:7999` through `:8008`; the zero-encoding fixture includes
  `ds_direct_load` at `rdna4/test_encodings.h:1096`.
- RDNA4 Chapter 12.4 operand-syntax narrow match: direct load exposes only
  `VDST`, matching the manual example `DS_DIRECT_LOAD V4` at
  `rdna4/README.md:5888`; the implicit-use tracking gap for M0 is recorded in
  `RDNA4-RJ-165`.
- RDNA4 Chapter 12.4 test note: no dedicated C++ or Python execution tests were
  found for `ds_direct_load`, `vdsdir`, or `lds_direct`; the generic execution
  harness skips non-scalar instructions at
  `tests/instruction_execution_harness_test.cpp:140` and `:246`.
- RDNA4 Chapter 12.5 decode/build narrow match: generated VDS decode and
  builders preserve the expected raw fields, including `offset0`, `offset1`,
  `addr`, `data0`, `data1`, and `vdst`, at `rdna4/machine_insts.h:122`
  through `:132`, `rdna4/builders.h:328` through `:340`, and representative
  decode fixtures at `rdna4/test_encodings.h:1097`.
- RDNA4 Chapter 12.5 single-address narrow match: ordinary one-address DS
  load/store/atomic address calculation combines `{offset1,offset0}` and adds
  the per-lane VGPR address plus `LDS_BASE`, matching the manual formula, at
  `rdna4/README.md:5951` through `:5955` and
  `shared/addr_calc_scalar.h:60` through `:76`.
- RDNA4 Chapter 12.5 two-address narrow match: ordinary
  `LOAD/STORE_2ADDR_{B32,B64}` and stride64 load/store paths set
  `ds2_active`, compute the documented `offset*4`, `offset*8`, `offset*256`,
  or `offset*512` forms, and the local pipeline has second-load/store handling
  at `rdna4/vds.cpp:507`, `:1539`, `:2263`, and
  `vm/amdgpu/memory_pipeline.cpp:524`. The store-exchange exception is recorded
  in `RDNA4-RJ-168`.
- RDNA4 Chapter 12.5 narrow-data narrow match: signed loads set
  `sign_extend`, D16 variants set `d16_lo` or `d16_hi`, and memory completion
  applies sign extension and half-register merge behavior at
  `rdna4/vds.cpp:1610`, `rdna4/vds.cpp:3587`, and
  `vm/amdgpu/memory_pipeline.cpp:175`.
- RDNA4 Chapter 12.5 DScnt narrow match: DS executions set
  `WaitCounterType::DSCNT`, the memory pipeline issues the state-selected
  counter, RDNA4 wait instructions set DS wait targets, and unit tests cover
  counter increment/decrement behavior at `rdna4/vds.cpp:45`,
  `vm/amdgpu/memory_pipeline.h:62`, `rdna4/sopp.cpp:572`, and
  `tests/waitcnt_counter_test.cpp:65`.
- RDNA4 Chapter 12.5 test note: searches found no targeted execution tests for
  `ds_load_addtid`, `ds_store_addtid`, `storexchg_2addr`, `ds_pk_add`, or
  `ds_add_f32`; the explicit local coverage found in this slice is primarily
  generated encoding fixtures and wait-counter unit tests.
- RDNA4 Chapter 12.5.2 wrapper narrow match: generated RDNA4 wrappers exist for
  all scoped lane-permute opcodes and route to shared helpers at
  `rdna4/vds.cpp:1490` through `:1502`, `:3931` through `:3945`, `:3948`
  through `:3962`, and `:3965` through `:3979`.
- RDNA4 Chapter 12.5.2 no-storage narrow match: the lane-permute helpers
  operate through VGPR read/write regions and do not touch LDS storage, matching
  the manual's no-LDS-memory-storage statement at `rdna4/README.md:6021`;
  representative helper paths are at `shared/execute_shared.h:93` through
  `:123`, `:127` through `:156`, `:230` through `:261`, and `:299` through
  `:331`.
- RDNA4 Chapter 12.5.2 BPERMUTE narrow match: `DS_BPERMUTE_B32` value behavior
  initializes temporary results to zero and only reads selected source data when
  the selected source lane is active, matching disabled-source-zero behavior,
  at `shared/execute_shared.h:112` through `:123`; `DS_BPERMUTE_FI_B32`
  fetches selected source data regardless of source EXEC and writes only active
  destinations at `shared/execute_shared.h:146` through `:156`.
- RDNA4 Chapter 12.5.2 PERMUTE narrow match: `DS_PERMUTE_B32` loops over active
  source lanes in ascending lane order, writes into a zero-initialized
  temporary array, and then writes active destination lanes at
  `shared/execute_shared.h:249` through `:261`. This matches the manual figure's
  unused-destination zeroing and highest-numbered-source collision behavior for
  forward scatter.
- RDNA4 Chapter 12.5.2 byte-index narrow match: the PERMUTE/BPERMUTE/FI helpers
  add the encoded offset, divide by four, and modulo the lane group at
  `shared/execute_shared.h:116`, `:150`, and `:255`, matching byte-index and
  wrap behavior from `rdna4/README.md:6027` through `:6029`.
- RDNA4 Chapter 12.5.2 Wave64 boundary note: RDNA4 lane-permute wave64 behavior
  is documented by the manual at `rdna4/README.md:6025`, but broad RDNA4
  wave64 dispatch is already tracked in `RDNA4-RJ-016`; this slice did not add
  a duplicate lane-permute-specific wave64 dispatch gap.
- RDNA4 Chapter 12.5.2 test note: existing execution coverage found for
  swizzle uses full EXEC and broadcast/QDMode cases at
  `tests/instruction_execution_harness_test.cpp:3384` through `:3418`; no
  targeted execution tests were found for DS_PERMUTE/BPERMUTE/FI EXEC masks,
  wave64 behavior, DS_SWIZZLE FFT, or DS_SWIZZLE rotate.
- RDNA4 Chapter 12.5.3 decoder narrow match: generated RDNA4 opcode constants
  and decoder-table entries exist for `DS_BVH_STACK_PUSH4_POP1_RTN_B32`,
  `DS_BVH_STACK_PUSH8_POP1_RTN_B32`, and
  `DS_BVH_STACK_PUSH8_POP2_RTN_B64` at `rdna4/opcodes.h:1216` through
  `:1218` and `rdna4/decoder.cpp:9522` through `:9526`.
- RDNA4 Chapter 12.5.3 operand narrow match: generated constructors model
  `VDST` as B32/B32/B64, `ADDR` as a B32 input and output operand, `DATA0` as
  B32, and `DATA1` as B128/B256/B256 at `rdna4/vds.cpp:4056` through
  `:4112`, matching the XML operand shapes at `amdgpu_isa_rdna4.xml:49709`
  through `:49828`.
- RDNA4 Chapter 12.5.3 raw-field narrow match: the VDS machine instruction and
  builder preserve `offset0`, `offset1`, `addr`, `data0`, `data1`, and `vdst`
  fields at `rdna4/machine_insts.h:122` through `:132` and
  `rdna4/builders.h:319` through `:338`, so the manual's stack size and
  traversal flags are available to a future stack semantic implementation.
- RDNA4 Chapter 12.5.3 test note: searches found generated encoding fixtures
  for the three RDNA4 stack opcodes at `rdna4/test_encodings.h:1215` through
  `:1217`, but no targeted execution tests because the execute bodies are
  intentionally unimplemented today.
- RDNA3/RDNA3.5 boundary note: older RDNA stack handling uses the single
  `DS_BVH_STACK_RTN_B32` opcode with different `ENC_DS` offset semantics;
  local generated wrappers also throw `UnimplementedInst` at
  `rdna3/ds.cpp:4097` through `:4116` and
  `rdna3_5/ds.cpp:4097` through `:4116`. This is a related cross-family gap,
  but it is separate from the RDNA4 Chapter 12.5.3 push/pop instruction slice.
- RDNA4 Chapter 11.1 narrow match: generated machine-instruction structs
  preserve the raw VFLAT, VSCRATCH, and VGLOBAL field shapes, including
  `SADDR`, the XML-only `NV` field from `RDNA4-XML-148`, `OP`, `SVE`, `SCOPE`,
  `TH`, `VSRC`, `VDST`, `VADDR`, and `IOFFSET`, at `rdna4/machine_insts.h:217`
  through `:268`; the metadata and semantic gaps are recorded in
  `RDNA4-RJ-139` through `RDNA4-RJ-144`.
- RDNA4 Chapter 11.1 narrow match: flat, global, and scratch instruction
  families are not marked decode-only; the RDNA4 coverage exception file says
  they are tested through kernel simulation at `coverage_exceptions_rdna4.txt:21`
  through `:23`, and zero-encoding fixtures enumerate representative audited
  forms at `rdna4/test_encodings.h:1426`, `:1459`, `:1501` through `:1502`,
  and `:1543`.
- RDNA4 Chapter 11.1 narrow match: `GLOBAL_LOAD_ADDTID_B32` and
  `GLOBAL_STORE_ADDTID_B32` omit a `VADDR` operand and compute
  `base + ioffset + lane*4` at `rdna4/vglobal.cpp:716` through `:753` and
  `:757` through `:804`, matching the manual's no-VGPR-addressing note.
- RDNA4 Chapter 11.1 narrow match: scratch-base state is stored on each
  wavefront and initialized by the command processor for dispatch scratch
  allocations at `vm/amdgpu/wavefront.h:253` through `:265` and
  `vm/amdgpu/command_processor.cpp:342` through `:348`.
- RDNA4 Chapter 11.2 narrow match: ordinary Global `GV` and `GVS` runtime
  address calculation uses a 64-bit VGPR address when `SADDR` is null and a
  64-bit SGPR base plus one 32-bit VGPR offset when `SADDR` is present, at
  `rdna4/addr_calc.cpp:103` through `:129`.
- RDNA4 Chapter 11.2 narrow match: RDNA4 ADDTID global load/store bodies
  compute `base + IOFFSET + lane*4` at `rdna4/vglobal.cpp:728` through `:752`
  and `:780` through `:804`, matching the `GT` formula for the ordinary
  functional address path.
- RDNA4 Chapter 11.2 narrow match: the Scratch helper uses `SVE` and `SADDR`
  nullness to include or omit VGPR and SGPR offset sources at
  `rdna4/addr_calc.cpp:142` through `:151`, and the positive-source-selection
  cases are covered at `tests/shared_infra_test.cpp:3114` through `:3143`.
- Cross-architecture note: gfx1250 currently uses the same private lane-slice
  style for scratch addressing at `gfx1250/addr_calc.cpp:184` through `:209`.
  This mirrors the RDNA4 scratch swizzle/validity concern in `RDNA4-RJ-145`,
  but the gfx1250 manual contract was not audited in this RDNA4 Chapter 11.2
  slice.
- Chapter 11.3 narrow match: for address calculators that remove lanes from
  `VectorMemState::lane_mask`, vector load completion writes zero to those
  destination lanes at `vm/amdgpu/memory_pipeline.cpp:139` through `:169`.
  This supports existing buffer OOB behavior, but Chapter 11.3's
  Flat/Global/Scratch error causes and `MEMVIOL`/trap side effects are recorded
  in `RDNA4-RJ-148`.
- Chapter 11.3 test note: existing RDNA4/global/scratch address tests are
  positive helper checks at `tests/shared_infra_test.cpp:3061` through `:3143`,
  and the broad generated instruction execution harness skips memory
  instructions at `tests/instruction_execution_harness_test.cpp:135` through
  `:139`; no focused Chapter 11.3 fault tests were found in this slice.
- Chapter 11.4 execution narrow match: ordinary B32/B64/B96/B128
  Flat/Global/Scratch data payloads use `elem_size`/`num_elems` to cover one
  through four consecutive DWORDs, and store bodies read consecutive VGPRs such
  as `data_base + 0` through `data_base + 3` in the 128-bit paths.
- Chapter 11.4 execution narrow match: D16/D16_HI loads and stores are plumbed
  through `VectorMemState::d16_lo` / `d16_hi`; `vector_complete` merges loaded
  low/high halves with the old VGPR value at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:181` through `:193`,
  and D16_HI stores shift the selected source half before writing memory, for
  example at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/vflat.cpp:614`
  through `:670`.
- Chapter 11.4 race-detector narrow match: D16 LDS and global load events use
  byte masks derived from `d16_lo` / `d16_hi`, so the plugin records low-half
  accesses as `0x3` and high-half accesses as `0xC` at
  `lib/rocjitsu/src/rocjitsu/vm/plugins/race_detector/plugin.cpp:284` through
  `:293` and `:314` through `:321`.
- Chapter 11.5 decode narrow match: generated RDNA4 decoder inventory includes
  all four block opcodes through `decodeScratchLoadBlockVscratch` and
  `decodeGlobalLoadBlockVglobal`, with table entries at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/decoder.cpp:5920` through
  `:5925`, `:6167` through `:6172`, `:10166` through `:10167`, and `:10427`
  through `:10428`.
- Chapter 11.5 operand-width narrow match: the generated block load/store
  classes expose the maximum data span as 1024-bit `VDST` or `VSRC` operands
  at `rdna4/vglobal.cpp:2070` through `:2082`, `:2097` through `:2109`,
  `rdna4/vscratch.cpp:716` through `:728`, and `:743` through `:755`; the
  missing sparse active-DWORD behavior is recorded in `RDNA4-RJ-153`.
- Chapter 11.5 counter narrow match: block load/store currently uses one
  `LOADCNT` or `STORECNT` producer per issued instruction by setting
  `d->wait_counter_type` once in the generated execute body at
  `rdna4/vglobal.cpp:2089`, `:2116`, `rdna4/vscratch.cpp:735`, and `:762`.
  The counter classification itself relies on the existing generic vector
  memory pipeline.
- Chapter 11.5 test note: the focused codegen test asserts only that the block
  instructions derive `flat_load`/`flat_store` semantics with `elem_size=4` and
  `num_elems=32` at `lib/python/amdisa/tests/test_sema_derive.py:1751` through
  `:1758`. This confirms full-mask sizing, but no execution tests were found in
  this slice for sparse `M0`, `M0==0`, block VGPR OOR handling, or illegal
  scratch `ST` block encodings.
- Chapter 11.6 decode narrow match: generated RDNA4 decoder support exists for
  both transpose global loads at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/decoder.cpp:6179` through
  `:6185`, with VGLOBAL dispatch table entries at `:10431` through `:10432`.
- Chapter 11.6 codegen narrow match: the semantics layer derives
  `GLOBAL_LOAD_TR_B64` and `GLOBAL_LOAD_TR_B128` as `flat_load` operations with
  transpose kinds `TR_B8` and `TR_B16`, respectively, at
  `lib/python/amdisa/semantics.py:1786` through `:1806` and `:1999` through
  `:2012`; the generator emits `d->transpose` for transpose loads at
  `lib/python/amdisa/codegen/_generator.py:4620` through `:4643`.
- Chapter 11.6 execution narrow match: generated RDNA4 runtime marks both
  transpose loads as `LOADCNT` global loads and assigns `d->transpose = 4` for
  `GLOBAL_LOAD_TR_B128` and `d->transpose = 3` for `GLOBAL_LOAD_TR_B64` at
  `rdna4/vglobal.cpp:2249` through `:2257` and `:2277` through `:2285`; the
  global memory pipeline applies `transpose_response` before vector writeback
  at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:498` through
  `:503`.
- Chapter 11.6 encoding/test note: zero-encoding fixtures include
  `global_load_tr_b128` and `global_load_tr_b64` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/test_encodings.h:1539`
  through `:1540`, but the generic instruction execution harness skips all
  non-scalar instructions at `emulation/rocjitsu/tests/instruction_execution_harness_test.cpp:141`
  through `:149`. Searches found codegen derivation tests for the mnemonic
  shapes at `lib/python/amdisa/tests/test_sema_derive.py:1673` through `:1690`,
  but no focused RDNA4 global transpose-load execution tests for wave32 mapping,
  wave64 lane 0-31 address sourcing, reduced wave64 write width, zero-`EXEC`
  `S_NOP`, or partial-`EXEC` undefined handling.
- RDNA4 Chapter 10.1 narrow match: generated machine-instruction structs
  preserve the raw VIMAGE and VSAMPLE field shapes, including `dim`, `r128`,
  `d16`, `a16`, the XML-only `nv` field from `RDNA4-XML-148`, `op`, `dmask`,
  `vdata`, `rsrc`, `scope`, `th`, `tfe`, `lwe`, `unorm`, `samp`, and `vaddr*`
  at `rdna4/machine_insts.h:154` through `:200`; the missing pieces are operand
  exposure, disassembly, descriptor interpretation, sampler interpretation, and
  execution semantics recorded in `RDNA4-RJ-125` through `RDNA4-RJ-138`.
- RDNA4 Chapter 10.1-10.3 narrow match: generated RDNA4 classes and decoder
  inventory exist for the audited image/sample families, including loads,
  stores, atomics, `IMAGE_GET_RESINFO`, `IMAGE_MSAA_LOAD`, `IMAGE_SAMPLE*`,
  gather forms, and BVH forms. Representative decoder tables are at
  `rdna4/decoder.cpp:8008` through `:8025` and `:9585` through `:9610`, and
  zero-encoding fixtures enumerate the families at `rdna4/test_encodings.h:1309`
  through `:1342`. The current coverage exception records that the image
  simulation itself is intentionally absent.
- RDNA4 Chapter 10.6 narrow match: generated VSAMPLE code preserves the raw
  `SAMP` selector and exposes ordinary sample/gather/get-LOD sampler operands
  as 128-bit SGPR operands; `IMAGE_MSAA_LOAD` omits `samp`, matching the manual
  statement that it uses VSAMPLE but not a sampler.
- RDNA4 Chapter 10.7 narrow match: generated VBUFFER code preserves the raw
  instruction-side typed-buffer `format` field and exposes it through the test
  builder at `rdna4/machine_insts.h:134` through `:152` and
  `rdna4/builders.h:343` through `:377`; the missing enum mapping and
  conversion semantics are recorded in `RDNA4-RJ-131`.
- RDNA4 Chapter 10.8 narrow match: generated RDNA4 split wait instructions
  decode and execute `S_WAIT_LOADCNT`, `S_WAIT_STORECNT`, `S_WAIT_SAMPLECNT`,
  `S_WAIT_BVHCNT`, `S_WAIT_EXPCNT`, `S_WAIT_DSCNT`, `S_WAIT_KMCNT`,
  `S_WAIT_LOADCNT_DSCNT`, and `S_WAIT_STORECNT_DSCNT`; representative execute
  paths set wait targets at `rdna4/sopp.cpp:497` through `:620`.
- RDNA4 Chapter 10.8 narrow match: implemented RDNA4 buffer/global/flat/scratch
  paths use split `LOADCNT` and `STORECNT` wait-counter types for many load and
  store instructions, and the memory pipeline increments/releases the selected
  counter at `vm/amdgpu/memory_pipeline.h:62` through `:88`.
- RDNA4 Chapter 10.9 narrow match: generated RDNA4 decode tables contain the
  four BVH ray-tracing VIMAGE opcodes, with `IMAGE_BVH_INTERSECT_RAY` and
  `IMAGE_BVH64_INTERSECT_RAY` in the low opcode table at
  `rdna4/decoder.cpp:8035` through `:8036` and
  `IMAGE_BVH_DUAL_INTERSECT_RAY` plus `IMAGE_BVH8_INTERSECT_RAY` at
  `:8138` through `:8139`.
- RDNA4 Chapter 10.9 narrow match: generated BVH constructors preserve the
  manual's destination/resource operand footprint at the coarse level:
  128-bit VDATA and 128-bit RSRC for single-node BVH/BVH64, and 320-bit VDATA
  and 128-bit RSRC for dual/BVH8, at `rdna4/vimage.cpp:402` through `:457`.
- RDNA4 Chapter 10.10 narrow match: generated machine-instruction structs
  preserve the raw PRT-related selector fields (`dmask`, `tfe`, `lwe`, and
  `rsrc`) for VIMAGE/VSAMPLE at `rdna4/machine_insts.h:154` through `:196`,
  and `coverage_exceptions_rdna4.txt:27` explicitly records image instructions
  as decode-only. The missing resource-enabled PRT behavior, extra status VGPR,
  and destination-range/nullification rules are recorded in `RDNA4-RJ-138`.
- RDNA4 Chapter 9.1 narrow match: generated machine-instruction state captures
  the VBUFFER instruction-word field shape, including `soffset`, the XML-only
  `nv` field from `RDNA4-XML-148`, `op`, `tfe`, `vdata`, `rsrc`, `scope`, `th`,
  `format`, `offen`, `idxen`, `vaddr`, and `ioffset` at
  `rdna4/machine_insts.h:134` through `:152`.
- RDNA4 Chapter 9.2 narrow match: generated VBUFFER constructors compute the
  exposed `VADDR` operand size from `IDXEN`/`OFFEN` through
  `vbuffer_vaddr_bits`, so disassembly/operand metadata distinguishes zero,
  one, and two address VGPR forms even though runtime address calculation is
  incomplete.
- RDNA4 Chapter 9.4 narrow match: RDNA4 tests cover the first-SGPR `RSRC`
  selector, optional `SOFFSET` including `M0`, `OFFEN`, and offset-part wrapping
  at `tests/shared_infra_test.cpp:3240` through `:3335`; descriptor bounds and
  index handling are recorded in `RDNA4-RJ-119`.
- RDNA4 Chapter 9.2/9.3.1 narrow match: plain D16 and D16_HI unformatted buffer
  loads/stores use `d16_lo`/`d16_hi` state and high-half store extraction, for
  example D16 loads at `rdna4/vbuffer.cpp:877` through `:1045` and D16_HI stores
  at `rdna4/vbuffer.cpp:1065` through `:1126`; formatted D16 conversion remains
  in `RDNA4-RJ-121`.
- RDNA4 Chapter 8.1 SMEM decode inventory exists for the audited load and
  invalidation opcodes: generated classes cover `S_LOAD_B32/B64/B96/B128/B256/B512`,
  `S_LOAD_I8/U8/I16/U16`, `S_BUFFER_LOAD_B32/B64/B96/B128/B256/B512`,
  `S_BUFFER_LOAD_I8/U8/I16/U16`, and `S_DCACHE_INV` in `rdna4/smem.cpp`.
- RDNA4 Chapter 8.1.3 narrow-load result extension is modeled after address
  calculation: generated I8/I16 forms set `sign_extend = true`, U8/U16 forms
  set `sign_extend = false`, and the scalar memory pipeline uses
  `load_bytes()` plus `extend_scalar_load()` for `elem_size < 4` at
  `vm/amdgpu/memory_pipeline.cpp:204` through `:212`.
- RDNA4 Chapter 8.1.4 `S_DCACHE_INV` is implemented as an operandless scalar
  cache invalidation: the generated class has no sources or destinations and
  calls `wf.cu().l1_scalar().invalidate_all()` at `rdna4/smem.cpp:564`
  through `:571`; the `KMcnt` dependency-counter side is recorded in
  `RDNA4-RJ-023`.
- RDNA4 Chapter 8.4 SBASE evenness is represented structurally: generated SMEM
  constructors publish `sbase` as `inst->sbase * 2`, for example
  `S_BUFFER_LOAD_B32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/smem.cpp:294` through
  `:305`, and runtime address calculation uses the same `inst.sbase * 2`
  mapping at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.cpp:56`
  through `:62`; range-check consequences are recorded in `RDNA4-RJ-117`.
- RDNA4 Chapter 8.5 prefetch decode coverage exists for all five scalar prefetch
  forms: generated constructors publish the expected base/direct-offset,
  length-offset, and immediate-length operands for `S_PREFETCH_INST`,
  `S_PREFETCH_INST_PC_REL`, `S_PREFETCH_DATA`, `S_BUFFER_PREFETCH_DATA`, and
  `S_PREFETCH_DATA_PC_REL` in
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna4/smem.cpp:603` through `:682`.
- RDNA4 Chapter 8.5 no-`KMcnt` behavior is structurally satisfied for scalar
  prefetches because their constructors do not set `MEMORY_OP` and their
  execute bodies do not enter `MemoryPipeline::issue`; the missing cache and
  mode behavior is recorded in `RDNA4-RJ-118`.
- RDNA4 SOPK instruction sizing and execution special-case the manual's
  `S_SETREG_IMM32_B32` literal exception: opcode 19 is 64-bit and execution
  reads `literal_`; the gap above is specifically in generated
  source-operand/disassembly metadata.
- RDNA4 SALU signed-overflow execution matches Chapter 6.3/6.4 despite the XML
  description gap: `S_ADD_CO_I32` and `S_SUB_CO_I32` call
  `signed_add_overflows` / `signed_sub_overflows` at
  `shared/execute_shared.h:385` through `:390` and `:2579` through `:2584`,
  and SOPK `S_ADDK_CO_I32` uses the same signed-overflow helper at
  `rdna4/sopk.cpp:79`.
- RDNA4 SALU unsigned extended arithmetic execution reads SCC as carry/borrow
  input and writes carry-out for the representative add/sub carry forms at
  `shared/execute_shared.h:372` through `:379` and `:2566` through `:2574`.
- RDNA4 Chapter 6.5 conditional moves execute as SCC consumers without SCC
  writes in the checked bodies: `S_CMOV*` at `shared/execute_shared.h:1081`
  through `:1101` and `S_CSELECT*` at `:1511` through `:1524`.
- RDNA4 Chapter 6.6/6.7 representative compare and bit operations execute the
  expected SCC rules: bit compares mask the bit index and write SCC at
  `shared/execute_shared.h:897` through `:925`, and the shift-add helpers
  shift source 0, add source 1, and write unsigned overflow at
  `shared/execute_shared.h:1784` through `:1828`.
- RDNA4 `S_CVT_HI_F32_F16` register sources are implemented with the expected
  high-half shift at `shared/execute_shared.h:1583` through `:1589`, and the
  existing scalar conversion SCC test covers a register value with F16 `1.0` in
  bits 31:16 at `tests/scalar_scc_test.cpp:950`; the gap above is limited to
  the literal-source path.
- RDNA4 aperture source selectors 235 through 238 resolve from wavefront
  aperture state in both 32-bit and 64-bit scalar reads at
  `rdna4/operand.cpp:797` through `:803` and `rdna4/operand.cpp:906` through
  `:912`, and `ComputeUnit` copies configured aperture bounds into the
  wavefront at `vm/amdgpu/compute_unit.cpp:146` through `:147`.
- RDNA4 `S_FMAC_F16` correctly reports an old-destination use because the XML
  and execute body model it as an accumulator at
  `amdgpu_isa_rdna4.xml:31216` through `:31234` and
  `shared/execute_shared.h:1747` through `:1753`.
- RDNA4 Chapter 6.7 count/scan corner cases match the manual prose in the
  checked bodies: `S_CLS*`, `S_CLZ*`, and `S_CTZ*` return `-1` for the
  documented no-match cases at `shared/execute_shared.h:1035` through `:1078`
  and `:1527` through `:1545`.
- RDNA4 `SOPP` and `SOP1` 32-bit field shapes match the manual's Chapter 5.1
  format prose through the generated machine-instruction structs.
- RDNA4 `S_ENDPGM` functionally ends the wave through `wf.end()` at
  `rdna4/sopp.cpp:362` through `:369`, and `Wavefront::end()` drains pending
  waits before halting at `vm/amdgpu/wavefront.h:446` through `:453`.
- RDNA4 `S_VERSION` is a no-op via `execute_s_version_sopk` at
  `shared/execute_shared.h:2672`, matching the manual's no-op/version-comment
  intent.
- RDNA4 `S_SENDMSG_RTN_B32/B64` construct ordinary `SSRC0` values as
  `OPR_SENDMSG_RTN` enum operands rather than SGPR operands at
  `rdna4/sop1.cpp:1241` through `:1277`; the `0xff` literal-sentinel collision
  is recorded above.
- RDNA4 has basic `S_WAIT_KMCNT` threshold plumbing and generic `KMCNT` counter
  storage; field masking and producer-completion gaps are recorded in
  `RDNA4-RJ-023`, `RDNA4-RJ-025`, `RDNA4-RJ-049`, and `RDNA4-RJ-115`.
- RDNA4 `RTN_GET_REALTIME` has a partial functional value source through
  `engine->global_time()` for B32/B64, even though it does not model the
  `KMcnt` return protocol.
- RDNA4 Chapter 5.5 branch opcode coverage exists for `S_BRANCH`, all six
  `S_CBRANCH_*` forms, `S_GETPC_B64`, `S_SETPC_B64`, `S_SWAPPC_B64`, and
  `S_CALL_B64` in generated opcode/decoder tables.
- RDNA4 branch execute paths sign-extend `SIMM16`, scale it by four bytes, and
  apply the manual next-PC base for `S_BRANCH`, `S_CBRANCH_*`, and `S_CALL_B64`
  through `branch_offset_bytes()` and the runtime `wf.pc + 4 + offset * 4`
  updates.
- RDNA4 conditional branch execute paths match the Chapter 5.5 condition table:
  SCC branches read `wf.read_scc()`, VCC branches test the live-lane VCC mask,
  and EXEC branches test `wf.exec()` zero/nonzero.
- RDNA4 `S_GETPC_B64`, `S_SWAPPC_B64`, and `S_CALL_B64` write the
  next-instruction PC as the manual describes; `S_SETPC_B64` and
  `S_SWAPPC_B64` generated execute bodies set `PC` from the source operand.
- RDNA4 Chapter 5.6 split-barrier decode coverage exists for
  `S_BARRIER_SIGNAL`, `S_BARRIER_SIGNAL_ISFIRST`, and `S_BARRIER_WAIT` in
  generated opcode/decoder tables; `S_BARRIER_WAIT` is decoded as SOPP opcode
  `20` with a `SIMM16` source and therefore has no `M0` operand path.
- RDNA4 `OPR_SSRC_BARRIER_ID` structurally includes `m0` and integer inline
  selectors for the signal instructions, and the current CU model releases a
  simple barrier wait only after all live sibling waves in the same
  dispatch/work-group have parked in the generic barrier state.
- RDNA4 `S_NOP` has no functional state effect via `execute_s_nop_sopp` at
  `shared/execute_shared.h:2138`; exact repeat/timing behavior is outside the
  current functional model.
- RDNA4 generates decode/operand plumbing for `S_CLAUSE`, `S_DELAY_ALU`, and
  `S_WAIT_ALU` at `rdna4/sopp.cpp:64` through `:100`; the gaps above are in
  validation, clause stream state, and delay/wait execution.
- RDNA4 split wait opcode constants, decoder table entries, and generated
  classes exist for the detailed Chapter 5.7.1 wait opcodes, including
  `S_WAIT_LOADCNT`, `S_WAIT_STORECNT`, `S_WAIT_SAMPLECNT`, `S_WAIT_BVHCNT`,
  `S_WAIT_EXPCNT`, `S_WAIT_DSCNT`, `S_WAIT_KMCNT`,
  `S_WAIT_LOADCNT_DSCNT`, and `S_WAIT_STORECNT_DSCNT`.
- Rocjitsu stores wait counters per wave, stalls waves in `WAITCNT` while
  thresholds are unsatisfied, and wakes them when counters reach target through
  `Wavefront::release_wait_counter` and `ComputeUnitCore::update_wf_states`;
  the gaps above are in exact producer classes, field masking, and detailed
  ordering behavior.
- Ordinary generated producers cover common SMEM, vector load/store, and LDS
  cases by mapping SMEM to `KMCNT`, vector memory loads/stores to
  `LOADCNT`/`STORECNT`, and DS/LDS to `DSCNT`; narrower counter and ordering
  gaps are recorded above.
- RDNA4 Chapter 8.2 narrow match: ordinary SMEM load bodies set
  `wait_counter_type = KMCNT` for the audited load forms before entering the
  generic memory pipeline; the width/counting and timing limitations are
  recorded in `RDNA4-RJ-049` and `RDNA4-RJ-115`.
- `DScnt` and `KMcnt` also feed the legacy `LGKMcnt` alias in
  `WaitCounters`, preserving compatibility with broader waitcnt-style code
  paths while still keeping separate RDNA3+ target fields.
- RDNA4 structurally generates several instruction families needed for clause
  typing, including SMEM, flat, DS/LDS, image, sample, and export instructions;
  this does not provide the missing clause membership table or runtime state.
- RDNA4 `V_S_*` pseudo-scalar transcendental instructions are generated as
  vector/VALU-style instructions, for example `v_s_exp_f32` appears in
  `rdna4/test_encodings.h:809`; the gap is not their basic classification as
  VALU-like operations.
- `S_ROUND_MODE` and `S_DENORM_MODE` were already recorded as
  `RDNA4-RJ-013`; this Chapter 5 slice reconfirmed that existing gap rather
  than adding a duplicate entry.
- RDNA4 Chapter 7.2 reconfirmed `RDNA4-RJ-013`: `S_ROUND_MODE` and
  `S_DENORM_MODE` still dispatch to empty helpers at
  `shared/execute_shared.h:1644` through `:1645` and `:2423` through `:2425`.
- RDNA4 generated `V_CMP*`/`V_CMPX*` paths initialize their result masks from
  zero and write only lanes whose comparisons are true, matching the Chapter
  7.2 inactive-lane-zero rule for ordinary compare masks; DPP/CMPX-specific
  behavior remains covered by `RDNA4-RJ-074`.
- RDNA4 VOP3P DPP decode did not prove to be a live DOT4/DOT8/WMMA legality
  bug in this slice: the checked XML exposes VOP3P DPP encoding alternatives
  only for `V_DOT2_F32_F16`, `V_DOT2_F32_BF16`, and `V_FMA_MIX*`, not for the
  audited DOT4/DOT8/WMMA opcodes.
- RDNA4 Chapter 7.7 packed opcode/decode inventory is present for the audited
  non-DOT packed VOP3P names: the manual lists the PK16/F16/MIX opcodes, and
  rocjitsu defines matching opcodes and decoder entries; `V_PK_FMAC_F16` is
  correctly a VOP2 form rather than a missing VOP3P class.
- RDNA4 generated VOP3P machine-instruction state exposes the expected packed
  field shape: `neg_hi`, split `opsel_hi`, `opsel`, `clamp`, and `neg`, with
  no ABS or OMOD field in the VOP3P format.
- RDNA4 ordinary packed I16/U16/F16 scalar execution implements basic
  `OPSEL`/`OPSEL_HI` half selection for sources; non-default selector handling
  falls back out of SIMD helpers where needed.
- RDNA4 MIX source selection, inline-constant F16 conversion when a MIX source
  is selected as F16, and `NEG_HI` as ABS are implemented in the scalar MIX
  bodies; the remaining MIX execution gap in this slice is the non-fused
  arithmetic recorded above.
- RDNA4 VOPD destination parity is reconstructed from `VDSTY[7:1]` and
  `~VDSTX[0]` in the constructor at `rdna4/vopd.cpp:230` through `:232`,
  matching the microcode field shape.
- RDNA4 VOPD source classes match the basic field roles: `SRC0` uses generic
  `OPR_SRC`, while `VSRC1` uses `OPR_VGPR`, at `rdna4/vopd.cpp:245` through
  `:248`.
- RDNA4 VOPD literal sizing is broader than the XML condition: rocjitsu treats
  either `SRCX0 == 255`, `SRCY0 == 255`, or any FMAAK/FMAMK slot as a
  96-bit-literal instruction at `rdna4/vopd.cpp:233` through `:238`.
- RDNA4 VOPD computes both slot results before either destination write at
  `rdna4/vopd.cpp:350` through `:354`, matching the allowed
  Y-overwrites-X-source case for ordinary functional execution.
- RDNA4 DPP marker constants and the DPP8FI distinction exist in shared helper
  code as `SRC_DPP == 250`, `SRC_DPP8_FI_0 == 233`, and
  `SRC_DPP8_FI_1 == 234`.
- RDNA4 machine-instruction structs model DPP16 and DPP8 suffix fields across
  VOP1, VOP2, VOPC, VOP3, VOP3P, and VOP3_SDST formats; the gaps above are in
  sizing, validation, modifier application, disassembly, and DBT handling.
- Core 32-bit DPP helper behavior exists for quad permute, row shift, row
  rotate, row share, row xmask, row/bank masks, `BC`, and `FI`, and existing
  helper/runtime tests cover representative FI and write-mask cases.
- RDNA4 pseudo-scalar `V_S_*` instructions are present in generated VOP3 code
  and use scalar operand classes for their ordinary source/destination slots;
  the gaps above are in special `EXEC`, destination-legality, and selector
  behavior.
- RDNA4 generated code implements the basic in-range `V_MOVRELD_B32` and
  `V_MOVRELS_B32` move paths for VOP1 and VOP3 encodings; the gaps above are
  in signed `M0`, out-of-range behavior, and the remaining indexed forms.
- RDNA4 Chapter 7.1 basic format shapes are present in generated
  `MachineInst` structs for VOP1, VOP2, VOPC, VOP3, VOP3P, and VOP3SD; the
  gaps above are in legality, disassembly, and corner semantics.
- RDNA4 generated opcodes respect the manual's non-promotable VOP1/VOP2 list:
  SWAP/SWAPREL/PERMLANE64_B32 are generated as VOP1 forms, and
  FMAMK/FMAAK/PK_FMAC as VOP2 forms, with no matching generated VOP3 classes
  found in this slice.
- Representative VOP2 `ACCUM` decode/disassembly plumbing captures the
  destination as an implied source: `V_PK_FMAC_F16` makes `vdst` both a source
  and destination at `rdna4/vop2.cpp:6464` through `:6475`.
- RDNA4 Chapter 7.3 explicit non-VOP3P VALU inventory is generated: every
  explicit mnemonic in the table has a generated RDNA4 mnemonic after
  normalizing `_e32` suffixes, and the generic compare rows expand to concrete
  VOPC/VOP3 compare instruction names.
- RDNA4 Chapter 7.3 compare opcode expansion follows the detailed Chapter 15
  opcode tables: generated names use `NE` for integer not-equal predicates and
  have no concrete `F` or `T` compare opcodes, matching XML and Chapter 15
  rather than the compact summary wording in 7.3.
- RDNA4 Chapter 7.4 compact true16 VOP1/VOP2/VOPC operands are modeled by the
  generated operand layer: values 0-127 select `vN.l`, values 128-255 select
  `vN.h`, destination half writes merge with the previous dword, and
  representative VOP1/VOP2/VOPC constructors pass the packed-16 source flag at
  `rdna4/operand.cpp:34` through `:58`, `rdna4/vop1.cpp:3232` through
  `:3237`, `rdna4/vop2.cpp:4839` through `:4846`, and
  `rdna4/vopc.cpp:23` through `:29`.
- RDNA4 Chapter 7.4 ordinary VOP3 true16 source OPSEL behavior is implemented
  through shared helpers: `read_vop3_true16_src` applies `OPSEL[0:2]`, and
  representative `v_add_f16` code uses it at `shared/simd_glue.h:270` through
  `:282` and `rdna4/vop3.cpp:9167` through `:9177`. Destination half merging
  through `OPSEL[3]` is implemented for the register reference chosen by
  `write_vop3_true16_dst`, but the raw `VDST=128..255` address collapse is
  tracked by `RDNA4-RJ-097`.
- RDNA4 VOP3SD inventory matches the manual's limited use: add/sub carry,
  div-scale, and mad-carry classes decode as `Vop3SdstEnc`, while `V_CMP*`
  classes decode as VOP3.
- The single 32-bit literal model for VOP3-family encodings is consistent with
  Chapter 7.1's 96-bit VOP3-plus-literal form; the manual's performance caveat
  for excessive VOP3 literal use is outside this functional decoder audit.
- Dense RDNA4 WMMA FP8/BF8 generated bodies select the expected FP8/BF8
  extraction helpers per mnemonic, and existing SIMD exact tests cover those
  helper combinations in `tests/simd_correctness/wmma_simd_exact_test.cpp:145`
  through `:158`. The remaining WMMA/SWMMAC layout, hazard, and
  illegal-modifier issues are recorded in `RDNA4-RJ-056` through
  `RDNA4-RJ-062`.
- RDNA4 opcode/decode coverage exists for all audited Chapter 7.12 dense WMMA
  and SWMMAC names through generated decoder tables and baseline encodings;
  LLVM spot checks distinguish default SWMMAC encoding from `index_key:1`, with
  disassembly mismatches recorded in `RDNA4-RJ-061`.
- SWMMAC operand grouping is modeled: generated RDNA4 bodies use `SRC2` as the
  sparse index base, alias `VDST` as the accumulator source, and write back
  through the same destination register group.
- The dense C/D wave64 output helper matches the manual's 16x16 result chunking
  formula for the audited output shapes, and existing shared-infra coverage
  checks representative wave64 output lanes.
- The sparse 2:4 expansion after index-set selection follows the manual's
  `idx0 < idx1` placement model through `swmmac_dense_k`.
- Whole-wave functional execution is implemented in the matrix helpers by
  iterating/writing all result lanes rather than gating on the current `EXEC`
  mask. Existing coverage still seeds full `EXEC`, so zero-`EXEC` regression
  coverage remains thin.
- RDNA4 Chapter 12.1.2 narrow match: indexed DS load/store/atomic plumbing
  exists for RDNA4. Generated VDS instructions use VGPR address/data operands,
  route through `LOCAL_MEM`, and complete through `LocalMemPipeline`; examples
  include `DS_STORE_B32` at `rdna4/vds.cpp:461` through `:490`, DS address
  calculation at `shared/addr_calc_scalar.h:69` through `:75`, and local load
  and store handling at `vm/amdgpu/memory_pipeline.cpp:520` through `:596`.
- RDNA4 Chapter 12.1.1 narrow match: CU versus WGP placement is represented
  structurally. `COMPUTE_PGM_RSRC1.WGP_MODE` is read only when the target ISA
  supports WGP mode at `vm/amdgpu/command_processor.cpp:976` through `:977`,
  WGP mode selects a sibling-CU LDS backing in `vm/amdgpu/spi.h:193` through
  `:239`, and the gap above is the per-workgroup size contract rather than
  complete absence of WGP placement.
- RDNA4 Chapter 12 intro narrow match: loads from memory into LDS have runtime
  paths outside VDSDIR. Buffer/global load-to-LDS style memory operations can
  complete into LDS through `complete_lds_dst_load` at
  `vm/amdgpu/memory_pipeline.cpp:124` through `:127`, and flat shared-aperture
  accesses can be routed to `LOCAL_MEM` at `vm/amdgpu/compute_unit.cpp:272`
  through `:281`.

## Cross-Architecture Notes Found During This Slice

During the RDNA4 Chapter 5.6 pass, `gfx1250` was useful as a contrast case:
it generates `S_GET_BARRIER_STATE` at `gfx1250/sop1.cpp:1816` through `:1839`,
but the generated execute body writes zero at `:1840` rather than packing
`valid`, `memberCnt`, and `signalCnt`. This should be carried into a dedicated
gfx1250 barrier audit rather than treated as generic RDNA4 coverage.

CDNA4 uses the same OCP helpers for `V_CVT_{PK,SR}_{FP8,BF8}_F32` at
`lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna4/vop3.cpp:11329` through
`:11432`, and its manual has the same `FP16_OVFL` table at
`workspace_docs/amdgpu-isa-manuals/cdna4/README.md:2402` through `:2418`.
CDNA3 uses FNUZ helpers for the same instruction names at
`lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/vop3.cpp:6086` through
`:6192`; CDNA3 still needs a separate FNUZ-versus-manual audit because its
manual also has the `FP16_OVFL` table and `SH_MEM_CONFIG` requirement at
`workspace_docs/amdgpu-isa-manuals/cdna3/README.md:2034` through `:2048`.

LLVM assembler sanity checks accepted CDNA3/CDNA4 non-VGPR source forms that
the CDNA manuals say are illegal. Rocjitsu currently follows the XML-derived
operand classes for those sources. Hardware behavior and LLVM diagnostic intent
remain unaudited.
