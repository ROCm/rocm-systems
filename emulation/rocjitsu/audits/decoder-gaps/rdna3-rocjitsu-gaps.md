# RDNA3 Rocjitsu Gaps

Architecture: RDNA3

Manual source: `workspace_docs/amdgpu-isa-manuals/rdna3/README.md`

Rocjitsu source: `emulation/rocjitsu`

## Coverage

| Area | Status | Notes |
| --- | --- | --- |
| RDNA3 Chapter 1 introduction / hardware overview | Audited statically | Checked RDNA3 config/topology, ISA wave-size traits, command-processor dispatch setup, WGP-mode/LDS placement, GDS handling, L1/L2/memory-side cache surfaces, completion interrupts, and architecture-level boundaries. Detailed operational gaps remain tracked under later wave, LDS, cache, memory, and exception rows. |
| RDNA3 Chapter 2 shader concepts / wave modes / shader types / work-groups / padding | Audited statically | Checked fixed wave-size dispatch, compute grid/workitem initialization, graphics launch absence, WGP-mode placement, workgroup and LDS capacity handling, LDSDIR WGP-mode exclusions, barrier behavior, shader-padding/prefetch instruction shells, and existing detailed wave/barrier/LDS/export gaps. |
| RDNA3 Chapter 3.1-3.2 wave state / PC / EXEC / skipping | Audited statically | Checked wavefront state storage, scalar special-source access, `STATUS` layout, direct-PC and branch execution, HWREG access, trap/TBA/TMA paths, wait-counter accounting, vector-memory/DS lane-mask setup, and zero-`EXEC` scheduling behavior. |
| RDNA3 Chapter 3.3.1 SGPR allocation / VCC / alignment / out-of-range | Audited statically | Checked scalar operand resolution, TTMP/VCC/special selector handling, scalar64 reads/writes, physical SGPR access, `S_MOVREL*` helpers, VCC consumer/producer execution, and C++ def-use metadata surfaces. |
| RDNA3 Chapter 3.3.2 VGPR allocation / deallocation / out-of-range | Audited statically | Checked VGPR allocation state, zero-count allocation handling, `S_SENDMSG` deallocation, operand VGPR read/write paths, SIMD region access, VMEM address VGPR collection, `V_MOVREL*`/`V_SWAPREL`, and VOPD source/destination behavior. |
| RDNA3 Chapter 3.3.3 memory alignment / memory-operation out-of-range | Audited statically | Checked memory pipeline issue/writeback, vector memory address helpers, generated flat/buffer/DS return paths, image instruction stubs, LDS/GPU memory byte accessors, atomic RMW paths, and SH_MEM_CONFIG/MEMVIOL state surfaces. |
| RDNA3 Chapter 3.3.4 LDS allocation / placement / alignment / out-of-range | Audited statically | Checked LDS capacity and placement, CU/WGP backing stores, per-wave LDS base assignment, DS address helpers, generated DS special forms, LDS load/store/atomic helpers, and `LDS_CONFIG`/MEMVIOL state surfaces. |
| RDNA3 Chapter 3.4.1-3.4.2 STATUS / MODE registers | Audited statically | Checked generated STATUS bitfield layout, raw STATUS/MODE storage, HWREG get/set behavior, dispatch/reset status initialization, `S_ROUND_MODE`/`S_DENORM_MODE`, representative floating-point min/max helpers, and trap/export/exception state surfaces. |
| RDNA3 Chapter 3.4.3-3.4.6 M0 / NULL / SCC / VCC / VCCZ | Audited statically | Checked wavefront M0/VCC/SCC storage, scalar special-source and destination handling for NULL/M0/VCC/SCC/VCCZ, representative SALU SCC updates, SCC/VCC branch consumers, VOPC compare result writes, and VCC carry-destination paths. |
| RDNA3 Chapter 3.4.7-3.4.8 FLAT_SCRATCH / hardware internal registers | Audited statically | Checked scratch-base storage/reset, dispatch initialization, scratch/flat-private address calculation, scalar selector handling for FLAT_SCRATCH-like aliases, generated operand bounds, and RDNA3 S_GETREG/S_SETREG HWREG coverage. |
| RDNA3 Chapter 3.4.9 trap and exception registers | Audited statically | Checked TTMP privilege overlap, TBA/TMA KFD storage and return-message behavior, `S_TRAP` generation/execution metadata, `S_RFE_B64`, STATUS/MODE trap-control fields, TRAPSTS storage absence, and existing trap/exception gap coverage. |
| RDNA3 Chapter 3.4.10 time counters | Audited statically | Checked `SHADER_CYCLES` via `S_GETREG`, `REALTIME` via `S_SENDMSG_RTN_B64`, simulation clock sources, and wait-counter behavior. |
| RDNA3 Chapter 3.5.1-3.5.2 initial `EXEC` / `FLAT_SCRATCH` state | Audited statically | Checked dispatch-time active-lane mask seeding, scratch-base reset, private-segment scratch initialization, and flat-scratch user-SGPR mirroring against the manual's launch-state rules. |
| RDNA3 Chapter 3.5.3 SGPR initialization | Audited statically | Checked graphics-stage SGPR preload boundaries, AMDHSA compute user-SGPR setup, compute workgroup-id system SGPRs, and `TG_SIZE` launch payload handling. |
| RDNA3 Chapter 3.5.4 VGPR initialization | Audited statically | Checked HS/GS combined-stage VGPR payload tables, GS fast-launch rows, PS input CAM and `SPI_PS_INPUT_ENA`/`SPI_PS_INPUT_ADDR` routing, plus rocjitsu compute workitem-ID VGPR initialization. |
| RDNA3 Chapter 3.5.5 LDS initialization | Audited statically | Checked PS-only prelaunch LDS vertex-parameter preload against rocjitsu LDS backing, compute LDS setup, LDSDIR shells, and graphics launch boundaries. |
| RDNA3 Chapter 4 shader instruction set / common fields / cache controls | Audited statically | Checked common selector and literal handling, reserved selectors, unused padding fields, and generated SLC/GLC/DLC cache-flag mapping. |
| RDNA3 Chapter 5 program flow control | Audited statically | Checked program-control, scheduling, clause, message, branch, barrier, wait-counter, wait-event, and `S_DELAY_ALU` behavior against generated SOP1/SOPK/SOPP execution and VM wait/barrier state. |
| RDNA3 Chapter 6.1/6.2 SALU formats and operands | Audited statically | Checked generated SALU decode shape, scalar operand selectors, literal availability, out-of-range/alignment overlap, and `S_SETREG_IMM32_B32` literal decode/execution/disassembly metadata. |
| RDNA3 Chapter 6.3-6.7 SALU SCC/arithmetic/conditional/compare/bit-wise behavior | Audited statically | Checked representative execute bodies, implicit SCC metadata, WREXEC destination restrictions, relative SGPR moves, bit-compare masks, and detailed arithmetic/bitwise behavior. |
| RDNA3 Chapter 6.8-6.9 SALU access and memory aperture query | Audited statically | Checked state-register instructions, `S_ROUND_MODE`/`S_DENORM_MODE`, and aperture selector resolution against existing HWREG/MODE/aperture state gaps. |
| RDNA3 Chapter 7.1 VALU microcode encodings | Audited statically | Checked generated VOP3/VOP3P/VOP3SD raw field storage, literal sizing, disassembly modifier hooks, VOP3 promotion boundaries, and representative VOP3SD scalar-destination paths. |
| RDNA3 Chapter 7.2.1-7.2.2 and 7.2.5-7.2.6 VALU operand/source rules | Audited statically | Checked non-standard operand constructors, readlane/writelane lane selectors, CNDMASK/carry/div-fmas scalar paths, literal selector rewrites, source-combination validation boundaries, and wave64 SGPR/mask boundaries. |
| RDNA3 Chapter 7.2.3-7.2.4 and 7.2.7-7.2.8 VALU output/mode/edge rules | Audited statically | Checked output destinations, `V_CMPX`/carry output behavior, representative VOP3 floating-point `OMOD`/`CLAMP` paths, integer `CLAMP` handling, wave64 SGPR destination hazards, MODE round/denorm boundaries, VALU out-of-range boundaries, and `PERMLANE` sequencing hazards. |
| RDNA3 Chapter 7.3 VALU instruction inventory | Audited statically | Checked generated mnemonic coverage, VOP1/VOP2/VOP3/VOP3SD classification, compare-family expansion, baseline encodings, and unconditional execute stubs for inventory-listed instructions. |
| RDNA3 Chapter 7.4 16-bit Math and VGPRs | Audited statically | Checked compact true16 VGPR half-addressing in VOP1/VOP2/VOPC operands, OPSEL-based VOP3/VOP3P/VINTERP true16 destination handling, VINTERP F16 constructors, and shared true16 source/destination helpers. |
| RDNA3 VOP3P packed math | Audited statically | Checked generated packed F16/I16/U16 helpers, MIX helpers, DOT2 helpers, inline constants, selector handling, clamp behavior, and DPP reachability for this slice. |
| RDNA3 Chapter 7.6 / VOPD dual issue | Audited statically | Checked handwritten VOPD decode, literal sizing, slot dispatch, disassembly, source/destination metadata, implicit VCC handling, pair legality boundaries, ignored MOV fields, and paired writeback behavior. |
| RDNA3 Chapter 7.7 DPP | Audited statically | Checked shared DPP16/DPP8 helpers, generated DPP suffix constructors, ordinary VGPR destination write masking, compact/VOP3 compare mask handling, DPP legality exclusions, DPP8 FI behavior, and overlap with VOPD/VINTERP packed-math exclusions. |
| RDNA3 Chapter 7.8 VGPR Indexing | Audited statically | Checked `M0`-indexed `V_MOVREL*`/`V_SWAPREL` formulas, unsigned and split-index uses of `M0`, generated VOP1/VOP3 constructors and execute bodies, and overlap with the existing indexed-VGPR execution gap. |
| RDNA3 Chapter 7.9 WMMA | Audited statically | Checked generated WMMA constructors and shared GFX11 WMMA helpers against opcode availability, source operand classes, C-only inline constants, DPP/literal legality, wave-size/layout helpers, `NEG`/`NEG_HI` overloads, and dependent-WMMA scheduling. |
| RDNA3 Chapter 8 scalar memory operations | Audited statically | Checked generated SMEM decode/constructors, raw scalar-load addressing, scalar buffer-resource addressing, SOFFSET metadata, scalar memory pipeline behavior, cache invalidation paths, wait-counter accounting, and overlap with existing cache-flag/alignment/clause findings. |
| RDNA3 Chapter 9 vector buffer operations | Audited statically | Checked generated MUBUF/MTBUF decode/constructors, address calculation, descriptor use, data-format/D16 behavior, TFE/cache/atomic handling, memory-pipeline writeback, and overlap with existing cache-flag/alignment/return-window findings. |
| RDNA3 Chapter 10 vector image operations | Audited statically | Checked generated MIMG decode/constructors, image load/store/atomic/query/sample/gather execution bodies, BVH execution hooks, MIMG operand-window metadata, and overlap with existing return-window/cache/vector-memory gaps. |
| RDNA3 Chapter 11 global/scratch/flat address space | Audited statically | Checked generated FLAT/GLOBAL/SCRATCH decode/constructors, mnemonic rewriting, operand metadata, address calculation, aperture routing, scratch setup, `GLOBAL_ATOMIC_CSUB_U32`, D16/data movement, and overlap with existing wait-counter and memory-error gaps. |
| RDNA3 Chapter 12 data share operations | Audited statically | Checked generated DS/LDSDIR decode/constructors, ordinary LDS load/store/atomic execution, single/dual address calculation, ADDTID, append/consume, `DS_WRAP_RTN_B32`, permute/swizzle, GDS/GWS/streamout boundaries, LDSDIR stubs, BVH stack overlap, wait-counter classification, and existing LDS allocation/alignment gaps. |
| RDNA3 Chapter 13.1-13.3 float memory atomics | Audited statically | Checked generated LDS F32/F64 and cache F32 float atomic constructors, shared memory-pipeline RMW helpers, min/max/add/compare-store behavior, denorm/MODE boundaries, NaN/signed-zero rules, and overlap with the existing broad MODE-control gap. |
| RDNA3 Chapter 13.4 GWS and ordered count | Audited statically | Checked generated GWS and ordered-count constructors, execution stubs, memory-op classification, wait-counter participation, GWS resource-state absence, ordered-count field handling, `EXEC==0` behavior, and overlap with existing GDS absence. |
| RDNA3 Chapter 14 exports | Audited statically | Checked generated EXP decode/constructor, target/source metadata, implicit `EXEC`/`M0` handling, no-op execution, `EXPCNT` producer absence, export-buffer state absence, `DONE`/target/dual-source rules, `SKIP_EXPORT`, and overlap with existing STATUS/zero-EXEC gaps. |
| RDNA3 Chapter 15 remaining microcode formats | Audited statically | Checked generated scalar/control field structs and literal sizing, VOP1/VOP2/VOPC/VOP3/VOP3SD/VINTERP raw field structs, DPP extension scaffolding, VINTERP no-extension sizing, and boundaries with existing VALU/SALU/DPP/VINTERP semantic gaps. |
| RDNA3 Chapter 16.1-16.2 SOP2/SOPK definitions | Audited statically | Checked generated SOP2 helper dispatch, scalar arithmetic/SCC, shifts, BFE/BFM, min/max, multiply, conditional select, pack forms, SOPK version/cmov/compare/add/mul/HWREG/call forms, and split wait-count operand semantics. |
| RDNA3 Chapter 16.3-16.5 SOP1/SOPC/SOPP definitions | Audited statically | Checked generated SOP1 bit/count/EXEC-save/relative/direct-PC/send-message paths, SOPC compare/bit-compare helpers, SOPP branch/wait/control/end-program/message/barrier paths, and overlap with existing state/control gaps. |
| RDNA3 Chapter 16.6 SMEM definitions | Audited statically | Checked generated SMEM scalar-load, scalar-buffer-load, `S_GL1_INV`, and `S_DCACHE_INV` instruction shells against Chapter 8 scalar-memory execution and the existing descriptor, metadata, wait-counter, and group-restriction findings. |
| RDNA3 Chapter 16.7-16.8 VOP2/VOP1 definitions | Audited statically | Checked generated VOP2/VOP1 opcode coverage, representative compact execution bodies, literal-only FMA forms, min/max/MODE overlap, relative-indexed moves, swap/permlane behavior, VOP3 aliases, and `V_PIPEFLUSH` side effects. |
| RDNA3 Chapter 16.9 VOPC definitions | Audited statically | Checked generated compact compare constructors, VOP3 aliases, VCC/EXEC runtime writes, DPP/SDWA handling, and compare-result metadata against manual/XML result-mask definitions. |
| RDNA3 Chapter 16.10 VOP3P packed/DOT definitions | Audited statically | Checked definition rows 0-26 for ordinary packed arithmetic, shifts, DOT2 F16/BF16, DOT4/DOT8 integer signedness, literal/DPP reachability, clamp handling, and overlap with existing packed-math gaps. |
| RDNA3 Chapter 16.10 VOP3P WMMA definitions | Audited statically | Checked WMMA definition rows 64-69, full-`EXEC` execution pseudocode, and overlap with generated WMMA constructors/helpers and the Chapter 7.9 WMMA findings. |
| RDNA3 Chapter 16.11 VOPD definitions | Audited statically | Checked generated VOPD MOV, CNDMASK, DOT2ACC, slot opcode, pair-legality, implicit VCC, ignored-MOV-field, and paired-exception behavior against Chapter 7.6 dual-issue findings. |
| RDNA3 Chapter 16.12 VOP3/VOP3SD definitions | Audited statically | Checked generated VOP3/VOP3SD opcode coverage, VOP3SD restriction, compare result operands, DPP/literal handling, true16 literal selection, generic modifiers, min/max/MODE overlap, and representative compare-DPP runtime masking. |
| RDNA3 Chapter 16.13 VINTERP definitions | Audited statically | Checked generated VINTERP constructors and execute bodies against F32/F16 interpolation formulas, fixed DPP8 source selection, RTZ variants, and OPSEL-controlled true16 source/destination behavior. |
| RDNA3 Chapter 16.14-16.15 LDSDIR and LDS/GDS definitions | Audited statically | Checked generated LDSDIR and DS/GDS constructors, decode, ordinary LDS execution, ADDTID, append/consume, permute/swizzle, GDS/GWS/streamout boundaries, LDSDIR stubs, and overlap with Chapter 12 data-share findings. |
| RDNA3 Chapter 16.16-16.17 MUBUF/MTBUF definitions | Audited statically | Checked generated MUBUF/MTBUF constructors, decode, raw/formatted/D16/cache/atomic buffer execution boundaries, descriptor/address/data handling, and overlap with Chapter 9 vector-buffer findings. |
| RDNA3 Chapter 16.18 MIMG definitions | Audited statically | Checked generated image load/store/atomic/query/MSAA/BVH/sample/gather constructors and execution stubs against Chapter 10 image findings, including operand-window and BVH coverage boundaries. |
| RDNA3 Chapter 16.19 EXPORT definitions | Audited statically | Checked generated EXP raw fields, constructor operands, disassembly modifiers, target enum coverage, implicit `EXEC`/`M0` metadata, `VM`/valid-mask wording, and overlap with Chapter 14 export execution gaps. |
| RDNA3 Chapter 16.20 FLAT/Scratch/Global definitions | Audited statically | Checked generated FLAT/GLOBAL/SCRATCH constructors, decode, mnemonic rewriting, operand metadata, address/aperture handling, `GLOBAL_ATOMIC_CSUB_U32` coverage, D16/data movement, and overlap with Chapter 11 flat/global/scratch findings. |
| RDNA3 `DS_BVH_STACK_RTN_B32` | Audited statically | Checked generated decode tables, operand metadata, execution stub, memory-op classification, and encoding fixture coverage. |
| Remaining RDNA3 rocjitsu surface | Not started | Full decoder/autogen/runtime fuzzing remains outside the rows above. |

## Gaps

### RDNA3-RJ-001: Packed F16 inline constants use FP32 bit patterns

Manual evidence:

- RDNA3 section 7.5.1 says inline constants used with packed math produce a
  value only in the low 16 bits, and inline constants used with float 16-bit
  sources produce an F16 constant value, at `rdna3/README.md:2648` through
  `:2656`.

Rocjitsu evidence:

- The generated ordinary packed-F16 VOP3P constructors use 32-bit `OPR_SRC`
  operands and replace literal extensions with 32-bit `OPR_SIMM32` operands at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/vop3p.cpp:1059` through
  `:1413`.
- `Operand::read_lane()` selects `resolve_src_scalar16()` only when the operand
  size is 16 bits; otherwise it uses `resolve_src_scalar()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/operand.cpp:1041` through
  `:1059`.
- The 32-bit inline resolver returns FP32 bit patterns such as `0x3F000000` for
  `0.5f` at `operand.cpp:703` through `:748`, while the 16-bit resolver returns
  F16 encodings such as `0x3800` at `operand.cpp:775` through `:789`.
- Shared packed-F16 execution reads raw 32-bit values and then selects 16-bit
  halves, for example `execute_v_pk_add_f16_vop3p` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:16664`
  through `:16698` and `execute_v_pk_fma_f16_vop3p` at `:16839` through
  `:16884`.

Impact:

An inline constant such as 0.5 is read as `0x3f000000` for ordinary packed-F16
VOP3P, so selecting the low half yields zero and selecting the high half yields
`0x3f00` instead of the manual's F16 inline constant `0x3800`.

### RDNA3-RJ-002: Ordinary packed I16/U16/F16 VOP3P arithmetic ignores `CLMP`

Manual evidence:

- The RDNA3 VOP3P packed-math field table defines `CLMP` as a result clamp:
  float arithmetic clamps to `[0, 1.0]`, signed integer arithmetic clamps to
  `[min_int, +max_int]`, and unsigned integer arithmetic clamps to
  `[0, +max_uint]` at `rdna3/README.md:2627` through `:2628`.

Rocjitsu evidence:

- Ordinary packed-F16 helpers execute without testing `inst.inst_.clamp`, for
  example `V_PK_ADD_F16`, `V_PK_FMA_F16`, `V_PK_MAX_F16`, `V_PK_MIN_F16`, and
  `V_PK_MUL_F16` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:16664`
  through `:16698`, `:16839` through `:16884`, `:17089` through `:17127`,
  `:17192` through `:17230`, and `:17316` through `:17350`.
- Ordinary packed integer helpers wrap or shift 16-bit values without checking
  `inst.inst_.clamp`, for example `V_PK_ADD_I16`, `V_PK_ADD_U16`,
  `V_PK_MAD_I16`, and `V_PK_SUB_I16` at `execute_shared.h:16744` through
  `:16813`, `:17008` through `:17052`, and `:17428` through `:17464`.
- DOT2 and MIX helpers in the same shared file do apply `inst.inst_.clamp`,
  for example `execute_v_dot2_f32_bf16_vop3p` and
  `execute_v_dot2_f32_f16_vop3p` at `execute_shared.h:10414` through `:10483`,
  and the MIX helpers at `:11470` through `:11595`.

Impact:

Packed arithmetic results outside the manual's clamp range remain wrapped or
unclamped when the VOP3P `CLMP` bit is set.

### RDNA3-RJ-003: `V_FMA_MIX*` uses multiply-add instead of fused FMA

Manual/XML evidence:

- RDNA3 instruction definitions 32-34 describe `V_FMA_MIX_F32`,
  `V_FMA_MIXLO_F16`, and `V_FMA_MIXHI_F16` as fused multiply-add operations and
  use `fma(...)` in pseudocode at `rdna3/README.md:15422` through `:15480`.
- The XML entries for `V_FMA_MIX_F32`, `V_FMA_MIXLO_F16`, and
  `V_FMA_MIXHI_F16` are present at `amdgpu_isa_rdna3.xml:121083`, `:121409`,
  and `:121735` and describe fused multiply-add behavior.

Rocjitsu evidence:

- `execute_v_fma_mix_f32_vop3p`, `execute_v_fma_mixhi_f16_vop3p`, and
  `execute_v_fma_mixlo_f16_vop3p` compute `float result = a * b + c;` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:11470`
  through `:11595`.

Impact:

One-rounding FMA cases can differ from hardware and the RDNA3 ISA. This is a
real RDNA3 semantic mismatch; unlike CDNA3/CDNA4, the audited RDNA3 manual and
XML sources both say fused.

### RDNA3-RJ-004: `DS_BVH_STACK_RTN_B32` decodes but is not executable

Manual/XML evidence:

- The RDNA3 manual describes `DS_BVH_STACK_RTN_B32` as an LDS-backed per-thread
  BVH stack operation with packed `ADDR` state, `OFFSET1[5:4]` stack-size
  selection, `DATA_VALID` filtering, push/pop behavior, and memory
  invalidation at `rdna3/README.md:4919` through `:4940` and
  `rdna3/README.md:23530` through `:23579`.
- The RDNA3 XML lists opcode 173 with explicit `VDST`, in-out `ADDR`, `DATA0`,
  `DATA1`, and implicit DSMEM operands at `amdgpu_isa_rdna3.xml:24285` through
  `:24333`.

Rocjitsu evidence:

- Rocjitsu generates `DsBvhStackRtnB32Ds` with the expected explicit operands
  and in-out `ADDR` modeling, but `execute_impl()` throws
  `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/ds.cpp:4097` through
  `:4116`.
- The Python semantic classifier falls unrecognized DS instructions through to
  semantic class `nop` at `lib/python/amdisa/semantics.py:2418` through
  `:2429`, and the generator emits `UnimplementedInst` for `nop` at
  `lib/python/amdisa/codegen/_generator.py:3377` through `:3378`.

Impact:

Any shader containing this legal RDNA3 stack operation decodes to a concrete
instruction object, but execution stops at runtime instead of updating LDS,
`VDST`, and `ADDR`.

### RDNA3-RJ-005: `DS_BVH_STACK_RTN_B32` is not marked as a memory operation

Manual/XML evidence:

- The instruction is described as an LDS instruction for pushing/popping a
  per-thread stack at `rdna3/README.md:4919` through `:4923` and
  `rdna3/README.md:23530` through `:23579`.
- The XML entry includes implicit DSMEM input/output operands and places the
  instruction in the `VMEM`/`DATA_SHARE` functional group at
  `amdgpu_isa_rdna3.xml:24317` through `:24333`.

Rocjitsu evidence:

- The generated `DsBvhStackRtnB32Ds` constructor sets source/destination
  operands but does not set `flags_ |= MEMORY_OP` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/ds.cpp:4097` through
  `:4111`.
- The generator only emits `MEMORY_OP` for recognized memory semantic classes,
  including DS read/write/atomic-style classes, at
  `lib/python/amdisa/codegen/_generator.py:6208` through `:6240` and
  `:6471` through `:6472`; this instruction currently falls through to
  `nop`.

Impact:

Analysis or scheduling code that relies on rocjitsu's memory-operation flag
will treat this LDS side-effecting instruction as non-memory until the semantic
classification is corrected.

### RDNA3-RJ-006: RDNA3 GDS is decoded but not modeled

Manual evidence:

- Chapter 1.1 defines GDS as scratch memory shared by all shader engines, with
  append-operation support, at `rdna3/README.md:385`.
- Chapter 1.2.2.2 says RDNA3 devices use a 4KiB GDS visible to waves of a
  kernel on all WGPs, with 128 bytes per cycle of access, two integer atomic
  units, preload/writeback around launch, and unordered plus domain-launch
  ordered append/consume support, at `rdna3/README.md:497` through `:499`.

Rocjitsu evidence:

- The KFD sysfs emulation advertises `gds_size_in_kb 0` for GPU nodes at
  `lib/rocjitsu/src/rocjitsu/kmd/linux/sysfs.cpp:320` through `:329`.
- The device config struct has LDS, L1, and L2 fields but no GDS size field at
  `lib/rocjitsu/src/rocjitsu/config/kfd_device_config.h:42` through `:50`.
- Generated RDNA3 DS execution bodies throw `UnimplementedInst` when the DS
  `gds` bit is set; for example `DsAddU32Ds` checks `inst_.gds` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/ds.cpp:38` through `:40`.
- Searches found GDS wait-counter naming and generated `gds` decode checks, but
  no RDNA3 GDS backing store, preload/writeback path, atomic units, or
  append/consume implementation.

Impact:

Legal RDNA3 DS encodings that select GDS can decode, but they stop at
execution instead of accessing a shared 4KiB GDS surface. Runtime-visible device
properties also report no GDS even though Chapter 1 says RDNA3 devices have
one.

### RDNA3-RJ-007: RDNA3 Wave64 dispatch is not selectable

Manual evidence:

- Chapter 2.1 says RDNA3 supports wave32 and wave64, and that shaders are
  compiled for one fixed wave size at `rdna3/README.md:541` through `:543`.
- The same section defines wave64 issue behavior, second-pass scalar
  input/output increments, and wave32 upper-`EXEC`/`VCC` ignored behavior at
  `rdna3/README.md:545` through `:563`.

Rocjitsu evidence:

- `RdnaIsaBase` sets `WF_SIZE = 32` and `WF_SIZE_MAX = 64` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/rdna_isa_base.h:50`
  through `:52`.
- `IsaExecComputeUnit` constructs the execution CU with `Isa::WF_SIZE` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.h:683` through `:685`.
- The W7900 RDNA3 config advertises `wave_front_size` 32 at
  `configs/gfx1100_w7900.json:25`.
- `CommandProcessor::process_aql_packet` computes waves per workgroup from
  `cus_[0]->wf_size()` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:941` through
  `:944`; the AMDHSA `ENABLE_WAVEFRONT_SIZE32` kernel descriptor bit is
  defined at
  `lib/rocjitsu/external_headers/hsa_headers/hsa/AMDHSAKernelDescriptor.h:190`,
  but is not consumed in that path.

Impact:

RDNA3 wave64 kernels are dispatched and executed as wave32 in rocjitsu, so
wave64-specific `EXEC`/`VCC` behavior, issue passes, scalar carry/divergence
increments, and VGPR allocation granularity are not modeled.

### RDNA3-RJ-008: Workgroup size and residency limits are resource-only

Manual evidence:

- Chapter 2.3 says a WGP supports up to 32 workgroups in flight and a maximum
  of 1024 work-items per workgroup at `rdna3/README.md:588`.
- The same section says single-wave workgroups do not count against the
  32-workgroup limit, do not allocate a barrier resource, and treat barrier
  operations as `S_NOP` at `rdna3/README.md:590`.

Rocjitsu evidence:

- The command processor derives `wg_size` from
  `workgroup_size_x * workgroup_size_y * workgroup_size_z` and computes
  `wfs_per_wg` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:941` through
  `:944`, then stores the dimensions directly in `DispatchEntry` at
  `command_processor.cpp:1045` through `:1047`.
- `ComputeUnitCore::can_accept_workgroup()` admits workgroups based on free
  wave slots, register blocks, and LDS capacity at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:215` through `:251`.
  It does not enforce a max-1024 work-item contract.
- The WGP allocation path increments `active_workgroups` and advances LDS
  allocation at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/spi.h:203` through
  `:237`, but searches found no 32-workgroup cap or single-wave barrier
  resource elision rule.
- `S_BARRIER` always transitions the wave to barrier state through
  `execute_s_barrier_sopp` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:745`
  through `:746`; barrier resolution waits for all wavefronts in the workgroup
  at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:314` through
  `:333`.

Impact:

Rocjitsu can accept and schedule RDNA3 launches whose workgroup dimensions or
resident-workgroup count are outside the Chapter 2 hardware contract. It also
models single-wave barriers as ordinary barrier state transitions instead of a
launch-time no-barrier-resource / `S_NOP` special case.

### RDNA3-RJ-009: Graphics shader launch modes are not modeled

Manual evidence:

- Chapter 2.2.2 describes pixel, geometry, and hull shader waves, says the
  normal NGG geometry-engine launch initializes VGPRs with primitive/index and
  vertex-buffer data, and describes mesh-shader plus amplification-shader
  launch modes at `rdna3/README.md:571` through `:584`.

Rocjitsu evidence:

- `DispatchEntry` is compute/AQL shaped, with kernel dispatch identifiers,
  grid sizes, workgroup sizes, workgroup IDs, and WGP mode at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/dispatch_entry.h:43` through `:87`.
- `CommandProcessor::init_wavefront_regs` initializes AMDHSA compute SGPRs and
  local work-item IDs at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:174` through
  `:321`.
- Searches found no PS/GS/HS, geometry-engine, mesh-shader, or amplification
  shader launch path beyond generic shader-engine topology names.

Impact:

Rocjitsu is compute-dispatch oriented for this layer. It cannot emulate or
validate Chapter 2 graphics-stage wave creation, geometry-engine VGPR payloads,
mesh shader launch conversion, or amplification shader control.

### RDNA3-RJ-010: LDSDIR instructions are stubbed and WGP-mode legality is not enforced

Manual evidence:

- Chapter 2.3 says WGP mode exposes one contiguous WGP LDS and allows waves in
  a workgroup to be distributed across both CUs, but `LDS_PARAM_LOAD` and
  `LDS_DIRECT_LOAD` are not supported in WGP mode at `rdna3/README.md:600`
  through `:602`.

Rocjitsu evidence:

- RDNA3 generates concrete `LdsParamLoadLdsdir` and `LdsDirectLoadLdsdir`
  instruction classes at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/ldsdir.cpp:20` through
  `:46`.
- Both classes dispatch to shared helpers, but
  `execute_lds_param_load_ldsdir` and `execute_lds_direct_load_ldsdir` are
  empty at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:337`
  through `:343`.
- The command processor records `dp.wgp_mode` for placement and LDS-capacity
  checks at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:976`
  through `:999`, but the LDSDIR execute path has no WGP-mode rejection or
  decode-time legality guard.

Impact:

RDNA3 LDS parameter/direct load instructions decode but do not transfer data in
either CU or WGP mode. WGP-mode shaders that contain those unsupported
instructions can pass decode and execute as no-ops instead of being rejected or
flagged.

### RDNA3-RJ-011: RDNA3 HWREG get/set uses the wrong and incomplete state map

Manual/XML evidence:

- Chapter 3.1 lists `STATUS`, `MODE`, and `TRAPSTS` as 32-bit shader-visible
  state and lists `TBA`, `TMA`, `TTMP0`-`TTMP15`, and wait counters in the same
  visible-state table at `rdna3/README.md:636` through `:651`.
- The RDNA3 XML `OPR_HWREG` map assigns `HW_REG_MODE` to ID 1,
  `HW_REG_STATUS` to ID 2, `HW_REG_TRAPSTS` to ID 3, `HW_REG_GPR_ALLOC` to ID 5,
  `HW_REG_LDS_ALLOC` to ID 6, `HW_REG_IB_STS` to ID 7, and PC/TBA/flat-scratch
  IDs at `amdgpu_isa_rdna3.xml:174323` through `:174440`.
- XML represents `S_GETREG_B32` and `S_SETREG_B32` as HWREG access
  instructions at `amdgpu_isa_rdna3.xml:55994` and `:56032`.

Rocjitsu evidence:

- RDNA3 `SGetregB32Sopk::execute_impl` treats HWREG ID 1 as `wf.status_raw()`
  and handles only IDs 1, 4, 5, 6, and 7 before logging unhandled IDs at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/sopk.cpp:287` through
  `:317`.
- RDNA3 `SSetregB32Sopk` and `SSetregImm32B32Sopk` also only splice writes
  into `wf.status_raw()` for ID 1 at `rdna3/sopk.cpp:336` through `:356` and
  `:373` through `:393`.
- `Wavefront` stores raw MODE separately from raw STATUS at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:95` through `:110`, and the
  concrete RDNA3 wavefront exposes a separate `StatusReg` through
  `status_raw()`/`set_status_raw()` at `wavefront.h:608` through `:617`.

Impact:

RDNA3 `s_getreg_b32 hwreg(MODE,...)` observes STATUS in rocjitsu, while
architectural STATUS ID 2, TRAPSTS ID 3, TBA/TMA, PC, flat scratch, and several
allocation/status registers are unreachable or mapped to unrelated ad hoc
values. `s_setreg*` also writes STATUS through the MODE ID instead of applying
the architectural MODE/STATUS/TRAPSTS access policy.

### RDNA3-RJ-012: Raw `STATUS.EXECZ` and `STATUS.VCCZ` can drift from live masks

Manual evidence:

- Chapter 3.1 defines `EXECZ` and `VCCZ` as summary flags that indicate whether
  the live `EXEC` or `VCC` mask is all zero, using only the low 32 bits for
  wave32 at `rdna3/README.md:631` through `:633`.
- Chapter 3.2.2 repeats that wave32 hardware acts only on `EXEC[31:0]`, and
  `EXECZ` reflects that low-half state at `rdna3/README.md:669` through `:671`.

Rocjitsu evidence:

- The RDNA3 `StatusReg` layout includes `EXECZ` at bit 9 and `VCCZ` at bit 10
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/isa.h:43` through `:46`.
- `Wavefront::set_exec()`, `set_exec_raw()`, and `set_vcc()` update only the
  live mask fields at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:211`
  through `:238`.
- The scalar special-source resolver correctly synthesizes live `VCCZ` and
  `EXECZ` values from `wf.vcc()`/`wf.exec()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/operand.cpp:763` through
  `:768`, but raw `status_raw()` remains an independently stored field exposed
  by HWREG access as described in `RDNA3-RJ-011`.

Impact:

Code that reads the special scalar `EXECZ`/`VCCZ` sources observes live masks,
but raw `STATUS` inspection can observe stale or manually written summary bits
that no longer match the architectural `EXEC`/`VCC` state.

### RDNA3-RJ-013: Trap, TBA/TMA, and trap-return state is not modeled

Manual/XML evidence:

- Chapter 3.1 lists `TRAPSTS`, `TBA`, `TMA`, and `TTMP0`-`TTMP15` as
  shader-visible trap state at `rdna3/README.md:639` through `:647`.
- Chapter 3.2.1 says `S_RFE_B64` is one of the direct PC operations and that
  `S_TRAP` saves the PC of the trap instruction itself at
  `rdna3/README.md:659` through `:661`.
- XML exposes `HW_REG_TRAPSTS`, `HW_REG_SHADER_TBA_LO`,
  `HW_REG_SHADER_TBA_HI`, and the TMA/TBA return-message IDs at
  `amdgpu_isa_rdna3.xml:174333`, `:174378` through `:174383`, and
  `:175258` through `:175278`.

Rocjitsu evidence:

- `Wavefront` stores status, mode, `EXEC`, `VCC`, `M0`, scratch/aperture bases,
  and wait counters, but has no `TRAPSTS`, `TBA`, or `TMA` state members at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:532` through `:545`.
- RDNA3 `S_TRAP` is marked `PROGRAM_TERMINATOR` and throws
  `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/sopp.cpp:156` through
  `:168`.
- Shared `execute_s_rfe_b64_sop1` is empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2405`
  through `:2406`, and the RDNA3 `S_RFE_B64` class dispatches to that helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/sop1.cpp:1227` through
  `:1239`.
- RDNA3 `S_SENDMSG_RTN_B32` only returns a nonzero value for message `0x83` and
  returns zero for message IDs including `0x82`, `0x85`, and `0x86` at
  `rdna3/sop1.cpp:1256` through `:1281`; the XML names those values
  `MSG_RTN_GET_TMA`, `MSG_RTN_GET_TBA`, and `MSG_RTN_GET_TBA_TO_PC`.

Impact:

Trap handlers, trap status inspection, TBA/TMA queries, trap-entry PC save, and
trap return cannot be emulated faithfully. Legal trap-control instructions
either throw, no-op, or return zero instead of mutating the Chapter 3 trap
state.

### RDNA3-RJ-014: Indirect PC writes do not force DWORD alignment and zero targets halt

Manual evidence:

- Chapter 3.2.1 says the PC is a DWORD-aligned byte address whose low two bits
  are forced to zero at `rdna3/README.md:657`.
- The same section says `S_SETPC_B64`, `S_RFE_B64`, and `S_SWAPPC_B64` transfer
  the PC to and from even-aligned SGPR pairs, while branches are relative to the
  next instruction at `rdna3/README.md:659` through `:661`.

Rocjitsu evidence:

- `SSetpcB64Sop1::execute_impl` assigns
  `wf.pc = read_scalar64(ssrc0) - size_` without masking low address bits at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/sop1.cpp:1201` through
  `:1203`.
- `SSwappcB64Sop1::execute_impl` likewise reads the raw scalar target, assigns
  it minus `size_`, and writes the next PC to the destination at
  `rdna3/sop1.cpp:1221` through `:1224`.
- The outer CU loop special-cases `s_setpc` and `s_swappc`: if the raw scalar
  target is zero it halts the wave before executing the instruction at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:395` through `:407`.

Impact:

Misaligned indirect branch targets keep their low bits in rocjitsu instead of
being forced to DWORD alignment, and an indirect branch to address zero becomes
a wave halt rather than an architectural PC update.

### RDNA3-RJ-015: `EXEC==0` skip and counter-dependent issue rules are not modeled

Manual evidence:

- Chapter 3.2.3 says hardware may skip vector instructions when `EXEC==0`,
  with specific no-skip exceptions for SGPR/VCC writers, WMMA/SWMMA,
  `V_READLANE`/`V_WRITELANE`, buffer invalidations, selected wave64 issue-twice
  cases, export `Done`/`POS0`/`SKIP_EXPORT` behavior, and VMEM/LDS skip
  decisions that depend on outstanding VM/VScnt, EXPcnt, or LGKMcnt at
  `rdna3/README.md:673` through `:694`.

Rocjitsu evidence:

- `ComputeUnitCore::issue_instruction()` executes every decoded instruction
  through `execute_instruction(inst, *active)` without a zero-`EXEC` scheduler
  skip gate at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:393`
  through `:439`.
- Vector-buffer, flat/global/scratch, and DS address calculation seed
  `VectorMemState::lane_mask` directly from `wf.exec()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h:48`
  through `:53`, `shared/addr_calc_flat.h:38` through `:44`, and
  `shared/addr_calc_scalar.h:60` through `:65`.
- Memory instructions are still routed to a memory pipeline when
  `inst->is_memory_op()` is true at `compute_unit.cpp:430` through `:435`, and
  `MemoryPipeline::issue()` increments the selected wait counter before
  initiating access at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.h:62` through `:78`.

Impact:

For many VALU cases the per-lane mask can make final VGPR contents match, but
rocjitsu does not model the Chapter 3 zero-`EXEC` issue decision itself.
Counter-visible cases can also diverge: a zero-lane VMEM/LDS instruction can
still increment a wait counter in rocjitsu even when the manual permits or
requires the instruction to be skipped under specific outstanding-counter
conditions.

### RDNA3-RJ-016: SGPR out-of-range and TTMP privilege rules are not enforced

Manual evidence:

- Section 3.3.1.4 defines scalar selector regions, says out-of-range source
  SGPRs return zero and out-of-range destination SGPR writes are ignored, and
  says TTMP reads return zero and TTMP writes are ignored outside trap mode at
  `rdna3/README.md:735` through `:750`.
- The same section applies these rules to SALU, VALU, VMEM, and SMEM scalar
  operands, including SMEM destination-range suppression at `rdna3/README.md:752`
  through `:762`, and says `S_MOVREL` source OOR uses `S0` while destination OOR
  writes nothing at `rdna3/README.md:763` through `:771`.

Rocjitsu evidence:

- RDNA3 scalar source resolution reads SGPR and TTMP selector encodings directly
  from `wf.sgpr_alloc().base + ev` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/operand.cpp:703` through
  `:715`, without checking `wf.num_sgprs()` or `STATUS.PRIV`.
- RDNA3 64-bit scalar source resolution reads `ev` and `ev + 1` directly for
  SGPR and TTMP ranges at `rdna3/operand.cpp:810` through `:827`.
- RDNA3 scalar writes and scalar64 writes write SGPR/TTMP selector encodings
  directly at `rdna3/operand.cpp:870` through `:943`, without privilege or
  out-of-range suppression.
- The underlying CU SGPR accessors index the physical SGPR file directly at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.h:379` through `:392`.

Impact:

Programs that rely on architected scalar out-of-range behavior, TTMP user-mode
zeroing, TTMP write suppression, or failed-TTMP SCC preservation can execute with
stale/physical SGPR data in rocjitsu instead of the manual-defined zero/NOP
behavior.

### RDNA3-RJ-017: Multi-dword SGPR alignment and `S_MOVREL` range rules are not enforced

Manual evidence:

- Section 3.3.1.3 says 64-bit SGPR data must be even-aligned, wider data must be
  quad-aligned, misaligned multi-dword scalar operands are illegal/unpredictable,
  and hardware enforces alignment by ignoring low address bits; `*MOVREL*_B64`
  also ignores the low bit of the index at `rdna3/README.md:714` through `:731`.
- Section 3.3.1.4 says multi-DWORD operands and GPR indexing must not cross
  scalar regions, and `S_MOVREL` indexing must stay inside the base SGPR/VCC or
  TTMP range at `rdna3/README.md:739` through `:771`.

Rocjitsu evidence:

- RDNA3 `resolve_src_scalar64()` and `resolve_dst_write64()` use the encoded
  scalar selector as-is and access `ev + 1` for the high dword at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/operand.cpp:810` through
  `:827` and `:914` through `:943`; they do not clear low selector bits or
  enforce quad alignment.
- RDNA3 `SMovrelsB32Sop1`/`SMovrelsB64Sop1` and
  `SMovreldB32Sop1`/`SMovreldB64Sop1` truncate `M0` to eight bits and compute
  `base + index * width_words` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/sop1.cpp:1047` through
  `:1051`, `:1076` through `:1080`, `:1105` through `:1109`, and `:1134`
  through `:1138`.
- RDNA3 `SMovrelsd2B32Sop1` uses `M0[7:0]` for source and `M0[15:8]` for
  destination at `rdna3/sop1.cpp:1163` through `:1169`, rather than enforcing the
  manual's SGPR/TTMP base-range and out-of-range rules.

Impact:

Odd 64-bit SGPR selectors, region-crossing pairs, and indexed scalar moves can
read or write the wrong scalar registers instead of using the manual-defined
alignment masking, `S0` fallback, or destination write suppression.

### RDNA3-RJ-018: VCC dependency metadata is incomplete

Manual evidence:

- Section 3.3.1.2 says VCC is written by `V_CMP` and integer vector ADD/SUB,
  implicitly read by `V_ADD_CI`, `V_SUB_CI`, `V_CNDMASK`, and `V_DIV_FMAS`, and
  is a named SGPR pair subject to the same dependency checks as any other SGPR at
  `rdna3/README.md:708` through `:710`.

Rocjitsu evidence:

- `RegisterSet` defines `RegClass::VCC` but documents it as not currently
  tracked, and the set operations ignore untracked classes at
  `lib/rocjitsu/src/rocjitsu/isa/register_set.h:50` through `:60` and
  `lib/rocjitsu/src/rocjitsu/isa/register_set.cpp:37` through `:51`.
- RDNA3 `Operand::to_register_ref()` maps only ordinary SGPR selector ranges for
  scalar operand types and drops `OPR_VCC` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/operand.cpp:561` through
  `:685`.
- The E32 `V_CNDMASK_B32` and `V_ADD_CO_CI_U32` constructors list only explicit
  VGPR/SRC operands at `rdna3/vop2.cpp:23` through `:33` and `:3182` through
  `:3192`, while their execution reads and/or writes VCC at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:8257`
  through `:8267`, `:2834` through `:2852`, and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/simd_glue.h:592` through
  `:628`.
- The VOP3 `V_DIV_FMAS_*` constructors list only `VDST` and `SRC0`-`SRC2` at
  `rdna3/vop3.cpp:11794` through `:11806` and `:11873` through `:11885`, while
  execution reads fixed VCC at `execute_shared.h:10231` through `:10268`.

Impact:

Execution can consume and update VCC correctly for many common cases, but
rocjitsu's public def-use/liveness metadata can miss the VCC SGPR-pair
dependency that the manual says participates in ordinary SGPR dependency checks.

### RDNA3-RJ-019: VGPR deallocation and physical allocation policy are incomplete

Manual evidence:

- Section 3.3.2.1 says VGPRs are allocated in blocks of 16 for wave32 or 8 for
  wave64, with a 24/12-register exception for 1536-VGPR-per-SIMD devices, and
  that waves cannot be created with zero VGPRs at `rdna3/README.md:775` through
  `:779`.
- The same section says a wave may deallocate all VGPRs via `S_SENDMSG`, cannot
  reallocate them afterward, and may only terminate after deallocation at
  `rdna3/README.md:779`.

Rocjitsu evidence:

- The shared register allocator rejects zero-count allocations at
  `lib/simdojo/include/simdojo/components/register_file.h:44` through `:50`, so
  zero-VGPR dispatch is already blocked.
- RDNA3 compute-unit setup uses the configured `vgprs_per_wf` as the physical
  per-slot block size at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.h:693` through `:694`, and
  dispatch records the requested per-wave VGPR count directly in
  `wf->vgpr_alloc_.count` and `wf->num_vgprs_` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:139` through `:142`.
- The default config path sets RDNA3 `vgprs_per_wf` to 256 unless overridden at
  `lib/rocjitsu/src/rocjitsu/config/config_loader.cpp:72` through `:75` and
  `:461` through `:466`; no code found derives the physical allocation block
  size from RDNA3 wave32/wave64 or the 1536-VGPR/SIMD exception.
- `S_SENDMSG` decodes to an executable instruction, but the shared execution
  helper is a no-op at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2427`
  through `:2428`.

Impact:

The zero-VGPR creation rule is covered, but `MSG_DEALLOC_VGPRS` does not release
VGPRs or suppress later vector instructions, and the simulator does not encode
RDNA3 physical VGPR allocation granularity as an ISA policy. Resource/occupancy
behavior and post-deallocation execution can diverge from the manual.

### RDNA3-RJ-020: VGPR out-of-range source and destination consequences are not modeled

Manual evidence:

- Section 3.3.2.2 defines out-of-range VGPR operands from `Vs`/`Ve`, says
  out-of-range VGPR destinations make the instruction a NOP, says multi-
  destination VALU instructions suppress all GPR writes if any destination is
  out of range, and says out-of-range source VGPRs use `VGPR0` for VALU,
  VMEM, and export, with per-dword fallback for memory source groups at
  `rdna3/README.md:783` through `:815`.
- The same consequence table gives VOPD a different source rule: force the
  source address to `VGPRaddr % 4` at `rdna3/README.md:811` through `:812`.

Rocjitsu evidence:

- RDNA3 `Operand::read_lane()`, `write_lane()`, `read_lane64()`, and
  `write_lane64()` form `wf.vgpr_alloc().base + voff` and call physical VGPR
  accessors at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/operand.cpp:1041` through
  `:1108`; they do not compare `voff`/`voff + width - 1` against
  `wf.num_vgprs()` and do not substitute `VGPR0` or suppress writes.
- The SIMD operand chunk path similarly reads
  `wf.vgpr_alloc().base + voff` through a physical region at
  `lib/rocjitsu/src/rocjitsu/isa/isa_operand_simd_inl.h:53` through `:61`.
- VMEM address collection for buffer helpers reads physical regions starting at
  `wf.vgpr_alloc().base + inst.vaddr`, with no per-dword out-of-range fallback,
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h:98`
  through `:104` and `:182` through `:188`.
- Physical `RegisterAccess` and `RegisterFile` operations expose raw-index
  reads/writes and assertions, not architecture-level fallback/NOP behavior, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/register_access.h:724` through `:760`
  and `lib/simdojo/include/simdojo/components/register_file.h:77` through `:90`.
- RDNA3 VOPD execution reads its slot operands through normal `RegisterAccess`
  and writes both destinations through normal `write_lane()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/vopd.cpp:134` through
  `:138` and `:345` through `:354`; no VOPD-specific `% 4` source fallback or
  all-destination suppression is applied.

Impact:

Out-of-range VGPR operands can read or write physical storage, assert, or follow
normal operand behavior instead of the manual's `VGPR0` fallback, destination
NOP, all-destination suppression, and VOPD modulo source rule.

### RDNA3-RJ-021: `V_MOVREL*` and `V_SWAPREL` do not implement the manual's indexed VGPR contract

Manual evidence:

- Section 3.3.2.2 defines `V_MOVREL` indexed out-of-range checks, including
  `Index > 255` and `(Vs + M0)`/`(Ve + M0)` beyond `VGPR_SIZE`, at
  `rdna3/README.md:797` through `:803`.
- The consequence table says out-of-range destinations make the instruction a
  NOP, source VGPRs fall back to `VGPR0`, and `V_SWAP`/`V_SWAPREL` discard the
  whole instruction if either destination is out of range at
  `rdna3/README.md:804` through `:815`.
- Chapter 7.8 separately lists the indexed VALU formulas, including unsigned
  `M0[31:0]` for `V_MOVRELD_B32`, `V_MOVRELS_B32`, and `V_MOVRELSD_B32`, plus
  split `M0[9:0]` source and `M0[25:16]` destination indexes for
  `V_MOVRELSD_2_B32` and `V_SWAPREL_B32`, at `rdna3/README.md:2839` through
  `:2856`.

Rocjitsu evidence:

- RDNA3 `V_MOVRELD_B32` and `V_MOVRELS_B32` execute for in-range operands, but
  they compute `static_cast<int32_t>(wf.m0())`, throw `UnimplementedInst` when
  the indexed operand is negative or beyond `wf.vgpr_alloc().count`, and then
  form a normal `OPR_VGPR` operand at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/vop1.cpp:6778` through
  `:6790` and `:6908` through `:6920`; the VOP3 forms do the same at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/vop3.cpp:3548` through
  `:3560` and `:3625` through `:3637`.
- RDNA3 `V_MOVRELSD_B32` and `V_MOVRELSD_2_B32` decode, but both E32 execute
  paths throw `UnimplementedInst` at `rdna3/vop1.cpp:6980` through `:6983` and
  `:7017` through `:7019`; both VOP3 execute paths throw at
  `rdna3/vop3.cpp:3686` through `:3688` and `:3723` through `:3725`.
- RDNA3 `V_SWAPREL_B32` marks both explicit operands as source and destination
  operands in the constructor at `rdna3/vop1.cpp:10539` through `:10549`, but
  execution throws `UnimplementedInst` at `:10573` through `:10575`.

Impact:

Legal indexed VGPR moves and swaps can stop execution instead of applying the
manual's `VGPR0` source fallback, destination-NOP/all-discard behavior, and
separate source/destination indexing rules.

### RDNA3-RJ-022: Memory return-VGPR range nullification and PRT/DMASK return-window rules are not modeled

Manual evidence:

- Section 3.3.3 says memory/LDS/GDS reads and atomics with return must be
  nullified as if `EXEC` were zero when any destination VGPR is out of range,
  and that the test covers every VGPR that could be returned, such as `VDST` to
  `VDST+3` for `BUFFER_LOAD_B128`, at `rdna3/README.md:820` through `:826`.
- The same section says the destination check includes the extra PRT return
  VGPR whether the texture system returns it or not, says atomics with
  out-of-range destination VGPRs are issued with `EXEC=0`, says image loads and
  stores consider `DMASK` bits for out-of-bounds determination, and says `VDST`
  is checked only for LDS/GDS/memory atomics that actually return a value at
  `rdna3/README.md:825` through `:828`.

Rocjitsu evidence:

- Generated RDNA3 `BUFFER_LOAD_B128` records a 128-bit `VDATA` destination and
  then executes by setting `d->dst_reg_base`, `d->elem_size = 4`, and
  `d->num_elems = 4` before address calculation at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/mubuf.cpp:608` through
  `:636`; no pre-issue check nullifies the operation if `VDATA+3` is outside
  `wf.num_vgprs()`.
- Generated RDNA3 buffer atomics with return similarly set `d->dst_reg_base`
  from `inst_.vdata` and make return behavior depend on `GLC` at
  `rdna3/mubuf.cpp:1233` through `:1261`, while the DS atomic generator sets a
  returned `VDST` base for local-memory atomics at
  `lib/python/amdisa/codegen/_generator.py:4978` through `:4992`; neither path
  applies the manual's returned-VGPR range nullification before queuing work.
- The common vector memory state stores `exec_mask`, `lane_mask`,
  `dst_reg_base`, `elem_size`, and `num_elems` but has no return-VGPR range or
  PRT-return metadata at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:87` through `:112`.
- Common vector memory completion only computes the number of destination VGPRs
  from `num_elems * elem_size` and writes `d.dst_reg_base + i`; it zeroes lanes
  excluded by memory-address OOB handling, but does not nullify the operation
  for an out-of-range destination window at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:129` through `:195`.
- RDNA3 image instructions decode `DMASK`/`TFE` fields in machine structs and
  builders, but representative `image_load` execution is a minimal stub at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/mimg.cpp:20` through `:35`,
  so the manual's PRT extra-return and `DMASK`-driven range/OOB rules are not
  modeled by an image execution path.

Impact:

When a memory load or returning atomic targets a partially out-of-range VGPR
window, rocjitsu can still issue memory work and write any physical registers it
can reach instead of nullifying the whole operation with an empty `EXEC`.
Image/PRT cases are additionally under-modeled because the return window cannot
be derived from `DMASK`/`TFE` execution semantics today.

### RDNA3-RJ-023: `SH_MEM_CONFIG` alignment mode and atomic alignment `MEMVIOL` are not modeled

Manual evidence:

- Section 3.3.3 says VMEM texture alignment is controlled by
  `SH_MEM_CONFIG.alignment_mode`, and that the same setting affects LDS,
  flat/scratch/global operations at `rdna3/README.md:830` through `:834`.
- The section defines formatted-buffer alignment by data-format width and says
  atomics must be aligned to the data size or trigger `MEMVIOL` at
  `rdna3/README.md:836` through `:842`.

Rocjitsu evidence:

- Searches found no rocjitsu state or helper for `SH_MEM_CONFIG`,
  `alignment_mode`, or a memory-violation status path; the only `TRAPSTS`
  match in the implementation surface is semantic fingerprint naming, not an
  executable memory-violation update.
- RDNA3 scalar-memory address calculation does mask addresses down to a DWORD
  boundary at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/addr_calc.cpp:43`
  through `:50`, but the VMEM/flat/DS helpers build per-lane addresses without
  consulting an alignment-mode register: buffer address calculation handles
  descriptor bounds at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h:39`
  through `:132`, flat/global/scratch address calculation directly forms
  addresses at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/addr_calc.cpp:53`
  through `:123`, and DS address calculation directly adds the DS offset and
  LDS base at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:53`
  through `:76`.
- GPU memory byte/dword/qword helpers perform unaligned `memcpy` at the current
  address when the translated page contains the bytes, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/gpu_memory.h:227` through `:296`.
- Global and LDS atomic RMW paths iterate over `d.per_lane_addr[lane]` and
  perform the RMW without checking `ea % elem_size` or setting a MEMVIOL-like
  state at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:315`
  through `:380` and `:384` through `:420`.

Impact:

Misaligned VMEM/LDS/flat/global/scratch operations and atomics follow the host
byte-addressable memory model rather than the manual's alignment-mode and
atomic-MEMVIOL policy. This can make malformed shaders silently read/write data
where hardware would align, allow, or report a memory violation depending on
mode and opcode.

### RDNA3-RJ-024: LDS allocation granularity and per-workgroup bounds are not modeled accurately

Manual evidence:

- Section 3.3.4 says LDS allocations are 0-64 KiB per wave/workgroup, shared by
  all waves in the workgroup, allocated in 1024-byte blocks, and all LDS
  accesses are restricted to the allocated space at `rdna3/README.md:846`
  through `:847`.
- The same section describes the two 64 KiB LDS blocks, CU-mode same-side
  placement and no-cross/no-wrap rule, and WGP-mode placement or straddling
  behavior at `rdna3/README.md:848` through `:852`.

Rocjitsu evidence:

- Dispatch validation and placement round `group_segment_fixed_size` to 256
  bytes, not 1024 bytes, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:99` through
  `:101`, `:988` through `:999`,
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:243` through `:249`,
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.h:239` through `:245`, and
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/spi.h:203` through `:239`.
- WGP-mode uses a sibling-CU backing store whose size is the sum of the two CU
  LDS sizes at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/spi.h:270` through `:273`,
  and wavefronts carry only a placement-selected `lds_base`, not an allocated
  byte size, at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:167` through
  `:176` and `:512` through `:513`.
- DS address calculation adds the DS byte offset and `wf.lds_base()` directly at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:53`
  through `:76`; generated RDNA3 DS special forms do the same for scaled
  offsets at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/ds.cpp:535`
  through `:608`, `:1978` through `:2025`, and `:2590` through `:2653`.
- LDS read/write helpers check only the total backing vector size, returning
  zero or dropping writes when the backing is crossed, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/lds.h:31` through `:107` and `:136`
  through `:168`; the local-memory pipeline passes those backing-relative
  addresses straight through at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:506` through `:527`
  and `:590` through `:594`.

Impact:

A shader can access bytes outside its requested LDS allocation but still inside
the simulated CU/WGP LDS backing, and rocjitsu will read or write that storage
instead of applying the manual's per-allocation out-of-range behavior. The
256-byte allocation accounting can also admit or pack resident workgroups in a
layout that does not match the documented 1024-byte block allocation.

### RDNA3-RJ-025: LDS-specific alignment mode, `LDS_CONFIG` reporting, and MEMVIOL controls are not modeled

Manual evidence:

- Section 3.3.4.1 says DS load/store sizes may be byte-aligned only when
  alignment mode is unaligned; otherwise LDS forces alignment by zeroing low
  address bits, and 32-bit/64-bit atomics require 4-byte/8-byte alignment at
  `rdna3/README.md:856` through `:859`.
- The same section says LDS operations report `MEMVIOL` for out-of-range
  addresses when `LDS_CONFIG.ADDR_OUT_OF_RANGE_REPORTING` is set, report
  `MEMVIOL` for misaligned LDS accesses in STRICT or DWORD_STRICT mode, and use
  native LDS/GDS alignment masks for B8/B16/B32/B64/B96/B128 accesses at
  `rdna3/README.md:860` through `:890`.

Rocjitsu evidence:

- Searches found no rocjitsu state or helper for `LDS_CONFIG`,
  `ADDR_OUT_OF_RANGE_REPORTING`, STRICT/DWORD_STRICT alignment modes, or
  `MEMVIOL`; the implementation search surface only found an unrelated
  `out_of_range` exception string in liveness analysis.
- The shared DS address helper directly computes `addr + offset + wf.lds_base()`
  without an alignment mask at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:53`
  through `:76`, and generated RDNA3 DS special forms similarly add scaled
  offsets without consulting alignment state at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/ds.cpp:535` through `:608`
  and `:2590` through `:2653`.
- LDS vector loads/stores and single-value helpers return zero or drop writes
  only when the total backing vector is crossed at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/lds.h:31` through `:107` and `:136`
  through `:168`; they do not report a violation or distinguish alignment
  modes.
- LDS atomic RMW reads and writes through `Lds::read32`/`read64` and
  `write32`/`write64` without checking 4-byte or 8-byte alignment or recording
  `MEMVIOL` at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:383`
  through `:470`.

Impact:

Misaligned and out-of-range LDS accesses are treated only as host-side backing
array accesses. Rocjitsu can silently align differently, access misaligned bytes,
return zero, or drop writes where hardware behavior depends on LDS alignment
mode, `LDS_CONFIG.ADDR_OUT_OF_RANGE_REPORTING`, and `MEMVIOL` reporting.

### RDNA3-RJ-026: STATUS field layout, write permissions, and wave-creation state are incomplete

Manual evidence:

- Section 3.4.1 says STATUS fields are readable but not normally writable by the
  shader, with only selected fields writable in the trap handler, and that these
  bits are initialized at wave-creation time at `rdna3/README.md:912` through
  `:915`.
- The STATUS table includes export, barrier, halt, trap, validity, performance,
  conditional-debug, VGPR-release, LDS-parameter, GS-allocation, must-export,
  idle, and scratch-enable bits at `rdna3/README.md:920` through `:948`.

Rocjitsu evidence:

- The RDNA3 `StatusReg` bitfield models only `SCC`, priority, `PRIV`,
  `TRAP_EN`, `EXECZ`, `VCCZ`, `IN_TG`, `IN_BARRIER`, `HALT`, `TRAP`, `VALID`,
  `ECC_ERR`, and `ALLOW_REPLAY` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/isa.h:22` through `:61`;
  it omits manual STATUS fields such as `EXPORT_RDY`, `SKIP_EXPORT`,
  `PERF_EN`, `CDBG_USER`, `CDBG_SYS`, `FATAL_HALT`, `NO_VGPRS`,
  `LDS_PARAM_RDY`, `MUST_GS_ALLOC`, `MUST_EXPORT`, `IDLE`, and `SCRATCH_EN`.
- `Wavefront::read_scc()` and `write_scc()` update only bit 0 at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:414` through `:424`, and the
  raw STATUS accessors simply cast or assign the `StatusReg` at
  `wavefront.h:608` through `:617`.
- `Wavefront::reset()` explicitly does not change the status register while
  clearing other dynamic dispatch state at `wavefront.h:459` through `:491`,
  and `ComputeUnitCore::dispatch_wf()` initializes PC, register allocations,
  `EXEC`, `VCC`, `M0`, apertures, and run state, but does not initialize the
  manual STATUS bits at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:136` through `:150`.
- The existing `RDNA3-RJ-011` and `RDNA3-RJ-012` gaps cover the HWREG selector
  bug and stale raw `STATUS.EXECZ`/`STATUS.VCCZ`; this gap covers the remaining
  STATUS field inventory, permissions, and initialization semantics.

Impact:

Shaders or debug paths that read STATUS can observe a partial, stale, or
non-architectural status word. Export waits, skipped exports, LDS parameter
readiness, wave validity/idle state, VGPR release state, scratch enable, trap
write restrictions, and wave-creation initialization are not represented by the
RDNA3 STATUS model.

### RDNA3-RJ-027: MODE floating-point and exception controls are not applied

Manual evidence:

- Section 3.4.2 says MODE `FP_ROUND` controls VALU floating-point rounding,
  `FP_DENORM` controls denormal flushing for VALU, LDS, and VMEM atomics,
  `DX10_CLAMP` controls NaN clamp behavior and exception suppression with
  `CLAMP`, and `IEEE` changes min/max NaN and signed-zero behavior at
  `rdna3/README.md:952` through `:963`.
- The same MODE table says `TRAP_AFTER_INST` forces trap-handler entry after
  each instruction when traps are enabled, `EXCP_EN` controls exception enables,
  and `FP16_OVFL` clamps overflowed FP16 VALU results to max finite while
  preserving true infinities at `rdna3/README.md:964` through `:973`.

Rocjitsu evidence:

- `Wavefront` stores raw MODE and synchronizes only the VGPR-MSB subfield used
  by register indexing at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:103`
  through `:128` and `:535` through `:537`.
- RDNA3 `S_ROUND_MODE` and `S_DENORM_MODE` decode and dispatch to shared helper
  functions, but those helpers are empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1643`
  through `:1645` and `:2422` through `:2425`.
- Representative RDNA3 floating-point helpers use host `std::fmax`/`std::fmin`
  or SIMD equivalents without consulting `wf.mode_raw()`, for example
  `V_MAX_F32` at `execute_shared.h:13877` through `:13905` and `V_MIN_F32` at
  `:15184` through `:15212`.
- Searches found no RDNA3 runtime state or execution path for
  `TRAP_AFTER_INST`, `EXCP_EN`, `DX10_CLAMP`, or `FP16_OVFL` outside manual/test
  names and comments, and the memory/atomic paths audited earlier do not consult
  MODE denormal fields.

Impact:

Changing MODE through the documented instructions or HWREG paths does not alter
the RDNA3 floating-point, denormal, trap-after-instruction, exception-enable, or
FP16-overflow behavior. Programs that depend on non-default rounding, denormal
handling, DX10/IEEE min/max differences, or FP16 overflow clamping can match the
host default instead of the architectural MODE state.

### RDNA3-RJ-028: RDNA3 scalar resolution retains non-RDNA3 FLAT_SCRATCH aliases

Manual/XML evidence:

- RDNA3 Chapter 3.1 lists `S0-S105` as scalar general-purpose registers and
  `FLAT_SCRATCH` as separate wave state at `rdna3/README.md:624` through
  `:635`.
- RDNA3 scalar and vector operand tables identify source values `0-105` as
  `SGPR0 - SGPR105`, while VOP3/VOP3P source values `209-232` are reserved, at
  `rdna3/README.md:5418` through `:5425`,
  `rdna3/README.md:6158` through `:6161`, and `rdna3/README.md:6505` through
  `:6508`.
- RDNA3 XML likewise lists `s102` and `s103` as ordinary scalar predefined
  values in `OPR_SDST` and `OPR_SSRC` at `amdgpu_isa_rdna3.xml:175849` through
  `:175857` and `amdgpu_isa_rdna3.xml:183700` through `:183708`.
- CDNA3/CDNA4 are different: their manual operand tables name selector
  `102/103` as `FLAT_SCRATCH_LO/HI` at `cdna3/README.md:1267` through `:1273`
  and `cdna4/README.md:1295` through `:1299`, so the alias is not a
  cross-architecture rule.

Rocjitsu evidence:

- The generated RDNA3 operand bounds still classify scalar source and
  destination values through 105 as SGPRs at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/operand_types.h:99`
  through `:110` and `:136` through `:168`, and display/def-use code reports
  those values as SGPRs at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/operand.cpp:248` through
  `:251` and `:613` through `:618`.
- Execution special-cases source selector `102` and `103` before the ordinary
  SGPR range, returning `wf.scratch_base()` low/high halves at
  `rdna3/operand.cpp:703` through `:709`; the 64-bit source path returns the
  whole scratch base for selector `102` at `:810` through `:818`.
- The same resolver also accepts reserved source selector values `230` and
  `231` as scratch-base low/high halves at `rdna3/operand.cpp:728` through
  `:731` and `:839` through `:840`, even though the generated RDNA3 source enum
  has no flat-scratch source names in that range.
- Destination writes to selector `102` and `103` mutate `wf.scratch_base()`
  instead of SGPR102/SGPR103 at `rdna3/operand.cpp:870` through `:883`, and the
  64-bit destination path replaces the full scratch base for selector `102` at
  `:914` through `:924`.

Impact:

RDNA3 instructions that legally read or write SGPR102/SGPR103 can observe or
clobber rocjitsu's per-wave scratch base instead of the scalar register file.
Disassembly and dependency metadata can still report an SGPR operand while the
executor accesses scratch state, and invalid source encodings `230/231` can be
given non-reserved behavior.

### RDNA3-RJ-029: FLAT_SCRATCH launch alignment and access policy are incomplete

Manual evidence:

- Section 3.4.7 says `FLAT_SCRATCH` is initialized by wave-launch hardware for
  waves with scratch allocation, returns zero when no scratch is allocated, is
  read-only outside the trap handler, and is a 256-byte-aligned byte address at
  `rdna3/README.md:1042` through `:1052`.
- Section 3.4 and 3.4.8 also state that `FLAT_SCRATCH_LO/HI` are read-only
  HWREG targets writable only while in a trap handler at
  `rdna3/README.md:892` through `:910` and `rdna3/README.md:1080` through
  `:1088`.

Rocjitsu evidence:

- `Wavefront` stores a raw scratch base and exposes an unchecked
  `set_scratch_base()` setter at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:253` through `:259`; reset
  zeroes `scratch_base_` and `scratch_lane_size_` at `wavefront.h:474` through
  `:483`.
- Dispatch initializes scratch only when `private_segment_fixed_size > 0`, then
  derives each wave base as
  `scratch_pool + global_wave_idx * private_segment_fixed_size * wf_size` and
  stores it without any 256-byte alignment check or rounding at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:329` through
  `:348`.
- Scratch and flat-private address helpers consume the stored base directly at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/addr_calc.cpp:68` through
  `:84` and `:95` through `:112`.
- Existing `RDNA3-RJ-011` covers the incomplete RDNA3 HWREG get/set map for
  `FLAT_SCRATCH_LO/HI`; `RDNA3-RJ-028` covers the scalar-selector path that can
  mutate the scratch base outside the documented trap-only write route.

Impact:

Rocjitsu can hand later waves a FLAT_SCRATCH value that is not 256-byte aligned
when the backing address or per-wave scratch pitch is not aligned, and the
access-control contract is split across incomplete HWREG handling plus generic
scalar-selector writes. Kernels or probes that depend on the architectural
alignment or read-only/trap-only behavior can diverge from RDNA3 hardware.

### RDNA3-RJ-030: Time-counter accesses are incomplete and synchronous

Manual/XML evidence:

- RDNA3 Section 3.4.10 says `SHADER_CYCLES` is a 20-bit graphics-core-clock
  counter read with `S_GETREG`, not synchronized across SIMDs and intended only
  for within-wave deltas, at `rdna3/README.md:1163` through `:1170`.
- The same section says `REALTIME` is a 64-bit fixed-frequency clock read with
  `S_SENDMSG_RTN_B64` followed by `S_WAITCNT LGKMcnt == 0` at
  `rdna3/README.md:1172` through `:1179`.
- XML exposes `HW_REG_SHADER_CYCLES` and `MSG_RTN_GET_REALTIME` at
  `amdgpu_isa_rdna3.xml:174437` through `:174440` and
  `amdgpu_isa_rdna3.xml:175263` through `:175265`.

Rocjitsu evidence:

- RDNA3 `SGetregB32Sopk::execute_impl` handles only HWREG IDs 1, 4, 5, 6, and
  7 before warning and returning zero for other IDs at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/sopk.cpp:287` through
  `:317`; `HW_REG_SHADER_CYCLES` ID 29 therefore falls into the default path.
  This is the time-specific consequence of the broader HWREG map gap tracked in
  `RDNA3-RJ-011`.
- `S_SENDMSG_RTN_B32` and `S_SENDMSG_RTN_B64` return `engine->global_time()` for
  message `0x83` and write the scalar destination immediately at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/sop1.cpp:1256` through
  `:1281` and `:1299` through `:1324`; the shared generated template has the
  same synchronous write behavior at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2431`
  through `:2487`.
- `SimulationEngine::global_time()` is the latest processed simulation tick at
  `lib/simdojo/include/simdojo/sim/simulation.h:268` through `:270`. The
  send-message path does not model a 100 MHz REFCLK domain, 64-bit idle-running
  realtime source, or an outstanding LGKM dependency that must be retired by
  `S_WAITCNT`.

Impact:

`S_GETREG(SHADER_CYCLES)` returns zero rather than a 20-bit shader-cycle value,
and `S_SENDMSG_RTN_B64 REALTIME` produces a synchronous simulator tick instead
of an LGKM-counted return message in the architectural realtime clock domain.
Timing probes and code that relies on the documented wait dependency can
observe non-hardware behavior.

### RDNA3-RJ-031: Optional `TG_SIZE` system SGPR launch payload is not initialized

Manual evidence:

- Section 3.5.3.4 says compute launch appends a packed system SGPR containing
  `{first_wave, 6'h00, wave_id_in_group[4:0], 2'h0,
  ordered_append_term[11:0], work_group_size_in_waves[5:0]}` when
  `COMPUTE_PGM_RSRC2.tg_size_en` is set at `rdna3/README.md:1280` through
  `:1290`.

Rocjitsu evidence:

- The HSA descriptor header defines
  `COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO` at
  `lib/rocjitsu/external_headers/hsa_headers/hsa/AMDHSAKernelDescriptor.h:126`
  through `:148`.
- `DispatchEntry` carries user-SGPR count, workgroup-id enable flags, and
  wave/workgroup dimensions, but no workgroup-info enable or stored `TG_SIZE`
  payload field at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/dispatch_entry.h:45` through `:85`.
- `CommandProcessor::process_aql_packet()` reads user-SGPR count and the
  workgroup-id enable bits, but does not read
  `COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:930` through
  `:1045`.
- `CommandProcessor::init_wavefront_regs()` writes AMDHSA user SGPRs, optional
  kernarg preload values, and enabled `workgroup_id_{x,y,z}` SGPRs, then moves
  on to RDNA4/gfx1250 TTMP payload and VGPR workitem IDs; there is no `TG_SIZE`
  system SGPR write at `command_processor.cpp:174` through `:285`.
- Static searches for `ENABLE_SGPR_WORKGROUP_INFO`, `TG_SIZE`, `tg_size`,
  `first_wave`, `wave_id_in_group`, `ordered_append`, and
  `work_group_size_in_waves` found no rocjitsu launch payload implementation
  outside the descriptor constant.

Impact:

RDNA3 compute kernels that request the `TG_SIZE` system SGPR receive whatever
stale or zero SGPR value happens to be in the next packed SGPR slot instead of
the documented wave/workgroup-size payload.

### RDNA3-RJ-032: 2D/3D `work_group_id0` is linearized when only the X SGPR is enabled

Manual evidence:

- Section 3.5.3 says enabled SGPR launch values are packed consecutively with
  no alignment gaps at `rdna3/README.md:1201` through `:1203`.
- The compute table says `COMPUTE_PGM_RSRC2.tgid_x_en`,
  `COMPUTE_PGM_RSRC2.tgid_y_en`, and `COMPUTE_PGM_RSRC2.tgid_z_en` append
  `work_group_id0[31:0]`, `work_group_id1[31:0]`, and
  `work_group_id2[31:0]` respectively at `rdna3/README.md:1284` through
  `:1290`.

Rocjitsu evidence:

- `CommandProcessor::process_aql_packet()` keeps multidimensional dispatch
  decomposition for `num_dims >= 2`, setting `dp.grid_wgs_x` to the X
  workgroup count and noting that only 1D dispatches flatten the grid into
  `workgroup_id_x` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:1015` through
  `:1023`.
- `CommandProcessor::init_wavefront_regs()` computes `grid_wg_id_x` from
  `global_wg_id % grid_wgs_x`, but writes a linear `global_wg_id` to the X
  SGPR whenever Y and Z SGPRs are disabled:
  `uint32_t wg_id_x = (pkt.enable_wg_id_y || pkt.enable_wg_id_z) ?
  grid_wg_id_x : global_wg_id;` at `command_processor.cpp:265` through `:284`.

Impact:

For a 2D or 3D dispatch where the kernel requests only `tgid_x_en`, rocjitsu
writes the linear workgroup ordinal instead of the X coordinate. For example,
with `grid_wgs_x == 3`, `global_wg_id == 4` should initialize
`work_group_id0` as `1`, but rocjitsu writes `4`.

### RDNA3-RJ-033: Graphics-stage VGPR launch payloads and PS input routing are not modeled

Manual evidence:

- Section 3.5.4 maps HS(+LS) and GS(+ES) combined-stage initial VGPR payloads
  into `VGPR0` through `VGPR8`, including patch/control-point IDs, LS vertex
  data, GS offsets, primitive/payload data, RT index/edge flags/instance ID,
  ES parameter data, and GS Fast Launch variants at `rdna3/README.md:1294`
  through `:1306`.
- Section 3.5.4.1 says pixel shader VGPR loading uses a CAM from VS outputs to
  PS inputs and is controlled by `SPI_PS_INPUT_ENA` and
  `SPI_PS_INPUT_ADDR`; it defines full-load destinations, packed destinations
  when disabled terms compact toward `VGPR0`, and skipped destinations when
  `ADDR` bits are set without corresponding `ENA` bits at
  `rdna3/README.md:1308` through `:1417`.

Rocjitsu evidence:

- `DispatchEntry` is compute/AQL shaped, with kernel dispatch IDs, grid sizes,
  workgroup sizes, and workgroup ID controls, but no PS/GS/HS shader-stage
  launch payload fields or pixel-input register state at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/dispatch_entry.h:43` through `:87`.
- `CommandProcessor::init_wavefront_regs()` initializes AMDHSA compute SGPRs,
  workgroup IDs, workitem-ID VGPRs, and scratch state. Its only initial VGPR
  writes are local workitem IDs derived from the AQL packet at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:300` through
  `:321`.
- Static searches for `SPI_PS_INPUT`, `PS_INPUT`, `PERSP_`, `LINEAR_`,
  `POS_X`, `FRONT_FACE`, `ANCILLARY`, `SAMPLE_COVERAGE`, `FAST_LAUNCH`,
  `barycentric`, `primitive ID`, and `patch ID` found no graphics-stage VGPR
  launch implementation in rocjitsu's VM/ISA generator paths.
- The broader graphics-launch absence is already recorded as `RDNA3-RJ-009`;
  this finding records the concrete Chapter 3.5.4 VGPR payload and routing
  consequences.

Impact:

RDNA3 graphics shaders that consume hardware-initialized HS/GS/PS VGPRs cannot
be launched with the documented initial state. Pixel shaders in particular
cannot observe the PS input CAM, `ENA`/`ADDR` destination packing, or ancillary
packed fields described by the manual.

### RDNA3-RJ-034: Pixel-shader LDS launch preload is not modeled

Manual evidence:

- Section 3.5.5 says only pixel shader waves have LDS pre-initialized before
  launch, with LDS preloaded with vertex parameter data that can be
  interpolated using barycentrics I/J to compute per-pixel parameters, at
  `rdna3/README.md:1419` through `:1421`.

Rocjitsu evidence:

- `DispatchEntry` is compute/AQL shaped, with kernel dispatch IDs, grid sizes,
  workgroup sizes, and workgroup ID controls, but no PS launch state, graphics
  parameter payload, or pixel-input LDS preload fields at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/dispatch_entry.h:43` through `:87`.
- `CommandProcessor::dispatch_workgroups()` assigns an LDS base and backing
  store before wave register initialization at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:693` through
  `:720`, and `process_aql_packet()` validates compute LDS capacity from
  `group_segment_fixed_size` at `command_processor.cpp:976` through `:999`.
  Neither path populates LDS with pixel vertex parameter data before launch.
- `Lds` constructs a zeroed byte backing and exposes raw LDS read/write APIs at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/lds.h:17` through `:27` and
  `lds.h:31` through `:107`; it has no launch-time vertex-parameter preload
  helper.
- Static searches for `vertex parameter`, `barycentric`, `PS_INPUT`,
  `SPI_PS_INPUT`, and related pixel-input terms found no graphics/pixel-shader
  LDS preload implementation in rocjitsu's VM/ISA paths. Searches do find
  generated LDSDIR instruction shells, but `execute_lds_param_load_ldsdir` and
  `execute_lds_direct_load_ldsdir` are empty shared stubs at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:337`
  through `:343`.

Impact:

Pixel shaders that rely on hardware-preloaded LDS parameter data cannot observe
the documented initial LDS contents, and LDSDIR parameter loads have no backing
launch data to consume.

### RDNA3-RJ-035: Reserved common source selectors still execute with legacy meanings

Manual evidence:

- Chapter 4.1 marks source selector values `209-232`, `239`, `249`, `251`,
  `252`, and `254` as reserved, while `250` is `DPP16`, at
  `rdna3/README.md:1456` through `:1493`.

Rocjitsu evidence:

- RDNA3 scalar source resolution assigns meanings to manual-reserved values
  `230`, `231`, `249`, `251`, and `252`, and treats selector `250` as `NULL`,
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/operand.cpp:728`
  through `:768`.
- `can_resolve_src_scalar()` reports these values as resolvable by including
  `(ev >= 240 && ev <= 253)` and the non-RDNA3 `230`/`231` aliases at
  `operand.cpp:800` through `:807`.
- `RDNA3-RJ-028` already records the `230`/`231` FLAT_SCRATCH alias issue; this
  entry records the rest of the Chapter 4 common-source table drift.

Impact:

Manual-invalid RDNA3 source encodings can execute with legacy scratch, null,
VCCZ, EXECZ, or POPS-related meanings instead of being rejected or treated as
reserved/noncanonical.

### RDNA3-RJ-036: Signed 64-bit literal operands are zero-extended

Manual evidence:

- Chapter 4.1 says a 32-bit literal used in a 64-bit signed integer operation
  is sign-extended to 64 bits, while unsigned 64-bit and binary 64-bit
  operations zero-extend it, at `rdna3/README.md:1452` through `:1454`.

Rocjitsu evidence:

- `read_immediate64()` zero-extends every `OPR_SIMM32` value at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/operand.cpp:961` through
  `:964`, and both lane and scalar 64-bit immediate reads call it at
  `operand.cpp:1094` through `:1098` and `:1111` through `:1116`.
- RDNA3 `V_MAD_I64_I32` has a signed 64-bit `SRC2` operand and literal
  encodings for that operand in XML at `amdgpu_isa_rdna3.xml:106614` through
  `:106650` and `:106765` through `:106799`.
- The generated constructor builds literal `SRC2` for `V_MAD_I64_I32` as
  `Operand(64, OperandType::OPR_SIMM32, ...)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/vop3.cpp:37123` through
  `:37126`, and execution reads it with `read_lane64(inst.src2, lane)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:13051`
  through `:13055`.

Impact:

For a signed 64-bit literal such as `0xffffffff`, rocjitsu supplies
`0x00000000ffffffff` instead of the architectural `0xffffffffffffffff`.

### RDNA3-RJ-037: GFX11 cache flag mapping treats SLC/DLC as CDNA-style scope/NT bits

Manual evidence:

- Chapter 4.1.1 defines RDNA3 SLC/GLC/DLC cache behavior at
  `rdna3/README.md:1495` through `:1565`: GLC is the load-scope bit
  (`0=CU`, `1=DEVICE`), stores and atomics are device-scope, SLC is the
  graphics-client temporal hint, DLC is the Infinity Cache temporal hint, and
  atomic GLC selects whether the pre-operation value is returned.

Rocjitsu evidence:

- `shared/gfx11_cache_flags.h` documents RDNA3/RDNA3.5 as `GLC`/`SLC` 2-bit
  scope plus `DLC` non-temporal, with `scope = (SLC << 1) | GLC`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/gfx11_cache_flags.h:7`
  through `:20`.
- `mtype_from_flags_gfx11()` forwards that scope and `DLC` to the CDNA-style
  `mtype_from_scope_nt()` helper at `gfx11_cache_flags.h:32` through `:43`;
  the shared helper maps scope values `2+` to `Mtype::UC` and `nt` to
  `Mtype::NT` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/gfx940_cache_flags.h:29`
  through `:42`.
- Generated RDNA3 memory instructions use this helper, for example SMEM
  `S_LOAD_B32` at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/smem.cpp:45`
  through `:55`, MUBUF `BUFFER_LOAD_U8` at `rdna3/mubuf.cpp:407` through
  `:416`, and FLAT `FLAT_LOAD_U8` at `rdna3/flat.cpp:50` through `:59`.
- The vector-memory pipeline passes only `d.mtype` and `d.non_temporal` into the
  vector L1 at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:488`
  through `:494`, and the vector L1 bypasses/invalidate-paths on `Mtype::UC`,
  the separate `non_temporal` flag, or forced bypass at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/l1_vector_cache.cpp:92` through `:100`.

Impact:

For a load with `GLC=0,SLC=1,DLC=0`, the manual table keeps CU scope and applies
stream/non-temporal hints to graphics caches, but rocjitsu computes scope `2`
and treats the access as system/uncached. Conversely, DLC is collapsed into the
shared `Mtype::NT` model rather than the manual's MALL-specific hint.

### RDNA3-RJ-038: Reserved padding fields are accepted silently

Manual evidence:

- The Chapter 4 introduction says unused fields that are not SGPR
  source/destination fields are typically set to zero at `rdna3/README.md:1440`.

Rocjitsu evidence:

- RDNA3 machine-instruction structs expose padding fields such as `pad_15_17`,
  `pad_53_56`, `pad_23`, and `pad_59_60` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/machine_insts.h:33`
  through `:37` and `:84` through `:104`.
- Encoding constructors copy the bitfield struct and derive opcode/encoding
  without validating padding, for example `Smem::Smem` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/encodings.cpp:102`
  through `:108`.

Impact:

Noncanonical or reserved encodings with nonzero padding are accepted silently.
The manual wording is `typically`, so this is a canonicalization/legality gap
rather than a proven execution-semantics mismatch.

### RDNA3-RJ-039: `S_CODE_END` executes as a no-op

Manual evidence:

- Section 5.1 says `S_CODE_END` is treated as an illegal instruction and is used
  to pad past the end of shaders at `rdna3/README.md:1612` through `:1613`.

Rocjitsu evidence:

- RDNA3 `SCodeEndSopp` decodes and dispatches through
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/sopp.cpp:196` through
  `:203`.
- The shared `execute_s_code_end_sopp` helper is empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1508`.

Impact:

Falling through past shader code continues through padding instead of raising
the manual's illegal-instruction behavior.

### RDNA3-RJ-040: Wave-control sleep, wakeup, kill, halt, and priority effects are incomplete

Manual evidence:

- Section 5.1 says `S_SETKILL` sets the kill bit and causes immediate program
  end, `S_SETHALT` sets or clears `HALT`/fatal-halt bits subject to trap
  privilege, `S_SLEEP` sleeps for approximately `64 * SIMM16[6:0]` clocks,
  `S_WAKEUP` wakes same-work-group waves from sleep, and `S_SETPRIO` sets
  `USER_PRIO` and defines the priority formula at `rdna3/README.md:1585`
  through `:1599`.

Rocjitsu evidence:

- RDNA3 generated `S_SETKILL`, `S_SETHALT`, `S_SLEEP`, `S_WAKEUP`, and
  `S_SETPRIO` dispatch through `rdna3/sopp.cpp:30` through `:62` and
  `rdna3/sopp.cpp:442` through `:459`.
- The shared helpers for `S_SETKILL`, `S_SETHALT`, `S_SETPRIO`, and `S_WAKEUP`
  are empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2527`
  through `:2533` and `execute_shared.h:2690`.
- `execute_s_sleep_sopp` only requests a functional yield at
  `execute_shared.h:2555` through `:2558`; it does not track sleep duration or
  the `S_WAKEUP` interaction.

Impact:

Programs that use debug kill/halt, priority changes, or inter-wave sleep/wakeup
behavior continue without the manual's wave-state or scheduling effects.

### RDNA3-RJ-041: Instruction clauses are not modeled

Manual evidence:

- Section 5.1 defines `S_CLAUSE` length and break fields at
  `rdna3/README.md:1603` through `:1606`.
- Section 5.2 defines same-type uninterrupted clause execution, legal and
  illegal clause members, clause-internal instructions, sub-clause break rules,
  `EXEC==0` start/mid-clause behavior, skipped-first-instruction behavior, and
  clause-break causes at `rdna3/README.md:1619` through `:1687`.

Rocjitsu evidence:

- RDNA3 `SClauseSopp` decodes `OPR_CLAUSE` and dispatches through
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/sopp.cpp:77` through `:85`.
- The shared `execute_s_clause_sopp` helper is empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1032`.
- `WfState` has `HALTED`, `RUNNING`, `WAITCNT`, `BARRIER`, and `ENDING`, but no
  active-clause state, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:31` through `:38`.
- `ComputeUnitCore::step()` issues one instruction from each running wavefront
  in sequence at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:442`
  through `:449`; no uninterrupted same-type clause arbitration or membership
  validation is visible.

Impact:

Rocjitsu cannot model whether a clause starts, which instruction type it binds
to, whether a later instruction remains legal in the clause, or when hardware
would break the clause.

### RDNA3-RJ-042: Send-message execution and `LGKMcnt` behavior are incomplete

Manual evidence:

- Section 5.1 says `S_SENDMSG` sends an upstream message without an enforced
  wait before it, `S_SENDMSG_RTN_B32/B64` return data through `LGKMcnt`, and
  `S_SENDMSGHALT` sends a message and then halts at `rdna3/README.md:1614`
  through `:1616`.
- Section 5.3 says `S_SENDMSG` and `S_SENDMSG_RTN` use `M0` payloads, return
  messages increment `LGKMcnt` by 2 and decrement once on send plus once on
  data return, unlisted message codes are reserved/illegal, and the listed
  messages include interrupt, tessfactor, VGPR deallocation, GS allocation,
  doorbell, draw/dispatch ID, TMA, realtime, save-wave, TBA, and illegal return
  at `rdna3/README.md:1688` through `:1727`.

Rocjitsu evidence:

- RDNA3 `S_SENDMSG` and `S_SENDMSGHALT` dispatch through
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/sopp.cpp:462` through
  `:484`, but the shared helpers are empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2427`
  through `:2428` and `:2491` through `:2492`.
- RDNA3 `S_SENDMSG_RTN_B32/B64` decode `OPR_SENDMSG` operands at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/sop1.cpp:1241` through
  `:1297`, but execution only synthesizes a value for message `0x83` and
  otherwise writes zero at `rdna3/sop1.cpp:1256` through `:1281` and `:1299`
  through `:1324`.
- The return-message path writes the scalar destination immediately and does not
  increment or retire `LGKMcnt`; the broader wait-counter data structure has
  `LGKMCNT`/`KMCNT` support at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wait_counters.h:76` through `:99`.

Impact:

Message side effects, `M0` payload use, returned values other than realtime,
reserved/illegal message handling, send-and-halt behavior, and return-message
wait semantics do not match the manual.

### RDNA3-RJ-043: `S_WAIT_EVENT` executes as a no-op

Manual evidence:

- Section 5.6 says `S_WAIT_EVENT` waits for an event before proceeding,
  `SIMM16[0] == 0` waits for export-ready, other bits are reserved, and
  exceptions wait for this to complete before processing at
  `rdna3/README.md:1779`.

Rocjitsu evidence:

- RDNA3 `SWaitEventSopp` sets `flags_ |= WAITCNT` but its execute body only
  discards the wavefront reference at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/sopp.cpp:144` through
  `:154`.
- The shared `execute_s_wait_event_sopp` helper is also empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2678`
  through `:2680`.

Impact:

Code that relies on export-ready/event waiting or exception ordering around
`S_WAIT_EVENT` continues immediately.

### RDNA3-RJ-044: FLAT instructions do not model dual `LGKMcnt` plus vector-count participation

Manual evidence:

- Section 5.6 says FLAT instructions use both `LGKMcnt` and either `VMcnt` or
  `VScnt` at `rdna3/README.md:1792` through `:1797`.
- Chapter 11 repeats that FLAT instructions execute as both LDS and global
  instructions, increment both `VMcnt` or `VScnt` and `LGKMcnt`, and are not
  done until both have decremented at `rdna3/README.md:4326`.

Rocjitsu evidence:

- `VectorMemState` stores a single `wait_counter_type` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:85` through `:103`, and
  `MemoryPipeline::issue()` increments only that selected counter at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.h:62` through `:87`.
- Representative RDNA3 FLAT loads set only `WaitCounterType::LOADCNT` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/flat.cpp:50` through `:59`,
  and representative FLAT stores set only `WaitCounterType::STORECNT` at
  `rdna3/flat.cpp:348` through `:356`.
- DS/LDS instructions can use `WaitCounterType::DSCNT`, which also contributes
  to `LGKMcnt`, but the FLAT path does not hold an additional LDS/LGKM counter
  alongside the vector load/store counter.

Impact:

`S_WAITCNT LGKMcnt==0` can miss outstanding FLAT work in rocjitsu, and
mixed-wait tests cannot observe the documented dual-counter completion rule.

### RDNA3-RJ-045: `S_SETREG_IMM32_B32` hides its literal source operand from metadata

Manual/XML evidence:

- Chapter 6.2 says SALU literals are normally unavailable to SOPK, with
  `S_SETREG_IMM32_B32` as the exception at `rdna3/README.md:1949`.
- Chapter 6.8 says `S_SETREG_IMM32_B32` gets 32-bit data from a literal
  constant and is therefore a 64-bit instruction at `rdna3/README.md:2080`.
- The detailed instruction definition says this instruction requires a 32-bit
  literal at `rdna3/README.md:8043`.
- XML models the instruction with an `OPR_HWREG` selector and an explicit
  `LITERAL` / `OPR_SIMM32` source operand at `amdgpu_isa_rdna3.xml:56070`
  through `:56090`.

Rocjitsu evidence:

- RDNA3 `Sopk::hasImpliedLiteral()` returns true for opcode 19 and reads the
  extension word into `literal_` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/encodings.cpp:74` through
  `:82`.
- `SSetregImm32B32Sopk` constructs only the `OPR_HWREG` operand, stores it in
  `dst_operands_`, and sets `num_src_ = 0` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/sopk.cpp:358` through
  `:365`.
- Its `implicit_uses()` hook expands the HWREG selector operand, not the
  literal extension word, at `rdna3/sopk.cpp:367` through `:371`.
- Execution uses `literal_` as the source data at `rdna3/sopk.cpp:373` through
  `:393`, so the execution path sees the literal even though operand metadata
  does not.
- The common disassembler prints only declared destination and source operands
  at `lib/rocjitsu/src/rocjitsu/isa/instruction.h:239` through `:252`.

Impact:

Rocjitsu can execute `s_setreg_imm32_b32`, but operand-based consumers and
disassembly do not see the 32-bit input required by the manual and XML. This
can hide the literal in metadata-driven tests or analyses even though the
runtime state mutation uses the extension word.

### RDNA3-RJ-046: WREXEC accepts forbidden `EXEC` explicit destinations

Manual/XML evidence:

- Chapter 6.7 says `S_AND_NOT{0,1}_WREXEC_B{32,64}` writes both `D` and `EXEC`,
  but `D cannot be EXEC` at `rdna3/README.md:2067`.
- XML encodes the explicit WREXEC destination as `OPR_SREG`, for example
  `S_AND_NOT0_WREXEC_B32` at `amdgpu_isa_rdna3.xml:45256` through `:45288`.
- `OPR_SREG` includes SGPRs, TTMPs, VCC, and `NULL`, but not `M0` or `EXEC`,
  at `amdgpu_isa_rdna3.xml:181440` through `:181475`; generated RDNA3
  `OpSelSreg` constants mirror that value set at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/operand_types.h:192`
  through `:201`.

Rocjitsu evidence:

- Generated WREXEC constructors build `sdst` as `OperandType::OPR_SREG` from
  the raw `SDST` field but do not validate the raw value against the `OPR_SREG`
  selector set, for example `rdna3/sop1.cpp:956` through `:1028`.
- The shared WREXEC helpers write the explicit destination and then update
  `EXEC` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:510`
  through `:529` and `:571` through `:590`.
- RDNA3 scalar write resolution accepts raw destination selectors `126` and
  `127` as `EXEC_LO`/`EXEC_HI` for 32-bit writes, and raw selector `126` as
  full `EXEC` for 64-bit writes, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/operand.cpp:900` through
  `:908` and `:914` through `:944`.

Impact:

Invalid WREXEC encodings with `SDST=exec_lo`, `exec_hi`, or 64-bit `exec` can
execute and mutate `EXEC` through the explicit destination path before the
instruction's implicit EXEC write. Hardware should reject or otherwise not
treat those raw selectors as legal explicit destinations for this instruction
class.

### RDNA3-RJ-047: SALU implicit SCC operands are missing from generated def-use metadata

Manual/XML evidence:

- Chapter 6.3 says most SALU instructions write SCC, signed arithmetic uses
  overflow, bit/logical operations use result-nonzero, and extended arithmetic
  uses SCC as carry/borrow input at `rdna3/README.md:1957` through `:1966`.
- Chapter 6.5 says scalar conditional moves use SCC and do not write it at
  `rdna3/README.md:2004` through `:2012`.
- Chapter 6.6 says SOPC comparisons and bit compares set SCC at
  `rdna3/README.md:2016` through `:2025`.
- XML exposes these SCC dependencies structurally. For example, `S_ADD_I32`
  has an implicit SCC output at `amdgpu_isa_rdna3.xml:46399` through `:46430`,
  `S_CMOVK_I32` has an implicit SCC input at `:55354` through `:55382`, and
  `S_BITCMP0_B32` has an implicit SCC output at `:54594` through `:54623`.

Rocjitsu evidence:

- `DefUseChain` collects explicit operands and then calls `implicit_defs()` and
  `implicit_uses()` at
  `lib/rocjitsu/src/rocjitsu/analysis/def_use_chain.cpp:28` through `:44`.
- The base `Instruction` hooks are empty at
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:219` through `:222`.
- Representative SCC producer/consumer classes do not override those hooks:
  `SAddI32Sop2` is declared without SCC hooks at `rdna3/sop2.h:35` through
  `:43`, and its constructor adds only explicit `SDST`, `SSRC0`, and `SSRC1`
  operands at `rdna3/sop2.cpp:64` through `:82`.
- `SCselectB32Sop2` / `SCselectB64Sop2` add only explicit operands at
  `rdna3/sop2.cpp:1089` through `:1150`, while execution reads SCC at
  `shared/execute_shared.h:1511` through `:1524`.
- SOPC bit-compare classes similarly add only explicit source operands, for
  example `SBitcmp0B32Sopc` at `rdna3/sopc.cpp:296` through `:315`.

Impact:

Runtime execution can still be correct, but operand-based analyses do not see
SCC as a dependency or clobber for ordinary SALU SCC producers and consumers.
That can hide ordering hazards in def-use based transforms, schedulers, or
metadata-driven tests.

### RDNA3-RJ-048: Generic VOP3-family modifier fields are hidden in disassembly

Manual/XML evidence:

- Chapter 7.1 defines VOP3 `ABS`, `OPSEL`, `CLAMP`, `OMOD`, and `NEG` fields at
  `rdna3/README.md:2169` through `:2177`.
- Chapter 7.5 defines VOP3P `NEG`, `NEG_HI`, `OPSEL`, `OPSEL_HI`, and `CLMP`
  fields for packed math at `rdna3/README.md:2581` through `:2628`.
- XML encodes these fields in the corresponding VOP3 and VOP3P formats, for
  example VOP3 literal fields at `amdgpu_isa_rdna3.xml:6529` through `:6606`.

Rocjitsu evidence:

- Generated RDNA3 raw instruction structs store VOP3 `abs`, `op_sel`, `clamp`,
  `omod`, and `neg` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/machine_insts.h:63`
  through `:75`, and VOP3P `neg_hi`, `op_sel`, `op_sel_hi_2`, `clamp`,
  `op_sel_hi`, and `neg` at `:77` through `:90`.
- `Vop3`, `Vop3p`, and `Vop3SdstEnc` do not override `build_modifiers()` in
  `rdna3/encodings.h:485` through `:529` and `:594` through `:610`.
- The common disassembler prints explicit operands and then calls
  `build_modifiers()`, whose default implementation emits nothing, at
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:239` through `:274`.

Impact:

Rocjitsu decodes the raw modifier bits, and execution may use some of them, but
the disassembly string drops them. Metadata-driven tests or diagnostics can
therefore report identical text for encodings with different VOP3/VOP3P
modifier state.

### RDNA3-RJ-049: Generic VOP3 source modifiers, `OMOD`, and `OPSEL` legality is not validated

Manual/XML evidence:

- Chapter 7.1 restricts VOP3 `NEG`, `ABS`, and `OMOD` to floating-point use and
  says `OPSEL` must be zero for non-16-bit operands/results at
  `rdna3/README.md:2169` through `:2177`.
- Section 7.2.2.1 says input modifiers are undefined for non-FP inputs except
  selected move/cndmask forms and are unsupported for integer/bitwise,
  readlane, writelane, permlane, and QSAD at `rdna3/README.md:2264` through
  `:2276`.
- Section 7.2.2.4 says VOP3 `OPSEL` is usable only by an allow-listed subset of
  VOP3 and promoted VOP1/VOP2/VOPC instructions at `rdna3/README.md:2308`
  through `:2318`.

Rocjitsu evidence:

- `Vop3MachineInst` preserves the raw VOP3 modifier and selector bits at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/machine_insts.h:63`
  through `:75`, but the generated VOP3 base class only sizes literals and does
  not validate modifier legality at `rdna3/encodings.cpp:201` through `:209`.
- Representative integer VOP3 forms such as `VAddNcU32Vop3` construct ordinary
  32-bit operands from the raw sources at `rdna3/vop3.cpp:7820` through
  `:7838`; execution ignores `abs`, `op_sel`, `omod`, and `neg` while
  performing the add at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:3203`
  through `:3212`. Legal integer `CLAMP` semantics are tracked separately in
  `RDNA3-RJ-053`.

Impact:

Invalid encodings with nonzero VOP3 source modifiers, `OMOD`, or selectors can
decode and run as if the forbidden bits were absent. That hides
illegal-instruction cases and can make opcode-fuzzer output look semantically
valid when the manual requires zero or forbids the modifier.

### RDNA3-RJ-050: VALU scalar destination selectors can mutate forbidden special registers

Manual/XML evidence:

- Chapter 7.1 says `VDST` used as a scalar destination for `V_READLANE` and
  `V_CMP` cannot be `M0` or `EXEC`, and says VOP3SD `SDST` cannot be `M0` or
  `EXEC` while supporting `NULL`, at `rdna3/README.md:2168` through `:2169`.
- XML uses `OPR_SREG` for representative scalar destinations such as
  `V_READLANE_B32` `VDST` at `amdgpu_isa_rdna3.xml:113649` through `:113654`
  and `V_ADD_CO_CI_U32` `SDST` at `amdgpu_isa_rdna3.xml:77139` through
  `:77143`.

Rocjitsu evidence:

- `VReadlaneB32Vop3` builds scalar `vdst` directly from the raw VOP3 `VDST`
  field as `OPR_SREG` and writes through `RegisterAccess::write_scalar()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/vop3.cpp:19516` through
  `:19523` and `:19572` through `:19573`.
- VOP3SD carry forms build `sdst` directly from the raw `SDST` field as
  `OPR_SREG`, for example `VAddCoCiU32Vop3SdstEnc` at `rdna3/vop3.cpp:36608`
  through `:36615`; the shared carry helper writes that scalar destination at
  `shared/execute_shared.h:2882` through `:2885`.
- RDNA3 scalar write resolution accepts raw selector 125 as `M0`, 126/127 as
  `EXEC_LO`/`EXEC_HI` for 32-bit writes, and 126 as full `EXEC` for 64-bit
  writes at `rdna3/operand.cpp:897` through `:943`.

Impact:

Illegal VALU encodings with scalar-destination selectors for `M0` or `EXEC`
can mutate those special registers through the ordinary destination path,
instead of being rejected or treated as invalid according to the manual/XML
operand class.

### RDNA3-RJ-051: `V_READLANE` and `V_WRITELANE` use unmasked lane selectors

Manual/XML evidence:

- Section 7.2.1 says readlane lane-select ignores upper bits and is limited to
  0-31 for wave32 or 0-63 for wave64 at `rdna3/README.md:2239` through
  `:2247`.
- XML identifies the lane-select source as `OPR_SSRC_LANESEL` for
  `V_READLANE_B32` and `V_WRITELANE_B32` at
  `amdgpu_isa_rdna3.xml:113641` through `:113665` and
  `amdgpu_isa_rdna3.xml:113685` through `:113709`.

Rocjitsu evidence:

- `VReadlaneB32Vop3::execute_impl()` reads the scalar lane selector and passes
  it unmodified to `RegisterAccess::read_lane()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/vop3.cpp:19572` through
  `:19573`.
- `VWritelaneB32Vop3::execute_impl()` reads the scalar lane selector and passes
  it unmodified to `RegisterAccess::write_lane()` at `rdna3/vop3.cpp:19647`
  through `:19649`.
- `Operand::read_lane()` and `Operand::write_lane()` forward the lane value to
  the VGPR accessors without applying a wave-size mask at
  `rdna3/operand.cpp:1041` through `:1059` and `:1066` through `:1082`.

Impact:

Selectors with high bits set can address the wrong lane or fall outside the
implemented VGPR-lane range instead of matching the documented masked hardware
behavior.

### RDNA3-RJ-052: VOP3-family source-combination legality is not validated

Manual/XML evidence:

- Section 7.2.2.3 limits VALU instructions to at most two scalar values, permits
  one literal extension, counts implicit VCC uses for add/sub carry, div-fmas,
  and cndmask, gives special 64-bit shift rules, and adds wave64 mask-source
  sharing restrictions at `rdna3/README.md:2288` through `:2306`.
- Section 7.2.5 and 7.2.6 identify mask/carry SGPR uses and distinguish
  broadcast SGPR data from per-lane mask bits at `rdna3/README.md:2398`
  through `:2422`.

Rocjitsu evidence:

- Generated RDNA3 VOP3, VOP3P, and VOP3SD base constructors size a literal
  extension for any matching raw selector combination, including
  `has_lit_0_has_lit_1`, `has_lit_0_has_lit_2`, and
  `has_lit_0_has_lit_1_has_lit_2`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/encodings.cpp:201`
  through `:209`, `:253` through `:261`, and `:431` through `:440`.
- Representative VOP3 paths construct all raw sources directly with broad
  operand classes before execution, such as `VDivFmasF32Vop3` using three
  `OPR_SRC` operands at `rdna3/vop3.cpp:11794` through `:11818` and
  `VAddCoCiU32Vop3SdstEnc` using two `OPR_SRC` values plus an `OPR_SREG` mask
  input at `rdna3/vop3.cpp:36608` through `:36634`.
- The audited paths did not find a cross-operand validator for scalar-source
  counts, implicit VCC accounting, literal size accounting, DPP/literal
  exclusion, or wave64 mask-source sharing.

Impact:

Rocjitsu can accept source combinations that the manual says are illegal, and
wave64 mask/carry source constraints are not represented even at decode time.
This is partly masked by the current RDNA3 wave32-only runtime, but it remains a
decoder/validation gap.

### RDNA3-RJ-053: VOP3 integer arithmetic ignores supported `CLAMP` saturation

Manual/XML evidence:

- Section 7.2.3.1 says `CLAMP` clamps integer results to the signed or unsigned
  representable range, and its unsupported-opcode list excludes ordinary
  arithmetic forms such as `V_ADD_NC_{I,U}32`, `V_SUB_NC_{I,U}32`, and
  `V_MUL_LO_U32`, at `rdna3/README.md:2353` through `:2371`.
- The XML generic VOP3 `CLAMP` field repeats the signed/unsigned integer clamp
  meanings and references a non-structured `OPF_NOCLAMP` list at
  `amdgpu_isa_rdna3.xml:6539` through `:6540`.

Rocjitsu evidence:

- Representative generated integer VOP3 constructors preserve the raw VOP3
  `clamp` field, such as `VAddNcU32Vop3` at `rdna3/vop3.cpp:7820` through
  `:7838`.
- `execute_v_add_nc_i32_vop3` and `execute_v_add_nc_u32_vop3` use plain
  wraparound addition and never inspect `inst.inst_.clamp` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:3163`
  through `:3172` and `:3203` through `:3212`.
- `execute_v_sub_nc_i32_vop3`, `execute_v_sub_nc_u32_vop3`, and
  `execute_v_mul_lo_u32_vop3` likewise perform plain subtraction/multiply
  without `clamp` handling at `shared/execute_shared.h:18716` through `:18724`,
  `:18756` through `:18764`, and `:16483` through `:16491`.
- The integer SIMD helper documents that these VOP3 integer paths apply no
  result modifiers and that integer clamp would mean saturate, at
  `shared/simd_glue.h:1367` through `:1372`.

Impact:

For supported integer arithmetic with `CLAMP==1`, rocjitsu can produce the
wraparound result instead of the saturated signed/unsigned result described by
the manual.

### RDNA3-RJ-054: `PERMLANE`-after-`CMPX` stream hazard is not tracked

Manual/XML evidence:

- Section 7.2.8 says `V_PERMLANE` may not occur immediately after `V_CMPX`;
  another VALU instruction such as `V_NOP` must be inserted, at
  `rdna3/README.md:2432` through `:2434`.
- XML entries describe the standalone `V_PERMLANE16_B32`,
  `V_PERMLANEX16_B32`, and `V_PERMLANE64_B32` operations, but do not encode the
  adjacency hazard, at `amdgpu_isa_rdna3.xml:101888` through `:102022` and
  `amdgpu_isa_rdna3.xml:69851` through `:69880`.

Rocjitsu evidence:

- Generated `VPermlane16B32Vop3` and `VPermlanex16B32Vop3` constructors and
  execute paths perform local operand setup/execution with no previous-opcode
  check at `rdna3/vop3.cpp:14825` through `:15013`.
- `VPermlane64B32Vop1` similarly dispatches directly to the shared permute
  executor at `rdna3/vop1.cpp:10422` through `:10512`, and the shared helper
  only snapshots and swaps lanes at `shared/execute_shared.h:16642` through
  `:16654`.
- The compute-unit execution path dispatches the current decoded instruction via
  `execute_instruction(inst, *active)` and the concrete override calls
  `inst->execute(*inst, &wf)` directly at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:393` through `:412` and
  `vm/amdgpu/compute_unit.h:753` through `:756`. The searched RDNA3 VALU
  execution path did not find previous-VALU tracking for `V_CMPX`.

Impact:

Rocjitsu can execute instruction streams that RDNA3 requires software to avoid
or pad, so decoder/runtime tests will not expose missing scheduling around the
`V_CMPX` to `V_PERMLANE*` hazard.

### RDNA3-RJ-055: Wave64 VALU same-SGPR read/write hazards are not validated

Manual/XML evidence:

- Section 7.2.3.2 says wave64 VALU instructions may issue as two wave32 passes
  and may not both read and write the same SGPR value, because the first pass can
  overwrite a scalar value needed by the second pass, at `rdna3/README.md:2373`
  through `:2375`.
- Section 7.2.5 and 7.2.6 identify VALU scalar mask/carry inputs and outputs at
  `rdna3/README.md:2398` through `:2422`.

Rocjitsu evidence:

- RDNA3 wave64 execution is not selectable today, as tracked by `RDNA3-RJ-007`,
  but decode/construction still accepts the affected scalar operand tuples.
- `VAddCoCiU32Vop3SdstEnc` binds `sdst` as a 64-bit `OPR_SREG` destination and
  `src2` as a 64-bit `OPR_SREG` carry input at `rdna3/vop3.cpp:36608` through
  `:36615`, then the shared helper reads `src2` and writes `sdst` at
  `shared/execute_shared.h:2865` through `:2885`.
- The generated VOP3/VOP3SD base constructors only size literal forms and do not
  validate scalar source/destination overlap at `rdna3/encodings.cpp:201`
  through `:209` and `:431` through `:440`.

Impact:

Even before RDNA3 wave64 execution is modeled, rocjitsu lacks validation for a
manual wave64 hazard on VALU scalar operands. Encodings that would be
unpredictable on hardware can be accepted as ordinary decoded instructions.

### RDNA3-RJ-056: Several Chapter 7.3 VALU instructions decode but always throw

Manual/XML evidence:

- Chapter 7.3 lists `V_PK_FMAC_F16`, `V_TRIG_PREOP_F64`,
  `V_MQSAD_PK_U16_U8`, `V_MQSAD_U32_U8`, `V_MULLIT_F32`,
  `V_QSAD_PK_U16_U8`, and `V_SAT_PK_U8_I16` in the non-VOP3P VALU inventory at
  `rdna3/README.md:2480`, `:2490`, `:2514` through `:2517`, `:2522`, and
  `:2524`.
- The detailed opcode tables assign these instructions concrete VOP1, VOP2, or
  VOP3 opcode slots, for example `V_PK_FMAC_F16` at `rdna3/README.md:5831`,
  `V_SAT_PK_U8_I16` at `:5902`, `V_MULLIT_F32` at `:6321`, QSAD/MQSAD at
  `:6347` through `:6349`, and `V_TRIG_PREOP_F64` at `:6188`.
- The RDNA3 XML contains these affected instructions as concrete VALU entries,
  for example `V_SAT_PK_U8_I16` at `amdgpu_isa_rdna3.xml:69262`,
  `V_PK_FMAC_F16` at `:81952`, `V_MULLIT_F32` at `:86897`, QSAD/MQSAD at
  `:93309` through `:93843`, and `V_TRIG_PREOP_F64` at `:112733`.

Rocjitsu evidence:

- Generated baseline encodings exist for the affected names in
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/test_encodings.h:308`,
  `:549`, `:626`, `:684`, `:704` through `:706`, and `:777`.
- `VPkFmacF16Vop2::execute_impl` unconditionally throws
  `util::UnimplementedInst` at `rdna3/vop2.cpp:6103` through `:6105`.
- `VSatPkU8I16Vop1::execute_impl` and `VSatPkU8I16Vop3::execute_impl`
  unconditionally throw at `rdna3/vop1.cpp:9822` through `:9824` and
  `rdna3/vop3.cpp:5653` through `:5655`.
- `VMullitF32Vop3::execute_impl`, `VQsadPkU16U8Vop3::execute_impl`,
  `VMqsadPkU16U8Vop3::execute_impl`, `VMqsadU32U8Vop3::execute_impl`, and
  `VTrigPreopF64Vop3::execute_impl` likewise throw at
  `rdna3/vop3.cpp:10532` through `:10534`, `:12080` through `:12180`, and
  `:18999` through `:19001`.
- The separate `V_MOVRELSD*` and `V_SWAPREL_B32` unimplemented or partial bodies
  are already tracked by `RDNA3-RJ-021`.

Impact:

These inventory-listed VALU instructions can decode and disassemble, but any
functional execution path reaches an unconditional unimplemented exception.
Baseline encoding tests therefore prove table presence, not executable
semantics, for this subset of Chapter 7.3.

### RDNA3-RJ-057: OPSEL-based true16 destinations `128..255` alias to `v0..v127`

Manual evidence:

- Section 7.4 says VOP3, VOP3P, and VINTERP true16 encodings use
  `SRC/DST[7:0]` as the 32-bit VGPR address and `OPSEL` as the high/low half
  selector, so a wave can address 512 16-bit VGPRs at `rdna3/README.md:2565`
  through `:2567`.

Rocjitsu evidence:

- RDNA3 `packed_16bit_vgpr_dst` unconditionally treats any 16-bit `OPR_VGPR`
  destination value `128..255` as the high half of `v0..v127` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/operand.cpp:50` through
  `:58`.
- `Operand::to_register_ref()` applies that compact-E32 destination collapse
  before normal VGPR resolution at `rdna3/operand.cpp:561` through `:570`.
- `write_vop3_true16_dst()` then uses the collapsed register reference and
  applies `OPSEL[3]` only as the destination half selector at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/simd_glue.h:297` through
  `:321`.
- Representative affected generated paths construct 16-bit `OPR_VGPR`
  destinations from raw VOP3/VOP3P/VINTERP `VDST` fields:
  `rdna3/vop3.cpp:580` through `:584`, `:8185` through `:8189`,
  `rdna3/vop3p.cpp:1983` through `:1989`, and
  `rdna3/vinterp.cpp:76` through `:82`.

Impact:

For OPSEL-based true16 instructions, raw `VDST=128` with `OPSEL[3]=0` should
write `v128.l`, but rocjitsu resolves the destination as `v0.l`. The same
collapse affects high-half writes and liveness register references for raw
destinations `128..255`.

### RDNA3-RJ-058: `VINTERP` instructions decode but execute as no-ops

Manual/XML evidence:

- Section 7.4 groups VINTERP with VOP3 and VOP3P as an OPSEL-based 16-bit
  encoding at `rdna3/README.md:2555` through `:2567`.
- Section 12.3.1 says 16-bit interpolation uses OPSEL to select the upper or
  lower half, gives real P10/P2 F16 formulas, and notes RTZ variants at
  `rdna3/README.md:4718` through `:4733`.
- Chapter 16.13 gives executable formulas for the F32 P10/P2 interpolation
  passes at `rdna3/README.md:21942` through `:21980`, and the F16/RTZ forms at
  `rdna3/README.md:21983` through `:22027`.
- The VINTERP format table carries `VDST`, `WAITEXP`, `OPSEL`, `CLMP`, source
  fields, and the F16/RTZ opcodes at `rdna3/README.md:6716` through `:6778`.
- XML lists all six VINTERP opcodes with interpolation descriptions and
  concrete operand widths at `amdgpu_isa_rdna3.xml:57542` through `:57810`.

Rocjitsu evidence:

- RDNA3 generates concrete constructors for the two F32 VINTERP forms at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/vinterp.cpp:19` through
  `:44` and for the four F16 VINTERP forms at `:57` through `:82` and `:101`
  through `:126`.
- Every generated VINTERP `execute_impl` body is a compute-simulation no-op:
  `rdna3/vinterp.cpp:34` through `:35`, `:53` through `:54`, `:72` through
  `:73`, `:97` through `:98`, `:116` through `:117`, and `:141` through `:142`.

Impact:

Decoded VINTERP instructions leave their destination unchanged instead of
performing the fixed-DPP interpolation. The F16 forms also skip true16 source
half selection, RTZ rounding where applicable, and OPSEL-controlled destination
half update.

### RDNA3-RJ-059: VOPD `DOT2ACC` opcodes decode but are not executable

Manual evidence:

- Section 7.6 lists `DOT2ACC_F32_F16` and `DOT2ACC_F32_BF16` as VOPD `SRC2`
  users that read the destination operand at `rdna3/README.md:2699`.
- The VOPD OPX and OPY opcode tables list `V_DUAL_DOT2ACC_F32_F16` and
  `V_DUAL_DOT2ACC_F32_BF16` at `rdna3/README.md:6625` through `:6649`.
- Chapter 16.11 says both DOT2ACC forms accumulate with the destination and use
  the initial destination value as `S2` at `rdna3/README.md:15591` through
  `:15593` and `rdna3/README.md:15643` through `:15646`.

Rocjitsu evidence:

- RDNA3 `Vopd::op_name` names both DOT2ACC opcodes at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/vopd.cpp:87` through
  `:90`, and decode smoke coverage includes DOT2ACC VOPD encodings at
  `tests/decode_smoke_test.cpp:263` through `:278`.
- `Vopd::execute_slot` has no DOT2ACC case and falls through to
  `UnimplementedInst` at `rdna3/vopd.cpp:134` through `:209`.
- `init_operands()` source collection special-cases `FMAC`, `MOV`, and
  `CNDMASK`, but not DOT2ACC, so the destination-as-`SRC2` accumulator read is
  not included in public source metadata at `rdna3/vopd.cpp:266` through
  `:315`.

Impact:

Valid RDNA3 VOPD DOT2ACC instructions can decode and disassemble, but fail at
execution and expose incomplete source/dependency metadata.

### RDNA3-RJ-060: VOPD decoder accepts reserved or invalid opcode pairs

Manual evidence:

- Section 15.3.7 gives finite OPX opcode values `0` through `13` and OPY
  opcode values `0` through `13`, `16`, `17`, and `18` at
  `rdna3/README.md:6625` through `:6649`.

Rocjitsu evidence:

- RDNA3 `Decoder::decode` routes any instruction whose top bits satisfy
  `Vopd::is_vopd()` to the VOPD constructor at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/decoder.cpp:16` through
  `:18`.
- `Vopd::is_vopd()` checks only `(word0 >> 26) == 0x32` at
  `rdna3/vopd.cpp:56` through `:59`.
- Unknown slot opcodes format as `v_dual_unknown` at `rdna3/vopd.cpp:97`
  through `:99`, and only throw if execution reaches the default case at
  `rdna3/vopd.cpp:206` through `:207`.

Impact:

Reserved VOPD encodings can be represented and disassembled as VOPD instead of
being rejected at decode time, and the failure is deferred until slot
evaluation.

### RDNA3-RJ-061: VOPD pair legality restrictions are not enforced

Manual evidence:

- Section 7.6 says VOPD is legal only for wave32 and hardware does not function
  correctly unless all restrictions are met, at `rdna3/README.md:2676`.
- The restrictions include per-slot VGPR count, scalar/literal combinations,
  `VSRC1` VGPR-only use, source-cache bank and port limits, `SRC2` even/odd
  pairing, one literal, one even and one odd destination, instruction
  independence, no DPP, and wave32-only at `rdna3/README.md:2687` through
  `:2706`.

Rocjitsu evidence:

- The RDNA3 VOPD constructor extracts fields, builds operands, infers literal
  size, reconstructs the Y destination parity bit, and initializes slot operands
  at `rdna3/vopd.cpp:211` through `:315`.
- No validation in that path checks SGPR/literal counts, source-cache bank/port
  limits, `SRC2` parity, pair independence, DPP selector use, or wave32-only
  legality.

Impact:

The functional model can accept and often execute deterministic software
results for encodings the manual says are illegal and may not function
correctly on hardware, hiding compiler or assembler legality bugs.

### RDNA3-RJ-062: Illegal VOPD DPP selectors are not diagnosed as VOPD legality errors

Manual evidence:

- Section 7.6 says a VOPD pair must not use DPP at `rdna3/README.md:2705`.
- Section 15.3.7 says VOPD can be followed by a 32-bit literal constant, but
  not a DPP control DWORD, at `rdna3/README.md:6573`.
- The VOPD source selector table still assigns values `233`, `234`, and `250`
  to DPP selectors at `rdna3/README.md:6611` through `:6614`.

Rocjitsu evidence:

- RDNA3 VOPD routes non-literal source-0 selectors through generic `OPR_SRC` at
  `rdna3/vopd.cpp:40` through `:48`.
- The RDNA3 `OPR_SRC` enum lists ordinary SGPR, inline, aperture, literal, SCC,
  and VGPR selector names but no DPP selector names at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/operand_types.h:137`
  through `:167`.
- Generic scalar source resolution treats selector `250` as zero/`NULL`, while
  selectors `233` and `234` fall through to a generic unsupported scalar-read
  exception rather than a VOPD-specific DPP legality diagnostic at
  `rdna3/operand.cpp:703` through `:809`.

Impact:

Illegal VOPD encodings using DPP selector values can be decoded as ordinary
source operands and may either mis-execute or fail later with a generic operand
error instead of being rejected as invalid VOPD encodings.

### RDNA3-RJ-063: VOPD implicit VCC source accounting is incomplete

Manual evidence:

- Section 7.6 says `V_CNDMASK_B32` is the VOP2 form that uses VCC, and VCC
  counts as one SGPR read at `rdna3/README.md:2725`.

Rocjitsu evidence:

- RDNA3 VOPD marks CNDMASK slots with `uses_vcc` at `rdna3/vopd.cpp:269`
  through `:270`, and execution reads `wf.vcc()` for those slots at
  `rdna3/vopd.cpp:187` through `:190`.
- `add_slot_sources()` omits any implicit VCC operand when `uses_vcc` is true
  at `rdna3/vopd.cpp:300` through `:305`.

Impact:

Downstream dependency or legality logic cannot see the implicit VCC read needed
to enforce VOPD scalar-source limits or reason about CNDMASK dependencies.

### RDNA3-RJ-064: VOPD `MOV` reads the ignored `VSRC1` field during execution

Manual/XML evidence:

- Section 7.6 says `vsrc1X` and `vsrc1Y` are ignored for `V_MOV_B32` at
  `rdna3/README.md:2718` through `:2719`.
- The RDNA3 XML `V_DUAL_MOV_B32` entries expose only destination plus
  `SRC0`/literal operands, not `VSRC1`, at `amdgpu_isa_rdna3.xml:168999`
  through `:169075`.

Rocjitsu evidence:

- `Vopd::execute_slot` unconditionally reads `slot.src1` before switching on
  the slot opcode at `rdna3/vopd.cpp:134` through `:136`.
- The `V_MOV_B32` case later returns only `src0` at `rdna3/vopd.cpp:185`
  through `:186`.
- Source collection and disassembly correctly omit `src1` for MOV at
  `rdna3/vopd.cpp:294` through `:299` and `rdna3/vopd.cpp:321` through
  `:323`, so execution-time behavior disagrees with the advertised source
  list.

Impact:

MOV slots can perform a spurious VGPR read that hardware ignores. This can
distort dependency instrumentation and out-of-range/source-side-effect modeling
even when the final data result is unchanged.

### RDNA3-RJ-065: VOPD paired-exception coalescing is not modeled

Manual evidence:

- Section 7.6 says VOPD instruction pairs generate only a single exception if
  either or both paired operations raise an exception at `rdna3/README.md:2727`.

Rocjitsu evidence:

- `Vopd::execute_slot` directly performs each slot's arithmetic at
  `rdna3/vopd.cpp:134` through `:209`.
- `Vopd::execute_impl` evaluates X then Y and writes both results at
  `rdna3/vopd.cpp:345` through `:355`; no VOPD-level exception aggregation or
  coalescing hook was found in this path.

Impact:

If rocjitsu grows or uses FP exception/trap fidelity for VOPD, the current slot
execution structure does not encode the manual's single-exception rule.

### RDNA3-RJ-066: `S_BUFFER_LOAD_*` ignores scalar buffer descriptors and bounds

Manual evidence:

- Section 8.1.2 says scalar buffer loads use the buffer resource descriptor's
  `base_address`, `stride`, and `num_records`, use `stride` only for bounds,
  require nonnegative `inst_offset`, and compute `m_size` from
  `(stride == 0 ? 1 : stride) * num_records` at `rdna3/README.md:2962`
  through `:2985`.
- Section 8.4 says `SBASE` for `S_BUFFER_LOAD` names a four-SGPR-aligned
  descriptor, out-of-range `SBASE` uses `SGPR0`, and buffer out-of-range DWORDs
  return zero, including partial multi-DWORD loads, at `rdna3/README.md:3023`
  through `:3041`.

Rocjitsu evidence:

- The generated `SBufferLoad*Smem` constructors expose `SBASE` as a 128-bit
  scalar operand, but each execute body calls the same `smem_calculate_address`
  helper used by raw `S_LOAD_*` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/smem.cpp:165` through
  `:297`.
- `smem_calculate_address()` reads only two scalar registers as a raw 64-bit
  base, adds the signed `OFFSET` plus `SOFFSET`, and clears low address bits at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/addr_calc.cpp:43` through
  `:50`; it does not read descriptor stride or `num_records`, form the
  48-bit scalar-buffer base, or reject negative buffer `OFFSET`.
- `ScalarMemPipeline` loads `num_dwords` directly from scalar L1 and writes
  every returned DWORD to SGPRs at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:203` through
  `:227`; no per-DWORD scalar-buffer bounds mask or zero fill is visible.

Impact:

Scalar buffer loads can read from the wrong address and return real memory for
DWORDs that hardware would treat as out of range or `MEMVIOL`, so tests using a
legal resource descriptor do not exercise the Chapter 8 scalar-buffer contract.

### RDNA3-RJ-067: SMEM `SOFFSET` is hidden from source metadata

Manual/XML evidence:

- Section 8.1 defines `SOFFSET` as an SGPR, `M0`, or `NULL` unsigned byte
  offset at `rdna3/README.md:2929` through `:2931`.
- XML models `SOFFSET` as an explicit `OPR_SMEM_OFFSET` source operand on
  scalar loads and scalar buffer loads at `amdgpu_isa_rdna3.xml:41087` through
  `:41091` and `amdgpu_isa_rdna3.xml:41347` through `:41351`, with
  `OPR_SMEM_OFFSET` partitioned into `M0`, `NULL`, and scalar-register
  subtypes at `amdgpu_isa_rdna3.xml:175329` through `:175335`.

Rocjitsu evidence:

- RDNA3 generated SMEM constructors build the public `soffset` operand with
  `make_smem_offset()`, which always returns `Operand(32, OPR_SIMM32,
  enc->offset)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/smem.cpp:26` through
  `:29`.
- The constructors then publish that operand as `src_operands_[1]` for both
  raw and buffer SMEM loads, for example at `rdna3/smem.cpp:32` through `:42`
  and `:165` through `:177`.
- Execution still reads the raw instruction `soffset` field through
  `read_smem_offset()`, including `NULL`, `M0`, and SGPR cases, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/addr_calc.cpp:31` through
  `:39`.

Impact:

Address execution can use `SOFFSET`, but disassembly, def-use, dependency, and
fuzzing surfaces see the immediate `OFFSET` as the second source instead of the
actual SGPR/M0/NULL `SOFFSET` dependency.

### RDNA3-RJ-068: SMEM wait-counter accounting is width-insensitive and skips invalidates

Manual evidence:

- Section 8.2 says `LGKMcnt` increments by 1 for single-DWORD fetches or cache
  invalidates, by 2 for fetches of two or more DWORDs, and decrements by the
  same amount on completion at `rdna3/README.md:2995` through `:3001`.
- The same section says scalar cache invalidates are not known complete until
  the shader waits for `LGKMcnt == 0` at `rdna3/README.md:3003` through
  `:3005`.

Rocjitsu evidence:

- Every generated RDNA3 scalar load records `d->num_dwords`, but sets only one
  `WaitCounterType::KMCNT` token, for example `S_LOAD_B32` and `S_LOAD_B512`
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/smem.cpp:45` through
  `:162`.
- `MemoryPipeline::issue()` increments exactly once for the selected counter
  and decrements exactly once on completion at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.h:62` through `:87`;
  `WaitCounters::increment(KMCNT)` increments `KMCNT` and `LGKMCNT` by one at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wait_counters.h:76` through `:99`.
- `S_GL1_INV` and `S_DCACHE_INV` have no `MEMORY_OP` flag or scalar-memory
  state and execute direct cache invalidations at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/smem.cpp:300` through
  `:315` and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1763`
  through `:1768`.

Impact:

Multi-DWORD scalar loads contribute one wait-counter event instead of two, and
cache invalidations do not participate in `LGKMcnt`, so wait-count-sensitive
tests cannot observe the Chapter 8 completion contract.

### RDNA3-RJ-069: SMEM invalidate group restrictions are not modeled

Manual evidence:

- Section 8.3 says scalar memory instructions issue in groups and that `INV`
  instructions must be in a group by themselves and may not be in a clause at
  `rdna3/README.md:3007` through `:3015`.

Rocjitsu evidence:

- `S_GL1_INV` and `S_DCACHE_INV` decode as ordinary operandless SMEM
  instructions and execute direct cache invalidation helpers at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/smem.cpp:300` through
  `:315`.
- The existing clause audit found `S_CLAUSE` is a no-op and no active-clause
  state is tracked (`RDNA3-RJ-041`), and no separate SMEM group-state tracker
  was found in the RDNA3 execute path.

Impact:

Rocjitsu can execute scalar cache invalidations adjacent to other SMEM
instructions or inside an apparent clause without diagnosing the manual's
instruction-stream restriction.

### RDNA3-RJ-070: Buffer address and bounds calculation uses a simplified one-bit model

Manual evidence:

- Section 9.4 defines buffer address calculation from descriptor base, SGPR
  offset, instruction offset, `IDXEN`/`OFFEN`, VGPR index/offset, stride,
  add-tid, and optional swizzle fields at `rdna3/README.md:3363` through
  `:3412`.
- Section 9.4.1 defines the two-bit `OOB_SELECT` modes, payload-inclusive
  bounds checks, all-or-nothing formatted/atomic behavior, and per-component
  MUBUF versus MTBUF B64/B96/B128 range behavior at `rdna3/README.md:3414`
  through `:3432`.
- Sections 9.4.1.1 through 9.4.2 give different structured, raw, scratch,
  scalar, and swizzled address formulas at `rdna3/README.md:3434` through
  `:3501`.

Rocjitsu evidence:

- `mubuf_calculate_addresses()` reads descriptor base, stride, `num_records`,
  and one `oob_raw` bit from `srd3 >> 31`, then always forms
  `index * stride + offset + soffset` before a lane-level OOB check at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h:57`
  through `:133`.
- The MTBUF helper duplicates the same model at
  `addr_calc_buffer.h:169` through `:217`.
- `VectorMemState` carries one `lane_mask` and a single
  `elem_size`/`num_elems` tuple at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:92` through `:111`, and
  memory completion zeroes whole lanes rather than tracking MUBUF
  per-component OOB results.

Impact:

Rocjitsu can compute the wrong address or bounds result for descriptors that
use RDNA3 `OOB_SELECT` values, raw mode with nonzero stride, scratch/add-tid
addressing, swizzled addressing, payload-inclusive bounds, or B64/B96/B128
partial-component OOB cases.

### RDNA3-RJ-071: Formatted MUBUF/MTBUF conversion semantics are missing or fixed-width

Manual evidence:

- Section 9.2 says formatted buffer loads/stores return float, unsigned, or
  signed integer VGPR data based on memory format, and D16 variants pack
  components into VGPR halves at `rdna3/README.md:3246` through `:3312`.
- Section 9.3 says MTBUF data format comes from the instruction, MUBUF
  `FORMAT` instructions use the resource, `DST_SEL` can come from the resource,
  and format/opcode component-count mismatches have defined behavior at
  `rdna3/README.md:3313` through `:3362`.
- Chapter 16.16 and 16.17 define MUBUF and MTBUF formatted operations in terms
  of `ConvertFromFormat`/`ConvertToFormat`, including D16 preservation and
  packing, at `rdna3/README.md:23777` through `:24768`.

Rocjitsu evidence:

- RDNA3 MUBUF formatted loads/stores, including D16 formatted variants, throw
  `util::UnimplementedInst`, for example `BUFFER_LOAD_FORMAT_X` through
  `BUFFER_STORE_FORMAT_XYZW` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/mubuf.cpp:54` through
  `:210`, D16 formatted forms at `:230` through `:386`, and D16_HI format-X
  forms at `:1184` through `:1208`.
- RDNA3 MTBUF non-D16 formatted instructions execute as fixed 32-bit
  components by setting `elem_size = 4` and `num_elems` from the opcode, for
  example at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/mtbuf.cpp:54`
  through `:157`, while D16 MTBUF formatted variants throw at `:349` through
  `:505`.
- The shared vector memory state has no resource-format or `DST_SEL` fields at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:92` through `:111`.

Impact:

Legal formatted MUBUF instructions are not executable, and legal MTBUF
formatted instructions that require non-B32 memory formats, conversion, D16
packing, resource validity, `DST_SEL`, or component-count mismatch behavior can
produce byte-copy results instead of texture-format results.

### RDNA3-RJ-072: Buffer `SOFFSET` special selectors are bypassed during execution

Manual/XML evidence:

- Section 9.1 says buffer `SOFFSET` can be an SGPR, `M0`, `NULL`, or an inline
  constant at `rdna3/README.md:3116` through `:3123`.
- XML uses `OPR_SREG_M0_INL` for buffer `SOFFSET`; the generated RDNA3 operand
  type enumerates SGPRs, `NULL` as 124, `M0` as 125, positive inline constants
  from 128, negative inline constants from 193, and float inline constants from
  240 at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/operand_types.h:203`
  through `:225`.

Rocjitsu evidence:

- Generated MUBUF and MTBUF constructors expose `SOFFSET` as
  `OPR_SREG_M0_INL`, for example
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/mubuf.cpp:43` through
  `:45` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/mtbuf.cpp:42`
  through `:44`.
- The address helpers do not call the operand resolver. They special-case only
  selector `0x80` as zero, and otherwise read
  `read_sgpr(wf.sgpr_alloc().base + inst.soffset)` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h:63`
  through `:67` for MUBUF and `:175` through `:178` for MTBUF.
- The scalar resolver already knows how to return zero for `NULL`, `wf.m0()`
  for `M0`, and inline constant values for ordinary scalar sources at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/operand.cpp:703` through
  `:735`, but that path is bypassed here.

Impact:

Buffer instructions with `SOFFSET = NULL`, `M0`, or inline constants other than
zero can use an SGPR-numbered value instead of the architectural offset value,
changing the effective address and bounds decision.

### RDNA3-RJ-073: `BUFFER_ATOMIC_CSUB_U32` executes as ordinary subtract

Manual/XML evidence:

- Chapter 16.16 defines `BUFFER_ATOMIC_CSUB_U32` as subtracting the data value
  from memory and clamping the stored result to zero if `old_value < DATA` at
  `rdna3/README.md:24251` through `:24269`.
- XML describes `BUFFER_ATOMIC_CSUB_U32` as clamped unsigned subtract and says
  the original buffer value is returned to the vector register when `GLC` is
  set at `amdgpu_isa_rdna3.xml:39339` through `:39375`.

Rocjitsu evidence:

- The generated RDNA3 `BufferAtomicCsubU32Mubuf` execute body sets
  `d->atomic_op = amdgpu::AtomicOp::SUB` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/mubuf.cpp:1422` through
  `:1429`.
- `AtomicOp` has no CSUB value at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:45` through `:68`, and
  `apply_int_atomic(AtomicOp::SUB)` computes `old_val - src_val` without
  clamping at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:247`
  through `:260`.

Impact:

When `DATA` is larger than the memory value, rocjitsu wraps the unsigned
subtraction instead of storing zero.

### RDNA3-RJ-074: Buffer `TFE` status return is not modeled

Manual evidence:

- Section 9.1 says `TFE` enables partially resident texture fault reporting for
  buffer instructions and writes a status word to the VGPR after the last
  fetch-destination VGPR on a NACK at `rdna3/README.md:3124` through `:3129`.

Rocjitsu evidence:

- RDNA3 machine instruction structs store `tfe` for MUBUF and MTBUF at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/machine_insts.h:131`
  through `:162`.
- Searches of the generated RDNA3 MUBUF/MTBUF execution files, shared buffer
  helpers, and vector memory pipeline found no runtime use of `inst_.tfe`.
- `VectorMemState` tracks destination base, element size, lane mask, D16 flags,
  and atomic state but has no status-return field or extra destination window
  metadata at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:92` through
  `:111`.

Impact:

Rocjitsu decodes the `TFE` bit but cannot append the manual's fault-status VGPR
or account for the extra returned register on fault-enabled buffer operations.
Destination-window nullification overlap remains tracked under `RDNA3-RJ-022`.

### RDNA3-RJ-075: `BUFFER_GL0_INV` and `BUFFER_GL1_INV` use the same coarse cache action

Manual evidence:

- The Chapter 9 MUBUF table says `BUFFER_GL0_INV` invalidates the shader L0
  texture cache associated with the wave, while `BUFFER_GL1_INV` invalidates
  the GL1 cache associated with the wave's VMID at `rdna3/README.md:3238`
  through `:3244`.

Rocjitsu evidence:

- `execute_buffer_gl0_inv_mubuf()` and `execute_buffer_gl1_inv_mubuf()` both
  invalidate the CU vector L1 and flush all L2 state for the process at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:29`
  through `:43`.

Impact:

The two cache invalidation opcodes are functionally conflated and over-flush
relative to the manual's distinct L0-versus-GL1 scopes, so cache-sensitive tests
cannot distinguish their documented behavior.

### RDNA3-RJ-076: Ordinary MIMG image operations execute as no-ops

Manual evidence:

- Chapter 10 defines MIMG load, store, atomic, query, MSAA, sample, gather,
  TFE/LWE, D16/A16/G16, address, resource, sampler, data-format, and dependency
  behavior at `rdna3/README.md:3571` through `:4079`.
- Chapter 16.18 defines the corresponding `IMAGE_*` opcodes and atomic
  pseudocode at `rdna3/README.md:24770` through `:25207`.

Rocjitsu evidence:

- Generated RDNA3 image load and store execute bodies only cast away the wavefront
  and state that the image path is not implemented, for example
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/mimg.cpp:32` through `:194`.
- Generated image atomic, query, sample, gather, and G16 sample bodies also do
  nothing, for example atomic/query bodies at `mimg.cpp:213` through `:447` and
  sample/gather bodies at `mimg.cpp:518` through `:1583`.
- The code generator explicitly emits these image-pipeline stubs for
  `image_load`, `image_store`, `image_atomic`, `image_sample`, `image_query`, and
  `image_bvh` at `lib/python/amdisa/codegen/_generator.py:4241` through `:4256`.

Impact:

Legal MIMG loads, stores, atomics, queries, samples, and gathers decode but have
no architectural effect. This loses memory transactions, return data, format
conversion, sampler behavior, atomic read-modify-write behavior, and wait-counter
effects.

### RDNA3-RJ-077: MIMG operand windows do not track `DMASK`, `TFE`/`LWE`, `DIM`, `A16`, or NSA semantics

Manual evidence:

- Section 10.1 says `DMASK` selects the number and order of returned or stored
  components, `TFE` and `LWE` append status data after the data return, and D16
  and A16 change packed data/address component handling at `rdna3/README.md:3587`
  through `:3675`.
- Sections 10.1.5 through 10.4 define NSA grouping and per-op/per-DIM address and
  data VGPR usage at `rdna3/README.md:3676` through `:3904`.

Rocjitsu evidence:

- RDNA3 machine structs preserve the raw MIMG and `MIMG_NSA1` fields, including
  `nsa`, `dim`, `dmask`, `a16`, `d16`, `tfe`, `lwe`, `ssamp`, and NSA address
  fields at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/machine_insts.h:165`
  through `:188` and `:253` through `:280`.
- Generated MIMG constructors bind fixed-width operands from the mnemonic class,
  such as `IMAGE_LOAD` with 128-bit `VDATA` and 128-bit `VADDR` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/mimg.cpp:20` through `:29`,
  `IMAGE_GET_RESINFO` with 128-bit `VDATA` and 32-bit `VADDR` at `:433` through
  `:444`, and sampler forms with fixed `VADDR` widths at `:503` through `:529`.
- Searches of the generated MIMG constructors and execution bodies found no path
  that resizes source or destination operand windows from `inst_.dmask`,
  `inst_.tfe`, `inst_.lwe`, `inst_.dim`, `inst_.a16`, or the NSA grouping fields.

Impact:

Even before full image execution exists, def-use metadata and any future generic
operand walker cannot tell the precise address/data/status VGPR window for a
concrete MIMG instruction. The manual's dynamic destination size for `DMASK` and
TFE/LWE status is therefore not represented.

### RDNA3-RJ-078: BVH MIMG execution is empty or unimplemented

Manual evidence:

- Section 10.9 and Chapter 16.18 define `IMAGE_BVH_INTERSECT_RAY` and
  `IMAGE_BVH64_INTERSECT_RAY`, including return data for box and triangle nodes,
  32-bit versus 64-bit node pointers, A16 packed ray-direction forms, NSA groups,
  and encoding restrictions at `rdna3/README.md:4081` through `:4180` and
  `rdna3/README.md:24973` through `:25095`.

Rocjitsu evidence:

- `ImageBvhIntersectRayMimg::execute_impl()` calls
  `amdgpu::execute_image_bvh_intersect_ray_mimg()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/mimg.cpp:481` through `:482`,
  but that shared helper is an empty inline function at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:333`
  through `:335`.
- `ImageBvh64IntersectRayMimg::execute_impl()` throws `util::UnimplementedInst`
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/mimg.cpp:498` through
  `:500`.

Impact:

The 32-bit BVH opcode silently produces no ray-intersection results, while the
64-bit form traps as unimplemented. Neither path models the manual's ray-node
fetch, result packing, A16 layout, or BVH encoding restrictions.

### RDNA3-RJ-079: `GLOBAL_ATOMIC_CSUB_U32` is missing from RDNA3 generated decode

Manual/XML evidence:

- Chapter 11 lists `GLOBAL_ATOMIC_CSUB_U32` as the GLOBAL-only opcode 55 and says
  `GLC` must be set to 1 at `rdna3/README.md:4316`; Chapter 15.9 lists it in the
  GLOBAL opcode table at `rdna3/README.md:7195`.
- Chapter 16.20 defines the clamped subtract operation and return value at
  `rdna3/README.md:26227` through `:26240`.
- XML also has `GLOBAL_ATOMIC_CSUB_U32` with opcode 55 and a natural-language
  clamp description at `amdgpu_isa_rdna3.xml:24041` through `:24086`.

Rocjitsu evidence:

- Searches of the generated RDNA3 FLAT/GLOBAL surface found no
  `GlobalAtomicCsub*` class, decoder method, opcode constant, or encoding fixture;
  the only RDNA3 `Csub` matches are for `BUFFER_ATOMIC_CSUB_U32` in `mubuf.cpp`,
  `decoder.cpp`, `opcodes.h`, and `test_encodings.h`.
- The shared RDNA3 `sub_decode_flat` table implements opcode slots 51-54 and
  56-64, but slot 55 is `decodeInvalid` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/decoder.cpp:8567` through
  `:8572`.
- The generated RDNA3 flat encoding fixture jumps from
  `flat_atomic_sub_u32` to `flat_atomic_min_i32`, with no
  `global_atomic_csub_u32` or `flat_atomic_csub_u32`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/test_encodings.h:1342`
  through `:1347`.

Impact:

A legal RDNA3 `GLOBAL_ATOMIC_CSUB_U32` encoding decodes as invalid in rocjitsu,
so kernels using the GLOBAL clamped-subtract atomic cannot execute or disassemble
through the generated RDNA3 path. `RDNA3-RJ-073` is separate and covers the
generated BUFFER CSUB arithmetic path.

### RDNA3-RJ-080: FLAT/GLOBAL/SCRATCH address and aperture handling is simplified

Manual evidence:

- Chapter 11 says flat addresses are tested per work-item against memory aperture
  base/size state before `inst_offset` is added, with address classes for normal
  global, private scratch, shared LDS, and invalid space at
  `rdna3/README.md:4218` through `:4230` and `rdna3/README.md:4336` through
  `:4343`.
- The field and format tables say FLAT uses a 12-bit unsigned offset whose MSB is
  ignored/forced zero, while GLOBAL/SCRATCH use signed 13-bit offsets, at
  `rdna3/README.md:4251` and `rdna3/README.md:7137` through `:7148`.
- Section 11.2 gives the SEG-specific global, scratch, and flat address formulas,
  including flat LDS/shared address calculation and the SADDR/SVE mode table, at
  `rdna3/README.md:4368` through `:4468`; Section 11.3 says invalid-aperture
  checks occur before offsets and other error checks after offsets at
  `rdna3/README.md:4470` through `:4483`.

Rocjitsu evidence:

- RDNA3 `flat_calculate_addresses()` sign-extends `inst.offset` as 13 bits for all
  SEG values at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/addr_calc.cpp:53` through
  `:61`, so flat offsets with bit 12 set subtract from the address instead of
  ignoring that bit.
- The helper forms the final address as `saddr_val + vaddr + offset` before
  private-aperture classification, and the private check only compares the high
  32 bits against `private_aperture_base() >> 32`, at `rdna3/addr_calc.cpp:89`
  through `:113`.
- Shared-aperture routing is performed later in `ComputeUnitCore::route_memory_inst`
  by probing the first active lane, rewriting every active lane by subtracting the
  shared aperture base, retagging the whole instruction as `LOCAL_MEM`, and
  issuing it to LDS at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:263` through `:281`.

Impact:

Rocjitsu can compute different addresses from hardware for flat offsets, aperture
crossings, invalid apertures, and mixed-lane flat operations. A GLOBAL access that
falls into the shared aperture can also be routed to LDS by the generic
`GLOBAL_MEM` routing path instead of producing the manual's GLOBAL-to-LDS
`MEMVIOL`. The dual FLAT wait-counter part of the same manual section is already
tracked under `RDNA3-RJ-044`.

### RDNA3-RJ-081: Scratch/private memory uses a contiguous per-lane layout instead of scratch swizzle

Manual evidence:

- Section 11.2 defines scratch addressing as
  `SCRATCH_BASE + SWIZZLE(offset, ThreadID)` for SV/SS/SVS/ST modes, says SGPR and
  VGPR offsets are unsigned 32-bit byte offsets, and defines the hard-coded
  scratch swizzle formula using wave32 or wave64 stride at
  `rdna3/README.md:4394` through `:4429`.
- The flat-private address formula also maps flat private addresses through
  `FLAT_SCRATCH` plus the scratch swizzle at `rdna3/README.md:4459` through
  `:4466`.

Rocjitsu evidence:

- Direct scratch SEG handling computes addresses as
  `scratch_base + lane * scratch_lane_size + vaddr + saddr_val + offset` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/addr_calc.cpp:62` through
  `:85`.
- Flat private-aperture handling uses the same contiguous lane-slice model after
  the aperture check at `rdna3/addr_calc.cpp:95` through `:113`.
- Dispatch scratch setup allocates a per-wave region of
  `private_segment_fixed_size * wf_size`, stores `wave_scratch` as the wave
  scratch base, and records `private_segment_fixed_size` as `scratch_lane_size` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:323` through `:360`.

Impact:

Private scratch accesses are internally consistent for rocjitsu's simplified
allocation model, but they do not match the documented hardware swizzle address
layout. Code or probes that depend on the physical scratch address pattern, flat
private aperture translation, or cross-lane scratch interleaving will diverge.

### RDNA3-RJ-082: FLAT/GLOBAL/SCRATCH operand metadata misses RDNA null and scratch `SVE` address modes

Manual evidence:

- The Chapter 11 field table says `ADDR` is a 64-bit address for FLAT and
  GLOBAL/SCRATCH when `SADDR` is NULL, but becomes a 32-bit offset for
  GLOBAL/SCRATCH when `SADDR` participates; for SCRATCH, `ADDR` supplies an
  offset only if `SVE=1` at `rdna3/README.md:4241` through `:4253`.
- Chapter 15.9 says `SADDR` is disabled by `NULL` or `0x7f`, and repeats that
  Scratch uses `SVE` to decide whether a VGPR offset participates, at
  `rdna3/README.md:7146` through `:7155`; Section 11.2 lists the ST/SS/SV/SVS
  modes at `rdna3/README.md:4439` through `:4456`.

Rocjitsu evidence:

- Runtime address calculation treats both `0x7c` and `0x7f` as disabled SADDR
  encodings at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/addr_calc.cpp:22` through
  `:25`.
- The generator's flat constructor patch-up checks only `inst_.saddr != 0x7F`
  when deciding whether to add an `saddr` operand, and it always rewrites Scratch
  `addr` to a 32-bit VGPR operand without considering `SVE`, at
  `lib/python/amdisa/codegen/_generator.py:6246` through `:6279`; generated
  `FlatLoadU8Flat` shows the same pattern at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/flat.cpp:26` through `:47`.
- `Flat::implicit_uses()` likewise skips only `0x7f`, not RDNA's `NULL` selector
  value 124, before adding a scratch/global SADDR register use at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/encodings.cpp:421` through
  `:428`.

Impact:

Decoded/disassembled metadata can report a 32-bit VGPR offset and a null SADDR
source for `GLOBAL` encodings where hardware treats `ADDR` as a 64-bit VGPR
address, and can report an unused Scratch VGPR source when `SVE=0`. Execution
mostly uses the runtime `has_saddr()`/`SVE` checks, so this is primarily a
def-use, disassembly, and instrumentation metadata gap.

### RDNA3-RJ-083: DS ADDTID address calculation does not use the documented `M0 + TID*4` form

Manual evidence:

- Chapter 12 says the `M0` register is used by `ds_load_addtid_b32` and
  `ds_write_addtid_b32` as a byte address, and is otherwise not used for most
  LDS-indexed operations, at `rdna3/README.md:4793` through `:4795`.
- The ADDTID address formula is
  `LDS_BASE + {InstOffset1, InstOffset0} + TID(0..63)*4 + M0`, with no VGPR
  address contribution and DWORD-aligned `M0`, at `rdna3/README.md:4833`
  through `:4840`.

Rocjitsu evidence:

- `DS_STORE_ADDTID_B32` has no explicit `ADDR` operand, but its execution path
  calls the generic `ds_calculate_addresses()` helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/ds.cpp:4128` through
  `:4137`.
- The shared DS helper always reads the encoded `inst.addr` VGPR and computes
  `VGPR[ADDR] + {offset1,offset0} + lds_base`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:60`
  through `:75`. It does not apply `M0` or `TID*4`.
- `DS_LOAD_ADDTID_B32` uses a custom helper, but that helper interprets
  `M0[24:16]` as a per-lane stride and computes
  `lane * ds_stride_bytes + offset + lds_base` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:160`
  through `:185`. It ignores the manual's `M0[15:0]` byte offset and hardcoded
  `TID*4` term.

Impact:

Legal RDNA3 ADDTID DS instructions access the wrong LDS locations. Store ADDTID
can depend on an unused encoded `ADDR` field, while load ADDTID treats `M0` as a
stride selector rather than a byte base offset.

### RDNA3-RJ-084: DS permute/swizzle helpers bypass DS wait-counter accounting

Manual evidence:

- Chapter 12 says LDS indexed and atomic instructions use `LGKMcnt`, increment
  the counter when issued, decrement it on completion, and stay in-order with
  other LDS instructions from the same wave at `rdna3/README.md:4765` through
  `:4775`.
- The same chapter lists `DS_PERMUTE_B32` and `DS_BPERMUTE_B32` in the LDS
  indexed load/store group and then says they use LDS hardware, do not use
  memory storage, and may run without LDS allocation at `rdna3/README.md:4797`
  through `:4815` and `:4902` through `:4915`.
- Chapter 16.15 defines `DS_SWIZZLE_B32` as a DS dword swizzle that does not
  read or write DS memory banks and returns zero for invalid source threads at
  `rdna3/README.md:22587` through `:22595`.

Rocjitsu evidence:

- Generated ordinary DS memory operations allocate `VectorMemState`, set
  `wait_counter_type = WaitCounterType::DSCNT`, and run through the local memory
  pipeline. `DS_STORE_ADDTID_B32` is one example at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/ds.cpp:4128` through
  `:4147`.
- `DS_SWIZZLE_B32`, `DS_PERMUTE_B32`, and `DS_BPERMUTE_B32` constructors do not
  set `MEMORY_OP`; they call immediate shared helpers directly at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/ds.cpp:1747` through `:1759`
  and `:4164` through `:4195`.
- Those shared helpers read and write VGPR regions directly and never create a
  `VectorMemState` or set `DSCNT`/`LGKMcnt` participation at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:59`
  through `:90`, `:196` through `:227`, and `:265` through `:296`.

Impact:

Programs that rely on `S_WAITCNT LGKMcnt`/DS-count ordering around LDS
cross-lane operations can observe rocjitsu completing permute/swizzle results
immediately, without the documented DS wait-counter dependency.

### RDNA3-RJ-085: `DS_WRAP_RTN_B32` is decoded but unimplemented

Manual/XML evidence:

- Chapter 16.15 defines `DS_WRAP_RTN_B32` as opcode 52 for ring-buffer wrap
  management: it reads `tmp = MEM[ADDR].u`, writes
  `tmp - DATA.u` when `tmp >= DATA.u` or `tmp + DATA2.u` otherwise, and returns
  the original value at `rdna3/README.md:22574` through `:22585`.
- The RDNA3 XML has a `DS_WRAP_RTN_B32` entry with `VDST`, `ADDR`, `DATA0`,
  `DATA1`, implicit DSMEM input/output operands, and
  `VMEM`/`ATOMIC`/`DATA_SHARE` classification at
  `amdgpu_isa_rdna3.xml:14772` through `:14823`.

Rocjitsu evidence:

- Generated RDNA3 has a `DsWrapRtnB32Ds` constructor that exposes `VDST`,
  `ADDR`, `DATA0`, and `DATA1`, but `execute_impl()` immediately throws
  `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/ds.cpp:1727` through
  `:1744`.

Impact:

Legal RDNA3 `ds_wrap_rtn_b32` programs decode but cannot execute the documented
atomic wrap update or return the pre-op LDS value.

### RDNA3-RJ-086: Float atomic RMW helpers do not implement Chapter 13 edge semantics

Manual evidence:

- Chapter 13 says float-atomic-add ignores `MODE.round` and is fixed to
  round-to-nearest-even, at `rdna3/README.md:5015` through `:5017`.
- Section 13.2 gives different denormal behavior for LDS add, LDS
  compare/min/max, cache compare/min/max, and cache add, including the rule that
  memory atomic add always flushes input denormals, at `rdna3/README.md:5021`
  through `:5053`.
- Section 13.3 defines float-atomic SNaN quieting, min/max ordering across NaN,
  infinities, denormals, and signed zeros, compare-swap's non-NaN floating
  equality with `+0 == -0`, and float-add NaN/infinity/zero special cases at
  `rdna3/README.md:5064` through `:5104`.

Rocjitsu evidence:

- Generated DS F32 atomics set `AtomicOp::CMPSWAP`, `FMIN`, `FMAX`, and `FADD`
  before entering the shared memory path at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/ds.cpp:666` through `:790`,
  and DS F64 compare/min/max forms use the same `CMPSWAP`/`FMIN`/`FMAX`
  operation IDs at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/ds.cpp:2725` through
  `:2812`.
- Generated FLAT/GLOBAL/SCRATCH F32 and buffer F32 atomics use the same
  `CMPSWAP`, `FMIN`, `FMAX`, and `FADD` operation IDs at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/flat.cpp:2294` through
  `:2452` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/mubuf.cpp:2418`
  through `:2552`.
- The shared memory pipeline treats only `FADD`/`FMIN`/`FMAX` as FP operations
  and implements them with host `+`, `std::fmin`, and `std::fmax` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:286` through `:296`
  and `:321` through `:343`. It does not apply the Chapter 13 denormal table,
  SNaN-to-QNaN conversion, NaN payload selection, or signed-zero ordering.
- `AtomicOp::CMPSWAP` remains on the integer path and compares raw integer
  values at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:250`
  through `:253` and `:345` through `:349`,
  so FP compare-store does not implement the manual's non-NaN floating equality
  and `+0 == -0` rule.

Impact:

RDNA3 float atomic edge cases can produce host-library or raw-integer results
instead of the manual result. Examples include cache F32 add with denormal
inputs, LDS F32/F64 min/max involving SNaN/QNaN or `-0`/`+0`, and F32/F64
compare-store where the memory value and compare source are opposite-signed
zeros.

### RDNA3-RJ-087: GWS and ordered-count instructions are decoded but unimplemented

Manual evidence:

- Section 13.4 says GWS instructions use `LGKMcnt`, must be issued as a
  single-instruction clause with surrounding `S_WAITCNT LGKMcnt==0`, and have
  NACK/replay restrictions on source and destination VGPR aliasing at
  `rdna3/README.md:5106` through `:5122`.
- Section 13.4.2 says GDS/GWS instructions are single-lane, have special
  `EXEC==0` behavior for ordered count and selected GWS opcodes, and still use
  LGKMcnt-visible traffic for other `EXEC==0` GDS/GWS opcodes at
  `rdna3/README.md:5124` through `:5132`.
- Sections 13.4.3 and 13.4.4 define ordered-count queueing, field
  reinterpretation, append/consume full-`EXEC` counting, GWS resource state,
  `M0[21:16]` resource virtualization, clamping/NACK behavior, and
  semaphore/barrier pseudocode at `rdna3/README.md:5134` through `:5251`.

Rocjitsu evidence:

- Generated RDNA3 constructors exist for `ds_gws_sema_release_all`,
  `ds_gws_init`, `ds_gws_sema_v`, `ds_gws_sema_br`, `ds_gws_sema_p`, and
  `ds_gws_barrier`, but their `execute_impl()` bodies unconditionally throw
  `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/ds.cpp:806` through `:881`.
- `DS_ORDERED_COUNT` similarly has a generated constructor but its execution
  body throws unconditionally at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/ds.cpp:2028` through
  `:2041`.
- These constructors do not set `MEMORY_OP`, do not create `VectorMemState`, and
  do not set `WaitCounterType::DSCNT`; searches found no RDNA3 runtime model
  for GWS resource state, ordered-count queues, GWS NACK/replay, or the
  Chapter 13.4 `EXEC==0` exceptions.

Impact:

Legal RDNA3 GWS and ordered-count programs can decode and disassemble, but they
cannot execute, cannot participate in the documented LGKM/DS wait-counter
contract, and cannot model ordered append/consume, semaphore, barrier, or
NACK/replay behavior. This is distinct from the broader GDS backing-store gap in
`RDNA3-RJ-006`.

### RDNA3-RJ-088: `EXP` executes as a no-op and does not model export state or `EXPCNT`

Manual evidence:

- Chapter 14 says exports copy enabled `EXEC` lanes from VGPRs to graphics
  export buffers, each target can be exported only once, `DONE` identifies the
  last pixel or position export, primitive exports must set `DONE`, and
  `ROW_EN` uses `M0` for mesh position/primitive rows at `rdna3/README.md:5253`
  through `:5288`.
- Section 14.1 defines pixel export obligations, depth/MRT0/sample-mask
  ordering, and dual-source blend adjacency plus lane-mask transforms at
  `rdna3/README.md:5290` through `:5317`.
- Section 14.3 says export issue increments `EXPCNT`, completion later reads
  `EXEC` and VGPR data and decrements `EXPCNT`, `S_WAITCNT` on `EXPCNT` protects
  those sources, same-type exports complete in order, and `STATUS.SKIP_EXPORT`
  makes exports act as NOPs at `rdna3/README.md:5325` through `:5333`.
- Chapter 16.19 says pixel shaders must export to a color, depth, or `NULL`
  target with the `VM` bit set to communicate the pixel-valid mask, only one
  pixel export type may set `DONE`, vertex shaders need position plus parameter
  exports, and the final position export must set `DONE` at
  `rdna3/README.md:25211` through `:25213`.

Rocjitsu evidence:

- Generated RDNA3 `ExpExp` exposes `TGT` and four VGPR sources, but does not add
  the XML/manual implicit `EXEC` or `M0` operands to its source list at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/exp.cpp:19` through `:33`.
- `ExpExp::execute_impl()` ignores the wavefront and returns as a compute
  simulation no-op at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/exp.cpp:35` through `:37`.
- The RDNA3 `Exp` encoding base does not override implicit-use handling or
  install export-specific state at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/encodings.h:577` through
  `:582`.
- Searches found `EXPCNT` counter storage and wait instructions, but no RDNA3
  export producer path that increments/decrements `WaitCounterType::EXPCNT`, no
  export-buffer state, no target-once or `DONE` tracking, no dual-source
  lane-mask handling, and no `STATUS.SKIP_EXPORT` execution branch.

Impact:

Graphics shaders that rely on RDNA3 exports can decode but do not produce
position, primitive, color, Z, or dual-source blend output in rocjitsu.
Pixel valid-mask/`VM`, `DONE`, parameter/position-output, and `NULL`-target
contracts also have no execution effect. `S_WAITCNT EXPCNT` cannot protect
export source VGPRs or `EXEC` from overwrite because `EXP` never creates an
outstanding export event.

### RDNA3-RJ-089: Split SOPK wait instructions ignore SGPR threshold operands

Manual evidence:

- Chapter 16.2 defines `S_WAITCNT_VSCNT`, `S_WAITCNT_VMCNT`, and
  `S_WAITCNT_LGKMCNT` thresholds as `S0.u[5:0] + S1.u[5:0]`, with a six-bit
  comparison and no clamping on add overflow, at `rdna3/README.md:8092`
  through `:8124` and `:8145` through `:8160`.
- The same section defines `S_WAITCNT_EXPCNT` as
  `S0.u[2:0] + S1.u[2:0]`, with a three-bit comparison and no clamping, at
  `rdna3/README.md:8128` through `:8142`.
- Literal-only waits are encoded by writing `NULL` for the GPR argument, so a
  non-`NULL` SGPR operand is architecturally meaningful at
  `rdna3/README.md:8105` through `:8107`, `:8122` through `:8124`, `:8139`
  through `:8141`, and `:8158` through `:8160`.

Rocjitsu evidence:

- The RDNA3 split wait constructors keep both the `SDST`/`OPR_SDST_NULL`
  operand and the `SIMM16` operand in `src_operands_` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/sopk.cpp:417` through
  `:477`.
- Their execute bodies read only `simm16.encoding_value_` and set the wait
  target from that immediate value at `rdna3/sopk.cpp:429` through `:483`. No
  code reads the `sdst` operand low bits, performs the documented low-bit
  addition, or masks the `EXPCNT` form to a three-bit comparison.

Impact:

`s_waitcnt_vscnt`, `s_waitcnt_vmcnt`, `s_waitcnt_expcnt`, and
`s_waitcnt_lgkmcnt` behave correctly only for literal-only/`NULL` uses where
the SGPR contribution is zero. Programs using the legal SGPR-plus-immediate
forms can wait on the wrong threshold in rocjitsu.

### RDNA3-RJ-090: Debug-condition SOPP branches always fall through

Manual evidence:

- Chapter 3.4.1 defines `STATUS.CDBG_USER` and `STATUS.CDBG_SYS` as
  conditional-debug bits that can be used in conditional branches at
  `rdna3/README.md:940` through `:941`.
- Chapter 16.5 defines `S_CBRANCH_CDBGSYS`, `S_CBRANCH_CDBGUSER`,
  `S_CBRANCH_CDBGSYS_OR_USER`, and `S_CBRANCH_CDBGSYS_AND_USER` as PC-relative
  conditional branches on those status bits at `rdna3/README.md:9851` through
  `:9907`.

Rocjitsu evidence:

- The four RDNA3 debug-branch constructors dispatch to shared helpers at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/sopp.cpp:361` through
  `:410`, but unlike the SCC/VCC/EXEC branch constructors they do not set
  `flags_ |= COND_BRANCH`.
- The shared helpers for `execute_s_cbranch_cdbgsys_sopp`,
  `execute_s_cbranch_cdbguser_sopp`,
  `execute_s_cbranch_cdbgsys_or_user_sopp`, and
  `execute_s_cbranch_cdbgsys_and_user_sopp` are empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:997`
  through `:1011`.

Impact:

Programs that use the RDNA3 conditional-debug branch opcodes always fall
through in rocjitsu, and branch metadata users do not see them as conditional
branches.

### RDNA3-RJ-091: `V_PIPEFLUSH` executes as an empty helper

Manual/XML evidence:

- Chapter 16.8 defines compact `V_PIPEFLUSH` opcode 27 as flushing the VALU
  destination cache at `rdna3/README.md:11225` through `:11227`.
- The VOP3 alias has the same wording at `rdna3/README.md:16041` through
  `:16045`.
- XML carries the operandless `V_PIPEFLUSH` instruction and describes flushing
  the vector ALU pipeline through the destination cache at
  `amdgpu_isa_rdna3.xml:61283` through `:61297`.

Rocjitsu evidence:

- Generated RDNA3 VOP1 and VOP3 constructors expose `V_PIPEFLUSH` as operandless
  instructions at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/vop1.cpp:3143` through
  `:3148` and `rdna3/vop3.cpp:1717` through `:1722`.
- Both generated execute paths delegate to shared helpers at
  `rdna3/vop1.cpp:3206` and `rdna3/vop3.cpp:1738`.
- The shared helpers are empty no-ops at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:16658`
  and `:16661`.

Impact:

Rocjitsu accepts and executes `V_PIPEFLUSH`, but it does not model the manual's
pipeline or destination-cache flush side effect. Any program depending on that
ordering/cache behavior is treated like it executed an ordinary no-op.

### RDNA3-RJ-092: Compact VOPC VCC/EXEC result masks are missing from metadata

Manual/XML evidence:

- Chapter 16.9 says VOPC compares produce one result bit per lane into `VCC` or
  `EXEC` at `rdna3/README.md:12334`.
- The detailed `V_CMP_LT_F16` definition stores to `D0.u64[laneId]` and says
  `D0 = VCC in VOPC encoding` at `rdna3/README.md:12427` through `:12438`.
- The detailed `V_CMPX_LT_F16` definition stores to `EXEC.u64[laneId]` and says
  to write only `EXEC` with `SDST` set to `EXEC_LO` at `rdna3/README.md:13832`
  through `:13842`.
- XML carries the VOP3 alias result as an `OPR_SREG`/`VDST` output for
  `V_CMP_LT_F16` at `amdgpu_isa_rdna3.xml:120250` through `:120276`, and the
  `CMPX` result as `OPR_EXEC` plus implicit `OPR_SDST_EXEC` for
  `V_CMPX_LT_F16` at `amdgpu_isa_rdna3.xml:142262` through `:142293`.

Rocjitsu evidence:

- Generated compact `VCmpLtF16Vopc` publishes only `src0` and `vsrc1` and sets
  `num_dst_ = 0` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/vopc.cpp:134` through
  `:149`.
- Generated compact `VCmpxFF16Vopc` likewise publishes only `src0` and `vsrc1`
  and sets `num_dst_ = 0` at `rdna3/vopc.cpp:10404` through `:10419`.
- The `Vopc` base class declares no `implicit_defs()` override at
  `rdna3/encodings.h:427` through `:452`, and its constructor only records
  common encoding fields at `rdna3/encodings.cpp:149` through `:157`; the base
  `Instruction::implicit_defs()` hook is empty at
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:221` through `:222`.
- Execution still mutates hidden masks: shared non-`CMPX` helpers write
  `wf.set_vcc(vcc)`, for example `execute_v_cmp_lt_f16_vopc` at
  `shared/execute_shared.h:6151` through `:6165`, while generated `CMPX`
  execute bodies write `EXEC`, for example `rdna3/vopc.cpp:10498` through
  `:10512`.

Impact:

Def-use, liveness, scheduling, and legality passes that rely on generated
operand metadata cannot see compact VOPC compare definitions of `VCC` or
`EXEC`. A compare result can therefore be moved, removed, or consumed out of
order even though the runtime execute path mutates the live wavefront masks.

### RDNA3-RJ-093: VOP3 compare-DPP paths do not zero masked scalar/EXEC result bits

Manual/XML evidence:

- Chapter 7.7 says DPP may be used with VOP3, excluding the listed 64-bit and
  special opcodes, while VOPC 64-bit opcodes are excluded separately at
  `rdna3/README.md:2741` through `:2762`.
- The same DPP section says `V_CMP` and `V_CMPX` write the full mask, not a
  partial mask; with DPP and `bound_ctrl=0`, disabled `EXEC` lanes receive zero,
  and `FI=1` still does not turn on inactive lanes for `V_CMPX` at
  `rdna3/README.md:2764`.
- DPP16 row and bank masks say that for VOPC, disabled-lane SGPR/VCC bits
  receive zero at `rdna3/README.md:2779` through `:2788`.
- Chapter 16.12 says all non-`VOP3SD` VALU instructions use VOP3 at
  `rdna3/README.md:15659` through `:15688`; representative `CMPX` definitions
  write only `EXEC` with `SDST` set to `EXEC_LO`, for example
  `V_CMPX_T_U32` at `rdna3/README.md:21631` through `:21641`.
- XML carries VOP3 compare-result operands as scalar destinations, for example
  `V_CMP_LT_F16` at `amdgpu_isa_rdna3.xml:120250` through `:120276`, and
  `CMPX` forms as `OPR_EXEC` plus implicit `OPR_SDST_EXEC`, for example
  `V_CMPX_F_U64` at `amdgpu_isa_rdna3.xml:165593` through `:165624`, so the
  result is a scalar/EXEC mask rather than a VGPR destination.

Rocjitsu evidence:

- Generated RDNA3 VOP3 compare constructors accept DPP8 and DPP16 source-0
  marker forms; for example `VCmpLtF16Vop3` loads the DPP suffix state at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/vop3.cpp:20086` through
  `:20102`.
- `VCmpLtF16Vop3::execute_impl` computes a full scalar compare mask across
  active lanes and writes it to the scalar destination before computing a DPP
  write mask at `rdna3/vop3.cpp:20123` through `:20159`; the later mask path
  restores a VGPR destination at `:20160` through `:20167`, not scalar
  compare-result bits.
- `VCmpxFU64Vop3::execute_impl` similarly accepts DPP suffix state, calls
  `wf.set_exec(result)`, and then runs a DPP restore path that writes a VGPR
  indexed by `VDST` at `rdna3/vop3.cpp:35600` through `:35675`.
- The related compact VOPC metadata issue is tracked separately in
  `RDNA3-RJ-092`; this finding covers VOP3-encoded compare execution.

Impact:

RDNA3 VOP3 `V_CMP*`/`V_CMPX*` encodings with DPP can leave row/bank-masked or
inactive-lane scalar/EXEC result bits set according to the comparison instead
of forcing those bits to zero. The generic old-destination restore is VGPR-only,
so it cannot implement the scalar mask semantics required for compare
instructions.

### RDNA3-RJ-094: DPP16 inactive-source lanes ignore `BOUND_CTRL=0` write-disable

Manual/XML evidence:

- Chapter 7.7 defines `BOUND_CTRL` as the selector between writing zero and
  disabling the write when a DPP source lane is invalid or out of range at
  `rdna3/README.md:2791`; `FI=0` says inactive source lanes use `bound_ctrl`,
  while `FI=1` fetches inactive lanes but still uses `bound_ctrl` for
  out-of-range lanes at `rdna3/README.md:2792`.
- The BC/FI table explicitly says `BC=0/FI=0` disables the destination write
  for both out-of-range source lanes and in-range but disabled source lanes at
  `rdna3/README.md:2806` through `:2815`.
- XML's DPP16 field records also describe `BOUND_CTRL=0` as clearing
  write-enable for invalid shared data and `FI=0` as reading zero from inactive
  lanes at `amdgpu_isa_rdna3.xml:5507` through `:5538`.

Rocjitsu evidence:

- The shared DPP16 destination write mask only accounts for row/bank masks and
  out-of-range permutations; it has no `EXEC`/inactive-source input at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h:239`
  through `:269`.
- `dpp_read()` receives `exec_mask`, but for `FI=0` it returns zero for an
  inactive source lane regardless of `bound_ctrl` at
  `shared/dpp_sdwa_ops.h:292` through `:305`; `apply_dpp()` passes the current
  `EXEC` mask into this helper at `:324` through `:340`.
- Generated DPP users restore the old VGPR destination only for lanes excluded
  by `dpp_write_mask()`. For example, `VCndmaskB32Vop2::execute_impl` snapshots
  the old destination, applies DPP, executes the ALU op, then restores only
  lanes not present in `dpp_write_mask()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/vop2.cpp:57` through
  `:128`.

Impact:

For DPP16 instructions whose active destination lane reads from an inactive
source lane with `BC=0/FI=0`, rocjitsu supplies zero to the ALU and writes the
result. The manual requires that lane's destination write to be disabled, so the
old VGPR value should survive.

### RDNA3-RJ-095: Compact VOPC DPP paths preserve masked `VCC`/`EXEC` bits instead of zeroing them

Manual/XML evidence:

- Chapter 7.7 says `V_CMP`/`V_CMPX` write a full mask under DPP, and with
  `bound_ctrl=0` lanes that would otherwise not write receive zero; `FI=1`
  still does not turn on inactive lanes for `V_CMPX` at
  `rdna3/README.md:2764`.
- The DPP16 row and bank mask descriptions say that for VOPC, disabled
  SGPR/VCC bits receive zero at `rdna3/README.md:2779` through `:2788`.
- XML's VOPC DPP16 encoding description says compact compares respect `EXEC`
  and zero the VCC result bit when `EXEC[laneId]` is zero, and the same encoding
  records row/bank/bound/FI fields at `amdgpu_isa_rdna3.xml:11776` through
  `:11845`.

Rocjitsu evidence:

- `VCmpFF16Vopc::execute_impl` snapshots old `VCC`, computes a DPP write mask,
  executes the compare, and then merges masked-off bits from the old `VCC`
  value rather than forcing them to zero at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/vopc.cpp:58` through
  `:121`.
- `VCmpxFF16Vopc::execute_impl` does the same for both `VCC` and `EXEC`:
  it snapshots old masks, writes `EXEC`, and merges masked-off `VCC`/`EXEC`
  bits back from the old values at `rdna3/vopc.cpp:10439` through `:10512`.
- Shared compact VOPC compare helpers do produce a fresh zero-initialized mask
  for ordinary inactive destination lanes, for example
  `execute_v_cmp_lt_f16_vopc` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:6151`
  through `:6165`; the generated DPP merge overwrites that full-mask behavior
  for row/bank/out-of-range lanes.

Impact:

Compact RDNA3 `V_CMP*`/`V_CMPX*` DPP instructions can leave stale `VCC` or
`EXEC` bits set for row/bank-masked lanes and `BOUND_CTRL=0` invalid-source
lanes. Control flow or later scalar-mask consumers can observe a lane as active
or true where the ISA requires a zero result bit.

### RDNA3-RJ-096: WMMA constructors accept illegal literal-extension and DPP source forms

Manual/XML evidence:

- Chapter 7.9 says WMMA is encoded with VOP3P, lists only the six matrix
  instructions, and says inline constants can only be used for the C matrix at
  `rdna3/README.md:2860` through `:2883` and `:2893`.
- Chapter 7.7's DPP support table explicitly lists WMMA under the VOP3P
  "NO DPP" set at `rdna3/README.md:2757`.
- XML records the WMMA entries with VGPR-only A/B operands and
  `OPR_SRC_VGPR_OR_INLINE` C operands, with no WMMA `VOP3P_INST_LITERAL`,
  DPP8, or DPP16 encodings, at `amdgpu_isa_rdna3.xml:119685` through
  `:119976`.

Rocjitsu evidence:

- The generated RDNA3 WMMA constructors all rewrite `SRC0 == 255`,
  `SRC1 == 255`, and `SRC2 == 255` into `OPR_SIMM32` operands, even though A/B
  are VGPR-only and WMMA has no XML literal-extension encoding. Representative
  ranges are `VWmmaF3216x16x16F16Vop3p` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/vop3p.cpp:2155` through
  `:2180`, `VWmmaF3216x16x16Bf16Vop3p` at `:2246` through `:2269`,
  `VWmmaF1616x16x16F16Vop3p` at `:2333` through `:2358`,
  `VWmmaBf1616x16x16Bf16Vop3p` at `:2423` through `:2447`,
  `VWmmaI3216x16x16Iu8Vop3p` at `:2511` through `:2536`, and
  `VWmmaI3216x16x16Iu4Vop3p` at `:2608` through `:2632`.
- The same constructors accept DPP8 and DPP16 source-0 selector forms and store
  DPP suffix state, for example at `rdna3/vop3p.cpp:2181` through `:2197`,
  `:2270` through `:2286`, `:2359` through `:2375`, `:2448` through `:2464`,
  `:2537` through `:2553`, and `:2633` through `:2649`.
- The execute paths then compute WMMA source bases from `src0.encoding_value_`
  and `src1.encoding_value_` directly, for example at `rdna3/vop3p.cpp:2224`
  through `:2227` and `:2588` through `:2590`, so illegal literal or DPP forms
  are not diagnosed as invalid WMMA encodings before execution.

Impact:

Invalid RDNA3 WMMA encodings can decode, disassemble, and execute as if they
were supported source forms. That can hide assembler/decoder legality bugs and
produce misleading results for illegal A/B literals, illegal C literal
extensions, or illegal DPP suffixes.

### RDNA3-RJ-097: WMMA F16/BF16 `NEG`/`NEG_HI` semantics are incomplete

Manual/XML evidence:

- Chapter 7.9 says F16/BF16 WMMA repurposes `NEG[1:0]` for SRC1/SRC0 low-half
  negation, `NEG_HI[1:0]` for SRC1/SRC0 high-half negation, and
  `{NEG_HI[2], NEG[2]}` for SRC2 `{ABS, NEG}` at `rdna3/README.md:2872`.
- The same paragraph says integer IU8/IU4 WMMA uses only `NEG[1:0]` for A/B
  signedness and requires `NEG[2]` plus `NEG_HI[2:0]` to be zero at
  `rdna3/README.md:2872`.

Rocjitsu evidence:

- The F32-output F16/BF16 WMMA execute paths pass plain `extract_f16` or
  `extract_bf16` functions for A/B and only forward the C modifier produced by
  `wmma_c_modifier(inst_.neg, inst_.neg_hi)`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/vop3p.cpp:2224` through
  `:2227` and `:2313` through `:2316`.
- The F16/BF16-output WMMA execute paths also pass plain A/B extractors, but do
  not pass any C modifier into `exec_gfx11_wmma_f16` or
  `exec_gfx11_wmma_bf16`, at `rdna3/vop3p.cpp:2402` through `:2405` and
  `:2491` through `:2494`.
- The shared packed16 WMMA helper has no C-modifier parameter and initializes a
  floating accumulator directly from `const_acc` or `read_acc`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:1614` through
  `:1700`. The only shared C-modifier helper applies `{ABS, NEG}` to C in the
  F32-output path at `shared/mma_exec.h:100` through `:110` and `:1571`
  through `:1611`.
- The integer WMMA paths do use `NEG[1:0]` to choose signed or unsigned A/B
  extractors, but do not reject nonzero modifier bits that the manual says must
  be zero, at `rdna3/vop3p.cpp:2580` through `:2590` and `:2676` through
  `:2686`.

Impact:

RDNA3 WMMA F16/BF16 instructions with A/B low/high negation or packed16 C
`ABS`/`NEG` modifiers can produce unmodified results. Integer WMMA instructions
with reserved modifier bits set are accepted instead of being diagnosed as
illegal encodings.

### RDNA3-RJ-098: Dependent WMMA adjacency hazard is not tracked

Manual/XML evidence:

- Chapter 7.9 says back-to-back dependent WMMA instructions require one
  `V_NOP` or independent VALU instruction between them when the first
  instruction's D matrix is the same as or overlaps the second instruction's
  A/B matrices, at `rdna3/README.md:2895`.
- XML lists the standalone WMMA instruction entries and operand shapes at
  `amdgpu_isa_rdna3.xml:119685` through `:119976`, but does not encode this
  instruction-stream adjacency rule.

Rocjitsu evidence:

- The compute-unit path dispatches each decoded instruction directly through
  `execute_instruction(inst, *active)` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:393` through `:412`,
  and the RDNA3 compute-unit override calls the instruction's execute callback
  without stream-history validation at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.h:753` through `:756`.
- A search of the RDNA3 VM/VALU execution path found no previous-instruction
  tracking for WMMA D/A/B overlap analogous to the manual's required padding.

Impact:

Rocjitsu can execute instruction streams that RDNA3 requires software to pad
around dependent WMMA operations, so decoder/runtime tests will not expose
missing scheduling around this WMMA hazard.

### RDNA3-RJ-099: `V_DOT2_F32_BF16` converts BF16 inputs as FP16

Manual/XML evidence:

- Chapter 16.10 defines `V_DOT2_F32_BF16` as a dot product of packed BF16
  values converted to F32 before multiplication and accumulation at
  `rdna3/README.md:15404` through `:15418`.
- Section 7.5.1 also says BF16 inline constants use the upper 16 bits of an
  FP32 constant, and the DOT exception table names `DOT2_F32_BF16` as the BF16
  form at `rdna3/README.md:2652` and `:2670`.
- XML records `V_DOT2_F32_BF16` with `FMT_NUM_PK2_BF16` for SRC0/SRC1 and
  `FMT_NUM_F32` for SRC2/VDST at `amdgpu_isa_rdna3.xml:118369` through
  `:118399`.

Rocjitsu evidence:

- The generated RDNA3 `VDot2F32Bf16Vop3p` class dispatches to the shared
  `execute_v_dot2_f32_bf16_vop3p` helper at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/vop3p.cpp:1825` through
  `:1887`.
- That shared helper selects the 16-bit halves from SRC0/SRC1 but converts all
  four selected BF16 halves with `util::f16_to_f32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:10421`
  through `:10430`.
- The shared MMA code has distinct F16 and BF16 extractors and converts BF16
  halves with `util::bf16_to_f32`, for example at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:636` through
  `:650` and `:2645` through `:2655`.

Impact:

`V_DOT2_F32_BF16` produces the dot product of FP16-decoded bit patterns rather
than the manual's BF16 values. This affects both ordinary packed BF16 register
sources and BF16 inline-constant cases.

### RDNA3-RJ-100: Integer VOP3P DOT `CLMP` is missing or clamps signed results as unsigned

Manual/XML evidence:

- The RDNA3 VOP3P field table defines `CLMP` as a result clamp: signed integer
  arithmetic clamps to `[min_int, +max_int]`, and unsigned integer arithmetic
  clamps to `[0, +max_uint]`, at `rdna3/README.md:2627` through `:2628`.
- Chapter 16.10 defines `V_DOT4_I32_IU8` and `V_DOT8_I32_IU4` as signed-domain
  DOT operations whose `NEG[0]`/`NEG[1]` bits choose signed or unsigned
  interpretation for SRC0/SRC1, at `rdna3/README.md:15314` through `:15335`
  and `:15360` through `:15385`.
- The same section defines `V_DOT4_U32_U8` and `V_DOT8_U32_U4` as unsigned DOT
  operations at `rdna3/README.md:15341` through `:15356` and `:15387` through
  `:15402`.

Rocjitsu evidence:

- `execute_v_dot4_i32_iu8_vop3p` and `execute_v_dot8_i32_iu4_vop3p` use
  `NEG[0]`/`NEG[1]` for source signedness, but accumulate into `uint32_t` and,
  when `inst.inst_.clamp` is set, cast the wrapped sum to `int32_t` and force
  any negative result to zero at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:10682`
  through `:10699` and `:10803` through `:10820`.
- `execute_v_dot4_u32_u8_vop3p` and `execute_v_dot8_u32_u4_vop3p` perform
  unsigned accumulation and write the wrapped `uint32_t` result without
  inspecting `inst.inst_.clamp` at `shared/execute_shared.h:10704` through
  `:10721` and `:10825` through `:10842`.

Impact:

Signed DOT results that are valid negative `int32_t` values are clamped to zero
instead of being preserved inside the signed range, and unsigned DOT overflow
wraps instead of saturating when `CLMP==1`. This is separate from
`RDNA3-RJ-002`, which covers ordinary packed I16/U16/F16 arithmetic.

### RDNA3-RJ-101: `EXP` disassembly hides selectors, finality, and valid-mask fields

Manual/XML evidence:

- Section 15.10.1 defines `EXP` fields for `EN`, `TARGET`, `DONE`, `ROW`, and
  `VSRC0` through `VSRC3` at `rdna3/README.md:7249` through `:7271`.
- Chapter 16.19 says pixel shaders must use a color, depth, or `NULL` target
  with `VM` set for the pixel-valid mask, and defines the pixel/vertex `DONE`
  obligations at `rdna3/README.md:25211` through `:25213`.
- XML `ENC_EXP` records `EN`, `DONE`, `ROW_EN`, `TGT`, and `VSRC0` through
  `VSRC3`, and its prose shows disabled channels as `off`, at
  `shared/machine-readable-isa/isa/amdgpu_isa_rdna3.xml:4733` through `:4865`.

Rocjitsu evidence:

- `ExpMachineInst` stores `en`, `done`, and `row_en`, but bit 12 is generated as
  `pad_12` rather than a `vm` field at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/machine_insts.h:190`
  through `:197`.
- `ExpBuilderFields` and `build_exp()` expose `en`, `done`, and `row_en`, but
  no `vm` field at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/builders.h:458` through
  `:481`.
- `ExpExp` always constructs four `OPR_VGPR` sources and one `OPR_TGT`
  destination, regardless of `EN`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/exp.cpp:19` through `:33`.
- Generic `Instruction::disassemble()` prints only destination operands, source
  operands, and `build_modifiers()` output at
  `lib/rocjitsu/src/rocjitsu/isa/instruction.h:239` through `:255`, while
  RDNA3 `Exp` does not override `build_modifiers()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/encodings.h:577` through
  `:582`.
- The generated `OPR_TGT` enum and formatter name MRT0-7, MRTZ, POS0-4, PRIM,
  and dual-source blend targets, but not a `NULL` target at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/operand_types.h:275`
  through `:293` and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/operand.cpp:485` through
  `:520`.

Impact:

Decoded RDNA3 `EXP` text and operand metadata lose the enabled-channel mask
(`off` channels), `DONE`, `ROW_EN`, the Chapter 16.19 `VM` valid-mask bit, and
the `NULL` export target name. Tools consuming rocjitsu disassembly or def-use
metadata cannot distinguish final exports, row exports, disabled channels, or
valid-mask exports even before considering the broader no-op execution gap in
`RDNA3-RJ-088`.

## No-Gap Notes

- Chapter 15.1 scalar/control decode narrow match: RDNA3 imports shared
  field-identical SOP1/SOPC/SOPP/SOPK/SOP2 structs at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/machine_insts.h:18` through
  `:26`, and those shared structs preserve the Chapter 15 bit positions at
  `shared/machine_insts_scalar.h:28` through `:90`. The generated encoding
  classes size ordinary and literal scalar encodings at `rdna3/encodings.cpp:24`
  through `:100`, including the SOPK implied-literal case for
  `S_SETREG_IMM32_B32`.
- Chapter 16.3 SOP1 execution boundary: generated RDNA3 SOP1 constructors and
  shared helpers cover ordinary moves, conditional moves, bit-reverse/count,
  bitset, bitreplicate, quadmask/WQM, and EXEC save/write forms; existing gaps
  continue to cover relative SGPR indexing, direct PC alignment/trap return,
  message-return timing, and hidden implicit SCC metadata.
- Chapter 16.4 SOPC execution narrow match: generated SOPC compare and
  bit-compare forms dispatch to shared helpers that update SCC and mask bit
  indexes to five or six bits at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:897`
  through `:925` and `:1116` through `:1247`. The remaining SCC concern is the
  metadata gap in `RDNA3-RJ-047`.
- Chapter 16.5 SOPP execution boundary: ordinary SCC/VCC/EXEC branches,
  monolithic waitcnt, barrier entry, and `S_ENDPGM*` termination have executable
  paths; existing gaps cover no-op control side effects, clauses, messages,
  wait producers, wait-event/export readiness, debug branches, trap state, and
  missing producer coverage for end-program implicit waits.
- Chapter 16.10 ordinary VOP3P packed-operation boundary: the generated RDNA3
  packed I16/U16/F16 helpers dispatch to shared code that implements the
  manual's per-half arithmetic, OPSEL/OPSEL_HI source selection, and documented
  shift-count halves for representative rows such as `V_PK_ADD_F16`,
  `V_PK_ADD_I16`, `V_PK_LSHLREV_B16`, and `V_PK_MAD_I16` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:16664`
  through `:16698`, `:16747` through `:16772`, `:16950` through `:16975`, and
  `:17008` through `:17046`. Existing gaps cover inline constants
  (`RDNA3-RJ-001`), ordinary packed `CLMP` (`RDNA3-RJ-002`), BF16 DOT
  conversion (`RDNA3-RJ-099`), and integer DOT clamp behavior
  (`RDNA3-RJ-100`).
- Chapter 15.3.1-15.3.5 VALU decode narrow match: generated RDNA3 machine
  structs preserve VOP1, VOPC, VOP2, VOP3, VOP3P, VINTERP, and VOP3SD raw field
  layouts at `rdna3/machine_insts.h:41` through `:105` and `:440` through
  `:505`; generated builders also expose the VOP3SD caller-controlled fields at
  `rdna3/builders.h:517` through `:544`. Existing VALU gaps are about execution
  semantics, source legality, true16 handling, and MODE policy, not missing raw
  field storage.
- Chapter 15.3.8-15.3.9 DPP extension narrow match: the generated VOP1, VOPC,
  VOP2, VOP3, VOP3P, and VOP3SD classes carry DPP16/DPP8 state and delegate
  operand slots at `rdna3/encodings.h:398` through `:528` and `:594` through
  `:614`; the generated constructors decode DPP lane selectors and DPP control
  fields before applying shared DPP helpers, for example in
  `rdna3/vop3p.cpp:53` through `:89`. Remaining DPP runtime issues are tracked
  under `RDNA3-RJ-093` through `RDNA3-RJ-095`.
- Chapter 15.4 VINTERP field narrow match: RDNA3 `VinterpMachineInst` stores
  `VDST`, `WAIT_EXP`, `OPSEL`, `CLAMP`, `OP`, `SRC0..SRC2`, and `NEG` at
  `rdna3/machine_insts.h:93` through `:105`, and the base `Vinterp` constructor
  remains 64-bit with no literal/DPP size extension at `rdna3/encodings.cpp:305`
  through `:311`. The existing VINTERP execution gaps are about interpolation,
  fixed DPP8 behavior, `EXPCNT` waiting, and true16 semantics, not field decode.
- Chapter 14 / 16.19 export raw-decode boundary: generated RDNA3 decodes the
  singleton `EXP` opcode and preserves the base raw target/source fields.
  `RDNA3-RJ-088` records missing graphics export state and `EXPCNT` execution
  semantics, while `RDNA3-RJ-101` records missing user-visible
  `EN`/`DONE`/`ROW_EN`/`VM`/`NULL`-target metadata. Existing `RDNA3-RJ-015` and
  `RDNA3-RJ-026` continue to cover zero-`EXEC` scheduling and missing
  `STATUS.SKIP_EXPORT` storage more generally.
- Chapter 13.4 GWS/ordered-count decode narrow match: generated RDNA3 has
  opcode constants, decode entries, constructors, and encoding fixtures for the
  audited GWS and ordered-count opcodes. `RDNA3-RJ-087` records execution,
  wait-counter, and GWS/ordered-count state semantics; `RDNA3-RJ-006` continues
  to cover ordinary GDS backing storage and append/consume support.
- Chapter 13 float-atomic decode narrow match: generated RDNA3 has concrete DS
  F32/F64 and FLAT/GLOBAL/SCRATCH/buffer F32 constructors for the audited
  opcode shells, sets the expected broad element widths and memory atomic
  operation classes, and routes implemented forms through the local or global
  memory pipelines. `RDNA3-RJ-086` records missing numeric edge semantics in the
  shared RMW helpers, while the broader non-application of `MODE` controls
  remains tracked under `RDNA3-RJ-027`.
- Chapter 12 ordinary DS field/decode narrow match: generated RDNA3 has
  concrete constructors for the audited ordinary DS load/store/atomic opcode
  shells, and implemented memory forms set `MEMORY_OP`, `VectorMemState` byte
  widths, load/store direction, and `WaitCounterType::DSCNT` before entering
  `LocalMemPipeline`. `RDNA3-RJ-083` is limited to the special ADDTID address
  form, `RDNA3-RJ-084` is limited to cross-lane DS helpers that bypass the
  memory pipeline, and `RDNA3-RJ-085` records the unimplemented `DS_WRAP_RTN_B32`
  body.
- Chapter 12 single- and double-address LDS address narrow match: the shared
  DS helper uses `{OFFSET1, OFFSET0}` as a 16-bit byte offset for ordinary
  single-address forms at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:60`
  through `:75`, while generated `*_2addr_*` bodies use separate scaled
  `OFFSET0`/`OFFSET1` addresses for B32/B64 and stride64 forms. The separate
  ADDTID formula mismatch is `RDNA3-RJ-083`.
- Chapter 12 LDSDIR/GDS/BVH boundary: LDSDIR no-op execution remains tracked
  under `RDNA3-RJ-010`, RDNA3 GDS absence remains tracked under `RDNA3-RJ-006`,
  LDS allocation/alignment behavior remains tracked under `RDNA3-RJ-024` and
  `RDNA3-RJ-025`, and `DS_BVH_STACK_RTN_B32` remains tracked under
  `RDNA3-RJ-004`/`RDNA3-RJ-005`. The Chapter 12 pass did not duplicate those
  preexisting findings.
- Chapter 11 flat mnemonic narrow match: RDNA3 uses the generated `flat_mnemonic`
  helper to rewrite `flat_` mnemonics to `scratch_` or `global_` based on `SEG` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/encodings.cpp:14` through
  `:20`, and the `Flat` encoding base installs that owned mnemonic at
  `rdna3/encodings.cpp:398` through `:405`.
- Chapter 11 raw load/store/D16 data narrow match: representative generated
  FLAT/GLOBAL/SCRATCH load and store bodies set byte widths, sign extension,
  D16/D16_HI flags, and load/store state before the common memory pipeline. The
  gaps above are about address selection, aperture/error behavior, missing CSUB,
  and metadata, not ordinary in-range byte movement.
- Chapter 11 wait-counter boundary: the Chapter 11 dual-counter FLAT rule is
  already recorded under `RDNA3-RJ-044`, and the generic alignment/MEMVIOL state
  gap remains recorded under `RDNA3-RJ-023`; `RDNA3-RJ-080` adds the
  Chapter-11-specific aperture/offset consequences.
- Chapter 10 MIMG decode-field narrow match: RDNA3 machine structs preserve the
  expected raw MIMG fields for `NSA`, `DIM`, `UNORM`, `DMASK`, cache bits, `R128`,
  `A16`, `D16`, opcode, `VADDR`, `VDATA`, `SRSRC`, `TFE`, `LWE`, `SSAMP`, and the
  NSA address-extension fields at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/machine_insts.h:165` through
  `:188` and `:253` through `:280`.
- Chapter 10 MIMG constructor-shell narrow match: generated RDNA3 constructors
  expose the broad image operand directions and descriptor classes for
  load/store/atomic/query/sample/gather forms, including `SRSRC` and `SSAMP`
  where applicable. `RDNA3-RJ-076` and `RDNA3-RJ-077` are about missing runtime
  semantics and dynamic field-dependent operand windows, not total absence of
  image opcode construction.
- Chapter 10 BVH decode-shell narrow match: generated RDNA3 has distinct
  `IMAGE_BVH_INTERSECT_RAY` and `IMAGE_BVH64_INTERSECT_RAY` classes with 128-bit
  result operands, 352-bit versus 384-bit address operands, and 128-bit resource
  operands at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/mimg.cpp:468`
  through `:500`. The execution gap remains `RDNA3-RJ-078`.
- Chapter 9 MUBUF/MTBUF decode narrow match: RDNA3 machine structs preserve the
  expected raw MUBUF and MTBUF fields, including `OFFSET`, cache bits, opcode,
  `FORMAT` for MTBUF, `VADDR`, `VDATA`, `SRSRC`, `TFE`, `OFFEN`, `IDXEN`, and
  `SOFFSET`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/machine_insts.h:131`
  through `:162`.
- Chapter 9 VADDR operand-size narrow match: generated MUBUF/MTBUF
  constructors call `buffer_vaddr_bits()` so `VADDR` is zero, 32, or 64 bits
  based on `IDXEN`/`OFFEN`, matching the manual's address VGPR table for decode
  metadata. The address arithmetic gap is recorded separately under
  `RDNA3-RJ-070`.
- Chapter 9 raw buffer load/store narrow match: the generated raw MUBUF
  B8/B16/B32/B64/B96/B128 and D16/D16_HI B8/B16 execute bodies select the
  expected element sizes, sign extension, destination halves, and store-source
  halves for ordinary in-range byte-copy cases. Formatted conversion and full
  descriptor behavior are tracked under `RDNA3-RJ-070` and `RDNA3-RJ-071`.
- Chapter 9 atomic return narrow match: generated buffer atomics set
  `is_load = (inst_.glc != 0)`, and the common atomic RMW path stores old
  values in `response_data` for GLC returns. `RDNA3-RJ-073` is specific to the
  `CSUB` arithmetic operation, and cache-flag scope remains tracked under
  `RDNA3-RJ-037`.
- Chapter 8 raw `S_LOAD_*` address narrow match: RDNA3 `smem_calculate_address`
  reads the raw 64-bit base from `SBASE * 2`, adds signed `OFFSET` and
  SGPR/M0/NULL `SOFFSET`, and clears the final two address bits at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/addr_calc.cpp:31` through
  `:50`, matching the non-buffer formula in section 8.1.1 for the ordinary
  in-range case.
- Chapter 8 SMEM opcode and load-width narrow match: generated RDNA3 SMEM
  constructors cover `S_LOAD_B32/B64/B128/B256/B512`,
  `S_BUFFER_LOAD_B32/B64/B128/B256/B512`, `S_GL1_INV`, and `S_DCACHE_INV`, and
  the scalar-load bodies pass the expected 1/2/4/8/16 DWORD widths into
  `ScalarMemState` at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/smem.cpp:32`
  through `:315`.
- Chapter 8 scalar invalidate functional boundary: `S_GL1_INV` invalidates the
  vector L1 and flushes L2 through the shared helper, while `S_DCACHE_INV`
  invalidates scalar L1 at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1763`
  through `:1768` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/smem.cpp:308`
  through `:315`. The wait-counter and group/isolation semantics remain gaps
  under `RDNA3-RJ-068` and `RDNA3-RJ-069`.
- Chapter 8 cache-flag boundary: RDNA3 SMEM uses the shared GFX11
  `mtype_from_flags_gfx11()` helper in generated load bodies; the mismatch
  between that helper and RDNA3 cache-control prose is already tracked under
  `RDNA3-RJ-037`, so the Chapter 8 slice does not duplicate it.
- Chapter 7.6 VOPD decode/field narrow match: RDNA3 recognizes the VOPD prefix
  in the decoder, extracts the X/Y opcode and source fields, reconstructs the Y
  destination low bit from `!VDSTX[0]`, and produces an executable `Vopd` object
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/decoder.cpp:16` through
  `:18` and `rdna3/vopd.cpp:211` through `:355`.
- Chapter 7.6 VOPD literal-size narrow match: rocjitsu sizes VOPD as 96 bits
  when either `SRCX0` or `SRCY0` is the literal selector, or when either slot
  is an always-literal FMAAK/FMAMK form, at `rdna3/vopd.cpp:234` through
  `:243`. This follows the manual's Y-side literal capability and is broader
  than the XML condition recorded in `RDNA3-XML-044`.
- Chapter 7.6 VOPD ordinary execution boundary: rocjitsu implements the
  inspected non-DOT2ACC slot operations, including fused FMAC/FMAAK/FMAMK,
  ordinary F32 add/sub/mul, DX9 zero multiply, MOV, CNDMASK through VCC, max,
  min, ADD_NC, LSHLREV, and AND at `rdna3/vopd.cpp:134` through `:209`. The
  remaining VOPD-specific defects are recorded in `RDNA3-RJ-059` through
  `RDNA3-RJ-065`.
- Chapter 7.6 VOPD paired read/write narrow match: `Vopd::execute_impl()`
  computes both X and Y lane results before writing either destination at
  `rdna3/vopd.cpp:345` through `:355`, matching the manual's read-old rule for
  pairs where one operation reads a VGPR written by the other. Pair legality
  validation remains a separate gap under `RDNA3-RJ-061`.
- Chapter 7.4 compact true16 operand narrow match: RDNA3 compact 16-bit source
  operands map `0..127` to `vN.l` and `128..255` to `vN.h` through
  `packed_16bit_vgpr_source` at `rdna3/operand.cpp:34` through `:47`, and compact
  16-bit destinations use the corresponding compact register-reference path
  through `packed_16bit_vgpr_dst` at `:50` through `:58`. `RDNA3-RJ-057` is
  about reusing that compact collapse for OPSEL-based VOP3/VOP3P/VINTERP
  destinations.
- Chapter 7.4 VOP3 true16 helper narrow match: ordinary generated VOP3 true16
  bodies use `read_vop3_true16_src()` and `write_vop3_true16_dst()` to select
  source halves and merge the OPSEL-selected destination half at
  `shared/simd_glue.h:270` through `:321`; representative F16 VOP3 bodies call
  those helpers at `rdna3/vop3.cpp:640` through `:643` and `:8266` through
  `:8298`.
- Chapter 3 core-state storage narrow match: `Wavefront` stores PC, live
  `EXEC`, live `VCC`, `M0`, raw MODE, scratch/aperture bases, and wait counters
  at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:211` through `:283` and
  `:532` through `:545`. The gaps above concern state fidelity, access
  semantics, and trap/control behavior, not total absence of a wavefront state
  object.
- Chapter 3 wave32 mask narrow match: RDNA3 runs as wave32 today
  (`RdnaIsaBase::WF_SIZE = 32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/rdna_isa_base.h:50`
  through `:52`), `Wavefront::exec()` masks live lanes at
  `wavefront.h:211` through `:230`, and the RDNA3 scalar special-source path
  computes `EXECZ`/`VCCZ` from only active wave lanes at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/operand.cpp:763` through
  `:768`. `RDNA3-RJ-007` tracks the missing wave64 dispatch selection.
- Chapter 3 branch-condition narrow match: RDNA3 `S_CBRANCH_VCCZ` and
  `S_CBRANCH_EXECZ` test live low-half masks through `wf.vcc()`/`wf.exec()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/sopp.cpp:284` through
  `:335`.
- Chapter 3 PC-relative branch narrow match: ordinary `S_BRANCH` and
  `S_CBRANCH_*` helpers compute `wf.pc + 4 + offset * 4 - size_`, and the CU
  adds `size_` after execution at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/sopp.cpp:220` through
  `:223` and `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:439`,
  matching the manual's next-instruction-relative branch formula.
- Chapter 3 direct-PC read narrow match: RDNA3 `S_GETPC_B64`, `S_SWAPPC_B64`,
  and `S_CALL_B64` write the next-instruction PC to their scalar destination at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/sop1.cpp:1183` through
  `:1185`, `:1221` through `:1224`, and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/sopk.cpp:411` through
  `:414`.
- Chapter 3.3.1 basic scalar-selector narrow match: RDNA3 operand resolution
  names and handles ordinary SGPRs, VCC, TTMPs, `NULL`, `M0`, and `EXEC` selector
  encodings at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/operand.cpp:703`
  through `:723` and `:870` through `:910`; `RDNA3-RJ-016` and
  `RDNA3-RJ-017` are about allocation bounds, privilege, range-crossing, and
  alignment behavior.
- Chapter 3.3.1 VCC execution narrow match: representative RDNA3 VOP2/VOP3
  execution reads or writes VCC for `V_CNDMASK`, carry add/sub, and
  `V_DIV_FMAS`, for example at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:8257`
  through `:8288`, `:2834` through `:2862`, `:18470` through `:18515`, and
  `:10231` through `:10268`.
  `RDNA3-RJ-018` is about public dependency metadata, not total absence of VCC
  execution behavior.
- Chapter 3.3.2 zero-VGPR dispatch boundary: the shared register allocator
  refuses zero-count allocations at
  `lib/simdojo/include/simdojo/components/register_file.h:44` through `:50`;
  `RDNA3-RJ-019` is about physical granularity and `S_SENDMSG` deallocation,
  not the zero-allocation predicate.
- Chapter 3.3.2 basic VGPR state narrow match: `Wavefront` exposes
  `num_vgprs()` and `vgpr_alloc()` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:91` through `:93` and
  `:200` through `:202`, and dispatch fills those fields at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:139` through `:142`.
  The gaps above are about architecture-specific range consequences and
  deallocation, not total absence of VGPR allocation state.
- Chapter 3.3.2 VOPD narrow match: RDNA3 recognizes VOPD opcodes in the decoder
  and has an executable `Vopd` class at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/decoder.cpp:17` through
  `:18` and `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/vopd.cpp:345`
  through `:354`; `RDNA3-RJ-020` is limited to the source/destination
  out-of-range exception behavior.
- Chapter 3.3.3 buffer-resource OOB narrow match: `mubuf_calculate_addresses`
  removes lanes that fail buffer descriptor bounds checks and `vector_complete`
  zeroes those OOB lanes for loads at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h:68`
  through `:132` and
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:139` through `:164`.
  `RDNA3-RJ-022` is about destination register range/return-window
  nullification, not the existence of basic buffer bounds masking.
- Chapter 3.3.3 scalar-memory alignment boundary: RDNA3 `smem_calculate_address`
  clears low address bits before SMEM access at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/addr_calc.cpp:43` through
  `:50`; `RDNA3-RJ-023` is limited to the manual's `SH_MEM_CONFIG`-controlled
  VMEM/LDS/flat/global/scratch alignment behavior and atomic MEMVIOL.
- Chapter 3.3.4 LDS backing OOB narrow match: `Lds` returns zero for loads and
  drops writes when an address crosses the total simulated LDS backing at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/lds.h:31` through `:107` and `:136`
  through `:168`. `RDNA3-RJ-024` and `RDNA3-RJ-025` are about the missing
  per-workgroup allocation boundary, documented allocation granularity,
  side-placement constraints, alignment modes, and violation reporting.
- Chapter 3.3.4 CU/WGP LDS capacity boundary: rocjitsu records the WGP-mode bit,
  validates requested LDS against CU-local or WGP-paired capacity, and uses a
  sibling-CU LDS backing in WGP mode at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:976` through
  `:999` and `lib/rocjitsu/src/rocjitsu/vm/amdgpu/spi.h:203` through `:239`.
  The gaps are the finer allocation, bounds, side, and alignment semantics.
- Chapter 3.4.1 STATUS narrow match: rocjitsu stores a raw per-wave STATUS word,
  updates SCC through `Wavefront::write_scc()`, and synthesizes scalar
  `EXECZ`/`VCCZ` special-source reads from live masks. `RDNA3-RJ-026` is about
  missing field inventory, initialization, write permissions, and non-SCC
  dynamic behavior.
- Chapter 3.4.2 MODE narrow match: rocjitsu stores a raw per-wave MODE word and
  keeps the VGPR-MSB indexing subfield synchronized with `S_SET_VGPR_MSB` state.
  `RDNA3-RJ-027` is about the manual's floating-point, denormal, trap,
  exception, and FP16-overflow controls not affecting execution.
- Chapter 3.4.3 M0 storage/operand narrow match: `Wavefront` stores one raw
  per-wave `M0` value and RDNA3 scalar operand resolution reads and writes the
  `M0` selector at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:240` through `:246` and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/operand.cpp:716` through
  `:719` / `:897` through `:900`. The inspected explicit write-capable
  `OPR_SDST` constructors are SOP-side, for example
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/sop1.cpp:24` through `:31`,
  `rdna3/sop2.cpp:20` through `:29`, and `rdna3/sopk.cpp:23` through `:30`;
  non-SALU consumers such as DS ADDTID read `wf.m0()` as input state at
  `rdna3/ds.cpp:1985` through `:1989` and `:2019` through `:2023`.
  Instruction-consumer gaps remain tracked where they belong, including LDSDIR
  execution in `RDNA3-RJ-010`, GDS in `RDNA3-RJ-006`, LDS
  allocation/alignment in `RDNA3-RJ-024` and `RDNA3-RJ-025`, export graphics
  paths in `RDNA3-RJ-009`, and selected send-message side effects in earlier
  control-flow audit entries.
- Chapter 3.4.4-3.4.5 NULL/SCC narrow match: RDNA3 scalar reads return zero for
  `NULL`, destination writes to `NULL` return before mutating state, and
  representative SALU helpers still update SCC around destination writes; for
  example `S_ADDK_I32` writes SCC then writes `SDST` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:454`
  through `:466`, with the NULL no-op handled by the destination resolver.
  SCC consumers also route through `Wavefront::read_scc()` for conditional
  branches at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/sopp.cpp:240`
  through `:267`.
- Chapter 3.4.6 VCC/VCCZ narrow match: `Wavefront` stores raw VCC, scalar
  operands expose `VCC_LO`/`VCC_HI` and whole `VCC`, VOPC compare helpers build
  a zero-initialized mask and set only `EXEC`-active passing lanes before
  `wf.set_vcc(vcc)`, and `S_CBRANCH_VCCZ`/`S_CBRANCH_VCCNZ` plus scalar-source
  `VCCZ` synthesize the wave-size masked live value at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:232` through `:238`,
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/operand.cpp:710` through
  `:713`, `:762` through `:765`, `:885` through `:927`,
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:4127`
  through `:4160`, and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/sopp.cpp:284` through
  `:315`. `RDNA3-RJ-012` still covers stale raw `STATUS.VCCZ`, and
  `RDNA3-RJ-018` covers public VCC dependency/source-accounting metadata.
- Chapter 3.4.7 scratch-address narrow match: rocjitsu has per-wave scratch
  base and per-lane scratch-size state, zeroes both on reset, initializes them
  for dispatches with private scratch, and feeds that state into scratch and
  flat-private address calculation. `RDNA3-RJ-028` and `RDNA3-RJ-029` are about
  RDNA3 selector aliasing, alignment, and access-policy fidelity, not total
  absence of scratch-base state.
- Chapter 3.4.8 HWREG boundary: `RDNA3-RJ-011` already records the wrong and
  incomplete RDNA3 `S_GETREG`/`S_SETREG` map, including the missing
  `FLAT_SCRATCH_LO/HI`, `HW_ID1/2`, `SH_MEM_BASES`, `PC`, and `FLUSH_IB`
  behavior. This slice keeps that existing finding as the HWREG execution gap
  rather than duplicating it.
- Chapter 3.4.9 trap/exception boundary: existing gaps cover the rocjitsu
  behavior found in this slice. `RDNA3-RJ-013` covers missing `TRAPSTS`,
  TBA/TMA wave state, `S_TRAP`, `S_RFE_B64`, and TMA/TBA return-message
  execution; `RDNA3-RJ-016` covers TTMP user-mode read-zero/write-ignore
  behavior; `RDNA3-RJ-026` covers incomplete STATUS `PRIV`/`TRAP_EN` and trap
  status fields; and `RDNA3-RJ-027` covers `MODE.EXCP_EN` and
  `TRAP_AFTER_INST` not affecting execution.
- Chapter 3.4.9 trap-debug host surface boundary: the KFD shim stores
  `trap_tba_addr` and `trap_tma_addr` for `AMDKFD_IOC_SET_TRAP_HANDLER` and
  stores debugger exception masks for `AMDKFD_IOC_DBG_TRAP_ENABLE` at
  `lib/rocjitsu/src/rocjitsu/kmd/linux/kfd_process.h:83` through `:85`,
  `:176` through `:187`, and
  `lib/rocjitsu/src/rocjitsu/kmd/linux/simulated_kfd.cpp:675` through `:686`
  / `:1964` through `:1990`. That is host/API bookkeeping; the command
  processor comment says TBA/TMA have no reader yet, and the ISA executor gaps
  remain in `RDNA3-RJ-013`.
- Chapter 3.4.10 REALTIME decode narrow match: rocjitsu recognizes the
  `S_SENDMSG_RTN_B32/B64` instruction forms and special-cases message `0x83`,
  matching XML's `MSG_RTN_GET_REALTIME` selector. `RDNA3-RJ-030` is about the
  clock domain, synchronous writeback, missing LGKM dependency, and missing
  `SHADER_CYCLES` HWREG behavior.
- Chapter 3.5.1 compute `EXEC` initialization narrow match: the command
  processor computes an initial active-lane mask from the dispatch grid,
  workgroup dimensions, wave index, and wave size in
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/dispatch_entry.h:206` through `:238`,
  then writes it with `wf->set_exec(...)` before register initialization at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:710` through
  `:720`. Existing tests cover a wave64 grid tail and a 3D tail with
  workgroup offset at `emulation/rocjitsu/tests/amdgpu_vm_test.cpp:581`
  through `:605`. This covers the ordinary compute-dispatch active-lane
  contract; graphics launch and wave64 configurability remain broader launch
  gaps in `RDNA3-RJ-009` and `RDNA3-RJ-007`.
- Chapter 3.5.1 null-wave boundary: the audited compute path derives masks only
  for waves it actually schedules from `wfs_per_workgroup`, so ordinary compute
  dispatch does not create extra all-zero waves. The manual's null-wave launch
  cases are not modeled as a separate producer in rocjitsu's current compute
  dispatch surface; this is treated as part of the broader non-compute launch
  boundary rather than a duplicate finding in this slice.
- Chapter 3.5.2 `FLAT_SCRATCH` initialization narrow match: `Wavefront::reset`
  zeroes `scratch_base_` and `scratch_lane_size_` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:477` through `:483`, and
  dispatches with `private_segment_fixed_size > 0` set a per-wave scratch base,
  lane size, and optional `flat_scratch_init` user SGPR pair at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:323` through
  `:361`. The remaining scratch fidelity issues are the RDNA3 scalar-selector
  aliasing in `RDNA3-RJ-028` and the alignment/access-policy gap in
  `RDNA3-RJ-029`.
- Chapter 3.5.3 AMDHSA compute user-SGPR narrow match:
  `CommandProcessor::init_wavefront_regs()` packs enabled AMDHSA user SGPR
  payloads in descriptor order, including private-segment buffer, dispatch
  pointer, queue pointer, kernarg pointer, dispatch ID, flat-scratch init,
  private-segment size, and kernarg preload values at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:181` through
  `:256`. The `TG_SIZE` and multidimensional X-ID issues are recorded
  separately as `RDNA3-RJ-031` and `RDNA3-RJ-032`.
- Chapter 3.5.3 graphics-stage boundary: PS/GS/HS SGPR preload tables require
  graphics shader launch state that rocjitsu does not model today. That remains
  under the existing graphics-launch gap `RDNA3-RJ-009` rather than being
  duplicated for each 3.5.3 graphics SGPR row.
- Chapter 3.5.4 compute workitem-ID VGPR narrow match: rocjitsu reads
  `COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID` into the dispatch entry at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:1037` through
  `:1045`, computes per-lane local IDs in
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/dispatch_entry.h:196` through `:204`,
  packs GFX11+ IDs into `v0[9:0]`, `v0[19:10]`, and `v0[29:20]` at
  `dispatch_entry.h:186` through `:193`, and writes the requested components at
  `command_processor.cpp:300` through `:321`. Existing dispatch tests cover
  packed-TID component selection at `emulation/rocjitsu/tests/amdgpu_vm_test.cpp:481`
  through `:515`.
- Chapter 3.5.5 compute LDS backing narrow match:
  `CommandProcessor::dispatch_workgroups()` assigns each wave an LDS base and
  backing store, and `process_aql_packet()` validates requested compute LDS
  capacity. `RDNA3-RJ-034` concerns pixel-shader parameter preload; LDSDIR
  instruction execution remains covered by `RDNA3-RJ-010`, and LDS
  allocation/reporting fidelity remains covered by `RDNA3-RJ-024` and
  `RDNA3-RJ-025`.
- Chapter 4 instruction-group/decode boundary: generated RDNA3 decode covers
  the broad instruction families named by the Chapter 4 introduction. The new
  Chapter 4 runtime gaps are common-source reserved selector handling, signed
  literal materialization, SLC/GLC/DLC policy, and padding canonicalization.
- Chapter 4.1 `NULL` destination boundary: Chapter 3.4.4 clarifies that when
  `NULL` is used as an SALU destination, the instruction executes, `SDST` is
  not written, and SCC is updated if the instruction normally updates SCC at
  `rdna3/README.md:999` through `:1004`. Rocjitsu's SALU helpers follow that
  clarified behavior for representative SCC-updating forms, as noted in the
  Chapter 3.4.4-3.4.5 NULL/SCC narrow match above.
- Chapter 5 ordinary branch narrow match: RDNA3 `S_BRANCH` and the SCC/VCC/EXEC
  conditional branch helpers sign-extend `SIMM16`, use instruction-count
  offsets, and update `wf.pc` relative to the next instruction at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/sopp.cpp:206` through
  `:358`, matching the Chapter 5.4 branch formula for those condition classes.
- Chapter 5 direct-PC boundary: RDNA3 `S_GETPC_B64`, `S_SETPC_B64`, and
  `S_SWAPPC_B64` have executable direct-PC helpers, but low-bit alignment and
  zero-target behavior remain tracked under `RDNA3-RJ-014`, and `S_RFE_B64`
  trap-return behavior remains tracked under `RDNA3-RJ-013`.
- Chapter 5 barrier narrow match: RDNA3 `S_BARRIER` moves the wave to
  `WfState::BARRIER`, and `ComputeUnitCore::update_wf_states()` releases waves
  in the same dispatch/workgroup once all non-halted peers are at the barrier,
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:745`
  through `:746` and
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:313` through `:335`.
  Remaining launch/resource limits are tracked under earlier Chapter 2/3 gaps.
- Chapter 5 wait-counter narrow match: RDNA3 `S_WAITCNT` decodes the GFX11
  `EXP/LGKM/VM` fields and sets wait thresholds at `rdna3/sopp.cpp:115`
  through `:130`; split `S_WAITCNT_VSCNT`, `S_WAITCNT_VMCNT`,
  `S_WAITCNT_EXPCNT`, and `S_WAITCNT_LGKMCNT` literal-only behavior is limited
  by the dynamic SGPR threshold gap in `RDNA3-RJ-089`; and memory pipelines
  increment/retire the selected counter at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.h:62` through `:87`.
  `RDNA3-RJ-042` and `RDNA3-RJ-044` cover the missing message and dual-FLAT
  producers.
- Chapter 5 `S_DELAY_ALU` runtime boundary: generated RDNA3 preserves and
  disassembles the raw `S_DELAY_ALU` operand through `OPR_DELAY`, and executing
  it as a no-op is acceptable for functional emulation because Section 5.7 says
  it is optional and not required for correct operation. The XML gap records the
  missing stream-history metadata for validators and schedulers.
- Chapter 6 signed-overflow execution narrow match: RDNA3 signed add/sub forms
  call the shared signed-overflow helpers despite the XML description wording
  gap; representative bodies include `S_ADD_I32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:418`
  through `:423`, `S_ADDK_I32` at `:455` through `:466`, and signed subtract
  at `:2576` through `:2584`.
- Chapter 6 conditional, compare, and bit-operation execution narrow match:
  checked representative runtime bodies read or write SCC according to the
  manual, including `S_CMOVK_I32` at `shared/execute_shared.h:1095` through
  `:1101`, `S_CSELECT_*` at `:1511` through `:1524`, and bit-compare helpers
  at `:897` through `:925`. `RDNA3-RJ-047` is limited to def-use metadata.
- Chapter 6.8 HWREG/MODE boundary: existing gaps cover the runtime behavior
  found in this slice. `RDNA3-RJ-011` covers the incomplete/wrong RDNA3
  `S_GETREG`/`S_SETREG` HWREG map, and `RDNA3-RJ-027` covers empty
  `S_ROUND_MODE`/`S_DENORM_MODE` helpers plus missing MODE effects.
- Chapter 6.9 aperture selector narrow match: once wavefront aperture state is
  populated, RDNA3 scalar operand resolution returns shared/private base and
  limit selectors from that state for both 32-bit and 64-bit reads at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/operand.cpp:749` through
  `:756` and `:858` through `:865`. Updating that state through architectural
  `SH_MEM_BASES` remains part of the existing HWREG gap.
- Chapter 7.1 encoding-shell narrow match: generated RDNA3 structs preserve the
  raw VOP3, VOP3P, and VOP3SD fields and the generated base constructors size
  literal-extension forms. The gaps above are about missing legality,
  disassembly, and semantic policy rather than absence of basic decode storage.
- Chapter 7.2 operand-shell narrow match: representative generated constructors
  bind readlane/writelane, cndmask, div-fmas, and VOP3SD carry operands to the
  expected broad source and scalar-destination classes. The gaps above are
  about validation, lane masking, and wave64 mask behavior layered on top of
  those operand shells.
- Chapter 7.2.3 floating-point VOP3 modifier narrow match: representative F16,
  F32, and F64 VOP3 bodies apply source modifiers, `OMOD`, and floating-point
  `[0, 1]` clamp at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2958`
  through `:2998`, `:3018` through `:3056`, and `:3075` through `:3115`.
  SIMD helpers also apply floating-point destination modifiers at
  `shared/simd_glue.h:357` through `:398`. The gaps above are about integer
  clamp saturation, invalid modifier validation, and MODE/denorm/IEEE effects
  already tracked in `RDNA3-RJ-027`.
- Chapter 7.2.4 and 7.2.7 boundary: the MODE round/denorm behavior in this slice
  maps to existing `RDNA3-RJ-027`, and the VALU source/destination GPR
  out-of-range consequences map to existing `RDNA3-RJ-020`.
- Chapter 7.3 explicit non-VOP3P VALU inventory is generated: every explicit
  mnemonic in the table has a generated RDNA3 mnemonic after normalizing
  `_e32` suffixes and XML/manual aliases, and the generic compare rows expand to
  concrete VOPC/VOP3 compare instruction names.
- Chapter 7.3 compare opcode expansion follows the detailed Chapter 15 opcode
  tables: generated names use `NE` for integer not-equal predicates and do not
  materialize compact summary predicates that are absent from the detailed
  opcode rows, such as I16/U16 `F`/`T`.
- Chapter 2 compute dispatch narrow match: the command processor and
  dispatch-entry paths create waves over a 1D/2D/3D workgroup grid and
  initialize local work-item IDs, matching the broad compute-shader launch
  concept in Chapter 2. `RDNA3-RJ-007` and `RDNA3-RJ-008` track missing wave64
  and validation pieces.
- Chapter 2 CU/WGP LDS placement boundary: rocjitsu records the RDNA
  `COMPUTE_PGM_RSRC1.WGP_MODE` bit in `DispatchEntry`, validates LDS capacity
  against either CU-local or WGP-paired LDS, and uses a sibling-CU WGP LDS
  backing in WGP mode. This covers the broad LDS-placement distinction; the
  remaining Chapter 2 WGP-mode legality issue is `RDNA3-RJ-010`.
- Chapter 2 shader-padding boundary: generated RDNA3 decodes
  `S_SET_INST_PREFETCH_DISTANCE`, `S_CODE_END`, and `S_ENDPGM`; `S_CODE_END`
  and `S_SET_INST_PREFETCH_DISTANCE` execute as no-ops at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1508`
  and `:2523` through `:2524`. Rocjitsu does not model instruction prefetch, so
  the Chapter 2 64-DWORD padding rule is a code-object validation/host safety
  requirement rather than a functional executor state mutation.
- Chapter 1 architecture-model boundary: rocjitsu has RDNA3 ISA traits, a
  GFX1100/W7900 config with shader-engine/CU/cache/topology fields, a
  command-processor dispatch path, WGP-mode LDS placement, L1/L2 cache objects,
  and completion callback plumbing. The remaining differences are detailed
  operational gaps tracked under later sections or newly recorded in
  `RDNA3-RJ-006`.
- Chapter 1 LDS narrow match: the W7900 config reports 64KiB LDS per CU at
  `configs/gfx1100_w7900.json:29`, and WGP-mode LDS capacity is built from a
  paired-CU sum in `lib/rocjitsu/src/rocjitsu/vm/amdgpu/spi.h:270` through
  `:273`, matching the manual's 128KiB-per-WGP / 64KiB-per-workgroup storage
  scale. Bank geometry and LDS atomic-unit timing are not modeled here.
- Chapter 1 dispatch/cache boundary: the command processor derives 1D/2D/3D
  workgroup dispatch state and validates LDS capacity at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:941` through
  `:999`, and it invalidates CU caches for AQL acquire fences at `:1052`
  through `:1059`. This is a functional compute-dispatch/cache surface, not a
  full model of the hardware command processor, memory controller, DMA engine,
  or relaxed-consistency acknowledgment protocol.
- `V_PK_FMAC_F16` is generated as VOP2, not VOP3P, matching the XML `ENC_VOP2`
  entry at `amdgpu_isa_rdna3.xml:84100`; it should not be treated as a missing
  RDNA3 VOP3P instruction.
- MIX execution implements the special selector mapping, treats `NEG_HI` as
  absolute value, handles inline constants selected as F16 through the MIX
  helper path, applies `CLMP`, and preserves the untouched destination half for
  MIXLO/MIXHI at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:11470`
  through `:11595`.
- DOT2 F32 BF16/F16 helpers apply `CLMP` and use `OPSEL`/`OPSEL_HI` for the
  two packed components at `execute_shared.h:10414` through `:10483`; broader
  DOT inline-constant exception behavior still needs a dedicated pass.
- The ordinary `V_PK_*` constructors contain shared DPP handling scaffolding,
  but the inspected XML entry for `V_PK_FMA_F16` exposes base and literal VOP3P
  encodings rather than DPP encodings, matching the manual's no-DPP rule for
  `V_PK_*`.
- RDNA3 WMMA opcode coverage narrow match: generated RDNA3 contains concrete
  constructors and decode entries for all six Chapter 7.9 WMMA opcodes, with
  A/B VGPR source sizes and 256-bit C/D sizes matching the XML/manual table.
  The remaining WMMA issues are source-legality validation, modifier semantics,
  and stream hazards tracked in `RDNA3-RJ-096` through `RDNA3-RJ-098`.
- Chapter 16.10 WMMA full-`EXEC` execution narrow match: each WMMA definition
  saves `EXEC`, forces `EXEC` to all ones, evaluates the matrix operation, and
  restores `EXEC` at `rdna3/README.md:15485` through `:15550`. The shared GFX11
  WMMA helpers do not predicate legal base-form result production on
  `wf.exec()`: F32-output writes collected results at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/mma_exec.h:1584` through
  `:1602`, packed16-output writes every output lane/register at `:1629`
  through `:1670`, and I32-output writes collected results at `:3277` through
  `:3295`. Illegal DPP acceptance remains tracked separately in `RDNA3-RJ-096`.
- `DS_BVH_STACK_RTN_B32` is present in the RDNA3 opcode constants and decode
  table at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/opcodes.h:1137`,
  `opcodes.h:2074`, and `decoder.cpp:4557` through `:4558` /
  `decoder.cpp:8428`; this is an execution/metadata gap, not a missing decode.
- The generated RDNA3 constructor models the explicit operands and in-out
  `ADDR` path at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/ds.cpp:4097`
  through `:4111`, matching the XML operand direction and widths.
- The generated encoding fixture includes `ds_bvh_stack_rtn_b32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3/test_encodings.h:1138`.
