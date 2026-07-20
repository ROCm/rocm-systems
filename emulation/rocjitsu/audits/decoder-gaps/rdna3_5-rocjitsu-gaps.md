# RDNA3.5 Rocjitsu Gaps

Architecture: RDNA3.5

Manual source: `workspace_docs/amdgpu-isa-manuals/rdna3.5/README.md`

Rocjitsu source: `emulation/rocjitsu`

## Coverage

| Area | Status | Notes |
| --- | --- | --- |
| RDNA3.5 Chapter 1 introduction / hardware overview | Audited statically | Checked RDNA3.5 ISA trait binding, shared RDNA wave/register traits, XCD/SE/CU topology shells, command-processor dispatch setup, WGP-mode LDS placement, GDS handling, L1/L2/cache surfaces, completion interrupt plumbing, and architecture-level boundaries. Detailed operational gaps remain tracked under later wave, LDS, cache, memory, and exception rows. |
| RDNA3.5 Chapter 2 shader concepts / wave modes / shader types / work-groups / padding | Audited statically | Checked fixed wave-size dispatch, compute grid/workitem initialization, graphics launch absence, WGP-mode placement, workgroup and LDS capacity handling, LDSDIR WGP-mode exclusions, barrier behavior, shader-padding/prefetch instruction shells, and existing detailed wave/barrier/LDS/export gaps. |
| RDNA3.5 Chapter 3.1-3.2 wave state / PC / EXEC / skipping | Audited statically | Checked wavefront state storage, scalar special-source access, `STATUS` layout, direct-PC and branch execution, HWREG access, trap/TBA/TMA paths, wait-counter accounting, vector-memory/DS lane-mask setup, and zero-`EXEC` scheduling behavior. |
| RDNA3.5 Chapter 3.3.1 SGPR storage state | Audited statically | Checked SGPR/VCC/TTMP allocation, scalar selector accessors, 64-bit alignment behavior, TTMP privilege, scalar out-of-range fallbacks, SAVEEXEC/WREXEC destination behavior, and S_MOVREL relative-index helpers. |
| RDNA3.5 Chapter 3.3.2 VGPR storage state | Audited statically | Checked launch VGPR allocation bookkeeping, operand VGPR reads/writes, `V_MOVREL` helpers, `S_SENDMSG` deallocate-VGPR handling, and memory/export/DS direct VGPR base calculations. |
| RDNA3.5 Chapter 3.3.3 memory alignment and out-of-range behavior | Audited statically | Checked memory-return destination OOR nullification, memory source OOR, PRT extra-destination sizing, image `DMASK`, vector-memory alignment-mode/MEMVIOL handling, and existing address-OOB lane behavior. |
| RDNA3.5 Chapter 3.3.4 LDS allocation, placement, alignment, and out-of-range behavior | Audited statically | Checked LDS dispatch allocation granularity, CU/WGP placement, per-workgroup LDS bounds, DS address calculation, LDS backing OOB behavior, source/destination VGPR range overlap, and LDS_CONFIG/alignment/MEMVIOL surfaces. |
| RDNA3.5 Chapter 3.4.1-3.4.2 STATUS and MODE registers | Audited statically | Checked STATUS bitfield layout, raw STATUS accessors, wave creation/reset, HWREG get/set overlap, raw MODE storage, S_ROUND_MODE/S_DENORM_MODE helpers, and representative FP execution paths. |
| RDNA3.5 Chapter 3.4.3 M0 | Audited statically | Checked wavefront M0 storage, scalar source/destination resolution, SMEM and buffer M0 operand paths, LDSDIR/GDS/DS ADDTID/sendmsg/export overlap, and existing consumer-specific gap coverage. |
| RDNA3.5 Chapter 3.4.4-3.4.6 NULL / SCC / VCC / VCCZ | Audited statically | Checked NULL scalar source/destination behavior, SALU SCC updates with NULL destinations, SCC consumers, VCC storage and scalar exposure, VOPC compare mask construction, wave32 VCCZ masking, and VCCZ branch handling against existing gap boundaries. |
| RDNA3.5 Chapter 3.4.7-3.4.8 FLAT_SCRATCH / hardware internal registers | Audited statically | Checked scratch-base storage/reset, dispatch initialization, scratch and flat-private address calculation, scalar selector handling for FLAT_SCRATCH-like aliases, generated operand bounds, and RDNA3.5 S_GETREG/S_SETREG HWREG coverage. |
| RDNA3.5 Chapter 3.4.9 trap and exception registers | Audited statically | Checked TTMP privilege overlap, trap entry/return shells, TBA/TMA return messages, `STATUS.TRAP_EN`, `MODE.EXCP_EN`, `TRAPSTS` fields, KFD trap-handler/debug surfaces, and existing trap/MODE/STATUS gap coverage. |
| RDNA3.5 Chapter 3.4.10 time counters | Audited statically | Checked `SHADER_CYCLES` via `S_GETREG`, `REALTIME` via `S_SENDMSG_RTN_B64`, simulation clock sources, and wait-counter behavior. |
| RDNA3.5 Chapter 3.5.1-3.5.2 initial `EXEC` / `FLAT_SCRATCH` state | Audited statically | Checked dispatch-time active-lane mask seeding, scratch-base reset, private-segment scratch initialization, and flat-scratch user-SGPR mirroring against the manual's launch-state rules. |
| RDNA3.5 Chapter 3.5.3 SGPR initialization | Audited statically | Checked graphics-stage SGPR preload boundaries, AMDHSA compute user-SGPR setup, compute workgroup-id system SGPRs, `TG_SIZE`, and RDNA3.5 compute TTMP launch payload handling. |
| RDNA3.5 VOP3P packed math | Audited statically | Checked generated packed F16/I16/U16 helpers, MIX helpers, DOT2 helpers, inline constants, selector handling, clamp behavior, and DPP reachability for this slice. |
| RDNA3.5 `DS_BVH_STACK_RTN_B32` | Audited statically | Checked generated decode tables, operand metadata, execution stub, memory-op classification, and encoding fixture coverage. |
| Remaining RDNA3.5 rocjitsu surface | Not started | Full decoder/autogen/runtime fuzzing remains outside the rows above. |

## Gaps

### RDNA3_5-RJ-001: Packed F16 inline constants use FP32 bit patterns

Manual evidence:

- RDNA3.5 section 7.5.1 says inline constants used with packed math produce a
  value only in the low 16 bits, and inline constants used with float 16-bit
  sources produce an F16 constant value, at `rdna3.5/README.md:2678` through
  `:2686`.

Rocjitsu evidence:

- The generated ordinary packed-F16 VOP3P constructors use 32-bit `OPR_SRC`
  operands and replace literal extensions with 32-bit `OPR_SIMM32` operands at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/vop3p.cpp:1059` through
  `:1413`.
- `Operand::read_lane()` selects `resolve_src_scalar16()` only when the operand
  size is 16 bits; otherwise it uses `resolve_src_scalar()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/operand.cpp:1048` through
  `:1066`.
- The 32-bit inline resolver returns FP32 bit patterns such as `0x3F000000` for
  `0.5f` at `operand.cpp:709` through `:754`, while the 16-bit resolver returns
  F16 encodings such as `0x3800` at `operand.cpp:781` through `:795`.
- Shared packed-F16 execution reads raw 32-bit values and then selects 16-bit
  halves, for example `execute_v_pk_add_f16_vop3p` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:16664`
  through `:16698` and `execute_v_pk_fma_f16_vop3p` at `:16839` through
  `:16884`.

Impact:

An inline constant such as 0.5 is read as `0x3f000000` for ordinary packed-F16
VOP3P, so selecting the low half yields zero and selecting the high half yields
`0x3f00` instead of the manual's F16 inline constant `0x3800`.

### RDNA3_5-RJ-002: Ordinary packed I16/U16/F16 VOP3P arithmetic ignores `CLMP`

Manual evidence:

- The RDNA3.5 VOP3P packed-math field table defines `CLMP` as a result clamp:
  float arithmetic clamps to `[0, 1.0]`, signed integer arithmetic clamps to
  `[min_int, +max_int]`, and unsigned integer arithmetic clamps to
  `[0, +max_uint]` at `rdna3.5/README.md:2657` through `:2658`.

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

### RDNA3_5-RJ-003: `V_FMA_MIX*` uses multiply-add instead of fused FMA

Manual/XML evidence:

- RDNA3.5 instruction definitions 32-34 describe `V_FMA_MIX_F32`,
  `V_FMA_MIXLO_F16`, and `V_FMA_MIXHI_F16` as fused multiply-add operations and
  use `fma(...)` in pseudocode at `rdna3.5/README.md:16215` through `:16282`.
- The XML entries for `V_FMA_MIX_F32`, `V_FMA_MIXLO_F16`, and
  `V_FMA_MIXHI_F16` are present at `amdgpu_isa_rdna3_5.xml:123059`, `:123385`,
  and `:123711` and describe fused multiply-add behavior.

Rocjitsu evidence:

- `execute_v_fma_mix_f32_vop3p`, `execute_v_fma_mixhi_f16_vop3p`, and
  `execute_v_fma_mixlo_f16_vop3p` compute `float result = a * b + c;` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:11470`
  through `:11595`.

Impact:

One-rounding FMA cases can differ from hardware and the RDNA3.5 ISA. This is a
real semantic mismatch; the audited RDNA3.5 manual and XML sources both say
fused.

### RDNA3_5-RJ-004: `DS_BVH_STACK_RTN_B32` decodes but is not executable

Manual/XML evidence:

- The RDNA3.5 manual describes `DS_BVH_STACK_RTN_B32` as an LDS-backed
  per-thread BVH stack operation with packed `ADDR` state, `OFFSET1[5:4]`
  stack-size selection, `DATA_VALID` filtering, push/pop behavior, and memory
  invalidation at `rdna3.5/README.md:4952` through `:4973` and
  `rdna3.5/README.md:24705` through `:24750`.
- The RDNA3.5 XML lists opcode 173 with explicit `VDST`, in-out `ADDR`,
  `DATA0`, `DATA1`, and implicit DSMEM operands at
  `amdgpu_isa_rdna3_5.xml:21850` through `:21898`.

Rocjitsu evidence:

- Rocjitsu generates `DsBvhStackRtnB32Ds` with the expected explicit operands
  and in-out `ADDR` modeling, but `execute_impl()` throws
  `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/ds.cpp:4097` through
  `:4116`.
- The Python semantic classifier falls unrecognized DS instructions through to
  semantic class `nop` at `lib/python/amdisa/semantics.py:2418` through
  `:2429`, and the generator emits `UnimplementedInst` for `nop` at
  `lib/python/amdisa/codegen/_generator.py:3377` through `:3378`.

Impact:

Any shader containing this legal RDNA3.5 stack operation decodes to a concrete
instruction object, but execution stops at runtime instead of updating LDS,
`VDST`, and `ADDR`.

### RDNA3_5-RJ-005: `DS_BVH_STACK_RTN_B32` is not marked as a memory operation

Manual/XML evidence:

- The instruction is described as an LDS instruction for pushing/popping a
  per-thread stack at `rdna3.5/README.md:4952` through `:4956` and
  `rdna3.5/README.md:24705` through `:24750`.
- The XML entry includes implicit DSMEM input/output operands and places the
  instruction in the `VMEM`/`DATA_SHARE` functional group at
  `amdgpu_isa_rdna3_5.xml:21882` through `:21898`.

Rocjitsu evidence:

- The generated `DsBvhStackRtnB32Ds` constructor sets source/destination
  operands but does not set `flags_ |= MEMORY_OP` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/ds.cpp:4097` through
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

### RDNA3_5-RJ-006: RDNA3.5 GDS is decoded but not modeled

Manual evidence:

- Chapter 1.1 defines GDS as scratch memory shared by all shader engines, with
  append-operation support, at `rdna3.5/README.md:387`.
- Chapter 1.2.2.2 says RDNA3.5 devices use a 4KiB GDS visible to waves of a
  kernel on all WGPs, with 128 bytes per cycle of access, two integer atomic
  units, preload/writeback around launch, and unordered plus domain-launch
  ordered append/consume support, at `rdna3.5/README.md:501`.

Rocjitsu evidence:

- The KFD sysfs emulation advertises `gds_size_in_kb 0` for GPU nodes at
  `lib/rocjitsu/src/rocjitsu/kmd/linux/sysfs.cpp:327` through `:329`.
- The device config struct has an LDS size field but no GDS size field at
  `lib/rocjitsu/src/rocjitsu/config/kfd_device_config.h:34` through `:45`.
- Generated RDNA3.5 DS execution bodies throw `UnimplementedInst` when the DS
  `gds` bit is set; for example `DsAddU32Ds` checks `inst_.gds` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/ds.cpp:38` through `:40`.
- `DS_CONSUME` and `DS_APPEND` are implemented as LDS counter operations and
  also reject `inst_.gds` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/ds.cpp:1969` through
  `:1991` and `:2003` through `:2025`; `DS_ORDERED_COUNT` is unimplemented at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/ds.cpp:2039` through
  `:2041`. The dynamic atomic kinds label append/consume as LDS-only at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:65` through `:66`.

Impact:

Legal RDNA3.5 DS encodings that select GDS can decode, but they stop at
execution instead of accessing a shared 4KiB GDS surface. Runtime-visible device
properties also report no GDS even though Chapter 1 says RDNA3.5 devices have
one.

### RDNA3_5-RJ-007: WGP-mode dispatch allows a single workgroup to reserve more than 64KiB of LDS

Manual evidence:

- Chapter 1.2.2.1 says each WGP has 128KiB of LDS, but a single workgroup may
  allocate up to 64KiB of LDS space at `rdna3.5/README.md:497`.

Rocjitsu evidence:

- `CommandProcessor::process_aql_packet()` uses `spi->max_wgp_lds_bytes()` as
  the WGP-mode LDS capacity and rejects only requests larger than that capacity
  at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:980` through
  `:999`.
- `ShaderProcessorInput::max_wgp_lds_bytes()` returns the size of the WGP LDS
  backing store at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/spi.h:259` through
  `:265`, and that backing store is constructed as the sum of the two sibling
  CU LDS sizes at `spi.h:270` through `:273`.
- WGP-mode placement allocates the aligned `group_segment_fixed_size` from this
  combined WGP LDS pool and advances `next_lds_alloc` without a separate
  64KiB-per-workgroup cap at `spi.h:218` through `:239`.

Impact:

With the normal 64KiB-per-CU topology, a WGP-mode dispatch requesting between
64KiB and 128KiB of LDS can pass validation and receive one contiguous
simulated WGP allocation. The manual's Chapter 1 storage scale is preserved, but
the per-workgroup allocation limit is not enforced.

### RDNA3_5-RJ-008: RDNA3.5 Wave64 dispatch is not selectable

Manual evidence:

- Chapter 2.1 says RDNA3.5 supports wave32 and wave64, and that shaders are
  compiled for one fixed wave size at `rdna3.5/README.md:543` through `:545`.
- The same section defines wave64 issue behavior, second-pass scalar
  input/output increments, and wave32 upper-`EXEC`/`VCC` ignored behavior at
  `rdna3.5/README.md:547` through `:565`.

Rocjitsu evidence:

- `RdnaIsaBase` sets `WF_SIZE = 32` and `WF_SIZE_MAX = 64` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/rdna_isa_base.h:50`
  through `:52`.
- `IsaExecComputeUnit` constructs the execution CU with `Isa::WF_SIZE` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.h:683` through `:685`.
- The RDNA3.5 config advertises `wave_front_size` 32 at
  `configs/gfx1151.json:25`.
- `CommandProcessor::process_aql_packet` computes waves per workgroup from
  `cus_[0]->wf_size()` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:941` through
  `:944`; the AMDHSA `ENABLE_WAVEFRONT_SIZE32` kernel descriptor bit is
  defined at
  `lib/rocjitsu/external_headers/hsa_headers/hsa/AMDHSAKernelDescriptor.h:190`,
  but is not consumed in that functional VM dispatch path.
- The DBT kernel-descriptor translator separately recognizes the RDNA wave-size
  descriptor bit at
  `lib/rocjitsu/src/rocjitsu/code/dbt/kernel_descriptor_translator.cpp:321`
  through `:332`, so this is scoped to the functional execution path rather
  than every rocjitsu descriptor consumer.

Impact:

RDNA3.5 wave64 kernels are dispatched and executed as wave32 in rocjitsu, so
wave64-specific `EXEC`/`VCC` behavior, issue passes, scalar carry/divergence
increments, and VGPR allocation granularity are not modeled.

### RDNA3_5-RJ-009: Workgroup size and residency limits are resource-only

Manual evidence:

- Chapter 2.3 says a WGP supports up to 32 workgroups in flight and a maximum
  of 1024 work-items per workgroup at `rdna3.5/README.md:590`.
- The same section says single-wave workgroups do not count against the
  32-workgroup limit, do not allocate a barrier resource, and treat barrier
  operations as `S_NOP` at `rdna3.5/README.md:592`.

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
  `:239`, but searches found no 32-workgroup cap or single-wave barrier
  resource elision rule.
- `S_BARRIER` always transitions the wave to barrier state through
  `execute_s_barrier_sopp` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:744`
  through `:747`; barrier resolution waits for all wavefronts in the workgroup
  at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:313` through
  `:337`.

Impact:

Rocjitsu can accept and schedule RDNA3.5 launches whose workgroup dimensions or
resident-workgroup count are outside the Chapter 2 hardware contract. It also
models single-wave barriers as ordinary barrier state transitions instead of a
launch-time no-barrier-resource / `S_NOP` special case.

### RDNA3_5-RJ-010: Graphics shader launch modes are not modeled

Manual evidence:

- Chapter 2.2.2 describes pixel, geometry, and hull shader waves, says the
  normal NGG geometry-engine launch initializes VGPRs with primitive/index and
  vertex-buffer data, and describes mesh-shader plus amplification-shader
  launch modes at `rdna3.5/README.md:573` through `:586`.

Rocjitsu evidence:

- `DispatchEntry` is compute/AQL shaped, with kernel dispatch identifiers,
  grid sizes, workgroup sizes, workgroup IDs, and WGP mode at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/dispatch_entry.h:43` through `:87`.
- `CommandProcessor::init_wavefront_regs` initializes AMDHSA compute SGPRs and
  local work-item IDs at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:174` through
  `:325`.
- Searches found no PS/GS/HS, geometry-engine, mesh-shader, or amplification
  shader launch path beyond generic shader-engine topology names.

Impact:

Rocjitsu is compute-dispatch oriented for this layer. It cannot emulate or
validate Chapter 2 graphics-stage wave creation, geometry-engine VGPR payloads,
mesh shader launch conversion, or amplification shader control.

### RDNA3_5-RJ-011: LDSDIR instructions are stubbed and WGP-mode legality is not enforced

Manual evidence:

- Chapter 2.3 says WGP mode exposes one contiguous WGP LDS and allows waves in
  a workgroup to be distributed across both CUs, but `LDS_PARAM_LOAD` and
  `LDS_DIRECT_LOAD` are not supported in WGP mode at `rdna3.5/README.md:602`
  through `:604`.

Rocjitsu evidence:

- RDNA3.5 generates concrete `LdsParamLoadLdsdir` and
  `LdsDirectLoadLdsdir` instruction classes at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/ldsdir.cpp:20` through
  `:46`.
- Both classes dispatch to shared helpers, but
  `execute_lds_param_load_ldsdir` and `execute_lds_direct_load_ldsdir` are
  empty at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:337`
  through `:343`.
- The command processor records `dp.wgp_mode` for placement and LDS-capacity
  checks at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:980`
  through `:999`, but the LDSDIR execute path has no WGP-mode rejection or
  decode-time legality guard.

Impact:

RDNA3.5 LDS parameter/direct load instructions decode but do not transfer data
in either CU or WGP mode. WGP-mode shaders that contain those unsupported
instructions can pass decode and execute as no-ops instead of being rejected or
flagged.

### RDNA3_5-RJ-012: RDNA3.5 HWREG get/set uses the wrong and incomplete state map

Manual/XML evidence:

- Chapter 3.1 lists `STATUS`, `MODE`, and `TRAPSTS` as 32-bit shader-visible
  state and lists `TBA`, `TMA`, `TTMP0`-`TTMP15`, and wait counters in the same
  visible-state table at `rdna3.5/README.md:638` through `:653`.
- The RDNA3.5 XML `OPR_HWREG` map assigns `hw_reg_mode` to ID 1,
  `hw_reg_status` to ID 2, `hw_reg_trapsts` to ID 3, `hw_reg_gpr_alloc` to ID
  5, `hw_reg_lds_alloc` to ID 6, `hw_reg_ib_sts` to ID 7, and PC/TBA/flat
  scratch IDs at `amdgpu_isa_rdna3_5.xml:177147` through `:177280`.
- XML represents `S_GETREG_B32`, `S_SETREG_B32`, and
  `S_SETREG_IMM32_B32` as HWREG access instructions at
  `amdgpu_isa_rdna3_5.xml:60462` through `:60547`.

Rocjitsu evidence:

- RDNA3.5 `SGetregB32Sopk::execute_impl` treats HWREG ID 1 as
  `wf.status_raw()` and handles only IDs 1, 4, 5, 6, and 7 before logging
  unhandled IDs at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/sopk.cpp:287` through
  `:317`.
- RDNA3.5 `SSetregB32Sopk` and `SSetregImm32B32Sopk` also only splice writes
  into `wf.status_raw()` for ID 1 at `rdna3_5/sopk.cpp:336` through `:356` and
  `:373` through `:392`.
- `Wavefront` stores raw MODE separately from raw STATUS at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:95` through `:110`, and the
  concrete RDNA3.5 wavefront exposes a separate `StatusReg` through
  `status_raw()`/`set_status_raw()` at `wavefront.h:608` through `:617`.

Impact:

RDNA3.5 `s_getreg_b32 hwreg(MODE,...)` observes STATUS in rocjitsu, while
architectural STATUS ID 2, TRAPSTS ID 3, TBA/TMA, PC, flat scratch, and several
allocation/status registers are unreachable or mapped to unrelated ad hoc
values. `s_setreg*` also writes STATUS through the MODE ID instead of applying
the architectural MODE/STATUS/TRAPSTS access policy.

### RDNA3_5-RJ-013: Raw `STATUS.EXECZ` and `STATUS.VCCZ` can drift from live masks

Manual evidence:

- Chapter 3.1 defines `EXECZ` and `VCCZ` as summary flags that indicate whether
  the live `EXEC` or `VCC` mask is all zero, using only the low 32 bits for
  wave32 at `rdna3.5/README.md:632` through `:635`.
- Chapter 3.2.2 repeats that wave32 hardware acts only on `EXEC[31:0]`, and
  `EXECZ` reflects that low-half state at `rdna3.5/README.md:669` through
  `:673`.

Rocjitsu evidence:

- The RDNA3.5 `StatusReg` layout includes `EXECZ` at bit 9 and `VCCZ` at bit 10
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/isa.h:43` through
  `:46`.
- `Wavefront::set_exec()`, `set_exec_raw()`, and `set_vcc()` update only the
  live mask fields at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:211`
  through `:238`.
- The scalar special-source resolver correctly synthesizes live `VCCZ` and
  `EXECZ` values from `wf.vcc()`/`wf.exec()` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/operand.cpp:768` through
  `:774`, but raw `status_raw()` remains an independently stored field exposed
  by HWREG access as described in `RDNA3_5-RJ-012`.

Impact:

Code that reads the special scalar `EXECZ`/`VCCZ` sources observes live masks,
but raw `STATUS` inspection can observe stale or manually written summary bits
that no longer match the architectural `EXEC`/`VCC` state.

### RDNA3_5-RJ-014: Trap, TBA/TMA, and trap-return state is not modeled

Manual/XML evidence:

- Chapter 3.1 lists `TRAPSTS`, `TBA`, `TMA`, and `TTMP0`-`TTMP15` as
  shader-visible trap state at `rdna3.5/README.md:641` through `:649`.
- Chapter 3.2.1 says `S_RFE_B64` is one of the direct PC operations and that
  `S_TRAP` saves the PC of the trap instruction itself at
  `rdna3.5/README.md:661` through `:663`.
- XML exposes `hw_reg_trapsts`, `hw_reg_shader_tba_lo`,
  `hw_reg_shader_tba_hi`, and the TMA/TBA return-message IDs at
  `amdgpu_isa_rdna3_5.xml:177171` through `:177224` and `:178184` through
  `:178204`.

Rocjitsu evidence:

- `Wavefront` stores status, mode, `EXEC`, `VCC`, `M0`, scratch/aperture bases,
  and wait counters, but has no `TRAPSTS`, `TBA`, or `TMA` state members at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:532` through `:545`.
- RDNA3.5 `S_TRAP` is marked `PROGRAM_TERMINATOR` and throws
  `util::UnimplementedInst` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/sopp.cpp:158` through
  `:170`.
- Shared `execute_s_rfe_b64_sop1` is empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2405`
  through `:2406`, and the RDNA3.5 `S_RFE_B64` class dispatches to that helper
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/sop1.cpp:1227`
  through `:1239`.
- Shared `S_SENDMSG_RTN_*` only returns a nonzero value for message `0x83` and
  returns zero for message IDs including `0x82`, `0x85`, and `0x86` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2431`
  through `:2478`; the XML names those values `msg_rtn_get_tma`,
  `msg_rtn_get_tba`, and `msg_rtn_get_tba_to_pc`.

Impact:

Trap handlers, trap status inspection, TBA/TMA queries, trap-entry PC save, and
trap return cannot be emulated faithfully. Legal trap-control instructions
either throw, no-op, or return zero instead of mutating the Chapter 3 trap
state.

### RDNA3_5-RJ-015: Indirect PC writes do not force DWORD alignment and zero targets halt

Manual evidence:

- Chapter 3.2.1 says the PC is a DWORD-aligned byte address whose low two bits
  are forced to zero at `rdna3.5/README.md:659`.
- The same section says `S_SETPC_B64`, `S_RFE_B64`, and `S_SWAPPC_B64` transfer
  the PC to and from even-aligned SGPR pairs, while branches are relative to the
  next instruction at `rdna3.5/README.md:661` through `:663`.

Rocjitsu evidence:

- `SSetpcB64Sop1::execute_impl` assigns
  `wf.pc = read_scalar64(ssrc0) - size_` without masking low address bits at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/sop1.cpp:1201` through
  `:1203`.
- `SSwappcB64Sop1::execute_impl` likewise reads the raw scalar target, assigns
  it minus `size_`, and writes the next PC to the destination at
  `rdna3_5/sop1.cpp:1221` through `:1224`.
- The outer CU loop special-cases `s_setpc` and `s_swappc`: if the raw scalar
  target is zero it halts the wave before executing the instruction at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:395` through `:407`.

Impact:

Misaligned indirect branch targets keep their low bits in rocjitsu instead of
being forced to DWORD alignment, and an indirect branch to address zero becomes
a wave halt rather than an architectural PC update.

### RDNA3_5-RJ-016: `EXEC==0` skip and counter-dependent issue rules are not modeled

Manual evidence:

- Chapter 3.2.3 says hardware may skip vector instructions when `EXEC==0`,
  with specific no-skip exceptions for SGPR/VCC writers, WMMA, `V_READLANE` /
  `V_WRITELANE`, buffer invalidations, selected wave64 issue-twice cases,
  export `Done`/`POS0`/`SKIP_EXPORT` behavior, and VMEM/LDS skip decisions that
  depend on outstanding VM/VScnt, EXPcnt, or LGKMcnt at
  `rdna3.5/README.md:675` through `:696`.

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

### RDNA3_5-RJ-017: Scalar 64-bit SGPR alignment is not enforced

Manual evidence:

- Chapter 3.3.1.3 says 64-bit SGPR operands must be even aligned, values wider
  than 64 bits require quad alignment, the low dword is in `SGPR[n]`, and
  hardware enforces alignment by ignoring low address bits at
  `rdna3.5/README.md:714` through `:733`.

Rocjitsu evidence:

- RDNA3.5 scalar 64-bit reads accept encoding values through 105 and read
  `ev`/`ev+1` directly, and TTMP 64-bit reads accept 108 through 122 and also
  read `ev`/`ev+1` directly, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/operand.cpp:816` through
  `:833`.
- RDNA3.5 scalar 64-bit writes similarly write `ev`/`ev+1` directly for SGPR
  and TTMP destinations at `rdna3_5/operand.cpp:920` through `:940`.
- The low-level SGPR accessor simply reads/writes the physical register index it
  is given at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/register_access.h:700`
  through `:719` and
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.h:379` through `:392`.

Impact:

An instruction encoded with an odd 64-bit scalar register pair can read or write
the odd/even-next pair in rocjitsu. Hardware forces the low address bit to zero,
so the same encoding should address the even-aligned pair instead.

### RDNA3_5-RJ-018: TTMP privilege and scalar out-of-range fallbacks are not modeled

Manual evidence:

- Chapter 3.3.1.4 says out-of-range scalar sources return zero and
  out-of-range scalar writes are ignored at `rdna3.5/README.md:747` through
  `:750`.
- The same section says user-mode TTMP reads return zero, user-mode TTMP writes
  are ignored, and SALU instructions that fail to write a TTMP do not update SCC
  at `rdna3.5/README.md:752` through `:754`.
- Chapter 3.4 defines `STATUS.PRIV` as the bit that indicates trap-handler
  privileged mode and grants TTMP write access at `rdna3.5/README.md:928`
  through `:930`.

Rocjitsu evidence:

- RDNA3.5 `StatusReg` exposes a `PRIV` bit at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/isa.h:33` through `:40`,
  but searches found no RDNA3.5 operand or execution path consulting `PRIV` for
  TTMP access.
- RDNA3.5 scalar reads access TTMP encodings 108-123 directly through the SGPR
  file at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/operand.cpp:709`
  through `:721`, and scalar writes access the same range directly at
  `rdna3_5/operand.cpp:876` through `:901`.
- Unsupported scalar source and destination encodings throw
  `std::logic_error` rather than returning zero or ignoring the write at
  `rdna3_5/operand.cpp:776` through `:778`, `:873`, and `:917` through `:950`.
- Most SALU helpers update SCC after destination writes, for example
  `execute_s_not_b32_sop1` writes the scalar destination then writes SCC at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2177`
  through `:2188`.

Impact:

User shaders can read and write TTMP storage in rocjitsu even when
`STATUS.PRIV==0`; invalid scalar operands can terminate execution with an
exception instead of using the architecture's zero/ignored fallback; and failed
TTMP writes are not prevented from updating SCC.

### RDNA3_5-RJ-019: S_MOVREL does not enforce scalar-region and B64 index rules

Manual evidence:

- Chapter 3.3.1.4 says `S_MOVREL` indexing is allowed only within SGPRs and
  TTMPs, must stay within the base range, must not reach M0/EXEC/inline
  constants, uses S0 for out-of-range sources, writes nothing for out-of-range
  destinations, and for `*MOVREL*_B64` the index low bit is ignored at
  `rdna3.5/README.md:765` through `:773`.

Rocjitsu evidence:

- `S_MOVRELS_B32/B64` compute `src_reg = base + index * width_words`, build a
  normal `OPR_SSRC` operand, and then read it through the regular scalar
  accessor at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/sop1.cpp:1047` through
  `:1088`.
- `S_MOVRELD_B32/B64` compute `dst_reg = base + index * width_words`, build a
  normal `OPR_SDST` operand, and then write it through the regular scalar
  accessor at `rdna3_5/sop1.cpp:1105` through `:1145`.
- `S_MOVRELSD_2_B32` separately adds the two M0 byte fields to source and
  destination encodings and then uses normal scalar read/write accessors at
  `rdna3_5/sop1.cpp:1163` through `:1171`.

Impact:

Relative scalar moves can cross from SGPR/VCC into TTMP or special scalar-source
encodings in rocjitsu, write M0/EXEC through an out-of-range indexed
destination, throw on unsupported indexed operands, or use `index * 2` for B64
instead of masking the low index bit as hardware does.

### RDNA3_5-RJ-020: SAVEEXEC/WREXEC invalid destinations can affect special state or block EXEC updates

Manual/XML evidence:

- Chapter 3.3.1.4 says normal SALU out-of-range rules apply, but `WREXEC` and
  `SAVEEXEC` write the `EXEC` mask even when `SDST` is out of range at
  `rdna3.5/README.md:747` through `:756`.
- XML marks SAVEEXEC/WREXEC scalar destinations as `OPR_SREG`, not the broader
  `OPR_SSRC` set, and exposes implicit `EXEC`/`SCC` operands at
  `amdgpu_isa_rdna3_5.xml:40989` through `:41023` and `:42753` through
  `:42789`.

Rocjitsu evidence:

- RDNA3.5 generated SAVEEXEC/WREXEC constructors use `OperandType::OPR_SREG`
  for `SDST`, for example `SAndSaveexecB32Sop1` and `SAndSaveexecB64Sop1` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/sop1.cpp:576` through
  `:599`.
- The runtime scalar write helpers ignore the operand type and branch only on
  the numeric encoding: 32-bit writes can update M0/EXEC encodings 125-127, and
  unsupported 64-bit destinations throw, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/operand.cpp:876` through
  `:950`.
- SAVEEXEC/WREXEC helpers write `SDST` before updating `EXEC`, for example
  `execute_s_and_saveexec_b32_sop1`, `execute_s_and_saveexec_b64_sop1`,
  `execute_s_and_not0_wrexec_b32_sop1`, and
  `execute_s_and_not0_wrexec_b64_sop1` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:510`
  through `:528` and `:593` through `:609`.

Impact:

Invalid destination encodings for these instructions can mutate M0/EXEC_HI or
throw before the architectural `EXEC` update occurs, instead of ignoring the
invalid scalar destination while still applying the SAVEEXEC/WREXEC `EXEC`
side effect.

### RDNA3_5-RJ-021: VGPR out-of-range consequences are not enforced for ordinary operands

Manual evidence:

- Chapter 3.3.2.2 defines VGPR OOR using the first and last dword touched by an
  operand; destination OOR turns the instruction into a NOP, `V_SWAP` and
  `V_SWAPREL` discard when either destination is OOR, multiple-destination VALU
  instructions write no GPRs if any destination is OOR, and source OOR falls
  back to VGPR0 for VMEM/export and VALU sources, with VOPD using a separate
  source-address rule, at `rdna3.5/README.md:783` through `:815`.

Rocjitsu evidence:

- RDNA3.5 operand reads and writes convert VGPR operands to
  `wf.vgpr_alloc().base + voff` and then call raw VGPR accessors without
  checking `voff` or multi-dword width against `wf.vgpr_alloc().count`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/operand.cpp:1035`
  through `:1120`.
- `Isa::resolved_vgpr_offset()` accepts every 8-bit `OPR_VGPR` and every
  9-bit source value 256-511 as an offset, without consulting the wave's
  allocated VGPR count, at `rdna3_5/operand.cpp:978` through `:986`.
- The physical VGPR access layer reads and writes the supplied physical index
  directly at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/register_access.h:722`
  through `:742` and
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.h:702` through `:720`.
- Memory, flat/global/scratch, DS, and export setup paths also form physical
  VGPR bases directly from instruction fields, for example
  `shared/execute_shared.h:165`, `rdna3_5/mubuf.cpp:409`,
  `rdna3_5/ds.cpp:42`, and `rdna3_5/flat.cpp:52`.

Impact:

An instruction that addresses beyond the wave's allocated `VGPR_SIZE` can read
or write whatever physical VGPR slot the calculated index names, or reach
implementation-defined storage, instead of applying the ISA's VGPR0 fallback,
NOP/nullification, or all-destination suppression rules.

### RDNA3_5-RJ-022: V_MOVREL out-of-range cases throw instead of applying VGPR0/NOP rules

Manual evidence:

- Chapter 3.3.2.2 says `V_MOVREL` indexed operands are out of range when the
  index is greater than 255 or when `Vs + M0` or `Ve + M0` reaches
  `VGPR_SIZE`; source OOR uses VGPR0 and destination OOR makes the instruction a
  NOP, at `rdna3.5/README.md:801` through `:815`.

Rocjitsu evidence:

- RDNA3.5 `V_MOVRELD_B32` resolves the destination base, adds signed `M0`, and
  throws `util::UnimplementedInst` when the relative index is outside
  `wf.vgpr_alloc().count`, at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/vop1.cpp:6778` through
  `:6793`.
- RDNA3.5 `V_MOVRELS_B32` similarly throws for out-of-range indexed sources at
  `rdna3_5/vop1.cpp:6908` through `:6922`.
- The VOP3 `V_MOVRELD_B32` and `V_MOVRELS_B32` variants use the same
  throw-on-OOR pattern at `rdna3_5/vop3.cpp:3548` through `:3563` and
  `rdna3_5/vop3.cpp:3625` through `:3639`.

Impact:

Programs that rely on architectural `V_MOVREL` OOR behavior can terminate
emulation with an unimplemented-instruction exception instead of reading VGPR0
or leaving the destination untouched.

### RDNA3_5-RJ-023: msg_dealloc_vgprs does not deallocate VGPRs or suppress vector instructions

Manual/XML evidence:

- Chapter 3.3.2.1 says a wave may deallocate all VGPRs via `S_SENDMSG`; after
  that it cannot reallocate them and the only valid action is termination, at
  `rdna3.5/README.md:781`.
- XML's `msg_dealloc_vgprs` description says the wave has no VGPRs or scratch
  memory after the message, context save is not responded to, and all vector
  instructions are ignored at `amdgpu_isa_rdna3_5.xml:178068`.

Rocjitsu evidence:

- Shared `S_SENDMSG` execution is an empty stub at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2428`.
- `Wavefront` stores `num_vgprs_` and `vgpr_alloc_`, and dispatch initializes
  them at launch, but searches found no `S_SENDMSG` path that clears
  `num_vgprs_`, frees the VGPR allocation, marks a no-VGPR state, or suppresses
  later vector instruction execution. The relevant storage and launch writes
  are at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:83` through `:93`,
  `wavefront.h:200` through `:202`, and
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:119` through `:146`.

Impact:

After `s_sendmsg msg_dealloc_vgprs`, rocjitsu keeps the VGPR allocation live
and continues executing later vector instructions, while the manual/XML
contract says the wave has no VGPRs and vector instructions are ignored.

### RDNA3_5-RJ-024: Memory-return destination OOR does not nullify by clearing EXEC

Manual evidence:

- Chapter 3.3.3 says memory, LDS, and GDS reads and atomics with return are
  nullified when any destination VGPR is out of range, by issuing the
  instruction as if `EXEC` were cleared to zero. The range test covers all
  VGPRs that could be returned, includes the extra PRT VGPR, and nullifies
  return atomics with out-of-range destinations at `rdna3.5/README.md:820`
  through `:829`.
- Chapter 10.10 says PRT texture fetches can write status to
  `DST_VGPR+Num_VGPRS`, and the shader must allocate that extra VGPR at
  `rdna3.5/README.md:4238` through `:4248`.

Rocjitsu evidence:

- `VectorMemState` stores `dst_reg_base`, `elem_size`, and `num_elems`, but has
  no field for the architectural VGPR allocation bound, pre-issue
  destination-validity result, or PRT extra destination at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:87` through `:140`.
- Shared vector-memory completion computes the destination VGPR count, zeroes
  address-OOB lanes for buffer loads, and writes `d.dst_reg_base + i` directly
  for active lanes, without checking the destination range or converting the
  operation to an `EXEC==0` issue at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:119` through `:205`.
- DS read2 completion writes `d.ds2_dst_reg_base + i` directly, also without a
  destination-range nullification check, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:590` through `:635`.
- Generated memory instruction setup forms physical destination bases directly
  from instruction fields, for example `BUFFER_LOAD_U8`, `FLAT_LOAD_U8`, and
  `DS_ADD_U32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/mubuf.cpp:400` through
  `:414`, `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/flat.cpp:50`
  through `:64`, and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/ds.cpp:38` through `:55`.

Impact:

Out-of-range memory-return destinations can write physical VGPRs past the
wave's allocation instead of issuing a null operation. PRT fetches also lack the
extra-status-VGPR range check that the manual says must nullify the fetch.

### RDNA3_5-RJ-025: Memory source-OOR and alignment/MEMVIOL policy are not modeled

Manual evidence:

- Chapter 3.3.3 says memory, LDS, and GDS reads and atomics with return produce
  undefined data if any source VGPR or SGPR is out of range at
  `rdna3.5/README.md:822` through `:825`.
- The same section defines `SH_MEM_CONFIG.alignment_mode`, `DWORD` automatic
  alignment, `UNALIGNED` mode, formatted-operation alignment requirements, and
  MEMVIOL for atomics that are not aligned to the data size at
  `rdna3.5/README.md:834` through `:844`.

Rocjitsu evidence:

- Buffer address calculation reads SRD SGPRs, `soffset`, and optional VGPR
  address regions directly from `wf.sgpr_alloc().base + ...` and
  `wf.vgpr_alloc().base + ...` without source-range classification at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h:51`
  through `:106`.
- Flat/scratch/global and DS address calculation use the same direct source
  reads for SGPR/VGPR address operands at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h:35`
  through `:80` and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:55`
  through `:85`.
- Generated DS atomics such as `DS_ADD_U32` build an atomic memory operation
  and read `DATA0` directly from a physical VGPR base, with no alignment-mode
  or MEMVIOL check at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/ds.cpp:38` through `:55`.
- Searches in the runtime found no `SH_MEM_CONFIG`, `alignment_mode`, or
  `MEMVIOL` model in the RDNA3.5 vector-memory path; `VectorMemState` carries
  request data and lane masks but no alignment mode or memory-violation status
  at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:87` through `:140`.

Impact:

Memory operations with source operands outside the wave's legal GPR range read
some concrete simulated register value instead of applying the manual's
undefined-data source rule. Misaligned memory and atomic cases also execute
without the configured alignment-mode and MEMVIOL behavior described by the ISA.

### RDNA3_5-RJ-026: Image DMASK and PRT destination sizing are not modeled

Manual evidence:

- Chapter 3.3.3 says PRT extra VGPRs participate in destination out-of-range
  nullification and image loads and stores consider `DMASK` bits when making an
  out-of-bounds determination at `rdna3.5/README.md:825` through `:829`.
- Chapter 10.10 describes the extra PRT status DWORD at
  `DST_VGPR+Num_VGPRS` when `MIN_LOD_WARN` enables PRT behavior at
  `rdna3.5/README.md:4238` through `:4248`.

Rocjitsu evidence:

- The visible RDNA3.5 image-load execution slice is a minimal stub: for example
  `ImageMsaaLoadMimg::execute_impl()` ignores the wavefront at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/mimg.cpp:450` through
  `:470`.
- The shared vector-memory state used by implemented memory paths has ordinary
  destination sizing and D16 flags, but no `DMASK`, PRT status, or texture-fail
  destination metadata at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/mem_state.h:87` through `:140`.

Impact:

Even where image fields decode, rocjitsu does not model the manual's
`DMASK`-dependent destination footprint or the PRT extra-status destination
that participates in out-of-range nullification.

### RDNA3_5-RJ-027: LDS allocation granularity and per-workgroup bounds are not modeled accurately

Manual evidence:

- Chapter 3.3.4 says LDS allocations are 0-64 KiB per wave/workgroup, shared by
  all waves in the workgroup, allocated in 1024-byte blocks, and all LDS
  accesses are restricted to the allocated space at `rdna3.5/README.md:846`
  through `:847`.
- The same section describes the two 64 KiB LDS blocks, CU-mode same-side
  placement and no-cross/no-wrap rule, WGP-mode placement or straddling
  behavior, and pixel-parameter side placement at `rdna3.5/README.md:848`
  through `:855`.

Rocjitsu evidence:

- Dispatch validation and placement round `group_segment_fixed_size` to 256
  bytes, not 1024 bytes, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:99` through
  `:101`, `:988` through `:999`,
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:243` through `:249`,
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.h:239` through `:245`, and
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/spi.h:232` through `:239`.
- WGP-mode uses a sibling-CU backing store whose size is the sum of the two CU
  LDS sizes at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/spi.h:270` through `:273`,
  and wavefronts carry only a placement-selected `lds_base`, not an allocated
  byte size, at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:167` through
  `:176` and `:512` through `:513`.
- DS address calculation adds the DS byte offset and `wf.lds_base()` directly
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:55`
  through `:85`.
- LDS read/write helpers check only the total backing vector size, returning
  zero or dropping writes when the backing is crossed, at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/lds.h:31` through `:107` and `:136`
  through `:168`; the local-memory pipeline passes those backing-relative
  addresses straight through at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:506` through `:527`
  and `:590` through `:635`.

Impact:

A shader can access bytes outside its requested LDS allocation but still inside
the simulated CU/WGP LDS backing, and rocjitsu will read or write that storage
instead of applying the manual's per-allocation out-of-range behavior. The
256-byte allocation accounting can also admit or pack resident workgroups in a
layout that does not match the documented 1024-byte block allocation.

### RDNA3_5-RJ-028: LDS-specific alignment mode, `LDS_CONFIG` reporting, and MEMVIOL controls are not modeled

Manual evidence:

- Chapter 3.3.4.1 says DS load/store sizes may be byte-aligned only when
  alignment mode is unaligned; otherwise LDS forces alignment by zeroing low
  address bits, and 32-bit/64-bit atomics require 4-byte/8-byte alignment at
  `rdna3.5/README.md:856` through `:859`.
- The same section says LDS operations report `MEMVIOL` for out-of-range
  addresses when `LDS_CONFIG.ADDR_OUT_OF_RANGE_REPORTING` is set, report
  `MEMVIOL` for misaligned LDS accesses in STRICT or DWORD_STRICT mode, and use
  native LDS/GDS alignment masks for B8/B16/B32/B64/B96/B128 accesses at
  `rdna3.5/README.md:860` through `:890`.

Rocjitsu evidence:

- Searches found no rocjitsu state or helper for `LDS_CONFIG`,
  `ADDR_OUT_OF_RANGE_REPORTING`, STRICT/DWORD_STRICT alignment modes, or
  `MEMVIOL` in the RDNA3.5 LDS path.
- The shared DS address helper directly computes `addr + offset + wf.lds_base()`
  without an alignment mask at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h:55`
  through `:85`.
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
array accesses. Rocjitsu can silently align differently, access misaligned
bytes, return zero, or drop writes where hardware behavior depends on LDS
alignment mode, `LDS_CONFIG.ADDR_OUT_OF_RANGE_REPORTING`, and `MEMVIOL`
reporting.

### RDNA3_5-RJ-029: STATUS field layout, write permissions, and wave-creation state are incomplete

Manual evidence:

- Chapter 3.4.1 says STATUS fields are readable but not normally writable by
  the shader, with only selected fields writable in the trap handler, and that
  these bits are initialized at wave-creation time at
  `rdna3.5/README.md:914` through `:916`.
- The STATUS table includes export, barrier, halt, trap, validity, performance,
  conditional-debug, VGPR-release, LDS-parameter, GS-allocation, must-export,
  idle, and scratch-enable bits at `rdna3.5/README.md:922` through `:950`.

Rocjitsu evidence:

- The RDNA3.5 `StatusReg` bitfield models only `SCC`, priority, `PRIV`,
  `TRAP_EN`, `EXECZ`, `VCCZ`, `IN_TG`, `IN_BARRIER`, `HALT`, `TRAP`, `VALID`,
  `ECC_ERR`, and `ALLOW_REPLAY` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/isa.h:22` through `:61`;
  it omits manual STATUS fields such as `EXPORT_RDY`, `SKIP_EXPORT`,
  `PERF_EN`, `CDBG_USER`, `CDBG_SYS`, `FATAL_HALT`, `NO_VGPRS`,
  `LDS_PARAM_RDY`, `MUST_GS_ALLOC`, `MUST_EXPORT`, `IDLE`, and `SCRATCH_EN`.
- `Wavefront::read_scc()` and `write_scc()` update only bit 0 at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:414` through `:424`, and
  the raw STATUS accessors simply cast or assign the `StatusReg` at
  `wavefront.h:608` through `:617`.
- `Wavefront::reset()` explicitly does not change the status register while
  clearing other dynamic dispatch state at `wavefront.h:459` through `:491`,
  and `ComputeUnitCore::dispatch_wf()` initializes PC, register allocations,
  `EXEC`, `VCC`, `M0`, apertures, and run state, but does not initialize the
  manual STATUS bits at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:136` through `:152`.
- `RDNA3_5-RJ-012` and `RDNA3_5-RJ-013` cover the HWREG selector mismatch and
  stale raw `STATUS.EXECZ`/`STATUS.VCCZ`; this gap covers the remaining STATUS
  field inventory, permissions, and initialization semantics.

Impact:

Shaders or debug paths that read STATUS can observe a partial, stale, or
non-architectural status word. Export waits, skipped exports, LDS parameter
readiness, wave validity/idle state, VGPR release state, scratch enable, trap
write restrictions, and wave-creation initialization are not represented by the
RDNA3.5 STATUS model.

### RDNA3_5-RJ-030: MODE floating-point and exception controls are not applied

Manual evidence:

- Chapter 3.4.2 says MODE `FP_ROUND` controls VALU floating-point rounding,
  `FP_DENORM` controls denormal flushing for VALU, LDS, and VMEM atomics,
  `DX10_CLAMP` controls NaN clamp behavior and exception suppression with
  `CLAMP`, and `IEEE` changes min/max NaN and signed-zero behavior at
  `rdna3.5/README.md:954` through `:967`.
- The same MODE table says `TRAP_AFTER_INST` forces trap-handler entry after
  each instruction when traps are enabled, `EXCP_EN` controls exception
  enables, and `FP16_OVFL` clamps overflowed FP16 VALU results to max finite
  while preserving true infinities at `rdna3.5/README.md:968` through `:977`.

Rocjitsu evidence:

- `Wavefront` stores raw MODE and synchronizes only the VGPR-MSB subfield used
  by register indexing at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:103`
  through `:128` and `:535` through `:537`.
- RDNA3.5 `S_ROUND_MODE` and `S_DENORM_MODE` decode and dispatch to shared
  helper functions at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/sopp.cpp:171` through
  `:195`, but those helpers are empty at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1643`
  through `:1646` and `:2422` through `:2425`.
- Representative floating-point helpers use host `std::fmax`/`std::fmin` or
  SIMD equivalents without consulting `wf.mode_raw()`, for example
  `V_MAX_F32` at `execute_shared.h:13877` through `:13905` and `V_MIN_F32` at
  `:15184` through `:15212`.
- Searches found no RDNA3.5 runtime state or execution path for
  `TRAP_AFTER_INST`, `EXCP_EN`, `DX10_CLAMP`, or `FP16_OVFL` outside manual/test
  names and comments, and the memory/atomic paths audited earlier do not
  consult MODE denormal fields.

Impact:

Changing MODE through the documented instructions or HWREG paths does not alter
RDNA3.5 floating-point, denormal, trap-after-instruction, exception-enable, or
FP16-overflow behavior. Programs that depend on non-default rounding, denormal
handling, DX10/IEEE min/max differences, or FP16 overflow clamping can match the
host default instead of the architectural MODE state.

### RDNA3_5-RJ-031: RDNA3.5 scalar resolution retains non-RDNA3 FLAT_SCRATCH aliases

Manual/XML evidence:

- Chapter 3.1 lists `FLAT_SCRATCH` as separate wave state from scalar GPRs at
  `rdna3.5/README.md:637`.
- Chapter 3.3.1 says scalar selectors `0-105` are SGPRs, selector `106/107`
  is VCC, selectors `108-123` are TTMPs, and `124-127` are NULL/M0/EXEC at
  `rdna3.5/README.md:739` through `:750`. The scalar source table repeats
  `0-105` as `SGPR 0 105` and marks `209-232` reserved at
  `rdna3.5/README.md:1479` through `:1492`.
- RDNA3.5 XML likewise lists scalar GPRs as an incrementing `s0` range and
  then separately enumerates `VCC_LO`, `VCC_HI`, and `NULL` at
  `amdgpu_isa_rdna3_5.xml:179558` through `:179565` and
  `amdgpu_isa_rdna3_5.xml:179511` through `:179524`. Searches found no
  RDNA3.5 XML scalar source names for `SRC_FLAT_SCRATCH_BASE_LO/HI`; hits for
  values `230/231` are VGPR names, not scalar flat-scratch selectors.

Rocjitsu evidence:

- The generated RDNA3.5 operand bounds classify scalar source and destination
  values through 105 as SGPRs at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/operand_types.h:102`
  through `:150`, and display code reports those source values as SGPR names
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/operand.cpp:248`
  through `:251`.
- Execution special-cases source selector `102` and `103` before the ordinary
  SGPR range, returning `wf.scratch_base()` low/high halves at
  `rdna3_5/operand.cpp:709` through `:737`; the 64-bit source path returns the
  whole scratch base for selector `102` at `:816` through `:846`.
- The same resolver accepts reserved scalar source selector values `230` and
  `231` as scratch-base low/high aliases through `can_resolve_src_scalar()` and
  the read helpers at `rdna3_5/operand.cpp:810` through `:846`.
- Destination writes to selector `102` and `103` mutate `wf.scratch_base()`
  instead of SGPR102/SGPR103 at `rdna3_5/operand.cpp:876` through `:884`, and
  the 64-bit destination path replaces the full scratch base for selector `102`
  at `rdna3_5/operand.cpp:920` through `:923`.

Impact:

RDNA3.5 instructions that legally read or write SGPR102/SGPR103 can observe or
clobber rocjitsu's per-wave scratch base instead of the scalar register file.
Disassembly and dependency metadata can still report an SGPR operand while the
executor accesses scratch state, and reserved scalar source encodings `230/231`
can receive non-reserved behavior.

### RDNA3_5-RJ-032: FLAT_SCRATCH launch alignment and access policy are incomplete

Manual evidence:

- Chapter 3.4.7 says `FLAT_SCRATCH` is initialized by wave-launch hardware for
  waves with scratch allocation, returns zero when no scratch is allocated, is
  read-only outside the trap handler, and is a 256-byte-aligned byte address at
  `rdna3.5/README.md:1046` through `:1056`.
- Chapter 3.4 and 3.4.8 also state that `FLAT_SCRATCH_LO/HI` are read-only
  HWREG targets writable only while in a trap handler at
  `rdna3.5/README.md:904` through `:912` and `rdna3.5/README.md:1084`
  through `:1092`.

Rocjitsu evidence:

- `Wavefront` stores a raw scratch base and exposes an unchecked
  `set_scratch_base()` setter at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:253` through `:259`; reset
  zeroes `scratch_base_` and `scratch_lane_size_` at `wavefront.h:477`
  through `:483`.
- Dispatch initializes scratch only when `private_segment_fixed_size > 0`,
  derives each wave base as
  `scratch_pool + global_wave_idx * private_segment_fixed_size * wf_size`, and
  stores it without a 256-byte alignment check or rounding at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:329` through
  `:348`.
- Scratch and flat-private address helpers consume the stored base directly at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/addr_calc.cpp:68`
  through `:84` and `:95` through `:112`.
- Existing `RDNA3_5-RJ-012` covers the incomplete RDNA3.5 HWREG get/set map for
  `FLAT_SCRATCH_LO/HI`; `RDNA3_5-RJ-031` covers the scalar-selector path that
  can mutate the scratch base outside the documented trap-only write route.

Impact:

Rocjitsu can hand later waves a `FLAT_SCRATCH` value that is not 256-byte
aligned when the backing address or per-wave scratch pitch is not aligned, and
the access-control contract is split across incomplete HWREG handling plus
generic scalar-selector writes. Kernels or probes that depend on the
architectural alignment or read-only/trap-only behavior can diverge from
RDNA3.5 hardware.

### RDNA3_5-RJ-033: Time-counter accesses are incomplete and synchronous

Manual/XML evidence:

- RDNA3.5 Section 3.4.10 says `SHADER_CYCLES` is a 20-bit graphics-core-clock
  counter read with `S_GETREG`, not synchronized across SIMDs and intended only
  for within-wave deltas, at `rdna3.5/README.md:1167` through `:1174`.
- The same section says `REALTIME` is a 64-bit fixed-frequency clock read with
  `S_SENDMSG_RTN_B64` followed by `S_WAITCNT LGKMcnt == 0` at
  `rdna3.5/README.md:1176` through `:1183`.
- XML exposes `hw_reg_shader_cycles` and `msg_rtn_get_realtime` at
  `amdgpu_isa_rdna3_5.xml:177277` through `:177279` and
  `amdgpu_isa_rdna3_5.xml:178189` through `:178191`.

Rocjitsu evidence:

- RDNA3.5 `SGetregB32Sopk::execute_impl` handles only HWREG IDs 1, 4, 5, 6,
  and 7 before warning and returning zero for other IDs at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/sopk.cpp:287` through
  `:317`; `hw_reg_shader_cycles` ID 29 therefore falls into the default path.
  This is the time-specific consequence of the broader HWREG map gap tracked in
  `RDNA3_5-RJ-012`.
- RDNA3.5 `S_SENDMSG_RTN_B32` and `S_SENDMSG_RTN_B64` dispatch to shared
  helpers at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/sop1.cpp:1241`
  through `:1277`. The shared helpers return `engine->global_time()` for
  message `0x83` and write the scalar destination immediately at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:2431`
  through `:2488`.
- `SimulationEngine::global_time()` is the latest processed simulation tick at
  `lib/simdojo/include/simdojo/sim/simulation.h:268` through `:270`. The
  send-message path does not model a 100 MHz REFCLK domain, a 64-bit
  idle-running realtime source, or an outstanding LGKM dependency that must be
  retired by `S_WAITCNT`.

Impact:

`S_GETREG(SHADER_CYCLES)` returns zero rather than a 20-bit shader-cycle value,
and `S_SENDMSG_RTN_B64 REALTIME` produces a synchronous simulator tick instead
of an LGKM-counted return message in the architectural realtime clock domain.
Timing probes and code that relies on the documented wait dependency can
observe non-hardware behavior.

### RDNA3_5-RJ-034: Optional `TG_SIZE` system SGPR launch payload is not initialized

Manual evidence:

- Section 3.5.3.4 says compute launch appends a packed system SGPR containing
  `{first_wave, 6'h00, wave_id_in_group[4:0], 2'h0,
  ordered_append_term[11:0], work_group_size_in_waves[5:0]}` when
  `COMPUTE_PGM_RSRC2.tg_size_en` is set at `rdna3.5/README.md:1284`
  through `:1294`.

Rocjitsu evidence:

- The HSA descriptor header defines
  `COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO` at
  `lib/rocjitsu/external_headers/hsa_headers/hsa/AMDHSAKernelDescriptor.h:126`
  through `:148`.
- `DispatchEntry` carries user-SGPR count, workgroup-id enable flags, and
  wave/workgroup dimensions, but no workgroup-info enable or stored `TG_SIZE`
  payload field at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/dispatch_entry.h:43` through `:92`.
- `CommandProcessor::process_aql_packet()` reads user-SGPR count and the
  workgroup-id enable bits, but does not read
  `COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:930` through
  `:1048`.
- `CommandProcessor::init_wavefront_regs()` writes AMDHSA user SGPRs, optional
  kernarg preload values, and enabled `workgroup_id_{x,y,z}` SGPRs, then moves
  on to RDNA4/gfx1250 TTMP payload and VGPR workitem IDs; there is no
  `TG_SIZE` system SGPR write at `command_processor.cpp:181` through `:305`.
- Static searches for `ENABLE_SGPR_WORKGROUP_INFO`, `TG_SIZE`, `tg_size`,
  `first_wave`, `wave_id_in_group`, `ordered_append`, and
  `work_group_size_in_waves` found no rocjitsu launch payload implementation
  outside the descriptor constant.

Impact:

RDNA3.5 compute kernels that request the `TG_SIZE` system SGPR receive whatever
stale or zero SGPR value happens to be in the next packed SGPR slot instead of
the documented wave/workgroup-size payload.

### RDNA3_5-RJ-035: 2D/3D `work_group_id0` is linearized when only the X SGPR is enabled

Manual evidence:

- Section 3.5.3 says enabled SGPR launch values are packed consecutively with
  no alignment gaps at `rdna3.5/README.md:1203` through `:1207`.
- The compute table says `COMPUTE_PGM_RSRC2.tgid_x_en`,
  `COMPUTE_PGM_RSRC2.tgid_y_en`, and `COMPUTE_PGM_RSRC2.tgid_z_en` append
  `work_group_id0[31:0]`, `work_group_id1[31:0]`, and
  `work_group_id2[31:0]` respectively at `rdna3.5/README.md:1288` through
  `:1294`.

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
  grid_wg_id_x : global_wg_id;` at `command_processor.cpp:265` through
  `:284`.

Impact:

For a 2D or 3D dispatch where the kernel requests only `tgid_x_en`, rocjitsu
writes the linear workgroup ordinal instead of the X coordinate. For example,
with `grid_wgs_x == 3`, `global_wg_id == 4` should initialize
`work_group_id0` as `1`, but rocjitsu writes `4`.

### RDNA3_5-RJ-036: RDNA3.5 compute TTMP6-11 launch payloads are not initialized

Manual evidence:

- Section 3.5.3.4 says compute launch initializes `TTMP4` and `TTMP5` to zero,
  `TTMP6` and `TTMP7` to the dispatch-packet address, `TTMP8` through
  `TTMP10` to dispatch-grid X/Y/Z, and `TTMP11` to
  `{26'b0, wave_id_in_workgroup[5:0]}` at `rdna3.5/README.md:1284` through
  `:1308`.

Rocjitsu evidence:

- `CommandProcessor::init_wavefront_regs()` receives `wf_index_in_wg` and has
  the dispatch pointer/grid fields available through `DispatchEntry`, but the
  only TTMP launch write path is guarded for
  `ROCJITSU_CODE_ARCH_GFX1250` or `ROCJITSU_CODE_ARCH_RDNA4` at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:174` through
  `:298`.
- That RDNA4/gfx1250 branch writes only `TTMP7` and `TTMP9`, using RDNA4's
  different packed workgroup-id payload; it does not initialize RDNA3.5
  `TTMP4` through `TTMP11` and does not write the dispatch-packet address,
  dispatch grid X/Y/Z, or wave-id payload required by RDNA3.5.
- `DispatchEntry` stores grid dimensions and, for host-accessible dispatches,
  a dispatch pointer, but it has no explicit TTMP launch-payload state at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/dispatch_entry.h:43` through `:92`.
- Static searches for `TTMP6`, `TTMP7`, `TTMP8`, `TTMP9`, `TTMP10`, `TTMP11`,
  `dispatch grid`, `dispatch packet addr`, and `wave_id_in_workgroup` found no
  RDNA3.5 launch initialization outside scalar operand naming tables and the
  RDNA4/gfx1250 branch.

Impact:

RDNA3.5 compute or trap/debug code that reads `TTMP6` through `TTMP11` observes
zero or stale simulator SGPR storage instead of the documented dispatch-packet,
grid, and wave-id launch metadata.

## No-Gap Notes

- Chapter 3.1-3.2 `EXEC` active-lane narrow match: `Wavefront::exec()` masks to
  the active lane width, while `exec_raw()` preserves upper bits for wave32
  scalar uses at `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:211` through
  `:230`.
- Chapter 3.3.2 allocation-bookkeeping narrow match: rocjitsu does track
  maximum and allocated VGPR counts on the wavefront, decodes the AMDHSA
  granulated VGPR count at dispatch, and allocates a per-wave VGPR slice at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:83` through `:93`,
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:932` through
  `:968`, and `lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp:119`
  through `:146`. `RDNA3_5-RJ-021` through `RDNA3_5-RJ-023` cover missing
  runtime edge behavior rather than total absence of VGPR allocation state.
- Chapter 3.3.3 address-OOB lane narrow match: buffer address calculation does
  maintain `lane_mask`/`exec_mask` and excludes address-out-of-bounds lanes,
  while vector-memory completion zeroes address-OOB load lanes and drops store
  lanes via the lane mask at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h:35`
  through `:115` and
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_pipeline.cpp:137` through
  `:168`. `RDNA3_5-RJ-024` through `RDNA3_5-RJ-026` cover destination GPR
  range, PRT/DMASK sizing, source-OOR, and alignment/MEMVIOL behavior rather
  than the existing buffer address-OOB lane mask path.
- Chapter 3.3.4 VGPR-range overlap: the LDS source-VGPR and destination-VGPR
  range rules are covered by `RDNA3_5-RJ-021` for generic VGPR OOR behavior
  and `RDNA3_5-RJ-024` for memory-return destination nullification. The LDS
  findings here are limited to allocation, placement, alignment, and
  LDS-specific violation reporting.
- Chapter 3.3.4 LDS backing OOB narrow match: `Lds` returns zero for loads and
  drops writes when an address crosses the total simulated LDS backing at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/lds.h:31` through `:107` and `:136`
  through `:168`. `RDNA3_5-RJ-027` and `RDNA3_5-RJ-028` are about the missing
  per-workgroup allocation boundary, documented allocation granularity,
  side-placement constraints, alignment modes, and violation reporting.
- Chapter 3.4.1-3.4.2 HWREG/summary boundary: `RDNA3_5-RJ-012` records the
  incorrect and incomplete RDNA3.5 HWREG get/set map, and `RDNA3_5-RJ-013`
  records stale raw `STATUS.EXECZ`/`STATUS.VCCZ`. `RDNA3_5-RJ-029` and
  `RDNA3_5-RJ-030` cover the broader STATUS field inventory/initialization and
  MODE behavioral effects found in this slice.
- Chapter 3.4.3 M0 storage/operand narrow match: `Wavefront` stores one raw
  per-wave `M0` value, and RDNA3.5 scalar operand resolution reads and writes
  the `M0` selector at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:240` through `:246` and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/operand.cpp:720`
  through `:725`, `:834` through `:838`, and `:900` through `:907`.
  Instruction-consumer gaps remain tracked where they belong: S_MOVREL in
  `RDNA3_5-RJ-019`, LDSDIR in `RDNA3_5-RJ-011`, GDS in `RDNA3_5-RJ-006`,
  LDS allocation/alignment in `RDNA3_5-RJ-027` and `RDNA3_5-RJ-028`, sendmsg
  side effects in `RDNA3_5-RJ-023`, and export/graphics launch in
  `RDNA3_5-RJ-010`.
- Chapter 3.4.4-3.4.5 NULL/SCC narrow match: RDNA3.5 scalar source resolution
  returns zero for `NULL`, destination writes to `NULL` return without mutating
  state, `Wavefront` stores SCC in raw STATUS bit 0, and representative shared
  SALU helpers update or consume SCC independently of the destination write at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/operand.cpp:722`
  through `:777`, `:903` through `:904`,
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:414` through `:424`, and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:1080`
  through `:1099` and `:1116` through `:1130`.
- Chapter 3.4.6 VCC/VCCZ narrow match: `Wavefront` stores raw VCC, RDNA3.5
  scalar operands expose VCC low/high and whole-VCC reads/writes, VOPC compare
  helpers build an `EXEC`-masked result and write VCC or an SGPR destination,
  and VCCZ branches synthesize the wave-size-masked live VCC value at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:232` through `:238`,
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/operand.cpp:716` through
  `:771` and `:891` through `:933`,
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:4490`
  through `:4518`, and
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/sopp.cpp:286` through
  `:313`. Raw `STATUS.VCCZ` drift remains the already-recorded
  `RDNA3_5-RJ-013` boundary, not a new VCC/VCCZ execution gap.
- Chapter 3.4.7 scratch-address narrow match: rocjitsu has per-wave scratch
  base and per-lane scratch-size state, zeroes both on reset, initializes them
  for dispatches with private scratch, and feeds that state into scratch and
  flat-private address calculation. `RDNA3_5-RJ-031` and `RDNA3_5-RJ-032` are
  about selector aliasing, alignment, and access-policy fidelity, not total
  absence of scratch-base state.
- Chapter 3.4.8 HWREG boundary: `RDNA3_5-RJ-012` already records the wrong and
  incomplete RDNA3.5 `S_GETREG`/`S_SETREG` map, including the missing
  `FLAT_SCRATCH_LO/HI`, `SH_MEM_BASES`, `PC`, and other HWREG behaviors. This
  slice keeps that existing finding as the HWREG execution gap rather than
  duplicating it.
- Chapter 3.4.9 trap/exception boundary: the runtime gaps exposed by this
  slice are already covered by existing findings. `RDNA3_5-RJ-014` covers
  missing `TRAPSTS`, TBA/TMA, `S_TRAP`, `S_RFE_B64`, and TMA/TBA
  `S_SENDMSG_RTN` behavior; `RDNA3_5-RJ-018` covers user-mode TTMP access;
  `RDNA3_5-RJ-029` covers `STATUS.TRAP_EN` and other STATUS field
  inventory/initialization behavior; and `RDNA3_5-RJ-030` covers
  `MODE.EXCP_EN` and trap-after-instruction behavior.
- Chapter 3.4.9 KFD trap-debug host-surface boundary: rocjitsu's KFD emulation
  does store trap-handler TBA/TMA ioctl values and debugger exception masks at
  `lib/rocjitsu/src/rocjitsu/kmd/linux/kfd_process.h:84` through `:85`,
  `lib/rocjitsu/src/rocjitsu/kmd/linux/kfd_process.h:185` through `:187`,
  and `lib/rocjitsu/src/rocjitsu/kmd/linux/simulated_kfd.cpp:675` through
  `:689` and `:1964` through `:1989`. The ioctl layer explicitly notes that
  TBA/TMA have no consumer yet, so this remains part of `RDNA3_5-RJ-014`
  rather than a separate ISA executor finding.
- Chapter 3.4.10 REALTIME decode narrow match: rocjitsu recognizes the
  `S_SENDMSG_RTN_B32/B64` instruction forms and special-cases message `0x83`,
  matching XML's `msg_rtn_get_realtime` selector. `RDNA3_5-RJ-033` is about
  the clock domain, synchronous writeback, missing LGKM dependency, and missing
  `SHADER_CYCLES` HWREG behavior.
- Chapter 3.5.1 compute `EXEC` initialization narrow match: the command
  processor computes an initial active-lane mask from the dispatch grid,
  workgroup dimensions, wave index, and wave size in
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/dispatch_entry.h:206` through `:238`,
  then writes it with `wf->set_exec(...)` before register initialization at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:710` through
  `:720`. Existing tests cover a wave64 grid tail and a 3D tail with workgroup
  offset at `tests/amdgpu_vm_test.cpp:581` through `:605`. This covers the
  ordinary compute-dispatch active-lane contract; graphics launch and wave64
  configurability remain broader launch gaps in `RDNA3_5-RJ-010` and
  `RDNA3_5-RJ-008`.
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
  `:362`. The remaining scratch fidelity issues are the RDNA3.5
  scalar-selector aliasing in `RDNA3_5-RJ-031` and the alignment/access-policy
  gap in `RDNA3_5-RJ-032`.
- Chapter 3.5.3 AMDHSA compute user-SGPR narrow match: rocjitsu follows the
  AMDHSA descriptor order for private-segment buffer, dispatch pointer, queue
  pointer, kernarg pointer, dispatch ID, flat-scratch init, private segment
  size, and kernarg preload values at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:181` through
  `:256`. The `TG_SIZE`, multidimensional X-ID, and RDNA3.5 TTMP payload gaps
  are recorded separately in `RDNA3_5-RJ-034` through `RDNA3_5-RJ-036`.
- Chapter 3.5.3 graphics-stage boundary: the PS, GS, and HS SGPR preload
  tables require graphics-stage launch state and SPI shader-register plumbing
  that rocjitsu does not model. That remains under the existing broader
  graphics launch gap in `RDNA3_5-RJ-010`, rather than a duplicate finding in
  this compute launch slice.
- Chapter 3.3.1 scalar-selector storage narrow match: RDNA3.5 operand metadata
  defines the SGPR, VCC, TTMP, NULL, M0, and EXEC selector ranges, and
  `Wavefront` stores live `EXEC`, `VCC`, `M0`, and raw status separately at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/operand_types.h:196`
  through `:254` and
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wavefront.h:211` through `:246`.
  `RDNA3_5-RJ-017` through `RDNA3_5-RJ-020` cover the missing dynamic edge
  behavior rather than absence of the selector namespace.
- Chapter 3.2.1 branch offset narrow match: ordinary branch and call execution
  use `wf.pc + 4 + offset * 4 - size_`, matching the manual's
  next-instruction-relative branch formula at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/sopp.cpp:222` through
  `:224` and `rdna3_5/sopk.cpp:411` through `:414`.
- Chapter 2 compute dispatch narrow match: `DispatchEntry` and the command
  processor create waves over a 1D/2D/3D compute grid and initialize per-workitem
  IDs at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:174` through
  `:325` and `:1045` through `:1047`. `RDNA3_5-RJ-008` and
  `RDNA3_5-RJ-009` track the missing wave64 and hardware-limit contracts.
- Chapter 2 CU/WGP LDS placement boundary: `DispatchEntry` carries `wgp_mode`
  and SPI placement distinguishes CU/WGP LDS allocation paths. `RDNA3_5-RJ-007`
  covers the missing 64KiB per-workgroup LDS cap, while `RDNA3_5-RJ-011`
  covers LDSDIR no-op behavior and the missing WGP-mode legality check.
- Chapter 2 shader-padding boundary: generated RDNA3.5 decode and execution
  contain `S_SET_INST_PREFETCH_DISTANCE`, `S_CODE_END`, and `S_ENDPGM` shells
  at `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/sopp.cpp:64` through
  `:74`, `:198` through `:205`, and `:415` through `:422`. The 64-DWORD
  padding requirement is a shader-buffer/code-object contract outside normal
  instruction execution.
- Chapter 1 architecture-model boundary: rocjitsu has RDNA3.5 ISA traits at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/isa.h:63` through `:83`,
  a structural XCD/command-processor/shader-engine/CU model at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/xcd.cpp:16` through `:49`, and
  command-processor dispatch setup that derives workgroup and LDS state at
  `lib/rocjitsu/src/rocjitsu/vm/amdgpu/command_processor.cpp:916` through
  `:999`. The remaining differences are detailed operational gaps such as
  `RDNA3_5-RJ-006` and `RDNA3_5-RJ-007`, or are deferred to later wave, cache,
  memory, and exception chapters.
- `V_PK_FMAC_F16` is generated as VOP2, not VOP3P, matching the XML `ENC_VOP2`
  entry at `amdgpu_isa_rdna3_5.xml:86076`; it should not be treated as a
  missing RDNA3.5 VOP3P instruction.
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
- `DS_BVH_STACK_RTN_B32` is present in the RDNA3.5 opcode constants and decode
  table at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/opcodes.h:1197`,
  `opcodes.h:2200`, and `decoder.cpp:4789` through `:4790` /
  `decoder.cpp:8692`; this is an execution/metadata gap, not a missing decode.
- The generated RDNA3.5 constructor models the explicit operands and in-out
  `ADDR` path at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/ds.cpp:4097` through
  `:4111`, matching the XML operand direction and widths.
- The generated encoding fixture includes `ds_bvh_stack_rtn_b32` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/rdna3_5/test_encodings.h:1196`.
