// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef CONSAN_CASE
#define CONSAN_CASE 1
#endif
#ifndef CONSAN_KERNEL
#define CONSAN_KERNEL lds_producer_consumer
#endif

#ifndef CONSAN_TARGET
#define CONSAN_TARGET "amdgcn-amd-amdhsa--gfx1100"
#endif
#ifndef CONSAN_WAVE_SIZE
#define CONSAN_WAVE_SIZE 32
#endif
#ifndef CONSAN_LEGACY
#define CONSAN_LEGACY 0
#endif
#ifndef CONSAN_GFX12
#define CONSAN_GFX12 0
#endif
#ifndef CONSAN_MAX_WORKGROUP
#define CONSAN_MAX_WORKGROUP 1024
#endif
#if CONSAN_LEGACY
#define LOAD32 s_load_dword
#define LOAD64 s_load_dwordx2
#define DS_STORE ds_write_b32
#define DS_LOAD ds_read_b32
#define GLOBAL_STORE global_store_dword
#define VECTOR_ADD v_add_u32
#else
#define LOAD32 s_load_b32
#define LOAD64 s_load_b64
#define DS_STORE ds_store_b32
#define DS_LOAD ds_load_b32
#define GLOBAL_STORE global_store_b32
#define VECTOR_ADD v_add_nc_u32
#endif
#if CONSAN_WAVE_SIZE == 64
#define MASK_EXEC exec
#define MASK_VCC vcc
#define MASK_SAVE s[8:9]
#define MASK_MOVE s_mov_b64
#define MASK_AND_SAVE s_and_saveexec_b64
#define WAVE_SHIFT 6
#else
#define MASK_EXEC exec_lo
#define MASK_VCC vcc_lo
#define MASK_SAVE s8
#define MASK_MOVE s_mov_b32
#define MASK_AND_SAVE s_and_saveexec_b32
#define WAVE_SHIFT 5
#endif
#ifdef CONSAN_GROUP_BYTES
#define GROUP_BYTES CONSAN_GROUP_BYTES
#elif CONSAN_CASE >= 10
#define GROUP_BYTES 128
#else
#define GROUP_BYTES 8
#endif

.amdgcn_target CONSAN_TARGET
.macro consan_wait_all
#if CONSAN_GFX12
  s_wait_kmcnt 0
  s_wait_dscnt 0
  s_wait_loadcnt 0
#else
  s_waitcnt lgkmcnt(0)
#endif
.endm
.macro consan_wait_stores
#if CONSAN_GFX12
  s_wait_storecnt 0
#elif CONSAN_LEGACY
  s_waitcnt vmcnt(0)
#else
  s_waitcnt_vscnt null, 0
#endif
.endm
.text
.globl CONSAN_KERNEL
.type CONSAN_KERNEL,@function
.p2align 8
CONSAN_KERNEL:
  LOAD32 s4, s[0:1], 0
  LOAD64 s[6:7], s[0:1], 8
  consan_wait_all
  v_mov_b32 v1, 0
  v_mov_b32 v2, s4
  v_mov_b32 v5, 0

#if CONSAN_CASE == 10 || CONSAN_CASE == 11 || CONSAN_CASE == 12
  // Distinct addresses exercise grouping, bounded reservation and empty EXEC.
  v_lshlrev_b32 v1, 2, v0
#if CONSAN_CASE == 10
  v_cmp_gt_u32 MASK_VCC, 4, v0
#elif CONSAN_CASE == 11
  v_cmp_gt_u32 MASK_VCC, 5, v0
#else
  v_cmp_eq_u32 MASK_VCC, 1024, v0
#endif
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
  DS_STORE v1, v2
#elif CONSAN_CASE == 13
  v_cmp_eq_u32 MASK_VCC, 0, v0
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
  s_cmp_eq_u32 s4, 7
  s_cbranch_scc1 .Lskip_store
  DS_STORE v1, v2 offset:4
.Lskip_store:
  DS_STORE v1, v2
#elif CONSAN_CASE == 2 || CONSAN_CASE == 6
  // Each wave writes and reads its own word. The independent numeric oracle
  // is output[0] = input, output[1] = input + 1, with no cross-wave dependency.
  v_lshrrev_b32 v4, WAVE_SHIFT, v0
  v_lshlrev_b32 v1, 2, v4
  v_mov_b32 v5, v1
  VECTOR_ADD v2, s4, v4
  v_and_b32 v4, (CONSAN_WAVE_SIZE-1), v0
  v_cmp_eq_u32 MASK_VCC, 0, v4
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
  DS_STORE v1, v2
  consan_wait_all
#if CONSAN_CASE == 6
  DS_LOAD v1, v1
  consan_wait_all
  GLOBAL_STORE v5, v1, s[6:7]
#else
  DS_LOAD v3, v1
  consan_wait_all
  GLOBAL_STORE v5, v3, s[6:7]
#endif
#elif CONSAN_CASE == 4 || CONSAN_CASE == 5
#if CONSAN_CASE == 4
  v_cmp_gt_u32 MASK_VCC, 2, v0
#else
  v_cmp_eq_u32 MASK_VCC, 0, v0
#endif
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
  DS_STORE v1, v2
  GLOBAL_STORE v5, v2, s[6:7]
#else
#if CONSAN_CASE == 14
  v_cmp_eq_u32 MASK_VCC, (CONSAN_WAVE_SIZE-1), v0
#else
  v_cmp_eq_u32 MASK_VCC, 0, v0
#endif
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
#if CONSAN_CASE == 3
  DS_LOAD v2, v1
#elif CONSAN_CASE == 8
  DS_STORE v1, v2 offset:4
#elif CONSAN_CASE == 9
  ds_store_b64 v1, v[2:3]
#else
  DS_STORE v1, v2
#endif
  MASK_MOVE MASK_EXEC, MASK_SAVE
#if CONSAN_CASE == 7
  s_barrier
#endif
#if CONSAN_CASE == 14
  v_cmp_eq_u32 MASK_VCC, (2*CONSAN_WAVE_SIZE-1), v0
#else
  v_cmp_eq_u32 MASK_VCC, CONSAN_WAVE_SIZE, v0
#endif
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
  DS_LOAD v3, v1
  consan_wait_all
  GLOBAL_STORE v5, v3, s[6:7]
#endif
  MASK_MOVE MASK_EXEC, MASK_SAVE
  consan_wait_all
  consan_wait_stores
  s_endpgm
.Lkernel_end:
.size CONSAN_KERNEL, .Lkernel_end-CONSAN_KERNEL

.rodata
.p2align 6
.amdhsa_kernel CONSAN_KERNEL
  .amdhsa_group_segment_fixed_size GROUP_BYTES
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  // gfx1100 wave32 requires at least 16 user+system SGPR inputs. The
  // kernarg pointer stays at s[0:1]; padding moves workgroup X to s15.
  .amdhsa_user_sgpr_count 15
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 16
#if CONSAN_LEGACY
  .amdhsa_accum_offset 8
#endif
#if !CONSAN_LEGACY
#if CONSAN_WAVE_SIZE == 32
  .amdhsa_wavefront_size32 1
#else
  .amdhsa_wavefront_size32 0
#endif
#endif
  .amdhsa_reserve_vcc 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.target: CONSAN_TARGET
amdhsa.kernels:
  - .name: CONSAN_KERNEL
    .symbol: CONSAN_KERNEL.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: GROUP_BYTES
    .private_segment_fixed_size: 0
    .sgpr_count: 16
    .vgpr_count: 8
    .wavefront_size: CONSAN_WAVE_SIZE
    .max_flat_workgroup_size: CONSAN_MAX_WORKGROUP
    .args:
      - .name: input
        .size: 4
        .offset: 0
        .value_kind: by_value
      - .name: output
        .size: 8
        .offset: 8
        .value_kind: global_buffer
        .address_space: global
...
.end_amdgpu_metadata
