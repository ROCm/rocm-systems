// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#define CONSAN_TARGET "amdgcn-amd-amdhsa--gfx1250"
#define CONSAN_METADATA_TARGET CONSAN_TARGET
#define CONSAN_KERNEL lds_producer_consumer
#define CONSAN_WAVE_SIZE 32
#define CONSAN_LEGACY 0
#define GROUP_BYTES 128
#define CONSAN_MAX_WORKGROUP 1024

.amdgcn_target CONSAN_TARGET
.text
.globl CONSAN_KERNEL
.type CONSAN_KERNEL,@function
.p2align 8
CONSAN_KERNEL:
  // gfx1250 supplies workgroup X in TTMP9. Keep a guest copy for the
  // final-state oracle and use it even after instrumentation callbacks.
  s_mov_b32 s15, ttmp9
  s_load_b32 s4, s[0:1], 0
  s_load_b32 s5, s[0:1], 4
  s_load_b64 s[6:7], s[0:1], 8
  s_wait_kmcnt 0
  s_mul_i32 s10, s15, s5
  v_lshrrev_b32 v4, 5, v0
  v_add_nc_u32 v5, s10, v4
  v_lshlrev_b32 v5, 2, v5
  v_mov_b32 v6, s15
  v_lshlrev_b32 v6, 4, v6
  v_add_nc_u32 v2, s4, v4
  v_add_nc_u32 v2, v6, v2
  v_mov_b32 v1, 0
  v_and_b32 v7, 31, v0
#if CONSAN_GROUP_CASE == 1 || CONSAN_GROUP_CASE == 2 || CONSAN_GROUP_CASE == 5
  v_lshlrev_b32 v1, 2, v4
  v_cmp_eq_u32 vcc_lo, 0, v7
  s_and_saveexec_b32 s8, vcc_lo
  ds_store_b32 v1, v2
#if CONSAN_GROUP_CASE == 2 || CONSAN_GROUP_CASE == 5
  s_mov_b32 exec_lo, s8
#if CONSAN_GROUP_CASE == 5
  s_mov_b32 exec_lo, 0
#endif
  s_wait_dscnt 0
  s_barrier_signal -1
  s_barrier_wait -1
  s_mov_b32 exec_lo, s8
  v_cmp_eq_u32 vcc_lo, 0, v7
  s_and_saveexec_b32 s8, vcc_lo
  v_mov_b32 v1, 0
#else
  s_wait_dscnt 0
#endif
  ds_load_b32 v3, v1
#elif CONSAN_GROUP_CASE == 6
  // Different waves in different workgroups share numeric offsets, not LDS.
  s_cmp_eq_u32 s15, 0
  s_cbranch_scc0 .Lreader
  v_cmp_eq_u32 vcc_lo, 0, v0
  s_and_saveexec_b32 s8, vcc_lo
  ds_store_b32 v1, v2
  s_mov_b32 exec_lo, s8
  s_branch .Ldone
.Lreader:
  v_cmp_eq_u32 vcc_lo, 32, v0
  s_and_saveexec_b32 s8, vcc_lo
  ds_load_b32 v3, v1
#else
  v_cmp_eq_u32 vcc_lo, 0, v0
  s_and_saveexec_b32 s8, vcc_lo
#if CONSAN_GROUP_CASE == 4
  ds_load_b32 v2, v1
#else
  ds_store_b32 v1, v2
#endif
  s_mov_b32 exec_lo, s8
  v_cmp_eq_u32 vcc_lo, 32, v0
  s_and_saveexec_b32 s8, vcc_lo
  ds_load_b32 v3, v1
#endif
  s_wait_dscnt 0
  global_store_b32 v5, v3, s[6:7]
  s_mov_b32 exec_lo, s8
.Ldone:
  s_wait_dscnt 0
  s_wait_storecnt 0
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
  // Keep the conventional kernarg pointer at s[0:1] and reserve s15 for
  // the guest copy of the TTMP9 workgroup-X input used by the state oracle.
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
amdhsa.target: CONSAN_METADATA_TARGET
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
      - .name: waves_per_workgroup
        .size: 4
        .offset: 4
        .value_kind: by_value
      - .name: output
        .size: 8
        .offset: 8
        .value_kind: global_buffer
        .address_space: global
...
.end_amdgpu_metadata
