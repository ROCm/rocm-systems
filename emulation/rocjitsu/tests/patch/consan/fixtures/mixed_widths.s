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
  s_load_b32 s4, s[0:1], 0
  s_load_b64 s[6:7], s[0:1], 8
  s_wait_kmcnt 0
  v_mov_b32 v1, 0
  v_mov_b32 v2, s4
  v_mov_b32 v3, 1
  v_add_nc_u32 v3, s4, v3
  v_mov_b32 v5, 0
#if CONSAN_MIXED == 2
  v_add_nc_u32 v2, 0x8000, v2
  v_lshlrev_b32 v2, 16, v2
#endif
  v_cmp_eq_u32 vcc_lo, 0, v0
  s_and_saveexec_b32 s8, vcc_lo
#if CONSAN_MIXED == 2
  ds_store_b32 v1, v2
#elif CONSAN_MIXED == 4
  ds_load_b64 v[2:3], v1
#else
  ds_store_b64 v1, v[2:3]
#endif
  s_mov_b32 exec_lo, s8
#if CONSAN_MIXED == 5
  s_wait_dscnt 0
  s_barrier_signal -1
  s_barrier_wait -1
#endif
  v_cmp_eq_u32 vcc_lo, 32, v0
  s_and_saveexec_b32 s8, vcc_lo
#if CONSAN_MIXED == 2
  ds_load_u16 v3, v1 offset:2
#elif CONSAN_MIXED == 3
  ds_load_b32 v3, v1 offset:8
#else
  ds_load_b32 v3, v1 offset:4
#endif
  s_wait_dscnt 0
  global_store_b32 v5, v3, s[6:7]
  s_mov_b32 exec_lo, s8
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
  // Keep the kernarg pointer at s[0:1] and pad user inputs so the original
  // workgroup-X input occupies s15.
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
      - .name: output
        .size: 8
        .offset: 8
        .value_kind: global_buffer
        .address_space: global
...
.end_amdgpu_metadata
