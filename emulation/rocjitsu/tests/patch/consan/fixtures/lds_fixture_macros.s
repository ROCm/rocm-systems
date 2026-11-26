// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Assembly definitions for the native LDS and Group-FLAT fixtures.
#ifndef CONSAN_TARGET
#define CONSAN_TARGET "amdgcn-amd-amdhsa--gfx1250"
#endif
#define CONSAN_METADATA_TARGET CONSAN_TARGET
#define CONSAN_KERNEL lds_producer_consumer
#ifndef CONSAN_WAVE_SIZE
#define CONSAN_WAVE_SIZE 32
#endif
#ifndef CONSAN_LEGACY
#define CONSAN_LEGACY 0
#endif
#ifndef CONSAN_GFX12
#define CONSAN_GFX12 1
#endif
#if CONSAN_LEGACY
#define LOAD32 s_load_dword
#define LOAD64 s_load_dwordx2
#define DS_STORE_B8 ds_write_b8
#define DS_STORE_B16 ds_write_b16
#define DS_STORE_B32 ds_write_b32
#define DS_STORE_B64 ds_write_b64
#define DS_STORE_B96 ds_write_b96
#define DS_STORE_B128 ds_write_b128
#define DS_LOAD_U8 ds_read_u8
#define DS_LOAD_I8 ds_read_i8
#define DS_LOAD_B32 ds_read_b32
#define DS_LOAD_U16 ds_read_u16
#define DS_LOAD_I16 ds_read_i16
#define DS_READ_D16_OP(type) ds_read_##type
#define DS_WRITE_D16_OP(type) ds_write_##type
#define DS_LOAD_B96 ds_read_b96
#define DS_LOAD_B64 ds_read_b64
#define DS_LOAD_B128 ds_read_b128
#define GLOBAL_STORE global_store_dword
#define VECTOR_ADD v_add_u32
#else
#define LOAD32 s_load_b32
#define LOAD64 s_load_b64
#define DS_STORE_B8 ds_store_b8
#define DS_STORE_B16 ds_store_b16
#define DS_STORE_B32 ds_store_b32
#define DS_STORE_B64 ds_store_b64
#define DS_STORE_B96 ds_store_b96
#define DS_STORE_B128 ds_store_b128
#define DS_LOAD_U8 ds_load_u8
#define DS_LOAD_I8 ds_load_i8
#define DS_LOAD_B32 ds_load_b32
#define DS_LOAD_U16 ds_load_u16
#define DS_LOAD_I16 ds_load_i16
#define DS_READ_D16_OP(type) ds_load_##type
#define DS_WRITE_D16_OP(type) ds_store_##type
#define DS_LOAD_B96 ds_load_b96
#define DS_LOAD_B64 ds_load_b64
#define DS_LOAD_B128 ds_load_b128
#define GLOBAL_STORE global_store_b32
#define VECTOR_ADD v_add_nc_u32
#endif
#if CONSAN_WAVE_SIZE == 64
#define MASK_EXEC exec
#define MASK_VCC vcc
#define MASK_SAVE s[8:9]
#define MASK_PEER_SAVE s[10:11]
#define MASK_MOVE s_mov_b64
#define MASK_AND_SAVE s_and_saveexec_b64
#define WAVE_SHIFT 6
#else
#define MASK_EXEC exec_lo
#define MASK_VCC vcc_lo
#define MASK_SAVE s8
#define MASK_PEER_SAVE s10
#define MASK_MOVE s_mov_b32
#define MASK_AND_SAVE s_and_saveexec_b32
#define WAVE_SHIFT 5
#endif
#if defined(CONSAN_DUAL_WRITE) || defined(CONSAN_TRANSPOSE_BITS)
#define GROUP_BYTES 4096
#else
#define GROUP_BYTES 1024
#endif
#define CONSAN_MAX_WORKGROUP 1024
#ifndef CONSAN_WIDTH
#define CONSAN_WIDTH 16
#endif
#ifndef CONSAN_SIGNED
#define CONSAN_SIGNED 0
#endif
#ifndef CONSAN_HIGH
#define CONSAN_HIGH 0
#endif
#ifndef CONSAN_ALIAS
#define CONSAN_ALIAS 0
#endif
#ifndef CONSAN_B96_PEER_OFFSET
#define CONSAN_B96_PEER_OFFSET 0
#endif
#ifndef CONSAN_COLLISION
#define CONSAN_COLLISION 0
#endif

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

.macro consan_kernel_descriptor
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
  .amdhsa_next_free_vgpr 16
  .amdhsa_next_free_sgpr 16
#if CONSAN_LEGACY
  .amdhsa_accum_offset 16
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
    .vgpr_count: 16
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
.endm
