// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "lds_fixture_macros.s"

#if CONSAN_WIDTH == 64
#define STORE DS_STORE_B64
#define LOAD DS_LOAD_B64
#define DATA v[2:3]
#define RESULT v[8:9]
#define HIGH_DATA v3
#define HIGH_RESULT v9
#elif CONSAN_WIDTH == 96
#define STORE DS_STORE_B96
#define LOAD DS_LOAD_B96
#define DATA v[2:4]
#define RESULT v[8:10]
#define HIGH_DATA v4
#define HIGH_RESULT v10
#elif CONSAN_WIDTH == 128
#define STORE DS_STORE_B128
#define LOAD DS_LOAD_B128
#define DATA v[2:5]
#define RESULT v[8:11]
#define HIGH_DATA v5
#define HIGH_RESULT v11
#endif

.amdgcn_target CONSAN_TARGET
.text
.globl CONSAN_KERNEL
.type CONSAN_KERNEL,@function
.p2align 8
CONSAN_KERNEL:
  LOAD32 s4, s[0:1], 0
  LOAD64 s[6:7], s[0:1], 8
  consan_wait_all
  s_mov_b32 s10, 0x89abcdef
  s_mov_b32 s11, 0x13579bdf
  s_mov_b32 s12, 0xfedcba98
  v_mov_b32 v1, 0
#if CONSAN_WAVE_SIZE == 64
  VECTOR_ADD v7, -62, v0
#define WRITER_INDEX v7
#define PEER_WORKITEM 127
#else
#define WRITER_INDEX v0
#define PEER_WORKITEM CONSAN_WAVE_SIZE
#endif
  v_cmp_gt_u32 MASK_VCC, 2, WRITER_INDEX
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
  v_mov_b32 v1, 0
  v_mov_b32 v2, s4
  v_mov_b32 v3, s10
  v_mov_b32 v4, s11
  v_mov_b32 v5, s12
#ifdef CONSAN_VARYING_VALUE
  v_and_b32 HIGH_DATA, -2, HIGH_DATA
  VECTOR_ADD HIGH_DATA, WRITER_INDEX, HIGH_DATA
#endif
  STORE v1, DATA
  consan_wait_all
  v_mov_b32 v9, 0x01234567
  v_mov_b32 v10, 0x76543210
  v_mov_b32 v11, 0xa5a5a5a5
  LOAD RESULT, v1
  consan_wait_all
#ifdef CONSAN_VARYING_VALUE
  // Either writer can win bit zero. Check the remaining LDS bits and expose
  // each writer's original low bit in its first output word.
  v_and_b32 HIGH_RESULT, -2, HIGH_RESULT
#if CONSAN_WIDTH != 128
  VECTOR_ADD HIGH_RESULT, 1, HIGH_RESULT
#endif
  v_and_b32 v13, 1, HIGH_DATA
  VECTOR_ADD v8, v13, v8
#else
  VECTOR_ADD v8, WRITER_INDEX, v8
#endif
  v_lshlrev_b32 v6, 4, WRITER_INDEX
  GLOBAL_STORE v6, v8, s[6:7]
  VECTOR_ADD v6, 4, v6
  GLOBAL_STORE v6, v9, s[6:7]
  VECTOR_ADD v6, 4, v6
  GLOBAL_STORE v6, v10, s[6:7]
  VECTOR_ADD v6, 4, v6
  GLOBAL_STORE v6, v11, s[6:7]
  MASK_MOVE MASK_EXEC, MASK_SAVE
  v_cmp_eq_u32 MASK_VCC, PEER_WORKITEM, v0
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
  DS_LOAD_B32 v14, v1 offset:(CONSAN_WIDTH/8-4)
  consan_wait_all
  v_mov_b32 v14, 0
  MASK_MOVE MASK_EXEC, MASK_SAVE
  consan_wait_stores
  s_endpgm
.Lkernel_end:
.size CONSAN_KERNEL, .Lkernel_end-CONSAN_KERNEL

consan_kernel_descriptor
