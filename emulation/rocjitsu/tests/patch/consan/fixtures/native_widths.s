// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "lds_fixture_macros.s"

.amdgcn_target CONSAN_TARGET
.text
.globl CONSAN_KERNEL
.type CONSAN_KERNEL,@function
.p2align 8
CONSAN_KERNEL:
#ifdef CONSAN_ATOMIC
#include "native_atomic.inc"
#elif defined(CONSAN_TRANSPOSE_BITS)
#include "native_transpose.inc"
#else
  LOAD32 s4, s[0:1], 0
  LOAD64 s[6:7], s[0:1], 8
  consan_wait_all
  v_lshrrev_b32 v6, WAVE_SHIFT, v0
#ifdef CONSAN_DUAL_WRITE
#if CONSAN_COLLISION
  v_lshlrev_b32 v1, 3, v0
  VECTOR_ADD v1, 8, v1
#elif CONSAN_DUAL_PEER
  v_mov_b32 v1, 8
#else
  v_lshlrev_b32 v1, 5, v6
  VECTOR_ADD v1, 8, v1
#endif
#elif defined(CONSAN_B8_PEER_STRIDE)
#if CONSAN_B8_PEER_STRIDE == 0
  v_mov_b32 v1, 0
#elif CONSAN_B8_PEER_STRIDE == 1
  v_mov_b32 v1, v6
#else
  v_lshlrev_b32 v1, 1, v6
#endif
#elif defined(CONSAN_D16_PEER_OFFSET)
  v_mov_b32 v1, 0
#elif CONSAN_WIDTH == 8 || CONSAN_WIDTH == 17
  // Three waves occupy the last three dwords of the allocation.
  v_lshlrev_b32 v1, 2, v6
  VECTOR_ADD v1, (GROUP_BYTES-12), v1
#else
  v_lshlrev_b32 v1, 4, v6
#endif
#if CONSAN_WIDTH == 96
  VECTOR_ADD v1, 4, v1
#endif
  v_lshlrev_b32 v7, 4, v6
  VECTOR_ADD v2, s4, v6
#if CONSAN_WIDTH == 8
  VECTOR_ADD v2, 0x80, v2
#elif CONSAN_WIDTH == 16
  VECTOR_ADD v2, 0x8000, v2
#endif
  v_mov_b32 v3, 0x89abcdef
  v_mov_b32 v4, 0x13579bdf
  v_mov_b32 v5, 0xfedcba98
  v_mov_b32 v8, 0xdeadbeef
  v_mov_b32 v9, 0x01234567
  v_mov_b32 v10, 0x76543210
  v_mov_b32 v11, 0xa5a5a5a5
  v_and_b32 v12, (CONSAN_WAVE_SIZE-1), v0
#if CONSAN_COLLISION
  v_cmp_gt_u32 MASK_VCC, 2, v12
#else
  v_cmp_eq_u32 MASK_VCC, ((CONSAN_WAVE_SIZE-1)*CONSAN_HIGH), v12
#endif
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
#ifdef CONSAN_DUAL_WRITE
#include "native_dual.inc"
#elif defined(CONSAN_B8_PEER_STRIDE)
  DS_STORE_B8 v1, v2 offset:1020
  v_and_b32 v8, 0xff, v2
#elif CONSAN_WIDTH == 8
  DS_STORE_B32 v1, v9
  consan_wait_all
  DS_STORE_B8 v1, v2 offset:3
  consan_wait_all
#if CONSAN_SIGNED
#define DS_LOAD_BYTE DS_LOAD_I8
#else
#define DS_LOAD_BYTE DS_LOAD_U8
#endif
#if CONSAN_ALIAS
  v_mov_b32 v8, v1
  DS_LOAD_BYTE v8, v8 offset:3
#else
  DS_LOAD_BYTE v8, v1 offset:3
#endif
  DS_LOAD_B32 v9, v1
#elif CONSAN_WIDTH == 96
#if CONSAN_B96_PEER_OFFSET
  // Only wave zero writes the 12-byte range. The next wave writes one dword.
  v_cmp_eq_u32 MASK_VCC, 0, v6
  MASK_AND_SAVE MASK_PEER_SAVE, MASK_VCC
  DS_STORE_B96 v1, v[2:4] offset:12
  MASK_MOVE MASK_EXEC, MASK_PEER_SAVE
  v_cmp_eq_u32 MASK_VCC, 1, v6
  MASK_AND_SAVE MASK_PEER_SAVE, MASK_VCC
  DS_STORE_B32 v1, v2 offset:CONSAN_B96_PEER_OFFSET
  MASK_MOVE MASK_EXEC, MASK_PEER_SAVE
  v_mov_b32 v8, v2
  v_mov_b32 v9, v3
  v_mov_b32 v10, v4
#else
  DS_STORE_B96 v1, v[2:4] offset:12
  consan_wait_all
#if CONSAN_ALIAS
  v_mov_b32 v8, v1
  DS_LOAD_B96 v[8:10], v8 offset:12
#else
  DS_LOAD_B96 v[8:10], v1 offset:12
#endif
#endif
#elif CONSAN_WIDTH == 17
#define DS_READ_D16(type) DS_READ_D16_OP(type)
#define DS_WRITE_D16(type) DS_WRITE_D16_OP(type)
#include "native_d16.inc"
#elif CONSAN_WIDTH == 16
  DS_STORE_B16 v1, v2 offset:2
  consan_wait_all
#if CONSAN_SIGNED
  DS_LOAD_I16 v8, v1 offset:2
#else
  DS_LOAD_U16 v8, v1 offset:2
#endif
#elif CONSAN_WIDTH == 64
  DS_STORE_B64 v1, v[2:3] offset:8
  consan_wait_all
  DS_LOAD_B64 v[8:9], v1 offset:8
#elif CONSAN_WIDTH == 128
  DS_STORE_B128 v1, v[2:5] offset:16
  consan_wait_all
#if CONSAN_ALIAS
  v_mov_b32 v8, v1
  DS_LOAD_B128 v[8:11], v8 offset:16
#else
  DS_LOAD_B128 v[8:11], v1 offset:16
#endif
#endif
  consan_wait_all
  GLOBAL_STORE v7, v8, s[6:7]
  VECTOR_ADD v7, 4, v7
  GLOBAL_STORE v7, v9, s[6:7]
  VECTOR_ADD v7, 4, v7
  GLOBAL_STORE v7, v10, s[6:7]
  VECTOR_ADD v7, 4, v7
  GLOBAL_STORE v7, v11, s[6:7]
  MASK_MOVE MASK_EXEC, MASK_SAVE
  consan_wait_stores
  s_endpgm
#endif
.Lkernel_end:
.size CONSAN_KERNEL, .Lkernel_end-CONSAN_KERNEL

consan_kernel_descriptor
