// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#if CONSAN_TRANSPOSE_BITS == 4
#if CONSAN_LEGACY
#define TRANSPOSE ds_read_b64_tr_b4
#else
#define TRANSPOSE ds_load_tr4_b64
#endif
#define TR_LAST 9
#elif CONSAN_TRANSPOSE_BITS == 6
#if CONSAN_LEGACY
#define TRANSPOSE ds_read_b96_tr_b6
#else
#define TRANSPOSE ds_load_tr6_b96
#endif
#define TR_LAST 10
#elif CONSAN_TRANSPOSE_BITS == 8
#if CONSAN_LEGACY
#define TRANSPOSE ds_read_b64_tr_b8
#else
#define TRANSPOSE ds_load_tr8_b64
#endif
#define TR_LAST 9
#elif CONSAN_LEGACY
#define TRANSPOSE ds_read_b64_tr_b16
#define TR_LAST 9
#else
#define TRANSPOSE ds_load_tr16_b128
#define TR_LAST 11
#endif
#define TR_BYTES ((TR_LAST-7)*4)

  LOAD64 s[6:7], s[0:1], 8
  consan_wait_all
  v_and_b32 v12, (CONSAN_WAVE_SIZE-1), v0
  v_lshlrev_b32 v7, 4, v0
#if defined(TR_PEER)
  v_lshrrev_b32 v6, WAVE_SHIFT, v0
  v_cmp_eq_u32 MASK_VCC, 0, v6
  s_cbranch_vccz .Lpeer
  // Four noncontiguous address groups fit the per-site report budget.
  v_and_b32 v8, 3, v12
  v_lshlrev_b32 v8, 5, v8
#else
#ifdef TR_SHORT
  // Lanes outside the four-workitem dispatch still issue physical requests.
  MASK_MOVE MASK_SAVE, MASK_EXEC
  MASK_MOVE MASK_EXEC, -1
  v_mov_b32 v8, 0
  MASK_MOVE MASK_EXEC, MASK_SAVE
#endif
  v_lshlrev_b32 v8, 4, v12
  v_lshlrev_b32 v1, 8, v12
  VECTOR_ADD v1, v12, v1
  v_lshlrev_b32 v2, 16, v1
  VECTOR_ADD v1, v2, v1
  VECTOR_ADD v2, 0x10305070, v1
  VECTOR_ADD v3, 0x21416181, v1
  VECTOR_ADD v4, 0x32527292, v1
  VECTOR_ADD v5, 0x436383a3, v1
  DS_STORE_B128 v8, v[2:5] offset:2048
  consan_wait_all
#endif
  v_mov_b32 v9, 0x12345678
  v_mov_b32 v10, 0x9abcdef0
  v_mov_b32 v11, 0xfedcba98
  MASK_MOVE MASK_SAVE, MASK_EXEC
#if defined(TR_EMPTY)
  MASK_MOVE MASK_EXEC, 0
#elif defined(TR_PARTIAL)
  v_cmp_eq_u32 MASK_VCC, (CONSAN_WAVE_SIZE-1), v12
  MASK_MOVE MASK_EXEC, MASK_VCC
#endif
  TRANSPOSE v[8:TR_LAST], v8 offset:2048
  consan_wait_all
  MASK_MOVE MASK_EXEC, MASK_SAVE
#if defined(TR_PEER)
  s_branch .Lpeer_output
.Lpeer:
  v_mov_b32 v1, 2112
  v_mov_b32 v2, 7
  v_cmp_eq_u32 MASK_VCC, 0, v12
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
#if defined(TR_ADJACENT)
  DS_STORE_B32 v1, v2 offset:TR_BYTES
#else
  DS_STORE_B32 v1, v2 offset:(TR_BYTES-4)
#endif
  consan_wait_all
  MASK_MOVE MASK_EXEC, MASK_SAVE
.Lpeer_output:
  // Concurrent peer reads have no independent numeric oracle.
  v_mov_b32 v8, v12
  v_mov_b32 v9, 0x12345678
  v_mov_b32 v10, 0x9abcdef0
  v_mov_b32 v11, 0xfedcba98
#endif
  GLOBAL_STORE v7, v8, s[6:7]
  VECTOR_ADD v7, 4, v7
  GLOBAL_STORE v7, v9, s[6:7]
  VECTOR_ADD v7, 4, v7
  GLOBAL_STORE v7, v10, s[6:7]
  VECTOR_ADD v7, 4, v7
  GLOBAL_STORE v7, v11, s[6:7]
  consan_wait_stores
  s_endpgm
