// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifdef ATOMIC_U64
#define ADD ds_add_u64 v3, v[8:9] offset:8
#define SIZE 8
#define SEED DS_STORE_B64 v3, v[4:5] offset:8
#define READ_RESULT DS_LOAD_B64 v[8:9], v3 offset:8
#elif defined(ATOMIC_F32)
#define ADD ds_add_f32 v3, v7 offset:4
#define SIZE 4
#define SEED DS_STORE_B32 v3, v4 offset:4
#define READ_RESULT DS_LOAD_B32 v8, v3 offset:4
#else
#define ADD ds_add_u32 v3, v7 offset:4
#define SIZE 4
#define SEED DS_STORE_B32 v3, v2 offset:4
#define READ_RESULT DS_LOAD_B32 v8, v3 offset:4
#endif

  LOAD32 s4, s[0:1], 0
  LOAD64 s[6:7], s[0:1], 8
  consan_wait_all
  v_lshrrev_b32 v6, WAVE_SHIFT, v0
  v_and_b32 v12, (CONSAN_WAVE_SIZE-1), v0
  VECTOR_ADD v2, s4, v6
#ifdef ATOMIC_U64
  v_mov_b32 v4, 0xfffffffe
  v_mov_b32 v5, 3
  v_mov_b32 v8, 1
  v_mov_b32 v9, 0
#elif defined(ATOMIC_F32)
  v_mov_b32 v4, 0x3fc00000  // 1.5f
  v_mov_b32 v7, 0x3f000000  // 0.5f
#else
  v_mov_b32 v7, 1
#endif
#ifdef ATOMIC_COUNT
  // Seed and read one value per wave. Two high lanes each apply one update.
#if SIZE == 8
  v_lshlrev_b32 v3, 4, v6
#else
  v_lshlrev_b32 v3, 3, v6
#endif
  VECTOR_ADD v3, (1024-4*SIZE), v3
  v_cmp_eq_u32 MASK_VCC, (CONSAN_WAVE_SIZE-1), v12
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
  SEED
  consan_wait_all
  MASK_MOVE MASK_EXEC, MASK_SAVE

  v_lshrrev_b32 v11, 1, v12
  v_cmp_eq_u32 MASK_VCC, ((CONSAN_WAVE_SIZE/2)-1), v11
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
  ADD
  consan_wait_all
  MASK_MOVE MASK_EXEC, MASK_SAVE

  v_cmp_eq_u32 MASK_VCC, (CONSAN_WAVE_SIZE-1), v12
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
  READ_RESULT
  consan_wait_all
#else
  v_mov_b32 v3, (1024-3*SIZE)
  v_cmp_eq_u32 MASK_VCC, (CONSAN_WAVE_SIZE-1), v12
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
  v_cmp_eq_u32 MASK_VCC, 0, v6
  MASK_AND_SAVE MASK_PEER_SAVE, MASK_VCC
  ADD
  MASK_MOVE MASK_EXEC, MASK_PEER_SAVE
  v_cmp_eq_u32 MASK_VCC, 1, v6
  MASK_AND_SAVE MASK_PEER_SAVE, MASK_VCC
#if defined(ATOMIC_READ)
  DS_LOAD_B32 v8, v3 offset:(2*SIZE-4)
#elif defined(ATOMIC_WRITE)
  DS_STORE_B32 v3, v2 offset:(2*SIZE-4)
#elif defined(ATOMIC_ATOMIC)
  ADD
#elif defined(ATOMIC_ADJACENT)
  DS_STORE_B32 v3, v2 offset:(2*SIZE)
#else
#error Unknown atomic peer
#endif
  consan_wait_all
  MASK_MOVE MASK_EXEC, MASK_PEER_SAVE
  // The racing access has no deterministic value oracle.
  v_mov_b32 v8, v2
#endif
#if defined(ATOMIC_COUNT) && SIZE == 8
  v_lshlrev_b32 v1, 3, v6
  GLOBAL_STORE v1, v8, s[6:7]
  VECTOR_ADD v1, 4, v1
  GLOBAL_STORE v1, v9, s[6:7]
#else
  v_lshlrev_b32 v1, 2, v6
  GLOBAL_STORE v1, v8, s[6:7]
#endif
  MASK_MOVE MASK_EXEC, MASK_SAVE
  consan_wait_stores
  s_endpgm
