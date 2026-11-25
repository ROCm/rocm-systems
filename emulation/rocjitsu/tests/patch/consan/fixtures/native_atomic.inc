// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#if defined(ATOMIC_U64) || defined(ATOMIC_F64)
#ifdef ATOMIC_F64
#define ATOMIC_INSTRUCTION ds_add_f64 v3, v[8:9] offset:8
#else
#define ATOMIC_INSTRUCTION ds_add_u64 v3, v[8:9] offset:8
#endif
#define SIZE 8
#define SEED DS_STORE_B64 v3, v[4:5] offset:8
#define READ_RESULT DS_LOAD_B64 v[8:9], v3 offset:8
#elif defined(ATOMIC_CAS)
// DATA0 is the comparison on gfx942/gfx950 and the replacement on later forms.
#if CONSAN_LEGACY
#define ATOMIC_INSTRUCTION ds_cmpst_rtn_b32 v3, v3, v2, v7 offset:4
#else
#define ATOMIC_INSTRUCTION ds_cmpstore_rtn_b32 v3, v3, v7, v2 offset:4
#endif
#define SIZE 4
#define SEED DS_STORE_B32 v3, v4 offset:4
#define READ_RESULT DS_LOAD_B32 v8, v13 offset:4
#elif defined(ATOMIC_F32)
#define ATOMIC_INSTRUCTION ds_add_f32 v3, v7 offset:4
#define SIZE 4
#define SEED DS_STORE_B32 v3, v4 offset:4
#define READ_RESULT DS_LOAD_B32 v8, v3 offset:4
#else
#define ATOMIC_INSTRUCTION ds_add_u32 v3, v7 offset:4
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
#elif defined(ATOMIC_F64)
  v_mov_b32 v4, 0
  v_mov_b32 v5, 0x3ff80000  // 1.5
  v_mov_b32 v8, 0
  v_mov_b32 v9, 0x3fe00000  // 0.5
#elif defined(ATOMIC_CAS)
  v_mov_b32 v4, s4
  v_mov_b32 v7, 0x2468
#elif defined(ATOMIC_F32)
  v_mov_b32 v4, 0x3fc00000  // 1.5f
  v_mov_b32 v7, 0x3f000000  // 0.5f
#else
  v_mov_b32 v7, 1
#endif
#ifdef ATOMIC_COUNT
  // Seed and read one value per wave.
#if SIZE == 8
  v_lshlrev_b32 v3, 4, v6
#else
  v_lshlrev_b32 v3, 3, v6
#endif
  VECTOR_ADD v3, (1024-4*SIZE), v3
#ifdef ATOMIC_CAS
  // v3 is the guest result as well as the address. Use its saved value for the readback.
  v_mov_b32 v13, v3
#endif
  v_cmp_eq_u32 MASK_VCC, (CONSAN_WAVE_SIZE-1), v12
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
  SEED
  consan_wait_all
  MASK_MOVE MASK_EXEC, MASK_SAVE

#ifdef ATOMIC_CAS
  // Wave zero matches the seed; wave one compares input+1 and keeps the seed.
  v_cmp_eq_u32 MASK_VCC, (CONSAN_WAVE_SIZE-1), v12
#else
  v_lshrrev_b32 v11, 1, v12
  v_cmp_eq_u32 MASK_VCC, ((CONSAN_WAVE_SIZE/2)-1), v11
#endif
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
  ATOMIC_INSTRUCTION
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
  ATOMIC_INSTRUCTION
#ifdef ATOMIC_CAS
  consan_wait_all
#endif
  MASK_MOVE MASK_EXEC, MASK_PEER_SAVE
  v_cmp_eq_u32 MASK_VCC, 1, v6
  MASK_AND_SAVE MASK_PEER_SAVE, MASK_VCC
#if defined(ATOMIC_READ)
  DS_LOAD_B32 v8, v3 offset:(2*SIZE-4)
#elif defined(ATOMIC_WRITE)
  DS_STORE_B32 v3, v2 offset:(2*SIZE-4)
#elif defined(ATOMIC_ATOMIC)
  ATOMIC_INSTRUCTION
#elif defined(ATOMIC_ADJACENT)
  DS_STORE_B32 v3, v2 offset:(2*SIZE)
#else
#error Unknown atomic peer
#endif
  consan_wait_all
  MASK_MOVE MASK_EXEC, MASK_PEER_SAVE
  // The racing access has no deterministic value oracle.
  v_mov_b32 v8, v2
#ifdef ATOMIC_CAS
  v_mov_b32 v3, (1024-3*SIZE)
#endif
#endif
#if defined(ATOMIC_COUNT) && defined(ATOMIC_CAS)
  v_lshlrev_b32 v1, 4, v6
  GLOBAL_STORE v1, v3, s[6:7]
  VECTOR_ADD v1, 4, v1
  GLOBAL_STORE v1, v8, s[6:7]
  MASK_MOVE MASK_EXEC, MASK_SAVE
  v_cmp_eq_u32 MASK_VCC, (CONSAN_WAVE_SIZE-2), v12
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
  v_lshlrev_b32 v1, 4, v6
  VECTOR_ADD v1, 8, v1
  GLOBAL_STORE v1, v3, s[6:7]
#elif defined(ATOMIC_COUNT) && SIZE == 8
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
