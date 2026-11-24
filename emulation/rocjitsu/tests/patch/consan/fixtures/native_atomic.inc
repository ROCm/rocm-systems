// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

  LOAD32 s4, s[0:1], 0
  LOAD64 s[6:7], s[0:1], 8
  consan_wait_all
  v_lshrrev_b32 v6, WAVE_SHIFT, v0
  v_and_b32 v12, (CONSAN_WAVE_SIZE-1), v0
  VECTOR_ADD v2, s4, v6
  v_mov_b32 v7, 1
#ifdef ATOMIC_COUNT
  // Seed and read one word per wave. Two high lanes each add one to it.
  v_lshlrev_b32 v3, 3, v6
  VECTOR_ADD v3, 1008, v3
  v_cmp_eq_u32 MASK_VCC, (CONSAN_WAVE_SIZE-1), v12
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
  DS_STORE_B32 v3, v2 offset:4
  consan_wait_all
  MASK_MOVE MASK_EXEC, MASK_SAVE

  v_lshrrev_b32 v11, 1, v12
  v_cmp_eq_u32 MASK_VCC, ((CONSAN_WAVE_SIZE/2)-1), v11
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
  ds_add_u32 v3, v7 offset:4
  consan_wait_all
  MASK_MOVE MASK_EXEC, MASK_SAVE

  v_cmp_eq_u32 MASK_VCC, (CONSAN_WAVE_SIZE-1), v12
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
  DS_LOAD_B32 v8, v3 offset:4
  consan_wait_all
#else
  v_mov_b32 v3, 1012
  v_cmp_eq_u32 MASK_VCC, (CONSAN_WAVE_SIZE-1), v12
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
  v_cmp_eq_u32 MASK_VCC, 0, v6
  MASK_AND_SAVE MASK_PEER_SAVE, MASK_VCC
  ds_add_u32 v3, v7 offset:4
  MASK_MOVE MASK_EXEC, MASK_PEER_SAVE
  v_cmp_eq_u32 MASK_VCC, 1, v6
  MASK_AND_SAVE MASK_PEER_SAVE, MASK_VCC
#if defined(ATOMIC_READ)
  DS_LOAD_B32 v8, v3 offset:4
#elif defined(ATOMIC_WRITE)
  DS_STORE_B32 v3, v2 offset:4
#elif defined(ATOMIC_ATOMIC)
  ds_add_u32 v3, v7 offset:4
#elif defined(ATOMIC_ADJACENT)
  DS_STORE_B32 v3, v2 offset:8
#else
#error Unknown atomic peer
#endif
  consan_wait_all
  MASK_MOVE MASK_EXEC, MASK_PEER_SAVE
  // The racing access has no deterministic value oracle.
  v_mov_b32 v8, v2
#endif
  v_lshlrev_b32 v1, 2, v6
  GLOBAL_STORE v1, v8, s[6:7]
  MASK_MOVE MASK_EXEC, MASK_SAVE
  consan_wait_stores
  s_endpgm
