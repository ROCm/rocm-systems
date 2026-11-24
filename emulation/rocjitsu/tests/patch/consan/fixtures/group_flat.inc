// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

  LOAD32 s4, s[0:1], 0
  LOAD64 s[6:7], s[0:1], 8
  consan_wait_all
  v_lshrrev_b32 v6, WAVE_SHIFT, v0
  v_and_b32 v12, (CONSAN_WAVE_SIZE-1), v0
  VECTOR_ADD v8, s4, v6
  v_mov_b32 v5, s4
  VECTOR_ADD v5, 1, v5
  v_mov_b32 v4, 1020
  s_mov_b64 s[4:5], src_shared_base

  v_cmp_eq_u32 MASK_VCC, (CONSAN_WAVE_SIZE-1), v12
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
  v_cmp_eq_u32 MASK_VCC, 0, v6
  MASK_AND_SAVE MASK_PEER_SAVE, MASK_VCC
  DS_STORE_B32 v4, v5
  MASK_MOVE MASK_EXEC, MASK_PEER_SAVE
#ifdef FLAT_ORDERED
  s_waitcnt 0
  s_barrier
#endif

#ifdef FLAT_EMPTY
  v_cmp_eq_u32 MASK_VCC, 2, v6
#else
  v_cmp_eq_u32 MASK_VCC, 1, v6
#endif
  MASK_AND_SAVE MASK_PEER_SAVE, MASK_VCC
  v_mov_b32_e32 v3, s5
#if defined(FLAT_ADJACENT)
  v_mov_b32_e32 v2, 1016
#elif defined(FLAT_OOB)
  v_mov_b32_e32 v2, 1024
#else
  v_mov_b32_e32 v2, 1020
#endif

  // Alias the result with the low address half to exercise the BeforeInst probe.
  flat_load_dword v2, v[2:3]
  s_waitcnt 0
#ifdef FLAT_ORDERED
  v_mov_b32 v8, v2
#endif
  // The racing read has an unconstrained value; discard it before comparing registers.
  v_mov_b32 v2, 0
  MASK_MOVE MASK_EXEC, MASK_PEER_SAVE
  v_lshlrev_b32 v1, 2, v6
  GLOBAL_STORE v1, v8, s[6:7]
  MASK_MOVE MASK_EXEC, MASK_SAVE
  consan_wait_stores
  s_endpgm
