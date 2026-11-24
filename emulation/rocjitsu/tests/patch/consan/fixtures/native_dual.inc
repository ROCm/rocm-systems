// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Keep the two-site alias/cross-ordinal regression independent of the scalar
// oracle accesses in the extended native fixtures.
#ifdef CONSAN_DUAL_REGRESSION
#if CONSAN_WIDTH == 32
#define FIRST_SOURCE v2
#define SECOND_SOURCE v3
#define DUAL_RESULT v[8:9]
#define COLLISION_SHIFT 3
#else
#define FIRST_SOURCE v[2:3]
#define SECOND_SOURCE v[4:5]
#define DUAL_RESULT v[8:11]
#define COLLISION_SHIFT 4
#endif

#if CONSAN_COLLISION
  // Different lane bases make opposite ordinals address the same element.
  v_lshlrev_b32 v1, COLLISION_SHIFT, v12
  CONSAN_DUAL_WRITE v1, FIRST_SOURCE, SECOND_SOURCE offset0:3 offset1:5
  // Preserve an independent output oracle without reading the colliding bytes.
  v_mov_b32 v8, v2
  v_mov_b32 v9, v3
#else
  // Separate waves' ranges so every result word has a deterministic value.
  v_lshlrev_b32 v1, 6, v6
  CONSAN_DUAL_WRITE v1, FIRST_SOURCE, SECOND_SOURCE offset0:3 offset1:5
  consan_wait_all
  // The load overwrites its own address VGPR. Probes must capture it first.
  v_mov_b32 v8, v1
  CONSAN_DUAL_READ DUAL_RESULT, v8 offset0:3 offset1:5
#endif
#else
#if CONSAN_WIDTH == 32
#define FIRST_SOURCE v2
#define SECOND_SOURCE v3
#define FIRST_RESULT v8
#define SECOND_RESULT v9
#define DUAL_RESULT v[8:9]
#define STORE_ELEMENT DS_STORE_B32
#define LOAD_ELEMENT DS_LOAD_B32
#else
#define FIRST_SOURCE v[2:3]
#define SECOND_SOURCE v[4:5]
#define FIRST_RESULT v[8:9]
#define SECOND_RESULT v[10:11]
#define DUAL_RESULT v[8:11]
#define STORE_ELEMENT DS_STORE_B64
#define LOAD_ELEMENT DS_LOAD_B64
#endif

#if CONSAN_COLLISION
  CONSAN_DUAL_WRITE v1, FIRST_SOURCE, SECOND_SOURCE offset0:3 offset1:5
  consan_wait_all
  v_cmp_eq_u32 MASK_VCC, 0, v12
  MASK_AND_SAVE MASK_PEER_SAVE, MASK_VCC
  v_mov_b32 v8, v2
  v_mov_b32 v9, v3
#else
  v_cmp_eq_u32 MASK_VCC, 0, v6
  MASK_AND_SAVE MASK_PEER_SAVE, MASK_VCC
#if CONSAN_DUAL_PEER
  CONSAN_DUAL_WRITE v1, FIRST_SOURCE, SECOND_SOURCE offset0:3 offset1:5
  consan_wait_all
  CONSAN_DUAL_READ DUAL_RESULT, v1 offset0:3 offset1:5
#else
  // Ordinary accesses use explicit byte addresses for an independent value oracle.
  STORE_ELEMENT v1, FIRST_SOURCE offset:(3*CONSAN_DUAL_SCALE)
  STORE_ELEMENT v1, SECOND_SOURCE offset:(5*CONSAN_DUAL_SCALE)
  consan_wait_all
  v_mov_b32 v8, v1
  CONSAN_DUAL_READ DUAL_RESULT, v8 offset0:3 offset1:5
#endif
  consan_wait_all
  MASK_MOVE MASK_EXEC, MASK_PEER_SAVE
  v_cmp_eq_u32 MASK_VCC, 1, v6
  MASK_AND_SAVE MASK_PEER_SAVE, MASK_VCC
#if CONSAN_DUAL_PEER
  DS_STORE_B32 v1, v2 offset:(CONSAN_DUAL_PEER*CONSAN_DUAL_SCALE)
#else
  CONSAN_DUAL_WRITE v1, FIRST_SOURCE, SECOND_SOURCE offset0:3 offset1:5
  consan_wait_all
  LOAD_ELEMENT FIRST_RESULT, v1 offset:(3*CONSAN_DUAL_SCALE)
  LOAD_ELEMENT SECOND_RESULT, v1 offset:(5*CONSAN_DUAL_SCALE)
#endif
  consan_wait_all
  MASK_MOVE MASK_EXEC, MASK_PEER_SAVE
#if CONSAN_DUAL_PEER
  // The racing load's payload is discarded by the guest.
  v_mov_b32 v8, v2
  v_mov_b32 v9, v3
#if CONSAN_WIDTH == 64
  v_mov_b32 v10, v4
  v_mov_b32 v11, v5
#endif
#endif
#endif
#endif
