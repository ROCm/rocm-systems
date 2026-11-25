// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#if defined(CONSAN_D16_PEER_OFFSET)
  // The peer writes either the containing dword or the adjacent dword.
  v_cmp_eq_u32 MASK_VCC, 0, v6
  MASK_AND_SAVE MASK_PEER_SAVE, MASK_VCC
  DS_WRITE_D16(b8_d16_hi) v1, v3 offset:3
  DS_WRITE_D16(b16_d16_hi) v1, v3 offset:2
  consan_wait_all
  DS_READ_D16(u8_d16) v8, v1 offset:3
  DS_READ_D16(u8_d16_hi) v9, v1 offset:3
  DS_READ_D16(i8_d16) v10, v1 offset:3
  DS_READ_D16(i8_d16_hi) v11, v1 offset:3
  consan_wait_all
  DS_READ_D16(u16_d16) v8, v1 offset:2
  DS_READ_D16(u16_d16_hi) v9, v1 offset:2
  consan_wait_all
  MASK_MOVE MASK_EXEC, MASK_PEER_SAVE
  v_cmp_eq_u32 MASK_VCC, 1, v6
  MASK_AND_SAVE MASK_PEER_SAVE, MASK_VCC
  DS_STORE_B32 v1, v2 offset:CONSAN_D16_PEER_OFFSET
  MASK_MOVE MASK_EXEC, MASK_PEER_SAVE
  // Keep the deliberately unordered read payload out of the guest output oracle.
  v_mov_b32 v8, v2
  v_mov_b32 v9, v3
  v_mov_b32 v10, v4
  v_mov_b32 v11, v5
#else
  DS_STORE_B32 v1, v11
  consan_wait_all
#if CONSAN_D16_HALF
#define D16_SOURCE b16_d16_hi
#define D16_LOW u16_d16
#define D16_HIGH u16_d16_hi
#define D16_OFFSET 2
#else
#define D16_SOURCE b8_d16_hi
#define D16_OFFSET 3
#if CONSAN_SIGNED
#define D16_LOW i8_d16
#define D16_HIGH i8_d16_hi
#else
#define D16_LOW u8_d16
#define D16_HIGH u8_d16_hi
#endif
#endif
  DS_WRITE_D16(D16_SOURCE) v1, v3 offset:D16_OFFSET
  consan_wait_all
  DS_READ_D16(D16_LOW) v8, v1 offset:D16_OFFSET
  v_mov_b32 v9, v1
  DS_READ_D16(D16_HIGH) v9, v9 offset:D16_OFFSET
  DS_LOAD_B32 v10, v1
#endif
