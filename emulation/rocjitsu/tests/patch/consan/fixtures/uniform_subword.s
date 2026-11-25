// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "lds_fixture_macros.s"

#if CONSAN_WIDTH == 8
#define READBACK DS_LOAD_U8
#if CONSAN_STORE_HIGH
#define STORE DS_WRITE_D16_OP(b8_d16_hi)
#else
#define STORE DS_STORE_B8
#endif
#else
#define READBACK DS_LOAD_U16
#if CONSAN_STORE_HIGH
#define STORE DS_WRITE_D16_OP(b16_d16_hi)
#else
#define STORE DS_STORE_B16
#endif
#endif

.amdgcn_target CONSAN_TARGET
.text
.globl CONSAN_KERNEL
.type CONSAN_KERNEL,@function
.p2align 8
CONSAN_KERNEL:
  LOAD64 s[6:7], s[0:1], 8
  consan_wait_all
  s_mov_b32 s12, 0x3142596c
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
  v_lshlrev_b32 v6, 2, WRITER_INDEX
  VECTOR_ADD v6, 32, v6
  VECTOR_ADD v3, 0x7280, WRITER_INDEX
  DS_STORE_B32 v6, v3
  consan_wait_all
  v_mov_b32 v2, s12
#if CONSAN_STORE_HIGH == CONSAN_VARYING_VALUE
  DS_READ_D16_OP(u16_d16_hi) v2, v6
#else
  DS_READ_D16_OP(u16_d16) v2, v6
#endif
  consan_wait_all
  v_mov_b32 v1, 0
  STORE v1, v2
  consan_wait_all
  READBACK v8, v1
  consan_wait_all
#if CONSAN_VARYING_VALUE
  // Only bit zero depends on which writer wins. Publish the full source
  // register beside the stable readback so both lane values remain observable.
  v_and_b32 v8, -2, v8
#endif
  v_lshlrev_b32 v6, 3, WRITER_INDEX
  GLOBAL_STORE v6, v2, s[6:7]
  VECTOR_ADD v6, 4, v6
  GLOBAL_STORE v6, v8, s[6:7]
  MASK_MOVE MASK_EXEC, MASK_SAVE
  v_cmp_eq_u32 MASK_VCC, PEER_WORKITEM, v0
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
  DS_LOAD_B32 v14, v1
  consan_wait_all
  v_mov_b32 v14, 0
  MASK_MOVE MASK_EXEC, MASK_SAVE
  consan_wait_stores
  s_endpgm
.Lkernel_end:
.size CONSAN_KERNEL, .Lkernel_end-CONSAN_KERNEL

consan_kernel_descriptor
