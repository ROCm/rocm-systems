// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "lds_fixture_macros.s"

.amdgcn_target CONSAN_TARGET
.text
.globl CONSAN_KERNEL
.type CONSAN_KERNEL,@function
.p2align 8
CONSAN_KERNEL:
#if CONSAN_WIDTH == 64
#define WIDTH 8
#define SHIFT 3
#define REGS v[4:5]
#define DATA v[8:9]
#define DS_SEED DS_STORE_B64
#define DS_READ DS_LOAD_B64
#if CONSAN_LEGACY
#define FLAT_READ flat_load_dwordx2
#define FLAT_WRITE flat_store_dwordx2
#else
#define FLAT_READ flat_load_b64
#define FLAT_WRITE flat_store_b64
#endif
#else
#define WIDTH 16
#define SHIFT 4
#define REGS v[4:7]
#define DATA v[8:11]
#define DS_SEED DS_STORE_B128
#define DS_READ DS_LOAD_B128
#if CONSAN_LEGACY
#define FLAT_READ flat_load_dwordx4
#define FLAT_WRITE flat_store_dwordx4
#else
#define FLAT_READ flat_load_b128
#define FLAT_WRITE flat_store_b128
#endif
#endif

.macro flat_wait
#if CONSAN_GFX12
  consan_wait_all
#else
  s_waitcnt 0
#endif
.endm

.macro address_pair
  v_mov_b32 v4, v15
  v_mov_b32 v5, s5
.endm

.macro save_word reg
  GLOBAL_STORE v1, \reg, s[6:7]
  VECTOR_ADD v1, 4, v1
.endm

.macro save_tuple
  save_word v4
  save_word v5
#if WIDTH == 16
  save_word v6
  save_word v7
#endif
.endm

  LOAD32 s4, s[0:1], 0
  LOAD64 s[6:7], s[0:1], 8
  flat_wait
  v_lshrrev_b32 v12, WAVE_SHIFT, v0
  v_and_b32 v13, (CONSAN_WAVE_SIZE-1), v0
  VECTOR_ADD v14, s4, v12
  s_mov_b64 s[4:5], src_shared_base
#ifdef FLAT_VALUES
  v_lshlrev_b32 v15, SHIFT, v12
  VECTOR_ADD v15, (1024-2*WIDTH), v15
  v_lshlrev_b32 v1, (SHIFT+1), v12
#else
  v_mov_b32 v15, (1024-2*WIDTH)
  v_lshlrev_b32 v1, 2, v12
#endif
  v_cmp_eq_u32 MASK_VCC, (CONSAN_WAVE_SIZE-1), v13
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
#ifdef FLAT_VALUES
  v_mov_b32 v8, v14
  v_mov_b32 v9, 0x12345678
  v_mov_b32 v10, 0x87654321
  v_mov_b32 v11, 0xfedcba98
  DS_SEED v15, DATA
  flat_wait
#else
  v_cmp_eq_u32 MASK_VCC, 0, v12
  MASK_AND_SAVE MASK_PEER_SAVE, MASK_VCC
#endif
  address_pair
  // Preserve both pointer halves even when overwritten by the result tuple.
  FLAT_READ REGS, v[4:5]
  flat_wait
#ifdef FLAT_VALUES
  save_tuple
#endif
  address_pair
  v_mov_b32 v8, 0x13579bdf
  v_mov_b32 v9, 0x2468ace0
  v_mov_b32 v10, 0x10203040
  v_mov_b32 v11, v14
  FLAT_WRITE v[4:5], DATA
  flat_wait
  consan_wait_stores
#ifdef FLAT_VALUES
  DS_READ REGS, v15
  flat_wait
  save_tuple
#else
  v_mov_b32 v6, 0
  v_mov_b32 v7, 0
  MASK_MOVE MASK_EXEC, MASK_PEER_SAVE
  v_cmp_eq_u32 MASK_VCC, 1, v12
  MASK_AND_SAVE MASK_PEER_SAVE, MASK_VCC
#ifdef FLAT_ADJACENT
  DS_STORE_B32 v15, v14 offset:WIDTH
#else
  DS_STORE_B32 v15, v14 offset:(WIDTH-4)
#endif
  MASK_MOVE MASK_EXEC, MASK_PEER_SAVE
  GLOBAL_STORE v1, v14, s[6:7]
#endif
  MASK_MOVE MASK_EXEC, MASK_SAVE
  consan_wait_stores
  s_endpgm
.Lkernel_end:
.size CONSAN_KERNEL, .Lkernel_end-CONSAN_KERNEL

consan_kernel_descriptor
