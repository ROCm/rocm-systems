// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "lds_fixture_macros.s"

.amdgcn_target CONSAN_TARGET
.text
.globl CONSAN_KERNEL
.type CONSAN_KERNEL,@function
.p2align 8
CONSAN_KERNEL:
#if CONSAN_LEGACY
#define HALF_STORE flat_store_short
#define HALF_STORE_HI flat_store_short_d16_hi
#define HALF_LOAD_U flat_load_ushort
#define HALF_LOAD_I flat_load_sshort
#define HALF_LOAD_LO flat_load_short_d16
#define HALF_LOAD_HI flat_load_short_d16_hi
#else
#define HALF_STORE flat_store_b16
#define HALF_STORE_HI flat_store_d16_hi_b16
#define HALF_LOAD_U flat_load_u16
#define HALF_LOAD_I flat_load_i16
#define HALF_LOAD_LO flat_load_d16_b16
#define HALF_LOAD_HI flat_load_d16_hi_b16
#endif

.macro flat_wait
#if CONSAN_GFX12
  consan_wait_all
#else
  s_waitcnt 0
#endif
.endm

.macro save_result reg
#ifdef FLAT_VALUES
  GLOBAL_STORE v7, \reg, s[6:7]
  VECTOR_ADD v7, 4, v7
#endif
.endm

  LOAD32 s4, s[0:1], 0
  LOAD64 s[6:7], s[0:1], 8
  flat_wait
  v_lshrrev_b32 v6, WAVE_SHIFT, v0
  v_and_b32 v12, (CONSAN_WAVE_SIZE-1), v0
  VECTOR_ADD v13, s4, v6
  s_mov_b64 s[4:5], src_shared_base
  v_mov_b32 v5, 0xabcd8123
#ifdef FLAT_VALUES
  // One isolated aligned dword per wave, including the last two allocated bytes.
  v_lshlrev_b32 v4, 3, v6
  VECTOR_ADD v4, 1012, v4
  v_lshlrev_b32 v7, 4, v6
  v_lshlrev_b32 v1, 3, v6
  VECTOR_ADD v7, v1, v7
#else
  v_mov_b32 v4, 1020
  v_lshlrev_b32 v7, 2, v6
#endif
  v_cmp_eq_u32 MASK_VCC, (CONSAN_WAVE_SIZE-1), v12
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
#ifndef FLAT_VALUES
  v_cmp_eq_u32 MASK_VCC, 0, v6
  MASK_AND_SAVE MASK_PEER_SAVE, MASK_VCC
#else
  v_mov_b32 v11, 0x11223344
  DS_STORE_B32 v4, v11
  flat_wait
#endif

  VECTOR_ADD v2, 2, v4
  v_mov_b32_e32 v3, s5
  HALF_STORE v[2:3], v5
  flat_wait
  consan_wait_stores
#ifdef FLAT_VALUES
  DS_LOAD_B32 v9, v4
  flat_wait
  save_result v9
#endif
  HALF_LOAD_U v9, v[2:3]
  flat_wait
  save_result v9
  HALF_LOAD_I v9, v[2:3]
  flat_wait
  save_result v9
  v_mov_b32 v9, 0xdeadbeef
  HALF_LOAD_LO v9, v[2:3]
  flat_wait
  save_result v9
  // A D16 high write aliases the address low VGPR. Save its result before restoring the pointer.
  HALF_LOAD_HI v2, v[2:3]
  flat_wait
  save_result v2
  VECTOR_ADD v2, 2, v4
  HALF_STORE_HI v[2:3], v5
  flat_wait
  consan_wait_stores
#ifdef FLAT_VALUES
  DS_LOAD_B32 v9, v4
  flat_wait
  save_result v9
#else
  // Race payloads do not participate in the register or host output oracle.
  v_mov_b32 v9, 0
  MASK_MOVE MASK_EXEC, MASK_PEER_SAVE
  v_cmp_eq_u32 MASK_VCC, 1, v6
  MASK_AND_SAVE MASK_PEER_SAVE, MASK_VCC
#ifdef FLAT_ADJACENT
  DS_STORE_B16 v4, v5
#else
  DS_STORE_B16 v4, v5 offset:2
#endif
  MASK_MOVE MASK_EXEC, MASK_PEER_SAVE
  GLOBAL_STORE v7, v13, s[6:7]
#endif
  MASK_MOVE MASK_EXEC, MASK_SAVE
  consan_wait_stores
  s_endpgm
.Lkernel_end:
.size CONSAN_KERNEL, .Lkernel_end-CONSAN_KERNEL

consan_kernel_descriptor
