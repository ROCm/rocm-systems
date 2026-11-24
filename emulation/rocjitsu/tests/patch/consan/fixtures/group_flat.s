// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "lds_fixture_macros.s"

.amdgcn_target CONSAN_TARGET
.text
#ifdef FLAT_HELPER
// hip_moi::context::{lds_load_at,lds_store_at}<unsigned>, instantiated with the
// device compiler against hip_moi/context.hpp. These local symbols exercise
// the API contract on a bounded body containing the terminal *ptr access.
#if defined(FLAT_STORE) || defined(HELPER_WRONG)
#define HELPER _ZN7hip_moi7context12lds_store_atIjEEvPT_S2_jNS_7site_idE
#elif defined(HELPER_OTHER)
#define HELPER _ZN7hip_moi7context11lds_load_atIiEET_PKS2_jNS_7site_idE
#elif defined(HELPER_LOOKALIKE)
#define HELPER _ZN7hip_moi7context12lds_load_at2IjEET_PKS2_jNS_7site_idE
#else
#define HELPER _ZN7hip_moi7context11lds_load_atIjEET_PKS2_jNS_7site_idE
#endif
#if CONSAN_LEGACY
#define HELPER_LOAD flat_load_dword
#define HELPER_STORE flat_store_dword
#else
#define HELPER_LOAD flat_load_b32
#define HELPER_STORE flat_store_b32
#endif
.macro consan_flat_helper_body
  .local HELPER
  .type HELPER,@function
HELPER:
#ifdef HELPER_EARLIER
  HELPER_LOAD v10, v[2:3]
#if CONSAN_GFX12
  s_wait_loadcnt 0
  s_wait_dscnt 0
#else
  s_waitcnt 0
#endif
#endif
#if defined(HELPER_WIDTH)
#if CONSAN_LEGACY
  flat_load_dwordx2 v[10:11], v[2:3]
#else
  flat_load_b64 v[10:11], v[2:3]
#endif
#elif defined(FLAT_STORE)
  HELPER_STORE v[2:3], v5
#else
  HELPER_LOAD v2, v[2:3]
#endif
#ifdef HELPER_CLOBBER
  s_mov_b32 s12, 0
#endif
#ifdef HELPER_BADRETURN
  s_setpc_b64 s[10:11]
#else
  s_setpc_b64 s[12:13]
#endif
  .size HELPER, .-HELPER
#ifdef HELPER_ALIAS
  .local overlapping_lds_function
  .type overlapping_lds_function,@function
  .set overlapping_lds_function, HELPER
  .size overlapping_lds_function, .-HELPER
#endif
.endm
#if defined(FLAT_STORE) && !defined(HELPER_AFTER_CALLER)
// Keep the store helper's original PC before its later native LDS peer.
consan_flat_helper_body
#endif
#endif

.globl CONSAN_KERNEL
.type CONSAN_KERNEL,@function
.p2align 8
CONSAN_KERNEL:
#if CONSAN_LEGACY
#define FLAT_READ flat_load_dword
#define FLAT_WRITE flat_store_dword
#else
#define FLAT_READ flat_load_b32
#define FLAT_WRITE flat_store_b32
#endif

#if defined(FLAT_HELPER)
#define FLAT_IMMEDIATE 0
#elif defined(FLAT_BACKWARD)
#define FLAT_IMMEDIATE -4
#elif defined(FLAT_UNDERFLOW) || (CONSAN_GFX12 && defined(FLAT_OFFSET))
// Enter with a positive offset beyond the last dword, then return to it.
#define FLAT_IMMEDIATE -4
#elif defined(FLAT_STORE) || defined(FLAT_OFFSET) || defined(FLAT_OVERFLOW)
#define FLAT_IMMEDIATE 4
#else
#define FLAT_IMMEDIATE 0
#endif

.macro flat_wait_memory
#if CONSAN_GFX12
  s_wait_loadcnt 0
  s_wait_storecnt 0
  s_wait_dscnt 0
#else
  s_waitcnt 0
#endif
.endm

.macro group_flat_pointer
#if defined(FLAT_SCALAR) || defined(FLAT_BACKWARD)
  // The scalar low word changes after dispatch. Make its signed displacement
  // select the last legal dword with and without scaling.
#if defined(FLAT_BACKWARD)
  s_mov_b32 s4, 1056
  v_mov_b32_e32 v2, -8
#elif defined(FLAT_STORE)
  s_mov_b32 s4, 1040
  v_mov_b32_e32 v2, -24
#else
  s_mov_b32 s4, 988
  v_mov_b32_e32 v2, 8
#endif
#else
  // Build the entire pair under the same EXEC as the dereference.
#if defined(FLAT_NONGROUP)
  v_mov_b32_e32 v3, s1
  v_mov_b32_e32 v2, s0
#else
  v_mov_b32_e32 v3, s5
#if defined(FLAT_ALTERNATE)
  v_mov_b32_e32 v2, s6
#elif defined(FLAT_ADJACENT) && !defined(FLAT_STORE)
  v_mov_b32_e32 v2, (1016-FLAT_IMMEDIATE)
#elif defined(FLAT_UNDERFLOW)
  v_mov_b32_e32 v2, 0
#elif defined(FLAT_OVERFLOW)
  v_mov_b32_e32 v2, -4
#elif defined(FLAT_OOB)
  v_mov_b32_e32 v2, (1024-FLAT_IMMEDIATE)
#else
  v_mov_b32_e32 v2, (1020-FLAT_IMMEDIATE)
#endif
#endif
#endif
.endm

  LOAD32 s4, s[0:1], 0
  LOAD64 s[6:7], s[0:1], 8
  consan_wait_all
  v_lshrrev_b32 v6, WAVE_SHIFT, v0
  v_and_b32 v12, (CONSAN_WAVE_SIZE-1), v0
  VECTOR_ADD v8, s4, v6
  v_mov_b32 v5, s4
  VECTOR_ADD v5, 1, v5
#if defined(FLAT_STORE) && (defined(FLAT_ADJACENT) || defined(FLAT_TAIL))
  v_mov_b32 v4, 1016
#else
  v_mov_b32 v4, 1020
#endif
#if defined(HELPER_PRIVATE)
  s_mov_b64 s[4:5], src_private_base
#else
  s_mov_b64 s[4:5], src_shared_base
#endif
#if defined(FLAT_LIKELY) || defined(FLAT_ALTERNATE)
  // s7 holds the high word of the output pointer loaded from kernargs.
  // The two controls use opposite SCC values with the same pointer select.
#ifdef FLAT_ALTERNATE
  s_cmp_eq_u32 0, 1
#else
  s_cmp_eq_u32 0, 0
#endif
  s_cselect_b32 s5, s5, s7
#elif defined(FLAT_UNKNOWN)
  // A numerical identity has no recognized aperture provenance.
  s_add_u32 s5, s5, 0
#endif

  v_cmp_eq_u32 MASK_VCC, (CONSAN_WAVE_SIZE-1), v12
  MASK_AND_SAVE MASK_SAVE, MASK_VCC
  v_cmp_eq_u32 MASK_VCC, 0, v6
  MASK_AND_SAVE MASK_PEER_SAVE, MASK_VCC
#if defined(FLAT_STORE)
  group_flat_pointer
#ifdef FLAT_HELPER
  s_call_b64 s[12:13], HELPER
#elif defined(FLAT_BACKWARD)
  FLAT_WRITE v2, v5, s[4:5] offset:FLAT_IMMEDIATE scale_offset
#elif defined(FLAT_SCALAR)
  FLAT_WRITE v2, v5, s[4:5] offset:FLAT_IMMEDIATE
#else
  FLAT_WRITE v[2:3], v5 offset:FLAT_IMMEDIATE
#endif
#elif defined(FLAT_READREAD)
  DS_LOAD_B32 v10, v4
  flat_wait_memory
  v_mov_b32 v10, 0
#else
  DS_STORE_B32 v4, v5
#endif
  MASK_MOVE MASK_EXEC, MASK_PEER_SAVE
#ifdef FLAT_ORDERED
  flat_wait_memory
#if CONSAN_GFX12
  s_barrier_signal -1
  s_barrier_wait -1
#else
  s_barrier
#endif
#endif

#ifdef FLAT_EMPTY
  v_cmp_eq_u32 MASK_VCC, 2, v6
#else
  v_cmp_eq_u32 MASK_VCC, 1, v6
#endif
  MASK_AND_SAVE MASK_PEER_SAVE, MASK_VCC
#ifdef FLAT_STORE
#ifdef FLAT_TAIL
  DS_LOAD_B64 v[10:11], v4
#else
  DS_LOAD_B32 v10, v4
#endif
  flat_wait_memory
#ifdef FLAT_ORDERED
  v_mov_b32 v8, v10
#endif
  v_mov_b32 v10, 0
  v_mov_b32 v11, 0
#else
  group_flat_pointer
  // Alias the result with the low address half to exercise the BeforeInst probe.
#ifdef FLAT_HELPER
  s_call_b64 s[12:13], HELPER
#elif defined(FLAT_SCALAR) || defined(FLAT_BACKWARD)
  FLAT_READ v2, v2, s[4:5] offset:FLAT_IMMEDIATE scale_offset
#else
  FLAT_READ v2, v[2:3] offset:FLAT_IMMEDIATE
#endif
  flat_wait_memory
#ifdef FLAT_ORDERED
  v_mov_b32 v8, v2
#endif
  // The racing read has an unconstrained value; discard it before comparing registers.
  v_mov_b32 v2, 0
#endif
  MASK_MOVE MASK_EXEC, MASK_PEER_SAVE
  v_lshlrev_b32 v1, 2, v6
  GLOBAL_STORE v1, v8, s[6:7]
  MASK_MOVE MASK_EXEC, MASK_SAVE
  consan_wait_stores
  s_endpgm
.Lkernel_end:
.size CONSAN_KERNEL, .Lkernel_end-CONSAN_KERNEL
#if defined(FLAT_HELPER) && (!defined(FLAT_STORE) || defined(HELPER_AFTER_CALLER))
consan_flat_helper_body
#endif

consan_kernel_descriptor
