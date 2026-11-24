// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Diagnostic source handoff. Mask 7 is the whole original image. Masks 1/2
// exist only to establish existing single-entry preparation controls.
#ifndef MK_ENTRY_MASK
#define MK_ENTRY_MASK 7
#endif
#ifndef MK_BARRIER
#define MK_BARRIER 0
#endif
#ifndef MK_LOOP
#define MK_LOOP 0
#endif
#ifndef MK_MIXED
#define MK_MIXED 0
#endif
#ifndef MK_NATIVE_GLOBAL
#define MK_NATIVE_GLOBAL 0
#endif
#ifndef MK_UNREACHABLE_PC_READ
#define MK_UNREACHABLE_PC_READ 0
#endif
.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

.macro wait_all
  s_wait_kmcnt 0
  s_wait_dscnt 0
  s_wait_loadcnt 0
.endm

.text
#if MK_ENTRY_MASK & 1
.globl mk_a
.type mk_a,@function
.p2align 8
mk_a:
  s_load_b32 s4, s[0:1], 0
  s_load_b64 s[6:7], s[0:1], 8
  wait_all
  v_lshrrev_b32 v4, 5, v0
  v_lshlrev_b32 v1, 2, v4
  v_add_nc_u32 v2, s4, v4
#if MK_LOOP
  s_mov_b32 s10, 2
.Lmulti_loop_a:
#endif
  v_and_b32 v4, 31, v0
  v_cmp_eq_u32 vcc_lo, 0, v4
  s_and_saveexec_b32 s8, vcc_lo
  ds_store_b32 v1, v2
  wait_all
#if MK_BARRIER
  s_mov_b32 exec_lo, s8
  s_wait_dscnt 0
  s_barrier_signal -1
  s_barrier_wait -1
  v_cmp_eq_u32 vcc_lo, 0, v4
  s_and_saveexec_b32 s8, vcc_lo
#endif
  ds_load_b32 v3, v1
  wait_all
  global_store_b32 v1, v3, s[6:7]
  s_mov_b32 exec_lo, s8
#if MK_LOOP
  s_sub_u32 s10, s10, 1
  s_cmp_eq_u32 s10, 0
  s_cbranch_scc0 .Lmulti_loop_a
#endif
  wait_all
  s_wait_storecnt 0
  s_endpgm
.Lend_a:
.size mk_a, .Lend_a-mk_a
#endif

#if MK_ENTRY_MASK & 2
.globl mk_b
.type mk_b,@function
.p2align 8
mk_b:
  s_load_b64 s[20:21], s[0:1], 0
  s_load_b32 s22, s[0:1], 16
  s_load_b32 s23, s[0:1], 24
  wait_all
  v_lshrrev_b32 v16, 5, v0
  v_lshlrev_b32 v17, 2, v16
  v_mov_b32 v18, s22
  v_add_nc_u32 v18, s23, v18
  v_add_nc_u32 v18, v16, v18
#if MK_LOOP
  s_mov_b32 s25, 2
.Lmulti_loop_b:
#endif
  v_and_b32 v19, 31, v0
  v_cmp_eq_u32 vcc_lo, 0, v19
  s_and_saveexec_b32 s24, vcc_lo
  ds_store_b32 v17, v18 offset:128
  wait_all
#if (MK_BARRIER && !MK_MIXED)
  s_mov_b32 exec_lo, s24
  s_wait_dscnt 0
  s_barrier_signal -1
  s_barrier_wait -1
  v_cmp_eq_u32 vcc_lo, 0, v19
  s_and_saveexec_b32 s24, vcc_lo
#endif
  ds_load_b32 v21, v17 offset:128
  wait_all
  global_store_b32 v17, v21, s[20:21]
  s_mov_b32 exec_lo, s24
#if MK_LOOP
  s_sub_u32 s25, s25, 1
  s_cmp_eq_u32 s25, 0
  s_cbranch_scc0 .Lmulti_loop_b
#endif
  wait_all
  s_wait_storecnt 0
  s_endpgm
.Lend_b:
.size mk_b, .Lend_b-mk_b
#endif

#if MK_ENTRY_MASK & 4
.globl mk_c
.type mk_c,@function
.p2align 8
mk_c:
  s_load_b64 s[16:17], s[0:1], 0
  s_load_b32 s18, s[0:1], 16
  wait_all
  v_lshrrev_b32 v8, 5, v0
  v_lshlrev_b32 v9, 2, v8
  v_lshlrev_b32 v10, 1, v8
  v_add_nc_u32 v10, s18, v10
  v_and_b32 v11, 31, v0
  v_cmp_eq_u32 vcc_lo, 0, v11
  s_and_saveexec_b32 s19, vcc_lo
  ds_store_b32 v9, v10 offset:32
  wait_all
  ds_load_b32 v13, v9 offset:32
  wait_all
  global_store_b32 v9, v13, s[16:17]
#if MK_NATIVE_GLOBAL
#if MK_UNREACHABLE_PC_READ
  s_branch .Lnative_global_done
#else
  // Keep the PC read in a successor so admission must inspect every block.
  s_branch .Lnative_global_pc_read
#endif
.Lnative_global_pc_read:
  s_get_pc_i64 s[20:21]
.Lnative_global_pc:
  s_add_u32 s20, s20, native_global - .Lnative_global_pc
  s_addc_u32 s21, s21, 0
  global_store_b32 v9, v13, s[20:21]
.Lnative_global_done:
#endif
  s_mov_b32 exec_lo, s19
  wait_all
  s_wait_storecnt 0
  s_endpgm
.Lend_c:
.size mk_c, .Lend_c-mk_c
#endif

.macro descriptor name, args, sgprs, vgprs, lds
.rodata
.p2align 6
.amdhsa_kernel \name
  .amdhsa_group_segment_fixed_size \lds
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size \args
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_count 15
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr \vgprs
  .amdhsa_next_free_sgpr \sgprs
  .amdhsa_wavefront_size32 1
  .amdhsa_reserve_vcc 1
.end_amdhsa_kernel
.endm
#if MK_ENTRY_MASK & 1
  descriptor mk_a, 16, 16, 8, 8
#endif
#if MK_ENTRY_MASK & 2
  descriptor mk_b, 32, 32, 24, 256
#endif
#if MK_ENTRY_MASK & 4
  descriptor mk_c, 24, 24, 16, 64
#endif

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.target: "amdgcn-amd-amdhsa--gfx1250"
amdhsa.kernels:
#if MK_ENTRY_MASK & 1
  - .name: mk_a
    .symbol: mk_a.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 8
    .private_segment_fixed_size: 0
    .sgpr_count: 16
    .vgpr_count: 8
    .wavefront_size: 32
    .max_flat_workgroup_size: 64
    .args:
      - .name: seed
        .size: 4
        .offset: 0
        .value_kind: by_value
      - .name: output
        .size: 8
        .offset: 8
        .value_kind: global_buffer
        .address_space: global
#endif
#if MK_ENTRY_MASK & 2
  - .name: mk_b
    .symbol: mk_b.kd
    .kernarg_segment_size: 32
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 256
    .private_segment_fixed_size: 0
    .sgpr_count: 32
    .vgpr_count: 24
    .wavefront_size: 32
    .max_flat_workgroup_size: 64
    .args:
      - .name: output
        .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
      - .name: seed
        .size: 4
        .offset: 16
        .value_kind: by_value
      - .name: bias
        .size: 4
        .offset: 24
        .value_kind: by_value
#endif
#if MK_ENTRY_MASK & 4
  - .name: mk_c
    .symbol: mk_c.kd
    .kernarg_segment_size: 24
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 64
    .private_segment_fixed_size: 0
    .sgpr_count: 24
    .vgpr_count: 16
    .wavefront_size: 32
    .max_flat_workgroup_size: 64
    .args:
      - .name: output
        .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
      - .name: seed
        .size: 4
        .offset: 16
        .value_kind: by_value
#endif
...
.end_amdgpu_metadata

#if MK_NATIVE_GLOBAL
.bss
.p2align 8
.local native_global
.type native_global,@object
native_global:
  .zero 8
.size native_global, .-native_global
#endif
