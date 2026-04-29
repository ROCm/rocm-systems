; Copyright (c) 2026 Advanced Micro Devices, Inc.
; SPDX-License-Identifier: MIT
;
; Human-readable CDNA4 assembly extracted from Triton 3.6.0 for gfx950.
; Do not check in the Python source as a standalone kernel; it is included here
; only as comments so this assembly fixture remains auditable.
;
; Triton source:
; import triton
; import triton.language as tl
;
; @triton.jit
; def triton_cdna4_matmul_kernel(A, B, C, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, K_TOTAL: tl.constexpr):
;     offs_m = tl.arange(0, BLOCK_M)
;     offs_n = tl.arange(0, BLOCK_N)
;     offs_k = tl.arange(0, BLOCK_K)
;     acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
;     for k_base in range(0, K_TOTAL, BLOCK_K):
;         k = k_base + offs_k
;         a = tl.load(A + offs_m[:, None] * K_TOTAL + k[None, :])
;         b = tl.load(B + offs_n[:, None] * K_TOTAL + k[None, :])
;         acc += tl.dot(a, tl.trans(b))
;     tl.store(C + offs_m[:, None] * BLOCK_N + offs_n[None, :], acc)
;
; Compile options: target=hip:gfx950, warp_size=64, num_warps=4, num_stages=2,
; BLOCK_M=16, BLOCK_N=16, BLOCK_K=16, K_TOTAL=32. B is stored as N x K.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	triton_cdna4_matmul_kernel      ; -- Begin function triton_cdna4_matmul_kernel
	.p2align	8
	.type	triton_cdna4_matmul_kernel,@function
triton_cdna4_matmul_kernel:             ; @triton_cdna4_matmul_kernel
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.3:
	.file	1 "/home/kunwar/Work/runtime-evolution/rocm-systems-rocjitsu/experimental/rocjitsu/tests/kernels" "triton_cdna4_matmul.py"
	.loc	1 5 0 prologue_end              ; triton_cdna4_matmul.py:5:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.4:
.LBB0_0:
.Ltmp1:
	.loc	1 12 24 is_stmt 1               ; triton_cdna4_matmul.py:12:24
	v_lshlrev_b32_e32 v2, 2, v0
	.loc	1 6 26                          ; triton_cdna4_matmul.py:6:26
	v_and_b32_e32 v1, 15, v0
	.loc	1 12 24                         ; triton_cdna4_matmul.py:12:24
	v_and_b32_e32 v2, 0x3c0, v2
	v_mov_b32_e32 v3, 0
	v_lshl_add_u64 v[4:5], s[2:3], 0, v[2:3]
	.loc	1 13 24                         ; triton_cdna4_matmul.py:13:24
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[2:3]
	.loc	1 12 52                         ; triton_cdna4_matmul.py:12:52
	v_lshlrev_b32_e32 v2, 1, v1
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	.loc	1 12 20 is_stmt 0               ; triton_cdna4_matmul.py:12:20
	global_load_ushort v8, v[4:5], off
	.loc	1 13 52 is_stmt 1               ; triton_cdna4_matmul.py:13:52
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[2:3]
	.loc	1 13 20 is_stmt 0               ; triton_cdna4_matmul.py:13:20
	global_load_ushort v2, v[6:7], off
	.loc	1 12 42 is_stmt 1               ; triton_cdna4_matmul.py:12:42
	v_lshlrev_b32_e32 v9, 1, v0
	.loc	1 12 20 is_stmt 0               ; triton_cdna4_matmul.py:12:20
	v_bfe_i32 v10, v0, 7, 1
	v_and_b32_e32 v9, 0xfe, v9
	v_and_b32_e32 v10, 0x108, v10
	v_xad_u32 v10, v10, v9, 0
	global_load_ushort v5, v[4:5], off offset:32
	v_and_b32_e32 v4, 48, v0
	s_movk_i32 s0, 0x108
	s_movk_i32 s1, 0xe0
	s_waitcnt vmcnt(2)
	ds_write_b16 v10, v8
	.loc	1 13 20 is_stmt 1               ; triton_cdna4_matmul.py:13:20
	s_waitcnt vmcnt(1)
	ds_write_b16 v10, v2 offset:512
	.loc	1 12 20                         ; triton_cdna4_matmul.py:12:20
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 13 20                         ; triton_cdna4_matmul.py:13:20
	global_load_ushort v2, v[6:7], off offset:32
	.loc	1 12 20                         ; triton_cdna4_matmul.py:12:20
	v_bfe_i32 v7, v0, 3, 1
	v_lshrrev_b32_e32 v8, 1, v4
	v_lshlrev_b32_e32 v6, 5, v0
	v_bitop3_b32 v7, v7, v8, s0 bitop3:0x6c
	v_and_or_b32 v6, v6, s1, v7
	v_add_u32_e32 v11, 0, v6
	ds_read2st64_b64 v[6:9], v11 offset1:1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(1)
	ds_write_b16 v10, v5
	.loc	1 14 25                         ; triton_cdna4_matmul.py:14:25
	v_mfma_f32_16x16x16_f16 a[0:3], v[8:9], v[6:7], 0
	.loc	1 15 62                         ; triton_cdna4_matmul.py:15:62
	v_and_b32_e32 v0, 0xc0, v0
	v_cmp_eq_u32_e32 vcc, 0, v0
	.loc	1 13 20                         ; triton_cdna4_matmul.py:13:20
	s_waitcnt vmcnt(0)
	ds_write_b16 v10, v2 offset:512
	.loc	1 12 20                         ; triton_cdna4_matmul.py:12:20
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2st64_b64 v[6:9], v11 offset1:1
	.loc	1 14 25                         ; triton_cdna4_matmul.py:14:25
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 a[0:3], v[8:9], v[6:7], a[0:3]
	.loc	1 15 62                         ; triton_cdna4_matmul.py:15:62
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:                                ; %.critedge
	.loc	1 15 17 is_stmt 0               ; triton_cdna4_matmul.py:15:17
	v_lshlrev_b32_e32 v2, 6, v1
	v_lshl_add_u64 v[0:1], s[6:7], 0, v[2:3]
	.loc	1 15 45                         ; triton_cdna4_matmul.py:15:45
	v_mov_b32_e32 v5, v3
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[4:5]
	.loc	1 15 62                         ; triton_cdna4_matmul.py:15:62
	s_nop 1
	global_store_dwordx4 v[0:1], a[0:3], off
.LBB0_2:                                ; %.critedge4
	.loc	1 15 4                          ; triton_cdna4_matmul.py:15:4
	s_endpgm
.Ltmp2:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel triton_cdna4_matmul_kernel
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 12
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 10
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 16
		.amdhsa_next_free_sgpr 12
		.amdhsa_accum_offset 12
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	triton_cdna4_matmul_kernel, .Lfunc_end0-triton_cdna4_matmul_kernel
	.cfi_endproc
                                        ; -- End function
	.set triton_cdna4_matmul_kernel.num_vgpr, 12
	.set triton_cdna4_matmul_kernel.num_agpr, 4
	.set triton_cdna4_matmul_kernel.numbered_sgpr, 12
	.set triton_cdna4_matmul_kernel.num_named_barrier, 0
	.set triton_cdna4_matmul_kernel.private_seg_size, 0
	.set triton_cdna4_matmul_kernel.uses_vcc, 1
	.set triton_cdna4_matmul_kernel.uses_flat_scratch, 0
	.set triton_cdna4_matmul_kernel.has_dyn_sized_stack, 0
	.set triton_cdna4_matmul_kernel.has_recursion, 0
	.set triton_cdna4_matmul_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 596
; TotalNumSgprs: 18
; NumVgprs: 12
; NumAgprs: 4
; TotalNumVgprs: 16
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 16
; AccumOffset: 12
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 12
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 2
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	0                               ; DW_CHILDREN_no
	.byte	37                              ; DW_AT_producer
	.byte	14                              ; DW_FORM_strp
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	14                              ; DW_FORM_strp
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x1f DW_TAG_compile_unit
	.long	.Linfo_string0                  ; DW_AT_producer
	.short	2                               ; DW_AT_language
	.long	.Linfo_string1                  ; DW_AT_name
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.long	.Linfo_string2                  ; DW_AT_comp_dir
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"triton_cdna4_matmul.py"        ; string offset=7
.Linfo_string2:
	.asciz	"/home/kunwar/Work/runtime-evolution/rocm-systems-rocjitsu/experimental/rocjitsu/tests/kernels" ; string offset=30
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     4
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .max_flat_workgroup_size: 256
    .name:           triton_cdna4_matmul_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         triton_cdna4_matmul_kernel.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     16
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
