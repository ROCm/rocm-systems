// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl empty_kernel
.p2align 8
.type empty_kernel,@function
empty_kernel:
  v_mov_b32_e32 v0, 0
  s_endpgm
.Lfunc_end0:
  .size empty_kernel, .Lfunc_end0-empty_kernel

.rodata
.p2align 8
.amdhsa_kernel empty_kernel
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
  .amdhsa_inst_pref_size 7
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: empty_kernel
      .symbol: empty_kernel.kd
      .sgpr_count: 8
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
