// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl empty_kernel
.p2align 8
.type empty_kernel,@function
empty_kernel:
  s_endpgm
.Lfunc_end0:
  .size empty_kernel, .Lfunc_end0-empty_kernel
