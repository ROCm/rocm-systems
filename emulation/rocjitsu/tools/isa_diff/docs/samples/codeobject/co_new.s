.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.globl k
.p2align 8
.type k,@function
k:
  s_load_dwordx2 s[2:3], s[0:1], 0x0
  s_waitcnt lgkmcnt(0)
  s_add_u32 s4, s2, s3
  s_addc_u32 s5, 0, 0
  s_lshl_b64 s[8:9], s[4:5], 1
  s_lshl_b64 s[8:9], s[8:9], 1
  s_cmp_lg_u32 s4, 0
  s_cselect_b32 s7, s8, s9
  s_endpgm
.Lfunc_end0:
  .size k, .Lfunc_end0-k
.rodata
.p2align 6
.amdhsa_kernel k
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 16
  .amdhsa_accum_offset 4
.end_amdhsa_kernel
