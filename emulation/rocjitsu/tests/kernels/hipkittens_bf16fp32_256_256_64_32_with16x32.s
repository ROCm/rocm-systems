// Generated from the HipKittens gfx950 code object for DBT tests.
// Source: https://github.com/HazyResearch/HipKittens/blob/main/kernels/gemm/bf16fp32/256_256_64_32_with16x32.cpp
// Original hsaco sha256: 4e8a2f1de147547101c87952e341fd4b0121c22327c620477f585bf3b3b10321
// Instruction words are exact; comments are llvm-objdump decoded assembly.
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.protected _Z8micro_tk13micro_globalsiii
.globl _Z8micro_tk13micro_globalsiii
.p2align 8
.type _Z8micro_tk13micro_globalsiii,@function
_Z8micro_tk13micro_globalsiii:
	.long	0xC0060280, 0x000000B8	; s_load_dwordx2 s[10:11], s[0:1], 0xb8
	.long	0xC0060100, 0x00000000	; s_load_dwordx2 s[4:5], s[0:1], 0x0
	.long	0xC0060400, 0x00000020	; s_load_dwordx2 s[16:17], s[0:1], 0x20
	.long	0xC0060200, 0x00000030	; s_load_dwordx2 s[8:9], s[0:1], 0x30
	.long	0xC0060480, 0x00000050	; s_load_dwordx2 s[18:19], s[0:1], 0x50
	.long	0xBF8CC07F	; s_waitcnt lgkmcnt(0)
	.long	0x9203030A	; s_mul_i32 s3, s10, s3
	.long	0x81060203	; s_add_i32 s6, s3, s2
	.long	0x92020A0B	; s_mul_i32 s2, s11, s10
	.long	0x90039F02	; s_ashr_i32 s3, s2, 31
	.long	0x8F039703	; s_lshr_b32 s3, s3, 23
	.long	0x81020302	; s_add_i32 s2, s2, s3
	.long	0x8602FF02, 0xFFFFFE00	; s_and_b32 s2, s2, 0xfffffe00
	.long	0xBF020206	; s_cmp_gt_i32 s6, s2
	.long	0xBF850013	; s_cbranch_scc1 19
	.long	0x90029F06	; s_ashr_i32 s2, s6, 31
	.long	0x8F039D02	; s_lshr_b32 s3, s2, 29
	.long	0x81030306	; s_add_i32 s3, s6, s3
	.long	0x90078303	; s_ashr_i32 s7, s3, 3
	.long	0x860AFF03, 0x03FFFFF8	; s_and_b32 s10, s3, 0x3fffff8
	.long	0x90039F03	; s_ashr_i32 s3, s3, 31
	.long	0x8F029702	; s_lshr_b32 s2, s2, 23
	.long	0x8F039A03	; s_lshr_b32 s3, s3, 26
	.long	0x818A0A06	; s_sub_i32 s10, s6, s10
	.long	0x81020206	; s_add_i32 s2, s6, s2
	.long	0x81030307	; s_add_i32 s3, s7, s3
	.long	0x8903BF03	; s_andn2_b32 s3, s3, 63
	.long	0x8602FF02, 0xFFFFFE00	; s_and_b32 s2, s2, 0xfffffe00
	.long	0x8E06860A	; s_lshl_b32 s6, s10, 6
	.long	0x81830307	; s_sub_i32 s3, s7, s3
	.long	0x81020602	; s_add_i32 s2, s2, s6
	.long	0x81060302	; s_add_i32 s6, s2, s3
	.long	0xBE8A01EB	; s_mov_b64 s[10:11], src_shared_base
	.long	0xBF07C180	; s_cmp_lg_u32 0, -1
	.long	0x850A8080	; s_cselect_b32 s10, 0, 0
	.long	0x8507800B	; s_cselect_b32 s7, s11, 0
	.long	0x86028F0A	; s_and_b32 s2, s10, 15
	.long	0x860BD00A	; s_and_b32 s11, s10, -16
	.long	0xC00A0300, 0x000000A8	; s_load_dwordx4 s[12:15], s[0:1], 0xa8
	.long	0x800B900B	; s_add_u32 s11, s11, 16
	.long	0xBE830080	; s_mov_b32 s3, 0
	.long	0xBF8CC07F	; s_waitcnt lgkmcnt(0)
	.long	0x820F8007	; s_addc_u32 s15, s7, 0
	.long	0xBF128002	; s_cmp_eq_u64 s[2:3], 0
	.long	0x85160B0A	; s_cselect_b32 s22, s10, s11
	.long	0x85020F07	; s_cselect_b32 s2, s7, s15
	.long	0x8007FF16, 0x00010000	; s_add_u32 s7, s22, 0x10000
	.long	0x860AD007	; s_and_b32 s10, s7, -16
	.long	0x86028F07	; s_and_b32 s2, s7, 15
	.long	0x800A900A	; s_add_u32 s10, s10, 16
	.long	0xBF128002	; s_cmp_eq_u64 s[2:3], 0
	.long	0x85170A07	; s_cselect_b32 s23, s7, s10
	.long	0x8102FF0D, 0x000000FF	; s_add_i32 s2, s13, 0xff
	.long	0x90079F02	; s_ashr_i32 s7, s2, 31
	.long	0x8F079807	; s_lshr_b32 s7, s7, 24
	.long	0x81020702	; s_add_i32 s2, s2, s7
	.long	0x90028802	; s_ashr_i32 s2, s2, 8
	.long	0x8E078302	; s_lshl_b32 s7, s2, 3
	.long	0xBE8A3007	; s_abs_i32 s10, s7
	.long	0x7E020C0A	; v_cvt_f32_u32_e32 v1, s10
	.long	0x810BFF0C, 0x000000FF	; s_add_i32 s11, s12, 0xff
	.long	0x81940A80	; s_sub_i32 s20, 0, s10
	.long	0x900F9F0B	; s_ashr_i32 s15, s11, 31
	.long	0x7E024701	; v_rcp_iflag_f32_e32 v1, v1
	.long	0x8F0F980F	; s_lshr_b32 s15, s15, 24
	.long	0x810B0F0B	; s_add_i32 s11, s11, s15
	.long	0xBE8F3006	; s_abs_i32 s15, s6
	.long	0x0A0202FF, 0x4F7FFFFE	; v_mul_f32_e32 v1, 0x4f7ffffe, v1
	.long	0x7E020F01	; v_cvt_u32_f32_e32 v1, v1
	.long	0x88020206	; s_xor_b32 s2, s6, s2
	.long	0x900B880B	; s_ashr_i32 s11, s11, 8
	.long	0x90029F02	; s_ashr_i32 s2, s2, 31
	.long	0x7E2A0501	; v_readfirstlane_b32 s21, v1
	.long	0x92141514	; s_mul_i32 s20, s20, s21
	.long	0x96141415	; s_mul_hi_u32 s20, s21, s20
	.long	0x81151415	; s_add_i32 s21, s21, s20
	.long	0x9614150F	; s_mul_hi_u32 s20, s15, s21
	.long	0x92150A14	; s_mul_i32 s21, s20, s10
	.long	0x818F150F	; s_sub_i32 s15, s15, s21
	.long	0x81158114	; s_add_i32 s21, s20, 1
	.long	0x81980A0F	; s_sub_i32 s24, s15, s10
	.long	0xBF090A0F	; s_cmp_ge_u32 s15, s10
	.long	0x85141415	; s_cselect_b32 s20, s21, s20
	.long	0x850F0F18	; s_cselect_b32 s15, s24, s15
	.long	0x81158114	; s_add_i32 s21, s20, 1
	.long	0xBF090A0F	; s_cmp_ge_u32 s15, s10
	.long	0x850A1415	; s_cselect_b32 s10, s21, s20
	.long	0x880A020A	; s_xor_b32 s10, s10, s2
	.long	0x8182020A	; s_sub_i32 s2, s10, s2
	.long	0x8E0A8302	; s_lshl_b32 s10, s2, 3
	.long	0x818B0A0B	; s_sub_i32 s11, s11, s10
	.long	0x830B880B	; s_min_i32 s11, s11, 8
	.long	0xBE8F300B	; s_abs_i32 s15, s11
	.long	0x7E020C0F	; v_cvt_f32_u32_e32 v1, s15
	.long	0x81940F80	; s_sub_i32 s20, 0, s15
	.long	0x92020702	; s_mul_i32 s2, s2, s7
	.long	0x81820206	; s_sub_i32 s2, s6, s2
	.long	0x7E024701	; v_rcp_iflag_f32_e32 v1, v1
	.long	0xBE873002	; s_abs_i32 s7, s2
	.long	0x88060B02	; s_xor_b32 s6, s2, s11
	.long	0x90069F06	; s_ashr_i32 s6, s6, 31
	.long	0x0A0202FF, 0x4F7FFFFE	; v_mul_f32_e32 v1, 0x4f7ffffe, v1
	.long	0x7E020F01	; v_cvt_u32_f32_e32 v1, v1
	.long	0x20040086	; v_lshrrev_b32_e32 v2, 6, v0
	.long	0x2406048A	; v_lshlrev_b32_e32 v3, 10, v2
	.long	0x24080084	; v_lshlrev_b32_e32 v4, 4, v0
	.long	0x7E2A0501	; v_readfirstlane_b32 s21, v1
	.long	0x92141514	; s_mul_i32 s20, s20, s21
	.long	0x96141415	; s_mul_hi_u32 s20, s21, s20
	.long	0x81151415	; s_add_i32 s21, s21, s20
	.long	0x96141507	; s_mul_hi_u32 s20, s7, s21
	.long	0x92150F14	; s_mul_i32 s21, s20, s15
	.long	0x81871507	; s_sub_i32 s7, s7, s21
	.long	0x81158114	; s_add_i32 s21, s20, 1
	.long	0x81980F07	; s_sub_i32 s24, s7, s15
	.long	0xBF090F07	; s_cmp_ge_u32 s7, s15
	.long	0x85141415	; s_cselect_b32 s20, s21, s20
	.long	0x85070718	; s_cselect_b32 s7, s24, s7
	.long	0x81158114	; s_add_i32 s21, s20, 1
	.long	0xBF090F07	; s_cmp_ge_u32 s7, s15
	.long	0x85071415	; s_cselect_b32 s7, s21, s20
	.long	0x88070607	; s_xor_b32 s7, s7, s6
	.long	0x818F0607	; s_sub_i32 s15, s7, s6
	.long	0x92060B0F	; s_mul_i32 s6, s15, s11
	.long	0x81940602	; s_sub_i32 s20, s2, s6
	.long	0x8E079110	; s_lshl_b32 s7, s16, 17
	.long	0x81140A14	; s_add_i32 s20, s20, s10
	.long	0x92020C10	; s_mul_i32 s2, s16, s12
	.long	0x870BF507	; s_or_b32 s11, s7, -2.0
	.long	0xBE8A0003	; s_mov_b32 s10, s3
	.long	0x8E068102	; s_lshl_b32 s6, s2, 1
	.long	0x8602FF10, 0x00001FFF	; s_and_b32 s2, s16, 0x1fff
	.long	0x878A040A	; s_or_b64 s[10:11], s[10:11], s[4:5]
	.long	0xBF068002	; s_cmp_eq_u32 s2, 0
	.long	0x92020D12	; s_mul_i32 s2, s18, s13
	.long	0x85050B05	; s_cselect_b32 s5, s5, s11
	.long	0x85040A04	; s_cselect_b32 s4, s4, s10
	.long	0x8E0A8102	; s_lshl_b32 s10, s2, 1
	.long	0x8E029112	; s_lshl_b32 s2, s18, 17
	.long	0x870DF502	; s_or_b32 s13, s2, -2.0
	.long	0xBE8C0003	; s_mov_b32 s12, s3
	.long	0x860BFF12, 0x00001FFF	; s_and_b32 s11, s18, 0x1fff
	.long	0x8782080C	; s_or_b64 s[2:3], s[12:13], s[8:9]
	.long	0xBF06800B	; s_cmp_eq_u32 s11, 0
	.long	0x85090309	; s_cselect_b32 s9, s9, s3
	.long	0x85080208	; s_cselect_b32 s8, s8, s2
	.long	0x68020616	; v_add_u32_e32 v1, s22, v3
	.long	0x8018FF16, 0x00004000	; s_add_u32 s24, s22, 0x4000
	.long	0x7E420501	; v_readfirstlane_b32 s33, v1
	.long	0x68020618	; v_add_u32_e32 v1, s24, v3
	.long	0x801BFF17, 0x00004000	; s_add_u32 s27, s23, 0x4000
	.long	0x7E440501	; v_readfirstlane_b32 s34, v1
	.long	0x68020617	; v_add_u32_e32 v1, s23, v3
	.long	0xD1C80005, 0x02110500	; v_bfe_u32 v5, v0, 2, 4
	.long	0x7E460501	; v_readfirstlane_b32 s35, v1
	.long	0x6802061B	; v_add_u32_e32 v1, s27, v3
	.long	0xD2340504, 0x8A820900	; v_bitop3_b32 v4, v0, v4, 32 bitop3:0x6c
	.long	0x7E480501	; v_readfirstlane_b32 s36, v1
	.long	0x20020083	; v_lshrrev_b32_e32 v1, 3, v0
	.long	0xD2010001, 0x04156101	; v_and_or_b32 v1, v1, 48, v5
	.long	0x200A0081	; v_lshrrev_b32_e32 v5, 1, v0
	.long	0x20080881	; v_lshrrev_b32_e32 v4, 1, v4
	.long	0x260A0AA0	; v_and_b32_e32 v5, 32, v5
	.long	0xD2010004, 0x04153104	; v_and_or_b32 v4, v4, 24, v5
	.long	0xD1E80206, 0x04102101	; v_mad_u64_u32 v[6:7], s[2:3], v1, s16, v[4:5]
	.long	0x8E028610	; s_lshl_b32 s2, s16, 6
	.long	0x8E15880F	; s_lshl_b32 s21, s15, 8
	.long	0xD1FE0084, 0x02040506	; v_add_lshl_u32 v132, v6, s2, 1
	.long	0xD1E80204, 0x04102501	; v_mad_u64_u32 v[4:5], s[2:3], v1, s18, v[4:5]
	.long	0x92251512	; s_mul_i32 s37, s18, s21
	.long	0x8E028612	; s_lshl_b32 s2, s18, 6
	.long	0x8E0C8125	; s_lshl_b32 s12, s37, 1
	.long	0xD1FE0089, 0x02040504	; v_add_lshl_u32 v137, v4, s2, 1
	.long	0xBE82000C	; s_mov_b32 s2, s12
	.long	0xBE830023	; s_mov_b32 s3, s35
	.long	0xBE8700FF, 0x00110000	; s_mov_b32 s7, 0x110000
	.long	0xBE8D0003	; s_mov_b32 s13, s3
	.long	0x8E2A8814	; s_lshl_b32 s42, s20, 8
	.long	0xBE8B0007	; s_mov_b32 s11, s7
	.long	0x25100881	; v_lshlrev_b32_e32 v136, 1, v4
	.long	0xBEFC0080	; s_mov_b32 m0, 0
	.long	0xB7032000	; s_addk_i32 s3, 0x2000
	.long	0x92262A10	; s_mul_i32 s38, s16, s42
	.long	0xBEFC000D	; s_mov_b32 m0, s13
	.long	0xE05D1000, 0x02020088	; buffer_load_dwordx4 v136, s[8:11], s2 offen lds
	.long	0x8E0D8126	; s_lshl_b32 s13, s38, 1
	.long	0xBEFC0003	; s_mov_b32 m0, s3
	.long	0xE05D1000, 0x02020089	; buffer_load_dwordx4 v137, s[8:11], s2 offen lds
	.long	0xBE82000D	; s_mov_b32 s2, s13
	.long	0xBE830021	; s_mov_b32 s3, s33
	.long	0x250A0C81	; v_lshlrev_b32_e32 v133, 1, v6
	.long	0xBE8F0003	; s_mov_b32 s15, s3
	.long	0xB7032000	; s_addk_i32 s3, 0x2000
	.long	0x801CFF16, 0x00008000	; s_add_u32 s28, s22, 0x8000
	.long	0xBEFC000F	; s_mov_b32 m0, s15
	.long	0xE05D1000, 0x02010085	; buffer_load_dwordx4 v133, s[4:7], s2 offen lds
	.long	0x6808061C	; v_add_u32_e32 v4, s28, v3
	.long	0xBEFC0003	; s_mov_b32 m0, s3
	.long	0xE05D1000, 0x02010084	; buffer_load_dwordx4 v132, s[4:7], s2 offen lds
	.long	0x8702FF15, 0x00000080	; s_or_b32 s2, s21, 0x80
	.long	0x92020212	; s_mul_i32 s2, s18, s2
	.long	0x8E0F8102	; s_lshl_b32 s15, s2, 1
	.long	0xBE82000F	; s_mov_b32 s2, s15
	.long	0xBE830024	; s_mov_b32 s3, s36
	.long	0xBE990003	; s_mov_b32 s25, s3
	.long	0xB7032000	; s_addk_i32 s3, 0x2000
	.long	0x7E4E0504	; v_readfirstlane_b32 s39, v4
	.long	0xBEFC0019	; s_mov_b32 m0, s25
	.long	0xE05D1000, 0x02020088	; buffer_load_dwordx4 v136, s[8:11], s2 offen lds
	.long	0x801DFF17, 0x00008000	; s_add_u32 s29, s23, 0x8000
	.long	0xBEFC0003	; s_mov_b32 m0, s3
	.long	0xE05D1000, 0x02020089	; buffer_load_dwordx4 v137, s[8:11], s2 offen lds
	.long	0x8702FF2A, 0x00000080	; s_or_b32 s2, s42, 0x80
	.long	0x921E0210	; s_mul_i32 s30, s16, s2
	.long	0x8E02811E	; s_lshl_b32 s2, s30, 1
	.long	0xBE830022	; s_mov_b32 s3, s34
	.long	0xBE990003	; s_mov_b32 s25, s3
	.long	0xB7032000	; s_addk_i32 s3, 0x2000
	.long	0x801AFF17, 0x0000C000	; s_add_u32 s26, s23, 0xc000
	.long	0xBEFC0019	; s_mov_b32 m0, s25
	.long	0xE05D1000, 0x02010085	; buffer_load_dwordx4 v133, s[4:7], s2 offen lds
	.long	0x8019FF16, 0x0000C000	; s_add_u32 s25, s22, 0xc000
	.long	0xBEFC0003	; s_mov_b32 m0, s3
	.long	0xE05D1000, 0x02010084	; buffer_load_dwordx4 v132, s[4:7], s2 offen lds
	.long	0x68080619	; v_add_u32_e32 v4, s25, v3
	.long	0x20020088	; v_lshrrev_b32_e32 v1, 8, v0
	.long	0x7E3E0504	; v_readfirstlane_b32 s31, v4
	.long	0x6808061D	; v_add_u32_e32 v4, s29, v3
	.long	0x6806061A	; v_add_u32_e32 v3, s26, v3
	.long	0x7E500504	; v_readfirstlane_b32 s40, v4
	.long	0x7E520503	; v_readfirstlane_b32 s41, v3
	.long	0x7D940281	; v_cmp_eq_u32_e32 vcc, 1, v1
	.long	0xBE82206A	; s_and_saveexec_b64 s[2:3], vcc
	.long	0xBF880001	; s_cbranch_execz 1
	.long	0xBF8A0000	; s_barrier
	.long	0x87FE027E	; s_or_b64 exec, exec, s[2:3]
	.long	0x8702FF0C, 0x00000080	; s_or_b32 s2, s12, 0x80
	.long	0xBE830028	; s_mov_b32 s3, s40
	.long	0xBF8C0F74	; s_waitcnt vmcnt(4)
	.long	0xBF8A0000	; s_barrier
	.long	0xBE8C0003	; s_mov_b32 s12, s3
	.long	0xB7032000	; s_addk_i32 s3, 0x2000
	.long	0x27060483	; v_and_b32_e32 v131, 3, v2
	.long	0xBEFC000C	; s_mov_b32 m0, s12
	.long	0xE05D1000, 0x02020088	; buffer_load_dwordx4 v136, s[8:11], s2 offen lds
	.long	0x2405068B	; v_lshlrev_b32_e32 v2, 11, v131
	.long	0xBEFC0003	; s_mov_b32 m0, s3
	.long	0xE05D1000, 0x02020089	; buffer_load_dwordx4 v137, s[8:11], s2 offen lds
	.long	0x8702FF0D, 0x00000080	; s_or_b32 s2, s13, 0x80
	.long	0xBE830027	; s_mov_b32 s3, s39
	.long	0xBE8C0003	; s_mov_b32 s12, s3
	.long	0xB7032000	; s_addk_i32 s3, 0x2000
	.long	0x7E0A0280	; v_mov_b32_e32 v5, 0
	.long	0xBEFC000C	; s_mov_b32 m0, s12
	.long	0xE05D1000, 0x02010085	; buffer_load_dwordx4 v133, s[4:7], s2 offen lds
	.long	0x271400B0	; v_and_b32_e32 v138, 48, v0
	.long	0xBEFC0003	; s_mov_b32 m0, s3
	.long	0xE05D1000, 0x02010084	; buffer_load_dwordx4 v132, s[4:7], s2 offen lds
	.long	0x8702FF0F, 0x00000080	; s_or_b32 s2, s15, 0x80
	.long	0xBE830029	; s_mov_b32 s3, s41
	.long	0xBE8C0003	; s_mov_b32 s12, s3
	.long	0xB7032000	; s_addk_i32 s3, 0x2000
	.long	0x25160086	; v_lshlrev_b32_e32 v139, 6, v0
	.long	0xBEFC000C	; s_mov_b32 m0, s12
	.long	0xE05D1000, 0x02020088	; buffer_load_dwordx4 v136, s[8:11], s2 offen lds
	.long	0x25180082	; v_lshlrev_b32_e32 v140, 2, v0
	.long	0xBEFC0003	; s_mov_b32 m0, s3
	.long	0xE05D1000, 0x02020089	; buffer_load_dwordx4 v137, s[8:11], s2 offen lds
	.long	0x90029F0E	; s_ashr_i32 s2, s14, 31
	.long	0x8F029A02	; s_lshr_b32 s2, s2, 26
	.long	0x812B020E	; s_add_i32 s43, s14, s2
	.long	0xB20E00BF	; s_cmpk_gt_i32 s14, 0xbf
	.long	0x2504028C	; v_lshlrev_b32_e32 v130, 12, v1
	.long	0xBF8C0F76	; s_waitcnt vmcnt(6)
	.long	0xBF8A0000	; s_barrier
	.long	0xBF850007	; s_cbranch_scc1 7
	.long	0x260716FF, 0x000003C0	; v_and_b32_e32 v3, 0x3c0, v139
	.long	0x260918A0	; v_and_b32_e32 v4, 32, v140
	.long	0xD2340686, 0xC62A0903	; v_bitop3_b32 v134, v3, v4, v138 bitop3:0x36
	.long	0xBE8E0180	; s_mov_b64 s[14:15], 0
	.long	0xBF820001	; s_branch 1
	.long	0xBE8E01C1	; s_mov_b64 s[14:15], -1
	.long	0xC0060080, 0x00000060	; s_load_dwordx2 s[2:3], s[0:1], 0x60
	.long	0xC0060300, 0x00000080	; s_load_dwordx2 s[12:13], s[0:1], 0x80
	.long	0xBF8CC07F	; s_waitcnt lgkmcnt(0)
	.long	0x900D862B	; s_ashr_i32 s13, s43, 6
	.long	0x89EA0E7E	; s_andn2_b64 vcc, exec, s[14:15]
	.long	0x250E0481	; v_lshlrev_b32_e32 v135, 1, v2
	.long	0x7E080280	; v_mov_b32_e32 v4, 0
	.long	0x7E060280	; v_mov_b32_e32 v3, 0
	.long	0x7E040280	; v_mov_b32_e32 v2, 0
	.long	0x7E120280	; v_mov_b32_e32 v9, 0
	.long	0x7E100280	; v_mov_b32_e32 v8, 0
	.long	0x7E0E0280	; v_mov_b32_e32 v7, 0
	.long	0x7E0C0280	; v_mov_b32_e32 v6, 0
	.long	0x7E1A0280	; v_mov_b32_e32 v13, 0
	.long	0x7E180280	; v_mov_b32_e32 v12, 0
	.long	0x7E160280	; v_mov_b32_e32 v11, 0
	.long	0x7E140280	; v_mov_b32_e32 v10, 0
	.long	0x7E220280	; v_mov_b32_e32 v17, 0
	.long	0x7E200280	; v_mov_b32_e32 v16, 0
	.long	0x7E1E0280	; v_mov_b32_e32 v15, 0
	.long	0x7E1C0280	; v_mov_b32_e32 v14, 0
	.long	0x7E2A0280	; v_mov_b32_e32 v21, 0
	.long	0x7E280280	; v_mov_b32_e32 v20, 0
	.long	0x7E260280	; v_mov_b32_e32 v19, 0
	.long	0x7E240280	; v_mov_b32_e32 v18, 0
	.long	0x7E320280	; v_mov_b32_e32 v25, 0
	.long	0x7E300280	; v_mov_b32_e32 v24, 0
	.long	0x7E2E0280	; v_mov_b32_e32 v23, 0
	.long	0x7E2C0280	; v_mov_b32_e32 v22, 0
	.long	0x7E3A0280	; v_mov_b32_e32 v29, 0
	.long	0x7E380280	; v_mov_b32_e32 v28, 0
	.long	0x7E360280	; v_mov_b32_e32 v27, 0
	.long	0x7E340280	; v_mov_b32_e32 v26, 0
	.long	0x7E420280	; v_mov_b32_e32 v33, 0
	.long	0x7E400280	; v_mov_b32_e32 v32, 0
	.long	0x7E3E0280	; v_mov_b32_e32 v31, 0
	.long	0x7E3C0280	; v_mov_b32_e32 v30, 0
	.long	0x7E4A0280	; v_mov_b32_e32 v37, 0
	.long	0x7E480280	; v_mov_b32_e32 v36, 0
	.long	0x7E460280	; v_mov_b32_e32 v35, 0
	.long	0x7E440280	; v_mov_b32_e32 v34, 0
	.long	0x7E520280	; v_mov_b32_e32 v41, 0
	.long	0x7E500280	; v_mov_b32_e32 v40, 0
	.long	0x7E4E0280	; v_mov_b32_e32 v39, 0
	.long	0x7E4C0280	; v_mov_b32_e32 v38, 0
	.long	0x7E5A0280	; v_mov_b32_e32 v45, 0
	.long	0x7E580280	; v_mov_b32_e32 v44, 0
	.long	0x7E560280	; v_mov_b32_e32 v43, 0
	.long	0x7E540280	; v_mov_b32_e32 v42, 0
	.long	0x7E620280	; v_mov_b32_e32 v49, 0
	.long	0x7E600280	; v_mov_b32_e32 v48, 0
	.long	0x7E5E0280	; v_mov_b32_e32 v47, 0
	.long	0x7E5C0280	; v_mov_b32_e32 v46, 0
	.long	0x7E6A0280	; v_mov_b32_e32 v53, 0
	.long	0x7E680280	; v_mov_b32_e32 v52, 0
	.long	0x7E660280	; v_mov_b32_e32 v51, 0
	.long	0x7E640280	; v_mov_b32_e32 v50, 0
	.long	0x7E720280	; v_mov_b32_e32 v57, 0
	.long	0x7E700280	; v_mov_b32_e32 v56, 0
	.long	0x7E6E0280	; v_mov_b32_e32 v55, 0
	.long	0x7E6C0280	; v_mov_b32_e32 v54, 0
	.long	0x7E7A0280	; v_mov_b32_e32 v61, 0
	.long	0x7E780280	; v_mov_b32_e32 v60, 0
	.long	0x7E760280	; v_mov_b32_e32 v59, 0
	.long	0x7E740280	; v_mov_b32_e32 v58, 0
	.long	0x7E820280	; v_mov_b32_e32 v65, 0
	.long	0x7E800280	; v_mov_b32_e32 v64, 0
	.long	0x7E7E0280	; v_mov_b32_e32 v63, 0
	.long	0x7E7C0280	; v_mov_b32_e32 v62, 0
	.long	0x7E8A0280	; v_mov_b32_e32 v69, 0
	.long	0x7E880280	; v_mov_b32_e32 v68, 0
	.long	0x7E860280	; v_mov_b32_e32 v67, 0
	.long	0x7E840280	; v_mov_b32_e32 v66, 0
	.long	0x7E920280	; v_mov_b32_e32 v73, 0
	.long	0x7E900280	; v_mov_b32_e32 v72, 0
	.long	0x7E8E0280	; v_mov_b32_e32 v71, 0
	.long	0x7E8C0280	; v_mov_b32_e32 v70, 0
	.long	0x7E9A0280	; v_mov_b32_e32 v77, 0
	.long	0x7E980280	; v_mov_b32_e32 v76, 0
	.long	0x7E960280	; v_mov_b32_e32 v75, 0
	.long	0x7E940280	; v_mov_b32_e32 v74, 0
	.long	0x7EA20280	; v_mov_b32_e32 v81, 0
	.long	0x7EA00280	; v_mov_b32_e32 v80, 0
	.long	0x7E9E0280	; v_mov_b32_e32 v79, 0
	.long	0x7E9C0280	; v_mov_b32_e32 v78, 0
	.long	0x7EAA0280	; v_mov_b32_e32 v85, 0
	.long	0x7EA80280	; v_mov_b32_e32 v84, 0
	.long	0x7EA60280	; v_mov_b32_e32 v83, 0
	.long	0x7EA40280	; v_mov_b32_e32 v82, 0
	.long	0x7EB20280	; v_mov_b32_e32 v89, 0
	.long	0x7EB00280	; v_mov_b32_e32 v88, 0
	.long	0x7EAE0280	; v_mov_b32_e32 v87, 0
	.long	0x7EAC0280	; v_mov_b32_e32 v86, 0
	.long	0x7EBA0280	; v_mov_b32_e32 v93, 0
	.long	0x7EB80280	; v_mov_b32_e32 v92, 0
	.long	0x7EB60280	; v_mov_b32_e32 v91, 0
	.long	0x7EB40280	; v_mov_b32_e32 v90, 0
	.long	0x7EC20280	; v_mov_b32_e32 v97, 0
	.long	0x7EC00280	; v_mov_b32_e32 v96, 0
	.long	0x7EBE0280	; v_mov_b32_e32 v95, 0
	.long	0x7EBC0280	; v_mov_b32_e32 v94, 0
	.long	0x7ECA0280	; v_mov_b32_e32 v101, 0
	.long	0x7EC80280	; v_mov_b32_e32 v100, 0
	.long	0x7EC60280	; v_mov_b32_e32 v99, 0
	.long	0x7EC40280	; v_mov_b32_e32 v98, 0
	.long	0x7ED20280	; v_mov_b32_e32 v105, 0
	.long	0x7ED00280	; v_mov_b32_e32 v104, 0
	.long	0x7ECE0280	; v_mov_b32_e32 v103, 0
	.long	0x7ECC0280	; v_mov_b32_e32 v102, 0
	.long	0x7EDA0280	; v_mov_b32_e32 v109, 0
	.long	0x7ED80280	; v_mov_b32_e32 v108, 0
	.long	0x7ED60280	; v_mov_b32_e32 v107, 0
	.long	0x7ED40280	; v_mov_b32_e32 v106, 0
	.long	0x7EE20280	; v_mov_b32_e32 v113, 0
	.long	0x7EE00280	; v_mov_b32_e32 v112, 0
	.long	0x7EDE0280	; v_mov_b32_e32 v111, 0
	.long	0x7EDC0280	; v_mov_b32_e32 v110, 0
	.long	0x7EEA0280	; v_mov_b32_e32 v117, 0
	.long	0x7EE80280	; v_mov_b32_e32 v116, 0
	.long	0x7EE60280	; v_mov_b32_e32 v115, 0
	.long	0x7EE40280	; v_mov_b32_e32 v114, 0
	.long	0x7EF20280	; v_mov_b32_e32 v121, 0
	.long	0x7EF00280	; v_mov_b32_e32 v120, 0
	.long	0x7EEE0280	; v_mov_b32_e32 v119, 0
	.long	0x7EEC0280	; v_mov_b32_e32 v118, 0
	.long	0x7EFA0280	; v_mov_b32_e32 v125, 0
	.long	0x7EF80280	; v_mov_b32_e32 v124, 0
	.long	0x7EF60280	; v_mov_b32_e32 v123, 0
	.long	0x7EF40280	; v_mov_b32_e32 v122, 0
	.long	0x7F020280	; v_mov_b32_e32 v129, 0
	.long	0x7F000280	; v_mov_b32_e32 v128, 0
	.long	0x7EFE0280	; v_mov_b32_e32 v127, 0
	.long	0x7EFC0280	; v_mov_b32_e32 v126, 0
	.long	0xBF8702A5	; s_cbranch_vccnz 677
	.long	0x9400FF10, 0x00200000	; s_bfe_i64 s[0:1], s[16:17], 0x200000
	.long	0x260516FF, 0x000003C0	; v_and_b32_e32 v2, 0x3c0, v139
	.long	0x260718A0	; v_and_b32_e32 v3, 32, v140
	.long	0x8001FF2A, 0x00000080	; s_add_u32 s1, s42, 0x80
	.long	0x9410FF12, 0x00200000	; s_bfe_i64 s[16:17], s[18:19], 0x200000
	.long	0xD2340686, 0xC62A0702	; v_bitop3_b32 v134, v2, v3, v138 bitop3:0x36
	.long	0x7E040280	; v_mov_b32_e32 v2, 0
	.long	0x2406028D	; v_lshlrev_b32_e32 v3, 13, v1
	.long	0x920F0001	; s_mul_i32 s15, s1, s0
	.long	0x8000FF15, 0x00000080	; s_add_u32 s0, s21, 0x80
	.long	0x810EC20D	; s_add_i32 s14, s13, -2
	.long	0xD1FF008A, 0x061B0E17	; v_add3_u32 v138, s23, v135, v134
	.long	0xD1FF008B, 0x061A0616	; v_add3_u32 v139, s22, v3, v134
	.long	0xD1FF008C, 0x061B0E1B	; v_add3_u32 v140, s27, v135, v134
	.long	0xD1FF008D, 0x061A0618	; v_add3_u32 v141, s24, v3, v134
	.long	0xD1FF008E, 0x061B0E1D	; v_add3_u32 v142, s29, v135, v134
	.long	0xD1FF008F, 0x061A061C	; v_add3_u32 v143, s28, v3, v134
	.long	0xD1FF0090, 0x061B0E1A	; v_add3_u32 v144, s26, v135, v134
	.long	0xD1FF0091, 0x061A0619	; v_add3_u32 v145, s25, v3, v134
	.long	0x92101000	; s_mul_i32 s16, s0, s16
	.long	0xBE910080	; s_mov_b32 s17, 0
	.long	0xBE800180	; s_mov_b64 s[0:1], 0
	.long	0x7E060302	; v_mov_b32_e32 v3, v2
	.long	0x7E080302	; v_mov_b32_e32 v4, v2
	.long	0x7E0A0302	; v_mov_b32_e32 v5, v2
	.long	0x7E0C0302	; v_mov_b32_e32 v6, v2
	.long	0x7E0E0302	; v_mov_b32_e32 v7, v2
	.long	0x7E100302	; v_mov_b32_e32 v8, v2
	.long	0x7E120302	; v_mov_b32_e32 v9, v2
	.long	0x7E140302	; v_mov_b32_e32 v10, v2
	.long	0x7E160302	; v_mov_b32_e32 v11, v2
	.long	0x7E180302	; v_mov_b32_e32 v12, v2
	.long	0x7E1A0302	; v_mov_b32_e32 v13, v2
	.long	0x7E1C0302	; v_mov_b32_e32 v14, v2
	.long	0x7E1E0302	; v_mov_b32_e32 v15, v2
	.long	0x7E200302	; v_mov_b32_e32 v16, v2
	.long	0x7E220302	; v_mov_b32_e32 v17, v2
	.long	0x7E240302	; v_mov_b32_e32 v18, v2
	.long	0x7E260302	; v_mov_b32_e32 v19, v2
	.long	0x7E280302	; v_mov_b32_e32 v20, v2
	.long	0x7E2A0302	; v_mov_b32_e32 v21, v2
	.long	0x7E2C0302	; v_mov_b32_e32 v22, v2
	.long	0x7E2E0302	; v_mov_b32_e32 v23, v2
	.long	0x7E300302	; v_mov_b32_e32 v24, v2
	.long	0x7E320302	; v_mov_b32_e32 v25, v2
	.long	0x7E340302	; v_mov_b32_e32 v26, v2
	.long	0x7E360302	; v_mov_b32_e32 v27, v2
	.long	0x7E380302	; v_mov_b32_e32 v28, v2
	.long	0x7E3A0302	; v_mov_b32_e32 v29, v2
	.long	0x7E3C0302	; v_mov_b32_e32 v30, v2
	.long	0x7E3E0302	; v_mov_b32_e32 v31, v2
	.long	0x7E400302	; v_mov_b32_e32 v32, v2
	.long	0x7E420302	; v_mov_b32_e32 v33, v2
	.long	0x7E440302	; v_mov_b32_e32 v34, v2
	.long	0x7E460302	; v_mov_b32_e32 v35, v2
	.long	0x7E480302	; v_mov_b32_e32 v36, v2
	.long	0x7E4A0302	; v_mov_b32_e32 v37, v2
	.long	0x7E4C0302	; v_mov_b32_e32 v38, v2
	.long	0x7E4E0302	; v_mov_b32_e32 v39, v2
	.long	0x7E500302	; v_mov_b32_e32 v40, v2
	.long	0x7E520302	; v_mov_b32_e32 v41, v2
	.long	0x7E540302	; v_mov_b32_e32 v42, v2
	.long	0x7E560302	; v_mov_b32_e32 v43, v2
	.long	0x7E580302	; v_mov_b32_e32 v44, v2
	.long	0x7E5A0302	; v_mov_b32_e32 v45, v2
	.long	0x7E5C0302	; v_mov_b32_e32 v46, v2
	.long	0x7E5E0302	; v_mov_b32_e32 v47, v2
	.long	0x7E600302	; v_mov_b32_e32 v48, v2
	.long	0x7E620302	; v_mov_b32_e32 v49, v2
	.long	0x7E640302	; v_mov_b32_e32 v50, v2
	.long	0x7E660302	; v_mov_b32_e32 v51, v2
	.long	0x7E680302	; v_mov_b32_e32 v52, v2
	.long	0x7E6A0302	; v_mov_b32_e32 v53, v2
	.long	0x7E6C0302	; v_mov_b32_e32 v54, v2
	.long	0x7E6E0302	; v_mov_b32_e32 v55, v2
	.long	0x7E700302	; v_mov_b32_e32 v56, v2
	.long	0x7E720302	; v_mov_b32_e32 v57, v2
	.long	0x7E740302	; v_mov_b32_e32 v58, v2
	.long	0x7E760302	; v_mov_b32_e32 v59, v2
	.long	0x7E780302	; v_mov_b32_e32 v60, v2
	.long	0x7E7A0302	; v_mov_b32_e32 v61, v2
	.long	0x7E7C0302	; v_mov_b32_e32 v62, v2
	.long	0x7E7E0302	; v_mov_b32_e32 v63, v2
	.long	0x7E800302	; v_mov_b32_e32 v64, v2
	.long	0x7E820302	; v_mov_b32_e32 v65, v2
	.long	0x7E840302	; v_mov_b32_e32 v66, v2
	.long	0x7E860302	; v_mov_b32_e32 v67, v2
	.long	0x7E880302	; v_mov_b32_e32 v68, v2
	.long	0x7E8A0302	; v_mov_b32_e32 v69, v2
	.long	0x7E8C0302	; v_mov_b32_e32 v70, v2
	.long	0x7E8E0302	; v_mov_b32_e32 v71, v2
	.long	0x7E900302	; v_mov_b32_e32 v72, v2
	.long	0x7E920302	; v_mov_b32_e32 v73, v2
	.long	0x7E940302	; v_mov_b32_e32 v74, v2
	.long	0x7E960302	; v_mov_b32_e32 v75, v2
	.long	0x7E980302	; v_mov_b32_e32 v76, v2
	.long	0x7E9A0302	; v_mov_b32_e32 v77, v2
	.long	0x7E9C0302	; v_mov_b32_e32 v78, v2
	.long	0x7E9E0302	; v_mov_b32_e32 v79, v2
	.long	0x7EA00302	; v_mov_b32_e32 v80, v2
	.long	0x7EA20302	; v_mov_b32_e32 v81, v2
	.long	0x7EA40302	; v_mov_b32_e32 v82, v2
	.long	0x7EA60302	; v_mov_b32_e32 v83, v2
	.long	0x7EA80302	; v_mov_b32_e32 v84, v2
	.long	0x7EAA0302	; v_mov_b32_e32 v85, v2
	.long	0x7EAC0302	; v_mov_b32_e32 v86, v2
	.long	0x7EAE0302	; v_mov_b32_e32 v87, v2
	.long	0x7EB00302	; v_mov_b32_e32 v88, v2
	.long	0x7EB20302	; v_mov_b32_e32 v89, v2
	.long	0x7EB40302	; v_mov_b32_e32 v90, v2
	.long	0x7EB60302	; v_mov_b32_e32 v91, v2
	.long	0x7EB80302	; v_mov_b32_e32 v92, v2
	.long	0x7EBA0302	; v_mov_b32_e32 v93, v2
	.long	0x7EBC0302	; v_mov_b32_e32 v94, v2
	.long	0x7EBE0302	; v_mov_b32_e32 v95, v2
	.long	0x7EC00302	; v_mov_b32_e32 v96, v2
	.long	0x7EC20302	; v_mov_b32_e32 v97, v2
	.long	0x7EC40302	; v_mov_b32_e32 v98, v2
	.long	0x7EC60302	; v_mov_b32_e32 v99, v2
	.long	0x7EC80302	; v_mov_b32_e32 v100, v2
	.long	0x7ECA0302	; v_mov_b32_e32 v101, v2
	.long	0x7ECC0302	; v_mov_b32_e32 v102, v2
	.long	0x7ECE0302	; v_mov_b32_e32 v103, v2
	.long	0x7ED00302	; v_mov_b32_e32 v104, v2
	.long	0x7ED20302	; v_mov_b32_e32 v105, v2
	.long	0x7ED40302	; v_mov_b32_e32 v106, v2
	.long	0x7ED60302	; v_mov_b32_e32 v107, v2
	.long	0x7ED80302	; v_mov_b32_e32 v108, v2
	.long	0x7EDA0302	; v_mov_b32_e32 v109, v2
	.long	0x7EDC0302	; v_mov_b32_e32 v110, v2
	.long	0x7EDE0302	; v_mov_b32_e32 v111, v2
	.long	0x7EE00302	; v_mov_b32_e32 v112, v2
	.long	0x7EE20302	; v_mov_b32_e32 v113, v2
	.long	0x7EE40302	; v_mov_b32_e32 v114, v2
	.long	0x7EE60302	; v_mov_b32_e32 v115, v2
	.long	0x7EE80302	; v_mov_b32_e32 v116, v2
	.long	0x7EEA0302	; v_mov_b32_e32 v117, v2
	.long	0x7EEC0302	; v_mov_b32_e32 v118, v2
	.long	0x7EEE0302	; v_mov_b32_e32 v119, v2
	.long	0x7EF00302	; v_mov_b32_e32 v120, v2
	.long	0x7EF20302	; v_mov_b32_e32 v121, v2
	.long	0x7EF40302	; v_mov_b32_e32 v122, v2
	.long	0x7EF60302	; v_mov_b32_e32 v123, v2
	.long	0x7EF80302	; v_mov_b32_e32 v124, v2
	.long	0x7EFA0302	; v_mov_b32_e32 v125, v2
	.long	0x7EFC0302	; v_mov_b32_e32 v126, v2
	.long	0x7EFE0302	; v_mov_b32_e32 v127, v2
	.long	0x7F000302	; v_mov_b32_e32 v128, v2
	.long	0x7F020302	; v_mov_b32_e32 v129, v2
	.long	0xD9FE0000, 0x9200008A	; ds_read_b128 v[146:149], v138
	.long	0xD9FE0400, 0x9600008A	; ds_read_b128 v[150:153], v138 offset:1024
	.long	0xD9FE0800, 0x9A00008A	; ds_read_b128 v[154:157], v138 offset:2048
	.long	0xD9FE0C00, 0x9E00008A	; ds_read_b128 v[158:161], v138 offset:3072
	.long	0xD9FE0000, 0xA200008B	; ds_read_b128 v[162:165], v139
	.long	0xD9FE0400, 0xA600008B	; ds_read_b128 v[166:169], v139 offset:1024
	.long	0xD9FE0800, 0xAA00008B	; ds_read_b128 v[170:173], v139 offset:2048
	.long	0xD9FE0C00, 0xAE00008B	; ds_read_b128 v[174:177], v139 offset:3072
	.long	0xD9FE1000, 0xB200008B	; ds_read_b128 v[178:181], v139 offset:4096
	.long	0x8012000F	; s_add_u32 s18, s15, s0
	.long	0xD9FE1400, 0xB600008B	; ds_read_b128 v[182:185], v139 offset:5120
	.long	0x8013C012	; s_add_u32 s19, s18, 64
	.long	0xD9FE1800, 0xBA00008B	; ds_read_b128 v[186:189], v139 offset:6144
	.long	0x8E138113	; s_lshl_b32 s19, s19, 1
	.long	0xBEAA001F	; s_mov_b32 s42, s31
	.long	0xD9FE1C00, 0xBE00008B	; ds_read_b128 v[190:193], v139 offset:7168
	.long	0xBEAB002A	; s_mov_b32 s43, s42
	.long	0xB72A2000	; s_addk_i32 s42, 0x2000
	.long	0xBF800000	; s_nop 0
	.long	0xBEFC002B	; s_mov_b32 m0, s43
	.long	0xE05D1000, 0x13010085	; buffer_load_dwordx4 v133, s[4:7], s19 offen lds
	.long	0xBF800000	; s_nop 0
	.long	0xBEFC002A	; s_mov_b32 m0, s42
	.long	0xE05D1000, 0x13010084	; buffer_load_dwordx4 v132, s[4:7], s19 offen lds
	.long	0xBF8CC87F	; s_waitcnt lgkmcnt(8)
	.long	0xBF8A0000	; s_barrier
	.long	0xBF8CC07F	; s_waitcnt lgkmcnt(0)
	.long	0xBF8F0001	; s_setprio 1
	.long	0xD3B5007E, 0x05FB25A2	; v_mfma_f32_16x16x32_bf16 v[126:129], v[162:165], v[146:149], v[126:129]
	.long	0xD3B5007A, 0x05EB35A2	; v_mfma_f32_16x16x32_bf16 v[122:125], v[162:165], v[154:157], v[122:125]
	.long	0xD3B50076, 0x05DB25AA	; v_mfma_f32_16x16x32_bf16 v[118:121], v[170:173], v[146:149], v[118:121]
	.long	0xD3B50072, 0x05CB35AA	; v_mfma_f32_16x16x32_bf16 v[114:117], v[170:173], v[154:157], v[114:117]
	.long	0xD3B5006E, 0x05BB25B2	; v_mfma_f32_16x16x32_bf16 v[110:113], v[178:181], v[146:149], v[110:113]
	.long	0xD3B5006A, 0x05AB35B2	; v_mfma_f32_16x16x32_bf16 v[106:109], v[178:181], v[154:157], v[106:109]
	.long	0xD3B50066, 0x059B25BA	; v_mfma_f32_16x16x32_bf16 v[102:105], v[186:189], v[146:149], v[102:105]
	.long	0xD3B50062, 0x058B35BA	; v_mfma_f32_16x16x32_bf16 v[98:101], v[186:189], v[154:157], v[98:101]
	.long	0xD3B5007E, 0x05FB2DA6	; v_mfma_f32_16x16x32_bf16 v[126:129], v[166:169], v[150:153], v[126:129]
	.long	0xD3B5007A, 0x05EB3DA6	; v_mfma_f32_16x16x32_bf16 v[122:125], v[166:169], v[158:161], v[122:125]
	.long	0xD3B50076, 0x05DB2DAE	; v_mfma_f32_16x16x32_bf16 v[118:121], v[174:177], v[150:153], v[118:121]
	.long	0xD3B50072, 0x05CB3DAE	; v_mfma_f32_16x16x32_bf16 v[114:117], v[174:177], v[158:161], v[114:117]
	.long	0xD3B5006E, 0x05BB2DB6	; v_mfma_f32_16x16x32_bf16 v[110:113], v[182:185], v[150:153], v[110:113]
	.long	0xD3B5006A, 0x05AB3DB6	; v_mfma_f32_16x16x32_bf16 v[106:109], v[182:185], v[158:161], v[106:109]
	.long	0xD3B50066, 0x059B2DBE	; v_mfma_f32_16x16x32_bf16 v[102:105], v[190:193], v[150:153], v[102:105]
	.long	0xD3B50062, 0x058B3DBE	; v_mfma_f32_16x16x32_bf16 v[98:101], v[190:193], v[158:161], v[98:101]
	.long	0xBF8F0000	; s_setprio 0
	.long	0xBF8A0000	; s_barrier
	.long	0xD9FE0000, 0xC200008C	; ds_read_b128 v[194:197], v140
	.long	0x80130025	; s_add_u32 s19, s37, s0
	.long	0xD9FE0400, 0xC600008C	; ds_read_b128 v[198:201], v140 offset:1024
	.long	0x802AFF13, 0x00000080	; s_add_u32 s42, s19, 0x80
	.long	0xD9FE0800, 0xCA00008C	; ds_read_b128 v[202:205], v140 offset:2048
	.long	0x8E2A812A	; s_lshl_b32 s42, s42, 1
	.long	0xBEAB0023	; s_mov_b32 s43, s35
	.long	0xD9FE0C00, 0xCE00008C	; ds_read_b128 v[206:209], v140 offset:3072
	.long	0xBEAC002B	; s_mov_b32 s44, s43
	.long	0xB72B2000	; s_addk_i32 s43, 0x2000
	.long	0x81118211	; s_add_i32 s17, s17, 2
	.long	0xBEFC002C	; s_mov_b32 m0, s44
	.long	0xE05D1000, 0x2A020088	; buffer_load_dwordx4 v136, s[8:11], s42 offen lds
	.long	0xBF800000	; s_nop 0
	.long	0xBEFC002B	; s_mov_b32 m0, s43
	.long	0xE05D1000, 0x2A020089	; buffer_load_dwordx4 v137, s[8:11], s42 offen lds
	.long	0xBF8A0000	; s_barrier
	.long	0xBF8CC07F	; s_waitcnt lgkmcnt(0)
	.long	0xBF8F0001	; s_setprio 1
	.long	0xD3B5005E, 0x057B85A2	; v_mfma_f32_16x16x32_bf16 v[94:97], v[162:165], v[194:197], v[94:97]
	.long	0xD3B5005A, 0x056B95A2	; v_mfma_f32_16x16x32_bf16 v[90:93], v[162:165], v[202:205], v[90:93]
	.long	0xD3B50056, 0x055B85AA	; v_mfma_f32_16x16x32_bf16 v[86:89], v[170:173], v[194:197], v[86:89]
	.long	0xD3B50052, 0x054B95AA	; v_mfma_f32_16x16x32_bf16 v[82:85], v[170:173], v[202:205], v[82:85]
	.long	0xD3B5004E, 0x053B85B2	; v_mfma_f32_16x16x32_bf16 v[78:81], v[178:181], v[194:197], v[78:81]
	.long	0xD3B5004A, 0x052B95B2	; v_mfma_f32_16x16x32_bf16 v[74:77], v[178:181], v[202:205], v[74:77]
	.long	0xD3B50046, 0x051B85BA	; v_mfma_f32_16x16x32_bf16 v[70:73], v[186:189], v[194:197], v[70:73]
	.long	0xD3B50042, 0x050B95BA	; v_mfma_f32_16x16x32_bf16 v[66:69], v[186:189], v[202:205], v[66:69]
	.long	0xD3B5005E, 0x057B8DA6	; v_mfma_f32_16x16x32_bf16 v[94:97], v[166:169], v[198:201], v[94:97]
	.long	0xD3B5005A, 0x056B9DA6	; v_mfma_f32_16x16x32_bf16 v[90:93], v[166:169], v[206:209], v[90:93]
	.long	0xD3B50056, 0x055B8DAE	; v_mfma_f32_16x16x32_bf16 v[86:89], v[174:177], v[198:201], v[86:89]
	.long	0xD3B50052, 0x054B9DAE	; v_mfma_f32_16x16x32_bf16 v[82:85], v[174:177], v[206:209], v[82:85]
	.long	0xD3B5004E, 0x053B8DB6	; v_mfma_f32_16x16x32_bf16 v[78:81], v[182:185], v[198:201], v[78:81]
	.long	0xD3B5004A, 0x052B9DB6	; v_mfma_f32_16x16x32_bf16 v[74:77], v[182:185], v[206:209], v[74:77]
	.long	0xD3B50046, 0x051B8DBE	; v_mfma_f32_16x16x32_bf16 v[70:73], v[190:193], v[198:201], v[70:73]
	.long	0xD3B50042, 0x050B9DBE	; v_mfma_f32_16x16x32_bf16 v[66:69], v[190:193], v[206:209], v[66:69]
	.long	0xBF8F0000	; s_setprio 0
	.long	0xBF8A0000	; s_barrier
	.long	0xD9FE0000, 0xA200008D	; ds_read_b128 v[162:165], v141
	.long	0xD9FE0400, 0xA600008D	; ds_read_b128 v[166:169], v141 offset:1024
	.long	0xD9FE0800, 0xAA00008D	; ds_read_b128 v[170:173], v141 offset:2048
	.long	0xD9FE0C00, 0xAE00008D	; ds_read_b128 v[174:177], v141 offset:3072
	.long	0xD9FE1000, 0xB200008D	; ds_read_b128 v[178:181], v141 offset:4096
	.long	0x802A0026	; s_add_u32 s42, s38, s0
	.long	0xD9FE1400, 0xB600008D	; ds_read_b128 v[182:185], v141 offset:5120
	.long	0x802BFF2A, 0x00000080	; s_add_u32 s43, s42, 0x80
	.long	0xD9FE1800, 0xBA00008D	; ds_read_b128 v[186:189], v141 offset:6144
	.long	0x8E2B812B	; s_lshl_b32 s43, s43, 1
	.long	0xBEAC0021	; s_mov_b32 s44, s33
	.long	0xD9FE1C00, 0xBE00008D	; ds_read_b128 v[190:193], v141 offset:7168
	.long	0xBEAD002C	; s_mov_b32 s45, s44
	.long	0xB72C2000	; s_addk_i32 s44, 0x2000
	.long	0xBF800000	; s_nop 0
	.long	0xBEFC002D	; s_mov_b32 m0, s45
	.long	0xE05D1000, 0x2B010085	; buffer_load_dwordx4 v133, s[4:7], s43 offen lds
	.long	0xBF800000	; s_nop 0
	.long	0xBEFC002C	; s_mov_b32 m0, s44
	.long	0xE05D1000, 0x2B010084	; buffer_load_dwordx4 v132, s[4:7], s43 offen lds
	.long	0xBF8A0000	; s_barrier
	.long	0xBF8CC07F	; s_waitcnt lgkmcnt(0)
	.long	0xBF8F0001	; s_setprio 1
	.long	0xD3B5003E, 0x04FB25A2	; v_mfma_f32_16x16x32_bf16 v[62:65], v[162:165], v[146:149], v[62:65]
	.long	0xD3B5003A, 0x04EB35A2	; v_mfma_f32_16x16x32_bf16 v[58:61], v[162:165], v[154:157], v[58:61]
	.long	0xD3B50036, 0x04DB25AA	; v_mfma_f32_16x16x32_bf16 v[54:57], v[170:173], v[146:149], v[54:57]
	.long	0xD3B50032, 0x04CB35AA	; v_mfma_f32_16x16x32_bf16 v[50:53], v[170:173], v[154:157], v[50:53]
	.long	0xD3B5002E, 0x04BB25B2	; v_mfma_f32_16x16x32_bf16 v[46:49], v[178:181], v[146:149], v[46:49]
	.long	0xD3B5002A, 0x04AB35B2	; v_mfma_f32_16x16x32_bf16 v[42:45], v[178:181], v[154:157], v[42:45]
	.long	0xD3B50026, 0x049B25BA	; v_mfma_f32_16x16x32_bf16 v[38:41], v[186:189], v[146:149], v[38:41]
	.long	0xD3B50022, 0x048B35BA	; v_mfma_f32_16x16x32_bf16 v[34:37], v[186:189], v[154:157], v[34:37]
	.long	0xD3B5003E, 0x04FB2DA6	; v_mfma_f32_16x16x32_bf16 v[62:65], v[166:169], v[150:153], v[62:65]
	.long	0xD3B5003A, 0x04EB3DA6	; v_mfma_f32_16x16x32_bf16 v[58:61], v[166:169], v[158:161], v[58:61]
	.long	0xD3B50036, 0x04DB2DAE	; v_mfma_f32_16x16x32_bf16 v[54:57], v[174:177], v[150:153], v[54:57]
	.long	0xD3B50032, 0x04CB3DAE	; v_mfma_f32_16x16x32_bf16 v[50:53], v[174:177], v[158:161], v[50:53]
	.long	0xD3B5002E, 0x04BB2DB6	; v_mfma_f32_16x16x32_bf16 v[46:49], v[182:185], v[150:153], v[46:49]
	.long	0xD3B5002A, 0x04AB3DB6	; v_mfma_f32_16x16x32_bf16 v[42:45], v[182:185], v[158:161], v[42:45]
	.long	0xD3B50026, 0x049B2DBE	; v_mfma_f32_16x16x32_bf16 v[38:41], v[190:193], v[150:153], v[38:41]
	.long	0xD3B50022, 0x048B3DBE	; v_mfma_f32_16x16x32_bf16 v[34:37], v[190:193], v[158:161], v[34:37]
	.long	0xBF8F0000	; s_setprio 0
	.long	0xBF8A0000	; s_barrier
	.long	0xD9FE0000, 0x9200008E	; ds_read_b128 v[146:149], v142
	.long	0x802B0010	; s_add_u32 s43, s16, s0
	.long	0xD9FE0400, 0x9600008E	; ds_read_b128 v[150:153], v142 offset:1024
	.long	0x802CFF2B, 0x00000080	; s_add_u32 s44, s43, 0x80
	.long	0xD9FE0800, 0x9A00008E	; ds_read_b128 v[154:157], v142 offset:2048
	.long	0x8E2C812C	; s_lshl_b32 s44, s44, 1
	.long	0xBEAD0024	; s_mov_b32 s45, s36
	.long	0xD9FE0C00, 0x9E00008E	; ds_read_b128 v[158:161], v142 offset:3072
	.long	0xBEAE002D	; s_mov_b32 s46, s45
	.long	0xB72D2000	; s_addk_i32 s45, 0x2000
	.long	0xBF800000	; s_nop 0
	.long	0xBEFC002E	; s_mov_b32 m0, s46
	.long	0xE05D1000, 0x2C020088	; buffer_load_dwordx4 v136, s[8:11], s44 offen lds
	.long	0xBF800000	; s_nop 0
	.long	0xBEFC002D	; s_mov_b32 m0, s45
	.long	0xE05D1000, 0x2C020089	; buffer_load_dwordx4 v137, s[8:11], s44 offen lds
	.long	0xBF8C0F76	; s_waitcnt vmcnt(6)
	.long	0xBF8A0000	; s_barrier
	.long	0xBF8F0001	; s_setprio 1
	.long	0xD3B5001E, 0x047B85A2	; v_mfma_f32_16x16x32_bf16 v[30:33], v[162:165], v[194:197], v[30:33]
	.long	0xD3B5001A, 0x046B95A2	; v_mfma_f32_16x16x32_bf16 v[26:29], v[162:165], v[202:205], v[26:29]
	.long	0xD3B50016, 0x045B85AA	; v_mfma_f32_16x16x32_bf16 v[22:25], v[170:173], v[194:197], v[22:25]
	.long	0xD3B50012, 0x044B95AA	; v_mfma_f32_16x16x32_bf16 v[18:21], v[170:173], v[202:205], v[18:21]
	.long	0xD3B5000E, 0x043B85B2	; v_mfma_f32_16x16x32_bf16 v[14:17], v[178:181], v[194:197], v[14:17]
	.long	0xD3B5000A, 0x042B95B2	; v_mfma_f32_16x16x32_bf16 v[10:13], v[178:181], v[202:205], v[10:13]
	.long	0xD3B50006, 0x041B85BA	; v_mfma_f32_16x16x32_bf16 v[6:9], v[186:189], v[194:197], v[6:9]
	.long	0xD3B50002, 0x040B95BA	; v_mfma_f32_16x16x32_bf16 v[2:5], v[186:189], v[202:205], v[2:5]
	.long	0xD3B5001E, 0x047B8DA6	; v_mfma_f32_16x16x32_bf16 v[30:33], v[166:169], v[198:201], v[30:33]
	.long	0xD3B5001A, 0x046B9DA6	; v_mfma_f32_16x16x32_bf16 v[26:29], v[166:169], v[206:209], v[26:29]
	.long	0xD3B50016, 0x045B8DAE	; v_mfma_f32_16x16x32_bf16 v[22:25], v[174:177], v[198:201], v[22:25]
	.long	0xD3B50012, 0x044B9DAE	; v_mfma_f32_16x16x32_bf16 v[18:21], v[174:177], v[206:209], v[18:21]
	.long	0xD3B5000E, 0x043B8DB6	; v_mfma_f32_16x16x32_bf16 v[14:17], v[182:185], v[198:201], v[14:17]
	.long	0xD3B5000A, 0x042B9DB6	; v_mfma_f32_16x16x32_bf16 v[10:13], v[182:185], v[206:209], v[10:13]
	.long	0xD3B50006, 0x041B8DBE	; v_mfma_f32_16x16x32_bf16 v[6:9], v[190:193], v[198:201], v[6:9]
	.long	0xD3B50002, 0x040B9DBE	; v_mfma_f32_16x16x32_bf16 v[2:5], v[190:193], v[206:209], v[2:5]
	.long	0xBF8F0000	; s_setprio 0
	.long	0xBF8A0000	; s_barrier
	.long	0xD9FE0000, 0xA200008F	; ds_read_b128 v[162:165], v143
	.long	0xD9FE0400, 0xA600008F	; ds_read_b128 v[166:169], v143 offset:1024
	.long	0xD9FE0800, 0xAA00008F	; ds_read_b128 v[170:173], v143 offset:2048
	.long	0xD9FE0C00, 0xAE00008F	; ds_read_b128 v[174:177], v143 offset:3072
	.long	0xD9FE1000, 0xB200008F	; ds_read_b128 v[178:181], v143 offset:4096
	.long	0xD9FE1400, 0xB600008F	; ds_read_b128 v[182:185], v143 offset:5120
	.long	0x8012FF12, 0x00000080	; s_add_u32 s18, s18, 0x80
	.long	0xD9FE1800, 0xBA00008F	; ds_read_b128 v[186:189], v143 offset:6144
	.long	0x8E128112	; s_lshl_b32 s18, s18, 1
	.long	0xBEAC0022	; s_mov_b32 s44, s34
	.long	0xD9FE1C00, 0xBE00008F	; ds_read_b128 v[190:193], v143 offset:7168
	.long	0xBEAD002C	; s_mov_b32 s45, s44
	.long	0xB72C2000	; s_addk_i32 s44, 0x2000
	.long	0xBF800000	; s_nop 0
	.long	0xBEFC002D	; s_mov_b32 m0, s45
	.long	0xE05D1000, 0x12010085	; buffer_load_dwordx4 v133, s[4:7], s18 offen lds
	.long	0xBF800000	; s_nop 0
	.long	0xBEFC002C	; s_mov_b32 m0, s44
	.long	0xE05D1000, 0x12010084	; buffer_load_dwordx4 v132, s[4:7], s18 offen lds
	.long	0xBF8CC87F	; s_waitcnt lgkmcnt(8)
	.long	0xBF8A0000	; s_barrier
	.long	0xBF8CC07F	; s_waitcnt lgkmcnt(0)
	.long	0xBF8F0001	; s_setprio 1
	.long	0xD3B5007E, 0x05FB25A2	; v_mfma_f32_16x16x32_bf16 v[126:129], v[162:165], v[146:149], v[126:129]
	.long	0xD3B5007A, 0x05EB35A2	; v_mfma_f32_16x16x32_bf16 v[122:125], v[162:165], v[154:157], v[122:125]
	.long	0xD3B50076, 0x05DB25AA	; v_mfma_f32_16x16x32_bf16 v[118:121], v[170:173], v[146:149], v[118:121]
	.long	0xD3B50072, 0x05CB35AA	; v_mfma_f32_16x16x32_bf16 v[114:117], v[170:173], v[154:157], v[114:117]
	.long	0xD3B5006E, 0x05BB25B2	; v_mfma_f32_16x16x32_bf16 v[110:113], v[178:181], v[146:149], v[110:113]
	.long	0xD3B5006A, 0x05AB35B2	; v_mfma_f32_16x16x32_bf16 v[106:109], v[178:181], v[154:157], v[106:109]
	.long	0xD3B50066, 0x059B25BA	; v_mfma_f32_16x16x32_bf16 v[102:105], v[186:189], v[146:149], v[102:105]
	.long	0xD3B50062, 0x058B35BA	; v_mfma_f32_16x16x32_bf16 v[98:101], v[186:189], v[154:157], v[98:101]
	.long	0xD3B5007E, 0x05FB2DA6	; v_mfma_f32_16x16x32_bf16 v[126:129], v[166:169], v[150:153], v[126:129]
	.long	0xD3B5007A, 0x05EB3DA6	; v_mfma_f32_16x16x32_bf16 v[122:125], v[166:169], v[158:161], v[122:125]
	.long	0xD3B50076, 0x05DB2DAE	; v_mfma_f32_16x16x32_bf16 v[118:121], v[174:177], v[150:153], v[118:121]
	.long	0xD3B50072, 0x05CB3DAE	; v_mfma_f32_16x16x32_bf16 v[114:117], v[174:177], v[158:161], v[114:117]
	.long	0xD3B5006E, 0x05BB2DB6	; v_mfma_f32_16x16x32_bf16 v[110:113], v[182:185], v[150:153], v[110:113]
	.long	0xD3B5006A, 0x05AB3DB6	; v_mfma_f32_16x16x32_bf16 v[106:109], v[182:185], v[158:161], v[106:109]
	.long	0xD3B50066, 0x059B2DBE	; v_mfma_f32_16x16x32_bf16 v[102:105], v[190:193], v[150:153], v[102:105]
	.long	0xD3B50062, 0x058B3DBE	; v_mfma_f32_16x16x32_bf16 v[98:101], v[190:193], v[158:161], v[98:101]
	.long	0xBF8F0000	; s_setprio 0
	.long	0xBF8A0000	; s_barrier
	.long	0xD9FE0000, 0xC2000090	; ds_read_b128 v[194:197], v144
	.long	0xD9FE0400, 0xC6000090	; ds_read_b128 v[198:201], v144 offset:1024
	.long	0x8012FF13, 0x000000C0	; s_add_u32 s18, s19, 0xc0
	.long	0xD9FE0800, 0xCA000090	; ds_read_b128 v[202:205], v144 offset:2048
	.long	0x8E128112	; s_lshl_b32 s18, s18, 1
	.long	0xBE930028	; s_mov_b32 s19, s40
	.long	0xD9FE0C00, 0xCE000090	; ds_read_b128 v[206:209], v144 offset:3072
	.long	0xBEAC0013	; s_mov_b32 s44, s19
	.long	0xB7132000	; s_addk_i32 s19, 0x2000
	.long	0xBF800000	; s_nop 0
	.long	0xBEFC002C	; s_mov_b32 m0, s44
	.long	0xE05D1000, 0x12020088	; buffer_load_dwordx4 v136, s[8:11], s18 offen lds
	.long	0xBF800000	; s_nop 0
	.long	0xBEFC0013	; s_mov_b32 m0, s19
	.long	0xE05D1000, 0x12020089	; buffer_load_dwordx4 v137, s[8:11], s18 offen lds
	.long	0xBF8A0000	; s_barrier
	.long	0xBF8CC07F	; s_waitcnt lgkmcnt(0)
	.long	0xBF8F0001	; s_setprio 1
	.long	0xD3B5005E, 0x057B85A2	; v_mfma_f32_16x16x32_bf16 v[94:97], v[162:165], v[194:197], v[94:97]
	.long	0xD3B5005A, 0x056B95A2	; v_mfma_f32_16x16x32_bf16 v[90:93], v[162:165], v[202:205], v[90:93]
	.long	0xD3B50056, 0x055B85AA	; v_mfma_f32_16x16x32_bf16 v[86:89], v[170:173], v[194:197], v[86:89]
	.long	0xD3B50052, 0x054B95AA	; v_mfma_f32_16x16x32_bf16 v[82:85], v[170:173], v[202:205], v[82:85]
	.long	0xD3B5004E, 0x053B85B2	; v_mfma_f32_16x16x32_bf16 v[78:81], v[178:181], v[194:197], v[78:81]
	.long	0xD3B5004A, 0x052B95B2	; v_mfma_f32_16x16x32_bf16 v[74:77], v[178:181], v[202:205], v[74:77]
	.long	0xD3B50046, 0x051B85BA	; v_mfma_f32_16x16x32_bf16 v[70:73], v[186:189], v[194:197], v[70:73]
	.long	0xD3B50042, 0x050B95BA	; v_mfma_f32_16x16x32_bf16 v[66:69], v[186:189], v[202:205], v[66:69]
	.long	0xD3B5005E, 0x057B8DA6	; v_mfma_f32_16x16x32_bf16 v[94:97], v[166:169], v[198:201], v[94:97]
	.long	0xD3B5005A, 0x056B9DA6	; v_mfma_f32_16x16x32_bf16 v[90:93], v[166:169], v[206:209], v[90:93]
	.long	0xD3B50056, 0x055B8DAE	; v_mfma_f32_16x16x32_bf16 v[86:89], v[174:177], v[198:201], v[86:89]
	.long	0xD3B50052, 0x054B9DAE	; v_mfma_f32_16x16x32_bf16 v[82:85], v[174:177], v[206:209], v[82:85]
	.long	0xD3B5004E, 0x053B8DB6	; v_mfma_f32_16x16x32_bf16 v[78:81], v[182:185], v[198:201], v[78:81]
	.long	0xD3B5004A, 0x052B9DB6	; v_mfma_f32_16x16x32_bf16 v[74:77], v[182:185], v[206:209], v[74:77]
	.long	0xD3B50046, 0x051B8DBE	; v_mfma_f32_16x16x32_bf16 v[70:73], v[190:193], v[198:201], v[70:73]
	.long	0xD3B50042, 0x050B9DBE	; v_mfma_f32_16x16x32_bf16 v[66:69], v[190:193], v[206:209], v[66:69]
	.long	0xBF8F0000	; s_setprio 0
	.long	0xBF8A0000	; s_barrier
	.long	0xD9FE0000, 0xA2000091	; ds_read_b128 v[162:165], v145
	.long	0xD9FE0400, 0xA6000091	; ds_read_b128 v[166:169], v145 offset:1024
	.long	0xD9FE0800, 0xAA000091	; ds_read_b128 v[170:173], v145 offset:2048
	.long	0xD9FE0C00, 0xAE000091	; ds_read_b128 v[174:177], v145 offset:3072
	.long	0xD9FE1000, 0xB2000091	; ds_read_b128 v[178:181], v145 offset:4096
	.long	0xD9FE1400, 0xB6000091	; ds_read_b128 v[182:185], v145 offset:5120
	.long	0x8012FF2A, 0x000000C0	; s_add_u32 s18, s42, 0xc0
	.long	0xD9FE1800, 0xBA000091	; ds_read_b128 v[186:189], v145 offset:6144
	.long	0x8E128112	; s_lshl_b32 s18, s18, 1
	.long	0xBE930027	; s_mov_b32 s19, s39
	.long	0xD9FE1C00, 0xBE000091	; ds_read_b128 v[190:193], v145 offset:7168
	.long	0xBEAA0013	; s_mov_b32 s42, s19
	.long	0xB7132000	; s_addk_i32 s19, 0x2000
	.long	0xBF800000	; s_nop 0
	.long	0xBEFC002A	; s_mov_b32 m0, s42
	.long	0xE05D1000, 0x12010085	; buffer_load_dwordx4 v133, s[4:7], s18 offen lds
	.long	0xBF800000	; s_nop 0
	.long	0xBEFC0013	; s_mov_b32 m0, s19
	.long	0xE05D1000, 0x12010084	; buffer_load_dwordx4 v132, s[4:7], s18 offen lds
	.long	0xBF8A0000	; s_barrier
	.long	0xBF8CC07F	; s_waitcnt lgkmcnt(0)
	.long	0xBF8F0001	; s_setprio 1
	.long	0xD3B5003E, 0x04FB25A2	; v_mfma_f32_16x16x32_bf16 v[62:65], v[162:165], v[146:149], v[62:65]
	.long	0xD3B5003A, 0x04EB35A2	; v_mfma_f32_16x16x32_bf16 v[58:61], v[162:165], v[154:157], v[58:61]
	.long	0xD3B50036, 0x04DB25AA	; v_mfma_f32_16x16x32_bf16 v[54:57], v[170:173], v[146:149], v[54:57]
	.long	0xD3B50032, 0x04CB35AA	; v_mfma_f32_16x16x32_bf16 v[50:53], v[170:173], v[154:157], v[50:53]
	.long	0xD3B5002E, 0x04BB25B2	; v_mfma_f32_16x16x32_bf16 v[46:49], v[178:181], v[146:149], v[46:49]
	.long	0xD3B5002A, 0x04AB35B2	; v_mfma_f32_16x16x32_bf16 v[42:45], v[178:181], v[154:157], v[42:45]
	.long	0xD3B50026, 0x049B25BA	; v_mfma_f32_16x16x32_bf16 v[38:41], v[186:189], v[146:149], v[38:41]
	.long	0xD3B50022, 0x048B35BA	; v_mfma_f32_16x16x32_bf16 v[34:37], v[186:189], v[154:157], v[34:37]
	.long	0xD3B5003E, 0x04FB2DA6	; v_mfma_f32_16x16x32_bf16 v[62:65], v[166:169], v[150:153], v[62:65]
	.long	0xD3B5003A, 0x04EB3DA6	; v_mfma_f32_16x16x32_bf16 v[58:61], v[166:169], v[158:161], v[58:61]
	.long	0xD3B50036, 0x04DB2DAE	; v_mfma_f32_16x16x32_bf16 v[54:57], v[174:177], v[150:153], v[54:57]
	.long	0xD3B50032, 0x04CB3DAE	; v_mfma_f32_16x16x32_bf16 v[50:53], v[174:177], v[158:161], v[50:53]
	.long	0xD3B5002E, 0x04BB2DB6	; v_mfma_f32_16x16x32_bf16 v[46:49], v[182:185], v[150:153], v[46:49]
	.long	0xD3B5002A, 0x04AB3DB6	; v_mfma_f32_16x16x32_bf16 v[42:45], v[182:185], v[158:161], v[42:45]
	.long	0xD3B50026, 0x049B2DBE	; v_mfma_f32_16x16x32_bf16 v[38:41], v[190:193], v[150:153], v[38:41]
	.long	0xD3B50022, 0x048B3DBE	; v_mfma_f32_16x16x32_bf16 v[34:37], v[190:193], v[158:161], v[34:37]
	.long	0xBF8F0000	; s_setprio 0
	.long	0xBF8A0000	; s_barrier
	.long	0x8012FF2B, 0x000000C0	; s_add_u32 s18, s43, 0xc0
	.long	0x8E128112	; s_lshl_b32 s18, s18, 1
	.long	0xBE930029	; s_mov_b32 s19, s41
	.long	0xBEAA0013	; s_mov_b32 s42, s19
	.long	0xB7132000	; s_addk_i32 s19, 0x2000
	.long	0xBF800000	; s_nop 0
	.long	0xBEFC002A	; s_mov_b32 m0, s42
	.long	0xE05D1000, 0x12020088	; buffer_load_dwordx4 v136, s[8:11], s18 offen lds
	.long	0xBF800000	; s_nop 0
	.long	0xBEFC0013	; s_mov_b32 m0, s19
	.long	0xE05D1000, 0x12020089	; buffer_load_dwordx4 v137, s[8:11], s18 offen lds
	.long	0xBF8C0F76	; s_waitcnt vmcnt(6)
	.long	0xBF8A0000	; s_barrier
	.long	0xBF8F0001	; s_setprio 1
	.long	0xD3B5001E, 0x047B85A2	; v_mfma_f32_16x16x32_bf16 v[30:33], v[162:165], v[194:197], v[30:33]
	.long	0xD3B5001A, 0x046B95A2	; v_mfma_f32_16x16x32_bf16 v[26:29], v[162:165], v[202:205], v[26:29]
	.long	0xD3B50016, 0x045B85AA	; v_mfma_f32_16x16x32_bf16 v[22:25], v[170:173], v[194:197], v[22:25]
	.long	0xD3B50012, 0x044B95AA	; v_mfma_f32_16x16x32_bf16 v[18:21], v[170:173], v[202:205], v[18:21]
	.long	0xD3B5000E, 0x043B85B2	; v_mfma_f32_16x16x32_bf16 v[14:17], v[178:181], v[194:197], v[14:17]
	.long	0xD3B5000A, 0x042B95B2	; v_mfma_f32_16x16x32_bf16 v[10:13], v[178:181], v[202:205], v[10:13]
	.long	0xD3B50006, 0x041B85BA	; v_mfma_f32_16x16x32_bf16 v[6:9], v[186:189], v[194:197], v[6:9]
	.long	0xD3B50002, 0x040B95BA	; v_mfma_f32_16x16x32_bf16 v[2:5], v[186:189], v[202:205], v[2:5]
	.long	0xD3B5001E, 0x047B8DA6	; v_mfma_f32_16x16x32_bf16 v[30:33], v[166:169], v[198:201], v[30:33]
	.long	0xD3B5001A, 0x046B9DA6	; v_mfma_f32_16x16x32_bf16 v[26:29], v[166:169], v[206:209], v[26:29]
	.long	0xD3B50016, 0x045B8DAE	; v_mfma_f32_16x16x32_bf16 v[22:25], v[174:177], v[198:201], v[22:25]
	.long	0xD3B50012, 0x044B9DAE	; v_mfma_f32_16x16x32_bf16 v[18:21], v[174:177], v[206:209], v[18:21]
	.long	0xD3B5000E, 0x043B8DB6	; v_mfma_f32_16x16x32_bf16 v[14:17], v[182:185], v[198:201], v[14:17]
	.long	0xD3B5000A, 0x042B9DB6	; v_mfma_f32_16x16x32_bf16 v[10:13], v[182:185], v[206:209], v[10:13]
	.long	0xD3B50006, 0x041B8DBE	; v_mfma_f32_16x16x32_bf16 v[6:9], v[190:193], v[198:201], v[6:9]
	.long	0xD3B50002, 0x040B9DBE	; v_mfma_f32_16x16x32_bf16 v[2:5], v[190:193], v[206:209], v[2:5]
	.long	0xBF8F0000	; s_setprio 0
	.long	0x8000FF00, 0x00000080	; s_add_u32 s0, s0, 0x80
	.long	0x82018001	; s_addc_u32 s1, s1, 0
	.long	0xBF030E11	; s_cmp_ge_i32 s17, s14
	.long	0xBF8A0000	; s_barrier
	.long	0xBF84FDFE	; s_cbranch_scc0 65022
	.long	0xD1FF0089, 0x061B0E17	; v_add3_u32 v137, s23, v135, v134
	.long	0xD9FE0000, 0x8A000089	; ds_read_b128 v[138:141], v137
	.long	0xD9FE0400, 0x8E000089	; ds_read_b128 v[142:145], v137 offset:1024
	.long	0xD9FE0800, 0x92000089	; ds_read_b128 v[146:149], v137 offset:2048
	.long	0xD9FE0C00, 0x96000089	; ds_read_b128 v[150:153], v137 offset:3072
	.long	0x25050481	; v_lshlrev_b32_e32 v130, 1, v130
	.long	0xD1FF0089, 0x061B0416	; v_add3_u32 v137, s22, v130, v134
	.long	0xD9FE0000, 0x9A000089	; ds_read_b128 v[154:157], v137
	.long	0xD9FE0400, 0x9E000089	; ds_read_b128 v[158:161], v137 offset:1024
	.long	0xD9FE0800, 0xA2000089	; ds_read_b128 v[162:165], v137 offset:2048
	.long	0xD9FE0C00, 0xA6000089	; ds_read_b128 v[166:169], v137 offset:3072
	.long	0x8E00860D	; s_lshl_b32 s0, s13, 6
	.long	0xD9FE1000, 0xAA000089	; ds_read_b128 v[170:173], v137 offset:4096
	.long	0x8180C000	; s_sub_i32 s0, s0, 64
	.long	0xD9FE1400, 0xAE000089	; ds_read_b128 v[174:177], v137 offset:5120
	.long	0x8000001E	; s_add_u32 s0, s30, s0
	.long	0xD9FE1800, 0xB2000089	; ds_read_b128 v[178:181], v137 offset:6144
	.long	0x8E008100	; s_lshl_b32 s0, s0, 1
	.long	0xD9FE1C00, 0xB6000089	; ds_read_b128 v[182:185], v137 offset:7168
	.long	0xBE81001F	; s_mov_b32 s1, s31
	.long	0xBEFC0080	; s_mov_b32 m0, 0
	.long	0xBEFC0001	; s_mov_b32 m0, s1
	.long	0x8101FF1F, 0x00002000	; s_add_i32 s1, s31, 0x2000
	.long	0xE05D1000, 0x00010085	; buffer_load_dwordx4 v133, s[4:7], s0 offen lds
	.long	0x21100082	; v_lshrrev_b32_e32 v136, 2, v0
	.long	0xBEFC0001	; s_mov_b32 m0, s1
	.long	0xE05D1000, 0x00010084	; buffer_load_dwordx4 v132, s[4:7], s0 offen lds
	.long	0xBF8A0000	; s_barrier
	.long	0xBF8CC07F	; s_waitcnt lgkmcnt(0)
	.long	0xBF8F0001	; s_setprio 1
	.long	0xD3B5007E, 0x05FB159A	; v_mfma_f32_16x16x32_bf16 v[126:129], v[154:157], v[138:141], v[126:129]
	.long	0xD3B5007A, 0x05EB259A	; v_mfma_f32_16x16x32_bf16 v[122:125], v[154:157], v[146:149], v[122:125]
	.long	0xD3B50076, 0x05DB15A2	; v_mfma_f32_16x16x32_bf16 v[118:121], v[162:165], v[138:141], v[118:121]
	.long	0xD3B50072, 0x05CB25A2	; v_mfma_f32_16x16x32_bf16 v[114:117], v[162:165], v[146:149], v[114:117]
	.long	0xD3B5006E, 0x05BB15AA	; v_mfma_f32_16x16x32_bf16 v[110:113], v[170:173], v[138:141], v[110:113]
	.long	0xD3B5006A, 0x05AB25AA	; v_mfma_f32_16x16x32_bf16 v[106:109], v[170:173], v[146:149], v[106:109]
	.long	0xD3B50066, 0x059B15B2	; v_mfma_f32_16x16x32_bf16 v[102:105], v[178:181], v[138:141], v[102:105]
	.long	0xD3B50062, 0x058B25B2	; v_mfma_f32_16x16x32_bf16 v[98:101], v[178:181], v[146:149], v[98:101]
	.long	0xD3B5007E, 0x05FB1D9E	; v_mfma_f32_16x16x32_bf16 v[126:129], v[158:161], v[142:145], v[126:129]
	.long	0xD3B5007A, 0x05EB2D9E	; v_mfma_f32_16x16x32_bf16 v[122:125], v[158:161], v[150:153], v[122:125]
	.long	0xD3B50076, 0x05DB1DA6	; v_mfma_f32_16x16x32_bf16 v[118:121], v[166:169], v[142:145], v[118:121]
	.long	0xD3B50072, 0x05CB2DA6	; v_mfma_f32_16x16x32_bf16 v[114:117], v[166:169], v[150:153], v[114:117]
	.long	0xD3B5006E, 0x05BB1DAE	; v_mfma_f32_16x16x32_bf16 v[110:113], v[174:177], v[142:145], v[110:113]
	.long	0xD3B5006A, 0x05AB2DAE	; v_mfma_f32_16x16x32_bf16 v[106:109], v[174:177], v[150:153], v[106:109]
	.long	0xD3B50066, 0x059B1DB6	; v_mfma_f32_16x16x32_bf16 v[102:105], v[182:185], v[142:145], v[102:105]
	.long	0xD3B50062, 0x058B2DB6	; v_mfma_f32_16x16x32_bf16 v[98:101], v[182:185], v[150:153], v[98:101]
	.long	0xBF8F0000	; s_setprio 0
	.long	0xBF8A0000	; s_barrier
	.long	0xD1FF0084, 0x061B0E1B	; v_add3_u32 v132, s27, v135, v134
	.long	0xD9FE0000, 0xBA000084	; ds_read_b128 v[186:189], v132
	.long	0xD9FE0400, 0xBE000084	; ds_read_b128 v[190:193], v132 offset:1024
	.long	0xD9FE0800, 0xC2000084	; ds_read_b128 v[194:197], v132 offset:2048
	.long	0xD9FE0C00, 0xC6000084	; ds_read_b128 v[198:201], v132 offset:3072
	.long	0xBF8A0000	; s_barrier
	.long	0xBF8CC07F	; s_waitcnt lgkmcnt(0)
	.long	0xBF8F0001	; s_setprio 1
	.long	0xD3B5005E, 0x057B759A	; v_mfma_f32_16x16x32_bf16 v[94:97], v[154:157], v[186:189], v[94:97]
	.long	0xD3B5005A, 0x056B859A	; v_mfma_f32_16x16x32_bf16 v[90:93], v[154:157], v[194:197], v[90:93]
	.long	0xD3B50056, 0x055B75A2	; v_mfma_f32_16x16x32_bf16 v[86:89], v[162:165], v[186:189], v[86:89]
	.long	0xD3B5004E, 0x053B75AA	; v_mfma_f32_16x16x32_bf16 v[78:81], v[170:173], v[186:189], v[78:81]
	.long	0xD3B50046, 0x051B75B2	; v_mfma_f32_16x16x32_bf16 v[70:73], v[178:181], v[186:189], v[70:73]
	.long	0xD3B5005E, 0x057B7D9E	; v_mfma_f32_16x16x32_bf16 v[94:97], v[158:161], v[190:193], v[94:97]
	.long	0xD3B5005A, 0x056B8D9E	; v_mfma_f32_16x16x32_bf16 v[90:93], v[158:161], v[198:201], v[90:93]
	.long	0xD3B50056, 0x055B7DA6	; v_mfma_f32_16x16x32_bf16 v[86:89], v[166:169], v[190:193], v[86:89]
	.long	0xD3B50052, 0x054B85A2	; v_mfma_f32_16x16x32_bf16 v[82:85], v[162:165], v[194:197], v[82:85]
	.long	0xD3B5004E, 0x053B7DAE	; v_mfma_f32_16x16x32_bf16 v[78:81], v[174:177], v[190:193], v[78:81]
	.long	0xD3B5004A, 0x052B85AA	; v_mfma_f32_16x16x32_bf16 v[74:77], v[170:173], v[194:197], v[74:77]
	.long	0xD3B50046, 0x051B7DB6	; v_mfma_f32_16x16x32_bf16 v[70:73], v[182:185], v[190:193], v[70:73]
	.long	0xD3B50042, 0x050B85B2	; v_mfma_f32_16x16x32_bf16 v[66:69], v[178:181], v[194:197], v[66:69]
	.long	0xD3B5009A, 0x054B8DA6	; v_mfma_f32_16x16x32_bf16 v[154:157], v[166:169], v[198:201], v[82:85]
	.long	0xD3B5009E, 0x052B8DAE	; v_mfma_f32_16x16x32_bf16 v[158:161], v[174:177], v[198:201], v[74:77]
	.long	0xD3B500A2, 0x050B8DB6	; v_mfma_f32_16x16x32_bf16 v[162:165], v[182:185], v[198:201], v[66:69]
	.long	0xBF8F0000	; s_setprio 0
	.long	0xBF8A0000	; s_barrier
	.long	0xD1FF0084, 0x061B0418	; v_add3_u32 v132, s24, v130, v134
	.long	0xD9FE0000, 0x42000084	; ds_read_b128 v[66:69], v132
	.long	0xD9FE0400, 0x4A000084	; ds_read_b128 v[74:77], v132 offset:1024
	.long	0xD9FE0800, 0x52000084	; ds_read_b128 v[82:85], v132 offset:2048
	.long	0xD9FE0C00, 0xA6000084	; ds_read_b128 v[166:169], v132 offset:3072
	.long	0xD9FE1000, 0xAA000084	; ds_read_b128 v[170:173], v132 offset:4096
	.long	0xD9FE1400, 0xAE000084	; ds_read_b128 v[174:177], v132 offset:5120
	.long	0xD9FE1800, 0xB2000084	; ds_read_b128 v[178:181], v132 offset:6144
	.long	0xD9FE1C00, 0xB6000084	; ds_read_b128 v[182:185], v132 offset:7168
	.long	0xBF8C0F74	; s_waitcnt vmcnt(4)
	.long	0xBF8A0000	; s_barrier
	.long	0xBF8CC07F	; s_waitcnt lgkmcnt(0)
	.long	0xBF8F0001	; s_setprio 1
	.long	0xD3B5003E, 0x04FB1542	; v_mfma_f32_16x16x32_bf16 v[62:65], v[66:69], v[138:141], v[62:65]
	.long	0xD3B50036, 0x04DB1552	; v_mfma_f32_16x16x32_bf16 v[54:57], v[82:85], v[138:141], v[54:57]
	.long	0xD3B50032, 0x04CB2552	; v_mfma_f32_16x16x32_bf16 v[50:53], v[82:85], v[146:149], v[50:53]
	.long	0xD3B5002E, 0x04BB15AA	; v_mfma_f32_16x16x32_bf16 v[46:49], v[170:173], v[138:141], v[46:49]
	.long	0xD3B5002A, 0x04AB25AA	; v_mfma_f32_16x16x32_bf16 v[42:45], v[170:173], v[146:149], v[42:45]
	.long	0xD3B50026, 0x049B15B2	; v_mfma_f32_16x16x32_bf16 v[38:41], v[178:181], v[138:141], v[38:41]
	.long	0xD3B50022, 0x048B25B2	; v_mfma_f32_16x16x32_bf16 v[34:37], v[178:181], v[146:149], v[34:37]
	.long	0xD3B5001E, 0x047B7542	; v_mfma_f32_16x16x32_bf16 v[30:33], v[66:69], v[186:189], v[30:33]
	.long	0xD3B5001A, 0x046B8542	; v_mfma_f32_16x16x32_bf16 v[26:29], v[66:69], v[194:197], v[26:29]
	.long	0xD3B50016, 0x045B7552	; v_mfma_f32_16x16x32_bf16 v[22:25], v[82:85], v[186:189], v[22:25]
	.long	0xD3B50012, 0x044B8552	; v_mfma_f32_16x16x32_bf16 v[18:21], v[82:85], v[194:197], v[18:21]
	.long	0xD3B5000E, 0x043B75AA	; v_mfma_f32_16x16x32_bf16 v[14:17], v[170:173], v[186:189], v[14:17]
	.long	0xD3B5000A, 0x042B85AA	; v_mfma_f32_16x16x32_bf16 v[10:13], v[170:173], v[194:197], v[10:13]
	.long	0xD3B50006, 0x041B75B2	; v_mfma_f32_16x16x32_bf16 v[6:9], v[178:181], v[186:189], v[6:9]
	.long	0xD3B50002, 0x040B85B2	; v_mfma_f32_16x16x32_bf16 v[2:5], v[178:181], v[194:197], v[2:5]
	.long	0xD3B5003E, 0x04FB1D4A	; v_mfma_f32_16x16x32_bf16 v[62:65], v[74:77], v[142:145], v[62:65]
	.long	0xD3B5003A, 0x04EB2542	; v_mfma_f32_16x16x32_bf16 v[58:61], v[66:69], v[146:149], v[58:61]
	.long	0xD3B50036, 0x04DB1DA6	; v_mfma_f32_16x16x32_bf16 v[54:57], v[166:169], v[142:145], v[54:57]
	.long	0xD3B50032, 0x04CB2DA6	; v_mfma_f32_16x16x32_bf16 v[50:53], v[166:169], v[150:153], v[50:53]
	.long	0xD3B5002E, 0x04BB1DAE	; v_mfma_f32_16x16x32_bf16 v[46:49], v[174:177], v[142:145], v[46:49]
	.long	0xD3B5002A, 0x04AB2DAE	; v_mfma_f32_16x16x32_bf16 v[42:45], v[174:177], v[150:153], v[42:45]
	.long	0xD3B50026, 0x049B1DB6	; v_mfma_f32_16x16x32_bf16 v[38:41], v[182:185], v[142:145], v[38:41]
	.long	0xD3B50022, 0x048B2DB6	; v_mfma_f32_16x16x32_bf16 v[34:37], v[182:185], v[150:153], v[34:37]
	.long	0xD3B5001E, 0x047B7D4A	; v_mfma_f32_16x16x32_bf16 v[30:33], v[74:77], v[190:193], v[30:33]
	.long	0xD3B5001A, 0x046B8D4A	; v_mfma_f32_16x16x32_bf16 v[26:29], v[74:77], v[198:201], v[26:29]
	.long	0xD3B50016, 0x045B7DA6	; v_mfma_f32_16x16x32_bf16 v[22:25], v[166:169], v[190:193], v[22:25]
	.long	0xD3B50012, 0x044B8DA6	; v_mfma_f32_16x16x32_bf16 v[18:21], v[166:169], v[198:201], v[18:21]
	.long	0xD3B5000E, 0x043B7DAE	; v_mfma_f32_16x16x32_bf16 v[14:17], v[174:177], v[190:193], v[14:17]
	.long	0xD3B5000A, 0x042B8DAE	; v_mfma_f32_16x16x32_bf16 v[10:13], v[174:177], v[198:201], v[10:13]
	.long	0xD3B50006, 0x041B7DB6	; v_mfma_f32_16x16x32_bf16 v[6:9], v[182:185], v[190:193], v[6:9]
	.long	0xD3B50002, 0x040B8DB6	; v_mfma_f32_16x16x32_bf16 v[2:5], v[182:185], v[198:201], v[2:5]
	.long	0xD3B500CA, 0x04EB2D4A	; v_mfma_f32_16x16x32_bf16 v[202:205], v[74:77], v[150:153], v[58:61]
	.long	0xBF8F0000	; s_setprio 0
	.long	0xBF8A0000	; s_barrier
	.long	0xBF800000	; s_nop 0
	.long	0xD1FF003A, 0x061B0E1D	; v_add3_u32 v58, s29, v135, v134
	.long	0xD9FE0000, 0x8A00003A	; ds_read_b128 v[138:141], v58
	.long	0xD9FE0400, 0x8E00003A	; ds_read_b128 v[142:145], v58 offset:1024
	.long	0xD9FE0800, 0x9200003A	; ds_read_b128 v[146:149], v58 offset:2048
	.long	0xD9FE0C00, 0x9600003A	; ds_read_b128 v[150:153], v58 offset:3072
	.long	0xD1FF004A, 0x061B041C	; v_add3_u32 v74, s28, v130, v134
	.long	0xD9FE0000, 0x3A00004A	; ds_read_b128 v[58:61], v74
	.long	0xD9FE0400, 0x4200004A	; ds_read_b128 v[66:69], v74 offset:1024
	.long	0xD9FE0800, 0xA600004A	; ds_read_b128 v[166:169], v74 offset:2048
	.long	0xD9FE0C00, 0xAA00004A	; ds_read_b128 v[170:173], v74 offset:3072
	.long	0xD9FE1000, 0xAE00004A	; ds_read_b128 v[174:177], v74 offset:4096
	.long	0xD9FE1400, 0xB200004A	; ds_read_b128 v[178:181], v74 offset:5120
	.long	0xD9FE1800, 0xB600004A	; ds_read_b128 v[182:185], v74 offset:6144
	.long	0xD9FE1C00, 0xBA00004A	; ds_read_b128 v[186:189], v74 offset:7168
	.long	0xBF8C0F72	; s_waitcnt vmcnt(2)
	.long	0xBF8A0000	; s_barrier
	.long	0xBF8CC07F	; s_waitcnt lgkmcnt(0)
	.long	0xBF8F0001	; s_setprio 1
	.long	0xD3B5004A, 0x05FB153A	; v_mfma_f32_16x16x32_bf16 v[74:77], v[58:61], v[138:141], v[126:129]
	.long	0xD3B5007E, 0x052B1D42	; v_mfma_f32_16x16x32_bf16 v[126:129], v[66:69], v[142:145], v[74:77]
	.long	0xD3B5004A, 0x05EB253A	; v_mfma_f32_16x16x32_bf16 v[74:77], v[58:61], v[146:149], v[122:125]
	.long	0xD3B5007A, 0x052B2D42	; v_mfma_f32_16x16x32_bf16 v[122:125], v[66:69], v[150:153], v[74:77]
	.long	0xD3B5004A, 0x05DB15A6	; v_mfma_f32_16x16x32_bf16 v[74:77], v[166:169], v[138:141], v[118:121]
	.long	0xD3B50076, 0x052B1DAA	; v_mfma_f32_16x16x32_bf16 v[118:121], v[170:173], v[142:145], v[74:77]
	.long	0xD3B5004A, 0x05CB25A6	; v_mfma_f32_16x16x32_bf16 v[74:77], v[166:169], v[146:149], v[114:117]
	.long	0xD3B50072, 0x052B2DAA	; v_mfma_f32_16x16x32_bf16 v[114:117], v[170:173], v[150:153], v[74:77]
	.long	0xD3B5004A, 0x05BB15AE	; v_mfma_f32_16x16x32_bf16 v[74:77], v[174:177], v[138:141], v[110:113]
	.long	0xD3B5006E, 0x052B1DB2	; v_mfma_f32_16x16x32_bf16 v[110:113], v[178:181], v[142:145], v[74:77]
	.long	0xD3B5004A, 0x05AB25AE	; v_mfma_f32_16x16x32_bf16 v[74:77], v[174:177], v[146:149], v[106:109]
	.long	0xD3B5006A, 0x052B2DB2	; v_mfma_f32_16x16x32_bf16 v[106:109], v[178:181], v[150:153], v[74:77]
	.long	0xD3B5004A, 0x059B15B6	; v_mfma_f32_16x16x32_bf16 v[74:77], v[182:185], v[138:141], v[102:105]
	.long	0xD3B50052, 0x052B1DBA	; v_mfma_f32_16x16x32_bf16 v[82:85], v[186:189], v[142:145], v[74:77]
	.long	0xD3B5004A, 0x058B25B6	; v_mfma_f32_16x16x32_bf16 v[74:77], v[182:185], v[146:149], v[98:101]
	.long	0xD3B5004A, 0x052B2DBA	; v_mfma_f32_16x16x32_bf16 v[74:77], v[186:189], v[150:153], v[74:77]
	.long	0xBF8F0000	; s_setprio 0
	.long	0xBF8A0000	; s_barrier
	.long	0xD1FF0062, 0x061B0E1A	; v_add3_u32 v98, s26, v135, v134
	.long	0xD9FE0000, 0xBE000062	; ds_read_b128 v[190:193], v98
	.long	0xD9FE0400, 0xC2000062	; ds_read_b128 v[194:197], v98 offset:1024
	.long	0xD9FE0800, 0xC6000062	; ds_read_b128 v[198:201], v98 offset:2048
	.long	0xD9FE0C00, 0xCE000062	; ds_read_b128 v[206:209], v98 offset:3072
	.long	0xBF8C0F70	; s_waitcnt vmcnt(0)
	.long	0xBF8A0000	; s_barrier
	.long	0xBF8CC07F	; s_waitcnt lgkmcnt(0)
	.long	0xBF8F0001	; s_setprio 1
	.long	0xD3B5005E, 0x057B7D3A	; v_mfma_f32_16x16x32_bf16 v[94:97], v[58:61], v[190:193], v[94:97]
	.long	0xD3B5003A, 0x056B8D3A	; v_mfma_f32_16x16x32_bf16 v[58:61], v[58:61], v[198:201], v[90:93]
	.long	0xD3B50062, 0x04EB9D42	; v_mfma_f32_16x16x32_bf16 v[98:101], v[66:69], v[206:209], v[58:61]
	.long	0xD3B5003A, 0x055B7DA6	; v_mfma_f32_16x16x32_bf16 v[58:61], v[166:169], v[190:193], v[86:89]
	.long	0xD3B50066, 0x057B8542	; v_mfma_f32_16x16x32_bf16 v[102:105], v[66:69], v[194:197], v[94:97]
	.long	0xD3B5005E, 0x04EB85AA	; v_mfma_f32_16x16x32_bf16 v[94:97], v[170:173], v[194:197], v[58:61]
	.long	0xD3B5003A, 0x066B8DA6	; v_mfma_f32_16x16x32_bf16 v[58:61], v[166:169], v[198:201], v[154:157]
	.long	0xD3B5005A, 0x04EB9DAA	; v_mfma_f32_16x16x32_bf16 v[90:93], v[170:173], v[206:209], v[58:61]
	.long	0xD3B5003A, 0x053B7DAE	; v_mfma_f32_16x16x32_bf16 v[58:61], v[174:177], v[190:193], v[78:81]
	.long	0xD3B50056, 0x04EB85B2	; v_mfma_f32_16x16x32_bf16 v[86:89], v[178:181], v[194:197], v[58:61]
	.long	0xD3B5003A, 0x067B8DAE	; v_mfma_f32_16x16x32_bf16 v[58:61], v[174:177], v[198:201], v[158:161]
	.long	0xD3B5004E, 0x04EB9DB2	; v_mfma_f32_16x16x32_bf16 v[78:81], v[178:181], v[206:209], v[58:61]
	.long	0xD3B5003A, 0x051B7DB6	; v_mfma_f32_16x16x32_bf16 v[58:61], v[182:185], v[190:193], v[70:73]
	.long	0xD3B50042, 0x04EB85BA	; v_mfma_f32_16x16x32_bf16 v[66:69], v[186:189], v[194:197], v[58:61]
	.long	0xD3B5003A, 0x068B8DB6	; v_mfma_f32_16x16x32_bf16 v[58:61], v[182:185], v[198:201], v[162:165]
	.long	0xD3B5003A, 0x04EB9DBA	; v_mfma_f32_16x16x32_bf16 v[58:61], v[186:189], v[206:209], v[58:61]
	.long	0xBF8F0000	; s_setprio 0
	.long	0xBF8A0000	; s_barrier
	.long	0xD1FF0046, 0x061B0419	; v_add3_u32 v70, s25, v130, v134
	.long	0xD9FE0000, 0x84000046	; ds_read_b128 v[132:135], v70
	.long	0xD9FE0400, 0x9A000046	; ds_read_b128 v[154:157], v70 offset:1024
	.long	0xD9FE0800, 0x9E000046	; ds_read_b128 v[158:161], v70 offset:2048
	.long	0xD9FE0C00, 0xA2000046	; ds_read_b128 v[162:165], v70 offset:3072
	.long	0xD9FE1000, 0xA6000046	; ds_read_b128 v[166:169], v70 offset:4096
	.long	0xD9FE1400, 0xAA000046	; ds_read_b128 v[170:173], v70 offset:5120
	.long	0xD9FE1800, 0xAE000046	; ds_read_b128 v[174:177], v70 offset:6144
	.long	0xD9FE1C00, 0xB2000046	; ds_read_b128 v[178:181], v70 offset:7168
	.long	0xBF8A0000	; s_barrier
	.long	0xBF8CC07F	; s_waitcnt lgkmcnt(0)
	.long	0xBF8F0001	; s_setprio 1
	.long	0xD3B5003E, 0x04FB1584	; v_mfma_f32_16x16x32_bf16 v[62:65], v[132:135], v[138:141], v[62:65]
	.long	0xD3B50046, 0x04FB1D9A	; v_mfma_f32_16x16x32_bf16 v[70:73], v[154:157], v[142:145], v[62:65]
	.long	0xD3B5003E, 0x072B2584	; v_mfma_f32_16x16x32_bf16 v[62:65], v[132:135], v[146:149], v[202:205]
	.long	0xD3B50036, 0x04DB159E	; v_mfma_f32_16x16x32_bf16 v[54:57], v[158:161], v[138:141], v[54:57]
	.long	0xD3B50032, 0x04CB259E	; v_mfma_f32_16x16x32_bf16 v[50:53], v[158:161], v[146:149], v[50:53]
	.long	0xD3B5002E, 0x04BB15A6	; v_mfma_f32_16x16x32_bf16 v[46:49], v[166:169], v[138:141], v[46:49]
	.long	0xD3B5002A, 0x04AB25A6	; v_mfma_f32_16x16x32_bf16 v[42:45], v[166:169], v[146:149], v[42:45]
	.long	0xD3B50026, 0x049B15AE	; v_mfma_f32_16x16x32_bf16 v[38:41], v[174:177], v[138:141], v[38:41]
	.long	0xD3B50022, 0x048B25AE	; v_mfma_f32_16x16x32_bf16 v[34:37], v[174:177], v[146:149], v[34:37]
	.long	0xD3B5001E, 0x047B7D84	; v_mfma_f32_16x16x32_bf16 v[30:33], v[132:135], v[190:193], v[30:33]
	.long	0xD3B5001A, 0x046B8D84	; v_mfma_f32_16x16x32_bf16 v[26:29], v[132:135], v[198:201], v[26:29]
	.long	0xD3B50016, 0x045B7D9E	; v_mfma_f32_16x16x32_bf16 v[22:25], v[158:161], v[190:193], v[22:25]
	.long	0xD3B50012, 0x044B8D9E	; v_mfma_f32_16x16x32_bf16 v[18:21], v[158:161], v[198:201], v[18:21]
	.long	0xD3B5000E, 0x043B7DA6	; v_mfma_f32_16x16x32_bf16 v[14:17], v[166:169], v[190:193], v[14:17]
	.long	0xD3B5000A, 0x042B8DA6	; v_mfma_f32_16x16x32_bf16 v[10:13], v[166:169], v[198:201], v[10:13]
	.long	0xD3B50006, 0x041B7DAE	; v_mfma_f32_16x16x32_bf16 v[6:9], v[174:177], v[190:193], v[6:9]
	.long	0xD3B50002, 0x040B8DAE	; v_mfma_f32_16x16x32_bf16 v[2:5], v[174:177], v[198:201], v[2:5]
	.long	0xD3B5003E, 0x04FB2D9A	; v_mfma_f32_16x16x32_bf16 v[62:65], v[154:157], v[150:153], v[62:65]
	.long	0xD3B50036, 0x04DB1DA2	; v_mfma_f32_16x16x32_bf16 v[54:57], v[162:165], v[142:145], v[54:57]
	.long	0xD3B50032, 0x04CB2DA2	; v_mfma_f32_16x16x32_bf16 v[50:53], v[162:165], v[150:153], v[50:53]
	.long	0xD3B5002E, 0x04BB1DAA	; v_mfma_f32_16x16x32_bf16 v[46:49], v[170:173], v[142:145], v[46:49]
	.long	0xD3B5002A, 0x04AB2DAA	; v_mfma_f32_16x16x32_bf16 v[42:45], v[170:173], v[150:153], v[42:45]
	.long	0xD3B50026, 0x049B1DB2	; v_mfma_f32_16x16x32_bf16 v[38:41], v[178:181], v[142:145], v[38:41]
	.long	0xD3B50022, 0x048B2DB2	; v_mfma_f32_16x16x32_bf16 v[34:37], v[178:181], v[150:153], v[34:37]
	.long	0xD3B5001E, 0x047B859A	; v_mfma_f32_16x16x32_bf16 v[30:33], v[154:157], v[194:197], v[30:33]
	.long	0xD3B5001A, 0x046B9D9A	; v_mfma_f32_16x16x32_bf16 v[26:29], v[154:157], v[206:209], v[26:29]
	.long	0xD3B50016, 0x045B85A2	; v_mfma_f32_16x16x32_bf16 v[22:25], v[162:165], v[194:197], v[22:25]
	.long	0xD3B50012, 0x044B9DA2	; v_mfma_f32_16x16x32_bf16 v[18:21], v[162:165], v[206:209], v[18:21]
	.long	0xD3B5000E, 0x043B85AA	; v_mfma_f32_16x16x32_bf16 v[14:17], v[170:173], v[194:197], v[14:17]
	.long	0xD3B5000A, 0x042B9DAA	; v_mfma_f32_16x16x32_bf16 v[10:13], v[170:173], v[206:209], v[10:13]
	.long	0xD3B50006, 0x041B85B2	; v_mfma_f32_16x16x32_bf16 v[6:9], v[178:181], v[194:197], v[6:9]
	.long	0xD3B50002, 0x040B9DB2	; v_mfma_f32_16x16x32_bf16 v[2:5], v[178:181], v[206:209], v[2:5]
	.long	0xBF8F0000	; s_setprio 0
	.long	0xB0000100	; s_movk_i32 s0, 0x100
	.long	0x7D980000	; v_cmp_gt_u32_e32 vcc, s0, v0
	.long	0xBF8A0000	; s_barrier
	.long	0xBE80206A	; s_and_saveexec_b64 s[0:1], vcc
	.long	0xBF880001	; s_cbranch_execz 1
	.long	0xBF8A0000	; s_barrier
	.long	0x87FE007E	; s_or_b64 exec, exec, s[0:1]
	.long	0x24020286	; v_lshlrev_b32_e32 v1, 6, v1
	.long	0xD20000A8, 0x04051014	; v_lshl_or_b32 v168, s20, 8, v1
	.long	0x2603108C	; v_and_b32_e32 v1, 12, v136
	.long	0x2600008F	; v_and_b32_e32 v0, 15, v0
	.long	0xD1E80000, 0x04001901	; v_mad_u64_u32 v[0:1], s[0:1], v1, s12, v[0:1]
	.long	0xD2000082, 0x00550B83	; v_lshl_or_b32 v130, v131, 5, s21
	.long	0x2202009F	; v_ashrrev_i32_e32 v1, 31, v0
	.long	0xD1E90084, 0x0203500C	; v_mad_i64_i32 v[132:133], s[0:1], s12, v168, 0
	.long	0x2307049F	; v_ashrrev_i32_e32 v131, 31, v130
	.long	0xD28F0086, 0x00020081	; v_lshlrev_b64 v[134:135], 1, v[0:1]
	.long	0x6800000C	; v_add_u32_e32 v0, s12, v0
	.long	0xD2080084, 0x00090384	; v_lshl_add_u64 v[132:133], v[132:133], 1, s[2:3]
	.long	0xD28F0082, 0x00030481	; v_lshlrev_b64 v[130:131], 1, v[130:131]
	.long	0x2202009F	; v_ashrrev_i32_e32 v1, 31, v0
	.long	0xD2080084, 0x06090184	; v_lshl_add_u64 v[132:133], v[132:133], 0, v[130:131]
	.long	0xD28F008A, 0x00020081	; v_lshlrev_b64 v[138:139], 1, v[0:1]
	.long	0x6800000C	; v_add_u32_e32 v0, s12, v0
	.long	0xD2080088, 0x06190184	; v_lshl_add_u64 v[136:137], v[132:133], 0, v[134:135]
	.long	0xD208008C, 0x06290184	; v_lshl_add_u64 v[140:141], v[132:133], 0, v[138:139]
	.long	0x2202009F	; v_ashrrev_i32_e32 v1, 31, v0
	.long	0xDC6C8000, 0x007F7E88	; global_store_short_d16_hi v[136:137], v126, off
	.long	0xDC6C8000, 0x007F7F8C	; global_store_short_d16_hi v[140:141], v127, off
	.long	0xD28F007E, 0x00020081	; v_lshlrev_b64 v[126:127], 1, v[0:1]
	.long	0x6800000C	; v_add_u32_e32 v0, s12, v0
	.long	0x2202009F	; v_ashrrev_i32_e32 v1, 31, v0
	.long	0x92008D0C	; s_mul_i32 s0, s12, 13
	.long	0xD28F0090, 0x00020081	; v_lshlrev_b64 v[144:145], 1, v[0:1]
	.long	0x68000000	; v_add_u32_e32 v0, s0, v0
	.long	0xD208008E, 0x05F90184	; v_lshl_add_u64 v[142:143], v[132:133], 0, v[126:127]
	.long	0xD2080092, 0x06410184	; v_lshl_add_u64 v[146:147], v[132:133], 0, v[144:145]
	.long	0x2202009F	; v_ashrrev_i32_e32 v1, 31, v0
	.long	0xDC6C8000, 0x007F808E	; global_store_short_d16_hi v[142:143], v128, off
	.long	0xDC6C8000, 0x007F8192	; global_store_short_d16_hi v[146:147], v129, off
	.long	0xDC6C8020, 0x007F7A88	; global_store_short_d16_hi v[136:137], v122, off offset:32
	.long	0xDC6C8020, 0x007F7B8C	; global_store_short_d16_hi v[140:141], v123, off offset:32
	.long	0xDC6C8020, 0x007F7C8E	; global_store_short_d16_hi v[142:143], v124, off offset:32
	.long	0xDC6C8020, 0x007F7D92	; global_store_short_d16_hi v[146:147], v125, off offset:32
	.long	0xD28F007A, 0x00020081	; v_lshlrev_b64 v[122:123], 1, v[0:1]
	.long	0x6800000C	; v_add_u32_e32 v0, s12, v0
	.long	0x2202009F	; v_ashrrev_i32_e32 v1, 31, v0
	.long	0xD28F0080, 0x00020081	; v_lshlrev_b64 v[128:129], 1, v[0:1]
	.long	0x6800000C	; v_add_u32_e32 v0, s12, v0
	.long	0xD208007C, 0x05E90184	; v_lshl_add_u64 v[124:125], v[132:133], 0, v[122:123]
	.long	0xD2080094, 0x06010184	; v_lshl_add_u64 v[148:149], v[132:133], 0, v[128:129]
	.long	0x2202009F	; v_ashrrev_i32_e32 v1, 31, v0
	.long	0xDC6C8000, 0x007F767C	; global_store_short_d16_hi v[124:125], v118, off
	.long	0xDC6C8000, 0x007F7794	; global_store_short_d16_hi v[148:149], v119, off
	.long	0xD28F0076, 0x00020081	; v_lshlrev_b64 v[118:119], 1, v[0:1]
	.long	0x6800000C	; v_add_u32_e32 v0, s12, v0
	.long	0x2202009F	; v_ashrrev_i32_e32 v1, 31, v0
	.long	0xD28F0098, 0x00020081	; v_lshlrev_b64 v[152:153], 1, v[0:1]
	.long	0x68000000	; v_add_u32_e32 v0, s0, v0
	.long	0xD2080096, 0x05D90184	; v_lshl_add_u64 v[150:151], v[132:133], 0, v[118:119]
	.long	0xD208009A, 0x06610184	; v_lshl_add_u64 v[154:155], v[132:133], 0, v[152:153]
	.long	0x2202009F	; v_ashrrev_i32_e32 v1, 31, v0
	.long	0xDC6C8000, 0x007F7896	; global_store_short_d16_hi v[150:151], v120, off
	.long	0xDC6C8000, 0x007F799A	; global_store_short_d16_hi v[154:155], v121, off
	.long	0xDC6C8020, 0x007F727C	; global_store_short_d16_hi v[124:125], v114, off offset:32
	.long	0xDC6C8020, 0x007F7394	; global_store_short_d16_hi v[148:149], v115, off offset:32
	.long	0xDC6C8020, 0x007F7496	; global_store_short_d16_hi v[150:151], v116, off offset:32
	.long	0xDC6C8020, 0x007F759A	; global_store_short_d16_hi v[154:155], v117, off offset:32
	.long	0xD28F0072, 0x00020081	; v_lshlrev_b64 v[114:115], 1, v[0:1]
	.long	0x6800000C	; v_add_u32_e32 v0, s12, v0
	.long	0x2202009F	; v_ashrrev_i32_e32 v1, 31, v0
	.long	0xD28F0078, 0x00020081	; v_lshlrev_b64 v[120:121], 1, v[0:1]
	.long	0x6800000C	; v_add_u32_e32 v0, s12, v0
	.long	0xD2080074, 0x05C90184	; v_lshl_add_u64 v[116:117], v[132:133], 0, v[114:115]
	.long	0xD208009C, 0x05E10184	; v_lshl_add_u64 v[156:157], v[132:133], 0, v[120:121]
	.long	0x2202009F	; v_ashrrev_i32_e32 v1, 31, v0
	.long	0xDC6C8000, 0x007F6E74	; global_store_short_d16_hi v[116:117], v110, off
	.long	0xDC6C8000, 0x007F6F9C	; global_store_short_d16_hi v[156:157], v111, off
	.long	0xD28F006E, 0x00020081	; v_lshlrev_b64 v[110:111], 1, v[0:1]
	.long	0x6800000C	; v_add_u32_e32 v0, s12, v0
	.long	0x2202009F	; v_ashrrev_i32_e32 v1, 31, v0
	.long	0xD28F00A0, 0x00020081	; v_lshlrev_b64 v[160:161], 1, v[0:1]
	.long	0x68000000	; v_add_u32_e32 v0, s0, v0
	.long	0xD208009E, 0x05B90184	; v_lshl_add_u64 v[158:159], v[132:133], 0, v[110:111]
	.long	0xD20800A2, 0x06810184	; v_lshl_add_u64 v[162:163], v[132:133], 0, v[160:161]
	.long	0x2202009F	; v_ashrrev_i32_e32 v1, 31, v0
	.long	0xDC6C8000, 0x007F709E	; global_store_short_d16_hi v[158:159], v112, off
	.long	0xDC6C8000, 0x007F71A2	; global_store_short_d16_hi v[162:163], v113, off
	.long	0xDC6C8020, 0x007F6A74	; global_store_short_d16_hi v[116:117], v106, off offset:32
	.long	0xDC6C8020, 0x007F6B9C	; global_store_short_d16_hi v[156:157], v107, off offset:32
	.long	0xDC6C8020, 0x007F6C9E	; global_store_short_d16_hi v[158:159], v108, off offset:32
	.long	0xDC6C8020, 0x007F6DA2	; global_store_short_d16_hi v[162:163], v109, off offset:32
	.long	0xD28F006A, 0x00020081	; v_lshlrev_b64 v[106:107], 1, v[0:1]
	.long	0x6800000C	; v_add_u32_e32 v0, s12, v0
	.long	0x2202009F	; v_ashrrev_i32_e32 v1, 31, v0
	.long	0xD28F0070, 0x00020081	; v_lshlrev_b64 v[112:113], 1, v[0:1]
	.long	0x6800000C	; v_add_u32_e32 v0, s12, v0
	.long	0xD208006C, 0x05A90184	; v_lshl_add_u64 v[108:109], v[132:133], 0, v[106:107]
	.long	0xD20800A4, 0x05C10184	; v_lshl_add_u64 v[164:165], v[132:133], 0, v[112:113]
	.long	0x2202009F	; v_ashrrev_i32_e32 v1, 31, v0
	.long	0xDC6C8000, 0x007F526C	; global_store_short_d16_hi v[108:109], v82, off
	.long	0xDC6C8000, 0x007F53A4	; global_store_short_d16_hi v[164:165], v83, off
	.long	0xD28F0052, 0x00020081	; v_lshlrev_b64 v[82:83], 1, v[0:1]
	.long	0x6800000C	; v_add_u32_e32 v0, s12, v0
	.long	0x2202009F	; v_ashrrev_i32_e32 v1, 31, v0
	.long	0xD28F0000, 0x00020081	; v_lshlrev_b64 v[0:1], 1, v[0:1]
	.long	0xD20800A6, 0x05490184	; v_lshl_add_u64 v[166:167], v[132:133], 0, v[82:83]
	.long	0xD2080084, 0x04010184	; v_lshl_add_u64 v[132:133], v[132:133], 0, v[0:1]
	.long	0xDC6C8000, 0x007F54A6	; global_store_short_d16_hi v[166:167], v84, off
	.long	0xDC6C8000, 0x007F5584	; global_store_short_d16_hi v[132:133], v85, off
	.long	0xDC6C8020, 0x007F4A6C	; global_store_short_d16_hi v[108:109], v74, off offset:32
	.long	0xDC6C8020, 0x007F4BA4	; global_store_short_d16_hi v[164:165], v75, off offset:32
	.long	0xDC6C8020, 0x007F4CA6	; global_store_short_d16_hi v[166:167], v76, off offset:32
	.long	0xDC6C8020, 0x007F4D84	; global_store_short_d16_hi v[132:133], v77, off offset:32
	.long	0xDC6C8100, 0x007F6688	; global_store_short_d16_hi v[136:137], v102, off offset:256
	.long	0xDC6C8100, 0x007F678C	; global_store_short_d16_hi v[140:141], v103, off offset:256
	.long	0xDC6C8100, 0x007F688E	; global_store_short_d16_hi v[142:143], v104, off offset:256
	.long	0xDC6C8100, 0x007F6992	; global_store_short_d16_hi v[146:147], v105, off offset:256
	.long	0xDC6C8120, 0x007F6288	; global_store_short_d16_hi v[136:137], v98, off offset:288
	.long	0xDC6C8120, 0x007F638C	; global_store_short_d16_hi v[140:141], v99, off offset:288
	.long	0xDC6C8120, 0x007F648E	; global_store_short_d16_hi v[142:143], v100, off offset:288
	.long	0xDC6C8120, 0x007F6592	; global_store_short_d16_hi v[146:147], v101, off offset:288
	.long	0xDC6C8100, 0x007F5E7C	; global_store_short_d16_hi v[124:125], v94, off offset:256
	.long	0xDC6C8100, 0x007F5F94	; global_store_short_d16_hi v[148:149], v95, off offset:256
	.long	0xDC6C8100, 0x007F6096	; global_store_short_d16_hi v[150:151], v96, off offset:256
	.long	0xDC6C8100, 0x007F619A	; global_store_short_d16_hi v[154:155], v97, off offset:256
	.long	0xDC6C8120, 0x007F5A7C	; global_store_short_d16_hi v[124:125], v90, off offset:288
	.long	0xDC6C8120, 0x007F5B94	; global_store_short_d16_hi v[148:149], v91, off offset:288
	.long	0xDC6C8120, 0x007F5C96	; global_store_short_d16_hi v[150:151], v92, off offset:288
	.long	0xDC6C8120, 0x007F5D9A	; global_store_short_d16_hi v[154:155], v93, off offset:288
	.long	0xDC6C8100, 0x007F5674	; global_store_short_d16_hi v[116:117], v86, off offset:256
	.long	0xDC6C8100, 0x007F579C	; global_store_short_d16_hi v[156:157], v87, off offset:256
	.long	0xDC6C8100, 0x007F589E	; global_store_short_d16_hi v[158:159], v88, off offset:256
	.long	0xDC6C8100, 0x007F59A2	; global_store_short_d16_hi v[162:163], v89, off offset:256
	.long	0xDC6C8120, 0x007F4E74	; global_store_short_d16_hi v[116:117], v78, off offset:288
	.long	0xDC6C8120, 0x007F4F9C	; global_store_short_d16_hi v[156:157], v79, off offset:288
	.long	0xDC6C8120, 0x007F509E	; global_store_short_d16_hi v[158:159], v80, off offset:288
	.long	0xDC6C8120, 0x007F51A2	; global_store_short_d16_hi v[162:163], v81, off offset:288
	.long	0xDC6C8100, 0x007F426C	; global_store_short_d16_hi v[108:109], v66, off offset:256
	.long	0xDC6C8100, 0x007F43A4	; global_store_short_d16_hi v[164:165], v67, off offset:256
	.long	0xDC6C8100, 0x007F44A6	; global_store_short_d16_hi v[166:167], v68, off offset:256
	.long	0xDC6C8100, 0x007F4584	; global_store_short_d16_hi v[132:133], v69, off offset:256
	.long	0xDC6C8120, 0x007F3A6C	; global_store_short_d16_hi v[108:109], v58, off offset:288
	.long	0xDC6C8120, 0x007F3BA4	; global_store_short_d16_hi v[164:165], v59, off offset:288
	.long	0xDC6C8120, 0x007F3CA6	; global_store_short_d16_hi v[166:167], v60, off offset:288
	.long	0xDC6C8120, 0x007F3D84	; global_store_short_d16_hi v[132:133], v61, off offset:288
	.long	0x287550FF, 0x00000080	; v_or_b32_e32 v58, 0x80, v168
	.long	0xD1E9003A, 0x0202740C	; v_mad_i64_i32 v[58:59], s[0:1], s12, v58, 0
	.long	0xD208003A, 0x0009033A	; v_lshl_add_u64 v[58:59], v[58:59], 1, s[2:3]
	.long	0xD208003A, 0x0609013A	; v_lshl_add_u64 v[58:59], v[58:59], 0, v[130:131]
	.long	0xD208003C, 0x0619013A	; v_lshl_add_u64 v[60:61], v[58:59], 0, v[134:135]
	.long	0xD2080042, 0x0629013A	; v_lshl_add_u64 v[66:67], v[58:59], 0, v[138:139]
	.long	0xDC6C8000, 0x007F463C	; global_store_short_d16_hi v[60:61], v70, off
	.long	0xDC6C8000, 0x007F4742	; global_store_short_d16_hi v[66:67], v71, off
	.long	0xD2080044, 0x05F9013A	; v_lshl_add_u64 v[68:69], v[58:59], 0, v[126:127]
	.long	0xD2080046, 0x0641013A	; v_lshl_add_u64 v[70:71], v[58:59], 0, v[144:145]
	.long	0xDC6C8000, 0x007F4844	; global_store_short_d16_hi v[68:69], v72, off
	.long	0xDC6C8000, 0x007F4946	; global_store_short_d16_hi v[70:71], v73, off
	.long	0xDC6C8020, 0x007F3E3C	; global_store_short_d16_hi v[60:61], v62, off offset:32
	.long	0xDC6C8020, 0x007F3F42	; global_store_short_d16_hi v[66:67], v63, off offset:32
	.long	0xDC6C8020, 0x007F4044	; global_store_short_d16_hi v[68:69], v64, off offset:32
	.long	0xDC6C8020, 0x007F4146	; global_store_short_d16_hi v[70:71], v65, off offset:32
	.long	0xD208003E, 0x05E9013A	; v_lshl_add_u64 v[62:63], v[58:59], 0, v[122:123]
	.long	0xD2080040, 0x0601013A	; v_lshl_add_u64 v[64:65], v[58:59], 0, v[128:129]
	.long	0xDC6C8000, 0x007F363E	; global_store_short_d16_hi v[62:63], v54, off
	.long	0xDC6C8000, 0x007F3740	; global_store_short_d16_hi v[64:65], v55, off
	.long	0xD2080036, 0x05D9013A	; v_lshl_add_u64 v[54:55], v[58:59], 0, v[118:119]
	.long	0xD2080048, 0x0661013A	; v_lshl_add_u64 v[72:73], v[58:59], 0, v[152:153]
	.long	0xDC6C8000, 0x007F3836	; global_store_short_d16_hi v[54:55], v56, off
	.long	0xDC6C8000, 0x007F3948	; global_store_short_d16_hi v[72:73], v57, off
	.long	0xDC6C8020, 0x007F323E	; global_store_short_d16_hi v[62:63], v50, off offset:32
	.long	0xDC6C8020, 0x007F3340	; global_store_short_d16_hi v[64:65], v51, off offset:32
	.long	0xDC6C8020, 0x007F3436	; global_store_short_d16_hi v[54:55], v52, off offset:32
	.long	0xDC6C8020, 0x007F3548	; global_store_short_d16_hi v[72:73], v53, off offset:32
	.long	0xD2080032, 0x05C9013A	; v_lshl_add_u64 v[50:51], v[58:59], 0, v[114:115]
	.long	0xD2080034, 0x05E1013A	; v_lshl_add_u64 v[52:53], v[58:59], 0, v[120:121]
	.long	0xDC6C8000, 0x007F2E32	; global_store_short_d16_hi v[50:51], v46, off
	.long	0xDC6C8000, 0x007F2F34	; global_store_short_d16_hi v[52:53], v47, off
	.long	0xD208002E, 0x05B9013A	; v_lshl_add_u64 v[46:47], v[58:59], 0, v[110:111]
	.long	0xD2080038, 0x0681013A	; v_lshl_add_u64 v[56:57], v[58:59], 0, v[160:161]
	.long	0xDC6C8000, 0x007F302E	; global_store_short_d16_hi v[46:47], v48, off
	.long	0xDC6C8000, 0x007F3138	; global_store_short_d16_hi v[56:57], v49, off
	.long	0xDC6C8020, 0x007F2A32	; global_store_short_d16_hi v[50:51], v42, off offset:32
	.long	0xDC6C8020, 0x007F2B34	; global_store_short_d16_hi v[52:53], v43, off offset:32
	.long	0xDC6C8020, 0x007F2C2E	; global_store_short_d16_hi v[46:47], v44, off offset:32
	.long	0xDC6C8020, 0x007F2D38	; global_store_short_d16_hi v[56:57], v45, off offset:32
	.long	0xD208002A, 0x05A9013A	; v_lshl_add_u64 v[42:43], v[58:59], 0, v[106:107]
	.long	0xD208002C, 0x05C1013A	; v_lshl_add_u64 v[44:45], v[58:59], 0, v[112:113]
	.long	0xDC6C8000, 0x007F262A	; global_store_short_d16_hi v[42:43], v38, off
	.long	0xDC6C8000, 0x007F272C	; global_store_short_d16_hi v[44:45], v39, off
	.long	0xD2080026, 0x0549013A	; v_lshl_add_u64 v[38:39], v[58:59], 0, v[82:83]
	.long	0xD2080000, 0x0401013A	; v_lshl_add_u64 v[0:1], v[58:59], 0, v[0:1]
	.long	0xDC6C8000, 0x007F2826	; global_store_short_d16_hi v[38:39], v40, off
	.long	0xDC6C8000, 0x007F2900	; global_store_short_d16_hi v[0:1], v41, off
	.long	0xDC6C8020, 0x007F222A	; global_store_short_d16_hi v[42:43], v34, off offset:32
	.long	0xDC6C8020, 0x007F232C	; global_store_short_d16_hi v[44:45], v35, off offset:32
	.long	0xDC6C8020, 0x007F2426	; global_store_short_d16_hi v[38:39], v36, off offset:32
	.long	0xDC6C8020, 0x007F2500	; global_store_short_d16_hi v[0:1], v37, off offset:32
	.long	0xDC6C8100, 0x007F1E3C	; global_store_short_d16_hi v[60:61], v30, off offset:256
	.long	0xDC6C8100, 0x007F1F42	; global_store_short_d16_hi v[66:67], v31, off offset:256
	.long	0xDC6C8100, 0x007F2044	; global_store_short_d16_hi v[68:69], v32, off offset:256
	.long	0xDC6C8100, 0x007F2146	; global_store_short_d16_hi v[70:71], v33, off offset:256
	.long	0xDC6C8120, 0x007F1A3C	; global_store_short_d16_hi v[60:61], v26, off offset:288
	.long	0xDC6C8120, 0x007F1B42	; global_store_short_d16_hi v[66:67], v27, off offset:288
	.long	0xDC6C8120, 0x007F1C44	; global_store_short_d16_hi v[68:69], v28, off offset:288
	.long	0xDC6C8120, 0x007F1D46	; global_store_short_d16_hi v[70:71], v29, off offset:288
	.long	0xDC6C8100, 0x007F163E	; global_store_short_d16_hi v[62:63], v22, off offset:256
	.long	0xDC6C8100, 0x007F1740	; global_store_short_d16_hi v[64:65], v23, off offset:256
	.long	0xDC6C8100, 0x007F1836	; global_store_short_d16_hi v[54:55], v24, off offset:256
	.long	0xDC6C8100, 0x007F1948	; global_store_short_d16_hi v[72:73], v25, off offset:256
	.long	0xDC6C8120, 0x007F123E	; global_store_short_d16_hi v[62:63], v18, off offset:288
	.long	0xDC6C8120, 0x007F1340	; global_store_short_d16_hi v[64:65], v19, off offset:288
	.long	0xDC6C8120, 0x007F1436	; global_store_short_d16_hi v[54:55], v20, off offset:288
	.long	0xDC6C8120, 0x007F1548	; global_store_short_d16_hi v[72:73], v21, off offset:288
	.long	0xDC6C8100, 0x007F0E32	; global_store_short_d16_hi v[50:51], v14, off offset:256
	.long	0xDC6C8100, 0x007F0F34	; global_store_short_d16_hi v[52:53], v15, off offset:256
	.long	0xDC6C8100, 0x007F102E	; global_store_short_d16_hi v[46:47], v16, off offset:256
	.long	0xDC6C8100, 0x007F1138	; global_store_short_d16_hi v[56:57], v17, off offset:256
	.long	0xDC6C8120, 0x007F0A32	; global_store_short_d16_hi v[50:51], v10, off offset:288
	.long	0xDC6C8120, 0x007F0B34	; global_store_short_d16_hi v[52:53], v11, off offset:288
	.long	0xDC6C8120, 0x007F0C2E	; global_store_short_d16_hi v[46:47], v12, off offset:288
	.long	0xDC6C8120, 0x007F0D38	; global_store_short_d16_hi v[56:57], v13, off offset:288
	.long	0xDC6C8100, 0x007F062A	; global_store_short_d16_hi v[42:43], v6, off offset:256
	.long	0xDC6C8100, 0x007F072C	; global_store_short_d16_hi v[44:45], v7, off offset:256
	.long	0xDC6C8100, 0x007F0826	; global_store_short_d16_hi v[38:39], v8, off offset:256
	.long	0xDC6C8100, 0x007F0900	; global_store_short_d16_hi v[0:1], v9, off offset:256
	.long	0xDC6C8120, 0x007F022A	; global_store_short_d16_hi v[42:43], v2, off offset:288
	.long	0xDC6C8120, 0x007F032C	; global_store_short_d16_hi v[44:45], v3, off offset:288
	.long	0xDC6C8120, 0x007F0426	; global_store_short_d16_hi v[38:39], v4, off offset:288
	.long	0xDC6C8120, 0x007F0500	; global_store_short_d16_hi v[0:1], v5, off offset:288
	.long	0xBF810000	; s_endpgm
.Lfunc_end0:
	.size _Z8micro_tk13micro_globalsiii, .Lfunc_end0-_Z8micro_tk13micro_globalsiii

.section .rodata,"a",@progbits
.p2align 6
.protected _Z8micro_tk13micro_globalsiii.kd
.globl _Z8micro_tk13micro_globalsiii.kd
.type _Z8micro_tk13micro_globalsiii.kd,@object
_Z8micro_tk13micro_globalsiii.kd:
	.long	0x00000000
	.long	0x00000000
	.long	0x000001b8
	.long	0x00000000
	.quad	_Z8micro_tk13micro_globalsiii - _Z8micro_tk13micro_globalsiii.kd
	.byte	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
	.byte	0x00, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00, 0x9a, 0x01, 0xaf, 0x00, 0x84, 0x01, 0x00, 0x00
	.byte	0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
	.size _Z8micro_tk13micro_globalsiii.kd, .-_Z8micro_tk13micro_globalsiii.kd
	.set _Z8micro_tk13micro_globalsiii.private_seg_size, 0
	.set _Z8micro_tk13micro_globalsiii.num_vgpr, 210
	.set _Z8micro_tk13micro_globalsiii.num_agpr, 0
	.set _Z8micro_tk13micro_globalsiii.numbered_sgpr, 47
	.set _Z8micro_tk13micro_globalsiii.uses_vcc, 1
	.set _Z8micro_tk13micro_globalsiii.uses_flat_scratch, 0
	.set _Z8micro_tk13micro_globalsiii.has_dyn_sized_stack, 0
	.set _Z8micro_tk13micro_globalsiii.has_recursion, 0

	.section .AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.section ".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           168
        .value_kind:     by_value
      - .offset:         168
        .size:           4
        .value_kind:     by_value
      - .offset:         172
        .size:           4
        .value_kind:     by_value
      - .offset:         176
        .size:           4
        .value_kind:     by_value
      - .offset:         184
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         188
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         192
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         196
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         198
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         200
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         202
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         204
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         206
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         224
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         232
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         240
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         248
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         304
        .size:           4
        .value_kind:     hidden_dynamic_lds_size
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 440
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z8micro_tk13micro_globalsiii
    .private_segment_fixed_size: 0
    .sgpr_count:     53
    .sgpr_spill_count: 0
    .symbol:         _Z8micro_tk13micro_globalsiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     210
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
