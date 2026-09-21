// Generated from the HipKittens gfx950 code object for DBT tests.
// Source: https://github.com/HazyResearch/HipKittens/blob/main/kernels/gemm/bf16fp32/256_256_64_32_with32x16.cpp
// Original hsaco sha256: c495e1f5329b6a394f041ab61b6b46bb3cd45ffbaf3201e6d51731f11dbc77c4
// Instruction words are exact; comments are llvm-objdump decoded assembly.
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.protected _Z8micro_tk13micro_globals
.globl _Z8micro_tk13micro_globals
.p2align 8
.type _Z8micro_tk13micro_globals,@function
_Z8micro_tk13micro_globals:
	.long	0xBE8401EB	; s_mov_b64 s[4:5], src_shared_base
	.long	0xBF07C180	; s_cmp_lg_u32 0, -1
	.long	0x85048005	; s_cselect_b32 s4, s5, 0
	.long	0x85058080	; s_cselect_b32 s5, 0, 0
	.long	0x86108F05	; s_and_b32 s16, s5, 15
	.long	0x8606D005	; s_and_b32 s6, s5, -16
	.long	0x80069006	; s_add_u32 s6, s6, 16
	.long	0xBE910080	; s_mov_b32 s17, 0
	.long	0x82078004	; s_addc_u32 s7, s4, 0
	.long	0xBF128010	; s_cmp_eq_u64 s[16:17], 0
	.long	0x85120605	; s_cselect_b32 s18, s5, s6
	.long	0x85130704	; s_cselect_b32 s19, s4, s7
	.long	0x8006FF12, 0x00010000	; s_add_u32 s6, s18, 0x10000
	.long	0x82078013	; s_addc_u32 s7, s19, 0
	.long	0x86108F06	; s_and_b32 s16, s6, 15
	.long	0x8604D006	; s_and_b32 s4, s6, -16
	.long	0x80089004	; s_add_u32 s8, s4, 16
	.long	0xC0060100, 0x00000098	; s_load_dwordx2 s[4:5], s[0:1], 0x98
	.long	0x82098007	; s_addc_u32 s9, s7, 0
	.long	0xBF128010	; s_cmp_eq_u64 s[16:17], 0
	.long	0x85150907	; s_cselect_b32 s21, s7, s9
	.long	0x85140806	; s_cselect_b32 s20, s6, s8
	.long	0xBF8CC07F	; s_waitcnt lgkmcnt(0)
	.long	0x92030304	; s_mul_i32 s3, s4, s3
	.long	0x81020203	; s_add_i32 s2, s3, s2
	.long	0x92030405	; s_mul_i32 s3, s5, s4
	.long	0x90049F02	; s_ashr_i32 s4, s2, 31
	.long	0x8F049D04	; s_lshr_b32 s4, s4, 29
	.long	0x81040402	; s_add_i32 s4, s2, s4
	.long	0x90058304	; s_ashr_i32 s5, s4, 3
	.long	0x8604C804	; s_and_b32 s4, s4, -8
	.long	0x81820402	; s_sub_i32 s2, s2, s4
	.long	0x90049F03	; s_ashr_i32 s4, s3, 31
	.long	0x8F049D04	; s_lshr_b32 s4, s4, 29
	.long	0x81030403	; s_add_i32 s3, s3, s4
	.long	0x90038303	; s_ashr_i32 s3, s3, 3
	.long	0x92020302	; s_mul_i32 s2, s2, s3
	.long	0x81020502	; s_add_i32 s2, s2, s5
	.long	0x90039F02	; s_ashr_i32 s3, s2, 31
	.long	0x8F039803	; s_lshr_b32 s3, s3, 24
	.long	0x81030302	; s_add_i32 s3, s2, s3
	.long	0x90048803	; s_ashr_i32 s4, s3, 8
	.long	0x8E048304	; s_lshl_b32 s4, s4, 3
	.long	0x818504A0	; s_sub_i32 s5, 32, s4
	.long	0x83058805	; s_min_i32 s5, s5, 8
	.long	0xBE863005	; s_abs_i32 s6, s5
	.long	0x7E020C06	; v_cvt_f32_u32_e32 v1, s6
	.long	0x81880680	; s_sub_i32 s8, 0, s6
	.long	0x8603FF03, 0xFFFFFF00	; s_and_b32 s3, s3, 0xffffff00
	.long	0x81820302	; s_sub_i32 s2, s2, s3
	.long	0x7E024701	; v_rcp_iflag_f32_e32 v1, v1
	.long	0xBE873002	; s_abs_i32 s7, s2
	.long	0x88030502	; s_xor_b32 s3, s2, s5
	.long	0x90039F03	; s_ashr_i32 s3, s3, 31
	.long	0x0A0202FF, 0x4F7FFFFE	; v_mul_f32_e32 v1, 0x4f7ffffe, v1
	.long	0x7E020F01	; v_cvt_u32_f32_e32 v1, v1
	.long	0x20060086	; v_lshrrev_b32_e32 v3, 6, v0
	.long	0x24040684	; v_lshlrev_b32_e32 v2, 4, v3
	.long	0x240C0084	; v_lshlrev_b32_e32 v6, 4, v0
	.long	0x7E120501	; v_readfirstlane_b32 s9, v1
	.long	0x92080908	; s_mul_i32 s8, s8, s9
	.long	0x96080809	; s_mul_hi_u32 s8, s9, s8
	.long	0x81090809	; s_add_i32 s9, s9, s8
	.long	0x96080907	; s_mul_hi_u32 s8, s7, s9
	.long	0x92090608	; s_mul_i32 s9, s8, s6
	.long	0x81870907	; s_sub_i32 s7, s7, s9
	.long	0x81098108	; s_add_i32 s9, s8, 1
	.long	0x818A0607	; s_sub_i32 s10, s7, s6
	.long	0xBF090607	; s_cmp_ge_u32 s7, s6
	.long	0x85080809	; s_cselect_b32 s8, s9, s8
	.long	0x8507070A	; s_cselect_b32 s7, s10, s7
	.long	0x81098108	; s_add_i32 s9, s8, 1
	.long	0xBF090607	; s_cmp_ge_u32 s7, s6
	.long	0x85060809	; s_cselect_b32 s6, s9, s8
	.long	0x88060306	; s_xor_b32 s6, s6, s3
	.long	0x260200A0	; v_and_b32_e32 v1, 32, v0
	.long	0x26080490	; v_and_b32_e32 v4, 16, v2
	.long	0x818A0306	; s_sub_i32 s10, s6, s3
	.long	0xD2340601, 0xC4060D04	; v_bitop3_b32 v1, v4, v6, v1 bitop3:0x36
	.long	0x9203050A	; s_mul_i32 s3, s10, s5
	.long	0x20020281	; v_lshrrev_b32_e32 v1, 1, v1
	.long	0x260404A0	; v_and_b32_e32 v2, 32, v2
	.long	0x81820302	; s_sub_i32 s2, s2, s3
	.long	0xD2010002, 0x04093101	; v_and_or_b32 v2, v1, 24, v2
	.long	0x20020083	; v_lshrrev_b32_e32 v1, 3, v0
	.long	0x81060204	; s_add_i32 s6, s4, s2
	.long	0x20080082	; v_lshrrev_b32_e32 v4, 2, v0
	.long	0x260A02A0	; v_and_b32_e32 v5, 32, v1
	.long	0xC00201C0, 0x00000020	; s_load_dword s7, s[0:1], 0x20
	.long	0xC0060080, 0x00000000	; s_load_dwordx2 s[2:3], s[0:1], 0x0
	.long	0xC0060200, 0x00000030	; s_load_dwordx2 s[8:9], s[0:1], 0x30
	.long	0xC00202C0, 0x00000050	; s_load_dword s11, s[0:1], 0x50
	.long	0xD2010007, 0x04153F04	; v_and_or_b32 v7, v4, 31, v5
	.long	0xBF8CC07F	; s_waitcnt lgkmcnt(0)
	.long	0xD1E80404, 0x04080F07	; v_mad_u64_u32 v[4:5], s[4:5], v7, s7, v[2:3]
	.long	0x8E048607	; s_lshl_b32 s4, s7, 6
	.long	0x25160881	; v_lshlrev_b32_e32 v139, 1, v4
	.long	0x68080804	; v_add_u32_e32 v4, s4, v4
	.long	0x251A0881	; v_lshlrev_b32_e32 v141, 1, v4
	.long	0x68080804	; v_add_u32_e32 v4, s4, v4
	.long	0x251C0881	; v_lshlrev_b32_e32 v142, 1, v4
	.long	0xD1FE008F, 0x02040904	; v_add_lshl_u32 v143, v4, s4, 1
	.long	0xD1E80404, 0x04081707	; v_mad_u64_u32 v[4:5], s[4:5], v7, s11, v[2:3]
	.long	0x8E04860B	; s_lshl_b32 s4, s11, 6
	.long	0xBF800000	; s_nop 0
	.long	0x68040804	; v_add_u32_e32 v2, s4, v4
	.long	0x25220481	; v_lshlrev_b32_e32 v145, 1, v2
	.long	0x68040404	; v_add_u32_e32 v2, s4, v2
	.long	0x8E108806	; s_lshl_b32 s16, s6, 8
	.long	0x25240481	; v_lshlrev_b32_e32 v146, 1, v2
	.long	0xD1FE0093, 0x02040902	; v_add_lshl_u32 v147, v2, s4, 1
	.long	0x96851007	; s_mul_hi_i32 s5, s7, s16
	.long	0x92041007	; s_mul_i32 s4, s7, s16
	.long	0x7E040280	; v_mov_b32_e32 v2, 0
	.long	0x25200881	; v_lshlrev_b32_e32 v144, 1, v4
	.long	0x8E848104	; s_lshl_b64 s[4:5], s[4:5], 1
	.long	0x26080CFF, 0x00001C00	; v_and_b32_e32 v4, 0x1c00, v6
	.long	0x7E0A0302	; v_mov_b32_e32 v5, v2
	.long	0x80040402	; s_add_u32 s4, s2, s4
	.long	0xD2080082, 0x04110012	; v_lshl_add_u64 v[130:131], s[18:19], 0, v[4:5]
	.long	0x82050503	; s_addc_u32 s5, s3, s5
	.long	0x8E028907	; s_lshl_b32 s2, s7, 9
	.long	0xBE8300FF, 0x00110000	; s_mov_b32 s3, 0x110000
	.long	0x7E180582	; v_readfirstlane_b32 s12, v130
	.long	0x680D04FF, 0x00002000	; v_add_u32_e32 v6, 0x2000, v130
	.long	0xBE860002	; s_mov_b32 s6, s2
	.long	0xBE870003	; s_mov_b32 s7, s3
	.long	0xBEFC000C	; s_mov_b32 m0, s12
	.long	0x7E180506	; v_readfirstlane_b32 s12, v6
	.long	0x680D04FF, 0x00004000	; v_add_u32_e32 v6, 0x4000, v130
	.long	0xE05D1000, 0x8001008B	; buffer_load_dwordx4 v139, s[4:7], 0 offen lds
	.long	0xBEFC000C	; s_mov_b32 m0, s12
	.long	0x7E180506	; v_readfirstlane_b32 s12, v6
	.long	0x680D04FF, 0x00006000	; v_add_u32_e32 v6, 0x6000, v130
	.long	0xE05D1000, 0x8001008D	; buffer_load_dwordx4 v141, s[4:7], 0 offen lds
	.long	0xBEFC000C	; s_mov_b32 m0, s12
	.long	0x7E180506	; v_readfirstlane_b32 s12, v6
	.long	0xE05D1000, 0x8001008E	; buffer_load_dwordx4 v142, s[4:7], 0 offen lds
	.long	0xBEFC000C	; s_mov_b32 m0, s12
	.long	0x8E18880A	; s_lshl_b32 s24, s10, 8
	.long	0xE05D1000, 0x8001008F	; buffer_load_dwordx4 v143, s[4:7], 0 offen lds
	.long	0x9687180B	; s_mul_hi_i32 s7, s11, s24
	.long	0x9206180B	; s_mul_i32 s6, s11, s24
	.long	0x8E868106	; s_lshl_b64 s[6:7], s[6:7], 1
	.long	0x80080608	; s_add_u32 s8, s8, s6
	.long	0xD2080084, 0x04110014	; v_lshl_add_u64 v[132:133], s[20:21], 0, v[4:5]
	.long	0x82090709	; s_addc_u32 s9, s9, s7
	.long	0x8E0E890B	; s_lshl_b32 s14, s11, 9
	.long	0x7E0C0584	; v_readfirstlane_b32 s6, v132
	.long	0x680908FF, 0x00002000	; v_add_u32_e32 v4, 0x2000, v132
	.long	0xBE8A000E	; s_mov_b32 s10, s14
	.long	0xBE8B0003	; s_mov_b32 s11, s3
	.long	0xBEFC0006	; s_mov_b32 m0, s6
	.long	0x7E0C0504	; v_readfirstlane_b32 s6, v4
	.long	0x680908FF, 0x00004000	; v_add_u32_e32 v4, 0x4000, v132
	.long	0xE05D1000, 0x80020090	; buffer_load_dwordx4 v144, s[8:11], 0 offen lds
	.long	0xBEFC0006	; s_mov_b32 m0, s6
	.long	0x7E0C0504	; v_readfirstlane_b32 s6, v4
	.long	0x680908FF, 0x00006000	; v_add_u32_e32 v4, 0x6000, v132
	.long	0xE05D1000, 0x80020091	; buffer_load_dwordx4 v145, s[8:11], 0 offen lds
	.long	0xBEFC0006	; s_mov_b32 m0, s6
	.long	0x7E0C0504	; v_readfirstlane_b32 s6, v4
	.long	0xE05D1000, 0x80020092	; buffer_load_dwordx4 v146, s[8:11], 0 offen lds
	.long	0xBEFC0006	; s_mov_b32 m0, s6
	.long	0x21060088	; v_lshrrev_b32_e32 v131, 8, v0
	.long	0xE05D1000, 0x80020093	; buffer_load_dwordx4 v147, s[8:11], 0 offen lds
	.long	0xC0060280, 0x00000060	; s_load_dwordx2 s[10:11], s[0:1], 0x60
	.long	0xC0060180, 0x00000080	; s_load_dwordx2 s[6:7], s[0:1], 0x80
	.long	0xBF8CC07F	; s_waitcnt lgkmcnt(0)
	.long	0xBE870081	; s_mov_b32 s7, 1
	.long	0x7D950681	; v_cmp_eq_u32_e32 vcc, 1, v131
	.long	0xBF8C0000	; s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	.long	0xBF8A0000	; s_barrier
	.long	0xBE80206A	; s_and_saveexec_b64 s[0:1], vcc
	.long	0xBF880001	; s_cbranch_execz 1
	.long	0xBF8A0000	; s_barrier
	.long	0x87FE007E	; s_or_b64 exec, exec, s[0:1]
	.long	0x270A0683	; v_and_b32_e32 v133, 3, v3
	.long	0x24060086	; v_lshlrev_b32_e32 v3, 6, v0
	.long	0x20080081	; v_lshrrev_b32_e32 v4, 1, v0
	.long	0x260606FF, 0x000007C0	; v_and_b32_e32 v3, 0x7c0, v3
	.long	0x26080890	; v_and_b32_e32 v4, 16, v4
	.long	0x240C0082	; v_lshlrev_b32_e32 v6, 2, v0
	.long	0x260E0090	; v_and_b32_e32 v7, 16, v0
	.long	0x280A0704	; v_or_b32_e32 v5, v4, v3
	.long	0xD2010006, 0x041D4106	; v_and_or_b32 v6, v6, 32, v7
	.long	0xD2340695, 0xC40E0D04	; v_bitop3_b32 v149, v4, v6, v3 bitop3:0x36
	.long	0xD2340694, 0xC2820D05	; v_bitop3_b32 v148, v5, v6, 32 bitop3:0x36
	.long	0x2409068E	; v_lshlrev_b32_e32 v4, 14, v131
	.long	0x7E0A0302	; v_mov_b32_e32 v5, v2
	.long	0xD2080086, 0x04110012	; v_lshl_add_u64 v[134:135], s[18:19], 0, v[4:5]
	.long	0x24090A8D	; v_lshlrev_b32_e32 v4, 13, v133
	.long	0xD2080088, 0x04110014	; v_lshl_add_u64 v[136:137], s[20:21], 0, v[4:5]
	.long	0xBE9601FF, 0x00000080	; s_mov_b64 s[22:23], 0x80
	.long	0x7E060302	; v_mov_b32_e32 v3, v2
	.long	0x7E080302	; v_mov_b32_e32 v4, v2
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
	.long	0x2519068D	; v_lshlrev_b32_e32 v140, 13, v131
	.long	0x25150A8C	; v_lshlrev_b32_e32 v138, 12, v133
	.long	0x8E0C8F11	; s_lshl_b32 s12, s17, 15
	.long	0x690F0C0C	; v_add_u32_e32 v135, s12, v134
	.long	0x69130F95	; v_add_u32_e32 v137, v149, v135
	.long	0xD9FE0000, 0x96000089	; ds_read_b128 v[150:153], v137
	.long	0xD9FE1000, 0x9A000089	; ds_read_b128 v[154:157], v137 offset:4096
	.long	0xD9FE2000, 0x9E000089	; ds_read_b128 v[158:161], v137 offset:8192
	.long	0xD9FE3000, 0xA2000089	; ds_read_b128 v[162:165], v137 offset:12288
	.long	0x69130F94	; v_add_u32_e32 v137, v148, v135
	.long	0xD9FE0000, 0xA6000089	; ds_read_b128 v[166:169], v137
	.long	0xD9FE1000, 0xAA000089	; ds_read_b128 v[170:173], v137 offset:4096
	.long	0x80001604	; s_add_u32 s0, s4, s22
	.long	0xD9FE2000, 0xAE000089	; ds_read_b128 v[174:177], v137 offset:8192
	.long	0x82011705	; s_addc_u32 s1, s5, s23
	.long	0x8E138F07	; s_lshl_b32 s19, s7, 15
	.long	0xD9FE3000, 0xB2000089	; ds_read_b128 v[178:181], v137 offset:12288
	.long	0x69130413	; v_add_u32_e32 v137, s19, v130
	.long	0x696D12FF, 0x00002000	; v_add_u32_e32 v182, 0x2000, v137
	.long	0x7E1A0589	; v_readfirstlane_b32 s13, v137
	.long	0xBEFC000D	; s_mov_b32 m0, s13
	.long	0x7E1A05B6	; v_readfirstlane_b32 s13, v182
	.long	0x696D12FF, 0x00004000	; v_add_u32_e32 v182, 0x4000, v137
	.long	0xE05D1000, 0x8000008B	; buffer_load_dwordx4 v139, s[0:3], 0 offen lds
	.long	0xBEFC000D	; s_mov_b32 m0, s13
	.long	0x7E1A05B6	; v_readfirstlane_b32 s13, v182
	.long	0x691312FF, 0x00006000	; v_add_u32_e32 v137, 0x6000, v137
	.long	0xE05D1000, 0x8000008D	; buffer_load_dwordx4 v141, s[0:3], 0 offen lds
	.long	0xBEFC000D	; s_mov_b32 m0, s13
	.long	0x7E1A0589	; v_readfirstlane_b32 s13, v137
	.long	0xE05D1000, 0x8000008E	; buffer_load_dwordx4 v142, s[0:3], 0 offen lds
	.long	0xBEFC000D	; s_mov_b32 m0, s13
	.long	0x6913100C	; v_add_u32_e32 v137, s12, v136
	.long	0xE05D1000, 0x8000008F	; buffer_load_dwordx4 v143, s[0:3], 0 offen lds
	.long	0x69751395	; v_add_u32_e32 v186, v149, v137
	.long	0xD9FE0000, 0xB60000BA	; ds_read_b128 v[182:185], v186
	.long	0xD9FE1000, 0xBA0000BA	; ds_read_b128 v[186:189], v186 offset:4096
	.long	0x698D0813	; v_add_u32_e32 v198, s19, v132
	.long	0x69851394	; v_add_u32_e32 v194, v148, v137
	.long	0xD9FE0000, 0xBE0000C2	; ds_read_b128 v[190:193], v194
	.long	0x800C1608	; s_add_u32 s12, s8, s22
	.long	0x7E0005C6	; v_readfirstlane_b32 s0, v198
	.long	0x698F8CFF, 0x00002000	; v_add_u32_e32 v199, 0x2000, v198
	.long	0xD9FE1000, 0xC20000C2	; ds_read_b128 v[194:197], v194 offset:4096
	.long	0x820D1709	; s_addc_u32 s13, s9, s23
	.long	0xBE8F0003	; s_mov_b32 s15, s3
	.long	0xBEFC0000	; s_mov_b32 m0, s0
	.long	0x7E0005C7	; v_readfirstlane_b32 s0, v199
	.long	0x698F8CFF, 0x00004000	; v_add_u32_e32 v199, 0x4000, v198
	.long	0xE05D1000, 0x80030090	; buffer_load_dwordx4 v144, s[12:15], 0 offen lds
	.long	0xBEFC0000	; s_mov_b32 m0, s0
	.long	0x7E0005C7	; v_readfirstlane_b32 s0, v199
	.long	0x698D8CFF, 0x00006000	; v_add_u32_e32 v198, 0x6000, v198
	.long	0xE05D1000, 0x80030091	; buffer_load_dwordx4 v145, s[12:15], 0 offen lds
	.long	0xBEFC0000	; s_mov_b32 m0, s0
	.long	0x7E0005C6	; v_readfirstlane_b32 s0, v198
	.long	0xE05D1000, 0x80030092	; buffer_load_dwordx4 v146, s[12:15], 0 offen lds
	.long	0xBEFC0000	; s_mov_b32 m0, s0
	.long	0xBF800000	; s_nop 0
	.long	0xE05D1000, 0x80030093	; buffer_load_dwordx4 v147, s[12:15], 0 offen lds
	.long	0xBF8A0000	; s_barrier
	.long	0xBF8CC07F	; s_waitcnt lgkmcnt(0)
	.long	0xBF8F0001	; s_setprio 1
	.long	0xD3B70072, 0x05CB6D96	; v_mfma_f32_32x32x16_bf16 v[114:129], v[150:153], v[182:185], v[114:129]
	.long	0xD3B70062, 0x058B7596	; v_mfma_f32_32x32x16_bf16 v[98:113], v[150:153], v[186:189], v[98:113]
	.long	0xD3B70052, 0x054B6D9A	; v_mfma_f32_32x32x16_bf16 v[82:97], v[154:157], v[182:185], v[82:97]
	.long	0xD3B70042, 0x050B759A	; v_mfma_f32_32x32x16_bf16 v[66:81], v[154:157], v[186:189], v[66:81]
	.long	0xD3B70032, 0x04CB6D9E	; v_mfma_f32_32x32x16_bf16 v[50:65], v[158:161], v[182:185], v[50:65]
	.long	0xD3B70022, 0x048B759E	; v_mfma_f32_32x32x16_bf16 v[34:49], v[158:161], v[186:189], v[34:49]
	.long	0xD3B70012, 0x044B6DA2	; v_mfma_f32_32x32x16_bf16 v[18:33], v[162:165], v[182:185], v[18:33]
	.long	0xD3B70002, 0x040B75A2	; v_mfma_f32_32x32x16_bf16 v[2:17], v[162:165], v[186:189], v[2:17]
	.long	0xD3B70072, 0x05CB7DA6	; v_mfma_f32_32x32x16_bf16 v[114:129], v[166:169], v[190:193], v[114:129]
	.long	0xD3B70062, 0x058B85A6	; v_mfma_f32_32x32x16_bf16 v[98:113], v[166:169], v[194:197], v[98:113]
	.long	0xD3B70052, 0x054B7DAA	; v_mfma_f32_32x32x16_bf16 v[82:97], v[170:173], v[190:193], v[82:97]
	.long	0xD3B70042, 0x050B85AA	; v_mfma_f32_32x32x16_bf16 v[66:81], v[170:173], v[194:197], v[66:81]
	.long	0xD3B70032, 0x04CB7DAE	; v_mfma_f32_32x32x16_bf16 v[50:65], v[174:177], v[190:193], v[50:65]
	.long	0xD3B70022, 0x048B85AE	; v_mfma_f32_32x32x16_bf16 v[34:49], v[174:177], v[194:197], v[34:49]
	.long	0xD3B70012, 0x044B7DB2	; v_mfma_f32_32x32x16_bf16 v[18:33], v[178:181], v[190:193], v[18:33]
	.long	0xD3B70002, 0x040B85B2	; v_mfma_f32_32x32x16_bf16 v[2:17], v[178:181], v[194:197], v[2:17]
	.long	0xBF8F0000	; s_setprio 0
	.long	0x690F0EFF, 0x00000800	; v_add_u32_e32 v135, 0x800, v135
	.long	0xBF8A0000	; s_barrier
	.long	0x69450F95	; v_add_u32_e32 v162, v149, v135
	.long	0xD9FE0000, 0x960000A2	; ds_read_b128 v[150:153], v162
	.long	0xD9FE1000, 0x9A0000A2	; ds_read_b128 v[154:157], v162 offset:4096
	.long	0xD9FE2000, 0x9E0000A2	; ds_read_b128 v[158:161], v162 offset:8192
	.long	0xD9FE3000, 0xA20000A2	; ds_read_b128 v[162:165], v162 offset:12288
	.long	0x690F0F94	; v_add_u32_e32 v135, v148, v135
	.long	0xD9FE0000, 0xA6000087	; ds_read_b128 v[166:169], v135
	.long	0xD9FE1000, 0xAA000087	; ds_read_b128 v[170:173], v135 offset:4096
	.long	0xD9FE2000, 0xAE000087	; ds_read_b128 v[174:177], v135 offset:8192
	.long	0xD9FE3000, 0xB2000087	; ds_read_b128 v[178:181], v135 offset:12288
	.long	0x690F12FF, 0x00000800	; v_add_u32_e32 v135, 0x800, v137
	.long	0x69130F95	; v_add_u32_e32 v137, v149, v135
	.long	0xD9FE0000, 0xB6000089	; ds_read_b128 v[182:185], v137
	.long	0xD9FE1000, 0xBA000089	; ds_read_b128 v[186:189], v137 offset:4096
	.long	0x690F0F94	; v_add_u32_e32 v135, v148, v135
	.long	0xD9FE0000, 0xBE000087	; ds_read_b128 v[190:193], v135
	.long	0xD9FE1000, 0xC2000087	; ds_read_b128 v[194:197], v135 offset:4096
	.long	0xBF8C0000	; s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	.long	0xBF8A0000	; s_barrier
	.long	0xBF8CC07F	; s_waitcnt lgkmcnt(0)
	.long	0xBF8F0001	; s_setprio 1
	.long	0xD3B70072, 0x05CB6D96	; v_mfma_f32_32x32x16_bf16 v[114:129], v[150:153], v[182:185], v[114:129]
	.long	0xD3B70062, 0x058B7596	; v_mfma_f32_32x32x16_bf16 v[98:113], v[150:153], v[186:189], v[98:113]
	.long	0xD3B70052, 0x054B6D9A	; v_mfma_f32_32x32x16_bf16 v[82:97], v[154:157], v[182:185], v[82:97]
	.long	0xD3B70042, 0x050B759A	; v_mfma_f32_32x32x16_bf16 v[66:81], v[154:157], v[186:189], v[66:81]
	.long	0xD3B70032, 0x04CB6D9E	; v_mfma_f32_32x32x16_bf16 v[50:65], v[158:161], v[182:185], v[50:65]
	.long	0xD3B70022, 0x048B759E	; v_mfma_f32_32x32x16_bf16 v[34:49], v[158:161], v[186:189], v[34:49]
	.long	0xD3B70012, 0x044B6DA2	; v_mfma_f32_32x32x16_bf16 v[18:33], v[162:165], v[182:185], v[18:33]
	.long	0xD3B70002, 0x040B75A2	; v_mfma_f32_32x32x16_bf16 v[2:17], v[162:165], v[186:189], v[2:17]
	.long	0xD3B70072, 0x05CB7DA6	; v_mfma_f32_32x32x16_bf16 v[114:129], v[166:169], v[190:193], v[114:129]
	.long	0xD3B70062, 0x058B85A6	; v_mfma_f32_32x32x16_bf16 v[98:113], v[166:169], v[194:197], v[98:113]
	.long	0xD3B70052, 0x054B7DAA	; v_mfma_f32_32x32x16_bf16 v[82:97], v[170:173], v[190:193], v[82:97]
	.long	0xD3B70042, 0x050B85AA	; v_mfma_f32_32x32x16_bf16 v[66:81], v[170:173], v[194:197], v[66:81]
	.long	0xD3B70032, 0x04CB7DAE	; v_mfma_f32_32x32x16_bf16 v[50:65], v[174:177], v[190:193], v[50:65]
	.long	0xD3B70022, 0x048B85AE	; v_mfma_f32_32x32x16_bf16 v[34:49], v[174:177], v[194:197], v[34:49]
	.long	0xD3B70012, 0x044B7DB2	; v_mfma_f32_32x32x16_bf16 v[18:33], v[178:181], v[190:193], v[18:33]
	.long	0xD3B70002, 0x040B85B2	; v_mfma_f32_32x32x16_bf16 v[2:17], v[178:181], v[194:197], v[2:17]
	.long	0xBF8F0000	; s_setprio 0
	.long	0x88118111	; s_xor_b32 s17, s17, 1
	.long	0x88078107	; s_xor_b32 s7, s7, 1
	.long	0x8016FF16, 0x00000080	; s_add_u32 s22, s22, 0x80
	.long	0x82178017	; s_addc_u32 s23, s23, 0
	.long	0xB1164000	; s_cmpk_eq_i32 s22, 0x4000
	.long	0xBF8A0000	; s_barrier
	.long	0xBF84FF3A	; s_cbranch_scc0 65338
	.long	0x8E008F11	; s_lshl_b32 s0, s17, 15
	.long	0x81010012	; s_add_i32 s1, s18, s0
	.long	0xD1FD0082, 0x0005038C	; v_lshl_add_u32 v130, v140, 1, s1
	.long	0x69090595	; v_add_u32_e32 v132, v149, v130
	.long	0xD9FE0000, 0x86000084	; ds_read_b128 v[134:137], v132
	.long	0xD9FE1000, 0x8C000084	; ds_read_b128 v[140:143], v132 offset:4096
	.long	0xD9FE2000, 0x90000084	; ds_read_b128 v[144:147], v132 offset:8192
	.long	0xD9FE3000, 0x96000084	; ds_read_b128 v[150:153], v132 offset:12288
	.long	0x69090594	; v_add_u32_e32 v132, v148, v130
	.long	0xD9FE0000, 0x9A000084	; ds_read_b128 v[154:157], v132
	.long	0xD9FE1000, 0x9E000084	; ds_read_b128 v[158:161], v132 offset:4096
	.long	0xD9FE2000, 0xA2000084	; ds_read_b128 v[162:165], v132 offset:8192
	.long	0x81000014	; s_add_i32 s0, s20, s0
	.long	0xD9FE3000, 0xA6000084	; ds_read_b128 v[166:169], v132 offset:12288
	.long	0xD1FD0084, 0x0001038A	; v_lshl_add_u32 v132, v138, 1, s0
	.long	0x69150995	; v_add_u32_e32 v138, v149, v132
	.long	0xD9FE0000, 0xAA00008A	; ds_read_b128 v[170:173], v138
	.long	0xD9FE1000, 0xAE00008A	; ds_read_b128 v[174:177], v138 offset:4096
	.long	0x69150994	; v_add_u32_e32 v138, v148, v132
	.long	0xD9FE0000, 0xB200008A	; ds_read_b128 v[178:181], v138
	.long	0xD9FE1000, 0xB600008A	; ds_read_b128 v[182:185], v138 offset:4096
	.long	0xBF8A0000	; s_barrier
	.long	0xBF8CC07F	; s_waitcnt lgkmcnt(0)
	.long	0xBF8F0001	; s_setprio 1
	.long	0xD3B70072, 0x05CB5586	; v_mfma_f32_32x32x16_bf16 v[114:129], v[134:137], v[170:173], v[114:129]
	.long	0xD3B70062, 0x058B5D86	; v_mfma_f32_32x32x16_bf16 v[98:113], v[134:137], v[174:177], v[98:113]
	.long	0xD3B70052, 0x054B558C	; v_mfma_f32_32x32x16_bf16 v[82:97], v[140:143], v[170:173], v[82:97]
	.long	0xD3B70042, 0x050B5D8C	; v_mfma_f32_32x32x16_bf16 v[66:81], v[140:143], v[174:177], v[66:81]
	.long	0xD3B70032, 0x04CB5590	; v_mfma_f32_32x32x16_bf16 v[50:65], v[144:147], v[170:173], v[50:65]
	.long	0xD3B70022, 0x048B5D90	; v_mfma_f32_32x32x16_bf16 v[34:49], v[144:147], v[174:177], v[34:49]
	.long	0xD3B70012, 0x044B5596	; v_mfma_f32_32x32x16_bf16 v[18:33], v[150:153], v[170:173], v[18:33]
	.long	0xD3B70002, 0x040B5D96	; v_mfma_f32_32x32x16_bf16 v[2:17], v[150:153], v[174:177], v[2:17]
	.long	0xD3B70072, 0x05CB659A	; v_mfma_f32_32x32x16_bf16 v[114:129], v[154:157], v[178:181], v[114:129]
	.long	0xD3B70062, 0x058B6D9A	; v_mfma_f32_32x32x16_bf16 v[98:113], v[154:157], v[182:185], v[98:113]
	.long	0xD3B70052, 0x054B659E	; v_mfma_f32_32x32x16_bf16 v[82:97], v[158:161], v[178:181], v[82:97]
	.long	0xD3B70042, 0x050B6D9E	; v_mfma_f32_32x32x16_bf16 v[66:81], v[158:161], v[182:185], v[66:81]
	.long	0xD3B70032, 0x04CB65A2	; v_mfma_f32_32x32x16_bf16 v[50:65], v[162:165], v[178:181], v[50:65]
	.long	0xD3B70022, 0x048B6DA2	; v_mfma_f32_32x32x16_bf16 v[34:49], v[162:165], v[182:185], v[34:49]
	.long	0xD3B70012, 0x044B65A6	; v_mfma_f32_32x32x16_bf16 v[18:33], v[166:169], v[178:181], v[18:33]
	.long	0xD3B70002, 0x040B6DA6	; v_mfma_f32_32x32x16_bf16 v[2:17], v[166:169], v[182:185], v[2:17]
	.long	0xBF8F0000	; s_setprio 0
	.long	0x690504FF, 0x00000800	; v_add_u32_e32 v130, 0x800, v130
	.long	0xBF8A0000	; s_barrier
	.long	0x69250595	; v_add_u32_e32 v146, v149, v130
	.long	0xD9FE0000, 0x86000092	; ds_read_b128 v[134:137], v146
	.long	0xD9FE1000, 0x8A000092	; ds_read_b128 v[138:141], v146 offset:4096
	.long	0xD9FE2000, 0x8E000092	; ds_read_b128 v[142:145], v146 offset:8192
	.long	0xD9FE3000, 0x96000092	; ds_read_b128 v[150:153], v146 offset:12288
	.long	0x69050594	; v_add_u32_e32 v130, v148, v130
	.long	0xD9FE0000, 0x9A000082	; ds_read_b128 v[154:157], v130
	.long	0xD9FE1000, 0x9E000082	; ds_read_b128 v[158:161], v130 offset:4096
	.long	0xD9FE2000, 0xA2000082	; ds_read_b128 v[162:165], v130 offset:8192
	.long	0xD9FE3000, 0xA6000082	; ds_read_b128 v[166:169], v130 offset:12288
	.long	0x690508FF, 0x00000800	; v_add_u32_e32 v130, 0x800, v132
	.long	0x69090595	; v_add_u32_e32 v132, v149, v130
	.long	0xD9FE0000, 0xAA000084	; ds_read_b128 v[170:173], v132
	.long	0xD9FE1000, 0xAE000084	; ds_read_b128 v[174:177], v132 offset:4096
	.long	0x69050594	; v_add_u32_e32 v130, v148, v130
	.long	0xD9FE0000, 0x92000082	; ds_read_b128 v[146:149], v130
	.long	0xD9FE1000, 0xB2000082	; ds_read_b128 v[178:181], v130 offset:4096
	.long	0xBF8A0000	; s_barrier
	.long	0xBF8CC07F	; s_waitcnt lgkmcnt(0)
	.long	0xBF8F0001	; s_setprio 1
	.long	0xD3B70072, 0x05CB5586	; v_mfma_f32_32x32x16_bf16 v[114:129], v[134:137], v[170:173], v[114:129]
	.long	0xD3B70062, 0x058B5D86	; v_mfma_f32_32x32x16_bf16 v[98:113], v[134:137], v[174:177], v[98:113]
	.long	0xD3B70052, 0x054B558A	; v_mfma_f32_32x32x16_bf16 v[82:97], v[138:141], v[170:173], v[82:97]
	.long	0xD3B70042, 0x050B5D8A	; v_mfma_f32_32x32x16_bf16 v[66:81], v[138:141], v[174:177], v[66:81]
	.long	0xD3B70032, 0x04CB558E	; v_mfma_f32_32x32x16_bf16 v[50:65], v[142:145], v[170:173], v[50:65]
	.long	0xD3B70022, 0x048B5D8E	; v_mfma_f32_32x32x16_bf16 v[34:49], v[142:145], v[174:177], v[34:49]
	.long	0xD3B70012, 0x044B5596	; v_mfma_f32_32x32x16_bf16 v[18:33], v[150:153], v[170:173], v[18:33]
	.long	0xD3B70002, 0x040B5D96	; v_mfma_f32_32x32x16_bf16 v[2:17], v[150:153], v[174:177], v[2:17]
	.long	0xD3B70072, 0x05CB259A	; v_mfma_f32_32x32x16_bf16 v[114:129], v[154:157], v[146:149], v[114:129]
	.long	0xD3B70062, 0x058B659A	; v_mfma_f32_32x32x16_bf16 v[98:113], v[154:157], v[178:181], v[98:113]
	.long	0xD3B70052, 0x054B259E	; v_mfma_f32_32x32x16_bf16 v[82:97], v[158:161], v[146:149], v[82:97]
	.long	0xD3B70042, 0x050B659E	; v_mfma_f32_32x32x16_bf16 v[66:81], v[158:161], v[178:181], v[66:81]
	.long	0xD3B70032, 0x04CB25A2	; v_mfma_f32_32x32x16_bf16 v[50:65], v[162:165], v[146:149], v[50:65]
	.long	0xD3B70022, 0x048B65A2	; v_mfma_f32_32x32x16_bf16 v[34:49], v[162:165], v[178:181], v[34:49]
	.long	0xD3B70012, 0x044B25A6	; v_mfma_f32_32x32x16_bf16 v[18:33], v[166:169], v[146:149], v[18:33]
	.long	0xD3B70002, 0x040B65A6	; v_mfma_f32_32x32x16_bf16 v[2:17], v[166:169], v[178:181], v[2:17]
	.long	0xBF8F0000	; s_setprio 0
	.long	0xD1FD0083, 0x00410F83	; v_lshl_add_u32 v131, v131, 7, s16
	.long	0xD2000082, 0x00610D85	; v_lshl_or_b32 v130, v133, 6, s24
	.long	0xD1E90084, 0x02030606	; v_mad_i64_i32 v[132:133], s[0:1], s6, v131, 0
	.long	0x2307049F	; v_ashrrev_i32_e32 v131, 31, v130
	.long	0xD2080084, 0x00290384	; v_lshl_add_u64 v[132:133], v[132:133], 1, s[10:11]
	.long	0x272C0284	; v_and_b32_e32 v150, 4, v1
	.long	0x2600009F	; v_and_b32_e32 v0, 31, v0
	.long	0xD2080082, 0x06110382	; v_lshl_add_u64 v[130:131], v[130:131], 1, v[132:133]
	.long	0xD1E80084, 0x04000D96	; v_mad_u64_u32 v[132:133], s[0:1], v150, s6, v[0:1]
	.long	0x230B089F	; v_ashrrev_i32_e32 v133, 31, v132
	.long	0xD2080086, 0x06090384	; v_lshl_add_u64 v[134:135], v[132:133], 1, v[130:131]
	.long	0x69090806	; v_add_u32_e32 v132, s6, v132
	.long	0x230B089F	; v_ashrrev_i32_e32 v133, 31, v132
	.long	0xBF8A0000	; s_barrier
	.long	0xDC6C8000, 0x007F7286	; global_store_short_d16_hi v[134:135], v114, off
	.long	0xD2080088, 0x06090384	; v_lshl_add_u64 v[136:137], v[132:133], 1, v[130:131]
	.long	0x68E50806	; v_add_u32_e32 v114, s6, v132
	.long	0xDC6C8000, 0x007F7388	; global_store_short_d16_hi v[136:137], v115, off
	.long	0x22E6E49F	; v_ashrrev_i32_e32 v115, 31, v114
	.long	0xD2080084, 0x06090372	; v_lshl_add_u64 v[132:133], v[114:115], 1, v[130:131]
	.long	0x68E4E406	; v_add_u32_e32 v114, s6, v114
	.long	0x22E6E49F	; v_ashrrev_i32_e32 v115, 31, v114
	.long	0x92008506	; s_mul_i32 s0, s6, 5
	.long	0xD208008A, 0x06090372	; v_lshl_add_u64 v[138:139], v[114:115], 1, v[130:131]
	.long	0x68E4E400	; v_add_u32_e32 v114, s0, v114
	.long	0x22E6E49F	; v_ashrrev_i32_e32 v115, 31, v114
	.long	0xDC6C8000, 0x007F7484	; global_store_short_d16_hi v[132:133], v116, off
	.long	0xDC6C8000, 0x007F758A	; global_store_short_d16_hi v[138:139], v117, off
	.long	0xD2080074, 0x06090372	; v_lshl_add_u64 v[116:117], v[114:115], 1, v[130:131]
	.long	0x68E4E406	; v_add_u32_e32 v114, s6, v114
	.long	0x22E6E49F	; v_ashrrev_i32_e32 v115, 31, v114
	.long	0xD208008C, 0x06090372	; v_lshl_add_u64 v[140:141], v[114:115], 1, v[130:131]
	.long	0x68E4E406	; v_add_u32_e32 v114, s6, v114
	.long	0x22E6E49F	; v_ashrrev_i32_e32 v115, 31, v114
	.long	0xDC6C8000, 0x007F7674	; global_store_short_d16_hi v[116:117], v118, off
	.long	0xDC6C8000, 0x007F778C	; global_store_short_d16_hi v[140:141], v119, off
	.long	0xD2080076, 0x06090372	; v_lshl_add_u64 v[118:119], v[114:115], 1, v[130:131]
	.long	0x68E4E406	; v_add_u32_e32 v114, s6, v114
	.long	0x22E6E49F	; v_ashrrev_i32_e32 v115, 31, v114
	.long	0xD208008E, 0x06090372	; v_lshl_add_u64 v[142:143], v[114:115], 1, v[130:131]
	.long	0x68E4E400	; v_add_u32_e32 v114, s0, v114
	.long	0x22E6E49F	; v_ashrrev_i32_e32 v115, 31, v114
	.long	0xDC6C8000, 0x007F7876	; global_store_short_d16_hi v[118:119], v120, off
	.long	0xDC6C8000, 0x007F798E	; global_store_short_d16_hi v[142:143], v121, off
	.long	0xD2080078, 0x06090372	; v_lshl_add_u64 v[120:121], v[114:115], 1, v[130:131]
	.long	0x68E4E406	; v_add_u32_e32 v114, s6, v114
	.long	0x22E6E49F	; v_ashrrev_i32_e32 v115, 31, v114
	.long	0xD2080090, 0x06090372	; v_lshl_add_u64 v[144:145], v[114:115], 1, v[130:131]
	.long	0x68E4E406	; v_add_u32_e32 v114, s6, v114
	.long	0x22E6E49F	; v_ashrrev_i32_e32 v115, 31, v114
	.long	0xDC6C8000, 0x007F7A78	; global_store_short_d16_hi v[120:121], v122, off
	.long	0xDC6C8000, 0x007F7B90	; global_store_short_d16_hi v[144:145], v123, off
	.long	0xD208007A, 0x06090372	; v_lshl_add_u64 v[122:123], v[114:115], 1, v[130:131]
	.long	0x68E4E406	; v_add_u32_e32 v114, s6, v114
	.long	0x22E6E49F	; v_ashrrev_i32_e32 v115, 31, v114
	.long	0xD2080092, 0x06090372	; v_lshl_add_u64 v[146:147], v[114:115], 1, v[130:131]
	.long	0x68E4E400	; v_add_u32_e32 v114, s0, v114
	.long	0x22E6E49F	; v_ashrrev_i32_e32 v115, 31, v114
	.long	0xDC6C8000, 0x007F7C7A	; global_store_short_d16_hi v[122:123], v124, off
	.long	0xDC6C8000, 0x007F7D92	; global_store_short_d16_hi v[146:147], v125, off
	.long	0xD208007C, 0x06090372	; v_lshl_add_u64 v[124:125], v[114:115], 1, v[130:131]
	.long	0x68E4E406	; v_add_u32_e32 v114, s6, v114
	.long	0x22E6E49F	; v_ashrrev_i32_e32 v115, 31, v114
	.long	0xD2080094, 0x06090372	; v_lshl_add_u64 v[148:149], v[114:115], 1, v[130:131]
	.long	0x68E4E406	; v_add_u32_e32 v114, s6, v114
	.long	0x22E6E49F	; v_ashrrev_i32_e32 v115, 31, v114
	.long	0xDC6C8000, 0x007F7E7C	; global_store_short_d16_hi v[124:125], v126, off
	.long	0xDC6C8000, 0x007F7F94	; global_store_short_d16_hi v[148:149], v127, off
	.long	0xD208007E, 0x06090372	; v_lshl_add_u64 v[126:127], v[114:115], 1, v[130:131]
	.long	0x68E4E406	; v_add_u32_e32 v114, s6, v114
	.long	0x22E6E49F	; v_ashrrev_i32_e32 v115, 31, v114
	.long	0xD2080072, 0x06090372	; v_lshl_add_u64 v[114:115], v[114:115], 1, v[130:131]
	.long	0xDC6C8000, 0x007F807E	; global_store_short_d16_hi v[126:127], v128, off
	.long	0xDC6C8000, 0x007F8172	; global_store_short_d16_hi v[114:115], v129, off
	.long	0xDC6C8040, 0x007F6286	; global_store_short_d16_hi v[134:135], v98, off offset:64
	.long	0xDC6C8040, 0x007F6388	; global_store_short_d16_hi v[136:137], v99, off offset:64
	.long	0xDC6C8040, 0x007F6484	; global_store_short_d16_hi v[132:133], v100, off offset:64
	.long	0xDC6C8040, 0x007F658A	; global_store_short_d16_hi v[138:139], v101, off offset:64
	.long	0xDC6C8040, 0x007F6674	; global_store_short_d16_hi v[116:117], v102, off offset:64
	.long	0xDC6C8040, 0x007F678C	; global_store_short_d16_hi v[140:141], v103, off offset:64
	.long	0xDC6C8040, 0x007F6876	; global_store_short_d16_hi v[118:119], v104, off offset:64
	.long	0xDC6C8040, 0x007F698E	; global_store_short_d16_hi v[142:143], v105, off offset:64
	.long	0xDC6C8040, 0x007F6A78	; global_store_short_d16_hi v[120:121], v106, off offset:64
	.long	0xDC6C8040, 0x007F6B90	; global_store_short_d16_hi v[144:145], v107, off offset:64
	.long	0xDC6C8040, 0x007F6C7A	; global_store_short_d16_hi v[122:123], v108, off offset:64
	.long	0xDC6C8040, 0x007F6D92	; global_store_short_d16_hi v[146:147], v109, off offset:64
	.long	0xDC6C8040, 0x007F6E7C	; global_store_short_d16_hi v[124:125], v110, off offset:64
	.long	0xDC6C8040, 0x007F6F94	; global_store_short_d16_hi v[148:149], v111, off offset:64
	.long	0xDC6C8040, 0x007F707E	; global_store_short_d16_hi v[126:127], v112, off offset:64
	.long	0xDC6C8040, 0x007F7172	; global_store_short_d16_hi v[114:115], v113, off offset:64
	.long	0x28C52CA0	; v_or_b32_e32 v98, 32, v150
	.long	0xD1E80262, 0x04000D62	; v_mad_u64_u32 v[98:99], s[2:3], v98, s6, v[0:1]
	.long	0x22C6C49F	; v_ashrrev_i32_e32 v99, 31, v98
	.long	0xD2080064, 0x06090362	; v_lshl_add_u64 v[100:101], v[98:99], 1, v[130:131]
	.long	0x68C4C406	; v_add_u32_e32 v98, s6, v98
	.long	0x22C6C49F	; v_ashrrev_i32_e32 v99, 31, v98
	.long	0xDC6C8000, 0x007F5264	; global_store_short_d16_hi v[100:101], v82, off
	.long	0xD2080066, 0x06090362	; v_lshl_add_u64 v[102:103], v[98:99], 1, v[130:131]
	.long	0x68A4C406	; v_add_u32_e32 v82, s6, v98
	.long	0xDC6C8000, 0x007F5366	; global_store_short_d16_hi v[102:103], v83, off
	.long	0x22A6A49F	; v_ashrrev_i32_e32 v83, 31, v82
	.long	0xD2080062, 0x06090352	; v_lshl_add_u64 v[98:99], v[82:83], 1, v[130:131]
	.long	0x68A4A406	; v_add_u32_e32 v82, s6, v82
	.long	0x22A6A49F	; v_ashrrev_i32_e32 v83, 31, v82
	.long	0xD2080068, 0x06090352	; v_lshl_add_u64 v[104:105], v[82:83], 1, v[130:131]
	.long	0x68A4A400	; v_add_u32_e32 v82, s0, v82
	.long	0x22A6A49F	; v_ashrrev_i32_e32 v83, 31, v82
	.long	0xDC6C8000, 0x007F5462	; global_store_short_d16_hi v[98:99], v84, off
	.long	0xDC6C8000, 0x007F5568	; global_store_short_d16_hi v[104:105], v85, off
	.long	0xD2080054, 0x06090352	; v_lshl_add_u64 v[84:85], v[82:83], 1, v[130:131]
	.long	0x68A4A406	; v_add_u32_e32 v82, s6, v82
	.long	0x22A6A49F	; v_ashrrev_i32_e32 v83, 31, v82
	.long	0xD208006A, 0x06090352	; v_lshl_add_u64 v[106:107], v[82:83], 1, v[130:131]
	.long	0x68A4A406	; v_add_u32_e32 v82, s6, v82
	.long	0x22A6A49F	; v_ashrrev_i32_e32 v83, 31, v82
	.long	0xDC6C8000, 0x007F5654	; global_store_short_d16_hi v[84:85], v86, off
	.long	0xDC6C8000, 0x007F576A	; global_store_short_d16_hi v[106:107], v87, off
	.long	0xD2080056, 0x06090352	; v_lshl_add_u64 v[86:87], v[82:83], 1, v[130:131]
	.long	0x68A4A406	; v_add_u32_e32 v82, s6, v82
	.long	0x22A6A49F	; v_ashrrev_i32_e32 v83, 31, v82
	.long	0xD208006C, 0x06090352	; v_lshl_add_u64 v[108:109], v[82:83], 1, v[130:131]
	.long	0x68A4A400	; v_add_u32_e32 v82, s0, v82
	.long	0x22A6A49F	; v_ashrrev_i32_e32 v83, 31, v82
	.long	0xDC6C8000, 0x007F5856	; global_store_short_d16_hi v[86:87], v88, off
	.long	0xDC6C8000, 0x007F596C	; global_store_short_d16_hi v[108:109], v89, off
	.long	0xD2080058, 0x06090352	; v_lshl_add_u64 v[88:89], v[82:83], 1, v[130:131]
	.long	0x68A4A406	; v_add_u32_e32 v82, s6, v82
	.long	0x22A6A49F	; v_ashrrev_i32_e32 v83, 31, v82
	.long	0xD208006E, 0x06090352	; v_lshl_add_u64 v[110:111], v[82:83], 1, v[130:131]
	.long	0x68A4A406	; v_add_u32_e32 v82, s6, v82
	.long	0x22A6A49F	; v_ashrrev_i32_e32 v83, 31, v82
	.long	0xDC6C8000, 0x007F5A58	; global_store_short_d16_hi v[88:89], v90, off
	.long	0xDC6C8000, 0x007F5B6E	; global_store_short_d16_hi v[110:111], v91, off
	.long	0xD208005A, 0x06090352	; v_lshl_add_u64 v[90:91], v[82:83], 1, v[130:131]
	.long	0x68A4A406	; v_add_u32_e32 v82, s6, v82
	.long	0x22A6A49F	; v_ashrrev_i32_e32 v83, 31, v82
	.long	0xD2080070, 0x06090352	; v_lshl_add_u64 v[112:113], v[82:83], 1, v[130:131]
	.long	0x68A4A400	; v_add_u32_e32 v82, s0, v82
	.long	0x22A6A49F	; v_ashrrev_i32_e32 v83, 31, v82
	.long	0xDC6C8000, 0x007F5C5A	; global_store_short_d16_hi v[90:91], v92, off
	.long	0xDC6C8000, 0x007F5D70	; global_store_short_d16_hi v[112:113], v93, off
	.long	0xD208005C, 0x06090352	; v_lshl_add_u64 v[92:93], v[82:83], 1, v[130:131]
	.long	0x68A4A406	; v_add_u32_e32 v82, s6, v82
	.long	0x22A6A49F	; v_ashrrev_i32_e32 v83, 31, v82
	.long	0xD2080072, 0x06090352	; v_lshl_add_u64 v[114:115], v[82:83], 1, v[130:131]
	.long	0x68A4A406	; v_add_u32_e32 v82, s6, v82
	.long	0x22A6A49F	; v_ashrrev_i32_e32 v83, 31, v82
	.long	0xDC6C8000, 0x007F5E5C	; global_store_short_d16_hi v[92:93], v94, off
	.long	0xDC6C8000, 0x007F5F72	; global_store_short_d16_hi v[114:115], v95, off
	.long	0xD208005E, 0x06090352	; v_lshl_add_u64 v[94:95], v[82:83], 1, v[130:131]
	.long	0x68A4A406	; v_add_u32_e32 v82, s6, v82
	.long	0x22A6A49F	; v_ashrrev_i32_e32 v83, 31, v82
	.long	0xD2080052, 0x06090352	; v_lshl_add_u64 v[82:83], v[82:83], 1, v[130:131]
	.long	0xDC6C8000, 0x007F605E	; global_store_short_d16_hi v[94:95], v96, off
	.long	0xDC6C8000, 0x007F6152	; global_store_short_d16_hi v[82:83], v97, off
	.long	0xDC6C8040, 0x007F4264	; global_store_short_d16_hi v[100:101], v66, off offset:64
	.long	0xDC6C8040, 0x007F4366	; global_store_short_d16_hi v[102:103], v67, off offset:64
	.long	0xDC6C8040, 0x007F4462	; global_store_short_d16_hi v[98:99], v68, off offset:64
	.long	0xDC6C8040, 0x007F4568	; global_store_short_d16_hi v[104:105], v69, off offset:64
	.long	0xDC6C8040, 0x007F4654	; global_store_short_d16_hi v[84:85], v70, off offset:64
	.long	0xDC6C8040, 0x007F476A	; global_store_short_d16_hi v[106:107], v71, off offset:64
	.long	0xDC6C8040, 0x007F4856	; global_store_short_d16_hi v[86:87], v72, off offset:64
	.long	0xDC6C8040, 0x007F496C	; global_store_short_d16_hi v[108:109], v73, off offset:64
	.long	0xDC6C8040, 0x007F4A58	; global_store_short_d16_hi v[88:89], v74, off offset:64
	.long	0xDC6C8040, 0x007F4B6E	; global_store_short_d16_hi v[110:111], v75, off offset:64
	.long	0xDC6C8040, 0x007F4C5A	; global_store_short_d16_hi v[90:91], v76, off offset:64
	.long	0xDC6C8040, 0x007F4D70	; global_store_short_d16_hi v[112:113], v77, off offset:64
	.long	0xDC6C8040, 0x007F4E5C	; global_store_short_d16_hi v[92:93], v78, off offset:64
	.long	0xDC6C8040, 0x007F4F72	; global_store_short_d16_hi v[114:115], v79, off offset:64
	.long	0xDC6C8040, 0x007F505E	; global_store_short_d16_hi v[94:95], v80, off offset:64
	.long	0xDC6C8040, 0x007F5152	; global_store_short_d16_hi v[82:83], v81, off offset:64
	.long	0x28852CC0	; v_or_b32_e32 v66, 64, v150
	.long	0xD1E80242, 0x04000D42	; v_mad_u64_u32 v[66:67], s[2:3], v66, s6, v[0:1]
	.long	0x2286849F	; v_ashrrev_i32_e32 v67, 31, v66
	.long	0xD2080044, 0x06090342	; v_lshl_add_u64 v[68:69], v[66:67], 1, v[130:131]
	.long	0x68848406	; v_add_u32_e32 v66, s6, v66
	.long	0x2286849F	; v_ashrrev_i32_e32 v67, 31, v66
	.long	0xDC6C8000, 0x007F3244	; global_store_short_d16_hi v[68:69], v50, off
	.long	0xD2080046, 0x06090342	; v_lshl_add_u64 v[70:71], v[66:67], 1, v[130:131]
	.long	0x68648406	; v_add_u32_e32 v50, s6, v66
	.long	0xDC6C8000, 0x007F3346	; global_store_short_d16_hi v[70:71], v51, off
	.long	0x2266649F	; v_ashrrev_i32_e32 v51, 31, v50
	.long	0xD2080042, 0x06090332	; v_lshl_add_u64 v[66:67], v[50:51], 1, v[130:131]
	.long	0x68646406	; v_add_u32_e32 v50, s6, v50
	.long	0x2266649F	; v_ashrrev_i32_e32 v51, 31, v50
	.long	0xD2080048, 0x06090332	; v_lshl_add_u64 v[72:73], v[50:51], 1, v[130:131]
	.long	0x68646400	; v_add_u32_e32 v50, s0, v50
	.long	0x2266649F	; v_ashrrev_i32_e32 v51, 31, v50
	.long	0xDC6C8000, 0x007F3442	; global_store_short_d16_hi v[66:67], v52, off
	.long	0xDC6C8000, 0x007F3548	; global_store_short_d16_hi v[72:73], v53, off
	.long	0xD2080034, 0x06090332	; v_lshl_add_u64 v[52:53], v[50:51], 1, v[130:131]
	.long	0x68646406	; v_add_u32_e32 v50, s6, v50
	.long	0x2266649F	; v_ashrrev_i32_e32 v51, 31, v50
	.long	0xD208004A, 0x06090332	; v_lshl_add_u64 v[74:75], v[50:51], 1, v[130:131]
	.long	0x68646406	; v_add_u32_e32 v50, s6, v50
	.long	0x2266649F	; v_ashrrev_i32_e32 v51, 31, v50
	.long	0xDC6C8000, 0x007F3634	; global_store_short_d16_hi v[52:53], v54, off
	.long	0xDC6C8000, 0x007F374A	; global_store_short_d16_hi v[74:75], v55, off
	.long	0xD2080036, 0x06090332	; v_lshl_add_u64 v[54:55], v[50:51], 1, v[130:131]
	.long	0x68646406	; v_add_u32_e32 v50, s6, v50
	.long	0x2266649F	; v_ashrrev_i32_e32 v51, 31, v50
	.long	0xD208004C, 0x06090332	; v_lshl_add_u64 v[76:77], v[50:51], 1, v[130:131]
	.long	0x68646400	; v_add_u32_e32 v50, s0, v50
	.long	0x2266649F	; v_ashrrev_i32_e32 v51, 31, v50
	.long	0xDC6C8000, 0x007F3836	; global_store_short_d16_hi v[54:55], v56, off
	.long	0xDC6C8000, 0x007F394C	; global_store_short_d16_hi v[76:77], v57, off
	.long	0xD2080038, 0x06090332	; v_lshl_add_u64 v[56:57], v[50:51], 1, v[130:131]
	.long	0x68646406	; v_add_u32_e32 v50, s6, v50
	.long	0x2266649F	; v_ashrrev_i32_e32 v51, 31, v50
	.long	0xD208004E, 0x06090332	; v_lshl_add_u64 v[78:79], v[50:51], 1, v[130:131]
	.long	0x68646406	; v_add_u32_e32 v50, s6, v50
	.long	0x2266649F	; v_ashrrev_i32_e32 v51, 31, v50
	.long	0xDC6C8000, 0x007F3A38	; global_store_short_d16_hi v[56:57], v58, off
	.long	0xDC6C8000, 0x007F3B4E	; global_store_short_d16_hi v[78:79], v59, off
	.long	0xD208003A, 0x06090332	; v_lshl_add_u64 v[58:59], v[50:51], 1, v[130:131]
	.long	0x68646406	; v_add_u32_e32 v50, s6, v50
	.long	0x2266649F	; v_ashrrev_i32_e32 v51, 31, v50
	.long	0xD2080050, 0x06090332	; v_lshl_add_u64 v[80:81], v[50:51], 1, v[130:131]
	.long	0x68646400	; v_add_u32_e32 v50, s0, v50
	.long	0x2266649F	; v_ashrrev_i32_e32 v51, 31, v50
	.long	0xDC6C8000, 0x007F3C3A	; global_store_short_d16_hi v[58:59], v60, off
	.long	0xDC6C8000, 0x007F3D50	; global_store_short_d16_hi v[80:81], v61, off
	.long	0xD208003C, 0x06090332	; v_lshl_add_u64 v[60:61], v[50:51], 1, v[130:131]
	.long	0x68646406	; v_add_u32_e32 v50, s6, v50
	.long	0x2266649F	; v_ashrrev_i32_e32 v51, 31, v50
	.long	0xD2080052, 0x06090332	; v_lshl_add_u64 v[82:83], v[50:51], 1, v[130:131]
	.long	0x68646406	; v_add_u32_e32 v50, s6, v50
	.long	0x2266649F	; v_ashrrev_i32_e32 v51, 31, v50
	.long	0xDC6C8000, 0x007F3E3C	; global_store_short_d16_hi v[60:61], v62, off
	.long	0xDC6C8000, 0x007F3F52	; global_store_short_d16_hi v[82:83], v63, off
	.long	0xD208003E, 0x06090332	; v_lshl_add_u64 v[62:63], v[50:51], 1, v[130:131]
	.long	0x68646406	; v_add_u32_e32 v50, s6, v50
	.long	0x2266649F	; v_ashrrev_i32_e32 v51, 31, v50
	.long	0xD2080032, 0x06090332	; v_lshl_add_u64 v[50:51], v[50:51], 1, v[130:131]
	.long	0xDC6C8000, 0x007F403E	; global_store_short_d16_hi v[62:63], v64, off
	.long	0xDC6C8000, 0x007F4132	; global_store_short_d16_hi v[50:51], v65, off
	.long	0xDC6C8040, 0x007F2244	; global_store_short_d16_hi v[68:69], v34, off offset:64
	.long	0xDC6C8040, 0x007F2346	; global_store_short_d16_hi v[70:71], v35, off offset:64
	.long	0xDC6C8040, 0x007F2442	; global_store_short_d16_hi v[66:67], v36, off offset:64
	.long	0xDC6C8040, 0x007F2548	; global_store_short_d16_hi v[72:73], v37, off offset:64
	.long	0xDC6C8040, 0x007F2634	; global_store_short_d16_hi v[52:53], v38, off offset:64
	.long	0xDC6C8040, 0x007F274A	; global_store_short_d16_hi v[74:75], v39, off offset:64
	.long	0xDC6C8040, 0x007F2836	; global_store_short_d16_hi v[54:55], v40, off offset:64
	.long	0xDC6C8040, 0x007F294C	; global_store_short_d16_hi v[76:77], v41, off offset:64
	.long	0xDC6C8040, 0x007F2A38	; global_store_short_d16_hi v[56:57], v42, off offset:64
	.long	0xDC6C8040, 0x007F2B4E	; global_store_short_d16_hi v[78:79], v43, off offset:64
	.long	0xDC6C8040, 0x007F2C3A	; global_store_short_d16_hi v[58:59], v44, off offset:64
	.long	0xDC6C8040, 0x007F2D50	; global_store_short_d16_hi v[80:81], v45, off offset:64
	.long	0xDC6C8040, 0x007F2E3C	; global_store_short_d16_hi v[60:61], v46, off offset:64
	.long	0xDC6C8040, 0x007F2F52	; global_store_short_d16_hi v[82:83], v47, off offset:64
	.long	0xDC6C8040, 0x007F303E	; global_store_short_d16_hi v[62:63], v48, off offset:64
	.long	0xDC6C8040, 0x007F3132	; global_store_short_d16_hi v[50:51], v49, off offset:64
	.long	0x28452CFF, 0x00000060	; v_or_b32_e32 v34, 0x60, v150
	.long	0xD1E80222, 0x04000D22	; v_mad_u64_u32 v[34:35], s[2:3], v34, s6, v[0:1]
	.long	0x2246449F	; v_ashrrev_i32_e32 v35, 31, v34
	.long	0xD2080024, 0x06090322	; v_lshl_add_u64 v[36:37], v[34:35], 1, v[130:131]
	.long	0x68444406	; v_add_u32_e32 v34, s6, v34
	.long	0x2246449F	; v_ashrrev_i32_e32 v35, 31, v34
	.long	0xDC6C8000, 0x007F1224	; global_store_short_d16_hi v[36:37], v18, off
	.long	0xD2080026, 0x06090322	; v_lshl_add_u64 v[38:39], v[34:35], 1, v[130:131]
	.long	0x68244406	; v_add_u32_e32 v18, s6, v34
	.long	0xDC6C8000, 0x007F1326	; global_store_short_d16_hi v[38:39], v19, off
	.long	0x2226249F	; v_ashrrev_i32_e32 v19, 31, v18
	.long	0xD2080022, 0x06090312	; v_lshl_add_u64 v[34:35], v[18:19], 1, v[130:131]
	.long	0x68242406	; v_add_u32_e32 v18, s6, v18
	.long	0x2226249F	; v_ashrrev_i32_e32 v19, 31, v18
	.long	0xD2080028, 0x06090312	; v_lshl_add_u64 v[40:41], v[18:19], 1, v[130:131]
	.long	0x68242400	; v_add_u32_e32 v18, s0, v18
	.long	0x2226249F	; v_ashrrev_i32_e32 v19, 31, v18
	.long	0xDC6C8000, 0x007F1422	; global_store_short_d16_hi v[34:35], v20, off
	.long	0xDC6C8000, 0x007F1528	; global_store_short_d16_hi v[40:41], v21, off
	.long	0xD2080014, 0x06090312	; v_lshl_add_u64 v[20:21], v[18:19], 1, v[130:131]
	.long	0x68242406	; v_add_u32_e32 v18, s6, v18
	.long	0x2226249F	; v_ashrrev_i32_e32 v19, 31, v18
	.long	0xD208002A, 0x06090312	; v_lshl_add_u64 v[42:43], v[18:19], 1, v[130:131]
	.long	0x68242406	; v_add_u32_e32 v18, s6, v18
	.long	0x2226249F	; v_ashrrev_i32_e32 v19, 31, v18
	.long	0xDC6C8000, 0x007F1614	; global_store_short_d16_hi v[20:21], v22, off
	.long	0xDC6C8000, 0x007F172A	; global_store_short_d16_hi v[42:43], v23, off
	.long	0xD2080016, 0x06090312	; v_lshl_add_u64 v[22:23], v[18:19], 1, v[130:131]
	.long	0x68242406	; v_add_u32_e32 v18, s6, v18
	.long	0x2226249F	; v_ashrrev_i32_e32 v19, 31, v18
	.long	0xD208002C, 0x06090312	; v_lshl_add_u64 v[44:45], v[18:19], 1, v[130:131]
	.long	0x68242400	; v_add_u32_e32 v18, s0, v18
	.long	0x2226249F	; v_ashrrev_i32_e32 v19, 31, v18
	.long	0xDC6C8000, 0x007F1816	; global_store_short_d16_hi v[22:23], v24, off
	.long	0xDC6C8000, 0x007F192C	; global_store_short_d16_hi v[44:45], v25, off
	.long	0xD2080018, 0x06090312	; v_lshl_add_u64 v[24:25], v[18:19], 1, v[130:131]
	.long	0x68242406	; v_add_u32_e32 v18, s6, v18
	.long	0x2226249F	; v_ashrrev_i32_e32 v19, 31, v18
	.long	0xD208002E, 0x06090312	; v_lshl_add_u64 v[46:47], v[18:19], 1, v[130:131]
	.long	0x68242406	; v_add_u32_e32 v18, s6, v18
	.long	0x2226249F	; v_ashrrev_i32_e32 v19, 31, v18
	.long	0xDC6C8000, 0x007F1A18	; global_store_short_d16_hi v[24:25], v26, off
	.long	0xDC6C8000, 0x007F1B2E	; global_store_short_d16_hi v[46:47], v27, off
	.long	0xD208001A, 0x06090312	; v_lshl_add_u64 v[26:27], v[18:19], 1, v[130:131]
	.long	0x68242406	; v_add_u32_e32 v18, s6, v18
	.long	0x2226249F	; v_ashrrev_i32_e32 v19, 31, v18
	.long	0xD2080030, 0x06090312	; v_lshl_add_u64 v[48:49], v[18:19], 1, v[130:131]
	.long	0x68242400	; v_add_u32_e32 v18, s0, v18
	.long	0x2226249F	; v_ashrrev_i32_e32 v19, 31, v18
	.long	0xDC6C8000, 0x007F1C1A	; global_store_short_d16_hi v[26:27], v28, off
	.long	0xDC6C8000, 0x007F1D30	; global_store_short_d16_hi v[48:49], v29, off
	.long	0xD208001C, 0x06090312	; v_lshl_add_u64 v[28:29], v[18:19], 1, v[130:131]
	.long	0x68242406	; v_add_u32_e32 v18, s6, v18
	.long	0x2226249F	; v_ashrrev_i32_e32 v19, 31, v18
	.long	0x280202FF, 0x0000007B	; v_or_b32_e32 v1, 0x7b, v1
	.long	0xD2080032, 0x06090312	; v_lshl_add_u64 v[50:51], v[18:19], 1, v[130:131]
	.long	0x68242406	; v_add_u32_e32 v18, s6, v18
	.long	0xD1E80000, 0x04000D01	; v_mad_u64_u32 v[0:1], s[0:1], v1, s6, v[0:1]
	.long	0x2226249F	; v_ashrrev_i32_e32 v19, 31, v18
	.long	0x2202009F	; v_ashrrev_i32_e32 v1, 31, v0
	.long	0xD2080012, 0x06090312	; v_lshl_add_u64 v[18:19], v[18:19], 1, v[130:131]
	.long	0xD2080000, 0x06090300	; v_lshl_add_u64 v[0:1], v[0:1], 1, v[130:131]
	.long	0xDC6C8000, 0x007F1E1C	; global_store_short_d16_hi v[28:29], v30, off
	.long	0xDC6C8000, 0x007F1F32	; global_store_short_d16_hi v[50:51], v31, off
	.long	0xDC6C8000, 0x007F2012	; global_store_short_d16_hi v[18:19], v32, off
	.long	0xDC6C8000, 0x007F2100	; global_store_short_d16_hi v[0:1], v33, off
	.long	0xDC6C8040, 0x007F0224	; global_store_short_d16_hi v[36:37], v2, off offset:64
	.long	0xDC6C8040, 0x007F0326	; global_store_short_d16_hi v[38:39], v3, off offset:64
	.long	0xDC6C8040, 0x007F0422	; global_store_short_d16_hi v[34:35], v4, off offset:64
	.long	0xDC6C8040, 0x007F0528	; global_store_short_d16_hi v[40:41], v5, off offset:64
	.long	0xDC6C8040, 0x007F0614	; global_store_short_d16_hi v[20:21], v6, off offset:64
	.long	0xDC6C8040, 0x007F072A	; global_store_short_d16_hi v[42:43], v7, off offset:64
	.long	0xDC6C8040, 0x007F0816	; global_store_short_d16_hi v[22:23], v8, off offset:64
	.long	0xDC6C8040, 0x007F092C	; global_store_short_d16_hi v[44:45], v9, off offset:64
	.long	0xDC6C8040, 0x007F0A18	; global_store_short_d16_hi v[24:25], v10, off offset:64
	.long	0xDC6C8040, 0x007F0B2E	; global_store_short_d16_hi v[46:47], v11, off offset:64
	.long	0xDC6C8040, 0x007F0C1A	; global_store_short_d16_hi v[26:27], v12, off offset:64
	.long	0xDC6C8040, 0x007F0D30	; global_store_short_d16_hi v[48:49], v13, off offset:64
	.long	0xDC6C8040, 0x007F0E1C	; global_store_short_d16_hi v[28:29], v14, off offset:64
	.long	0xDC6C8040, 0x007F0F32	; global_store_short_d16_hi v[50:51], v15, off offset:64
	.long	0xDC6C8040, 0x007F1012	; global_store_short_d16_hi v[18:19], v16, off offset:64
	.long	0xDC6C8040, 0x007F1100	; global_store_short_d16_hi v[0:1], v17, off offset:64
	.long	0xBF810000	; s_endpgm
.Lfunc_end0:
	.size _Z8micro_tk13micro_globals, .Lfunc_end0-_Z8micro_tk13micro_globals

.section .rodata,"a",@progbits
.p2align 6
.protected _Z8micro_tk13micro_globals.kd
.globl _Z8micro_tk13micro_globals.kd
.type _Z8micro_tk13micro_globals.kd,@object
_Z8micro_tk13micro_globals.kd:
	.long	0x00000000
	.long	0x00000000
	.long	0x00000198
	.long	0x00000000
	.quad	_Z8micro_tk13micro_globals - _Z8micro_tk13micro_globals.kd
	.byte	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
	.byte	0x00, 0x00, 0x00, 0x00, 0x31, 0x00, 0x00, 0x00, 0xd8, 0x00, 0xaf, 0x00, 0x84, 0x01, 0x00, 0x00
	.byte	0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
	.size _Z8micro_tk13micro_globals.kd, .-_Z8micro_tk13micro_globals.kd
	.set _Z8micro_tk13micro_globals.private_seg_size, 0
	.set _Z8micro_tk13micro_globals.num_vgpr, 200
	.set _Z8micro_tk13micro_globals.num_agpr, 0
	.set _Z8micro_tk13micro_globals.numbered_sgpr, 25
	.set _Z8micro_tk13micro_globals.uses_vcc, 1
	.set _Z8micro_tk13micro_globals.uses_flat_scratch, 0
	.set _Z8micro_tk13micro_globals.has_dyn_sized_stack, 0
	.set _Z8micro_tk13micro_globals.has_recursion, 0

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
        .size:           152
        .value_kind:     by_value
      - .offset:         152
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         156
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         160
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         164
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         166
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         168
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         170
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         172
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         174
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         192
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         200
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         208
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         216
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         272
        .size:           4
        .value_kind:     hidden_dynamic_lds_size
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 408
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z8micro_tk13micro_globals
    .private_segment_fixed_size: 0
    .sgpr_count:     31
    .sgpr_spill_count: 0
    .symbol:         _Z8micro_tk13micro_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     200
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
