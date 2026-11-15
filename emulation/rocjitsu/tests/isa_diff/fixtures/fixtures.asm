# rj_isa_diff validator fixtures (gfx950). Each "=== name" section is one kernel body.
# Regenerate fixtures.inc with regen_fixtures.sh (needs llvm-mc; tests never run LLVM).

=== shift_old
s_add_u32 s4, s2, s3
s_addc_u32 s5, s0, s1
s_cselect_b32 s6, s4, s5
s_lshl_b64 s[8:9], s[4:5], 2
s_cmp_lg_u32 s6, 0
s_cselect_b32 s7, s8, s9
s_endpgm

=== shift_new
s_add_u32 s4, s2, s3
s_addc_u32 s5, s0, s1
s_cselect_b32 s6, s4, s5
s_lshl_b64 s[8:9], s[4:5], 1
s_lshl_b64 s[8:9], s[8:9], 1
s_cmp_lg_u32 s6, 0
s_cselect_b32 s7, s8, s9
s_endpgm

=== deadtemp_old
s_mov_b32 s10, 0x11
s_add_u32 s4, s2, s3
s_mov_b32 s10, s4
s_cmp_eq_u32 s10, 0
s_endpgm

=== deadtemp_new
s_mov_b32 s10, 0x22
s_add_u32 s4, s2, s3
s_mov_b32 s10, s4
s_cmp_eq_u32 s10, 0
s_endpgm

=== scclive_old
s_add_u32 s4, s2, s3
s_cselect_b32 s5, s4, 0
s_cmp_eq_u32 s5, 0
s_store_dword s5, s[0:1], 0x0
s_endpgm

=== scclive_new
s_add_i32 s4, s2, s3
s_cselect_b32 s5, s4, 0
s_cmp_eq_u32 s5, 0
s_store_dword s5, s[0:1], 0x0
s_endpgm

=== sccdead_old
s_add_u32 s4, s2, s3
s_cmp_eq_u32 s4, 0
s_cselect_b32 s5, s4, 0
s_cmp_eq_u32 s5, 0
s_endpgm

=== sccdead_new
s_add_i32 s4, s2, s3
s_cmp_eq_u32 s4, 0
s_cselect_b32 s5, s4, 0
s_cmp_eq_u32 s5, 0
s_endpgm

=== clobber_old
s_add_u32 s4, s2, s3
s_cmp_eq_u32 s4, 0
s_add_u32 s21, s20, s4
s_cmp_eq_u32 s21, 0
s_endpgm

=== clobber_new
s_add_u32 s4, s2, s3
s_mov_b32 s20, 0
s_cmp_eq_u32 s4, 0
s_add_u32 s21, s20, s4
s_cmp_eq_u32 s21, 0
s_endpgm

=== vcclive_old
s_mov_b32 vcc_lo, s2
s_cmp_eq_u32 s3, 0
s_cselect_b64 s[6:7], vcc, 0
s_cmp_eq_u32 s6, 0
s_endpgm

=== vcclive_new
s_mov_b64 vcc, s[2:3]
s_cmp_eq_u32 s3, 0
s_cselect_b64 s[6:7], vcc, 0
s_cmp_eq_u32 s6, 0
s_endpgm

=== vccdead_old
s_mov_b32 vcc_lo, s2
s_mov_b32 vcc_hi, 0
s_cmp_eq_u32 s3, 0
s_cselect_b64 s[6:7], vcc, 0
s_cmp_eq_u32 s6, 0
s_endpgm

=== vccdead_new
s_mov_b64 vcc, s[2:3]
s_mov_b32 vcc_hi, 0
s_cmp_eq_u32 s3, 0
s_cselect_b64 s[6:7], vcc, 0
s_cmp_eq_u32 s6, 0
s_endpgm

=== overlap_old
s_lshl_b64 s[4:5], s[4:5], 3
s_add_u32 s6, s6, s7
s_cmp_eq_u32 s4, 0
s_cselect_b64 s[8:9], s[4:5], s[6:7]
s_cmp_eq_u32 s8, 0
s_endpgm

=== overlap_new
s_lshl_b64 s[4:5], s[4:5], 1
s_lshl_b64 s[4:5], s[4:5], 2
s_add_u32 s6, s7, s6
s_cmp_eq_u32 s4, 0
s_cselect_b64 s[8:9], s[4:5], s[6:7]
s_cmp_eq_u32 s8, 0
s_endpgm

=== constraint_old
s_and_b32 s4, s4, 0xff
s_lshr_b32 s5, s4, 4
s_add_u32 s6, s5, s4
s_cmp_eq_u32 s6, 0
s_endpgm

=== constraint_new
s_and_b32 s4, s4, 0xff
s_bfe_u32 s5, s4, 0x40004
s_add_u32 s6, s5, s4
s_cmp_eq_u32 s6, 0
s_endpgm

=== twohunk_old
s_add_u32 s4, s2, 1
s_mov_b32 s9, 7
s_add_u32 s4, s4, 1
s_cmp_lg_u32 s4, 0
s_cselect_b32 s5, s9, 0
s_cmp_eq_u32 s5, 0
s_endpgm

=== twohunk_new
s_add_u32 s4, s2, 2
s_mov_b32 s9, 7
s_add_u32 s4, s4, 0
s_cmp_lg_u32 s4, 0
s_cselect_b32 s5, s9, 0
s_cmp_eq_u32 s5, 0
s_endpgm

=== rename_old
s_mov_b32 s4, s2
s_cmp_eq_u32 s3, 0
s_cbranch_scc1 BB0
s_add_u32 s4, s4, 1
BB0:
s_nop 0
s_add_u32 s6, s4, s5
s_cmp_lg_u32 s6, 0
s_endpgm

=== rename_new
s_mov_b32 s8, s2
s_cmp_eq_u32 s3, 0
s_cbranch_scc1 BB0
s_add_u32 s8, s8, 1
BB0:
s_nop 0
s_add_u32 s6, s8, s5
s_cmp_lg_u32 s6, 0
s_endpgm

=== setpc_old
s_add_u32 s4, s2, s3
s_cmp_eq_u32 s4, 0
s_setpc_b64 s[0:1]

=== setpc_new
s_add_u32 s4, s3, s2
s_cmp_eq_u32 s4, 0
s_setpc_b64 s[0:1]

=== exec_old
s_and_b64 exec, exec, s[2:3]
s_mov_b64 s[6:7], exec
s_cselect_b32 s8, 1, 0
s_cmp_eq_u32 s8, 0
s_endpgm

=== exec_new
s_and_b64 exec, s[2:3], exec
s_mov_b64 s[6:7], exec
s_cselect_b32 s8, 1, 0
s_cmp_eq_u32 s8, 0
s_endpgm

=== waitcnt_old
s_load_dword s4, s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_add_u32 s5, s4, 1
s_endpgm

=== waitcnt_new
s_load_dword s4, s[0:1], 0x0
s_waitcnt vmcnt(0) lgkmcnt(0)
s_add_u32 s5, s4, 1
s_endpgm

=== smem_old
s_load_dword s4, s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_add_u32 s5, s4, 1
s_endpgm

=== smem_new
s_load_dword s4, s[0:1], 0x4
s_waitcnt lgkmcnt(0)
s_add_u32 s5, s4, 1
s_endpgm

=== getpc_old
s_getpc_b64 s[0:1]
s_add_u32 s0, s0, 0x10
s_addc_u32 s1, s1, 0
s_load_dword s4, s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_endpgm

=== getpc_new
s_getpc_b64 s[0:1]
s_add_u32 s0, s0, 0x20
s_addc_u32 s1, s1, 0
s_load_dword s4, s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_endpgm

=== getreg_old
s_getreg_b32 s4, hwreg(HW_REG_MODE)
s_cmp_eq_u32 s4, 0
s_endpgm

=== getreg_new
s_getreg_b32 s4, hwreg(HW_REG_STATUS)
s_cmp_eq_u32 s4, 0
s_endpgm

=== unreachable_old
s_add_u32 s4, s2, s3
s_endpgm
s_mov_b32 s5, 1

=== unreachable_new
s_add_u32 s4, s2, s3
s_endpgm
s_mov_b32 s5, 2

=== cachedead_old
s_mov_b32 s10, 1
s_mov_b32 s10, 0
s_cmp_eq_u32 s10, 0
s_endpgm

=== cachedead_new
s_mov_b32 s10, 2
s_mov_b32 s10, 0
s_cmp_eq_u32 s10, 0
s_endpgm

=== cachelive_old
s_mov_b32 s10, 1
s_add_u32 s11, s10, 1
s_cmp_eq_u32 s11, 0
s_endpgm

=== cachelive_new
s_mov_b32 s10, 2
s_add_u32 s11, s10, 1
s_cmp_eq_u32 s11, 0
s_endpgm

=== effects
s_add_u32 s4, s2, s3
s_addc_u32 s5, s0, s1
s_cselect_b64 s[14:15], s[2:3], vcc
s_mov_b32 vcc_hi, s1
s_and_b64 exec, exec, s[2:3]
s_and_saveexec_b64 s[2:3], s[4:5]
s_cmov_b32 s6, s7
s_mov_b32 m0, s2
s_mov_b32 s9, src_scc
s_lshl_b64 s[6:7], s[4:5], 3
s_cbranch_scc0 1
s_cbranch_execz 1
s_cbranch_vccnz 1
s_getpc_b64 s[18:19]
s_waitcnt 0
s_load_dword s20, s[0:1], 0x0
v_mov_b32 v0, s1
s_mov_b32 s0, ttmp0
s_getreg_b32 s4, hwreg(HW_REG_MODE)
s_movrels_b32 s0, s1
s_endpgm

=== roles
s_lshl_b32 s4, s2, s3
s_bfe_u32 s5, s6, s7
s_sext_i32_i8 s8, s9
s_cmp_gt_u32 s10, 0x12345
s_add_u32 s11, s12, s13
s_lshl_b64 s[14:15], s[16:17], s18
s_cmp_eq_u64 s[20:21], s[22:23]
s_endpgm

=== offbyone_old
s_cmp_gt_u32 s2, 0x12345
s_cselect_b32 s4, 1, 0
s_cmp_eq_u32 s4, 0
s_endpgm

=== offbyone_bug
s_cmp_ge_u32 s2, 0x12345
s_cselect_b32 s4, 1, 0
s_cmp_eq_u32 s4, 0
s_endpgm

=== offbyone_fixed
s_cmp_ge_u32 s2, 0x12346
s_cselect_b32 s4, 1, 0
s_cmp_eq_u32 s4, 0
s_endpgm

=== pending_conflict_old
s_load_dword s4, s[0:1], 0x0
s_add_u32 s5, s4, 1
s_waitcnt lgkmcnt(0)
s_store_dword s5, s[0:1], 0x4
s_endpgm

=== pending_conflict_new
s_load_dword s4, s[0:1], 0x0
s_add_u32 s5, 1, s4
s_waitcnt lgkmcnt(0)
s_store_dword s5, s[0:1], 0x4
s_endpgm

=== pending_disjoint_old
s_load_dword s10, s[0:1], 0x0
s_add_u32 s5, s4, 1
s_waitcnt lgkmcnt(0)
s_add_u32 s6, s10, s5
s_store_dword s6, s[0:1], 0x4
s_endpgm

=== pending_disjoint_new
s_load_dword s10, s[0:1], 0x0
s_add_u32 s5, 1, s4
s_waitcnt lgkmcnt(0)
s_add_u32 s6, s10, s5
s_store_dword s6, s[0:1], 0x4
s_endpgm

=== pending_waited_old
s_load_dword s4, s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_add_u32 s5, s4, 1
s_store_dword s5, s[0:1], 0x4
s_endpgm

=== pending_waited_new
s_load_dword s4, s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_add_u32 s5, 1, s4
s_store_dword s5, s[0:1], 0x4
s_endpgm

=== pending_partial_old
s_load_dword s4, s[0:1], 0x0
s_load_dword s10, s[0:1], 0x8
s_waitcnt lgkmcnt(1)
s_add_u32 s5, s4, 1
s_waitcnt lgkmcnt(0)
s_add_u32 s6, s10, s5
s_store_dword s6, s[0:1], 0x4
s_endpgm

=== pending_partial_new
s_load_dword s4, s[0:1], 0x0
s_load_dword s10, s[0:1], 0x8
s_waitcnt lgkmcnt(1)
s_add_u32 s5, 1, s4
s_waitcnt lgkmcnt(0)
s_add_u32 s6, s10, s5
s_store_dword s6, s[0:1], 0x4
s_endpgm

=== comp_disjoint_old
s_getreg_b32 s4, hwreg(HW_REG_MODE)
s_cmp_eq_u32 s6, 0
s_cbranch_scc1 BB0
s_nop 0
BB0:
s_nop 0
s_add_u32 s5, s2, s3
s_store_dword s5, s[0:1], 0x0
s_endpgm

=== comp_disjoint_new
s_getreg_b32 s4, hwreg(HW_REG_STATUS)
s_cmp_eq_u32 s6, 0
s_cbranch_scc1 BB0
s_nop 0
BB0:
s_nop 0
s_add_u32 s5, s3, s2
s_store_dword s5, s[0:1], 0x0
s_endpgm

=== comp_conflict_old
s_getreg_b32 s4, hwreg(HW_REG_MODE)
s_cmp_eq_u32 s6, 0
s_cbranch_scc1 BB0
s_nop 0
BB0:
s_nop 0
s_add_u32 s5, s2, s4
s_store_dword s5, s[0:1], 0x0
s_endpgm

=== comp_conflict_new
s_getreg_b32 s4, hwreg(HW_REG_STATUS)
s_cmp_eq_u32 s6, 0
s_cbranch_scc1 BB0
s_nop 0
BB0:
s_nop 0
s_add_u32 s5, s4, s2
s_store_dword s5, s[0:1], 0x0
s_endpgm

=== expand_masked_old
s_lshl_b32 s4, s2, 1
s_and_b32 s4, s4, 1
s_store_dword s4, s[0:1], 0x0
s_endpgm

=== expand_masked_new
s_lshl_b32 s4, s2, 2
s_and_b32 s4, s4, 1
s_store_dword s4, s[0:1], 0x0
s_endpgm

=== valutemp_old
s_add_u32 s4, s2, s3
s_mov_b32 s8, s4
s_add_u32 s5, s8, 1
v_mov_b32 v2, s5
v_mov_b32 v0, 0
global_store_dword v0, v2, s[6:7]
s_endpgm

=== valutemp_new
s_add_u32 s4, s2, s3
s_mov_b32 s10, s4
s_add_u32 s5, s10, 1
v_mov_b32 v2, s5
v_mov_b32 v0, 0
global_store_dword v0, v2, s[6:7]
s_endpgm

=== valu_old
v_add_u32_e32 v2, v0, v1
v_lshlrev_b32_e32 v3, 2, v2
global_store_dword v4, v3, s[0:1]
s_endpgm

=== valu_new
v_add_u32_e32 v2, v1, v0
v_lshlrev_b32_e32 v3, 1, v2
v_lshlrev_b32_e32 v3, 1, v3
global_store_dword v4, v3, s[0:1]
s_endpgm

=== valuwrong_new
v_add_u32_e32 v2, v1, v0
v_lshlrev_b32_e32 v3, 3, v2
global_store_dword v4, v3, s[0:1]
s_endpgm

=== valuclobber_old
v_mov_b32_e32 v3, v2
global_store_dword v4, v3, s[0:1]
s_endpgm

=== valuclobber_new
s_mov_b64 s[10:11], exec
s_mov_b64 exec, -1
v_mov_b32_e32 v3, v2
s_mov_b64 exec, s[10:11]
global_store_dword v4, v3, s[0:1]
s_endpgm

=== valudead_old
v_add_u32_e32 v5, v0, v1
v_mov_b32_e32 v3, v5
global_store_dword v4, v3, s[0:1]
s_endpgm

=== valudead_new
v_add_u32_e32 v6, v0, v1
v_mov_b32_e32 v3, v6
global_store_dword v4, v3, s[0:1]
s_endpgm

=== valupending_old
global_load_dword v0, v4, s[0:1]
v_add_u32_e32 v2, v0, v1
s_waitcnt vmcnt(0)
global_store_dword v4, v2, s[0:1]
s_endpgm

=== valupending_new
global_load_dword v0, v4, s[0:1]
v_add_u32_e32 v2, v1, v0
s_waitcnt vmcnt(0)
global_store_dword v4, v2, s[0:1]
s_endpgm

=== valucvt
v_cvt_f32_f16_e32 v1, v0
global_store_dword v2, v1, s[0:1]
s_endpgm
