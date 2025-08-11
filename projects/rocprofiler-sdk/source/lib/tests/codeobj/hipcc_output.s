s_load_b64 s[0:1], s[0:1], 0x0
s_getreg_b32 s2, hwreg(HW_REG_SHADER_CYCLES, 0, 20)
s_waitcnt vmcnt(0) lgkmcnt(0)
s_waitcnt_vscnt null, 0x0
s_barrier
s_waitcnt vmcnt(0) lgkmcnt(0)
s_waitcnt_vscnt null, 0x0
buffer_gl0_inv
s_getreg_b32 s3, hwreg(HW_REG_SHADER_CYCLES, 0, 20)
s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
s_sub_u32 s4, s3, s2
s_subb_u32 s5, 0, 0
v_cmp_lt_i64_e64 s3, s[4:5], s[0:1]
s_delay_alu instid0(VALU_DEP_1)
s_and_b32 vcc_lo, exec_lo, s3
s_cbranch_vccnz 65520
s_endpgm
