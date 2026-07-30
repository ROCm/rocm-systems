
build/heliosr-2b805-d7-3/115499_gfx1250_code_object_id_2.out:	file format elf64-amdgpu

Disassembly of section .text:

0000000000005500 <__amd_rocclr_fillBufferAligned>:
; __amd_rocclr_fillBufferAligned():
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1       // 000000005500: B9800641 00000001
	v_mad_u32 v0, s11, ttmp9, v0                               // 000000005508: D6350000 0400EA0B
	v_mov_b32_e32 v1, 0                                        // 000000005510: 7E020280
	s_mov_b32 s0, s7                                           // 000000005514: BE800007
	s_mov_b32 s7, 0                                            // 000000005518: BE870080
	s_bcnt1_i32_b32 s1, s0                                     // 00000000551C: BE811800
	s_mov_b32 s15, -1                                          // 000000005520: BE8F00C1
	s_cmp_lg_u32 s1, 1                                         // 000000005524: BF078101
	s_delay_alu instid0(VALU_DEP_2)                            // 000000005528: BF870002
	v_mul_u64_e32 v[0:1], s[6:7], v[0:1]                       // 00000000552C: 54000006
	s_cbranch_scc1 148                                         // 000000005530: BFA20094 <__amd_rocclr_fillBufferAligned+0x284>
	s_ctz_i32_b32 s14, s0                                      // 000000005534: BE8E0800
	s_mov_b32 s0, -1                                           // 000000005538: BE8000C1
	s_cmp_lt_i32 s14, 3                                        // 00000000553C: BF04830E
	s_mov_b32 s15, 0                                           // 000000005540: BE8F0080
	s_cbranch_scc1 90                                          // 000000005544: BFA2005A <__amd_rocclr_fillBufferAligned+0x1b0>
	s_cmp_gt_i32 s14, 3                                        // 000000005548: BF02830E
	s_cbranch_scc0 45                                          // 00000000554C: BFA1002D <__amd_rocclr_fillBufferAligned+0x104>
	s_cmp_eq_u32 s14, 4                                        // 000000005550: BF06840E
	s_mov_b32 s15, -1                                          // 000000005554: BE8F00C1
	s_cbranch_scc0 41                                          // 000000005558: BFA10029 <__amd_rocclr_fillBufferAligned+0x100>
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000555C: BF8700A1
	v_lshl_add_u64 v[2:3], v[0:1], 4, s[2:3]                   // 000000005560: D6520002 00090900
	s_mov_b32 s15, exec_lo                                     // 000000005568: BE8F007E
	v_cmpx_gt_u64_e64 s[8:9], v[2:3]                           // 00000000556C: D4DC007E 02020408
	s_cbranch_execz 32                                         // 000000005574: BFA50020 <__amd_rocclr_fillBufferAligned+0xf8>
	s_cmp_lg_u32 s6, 0                                         // 000000005578: BF078006
	s_mov_b32 s11, 0                                           // 00000000557C: BE8B0080
	s_cselect_b32 s16, -1, 0                                   // 000000005580: 981080C1
	s_lshl_b64 s[0:1], s[10:11], 4                             // 000000005584: 8480840A
	s_branch 6                                                 // 000000005588: BFA00006 <__amd_rocclr_fillBufferAligned+0xa4>
	v_add_nc_u64_e32 v[2:3], s[0:1], v[2:3]                    // 00000000558C: 50040400
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000005590: BF8704A1
	v_cmp_le_u64_e32 vcc_lo, s[8:9], v[2:3]                    // 000000005594: 7CB60408
	s_or_b32 s11, vcc_lo, s11                                  // 000000005598: 8C0B0B6A
	s_and_not1_b32 exec_lo, exec_lo, s11                       // 00000000559C: 917E0B7E
	s_cbranch_execz 21                                         // 0000000055A0: BFA50015 <__amd_rocclr_fillBufferAligned+0xf8>
	v_mov_b64_e32 v[4:5], v[2:3]                               // 0000000055A4: 7E083B02
	s_and_not1_b32 vcc_lo, exec_lo, s16                        // 0000000055A8: 916A107E
	s_mov_b64 s[12:13], s[4:5]                                 // 0000000055AC: BE8C0104
	s_mov_b32 s17, s6                                          // 0000000055B0: BE910006
	s_cbranch_vccnz 65525                                      // 0000000055B4: BFA4FFF5 <__amd_rocclr_fillBufferAligned+0x8c>
	s_load_b128 s[20:23], s[12:13], 0x0 nv                     // 0000000055B8: F4104506 F8000000
	s_add_co_i32 s17, s17, -1                                  // 0000000055C0: 8111C111
	s_wait_xcnt 0x0                                            // 0000000055C4: BFC50000
	s_add_nc_u64 s[12:13], s[12:13], 16                        // 0000000055C8: A98C900C
	s_cmp_eq_u32 s17, 0                                        // 0000000055CC: BF068011
	s_wait_kmcnt 0x0                                           // 0000000055D0: BFC70000
	v_mov_b64_e32 v[6:7], s[20:21]                             // 0000000055D4: 7E0C3A14
	v_mov_b64_e32 v[8:9], s[22:23]                             // 0000000055D8: 7E103A16
	global_store_b128 v[4:5], v[6:9], off                      // 0000000055DC: EE07407C 03000000 00000004
	s_wait_xcnt 0x0                                            // 0000000055E8: BFC50000
	v_add_nc_u64_e32 v[4:5], 16, v[4:5]                        // 0000000055EC: 50080890
	s_cbranch_scc0 65521                                       // 0000000055F0: BFA1FFF1 <__amd_rocclr_fillBufferAligned+0xb8>
	s_branch 65509                                             // 0000000055F4: BFA0FFE5 <__amd_rocclr_fillBufferAligned+0x8c>
	s_or_b32 exec_lo, exec_lo, s15                             // 0000000055F8: 8C7E0F7E
	s_mov_b32 s15, 0                                           // 0000000055FC: BE8F0080
	s_mov_b32 s0, 0                                            // 000000005600: BE800080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005604: BF870009
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000005608: 8B6A007E
	s_cbranch_vccz 39                                          // 00000000560C: BFA30027 <__amd_rocclr_fillBufferAligned+0x1ac>
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000005610: BF8700A1
	v_lshl_add_u64 v[2:3], v[0:1], 3, s[2:3]                   // 000000005614: D6520002 00090700
	s_mov_b32 s16, exec_lo                                     // 00000000561C: BE90007E
	v_cmpx_gt_u64_e64 s[8:9], v[2:3]                           // 000000005620: D4DC007E 02020408
	s_cbranch_execz 31                                         // 000000005628: BFA5001F <__amd_rocclr_fillBufferAligned+0x1a8>
	s_cmp_lg_u32 s6, 0                                         // 00000000562C: BF078006
	s_mov_b32 s11, 0                                           // 000000005630: BE8B0080
	s_cselect_b32 s17, -1, 0                                   // 000000005634: 981180C1
	s_lshl_b64 s[0:1], s[10:11], 3                             // 000000005638: 8480830A
	s_branch 6                                                 // 00000000563C: BFA00006 <__amd_rocclr_fillBufferAligned+0x158>
	v_add_nc_u64_e32 v[2:3], s[0:1], v[2:3]                    // 000000005640: 50040400
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000005644: BF8704A1
	v_cmp_le_u64_e32 vcc_lo, s[8:9], v[2:3]                    // 000000005648: 7CB60408
	s_or_b32 s11, vcc_lo, s11                                  // 00000000564C: 8C0B0B6A
	s_and_not1_b32 exec_lo, exec_lo, s11                       // 000000005650: 917E0B7E
	s_cbranch_execz 20                                         // 000000005654: BFA50014 <__amd_rocclr_fillBufferAligned+0x1a8>
	v_mov_b64_e32 v[4:5], v[2:3]                               // 000000005658: 7E083B02
	s_and_not1_b32 vcc_lo, exec_lo, s17                        // 00000000565C: 916A117E
	s_mov_b64 s[12:13], s[4:5]                                 // 000000005660: BE8C0104
	s_mov_b32 s18, s6                                          // 000000005664: BE920006
	s_cbranch_vccnz 65525                                      // 000000005668: BFA4FFF5 <__amd_rocclr_fillBufferAligned+0x140>
	s_load_b64 s[20:21], s[12:13], 0x0 nv                      // 00000000566C: F4102506 F8000000
	s_add_co_i32 s18, s18, -1                                  // 000000005674: 8112C112
	s_wait_xcnt 0x0                                            // 000000005678: BFC50000
	s_add_nc_u64 s[12:13], s[12:13], 8                         // 00000000567C: A98C880C
	s_cmp_eq_u32 s18, 0                                        // 000000005680: BF068012
	s_wait_kmcnt 0x0                                           // 000000005684: BFC70000
	v_mov_b64_e32 v[6:7], s[20:21]                             // 000000005688: 7E0C3A14
	global_store_b64 v[4:5], v[6:7], off                       // 00000000568C: EE06C07C 03000000 00000004
	s_wait_xcnt 0x0                                            // 000000005698: BFC50000
	v_add_nc_u64_e32 v[4:5], 8, v[4:5]                         // 00000000569C: 50080888
	s_cbranch_scc0 65522                                       // 0000000056A0: BFA1FFF2 <__amd_rocclr_fillBufferAligned+0x16c>
	s_branch 65510                                             // 0000000056A4: BFA0FFE6 <__amd_rocclr_fillBufferAligned+0x140>
	s_or_b32 exec_lo, exec_lo, s16                             // 0000000056A8: 8C7E107E
	s_mov_b32 s0, 0                                            // 0000000056AC: BE800080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000056B0: BF870009
	s_and_b32 vcc_lo, exec_lo, s0                              // 0000000056B4: 8B6A007E
	s_cbranch_vccz 50                                          // 0000000056B8: BFA30032 <__amd_rocclr_fillBufferAligned+0x284>
	s_cmp_gt_i32 s14, 1                                        // 0000000056BC: BF02810E
	s_mov_b32 s0, -1                                           // 0000000056C0: BE8000C1
	s_cbranch_scc0 40                                          // 0000000056C4: BFA10028 <__amd_rocclr_fillBufferAligned+0x268>
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000056C8: BF8700A1
	v_lshl_add_u64 v[2:3], v[0:1], 2, s[2:3]                   // 0000000056CC: D6520002 00090500
	s_mov_b32 s7, exec_lo                                      // 0000000056D4: BE87007E
	v_cmpx_gt_u64_e64 s[8:9], v[2:3]                           // 0000000056D8: D4DC007E 02020408
	s_cbranch_execz 31                                         // 0000000056E0: BFA5001F <__amd_rocclr_fillBufferAligned+0x260>
	s_cmp_lg_u32 s6, 0                                         // 0000000056E4: BF078006
	s_mov_b32 s11, 0                                           // 0000000056E8: BE8B0080
	s_cselect_b32 s16, -1, 0                                   // 0000000056EC: 981080C1
	s_lshl_b64 s[0:1], s[10:11], 2                             // 0000000056F0: 8480820A
	s_branch 6                                                 // 0000000056F4: BFA00006 <__amd_rocclr_fillBufferAligned+0x210>
	v_add_nc_u64_e32 v[2:3], s[0:1], v[2:3]                    // 0000000056F8: 50040400
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000056FC: BF8704A1
	v_cmp_le_u64_e32 vcc_lo, s[8:9], v[2:3]                    // 000000005700: 7CB60408
	s_or_b32 s11, vcc_lo, s11                                  // 000000005704: 8C0B0B6A
	s_and_not1_b32 exec_lo, exec_lo, s11                       // 000000005708: 917E0B7E
	s_cbranch_execz 20                                         // 00000000570C: BFA50014 <__amd_rocclr_fillBufferAligned+0x260>
	v_mov_b64_e32 v[4:5], v[2:3]                               // 000000005710: 7E083B02
	s_and_not1_b32 vcc_lo, exec_lo, s16                        // 000000005714: 916A107E
	s_mov_b64 s[12:13], s[4:5]                                 // 000000005718: BE8C0104
	s_mov_b32 s17, s6                                          // 00000000571C: BE910006
	s_cbranch_vccnz 65525                                      // 000000005720: BFA4FFF5 <__amd_rocclr_fillBufferAligned+0x1f8>
	s_load_b32 s18, s[12:13], 0x0 nv                           // 000000005724: F4100486 F8000000
	s_add_co_i32 s17, s17, -1                                  // 00000000572C: 8111C111
	s_wait_xcnt 0x0                                            // 000000005730: BFC50000
	s_add_nc_u64 s[12:13], s[12:13], 4                         // 000000005734: A98C840C
	s_cmp_eq_u32 s17, 0                                        // 000000005738: BF068011
	s_wait_kmcnt 0x0                                           // 00000000573C: BFC70000
	v_mov_b32_e32 v6, s18                                      // 000000005740: 7E0C0212
	global_store_b32 v[4:5], v6, off                           // 000000005744: EE06807C 03000000 00000004
	s_wait_xcnt 0x0                                            // 000000005750: BFC50000
	v_add_nc_u64_e32 v[4:5], 4, v[4:5]                         // 000000005754: 50080884
	s_cbranch_scc0 65522                                       // 000000005758: BFA1FFF2 <__amd_rocclr_fillBufferAligned+0x224>
	s_branch 65510                                             // 00000000575C: BFA0FFE6 <__amd_rocclr_fillBufferAligned+0x1f8>
	s_or_b32 exec_lo, exec_lo, s7                              // 000000005760: 8C7E077E
	s_mov_b32 s0, 0                                            // 000000005764: BE800080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005768: BF870009
	s_and_not1_b32 vcc_lo, exec_lo, s0                         // 00000000576C: 916A007E
	s_mov_b32 s7, 0                                            // 000000005770: BE870080
	s_cbranch_vccnz 3                                          // 000000005774: BFA40003 <__amd_rocclr_fillBufferAligned+0x284>
	s_cmp_lg_u32 s14, 1                                        // 000000005778: BF07810E
	s_mov_b32 s7, -1                                           // 00000000577C: BE8700C1
	s_cselect_b32 s15, -1, 0                                   // 000000005780: 980F80C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005784: BF870009
	s_and_b32 vcc_lo, exec_lo, s15                             // 000000005788: 8B6A0F7E
	s_cbranch_vccz 37                                          // 00000000578C: BFA30025 <__amd_rocclr_fillBufferAligned+0x324>
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000005790: BF8700A1
	v_add_nc_u64_e32 v[2:3], s[2:3], v[0:1]                    // 000000005794: 50040002
	s_mov_b32 s7, exec_lo                                      // 000000005798: BE87007E
	v_cmpx_gt_u64_e64 s[8:9], v[2:3]                           // 00000000579C: D4DC007E 02020408
	s_cbranch_execz 29                                         // 0000000057A4: BFA5001D <__amd_rocclr_fillBufferAligned+0x31c>
	v_mov_b32_e32 v4, 0                                        // 0000000057A8: 7E080280
	s_cmp_lg_u32 s6, 0                                         // 0000000057AC: BF078006
	s_mov_b32 s11, 0                                           // 0000000057B0: BE8B0080
	s_cselect_b32 s12, -1, 0                                   // 0000000057B4: 980C80C1
	s_mov_b32 s13, s11                                         // 0000000057B8: BE8D000B
	s_branch 6                                                 // 0000000057BC: BFA00006 <__amd_rocclr_fillBufferAligned+0x2d8>
	v_add_nc_u64_e32 v[2:3], s[10:11], v[2:3]                  // 0000000057C0: 5004040A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000057C4: BF8704A1
	v_cmp_le_u64_e32 vcc_lo, s[8:9], v[2:3]                    // 0000000057C8: 7CB60408
	s_or_b32 s13, vcc_lo, s13                                  // 0000000057CC: 8C0D0D6A
	s_and_not1_b32 exec_lo, exec_lo, s13                       // 0000000057D0: 917E0D7E
	s_cbranch_execz 17                                         // 0000000057D4: BFA50011 <__amd_rocclr_fillBufferAligned+0x31c>
	s_and_not1_b32 vcc_lo, exec_lo, s12                        // 0000000057D8: 916A0C7E
	s_cbranch_vccnz 65528                                      // 0000000057DC: BFA4FFF8 <__amd_rocclr_fillBufferAligned+0x2c0>
	s_mov_b64 s[0:1], 0                                        // 0000000057E0: BE800180
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 0000000057E4: BF8704C9
	s_add_nc_u64 s[14:15], s[4:5], s[0:1]                      // 0000000057E8: A98E0004
	v_add_nc_u64_e32 v[6:7], s[0:1], v[2:3]                    // 0000000057EC: 500C0400
	global_load_u8 v5, v4, s[14:15] nv                         // 0000000057F0: EE04008E 00000005 00000004
	s_add_nc_u64 s[0:1], s[0:1], 1                             // 0000000057FC: A9808100
	s_cmp_eq_u32 s6, s0                                        // 000000005800: BF060006
	s_wait_loadcnt 0x0                                         // 000000005804: BFC00000
	global_store_b8 v[6:7], v5, off                            // 000000005808: EE06007C 02800000 00000006
	s_cbranch_scc0 65523                                       // 000000005814: BFA1FFF3 <__amd_rocclr_fillBufferAligned+0x2e4>
	s_branch 65513                                             // 000000005818: BFA0FFE9 <__amd_rocclr_fillBufferAligned+0x2c0>
	s_or_b32 exec_lo, exec_lo, s7                              // 00000000581C: 8C7E077E
	s_mov_b32 s7, 0                                            // 000000005820: BE870080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005824: BF870009
	s_and_not1_b32 vcc_lo, exec_lo, s7                         // 000000005828: 916A077E
	s_cbranch_vccnz 39                                         // 00000000582C: BFA40027 <__amd_rocclr_fillBufferAligned+0x3cc>
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000005830: BF8700A1
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[2:3]                   // 000000005834: D6520000 00090300
	s_mov_b32 s0, exec_lo                                      // 00000000583C: BE80007E
	v_cmpx_gt_u64_e64 s[8:9], v[0:1]                           // 000000005840: D4DC007E 02020008
	s_cbranch_execz 32                                         // 000000005848: BFA50020 <__amd_rocclr_fillBufferAligned+0x3cc>
	v_mov_b32_e32 v4, 0                                        // 00000000584C: 7E080280
	s_cmp_lg_u32 s6, 0                                         // 000000005850: BF078006
	s_mov_b32 s11, 0                                           // 000000005854: BE8B0080
	s_cselect_b32 s7, -1, 0                                    // 000000005858: 980780C1
	s_lshl_b64 s[0:1], s[10:11], 1                             // 00000000585C: 8480810A
	s_branch 6                                                 // 000000005860: BFA00006 <__amd_rocclr_fillBufferAligned+0x37c>
	v_add_nc_u64_e32 v[0:1], s[0:1], v[0:1]                    // 000000005864: 50000000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000005868: BF8704A1
	v_cmp_le_u64_e32 vcc_lo, s[8:9], v[0:1]                    // 00000000586C: 7CB60008
	s_or_b32 s11, vcc_lo, s11                                  // 000000005870: 8C0B0B6A
	s_and_not1_b32 exec_lo, exec_lo, s11                       // 000000005874: 917E0B7E
	s_cbranch_execz 20                                         // 000000005878: BFA50014 <__amd_rocclr_fillBufferAligned+0x3cc>
	v_mov_b64_e32 v[2:3], v[0:1]                               // 00000000587C: 7E043B00
	s_and_not1_b32 vcc_lo, exec_lo, s7                         // 000000005880: 916A077E
	s_mov_b64 s[2:3], s[4:5]                                   // 000000005884: BE820104
	s_mov_b32 s10, s6                                          // 000000005888: BE8A0006
	s_cbranch_vccnz 65525                                      // 00000000588C: BFA4FFF5 <__amd_rocclr_fillBufferAligned+0x364>
	global_load_u16 v5, v4, s[2:3] nv                          // 000000005890: EE048082 00000005 00000004
	s_add_co_i32 s10, s10, -1                                  // 00000000589C: 810AC10A
	s_wait_xcnt 0x0                                            // 0000000058A0: BFC50000
	s_add_nc_u64 s[2:3], s[2:3], 2                             // 0000000058A4: A9828202
	s_cmp_eq_u32 s10, 0                                        // 0000000058A8: BF06800A
	s_wait_loadcnt 0x0                                         // 0000000058AC: BFC00000
	global_store_b16 v[2:3], v5, off                           // 0000000058B0: EE06407C 02800000 00000002
	s_wait_xcnt 0x0                                            // 0000000058BC: BFC50000
	v_add_nc_u64_e32 v[2:3], 2, v[2:3]                         // 0000000058C0: 50040482
	s_cbranch_scc0 65522                                       // 0000000058C4: BFA1FFF2 <__amd_rocclr_fillBufferAligned+0x390>
	s_branch 65510                                             // 0000000058C8: BFA0FFE6 <__amd_rocclr_fillBufferAligned+0x364>
	s_endpgm                                                   // 0000000058CC: BFB00000
	s_nop 0                                                    // 0000000058D0: BF800000
	s_nop 0                                                    // 0000000058D4: BF800000
	s_nop 0                                                    // 0000000058D8: BF800000
	s_nop 0                                                    // 0000000058DC: BF800000
	s_nop 0                                                    // 0000000058E0: BF800000
	s_nop 0                                                    // 0000000058E4: BF800000
	s_nop 0                                                    // 0000000058E8: BF800000
	s_nop 0                                                    // 0000000058EC: BF800000
	s_nop 0                                                    // 0000000058F0: BF800000
	s_nop 0                                                    // 0000000058F4: BF800000
	s_nop 0                                                    // 0000000058F8: BF800000
	s_nop 0                                                    // 0000000058FC: BF800000

0000000000005900 <__amd_rocclr_fillBufferAligned2D>:
; __amd_rocclr_fillBufferAligned2D():
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1       // 000000005900: B9800641 00000001
	s_clause 0x1                                               // 000000005908: BF850001
	s_load_b32 s13, s[0:1], 0x5c nv                            // 00000000590C: F4100340 F800005C
	s_load_b128 s[24:27], s[0:1], 0x78 nv                      // 000000005914: F4104600 F8000078
	v_and_b32_e32 v1, 0x3ff, v0                                // 00000000591C: 360200FF 000003FF
	s_load_b128 s[20:23], s[0:1], 0x40 nv                      // 000000005924: F4104500 F8000040
	v_mov_b32_e32 v3, 0                                        // 00000000592C: 7E060280
	v_bfe_u32 v4, v0, 10, 10                                   // 000000005930: D6100004 02291500
	s_wait_kmcnt 0x0                                           // 000000005938: BFC70000
	s_and_b32 s18, s13, 0xffff                                 // 00000000593C: 8B12FF0D 0000FFFF
	s_lshr_b32 s0, s13, 16                                     // 000000005944: 8500900D
	v_mad_u32 v2, ttmp9, s18, v1                               // 000000005948: D6350002 04042475
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000005950: BF870121
	v_add_nc_u64_e32 v[0:1], s[24:25], v[2:3]                  // 000000005954: 50000418
	v_mad_u32 v2, ttmp7, s0, v4                                // 000000005958: D6350002 04100073
	v_cmp_gt_u64_e32 vcc_lo, s[16:17], v[0:1]                  // 000000005960: 7CB80010
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005964: BF870092
	v_add_nc_u64_e32 v[2:3], s[26:27], v[2:3]                  // 000000005968: 5004041A
	v_cmp_gt_u64_e64 s0, s[20:21], v[2:3]                      // 00000000596C: D45C0000 02020414
	s_and_b32 s0, vcc_lo, s0                                   // 000000005974: 8B00006A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005978: BF870009
	s_and_saveexec_b32 s1, s0                                  // 00000000597C: BE812000
	s_cbranch_execz 115                                        // 000000005980: BFA50073 <__amd_rocclr_fillBufferAligned2D+0x250>
	v_mad_nc_u64_u32 v[0:1], v2, s22, v[0:1]                   // 000000005984: D6FA0000 04002D02
	s_cmp_eq_u64 s[8:9], 0                                     // 00000000598C: BF108008
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005990: BF870091
	v_mad_u32 v1, v3, s22, v1                                  // 000000005994: D6350001 04042D03
	v_mad_u32 v1, v2, s23, v1                                  // 00000000599C: D6350001 04042F02
	s_cbranch_scc1 27                                          // 0000000059A4: BFA2001B <__amd_rocclr_fillBufferAligned2D+0x114>
	s_cmp_eq_u32 s12, 0                                        // 0000000059A8: BF06800C
	s_mov_b32 s13, 0                                           // 0000000059AC: BE8D0080
	s_cbranch_scc1 21                                          // 0000000059B0: BFA20015 <__amd_rocclr_fillBufferAligned2D+0x108>
	s_lshl_b64 s[0:1], s[14:15], 3                             // 0000000059B4: 8480830E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000059B8: BF870009
	s_add_nc_u64 s[0:1], s[8:9], s[0:1]                        // 0000000059BC: A9800008
	s_mov_b32 s8, s12                                          // 0000000059C0: BE88000C
	v_lshl_add_u64 v[2:3], v[0:1], 3, s[0:1]                   // 0000000059C4: D6520002 00010700
	s_mov_b64 s[0:1], s[10:11]                                 // 0000000059CC: BE80010A
	s_load_b64 s[16:17], s[0:1], 0x0 nv                        // 0000000059D0: F4102400 F8000000
	s_add_co_i32 s8, s8, -1                                    // 0000000059D8: 8108C108
	s_wait_xcnt 0x0                                            // 0000000059DC: BFC50000
	s_add_nc_u64 s[0:1], s[0:1], 8                             // 0000000059E0: A9808800
	s_cmp_eq_u32 s8, 0                                         // 0000000059E4: BF068008
	s_wait_kmcnt 0x0                                           // 0000000059E8: BFC70000
	v_mov_b64_e32 v[4:5], s[16:17]                             // 0000000059EC: 7E083A10
	global_store_b64 v[2:3], v[4:5], off                       // 0000000059F0: EE06C07C 02000000 00000002
	s_wait_xcnt 0x0                                            // 0000000059FC: BFC50000
	v_add_nc_u64_e32 v[2:3], 8, v[2:3]                         // 000000005A00: 50040488
	s_cbranch_scc0 65522                                       // 000000005A04: BFA1FFF2 <__amd_rocclr_fillBufferAligned2D+0xd0>
	s_and_not1_b32 vcc_lo, exec_lo, s13                        // 000000005A08: 916A0D7E
	s_cbranch_vccz 1                                           // 000000005A0C: BFA30001 <__amd_rocclr_fillBufferAligned2D+0x114>
	s_branch 79                                                // 000000005A10: BFA0004F <__amd_rocclr_fillBufferAligned2D+0x250>
	s_cmp_eq_u64 s[6:7], 0                                     // 000000005A14: BF108006
	s_cbranch_scc1 27                                          // 000000005A18: BFA2001B <__amd_rocclr_fillBufferAligned2D+0x188>
	s_cmp_eq_u32 s12, 0                                        // 000000005A1C: BF06800C
	s_mov_b32 s8, 0                                            // 000000005A20: BE880080
	s_cbranch_scc1 21                                          // 000000005A24: BFA20015 <__amd_rocclr_fillBufferAligned2D+0x17c>
	s_lshl_b64 s[0:1], s[14:15], 2                             // 000000005A28: 8480820E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005A2C: BF870009
	s_add_nc_u64 s[0:1], s[6:7], s[0:1]                        // 000000005A30: A9800006
	s_mov_b32 s6, s12                                          // 000000005A34: BE86000C
	v_lshl_add_u64 v[2:3], v[0:1], 2, s[0:1]                   // 000000005A38: D6520002 00010500
	s_mov_b64 s[0:1], s[10:11]                                 // 000000005A40: BE80010A
	s_load_b32 s7, s[0:1], 0x0 nv                              // 000000005A44: F41001C0 F8000000
	s_add_co_i32 s6, s6, -1                                    // 000000005A4C: 8106C106
	s_wait_xcnt 0x0                                            // 000000005A50: BFC50000
	s_add_nc_u64 s[0:1], s[0:1], 4                             // 000000005A54: A9808400
	s_cmp_eq_u32 s6, 0                                         // 000000005A58: BF068006
	s_wait_kmcnt 0x0                                           // 000000005A5C: BFC70000
	v_mov_b32_e32 v4, s7                                       // 000000005A60: 7E080207
	global_store_b32 v[2:3], v4, off                           // 000000005A64: EE06807C 02000000 00000002
	s_wait_xcnt 0x0                                            // 000000005A70: BFC50000
	v_add_nc_u64_e32 v[2:3], 4, v[2:3]                         // 000000005A74: 50040484
	s_cbranch_scc0 65522                                       // 000000005A78: BFA1FFF2 <__amd_rocclr_fillBufferAligned2D+0x144>
	s_and_not1_b32 vcc_lo, exec_lo, s8                         // 000000005A7C: 916A087E
	s_cbranch_vccz 1                                           // 000000005A80: BFA30001 <__amd_rocclr_fillBufferAligned2D+0x188>
	s_branch 50                                                // 000000005A84: BFA00032 <__amd_rocclr_fillBufferAligned2D+0x250>
	s_cmp_eq_u64 s[4:5], 0                                     // 000000005A88: BF108004
	s_cbranch_scc1 27                                          // 000000005A8C: BFA2001B <__amd_rocclr_fillBufferAligned2D+0x1fc>
	s_cmp_eq_u32 s12, 0                                        // 000000005A90: BF06800C
	s_mov_b32 s6, 0                                            // 000000005A94: BE860080
	s_cbranch_scc1 21                                          // 000000005A98: BFA20015 <__amd_rocclr_fillBufferAligned2D+0x1f0>
	s_lshl_b64 s[0:1], s[14:15], 1                             // 000000005A9C: 8480810E
	v_mov_b32_e32 v4, 0                                        // 000000005AA0: 7E080280
	s_add_nc_u64 s[0:1], s[4:5], s[0:1]                        // 000000005AA4: A9800004
	s_mov_b32 s4, s12                                          // 000000005AA8: BE84000C
	v_lshl_add_u64 v[2:3], v[0:1], 1, s[0:1]                   // 000000005AAC: D6520002 00010300
	s_mov_b64 s[0:1], s[10:11]                                 // 000000005AB4: BE80010A
	global_load_u16 v5, v4, s[0:1] nv                          // 000000005AB8: EE048080 00000005 00000004
	s_add_co_i32 s4, s4, -1                                    // 000000005AC4: 8104C104
	s_wait_xcnt 0x0                                            // 000000005AC8: BFC50000
	s_add_nc_u64 s[0:1], s[0:1], 2                             // 000000005ACC: A9808200
	s_cmp_eq_u32 s4, 0                                         // 000000005AD0: BF068004
	s_wait_loadcnt 0x0                                         // 000000005AD4: BFC00000
	global_store_b16 v[2:3], v5, off                           // 000000005AD8: EE06407C 02800000 00000002
	s_wait_xcnt 0x0                                            // 000000005AE4: BFC50000
	v_add_nc_u64_e32 v[2:3], 2, v[2:3]                         // 000000005AE8: 50040482
	s_cbranch_scc0 65522                                       // 000000005AEC: BFA1FFF2 <__amd_rocclr_fillBufferAligned2D+0x1b8>
	s_and_not1_b32 vcc_lo, exec_lo, s6                         // 000000005AF0: 916A067E
	s_cbranch_vccz 1                                           // 000000005AF4: BFA30001 <__amd_rocclr_fillBufferAligned2D+0x1fc>
	s_branch 21                                                // 000000005AF8: BFA00015 <__amd_rocclr_fillBufferAligned2D+0x250>
	s_cmp_eq_u64 s[2:3], 0                                     // 000000005AFC: BF108002
	s_cbranch_scc1 19                                          // 000000005B00: BFA20013 <__amd_rocclr_fillBufferAligned2D+0x250>
	s_cmp_eq_u32 s12, 0                                        // 000000005B04: BF06800C
	s_cbranch_scc1 17                                          // 000000005B08: BFA20011 <__amd_rocclr_fillBufferAligned2D+0x250>
	s_add_nc_u64 s[0:1], s[2:3], s[14:15]                      // 000000005B0C: A9800E02
	v_mov_b32_e32 v2, 0                                        // 000000005B10: 7E040280
	v_add_nc_u64_e32 v[0:1], s[0:1], v[0:1]                    // 000000005B14: 50000000
	global_load_u8 v3, v2, s[10:11] nv                         // 000000005B18: EE04008A 00000003 00000002
	s_add_co_i32 s12, s12, -1                                  // 000000005B24: 810CC10C
	s_wait_xcnt 0x0                                            // 000000005B28: BFC50000
	s_add_nc_u64 s[10:11], s[10:11], 1                         // 000000005B2C: A98A810A
	s_cmp_eq_u32 s12, 0                                        // 000000005B30: BF06800C
	s_wait_loadcnt 0x0                                         // 000000005B34: BFC00000
	global_store_b8 v[0:1], v3, off                            // 000000005B38: EE06007C 01800000 00000000
	s_wait_xcnt 0x0                                            // 000000005B44: BFC50000
	v_add_nc_u64_e32 v[0:1], 1, v[0:1]                         // 000000005B48: 50000081
	s_cbranch_scc0 65522                                       // 000000005B4C: BFA1FFF2 <__amd_rocclr_fillBufferAligned2D+0x218>
	s_endpgm                                                   // 000000005B50: BFB00000
	s_nop 0                                                    // 000000005B54: BF800000
	s_nop 0                                                    // 000000005B58: BF800000
	s_nop 0                                                    // 000000005B5C: BF800000
	s_nop 0                                                    // 000000005B60: BF800000
	s_nop 0                                                    // 000000005B64: BF800000
	s_nop 0                                                    // 000000005B68: BF800000
	s_nop 0                                                    // 000000005B6C: BF800000
	s_nop 0                                                    // 000000005B70: BF800000
	s_nop 0                                                    // 000000005B74: BF800000
	s_nop 0                                                    // 000000005B78: BF800000
	s_nop 0                                                    // 000000005B7C: BF800000
	s_nop 0                                                    // 000000005B80: BF800000
	s_nop 0                                                    // 000000005B84: BF800000
	s_nop 0                                                    // 000000005B88: BF800000
	s_nop 0                                                    // 000000005B8C: BF800000
	s_nop 0                                                    // 000000005B90: BF800000
	s_nop 0                                                    // 000000005B94: BF800000
	s_nop 0                                                    // 000000005B98: BF800000
	s_nop 0                                                    // 000000005B9C: BF800000
	s_nop 0                                                    // 000000005BA0: BF800000
	s_nop 0                                                    // 000000005BA4: BF800000
	s_nop 0                                                    // 000000005BA8: BF800000
	s_nop 0                                                    // 000000005BAC: BF800000
	s_nop 0                                                    // 000000005BB0: BF800000
	s_nop 0                                                    // 000000005BB4: BF800000
	s_nop 0                                                    // 000000005BB8: BF800000
	s_nop 0                                                    // 000000005BBC: BF800000
	s_nop 0                                                    // 000000005BC0: BF800000
	s_nop 0                                                    // 000000005BC4: BF800000
	s_nop 0                                                    // 000000005BC8: BF800000
	s_nop 0                                                    // 000000005BCC: BF800000
	s_nop 0                                                    // 000000005BD0: BF800000
	s_nop 0                                                    // 000000005BD4: BF800000
	s_nop 0                                                    // 000000005BD8: BF800000
	s_nop 0                                                    // 000000005BDC: BF800000
	s_nop 0                                                    // 000000005BE0: BF800000
	s_nop 0                                                    // 000000005BE4: BF800000
	s_nop 0                                                    // 000000005BE8: BF800000
	s_nop 0                                                    // 000000005BEC: BF800000
	s_nop 0                                                    // 000000005BF0: BF800000
	s_nop 0                                                    // 000000005BF4: BF800000
	s_nop 0                                                    // 000000005BF8: BF800000
	s_nop 0                                                    // 000000005BFC: BF800000

0000000000005c00 <__amd_rocclr_copyBuffer>:
; __amd_rocclr_copyBuffer():
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1       // 000000005C00: B9800641 00000001
	v_mad_u32 v0, s13, ttmp9, v0                               // 000000005C08: D6350000 0400EA0D
	v_mov_b32_e32 v1, 0                                        // 000000005C10: 7E020280
	s_cmp_eq_u32 s9, 16                                        // 000000005C14: BF069009
	s_mov_b32 s0, -1                                           // 000000005C18: BE8000C1
	s_cbranch_scc1 35                                          // 000000005C1C: BFA20023 <__amd_rocclr_copyBuffer+0xac>
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000005C20: BF8700A1
	v_lshl_add_u64 v[2:3], v[0:1], 2, s[4:5]                   // 000000005C24: D6520002 00110500
	s_mov_b32 s9, exec_lo                                      // 000000005C2C: BE89007E
	v_cmpx_gt_u64_e64 s[10:11], v[2:3]                         // 000000005C30: D4DC007E 0202040A
	s_cbranch_execz 26                                         // 000000005C38: BFA5001A <__amd_rocclr_copyBuffer+0xa4>
	v_mov_b64_e32 v[6:7], v[0:1]                               // 000000005C3C: 7E0C3B00
	v_lshl_add_u64 v[4:5], v[0:1], 2, s[2:3]                   // 000000005C40: D6520004 00090500
	s_mov_b32 s13, 0                                           // 000000005C48: BE8D0080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005C4C: BF870009
	s_lshl_b64 s[0:1], s[12:13], 2                             // 000000005C50: 8480820C
	s_mov_b32 s14, s13                                         // 000000005C54: BE8E000D
	global_load_b32 v10, v[4:5], off                           // 000000005C58: EE05007C 0000000A 00000004
	v_add_nc_u64_e32 v[6:7], s[12:13], v[6:7]                  // 000000005C64: 500C0C0C
	s_wait_xcnt 0x0                                            // 000000005C68: BFC50000
	v_add_nc_u64_e32 v[4:5], s[0:1], v[4:5]                    // 000000005C6C: 50080800
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005C70: BF870092
	v_lshl_add_u64 v[8:9], v[6:7], 2, s[4:5]                   // 000000005C74: D6520008 00110506
	v_cmp_le_u64_e32 vcc_lo, s[10:11], v[8:9]                  // 000000005C7C: 7CB6100A
	s_or_b32 s14, vcc_lo, s14                                  // 000000005C80: 8C0E0E6A
	s_wait_loadcnt 0x0                                         // 000000005C84: BFC00000
	global_store_b32 v[2:3], v10, off                          // 000000005C88: EE06807C 05000000 00000002
	s_wait_xcnt 0x0                                            // 000000005C94: BFC50000
	v_add_nc_u64_e32 v[2:3], s[0:1], v[2:3]                    // 000000005C98: 50040400
	s_and_not1_b32 exec_lo, exec_lo, s14                       // 000000005C9C: 917E0E7E
	s_cbranch_execnz 65517                                     // 000000005CA0: BFA6FFED <__amd_rocclr_copyBuffer+0x58>
	s_or_b32 exec_lo, exec_lo, s9                              // 000000005CA4: 8C7E097E
	s_mov_b32 s0, 0                                            // 000000005CA8: BE800080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005CAC: BF870009
	s_and_not1_b32 vcc_lo, exec_lo, s0                         // 000000005CB0: 916A007E
	s_cbranch_vccnz 34                                         // 000000005CB4: BFA40022 <__amd_rocclr_copyBuffer+0x140>
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000005CB8: BF8700A2
	v_lshl_add_u64 v[2:3], v[0:1], 4, s[4:5]                   // 000000005CBC: D6520002 00110900
	s_mov_b32 s9, exec_lo                                      // 000000005CC4: BE89007E
	v_cmpx_gt_u64_e64 s[10:11], v[2:3]                         // 000000005CC8: D4DC007E 0202040A
	s_cbranch_execz 26                                         // 000000005CD0: BFA5001A <__amd_rocclr_copyBuffer+0x13c>
	v_mov_b64_e32 v[6:7], v[0:1]                               // 000000005CD4: 7E0C3B00
	v_lshl_add_u64 v[4:5], v[0:1], 4, s[2:3]                   // 000000005CD8: D6520004 00090900
	s_mov_b32 s13, 0                                           // 000000005CE0: BE8D0080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005CE4: BF870009
	s_lshl_b64 s[0:1], s[12:13], 4                             // 000000005CE8: 8480840C
	s_mov_b32 s14, s13                                         // 000000005CEC: BE8E000D
	global_load_b128 v[8:11], v[4:5], off                      // 000000005CF0: EE05C07C 00000008 00000004
	v_add_nc_u64_e32 v[6:7], s[12:13], v[6:7]                  // 000000005CFC: 500C0C0C
	s_wait_xcnt 0x0                                            // 000000005D00: BFC50000
	v_add_nc_u64_e32 v[4:5], s[0:1], v[4:5]                    // 000000005D04: 50080800
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000005D08: BF870092
	v_lshl_add_u64 v[12:13], v[6:7], 4, s[4:5]                 // 000000005D0C: D652000C 00110906
	v_cmp_le_u64_e32 vcc_lo, s[10:11], v[12:13]                // 000000005D14: 7CB6180A
	s_or_b32 s14, vcc_lo, s14                                  // 000000005D18: 8C0E0E6A
	s_wait_loadcnt 0x0                                         // 000000005D1C: BFC00000
	global_store_b128 v[2:3], v[8:11], off                     // 000000005D20: EE07407C 04000000 00000002
	s_wait_xcnt 0x0                                            // 000000005D2C: BFC50000
	v_add_nc_u64_e32 v[2:3], s[0:1], v[2:3]                    // 000000005D30: 50040400
	s_and_not1_b32 exec_lo, exec_lo, s14                       // 000000005D34: 917E0E7E
	s_cbranch_execnz 65517                                     // 000000005D38: BFA6FFED <__amd_rocclr_copyBuffer+0xf0>
	s_or_b32 exec_lo, exec_lo, s9                              // 000000005D3C: 8C7E097E
	s_mov_b32 s9, 0                                            // 000000005D40: BE890080
	v_cmp_eq_u32_e32 vcc_lo, 0, v0                             // 000000005D44: 7C940080
	s_sub_nc_u64 s[0:1], s[6:7], s[8:9]                        // 000000005D48: AA000806
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000005D4C: BF8704A9
	v_cmp_lt_u64_e64 s0, s[0:1], s[6:7]                        // 000000005D50: D4590000 02000C00
	s_and_b32 s0, s0, vcc_lo                                   // 000000005D58: 8B006A00
	s_and_saveexec_b32 s1, s0                                  // 000000005D5C: BE812000
	s_cbranch_execz 21                                         // 000000005D60: BFA50015 <__amd_rocclr_copyBuffer+0x1b8>
	v_mov_b32_e32 v0, 0                                        // 000000005D64: 7E000280
	s_sub_nc_u64 s[0:1], 0, s[8:9]                             // 000000005D68: AA000880
	s_add_nc_u64 s[4:5], s[4:5], s[6:7]                        // 000000005D6C: A9840604
	s_add_nc_u64 s[2:3], s[2:3], s[6:7]                        // 000000005D70: A9820602
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005D74: BF870009
	s_add_nc_u64 s[6:7], s[2:3], s[0:1]                        // 000000005D78: A9860002
	global_load_u8 v1, v0, s[6:7]                              // 000000005D7C: EE040006 00000001 00000000
	s_wait_xcnt 0x0                                            // 000000005D88: BFC50000
	s_add_nc_u64 s[6:7], s[4:5], s[0:1]                        // 000000005D8C: A9860004
	s_add_co_u32 s0, s0, 1                                     // 000000005D90: 80008100
	s_add_co_ci_u32 s1, s1, 0                                  // 000000005D94: 82018001
	s_cselect_b32 s8, -1, 0                                    // 000000005D98: 980880C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005D9C: BF870009
	s_and_not1_b32 vcc_lo, exec_lo, s8                         // 000000005DA0: 916A087E
	s_wait_loadcnt 0x0                                         // 000000005DA4: BFC00000
	global_store_b8 v0, v1, s[6:7]                             // 000000005DA8: EE060006 00800000 00000000
	s_cbranch_vccnz 65519                                      // 000000005DB4: BFA4FFEF <__amd_rocclr_copyBuffer+0x174>
	s_endpgm                                                   // 000000005DB8: BFB00000
	s_nop 0                                                    // 000000005DBC: BF800000
	s_nop 0                                                    // 000000005DC0: BF800000
	s_nop 0                                                    // 000000005DC4: BF800000
	s_nop 0                                                    // 000000005DC8: BF800000
	s_nop 0                                                    // 000000005DCC: BF800000
	s_nop 0                                                    // 000000005DD0: BF800000
	s_nop 0                                                    // 000000005DD4: BF800000
	s_nop 0                                                    // 000000005DD8: BF800000
	s_nop 0                                                    // 000000005DDC: BF800000
	s_nop 0                                                    // 000000005DE0: BF800000
	s_nop 0                                                    // 000000005DE4: BF800000
	s_nop 0                                                    // 000000005DE8: BF800000
	s_nop 0                                                    // 000000005DEC: BF800000
	s_nop 0                                                    // 000000005DF0: BF800000
	s_nop 0                                                    // 000000005DF4: BF800000
	s_nop 0                                                    // 000000005DF8: BF800000
	s_nop 0                                                    // 000000005DFC: BF800000

0000000000005e00 <__amd_rocclr_copyBufferBatch>:
; __amd_rocclr_copyBufferBatch():
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1       // 000000005E00: B9800641 00000001
	s_mov_b32 s0, ttmp7                                        // 000000005E08: BE800073
	s_mov_b32 s1, 0                                            // 000000005E0C: BE810080
	v_mov_b32_e32 v1, 0                                        // 000000005E10: 7E020280
	s_lshl_b64 s[0:1], s[0:1], 5                               // 000000005E14: 84808500
	s_mov_b32 s6, s5                                           // 000000005E18: BE860005
	s_add_nc_u64 s[0:1], s[2:3], s[0:1]                        // 000000005E1C: A9800002
	s_load_b256 s[8:15], s[0:1], 0x0                           // 000000005E20: F4006200 F8000000
	v_mad_nc_u64_u32 v[2:3], s4, ttmp9, v[0:1]                 // 000000005E28: D6FA0002 0400EA04
	s_wait_xcnt 0x0                                            // 000000005E30: BFC50000
	s_mov_b32 s1, -1                                           // 000000005E34: BE8100C1
	s_wait_kmcnt 0x0                                           // 000000005E38: BFC70000
	s_delay_alu instid0(VALU_DEP_1)                            // 000000005E3C: BF870001
	v_cmp_gt_u64_e64 s0, s[12:13], v[2:3]                      // 000000005E40: D45C0000 0202040C
	s_cmp_eq_u32 s14, 16                                       // 000000005E48: BF06900E
	s_cbranch_scc1 29                                          // 000000005E4C: BFA2001D <__amd_rocclr_copyBufferBatch+0xc4>
	s_and_saveexec_b32 s1, s0                                  // 000000005E50: BE812000
	s_cbranch_execz 25                                         // 000000005E54: BFA50019 <__amd_rocclr_copyBufferBatch+0xbc>
	v_lshlrev_b64_e32 v[6:7], 2, v[2:3]                        // 000000005E58: 3E0C0482
	v_mov_b64_e32 v[8:9], v[2:3]                               // 000000005E5C: 7E103B02
	s_mov_b32 s7, 0                                            // 000000005E60: BE870080
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000005E64: BF870129
	s_lshl_b64 s[2:3], s[6:7], 2                               // 000000005E68: 84828206
	s_mov_b32 s4, s7                                           // 000000005E6C: BE840007
	v_add_nc_u64_e32 v[4:5], s[10:11], v[6:7]                  // 000000005E70: 50080C0A
	v_add_nc_u64_e32 v[6:7], s[8:9], v[6:7]                    // 000000005E74: 500C0C08
	global_load_b32 v1, v[6:7], off                            // 000000005E78: EE05007C 00000001 00000006
	v_add_nc_u64_e32 v[8:9], s[6:7], v[8:9]                    // 000000005E84: 50101006
	s_wait_xcnt 0x0                                            // 000000005E88: BFC50000
	v_add_nc_u64_e32 v[6:7], s[2:3], v[6:7]                    // 000000005E8C: 500C0C02
	s_delay_alu instid0(VALU_DEP_2)                            // 000000005E90: BF870002
	v_cmp_le_u64_e32 vcc_lo, s[12:13], v[8:9]                  // 000000005E94: 7CB6100C
	s_or_b32 s4, vcc_lo, s4                                    // 000000005E98: 8C04046A
	s_wait_loadcnt 0x0                                         // 000000005E9C: BFC00000
	global_store_b32 v[4:5], v1, off                           // 000000005EA0: EE06807C 00800000 00000004
	s_wait_xcnt 0x0                                            // 000000005EAC: BFC50000
	v_add_nc_u64_e32 v[4:5], s[2:3], v[4:5]                    // 000000005EB0: 50080802
	s_and_not1_b32 exec_lo, exec_lo, s4                        // 000000005EB4: 917E047E
	s_cbranch_execnz 65519                                     // 000000005EB8: BFA6FFEF <__amd_rocclr_copyBufferBatch+0x78>
	s_or_b32 exec_lo, exec_lo, s1                              // 000000005EBC: 8C7E017E
	s_mov_b32 s1, 0                                            // 000000005EC0: BE810080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005EC4: BF870009
	s_and_not1_b32 vcc_lo, exec_lo, s1                         // 000000005EC8: 916A017E
	s_cbranch_vccnz 27                                         // 000000005ECC: BFA4001B <__amd_rocclr_copyBufferBatch+0x13c>
	s_and_saveexec_b32 s2, s0                                  // 000000005ED0: BE822000
	s_cbranch_execz 24                                         // 000000005ED4: BFA50018 <__amd_rocclr_copyBufferBatch+0x138>
	v_lshlrev_b64_e32 v[6:7], 4, v[2:3]                        // 000000005ED8: 3E0C0484
	s_mov_b32 s7, 0                                            // 000000005EDC: BE870080
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000005EE0: BF8700A9
	s_lshl_b64 s[0:1], s[6:7], 4                               // 000000005EE4: 84808406
	s_mov_b32 s3, s7                                           // 000000005EE8: BE830007
	v_add_nc_u64_e32 v[4:5], s[10:11], v[6:7]                  // 000000005EEC: 50080C0A
	v_add_nc_u64_e32 v[6:7], s[8:9], v[6:7]                    // 000000005EF0: 500C0C08
	global_load_b128 v[8:11], v[6:7], off                      // 000000005EF4: EE05C07C 00000008 00000006
	v_add_nc_u64_e32 v[2:3], s[6:7], v[2:3]                    // 000000005F00: 50040406
	s_wait_xcnt 0x0                                            // 000000005F04: BFC50000
	v_add_nc_u64_e32 v[6:7], s[0:1], v[6:7]                    // 000000005F08: 500C0C00
	s_delay_alu instid0(VALU_DEP_2)                            // 000000005F0C: BF870002
	v_cmp_le_u64_e32 vcc_lo, s[12:13], v[2:3]                  // 000000005F10: 7CB6040C
	s_or_b32 s3, vcc_lo, s3                                    // 000000005F14: 8C03036A
	s_wait_loadcnt 0x0                                         // 000000005F18: BFC00000
	global_store_b128 v[4:5], v[8:11], off                     // 000000005F1C: EE07407C 04000000 00000004
	s_wait_xcnt 0x0                                            // 000000005F28: BFC50000
	v_add_nc_u64_e32 v[4:5], s[0:1], v[4:5]                    // 000000005F2C: 50080800
	s_and_not1_b32 exec_lo, exec_lo, s3                        // 000000005F30: 917E037E
	s_cbranch_execnz 65519                                     // 000000005F34: BFA6FFEF <__amd_rocclr_copyBufferBatch+0xf4>
	s_or_b32 exec_lo, exec_lo, s2                              // 000000005F38: 8C7E027E
	v_or_b32_e32 v0, ttmp9, v0                                 // 000000005F3C: 38000075
	s_cmp_lg_u32 s15, 0                                        // 000000005F40: BF07800F
	s_mov_b32 s1, 0                                            // 000000005F44: BE810080
	s_cselect_b32 s0, -1, 0                                    // 000000005F48: 980080C1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000005F4C: BF8704A1
	v_cmp_eq_u32_e32 vcc_lo, 0, v0                             // 000000005F50: 7C940080
	s_and_b32 s0, vcc_lo, s0                                   // 000000005F54: 8B00006A
	s_and_saveexec_b32 s2, s0                                  // 000000005F58: BE822000
	s_cbranch_execz 27                                         // 000000005F5C: BFA5001B <__amd_rocclr_copyBufferBatch+0x1cc>
	s_mov_b32 s0, s14                                          // 000000005F60: BE80000E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000005F64: BF8704A9
	s_mul_u64 s[4:5], s[12:13], s[0:1]                         // 000000005F68: AA84000C
	s_mov_b32 s0, s15                                          // 000000005F6C: BE80000F
	s_add_nc_u64 s[2:3], s[4:5], s[0:1]                        // 000000005F70: A9820004
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000005F74: BF870009
	v_cmp_ge_u64_e64 s2, s[4:5], s[2:3]                        // 000000005F78: D45E0002 02000404
	s_and_b32 vcc_lo, exec_lo, s2                              // 000000005F80: 8B6A027E
	s_cbranch_vccnz 17                                         // 000000005F84: BFA40011 <__amd_rocclr_copyBufferBatch+0x1cc>
	v_mov_b32_e32 v0, 0                                        // 000000005F88: 7E000280
	s_add_nc_u64 s[2:3], s[10:11], s[4:5]                      // 000000005F8C: A982040A
	s_add_nc_u64 s[4:5], s[8:9], s[4:5]                        // 000000005F90: A9840408
	global_load_u8 v1, v0, s[4:5]                              // 000000005F94: EE040004 00000001 00000000
	s_add_nc_u64 s[0:1], s[0:1], -1                            // 000000005FA0: A980C100
	s_wait_xcnt 0x0                                            // 000000005FA4: BFC50000
	s_add_nc_u64 s[4:5], s[4:5], 1                             // 000000005FA8: A9848104
	s_cmp_lg_u64 s[0:1], 0                                     // 000000005FAC: BF118000
	s_wait_loadcnt 0x0                                         // 000000005FB0: BFC00000
	global_store_b8 v0, v1, s[2:3]                             // 000000005FB4: EE060002 00800000 00000000
	s_wait_xcnt 0x0                                            // 000000005FC0: BFC50000
	s_add_nc_u64 s[2:3], s[2:3], 1                             // 000000005FC4: A9828102
	s_cbranch_scc1 65522                                       // 000000005FC8: BFA2FFF2 <__amd_rocclr_copyBufferBatch+0x194>
	s_endpgm                                                   // 000000005FCC: BFB00000
	s_nop 0                                                    // 000000005FD0: BF800000
	s_nop 0                                                    // 000000005FD4: BF800000
	s_nop 0                                                    // 000000005FD8: BF800000
	s_nop 0                                                    // 000000005FDC: BF800000
	s_nop 0                                                    // 000000005FE0: BF800000
	s_nop 0                                                    // 000000005FE4: BF800000
	s_nop 0                                                    // 000000005FE8: BF800000
	s_nop 0                                                    // 000000005FEC: BF800000
	s_nop 0                                                    // 000000005FF0: BF800000
	s_nop 0                                                    // 000000005FF4: BF800000
	s_nop 0                                                    // 000000005FF8: BF800000
	s_nop 0                                                    // 000000005FFC: BF800000

0000000000006000 <__amd_rocclr_copyBufferAligned>:
; __amd_rocclr_copyBufferAligned():
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1       // 000000006000: B9800641 00000001
	s_load_b64 s[0:1], s[0:1], 0x58 nv                         // 000000006008: F4102000 F8000058
	s_and_b32 s13, s17, 0xffff                                 // 000000006010: 8B0DFF11 0000FFFF
	v_mov_b32_e32 v1, 0                                        // 000000006018: 7E020280
	v_mad_u32 v0, ttmp9, s13, v0                               // 00000000601C: D6350000 04001A75
	s_wait_kmcnt 0x0                                           // 000000006024: BFC70000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000006028: BF8700A1
	v_add_nc_u64_e32 v[0:1], s[0:1], v[0:1]                    // 00000000602C: 50000000
	s_mov_b32 s0, exec_lo                                      // 000000006030: BE80007E
	v_cmpx_gt_u64_e64 s[10:11], v[0:1]                         // 000000006034: D4DC007E 0202000A
	s_cbranch_execz 32                                         // 00000000603C: BFA50020 <__amd_rocclr_copyBufferAligned+0xc0>
	v_add_nc_u64_e32 v[2:3], s[6:7], v[0:1]                    // 000000006040: 50040006
	v_add_nc_u64_e32 v[0:1], s[8:9], v[0:1]                    // 000000006044: 50000008
	s_cmp_lg_u32 s12, 16                                       // 000000006048: BF07900C
	s_mov_b32 s0, -1                                           // 00000000604C: BE8000C1
	s_cbranch_scc0 14                                          // 000000006050: BFA1000E <__amd_rocclr_copyBufferAligned+0x8c>
	s_delay_alu instid0(VALU_DEP_2)                            // 000000006054: BF870002
	v_lshl_add_u64 v[4:5], v[2:3], 2, s[2:3]                   // 000000006058: D6520004 00090502
	s_mov_b32 s0, 0                                            // 000000006060: BE800080
	global_load_b32 v6, v[4:5], off                            // 000000006064: EE05007C 00000006 00000004
	s_wait_xcnt 0x0                                            // 000000006070: BFC50000
	v_lshl_add_u64 v[4:5], v[0:1], 2, s[4:5]                   // 000000006074: D6520004 00110500
	s_wait_loadcnt 0x0                                         // 00000000607C: BFC00000
	global_store_b32 v[4:5], v6, off                           // 000000006080: EE06807C 03000000 00000004
	s_and_not1_b32 vcc_lo, exec_lo, s0                         // 00000000608C: 916A007E
	s_cbranch_vccnz 11                                         // 000000006090: BFA4000B <__amd_rocclr_copyBufferAligned+0xc0>
	v_lshl_add_u64 v[2:3], v[2:3], 4, s[2:3]                   // 000000006094: D6520002 00090902
	v_lshl_add_u64 v[0:1], v[0:1], 4, s[4:5]                   // 00000000609C: D6520000 00110900
	global_load_b128 v[2:5], v[2:3], off                       // 0000000060A4: EE05C07C 00000002 00000002
	s_wait_loadcnt 0x0                                         // 0000000060B0: BFC00000
	global_store_b128 v[0:1], v[2:5], off                      // 0000000060B4: EE07407C 01000000 00000000
	s_endpgm                                                   // 0000000060C0: BFB00000
	s_nop 0                                                    // 0000000060C4: BF800000
	s_nop 0                                                    // 0000000060C8: BF800000
	s_nop 0                                                    // 0000000060CC: BF800000
	s_nop 0                                                    // 0000000060D0: BF800000
	s_nop 0                                                    // 0000000060D4: BF800000
	s_nop 0                                                    // 0000000060D8: BF800000
	s_nop 0                                                    // 0000000060DC: BF800000
	s_nop 0                                                    // 0000000060E0: BF800000
	s_nop 0                                                    // 0000000060E4: BF800000
	s_nop 0                                                    // 0000000060E8: BF800000
	s_nop 0                                                    // 0000000060EC: BF800000
	s_nop 0                                                    // 0000000060F0: BF800000
	s_nop 0                                                    // 0000000060F4: BF800000
	s_nop 0                                                    // 0000000060F8: BF800000
	s_nop 0                                                    // 0000000060FC: BF800000

0000000000006100 <__amd_rocclr_copyBufferRect>:
; __amd_rocclr_copyBufferRect():
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1       // 000000006100: B9800641 00000001
	s_clause 0x1                                               // 000000006108: BF850001
	s_load_b64 s[6:7], s[0:1], 0x8c nv                         // 00000000610C: F4102180 F800008C
	s_load_b128 s[36:39], s[0:1], 0xa8 nv                      // 000000006114: F4104900 F80000A8
	v_and_b32_e32 v1, 0x3ff, v0                                // 00000000611C: 360200FF 000003FF
	v_mov_b32_e32 v7, 0                                        // 000000006124: 7E0E0280
	s_and_b32 s16, ttmp7, 0xffff                               // 000000006128: 8B10FF73 0000FFFF
	s_wait_kmcnt 0x0                                           // 000000006130: BFC70000
	s_and_b32 s8, s6, 0xffff                                   // 000000006134: 8B08FF06 0000FFFF
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000613C: BF870009
	v_mad_u32 v6, ttmp9, s8, v1                                // 000000006140: D6350006 04041075
	v_bfe_u32 v1, v0, 10, 10                                   // 000000006148: D6100001 02291500
	s_clause 0x1                                               // 000000006150: BF850001
	s_load_b256 s[24:31], s[0:1], 0x60 nv                      // 000000006154: F4106600 F8000060
	s_load_b64 s[8:9], s[0:1], 0xb8 nv                         // 00000000615C: F4102200 F80000B8
	s_wait_xcnt 0x0                                            // 000000006164: BFC50000
	s_lshr_b32 s0, s6, 16                                      // 000000006168: 85009006
	v_bfe_u32 v0, v0, 20, 10                                   // 00000000616C: D6100000 02292900
	s_and_b32 s1, s7, 0xffff                                   // 000000006174: 8B01FF07 0000FFFF
	v_add_nc_u64_e32 v[2:3], s[36:37], v[6:7]                  // 00000000617C: 50040C24
	v_mad_u32 v6, s16, s0, v1                                  // 000000006180: D6350006 04040010
	s_lshr_b32 s0, ttmp7, 16                                   // 000000006188: 85009073
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)// 00000000618C: BF8701C1
	v_add_nc_u64_e32 v[4:5], s[38:39], v[6:7]                  // 000000006190: 50080C26
	v_mad_u32 v6, s0, s1, v0                                   // 000000006194: D6350006 04000200
	s_wait_kmcnt 0x0                                           // 00000000619C: BFC70000
	v_cmp_gt_u64_e32 vcc_lo, s[24:25], v[2:3]                  // 0000000061A0: 7CB80418
	v_cmp_gt_u64_e64 s0, s[26:27], v[4:5]                      // 0000000061A4: D45C0000 0202081A
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000061AC: BF8700A3
	v_add_nc_u64_e32 v[0:1], s[8:9], v[6:7]                    // 0000000061B0: 50000C08
	s_and_b32 s0, vcc_lo, s0                                   // 0000000061B4: 8B00006A
	v_cmp_gt_u64_e64 s1, s[28:29], v[0:1]                      // 0000000061B8: D45C0001 0202001C
	s_and_b32 s0, s0, s1                                       // 0000000061C0: 8B000100
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000061C4: BF870009
	s_and_saveexec_b32 s1, s0                                  // 0000000061C8: BE812000
	s_cbranch_execz 42                                         // 0000000061CC: BFA5002A <__amd_rocclr_copyBufferRect+0x178>
	s_add_nc_u64 s[0:1], s[2:3], s[14:15]                      // 0000000061D0: A9800E02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000061D4: BF8704A9
	v_add_nc_u64_e32 v[6:7], s[0:1], v[2:3]                    // 0000000061D8: 500C0400
	s_add_nc_u64 s[0:1], s[4:5], s[22:23]                      // 0000000061DC: A9801604
	v_add_nc_u64_e32 v[2:3], s[0:1], v[2:3]                    // 0000000061E0: 50040400
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000061E4: BF870112
	v_mad_nc_u64_u32 v[6:7], v4, s10, v[6:7]                   // 0000000061E8: D6FA0006 04181504
	v_mad_nc_u64_u32 v[2:3], v4, s18, v[2:3]                   // 0000000061F0: D6FA0002 04082504
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000061F8: BF870112
	v_mad_u32 v7, v5, s10, v7                                  // 0000000061FC: D6350007 041C1505
	v_mad_u32 v3, v5, s18, v3                                  // 000000006204: D6350003 040C2505
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000620C: BF870112
	v_mad_u32 v7, v4, s11, v7                                  // 000000006210: D6350007 041C1704
	v_mad_u32 v3, v4, s19, v3                                  // 000000006218: D6350003 040C2704
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000006220: BF870112
	v_mad_nc_u64_u32 v[6:7], v0, s12, v[6:7]                   // 000000006224: D6FA0006 04181900
	v_mad_nc_u64_u32 v[2:3], v0, s20, v[2:3]                   // 00000000622C: D6FA0002 04082900
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000006234: BF870112
	v_mad_u32 v7, v1, s12, v7                                  // 000000006238: D6350007 041C1901
	v_mad_u32 v1, v1, s20, v3                                  // 000000006240: D6350001 040C2901
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000006248: BF870112
	v_mad_u32 v7, v0, s13, v7                                  // 00000000624C: D6350007 041C1B00
	v_mad_u32 v3, v0, s21, v1                                  // 000000006254: D6350003 04042B00
	global_load_u8 v6, v[6:7], off                             // 00000000625C: EE04007C 00000006 00000006
	s_wait_loadcnt 0x0                                         // 000000006268: BFC00000
	global_store_b8 v[2:3], v6, off                            // 00000000626C: EE06007C 03000000 00000002
	s_endpgm                                                   // 000000006278: BFB00000
	s_nop 0                                                    // 00000000627C: BF800000
	s_nop 0                                                    // 000000006280: BF800000
	s_nop 0                                                    // 000000006284: BF800000
	s_nop 0                                                    // 000000006288: BF800000
	s_nop 0                                                    // 00000000628C: BF800000
	s_nop 0                                                    // 000000006290: BF800000
	s_nop 0                                                    // 000000006294: BF800000
	s_nop 0                                                    // 000000006298: BF800000
	s_nop 0                                                    // 00000000629C: BF800000
	s_nop 0                                                    // 0000000062A0: BF800000
	s_nop 0                                                    // 0000000062A4: BF800000
	s_nop 0                                                    // 0000000062A8: BF800000
	s_nop 0                                                    // 0000000062AC: BF800000
	s_nop 0                                                    // 0000000062B0: BF800000
	s_nop 0                                                    // 0000000062B4: BF800000
	s_nop 0                                                    // 0000000062B8: BF800000
	s_nop 0                                                    // 0000000062BC: BF800000
	s_nop 0                                                    // 0000000062C0: BF800000
	s_nop 0                                                    // 0000000062C4: BF800000
	s_nop 0                                                    // 0000000062C8: BF800000
	s_nop 0                                                    // 0000000062CC: BF800000
	s_nop 0                                                    // 0000000062D0: BF800000
	s_nop 0                                                    // 0000000062D4: BF800000
	s_nop 0                                                    // 0000000062D8: BF800000
	s_nop 0                                                    // 0000000062DC: BF800000
	s_nop 0                                                    // 0000000062E0: BF800000
	s_nop 0                                                    // 0000000062E4: BF800000
	s_nop 0                                                    // 0000000062E8: BF800000
	s_nop 0                                                    // 0000000062EC: BF800000
	s_nop 0                                                    // 0000000062F0: BF800000
	s_nop 0                                                    // 0000000062F4: BF800000
	s_nop 0                                                    // 0000000062F8: BF800000
	s_nop 0                                                    // 0000000062FC: BF800000

0000000000006300 <__amd_rocclr_copyBufferRectAligned>:
; __amd_rocclr_copyBufferRectAligned():
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1       // 000000006300: B9800641 00000001
	s_clause 0x1                                               // 000000006308: BF850001
	s_load_b64 s[6:7], s[0:1], 0x8c nv                         // 00000000630C: F4102180 F800008C
	s_load_b128 s[36:39], s[0:1], 0xa8 nv                      // 000000006314: F4104900 F80000A8
	v_and_b32_e32 v1, 0x3ff, v0                                // 00000000631C: 360200FF 000003FF
	v_mov_b32_e32 v7, 0                                        // 000000006324: 7E0E0280
	s_and_b32 s16, ttmp7, 0xffff                               // 000000006328: 8B10FF73 0000FFFF
	s_wait_kmcnt 0x0                                           // 000000006330: BFC70000
	s_and_b32 s8, s6, 0xffff                                   // 000000006334: 8B08FF06 0000FFFF
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000633C: BF870009
	v_mad_u32 v6, ttmp9, s8, v1                                // 000000006340: D6350006 04041075
	v_bfe_u32 v1, v0, 10, 10                                   // 000000006348: D6100001 02291500
	s_clause 0x1                                               // 000000006350: BF850001
	s_load_b256 s[24:31], s[0:1], 0x60 nv                      // 000000006354: F4106600 F8000060
	s_load_b64 s[8:9], s[0:1], 0xb8 nv                         // 00000000635C: F4102200 F80000B8
	s_wait_xcnt 0x0                                            // 000000006364: BFC50000
	s_lshr_b32 s0, s6, 16                                      // 000000006368: 85009006
	v_bfe_u32 v0, v0, 20, 10                                   // 00000000636C: D6100000 02292900
	s_and_b32 s1, s7, 0xffff                                   // 000000006374: 8B01FF07 0000FFFF
	v_add_nc_u64_e32 v[2:3], s[36:37], v[6:7]                  // 00000000637C: 50040C24
	v_mad_u32 v6, s16, s0, v1                                  // 000000006380: D6350006 04040010
	s_lshr_b32 s0, ttmp7, 16                                   // 000000006388: 85009073
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)// 00000000638C: BF8701C1
	v_add_nc_u64_e32 v[4:5], s[38:39], v[6:7]                  // 000000006390: 50080C26
	v_mad_u32 v6, s0, s1, v0                                   // 000000006394: D6350006 04000200
	s_wait_kmcnt 0x0                                           // 00000000639C: BFC70000
	v_cmp_gt_u64_e32 vcc_lo, s[24:25], v[2:3]                  // 0000000063A0: 7CB80418
	v_cmp_gt_u64_e64 s0, s[26:27], v[4:5]                      // 0000000063A4: D45C0000 0202081A
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000063AC: BF8700A3
	v_add_nc_u64_e32 v[0:1], s[8:9], v[6:7]                    // 0000000063B0: 50000C08
	s_and_b32 s0, vcc_lo, s0                                   // 0000000063B4: 8B00006A
	v_cmp_gt_u64_e64 s1, s[28:29], v[0:1]                      // 0000000063B8: D45C0001 0202001C
	s_and_b32 s0, s0, s1                                       // 0000000063C0: 8B000100
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000063C4: BF870009
	s_and_saveexec_b32 s1, s0                                  // 0000000063C8: BE812000
	s_cbranch_execz 63                                         // 0000000063CC: BFA5003F <__amd_rocclr_copyBufferRectAligned+0x1cc>
	v_add_nc_u64_e32 v[6:7], s[14:15], v[2:3]                  // 0000000063D0: 500C040E
	v_add_nc_u64_e32 v[2:3], s[22:23], v[2:3]                  // 0000000063D4: 50040416
	s_cmp_lg_u64 s[30:31], 16                                  // 0000000063D8: BF11901E
	s_mov_b32 s0, -1                                           // 0000000063DC: BE8000C1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000063E0: BF870112
	v_mad_nc_u64_u32 v[6:7], v4, s10, v[6:7]                   // 0000000063E4: D6FA0006 04181504
	v_mad_nc_u64_u32 v[8:9], v4, s18, v[2:3]                   // 0000000063EC: D6FA0008 04082504
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000063F4: BF870112
	v_mad_u32 v2, v5, s10, v7                                  // 0000000063F8: D6350002 041C1505
	v_mad_u32 v3, v5, s18, v9                                  // 000000006400: D6350003 04242505
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000006408: BF870112
	v_mad_u32 v7, v4, s11, v2                                  // 00000000640C: D6350007 04081704
	v_mad_u32 v9, v4, s19, v3                                  // 000000006414: D6350009 040C2704
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000641C: BF870112
	v_mad_nc_u64_u32 v[2:3], v0, s12, v[6:7]                   // 000000006420: D6FA0002 04181900
	v_mad_nc_u64_u32 v[4:5], v0, s20, v[8:9]                   // 000000006428: D6FA0004 04202900
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000006430: BF870112
	v_mad_u32 v3, v1, s12, v3                                  // 000000006434: D6350003 040C1901
	v_mad_u32 v1, v1, s20, v5                                  // 00000000643C: D6350001 04142901
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000006444: BF870112
	v_mad_u32 v3, v0, s13, v3                                  // 000000006448: D6350003 040C1B00
	v_mad_u32 v5, v0, s21, v1                                  // 000000006450: D6350005 04042B00
	s_cbranch_scc0 14                                          // 000000006458: BFA1000E <__amd_rocclr_copyBufferRectAligned+0x194>
	s_delay_alu instid0(VALU_DEP_2)                            // 00000000645C: BF870002
	v_lshl_add_u64 v[0:1], v[2:3], 2, s[2:3]                   // 000000006460: D6520000 00090502
	s_mov_b32 s0, 0                                            // 000000006468: BE800080
	global_load_b32 v6, v[0:1], off                            // 00000000646C: EE05007C 00000006 00000000
	s_wait_xcnt 0x0                                            // 000000006478: BFC50000
	v_lshl_add_u64 v[0:1], v[4:5], 2, s[4:5]                   // 00000000647C: D6520000 00110504
	s_wait_loadcnt 0x0                                         // 000000006484: BFC00000
	global_store_b32 v[0:1], v6, off                           // 000000006488: EE06807C 03000000 00000000
	s_and_not1_b32 vcc_lo, exec_lo, s0                         // 000000006494: 916A007E
	s_cbranch_vccnz 12                                         // 000000006498: BFA4000C <__amd_rocclr_copyBufferRectAligned+0x1cc>
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000649C: BF870112
	v_lshl_add_u64 v[0:1], v[2:3], 4, s[2:3]                   // 0000000064A0: D6520000 00090902
	v_lshl_add_u64 v[4:5], v[4:5], 4, s[4:5]                   // 0000000064A8: D6520004 00110904
	global_load_b128 v[0:3], v[0:1], off                       // 0000000064B0: EE05C07C 00000000 00000000
	s_wait_loadcnt 0x0                                         // 0000000064BC: BFC00000
	global_store_b128 v[4:5], v[0:3], off                      // 0000000064C0: EE07407C 00000000 00000004
	s_endpgm                                                   // 0000000064CC: BFB00000
	s_nop 0                                                    // 0000000064D0: BF800000
	s_nop 0                                                    // 0000000064D4: BF800000
	s_nop 0                                                    // 0000000064D8: BF800000
	s_nop 0                                                    // 0000000064DC: BF800000
	s_nop 0                                                    // 0000000064E0: BF800000
	s_nop 0                                                    // 0000000064E4: BF800000
	s_nop 0                                                    // 0000000064E8: BF800000
	s_nop 0                                                    // 0000000064EC: BF800000
	s_nop 0                                                    // 0000000064F0: BF800000
	s_nop 0                                                    // 0000000064F4: BF800000
	s_nop 0                                                    // 0000000064F8: BF800000
	s_nop 0                                                    // 0000000064FC: BF800000

0000000000006500 <__amd_rocclr_batchMemOp>:
; __amd_rocclr_batchMemOp():
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1       // 000000006500: B9800641 00000001
	s_add_nc_u64 s[0:1], s[0:1], 16                            // 000000006508: A9809000
	s_and_b32 s4, 0xffff, s9                                   // 00000000650C: 8B0409FF 0000FFFF
	s_load_b64 s[0:1], s[0:1], 0x28 nv                         // 000000006514: F4102000 F8000028
	v_mad_u32 v0, ttmp9, s4, v0                                // 00000000651C: D6350000 04000875
	s_wait_kmcnt 0x0                                           // 000000006524: BFC70000
	s_mul_u64 s[0:1], s[0:1], 48                               // 000000006528: AA80B000
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000652C: BF870009
	s_add_nc_u64 s[0:1], s[2:3], s[0:1]                        // 000000006530: A9800002
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)    // 000000006534: BF870481
	v_mad_nc_u64_u32 v[4:5], v0, 48, s[0:1]                    // 000000006538: D6FA0004 00016100
	s_mov_b32 s0, exec_lo                                      // 000000006540: BE80007E
	global_load_b32 v0, v[4:5], off                            // 000000006544: EE05007C 00000000 00000004
	s_wait_loadcnt 0x0                                         // 000000006550: BFC00000
	v_cmpx_lt_i32_e32 3, v0                                    // 000000006554: 7D820083
	s_xor_b32 s0, exec_lo, s0                                  // 000000006558: 8D00007E
	s_cbranch_execz 156                                        // 00000000655C: BFA5009C <__amd_rocclr_batchMemOp+0x2d0>
	s_mov_b32 s1, exec_lo                                      // 000000006560: BE81007E
	v_cmpx_lt_i32_e32 4, v0                                    // 000000006564: 7D820084
	s_xor_b32 s1, exec_lo, s1                                  // 000000006568: 8D01017E
	s_cbranch_execz 12                                         // 00000000656C: BFA5000C <__amd_rocclr_batchMemOp+0xa0>
	s_mov_b32 s2, exec_lo                                      // 000000006570: BE82007E
	v_cmpx_eq_u32_e32 5, v0                                    // 000000006574: 7D940085
	s_cbranch_execz 7                                          // 000000006578: BFA50007 <__amd_rocclr_batchMemOp+0x98>
	global_load_b128 v[0:3], v[4:5], off offset:8              // 00000000657C: EE05C07C 00000000 00000804
	s_wait_loadcnt 0x0                                         // 000000006588: BFC00000
	flat_store_b64 v[0:1], v[2:3] scope:SCOPE_SYS              // 00000000658C: EC06C07C 010C0000 00000000
	s_wait_xcnt 0x0                                            // 000000006598: BFC50000
	s_or_b32 exec_lo, exec_lo, s2                              // 00000000659C: 8C7E027E
	s_and_not1_saveexec_b32 s1, s1                             // 0000000065A0: BE813001
	s_cbranch_execz 136                                        // 0000000065A4: BFA50088 <__amd_rocclr_batchMemOp+0x2c8>
	s_clause 0x1                                               // 0000000065A8: BF850001
	global_load_b32 v6, v[4:5], off offset:24                  // 0000000065AC: EE05007C 00000006 00001804
	global_load_b128 v[0:3], v[4:5], off offset:8              // 0000000065B8: EE05C07C 00000000 00000804
	s_mov_b32 s2, exec_lo                                      // 0000000065C4: BE82007E
	s_wait_loadcnt 0x1                                         // 0000000065C8: BFC00001
	s_wait_xcnt 0x0                                            // 0000000065CC: BFC50000
	v_cmpx_lt_i32_e32 1, v6                                    // 0000000065D0: 7D820C81
	s_xor_b32 s2, exec_lo, s2                                  // 0000000065D4: 8D02027E
	s_cbranch_execz 65                                         // 0000000065D8: BFA50041 <__amd_rocclr_batchMemOp+0x1e0>
	s_mov_b32 s3, exec_lo                                      // 0000000065DC: BE83007E
	v_cmpx_lt_i32_e32 2, v6                                    // 0000000065E0: 7D820C82
	s_xor_b32 s3, exec_lo, s3                                  // 0000000065E4: 8D03037E
	s_cbranch_execz 30                                         // 0000000065E8: BFA5001E <__amd_rocclr_batchMemOp+0x164>
	s_mov_b32 s4, exec_lo                                      // 0000000065EC: BE84007E
	v_cmpx_eq_u32_e32 3, v6                                    // 0000000065F0: 7D940C83
	s_cbranch_execz 26                                         // 0000000065F4: BFA5001A <__amd_rocclr_batchMemOp+0x160>
	s_wait_loadcnt 0x0                                         // 0000000065F8: BFC00000
	flat_load_b64 v[4:5], v[0:1] scope:SCOPE_SYS               // 0000000065FC: EC05407C 000C0004 00000000
	s_wait_loadcnt_dscnt 0x0                                   // 000000006608: BFC80000
	v_or_b32_e32 v5, v5, v3                                    // 00000000660C: 380A0705
	v_or_b32_e32 v4, v4, v2                                    // 000000006610: 38080504
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006614: BF870001
	v_cmp_eq_u64_e32 vcc_lo, -1, v[4:5]                        // 000000006618: 7CB408C1
	s_wait_xcnt 0x0                                            // 00000000661C: BFC50000
	s_and_b32 exec_lo, exec_lo, vcc_lo                         // 000000006620: 8B7E6A7E
	s_cbranch_execz 14                                         // 000000006624: BFA5000E <__amd_rocclr_batchMemOp+0x160>
	s_mov_b32 s5, 0                                            // 000000006628: BE850080
	s_sleep 1                                                  // 00000000662C: BF830001
	flat_load_b64 v[4:5], v[0:1] scope:SCOPE_SYS               // 000000006630: EC05407C 000C0004 00000000
	s_wait_loadcnt_dscnt 0x0                                   // 00000000663C: BFC80000
	v_or_b32_e32 v5, v5, v3                                    // 000000006640: 380A0705
	v_or_b32_e32 v4, v4, v2                                    // 000000006644: 38080504
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006648: BF870001
	v_cmp_ne_u64_e32 vcc_lo, -1, v[4:5]                        // 00000000664C: 7CBA08C1
	s_or_b32 s5, vcc_lo, s5                                    // 000000006650: 8C05056A
	s_wait_xcnt 0x0                                            // 000000006654: BFC50000
	s_and_not1_b32 exec_lo, exec_lo, s5                        // 000000006658: 917E057E
	s_cbranch_execnz 65523                                     // 00000000665C: BFA6FFF3 <__amd_rocclr_batchMemOp+0x12c>
	s_or_b32 exec_lo, exec_lo, s4                              // 000000006660: 8C7E047E
	s_and_not1_saveexec_b32 s3, s3                             // 000000006664: BE833003
	s_cbranch_execz 27                                         // 000000006668: BFA5001B <__amd_rocclr_batchMemOp+0x1d8>
	s_wait_loadcnt 0x0                                         // 00000000666C: BFC00000
	flat_load_b64 v[4:5], v[0:1] scope:SCOPE_SYS               // 000000006670: EC05407C 000C0004 00000000
	s_mov_b32 s4, exec_lo                                      // 00000000667C: BE84007E
	s_wait_loadcnt_dscnt 0x0                                   // 000000006680: BFC80000
	v_and_b32_e32 v5, v5, v3                                   // 000000006684: 360A0705
	v_and_b32_e32 v4, v4, v2                                   // 000000006688: 36080504
	s_wait_xcnt 0x0                                            // 00000000668C: BFC50000
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006690: BF870001
	v_cmpx_eq_u64_e32 0, v[4:5]                                // 000000006694: 7DB40880
	s_cbranch_execz 14                                         // 000000006698: BFA5000E <__amd_rocclr_batchMemOp+0x1d4>
	s_mov_b32 s5, 0                                            // 00000000669C: BE850080
	s_sleep 1                                                  // 0000000066A0: BF830001
	flat_load_b64 v[4:5], v[0:1] scope:SCOPE_SYS               // 0000000066A4: EC05407C 000C0004 00000000
	s_wait_loadcnt_dscnt 0x0                                   // 0000000066B0: BFC80000
	v_and_b32_e32 v5, v5, v3                                   // 0000000066B4: 360A0705
	v_and_b32_e32 v4, v4, v2                                   // 0000000066B8: 36080504
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000066BC: BF870001
	v_cmp_ne_u64_e32 vcc_lo, 0, v[4:5]                         // 0000000066C0: 7CBA0880
	s_or_b32 s5, vcc_lo, s5                                    // 0000000066C4: 8C05056A
	s_wait_xcnt 0x0                                            // 0000000066C8: BFC50000
	s_and_not1_b32 exec_lo, exec_lo, s5                        // 0000000066CC: 917E057E
	s_cbranch_execnz 65523                                     // 0000000066D0: BFA6FFF3 <__amd_rocclr_batchMemOp+0x1a0>
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000066D4: 8C7E047E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000066D8: BF870009
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000066DC: 8C7E037E
	s_and_not1_saveexec_b32 s2, s2                             // 0000000066E0: BE823002
	s_cbranch_execz 54                                         // 0000000066E4: BFA50036 <__amd_rocclr_batchMemOp+0x2c0>
	s_mov_b32 s3, exec_lo                                      // 0000000066E8: BE83007E
	v_cmpx_lt_i32_e32 0, v6                                    // 0000000066EC: 7D820C80
	s_xor_b32 s3, exec_lo, s3                                  // 0000000066F0: 8D03037E
	s_cbranch_execz 22                                         // 0000000066F4: BFA50016 <__amd_rocclr_batchMemOp+0x250>
	s_wait_loadcnt 0x0                                         // 0000000066F8: BFC00000
	flat_load_b64 v[4:5], v[0:1] scope:SCOPE_SYS               // 0000000066FC: EC05407C 000C0004 00000000
	s_mov_b32 s4, exec_lo                                      // 000000006708: BE84007E
	s_wait_loadcnt_dscnt 0x0                                   // 00000000670C: BFC80000
	s_wait_xcnt 0x0                                            // 000000006710: BFC50000
	v_cmpx_ne_u64_e64 v[4:5], v[2:3]                           // 000000006714: D4DD007E 02020504
	s_cbranch_execz 11                                         // 00000000671C: BFA5000B <__amd_rocclr_batchMemOp+0x24c>
	s_mov_b32 s5, 0                                            // 000000006720: BE850080
	s_sleep 1                                                  // 000000006724: BF830001
	flat_load_b64 v[4:5], v[0:1] scope:SCOPE_SYS               // 000000006728: EC05407C 000C0004 00000000
	s_wait_loadcnt_dscnt 0x0                                   // 000000006734: BFC80000
	v_cmp_eq_u64_e32 vcc_lo, v[4:5], v[2:3]                    // 000000006738: 7CB40504
	s_or_b32 s5, vcc_lo, s5                                    // 00000000673C: 8C05056A
	s_wait_xcnt 0x0                                            // 000000006740: BFC50000
	s_and_not1_b32 exec_lo, exec_lo, s5                        // 000000006744: 917E057E
	s_cbranch_execnz 65526                                     // 000000006748: BFA6FFF6 <__amd_rocclr_batchMemOp+0x224>
	s_or_b32 exec_lo, exec_lo, s4                              // 00000000674C: 8C7E047E
	s_and_not1_saveexec_b32 s3, s3                             // 000000006750: BE833003
	s_cbranch_execz 24                                         // 000000006754: BFA50018 <__amd_rocclr_batchMemOp+0x2b8>
	s_mov_b32 s4, exec_lo                                      // 000000006758: BE84007E
	v_cmpx_eq_u32_e32 0, v6                                    // 00000000675C: 7D940C80
	s_cbranch_execz 20                                         // 000000006760: BFA50014 <__amd_rocclr_batchMemOp+0x2b4>
	s_wait_loadcnt 0x0                                         // 000000006764: BFC00000
	flat_load_b64 v[4:5], v[0:1] scope:SCOPE_SYS               // 000000006768: EC05407C 000C0004 00000000
	s_wait_loadcnt_dscnt 0x0                                   // 000000006774: BFC80000
	v_cmp_lt_u64_e32 vcc_lo, v[4:5], v[2:3]                    // 000000006778: 7CB20504
	s_wait_xcnt 0x0                                            // 00000000677C: BFC50000
	s_and_b32 exec_lo, exec_lo, vcc_lo                         // 000000006780: 8B7E6A7E
	s_cbranch_execz 11                                         // 000000006784: BFA5000B <__amd_rocclr_batchMemOp+0x2b4>
	s_mov_b32 s5, 0                                            // 000000006788: BE850080
	s_sleep 1                                                  // 00000000678C: BF830001
	flat_load_b64 v[4:5], v[0:1] scope:SCOPE_SYS               // 000000006790: EC05407C 000C0004 00000000
	s_wait_loadcnt_dscnt 0x0                                   // 00000000679C: BFC80000
	v_cmp_ge_u64_e32 vcc_lo, v[4:5], v[2:3]                    // 0000000067A0: 7CBC0504
	s_or_b32 s5, vcc_lo, s5                                    // 0000000067A4: 8C05056A
	s_wait_xcnt 0x0                                            // 0000000067A8: BFC50000
	s_and_not1_b32 exec_lo, exec_lo, s5                        // 0000000067AC: 917E057E
	s_cbranch_execnz 65526                                     // 0000000067B0: BFA6FFF6 <__amd_rocclr_batchMemOp+0x28c>
	s_or_b32 exec_lo, exec_lo, s4                              // 0000000067B4: 8C7E047E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000067B8: BF870009
	s_or_b32 exec_lo, exec_lo, s3                              // 0000000067BC: 8C7E037E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000067C0: BF870009
	s_or_b32 exec_lo, exec_lo, s2                              // 0000000067C4: 8C7E027E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000067C8: BF870009
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000067CC: 8C7E017E
	s_and_not1_saveexec_b32 s0, s0                             // 0000000067D0: BE803000
	s_cbranch_execz 287                                        // 0000000067D4: BFA5011F <__amd_rocclr_batchMemOp+0x754>
	s_mov_b32 s0, exec_lo                                      // 0000000067D8: BE80007E
	s_wait_loadcnt 0x0                                         // 0000000067DC: BFC00000
	v_cmpx_lt_i32_e32 1, v0                                    // 0000000067E0: 7D820081
	s_xor_b32 s0, exec_lo, s0                                  // 0000000067E4: 8D00007E
	s_cbranch_execz 25                                         // 0000000067E8: BFA50019 <__amd_rocclr_batchMemOp+0x350>
	s_mov_b32 s1, exec_lo                                      // 0000000067EC: BE81007E
	v_cmpx_eq_u32_e32 2, v0                                    // 0000000067F0: 7D940082
	s_cbranch_execz 20                                         // 0000000067F4: BFA50014 <__amd_rocclr_batchMemOp+0x348>
	global_load_b96 v[0:2], v[4:5], off offset:8               // 0000000067F8: EE05807C 00000000 00000804
	s_mov_b32 s2, exec_lo                                      // 000000006804: BE82007E
	s_wait_loadcnt 0x0                                         // 000000006808: BFC00000
	s_wait_xcnt 0x0                                            // 00000000680C: BFC50000
	v_cmpx_ne_u64_e32 0, v[0:1]                                // 000000006810: 7DBA0080
	s_xor_b32 s2, exec_lo, s2                                  // 000000006814: 8D02027E
	s_cbranch_execz 3                                          // 000000006818: BFA50003 <__amd_rocclr_batchMemOp+0x328>
	flat_store_b32 v[0:1], v2 scope:SCOPE_SYS                  // 00000000681C: EC06807C 010C0000 00000000
	s_wait_xcnt 0x0                                            // 000000006828: BFC50000
	s_and_not1_saveexec_b32 s2, s2                             // 00000000682C: BE823002
	s_cbranch_execz 5                                          // 000000006830: BFA50005 <__amd_rocclr_batchMemOp+0x348>
	v_mov_b64_e32 v[0:1], 0                                    // 000000006834: 7E003A80
	v_mov_b32_e32 v3, 0                                        // 000000006838: 7E060280
	global_store_b64 v[0:1], v[2:3], off scope:SCOPE_SYS       // 00000000683C: EE06C07C 010C0000 00000000
	s_wait_xcnt 0x0                                            // 000000006848: BFC50000
	s_or_b32 exec_lo, exec_lo, s1                              // 00000000684C: 8C7E017E
	s_and_not1_saveexec_b32 s0, s0                             // 000000006850: BE803000
	s_cbranch_execz 255                                        // 000000006854: BFA500FF <__amd_rocclr_batchMemOp+0x754>
	s_mov_b32 s0, exec_lo                                      // 000000006858: BE80007E
	v_cmpx_eq_u32_e32 1, v0                                    // 00000000685C: 7D940081
	s_cbranch_execz 252                                        // 000000006860: BFA500FC <__amd_rocclr_batchMemOp+0x754>
	s_clause 0x1                                               // 000000006864: BF850001
	global_load_b32 v6, v[4:5], off offset:24                  // 000000006868: EE05007C 00000006 00001804
	global_load_b96 v[0:2], v[4:5], off offset:8               // 000000006874: EE05807C 00000000 00000804
	v_mov_b32_e32 v3, 0                                        // 000000006880: 7E060280
	s_mov_b32 s0, exec_lo                                      // 000000006884: BE80007E
	s_wait_loadcnt 0x1                                         // 000000006888: BFC00001
	s_wait_xcnt 0x0                                            // 00000000688C: BFC50000
	v_cmpx_lt_i32_e32 1, v6                                    // 000000006890: 7D820C81
	s_xor_b32 s0, exec_lo, s0                                  // 000000006894: 8D00007E
	s_cbranch_execz 127                                        // 000000006898: BFA5007F <__amd_rocclr_batchMemOp+0x598>
	s_mov_b32 s1, exec_lo                                      // 00000000689C: BE81007E
	v_cmpx_lt_i32_e32 2, v6                                    // 0000000068A0: 7D820C82
	s_xor_b32 s1, exec_lo, s1                                  // 0000000068A4: 8D01017E
	s_cbranch_execz 59                                         // 0000000068A8: BFA5003B <__amd_rocclr_batchMemOp+0x498>
	s_mov_b32 s2, exec_lo                                      // 0000000068AC: BE82007E
	v_cmpx_eq_u32_e32 3, v6                                    // 0000000068B0: 7D940C83
	s_cbranch_execz 55                                         // 0000000068B4: BFA50037 <__amd_rocclr_batchMemOp+0x494>
	s_mov_b32 s3, exec_lo                                      // 0000000068B8: BE83007E
	s_wait_loadcnt 0x0                                         // 0000000068BC: BFC00000
	v_cmpx_ne_u64_e32 0, v[0:1]                                // 0000000068C0: 7DBA0080
	s_xor_b32 s3, exec_lo, s3                                  // 0000000068C4: 8D03037E
	s_cbranch_execz 24                                         // 0000000068C8: BFA50018 <__amd_rocclr_batchMemOp+0x42c>
	flat_load_b32 v3, v[0:1] scope:SCOPE_SYS                   // 0000000068CC: EC05007C 000C0003 00000000
	s_mov_b32 s4, exec_lo                                      // 0000000068D8: BE84007E
	s_wait_loadcnt_dscnt 0x0                                   // 0000000068DC: BFC80000
	v_or_b32_e32 v3, v3, v2                                    // 0000000068E0: 38060503
	s_wait_xcnt 0x0                                            // 0000000068E4: BFC50000
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000068E8: BF870001
	v_cmpx_eq_u32_e32 -1, v3                                   // 0000000068EC: 7D9406C1
	s_cbranch_execz 13                                         // 0000000068F0: BFA5000D <__amd_rocclr_batchMemOp+0x428>
	s_mov_b32 s5, 0                                            // 0000000068F4: BE850080
	s_sleep 1                                                  // 0000000068F8: BF830001
	flat_load_b32 v3, v[0:1] scope:SCOPE_SYS                   // 0000000068FC: EC05007C 000C0003 00000000
	s_wait_loadcnt_dscnt 0x0                                   // 000000006908: BFC80000
	v_or_b32_e32 v3, v3, v2                                    // 00000000690C: 38060503
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006910: BF870001
	v_cmp_ne_u32_e32 vcc_lo, -1, v3                            // 000000006914: 7C9A06C1
	s_or_b32 s5, vcc_lo, s5                                    // 000000006918: 8C05056A
	s_wait_xcnt 0x0                                            // 00000000691C: BFC50000
	s_and_not1_b32 exec_lo, exec_lo, s5                        // 000000006920: 917E057E
	s_cbranch_execnz 65524                                     // 000000006924: BFA6FFF4 <__amd_rocclr_batchMemOp+0x3f8>
	s_or_b32 exec_lo, exec_lo, s4                              // 000000006928: 8C7E047E
	s_and_not1_saveexec_b32 s3, s3                             // 00000000692C: BE833003
	s_cbranch_execz 24                                         // 000000006930: BFA50018 <__amd_rocclr_batchMemOp+0x494>
	v_mov_b64_e32 v[0:1], 0                                    // 000000006934: 7E003A80
	global_load_b64 v[4:5], v[0:1], off scope:SCOPE_SYS        // 000000006938: EE05407C 000C0004 00000000
	s_wait_loadcnt 0x0                                         // 000000006944: BFC00000
	v_or_b32_e32 v4, v4, v2                                    // 000000006948: 38080504
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000694C: BF870001
	v_cmp_eq_u64_e32 vcc_lo, -1, v[4:5]                        // 000000006950: 7CB408C1
	s_wait_xcnt 0x0                                            // 000000006954: BFC50000
	s_and_b32 exec_lo, exec_lo, vcc_lo                         // 000000006958: 8B7E6A7E
	s_cbranch_execz 13                                         // 00000000695C: BFA5000D <__amd_rocclr_batchMemOp+0x494>
	s_mov_b32 s3, 0                                            // 000000006960: BE830080
	s_sleep 1                                                  // 000000006964: BF830001
	global_load_b64 v[4:5], v[0:1], off scope:SCOPE_SYS        // 000000006968: EE05407C 000C0004 00000000
	s_wait_loadcnt 0x0                                         // 000000006974: BFC00000
	v_or_b32_e32 v4, v4, v2                                    // 000000006978: 38080504
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000697C: BF870001
	v_cmp_ne_u64_e32 vcc_lo, -1, v[4:5]                        // 000000006980: 7CBA08C1
	s_or_b32 s3, vcc_lo, s3                                    // 000000006984: 8C03036A
	s_wait_xcnt 0x0                                            // 000000006988: BFC50000
	s_and_not1_b32 exec_lo, exec_lo, s3                        // 00000000698C: 917E037E
	s_cbranch_execnz 65524                                     // 000000006990: BFA6FFF4 <__amd_rocclr_batchMemOp+0x464>
	s_or_b32 exec_lo, exec_lo, s2                              // 000000006994: 8C7E027E
	s_and_not1_saveexec_b32 s1, s1                             // 000000006998: BE813001
	s_cbranch_execz 60                                         // 00000000699C: BFA5003C <__amd_rocclr_batchMemOp+0x590>
	s_mov_b32 s2, exec_lo                                      // 0000000069A0: BE82007E
	s_wait_loadcnt 0x0                                         // 0000000069A4: BFC00000
	v_cmpx_ne_u64_e32 0, v[0:1]                                // 0000000069A8: 7DBA0080
	s_xor_b32 s2, exec_lo, s2                                  // 0000000069AC: 8D02027E
	s_cbranch_execz 24                                         // 0000000069B0: BFA50018 <__amd_rocclr_batchMemOp+0x514>
	flat_load_b32 v3, v[0:1] scope:SCOPE_SYS                   // 0000000069B4: EC05007C 000C0003 00000000
	s_mov_b32 s4, 0                                            // 0000000069C0: BE840080
	s_mov_b32 s3, exec_lo                                      // 0000000069C4: BE83007E
	s_wait_loadcnt_dscnt 0x0                                   // 0000000069C8: BFC80000
	v_and_b32_e32 v3, v3, v2                                   // 0000000069CC: 36060503
	s_wait_xcnt 0x0                                            // 0000000069D0: BFC50000
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000069D4: BF870001
	v_cmpx_eq_u32_e32 0, v3                                    // 0000000069D8: 7D940680
	s_cbranch_execz 12                                         // 0000000069DC: BFA5000C <__amd_rocclr_batchMemOp+0x510>
	s_sleep 1                                                  // 0000000069E0: BF830001
	flat_load_b32 v3, v[0:1] scope:SCOPE_SYS                   // 0000000069E4: EC05007C 000C0003 00000000
	s_wait_loadcnt_dscnt 0x0                                   // 0000000069F0: BFC80000
	v_and_b32_e32 v3, v3, v2                                   // 0000000069F4: 36060503
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000069F8: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v3                             // 0000000069FC: 7C9A0680
	s_or_b32 s4, vcc_lo, s4                                    // 000000006A00: 8C04046A
	s_wait_xcnt 0x0                                            // 000000006A04: BFC50000
	s_and_not1_b32 exec_lo, exec_lo, s4                        // 000000006A08: 917E047E
	s_cbranch_execnz 65524                                     // 000000006A0C: BFA6FFF4 <__amd_rocclr_batchMemOp+0x4e0>
	s_or_b32 exec_lo, exec_lo, s3                              // 000000006A10: 8C7E037E
	s_and_not1_saveexec_b32 s2, s2                             // 000000006A14: BE823002
	s_cbranch_execz 27                                         // 000000006A18: BFA5001B <__amd_rocclr_batchMemOp+0x588>
	v_mov_b64_e32 v[0:1], 0                                    // 000000006A1C: 7E003A80
	s_mov_b32 s3, exec_lo                                      // 000000006A20: BE83007E
	global_load_b64 v[4:5], v[0:1], off scope:SCOPE_SYS        // 000000006A24: EE05407C 000C0004 00000000
	s_wait_loadcnt 0x0                                         // 000000006A30: BFC00000
	v_dual_mov_b32 v5, 0 :: v_dual_bitop2_b32 v4, v4, v2 bitop3:0x40// 000000006A34: CF212080 00000104 04400205
	s_wait_xcnt 0x0                                            // 000000006A40: BFC50000
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006A44: BF870001
	v_cmpx_eq_u64_e32 0, v[4:5]                                // 000000006A48: 7DB40880
	s_cbranch_execz 13                                         // 000000006A4C: BFA5000D <__amd_rocclr_batchMemOp+0x584>
	s_mov_b32 s4, 0                                            // 000000006A50: BE840080
	s_sleep 1                                                  // 000000006A54: BF830001
	global_load_b64 v[6:7], v[0:1], off scope:SCOPE_SYS        // 000000006A58: EE05407C 000C0006 00000000
	s_wait_loadcnt 0x0                                         // 000000006A64: BFC00000
	v_and_b32_e32 v4, v6, v2                                   // 000000006A68: 36080506
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006A6C: BF870001
	v_cmp_ne_u64_e32 vcc_lo, 0, v[4:5]                         // 000000006A70: 7CBA0880
	s_or_b32 s4, vcc_lo, s4                                    // 000000006A74: 8C04046A
	s_wait_xcnt 0x0                                            // 000000006A78: BFC50000
	s_and_not1_b32 exec_lo, exec_lo, s4                        // 000000006A7C: 917E047E
	s_cbranch_execnz 65524                                     // 000000006A80: BFA6FFF4 <__amd_rocclr_batchMemOp+0x554>
	s_or_b32 exec_lo, exec_lo, s3                              // 000000006A84: 8C7E037E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000006A88: BF870009
	s_or_b32 exec_lo, exec_lo, s2                              // 000000006A8C: 8C7E027E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000006A90: BF870009
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006A94: 8C7E017E
	s_and_not1_saveexec_b32 s0, s0                             // 000000006A98: BE803000
	s_cbranch_execz 109                                        // 000000006A9C: BFA5006D <__amd_rocclr_batchMemOp+0x754>
	s_mov_b32 s0, exec_lo                                      // 000000006AA0: BE80007E
	v_cmpx_lt_i32_e32 0, v6                                    // 000000006AA4: 7D820C80
	s_xor_b32 s0, exec_lo, s0                                  // 000000006AA8: 8D00007E
	s_cbranch_execz 52                                         // 000000006AAC: BFA50034 <__amd_rocclr_batchMemOp+0x680>
	s_mov_b32 s1, exec_lo                                      // 000000006AB0: BE81007E
	s_wait_loadcnt 0x0                                         // 000000006AB4: BFC00000
	v_cmpx_ne_u64_e32 0, v[0:1]                                // 000000006AB8: 7DBA0080
	s_xor_b32 s1, exec_lo, s1                                  // 000000006ABC: 8D01017E
	s_cbranch_execz 21                                         // 000000006AC0: BFA50015 <__amd_rocclr_batchMemOp+0x618>
	flat_load_b32 v3, v[0:1] scope:SCOPE_SYS                   // 000000006AC4: EC05007C 000C0003 00000000
	s_mov_b32 s2, exec_lo                                      // 000000006AD0: BE82007E
	s_wait_loadcnt_dscnt 0x0                                   // 000000006AD4: BFC80000
	s_wait_xcnt 0x0                                            // 000000006AD8: BFC50000
	v_cmpx_ne_u32_e64 v3, v2                                   // 000000006ADC: D4CD007E 02020503
	s_cbranch_execz 11                                         // 000000006AE4: BFA5000B <__amd_rocclr_batchMemOp+0x614>
	s_mov_b32 s3, 0                                            // 000000006AE8: BE830080
	s_sleep 1                                                  // 000000006AEC: BF830001
	flat_load_b32 v3, v[0:1] scope:SCOPE_SYS                   // 000000006AF0: EC05007C 000C0003 00000000
	s_wait_loadcnt_dscnt 0x0                                   // 000000006AFC: BFC80000
	v_cmp_eq_u32_e32 vcc_lo, v3, v2                            // 000000006B00: 7C940503
	s_or_b32 s3, vcc_lo, s3                                    // 000000006B04: 8C03036A
	s_wait_xcnt 0x0                                            // 000000006B08: BFC50000
	s_and_not1_b32 exec_lo, exec_lo, s3                        // 000000006B0C: 917E037E
	s_cbranch_execnz 65526                                     // 000000006B10: BFA6FFF6 <__amd_rocclr_batchMemOp+0x5ec>
	s_or_b32 exec_lo, exec_lo, s2                              // 000000006B14: 8C7E027E
	s_and_not1_saveexec_b32 s1, s1                             // 000000006B18: BE813001
	s_cbranch_execz 22                                         // 000000006B1C: BFA50016 <__amd_rocclr_batchMemOp+0x678>
	v_mov_b64_e32 v[0:1], 0                                    // 000000006B20: 7E003A80
	s_mov_b32 s2, exec_lo                                      // 000000006B24: BE82007E
	global_load_b64 v[4:5], v[0:1], off scope:SCOPE_SYS        // 000000006B28: EE05407C 000C0004 00000000
	s_wait_loadcnt 0x0                                         // 000000006B34: BFC00000
	s_wait_xcnt 0x0                                            // 000000006B38: BFC50000
	v_cmpx_ne_u64_e64 v[4:5], v[2:3]                           // 000000006B3C: D4DD007E 02020504
	s_cbranch_execz 11                                         // 000000006B44: BFA5000B <__amd_rocclr_batchMemOp+0x674>
	s_mov_b32 s3, 0                                            // 000000006B48: BE830080
	s_sleep 1                                                  // 000000006B4C: BF830001
	global_load_b64 v[4:5], v[0:1], off scope:SCOPE_SYS        // 000000006B50: EE05407C 000C0004 00000000
	s_wait_loadcnt 0x0                                         // 000000006B5C: BFC00000
	v_cmp_eq_u64_e32 vcc_lo, v[4:5], v[2:3]                    // 000000006B60: 7CB40504
	s_or_b32 s3, vcc_lo, s3                                    // 000000006B64: 8C03036A
	s_wait_xcnt 0x0                                            // 000000006B68: BFC50000
	s_and_not1_b32 exec_lo, exec_lo, s3                        // 000000006B6C: 917E037E
	s_cbranch_execnz 65526                                     // 000000006B70: BFA6FFF6 <__amd_rocclr_batchMemOp+0x64c>
	s_or_b32 exec_lo, exec_lo, s2                              // 000000006B74: 8C7E027E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000006B78: BF870009
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006B7C: 8C7E017E
	s_and_not1_saveexec_b32 s0, s0                             // 000000006B80: BE803000
	s_cbranch_execz 51                                         // 000000006B84: BFA50033 <__amd_rocclr_batchMemOp+0x754>
	v_cmp_eq_u32_e32 vcc_lo, 0, v6                             // 000000006B88: 7C940C80
	s_and_b32 exec_lo, exec_lo, vcc_lo                         // 000000006B8C: 8B7E6A7E
	s_cbranch_execz 48                                         // 000000006B90: BFA50030 <__amd_rocclr_batchMemOp+0x754>
	s_mov_b32 s0, exec_lo                                      // 000000006B94: BE80007E
	s_wait_loadcnt 0x0                                         // 000000006B98: BFC00000
	v_cmpx_ne_u64_e32 0, v[0:1]                                // 000000006B9C: 7DBA0080
	s_xor_b32 s0, exec_lo, s0                                  // 000000006BA0: 8D00007E
	s_cbranch_execz 21                                         // 000000006BA4: BFA50015 <__amd_rocclr_batchMemOp+0x6fc>
	flat_load_b32 v3, v[0:1] scope:SCOPE_SYS                   // 000000006BA8: EC05007C 000C0003 00000000
	s_mov_b32 s1, exec_lo                                      // 000000006BB4: BE81007E
	s_wait_loadcnt_dscnt 0x0                                   // 000000006BB8: BFC80000
	s_wait_xcnt 0x0                                            // 000000006BBC: BFC50000
	v_cmpx_lt_u32_e64 v3, v2                                   // 000000006BC0: D4C9007E 02020503
	s_cbranch_execz 11                                         // 000000006BC8: BFA5000B <__amd_rocclr_batchMemOp+0x6f8>
	s_mov_b32 s2, 0                                            // 000000006BCC: BE820080
	s_sleep 1                                                  // 000000006BD0: BF830001
	flat_load_b32 v3, v[0:1] scope:SCOPE_SYS                   // 000000006BD4: EC05007C 000C0003 00000000
	s_wait_loadcnt_dscnt 0x0                                   // 000000006BE0: BFC80000
	v_cmp_ge_u32_e32 vcc_lo, v3, v2                            // 000000006BE4: 7C9C0503
	s_or_b32 s2, vcc_lo, s2                                    // 000000006BE8: 8C02026A
	s_wait_xcnt 0x0                                            // 000000006BEC: BFC50000
	s_and_not1_b32 exec_lo, exec_lo, s2                        // 000000006BF0: 917E027E
	s_cbranch_execnz 65526                                     // 000000006BF4: BFA6FFF6 <__amd_rocclr_batchMemOp+0x6d0>
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006BF8: 8C7E017E
	s_and_not1_saveexec_b32 s0, s0                             // 000000006BFC: BE803000
	s_cbranch_execz 20                                         // 000000006C00: BFA50014 <__amd_rocclr_batchMemOp+0x754>
	v_mov_b64_e32 v[0:1], 0                                    // 000000006C04: 7E003A80
	global_load_b64 v[4:5], v[0:1], off scope:SCOPE_SYS        // 000000006C08: EE05407C 000C0004 00000000
	s_wait_loadcnt 0x0                                         // 000000006C14: BFC00000
	v_cmp_lt_u64_e32 vcc_lo, v[4:5], v[2:3]                    // 000000006C18: 7CB20504
	s_wait_xcnt 0x0                                            // 000000006C1C: BFC50000
	s_and_b32 exec_lo, exec_lo, vcc_lo                         // 000000006C20: 8B7E6A7E
	s_cbranch_execz 11                                         // 000000006C24: BFA5000B <__amd_rocclr_batchMemOp+0x754>
	s_mov_b32 s0, 0                                            // 000000006C28: BE800080
	s_sleep 1                                                  // 000000006C2C: BF830001
	global_load_b64 v[4:5], v[0:1], off scope:SCOPE_SYS        // 000000006C30: EE05407C 000C0004 00000000
	s_wait_loadcnt 0x0                                         // 000000006C3C: BFC00000
	v_cmp_ge_u64_e32 vcc_lo, v[4:5], v[2:3]                    // 000000006C40: 7CBC0504
	s_or_b32 s0, vcc_lo, s0                                    // 000000006C44: 8C00006A
	s_wait_xcnt 0x0                                            // 000000006C48: BFC50000
	s_and_not1_b32 exec_lo, exec_lo, s0                        // 000000006C4C: 917E007E
	s_cbranch_execnz 65526                                     // 000000006C50: BFA6FFF6 <__amd_rocclr_batchMemOp+0x72c>
	s_endpgm                                                   // 000000006C54: BFB00000
	s_nop 0                                                    // 000000006C58: BF800000
	s_nop 0                                                    // 000000006C5C: BF800000
	s_nop 0                                                    // 000000006C60: BF800000
	s_nop 0                                                    // 000000006C64: BF800000
	s_nop 0                                                    // 000000006C68: BF800000
	s_nop 0                                                    // 000000006C6C: BF800000
	s_nop 0                                                    // 000000006C70: BF800000
	s_nop 0                                                    // 000000006C74: BF800000
	s_nop 0                                                    // 000000006C78: BF800000
	s_nop 0                                                    // 000000006C7C: BF800000
	s_nop 0                                                    // 000000006C80: BF800000
	s_nop 0                                                    // 000000006C84: BF800000
	s_nop 0                                                    // 000000006C88: BF800000
	s_nop 0                                                    // 000000006C8C: BF800000
	s_nop 0                                                    // 000000006C90: BF800000
	s_nop 0                                                    // 000000006C94: BF800000
	s_nop 0                                                    // 000000006C98: BF800000
	s_nop 0                                                    // 000000006C9C: BF800000
	s_nop 0                                                    // 000000006CA0: BF800000
	s_nop 0                                                    // 000000006CA4: BF800000
	s_nop 0                                                    // 000000006CA8: BF800000
	s_nop 0                                                    // 000000006CAC: BF800000
	s_nop 0                                                    // 000000006CB0: BF800000
	s_nop 0                                                    // 000000006CB4: BF800000
	s_nop 0                                                    // 000000006CB8: BF800000
	s_nop 0                                                    // 000000006CBC: BF800000
	s_nop 0                                                    // 000000006CC0: BF800000
	s_nop 0                                                    // 000000006CC4: BF800000
	s_nop 0                                                    // 000000006CC8: BF800000
	s_nop 0                                                    // 000000006CCC: BF800000
	s_nop 0                                                    // 000000006CD0: BF800000
	s_nop 0                                                    // 000000006CD4: BF800000
	s_nop 0                                                    // 000000006CD8: BF800000
	s_nop 0                                                    // 000000006CDC: BF800000
	s_nop 0                                                    // 000000006CE0: BF800000
	s_nop 0                                                    // 000000006CE4: BF800000
	s_nop 0                                                    // 000000006CE8: BF800000
	s_nop 0                                                    // 000000006CEC: BF800000
	s_nop 0                                                    // 000000006CF0: BF800000
	s_nop 0                                                    // 000000006CF4: BF800000
	s_nop 0                                                    // 000000006CF8: BF800000
	s_nop 0                                                    // 000000006CFC: BF800000

0000000000006d00 <__amd_rocclr_streamOpsWrite>:
; __amd_rocclr_streamOpsWrite():
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1       // 000000006D00: B9800641 00000001
	s_cmp_eq_u64 s[2:3], 0                                     // 000000006D08: BF108002
	s_cbranch_scc1 12                                          // 000000006D0C: BFA2000C <__amd_rocclr_streamOpsWrite+0x40>
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s6              // 000000006D10: CA100080 00000006
	global_store_b32 v0, v1, s[2:3] scope:SCOPE_SYS            // 000000006D18: EE068002 008C0000 00000000
	s_cbranch_execnz 5                                         // 000000006D24: BFA60005 <__amd_rocclr_streamOpsWrite+0x3c>
	v_mov_b64_e32 v[0:1], s[6:7]                               // 000000006D28: 7E003A06
	v_mov_b32_e32 v2, 0                                        // 000000006D2C: 7E040280
	global_store_b64 v2, v[0:1], s[4:5] scope:SCOPE_SYS        // 000000006D30: EE06C004 000C0000 00000002
	s_endpgm                                                   // 000000006D3C: BFB00000
	s_branch 65529                                             // 000000006D40: BFA0FFF9 <__amd_rocclr_streamOpsWrite+0x28>
	s_nop 0                                                    // 000000006D44: BF800000
	s_nop 0                                                    // 000000006D48: BF800000
	s_nop 0                                                    // 000000006D4C: BF800000
	s_nop 0                                                    // 000000006D50: BF800000
	s_nop 0                                                    // 000000006D54: BF800000
	s_nop 0                                                    // 000000006D58: BF800000
	s_nop 0                                                    // 000000006D5C: BF800000
	s_nop 0                                                    // 000000006D60: BF800000
	s_nop 0                                                    // 000000006D64: BF800000
	s_nop 0                                                    // 000000006D68: BF800000
	s_nop 0                                                    // 000000006D6C: BF800000
	s_nop 0                                                    // 000000006D70: BF800000
	s_nop 0                                                    // 000000006D74: BF800000
	s_nop 0                                                    // 000000006D78: BF800000
	s_nop 0                                                    // 000000006D7C: BF800000
	s_nop 0                                                    // 000000006D80: BF800000
	s_nop 0                                                    // 000000006D84: BF800000
	s_nop 0                                                    // 000000006D88: BF800000
	s_nop 0                                                    // 000000006D8C: BF800000
	s_nop 0                                                    // 000000006D90: BF800000
	s_nop 0                                                    // 000000006D94: BF800000
	s_nop 0                                                    // 000000006D98: BF800000
	s_nop 0                                                    // 000000006D9C: BF800000
	s_nop 0                                                    // 000000006DA0: BF800000
	s_nop 0                                                    // 000000006DA4: BF800000
	s_nop 0                                                    // 000000006DA8: BF800000
	s_nop 0                                                    // 000000006DAC: BF800000
	s_nop 0                                                    // 000000006DB0: BF800000
	s_nop 0                                                    // 000000006DB4: BF800000
	s_nop 0                                                    // 000000006DB8: BF800000
	s_nop 0                                                    // 000000006DBC: BF800000
	s_nop 0                                                    // 000000006DC0: BF800000
	s_nop 0                                                    // 000000006DC4: BF800000
	s_nop 0                                                    // 000000006DC8: BF800000
	s_nop 0                                                    // 000000006DCC: BF800000
	s_nop 0                                                    // 000000006DD0: BF800000
	s_nop 0                                                    // 000000006DD4: BF800000
	s_nop 0                                                    // 000000006DD8: BF800000
	s_nop 0                                                    // 000000006DDC: BF800000
	s_nop 0                                                    // 000000006DE0: BF800000
	s_nop 0                                                    // 000000006DE4: BF800000
	s_nop 0                                                    // 000000006DE8: BF800000
	s_nop 0                                                    // 000000006DEC: BF800000
	s_nop 0                                                    // 000000006DF0: BF800000
	s_nop 0                                                    // 000000006DF4: BF800000
	s_nop 0                                                    // 000000006DF8: BF800000
	s_nop 0                                                    // 000000006DFC: BF800000

0000000000006e00 <__amd_rocclr_streamOpsIncrement>:
; __amd_rocclr_streamOpsIncrement():
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1       // 000000006E00: B9800641 00000001
	s_cmp_eq_u64 s[2:3], 0                                     // 000000006E08: BF108002
	s_cbranch_scc1 22                                          // 000000006E0C: BFA20016 <__amd_rocclr_streamOpsIncrement+0x68>
	s_mov_b32 s8, exec_lo                                      // 000000006E10: BE88007E
	s_mov_b32 s0, 0                                            // 000000006E14: BE800080
	v_mbcnt_lo_u32_b32 v0, s8, 0                               // 000000006E18: D71F0000 02010008
	s_mov_b32 s1, exec_lo                                      // 000000006E20: BE81007E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006E24: BF870001
	v_cmpx_eq_u32_e32 0, v0                                    // 000000006E28: 7D940080
	s_cbranch_execz 8                                          // 000000006E2C: BFA50008 <__amd_rocclr_streamOpsIncrement+0x50>
	s_bcnt1_i32_b32 s8, s8                                     // 000000006E30: BE881808
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000006E34: BF870499
	s_mul_i32 s8, s6, s8                                       // 000000006E38: 96080806
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s8              // 000000006E3C: CA100080 00000008
	global_atomic_add_u32 v0, v1, s[2:3] scope:SCOPE_SYS       // 000000006E44: EE0D4002 008C0000 00000000
	s_wait_xcnt 0x0                                            // 000000006E50: BFC50000
	s_or_b32 exec_lo, exec_lo, s1                              // 000000006E54: 8C7E017E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000006E58: BF870009
	s_and_not1_b32 vcc_lo, exec_lo, s0                         // 000000006E5C: 916A007E
	s_cbranch_vccz 1                                           // 000000006E60: BFA30001 <__amd_rocclr_streamOpsIncrement+0x68>
	s_endpgm                                                   // 000000006E64: BFB00000
	s_mov_b32 s0, exec_lo                                      // 000000006E68: BE80007E
	s_mov_b32 s1, 0                                            // 000000006E6C: BE810080
	v_mbcnt_lo_u32_b32 v0, s0, 0                               // 000000006E70: D71F0000 02010000
	s_mov_b32 s2, exec_lo                                      // 000000006E78: BE82007E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000006E7C: BF870001
	v_cmpx_eq_u32_e32 0, v0                                    // 000000006E80: 7D940080
	s_cbranch_execz 65527                                      // 000000006E84: BFA5FFF7 <__amd_rocclr_streamOpsIncrement+0x64>
	s_bcnt1_i32_b32 s0, s0                                     // 000000006E88: BE801800
	v_mov_b32_e32 v2, 0                                        // 000000006E8C: 7E040280
	s_mul_u64 s[0:1], s[6:7], s[0:1]                           // 000000006E90: AA800006
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000006E94: BF870009
	v_mov_b64_e32 v[0:1], s[0:1]                               // 000000006E98: 7E003A00
	global_atomic_add_u64 v2, v[0:1], s[4:5] scope:SCOPE_SYS   // 000000006E9C: EE10C004 000C0000 00000002
	s_endpgm                                                   // 000000006EA8: BFB00000
	s_nop 0                                                    // 000000006EAC: BF800000
	s_nop 0                                                    // 000000006EB0: BF800000
	s_nop 0                                                    // 000000006EB4: BF800000
	s_nop 0                                                    // 000000006EB8: BF800000
	s_nop 0                                                    // 000000006EBC: BF800000
	s_nop 0                                                    // 000000006EC0: BF800000
	s_nop 0                                                    // 000000006EC4: BF800000
	s_nop 0                                                    // 000000006EC8: BF800000
	s_nop 0                                                    // 000000006ECC: BF800000
	s_nop 0                                                    // 000000006ED0: BF800000
	s_nop 0                                                    // 000000006ED4: BF800000
	s_nop 0                                                    // 000000006ED8: BF800000
	s_nop 0                                                    // 000000006EDC: BF800000
	s_nop 0                                                    // 000000006EE0: BF800000
	s_nop 0                                                    // 000000006EE4: BF800000
	s_nop 0                                                    // 000000006EE8: BF800000
	s_nop 0                                                    // 000000006EEC: BF800000
	s_nop 0                                                    // 000000006EF0: BF800000
	s_nop 0                                                    // 000000006EF4: BF800000
	s_nop 0                                                    // 000000006EF8: BF800000
	s_nop 0                                                    // 000000006EFC: BF800000

0000000000006f00 <__amd_rocclr_streamOpsDecrement>:
; __amd_rocclr_streamOpsDecrement():
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1       // 000000006F00: B9800641 00000001
	s_cmp_eq_u64 s[2:3], 0                                     // 000000006F08: BF108002
	s_cbranch_scc1 12                                          // 000000006F0C: BFA2000C <__amd_rocclr_streamOpsDecrement+0x40>
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s6              // 000000006F10: CA100080 00000006
	flat_atomic_sub_u32 v0, v1, s[2:3] scope:SCOPE_SYS         // 000000006F18: EC0D8002 008C0000 00000000
	s_cbranch_execnz 5                                         // 000000006F24: BFA60005 <__amd_rocclr_streamOpsDecrement+0x3c>
	v_mov_b64_e32 v[0:1], s[6:7]                               // 000000006F28: 7E003A06
	v_mov_b32_e32 v2, 0                                        // 000000006F2C: 7E040280
	flat_atomic_sub_u64 v2, v[0:1], s[4:5] scope:SCOPE_SYS     // 000000006F30: EC110004 000C0000 00000002
	s_endpgm                                                   // 000000006F3C: BFB00000
	s_branch 65529                                             // 000000006F40: BFA0FFF9 <__amd_rocclr_streamOpsDecrement+0x28>
	s_nop 0                                                    // 000000006F44: BF800000
	s_nop 0                                                    // 000000006F48: BF800000
	s_nop 0                                                    // 000000006F4C: BF800000
	s_nop 0                                                    // 000000006F50: BF800000
	s_nop 0                                                    // 000000006F54: BF800000
	s_nop 0                                                    // 000000006F58: BF800000
	s_nop 0                                                    // 000000006F5C: BF800000
	s_nop 0                                                    // 000000006F60: BF800000
	s_nop 0                                                    // 000000006F64: BF800000
	s_nop 0                                                    // 000000006F68: BF800000
	s_nop 0                                                    // 000000006F6C: BF800000
	s_nop 0                                                    // 000000006F70: BF800000
	s_nop 0                                                    // 000000006F74: BF800000
	s_nop 0                                                    // 000000006F78: BF800000
	s_nop 0                                                    // 000000006F7C: BF800000
	s_nop 0                                                    // 000000006F80: BF800000
	s_nop 0                                                    // 000000006F84: BF800000
	s_nop 0                                                    // 000000006F88: BF800000
	s_nop 0                                                    // 000000006F8C: BF800000
	s_nop 0                                                    // 000000006F90: BF800000
	s_nop 0                                                    // 000000006F94: BF800000
	s_nop 0                                                    // 000000006F98: BF800000
	s_nop 0                                                    // 000000006F9C: BF800000
	s_nop 0                                                    // 000000006FA0: BF800000
	s_nop 0                                                    // 000000006FA4: BF800000
	s_nop 0                                                    // 000000006FA8: BF800000
	s_nop 0                                                    // 000000006FAC: BF800000
	s_nop 0                                                    // 000000006FB0: BF800000
	s_nop 0                                                    // 000000006FB4: BF800000
	s_nop 0                                                    // 000000006FB8: BF800000
	s_nop 0                                                    // 000000006FBC: BF800000
	s_nop 0                                                    // 000000006FC0: BF800000
	s_nop 0                                                    // 000000006FC4: BF800000
	s_nop 0                                                    // 000000006FC8: BF800000
	s_nop 0                                                    // 000000006FCC: BF800000
	s_nop 0                                                    // 000000006FD0: BF800000
	s_nop 0                                                    // 000000006FD4: BF800000
	s_nop 0                                                    // 000000006FD8: BF800000
	s_nop 0                                                    // 000000006FDC: BF800000
	s_nop 0                                                    // 000000006FE0: BF800000
	s_nop 0                                                    // 000000006FE4: BF800000
	s_nop 0                                                    // 000000006FE8: BF800000
	s_nop 0                                                    // 000000006FEC: BF800000
	s_nop 0                                                    // 000000006FF0: BF800000
	s_nop 0                                                    // 000000006FF4: BF800000
	s_nop 0                                                    // 000000006FF8: BF800000
	s_nop 0                                                    // 000000006FFC: BF800000

0000000000007000 <__amd_rocclr_streamOpsWait>:
; __amd_rocclr_streamOpsWait():
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1       // 000000007000: B9800641 00000001
	v_cmp_lt_i64_e64 s0, s[8:9], 2                             // 000000007008: D4510000 02010408
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000007010: 8B6A007E
	s_mov_b32 s0, -1                                           // 000000007014: BE8000C1
	s_cbranch_vccnz 85                                         // 000000007018: BFA40055 <__amd_rocclr_streamOpsWait+0x170>
	v_cmp_lt_i64_e64 s0, s[8:9], 3                             // 00000000701C: D4510000 02010608
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000007024: 8B6A007E
	s_mov_b32 s0, -1                                           // 000000007028: BE8000C1
	s_cbranch_vccnz 37                                         // 00000000702C: BFA40025 <__amd_rocclr_streamOpsWait+0xc4>
	s_cmp_eq_u64 s[8:9], 3                                     // 000000007030: BF108308
	s_cbranch_scc0 34                                          // 000000007034: BFA10022 <__amd_rocclr_streamOpsWait+0xc0>
	s_cmp_lg_u64 s[2:3], 0                                     // 000000007038: BF118002
	s_cbranch_scc0 14                                          // 00000000703C: BFA1000E <__amd_rocclr_streamOpsWait+0x78>
	v_mov_b32_e32 v0, 0                                        // 000000007040: 7E000280
	global_load_b32 v1, v0, s[2:3] scope:SCOPE_SYS             // 000000007044: EE050002 000C0001 00000000
	s_wait_loadcnt 0x0                                         // 000000007050: BFC00000
	v_readfirstlane_b32 s0, v1                                 // 000000007054: 7E000501
	s_or_b32 s0, s0, s6                                        // 000000007058: 8C000600
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000705C: BF870499
	s_and_b32 s0, s0, s10                                      // 000000007060: 8B000A00
	s_cmp_lg_u32 s0, s10                                       // 000000007064: BF070A00
	s_cbranch_scc1 2                                           // 000000007068: BFA20002 <__amd_rocclr_streamOpsWait+0x74>
	s_sleep 1                                                  // 00000000706C: BF830001
	s_branch 65524                                             // 000000007070: BFA0FFF4 <__amd_rocclr_streamOpsWait+0x44>
	s_mov_b32 s0, 0                                            // 000000007074: BE800080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000007078: BF870009
	s_and_not1_b32 vcc_lo, exec_lo, s0                         // 00000000707C: 916A007E
	s_cbranch_vccnz 15                                         // 000000007080: BFA4000F <__amd_rocclr_streamOpsWait+0xc0>
	v_mov_b32_e32 v0, 0                                        // 000000007084: 7E000280
	global_load_b64 v[2:3], v0, s[4:5] scope:SCOPE_SYS         // 000000007088: EE054004 000C0002 00000000
	s_wait_loadcnt 0x0                                         // 000000007094: BFC00000
	v_or_b32_e32 v1, s6, v2                                    // 000000007098: 38020406
	v_or_b32_e32 v3, s7, v3                                    // 00000000709C: 38060607
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000070A0: BF870112
	v_and_b32_e32 v2, s10, v1                                  // 0000000070A4: 3604020A
	v_and_b32_e32 v3, s11, v3                                  // 0000000070A8: 3606060B
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000070AC: BF870001
	v_cmp_ne_u64_e32 vcc_lo, s[10:11], v[2:3]                  // 0000000070B0: 7CBA040A
	s_cbranch_vccnz 2                                          // 0000000070B4: BFA40002 <__amd_rocclr_streamOpsWait+0xc0>
	s_sleep 1                                                  // 0000000070B8: BF830001
	s_branch 65522                                             // 0000000070BC: BFA0FFF2 <__amd_rocclr_streamOpsWait+0x88>
	s_mov_b32 s0, 0                                            // 0000000070C0: BE800080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000070C4: BF870009
	s_and_not1_b32 vcc_lo, exec_lo, s0                         // 0000000070C8: 916A007E
	s_cbranch_vccnz 39                                         // 0000000070CC: BFA40027 <__amd_rocclr_streamOpsWait+0x16c>
	s_and_b64 s[0:1], s[10:11], s[6:7]                         // 0000000070D0: 8B80060A
	s_cmp_lg_u64 s[2:3], 0                                     // 0000000070D4: BF118002
	s_cbranch_scc0 20                                          // 0000000070D8: BFA10014 <__amd_rocclr_streamOpsWait+0x12c>
	v_mov_b32_e32 v0, 0                                        // 0000000070DC: 7E000280
	s_mov_b32 s12, 0                                           // 0000000070E0: BE8C0080
	global_load_b32 v1, v0, s[2:3] scope:SCOPE_SYS             // 0000000070E4: EE050002 000C0001 00000000
	s_wait_loadcnt 0x0                                         // 0000000070F0: BFC00000
	v_and_b32_e32 v1, s0, v1                                   // 0000000070F4: 36020200
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000070F8: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v1                             // 0000000070FC: 7C9A0280
	s_cbranch_vccnz 11                                         // 000000007100: BFA4000B <__amd_rocclr_streamOpsWait+0x130>
	s_sleep 1                                                  // 000000007104: BF830001
	global_load_b32 v1, v0, s[2:3] scope:SCOPE_SYS             // 000000007108: EE050002 000C0001 00000000
	s_wait_loadcnt 0x0                                         // 000000007114: BFC00000
	v_and_b32_e32 v1, s0, v1                                   // 000000007118: 36020200
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000711C: BF870001
	v_cmp_ne_u32_e32 vcc_lo, 0, v1                             // 000000007120: 7C9A0280
	s_cbranch_vccz 65527                                       // 000000007124: BFA3FFF7 <__amd_rocclr_streamOpsWait+0x104>
	s_branch 1                                                 // 000000007128: BFA00001 <__amd_rocclr_streamOpsWait+0x130>
	s_mov_b32 s12, -1                                          // 00000000712C: BE8C00C1
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000007130: BF870009
	s_and_not1_b32 vcc_lo, exec_lo, s12                        // 000000007134: 916A0C7E
	s_cbranch_vccnz 12                                         // 000000007138: BFA4000C <__amd_rocclr_streamOpsWait+0x16c>
	v_mov_b32_e32 v0, 0                                        // 00000000713C: 7E000280
	global_load_b64 v[2:3], v0, s[4:5] scope:SCOPE_SYS         // 000000007140: EE054004 000C0002 00000000
	s_wait_loadcnt 0x0                                         // 00000000714C: BFC00000
	v_and_b32_e32 v2, s0, v2                                   // 000000007150: 36040400
	v_and_b32_e32 v3, s1, v3                                   // 000000007154: 36060601
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007158: BF870001
	v_cmp_ne_u64_e32 vcc_lo, 0, v[2:3]                         // 00000000715C: 7CBA0480
	s_cbranch_vccnz 2                                          // 000000007160: BFA40002 <__amd_rocclr_streamOpsWait+0x16c>
	s_sleep 1                                                  // 000000007164: BF830001
	s_branch 65525                                             // 000000007168: BFA0FFF5 <__amd_rocclr_streamOpsWait+0x140>
	s_mov_b32 s0, 0                                            // 00000000716C: BE800080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000007170: BF870009
	s_and_not1_b32 vcc_lo, exec_lo, s0                         // 000000007174: 916A007E
	s_cbranch_vccnz 67                                         // 000000007178: BFA40043 <__amd_rocclr_streamOpsWait+0x288>
	v_cmp_gt_i64_e64 s0, s[8:9], 0                             // 00000000717C: D4540000 02010008
	s_and_b32 vcc_lo, exec_lo, s0                              // 000000007184: 8B6A007E
	s_mov_b32 s0, -1                                           // 000000007188: BE8000C1
	s_cbranch_vccz 30                                          // 00000000718C: BFA3001E <__amd_rocclr_streamOpsWait+0x208>
	s_cmp_lg_u64 s[2:3], 0                                     // 000000007190: BF118002
	s_cbranch_scc0 12                                          // 000000007194: BFA1000C <__amd_rocclr_streamOpsWait+0x1c8>
	v_mov_b32_e32 v0, 0                                        // 000000007198: 7E000280
	global_load_b32 v1, v0, s[2:3] scope:SCOPE_SYS             // 00000000719C: EE050002 000C0001 00000000
	s_wait_loadcnt 0x0                                         // 0000000071A8: BFC00000
	v_and_b32_e32 v1, s10, v1                                  // 0000000071AC: 3602020A
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000071B0: BF870001
	v_cmp_eq_u32_e32 vcc_lo, s6, v1                            // 0000000071B4: 7C940206
	s_cbranch_vccnz 2                                          // 0000000071B8: BFA40002 <__amd_rocclr_streamOpsWait+0x1c4>
	s_sleep 1                                                  // 0000000071BC: BF830001
	s_branch 65526                                             // 0000000071C0: BFA0FFF6 <__amd_rocclr_streamOpsWait+0x19c>
	s_mov_b32 s0, 0                                            // 0000000071C4: BE800080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000071C8: BF870009
	s_and_not1_b32 vcc_lo, exec_lo, s0                         // 0000000071CC: 916A007E
	s_cbranch_vccnz 12                                         // 0000000071D0: BFA4000C <__amd_rocclr_streamOpsWait+0x204>
	v_mov_b32_e32 v0, 0                                        // 0000000071D4: 7E000280
	global_load_b64 v[2:3], v0, s[4:5] scope:SCOPE_SYS         // 0000000071D8: EE054004 000C0002 00000000
	s_wait_loadcnt 0x0                                         // 0000000071E4: BFC00000
	v_and_b32_e32 v2, s10, v2                                  // 0000000071E8: 3604040A
	v_and_b32_e32 v3, s11, v3                                  // 0000000071EC: 3606060B
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000071F0: BF870001
	v_cmp_eq_u64_e32 vcc_lo, s[6:7], v[2:3]                    // 0000000071F4: 7CB40406
	s_cbranch_vccnz 2                                          // 0000000071F8: BFA40002 <__amd_rocclr_streamOpsWait+0x204>
	s_sleep 1                                                  // 0000000071FC: BF830001
	s_branch 65525                                             // 000000007200: BFA0FFF5 <__amd_rocclr_streamOpsWait+0x1d8>
	s_mov_b32 s0, 0                                            // 000000007204: BE800080
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000007208: BF870009
	s_and_not1_b32 vcc_lo, exec_lo, s0                         // 00000000720C: 916A007E
	s_cbranch_vccnz 29                                         // 000000007210: BFA4001D <__amd_rocclr_streamOpsWait+0x288>
	s_cmp_lg_u64 s[8:9], 0                                     // 000000007214: BF118008
	s_cbranch_scc1 27                                          // 000000007218: BFA2001B <__amd_rocclr_streamOpsWait+0x288>
	s_cmp_lg_u64 s[2:3], 0                                     // 00000000721C: BF118002
	s_cbranch_scc0 13                                          // 000000007220: BFA1000D <__amd_rocclr_streamOpsWait+0x258>
	v_mov_b32_e32 v0, 0                                        // 000000007224: 7E000280
	global_load_b32 v1, v0, s[2:3] scope:SCOPE_SYS             // 000000007228: EE050002 000C0001 00000000
	s_wait_loadcnt 0x0                                         // 000000007234: BFC00000
	v_and_b32_e32 v1, s10, v1                                  // 000000007238: 3602020A
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000723C: BF870001
	v_cmp_le_u32_e32 vcc_lo, s6, v1                            // 000000007240: 7C960206
	s_cbranch_vccnz 2                                          // 000000007244: BFA40002 <__amd_rocclr_streamOpsWait+0x250>
	s_sleep 1                                                  // 000000007248: BF830001
	s_branch 65526                                             // 00000000724C: BFA0FFF6 <__amd_rocclr_streamOpsWait+0x228>
	s_cbranch_execz 1                                          // 000000007250: BFA50001 <__amd_rocclr_streamOpsWait+0x258>
	s_branch 12                                                // 000000007254: BFA0000C <__amd_rocclr_streamOpsWait+0x288>
	v_mov_b32_e32 v0, 0                                        // 000000007258: 7E000280
	global_load_b64 v[2:3], v0, s[4:5] scope:SCOPE_SYS         // 00000000725C: EE054004 000C0002 00000000
	s_wait_loadcnt 0x0                                         // 000000007268: BFC00000
	v_and_b32_e32 v2, s10, v2                                  // 00000000726C: 3604040A
	v_and_b32_e32 v3, s11, v3                                  // 000000007270: 3606060B
	s_delay_alu instid0(VALU_DEP_1)                            // 000000007274: BF870001
	v_cmp_le_u64_e32 vcc_lo, s[6:7], v[2:3]                    // 000000007278: 7CB60406
	s_cbranch_vccnz 2                                          // 00000000727C: BFA40002 <__amd_rocclr_streamOpsWait+0x288>
	s_sleep 1                                                  // 000000007280: BF830001
	s_branch 65525                                             // 000000007284: BFA0FFF5 <__amd_rocclr_streamOpsWait+0x25c>
	s_endpgm                                                   // 000000007288: BFB00000
	s_nop 0                                                    // 00000000728C: BF800000
	s_nop 0                                                    // 000000007290: BF800000
	s_nop 0                                                    // 000000007294: BF800000
	s_nop 0                                                    // 000000007298: BF800000
	s_nop 0                                                    // 00000000729C: BF800000
	s_nop 0                                                    // 0000000072A0: BF800000
	s_nop 0                                                    // 0000000072A4: BF800000
	s_nop 0                                                    // 0000000072A8: BF800000
	s_nop 0                                                    // 0000000072AC: BF800000
	s_nop 0                                                    // 0000000072B0: BF800000
	s_nop 0                                                    // 0000000072B4: BF800000
	s_nop 0                                                    // 0000000072B8: BF800000
	s_nop 0                                                    // 0000000072BC: BF800000
	s_nop 0                                                    // 0000000072C0: BF800000
	s_nop 0                                                    // 0000000072C4: BF800000
	s_nop 0                                                    // 0000000072C8: BF800000
	s_nop 0                                                    // 0000000072CC: BF800000
	s_nop 0                                                    // 0000000072D0: BF800000
	s_nop 0                                                    // 0000000072D4: BF800000
	s_nop 0                                                    // 0000000072D8: BF800000
	s_nop 0                                                    // 0000000072DC: BF800000
	s_nop 0                                                    // 0000000072E0: BF800000
	s_nop 0                                                    // 0000000072E4: BF800000
	s_nop 0                                                    // 0000000072E8: BF800000
	s_nop 0                                                    // 0000000072EC: BF800000
	s_nop 0                                                    // 0000000072F0: BF800000
	s_nop 0                                                    // 0000000072F4: BF800000
	s_nop 0                                                    // 0000000072F8: BF800000
	s_nop 0                                                    // 0000000072FC: BF800000

0000000000007300 <__amd_rocclr_initHeap>:
; __amd_rocclr_initHeap():
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1       // 000000007300: B9800641 00000001
	s_mov_b32 s0, s7                                           // 000000007308: BE800007
	s_cmp_eq_u32 s6, 0                                         // 00000000730C: BF068006
	s_mov_b32 s8, 0                                            // 000000007310: BE880080
	s_cbranch_scc1 102                                         // 000000007314: BFA20066 <__amd_rocclr_initHeap+0x1b0>
	s_mov_b32 s9, s8                                           // 000000007318: BE890008
	s_mov_b32 s10, s8                                          // 00000000731C: BE8A0008
	s_mov_b32 s11, s8                                          // 000000007320: BE8B0008
	v_mov_b64_e32 v[2:3], s[8:9]                               // 000000007324: 7E043A08
	v_mov_b64_e32 v[4:5], s[10:11]                             // 000000007328: 7E083A0A
	s_clause 0x1f                                              // 00000000732C: BF85001F
	global_store_b128 v0, v[2:5], s[2:3] scale_offset          // 000000007330: EE074002 01010000 00000000
	global_store_b128 v0, v[2:5], s[2:3] offset:4096 scale_offset// 00000000733C: EE074002 01010000 00100000
	global_store_b128 v0, v[2:5], s[2:3] offset:8192 scale_offset// 000000007348: EE074002 01010000 00200000
	global_store_b128 v0, v[2:5], s[2:3] offset:12288 scale_offset// 000000007354: EE074002 01010000 00300000
	global_store_b128 v0, v[2:5], s[2:3] offset:16384 scale_offset// 000000007360: EE074002 01010000 00400000
	global_store_b128 v0, v[2:5], s[2:3] offset:20480 scale_offset// 00000000736C: EE074002 01010000 00500000
	global_store_b128 v0, v[2:5], s[2:3] offset:24576 scale_offset// 000000007378: EE074002 01010000 00600000
	global_store_b128 v0, v[2:5], s[2:3] offset:28672 scale_offset// 000000007384: EE074002 01010000 00700000
	global_store_b128 v0, v[2:5], s[2:3] offset:32768 scale_offset// 000000007390: EE074002 01010000 00800000
	global_store_b128 v0, v[2:5], s[2:3] offset:36864 scale_offset// 00000000739C: EE074002 01010000 00900000
	global_store_b128 v0, v[2:5], s[2:3] offset:40960 scale_offset// 0000000073A8: EE074002 01010000 00A00000
	global_store_b128 v0, v[2:5], s[2:3] offset:45056 scale_offset// 0000000073B4: EE074002 01010000 00B00000
	global_store_b128 v0, v[2:5], s[2:3] offset:49152 scale_offset// 0000000073C0: EE074002 01010000 00C00000
	global_store_b128 v0, v[2:5], s[2:3] offset:53248 scale_offset// 0000000073CC: EE074002 01010000 00D00000
	global_store_b128 v0, v[2:5], s[2:3] offset:57344 scale_offset// 0000000073D8: EE074002 01010000 00E00000
	global_store_b128 v0, v[2:5], s[2:3] offset:61440 scale_offset// 0000000073E4: EE074002 01010000 00F00000
	global_store_b128 v0, v[2:5], s[2:3] offset:65536 scale_offset// 0000000073F0: EE074002 01010000 01000000
	global_store_b128 v0, v[2:5], s[2:3] offset:69632 scale_offset// 0000000073FC: EE074002 01010000 01100000
	global_store_b128 v0, v[2:5], s[2:3] offset:73728 scale_offset// 000000007408: EE074002 01010000 01200000
	global_store_b128 v0, v[2:5], s[2:3] offset:77824 scale_offset// 000000007414: EE074002 01010000 01300000
	global_store_b128 v0, v[2:5], s[2:3] offset:81920 scale_offset// 000000007420: EE074002 01010000 01400000
	global_store_b128 v0, v[2:5], s[2:3] offset:86016 scale_offset// 00000000742C: EE074002 01010000 01500000
	global_store_b128 v0, v[2:5], s[2:3] offset:90112 scale_offset// 000000007438: EE074002 01010000 01600000
	global_store_b128 v0, v[2:5], s[2:3] offset:94208 scale_offset// 000000007444: EE074002 01010000 01700000
	global_store_b128 v0, v[2:5], s[2:3] offset:98304 scale_offset// 000000007450: EE074002 01010000 01800000
	global_store_b128 v0, v[2:5], s[2:3] offset:102400 scale_offset// 00000000745C: EE074002 01010000 01900000
	global_store_b128 v0, v[2:5], s[2:3] offset:106496 scale_offset// 000000007468: EE074002 01010000 01A00000
	global_store_b128 v0, v[2:5], s[2:3] offset:110592 scale_offset// 000000007474: EE074002 01010000 01B00000
	global_store_b128 v0, v[2:5], s[2:3] offset:114688 scale_offset// 000000007480: EE074002 01010000 01C00000
	global_store_b128 v0, v[2:5], s[2:3] offset:118784 scale_offset// 00000000748C: EE074002 01010000 01D00000
	global_store_b128 v0, v[2:5], s[2:3] offset:122880 scale_offset// 000000007498: EE074002 01010000 01E00000
	global_store_b128 v0, v[2:5], s[2:3] offset:126976 scale_offset// 0000000074A4: EE074002 01010000 01F00000
	s_wait_storecnt 0x0                                        // 0000000074B0: BFC10000
	global_wb scope:SCOPE_DEV                                  // 0000000074B4: EE0B007C 00080000 00000000
	s_wait_storecnt 0x0                                        // 0000000074C0: BFC10000
	s_barrier_signal -1                                        // 0000000074C4: BE804EC1
	s_mov_b32 s1, exec_lo                                      // 0000000074C8: BE81007E
	s_barrier_wait 0xffff                                      // 0000000074CC: BF94FFFF
	v_cmpx_gt_u32_e32 16, v0                                   // 0000000074D0: 7D980090
	s_cbranch_execz 6                                          // 0000000074D4: BFA50006 <__amd_rocclr_initHeap+0x1f0>
	v_dual_mov_b32 v2, 0x100 :: v_dual_lshlrev_b32 v1, 7, v0   // 0000000074D8: CA2200FF 02000087 00000100
	global_store_b32 v1, v2, s[2:3] offset:4096 scope:SCOPE_DEV// 0000000074E4: EE068002 01080000 00100001
	s_wait_xcnt 0x0                                            // 0000000074F0: BFC50000
	s_or_b32 exec_lo, exec_lo, s1                              // 0000000074F4: 8C7E017E
	s_mov_b32 s1, 0                                            // 0000000074F8: BE810080
	s_mov_b32 s6, exec_lo                                      // 0000000074FC: BE86007E
	v_cmpx_eq_u32_e32 0, v0                                    // 000000007500: 7D940080
	s_cbranch_execz 17                                         // 000000007504: BFA50011 <__amd_rocclr_initHeap+0x24c>
	s_lshl_b64 s[0:1], s[0:1], 21                              // 000000007508: 84809500
	v_mov_b64_e32 v[4:5], s[4:5]                               // 00000000750C: 7E083A04
	s_add_nc_u64 s[0:1], s[0:1], s[4:5]                        // 000000007510: A9800400
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000007514: BF870009
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v0, s0              // 000000007518: CF208080 00000000 00000006
	v_dual_mov_b32 v1, s1 :: v_dual_mov_b32 v2, s4             // 000000007524: CA100001 01020004
	v_mov_b32_e32 v3, s5                                       // 00000000752C: 7E060205
	s_clause 0x1                                               // 000000007530: BF850001
	global_store_b64 v6, v[4:5], s[2:3] offset:108544 scope:SCOPE_DEV// 000000007534: EE06C002 02080000 01A80006
	global_store_b128 v6, v[0:3], s[2:3] offset:108552         // 000000007540: EE074002 00000000 01A80806
	s_endpgm                                                   // 00000000754C: BFB00000
	s_code_end                                                 // 000000007550: BF9F0000
	s_code_end                                                 // 000000007554: BF9F0000
	s_code_end                                                 // 000000007558: BF9F0000
	s_code_end                                                 // 00000000755C: BF9F0000
	s_code_end                                                 // 000000007560: BF9F0000
	s_code_end                                                 // 000000007564: BF9F0000
	s_code_end                                                 // 000000007568: BF9F0000
	s_code_end                                                 // 00000000756C: BF9F0000
	s_code_end                                                 // 000000007570: BF9F0000
	s_code_end                                                 // 000000007574: BF9F0000
	s_code_end                                                 // 000000007578: BF9F0000
	s_code_end                                                 // 00000000757C: BF9F0000
	s_code_end                                                 // 000000007580: BF9F0000
	s_code_end                                                 // 000000007584: BF9F0000
	s_code_end                                                 // 000000007588: BF9F0000
	s_code_end                                                 // 00000000758C: BF9F0000
	s_code_end                                                 // 000000007590: BF9F0000
	s_code_end                                                 // 000000007594: BF9F0000
	s_code_end                                                 // 000000007598: BF9F0000
	s_code_end                                                 // 00000000759C: BF9F0000
	s_code_end                                                 // 0000000075A0: BF9F0000
	s_code_end                                                 // 0000000075A4: BF9F0000
	s_code_end                                                 // 0000000075A8: BF9F0000
	s_code_end                                                 // 0000000075AC: BF9F0000
	s_code_end                                                 // 0000000075B0: BF9F0000
	s_code_end                                                 // 0000000075B4: BF9F0000
	s_code_end                                                 // 0000000075B8: BF9F0000
	s_code_end                                                 // 0000000075BC: BF9F0000
	s_code_end                                                 // 0000000075C0: BF9F0000
	s_code_end                                                 // 0000000075C4: BF9F0000
	s_code_end                                                 // 0000000075C8: BF9F0000
	s_code_end                                                 // 0000000075CC: BF9F0000
	s_code_end                                                 // 0000000075D0: BF9F0000
	s_code_end                                                 // 0000000075D4: BF9F0000
	s_code_end                                                 // 0000000075D8: BF9F0000
	s_code_end                                                 // 0000000075DC: BF9F0000
	s_code_end                                                 // 0000000075E0: BF9F0000
	s_code_end                                                 // 0000000075E4: BF9F0000
	s_code_end                                                 // 0000000075E8: BF9F0000
	s_code_end                                                 // 0000000075EC: BF9F0000
	s_code_end                                                 // 0000000075F0: BF9F0000
	s_code_end                                                 // 0000000075F4: BF9F0000
	s_code_end                                                 // 0000000075F8: BF9F0000
	s_code_end                                                 // 0000000075FC: BF9F0000
	s_code_end                                                 // 000000007600: BF9F0000
	s_code_end                                                 // 000000007604: BF9F0000
	s_code_end                                                 // 000000007608: BF9F0000
	s_code_end                                                 // 00000000760C: BF9F0000
	s_code_end                                                 // 000000007610: BF9F0000
	s_code_end                                                 // 000000007614: BF9F0000
	s_code_end                                                 // 000000007618: BF9F0000
	s_code_end                                                 // 00000000761C: BF9F0000
	s_code_end                                                 // 000000007620: BF9F0000
	s_code_end                                                 // 000000007624: BF9F0000
	s_code_end                                                 // 000000007628: BF9F0000
	s_code_end                                                 // 00000000762C: BF9F0000
	s_code_end                                                 // 000000007630: BF9F0000
	s_code_end                                                 // 000000007634: BF9F0000
	s_code_end                                                 // 000000007638: BF9F0000
	s_code_end                                                 // 00000000763C: BF9F0000
	s_code_end                                                 // 000000007640: BF9F0000
	s_code_end                                                 // 000000007644: BF9F0000
	s_code_end                                                 // 000000007648: BF9F0000
	s_code_end                                                 // 00000000764C: BF9F0000
	s_code_end                                                 // 000000007650: BF9F0000
	s_code_end                                                 // 000000007654: BF9F0000
	s_code_end                                                 // 000000007658: BF9F0000
	s_code_end                                                 // 00000000765C: BF9F0000
	s_code_end                                                 // 000000007660: BF9F0000
	s_code_end                                                 // 000000007664: BF9F0000
	s_code_end                                                 // 000000007668: BF9F0000
	s_code_end                                                 // 00000000766C: BF9F0000
	s_code_end                                                 // 000000007670: BF9F0000
	s_code_end                                                 // 000000007674: BF9F0000
	s_code_end                                                 // 000000007678: BF9F0000
	s_code_end                                                 // 00000000767C: BF9F0000
	s_code_end                                                 // 000000007680: BF9F0000
	s_code_end                                                 // 000000007684: BF9F0000
	s_code_end                                                 // 000000007688: BF9F0000
	s_code_end                                                 // 00000000768C: BF9F0000
	s_code_end                                                 // 000000007690: BF9F0000
	s_code_end                                                 // 000000007694: BF9F0000
	s_code_end                                                 // 000000007698: BF9F0000
	s_code_end                                                 // 00000000769C: BF9F0000
	s_code_end                                                 // 0000000076A0: BF9F0000
	s_code_end                                                 // 0000000076A4: BF9F0000
	s_code_end                                                 // 0000000076A8: BF9F0000
	s_code_end                                                 // 0000000076AC: BF9F0000
	s_code_end                                                 // 0000000076B0: BF9F0000
	s_code_end                                                 // 0000000076B4: BF9F0000
	s_code_end                                                 // 0000000076B8: BF9F0000
	s_code_end                                                 // 0000000076BC: BF9F0000
	s_code_end                                                 // 0000000076C0: BF9F0000
	s_code_end                                                 // 0000000076C4: BF9F0000
	s_code_end                                                 // 0000000076C8: BF9F0000
	s_code_end                                                 // 0000000076CC: BF9F0000
	s_code_end                                                 // 0000000076D0: BF9F0000
	s_code_end                                                 // 0000000076D4: BF9F0000
	s_code_end                                                 // 0000000076D8: BF9F0000
	s_code_end                                                 // 0000000076DC: BF9F0000
	s_code_end                                                 // 0000000076E0: BF9F0000
	s_code_end                                                 // 0000000076E4: BF9F0000
	s_code_end                                                 // 0000000076E8: BF9F0000
	s_code_end                                                 // 0000000076EC: BF9F0000
	s_code_end                                                 // 0000000076F0: BF9F0000
	s_code_end                                                 // 0000000076F4: BF9F0000
	s_code_end                                                 // 0000000076F8: BF9F0000
	s_code_end                                                 // 0000000076FC: BF9F0000

Disassembly of section :

0000000000009000 <>:
; _DYNAMIC():
	global_wb                                                  // 000000009000: EE0B007C 00000000 00000000
	v_nop                                                      // 00000000900C: 7E000000
	s_get_pc_i64 s[26:27]                                      // 000000009010: BE9A4700
	s_add_co_u32 s26, s26, 0xffffc4ec                          // 000000009014: 801AFF1A FFFFC4EC
	s_add_co_ci_u32 s27, s27, -1                               // 00000000901C: 821BC11B
	s_set_pc_i64 s[26:27]                                      // 000000009020: BE80481A
	s_code_end                                                 // 000000009024: BF9F0000
	s_code_end                                                 // 000000009028: BF9F0000
	s_code_end                                                 // 00000000902C: BF9F0000
	s_code_end                                                 // 000000009030: BF9F0000
	s_code_end                                                 // 000000009034: BF9F0000
	s_code_end                                                 // 000000009038: BF9F0000
	s_code_end                                                 // 00000000903C: BF9F0000
	s_code_end                                                 // 000000009040: BF9F0000
	s_code_end                                                 // 000000009044: BF9F0000
	s_code_end                                                 // 000000009048: BF9F0000
	s_code_end                                                 // 00000000904C: BF9F0000
	s_code_end                                                 // 000000009050: BF9F0000
	s_code_end                                                 // 000000009054: BF9F0000
	s_code_end                                                 // 000000009058: BF9F0000
	s_code_end                                                 // 00000000905C: BF9F0000
	s_code_end                                                 // 000000009060: BF9F0000
	s_code_end                                                 // 000000009064: BF9F0000
	s_code_end                                                 // 000000009068: BF9F0000
	s_code_end                                                 // 00000000906C: BF9F0000
	s_code_end                                                 // 000000009070: BF9F0000
	s_code_end                                                 // 000000009074: BF9F0000
	s_code_end                                                 // 000000009078: BF9F0000
	s_code_end                                                 // 00000000907C: BF9F0000
	s_code_end                                                 // 000000009080: BF9F0000
	s_code_end                                                 // 000000009084: BF9F0000
	s_code_end                                                 // 000000009088: BF9F0000
	s_code_end                                                 // 00000000908C: BF9F0000
	s_code_end                                                 // 000000009090: BF9F0000
	s_code_end                                                 // 000000009094: BF9F0000
	s_code_end                                                 // 000000009098: BF9F0000
	s_code_end                                                 // 00000000909C: BF9F0000
	s_code_end                                                 // 0000000090A0: BF9F0000
	s_code_end                                                 // 0000000090A4: BF9F0000
	s_code_end                                                 // 0000000090A8: BF9F0000
	s_code_end                                                 // 0000000090AC: BF9F0000
	s_code_end                                                 // 0000000090B0: BF9F0000
	s_code_end                                                 // 0000000090B4: BF9F0000
	s_code_end                                                 // 0000000090B8: BF9F0000
	s_code_end                                                 // 0000000090BC: BF9F0000
	s_code_end                                                 // 0000000090C0: BF9F0000
	s_code_end                                                 // 0000000090C4: BF9F0000
	s_code_end                                                 // 0000000090C8: BF9F0000
	s_code_end                                                 // 0000000090CC: BF9F0000
	s_code_end                                                 // 0000000090D0: BF9F0000
	s_code_end                                                 // 0000000090D4: BF9F0000
	s_code_end                                                 // 0000000090D8: BF9F0000
	s_code_end                                                 // 0000000090DC: BF9F0000
	s_code_end                                                 // 0000000090E0: BF9F0000
	s_code_end                                                 // 0000000090E4: BF9F0000
	s_code_end                                                 // 0000000090E8: BF9F0000
	s_code_end                                                 // 0000000090EC: BF9F0000
	s_code_end                                                 // 0000000090F0: BF9F0000
	s_code_end                                                 // 0000000090F4: BF9F0000
	s_code_end                                                 // 0000000090F8: BF9F0000
	s_code_end                                                 // 0000000090FC: BF9F0000
	global_wb                                                  // 000000009100: EE0B007C 00000000 00000000
	v_nop                                                      // 00000000910C: 7E000000
	s_get_pc_i64 s[8:9]                                        // 000000009110: BE884700
	s_add_co_u32 s8, s8, 0xffffdbec                            // 000000009114: 8008FF08 FFFFDBEC
	s_add_co_ci_u32 s9, s9, -1                                 // 00000000911C: 8209C109
	s_set_pc_i64 s[8:9]                                        // 000000009120: BE804808
	s_code_end                                                 // 000000009124: BF9F0000
	s_code_end                                                 // 000000009128: BF9F0000
	s_code_end                                                 // 00000000912C: BF9F0000
	s_code_end                                                 // 000000009130: BF9F0000
	s_code_end                                                 // 000000009134: BF9F0000
	s_code_end                                                 // 000000009138: BF9F0000
	s_code_end                                                 // 00000000913C: BF9F0000
	s_code_end                                                 // 000000009140: BF9F0000
	s_code_end                                                 // 000000009144: BF9F0000
	s_code_end                                                 // 000000009148: BF9F0000
	s_code_end                                                 // 00000000914C: BF9F0000
	s_code_end                                                 // 000000009150: BF9F0000
	s_code_end                                                 // 000000009154: BF9F0000
	s_code_end                                                 // 000000009158: BF9F0000
	s_code_end                                                 // 00000000915C: BF9F0000
	s_code_end                                                 // 000000009160: BF9F0000
	s_code_end                                                 // 000000009164: BF9F0000
	s_code_end                                                 // 000000009168: BF9F0000
	s_code_end                                                 // 00000000916C: BF9F0000
	s_code_end                                                 // 000000009170: BF9F0000
	s_code_end                                                 // 000000009174: BF9F0000
	s_code_end                                                 // 000000009178: BF9F0000
	s_code_end                                                 // 00000000917C: BF9F0000
	s_code_end                                                 // 000000009180: BF9F0000
	s_code_end                                                 // 000000009184: BF9F0000
	s_code_end                                                 // 000000009188: BF9F0000
	s_code_end                                                 // 00000000918C: BF9F0000
	s_code_end                                                 // 000000009190: BF9F0000
	s_code_end                                                 // 000000009194: BF9F0000
	s_code_end                                                 // 000000009198: BF9F0000
	s_code_end                                                 // 00000000919C: BF9F0000
	s_code_end                                                 // 0000000091A0: BF9F0000
	s_code_end                                                 // 0000000091A4: BF9F0000
	s_code_end                                                 // 0000000091A8: BF9F0000
	s_code_end                                                 // 0000000091AC: BF9F0000
	s_code_end                                                 // 0000000091B0: BF9F0000
	s_code_end                                                 // 0000000091B4: BF9F0000
	s_code_end                                                 // 0000000091B8: BF9F0000
	s_code_end                                                 // 0000000091BC: BF9F0000
	s_code_end                                                 // 0000000091C0: BF9F0000
	s_code_end                                                 // 0000000091C4: BF9F0000
	s_code_end                                                 // 0000000091C8: BF9F0000
	s_code_end                                                 // 0000000091CC: BF9F0000
	s_code_end                                                 // 0000000091D0: BF9F0000
	s_code_end                                                 // 0000000091D4: BF9F0000
	s_code_end                                                 // 0000000091D8: BF9F0000
	s_code_end                                                 // 0000000091DC: BF9F0000
	s_code_end                                                 // 0000000091E0: BF9F0000
	s_code_end                                                 // 0000000091E4: BF9F0000
	s_code_end                                                 // 0000000091E8: BF9F0000
	s_code_end                                                 // 0000000091EC: BF9F0000
	s_code_end                                                 // 0000000091F0: BF9F0000
	s_code_end                                                 // 0000000091F4: BF9F0000
	s_code_end                                                 // 0000000091F8: BF9F0000
	s_code_end                                                 // 0000000091FC: BF9F0000
	global_wb                                                  // 000000009200: EE0B007C 00000000 00000000
	v_nop                                                      // 00000000920C: 7E000000
	s_get_pc_i64 s[8:9]                                        // 000000009210: BE884700
	s_add_co_u32 s8, s8, 0xffffdcec                            // 000000009214: 8008FF08 FFFFDCEC
	s_add_co_ci_u32 s9, s9, -1                                 // 00000000921C: 8209C109
	s_set_pc_i64 s[8:9]                                        // 000000009220: BE804808
	s_code_end                                                 // 000000009224: BF9F0000
	s_code_end                                                 // 000000009228: BF9F0000
	s_code_end                                                 // 00000000922C: BF9F0000
	s_code_end                                                 // 000000009230: BF9F0000
	s_code_end                                                 // 000000009234: BF9F0000
	s_code_end                                                 // 000000009238: BF9F0000
	s_code_end                                                 // 00000000923C: BF9F0000
	s_code_end                                                 // 000000009240: BF9F0000
	s_code_end                                                 // 000000009244: BF9F0000
	s_code_end                                                 // 000000009248: BF9F0000
	s_code_end                                                 // 00000000924C: BF9F0000
	s_code_end                                                 // 000000009250: BF9F0000
	s_code_end                                                 // 000000009254: BF9F0000
	s_code_end                                                 // 000000009258: BF9F0000
	s_code_end                                                 // 00000000925C: BF9F0000
	s_code_end                                                 // 000000009260: BF9F0000
	s_code_end                                                 // 000000009264: BF9F0000
	s_code_end                                                 // 000000009268: BF9F0000
	s_code_end                                                 // 00000000926C: BF9F0000
	s_code_end                                                 // 000000009270: BF9F0000
	s_code_end                                                 // 000000009274: BF9F0000
	s_code_end                                                 // 000000009278: BF9F0000
	s_code_end                                                 // 00000000927C: BF9F0000
	s_code_end                                                 // 000000009280: BF9F0000
	s_code_end                                                 // 000000009284: BF9F0000
	s_code_end                                                 // 000000009288: BF9F0000
	s_code_end                                                 // 00000000928C: BF9F0000
	s_code_end                                                 // 000000009290: BF9F0000
	s_code_end                                                 // 000000009294: BF9F0000
	s_code_end                                                 // 000000009298: BF9F0000
	s_code_end                                                 // 00000000929C: BF9F0000
	s_code_end                                                 // 0000000092A0: BF9F0000
	s_code_end                                                 // 0000000092A4: BF9F0000
	s_code_end                                                 // 0000000092A8: BF9F0000
	s_code_end                                                 // 0000000092AC: BF9F0000
	s_code_end                                                 // 0000000092B0: BF9F0000
	s_code_end                                                 // 0000000092B4: BF9F0000
	s_code_end                                                 // 0000000092B8: BF9F0000
	s_code_end                                                 // 0000000092BC: BF9F0000
	s_code_end                                                 // 0000000092C0: BF9F0000
	s_code_end                                                 // 0000000092C4: BF9F0000
	s_code_end                                                 // 0000000092C8: BF9F0000
	s_code_end                                                 // 0000000092CC: BF9F0000
	s_code_end                                                 // 0000000092D0: BF9F0000
	s_code_end                                                 // 0000000092D4: BF9F0000
	s_code_end                                                 // 0000000092D8: BF9F0000
	s_code_end                                                 // 0000000092DC: BF9F0000
	s_code_end                                                 // 0000000092E0: BF9F0000
	s_code_end                                                 // 0000000092E4: BF9F0000
	s_code_end                                                 // 0000000092E8: BF9F0000
	s_code_end                                                 // 0000000092EC: BF9F0000
	s_code_end                                                 // 0000000092F0: BF9F0000
	s_code_end                                                 // 0000000092F4: BF9F0000
	s_code_end                                                 // 0000000092F8: BF9F0000
	s_code_end                                                 // 0000000092FC: BF9F0000
	global_wb                                                  // 000000009300: EE0B007C 00000000 00000000
	v_nop                                                      // 00000000930C: 7E000000
	s_get_pc_i64 s[42:43]                                      // 000000009310: BEAA4700
	s_add_co_u32 s42, s42, 0xffffcdec                          // 000000009314: 802AFF2A FFFFCDEC
	s_add_co_ci_u32 s43, s43, -1                               // 00000000931C: 822BC12B
	s_set_pc_i64 s[42:43]                                      // 000000009320: BE80482A
	s_code_end                                                 // 000000009324: BF9F0000
	s_code_end                                                 // 000000009328: BF9F0000
	s_code_end                                                 // 00000000932C: BF9F0000
	s_code_end                                                 // 000000009330: BF9F0000
	s_code_end                                                 // 000000009334: BF9F0000
	s_code_end                                                 // 000000009338: BF9F0000
	s_code_end                                                 // 00000000933C: BF9F0000
	s_code_end                                                 // 000000009340: BF9F0000
	s_code_end                                                 // 000000009344: BF9F0000
	s_code_end                                                 // 000000009348: BF9F0000
	s_code_end                                                 // 00000000934C: BF9F0000
	s_code_end                                                 // 000000009350: BF9F0000
	s_code_end                                                 // 000000009354: BF9F0000
	s_code_end                                                 // 000000009358: BF9F0000
	s_code_end                                                 // 00000000935C: BF9F0000
	s_code_end                                                 // 000000009360: BF9F0000
	s_code_end                                                 // 000000009364: BF9F0000
	s_code_end                                                 // 000000009368: BF9F0000
	s_code_end                                                 // 00000000936C: BF9F0000
	s_code_end                                                 // 000000009370: BF9F0000
	s_code_end                                                 // 000000009374: BF9F0000
	s_code_end                                                 // 000000009378: BF9F0000
	s_code_end                                                 // 00000000937C: BF9F0000
	s_code_end                                                 // 000000009380: BF9F0000
	s_code_end                                                 // 000000009384: BF9F0000
	s_code_end                                                 // 000000009388: BF9F0000
	s_code_end                                                 // 00000000938C: BF9F0000
	s_code_end                                                 // 000000009390: BF9F0000
	s_code_end                                                 // 000000009394: BF9F0000
	s_code_end                                                 // 000000009398: BF9F0000
	s_code_end                                                 // 00000000939C: BF9F0000
	s_code_end                                                 // 0000000093A0: BF9F0000
	s_code_end                                                 // 0000000093A4: BF9F0000
	s_code_end                                                 // 0000000093A8: BF9F0000
	s_code_end                                                 // 0000000093AC: BF9F0000
	s_code_end                                                 // 0000000093B0: BF9F0000
	s_code_end                                                 // 0000000093B4: BF9F0000
	s_code_end                                                 // 0000000093B8: BF9F0000
	s_code_end                                                 // 0000000093BC: BF9F0000
	s_code_end                                                 // 0000000093C0: BF9F0000
	s_code_end                                                 // 0000000093C4: BF9F0000
	s_code_end                                                 // 0000000093C8: BF9F0000
	s_code_end                                                 // 0000000093CC: BF9F0000
	s_code_end                                                 // 0000000093D0: BF9F0000
	s_code_end                                                 // 0000000093D4: BF9F0000
	s_code_end                                                 // 0000000093D8: BF9F0000
	s_code_end                                                 // 0000000093DC: BF9F0000
	s_code_end                                                 // 0000000093E0: BF9F0000
	s_code_end                                                 // 0000000093E4: BF9F0000
	s_code_end                                                 // 0000000093E8: BF9F0000
	s_code_end                                                 // 0000000093EC: BF9F0000
	s_code_end                                                 // 0000000093F0: BF9F0000
	s_code_end                                                 // 0000000093F4: BF9F0000
	s_code_end                                                 // 0000000093F8: BF9F0000
	s_code_end                                                 // 0000000093FC: BF9F0000
	global_wb                                                  // 000000009400: EE0B007C 00000000 00000000
	v_nop                                                      // 00000000940C: 7E000000
	s_get_pc_i64 s[12:13]                                      // 000000009410: BE8C4700
	s_add_co_u32 s12, s12, 0xffffd9ec                          // 000000009414: 800CFF0C FFFFD9EC
	s_add_co_ci_u32 s13, s13, -1                               // 00000000941C: 820DC10D
	s_set_pc_i64 s[12:13]                                      // 000000009420: BE80480C
	s_code_end                                                 // 000000009424: BF9F0000
	s_code_end                                                 // 000000009428: BF9F0000
	s_code_end                                                 // 00000000942C: BF9F0000
	s_code_end                                                 // 000000009430: BF9F0000
	s_code_end                                                 // 000000009434: BF9F0000
	s_code_end                                                 // 000000009438: BF9F0000
	s_code_end                                                 // 00000000943C: BF9F0000
	s_code_end                                                 // 000000009440: BF9F0000
	s_code_end                                                 // 000000009444: BF9F0000
	s_code_end                                                 // 000000009448: BF9F0000
	s_code_end                                                 // 00000000944C: BF9F0000
	s_code_end                                                 // 000000009450: BF9F0000
	s_code_end                                                 // 000000009454: BF9F0000
	s_code_end                                                 // 000000009458: BF9F0000
	s_code_end                                                 // 00000000945C: BF9F0000
	s_code_end                                                 // 000000009460: BF9F0000
	s_code_end                                                 // 000000009464: BF9F0000
	s_code_end                                                 // 000000009468: BF9F0000
	s_code_end                                                 // 00000000946C: BF9F0000
	s_code_end                                                 // 000000009470: BF9F0000
	s_code_end                                                 // 000000009474: BF9F0000
	s_code_end                                                 // 000000009478: BF9F0000
	s_code_end                                                 // 00000000947C: BF9F0000
	s_code_end                                                 // 000000009480: BF9F0000
	s_code_end                                                 // 000000009484: BF9F0000
	s_code_end                                                 // 000000009488: BF9F0000
	s_code_end                                                 // 00000000948C: BF9F0000
	s_code_end                                                 // 000000009490: BF9F0000
	s_code_end                                                 // 000000009494: BF9F0000
	s_code_end                                                 // 000000009498: BF9F0000
	s_code_end                                                 // 00000000949C: BF9F0000
	s_code_end                                                 // 0000000094A0: BF9F0000
	s_code_end                                                 // 0000000094A4: BF9F0000
	s_code_end                                                 // 0000000094A8: BF9F0000
	s_code_end                                                 // 0000000094AC: BF9F0000
	s_code_end                                                 // 0000000094B0: BF9F0000
	s_code_end                                                 // 0000000094B4: BF9F0000
	s_code_end                                                 // 0000000094B8: BF9F0000
	s_code_end                                                 // 0000000094BC: BF9F0000
	s_code_end                                                 // 0000000094C0: BF9F0000
	s_code_end                                                 // 0000000094C4: BF9F0000
	s_code_end                                                 // 0000000094C8: BF9F0000
	s_code_end                                                 // 0000000094CC: BF9F0000
	s_code_end                                                 // 0000000094D0: BF9F0000
	s_code_end                                                 // 0000000094D4: BF9F0000
	s_code_end                                                 // 0000000094D8: BF9F0000
	s_code_end                                                 // 0000000094DC: BF9F0000
	s_code_end                                                 // 0000000094E0: BF9F0000
	s_code_end                                                 // 0000000094E4: BF9F0000
	s_code_end                                                 // 0000000094E8: BF9F0000
	s_code_end                                                 // 0000000094EC: BF9F0000
	s_code_end                                                 // 0000000094F0: BF9F0000
	s_code_end                                                 // 0000000094F4: BF9F0000
	s_code_end                                                 // 0000000094F8: BF9F0000
	s_code_end                                                 // 0000000094FC: BF9F0000
	global_wb                                                  // 000000009500: EE0B007C 00000000 00000000
	v_nop                                                      // 00000000950C: 7E000000
	s_get_pc_i64 s[16:17]                                      // 000000009510: BE904700
	s_add_co_u32 s16, s16, 0xffffdaec                          // 000000009514: 8010FF10 FFFFDAEC
	s_add_co_ci_u32 s17, s17, -1                               // 00000000951C: 8211C111
	s_set_pc_i64 s[16:17]                                      // 000000009520: BE804810
	s_code_end                                                 // 000000009524: BF9F0000
	s_code_end                                                 // 000000009528: BF9F0000
	s_code_end                                                 // 00000000952C: BF9F0000
	s_code_end                                                 // 000000009530: BF9F0000
	s_code_end                                                 // 000000009534: BF9F0000
	s_code_end                                                 // 000000009538: BF9F0000
	s_code_end                                                 // 00000000953C: BF9F0000
	s_code_end                                                 // 000000009540: BF9F0000
	s_code_end                                                 // 000000009544: BF9F0000
	s_code_end                                                 // 000000009548: BF9F0000
	s_code_end                                                 // 00000000954C: BF9F0000
	s_code_end                                                 // 000000009550: BF9F0000
	s_code_end                                                 // 000000009554: BF9F0000
	s_code_end                                                 // 000000009558: BF9F0000
	s_code_end                                                 // 00000000955C: BF9F0000
	s_code_end                                                 // 000000009560: BF9F0000
	s_code_end                                                 // 000000009564: BF9F0000
	s_code_end                                                 // 000000009568: BF9F0000
	s_code_end                                                 // 00000000956C: BF9F0000
	s_code_end                                                 // 000000009570: BF9F0000
	s_code_end                                                 // 000000009574: BF9F0000
	s_code_end                                                 // 000000009578: BF9F0000
	s_code_end                                                 // 00000000957C: BF9F0000
	s_code_end                                                 // 000000009580: BF9F0000
	s_code_end                                                 // 000000009584: BF9F0000
	s_code_end                                                 // 000000009588: BF9F0000
	s_code_end                                                 // 00000000958C: BF9F0000
	s_code_end                                                 // 000000009590: BF9F0000
	s_code_end                                                 // 000000009594: BF9F0000
	s_code_end                                                 // 000000009598: BF9F0000
	s_code_end                                                 // 00000000959C: BF9F0000
	s_code_end                                                 // 0000000095A0: BF9F0000
	s_code_end                                                 // 0000000095A4: BF9F0000
	s_code_end                                                 // 0000000095A8: BF9F0000
	s_code_end                                                 // 0000000095AC: BF9F0000
	s_code_end                                                 // 0000000095B0: BF9F0000
	s_code_end                                                 // 0000000095B4: BF9F0000
	s_code_end                                                 // 0000000095B8: BF9F0000
	s_code_end                                                 // 0000000095BC: BF9F0000
	s_code_end                                                 // 0000000095C0: BF9F0000
	s_code_end                                                 // 0000000095C4: BF9F0000
	s_code_end                                                 // 0000000095C8: BF9F0000
	s_code_end                                                 // 0000000095CC: BF9F0000
	s_code_end                                                 // 0000000095D0: BF9F0000
	s_code_end                                                 // 0000000095D4: BF9F0000
	s_code_end                                                 // 0000000095D8: BF9F0000
	s_code_end                                                 // 0000000095DC: BF9F0000
	s_code_end                                                 // 0000000095E0: BF9F0000
	s_code_end                                                 // 0000000095E4: BF9F0000
	s_code_end                                                 // 0000000095E8: BF9F0000
	s_code_end                                                 // 0000000095EC: BF9F0000
	s_code_end                                                 // 0000000095F0: BF9F0000
	s_code_end                                                 // 0000000095F4: BF9F0000
	s_code_end                                                 // 0000000095F8: BF9F0000
	s_code_end                                                 // 0000000095FC: BF9F0000
	global_wb                                                  // 000000009600: EE0B007C 00000000 00000000
	v_nop                                                      // 00000000960C: 7E000000
	s_get_pc_i64 s[30:31]                                      // 000000009610: BE9E4700
	s_add_co_u32 s30, s30, 0xffffc2ec                          // 000000009614: 801EFF1E FFFFC2EC
	s_add_co_ci_u32 s31, s31, -1                               // 00000000961C: 821FC11F
	s_set_pc_i64 s[30:31]                                      // 000000009620: BE80481E
	s_code_end                                                 // 000000009624: BF9F0000
	s_code_end                                                 // 000000009628: BF9F0000
	s_code_end                                                 // 00000000962C: BF9F0000
	s_code_end                                                 // 000000009630: BF9F0000
	s_code_end                                                 // 000000009634: BF9F0000
	s_code_end                                                 // 000000009638: BF9F0000
	s_code_end                                                 // 00000000963C: BF9F0000
	s_code_end                                                 // 000000009640: BF9F0000
	s_code_end                                                 // 000000009644: BF9F0000
	s_code_end                                                 // 000000009648: BF9F0000
	s_code_end                                                 // 00000000964C: BF9F0000
	s_code_end                                                 // 000000009650: BF9F0000
	s_code_end                                                 // 000000009654: BF9F0000
	s_code_end                                                 // 000000009658: BF9F0000
	s_code_end                                                 // 00000000965C: BF9F0000
	s_code_end                                                 // 000000009660: BF9F0000
	s_code_end                                                 // 000000009664: BF9F0000
	s_code_end                                                 // 000000009668: BF9F0000
	s_code_end                                                 // 00000000966C: BF9F0000
	s_code_end                                                 // 000000009670: BF9F0000
	s_code_end                                                 // 000000009674: BF9F0000
	s_code_end                                                 // 000000009678: BF9F0000
	s_code_end                                                 // 00000000967C: BF9F0000
	s_code_end                                                 // 000000009680: BF9F0000
	s_code_end                                                 // 000000009684: BF9F0000
	s_code_end                                                 // 000000009688: BF9F0000
	s_code_end                                                 // 00000000968C: BF9F0000
	s_code_end                                                 // 000000009690: BF9F0000
	s_code_end                                                 // 000000009694: BF9F0000
	s_code_end                                                 // 000000009698: BF9F0000
	s_code_end                                                 // 00000000969C: BF9F0000
	s_code_end                                                 // 0000000096A0: BF9F0000
	s_code_end                                                 // 0000000096A4: BF9F0000
	s_code_end                                                 // 0000000096A8: BF9F0000
	s_code_end                                                 // 0000000096AC: BF9F0000
	s_code_end                                                 // 0000000096B0: BF9F0000
	s_code_end                                                 // 0000000096B4: BF9F0000
	s_code_end                                                 // 0000000096B8: BF9F0000
	s_code_end                                                 // 0000000096BC: BF9F0000
	s_code_end                                                 // 0000000096C0: BF9F0000
	s_code_end                                                 // 0000000096C4: BF9F0000
	s_code_end                                                 // 0000000096C8: BF9F0000
	s_code_end                                                 // 0000000096CC: BF9F0000
	s_code_end                                                 // 0000000096D0: BF9F0000
	s_code_end                                                 // 0000000096D4: BF9F0000
	s_code_end                                                 // 0000000096D8: BF9F0000
	s_code_end                                                 // 0000000096DC: BF9F0000
	s_code_end                                                 // 0000000096E0: BF9F0000
	s_code_end                                                 // 0000000096E4: BF9F0000
	s_code_end                                                 // 0000000096E8: BF9F0000
	s_code_end                                                 // 0000000096EC: BF9F0000
	s_code_end                                                 // 0000000096F0: BF9F0000
	s_code_end                                                 // 0000000096F4: BF9F0000
	s_code_end                                                 // 0000000096F8: BF9F0000
	s_code_end                                                 // 0000000096FC: BF9F0000
	global_wb                                                  // 000000009700: EE0B007C 00000000 00000000
	v_nop                                                      // 00000000970C: 7E000000
	s_get_pc_i64 s[18:19]                                      // 000000009710: BE924700
	s_add_co_u32 s18, s18, 0xffffc4ec                          // 000000009714: 8012FF12 FFFFC4EC
	s_add_co_ci_u32 s19, s19, -1                               // 00000000971C: 8213C113
	s_set_pc_i64 s[18:19]                                      // 000000009720: BE804812
	s_code_end                                                 // 000000009724: BF9F0000
	s_code_end                                                 // 000000009728: BF9F0000
	s_code_end                                                 // 00000000972C: BF9F0000
	s_code_end                                                 // 000000009730: BF9F0000
	s_code_end                                                 // 000000009734: BF9F0000
	s_code_end                                                 // 000000009738: BF9F0000
	s_code_end                                                 // 00000000973C: BF9F0000
	s_code_end                                                 // 000000009740: BF9F0000
	s_code_end                                                 // 000000009744: BF9F0000
	s_code_end                                                 // 000000009748: BF9F0000
	s_code_end                                                 // 00000000974C: BF9F0000
	s_code_end                                                 // 000000009750: BF9F0000
	s_code_end                                                 // 000000009754: BF9F0000
	s_code_end                                                 // 000000009758: BF9F0000
	s_code_end                                                 // 00000000975C: BF9F0000
	s_code_end                                                 // 000000009760: BF9F0000
	s_code_end                                                 // 000000009764: BF9F0000
	s_code_end                                                 // 000000009768: BF9F0000
	s_code_end                                                 // 00000000976C: BF9F0000
	s_code_end                                                 // 000000009770: BF9F0000
	s_code_end                                                 // 000000009774: BF9F0000
	s_code_end                                                 // 000000009778: BF9F0000
	s_code_end                                                 // 00000000977C: BF9F0000
	s_code_end                                                 // 000000009780: BF9F0000
	s_code_end                                                 // 000000009784: BF9F0000
	s_code_end                                                 // 000000009788: BF9F0000
	s_code_end                                                 // 00000000978C: BF9F0000
	s_code_end                                                 // 000000009790: BF9F0000
	s_code_end                                                 // 000000009794: BF9F0000
	s_code_end                                                 // 000000009798: BF9F0000
	s_code_end                                                 // 00000000979C: BF9F0000
	s_code_end                                                 // 0000000097A0: BF9F0000
	s_code_end                                                 // 0000000097A4: BF9F0000
	s_code_end                                                 // 0000000097A8: BF9F0000
	s_code_end                                                 // 0000000097AC: BF9F0000
	s_code_end                                                 // 0000000097B0: BF9F0000
	s_code_end                                                 // 0000000097B4: BF9F0000
	s_code_end                                                 // 0000000097B8: BF9F0000
	s_code_end                                                 // 0000000097BC: BF9F0000
	s_code_end                                                 // 0000000097C0: BF9F0000
	s_code_end                                                 // 0000000097C4: BF9F0000
	s_code_end                                                 // 0000000097C8: BF9F0000
	s_code_end                                                 // 0000000097CC: BF9F0000
	s_code_end                                                 // 0000000097D0: BF9F0000
	s_code_end                                                 // 0000000097D4: BF9F0000
	s_code_end                                                 // 0000000097D8: BF9F0000
	s_code_end                                                 // 0000000097DC: BF9F0000
	s_code_end                                                 // 0000000097E0: BF9F0000
	s_code_end                                                 // 0000000097E4: BF9F0000
	s_code_end                                                 // 0000000097E8: BF9F0000
	s_code_end                                                 // 0000000097EC: BF9F0000
	s_code_end                                                 // 0000000097F0: BF9F0000
	s_code_end                                                 // 0000000097F4: BF9F0000
	s_code_end                                                 // 0000000097F8: BF9F0000
	s_code_end                                                 // 0000000097FC: BF9F0000
	global_wb                                                  // 000000009800: EE0B007C 00000000 00000000
	v_nop                                                      // 00000000980C: 7E000000
	s_get_pc_i64 s[18:19]                                      // 000000009810: BE924700
	s_add_co_u32 s18, s18, 0xffffc5ec                          // 000000009814: 8012FF12 FFFFC5EC
	s_add_co_ci_u32 s19, s19, -1                               // 00000000981C: 8213C113
	s_set_pc_i64 s[18:19]                                      // 000000009820: BE804812
	s_code_end                                                 // 000000009824: BF9F0000
	s_code_end                                                 // 000000009828: BF9F0000
	s_code_end                                                 // 00000000982C: BF9F0000
	s_code_end                                                 // 000000009830: BF9F0000
	s_code_end                                                 // 000000009834: BF9F0000
	s_code_end                                                 // 000000009838: BF9F0000
	s_code_end                                                 // 00000000983C: BF9F0000
	s_code_end                                                 // 000000009840: BF9F0000
	s_code_end                                                 // 000000009844: BF9F0000
	s_code_end                                                 // 000000009848: BF9F0000
	s_code_end                                                 // 00000000984C: BF9F0000
	s_code_end                                                 // 000000009850: BF9F0000
	s_code_end                                                 // 000000009854: BF9F0000
	s_code_end                                                 // 000000009858: BF9F0000
	s_code_end                                                 // 00000000985C: BF9F0000
	s_code_end                                                 // 000000009860: BF9F0000
	s_code_end                                                 // 000000009864: BF9F0000
	s_code_end                                                 // 000000009868: BF9F0000
	s_code_end                                                 // 00000000986C: BF9F0000
	s_code_end                                                 // 000000009870: BF9F0000
	s_code_end                                                 // 000000009874: BF9F0000
	s_code_end                                                 // 000000009878: BF9F0000
	s_code_end                                                 // 00000000987C: BF9F0000
	s_code_end                                                 // 000000009880: BF9F0000
	s_code_end                                                 // 000000009884: BF9F0000
	s_code_end                                                 // 000000009888: BF9F0000
	s_code_end                                                 // 00000000988C: BF9F0000
	s_code_end                                                 // 000000009890: BF9F0000
	s_code_end                                                 // 000000009894: BF9F0000
	s_code_end                                                 // 000000009898: BF9F0000
	s_code_end                                                 // 00000000989C: BF9F0000
	s_code_end                                                 // 0000000098A0: BF9F0000
	s_code_end                                                 // 0000000098A4: BF9F0000
	s_code_end                                                 // 0000000098A8: BF9F0000
	s_code_end                                                 // 0000000098AC: BF9F0000
	s_code_end                                                 // 0000000098B0: BF9F0000
	s_code_end                                                 // 0000000098B4: BF9F0000
	s_code_end                                                 // 0000000098B8: BF9F0000
	s_code_end                                                 // 0000000098BC: BF9F0000
	s_code_end                                                 // 0000000098C0: BF9F0000
	s_code_end                                                 // 0000000098C4: BF9F0000
	s_code_end                                                 // 0000000098C8: BF9F0000
	s_code_end                                                 // 0000000098CC: BF9F0000
	s_code_end                                                 // 0000000098D0: BF9F0000
	s_code_end                                                 // 0000000098D4: BF9F0000
	s_code_end                                                 // 0000000098D8: BF9F0000
	s_code_end                                                 // 0000000098DC: BF9F0000
	s_code_end                                                 // 0000000098E0: BF9F0000
	s_code_end                                                 // 0000000098E4: BF9F0000
	s_code_end                                                 // 0000000098E8: BF9F0000
	s_code_end                                                 // 0000000098EC: BF9F0000
	s_code_end                                                 // 0000000098F0: BF9F0000
	s_code_end                                                 // 0000000098F4: BF9F0000
	s_code_end                                                 // 0000000098F8: BF9F0000
	s_code_end                                                 // 0000000098FC: BF9F0000
	global_wb                                                  // 000000009900: EE0B007C 00000000 00000000
	v_nop                                                      // 00000000990C: 7E000000
	s_get_pc_i64 s[20:21]                                      // 000000009910: BE944700
	s_add_co_u32 s20, s20, 0xffffc6ec                          // 000000009914: 8014FF14 FFFFC6EC
	s_add_co_ci_u32 s21, s21, -1                               // 00000000991C: 8215C115
	s_set_pc_i64 s[20:21]                                      // 000000009920: BE804814
	s_code_end                                                 // 000000009924: BF9F0000
	s_code_end                                                 // 000000009928: BF9F0000
	s_code_end                                                 // 00000000992C: BF9F0000
	s_code_end                                                 // 000000009930: BF9F0000
	s_code_end                                                 // 000000009934: BF9F0000
	s_code_end                                                 // 000000009938: BF9F0000
	s_code_end                                                 // 00000000993C: BF9F0000
	s_code_end                                                 // 000000009940: BF9F0000
	s_code_end                                                 // 000000009944: BF9F0000
	s_code_end                                                 // 000000009948: BF9F0000
	s_code_end                                                 // 00000000994C: BF9F0000
	s_code_end                                                 // 000000009950: BF9F0000
	s_code_end                                                 // 000000009954: BF9F0000
	s_code_end                                                 // 000000009958: BF9F0000
	s_code_end                                                 // 00000000995C: BF9F0000
	s_code_end                                                 // 000000009960: BF9F0000
	s_code_end                                                 // 000000009964: BF9F0000
	s_code_end                                                 // 000000009968: BF9F0000
	s_code_end                                                 // 00000000996C: BF9F0000
	s_code_end                                                 // 000000009970: BF9F0000
	s_code_end                                                 // 000000009974: BF9F0000
	s_code_end                                                 // 000000009978: BF9F0000
	s_code_end                                                 // 00000000997C: BF9F0000
	s_code_end                                                 // 000000009980: BF9F0000
	s_code_end                                                 // 000000009984: BF9F0000
	s_code_end                                                 // 000000009988: BF9F0000
	s_code_end                                                 // 00000000998C: BF9F0000
	s_code_end                                                 // 000000009990: BF9F0000
	s_code_end                                                 // 000000009994: BF9F0000
	s_code_end                                                 // 000000009998: BF9F0000
	s_code_end                                                 // 00000000999C: BF9F0000
	s_code_end                                                 // 0000000099A0: BF9F0000
	s_code_end                                                 // 0000000099A4: BF9F0000
	s_code_end                                                 // 0000000099A8: BF9F0000
	s_code_end                                                 // 0000000099AC: BF9F0000
	s_code_end                                                 // 0000000099B0: BF9F0000
	s_code_end                                                 // 0000000099B4: BF9F0000
	s_code_end                                                 // 0000000099B8: BF9F0000
	s_code_end                                                 // 0000000099BC: BF9F0000
	s_code_end                                                 // 0000000099C0: BF9F0000
	s_code_end                                                 // 0000000099C4: BF9F0000
	s_code_end                                                 // 0000000099C8: BF9F0000
	s_code_end                                                 // 0000000099CC: BF9F0000
	s_code_end                                                 // 0000000099D0: BF9F0000
	s_code_end                                                 // 0000000099D4: BF9F0000
	s_code_end                                                 // 0000000099D8: BF9F0000
	s_code_end                                                 // 0000000099DC: BF9F0000
	s_code_end                                                 // 0000000099E0: BF9F0000
	s_code_end                                                 // 0000000099E4: BF9F0000
	s_code_end                                                 // 0000000099E8: BF9F0000
	s_code_end                                                 // 0000000099EC: BF9F0000
	s_code_end                                                 // 0000000099F0: BF9F0000
	s_code_end                                                 // 0000000099F4: BF9F0000
	s_code_end                                                 // 0000000099F8: BF9F0000
	s_code_end                                                 // 0000000099FC: BF9F0000
	global_wb                                                  // 000000009A00: EE0B007C 00000000 00000000
	v_nop                                                      // 000000009A0C: 7E000000
	s_get_pc_i64 s[42:43]                                      // 000000009A10: BEAA4700
	s_add_co_u32 s42, s42, 0xffffc8ec                          // 000000009A14: 802AFF2A FFFFC8EC
	s_add_co_ci_u32 s43, s43, -1                               // 000000009A1C: 822BC12B
	s_set_pc_i64 s[42:43]                                      // 000000009A20: BE80482A
	s_code_end                                                 // 000000009A24: BF9F0000
	s_code_end                                                 // 000000009A28: BF9F0000
	s_code_end                                                 // 000000009A2C: BF9F0000
	s_code_end                                                 // 000000009A30: BF9F0000
	s_code_end                                                 // 000000009A34: BF9F0000
	s_code_end                                                 // 000000009A38: BF9F0000
	s_code_end                                                 // 000000009A3C: BF9F0000
	s_code_end                                                 // 000000009A40: BF9F0000
	s_code_end                                                 // 000000009A44: BF9F0000
	s_code_end                                                 // 000000009A48: BF9F0000
	s_code_end                                                 // 000000009A4C: BF9F0000
	s_code_end                                                 // 000000009A50: BF9F0000
	s_code_end                                                 // 000000009A54: BF9F0000
	s_code_end                                                 // 000000009A58: BF9F0000
	s_code_end                                                 // 000000009A5C: BF9F0000
	s_code_end                                                 // 000000009A60: BF9F0000
	s_code_end                                                 // 000000009A64: BF9F0000
	s_code_end                                                 // 000000009A68: BF9F0000
	s_code_end                                                 // 000000009A6C: BF9F0000
	s_code_end                                                 // 000000009A70: BF9F0000
	s_code_end                                                 // 000000009A74: BF9F0000
	s_code_end                                                 // 000000009A78: BF9F0000
	s_code_end                                                 // 000000009A7C: BF9F0000
	s_code_end                                                 // 000000009A80: BF9F0000
	s_code_end                                                 // 000000009A84: BF9F0000
	s_code_end                                                 // 000000009A88: BF9F0000
	s_code_end                                                 // 000000009A8C: BF9F0000
	s_code_end                                                 // 000000009A90: BF9F0000
	s_code_end                                                 // 000000009A94: BF9F0000
	s_code_end                                                 // 000000009A98: BF9F0000
	s_code_end                                                 // 000000009A9C: BF9F0000
	s_code_end                                                 // 000000009AA0: BF9F0000
	s_code_end                                                 // 000000009AA4: BF9F0000
	s_code_end                                                 // 000000009AA8: BF9F0000
	s_code_end                                                 // 000000009AAC: BF9F0000
	s_code_end                                                 // 000000009AB0: BF9F0000
	s_code_end                                                 // 000000009AB4: BF9F0000
	s_code_end                                                 // 000000009AB8: BF9F0000
	s_code_end                                                 // 000000009ABC: BF9F0000
	s_code_end                                                 // 000000009AC0: BF9F0000
	s_code_end                                                 // 000000009AC4: BF9F0000
	s_code_end                                                 // 000000009AC8: BF9F0000
	s_code_end                                                 // 000000009ACC: BF9F0000
	s_code_end                                                 // 000000009AD0: BF9F0000
	s_code_end                                                 // 000000009AD4: BF9F0000
	s_code_end                                                 // 000000009AD8: BF9F0000
	s_code_end                                                 // 000000009ADC: BF9F0000
	s_code_end                                                 // 000000009AE0: BF9F0000
	s_code_end                                                 // 000000009AE4: BF9F0000
	s_code_end                                                 // 000000009AE8: BF9F0000
	s_code_end                                                 // 000000009AEC: BF9F0000
	s_code_end                                                 // 000000009AF0: BF9F0000
	s_code_end                                                 // 000000009AF4: BF9F0000
	s_code_end                                                 // 000000009AF8: BF9F0000
	s_code_end                                                 // 000000009AFC: BF9F0000
	global_wb                                                  // 000000009B00: EE0B007C 00000000 00000000
	v_nop                                                      // 000000009B0C: 7E000000
	s_get_pc_i64 s[12:13]                                      // 000000009B10: BE8C4700
	s_add_co_u32 s12, s12, 0xffffc9ec                          // 000000009B14: 800CFF0C FFFFC9EC
	s_add_co_ci_u32 s13, s13, -1                               // 000000009B1C: 820DC10D
	s_set_pc_i64 s[12:13]                                      // 000000009B20: BE80480C
	s_code_end                                                 // 000000009B24: BF9F0000
	s_code_end                                                 // 000000009B28: BF9F0000
	s_code_end                                                 // 000000009B2C: BF9F0000
	s_code_end                                                 // 000000009B30: BF9F0000
	s_code_end                                                 // 000000009B34: BF9F0000
	s_code_end                                                 // 000000009B38: BF9F0000
	s_code_end                                                 // 000000009B3C: BF9F0000
	s_code_end                                                 // 000000009B40: BF9F0000
	s_code_end                                                 // 000000009B44: BF9F0000
	s_code_end                                                 // 000000009B48: BF9F0000
	s_code_end                                                 // 000000009B4C: BF9F0000
	s_code_end                                                 // 000000009B50: BF9F0000
	s_code_end                                                 // 000000009B54: BF9F0000
	s_code_end                                                 // 000000009B58: BF9F0000
	s_code_end                                                 // 000000009B5C: BF9F0000
	s_code_end                                                 // 000000009B60: BF9F0000
	s_code_end                                                 // 000000009B64: BF9F0000
	s_code_end                                                 // 000000009B68: BF9F0000
	s_code_end                                                 // 000000009B6C: BF9F0000
	s_code_end                                                 // 000000009B70: BF9F0000
	s_code_end                                                 // 000000009B74: BF9F0000
	s_code_end                                                 // 000000009B78: BF9F0000
	s_code_end                                                 // 000000009B7C: BF9F0000
	s_code_end                                                 // 000000009B80: BF9F0000
	s_code_end                                                 // 000000009B84: BF9F0000
	s_code_end                                                 // 000000009B88: BF9F0000
	s_code_end                                                 // 000000009B8C: BF9F0000
	s_code_end                                                 // 000000009B90: BF9F0000
	s_code_end                                                 // 000000009B94: BF9F0000
	s_code_end                                                 // 000000009B98: BF9F0000
	s_code_end                                                 // 000000009B9C: BF9F0000
	s_code_end                                                 // 000000009BA0: BF9F0000
	s_code_end                                                 // 000000009BA4: BF9F0000
	s_code_end                                                 // 000000009BA8: BF9F0000
	s_code_end                                                 // 000000009BAC: BF9F0000
	s_code_end                                                 // 000000009BB0: BF9F0000
	s_code_end                                                 // 000000009BB4: BF9F0000
	s_code_end                                                 // 000000009BB8: BF9F0000
	s_code_end                                                 // 000000009BBC: BF9F0000
	s_code_end                                                 // 000000009BC0: BF9F0000
	s_code_end                                                 // 000000009BC4: BF9F0000
	s_code_end                                                 // 000000009BC8: BF9F0000
	s_code_end                                                 // 000000009BCC: BF9F0000
	s_code_end                                                 // 000000009BD0: BF9F0000
	s_code_end                                                 // 000000009BD4: BF9F0000
	s_code_end                                                 // 000000009BD8: BF9F0000
	s_code_end                                                 // 000000009BDC: BF9F0000
	s_code_end                                                 // 000000009BE0: BF9F0000
	s_code_end                                                 // 000000009BE4: BF9F0000
	s_code_end                                                 // 000000009BE8: BF9F0000
	s_code_end                                                 // 000000009BEC: BF9F0000
	s_code_end                                                 // 000000009BF0: BF9F0000
	s_code_end                                                 // 000000009BF4: BF9F0000
	s_code_end                                                 // 000000009BF8: BF9F0000
	s_code_end                                                 // 000000009BFC: BF9F0000
	global_wb                                                  // 000000009C00: EE0B007C 00000000 00000000
	v_nop                                                      // 000000009C0C: 7E000000
	s_get_pc_i64 s[12:13]                                      // 000000009C10: BE8C4700
	s_add_co_u32 s12, s12, 0xffffd6ec                          // 000000009C14: 800CFF0C FFFFD6EC
	s_add_co_ci_u32 s13, s13, -1                               // 000000009C1C: 820DC10D
	s_set_pc_i64 s[12:13]                                      // 000000009C20: BE80480C
	s_code_end                                                 // 000000009C24: BF9F0000
	s_code_end                                                 // 000000009C28: BF9F0000
	s_code_end                                                 // 000000009C2C: BF9F0000
	s_code_end                                                 // 000000009C30: BF9F0000
	s_code_end                                                 // 000000009C34: BF9F0000
	s_code_end                                                 // 000000009C38: BF9F0000
	s_code_end                                                 // 000000009C3C: BF9F0000
	s_code_end                                                 // 000000009C40: BF9F0000
	s_code_end                                                 // 000000009C44: BF9F0000
	s_code_end                                                 // 000000009C48: BF9F0000
	s_code_end                                                 // 000000009C4C: BF9F0000
	s_code_end                                                 // 000000009C50: BF9F0000
	s_code_end                                                 // 000000009C54: BF9F0000
	s_code_end                                                 // 000000009C58: BF9F0000
	s_code_end                                                 // 000000009C5C: BF9F0000
	s_code_end                                                 // 000000009C60: BF9F0000
	s_code_end                                                 // 000000009C64: BF9F0000
	s_code_end                                                 // 000000009C68: BF9F0000
	s_code_end                                                 // 000000009C6C: BF9F0000
	s_code_end                                                 // 000000009C70: BF9F0000
	s_code_end                                                 // 000000009C74: BF9F0000
	s_code_end                                                 // 000000009C78: BF9F0000
	s_code_end                                                 // 000000009C7C: BF9F0000
	s_code_end                                                 // 000000009C80: BF9F0000
	s_code_end                                                 // 000000009C84: BF9F0000
	s_code_end                                                 // 000000009C88: BF9F0000
	s_code_end                                                 // 000000009C8C: BF9F0000
	s_code_end                                                 // 000000009C90: BF9F0000
	s_code_end                                                 // 000000009C94: BF9F0000
	s_code_end                                                 // 000000009C98: BF9F0000
	s_code_end                                                 // 000000009C9C: BF9F0000
	s_code_end                                                 // 000000009CA0: BF9F0000
	s_code_end                                                 // 000000009CA4: BF9F0000
	s_code_end                                                 // 000000009CA8: BF9F0000
	s_code_end                                                 // 000000009CAC: BF9F0000
	s_code_end                                                 // 000000009CB0: BF9F0000
	s_code_end                                                 // 000000009CB4: BF9F0000
	s_code_end                                                 // 000000009CB8: BF9F0000
	s_code_end                                                 // 000000009CBC: BF9F0000
	s_code_end                                                 // 000000009CC0: BF9F0000
	s_code_end                                                 // 000000009CC4: BF9F0000
	s_code_end                                                 // 000000009CC8: BF9F0000
	s_code_end                                                 // 000000009CCC: BF9F0000
	s_code_end                                                 // 000000009CD0: BF9F0000
	s_code_end                                                 // 000000009CD4: BF9F0000
	s_code_end                                                 // 000000009CD8: BF9F0000
	s_code_end                                                 // 000000009CDC: BF9F0000
	s_code_end                                                 // 000000009CE0: BF9F0000
	s_code_end                                                 // 000000009CE4: BF9F0000
	s_code_end                                                 // 000000009CE8: BF9F0000
	s_code_end                                                 // 000000009CEC: BF9F0000
	s_code_end                                                 // 000000009CF0: BF9F0000
	s_code_end                                                 // 000000009CF4: BF9F0000
	s_code_end                                                 // 000000009CF8: BF9F0000
	s_code_end                                                 // 000000009CFC: BF9F0000
