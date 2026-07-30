
/tmp/tmp.9EIkB6moFK/gfx1250.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001b00 <_Z7kernel1ii>:
; _Z7kernel1ii():
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:66
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1       // 000000001B00: B9800641 00000001
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:69
	s_load_b64 s[0:1], s[0:1], 0x0 nv                          // 000000001B08: F4102000 F8000000
	s_wait_kmcnt 0x0                                           // 000000001B10: BFC70000
	s_cmp_lt_i32 s1, 1                                         // 000000001B14: BF048101
	s_cbranch_scc1 111                                         // 000000001B18: BFA2006F <_Z7kernel1ii+0x1d8>
	s_add_co_i32 s1, s1, -1                                    // 000000001B1C: 8101C101
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:71
	s_nop 1                                                    // 000000001B20: BF800001
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:69
	s_cmp_eq_u32 s1, 0                                         // 000000001B24: BF068001
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:71
	s_nop 1                                                    // 000000001B28: BF800001
	s_nop 1                                                    // 000000001B2C: BF800001
	s_nop 1                                                    // 000000001B30: BF800001
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:72
	v_mov_b32_e32 v0, s0                                       // 000000001B34: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:73
	v_mov_b32_e32 v0, s0                                       // 000000001B38: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:74
	v_mov_b32_e32 v0, s0                                       // 000000001B3C: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:75
	v_mov_b32_e32 v0, s0                                       // 000000001B40: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:76
	v_mov_b32_e32 v0, s0                                       // 000000001B44: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:77
	v_mov_b32_e32 v0, s0                                       // 000000001B48: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:78
	v_mov_b32_e32 v0, s0                                       // 000000001B4C: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:79
	v_mov_b32_e32 v0, s0                                       // 000000001B50: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:80
	v_mov_b32_e32 v0, s0                                       // 000000001B54: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:81
	v_mov_b32_e32 v0, s0                                       // 000000001B58: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:82
	v_mov_b32_e32 v0, s0                                       // 000000001B5C: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:83
	v_mov_b32_e32 v0, s0                                       // 000000001B60: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:84
	v_mov_b32_e32 v0, s0                                       // 000000001B64: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:85
	v_mov_b32_e32 v0, s0                                       // 000000001B68: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:86
	v_mov_b32_e32 v0, s0                                       // 000000001B6C: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:87
	v_mov_b32_e32 v0, s0                                       // 000000001B70: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:88
	v_mov_b32_e32 v0, s0                                       // 000000001B74: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:89
	v_mov_b32_e32 v0, s0                                       // 000000001B78: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:90
	v_mov_b32_e32 v0, s0                                       // 000000001B7C: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:91
	v_mov_b32_e32 v0, s0                                       // 000000001B80: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:92
	v_mov_b32_e32 v0, s0                                       // 000000001B84: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:93
	v_mov_b32_e32 v0, s0                                       // 000000001B88: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:94
	v_mov_b32_e32 v0, s0                                       // 000000001B8C: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:95
	v_mov_b32_e32 v0, s0                                       // 000000001B90: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:96
	v_mov_b32_e32 v0, s0                                       // 000000001B94: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:97
	v_mov_b32_e32 v0, s0                                       // 000000001B98: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:98
	v_mov_b32_e32 v0, s0                                       // 000000001B9C: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:99
	v_mov_b32_e32 v0, s0                                       // 000000001BA0: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:100
	v_mov_b32_e32 v0, s0                                       // 000000001BA4: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:101
	v_mov_b32_e32 v0, s0                                       // 000000001BA8: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:102
	v_mov_b32_e32 v0, s0                                       // 000000001BAC: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:103
	v_mov_b32_e32 v0, s0                                       // 000000001BB0: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:104
	v_mov_b32_e32 v0, s0                                       // 000000001BB4: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:105
	v_mov_b32_e32 v0, s0                                       // 000000001BB8: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:106
	v_mov_b32_e32 v0, s0                                       // 000000001BBC: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:107
	v_mov_b32_e32 v0, s0                                       // 000000001BC0: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:108
	v_mov_b32_e32 v0, s0                                       // 000000001BC4: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:109
	v_mov_b32_e32 v0, s0                                       // 000000001BC8: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:110
	v_mov_b32_e32 v0, s0                                       // 000000001BCC: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:111
	v_mov_b32_e32 v0, s0                                       // 000000001BD0: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:112
	v_mov_b32_e32 v0, s0                                       // 000000001BD4: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:113
	v_mov_b32_e32 v0, s0                                       // 000000001BD8: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:114
	v_mov_b32_e32 v0, s0                                       // 000000001BDC: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:115
	v_mov_b32_e32 v0, s0                                       // 000000001BE0: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:116
	v_mov_b32_e32 v0, s0                                       // 000000001BE4: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:117
	v_mov_b32_e32 v0, s0                                       // 000000001BE8: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:118
	v_mov_b32_e32 v0, s0                                       // 000000001BEC: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:119
	v_mov_b32_e32 v0, s0                                       // 000000001BF0: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:120
	v_mov_b32_e32 v0, s0                                       // 000000001BF4: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:121
	v_mov_b32_e32 v0, s0                                       // 000000001BF8: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:122
	v_mov_b32_e32 v0, s0                                       // 000000001BFC: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:123
	v_mov_b32_e32 v0, s0                                       // 000000001C00: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:124
	v_mov_b32_e32 v0, s0                                       // 000000001C04: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:125
	v_mov_b32_e32 v0, s0                                       // 000000001C08: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:126
	v_mov_b32_e32 v0, s0                                       // 000000001C0C: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:127
	v_mov_b32_e32 v0, s0                                       // 000000001C10: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:128
	v_mov_b32_e32 v0, s0                                       // 000000001C14: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:129
	v_mov_b32_e32 v0, s0                                       // 000000001C18: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:130
	v_mov_b32_e32 v0, s0                                       // 000000001C1C: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:131
	v_mov_b32_e32 v0, s0                                       // 000000001C20: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:132
	v_mov_b32_e32 v0, s0                                       // 000000001C24: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:133
	v_mov_b32_e32 v0, s0                                       // 000000001C28: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:134
	v_mov_b32_e32 v0, s0                                       // 000000001C2C: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:135
	v_mov_b32_e32 v0, s0                                       // 000000001C30: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:136
	v_mov_b32_e32 v0, s0                                       // 000000001C34: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:137
	v_mov_b32_e32 v0, s0                                       // 000000001C38: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:138
	v_mov_b32_e32 v0, s0                                       // 000000001C3C: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:139
	v_mov_b32_e32 v0, s0                                       // 000000001C40: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:140
	v_mov_b32_e32 v0, s0                                       // 000000001C44: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:141
	v_mov_b32_e32 v0, s0                                       // 000000001C48: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:142
	v_mov_b32_e32 v0, s0                                       // 000000001C4C: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:143
	v_mov_b32_e32 v0, s0                                       // 000000001C50: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:144
	v_mov_b32_e32 v0, s0                                       // 000000001C54: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:145
	v_mov_b32_e32 v0, s0                                       // 000000001C58: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:146
	v_mov_b32_e32 v0, s0                                       // 000000001C5C: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:147
	v_mov_b32_e32 v0, s0                                       // 000000001C60: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:148
	v_mov_b32_e32 v0, s0                                       // 000000001C64: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:149
	v_mov_b32_e32 v0, s0                                       // 000000001C68: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:150
	v_mov_b32_e32 v0, s0                                       // 000000001C6C: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:151
	v_mov_b32_e32 v0, s0                                       // 000000001C70: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:152
	v_mov_b32_e32 v0, s0                                       // 000000001C74: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:153
	v_mov_b32_e32 v0, s0                                       // 000000001C78: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:154
	v_mov_b32_e32 v0, s0                                       // 000000001C7C: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:155
	v_mov_b32_e32 v0, s0                                       // 000000001C80: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:156
	v_mov_b32_e32 v0, s0                                       // 000000001C84: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:157
	v_mov_b32_e32 v0, s0                                       // 000000001C88: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:158
	v_mov_b32_e32 v0, s0                                       // 000000001C8C: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:159
	v_mov_b32_e32 v0, s0                                       // 000000001C90: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:160
	v_mov_b32_e32 v0, s0                                       // 000000001C94: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:161
	v_mov_b32_e32 v0, s0                                       // 000000001C98: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:162
	v_mov_b32_e32 v0, s0                                       // 000000001C9C: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:163
	v_mov_b32_e32 v0, s0                                       // 000000001CA0: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:164
	v_mov_b32_e32 v0, s0                                       // 000000001CA4: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:165
	v_mov_b32_e32 v0, s0                                       // 000000001CA8: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:166
	v_mov_b32_e32 v0, s0                                       // 000000001CAC: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:167
	v_mov_b32_e32 v0, s0                                       // 000000001CB0: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:168
	v_mov_b32_e32 v0, s0                                       // 000000001CB4: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:169
	v_mov_b32_e32 v0, s0                                       // 000000001CB8: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:170
	v_mov_b32_e32 v0, s0                                       // 000000001CBC: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:171
	v_mov_b32_e32 v0, s0                                       // 000000001CC0: 7E000200
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:172
	s_nop 1                                                    // 000000001CC4: BF800001
	s_nop 1                                                    // 000000001CC8: BF800001
	s_nop 1                                                    // 000000001CCC: BF800001
	s_nop 1                                                    // 000000001CD0: BF800001
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:69
	s_cbranch_scc0 65425                                       // 000000001CD4: BFA1FF91 <_Z7kernel1ii+0x1c>
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:174
	s_endpgm                                                   // 000000001CD8: BFB00000
	s_nop 0                                                    // 000000001CDC: BF800000
	s_nop 0                                                    // 000000001CE0: BF800000
	s_nop 0                                                    // 000000001CE4: BF800000
	s_nop 0                                                    // 000000001CE8: BF800000
	s_nop 0                                                    // 000000001CEC: BF800000
	s_nop 0                                                    // 000000001CF0: BF800000
	s_nop 0                                                    // 000000001CF4: BF800000
	s_nop 0                                                    // 000000001CF8: BF800000
	s_nop 0                                                    // 000000001CFC: BF800000

0000000000001d00 <_Z7kernel2ii>:
; _Z7kernel2ii():
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:178
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1       // 000000001D00: B9800641 00000001
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:181
	s_load_b64 s[0:1], s[0:1], 0x0 nv                          // 000000001D08: F4102000 F8000000
	s_wait_kmcnt 0x0                                           // 000000001D10: BFC70000
	s_cmp_lt_i32 s1, 1                                         // 000000001D14: BF048101
	s_cbranch_scc1 111                                         // 000000001D18: BFA2006F <_Z7kernel2ii+0x1d8>
	s_add_co_i32 s1, s1, -1                                    // 000000001D1C: 8101C101
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:183
	s_nop 1                                                    // 000000001D20: BF800001
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:181
	s_cmp_eq_u32 s1, 0                                         // 000000001D24: BF068001
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:183
	s_nop 1                                                    // 000000001D28: BF800001
	s_nop 1                                                    // 000000001D2C: BF800001
	s_nop 1                                                    // 000000001D30: BF800001
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:184
	s_mov_b32 s2, s0                                           // 000000001D34: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:185
	s_mov_b32 s2, s0                                           // 000000001D38: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:186
	s_mov_b32 s2, s0                                           // 000000001D3C: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:187
	s_mov_b32 s2, s0                                           // 000000001D40: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:188
	s_mov_b32 s2, s0                                           // 000000001D44: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:189
	s_mov_b32 s2, s0                                           // 000000001D48: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:190
	s_mov_b32 s2, s0                                           // 000000001D4C: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:191
	s_mov_b32 s2, s0                                           // 000000001D50: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:192
	s_mov_b32 s2, s0                                           // 000000001D54: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:193
	s_mov_b32 s2, s0                                           // 000000001D58: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:194
	s_mov_b32 s2, s0                                           // 000000001D5C: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:195
	s_mov_b32 s2, s0                                           // 000000001D60: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:196
	s_mov_b32 s2, s0                                           // 000000001D64: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:197
	s_mov_b32 s2, s0                                           // 000000001D68: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:198
	s_mov_b32 s2, s0                                           // 000000001D6C: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:199
	s_mov_b32 s2, s0                                           // 000000001D70: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:200
	s_mov_b32 s2, s0                                           // 000000001D74: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:201
	s_mov_b32 s2, s0                                           // 000000001D78: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:202
	s_mov_b32 s2, s0                                           // 000000001D7C: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:203
	s_mov_b32 s2, s0                                           // 000000001D80: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:204
	s_mov_b32 s2, s0                                           // 000000001D84: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:205
	s_mov_b32 s2, s0                                           // 000000001D88: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:206
	s_mov_b32 s2, s0                                           // 000000001D8C: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:207
	s_mov_b32 s2, s0                                           // 000000001D90: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:208
	s_mov_b32 s2, s0                                           // 000000001D94: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:209
	s_mov_b32 s2, s0                                           // 000000001D98: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:210
	s_mov_b32 s2, s0                                           // 000000001D9C: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:211
	s_mov_b32 s2, s0                                           // 000000001DA0: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:212
	s_mov_b32 s2, s0                                           // 000000001DA4: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:213
	s_mov_b32 s2, s0                                           // 000000001DA8: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:214
	s_mov_b32 s2, s0                                           // 000000001DAC: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:215
	s_mov_b32 s2, s0                                           // 000000001DB0: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:216
	s_mov_b32 s2, s0                                           // 000000001DB4: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:217
	s_mov_b32 s2, s0                                           // 000000001DB8: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:218
	s_mov_b32 s2, s0                                           // 000000001DBC: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:219
	s_mov_b32 s2, s0                                           // 000000001DC0: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:220
	s_mov_b32 s2, s0                                           // 000000001DC4: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:221
	s_mov_b32 s2, s0                                           // 000000001DC8: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:222
	s_mov_b32 s2, s0                                           // 000000001DCC: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:223
	s_mov_b32 s2, s0                                           // 000000001DD0: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:224
	s_mov_b32 s2, s0                                           // 000000001DD4: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:225
	s_mov_b32 s2, s0                                           // 000000001DD8: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:226
	s_mov_b32 s2, s0                                           // 000000001DDC: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:227
	s_mov_b32 s2, s0                                           // 000000001DE0: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:228
	s_mov_b32 s2, s0                                           // 000000001DE4: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:229
	s_mov_b32 s2, s0                                           // 000000001DE8: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:230
	s_mov_b32 s2, s0                                           // 000000001DEC: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:231
	s_mov_b32 s2, s0                                           // 000000001DF0: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:232
	s_mov_b32 s2, s0                                           // 000000001DF4: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:233
	s_mov_b32 s2, s0                                           // 000000001DF8: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:234
	s_mov_b32 s2, s0                                           // 000000001DFC: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:235
	s_mov_b32 s2, s0                                           // 000000001E00: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:236
	s_mov_b32 s2, s0                                           // 000000001E04: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:237
	s_mov_b32 s2, s0                                           // 000000001E08: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:238
	s_mov_b32 s2, s0                                           // 000000001E0C: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:239
	s_mov_b32 s2, s0                                           // 000000001E10: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:240
	s_mov_b32 s2, s0                                           // 000000001E14: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:241
	s_mov_b32 s2, s0                                           // 000000001E18: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:242
	s_mov_b32 s2, s0                                           // 000000001E1C: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:243
	s_mov_b32 s2, s0                                           // 000000001E20: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:244
	s_mov_b32 s2, s0                                           // 000000001E24: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:245
	s_mov_b32 s2, s0                                           // 000000001E28: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:246
	s_mov_b32 s2, s0                                           // 000000001E2C: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:247
	s_mov_b32 s2, s0                                           // 000000001E30: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:248
	s_mov_b32 s2, s0                                           // 000000001E34: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:249
	s_mov_b32 s2, s0                                           // 000000001E38: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:250
	s_mov_b32 s2, s0                                           // 000000001E3C: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:251
	s_mov_b32 s2, s0                                           // 000000001E40: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:252
	s_mov_b32 s2, s0                                           // 000000001E44: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:253
	s_mov_b32 s2, s0                                           // 000000001E48: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:254
	s_mov_b32 s2, s0                                           // 000000001E4C: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:255
	s_mov_b32 s2, s0                                           // 000000001E50: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:256
	s_mov_b32 s2, s0                                           // 000000001E54: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:257
	s_mov_b32 s2, s0                                           // 000000001E58: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:258
	s_mov_b32 s2, s0                                           // 000000001E5C: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:259
	s_mov_b32 s2, s0                                           // 000000001E60: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:260
	s_mov_b32 s2, s0                                           // 000000001E64: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:261
	s_mov_b32 s2, s0                                           // 000000001E68: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:262
	s_mov_b32 s2, s0                                           // 000000001E6C: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:263
	s_mov_b32 s2, s0                                           // 000000001E70: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:264
	s_mov_b32 s2, s0                                           // 000000001E74: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:265
	s_mov_b32 s2, s0                                           // 000000001E78: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:266
	s_mov_b32 s2, s0                                           // 000000001E7C: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:267
	s_mov_b32 s2, s0                                           // 000000001E80: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:268
	s_mov_b32 s2, s0                                           // 000000001E84: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:269
	s_mov_b32 s2, s0                                           // 000000001E88: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:270
	s_mov_b32 s2, s0                                           // 000000001E8C: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:271
	s_mov_b32 s2, s0                                           // 000000001E90: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:272
	s_mov_b32 s2, s0                                           // 000000001E94: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:273
	s_mov_b32 s2, s0                                           // 000000001E98: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:274
	s_mov_b32 s2, s0                                           // 000000001E9C: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:275
	s_mov_b32 s2, s0                                           // 000000001EA0: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:276
	s_mov_b32 s2, s0                                           // 000000001EA4: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:277
	s_mov_b32 s2, s0                                           // 000000001EA8: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:278
	s_mov_b32 s2, s0                                           // 000000001EAC: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:279
	s_mov_b32 s2, s0                                           // 000000001EB0: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:280
	s_mov_b32 s2, s0                                           // 000000001EB4: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:281
	s_mov_b32 s2, s0                                           // 000000001EB8: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:282
	s_mov_b32 s2, s0                                           // 000000001EBC: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:283
	s_mov_b32 s2, s0                                           // 000000001EC0: BE820000
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:284
	s_nop 1                                                    // 000000001EC4: BF800001
	s_nop 1                                                    // 000000001EC8: BF800001
	s_nop 1                                                    // 000000001ECC: BF800001
	s_nop 1                                                    // 000000001ED0: BF800001
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:181
	s_cbranch_scc0 65425                                       // 000000001ED4: BFA1FF91 <_Z7kernel2ii+0x1c>
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:286
	s_endpgm                                                   // 000000001ED8: BFB00000
	s_nop 0                                                    // 000000001EDC: BF800000
	s_nop 0                                                    // 000000001EE0: BF800000
	s_nop 0                                                    // 000000001EE4: BF800000
	s_nop 0                                                    // 000000001EE8: BF800000
	s_nop 0                                                    // 000000001EEC: BF800000
	s_nop 0                                                    // 000000001EF0: BF800000
	s_nop 0                                                    // 000000001EF4: BF800000
	s_nop 0                                                    // 000000001EF8: BF800000
	s_nop 0                                                    // 000000001EFC: BF800000

0000000000001f00 <_Z7kernel3fi>:
; _Z7kernel3fi():
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:290
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1       // 000000001F00: B9800641 00000001
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:296
	s_load_b64 s[0:1], s[0:1], 0x0 nv                          // 000000001F08: F4102000 F8000000
	s_wait_kmcnt 0x0                                           // 000000001F10: BFC70000
	s_cmp_lt_i32 s1, 1                                         // 000000001F14: BF048101
	s_cbranch_scc1 234                                         // 000000001F18: BFA200EA <_Z7kernel3fi+0x3c4>
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:291
	v_cvt_f64_u32_e32 v[2:3], v0                               // 000000001F1C: 7E042D00
	v_and_b32_e32 v1, 1, v0                                    // 000000001F20: 36020081
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:293
	v_cvt_f32_u32_e32 v0, v0                                   // 000000001F24: 7E000D00
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001F28: BF870002
	v_cmp_eq_u32_e32 vcc_lo, 1, v1                             // 000000001F2C: 7C940281
	s_branch 5                                                 // 000000001F30: BFA00005 <_Z7kernel3fi+0x48>
	s_or_b32 exec_lo, exec_lo, s2                              // 000000001F34: 8C7E027E
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:296
	s_add_co_i32 s1, s1, -1                                    // 000000001F38: 8101C101
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001F3C: BF870009
	s_cmp_eq_u32 s1, 0                                         // 000000001F40: BF068001
	s_cbranch_scc1 223                                         // 000000001F44: BFA200DF <_Z7kernel3fi+0x3c4>
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:298
	s_and_saveexec_b32 s2, vcc_lo                              // 000000001F48: BE82206A
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001F4C: BF870009
	s_xor_b32 s2, exec_lo, s2                                  // 000000001F50: 8D02027E
	s_cbranch_execz 108                                        // 000000001F54: BFA5006C <_Z7kernel3fi+0x208>
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:405
	s_nop 1                                                    // 000000001F58: BF800001
	s_nop 1                                                    // 000000001F5C: BF800001
	s_nop 1                                                    // 000000001F60: BF800001
	s_nop 1                                                    // 000000001F64: BF800001
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:406
	v_rcp_f32_e32 v0, v0                                       // 000000001F68: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:407
	v_rcp_f32_e32 v0, v0                                       // 000000001F6C: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:408
	v_rcp_f32_e32 v0, v0                                       // 000000001F70: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:409
	v_rcp_f32_e32 v0, v0                                       // 000000001F74: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:410
	v_rcp_f32_e32 v0, v0                                       // 000000001F78: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:411
	v_rcp_f32_e32 v0, v0                                       // 000000001F7C: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:412
	v_rcp_f32_e32 v0, v0                                       // 000000001F80: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:413
	v_rcp_f32_e32 v0, v0                                       // 000000001F84: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:414
	v_rcp_f32_e32 v0, v0                                       // 000000001F88: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:415
	v_rcp_f32_e32 v0, v0                                       // 000000001F8C: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:416
	v_rcp_f32_e32 v0, v0                                       // 000000001F90: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:417
	v_rcp_f32_e32 v0, v0                                       // 000000001F94: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:418
	v_rcp_f32_e32 v0, v0                                       // 000000001F98: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:419
	v_rcp_f32_e32 v0, v0                                       // 000000001F9C: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:420
	v_rcp_f32_e32 v0, v0                                       // 000000001FA0: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:421
	v_rcp_f32_e32 v0, v0                                       // 000000001FA4: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:422
	v_rcp_f32_e32 v0, v0                                       // 000000001FA8: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:423
	v_rcp_f32_e32 v0, v0                                       // 000000001FAC: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:424
	v_rcp_f32_e32 v0, v0                                       // 000000001FB0: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:425
	v_rcp_f32_e32 v0, v0                                       // 000000001FB4: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:426
	v_rcp_f32_e32 v0, v0                                       // 000000001FB8: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:427
	v_rcp_f32_e32 v0, v0                                       // 000000001FBC: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:428
	v_rcp_f32_e32 v0, v0                                       // 000000001FC0: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:429
	v_rcp_f32_e32 v0, v0                                       // 000000001FC4: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:430
	v_rcp_f32_e32 v0, v0                                       // 000000001FC8: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:431
	v_rcp_f32_e32 v0, v0                                       // 000000001FCC: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:432
	v_rcp_f32_e32 v0, v0                                       // 000000001FD0: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:433
	v_rcp_f32_e32 v0, v0                                       // 000000001FD4: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:434
	v_rcp_f32_e32 v0, v0                                       // 000000001FD8: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:435
	v_rcp_f32_e32 v0, v0                                       // 000000001FDC: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:436
	v_rcp_f32_e32 v0, v0                                       // 000000001FE0: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:437
	v_rcp_f32_e32 v0, v0                                       // 000000001FE4: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:438
	v_rcp_f32_e32 v0, v0                                       // 000000001FE8: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:439
	v_rcp_f32_e32 v0, v0                                       // 000000001FEC: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:440
	v_rcp_f32_e32 v0, v0                                       // 000000001FF0: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:441
	v_rcp_f32_e32 v0, v0                                       // 000000001FF4: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:442
	v_rcp_f32_e32 v0, v0                                       // 000000001FF8: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:443
	v_rcp_f32_e32 v0, v0                                       // 000000001FFC: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:444
	v_rcp_f32_e32 v0, v0                                       // 000000002000: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:445
	v_rcp_f32_e32 v0, v0                                       // 000000002004: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:446
	v_rcp_f32_e32 v0, v0                                       // 000000002008: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:447
	v_rcp_f32_e32 v0, v0                                       // 00000000200C: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:448
	v_rcp_f32_e32 v0, v0                                       // 000000002010: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:449
	v_rcp_f32_e32 v0, v0                                       // 000000002014: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:450
	v_rcp_f32_e32 v0, v0                                       // 000000002018: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:451
	v_rcp_f32_e32 v0, v0                                       // 00000000201C: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:452
	v_rcp_f32_e32 v0, v0                                       // 000000002020: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:453
	v_rcp_f32_e32 v0, v0                                       // 000000002024: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:454
	v_rcp_f32_e32 v0, v0                                       // 000000002028: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:455
	v_rcp_f32_e32 v0, v0                                       // 00000000202C: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:456
	v_rcp_f32_e32 v0, v0                                       // 000000002030: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:457
	v_rcp_f32_e32 v0, v0                                       // 000000002034: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:458
	v_rcp_f32_e32 v0, v0                                       // 000000002038: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:459
	v_rcp_f32_e32 v0, v0                                       // 00000000203C: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:460
	v_rcp_f32_e32 v0, v0                                       // 000000002040: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:461
	v_rcp_f32_e32 v0, v0                                       // 000000002044: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:462
	v_rcp_f32_e32 v0, v0                                       // 000000002048: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:463
	v_rcp_f32_e32 v0, v0                                       // 00000000204C: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:464
	v_rcp_f32_e32 v0, v0                                       // 000000002050: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:465
	v_rcp_f32_e32 v0, v0                                       // 000000002054: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:466
	v_rcp_f32_e32 v0, v0                                       // 000000002058: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:467
	v_rcp_f32_e32 v0, v0                                       // 00000000205C: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:468
	v_rcp_f32_e32 v0, v0                                       // 000000002060: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:469
	v_rcp_f32_e32 v0, v0                                       // 000000002064: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:470
	v_rcp_f32_e32 v0, v0                                       // 000000002068: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:471
	v_rcp_f32_e32 v0, v0                                       // 00000000206C: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:472
	v_rcp_f32_e32 v0, v0                                       // 000000002070: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:473
	v_rcp_f32_e32 v0, v0                                       // 000000002074: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:474
	v_rcp_f32_e32 v0, v0                                       // 000000002078: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:475
	v_rcp_f32_e32 v0, v0                                       // 00000000207C: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:476
	v_rcp_f32_e32 v0, v0                                       // 000000002080: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:477
	v_rcp_f32_e32 v0, v0                                       // 000000002084: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:478
	v_rcp_f32_e32 v0, v0                                       // 000000002088: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:479
	v_rcp_f32_e32 v0, v0                                       // 00000000208C: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:480
	v_rcp_f32_e32 v0, v0                                       // 000000002090: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:481
	v_rcp_f32_e32 v0, v0                                       // 000000002094: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:482
	v_rcp_f32_e32 v0, v0                                       // 000000002098: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:483
	v_rcp_f32_e32 v0, v0                                       // 00000000209C: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:484
	v_rcp_f32_e32 v0, v0                                       // 0000000020A0: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:485
	v_rcp_f32_e32 v0, v0                                       // 0000000020A4: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:486
	v_rcp_f32_e32 v0, v0                                       // 0000000020A8: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:487
	v_rcp_f32_e32 v0, v0                                       // 0000000020AC: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:488
	v_rcp_f32_e32 v0, v0                                       // 0000000020B0: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:489
	v_rcp_f32_e32 v0, v0                                       // 0000000020B4: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:490
	v_rcp_f32_e32 v0, v0                                       // 0000000020B8: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:491
	v_rcp_f32_e32 v0, v0                                       // 0000000020BC: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:492
	v_rcp_f32_e32 v0, v0                                       // 0000000020C0: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:493
	v_rcp_f32_e32 v0, v0                                       // 0000000020C4: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:494
	v_rcp_f32_e32 v0, v0                                       // 0000000020C8: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:495
	v_rcp_f32_e32 v0, v0                                       // 0000000020CC: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:496
	v_rcp_f32_e32 v0, v0                                       // 0000000020D0: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:497
	v_rcp_f32_e32 v0, v0                                       // 0000000020D4: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:498
	v_rcp_f32_e32 v0, v0                                       // 0000000020D8: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:499
	v_rcp_f32_e32 v0, v0                                       // 0000000020DC: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:500
	v_rcp_f32_e32 v0, v0                                       // 0000000020E0: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:501
	v_rcp_f32_e32 v0, v0                                       // 0000000020E4: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:502
	v_rcp_f32_e32 v0, v0                                       // 0000000020E8: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:503
	v_rcp_f32_e32 v0, v0                                       // 0000000020EC: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:504
	v_rcp_f32_e32 v0, v0                                       // 0000000020F0: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:505
	v_rcp_f32_e32 v0, v0                                       // 0000000020F4: 7E005500
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:506
	s_nop 1                                                    // 0000000020F8: BF800001
	s_nop 1                                                    // 0000000020FC: BF800001
	s_nop 1                                                    // 000000002100: BF800001
	s_nop 1                                                    // 000000002104: BF800001
	s_and_not1_saveexec_b32 s2, s2                             // 000000002108: BE823002
	s_cbranch_execz 65417                                      // 00000000210C: BFA5FF89 <_Z7kernel3fi+0x34>
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:300
	s_nop 1                                                    // 000000002110: BF800001
	s_nop 1                                                    // 000000002114: BF800001
	s_nop 1                                                    // 000000002118: BF800001
	s_nop 1                                                    // 00000000211C: BF800001
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:301
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002120: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:302
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002124: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:303
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002128: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:304
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 00000000212C: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:305
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002130: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:306
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002134: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:307
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002138: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:308
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 00000000213C: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:309
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002140: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:310
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002144: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:311
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002148: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:312
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 00000000214C: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:313
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002150: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:314
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002154: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:315
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002158: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:316
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 00000000215C: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:317
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002160: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:318
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002164: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:319
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002168: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:320
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 00000000216C: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:321
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002170: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:322
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002174: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:323
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002178: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:324
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 00000000217C: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:325
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002180: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:326
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002184: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:327
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002188: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:328
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 00000000218C: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:329
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002190: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:330
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002194: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:331
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002198: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:332
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 00000000219C: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:333
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021A0: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:334
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021A4: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:335
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021A8: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:336
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021AC: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:337
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021B0: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:338
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021B4: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:339
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021B8: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:340
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021BC: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:341
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021C0: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:342
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021C4: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:343
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021C8: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:344
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021CC: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:345
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021D0: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:346
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021D4: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:347
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021D8: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:348
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021DC: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:349
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021E0: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:350
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021E4: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:351
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021E8: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:352
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021EC: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:353
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021F0: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:354
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021F4: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:355
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021F8: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:356
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000021FC: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:357
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002200: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:358
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002204: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:359
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002208: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:360
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 00000000220C: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:361
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002210: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:362
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002214: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:363
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002218: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:364
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 00000000221C: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:365
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002220: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:366
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002224: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:367
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002228: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:368
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 00000000222C: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:369
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002230: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:370
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002234: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:371
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002238: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:372
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 00000000223C: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:373
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002240: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:374
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002244: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:375
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002248: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:376
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 00000000224C: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:377
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002250: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:378
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002254: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:379
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002258: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:380
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 00000000225C: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:381
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002260: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:382
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002264: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:383
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002268: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:384
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 00000000226C: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:385
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002270: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:386
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002274: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:387
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002278: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:388
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 00000000227C: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:389
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002280: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:390
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002284: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:391
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002288: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:392
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 00000000228C: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:393
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002290: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:394
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002294: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:395
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 000000002298: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:396
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 00000000229C: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:397
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000022A0: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:398
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000022A4: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:399
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000022A8: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:400
	v_rcp_f64_e32 v[2:3], v[2:3]                               // 0000000022AC: 7E045F02
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:401
	s_nop 1                                                    // 0000000022B0: BF800001
	s_nop 1                                                    // 0000000022B4: BF800001
	s_nop 1                                                    // 0000000022B8: BF800001
	s_nop 1                                                    // 0000000022BC: BF800001
	s_branch 65308                                             // 0000000022C0: BFA0FF1C <_Z7kernel3fi+0x34>
; /home/vlaindic/git/rocm-systems/projects/rocprofiler-sdk/tests/bin/pc-sampling/exec-mask-manipulation/exec_mask_manipulation.cpp:509
	s_endpgm                                                   // 0000000022C4: BFB00000
	s_code_end                                                 // 0000000022C8: BF9F0000
	s_code_end                                                 // 0000000022CC: BF9F0000
	s_code_end                                                 // 0000000022D0: BF9F0000
	s_code_end                                                 // 0000000022D4: BF9F0000
	s_code_end                                                 // 0000000022D8: BF9F0000
	s_code_end                                                 // 0000000022DC: BF9F0000
	s_code_end                                                 // 0000000022E0: BF9F0000
	s_code_end                                                 // 0000000022E4: BF9F0000
	s_code_end                                                 // 0000000022E8: BF9F0000
	s_code_end                                                 // 0000000022EC: BF9F0000
	s_code_end                                                 // 0000000022F0: BF9F0000
	s_code_end                                                 // 0000000022F4: BF9F0000
	s_code_end                                                 // 0000000022F8: BF9F0000
	s_code_end                                                 // 0000000022FC: BF9F0000
	s_code_end                                                 // 000000002300: BF9F0000
	s_code_end                                                 // 000000002304: BF9F0000
	s_code_end                                                 // 000000002308: BF9F0000
	s_code_end                                                 // 00000000230C: BF9F0000
	s_code_end                                                 // 000000002310: BF9F0000
	s_code_end                                                 // 000000002314: BF9F0000
	s_code_end                                                 // 000000002318: BF9F0000
	s_code_end                                                 // 00000000231C: BF9F0000
	s_code_end                                                 // 000000002320: BF9F0000
	s_code_end                                                 // 000000002324: BF9F0000
	s_code_end                                                 // 000000002328: BF9F0000
	s_code_end                                                 // 00000000232C: BF9F0000
	s_code_end                                                 // 000000002330: BF9F0000
	s_code_end                                                 // 000000002334: BF9F0000
	s_code_end                                                 // 000000002338: BF9F0000
	s_code_end                                                 // 00000000233C: BF9F0000
	s_code_end                                                 // 000000002340: BF9F0000
	s_code_end                                                 // 000000002344: BF9F0000
	s_code_end                                                 // 000000002348: BF9F0000
	s_code_end                                                 // 00000000234C: BF9F0000
	s_code_end                                                 // 000000002350: BF9F0000
	s_code_end                                                 // 000000002354: BF9F0000
	s_code_end                                                 // 000000002358: BF9F0000
	s_code_end                                                 // 00000000235C: BF9F0000
	s_code_end                                                 // 000000002360: BF9F0000
	s_code_end                                                 // 000000002364: BF9F0000
	s_code_end                                                 // 000000002368: BF9F0000
	s_code_end                                                 // 00000000236C: BF9F0000
	s_code_end                                                 // 000000002370: BF9F0000
	s_code_end                                                 // 000000002374: BF9F0000
	s_code_end                                                 // 000000002378: BF9F0000
	s_code_end                                                 // 00000000237C: BF9F0000
	s_code_end                                                 // 000000002380: BF9F0000
	s_code_end                                                 // 000000002384: BF9F0000
	s_code_end                                                 // 000000002388: BF9F0000
	s_code_end                                                 // 00000000238C: BF9F0000
	s_code_end                                                 // 000000002390: BF9F0000
	s_code_end                                                 // 000000002394: BF9F0000
	s_code_end                                                 // 000000002398: BF9F0000
	s_code_end                                                 // 00000000239C: BF9F0000
	s_code_end                                                 // 0000000023A0: BF9F0000
	s_code_end                                                 // 0000000023A4: BF9F0000
	s_code_end                                                 // 0000000023A8: BF9F0000
	s_code_end                                                 // 0000000023AC: BF9F0000
	s_code_end                                                 // 0000000023B0: BF9F0000
	s_code_end                                                 // 0000000023B4: BF9F0000
	s_code_end                                                 // 0000000023B8: BF9F0000
	s_code_end                                                 // 0000000023BC: BF9F0000
	s_code_end                                                 // 0000000023C0: BF9F0000
	s_code_end                                                 // 0000000023C4: BF9F0000
	s_code_end                                                 // 0000000023C8: BF9F0000
	s_code_end                                                 // 0000000023CC: BF9F0000
	s_code_end                                                 // 0000000023D0: BF9F0000
	s_code_end                                                 // 0000000023D4: BF9F0000
	s_code_end                                                 // 0000000023D8: BF9F0000
	s_code_end                                                 // 0000000023DC: BF9F0000
	s_code_end                                                 // 0000000023E0: BF9F0000
	s_code_end                                                 // 0000000023E4: BF9F0000
	s_code_end                                                 // 0000000023E8: BF9F0000
	s_code_end                                                 // 0000000023EC: BF9F0000
	s_code_end                                                 // 0000000023F0: BF9F0000
	s_code_end                                                 // 0000000023F4: BF9F0000
	s_code_end                                                 // 0000000023F8: BF9F0000
	s_code_end                                                 // 0000000023FC: BF9F0000
	s_code_end                                                 // 000000002400: BF9F0000
	s_code_end                                                 // 000000002404: BF9F0000
	s_code_end                                                 // 000000002408: BF9F0000
	s_code_end                                                 // 00000000240C: BF9F0000
	s_code_end                                                 // 000000002410: BF9F0000
	s_code_end                                                 // 000000002414: BF9F0000
	s_code_end                                                 // 000000002418: BF9F0000
	s_code_end                                                 // 00000000241C: BF9F0000
	s_code_end                                                 // 000000002420: BF9F0000
	s_code_end                                                 // 000000002424: BF9F0000
	s_code_end                                                 // 000000002428: BF9F0000
	s_code_end                                                 // 00000000242C: BF9F0000
	s_code_end                                                 // 000000002430: BF9F0000
	s_code_end                                                 // 000000002434: BF9F0000
	s_code_end                                                 // 000000002438: BF9F0000
	s_code_end                                                 // 00000000243C: BF9F0000
	s_code_end                                                 // 000000002440: BF9F0000
	s_code_end                                                 // 000000002444: BF9F0000
	s_code_end                                                 // 000000002448: BF9F0000
	s_code_end                                                 // 00000000244C: BF9F0000
	s_code_end                                                 // 000000002450: BF9F0000
	s_code_end                                                 // 000000002454: BF9F0000
	s_code_end                                                 // 000000002458: BF9F0000
	s_code_end                                                 // 00000000245C: BF9F0000
	s_code_end                                                 // 000000002460: BF9F0000
	s_code_end                                                 // 000000002464: BF9F0000
	s_code_end                                                 // 000000002468: BF9F0000
	s_code_end                                                 // 00000000246C: BF9F0000
	s_code_end                                                 // 000000002470: BF9F0000
	s_code_end                                                 // 000000002474: BF9F0000
	s_code_end                                                 // 000000002478: BF9F0000
	s_code_end                                                 // 00000000247C: BF9F0000
