////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

/// Trap Handler V2 source
.set DOORBELL_ID_SIZE                              , 10
.set DOORBELL_ID_MASK                              , ((1 << DOORBELL_ID_SIZE) - 1)
.set EC_QUEUE_WAVE_ABORT_M0                        , (1 << (DOORBELL_ID_SIZE + 0))
.set EC_QUEUE_WAVE_TRAP_M0                         , (1 << (DOORBELL_ID_SIZE + 1))
.set EC_QUEUE_WAVE_MATH_ERROR_M0                   , (1 << (DOORBELL_ID_SIZE + 2))
.set EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION_M0          , (1 << (DOORBELL_ID_SIZE + 3))
.set EC_QUEUE_WAVE_MEMORY_VIOLATION_M0             , (1 << (DOORBELL_ID_SIZE + 4))
.set EC_QUEUE_WAVE_APERTURE_VIOLATION_M0           , (1 << (DOORBELL_ID_SIZE + 5))

.set SQ_WAVE_EXCP_FLAG_PRIV_ADDR_WATCH_MASK        , (1 << 4) - 1
.set SQ_WAVE_EXCP_FLAG_PRIV_MEMVIOL_SHIFT          , 4
.set SQ_WAVE_EXCP_FLAG_PRIV_SAVE_CONTEXT           , 5
.set SQ_WAVE_EXCP_FLAG_PRIV_ILLEGAL_INST_SHIFT     , 6
.set SQ_WAVE_EXCP_FLAG_PRIV_HT_SHIFT               , 7
.set SQ_WAVE_EXCP_FLAG_PRIV_WAVE_START_SHIFT       , 8
.set SQ_WAVE_EXCP_FLAG_PRIV_WAVE_END_SHIFT         , 9
.set SQ_WAVE_EXCP_FLAG_PRIV_PERF_SNAPSHOT_SHIFT    , 10
.set SQ_WAVE_EXCP_FLAG_PRIV_TRAP_AFTER_INST_SHIFT  , 11
.set SQ_WAVE_EXCP_FLAG_PRIV_XNACK_ERROR_SHIFT      , 12

.set SQ_WAVE_EXCP_FLAG_USER_MATH_EXCP_SHIFT        , 0
.set SQ_WAVE_EXCP_FLAG_USER_MATH_EXCP_SIZE         , 7

.set SQ_WAVE_TRAP_CTRL_MATH_EXCP_MASK              , ((1 << 7) - 1)
.set SQ_WAVE_TRAP_CTRL_ADDR_WATCH_SHIFT            , 7
.set SQ_WAVE_TRAP_CTRL_WAVE_END_SHIFT              , 8
.set SQ_WAVE_TRAP_CTRL_TRAP_AFTER_INST             , 9

// The PC is dword (32bit) aligned, so the 2 LSBs are always zero.
.set SQ_WAVE_PC_LO_ADDRESS_MASK                    , 0xFFFFFFFC
.if .amdgcn.gfx_generation_minor == 0
  .set SQ_WAVE_PC_HI_ADDRESS_MASK                  , 0xFFFF
.else
  .set SQ_WAVE_PC_HI_ADDRESS_MASK                  , 0x1FFFFFF
.endif
.set SQ_WAVE_PC_HI_TRAP_ID_BFE                     , (SQ_WAVE_PC_HI_TRAP_ID_SHIFT | (SQ_WAVE_PC_HI_TRAP_ID_SIZE << 16))
.set SQ_WAVE_PC_HI_TRAP_ID_SHIFT                   , 28
.set SQ_WAVE_PC_HI_TRAP_ID_SIZE                    , 4

.set SQ_WAVE_STATE_PRIV_HALT_BFE                   , (SQ_WAVE_STATE_PRIV_HALT_SHIFT | (1 << 16))
.set SQ_WAVE_STATE_PRIV_HALT_SHIFT                 , 14
.set SQ_WAVE_STATE_PRIV_BARRIER_COMPLETE_SHIFT     , 2

.set TRAP_ID_ABORT                                 , 2
.set TRAP_ID_DEBUGTRAP                             , 3
.if .amdgcn.gfx_generation_minor == 0
  .set TTMP1_SCHED_MODE_MASK                       , 0xC000000
.endif

.set TTMP6_SAVED_STATUS_HALT_MASK                  , (1 << TTMP6_SAVED_STATUS_HALT_SHIFT)
.set TTMP6_SAVED_STATUS_HALT_SHIFT                 , 29
.set TTMP6_WAVE_STOPPED_SHIFT                      , 30
.if .amdgcn.gfx_generation_minor == 0
  .set TTMP6_SAVED_TRAP_ID_BFE                     , (TTMP6_SAVED_TRAP_ID_SHIFT | (TTMP6_SAVED_TRAP_ID_SIZE << 16))
  .set TTMP6_SAVED_TRAP_ID_MASK                    , (((1 << TTMP6_SAVED_TRAP_ID_SIZE) - 1) << TTMP6_SAVED_TRAP_ID_SHIFT)
  .set TTMP6_SAVED_TRAP_ID_SHIFT                   , 25
  .set TTMP6_SAVED_TRAP_ID_SIZE                    , 4
.endif
.set TTMP8_DEBUG_FLAG_SHIFT                        , 31

.set TTMP11_DEBUG_ENABLED_SHIFT                    , 23
.if .amdgcn.gfx_generation_minor == 0
  .set TTMP11_SCHED_MODE_SHIFT                     , 26
  .set TTMP11_SCHED_MODE_SIZE                      , 2
  .set TTMP11_SCHED_MODE_MASK                      , 0xC000000
  .set TTMP11_SCHED_MODE_BFE                       , (TTMP11_SCHED_MODE_SHIFT | (TTMP11_SCHED_MODE_SIZE << 16))
.endif
.if .amdgcn.gfx_generation_minor == 5
  .set TTMP11_SAVED_TRAP_ID_SHIFT                  , 28
  .set TTMP11_SAVED_TRAP_ID_SIZE                   , 4
  .set TTMP11_SAVED_TRAP_ID_MASK                   , (((1 << TTMP11_SAVED_TRAP_ID_SIZE) - 1) << TTMP11_SAVED_TRAP_ID_SHIFT)
  .set TTMP11_SAVED_TRAP_ID_BFE                    , (TTMP11_SAVED_TRAP_ID_SHIFT | (TTMP11_SAVED_TRAP_ID_SIZE << 16))

  .set TTMP11_FXPTR_SHIFT                          , 14
  .set TTMP11_REPLAY_W64H_SHIFT                    , 21
  .set TTMP11_FIRST_REPLAY_SHIFT                   , 22
.endif

.if .amdgcn.gfx_generation_minor == 0
  .set TTMP_PC_HI_SHIFT                            , 7
.endif
.set TTMP13_HT_FLAG_BIT_SHIFT                      , 22           // TTMP13 bit for host-trap
.set TTMP13_STOCH_FLAG_BIT_SHIFT                   , 23           // TTMP13 bit for stochastic
.set TTMP13_BUF_ID_BIT_POSITION                    , 31           // TTMP13 bit position for buffer ID

.set TTMP8_DISPATCH_ID_MASK                        , 0X1FFFFFF
// Per-sample data layout within the device buffer. Each sample is 64 bytes.
// These are offsets from the start of a specific sample slot in the device buffer.

.set SAMPLE_OFF_BYTES_PER_SAMPLE                   , 0x40         // 64 bytes per sample slot
.set SAMPLE_OFF_PC_HOST                            , 0x00         // original PC (host trap only)
.set SAMPLE_OFF_EXEC_LOHI                          , 0x08         // saved EXEC low/high
.set SAMPLE_OFF_WGID_XY                            , 0x10         // WG id X / Y
.set SAMPLE_OFF_WGID_Z                             , 0x18         // WG id Z (32-bit)
.set SAMPLE_OFF_BITFIELD                           , 0x1C         // wave_in_wg[5:0] | reserved_wg[7:6] | chiplet[10:8] | reserved[31:11]
.set SAMPLE_OFF_TIMESTAMP                          , 0x30         // 64 bit realtime counter
.set SAMPLE_OFF_HW_ID                              , 0x20         // Combined HW_ID (HW_ID1 + HW_ID2)
.set SAMPLE_OFF_SNAPSHOT_DATA                      , 0x24         // Performance snapshot data
.set SAMPLE_OFF_CORRELATION                        , 0x38         // doorbell + dispatch id
.set SAMPLE_OFF_BUF_WRITTEN_VAL                    , 0x10         // Offset to buf_written_val0/1 in pcs_sampling_data_t
.set SAMPLE_OFF_WATERMARK_FIELD                    , 0x14         // Offset to watermark field in pcs_sampling_data_t
.set SAMPLE_OFF_BUF_SIZE                           , 0x8          // Offset to buf_size in pcs_sampling_data_t
.set SAMPLE_OFF_DONE_SIG0                          , 0x18         // Offset for done_sig0 (hsa_signal_t handle for buffer 0)
.set SAMPLE_OFF_DONE_SIG1                          , 0x28         // Offset for done_sig1 (hsa_signal_t handle for buffer 1)
.set SAMPLE_OFF_SIGNAL_VALUE                       , 0x8          // Offset within signal structure to value field
.set SAMPLE_OFF_EVENT_MAILBOX0                     , 0x10         // Offset for event mailbox pointer for buffer 0
.set SAMPLE_OFF_EVENT_MAILBOX1                     , 0x20         // Offset for event mailbox pointer for buffer 1

.set WAVE_ID_MASK                                  , 0x1f         // Mask to extract Wave ID from TTMP register.
.set WAVE_ID_WG_BIT_POSITION                       , 25           // Wave ID is stored in bits [29:25] of ttmp8, so we need to shift it right by 25 bits.
.set BUF_INDEX_MASK                                , 0x7fffffff   // Extract bit31 from the buffer index in the device buffer.
.set SAMPLE_INDEX_WIDTH                            , 31           // The sample index is 63 bits; the high part is 31 bits.

.if .amdgcn.gfx_generation_minor == 5
  .set SQ_WAVE_XNACK_STATE_PRIV_FXPTR_SHIFT        , 0
  .set SQ_WAVE_XNACK_STATE_PRIV_FXPTR_SIZE         , 7
  .set SQ_WAVE_XNACK_STATE_PRIV_REPLAY_W64H_SHIFT  , 16
  .set SQ_WAVE_XNACK_STATE_PRIV_REPLAY_W64H_SIZE   , 1
  .set SQ_WAVE_XNACK_STATE_PRIV_FIRST_REPLAY_SHIFT , 18
  .set SQ_WAVE_XNACK_STATE_PRIV_FIRST_REPLAY_SIZE  , 1
// XNACK_MASK storage locations in ttmp11
  .set TTMP11_XNACK_MASK_LO_SIZE                   , 14
  .set TTMP11_XNACK_MASK_LO_MASK                   , ((1 << TTMP11_XNACK_MASK_LO_SIZE) - 1)
  .set TTMP11_XNACK_MASK_HI_SHIFT                  , 28           // Bits [31:28] for XNACK_MASK[17:14]
  .set TTMP11_XNACK_MASK_HI_SIZE                   , 4
  .set TTMP11_XNACK_MASK_HI_MASK                   , (((1 << TTMP11_XNACK_MASK_HI_SIZE) - 1) << TTMP11_XNACK_MASK_HI_SHIFT)
.endif

//////////////////////////////////////////////////////////////////
// GENERATION-SPECIFIC MACROS FOR REGISTER USAGE
//////////////////////////////////////////////////////////////////

// These macros abstract register usage differences between GFX12.0 and GFX12.5
// GFX12.5 uses vcc_hi as scratch, GFX12.0 uses ttmp6

.macro V_READLANE_B32 vsrc, lane
  .if .amdgcn.gfx_generation_minor >= 5
      v_readlane_b32    vcc_hi, \vsrc, \lane
  .else
      v_readlane_b32    ttmp6, \vsrc, \lane
  .endif
.endm

.macro S_MUL_I32  sdst, src1
  .if .amdgcn.gfx_generation_minor >= 5
      s_mul_i32    \sdst, vcc_hi, \src1
  .else
      s_mul_i32    \sdst, ttmp6, \src1
  .endif
.endm

.macro S_MUL_HI_U32 sdst, src1
  .if .amdgcn.gfx_generation_minor >= 5
      s_mul_hi_u32    \sdst, vcc_hi, \src1
  .else
      s_mul_hi_u32    \sdst, ttmp6, \src1
  .endif
.endm

.macro S_BFE_U32 src0, src1
  .if .amdgcn.gfx_generation_minor >= 5
      s_bfe_u32     vcc_hi,  \src0, \src1
  .else
      s_bfe_u32     ttmp6,  \src0, \src1
  .endif
.endm

.macro V_WRITELANE_B32 vdst, lane
  .if .amdgcn.gfx_generation_minor >= 5
      v_writelane_b32    \vdst, vcc_hi, \lane
  .else
      v_writelane_b32    \vdst, ttmp6, \lane
  .endif
.endm

.macro S_GETREG_B32  hw_reg_name
  .if .amdgcn.gfx_generation_minor >= 5
      s_getreg_b32   vcc_hi, hwreg(\hw_reg_name)
  .else
      s_getreg_b32   ttmp6, hwreg(\hw_reg_name)
  .endif
.endm

.macro S_ADD_U32  sdst, src0
  .if .amdgcn.gfx_generation_minor >= 5
      s_add_u32    \sdst, \src0, vcc_hi
  .else
      s_add_u32    \sdst, \src0, ttmp6
  .endif
.endm

.macro S_LSHR_B32 arg1,arg2
  .if .amdgcn.gfx_generation_minor >= 5
      s_lshr_b32    vcc_hi, \arg1, \arg2
  .else
      s_lshr_b32    ttmp6, \arg1, \arg2
  .endif
.endm

.macro S_MULK_I32 arg1
  .if .amdgcn.gfx_generation_minor >= 5
      s_mulk_i32    vcc_hi,\arg1
  .else
      s_mulk_i32    ttmp6,\arg1
  .endif
.endm

// Macro to save XNACK_MASK register state at trap entry (GFX12.5 only)
// Required to preserve XNACK state across trap handler execution
.macro SAVE_XNACK_MASK
  .if .amdgcn.gfx_generation_minor == 5
      s_getreg_b32  ttmp4, hwreg(HW_REG_XNACK_MASK)         // Read full 32-bit XNACK_MASK (ttmp4 is available)
      // Save bits [13:0] to ttmp11[13:0]
      s_and_b32     ttmp5, ttmp4, TTMP11_XNACK_MASK_LO_MASK // ttmp5 is available
      s_andn2_b32   ttmp11, ttmp11, TTMP11_XNACK_MASK_LO_MASK
      s_or_b32      ttmp11, ttmp11, ttmp5
      // Save bits [17:14] to ttmp11[31:28]
      s_bfe_u32     ttmp5, ttmp4, (14 | (4 << 16))          // Extract bits [17:14]
      s_lshl_b32    ttmp5, ttmp5, TTMP11_XNACK_MASK_HI_SHIFT
      s_andn2_b32   ttmp11, ttmp11, TTMP11_XNACK_MASK_HI_MASK
      s_or_b32      ttmp11, ttmp11, ttmp5
      // Save remaining bits [31:18] to ttmp13[13:0]
      s_bfe_u32     vcc_hi, ttmp4, (18 | (14 << 16))       // Extract bits ttmp4[31:18]
      s_andn2_b32   ttmp13, ttmp13, 0x3FFF                 // Clear ttmp13 [13:0] of ttmp13
      s_or_b32      ttmp13, ttmp13, vcc_hi
  .endif
.endm
// Macro to restore XNACK_MASK after VMEM operations (GFX12.5 only)
.macro RESTORE_XNACK_MASK
  .if .amdgcn.gfx_generation_minor == 5
      // Reconstruct XNACK_MASK from stored pieces
      s_and_b32     ttmp2, ttmp11, TTMP11_XNACK_MASK_LO_MASK    // Get bits [13:0] (ttmp2 safe at exit)
      s_bfe_u32     ttmp4, ttmp11, (TTMP11_XNACK_MASK_HI_SHIFT | (4 << 16))  // Get bits [31:28] from ttmp11
      s_lshl_b32    ttmp4, ttmp4, 14                        // Shift to position [17:14]
      s_or_b32      ttmp2, ttmp2, ttmp4                     // Combine bits [17:0]
      s_bfe_u32     vcc_hi, ttmp13, (0 | (14 << 16))        // Extract remaining XNACK_MASK bits from ttmp13[13:0]
      s_lshl_b32    ttmp4, vcc_hi, 18                       // Shift vcc_hi to position [31:18]
      s_or_b32      ttmp2, ttmp2, ttmp4                     // Final 32-bit XNACK_MASK
      s_setreg_b32  hwreg(HW_REG_XNACK_MASK), ttmp2         // Restore XNACK_MASK
  .endif
.endm

// Macro to save MSB bits from MODE[19:12] before using v[0:3] in trap handler (GFX12.5 only)
// The MODE register contains VGPR bank selection bits that must be preserved.
// Saves DST[13:12], SRC0[15:14], SRC1[17:16], SRC2[19:18] VGPR_MSB fields
// Storage: ttmp13[21:14] (these bits are free and don't conflict with flags in bits 22,23,31)
.macro SAVE_MODE_VGPR_MSBS
  .if .amdgcn.gfx_generation_minor == 5
      s_getreg_b32  vcc_hi, hwreg(HW_REG_MODE, 12, 8)       // Read MODE[19:12] (VGPR MSB bits of MODE register)
      s_and_b32     vcc_hi, vcc_hi, 0xFF                     // Mask to get only 8 bits
      s_lshl_b32    vcc_hi, vcc_hi, 14                       // Shift vcc_hi to position [21:14]
      s_and_b32     ttmp13, ttmp13, 0xFFC03FFF               // Clear bits [21:14] of ttmp13
      s_or_b32      ttmp13, ttmp13, vcc_hi                   // Store 8 VGPR MSB bits to ttmp13[21:14]
      s_mov_b32     vcc_hi, 0                                // Clear vcc_hi to prepare for MODE register reset
      s_setreg_b32  hwreg(HW_REG_MODE, 12, 8), vcc_hi        // Reset MODE[19:12]
  .endif
.endm

// Macro to restore VGPR bits [19:12] of MODE register before returning to user shader (GFX12.5 only)
.macro RESTORE_MODE_VGPR_MSBS
  .if .amdgcn.gfx_generation_minor == 5
      s_bfe_u32     vcc_hi, ttmp13, (14 | (8 << 16))  // Get 8 MSB bits from ttmp13[21:14]
      s_setreg_b32  hwreg(HW_REG_MODE, 12, 8), vcc_hi // Restore MODE[19:12] (VGPR MSB bits of MODE register)
  .endif
.endm

  // Macro to store the Correlation ID (Dispatch ID and Doorbell ID) into the current sample slot
  //
  // Assumes the following registers are set before it is called:
  //   v[0:1]:Must contain the 64-bit base address of the target sample slot
  //   ttmp8 :Must contain the dispatch ID in bits [24:0]
  //   exec  :Must be set to 0x1 to ensure operations apply only to lane 0
  //
  // Clobbers the following registers:
  //   v[2:3]:Used for [dispatch_id, doorbell_id]
  //   s_scratch:Used to stash the doorbell ID value.
.macro STORE_CORRELATION_ID s_scratch
  v_mov_b32         v2, 0
  v_mov_b32         v3, 0
  s_sendmsg_rtn_b32 \s_scratch, sendmsg(MSG_RTN_GET_DOORBELL)
  s_wait_kmcnt      0
  s_and_b32         \s_scratch, \s_scratch, DOORBELL_ID_MASK
  v_writelane_b32   v3, \s_scratch, 0
  s_and_b32         \s_scratch, ttmp8, TTMP8_DISPATCH_ID_MASK
  v_writelane_b32   v2, \s_scratch, 0
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_CORRELATION, scope:SCOPE_SYS
.endm

  // Macro to store the HW_ID registers into the current sample slot
  //
  // Assumes the following registers are set before it is called:
  //   v[0:1]: Must contain the 64-bit base address of the target sample slot.
  //   exec  : Must be set to 0x1 to ensure operations apply only to lane 0.
  //
  // Clobbers the following registers:
  //   v[2:3]: Used to stash the data for the global store.
  //  s_scratch1: Used to stash HW_ID1 and HW_ID2 values.
  //  NOTE: GFX12.5+ version uses exec_lo as temporary holder to avoid needing s_scratch2
  //        This preserves exec_hi completely (for XNACK_MASK storage)
.macro STORE_HW_ID s_scratch1
  // Current ROCr API determines single dword for HW_ID, while this information is scattered across two
  // dword registers HW_ID1 and HW_ID2 on GFX10+ architectures.
  // Thus, we combine values from HW_ID1 and HW_ID2 into a single dword HW_ID with the following layout:
  // WAVE_ID[4:0]
  // QUEUE_ID[8:5]
  // RESERVED [9]
  // WGP_ID[13:10]
  // SIMD_ID[15:14]
  // SA_ID[16]
  // ME_ID[17]
  // SE_ID[19:18]
  // PIPE_ID[21:20]
  // RESERVED [22]
  // WG_ID[27:23]
  // VM_ID[31:28]

  // Note: We don't show DP_RATE and STATE_ID that are useless for compute kernels
  // Also, we reduced SE_ID to 2 bits as there's only a maximum of 4 SEs on existing gfx12.0 parts
  // Finally, ME_ID is reduced to 1 bit as wavefronts are dispatched from either ME0 or ME1 in gfx12.
  // Bits 9 and 22 are reserved for a future use.
  v_mov_b32         v2, 0
  v_mov_b32         v3, 0
  s_getreg_b32      \s_scratch1, hwreg(HW_REG_HW_ID1)
  v_and_b32         v2, \s_scratch1, 0x1feffcff               // Mask to extract fields from HW_ID1 (WAVE_ID, WGP_ID, SA_ID, SE_ID on GFX12.0)
  v_and_b32         v3, \s_scratch1, 0x00000300               // Mask to extract SIMD_ID[9:8]
  v_lshl_or_b32     v2, v3, 6, v2                             // Shift SIMD_ID to bits [15:14]
  s_getreg_b32      \s_scratch1, hwreg(HW_REG_HW_ID2)         // Get HW_ID2
  v_and_b32         v3, \s_scratch1, 0x0f000000               // Mask to extract WAVE_ID[27:24]
  v_lshl_or_b32     v2, v3, 4, v2                             // Shift WAVE_ID to bits [4:0]
  v_and_b32         v3, \s_scratch1, 0x001f0000               // Mask to extract WG_ID[20:16]
  v_lshl_or_b32     v2, v3, 7, v2                             // Shift WG_ID to bits [27:23]
  v_and_b32         v3, \s_scratch1, 0x00000100               // Mask to extract ME_ID[8]
  v_lshl_or_b32     v2, v3, 9, v2                             // Shift ME_ID to bit [17]
  v_and_b32         v3, \s_scratch1, 0x00000030               // Mask to extract PIPE_ID[5:4]
  v_lshl_or_b32     v2, v3, 16, v2                            // Shift PIPE_ID to bits [21:20]
  v_and_b32         v3, \s_scratch1, 0x0000000f               // Mask to extract QUEUE_ID[3:0]
  v_lshl_or_b32     v2, v3, 5, v2                             // Shift QUEUE_ID to bits [8:5]

.if .amdgcn.gfx_generation_minor >= 5
  // For gfx12.5+, get SE_ID and chiplet from MSG_RTN_GET_SE_AID_ID.
  // Regspec MSG_RTN_GET_SE_AID_ID (0x87) returns:
  //   data[3:0]   = SE_ID (4 bits)
  //   data[19:16] = Virtual_XCC_ID (4 bits)
  //
  // Use exec_lo as temporary holder to avoid needing second scratch register
  // This keeps exec_hi completely preserved with XNACK_MASK[31:18]
  // EXEC_LO starts at 0x1 and is restored to 0x1 after each use
  
  s_sendmsg_rtn_b32 \s_scratch1, sendmsg(MSG_RTN_GET_SE_AID_ID)
  v_and_b32         v2, 0xFFC3FFFF, v2                        // Clear SE_ID bits [21:18] in v2 while waiting
  s_wait_kmcnt      0
  
  s_mov_b32         exec_lo, \s_scratch1
  
  // Extract and position SE_ID bits
  s_bfe_u32         \s_scratch1, exec_lo, (0 | (2 << 16))     // Extract lower 2 bits from the SE_ID[3:0] from exec_lo
  s_lshl_b32        \s_scratch1, \s_scratch1, 18              // Shift to bit position 18
  s_mov_b32         exec_lo, 0x1                              // Restore exec_lo to 0x1
  v_or_b32          v2, \s_scratch1, v2                       // OR the new SE_ID bits into v2

  // Construct and store chiplet_and_wave_id bitfield
  // Bitfield layout: wave_in_wg[5:0] | reserved_wg[7:6] | chiplet[10:8] | reserved[31:11]
  s_sendmsg_rtn_b32 \s_scratch1, sendmsg(MSG_RTN_GET_SE_AID_ID)
  
  // Extract wave_in_wg while waiting for sendmsg
  s_bfe_u32         exec_lo, ttmp8, (WAVE_ID_WG_BIT_POSITION | (5 << 16)) // Extract 5 bits of wave_in_wg from ttmp8[29:25]
  s_and_b32         exec_lo, exec_lo, 0x3F                    // Mask to bits [5:0]
  
  s_wait_kmcnt      0
  
  // Extract and position chiplet (Virtual_XCC_ID)
  s_bfe_u32         \s_scratch1, \s_scratch1, (16 | (3 << 16)) // Extract lower 3 bits from Virtual_XCC_ID[19:16]
  s_lshl_b32        \s_scratch1, \s_scratch1, 8                // Shift to bit position [10:8]
  
  // Combine chiplet and wave_in_wg
  s_or_b32          \s_scratch1, \s_scratch1, exec_lo          // Combine chiplet[10:8] with wave_in_wg[5:0]
  
  v_writelane_b32   v3, \s_scratch1, 0                         // Store bitfield in v3
  s_mov_b32         exec_lo, 0x1                               // Restore exec_lo to 0x1
  global_store_b32  v[0:1], v3, off, offset:SAMPLE_OFF_BITFIELD, scope:SCOPE_SYS
.endif

  global_store_b32  v[0:1], v2, off, offset:SAMPLE_OFF_HW_ID, scope:SCOPE_SYS
.endm

// ABI (Application Binary Interface) between first and second-level trap handler:
//   ttmp0: PC_LO[31:0] (Program Counter Low)
//   ttmp1: TrapID[3:0], SCHED_MODE[1:0], 0[9:0], PC_HI[15:0] (Program Counter High)
//   ttmp11: ?[7:0], DebugEnabled[0], PRESERVED[15:0], ?[6:0]
//   ttmp12: SQ_WAVE_STATE_PRIV (Private wave state register value).
//   ttmp14: TMA[31:0] - TMA_LO (Trap Memory Argument Low - base address for trap handler data, low 32 bits).
//   ttmp15: TTMA[63:32] - TMA_HI (Trap Memory Argument High - base address for trap handler data, high 32 bits).
//   For PC Sampling, this points to pcs_hosttrap_data_ or pcs_stochastic_data_
 trap_entry:
 .if .amdgcn.gfx_generation_minor == 0
    // Save SCHED_MODE from ttmp1[27:26] into ttmp11[27:26]. We will restore it on exit
    s_andn2_b32         ttmp11, ttmp11, TTMP11_SCHED_MODE_MASK
    s_and_b32           ttmp2,  ttmp1, TTMP1_SCHED_MODE_MASK
    s_or_b32            ttmp11, ttmp11, ttmp2
  .endif
  s_mov_b32           ttmp3, 0

.check_hosttrap:

  // ttmp[14:15] points to TMA.
  // Scratch registers: ttmp[2:3], ttmp[4:5], ttmp10, ttmp13
  s_getreg_b32      ttmp2, hwreg(HW_REG_EXCP_FLAG_PRIV)     // On gfx12, EXCP_FLAG_PRIV.b7
  s_bitcmp1_b32     ttmp2, SQ_WAVE_EXCP_FLAG_PRIV_HT_SHIFT  // Test Host Trap bit.
  s_cbranch_scc0    .check_stochastic                       // If not HT, check for stochastic sampling

  // It's a Host Trap event.
  s_load_b64        ttmp[14:15], ttmp[14:15], 0x0, scope:SCOPE_CU  // ttmp[14:15]=*host_trap_buffers
  s_bitset1_b32     ttmp13, TTMP13_HT_FLAG_BIT_SHIFT         // set bit 22 in TTMP13

  // Clear the Host Trap flag in the hardware register to acknowledge the event
  s_setreg_imm32_b32 hwreg(HW_REG_EXCP_FLAG_PRIV, SQ_WAVE_EXCP_FLAG_PRIV_HT_SHIFT,1), 0
  s_wait_kmcnt      0                                       // Ensure previous load is complete.
  s_branch          .profile_trap_handlers

.check_stochastic:
  s_getreg_b32      ttmp2, hwreg(HW_REG_EXCP_FLAG_PRIV)     // EXCP_FLAG_PRIV.b10=stochastic_sample_trap
  s_bitcmp1_b32     ttmp2, SQ_WAVE_EXCP_FLAG_PRIV_PERF_SNAPSHOT_SHIFT // Test Performance Snapshot bit.

  s_cbranch_scc0    .handle_sw_trap                         // If not Stochastic, continue to check trap ID

  s_load_b64        ttmp[14:15], ttmp[14:15], 0x8, scope:SCOPE_CU  // ttmp[14:15]=*stoch_trap_buf
  s_wait_kmcnt      0

  s_bitset1_b32     ttmp13, TTMP13_STOCH_FLAG_BIT_SHIFT      // set bit 21 in TTMP13

  s_setreg_imm32_b32 hwreg(HW_REG_EXCP_FLAG_PRIV, SQ_WAVE_EXCP_FLAG_PRIV_PERF_SNAPSHOT_SHIFT,1), 0 // Clear the perf_snapshot flag
  s_branch          .profile_trap_handlers

.handle_sw_trap:
  // Check if this is a trap (s_trap instruction) or a hardware exception.
  // Extract TrapID from ttmp1 (which contains PC_HI).
  // Branch if not a trap (an exception instead).
  s_bfe_u32         ttmp2, ttmp1, SQ_WAVE_PC_HI_TRAP_ID_BFE // ttmp2 = TrapID
  s_cbranch_scc0    .check_exceptions			    // If TrapID is 0, it's an exception, so branch.

  // If caused by s_trap then advance PC, then figure out the trap ID:
  // - if trapID is DEBUGTRAP and debugger is attach, report WAVE_TRAP,
  // - if trapID is ABORTTRAP, report WAVE_ABORT,
  // - report WAVE_TRAP for any other trap ID.
  s_add_u32         ttmp0, ttmp0, 0x4                       // PC_LO += 4
  s_addc_u32        ttmp1, ttmp1, 0x0                       // PC_HI += carry.

  // If llvm.debugtrap and debugger is not attached.
  s_cmp_eq_u32      ttmp2, TRAP_ID_DEBUGTRAP
  s_cbranch_scc0    .not_debug_trap

  s_bitcmp1_b32     ttmp11, TTMP11_DEBUG_ENABLED_SHIFT
  s_cbranch_scc0    .check_exceptions
  s_or_b32          ttmp3, ttmp3, EC_QUEUE_WAVE_TRAP_M0

.not_debug_trap:
  s_cmp_eq_u32      ttmp2, TRAP_ID_ABORT
  s_cbranch_scc0    .not_abort_trap
  s_or_b32          ttmp3, ttmp3, EC_QUEUE_WAVE_ABORT_M0
  s_branch          .check_exceptions

.not_abort_trap:
  s_or_b32          ttmp3, ttmp3, EC_QUEUE_WAVE_TRAP_M0

  s_bitcmp1_b32     ttmp8, TTMP8_DEBUG_FLAG_SHIFT
  s_cbranch_scc0    .check_exceptions

  // We need to explicitly look for all exceptions we want to report to the
  // host:
  // - EXCP_FLAG_PRIV.XNACK_ERROR (&& EXCP_FLAG_PRIV.MEMVIOL)
  //                                                 -> WAVE_MEMORY_VIOLATION
  // - EXCP_FLAG_PRIV.MEMVIOL (and !EXCP_FLAG_PRIV.XNACK_ERROR)
  //                                                 -> WAVE_APERTURE_VIOLATION
  // - EXCP_FLAG_PRIV.ILLEGAL_INST                   -> WAVE_ILLEGAL_INSTRUCTION
  // - EXCP_FLAG_PRIV.WAVE_START                     -> WAVE_TRAP
  // - EXCP_FLAG_PRIV.WAVE_END && TRAP_CTRL.WAVE_END -> WAVE_TRAP
  // - TRAP_CTRL.TRAP_AFTER_INST                     -> WAVE_TRAP
  // - EXCP_FLAG_PRIV.ADDR_WATCH && TRAP_CTL.WATCH   -> WAVE_TRAP
  // - (EXCP_FLAG_USER[ALU] & TRAP_CTRL[ALU]) != 0   -> WAVE_MATH_ERROR
.check_exceptions:
  s_getreg_b32      ttmp2, hwreg(HW_REG_EXCP_FLAG_PRIV)
  s_getreg_b32      ttmp13, hwreg(HW_REG_TRAP_CTRL)

  s_bitcmp1_b32     ttmp2, SQ_WAVE_EXCP_FLAG_PRIV_XNACK_ERROR_SHIFT
  s_cbranch_scc0    .not_memory_violation
  s_or_b32          ttmp3, ttmp3, EC_QUEUE_WAVE_MEMORY_VIOLATION_M0

  // Aperture violation requires XNACK_ERROR == 0.
  s_branch          .not_aperture_violation

.not_memory_violation:
  s_bitcmp1_b32     ttmp2, SQ_WAVE_EXCP_FLAG_PRIV_MEMVIOL_SHIFT
  s_cbranch_scc0    .not_aperture_violation
  s_or_b32          ttmp3, ttmp3, EC_QUEUE_WAVE_APERTURE_VIOLATION_M0

.not_aperture_violation:
  s_bitcmp1_b32     ttmp2, SQ_WAVE_EXCP_FLAG_PRIV_ILLEGAL_INST_SHIFT
  s_cbranch_scc0    .not_illegal_instruction
  s_or_b32          ttmp3, ttmp3, EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION_M0

.not_illegal_instruction:
  s_bitcmp1_b32     ttmp2, SQ_WAVE_EXCP_FLAG_PRIV_WAVE_START_SHIFT
  s_cbranch_scc0    .not_wave_end
  s_or_b32          ttmp3, ttmp3, EC_QUEUE_WAVE_TRAP_M0

.not_wave_start:
  s_bitcmp1_b32     ttmp2, SQ_WAVE_EXCP_FLAG_PRIV_WAVE_END_SHIFT
  s_cbranch_scc0    .not_wave_end
  s_bitcmp1_b32     ttmp13, SQ_WAVE_TRAP_CTRL_WAVE_END_SHIFT
  s_cbranch_scc0    .not_wave_end
  s_or_b32          ttmp3, ttmp3, EC_QUEUE_WAVE_TRAP_M0

.not_wave_end:
  s_bitcmp1_b32     ttmp13, SQ_WAVE_TRAP_CTRL_TRAP_AFTER_INST
  s_cbranch_scc0    .not_trap_after_inst
  s_or_b32          ttmp3, ttmp3, EC_QUEUE_WAVE_TRAP_M0

.not_trap_after_inst:
  s_and_b32         ttmp2, ttmp2, SQ_WAVE_EXCP_FLAG_PRIV_ADDR_WATCH_MASK
  s_cbranch_scc0    .not_addr_watch
  s_bitcmp1_b32     ttmp13, SQ_WAVE_TRAP_CTRL_ADDR_WATCH_SHIFT
  s_cbranch_scc0    .not_addr_watch
  s_or_b32          ttmp3, ttmp3, EC_QUEUE_WAVE_TRAP_M0

.not_addr_watch:
  s_getreg_b32      ttmp2, hwreg(HW_REG_EXCP_FLAG_USER, SQ_WAVE_EXCP_FLAG_USER_MATH_EXCP_SHIFT, SQ_WAVE_EXCP_FLAG_USER_MATH_EXCP_SIZE)
  s_and_b32         ttmp13, ttmp13, SQ_WAVE_TRAP_CTRL_MATH_EXCP_MASK
  s_and_b32         ttmp2, ttmp2, ttmp13
  s_cbranch_scc0    .not_math_exception
  s_or_b32          ttmp3, ttmp3, EC_QUEUE_WAVE_MATH_ERROR_M0

.not_math_exception:
  s_cmp_eq_u32      ttmp3, 0
  // This was not a s_trap we are interested in or an exception, return to
  // the user code.
  s_cbranch_scc1    .exit_trap

.send_interrupt:
  // Fetch doorbell id for our queue.
  s_sendmsg_rtn_b32 ttmp2, sendmsg(MSG_RTN_GET_DOORBELL)
  s_wait_kmcnt      0
  s_and_b32         ttmp2, ttmp2, DOORBELL_ID_MASK
  s_or_b32          ttmp3, ttmp2, ttmp3

.if .amdgcn.gfx_generation_minor == 0
  // Save trap id and halt status in ttmp6.
  s_andn2_b32       ttmp6, ttmp6, (TTMP6_SAVED_TRAP_ID_MASK | TTMP6_SAVED_STATUS_HALT_MASK)
  s_bfe_u32         ttmp2, ttmp1, SQ_WAVE_PC_HI_TRAP_ID_BFE
  s_min_u32         ttmp2, ttmp2, 0xF
  s_lshl_b32        ttmp2, ttmp2, TTMP6_SAVED_TRAP_ID_SHIFT
  s_or_b32          ttmp6, ttmp6, ttmp2
  s_bfe_u32         ttmp2, ttmp12, SQ_WAVE_STATE_PRIV_HALT_BFE
  s_lshl_b32        ttmp2, ttmp2, TTMP6_SAVED_STATUS_HALT_SHIFT
  s_or_b32          ttmp6, ttmp6, ttmp2
.elseif .amdgcn.gfx_generation_minor == 5
  // Save halt status in ttmp6
  s_andn2_b32       ttmp6, ttmp6, TTMP6_SAVED_STATUS_HALT_MASK
  s_bfe_u32         ttmp2, ttmp12, SQ_WAVE_STATE_PRIV_HALT_BFE
  s_lshl_b32        ttmp2, ttmp2, TTMP6_SAVED_STATUS_HALT_SHIFT
  s_or_b32          ttmp6, ttmp6, ttmp2

  // Save the trap id in ttmp11
  s_andn2_b32       ttmp11, ttmp11, TTMP11_SAVED_TRAP_ID_MASK
  s_bfe_u32         ttmp2, ttmp1, SQ_WAVE_PC_HI_TRAP_ID_BFE
  s_min_u32         ttmp2, ttmp2, 0xF
  s_lshl_b32        ttmp2, ttmp2, TTMP11_SAVED_TRAP_ID_SHIFT
  s_or_b32          ttmp11, ttmp11, ttmp2
.endif

  // m0 = interrupt data = (exception_code << DOORBELL_ID_SIZE) | doorbell_id
  s_mov_b32         ttmp2, m0
  s_mov_b32         m0, ttmp3
  s_sendmsg         sendmsg(MSG_INTERRUPT)
  // Wait for the message to go out.
  s_wait_kmcnt      0
  s_mov_b32         m0, ttmp2

.if .amdgcn.gfx_generation_minor == 0
  // Parking the wave requires saving the original pc in the preserved ttmps.
  // Register layout before parking the wave:
  //
  // ttmp10: ?[31:0]
  // ttmp11: 1st_level_ttmp11[31:28] SCHED_MODE[1:0] 1st_level_ttmp11[25:23] 0[15:0] 1st_level_ttmp11[6:0]
  //
  // After parking the wave:
  //
  // ttmp10: pc_lo[31:0]
  // ttmp11: 1st_level_ttmp11[31:28] SCHED_MODE[1:0] 1st_level_ttmp11[25:23] pc_hi[15:0] 1st_level_ttmp11[6:0]
  //
  // Save the PC
  s_mov_b32         ttmp10, ttmp0
  s_and_b32         ttmp1, ttmp1, SQ_WAVE_PC_HI_ADDRESS_MASK
  s_lshl_b32        ttmp1, ttmp1, TTMP_PC_HI_SHIFT
  s_andn2_b32       ttmp11, ttmp11, (SQ_WAVE_PC_HI_ADDRESS_MASK << TTMP_PC_HI_SHIFT)
  s_or_b32          ttmp11, ttmp11, ttmp1

  // Park the wave
  s_getpc_b64       [ttmp0, ttmp1]
  s_add_u32         ttmp0, ttmp0, .parked - .
  s_addc_u32        ttmp1, ttmp1, 0x0
.endif

.halt_wave:
  // Halt the wavefront upon restoring STATUS below.
  s_bitset1_b32     ttmp6, TTMP6_WAVE_STOPPED_SHIFT
  s_bitset1_b32     ttmp12, SQ_WAVE_STATE_PRIV_HALT_SHIFT

  // Initialize TTMP registers
  s_bitcmp1_b32     ttmp8, TTMP8_DEBUG_FLAG_SHIFT
  s_cbranch_scc1    .ttmps_initialized
  s_mov_b32         ttmp4, 0
  s_mov_b32         ttmp5, 0
  s_bitset1_b32     ttmp8, TTMP8_DEBUG_FLAG_SHIFT
.ttmps_initialized:
  s_branch          .exit_trap

.profile_trap_handlers:
  // Register state at the start of profile_trap_handlers:
  //
  // ttmp0:  PC_LO[31:0] - Contains program counter low bits
  // ttmp1: For gfx_generation_minor >= 5 (57-bit PC)
  //        PC_HI[24:0] - Contains program counter high bits
  //        Otherwise PC_HI[15:0] - Contains program counter high bits (47-bit PC)
  // ttmp2: Available - Can be freely used
  // ttmp3: Contains exception flags set earlier
  // ttmp4:  Available - Can be freely used
  // ttmp5:  Available - Can be freely used
  // ttmp6: For gfx_generation_minor >= 5
  //          Contains workgroup info if clusters are enabled. Not safe to use as scratch
  //        Else - Initially contains flags  - trap ID and halt status
  //               Reused after saving
  // ttmp7:  Contains WGID_Y in high 16 bits, WGID_Z in low 16 bits
  // ttmp8: For gfx_generation_minor >= 5
  //          Contains dispatch ID in bits [24:0] and debug flag. Not safe to use as scratch
  //        Else - Contains dispatch ID in bits [24:0] and debug flag
  //               Reused after saving
  // ttmp9:  Contains WGID_X
  // ttmp10: Available - Used next to save exec_lo
  //         For gfx12.0, ttmp10 holds persistent state used by the debugger,
  //         so PC sampling should not be used simultaneously with the debugger.
  // ttmp11: For gfx_generation_minor >= 5
  //          Contains debug/XNACK flags. Bits [13:0] and [31:28] are usable.
  //         Else - Contains debug flags - Used next to save exec_hi
  // ttmp12: Contains SQ_WAVE_STATE_PRIV
  // ttmp13: Contains flag bits for sampling type - HT_FLAG_BIT or STOCH_FLAG_BIT
  //         - TTMP13_HT_FLAG_BIT_SHIFT (bit 22) for host trap
  //         - TTMP13_STOCH_FLAG_BIT_SHIFT (bit 23) for stochastic sampling
  //         - TTMP13_BUF_ID_BIT_POSITION (bit 31) for buffer selection
  //         For gfx12.5:
  //         - Bits [13:0] used to save XNACK_MASK[31:18]
  //         - Bits [21:14] used to save MODE VGPR MSBs
  // ttmp14: Contains TMA_LO - base address for trap handler data, low 32 bits
  // ttmp15: Contains TMA_HI - base address for trap handler data, high 32 bits
  //
  // v[0:3] contain user shader data that must be preserved/restored
  // exec: Contains user's execution mask

  s_mov_b32         ttmp10, exec_lo                         // Save exec_lo to ttmp10
.if .amdgcn.gfx_generation_minor >= 5
  s_mov_b32 exec_hi, 0x0                                    // wave32 mode: exec_hi unused by hardware, safe as scratch
  SAVE_XNACK_MASK
  SAVE_MODE_VGPR_MSBS                                       // Save 8 VGPR bits of MODE[19:12] to ttmp13[7:0]
 .else
  s_mov_b32 ttmp11, exec_hi
  s_mov_b32 exec_hi, 0x0
.endif

  s_mov_b32 exec_lo, 0x1                                    // turn on lane 0 only

  v_readlane_b32    ttmp2, v0, 0                            // Save out lane 0's first VGPR
  v_readlane_b32    ttmp3, v1, 0                            // Save out lane 0's second VGPR
  // At this point, ttmp[4:5],v[0:1] are free.
  // For amdgcn.gfx_generation_minor >= 5, ttmp6[27:0] contains workgroup cluster info (protected).
  // Only ttmp6[31:28] can be modified for halt status storage.
  // Atomically get current sample slot index and select buffer
  // pcs_sampling_data_t.buf_write_val (uint64_t) stores:
  //   Bit 63: current_buffer_id (0 or 1)
  //   Bits 62-0: current_sample_index_in_buffer
  // v0 = 1 (value to add to the low part of buf_write_val)
  // v1 = 0 (value to add to the high part of buf_write_val, bit 63 is buffer selector)

  v_mov_b32         v0, 1
  v_mov_b32         v1, 0
  global_atomic_add_u64 v[0:1], v1, v[0:1], ttmp[14:15], scope:SCOPE_SYS th:TH_ATOMIC_RETURN
  s_wait_loadcnt    0                                       // Wait for atomic operation to complete and return value

  // At this point, ttmp[4:5] is free. ttmp6 is free if amdgcn.gfx_generation_minor < 5
  // v[0:1] (lane 0) now holds the previous value of buf_write_val.
  // This previous value gives the slot index for the current sample.

  v_readlane_b32    ttmp4, v1, 0x0                          // ttmp4 = high 32 bits of previous buf_write_val[63:32], i.e., bit 63 of previous buf_write_val
  s_lshr_b32        ttmp4, ttmp4, 31                        // ttmp4 = previous_buffer_id (0 or 1, from bit 63 of original uint64_t)
                                                            // This ttmp4 is used to select which buffer's metadata (size, watermark, signal) to use.
                                                            // It's also used to calculate the base address of the sample buffer.
.if .amdgcn.gfx_generation_minor < 5
  s_mov_b32         ttmp6, ttmp4                            // GFX12.0: Save buffer_id to ttmp6 (scratch) before ttmp4 is reused
.endif
  s_bitset0_b32     ttmp13, TTMP13_BUF_ID_BIT_POSITION       // Clear our local buffer full flag for now
  s_cmp_eq_u32      ttmp4, 0                                // store off buf_to_use
  s_cbranch_scc1    .skip_bufbit_set                        // into bit31 of ttmp13
  s_bitset1_b32     ttmp13, TTMP13_BUF_ID_BIT_POSITION

.skip_bufbit_set:
  // ttmp[2:3]=v[0:1]-backup, ttmp[4:5]=free
  // ttmp[10:11]=EXEC backup. ttmp[14:15]=tma
  // v[0:1].lane0=local_entry, v[2:3]=original, EXEC=0x1

  v_bfe_u32         v1, v1, 0, SAMPLE_INDEX_WIDTH           // v[0:1] = new local_entry
                                                            // removes bit 31 from v1, returning v1 & 0x7FFFFFFF.
  v_readlane_b32    ttmp5, v1, 0                            // ttmp5 = high 31 bits of sample index (if index > 2^32-1).
  s_cmp_lg_u32      ttmp5, 0                                // Check if sample index is very large (overflowed 32 bits).

  s_cbranch_scc1    .lost_sample                            // If ttmp5 > 0, index is too large, treat as lost sample.

  s_load_b32        ttmp5, ttmp[14:15], SAMPLE_OFF_BUF_SIZE, scope:SCOPE_CU // ttmp5 = pcs_sampling_data_t.buf_size
  v_readlane_b32    ttmp4, v0, 0                            // ttmp4 = sample_index_for_current_sample (from v0)
  s_wait_kmcnt      0                                       // Wait for buf_size load.

  s_cmp_ge_u32      ttmp4, ttmp5                            // if local_entry >= buf_size
  s_cbranch_scc1    .lost_sample                            // If index >= buf_size, buffer is full, sample is lost.
                                                            // This also sets TTMP13_BUF_ID_BIT_POSITION implicitly by branching.

  // Register state before calculating the sample buffer address:
  // ttmp2 = backup of original shader's v0
  // ttmp3 = backup of original shader's v1
  // ttmp4 = sample_index_for_current_sample (from v0)
  // ttmp5 = buf_size
  // For GFX12.0: ttmp6 = buffer_id (0 or 1)
  // For GFX12.5: ttmp6 contains protected workgroup info, buffer_id stored in ttmp13.b31
  // ttmp[10:11] = original shader's [exec_lo, exec_hi]
  // ttmp[14:15] = base_address_of_pcs_sampling_data_t (TMA)
  // ttmp13.b31 = buffer_id (0 or 1, same as ttmp6 for GFX12.0)
  // v[0:1].lane0 = sample index value from atomic
  // v[2:3] = original user shader's v[2:3] values
  // exec = backup of user shader's v[0:1]
  s_mov_b64         exec, ttmp[2:3]                          // stash into EXEC to free up ttmp[2:3]

  // Calculate the base address of the selected sample buffer (buffer0 or buffer1).
  // The buffers are located after the pcs_sampling_data_t struct header (0x40 bytes).
  // Formula: TMA + 0x40 + (buffer_id * buf_size * 64_bytes_per_sample)
.if .amdgcn.gfx_generation_minor >= 5
  s_mov_b32         vcc_hi, 0                                // Initialize vcc_hi to 0
   // Get buffer_id (0 or 1) from ttmp13 bit 31 into a scratch register.
  s_bfe_u32         vcc_hi, ttmp13, (TTMP13_BUF_ID_BIT_POSITION | (1 << 16)) // vcc_hi = buffer_id

  // Calculate the byte offset for the selected buffer: buf_size * buffer_id
  // Result is a 64-bit value in ttmp[2:3].
  s_mul_i32         ttmp2, ttmp5, vcc_hi                   // vcc_hi = buf_size * buffer_id (low 32 bits)
  s_mul_hi_u32      ttmp3, ttmp5, vcc_hi                   // vcc_hi = buf_size * buffer_id (high 32 bits)
.else
  s_mul_i32         ttmp2, ttmp5, ttmp6                    // low 32 bits
  s_mul_hi_u32      ttmp3, ttmp5, ttmp6                    // high 32 bits
.endif

  // Multiply by 64 bytes per sample slot (shift left by 6 bits)
  // This converts from units of samples to units of bytes
  s_lshl_b64        ttmp[2:3], ttmp[2:3], 6                 // ttmp[2:3] = buf_size * buffer_id * 64
  // Add the size of the pcs_sampling_data_t header to get the total offset from TMA.
  // The sample buffers start right after the header.
  s_add_u32         ttmp2, ttmp2, SAMPLE_OFF_BYTES_PER_SAMPLE // ttmp2 = total_offset_lo = buf_size * buffer_id * 64 + SAMPLE_OFF_BYTES_PER_SAMPLE
  s_addc_u32        ttmp3, ttmp3, 0                           // ttmp3 = total_offset_hi = buf_size * buffer_id * 64 + SAMPLE_OFF_BYTES_PER_SAMPLE + carry
  // Calculate the final buffer base address: TMA + total_offset.
  // Store the result in ttmp[4:5], which are free.
  s_add_u32         ttmp4, ttmp14, ttmp2                    // ttmp4 = TMA_base_lo + total_offset_lo. This is low part of &bufferX
  s_addc_u32        ttmp5, ttmp15, ttmp3                    // ttmp5 = TMA_base_hi + total_offset_hi + carry. This is high part of &bufferX
                                                            // ttmp[4:5] now correctly points to the base of the selected sample buffer array
  // At this point: exec_lo=0x1, exec_hi=0 or XNACK_MASK[31:18]
  // GFX12.5: exec_lo contains backup of user v0, exec_hi contains XNACK_MASK[31:18]
  // GFX12.0: exec contains backup of user v[0:1] (from ttmp[2:3])
  s_bitcmp1_b32     ttmp13, TTMP13_HT_FLAG_BIT_SHIFT        // if ttmp13.b22==1, this is hosttrap
  s_cbranch_scc1    .fill_sample_ht
  s_bitcmp1_b32     ttmp13, TTMP13_STOCH_FLAG_BIT_SHIFT
  s_cbranch_scc1    .fill_sample_stoch

  s_mov_b64         ttmp[2:3], exec                         // Restore user v[0:1] backup to ttmp[2:3]
  v_readlane_b32    ttmp4, v2, 0                            // Backup user v[2:3] to ttmp[4:5] for restore.
  v_readlane_b32    ttmp5, v3, 0                            // ttmp[4:5] now holds original user v[2:3] values
  s_branch          .restore_vector_before_exit_trap

.fill_sample_ht:
  // At this point, v[0:1] is local_entry (but v1 is 0)
  // v[2:3] is original user-data
  // ttmp[2:3] is free
  // ttmp[4:5] holds &buffer
  // ttmp6 holds buf_to_use
  // ttmp[10:11] holds original shader's [exec_lo,exec_hi]
  // [ttmp14:15]=‘tma', ttmp13.b31 = buf_to_use
  // EXEC holds holds backup of original shader's v[0:1]
  V_READLANE_B32    v0, 0                                   // v[0] = local_entry (from v[0])
  S_MUL_I32         ttmp2, SAMPLE_OFF_BYTES_PER_SAMPLE      // ttmp2 = local_entry * SAMPLE_OFF_BYTES_PER_SAMPLE
  S_MUL_HI_U32      ttmp3, SAMPLE_OFF_BYTES_PER_SAMPLE      // ttmp3 = local_entry * SAMPLE_OFF_BYTES_PER_SAMPLE (high part)
  s_add_u32         ttmp2, ttmp2, ttmp4                     // ttmp2 = &bufferX[local_entry] (low part)
  s_addc_u32        ttmp3, ttmp3, ttmp5                     // ttmp[2:3]=&bufferX[local_entry]
  v_readlane_b32    ttmp4, v2, 0x0                          // ttmp[4:5] now holds backup of
  v_readlane_b32    ttmp5, v3, 0x0                          // user-data from v[2:3]
  v_writelane_b32   v0, ttmp2, 0x0                          // v[0] = &buffer[local_entry]
  v_writelane_b32   v1, ttmp3, 0x0                          // v[0:1]=&buffer[local_entry]

  s_sendmsg_rtn_b64 ttmp[2:3], sendmsg(MSG_RTN_GET_REALTIME)// Get the current timestamp
  s_wait_kmcnt      0                                       // Wait for timestamp

  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
  // ttmp[2:3] holds the thing we want to store
  // ttmp[4:5] holds backup of original shaders v[2:3]
  // ttmp6     ttmp6 is free if amdgcn.gfx_generation_minor < 5
  // ttmp[10:11] holds original shaders [exec_lo,exec_hi]
  // ttmp[14:15]=tma, ttmp13.b31 = buf_to_use
  // EXEC holds backup of original shaders v[0:1]

  v_writelane_b32   v2, ttmp2, 0                            // bring output data to v[2:3]
  v_writelane_b32   v3, ttmp3, 0                            // v[2:3] = timestamp

  // ttmp[2:3] now free after moving to v[2:3]
  s_mov_b64         ttmp[2:3], exec                         // Save user v[0:1] backup from exec to ttmp[2:3] (exec will be modified)
.if .amdgcn.gfx_generation_minor >= 5
  s_mov_b32         exec_lo, 1                              // Set exec_lo to 1 (exec_hi unused/preserved)
.else
  s_mov_b64         exec, 1                                 // Set exec to lane 0 for vector stores
.endif

  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_TIMESTAMP, scope:SCOPE_SYS // store out timestamp

  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader's v[0:1]
  // ttmp[4:5] holds backup of original shader's v[2:3]
  // ttmp6     ttmp6 is free if amdgcn.gfx_generation_minor < 5
  // ttmp[10:11] holds original shader's [exec_lo,exec_hi]
  // ttmp[14:15]=‘tma', ttmp13.b31 = buf_to_use
  // EXEC is 0x1

  s_and_b32         ttmp1, ttmp1, SQ_WAVE_PC_HI_ADDRESS_MASK // Clear out extra data from PC_HI
  v_writelane_b32   v2, ttmp0, 0                             // v[2] = PC_LO
  v_writelane_b32   v3, ttmp1, 0                             // v[3] = PC_HI
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_PC_HOST, scope:SCOPE_SYS  // store out PC

  v_writelane_b32   v2, ttmp10, 0                            // v[2] = exec_lo

.if .amdgcn.gfx_generation_minor >= 5
  v_mov_b32         v3, 0                                    // exec_hi is not used in wave32
.else
  v_writelane_b32   v3, ttmp11, 0                            // v[3] = exec_hi
.endif
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_EXEC_LOHI, scope:SCOPE_SYS  // store out original EXEC

  // Store Workgroup ID X and Y at offset SAMPLE_OFF_WGID_XY (0x10).
  // ttmp9 = WGID_X (from first-level handler).
  // ttmp7 contains WGID_Y in high 16 bits.
  v_writelane_b32   v2, ttmp9, 0                             // wg_id_x
  S_BFE_U32         ttmp7, (16<<16)                          // extract bits 31:16, wg_id_y
  V_WRITELANE_B32   v3,0                                     // wg_id_y
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_WGID_XY, scope:SCOPE_SYS  // store wg_id_x and wg_id_y

  // Store Workgroup ID Z at offset 0x18 (32-bit).
  // ttmp7 contains WGID_Z in high 16 bits [31:16].
.if .amdgcn.gfx_generation_minor >= 5
  s_bfe_u32         vcc_hi, ttmp7, (16 | (16 << 16))         // Extract WGID_Z[15:0] from ttmp7[31:16]
  v_writelane_b32   v2, vcc_hi, 0                            // Store WGID_Z in v2
.else
  s_bfe_u32         ttmp6, ttmp7, (16 | (16 << 16))          // Extract WGID_Z[15:0] from ttmp7[31:16]
  v_writelane_b32   v2, ttmp6, 0                             // Store WGID_Z in v2
.endif
  global_store_b32  v[0:1], v2, off, offset:SAMPLE_OFF_WGID_Z, scope:SCOPE_SYS

  // Note: Bitfield word (wave_in_wg and chiplet) is now stored by STORE_HW_ID macro below
  // for gfx12.5+, which extracts both from the MSG_RTN_GET_SE_AID_ID response.
  // For gfx12.0, only wave_in_wg is stored here:
.if .amdgcn.gfx_generation_minor < 5
  // Construct bitfield word at offset 0x1C for gfx12.0:
  // Bits [5:0]   = wave_in_wg (5 bits from ttmp8[29:25])
  // Bits [31:6]  = reserved = 0
  s_bfe_u32         ttmp6, ttmp8, (WAVE_ID_WG_BIT_POSITION | (5 << 16)) // Extract 5 bits (use ttmp6, not vcc_hi which has user data)
  s_and_b32         ttmp6, ttmp6, 0x3F                       // Mask to bits [5:0]
  v_writelane_b32   v2, ttmp6, 0                             // Store bitfield in v2
  global_store_b32  v[0:1], v2, off, offset:SAMPLE_OFF_BITFIELD, scope:SCOPE_SYS
.endif

  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader's v[0:1]
  // ttmp[4:5] holds backup of original shader's v[2:3]
  // ttmp6     ttmp6 is free if amdgcn.gfx_generation_minor < 5
  // ttmp[10:11] holds original shader's [exec_lo,exec_hi]
  // ttmp[14:15]=tma, ttmp13.b31 = buf_to_use
  // EXEC is 0x1

.if .amdgcn.gfx_generation_minor >= 5
  STORE_HW_ID vcc_hi                                        // Uses exec_lo internally as temp holder (preserves exec_hi)
.else
  STORE_HW_ID ttmp6
.endif

  // The following is still true
  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader's v[0:1]
  // ttmp[4:5] holds backup of original shader's v[2:3]
  // ttmp6 = free
  // ttmp[10:11] holds original shader's [exec_lo,exec_hi]
  // ttmp[14:15]=tma, ttmp13.b31 = buf_to_use
  // EXEC is 0x1

  // Store wave_in_group and chiplet information in the following format:
  // Bits [5:0]   = wave_in_wg (5 bits from ttmp8[29:25])
  // bits [10:8]  = chiplet (zero on gfx12.0)
  // Bits [7:6] and [31:11] = reserved and must be zero

  s_bfe_u32         ttmp6, ttmp8, (WAVE_ID_WG_BIT_POSITION | (5 << 16)) // Extract 5 bits
  v_writelane_b32   v2, ttmp6, 0                            // Store wave_in_group in v2

  // Write wave_in_group and chiplet (0 on gfx12.0)
  global_store_b32  v[0:1], v2, off, offset:SAMPLE_OFF_BITFIELD, scope:SCOPE_SYS

  // The following is still true as we get ready to jump to correlation ID check
  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader’s v[0:1]
  // ttmp[4:5] holds backup of original shader’s v[2:3]
  // ttmp6 = free on gfx12.0, but reserved on gfx12.5
  // ttmp[10:11] holds original shader’s [exec_lo,exec_hi]
  // ttmp[14:15=‘tma’, ttmp13.b31 = buf_to_use
  // EXEC is 0x1
  // The following is still true
  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader's v[0:1]
  // ttmp[4:5] holds backup of original shader's v[2:3]
  // ttmp6     ttmp6 is free if amdgcn.gfx_generation_minor < 5
  // ttmp[10:11] holds original shader's [exec_lo,exec_hi]
  // ttmp[14:15]=tma, ttmp13.b31 = buf_to_use
  // EXEC is 0x1

.if .amdgcn.gfx_generation_minor >= 5
    STORE_CORRELATION_ID  vcc_hi
.else
    STORE_CORRELATION_ID  ttmp6
.endif

  // Ensure all stores have completed before returning and incrementing written_val
  s_wait_storecnt   0

  // Still true after returning back from correlation ID check
  // v[0:1] = &buffer[local_entry] (needed for upcoming atomic operation)
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader's v[0:1]
  // ttmp[4:5] holds backup of original shader's v[2:3]
  // For gfx_generation_minor == 0:
  //   ttmp6 holds buf_to_use
  // For gfx_generation_minor == 5 (GFX12.5):
  //   ttmp4 holds buf_to_use, ttmp6 contains workgroup info and must not be modified
  // ttmp[10:11] holds original shader's [exec_lo,exec_hi]
  // ttmp[14:15]=tma, ttmp13.b31 = buf_to_use
  // EXEC is 0x1
  s_branch          .ret_from_fill_sample

.fill_sample_stoch:
  // v0 contains local_entry, v1 is free
  // v[2:3] is original user-data
  // ttmp[2:3] is free
  // ttmp[4:5] holds &buffer
  // ttmp6 buffer_id (0 or 1) for GFX12.0, workgroup info for GFX12.5+
  // ttmp[10:11] holds original shader's [exec_lo,exec_hi]
  // [ttmp14:15]=TMA base address, ttmp13.b31 = buf_to_use
  // EXEC  holds backup of original shader's v[0:1]
  // Calculate address of sample slot: &buffer[local_entry * 64]
  V_READLANE_B32    v0, 0x0                                   // Extract local_entry from v[0] lane 0
  S_MUL_I32         ttmp2, SAMPLE_OFF_BYTES_PER_SAMPLE        // ttmp2 = local_entry * 64 (low 32 bits)
  S_MUL_HI_U32      ttmp3, SAMPLE_OFF_BYTES_PER_SAMPLE        // ttmp3 = local_entry * 64 (high 32 bits)
  s_add_u32         ttmp2, ttmp2, ttmp4                       // Add buffer base address (low)
  s_addc_u32        ttmp3, ttmp3, ttmp5                       // Add buffer base address (high) + carry

  // Backup original user data from v[2:3] before overwriting
  v_readlane_b32    ttmp4, v2, 0x0                            // Save user v[2] to ttmp4
  v_readlane_b32    ttmp5, v3, 0x0                            // Save user v[3] to ttmp5

  // Store calculated sample address in v[0:1] for subsequent global stores
  v_writelane_b32   v0, ttmp2, 0x0                            // v[0] = sample address (low)
  v_writelane_b32   v1, ttmp3, 0x0                            // v[1] = sample address (high)
  s_sendmsg_rtn_b64 ttmp[2:3], sendmsg(MSG_RTN_GET_REALTIME)  // Request realtime clock
  s_wait_kmcnt      0                                         // Wait for timestamp

  // v[0:1] = &buffer[local_entry] (sample slot address)
  // v[2:3] = free
  // ttmp[2:3] = 64-bit timestamp value
  // ttmp[4:5] = holds backup of original shader's v[2:3]
  // ttmp6 =  free for GFX12.0, workgroup info for GFX12.5+
  // ttmp[10:11] holds original shader's [exec_lo,exec_hi]
  // ttmp[14:15] = TMA base address, ttmp13.b31 = buffer_id
  // EXEC holds backup of original shader's v[0:1]

  v_writelane_b32   v2, ttmp2, 0                              // v[2] = timestamp (low)
  v_writelane_b32   v3, ttmp3, 0                              // v[3] = timestamp (high)

  // ttmp[2:3] now free after moving to v[2:3]
  s_mov_b64         ttmp[2:3], exec                           // Save user v[0:1] backup from exec to ttmp[2:3]
.if .amdgcn.gfx_generation_minor >= 5
  s_mov_b32         exec_lo, 1
.else
  s_mov_b64         exec, 1                                   // Enable only lane 0 for vector ops
.endif

 // Store timestamp at offset 0x30 (SAMPLE_OFF_TIMESTAMP) within sample slot
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_TIMESTAMP, scope:SCOPE_SYS  // store out timestamp

  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader's v[0:1]
  // ttmp[4:5] holds backup of original shader's v[2:3]
  // ttmp6     ttmp6 is free if amdgcn.gfx_generation_minor < 5
  // ttmp[10:11] holds original shader's [exec_lo,exec_hi]
  // ttmp[14:15] = TMA base address, ttmp13.b31 = buffer_id
  // EXEC is 0x1 (lane 0 only)

  v_writelane_b32   v2, ttmp10, 0                            // v[2] = exec_lo
.if .amdgcn.gfx_generation_minor >= 5
  v_mov_b32         v3, 0                                    // GFX12.5+ uses wave32, exec_hi = 0
.else
  v_writelane_b32   v3, ttmp11, 0                            // GFX12.0 uses wave64, store exec_hi
.endif
  // Store exec state at offset 0x08 (SAMPLE_OFF_EXEC_LOHI)
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_EXEC_LOHI, scope:SCOPE_SYS

  // ttmp9 contains WGID_X, ttmp7 contains WGID_Y in upper 16 bits
  v_writelane_b32   v2, ttmp9, 0                             // wg_id_x
  S_BFE_U32         ttmp7, (0 | (16 << 16))                  // extract bits [15:0] = wg_id_y
  V_WRITELANE_B32   v3, 0                                    // wg_id_y
  // Store at offset 0x10 (SAMPLE_OFF_WGID_XY)
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_WGID_XY, scope:SCOPE_SYS

  // STORE WORKGROUP ID Z at offset 0x18 (32-bit)
  // ttmp7 contains WGID_Z in high 16 bits [31:16]
.if .amdgcn.gfx_generation_minor >= 5
  s_bfe_u32         vcc_hi, ttmp7, (16 | (16 << 16))         // Extract WGID_Z[15:0] from ttmp7[31:16]
  v_writelane_b32   v2, vcc_hi, 0                            // Store WGID_Z in v2
.else
  s_bfe_u32         ttmp6, ttmp7, (16 | (16 << 16))          // Extract WGID_Z[15:0] from ttmp7[31:16]
  v_writelane_b32   v2, ttmp6, 0                             // Store WGID_Z in v2
.endif
  global_store_b32  v[0:1], v2, off, offset:SAMPLE_OFF_WGID_Z, scope:SCOPE_SYS

  // Note: Bitfield word (wave_in_wg and chiplet) is now stored by STORE_HW_ID macro below
  // for gfx12.5+, which extracts both from the MSG_RTN_GET_SE_AID_ID response.
  // For gfx12.0, only wave_in_wg is stored here:
.if .amdgcn.gfx_generation_minor < 5
  // Construct bitfield word at offset 0x1C for gfx12.0:
  // Bits [5:0]   = wave_in_wg (5 bits from ttmp8[29:25])
  // Bits [31:6]  = reserved = 0
  s_bfe_u32         ttmp6, ttmp8, (WAVE_ID_WG_BIT_POSITION | (5 << 16)) // Extract 5 bits (use ttmp6, not vcc_hi which has user data)
  s_and_b32         ttmp6, ttmp6, 0x3F                       // Mask to bits [5:0]
  v_writelane_b32   v2, ttmp6, 0                             // Store bitfield in v2
  global_store_b32  v[0:1], v2, off, offset:SAMPLE_OFF_BITFIELD, scope:SCOPE_SYS
.endif

.if .amdgcn.gfx_generation_minor >= 5
  STORE_HW_ID vcc_hi                                        // Uses exec_lo internally as temp holder (preserves exec_hi)
.else
  STORE_HW_ID ttmp6
.endif

  // Read performance SNAPSHOT registers and store at offset 0x28 (SAMPLE_OFF_SNAPSHOT_DATA + 4)
  S_GETREG_B32      HW_REG_PERF_SNAPSHOT_DATA1              // Read snapshot data register 1
  V_WRITELANE_B32   v2, 0x0                                 // stash DATA1 in v2
  S_GETREG_B32      HW_REG_PERF_SNAPSHOT_DATA2              // Read snapshot data register 2
  V_WRITELANE_B32   v3, 0x0                                 // stash DATA2 in v3
  // Store snapshot DATA1 and DATA2 at offset SAMPLE_OFF_SNAPSHOT_DATA + 4
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_SNAPSHOT_DATA + 4, scope:SCOPE_SYS

.if .amdgcn.gfx_generation_minor >= 5
  // Store main snapshot data at offset 0x24 (SAMPLE_OFF_SNAPSHOT_DATA)
  S_GETREG_B32      HW_REG_PERF_SNAPSHOT_DATA
  V_WRITELANE_B32   v2, 0
  global_store_b32  v[0:1], v2, off, offset:SAMPLE_OFF_SNAPSHOT_DATA, scope:SCOPE_SYS  // store perf snapshot DATA
.endif

  // For stochastic sampling, use PC from snapshot registers (actual sampled instruction)
  // Trap PC points to trap handler entry, not the interrupted instruction
  S_GETREG_B32      HW_REG_PERF_SNAPSHOT_PC_LO              // Read performance snapshot PC_LO register
  V_WRITELANE_B32   v2, 0x0                                 // stash PC_LO in v2
  S_GETREG_B32      HW_REG_PERF_SNAPSHOT_PC_HI              // Read performance snapshot PC_HI register
  V_WRITELANE_B32   v3, 0x0                                 // stash PC_HI in v3

.if .amdgcn.gfx_generation_minor == 0
  // Store PERF_SNAPSHOT_DATA at offset 0x24
  // We access PERF_SNAPSHOT_DATA last on gfx12.0 as it contains valid bit indicating if the
  // sample is valid and being read by the sampled wave.
  s_getreg_b32      ttmp6, hwreg(HW_REG_PERF_SNAPSHOT_DATA)
  v_writelane_b32   v2, ttmp6, 0
  global_store_b32  v[0:1], v2, off, offset:SAMPLE_OFF_SNAPSHOT_DATA, scope:SCOPE_SYS  // store perf snapshot DATA
.endif
  // Store at offset 0x00 (SAMPLE_OFF_PC_HOST)
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_PC_HOST, scope:SCOPE_SYS

  // The following is still true as we get ready to jump to correlation ID check
  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader's v[0:1]
  // ttmp[4:5] holds backup of original shader's v[2:3]
  // ttmp6     ttmp6 is free if amdgcn.gfx_generation_minor < 5
  // ttmp[10:11] holds original shader's [exec_lo,exec_hi]
  // ttmp[14:15]=tma, ttmp13.b31 tells us buf_to_use
  // EXEC is 0x1

.if .amdgcn.gfx_generation_minor >= 5
  STORE_CORRELATION_ID  vcc_hi
.else
  STORE_CORRELATION_ID  ttmp6
.endif

  // Ensure all stores have completed before returning and incrementing written_val
  s_wait_storecnt   0

  // SAMPLE DATA COMPLETION AND BUFFER MANAGEMENT
  // This section handles incrementing the written sample count and
  // signaling the host when watermark is reached.

.ret_from_fill_sample:
  // v[0:1] = free
  // v[2:3] = free
  // ttmp[2:3] = holds backup of original shader's v[0:1]
  // ttmp[4:5] = holds backup of original shader's v[2:3]
  // ttmp6 = free for GFX12.0, protected workgroup info for GFX12.5+
  // ttmp[10:11] = holds original shader's [exec_lo,exec_hi]
  // ttmp[14:15] = TMA base address, ttmp13.b31 = buffer_id (0 or 1)
  // EXEC = 0x1

  // Calculate offset to buf_written_val for current buffer
  // buf_written_val0 at offset 0x10, buf_written_val1 at offset 0x20
  S_LSHR_B32        ttmp13, TTMP13_BUF_ID_BIT_POSITION      // Extract buffer_id from bit 31 into scratch register
  S_MULK_I32        SAMPLE_OFF_BUF_WRITTEN_VAL              // Multiply buffer_id by 16 (0x10) to get offset
  S_ADD_U32         ttmp14, ttmp14                          // Add offset to TMA base (low)
  s_addc_u32        ttmp15, ttmp15, 0                       // Add carry to TMA base (high)

  // ttmp[14:15] now points to buf_written_valX - SAMPLE_OFF_BUF_WRITTEN_VAL
  // Atomically increment the chosen buf_written_val.
  // v0 = 0 (value to add - low part), v1 = 1 (value to add - high part, effectively just adding 1 to uint32_t)

  v_mov_b32         v0, 0                                   // want to atomic increment
  v_mov_b32         v1, 1                                   // buf_written_valX
 
  // Perform atomic add and return previous value
  global_atomic_add_u32 v0, v0, v1, ttmp[14:15], offset:SAMPLE_OFF_BUF_WRITTEN_VAL, scope:SCOPE_SYS th:TH_ATOMIC_RETURN
  s_wait_loadcnt    0

  // Check Watermark and Signal Host
  // v0 = done, v1 = free, v[2:3] = free
  // ttmp[2:3] holds backup of original shader's v[0:1]
  // ttmp[4:5] holds backup of original shader's v[2:3]
  // ttmp6     ttmp6 is free if amdgcn.gfx_generation_minor < 5
  // ttmp[10:11] holds original shader's [exec_lo,exec_hi]
  // ttmp[14:15] = address of buf_written_valX (after offset addition)
  // EXEC=0x1

  s_mov_b64         exec, ttmp[4:5]                         // stash user's v[2:3] in EXEC
  s_load_b32        ttmp5, ttmp[14:15], SAMPLE_OFF_WATERMARK_FIELD, scope:SCOPE_CU            // load watermark threshold into ttmp5
  v_readlane_b32    ttmp4, v0, 0                            // Get previous written count
  s_wait_kmcnt      0                                       // wait for watermark to load

  // Check if we should signal the host
  s_cmp_lg_u32      ttmp4, ttmp5                            // Compare previous_count with watermark
  s_add_u32         ttmp4, ttmp4, 1                         // Calculate current count (previous + 1)
  s_cmp_lt_u32      ttmp4, ttmp5                            // if (current_sample_count < watermark), don't signal

  // Restore user data and execution state
  s_mov_b64         ttmp[4:5], exec                         // restore user's v[2:3]
.if .amdgcn.gfx_generation_minor >= 5
  s_mov_b32         exec_lo, 1
.else
  s_mov_b64         exec, 1                                 // Set exec to lane 0 only
.endif
  s_cbranch_scc1    .restore_vector_before_exit_trap        // Skip signaling if below watermark

  // Host signalling part when whatermark is reached
.send_signal:
  // v[0:3] = free, ttmp[2:5] = backups of original v[0:3]
  // ttmp[10:11] holds original shaders [exec_lo,exec_hi]
  // ttmp[14:15]=buf_written_valX-0x10, EXEC=0x1
  // write done-signal and optional interrupt

  // Watermark reached or exceeded. Signal the host.
  // Load the hsa_signal_t handle for the current buffer.
  // done_sig0 is at offset 0x18. done_sig1 is at 0x28.
  // addr = ttmp[14:15] + 0x18 + (buffer_id * 0x10).
  // ttmp0 still holds buffer_id * 0x10.
  s_load_b64        ttmp[14:15], ttmp[14:15], SAMPLE_OFF_DONE_SIG0, scope:SCOPE_CU // load done_sig into ttmp[14:15]
.if .amdgcn.gfx_generation_minor >= 5
  s_mov_b32         exec_lo, 1
.else
  s_mov_b64         exec, 1                                 // Set exec to lane 0 only
.endif
  s_wait_kmcnt      0                                       // Wait for done signal to load

  // Zero out the signal value to notify host
  v_mov_b32         v0, 0                                   // v[0] = 0 (value to store)
  v_mov_b32         v1, 0                                   // value to store into v[0:1]
  v_writelane_b32   v2, ttmp14, 0                           // v[2] = done signal address (low part)
  v_writelane_b32   v3, ttmp15, 0                           // Put signal address into v[2:3]

  // Write to signal value field (offset 0x8 within signal structure)
  global_store_b64  v[2:3], v[0:1], off, offset:SAMPLE_OFF_SIGNAL_VALUE, scope:SCOPE_SYS

  // Load event ID and mailbox pointer for interrupt generation
.if .amdgcn.gfx_generation_minor >= 5
  s_load_b32        vcc_hi, ttmp[14:15], SAMPLE_OFF_DONE_SIG0, scope:SCOPE_CU // load event_id into vcc_hi
.else
  s_load_b32        ttmp6, ttmp[14:15], SAMPLE_OFF_DONE_SIG0, scope:SCOPE_CU  // load event_id into ttmp6
.endif
  s_load_b64        ttmp[14:15], ttmp[14:15], SAMPLE_OFF_EVENT_MAILBOX0, scope:SCOPE_CU     // load event mailbox ptr into 14:15
  s_wait_kmcnt      0

  // Check if interrupt should be sent (null mailbox or zero event_id means no interrupt)
  s_cmp_eq_u64      ttmp[14:15], 0                          // null mailbox means no interrupt
  s_cbranch_scc1    .restore_vector_before_exit_trap

.if .amdgcn.gfx_generation_minor >= 5
  s_cmp_eq_u32      vcc_hi, 0                               // event_id zero means no interrupt
.else
  s_cmp_eq_u32      ttmp6, 0                                // event_id zero means no interrupt
.endif
  s_cbranch_scc1    .restore_vector_before_exit_trap
  v_writelane_b32   v2, ttmp14, 0                           // v[2] = mailbox address (low part)
  v_writelane_b32   v3, ttmp15, 0                           // v[3] = mailbox address (high part)

  s_wait_storecnt   0                                       // Ensure mailbox address is ready before sending interrupt
  V_WRITELANE_B32   v0, 0x0                                 // v[0] = 0 (event ID low part)
  global_store_b32  v[2:3], v0, off, offset:0x0, scope:SCOPE_SYS // Send event ID to the mailbox
  s_wait_storecnt   0
  s_mov_b32         ttmp14, m0                              // Backup m0 (event ID low part) to ttmp14
  v_readlane_b32    ttmp15, v0, 0                           // Read event ID low part from v[0] into ttmp15
  s_mov_b32         m0, ttmp15                              // Set m0 to event ID (low part)
  s_sendmsg         sendmsg(MSG_INTERRUPT)                  // send interrupt message to host
  s_wait_kmcnt      0                                       // Wait for interrupt to complete
  s_mov_b32         m0, ttmp14                              // Restore m0 to original value (event ID low part)

  // v[0:1] = free
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader's v[0:1]
  // ttmp[4:5] holds backup of original shader's v[2:3]
  // ttmp6 = free
  // ttmp[10:11] holds original shader's [exec_lo,exec_hi]
  // ttmp[14:15]=somewhere in tma region, EXEC is junk

.restore_vector_before_exit_trap:
  // Restore all vector registers to original user shader state
  // ttmp[2:5] contain backups of original user v[0:3]
  v_writelane_b32   v2, ttmp4, 0                            // restore v[2:3] to user data
  v_writelane_b32   v3, ttmp5, 0                            // v[2:3] = original user data
  v_writelane_b32   v0, ttmp2, 0                            // restore v[0:1] to user data
  v_writelane_b32   v1, ttmp3, 0                            // v[0:1] = original user data

  // Handle cases where sample cannot be stored (buffer full, overflow, etc.)
.lost_sample:
  // v0 contains local_entry, v1 is free
  // v[2:3] is original user-data
  // ttmp[2:3] [local_entry, buf_size]
  // ttmp[4:5] = free
  // ttmp6=buffer_id for GFX12.0, workgroup info for GFX12.5+
  // ttmp[10:11] holds original shader's [exec_lo,exec_hi]
  // ttmp13.b31 = buffer_id
  // ttmp[14:15]=tma
  // EXEC=0x1
  // Restore vector registers before exiting

  s_bitcmp1_b32     ttmp13, TTMP13_STOCH_FLAG_BIT_SHIFT     // Check if stochastic sampling

  s_cbranch_scc0    .lost_sample_restore                    // If not, skip lock release
  S_GETREG_B32      HW_REG_PERF_SNAPSHOT_PC_HI              //  Read PC_HI register to release snapshot lock

.lost_sample_restore:
  v_writelane_b32   v0, ttmp2, 0                            // restore v[0:1] to user data
  v_writelane_b32   v1, ttmp3, 0                            // v[0:1] = original user data
.if .amdgcn.gfx_generation_minor >= 5
  RESTORE_MODE_VGPR_MSBS                                    // Restore 8 VGPR MSB bits from ttmp13[7:0] to MODE[19:12]
  RESTORE_XNACK_MASK                                        // Restore 32 bits of XNACK_MASK stored in ttmp11 and ttmp13
  s_mov_b32         exec_lo, ttmp10                         // restore exec mask
  s_mov_b32         exec_hi, 0
  s_mov_b32         vcc_hi,0                                // vcc_hi is not used in wave32
.else
  s_mov_b64         exec, ttmp[10:11]
.endif

.exit_trap:

.if .amdgcn.gfx_generation_minor == 0
    // Restore ttmp11[27:26] into SCHED_MODE[0:1]
    s_bfe_u32         ttmp2, ttmp11, TTMP11_SCHED_MODE_BFE
    s_setreg_b32      hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), ttmp2
.endif

.if .amdgcn.gfx_generation_minor == 5
 // VGPR MSB Fixup Function for GFX12.5
 // This function detects if we're returning to a VALU instruction followed by
 // s_set_vgpr_msb and restores the correct VGPR bank selection from SIMM16[15:8].
 //
 // TTMP usage at entry:
 // ttmp0: PC[31:0]
 // ttmp1: 31:28 : trap_id[3:0], 24:0 : PC[56:32]
 // ttmp2, ttmp3, ttmp10, ttmp13, ttmp14, ttmp15: free for use
 //
 // Register Safety:
 // - Uses ttmp2, ttmp3, ttmp10, ttmp13, ttmp14, ttmp15 (all safe, avoid ttmp6/ttmp11)
 // - Preserves ttmp0, ttmp1 (PC values)
 // - Does NOT modify user-visible state except MODE register (intentional fixup)

 // In this case no further progress is expected so fixup is not needed
  s_getreg_b32   ttmp10, hwreg(HW_REG_EXCP_FLAG_PRIV)
  s_bitcmp1_b32  ttmp10, SQ_WAVE_EXCP_FLAG_PRIV_MEMVIOL_SHIFT
  s_cbranch_scc1 .fixup_done

.fixup_start:
  s_and_b32         ttmp1, ttmp1, 0xfffffff                 // Zero out trap_id[3:0] from ttmp1
  s_load_b64        ttmp[14:15], ttmp[0:1], 0, scope:SCOPE_CU // Load the 2 instruction DW we are returning to
  s_wait_kmcnt      0
  s_load_b64        ttmp[2:3], ttmp[0:1], 8, scope:SCOPE_CU // Load the next 2 instruction DW, just in case
  s_and_b32         ttmp10, ttmp14, 0x80000000              // Check bit 31 in the first DWORD
                                                            // SCC set if ttmp10 is != 0, i.e. if bit 31 == 1
  s_cbranch_scc1    .fixup_not_vop12c                       // If bit 31 is 1, we are not VOP1, VOP2, or VOP3C
 // Fall through here means bit 31 == 0, meaning we are VOP1, VOP2, or VOPC
 // Size of instruction depends on Opcode or SRC0_9
 // Check for VOP2 opcode
  s_bfe_u32         ttmp10, ttmp14, (25 | (6 << 0x10))      // Check bits 30:25 for VOP2 Opcode
 // VOP2 V_FMAMK_F64 of V_FMAAK_F64 has implied 64-bit literal, 3 DW
  s_sub_co_i32      ttmp13, ttmp10, 0x23                    // V_FMAMK_F64 is 0x23, V_FMAAK_F64 is 0x24
  s_cmp_le_u32      ttmp13, 0x1                             // 0==0x23, 1==0x24
  s_cbranch_scc1    .fixup_three_dword                      // If either, this is 3 DWORD inst
 // VOP2 V_FMAMK_F32, V_FMAAK_F32, V_FMAMK_F16, V_FMAAK_F16, 2 DW
  s_sub_co_i32      ttmp13, ttmp10, 0x2c                    // V_FMAMK_F32 is 0x2c, V_FMAAK_F32 is 0x2d
  s_cmp_le_u32      ttmp13, 0x1                             // 0==0x2c, 1==0x2d
  s_cbranch_scc1    .fixup_two_dword                        // If either, this is 2 DWORD inst
  s_sub_co_i32      ttmp13, ttmp10, 0x37                    // V_FMAMK_F16 is 0x37, V_FMAAK_F16 is 0x38
  s_cmp_le_u32      ttmp13, 0x1                             // 0==0x37, 1==0x38
  s_cbranch_scc1    .fixup_two_dword                        // If either, this is 2 DWORD inst
 // Check SRC0_9 for VOP1, VOP2, and VOPC
  s_and_b32         ttmp10, ttmp14, 0x1ff                   // Check bits 8:0 for SRC0_9
 // Literal constant 64 is 3 DWORDs
  s_cmp_eq_u32      ttmp10, 0xfe                            // 0xfe == 254 == Literal constant64
  s_cbranch_scc1    .fixup_three_dword                      // 3 DWORD inst
 // Literal constant 32, DPP16, DPP8, and DPP8FI are 2 DWORDs
  s_cmp_eq_u32      ttmp10, 0xff                            // 0xff == 255 = Literal constant32
  s_cbranch_scc1    .fixup_two_dword                        // 2 DWORD inst
  s_cmp_eq_u32      ttmp10, 0xfa                            // 0xfa == 250 = DPP16
  s_cbranch_scc1    .fixup_two_dword                        // 2 DWORD inst
  s_sub_co_i32      ttmp13, ttmp10, 0xe9                    // DPP8 is 0xe9, DPP8FI is 0xea
  s_cmp_le_u32      ttmp13, 0x1                             // 0==0xe9, 1==0xea
  s_cbranch_scc1    .fixup_two_dword                        // If either, this is 2 DWORD inst
 // Instruction is 1 DWORD otherwise
.fixup_one_dword:
 // Check if TTMP15 contains the value for S_SET_VGPR_MSB instruction
  s_and_b32         ttmp10, ttmp15, 0xffff0000              // Check encoding in upper 16 bits
  s_cmp_eq_u32      ttmp10, 0xbf860000                      // Check if SOPP (9b'10_1111111) and S_SET_VGPR_MSB (7b'0000110)
  s_cbranch_scc0    .fixup_done                             // No problem, no fixup needed
 // VALU op followed by a S_SET_VGPR_MSB. Need to pull SIMM[15:8] to fix up MODE.*_VGPR_MSB
  s_bfe_u32         ttmp10, ttmp15, (14 | (2 << 0x10))      // Shift SIMM[15:14] over to 1:0, Dst
  s_and_b32         ttmp13, ttmp15, 0x3f00                  // Mask to get SIMM[13:8] only
  s_lshr_b32        ttmp13, ttmp13, 6                       // Shift SIMM[13:8] into 7:2, Src2, Src1, Src0
  s_or_b32          ttmp10, ttmp10, ttmp13                  // Src2, Src1, Src0, Dst --> format in MODE register
  s_setreg_b32      hwreg(HW_REG_MODE, 12, 8), ttmp10  // Write value into MODE[19:12]
  s_branch          .fixup_done
.fixup_not_vop12c:
 // ttmp[0:1]: {8b'0} PC[56:0]
 // ttmp2: PC+2 value (not waitcnt'ed yet)
 // ttmp3: PC+3 value (not waitcnt'ed yet)
 // ttmp10, ttmp13: free
 // ttmp14: PC+0 value
 // ttmp15: PC+1 value
 // Not VOP1, VOP2, or VOPC.
 // Check if we are VOP3 or VOP3SD
  s_and_b32         ttmp10, ttmp14, 0xfc000000              // Bits 31:26
  s_cmp_eq_u32      ttmp10, 0xd4000000                      // If 31:26 = 0x35, this is VOP3 or VOP3SD
  s_cbranch_scc1    .fixup_check_vop3                       // If VOP3 or VOP3SD, need to check SRC2_9, SRC1_9, SRC0_9
 // Not VOP1, VOP2, VOPC, VOP3, or VOP3SD.
 // Check for VOPD
  s_cmp_eq_u32      ttmp10, 0xc8000000                      // If 31:26 = 0x32, this is VOPD
  s_cbranch_scc1    .fixup_check_vopd                       // If VOPD, need to check OpX, OpY, SRCX0 and SRCY0
 // Not VOP1, VOP2, VOPC, VOP3, VOP3SD, VOPD.
 // Check if we are VOPD3
  s_and_b32         ttmp10, ttmp14, 0xff000000              // Bits 31:24
  s_cmp_eq_u32      ttmp10, 0xcf000000                      // If 31:24 = 0xcf, this is VOPD3
  s_cbranch_scc1    .fixup_three_dword                      // If VOPD3, 3 DWORD inst
 // Not VOP1, VOP2, VOPC, VOP3, VOP3SD, VOPD, or VOPD3.
 // Check if we are in the middle of VOP3PX.
  s_and_b32         ttmp13, ttmp14, 0xffff0000              // Bits 31:16
  s_cmp_eq_u32      ttmp13, 0xcc330000                      // If 31:16 = 0xcc33, this is 8 bytes past VOP3PX
  s_cbranch_scc1    .fixup_vop3px_middle
  s_cmp_eq_u32      ttmp13, 0xcc880000                      // If 31:16 = 0xcc88, this is 8 bytes past VOP3PX
  s_cbranch_scc1    .fixup_vop3px_middle
 // Might be in VOP3P, but we must ensure we are not VOP3PX2
  s_and_b32         ttmp13, ttmp14, 0xffff0000              // Bits 31:16
  s_cmp_eq_u32      ttmp13, 0xcc350000                      // If 31:16 = 0xcc35, this is VOP3PX2
  s_cbranch_scc1    .fixup_done                             // If VOP3PX2, no fixup needed
  s_cmp_eq_u32      ttmp13, 0xcc3a0000                      // If 31:16 = 0xcc3a, this is VOP3PX2
  s_cbranch_scc1    .fixup_done                             // If VOP3PX2, no fixup needed
 // Check if we are VOP3P
  s_cmp_eq_u32      ttmp10, 0xcc000000                      // If 31:24 = 0xcc, this is VOP3P
  s_cbranch_scc0    .fixup_done                             // Not in VOP3P, so instruction is not VOP1, VOP2,
                                                            // VOPC, VOP3, VOP3SD, VOP3P, VOPD, or VOPD3
                                                            // No fixup needed.
 // Fall-through if we are in VOP3P to check SRC2_9, SRC1_9, and SRC0_9
.fixup_check_vop3:
 // Start with Src0, which is in bits 8:0 of second instruction DW, ttmp15
  s_and_b32         ttmp10, ttmp15, 0x1ff                   // Mask out unused bits
 // Src0_9 == Literal constant 32, DPP16, DPP8, and DPP8FI means 3 DWORDs
  s_cmp_eq_u32      ttmp10, 0xff                            // 0xff == 255 = Literal constant32
  s_cbranch_scc1    .fixup_three_dword                      // 3 DWORD inst
  s_cmp_eq_u32      ttmp10, 0xfa                            // 0xfa == 250 = DPP16
  s_cbranch_scc1    .fixup_three_dword                      // 3 DWORD inst
  s_sub_co_i32      ttmp10, ttmp10, 0xe9                    // DPP8 is 0xe9, DPP8FI is 0xea
  s_cmp_le_u32      ttmp10, 0x1                             // 0==0xe9, 1==0xea
  s_cbranch_scc1    .fixup_three_dword                      // If either, this is 3 DWORD inst
  s_and_b32         ttmp10, ttmp15, 0x3fe00                 // Next is Src1, which is in 17:9
  s_cmp_eq_u32      ttmp10, 0x1fe00                         // 0xff == 255 = Literal constant32
  s_cbranch_scc1    .fixup_three_dword                      // 3 DWORD inst
  s_and_b32         ttmp10, ttmp15, 0x7fc0000               // Next is Src2, which is in 26:18
  s_cmp_eq_u32      ttmp10, 0x3fc0000                       // 0xff == 255 = Literal constant32
  s_cbranch_scc1    .fixup_three_dword                      // 3 DWORD inst
  s_branch          .fixup_two_dword                        // No special encodings, VOP3* is 2 Dword

.fixup_check_vopd:
 // OpX being V_DUAL_FMA*K_F32 means 3 DWORDs
  s_bfe_u32         ttmp10, ttmp14, (22 | (4 << 0x10))      // OPX is bits 25:22
  s_sub_co_i32      ttmp10, ttmp10, 0x1                     // V_DUAL_FMAAK_F32 is 0x1, V_DUAL_FMAMK_F32 is 0x2
  s_cmp_le_u32      ttmp10, 0x1                             // 0==0x1, 1==0x2
  s_cbranch_scc1    .fixup_three_dword                      // If either, this is 3 DWORD inst
 // OpY being V_DUAL_FMA*K_F32 means 3 DWORDs
  s_bfe_u32         ttmp10, ttmp14, (17 | (5 << 0x10))      // OPY is bits 21:17
  s_sub_co_i32      ttmp10, ttmp10, 0x1                     // V_DUAL_FMAAK_F32 is 0x1, V_DUAL_FMAMK_F32 is 0x2
  s_cmp_le_u32      ttmp10, 0x1                             // 0==0x1, 1==0x2
  s_cbranch_scc1    .fixup_three_dword                      // If either, this is 3 DWORD inst
 // SRCX0 == Literal constant 32 means 3 DWORDs
  s_and_b32         ttmp10, ttmp14, 0x1ff                   // SRCX0 is in bits 8:0 of 1st DWORD
  s_cmp_eq_u32      ttmp10, 0xff                            // 0xff == 255 = Literal constant32
  s_cbranch_scc1    .fixup_three_dword                      // 3 DWORD inst
 // SRCY0 == Literal constant 32 means 3 DWORDs
  s_and_b32         ttmp10, ttmp15, 0x1ff                   // SRCY0 is in bits 8:0 of 2nd DWORD
  s_cmp_eq_u32      ttmp10, 0xff                            // 0xff == 255 = Literal constant32
  s_cbranch_scc1    .fixup_three_dword                      // 3 DWORD inst
                                                            // If otherwise, no special encodings. Default VOPD is 2 Dword
                                                            // Fall-thru if true, because this is a 2 DWORD inst
.fixup_two_dword:
  s_wait_kmcnt      0                                       // Wait for PC+2 and PC+3 to arrive in ttmp2 and ttmp3
  s_mov_b32         ttmp15, ttmp2                           // Move possible S_SET_VGPR_MSB into ttmp15
  s_branch          .fixup_one_dword                        // Go to common logic that checks if it is S_SET_VGPR_MSB
.fixup_three_dword:
  s_wait_kmcnt      0                                       // Wait for PC+2 and PC+3 to arrive in ttmp2 and ttmp3
  s_mov_b32         ttmp15, ttmp3                           // Move possible S_SET_VGPR_MSB into ttmp15
  s_branch          .fixup_one_dword                        // Go to common logic that checks if it is S_SET_VGPR_MSB
.fixup_vop3px_middle:
  s_sub_co_u32      ttmp0, ttmp0, 8                         // Rewind PC 8 bytes to beginning of instruction
  s_sub_co_ci_u32   ttmp1, ttmp1, 0
  s_branch          .fixup_two_dword                        // 2 DWORD inst (2nd half of a 4 DWORD inst)
.fixup_done:

  s_wait_idle                                               // Required by SPG before reading/writing XNACK_STATE_PRIV

  // If SAVE_CONTEXT was set, re-assert it to ensure the trap handler is
  // re-entered.
  s_getreg_b32      ttmp2, hwreg(HW_REG_EXCP_FLAG_PRIV, SQ_WAVE_EXCP_FLAG_PRIV_SAVE_CONTEXT, 1)
  s_and_b32         ttmp2, ttmp2, ttmp2
  s_cbranch_scc0    .no_ctx_save
  s_setreg_b32      hwreg(HW_REG_EXCP_FLAG_PRIV, SQ_WAVE_EXCP_FLAG_PRIV_SAVE_CONTEXT, 1), ttmp2
.no_ctx_save:

  // Restore SQ_WAVE_XNACK_STATE_PRIV
  s_lshr_b32        ttmp2, ttmp11, TTMP11_FIRST_REPLAY_SHIFT
  s_setreg_b32      hwreg(HW_REG_XNACK_STATE_PRIV, SQ_WAVE_XNACK_STATE_PRIV_FIRST_REPLAY_SHIFT, SQ_WAVE_XNACK_STATE_PRIV_FIRST_REPLAY_SIZE), ttmp2
  s_lshr_b32        ttmp2, ttmp11, TTMP11_REPLAY_W64H_SHIFT
  s_setreg_b32      hwreg(HW_REG_XNACK_STATE_PRIV, SQ_WAVE_XNACK_STATE_PRIV_REPLAY_W64H_SHIFT, SQ_WAVE_XNACK_STATE_PRIV_REPLAY_W64H_SIZE), ttmp2
  s_lshr_b32        ttmp2, ttmp11, TTMP11_FXPTR_SHIFT
  s_setreg_b32      hwreg(HW_REG_XNACK_STATE_PRIV, SQ_WAVE_XNACK_STATE_PRIV_FXPTR_SHIFT, SQ_WAVE_XNACK_STATE_PRIV_FXPTR_SIZE), ttmp2
.endif //.amdgcn.gfx_generation_minor == 5

  // Restore SQ_WAVE_STATUS.
  s_and_b64         exec, exec, exec                        // Restore STATUS.EXECZ, not writable by s_setreg_b32
  s_and_b64         vcc, vcc, vcc                           // Restore STATUS.VCCZ, not writable by s_setreg_b32
  s_setreg_b32      hwreg(HW_REG_STATE_PRIV, 0, SQ_WAVE_STATE_PRIV_BARRIER_COMPLETE_SHIFT), ttmp12
  s_lshr_b32        ttmp12, ttmp12, (SQ_WAVE_STATE_PRIV_BARRIER_COMPLETE_SHIFT + 1)
  s_setreg_b32      hwreg(HW_REG_STATE_PRIV, SQ_WAVE_STATE_PRIV_BARRIER_COMPLETE_SHIFT + 1, 32 - SQ_WAVE_STATE_PRIV_BARRIER_COMPLETE_SHIFT - 1), ttmp12

  // Return to original (possibly modified) PC.
  s_rfe_b64         [ttmp0, ttmp1]

.parked:
  s_trap            0x2
  s_branch          .parked

// Add s_code_end padding so instruction prefetch always has something to read.
.rept (256 - ((. - trap_entry) % 64)) / 4
  s_code_end
.endr
