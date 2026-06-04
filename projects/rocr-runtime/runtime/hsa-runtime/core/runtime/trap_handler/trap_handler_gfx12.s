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

.if .amdgcn.gfx_generation_minor == 0
  .set SQ_WAVE_PC_HI_ADDRESS_MASK                  , 0xFFFF
.else
  .set SQ_WAVE_PC_HI_ADDRESS_MASK                  , 0x1FFFFFF
.endif
.set SQ_WAVE_PC_HI_TRAP_ID_BFE                     , (SQ_WAVE_PC_HI_TRAP_ID_SHIFT | (SQ_WAVE_PC_HI_TRAP_ID_SIZE << 16))
.set SQ_WAVE_PC_HI_TRAP_ID_SHIFT                   , 28
.set SQ_WAVE_PC_HI_TRAP_ID_SIZE                    , 4
.set SQ_WAVE_PC_HI_TRAP_ID_MASK                    , (((1 << SQ_WAVE_PC_HI_TRAP_ID_SIZE) - 1) << SQ_WAVE_PC_HI_TRAP_ID_SHIFT)

.set SQ_WAVE_STATE_PRIV_SCC_SHIFT                  , 9
.set SQ_WAVE_STATE_PRIV_HALT_SHIFT                 , 14

.set TRAP_ID_ABORT                                 , 2
.set TRAP_ID_DEBUGTRAP                             , 3
.if .amdgcn.gfx_generation_minor == 0
  .set TTMP1_SCHED_MODE_MASK                       , 0xC000000
.endif

.set TTMP6_SAVED_STATUS_HALT_MASK                  , (1 << TTMP6_SAVED_STATUS_HALT_SHIFT)
.set TTMP6_SAVED_STATUS_HALT_SHIFT                 , 29
.set TTMP6_WAVE_STOPPED_SHIFT                      , 30
.set TTMP6_SCC_SHIFT                               , 31
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

  .set SQ_WAVE_MODE_MSB_SHIFT                      , 12
  .set SQ_WAVE_MODE_MSB_SIZE                       , 8
  .set TTMP11_WAVE_MODE_MSB_MASK                   , ((1 << SQ_WAVE_MODE_MSB_SIZE) - 1)
.endif

.if .amdgcn.gfx_generation_minor == 0
  .set TTMP_PC_HI_SHIFT                            , 7
.endif

.set TTMP1_BUF_ID_BIT_POSITION                    , 25           // TTMP1 bit position for buffer ID

.set TTMP8_DISPATCH_ID_MASK                        , 0X1FFFFFF
// Per-sample data layout within the device buffer. Each sample is 64 bytes.
// These are offsets from the start of a specific sample slot in the device buffer.

.set SAMPLE_OFF_BYTES_PER_SAMPLE                   , 0x40         // 64 bytes per sample slot
.set SAMPLE_OFF_PC_HOST                            , 0x00         // original PC (host trap only)
.set SAMPLE_OFF_EXEC_LOHI                          , 0x08         // saved EXEC low/high
.set SAMPLE_OFF_WGID_XY                            , 0x10         // WG id X / Y
.set SAMPLE_OFF_WGID_Z                             , 0x18         // WG id Z (32-bit)
.set SAMPLE_OFF_WAVE_IN_GROUP_CHIPLET              , 0x1C         // wave_in_wg[5:0] | reserved_wg[7:6] | chiplet[10:8] | reserved[31:11]
.set SAMPLE_OFF_TIMESTAMP                          , 0x30         // 64 bit realtime counter
.set SAMPLE_OFF_HW_ID                              , 0x20         // Combined HW_ID (HW_ID1 + HW_ID2)
.set SAMPLE_OFF_SNAPSHOT_DATA                      , 0x24         // Performance snapshot data
.set SAMPLE_OFF_CORRELATION                        , 0x38         // doorbell + dispatch id
.set SAMPLE_OFF_BUF_WRITTEN_VAL                    , 0x10         // Offset to buf_written_val0/1 in pcs_sampling_data_t
.set SAMPLE_OFF_WATERMARK_FIELD                    , 0x14         // Offset to watermark field in pcs_sampling_data_t
.set SAMPLE_OFF_BUF_SIZE                           , 0x8          // Offset to buf_size in pcs_sampling_data_t
.set SAMPLE_OFF_DONE_SIG0                          , 0x18         // Offset for done_sig0 (hsa_signal_t handle for buffer 0)
.set SAMPLE_OFF_SIGNAL_VALUE                       , 0x8          // Offset within signal structure to value field
.set SAMPLE_OFF_SIGNAL_EVENT_ID                    , 0x18         // Offset within signal structure to event_id field
.set SAMPLE_OFF_EVENT_MAILBOX_PTR                  , 0x10         // Offset for event mailbox pointer for current PC sampling buffer

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

// ABI between first and second level trap handler:
//
//   gfx12
//     { ttmp1, ttmp0 } = TrapID[3:0], SCHED_MODE[1:0], zeros, PC[47:0]
//     ttmp11 = 0[7:0], DebugEnabled[0], 0[15:0], NoScratch[0], 0[5:0]
//   gfx12.5
//     { ttmp1, ttmp0 } = TrapID[3:0], 0[2:0], PC[56:0]
//     ttmp11 = 0[7:0], DebugEnabled[0], SQ_WAVE_XNACK_STATE_PRIV[18],
//              SQ_WAVE_XNACK_STATE_PRIV[16], SQ_WAVE_XNACK_STATE_PRIV[6:0], 0[13:0]
//
//     ttmp12 = SQ_WAVE_STATE_PRIV (Private wave state register value).
//     ttmp14 = TMA[31:0] - TMA_LO (Trap Memory Argument Low - base address for trap handler data, low 32 bits).
//     ttmp15 = TTMA[63:32] - TMA_HI (Trap Memory Argument High - base address for trap handler data, high 32 bits).
//
// Restricted register list:
//   gfx12:
//     ttmp[0:1] - Must be preserved for RFE
//     ttmp[7:9] - Contain workgroup information, must be preserved
//     ttmp12    - Contains SQ_WAVE_STATE_PRIV, SCC and SLEEP_WAKEUP bits must be preserved
//     ttmp[14:15] - Contain TMA address, must be preserved
//
//   gfx12.5:
//     ttmp[0:1] - Must be preserved for RFE
//     ttmp6    -  Contains workgroup info if clusters enabled, only bits [31:28] can be modified
//     ttmp[7:9] - Contain workgroup information, must be preserved
//     ttmp11      - Only bits [13:0] and [31:28] can be modified, bits [27:14] contain XNACK/debugger state
//     ttmp12    - Contains SQ_WAVE_STATE_PRIV, SCC and SLEEP_WAKEUP bits must be preserved
//     ttmp[14:15] - Contain TMA address, must be preserved
//
// Safe to use as scratch:
//   gfx12: ttmp[2:6], ttmp10, ttmp13
//   gfx12.5: ttmp[2:5], ttmp10, ttmp13 (bits [21:0] used for XNACK/MODE), exec_hi, ttmp13

 trap_entry:
 .if .amdgcn.gfx_generation_minor == 0
    // Save SCHED_MODE from ttmp1[27:26] into ttmp11[27:26]. We will restore it on exit
    s_andn2_b32         ttmp11, ttmp11, TTMP11_SCHED_MODE_MASK
    s_and_b32           ttmp2,  ttmp1, TTMP1_SCHED_MODE_MASK
    s_or_b32            ttmp11, ttmp11, ttmp2
  .endif
  s_mov_b32           ttmp3, 0                               // Clear ttmp3 as it will contain the exception code

  // Move SQ_WAVE_STATE_PRIV.SCC bit from ttmp12[9] to ttmp6[31].
  // This frees ttmp12 as a scratch register.
  s_bfe_u32         ttmp2, ttmp12, (SQ_WAVE_STATE_PRIV_SCC_SHIFT | (1 << 16))
  s_lshl_b32        ttmp2, ttmp2, TTMP6_SCC_SHIFT
  s_bitset0_b32     ttmp6, TTMP6_SCC_SHIFT
  s_or_b32          ttmp6, ttmp6, ttmp2

  // To avoid more overhead on the critical sample processing path, we decided to give a priority
  // to host-trap and perf_snapshot trap over the s_trap and halt.
.check_hosttrap:
  // ttmp[14:15] points to TMA.
  // Scratch registers: ttmp[2:3], ttmp[4:5], ttmp10, ttmp13
  s_getreg_b32      ttmp2, hwreg(HW_REG_EXCP_FLAG_PRIV)     // On gfx12, EXCP_FLAG_PRIV.b7
  s_bitcmp1_b32     ttmp2, SQ_WAVE_EXCP_FLAG_PRIV_HT_SHIFT  // Test Host Trap bit.
  s_cbranch_scc0    .check_stochastic                       // If not HT, check for stochastic sampling

  // It's a Host Trap event.
  s_load_b64        ttmp[14:15], ttmp[14:15], 0x0, scope:SCOPE_CU  // ttmp[14:15]=*host_trap_buffers
  s_wait_kmcnt      0                                       // Ensure previous load is complete.
  s_branch          .profile_trap_handlers

.check_stochastic:
  // ttmp2 already contains HW_REG_EXCP_FLAG_PRIV from .check_hosttrap
  s_bitcmp1_b32     ttmp2, SQ_WAVE_EXCP_FLAG_PRIV_PERF_SNAPSHOT_SHIFT // Test Performance Snapshot bit.

  s_cbranch_scc0    .handle_sw_trap                       // If not Stochastic, continue to check trap ID

  s_load_b64        ttmp[14:15], ttmp[14:15], 0x8, scope:SCOPE_CU  // ttmp[14:15]=*stoch_trap_buf
  s_wait_kmcnt      0
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
  // Save trap id and halt status in ttmp6
  s_andn2_b32       ttmp6, ttmp6, (TTMP6_SAVED_TRAP_ID_MASK | TTMP6_SAVED_STATUS_HALT_MASK)
  s_bfe_u32         ttmp2, ttmp1, SQ_WAVE_PC_HI_TRAP_ID_BFE
  s_min_u32         ttmp2, ttmp2, 0xF
  s_lshl_b32        ttmp2, ttmp2, TTMP6_SAVED_TRAP_ID_SHIFT
  s_or_b32          ttmp6, ttmp6, ttmp2
  s_getreg_b32      ttmp2, hwreg(HW_REG_WAVE_STATE_PRIV, SQ_WAVE_STATE_PRIV_HALT_SHIFT, 1)
  s_lshl_b32        ttmp2, ttmp2, TTMP6_SAVED_STATUS_HALT_SHIFT
  s_or_b32          ttmp6, ttmp6, ttmp2
.elseif .amdgcn.gfx_generation_minor == 5
  // Save halt status in ttmp6.
  s_andn2_b32       ttmp6, ttmp6, TTMP6_SAVED_STATUS_HALT_MASK
  s_getreg_b32      ttmp2, hwreg(HW_REG_WAVE_STATE_PRIV, SQ_WAVE_STATE_PRIV_HALT_SHIFT, 1)
  s_lshl_b32        ttmp2, ttmp2, TTMP6_SAVED_STATUS_HALT_SHIFT
  s_or_b32          ttmp6, ttmp6, ttmp2

  // Save the trap id in ttmp11.
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
  // Halt the wavefront.
  s_bitset1_b32     ttmp6, TTMP6_WAVE_STOPPED_SHIFT
  s_setreg_imm32_b32 hwreg(HW_REG_STATE_PRIV, SQ_WAVE_STATE_PRIV_HALT_SHIFT, 1), 1

  // Initialize TTMP registers
  s_bitcmp1_b32     ttmp8, TTMP8_DEBUG_FLAG_SHIFT
  s_cbranch_scc1    .ttmps_initialized
  s_mov_b32         ttmp4, 0
  s_mov_b32         ttmp5, 0
  s_bitset1_b32     ttmp8, TTMP8_DEBUG_FLAG_SHIFT
.ttmps_initialized:
  s_branch          .exit_trap

.profile_trap_handlers:
.if .amdgcn.gfx_generation_minor >= 5
  // Do the setup for gfx1250+
  // NOTE: When in PRIV=1, scheduling happens as if SCHED_MODE[1:0] was 0 (normal mode)
  // so no need to save/restore SCHED_MODE bits.

  // save xnack_mask in ttmp12
  s_getreg_b32 ttmp12,  hwreg(HW_REG_XNACK_MASK)

.if .amdgcn.gfx_generation_minor == 5
  // MI450, save MODE.MSB to ttmp11[7:0], and clear MODE.MSB
  s_getreg_b32 ttmp2, hwreg(HW_REG_WAVE_MODE, SQ_WAVE_MODE_MSB_SHIFT, SQ_WAVE_MODE_MSB_SIZE)
  s_andn2_b32 ttmp11, ttmp11, TTMP11_WAVE_MODE_MSB_MASK
  s_or_b32 ttmp11, ttmp11, ttmp2

  s_mov_b32 ttmp2, 0
  s_setreg_b32 hwreg(HW_REG_WAVE_MODE, SQ_WAVE_MODE_MSB_SHIFT, SQ_WAVE_MODE_MSB_SIZE), ttmp2
.endif

.else
  // GFX12.0 -> save ttmp11 (contains SCHED_MODE[1:0] at bits [27:26]) into ttmp12
  // before overwriting ttmp11 with exec_hi. ttmp12 is free since SCC moved to ttmp6[31].
  s_mov_b32 ttmp12, ttmp11
  s_mov_b32 ttmp11, exec_hi
.endif

  // TTMP assignments for profiler
  //
  // ttmp0 = PC[31:2], 0, 0
  // ttmp1 =
  //  gfx12.0: 
  //    ttmp1[31:26] = 0 (free to use),
  //    ttmp1[25]    = buff_id bit for PC sampling
  //    ttmp1[24:16] = 0 (free to use)
  //    ttmp1[15: 0] = PC[47:32]
  //  gfx12.5:
  //    ttmp1[31:26] = 0 (free to use)
  //    ttmp1[25]    = buff_id for PC sampling
  //    ttmp1[24: 0] = PC[56:32]
  // ttmp6 =
  //      gfx12.0: STATE_PRIV.SCC, 30b0 (free to use)
  //   >= gfx12.5: STATE_PRIV.SCC, 3b0 (free to use), cluster coordination
  // ttmp7 = WGP Y/Z
  // ttmp8 = 0 (unusable), yz valid, wave_in_wg, dispatch_idx
  // ttmp9 = WGP X
  // ttmp10 = EXEC_LO
  // ttmp11 =
  //   gfx12.0: EXEC_HI
  //   gfx12.5:
  //     ttmp11[22:14] = XNACK_STATE_PRIV[18, 16, 6:0]
  //     ttmp11[7:0]   = MODE.MSB[7:0]
  // ttmp12 = 
  //      gfx12.0: 4b0 (free to use), SCHED_MODE[1:0], 26b0 (free to use)
  //   >= gfx12.5: XNACK_MASK
  // ttmp14 = TMA_LO-ish
  // ttmp15 = TMA_HI-ish
  //
  // v[0:3] contain user shader data that must be preserved/restored
  // exec: Contains user's execution mask
  //       gfx12.0: both exec_lo and exec_hi must be preserved
  //    >= gfx12.5: only exec_lo is preserved
  // ttmp[2:5] and ttmp13 free to use as scratch registers 

  s_mov_b32         ttmp10, exec_lo                         // Save exec_lo to ttmp10

  s_mov_b64 exec, 0x1                                       // turn on lane 0 only

  v_readlane_b32    ttmp2, v0, 0                            // Save out lane 0's first VGPR
  v_readlane_b32    ttmp3, v1, 0                            // Save out lane 0's second VGPR

  // At this point, ttmp[4:5],v[0:1] are free.
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

  // At this point, ttmp[4:5] is free. ttmp13 is free
  // v[0:1] (lane 0) now holds the previous value of buf_write_val.
  // This previous value gives the slot index for the current sample.

  v_readlane_b32    ttmp4, v1, 0x0                          // ttmp4 = high 32 bits of previous buf_write_val[63:32], i.e., bit 63 of previous buf_write_val
  s_lshr_b32        ttmp4, ttmp4, 31                        // ttmp4 = previous_buffer_id (0 or 1, from bit 63 of original uint64_t)
                                                            // This ttmp4 is used to select which buffer's metadata (size, watermark, signal) to use.
                                                            // It's also used to calculate the base address of the sample buffer.
  s_bitset0_b32     ttmp1, TTMP1_BUF_ID_BIT_POSITION        // Clear our local buffer full flag for now
  s_cmp_eq_u32      ttmp4, 0                                // Check the value of the buf_to_use, and update ttmp1's buffer_id accordingly
  s_cbranch_scc1    .skip_bufbit_set                        // buffer_id (buf_to_use) remains zero
  s_bitset1_b32     ttmp1, TTMP1_BUF_ID_BIT_POSITION        // buffer_id (buf_to_use) is 1.

.skip_bufbit_set:
  // ttmp[2:3]=v[0:1]-backup,
  // ttmp[4:5]=free
  // ttmp[14:15]=tma
  // ttmp13 free
  // v[0:1].lane0=local_entry,
  // v[2:3]=original, EXEC=0x1

  v_bfe_u32         v1, v1, 0, SAMPLE_INDEX_WIDTH           // v[0:1] = new local_entry
                                                            // removes bit 31 from v1, returning v1 & 0x7FFFFFFF.
  v_readlane_b32    ttmp5, v1, 0                            // ttmp5 = high 31 bits of sample index (if index > 2^32-1).
  s_cmp_lg_u32      ttmp5, 0                                // Check if sample index is very large (overflowed 32 bits).

  s_cbranch_scc1    .lost_sample                            // If ttmp5 > 0, index is too large, treat as lost sample.

  s_load_b32        ttmp5, ttmp[14:15], SAMPLE_OFF_BUF_SIZE, scope:SCOPE_CU // ttmp5 = pcs_sampling_data_t.buf_size
  v_readlane_b32    ttmp4, v0, 0                            // ttmp4 = sample_index_for_current_sample (from v0)
  s_wait_kmcnt      0                                       // Wait for buf_size load.

  s_cmp_ge_u32      ttmp4, ttmp5                            // if local_entry >= buf_size
  s_cbranch_scc0    .process_sample                         // If index >= buf_size, buffer is full, sample is lost (.lost_sample path is hit).
                                                            // Otherwise, process sample (jump to .process_sample path).

.lost_sample:
  // Handle cases where sample cannot be stored (buffer full, overflow, etc.)
  // v[0:1] contains local_entry without bit v1[63],
  // v[2:3] is original user-data
  // ttmp1[25] = buffer_id
  // ttmp[2:3] = original v[0:1]
  // ttmp[4:5] = free
  // ttmp[10:11] holds original shader's data
  // ttmp13 is free
  // ttmp[14:15]=tma
  // EXEC=0x1

  // Before restoring other vector registers, we need to ensure the perf_snapshot sampling lock
  // has been released in case this stochastic sample is lost. Otherwise, we might never release a
  // perf_snapshot sampling lock, causing no other waves to be sampled by perf_snapshot.

  // Testing if the trap is caused by perf_snapshot (stochastic sampling HW).
  s_getreg_b32      ttmp13, hwreg(HW_REG_EXCP_FLAG_PRIV)                // On gfx12, EXCP_FLAG_PRIV.b7
  s_bitcmp1_b32     ttmp13, SQ_WAVE_EXCP_FLAG_PRIV_PERF_SNAPSHOT_SHIFT  // Test perf_snapshot (stochastic) bit.
  s_cbranch_scc0    .lost_sample_restore                                // If not, skip lock release
  s_getreg_b32      ttmp13, hwreg(HW_REG_PERF_SNAPSHOT_PC_HI)           // Otherwise, read PC_HI register to release snapshot lock
  s_branch .lost_sample_restore                                         // and branch to restore original user shader state

.process_sample:
  // Register state before calculating the sample buffer address:
  // ttmp1[25] = buffer_id
  // ttmp[2:3] = backup of original shader's v[0:1]
  // ttmp4 = sample_index_for_current_sample (from v0)
  //         Free to use, unless we override v0
  // ttmp5 = buf_size
  // ttmp13 free
  // ttmp[14:15] = base_address_of_pcs_sampling_data_t (TMA)
  // v[0:1].lane0 = sample index value from atomic
  // v[2:3] = original user shader's v[2:3] values

  // backup ttmp[2:3] to (exec_lo, ttmp4)
  s_mov_b32 exec_lo, ttmp2
  s_mov_b32 ttmp4, ttmp3

  // Calculate the base address of the selected sample buffer (buffer0 or buffer1).
  // The buffers are located after the pcs_sampling_data_t struct header (0x40 bytes).
  // Formula: TMA + 0x40 + (buffer_id * buf_size * 64_bytes_per_sample)
  // Get buffer_id (0 or 1) from ttmp1[25] into a scratch register.
  s_bfe_u32         ttmp13, ttmp1, (TTMP1_BUF_ID_BIT_POSITION | (1 << 16)) // ttmp13 = buffer_id

  // Calculate the byte offset for the selected buffer: buf_size * buffer_id
  // Result is a 64-bit value in ttmp[2:3].
  s_mul_i32         ttmp2, ttmp5, ttmp13                   // ttmp2 = buf_size * buffer_id (low 32 bits)
  s_mul_hi_u32      ttmp3, ttmp5, ttmp13                   // ttmp3 = buf_size * buffer_id (high 32 bits)

  // Multiply by 64 bytes per sample slot (shift left by 6 bits)
  // This converts from units of samples to units of bytes
  s_lshl_b64        ttmp[2:3], ttmp[2:3], 6                 // ttmp[2:3] = buf_size * buffer_id * 64
  // Add the size of the pcs_sampling_data_t header to get the total offset from TMA.
  // The sample buffers start right after the header.
  s_add_u32         ttmp2, ttmp2, SAMPLE_OFF_BYTES_PER_SAMPLE // ttmp2 = total_offset_lo = buf_size * buffer_id * 64 + SAMPLE_OFF_BYTES_PER_SAMPLE
  s_addc_u32        ttmp3, ttmp3, 0                           // ttmp3 = total_offset_hi = buf_size * buffer_id * 64 + SAMPLE_OFF_BYTES_PER_SAMPLE + carry
  // Calculate the final buffer base address: TMA + total_offset.
  // Store the result in ttmp[4:5], which are free.
  s_add_u32         ttmp2, ttmp14, ttmp2                    // ttmp2 = TMA_base_lo + total_offset_lo. This is low part of &bufferX
  s_addc_u32        ttmp3, ttmp15, ttmp3                    // ttmp3 = TMA_base_hi + total_offset_hi + carry. This is high part of &bufferX
                                                            // ttmp[2:3] now correctly points to the base of the selected sample buffer array

.fill_sample_common:
  // This is a common path for filling fields shared by host-trap and stochastic PC sampling:
  // timestamp, exec, workgroup information, HW_ID, and correlation ID.
  //
  // At this point, v[0:1] is local_entry (but v1 is 0)
  // v[2:3] is original user-data
  // ttmp[2:3] holds &buffer
  // exec_lo holds the backup of the v0
  // ttmp4 holds the backup of the v1
  // ttmp5 is free
  // ttmp13 is free
  // ttmp[10:11] contains user shader backup
  // [ttmp14:15]=‘tma', ttmp1[25] = buf_to_use
  v_readlane_b32    ttmp13, v0, 0                              // v[0] = local_entry (from v[0])
  s_mul_i32         ttmp5, ttmp13, SAMPLE_OFF_BYTES_PER_SAMPLE  // ttmp5 = local_entry * SAMPLE_OFF_BYTES_PER_SAMPLE
  s_mul_hi_u32      ttmp13, ttmp13, SAMPLE_OFF_BYTES_PER_SAMPLE // ttmp13 = local_entry * SAMPLE_OFF_BYTES_PER_SAMPLE (high part)
  s_add_u32         ttmp2, ttmp2, ttmp5                      //
  s_addc_u32        ttmp3, ttmp3, ttmp13                     // ttmp[2:3] = &bufferX[local_entry]
  v_writelane_b32   v0, ttmp2, 0x0                           //
  v_writelane_b32   v1, ttmp3, 0x0                           // v[0:1] = &buffer[local_entry]

  s_sendmsg_rtn_b64 ttmp[2:3], sendmsg(MSG_RTN_GET_REALTIME)// Get the current timestamp
  s_wait_kmcnt      0                                       // Wait for timestamp

  v_readlane_b32    ttmp5, v2, 0x0                           // ttmp5 and ttmp13 now holds backup of
  v_readlane_b32    ttmp13, v3, 0x0                          // user-data from v[2:3]

  // v[0:1] = &buffer[local_entry]
  // v[2:3] free
  // ttmp[2:3] holds sample timestamp we want to store
  // exec_lo holds the backup of the v0
  // ttmp4 holds backup of v1
  // ttmp5 holds the backup of v2
  // ttmp13 holds the backup of v3
  // ttmp[14:15]=tma, ttmp1[25] = buf_to_use

  v_writelane_b32   v2, ttmp2, 0                            // bring output data to v[2:3]
  v_writelane_b32   v3, ttmp3, 0                            // v[2:3] = timestamp

  // ttmp[2:3] now free after moving to v[2:3], so return ttmp[2:5] = v[0:3]
  s_mov_b32         ttmp2, exec_lo
  s_mov_b32         ttmp3, ttmp4
  s_mov_b32         ttmp4, ttmp5
  s_mov_b32         ttmp5, ttmp13
  s_mov_b64         exec, 1                                 // Set exec to lane 0 for vector stores

  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_TIMESTAMP, scope:SCOPE_SYS // store out timestamp

  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader's v[0:1]
  // ttmp[4:5] holds backup of original shader's v[2:3]
  // ttmp[10:11] holds backups from user shader
  // ttmp13 is free
  // ttmp[14:15]=‘tma', ttmp1[25] = buf_to_use
  // EXEC is 0x1

  // Save exec for the sample. exec_lo is inside ttmp10, while exec_hi in ttmp11 for gfx120*
  v_writelane_b32   v2, ttmp10, 0                            // v[2] = exec_lo
.if .amdgcn.gfx_generation_minor >= 5
  v_mov_b32         v3, 0                                    // exec_hi is not used in wave32
.else
  v_writelane_b32   v3, ttmp11, 0                            // v[3] = exec_hi
.endif
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_EXEC_LOHI, scope:SCOPE_SYS  // store out original EXEC

  // Store Workgroup ID X and Y at offset SAMPLE_OFF_WGID_XY (0x10).
  // ttmp9 = WGID_X (from first-level handler).
  // ttmp7 contains WGID_Y in low 16 bits.
  v_writelane_b32   v2, ttmp9, 0                             // wg_id_x
  s_bfe_u32         ttmp13, ttmp7, (0 | 16<<16)              // extract bits tttmp7[15:0] representing wg_id_y
  v_writelane_b32   v3, ttmp13, 0                            // wg_id_y
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_WGID_XY, scope:SCOPE_SYS  // store wg_id_x and wg_id_y

  // Store Workgroup ID Z at offset 0x18 (32-bit).
  // ttmp7 contains WGID_Z in high 16 bits [31:16].
  s_bfe_u32         ttmp13, ttmp7, (16 | (16 << 16))         // Extract WGID_Z[15:0] from ttmp7[31:16]
  v_writelane_b32   v2, ttmp13, 0                            // Store WGID_Z in v2
  global_store_b32  v[0:1], v2, off, offset:SAMPLE_OFF_WGID_Z, scope:SCOPE_SYS  // store wg_id_z

  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader's v[0:1]
  // ttmp[4:5] holds backup of original shader's v[2:3]
  // ttmp[10:11] holds original shader's information (save/restore)
  // ttmp13 is free
  // ttmp[14:15]=tma, ttmp1[25] = buf_to_use
  // EXEC is 0x1

  // Store HW_ID values spanned across multiple registers
  // Current ROCr API determines single dword for HW_ID, while this information is scattered across:
  //    gfx12.0: two dword registers HW_ID1 and HW_ID2 on GFX10+ architectures.
  // >= gfx12.5: three registers HW_ID1, HW_ID2, and AID_ID
  // Thus, we combine values from multiple registers listed abot into a single dword HW_ID with
  // the following layout:
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
  s_getreg_b32      ttmp13, hwreg(HW_REG_HW_ID1)
  v_and_b32         v2, ttmp13, 0x1feffcff               // Mask to extract fields from HW_ID1 (WAVE_ID, WGP_ID, SA_ID, SE_ID on GFX12.0)
  v_and_b32         v3, ttmp13, 0x00000300               // Mask to extract SIMD_ID[9:8]
  v_lshl_or_b32     v2, v3, 6, v2                        // Shift SIMD_ID to bits [15:14]
  s_getreg_b32      ttmp13, hwreg(HW_REG_HW_ID2)         // Get HW_ID2
  v_and_b32         v3, ttmp13, 0x0f000000               // Mask to extract WAVE_ID[27:24]
  v_lshl_or_b32     v2, v3, 4, v2                        // Shift WAVE_ID to bits [4:0]
  v_and_b32         v3, ttmp13, 0x001f0000               // Mask to extract WG_ID[20:16]
  v_lshl_or_b32     v2, v3, 7, v2                        // Shift WG_ID to bits [27:23]
  v_and_b32         v3, ttmp13, 0x00000100               // Mask to extract ME_ID[8]
  v_lshl_or_b32     v2, v3, 9, v2                        // Shift ME_ID to bit [17]
  v_and_b32         v3, ttmp13, 0x00000030               // Mask to extract PIPE_ID[5:4]
  v_lshl_or_b32     v2, v3, 16, v2                       // Shift PIPE_ID to bits [21:20]
  v_and_b32         v3, ttmp13, 0x0000000f               // Mask to extract QUEUE_ID[3:0]
  v_lshl_or_b32     v2, v3, 5, v2                        // Shift QUEUE_ID to bits [8:5]

.if .amdgcn.gfx_generation_minor >= 5
  // SE_ID resides in MSG_RTN_GET_SE_AID_ID[3:0].
  
  s_sendmsg_rtn_b32 ttmp13, sendmsg(MSG_RTN_GET_SE_AID_ID)
  v_and_b32         v2, 0xFFF3FFFF, v2                        // Clear SE_ID bits [19:18] in v2 while waiting
  s_wait_kmcnt      0

  // Cache the full SE_AID_ID in v3 to avoid a second message later (for XCC_ID extraction)
  v_writelane_b32   v3, ttmp13, 0

  // Extract and position SE_ID bits
  s_bfe_u32         ttmp13, ttmp13, (0 | (2 << 16))      // Extract lower 2 bits from the SE_ID[3:0] from ttmp13
  s_lshl_b32        ttmp13, ttmp13, 18                   // Shift to bit position 18
  v_or_b32          v2, ttmp13, v2                       // OR the new SE_ID bits into v2

.endif
  // Store HW_ID information
  global_store_b32  v[0:1], v2, off, offset:SAMPLE_OFF_HW_ID, scope:SCOPE_SYS

  // The following is still true
  // v[0:1] = &buffer[local_entry]
  // v2 = free, v3 = cached SE_AID_ID (gfx12.5+), free (gfx12.0)
  // ttmp[2:3] holds backup of original shader's v[0:1]
  // ttmp[4:5] holds backup of original shader's v[2:3]
  // ttmp[10:11] holds original shader's information (save/restore)
  // ttmp13 is free
  // ttmp[14:15]=tma, ttmp1[25] = buf_to_use
  // EXEC is 0x1

  // Store wave_in_group and chiplet information in the following format:
  // Bits [5:0]   = wave_in_wg (5 bits from ttmp8[29:25])
  // bits [10:8]  = chiplet (zero on gfx12.0)
  // Bits [7:6] and [31:11] = reserved and must be zero

  s_bfe_u32         ttmp13, ttmp8, (WAVE_ID_WG_BIT_POSITION | (5 << 16)) // Extract 5 bits (use ttmp13)
  v_writelane_b32   v2, ttmp13, 0                             // Store wave_in_group in v2

.if .amdgcn.gfx_generation_minor >= 5
  // Chiplet (XCC_ID) resides in MSG_RTN_GET_SE_AID_ID[19:16].
  // NOTE: on gfx12.0 there are no chiplets

  // Reuse SE_AID_ID value cached in v3 to avoid redundant message traffic
  v_readlane_b32    ttmp13, v3, 0

  s_bfe_u32         ttmp13, ttmp13, (16 | (3 << 16)) // Extract lower 3 bits from Virtual_XCC_ID[19:16]
  s_lshl_b32        ttmp13, ttmp13, 8                // Shift to bit position [10:8]
  v_or_b32          v2, ttmp13, v2                   // move chiplet into the v2 already containing wave_in_group
.endif

  // Write wave_in_group and chiplet
  global_store_b32  v[0:1], v2, off, offset:SAMPLE_OFF_WAVE_IN_GROUP_CHIPLET, scope:SCOPE_SYS

  // The following is still true
  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader's v[0:1]
  // ttmp[4:5] holds backup of original shader's v[2:3]
  // ttmp[10:11] holds original shader's information (save/restore)
  // ttmp13 is free
  // ttmp[14:15]=tma, ttmp1[25] = buf_to_use
  // EXEC is 0x1

  // Store the correlation ID contained of dispatch_id (ttmp8[24:0]) + doorbell_id
  s_and_b32         ttmp13, ttmp8, TTMP8_DISPATCH_ID_MASK
  v_writelane_b32   v2, ttmp13, 0
  s_sendmsg_rtn_b32 ttmp13, sendmsg(MSG_RTN_GET_DOORBELL)
  s_wait_kmcnt      0
  s_and_b32         ttmp13, ttmp13, DOORBELL_ID_MASK
  v_writelane_b32   v3, ttmp13, 0
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_CORRELATION, scope:SCOPE_SYS

  // The following is still true
  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader's v[0:1]
  // ttmp[4:5] holds backup of original shader's v[2:3]
  // ttmp[10:11] holds original shader's information (save/restore)
  // ttmp13 is free
  // ttmp[14:15]=tma, ttmp1[25] = buf_to_use
  // EXEC is 0x1

  // Check perf_snapshot bit to determine if a trap caused by stochastic sampling.
  s_getreg_b32      ttmp13, hwreg(HW_REG_EXCP_FLAG_PRIV)     // On gfx12, EXCP_FLAG_PRIV.b7
  s_bitcmp1_b32     ttmp13, SQ_WAVE_EXCP_FLAG_PRIV_PERF_SNAPSHOT_SHIFT // Test Performance Snapshot bit.
  s_cbranch_scc1    .fill_sample_stoch  // jump if a trap is caused by the perf_snapshot block

.fill_sample_ht:
  // The following is still true
  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader's v[0:1]
  // ttmp[4:5] holds backup of original shader's v[2:3]
  // ttmp[10:11] holds original shader's information (save/restore)
  // ttmp13 is free
  // ttmp[14:15]=tma, ttmp1[25] = buf_to_use
  // EXEC is 0x1

  v_writelane_b32   v2, ttmp0, 0                             // v[2] = PC_LO
  v_writelane_b32   v3, ttmp1, 0                             // v[3] = PC_HI
  v_and_b32         v3, v3, SQ_WAVE_PC_HI_ADDRESS_MASK       // clear out PC_HI's MSBs scratch bits
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_PC_HOST, scope:SCOPE_SYS  // store out PC

  // Ensure all stores have completed before returning and incrementing written_val
  s_wait_storecnt   0

  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader's v[0:1]
  // ttmp[4:5] holds backup of original shader's v[2:3]
  // ttmp[10:11] holds original shader's information (save/restore)
  // ttmp13 is free
  // ttmp[14:15]=tma, ttmp1[25] = buf_to_use
  // EXEC is 0x1
  s_branch          .ret_from_fill_sample

.fill_sample_stoch:
  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader's v[0:1]
  // ttmp[4:5] holds backup of original shader's v[2:3]
  // ttmp[10:11] holds original shader's information (save/restore)
  // ttmp13 is free
  // ttmp[14:15]=tma, ttmp1[25] = buf_to_use
  // EXEC is 0x1

  // Read performance SNAPSHOT registers and store at offset 0x28 (SAMPLE_OFF_SNAPSHOT_DATA + 4)
  s_getreg_b32      ttmp13, hwreg(HW_REG_PERF_SNAPSHOT_DATA1)       // Read snapshot data register 1
  v_writelane_b32   v2, ttmp13, 0x0                                 // stash DATA1 in v2
  s_getreg_b32      ttmp13, hwreg(HW_REG_PERF_SNAPSHOT_DATA2)       // Read snapshot data register 2
  v_writelane_b32   v3, ttmp13, 0x0                                 // stash DATA2 in v3
  // Store snapshot DATA1 and DATA2 at offset SAMPLE_OFF_SNAPSHOT_DATA + 4
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_SNAPSHOT_DATA + 4, scope:SCOPE_SYS

.if .amdgcn.gfx_generation_minor >= 5
  // On gfx1250+, the PC_HI must be read last to release the perf_snapshot lock.
  // Store main snapshot data at offset 0x24 (SAMPLE_OFF_SNAPSHOT_DATA)
  s_getreg_b32      ttmp13, hwreg(HW_REG_PERF_SNAPSHOT_DATA)
  v_writelane_b32   v2, ttmp13, 0
  global_store_b32  v[0:1], v2, off, offset:SAMPLE_OFF_SNAPSHOT_DATA, scope:SCOPE_SYS  // store perf snapshot DATA
.endif

  // For stochastic sampling, use PC from snapshot registers (actual sampled instruction)
  // Trap PC points to trap handler entry, not the interrupted instruction
  s_getreg_b32      ttmp13, hwreg(HW_REG_PERF_SNAPSHOT_PC_LO)       // Read performance snapshot PC_LO register
  v_writelane_b32   v2, ttmp13, 0x0                                 // stash PC_LO in v2
  s_getreg_b32      ttmp13, hwreg(HW_REG_PERF_SNAPSHOT_PC_HI)       // Read performance snapshot PC_HI register
  v_writelane_b32   v3, ttmp13, 0x0                                 // stash PC_HI in v3
  // Store at offset 0x00 (SAMPLE_OFF_PC_HOST)
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_PC_HOST, scope:SCOPE_SYS

.if .amdgcn.gfx_generation_minor == 0
  // As the perf_snapshot lock doesn't exist on GFX120*, we need to read DATA last as it contains the VALID bit.
  // Store main snapshot data at offset 0x24 (SAMPLE_OFF_SNAPSHOT_DATA)
  s_getreg_b32      ttmp13, hwreg(HW_REG_PERF_SNAPSHOT_DATA)
  v_writelane_b32   v2, ttmp13, 0
  global_store_b32  v[0:1], v2, off, offset:SAMPLE_OFF_SNAPSHOT_DATA, scope:SCOPE_SYS  // store perf snapshot DATA
.endif

  // Ensure all stores have completed before returning and incrementing written_val
  s_wait_storecnt   0

  // SAMPLE DATA COMPLETION AND BUFFER MANAGEMENT
  // This section handles incrementing the written sample count and
  // signaling the host when watermark is reached.

.ret_from_fill_sample:
  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader's v[0:1]
  // ttmp[4:5] holds backup of original shader's v[2:3]
  // ttmp[10:11] holds original shader's information (save/restore)
  // ttmp13 is free
  // ttmp[14:15]=tma, ttmp1[25] = buf_to_use
  // EXEC is 0x1

  // Calculate offset to buf_written_val for current buffer
  // buf_written_val0 at offset 0x10, buf_written_val1 at offset 0x20
  s_bfe_u32         ttmp13, ttmp1, (TTMP1_BUF_ID_BIT_POSITION | 1 << 16) // Extract buffer_id from ttmp1[25] into scratch register
  s_mulk_i32        ttmp13, SAMPLE_OFF_BUF_WRITTEN_VAL              // Multiply buffer_id by 16 (0x10) to get offset
  s_add_u32         ttmp14, ttmp14, ttmp13                          // Add offset to TMA base (low)
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
  // v0 = done (previous buff_written_val index), v1 = free, v[2:3] = free
  // ttmp[2:3] holds backup of original shader's v[0:1]
  // ttmp[4:5] holds backup of original shader's v[2:3]
  // ttmp[10:11] holds original shader's [exec_lo,exec_hi]
  // ttmp13 is free
  // ttmp[14:15] = TMA + buffer_id * SAMPLE_OFF_BUF_WRITTEN_VAL
  // EXEC=0x1

  s_mov_b32         exec_lo, ttmp4  // backup v2 to exec_lo
  s_mov_b32         ttmp13, ttmp5   // backup v3 to ttmp13

  s_load_b32        ttmp5, ttmp[14:15], SAMPLE_OFF_WATERMARK_FIELD, scope:SCOPE_CU            // load watermark threshold into ttmp5
  v_readlane_b32    ttmp4, v0, 0                            // Get previous written count
  s_add_u32         ttmp4, ttmp4, 1                         // Find current sample count (previous + 1)
  s_wait_kmcnt      0                                       // wait for watermark to load

  // Check if we should signal the host (only trap handler instances that observes ttmp4 == tmp5 signals host)
  s_cmp_lg_u32      ttmp4, ttmp5                            // Compare buff_written_val with watermark (fails if ttmp4 == ttmp5)

  // Restore user data and execution state
  s_mov_b32         ttmp4, exec_lo                          // restore v2 to ttmp4
  s_mov_b32         ttmp5, ttmp13                           // restore v3 to ttmp5 
  s_mov_b64         exec, 1                                 // Set exec to lane 0 only

  s_cbranch_scc1    .restore_vector_before_exit_trap        // Skip signaling if below/above watermark (ttmp4 != ttmp5 succeeds)

  // Host signalling part when whatermark is reached
.send_signal:
  // v[0:3] = free, ttmp[2:5] = backups of original v[0:3]
  // ttmp[10:11] holds original shaders data
  // ttmp[14:15]=buf_written_valX-0x10, EXEC=0x1
  // ttmp13 is free
  // EXEC=0x1
  // write done-signal and optional interrupt

  // Watermark reached or exceeded. Signal the host.
  // Load the hsa_signal_t handle for the current buffer.
  // done_sig0 is at offset 0x18. done_sig1 is at 0x28.
  // addr = ttmp[14:15] + 0x18 + (buffer_id * 0x10).
  s_load_b64        ttmp[14:15], ttmp[14:15], SAMPLE_OFF_DONE_SIG0, scope:SCOPE_CU // load done_sig into ttmp[14:15]
  s_wait_kmcnt      0                                       // Wait for done signal to load

  // Zero out the signal value to notify host
  v_mov_b32         v0, 0                                   // v[0] = 0 (value to store)
  v_mov_b32         v1, 0                                   // value to store into v[0:1]
  v_writelane_b32   v2, ttmp14, 0                           // v[2] = done signal address (low part)
  v_writelane_b32   v3, ttmp15, 0                           // Put signal address into v[2:3]

  // Write to signal value field (offset 0x8 within signal structure); namely amd_signal_t.value=v[0:1]
  global_store_b64  v[2:3], v[0:1], off, offset:SAMPLE_OFF_SIGNAL_VALUE, scope:SCOPE_SYS

  // Load event ID and mailbox pointer for interrupt generation
  s_load_b32        ttmp13, ttmp[14:15], SAMPLE_OFF_SIGNAL_EVENT_ID, scope:SCOPE_CU // load event_id into ttmp13
  s_load_b64        ttmp[14:15], ttmp[14:15], SAMPLE_OFF_EVENT_MAILBOX_PTR, scope:SCOPE_CU     // load event mailbox ptr into 14:15
  s_wait_kmcnt      0

  // Check if interrupt should be sent (null mailbox or zero event_id means no interrupt)
  s_cmp_eq_u64      ttmp[14:15], 0                          // null mailbox means no interrupt
  s_cbranch_scc1    .restore_vector_before_exit_trap

  s_cmp_eq_u32      ttmp13, 0                               // event_id zero means no interrupt
  s_cbranch_scc1    .restore_vector_before_exit_trap

  v_writelane_b32   v2, ttmp14, 0                           // v[2] = mailbox address (low part)
  v_writelane_b32   v3, ttmp15, 0                           // v[3] = mailbox address (high part)

  s_wait_storecnt   0                                       // wait for signal value 0 to be written to amd_signal_t.value

  v_writelane_b32   v0, ttmp13, 0x0                         // v[0] = 0 (event ID low part)
  global_store_b32  v[2:3], v0, off, offset:0x0, scope:SCOPE_SYS // Send event ID to the mailbox
  s_wait_storecnt   0
  s_mov_b32         ttmp14, m0                              // Backup m0 (event ID low part) to ttmp14
  v_readlane_b32    ttmp15, v0, 0                           // Read event ID low part from v0 into ttmp15
  s_mov_b32         m0, ttmp15                              // Set m0 to event ID (low part)
  s_sendmsg         sendmsg(MSG_INTERRUPT)                  // send interrupt message to host
  s_wait_kmcnt      0                                       // Wait for interrupt to complete
  s_mov_b32         m0, ttmp14                              // Restore m0 to original value (event ID low part)

  // v[0:1] = free
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader's v[0:1]
  // ttmp[4:5] holds backup of original shader's v[2:3]
  // ttmp[10:11] holds original shader data
  // ttmp13 is free
  // ttmp[14:15] is free
  // EXEC=1

.restore_vector_before_exit_trap:
  // v[0:3] = free
  // ttmp[2:5] = backup of the user's v[0:3]
  // ttmp[10:11] users data to backup
  // ttmp13 is free
  // ttmp[14:15] = TMA + buffer_id * SAMPLE_OFF_BUF_WRITTEN_VAL (or free if we come from above .send_signal path)

  // Restore original v[2:3] from ttmp[4:5]
  v_writelane_b32   v2, ttmp4, 0                            // restore v[2:3] to user data
  v_writelane_b32   v3, ttmp5, 0                            // v[2:3] = original user data

.lost_sample_restore:
  // v[0:1] contains local_entry when branched from .lost_sample; otherwise free.
  // v[2:3] is original user-data
  // ttmp1[25] = buffer_id
  // ttmp[2:3] = original v[0:1]
  // ttmp[4:5] = free
  // ttmp[10:11] holds original shader's data (EXEC mask)
  // ttmp13 is free
  // ttmp[14:15]=tma
  // EXEC=0x1

  // restore v[0:1] from ttmp[2:3]
  v_writelane_b32   v0, ttmp2, 0                            // restore v[0:1] to user data
  v_writelane_b32   v1, ttmp3, 0                            // v[0:1] = original user data

  // zero out ttmp1[25] holding buff_id
  s_bitset0_b32     ttmp1, TTMP1_BUF_ID_BIT_POSITION

.if .amdgcn.gfx_generation_minor >= 5
.if  .amdgcn.gfx_generation_minor == 5
  // Restore MODE.MSB
  s_setreg_b32 hwreg(HW_REG_WAVE_MODE, SQ_WAVE_MODE_MSB_SHIFT, SQ_WAVE_MODE_MSB_SIZE), ttmp11
.endif
  // Restore XNACK_MASK
  s_setreg_b32 hwreg(HW_REG_XNACK_MASK), ttmp12

  // Restore exec_lo
  s_mov_b32         exec_lo, ttmp10                         // restore exec mask low bits
.else
  // Restore exec on gfx12.0
  s_mov_b64         exec, ttmp[10:11]
  // Restore ttmp11 from ttmp12 (SCHED_MODE[1:0] at bits [27:26], saved before exec_hi overwrite)
  s_mov_b32         ttmp11, ttmp12
.endif

.exit_pcs_trap:
  // As we don't support host-trap and stochastic at the same time, no need for additional check.
  // Thus, proceed with clearing both bits.
  // Clear the Host Trap flag in the hardware register to acknowledge the event
  s_setreg_imm32_b32 hwreg(HW_REG_EXCP_FLAG_PRIV, SQ_WAVE_EXCP_FLAG_PRIV_HT_SHIFT,1), 0
  // Clear the perf_snapshot flag
  s_setreg_imm32_b32 hwreg(HW_REG_EXCP_FLAG_PRIV, SQ_WAVE_EXCP_FLAG_PRIV_PERF_SNAPSHOT_SHIFT,1), 0

  // v[0:3] original user data
  // ttmp[0:1]
  //      gfx12.0: 0..., SCHED_MOD[1:0], 0..., original PC
  //   >= gfx12.5: original PC (no trap_id)
  // ttmp[2:5] free
  // ttmp10 is free (ttmp11 is free on gfx12.5)
  // ttmp12 is free
  // ttmp13 is free
  // ttmp[14:15] is free
  // EXEC=original user shader exec mask

.exit_trap:

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
  // Zero out trap_id[3:0] and scratch bits (if any) from ttmp1.
  // Two cases worth noting:
  // 1. perf_snapshot stochastic trap SQ_WAVE_EXCP_FLAG_PRIV_PERF_SNAPSHOT_SHIFT
  //    and `s_trap NON_ZERO_TRAP_ID` occured simultaneously.
  //    In that case, we'll process a stochastic trap, remove the TRAP_ID here,
  //    and execute s_rfe. As the PC inside ttmp[1:0] is not advanced, the `s_trap NON_ZERO_TRAP_ID`
  //    will be re-executed and properly processed in the trap handler reentry.
  // 2. host-trap occured - trap_id is zero for host-trap, so the following would clean only scratch bits.
  s_and_b32         ttmp1, ttmp1, SQ_WAVE_PC_HI_ADDRESS_MASK
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

.if .amdgcn.gfx_generation_minor == 0
  // Extract SCHED_MODE from ttmp11 so we can restore it just before s_rfe.
  s_bfe_u32         ttmp2, ttmp11, TTMP11_SCHED_MODE_BFE
.endif

  // Restore SQ_WAVE_STATUS.
  s_and_b64         exec, exec, exec                        // Restore STATUS.EXECZ, not writable by s_setreg_b32
  s_and_b64         vcc, vcc, vcc                           // Restore STATUS.VCCZ, not writable by s_setreg_b32

  // Restore SQ_WAVE_STATE_PRIV: only SCC (bit 9), as the trap handler
  // does not modify other SQ_WAVE_STATE_PRIV bits.
  // SCC was saved in ttmp6[31] at trap entry.
  s_lshr_b32        ttmp3, ttmp6, TTMP6_SCC_SHIFT
  s_setreg_b32      hwreg(HW_REG_STATE_PRIV, SQ_WAVE_STATE_PRIV_SCC_SHIFT, 1), ttmp3

  // Zero out bit 31 in a persistent ttmp6 register before s_rfe.
  s_bitset0_b32     ttmp6, TTMP6_SCC_SHIFT

.if .amdgcn.gfx_generation_minor == 0
  s_setreg_b32      hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), ttmp2
.endif

  // Return to original (possibly modified) PC.
  s_rfe_b64         [ttmp0, ttmp1]

.parked:
  s_trap            0x2
  s_branch          .parked

// Add s_code_end padding so instruction prefetch always has something to read.
.rept (256 - ((. - trap_entry) % 64)) / 4
  s_code_end
.endr
