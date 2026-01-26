////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2024, Advanced Micro Devices, Inc. All rights reserved.
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
.endif
// The PC is dword (32bit) aligned, so the 2 LSBs are always zero.
.set SQ_WAVE_PC_LO_ADDRESS_MASK                    , 0xFFFFFFFC
.set SQ_WAVE_PC_HI_TRAP_ID_BFE                     , (SQ_WAVE_PC_HI_TRAP_ID_SHIFT | (SQ_WAVE_PC_HI_TRAP_ID_SIZE << 16))
.set SQ_WAVE_PC_HI_TRAP_ID_SHIFT                   , 28
.set SQ_WAVE_PC_HI_TRAP_ID_SIZE                    , 4
.set SQ_WAVE_PC_HI_TRAP_ID_MASK                    , (((1 << SQ_WAVE_PC_HI_TRAP_ID_SIZE) - 1) << SQ_WAVE_PC_HI_TRAP_ID_SHIFT)

.set SQ_WAVE_STATE_PRIV_HALT_BFE                   , (SQ_WAVE_STATE_PRIV_HALT_SHIFT | (1 << 16))
.set SQ_WAVE_STATE_PRIV_HALT_SHIFT                 , 14
.set SQ_WAVE_STATE_PRIV_BARRIER_COMPLETE_SHIFT     , 2

.set TRAP_ID_ABORT                                 , 2
.set TRAP_ID_DEBUGTRAP                             , 3

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

// ABI between first and second level trap handler:
//
//   gfx12
//     { ttmp1, ttmp0 } = TrapID[3:0], zeros, PC[47:0]
//     ttmp11 = 0[7:0], DebugEnabled[0], 0[15:0], NoScratch[0], 0[5:0]
//
//     ttmp12 = SQ_WAVE_STATE_PRIV (Private wave state register value).
//     ttmp14 = TMA[31:0] - TMA_LO (Trap Memory Argument Low - base address for trap handler data, low 32 bits).
//     ttmp15 = TTMA[63:32] - TMA_HI (Trap Memory Argument High - base address for trap handler data, high 32 bits).
//
// Restricted register list:
//   gfx12:
//     ttmp[0:1] - Must be preserved for RFE
//     ttmp[7:9] - Contain workgroup information, must be preserved
//     ttmp12    - Contains SQ_WAVE_STATE_PRIV, must be preserved
//     ttmp[14:15] - Contain TMA address, must be preserved
//
// Safe to use as scratch:
//   gfx12: ttmp[2:6], ttmp10, ttmp13

 trap_entry:
  s_mov_b32         ttmp3, 0                                 // Clear ttmp3 as it will contain the exception code

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
  s_getreg_b32      ttmp2, hwreg(HW_REG_EXCP_FLAG_PRIV)     // EXCP_FLAG_PRIV.b10=stochastic_sample_trap
  s_bitcmp1_b32     ttmp2, SQ_WAVE_EXCP_FLAG_PRIV_PERF_SNAPSHOT_SHIFT // Test Performance Snapshot bit.

  s_cbranch_scc0    .handle_sw_trap                       // If not Stochastic, continue to check trap ID

  s_load_b64        ttmp[14:15], ttmp[14:15], 0x8, scope:SCOPE_CU  // ttmp[14:15]=*stoch_trap_buf
  s_wait_kmcnt      0
  s_branch          .profile_trap_handlers

  // To avoid more overhead on the critical sample processing path, we decided to give a priority
  // to host-trap and perf_snapshot trap over the s_trap and halt.
  // For host-trap, trap_id should always be zero.
  // For stochastic, it's possible that both SQ_WAVE_EXCP_FLAG_PRIV_PERF_SNAPSHOT_SHIFT and `s_trap NON_ZERO_TRAP_ID`
  // occured simultaneously. In that case, we'll process a stochastic trap, remove the TRAP_ID
  // and execute s_rfe. As the PC inside ttmp[1:0] is not advanced, then the `s_trap NON_ZERO_TRAP_ID`
  // will be rexecuted and NON_ZERO_TRAP_ID will be properly processed.
  // TODO: If wave is halted, should we process sample or just ignore it?

.handle_sw_trap:
  // Check if this is a trap (s_trap instruction) or a hardware exception.
  // Extract TrapID from ttmp1 (which contains PC_HI).
  // Branch if not a trap (an exception instead).
  s_bfe_u32         ttmp2, ttmp1, SQ_WAVE_PC_HI_TRAP_ID_BFE // ttmp2 = TrapID
  s_cbranch_scc0    .check_exceptions			                  // If TrapID is 0, it's an exception, so branch.

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
  // ttmp11: 1st_level_ttmp11[31:23] 0[15:0] 1st_level_ttmp11[6:0]
  //
  // After parking the wave:
  //
  // ttmp10: pc_lo[31:0]
  // ttmp11: 1st_level_ttmp11[31:23] pc_hi[15:0] 1st_level_ttmp11[6:0]
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
  s_bitset1_b32     ttmp12, SQ_WAVE_STATE_PRIV_HALT_SHIFT  // FIXME: when we're allowed to halt a wave?

  // Initialize TTMP registers
  s_bitcmp1_b32     ttmp8, TTMP8_DEBUG_FLAG_SHIFT
  s_cbranch_scc1    .ttmps_initialized
  s_mov_b32         ttmp4, 0
  s_mov_b32         ttmp5, 0
  s_bitset1_b32     ttmp8, TTMP8_DEBUG_FLAG_SHIFT
.ttmps_initialized:
  s_branch          .exit_trap

.profile_trap_handlers:
  // GFX12.0 -> backup exec_hi into ttmp11
  s_mov_b32 ttmp11, exec_hi

  // TTMP assignments for profiler
  //
  // ttmp0 =
  //      gfx12.0: PC[31:2], 0, 0
  // ttmp1 =
  //  gfx12.0:
  //    ttmp1[31:28] = 0 (free to use),
  //    ttmp1[27:26] = SCHED_MODE[1:0]
  //    ttmp1[25]    = buff_id bit for PC sampling
  //    ttmp1[24:16] = 0 (free to use)
  //    ttmp1[15: 0] = PC[47:32]
  // ttmp6 =
  //      gfx12.0: free
  // ttmp7 = WGP Y/Z
  // ttmp8 = 0 (unusable), yz valid, wave_in_wg, dispatch_idx
  // ttmp9 = WGP X
  // ttmp10 = EXEC_LO
  // ttmp11 =
  //   gfx12.0: EXEC_HI
  // ttmp12 =
  //      gfx12.0: free to use
  // ttmp14 = TMA_LO-ish
  // ttmp15 = TMA_HI-ish
  //
  // v[0:3] contain user shader data that must be preserved/restored
  // exec: Contains user's execution mask
  //       gfx12.0: both exec_lo and exec_hi must be preserved
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

  // Before restoring other vector registers, we need to free perf_snapshot resources.

  // Testing if the trap is caused by perf_snapshot (stochastic sampling HW).
  s_getreg_b32      ttmp13, hwreg(HW_REG_EXCP_FLAG_PRIV)                // On gfx12, EXCP_FLAG_PRIV.b7
  s_bitcmp1_b32     ttmp13, SQ_WAVE_EXCP_FLAG_PRIV_PERF_SNAPSHOT_SHIFT  // Test perf_snapshot (stochastic) bit.
  s_cbranch_scc0    .lost_sample_restore                                // If not, just restore sample
  s_getreg_b32      ttmp13, hwreg(HW_REG_PERF_SNAPSHOT_PC_HI)           // Otherwise, free perf_snapshot resources
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
  v_writelane_b32   v3, ttmp11, 0                            // v[3] = exec_hi
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

  // Store HW_ID information
  global_store_b32  v[0:1], v2, off, offset:SAMPLE_OFF_HW_ID, scope:SCOPE_SYS

  // The following is still true
  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
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

  // Write wave_in_group and chiplet (0 on gfx12.0)
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

  // Clear out 2 LSBs of the PC_LO (used as scratch bits in ttmp0)
  v_writelane_b32   v2, ttmp0, 0                             // v[2] = PC_LO
  v_and_b32         v2, v2, SQ_WAVE_PC_LO_ADDRESS_MASK       // clear out scratch bits
  // Clear out 7 MSBs of PC_HI (used as scratch bits in ttmp1)
  v_writelane_b32   v3, ttmp1, 0                             // v[3] = PC_HI
  v_and_b32         v3, v3, SQ_WAVE_PC_HI_ADDRESS_MASK       // clear out scratch bits
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

  // Store main snapshot data at offset 0x24 (SAMPLE_OFF_SNAPSHOT_DATA)
  s_getreg_b32      ttmp13, hwreg(HW_REG_PERF_SNAPSHOT_DATA)
  v_writelane_b32   v2, ttmp13, 0
  global_store_b32  v[0:1], v2, off, offset:SAMPLE_OFF_SNAPSHOT_DATA, scope:SCOPE_SYS  // store perf snapshot DATA

  // For stochastic sampling, use PC from snapshot registers (actual sampled instruction)
  // Trap PC points to trap handler entry, not the interrupted instruction
  s_getreg_b32      ttmp13, hwreg(HW_REG_PERF_SNAPSHOT_PC_LO)       // Read performance snapshot PC_LO register
  v_writelane_b32   v2, ttmp13, 0x0                                 // stash PC_LO in v2
  s_getreg_b32      ttmp13, hwreg(HW_REG_PERF_SNAPSHOT_PC_HI)       // Read performance snapshot PC_HI register
  v_writelane_b32   v3, ttmp13, 0x0                                 // stash PC_HI in v3

  // Store at offset 0x00 (SAMPLE_OFF_PC_HOST)
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_PC_HOST, scope:SCOPE_SYS

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
  s_wait_kmcnt      0                                       // wait for watermark to load

  // Check if we should signal the host (only trap handler instances that observes ttmp4 == tmp5 signals host)
  s_cmp_lg_u32      ttmp4, ttmp5                            // Compare buff_written_val with watermark (fails if ttmp4 == ttmp5)

  // FIMXE: If we keep this piece of code, is it possible that multiple signals are sent for the same buffer???
  // s_add_u32         ttmp4, ttmp4, 1                         // Calculate current count (previous + 1)
  // s_cmp_lt_u32      ttmp4, ttmp5                            // if (current_sample_count < watermark), don't signal

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
  // v0 contains local_entry, v1 is free (or is free if we came from the .restore_vector_before_exit_trap path)
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

  // Restore exec on gfx12.0
  s_mov_b64         exec, ttmp[10:11]

  // TODO: check whether we need additional checking if either Host-Trap or Stochastic is set
  // (probably no, as we don't support both at the same time)
  // Clear the Host Trap flag in the hardware register to acknowledge the event
  s_setreg_imm32_b32 hwreg(HW_REG_EXCP_FLAG_PRIV, SQ_WAVE_EXCP_FLAG_PRIV_HT_SHIFT,1), 0
  // Clear the perf_snapshot flag
  s_setreg_imm32_b32 hwreg(HW_REG_EXCP_FLAG_PRIV, SQ_WAVE_EXCP_FLAG_PRIV_PERF_SNAPSHOT_SHIFT,1), 0 

  // v[0:3] original user data
  // ttmp[0:1]
  //      gfx12.0: 0..., SCHED_MOD[1:0], 0..., original PC
  // ttmp[2:5] free
  // ttmp10 is free (ttmp11 is free on gfx12.0)
  // ttmp13 is free
  // ttmp[14:15] is free
  // EXEC=original user shader exec mask

.exit_trap:
  // Restore SQ_WAVE_STATUS.
  s_and_b64         exec, exec, exec                        // Restore STATUS.EXECZ, not writable by s_setreg_b32
  s_and_b64         vcc, vcc, vcc                           // Restore STATUS.VCCZ, not writable by s_setreg_b32

  // Restore SQ_WAVE_STATUS.SCC and SQ_WAVE_STATUS.SLEEP_PRIV
  s_bfe_u32         ttmp2, ttmp12, (9 | (1 << 16))
  s_bfe_u32         ttmp3, ttmp12, (1 | (1 << 16))
  s_setreg_b32      hwreg(HW_REG_STATE_PRIV, 9, 1), ttmp2
  s_setreg_b32      hwreg(HW_REG_STATE_PRIV, 1, 1), ttmp3
  // TODO: what about halt bit?

  s_rfe_b64         [ttmp0, ttmp1]

.parked:
  s_trap            0x2
  s_branch          .parked

// Add s_code_end padding so instruction prefetch always has something to read.
.rept (256 - ((. - trap_entry) % 64)) / 4
  s_code_end
.endr
