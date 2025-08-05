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
.set DOORBELL_ID_SIZE                          , 10
.set DOORBELL_ID_MASK                          , ((1 << DOORBELL_ID_SIZE) - 1)
.set EC_QUEUE_WAVE_ABORT_M0                    , (1 << (DOORBELL_ID_SIZE + 0))
.set EC_QUEUE_WAVE_TRAP_M0                     , (1 << (DOORBELL_ID_SIZE + 1))
.set EC_QUEUE_WAVE_MATH_ERROR_M0               , (1 << (DOORBELL_ID_SIZE + 2))
.set EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION_M0      , (1 << (DOORBELL_ID_SIZE + 3))
.set EC_QUEUE_WAVE_MEMORY_VIOLATION_M0         , (1 << (DOORBELL_ID_SIZE + 4))
.set EC_QUEUE_WAVE_APERTURE_VIOLATION_M0       , (1 << (DOORBELL_ID_SIZE + 5))

.set SQ_WAVE_EXCP_FLAG_PRIV_ADDR_WATCH_MASK    , (1 << 4) - 1
.set SQ_WAVE_EXCP_FLAG_PRIV_MEMVIOL_SHIFT      , 4
.set SQ_WAVE_EXCP_FLAG_PRIV_ILLEGAL_INST_SHIFT , 6
.set SQ_WAVE_EXCP_FLAG_PRIV_HT_SHIFT           , 7
.set SQ_WAVE_EXCP_FLAG_PRIV_WAVE_START_SHIFT   , 8
.set SQ_WAVE_EXCP_FLAG_PRIV_WAVE_END_SHIFT     , 9
.set SQ_WAVE_EXCP_FLAG_PRIV_PERF_SNAPSHOT      , 10
.set SQ_WAVE_EXCP_FLAG_PRIV_TRAP_AFTER_INST_SHIFT , 11
.set SQ_WAVE_EXCP_FLAG_PRIV_XNACK_ERROR_SHIFT  , 12

.set SQ_WAVE_EXCP_FLAG_USER_MATH_EXCP_SHIFT    , 0
.set SQ_WAVE_EXCP_FLAG_USER_MATH_EXCP_SIZE     , 7

.set SQ_WAVE_TRAP_CTRL_MATH_EXCP_MASK          , ((1 << 7) - 1)
.set SQ_WAVE_TRAP_CTRL_ADDR_WATCH_SHIFT        , 7
.set SQ_WAVE_TRAP_CTRL_WAVE_END_SHIFT          , 8
.set SQ_WAVE_TRAP_CTRL_TRAP_AFTER_INST         , 9

.set SQ_WAVE_PC_HI_ADDRESS_MASK                , 0xFFFF
.set SQ_WAVE_PC_HI_TRAP_ID_BFE                 , (SQ_WAVE_PC_HI_TRAP_ID_SHIFT | (SQ_WAVE_PC_HI_TRAP_ID_SIZE << 16))
.set SQ_WAVE_PC_HI_TRAP_ID_SHIFT               , 28
.set SQ_WAVE_PC_HI_TRAP_ID_SIZE                , 4
.set SQ_WAVE_STATE_PRIV_HALT_BFE               , (SQ_WAVE_STATE_PRIV_HALT_SHIFT | (1 << 16))
.set SQ_WAVE_STATE_PRIV_HALT_SHIFT             , 14
.set SQ_WAVE_STATE_PRIV_BARRIER_COMPLETE_SHIFT , 2

.set TRAP_ID_ABORT                             , 2
.set TRAP_ID_DEBUGTRAP                         , 3
.set TTMP6_SAVED_STATUS_HALT_MASK              , (1 << TTMP6_SAVED_STATUS_HALT_SHIFT)
.set TTMP6_SAVED_STATUS_HALT_SHIFT             , 29
.set TTMP6_SAVED_TRAP_ID_BFE                   , (TTMP6_SAVED_TRAP_ID_SHIFT | (TTMP6_SAVED_TRAP_ID_SIZE << 16))
.set TTMP6_SAVED_TRAP_ID_MASK                  , (((1 << TTMP6_SAVED_TRAP_ID_SIZE) - 1) << TTMP6_SAVED_TRAP_ID_SHIFT)
.set TTMP6_SAVED_TRAP_ID_SHIFT                 , 25
.set TTMP6_SAVED_TRAP_ID_SIZE                  , 4
.set TTMP6_WAVE_STOPPED_SHIFT                  , 30
.set TTMP8_DEBUG_FLAG_SHIFT                    , 31
.set TTMP11_DEBUG_ENABLED_SHIFT                , 23
.set TTMP_PC_HI_SHIFT                          , 7

.set TTMP13_HT_FLAG_BIT                        , 22           // TTMP13 bit for host‑trap
.set TTMP13_STOCH_FLAG_BIT                     , 21           // TTMP13 bit for stochastic
.set TTMP13_BUF_FULL_BIT                       , 31           // TTMP13 bit – buf full mark
.set TTMP8_DISPATCH_ID_MASK                    , 0X1FFFFFF
// Per-sample data layout within the device buffer. Each sample is 64 bytes.
// These are offsets from the start of a specific sample slot in the device buffer.

.set SAMPLE_OFF_BYTES_PER_SAMPLE               , 0x40         // bytes per sample slot

.set SAMPLE_OFF_PC_HOST                        , 0x00         // original PC (host only)
.set SAMPLE_OFF_EXEC_LOHI                      , 0x08         // saved EXEC low/high
.set SAMPLE_OFF_WGID_XY                        , 0x10         // WG id X / Y
.set SAMPLE_OFF_WGID_Z_WAVE                    , 0x18         // WG id Z
.set SAMPLE_OFF_TIMESTAMP                      , 0x30         // 64 bit realtime counter
.set SAMPLE_OFF_HW_ID                          , 0x20         // HW_ID (values combined from the HW_ID1 + HW_ID2)
.set SAMPLE_OFF_SNAPSHOT_DATA                  , 0x24
.set SAMPLE_OFF_CORRELATION                    , 0x38         // doorbell + dispatch id
.set SAMPLE_OFF_BUF_WRITTEN_VAL                , 0x10         // Offset to buf_written_val0/1 in pcs_sampling_data_t
.set SAMPLE_OFF_BUF_SIZE                       , 0x8          // Offset to buf_size in pcs_sampling_data_t
.set SAMPLE_OFF_DONE_SIG0                      , 0x18         // Offset for done_sig0 (hsa_signal_t handle for buffer 0)
.set SAMPLE_OFF_DONE_SIG1                      , 0x28         // Offset for done_sig1 (hsa_signal_t handle for buffer 1)
.set SAMPLE_OFF_SIGNAL_VALUE                   , 0x8          // Offset within signal structure to value field
.set SAMPLE_OFF_EVENT_MAILBOX0                 , 0x10         // Offset for event mailbox pointer for buffer 0
.set SAMPLE_OFF_EVENT_MAILBOX1                 , 0x20         // Offset for event mailbox pointer for buffer 1

.set WAVE_ID_MASK                              , 0x1f         // Mask to extract Wave ID from TTMP register.
.set BUF_INDEX_MASK                            , 0x7fffffff   // strip bit31 from add_x2
.set SAMPLE_OFF_BUF_WRITTEN_VAL                , 0x10         // Offset to buf_written_val0/1 in pcs_sampling_data_t
.set SAMPLE_INDEX_WIDTH                        , 31           // The sample index is 63 bits; the high part is 31 bits.

.set HW_REG_SHADER_HW_ID1                      , 0xf817
.set HW_REG_SHADER_HW_ID2                      , 0xf818
.set HW_REG_SQ_PERF_SNAPSHOT_PC_LO             , 0xf80b
.set HW_REG_SQ_PERF_SNAPSHOT_PC_HI             , 0xf80c
.set HW_REG_SQ_PERF_SNAPSHOT_DATA1             , 0xf80f
.set HW_REG_SQ_PERF_SNAPSHOT_DATA2             , 0xf810
.set HW_REG_SQ_PERF_SNAPSHOT_DATA              , 0xf81b

  // Macro to store the Correlation ID (Dispatch ID and Doorbell ID) into the current sample slot
  //
  // Assumes the following registers are set before it is called:
  //   v[0:1]:Must contain the 64-bit base address of the target sample slot
  //   ttmp8 :Must contain the dispatch ID in bits [24:0]
  //   exec  :Must be set to 0x1 to ensure operations apply only to lane 0
  //
  // Clobbers the following registers:
  //   v[2:3]:Used for [dispatch_id, doorbell_id]
  //   ttmp6 :Used as scratch register
.macro STORE_CORRELATION_ID
  s_sendmsg_rtn_b32 ttmp6, sendmsg(MSG_RTN_GET_DOORBELL)    // Gets current queue's doorbell ID into ttmp6.
  s_wait_kmcnt      0
  s_and_b32         ttmp6, ttmp6, DOORBELL_ID_MASK          // Mask to get actual doorbell ID.
  v_writelane_b32   v3, ttmp6, 0                            // Store doorbell ID into high part of v[2:3] (via v3).
  s_and_b32         ttmp6, ttmp8, TTMP8_DISPATCH_ID_MASK    // Get dispatch ID from ttmp8 into ttmp6
  v_writelane_b32   v2, ttmp6, 0                            // Store dispatch ID into low part of v[2:3] (via v2)
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_CORRELATION, scope:SCOPE_SYS  // Store {dispatch_id, doorbell_id} into sample slot.
                                                                       // v[0:1] = sample slot base address.
                                                                       // v[2] = dispatch_id, v[3] = doorbell_id.
.endm

  // Macro to store the HW_ID registers into the current sample slot
  //
  // Assumes the following registers are set before it is called:
  //   v[0:1]: Must contain the 64-bit base address of the target sample slot.
  //   exec  : Must be set to 0x1 to ensure operations apply only to lane 0.
  //
  // Clobbers the following registers:
  //   v[2:3]: Used to stage the data for the global store.
  //   ttmp6 : Used as scratch registers.
.macro STORE_HW_ID
  // Current ROCr API determines single dword for HW_ID, while this information is scattered accross two
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

  s_getreg_b32      ttmp6, HW_REG_SHADER_HW_ID1             // Put HW_ID1 in ttmp6
  v_and_b32         v2, ttmp6, 0x1feffcff                   // Mask DP_RATE, SE_ID[2] and SIMD_ID
  v_and_b32         v3, ttmp6, 0x300                        // Put SIMD_ID into ttmp6[8:9]
  v_lshl_or_b32     v2, v3, 6, v2                           // Put SIMD_ID into v2[15:14]
  s_getreg_b32      ttmp6, HW_REG_SHADER_HW_ID2             // Put HW_ID2 in ttmp6
  v_and_b32         v3, ttmp6, 0xf000000                    // v3 = VM_ID in bits 27:24
  v_lshl_or_b32     v2, v3, 4, v2                           // Put VM_ID into v2[31:28]
  v_and_b32         v3, ttmp6, 0x1f0000                     // v3 = WG_ID in bits 20:16
  v_lshl_or_b32     v2, v3, 7, v2                           // Put WG_ID in v2[27:23]
  v_and_b32         v3, ttmp6, 0x100                        // v3 = ME_ID[0] in bit 8
  v_lshl_or_b32     v2, v3, 9, v2                           // Put ME_ID in v2[17]
  v_and_b32         v3, ttmp6, 0x30                         // v3 = PIPE_ID in bits 5:4
  v_lshl_or_b32     v2, v3, 16, v2                          // Put PIPE_ID in v2[21:20]
  v_and_b32         v3, ttmp6, 0xf                          // v3 = QUEUE_ID in bits 3:0
  v_lshl_or_b32     v2, v3, 5, v2                           // Put QUEUE_ID in v2[8:5]
  global_store_b32  v[0:1], v2, off, offset:SAMPLE_OFF_HW_ID, scope:SCOPE_SYS  // store HW_ID
.endm

// ABI (Application Binary Interface) between first and second-level trap handler:
//   ttmp0: PC_LO[31:0] (Program Counter Low)
//   ttmp1: PC_HI[15:0] (Program Counter High, bits 0-15), TrapID[3:0] (in bits 28-31 of original PC_HI)
//   ttmp11: 0[7:0], DebugEnabled[0], 0[15:0], NoScratch[0], 0[5:0]
//   ttmp12: SQ_WAVE_STATE_PRIV (Private wave state register value).
//   ttmp14: TMA[31:0] - TMA_LO (Trap Memory Argument Low - base address for trap handler data, low 32 bits).
//   ttmp15: TTMA[63:32] - TMA_HI (Trap Memory Argument High - base address for trap handler data, high 32 bits).
//   For PC Sampling, this points to pcs_hosttrap_data_ or pcs_stochastic_data_
 trap_entry:

  s_mov_b32         ttmp3, 0

.check_hosttrap:

  // ttmp[14:15] points to TMA.
  // Available: ttmp[2:3], ttmp[4:5], ttmp6, ttmp[10:11]
  s_getreg_b32      ttmp2, hwreg(HW_REG_EXCP_FLAG_PRIV)     // On gfx12, EXCP_FLAG_PRIV.b7
  s_bitcmp1_b32     ttmp2, SQ_WAVE_EXCP_FLAG_PRIV_HT_SHIFT
  s_cbranch_scc0    .check_stochastic

  // It's a Host Trap event.
  s_load_b64        ttmp[14:15], ttmp[14:15], 0x0, scope:SCOPE_CU         // ttmp[14:15]=*host_trap_buffers
  s_bitset1_b32     ttmp13, TTMP13_HT_FLAG_BIT              // set bit 22 in TTMP13

  // Clear the Host Trap flag in the hardware register to acknowledge the event
  s_setreg_imm32_b32 hwreg(HW_REG_EXCP_FLAG_PRIV, SQ_WAVE_EXCP_FLAG_PRIV_HT_SHIFT,1), 0
  s_wait_kmcnt      0                                       // Ensure previous load is complete.
  s_branch          .profile_trap_handlers

.check_stochastic:
  s_getreg_b32      ttmp2, hwreg(HW_REG_EXCP_FLAG_PRIV)     // EXCP_FLAG_PRIV.b10=stochastic_sample_trap
  s_bitcmp1_b32     ttmp2, SQ_WAVE_EXCP_FLAG_PRIV_PERF_SNAPSHOT // Test Performance Snapshot bit.

  s_cbranch_scc0    .check_exceptions                       // If not Stochastic, check for other exceptions.

  s_load_b64           ttmp[14:15], ttmp[14:15], 0x8, scope:SCOPE_CU         // ttmp[14:15]=*stoch_trap_buf
  s_wait_kmcnt      0

  s_bitset1_b32     ttmp13, TTMP13_STOCH_FLAG_BIT           // set bit 21 in TTMP13

  s_setreg_imm32_b32 hwreg(HW_REG_EXCP_FLAG_PRIV, SQ_WAVE_EXCP_FLAG_PRIV_PERF_SNAPSHOT,1), 0 // Clear the perf_snapshot flag
  s_branch          .profile_trap_handlers

  // Check if this is a trap (s_trap instruction) or a hardware exception.
  // Extract TrapID from ttmp1 (which contains PC_HI).
  // Branch if not a trap (an exception instead).
  s_bfe_u32         ttmp2, ttmp1, SQ_WAVE_PC_HI_TRAP_ID_BFE // ttmp2 = TrapID
  s_cbranch_scc0       .check_exceptions			             // If TrapID is 0, it's an exception, so branch.

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

  // Save trap id and halt status in ttmp6.
  s_andn2_b32       ttmp6, ttmp6, (TTMP6_SAVED_TRAP_ID_MASK | TTMP6_SAVED_STATUS_HALT_MASK)
  s_bfe_u32         ttmp2, ttmp1, SQ_WAVE_PC_HI_TRAP_ID_BFE
  s_min_u32         ttmp2, ttmp2, 0xF
  s_lshl_b32        ttmp2, ttmp2, TTMP6_SAVED_TRAP_ID_SHIFT
  s_or_b32          ttmp6, ttmp6, ttmp2
  s_bfe_u32         ttmp2, ttmp12, SQ_WAVE_STATE_PRIV_HALT_BFE
  s_lshl_b32        ttmp2, ttmp2, TTMP6_SAVED_STATUS_HALT_SHIFT
  s_or_b32          ttmp6, ttmp6, ttmp2

  // m0 = interrupt data = (exception_code << DOORBELL_ID_SIZE) | doorbell_id
  s_mov_b32         ttmp2, m0
  s_mov_b32         m0, ttmp3
  s_sendmsg         sendmsg(MSG_INTERRUPT)
  // Wait for the message to go out.
  s_wait_kmcnt      0
  s_mov_b32         m0, ttmp2

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
  // ttmp1:  PC_HI[15:0] - Contains program counter high bits
  // ttmp2:  Contains HW_REG_EXCP_FLAG_PRIV
  // ttmp3:  Initialized to 0, available for use
  // ttmp4:  Available - Can be freely used
  // ttmp5:  Available - Can be freely used
  // ttmp6:  Initially contains flags  - trap ID and halt status - reused after saving
  // ttmp7:  Contains WGID_Y in high 16 bits, WGID_Z in low 16 bits
  // ttmp8:  Contains dispatch ID in bits [24:0] and debug flag
  // ttmp9:  Contains WGID_X
  // ttmp10: Available - Used next to save exec_lo
  // ttmp11: Contains debug flags - Used next to save exec_hi
  // ttmp12: Contains SQ_WAVE_STATE_PRIV
  // ttmp13: Contains flag bits for sampling type - HT_FLAG_BIT or STOCH_FLAG_BIT
  // ttmp[14:15]: Contains HT or ST buffer base address
  //
  // v[0:3] contain user shader data that must be preserved/restored
  // exec: Contains user's execution mask
  s_mov_b64         ttmp[10:11], exec                       // save exec to ttmp[10:11]
  s_mov_b64         exec, 0x1                               // turn on lane 0 only

  v_readlane_b32    ttmp2, v0, 0
  v_readlane_b32    ttmp3, v1, 0                            // Save out lane 0’s first 2 VGPRs

  // At this point, ttmp[4:5], ttmp6 and v[0:1] are free
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

  // At this point, ttmp[4:5] and ttmp6 are free
  // v[0:1] (lane 0) now holds the previous value of buf_write_val.
  // This previous value gives the slot index for the current sample.

  v_readlane_b32    ttmp6, v1, 0x0                          // previous buf_write_val[63:32]
  s_lshr_b32        ttmp6, ttmp6, TTMP13_BUF_FULL_BIT       // ttmp6 = previous_buffer_id (0 or 1, from bit 63 of original uint64_t)
                                                            // This ttmp6 is used to select which buffer's metadata (size, watermark, signal) to use.
                                                            // It's also used to calculate the base address of the sample buffer.
  s_bitset0_b32     ttmp13, TTMP13_BUF_FULL_BIT             // Clear our local buffer full flag for now

  s_cmp_eq_u32      ttmp6, 0                                // store off buf_to_use
  s_cbranch_scc1    .skip_bufbit_set                        // into bit31 of ttmp13
  s_bitset1_b32     ttmp13, TTMP13_BUF_FULL_BIT

.skip_bufbit_set:
  // ttmp[2:3]=v[0:1]-backup, ttmp[4:5]=free, ttmp6=buf_to_use (also in ttmp13.b31)
  // ttmp[10:11]=EXEC backup. ttmp[14:15]=tma
  // v[0:1].lane0=local_entry, v[2:3]=original, EXEC=0x1

  v_bfe_u32         v1, v1, 0, SAMPLE_INDEX_WIDTH           // v[0:1] = new local_entry
                                                            // removes bit 31 from v1, returning v1 & 0x7FFFFFFF.

  v_readlane_b32    ttmp5, v1, 0                            // ttmp5 = high 31 bits of sample index (if index > 2^32-1).
  s_cmp_lg_u32      ttmp5, 0                                // Check if sample index is very large (overflowed 32 bits).

  s_cbranch_scc1    .lost_sample                            // If ttmp5 > 0, index is too large, treat as lost sample.

  s_load_b32           ttmp5, ttmp[14:15], SAMPLE_OFF_BUF_SIZE, scope:SCOPE_CU // ttmp5 = pcs_sampling_data_t.buf_size
  v_readlane_b32    ttmp4, v0, 0                            // ttmp4 = sample_index_for_current_sample (from v0)
  s_wait_kmcnt      0                                       // Wait for buf_size load.

  s_cmp_ge_u32      ttmp4, ttmp5                            // if local_entry >= buf_size
  s_cbranch_scc1    .lost_sample                            // If index >= buf_size, buffer is full, sample is lost.
                                                            // This also sets TTMP13_BUF_FULL_BIT implicitly by branching.

  // Register state before calculating the sample buffer address:
  // ttmp2 = backup of original shader's v0
  // ttmp3 = backup of original shader's v1
  // ttmp4 = sample_index_for_current_sample (from v0)
  // ttmp5 = buf_size
  // ttmp6 = buffer_id (0 or 1)
  // ttmp[10:11] = original shader's [exec_lo, exec_hi]
  // ttmp[14:15] = base_address_of_pcs_sampling_data_t (TMA)
  // ttmp13.b31 = buffer_id (0 or 1, same as ttmp6)
  // v[0:1].lane0 = sample index value from atomic
  // v[2:3] = original user shader's v[2:3] values
  // exec = backup of user shader's v[0:1]
  s_mov_b64         exec, ttmp[2:3]                         // stash into EXEC to free up ttmp

  // Calculate the base address of the correct sample buffer (buffer0 or buffer1).
  // The buffers are located after the pcs_sampling_data_t struct header.
  // Address = (TMA + SAMPLE_OFF_BYTES_PER_SAMPLE) + (buffer_id * buf_size * 64)
  s_mul_i32         ttmp2, ttmp5, ttmp6                     // low 32 bits
  s_mul_hi_u32      ttmp3, ttmp5, ttmp6                     // high 32 bits

  // Multiply by 64 bytes per sample slot (shift left by 6 bits)
  // This converts from units of samples to units of bytes
  s_lshl_b64        ttmp[2:3], ttmp[2:3], 6
  s_add_u32         ttmp2, ttmp2, SAMPLE_OFF_BYTES_PER_SAMPLE
  s_addc_u32        ttmp3, ttmp3, 0
  s_add_u32         ttmp4, ttmp14, ttmp2                    // ttmp4 = TMA_base_lo + total_offset_lo. This is low part of &bufferX
  s_addc_u32        ttmp5, ttmp15, ttmp3                    // ttmp5 = TMA_base_hi + total_offset_hi + carry. This is high part of &bufferX
                                                            // ttmp[4:5] now correctly points to the base of the selected sample buffer array

  s_bitcmp1_b32     ttmp13, TTMP13_HT_FLAG_BIT              // if ttmp13.b22==1, this is hosttrap
  s_cbranch_scc1    .fill_sample_ht
  s_bitcmp1_b32     ttmp13, TTMP13_STOCH_FLAG_BIT
  s_cbranch_scc1    .fill_sample_stoch

  s_mov_b64         ttmp[2:3], exec                         // Restore user v[0:1] backup to ttmp[2:3]
  v_readlane_b32    ttmp4, v2, 0                            // Backup user v[2:3] to ttmp[4:5] for restore.
  v_readlane_b32    ttmp5, v3, 0
  s_branch          .restore_vector_before_exit_trap

.fill_sample_ht:
  // At this point, v[0:1] is local_entry (but v1 is 0)
  // v[2:3] is original user-data
  // ttmp[2:3] is free
  // ttmp[4:5] holds &buffer
  // ttmp6 holds buf_to_use
  // ttmp[10:11] holds original shader’s [exec_lo,exec_hi]
  // [ttmp14:15]=‘tma’, ttmp13.b31 = buf_to_use
  // EXEC holds holds backup of original shader’s v[0:1]

  v_readlane_b32    ttmp6, v0, 0                              // ttmp6=local_entry
  s_mul_i32         ttmp2, ttmp6, SAMPLE_OFF_BYTES_PER_SAMPLE // into buffer for 64B objects
  s_mul_hi_u32      ttmp3, ttmp6, SAMPLE_OFF_BYTES_PER_SAMPLE // ttmp[2:3] now holds the offset
  s_add_u32         ttmp2, ttmp2, ttmp4
  s_addc_u32        ttmp3, ttmp3, ttmp5                     // ttmp[2:3]=&bufferX[local_entry]
  v_readlane_b32    ttmp4, v2, 0x0                          // ttmp[4:5] now holds backup of
  v_readlane_b32    ttmp5, v3, 0x0                          // user-data from v[2:3]
  v_writelane_b32   v0, ttmp2, 0x0
  v_writelane_b32   v1, ttmp3, 0x0                          // v[0:1]=&buffer[local_entry]

  s_sendmsg_rtn_b64 ttmp[2:3], sendmsg(MSG_RTN_GET_REALTIME)
  s_wait_kmcnt      0                                       // Wait for timestamp

  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
  // ttmp[2:3] holds the thing we want to store
  // ttmp[4:5] holds backup of original shaders v[2:3]
  // ttmp6 = free
  // ttmp[10:11] holds original shaders [exec_lo,exec_hi]
  // ttmp[14:15]=tma, ttmp13.b31 = buf_to_use
  // EXEC holds backup of original shaders v[0:1]

  v_writelane_b32   v2, ttmp2, 0                            // bring output data to v[2:3]
  v_writelane_b32   v3, ttmp3, 0

  s_mov_b64         ttmp[2:3], exec                         // vector stores need EXEC set
  s_mov_b64         exec, 1                                 // so ttmp[2:3] holds it for now

  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_TIMESTAMP, scope:SCOPE_SYS // store out timestamp

  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader’s v[0:1]
  // ttmp[4:5] holds backup of original shader’s v[2:3]
  // ttmp6 = free
  // ttmp[10:11] holds original shader’s [exec_lo,exec_hi]
  // ttmp[14:15]=‘tma’, ttmp13.b31 = buf_to_use
  // EXEC is 0x1

  s_and_b32         ttmp1, ttmp1, SQ_WAVE_PC_HI_ADDRESS_MASK // Clear out extra data from PC_HI
  v_writelane_b32   v2, ttmp0, 0
  v_writelane_b32   v3, ttmp1, 0
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_PC_HOST, scope:SCOPE_SYS  // store out PC

  v_writelane_b32   v2, ttmp10, 0
  v_writelane_b32   v3, ttmp11, 0
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_EXEC_LOHI, scope:SCOPE_SYS  // store out original EXEC

  // Store Workgroup ID X and Y at offset SAMPLE_OFF_WGID_XY (0x10).
  // ttmp9 = WGID_X (from first-level handler).
  // ttmp7 contains WGID_Y in high 16 bits.
  v_writelane_b32   v2, ttmp9, 0                            // wg_id_x
  s_bfe_u32         ttmp6, ttmp7, (16<<16)                  // extract bits 15:0, wg_id_y
  v_writelane_b32   v3, ttmp6, 0
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_WGID_XY, scope:SCOPE_SYS  // store wg_id_x and wg_id_y

  // Store Workgroup ID Z and Wave ID at offset SAMPLE_OFF_WGID_Z_WAVE (0x18).
  // ttmp7 contains WGID_Z in low 16 bits.
  // ttmp11 contains Wave ID in low 6 bits (from EXEC_hi).
  s_bfe_u32         ttmp6, ttmp7, (16|16<<16)               // extract bits 31:16, wg_id_z
  v_writelane_b32   v2, ttmp6, 0
  v_writelane_b32   v3, ttmp8, 0x0                          // wave_in_wg is bits 29:25
  v_lshrrev_b32     v3, 25, v3                              // Shift wave_in_wg to 4:0
  v_and_b32         v3, v3, WAVE_ID_MASK                    // put (ttmp8>>25)&0x1f into v3
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_WGID_Z_WAVE, scope:SCOPE_SYS  // store wg_id_z and wave_id

  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader’s v[0:1]
  // ttmp[4:5] holds backup of original shader’s v[2:3]
  // ttmp6 = free
  // ttmp[10:11] holds original shader’s [exec_lo,exec_hi]
  // ttmp[14:15]=‘tma’, ttmp13.b31 = buf_to_use
  // EXEC is 0x1
  // Get HW_ID1 & 2 with S_GETREG_B32 with size=32 (F8 in upper bits), offset=0, and:
  // HW_ID1 = 23 (0x17), HW_ID2 = 24 (0x18)

  STORE_HW_ID

  // The following is still true as we get ready to jump to correlation ID check
  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader’s v[0:1]
  // ttmp[4:5] holds backup of original shader’s v[2:3]
  // ttmp6 = free
  // ttmp[10:11] holds original shader’s [exec_lo,exec_hi]
  // ttmp[14:15=‘tma’, ttmp13.b31 = buf_to_use
  // EXEC is 0x1

  STORE_CORRELATION_ID
  // Ensure all stores have completed before returning and incrementing written_val
  s_wait_storecnt   0

  // Still true after returning back from correlation ID check
  // v[0:1] = &buffer[local_entry], but we no longer need it
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader’s v[0:1]
  // ttmp[4:5] holds backup of original shader’s v[2:3]
  // ttmp6 = free
  // ttmp[10:11] holds original shader’s [exec_lo,exec_hi]
  // ttmp[14:15]=‘tma’, ttmp13.b31 = buf_to_use
  // EXEC is 0x1
  //
  s_branch          .ret_from_fill_sample

.fill_sample_stoch:
  // v0 contains local_entry, v1 is free
  // v[2:3] is original user-data
  // ttmp[2:3] is free
  // ttmp[4:5] holds &buffer
  // ttmp6 holds buf_to_use
  // ttmp[10:11] holds original shader’s [exec_lo,exec_hi]
  // [ttmp14:15]=‘tma’, ttmp13.b31 = buf_to_use
  // EXEC holds holds backup of original shader’s v[0:1]

  v_readlane_b32    ttmp6, v0, 0x0                            // ttmp2=local_entry
  s_mul_i32         ttmp2, ttmp6, SAMPLE_OFF_BYTES_PER_SAMPLE // into buffer for 64B objects
  s_mul_hi_u32      ttmp3, ttmp6, SAMPLE_OFF_BYTES_PER_SAMPLE // ttmp[2:3] now holds the offset
  s_add_u32         ttmp2, ttmp2, ttmp4
  s_addc_u32        ttmp3, ttmp3, ttmp5                       // ttmp[2:3]=&bufferX[local_entry]
  v_readlane_b32    ttmp4, v2, 0x0                            // ttmp[4:5] now holds backup of
  v_readlane_b32    ttmp5, v3, 0x0                            // user-data from v[2:3]
  v_writelane_b32   v0, ttmp2, 0x0
  v_writelane_b32   v1, ttmp3, 0x0                            // v[0:1]=&buffer[local_entry]
  s_sendmsg_rtn_b64 ttmp[2:3], sendmsg(MSG_RTN_GET_REALTIME)
  s_wait_kmcnt      0                                         // Wait for timestamp

  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
  // ttmp[2:3] holds the thing we want to store
  // ttmp[4:5] holds backup of original shader’s v[2:3]
  // ttmp6 = free
  // ttmp[10:11] holds original shader’s [exec_lo,exec_hi]
  // ttmp[14:15]=‘tma’, ttmp13.b31 = buf_to_use
  // EXEC holds backup of original shader’s v[0:1]

  v_writelane_b32   v2, ttmp2, 0                            // bring output data to v[2:3]
  v_writelane_b32   v3, ttmp3, 0
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_TIMESTAMP, scope:SCOPE_SYS  // store out timestamp

  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader’s v[0:1]
  // ttmp[4:5] holds backup of original shader’s v[2:3]
  // ttmp6 = free
  // ttmp[10:11] holds original shader’s [exec_lo,exec_hi]
  // ttmp[14:15]=‘tma’, ttmp13.b31 = buf_to_use
  // EXEC is 0x1
  v_writelane_b32   v2, ttmp10, 0
  v_writelane_b32   v3, ttmp11, 0
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_EXEC_LOHI, scope:SCOPE_SYS  // store out original EXEC
  v_writelane_b32   v2, ttmp9, 0                            // wg_id_x
  s_bfe_u32         ttmp6, ttmp7, (0 | (16 << 16))          // extract bits 15:0, wg_id_y
  v_writelane_b32   v3, ttmp6, 0
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_WGID_XY, scope:SCOPE_SYS  // store wg_id_x and wg_id_y
  s_bfe_u32         ttmp6, ttmp7, (16|16<<16)               // extract bits 31:16, wg_id_z
  v_writelane_b32   v2, ttmp6, 0                            // put wg_id_z in v2
  v_writelane_b32   v3, ttmp8, 0x0                          // wave_in_wg is bits 29:25

  v_lshrrev_b32     v3, 25, v3                              // Shift wave_in_wg to 4:0

  v_and_b32         v3, v3, WAVE_ID_MASK                    // put (ttmp8>>25)&0x1f into v3
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_WGID_Z_WAVE, scope:SCOPE_SYS  // store wg_id_z and wave_id

  STORE_HW_ID

  //Read SNAPSHOT Data
  s_getreg_b32      ttmp6, HW_REG_SQ_PERF_SNAPSHOT_DATA1
  v_writelane_b32   v2, ttmp6, 0x0
  s_getreg_b32      ttmp6, HW_REG_SQ_PERF_SNAPSHOT_DATA2
  v_writelane_b32   v3, ttmp6, 0x0
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_SNAPSHOT_DATA + 4, scope:SCOPE_SYS  // store snapshot DATA1 and DATA2

  s_getreg_b32      ttmp2, HW_REG_SQ_PERF_SNAPSHOT_DATA
  v_writelane_b32   v2, ttmp2, 0
  global_store_b32  v[0:1], v2, off, offset:SAMPLE_OFF_SNAPSHOT_DATA, scope:SCOPE_SYS  // store perf snapshot DATA

  s_getreg_b32      ttmp6, HW_REG_SQ_PERF_SNAPSHOT_PC_LO
  v_writelane_b32   v2, ttmp6, 0x0
  s_getreg_b32      ttmp6, HW_REG_SQ_PERF_SNAPSHOT_PC_HI
  v_writelane_b32   v3, ttmp6, 0x0
  global_store_b64  v[0:1], v[2:3], off, offset:SAMPLE_OFF_PC_HOST, scope:SCOPE_SYS  // store PC_HI:PC_LO

  // The following is still true as we get ready to jump to correlation ID check
  // v[0:1] = &buffer[local_entry]
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader’s v[0:1]
  // ttmp[4:5] holds backup of original shader’s v[2:3]
  // ttmp6 = free
  // ttmp[10:11] holds original shader’s [exec_lo,exec_hi]
  // ttmp[14:15]=tma, ttmp13.b31 tells us buf_to_use
  // EXEC is 0x1

  STORE_CORRELATION_ID
  // Ensure all stores have completed before returning and incrementing written_val
  s_wait_storecnt   0

.ret_from_fill_sample:
  // v[0:1] = free
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader’s v[0:1]
  // ttmp[4:5] holds backup of original shader’s v[2:3]
  // ttmp6 = free
  // ttmp[10:11] holds original shader’s [exec_lo,exec_hi]
  // ttmp[14:15]=‘tma’, ttmp13.b31 tells us buf_to_use
  // EXEC is 0x1

  // Sample data has been written to the device buffer.
  // Now, atomically increment the count of written samples for the current buffer.
  // This is pcs_sampling_data_t.buf_written_val0 or buf_written_val1.
  s_lshr_b32        ttmp6, ttmp13, 31                       // ttmp6 is buf_to_use
  s_mulk_i32        ttmp6, 0x10                             // ttmp6=offset from
                                                            // written_val0 to written_val_X
  s_add_u32         ttmp14, ttmp14, ttmp6                   // now ttmp[14:15] points to base for
  s_addc_u32        ttmp15, ttmp15, 0                       // buf_written_valX atomic operation

  // Atomically increment the chosen buf_written_val.
  // v0 = 0 (value to add - low part), v1 = 1 (value to add - high part, effectively just adding 1 to uint32_t)

  v_mov_b32         v0, 0                                   // want to atomic increment
  v_mov_b32         v1, 1                                   // buf_written_valX
  global_atomic_add_u32 v0, v0, v1, ttmp[14:15], offset:SAMPLE_OFF_BUF_WRITTEN_VAL, scope:SCOPE_SYS th:TH_ATOMIC_RETURN
  s_wait_loadcnt    0

  // v0 = done, v1 = free, v[2:3] = free
  // ttmp[2:3] holds backup of original shader’s v[0:1]
  // ttmp[4:5] holds backup of original shader’s v[2:3]
  // ttmp6 = free
  // ttmp[10:11] holds original shader’s [exec_lo,exec_hi]
  // ttmp[14:15]=buf_written_valX-0x10, EXEC=0x1
  // Check Watermark and Signal Host

  s_mov_b64         exec, ttmp[4:5]                         // stash user’s v[2:3] in EXEC
  s_load_b32        ttmp5, ttmp[14:15], 0x14, scope:SCOPE_CU // load watermark into ttmp5
  v_readlane_b32    ttmp4, v0, 0                            // put done into ttmp4
  s_wait_kmcnt      0                                       // wait for watermark to load
  s_cmp_lg_u32      ttmp4, ttmp5                            // if done != watermark, exit
  s_add_u32         ttmp4, ttmp4, 1                         // ttmp4 is now current_sample_count (count_before_inc + 1)
  s_cmp_lt_u32      ttmp4, ttmp5                            // if (current_sample_count < watermark), don't signal
  s_mov_b64         ttmp[4:5], exec                         // restore user’s v[2:3]
  s_mov_b64         exec, 1
  s_cbranch_scc1    .restore_vector_before_exit_trap

.send_signal:
  // v[0:3] = free, ttmp[2:5] = backups of original v[0:3], ttmp6=free
  // ttmp[10:11] holds original shader’s [exec_lo,exec_hi]
  // ttmp[14:15]=buf_written_valX-0x10, EXEC=old copy of original shader v[2:3]
  // write done-signal and optional interrupt

  // Watermark reached or exceeded. Signal the host.
  // Load the hsa_signal_t handle for the current buffer.
  // done_sig0 is at offset 0x18. done_sig1 is at 0x28.
  // addr = ttmp[14:15] + 0x18 + (buffer_id * 0x10).
  // ttmp0 still holds buffer_id * 0x10.

  s_load_b64           ttmp[14:15], ttmp[14:15], SAMPLE_OFF_DONE_SIG0, scope:SCOPE_CU // load done_sig into ttmp[14:15]
  s_mov_b64         exec, 1
  s_wait_kmcnt      0

  v_mov_b32         v0, 0
  v_mov_b32         v1, 0                                   // value to store into v[0:1]
  v_writelane_b32   v2, ttmp14, 0
  v_writelane_b32   v3, ttmp15, 0                           // Put signal address into v[2:3]
  global_store_b64  v[2:3], v[0:1], off, offset:SAMPLE_OFF_SIGNAL_VALUE, scope:SCOPE_SYS // zero out signal value

  s_load_b32           ttmp6, ttmp[14:15], 0x18, scope:SCOPE_CU           // load event_id into ttmp6
  s_load_b64           ttmp[14:15], ttmp[14:15], SAMPLE_OFF_EVENT_MAILBOX0, scope:SCOPE_CU     // load event mailbox ptr into 14:15
  s_wait_kmcnt      0

  s_cmp_eq_u64      ttmp[14:15], 0                          // null mailbox means no interrupt
  s_cbranch_scc1    .restore_vector_before_exit_trap
  s_cmp_eq_u32      ttmp6, 0                                // event_id zero means no interrupt
  s_cbranch_scc1    .restore_vector_before_exit_trap
  v_writelane_b32   v2, ttmp14, 0
  v_writelane_b32   v3, ttmp15, 0                           // Put mailbox address into v[2:3]

  s_wait_storecnt   0
  v_writelane_b32   v0, ttmp6, 0x0                          // put event_id into v0
  global_store_b32  v[2:3], v0, off, offset:0x0, scope:SCOPE_SYS // Send event ID to the mailbox
  s_wait_storecnt   0
  s_mov_b32         ttmp14, m0                              // save off m0
  v_readlane_b32    ttmp15, v0, 0                           // Put ID into message payload
  s_mov_b32         m0, ttmp15
  s_sendmsg         sendmsg(MSG_INTERRUPT)                  // send interrupt message
  s_wait_kmcnt      0
  s_mov_b32         m0, ttmp14                              // restore m0

  // v[0:1] = free
  // v[2:3] = free
  // ttmp[2:3] holds backup of original shader’s v[0:1]
  // ttmp[4:5] holds backup of original shader’s v[2:3]
  // ttmp6 = free
  // ttmp[10:11] holds original shader’s [exec_lo,exec_hi]
  // ttmp[14:15]=somewhere in tma region, EXEC is junk

.restore_vector_before_exit_trap:
  v_writelane_b32   v2, ttmp4, 0
  v_writelane_b32   v3, ttmp5, 0

.lost_sample:
  // v0 contains local_entry, v1 is free
  // v[2:3] is original user-data
  // ttmp[2:3] [local_entry, buf_size]
  // ttmp[4:5] = free
  // ttmp6=buf_to_use (also in ttmp13.b31)
  // ttmp[10:11] holds original shader’s [exec_lo,exec_hi]
  // ttmp[14:15]=tma
  // EXEC=0x1
  // Restore vector registers before exiting

  s_bitcmp1_b32     ttmp13, TTMP13_STOCH_FLAG_BIT           // Check if stochastic sampling
  s_cbranch_scc0    .lost_sample_restore                    // If not, just restore and exit
  s_getreg_b32      ttmp6, HW_REG_SQ_PERF_SNAPSHOT_PC_HI    // Read PC_HI to release lock

.lost_sample_restore:
  v_writelane_b32   v0, ttmp2, 0                            // restore v[0:1] to user data
  v_writelane_b32   v1, ttmp3, 0
  s_mov_b64         exec, ttmp[10:11]                       // restore exec mask

.exit_trap:
  // Restore SQ_WAVE_STATUS.
  s_and_b64         exec, exec, exec                        // Restore STATUS.EXECZ, not writable by s_setreg_b32
  s_and_b64         vcc, vcc, vcc                           // Restore STATUS.VCCZ, not writable by s_setreg_b32
  s_setreg_b32      hwreg(HW_REG_STATE_PRIV, 0, SQ_WAVE_STATE_PRIV_BARRIER_COMPLETE_SHIFT), ttmp12
  s_lshr_b32        ttmp12, ttmp12, (SQ_WAVE_STATE_PRIV_BARRIER_COMPLETE_SHIFT + 1)
  s_setreg_b32      hwreg(HW_REG_STATE_PRIV, SQ_WAVE_STATE_PRIV_BARRIER_COMPLETE_SHIFT + 1, 32 - SQ_WAVE_STATE_PRIV_BARRIER_COMPLETE_SHIFT - 1), ttmp12

  s_rfe_b64         [ttmp0, ttmp1]

.parked:
  s_trap            0x2
  s_branch          .parked

// Add s_code_end padding so instruction prefetch always has something to read.
.rept (256 - ((. - trap_entry) % 64)) / 4
  s_code_end
.endr
