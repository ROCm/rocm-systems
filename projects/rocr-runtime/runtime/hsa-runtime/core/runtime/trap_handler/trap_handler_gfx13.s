/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
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
.set SQ_WAVE_EXCP_FLAG_PRIV_ILLEGAL_INST_SHIFT     , 6
.set SQ_WAVE_EXCP_FLAG_PRIV_WAVE_START_SHIFT       , 8
.set SQ_WAVE_EXCP_FLAG_PRIV_WAVE_END_SHIFT         , 9
.set SQ_WAVE_EXCP_FLAG_PRIV_TRAP_AFTER_INST_SHIFT  , 11
.set SQ_WAVE_EXCP_FLAG_PRIV_XNACK_ERROR_SHIFT      , 12

.set SQ_WAVE_EXCP_FLAG_USER_MATH_EXCP_SHIFT        , 0
.set SQ_WAVE_EXCP_FLAG_USER_MATH_EXCP_SIZE         , 7
.set SQ_WAVE_EXCP_FLAG_USER_MATH_EXCP_BFE          , (SQ_WAVE_EXCP_FLAG_USER_MATH_EXCP_SHIFT | (SQ_WAVE_EXCP_FLAG_USER_MATH_EXCP_SIZE << 16))
.set SQ_WAVE_EXCP_FLAG_USER_ALU_CONSUME_NAN_SHIFT  , 7

.set SQ_WAVE_TRAP_CTRL_MATH_EXCP_MASK              , ((1 << 7) - 1)
.set SQ_WAVE_TRAP_CTRL_ADDR_WATCH_SHIFT            , 7
.set SQ_WAVE_TRAP_CTRL_TRAP_AFTER_INST             , 9
.set SQ_WAVE_TRAP_CTRL_ALU_CONSUME_NAN_SHIFT       , 10

.set SQ_WAVE_PC_HI_TRAP_ID_BFE                     , (SQ_WAVE_PC_HI_TRAP_ID_SHIFT | (SQ_WAVE_PC_HI_TRAP_ID_SIZE << 16))
.set SQ_WAVE_PC_HI_TRAP_ID_SHIFT                   , 28
.set SQ_WAVE_PC_HI_TRAP_ID_SIZE                    , 4

.set SQ_WAVE_STATE_PRIV_HALT_SHIFT                 , 14

.set TRAP_ID_ABORT                                 , 2
.set TRAP_ID_DEBUGTRAP                             , 3
.set TTMP6_SAVED_STATUS_HALT_MASK                  , (1 << TTMP6_SAVED_STATUS_HALT_SHIFT)
.set TTMP6_SAVED_STATUS_HALT_SHIFT                 , 29
.set TTMP6_WAVE_STOPPED_SHIFT                      , 30

.set TTMP8_DEBUG_FLAG_SHIFT                        , 31

.set TTMP11_DEBUG_ENABLED_SHIFT                    , 23
.set TTMP11_SCC_SHIFT                              , 22
.set TTMP11_SCC_BFE                                , (TTMP11_SCC_SHIFT | (1 << 16))
.set TTMP11_SAVED_TRAP_ID_MASK                     , (((1 << TTMP11_SAVED_TRAP_ID_SIZE) - 1) << TTMP11_SAVED_TRAP_ID_SHIFT)
.set TTMP11_SAVED_TRAP_ID_SHIFT                    , 28
.set TTMP11_SAVED_TRAP_ID_SIZE                     , 4

// ABI (Application Binary Interface) between first and second-level trap handler:
//   ttmp0:  PC_LO[31:0]
//   ttmp1:  TrapId[3:0], 0[11:0], PC_HI[15:0]
//   ttmp11: ?[7:0], DebugEnabled[0], SCC[0], ?[21:0]
//   ttmp14: TMA[31:0]
//   ttmp15: TMA[63:32]
 trap_entry:
  s_mov_b32         ttmp3, 0                                 // Clear ttmp3 as it will contain the exception code

  // Check if this is a trap (s_trap instruction) or a hardware exception.
  // Extract TrapID from ttmp1 (which contains PC_HI).
  // Branch if not a trap (an exception instead).
  s_bfe_u32         ttmp2, ttmp1, SQ_WAVE_PC_HI_TRAP_ID_BFE // ttmp2 = TrapID
  s_cbranch_scc0    .check_exceptions			    // If TrapID is 0, it's an exception, so branch.

  // We have executed an s_trap instruction.
  //
  // +---------------------------------------------+--------------------------+
  // | Trap ID                                     | Action                   |
  // +---------------------------------------------+--------------------------+
  // | TRAP_ID_ABORT                               | raise WAVE_ABORT         |
  // | TRAP_ID_DEBUGTRAP and debugger not attached | incr PC, ignore          |
  // | other traps                                 | incr PC, raise WAVE_TRAP |
  // +---------------------------------------------+--------------------------+
  //
  // Handle the TRAP_ID_ABORT first as for abort, we want to keep the PC at the
  // trap instruction.
  s_cmp_eq_u32      ttmp2, TRAP_ID_ABORT
  s_cbranch_scc0    .not_abort_trap
  s_or_b32          ttmp3, ttmp3, EC_QUEUE_WAVE_ABORT_M0
  s_branch          .check_exceptions

.not_abort_trap:
  // For s_trap with an ID other than TRAP_ID_ABORT, advance the PC past the
  // s_trap instruction.
  s_add_u32         ttmp0, ttmp0, 0x4                       // PC_LO += 4
  s_addc_u32        ttmp1, ttmp1, 0x0                       // PC_HI += carry.

  // If llvm.debugtrap and debugger is not attached.
  s_cmp_eq_u32      ttmp2, TRAP_ID_DEBUGTRAP
  s_cbranch_scc0    .not_debug_trap

  s_bitcmp1_b32     ttmp11, TTMP11_DEBUG_ENABLED_SHIFT
  s_cbranch_scc0    .check_exceptions
  s_or_b32          ttmp3, ttmp3, EC_QUEUE_WAVE_TRAP_M0

.not_debug_trap:
  s_or_b32          ttmp3, ttmp3, EC_QUEUE_WAVE_TRAP_M0

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
  s_cbranch_scc0    .not_wave_start
  s_or_b32          ttmp3, ttmp3, EC_QUEUE_WAVE_TRAP_M0

.not_wave_start:
  s_bitcmp1_b32     ttmp2, SQ_WAVE_EXCP_FLAG_PRIV_WAVE_END_SHIFT
  s_cbranch_scc0    .not_wave_end
  s_or_b32          ttmp3, ttmp3, EC_QUEUE_WAVE_TRAP_M0

.not_wave_end:
  s_bitcmp1_b32     ttmp13, SQ_WAVE_TRAP_CTRL_TRAP_AFTER_INST
  s_cbranch_scc0    .not_trap_after_inst
  // TODO: check if the trap_after_inst handling was fixed, and if so ignore unless
  //       trap_ctrl.trap_after_inst is also set.
  s_or_b32          ttmp3, ttmp3, EC_QUEUE_WAVE_TRAP_M0

.not_trap_after_inst:
  s_and_b32         ttmp2, ttmp2, SQ_WAVE_EXCP_FLAG_PRIV_ADDR_WATCH_MASK
  s_cbranch_scc0    .not_addr_watch
  s_bitcmp1_b32     ttmp13, SQ_WAVE_TRAP_CTRL_ADDR_WATCH_SHIFT
  s_cbranch_scc0    .not_addr_watch
  s_or_b32          ttmp3, ttmp3, EC_QUEUE_WAVE_TRAP_M0

.not_addr_watch:
  s_getreg_b32      ttmp2, hwreg(HW_REG_EXCP_FLAG_USER)
  s_bitcmp1_b32     ttmp2, SQ_WAVE_EXCP_FLAG_USER_ALU_CONSUME_NAN_SHIFT
  s_cbranch_scc0    .not_alu_consume_nan
  s_bitcmp1_b32     ttmp13, SQ_WAVE_TRAP_CTRL_ALU_CONSUME_NAN_SHIFT
  s_cbranch_scc0    .not_alu_consume_nan
  s_or_b32          ttmp3, ttmp3, EC_QUEUE_WAVE_TRAP_M0

.not_alu_consume_nan:
  s_bfe_u32         ttmp2, ttmp2, SQ_WAVE_EXCP_FLAG_USER_MATH_EXCP_BFE
  s_and_b32         ttmp13, ttmp13, SQ_WAVE_TRAP_CTRL_MATH_EXCP_MASK
  s_and_b32         ttmp2, ttmp2, ttmp13
  s_cbranch_scc0    .not_math_exception
  s_or_b32          ttmp3, ttmp3, EC_QUEUE_WAVE_MATH_ERROR_M0

.not_math_exception:
  // If we did not find any exception we are interested in, return to user
  // code.
  s_cmp_eq_u32      ttmp3, 0
  s_cbranch_scc1    .exit_trap

.send_interrupt:
  // Fetch doorbell id for our queue.
  s_sendmsg_rtn_b32 ttmp2, sendmsg(MSG_RTN_GET_DOORBELL)
  s_wait_kmcnt      0
  s_and_b32         ttmp2, ttmp2, DOORBELL_ID_MASK
  s_or_b32          ttmp3, ttmp2, ttmp3

  // Save halt status in ttmp6
  s_andn2_b32       ttmp6, ttmp6, TTMP6_SAVED_STATUS_HALT_MASK
  s_getreg_b32      ttmp2, hwreg(HW_REG_WAVE_STATE_PRIV, SQ_WAVE_STATE_PRIV_HALT_SHIFT, 1)
  s_lshl_b32        ttmp2, ttmp2, TTMP6_SAVED_STATUS_HALT_SHIFT
  s_or_b32          ttmp6, ttmp6, ttmp2

  // Save the trap id in ttmp11
  s_andn2_b32       ttmp11, ttmp11, TTMP11_SAVED_TRAP_ID_MASK
  s_bfe_u32         ttmp2, ttmp1, SQ_WAVE_PC_HI_TRAP_ID_BFE
  s_lshl_b32        ttmp2, ttmp2, TTMP11_SAVED_TRAP_ID_SHIFT
  s_or_b32          ttmp11, ttmp11, ttmp2

  // m0 = interrupt data = (exception_code << DOORBELL_ID_SIZE) | doorbell_id
  s_mov_b32         ttmp2, m0
  s_mov_b32         m0, ttmp3
  s_sendmsg         sendmsg(MSG_INTERRUPT)
  // Wait for the message to go out.
  s_wait_kmcnt      0
  s_mov_b32         m0, ttmp2

.halt_wave:
  s_bitset1_b32     ttmp6, TTMP6_WAVE_STOPPED_SHIFT
  s_setreg_imm32_b32 hwreg(HW_REG_STATE_PRIV, SQ_WAVE_STATE_PRIV_HALT_SHIFT, 1), 1

  // Initialize TTMP registers
  s_bitcmp1_b32     ttmp8, TTMP8_DEBUG_FLAG_SHIFT
  s_cbranch_scc1    .ttmps_initialized
  s_mov_b32         ttmp4, 0
  s_mov_b32         ttmp5, 0
  s_bitset1_b32     ttmp8, TTMP8_DEBUG_FLAG_SHIFT
.ttmps_initialized:

.exit_trap:
  // Restore SCC (s_bfe sets scc = (extracted_value != 0)
  s_bfe_u32         ttmp2, ttmp11, TTMP11_SCC_BFE

  // And return to user code.
  s_rfe_b64         [ttmp0, ttmp1]

// Add s_code_end padding so instruction prefetch always has something to read.
.rept (256 - ((. - trap_entry) % 64)) / 4
  s_code_end
.endr
