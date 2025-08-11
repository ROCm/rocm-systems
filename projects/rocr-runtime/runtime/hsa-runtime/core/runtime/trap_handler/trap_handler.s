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
.set SQ_WAVE_PC_HI_ADDRESS_MASK              , 0xFFFF
.set SQ_WAVE_PC_HI_HT_SHIFT                  , 24
.set SQ_WAVE_PC_HI_TRAP_ID_SHIFT             , 16
.set SQ_WAVE_PC_HI_TRAP_ID_SIZE              , 8
.set SQ_WAVE_PC_HI_TRAP_ID_BFE               , (SQ_WAVE_PC_HI_TRAP_ID_SHIFT | (SQ_WAVE_PC_HI_TRAP_ID_SIZE << 16))
.set SQ_WAVE_STATUS_HALT_SHIFT               , 13
.set SQ_WAVE_STATUS_TRAP_SKIP_EXPORT_SHIFT   , 18
.set SQ_WAVE_STATUS_HALT_BFE                 , (SQ_WAVE_STATUS_HALT_SHIFT | (1 << 16))
.set SQ_WAVE_TRAPSTS_MEM_VIOL_SHIFT          , 8
.set SQ_WAVE_TRAPSTS_ILLEGAL_INST_SHIFT      , 11
.set SQ_WAVE_TRAPSTS_XNACK_ERROR_SHIFT       , 28
.set SQ_WAVE_TRAPSTS_MATH_EXCP               , 0x7F
.set SQ_WAVE_TRAPSTS_PERF_SNAPSHOT_SHIFT     , 26
.set SQ_WAVE_TRAPSTS_HOST_TRAP_SHIFT         , 22
.set SQ_WAVE_MODE_EXCP_EN_SHIFT              , 12
.set SQ_WAVE_MODE_EXCP_EN_SIZE               , 8
.set TRAP_ID_ABORT                           , 2
.set TRAP_ID_DEBUGTRAP                       , 3
.set DOORBELL_ID_SIZE                        , 10
.set DOORBELL_ID_MASK                        , ((1 << DOORBELL_ID_SIZE) - 1)
.set EC_QUEUE_WAVE_ABORT_M0                  , (1 << (DOORBELL_ID_SIZE + 0))
.set EC_QUEUE_WAVE_TRAP_M0                   , (1 << (DOORBELL_ID_SIZE + 1))
.set EC_QUEUE_WAVE_MATH_ERROR_M0             , (1 << (DOORBELL_ID_SIZE + 2))
.set EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION_M0    , (1 << (DOORBELL_ID_SIZE + 3))
.set EC_QUEUE_WAVE_MEMORY_VIOLATION_M0       , (1 << (DOORBELL_ID_SIZE + 4))
.set EC_QUEUE_WAVE_APERTURE_VIOLATION_M0     , (1 << (DOORBELL_ID_SIZE + 5))

.set TTMP6_SPI_TTMPS_SETUP_DISABLED_SHIFT    , 31
.set TTMP6_WAVE_STOPPED_SHIFT                , 30
.set TTMP6_SAVED_STATUS_HALT_SHIFT           , 29
.set TTMP6_SAVED_STATUS_HALT_MASK            , (1 << TTMP6_SAVED_STATUS_HALT_SHIFT)
.set TTMP6_SAVED_TRAP_ID_SHIFT               , 25
.set TTMP6_SAVED_TRAP_ID_SIZE                , 4
.set TTMP6_SAVED_TRAP_ID_MASK                , (((1 << TTMP6_SAVED_TRAP_ID_SIZE) - 1) << TTMP6_SAVED_TRAP_ID_SHIFT)
.set TTMP6_SAVED_TRAP_ID_BFE                 , (TTMP6_SAVED_TRAP_ID_SHIFT | (TTMP6_SAVED_TRAP_ID_SIZE << 16))

.set TTMP_PC_HI_SHIFT                        , 7
.set TTMP_DEBUG_ENABLED_SHIFT                , 23

.if .amdgcn.gfx_generation_number == 9
  .set TTMP_SAVE_RCNT_FIRST_REPLAY_SHIFT     , 26
  .set SQ_WAVE_IB_STS_FIRST_REPLAY_SHIFT     , 15
  .set SQ_WAVE_IB_STS_RCNT_FIRST_REPLAY_MASK , 0x1F8000
.elseif .amdgcn.gfx_generation_number == 10 && .amdgcn.gfx_generation_minor < 3
  .set TTMP_SAVE_REPLAY_W64H_SHIFT           , 31
  .set TTMP_SAVE_RCNT_FIRST_REPLAY_SHIFT     , 24
  .set SQ_WAVE_IB_STS_REPLAY_W64H_SHIFT      , 25
  .set SQ_WAVE_IB_STS_FIRST_REPLAY_SHIFT     , 15
  .set SQ_WAVE_IB_STS_RCNT_FIRST_REPLAY_MASK , 0x3F8000
  .set SQ_WAVE_IB_STS_REPLAY_W64H_MASK       , 0x2000000
.endif

// Defining TTMP_REG1 and TTMP_REG2 for clarity in comments
// TTMP_REG1 means ttmp6 register if gfx>=942 and means ttmp13 register if gfx<942
// TTMP_REG2 means ttmp11 register if gfx>=942 and means ttmp6 register if gfx<942

.if .amdgcn.gfx_generation_number == 9
  .set TTMP11_TTMPS_SETUP_SHIFT              , 31

.if (.amdgcn.gfx_generation_minor >= 4)
  .set TTMP11_WAVE_IN_WG_MASK                , 0x3F

  // Bit to indicate that this is a stochastic trap
  .set TTMP13_PCS_IS_STOCHASTIC              , 21

  // Bit to indicate that this is a host trap
  .set TTMP13_PCS_IS_HOSTTRAP                , 22

.else

  // Bit to indicate that this is a host trap
  .set TTMP11_PCS_IS_HOSTTRAP                , 22
.endif
.endif

.if (.amdgcn.gfx_generation_number == 9)

 .macro S_LOAD_DWORD_PCS_TTMP_REG1 base, offset
  .if (.amdgcn.gfx_generation_minor >= 4)
     s_load_dword      ttmp6, \base, \offset
  .else
     s_load_dword      ttmp13,\base, \offset
  .endif
 .endm

 .macro S_BITSET0_B32_PCS_TTMP_REG2 bit_index
  .if (.amdgcn.gfx_generation_minor >= 4)
     s_bitset0_b32     ttmp11, \bit_index
  .else
     s_bitset0_b32     ttmp6, \bit_index
  .endif
 .endm

 .macro S_BITSET1_B32_PCS_TTMP_REG2 bit_index
  .if (.amdgcn.gfx_generation_minor >= 4)
     s_bitset1_b32    ttmp11, \bit_index
  .else
     s_bitset1_b32    ttmp6, \bit_index
  .endif
 .endm

 .macro S_CMP_GE_U32_PCS_TTMP_REG1 src0
  .if (.amdgcn.gfx_generation_minor >= 4)
    s_cmp_ge_u32      \src0, ttmp6
  .else
    s_cmp_ge_u32      \src0, ttmp13
  .endif
 .endm

 .macro S_MOV_B32_SRC_PCS_TTMP_REG1 src0
  .if (.amdgcn.gfx_generation_minor >= 4)
    s_mov_b32        ttmp6, \src0
  .else
    s_mov_b32        ttmp13, \src0
  .endif
 .endm

 .macro S_MOV_B32_DST_PCS_TTMP_REG1 dst
  .if (.amdgcn.gfx_generation_minor >= 4)
    s_mov_b32       \dst, ttmp6
  .else
    s_mov_b32       \dst, ttmp13
  .endif
 .endm

 .macro S_LSHR_B32_PCS_TTMP_REG1_REG2 src1
  .if (.amdgcn.gfx_generation_minor >= 4)
    s_lshr_b32       ttmp6, ttmp11, \src1
  .else
    s_lshr_b32       ttmp13, ttmp6, \src1
  .endif
 .endm

 .macro  S_STORE_DWORD_PCS_TTMP_REG1 base, offset
  .if (.amdgcn.gfx_generation_minor >= 4)
    s_store_dword    ttmp6, \base, \offset
  .else
    s_store_dword    ttmp13, \base, \offset
  .endif
 .endm

 .macro S_MULK_I32_PCS_TTMP_REG1 const_val
  .if (.amdgcn.gfx_generation_minor >= 4)
    s_mulk_i32       ttmp6, \const_val
  .else
    s_mulk_i32       ttmp13, \const_val
  .endif
 .endm

 .macro S_ADD_U32_PCS_TTMP_REG1  dst, src0
  .if (.amdgcn.gfx_generation_minor >= 4)
    s_add_u32        \dst, \src0, ttmp6
  .else
    s_add_u32        \dst, \src0, ttmp13
  .endif
 .endm

 .macro S_CMP_LG_U32_PCS_TTMP_REG1 src0
  .if (.amdgcn.gfx_generation_minor >= 4)
    s_cmp_lg_u32     \src0, ttmp6
  .else
    s_cmp_lg_u32     \src0, ttmp13
  .endif
 .endm

.endif

// ABI between first and second level trap handler:
//   ttmp0  = PC[31:0]
//   ttmp8  = WorkgroupIdX
//   ttmp9  = WorkgroupIdY
//   ttmp10 = WorkgroupIdZ
//   ttmp12 = SQ_WAVE_STATUS
//   ttmp14 = TMA[31:0]
//   ttmp15 = TMA[63:32]
// gfx9:
//   ttmp1 = 0[2:0], PCRewind[3:0], HostTrap[0], TrapId[7:0], PC[47:32]
// For all gfx9 (except gfx940, gfx941, gfx942):
//   ttmp6 = 0[6:0], DispatchPktIndx[24:0]
//   ttmp11 = SQ_WAVE_IB_STS[20:15], 0[1:0], DebugEnabled[0], 0[15:0], NoScratch[0], WaveInWg[5:0]
//
// For gfx940/gfx941/gfx942:
//   ttmp11 = 0[0], DispatchPktIndx[24:0], WaveIdInWg[5:0]
//   ttmp13:
//       Bits 31:26 : SQ_WAVE_IB_STS[20:15] (1TH)
//            25:24 : 0 on 2TH entry. Used by 1st level TH but also
//                    free to be used in the 2nd level TH
//            23    : Debug Enabled (1TH)
//            22:0  : values are unspecified on 2TH entry. Free.
//
// gfx10:
//   ttmp1 = 0[0], PCRewind[5:0], HostTrap[0], TrapId[7:0], PC[47:32]
//
// gfx10/gfx11:
//   ttmp6 = 0[6:0], DispatchPktIndx[24:0]
//
// gfx1010:
//   ttmp11 = SQ_WAVE_IB_STS[25], SQ_WAVE_IB_STS[21:15], DebugEnabled[0], 0[15:0], NoScratch[0], WaveIdInWG[5:0]
//
// gfx1030/gfx1100:
//   ttmp11 = 0[7:0], DebugEnabled[0], 0[15:0], NoScratch[0], WaveIdInWG[5:0]
//
// ttmp[14:15] points to TMA2; Available: ttmp[2:3], ttmp[4:5]
//
// ttmp7 : gfx9, gfx1010, gfx1030, gfx11 - 31:0 : PC[31:0]  (2TH, DBG);
//       : gfx940 - free;
//       : gfx12 - ttmp7 - 31:16 : workgroup_z[15:0]  (SPI) and 15:0 : workgroup_y[15:0]  (SPI)

trap_entry:
  // Extract trap_id from ttmp2
  s_bfe_u32                             ttmp2, ttmp1, SQ_WAVE_PC_HI_TRAP_ID_BFE
  s_cbranch_scc0                        .not_s_trap                      // If trap_id == 0, it's not an s_trap nor host trap

  // Check if the it was an host trap.
  s_bitcmp1_b32                         ttmp1, SQ_WAVE_PC_HI_HT_SHIFT
  s_cbranch_scc0                        .not_host_trap

.if (.amdgcn.gfx_generation_number == 9) // PC_SAMPLING_GFX9
  // ttmp[14:15] is TMA2; Available: ttmp[2:3], ttmp[4:5], ttmp7, TTMP_REG1
  // Check if this is a host-trap. For now, if so, that means we are sampling
  //
  // TMA2 layout:
  //   [0x00] out_buf_t* host_trap_buffers;
  //   [0x08] out_buf_t* stochastic_trap_buffers;
  //
  // --- Start profile trap handlers GFX9 --- //
  // If the wave entered the trap handler: 
  // If on gfx9:
  // - Check SQ_WAVE_PC_HI_HT_SHIFT bit on TTMP1 register to
  //   identify if it was a host trap.
  // If a host trap is detected:
  // - Mark TTMP13(gfx94x) or TTMP11(gfx9) hosttrap bit
  // - Load host_trap_buffers
  // - Branch to the profile trap handler logic.
  //
  // If on gfx9.4+:
  // - Check TRAPSTS bit 26 (SQ_WAVE_TRAPSTS_PERF_SNAPSHOT_SHIFT) to
  //   identify stochastic traps.
  // If a stochastic trap is detected:
  // - Set bit 21 in TTMP13 to indicate a stochastic trap.
  // - Branch to the profile trap handler logic.

  s_load_dwordx2                        ttmp[2:3], ttmp[14:15], 0 glc   // ttmp[14:15]=*host_trap_buffers
.if .amdgcn.gfx_generation_minor >= 4
  s_setreg_imm32_b32                    hwreg(HW_REG_TRAPSTS, SQ_WAVE_TRAPSTS_HOST_TRAP_SHIFT, 1), 0
  s_bitset0_b32                         ttmp13, TTMP13_PCS_IS_STOCHASTIC
  s_bitset1_b32                         ttmp13, TTMP13_PCS_IS_HOSTTRAP   // set bit 22 in TTMP13
.else
  s_bitset1_b32                         ttmp11, TTMP11_PCS_IS_HOSTTRAP    // Set bit 22 in TTMP11
.endif
  s_waitcnt                             lgkmcnt(0)
  s_mov_b64                             ttmp[14:15], ttmp[2:3]          //now ttmp[14:15] = host_trap_buffers
  s_branch                              .profile_trap_handlers_gfx9     // Off to the profile handlers
.else
  // Ignore host traps.  They should be masked by the driver anyway.
  s_branch .not_s_trap
.endif

.not_host_trap:
  // It's an s_trap; advance the PC
  s_add_u32                             ttmp0, ttmp0, 0x4
  s_addc_u32                            ttmp1, ttmp1, 0x0

  // If llvm.debugtrap and debugger is not attached.
  s_cmp_eq_u32                          ttmp2, TRAP_ID_DEBUGTRAP
  s_cbranch_scc0                        .no_skip_debugtrap
.if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor < 4) || .amdgcn.gfx_generation_number >= 10
  s_bitcmp0_b32                         ttmp11, TTMP_DEBUG_ENABLED_SHIFT
.else
  s_bitcmp0_b32                         ttmp13, TTMP_DEBUG_ENABLED_SHIFT
.endif
  s_cbranch_scc0                        .no_skip_debugtrap

  // Ignore llvm.debugtrap.
  s_branch                              .exit_trap

.not_s_trap:
.if .amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor >= 4
  //Check for stochastic trap on gfx9.4+
  s_getreg_b32                          ttmp7, hwreg(HW_REG_TRAPSTS)             // On gfx94x, TRAPSTS bit 26 ...
  s_bitcmp1_b32                         ttmp7, SQ_WAVE_TRAPSTS_PERF_SNAPSHOT_SHIFT   // is stochastic_sample_trap
  s_cbranch_scc0                        .no_skip_debugtrap

  // Handle stochastic trap
  s_setreg_imm32_b32                    hwreg(HW_REG_TRAPSTS, SQ_WAVE_TRAPSTS_PERF_SNAPSHOT_SHIFT, 1), 0
  s_load_dwordx2                        ttmp[2:3], ttmp[14:15], 0x8 glc // ttmp[14:15]=*stoch_trap_buf
  s_bitset0_b32                         ttmp13, TTMP13_PCS_IS_HOSTTRAP
  s_bitset1_b32                         ttmp13, TTMP13_PCS_IS_STOCHASTIC  // set bit 25 in TTMP13
  s_waitcnt                             lgkmcnt(0)
  s_mov_b64                             ttmp[14:15], ttmp[2:3]
  s_branch                              .profile_trap_handlers_gfx9      // Off to the profile handlers
.else
  s_branch                              .no_skip_debugtrap
.endif // PC_SAMPLING_GFX9

.if (.amdgcn.gfx_generation_number == 9) // PC_SAMPLING_GFX9
  // tma->host_trap_buffers Offsets:
  //    [0x00]  uint64_t buf_write_val;
  //    [0x08]  uint32_t buf_size;
  //    [0x0c]  uint32_t reserved0;
  //    [0x10]  uint32_t buf_written_val0;
  //    [0x14]  uint32_t buf_watermark0;
  //    [0x18]  hsa_signal_t done_sig0;
  //    [0x20]  uint32_t buf_written_val1;
  //    [0x24]  uint32_t buf_watermark1;
  //    [0x28]  hsa_signal_t done_sig1;
  //    [0x30]  uint8_t  reserved1[16];
  //    [0x40]  sample_t buffer0[buf_size];
  //    [0x40+(buf_size*sizeof(sample_t))]sample_t buffer1[buf_size];
  //
  //__global__ void profiling_trap_handler(out_buf_t* tma) {
  //  uint64_t local_entry = atomicAdd(&tma->buf_write_val, 1);
  //  int buf_to_use = local_entry >> 63;
  //  local_entry &= (ULLONG_MAX >> 1);
  //
  //  if (local_entry < tma->buf_size) {
  //    sample_t *buf_base = buf_to_use ? tma->buffer1 : tma->buffer0;
  //    fill_sample(&buf_base[local_entry]); // reads TTMP11 as well
  //
  //    uint32_t * written = buf_to_use ? &(tma->buf_written_val1) :
  //                                      &(tma->buf_written_val0);
  //
  //    uint64_t done = __atomic_fetch_add(&written, 1,
  //                memory_order_release, memory_scope_system);
  //
  //    uint32_t watermark = buf_to_use ? tma->buf_watermark0 :
  //                                      tma->buf_watermark1;
  //    if (done == watermark) {
  //       hsa_signal_t done_sig = buf_to_use ? tma->done_sig1 :
  //                                            tma->done_sig0;
  //       send_signal(done_sig);
  //    }
  //  }
  //}

  // ttmp[14:15] is tma->host_trap_buffers; Available: ttmp[2:3], ttmp[4:5], ttmp7, ttmp13
.profile_trap_handlers_gfx9:
  s_mov_b64                             ttmp[2:3], 1                    // atomic increment buf_write_val
  s_atomic_add_x2                       ttmp[2:3], ttmp[14:15], glc     // ttmp[2:3] = packed local_entry
  S_LOAD_DWORD_PCS_TTMP_REG1            ttmp[14:15], 0x8                // TTMP_REG1 = tma->buf_size
  s_waitcnt                             lgkmcnt(0)
  s_lshr_b32                            ttmp7, ttmp3, 31                // ttmp7 = buf_to_use
  S_BITSET0_B32_PCS_TTMP_REG2           31                              // clear out TTMP_REG2  bit31
  s_cmp_eq_u32                          ttmp7, 0                        // store off buf_to_use ...
  s_cbranch_scc1                        .skip_ttmp_set_gfx9             // into bit31 of TTMP_REG2
  S_BITSET1_B32_PCS_TTMP_REG2           31
.skip_ttmp_set_gfx9:
  s_bfe_u64                             ttmp[2:3], ttmp[2:3], (63<<16)  // ttmp[2:3] = new local_entry
  s_cmp_lg_u32                          ttmp3, 0                        // if entry >= 2^32, always lost
  s_cbranch_scc1                        .pc_sampling_exit
  S_CMP_GE_U32_PCS_TTMP_REG1            ttmp2                           // if local_entry >= buf_size
  s_cbranch_scc1                        .pc_sampling_exit

  // ttmp2=local_entry, ttmp7=buf_to_use (also in bit31 of TTMP_REG2), TTMP_REG1=buf_size
  // ttmp[14:15] is tma->host_trap_buffers. Available: ttmp3, ttmp[4:5]
.if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor == 4)
  s_mul_i32                             ttmp6, ttmp6, ttmp7             // ttmp[4:5]=buf_size if ...
  s_mul_i32                             ttmp4, ttmp6, 0x40              // buf_to_use=1, 0 otherwise
  s_mul_hi_u32                          ttmp5, ttmp6, 0x40
.else
  s_mul_i32                             ttmp13, ttmp13, ttmp7           // ttmp[4:5]=buf_size if ...
  s_mul_i32                             ttmp4, ttmp13, 0x40             // buf_to_use=1, 0 otherwise
  s_mul_hi_u32                          ttmp5, ttmp13, 0x40
.endif

  s_add_u32                             ttmp4, ttmp4, 0x40              // now ttmp[4:5]=offset from ...
  s_addc_u32                            ttmp5, ttmp5, 0                 // tma to start of target buffer;
  s_add_u32                             ttmp4, ttmp14, ttmp4            // ttmp[4:5] now points to ...
  s_addc_u32                            ttmp5, ttmp15, ttmp5            // buffer0 or buffer1
  s_mov_b32                             ttmp7, ttmp2

 .if .amdgcn.gfx_generation_number == 9

 .if .amdgcn.gfx_generation_minor >= 4
  // Check if it's a stochastic trap
  s_bitcmp1_b32                         ttmp13, TTMP13_PCS_IS_STOCHASTIC
  s_cbranch_scc1                        .fill_sample_stochastic
  // Check if it's a host trap
  s_bitcmp1_b32                         ttmp13, TTMP13_PCS_IS_HOSTTRAP
  s_cbranch_scc1                        .fill_sample_hosttrap
.else
 // Check if it's a host trap
  s_bitcmp1_b32                         ttmp11, TTMP11_PCS_IS_HOSTTRAP
  s_cbranch_scc1                        .fill_sample_hosttrap

.endif
.endif
  // If neither bit is set, this is unexpected.
  // This branch is not expected to be taken.
  s_branch                              .no_skip_debugtrap

  // ttmp7 contains local_entry, ttmp[4:5] contains "&bufferX",
  // ttmp[14:15] holds 'tma->host_trap_buffers' pointer
  // ttmp[2:3] and ttmp13 are available for gathering perf sample info
  // ttmp[14:15] is live out

  // fill_sample(...) - begin //
  // typedef struct {
  // [0x00]  uint64_t pc;
  // [0x08]  uint64_t exec_mask;
  // [0x10]  uint32_t workgroup_id_x;
  // [0x14]  uint32_t workgroup_id_y;
  // [0x18]  uint32_t workgroup_id_z;
  // [0x1c]  uint32_t wave_in_wg : 6;
  //         uint32_t chiplet    : 3;    // Currently not used
  //         uint32_t reserved   : 23;
  // [0x20]  uint32_t hw_id;
  // [0x24]  uint32_t reserved0;
  // [0x28]  uint64_t reserved1;
  // [0x30]  uint64_t timestamp;
  // [0x38]  uint64_t correlation_id;
  // } perf_sample_hosttrap_v1_t;
  //
  // __device__ void fill_sample_hosttrap_v1(perf_sample_hosttrap_v1_t* buf) {
  //    buf->pc = ((ttmp1 & 0xffff) << 32) | ttmp0;
  //    buf->exec_mask = EXEC;
  //    buf->workgroup_id_x = ttmp8;
  //    buf->workgroup_id_y = ttmp9;
  //    buf->workgroup_id_z = ttmp10;
  //    buf->chiplet_and_wave_id = ttmp11 & 0x3f;
  //    buf->hw_id = s_getreg_b32(HW_REG_HW_ID);
  //    buf->timestamp = s_memrealtime;
  //    buf->correlation_id = get_correlation_id();
  // }
.fill_sample_hosttrap:
  s_mul_i32                             ttmp2, ttmp7, 0x40              // offset into buffer for 64B objects
  s_mul_hi_u32                          ttmp3, ttmp7, 0x40              // ttmp[2:3] will contain byte ...
  s_add_u32                             ttmp2, ttmp2, ttmp4
  s_addc_u32                            ttmp3, ttmp3, ttmp5             // ttmp[2:3]=&bufferX[local_entry]
  s_memrealtime                         ttmp[4:5]
  s_and_b32                             ttmp1, ttmp1, 0xffff            // clear out extra data from PC_HI
  s_store_dwordx2                       ttmp[0:1], ttmp[2:3]            // store PC
  s_waitcnt                             lgkmcnt(0)                      // wait for timestamp
  S_MOV_B32_SRC_PCS_TTMP_REG1           exec_lo
  S_STORE_DWORD_PCS_TTMP_REG1           ttmp[2:3], 0x8                  // store EXEC_LO
  S_MOV_B32_SRC_PCS_TTMP_REG1           exec_hi
  S_STORE_DWORD_PCS_TTMP_REG1           ttmp[2:3], 0xc                  // store EXEC_HI
  s_store_dwordx2                       ttmp[8:9], ttmp[2:3], 0x10      // store wg_id_x and wg_id_y
  s_store_dword                         ttmp10, ttmp[2:3], 0x18         // store wg_id_z
  s_store_dwordx2                       ttmp[4:5], ttmp[2:3], 0x30      // store timestamp

.if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor >= 4)
  s_getreg_b32                          ttmp4, hwreg(HW_REG_XCC_ID)     //store XCC_ID
  s_lshl_b32                            ttmp4, ttmp4, 8
  s_and_b32                             ttmp5, ttmp11, TTMP11_WAVE_IN_WG_MASK
  s_or_b32                              ttmp4, ttmp4, ttmp5
  s_store_dword                         ttmp4, ttmp[2:3], 0x1c          // store wave_in_wg
.else
  s_and_b32                             ttmp4, ttmp11, 0x3f
  s_store_dword                         ttmp4, ttmp[2:3], 0x1c          // store wave_in_wg
.endif
  s_getreg_b32                          ttmp4, hwreg(HW_REG_HW_ID)
  s_store_dword                         ttmp4, ttmp[2:3], 0x20          // store HW_ID

  s_branch                              .get_correlation_id

.if .amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor >= 4
.fill_sample_stochastic:
  s_mul_i32                             ttmp2, ttmp7, 0x40              // offset into buffer for 64B objects
  s_mul_hi_u32                          ttmp3, ttmp7, 0x40
  s_add_u32                             ttmp2, ttmp2, ttmp4
  s_addc_u32                            ttmp3, ttmp3, ttmp5             // ttmp[2:3]=&buffer[local_entry]
  s_memrealtime                         ttmp[4:5]
  s_waitcnt                             lgkmcnt(0)                      // Wait for timestamp
  s_store_dwordx2                       ttmp[4:5], ttmp[2:3] 0x30       // Store timestamp

  s_getreg_b32                          ttmp4, hwreg(HW_REG_SQ_PERF_SNAPSHOT_PC_LO)
  s_getreg_b32                          ttmp5, hwreg(HW_REG_SQ_PERF_SNAPSHOT_PC_HI)
  s_store_dwordx2                       ttmp[4:5], ttmp[2:3] 0x00       // store snapshot data
  s_getreg_b32                          ttmp5, hwreg(HW_REG_SQ_PERF_SNAPSHOT_DATA1)
  s_getreg_b32                          ttmp4, hwreg(HW_REG_SQ_PERF_SNAPSHOT_DATA)
  s_store_dwordx2                       ttmp[4:5], ttmp[2:3], 0x24            // store snapshot PC

  s_mov_b32                             ttmp6, exec_lo
  s_store_dword                         ttmp6, ttmp[2:3], 0x8           // store EXEC_LO
  s_mov_b32                             ttmp6, exec_hi
  s_store_dword                         ttmp6, ttmp[2:3], 0xc           // store EXEC_HI

  s_store_dwordx2                       ttmp[8:9], ttmp[2:3], 0x10      // store wg_id_x and wg_id_y
  s_store_dword                         ttmp10, ttmp[2:3], 0x18         // store wg_id_z
  s_getreg_b32                          ttmp4, hwreg(HW_REG_XCC_ID)
  s_lshl_b32                            ttmp4, ttmp4, 8
  s_and_b32                             ttmp5, ttmp11, TTMP11_WAVE_IN_WG_MASK
  s_or_b32                              ttmp4, ttmp4, ttmp5
  s_store_dword                         ttmp4, ttmp[2:3], 0x1c          // store chiplet_and_wave_id
  s_getreg_b32                          ttmp4, hwreg(HW_REG_HW_ID)
  s_store_dword                         ttmp4, ttmp[2:3], 0x20          // store HW_ID
  // ttmp[2:3]=&buffer[local_entry]; ttmp[4:5], ttmp[6:7] are free
  // ttmp[14:15]=ptr to ‘tma’ and is live out; ttmp11.b31 is buf_to_use, 0 or 1
  s_branch                              .get_correlation_id

.endif

.get_correlation_id:

  // get_correlation_id() -- begin //
  // Returns a value to use as a correlation ID.
  // Returns a 64bit number made up of the 9-bit queue ID and the
  // 25-bit dispatch_pkt concatenated together as:
  // Upper 32 bits: {23 0s}{9b queue_id}
  // Lower 32 bits: { 7 0s}{25b dispatch_pkt}
  // __device__ uint64_t get_correlation_id() {
  //   uint64_t output;
  //   // Get bottom 10 bits of queue's doorbell, in doorbell region.
  //   // Doorbell is 8B (3b per); region is 8K (13b total) so 10 bits.
  //   output = s_sendmsg(MSG_GET_DOORBELL);
  //   output &= 0x3ff;
  //   output <<= 32;
  //   // TTMP6 contains this packet dispatch ID modulus the queue size
  //   output |= TTMP6;
  //   return output;
  // }

  // ttmp[2:3] = &buffer[local_entry]
  // ttmp[4:5], ttmp7, and ttmp13 are free
  // ttmp[14:15] = tma->host_trap_buffers and is live out
  // ttmp6.b31 is buf_to_use, 0 or 1 and is live out

  s_mov_b64                             ttmp[4:5], exec                 // back up EXEC mask
  s_mov_b32                             exec_lo, 0x80000000             // prepare EXEC for doorbell spin
  s_sendmsg                             sendmsg(MSG_GET_DOORBELL)       // message 10, puts doorbell in EXEC
.wait_for_doorbell:
  s_nop                                 0x7                             // wait a bit for message to return
  s_bitcmp0_b32                         exec_lo, 0x1f                   // returned message  will 0 bit 31
  s_cbranch_scc0                        .wait_for_doorbell              // wait some more if no data yet
  s_mov_b32                             exec_hi, ttmp5                  // do not care about message[63:32]
  s_and_b32                             ttmp5, exec_lo, DOORBELL_ID_MASK // doorbell now in ttmp5
  s_mov_b32                             exec_lo, ttmp4                  // exec mask restored

.if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor >= 4)
  s_bfe_u32                             ttmp4, ttmp11, (6 | 25 << 16)    // extract dispatch ID from ttmp11
.else
  s_and_b32                             ttmp4, ttmp6, 0x1ffffff         // extract low 25 bits from ttmp6 (DispatchPktIndx[24:0])
.endif
  s_store_dwordx2                       ttmp[4:5], ttmp[2:3], 0x38      // ttmp[4:5] is correlation ID. Store correlation_id to sample
  // get_correlation_id() -- end //

  // complete stores before returning
  s_dcache_wb
  s_waitcnt                             lgkmcnt(0)
  // fill_sample(...) - end //

  // ttmp[2:3], ttmp[4:5], ttmp7, and ttmp13 are free
  // ttmp[14:15] = tma->host_trap_buffers; ttmp6.b31 is buf_to_use, 0 or 1
  S_LSHR_B32_PCS_TTMP_REG1_REG2         31                              // TTMP_REG1 is buf_to_use
  S_MULK_I32_PCS_TTMP_REG1              0x10                            // written_val0 to written_val_X
  S_ADD_U32_PCS_TTMP_REG1               ttmp14, ttmp14                  // now ttmp[14:15] points to ...
  s_addc_u32                            ttmp15, ttmp15, 0x0             // buf_written_valX-0x10
  s_mov_b32                             ttmp7, 1                        // atomic increment buf_written_valX
  s_atomic_add                          ttmp7, ttmp[14:15], 0x10 glc    // ttmp7 will contain 'done'
  S_LOAD_DWORD_PCS_TTMP_REG1            ttmp[14:15], 0x14               // TTMP_REG1 will hold watermark
  s_waitcnt                             lgkmcnt(0)
  S_CMP_LG_U32_PCS_TTMP_REG1            ttmp7                          // if 'done' not at watermark, exit
  s_cbranch_scc1                        .pc_sampling_exit

  // ttmp[2:3], [4:5], ttmp7, and ttmp13 are free
  // ttmp[14:15] = buf_written_valX-0x10

  // send_signal(...) - begin //
  //__device__ void send_signal(hsa_signal_t* signal) {
  //
  //   amd_signal_t *sig = (amd_signal_t *)signal->handle;
  //   __atomic_store(&(sig->value), 0, memory_order_relaxed, memory_scope_system);
  //   if (sig->event_mailbox_ptr != NULL && sig->event_id != NULL) {
  //     uint32_t id = sig->event_id;
  //     __atomic_store(sig->event_mailbox_ptr, id,
  //            memory_order_relaxed, memory_scope_system);
  //     __builtin_amdgcn_s_sendmsg(1, id);
  //   }
  //}
  // We jump to the trap handler exit after this, so no live-out registers except
  // those that must survive the trap handler

  s_load_dwordx2                        ttmp[2:3], ttmp[14:15], 0x18    // load done_sig into ttmp[2:3]
  s_waitcnt                             lgkmcnt(0)                      // it's actually an amd_signal_t*
  s_load_dwordx2                        ttmp[4:5], ttmp[2:3], 0x10      // load event mailbox ptr into 4:5
  s_load_dword                          ttmp7, ttmp[2:3], 0x18          // load event_id into ttmp7
  s_mov_b64                             ttmp[14:15], 0
  s_store_dwordx2                       ttmp[14:15], ttmp[2:3], 0x8 glc // zero out signal value
  s_waitcnt                             lgkmcnt(0)                      // wait for value store to complete
  s_cmp_eq_u64                          ttmp[4:5], 0
  s_cbranch_scc1                        .pc_sampling_exit               // null mailbox means no interrupt
  s_cmp_eq_u32                          ttmp7, 0
  s_cbranch_scc1                        .pc_sampling_exit               // event_id zero means no interrupt
  s_store_dword                         ttmp7, ttmp[4:5] glc            // send event ID to the mailbox
  s_waitcnt                             lgkmcnt(0)
  S_MOV_B32_SRC_PCS_TTMP_REG1           m0                              // save off m0
  s_mov_b32                             m0, ttmp7                       // put ID into message payload
  s_nop                                 0x0                             // Manually inserted wait states
  s_sendmsg                             sendmsg(MSG_INTERRUPT)          // send interrupt message
  s_waitcnt                             lgkmcnt(0)                      // wait for message to be sent
  S_MOV_B32_DST_PCS_TTMP_REG1           m0                              // restore m0
  // send_signal(...) - end //
.pc_sampling_exit:
  // We can receive regular exceptions while doing PC-Sampling so we need to make sure we
  // handle these exceptions here
  s_getreg_b32                          ttmp2, hwreg(HW_REG_TRAPSTS)
  s_getreg_b32                          ttmp3, hwreg(HW_REG_MODE, SQ_WAVE_MODE_EXCP_EN_SHIFT, SQ_WAVE_MODE_EXCP_EN_SIZE) // ttmp3[7:0] = MODE.EXCP_EN
  // Set bits corresponding to TRAPSTS.MEM_VIOL, TRAPSTS.ILLEGAL_INST and TRAPSTS.XNACK_ERROR
  s_or_b32                              ttmp3, ttmp3, (1 << SQ_WAVE_TRAPSTS_MEM_VIOL_SHIFT | 1 << SQ_WAVE_TRAPSTS_ILLEGAL_INST_SHIFT | 1 << SQ_WAVE_TRAPSTS_XNACK_ERROR_SHIFT)
  s_and_b32                             ttmp2, ttmp2, ttmp3
  // SCC will be 1 if either a maskable instruction was set, or one of MEM_VIOL, ILL_INST, XNACK_ERROR
  s_cbranch_scc1                        .no_skip_debugtrap              // if any of those are set, handle exceptions

  // Check for maskable exceptions
  s_getreg_b32                          ttmp3, hwreg(HW_REG_MODE, SQ_WAVE_MODE_EXCP_EN_SHIFT, SQ_WAVE_MODE_EXCP_EN_SIZE)
  s_and_b32                             ttmp3, ttmp2, ttmp3
  s_cbranch_scc1                        .no_skip_debugtrap

  // Since we are in PC sampling, it is safe to ignore watch1/2/3 and single step
  // as those should only be enabled by the debugger.
  // We could add them for completeness, i.e. check MODE.DEBUG_EN (bit 11)
  // and "MODE.EXCP_EN.WATCH (bit 19) && (TRAPSTS.EXCP_HI.ADDR_WATCH1 (bit 12) || TRAPSTS.EXCP_HI.ADDR_WATCH2 (bit 13) || TRAPSTS.EXCP_HI.ADDR_WATCH3 (bit 14)).
  s_branch                              .exit_trap

.endif // PC_SAMPLING_GFX9
.no_skip_debugtrap:
  // Save trap id and halt status in ttmp6.
  s_andn2_b32                           ttmp6, ttmp6, (TTMP6_SAVED_TRAP_ID_MASK | TTMP6_SAVED_STATUS_HALT_MASK)
  s_bfe_u32                             ttmp2, ttmp1, SQ_WAVE_PC_HI_TRAP_ID_BFE
  s_min_u32                             ttmp2, ttmp2, 0xF
  s_lshl_b32                            ttmp2, ttmp2, TTMP6_SAVED_TRAP_ID_SHIFT
  s_or_b32                              ttmp6, ttmp6, ttmp2
  s_bfe_u32                             ttmp2, ttmp12, SQ_WAVE_STATUS_HALT_BFE
  s_lshl_b32                            ttmp2, ttmp2, TTMP6_SAVED_STATUS_HALT_SHIFT
  s_or_b32                              ttmp6, ttmp6, ttmp2

  // Fetch doorbell id for our queue.
.if .amdgcn.gfx_generation_number < 11
  s_mov_b32                             ttmp2, exec_lo
  s_mov_b32                             ttmp3, exec_hi
  s_mov_b32                             exec_lo, 0x80000000
  s_sendmsg                             sendmsg(MSG_GET_DOORBELL)
.wait_sendmsg:
  s_nop                                 0x7
  s_bitcmp0_b32                         exec_lo, 0x1F
  s_cbranch_scc0                        .wait_sendmsg
  s_mov_b32                             exec_hi, ttmp3
  // Restore exec_lo, move the doorbell_id into ttmp3
  s_and_b32                             ttmp3, exec_lo, DOORBELL_ID_MASK
  s_mov_b32                             exec_lo, ttmp2
.else
  s_sendmsg_rtn_b32                     ttmp3, sendmsg(MSG_RTN_GET_DOORBELL)
  s_waitcnt                             lgkmcnt(0)
  s_and_b32                             ttmp3, ttmp3, DOORBELL_ID_MASK
.endif

  // Map trap reason to an exception code.
  s_getreg_b32                          ttmp2, hwreg(HW_REG_TRAPSTS)

  s_bitcmp1_b32                         ttmp2, SQ_WAVE_TRAPSTS_XNACK_ERROR_SHIFT
  s_cbranch_scc0                        .not_memory_violation
  s_or_b32                              ttmp3, ttmp3, EC_QUEUE_WAVE_MEMORY_VIOLATION_M0

  // Aperture violation requires XNACK_ERROR == 0.
  s_branch                              .not_aperture_violation

.not_memory_violation:
  s_bitcmp1_b32                         ttmp2, SQ_WAVE_TRAPSTS_MEM_VIOL_SHIFT
  s_cbranch_scc0                        .not_aperture_violation
  s_or_b32                              ttmp3, ttmp3, EC_QUEUE_WAVE_APERTURE_VIOLATION_M0

.not_aperture_violation:
  s_bitcmp1_b32                         ttmp2, SQ_WAVE_TRAPSTS_ILLEGAL_INST_SHIFT
  s_cbranch_scc0                        .not_illegal_instruction
  s_or_b32                              ttmp3, ttmp3, EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION_M0

.not_illegal_instruction:
  s_and_b32                             ttmp2, ttmp2, SQ_WAVE_TRAPSTS_MATH_EXCP
  s_cbranch_scc0                        .not_math_exception
  s_getreg_b32                          ttmp7, hwreg(HW_REG_MODE)
  s_lshl_b32                            ttmp2, ttmp2, SQ_WAVE_MODE_EXCP_EN_SHIFT
  s_and_b32                             ttmp2, ttmp2, ttmp7
  s_cbranch_scc0                        .not_math_exception
  s_or_b32                              ttmp3, ttmp3, EC_QUEUE_WAVE_MATH_ERROR_M0

.not_math_exception:
  s_bfe_u32                             ttmp2, ttmp6, TTMP6_SAVED_TRAP_ID_BFE
  s_cmp_eq_u32                          ttmp2, TRAP_ID_ABORT
  s_cbranch_scc0                        .not_abort_trap
  s_or_b32                              ttmp3, ttmp3, EC_QUEUE_WAVE_ABORT_M0

.not_abort_trap:
  // If no other exception was flagged then report a generic error.
  s_andn2_b32                           ttmp2, ttmp3, DOORBELL_ID_MASK
  s_cbranch_scc1                        .send_interrupt
  s_or_b32                              ttmp3, ttmp3, EC_QUEUE_WAVE_TRAP_M0

.send_interrupt:
  // m0 = interrupt data = (exception_code << DOORBELL_ID_SIZE) | doorbell_id
  s_mov_b32                             ttmp2, m0
  s_mov_b32                             m0, ttmp3
  s_nop                                 0x0                             // Manually inserted wait states
  s_sendmsg                             sendmsg(MSG_INTERRUPT)
  s_waitcnt                             lgkmcnt(0)                      // Wait for the message to go out.
  s_mov_b32                             m0, ttmp2

  // Parking the wave requires saving the original pc in the preserved ttmps.
  // Register layout before parking the wave:
  //
  // ttmp7: 0[31:0]
  // ttmp11: 1st_level_ttmp11[31:23] 0[15:0] 1st_level_ttmp11[6:0]
  //
  // After parking the wave:
  //
  // ttmp7:  pc_lo[31:0]
  // ttmp11: 1st_level_ttmp11[31:23] pc_hi[15:0] 1st_level_ttmp11[6:0]
.if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor < 4) || (.amdgcn.gfx_generation_number == 10 && .amdgcn.gfx_generation_minor < 3) || (.amdgcn.gfx_generation_number == 11)
  // Save the PC
  s_mov_b32                             ttmp7, ttmp0
  s_and_b32                             ttmp1, ttmp1, SQ_WAVE_PC_HI_ADDRESS_MASK
  s_lshl_b32                            ttmp1, ttmp1, TTMP_PC_HI_SHIFT
  s_andn2_b32                           ttmp11, ttmp11, (SQ_WAVE_PC_HI_ADDRESS_MASK << TTMP_PC_HI_SHIFT)
  s_or_b32                              ttmp11, ttmp11, ttmp1

  // Park the wave
  s_getpc_b64                           [ttmp0, ttmp1]
  s_add_u32                             ttmp0, ttmp0, .parked - .
  s_addc_u32                            ttmp1, ttmp1, 0x0
.endif

.halt_wave:
  // Halt the wavefront upon restoring STATUS below.
  s_bitset1_b32                         ttmp6, TTMP6_WAVE_STOPPED_SHIFT
  s_bitset1_b32                         ttmp12, SQ_WAVE_STATUS_HALT_SHIFT
  // Set WAVE.SKIP_EXPORT as a maker so the debugger knows the trap handler was
  // entered and has decided to halt the wavee.
  s_bitset1_b32                         ttmp12, SQ_WAVE_STATUS_TRAP_SKIP_EXPORT_SHIFT

.if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor >= 4)
  s_bitcmp1_b32                         ttmp11, TTMP11_TTMPS_SETUP_SHIFT
  s_cbranch_scc1                        .ttmps_initialized
  s_mov_b32                             ttmp4, 0
  s_mov_b32                             ttmp5, 0
  s_bitset0_b32                         ttmp6, TTMP6_SPI_TTMPS_SETUP_DISABLED_SHIFT
  s_bitset1_b32                         ttmp11, TTMP11_TTMPS_SETUP_SHIFT
.ttmps_initialized:
.endif

.exit_trap:
  // Restore SQ_WAVE_IB_STS.
.if .amdgcn.gfx_generation_number == 9
.if .amdgcn.gfx_generation_minor < 4
  s_lshr_b32                            ttmp2, ttmp11, (TTMP_SAVE_RCNT_FIRST_REPLAY_SHIFT - SQ_WAVE_IB_STS_FIRST_REPLAY_SHIFT)
.else
  s_lshr_b32                            ttmp2, ttmp13, (TTMP_SAVE_RCNT_FIRST_REPLAY_SHIFT - SQ_WAVE_IB_STS_FIRST_REPLAY_SHIFT)
.endif
  s_and_b32                             ttmp2, ttmp2, SQ_WAVE_IB_STS_RCNT_FIRST_REPLAY_MASK
  s_setreg_b32                          hwreg(HW_REG_IB_STS), ttmp2
.elseif .amdgcn.gfx_generation_number == 10 && .amdgcn.gfx_generation_minor < 3
  s_lshr_b32                            ttmp2, ttmp11, (TTMP_SAVE_RCNT_FIRST_REPLAY_SHIFT - SQ_WAVE_IB_STS_FIRST_REPLAY_SHIFT)
  s_and_b32                             ttmp3, ttmp2, SQ_WAVE_IB_STS_RCNT_FIRST_REPLAY_MASK
  s_lshr_b32                            ttmp2, ttmp11, (TTMP_SAVE_REPLAY_W64H_SHIFT - SQ_WAVE_IB_STS_REPLAY_W64H_SHIFT)
  s_and_b32                             ttmp2, ttmp2, SQ_WAVE_IB_STS_REPLAY_W64H_MASK
  s_or_b32                              ttmp2, ttmp2, ttmp3
  s_setreg_b32                          hwreg(HW_REG_IB_STS), ttmp2
.endif

  // Restore SQ_WAVE_STATUS.
  s_and_b64                             exec, exec, exec               // restore STATUS.EXECZ, not writable by s_setreg_b32
  s_and_b64                             vcc, vcc, vcc                  // restore STATUS.VCCZ, not writable by s_setreg_b32
  s_setreg_b32                          hwreg(HW_REG_STATUS), ttmp12

  // Return to original (possibly modified) PC.
  s_rfe_b64                             [ttmp0, ttmp1]

.parked:
  s_trap                                0x2
  s_branch                              .parked

// For gfx11, add padding instructions so we can ensure instruction cache
// prefetch always has something to load.
.if .amdgcn.gfx_generation_number == 11
.rept (256 - ((. - trap_entry) % 64)) / 4
  s_code_end
.endr
.endif
