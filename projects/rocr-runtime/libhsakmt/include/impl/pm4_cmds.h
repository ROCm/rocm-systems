/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>

#define mmCOMPUTE_NUM_THREAD_X              0x2E07
#define mmCOMPUTE_PGM_LO                    0x2E0C
#define mmCOMPUTE_DISPATCH_SCRATCH_BASE_LO  0x2E10
#define mmCOMPUTE_PGM_RSRC1                 0x2E12
#define mmCOMPUTE_PGM_RSRC3                 0x2E28
#define mmCOMPUTE_RESOURCE_LIMITS           0x2E15
#define mmCOMPUTE_USER_DATA_0               0x2E40

#define PM4_TYPE_SHIFT        30
#define PM4_COUNT_SHIFT       16
#define PM4_OPCODE_SHIFT      8
#define PM4_SHADER_TYPE_SHIFT 1

#define PM4_GFX_SHADER     0
#define PM4_COMPUTE_SHADER 1

#define PM4_TYPE3_HDR(_opc_, _count_) \
    (uint32_t)((3)                 << PM4_TYPE_SHIFT | \
               ((_count_) - 2)     << PM4_COUNT_SHIFT | \
               (_opc_)             << PM4_OPCODE_SHIFT) | \
               (PM4_COMPUTE_SHADER << PM4_SHADER_TYPE_SHIFT)

union PM4_MEC_TYPE_3_HEADER {
    struct {
        uint32_t reserved1 : 8; ///< reserved
        uint32_t opcode    : 8; ///< IT opcode
        uint32_t count     : 14;///< number of DWORDs - 1 in the information body.
        uint32_t type      : 2; ///< packet identifier. It should be 3 for type 3 packets
    };
    uint32_t u32All;
};

#define IT_DISPATCH_DIRECT 0x15
#define IT_ATOMIC_MEM      0x1E
#define IT_WRITE_DATA      0x37
#define IT_INDIRECT_BUFFER  0x3F
#define IT_COPY_DATA       0x40
#define IT_EVENT_WRITE     0x46
#define IT_RELEASE_MEM     0x49
#define IT_ACQUIRE_MEM     0x58
#define IT_SET_SH_REG      0x76

struct PM4_MEC_SET_SH_REG {
  union {
    PM4_MEC_TYPE_3_HEADER header;
    uint32_t ordinal1;
  };
  union {
    struct {
      uint32_t reg_offset:16;
      uint32_t reserved1:16;
    } bitfields2;
    uint32_t ordinal2;
  };
};

struct PM4_MEC_DISPATCH_DIRECT {
  union {
    PM4_MEC_TYPE_3_HEADER   header;
    uint32_t            ordinal1;
  };
  uint32_t dim_x;
  uint32_t dim_y;
  uint32_t dim_z;
  uint32_t dispatch_initiator;
};

// ------------------------------- MEC_EVENT_WRITE_event_index_enum -------------------------------
enum MEC_EVENT_WRITE_event_index_enum {
  event_index__mec_event_write__other                         =  0,
  event_index__mec_event_write__sample_pipelinestat           =  2,
  event_index__mec_event_write__cs_partial_flush              =  4,
  event_index__mec_event_write__sample_streamoutstats__GFX11  =  8,
  event_index__mec_event_write__sample_streamoutstats1__GFX11 =  9,
  event_index__mec_event_write__sample_streamoutstats2__GFX11 = 10,
  event_index__mec_event_write__sample_streamoutstats3__GFX11 = 11,
};

enum VGT_EVENT_TYPE {
  Reserved_0x00                                      = 0x00000000,
  SAMPLE_STREAMOUTSTATS1                             = 0x00000001,
  SAMPLE_STREAMOUTSTATS2                             = 0x00000002,
  SAMPLE_STREAMOUTSTATS3                             = 0x00000003,
  CACHE_FLUSH_TS                                     = 0x00000004,
  CONTEXT_DONE                                       = 0x00000005,
  CACHE_FLUSH                                        = 0x00000006,
  CS_PARTIAL_FLUSH                                   = 0x00000007,
  VGT_STREAMOUT_SYNC                                 = 0x00000008,
  VGT_STREAMOUT_RESET                                = 0x0000000a,
  END_OF_PIPE_INCR_DE                                = 0x0000000b,
  END_OF_PIPE_IB_END                                 = 0x0000000c,
  RST_PIX_CNT                                        = 0x0000000d,
  BREAK_BATCH                                        = 0x0000000e,
  VS_PARTIAL_FLUSH                                   = 0x0000000f,
  PS_PARTIAL_FLUSH                                   = 0x00000010,
  FLUSH_HS_OUTPUT                                    = 0x00000011,
  FLUSH_DFSM                                         = 0x00000012,
  RESET_TO_LOWEST_VGT                                = 0x00000013,
  CACHE_FLUSH_AND_INV_TS_EVENT                       = 0x00000014,
  CACHE_FLUSH_AND_INV_EVENT                          = 0x00000016,
  PERFCOUNTER_START                                  = 0x00000017,
  PERFCOUNTER_STOP                                   = 0x00000018,
  PIPELINESTAT_START                                 = 0x00000019,
  PIPELINESTAT_STOP                                  = 0x0000001a,
  PERFCOUNTER_SAMPLE                                 = 0x0000001b,
  SAMPLE_PIPELINESTAT                                = 0x0000001e,
  SO_VGTSTREAMOUT_FLUSH                              = 0x0000001f,
  SAMPLE_STREAMOUTSTATS                              = 0x00000020,
  RESET_VTX_CNT                                      = 0x00000021,
  BLOCK_CONTEXT_DONE                                 = 0x00000022,
  CS_CONTEXT_DONE                                    = 0x00000023,
  VGT_FLUSH                                          = 0x00000024,
  TGID_ROLLOVER                                      = 0x00000025,
  SQ_NON_EVENT                                       = 0x00000026,
  SC_SEND_DB_VPZ                                     = 0x00000027,
  BOTTOM_OF_PIPE_TS                                  = 0x00000028,
  FLUSH_SX_TS                                        = 0x00000029,
  DB_CACHE_FLUSH_AND_INV                             = 0x0000002a,
  FLUSH_AND_INV_DB_DATA_TS                           = 0x0000002b,
  FLUSH_AND_INV_DB_META                              = 0x0000002c,
  FLUSH_AND_INV_CB_DATA_TS                           = 0x0000002d,
  FLUSH_AND_INV_CB_META                              = 0x0000002e,
  CS_DONE                                            = 0x0000002f,
  PS_DONE                                            = 0x00000030,
  FLUSH_AND_INV_CB_PIXEL_DATA                        = 0x00000031,
  SX_CB_RAT_ACK_REQUEST                              = 0x00000032,
  THREAD_TRACE_START                                 = 0x00000033,
  THREAD_TRACE_STOP                                  = 0x00000034,
  THREAD_TRACE_MARKER                                = 0x00000035,
  THREAD_TRACE_FINISH                                = 0x00000037,
  PIXEL_PIPE_STAT_CONTROL                            = 0x00000038,
  PIXEL_PIPE_STAT_DUMP                               = 0x00000039,
  PIXEL_PIPE_STAT_RESET                              = 0x0000003a,
  CONTEXT_SUSPEND                                    = 0x0000003b,
  OFFCHIP_HS_DEALLOC                                 = 0x0000003c,
  ENABLE_NGG_PIPELINE                                = 0x0000003d,
  SET_FE_ID__GFX09                                   = 0x00000009,
  Available_0x1c__GFX09                              = 0x0000001c,
  Available_0x1d__GFX09                              = 0x0000001d,
  THREAD_TRACE_FLUSH__GFX09                          = 0x00000036,
  Reserved_0x3f__GFX09                               = 0x0000003f,
  ZPASS_DONE__GFX09_10                               = 0x00000015,
  ENABLE_LEGACY_PIPELINE__GFX09_10                   = 0x0000003e,
  Reserved_0x09__GFX10PLUS                           = 0x00000009,
  FLUSH_ES_OUTPUT__GFX10PLUS                         = 0x0000001c,
  BIN_CONF_OVERRIDE_CHECK__GFX10PLUS                 = 0x0000001d,
  THREAD_TRACE_DRAW__GFX10PLUS                       = 0x00000036,
  DRAW_DONE__GFX10PLUS                               = 0x0000003f,
  WAIT_SYNC__GFX11                                   = 0x00000015,
  ENABLE_PIPELINE_NOT_USED__GFX11                    = 0x0000003e,
};

struct PM4_MEC_EVENT_WRITE {
  union {
    PM4_MEC_TYPE_3_HEADER   header;
    uint32_t            ordinal1;
  };
  union {
    struct {
      uint32_t event_type:6;
      uint32_t reserved1:2;
      uint32_t event_index:4;
      uint32_t reserved2:19;
      uint32_t offload_enable:1;
    } bitfields2;
    uint32_t ordinal2;
  };
};

struct PM4_MEC_ATOMIC_MEM {
  union {
    PM4_MEC_TYPE_3_HEADER   header;
    uint32_t            ordinal1;
  };
  union {
    struct {
      uint32_t atomic:7;
      uint32_t reserved1:1;
      uint32_t command:4;
      uint32_t reserved2:13;
      uint32_t cache_policy:2;
      uint32_t reserved3:5;
    } bitfields2;
    uint32_t ordinal2;
  };
  uint32_t addr_lo;
  uint32_t addr_hi;
  uint32_t src_data_lo;
  uint32_t src_data_hi;
  uint32_t cmp_data_lo;
  uint32_t cmp_data_hi;
  union {
    struct {
      uint32_t loop_interval:13;
      uint32_t reserved4:19;
    } bitfields9;
    uint32_t ordinal9;
  };
};

struct PM4_MEC_WRITE_DATA {
  union {
    PM4_MEC_TYPE_3_HEADER   header;
    uint32_t            ordinal1;
  };
  union {
    struct {
      uint32_t reserved1:8;
      uint32_t dst_sel:4;
      uint32_t reserved2:4;
      uint32_t addr_incr:1;
      uint32_t reserved3:2;
      uint32_t resume_vf:1;
      uint32_t wr_confirm:1;
      uint32_t reserved4:4;
      uint32_t cache_policy:2;
      uint32_t reserved5:5;
    } bitfields2;
    uint32_t ordinal2;
  };
  union {
    struct {
      uint32_t dst_mmreg_addr:18;
      uint32_t reserved6:14;
    } bitfields3a;
    struct {
      uint32_t dst_gds_addr:16;
      uint32_t reserved7:16;
    } bitfields3b;
    struct {
      uint32_t reserved8:2;
      uint32_t dst_mem_addr_lo:30;
    } bitfields3c;
    uint32_t ordinal3;
  };
  uint32_t dst_mem_addr_hi;
  uint64_t write_data_value;
};

#define PERSISTENT_SPACE_START         0x00002c00

template <class T>
void GenerateSetShRegHeader(T* pm4, uint32_t reg_addr) {
  pm4->cmd_set_data.header.u32All = PM4_TYPE3_HDR(IT_SET_SH_REG,
                                                  sizeof(T) / sizeof(uint32_t));
  pm4->cmd_set_data.bitfields2.reg_offset = reg_addr - PERSISTENT_SPACE_START;
}

template <class T>
void GenerateCmdHeader(T* pm4, int op_code) {
  pm4->header.u32All = PM4_TYPE3_HDR(op_code, sizeof(T) / sizeof(uint32_t));
}

/// @brief Defines the Gpu command to dispatch a kernel. It embeds
/// various Gpu hardware specific data structures for initialization
/// and configuration before a dispatch begins to run
struct DispatchTemplate {

  /// @brief Structure used to initialize the group dimensions
  /// of a kernel dispatch and if performance counters are enabled
  struct DispatchDimensionRegs {
    PM4_MEC_SET_SH_REG cmd_set_data;
    uint32_t compute_num_thread_x;
    uint32_t compute_num_thread_y;
    uint32_t compute_num_thread_z;
  } dimension_regs;

  struct DispatchProgramRegs {
    PM4_MEC_SET_SH_REG cmd_set_data;
    uint32_t compute_pgm_lo;
    uint32_t compute_pgm_hi;
  } program_regs;

  struct DispatchProgramResourceRegs {
    PM4_MEC_SET_SH_REG cmd_set_data;
    uint32_t compute_pgm_rsrc1;
    uint32_t compute_pgm_rsrc2;
  } program_resource_regs;

  /// @brief Structure used to initialize parameters related to
  /// thread management i.e. number of waves to issue and number
  /// of Compute Units to use
  struct DispatchResourceRegs {
    PM4_MEC_SET_SH_REG cmd_set_data;
    uint32_t compute_resource_limits;
    uint32_t compute_static_thread_mgmt_se0;
    uint32_t compute_static_thread_mgmt_se1;
    uint32_t compute_tmpring_size;
    uint32_t compute_static_thread_mgmt_se2;
    uint32_t compute_static_thread_mgmt_se3;
  } resource_regs;

  /// @brief Structure used to pass handles of the Aql dispatch
  /// packet, Aql queue, Kernel argument address block, Scratch
  /// buffer
  struct DispatchComputeUserDataRegs {
    PM4_MEC_SET_SH_REG cmd_set_data;
    uint32_t compute_user_data[16];
  } compute_user_data_regs;

  /// @brief Structure used to configure Cache flush policy
  /// and dimensions of total work size
  PM4_MEC_DISPATCH_DIRECT dispatch_direct;
};

struct DispatchProgramResourceRegs {
    PM4_MEC_SET_SH_REG cmd_set_data;
    uint32_t compute_pgm_rsrc3;
};


/// @brief Structure used to issue a programing scratch command for gfx11+
struct SetScratchTemplate {
  PM4_MEC_SET_SH_REG cmd_set_data;
  uint32_t scratch_lo;
  uint32_t scratch_hi;
};

/// @brief Structure used to issue a Gpu Barrier command
struct BarrierTemplate {
  PM4_MEC_EVENT_WRITE event_write;
};

//--------------------MEC_ATOMIC_MEM--------------------
enum MEC_ATOMIC_MEM_command_enum {
  command__mec_atomic_mem__single_pass_atomic           = 0,
  command__mec_atomic_mem__loop_until_compare_satisfied = 1,
  command__mec_atomic_mem__wait_for_write_confirmation  = 2,
  command__mec_atomic_mem__send_and_continue            = 3,
};

enum MEC_ATOMIC_MEM_cache_policy_enum {
  cache_policy__mec_atomic_mem__lru     = 0,
  cache_policy__mec_atomic_mem__stream  = 1,
  cache_policy__mec_atomic_mem__noa     = 2,
  cache_policy__mec_atomic_mem__bypass  = 3,
};

enum TC_OP {
  TC_OP_READ                                         = 0x00000000,
  TC_OP_ATOMIC_FCMPSWAP_RTN_32                       = 0x00000001,
  TC_OP_ATOMIC_FMIN_RTN_32                           = 0x00000002,
  TC_OP_ATOMIC_FMAX_RTN_32                           = 0x00000003,
  TC_OP_RESERVED_FOP_RTN_32_0                        = 0x00000004,
  TC_OP_RESERVED_FOP_RTN_32_2                        = 0x00000006,
  TC_OP_ATOMIC_SWAP_RTN_32                           = 0x00000007,
  TC_OP_ATOMIC_CMPSWAP_RTN_32                        = 0x00000008,
  TC_OP_ATOMIC_FCMPSWAP_FLUSH_DENORM_RTN_32          = 0x00000009,
  TC_OP_ATOMIC_FMIN_FLUSH_DENORM_RTN_32              = 0x0000000a,
  TC_OP_ATOMIC_FMAX_FLUSH_DENORM_RTN_32              = 0x0000000b,
  TC_OP_PROBE_FILTER                                 = 0x0000000c,
  TC_OP_RESERVED_FOP_FLUSH_DENORM_RTN_32_2           = 0x0000000e,
  TC_OP_ATOMIC_ADD_RTN_32                            = 0x0000000f,
  TC_OP_ATOMIC_SUB_RTN_32                            = 0x00000010,
  TC_OP_ATOMIC_SMIN_RTN_32                           = 0x00000011,
  TC_OP_ATOMIC_UMIN_RTN_32                           = 0x00000012,
  TC_OP_ATOMIC_SMAX_RTN_32                           = 0x00000013,
  TC_OP_ATOMIC_UMAX_RTN_32                           = 0x00000014,
  TC_OP_ATOMIC_AND_RTN_32                            = 0x00000015,
  TC_OP_ATOMIC_OR_RTN_32                             = 0x00000016,
  TC_OP_ATOMIC_XOR_RTN_32                            = 0x00000017,
  TC_OP_ATOMIC_INC_RTN_32                            = 0x00000018,
  TC_OP_ATOMIC_DEC_RTN_32                            = 0x00000019,
  TC_OP_WBINVL1_VOL                                  = 0x0000001a,
  TC_OP_WBINVL1_SD                                   = 0x0000001b,
  TC_OP_RESERVED_NON_FLOAT_RTN_32_0                  = 0x0000001c,
  TC_OP_RESERVED_NON_FLOAT_RTN_32_1                  = 0x0000001d,
  TC_OP_RESERVED_NON_FLOAT_RTN_32_2                  = 0x0000001e,
  TC_OP_RESERVED_NON_FLOAT_RTN_32_3                  = 0x0000001f,
  TC_OP_WRITE                                        = 0x00000020,
  TC_OP_ATOMIC_FCMPSWAP_RTN_64                       = 0x00000021,
  TC_OP_ATOMIC_FMIN_RTN_64                           = 0x00000022,
  TC_OP_ATOMIC_FMAX_RTN_64                           = 0x00000023,
  TC_OP_RESERVED_FOP_RTN_64_0                        = 0x00000024,
  TC_OP_RESERVED_FOP_RTN_64_1                        = 0x00000025,
  TC_OP_RESERVED_FOP_RTN_64_2                        = 0x00000026,
  TC_OP_ATOMIC_SWAP_RTN_64                           = 0x00000027,
  TC_OP_ATOMIC_CMPSWAP_RTN_64                        = 0x00000028,
  TC_OP_ATOMIC_FCMPSWAP_FLUSH_DENORM_RTN_64          = 0x00000029,
  TC_OP_ATOMIC_FMIN_FLUSH_DENORM_RTN_64              = 0x0000002a,
  TC_OP_ATOMIC_FMAX_FLUSH_DENORM_RTN_64              = 0x0000002b,
  TC_OP_WBINVL2_SD                                   = 0x0000002c,
  TC_OP_RESERVED_FOP_FLUSH_DENORM_RTN_64_0           = 0x0000002d,
  TC_OP_RESERVED_FOP_FLUSH_DENORM_RTN_64_1           = 0x0000002e,
  TC_OP_ATOMIC_ADD_RTN_64                            = 0x0000002f,
  TC_OP_ATOMIC_SUB_RTN_64                            = 0x00000030,
  TC_OP_ATOMIC_SMIN_RTN_64                           = 0x00000031,
  TC_OP_ATOMIC_UMIN_RTN_64                           = 0x00000032,
  TC_OP_ATOMIC_SMAX_RTN_64                           = 0x00000033,
  TC_OP_ATOMIC_UMAX_RTN_64                           = 0x00000034,
  TC_OP_ATOMIC_AND_RTN_64                            = 0x00000035,
  TC_OP_ATOMIC_OR_RTN_64                             = 0x00000036,
  TC_OP_ATOMIC_XOR_RTN_64                            = 0x00000037,
  TC_OP_ATOMIC_INC_RTN_64                            = 0x00000038,
  TC_OP_ATOMIC_DEC_RTN_64                            = 0x00000039,
  TC_OP_WBL2_NC                                      = 0x0000003a,
  TC_OP_WBL2_WC                                      = 0x0000003b,
  TC_OP_RESERVED_NON_FLOAT_RTN_64_1                  = 0x0000003c,
  TC_OP_RESERVED_NON_FLOAT_RTN_64_2                  = 0x0000003d,
  TC_OP_RESERVED_NON_FLOAT_RTN_64_3                  = 0x0000003e,
  TC_OP_RESERVED_NON_FLOAT_RTN_64_4                  = 0x0000003f,
  TC_OP_WBINVL1                                      = 0x00000040,
  TC_OP_ATOMIC_FCMPSWAP_32                           = 0x00000041,
  TC_OP_ATOMIC_FMIN_32                               = 0x00000042,
  TC_OP_ATOMIC_FMAX_32                               = 0x00000043,
  TC_OP_RESERVED_FOP_32_0                            = 0x00000044,
  TC_OP_RESERVED_FOP_32_2                            = 0x00000046,
  TC_OP_ATOMIC_SWAP_32                               = 0x00000047,
  TC_OP_ATOMIC_CMPSWAP_32                            = 0x00000048,
  TC_OP_ATOMIC_FCMPSWAP_FLUSH_DENORM_32              = 0x00000049,
  TC_OP_ATOMIC_FMIN_FLUSH_DENORM_32                  = 0x0000004a,
  TC_OP_ATOMIC_FMAX_FLUSH_DENORM_32                  = 0x0000004b,
  TC_OP_INV_METADATA                                 = 0x0000004c,
  TC_OP_RESERVED_FOP_FLUSH_DENORM_32_2               = 0x0000004e,
  TC_OP_ATOMIC_ADD_32                                = 0x0000004f,
  TC_OP_ATOMIC_SUB_32                                = 0x00000050,
  TC_OP_ATOMIC_SMIN_32                               = 0x00000051,
  TC_OP_ATOMIC_UMIN_32                               = 0x00000052,
  TC_OP_ATOMIC_SMAX_32                               = 0x00000053,
  TC_OP_ATOMIC_UMAX_32                               = 0x00000054,
  TC_OP_ATOMIC_AND_32                                = 0x00000055,
  TC_OP_ATOMIC_OR_32                                 = 0x00000056,
  TC_OP_ATOMIC_XOR_32                                = 0x00000057,
  TC_OP_ATOMIC_INC_32                                = 0x00000058,
  TC_OP_ATOMIC_DEC_32                                = 0x00000059,
  TC_OP_INVL2_NC                                     = 0x0000005a,
  TC_OP_NOP_RTN0                                     = 0x0000005b,
  TC_OP_RESERVED_NON_FLOAT_32_1                      = 0x0000005c,
  TC_OP_RESERVED_NON_FLOAT_32_2                      = 0x0000005d,
  TC_OP_RESERVED_NON_FLOAT_32_3                      = 0x0000005e,
  TC_OP_RESERVED_NON_FLOAT_32_4                      = 0x0000005f,
  TC_OP_WBINVL2                                      = 0x00000060,
  TC_OP_ATOMIC_FCMPSWAP_64                           = 0x00000061,
  TC_OP_ATOMIC_FMIN_64                               = 0x00000062,
  TC_OP_ATOMIC_FMAX_64                               = 0x00000063,
  TC_OP_RESERVED_FOP_64_0                            = 0x00000064,
  TC_OP_RESERVED_FOP_64_1                            = 0x00000065,
  TC_OP_RESERVED_FOP_64_2                            = 0x00000066,
  TC_OP_ATOMIC_SWAP_64                               = 0x00000067,
  TC_OP_ATOMIC_CMPSWAP_64                            = 0x00000068,
  TC_OP_ATOMIC_FCMPSWAP_FLUSH_DENORM_64              = 0x00000069,
  TC_OP_ATOMIC_FMIN_FLUSH_DENORM_64                  = 0x0000006a,
  TC_OP_ATOMIC_FMAX_FLUSH_DENORM_64                  = 0x0000006b,
  TC_OP_RESERVED_FOP_FLUSH_DENORM_64_0               = 0x0000006c,
  TC_OP_RESERVED_FOP_FLUSH_DENORM_64_1               = 0x0000006d,
  TC_OP_RESERVED_FOP_FLUSH_DENORM_64_2               = 0x0000006e,
  TC_OP_ATOMIC_ADD_64                                = 0x0000006f,
  TC_OP_ATOMIC_SUB_64                                = 0x00000070,
  TC_OP_ATOMIC_SMIN_64                               = 0x00000071,
  TC_OP_ATOMIC_UMIN_64                               = 0x00000072,
  TC_OP_ATOMIC_SMAX_64                               = 0x00000073,
  TC_OP_ATOMIC_UMAX_64                               = 0x00000074,
  TC_OP_ATOMIC_AND_64                                = 0x00000075,
  TC_OP_ATOMIC_OR_64                                 = 0x00000076,
  TC_OP_ATOMIC_XOR_64                                = 0x00000077,
  TC_OP_ATOMIC_INC_64                                = 0x00000078,
  TC_OP_ATOMIC_DEC_64                                = 0x00000079,
  TC_OP_WBINVL2_NC                                   = 0x0000007a,
  TC_OP_NOP_ACK                                      = 0x0000007b,
  TC_OP_RESERVED_NON_FLOAT_64_1                      = 0x0000007c,
  TC_OP_RESERVED_NON_FLOAT_64_2                      = 0x0000007d,
  TC_OP_RESERVED_NON_FLOAT_64_3                      = 0x0000007e,
  TC_OP_RESERVED_NON_FLOAT_64_4                      = 0x0000007f,
  TC_OP_RESERVED_FOP_RTN_32_1__GFX09_10              = 0x00000005,
  TC_OP_RESERVED_FOP_FLUSH_DENORM_RTN_32_1__GFX09_10 = 0x0000000d,
  TC_OP_RESERVED_FOP_32_1__GFX09_10                  = 0x00000045,
  TC_OP_RESERVED_FOP_FLUSH_DENORM_32_1__GFX09_10     = 0x0000004d,
  TC_OP_RESERVED_FADD_RTN_32__GFX11                  = 0x00000005,
  TC_OP_ATOMIC_FADD_FLUSH_DENORM_RTN_32__GFX11       = 0x0000000d,
  TC_OP_RESERVED_FADD_32__GFX11                      = 0x00000045,
  TC_OP_ATOMIC_FADD_FLUSH_DENORM_32__GFX11           = 0x0000004d,
};

// Desc: Strucuture used to perform various atomic
// operations - add, subtract, increment, etc
struct AtomicTemplate {
  PM4_MEC_ATOMIC_MEM atomic;
};

/// @brief PM4 command to write a 64-bit value into a memory
/// location accessible to Gpu
struct WriteDataTemplate {
  PM4_MEC_WRITE_DATA write_data;
};

// ---------------------------------- MEC_COPY_DATA_src_sel_enum ----------------------------------
enum MEC_COPY_DATA_src_sel_enum {
  src_sel__mec_copy_data__mem_mapped_register     =  0,
  src_sel__mec_copy_data__tc_l2_obsolete          =  1,
  src_sel__mec_copy_data__tc_l2                   =  2,
  src_sel__mec_copy_data__gds                     =  3,
  src_sel__mec_copy_data__perfcounters            =  4,
  src_sel__mec_copy_data__immediate_data          =  5,
  src_sel__mec_copy_data__atomic_return_data      =  6,
  src_sel__mec_copy_data__gds_atomic_return_data0 =  7,
  src_sel__mec_copy_data__gds_atomic_return_data1 =  8,
  src_sel__mec_copy_data__gpu_clock_count         =  9,
  src_sel__mec_copy_data__system_clock_count      = 10,
  src_sel__mec_copy_data__ext32perfcntr           = 11,
};

// ---------------------------------- MEC_COPY_DATA_dst_sel_enum ----------------------------------
enum MEC_COPY_DATA_dst_sel_enum {
  dst_sel__mec_copy_data__mem_mapped_register =  0,
  dst_sel__mec_copy_data__tc_l2               =  2,
  dst_sel__mec_copy_data__gds                 =  3,
  dst_sel__mec_copy_data__perfcounters        =  4,
  dst_sel__mec_copy_data__tc_l2_obsolete      =  5,
  dst_sel__mec_copy_data__mem_mapped_reg_dc   =  6,
  dst_sel__mec_copy_data__ext32perfcntr       = 11,
};

// ------------------------------ MEC_COPY_DATA_src_cache_policy_enum ------------------------------
enum MEC_COPY_DATA_src_cache_policy_enum {
  src_cache_policy__mec_copy_data__lru    =  0,
  src_cache_policy__mec_copy_data__stream =  1,
  src_cache_policy__mec_copy_data__noa    =  2,
  src_cache_policy__mec_copy_data__bypass =  3,
};

// --------------------------------- MEC_COPY_DATA_count_sel_enum ---------------------------------
enum MEC_COPY_DATA_count_sel_enum {
  count_sel__mec_copy_data__32_bits_of_data =  0,
  count_sel__mec_copy_data__64_bits_of_data =  1,
};

// --------------------------------- MEC_COPY_DATA_wr_confirm_enum ---------------------------------
enum MEC_COPY_DATA_wr_confirm_enum {
  wr_confirm__mec_copy_data__do_not_wait_for_confirmation =  0,
  wr_confirm__mec_copy_data__wait_for_confirmation        =  1,
};

// ------------------------------ MEC_COPY_DATA_dst_cache_policy_enum ------------------------------
enum MEC_COPY_DATA_dst_cache_policy_enum {
  dst_cache_policy__mec_copy_data__lru    =  0,
  dst_cache_policy__mec_copy_data__stream =  1,
  dst_cache_policy__mec_copy_data__noa    =  2,
  dst_cache_policy__mec_copy_data__bypass =  3,
};

// ------------------------------- MEC_COPY_DATA_pq_exe_status_enum -------------------------------
enum MEC_COPY_DATA_pq_exe_status_enum {
  pq_exe_status__mec_copy_data__default      =  0,
  pq_exe_status__mec_copy_data__phase_update =  1,
};

// ------------------------------- MEC_WRITE_DATA_dst_sel_enum -------------------------------
enum MEC_WRITE_DATA_dst_sel_enum {
     dst_sel__mec_write_data__mem_mapped_register = 0,
     dst_sel__mec_write_data__tc_l2 = 2,
     dst_sel__mec_write_data__gds = 3,
     dst_sel__mec_write_data__memory = 5,
     dst_sel__mec_write_data__memory_mapped_adc_persistent_state = 6 };

// ------------------------------- MEC_WRITE_DATA_addr_incr_enum -------------------------------
enum MEC_WRITE_DATA_addr_incr_enum {
     addr_incr__mec_write_data__increment_address = 0,
     addr_incr__mec_write_data__do_not_increment_address = 1 };

// ------------------------------- MEC_WRITE_DATA_wr_confirm_enum -------------------------------
enum MEC_WRITE_DATA_wr_confirm_enum {
     wr_confirm__mec_write_data__do_not_wait_for_write_confirmation = 0,
     wr_confirm__mec_write_data__wait_for_write_confirmation = 1 };

// ------------------------------- MEC_WRITE_DATA_cache_policy_enum -------------------------------
enum MEC_WRITE_DATA_cache_policy_enum {
     cache_policy__mec_write_data__lru = 0,
     cache_policy__mec_write_data__stream = 1,
     cache_policy__mec_write_data__noa    =  2,
     cache_policy__mec_write_data__bypass =  3 };

typedef struct PM4_MEC_COPY_DATA {
  union {
    PM4_MEC_TYPE_3_HEADER header;  /// header
    uint32_t ordinal1;
  };
  union {
    struct {
      uint32_t src_sel : 4;
      uint32_t reserved1 : 4;
      uint32_t dst_sel : 4;
      uint32_t reserved2 : 1;
      uint32_t src_cache_policy : 2;
      uint32_t reserved3 : 1;
      uint32_t count_sel : 1;
      uint32_t reserved4 : 3;
      uint32_t wr_confirm : 1;
      uint32_t reserved5 : 4;
      uint32_t dst_cache_policy : 2;
      uint32_t reserved6 : 2;
      uint32_t pq_exe_status : 1;
      uint32_t reserved7 : 2;
    } bitfields2;
    uint32_t ordinal2;
  };
  union {
    struct {
      uint32_t src_reg_offset : 18;
      uint32_t reserved8 : 14;
    } bitfields3a;
    struct {
      uint32_t reserved9 : 2;
      uint32_t src_32b_addr_lo : 30;
    } bitfields3b;
    struct {
      uint32_t reserved10 : 3;
      uint32_t src_64b_addr_lo : 29;
    } bitfields3c;
    struct {
      uint32_t src_gds_addr_lo : 16;
      uint32_t reserved11 : 16;
    } bitfields3d;
    uint32_t imm_data;
    uint32_t ordinal3;
  };
  union {
    uint32_t src_memtc_addr_hi;
    uint32_t src_imm_data;
    uint32_t ordinal4;
  };
  union {
    struct {
      uint32_t dst_reg_offset : 18;
      uint32_t reserved12 : 14;
    } bitfields5a;
    struct {
      uint32_t reserved13 : 2;
      uint32_t dst_32b_addr_lo : 30;
    } bitfields5b;
    struct {
      uint32_t reserved14 : 3;
      uint32_t dst_64b_addr_lo : 29;
    } bitfields5c;
    struct {
      uint32_t dst_gds_addr_lo : 16;
      uint32_t reserved15 : 16;
    } bitfields5d;
    uint32_t ordinal5;
  };
  uint32_t dst_addr_hi;
} PM4MEC_COPY_DATA;
namespace gfx9 {

struct PM4_MEC_ACQUIRE_MEM {
  union {
    PM4_MEC_TYPE_3_HEADER   header;
    uint32_t            ordinal1;
  };
  union {
    struct {
      uint32_t coher_cntl:31;
      uint32_t reserved1:1;
    } bitfields2;
    uint32_t ordinal2;
  };
  uint32_t coher_size;
  union {
    struct {
      uint32_t coher_size_hi:8;
      uint32_t reserved2:24;
    } bitfields4;
    uint32_t ordinal4;
  };
  uint32_t coher_base_lo;
  union {
    struct {
      uint32_t coher_base_hi:24;
      uint32_t reserved3:8;
    } bitfields6;
    uint32_t ordinal6;
  };
  union {
    struct {
      uint32_t poll_interval:16;
      uint32_t reserved4:16;
    } bitfields7;
    uint32_t ordinal7;
  };
};

struct PM4_MEC_RELEASE_MEM {
    union {
        PM4_MEC_TYPE_3_HEADER   header;
        uint32_t            ordinal1;
    };
    union {
        struct {
            uint32_t event_type:6;
            uint32_t reserved1:2;
            uint32_t event_index:4;
            uint32_t tcl1_vol_action_ena:1;
            uint32_t tc_vol_action_ena:1;
            uint32_t reserved2:1;
            uint32_t tc_wb_action_ena:1;
            uint32_t tcl1_action_ena:1;
            uint32_t tc_action_ena:1;
            uint32_t reserved3:1;
            uint32_t tc_nc_action_ena:1;
            uint32_t tc_wc_action_ena:1;
            uint32_t tc_md_action_ena:1;
            uint32_t reserved4:3;
            uint32_t cache_policy:2;
            uint32_t reserved5:2;
            uint32_t pq_exe_status:1;
            uint32_t reserved6:2;
        } bitfields2;
        uint32_t ordinal2;
    };
    union {
        struct {
            uint32_t reserved7:16;
            uint32_t dst_sel:2;
            uint32_t reserved8:6;
            uint32_t int_sel:3;
            uint32_t reserved9:2;
            uint32_t data_sel:3;
        } bitfields3;
        uint32_t ordinal3;
    };
    union {
        struct {
            uint32_t reserved10:2;
            uint32_t address_lo_32b:30;
        } bitfields4a;
        struct {
            uint32_t reserved11:3;
            uint32_t address_lo_64b:29;
        } bitfields4b;
        uint32_t reserved12;
        uint32_t ordinal4;
    };
    union {
        uint32_t address_hi;
        uint32_t reserved13;
        uint32_t ordinal5;
    };
    union {
        uint32_t data_lo;
        uint32_t cmp_data_lo;
        struct {
            uint32_t dw_offset:16;
            uint32_t num_dwords:16;
        } bitfields6c;
        uint32_t reserved14;
        uint32_t ordinal6;
    };
    union {
        uint32_t data_hi;
        uint32_t cmp_data_hi;
        uint32_t reserved15;
        uint32_t reserved16;
        uint32_t ordinal7;
    };
    uint32_t int_ctxid;
};

struct PM4_MEC_WAIT_REG_MEM64 {
  union {
    PM4_MEC_TYPE_3_HEADER   header;
    uint32_t            ordinal1;
  };
  union {
    struct {
      uint32_t function:3;
      uint32_t reserved1:1;
      uint32_t mem_space:2;
      uint32_t operation:2;
      uint32_t reserved2:24;
    } bitfields2;
    uint32_t ordinal2;
  };
  union {
    struct {
      uint32_t reserved3:3;
      uint32_t mem_poll_addr_lo:29;
    } bitfields3a;
    struct {
      uint32_t reg_poll_addr:18;
      uint32_t reserved4:14;
    } bitfields3b;
    struct {
      uint32_t reg_write_addr1:18;
      uint32_t reserved5:14;
    } bitfields3c;
    uint32_t ordinal3;
  };
  union {
    uint32_t mem_poll_addr_hi;
    struct {
      uint32_t reg_write_addr2:18;
      uint32_t reserved6:14;
    } bitfields4b;
    uint32_t ordinal4;
  };
  uint32_t reference;
  uint32_t reference_hi;
  uint32_t mask;
  uint32_t mask_hi;
  union {
    struct {
      uint32_t poll_interval:16;
      uint32_t reserved7:16;
    } bitfields9;
    uint32_t ordinal9;
  };
};

/// @brief Structure used to configure the flushing of
/// various caches - instruction, constants, L1 and L2
struct AcquireMemTemplate {
  PM4_MEC_ACQUIRE_MEM acquire_mem;
};

struct EndofKernelNotifyTemplate {
  PM4_MEC_RELEASE_MEM release_mem;
};

/// @brief PM4 command to wait for a certain event before proceeding
/// to process another command on the queue
struct WaitRegMem64Template {
  PM4_MEC_WAIT_REG_MEM64 wait_reg_mem;
};

}  // gfx9 namespace

namespace gfx10 {

struct PM4_MEC_ACQUIRE_MEM {
  union {
    PM4_MEC_TYPE_3_HEADER   header;
    uint32_t            ordinal1;
  };
  uint32_t reserved1;
  uint32_t coher_size;
  union {
    struct {
      uint32_t coher_size_hi:8;
      uint32_t reserved2:24;
    } bitfields4;
    uint32_t ordinal4;
  };
  uint32_t coher_base_lo;
  union {
    struct {
      uint32_t coher_base_hi:24;
      uint32_t reserved3:8;
    } bitfields6;
    uint32_t ordinal6;
  };
  union {
    struct {
      uint32_t poll_interval:16;
      uint32_t reserved4:16;
    } bitfields7;
    uint32_t ordinal7;
  };
  union {
    struct {
      uint32_t gcr_cntl:19;
      uint32_t reserved4:13;
    } bitfields8;
    uint32_t ordinal8;
  };
};

struct PM4_MEC_RELEASE_MEM {
    union {
        PM4_MEC_TYPE_3_HEADER   header;
        uint32_t            ordinal1;
    };
    union {
        struct {
            uint32_t event_type:6;
            uint32_t reserved1:2;
            uint32_t event_index:4;
            uint32_t gcr_cntl:12;
            uint32_t reserved2:1;
            uint32_t cache_policy:2;
            uint32_t reserved3:2;
            uint32_t pq_exe_status:1;
            uint32_t reserved4:2;
        } bitfields2;
        uint32_t ordinal2;
    };
    union {
        struct {
            uint32_t reserved7:16;
            uint32_t dst_sel:2;
            uint32_t reserved8:2;
            uint32_t mes_intr_pipe:2;
            uint32_t mes_action_id:2;
            uint32_t int_sel:3;
            uint32_t reserved9:2;
            uint32_t data_sel:3;
        } bitfields3;
        uint32_t ordinal3;
    };
    union {
        struct {
            uint32_t reserved10:2;
            uint32_t address_lo_32b:30;
        } bitfields4a;
        struct {
            uint32_t reserved11:3;
            uint32_t address_lo_64b:29;
        } bitfields4b;
        uint32_t reserved12;
        uint32_t ordinal4;
    };
    union {
        uint32_t address_hi;
        uint32_t reserved13;
        uint32_t ordinal5;
    };
    union {
        uint32_t data_lo;
        uint32_t cmp_data_lo;
        struct {
            uint32_t dw_offset:16;
            uint32_t num_dwords:16;
        } bitfields6c;
        uint32_t reserved14;
        uint32_t ordinal6;
    };
    union {
        uint32_t data_hi;
        uint32_t cmp_data_hi;
        uint32_t reserved15;
        uint32_t reserved16;
        uint32_t ordinal7;
    };
    uint32_t int_ctxid;
};

struct PM4_MEC_WAIT_REG_MEM64 {
    union {
        PM4_MEC_TYPE_3_HEADER   header;            ///header
        uint32_t            ordinal1;
    };
    union {
        struct {
            uint32_t function:3;
            uint32_t reserved1:1;
            uint32_t mem_space:2;
            uint32_t operation:2;
            uint32_t reserved2:14;
            uint32_t mes_intr_pipe:2;
            uint32_t mes_action:1;
            uint32_t cache_policy:2;
            uint32_t reserved3:5;
        } bitfields2;
        uint32_t ordinal2;
    };
    union {
        struct {
            uint32_t reserved4:3;
            uint32_t mem_poll_addr_lo:29;
        } bitfields3a;
        struct {
            uint32_t reg_poll_addr:18;
            uint32_t reserved5:14;
        } bitfields3b;
        struct {
            uint32_t reg_write_addr1:18;
            uint32_t reserved6:14;
        } bitfields3c;
        uint32_t ordinal3;
    };
    union {
        uint32_t mem_poll_addr_hi;
        struct {
            uint32_t reg_write_addr2:18;
            uint32_t reserved7:14;
        } bitfields4b;
        uint32_t ordinal4;
    };
    uint32_t reference;
    uint32_t reference_hi;
    uint32_t mask;
    uint32_t mask_hi;
    union {
        struct {
            uint32_t poll_interval:16;
            uint32_t reserved8:15;
            uint32_t optimize_ace_offload_mode:1;
        } bitfields9;
        uint32_t ordinal9;
    };
};

/// @brief Structure used to configure the flushing of
/// various caches - instruction, constants, L1 and L2
struct AcquireMemTemplate {
  PM4_MEC_ACQUIRE_MEM acquire_mem;
};

struct EndofKernelNotifyTemplate {
  PM4_MEC_RELEASE_MEM release_mem;
};

struct WaitRegMem64Template {
  PM4_MEC_WAIT_REG_MEM64 wait_reg_mem;
};

} // gfx10 namespace

namespace gfx11 {

struct PM4_MEC_RELEASE_MEM {
    union {
        PM4_MEC_TYPE_3_HEADER header;
        uint32_t ordinal1;
    };
    union {
        struct {
            uint32_t event_type:6;
            uint32_t reserved1:2;
            uint32_t event_index:4;
            uint32_t gcr_cntl:13;
            uint32_t cache_policy:2;
            uint32_t reserved2:1;
            uint32_t pq_exe_status:1;
            uint32_t reserved3:1;
            uint32_t glk_inv:1;
            uint32_t reserved4:1;
        } bitfields2;
        uint32_t ordinal2;
    };
    union {
        struct {
            uint32_t reserved5:16;
            uint32_t dst_sel:2;
            uint32_t reserved6:2;
            uint32_t mes_intr_pipe:2;
            uint32_t mes_action_id:2;
            uint32_t int_sel:3;
            uint32_t reserved7:2;
            uint32_t data_sel:3;
        } bitfields3;
        uint32_t ordinal3;
    };
    union {
        struct {
            uint32_t reserved8:2;
            uint32_t address_lo_32b:30;
        } bitfields4a;
        struct {
            uint32_t reserved9:3;
            uint32_t address_lo_64b:29;
        } bitfields4b;
        uint32_t reserved10;
        uint32_t ordinal4;
    };
    union {
        uint32_t address_hi;
        uint32_t reserved11;
        uint32_t ordinal5;
    };
    union {
        uint32_t data_lo;
        uint32_t cmp_data_lo;
        struct {
            uint32_t dw_offset:16;
            uint32_t num_dwords:16;
        } bitfields6c;
        uint32_t reserved12;
        uint32_t ordinal6;
    };
    union {
        uint32_t data_hi;
        uint32_t cmp_data_hi;
        uint32_t reserved13;
        uint32_t reserved14;
        uint32_t ordinal7;
    };
    uint32_t int_ctxid;
};

struct EndofKernelNotifyTemplate {
  PM4_MEC_RELEASE_MEM release_mem;
};

} // gfx11 namespace

