/**
 * @file aql_gfx9_defs.h
 * @brief GFX9 (Vega) Architecture Definitions
 *
 * This file contains GFX9-specific constants, register definitions,
 * and PM4 packet formats for the Vega architecture family.
 *
 * Key differences from GFX12:
 * - Supports PRED_EXEC packets (unique to GFX9)
 * - Uses 7-dword ACQUIRE_MEM packets vs 8-dword in GFX12
 * - Has UCONFIG register space
 * - Different cache coherence mechanisms
 */

#ifndef AQL_GFX9_DEFS_H
#define AQL_GFX9_DEFS_H

#include "aql_types.h"

/*
 * GFX9 Architecture Information
 */
#define AQL_GFX9_ARCH_NAME              "GFX9_Vega"
#define AQL_GFX9_VERSION                9

/*
 * GFX9 Register Space Definitions
 */
#define AQL_GFX9_CONFIG_SPACE_START     0x00002000
#define AQL_GFX9_CONFIG_SPACE_END       0x00009FFF
#define AQL_GFX9_UCONFIG_SPACE_START    0x0000C000
#define AQL_GFX9_UCONFIG_SPACE_END      0x0000C0FF

/*
 * GFX9 PM4 Packet Types and Opcodes
 */
#define AQL_GFX9_PKT_TYPE_0             0
#define AQL_GFX9_PKT_TYPE_3             3

/* GFX9 PM4 Type-3 Opcodes */
#define AQL_GFX9_PKT3_NOP               0x10
#define AQL_GFX9_PKT3_WRITE_DATA        0x37
#define AQL_GFX9_PKT3_COPY_REG_DATA     0x2E
#define AQL_GFX9_PKT3_WAIT_REG_MEM      0x3C
#define AQL_GFX9_PKT3_INDIRECT_BUFFER   0x3F
#define AQL_GFX9_PKT3_ACQUIRE_MEM       0x58
#define AQL_GFX9_PKT3_PRED_EXEC         0x94

/* GFX9 Register Programming Opcodes */
#define AQL_GFX9_WRITE_UCONFIG_REG      0x13
#define AQL_GFX9_WRITE_CONFIG_REG       0x68
#define AQL_GFX9_WRITE_SH_REG           0x76

/*
 * GFX9 PM4 Packet Size Definitions
 */
#define AQL_GFX9_PKT3_WRITE_UCONFIG_REG_SIZE    3   /* header + reg + data */
#define AQL_GFX9_PKT3_WRITE_CONFIG_REG_SIZE     3   /* header + reg + data */
#define AQL_GFX9_PKT3_WRITE_SH_REG_SIZE         3   /* header + reg + data */
#define AQL_GFX9_PKT3_COPY_REG_DATA_SIZE        5   /* header + 4 dwords */
#define AQL_GFX9_PKT3_WAIT_REG_MEM_SIZE         7   /* header + 6 dwords */
#define AQL_GFX9_PKT3_INDIRECT_BUFFER_SIZE      4   /* header + 3 dwords */
#define AQL_GFX9_PKT3_ACQUIRE_MEM_SIZE          7   /* header + 6 dwords (GFX9 specific) */
#define AQL_GFX9_PKT3_PRED_EXEC_SIZE            3   /* header + 2 dwords (GFX9 specific) */
#define AQL_GFX9_PKT3_NOP_MIN_SIZE              2   /* header + 1 data dword minimum */

/*
 * GFX9 PM4 Packet Header Construction
 */
#define AQL_GFX9_PKT3_HEADER(opcode, count) \
    ((AQL_GFX9_PKT_TYPE_3 << 30) | ((count) << 16) | ((opcode) << 8))

#define AQL_GFX9_PKT0_HEADER(reg, count) \
    ((AQL_GFX9_PKT_TYPE_0 << 30) | ((count) << 16) | ((reg) >> 2))

/*
 * GFX9 ACQUIRE_MEM Packet Fields (7-dword format)
 */
#define AQL_GFX9_ACQUIRE_MEM_ENGINE_PFP     (0 << 31)
#define AQL_GFX9_ACQUIRE_MEM_ENGINE_ME      (1 << 31)
#define AQL_GFX9_ACQUIRE_MEM_CP_COHER_CNTL  0x00000000
#define AQL_GFX9_ACQUIRE_MEM_CP_COHER_SIZE  0xFFFFFFFF
#define AQL_GFX9_ACQUIRE_MEM_CP_COHER_BASE  0x00000000

/* GFX9 Cache Coherency Control Bits */
#define AQL_GFX9_ACQUIRE_MEM_TCL1_ACTION_ENA    (1 << 0)
#define AQL_GFX9_ACQUIRE_MEM_TC_ACTION_ENA      (1 << 1)
#define AQL_GFX9_ACQUIRE_MEM_CB_ACTION_ENA      (1 << 2)
#define AQL_GFX9_ACQUIRE_MEM_DB_ACTION_ENA      (1 << 3)
#define AQL_GFX9_ACQUIRE_MEM_SH_KCACHE_ACTION_ENA (1 << 4)
#define AQL_GFX9_ACQUIRE_MEM_SH_ICACHE_ACTION_ENA (1 << 5)

/*
 * GFX9 PRED_EXEC Packet Fields (GFX9 specific)
 */
#define AQL_GFX9_PRED_EXEC_XCC_MASK         0xFF
#define AQL_GFX9_PRED_EXEC_EXEC_COUNT_MASK  0xFFFFFF00

/*
 * GFX9 WAIT_REG_MEM Operations
 */
#define AQL_GFX9_WAIT_REG_MEM_FUNCTION_ALWAYS       0
#define AQL_GFX9_WAIT_REG_MEM_FUNCTION_LT           1
#define AQL_GFX9_WAIT_REG_MEM_FUNCTION_LE           2
#define AQL_GFX9_WAIT_REG_MEM_FUNCTION_EQ           3
#define AQL_GFX9_WAIT_REG_MEM_FUNCTION_NE           4
#define AQL_GFX9_WAIT_REG_MEM_FUNCTION_GE           5
#define AQL_GFX9_WAIT_REG_MEM_FUNCTION_GT           6

#define AQL_GFX9_WAIT_REG_MEM_SPACE_REG             0
#define AQL_GFX9_WAIT_REG_MEM_SPACE_MEM             1

/*
 * GFX9 COPY_REG_DATA Options
 */
#define AQL_GFX9_COPY_REG_DATA_SRC_SEL_REG          0
#define AQL_GFX9_COPY_REG_DATA_SRC_SEL_MEM          1
#define AQL_GFX9_COPY_REG_DATA_DST_SEL_REG          0
#define AQL_GFX9_COPY_REG_DATA_DST_SEL_MEM          1
#define AQL_GFX9_COPY_REG_DATA_WR_CONFIRM           (1 << 20)

/*
 * GFX9 GRBM Index Register Programming
 */
#define AQL_GFX9_GRBM_GFX_INDEX                     0x30800
#define AQL_GFX9_GRBM_GFX_INDEX_TYPE_ALL            0
#define AQL_GFX9_GRBM_GFX_INDEX_TYPE_SE             1
#define AQL_GFX9_GRBM_GFX_INDEX_TYPE_SA             2

#define AQL_GFX9_GRBM_GFX_INDEX_SE_SHIFT            16
#define AQL_GFX9_GRBM_GFX_INDEX_SA_SHIFT            8
#define AQL_GFX9_GRBM_GFX_INDEX_INSTANCE_SHIFT      0

/*
 * GFX9 Architecture Capabilities
 */
#define AQL_GFX9_HAS_PRED_EXEC                      true
#define AQL_GFX9_HAS_UCONFIG_SPACE                  true
#define AQL_GFX9_HAS_DUAL_SDMA                      true
#define AQL_GFX9_HAS_GL_CACHE_HIERARCHY             false
#define AQL_GFX9_HAS_SMN_ADDRESSING                 true

/*
 * GFX9 Real Hardware Performance Counter Select Registers
 * These are actual hardware register addresses extracted from AQLProfile
 * Source: /linux/registers/gc/gc_9_2_1_offset.h
 */
#define AQL_GFX9_PERF_SEL_BASE_CB                   0x3C01  /* Real hardware address */
#define AQL_GFX9_PERF_SEL_BASE_CPC                  0x3809  /* Real hardware address */
#define AQL_GFX9_PERF_SEL_BASE_CPF                  0x3807  /* Real hardware address */
#define AQL_GFX9_PERF_SEL_BASE_CPG                  0x3802  /* Real hardware address */
#define AQL_GFX9_PERF_SEL_BASE_DB                   0x3C40  /* Real hardware address */
#define AQL_GFX9_PERF_SEL_BASE_GDS                  0x3A80  /* Real hardware address */
#define AQL_GFX9_PERF_SEL_BASE_GRBM                 0x3840  /* Real hardware address */
#define AQL_GFX9_PERF_SEL_BASE_GRBMSE               0x8040  /* Placeholder - not found in registers */
#define AQL_GFX9_PERF_SEL_BASE_PA_SC                0x3940  /* Real hardware address */
#define AQL_GFX9_PERF_SEL_BASE_PA_SU                0x3900  /* Real hardware address */
#define AQL_GFX9_PERF_SEL_BASE_SPI                  0x2400  /* Placeholder - not found in registers */
#define AQL_GFX9_PERF_SEL_BASE_SQ                   0x39C0  /* Real hardware address */
#define AQL_GFX9_PERF_SEL_BASE_SX                   0x2100  /* Placeholder - not found in registers */
#define AQL_GFX9_PERF_SEL_BASE_TA                   0x2E00  /* Placeholder - not found in registers */
#define AQL_GFX9_PERF_SEL_BASE_TCP                  0x3B40  /* Real hardware address */
#define AQL_GFX9_PERF_SEL_BASE_TD                   0x2F00  /* Placeholder - not found in registers */
#define AQL_GFX9_PERF_SEL_BASE_TCA                  0x3400  /* Placeholder - not found in registers */
#define AQL_GFX9_PERF_SEL_BASE_TCC                  0x3B80  /* Real hardware address */

/*
 * GFX9 Performance Counter Register Offsets
 */
#define AQL_GFX9_PERF_COUNTER_SELECT_OFFSET         0x00
#define AQL_GFX9_PERF_COUNTER_SELECT1_OFFSET        0x01
#define AQL_GFX9_PERF_COUNTER_LO_OFFSET             0x02
#define AQL_GFX9_PERF_COUNTER_HI_OFFSET             0x03

/*
 * GFX9 Shader Engine Configuration
 */
#define AQL_GFX9_MAX_SHADER_ENGINES                 4
#define AQL_GFX9_MAX_SHADER_ARRAYS_PER_SE           1
#define AQL_GFX9_MAX_CU_PER_SA                      16

/*
 * GFX9 Cache Policy Definitions
 */
#define AQL_GFX9_CACHE_POLICY_LRU                   0
#define AQL_GFX9_CACHE_POLICY_STREAM                1
#define AQL_GFX9_CACHE_POLICY_NOA                   2
#define AQL_GFX9_CACHE_POLICY_BYPASS                3

/*
 * GFX9 SMN (System Management Network) Addressing
 */
#define AQL_GFX9_SMN_BASE                           0x01000000
#define AQL_GFX9_SMN_AID_SHIFT                      20

/*
 * GFX9 Register Space Layout Structure
 */
static const aql_register_spaces_t aql_gfx9_register_spaces = {
    .config_space_start = AQL_GFX9_CONFIG_SPACE_START,
    .config_space_end = AQL_GFX9_CONFIG_SPACE_END,
    .uconfig_space_start = AQL_GFX9_UCONFIG_SPACE_START,
    .uconfig_space_end = AQL_GFX9_UCONFIG_SPACE_END,
    .has_uconfig_space = true
};

/*
 * GFX9 Counter Block Register Information
 */
typedef struct {
    aql_block_id_t block_id;
    uint32_t perf_sel_base;
    uint32_t counter_count;
    const char* block_name;
} aql_gfx9_counter_block_info_t;

static const aql_gfx9_counter_block_info_t aql_gfx9_counter_blocks[] = {
    { AQL_BLOCK_CB,     AQL_GFX9_PERF_SEL_BASE_CB,     4, "CB" },     /* Real: 0x3C01 */
    { AQL_BLOCK_CPC,    AQL_GFX9_PERF_SEL_BASE_CPC,    2, "CPC" },    /* Real: 0x3809 */
    { AQL_BLOCK_CPF,    AQL_GFX9_PERF_SEL_BASE_CPF,    2, "CPF" },    /* Real: 0x3807 */
    { AQL_BLOCK_CPG,    AQL_GFX9_PERF_SEL_BASE_CPG,    2, "CPG" },    /* Real: 0x3802 */
    { AQL_BLOCK_DB,     AQL_GFX9_PERF_SEL_BASE_DB,     4, "DB" },     /* Real: 0x3C40 */
    { AQL_BLOCK_GDS,    AQL_GFX9_PERF_SEL_BASE_GDS,    4, "GDS" },    /* Real: 0x3A80 */
    { AQL_BLOCK_GRBM,   AQL_GFX9_PERF_SEL_BASE_GRBM,   2, "GRBM" },   /* Real: 0x3840 */
    { AQL_BLOCK_GRBM_SE, AQL_GFX9_PERF_SEL_BASE_GRBMSE, 4, "GRBMSE" }, /* Placeholder */
    { AQL_BLOCK_PA_SC,  AQL_GFX9_PERF_SEL_BASE_PA_SC,  8, "PA_SC" },  /* Real: 0x3940 */
    { AQL_BLOCK_PA_SU,  AQL_GFX9_PERF_SEL_BASE_PA_SU,  4, "PA_SU" },  /* Real: 0x3900 */
    { AQL_BLOCK_SPI,    AQL_GFX9_PERF_SEL_BASE_SPI,    6, "SPI" },    /* Placeholder */
    { AQL_BLOCK_SQ,     AQL_GFX9_PERF_SEL_BASE_SQ,     8, "SQ" },     /* Real: 0x39C0 */
    { AQL_BLOCK_SX,     AQL_GFX9_PERF_SEL_BASE_SX,     4, "SX" },     /* Placeholder */
    { AQL_BLOCK_TA,     AQL_GFX9_PERF_SEL_BASE_TA,     2, "TA" },     /* Placeholder */
    { AQL_BLOCK_TCP,    AQL_GFX9_PERF_SEL_BASE_TCP,    4, "TCP" },    /* Real: 0x3B40 */
    { AQL_BLOCK_TD,     AQL_GFX9_PERF_SEL_BASE_TD,     2, "TD" },     /* Placeholder */
    { AQL_BLOCK_TCA,    AQL_GFX9_PERF_SEL_BASE_TCA,    4, "TCA" },    /* Placeholder */
    { AQL_BLOCK_TCC,    AQL_GFX9_PERF_SEL_BASE_TCC,    4, "TCC" },    /* Real: 0x3B80 */
    { AQL_BLOCK_UNKNOWN, 0, 0, NULL } /* Terminator */
};

#endif /* AQL_GFX9_DEFS_H */